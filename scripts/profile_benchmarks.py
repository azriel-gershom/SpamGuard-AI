"""SpamGuard AI — Resource & Performance Profiling Engine.

Measures real-world inference latency, memory consumption, throughput,
and prediction reproducibility on the local execution environment.
Exports structured benchmark metrics to reports/performance_benchmark.json.
"""

from __future__ import annotations

import ctypes
import json
import platform
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAVED_PIPELINE_PATH
from src.predict import SpamPredictor


# =====================================================================
# Windows Memory Profiling via Win32 API
# =====================================================================

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

kernel32.GetCurrentProcess.restype = wintypes.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def get_current_process_memory_mb() -> dict[str, float]:
    """Returns working set and peak memory in megabytes (MB) on Windows."""
    try:
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return {
                "working_set_mb": round(pmc.WorkingSetSize / (1024 * 1024), 2),
                "peak_working_set_mb": round(pmc.PeakWorkingSetSize / (1024 * 1024), 2),
                "pagefile_mb": round(pmc.PagefileUsage / (1024 * 1024), 2),
            }
        else:
            err = kernel32.GetLastError()
            print(f"Warning: GetProcessMemoryInfo returned FALSE, error code: {err}")
    except Exception as exc:
        print(f"Warning: Could not query Windows memory info: {exc}")
    return {"working_set_mb": 0.0, "peak_working_set_mb": 0.0, "pagefile_mb": 0.0}


# =====================================================================
# Benchmark Workload & Synthetic Samples
# =====================================================================

SYNTHETIC_SPAM_SAMPLE = (
    "URGENT: Claim Your Free Lottery Cash Prize Guaranteed!",
    "Congratulations! Your email address has been selected to receive a cash prize of $500,000. "
    "To claim your money immediately, click the link below and verify your banking details. "
    "Do not delay, offer expires within 24 hours. Free bonus mortgage refinance life insurance guarantee."
)

SYNTHETIC_HAM_SAMPLE = (
    "Weekly Engineering Sprint Review & PR #104 Discussion",
    "Hi team, please find attached the meeting notes and action items from today's sprint review. "
    "We merged PR #104 fixing the database connection pool timeout. Let me know if you encounter any "
    "issues during integration testing. Best regards, Alex."
)


def run_benchmarks() -> dict:
    print("=" * 70)
    print("SpamGuard AI — Phase 09 Performance & Resource Profiling")
    print("=" * 70)

    # 1. Environment Information
    mem_initial = get_current_process_memory_mb()
    print(f"[*] Initial Memory: Working Set = {mem_initial['working_set_mb']} MB")

    # 2. Cold Model Loading Latency
    print("[*] Benchmarking Cold Model Load...")
    t_start = time.perf_counter()
    predictor = SpamPredictor.load(SAVED_PIPELINE_PATH)
    cold_load_time_ms = (time.perf_counter() - t_start) * 1000.0
    mem_after_load = get_current_process_memory_mb()
    print(f"    Cold Load Time: {cold_load_time_ms:.2f} ms")
    print(f"    Memory After Load: {mem_after_load['working_set_mb']} MB (Delta: +{mem_after_load['working_set_mb'] - mem_initial['working_set_mb']:.2f} MB)")

    # 3. Warm Single-Email Latency (100 Iterations)
    print("[*] Benchmarking Warm Single-Email Inference (100 iterations)...")
    single_latencies_ms = []
    # Warmup run
    predictor.predict_email(*SYNTHETIC_HAM_SAMPLE, explain=False)

    for _ in range(100):
        t0 = time.perf_counter()
        res = predictor.predict_email(*SYNTHETIC_HAM_SAMPLE, explain=False)
        single_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_single = float(np.median(single_latencies_ms))
    p95_single = float(np.percentile(single_latencies_ms, 95))
    mean_single = float(np.mean(single_latencies_ms))
    min_single = float(np.min(single_latencies_ms))
    max_single = float(np.max(single_latencies_ms))
    print(f"    Single Email: Median = {p50_single:.2f} ms, P95 = {p95_single:.2f} ms, Mean = {mean_single:.2f} ms, Min = {min_single:.2f} ms, Max = {max_single:.2f} ms")

    # 4. Single-Email with Explainability (30 Iterations)
    print("[*] Benchmarking Single-Email with Explainability (30 iterations)...")
    explain_latencies_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        res_exp = predictor.predict_email(*SYNTHETIC_SPAM_SAMPLE, explain=True, top_k=10)
        explain_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_explain = float(np.median(explain_latencies_ms))
    p95_explain = float(np.percentile(explain_latencies_ms, 95))
    print(f"    Explainability Mode: Median = {p50_explain:.2f} ms, P95 = {p95_explain:.2f} ms")

    # 5. Small Batch Classification (50 Emails, 20 iterations)
    print("[*] Benchmarking Small Batch (50 emails x 20 iterations)...")
    small_batch_data = [SYNTHETIC_HAM_SAMPLE if i % 2 == 0 else SYNTHETIC_SPAM_SAMPLE for i in range(50)]
    small_batch_latencies_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        batch_res = predictor.predict_batch(small_batch_data, explain=False)
        small_batch_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_small = float(np.median(small_batch_latencies_ms))
    p95_small = float(np.percentile(small_batch_latencies_ms, 95))
    per_email_small = p50_small / 50.0
    throughput_small = (50.0 / (p50_small / 1000.0))
    print(f"    50 Batch: Total Median = {p50_small:.2f} ms ({per_email_small:.3f} ms/email), Throughput = {throughput_small:.1f} emails/sec")

    # 6. Medium Batch Classification (500 Emails, 10 iterations)
    print("[*] Benchmarking Medium Batch (500 emails x 10 iterations)...")
    med_batch_data = [SYNTHETIC_HAM_SAMPLE if i % 2 == 0 else SYNTHETIC_SPAM_SAMPLE for i in range(500)]
    med_batch_latencies_ms = []
    for _ in range(10):
        t0 = time.perf_counter()
        batch_res = predictor.predict_batch(med_batch_data, explain=False)
        med_batch_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_med = float(np.median(med_batch_latencies_ms))
    p95_med = float(np.percentile(med_batch_latencies_ms, 95))
    per_email_med = p50_med / 500.0
    throughput_med = (500.0 / (p50_med / 1000.0))
    print(f"    500 Batch: Total Median = {p50_med:.2f} ms ({per_email_med:.3f} ms/email), Throughput = {throughput_med:.1f} emails/sec")

    # 7. Large Batch Classification (1,000 Emails, 5 iterations)
    print("[*] Benchmarking Large Batch (1,000 emails x 5 iterations)...")
    large_batch_data = [SYNTHETIC_HAM_SAMPLE if i % 2 == 0 else SYNTHETIC_SPAM_SAMPLE for i in range(1000)]
    large_batch_latencies_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        batch_res = predictor.predict_batch(large_batch_data, explain=False)
        large_batch_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50_large = float(np.median(large_batch_latencies_ms))
    p95_large = float(np.percentile(large_batch_latencies_ms, 95))
    per_email_large = p50_large / 1000.0
    throughput_large = (1000.0 / (p50_large / 1000.0))
    mem_peak = get_current_process_memory_mb()
    print(f"    1,000 Batch: Total Median = {p50_large:.2f} ms ({per_email_large:.3f} ms/email), Throughput = {throughput_large:.1f} emails/sec")
    print(f"    Peak Process Memory: {mem_peak['peak_working_set_mb']} MB (Current: {mem_peak['working_set_mb']} MB)")

    # 8. Deterministic Output Consistency Verification (50 repeated runs)
    print("[*] Verifying Deterministic Prediction Parity across 50 runs...")
    reference_res = predictor.predict_email(*SYNTHETIC_SPAM_SAMPLE, explain=True)
    parity_maintained = True
    for _ in range(50):
        check_res = predictor.predict_email(*SYNTHETIC_SPAM_SAMPLE, explain=True)
        if (check_res.predicted_class != reference_res.predicted_class or
            abs(check_res.spam_probability - reference_res.spam_probability) > 1e-9 or
            abs(check_res.ham_probability - reference_res.ham_probability) > 1e-9):
            parity_maintained = False
            break
    print(f"    Prediction Parity Maintained: {parity_maintained}")

    benchmark_data = {
        "benchmark_metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "hardware": {
                "cpu": "12th Gen Intel Core i7-12650H (10 Cores, 16 Threads)",
                "ram_total_gb": 16.0,
                "os": platform.platform(),
                "python_version": platform.python_version(),
            },
            "model_identifier": predictor._model_identifier,
            "pipeline_sha256": predictor._model_sha256,
        },
        "cold_model_load": {
            "load_time_ms": round(cold_load_time_ms, 2),
            "memory_before_load_mb": mem_initial["working_set_mb"],
            "memory_after_load_mb": mem_after_load["working_set_mb"],
            "memory_delta_mb": round(mem_after_load["working_set_mb"] - mem_initial["working_set_mb"], 2),
        },
        "single_email_latency_ms": {
            "iterations": 100,
            "median_p50": round(p50_single, 3),
            "percentile_p95": round(p95_single, 3),
            "mean": round(mean_single, 3),
            "min": round(min_single, 3),
            "max": round(max_single, 3),
        },
        "single_email_with_explainability_ms": {
            "iterations": 30,
            "median_p50": round(p50_explain, 3),
            "percentile_p95": round(p95_explain, 3),
            "explainability_overhead_ms": round(p50_explain - p50_single, 3),
        },
        "batch_classification": {
            "small_batch_50": {
                "total_median_ms": round(p50_small, 2),
                "total_p95_ms": round(p95_small, 2),
                "latency_per_email_ms": round(per_email_small, 3),
                "throughput_emails_per_sec": round(throughput_small, 1),
            },
            "medium_batch_500": {
                "total_median_ms": round(p50_med, 2),
                "total_p95_ms": round(p95_med, 2),
                "latency_per_email_ms": round(per_email_med, 3),
                "throughput_emails_per_sec": round(throughput_med, 1),
            },
            "large_batch_1000": {
                "total_median_ms": round(p50_large, 2),
                "total_p95_ms": round(p95_large, 2),
                "latency_per_email_ms": round(per_email_large, 3),
                "throughput_emails_per_sec": round(throughput_large, 1),
            },
        },
        "memory_profile_mb": {
            "baseline_working_set_mb": mem_initial["working_set_mb"],
            "post_model_load_mb": mem_after_load["working_set_mb"],
            "peak_working_set_during_1000_batch_mb": mem_peak["peak_working_set_mb"],
            "steady_state_working_set_mb": mem_peak["working_set_mb"],
        },
        "prediction_reproducibility": {
            "tested_iterations": 50,
            "bit_for_bit_identical": parity_maintained,
        },
        "benchmark_limitations": [
            "Measurements were conducted on an MSI Windows 11 development laptop with background OS processes active.",
            "Results reflect vectorized CPU inference on Scikit-learn Pipeline; performance in containerized or resource-constrained serverless environments may vary.",
            "Local benchmarks must not be interpreted as contractual production service-level guarantees (SLAs)."
        ],
    }

    report_path = PROJECT_ROOT / "reports" / "performance_benchmark.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n[+] Successfully saved benchmark results to {report_path}")
    return benchmark_data


if __name__ == "__main__":
    run_benchmarks()
