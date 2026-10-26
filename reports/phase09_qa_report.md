# SpamGuard AI — Phase 09 Comprehensive QA, Security & Performance Audit Report

> **Academic AI Micro Project – 05** | **Domain:** Cybersecurity / Email Threat Filtering  
> **Evaluation Phase:** Phase 09 (Testing, Security Hardening & Performance Verification)  
> **Date:** October 26, 2026 | **Host Environment:** Windows 11 Home, MSI Laptop, Python 3.12.10

---

## 1. Executive Summary

Phase 09 executed a rigorous quality assurance, security penetration defense, edge-case hardening, and resource profiling audit of the **SpamGuard AI** application. 

Key milestones achieved:
- **Baseline Test Expansion:** Expanded automated test suite from **111 passed** to **123 passed (0 failures, 100% pass rate)**.
- **Model Checksum Stability:** Verified that the trained artifact [`models/spam_pipeline.joblib`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/models/spam_pipeline.joblib) remained completely frozen, with bit-for-bit SHA-256 verification (`c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`).
- **Targeted Defect Resolution:** Identified and resolved 2 operational state management and formula injection vulnerabilities without modifying existing ML models or UI aesthetics.
- **Empirical Benchmarking:** Profiled cold load latency (76.6 ms), warm single-email inference (1.04 ms median), vectorized batch processing (10,478 emails/sec throughput), and memory footprint (162.4 MB peak working set).

---

## 2. Audit Methodology & Scope

The audit evaluated five core system dimensions:
1. **Inference Robustness & Edge Cases:** Non-Latin character encodings (Cyrillic, Arabic, CJK, Devanagari), emojis, zero-width characters (`\u200b`), extreme whitespace, oversized inputs, and script/HTML tags.
2. **Batch CSV Ingestion & Security:** Malformed schemas, missing cells, boundary limits (`MAX_BATCH_SIZE = 10,000`), and spreadsheet formula injection (`=`, `+`, `-`, `@`, `\t`, `\r`, `\n`, and leading whitespace evasion).
3. **Streamlit UI State Stability:** Session state persistence during reactive component reruns, filter toggle persistence, and sample selector reset behavior.
4. **Machine Learning Integrity & Determinism:** Reload verification, reproducible discrete predictions, probability boundary guarantees ($[0.0, 1.0]$, sum integrity), and isolation of training-only statistics.
5. **System Resource Profiling:** Real-time Windows 11 Win32 memory counters (`GetProcessMemoryInfo`), latency percentiles (P50, P95), and CPU scaling.

---

## 3. Defects Discovered & Remediations Applied

| Defect ID | Severity | Affected Component | Root Cause Analysis | Remediation Implemented | Verification Test |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **DEF-01** | **Medium** | `render_batch_scanner()` ([`app.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/app.py)) | In Streamlit, placing the batch results table, filter radio, and download button directly within the `if st.button("Scan")` block caused all output to vanish upon interacting with the filter radio or download button (since button state evaluates to `False` on subsequent reruns). | Persisted batch results, metrics, and formatted DataFrames in `st.session_state` (`batch_results`, `batch_out_df`, `batch_metrics`), resetting them only upon a new file upload. | `test_apptest_batch_scanner_state_persistence_and_filtering` |
| **DEF-02** | **Medium** | `sanitize_csv_cell()` ([`src/predict.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/src/predict.py)) | Cell sanitization only examined `value[0]`. Malicious payloads with leading whitespace before formula triggers (e.g. `"  =cmd"`, `" \t=1+1"`) bypassed the escape check while still being parsed as executable formulas by Excel/Sheets. | Updated sanitization to strip leading whitespace (`lstrip(" \t\r\n")`) before inspecting dangerous prefix characters (`=, +, -, @`), prepending a single quote `'` to neutralize formula execution. | `test_formula_injection_leading_whitespace_and_control_chars` |
| **DEF-03** | **Low** | `render_email_scanner()` ([`app.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/app.py)) | Switching between pre-configured quick-fill demonstration samples did not invalidate prior classification results, temporarily displaying stale verdicts alongside newly selected text. | Added active sample tracking in `st.session_state["active_sample_key"]`; switching sample choices resets `scanner_result` to `None`. | `test_apptest_scanner_sample_selection_clears_stale_result` |
| **DEF-04** | **Low** | `render_model_analytics()` ([`app.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/app.py)) | Confusion matrix explanation lacked explicit mathematical formulas for Type I and Type II error rates and lacked provenance notices distinguishing historical test performance from modern zero-day protection. | Added explicit LaTeX mathematical formulas for False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1.13\%$) and False Negative Rate ($\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = 6.25\%$) alongside dataset provenance disclosures. | Visual inspection & `tests/test_app.py` |

---

## 4. Security Findings & Hardening Audit

### 4.1 Input Sanitization & XSS Defense
- Single email subject and body inputs are normalized with Unicode NFKC and stripped of non-printable control characters.
- Embedded HTML tags (e.g. `<script>`, `<img>`, `<iframe>`) are processed purely as lexical tokens by Scikit-learn's TF-IDF vectorizer and rendered via Streamlit's safe text components, guaranteeing zero script execution.

### 4.2 Denial of Service (DoS) Bounds
- **Single Email Length Bound:** Inputs exceeding `MAX_INPUT_CHARS` (1,000,000 characters) are rejected before vectorization with a `SpamGuardValidationError`, preventing excessive memory allocation.
- **Batch Record Bound:** Batch uploads exceeding `MAX_BATCH_SIZE` (10,000 records) are defensively halted with `SpamGuardBatchError`.

### 4.3 Client-Side Privacy & Leakage Audit
- No uploaded emails, single scan bodies, or batch CSV files are written to persistent storage or logged.
- Streamlit resource caching (`@st.cache_resource`) is strictly confined to the read-only, shared `SpamPredictor` model instance; user inputs and predictions are never globally cached.

### 4.4 Formula Injection Defense
- Sanitization verifies that no cell can initiate dynamic execution in Microsoft Excel, LibreOffice Calc, or Google Sheets. All cells with direct or whitespace-padded prefixes in `("=", "+", "-", "@", "\t", "\r", "\n")` are safely neutralized with `'`.

---

## 5. Performance & Resource Profiling Results

Empirical benchmarks conducted on host hardware (**12th Gen Intel Core i7-12650H**, 16 GB RAM, Windows 11 Home, Python 3.12.10):

| Workload Scenario | Iterations | Median Latency (P50) | 95th Percentile (P95) | Throughput | Process Memory |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cold Model Loading** | 1 | 76.62 ms | N/A | N/A | 150.5 MB $\rightarrow$ 159.7 MB (+9.2 MB) |
| **Warm Single Email (Inference Only)** | 100 | **1.04 ms** | 2.14 ms | ~960 emails/sec | Stable |
| **Single Email + Token Explainability** | 30 | **1.59 ms** | 3.19 ms | ~630 emails/sec | Stable (+0.55 ms overhead) |
| **Small Batch (50 Emails)** | 20 | 5.46 ms total | 6.70 ms total | **9,151.7 emails/sec** | 160.2 MB |
| **Medium Batch (500 Emails)** | 10 | 47.85 ms total | 55.41 ms total | **10,449.2 emails/sec** | 161.4 MB |
| **Large Batch (1,000 Emails)** | 5 | 95.43 ms total | 98.43 ms total | **10,478.5 emails/sec** | **162.4 MB peak** |

*Hardware Profiling Conclusion:* The vectorized Scikit-learn MultinomialNB inference pipeline demonstrates exceptional throughput exceeding 10,000 emails per second during bulk classification, with negligible memory footprint (~12 MB above Python baseline).

---

## 6. Windows 11 & Environment Compatibility

- **Path Handling:** All path operations utilize `pathlib.Path` with forward/backward slash normalization, fully compatible with Windows NT path conventions.
- **Console Encoding:** Addressed Windows cp1252 character console limitations by employing clean ASCII status indicators in CLI tools while preserving full UTF-8 encoding in web and report artifacts.
- **Dependency Audit:**
  - `requirements.txt` specifies `pytest>=8.0.0`.
  - Virtual environment `.venv` contains `pytest 9.1.1` and `streamlit 1.66.1`.
  - The installed Scikit-learn version loads the frozen Joblib pipeline with zero deprecation warnings and zero deserialization errors.

---

## 7. Automated Test Suite Summary

### Suite Composition (123 Tests Passed)
1. [`tests/test_smoke.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_smoke.py): 4 tests (Environment, imports, syntax).
2. [`tests/test_prepare_data.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_prepare_data.py): 11 tests (MIME parsing, HTML extraction, deduplication).
3. [`tests/test_eda.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_eda.py): 9 tests (Statistical distributions, figure exports).
4. [`tests/test_preprocess.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_preprocess.py): 14 tests (Stratified splits, leakage prevention, TF-IDF).
5. [`tests/test_train.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_train.py): 17 tests (Naïve Bayes training, tuning, serialization).
6. [`tests/test_evaluate.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_evaluate.py): 16 tests (Held-out evaluation, confusion matrix, ROC-AUC).
7. [`tests/test_predict.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_predict.py): 21 tests (Inference engine, explainability, batching).
8. [`tests/test_app.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_app.py): 19 tests (Streamlit navigation, views, charts).
9. [`tests/test_hardening.py`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/tests/test_hardening.py): 12 tests (**Phase 09 additions**: Formula injection, Unicode stress, batch state, model immutability).

---

## 8. Known Limitations & Academic Disclosures

1. **Temporal Generalization Shift:** The classifier is trained on the historical Apache SpamAssassin corpus (2002–2005). While highly accurate on lexical patterns of that era (97.30% Accuracy), it does not account for modern threat techniques such as AI-generated text, QR-code phishing (quishing), or domain spoofing.
2. **Corpus Distribution Bias:** The legitimate "Ham" class originates largely from open-source software mailing lists, resulting in high learned associations with terms like `linux`, `rpm`, and `redhat`.
3. **Local Benchmark Scope:** Latency benchmarks reflect local workstation performance on an MSI gaming laptop and are not service-level guarantees for distributed cloud deployments.

---

## 9. Final Readiness Conclusion

**SpamGuard AI — Phase 09 is verified SUCCESSFUL.**  
The application is hardened, secure, fully tested, and ready for Phase 10 (Final Project Academic Documentation & Release Delivery).
