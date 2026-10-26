"""SpamGuard AI — Professional Cybersecurity Streamlit Application.

Phase 08:
- Interactive, multi-section web dashboard for email spam classification.
- Reuses the frozen, verified Phase 05/06 model and Phase 07 prediction engine.
- Features Overview Dashboard, Single Email Scanner, Batch CSV Scanner, Model Analytics, and About.
- Built strictly with Streamlit and Plotly; zero unapproved external web frameworks.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import (
    APP_SUBTITLE,
    APP_TITLE,
    APP_VERSION,
    FIGURES_DIR,
    MAX_BATCH_SIZE,
    MAX_INPUT_CHARS,
)
from src.predict import (
    SpamGuardBatchError,
    SpamGuardModelError,
    SpamGuardValidationError,
    get_default_predictor,
    predict_batch,
    predict_email,
    sanitize_csv_cell,
)
from src.ui_helpers import (
    COLOR_CARD_BG,
    COLOR_CARD_BORDER,
    COLOR_HAM,
    COLOR_MUTED,
    COLOR_NEUTRAL,
    COLOR_SPAM,
    COLOR_TEXT,
    SAMPLE_EMAILS,
    create_baseline_comparison_chart,
    create_confusion_matrix_chart,
    create_gauge_chart,
    format_features_dataframe,
    get_model_status,
    load_evaluation_summary,
)

logger = logging.getLogger("SpamGuardAI.App")

# =====================================================================
# 1. Page Configuration & Custom CSS
# =====================================================================

st.set_page_config(
    page_title=f"{APP_TITLE} | AI Cybersecurity Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
/* Main Background and Typography */
.main {{
    background-color: #0b0f19;
    color: {COLOR_TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

/* Header Banner */
.header-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    background-color: rgba(59, 130, 246, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}}

/* Glassmorphism Metric Cards */
.security-card {{
    background-color: {COLOR_CARD_BG};
    border: 1px solid {COLOR_CARD_BORDER};
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}}

/* Threat & Safety Result Banners */
.verdict-banner-spam {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
    border: 1.5px solid #ef4444;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    color: #fee2e2;
}}

.verdict-banner-ham {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.1) 100%);
    border: 1.5px solid #10b981;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    color: #d1fae5;
}}

.verdict-title {{
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Token Tags */
.token-tag-spam {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background-color: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.85rem;
    margin: 0.15rem;
}}

.token-tag-ham {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    background-color: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 6px;
    font-family: monospace;
    font-size: 0.85rem;
    margin: 0.15rem;
}}

/* Footer Note */
.academic-footer {{
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid {COLOR_CARD_BORDER};
    color: {COLOR_MUTED};
    font-size: 0.82rem;
    text-align: center;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# 2. Sidebar Navigation & System Telemetry
# =====================================================================

def render_sidebar() -> str:
    """Renders the persistent sidebar with status telemetry and navigation."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.7rem;">🛡️</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 700;">{APP_TITLE}</h2>
                    <span style="color: {COLOR_MUTED}; font-size: 0.78rem;">v{APP_VERSION} | AI Security Engine</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Live Model Integrity Telemetry
        status = get_model_status()
        if status["available"] and status["checksum_match"]:
            st.markdown(
                """
                <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 0.6rem; margin-bottom: 1rem;">
                    <div style="color: #10b981; font-weight: 600; font-size: 0.85rem;">● ML ENGINE ONLINE</div>
                    <div style="color: #94a3b8; font-size: 0.72rem; word-break: break-all;">SHA-256 Verified Match</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 0.6rem; margin-bottom: 1rem;">
                    <div style="color: #ef4444; font-weight: 600; font-size: 0.85rem;">⚠️ ENGINE WARNING</div>
                    <div style="color: #94a3b8; font-size: 0.72rem;">{status['status']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 🧭 Navigation")
        selected_tab = st.radio(
            "Select Module",
            options=[
                "🏠 Overview Dashboard",
                "🔍 Email Scanner",
                "📁 Batch CSV Scanner",
                "📊 Model Analytics",
                "ℹ️ About & Methodology",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 🎓 Academic Context")
        st.markdown(
            """
            - **Micro Project:** 05
            - **Domain:** Cybersecurity & NLP
            - **Course Alignment:** CO2, CO4, CO5
            - **Model:** Multinomial Naïve Bayes
            - **Features:** 20,000 TF-IDF n-grams
            """
        )

        st.markdown("---")
        st.caption("SpamGuard AI — Built for Academic College Demonstration")

    return selected_tab


# =====================================================================
# 3. View: Overview Dashboard
# =====================================================================

def render_overview():
    """Renders the executive summary dashboard and model KPI cards."""
    st.markdown('<span class="header-badge">● Operational Security Dashboard</span>', unsafe_allow_html=True)
    st.title("🛡️ SpamGuard AI")
    st.subheader(APP_SUBTITLE)

    st.markdown(
        """
        **SpamGuard AI** is an intelligent cybersecurity application designed to detect, classify, and neutralize 
        unwanted email threats. Combining **Term Frequency-Inverse Document Frequency (TF-IDF)** feature extraction 
        with a calibrated **Multinomial Naïve Bayes** probabilistic classifier, it achieves rapid, transparent, and 
        statistically rigorous email filtering without reliance on external black-box LLM APIs.
        """
    )

    # Load authentic held-out test evaluation report
    report = load_evaluation_summary()
    if report and "final_test_metrics" in report:
        m = report["final_test_metrics"]
        st.markdown("### 🏆 Held-Out Test Set Performance Benchmark")
        st.caption("Measured independently on 889 quarantined emails (617 Ham, 272 Spam) from the Apache SpamAssassin Public Corpus.")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Test Accuracy", f"{m['accuracy'] * 100:.2f}%", "+27.90% vs Baseline")
        with col2:
            st.metric("Spam Precision", f"{m['spam']['precision'] * 100:.2f}%", "Confidence on Flagged")
        with col3:
            st.metric("Spam Recall", f"{m['spam']['recall'] * 100:.2f}%", "Threats Caught (255/272)")
        with col4:
            st.metric("Spam F1-Score", f"{m['spam']['f1_score'] * 100:.2f}%", "Harmonic Balance")
        with col5:
            st.metric("ROC-AUC", f"{m['roc_auc']:.4f}", "Discrimination Score")
    else:
        st.warning("⚠️ Phase 06 evaluation report (reports/test_evaluation.json) not detected. Displaying baseline structure.")

    st.markdown("---")

    # Workflow Columns
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### 🔍 High-Throughput Detection")
        st.markdown(
            """
            - **Interactive Email Scanner:** Inspect raw subject lines and email text with real-time probability calibration and token attribution.
            - **Bulk CSV Ingestion:** Vectorized batch classification capable of scoring thousands of records per second with sanitized CSV exports.
            - **Mathematical Explainability:** Token-level log-odds contributions reveal why an email was classified as spam or legitimate.
            """
        )

    with c_right:
        st.markdown("### 🔒 Defensive Privacy & Security")
        st.markdown(
            """
            - **Client-Side Data Privacy:** User email text is never persisted to disk, transmitted to external cloud APIs, or logged.
            - **Formula Injection Defense:** Exported batch CSV files escape leading formula control characters (`=`, `+`, `-`, `@`).
            - **Denial-of-Service Capping:** Safeguards against payload exhaustion with strict 1MB text and 10,000-record batch bounds.
            """
        )


# =====================================================================
# 4. View: Single Email Scanner
# =====================================================================

def render_email_scanner():
    """Renders the interactive single-email classifier with token explainability."""
    st.markdown('<span class="header-badge">● Interactive Threat Scanner</span>', unsafe_allow_html=True)
    st.title("🔍 Email Threat Inspector")
    st.markdown("Enter an email subject and body to scan for phishing, commercial spam, and suspicious keywords.")

    # Initialize Session State
    if "scanner_subject" not in st.session_state:
        st.session_state["scanner_subject"] = ""
    if "scanner_body" not in st.session_state:
        st.session_state["scanner_body"] = ""
    if "scanner_result" not in st.session_state:
        st.session_state["scanner_result"] = None

    # Pre-configured Sample Selector for Easy Demonstration
    sample_key = st.selectbox(
        "⚡ Quick-Fill Demonstration Sample:",
        options=list(SAMPLE_EMAILS.keys()),
        index=0,
        help="Select a synthetic email sample to populate the inputs immediately for live testing.",
    )

    if sample_key != "Select a pre-configured sample...":
        sample_data = SAMPLE_EMAILS[sample_key]
        st.session_state["scanner_subject"] = sample_data["subject"]
        st.session_state["scanner_body"] = sample_data["body"]
        st.info(f"💡 **Sample Context:** {sample_data['description']}")

    # Input Form
    with st.form("single_email_form", clear_on_submit=False):
        subject_input = st.text_input(
            "Email Subject Line:",
            value=st.session_state["scanner_subject"],
            placeholder="e.g. Account Security Alert or Project Progress Update",
        )
        body_input = st.text_area(
            "Email Message Body:",
            value=st.session_state["scanner_body"],
            height=200,
            placeholder="Paste raw email message content here...",
        )

        col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 3])
        with col_btn1:
            submit_scan = st.form_submit_button("🔍 Scan Email", type="primary", use_container_width=True)
        with col_btn2:
            reset_scan = st.form_submit_button("🔄 Clear / Reset", use_container_width=True)

    if reset_scan:
        st.session_state["scanner_subject"] = ""
        st.session_state["scanner_body"] = ""
        st.session_state["scanner_result"] = None
        st.rerun()

    # Inference Execution
    if submit_scan:
        if not subject_input.strip() and not body_input.strip():
            st.error("⚠️ Input Validation Error: Please provide an email subject or message body to analyze.")
            st.session_state["scanner_result"] = None
        else:
            try:
                with st.spinner("Analyzing message with Naïve Bayes pipeline..."):
                    res = predict_email(subject=subject_input, body=body_input, explain=True)
                    st.session_state["scanner_result"] = res
            except SpamGuardValidationError as val_err:
                st.error(f"⚠️ Validation Error: {val_err}")
                st.session_state["scanner_result"] = None
            except SpamGuardModelError as mod_err:
                st.error(f"❌ Model Error: {mod_err}")
                st.session_state["scanner_result"] = None
            except Exception as exc:
                st.error(f"❌ Unexpected Error: {exc}")
                st.session_state["scanner_result"] = None

    # Render Classification Results
    res = st.session_state.get("scanner_result")
    if res is not None:
        st.markdown("---")
        st.markdown("### 📊 Classification Verdict")

        # Prominent Result Banner
        if res.is_spam:
            st.markdown(
                f"""
                <div class="verdict-banner-spam">
                    <div class="verdict-title">🚨 THREAT DETECTED: SPAM</div>
                    <div>This message has been flagged as suspicious or promotional spam with 
                    <b>{res.spam_probability * 100:.2f}%</b> estimated probability.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-banner-ham">
                    <div class="verdict-title">✅ VERIFIED: HAM (LEGITIMATE)</div>
                    <div>This message exhibits characteristics of legitimate correspondence with 
                    <b>{res.ham_probability * 100:.2f}%</b> estimated confidence.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Telemetry & Gauge Row
        col_metrics, col_gauge = st.columns([1.3, 1.0])

        with col_metrics:
            st.markdown("#### Message Telemetry")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Label", res.predicted_label)
                st.metric("Spam Probability", f"{res.spam_probability * 100:.2f}%")
            with m2:
                st.metric("Processing Latency", f"{res.processing_time_ms:.2f} ms")
                st.metric("Cleaned Characters", f"{res.cleaned_char_count:,}")

            st.caption(f"Engine: {res.model_identifier}")
            st.info(
                "💡 **Probability Disclaimer:** Probabilities reflect likelihood estimates based on training "
                "vocabulary frequencies. They do not constitute an absolute guarantee or proof of safety."
            )

        with col_gauge:
            gauge_fig = create_gauge_chart(res.spam_probability)
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Feature Explanations Section
        if res.top_features:
            st.markdown("#### 🔬 Token Attribution Analysis")
            st.markdown(
                "The following tokens present in this email exhibited the strongest statistical influence "
                "on the classifier's log-odds calculation:"
            )

            df_feats = format_features_dataframe(res.top_features)
            st.dataframe(df_feats, use_container_width=True, hide_index=True)
            st.caption(
                "* Note: Log-odds contributions are calculated as TF-IDF(w) * [log P(w|Spam) - log P(w|Ham)]. "
                "Positive values lean Spam; negative values lean Ham. These represent learned empirical correlations, not causal proof."
            )


# =====================================================================
# 5. View: Batch CSV Scanner
# =====================================================================

def render_batch_scanner():
    """Renders the vectorized bulk CSV classification interface."""
    st.markdown('<span class="header-badge">● Vectorized Batch Processing</span>', unsafe_allow_html=True)
    st.title("📁 Batch Email Scanner")
    st.markdown("Upload a CSV file containing multiple email records for rapid, vectorized classification.")

    # CSV Template Helper
    with st.expander("ℹ️ CSV Format Guidelines & Sample Template"):
        st.markdown(
            """
            Your uploaded CSV should contain either:
            - **`subject`** and **`body`** columns, OR
            - a single **`text`** column containing combined email content.
            
            Maximum batch size: **10,000 records**. Maximum single email length: **1,000,000 characters**.
            """
        )
        sample_template_df = pd.DataFrame([
            {"subject": "Urgent Claim Cash", "body": "You won free cash prize money bonus guarantee."},
            {"subject": "Sprint Review Notes", "body": "Please review the pull request patch and merge to main."},
            {"subject": "Refinance Mortgage", "body": "Lowest home loan interest rate guarantee available today."},
        ])
        csv_template_bytes = sample_template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Sample CSV Template",
            data=csv_template_bytes,
            file_name="spamguard_sample_template.csv",
            mime="text/csv",
        )

    # File Upload
    uploaded_file = st.file_uploader(
        "Select CSV file for batch scanning:",
        type=["csv"],
        help="Upload standard comma-separated text file (up to 10MB).",
    )

    if uploaded_file is not None:
        try:
            # Safe CSV ingestion
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV with **{len(df_batch):,} records**.")

            # Validate column presence
            has_subj_body = "subject" in df_batch.columns and "body" in df_batch.columns
            has_text = "text" in df_batch.columns
            has_body_only = "body" in df_batch.columns

            if not (has_subj_body or has_text or has_body_only):
                st.error(
                    f"❌ Schema Validation Error: Uploaded file contains columns {list(df_batch.columns)}. "
                    "Expected ('subject', 'body') or 'text'."
                )
                return

            # Display Preview
            st.markdown("#### Input Data Preview (First 5 Rows)")
            st.dataframe(df_batch.head(5), use_container_width=True)

            if len(df_batch) > MAX_BATCH_SIZE:
                st.error(
                    f"⚠️ Batch size limit exceeded: File contains {len(df_batch):,} rows. "
                    f"Maximum allowed limit is {MAX_BATCH_SIZE:,}."
                )
                return

            # Execute Batch Scan Button
            if st.button("🚀 Scan Entire Batch", type="primary", use_container_width=True):
                t_start = time.perf_counter()
                with st.spinner(f"Vectorizing and classifying {len(df_batch):,} emails..."):
                    results = predict_batch(df_batch, explain=False)
                t_total_ms = (time.perf_counter() - t_start) * 1000.0

                # Aggregate Metrics
                total_scanned = len(results)
                spam_count = sum(1 for r in results if r.is_spam)
                ham_count = total_scanned - spam_count
                spam_pct = (spam_count / total_scanned) * 100.0 if total_scanned > 0 else 0.0
                ham_pct = (ham_count / total_scanned) * 100.0 if total_scanned > 0 else 0.0

                st.markdown("---")
                st.markdown("### 📊 Batch Classification Summary")

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total Emails Scanned", f"{total_scanned:,}")
                with b2:
                    st.metric("Spam Detected", f"{spam_count:,}", f"{spam_pct:.1f}% of batch")
                with b3:
                    st.metric("Legitimate (Ham)", f"{ham_count:,}", f"{ham_pct:.1f}% of batch")
                with b4:
                    st.metric("Total Execution Time", f"{t_total_ms:.1f} ms", f"{t_total_ms/total_scanned:.2f} ms/email")

                # Build Clean Output DataFrame (Formula Injection Protected)
                out_rows = []
                for idx, r in enumerate(results):
                    out_rows.append({
                        "Row ID": idx + 1,
                        "Predicted Verdict": sanitize_csv_cell(r.predicted_label),
                        "Spam Probability": f"{r.spam_probability * 100:.2f}%",
                        "Ham Probability": f"{r.ham_probability * 100:.2f}%",
                        "Is Spam": r.is_spam,
                        "Latency (ms)": round(r.processing_time_ms, 2),
                    })
                df_out = pd.DataFrame(out_rows)

                # Results Table with Filter
                st.markdown("#### Detailed Scan Results")
                filter_choice = st.radio(
                    "Filter Results:",
                    options=["Show All", "Spam Only (Threats)", "Ham Only (Legitimate)"],
                    horizontal=True,
                )

                if filter_choice == "Spam Only (Threats)":
                    df_display = df_out[df_out["Is Spam"] == True]
                elif filter_choice == "Ham Only (Legitimate)":
                    df_display = df_out[df_out["Is Spam"] == False]
                else:
                    df_display = df_out

                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # CSV Export Button
                csv_export_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Clean Results (CSV)",
                    data=csv_export_bytes,
                    file_name="spamguard_batch_scan_results.csv",
                    mime="text/csv",
                    help="Exports sanitized classification outputs with formula injection defense.",
                )

        except Exception as exc:
            st.error(f"❌ Batch Processing Error: {exc}")


# =====================================================================
# 6. View: Model Performance Analytics
# =====================================================================

def render_model_analytics():
    """Renders the comprehensive Phase 06 empirical test performance dashboard."""
    st.markdown('<span class="header-badge">● Model Validation & Empirical Metrics</span>', unsafe_allow_html=True)
    st.title("📊 Model Performance Analytics")
    st.markdown(
        "Empirical evaluation results of the frozen **Multinomial Naïve Bayes** pipeline on the "
        "**untouched test partition** (`n = 889 emails`)."
    )

    report = load_evaluation_summary()
    if not report or "final_test_metrics" not in report:
        st.warning("⚠️ Test evaluation metrics could not be loaded from reports/test_evaluation.json.")
        return

    m = report["final_test_metrics"]
    b = report.get("majority_class_baseline", {})
    cm = m.get("confusion_matrix", {})

    # Top KPI Bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Test Accuracy", f"{m['accuracy'] * 100:.2f}%")
        st.caption("Ratio of correct classifications")
    with k2:
        st.metric("Spam Precision", f"{m['spam']['precision'] * 100:.2f}%")
        st.caption("True spam among flagged emails")
    with k3:
        st.metric("Spam Recall", f"{m['spam']['recall'] * 100:.2f}%")
        st.caption("True spam detected (255/272)")
    with k4:
        st.metric("Spam F1-Score", f"{m['spam']['f1_score'] * 100:.2f}%")
        st.caption("Harmonic balance of precision & recall")

    st.markdown("---")

    # Detailed Sub-Tabs
    tab_cm, tab_curves, tab_baseline, tab_gen = st.tabs([
        "🎯 Confusion Matrix",
        "📈 ROC & PR Curves",
        "⚖️ Baseline Comparison",
        "🔬 Generalization & Stability",
    ])

    with tab_cm:
        st.markdown("### Test Confusion Matrix Breakdown")
        col_cm_chart, col_cm_text = st.columns([1.2, 1.0])

        with col_cm_chart:
            fig_cm = create_confusion_matrix_chart(cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm_text:
            st.markdown(
                f"""
                #### Matrix Interpretation:
                - **True Negatives (`TN = {cm.get('tn', 610)}`):** 98.87% of legitimate emails were correctly allowed into the inbox.
                - **True Positives (`TP = {cm.get('tp', 255)}`):** 93.75% of malicious/promotional spam messages were caught.
                - **False Positives (`FP = {cm.get('fp', 7)}`):** Only 7 legitimate emails (1.13% Type I error) were misidentified as spam.
                - **False Negatives (`FN = {cm.get('fn', 17)}`):** Only 17 spam messages (6.25% Type II error) bypassed detection.
                
                *In email cybersecurity, keeping False Positives minimal is vital to ensure important communications are not lost.*
                """
            )

        # Static figure expander
        cm_fig_path = FIGURES_DIR / "test_confusion_matrix.png"
        if cm_fig_path.exists():
            with st.expander("View Publication-Quality Static Chart"):
                st.image(str(cm_fig_path), caption="Phase 06 Publication Confusion Matrix Heatmap")

    with tab_curves:
        st.markdown("### Discrimination Power: ROC & Precision-Recall Curves")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Area Under ROC Curve (ROC-AUC)", f"{m['roc_auc']:.4f}", "Excellent Separability")
            roc_fig_path = FIGURES_DIR / "test_roc_curve.png"
            if roc_fig_path.exists():
                st.image(str(roc_fig_path), caption="Receiver Operating Characteristic (ROC) Curve")
        with c2:
            st.metric("Average Precision (PR-AUC)", f"{m['average_precision']:.4f}", "High Precision across Recalls")
            pr_fig_path = FIGURES_DIR / "test_precision_recall_curve.png"
            if pr_fig_path.exists():
                st.image(str(pr_fig_path), caption="Precision-Recall Curve (Spam Positive Class)")

    with tab_baseline:
        st.markdown("### Model vs. Zero-Intelligence Majority Baseline")
        st.markdown(
            "A naive majority classifier that blindly predicts 'Ham' for every email would achieve "
            "**69.40% accuracy** while catching **0% of spam**. The chart below proves why accuracy alone "
            "is dangerously misleading on imbalanced security datasets."
        )
        fig_base = create_baseline_comparison_chart(m, b)
        st.plotly_chart(fig_base, use_container_width=True)

    with tab_gen:
        st.markdown("### Validation vs. Held-Out Test Generalization")
        st.markdown(
            "Comparing Phase 05 validation performance against Phase 06 test results verifies that the model "
            "did not overfit the validation set during hyperparameter selection ($\alpha = 0.1$)."
        )
        v_comp = report.get("validation_vs_test_comparison", {})
        val_m = v_comp.get("validation_phase_05", {})
        test_m = v_comp.get("final_test_phase_06", {})
        delta = v_comp.get("generalization_delta", {})

        gen_df = pd.DataFrame([
            {"Metric": "Overall Accuracy", "Phase 05 Validation": "97.86%", "Phase 06 Test": f"{test_m.get('accuracy', 0.973)*100:.2f}%", "Generalization Delta": f"{delta.get('accuracy_delta', -0.0056)*100:+.2f}%", "Status": "Stable (< 0.6% change)"},
            {"Metric": "Spam Precision", "Phase 05 Validation": "98.85%", "Phase 06 Test": f"{test_m.get('spam_precision', 0.9733)*100:.2f}%", "Generalization Delta": f"{delta.get('spam_precision_delta', -0.0152)*100:+.2f}%", "Status": "Robust (1.52% delta)"},
            {"Metric": "Spam Recall", "Phase 05 Validation": "94.14%", "Phase 06 Test": f"{test_m.get('spam_recall', 0.9375)*100:.2f}%", "Generalization Delta": f"{delta.get('spam_recall_delta', -0.0039)*100:+.2f}%", "Status": "Consistent (-0.39% delta)"},
            {"Metric": "Spam F1-Score", "Phase 05 Validation": "96.44%", "Phase 06 Test": f"{test_m.get('spam_f1_score', 0.9551)*100:.2f}%", "Generalization Delta": f"{delta.get('spam_f1_delta', -0.0093)*100:+.2f}%", "Status": "High Fidelity (< 1.0% change)"},
        ])
        st.dataframe(gen_df, use_container_width=True, hide_index=True)


# =====================================================================
# 7. View: About & Methodology
# =====================================================================

def render_about():
    """Renders academic documentation, architecture explanation, and project limits."""
    st.markdown('<span class="header-badge">● Academic Micro Project – 05</span>', unsafe_allow_html=True)
    st.title("ℹ️ About & System Architecture")

    st.markdown(
        """
        ### Project Overview
        **SpamGuard AI** is an academic Artificial Intelligence micro project developed for coursework in 
        Machine Learning, Natural Language Processing, and Cybersecurity.
        
        #### Course Outcome Alignment:
        - **CO2 (Algorithm Selection):** Justifying and implementing classical **Multinomial Naïve Bayes** for sparse text data over computationally heavy deep neural networks.
        - **CO4 (Model Engineering):** Building a leak-free, modular Scikit-learn Pipeline with unigram/bigram **TF-IDF feature extraction** (20,000 max features) and tuned smoothing ($\alpha = 0.1$).
        - **CO5 (Evaluation Metrics):** Implementing rigorous statistical metrics (Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC) on a strictly held-out test partition.
        """
    )

    st.markdown("---")

    # Pipeline Flowchart
    st.markdown("### 🔄 End-to-End Classification Pipeline")
    st.code(
        """
        +------------------+     +------------------------+     +-----------------------+
        | Raw Email Input  | --> | Unicode & Whitespace   | --> | TF-IDF Vectorizer     |
        | (Subject + Body) |     | Normalization (clean)  |     | (20,000 Features)     |
        +------------------+     +------------------------+     +-----------------------+
                                                                            |
                                                                            v
        +------------------+     +------------------------+     +-----------------------+
        | Output: Ham/Spam | <-- | Probability Calibration| <-- | MultinomialNB Model   |
        | + Feature Reason |     | (Softmax Likelihood)   |     | (alpha = 0.1)         |
        +------------------+     +------------------------+     +-----------------------+
        """,
        language="text",
    )

    st.markdown("---")
    st.markdown("### 📚 Dataset Provenance & Limitations")
    st.markdown(
        """
        - **Corpus Source:** Authentic email archives from the **Apache SpamAssassin Public Corpus**.
        - **Record Count:** 5,923 clean emails (4,109 Ham, 1,814 Spam) partitioned into stratified 70% Train, 15% Validation, and 15% Test sets.
        - **Temporal Shift Notice:** The corpus reflects early 2000s email formats. Modern phishing campaigns frequently employ homoglyphs, generative AI text, and dynamic redirect URLs.
        - **Security Warning:** A 'Ham' prediction means low historical spam probability based on text vocabulary. It does not certify that an email is free from zero-day malware or weaponized attachments.
        """
    )


# =====================================================================
# 8. Main Router
# =====================================================================

def main():
    selected_tab = render_sidebar()

    if selected_tab == "🏠 Overview Dashboard":
        render_overview()
    elif selected_tab == "🔍 Email Scanner":
        render_email_scanner()
    elif selected_tab == "📁 Batch CSV Scanner":
        render_batch_scanner()
    elif selected_tab == "📊 Model Analytics":
        render_model_analytics()
    elif selected_tab == "ℹ️ About & Methodology":
        render_about()

    # Academic Footer
    st.markdown(
        f"""
        <div class="academic-footer">
            <b>{APP_TITLE}</b> v{APP_VERSION} &bull; Academic Micro Project – 05 &bull; 
            Cybersecurity & Email Filtering &bull; Multinomial Naïve Bayes + TF-IDF Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
