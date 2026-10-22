"""SpamGuard AI — Streamlit Application Starter.

Academic Micro Project: AI-Based Spam Email Classifier.
Phase 01: Project Foundation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from src.config import APP_SUBTITLE, APP_TITLE, APP_VERSION

# Page Configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Minimal Cybersecurity / Modern Dark Aesthetics)
st.markdown(
    """
    <style>
    .main {
        padding-top: 1.5rem;
    }
    .security-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .card-box {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .tech-pill {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        background-color: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Information
with st.sidebar:
    st.markdown("### 🛡️ System Specifications")
    st.markdown(f"**Version:** `{APP_VERSION}`")
    st.markdown("**Domain:** Cybersecurity / Email Filtering")
    st.markdown("**Course Alignment:** CO2, CO4, CO5")
    st.markdown("---")
    st.markdown("### 📋 Implementation Roadmap")
    st.markdown("- [x] **Phase 01:** Foundation & Architecture")
    st.markdown("- [ ] **Phase 02:** Real Labeled Data Ingestion")
    st.markdown("- [ ] **Phase 03:** TF-IDF & Multinomial NB Model")
    st.markdown("- [ ] **Phase 04:** Standalone Inference Engine")
    st.markdown("- [ ] **Phase 05:** Streamlit Live Inspector UI")
    st.markdown("- [ ] **Phase 06:** Final Validation & Reports")
    st.markdown("---")
    st.caption("SpamGuard AI — Academic AI Micro Project")

# Header Section
st.markdown('<span class="security-badge">● System Online — Phase 01</span>', unsafe_allow_html=True)
st.title(f"🛡️ {APP_TITLE}")
st.subheader(APP_SUBTITLE)

st.markdown(
    """
    **SpamGuard AI** is an academic AI micro project engineered to detect and classify spam emails
    using machine learning and statistical Natural Language Processing (NLP). 
    It leverages **TF-IDF (Term Frequency-Inverse Document Frequency)** feature representation and a 
    **Multinomial Naïve Bayes** probabilistic classifier designed specifically for text classification tasks.
    """
)

# Foundation Status Notice
st.info(
    """
    ℹ️ **System Status: Foundation Stage Active (Phase 01)**  
    The machine learning pipeline and inference engine are not yet trained. 
    In adherence to sound engineering practices, no synthetic results or placeholder predictions are shown.
    Model training on authentic labeled email data will be integrated in subsequent phases.
    """,
    icon="⚙️",
)

st.markdown("---")

# Layout: Two Columns for Overview & Tech Stack
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🎯 Problem Statement & Objectives")
    st.markdown(
        """
        Unsolicited bulk email (spam) presents serious security and operational threats, 
        including phishing vectors, malware delivery, and wasted network throughput.
        
        **Core Engineering Objectives:**
        - **Authentic Dataset Training:** Ingest real, labeled email/message corpora without synthetic data.
        - **Leakage Prevention:** Isolate train and test partitions; fit TF-IDF vectorizers solely on training splits.
        - **Rigorous Evaluation:** Demonstrate genuine classification performance via Precision, Recall, and F1-Score.
        - **Decoupled Architecture:** Keep machine learning inference completely separate from the Streamlit presentation layer.
        """
    )

with col2:
    st.markdown("### 💻 Technology Stack")
    st.markdown(
        """
        <div class="card-box">
            <strong>Core Language:</strong><br>
            <span class="tech-pill">Python 3.12</span><br><br>
            <strong>Machine Learning & Statistics:</strong><br>
            <span class="tech-pill">Scikit-learn</span>
            <span class="tech-pill">TF-IDF Vectorizer</span>
            <span class="tech-pill">Multinomial Naïve Bayes</span>
            <span class="tech-pill">NumPy</span>
            <span class="tech-pill">Pandas</span>
            <span class="tech-pill">Joblib</span><br><br>
            <strong>Application & Visualization:</strong><br>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Plotly</span>
            <span class="tech-pill">Matplotlib</span><br><br>
            <strong>Quality Assurance:</strong><br>
            <span class="tech-pill">Pytest</span>
            <span class="tech-pill">Git</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Academic Requirements Section
st.markdown("### 🎓 Academic Competency Mapping")
ac_col1, ac_col2, ac_col3 = st.columns(3)

with ac_col1:
    st.metric(label="Competency CO2", value="Algorithm Selection", help="Multinomial Naïve Bayes probabilistic model for text classification")
    st.caption("Selection and rationale of Naïve Bayes for discrete feature frequency distributions.")

with ac_col2:
    st.metric(label="Competency CO4", value="Model Engineering", help="Data preprocessing, tokenization, and TF-IDF feature extraction")
    st.caption("Feature engineering with strict train/test isolation to prevent data snooping.")

with ac_col3:
    st.metric(label="Competency CO5", value="Evaluation Metrics", help="Precision, Recall, F1-Score, Confusion Matrix")
    st.caption("Quantitative performance validation on unseen test data.")
