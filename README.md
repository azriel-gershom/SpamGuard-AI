# SpamGuard AI — AI-Based Spam Email Classifier

> **Academic AI Micro Project** | **Domain:** Cybersecurity / Email Filtering  
> **Course Competency Mapping:** CO2 (Algorithm Selection), CO4 (Model Engineering), CO5 (Evaluation Metrics)

---

## 📌 Project Overview

### Problem Statement
Unsolicited electronic mail (spam) represents a persistent threat vector across digital communications. Spam encompasses phishing campaigns, malware dissemination, deceptive advertisements, and social engineering attempts. Traditional rule-based filters struggle to adapt to evolving lexical patterns and adversarial evasion techniques.

### Proposed Solution
**SpamGuard AI** provides an automated, machine-learning-driven text classification pipeline. By converting email text into numerical feature representations using **Term Frequency-Inverse Document Frequency (TF-IDF)** and applying a probabilistic **Multinomial Naïve Bayes** classifier, SpamGuard AI categorizes incoming emails as either **Ham (Legitimate)** or **Spam** with high computational efficiency and demonstrable statistical precision.

---

## 🧠 Core Algorithm: Multinomial Naïve Bayes

The core model selected for text categorization is the **Multinomial Naïve Bayes (MNB)** classifier:
- **Foundational Principle:** Applies Bayes' Theorem under the conditional independence assumption between features:
  $$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{n} P(x_i \mid c)$$
- **Why Multinomial NB?** It is tailored for discrete count data and term frequencies (such as word counts or TF-IDF weights) in text documents, outperforming Gaussian models on sparse textual representations.
- **Laplace Smoothing:** Uses additive smoothing ($\alpha = 1.0$) to mitigate the zero-probability problem for tokens unseen in specific classes during training.

---

## 🛠️ Technology Stack

| Category | Component / Tool | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core programming runtime |
| **Machine Learning** | Scikit-learn | TF-IDF Vectorization, Multinomial Naïve Bayes, Train/Test Split |
| **Data Processing** | Pandas, NumPy | Structured tabular processing and numerical operations |
| **Model Persistence** | Joblib | Model artifact serialization (`.joblib`) |
| **Interactive UI** | Streamlit | Real-time web dashboard for email classification |
| **Visualization** | Plotly, Matplotlib | Confusion matrices, probability charts, token importance |
| **Testing & Quality** | Pytest | Automated smoke, unit, and integration testing |
| **Version Control** | Git | Systematic phase-by-phase version control |

---

## 🏗️ Repository Architecture

```text
SpamGuard-AI/
├── app.py                  # Streamlit web application starter
├── requirements.txt        # Pinned project dependencies
├── README.md               # Project documentation and specifications
├── .gitignore              # Git ignore rules for venv, cache, and artifacts
├── src/
│   ├── __init__.py         # Core package definition
│   └── config.py           # Path constants and pipeline configurations
├── data/
│   ├── raw/                # Original authentic datasets (.gitkeep)
│   └── processed/          # Cleaned, tokenized datasets (.gitkeep)
├── models/                 # Serialized model & vectorizer artifacts (.gitkeep)
├── reports/
│   └── figures/            # Evaluation graphs, confusion matrices (.gitkeep)
├── tests/
│   └── test_smoke.py       # Pytest smoke test suite
└── docs/                   # Additional architecture & academic documentation
```

---

## 🚦 Implementation Status

### ✅ Implemented in Phase 01 (Foundation)
- [x] Initialized Git repository with clean `.gitignore`.
- [x] Verified Python 3.12.13 runtime and created `.venv` virtual environment.
- [x] Defined complete directory tree with `.gitkeep` placeholders.
- [x] Centralized project configuration (`src/config.py`).
- [x] Pinned and installed dependencies (`requirements.txt`).
- [x] Functional minimal Streamlit starter interface (`app.py`).
- [x] Pytest smoke test suite verifying paths, imports, and syntax.

### ⏳ Planned Future Phases
- [ ] **Phase 02 (Data Ingestion & Validation):** Acquire authentic, real-world labeled email data (e.g., SMS/Email spam corpus); clean and validate without leakage.
- [ ] **Phase 03 (Feature Engineering & Model Training):** Stratified train/test split; TF-IDF fit strictly on training set; Multinomial Naïve Bayes training; compute genuine Accuracy, Precision, Recall, and F1-Score.
- [ ] **Phase 04 (Inference Engine & Testing):** Standalone inference class with probability scoring and keyword explanations, isolated from UI.
- [ ] **Phase 05 (Streamlit Live Application):** Live interactive email testing, confidence meters, and evaluation analytics.
- [ ] **Phase 06 (Project Review & Finalization):** Final academic report documentation, end-to-end verification, and release tagging.

---

## ⚡ Setup & Installation

### 1. Prerequisites
Ensure Python 3.12 is installed on your system.

### 2. Clone / Open Project
```bash
cd /path/to/SpamGuard-AI
```

### 3. Activate Virtual Environment
```bash
# macOS / Linux:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Smoke Tests
```bash
pytest -v
```

### 6. Launch Streamlit Application
```bash
streamlit run app.py
```
By default, the application runs on `http://localhost:8501`.
