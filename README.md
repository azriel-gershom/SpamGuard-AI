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

## 📦 Dataset & Ingestion Pipeline (Phase 02)

### Official Data Source
SpamGuard AI utilizes authentic, real-world labeled email records from the **[Apache SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)**.

### Selected Archives
The following 5 standard archives are ingested over secure HTTPS:
1. `20030228_easy_ham.tar.bz2` (Standard legitimate personal & mailing list emails)
2. `20030228_easy_ham_2.tar.bz2` (Additional clean ham messages)
3. `20030228_hard_ham.tar.bz2` (Legitimate emails with formatting/HTML that closely mimic spam)
4. `20030228_spam.tar.bz2` (Authentic unsolicited commercial and fraudulent emails)
5. `20050311_spam_2.tar.bz2` (Modernized spam collection from subsequent campaigns)

### Label Definitions
Binary classification target:
- **Ham (`0`):** Legitimate emails (`easy_ham`, `easy_ham_2`, `hard_ham`).
- **Spam (`1`):** Unsolicited/malicious emails (`spam`, `spam_2`).

### Safe Email Parsing & Sanitization Architecture
Implemented in `src/prepare_data.py`:
- **Archive Ingestion Security:** Tar archives are opened and read directly as byte streams without extracting unverified members to the local filesystem. Members are verified to be regular files; symlinks, directories, path traversal sequences (`..`), null bytes, oversized members (>10MB), and administrative shell scripts (`cmds`) are strictly rejected.
- **MIME & Encoding Normalization:** Decodes multi-part MIME headers across multiple character sets (UTF-8, ISO-8859-1, Windows-1252, ASCII) via Python's standard `email` and `email.header` modules.
- **Body Extraction & HTML Fallback:** Walks message parts to extract `text/plain` bodies while strictly excluding binary attachments (`Content-Disposition: attachment`). When only HTML is present (`text/html`), a custom parser (`SafeHTMLTextExtractor`) safely extracts human-readable text while discarding `<script>`, `<style>`, and other non-content tags without executing or rendering HTML.
- **Leakage Prevention:** Server-generated transport headers, routing hops, and SpamAssassin-generated scoring tags (such as `X-Spam-Status`, `X-Spam-Flag`) are completely excluded. Only the decoded `subject` and `body` constitute the classification `text`.
- **Text Normalization:** Applies Unicode NFKC normalization, standardizes line breaks (`\r\n` $\rightarrow$ `\n`), strips null bytes, and trims whitespace while preserving spam indicators (URLs, monetary figures, capitalization, punctuation).
- **Deduplication & Conflict Resolution:** Calculates SHA-256 hashes of normalized content. Identical texts appearing across multiple archives are deduplicated deterministically. Conflicting-label instances (identical content flagged as both Ham and Spam) are flagged and excluded entirely.

### Dataset Metrics (`reports/data_summary.json`)
- **Total Source Archive Members:** 6,056 (including non-email administrative entries)
- **Parsed Email Messages:** 6,046
- **Duplicates Removed:** 123
- **Conflicting Label Messages:** 0
- **Final Usable Messages:** 5,923
  - **Ham (`0`):** 4,109 (69.37%)
  - **Spam (`1`):** 1,814 (30.63%)
- **Character Length Summary:** Minimum: 9 | Maximum: 194,844 | Mean: 2,063.1 | Median: 1,084.0

---

## 📊 Exploratory Data Analysis (Phase 03)

Implemented in `src/analyze_data.py`, Phase 03 conducts an exhaustive, descriptive examination of email characteristics without modifying the underlying data or fitting machine learning classifiers.

### Execution Command
```bash
python -m src.analyze_data
```

### Key Statistical Findings & Observable Patterns
1. **Class Imbalance & Ratio:**
   - 4,109 Ham (69.37%) vs. 1,814 Spam (30.63%), establishing an imbalance ratio of **2.27 : 1**.
   - A naive majority-class model would achieve 69.37% Accuracy while catching 0% of spam. This demonstrates why **Precision, Recall, and F1-score (CO5)** are essential evaluation metrics.
2. **Length Distributions:**
   - **Spam Emails:** Mean = 2,466.1 characters, Median = 1,226.0 characters, Mean Word Count = 320.8 words.
   - **Ham Emails:** Mean = 1,885.2 characters, Median = 1,022.0 characters, Mean Word Count = 273.7 words.
   - Both classes exhibit heavy-tailed distributions with extreme outliers. Spam messages have higher median lengths due to HTML fallback content, tracking tokens, and promotional disclaimers.
3. **Punctuation & Urgency Markers:**
   - **Exclamation Marks (`!`):** Present in **78.61%** of spam emails (mean = 5.84 per email), compared to only **33.56%** of ham emails (mean = 0.86 per email).
4. **Structural Lexical Indicators (URLs & Digits):**
   - **Digits:** Present in **93.83%** of spam and **89.39%** of ham.
   - **URLs:** Found in **85.08%** of ham emails and **56.34%** of spam emails. In developer mailing-list datasets (such as SpamAssassin), legitimate messages consistently append subscription, archive, and repository URLs in footers, demonstrating that "URLs imply spam" is a flawed heuristic.
5. **Top Distinctive Frequent Terms (Non-Stop-Words):**
   - **Spam:** `font`, `com`, `http`, `email`, `free`, `www`, `click`, `nbsp`, `money`, `business`, `mail`.
   - **Ham:** `http`, `com`, `www`, `net`, `list`, `linux`, `users`, `just`, `use`, `listinfo`, `lists`, `org`, `rpm`.

### Generated Visualizations (`reports/figures/`)
1. `class_distribution.png`: Bar chart of class counts and percentages with imbalance ratio annotation.
2. `email_length_distribution.png`: Character length histogram comparing Ham and Spam density (with 10,000 char visual clipping threshold disclosed).
3. `length_comparison.png`: Side-by-side boxplots showing median, IQR, and mean diamond markers.
4. `top_spam_words.png`: Horizontal bar chart of the top 20 frequent words in spam.
5. `top_ham_words.png`: Horizontal bar chart of the top 20 frequent words in ham.
6. `url_pattern_comparison.png`: Grouped comparative bar chart of URL, digit, and exclamation mark presence.

### Comprehensive Reports
- `reports/eda_summary.json`: Complete machine-readable summary statistics and distributions.
- `reports/eda_findings.md`: In-depth academic discussion covering data quality, observed class differences, class imbalance implications, and metric justification.

---

## 🔒 Text Preprocessing & Leakage-Safe Data Splitting (Phase 04)

Implemented in `src/preprocess.py`, Phase 04 standardizes email text and partitions the dataset into isolated, stratified subsets while strictly enforcing data leakage prevention protocols.

### Execution Command
```bash
python -m src.preprocess --split
```

### Partitioning Methodology & Purpose
The dataset is partitioned into three disjoint subsets:
- **Training Set (70% — 4,145 emails):** Teaches the model feature associations and class prior probabilities.
- **Validation Set (15% — 889 emails):** Used to tune hyperparameters, adjust decision thresholds, and validate convergence.
- **Test Set (15% — 889 emails):** Held out strictly for final, uncorrupted performance reporting.

### Why Stratification Matters
Given the natural **69.37% Ham / 30.63% Spam** class distribution, standard unstratified random sampling risks introducing varying class frequencies across subsets. Stratified sampling preserves the identical proportion in every partition:
- **Training:** 2,876 Ham (69.38%) | 1,269 Spam (30.62%)
- **Validation:** 616 Ham (69.29%) | 273 Spam (30.71%)
- **Test:** 617 Ham (69.40%) | 272 Spam (30.60%)

### Content-Group-Aware Splitting & Leakage Prevention
To guarantee zero data contamination:
1. **Zero Shared Message IDs:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Zero Identical Content Overlap:** Emails with identical normalized text (e.g., duplicated broadcast campaigns or matching newsletters) are grouped by content hash and assigned to the same partition, preventing identical text from appearing across train and test sets.
3. **100% Coverage:** Every single one of the 5,923 original emails is allocated to exactly one partition.
4. **TF-IDF Vocabulary Isolation:** The vectorizer factory (`build_tfidf_vectorizer()`) returns an **unfitted** instance. Fitting occurs strictly on the training partition in Phase 05, ensuring terms unique to validation/test sets cannot artificially influence vocabulary extraction or inverse document frequencies (IDF).

### Residual Leakage Consideration
While exact content and ID duplicates are strictly isolated, real-world mailing-list archives feature multi-turn conversation threads that quote preceding messages. Practitioners must recognize that residual lexical correlation can persist across related threads.

### Unfitted TF-IDF Configuration
- **Analyzer:** `word`
- **N-gram Range:** `(1, 2)` (unigrams and bigrams to capture phrases such as `free money` and `click here`)
- **Max Features:** `20,000` (bounds vocabulary to mitigate dimensionality curse)
- **Document Frequency Bounds:** `min_df=2` (prunes hapax legomena), `max_df=0.98` (prunes ubiquitous terms)
- **Sublinear Term Frequency:** `sublinear_tf=True` (scales term frequency logarithmically: $1 + \log(\text{tf})$)
- **Normalization:** `norm="l2"`

### Generated Split Artifacts
- `data/processed/splits/train.csv` (4,145 rows)
- `data/processed/splits/validation.csv` (889 rows)
- `data/processed/splits/test.csv` (889 rows)
- `reports/split_summary.json` (audit report capturing hashes, counts, and disjointness verification)

---

## 🏗️ Repository Architecture

```text
SpamGuard-AI/
├── app.py                  # Streamlit web application starter
├── requirements.txt        # Pinned project dependencies
├── README.md               # Project documentation and specifications
├── .gitignore              # Git ignore rules for venv, cache, raw/processed data, and splits
├── pytest.ini              # Pytest configuration (pythonpath = .)
├── src/
│   ├── __init__.py         # Core package definition
│   ├── config.py           # Path constants and pipeline configurations
│   ├── prepare_data.py     # Phase 02: Safe email download, parsing, and export
│   ├── analyze_data.py     # Phase 03: Exploratory data analysis & visualization engine
│   └── preprocess.py       # Phase 04: Text cleaning, stratified splitting & TF-IDF factory
├── data/
│   ├── raw/                # Downloaded Apache SpamAssassin tar.bz2 archives (gitignored)
│   └── processed/          # Cleaned emails.csv dataset (gitignored)
│       └── splits/         # Phase 04: train.csv, validation.csv, test.csv (gitignored)
├── models/                 # Serialized model & vectorizer artifacts (.gitkeep)
├── reports/
│   ├── data_summary.json   # Phase 02: Dataset ingestion statistics & SHA-256 hashes
│   ├── eda_summary.json    # Phase 03: Quantitative EDA summary metrics
│   ├── eda_findings.md     # Phase 03: Comprehensive academic findings report
│   ├── split_summary.json  # Phase 04: Partition statistics & leakage audit report
│   └── figures/            # Phase 03: Generated visualization charts
│       ├── class_distribution.png
│       ├── email_length_distribution.png
│       ├── length_comparison.png
│       ├── top_spam_words.png
│       ├── top_ham_words.png
│       └── url_pattern_comparison.png
├── tests/
│   ├── test_smoke.py       # Phase 01: Project foundation smoke tests
│   ├── test_prepare_data.py# Phase 02: Unit and integration tests for data parsing
│   ├── test_eda.py         # Phase 03: Unit and integration tests for EDA module
│   └── test_preprocess.py  # Phase 04: Unit and integration tests for preprocessing & splits
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

### ✅ Implemented in Phase 02 (Dataset Preparation)
- [x] Automated HTTPS download and SHA-256 verification of 5 official Apache SpamAssassin archives.
- [x] Secure in-memory archive extraction preventing directory traversal and symlink vulnerabilities.
- [x] Robust MIME subject decoding and character encoding fallbacks.
- [x] Multipart extraction prioritizing plain text with non-executing HTML text fallback.
- [x] Exclusion of attachments and transport/spam metadata to prevent data leakage.
- [x] Content-hash deduplication and conflicting-label exclusion.
- [x] Exported standardized dataset to `data/processed/emails.csv` (5,923 records).
- [x] Exported verified dataset metrics to `reports/data_summary.json`.
- [x] Comprehensive unit and integration test suite (`tests/test_prepare_data.py`).

### ✅ Implemented in Phase 03 (Exploratory Data Analysis)
- [x] Modular EDA module with automated validation, length statistics, and pattern metrics (`src/analyze_data.py`).
- [x] Verified dataset integrity: 5,923 valid emails, 0 missing texts/labels, 0 duplicates, 0 conflicting labels.
- [x] Generated 6 professional publication-quality figures under `reports/figures/`.
- [x] Exported comprehensive metrics to `reports/eda_summary.json`.
- [x] Authored thorough academic findings report (`reports/eda_findings.md`).
- [x] Pytest unit and integration test suite with 9 dedicated tests (`tests/test_eda.py`).

### ✅ Implemented in Phase 04 (Preprocessing & Splitting)
- [x] Safe, consistent text normalization (Unicode NFKC, line breaks, null bytes) (`src/preprocess.py`).
- [x] Group-aware stratified train (70%), validation (15%), and test (15%) splitting.
- [x] Enforced strict data leakage prevention (0 shared IDs, 0 shared content, 100% coverage).
- [x] Reusable, unfitted TF-IDF vectorizer factory (`build_tfidf_vectorizer`).
- [x] Exported partitioned CSV splits under `data/processed/splits/` (gitignored).
- [x] Generated deterministic audit report (`reports/split_summary.json`).
- [x] Automated unit and integration test suite with 14 tests (`tests/test_preprocess.py`).
- [x] Full test suite (38 tests) passing cleanly.

### ⏳ Planned Future Phases
- [ ] **Phase 05 (Model Training & Evaluation):** Fit TF-IDF and Multinomial Naïve Bayes pipeline on training data only; hyperparameter tuning on validation; final evaluation on test set (Accuracy, Precision, Recall, F1, Confusion Matrix); serialize artifacts (`.joblib`).
- [ ] **Phase 06 (Inference Engine & Testing):** Standalone inference module with confidence calibration and top keyword explanation, decoupled from UI.
- [ ] **Phase 07 (Streamlit Live Application):** Interactive web application with real-time inspection, probability visualizers, and evaluation metric dashboards.
- [ ] **Phase 08 (Project Review & Finalization):** Final academic report documentation, end-to-end verification, and release tagging.

---

## ⚡ Setup & Execution

### 1. Prerequisites
Ensure Python 3.12 is installed on your system.

### 2. Activate Virtual Environment
```bash
# macOS / Linux:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Dataset Preparation Pipeline (Phase 02)
```bash
python -m src.prepare_data --download
```

### 5. Run Exploratory Data Analysis Pipeline (Phase 03)
```bash
python -m src.analyze_data
```

### 6. Run Preprocessing & Splitting Pipeline (Phase 04)
```bash
python -m src.preprocess --split
```

### 7. Run Test Suite
```bash
pytest -v
```

### 8. Launch Streamlit Application
```bash
streamlit run app.py
```
By default, the application runs on `http://localhost:8501`.
