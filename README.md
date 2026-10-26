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

## 🤖 Naïve Bayes Model Development (Phase 05)

Implemented in `src/train.py`, Phase 05 trains, tunes, validates, and serializes the first end-to-end Multinomial Naïve Bayes email spam classification pipeline.

### Execution Command
```bash
python -m src.train
```

### Core Architecture & Academic Principles

1. **Multinomial Naïve Bayes Model:**
   The classification decision relies on Bayes' theorem under the conditional feature independence assumption:
   $$P(\text{class} \mid \mathbf{x}) \propto P(\text{class}) \prod_{i=1}^{d} P(w_i \mid \text{class})^{x_i}$$
   In log-space to prevent numerical underflow:
   $$\log P(\text{class} \mid \mathbf{x}) \propto \log P(\text{class}) + \sum_{i=1}^{d} x_i \log P(w_i \mid \text{class})$$
   where $x_i$ represents the TF-IDF weight of token $w_i$, and $P(w_i \mid \text{class})$ is estimated from training document term frequencies.

2. **Laplace / Lidstone Additive Smoothing ($\alpha$):**
   To resolve the zero-frequency problem where an unseen token $w_i$ in training would otherwise collapse $P(w_i \mid \text{class})$ to zero, Laplace/Lidstone smoothing adds parameter $\alpha$ to feature counts:
   $$\hat{P}(w_i \mid c) = \frac{N_{c, i} + \alpha}{N_c + \alpha \cdot |V|}$$
   where $N_{c, i}$ is the aggregate TF-IDF weight for token $i$ in class $c$, $N_c$ is the total weight across all vocabulary in class $c$, and $|V|$ is the vocabulary size (20,000 features).
   - $\alpha = 1.0$: Standard Laplace smoothing.
   - $\alpha < 1.0$: Lidstone smoothing, allowing rarer words stronger discriminative weight while retaining smoothing guarantees.

3. **Strict Train-Only Fitting (Leakage Prevention):**
   - The Scikit-learn `Pipeline([('tfidf', build_tfidf_vectorizer()), ('classifier', MultinomialNB(alpha=alpha))])` is fitted **exclusively on the 4,145 training emails**.
   - The TF-IDF vocabulary, document frequencies, and IDF weights are learned strictly from `train.csv`.
   - **Crucial Rule:** The 889 validation emails are transformed using the fitted training statistics without refitting. The 889 test emails are **not accessed or evaluated** during Phase 05.

### Baseline Comparison: Why Accuracy Alone Fails

To demonstrate the critical necessity of **Precision, Recall, and F1-score (CO5)** on imbalanced datasets, a majority-class baseline was fitted using `DummyClassifier(strategy="most_frequent")`:

| Model | Val Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** (`strategy="most_frequent"`) | 69.29% | 0.00% | 0.00% | 0.0000 |
| **Multinomial Naïve Bayes** ($\alpha = 0.1$) | **97.86%** | **98.85%** | **94.14%** | **0.9644** |

*Key Academic Finding:* The baseline achieves nearly 70% accuracy simply by predicting every email as legitimate ("Ham"), yet it catches zero spam emails. In real-world cybersecurity, high precision (protecting legitimate communications from false quarantines) and high recall (intercepting attacks) are required.

### Hyperparameter Selection on Validation Partition

The smoothing parameter $\alpha$ was tuned over fixed candidate values $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ using the 889 validation emails (616 ham, 273 spam). The candidate with the highest spam-class F1-score was selected:

| Candidate $\alpha$ | Val Accuracy | Val Precision (Spam) | Val Recall (Spam) | Val F1-Score (Spam) | True Positives | False Positives | False Negatives | True Negatives |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\mathbf{0.10}$ (Selected)** | **0.9786** | **0.9885** | **0.9414** | **0.9644** | **257** | **3** | **16** | **613** |
| $0.50$ | 0.9708 | 0.9881 | 0.9158 | 0.9506 | 250 | 3 | 23 | 613 |
| $1.00$ | 0.9640 | 0.9879 | 0.8938 | 0.9385 | 244 | 3 | 29 | 613 |
| $2.00$ | 0.9516 | 1.0000 | 0.8425 | 0.9145 | 230 | 0 | 43 | 616 |
| $5.00$ | 0.8841 | 1.0000 | 0.6227 | 0.7675 | 170 | 0 | 103 | 616 |

> [!IMPORTANT]
> **Validation Notice:** All metrics reported above are **Validation partition metrics** evaluated during model selection. The test partition (889 emails) remains completely untouched and held-out for independent evaluation in Phase 06.

### Model Persistence & Reload Verification

- The optimal pipeline ($\alpha = 0.1$) is serialized to `models/spam_pipeline.joblib` using Joblib.
- The artifact encapsulates the complete fitted TF-IDF vectorizer (20,000 features) and fitted MultinomialNB classifier.
- Post-save verification reloads the pipeline from disk, evaluates 100 validation samples, and verifies bit-for-bit identical discrete class predictions and probability distributions.
- Checksum: SHA-256 integrity hash is recorded in `reports/training_summary.json`.

### Model Interpretability & Learned Feature Associations

Log-probability differences $\log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham})$ capture tokens most strongly associated with each class (recorded in `reports/model_insights.json`):
- **Top Spam Indicators:** `mortgage` (+5.42 diff), `life insurance` (+4.78 diff), `spamassassin sightings` (+4.70 diff), `form below` (+4.69 diff), `refinance` (+4.60 diff), `optout` (+4.51 diff).
- **Top Ham Indicators:** `rpm list` (-5.23 diff), `newsisfree` (-5.14 diff), `freshrpms` (-5.10 diff), `razor` (-4.97 diff), `redhat` (-4.69 diff), `wrote` (-4.81 diff).
- *Disclaimer:* These reflect learned empirical statistical associations in historical data, not causal determinations.

### Historical Dataset Limitations

Users and evaluators should recognize that the Apache SpamAssassin dataset comprises historical emails (circa 2002–2005):
1. **Temporal Shift:** Modern spam tactics (e.g., Unicode zero-width spoofing, adversarial image text, shortened cloud URLs) are absent.
2. **Domain Bias:** The legitimate email corpus primarily reflects open-source developer mailing lists (Linux, RPM, Razor), skewing vocabulary toward developer discussions.

---

## 📈 Final Model Performance Evaluation (Phase 06)

Implemented in `src/evaluate.py`, Phase 06 evaluates the frozen, serialized pipeline artifact (`models/spam_pipeline.joblib`) strictly against the quarantined, held-out test partition (`data/processed/splits/test.csv`).

### Execution Command
```bash
python -m src.evaluate
```

### Empirical Test Partition Results (`n=889`: 617 Ham, 272 Spam)

| Evaluation Metric | Naïve Bayes Classifier | Majority Baseline | Delta vs Baseline |
|---|---:|---:|---:|
| **Test Accuracy** | **97.30%** | 69.40% | +27.90% |
| **Balanced Accuracy** | **96.31%** | 50.00% | +46.31% |
| **Spam Precision (PPV)** | **97.33%** | 0.00% | +97.33% |
| **Spam Recall (Sensitivity)** | **93.75%** | 0.00% | +93.75% |
| **Spam F1-Score** | **95.51%** | 0.00% | +95.51% |
| **Ham Precision** | **97.29%** | 69.40% | +27.89% |
| **Ham Recall (Specificity)** | **98.87%** | 100.00% | -1.13% |
| **Ham F1-Score** | **98.07%** | 81.94% | +16.13% |
| **Macro Average F1** | **96.79%** | 40.97% | +55.82% |
| **Weighted Average F1** | **97.29%** | 56.87% | +40.42% |
| **False Positive Rate (FPR)** | **1.13%** | 0.00% | +1.13% |
| **False Negative Rate (FNR)** | **6.25%** | 100.00% | -93.75% |
| **ROC-AUC** | **0.9965** | 0.5000 | +0.4965 |
| **Average Precision (PR-AUC)** | **0.9926** | 0.3060 | +0.6866 |

### Confusion Matrix Breakdown
- **True Negatives (`TN`):** 610 / 617 legitimate emails correctly allowed (98.87%).
- **True Positives (`TP`):** 255 / 272 spam emails caught and filtered (93.75%).
- **False Positives (`FP`):** 7 legitimate emails flagged (Type I error rate: 1.13%).
- **False Negatives (`FN`):** 17 spam emails missed (Type II error rate: 6.25%).

### Validation vs Test Generalization Stability
- **Accuracy:** 97.86% (Val) $\rightarrow$ 97.30% (Test) [$\Delta = -0.56\%$]
- **Spam Precision:** 98.85% (Val) $\rightarrow$ 97.33% (Test) [$\Delta = -1.52\%$]
- **Spam Recall:** 94.14% (Val) $\rightarrow$ 93.75% (Test) [$\Delta = -0.39\%$]
- **Spam F1-Score:** 96.44% (Val) $\rightarrow$ 95.51% (Test) [$\Delta = -0.93\%$]

The negligible variance ($<1.0\%$ change across accuracy, recall, and F1-score, with a minor 1.52% precision delta) demonstrates outstanding generalization and confirms the complete absence of test data contamination.

---

## 🔮 Standalone Prediction Engine (Phase 07)

Implemented in `src/predict.py`, Phase 07 provides an isolated, reliable, high-throughput prediction engine ready for integration with the Streamlit frontend (Phase 08) or external APIs.

### Architecture & Capabilities
1. **Cryptographic Model Verification:** Before deserialization, `SpamPredictor.load()` calculates the file's SHA-256 in 64KB blocks and verifies it against the frozen Phase 05 checksum (`c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`). Unverified or modified artifacts are rejected immediately.
2. **Deterministic Preprocessing:** Reproduces the exact Phase 02 subject/body combination (`{subject}\n\n{body}`) and applies Phase 04 text normalization (`clean_text`), guaranteeing that input whitespace, Unicode decomposition (NFKC), and special tokens are treated identically to training samples.
3. **Calibrated Probabilistic Scoring:** Extracts discrete predictions and class probabilities using dynamic class indices from `pipeline.classes_`. Validates that returned probabilities are finite, strictly bounded in $[0, 1]$, and sum to $1.0 \pm 10^{-4}$.
4. **Vectorized High-Throughput Batch Processing:** `predict_batch` accepts collections of dictionaries, tuples, or pandas DataFrames. All records are vectorized simultaneously in a single call to `pipeline.predict()` and `pipeline.predict_proba()`, providing sub-millisecond per-item latency while guaranteeing exact index preservation.
5. **Model Explainability (Log-Odds Contribution):** Features an optional explainability engine calculating token-level log-odds contributions:
   $$\text{contribution}_j = \text{TF-IDF}(w_j) \times \left( \log P(w_j \mid \text{Spam}) - \log P(w_j \mid \text{Ham}) \right)$$
   Tokens with positive contributions lean Spam; negative contributions lean Ham. Explanations reflect learned statistical associations from training data, without claiming causality.
6. **Defensive Security & Privacy:**
   - **Denial-of-Service Defense:** Caps input length at 1,000,000 characters (`MAX_INPUT_CHARS`) and batch size at 10,000 records (`MAX_BATCH_SIZE`).
   - **CSV Formula Injection Prevention:** Escapes dangerous leading formula prefixes (`=`, `+`, `-`, `@`, `\t`, `\r`) during batch export.
   - **Privacy Isolation:** Does not log or echo raw email bodies in batch reports or log streams.
7. **Dataset Limitations & Security Notice:**
   - The model was trained on the historical Apache SpamAssassin corpus (circa 2002–2005).
   - While highly effective on textual spam patterns, it cannot guarantee detection of modern adversarial evasions, zero-day phishing kits, or obfuscated image attachments. Model output represents a statistical probability estimate, not proof of safety.

### CLI Usage Examples
```bash
# Single email classification with feature explanations:
python -m src.predict --subject "Claim Prize" --text "Urgent claim your free cash prize bonus money" --explain

# JSON output mode:
python -m src.predict --subject "Meeting" --text "Team standup notes for Monday morning" --format json

# Secure classification via standard input (avoids shell history leaks):
echo "Confidential message body" | python -m src.predict --stdin

# Batch CSV classification:
python -m src.predict --batch-csv incoming_emails.csv --output-csv classified_emails.csv
```

---

## 🖥️ Streamlit Web Application (Phase 08)

SpamGuard AI includes a full-featured, interactive cybersecurity operations dashboard built with **Streamlit** (`app.py`), backed by the verified inference engine (`src/predict.py`) and UI helper utilities (`src/ui_helpers.py`).

### Launching the Dashboard

```bash
streamlit run app.py
```
*Default URL:* `http://localhost:8501`

### Dashboard Architecture & Sections

The application features persistent sidebar navigation, live cryptographic model integrity telemetry, and five dedicated operational sections:

1. **System Overview & Performance:**
   - Real-time model status badge and SHA-256 integrity indicator (`c652119...`).
   - Official held-out test evaluation KPIs (`reports/test_evaluation.json`): **97.30% Accuracy**, **97.33% Spam Precision**, **93.75% Spam Recall**, **95.51% Spam F1-Score**, **0.9965 ROC-AUC**.
   - Dataset partition breakdown (4,145 train, 889 validation, 889 test).
   - High-level pipeline architecture and operational workflow.

2. **Single Email Scanner:**
   - Interactive Subject and Body text areas with real-time character counters.
   - 4 realistic, pre-populated synthetic quick-fill templates:
     - 🎣 *Phishing Alert:* Deceptive bank account suspension lure.
     - 🎁 *Prize Spam:* Lottery winnings notification with urgent claim links.
     - 💻 *Engineering Ham:* Software pull request discussion and unit test updates.
     - 📚 *Academic Ham:* Faculty project submission deadline reminder.
   - Dynamic verdict banner (**HAM (LEGITIMATE)** with green badge vs. **SPAM DETECTED** with red warning).
   - Inference latency telemetry (in milliseconds).
   - Interactive **Plotly Probability Gauge** ($0\%$ to $100\%$) with calibrated green/amber/red threat thresholds.
   - **Explainability Breakdown:** Mathematical token contribution table displaying word-level log-odds scores:
     $$\text{Contribution}(w) = \text{TF-IDF}(w) \times [\log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham})]$$
     categorized as *Supports SPAM* (positive log-odds) or *Supports HAM* (negative log-odds).
   - Operational security & probabilistic classification disclaimer.

3. **Batch CSV Scanner:**
   - Drag-and-drop CSV file uploader.
   - Built-in **Download CSV Template** button providing a standard header format (`subject,body`).
   - Uploaded dataset preview displaying row count, columns, and data hygiene indicators.
   - Vectorized inference processing via `SpamPredictor.predict_batch()`.
   - Batch summary metrics: Total processed, Spam count, Ham count, and aggregate Spam Ratio.
   - Filterable interactive results table (All, Spam Only, Ham Only).
   - Sanitized CSV export button with automated formula injection neutralization (neutralizes `=, +, -, @` cell prefixes).

4. **Model Performance Analytics:**
   - Interactive **Plotly Confusion Matrix Heatmap** displaying normalized percentages and raw test counts (610 TN, 255 TP, 7 FP, 17 FN).
   - Side-by-side **ROC Curve** ($\text{AUC} = 0.9965$) and **Precision-Recall Curve** ($\text{AP} = 0.9926$) from Phase 06 test evaluation.
   - Interactive **Model vs. Majority-Class Baseline** comparative bar chart demonstrating superior discriminative power (+95.51% Spam F1 improvement).
   - Comprehensive **Validation vs. Test Generalization Table** proving absence of overfitting.

5. **About & Methodology:**
   - Explicit Academic Course Outcome mapping:
     - **CO2:** Text Representation & Algorithm Selection (TF-IDF + Multinomial Naïve Bayes).
     - **CO4:** Model Engineering & Hyperparameter Optimization ($\alpha = 0.1$ Lidstone smoothing).
     - **CO5:** Multi-Metric Performance Evaluation (Precision, Recall, F1, ROC-AUC under 2.27:1 class imbalance).
   - End-to-end data processing and inference flowchart.
   - Dataset provenance documentation (Apache SpamAssassin Public Corpus: 5,923 emails).
   - Scientific acknowledgment of historical limitations (temporal drift and open-source mailing list vocabulary bias).

---

## 🏗️ Repository Architecture

```text
SpamGuard-AI/
├── app.py                  # Phase 08: Streamlit operational web dashboard
├── requirements.txt        # Pinned project dependencies
├── README.md               # Project documentation and specifications
├── .gitignore              # Git ignore rules for venv, cache, raw/processed data, and splits
├── pytest.ini              # Pytest configuration (pythonpath = .)
├── src/
│   ├── __init__.py         # Core package definition
│   ├── config.py           # Path constants and pipeline configurations
│   ├── prepare_data.py     # Phase 02: Safe email download, parsing, and export
│   ├── analyze_data.py     # Phase 03: Exploratory data analysis & visualization engine
│   ├── preprocess.py       # Phase 04: Text cleaning, stratified splitting & TF-IDF factory
│   ├── train.py            # Phase 05: Naïve Bayes pipeline training, tuning & persistence
│   ├── evaluate.py         # Phase 06: Final model performance evaluation pipeline
│   ├── predict.py          # Phase 07: High-reliability prediction engine & CLI
│   └── ui_helpers.py       # Phase 08: Streamlit UI utilities, caching, samples & charts
├── data/
│   ├── raw/                # Downloaded Apache SpamAssassin tar.bz2 archives (gitignored)
│   └── processed/          # Cleaned emails.csv dataset (gitignored)
│       └── splits/         # Phase 04: train.csv, validation.csv, test.csv (gitignored)
├── models/
│   ├── .gitkeep
│   └── spam_pipeline.joblib# Phase 05: Serialized fitted pipeline (gitignored)
├── reports/
│   ├── data_summary.json   # Phase 02: Dataset ingestion statistics & SHA-256 hashes
│   ├── eda_summary.json    # Phase 03: Quantitative EDA summary metrics
│   ├── eda_findings.md     # Phase 03: Comprehensive academic findings report
│   ├── split_summary.json  # Phase 04: Partition statistics & leakage audit report
│   ├── training_summary.json # Phase 05: Training metadata, baseline & validation metrics
│   ├── model_insights.json # Phase 05: Top predictive token log probabilities
│   ├── test_evaluation.json# Phase 06: Final test evaluation metrics & comparison
│   ├── evaluation_findings.md # Phase 06: Comprehensive academic evaluation findings report
│   ├── phase09_qa_report.md# Phase 09: Comprehensive QA, security & hardening audit report
│   ├── performance_benchmark.json # Phase 09: Real hardware latency, throughput & memory metrics
│   └── figures/            # Generated visualization charts
├── scripts/
│   └── profile_benchmarks.py # Phase 09: Automated performance and memory profiling script
├── tests/
│   ├── test_smoke.py       # Phase 01: Project foundation smoke tests
│   ├── test_prepare_data.py# Phase 02: Unit and integration tests for data parsing
│   ├── test_eda.py         # Phase 03: Unit and integration tests for EDA module
│   ├── test_preprocess.py  # Phase 04: Unit and integration tests for preprocessing & splits
│   ├── test_train.py       # Phase 05: Unit and integration tests for training pipeline
│   ├── test_evaluate.py    # Phase 06: Unit and integration tests for evaluation pipeline
│   ├── test_predict.py     # Phase 07: Unit and integration tests for prediction engine
│   ├── test_app.py         # Phase 08: Streamlit AppTest UI integration test suite
│   └── test_hardening.py   # Phase 09: Security hardening, formula injection & edge cases
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

### ✅ Implemented in Phase 05 (Naïve Bayes Model Development)
- [x] Reusable `build_model(alpha)` factory integrating TF-IDF vectorizer and Multinomial Naïve Bayes.
- [x] Strict train-only fitting on 4,145 training emails (zero leakage from validation or test partitions).
- [x] Majority-class baseline benchmark (`DummyClassifier`) demonstrating failure of accuracy-only metric on imbalanced data.
- [x] Deterministic hyperparameter tuning across candidate alphas $[0.1, 0.5, 1.0, 2.0, 5.0]$ on 889 validation emails.
- [x] Optimal parameter selection: $\alpha = 0.1$ achieving **0.9644 validation F1-score**, 98.85% precision, 94.14% recall.
- [x] Pipeline serialization to `models/spam_pipeline.joblib` with SHA-256 checksum and reload prediction parity verification.
- [x] Model interpretability extraction comparing class-conditional log probabilities (`reports/model_insights.json`).
- [x] Complete training metadata and audit summary exported to `reports/training_summary.json`.
- [x] Automated unit and integration test suite with 17 dedicated tests (`tests/test_train.py`).
- [x] Entire project test suite (55 tests) passing cleanly.

### ✅ Implemented in Phase 06 (Final Model Performance Evaluation)
- [x] Independent evaluation pipeline (`src/evaluate.py`) evaluating frozen `models/spam_pipeline.joblib` on untouched held-out test partition (`n=889`: 617 Ham, 272 Spam).
- [x] Pre-inference SHA-256 integrity verification (`c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`).
- [x] Definitive test metrics: **97.30% Accuracy**, **97.33% Spam Precision**, **93.75% Spam Recall**, **95.51% Spam F1-Score**, **98.07% Ham F1-Score**, **0.9965 ROC-AUC**, **0.9926 Average Precision**.
- [x] Empirical comparison against majority-class baseline (demonstrating +95.51 percentage point improvement in Spam F1-score over zero-skill baseline).
- [x] Confusion matrix analysis: 610 TN (98.87%), 255 TP (93.75%), only 7 FP (1.13% Type I error), and 17 FN (6.25% Type II error).
- [x] Generated 5 publication-quality visualization charts in `reports/figures/` (Confusion Matrix, ROC Curve, Precision-Recall Curve, Model vs Baseline, Classification Metrics).
- [x] Generated comprehensive academic reports: `reports/test_evaluation.json` and `reports/evaluation_findings.md`.
- [x] Automated unit and integration test suite with 16 dedicated tests (`tests/test_evaluate.py`).
- [x] Entire project test suite (71 tests) passing cleanly on Windows environment.

### ✅ Implemented in Phase 07 (Prediction Engine)
- [x] Production-ready inference engine (`src/predict.py`) featuring `SpamPredictor` class and high-level `predict_email` / `predict_batch` APIs.
- [x] Cryptographic model verification ensuring SHA-256 integrity (`c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`) and pipeline architecture conformity before execution.
- [x] Exact Phase 02/04 text normalization consistency, Unicode NFKC decomposition, and punctuation/token preservation.
- [x] Calibrated probabilistic scoring with verified mathematical bounds $[0.0, 1.0]$ and sum integrity.
- [x] High-performance vectorized batch classification with strict index and ordering preservation.
- [x] Mathematical feature explainability extracting token-level log-odds contributions: $\text{TF-IDF}(w) \times [\log P(w|\text{Spam}) - \log P(w|\text{Ham})]$.
- [x] Defensive security: input length capping (1,000,000 chars), batch limiting (10,000 emails), and CSV formula injection neutralization.
- [x] Fully featured CLI entry point supporting single email, stdin streaming, and batch CSV processing with JSON/text formatting.
- [x] Comprehensive test suite with 21 dedicated unit and integration tests (`tests/test_predict.py`).
- [x] Entire project test suite (92 tests) passing cleanly with 0 failures.

### ✅ Implemented in Phase 08 (Complete Streamlit Frontend)
- [x] Professional multi-section cybersecurity operations dashboard (`app.py`) with persistent sidebar navigation across 5 operational views.
- [x] High-performance `@st.cache_resource` model caching with live SHA-256 cryptographic integrity verification badge (`c652119c99f0...`).
- [x] System Overview view integrating held-out test KPIs (**97.30% Accuracy**, **97.33% Precision**, **93.75% Recall**, **95.51% F1**, **0.9965 ROC-AUC**) and pipeline workflow.
- [x] Interactive Single Email Scanner featuring 4 realistic synthetic sample templates (Phishing, Prize Spam, Engineering Ham, Academic Ham), latency telemetry, and dynamic verdict banners.
- [x] Real-time interactive Plotly probability threat gauge ($0\%$ to $100\%$) with calibrated green/amber/red operational zones.
- [x] Mathematical token attribution explainability table displaying word-level log-odds contributions: $\text{TF-IDF}(w) \times [\log P(w|\text{Spam}) - \log P(w|\text{Ham})]$.
- [x] High-throughput Batch CSV Scanner with drag-and-drop upload, template CSV generator, data preview, vectorized `predict_batch()` processing, summary metrics, and filterable results table.
- [x] Secure sanitized batch CSV export neutralizing spreadsheet formula injection (`=, +, -, @`).
- [x] Model Performance Analytics tab rendering interactive Plotly confusion matrix heatmap, ROC and PR curves, baseline comparison chart, and validation vs. test generalization table.
- [x] Academic About & Methodology tab documenting course outcome mapping (CO2, CO4, CO5), architecture flowchart, Apache SpamAssassin dataset provenance, and historical limitations.
- [x] Dedicated UI helper module (`src/ui_helpers.py`) isolating report loading, sample emails, and Plotly visualization generation.
- [x] Comprehensive UI automated test suite with 19 dedicated unit and integration tests (`tests/test_app.py`) using `streamlit.testing.v1.AppTest`.
- [x] Entire project test suite (111 tests) passing cleanly on Windows environment.

### ✅ Implemented in Phase 09 (Testing, Security Hardening & Performance Verification)
- [x] Hardened CSV formula injection defense in `src/predict.py` and `app.py` against dangerous prefixes (`=, +, -, @, \t, \r, \n`) and leading whitespace evasion patterns.
- [x] Resolved Streamlit batch scanner state persistence defect, ensuring results table, metrics, and CSV downloads survive reactive filter changes.
- [x] Resolved single scanner quick-fill sample state defect, resetting previous results upon new sample selection.
- [x] Added mathematical formulas for False Positive Rate ($\text{FPR} = 1.13\%$) and False Negative Rate ($\text{FNR} = 6.25\%$) alongside dataset provenance disclosures in model analytics.
- [x] Automated resource and latency profiling on Windows 11 host (12th Gen Intel Core i7-12650H, 16 GB RAM): cold load 76.6 ms, warm inference 1.04 ms median, 10,478 emails/sec batch throughput, and 162.4 MB peak working set.
- [x] Exported empirical metrics to `reports/performance_benchmark.json` and authored exhaustive QA report (`reports/phase09_qa_report.md`).
- [x] Developed 12 dedicated automated regression tests in `tests/test_hardening.py` verifying formula injection, boundary cases, Unicode/adversarial text, and model immutability.
- [x] Verified complete automated test suite: **123 passed, 0 failed** across all 9 test suites.
- [x] Cryptographically confirmed model integrity: SHA-256 hash `c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0` strictly unchanged.

### ⏳ Planned Future Phases
- [ ] **Phase 10 (Final Academic Documentation & Release Delivery):** Complete project documentation, demonstration materials, academic report, and presentation preparation.

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

### 7. Run Model Training & Tuning Pipeline (Phase 05)
```bash
python -m src.train
```

### 8. Run Final Model Evaluation Pipeline (Phase 06)
```bash
python -m src.evaluate
```

### 9. Run Prediction Engine via CLI (Phase 07)
```bash
# Classify single email with feature contributions:
python -m src.predict --subject "Claim Prize" --text "Urgent claim your free cash prize bonus money" --explain

# Batch CSV classification:
python -m src.predict --batch-csv incoming.csv --output-csv results.csv
```

### 10. Run Test Suite
```bash
pytest -v
```

### 11. Launch Streamlit Application
```bash
streamlit run app.py
```
By default, the application runs on `http://localhost:8501`.
