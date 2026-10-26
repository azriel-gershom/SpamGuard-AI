# SpamGuard AI — Operational User Guide & Deployment Manual

> **Academic AI Micro Project – 05** | **Domain:** Cybersecurity / Email Threat Filtering  
> **Target Audience:** System Evaluators, Academic Examiners, and Security Analysts  
> **Version:** 1.0.0 (Phase 10 Final Release)

---

## 1. System Requirements

### Hardware Requirements
- **CPU:** Dual-core 2.0 GHz or higher (Tested on 12th Gen Intel Core i7-12650H, 10 Cores / 16 Threads).
- **RAM:** Minimum 4 GB required; 8 GB to 16 GB recommended for high-volume batch processing.
- **Disk Space:** Minimum 500 MB free space for repository, virtual environment, and model artifacts.

### Software Prerequisites
- **Operating System:** Microsoft Windows 10/11, macOS (12+), or Ubuntu Linux (22.04+).
- **Python Runtime:** Python **3.12.x** (64-bit).
- **Web Browser:** Any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, Apple Safari) with JavaScript enabled.

---

## 2. Environment Setup & Dependency Installation

### Step 1: Clone or Navigate to the Repository
Open PowerShell or your preferred terminal:
```powershell
cd "E:\Sem 3\AI\mp\SpamGuard-AI-Handoff\AI_mp"
```

### Step 2: Initialize Virtual Environment
Create and activate an isolated Python 3.12 virtual environment:
```powershell
# Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux:
# python3 -m venv .venv
# source .venv/bin/activate
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Pinned Dependencies
Install all required libraries specified in [`requirements.txt`](file:///e:/Sem%203/AI/mp/SpamGuard-AI-Handoff/AI_mp/requirements.txt):
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Artifact Integrity (Git-Ignored Assets)
Due to standard version control practices, binary models and raw datasets are git-ignored. Verify that the trained model pipeline exists locally:
```powershell
Get-FileHash models\spam_pipeline.joblib -Algorithm SHA256
```
**Expected Checksum:**  
`c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`

*(If the model file is not present, generate it deterministically using: `python -m src.prepare_data --download`, `python -m src.preprocess --split`, and `python -m src.train`)*

---

## 3. Launching the Streamlit Application

Execute the following command from the workspace root:
```powershell
streamlit run app.py
```
Upon execution, your default browser will open automatically to:
```text
http://localhost:8501
```

---

## 4. Navigating the Dashboard Sections

The application features persistent sidebar navigation across 5 operational modules:

### 4.1 System Overview (`🏠 Overview Dashboard`)
- **Model Status Indicator:** Real-time badge confirming pipeline authentication and SHA-256 integrity.
- **Held-Out Test Set KPIs:** Displays the official academic evaluation metrics evaluated on $n=889$ emails (Accuracy: **97.30%**, Precision: **97.33%**, Recall: **93.75%**, F1-Score: **95.51%**, ROC-AUC: **0.9965**).
- **System Boundaries:** Summary of architectural security limits (1MB text limit, 10,000 batch rows).

### 4.2 Single Email Scanner (`🔍 Email Scanner`)
- **Quick-Fill Synthetic Samples:** Use the dropdown selector to immediately load pre-configured realistic test cases:
  1. *Phishing Alert:* Bank account verification lure with urgent threat keywords.
  2. *Prize Spam:* Lottery winnings notification promising cash prizes.
  3. *Engineering Ham:* Software pull request discussion and code review comments.
  4. *Academic Ham:* Course microproject deadline reminder.
- **Custom Input:** Paste any custom email subject and message body into the provided input areas.
- **Scanning:** Click **`🔍 Scan Email`** to initiate inference.
- **Interpreting Results:**
  - **Verdict Banner:** Displays `🚨 THREAT DETECTED: SPAM` (red) or `✅ VERIFIED: HAM (LEGITIMATE)` (green).
  - **Threat Probability Gauge:** Interactive Plotly dial displaying calculated probability of spam [0% to 100%].
  - **Explainability Breakdown:** Ranked table of influential vocabulary terms showing whether each token pushes towards Spam or Ham based on its statistical log-odds difference.
- **Reset:** Click **`🔄 Clear / Reset`** to wipe inputs and restore blank state.

### 4.3 Batch CSV Scanner (`📁 Batch CSV Scanner`)
- **Sample Template:** Click **`📥 Download Sample CSV Template`** to download a verified CSV format with headers `subject,body`.
- **Uploading File:** Drag and drop or browse for a `.csv` file (maximum 10,000 rows).
- **Preview:** Verify the first 5 records in the uploaded preview table.
- **Execution:** Click **`🚀 Scan Entire Batch`** to run vectorized inference.
- **Summary Metrics:** View total emails processed, spam count, ham count, spam percentage, and total execution time.
- **Filtering Table:** Use the radio toggle to inspect:
  - *Show All* (Complete results).
  - *Spam Only (Threats)* (Quarantined emails).
  - *Ham Only (Legitimate)* (Passed emails).
- **Safe CSV Export:** Click **`📥 Export Clean Results (CSV)`** to download results. All cells starting with formula triggers (`=, +, -, @, \t, \r, \n`) are automatically neutralized to prevent spreadsheet formula injection.

### 4.4 Model Performance Analytics (`📊 Model Analytics`)
- **Interactive Confusion Matrix:** Visualizes the $2 \times 2$ matrix for the held-out test partition ($610$ True Negatives, $255$ True Positives, $7$ False Positives, $17$ False Negatives).
- **ROC & Precision-Recall Curves:** View high-resolution curves verifying strong discrimination ($\text{AUC} = 0.9965$, $\text{AP} = 0.9926$).
- **Baseline Comparison Chart:** Compares the trained Naïve Bayes model against a zero-intelligence majority-class classifier, highlighting why accuracy alone fails on imbalanced datasets.
- **Generalization Table:** Compares Phase 05 validation performance against Phase 06 test performance, proving absence of overfitting ($< 0.6\%$ delta).

### 4.5 About & Methodology (`ℹ️ About & Methodology`)
- Direct academic mapping to Course Competency Outcomes (CO2, CO4, CO5).
- Flowchart diagram of text preprocessing, TF-IDF vectorization, and inference.
- Academic provenance and ethical limitations of historical training corpora.

---

## 5. Command-Line Interface (CLI) Usage

For headless automation or scripting, the prediction engine supports direct CLI execution:

```powershell
# Classify single email with explainability:
python -m src.predict --subject "Urgent Prize" --text "Claim your free cash money bonus now" --explain

# Output structured JSON:
python -m src.predict --subject "Meeting" --text "Standup at 10 AM in conference room" --format json

# Secure classification via standard input (avoids shell history leaks):
echo "Confidential message body" | python -m src.predict --stdin

# Batch CSV classification:
python -m src.predict --batch-csv incoming.csv --output-csv results.csv
```

---

## 6. Running the Automated Test Suite

Verify complete test coverage across all 9 test suites:
```powershell
pytest -v
```
Expected output: **123 passed in ~11 seconds (0 failures)**.

---

## 7. Troubleshooting & FAQ

| Problem | Potential Cause | Verified Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: models/spam_pipeline.joblib` | Model artifact not found locally. | Verify that `models/spam_pipeline.joblib` is present and matches SHA-256 `c652119c99f0...`. |
| `SpamGuardModelError: Checksum mismatch` | Model file was modified or corrupted. | Restore the original model artifact from backup or regenerate via `python -m src.train`. |
| `SpamGuardBatchError: Schema Validation Error` | Uploaded CSV missing required columns. | Ensure your CSV includes either `subject` and `body` columns, or a single `text` column. |
| `SpamGuardValidationError: Email content is empty` | Input contains only spaces or null characters. | Enter at least one alphanumeric token in the Subject or Body field. |
| Streamlit port conflict (`Port 8501 is in use`) | Another instance is running. | Run `streamlit run app.py --server.port 8502` to use an alternate port. |

---

## 8. Privacy & Operational Security Notice

- **No Remote Telemetry:** All inference executes locally on CPU. No emails are transmitted to external servers, cloud APIs, or third-party AI services.
- **No Data Persistence:** Single email text and uploaded CSVs are processed in-memory and are never saved to disk or logged.
- **Academic Limitation Disclosure:** SpamGuard AI was trained on the Apache SpamAssassin corpus (2002–2005). While highly accurate on lexical spam indicators (97.30% test accuracy), a "Ham" verdict does not provide a universal guarantee against modern zero-day attacks, QR-code phishing (quishing), or advanced social engineering.
