# SpamGuard AI — System Architecture & Workflow Specifications

> **Academic AI Micro Project – 05** | **Domain:** Cybersecurity / Email Threat Filtering  
> **Course Outcomes:** CO2 (Algorithm Selection), CO4 (Model Engineering), CO5 (Evaluation Metrics)  
> **Status:** Phase 10 Final Release Architecture Specification

---

## 1. Overall System Architecture

The following diagram illustrates the multi-tier architectural layout of SpamGuard AI, separating the presentation layer, the modular inference engine, the persistence and artifact stores, and defensive security boundaries.

```mermaid
graph TD
    subgraph ClientLayer ["1. Client & Presentation Layer (Streamlit)"]
        UI_Dash["Overview Dashboard<br/>(System Telemetry & KPIs)"]
        UI_Single["Single Email Scanner<br/>(Live Threat Inspector)"]
        UI_Batch["Batch CSV Scanner<br/>(Vectorized Bulk Scanner)"]
        UI_Analytics["Model Performance Analytics<br/>(Confusion Matrix & Curves)"]
        UI_About["About & Methodology<br/>(CO Mapping & Disclosures)"]
    end

    subgraph HelperLayer ["2. UI Utility & Visualization Layer (src/ui_helpers.py)"]
        Cache["@st.cache_resource<br/>SpamPredictor Cache"]
        PlotlyEngine["Plotly & Matplotlib Engines<br/>(Gauge, CM Heatmap, Comparison)"]
        ReportLoader["Report Loader<br/>(test_evaluation.json, eda_summary.json)"]
        SampleLib["Synthetic Demonstration Library<br/>(Phishing, Prize, Engineering, Academic)"]
    end

    subgraph EngineLayer ["3. Core Inference & Security Engine (src/predict.py)"]
        Validator["Input Validator & Length Capper<br/>(MAX_INPUT_CHARS: 1MB)"]
        TextPrep["Text Combiner & Sanitizer<br/>(Unicode NFKC & Phase 02/04 Norm)"]
        Predictor["SpamPredictor Engine<br/>(Single & Vectorized Batch Predict)"]
        ExplainEngine["Log-Odds Feature Explainability<br/>(Token Attribution Calculator)"]
        Sanitizer["CSV Formula Injection Defense<br/>(Neutralizes =, +, -, @, \\t, \\r, \\n)"]
    end

    subgraph ModelLayer ["4. Machine Learning Pipeline (models/spam_pipeline.joblib)"]
        TFIDF["TF-IDF Vectorizer Step<br/>(Unigrams & Bigrams, 20,000 Features)"]
        MNB["Multinomial Naive Bayes Step<br/>(alpha = 0.1, Lidstone Smoothing)"]
        IntegrityCheck["SHA-256 Checksum Validator<br/>(c652119c99f0...)"]
    end

    subgraph DataStore ["5. Artifact & Report Store"]
        Splits["Quarantined Splits<br/>(train.csv, validation.csv, test.csv)"]
        Reports["Verified Audit Reports<br/>(test_evaluation.json, training_summary.json)"]
        Figures["Publication Figures<br/>(test_roc_curve.png, test_confusion_matrix.png)"]
    end

    %% Interactions
    UI_Dash --> ReportLoader
    UI_Single --> Predictor
    UI_Single --> PlotlyEngine
    UI_Single --> SampleLib
    UI_Batch --> Predictor
    UI_Batch --> Sanitizer
    UI_Analytics --> ReportLoader
    UI_Analytics --> PlotlyEngine
    UI_About --> Figures

    Cache --> Predictor
    ReportLoader --> Reports

    Predictor --> Validator
    Validator --> TextPrep
    TextPrep --> TFIDF
    TFIDF --> MNB
    MNB --> ExplainEngine
    Predictor --> IntegrityCheck
    IntegrityCheck --> ModelLayer
```

---

## 2. Dataset Preparation & Training Pipeline

This flowchart documents the complete Phase 02 to Phase 05 pipeline, illustrating how raw archives are converted into the frozen classifier without data leakage.

```mermaid
flowchart TD
    subgraph Ingestion ["Phase 02: Ingestion & Parsing (src/prepare_data.py)"]
        Raw["5 Apache SpamAssassin Archives<br/>(easy_ham, easy_ham_2, hard_ham, spam, spam_2)"]
        SafeExtract["In-Memory Tar Stream Reader<br/>(Path Traversal & Shell Script Rejection)"]
        MIMEDecode["MIME Header & Encoding Decoder<br/>(UTF-8, ISO-8859-1, Windows-1252)"]
        HTMLFallback["SafeHTMLTextExtractor<br/>(Non-executing Tag Stripper)"]
        Deduplicate["SHA-256 Deduplication & Conflict Filter<br/>(5,923 Valid Emails: 4,109 Ham, 1,814 Spam)"]
    end

    subgraph Splitting ["Phase 04: Stratified Partitioning (src/preprocess.py)"]
        Split["Stratified Random Split (Seed: 42)<br/>Zero ID & Content Leakage Enforced"]
        TrainData["train.csv (70% - 4,145 records)"]
        ValData["validation.csv (15% - 889 records)"]
        TestData["test.csv (15% - 889 records, QUARANTINED)"]
    end

    subgraph Training ["Phase 05: Model Training & Tuning (src/train.py)"]
        FitTFIDF["Fit TF-IDF Vectorizer strictly on train.csv<br/>(20,000 Features, Sublinear TF)"]
        GridSearch["Alpha Hyperparameter Optimization<br/>(Candidate alphas: 0.1, 0.5, 1.0, 2.0, 5.0)"]
        TuneOnVal["Evaluate on validation.csv<br/>(Selected: alpha = 0.1, Val F1 = 0.9644)"]
        Serialize["Joblib Serialization & Reload Verification<br/>(models/spam_pipeline.joblib)"]
    end

    Raw --> SafeExtract --> MIMEDecode --> HTMLFallback --> Deduplicate
    Deduplicate --> Split
    Split --> TrainData
    Split --> ValData
    Split --> TestData
    TrainData --> FitTFIDF
    FitTFIDF --> GridSearch
    ValData --> TuneOnVal
    GridSearch --> TuneOnVal
    TuneOnVal --> Serialize
```

---

## 3. Single Email Prediction Workflow

The single email workflow illustrates the life-cycle of a user request from text entry to token-level explainability.

```mermaid
sequenceDiagram
    autonumber
    actor User as Security Analyst
    participant UI as Streamlit UI (app.py)
    participant Engine as SpamPredictor (src/predict.py)
    participant Pipe as Scikit-Learn Pipeline
    participant Explainer as Log-Odds Calculator

    User->>UI: Selects Sample or Inputs Subject & Body
    User->>UI: Clicks 'Scan Email'
    UI->>Engine: predict_email(subject, body, explain=True)
    Engine->>Engine: prepare_input_text(): Validate bounds (len <= 1MB) & clean
    Engine->>Pipe: pipeline.predict([cleaned_text])
    Engine->>Pipe: pipeline.predict_proba([cleaned_text])
    Pipe-->>Engine: Class [0 or 1] and Probabilities [P(Ham), P(Spam)]
    Engine->>Engine: Assert finite & sum to 1.0
    Engine->>Explainer: explain_email(cleaned_text, top_k=10)
    Explainer->>Pipe: tfidf.transform([cleaned_text])
    Explainer->>Explainer: contribution(w) = TF-IDF(w) * [log P(w|Spam) - log P(w|Ham)]
    Explainer-->>Engine: Top-10 Token Contributions
    Engine-->>UI: PredictionResult Object
    UI->>UI: Render Verdict Banner (Green Ham vs. Red Threat)
    UI->>UI: Render Plotly Probability Gauge & Telemetry
    UI->>UI: Render Feature Contribution Table (Spam-leaning vs. Ham-leaning)
```

---

## 4. Batch Email Classification Workflow

The batch workflow handles high-throughput vectorized classification with spreadsheet formula injection defenses.

```mermaid
flowchart TD
    StartUpload["User Uploads CSV File (Drag & Drop)"] --> CheckSchema{"Schema Check:<br/>('subject', 'body') OR 'text'"}
    CheckSchema -- No --> ErrorSchema["Display Schema Validation Error"]
    CheckSchema -- Yes --> CheckBounds{"Row Count Check:<br/>len(df) <= 10,000"}
    CheckBounds -- Exceeded --> ErrorBounds["Display Batch Size Limit Error"]
    CheckBounds -- Valid --> ClickScan["User Clicks 'Scan Entire Batch'"]

    ClickScan --> Vectorize["Vectorized Text Preparation & Pipeline Inference<br/>(Predict & Predict_Proba on Entire Array)"]
    Vectorize --> CalcMetrics["Calculate Aggregate Batch Metrics<br/>(Total, Spam Count, Ham Count, Spam %)"]
    CalcMetrics --> StoreState["Persist Results in st.session_state<br/>(Preserves state across filter changes)"]

    StoreState --> RenderMetrics["Display Summary Metric Cards<br/>(Total Scanned, Spam Detected, Ham, Latency)"]
    RenderMetrics --> FilterRadio{"User Selects Filter:<br/>All / Spam Only / Ham Only"}
    FilterRadio --> RenderTable["Render Filtered Data Table<br/>(Predicted Verdict, Spam %, Ham %, Latency)"]
    RenderTable --> DownloadCSV["User Clicks 'Export Clean Results (CSV)'"]

    DownloadCSV --> SanitizeCells["Apply sanitize_csv_cell()<br/>Neutralize =, +, -, @, \\t, \\r, \\n with leading '"]
    SanitizeCells --> DeliverDownload["Deliver Sanitized CSV Stream to Browser"]
```

---

## 5. Model Evaluation Workflow (Phase 06)

This specification details the independent evaluation pipeline strictly assessing the frozen model on the held-out test partition.

```mermaid
flowchart LR
    subgraph InputQuarantine ["Held-Out Test Partition"]
        TestCSV["data/processed/splits/test.csv<br/>(n = 889: 617 Ham, 272 Spam)<br/>SHA-256: 587102a5..."]
    end

    subgraph IntegrityGate ["Pre-Evaluation Cryptographic Gate"]
        ModelJoblib["models/spam_pipeline.joblib<br/>SHA-256: c652119c..."]
        Gate{"Checksum Verification<br/>Actual == Expected?"}
    end

    subgraph EvaluationEngine ["Phase 06 Evaluation Engine (src/evaluate.py)"]
        Inference["Vectorized Inference on Test Set<br/>Zero refitting or parameter adjustment"]
        Confusion["Confusion Matrix Generation<br/>TN: 610, FP: 7, FN: 17, TP: 255"]
        Metrics["Compute Statistical Metrics<br/>Accuracy: 97.30%, Precision: 97.33%<br/>Recall: 93.75%, F1: 95.51%<br/>ROC-AUC: 0.9965, AP: 0.9926"]
        Baseline["Evaluate Majority-Class Baseline<br/>Accuracy: 69.40%, Precision: 0.00%<br/>Recall: 0.00%, F1: 0.00%"]
    end

    subgraph OutputReports ["Verified Persistent Outputs"]
        JSONReport["reports/test_evaluation.json"]
        MarkdownReport["reports/evaluation_findings.md"]
        CurvePlots["reports/figures/<br/>(test_roc_curve.png, test_precision_recall_curve.png)"]
    end

    ModelJoblib --> Gate
    Gate -- Checksum Mismatch --> Abort["Halt with Security Error"]
    Gate -- Authenticated --> Inference
    TestCSV --> Inference
    Inference --> Confusion
    Inference --> Metrics
    Inference --> Baseline
    Confusion --> JSONReport
    Metrics --> JSONReport
    Baseline --> JSONReport
    Metrics --> MarkdownReport
    Metrics --> CurvePlots
```
