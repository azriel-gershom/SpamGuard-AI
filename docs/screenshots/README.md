# SpamGuard AI — Application Visual Assets & Screenshot Capture Checklist

> **Academic AI Micro Project – 05** | **Phase 10 Release Documentation**  
> **Status:** Manual Verification & Evaluation Capture Checklist

---

## 1. Overview & Verification Notice

In compliance with academic evaluation standards, all visual assets in this directory must represent authentic application captures produced from the verified Streamlit dashboard. 

Because headless automated browser drivers (e.g. Playwright) were restricted from downloading external pre-compiled binaries in this environment, this checklist provides the exact, reproducible manual instructions for capturing evaluation screenshots using any standard browser (Chrome, Edge, Firefox).

---

## 2. Screenshot Capture Checklist

Launch the live application before capturing:
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Set browser resolution to **1920 × 1080 (100% zoom)** for optimal academic presentation clarity.

| File Name | Target Module | User Action / State | Required Visual Elements |
| :--- | :--- | :--- | :--- |
| **`01_overview_dashboard.png`** | Overview Dashboard (`🏠 Overview Dashboard`) | Default landing state upon application startup. | Model status badge (`Verified (c652119c...)`), 5 KPI metric cards (97.30% Accuracy, 97.33% Precision, 93.75% Recall, 95.51% F1, 0.9965 ROC-AUC), dataset breakdown. |
| **`02_single_scanner_initial.png`** | Single Email Scanner (`🔍 Email Scanner`) | Clean initial state before scanning. | Subject input field, body text area, quick-fill sample dropdown, 'Scan Email' and 'Clear/Reset' buttons. |
| **`03_single_scanner_spam_threat.png`** | Single Email Scanner (`🔍 Email Scanner`) | Select `🚨 Promotional Cash Prize Notification (Spam)` and click **Scan Email**. | Red `🚨 THREAT DETECTED: SPAM` banner (99.99% probability), Threat Probability Gauge dial pointing to red zone, Top Token Contributions table showing positive log-odds terms (`bonus`, `form below`). |
| **`04_single_scanner_ham_verified.png`** | Single Email Scanner (`🔍 Email Scanner`) | Select `✅ Team Engineering Standup Update (Ham)` and click **Scan Email**. | Green `✅ VERIFIED: HAM (LEGITIMATE)` banner (99.34% confidence), Threat Probability Gauge in safe green zone (<1%), explainability table showing negative log-odds terms (`sprint`, `deployment`). |
| **`05_batch_scanner_overview.png`** | Batch CSV Scanner (`📁 Batch CSV Scanner`) | Initial state with uploaded CSV template. | Expandable CSV guidelines, 'Download Sample CSV Template' button, drag-and-drop file uploader area, input preview table. |
| **`06_batch_scanner_results.png`** | Batch CSV Scanner (`📁 Batch CSV Scanner`) | Upload a batch and click **Scan Entire Batch**. | Batch Classification Summary metric cards (Total Scanned, Spam Detected, Ham Count, Latency), interactive filter radio ('Show All', 'Spam Only', 'Ham Only'), data table with probability columns, and 'Export Clean Results (CSV)' button. |
| **`07_analytics_confusion_matrix.png`** | Model Analytics (`📊 Model Analytics`) | Sub-tab: **🎯 Confusion Matrix**. | Top test KPI bar, interactive Plotly confusion matrix heatmap (TN: 610, TP: 255, FP: 7, FN: 17), mathematical error rate definitions ($\text{FPR}=1.13\%$, $\text{FNR}=6.25\%$), and historical provenance notice. |
| **`08_analytics_roc_pr_curves.png`** | Model Analytics (`📊 Model Analytics`) | Sub-tab: **📈 ROC & PR Curves**. | Side-by-side high-resolution plots of the Receiver Operating Characteristic curve ($\text{AUC} = 0.9965$) and Precision-Recall curve ($\text{AP} = 0.9926$). |
| **`09_analytics_baseline_comparison.png`** | Model Analytics (`📊 Model Analytics`) | Sub-tab: **⚖️ Baseline Comparison**. | Grouped Plotly bar chart comparing Naïve Bayes against the Majority-Class Baseline across Accuracy, Precision, Recall, and F1-score. |
| **`10_analytics_generalization.png`** | Model Analytics (`📊 Model Analytics`) | Sub-tab: **🔬 Generalization & Stability**. | Structured comparison table of Phase 05 validation metrics vs. Phase 06 test metrics with generalization delta column verifying stability ($<1.6\%$ change). |
| **`11_about_methodology.png`** | About & Methodology (`ℹ️ About & Methodology`) | Complete scroll view of About module. | Academic course outcome mapping (CO2, CO4, CO5), end-to-end pipeline architecture text/ASCII diagram, and historical dataset limitation disclosures. |

---

## 3. Storage & Presentation Integration

Save all captured PNG images in this directory (`docs/screenshots/`). They can be referenced in academic slide presentations, project viva portfolios, and institutional project documentation.
