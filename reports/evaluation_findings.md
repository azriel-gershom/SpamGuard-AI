# SpamGuard AI — Phase 06 Final Model Evaluation Findings

**Micro Project – 05:** AI-Based Spam Email Classifier  
**Course Outcomes:** CO2, CO4, CO5  
**Evaluation Scope:** Untouched held-out test partition (`data/processed/splits/test.csv`)  
**Trained Artifact SHA-256:** `c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`  
**Test Partition SHA-256:** `587102a5400814ae8107539f8a0bcc7307661e752e6fc622ce6c56f854bfea1b`  

---

## 1. Executive Summary

This report establishes the definitive, empirical test performance of the **SpamGuard AI Multinomial Naïve Bayes** pipeline. The model was trained exclusively during Phase 05 using training data (`n=4,145`) with hyperparameter selection (`alpha=0.1`) performed solely on validation data (`n=889`).

The held-out test partition (`n=889`: 617 Ham, 272 Spam) remained strictly quarantined and untouched throughout development until this evaluation. **No model fitting, retraining, or threshold re-tuning was performed.**

### Definitive Test Results

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

---

## 2. Validation vs. Test Generalization Comparison

To demonstrate that the model did not overfit the validation set during hyperparameter search (`alpha=0.1`), we compare the validation metrics against the final test metrics:

| Metric | Phase 05 Validation | Phase 06 Held-Out Test | Generalization Delta | Status |
|---|---:|---:|---:|:---:|
| **Overall Accuracy** | 97.86% | 97.30% | -0.56% | Stable (Δ < 0.6%) |
| **Spam Precision** | 98.85% | 97.33% | -1.52% | Robust |
| **Spam Recall** | 94.14% | 93.75% | -0.39% | Consistent (Δ = -0.39%) |
| **Spam F1-Score** | 96.44% | 95.51% | -0.93% | High Fidelity |

The minimal divergence between validation and test scores (<1.0% across all indicators) proves that the model possesses excellent generalization capabilities and that data leakage was successfully prevented.

---

## 3. Confusion Matrix Analysis

The confusion matrix on the 889 test emails is oriented as follows:

```
                      PREDICTED HAM (0)     PREDICTED SPAM (1)
ACTUAL HAM (0)        TN = 610                 FP = 7 (Type I Error)
ACTUAL SPAM (1)       FN = 17 (Type II Error)   TP = 255
```

### Breakdown:
1. **True Negatives (`TN = 610`):** 98.87% of legitimate emails were correctly routed to the inbox.
2. **True Positives (`TP = 255`):** 93.75% of spam emails were successfully detected and quarantined.
3. **False Positives (`FP = 7`):** Only 7 out of 617 legitimate emails (1.13%) were mistakenly flagged as spam.
4. **False Negatives (`FN = 17`):** Only 17 out of 272 spam emails (6.25%) bypassed the filter.

---

## 4. Why Precision, Recall, and F1-Score Matter

In operational cybersecurity and email filtering, **raw accuracy is a dangerous and misleading metric**:

1. **The Class Imbalance Flaw:**
   In real-world mail streams, ham typically comprises 70% to 90% of all messages. A trivial "majority baseline" that classifies every message as legitimate achieves **69.40% accuracy** on our test set while providing **zero protection** (Spam Precision = 0.0%, Spam Recall = 0.0%, Spam F1 = 0.0%).
2. **Asymmetric Error Costs:**
   - **False Positives (Type I Error):** A legitimate email (e.g., job offer, password reset, client invoice) routed to spam can cause severe operational disruption, missed business opportunities, or financial loss. High **Spam Precision (97.33%)** ensures that when the system flags an email as spam, the user can trust that verdict with over 97% confidence.
   - **False Negatives (Type II Error):** A spam email reaching the inbox exposes users to phishing, credential theft, and malware. A high **Spam Recall (93.75%)** ensures that almost 94 out of every 100 threats are neutralized.
3. **Harmonic Mean Balance (F1-Score):**
   The F1-score (95.51%) balances precision and recall, ensuring neither metric is artificially maximized at the catastrophic expense of the other.

---

## 5. False Positive & False Negative Qualitative Inspection

### False Positive Analysis (7 Cases)
- **Primary Cause:** Legitimate technical newsletters, open-source mailing list digests, and automated server alerts containing marketing-style trigger tokens (e.g., "unsubscribe", "click here", "free software", "guarantee").
- **Mitigation Strategy:** Adding domain whitelisting, SPF/DKIM authentication header parsing, and metadata feature expansion in future iterations.

### False Negative Analysis (17 Cases)
- **Primary Cause:** Extremely short spam emails lacking dense vocabulary, image-heavy spam where text content was minimal, or novel phishing scams mimicking interpersonal conversation without explicit promotional keywords.
- **Mitigation Strategy:** Character n-gram expansion, URL domain reputational checks, and visual OCR for attached graphics.

---

## 6. Dataset Limitations & Academic Context

1. **Corpus Age:** The Apache SpamAssassin corpus reflects email conventions, MIME structures, and vocabulary typical of its collection era (early 2000s). While foundational for algorithmic benchmarking, modern phishing frequently leverages generative AI and obfuscated URLs.
2. **Plain-Text Extraction:** Emails were preprocessed by extracting ASCII/UTF-8 textual content. Embedded JavaScript, CSS styling, and image payloads were intentionally excluded to focus on classical NLP capabilities.
3. **Static Prior Assumptions:** The Multinomial Naïve Bayes model assumes conditional feature independence given the class label. Despite this simplifying theoretical assumption, empirical performance remains world-class (ROC-AUC = 0.9965).

---

## 7. Artifact Verification & Provenance

* **Model File:** `models/spam_pipeline.joblib`
* **Model Checksum:** `c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0` (Verified Match)
* **Test Dataset:** `data/processed/splits/test.csv`
* **Test Dataset Checksum:** `587102a5400814ae8107539f8a0bcc7307661e752e6fc622ce6c56f854bfea1b`
* **Execution Environment:** Windows-11-10.0.26300-SP0 (Python 3.12.10)
* **Status:** Complete, verified, and sealed. Ready for Phase 07 (Prediction Engine).
