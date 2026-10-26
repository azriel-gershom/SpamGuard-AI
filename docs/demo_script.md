# SpamGuard AI — Live Project Demonstration Script

> **Academic AI Micro Project – 05** | **Course Competencies:** CO2, CO4, CO5  
> **Target Presentation Time:** 7–10 Minutes | **Presenter Role:** AI / ML Systems Engineer

---

## 1. Introduction & Problem Statement (1 Minute)

### Spoken Introduction
> *"Good morning, respected examiners and faculty members. Today, we are demonstrating **SpamGuard AI — AI-Based Spam Email Classifier**, an academic microproject addressing digital communication security and email threat filtering.*
>
> *Traditional rule-based filters rely on static blocklists and keyword heuristics that rapidly degrade when spammers alter vocabulary or obfuscate terms. Furthermore, imbalanced email data poses a critical design challenge: an email filter cannot simply maximize naive accuracy—it must protect critical legitimate correspondence from false quarantine while intercepting malicious campaigns.*
>
> *SpamGuard AI solves this by engineering a probabilistically calibrated NLP classifier combining **TF-IDF sublinear n-gram feature extraction** with **Multinomial Naïve Bayes** using optimal Lidstone smoothing ($\alpha = 0.1$)."*

---

## 2. System Architecture & Live Overview Dashboard (1.5 Minutes)

### Action in Application
1. Open the browser to `http://localhost:8501`.
2. Ensure the active view is set to **`🏠 Overview Dashboard`**.

### Spoken Narration
> *"As shown on the Overview Dashboard, our application features continuous cryptographic integrity tracking. The sidebar displays a live green badge verifying that the active classifier artifact matches our pre-computed SHA-256 checksum (`c652119c...`). This guarantees that model parameters remain completely frozen without runtime re-fitting.*
>
> *On the main panel, you see the official performance metrics evaluated on our **strictly held-out test partition of 889 authentic emails** from the Apache SpamAssassin corpus:
> - **Test Accuracy:** 97.30%
> - **Spam Precision:** 97.33% (protecting users from false accusations)
> - **Spam Recall:** 93.75% (catching over 93% of all attack messages)
> - **Spam F1-Score:** 95.51%
> - **ROC-AUC:** 0.9965, demonstrating near-ideal class separability.*
>
> *Notice also the dataset breakdown: 4,145 training emails, 889 validation emails used exclusively for hyperparameter tuning, and 889 test emails that were quarantined until final evaluation."*

---

## 3. Live Demonstration: Single Email Scanner (3 Minutes)

### Action in Application
1. Click **`🔍 Email Scanner`** in the sidebar.
2. Under **Quick-Fill Demonstration Sample**, select:  
   `🚨 Promotional Cash Prize Notification (Spam)`
3. Point out that the Subject and Body are populated immediately.
4. Click the primary button: **`🔍 Scan Email`**.

### Spoken Narration & Results Explanation
> *"First, let us examine an aggressive commercial spam lure promising cash prizes. When we click 'Scan Email', the inference engine normalizes the text and vectorizes it across 20,000 unigram and bigram features.*
>
> *The model outputs a prominent red alert banner: **🚨 THREAT DETECTED: SPAM**, with an estimated spam probability of **99.99%**.*
>
> *Below the verdict banner, look at the **Threat Probability Gauge**: the needle points into the deep red zone ($> 85\%$).*
>
> *More importantly, for cyber analysts, SpamGuard AI provides transparent mathematical explainability. The **Top Token Contributions** table lists the statistical log-odds difference for each term:
> $$\text{Contribution}(w) = \text{TF-IDF}(w) \times [\log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham})]$$
> Here, bigrams like `form below`, `free bonus`, and tokens like `bonus` exhibit massive positive log-odds, strongly driving the classifier towards the Spam classification.*
>
> *Now, let us test a legitimate message."*

### Action in Application
5. Change the sample selector to:  
   `✅ Team Engineering Standup Update (Ham)`
6. Click **`🔍 Scan Email`**.

### Spoken Narration & Results Explanation
> *"Notice the immediate contrast. The banner turns green: **✅ VERIFIED: HAM (LEGITIMATE)**, with a confidence of **99.34%** (estimated spam probability of only **0.66%**).*
>
> *Looking at the explainability table, engineering vocabulary such as `yesterday`, `authentication`, `deployment`, and `sprint` all exhibit negative log-odds, correctly pulling the classification toward legitimate ham.*
>
> *Finally, clicking **'🔄 Clear / Reset'** wipes all session state and input fields clean."*

---

## 4. Live Demonstration: Vectorized Batch CSV Scanner (2 Minutes)

### Action in Application
1. Navigate to **`📁 Batch CSV Scanner`**.
2. Click **`ℹ️ CSV Format Guidelines & Sample Template`** and briefly point out the **`📥 Download Sample CSV Template`** button.
3. In the file uploader, upload the template file (or a test batch).
4. Point out the **Input Data Preview** showing the first 5 records and column schema.
5. Click **`🚀 Scan Entire Batch`**.

### Spoken Narration
> *"In real-world enterprise environments, security gateways process millions of emails daily. SpamGuard AI includes a high-throughput vectorized batch scanner.*
>
> *When we execute 'Scan Entire Batch', the engine vectorizes all rows simultaneously. Notice the execution time: over **10,000 emails per second** on standard CPU hardware.*
>
> *The dashboard presents aggregate statistics: Total Scanned, Spam Detected, and Ham Count.*
>
> *We can interactively filter the results table using the radio toggle: clicking **'Spam Only (Threats)'** immediately filters the table to show only quarantined records. Notice that in our hardened Phase 09 release, Streamlit session state preserves the results table without disappearing upon filter changes.*
>
> *Finally, clicking **'📥 Export Clean Results (CSV)'** exports the sanitized data. To guard against spreadsheet formula injection attacks, our sanitizer neutralizes dangerous prefixes (`=, +, -, @`) by prefixing them with a single quote."*

---

## 5. Model Analytics, Confusion Matrix & Academic Evaluation (2 Minutes)

### Action in Application
1. Navigate to **`📊 Model Analytics`**.
2. Under the **Confusion Matrix** sub-tab, show the interactive Plotly heatmap.
3. Switch to the **ROC & PR Curves** sub-tab.
4. Switch to the **Baseline Comparison** sub-tab.

### Spoken Narration
> *"Now let us review the rigorous academic evaluation (CO5). In our confusion matrix:
> - **True Negatives:** 610 legitimate emails were correctly delivered.
> - **True Positives:** 255 spam messages were successfully caught.
> - **False Positives:** Only 7 legitimate emails were misflagged, establishing an exceptionally low **False Positive Rate of 1.13%** ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{7}{617}$). In cybersecurity, minimizing Type I error is essential so valid business communication is never discarded.
> - **False Negatives:** Only 17 spam messages bypassed filtering, giving a **False Negative Rate of 6.25%** ($\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = \frac{17}{272}$).
>
> *Under the Baseline Comparison tab, we show why accuracy alone is a flawed metric on imbalanced data. A naive majority-class classifier achieves 69.40% accuracy simply by predicting every email as legitimate—yet it catches exactly 0% of spam.*
>
> *Our Multinomial Naïve Bayes model elevates Spam F1-score from 0.00% to **95.51%**, with an ROC-AUC of **0.9965** and Average Precision of **0.9926**."*

---

## 6. Limitations, Ethical Disclosures & Conclusion (1 Minute)

### Spoken Conclusion
> *"In conclusion, SpamGuard AI demonstrates a complete, production-grade AI system adhering to academic rigor and engineering best practices:
> - **123 automated tests** passing with 100% pass rate.
> - Complete leakage prevention between training, validation, and test sets.
> - Zero external API dependencies, ensuring strict user privacy.
>
> *We also openly disclose historical dataset limitations: because the Apache SpamAssassin corpus dates from 2002–2005, modern adversarial tactics such as QR-code phishing or Unicode homoglyph spoofing require continuous model retraining with modern corpora.*
>
> *Thank you. We now welcome questions from the examination committee."*
