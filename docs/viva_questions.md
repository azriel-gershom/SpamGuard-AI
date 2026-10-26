# SpamGuard AI — Academic Viva Voce Preparation Guide

> **Academic AI Micro Project – 05** | **Course Competencies:** CO2, CO4, CO5  
> **Topic:** Comprehensive Examiner Questions & Model Technical Answers  
> **Target Audience:** Microproject Viva Candidates and Evaluators

---

### Q1: What is the primary objective of SpamGuard AI?
**Answer:**  
SpamGuard AI is an intelligent text classification application designed to automatically categorize incoming electronic mail into **Ham (Legitimate, class 0)** and **Spam (Malicious/Unsolicited, class 1)**. It addresses the asymmetric cost of email classification errors by optimizing for high precision and balanced recall using TF-IDF feature extraction and Multinomial Naïve Bayes.

---

### Q2: State Bayes' Theorem and explain how it applies to SpamGuard AI.
**Answer:**  
Bayes' Theorem calculates the posterior probability of a hypothesis (class $c \in \{\text{Ham}, \text{Spam}\}$) given observed feature evidence $\mathbf{x} = (x_1, x_2, \dots, x_n)$:
$$P(c \mid \mathbf{x}) = \frac{P(c) \cdot P(\mathbf{x} \mid c)}{P(\mathbf{x})}$$
In SpamGuard AI, $P(c)$ represents the class prior probability learned from training split frequencies, and $P(\mathbf{x} \mid c)$ represents the class-conditional probability of observing the specific email vocabulary given that the email is Spam or Ham.

---

### Q3: What is the "Naïve" assumption in Naïve Bayes, and why does it work so well in text classification?
**Answer:**  
The naïve assumption posits that all features $x_i$ are conditionally independent given the class label $c$:
$$P(\mathbf{x} \mid c) = \prod_{i=1}^{n} P(x_i \mid c)$$
Although natural language violates this assumption in practice (words correlate grammatically and semantically), empirical and theoretical research demonstrates that the decision boundary (the rank ordering of class probabilities) remains optimal even when probability calibration exhibits slight bias. In text classification with high-dimensional sparse representations, it prevents catastrophic overfitting and executes in linear time $O(N)$.

---

### Q4: Why did you select Multinomial Naïve Bayes over Gaussian or Bernoulli Naïve Bayes (CO2)?
**Answer:**  
- **Multinomial NB:** Models discrete word counts and continuous frequency weights (such as TF-IDF), taking word frequency and magnitude into account.
- **Bernoulli NB:** Considers only binary word presence/absence (0 or 1), discarding word frequency information.
- **Gaussian NB:** Assumes features follow a continuous normal bell-curve distribution, which is mathematically inappropriate for sparse, non-negative, heavily zero-inflated text features.  
Therefore, Multinomial Naïve Bayes is the mathematically optimal variant for TF-IDF feature spaces.

---

### Q5: What is TF-IDF and what are its two components?
**Answer:**  
TF-IDF (Term Frequency-Inverse Document Frequency) measures the statistical importance of a token in a document relative to an entire corpus:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
1. **Term Frequency (TF):** Measures how frequently token $t$ appears in document $d$. In our pipeline, we use sublinear TF scaling ($1 + \log(\text{tf})$) to dampen the influence of spammers repeating keywords.
2. **Inverse Document Frequency (IDF):** Measures how rare the token is across all $N$ documents in the training corpus:
   $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   Common words (e.g., "the", "and") receive low IDF weights, while discriminative terms receive high weights.

---

### Q6: What text preprocessing steps were applied in your pipeline?
**Answer:**  
Implemented in `src/preprocess.py`:
1. Unicode NFKC normalization to resolve composite characters and homoglyph variations.
2. Line-break standardization (`\r\n` $\rightarrow$ `\n`) and null-byte elimination.
3. Whitespace trimming while preserving critical spam indicators: currency symbols (`$`, `€`), punctuation marks (`!`, `?`), URLs, and numeric strings.
4. Word unigram and bigram extraction (`ngram_range=(1, 2)`) with a 20,000 maximum feature cap.

---

### Q7: What is class imbalance and what was the imbalance ratio in your dataset?
**Answer:**  
Class imbalance occurs when one class significantly outnumbers another in the training distribution. In SpamGuard AI's cleaned dataset of 5,923 emails:
- **Ham (0):** 4,109 emails (69.37%)
- **Spam (1):** 1,814 emails (30.63%)  
This establishes an imbalance ratio of **2.27 : 1**.

---

### Q8: Why is Accuracy an insufficient metric for evaluating an imbalanced spam filter (CO5)?
**Answer:**  
On imbalanced data, a naive majority-class classifier that blindly predicts "Ham" for every single email would achieve **69.37% Accuracy** while catching **0% of spam** (Precision = 0%, Recall = 0%, F1 = 0.00). In cybersecurity, an organization cannot tolerate an email gateway that allows 100% of attacks into users' inboxes simply to report 70% accuracy.

---

### Q9: Define Precision, Recall, and F1-score in the context of email spam detection.
**Answer:**  
- **Precision:** Out of all emails the model flagged as Spam, what fraction were genuinely Spam?
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  High precision prevents legitimate emails from being incorrectly quarantined.
- **Recall (Sensitivity):** Out of all actual Spam emails that arrived, what fraction did the model successfully intercept?
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  High recall ensures attacks do not bypass the filter.
- **F1-Score:** The harmonic mean balancing precision and recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### Q10: What were your final performance metrics on the held-out test dataset?
**Answer:**  
Evaluated on $n=889$ quarantined test emails (617 Ham, 272 Spam):
- **Accuracy:** 97.30%
- **Spam Precision:** 97.33% (255 TP vs. 7 FP)
- **Spam Recall:** 93.75% (255 TP vs. 17 FN)
- **Spam F1-Score:** 95.51%
- **Specificity (Ham Recall):** 98.87% (610 TN vs. 7 FP)
- **ROC-AUC:** 0.9965
- **Average Precision (PR-AUC):** 0.9926

---

### Q11: Explain your confusion matrix breakdown and the corresponding error rates.
**Answer:**  
On the 889 test emails:
- **True Negatives (`TN = 610`):** Legitimate emails correctly allowed to inbox.
- **True Positives (`TP = 255`):** Malicious spam messages caught.
- **False Positives (`FP = 7`):** Legitimate emails mistakenly flagged (Type I error).
- **False Negatives (`FN = 17`):** Spam emails that slipped through (Type II error).
- **False Positive Rate (FPR):** $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{7}{7 + 610} = \mathbf{1.13\%}$.
- **False Negative Rate (FNR):** $\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = \frac{17}{17 + 255} = \mathbf{6.25\%}$.

---

### Q12: In email security, which error is more dangerous: Type I (False Positive) or Type II (False Negative)?
**Answer:**  
**Type I (False Positive) is far more detrimental in email security.** If a spam filter misclassifies a critical client contract, medical report, or university admission offer as spam and quarantines it, the user may suffer severe financial, legal, or professional harm without knowing. Conversely, a False Negative merely requires the user to manually mark an unsolicited message as junk. Therefore, our model was tuned for high precision (97.33%) and low FPR (1.13%).

---

### Q13: What is additive smoothing, and why is it necessary in Naïve Bayes?
**Answer:**  
If a word $w$ never appeared in a class during training, the maximum likelihood estimate $P(w \mid c)$ would be zero. Because Naïve Bayes multiplies feature probabilities together, a single zero probability would force the entire posterior probability to zero, blinding the model to all other strong evidence:
$$P(x_i \mid c) = \frac{N_{c, i} + \alpha}{N_c + \alpha |V|}$$
Additive smoothing adds pseudo-counts ($\alpha$) to prevent the zero-probability problem.

---

### Q14: Differentiate Laplace smoothing from Lidstone smoothing. What was your selected $\alpha$?
**Answer:**  
- **Laplace Smoothing:** Sets $\alpha = 1.0$. While standard, $\alpha = 1.0$ can over-smooth probabilities on large vocabularies ($|V| = 20,000$), diminishing the discriminative power of rare but highly predictive tokens.
- **Lidstone Smoothing:** Uses fractional smoothing values ($0 < \alpha < 1.0$).
- **Our Selection:** Through hyperparameter tuning across candidate alphas $\{0.1, 0.5, 1.0, 2.0, 5.0\}$ on the validation set, **$\alpha = 0.1$** achieved the highest Spam F1-Score of **0.9644**, outperforming $\alpha = 1.0$ (0.9385 F1).

---

### Q15: How did you partition your dataset, and what were the exact sizes?
**Answer:**  
Implemented in `src/preprocess.py` using stratified random sampling with fixed seed 42:
1. **Training Partition (70%):** 4,145 emails (2,876 Ham, 1,269 Spam).
2. **Validation Partition (15%):** 889 emails (616 Ham, 273 Spam).
3. **Test Partition (15%):** 889 emails (617 Ham, 272 Spam).

---

### Q16: What is data leakage, and what explicit architectural controls prevented it in your project (CO4)?
**Answer:**  
Data leakage occurs when information from outside the training partition influences model fitting or feature engineering, inflating performance estimates. We prevented it through four strict controls:
1. **Stratified Content Disjointness:** Verified zero shared message IDs and zero identical content between partitions.
2. **Train-Only Vectorizer Fitting:** The Scikit-learn `TfidfVectorizer` was fitted strictly on `train.csv`. The vocabulary, document frequencies, and IDF weights were learned solely from training data.
3. **Validation-Only Tuning:** Hyperparameter $\alpha$ was tuned solely on `validation.csv`.
4. **Quarantined Test Evaluation:** `test.csv` was never accessed, inspected, or transformed until the frozen artifact was evaluated in Phase 06.

---

### Q17: What does ROC-AUC measure, and what does a score of 0.9965 indicate?
**Answer:**  
The Receiver Operating Characteristic Area Under the Curve (ROC-AUC) evaluates the classifier's ability to rank a randomly chosen positive (Spam) instance higher than a randomly chosen negative (Ham) instance across all possible classification thresholds from $0.0$ to $1.0$. A score of **0.9965** indicates near-perfect discrimination and confirms that the model's ranking ability is robust regardless of the specific decision threshold applied.

---

### Q18: What is Average Precision (PR-AUC) and why is it valuable for imbalanced classes?
**Answer:**  
Average Precision calculates the area under the Precision-Recall curve. Unlike ROC curves—which can look overly optimistic when the negative class is large because the False Positive Rate denominator ($FP + TN$) is dominated by $TN$—the Precision-Recall curve focuses strictly on the minority (positive) class. Our model achieved an Average Precision of **0.9926**, confirming exceptional precision across virtually all recall levels.

---

### Q19: How did you verify that your model did not overfit during hyperparameter tuning?
**Answer:**  
By comparing Phase 05 validation performance against Phase 06 held-out test performance:
- Accuracy: Validation 97.86% vs. Test 97.30% ($\Delta = -0.56\%$)
- Spam Precision: Validation 98.85% vs. Test 97.33% ($\Delta = -1.52\%$)
- Spam Recall: Validation 94.14% vs. Test 93.75% ($\Delta = -0.39\%$)
- Spam F1-Score: Validation 96.44% vs. Test 95.51% ($\Delta = -0.93\%$)  
All generalization deltas are below $1.6\%$, proving that the model generalizes robustly to unseen data without variance degradation.

---

### Q20: How does SpamGuard AI explain its predictions to security analysts?
**Answer:**  
Implemented in `SpamPredictor.explain_email()`: For any email, the engine transforms the text with TF-IDF and computes the token-level log-odds contribution for each active token $w$:
$$\text{Contribution}(w) = \text{TF-IDF}(w) \times \left[ \log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham}) \right]$$
If contribution $> 0$, the word increases the odds of Spam (e.g., `bonus`, `form below`, `guarantee`); if $< 0$, it supports Ham (e.g., `sprint`, `authentication`, `deployment`). The top-10 tokens are displayed in an interactive table with absolute influence rankings.

---

### Q21: How is the trained model serialized and verified against tampering?
**Answer:**  
The complete Scikit-learn Pipeline (combining the fitted TF-IDF vectorizer and fitted MultinomialNB classifier) is serialized using Joblib to `models/spam_pipeline.joblib`. Before every inference or Streamlit launch, `compute_file_sha256()` computes the block-wise SHA-256 hash and asserts it matches:  
`c652119c99f0db2abe0788969d002818478bb63be95e31e5f337eda8da6ddbb0`. If a mismatch occurs, loading is halted with a `SpamGuardModelError`.

---

### Q22: What is CSV Formula Injection (CSV Injection), and how does SpamGuard AI neutralize it?
**Answer:**  
CSV Formula Injection occurs when an attacker embeds spreadsheet command prefixes (`=`, `+`, `-`, `@`) or DDE commands into data. When an analyst exports the CSV and opens it in Microsoft Excel or Google Sheets, the spreadsheet software treats the text as a formula, potentially executing arbitrary shell commands.  
SpamGuard AI's `sanitize_csv_cell()` strips leading whitespace (`lstrip(" \t\r\n")`) and prepends a single apostrophe (`'`) to any cell starting with dangerous control characters, forcing spreadsheet applications to treat the cell strictly as literal text.

---

### Q23: What caching strategy did you employ in the Streamlit application and why?
**Answer:**  
We used `@st.cache_resource` for the `SpamPredictor` instance. Because the trained model is frozen, read-only, and shared across all sessions, caching the loaded object eliminates disk I/O and deserialization overhead on each user interaction. Conversely, user input text and batch CSV uploads are **never cached globally**, ensuring zero cross-session data leakage and strict user privacy.

---

### Q24: What were the measured performance benchmarks on your development hardware?
**Answer:**  
Profiled on an MSI laptop with a 12th Gen Intel Core i7-12650H CPU and 16 GB RAM:
- **Cold Model Load:** 76.62 ms
- **Warm Single Email Inference:** 1.04 ms median (P95: 2.14 ms)
- **Single Email with Token Explainability:** 1.59 ms median (0.55 ms overhead)
- **Vectorized Batch Classification:** 10,478 emails/sec throughput (0.095 ms per email)
- **Process Memory Footprint:** 162.4 MB peak working set (only ~9.2 MB model delta above baseline Python).

---

### Q25: What are the primary historical limitations of the Apache SpamAssassin dataset?
**Answer:**  
1. **Temporal Shift:** The corpus was collected circa 2002–2005. While lexical spam fundamentals (promotions, urgent claims, financial incentives) remain consistent, modern threats like QR-code phishing (quishing), adversarial zero-width spaces, and LLM-generated targeted spear-phishing are unrepresented.
2. **Domain Bias:** The legitimate "Ham" split consists primarily of open-source developer mailing lists (Linux kernel, RedHat RPM, Razor), skewing legitimate vocabulary toward software development terms.

---

### Q26: Does a "Ham" classification guarantee that an email is safe to open?
**Answer:**  
**No.** A "Ham" verdict signifies that the email's lexical content statistically aligns with legitimate patterns learned from historical data. It does not certify that links are safe, attachments are free from zero-day malware, or the sender address has not been spoofed via compromised credentials. In real enterprise environments, SpamGuard AI should operate alongside SPF/DKIM verification and sandboxed attachment analysis.

---

### Q27: How many automated tests exist in the project, and what areas do they cover?
**Answer:**  
There are **123 automated Pytest tests** with 100% pass rate covering:
1. `test_smoke.py` (4 tests): Directory structures, configuration constants, imports.
2. `test_prepare_data.py` (11 tests): In-memory archive extraction, MIME decoding, HTML parsing.
3. `test_eda.py` (9 tests): Statistical metric verification and figure generation.
4. `test_preprocess.py` (14 tests): Normalization, stratified splitting, leakage checks.
5. `test_train.py` (17 tests): Pipeline assembly, baseline benchmarking, tuning, persistence.
6. `test_evaluate.py` (16 tests): Test set quarantine, confusion matrix orientation, curve plotting.
7. `test_predict.py` (21 tests): Inference engine, probability calibration, CLI execution.
8. `test_app.py` (19 tests): Streamlit navigation, telemetry, chart construction.
9. `test_hardening.py` (12 tests): Formula injection, boundary testing, Unicode stress, session state.

---

### Q28: How would you extend SpamGuard AI in a future engineering phase?
**Answer:**  
1. **Continuous Online Learning:** Implement incremental learning algorithms (e.g., `MultinomialNB.partial_fit()`) or adaptive active learning to incorporate user feedback.
2. **Hybrid Transformer Ensembling:** Combine Naïve Bayes' sub-millisecond screening for high-confidence decisions with a lightweight distilled transformer (e.g., DistilBERT) for ambiguous edge cases.
3. **Transport Header Integration:** Parse SPF, DKIM, and DMARC authentication headers alongside body text for multi-vector threat scoring.
