# SpamGuard AI — Exploratory Data Analysis Findings

> **Academic AI Micro Project** | **Cybersecurity & Statistical Text Filtering**  
> **Course Competency Mapping:** CO2 (Algorithm Selection), CO4 (Model Engineering), CO5 (Evaluation Metrics)

---

## 1. Executive Summary & Dataset Composition

The exploratory data analysis examined the standardized **Apache SpamAssassin Public Corpus** prepared in Phase 02. The dataset contains **5,923 clean, non-empty email messages** categorized into binary classes:

- **Legitimate (Ham, Label 0):** 4,109 messages (69.37%)
- **Unsolicited (Spam, Label 1):** 1,814 messages (30.63%)
- **Class Imbalance Ratio:** 2.27:1 (Ham to Spam)

Every record contains valid non-empty classification text formed from decoded email subjects and sanitized message bodies. No synthetic records, artificial balancing, or fabricated metrics were introduced.

---

## 2. Key Statistical Findings & Observable Differences

### 2.1 Message Length Distributions
- **Spam Emails:**
  - Mean character length: **2,466.1** characters (Median: **1,226.0**)
  - Mean word count: **320.8** words (Median: **157.0**)
  - Minimum / Maximum: **9** / **126,044.0** characters
  - Interquartile Range (IQR): **743.25** to **2,255.25** characters
- **Ham Emails:**
  - Mean character length: **1,885.19** characters (Median: **1,022.0**)
  - Mean word count: **273.7** words (Median: **150.0**)
  - Minimum / Maximum: **45** / **194,844.0** characters
  - Interquartile Range (IQR): **570.0** to **1,714.0** characters

**Observation:** While both classes exhibit heavy-tailed right-skewed distributions with extreme outliers, spam emails have higher median character lengths (1,226.0 vs 1,022.0 chars) due to embedded styling metadata, marketing boilerplate, and HTML fallback formatting.

### 2.2 Punctuation & Lexical Indicators
- **Exclamation Marks (`!`):**
  - Present in **78.61%** of spam messages (Mean count: **5.84** per email).
  - Present in only **33.56%** of ham messages (Mean count: **0.86** per email).
  - Exclamation mark density serves as a noticeable indicator of urgency and promotional rhetoric in spam.
- **Digit Presence:**
  - Present in **93.83%** of spam and **89.39%** of ham. Both classes exhibit high numerical content (prices and phone numbers in spam; software versions, patches, and timestamps in ham).
- **URL Presence:**
  - Found in **85.08%** of ham emails compared to **56.34%** of spam emails.
  - *Critical Insight:* In mailing-list datasets (such as SpamAssassin), ham emails frequently contain automated administrative footers (e.g., listinfo, unsubscribe URLs, and source repositories). Therefore, the naive heuristic that "URLs imply spam" is statistically incorrect on this corpus.

### 2.3 Top Frequent Terms (Excluding Stop-Words)
- **Top Spam Terms:** `font` (4,037), `com` (3,677), `http` (3,080), `email` (2,711), `free` (2,705), `www` (2,028), `click` (1,869), `nbsp` (1,702), `money` (1,600), `business` (1,588)
  - Prominently features marketing keywords (`free`, `click`, `money`, `business`) and HTML styling artifacts (`font`, `nbsp`, `color`, `size`).
- **Top Ham Terms:** `http` (11,598), `com` (9,939), `www` (6,292), `net` (4,831), `list` (4,209), `linux` (3,849), `users` (2,706), `just` (2,388), `use` (2,365), `listinfo` (2,301)
  - Dominated by open-source technical discussions (`linux`, `users`, `lists`, `rpm`, `razor`, `mailman`).

---

## 3. Data Quality & Integrity Validation

- **Missing Values:** Zero missing values in primary classification text (`text`), target label (`label`), and unique identifier (`message_id`).
- **Subject Availability:** 99.73% of records have non-empty subjects.
- **Body Availability:** 99.63% of records have non-empty bodies.
- **Duplicates & Conflicts:** 0 duplicate message IDs, 0 duplicate content texts, and 0 conflicting labels exist in the prepared dataset.

---

## 4. Class Imbalance & Evaluation Metric Implications

The dataset contains a natural **69.37% Ham to 30.63% Spam** class distribution. 

### Why Accuracy Alone is Insufficient (CO5 Alignment):
A trivial majority-class classifier predicting "Ham" for every email would achieve an **Accuracy of 69.37%** while failing to intercept a single spam email (0% Recall, 0% Precision for Spam). Therefore, model evaluation must emphasize:
1. **Precision (Spam):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   Measures the proportion of emails flagged as spam that are truly spam. In email filtering, **False Positives (legitimate emails marked as spam) are catastrophic** because critical personal or business communications could be lost. High precision on the Spam class is paramount.
2. **Recall (Spam):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Measures the fraction of actual spam caught by the filter.
3. **F1-Score:**
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Provides a harmonic balance between precision and recall under class imbalance.

---

## 5. Architectural Alignment for Phase 04

The exploratory findings validate the selection of **TF-IDF + Multinomial Naïve Bayes** (CO2, CO4):
- **Multinomial Naïve Bayes** operates effectively on discrete feature frequencies, accommodating high-dimensional sparse representations typical of email vocabularies.
- **TF-IDF Weighting** will downweight ubiquitous corpus tokens (e.g., `http`, `com`, `list`) while boosting distinctive discriminatory terms (e.g., `free`, `click`, `guarantee`).
- **Strict Leakage Prevention:** Feature extraction parameters (vocabulary, IDF weights) must be fitted **strictly on training partitions** to avoid data leakage.

---

## 6. Historical Dataset Limitations

> [!WARNING]
> **Dataset Scope & Modern Threat Horizon**  
> The Apache SpamAssassin Public Corpus represents email patterns from the 2002–2006 era. Modern email threats frequently utilize sophisticated zero-day techniques, including AI-generated spear phishing, image-only steganography, obfuscated redirectors, and compromised legitimate cloud infrastructure. The exploratory findings presented here describe this historical benchmark and must not be interpreted as causal guarantees across contemporary enterprise email environments.
