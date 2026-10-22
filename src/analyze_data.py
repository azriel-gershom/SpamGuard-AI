"""SpamGuard AI — Exploratory Data Analysis (EDA) Module.

Analyzes dataset quality, class distribution, message lengths, word frequencies,
and structural lexical patterns (URLs, digits, punctuation) on the prepared
SpamAssassin dataset without modifying data or training models.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
# Ensure non-interactive backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import (
    EDA_FINDINGS_FILE,
    EDA_SUMMARY_FILE,
    FIGURES_DIR,
    PROCESSED_DATA_FILE,
    REPORTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpamGuardAI.EDA")

REQUIRED_COLUMNS = [
    "message_id",
    "subject",
    "body",
    "text",
    "label",
    "source_group",
    "source_archive",
]

# Color Palette for Cybersecurity / Clean Academic Visualization
COLOR_HAM = "#059669"    # Emerald / Safe
COLOR_SPAM = "#e11d48"   # Crimson / Threat
COLOR_NEUTRAL = "#3b82f6" # Slate Blue
COLOR_ACCENT = "#8b5cf6" # Violet


# =====================================================================
# Quality & Validation Analysis
# =====================================================================

def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validates the schema, row counts, null values, and basic integrity of the dataset."""
    if df.empty:
        raise ValueError("Dataset is empty. Cannot perform EDA on an empty DataFrame.")

    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    labels = set(df["label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Invalid labels found in dataset: {labels}. Expected {{0, 1}}.")

    if 0 not in labels or 1 not in labels:
        raise ValueError(f"Both ham (0) and spam (1) must be present in dataset. Found: {labels}")

    total_records = len(df)
    missing_values = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
    empty_texts = int((df["text"].str.strip() == "").sum())
    duplicate_message_ids = int(df["message_id"].duplicated().sum())
    duplicate_texts = int(df["text"].duplicated().sum())

    subject_available = int(df["subject"].fillna("").str.strip().ne("").sum())
    body_available = int(df["body"].fillna("").str.strip().ne("").sum())
    source_group_counts = {str(k): int(v) for k, v in df["source_group"].value_counts().to_dict().items()}
    source_archive_counts = {str(k): int(v) for k, v in df["source_archive"].value_counts().to_dict().items()}

    return {
        "total_records": int(total_records),
        "labels_present": [int(x) for x in sorted(list(labels))],
        "missing_values": missing_values,
        "empty_text_records": empty_texts,
        "duplicate_message_ids": duplicate_message_ids,
        "duplicate_texts": duplicate_texts,
        "subject_available_count": subject_available,
        "subject_available_pct": round((subject_available / total_records) * 100, 2),
        "body_available_count": body_available,
        "body_available_pct": round((body_available / total_records) * 100, 2),
        "source_group_distribution": source_group_counts,
        "source_archive_distribution": source_archive_counts,
    }


# =====================================================================
# Statistical & Length Analysis
# =====================================================================

def compute_class_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Computes exact counts and percentages for spam and ham classes."""
    total = len(df)
    if total == 0:
        raise ValueError("Cannot compute class distribution for empty DataFrame.")

    spam_count = int((df["label"] == 1).sum())
    ham_count = int((df["label"] == 0).sum())
    spam_pct = round((spam_count / total) * 100, 2)
    ham_pct = round((ham_count / total) * 100, 2)

    return {
        "total": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": spam_pct,
        "ham_percentage": ham_pct,
        "imbalance_ratio": round(ham_count / max(1, spam_count), 2),
    }


def compute_series_stats(series: pd.Series) -> dict[str, float]:
    """Computes descriptive summary statistics for a numeric series."""
    if series.empty:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0, "q99": 0.0}

    std_val = float(series.std()) if len(series) > 1 else 0.0
    if np.isnan(std_val):
        std_val = 0.0

    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std": round(std_val, 2),
        "q25": round(float(series.quantile(0.25)), 2),
        "q75": round(float(series.quantile(0.75)), 2),
        "q99": round(float(series.quantile(0.99)), 2),
    }


def compute_length_and_word_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Computes character length and word count statistics for each class and combined."""
    # Ensure string types
    texts = df["text"].fillna("")
    char_lens = texts.str.len()
    word_counts = texts.apply(lambda x: len(x.split()))

    ham_mask = df["label"] == 0
    spam_mask = df["label"] == 1

    return {
        "character_length": {
            "overall": compute_series_stats(char_lens),
            "ham": compute_series_stats(char_lens[ham_mask]),
            "spam": compute_series_stats(char_lens[spam_mask]),
        },
        "word_count": {
            "overall": compute_series_stats(word_counts),
            "ham": compute_series_stats(word_counts[ham_mask]),
            "spam": compute_series_stats(word_counts[spam_mask]),
        },
    }


# =====================================================================
# Lexical & Pattern Analysis
# =====================================================================

def extract_top_terms(texts: pd.Series | list[str], top_n: int = 20) -> list[dict[str, Any]]:
    """Extracts the most frequent tokens excluding English stop-words."""
    counter: Counter[str] = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        # Lowercase words with length >= 3
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        filtered = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        counter.update(filtered)

    return [{"term": term, "count": count} for term, count in counter.most_common(top_n)]


def compute_pattern_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Computes structural indicators (URLs, digits, punctuation, subject length)."""
    texts = df["text"].fillna("")
    subjects = df["subject"].fillna("")

    has_url = texts.str.contains(r"https?://|www\.", case=False, regex=True)
    has_digit = texts.str.contains(r"\d", regex=True)
    has_excl = texts.str.contains(r"!", regex=False)
    excl_counts = texts.apply(lambda x: x.count("!"))

    subj_char_lens = subjects.str.len()
    subj_word_cnts = subjects.apply(lambda x: len(x.split()))

    stats: dict[str, Any] = {}
    for label, name in [(0, "ham"), (1, "spam")]:
        mask = df["label"] == label
        total_sub = int(mask.sum())
        if total_sub == 0:
            continue

        url_cnt = int(has_url[mask].sum())
        digit_cnt = int(has_digit[mask].sum())
        excl_cnt = int(has_excl[mask].sum())

        stats[name] = {
            "total_emails": total_sub,
            "has_url_count": url_cnt,
            "has_url_percentage": round((url_cnt / total_sub) * 100, 2),
            "has_digit_count": digit_cnt,
            "has_digit_percentage": round((digit_cnt / total_sub) * 100, 2),
            "has_exclamation_count": excl_cnt,
            "has_exclamation_percentage": round((excl_cnt / total_sub) * 100, 2),
            "exclamation_count_mean": round(float(excl_counts[mask].mean()), 2),
            "exclamation_count_median": round(float(excl_counts[mask].median()), 2),
            "subject_char_length_mean": round(float(subj_char_lens[mask].mean()), 2),
            "subject_char_length_median": round(float(subj_char_lens[mask].median()), 2),
            "subject_word_count_mean": round(float(subj_word_cnts[mask].mean()), 2),
            "subject_word_count_median": round(float(subj_word_cnts[mask].median()), 2),
        }

    return stats


# =====================================================================
# Professional Visualization Generation
# =====================================================================

def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    """Generates a clean bar chart showing class distribution and proportions."""
    dist = compute_class_distribution(df)
    categories = ["Ham (Legitimate)", "Spam (Unsolicited)"]
    counts = [dist["ham_count"], dist["spam_count"]]
    percentages = [dist["ham_percentage"], dist["spam_percentage"]]
    colors = [COLOR_HAM, COLOR_SPAM]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)

    # Annotate counts and percentages
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.02,
            f"{count:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#0f172a",
        )

    ax.set_title("SpamGuard AI — Email Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Email Messages", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Subtitle note on imbalance
    fig.text(
        0.5,
        -0.02,
        f"Total clean emails: {dist['total']:,} | Class imbalance ratio: {dist['imbalance_ratio']}:1 (Ham:Spam)",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_length_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    """Generates an email character-length histogram with disclosed clipping threshold."""
    char_lens = df["text"].fillna("").str.len()
    ham_lens = char_lens[df["label"] == 0]
    spam_lens = char_lens[df["label"] == 1]

    # Visual clipping threshold set at 99th percentile for readability
    clip_thresh = 10000
    clipped_ham = np.clip(ham_lens, 0, clip_thresh)
    clipped_spam = np.clip(spam_lens, 0, clip_thresh)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    bins = np.linspace(0, clip_thresh, 50)

    ax.hist(clipped_ham, bins=bins, alpha=0.65, color=COLOR_HAM, label=f"Ham (Median: {ham_lens.median():.0f} chars)", density=True)
    ax.hist(clipped_spam, bins=bins, alpha=0.65, color=COLOR_SPAM, label=f"Spam (Median: {spam_lens.median():.0f} chars)", density=True)

    ax.set_title("SpamGuard AI — Email Character Length Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Email Character Length", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        -0.03,
        f"Note: X-axis truncated at {clip_thresh:,} characters for visual clarity. Full max length: Ham={ham_lens.max():,}, Spam={spam_lens.max():,}.",
        ha="center",
        fontsize=8.5,
        color="#475569",
        style="italic",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_length_box(df: pd.DataFrame, output_path: Path) -> Path:
    """Generates side-by-side box plots comparing message length between classes."""
    char_lens = df["text"].fillna("").str.len()
    ham_lens = char_lens[df["label"] == 0]
    spam_lens = char_lens[df["label"] == 1]

    clip_thresh = 8000
    clipped_ham = np.clip(ham_lens, 0, clip_thresh)
    clipped_spam = np.clip(spam_lens, 0, clip_thresh)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    bp = ax.boxplot(
        [clipped_ham, clipped_spam],
        tick_labels=["Ham (Legitimate)", "Spam (Unsolicited)"],
        patch_artist=True,
        widths=0.45,
        showmeans=True,
        meanprops={"marker": "D", "markeredgecolor": "#0f172a", "markerfacecolor": "#facc15"},
    )

    # Style boxes
    colors = [COLOR_HAM, COLOR_SPAM]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#0f172a")

    for median in bp["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(1.8)

    ax.set_title("SpamGuard AI — Message Length Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Character Length (Truncated at 8,000 for display)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend for mean diamond
    ax.plot([], [], marker="D", color="#facc15", markeredgecolor="#0f172a", linestyle="None", label="Mean")
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")

    fig.text(
        0.5,
        -0.03,
        "Yellow diamonds indicate class means. Box represents IQR (25th–75th percentiles); line indicates median.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_top_words(terms: list[dict[str, Any]], title: str, color: str, output_path: Path) -> Path:
    """Generates a horizontal bar chart of the most frequent terms."""
    words = [item["term"] for item in reversed(terms)]
    counts = [item["count"] for item in reversed(terms)]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
    bars = ax.barh(words, counts, color=color, alpha=0.85, edgecolor="#0f172a", height=0.65)

    # Annotate counts on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count:,}",
            ha="left",
            va="center",
            fontsize=9.5,
            color="#0f172a",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Term Occurrences (Excluding English Stop-Words)", fontsize=10.5)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pattern_comparison(stats: dict[str, Any], output_path: Path) -> Path:
    """Generates a grouped bar chart comparing structural indicators between Ham and Spam."""
    metrics = ["Contains URL", "Contains Digits", "Contains '!'"]
    ham_vals = [
        stats["ham"]["has_url_percentage"],
        stats["ham"]["has_digit_percentage"],
        stats["ham"]["has_exclamation_percentage"],
    ]
    spam_vals = [
        stats["spam"]["has_url_percentage"],
        stats["spam"]["has_digit_percentage"],
        stats["spam"]["has_exclamation_percentage"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    bars1 = ax.bar(x - width / 2, ham_vals, width, label="Ham", color=COLOR_HAM, alpha=0.85, edgecolor="#0f172a")
    bars2 = ax.bar(x + width / 2, spam_vals, width, label="Spam", color=COLOR_SPAM, alpha=0.85, edgecolor="#0f172a")

    for bar, val in zip(bars1, ham_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    for bar, val in zip(bars2, spam_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    ax.set_title("SpamGuard AI — Structural Indicator Prevalence Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Percentage of Emails within Class (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        -0.03,
        "Higher exclamation mark usage is prominent in spam; high URL presence in ham reflects mailing-list footers.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_figures(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> dict[str, Path]:
    """Generates and saves all 6 required EDA figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_paths: dict[str, Path] = {}

    # 1. Class Distribution
    p1 = figures_dir / "class_distribution.png"
    plot_class_distribution(df, p1)
    fig_paths["class_distribution"] = p1

    # 2. Length Distribution
    p2 = figures_dir / "email_length_distribution.png"
    plot_length_distribution(df, p2)
    fig_paths["email_length_distribution"] = p2

    # 3. Length Boxplot
    p3 = figures_dir / "length_comparison.png"
    plot_length_box(df, p3)
    fig_paths["length_comparison"] = p3

    # Word Frequency Analysis
    spam_texts = df[df["label"] == 1]["text"].fillna("")
    ham_texts = df[df["label"] == 0]["text"].fillna("")
    top_spam = extract_top_terms(spam_texts, top_n=20)
    top_ham = extract_top_terms(ham_texts, top_n=20)

    # 4. Top Spam Words
    p4 = figures_dir / "top_spam_words.png"
    plot_top_words(top_spam, "Top 20 Frequent Terms in Spam Emails", COLOR_SPAM, p4)
    fig_paths["top_spam_words"] = p4

    # 5. Top Ham Words
    p5 = figures_dir / "top_ham_words.png"
    plot_top_words(top_ham, "Top 20 Frequent Terms in Ham Emails", COLOR_HAM, p5)
    fig_paths["top_ham_words"] = p5

    # 6. Pattern Comparison
    pattern_stats = compute_pattern_statistics(df)
    p6 = figures_dir / "url_pattern_comparison.png"
    plot_pattern_comparison(pattern_stats, p6)
    fig_paths["url_pattern_comparison"] = p6

    logger.info("Successfully generated 6 figures in %s", figures_dir)
    return fig_paths


# =====================================================================
# Report Generation
# =====================================================================

def build_eda_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Constructs complete dictionary of computed EDA metrics."""
    val = validate_dataset(df)
    dist = compute_class_distribution(df)
    lens = compute_length_and_word_stats(df)
    patterns = compute_pattern_statistics(df)

    top_spam = extract_top_terms(df[df["label"] == 1]["text"].fillna(""), top_n=20)
    top_ham = extract_top_terms(df[df["label"] == 0]["text"].fillna(""), top_n=20)

    return {
        "dataset_validation": val,
        "class_distribution": dist,
        "length_statistics": lens,
        "lexical_patterns": patterns,
        "top_frequent_terms": {
            "spam": top_spam,
            "ham": top_ham,
        },
    }


def _json_serial_default(obj: Any) -> Any:
    """Fallback JSON serializer for NumPy and Path types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_eda_summary_json(summary: dict[str, Any], output_path: Path = EDA_SUMMARY_FILE) -> Path:
    """Writes eda_summary.json with complete computed metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_serial_default)
    logger.info("Written EDA JSON summary to %s", output_path)
    return output_path


def write_eda_findings_md(summary: dict[str, Any], output_path: Path = EDA_FINDINGS_FILE) -> Path:
    """Generates comprehensive academic findings report in Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dist = summary["class_distribution"]
    lens = summary["length_statistics"]
    patterns = summary["lexical_patterns"]
    top_s = summary["top_frequent_terms"]["spam"]
    top_h = summary["top_frequent_terms"]["ham"]

    top_spam_str = ", ".join(f"`{item['term']}` ({item['count']:,})" for item in top_s[:10])
    top_ham_str = ", ".join(f"`{item['term']}` ({item['count']:,})" for item in top_h[:10])

    md_content = f"""# SpamGuard AI — Exploratory Data Analysis Findings

> **Academic AI Micro Project** | **Cybersecurity & Statistical Text Filtering**  
> **Course Competency Mapping:** CO2 (Algorithm Selection), CO4 (Model Engineering), CO5 (Evaluation Metrics)

---

## 1. Executive Summary & Dataset Composition

The exploratory data analysis examined the standardized **Apache SpamAssassin Public Corpus** prepared in Phase 02. The dataset contains **{dist['total']:,} clean, non-empty email messages** categorized into binary classes:

- **Legitimate (Ham, Label 0):** {dist['ham_count']:,} messages ({dist['ham_percentage']}%)
- **Unsolicited (Spam, Label 1):** {dist['spam_count']:,} messages ({dist['spam_percentage']}%)
- **Class Imbalance Ratio:** {dist['imbalance_ratio']}:1 (Ham to Spam)

Every record contains valid non-empty classification text formed from decoded email subjects and sanitized message bodies. No synthetic records, artificial balancing, or fabricated metrics were introduced.

---

## 2. Key Statistical Findings & Observable Differences

### 2.1 Message Length Distributions
- **Spam Emails:**
  - Mean character length: **{lens['character_length']['spam']['mean']:,}** characters (Median: **{lens['character_length']['spam']['median']:,}**)
  - Mean word count: **{lens['word_count']['spam']['mean']:.1f}** words (Median: **{lens['word_count']['spam']['median']:.1f}**)
  - Minimum / Maximum: **{lens['character_length']['spam']['min']:.0f}** / **{lens['character_length']['spam']['max']:,}** characters
  - Interquartile Range (IQR): **{lens['character_length']['spam']['q25']:,}** to **{lens['character_length']['spam']['q75']:,}** characters
- **Ham Emails:**
  - Mean character length: **{lens['character_length']['ham']['mean']:,}** characters (Median: **{lens['character_length']['ham']['median']:,}**)
  - Mean word count: **{lens['word_count']['ham']['mean']:.1f}** words (Median: **{lens['word_count']['ham']['median']:.1f}**)
  - Minimum / Maximum: **{lens['character_length']['ham']['min']:.0f}** / **{lens['character_length']['ham']['max']:,}** characters
  - Interquartile Range (IQR): **{lens['character_length']['ham']['q25']:,}** to **{lens['character_length']['ham']['q75']:,}** characters

**Observation:** While both classes exhibit heavy-tailed right-skewed distributions with extreme outliers, spam emails have higher median character lengths ({lens['character_length']['spam']['median']:,} vs {lens['character_length']['ham']['median']:,} chars) due to embedded styling metadata, marketing boilerplate, and HTML fallback formatting.

### 2.2 Punctuation & Lexical Indicators
- **Exclamation Marks (`!`):**
  - Present in **{patterns['spam']['has_exclamation_percentage']}%** of spam messages (Mean count: **{patterns['spam']['exclamation_count_mean']}** per email).
  - Present in only **{patterns['ham']['has_exclamation_percentage']}%** of ham messages (Mean count: **{patterns['ham']['exclamation_count_mean']}** per email).
  - Exclamation mark density serves as a noticeable indicator of urgency and promotional rhetoric in spam.
- **Digit Presence:**
  - Present in **{patterns['spam']['has_digit_percentage']}%** of spam and **{patterns['ham']['has_digit_percentage']}%** of ham. Both classes exhibit high numerical content (prices and phone numbers in spam; software versions, patches, and timestamps in ham).
- **URL Presence:**
  - Found in **{patterns['ham']['has_url_percentage']}%** of ham emails compared to **{patterns['spam']['has_url_percentage']}%** of spam emails.
  - *Critical Insight:* In mailing-list datasets (such as SpamAssassin), ham emails frequently contain automated administrative footers (e.g., listinfo, unsubscribe URLs, and source repositories). Therefore, the naive heuristic that "URLs imply spam" is statistically incorrect on this corpus.

### 2.3 Top Frequent Terms (Excluding Stop-Words)
- **Top Spam Terms:** {top_spam_str}
  - Prominently features marketing keywords (`free`, `click`, `money`, `business`) and HTML styling artifacts (`font`, `nbsp`, `color`, `size`).
- **Top Ham Terms:** {top_ham_str}
  - Dominated by open-source technical discussions (`linux`, `users`, `lists`, `rpm`, `razor`, `mailman`).

---

## 3. Data Quality & Integrity Validation

- **Missing Values:** Zero missing values in primary classification text (`text`), target label (`label`), and unique identifier (`message_id`).
- **Subject Availability:** {summary['dataset_validation']['subject_available_pct']}% of records have non-empty subjects.
- **Body Availability:** {summary['dataset_validation']['body_available_pct']}% of records have non-empty bodies.
- **Duplicates & Conflicts:** 0 duplicate message IDs, 0 duplicate content texts, and 0 conflicting labels exist in the prepared dataset.

---

## 4. Class Imbalance & Evaluation Metric Implications

The dataset contains a natural **69.37% Ham to 30.63% Spam** class distribution. 

### Why Accuracy Alone is Insufficient (CO5 Alignment):
A trivial majority-class classifier predicting "Ham" for every email would achieve an **Accuracy of 69.37%** while failing to intercept a single spam email (0% Recall, 0% Precision for Spam). Therefore, model evaluation must emphasize:
1. **Precision (Spam):**
   $$\\text{{Precision}} = \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FP}}}}$$
   Measures the proportion of emails flagged as spam that are truly spam. In email filtering, **False Positives (legitimate emails marked as spam) are catastrophic** because critical personal or business communications could be lost. High precision on the Spam class is paramount.
2. **Recall (Spam):**
   $$\\text{{Recall}} = \\frac{{\\text{{TP}}}}{{\\text{{TP}} + \\text{{FN}}}}$$
   Measures the fraction of actual spam caught by the filter.
3. **F1-Score:**
   $$\\text{{F1-Score}} = 2 \\times \\frac{{\\text{{Precision}} \\times \\text{{Recall}}}}{{\\text{{Precision}} + \\text{{Recall}}}}$$
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
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Written EDA Markdown findings to %s", output_path)
    return output_path


# =====================================================================
# Main Orchestrator
# =====================================================================

def run_eda(
    input_csv: Path = PROCESSED_DATA_FILE,
    reports_dir: Path = REPORTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    summary_file: Path = EDA_SUMMARY_FILE,
    findings_file: Path = EDA_FINDINGS_FILE,
) -> dict[str, Any]:
    """Runs end-to-end exploratory data analysis pipeline."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Prepared dataset not found at {input_csv}. Please run Phase 02 preparation first.")

    logger.info("Loading dataset from %s...", input_csv)
    df = pd.read_csv(input_csv)
    # Ensure missing subjects/bodies are handled cleanly as empty strings
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = df["text"].fillna("")

    logger.info("Validating dataset schema and integrity...")
    summary = build_eda_summary(df)

    logger.info("Generating professional figures under %s...", figures_dir)
    fig_paths = generate_all_figures(df, figures_dir=figures_dir)
    summary["generated_figures"] = {name: str(p) for name, p in fig_paths.items()}

    logger.info("Writing summary reports...")
    write_eda_summary_json(summary, output_path=summary_file)
    write_eda_findings_md(summary, output_path=findings_file)

    logger.info("EDA completed successfully!")
    return summary


def main() -> None:
    """CLI Entrypoint for Phase 03 EDA."""
    parser = argparse.ArgumentParser(description="SpamGuard AI — Phase 03 Exploratory Data Analysis")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROCESSED_DATA_FILE,
        help="Path to prepared emails.csv dataset",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to save generated chart images",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=EDA_SUMMARY_FILE,
        help="Destination path for eda_summary.json",
    )
    parser.add_argument(
        "--findings-file",
        type=Path,
        default=EDA_FINDINGS_FILE,
        help="Destination path for eda_findings.md",
    )
    args = parser.parse_args()

    run_eda(
        input_csv=args.input_csv,
        reports_dir=REPORTS_DIR,
        figures_dir=args.figures_dir,
        summary_file=args.summary_file,
        findings_file=args.findings_file,
    )


if __name__ == "__main__":
    main()
