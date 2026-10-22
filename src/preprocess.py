"""SpamGuard AI — Text Preprocessing & Leakage-Safe Data Splitting.

Phase 04:
- Performs safe, consistent text normalization (Unicode NFKC, whitespace, null bytes).
- Implements reproducible stratified train/validation/test splitting (70% / 15% / 15%).
- Strictly prevents data leakage across partitions (zero shared IDs, zero identical content).
- Provides a reusable, unfitted TF-IDF vectorizer factory.
- Exports split CSV files and comprehensive summary reports without model fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_SUMMARY_FILE,
    SPLITS_DIR,
    TEST_RATIO,
    TEST_SPLIT_FILE,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TRAIN_RATIO,
    TRAIN_SPLIT_FILE,
    VAL_RATIO,
    VAL_SPLIT_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpamGuardAI.Preprocess")

SPLIT_COLUMNS = ["message_id", "text", "label", "source_group"]


# =====================================================================
# Text Preprocessing & Normalization
# =====================================================================

def clean_text(text: Any) -> str:
    """Performs safe, lightweight text normalization suitable for TF-IDF vectorization.

    Guarantees:
    - Handles None, non-string, or empty input gracefully (returns "").
    - Normalizes Unicode characters to standard compatibility decomposition (NFKC).
    - Standardizes all line breaks (CRLF and CR to LF).
    - Removes null bytes (0x00) and unprintable control characters (except newline and tab).
    - Collapses multiple horizontal spaces/tabs to a single space.
    - Limits excessive consecutive blank lines (max 2 consecutive newlines).
    - Preserves words, digits, URLs, email addresses, currency symbols, and punctuation (!, $, ?, @, %).
    - Does not truncate long emails or apply aggressive stem/stop-word removal.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return ""

    # Safe Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Standardize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Remove non-printable control characters except newline (\n) and tab (\t)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # Collapse consecutive horizontal spaces/tabs while preserving line breaks
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive consecutive newlines (more than 2 down to 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA-256 checksum of a file in 64KB blocks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =====================================================================
# Stratified Partitioning & Leakage Prevention
# =====================================================================

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into stratified, disjoint train, validation, and test partitions.

    Splits strictly:
    - Train: 70%
    - Validation: 15%
    - Test: 15%
    Using deterministic stratification by the target 'label'.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    required = {"message_id", "text", "label", "source_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns for splitting: {missing}")

    labels = set(df["label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Invalid label values found: {labels}. Expected {{0, 1}}.")

    if 0 not in labels or 1 not in labels:
        raise ValueError(f"Both ham (0) and spam (1) must be present to split. Found: {labels}")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0. Current sum: {total_ratio}")

    # Work on clean copy with normalized text
    df_clean = df.copy()
    df_clean["text"] = df_clean["text"].apply(clean_text)

    # Discard any edge-case messages that became genuinely empty after normalization
    empty_mask = df_clean["text"].str.strip() == ""
    if empty_mask.any():
        logger.warning("Dropping %d messages that normalized to empty string.", int(empty_mask.sum()))
        df_clean = df_clean[~empty_mask].copy()

    # Compute normalized content hash for leakage-safe group splitting
    df_clean["content_hash"] = df_clean["text"].apply(
        lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
    )

    # Deterministic grouping by content hash to prevent identical content appearing across splits
    group_df = (
        df_clean.groupby("content_hash")
        .agg(label=("label", "first"), count=("message_id", "count"))
        .reset_index()
        .sort_values("content_hash")
        .reset_index(drop=True)
    )

    # Step 1: Stratified train / holdout split on unique content groups
    min_count_step1 = group_df["label"].value_counts().min()
    stratify_step1 = group_df["label"] if min_count_step1 >= 2 else None
    train_groups, holdout_groups = train_test_split(
        group_df,
        train_size=train_ratio,
        stratify=stratify_step1,
        random_state=random_state,
    )

    # Step 2: Split holdout groups equally into validation and test groups
    holdout_test_ratio = test_ratio / (val_ratio + test_ratio)
    min_count_step2 = holdout_groups["label"].value_counts().min()
    stratify_step2 = holdout_groups["label"] if min_count_step2 >= 2 else None
    val_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=holdout_test_ratio,
        stratify=stratify_step2,
        random_state=random_state,
    )

    train_hashes = set(train_groups["content_hash"])
    val_hashes = set(val_groups["content_hash"])
    test_hashes = set(test_groups["content_hash"])

    # Map groups back to constituent email records
    train_df = (
        df_clean[df_clean["content_hash"].isin(train_hashes)][SPLIT_COLUMNS]
        .sort_values("message_id")
        .reset_index(drop=True)
    )
    val_df = (
        df_clean[df_clean["content_hash"].isin(val_hashes)][SPLIT_COLUMNS]
        .sort_values("message_id")
        .reset_index(drop=True)
    )
    test_df = (
        df_clean[df_clean["content_hash"].isin(test_hashes)][SPLIT_COLUMNS]
        .sort_values("message_id")
        .reset_index(drop=True)
    )

    # =================================================================
    # Mandatory Data Leakage & Disjointness Assertions
    # =================================================================
    train_ids = set(train_df["message_id"])
    val_ids = set(val_df["message_id"])
    test_ids = set(test_df["message_id"])

    # 1. Message ID Disjointness
    if train_ids & val_ids:
        raise AssertionError(f"Leakage: Train and Validation share {len(train_ids & val_ids)} message IDs")
    if train_ids & test_ids:
        raise AssertionError(f"Leakage: Train and Test share {len(train_ids & test_ids)} message IDs")
    if val_ids & test_ids:
        raise AssertionError(f"Leakage: Validation and Test share {len(val_ids & test_ids)} message IDs")

    # 2. Total Coverage
    all_split_ids = train_ids | val_ids | test_ids
    if len(all_split_ids) != len(df_clean):
        raise AssertionError(
            f"Coverage mismatch: Original={len(df_clean)}, Split union={len(all_split_ids)}"
        )

    # 3. Content Disjointness
    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    if train_texts & val_texts:
        raise AssertionError(f"Leakage: Train and Validation share {len(train_texts & val_texts)} identical texts")
    if train_texts & test_texts:
        raise AssertionError(f"Leakage: Train and Test share {len(train_texts & test_texts)} identical texts")
    if val_texts & test_texts:
        raise AssertionError(f"Leakage: Validation and Test share {len(val_texts & test_texts)} identical texts")

    logger.info(
        "Successfully verified partition independence: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df_clean),
    )
    return train_df, val_df, test_df


# =====================================================================
# Reusable TF-IDF Vectorizer Factory (Unfitted)
# =====================================================================

def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int | float = TFIDF_MIN_DF,
    max_df: int | float = TFIDF_MAX_DF,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Builds a standardized, unfitted TfidfVectorizer instance.

    Academic & ML Design Rationale:
    - analyzer='word': Standard lexical word tokenization.
    - ngram_range=(1, 2): Captures unigrams and bigrams (e.g., 'free money', 'click here')
      critical for identifying promotional spam rhetoric.
    - min_df=2: Prunes hapax legomena (single-occurrence typos and noise).
    - max_df=0.98: Prunes near-ubiquitous corpus words.
    - max_features=20000: Bounds vocabulary dimensionality to prevent memory explosion.
    - sublinear_tf=True: Applies 1 + log(tf) scaling to dampen the effect of repetitive keywords.
    - lowercase=True: Case-insensitivity at vectorization stage.
    - norm='l2': Unit vector Euclidean normalization.
    - strip_accents='unicode': Maps accented characters to base Latin characters.

    CRITICAL LEAKAGE RULE:
    This factory strictly returns an unfitted vectorizer. Vectorizer fitting must occur
    ONLY on the training split in Phase 05.
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        norm="l2",
        strip_accents="unicode",
    )


# =====================================================================
# Split Export & Summary Reporting
# =====================================================================

def export_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_dir: Path = SPLITS_DIR,
) -> dict[str, Path]:
    """Exports train.csv, validation.csv, and test.csv to data/processed/splits/."""
    splits_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": splits_dir / "train.csv",
        "validation": splits_dir / "validation.csv",
        "test": splits_dir / "test.csv",
    }

    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    val_df.to_csv(paths["validation"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")

    logger.info("Exported partitioned splits to %s", splits_dir)
    return paths


def generate_split_summary(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_file: Path,
    output_path: Path = SPLIT_SUMMARY_FILE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Generates deterministic JSON summary report of dataset partitions and leakage checks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def class_breakdown(sub_df: pd.DataFrame) -> dict[str, Any]:
        tot = len(sub_df)
        sp_cnt = int((sub_df["label"] == 1).sum())
        hm_cnt = int((sub_df["label"] == 0).sum())
        return {
            "total": tot,
            "ham_count": hm_cnt,
            "spam_count": sp_cnt,
            "ham_percentage": round((hm_cnt / tot) * 100, 2) if tot else 0.0,
            "spam_percentage": round((sp_cnt / tot) * 100, 2) if tot else 0.0,
        }

    input_sha256 = compute_file_sha256(input_file) if input_file.exists() else "unavailable"

    # Compute disjointness verification
    train_ids = set(train_df["message_id"])
    val_ids = set(val_df["message_id"])
    test_ids = set(test_df["message_id"])

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    summary: dict[str, Any] = {
        "dataset_name": "SpamGuard AI Labeled Corpus Splits",
        "input_dataset_path": str(input_file),
        "input_dataset_sha256": input_sha256,
        "input_record_count": len(df),
        "random_seed": random_state,
        "split_methodology": "Stratified random split (70% train, 15% validation, 15% test)",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "split_record_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "total": len(train_df) + len(val_df) + len(test_df),
        },
        "class_distributions": {
            "original": class_breakdown(df),
            "train": class_breakdown(train_df),
            "validation": class_breakdown(val_df),
            "test": class_breakdown(test_df),
        },
        "leakage_and_disjointness_checks": {
            "shared_message_ids_train_val": len(train_ids & val_ids),
            "shared_message_ids_train_test": len(train_ids & test_ids),
            "shared_message_ids_val_test": len(val_ids & test_ids),
            "shared_content_train_val": len(train_texts & val_texts),
            "shared_content_train_test": len(train_texts & test_texts),
            "shared_content_val_test": len(val_texts & test_texts),
            "complete_coverage": (len(train_ids | val_ids | test_ids) == len(df)),
        },
        "missing_value_checks": {
            "train_missing_text": int(train_df["text"].str.strip().eq("").sum()),
            "val_missing_text": int(val_df["text"].str.strip().eq("").sum()),
            "test_missing_text": int(test_df["text"].str.strip().eq("").sum()),
        },
        "tfidf_configuration": {
            "analyzer": "word",
            "ngram_range": list(TFIDF_NGRAM_RANGE),
            "max_features": TFIDF_MAX_FEATURES,
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "sublinear_tf": True,
            "lowercase": True,
            "norm": "l2",
            "strip_accents": "unicode",
            "fitted_in_phase_04": False,
        },
        "residual_leakage_note": (
            "Exact duplicate texts and identical message IDs have been verified to have zero overlap "
            "across partitions. However, residual leakage risk remains inherent to historical mailing-list "
            "corpora where multi-turn reply chains share identical quoted header snippets. Strict isolation "
            "of vectorizer fitting to the training partition ensures no statistical leakage of IDF weights."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Split summary report written to %s", output_path)
    return summary


# =====================================================================
# Pipeline Orchestrator
# =====================================================================

def run_preprocessing_and_splitting(
    input_file: Path = PROCESSED_DATA_FILE,
    splits_dir: Path = SPLITS_DIR,
    summary_file: Path = SPLIT_SUMMARY_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Loads processed dataset, validates, cleans, splits, exports, and summarizes."""
    if not input_file.exists():
        raise FileNotFoundError(f"Processed dataset not found at {input_file}. Please run Phase 02 preparation first.")

    logger.info("Loading processed dataset from %s...", input_file)
    df = pd.read_csv(input_file)

    logger.info("Splitting dataset into stratified train/val/test partitions...")
    train_df, val_df, test_df = split_dataset(df)

    logger.info("Exporting partitioned split CSVs...")
    export_splits(train_df, val_df, test_df, splits_dir=splits_dir)

    logger.info("Generating split summary report...")
    summary = generate_split_summary(
        df,
        train_df,
        val_df,
        test_df,
        input_file=input_file,
        output_path=summary_file,
    )

    logger.info("Phase 04 text preprocessing & splitting completed successfully.")
    return train_df, val_df, test_df, summary


def main() -> None:
    """CLI Entrypoint for Phase 04 Preprocessing & Splitting."""
    parser = argparse.ArgumentParser(description="SpamGuard AI — Text Preprocessing & Splitting")
    parser.add_argument(
        "--split",
        action="store_true",
        help="Execute dataset cleaning and stratified train/val/test splitting",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=PROCESSED_DATA_FILE,
        help="Path to input processed emails.csv",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory to save train.csv, validation.csv, test.csv",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=SPLIT_SUMMARY_FILE,
        help="Destination path for split_summary.json",
    )
    args = parser.parse_args()

    run_preprocessing_and_splitting(
        input_file=args.input_file,
        splits_dir=args.splits_dir,
        summary_file=args.summary_file,
    )


if __name__ == "__main__":
    main()
