"""Unit and integration tests for SpamGuard AI Phase 04 Text Preprocessing & Splitting."""

import numpy as np
import pandas as pd
import pytest

from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from src.preprocess import (
    SPLIT_COLUMNS,
    build_tfidf_vectorizer,
    clean_text,
    split_dataset,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_split_df() -> pd.DataFrame:
    """Creates a deterministic synthetic dataset fixture with 20 balanced records for testing splits."""
    rows = []
    # 14 Ham records (70% of 20)
    for i in range(14):
        rows.append({
            "message_id": f"ham_msg_{i:03d}",
            "text": f"Legitimate team status report number {i} regarding project milestones.",
            "label": 0,
            "source_group": "easy_ham",
        })
    # 6 Spam records (30% of 20)
    for i in range(6):
        rows.append({
            "message_id": f"spam_msg_{i:03d}",
            "text": f"URGENT: Claim your free prize ${i * 1000} now! Visit http://promo{i}.com today!",
            "label": 1,
            "source_group": "spam",
        })
    return pd.DataFrame(rows)


# =====================================================================
# Task 7 Tests: Text Preprocessing & Normalization
# =====================================================================

def test_unicode_normalization():
    """Verify clean_text applies NFKC decomposition to Unicode characters."""
    raw = "Caf\u0065\u0301 Special Deal"  # Combining acute accent
    cleaned = clean_text(raw)
    assert cleaned == "Café Special Deal"
    assert "\u0301" not in cleaned


def test_whitespace_normalization():
    """Verify clean_text standardizes CRLF, collapses horizontal spaces, and trims."""
    raw = "Header Line\r\n\r\n\r\n\r\nBody   with    spaces\tand\t\ttabs.\r\n"
    cleaned = clean_text(raw)
    assert cleaned == "Header Line\n\nBody with spaces and tabs."


def test_empty_and_missing_text_handling():
    """Verify clean_text gracefully handles None, empty string, and whitespace-only."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \t  \r\n  ") == ""
    assert clean_text(12345) == "12345"


def test_url_and_numeric_token_preservation():
    """Verify clean_text preserves URLs, domain paths, query strings, and monetary figures."""
    raw = "Visit https://secure.bank.com/login?id=884&user=test and claim $5,000.00 cash."
    cleaned = clean_text(raw)
    assert "https://secure.bank.com/login?id=884&user=test" in cleaned
    assert "$5,000.00" in cleaned


def test_punctuation_preservation():
    """Verify exclamation marks, question marks, and symbols are preserved for spam feature detection."""
    raw = "Congratulations!!! Are you ready??? Win 100% FREE cash now!!!"
    cleaned = clean_text(raw)
    assert "Congratulations!!!" in cleaned
    assert "ready???" in cleaned
    assert "100% FREE" in cleaned


# =====================================================================
# Task 7 Tests: Stratified Splitting & Leakage Prevention
# =====================================================================

def test_stratified_split_proportions(synthetic_split_df: pd.DataFrame):
    """Verify 70/15/15 ratio and preservation of spam/ham class proportions."""
    train, val, test = split_dataset(synthetic_split_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    total = len(synthetic_split_df)

    # 20 records -> Train: 14 (70%), Val: 3 (15%), Test: 3 (15%)
    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3

    # Ham: 14 total -> Train: 10, Val: 2, Test: 2
    assert (train["label"] == 0).sum() == 10
    assert (val["label"] == 0).sum() == 2
    assert (test["label"] == 0).sum() == 2

    # Spam: 6 total -> Train: 4, Val: 1, Test: 1
    assert (train["label"] == 1).sum() == 4
    assert (val["label"] == 1).sum() == 1
    assert (test["label"] == 1).sum() == 1


def test_split_reproducibility(synthetic_split_df: pd.DataFrame):
    """Verify identical random_state produces identical partitioned DataFrames."""
    t1, v1, s1 = split_dataset(synthetic_split_df, random_state=42)
    t2, v2, s2 = split_dataset(synthetic_split_df, random_state=42)

    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(s1, s2)


def test_message_id_disjointness(synthetic_split_df: pd.DataFrame):
    """Verify zero overlap in message IDs across train, validation, and test partitions."""
    train, val, test = split_dataset(synthetic_split_df, random_state=42)

    train_ids = set(train["message_id"])
    val_ids = set(val["message_id"])
    test_ids = set(test["message_id"])

    assert len(train_ids & val_ids) == 0, "Shared message IDs between Train and Validation"
    assert len(train_ids & test_ids) == 0, "Shared message IDs between Train and Test"
    assert len(val_ids & test_ids) == 0, "Shared message IDs between Validation and Test"


def test_identical_content_overlap_rejection():
    """Verify group-aware splitting assigns identical normalized texts to the SAME split."""
    # Create dataset where two different message IDs share identical content
    rows = [
        {"message_id": "m1", "text": "Identical newsletter text content", "label": 0, "source_group": "easy_ham"},
        {"message_id": "m2", "text": "Identical newsletter text content", "label": 0, "source_group": "easy_ham_2"},
        {"message_id": "m3", "text": "Unique email A", "label": 0, "source_group": "easy_ham"},
        {"message_id": "m4", "text": "Unique email B", "label": 0, "source_group": "easy_ham"},
        {"message_id": "m5", "text": "Spam email 1", "label": 1, "source_group": "spam"},
        {"message_id": "m6", "text": "Spam email 2", "label": 1, "source_group": "spam"},
        {"message_id": "m7", "text": "Spam email 3", "label": 1, "source_group": "spam"},
        {"message_id": "m8", "text": "Spam email 4", "label": 1, "source_group": "spam"},
        {"message_id": "m9", "text": "Spam email 5", "label": 1, "source_group": "spam"},
        {"message_id": "m10", "text": "Spam email 6", "label": 1, "source_group": "spam"},
    ]
    df = pd.DataFrame(rows)
    train, val, test = split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    # Content disjointness assertion
    train_texts = set(train["text"])
    val_texts = set(val["text"])
    test_texts = set(test["text"])

    assert len(train_texts & val_texts) == 0
    assert len(train_texts & test_texts) == 0
    assert len(val_texts & test_texts) == 0

    # m1 and m2 must be together in whichever partition they were placed
    target_split = None
    for split_name, s_df in [("train", train), ("val", val), ("test", test)]:
        ids = set(s_df["message_id"])
        if "m1" in ids:
            target_split = split_name
            assert "m2" in ids, f"m1 was in {split_name} but m2 was placed in another partition!"


def test_complete_dataset_coverage(synthetic_split_df: pd.DataFrame):
    """Verify union of split message IDs equals the original dataset records without drops or additions."""
    train, val, test = split_dataset(synthetic_split_df, random_state=42)
    original_ids = set(synthetic_split_df["message_id"])
    split_ids = set(train["message_id"]) | set(val["message_id"]) | set(test["message_id"])

    assert split_ids == original_ids
    assert len(train) + len(val) + len(test) == len(synthetic_split_df)


def test_invalid_label_handling():
    """Verify split_dataset raises ValueError on invalid label values or missing classes."""
    # Invalid label
    df_invalid = pd.DataFrame({
        "message_id": ["1", "2"],
        "text": ["A", "B"],
        "label": [0, 99],
        "source_group": ["g", "g"],
    })
    with pytest.raises(ValueError, match="Invalid label values"):
        split_dataset(df_invalid)

    # Missing class (only ham)
    df_single = pd.DataFrame({
        "message_id": ["1", "2"],
        "text": ["A", "B"],
        "label": [0, 0],
        "source_group": ["g", "g"],
    })
    with pytest.raises(ValueError, match="Both ham .* and spam .* must be present"):
        split_dataset(df_single)


def test_output_schema_validation(synthetic_split_df: pd.DataFrame):
    """Verify generated partitions adhere strictly to the target column schema."""
    train, val, test = split_dataset(synthetic_split_df, random_state=42)
    for part in [train, val, test]:
        assert list(part.columns) == SPLIT_COLUMNS
        assert part["text"].isnull().sum() == 0
        assert part["label"].isnull().sum() == 0
        assert set(part["label"].unique()).issubset({0, 1})


# =====================================================================
# Task 7 Tests: TF-IDF Vectorizer Factory & Leakage Demonstration
# =====================================================================

def test_tfidf_vectorizer_configuration():
    """Verify build_tfidf_vectorizer initializes an unfitted vectorizer with expected hyperparameters."""
    vec = build_tfidf_vectorizer()
    assert vec.ngram_range == TFIDF_NGRAM_RANGE
    assert vec.max_features == TFIDF_MAX_FEATURES
    assert vec.analyzer == "word"
    assert vec.lowercase is True
    assert vec.sublinear_tf is True
    assert vec.norm == "l2"
    # Ensure vectorizer is NOT fitted (does not have vocabulary_ attribute)
    assert not hasattr(vec, "vocabulary_")


def test_tfidf_training_only_vocabulary_isolation():
    """Verify strictly fitting on training data prevents vocabulary leakage from validation text."""
    train_corpus = [
        "python natural language processing pipeline",
        "multinomial naive bayes statistical classifier",
    ]
    val_corpus = [
        "quantum zero day exploit cryptography payload",
    ]

    # Initialize unfitted vectorizer
    vec = build_tfidf_vectorizer(min_df=1)

    # Fit strictly on train_corpus
    vec.fit(train_corpus)

    # Inspect learned vocabulary
    learned_vocab = set(vec.vocabulary_.keys())

    # Verify training tokens exist in learned vocabulary
    assert "python" in learned_vocab
    assert "classifier" in learned_vocab

    # Critical Leakage Test: Tokens appearing exclusively in validation must NOT exist in vocabulary!
    assert "quantum" not in learned_vocab
    assert "cryptography" not in learned_vocab
    assert "exploit" not in learned_vocab

    # Transform validation data using training vocabulary
    val_matrix = vec.transform(val_corpus)
    # Since none of the validation terms appeared in train, transformed representation must be all zeros
    assert val_matrix.nnz == 0
