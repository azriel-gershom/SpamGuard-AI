"""Unit and integration tests for SpamGuard AI Phase 03 Exploratory Data Analysis."""

from pathlib import Path

import pandas as pd
import pytest

from src.analyze_data import (
    build_eda_summary,
    compute_class_distribution,
    compute_length_and_word_stats,
    compute_pattern_statistics,
    compute_series_stats,
    extract_top_terms,
    generate_all_figures,
    validate_dataset,
)


@pytest.fixture
def synthetic_email_df() -> pd.DataFrame:
    """Creates a small, deterministic synthetic dataset fixture for testing EDA functions."""
    data = [
        {
            "message_id": "test_ham:001",
            "subject": "Team Sync Meeting",
            "body": "Hi team, please find the agenda for tomorrow's meeting at http://example.com/sync. Regards!",
            "text": "Team Sync Meeting\n\nHi team, please find the agenda for tomorrow's meeting at http://example.com/sync. Regards!",
            "label": 0,
            "source_group": "easy_ham",
            "source_archive": "easy_ham.tar.bz2",
        },
        {
            "message_id": "test_ham:002",
            "subject": "Code Review Notification",
            "body": "PR #42 has been reviewed and merged into main.",
            "text": "Code Review Notification\n\nPR #42 has been reviewed and merged into main.",
            "label": 0,
            "source_group": "easy_ham",
            "source_archive": "easy_ham.tar.bz2",
        },
        {
            "message_id": "test_spam:001",
            "subject": "Claim Free Cash Now!!!",
            "body": "Congratulations! You won $10,000 cash! Click http://spam.xyz now to claim your free reward!!!",
            "text": "Claim Free Cash Now!!!\n\nCongratulations! You won $10,000 cash! Click http://spam.xyz now to claim your free reward!!!",
            "label": 1,
            "source_group": "spam",
            "source_archive": "spam.tar.bz2",
        },
    ]
    return pd.DataFrame(data)


def test_class_percentage_calculation(synthetic_email_df: pd.DataFrame):
    """Verify class counts and percentage calculations."""
    dist = compute_class_distribution(synthetic_email_df)
    assert dist["total"] == 3
    assert dist["ham_count"] == 2
    assert dist["spam_count"] == 1
    assert dist["ham_percentage"] == 66.67
    assert dist["spam_percentage"] == 33.33
    assert dist["imbalance_ratio"] == 2.0


def test_email_length_statistics():
    """Verify character length and word count calculations."""
    series = pd.Series([100, 200, 300, 400, 500])
    stats = compute_series_stats(series)
    assert stats["min"] == 100.0
    assert stats["max"] == 500.0
    assert stats["mean"] == 300.0
    assert stats["median"] == 300.0
    assert stats["q25"] == 200.0
    assert stats["q75"] == 400.0


def test_token_frequency_calculation():
    """Verify extraction of frequent terms excluding English stop-words."""
    texts = [
        "Special prize win cash money prize",
        "Prize money free money win",
    ]
    top_terms = extract_top_terms(texts, top_n=3)
    term_dict = {t["term"]: t["count"] for t in top_terms}
    assert term_dict["prize"] == 3
    assert term_dict["money"] == 3
    assert term_dict["win"] == 2
    # Ensure stop-words like 'the', 'is' would be omitted
    assert "special" in [t["term"] for t in extract_top_terms(texts, top_n=5)]


def test_url_and_pattern_identification(synthetic_email_df: pd.DataFrame):
    """Verify detection of URLs, digits, and exclamation marks."""
    patterns = compute_pattern_statistics(synthetic_email_df)
    assert "ham" in patterns and "spam" in patterns

    # Ham has 1 URL out of 2 emails (50%)
    assert patterns["ham"]["has_url_percentage"] == 50.0
    # Ham has digits in PR #42 (50%)
    assert patterns["ham"]["has_digit_percentage"] == 50.0
    # Ham has 1 exclamation mark in email 1
    assert patterns["ham"]["has_exclamation_count"] == 1

    # Spam has 1 URL out of 1 email (100%)
    assert patterns["spam"]["has_url_percentage"] == 100.0
    # Spam has exclamation marks (100%)
    assert patterns["spam"]["has_exclamation_percentage"] == 100.0
    assert patterns["spam"]["exclamation_count_mean"] > 1.0


def test_empty_dataset_handling():
    """Verify that empty DataFrame inputs are caught and raise ValueError."""
    empty_df = pd.DataFrame(columns=["message_id", "subject", "body", "text", "label", "source_group", "source_archive"])
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(empty_df)


def test_invalid_label_rejection():
    """Verify validation catches invalid labels or missing classes."""
    # Invalid label value (e.g. 99)
    df_invalid = pd.DataFrame({
        "message_id": ["1", "2"],
        "subject": ["A", "B"],
        "body": ["A", "B"],
        "text": ["A", "B"],
        "label": [0, 99],
        "source_group": ["g", "g"],
        "source_archive": ["a", "a"],
    })
    with pytest.raises(ValueError, match="Invalid labels found"):
        validate_dataset(df_invalid)

    # Missing class (only ham)
    df_single_class = pd.DataFrame({
        "message_id": ["1", "2"],
        "subject": ["A", "B"],
        "body": ["A", "B"],
        "text": ["A", "B"],
        "label": [0, 0],
        "source_group": ["g", "g"],
        "source_archive": ["a", "a"],
    })
    with pytest.raises(ValueError, match="Both ham .* and spam .* must be present"):
        validate_dataset(df_single_class)


def test_generated_summary_structure(synthetic_email_df: pd.DataFrame):
    """Verify complete dictionary structure produced by build_eda_summary."""
    summary = build_eda_summary(synthetic_email_df)
    assert "dataset_validation" in summary
    assert "class_distribution" in summary
    assert "length_statistics" in summary
    assert "lexical_patterns" in summary
    assert "top_frequent_terms" in summary

    assert summary["dataset_validation"]["total_records"] == 3
    assert summary["class_distribution"]["spam_count"] == 1
    assert summary["class_distribution"]["ham_count"] == 2


def test_visualization_file_creation(synthetic_email_df: pd.DataFrame, tmp_path: Path):
    """Verify generate_all_figures generates all 6 expected PNG files."""
    test_fig_dir = tmp_path / "figures"
    fig_paths = generate_all_figures(synthetic_email_df, figures_dir=test_fig_dir)

    expected_files = [
        "class_distribution",
        "email_length_distribution",
        "length_comparison",
        "top_spam_words",
        "top_ham_words",
        "url_pattern_comparison",
    ]

    for key in expected_files:
        assert key in fig_paths, f"Missing figure key {key}"
        path = fig_paths[key]
        assert path.exists(), f"Figure file does not exist: {path}"
        assert path.stat().st_size > 0, f"Figure file is empty: {path}"


def test_deterministic_results(synthetic_email_df: pd.DataFrame):
    """Verify running summary computation repeatedly yields identical results."""
    summary1 = build_eda_summary(synthetic_email_df)
    summary2 = build_eda_summary(synthetic_email_df)

    assert summary1["class_distribution"] == summary2["class_distribution"]
    assert summary1["length_statistics"] == summary2["length_statistics"]
    assert summary1["lexical_patterns"] == summary2["lexical_patterns"]
    assert summary1["top_frequent_terms"] == summary2["top_frequent_terms"]
