"""Unit and integration tests for SpamGuard AI Phase 07 Prediction Engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    EXPECTED_PIPELINE_SHA256,
    MAX_BATCH_SIZE,
    MAX_INPUT_CHARS,
    SAVED_PIPELINE_PATH,
)
from src.predict import (
    FeatureContribution,
    PredictionResult,
    SpamGuardBatchError,
    SpamGuardModelError,
    SpamGuardValidationError,
    SpamPredictor,
    combine_subject_and_body,
    compute_file_sha256,
    get_default_predictor,
    load_model,
    predict_batch,
    predict_email,
    sanitize_csv_cell,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture(scope="module")
def predictor() -> SpamPredictor:
    """Provides a shared, validated SpamPredictor instance for testing."""
    return SpamPredictor.load(SAVED_PIPELINE_PATH, EXPECTED_PIPELINE_SHA256)


@pytest.fixture
def sample_spam_email() -> tuple[str, str]:
    """Synthetic obvious spam email message."""
    return (
        "Claim your prize now!",
        "Congratulations! You won free cash prize money and a special bonus offer. Guaranteed payout.",
    )


@pytest.fixture
def sample_ham_email() -> tuple[str, str]:
    """Synthetic legitimate communication message."""
    return (
        "Sprint meeting agenda and code review",
        "Hi team, please find attached the meeting notes, sprint progress report, and code patch update.",
    )


# =====================================================================
# 1. Trusted Model Loading
# =====================================================================

def test_trusted_model_loading(predictor: SpamPredictor):
    """Verify SpamPredictor loads the frozen Pipeline with expected structure."""
    assert isinstance(predictor._pipeline, Pipeline)
    assert "tfidf" in predictor._pipeline.named_steps
    assert "classifier" in predictor._pipeline.named_steps
    assert hasattr(predictor._pipeline.named_steps["classifier"], "classes_")
    assert list(predictor._pipeline.classes_) == [0, 1]
    assert len(predictor._vocab) == 20000

    # Test module-level load_model helper
    pipe = load_model(SAVED_PIPELINE_PATH)
    assert isinstance(pipe, Pipeline)


# =====================================================================
# 2. Model Checksum Verification
# =====================================================================

def test_model_checksum_verification():
    """Verify computed checksum matches the expected frozen Phase 05 fingerprint."""
    actual_hash = compute_file_sha256(SAVED_PIPELINE_PATH)
    assert actual_hash.lower() == EXPECTED_PIPELINE_SHA256.lower()


# =====================================================================
# 3. Rejection of Corrupted Artifacts
# =====================================================================

def test_rejection_of_corrupted_artifacts(tmp_path: Path):
    """Verify loading fails immediately when file hash does not match expected SHA."""
    corrupted_file = tmp_path / "corrupted_pipeline.joblib"
    corrupted_file.write_bytes(b"tampered_bytes_not_real_model")

    with pytest.raises(SpamGuardModelError, match="integrity verification failed"):
        SpamPredictor.load(corrupted_file, expected_sha=EXPECTED_PIPELINE_SHA256)


# =====================================================================
# 4. Valid Spam Predictions
# =====================================================================

def test_valid_spam_predictions(predictor: SpamPredictor, sample_spam_email: tuple[str, str]):
    """Verify synthetic spam email is classified as Spam (Class 1) with high probability."""
    subj, body = sample_spam_email
    res = predictor.predict_email(subj, body)

    assert isinstance(res, PredictionResult)
    assert res.predicted_class == 1
    assert res.predicted_label == "Spam"
    assert res.is_spam is True
    assert res.spam_probability > 0.50
    assert res.ham_probability < 0.50
    assert res.cleaned_char_count > 0
    assert res.processing_time_ms > 0.0


# =====================================================================
# 5. Valid Ham Predictions
# =====================================================================

def test_valid_ham_predictions(predictor: SpamPredictor, sample_ham_email: tuple[str, str]):
    """Verify synthetic legitimate email is classified as Ham (Class 0)."""
    subj, body = sample_ham_email
    res = predictor.predict_email(subj, body)

    assert isinstance(res, PredictionResult)
    assert res.predicted_class == 0
    assert res.predicted_label == "Ham"
    assert res.is_spam is False
    assert res.ham_probability > 0.50
    assert res.spam_probability < 0.50


# =====================================================================
# 6. Correct Class-Label Mapping
# =====================================================================

def test_correct_class_label_mapping(predictor: SpamPredictor):
    """Verify class 0 strictly maps to 'Ham' and class 1 strictly maps to 'Spam'."""
    res_spam = predictor.predict_email(body="free money claim cash prize now guarantee")
    assert res_spam.predicted_class == 1 and res_spam.predicted_label == "Spam"

    res_ham = predictor.predict_email(body="team project meeting agenda code review update")
    assert res_ham.predicted_class == 0 and res_ham.predicted_label == "Ham"


# =====================================================================
# 7. Probability Column Ordering
# =====================================================================

def test_probability_column_ordering(predictor: SpamPredictor):
    """Verify that ham_probability and spam_probability index by class label, not position."""
    assert predictor._ham_idx == predictor._class_to_idx[0]
    assert predictor._spam_idx == predictor._class_to_idx[1]
    assert {predictor._ham_idx, predictor._spam_idx} == {0, 1}


# =====================================================================
# 8. Probability Range and Sum Checks
# =====================================================================

def test_probability_range_and_sum_checks(predictor: SpamPredictor):
    """Verify probabilities are strictly bounded in [0, 1] and sum to 1.0."""
    test_texts = [
        "quick meeting schedule",
        "refinance your mortgage home loan cash offer",
        "linux kernel developer patch commit",
        "special discount clearance sale buy today",
    ]
    for text in test_texts:
        res = predictor.predict_email(body=text)
        assert 0.0 <= res.ham_probability <= 1.0
        assert 0.0 <= res.spam_probability <= 1.0
        assert abs((res.ham_probability + res.spam_probability) - 1.0) < 1e-4


# =====================================================================
# 9. Empty Input Rejection
# =====================================================================

def test_empty_input_rejection(predictor: SpamPredictor):
    """Verify empty or whitespace-only inputs raise SpamGuardValidationError."""
    with pytest.raises(SpamGuardValidationError, match="empty or contains only whitespace"):
        predictor.predict_email(subject="", body="")

    with pytest.raises(SpamGuardValidationError, match="empty or contains only whitespace"):
        predictor.predict_email(subject="   \t   ", body="   \n   ")

    with pytest.raises(SpamGuardValidationError, match="empty or contains only whitespace"):
        predictor.predict_email(subject=None, body=None)


# =====================================================================
# 10. Unicode Handling
# =====================================================================

def test_unicode_handling(predictor: SpamPredictor):
    """Verify Unicode characters, accents, non-Latin text, and emojis are handled safely."""
    unicode_msg = "Wïn frëë cäsh 💰 100% gúárántëë! Sonderangebot für Sie."
    res = predictor.predict_email(subject="Glückwunsch 🎉", body=unicode_msg)
    assert isinstance(res, PredictionResult)
    assert res.predicted_label in ("Ham", "Spam")


# =====================================================================
# 11. Text Normalization Consistency
# =====================================================================

def test_text_normalization_consistency(predictor: SpamPredictor):
    """Verify text preparation matches Phase 02 combination and Phase 04 cleaning."""
    subj = "  Urgent: Claim Prize  "
    body = "  Click here for free money.\r\n\r\nSecond line.  "

    combined = combine_subject_and_body(subj, body)
    assert combined == "Urgent: Claim Prize\n\nClick here for free money.\r\n\r\nSecond line."

    prepared = predictor.prepare_input_text(subj, body)
    assert "\r" not in prepared  # CRLF converted to LF
    assert "Urgent: Claim Prize\n\nClick here for free money.\n\nSecond line." in prepared


# =====================================================================
# 12. Individual and Batch Prediction Parity
# =====================================================================

def test_individual_and_batch_prediction_parity(
    predictor: SpamPredictor,
    sample_spam_email: tuple[str, str],
    sample_ham_email: tuple[str, str],
):
    """Verify predict_batch produces identical results to individual predict_email calls."""
    items = [
        {"subject": sample_spam_email[0], "body": sample_spam_email[1]},
        {"subject": sample_ham_email[0], "body": sample_ham_email[1]},
        {"subject": "Reminder", "body": "Please review project documentation before sprint ends."},
    ]

    batch_res = predictor.predict_batch(items)
    indiv_res = [predictor.predict_email(item["subject"], item["body"]) for item in items]

    assert len(batch_res) == len(indiv_res)
    for b_item, i_item in zip(batch_res, indiv_res):
        assert b_item.predicted_class == i_item.predicted_class
        assert b_item.predicted_label == i_item.predicted_label
        assert abs(b_item.spam_probability - i_item.spam_probability) < 1e-4
        assert abs(b_item.ham_probability - i_item.ham_probability) < 1e-4


# =====================================================================
# 13. Batch Output Ordering
# =====================================================================

def test_batch_output_ordering(predictor: SpamPredictor):
    """Verify batch output order strictly mirrors the input sequence."""
    inputs = [
        {"subject": "Email 1", "body": "cash prize free offer"},      # likely spam
        {"subject": "Email 2", "body": "team meeting project status"}, # likely ham
        {"subject": "Email 3", "body": "credit card debt refinance"},  # likely spam
        {"subject": "Email 4", "body": "linux kernel bug patch"},      # likely ham
    ]

    results = predictor.predict_batch(inputs)
    assert len(results) == 4
    # Check that alternating inputs result in expected classifications
    assert results[0].is_spam is True
    assert results[1].is_spam is False
    assert results[2].is_spam is True
    assert results[3].is_spam is False


# =====================================================================
# 14. CSV Input Validation & Formula Injection Sanitization
# =====================================================================

def test_csv_batch_and_formula_injection(tmp_path: Path, predictor: SpamPredictor):
    """Verify predict_csv handles DataFrames and escapes formula injection payloads."""
    # Test formula injection cell
    malicious_label = "=cmd|' /C calc'!A0"
    sanitized = sanitize_csv_cell(malicious_label)
    assert sanitized.startswith("'")

    safe_text = "Normal Text"
    assert sanitize_csv_cell(safe_text) == "Normal Text"

    # Create CSV file with batch items
    csv_file = tmp_path / "test_emails.csv"
    out_csv = tmp_path / "out_emails.csv"
    df = pd.DataFrame([
        {"subject": "Offer", "body": "claim your free cash prize"},
        {"subject": "Update", "body": "weekly sprint meeting code review notes"},
    ])
    df.to_csv(csv_file, index=False)

    out_df = predictor.predict_csv(csv_file, out_csv)
    assert len(out_df) == 2
    assert "row_id" in out_df.columns
    assert "predicted_class" in out_df.columns
    assert "spam_probability" in out_df.columns
    assert out_csv.exists()


def test_csv_missing_columns(tmp_path: Path, predictor: SpamPredictor):
    """Verify predict_csv rejects CSV without recognized subject/body or text columns."""
    bad_csv = tmp_path / "bad_columns.csv"
    pd.DataFrame({"invalid_col": ["hello", "world"]}).to_csv(bad_csv, index=False)

    with pytest.raises(SpamGuardBatchError, match="must contain"):
        predictor.predict_csv(bad_csv)


# =====================================================================
# 15. Oversized Input Handling
# =====================================================================

def test_oversized_input_handling(predictor: SpamPredictor):
    """Verify text exceeding MAX_INPUT_CHARS is rejected with validation error."""
    oversized_body = "word " * (MAX_INPUT_CHARS // 4 + 10)
    assert len(oversized_body) > MAX_INPUT_CHARS

    with pytest.raises(SpamGuardValidationError, match="exceeds maximum allowed limit"):
        predictor.predict_email(subject="Big", body=oversized_body)


# =====================================================================
# 16. Missing Artifact Errors
# =====================================================================

def test_missing_artifact_errors():
    """Verify SpamPredictor.load raises SpamGuardModelError for non-existent path."""
    with pytest.raises(SpamGuardModelError, match="not found"):
        SpamPredictor.load(Path("non_existent_model_dir/model.joblib"))


# =====================================================================
# 17. Inference Without Fitting or Retraining
# =====================================================================

def test_inference_without_fitting_or_retraining(predictor: SpamPredictor):
    """Verify that executing predictions does not alter model parameters."""
    nb = predictor._clf_step
    initial_log_prob = nb.feature_log_prob_.copy()
    initial_classes = nb.classes_.copy()

    # Run diverse inference
    _ = predictor.predict_email("Hello", "This is an important meeting update.")
    _ = predictor.predict_email("Win Cash", "Claim your free money prize today!")

    # Check identical weights
    np.testing.assert_array_equal(nb.feature_log_prob_, initial_log_prob)
    np.testing.assert_array_equal(nb.classes_, initial_classes)


# =====================================================================
# 18. Deterministic Predictions
# =====================================================================

def test_deterministic_predictions(predictor: SpamPredictor):
    """Verify identical inputs yield exact same predictions across repeated calls."""
    subj = "Test Subject Verification"
    body = "Reviewing the automated pipeline outputs for deterministic behavior."

    res1 = predictor.predict_email(subj, body)
    res2 = predictor.predict_email(subj, body)

    assert res1.predicted_class == res2.predicted_class
    assert res1.spam_probability == res2.spam_probability
    assert res1.ham_probability == res2.ham_probability


# =====================================================================
# 19. Feature Contribution Explainability
# =====================================================================

def test_feature_explanations(predictor: SpamPredictor):
    """Verify explain=True extracts valid FeatureContribution objects."""
    res = predictor.predict_email(
        subject="Mortgage Refinance Offer",
        body="Lower your mortgage interest rate today. Exclusive free cash savings.",
        explain=True,
        top_k=5,
    )

    assert len(res.top_features) > 0
    assert len(res.top_features) <= 5

    for feat in res.top_features:
        assert isinstance(feat, FeatureContribution)
        assert feat.token in predictor._vocab
        assert feat.weight > 0.0
        assert feat.association in ("Spam-leaning", "Ham-leaning")
        assert isinstance(feat.to_dict(), dict)

    # Top token for this email should include 'mortgage' or 'refinance'
    tokens = [f.token for f in res.top_features]
    assert any("mortgage" in t or "refinance" in t for t in tokens)


# =====================================================================
# 20. Module-Level Convenience Functions & Singleton
# =====================================================================

def test_module_level_helpers():
    """Verify predict_email and predict_batch convenience functions."""
    res = predict_email(subject="Hello", body="Meeting at 2pm today in the main boardroom.")
    assert isinstance(res, PredictionResult)
    assert res.predicted_label in ("Ham", "Spam")

    batch = predict_batch([
        {"subject": "Update", "body": "Project sprint report"},
        {"subject": "Free", "body": "Claim free cash prize now"},
    ])
    assert len(batch) == 2

    # Check singleton instance caching
    p1 = get_default_predictor()
    p2 = get_default_predictor()
    assert p1 is p2
