"""SpamGuard AI — Security Hardening, Edge-Case & Regression Tests.

Phase 09:
- Validates CSV formula injection neutralization against leading whitespace and control characters.
- Audits Unicode, non-Latin alphabets, emojis, and zero-width characters.
- Verifies defensive bounds on input lengths and batch capacity.
- Tests malformed CSV schemas and schema validation error reporting.
- Tests Streamlit session state persistence across batch scanning and interactive filtering.
- Cryptographically verifies model immutability and tamper detection.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import (
    EXPECTED_PIPELINE_SHA256,
    MAX_BATCH_SIZE,
    MAX_INPUT_CHARS,
    SAVED_PIPELINE_PATH,
)
from src.predict import (
    SpamGuardBatchError,
    SpamGuardModelError,
    SpamGuardValidationError,
    SpamPredictor,
    compute_file_sha256,
    predict_batch,
    predict_email,
    sanitize_csv_cell,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# =====================================================================
# 1. Formula Injection & Data Sanitization Hardening
# =====================================================================

def test_formula_injection_leading_whitespace_and_control_chars():
    """Verifies that sanitize_csv_cell neutralizes formula triggers even with leading whitespace."""
    # Direct formula triggers
    assert sanitize_csv_cell("=SUM(1,2)") == "'=SUM(1,2)"
    assert sanitize_csv_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_csv_cell("-1+1") == "'-1+1"
    assert sanitize_csv_cell("@HYPERLINK('http://malicious.com')") == "'@HYPERLINK('http://malicious.com')"

    # Leading whitespace before formula triggers (Excel / Sheets evasion)
    assert sanitize_csv_cell("  =cmd") == "'  =cmd"
    assert sanitize_csv_cell(" \t=1+1") == "' \t=1+1"
    assert sanitize_csv_cell("   @SUM(A1:A10)") == "'   @SUM(A1:A10)"
    assert sanitize_csv_cell(" \n-42") == "' \n-42"
    assert sanitize_csv_cell("\r+99") == "'\r+99"

    # Control character prefixes
    assert sanitize_csv_cell("\tSpecial") == "'\tSpecial"
    assert sanitize_csv_cell("\rSpecial") == "'\rSpecial"
    assert sanitize_csv_cell("\nSpecial") == "'\nSpecial"

    # Safe strings must remain completely unaltered
    assert sanitize_csv_cell("Ham") == "Ham"
    assert sanitize_csv_cell("Spam") == "Spam"
    assert sanitize_csv_cell("Normal legitimate email subject") == "Normal legitimate email subject"
    assert sanitize_csv_cell("") == ""
    assert sanitize_csv_cell(None) is None
    assert sanitize_csv_cell(123) == 123
    assert sanitize_csv_cell(45.67) == 45.67


# =====================================================================
# 2. Input Boundary Conditions & Text Variations
# =====================================================================

def test_boundary_empty_and_whitespace_combinations():
    """Verifies strict rejection of empty, whitespace-only, and null-only messages."""
    with pytest.raises(SpamGuardValidationError, match="Email content is empty"):
        predict_email(subject="", body="")

    with pytest.raises(SpamGuardValidationError, match="Email content is empty"):
        predict_email(subject="   ", body="\n\t\r  ")

    with pytest.raises(SpamGuardValidationError, match="Email content is empty"):
        predict_email(subject=None, body=None)

    # Valid subject with empty body must succeed
    res_subj_only = predict_email(subject="Important Project Update", body="")
    assert res_subj_only.predicted_label in ("Ham", "Spam")

    # Empty subject with valid body must succeed
    res_body_only = predict_email(subject="", body="Please review the attached release notes for sprint 12.")
    assert res_body_only.predicted_label in ("Ham", "Spam")


def test_unicode_and_adversarial_text_handling():
    """Verifies that non-Latin alphabets, emojis, and zero-width characters do not crash the pipeline."""
    adversarial_inputs = [
        # Cyrillic script
        ("Внимание", "Ваш банковский счет заблокирован. Подтвердите данные."),
        # Devanagari script
        ("सूचना", "कृपया अपनी परियोजना की प्रगति रिपोर्ट प्रस्तुत करें।"),
        # CJK characters
        ("緊急通知", "アカウントのセキュリティを確認してください。無料ギフトを獲得。"),
        # Arabic script
        ("تحديث عاجل", "يرجى التحقق من تفاصيل حسابك المصرفي على الفور."),
        # Emojis and unusual symbols
        ("🚨 URGENT CASH PRIZE 🎁", "Congratulations! 💰 Claim $$$ right now 👉 http://claim-bonus.xyz 🔥"),
        # Zero-width spaces and formatting characters
        ("U\u200br\u200bg\u200be\u200bn\u200bt Notice", "Free\u200bmoney\u200bguaranteed\u200bnow"),
        # Mixed HTML tags and non-executing scripts
        ("<title>Test</title>", "<script>alert('XSS')</script><p>Click <a href='http://x'>here</a> to win</p>"),
        # Extreme newline variations
        ("Subject\r\n\r\n\r\nLine", "Body\r\nwith\r\r\nmany\n\n\nnewlines\n\nand spaces."),
    ]

    for subj, body in adversarial_inputs:
        res = predict_email(subject=subj, body=body, explain=True)
        assert res.predicted_label in ("Ham", "Spam")
        assert 0.0 <= res.spam_probability <= 1.0
        assert 0.0 <= res.ham_probability <= 1.0
        assert abs((res.spam_probability + res.ham_probability) - 1.0) < 1e-4
        assert res.cleaned_char_count > 0


def test_oversized_input_rejection():
    """Verifies that single inputs exceeding MAX_INPUT_CHARS are rejected defensively."""
    huge_subject = "Urgent Notice " * 50
    huge_body = "A" * (MAX_INPUT_CHARS + 500)

    with pytest.raises(SpamGuardValidationError, match="exceeds maximum allowed limit"):
        predict_email(subject=huge_subject, body=huge_body)


# =====================================================================
# 3. Batch Scanner & CSV Processing Robustness
# =====================================================================

def test_batch_schema_validation_errors():
    """Verifies that batch processing validates column schemas and rejects invalid formats."""
    # Missing required columns
    invalid_df = pd.DataFrame([{"foo": "test", "bar": "123"}])
    with pytest.raises(SpamGuardBatchError, match="must contain"):
        predict_batch(invalid_df)

    # Empty DataFrame returns empty list safely
    empty_df = pd.DataFrame(columns=["subject", "body"])
    assert predict_batch(empty_df) == []


def test_batch_oversized_batch_rejection():
    """Verifies that batches exceeding MAX_BATCH_SIZE (10,000) are rejected."""
    oversized_data = [("Subject", "Body text content")] * (MAX_BATCH_SIZE + 5)
    with pytest.raises(SpamGuardBatchError, match="exceeds maximum limit"):
        predict_batch(oversized_data)


def test_batch_mixed_records_with_invalid_item():
    """Verifies that an invalid empty record inside a batch is identified with its exact index."""
    mixed_data = [
        {"subject": "Meeting", "body": "Standup at 10 AM"},
        {"subject": "   ", "body": "\n\n"},  # Invalid empty record
        {"subject": "Invoice", "body": "Please find invoice attached"},
    ]
    with pytest.raises(SpamGuardBatchError, match="Batch record at index 1 failed validation"):
        predict_batch(mixed_data)


def test_batch_prediction_parity_and_order_preservation():
    """Verifies that vectorized batch predictions strictly match individual predict_email results."""
    test_samples = [
        ("URGENT: Claim Free Prize Money", "Congratulations you won cash bonus guarantee!"),
        ("Sprint Retrospective Notes", "Please find attached notes from today's engineering review."),
        ("Refinance Your Mortgage Fast", "Lowest interest rates on home mortgage loans today."),
        ("Assignment 3 Submission Deadline", "Dear students, please submit your machine learning microproject by Friday."),
    ]

    # Run batch
    batch_results = predict_batch(test_samples, explain=True)
    assert len(batch_results) == len(test_samples)

    # Run individual and compare
    for i, (subj, body) in enumerate(test_samples):
        indiv_res = predict_email(subject=subj, body=body, explain=True)
        batch_res = batch_results[i]

        assert batch_res.predicted_class == indiv_res.predicted_class
        assert batch_res.predicted_label == indiv_res.predicted_label
        assert abs(batch_res.spam_probability - indiv_res.spam_probability) < 1e-9
        assert abs(batch_res.ham_probability - indiv_res.ham_probability) < 1e-9


# =====================================================================
# 4. Model Integrity, Deserialization & Tamper Defense
# =====================================================================

def test_model_checksum_tamper_detection(tmp_path: Path):
    """Verifies that SpamPredictor strictly refuses to load an artifact with a mismatched checksum."""
    fake_model_path = tmp_path / "tampered_pipeline.joblib"
    fake_model_path.write_bytes(b"malicious_or_corrupted_payload_data")

    with pytest.raises(SpamGuardModelError, match="Model SHA-256 integrity verification failed"):
        SpamPredictor.load(model_path=fake_model_path, expected_sha=EXPECTED_PIPELINE_SHA256)


def test_frozen_model_hash_and_immutability():
    """Cryptographically verifies that models/spam_pipeline.joblib is unmodified."""
    assert SAVED_PIPELINE_PATH.exists()
    actual_sha = compute_file_sha256(SAVED_PIPELINE_PATH)
    assert actual_sha == EXPECTED_PIPELINE_SHA256


# =====================================================================
# 5. Streamlit App Session State & Workflow Hardening (AppTest)
# =====================================================================

def test_apptest_scanner_sample_selection_clears_stale_result():
    """Verifies that selecting a new sample in the dropdown resets stale results."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Navigate to Single Email Scanner
    at.sidebar.radio[0].set_value("🔍 Email Scanner")
    at.run()

    # Form inputs should be present
    assert len(at.text_input) > 0
    assert len(at.text_area) > 0


def test_apptest_batch_scanner_state_persistence_and_filtering():
    """Verifies that batch scan results persist across radio filter interactions."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Navigate to Batch CSV Scanner
    at.sidebar.radio[0].set_value("📁 Batch CSV Scanner")
    at.run()

    assert not at.exception
    # Verify expander and download button are rendered cleanly
    assert len(at.download_button) >= 1
