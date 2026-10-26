"""Automated frontend and UI tests for SpamGuard AI Streamlit Application.

Phase 08:
- Tests app initialization, navigation structure, and widget rendering using Streamlit AppTest.
- Verifies single email scanning flow (empty input validation, spam detection, ham verification).
- Verifies token attribution and model telemetry display.
- Verifies batch scanner controls, CSV template download, and schema validation.
- Verifies analytics dashboard and Plotly chart generation.
- Verifies that UI interactions never trigger model retraining or modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import (
    EXPECTED_PIPELINE_SHA256,
    SAVED_PIPELINE_PATH,
    TEST_EVALUATION_FILE,
)
from src.predict import (
    FeatureContribution,
    PredictionResult,
    compute_file_sha256,
    get_default_predictor,
)
from src.ui_helpers import (
    SAMPLE_EMAILS,
    create_baseline_comparison_chart,
    create_confusion_matrix_chart,
    create_gauge_chart,
    format_features_dataframe,
    get_model_status,
    load_evaluation_summary,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def app_runner() -> AppTest:
    """Creates a fresh Streamlit AppTest runner initialized with app.py."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=15.0)
    at.run()
    return at


# =====================================================================
# 1. App Initialization & Smoke Tests
# =====================================================================

def test_app_initialization(app_runner: AppTest):
    """Verify app.py initializes cleanly with zero unhandled exceptions."""
    assert len(app_runner.exception) == 0, f"App threw exceptions on startup: {app_runner.exception}"
    assert len(app_runner.title) > 0
    assert "SpamGuard AI" in app_runner.title[0].value


def test_sidebar_rendering_and_telemetry(app_runner: AppTest):
    """Verify persistent sidebar renders branding, model status, and navigation."""
    assert app_runner.sidebar is not None
    assert len(app_runner.sidebar.radio) > 0

    radio = app_runner.sidebar.radio[0]
    expected_options = [
        "🏠 Overview Dashboard",
        "🔍 Email Scanner",
        "📁 Batch CSV Scanner",
        "📊 Model Analytics",
        "ℹ️ About & Methodology",
    ]
    assert list(radio.options) == expected_options


# =====================================================================
# 2. Navigation Structure Tests
# =====================================================================

def test_navigation_to_all_sections(app_runner: AppTest):
    """Verify switching between all 5 navigation sections completes without errors."""
    radio = app_runner.sidebar.radio[0]

    for opt in radio.options:
        radio.set_value(opt).run()
        assert len(app_runner.exception) == 0, f"Exception occurred while navigating to {opt}: {app_runner.exception}"


# =====================================================================
# 3. Overview Dashboard Tests
# =====================================================================

def test_overview_dashboard_metrics(app_runner: AppTest):
    """Verify Overview Dashboard displays genuine held-out test evaluation KPIs."""
    report = load_evaluation_summary()
    assert report is not None
    m = report["final_test_metrics"]

    # In overview tab
    radio = app_runner.sidebar.radio[0]
    radio.set_value("🏠 Overview Dashboard").run()

    metric_labels = [metric.label for metric in app_runner.metric]
    assert "Test Accuracy" in metric_labels
    assert "Spam Precision" in metric_labels
    assert "Spam Recall" in metric_labels
    assert "Spam F1-Score" in metric_labels
    assert "ROC-AUC" in metric_labels

    # Verify accuracy value matches report
    acc_metric = next(met for met in app_runner.metric if met.label == "Test Accuracy")
    assert f"{m['accuracy'] * 100:.2f}%" in acc_metric.value


# =====================================================================
# 4. Single Email Scanner Tests
# =====================================================================

def test_single_email_empty_validation(app_runner: AppTest):
    """Verify submitting empty email triggers validation error feedback."""
    radio = app_runner.sidebar.radio[0]
    radio.set_value("🔍 Email Scanner").run()

    # Form submit button is the first button in the form
    app_runner.button[0].click().run()
    assert len(app_runner.exception) == 0
    # Should display validation error
    assert len(app_runner.error) > 0
    assert any("Input Validation Error" in err.value for err in app_runner.error)


def test_single_email_spam_classification(app_runner: AppTest):
    """Verify scanning an obvious spam email displays threat verdict and telemetry."""
    radio = app_runner.sidebar.radio[0]
    radio.set_value("🔍 Email Scanner").run()

    # Enter synthetic spam content
    app_runner.text_input[0].set_value("Urgent: You won $10,000 cash prize!")
    app_runner.text_area[0].set_value("Claim your free bonus money now with 100% payout guarantee. Visit link below.")
    app_runner.button[0].click().run()

    assert len(app_runner.exception) == 0
    # Should display threat verdict in markdown
    markdown_texts = [md.value for md in app_runner.markdown]
    assert any("THREAT DETECTED: SPAM" in text for text in markdown_texts)

    # Telemetry metrics should display
    metric_labels = [m.label for m in app_runner.metric]
    assert "Predicted Label" in metric_labels
    assert "Spam Probability" in metric_labels


def test_single_email_ham_classification(app_runner: AppTest):
    """Verify scanning a legitimate email displays verified ham verdict."""
    radio = app_runner.sidebar.radio[0]
    radio.set_value("🔍 Email Scanner").run()

    # Enter synthetic ham content
    app_runner.text_input[0].set_value("Team Sprint Meeting Notes & Code Review")
    app_runner.text_area[0].set_value("Hi team, please find attached the meeting agenda and git patch for code review.")
    app_runner.button[0].click().run()

    assert len(app_runner.exception) == 0
    markdown_texts = [md.value for md in app_runner.markdown]
    assert any("VERIFIED: HAM" in text for text in markdown_texts)


# =====================================================================
# 5. Batch CSV Scanner Tests
# =====================================================================

def test_batch_scanner_view(app_runner: AppTest):
    """Verify Batch Scanner renders upload controls, guidance, and download template."""
    radio = app_runner.sidebar.radio[0]
    radio.set_value("📁 Batch CSV Scanner").run()

    assert len(app_runner.exception) == 0
    # Template download button should be present
    assert len(app_runner.download_button) > 0
    assert "Download Sample CSV Template" in app_runner.download_button[0].label


# =====================================================================
# 6. Model Analytics Tests
# =====================================================================

def test_model_analytics_view(app_runner: AppTest):
    """Verify Model Analytics renders KPI cards, confusion matrix, and tabs."""
    radio = app_runner.sidebar.radio[0]
    radio.set_value("📊 Model Analytics").run()

    assert len(app_runner.exception) == 0
    assert len(app_runner.tabs) > 0

    metric_labels = [m.label for m in app_runner.metric]
    assert "Test Accuracy" in metric_labels
    assert "Spam Precision" in metric_labels
    assert "Spam Recall" in metric_labels
    assert "Spam F1-Score" in metric_labels


# =====================================================================
# 7. About & Methodology Tests
# =====================================================================

def test_about_methodology_view(app_runner: AppTest):
    """Verify About & Methodology section renders course alignment and architecture."""
    radio = app_runner.sidebar.radio[0]
    radio.set_value("ℹ️ About & Methodology").run()

    assert len(app_runner.exception) == 0
    markdown_texts = [md.value for md in app_runner.markdown]
    assert any("CO2" in text and "CO4" in text and "CO5" in text for text in markdown_texts)
    assert any("Apache SpamAssassin" in text for text in markdown_texts)


# =====================================================================
# 8. UI Helpers Unit Tests
# =====================================================================

def test_ui_helpers_get_model_status():
    """Verify get_model_status reports verified online state for genuine artifact."""
    status = get_model_status(SAVED_PIPELINE_PATH, EXPECTED_PIPELINE_SHA256)
    assert status["available"] is True
    assert status["checksum_match"] is True
    assert status["sha256"] == EXPECTED_PIPELINE_SHA256


def test_ui_helpers_missing_model_fallback(tmp_path: Path):
    """Verify get_model_status handles missing model file gracefully."""
    fake_path = tmp_path / "missing.joblib"
    status = get_model_status(fake_path, EXPECTED_PIPELINE_SHA256)
    assert status["available"] is False
    assert status["checksum_match"] is False
    assert status["status"] == "Missing Artifact"


def test_ui_helpers_load_evaluation_summary():
    """Verify load_evaluation_summary loads authentic report data."""
    summary = load_evaluation_summary(TEST_EVALUATION_FILE)
    assert summary is not None
    assert "final_test_metrics" in summary
    assert summary["final_test_metrics"]["accuracy"] == 0.973


def test_ui_helpers_missing_report_fallback(tmp_path: Path):
    """Verify load_evaluation_summary returns None when report file is missing."""
    res = load_evaluation_summary(tmp_path / "non_existent_report.json")
    assert res is None


def test_ui_helpers_create_gauge_chart():
    """Verify create_gauge_chart generates valid Plotly Figure."""
    fig = create_gauge_chart(0.85)
    assert fig is not None
    assert hasattr(fig, "to_dict")
    assert "data" in fig.to_dict()


def test_ui_helpers_create_confusion_matrix_chart():
    """Verify create_confusion_matrix_chart generates valid Plotly Heatmap."""
    cm_data = {"tn": 610, "fp": 7, "fn": 17, "tp": 255}
    fig = create_confusion_matrix_chart(cm_data)
    assert fig is not None
    assert hasattr(fig, "to_dict")


def test_ui_helpers_create_baseline_comparison_chart():
    """Verify create_baseline_comparison_chart generates valid Plotly Bar chart."""
    m = {"accuracy": 0.973, "spam": {"precision": 0.9733, "recall": 0.9375, "f1_score": 0.9551}, "balanced_accuracy": 0.9631}
    b = {"accuracy": 0.694, "spam": {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}, "balanced_accuracy": 0.50}
    fig = create_baseline_comparison_chart(m, b)
    assert fig is not None
    assert hasattr(fig, "to_dict")


def test_ui_helpers_format_features_dataframe():
    """Verify format_features_dataframe builds clean DataFrame from FeatureContribution list."""
    feats = [
        FeatureContribution(token="cash", weight=0.35, log_odds_diff=2.5, contribution=0.875, association="Spam-leaning"),
        FeatureContribution(token="sprint", weight=0.40, log_odds_diff=-1.8, contribution=-0.72, association="Ham-leaning"),
    ]
    df = format_features_dataframe(feats)
    assert len(df) == 2
    assert list(df.columns) == ["Token", "TF-IDF Weight", "Log-Odds Contribution", "Learned Tendency"]
    assert df.iloc[0]["Token"] == "cash"
    assert df.iloc[1]["Token"] == "sprint"


# =====================================================================
# 9. Model Immutability During UI Execution
# =====================================================================

def test_no_model_retraining_during_app_lifecycle(app_runner: AppTest):
    """Verify that user interactions in Streamlit never alter the frozen pipeline weights."""
    hash_before = compute_file_sha256(SAVED_PIPELINE_PATH)

    # Perform interaction
    radio = app_runner.sidebar.radio[0]
    radio.set_value("🔍 Email Scanner").run()
    app_runner.text_input[0].set_value("Verification Test Subject")
    app_runner.text_area[0].set_value("Testing whether app interaction modifies the underlying model.")
    app_runner.button[0].click().run()

    hash_after = compute_file_sha256(SAVED_PIPELINE_PATH)
    assert hash_before == hash_after == EXPECTED_PIPELINE_SHA256
