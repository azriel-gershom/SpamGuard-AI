"""Unit and integration tests for SpamGuard AI Phase 06 Final Model Evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import (
    SAVED_PIPELINE_PATH,
    TEST_SPLIT_FILE,
)
from src.evaluate import (
    EXPECTED_MODEL_SHA256,
    analyze_misclassifications,
    build_evaluation_report,
    compute_comprehensive_metrics,
    compute_file_sha256,
    evaluate_majority_baseline,
    generate_all_visualizations,
    load_test_data,
    load_trusted_pipeline,
    plot_classification_metrics,
    plot_confusion_matrix,
    plot_model_vs_baseline,
    plot_precision_recall_curve,
    plot_roc_curve,
    write_markdown_findings,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_eval_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates deterministic ground truth, predictions, and probabilities.

    Orientation:
    - 60 actual Ham (0): 55 predicted Ham (TN), 5 predicted Spam (FP)
    - 40 actual Spam (1): 8 predicted Ham (FN), 32 predicted Spam (TP)
    """
    # 55 TN (0, 0), 5 FP (0, 1), 8 FN (1, 0), 32 TP (1, 1)
    y_true = np.array([0] * 60 + [1] * 40)
    y_pred = np.array([0] * 55 + [1] * 5 + [0] * 8 + [1] * 32)
    # Calibrated dummy probabilities
    y_proba = np.array([0.1] * 55 + [0.8] * 5 + [0.3] * 8 + [0.9] * 32)
    return y_true, y_pred, y_proba


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    """Creates a deterministic DataFrame representing test emails."""
    return pd.DataFrame({
        "message_id": ["m1", "m2", "m3", "m4"],
        "text": ["normal meeting text", "claim free prize now", "invoice attachment", "urgent verify account"],
        "label": [0, 1, 0, 1],
        "source_group": ["easy_ham", "spam", "easy_ham", "spam"],
    })


# =====================================================================
# Tests: Checksum & Pipeline Loading
# =====================================================================

def test_compute_file_sha256_valid(tmp_path: Path):
    """Verify SHA-256 computation matches known test vector."""
    import hashlib
    content = b"SpamGuard AI Verification"
    test_file = tmp_path / "sample.txt"
    test_file.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(test_file) == expected_hash


def test_compute_file_sha256_missing_file():
    """Verify compute_file_sha256 raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(Path("non_existent_file_xyz.bin"))


def test_load_trusted_pipeline_real_artifact():
    """Verify the Phase 05 serialized pipeline loads and passes integrity checks."""
    pipeline = load_trusted_pipeline(SAVED_PIPELINE_PATH, EXPECTED_MODEL_SHA256)
    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert hasattr(pipeline.named_steps["classifier"], "classes_")


def test_load_trusted_pipeline_checksum_mismatch(tmp_path: Path):
    """Verify load_trusted_pipeline rejects tampered model file."""
    fake_model = tmp_path / "tampered.joblib"
    fake_model.write_bytes(b"corrupted_binary_data")
    with pytest.raises(ValueError, match="Model SHA-256 mismatch"):
        load_trusted_pipeline(fake_model, expected_sha=EXPECTED_MODEL_SHA256)


# =====================================================================
# Tests: Test Data Loading
# =====================================================================

def test_load_test_data_success():
    """Verify load_test_data loads the held-out partition correctly."""
    X_test, y_test, df = load_test_data(TEST_SPLIT_FILE)
    assert len(df) == 889
    assert len(X_test) == 889
    assert len(y_test) == 889
    assert set(df["label"].unique()) == {0, 1}
    assert (y_test == 0).sum() == 617
    assert (y_test == 1).sum() == 272


def test_load_test_data_missing_column(tmp_path: Path):
    """Verify load_test_data validates schema columns."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["hello"], "label": [0]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_test_data(bad_csv)


def test_load_test_data_invalid_label(tmp_path: Path):
    """Verify load_test_data rejects labels other than 0 and 1."""
    bad_csv = tmp_path / "bad_label.csv"
    pd.DataFrame({
        "message_id": ["1"],
        "text": ["hello"],
        "label": [2],
    }).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Invalid labels"):
        load_test_data(bad_csv)


# =====================================================================
# Tests: Metric Computation & Confusion Matrix Orientation
# =====================================================================

def test_confusion_matrix_orientation(synthetic_eval_data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Verify confusion matrix orientation is strictly [[TN, FP], [FN, TP]]."""
    y_true, y_pred, y_proba = synthetic_eval_data
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba)
    cm = metrics["confusion_matrix"]

    assert cm["tn"] == 55
    assert cm["fp"] == 5
    assert cm["fn"] == 8
    assert cm["tp"] == 32
    assert cm["matrix"] == [[55, 5], [8, 32]]


def test_metric_calculation_formulas(synthetic_eval_data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Verify precision, recall, F1, specificity, FPR, and FNR exact calculations."""
    y_true, y_pred, y_proba = synthetic_eval_data
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba)

    # Total = 100
    assert metrics["accuracy"] == 0.8700  # (55 + 32) / 100 = 0.87
    # Spam (Class 1): TP=32, FP=5 -> Precision = 32/37 = 0.8649
    assert metrics["spam"]["precision"] == 0.8649
    # Spam (Class 1): TP=32, FN=8 -> Recall = 32/40 = 0.8000
    assert metrics["spam"]["recall"] == 0.8000
    # Ham (Class 0): TN=55, FN=8 -> Precision = 55/63 = 0.8730
    assert metrics["ham"]["precision"] == 0.8730
    # Ham (Class 0): TN=55, FP=5 -> Recall = 55/60 = 0.9167
    assert metrics["ham"]["recall"] == 0.9167
    # Specificity = TN / (TN + FP) = 55 / 60 = 0.9167
    assert metrics["specificity"] == 0.9167
    # FPR = FP / (TN + FP) = 5 / 60 = 0.0833
    assert metrics["false_positive_rate"] == 0.0833
    # FNR = FN / (TP + FN) = 8 / 40 = 0.2000
    assert metrics["false_negative_rate"] == 0.2000


def test_probability_metrics(synthetic_eval_data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Verify ROC-AUC and Average Precision calculations with class probabilities."""
    y_true, y_pred, y_proba = synthetic_eval_data
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba)

    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["average_precision"] is not None
    assert 0.0 <= metrics["average_precision"] <= 1.0


def test_majority_baseline_behavior(synthetic_eval_data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Verify majority baseline predicts Ham (0) with zero spam detection."""
    y_true, _, _ = synthetic_eval_data
    baseline = evaluate_majority_baseline(y_true)

    assert baseline["spam"]["recall"] == 0.0
    assert baseline["spam"]["precision"] == 0.0
    assert baseline["spam"]["f1_score"] == 0.0
    assert baseline["ham"]["recall"] == 1.0
    assert baseline["accuracy"] == 0.6000  # 60 ham out of 100 total
    assert baseline["balanced_accuracy"] == 0.5000


# =====================================================================
# Tests: Error Analysis & Misclassifications
# =====================================================================

def test_misclassifications_identification(synthetic_test_df: pd.DataFrame):
    """Verify identification of false positives and false negatives."""
    # Synthetic: 4 samples: [Ham, Spam, Ham, Spam]
    # Predict: [Spam, Spam, Ham, Ham] -> m1 is FP (0->1), m4 is FN (1->0)
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.85, 0.95, 0.10, 0.20])

    analysis = analyze_misclassifications(synthetic_test_df, y_pred, y_proba)
    assert analysis["false_positive_count"] == 1
    assert analysis["false_negative_count"] == 1
    assert analysis["false_positive_details"][0]["message_id"] == "m1"
    assert analysis["false_negative_details"][0]["message_id"] == "m4"


# =====================================================================
# Tests: Visualizations & Report Artifacts Generation
# =====================================================================

def test_visualizations_generation(tmp_path: Path, synthetic_eval_data: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Verify all 5 evaluation figures are created as valid image files."""
    y_true, y_pred, y_proba = synthetic_eval_data
    model_m = compute_comprehensive_metrics(y_true, y_pred, y_proba)
    base_m = evaluate_majority_baseline(y_true)

    fig_paths = generate_all_visualizations(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        model_metrics=model_m,
        baseline_metrics=base_m,
        figures_dir=tmp_path,
    )

    expected_keys = [
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
        "model_vs_baseline",
        "classification_metrics",
    ]
    for key in expected_keys:
        assert key in fig_paths
        p = fig_paths[key]
        assert p.exists()
        assert p.stat().st_size > 1000  # Non-trivial image size


def test_markdown_and_json_report_generation(
    tmp_path: Path,
    synthetic_eval_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    synthetic_test_df: pd.DataFrame,
):
    """Verify JSON summary and Markdown findings are written properly."""
    y_true, y_pred, y_proba = synthetic_eval_data
    model_m = compute_comprehensive_metrics(y_true, y_pred, y_proba)
    base_m = evaluate_majority_baseline(y_true)
    err = analyze_misclassifications(synthetic_test_df, np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))

    summary = build_evaluation_report(
        model_metrics=model_m,
        baseline_metrics=base_m,
        error_analysis=err,
        model_sha256="fake_sha",
        test_split_sha256="fake_test_sha",
        test_count=100,
    )

    json_path = tmp_path / "test_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)
    assert json_path.exists()

    md_path = tmp_path / "findings.md"
    written_md = write_markdown_findings(summary, md_path)
    assert written_md.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "# SpamGuard AI — Phase 06 Final Model Evaluation Findings" in content
    assert "Definitive Test Results" in content
    assert "Confusion Matrix Analysis" in content


# =====================================================================
# Tests: Model Immutability & Reproducibility
# =====================================================================

def test_model_immutability_during_evaluation():
    """Verify that running evaluation does not mutate pipeline weights or attributes."""
    pipeline = load_trusted_pipeline(SAVED_PIPELINE_PATH, EXPECTED_MODEL_SHA256)
    nb_step = pipeline.named_steps["classifier"]
    original_feature_log_prob = nb_step.feature_log_prob_.copy()
    original_class_count = nb_step.class_count_.copy()

    X_test, _, _ = load_test_data(TEST_SPLIT_FILE)
    _ = pipeline.predict(X_test.iloc[:50])
    _ = pipeline.predict_proba(X_test.iloc[:50])

    # Assert model parameters remained strictly identical
    np.testing.assert_array_equal(nb_step.feature_log_prob_, original_feature_log_prob)
    np.testing.assert_array_equal(nb_step.class_count_, original_class_count)


def test_deterministic_evaluation_output():
    """Verify evaluation metrics on the test partition are completely deterministic."""
    pipeline = load_trusted_pipeline(SAVED_PIPELINE_PATH, EXPECTED_MODEL_SHA256)
    X_test, y_test, _ = load_test_data(TEST_SPLIT_FILE)

    pred1 = pipeline.predict(X_test)
    pred2 = pipeline.predict(X_test)
    proba1 = pipeline.predict_proba(X_test)[:, 1]
    proba2 = pipeline.predict_proba(X_test)[:, 1]

    np.testing.assert_array_equal(pred1, pred2)
    np.testing.assert_array_almost_equal(proba1, proba2)

    m1 = compute_comprehensive_metrics(y_test, pred1, proba1)
    m2 = compute_comprehensive_metrics(y_test, pred2, proba2)
    assert m1 == m2
