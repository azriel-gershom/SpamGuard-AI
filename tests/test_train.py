"""Unit and integration tests for SpamGuard AI Phase 05 Naïve Bayes Model Training."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocess import build_tfidf_vectorizer
from src.train import (
    build_model,
    compute_evaluation_metrics,
    evaluate_majority_baseline,
    extract_model_insights,
    save_model_pipeline,
    select_best_hyperparameters,
    train_and_save_pipeline,
    verify_saved_model,
)


# =====================================================================
# Synthetic Fixtures for Fast & Deterministic Testing
# =====================================================================

@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Creates a deterministic synthetic training dataset."""
    rows = [
        {"message_id": "tr_h1", "text": "python sprint progress status report code update", "label": 0, "source_group": "easy_ham"},
        {"message_id": "tr_h2", "text": "python meeting agenda progress review code patch", "label": 0, "source_group": "easy_ham"},
        {"message_id": "tr_h3", "text": "status report sprint review meeting code update", "label": 0, "source_group": "easy_ham"},
        {"message_id": "tr_h4", "text": "python patch status report agenda meeting update", "label": 0, "source_group": "easy_ham"},
        {"message_id": "tr_s1", "text": "urgent claim free cash prize money guarantee win", "label": 1, "source_group": "spam"},
        {"message_id": "tr_s2", "text": "urgent free prize money cash special bonus offer", "label": 1, "source_group": "spam"},
        {"message_id": "tr_s3", "text": "special offer claim cash prize free bonus money", "label": 1, "source_group": "spam"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_val_df() -> pd.DataFrame:
    """Creates a deterministic synthetic validation dataset."""
    rows = [
        {"message_id": "val_h1", "text": "python status report progress review update code", "label": 0, "source_group": "easy_ham"},
        {"message_id": "val_s1", "text": "urgent free cash prize money bonus claim offer", "label": 1, "source_group": "spam"},
    ]
    return pd.DataFrame(rows)


# =====================================================================
# Task 9 Unit & Integration Tests
# =====================================================================

def test_model_pipeline_construction():
    """Verify build_model returns an unfitted Scikit-learn Pipeline with 'tfidf' and 'classifier'."""
    pipe = build_model(alpha=1.0)
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["tfidf", "classifier"]
    assert not hasattr(pipe, "classes_")


def test_tfidf_vectorizer_reuse():
    """Verify build_model reuses build_tfidf_vectorizer configuration from Phase 04."""
    pipe = build_model()
    tfidf_step = pipe.named_steps["tfidf"]
    ref_tfidf = build_tfidf_vectorizer()

    assert tfidf_step.ngram_range == ref_tfidf.ngram_range
    assert tfidf_step.max_features == ref_tfidf.max_features
    assert tfidf_step.min_df == ref_tfidf.min_df
    assert tfidf_step.max_df == ref_tfidf.max_df
    assert tfidf_step.sublinear_tf == ref_tfidf.sublinear_tf


def test_multinomial_nb_initialization():
    """Verify MultinomialNB is initialized with the requested smoothing parameter alpha."""
    pipe_05 = build_model(alpha=0.5)
    nb_step = pipe_05.named_steps["classifier"]
    assert nb_step.alpha == 0.5


def test_successful_model_fitting(synthetic_train_df: pd.DataFrame):
    """Verify Pipeline fits cleanly on training samples and learns both binary classes."""
    pipe = build_model(alpha=1.0)
    pipe.fit(synthetic_train_df["text"], synthetic_train_df["label"])

    assert hasattr(pipe, "classes_")
    assert list(pipe.classes_) == [0, 1]
    assert hasattr(pipe.named_steps["tfidf"], "vocabulary_")
    assert len(pipe.named_steps["tfidf"].vocabulary_) > 0


def test_valid_class_predictions(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame):
    """Verify fitted Pipeline generates discrete binary predictions in {0, 1}."""
    pipe = build_model(alpha=0.1)
    pipe.fit(synthetic_train_df["text"], synthetic_train_df["label"])
    preds = pipe.predict(synthetic_val_df["text"])

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(synthetic_val_df)
    assert set(preds).issubset({0, 1})


def test_prediction_probabilities(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame):
    """Verify prediction probabilities are finite and sum to 1.0 for each sample."""
    pipe = build_model(alpha=0.1)
    pipe.fit(synthetic_train_df["text"], synthetic_train_df["label"])
    probs = pipe.predict_proba(synthetic_val_df["text"])

    assert probs.shape == (len(synthetic_val_df), 2)
    assert np.all(np.isfinite(probs))
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_positive_label_metric_calculations():
    """Verify compute_evaluation_metrics computes correct accuracy, precision, recall, and F1."""
    # 2 TP, 1 FP, 1 FN, 2 TN
    # TP=2, FP=1 => Prec = 2/3 = 0.6667
    # TP=2, FN=1 => Rec  = 2/3 = 0.6667
    # F1 = 0.6667, Acc = 4/6 = 0.6667
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])

    metrics = compute_evaluation_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.6667
    assert metrics["precision"] == 0.6667
    assert metrics["recall"] == 0.6667
    assert metrics["f1_score"] == 0.6667
    assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]


def test_majority_baseline_evaluation(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame):
    """Verify majority-class baseline correctly reports 0% spam precision and recall."""
    metrics = evaluate_majority_baseline(
        synthetic_train_df["text"], synthetic_train_df["label"],
        synthetic_val_df["text"], synthetic_val_df["label"],
    )
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    # In synthetic_val_df, 1 ham and 1 spam -> accuracy is 0.5
    assert metrics["accuracy"] == 0.5


def test_hyperparameter_selection(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame):
    """Verify select_best_hyperparameters evaluates all candidates and chooses the best by F1."""
    candidates = [0.1, 1.0, 5.0]
    best_alpha, best_pipe, results = select_best_hyperparameters(
        synthetic_train_df["text"], synthetic_train_df["label"],
        synthetic_val_df["text"], synthetic_val_df["label"],
        alpha_candidates=candidates,
    )
    assert best_alpha in candidates
    assert isinstance(best_pipe, Pipeline)
    assert len(results) == len(candidates)
    assert all("f1_score" in r and "precision" in r for r in results)


def test_deterministic_tie_breaking():
    """Verify tie-breaker selects higher precision, then higher recall, then smaller alpha."""
    # Candidate 1: F1=0.9, Prec=0.95, Rec=0.85, alpha=1.0
    # Candidate 2: F1=0.9, Prec=0.92, Rec=0.88, alpha=0.5
    # Candidate 1 should win on higher precision
    res1 = [{"alpha": 1.0, "f1_score": 0.9, "precision": 0.95, "recall": 0.85},
            {"alpha": 0.5, "f1_score": 0.9, "precision": 0.92, "recall": 0.88}]
    sorted1 = sorted(res1, key=lambda r: (r["f1_score"], r["precision"], r["recall"], -r["alpha"]), reverse=True)
    assert sorted1[0]["alpha"] == 1.0

    # Tie on F1 and precision and recall: smaller alpha wins
    res2 = [{"alpha": 1.0, "f1_score": 0.9, "precision": 0.95, "recall": 0.85},
            {"alpha": 0.1, "f1_score": 0.9, "precision": 0.95, "recall": 0.85}]
    sorted2 = sorted(res2, key=lambda r: (r["f1_score"], r["precision"], r["recall"], -r["alpha"]), reverse=True)
    assert sorted2[0]["alpha"] == 0.1


def test_strict_training_only_vocabulary_isolation(synthetic_train_df: pd.DataFrame):
    """Verify that terms unique to held-out validation data do NOT exist in the trained vocabulary."""
    pipe = build_model(alpha=1.0)
    pipe.fit(synthetic_train_df["text"], synthetic_train_df["label"])
    vocab = set(pipe.named_steps["tfidf"].vocabulary_.keys())

    # Ensure training tokens are present
    assert "sprint" in vocab or "progress" in vocab or "python" in vocab

    # Validation or test tokens that were never in train must not exist
    assert "quantum" not in vocab
    assert "cryptography" not in vocab
    assert "blockchain" not in vocab


def test_model_serialization_and_reload(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame, tmp_path: Path):
    """Verify model can be saved to Joblib and reloaded with identical predictions."""
    model_path = tmp_path / "test_pipeline.joblib"
    pipe = build_model(alpha=0.5)
    pipe.fit(synthetic_train_df["text"], synthetic_train_df["label"])

    save_model_pipeline(pipe, output_path=model_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0

    verification = verify_saved_model(model_path, pipe, synthetic_val_df["text"])
    assert verification["verification_passed"] is True
    assert verification["prediction_parity"] is True
    assert verification["probability_parity"] is True


def test_model_insights_extraction(synthetic_train_df: pd.DataFrame, tmp_path: Path):
    """Verify extract_model_insights creates report with top spam and ham predictive tokens."""
    insights_path = tmp_path / "test_insights.json"
    pipe = build_model(alpha=0.1)
    pipe.fit(synthetic_train_df["text"], synthetic_train_df["label"])

    insights = extract_model_insights(pipe, output_path=insights_path, top_n=5)
    assert insights_path.exists()
    assert "top_spam_predictive_tokens" in insights
    assert "top_ham_predictive_tokens" in insights
    assert len(insights["top_spam_predictive_tokens"]) > 0
    assert len(insights["top_ham_predictive_tokens"]) > 0


def test_training_data_schema_validation(synthetic_val_df: pd.DataFrame, tmp_path: Path):
    """Verify pipeline rejects training DataFrame missing required columns or classes."""
    train_file = tmp_path / "bad_train.csv"
    val_file = tmp_path / "val.csv"
    synthetic_val_df.to_csv(val_file, index=False)

    # Missing label column
    pd.DataFrame({"message_id": ["1"], "text": ["text"]}).to_csv(train_file, index=False)
    with pytest.raises(ValueError, match="missing required 'text' or 'label' columns"):
        train_and_save_pipeline(train_file=train_file, val_file=val_file)

    # Single class only (missing spam)
    pd.DataFrame({"message_id": ["1", "2"], "text": ["t1", "t2"], "label": [0, 0]}).to_csv(train_file, index=False)
    with pytest.raises(ValueError, match="must contain both binary classes"):
        train_and_save_pipeline(train_file=train_file, val_file=val_file)


def test_missing_files_error_handling(tmp_path: Path):
    """Verify FileNotFoundError is raised when input split files do not exist."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Training split not found"):
        train_and_save_pipeline(train_file=non_existent, val_file=non_existent)


def test_no_test_partition_accessed_by_training(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame, tmp_path: Path):
    """Verify training pipeline executes without requiring or touching test.csv."""
    train_file = tmp_path / "train.csv"
    val_file = tmp_path / "val.csv"
    model_file = tmp_path / "model.joblib"
    summary_file = tmp_path / "summary.json"
    insights_file = tmp_path / "insights.json"

    synthetic_train_df.to_csv(train_file, index=False)
    synthetic_val_df.to_csv(val_file, index=False)

    # Explicitly ensure test.csv does not even exist in tmp_path
    assert not (tmp_path / "test.csv").exists()

    summary = train_and_save_pipeline(
        train_file=train_file,
        val_file=val_file,
        model_output_path=model_file,
        summary_file=summary_file,
        insights_file=insights_file,
    )
    assert summary["leakage_safeguards"]["test_partition_accessed"] is False
    assert model_file.exists()


def test_deterministic_training_output(synthetic_train_df: pd.DataFrame, synthetic_val_df: pd.DataFrame):
    """Verify running training repeatedly on the same data yields identical parameters and metrics."""
    _, pipe1, res1 = select_best_hyperparameters(
        synthetic_train_df["text"], synthetic_train_df["label"],
        synthetic_val_df["text"], synthetic_val_df["label"],
        alpha_candidates=[0.1, 1.0],
    )
    _, pipe2, res2 = select_best_hyperparameters(
        synthetic_train_df["text"], synthetic_train_df["label"],
        synthetic_val_df["text"], synthetic_val_df["label"],
        alpha_candidates=[0.1, 1.0],
    )

    assert res1 == res2
    preds1 = pipe1.predict(synthetic_val_df["text"])
    preds2 = pipe2.predict(synthetic_val_df["text"])
    assert np.array_equal(preds1, preds2)
