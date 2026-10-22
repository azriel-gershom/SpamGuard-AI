"""SpamGuard AI — Multinomial Naïve Bayes Model Training & Validation.

Phase 05:
- Builds reproducible Scikit-learn Pipeline combining TF-IDF and MultinomialNB.
- Trains strictly on training partition (data/processed/splits/train.csv).
- Establishes a majority-class baseline to prove accuracy alone is insufficient.
- Tunes Laplace smoothing parameter alpha across [0.1, 0.5, 1.0, 2.0, 5.0] on validation set.
- Selects optimal alpha based on spam-class F1-score with deterministic tie-breaking.
- Serializes complete fitted Pipeline to models/spam_pipeline.joblib.
- Verifies model reloading with prediction parity checks.
- Generates reports/training_summary.json and reports/model_insights.json.
- Strictly avoids accessing or evaluating the test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    ALPHA_CANDIDATES,
    MODEL_INSIGHTS_FILE,
    SAVED_PIPELINE_PATH,
    TRAIN_SPLIT_FILE,
    TRAINING_SUMMARY_FILE,
    VAL_SPLIT_FILE,
)
from src.preprocess import build_tfidf_vectorizer, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpamGuardAI.Train")


# =====================================================================
# Model Architecture Factory
# =====================================================================

def build_model(alpha: float = 1.0) -> Pipeline:
    """Constructs an unfitted Scikit-learn Pipeline with TF-IDF and Multinomial Naïve Bayes.

    Architecture:
    Step 1: 'tfidf' — TfidfVectorizer (unfitted baseline from Phase 04 factory).
    Step 2: 'classifier' — MultinomialNB(alpha=alpha).
    """
    tfidf = build_tfidf_vectorizer()
    classifier = MultinomialNB(alpha=alpha)

    return Pipeline([
        ("tfidf", tfidf),
        ("classifier", classifier),
    ])


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA-256 checksum of a file in 64KB blocks."""
    if not filepath.exists():
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =====================================================================
# Evaluation & Baseline Metrics
# =====================================================================

def compute_evaluation_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Computes accuracy, precision, recall, F1-score (pos_label=1), and confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
    }


def evaluate_majority_baseline(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
) -> dict[str, Any]:
    """Fits a DummyClassifier(strategy='most_frequent') to demonstrate the baseline."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_val)

    metrics = compute_evaluation_metrics(y_val, y_pred)
    logger.info(
        "Majority-Class Baseline (Val): Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )
    return metrics


# =====================================================================
# Hyperparameter Tuning on Validation Data
# =====================================================================

def select_best_hyperparameters(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    alpha_candidates: list[float] = ALPHA_CANDIDATES,
) -> tuple[float, Pipeline, list[dict[str, Any]]]:
    """Evaluates MultinomialNB alpha candidates on the validation set.

    Tie-breaking hierarchy:
    1. Highest spam F1-score
    2. Highest spam Precision
    3. Highest spam Recall
    4. Smallest alpha
    """
    candidate_results: list[dict[str, Any]] = []
    fitted_pipelines: dict[float, Pipeline] = {}

    for alpha in alpha_candidates:
        pipeline = build_model(alpha=alpha)
        # Fit strictly on training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[alpha] = pipeline

        # Predict on validation data
        y_val_pred = pipeline.predict(X_val)
        metrics = compute_evaluation_metrics(y_val, y_val_pred)

        record = {
            "alpha": float(alpha),
            **metrics,
        }
        candidate_results.append(record)
        logger.info(
            "Candidate Alpha=%.2f -> Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
            alpha,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        )

    # Sort deterministically using tie-breaking hierarchy
    # Note: negative alpha for ascending order when other metrics tie
    sorted_candidates = sorted(
        candidate_results,
        key=lambda r: (r["f1_score"], r["precision"], r["recall"], -r["alpha"]),
        reverse=True,
    )

    best_record = sorted_candidates[0]
    best_alpha = float(best_record["alpha"])
    best_pipeline = fitted_pipelines[best_alpha]

    logger.info(
        "Selected Best Alpha=%.2f (Val F1=%.4f, Precision=%.4f, Recall=%.4f, Acc=%.4f)",
        best_alpha,
        best_record["f1_score"],
        best_record["precision"],
        best_record["recall"],
        best_record["accuracy"],
    )
    return best_alpha, best_pipeline, candidate_results


# =====================================================================
# Serialization & Verification
# =====================================================================

def save_model_pipeline(pipeline: Pipeline, output_path: Path = SAVED_PIPELINE_PATH) -> Path:
    """Serializes the fitted Pipeline (TF-IDF + MultinomialNB) to disk via Joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path, compress=3)
    logger.info("Saved fitted pipeline to %s", output_path)
    return output_path


def verify_saved_model(
    saved_path: Path,
    original_pipeline: Pipeline,
    X_sample: pd.Series,
) -> dict[str, Any]:
    """Reloads the saved Joblib pipeline and validates prediction consistency."""
    if not saved_path.exists():
        raise FileNotFoundError(f"Saved model file not found: {saved_path}")

    # Reload model
    loaded_pipeline: Pipeline = joblib.load(saved_path)

    # Verify pipeline structure
    if not isinstance(loaded_pipeline, Pipeline):
        raise TypeError(f"Loaded object is not a sklearn Pipeline: {type(loaded_pipeline)}")

    step_names = [name for name, _ in loaded_pipeline.steps]
    if step_names != ["tfidf", "classifier"]:
        raise ValueError(f"Pipeline steps mismatch. Expected ['tfidf', 'classifier'], got {step_names}")

    # Verify classes
    classes = list(loaded_pipeline.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected classes in loaded model: {classes}. Expected [0, 1]")

    # Verify predictions on fixed sample
    orig_preds = original_pipeline.predict(X_sample)
    loaded_preds = loaded_pipeline.predict(X_sample)
    if not np.array_equal(orig_preds, loaded_preds):
        raise AssertionError("Loaded model predictions differ from original fitted pipeline!")

    # Verify prediction probabilities
    orig_probs = original_pipeline.predict_proba(X_sample)
    loaded_probs = loaded_pipeline.predict_proba(X_sample)
    if not np.allclose(orig_probs, loaded_probs, atol=1e-6):
        raise AssertionError("Loaded model probabilities differ from original fitted pipeline!")

    checksum = compute_file_sha256(saved_path)
    logger.info("Model reload verification PASSED (SHA-256: %s)", checksum)

    return {
        "verification_passed": True,
        "sample_size_evaluated": len(X_sample),
        "prediction_parity": True,
        "probability_parity": True,
        "classes": classes,
        "artifact_sha256": checksum,
    }


# =====================================================================
# Model Interpretability Analysis
# =====================================================================

def extract_model_insights(
    pipeline: Pipeline,
    output_path: Path = MODEL_INSIGHTS_FILE,
    top_n: int = 25,
) -> dict[str, Any]:
    """Extracts top predictive tokens using class-conditional log probability differences."""
    tfidf = pipeline.named_steps["tfidf"]
    classifier: MultinomialNB = pipeline.named_steps["classifier"]

    feature_names = np.array(tfidf.get_feature_names_out())
    # feature_log_prob_ has shape (2, n_features): index 0=Ham, index 1=Spam
    log_probs = classifier.feature_log_prob_
    log_diff = log_probs[1] - log_probs[0]  # Positive => more predictive of Spam; Negative => Ham

    # Sort descending for Spam
    spam_sorted_indices = np.argsort(log_diff)[::-1]
    # Sort ascending for Ham
    ham_sorted_indices = np.argsort(log_diff)

    # Filter out potential email addresses, person-specific handles, or numbers
    identifying_pattern = re.compile(r"[@_\d]|(^jm$)|(netnoteinc)", re.IGNORECASE)

    def collect_features(indices: np.ndarray, is_spam: bool) -> list[dict[str, Any]]:
        features = []
        for idx in indices:
            token = str(feature_names[idx])
            # Redact / omit identifying or internal technical addresses
            if identifying_pattern.search(token):
                continue
            features.append({
                "token": token,
                "log_prob_diff": round(float(log_diff[idx]), 4),
                "spam_log_prob": round(float(log_probs[1, idx]), 4),
                "ham_log_prob": round(float(log_probs[0, idx]), 4),
                "association": "Spam" if is_spam else "Ham",
            })
            if len(features) >= top_n:
                break
        return features

    top_spam_features = collect_features(spam_sorted_indices, is_spam=True)
    top_ham_features = collect_features(ham_sorted_indices, is_spam=False)

    insights: dict[str, Any] = {
        "methodology": (
            "Class-conditional log-probability difference: log P(token|Spam) - log P(token|Ham). "
            "Positive values indicate tokens with higher likelihood under Spam; negative values "
            "indicate higher likelihood under Ham. These reflect learned statistical associations, "
            "not causal determinations."
        ),
        "vocabulary_size": int(len(feature_names)),
        "top_spam_predictive_tokens": top_spam_features,
        "top_ham_predictive_tokens": top_ham_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    logger.info("Exported model interpretability report to %s", output_path)
    return insights


# =====================================================================
# Training Metadata Report
# =====================================================================

def _json_serial_default(obj: Any) -> Any:
    """Fallback JSON serializer for NumPy and Path types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def generate_training_summary(
    best_alpha: float,
    pipeline: Pipeline,
    candidate_results: list[dict[str, Any]],
    baseline_metrics: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    verification_results: dict[str, Any],
    train_file: Path = TRAIN_SPLIT_FILE,
    val_file: Path = VAL_SPLIT_FILE,
    model_path: Path = SAVED_PIPELINE_PATH,
    output_path: Path = TRAINING_SUMMARY_FILE,
) -> dict[str, Any]:
    """Constructs comprehensive, reproducible training summary metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tfidf = pipeline.named_steps["tfidf"]
    vocab_size = int(len(tfidf.vocabulary_))

    best_record = next(r for r in candidate_results if r["alpha"] == best_alpha)

    summary: dict[str, Any] = {
        "model_name": "SpamGuard AI Multinomial Naïve Bayes Classifier",
        "training_algorithm": "MultinomialNB with TF-IDF Vectorization",
        "saved_model_artifact_path": str(model_path),
        "model_artifact_sha256": str(verification_results.get("artifact_sha256", "")),
        "model_reload_verification": verification_results,
        "training_record_count": int(len(train_df)),
        "validation_record_count": int(len(val_df)),
        "dataset_checksums": {
            "train_csv_sha256": compute_file_sha256(train_file),
            "validation_csv_sha256": compute_file_sha256(val_file),
        },
        "class_distributions": {
            "training": {
                "ham_count": int((train_df["label"] == 0).sum()),
                "spam_count": int((train_df["label"] == 1).sum()),
            },
            "validation": {
                "ham_count": int((val_df["label"] == 0).sum()),
                "spam_count": int((val_df["label"] == 1).sum()),
            },
        },
        "hyperparameter_selection": {
            "alpha_candidates_tested": [float(r["alpha"]) for r in candidate_results],
            "candidate_evaluations": candidate_results,
            "selected_alpha": float(best_alpha),
            "selection_criterion": "Highest spam-class F1-score with precision tie-breaker",
        },
        "validation_performance": {
            "majority_class_baseline": baseline_metrics,
            "selected_multinomial_nb": best_record,
        },
        "feature_extraction": {
            "analyzer": "word",
            "ngram_range": [int(x) for x in tfidf.ngram_range],
            "max_features": int(tfidf.max_features) if tfidf.max_features else None,
            "min_df": int(tfidf.min_df) if isinstance(tfidf.min_df, (int, np.integer)) else float(tfidf.min_df),
            "max_df": float(tfidf.max_df),
            "sublinear_tf": bool(tfidf.sublinear_tf),
            "vocabulary_size_learned": vocab_size,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "leakage_safeguards": {
            "test_partition_accessed": False,
            "training_only_fit": True,
            "validation_only_tuning": True,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_serial_default)

    logger.info("Written training summary report to %s", output_path)
    return summary


# =====================================================================
# Main Pipeline Function
# =====================================================================

def train_and_save_pipeline(
    train_file: Path = TRAIN_SPLIT_FILE,
    val_file: Path = VAL_SPLIT_FILE,
    model_output_path: Path = SAVED_PIPELINE_PATH,
    summary_file: Path = TRAINING_SUMMARY_FILE,
    insights_file: Path = MODEL_INSIGHTS_FILE,
    alpha_candidates: list[float] = ALPHA_CANDIDATES,
) -> dict[str, Any]:
    """Executes the full Phase 05 model training, hyperparameter tuning, and verification workflow."""
    if not train_file.exists():
        raise FileNotFoundError(f"Training split not found at {train_file}. Run Phase 04 first.")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation split not found at {val_file}. Run Phase 04 first.")

    logger.info("Loading training and validation partitions...")
    train_df = pd.read_csv(train_file).fillna({"text": ""})
    val_df = pd.read_csv(val_file).fillna({"text": ""})

    # Validate inputs
    for name, df in [("Training", train_df), ("Validation", val_df)]:
        if df.empty:
            raise ValueError(f"{name} split is empty.")
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{name} split missing required 'text' or 'label' columns.")
        labels = set(df["label"].unique())
        if not labels.issubset({0, 1}) or len(labels) < 2:
            raise ValueError(f"{name} split must contain both binary classes {{0, 1}}. Found: {labels}")

    X_train, y_train = train_df["text"], train_df["label"]
    X_val, y_val = val_df["text"], val_df["label"]

    logger.info("Evaluating majority-class baseline on validation data...")
    baseline_metrics = evaluate_majority_baseline(X_train, y_train, X_val, y_val)

    logger.info("Tuning MultinomialNB smoothing parameter alpha across candidates %s...", alpha_candidates)
    best_alpha, best_pipeline, candidate_results = select_best_hyperparameters(
        X_train, y_train, X_val, y_val, alpha_candidates=alpha_candidates
    )

    logger.info("Serializing selected pipeline to %s...", model_output_path)
    save_model_pipeline(best_pipeline, output_path=model_output_path)

    logger.info("Verifying saved model artifact...")
    verification = verify_saved_model(model_output_path, best_pipeline, X_val.iloc[:100])

    logger.info("Extracting model interpretability insights...")
    insights = extract_model_insights(best_pipeline, output_path=insights_file)

    logger.info("Exporting training metadata summary...")
    summary = generate_training_summary(
        best_alpha=best_alpha,
        pipeline=best_pipeline,
        candidate_results=candidate_results,
        baseline_metrics=baseline_metrics,
        train_df=train_df,
        val_df=val_df,
        verification_results=verification,
        train_file=train_file,
        val_file=val_file,
        model_path=model_output_path,
        output_path=summary_file,
    )

    # Print concise terminal report
    best_val = next(r for r in candidate_results if r["alpha"] == best_alpha)
    print("\n" + "=" * 60)
    print(" SPAMGUARD AI — MODEL TRAINING REPORT (PHASE 05)")
    print("=" * 60)
    print(f"Algorithm:           Multinomial Naïve Bayes")
    print(f"Training Emails:     {len(train_df):,} (Ham: {(train_df['label']==0).sum():,}, Spam: {(train_df['label']==1).sum():,})")
    print(f"Validation Emails:   {len(val_df):,} (Ham: {(val_df['label']==0).sum():,}, Spam: {(val_df['label']==1).sum():,})")
    print(f"Vocabulary Size:     {insights['vocabulary_size']:,} n-gram features")
    print(f"Selected Alpha:      {best_alpha}")
    print("-" * 60)
    print(f"Majority Baseline:   Acc={baseline_metrics['accuracy']:.4f} | Spam F1={baseline_metrics['f1_score']:.4f}")
    print(f"Selected Model:      Acc={best_val['accuracy']:.4f} | Prec={best_val['precision']:.4f} | Rec={best_val['recall']:.4f} | F1={best_val['f1_score']:.4f}")
    print(f"Artifact Saved:      {model_output_path}")
    print(f"Verification:        PASSED (Parity Confirmed)")
    print("=" * 60 + "\n")

    return summary


def main() -> None:
    """CLI Entrypoint for Phase 05 Model Training."""
    parser = argparse.ArgumentParser(description="SpamGuard AI — Model Training (Phase 05)")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_SPLIT_FILE,
        help="Path to training partition CSV",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=VAL_SPLIT_FILE,
        help="Path to validation partition CSV",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=SAVED_PIPELINE_PATH,
        help="Destination path for serialized spam_pipeline.joblib",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=TRAINING_SUMMARY_FILE,
        help="Destination path for training_summary.json",
    )
    parser.add_argument(
        "--insights-output",
        type=Path,
        default=MODEL_INSIGHTS_FILE,
        help="Destination path for model_insights.json",
    )
    args = parser.parse_args()

    train_and_save_pipeline(
        train_file=args.train_file,
        val_file=args.val_file,
        model_output_path=args.model_output,
        summary_file=args.summary_output,
        insights_file=args.insights_output,
    )


if __name__ == "__main__":
    main()
