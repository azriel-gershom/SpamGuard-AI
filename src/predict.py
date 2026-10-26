"""SpamGuard AI — High-Reliability Prediction Engine.

Phase 07:
- Encapsulates inference logic for single and batch email classification.
- Validates model SHA-256 integrity and internal pipeline structure prior to inference.
- Reuses Phase 04 text normalization and Phase 02 subject/body combination conventions.
- Employs vectorized prediction for high-throughput batch classification.
- Provides mathematical feature contribution extraction (log-odds difference) for model explainability.
- Features robust defense against CSV formula injection and payload denial-of-service.
- CLI entry point supporting single email, stdin streaming, and batch CSV processing.
- Completely independent of Streamlit UI (ready for Phase 08 consumption).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_TOP_K_EXPLANATIONS,
    EXPECTED_PIPELINE_SHA256,
    MAX_BATCH_SIZE,
    MAX_INPUT_CHARS,
    SAVED_PIPELINE_PATH,
)
from src.preprocess import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpamGuardAI.Predict")


# =====================================================================
# Custom Exceptions
# =====================================================================

class SpamGuardError(Exception):
    """Base exception for all SpamGuard AI errors."""


class SpamGuardModelError(SpamGuardError):
    """Raised when model loading, integrity verification, or artifact structure fails."""


class SpamGuardValidationError(SpamGuardError):
    """Raised when an input email fails validation (empty, oversized, wrong type)."""


class SpamGuardBatchError(SpamGuardError):
    """Raised when batch inference input is invalid or exceeds bounds."""


# =====================================================================
# Structured Prediction Data Types
# =====================================================================

@dataclass(frozen=True)
class FeatureContribution:
    """Individual token contribution towards the log-odds of Spam vs. Ham.

    Attributes:
        token: Term from the fitted TF-IDF vocabulary.
        weight: TF-IDF feature weight in the evaluated email.
        log_odds_diff: Model parameter log P(token|Spam) - log P(token|Ham).
        contribution: weight * log_odds_diff (net push towards Spam or Ham).
        association: 'Spam-leaning' if contribution > 0 else 'Ham-leaning'.
    """
    token: str
    weight: float
    log_odds_diff: float
    contribution: float
    association: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "weight": round(self.weight, 4),
            "log_odds_diff": round(self.log_odds_diff, 4),
            "contribution": round(self.contribution, 4),
            "association": self.association,
        }


@dataclass(frozen=True)
class PredictionResult:
    """Complete, immutable output of a single email classification.

    Attributes:
        predicted_class: 0 (Ham) or 1 (Spam).
        predicted_label: Human-readable label ('Ham' or 'Spam').
        spam_probability: Estimated probability of Spam [0.0 - 1.0].
        ham_probability: Estimated probability of Ham [0.0 - 1.0].
        is_spam: Boolean convenience flag (True if predicted_class == 1).
        model_identifier: String description of the active model and hyperparameter.
        processing_time_ms: Classification execution time in milliseconds.
        cleaned_char_count: Character count of the normalized text.
        top_features: Influential tokens and their statistical contributions (optional).
    """
    predicted_class: int
    predicted_label: str
    spam_probability: float
    ham_probability: float
    is_spam: bool
    model_identifier: str
    processing_time_ms: float
    cleaned_char_count: int
    top_features: list[FeatureContribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "spam_probability": round(self.spam_probability, 4),
            "ham_probability": round(self.ham_probability, 4),
            "is_spam": self.is_spam,
            "model_identifier": self.model_identifier,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "cleaned_char_count": self.cleaned_char_count,
            "top_features": [f.to_dict() for f in self.top_features],
        }


# =====================================================================
# Integrity & Preprocessing Helpers
# =====================================================================

def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA-256 checksum of a file in 64KB blocks."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found for checksum verification: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def combine_subject_and_body(subject: Any, body: Any) -> str:
    """Combines subject and body following Phase 02 dataset preparation conventions.

    Reproduces the exact format:
    - Both subject and body present: '{subject}\\n\\n{body}'
    - Only body present: '{body}'
    - Only subject present: '{subject}'
    - Neither present: ''
    """
    subj_str = str(subject).strip() if subject is not None else ""
    body_str = str(body).strip() if body is not None else ""

    if subj_str and body_str:
        return f"{subj_str}\n\n{body_str}"
    elif body_str:
        return body_str
    elif subj_str:
        return subj_str
    return ""


def sanitize_csv_cell(value: Any) -> Any:
    """Prevents CSV formula injection by escaping leading control characters."""
    if isinstance(value, str):
        if value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
            return f"'{value}"
    return value


# =====================================================================
# SpamPredictor Engine Class
# =====================================================================

class SpamPredictor:
    """Reusable, production-grade inference engine for SpamGuard AI.

    Features:
    - Pre-inference model integrity validation.
    - Deterministic text sanitization matching training pipeline.
    - Vectorized single and batch prediction.
    - Feature contribution extraction via Naïve Bayes log-odds difference.
    """

    def __init__(self, pipeline: Pipeline, model_sha256: str = "") -> None:
        """Initializes the predictor with an authenticated Scikit-learn Pipeline.

        Use SpamPredictor.load(...) to construct instances safely.
        """
        self._pipeline = pipeline
        self._model_sha256 = model_sha256

        # Extract and validate pipeline components
        self._validate_pipeline_structure()

        # Class index mapping for safe probability extraction
        classes = list(self._pipeline.classes_)
        self._class_to_idx: dict[int, int] = {cls: idx for idx, cls in enumerate(classes)}
        self._ham_idx = self._class_to_idx[0]
        self._spam_idx = self._class_to_idx[1]

        # Extract vocabulary and parameter weights for explainability
        self._tfidf_step = self._pipeline.named_steps["tfidf"]
        self._clf_step = self._pipeline.named_steps["classifier"]
        self._vocab: dict[str, int] = self._tfidf_step.vocabulary_
        self._inv_vocab: dict[int, str] = {idx: token for token, idx in self._vocab.items()}

        # log P(token|Spam) - log P(token|Ham)
        # feature_log_prob_ has shape [2, n_features]
        self._log_odds_diff: np.ndarray = (
            self._clf_step.feature_log_prob_[self._spam_idx]
            - self._clf_step.feature_log_prob_[self._ham_idx]
        )

        self._model_identifier = f"MultinomialNB(alpha={self._clf_step.alpha}) + TF-IDF({len(self._vocab):,} features)"

    @classmethod
    def load(
        cls,
        model_path: Path = SAVED_PIPELINE_PATH,
        expected_sha: str | None = EXPECTED_PIPELINE_SHA256,
    ) -> SpamPredictor:
        """Safely loads and verifies the trained pipeline artifact.

        Raises:
            SpamGuardModelError: If model file is missing, corrupted, or checksum differs.
        """
        if not model_path.exists():
            raise SpamGuardModelError(f"Trained model artifact not found at {model_path}")

        try:
            actual_sha = compute_file_sha256(model_path)
        except Exception as exc:
            raise SpamGuardModelError(f"Failed to read model file for checksum calculation: {exc}") from exc

        if expected_sha is not None:
            if actual_sha.lower() != expected_sha.lower():
                raise SpamGuardModelError(
                    f"Model SHA-256 integrity verification failed! "
                    f"Expected: {expected_sha}, Actual: {actual_sha}."
                )

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            raise SpamGuardModelError(f"Failed to deserialize model artifact with Joblib: {exc}") from exc

        return cls(pipeline=pipeline, model_sha256=actual_sha)

    def _validate_pipeline_structure(self) -> None:
        """Verifies that the deserialized object conforms strictly to expected architecture."""
        if not isinstance(self._pipeline, Pipeline):
            raise SpamGuardModelError(
                f"Invalid model object type. Expected sklearn.pipeline.Pipeline, got {type(self._pipeline)}"
            )

        steps = dict(self._pipeline.steps)
        if "tfidf" not in steps or "classifier" not in steps:
            raise SpamGuardModelError(
                f"Pipeline missing required steps ('tfidf', 'classifier'). Found: {list(steps.keys())}"
            )

        classifier = steps["classifier"]
        if type(classifier).__name__ != "MultinomialNB":
            raise SpamGuardModelError(
                f"Classifier step must be MultinomialNB, got {type(classifier).__name__}"
            )

        if not hasattr(classifier, "classes_"):
            raise SpamGuardModelError("Classifier is not fitted (missing 'classes_' attribute)")

        classes = list(classifier.classes_)
        if sorted(classes) != [0, 1]:
            raise SpamGuardModelError(f"Expected binary classes [0, 1], found: {classes}")

        tfidf = steps["tfidf"]
        if not hasattr(tfidf, "vocabulary_") or not tfidf.vocabulary_:
            raise SpamGuardModelError("TF-IDF vectorizer step is unfitted or possesses an empty vocabulary")

    # =================================================================
    # Text Preprocessing & Validation
    # =================================================================

    def prepare_input_text(self, subject: Any = "", body: Any = "") -> str:
        """Combines and cleans email text, applying strict validation checks.

        Returns:
            Cleaned and normalized text ready for vectorization.

        Raises:
            SpamGuardValidationError: If input is empty, invalid, or exceeds size limits.
        """
        combined = combine_subject_and_body(subject, body)
        cleaned = clean_text(combined)

        if not cleaned:
            raise SpamGuardValidationError(
                "Email content is empty or contains only whitespace after cleaning."
            )

        if len(cleaned) > MAX_INPUT_CHARS:
            raise SpamGuardValidationError(
                f"Input text exceeds maximum allowed limit of {MAX_INPUT_CHARS:,} characters "
                f"({len(cleaned):,} received). Request rejected to prevent denial-of-service."
            )

        return cleaned

    # =================================================================
    # Single Email Prediction
    # =================================================================

    def predict_email(
        self,
        subject: Any = "",
        body: Any = "",
        explain: bool = False,
        top_k: int = DEFAULT_TOP_K_EXPLANATIONS,
    ) -> PredictionResult:
        """Classifies a single incoming email with calibrated probabilities.

        Args:
            subject: Email subject line.
            body: Email body text (plain text or extracted HTML text).
            explain: If True, computes token-level log-odds contributions.
            top_k: Number of influential tokens to return if explain=True.

        Returns:
            PredictionResult containing classification, probabilities, and metadata.

        Raises:
            SpamGuardValidationError: On invalid or empty input.
            SpamGuardModelError: If model inference produces invalid output.
        """
        t0 = time.perf_counter()
        cleaned_text = self.prepare_input_text(subject, body)

        # Execute frozen model inference
        try:
            preds = self._pipeline.predict([cleaned_text])
            probs = self._pipeline.predict_proba([cleaned_text])[0]
        except Exception as exc:
            raise SpamGuardModelError(f"Model prediction inference failed: {exc}") from exc

        duration_ms = (time.perf_counter() - t0) * 1000.0

        # Validate probability characteristics
        pred_class = int(preds[0])
        ham_prob = float(probs[self._ham_idx])
        spam_prob = float(probs[self._spam_idx])

        if not (np.isfinite(ham_prob) and np.isfinite(spam_prob)):
            raise SpamGuardModelError("Model returned non-finite (NaN or Inf) probability values")

        if not (0.0 <= ham_prob <= 1.0 and 0.0 <= spam_prob <= 1.0):
            raise SpamGuardModelError(
                f"Model probability out of bounds [0, 1]: Ham={ham_prob}, Spam={spam_prob}"
            )

        if abs((ham_prob + spam_prob) - 1.0) > 1e-4:
            raise SpamGuardModelError(
                f"Model probabilities do not sum to 1.0: sum = {ham_prob + spam_prob}"
            )

        # Optional feature explanations
        features: list[FeatureContribution] = []
        if explain:
            features = self.explain_email(cleaned_text, top_k=top_k)

        return PredictionResult(
            predicted_class=pred_class,
            predicted_label="Spam" if pred_class == 1 else "Ham",
            spam_probability=spam_prob,
            ham_probability=ham_prob,
            is_spam=(pred_class == 1),
            model_identifier=self._model_identifier,
            processing_time_ms=duration_ms,
            cleaned_char_count=len(cleaned_text),
            top_features=features,
        )

    # =================================================================
    # Feature Contribution Explainability
    # =================================================================

    def explain_email(self, cleaned_text: str, top_k: int = DEFAULT_TOP_K_EXPLANATIONS) -> list[FeatureContribution]:
        """Calculates token-level log-odds contributions for terms present in the email.

        Methodology:
        For each token $w_j$ in the vocabulary that appears in this email:
          contribution = TF-IDF(w_j) * [log P(w_j|Spam) - log P(w_j|Ham)]
        Tokens with positive contribution increase Spam odds; negative increase Ham odds.

        Disclaimer:
        Reflects statistical associations in training data; does NOT denote causal proof.
        """
        if not cleaned_text:
            return []

        try:
            tfidf_vec = self._tfidf_step.transform([cleaned_text])
        except Exception as exc:
            logger.warning("Could not compute TF-IDF transform for explanation: %s", exc)
            return []

        # Extract non-zero features present in this email
        _, non_zero_cols = tfidf_vec.nonzero()
        if len(non_zero_cols) == 0:
            return []

        data_weights = tfidf_vec.data
        contributions: list[FeatureContribution] = []

        for col_idx, weight in zip(non_zero_cols, data_weights):
            token = self._inv_vocab.get(int(col_idx), f"token_{col_idx}")
            diff = float(self._log_odds_diff[col_idx])
            contrib = float(weight * diff)
            assoc = "Spam-leaning" if contrib > 0 else "Ham-leaning"

            contributions.append(
                FeatureContribution(
                    token=token,
                    weight=float(weight),
                    log_odds_diff=diff,
                    contribution=contrib,
                    association=assoc,
                )
            )

        # Sort by absolute magnitude of contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions[:top_k]

    # =================================================================
    # Batch Classification
    # =================================================================

    def predict_batch(
        self,
        emails: Sequence[dict[str, Any]] | pd.DataFrame | Sequence[tuple[Any, Any]],
        explain: bool = False,
        top_k: int = DEFAULT_TOP_K_EXPLANATIONS,
    ) -> list[PredictionResult]:
        """Vectorized classification for multiple emails with strict ordering guarantees.

        Args:
            emails: Iterable of email dicts ({'subject': ..., 'body': ...}),
                    DataFrame with 'subject'/'body' (or 'text'), or tuples ((subject, body)).
            explain: If True, computes token-level explanations for each email.
            top_k: Number of top features to return per email if explain=True.

        Returns:
            List of PredictionResult objects matching the exact input order.

        Raises:
            SpamGuardBatchError: If batch exceeds size limits or records fail validation.
        """
        # Normalize input to list of (subject, body) pairs
        parsed_items: list[tuple[Any, Any]] = []

        if isinstance(emails, pd.DataFrame):
            if emails.empty:
                return []
            if "subject" in emails.columns and "body" in emails.columns:
                for _, row in emails.iterrows():
                    parsed_items.append((row["subject"], row["body"]))
            elif "text" in emails.columns:
                for _, row in emails.iterrows():
                    parsed_items.append(("", row["text"]))
            elif "body" in emails.columns:
                for _, row in emails.iterrows():
                    parsed_items.append(("", row["body"]))
            else:
                raise SpamGuardBatchError(
                    "DataFrame must contain ('subject', 'body') or 'text' columns."
                )
        elif isinstance(emails, (list, tuple)):
            if not emails:
                return []
            for item in emails:
                if isinstance(item, dict):
                    subj = item.get("subject", "")
                    body = item.get("body", item.get("text", ""))
                    parsed_items.append((subj, body))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    parsed_items.append((item[0], item[1]))
                elif isinstance(item, str):
                    parsed_items.append(("", item))
                else:
                    raise SpamGuardBatchError(
                        f"Unsupported batch item structure: {type(item)}"
                    )
        else:
            raise SpamGuardBatchError(
                f"Unsupported emails container type: {type(emails)}. Expected list, tuple, or DataFrame."
            )

        if len(parsed_items) > MAX_BATCH_SIZE:
            raise SpamGuardBatchError(
                f"Batch size {len(parsed_items):,} exceeds maximum limit of {MAX_BATCH_SIZE:,} emails."
            )

        # Preprocess and validate all items
        cleaned_texts: list[str] = []
        for idx, (subj, body) in enumerate(parsed_items):
            try:
                cleaned = self.prepare_input_text(subj, body)
                cleaned_texts.append(cleaned)
            except SpamGuardValidationError as val_err:
                raise SpamGuardBatchError(
                    f"Batch record at index {idx} failed validation: {val_err}"
                ) from val_err

        # Vectorized inference
        t0 = time.perf_counter()
        try:
            batch_preds = self._pipeline.predict(cleaned_texts)
            batch_probs = self._pipeline.predict_proba(cleaned_texts)
        except Exception as exc:
            raise SpamGuardModelError(f"Vectorized batch inference failed: {exc}") from exc
        total_duration_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_duration_ms / len(cleaned_texts) if cleaned_texts else 0.0

        # Construct results preserving exact index order
        results: list[PredictionResult] = []
        for i, cleaned in enumerate(cleaned_texts):
            pred_class = int(batch_preds[i])
            ham_prob = float(batch_probs[i, self._ham_idx])
            spam_prob = float(batch_probs[i, self._spam_idx])

            feats = self.explain_email(cleaned, top_k=top_k) if explain else []

            results.append(
                PredictionResult(
                    predicted_class=pred_class,
                    predicted_label="Spam" if pred_class == 1 else "Ham",
                    spam_probability=spam_prob,
                    ham_probability=ham_prob,
                    is_spam=(pred_class == 1),
                    model_identifier=self._model_identifier,
                    processing_time_ms=per_item_ms,
                    cleaned_char_count=len(cleaned),
                    top_features=feats,
                )
            )

        return results

    # =================================================================
    # CSV Batch Processing
    # =================================================================

    def predict_csv(
        self,
        input_csv_path: Path,
        output_csv_path: Path | None = None,
    ) -> pd.DataFrame:
        """Processes an input CSV and optionally exports results with formula injection defense."""
        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found at {input_csv_path}")

        try:
            df = pd.read_csv(input_csv_path)
        except Exception as exc:
            raise SpamGuardBatchError(f"Failed to read input CSV: {exc}") from exc

        results = self.predict_batch(df)

        # Build clean output DataFrame without echoing untrusted/sensitive input bodies
        output_rows = []
        for idx, res in enumerate(results):
            row_dict = {
                "row_id": idx,
                "predicted_class": res.predicted_class,
                "predicted_label": sanitize_csv_cell(res.predicted_label),
                "is_spam": res.is_spam,
                "spam_probability": round(res.spam_probability, 4),
                "ham_probability": round(res.ham_probability, 4),
                "processing_time_ms": round(res.processing_time_ms, 2),
            }
            output_rows.append(row_dict)

        out_df = pd.DataFrame(output_rows)

        if output_csv_path is not None:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(output_csv_path, index=False)
            logger.info("Saved batch predictions to %s", output_csv_path)

        return out_df


# =====================================================================
# Singleton Predictor & Convenience Functions
# =====================================================================

_DEFAULT_PREDICTOR: SpamPredictor | None = None


def get_default_predictor() -> SpamPredictor:
    """Returns the globally cached default SpamPredictor instance."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = SpamPredictor.load()
    return _DEFAULT_PREDICTOR


def load_model(model_path: Path = SAVED_PIPELINE_PATH) -> Pipeline:
    """Convenience function to load and return the underlying Scikit-learn Pipeline."""
    predictor = SpamPredictor.load(model_path)
    return predictor._pipeline


def predict_email(
    subject: Any = "",
    body: Any = "",
    explain: bool = False,
    predictor: SpamPredictor | None = None,
) -> PredictionResult:
    """Convenience function to classify a single email using the default predictor."""
    p = predictor or get_default_predictor()
    return p.predict_email(subject=subject, body=body, explain=explain)


def predict_batch(
    emails: Sequence[dict[str, Any]] | pd.DataFrame | Sequence[tuple[Any, Any]],
    explain: bool = False,
    predictor: SpamPredictor | None = None,
) -> list[PredictionResult]:
    """Convenience function to classify a batch of emails using the default predictor."""
    p = predictor or get_default_predictor()
    return p.predict_batch(emails=emails, explain=explain)


# =====================================================================
# CLI Interface
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SpamGuard AI — Email Spam Prediction Engine (Phase 07)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single email classification:
  python -m src.predict --subject "Prize Notification" --text "You won a cash prize! Claim now."

  # Classification via stdin:
  echo "Project meeting agenda for Monday morning" | python -m src.predict --stdin

  # Batch classification via CSV:
  python -m src.predict --batch-csv input.csv --output-csv results.csv

  # JSON formatted output with feature explanation:
  python -m src.predict --subject "Refinance" --text "Lower mortgage rate today" --explain --format json
        """,
    )
    parser.add_argument("--subject", type=str, default="", help="Email subject line")
    parser.add_argument("--body", "--text", dest="body", type=str, default="", help="Email body content")
    parser.add_argument("--stdin", action="store_true", help="Read email body from standard input")
    parser.add_argument("--batch-csv", type=Path, default=None, help="Input CSV path for batch inference")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path for batch inference")
    parser.add_argument("--explain", action="store_true", help="Show influential token contributions")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_EXPLANATIONS, help="Top K explanation tokens")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output display format")

    args = parser.parse_args()

    # Reconfigure stdout for UTF-8 compatibility on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Load predictor
    try:
        predictor = get_default_predictor()
    except Exception as exc:
        logger.error("Failed to initialize predictor: %s", exc)
        sys.exit(1)

    # 1. Batch CSV Mode
    if args.batch_csv is not None:
        try:
            out_df = predictor.predict_csv(args.batch_csv, args.output_csv)
            print(f"Batch processing complete: {len(out_df)} records evaluated.")
            print(out_df.head(10).to_string(index=False))
            sys.exit(0)
        except Exception as exc:
            logger.error("Batch CSV processing error: %s", exc)
            sys.exit(1)

    # 2. Stdin Mode
    body_text = args.body
    if args.stdin:
        if not sys.stdin.isatty():
            body_text = sys.stdin.read()
        else:
            print("Reading from stdin (Ctrl+Z then Enter on Windows, Ctrl+D on Unix to finish):")
            body_text = sys.stdin.read()

    if not args.subject and not body_text:
        parser.print_help()
        sys.exit(1)

    # 3. Single Email Prediction
    try:
        res = predictor.predict_email(
            subject=args.subject,
            body=body_text,
            explain=args.explain,
            top_k=args.top_k,
        )
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        sys.exit(1)

    # 4. Display Results
    if args.format == "json":
        print(json.dumps(res.to_dict(), indent=2))
    else:
        badge = "[SPAM]" if res.is_spam else "[HAM - Legitimate]"
        print("=" * 60)
        print(f"SpamGuard AI Classification: {badge}")
        print("=" * 60)
        print(f"Predicted Label:     {res.predicted_label} (Class {res.predicted_class})")
        print(f"Spam Probability:    {res.spam_probability * 100:.2f}%")
        print(f"Ham Probability:     {res.ham_probability * 100:.2f}%")
        print(f"Processing Time:     {res.processing_time_ms:.2f} ms")
        print(f"Cleaned Text Length: {res.cleaned_char_count:,} characters")
        print(f"Model Identifier:    {res.model_identifier}")

        if args.explain and res.top_features:
            print("\nInfluential Token Contributions (Statistical Association):")
            print(f"{'Token':<25} {'Weight':<10} {'Contribution':<15} {'Tendency'}")
            print("-" * 65)
            for feat in res.top_features:
                print(
                    f"{feat.token:<25} {feat.weight:<10.4f} "
                    f"{feat.contribution:<+15.4f} {feat.association}"
                )
            print("\n* Note: Learned statistical associations; not proof of causality.")
        print("=" * 60)


if __name__ == "__main__":
    main()
