"""Configuration settings and path management for SpamGuard AI."""

from pathlib import Path

# Base Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = BASE_DIR / "src"
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_DIR: Path = BASE_DIR / "docs"

# Serialized Model Artifact Paths
MODEL_PATH: Path = MODELS_DIR / "spam_classifier_nb.joblib"
VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH: Path = REPORTS_DIR / "model_metrics.json"

# Pipeline Configurations
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Application Metadata
APP_TITLE: str = "SpamGuard AI"
APP_SUBTITLE: str = "AI-Powered Email Spam Classification"
APP_VERSION: str = "0.1.0"
