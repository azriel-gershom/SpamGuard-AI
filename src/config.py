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

# Dataset Paths & Artifacts
PROCESSED_DATA_FILE: Path = PROCESSED_DATA_DIR / "emails.csv"
DATA_SUMMARY_FILE: Path = REPORTS_DIR / "data_summary.json"
EDA_SUMMARY_FILE: Path = REPORTS_DIR / "eda_summary.json"
EDA_FINDINGS_FILE: Path = REPORTS_DIR / "eda_findings.md"

# Dataset Splits (Phase 04)
SPLITS_DIR: Path = PROCESSED_DATA_DIR / "splits"
TRAIN_SPLIT_FILE: Path = SPLITS_DIR / "train.csv"
VAL_SPLIT_FILE: Path = SPLITS_DIR / "validation.csv"
TEST_SPLIT_FILE: Path = SPLITS_DIR / "test.csv"
SPLIT_SUMMARY_FILE: Path = REPORTS_DIR / "split_summary.json"

# Split Ratios & Seed
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# TF-IDF Feature Extraction Parameters
TFIDF_MAX_FEATURES: int = 20000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.98

# Corpus Archive Specifications (Apache SpamAssassin Public Corpus)
SPAMASSASSIN_BASE_URL: str = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_ARCHIVES: list[dict[str, str | int]] = [
    {"filename": "20030228_easy_ham.tar.bz2", "group": "easy_ham", "label": 0},
    {"filename": "20030228_easy_ham_2.tar.bz2", "group": "easy_ham_2", "label": 0},
    {"filename": "20030228_hard_ham.tar.bz2", "group": "hard_ham", "label": 0},
    {"filename": "20030228_spam.tar.bz2", "group": "spam", "label": 1},
    {"filename": "20050311_spam_2.tar.bz2", "group": "spam_2", "label": 1},
]

# Serialized Model Artifact Paths (Phase 05)
SAVED_PIPELINE_PATH: Path = MODELS_DIR / "spam_pipeline.joblib"
MODEL_PATH: Path = SAVED_PIPELINE_PATH  # Alias for backward compatibility
VECTORIZER_PATH: Path = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH: Path = REPORTS_DIR / "model_metrics.json"
TRAINING_SUMMARY_FILE: Path = REPORTS_DIR / "training_summary.json"
MODEL_INSIGHTS_FILE: Path = REPORTS_DIR / "model_insights.json"

# Phase 05 Hyperparameter Search Space
ALPHA_CANDIDATES: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0]

# Pipeline Configurations
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Application Metadata
APP_TITLE: str = "SpamGuard AI"
APP_SUBTITLE: str = "AI-Powered Email Spam Classification"
APP_VERSION: str = "0.1.0"
