"""SpamGuard AI — UI Helpers and Interactive Visualizations for Streamlit.

Phase 08:
- Provides cached metrics extraction from verified academic reports.
- Supplies realistic synthetic email samples for instant demonstration.
- Builds interactive Plotly figures (probability gauge, confusion matrix heatmap, baseline comparison).
- Encapsulates UI formatting, CSS styles, and model readiness status checks.
- Completely avoids modifying or retraining machine learning models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    EXPECTED_PIPELINE_SHA256,
    SAVED_PIPELINE_PATH,
    TEST_EVALUATION_FILE,
)
from src.predict import FeatureContribution, compute_file_sha256

logger = logging.getLogger("SpamGuardAI.UIHelpers")

# =====================================================================
# Theme Color Palette (Cybersecurity / Dark Aesthetic)
# =====================================================================
COLOR_BG_DARK = "#0b0f19"
COLOR_CARD_BG = "#131b2e"
COLOR_CARD_BORDER = "#1e293b"
COLOR_HAM = "#10b981"       # Emerald / Safe
COLOR_SPAM = "#ef4444"      # Crimson / Threat
COLOR_NEUTRAL = "#3b82f6"   # Cyan / Blue
COLOR_ACCENT = "#8b5cf6"    # Violet / Accent
COLOR_MUTED = "#94a3b8"     # Slate 400
COLOR_TEXT = "#f8fafc"      # White / Slate 50


# =====================================================================
# Synthetic Sample Library for Quick Demonstration
# =====================================================================
SAMPLE_EMAILS: dict[str, dict[str, str]] = {
    "Select a pre-configured sample...": {
        "subject": "",
        "body": "",
        "category": "None",
        "description": "Custom user input",
    },
    "🚨 Urgent Account Phishing Scam (Spam)": {
        "subject": "URGENT: Unauthorized access detected on your account!",
        "body": (
            "Dear valued customer,\n\n"
            "We have detected suspicious login attempts on your account from an unrecognized IP address. "
            "To prevent permanent suspension of your banking access, you must immediately verify your credentials.\n\n"
            "Click here to verify your identity now: http://secure-verify-update.phishing-test.com/login\n\n"
            "Failure to verify within 24 hours will result in automatic account termination.\n\n"
            "Security Team, Global Trust Bank."
        ),
        "category": "Spam",
        "description": "Urgency manipulation, credential theft lure, and suspicious link.",
    },
    "🚨 Promotional Cash Prize Notification (Spam)": {
        "subject": "Congratulations! You won free cash prize money and bonus offer",
        "body": (
            "You have been selected as the guaranteed winner of our exclusive $10,000 cash sweepstakes prize! "
            "Claim your free bonus money today with 100% guarantee payout.\n\n"
            "Visit our claim form below to receive your immediate wire transfer:\n"
            "http://claim-free-prize-bonus.com/redeem?id=99283\n\n"
            "Act fast! This limited bonus offer expires midnight tonight."
        ),
        "category": "Spam",
        "description": "Financial inducement, promotional hype, and unsolicited reward claim.",
    },
    "✅ Team Engineering Standup Update (Ham)": {
        "subject": "Sprint 34 Standup Notes & Feature Branch Code Review",
        "body": (
            "Hi team,\n\n"
            "Here is the summary of yesterday's sprint retrospective and today's agenda:\n"
            "1. Sarah merged the authentication pipeline refactoring patch.\n"
            "2. Dave is addressing unit test edge cases on the reporting module.\n"
            "3. The staging server deployment is scheduled for 4:00 PM today.\n\n"
            "Please review the open pull request on git and submit your comments before the sync meeting.\n\n"
            "Best regards,\nAlex Chen\nLead Software Engineer"
        ),
        "category": "Ham",
        "description": "Professional engineering communication, developer terminology, and code review.",
    },
    "✅ Academic Micro Project Submission (Ham)": {
        "subject": "AI Micro Project 05: SpamGuard AI Documentation & Report",
        "body": (
            "Respected Professor,\n\n"
            "Please find attached the final project documentation and evaluation metrics for "
            "Micro Project 05 (AI-Based Spam Email Classifier).\n\n"
            "We have evaluated the Multinomial Naïve Bayes pipeline against the held-out Apache SpamAssassin "
            "test partition, achieving 97.30% accuracy, 97.33% spam precision, and 95.51% F1-score.\n\n"
            "All unit test suites are passing with zero warnings.\n\n"
            "Sincerely,\nStudent Project Team"
        ),
        "category": "Ham",
        "description": "Formal academic correspondence referencing project deliverables.",
    },
}


# =====================================================================
# Model Status & Report Loaders
# =====================================================================

@st.cache_data(show_spinner=False)
def load_evaluation_summary(report_path: Path = TEST_EVALUATION_FILE) -> dict[str, Any] | None:
    """Safely loads genuine Phase 06 test evaluation metrics from disk."""
    if not report_path.exists():
        logger.warning("Test evaluation summary not found at %s", report_path)
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        logger.error("Failed to parse evaluation summary JSON: %s", exc)
        return None


@st.cache_data(show_spinner=False)
def get_model_status(
    model_path: Path = SAVED_PIPELINE_PATH,
    expected_sha: str = EXPECTED_PIPELINE_SHA256,
) -> dict[str, Any]:
    """Inspects the local model file and verifies its cryptographic integrity."""
    if not model_path.exists():
        return {
            "available": False,
            "status": "Missing Artifact",
            "sha256": "None",
            "checksum_match": False,
            "path": str(model_path),
        }

    try:
        actual_sha = compute_file_sha256(model_path)
        checksum_match = (actual_sha.lower() == expected_sha.lower())
        return {
            "available": True,
            "status": "Online & Authenticated" if checksum_match else "Integrity Failed",
            "sha256": actual_sha,
            "checksum_match": checksum_match,
            "path": str(model_path),
        }
    except Exception as exc:
        return {
            "available": False,
            "status": f"Read Error ({exc})",
            "sha256": "Error",
            "checksum_match": False,
            "path": str(model_path),
        }


# =====================================================================
# Interactive Plotly Visualizations
# =====================================================================

def create_gauge_chart(spam_probability: float) -> go.Figure:
    """Builds a responsive Plotly indicator gauge showing Spam Probability."""
    prob_pct = spam_probability * 100.0
    bar_color = COLOR_SPAM if prob_pct >= 50.0 else COLOR_HAM

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 38, "color": COLOR_TEXT, "family": "Inter, sans-serif"}},
            title={
                "text": "Estimated Spam Probability",
                "font": {"size": 15, "color": COLOR_MUTED, "family": "Inter, sans-serif"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLOR_MUTED,
                    "tickfont": {"size": 11, "color": COLOR_MUTED},
                },
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 1,
                "bordercolor": COLOR_CARD_BORDER,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [30, 70], "color": "rgba(59, 130, 246, 0.15)"},
                    {"range": [70, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": prob_pct,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin={"t": 35, "b": 15, "l": 25, "r": 25},
    )
    return fig


def create_confusion_matrix_chart(cm_data: dict[str, Any]) -> go.Figure:
    """Builds an interactive Plotly heatmap from the genuine Phase 06 test confusion matrix."""
    tn = cm_data.get("tn", 610)
    fp = cm_data.get("fp", 7)
    fn = cm_data.get("fn", 17)
    tp = cm_data.get("tp", 255)
    total = tn + fp + fn + tp

    z_matrix = [[tn, fp], [fn, tp]]
    text_annotations = [
        [
            f"<b>True Negative (TN)</b><br>{tn:,} emails<br>({(tn/total)*100:.1f}%)<br><i>Legitimate Allowed</i>",
            f"<b>False Positive (FP)</b><br>{fp:,} emails<br>({(fp/total)*100:.1f}%)<br><i>Type I Error</i>",
        ],
        [
            f"<b>False Negative (FN)</b><br>{fn:,} emails<br>({(fn/total)*100:.1f}%)<br><i>Type II Error</i>",
            f"<b>True Positive (TP)</b><br>{tp:,} emails<br>({(tp/total)*100:.1f}%)<br><i>Spam Filtered</i>",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=["Predicted Ham (0)", "Predicted Spam (1)"],
            y=["Actual Ham (0)", "Actual Spam (1)"],
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 13, "color": COLOR_TEXT, "family": "Inter, sans-serif"},
            colorscale=[
                [0.0, "#0f172a"],
                [0.1, "#1e3a8a"],
                [0.5, "#2563eb"],
                [1.0, "#38bdf8"],
            ],
            showscale=False,
        )
    )

    fig.update_layout(
        title={
            "text": "<b>Held-Out Test Confusion Matrix (n = 889 emails)</b>",
            "font": {"size": 15, "color": COLOR_TEXT},
            "x": 0.5,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin={"t": 50, "b": 30, "l": 50, "r": 30},
        xaxis={"tickfont": {"size": 12, "color": COLOR_TEXT}},
        yaxis={"tickfont": {"size": 12, "color": COLOR_TEXT}, "autorange": "reversed"},
    )
    return fig


def create_baseline_comparison_chart(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> go.Figure:
    """Builds an interactive Plotly bar chart comparing Naïve Bayes against the majority baseline."""
    categories = [
        "Overall Accuracy",
        "Spam Precision",
        "Spam Recall",
        "Spam F1-Score",
        "Balanced Accuracy",
    ]

    model_vals = [
        model_metrics.get("accuracy", 0.973) * 100,
        model_metrics.get("spam", {}).get("precision", 0.9733) * 100,
        model_metrics.get("spam", {}).get("recall", 0.9375) * 100,
        model_metrics.get("spam", {}).get("f1_score", 0.9551) * 100,
        model_metrics.get("balanced_accuracy", 0.9631) * 100,
    ]

    baseline_vals = [
        baseline_metrics.get("accuracy", 0.694) * 100,
        baseline_metrics.get("spam", {}).get("precision", 0.0) * 100,
        baseline_metrics.get("spam", {}).get("recall", 0.0) * 100,
        baseline_metrics.get("spam", {}).get("f1_score", 0.0) * 100,
        baseline_metrics.get("balanced_accuracy", 0.50) * 100,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="SpamGuard AI (MultinomialNB)",
            x=categories,
            y=model_vals,
            marker_color=COLOR_NEUTRAL,
            text=[f"{v:.1f}%" for v in model_vals],
            textposition="auto",
            textfont={"size": 11, "color": "#ffffff"},
        )
    )
    fig.add_trace(
        go.Bar(
            name="Majority-Class Baseline (Always Ham)",
            x=categories,
            y=baseline_vals,
            marker_color=COLOR_MUTED,
            text=[f"{v:.1f}%" for v in baseline_vals],
            textposition="auto",
            textfont={"size": 11, "color": "#ffffff"},
        )
    )

    fig.update_layout(
        barmode="group",
        title={
            "text": "<b>Classifier vs. Zero-Intelligence Baseline (Held-Out Test Set)</b>",
            "font": {"size": 15, "color": COLOR_TEXT},
            "x": 0.5,
        },
        yaxis={
            "title": "Performance (%)",
            "range": [0, 115],
            "tickfont": {"color": COLOR_MUTED},
            "gridcolor": COLOR_CARD_BORDER,
        },
        xaxis={"tickfont": {"color": COLOR_TEXT, "size": 11}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.25,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": COLOR_TEXT},
        },
        margin={"t": 45, "b": 60, "l": 40, "r": 20},
    )
    return fig


def format_features_dataframe(features: list[FeatureContribution]) -> pd.DataFrame:
    """Converts a list of FeatureContribution objects into a formatted DataFrame."""
    if not features:
        return pd.DataFrame(columns=["Token", "TF-IDF Weight", "Log-Odds Contribution", "Learned Tendency"])

    rows = []
    for f in features:
        rows.append({
            "Token": f.token,
            "TF-IDF Weight": f"{f.weight:.4f}",
            "Log-Odds Contribution": f"{f.contribution:+.4f}",
            "Learned Tendency": f.association,
        })
    return pd.DataFrame(rows)
