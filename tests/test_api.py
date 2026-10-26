"""Unit and integration tests for SpamGuard AI FastAPI endpoints."""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import EXPECTED_PIPELINE_SHA256

client = TestClient(app)


def test_api_health_endpoint():
    """Verify /api/health verifies model readiness and SHA-256 integrity."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert data["model_available"] is True
    assert data["model_sha256"].lower() == EXPECTED_PIPELINE_SHA256.lower()
    assert data["model_sha256_verified"] is True
    assert "MultinomialNB" in data["model_identifier"]
    assert "os" in data["system_info"]


def test_api_predict_spam_with_explainability():
    """Verify single spam prediction with token contribution explainability."""
    payload = {
        "subject": "Exclusive Cash Prize Reward",
        "body": "Congratulations! Claim your guaranteed $5,000 free bonus money cash payout immediately at http://claim.com",
        "explain": True,
        "top_k": 5,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 1
    assert data["predicted_label"] == "Spam"
    assert data["is_spam"] is True
    assert data["spam_probability"] > 0.7
    assert data["ham_probability"] < 0.3
    assert len(data["top_features"]) > 0
    # Top features have required fields
    feat = data["top_features"][0]
    assert "token" in feat
    assert "weight" in feat
    assert "log_odds_diff" in feat
    assert "contribution" in feat
    assert "association" in feat


def test_api_predict_ham():
    """Verify legitimate software engineering update classifies as Ham."""
    payload = {
        "subject": "Sprint 34 Standup Notes & Feature Branch",
        "body": "Hi team, please find attached the meeting notes and git patch for today's review.",
        "explain": False,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 0
    assert data["predicted_label"] == "Ham"
    assert data["is_spam"] is False
    assert data["ham_probability"] > 0.8
    assert data["top_features"] == []


def test_api_predict_empty_validation():
    """Verify empty email triggers 400 Bad Request."""
    payload = {"subject": "", "body": "   "}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "empty or contains only whitespace" in response.json()["detail"]


def test_api_predict_batch_json():
    """Verify batch JSON classification processes multiple items preserving order."""
    payload = {
        "emails": [
            {
                "id": "email-01",
                "subject": "Urgent Bank Wire Transfer",
                "body": "Urgent account suspension verify your login credentials now cash prize free bonus",
            },
            {
                "id": "email-02",
                "subject": "Team Meeting",
                "body": "Hi team, attached is the sprint backlog and documentation update for Friday.",
            },
        ],
        "explain": False,
    }
    response = client.post("/api/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert data["spam_count"] >= 1
    assert data["ham_count"] >= 1
    assert len(data["items"]) == 2
    assert data["items"][0]["id"] == "email-01"
    assert data["items"][1]["id"] == "email-02"


def test_api_predict_batch_csv_upload():
    """Verify batch CSV multipart upload and formula injection defenses."""
    csv_content = (
        "subject,body\n"
        "Prize Claim,Claim your free bonus money now\n"
        "Meeting,Team code review meeting scheduled for tomorrow\n"
    )
    files = {"file": ("test_emails.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict/csv", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["items"]) == 2


def test_api_metrics_endpoint():
    """Verify /api/metrics loads genuine Phase 06 test evaluation KPIs."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation" in data
    eval_metrics = data["evaluation"]["final_test_metrics"]
    assert eval_metrics["accuracy"] == 0.973
    assert eval_metrics["spam"]["precision"] == 0.9733
    assert eval_metrics["spam"]["recall"] == 0.9375
    assert eval_metrics["spam"]["f1_score"] == 0.9551
    assert eval_metrics["roc_auc"] == 0.9965
    assert eval_metrics["confusion_matrix"]["tp"] == 255
    assert eval_metrics["confusion_matrix"]["tn"] == 610
    assert eval_metrics["confusion_matrix"]["fp"] == 7
    assert eval_metrics["confusion_matrix"]["fn"] == 17
    assert "course_metadata" in data


def test_api_samples_endpoint():
    """Verify /api/samples returns the demonstration synthetic samples."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 4
    for sample in data["samples"]:
        assert "title" in sample
        assert "category" in sample
        assert "subject" in sample
        assert "body" in sample
