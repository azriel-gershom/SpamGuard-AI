"""SpamGuard AI — Thin High-Performance FastAPI Inference Service.

Exposes RESTful prediction, batch classification, and verified model telemetry
using the frozen, SHA-256 authenticated SpamPredictor engine.
"""

from __future__ import annotations

import io
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.best_first_search import (
    EmailRecord,
    build_email_indicator_graph,
    get_demo_investigation_emails,
    greedy_best_first_search,
)
from src.config import (
    DEFAULT_TOP_K_EXPLANATIONS,
    EXPECTED_PIPELINE_SHA256,
    MAX_BATCH_SIZE,
    MAX_INPUT_CHARS,
    SAVED_PIPELINE_PATH,
    TEST_EVALUATION_FILE,
)
from src.predict import (
    FeatureContribution,
    PredictionResult,
    SpamGuardBatchError,
    SpamGuardModelError,
    SpamGuardValidationError,
    compute_file_sha256,
    get_default_predictor,
    sanitize_csv_cell,
)
from src.ui_helpers import SAMPLE_EMAILS

logger = logging.getLogger("SpamGuardAI.API")

# =====================================================================
# Pydantic Schemas
# =====================================================================

class FeatureContributionModel(BaseModel):
    token: str
    weight: float
    log_odds_diff: float
    contribution: float
    association: str


class PredictRequest(BaseModel):
    subject: str = Field(default="", max_length=100000)
    body: str = Field(default="", max_length=MAX_INPUT_CHARS)
    explain: bool = Field(default=True)
    top_k: int = Field(default=DEFAULT_TOP_K_EXPLANATIONS, ge=1, le=50)


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    processing_time_ms: float
    cleaned_char_count: int
    model_identifier: str
    top_features: List[FeatureContributionModel]


class BatchEmailItem(BaseModel):
    id: Optional[str] = None
    subject: str = Field(default="", max_length=100000)
    body: str = Field(default="", max_length=MAX_INPUT_CHARS)


class BatchPredictRequest(BaseModel):
    emails: List[BatchEmailItem]
    explain: bool = Field(default=False)
    top_k: int = Field(default=5, ge=1, le=20)


class BatchItemResult(BaseModel):
    id: Optional[str] = None
    subject: str
    body_preview: str
    predicted_class: int
    predicted_label: str
    is_spam: bool
    spam_probability: float
    ham_probability: float
    processing_time_ms: float
    top_features: List[FeatureContributionModel] = Field(default_factory=list)


class BatchPredictResponse(BaseModel):
    total_count: int
    spam_count: int
    ham_count: int
    spam_percentage: float
    ham_percentage: float
    total_duration_ms: float
    avg_latency_ms: float
    items: List[BatchItemResult]


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_sha256: str
    model_sha256_verified: bool
    model_identifier: str
    system_info: dict[str, str]


# =====================================================================
# Threat Investigation Explorer (Greedy Best-First Search) Schemas
# =====================================================================

class InvestigationEmailItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=100)
    subject: str = Field(default="", max_length=10000)
    body: str = Field(default="", max_length=MAX_INPUT_CHARS)


class SearchRequest(BaseModel):
    emails: List[InvestigationEmailItem]
    start_email_id: str = Field(..., min_length=1, max_length=100)
    goal_email_id: str = Field(..., min_length=1, max_length=100)


class SearchNodeModel(BaseModel):
    node_id: str
    node_type: str
    display_label: str
    heuristic: float
    explanation: str
    step_order: Optional[int] = None


class GraphNodeVis(BaseModel):
    id: str
    label: str
    type: str  # "email" | "indicator"
    indicator_type: Optional[str] = None
    preview: Optional[str] = None
    is_path: bool = False
    is_explored: bool = False
    heuristic: Optional[float] = None


class GraphEdgeVis(BaseModel):
    source: str
    target: str
    is_path: bool = False


class GraphVisData(BaseModel):
    nodes: List[GraphNodeVis]
    edges: List[GraphEdgeVis]


class SearchResponse(BaseModel):
    found: bool
    start_email_id: str
    goal_email_id: str
    path: List[SearchNodeModel]
    explored: List[SearchNodeModel]
    heuristic_values: Dict[str, float]
    graph_summary: Dict[str, int]
    search_metrics: Dict[str, Any]
    message: str
    graph_visualization: GraphVisData


# =====================================================================
# FastAPI Application Factory
# =====================================================================

app = FastAPI(
    title="SpamGuard AI — Inference Engine",
    version="1.0.0",
    description="Thin, secure RESTful API powering the Anthracite-inspired SpamGuard AI experience.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS Middleware (supports Next.js dev server on 3000, Vite on 5173, and Streamlit on 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =====================================================================
# Routes
# =====================================================================

@app.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Verifies service readiness and cryptographic model integrity."""
    model_available = SAVED_PIPELINE_PATH.exists()
    actual_sha = ""
    is_verified = False
    model_identifier = "Unavailable"

    if model_available:
        try:
            actual_sha = compute_file_sha256(SAVED_PIPELINE_PATH)
            is_verified = (actual_sha.lower() == EXPECTED_PIPELINE_SHA256.lower())
            predictor = get_default_predictor()
            model_identifier = predictor._model_identifier
        except Exception as exc:
            logger.error("Health check model inspection error: %s", exc)
            return HealthResponse(
                status="DEGRADED",
                model_available=model_available,
                model_sha256=actual_sha,
                model_sha256_verified=False,
                model_identifier=f"Error: {exc}",
                system_info={
                    "os": platform.platform(),
                    "python": platform.python_version(),
                },
            )

    return HealthResponse(
        status="OPERATIONAL" if (model_available and is_verified) else "UNAVAILABLE",
        model_available=model_available,
        model_sha256=actual_sha,
        model_sha256_verified=is_verified,
        model_identifier=model_identifier,
        system_info={
            "os": platform.platform(),
            "python": platform.python_version(),
        },
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict_single(payload: PredictRequest) -> PredictResponse:
    """Classifies a single email and returns probability and explainability telemetry."""
    try:
        predictor = get_default_predictor()
    except SpamGuardModelError as me:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model engine unavailable: {me}",
        )

    try:
        res: PredictionResult = predictor.predict_email(
            subject=payload.subject,
            body=payload.body,
            explain=payload.explain,
            top_k=payload.top_k,
        )
    except SpamGuardValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as exc:
        logger.error("Single email inference exception: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed unexpectedly.",
        )

    return PredictResponse(
        predicted_class=res.predicted_class,
        predicted_label=res.predicted_label,
        is_spam=res.is_spam,
        spam_probability=round(res.spam_probability, 4),
        ham_probability=round(res.ham_probability, 4),
        processing_time_ms=round(res.processing_time_ms, 2),
        cleaned_char_count=res.cleaned_char_count,
        model_identifier=res.model_identifier,
        top_features=[
            FeatureContributionModel(
                token=f.token,
                weight=round(f.weight, 4),
                log_odds_diff=round(f.log_odds_diff, 4),
                contribution=round(f.contribution, 4),
                association=f.association,
            )
            for f in res.top_features
        ],
    )


@app.post("/api/predict/batch", response_model=BatchPredictResponse)
def predict_batch_json(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Performs high-throughput vectorized classification on a list of email objects."""
    if not payload.emails:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list cannot be empty.",
        )
    if len(payload.emails) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(payload.emails):,} exceeds maximum limit of {MAX_BATCH_SIZE:,}.",
        )

    try:
        predictor = get_default_predictor()
    except SpamGuardModelError as me:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model engine unavailable: {me}",
        )

    items_to_predict = [(e.subject, e.body) for e in payload.emails]

    t0 = time.perf_counter()
    try:
        raw_results: list[PredictionResult] = predictor.predict_batch(
            emails=items_to_predict,
            explain=payload.explain,
            top_k=payload.top_k,
        )
    except (SpamGuardValidationError, SpamGuardBatchError) as be:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(be),
        )
    except Exception as exc:
        logger.error("Vectorized batch classification error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch classification failed: {exc}",
        )

    total_duration_ms = (time.perf_counter() - t0) * 1000.0

    items_response: list[BatchItemResult] = []
    spam_count = 0
    for idx, (res, orig) in enumerate(zip(raw_results, payload.emails)):
        if res.is_spam:
            spam_count += 1
        body_prev = orig.body[:120].strip()
        if len(orig.body) > 120:
            body_prev += "..."
        items_response.append(
            BatchItemResult(
                id=orig.id or str(idx + 1),
                subject=sanitize_csv_cell(orig.subject) if orig.subject else "(No Subject)",
                body_preview=body_prev,
                predicted_class=res.predicted_class,
                predicted_label=res.predicted_label,
                is_spam=res.is_spam,
                spam_probability=round(res.spam_probability, 4),
                ham_probability=round(res.ham_probability, 4),
                processing_time_ms=round(res.processing_time_ms, 2),
                top_features=[
                    FeatureContributionModel(
                        token=f.token,
                        weight=round(f.weight, 4),
                        log_odds_diff=round(f.log_odds_diff, 4),
                        contribution=round(f.contribution, 4),
                        association=f.association,
                    )
                    for f in res.top_features
                ],
            )
        )

    total_count = len(items_response)
    ham_count = total_count - spam_count
    spam_pct = round((spam_count / total_count) * 100.0, 2) if total_count else 0.0
    ham_pct = round((ham_count / total_count) * 100.0, 2) if total_count else 0.0

    return BatchPredictResponse(
        total_count=total_count,
        spam_count=spam_count,
        ham_count=ham_count,
        spam_percentage=spam_pct,
        ham_percentage=ham_pct,
        total_duration_ms=round(total_duration_ms, 2),
        avg_latency_ms=round(total_duration_ms / total_count, 3) if total_count else 0.0,
        items=items_response,
    )


@app.post("/api/predict/csv", response_model=BatchPredictResponse)
async def predict_batch_csv(
    file: UploadFile = File(...),
    explain: bool = Query(default=False),
) -> BatchPredictResponse:
    """Parses an uploaded CSV file, executes batch inference, and applies formula injection defenses."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid .csv file.",
        )

    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB bound
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file exceeds 50MB size limit.",
            )
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {exc}",
        )

    # Normalize column names
    col_map = {c.strip().lower(): c for c in df.columns}
    has_body = "body" in col_map or "text" in col_map
    has_subject = "subject" in col_map

    if not has_body and not has_subject:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV must contain at least a 'body' (or 'text') or 'subject' column.",
        )

    body_col = col_map.get("body", col_map.get("text", ""))
    subj_col = col_map.get("subject", "")

    batch_items: list[BatchEmailItem] = []
    for idx, row in df.iterrows():
        b_val = str(row[body_col]) if body_col else ""
        s_val = str(row[subj_col]) if subj_col else ""
        batch_items.append(
            BatchEmailItem(
                id=str(idx + 1),
                subject=s_val,
                body=b_val,
            )
        )

    return predict_batch_json(
        BatchPredictRequest(emails=batch_items, explain=explain)
    )


@app.get("/api/metrics")
def get_evaluation_metrics() -> dict[str, Any]:
    """Returns verified academic test metrics and metadata from reports/test_evaluation.json."""
    if not TEST_EVALUATION_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test evaluation report not found on disk.",
        )

    try:
        with open(TEST_EVALUATION_FILE, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation report: {exc}",
        )

    # Load training and split summaries if present
    train_summary_file = PROJECT_ROOT / "reports" / "training_summary.json"
    split_summary_file = PROJECT_ROOT / "reports" / "split_summary.json"
    perf_file = PROJECT_ROOT / "reports" / "performance_benchmark.json"

    train_data = {}
    if train_summary_file.exists():
        try:
            with open(train_summary_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
        except Exception:
            pass

    split_data = {}
    if split_summary_file.exists():
        try:
            with open(split_summary_file, "r", encoding="utf-8") as f:
                split_data = json.load(f)
        except Exception:
            pass

    perf_data = {}
    if perf_file.exists():
        try:
            with open(perf_file, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
        except Exception:
            pass

    return {
        "evaluation": eval_data,
        "training": train_data,
        "splits": split_data,
        "benchmarks": perf_data,
        "course_metadata": {
            "title": "SpamGuard AI — AI-Based Spam Email Classifier",
            "course": "Semester 3 AI Micro Project – 05",
            "competencies": [
                {
                    "code": "CO2",
                    "title": "Informed Search & Algorithm Engineering",
                    "description": "Greedy Best-First Search on email-indicator bipartite graphs with Jaccard structural heuristics, alongside TF-IDF and Naive Bayes.",
                },
                {
                    "code": "CO4",
                    "title": "Model Engineering",
                    "description": "Leak-free Scikit-learn Pipeline with unigram/bigram TF-IDF vectorization and Lidstone smoothing (alpha=0.1).",
                },
                {
                    "code": "CO5",
                    "title": "Evaluation Metrics",
                    "description": "Multi-metric statistical evaluation on held-out test split (n=889) verifying Precision, Recall, F1, and ROC-AUC.",
                },
            ],
            "team_roles": [
                {
                    "name": "Samuvel Raj R",
                    "role": "Systems Architect & Defensive Engineering",
                    "responsibility": "Phase 06-10: Inference engine, FastAPI integration, Streamlit hardening, security audits",
                },
                {
                    "name": "Azriel Gershom Raj",
                    "role": "Machine Learning & Pipeline Engineer",
                    "responsibility": "Phase 01-05: Ingestion pipeline, EDA, leakage-safe text preprocessing, Naïve Bayes model tuning",
                },
                {
                    "name": "CO4 Model Engineering Specialist",
                    "role": "Data Verification & Feature Engineering",
                    "responsibility": "Stratified content-group splitting, MIME sanitization, and vocabulary isolation",
                },
                {
                    "name": "CO5 Quality & Evaluation Specialist",
                    "role": "Statistical Validation & Benchmark QA",
                    "responsibility": "Held-out test partition certification, baseline benchmarking, and hardware profiling",
                },
            ],
        },
    }


@app.get("/api/samples")
def get_samples() -> dict[str, Any]:
    """Returns the verified synthetic email samples library for instant demonstrations."""
    # Format samples cleanly for client cards
    formatted = []
    for title, s in SAMPLE_EMAILS.items():
        if s.get("category") == "None":
            continue
        formatted.append({
            "key": title,
            "title": title.replace("🚨 ", "").replace("✅ ", ""),
            "category": s.get("category"),
            "subject": s.get("subject"),
            "body": s.get("body"),
            "description": s.get("description"),
        })
    return {"samples": formatted}


# =====================================================================
# Threat Investigation Explorer (Greedy Best-First Search) Endpoints
# =====================================================================

@app.get("/api/search/demo-emails")
def get_search_demo_emails() -> dict[str, Any]:
    """Returns synthetic demonstration emails for Threat Investigation Explorer."""
    return {"emails": get_demo_investigation_emails()}


@app.post("/api/search/best-first", response_model=SearchResponse)
def search_best_first(payload: SearchRequest) -> SearchResponse:
    """Executes Greedy Best-First Search on the email-indicator bipartite graph."""
    if not payload.emails:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The email collection must contain at least one email record.",
        )

    if len(payload.emails) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum allowed investigation emails is 100 (received {len(payload.emails)}).",
        )

    # Check ID uniqueness
    email_ids = [e.id.strip() for e in payload.emails]
    if len(email_ids) != len(set(email_ids)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duplicate email identifier detected. All email IDs must be unique.",
        )

    start_clean = payload.start_email_id.strip()
    goal_clean = payload.goal_email_id.strip()

    if start_clean not in set(email_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Start email ID '{start_clean}' does not match any provided email records.",
        )

    if goal_clean not in set(email_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target email ID '{goal_clean}' does not match any provided email records.",
        )

    # Build EmailRecords and graph
    records = [
        EmailRecord(id=e.id.strip(), subject=e.subject or "", body=e.body or "")
        for e in payload.emails
    ]
    graph = build_email_indicator_graph(records)

    # Execute Greedy Best-First Search
    res = greedy_best_first_search(graph, start_clean, goal_clean)

    # Convert path and explored nodes to SearchNodeModel
    path_models = [
        SearchNodeModel(
            node_id=p.node_id,
            node_type=p.node_type,
            display_label=p.display_label,
            heuristic=p.heuristic,
            explanation=p.explanation,
            step_order=p.step_order,
        )
        for p in res.path
    ]

    explored_models = [
        SearchNodeModel(
            node_id=e.node_id,
            node_type=e.node_type,
            display_label=e.display_label,
            heuristic=e.heuristic,
            explanation=e.explanation,
            step_order=e.step_order,
        )
        for e in res.explored
    ]

    # Prepare graph visualization payload
    path_node_ids = {p.node_id for p in res.path}
    explored_node_ids = {e.node_id for e in res.explored}

    path_edge_set: set[tuple[str, str]] = set()
    if len(res.path) > 1:
        for idx in range(len(res.path) - 1):
            u, v = res.path[idx].node_id, res.path[idx + 1].node_id
            path_edge_set.add((min(u, v), max(u, v)))

    vis_nodes: List[GraphNodeVis] = []
    for nid, node in graph.nodes.items():
        h_val = res.heuristic_values.get(nid)
        vis_nodes.append(
            GraphNodeVis(
                id=nid,
                label=node.display_label,
                type=node.node_type,
                indicator_type=node.indicator_type,
                preview=node.email_preview,
                is_path=nid in path_node_ids,
                is_explored=nid in explored_node_ids,
                heuristic=h_val,
            )
        )

    vis_edges: List[GraphEdgeVis] = []
    seen_edges: set[tuple[str, str]] = set()
    for source_id, neighbors in graph.adj.items():
        for target_id in neighbors:
            edge_key = (min(source_id, target_id), max(source_id, target_id))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                vis_edges.append(
                    GraphEdgeVis(
                        source=source_id,
                        target=target_id,
                        is_path=edge_key in path_edge_set,
                    )
                )

    return SearchResponse(
        found=res.found,
        start_email_id=res.start_email_id,
        goal_email_id=res.goal_email_id,
        path=path_models,
        explored=explored_models,
        heuristic_values=res.heuristic_values,
        graph_summary=res.graph_summary,
        search_metrics=res.search_metrics,
        message=res.message,
        graph_visualization=GraphVisData(nodes=vis_nodes, edges=vis_edges),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
