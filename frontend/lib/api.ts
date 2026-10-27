/**
 * SpamGuard AI — Client API SDK
 * Connects frontend directly to the thin FastAPI backend with typed error handling and fallback support.
 */

export interface FeatureContribution {
  token: string;
  weight: number;
  log_odds_diff: number;
  contribution: number;
  association: "Spam-leaning" | "Ham-leaning" | string;
}

export interface PredictionResponse {
  predicted_class: 0 | 1;
  predicted_label: "Ham" | "Spam";
  is_spam: boolean;
  spam_probability: number;
  ham_probability: number;
  processing_time_ms: number;
  cleaned_char_count: number;
  model_identifier: string;
  top_features: FeatureContribution[];
}

export interface BatchItemResult {
  id?: string;
  subject: string;
  body_preview: string;
  predicted_class: number;
  predicted_label: "Ham" | "Spam";
  is_spam: boolean;
  spam_probability: number;
  ham_probability: number;
  processing_time_ms: number;
  top_features: FeatureContribution[];
}

export interface BatchResponse {
  total_count: number;
  spam_count: number;
  ham_count: number;
  spam_percentage: number;
  ham_percentage: number;
  total_duration_ms: number;
  avg_latency_ms: number;
  items: BatchItemResult[];
}

export interface SystemHealth {
  status: "OPERATIONAL" | "DEGRADED" | "UNAVAILABLE";
  model_available: boolean;
  model_sha256: string;
  model_sha256_verified: boolean;
  model_identifier: string;
  system_info: {
    os: string;
    python: string;
  };
}

export interface SyntheticSample {
  key: string;
  title: string;
  category: "Spam" | "Ham";
  subject: string;
  body: string;
  description: string;
}

export interface EvaluationMetricsData {
  evaluation: {
    metadata: {
      project_name: string;
      model_type: string;
      model_artifact_sha256: string;
      test_split_sha256: string;
      test_sample_count: number;
    };
    final_test_metrics: {
      accuracy: number;
      balanced_accuracy: number;
      specificity: number;
      false_positive_rate: number;
      false_negative_rate: number;
      spam: {
        precision: number;
        recall: number;
        f1_score: number;
        support: number;
      };
      ham: {
        precision: number;
        recall: number;
        f1_score: number;
        support: number;
      };
      roc_auc: number;
      average_precision: number;
      confusion_matrix: {
        tn: number;
        fp: number;
        fn: number;
        tp: number;
        matrix: number[][];
      };
      total_samples: number;
    };
    majority_class_baseline: {
      accuracy: number;
      roc_auc: number;
      spam: {
        f1_score: number;
      };
      confusion_matrix: {
        tn: number;
        fp: number;
        fn: number;
        tp: number;
      };
    };
  };
  course_metadata: {
    title: string;
    course: string;
    competencies: Array<{
      code: string;
      title: string;
      description: string;
    }>;
    team_roles: Array<{
      name: string;
      role: string;
      responsibility: string;
    }>;
  };
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health check failed: HTTP ${res.status}`);
  }
  return res.json();
}

export async function classifyEmail(
  subject: string,
  body: string,
  explain = true,
  top_k = 10
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject, body, explain, top_k }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Inference failed" }));
    throw new Error(err.detail || `Classification error: ${res.status}`);
  }
  return res.json();
}

export async function classifyBatchJson(
  emails: Array<{ id?: string; subject: string; body: string }>,
  explain = false
): Promise<BatchResponse> {
  const res = await fetch(`${API_BASE}/api/predict/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ emails, explain }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Batch inference failed" }));
    throw new Error(err.detail || `Batch error: ${res.status}`);
  }
  return res.json();
}

export async function classifyBatchCsv(
  file: File,
  explain = false
): Promise<BatchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/predict/csv?explain=${explain}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "CSV upload failed" }));
    throw new Error(err.detail || `CSV error: ${res.status}`);
  }
  return res.json();
}

export async function fetchMetrics(): Promise<EvaluationMetricsData> {
  const res = await fetch(`${API_BASE}/api/metrics`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Metrics error: ${res.status}`);
  }
  return res.json();
}

export async function fetchSamples(): Promise<{ samples: SyntheticSample[] }> {
  const res = await fetch(`${API_BASE}/api/samples`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Samples error: ${res.status}`);
  }
  return res.json();
}

// =====================================================================
// Threat Investigation Explorer (Greedy Best-First Search) Types & SDK
// =====================================================================

export interface InvestigationEmailItem {
  id: string;
  subject: string;
  body: string;
}

export interface SearchNodeModel {
  node_id: string;
  node_type: "email" | "indicator";
  display_label: string;
  heuristic: number;
  explanation: string;
  step_order?: number;
}

export interface GraphNodeVis {
  id: string;
  label: string;
  type: "email" | "indicator";
  indicator_type?: "domain" | "url" | "email";
  preview?: string;
  is_path: boolean;
  is_explored: boolean;
  heuristic?: number;
}

export interface GraphEdgeVis {
  source: string;
  target: string;
  is_path: boolean;
}

export interface SearchResponse {
  found: boolean;
  start_email_id: string;
  goal_email_id: string;
  path: SearchNodeModel[];
  explored: SearchNodeModel[];
  heuristic_values: Record<string, number>;
  graph_summary: {
    total_nodes: number;
    total_edges: number;
    email_count: number;
    indicator_count: number;
  };
  search_metrics: {
    nodes_explored: number;
    path_length: number;
    max_frontier_size: number;
  };
  message: string;
  graph_visualization: {
    nodes: GraphNodeVis[];
    edges: GraphEdgeVis[];
  };
}

export async function fetchSearchDemoEmails(): Promise<{ emails: InvestigationEmailItem[] }> {
  const res = await fetch(`${API_BASE}/api/search/demo-emails`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Demo emails error: ${res.status}`);
  }
  return res.json();
}

export async function runGreedyBestFirstSearch(
  emails: InvestigationEmailItem[],
  start_email_id: string,
  goal_email_id: string
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/api/search/best-first`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ emails, start_email_id, goal_email_id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Search failed" }));
    throw new Error(err.detail || `Search error: ${res.status}`);
  }
  return res.json();
}
