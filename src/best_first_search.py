"""SpamGuard AI — Threat Investigation Explorer & Greedy Best-First Search Engine.

Implements an informed graph search algorithm (Learning Unit II / CO2 Alignment)
over an email-indicator bipartite graph.

Educational & Scientific Clarification:
- Algorithm: Greedy Best-First Search (Priority(n) = h(n)).
- Heuristic: Structural Jaccard dissimilarity h(n) = 1 - J(I(n), I(goal)) to target.
- Purpose: Discovers connection paths through shared digital artifacts (URLs, domains, emails).
- Disclaimer: Graph connection indicates shared structural indicators, NOT proof of malice.
"""

from __future__ import annotations

import heapq
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("SpamGuardAI.Search")

# =====================================================================
# Indicator Extraction & Normalization
# =====================================================================

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)

# Robust URL extractor matching http and https
URL_REGEX = re.compile(
    r"https?://[^\s<>'\"{}|\\^`]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedIndicator:
    """Represents an extracted and normalized indicator node."""

    indicator_type: str  # "domain" | "url" | "email"
    value: str           # Normalized string value

    @property
    def node_id(self) -> str:
        """Unique bipartite graph node identifier."""
        return f"{self.indicator_type}:{self.value}"

    @property
    def display_label(self) -> str:
        """Human-readable display representation."""
        return f"{self.indicator_type.upper()}: {self.value}"


def normalize_hostname(hostname: str) -> str:
    """Normalizes hostnames and domain names consistently."""
    cleaned = hostname.strip().lower()
    # Strip leading/trailing dots and ports if present
    cleaned = cleaned.split(":")[0].strip(".")
    return cleaned


def normalize_url(raw_url: str) -> Tuple[str, str]:
    """Cleans a raw URL and returns (normalized_url, normalized_domain)."""
    # Strip trailing punctuation often adjacent to URLs in prose
    cleaned = raw_url.strip().rstrip(".,;:?!()[]{}'\"<>")
    try:
        parsed = urlparse(cleaned)
        scheme = (parsed.scheme or "http").lower()
        netloc = normalize_hostname(parsed.netloc)
        path = parsed.path.rstrip("/") if parsed.path and parsed.path != "/" else ""
        norm_url = f"{scheme}://{netloc}{path}" if path else f"{scheme}://{netloc}"
        return norm_url, netloc
    except Exception:
        clean_fallback = cleaned.lower()
        return clean_fallback, clean_fallback


def extract_indicators(text: str) -> Set[ExtractedIndicator]:
    """Extracts and normalizes URLs, domains, and email addresses from text."""
    indicators: Set[ExtractedIndicator] = set()
    if not text or not isinstance(text, str):
        return indicators

    # 1. Extract email addresses and their domain components
    for match in EMAIL_REGEX.finditer(text):
        raw_email = match.group(0).strip().lower()
        if "@" in raw_email:
            local, domain = raw_email.split("@", 1)
            domain = normalize_hostname(domain)
            if domain and "." in domain:
                indicators.add(ExtractedIndicator(indicator_type="email", value=f"{local}@{domain}"))
                indicators.add(ExtractedIndicator(indicator_type="domain", value=domain))

    # 2. Extract URLs and hostnames
    for match in URL_REGEX.finditer(text):
        raw_url = match.group(0)
        norm_url, hostname = normalize_url(raw_url)
        if hostname and "." in hostname:
            indicators.add(ExtractedIndicator(indicator_type="url", value=norm_url))
            indicators.add(ExtractedIndicator(indicator_type="domain", value=hostname))

    return indicators


# =====================================================================
# Bipartite Graph Data Structure
# =====================================================================

@dataclass
class EmailRecord:
    """Represents a supplied email document."""

    id: str
    subject: str = ""
    body: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.body}".strip()

    @property
    def display_name(self) -> str:
        subj = self.subject.strip()
        if subj:
            return f"{self.id} ({subj[:30]}...)" if len(subj) > 30 else f"{self.id} ({subj})"
        return self.id


@dataclass
class GraphNode:
    """Represents a node in the bipartite email-indicator graph."""

    node_id: str
    node_type: str  # "email" | "indicator"
    display_label: str
    indicator_type: Optional[str] = None  # "domain" | "url" | "email" if indicator
    email_preview: Optional[str] = None   # snippet if email


class EmailIndicatorGraph:
    """Undirected bipartite graph connecting email nodes to indicator nodes."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.adj: Dict[str, Set[str]] = {}
        self.email_indicators: Dict[str, Set[str]] = {}  # email_node_id -> set of indicator_node_ids
        self.email_ids: Set[str] = set()

    def add_email(self, email: EmailRecord) -> str:
        """Adds an email node and extracts its connecting indicators."""
        email_node_id = f"email:{email.id}"
        self.email_ids.add(email_node_id)

        preview = email.full_text[:120] + "..." if len(email.full_text) > 120 else email.full_text
        self.nodes[email_node_id] = GraphNode(
            node_id=email_node_id,
            node_type="email",
            display_label=email.display_name,
            email_preview=preview,
        )
        if email_node_id not in self.adj:
            self.adj[email_node_id] = set()

        # Extract indicators
        extracted = extract_indicators(email.full_text)
        indicator_ids: Set[str] = set()

        for ind in extracted:
            ind_node_id = ind.node_id
            indicator_ids.add(ind_node_id)

            if ind_node_id not in self.nodes:
                self.nodes[ind_node_id] = GraphNode(
                    node_id=ind_node_id,
                    node_type="indicator",
                    display_label=ind.display_label,
                    indicator_type=ind.indicator_type,
                )
            if ind_node_id not in self.adj:
                self.adj[ind_node_id] = set()

            # Undirected edges between email and indicator
            self.adj[email_node_id].add(ind_node_id)
            self.adj[ind_node_id].add(email_node_id)

        self.email_indicators[email_node_id] = indicator_ids
        return email_node_id

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_edges(self) -> int:
        # Sum of degrees / 2 for undirected graph
        return sum(len(neighbors) for neighbors in self.adj.values()) // 2

    @property
    def email_count(self) -> int:
        return len(self.email_ids)

    @property
    def indicator_count(self) -> int:
        return len(self.nodes) - len(self.email_ids)

    def summary(self) -> Dict[str, int]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "email_count": self.email_count,
            "indicator_count": self.indicator_count,
        }


def build_email_indicator_graph(emails: List[EmailRecord]) -> EmailIndicatorGraph:
    """Constructs a clean bipartite graph from a list of EmailRecord instances."""
    graph = EmailIndicatorGraph()
    for em in emails:
        graph.add_email(em)
    return graph


# =====================================================================
# Heuristic Function: Structural Target Similarity
# =====================================================================

def calculate_heuristic(
    node_id: str,
    graph: EmailIndicatorGraph,
    goal_email_node_id: str,
) -> Tuple[float, str]:
    """Calculates the documented structural heuristic h(n) toward goal_email_node_id.

    Definition:
    - Target email: h(goal) = 0.0.
    - Other email n: h(n) = 1.0 - J(I(n), I(goal)), where J is Jaccard similarity.
      If union is empty, h(n) = 1.0.
    - Indicator node i: h(i) = 0.0 if i in I(goal) else 1.0.

    Range: [0.0, 1.0]. Lower values indicate greater structural proximity to the goal.
    """
    if node_id == goal_email_node_id:
        return 0.0, "Target email node reached (h = 0.0)"

    goal_indicators = graph.email_indicators.get(goal_email_node_id, set())

    node = graph.nodes.get(node_id)
    if not node:
        return 1.0, "Unknown node (h = 1.0)"

    if node.node_type == "email":
        curr_indicators = graph.email_indicators.get(node_id, set())
        union = curr_indicators | goal_indicators
        if not union:
            return 1.0, "No extractable indicators in current or target email (h = 1.0)"

        intersection = curr_indicators & goal_indicators
        jaccard = len(intersection) / len(union)
        h_val = round(1.0 - jaccard, 4)
        explanation = (
            f"Email node: Jaccard similarity J={len(intersection)}/{len(union)} "
            f"({jaccard:.2f}) -> h = 1 - J = {h_val}"
        )
        return h_val, explanation

    elif node.node_type == "indicator":
        if node_id in goal_indicators:
            return 0.0, f"Indicator directly present in target email '{goal_email_node_id}' (h = 0.0)"
        else:
            return 1.0, f"Indicator not directly present in target email '{goal_email_node_id}' (h = 1.0)"

    return 1.0, "Default dissimilarity heuristic (h = 1.0)"


# =====================================================================
# Greedy Best-First Search Algorithm
# =====================================================================

@dataclass
class SearchNodeResult:
    """Represents a node visited or on the reconstructed path."""

    node_id: str
    node_type: str
    display_label: str
    heuristic: float
    explanation: str
    step_order: Optional[int] = None


@dataclass
class SearchResult:
    """Complete structured output from Greedy Best-First Search."""

    found: bool
    start_email_id: str
    goal_email_id: str
    path: List[SearchNodeResult]
    explored: List[SearchNodeResult]
    heuristic_values: Dict[str, float]
    graph_summary: Dict[str, int]
    search_metrics: Dict[str, Any]
    message: str


def greedy_best_first_search(
    graph: EmailIndicatorGraph,
    start_email_id: str,
    goal_email_id: str,
) -> SearchResult:
    """Executes pure Greedy Best-First Search on the email-indicator graph.

    Academic Integrity Note:
    - Priority = h(n).
    - It does NOT add path cost g(n) (which would make it A*).
    - Uses deterministic tie-breaking (h_value, entry_counter, node_id).
    """
    start_node_id = start_email_id if start_email_id.startswith("email:") else f"email:{start_email_id}"
    goal_node_id = goal_email_id if goal_email_id.startswith("email:") else f"email:{goal_email_id}"

    # Validation
    if start_node_id not in graph.nodes:
        return SearchResult(
            found=False,
            start_email_id=start_email_id,
            goal_email_id=goal_email_id,
            path=[],
            explored=[],
            heuristic_values={},
            graph_summary=graph.summary(),
            search_metrics={"nodes_explored": 0, "path_length": 0, "max_frontier_size": 0},
            message=f"Starting email '{start_email_id}' was not found in the graph.",
        )

    if goal_node_id not in graph.nodes:
        return SearchResult(
            found=False,
            start_email_id=start_email_id,
            goal_email_id=goal_email_id,
            path=[],
            explored=[],
            heuristic_values={},
            graph_summary=graph.summary(),
            search_metrics={"nodes_explored": 0, "path_length": 0, "max_frontier_size": 0},
            message=f"Target email '{goal_email_id}' was not found in the graph.",
        )

    # Calculate initial heuristic
    h_start, expl_start = calculate_heuristic(start_node_id, graph, goal_node_id)
    heuristic_values: Dict[str, float] = {start_node_id: h_start}

    # Special Case: Start is Goal
    if start_node_id == goal_node_id:
        start_node = graph.nodes[start_node_id]
        single_result = SearchNodeResult(
            node_id=start_node_id,
            node_type=start_node.node_type,
            display_label=start_node.display_label,
            heuristic=h_start,
            explanation=expl_start,
            step_order=1,
        )
        return SearchResult(
            found=True,
            start_email_id=start_email_id,
            goal_email_id=goal_email_id,
            path=[single_result],
            explored=[single_result],
            heuristic_values=heuristic_values,
            graph_summary=graph.summary(),
            search_metrics={"nodes_explored": 1, "path_length": 0, "max_frontier_size": 1},
            message="Starting email is identical to target email. Trivial path of length 0 found.",
        )

    # Priority Queue: (heuristic_value, tie_breaker_entry_id, node_id)
    # Strictly Priority(n) = h(n)
    entry_counter = 0
    frontier: List[Tuple[float, int, str]] = []
    heapq.heappush(frontier, (h_start, entry_counter, start_node_id))

    visited: Set[str] = set()
    predecessors: Dict[str, Optional[str]] = {start_node_id: None}
    explored: List[SearchNodeResult] = []
    max_frontier_size = len(frontier)
    step_order = 0
    found = False

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        h_curr, _, curr_id = heapq.heappop(frontier)

        if curr_id in visited:
            continue

        visited.add(curr_id)
        step_order += 1
        curr_node = graph.nodes[curr_id]
        _, curr_expl = calculate_heuristic(curr_id, graph, goal_node_id)

        explored.append(
            SearchNodeResult(
                node_id=curr_id,
                node_type=curr_node.node_type,
                display_label=curr_node.display_label,
                heuristic=h_curr,
                explanation=curr_expl,
                step_order=step_order,
            )
        )

        # Check goal condition upon expansion
        if curr_id == goal_node_id:
            found = True
            break

        # Expand adjacent neighbors deterministically (sort neighbor IDs for reproducible order)
        for neighbor_id in sorted(graph.adj.get(curr_id, set())):
            if neighbor_id not in visited:
                if neighbor_id not in predecessors:
                    predecessors[neighbor_id] = curr_id

                h_neighbor, expl_n = calculate_heuristic(neighbor_id, graph, goal_node_id)
                heuristic_values[neighbor_id] = h_neighbor

                entry_counter += 1
                heapq.heappush(frontier, (h_neighbor, entry_counter, neighbor_id))

    # Reconstruct path if found
    reconstructed_path: List[SearchNodeResult] = []
    if found:
        curr: Optional[str] = goal_node_id
        while curr is not None:
            c_node = graph.nodes[curr]
            h_c = heuristic_values.get(curr, 0.0)
            _, expl_c = calculate_heuristic(curr, graph, goal_node_id)
            reconstructed_path.append(
                SearchNodeResult(
                    node_id=curr,
                    node_type=c_node.node_type,
                    display_label=c_node.display_label,
                    heuristic=h_c,
                    explanation=expl_c,
                )
            )
            curr = predecessors.get(curr)
        reconstructed_path.reverse()

    path_length = max(0, len(reconstructed_path) - 1) if found else 0
    message = (
        f"Greedy Best-First Search successfully identified an investigation path of {path_length} hops "
        f"connecting '{start_email_id}' to '{goal_email_id}'."
        if found
        else f"No connecting indicator path exists between '{start_email_id}' and '{goal_email_id}' "
        f"within the constructed graph (disconnected components)."
    )

    return SearchResult(
        found=found,
        start_email_id=start_email_id,
        goal_email_id=goal_email_id,
        path=reconstructed_path,
        explored=explored,
        heuristic_values=heuristic_values,
        graph_summary=graph.summary(),
        search_metrics={
            "nodes_explored": len(explored),
            "path_length": path_length,
            "max_frontier_size": max_frontier_size,
        },
        message=message,
    )


# =====================================================================
# Demonstration Synthetic Dataset
# =====================================================================

def get_demo_investigation_emails() -> List[Dict[str, str]]:
    """Returns curated synthetic, harmless demonstration emails with shared indicators."""
    return [
        {
            "id": "Email-1",
            "subject": "Urgent Security Verification Required",
            "body": (
                "Attention user: Please verify your credentials immediately at "
                "https://auth-portal.security-update.example.com/verify to prevent suspension. "
                "Contact our support desk at alert@security-update.example.com."
            ),
        },
        {
            "id": "Email-2",
            "subject": "Critical Incident Dispatch Notification",
            "body": (
                "Security Operations update: Incident reported regarding portal "
                "https://auth-portal.security-update.example.com/verify. "
                "Escalated by supervisor agent at triage@threat-feed.example.org."
            ),
        },
        {
            "id": "Email-3",
            "subject": "Wire Transfer Confirmation Request",
            "body": (
                "Please review wire confirmation invoice via portal "
                "https://wire-transfer.financial-ops.example.net/invoice. "
                "Questions should be directed to triage@threat-feed.example.org."
            ),
        },
        {
            "id": "Email-4",
            "subject": "Urgent Payment Release Notice",
            "body": (
                "Final notice: Payment release pending confirmation at "
                "https://wire-transfer.financial-ops.example.net/invoice. "
                "Issued by billing department: billing@financial-ops.example.net."
            ),
        },
        {
            "id": "Email-5",
            "subject": "Weekly Engineering Team Standup",
            "body": (
                "Hi team, meeting notes and code review summaries have been posted "
                "to internal knowledge base https://wiki.internal-dev.example.edu/sprint42. "
                "Reach out at dev-team@internal-dev.example.edu."
            ),
        },
    ]

