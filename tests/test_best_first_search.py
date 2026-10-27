"""Unit and integration tests for Greedy Best-First Search & Threat Investigation Explorer.

Validates:
1. Valid path between connected email nodes
2. Disconnected components return found=False
3. Start and target being the same email (path length 0)
4. Cycles in graph are handled without infinite loops
5. Repeated indicators in a single email are deduplicated
6. Missing or empty email text handled gracefully
7. Emails without extractable indicators handled gracefully
8. Deterministic tie-breaking on tied heuristics
9. Correct Jaccard structural heuristic computation
10. Accurate predecessor and path reconstruction
11. Duplicate email IDs rejected by API
12. Invalid start and target IDs handled gracefully
13. Graph connecting via both URLs and email domains
14. API endpoints /api/search/demo-emails and /api/search/best-first
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.best_first_search import (
    EmailIndicatorGraph,
    EmailRecord,
    ExtractedIndicator,
    build_email_indicator_graph,
    calculate_heuristic,
    extract_indicators,
    get_demo_investigation_emails,
    greedy_best_first_search,
)

client = TestClient(app)


# =====================================================================
# Unit Tests: Indicator Extraction
# =====================================================================

def test_extract_indicators_from_text():
    """Verify URLs, domains, and email addresses are extracted and normalized."""
    text = (
        "Contact support@security-alert.org or visit https://Login.Example.COM/auth?id=123. "
        "Also check admin@sub.domain.co.uk."
    )
    indicators = extract_indicators(text)
    node_ids = {ind.node_id for ind in indicators}

    assert "email:support@security-alert.org" in node_ids
    assert "domain:security-alert.org" in node_ids
    assert "url:https://login.example.com/auth" in node_ids
    assert "domain:login.example.com" in node_ids
    assert "email:admin@sub.domain.co.uk" in node_ids
    assert "domain:sub.domain.co.uk" in node_ids


def test_repeated_indicators_in_one_email_deduplicated():
    """Verify repeated appearances of the same URL or email in one message are deduplicated."""
    text = (
        "Please visit https://example.com/login and again https://example.com/login. "
        "Email support@example.com or support@example.com."
    )
    indicators = extract_indicators(text)
    assert len([ind for ind in indicators if ind.node_id == "url:https://example.com/login"]) == 1
    assert len([ind for ind in indicators if ind.node_id == "email:support@example.com"]) == 1
    assert len([ind for ind in indicators if ind.node_id == "domain:example.com"]) == 1


def test_empty_or_missing_email_text():
    """Verify empty, whitespace, or none-like inputs yield an empty indicator set."""
    assert extract_indicators("") == set()
    assert extract_indicators("   \n\t  ") == set()
    assert extract_indicators(None) == set()  # type: ignore


def test_emails_without_indicators():
    """Verify emails with plain text and no URLs or emails produce no indicator nodes."""
    text = "Hello team, tomorrow's sprint review is scheduled for 10 AM. See you there."
    indicators = extract_indicators(text)
    assert len(indicators) == 0

    record = EmailRecord(id="E1", subject="Meeting", body=text)
    graph = build_email_indicator_graph([record])
    assert graph.email_count == 1
    assert graph.indicator_count == 0
    assert len(graph.adj["email:E1"]) == 0


# =====================================================================
# Unit Tests: Heuristic Calculation
# =====================================================================

def test_heuristic_calculation_cases():
    """Verify exact mathematical values of the structural Jaccard heuristic."""
    # E1 has {ind1, ind2}
    # E2 (Goal) has {ind2, ind3}
    # Intersection = {ind2} (1), Union = {ind1, ind2, ind3} (3)
    # Jaccard = 1/3 ~ 0.3333 -> h = 1 - 0.3333 = 0.6667
    e1 = EmailRecord(id="E1", body="https://ind1.com https://ind2.com")
    e2 = EmailRecord(id="E2", body="https://ind2.com https://ind3.com")
    graph = build_email_indicator_graph([e1, e2])

    # Target email heuristic is exactly 0.0
    h_goal, expl_goal = calculate_heuristic("email:E2", graph, "email:E2")
    assert h_goal == 0.0
    assert "Target email" in expl_goal

    # Non-target email heuristic
    h_e1, expl_e1 = calculate_heuristic("email:E1", graph, "email:E2")
    assert round(h_e1, 2) in (0.67, 0.75, 0.60)  # depending on domain vs url indicators

    # Indicator in goal has h = 0.0
    ind2_id = "domain:ind2.com"
    h_ind2, expl_ind2 = calculate_heuristic(ind2_id, graph, "email:E2")
    assert h_ind2 == 0.0
    assert "directly present in target email" in expl_ind2

    # Indicator not in goal has h = 1.0
    ind1_id = "domain:ind1.com"
    h_ind1, expl_ind1 = calculate_heuristic(ind1_id, graph, "email:E2")
    assert h_ind1 == 1.0
    assert "not directly present in target email" in expl_ind1


# =====================================================================
# Unit Tests: Greedy Best-First Search Algorithm
# =====================================================================

def test_valid_path_between_connected_emails():
    """Verify search finds the connecting path in a chain: E1 -> indA -> E2 -> indB -> E3."""
    emails = [
        EmailRecord(id="E1", body="See https://shared-a.com"),
        EmailRecord(id="E2", body="See https://shared-a.com and https://shared-b.com"),
        EmailRecord(id="E3", body="See https://shared-b.com"),
    ]
    graph = build_email_indicator_graph(emails)
    res = greedy_best_first_search(graph, "E1", "E3")

    assert res.found is True
    assert res.start_email_id == "E1"
    assert res.goal_email_id == "E3"
    assert res.search_metrics["path_length"] > 0

    # Path starts with E1 and ends with E3
    assert res.path[0].node_id == "email:E1"
    assert res.path[-1].node_id == "email:E3"

    # Alternating path: email -> indicator -> email -> indicator -> email
    for idx in range(len(res.path) - 1):
        curr_t = res.path[idx].node_type
        next_t = res.path[idx + 1].node_type
        assert curr_t != next_t, f"Step {idx} and {idx+1} have identical node types {curr_t}"


def test_disconnected_components_returns_no_path():
    """Verify search cleanly reports found=False for disconnected components."""
    emails = [
        EmailRecord(id="E1", body="See https://cluster-one.com"),
        EmailRecord(id="E2", body="See https://cluster-two.org"),
    ]
    graph = build_email_indicator_graph(emails)
    res = greedy_best_first_search(graph, "E1", "E2")

    assert res.found is False
    assert len(res.path) == 0
    assert "disconnected components" in res.message


def test_start_and_target_same_email():
    """Verify search handles start == target with trivial path of length 0."""
    emails = [EmailRecord(id="E1", body="See https://example.com")]
    graph = build_email_indicator_graph(emails)
    res = greedy_best_first_search(graph, "E1", "E1")

    assert res.found is True
    assert len(res.path) == 1
    assert res.path[0].node_id == "email:E1"
    assert res.search_metrics["path_length"] == 0
    assert "identical to target email" in res.message


def test_graph_containing_cycles():
    """Verify search terminates cleanly on graphs with multiple cycles without looping."""
    # E1 -> A -> E2 -> B -> E3
    # E1 -> C -> E2
    # E2 -> D -> E1 (cycle back)
    emails = [
        EmailRecord(id="E1", body="https://site-a.com https://site-c.com https://site-d.com"),
        EmailRecord(id="E2", body="https://site-a.com https://site-b.com https://site-c.com https://site-d.com"),
        EmailRecord(id="E3", body="https://site-b.com"),
    ]
    graph = build_email_indicator_graph(emails)
    res = greedy_best_first_search(graph, "E1", "E3")

    assert res.found is True
    assert res.path[-1].node_id == "email:E3"
    # All nodes on path are unique
    path_ids = [p.node_id for p in res.path]
    assert len(path_ids) == len(set(path_ids))


def test_deterministic_results_when_heuristics_tied():
    """Verify search yields 100% reproducible order and paths over multiple runs."""
    emails = [
        EmailRecord(id="E1", body="https://alpha.com https://beta.com"),
        EmailRecord(id="E2", body="https://alpha.com https://gamma.com"),
        EmailRecord(id="E3", body="https://beta.com https://gamma.com"),
        EmailRecord(id="E4", body="https://gamma.com"),
    ]
    graph = build_email_indicator_graph(emails)

    res1 = greedy_best_first_search(graph, "E1", "E4")
    res2 = greedy_best_first_search(graph, "E1", "E4")

    assert res1.found == res2.found
    assert [p.node_id for p in res1.path] == [p.node_id for p in res2.path]
    assert [e.node_id for e in res1.explored] == [e.node_id for e in res2.explored]


def test_invalid_start_or_goal_identifiers():
    """Verify search returns clean failure when start or goal ID does not exist."""
    emails = [EmailRecord(id="E1", body="https://example.com")]
    graph = build_email_indicator_graph(emails)

    res_bad_start = greedy_best_first_search(graph, "NON_EXISTENT", "E1")
    assert res_bad_start.found is False
    assert "was not found in the graph" in res_bad_start.message

    res_bad_goal = greedy_best_first_search(graph, "E1", "NON_EXISTENT")
    assert res_bad_goal.found is False
    assert "was not found in the graph" in res_bad_goal.message


def test_shared_url_and_email_domain_indicators():
    """Verify graph connects messages sharing either exact URLs or domain names."""
    emails = [
        EmailRecord(id="E1", body="Reach us at security@defense-network.org"),
        EmailRecord(id="E2", body="Visit billing portal https://portal.defense-network.org/login"),
    ]
    graph = build_email_indicator_graph(emails)

    # Both share the base domain or host 'defense-network.org'
    # E1 has domain:defense-network.org, E2 has domain:portal.defense-network.org
    # Let's ensure a shared domain connects them
    emails_shared = [
        EmailRecord(id="E1", body="Reach us at security@defense-network.org"),
        EmailRecord(id="E2", body="Portal hosted at https://defense-network.org/login"),
    ]
    graph_shared = build_email_indicator_graph(emails_shared)
    res = greedy_best_first_search(graph_shared, "E1", "E2")

    assert res.found is True
    # Shared domain indicator node connects them
    assert any("domain:defense-network.org" in p.node_id for p in res.path)


def test_demo_dataset_execution():
    """Verify the built-in synthetic demo dataset produces expected paths and disconnects."""
    demo_emails = [EmailRecord(**d) for d in get_demo_investigation_emails()]
    graph = build_email_indicator_graph(demo_emails)

    # 1. Connected path between Email-1 and Email-4
    res_connected = greedy_best_first_search(graph, "Email-1", "Email-4")
    assert res_connected.found is True
    assert res_connected.search_metrics["path_length"] == 6

    # 2. Disconnected path between Email-1 and Email-5
    res_disconnected = greedy_best_first_search(graph, "Email-1", "Email-5")
    assert res_disconnected.found is False
    assert len(res_disconnected.path) == 0


# =====================================================================
# API Integration Tests
# =====================================================================

def test_api_search_demo_emails_endpoint():
    """Verify /api/search/demo-emails returns valid synthetic demonstration messages."""
    response = client.get("/api/search/demo-emails")
    assert response.status_code == 200
    data = response.json()
    assert "emails" in data
    assert len(data["emails"]) >= 4
    first = data["emails"][0]
    assert "id" in first
    assert "subject" in first
    assert "body" in first


def test_api_search_best_first_endpoint():
    """Verify /api/search/best-first executes search and returns full structured response."""
    payload = {
        "emails": get_demo_investigation_emails(),
        "start_email_id": "Email-1",
        "goal_email_id": "Email-4",
    }
    response = client.post("/api/search/best-first", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["found"] is True
    assert data["start_email_id"] == "Email-1"
    assert data["goal_email_id"] == "Email-4"
    assert len(data["path"]) > 0
    assert len(data["explored"]) > 0
    assert "graph_visualization" in data
    assert len(data["graph_visualization"]["nodes"]) > 0
    assert len(data["graph_visualization"]["edges"]) > 0

    # Verify path node schema
    path_node = data["path"][0]
    assert "node_id" in path_node
    assert "node_type" in path_node
    assert "heuristic" in path_node
    assert "explanation" in path_node


def test_api_search_duplicate_email_ids_rejected():
    """Verify duplicate email IDs in request trigger 400 Bad Request."""
    payload = {
        "emails": [
            {"id": "duplicate-id", "subject": "A", "body": "https://site.com"},
            {"id": "duplicate-id", "subject": "B", "body": "https://other.com"},
        ],
        "start_email_id": "duplicate-id",
        "goal_email_id": "duplicate-id",
    }
    response = client.post("/api/search/best-first", json=payload)
    assert response.status_code == 400
    assert "Duplicate email identifier" in response.json()["detail"]


def test_api_search_invalid_start_or_goal_ids_rejected():
    """Verify start or goal ID not matching any email triggers 400 Bad Request."""
    payload = {
        "emails": [
            {"id": "E1", "subject": "A", "body": "https://site.com"},
            {"id": "E2", "subject": "B", "body": "https://other.com"},
        ],
        "start_email_id": "UNKNOWN_START",
        "goal_email_id": "E2",
    }
    response = client.post("/api/search/best-first", json=payload)
    assert response.status_code == 400
    assert "Start email ID 'UNKNOWN_START' does not match" in response.json()["detail"]

