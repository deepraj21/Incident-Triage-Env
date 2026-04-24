"""Unit tests for per-app dispatchers under server/apps/."""

import pytest

from models import IncidentTriageState
from server.apps import alerthub, chatops, obsly, repohub, ticketdesk


@pytest.fixture
def state():
    return IncidentTriageState()


@pytest.fixture
def scenario():
    return {
        "alert": {
            "id": "ALERT-42",
            "service": "payments-service",
            "message": "500 spike",
            "severity": "critical",
            "timestamp": "2026-03-26T14:32:00Z",
        },
        "services": {
            "payments-service": {
                "dependencies": ["database"],
                "dependents": ["api-gateway"],
                "logs": [
                    {"timestamp": "14:32", "severity": "error", "message": "NullPointerException at line 47"},
                    {"timestamp": "14:33", "severity": "info", "message": "startup ok"},
                ],
                "metrics": {
                    "error_rate": [{"ts": "14:32", "val": 15.2}],
                    "cpu": [{"ts": "14:32", "val": 40}],
                },
                "status": {"health": "degraded", "uptime": "99.1%", "active_alerts": ["ALERT-42"]},
                "deploys": [
                    {"timestamp": "2026-03-26T14:31:00Z", "author": "alice",
                     "diff_summary": "refactor payment processor", "diff_patch": "@@ -1 +1 @@"},
                ],
                "code_snippets": {
                    "PaymentProcessor.java": "class PaymentProcessor { void process() {} }",
                },
            },
            "database": {"dependencies": [], "dependents": ["payments-service"]},
        },
        "ticketdesk": {
            "tickets": [
                {"id": "ING-101", "title": "Payments timeouts", "body": "customers affected"},
                {"id": "ING-102", "title": "Unrelated DB cleanup"},
            ]
        },
        "chatops": {
            "oncall_lead": "bob",
            "channels": {
                "incidents": [{"user": "alice", "text": "investigating"}],
            },
        },
        "obsly": {"traces": {"trace-7": {"spans": [{"name": "db.query", "ms": 42}]}}},
    }


# -------------------------------------------------------- alerthub


def test_alerthub_list_and_get(scenario, state):
    out = alerthub.dispatch("list_alerts", {}, scenario, state)
    assert "ALERT-42" in out
    got = alerthub.dispatch("get_alert", {"alert_id": "ALERT-42"}, scenario, state)
    assert "payments-service" in got


def test_alerthub_ack(scenario, state):
    out = alerthub.dispatch("ack_alert", {"alert_id": "ALERT-42"}, scenario, state)
    assert "acked" in out
    assert "ALERT-42" in state.acked_alerts


def test_alerthub_ack_unknown(scenario, state):
    out = alerthub.dispatch("ack_alert", {"alert_id": "ALERT-nope"}, scenario, state)
    assert "cannot ack" in out
    assert state.acked_alerts == []


# -------------------------------------------------------- obsly


def test_obsly_query_logs(scenario, state):
    out = obsly.dispatch("query_logs", {"service": "payments-service"}, scenario, state)
    assert "NullPointerException" in out


def test_obsly_query_logs_keyword_filter(scenario, state):
    out = obsly.dispatch(
        "query_logs",
        {"service": "payments-service", "keyword": "nullpointer"},
        scenario, state,
    )
    assert "NullPointerException" in out
    assert "startup ok" not in out


def test_obsly_query_metric(scenario, state):
    out = obsly.dispatch(
        "query_metric",
        {"service": "payments-service", "metric": "error_rate"},
        scenario, state,
    )
    assert "15.2%" in out


def test_obsly_get_trace(scenario, state):
    out = obsly.dispatch("get_trace", {"trace_id": "trace-7"}, scenario, state)
    assert "db.query" in out


def test_obsly_unknown_service(scenario, state):
    out = obsly.dispatch("query_logs", {"service": "ghost"}, scenario, state)
    assert "Unknown service" in out


# -------------------------------------------------------- repohub


def test_repohub_recent_commits(scenario, state):
    out = repohub.dispatch("recent_commits", {"service": "payments-service"}, scenario, state)
    assert "alice" in out
    assert "refactor payment processor" in out


def test_repohub_get_diff_roundtrip(scenario, state):
    commits = repohub.dispatch("recent_commits", {"service": "payments-service"}, scenario, state)
    sha = commits.split(" | ")[0]
    out = repohub.dispatch("get_diff", {"commit_sha": sha}, scenario, state)
    assert sha in out
    assert "alice" in out


def test_repohub_get_file(scenario, state):
    out = repohub.dispatch(
        "get_file",
        {"service": "payments-service", "path": "PaymentProcessor.java"},
        scenario, state,
    )
    assert "class PaymentProcessor" in out


def test_repohub_list_files(scenario, state):
    out = repohub.dispatch("list_files", {"service": "payments-service"}, scenario, state)
    assert "PaymentProcessor.java" in out


def test_repohub_open_pr(scenario, state):
    out = repohub.dispatch(
        "open_pr",
        {
            "target_repo": "fintra/payments",
            "head_branch": "fix/npe",
            "title": "Null guard",
            "summary": "guards against null user",
            "diff_patch": "@@",
        },
        scenario, state,
    )
    assert "pr_url" in out
    assert len(state.opened_prs) == 1
    assert state.opened_prs[0]["target_repo"] == "fintra/payments"


# -------------------------------------------------------- ticketdesk


def test_ticketdesk_search(scenario, state):
    out = ticketdesk.dispatch("search_tickets", {"q": "payments"}, scenario, state)
    assert "ING-101" in out
    assert "ING-102" not in out


def test_ticketdesk_get_ticket(scenario, state):
    out = ticketdesk.dispatch("get_ticket", {"ticket_id": "ING-101"}, scenario, state)
    assert "Payments timeouts" in out


def test_ticketdesk_create_incident(scenario, state):
    out = ticketdesk.dispatch(
        "create_incident",
        {"title": "Payments 5xx", "body": "npe", "severity": "sev1"},
        scenario, state,
    )
    assert "INC-0001" in out
    assert state.created_tickets[0]["title"] == "Payments 5xx"


# -------------------------------------------------------- chatops


def test_chatops_post_update(scenario, state):
    out = chatops.dispatch(
        "post_update",
        {"channel": "incidents", "message": "rolling back v2.14.1"},
        scenario, state,
    )
    assert "posted" in out
    assert state.posted_updates[0]["channel"] == "incidents"


def test_chatops_read_channel(scenario, state):
    out = chatops.dispatch("read_channel", {"channel": "incidents"}, scenario, state)
    assert "investigating" in out


def test_chatops_page_oncall(scenario, state):
    out = chatops.dispatch("page_oncall", {"role": "payments-lead"}, scenario, state)
    assert "paged payments-lead" in out
    assert "bob" in out
    assert "payments-lead" in state.paged_roles
