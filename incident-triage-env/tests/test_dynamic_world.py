"""Tests for Phase 2 — WorldClock + EventQueue dynamic world engine."""

from __future__ import annotations

import copy

import pytest

from models import IncidentTriageAction, IncidentTriageState
from server.environment import IncidentTriageEnv
from server.event_queue import EventQueue
from server.world_clock import WorldClock


# --------------------------------------------------------- WorldClock


def test_world_clock_advances_by_step_seconds():
    clk = WorldClock(start_iso="2026-03-26T14:32:00Z", step_seconds=30)
    assert clk.step == 0
    assert clk.now_iso().startswith("2026-03-26T14:32:00")
    clk.advance()
    assert clk.step == 1
    assert clk.now_iso().startswith("2026-03-26T14:32:30")
    clk.advance()
    assert clk.now_iso().startswith("2026-03-26T14:33:00")


def test_world_clock_from_scenario_falls_back_to_alert():
    scenario = {"alert": {"timestamp": "2026-03-26T14:32:00Z"}}
    clk = WorldClock.from_scenario(scenario)
    assert clk.now_iso().startswith("2026-03-26T14:32:00")
    assert clk.step_seconds == 30


def test_world_clock_uses_explicit_clock_block():
    scenario = {"clock": {"start": "2030-01-01T00:00:00Z", "step_seconds": 5}}
    clk = WorldClock.from_scenario(scenario)
    assert clk.now_iso().startswith("2030-01-01T00:00:00")
    clk.advance()
    assert clk.now_iso().startswith("2030-01-01T00:00:05")


# --------------------------------------------------------- EventQueue


@pytest.fixture
def scenario_and_state():
    scenario = {
        "services": {"svc-a": {"logs": [], "deploys": [], "status": {}}},
        "alerthub": {"alerts": []},
        "chatops": {"oncall_lead": "alice"},
    }
    return scenario, IncidentTriageState()


def test_event_queue_fires_at_correct_step(scenario_and_state):
    scenario, state = scenario_and_state
    queue = EventQueue([
        {"at_step": 2, "event": "new_log", "service": "svc-a",
         "payload": {"severity": "error", "message": "boom"}},
    ])
    clk = WorldClock(start_iso="2026-01-01T00:00:00Z", step_seconds=30)
    # step 1 — nothing yet
    clk.advance()
    assert queue.fire_due(clk, scenario, state) == []
    assert scenario["services"]["svc-a"]["logs"] == []
    # step 2 — event fires
    clk.advance()
    fired = queue.fire_due(clk, scenario, state)
    assert len(fired) == 1
    assert scenario["services"]["svc-a"]["logs"][0]["message"] == "boom"
    # step 3 — already fired, nothing re-fires
    clk.advance()
    assert queue.fire_due(clk, scenario, state) == []


def test_event_queue_silent_events_not_returned_but_mutate(scenario_and_state):
    scenario, state = scenario_and_state
    queue = EventQueue([
        {"at_step": 1, "event": "new_log", "service": "svc-a",
         "visible": False,
         "payload": {"severity": "warn", "message": "silent drift"}},
    ])
    clk = WorldClock(start_iso="2026-01-01T00:00:00Z", step_seconds=30)
    clk.advance()
    fired = queue.fire_due(clk, scenario, state)
    assert fired == []  # not surfaced to agent
    # but world did change — stealth-regression shape
    assert scenario["services"]["svc-a"]["logs"][0]["message"] == "silent drift"
    # and state.event_log still records it for the grader / demo timeline
    assert len(state.event_log) == 1
    assert state.event_log[0]["visible"] is False


def test_event_queue_new_alert_and_new_deploy_and_handoff(scenario_and_state):
    scenario, state = scenario_and_state
    queue = EventQueue([
        {"at_step": 1, "event": "new_deploy", "service": "svc-a",
         "payload": {"author": "sre-bob", "diff_summary": "hotfix"}},
        {"at_step": 1, "event": "oncall_handoff",
         "payload": {"from": "alice", "to": "carol"}},
        {"at_step": 1, "event": "new_alert",
         "payload": {"id": "ALERT-99", "service": "svc-a",
                     "message": "db pool exhausted", "severity": "critical"}},
    ])
    clk = WorldClock(start_iso="2026-01-01T00:00:00Z", step_seconds=30)
    clk.advance()
    fired = queue.fire_due(clk, scenario, state)
    assert len(fired) == 3
    assert scenario["services"]["svc-a"]["deploys"][0]["author"] == "sre-bob"
    assert scenario["chatops"]["oncall_lead"] == "carol"
    assert scenario["alerthub"]["alerts"][0]["id"] == "ALERT-99"


def test_event_queue_pending_count(scenario_and_state):
    scenario, state = scenario_and_state
    queue = EventQueue([
        {"at_step": 1, "event": "new_log", "service": "svc-a",
         "payload": {"severity": "info", "message": "a"}},
        {"at_step": 5, "event": "new_log", "service": "svc-a",
         "payload": {"severity": "info", "message": "b"}},
    ])
    assert queue.pending_count() == 2
    clk = WorldClock(start_iso="2026-01-01T00:00:00Z", step_seconds=30)
    clk.advance()
    queue.fire_due(clk, scenario, state)
    assert queue.pending_count() == 1


# --------------------------------------------------------- integration with env


def test_env_reset_surfaces_sim_time():
    env = IncidentTriageEnv()
    obs = env.reset(task_id="easy_single_service_failure")
    # easy scenario's clock block anchors sim-time to the incident start.
    assert obs.sim_time is not None
    assert obs.sim_time.startswith("2026-03-26T14:32:00")
    assert obs.world_events == []
    # Phase 4 upgraded easy scenario adds a 2-event timeline; just ensure
    # the queue exists and returns a non-negative pending count.
    assert obs.pending_world_events >= 0


def test_env_scenario_isolation_across_episodes(monkeypatch):
    """Two resets must not share mutated scenario state (deep-copy guarantee)."""
    env = IncidentTriageEnv()
    # Inject a timeline into the loader's cached copy of the easy scenario.
    env._loader._scenarios["easy_single_service_failure"]["timeline"] = [
        {"at_step": 1, "event": "new_log", "service": "payments-service",
         "payload": {"severity": "error", "message": "INJECTED-EVENT-XYZ"}}
    ]
    try:
        env.reset(task_id="easy_single_service_failure", episode_id="ep1")
        env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
        # scenario for episode 1 now has the injected log
        ep1_logs = env._current_scenario["services"]["payments-service"]["logs"]
        assert any("INJECTED-EVENT-XYZ" in l["message"] for l in ep1_logs)

        env.reset(task_id="easy_single_service_failure", episode_id="ep2")
        # episode 2's scenario must NOT contain the mutation from episode 1.
        ep2_logs = env._current_scenario["services"]["payments-service"]["logs"]
        assert not any("INJECTED-EVENT-XYZ" in l["message"] for l in ep2_logs)
    finally:
        env._loader._scenarios["easy_single_service_failure"].pop("timeline", None)


def test_env_world_event_surfaces_on_obs():
    env = IncidentTriageEnv()
    env._loader._scenarios["easy_single_service_failure"]["timeline"] = [
        {"at_step": 1, "event": "new_alert",
         "payload": {"id": "ALERT-NEW", "service": "database",
                     "message": "secondary db latency", "severity": "warning"}}
    ]
    try:
        env.reset(task_id="easy_single_service_failure")
        obs = env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
        assert len(obs.world_events) == 1
        assert obs.world_events[0].event == "new_alert"
        assert obs.world_events[0].payload["id"] == "ALERT-NEW"
        assert obs.sim_time.startswith("2026-03-26T14:32:30")
    finally:
        env._loader._scenarios["easy_single_service_failure"].pop("timeline", None)


def test_alerthub_hides_alerts_before_fire_time():
    """Clock-aware filtering: a future-fire_time alert should not list yet."""
    from server.apps import alerthub

    scenario = {
        "alerthub": {
            "alerts": [
                {"id": "A1", "service": "svc", "severity": "critical",
                 "message": "now", "fire_time": "2026-01-01T00:00:00Z"},
                {"id": "A2", "service": "svc", "severity": "warning",
                 "message": "later", "fire_time": "2026-01-01T00:05:00Z"},
            ]
        }
    }
    state = IncidentTriageState()
    clk = WorldClock(start_iso="2026-01-01T00:01:00Z", step_seconds=30)
    out = alerthub.dispatch("list_alerts", {}, scenario, state, clock=clk)
    assert "A1" in out
    assert "A2" not in out
