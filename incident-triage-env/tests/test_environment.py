import pytest
from models import (
    IncidentTriageAction,
    RootCauseCategory,
    Remediation,
)
from server.environment import IncidentTriageEnv
from server.episode_store import get_episode


@pytest.fixture
def env():
    return IncidentTriageEnv()


def test_reset_returns_valid_observation(env):
    obs = env.reset(task_id="easy_single_service_failure")
    assert obs.done is False
    assert obs.reward is None
    assert obs.result is None
    assert obs.steps_remaining == 10
    assert obs.alert.service == "payments-service"
    assert obs.services_investigated == []


def test_reset_state_is_clean(env):
    env.reset(task_id="easy_single_service_failure")
    state = env.state
    assert state.step_count == 0
    assert state.steps_remaining == 10
    assert state.task_id == "easy_single_service_failure"
    assert state.services_investigated == []
    assert state.actions_taken == []


def test_step_query_logs(env):
    env.reset(task_id="easy_single_service_failure")
    action = IncidentTriageAction(
        action_type="query_logs",
        service="payments-service",
    )
    obs = env.step(action)
    assert obs.done is False
    assert obs.result is not None
    assert obs.result.action_type == "query_logs"
    assert "NullPointerException" in obs.result.data
    assert obs.steps_remaining == 9
    assert obs.reward is not None


def test_step_decrements_remaining(env):
    env.reset(task_id="easy_single_service_failure")
    for i in range(3):
        action = IncidentTriageAction(
            action_type="check_status",
            service="payments-service",
        )
        obs = env.step(action)
    assert obs.steps_remaining == 7
    assert env.state.step_count == 3


def test_step_tracks_investigation(env):
    env.reset(task_id="easy_single_service_failure")
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.step(IncidentTriageAction(action_type="query_metrics", service="database", metric="cpu"))
    state = env.state
    assert "payments-service" in state.services_investigated
    assert "database" in state.services_investigated
    assert "query_logs" in state.evidence_collected.get("payments-service", [])
    assert "query_metrics" in state.evidence_collected.get("database", [])


def test_submit_diagnosis_ends_episode(env):
    env.reset(task_id="easy_single_service_failure")
    # Do some investigation first
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.step(IncidentTriageAction(action_type="check_deploys", service="payments-service"))
    env.step(IncidentTriageAction(action_type="inspect_code", service="payments-service"))
    env.step(IncidentTriageAction(action_type="query_metrics", service="payments-service", metric="error_rate"))

    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    assert obs.done is True
    assert "Grader score" in obs.result.data


def test_step_budget_exhaustion(env):
    env.reset(task_id="easy_single_service_failure")
    # Use all 10 steps without submitting
    for i in range(10):
        action = IncidentTriageAction(
            action_type="check_status",
            service="payments-service",
        )
        obs = env.step(action)
    assert obs.done is True
    assert obs.steps_remaining == 0


def test_episode_result_stored(env):
    env.reset(task_id="easy_single_service_failure", episode_id="test-ep-123")
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.step(IncidentTriageAction(action_type="check_deploys", service="payments-service"))
    env.step(IncidentTriageAction(action_type="inspect_code", service="payments-service"))
    env.step(IncidentTriageAction(action_type="query_metrics", service="payments-service", metric="error_rate"))
    env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    episode = get_episode("test-ep-123")
    assert episode is not None
    assert episode["task_id"] == "easy_single_service_failure"
    assert episode["diagnosis_submitted"] is True
    assert episode["grader_score"]["score"] > 0.0


def test_medium_task_reset(env):
    obs = env.reset(task_id="medium_cascading_dependency")
    assert obs.steps_remaining == 15
    assert obs.alert.service == "api-gateway"


def test_hard_task_reset(env):
    obs = env.reset(task_id="hard_multi_signal_cascade")
    assert obs.steps_remaining == 20
    assert obs.alert.service == "api-gateway"


def test_invalid_service_returns_error(env):
    env.reset(task_id="easy_single_service_failure")
    action = IncidentTriageAction(
        action_type="query_logs",
        service="nonexistent-service",
    )
    obs = env.step(action)
    assert "Unknown service" in obs.result.data


def test_state_reflects_episode(env):
    env.reset(task_id="easy_single_service_failure", episode_id="state-test")
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    state = env.state
    assert state.episode_id == "state-test"
    assert state.step_count == 1
    assert state.steps_remaining == 9
