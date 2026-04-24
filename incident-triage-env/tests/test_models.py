import pytest
from models import (
    RootCauseCategory,
    Remediation,
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
    AlertInfo,
    ActionResult,
)


def test_query_logs_action_valid():
    action = IncidentTriageAction(
        action_type="query_logs",
        service="payments-service",
        keyword="error",
    )
    assert action.action_type == "query_logs"
    assert action.service == "payments-service"


def test_query_logs_action_missing_service_fails():
    with pytest.raises(Exception):
        IncidentTriageAction(action_type="query_logs")


def test_submit_diagnosis_valid():
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    assert action.action_type == "submit_diagnosis"
    assert action.root_cause_service == "payments-service"


def test_submit_diagnosis_missing_fields_fails():
    with pytest.raises(Exception):
        IncidentTriageAction(
            action_type="submit_diagnosis",
            root_cause_service="payments-service",
        )


def test_all_action_types_valid():
    for action_type in ["query_logs", "query_metrics", "check_deploys",
                        "trace_dependencies", "check_status", "inspect_code"]:
        action = IncidentTriageAction(
            action_type=action_type,
            service="test-service",
        )
        assert action.action_type == action_type


def test_query_metrics_with_metric():
    action = IncidentTriageAction(
        action_type="query_metrics",
        service="db",
        metric="cpu",
    )
    assert action.metric == "cpu"


def test_observation_creation():
    alert = AlertInfo(
        service="test",
        message="Test alert",
        severity="critical",
        timestamp="2026-01-01T00:00:00Z",
    )
    obs = IncidentTriageObservation(
        done=False,
        reward=0.5,
        alert=alert,
        steps_remaining=10,
    )
    assert obs.done is False
    assert obs.reward == 0.5
    assert obs.steps_remaining == 10
    assert obs.services_investigated == []


def test_observation_with_result():
    alert = AlertInfo(service="test", message="msg", severity="warning", timestamp="t")
    result = ActionResult(action_type="query_logs", data="some log data")
    obs = IncidentTriageObservation(
        done=False,
        alert=alert,
        result=result,
        steps_remaining=5,
    )
    assert obs.result.data == "some log data"


def test_state_creation():
    state = IncidentTriageState(
        episode_id="ep-1",
        step_count=3,
        task_id="easy_single_service_failure",
        steps_remaining=7,
        services_investigated=["payments-service"],
        actions_taken=[{"action_type": "query_logs", "service": "payments-service"}],
        evidence_collected={"payments-service": ["query_logs"]},
    )
    assert state.episode_id == "ep-1"
    assert state.step_count == 3
    assert state.steps_remaining == 7


def test_action_serialization():
    action = IncidentTriageAction(
        action_type="query_logs",
        service="payments-service",
        severity="error",
    )
    data = action.model_dump(exclude_none=True)
    # New tagged-union shape: {app, op, args}.
    assert data["app"] == "obsly"
    assert data["op"] == "query_logs"
    assert data["args"]["service"] == "payments-service"
    assert data["args"]["severity"] == "error"
    assert "keyword" not in data["args"]
    # Legacy convenience properties still work.
    assert action.action_type == "query_logs"
    assert action.service == "payments-service"


def test_new_shape_action_direct():
    action = IncidentTriageAction(
        app="repohub",
        op="open_pr",
        args={
            "target_repo": "fintra/payments-service",
            "head_branch": "fix/npe",
            "title": "Guard against null guest user",
        },
    )
    assert action.app.value == "repohub"
    assert action.op == "open_pr"
    assert action.args["target_repo"] == "fintra/payments-service"


def test_invalid_op_for_app_fails():
    with pytest.raises(Exception):
        IncidentTriageAction(app="obsly", op="open_pr", args={"service": "x"})


def test_missing_required_args_fails():
    with pytest.raises(Exception):
        IncidentTriageAction(app="repohub", op="open_pr", args={"title": "no repo"})


def test_root_cause_category_values():
    assert RootCauseCategory.BAD_DEPLOY.value == "bad_deploy"
    assert RootCauseCategory.RESOURCE_EXHAUSTION.value == "resource_exhaustion"


def test_remediation_values():
    assert Remediation.ROLLBACK_DEPLOY.value == "rollback_deploy"
    assert Remediation.FIX_CONFIG.value == "fix_config"
