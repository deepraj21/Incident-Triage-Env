import pytest
from models import (
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
    AlertInfo,
    RootCauseCategory,
    Remediation,
)
from server.grader import InvestigationRubric, DiagnosisScorer
from server.scenario_loader import ScenarioLoader


@pytest.fixture
def easy_scenario():
    loader = ScenarioLoader()
    return loader.get_scenario("easy_single_service_failure")


@pytest.fixture
def medium_scenario():
    loader = ScenarioLoader()
    return loader.get_scenario("medium_cascading_dependency")


@pytest.fixture
def rubric(easy_scenario):
    r = InvestigationRubric(easy_scenario)
    return r


@pytest.fixture
def alert():
    return AlertInfo(service="test", message="msg", severity="critical", timestamp="t")


def make_obs(alert, done=False, steps=10):
    return IncidentTriageObservation(
        done=done, alert=alert, steps_remaining=steps
    )


def test_direct_evidence_reward(rubric, alert):
    action = IncidentTriageAction(action_type="query_logs", service="payments-service")
    obs = make_obs(alert)
    reward = rubric(action, obs)
    assert reward > 0.1  # direct evidence = 0.12 + strategy bonuses


def test_contextual_evidence_reward(rubric, alert):
    action = IncidentTriageAction(action_type="query_logs", service="database")
    obs = make_obs(alert)
    reward = rubric(action, obs)
    assert reward >= 0.04  # contextual = 0.04


def test_redundant_query_no_reward(rubric, alert):
    action = IncidentTriageAction(action_type="query_logs", service="payments-service")
    obs = make_obs(alert)
    rubric(action, obs)  # first time
    reward2 = rubric(action, obs)  # redundant
    assert reward2 <= 0.0  # no info gain, possibly penalty


def test_rubric_reset(rubric, alert):
    action = IncidentTriageAction(action_type="query_logs", service="payments-service")
    obs = make_obs(alert)
    rubric(action, obs)
    rubric.reset()
    reward = rubric(action, obs)
    assert reward > 0.1  # direct evidence again after reset


def test_diagnosis_scorer_perfect():
    scorer = DiagnosisScorer()
    ground_truth = {
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
        "causal_chain": ["payments-service"],
        "failure_family": "change-induced",
    }
    diagnosis = {
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
    }
    state = IncidentTriageState(
        episode_id="ep",
        step_count=5,
        steps_remaining=5,
        task_id="easy_single_service_failure",
        services_investigated=["payments-service"],
        actions_taken=[
            {"action_type": "query_logs", "service": "payments-service"},
            {"action_type": "query_metrics", "service": "payments-service"},
            {"action_type": "check_deploys", "service": "payments-service"},
            {"action_type": "inspect_code", "service": "payments-service"},
            {"action_type": "submit_diagnosis", "service": ""},
        ],
    )
    result = scorer.score(ground_truth, diagnosis, state)
    assert result["diagnosis_submitted"] is True
    assert result["breakdown"]["root_cause_service"] == 0.20
    assert result["breakdown"]["root_cause_category"] == 0.15
    assert result["breakdown"]["remediation"] == 0.10
    assert result["breakdown"]["evidence_quality"] == 0.10
    assert result["score"] > 0.5


def test_diagnosis_scorer_no_diagnosis():
    scorer = DiagnosisScorer()
    result = scorer.score({}, None, IncidentTriageState())
    assert result["score"] == 0.0
    assert result["diagnosis_submitted"] is False


def test_diagnosis_scorer_partial_credit():
    scorer = DiagnosisScorer()
    ground_truth = {
        "root_cause_service": "auth-db",
        "root_cause_category": "config_change",
        "remediation": "fix_config",
        "causal_chain": ["auth-db", "user-service", "api-gateway"],
        "failure_family": "change-induced",
    }
    diagnosis = {
        "root_cause_service": "user-service",  # one hop away
        "root_cause_category": "bad_deploy",  # same failure family
        "remediation": "restart_service",  # wrong
    }
    state = IncidentTriageState(
        episode_id="ep",
        step_count=8,
        steps_remaining=7,
        services_investigated=["api-gateway", "user-service", "auth-db"],
        actions_taken=[{"action_type": f"a{i}", "service": f"s{i}"} for i in range(8)],
    )
    result = scorer.score(ground_truth, diagnosis, state)
    assert result["breakdown"]["root_cause_service"] == 0.10  # one hop
    assert result["breakdown"]["root_cause_category"] == 0.07  # same family


def test_diagnosis_scorer_shotgun_penalty():
    scorer = DiagnosisScorer()
    ground_truth = {
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
        "causal_chain": ["payments-service"],
    }
    diagnosis = {
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
    }
    state = IncidentTriageState(
        episode_id="ep",
        step_count=2,  # submitted in 2 steps — shotgun
        steps_remaining=8,
        services_investigated=["payments-service"],
        actions_taken=[
            {"action_type": "query_logs", "service": "payments-service"},
            {"action_type": "submit_diagnosis", "service": ""},
        ],
    )
    result = scorer.score(ground_truth, diagnosis, state)
    assert result["breakdown"]["penalties"] == -0.15


def test_diagnosis_scorer_circular_penalty():
    scorer = DiagnosisScorer()
    ground_truth = {
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
        "causal_chain": ["payments-service"],
    }
    diagnosis = {
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
    }
    # 5 non-submit queries to same service = circular (>= 4)
    state = IncidentTriageState(
        episode_id="ep",
        step_count=6,
        steps_remaining=4,
        services_investigated=["payments-service"],
        actions_taken=[
            {"action_type": "query_logs", "service": "payments-service"},
            {"action_type": "query_metrics", "service": "payments-service"},
            {"action_type": "check_deploys", "service": "payments-service"},
            {"action_type": "check_status", "service": "payments-service"},
            {"action_type": "inspect_code", "service": "payments-service"},
            {"action_type": "submit_diagnosis", "service": ""},
        ],
    )
    result = scorer.score(ground_truth, diagnosis, state)
    # 5 queries to payments-service (excluding submit) >= 4, so -0.10
    assert result["breakdown"]["penalties"] == -0.10
