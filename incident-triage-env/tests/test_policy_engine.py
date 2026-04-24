"""Tests for Phase 3 — PolicyEngine business-rule enforcement."""

from __future__ import annotations

import pytest

from models import (
    IncidentTriageAction,
    IncidentTriageState,
    Remediation,
    RootCauseCategory,
)
from server.environment import IncidentTriageEnv
from server.policy_engine import PolicyEngine


# --------------------------------------------------------- unit: matching


def _mk_state(past=None, step=1):
    state = IncidentTriageState(step_count=step)
    for a in past or []:
        state.actions_taken.append(a)
    return state


def test_matches_app_op_and_args_match():
    engine = PolicyEngine(
        rules=[
            {
                "id": "pen-rollback",
                "when": {
                    "app": "system",
                    "op": "submit_diagnosis",
                    "args_match": {"remediation": "rollback_deploy"},
                },
                "penalty": -0.5,
            }
        ]
    )
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    state = _mk_state(step=3)
    # No require_prior, no forbidden_if → match with no violation trigger → bonus=0, skipped.
    # But we set penalty-only: the rule is "pure stick"; on match without
    # any violation trigger, no-op. That's by design (documented).
    delta, violations = engine.evaluate(action, state)
    assert delta == 0.0
    assert violations == []


def test_forbidden_if_scenario_tag_triggers_penalty():
    engine = PolicyEngine(
        rules=[
            {
                "id": "no-forward-fix-in-freeze",
                "when": {"app": "repohub", "op": "open_pr"},
                "forbidden_if": {"scenario_tag": "change_freeze"},
                "penalty": -0.3,
            }
        ],
        tags=["change_freeze"],
    )
    action = IncidentTriageAction(
        app="repohub",
        op="open_pr",
        args={"target_repo": "r", "head_branch": "b", "title": "t"},
    )
    state = _mk_state(step=2)
    delta, violations = engine.evaluate(action, state)
    assert delta == pytest.approx(-0.3)
    assert violations == ["no-forward-fix-in-freeze"]
    assert "no-forward-fix-in-freeze" in state.policy_violations


def test_require_prior_missing_triggers_penalty():
    engine = PolicyEngine(
        rules=[
            {
                "id": "page-before-rollback",
                "when": {
                    "app": "system",
                    "op": "submit_diagnosis",
                    "args_match": {"remediation": "rollback_deploy"},
                },
                "require_prior": {"app": "chatops", "op": "page_oncall"},
                "penalty": -0.25,
            }
        ]
    )
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    state = _mk_state(step=4)
    delta, violations = engine.evaluate(action, state)
    assert delta == pytest.approx(-0.25)
    assert "page-before-rollback" in violations


def test_require_prior_satisfied_yields_bonus():
    engine = PolicyEngine(
        rules=[
            {
                "id": "page-before-rollback",
                "when": {
                    "app": "system",
                    "op": "submit_diagnosis",
                    "args_match": {"remediation": "rollback_deploy"},
                },
                "require_prior": {"app": "chatops", "op": "page_oncall"},
                "penalty": -0.25,
                "bonus": 0.15,
            }
        ]
    )
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    state = _mk_state(
        past=[
            {"app": "chatops", "op": "page_oncall", "args": {"role": "sre"}, "at_step": 2},
            {"app": "system", "op": "submit_diagnosis", "args": {}, "at_step": 5},
        ],
        step=5,
    )
    delta, violations = engine.evaluate(action, state)
    assert delta == pytest.approx(0.15)
    assert violations == []


def test_require_prior_within_steps_expires():
    engine = PolicyEngine(
        rules=[
            {
                "id": "page-recent",
                "when": {"app": "system", "op": "submit_diagnosis"},
                "require_prior": {"app": "chatops", "op": "page_oncall"},
                "require_prior_within_steps": 3,
                "penalty": -0.2,
            }
        ]
    )
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="svc",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    # Paged at step 1, now on step 10 → gap=9, outside window=3 → violation.
    state = _mk_state(
        past=[{"app": "chatops", "op": "page_oncall", "args": {}, "at_step": 1}],
        step=10,
    )
    delta, _ = engine.evaluate(action, state)
    assert delta == pytest.approx(-0.2)

    # Now paged at step 8, gap=2, within window → no violation.
    state2 = _mk_state(
        past=[{"app": "chatops", "op": "page_oncall", "args": {}, "at_step": 8}],
        step=10,
    )
    delta2, _ = engine.evaluate(action, state2)
    assert delta2 == 0.0


def test_max_occurrences_caps_repeated_matches():
    engine = PolicyEngine(
        rules=[
            {
                "id": "dont-spam-page",
                "when": {"app": "chatops", "op": "page_oncall"},
                "max_occurrences": 2,
                "penalty": -0.1,
            }
        ]
    )
    action = IncidentTriageAction(app="chatops", op="page_oncall", args={"role": "sre"})
    # Three past pages (including current) → count=3 > cap=2 → violation.
    state = _mk_state(
        past=[
            {"app": "chatops", "op": "page_oncall", "args": {}, "at_step": 1},
            {"app": "chatops", "op": "page_oncall", "args": {}, "at_step": 2},
            {"app": "chatops", "op": "page_oncall", "args": {}, "at_step": 3},
        ],
        step=3,
    )
    delta, violations = engine.evaluate(action, state)
    assert delta == pytest.approx(-0.1)
    assert "dont-spam-page" in violations


def test_summary_aggregates_and_compliance_score():
    engine = PolicyEngine(
        rules=[
            {"id": "r1", "when": {"app": "repohub", "op": "open_pr"},
             "forbidden_if": {"scenario_tag": "freeze"}, "penalty": -0.4},
        ],
        tags=["freeze"],
    )
    action = IncidentTriageAction(
        app="repohub", op="open_pr",
        args={"target_repo": "r", "head_branch": "b", "title": "t"},
    )
    state = _mk_state(step=1)
    engine.evaluate(action, state)
    summary = engine.summary(state)
    assert summary["num_violations"] == 1
    assert summary["total_penalty"] == pytest.approx(-0.4)
    assert summary["total_bonus"] == 0.0
    assert summary["compliance_score"] == pytest.approx(0.6)


def test_no_policies_is_zero_delta():
    engine = PolicyEngine(rules=[])
    action = IncidentTriageAction(action_type="query_logs", service="x")
    state = _mk_state(step=1)
    delta, violations = engine.evaluate(action, state)
    assert delta == 0.0
    assert violations == []


# --------------------------------------------------------- env integration


def test_env_surfaces_policy_delta_on_obs():
    env = IncidentTriageEnv()
    env._loader._scenarios["easy_single_service_failure"]["policies"] = [
        {
            "id": "forbid-open-pr-in-freeze",
            "when": {"app": "repohub", "op": "open_pr"},
            "forbidden_if": {"scenario_tag": "change_freeze"},
            "penalty": -0.5,
        }
    ]
    env._loader._scenarios["easy_single_service_failure"]["tags"] = ["change_freeze"]
    try:
        env.reset(task_id="easy_single_service_failure")
        obs = env.step(IncidentTriageAction(
            app="repohub", op="open_pr",
            args={"target_repo": "r", "head_branch": "b", "title": "t"},
        ))
        assert obs.policy_delta_this_step == pytest.approx(-0.5)
        assert "forbid-open-pr-in-freeze" in obs.policy_violations_this_step
        assert obs.reward is not None and obs.reward <= -0.4  # rubric + -0.5 policy
    finally:
        env._loader._scenarios["easy_single_service_failure"].pop("policies", None)
        env._loader._scenarios["easy_single_service_failure"].pop("tags", None)


def test_env_submit_diagnosis_respects_require_prior():
    env = IncidentTriageEnv()
    env._loader._scenarios["easy_single_service_failure"]["policies"] = [
        {
            "id": "page-before-rollback",
            "when": {
                "app": "system",
                "op": "submit_diagnosis",
                "args_match": {"remediation": "rollback_deploy"},
            },
            "require_prior": {"app": "chatops", "op": "page_oncall"},
            "penalty": -0.3,
            "bonus": 0.1,
        }
    ]
    try:
        # Case A: no prior page → penalty.
        env.reset(task_id="easy_single_service_failure", episode_id="no-page")
        obs = env.step(IncidentTriageAction(
            action_type="submit_diagnosis",
            root_cause_service="payments-service",
            root_cause_category=RootCauseCategory.BAD_DEPLOY,
            remediation=Remediation.ROLLBACK_DEPLOY,
        ))
        assert "page-before-rollback" in obs.policy_violations_this_step
        assert obs.policy_delta_this_step == pytest.approx(-0.3)

        # Case B: page before submit → bonus.
        env.reset(task_id="easy_single_service_failure", episode_id="with-page")
        env.step(IncidentTriageAction(
            app="chatops", op="page_oncall", args={"role": "sre"},
        ))
        obs = env.step(IncidentTriageAction(
            action_type="submit_diagnosis",
            root_cause_service="payments-service",
            root_cause_category=RootCauseCategory.BAD_DEPLOY,
            remediation=Remediation.ROLLBACK_DEPLOY,
        ))
        assert obs.policy_violations_this_step == []
        assert obs.policy_delta_this_step == pytest.approx(0.1)
    finally:
        env._loader._scenarios["easy_single_service_failure"].pop("policies", None)
