"""Tier A tests — UATSim app + RepoHub CI gates + their scenarios."""

from __future__ import annotations

import json

import pytest

from models import (
    AppName,
    IncidentTriageAction,
    IncidentTriageState,
    RootCauseCategory,
    Remediation,
)
from server.apps import APP_DISPATCH, uatsim
from server.apps.repohub import dispatch as repohub_dispatch
from server.environment import IncidentTriageEnv
from server.scenario_loader import EVAL_TASK_IDS, TASK_FILES, TRAIN_TASK_IDS, ScenarioLoader


# --------------------------------------------------------- UATSim dispatcher


@pytest.fixture
def uat_scenario():
    return {
        "uatsim": {
            "stages": [
                {"id": "smoke", "name": "smoke", "required": True, "status": "passed"},
                {"id": "regression", "name": "regression", "required": True,
                 "status": "skipped", "reason": "release-captain override"},
            ],
            "signoffs": [
                {"role": "qa-lead", "user": "mira", "signed": False},
                {"role": "product", "user": "priya", "signed": True},
            ],
            "uat_records": {
                "checkout-service": {
                    "passed_stages": ["smoke"],
                    "skipped_stages": ["regression"],
                    "signoff_complete": False,
                    "deploy_shipped_anyway": True,
                }
            },
        }
    }


def test_uatsim_is_registered_in_dispatch():
    assert "uatsim" in APP_DISPATCH


def test_list_stages_shows_skipped_flag(uat_scenario):
    state = IncidentTriageState()
    out = uatsim.dispatch("list_stages", {}, uat_scenario, state)
    assert "SKIPPED" in out
    assert "release-captain override" in out
    assert "REQUIRED" in out


def test_get_signoff_status_reports_missing(uat_scenario):
    state = IncidentTriageState()
    out = uatsim.dispatch("get_signoff_status", {}, uat_scenario, state)
    data = json.loads(out)
    assert data["complete"] is False
    assert any(m["role"] == "qa-lead" for m in data["missing"])


def test_check_uat_record_marks_state(uat_scenario):
    state = IncidentTriageState()
    out = uatsim.dispatch("check_uat_record", {"service": "checkout-service"},
                          uat_scenario, state)
    data = json.loads(out)
    assert data["signoff_complete"] is False
    assert data["deploy_shipped_anyway"] is True
    # Side-effect: state records the check so graders/policies can see it.
    assert "checkout-service" in state.evidence_collected["__uat_checked__"]


def test_check_uat_record_missing_service_returns_neutral(uat_scenario):
    state = IncidentTriageState()
    out = uatsim.dispatch("check_uat_record", {"service": "nonexistent"},
                          uat_scenario, state)
    data = json.loads(out)
    assert data["signoff_complete"] is True
    assert data["deploy_shipped_anyway"] is False


# --------------------------------------------------------- RepoHub CI gate


def test_repohub_ci_check_default_report_is_healthy():
    state = IncidentTriageState()
    out = repohub_dispatch(
        "ci_check", {"target_repo": "some/repo"}, {"services": {}}, state
    )
    data = json.loads(out)
    assert data["gate_status"] == "passed"
    assert data["coverage_pct"] == 92.0
    assert data["snyk"]["high"] == 0


def test_repohub_ci_check_returns_scenario_override():
    scenario = {
        "services": {},
        "repohub": {
            "ci_state": {
                "platform/notifications": {
                    "target_repo": "platform/notifications",
                    "coverage_pct": 74.2,
                    "coverage_delta_pct": -12.8,
                    "snyk": {"high": 1, "medium": 0, "low": 0, "waived": ["CVE-X"]},
                    "sonar": {"quality_gate": "failed"},
                    "raven": {"secrets_found": 0},
                    "gate_status": "failed_but_merged",
                }
            }
        },
    }
    state = IncidentTriageState()
    out = repohub_dispatch("ci_check", {"target_repo": "platform/notifications"},
                            scenario, state)
    data = json.loads(out)
    assert data["gate_status"] == "failed_but_merged"
    assert data["snyk"]["high"] == 1
    # Side-effect: records that CI was queried.
    assert "platform/notifications" in state.evidence_collected["__ci_checked__"]


def test_repohub_list_pr_history_surfaces_waived_findings():
    scenario = {
        "services": {},
        "repohub": {
            "pr_history": {
                "platform/notifications": [
                    {"number": 4412, "title": "v2.2", "author": "n",
                     "merged": True, "merged_at": "t", "ci_gate_status": "failed",
                     "waived_findings": ["CVE-2025-30417"]},
                ]
            }
        },
    }
    out = repohub_dispatch("list_pr_history", {"target_repo": "platform/notifications"},
                            scenario, IncidentTriageState())
    assert "CVE-2025-30417" in out
    assert "gate=failed" in out


# --------------------------------------------------------- scenario integration


def test_uat_scenario_is_registered_and_loadable():
    loader = ScenarioLoader()
    s = loader.get_scenario("medium_uat_skipped")
    assert "uatsim" in s
    assert "uat_bypassed" in s["tags"]
    assert s["ground_truth"]["root_cause_service"] == "checkout-service"


def test_pr_quality_scenario_is_registered_and_loadable():
    loader = ScenarioLoader()
    s = loader.get_scenario("hard_pr_quality_breach")
    ci = s["repohub"]["ci_state"]["platform/notifications-service"]
    assert ci["gate_status"] == "failed_but_merged"
    assert "CVE-2025-30417" in ci["snyk"]["waived"]
    assert "snyk_high_open" in s["tags"]


def test_pr_quality_scenario_in_eval_split():
    assert "hard_pr_quality_breach" in EVAL_TASK_IDS


def test_uat_scenario_in_train_split():
    assert "medium_uat_skipped" in TRAIN_TASK_IDS


# --------------------------------------------------------- end-to-end env runs


def test_uat_scenario_bonus_for_checking_uat_before_diagnosis():
    env = IncidentTriageEnv()
    env.reset(task_id="medium_uat_skipped")
    # check UAT record first
    env.step(IncidentTriageAction(
        app="uatsim", op="check_uat_record",
        args={"service": "checkout-service"},
    ))
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="checkout-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    # Both "check_uat_record → diagnosis" bonus and "rollback_deploy" bonus fire.
    assert obs.policy_delta_this_step > 0
    assert obs.policy_violations_this_step == []


def test_uat_scenario_penalty_for_skipping_uat_check():
    env = IncidentTriageEnv()
    env.reset(task_id="medium_uat_skipped")
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="checkout-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    # The require_prior UAT bonus should not fire; penalty applies.
    assert "bonus-check-uat-gate-before-diagnosis" in obs.policy_violations_this_step


def test_pr_quality_scenario_forbids_forward_fix_pr():
    env = IncidentTriageEnv()
    env.reset(task_id="hard_pr_quality_breach")
    obs = env.step(IncidentTriageAction(
        app="repohub", op="open_pr",
        args={
            "target_repo": "platform/notifications-service",
            "head_branch": "fix/ssrf-patch",
            "title": "forward-fix SSRF",
        },
    ))
    assert "forbid-forward-fix-pr-with-open-vulns" in obs.policy_violations_this_step
    assert obs.policy_delta_this_step < 0


def test_pr_quality_scenario_rewards_ci_check_then_rollback():
    env = IncidentTriageEnv()
    env.reset(task_id="hard_pr_quality_breach")
    env.step(IncidentTriageAction(
        app="repohub", op="ci_check",
        args={"target_repo": "platform/notifications-service"},
    ))
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="notifications-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    assert obs.policy_delta_this_step > 0
    assert obs.policy_violations_this_step == []


# --------------------------------------------------------- scenario shape


@pytest.mark.parametrize("task_id", ["medium_uat_skipped", "hard_pr_quality_breach"])
def test_tier_a_scenarios_have_full_phase5_ground_truth(task_id):
    loader = ScenarioLoader()
    gt = loader.get_scenario(task_id)["ground_truth"]
    assert "correct_pr" in gt
    assert "correct_blast_radius" in gt
    for key in ("target_repo", "touched_files", "keywords"):
        assert key in gt["correct_pr"]
    assert "affected_services" in gt["correct_blast_radius"]
