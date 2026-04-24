"""Tests for Phase 5 — multi-head grader (diagnosis + policy + blast + PR)."""

from __future__ import annotations

import pytest

from models import IncidentTriageAction, RootCauseCategory, Remediation
from server.environment import IncidentTriageEnv
from server.grader import (
    BlastRadiusScorer,
    CompositeScorer,
    PolicyComplianceScorer,
    PRProposalScorer,
    _f1,
    _window_overlap_fraction,
    _parse_iso,
)
from server.scenario_loader import TASK_FILES, ScenarioLoader


# ---------------------------------------------------------------- helpers


def test_f1_basic():
    assert _f1(set(), set()) == 1.0
    assert _f1({"a"}, set()) == 0.0
    assert _f1({"a", "b"}, {"a", "b"}) == 1.0
    assert 0 < _f1({"a", "b"}, {"a", "c"}) < 1


def test_window_overlap_perfect_and_disjoint():
    a = _parse_iso("2026-01-01T00:00:00Z")
    b = _parse_iso("2026-01-01T01:00:00Z")
    assert _window_overlap_fraction(a, b, a, b) == 1.0
    c = _parse_iso("2026-01-01T02:00:00Z")
    d = _parse_iso("2026-01-01T03:00:00Z")
    assert _window_overlap_fraction(a, b, c, d) == 0.0


# ---------------------------------------------------------------- PR scorer


def test_pr_scorer_perfect_match():
    gt = {
        "target_repo": "fintra/orders",
        "touched_files": ["OrderService.java"],
        "keywords": ["rollback", "v8.1"],
    }
    pr = {
        "target_repo": "fintra/orders",
        "head_branch": "revert/v8.2",
        "title": "Rollback to v8.1",
        "summary": "Reverting v8.2 which added strict discount_code check. See OrderService.java.",
        "diff_patch": "@@ OrderService.java @@ - throw ...",
    }
    out = PRProposalScorer().score(gt, pr)
    assert out["applicable"] is True
    assert out["score"] > 0.8


def test_pr_scorer_missing_submission():
    gt = {"target_repo": "x", "touched_files": ["a.py"], "keywords": ["foo"]}
    out = PRProposalScorer().score(gt, None)
    assert out["applicable"] is True
    assert out["score"] == 0.0


def test_pr_scorer_inapplicable_when_no_gt():
    out = PRProposalScorer().score({}, {"target_repo": "x"})
    assert out["applicable"] is False
    assert out["score"] is None


# ---------------------------------------------------------------- blast scorer


def test_blast_scorer_perfect():
    gt = {
        "affected_services": ["svc-a", "svc-b"],
        "estimated_requests_failed": 1000,
        "missed_regions": ["us-east-1"],
        "outage_window_start": "2026-01-01T00:00:00Z",
        "outage_window_end": "2026-01-01T00:30:00Z",
    }
    sub = dict(gt)
    out = BlastRadiusScorer().score(gt, sub)
    assert out["score"] > 0.95


def test_blast_scorer_partial_credit_on_magnitude():
    gt = {"affected_services": ["a"], "estimated_requests_failed": 1000,
          "missed_regions": [], "outage_window_start": None, "outage_window_end": None}
    sub = {"affected_services": ["a"], "estimated_requests_failed": 900,
           "missed_regions": [], "outage_window_start": None, "outage_window_end": None}
    out = BlastRadiusScorer().score(gt, sub)
    # close magnitude (<0.5 dex) should give full requests_failed credit
    assert out["breakdown"]["requests_failed"] == pytest.approx(0.20, abs=1e-3)


def test_blast_scorer_zero_on_none_submission():
    gt = {"affected_services": ["a"], "estimated_requests_failed": 10}
    assert BlastRadiusScorer().score(gt, None)["score"] == 0.0


# ---------------------------------------------------------------- policy


def test_policy_compliance_scorer_passes_through():
    summary = {"compliance_score": 0.7, "num_violations": 1,
               "total_penalty": -0.3, "total_bonus": 0.0}
    out = PolicyComplianceScorer().score(summary)
    assert out["score"] == 0.7
    assert out["breakdown"]["num_violations"] == 1


# ---------------------------------------------------------------- composite


def test_composite_rejects_zero_when_no_diagnosis():
    out = CompositeScorer().score(ground_truth={}, diagnosis={}, episode_state=None)
    assert out["score"] == 0.0
    assert out["diagnosis_submitted"] is False


def test_composite_redistributes_weights_when_pr_missing():
    """Scenario has no correct_pr → PR head inapplicable, weight redistributes."""
    env = IncidentTriageEnv()
    # Use easy scenario; temporarily drop its correct_pr so PR head is inapplicable.
    loader = env._loader
    easy = loader.get_scenario("easy_single_service_failure")
    saved_pr = easy["ground_truth"].pop("correct_pr", None)
    saved_br = easy["ground_truth"].pop("correct_blast_radius", None)
    try:
        env.reset(task_id="easy_single_service_failure")
        obs = env.step(IncidentTriageAction(
            action_type="submit_diagnosis",
            root_cause_service="payments-service",
            root_cause_category=RootCauseCategory.BAD_DEPLOY,
            remediation=Remediation.ROLLBACK_DEPLOY,
        ))
        bd = obs.grader_breakdown
        eff = bd["effective_weights"]
        assert "pr" not in eff
        assert "blast" not in eff
        # diagnosis and policy should absorb the freed weight (sum to 1)
        assert pytest.approx(sum(eff.values()), abs=1e-4) == 1.0
    finally:
        if saved_pr is not None:
            easy["ground_truth"]["correct_pr"] = saved_pr
        if saved_br is not None:
            easy["ground_truth"]["correct_blast_radius"] = saved_br


def test_composite_uses_all_four_heads_when_scenario_has_ground_truth():
    env = IncidentTriageEnv()
    env.reset(task_id="hard_freeze_violation")
    env.step(IncidentTriageAction(app="chatops", op="page_oncall", args={"role": "orders-lead"}))
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="orders-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
        pr_proposal={
            "target_repo": "commerce/orders-service",
            "head_branch": "revert/v8.2",
            "title": "Rollback to v8.1",
            "summary": "Revert v8.2 discount_code validation in OrderService.java",
            "diff_patch": "@@ OrderService.java @@",
        },
        blast_radius={
            "affected_services": ["orders-service", "api-gateway"],
            "estimated_requests_failed": 4000,
            "missed_regions": [],
            "outage_window_start": "2026-03-27T02:08:00Z",
            "outage_window_end": "2026-03-27T02:30:00Z",
        },
    ))
    bd = obs.grader_breakdown
    eff = bd["effective_weights"]
    # All four heads should be active.
    assert set(eff.keys()) == {"diagnosis", "policy", "blast", "pr"}
    assert pytest.approx(sum(eff.values()), abs=1e-4) == 1.0
    # Each head exposes its own sub-breakdown for the storytelling heatmap.
    for head in ("diagnosis", "policy", "blast", "pr"):
        assert "score" in bd["heads"][head]


def test_composite_per_head_scores_are_independent():
    """Submitting a correct diagnosis with junk blast radius should hurt the
    blast head without wiping the diagnosis head."""
    env = IncidentTriageEnv()
    env.reset(task_id="hard_freeze_violation")
    env.step(IncidentTriageAction(app="chatops", op="page_oncall", args={"role": "orders-lead"}))
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="orders-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
        blast_radius={
            "affected_services": ["unrelated"],
            "estimated_requests_failed": 1,
            "missed_regions": ["mars"],
        },
    ))
    heads = obs.grader_breakdown["heads"]
    assert heads["diagnosis"]["score"] > 0.3
    assert heads["blast"]["score"] < 0.3


# ---------------------------------------------------------------- scenario coverage


@pytest.mark.parametrize("task_id", list(TASK_FILES.keys()))
def test_all_scenarios_have_phase5_ground_truth(task_id):
    loader = ScenarioLoader()
    gt = loader.get_scenario(task_id)["ground_truth"]
    assert "correct_pr" in gt, f"{task_id} missing correct_pr"
    assert "correct_blast_radius" in gt, f"{task_id} missing correct_blast_radius"
    pr = gt["correct_pr"]
    for k in ("target_repo", "touched_files", "keywords"):
        assert k in pr
    br = gt["correct_blast_radius"]
    for k in ("affected_services", "estimated_requests_failed"):
        assert k in br
