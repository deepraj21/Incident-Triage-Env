"""Tests for Phase 4 — scenario authoring + variant generator + loader split."""

from __future__ import annotations

import pytest

from models import IncidentTriageAction, RootCauseCategory, Remediation
from server.environment import IncidentTriageEnv
from server.scenario_loader import (
    EVAL_TASK_IDS,
    TASK_FILES,
    TRAIN_TASK_IDS,
    ScenarioLoader,
)
from server.scenario_variants import apply_variant


# --------------------------------------------------- hero scenario coverage


@pytest.fixture(scope="module")
def loader():
    return ScenarioLoader()


@pytest.mark.parametrize("task_id", list(TASK_FILES.keys()))
def test_every_scenario_has_phase2_phase3_blocks(loader, task_id):
    s = loader.get_scenario(task_id)
    assert "clock" in s
    assert "start" in s["clock"]
    assert "step_seconds" in s["clock"]
    assert "timeline" in s
    assert "policies" in s
    assert "tags" in s
    # Multi-app surface present
    for required in ("alerthub", "chatops", "ticketdesk", "obsly"):
        assert required in s, f"{task_id} missing {required}"
    # Ground-truth contract intact
    gt = s["ground_truth"]
    for key in ("root_cause_service", "root_cause_category", "remediation",
                "causal_chain"):
        assert key in gt, f"{task_id} ground_truth missing {key}"


def test_train_eval_split_is_disjoint():
    train = set(TRAIN_TASK_IDS)
    eval_ = set(EVAL_TASK_IDS)
    assert train.isdisjoint(eval_)
    assert train | eval_ == set(TASK_FILES.keys())
    assert len(eval_) >= 2, "need at least 2 held-out tasks for before/after eval"


@pytest.mark.parametrize("task_id", list(TASK_FILES.keys()))
def test_new_scenarios_are_playable_end_to_end(task_id):
    env = IncidentTriageEnv()
    obs = env.reset(task_id=task_id)
    assert obs.sim_time is not None
    # Do a couple of benign queries to verify dispatchers survive the scenario.
    alert_svc = obs.alert.service
    env.step(IncidentTriageAction(action_type="query_logs", service=alert_svc))
    env.step(IncidentTriageAction(action_type="check_deploys", service=alert_svc))
    # Submit a diagnosis using ground-truth to ensure the grader accepts it.
    gt = env._loader.get_scenario(task_id)["ground_truth"]
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service=gt["root_cause_service"],
        root_cause_category=RootCauseCategory(gt["root_cause_category"]),
        remediation=Remediation(gt["remediation"]),
    ))
    assert obs.done is True
    assert "Grader score" in obs.result.data


# --------------------------------------------------- variant generator


def test_variant_is_deterministic_for_same_seed(loader):
    s = loader.get_scenario("easy_single_service_failure")
    a = apply_variant(s, seed=7)
    b = apply_variant(s, seed=7)
    assert a == b


def test_variant_differs_across_seeds(loader):
    s = loader.get_scenario("easy_single_service_failure")
    a = apply_variant(s, seed=1)
    b = apply_variant(s, seed=9999)
    # At minimum the oncall lead rename should make them non-identical.
    assert a != b


def test_variant_preserves_ground_truth(loader):
    for task_id in TASK_FILES:
        s = loader.get_scenario(task_id)
        for seed in (0, 1, 42, 777):
            v = apply_variant(s, seed=seed)
            assert v["ground_truth"] == s["ground_truth"], (
                f"{task_id} ground_truth mutated by seed={seed}"
            )
            assert v["task_id"] == s["task_id"]
            assert v["step_budget"] == s["step_budget"]


def test_variant_preserves_root_cause_logs(loader):
    """Logs mentioning the root-cause service must never be hidden."""
    s = loader.get_scenario("hard_region_failover")
    root_svc = s["ground_truth"]["root_cause_service"]
    original_root_logs = [
        log for log in s["services"][root_svc]["logs"]
        if root_svc.lower() in log["message"].lower()
    ]
    for seed in (1, 2, 3, 4, 5):
        v = apply_variant(s, seed=seed)
        variant_root_logs = [
            log for log in v["services"][root_svc]["logs"]
            if root_svc.lower() in log["message"].lower()
        ]
        # same set of root-cause messages, order may differ
        assert sorted(l["message"] for l in original_root_logs) == \
               sorted(l["message"] for l in variant_root_logs)


def test_env_reset_with_seed_is_stable():
    env = IncidentTriageEnv()
    env.reset(task_id="easy_single_service_failure", seed=42, episode_id="a")
    a_logs = env._current_scenario["services"]["payments-service"]["logs"]
    env.reset(task_id="easy_single_service_failure", seed=42, episode_id="b")
    b_logs = env._current_scenario["services"]["payments-service"]["logs"]
    assert a_logs == b_logs


def test_env_reset_different_seeds_diverge():
    env = IncidentTriageEnv()
    env.reset(task_id="easy_single_service_failure", seed=1, episode_id="a")
    a_lead = env._current_scenario["chatops"]["oncall_lead"]
    found_different = False
    for seed in range(2, 30):
        env.reset(task_id="easy_single_service_failure", seed=seed, episode_id=f"s{seed}")
        if env._current_scenario["chatops"]["oncall_lead"] != a_lead:
            found_different = True
            break
    assert found_different, "at least one seed in [2,30) should rename oncall lead"


# --------------------------------------------------- freeze-violation scenario policy


def test_freeze_scenario_penalizes_open_pr():
    env = IncidentTriageEnv()
    env.reset(task_id="hard_freeze_violation")
    # open_pr during freeze should trigger the policy penalty.
    obs = env.step(IncidentTriageAction(
        app="repohub", op="open_pr",
        args={"target_repo": "orders/main", "head_branch": "fix/bad", "title": "forward fix"}
    ))
    assert "forbid-pr-during-freeze" in obs.policy_violations_this_step
    assert obs.policy_delta_this_step < 0


def test_freeze_scenario_rewards_rollback_diagnosis():
    env = IncidentTriageEnv()
    env.reset(task_id="hard_freeze_violation")
    # page first so we get the page-before-rollback bonus too
    env.step(IncidentTriageAction(app="chatops", op="page_oncall", args={"role": "orders-lead"}))
    obs = env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="orders-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    # Both the rollback-bonus and the page-before-rollback bonus should fire; none violated.
    assert obs.policy_delta_this_step > 0
    assert obs.policy_violations_this_step == []


# --------------------------------------------------- stealth-regression silent events


def test_stealth_regression_events_are_silent():
    """The expert scenario's events fire silently — agent must re-poll to notice."""
    env = IncidentTriageEnv()
    env.reset(task_id="expert_stealth_regression")
    # Walk through a few benign steps; world_events on obs should stay empty
    # because all timeline entries have visible=false.
    any_visible = False
    for _ in range(5):
        obs = env.step(IncidentTriageAction(
            action_type="query_metrics",
            service="ml-model-server",
            metric="latency_p99",
        ))
        if obs.world_events:
            any_visible = True
            break
    assert not any_visible, "stealth scenario should not surface visible world events"
    # But the underlying scenario WAS mutated by the silent events.
    state = env.state
    assert len(state.event_log) >= 1
    assert all(e["visible"] is False for e in state.event_log)
