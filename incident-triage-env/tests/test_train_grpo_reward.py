"""Unit tests for the Phase 7 GRPO reward function (replay-and-grade).

These tests don't load an LLM — they only exercise the parsing + env-replay
path, which is CPU-only and runs in ~1 second. The goal is to catch
regressions in the reward signal before burning GPU time.
"""

from __future__ import annotations

import json

import pytest

from scripts.train_grpo import (
    parse_actions_from_completion,
    replay_and_grade,
    make_reward_func,
)


def test_parse_actions_accepts_clean_jsonl():
    comp = (
        '{"action_type":"check_status","service":"payments-service"}\n'
        '{"action_type":"query_logs","service":"payments-service","severity":"error"}\n'
    )
    actions = parse_actions_from_completion(comp)
    assert len(actions) == 2
    assert actions[0]["action_type"] == "check_status"


def test_parse_actions_tolerates_fences_and_blanks():
    comp = (
        "```json\n"
        '{"action_type":"check_status","service":"svc"}\n'
        "\n"
        "```\n"
        '{"action_type":"trace_dependencies","service":"svc"}\n'
    )
    actions = parse_actions_from_completion(comp)
    assert len(actions) == 2


def test_parse_actions_drops_prose_lines():
    comp = (
        "Let me investigate:\n"
        '{"action_type":"check_status","service":"svc"}\n'
        "Now checking logs\n"
    )
    actions = parse_actions_from_completion(comp)
    assert len(actions) == 1


def test_replay_correct_trajectory_scores_above_floor():
    """A valid submit_diagnosis with the right answer must score well above floor."""
    actions = [
        {"action_type": "check_status", "service": "payments-service"},
        {"action_type": "query_logs", "service": "payments-service", "severity": "error"},
        {"action_type": "check_deploys", "service": "payments-service"},
        {
            "action_type": "submit_diagnosis",
            "root_cause_service": "payments-service",
            "root_cause_category": "bad_deploy",
            "remediation": "rollback_deploy",
        },
    ]
    score = replay_and_grade("easy_single_service_failure", seed=0, actions=actions)
    assert score > 0.2, f"expected >0.2 for correct diagnosis, got {score}"


def test_replay_wrong_diagnosis_scores_lower_than_correct():
    correct = [{
        "action_type": "submit_diagnosis",
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy",
    }]
    wrong = [{
        "action_type": "submit_diagnosis",
        "root_cause_service": "database",
        "root_cause_category": "traffic_spike",
        "remediation": "scale_up",
    }]
    s_correct = replay_and_grade("easy_single_service_failure", 0, correct)
    s_wrong = replay_and_grade("easy_single_service_failure", 0, wrong)
    assert s_correct > s_wrong, f"correct {s_correct} must beat wrong {s_wrong}"


def test_replay_empty_trajectory_returns_floor():
    assert replay_and_grade("easy_single_service_failure", 0, []) == pytest.approx(0.001)


def test_replay_junk_trajectory_returns_floor():
    junk = [{"action_type": "not_a_real_action", "service": "svc"}]
    score = replay_and_grade("easy_single_service_failure", 0, junk)
    assert score <= 0.05


def test_reward_func_signature_matches_trl():
    """TRL calls reward_fn(prompts, completions, **row_kwargs)."""
    fn = make_reward_func()
    completions = [
        '{"action_type":"submit_diagnosis","root_cause_service":"payments-service",'
        '"root_cause_category":"bad_deploy","remediation":"rollback_deploy"}',
        '{"action_type":"submit_diagnosis","root_cause_service":"database",'
        '"root_cause_category":"traffic_spike","remediation":"scale_up"}',
    ]
    rewards = fn(
        prompts=["p1", "p2"],
        completions=completions,
        task_id=["easy_single_service_failure", "easy_single_service_failure"],
        seed=[0, 0],
    )
    assert len(rewards) == 2
    assert rewards[0] > rewards[1], "correct answer must score higher"


def test_reward_func_handles_chat_shaped_completions():
    """Newer TRL versions return completions as list[{role, content}] messages."""
    fn = make_reward_func()
    chat_completion = [
        {"role": "assistant", "content":
            '{"action_type":"submit_diagnosis","root_cause_service":"payments-service",'
            '"root_cause_category":"bad_deploy","remediation":"rollback_deploy"}'}
    ]
    rewards = fn(
        prompts=["p1"],
        completions=[chat_completion],
        task_id=["easy_single_service_failure"],
        seed=[0],
    )
    assert rewards[0] > 0.2
