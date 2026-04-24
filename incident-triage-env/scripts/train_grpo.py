"""Phase 7 — GRPO training for incident_triage_env.

Single-turn GRPO formulation (TRL-native, Colab-ready, scales to HF Spaces):

  prompt     = system-instructions + initial alert
  completion = model-generated JSON-action trajectory (one action per line)
  reward     = replay trajectory in a fresh env copy → composite grader score

Why single-turn: TRL's `GRPOTrainer` expects one generation per prompt. A
multi-turn rollout can be simulated inside the reward function by parsing the
completion into a list of actions and replaying them in the env — this keeps
the training loop clean and portable between Colab, Kaggle, and HF Spaces.

Hardware presets:
  --model Qwen/Qwen2.5-0.5B-Instruct  --lora-r 8     # M4 Air / Colab free-T4
  --model Qwen/Qwen2.5-1.5B-Instruct  --lora-r 16    # Kaggle P100
  --model Qwen/Qwen2.5-3B-Instruct    --lora-r 32    # HF Spaces A10G / A100

Run:
  # 1-iteration smoke test (uses the smallest model, group=2)
  python scripts/train_grpo.py --dry-run

  # real training run
  python scripts/train_grpo.py \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --num-iters 200 --group-size 4 --seeds 0 1 2 3

Outputs: a LoRA adapter directory under --output-dir that can be loaded with
`PeftModel.from_pretrained(base, adapter_path)` for eval via inference.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the env importable when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IncidentTriageAction  # noqa: E402
from server.environment import IncidentTriageEnv  # noqa: E402
from server.scenario_loader import TRAIN_TASK_IDS  # noqa: E402

# Heavy ML deps are imported lazily inside main() so that the parse + replay
# helpers (which are pure Python) can be unit-tested on machines without a
# torch / trl / peft install.


# --------------------------------------------------------------- prompt template

TRAIN_SYSTEM_PROMPT = """You are an SRE diagnosing a production incident. Output a JSON-action trajectory — one action per line — ending with submit_diagnosis.

Line format: each line is exactly one JSON object. No prose, no markdown fences, no commentary.

Investigation actions:
  {"action_type":"check_status","service":"<svc>"}
  {"action_type":"trace_dependencies","service":"<svc>"}
  {"action_type":"query_logs","service":"<svc>","severity":"error"}
  {"action_type":"query_metrics","service":"<svc>","metric":"latency_p99|error_rate|cpu|memory"}
  {"action_type":"check_deploys","service":"<svc>"}
  {"action_type":"inspect_code","service":"<svc>"}

Multi-app actions (optional but scored):
  {"app":"chatops","op":"page_oncall","args":{"role":"<role>"}}
  {"app":"alerthub","op":"list_alerts","args":{}}

Terminal (REQUIRED):
  {"action_type":"submit_diagnosis","root_cause_service":"<svc>","root_cause_category":"<cat>","remediation":"<rem>"}

Categories:    bad_deploy  resource_exhaustion  dependency_failure  config_change  traffic_spike  data_corruption
Remediations:  rollback_deploy  scale_up  restart_service  fix_config  enable_rate_limiting  failover_to_backup

Strategy: trace upstream from the alerted service; the root cause is the deepest upstream with real evidence, not something relaying errors. Respect step budget.
"""


def build_prompt(tokenizer, alert: dict, step_budget: int) -> str:
    """Render the training prompt via the tokenizer's chat template."""
    user_msg = (
        f"INCIDENT ALERT [{str(alert.get('severity', '?')).upper()}] "
        f"{alert.get('service', '?')}: {alert.get('message', '?')}\n"
        f"Sim-time: {alert.get('timestamp', '?')}  •  Step budget: {step_budget}"
    )
    messages = [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_training_dataset(tokenizer, seeds: list[int]):
    """One row per (task_id, seed) from the TRAIN split.

    Each row carries the prompt plus `task_id` + `seed` so the reward
    function can replay the trajectory against the correct scenario.
    """
    from datasets import Dataset  # lazy import — keeps unit tests torch-free
    rows: list[dict[str, Any]] = []
    probe = IncidentTriageEnv()
    for task_id in TRAIN_TASK_IDS:
        for seed in seeds:
            obs = probe.reset(
                task_id=task_id, seed=seed, episode_id=f"ds-{task_id}-{seed}"
            )
            rows.append({
                "prompt": build_prompt(
                    tokenizer, obs.alert.model_dump(), probe._step_budget
                ),
                "task_id": task_id,
                "seed": seed,
            })
    return Dataset.from_list(rows)


# --------------------------------------------------------------- parse + grade

def parse_actions_from_completion(text: str) -> list[dict]:
    """Greedy line-by-line JSON parse; tolerant of stray fences and whitespace."""
    actions: list[dict] = []
    for line in (text or "").splitlines():
        stripped = line.strip().strip("`")
        if not stripped or not stripped.startswith("{"):
            continue
        # Some models emit trailing commas — try a second parse after a cleanup.
        for candidate in (stripped, stripped.rstrip(",")):
            try:
                actions.append(json.loads(candidate))
                break
            except json.JSONDecodeError:
                continue
    return actions


def replay_and_grade(task_id: str, seed: int | None, actions: list[dict]) -> float:
    """Replay a parsed trajectory against a fresh env and return composite score.

    The reward is the terminal grader score (0.001–0.999). Tiny floors are
    applied so completely-malformed completions still rank below partial ones,
    which gives GRPO a usable gradient.
    """
    env = IncidentTriageEnv()
    env.reset(
        task_id=task_id,
        seed=seed,
        episode_id=f"rollout-{task_id}-{seed if seed is not None else 0}",
    )
    floor = 0.001
    if not actions:
        return floor

    saw_submit = False
    for raw in actions:
        try:
            action = IncidentTriageAction.model_validate(raw)
        except Exception:
            continue
        obs = env.step(action)
        if obs.done:
            saw_submit = True
            if obs.grader_breakdown:
                return float(obs.grader_breakdown.get("score", floor))
            break
    # Finished the loop without a submit — small credit for any valid actions
    # so the model learns "acting > spamming junk" before it learns "submit".
    return 0.01 if saw_submit is False else floor


def make_reward_func():
    """TRL reward-function signature: fn(prompts, completions, **meta) → list[float]."""

    def _reward(prompts: list[str], completions: list[str], **meta) -> list[float]:
        # TRL passes extra dataset columns as kwargs (lists, one per row).
        task_ids = meta.get("task_id") or []
        seeds = meta.get("seed") or []
        scores: list[float] = []
        for i, comp in enumerate(completions):
            # `completions` may be structured as chat messages in newer TRL
            # versions — normalise to a plain string either way.
            if isinstance(comp, list):
                comp_text = "".join(m.get("content", "") for m in comp if isinstance(m, dict))
            else:
                comp_text = str(comp)
            actions = parse_actions_from_completion(comp_text)
            tid = task_ids[i] if i < len(task_ids) else TRAIN_TASK_IDS[0]
            seed = seeds[i] if i < len(seeds) else None
            scores.append(replay_and_grade(tid, seed, actions))
        return scores

    return _reward


# --------------------------------------------------------------- device / args

def _pick_device(torch_mod) -> str:
    if torch_mod.cuda.is_available():
        return "cuda"
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO training for incident_triage_env")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Base model. 0.5B fits M4 Air / Colab free-T4; 1.5B on Kaggle; 3B on HF Spaces.")
    p.add_argument("--output-dir", default="./trained/grpo-run")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(4)),
                   help="Variant seeds to include in the training dataset.")
    p.add_argument("--group-size", type=int, default=4,
                   help="G in GRPO — completions generated per prompt.")
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Prompts per optimiser step (effective batch = batch * grad_accum).")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.04,
                   help="KL-to-reference coefficient in GRPO.")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--dry-run", action="store_true",
                   help="1 iter, group=2, 1 seed — smoke test only.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.dry_run:
        args.num_iters = 1
        args.group_size = 2
        args.batch_size = 2          # must be >= group_size so TRL can tile completions
        args.grad_accum = 1
        args.seeds = [0]
        args.max_new_tokens = 256

    # Lazy import: keeps the module loadable on CPU-only / no-torch environments
    # (e.g. the unit-test harness that exercises parse/replay in isolation).
    import torch  # noqa: WPS433
    from datasets import Dataset  # noqa: WPS433
    from peft import LoraConfig  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
    from trl import GRPOConfig, GRPOTrainer  # noqa: WPS433

    device = _pick_device(torch)
    print(f"[train_grpo] device={device}  model={args.model}  lora_r={args.lora_r}")
    print(f"[train_grpo] iters={args.num_iters}  group_size={args.group_size}  seeds={args.seeds}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Precision policy:
    #   CUDA  → bf16 (best on A10G/A100)
    #   MPS   → fp16 (bf16 unsupported on Apple Silicon through torch today)
    #   CPU   → fp32
    dtype = (
        torch.bfloat16 if device == "cuda"
        else torch.float16 if device == "mps"
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataset = build_training_dataset(tokenizer, args.seeds)
    print(f"[train_grpo] dataset rows={len(train_dataset)}  "
          f"(TRAIN_TASK_IDS × seeds = {len(TRAIN_TASK_IDS)}×{len(args.seeds)})")

    # TRL requires (per_device_batch * grad_accum * num_processes) % group_size == 0.
    # Round per_device_batch up to the next multiple of group_size so users don't
    # have to hand-tune these numbers.
    effective_batch = args.batch_size
    if effective_batch % args.group_size != 0:
        effective_batch = (
            (effective_batch // args.group_size) + 1
        ) * args.group_size
        print(f"[train_grpo] bumping per_device_train_batch_size "
              f"{args.batch_size} → {effective_batch} to satisfy group_size={args.group_size}")

    grpo_cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.group_size,
        max_completion_length=args.max_new_tokens,
        per_device_train_batch_size=effective_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,
        logging_steps=1,
        save_steps=args.save_every,
        max_steps=args.num_iters,
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        gradient_checkpointing=(device == "cuda"),  # MPS support is flaky
        report_to=os.environ.get("GRPO_REPORT_TO", "none"),
        remove_unused_columns=False,  # preserve task_id / seed columns for the reward fn
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func()],
        args=grpo_cfg,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_grpo] adapter saved → {args.output_dir}")


if __name__ == "__main__":
    main()
