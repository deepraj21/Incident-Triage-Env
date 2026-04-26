"""SFT alternative to train_grpo.py — faster, stabler on T4.

Generates expert trajectories by running the oracle (hand-authored optimal path
for each scenario, see scripts/benchmark_env.py) and trains Qwen-1.5B with
TRL's SFTTrainer + LoRA to imitate them. The same prompt / action-line format
as train_grpo.py is used, so the resulting adapter is interchangeable for
inference.

Why SFT not GRPO for hackathon time pressure:
  - 5–15 sec/step on T4 vs 8–10 min/step for GRPO (≈50× faster)
  - No group-rollout memory — fits T4 with full batch comfortably
  - Bounded by oracle quality (0.78–0.85 composite ceiling), but that's
    plenty of headroom over a baseline that scores 0.15–0.30

Usage (HF Space env):
    MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
    LORA_R=16  LORA_ALPHA=32
    NUM_EPOCHS=3                       # vs NUM_ITERS for GRPO
    BATCH_SIZE=4  GRAD_ACCUM=2
    LEARNING_RATE=2e-4                 # SFT can use a higher LR than GRPO
    MAX_SEQ_LENGTH=1024
    SEEDS="0 1 2 3 4 5 6 7"            # 8 seeds × 5 train tasks = 40 examples
    HUB_REPO=AbhishekMallick/incident-triage-sft
    HUB_TOKEN=hf_xxx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IncidentTriageAction, RootCauseCategory, Remediation  # noqa: E402
from server.environment import IncidentTriageEnv  # noqa: E402
from server.scenario_loader import TRAIN_TASK_IDS, ScenarioLoader  # noqa: E402


# Same system prompt as train_grpo.py so SFT and GRPO produce interchangeable
# adapters. Keeping the wording identical is critical — at inference we use
# the same string.
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


def _legacy(action_type: str, service: str, **extra) -> IncidentTriageAction:
    return IncidentTriageAction(action_type=action_type, service=service, **extra)


def _submit(service: str, cat: str, rem: str) -> IncidentTriageAction:
    return IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service=service,
        root_cause_category=RootCauseCategory(cat),
        remediation=Remediation(rem),
    )


def oracle_for(task_id: str, loader: ScenarioLoader) -> list[IncidentTriageAction]:
    """Hand-authored optimal trajectory per scenario.

    Mirrors scripts/benchmark_env.py — short, follows ground-truth causal chain,
    includes mandatory gate-checks (page_oncall, ci_check, check_uat_record).
    """
    s = loader.get_scenario(task_id)
    gt = s["ground_truth"]
    svc = gt["root_cause_service"]
    cat = gt["root_cause_category"]
    rem = gt["remediation"]

    actions = [
        _legacy("check_status", s["alert"]["service"]),
        _legacy("trace_dependencies", s["alert"]["service"]),
        _legacy("query_logs", svc, severity="error"),
        _legacy("check_deploys", svc),
    ]

    # Scenario-specific gate evidence (matches benchmark_env oracle).
    if task_id == "hard_freeze_violation":
        actions.append(IncidentTriageAction(
            app="chatops", op="page_oncall", args={"role": "orders-lead"}))
    elif task_id == "hard_region_failover":
        actions.append(IncidentTriageAction(
            app="chatops", op="page_oncall", args={"role": "platform-lead"}))
    elif task_id == "medium_uat_skipped":
        actions.append(IncidentTriageAction(
            app="uatsim", op="check_uat_record", args={"service": svc}))
    elif task_id == "hard_pr_quality_breach":
        actions.append(IncidentTriageAction(
            app="repohub", op="ci_check",
            args={"target_repo": gt["correct_pr"]["target_repo"]}))

    actions.append(_submit(svc, cat, rem))
    return actions


def action_to_jsonl_line(a: IncidentTriageAction) -> str:
    """Serialise one action as a JSON line, matching the inference prompt format."""
    if a.app.value == "system" and a.op == "submit_diagnosis":
        return json.dumps({
            "action_type": "submit_diagnosis",
            "root_cause_service": a.root_cause_service,
            "root_cause_category": a.root_cause_category.value,
            "remediation": a.remediation.value,
        }, separators=(",", ":"))
    if a.action_type and a.action_type != "submit_diagnosis":
        # Legacy verb shape
        d: dict[str, Any] = {"action_type": a.action_type}
        if a.service:
            d["service"] = a.service
        for k, v in (a.args or {}).items():
            if k not in d and v not in (None, ""):
                d[k] = v
        return json.dumps(d, separators=(",", ":"))
    # Multi-app shape
    return json.dumps({"app": a.app.value, "op": a.op,
                        "args": dict(a.args or {})}, separators=(",", ":"))


def build_prompt(tokenizer, alert: dict, step_budget: int) -> str:
    user_msg = (
        f"INCIDENT ALERT [{str(alert.get('severity', '?')).upper()}] "
        f"{alert.get('service', '?')}: {alert.get('message', '?')}\n"
        f"Sim-time: {alert.get('timestamp', '?')}  •  Step budget: {step_budget}"
    )
    return tokenizer.apply_chat_template([
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ], tokenize=False, add_generation_prompt=True)


def build_dataset(tokenizer, seeds: list[int]):
    """Return a HF Dataset with a single 'text' field (prompt + completion + eos).

    TRL 0.15.x SFTTrainer expects either `dataset_text_field` or formatting_func.
    We pre-concatenate so the trainer needs no extra config.
    """
    from datasets import Dataset  # noqa: WPS433
    rows = []
    probe = IncidentTriageEnv()
    loader = probe._loader
    eos = tokenizer.eos_token or ""
    for task_id in TRAIN_TASK_IDS:
        for seed in seeds:
            obs = probe.reset(task_id=task_id, seed=seed,
                                episode_id=f"sft-{task_id}-{seed}")
            prompt = build_prompt(tokenizer, obs.alert.model_dump(),
                                    probe._step_budget)
            actions = oracle_for(task_id, loader)
            completion = "\n".join(action_to_jsonl_line(a) for a in actions)
            rows.append({"text": prompt + completion + eos})
    return Dataset.from_list(rows)


def _pick_device(torch_mod) -> str:
    if torch_mod.cuda.is_available():
        return "cuda"
    if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT trainer for incident_triage_env (TRL+LoRA)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output-dir", default="/tmp/sft-out")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    if args.dry_run:
        args.num_epochs = 1
        args.seeds = [0]

    import torch  # noqa: WPS433
    from peft import LoraConfig  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
    from trl import SFTConfig, SFTTrainer  # noqa: WPS433

    device = _pick_device(torch)
    print(f"[train_sft] device={device}  model={args.model}  lora_r={args.lora_r}")
    print(f"[train_sft] epochs={args.num_epochs}  seeds={args.seeds}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (torch.bfloat16 if device == "cuda"
             else torch.float16 if device == "mps" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataset = build_dataset(tokenizer, args.seeds)
    print(f"[train_sft] dataset rows={len(train_dataset)} "
          f"(TRAIN_TASK_IDS × seeds = {len(TRAIN_TASK_IDS)}×{len(args.seeds)})")

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=1,
        save_steps=max(20, len(train_dataset) // 2),
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        gradient_checkpointing=(device == "cuda"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=os.environ.get("SFT_REPORT_TO", "none"),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[train_sft] adapter saved → {args.output_dir}")

    # Reuse the same trainer_state.json → reward_curve.csv extraction so plot_results
    # can render an SFT loss curve too.
    state_path = Path(args.output_dir) / "trainer_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            curve_rows = []
            for entry in state.get("log_history", []):
                if entry.get("step") is None:
                    continue
                curve_rows.append({
                    "step": entry["step"],
                    "reward": entry.get("loss"),  # SFT: rename loss→"reward" col
                    "loss": entry.get("loss"),
                    "kl": 0.0,
                })
            if curve_rows:
                cp = Path(args.output_dir) / "reward_curve.csv"
                with cp.open("w") as f:
                    f.write("step,reward,loss,kl\n")
                    for r in curve_rows:
                        f.write(f"{r['step']},{r['reward']},{r['loss']},0.0\n")
                print(f"[train_sft] reward curve → {cp}  ({len(curve_rows)} steps)")
        except Exception as e:
            print(f"[train_sft] WARN — reward curve extract: {e}")

    # Hub push
    hub_repo = os.environ.get("HUB_REPO")
    hub_token = os.environ.get("HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hub_repo and hub_token:
        try:
            from huggingface_hub import HfApi  # noqa: WPS433
            api = HfApi(token=hub_token)
            api.create_repo(repo_id=hub_repo, repo_type="model",
                              private=False, exist_ok=True)
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=hub_repo, repo_type="model",
                commit_message=f"SFT adapter — {args.model} @ {args.num_epochs} epochs, "
                               f"seeds={args.seeds}",
            )
            print(f"[train_sft] adapter pushed to https://huggingface.co/{hub_repo}")
            print("[train_sft] DONE — safe to pause this Space now.")
        except Exception as e:
            print(f"[train_sft] ERROR — Hub push failed: {e}")
    else:
        print("[train_sft] no HUB_REPO/HUB_TOKEN — adapter remains at output_dir")


if __name__ == "__main__":
    main()
