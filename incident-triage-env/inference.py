"""Hackathon-compliant inference for the Incident Triage Environment (Round 2).

Wires the LLM to every Phase 1–5 surface:
  - Multi-app action tagged-union ({app, op, args}) + legacy shorthand
  - Sim-time advancement and mid-episode world_events surfaced each step
  - Per-step policy feedback (violations + delta) fed back into the prompt
  - Terminal multi-head grader breakdown (diagnosis / policy / blast / pr)
  - Train / eval task split + optional seed for variant rollouts

STDOUT FORMAT (hackathon-required):
    [START] task=<id> env=<name> model=<name>
    [STEP]  step=N action=... reward=... done=... error=null
    [EVENT] step=N event=new_alert service=... payload={...}    (world_events)
    [POLICY] step=N delta=... violations=[...]                  (policy feedback)
    [BREAKDOWN] diagnosis=... policy=... blast=... pr=...       (terminal)
    [END]   success=... steps=N score=... rewards=...
    [EVAL]  train_avg=... eval_avg=...                          (aggregate)

Environment variables (hackathon-required):
    API_BASE_URL   — OpenAI-compatible endpoint (default: openrouter)
    MODEL_NAME     — model identifier
    HF_TOKEN       — API key (HF Space uses this)
    ENV_URL        — environment server URL (default: http://localhost:8000)

CLI:
    python inference.py                    # all 6 scenarios
    python inference.py --tasks eval       # held-out eval split only
    python inference.py --tasks train      # train split only
    python inference.py --seed 42          # deterministic variant
    python inference.py --url http://...   # override ENV_URL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, List, Optional

import httpx
import websockets
from openai import AsyncOpenAI


# ------------------------------------------------------------------ config

API_BASE_URL = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen/qwen3.6-plus:free")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

BENCHMARK = "incident_triage_env"
MAX_RETRIES = 2
TEMPERATURE = 0.1

TRAIN_TASK_IDS = [
    "easy_single_service_failure",
    "medium_cascading_dependency",
    "hard_region_failover",
    "hard_freeze_violation",
    "medium_uat_skipped",
]
EVAL_TASK_IDS = [
    "hard_multi_signal_cascade",
    "expert_stealth_regression",
    "hard_pr_quality_breach",
]
ALL_TASK_IDS = TRAIN_TASK_IDS + EVAL_TASK_IDS


# ------------------------------------------------------------------ stdout logging

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_event(step: int, ev: dict) -> None:
    payload = json.dumps(ev.get("payload", {}), separators=(",", ":"))
    svc = ev.get("service") or "-"
    print(f"[EVENT] step={step} event={ev.get('event')} service={svc} payload={payload}", flush=True)


def log_policy(step: int, delta: float, violations: list[str]) -> None:
    if delta == 0.0 and not violations:
        return
    print(f"[POLICY] step={step} delta={delta:+.2f} violations={json.dumps(violations)}", flush=True)


def log_breakdown(heads: dict, effective_weights: dict) -> None:
    parts = []
    for h in ("diagnosis", "policy", "blast", "pr"):
        head = heads.get(h) or {}
        score = head.get("score")
        weight = effective_weights.get(h)
        if score is None or weight is None:
            continue
        parts.append(f"{h}={score:.2f}@w{weight:.2f}")
    if parts:
        print(f"[BREAKDOWN] {' '.join(parts)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ------------------------------------------------------------------ system prompt

SYSTEM_PROMPT = """You are an expert SRE / oncall engineer diagnosing a live production incident.
The environment simulates a multi-app enterprise surface and a world-clock that advances one tick per action. New events (alerts, deploys, oncall handoffs, fresh logs) can fire mid-incident — react to them.

## Action formats

You can return EITHER the legacy investigation shorthand OR the explicit multi-app form.

Legacy investigation (recommended for data queries):
  {"action_type": "check_status", "service": "payments-service"}
  {"action_type": "trace_dependencies", "service": "payments-service"}
  {"action_type": "query_logs", "service": "payments-service", "severity": "error"}
  {"action_type": "query_metrics", "service": "payments-service", "metric": "error_rate"}
  {"action_type": "check_deploys", "service": "payments-service"}
  {"action_type": "inspect_code", "service": "payments-service"}

Multi-app form (for ChatOps / RepoHub / AlertHub / TicketDesk / Obsly):
  {"app": "chatops",    "op": "page_oncall",   "args": {"role": "payments-lead"}}
  {"app": "chatops",    "op": "post_update",   "args": {"channel": "incidents", "text": "..."}}
  {"app": "alerthub",   "op": "list_alerts",   "args": {}}
  {"app": "alerthub",   "op": "ack_alert",     "args": {"alert_id": "PG-ALERT-1"}}
  {"app": "ticketdesk", "op": "open_ticket",   "args": {"title": "...", "body": "...", "severity": "sev1"}}
  {"app": "repohub",    "op": "open_pr",       "args": {"target_repo": "...", "head_branch": "...", "title": "..."}}
  {"app": "obsly",      "op": "get_trace",     "args": {"trace_id": "..."}}

## Terminal action — submit_diagnosis (STRUCTURED)

When you are confident, submit a FULL diagnosis. Every field you fill in can earn points:

{
  "action_type": "submit_diagnosis",
  "root_cause_service": "orders-service",
  "root_cause_category": "bad_deploy",
  "remediation": "rollback_deploy",
  "pr_proposal": {
    "target_repo": "commerce/orders-service",
    "head_branch": "revert/v8.2",
    "title": "Rollback v8.2 — discount_code validation breaks clients",
    "summary": "Reverts the over-strict discount_code regex added in v8.2 which threw IllegalArgumentException on valid legacy codes. Touches OrderService.java.",
    "diff_patch": "@@ OrderService.java @@ - throw new IllegalArgumentException ..."
  },
  "blast_radius": {
    "affected_services": ["orders-service", "api-gateway"],
    "estimated_requests_failed": 4000,
    "missed_regions": [],
    "outage_window_start": "2026-03-27T02:08:00Z",
    "outage_window_end":   "2026-03-27T02:30:00Z"
  }
}

The grader has FOUR heads — diagnosis / policy-compliance / blast-radius / pr-proposal.
Skipping pr_proposal or blast_radius simply redistributes their weight to the other heads, but if you can fill them in credibly you will score higher.

## Enumerations

root_cause_category: bad_deploy | resource_exhaustion | dependency_failure | config_change | traffic_spike | data_corruption
remediation:         rollback_deploy | scale_up | restart_service | fix_config | enable_rate_limiting | failover_to_backup

## Investigation strategy

1. START with the alerted service — check_status, then trace_dependencies.
2. FOLLOW CHAINS UPSTREAM. If service A times out calling B, the root cause is B (or something B depends on) — keep tracing until you find where the failure ORIGINATES.
3. GATHER evidence on the deepest upstream culprit: query_logs (severity=error), check_deploys, query_metrics, inspect_code.
4. READ world_events and policy feedback each turn — they may reveal new signals or warn you that an action is forbidden.
5. RESPECT policies. Some scenarios require paging oncall BEFORE submitting a rollback. Change-freeze scenarios forbid forward-fix PRs — use rollback only.
6. SUBMIT with pr_proposal + blast_radius filled in when you can.

## Matching guide

- NullPointerException / IllegalArgumentException after a deploy → bad_deploy → rollback_deploy
- failover_enabled=false / pool size / config reload → config_change → fix_config
- Memory leak / OOM / GC pressure → resource_exhaustion → restart_service
- Gradual p99 drift with NO error logs → bad_deploy (stealth regression) → rollback_deploy
- Regional 5xx spike with other regions healthy → config_change (failover disabled) → fix_config

## Output discipline

Respond with EXACTLY ONE JSON object. No prose, no markdown fences, no commentary."""


# ------------------------------------------------------------------ json parser

def parse_json_from_response(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ------------------------------------------------------------------ observation formatter

def format_observation(obs: dict, step: int) -> str:
    parts: list[str] = []
    alert = obs.get("alert") or {}
    sim_time = obs.get("sim_time")

    if step == 0:
        parts.append(
            f"INCIDENT ALERT [{str(alert.get('severity', '?')).upper()}] "
            f"on {alert.get('service', '?')}: {alert.get('message', '?')}"
        )
        parts.append(f"Alert timestamp: {alert.get('timestamp', '?')}")
        if sim_time:
            parts.append(f"Sim-time now: {sim_time}")
        parts.append("\nYou are the oncall engineer. Investigate systematically, following dependencies upstream.")
    else:
        parts.append(f"Alert: {alert.get('service', '?')}: {alert.get('message', '?')}")
        if sim_time:
            parts.append(f"Sim-time: {sim_time}")

    # Result of the action we just took
    result = obs.get("result")
    if result:
        parts.append(f"\n--- Result of {result.get('action_type', '?')} ---")
        parts.append(str(result.get("data", "")))

    # Mid-episode world events (Phase 2)
    world_events = obs.get("world_events") or []
    if world_events:
        parts.append("\n*** WORLD EVENTS since last step ***")
        for ev in world_events:
            svc = ev.get("service") or "-"
            parts.append(f"  • [{ev.get('event')}] service={svc} payload={json.dumps(ev.get('payload', {}))}")

    pending = obs.get("pending_world_events") or 0
    if pending:
        parts.append(f"(still-pending scheduled world events: {pending})")

    # Per-step policy feedback (Phase 3)
    violations = obs.get("policy_violations_this_step") or []
    delta = obs.get("policy_delta_this_step") or 0.0
    if violations or delta:
        marker = "VIOLATION" if delta < 0 else "BONUS"
        parts.append(f"\n[POLICY {marker}] delta={delta:+.2f} rules={violations}")

    # Budget pressure
    steps_remaining = obs.get("steps_remaining")
    parts.append(f"\nSteps remaining: {steps_remaining}")
    investigated = obs.get("services_investigated") or []
    if investigated:
        parts.append(f"Services investigated: {', '.join(investigated)}")

    if isinstance(steps_remaining, int):
        if steps_remaining <= 2:
            parts.append("\n*** URGENT: submit_diagnosis NOW — budget almost gone. ***")
        elif steps_remaining <= 5:
            parts.append("\n** Warning: budget running low. Prepare to submit soon. **")

    return "\n".join(parts)


# ------------------------------------------------------------------ single run

async def run_task(
    client: AsyncOpenAI,
    base_url: str,
    task_id: str,
    seed: Optional[int] = None,
) -> dict:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with websockets.connect(ws_url) as ws:
            reset_args: dict[str, Any] = {"task_id": task_id}
            if seed is not None:
                reset_args["seed"] = seed
            await ws.send(json.dumps({"type": "reset", "data": reset_args}))
            response = json.loads(await ws.recv())
            data = response.get("data", response)
            obs = data.get("observation", data)
            done = data.get("done", obs.get("done", False))

            await ws.send(json.dumps({"type": "state"}))
            state_resp = json.loads(await ws.recv())
            state_data = state_resp.get("data", state_resp)
            episode_id = state_data.get("episode_id", "unknown")

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            final_grader_breakdown: dict = {}
            step = 0

            while not done:
                messages.append({"role": "user", "content": format_observation(obs, step)})

                action: Optional[dict] = None
                last_error: Optional[str] = None
                for retry in range(MAX_RETRIES + 1):
                    try:
                        completion = await client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            response_format={"type": "json_object"},
                            temperature=TEMPERATURE,
                        )
                        raw = completion.choices[0].message.content
                        action = parse_json_from_response(raw)
                        break
                    except Exception as e:
                        last_error = str(e)
                        if retry < MAX_RETRIES:
                            messages.append({
                                "role": "user",
                                "content": f"Your last response was not valid JSON: {e}. Return ONLY a JSON object.",
                            })

                if action is None:
                    log_end(success=False, steps=steps_taken, score=0.001, rewards=rewards)
                    return {"task_id": task_id, "score": 0.001, "steps": steps_taken,
                            "error": last_error, "breakdown": {}, "heads": {}}

                for junk in ("metadata", "reasoning", "explanation", "thought"):
                    action.pop(junk, None)

                await ws.send(json.dumps({"type": "step", "data": action}))
                response = json.loads(await ws.recv())
                data = response.get("data", response)
                obs = data.get("observation", data)
                done = data.get("done", obs.get("done", False))
                step += 1
                steps_taken = step

                reward = data.get("reward", obs.get("reward", 0.0)) or 0.0
                rewards.append(float(reward))

                messages.append({"role": "assistant", "content": json.dumps(action)})

                svc = action.get("service") or action.get("root_cause_service") or \
                      (action.get("args", {}) or {}).get("service", "")
                action_str = f"{action.get('action_type') or action.get('op', '?')}({svc})"
                log_step(step=step, action=action_str, reward=float(reward), done=done, error=None)

                for ev in obs.get("world_events") or []:
                    log_event(step, ev)
                log_policy(
                    step,
                    float(obs.get("policy_delta_this_step") or 0.0),
                    obs.get("policy_violations_this_step") or [],
                )

                if done and obs.get("grader_breakdown"):
                    final_grader_breakdown = obs["grader_breakdown"]

        # Terminal grader fetch
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                base_url.rstrip("/") + "/grader",
                json={"episode_id": episode_id},
                timeout=30.0,
            )
            if resp.status_code == 200:
                gd = resp.json()
                score = max(0.001, min(0.999, float(gd.get("score", 0.001))))
                heads = gd.get("heads", {}) or final_grader_breakdown.get("heads", {})
                eff_w = gd.get("effective_weights", {}) or final_grader_breakdown.get("effective_weights", {})
                success = score >= 0.3
                log_breakdown(heads, eff_w)
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
                return {
                    "task_id": task_id,
                    "score": score,
                    "steps": steps_taken,
                    "breakdown": gd.get("breakdown", {}),
                    "heads": heads,
                    "effective_weights": eff_w,
                }
            log_end(success=False, steps=steps_taken, score=0.001, rewards=rewards)
            return {"task_id": task_id, "score": 0.001, "steps": steps_taken,
                    "error": f"Grader returned {resp.status_code}", "breakdown": {}, "heads": {}}

    except Exception as e:
        log_end(success=False, steps=steps_taken, score=0.001, rewards=rewards)
        return {"task_id": task_id, "score": 0.001, "steps": steps_taken,
                "error": str(e), "breakdown": {}, "heads": {}}


# ------------------------------------------------------------------ orchestrator

def _pick_api_key() -> str:
    if "openrouter" in API_BASE_URL.lower():
        return os.environ.get("OPENROUTER_API_KEY") or HF_TOKEN
    if "huggingface" in API_BASE_URL.lower():
        return HF_TOKEN or os.environ.get("API_KEY", "")
    return HF_TOKEN or os.environ.get("OPENROUTER_API_KEY") or \
           os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY", "")


def _pick_tasks(which: str) -> list[str]:
    if which == "train":
        return list(TRAIN_TASK_IDS)
    if which == "eval":
        return list(EVAL_TASK_IDS)
    return list(ALL_TASK_IDS)


async def main_async(args: argparse.Namespace) -> dict:
    api_key = _pick_api_key()
    if not api_key:
        print("ERROR: set HF_TOKEN (or OPENROUTER_API_KEY / API_KEY).", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=api_key)
    base_url = args.url or ENV_URL
    task_ids = _pick_tasks(args.tasks)

    print(f"Model:        {MODEL_NAME}")
    print(f"API base:     {API_BASE_URL}")
    print(f"Env URL:      {base_url}")
    print(f"Tasks ({args.tasks}): {task_ids}")
    if args.seed is not None:
        print(f"Seed:         {args.seed}")

    results: dict[str, dict] = {}
    for task_id in task_ids:
        print(f"\n{'=' * 60}\nTask: {task_id}\n{'=' * 60}")
        results[task_id] = await run_task(client, base_url, task_id, seed=args.seed)

    # Aggregate + eval split summary
    print(f"\n{'=' * 60}\nRESULTS\n{'=' * 60}")
    train_scores = [results[t]["score"] for t in task_ids if t in TRAIN_TASK_IDS]
    eval_scores = [results[t]["score"] for t in task_ids if t in EVAL_TASK_IDS]
    for tid in task_ids:
        r = results[tid]
        tag = "eval" if tid in EVAL_TASK_IDS else "train"
        print(f"  [{tag}] {tid}: {r['score']:.3f} (steps={r['steps']})")

    train_avg = sum(train_scores) / len(train_scores) if train_scores else 0.0
    eval_avg = sum(eval_scores) / len(eval_scores) if eval_scores else 0.0
    all_avg = sum(r["score"] for r in results.values()) / len(results) if results else 0.0
    print(f"\n  train_avg={train_avg:.3f}  eval_avg={eval_avg:.3f}  overall={all_avg:.3f}")
    print(f"[EVAL] train_avg={train_avg:.3f} eval_avg={eval_avg:.3f} overall={all_avg:.3f}", flush=True)

    return {
        "model": MODEL_NAME,
        "seed": args.seed,
        "tasks_run": args.tasks,
        "results": results,
        "train_avg": train_avg,
        "eval_avg": eval_avg,
        "overall": all_avg,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 6 inference driver for incident_triage_env")
    p.add_argument("--tasks", choices=["all", "train", "eval"], default="all")
    p.add_argument("--seed", type=int, default=None, help="deterministic variant seed")
    p.add_argument("--url", type=str, default=None, help="override ENV_URL")
    p.add_argument("--out", type=str, default=None, help="write results JSON to path")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    out = asyncio.run(main_async(args))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote results to {args.out}")
