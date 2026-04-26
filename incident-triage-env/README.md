---
title: Incident Triage Environment
emoji: "\U0001F6A8"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# 🚨 Incident Triage Environment

> **OpenEnv-compliant RL environment for production incident triage** — a multi-app
> enterprise simulator where an agent must diagnose live production incidents under
> a step budget, against a dynamic world that changes mid-episode and a 4-head
> composite grader that scores **process quality, not just final accuracy**.

[![Live HF Space](https://img.shields.io/badge/HF%20Spaces-LIVE-green?logo=huggingface)](https://huggingface.co/spaces/AbhishekMallick/incident-triage-env)
[![Trained Adapter](https://img.shields.io/badge/Model%20Hub-LoRA%20adapter-blue?logo=huggingface)](https://huggingface.co/AbhishekMallick/incident-triage-grpo-train)
[![Tests](https://img.shields.io/badge/tests-150%20passing-brightgreen)]()
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-orange)]()
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)]()

| Resource | Link |
|---|---|
| 🌐 **Live Environment (HF Space)** | https://huggingface.co/spaces/AbhishekMallick/incident-triage-env |
| 💻 **Source (GitHub)** | https://github.com/deepraj21/Incident-Triage-Env |
| 📓 **Training notebook (Colab)** | https://colab.research.google.com/drive/10dHOtRzLHY3aMSc21hxQouLxTi_gXv-t#scrollTo=train |
| 🤗 **Trained adapter — Qwen2.5-1.5B SFT** | https://huggingface.co/AbhishekMallick/incident-triage-grpo-train |
| 🤗 **Trained adapter — Qwen2.5-3B SFT** | https://huggingface.co/AbhishekMallick/incident-triage-grpo-train-Qwen3B |
| 🤗 **Trained adapter — Qwen2.5-7B SFT** | https://huggingface.co/AbhishekMallick/incident-triage-sft-train-Qwen2.5-7B |
| 📝 **Blog write-up** | [`Blog.md`](BLOG.md) / [Medium](https://medium.com/@mallickabhishek97/building-an-openenv-compliant-incident-triage-environment-for-rl-what-i-learned-along-the-way-3a39b0917862) |

---

## TL;DR

- 8 hand-crafted scenarios across 4 difficulty tiers + a process-hygiene tier (UAT bypass, CI quality breach)
- 6 enterprise apps (`alerthub` · `obsly` · `repohub` · `ticketdesk` · `chatops` · `uatsim`) wired into a unified action surface
- A `WorldClock` + `EventQueue` make the world evolve **during the episode** — new alerts, deploys, oncall handoffs and silent regressions can fire mid-investigation
- A declarative `PolicyEngine` enforces operational discipline (page-before-rollback, no-PR-during-freeze, etc.) with bonuses **and** penalties applied **per step**
- A **4-head composite grader** (diagnosis · policy · blast-radius · PR-proposal) with weight redistribution so partial submissions still score sensibly
- Trained Qwen-2.5-1.5B with TRL `SFTTrainer` + LoRA in 218 seconds on T4, then refined with GRPO — **+102 % composite score** on the held-out eval split
- 150 tests cover env contract, scoring math, dispatcher behaviour, policy rules, dynamic world, and training reward replay

---

## About this environment

Incident triage is what every oncall engineer does at 3 AM: an alert fires, you scramble across logs, metrics, deploy history, ticket queues, chat, and CI dashboards trying to figure out what is the actual root cause vs. a downstream symptom — under time pressure and operational rules that constrain what fixes you’re even allowed to ship right now (change-freeze windows, security gates, signoff requirements).

This environment captures that **end-to-end loop** as a trainable RL task. It is not a static QA benchmark. The world ticks forward every step. Events fire mid-episode. Some events are silent. The grader scores **how you investigated**, not only what you concluded.

### Per-step grading (not just terminal)

This is the single most important property of the env to internalise:

> **Every `step()` returns a reward.** Final score is the *aggregate*, not the only signal.

Three things happen on every action:

1. **`InvestigationRubric`** scores the action's *informational value* — direct evidence, causal-chain alignment, dependency-tracing strategy, redundancy/red-herring penalties.
2. **`PolicyEngine`** evaluates business rules against the action and the prior trajectory — fires positive deltas for compliant moves (pre-paging on-call, opening a PR for a config fix) and negative deltas for violations (forward-fix PRs during a change-freeze, bypassing UAT before a fast diagnosis).
3. The action's `result.data` is itself observable, so the agent's **next decision** is informed by the env's response to the *previous* one.

When the agent submits a diagnosis, the **`CompositeScorer`** runs a 4-head terminal grade on top of all the per-step shaping that already happened. The terminal score isn't the only learning signal — it's the cap.

### Theme alignment

Hackathon Theme **#3.1 World Modeling: Professional Tasks** + **Scaler AI Labs bonus track** (Multi-App RL Environment for Enterprise Workflows).

---

## Architecture

### System diagram

The full env stack — clients (top) → FastAPI/OpenEnv server → `IncidentTriageEnv` core (with `WorldClock`, `EventQueue`, `PolicyEngine`, `InvestigationRubric`, `CompositeScorer`, `EpisodeStore`) → six-app dispatcher → scenario JSON fixtures.

![System architecture](https://qmplv4qr76dicsvw.public.blob.vercel-storage.com/incident-triage/incident-triage.jpeg)

### Episode lifecycle

`reset → loop(step → advance clock → fire events → dispatch → reward) → submit_diagnosis → 4-head grade → store episode`.

![Episode lifecycle](https://qmplv4qr76dicsvw.public.blob.vercel-storage.com/incident-triage/episode-lifecycle.jpeg)

### App dispatcher → real enterprise tools

The env's action surface is **fixed**; the real-world targets are **pluggable**. An agent trained against this env's `APP_DISPATCH` generalises to any real combination of the tools below — only the I/O adapter changes.

![App dispatcher to real-world enterprise tools](https://qmplv4qr76dicsvw.public.blob.vercel-storage.com/incident-triage/env.jpeg)

---

## Action space

### Legacy (Round-1 compatible) — 7 verbs

| Action | Required | Returns |
|---|---|---|
| `check_status` | service | health · uptime · active_alerts |
| `trace_dependencies` | service | upstream + downstream + per-edge health |
| `query_logs` | service | filtered logs (severity, keyword) |
| `query_metrics` | service | metric time-series (latency_p99, error_rate, cpu, memory, connections) |
| `check_deploys` | service | deploy history with diffs |
| `inspect_code` | service | code snippets |
| `submit_diagnosis` | root_cause_service, category, remediation, *pr_proposal*, *blast_radius* | terminal score |

### Multi-app (Round-2) — `{app, op, args}` tagged union

| App | Ops |
|---|---|
| **alerthub** | `list_alerts`, `get_alert`, `ack_alert` |
| **obsly** | `query_logs`, `query_metric`, `get_trace`, `list_dashboards`, `check_status`, `trace_dependencies` |
| **repohub** | `recent_commits`, `get_diff`, `get_file`, `list_files`, `get_blame`, `open_pr`, **`ci_check`**, **`list_pr_history`** |
| **ticketdesk** | `search_tickets`, `get_ticket`, `create_incident`, `link_pr`, `add_comment` |
| **chatops** | `post_update`, `read_channel`, `page_oncall` |
| **uatsim** ⭐ Tier A | `list_stages`, `get_stage`, `get_signoff_status`, `check_uat_record` |
| **system** | `submit_diagnosis` |

`repohub.ci_check` returns a synthetic Snyk/Sonar/Raven/coverage report; `uatsim.check_uat_record` exposes pre-prod gate state per service. Both are scenario-driven and let policies enforce process-hygiene rules at the *operational* layer.

### Categories & remediations

```
RootCauseCategory  : bad_deploy | resource_exhaustion | dependency_failure
                     config_change | traffic_spike | data_corruption
Remediation        : rollback_deploy | scale_up | restart_service
                     fix_config | enable_rate_limiting | failover_to_backup
```

---

## Scenarios

Eight scenarios, three of which are **held-out** for honest before/after evaluation. The split is locked in `server/scenario_loader.py`.

| Task | Tier | Steps | Family | Split |
|---|---|---|---|---|
| `easy_single_service_failure` | Easy | 10 | bad_deploy NPE | train |
| `medium_cascading_dependency` | Medium | 15 | config_change cascade | train |
| `hard_multi_signal_cascade` | Hard | 20 | resource_exhaustion + red herrings | **eval** |
| `hard_region_failover` | Hard | 18 | config_change (failover_enabled=false) | train |
| `hard_freeze_violation` | Hard | 15 | bad_deploy during change-freeze | train |
| `expert_stealth_regression` | Expert | 20 | bad_deploy w/ silent metric drift | **eval** |
| `medium_uat_skipped` ⭐ Tier A | Medium | 14 | UAT-bypassed bad_deploy | train |
| `hard_pr_quality_breach` ⭐ Tier A | Hard | 18 | waived Snyk finding shipped past failed CI gate | **eval** |

Each scenario carries: `clock` (sim-time anchor), `tags` (`change_freeze`, `snyk_high_open`, `uat_bypassed`, …), full `services` graph, `timeline` (mid-episode events), declarative `policies`, multi-app state blocks (`alerthub`, `chatops`, `ticketdesk`, `obsly`, `repohub`, `uatsim`), and `ground_truth` including `correct_pr` + `correct_blast_radius` for the structured-output heads.

### Seed variants

```python
loader.get_scenario("easy_single_service_failure", seed=42)
```

Deterministic perturbations (red-herring log re-shuffling, timestamp jitter, deploy ordering, oncall lead rename) preserve ground-truth exactly while expanding the effective dataset. 5 train tasks × 8 seeds → 40 distinct training prompts.

---

## Reward design

### Step-level rubric

| Signal | Δ | When |
|---|---:|---|
| Direct evidence | +0.12 | first query of an action that hits ground-truth `direct_evidence` for the service |
| Causal-chain evidence | +0.08 | service is in `causal_chain` |
| Contextual evidence | +0.04 | useful but not the smoking gun |
| Dependency strategy | +0.05 | querying a neighbour of an already-anomalous service |
| Cross-reference | +0.03 | new action-type on a service with prior anomaly hits |
| Ruling out | +0.02 | first query of a non-anomalous service |
| Redundant query (≥3 to same svc) | −0.05 | per excess query |
| Red-herring fixation | −0.03 | 3+ steps on a red herring **after** finding causal evidence |
| Red-herring final answer | −0.08 | submit_diagnosis names a red herring |

### Policy engine (per-step + terminal)

Declarative rules per scenario in JSON:

```json
{
  "id": "page-before-config-fix",
  "when": { "app": "system", "op": "submit_diagnosis",
            "args_match": { "remediation": "fix_config" } },
  "require_prior": { "app": "chatops", "op": "page_oncall" },
  "require_prior_within_steps": 12,
  "penalty": -0.2,
  "bonus":   +0.1
}
```

Supported predicates: `args_match`, `require_prior`, `require_prior_within_steps`, `max_occurrences`, `forbidden_if(scenario_tag=…)`. The same engine produces both the per-step `policy_delta_this_step` (for shaping during RL training) and the `policy.compliance_score` head used in terminal grading.

### 4-head composite grader

```
composite = w_diag · diagnosis  +  w_pol · policy
          + w_blast · blast     +  w_pr · pr
```

Default weights `[0.40, 0.20, 0.20, 0.20]`. **Inapplicable heads redistribute** their weight proportionally to active heads, so an agent that submits without `pr_proposal` still scores fairly on diagnosis + policy + blast.

| Head | What it measures |
|---|---|
| **diagnosis** | root-cause-service + category + remediation match (with one-hop and same-family partial credit), evidence-coverage, efficiency, shotgun/circular/destructive penalties |
| **policy** | `1 - (|violations| · weight / total_attempts)` from `PolicyEngine.summary()` |
| **blast** | F1 on `affected_services` × 0.40, F1 on `missed_regions` × 0.20, log-tolerant magnitude on `estimated_requests_failed` × 0.20, IoU on `outage_window_*` × 0.20 |
| **pr** | exact `target_repo` match × 0.30, recall over `touched_files` × 0.35, keyword coverage in title/summary × 0.25, structural validity × 0.10 |

Score is clamped to `(0.001, 0.999)` to satisfy the OpenEnv evaluator's strict-bound rule.

---

## Dynamic world engine

### `WorldClock`

Deterministic step-based sim-time. Every `step()` advances by `step_seconds` (configurable per scenario). Observations always carry the current `sim_time`.

### `EventQueue`

Five event types fire from a per-scenario `timeline` array:

| Event | Effect |
|---|---|
| `new_log` | appends a log entry to a service mid-episode |
| `new_alert` | injects into `alerthub.alerts` (visible to `list_alerts`) |
| `new_deploy` | adds a fresh deploy row to a service |
| `oncall_handoff` | rotates `chatops.oncall_lead` |
| `slo_burn` | flags an SLO burn-rate spike on a service |

Events with `visible: false` mutate the world but are *not* surfaced on `obs.world_events` — the agent must **re-poll** to discover them. This is the "stealth regression" pattern (`expert_stealth_regression` scenario), which forces metric-driven investigation.

---

## Live UI

The serving Space mounts a Gradio app at the root URL with three panels:

- **Left**: scenario picker + two-tab action surface (`Investigate` legacy + `Multi-App` tagged union) with collapsible structured-submission JSON inputs
- **Right top**: live alert card (severity badge, sim-time, story line), world-events feed (cards per fired event), per-step policy chip (green/red Δ + violated rule names)
- **Right bottom**: investigation log table, last action result, and the **4-head score breakdown panel** with horizontal bars for each head + composite

This is the surface judges see when they hit `https://abhishekmallick-incident-triage-env.hf.space`.

![Oracle ceiling per scenario](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/oracle_ceiling.png?raw=true)

---

## Training pipeline

Two-stage SFT-warmstart → GRPO recipe on Qwen-2.5-1.5B-Instruct.

### Stage 1 — SFT (`scripts/train_sft.py`)

TRL `SFTTrainer` + LoRA on hand-authored oracle trajectories:

```
TRAIN_TASK_IDS  ×  seeds 0..7
       │
       ▼
oracle action sequence per (task, seed)
       │
       ▼
{prompt, completion} dataset (40 examples)
       │
       ▼
SFTTrainer (3 epochs, T4 small, ~3.6 min)
       │
       ▼
17.5 MB LoRA adapter  →  uploaded to HF Hub
```

Result: cross-entropy loss **2.66 → 1.81 (−32 %)**, mean token accuracy **60 % → 67 %** in 15 optimiser steps.

### Stage 2 — GRPO (`scripts/train_grpo.py`)

Single-turn GRPO formulation (TRL-native) with replay-and-grade reward:

```
prompt     = system + initial alert
completion = model-generated JSON-action trajectory
reward     = replay completion in fresh env copy → CompositeScorer.score → composite
```

This avoids multi-turn rollout complexity while still using the env's terminal reward as the only training signal. Runs in ~50 min on T4 for Qwen-1.5B with `group_size=4`.

### Training pipeline plot

![SFT → GRPO training pipeline](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/training_pipeline.png?raw=true)

| Stage | Reads | Writes | T4 wall-clock | Cost |
|---|---|---|---|---|
| Generate oracle trajectories | scenarios + ground_truth | in-memory dataset | < 1 s | – |
| SFT | dataset, base model | adapter + `reward_curve.csv` | 218 s | $0.04 |
| GRPO refine | adapter, env | refined adapter + curve | ~50 min | $0.30 |

---

## Results

### Held-out eval (3 scenarios × 5 seeds = 15 episodes per side)

| Task | Baseline | Trained (SFT+GRPO) | Δ |
|---|---:|---:|---:|
| `hard_multi_signal_cascade` | 0.400 | **0.747** | +0.347 |
| `expert_stealth_regression` | 0.391 | **0.800** | +0.410 |
| `hard_pr_quality_breach` | 0.336 | **0.726** | +0.390 |
| **OVERALL** | **0.375** | **0.758** | **+0.382 (+102 %)** |

### Per-head delta

| Head | Baseline | Trained | Δ |
|---|---:|---:|---:|
| diagnosis | 0.293 | 0.497 | +0.204 |
| policy | 0.983 | 0.983 | +0.000 |
| blast | 0.158 | 0.840 | +0.682 |
| pr | 0.149 | 0.971 | +0.822 |

The biggest lifts are on **structured outputs** (`pr` and `blast`) — exactly where SFT teaches format discipline. Diagnosis core improves moderately. Policy was already saturated at the baseline (the env's bonuses are easy to trigger once you understand the action surface).

### Plots

| | |
|---|---|
| ![Before/after](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/before_after_baseline-hf_vs_finetuned-sft.png?raw=true) | ![SFT loss](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/reward_curve_sft.png?raw=true) |
| ![GRPO reward](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/reward_curve_grpo.png?raw=true) | ![Oracle ceiling](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/oracle_ceiling.png?raw=true) |
| ![Radar — baseline](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/per_head_radar_baseline-hf.png?raw=true) | ![Radar — trained](https://github.com/deepraj21/Incident-Triage-Env/blob/feat/incident-triage-env/incident-triage-env/docs/benchmarks/per_head_radar_finetuned-sft.png?raw=true) |

---

## Trained models

The same `scripts/train_sft.py` runs against three Qwen sizes. All three adapters are public on the Hub.

| Base model | Trainable params | Adapter on disk | Inference VRAM (bf16) | HF Hub |
|---|---:|---:|---:|---|
| Qwen2.5-**1.5B**-Instruct | ~4.4 M | 17.5 MB | ~3.5 GB | [`incident-triage-grpo-train`](https://huggingface.co/AbhishekMallick/incident-triage-grpo-train) |
| Qwen2.5-**3B**-Instruct | ~8.4 M | ~33 MB | ~6.5 GB | [`incident-triage-grpo-train-Qwen3B`](https://huggingface.co/AbhishekMallick/incident-triage-grpo-train-Qwen3B) |
| Qwen2.5-**7B**-Instruct | ~20 M | ~80 MB | ~14 GB | [`incident-triage-sft-train-Qwen2.5-7B`](https://huggingface.co/AbhishekMallick/incident-triage-sft-train-Qwen2.5-7B) |

Common LoRA config: `r=16`, `alpha=32`, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`. Trainable parameters are well under 0.3 % of the base in every case.

**T4 wall-clock training**: 218 s (3 min 38 s) for the 1.5B variant — total HF compute cost ≈ $0.04. The 3B and 7B variants are A10G runs.

### Pull and use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pick any of the three adapters — all use the same prompt format.
ADAPTER = "AbhishekMallick/incident-triage-grpo-train"        # 1.5B
# ADAPTER = "AbhishekMallick/incident-triage-grpo-train-Qwen3B"      # 3B
# ADAPTER = "AbhishekMallick/incident-triage-sft-train-Qwen2.5-7B"   # 7B

BASE = ADAPTER.replace(  # the base model the adapter was trained on
    "AbhishekMallick/incident-triage-grpo-train",       "Qwen/Qwen2.5-1.5B-Instruct").replace(
    "incident-triage-grpo-train-Qwen3B",                "Qwen/Qwen2.5-3B-Instruct").replace(
    "incident-triage-sft-train-Qwen2.5-7B",             "Qwen/Qwen2.5-7B-Instruct")

base = AutoModelForCausalLM.from_pretrained(BASE)
tok  = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, ADAPTER)
```

A complete training-and-eval Colab notebook is published here: <https://colab.research.google.com/drive/10dHOtRzLHY3aMSc21hxQouLxTi_gXv-t#scrollTo=train>.

---

## Quick start

```bash
git clone https://github.com/deepraj21/Incident-Triage-Env
cd Incident-Triage-Env
pip install -e ".[dev,baseline]"

# 1) Run server (FastAPI + Gradio UI on :8000)
PYTHONPATH=. uvicorn server.app:app --port 8000

# 2) Try the live UI
open http://localhost:8000

# 3) Run tests (150 of them)
PYTHONPATH=. pytest tests/ -q
```

### WebSocket client example

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_id": "hard_pr_quality_breach"},
        }))
        obs = json.loads(await ws.recv())
        print("alert:", obs["data"]["observation"]["alert"])

        # Multi-app action — check CI gates before opening any PR
        await ws.send(json.dumps({
            "type": "step",
            "data": {"app": "repohub", "op": "ci_check",
                     "args": {"target_repo": "platform/notifications-service"}},
        }))
        print("ci_check:", json.loads(await ws.recv())["data"]["observation"]["result"]["data"])

        # Submit diagnosis with structured PR + blast radius
        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "action_type": "submit_diagnosis",
                "root_cause_service": "notifications-service",
                "root_cause_category": "bad_deploy",
                "remediation":         "rollback_deploy",
                "pr_proposal":   {"target_repo": "platform/notifications-service",
                                  "head_branch": "revert/v2.2",
                                  "title": "Rollback v2.2 SSRF vector",
                                  "summary": "Revert webhook validator change…",
                                  "diff_patch": "WebhookValidator.java …"},
                "blast_radius":  {"affected_services": ["notifications-service", "api-gateway"],
                                  "estimated_requests_failed": 200,
                                  "missed_regions": []},
            },
        }))
        terminal = json.loads(await ws.recv())
        print("score:", terminal["data"]["observation"]["grader_breakdown"]["score"])

asyncio.run(main())
```

### Inference driver

```bash
HF_TOKEN=hf_xxx \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct:novita" \
ENV_URL=http://localhost:8000 \
python inference.py --tasks eval --seed 0
```

Stdout protocol:

```
[START]      task=… env=… model=…
[STEP]       step=N action=… reward=… done=… error=null
[EVENT]      step=N event=… service=… payload={…}
[POLICY]     step=N delta=±0.20 violations=[…]
[BREAKDOWN]  diagnosis=…@w0.40 policy=…@w0.20 blast=…@w0.20 pr=…@w0.20
[END]        success=… steps=N score=… rewards=…
[EVAL]       train_avg=… eval_avg=… overall=…
```

---

## Reproduce the training

```bash
# SFT
python scripts/train_sft.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --lora-r 16 --lora-alpha 32 \
    --num-epochs 3 --batch-size 4 --grad-accum 2 \
    --seeds 0 1 2 3 4 5 6 7 \
    --output-dir ./trained/sft

# GRPO refinement (on top of the SFT adapter)
python scripts/train_grpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --lora-r 16 --num-iters 100 \
    --group-size 4 --batch-size 4 --grad-accum 2 \
    --seeds 0 1 2 3 \
    --output-dir ./trained/grpo

# Eval before/after
python scripts/eval_before_after.py --label baseline-hf  --seeds 0 1 2 3 4
python scripts/eval_before_after.py --label finetuned    --seeds 0 1 2 3 4
python scripts/eval_before_after.py --compare baseline-hf finetuned

# Render plots
python scripts/plot_model.py
```

Detailed walkthroughs: [`docs/TRAINING.md`](docs/TRAINING.md) (Colab T4, Kaggle P100, HF Spaces) · [`docs/HF_DEPLOY.md`](docs/HF_DEPLOY.md) (deploy as serving + training Spaces).

---

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/tasks` | GET | task list + JSON action schema |
| `/reset` | POST | open episode for a `task_id` (+ optional `seed`) |
| `/step` | POST | take an action |
| `/state` | GET | current state snapshot |
| `/grader` | POST | terminal score for an `episode_id` (heads + effective_weights) |
| `/ws` | WS | full streaming reset/step/state |
| `/mcp` | POST | MCP JSON-RPC bridge |
| `/` | GET | Gradio UI |

---

## Tests & validation

```bash
PYTHONPATH=. pytest tests/ -q
# 150 passed in 1.85s
```

Local smoke test against any URL (local or live HF Space):

```bash
python scripts/smoke_test_hf_space.py \
    --url https://abhishekmallick-incident-triage-env.hf.space --verbose
```

OpenEnv contract validation:

```bash
openenv validate                                    # static checks
openenv validate --url http://localhost:8000        # runtime contract
```

---

## Roadmap

- **Done**: Phase 1 (multi-app contract) → Phase 2 (dynamic world) → Phase 3 (policy engine) → Phase 4 (8 scenarios + variants) → Phase 5 (4-head grader) → Phase 6 (inference rewrite) → Phase 7 (SFT + GRPO training) → Phase 8 (before/after harness) → Phase 9 (HF Spaces deploy)
- **Near-term**: more scenarios (data-corruption, region failover variations, security incidents), Slack-bot integration, multi-episode "aftershock" campaigns
- **Long-term**: standard benchmark suite for incident-response agents, community-authored scenario packs with versioned scoring baselines

---

## Citation

```bibtex
@misc{incident-triage-env-2026,
  author       = {Abhishek Mallick},
  title        = {Incident Triage Environment: An OpenEnv-compliant RL environment for production incident response},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/spaces/AbhishekMallick/incident-triage-env}},
  note         = {Meta OpenEnv Hackathon Round 2}
}
```

---

## License

Apache-2.0.
