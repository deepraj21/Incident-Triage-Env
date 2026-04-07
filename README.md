# Incident Triage RL Environment

An **OpenEnv-compliant reinforcement learning environment** for training LLM agents to diagnose production incidents across distributed microservices. Built for the OpenEnv Hackathon.

> **Live Demo:** [Hugging Face Space →](https://huggingface.co/spaces/AbhishekMallick/incident-triage-env)

---

## TL;DR

An RL-ready simulator where an agent receives a production alert and must investigate logs, metrics, deploys, and service dependencies to submit a graded root-cause diagnosis. Implements the OpenEnv `reset() / step() / state()` contract, ships as a Docker image, and exposes a Gradio UI for interactive evaluation.

**Stack:** Python 3.11 · FastAPI · Pydantic v2 · Gradio · Docker · OpenEnv SDK · HF Spaces

---

## Architecture

```
┌─────────────────┐     HTTP/JSON      ┌──────────────────────┐
│   Agent / LLM   │ ─────────────────▶ │  OpenEnv FastAPI     │
│  (policy π_θ)   │ ◀───────────────── │  Server (Docker)     │
└─────────────────┘   obs, reward, done└──────────┬───────────┘
        ▲                                         │
        │ GRPO / PPO                               ▼
        │ rollouts                      ┌──────────────────────┐
        │                               │  Env Core            │
        │                               │  ├─ Scenario Loader  │
        │                               │  ├─ Action Dispatcher│
        │                               │  ├─ Rubric (dense r) │
        │                               │  └─ Grader (terminal)│
        │                               └──────────┬───────────┘
        │                                          │
        │                               ┌──────────▼───────────┐
        │                               │  Fixture Store       │
        │                               │  logs / metrics /    │
        │                               │  deploys / deps /    │
        │                               │  code snapshots      │
        │                               └──────────────────────┘
```

## Episode Sequence

```
Agent                         Env Server                    Scenario Store
  │                                │                               │
  │  POST /reset {task_id}         │                               │
  │───────────────────────────────▶│  load_fixture(task_id)        │
  │                                │──────────────────────────────▶│
  │                                │◀──────────────────────────────│
  │  ◀─── Observation(alert) ──────│                               │
  │                                │                               │
  │  POST /step check_status       │                               │
  │───────────────────────────────▶│  dispatch + rubric.score()    │
  │  ◀── obs + r_t + remaining ────│                               │
  │                                │                               │
  │  POST /step trace_dependencies │                               │
  │───────────────────────────────▶│                               │
  │  ◀── obs + r_t ────────────────│                               │
  │                                │                               │
  │  POST /step query_logs(err)    │                               │
  │───────────────────────────────▶│                               │
  │  ◀── obs + r_t ────────────────│                               │
  │                                │                               │
  │  ... (investigate ≤ budget)    │                               │
  │                                │                               │
  │  POST /step submit_diagnosis   │                               │
  │───────────────────────────────▶│  grader.score(evidence,dx)    │
  │  ◀── final_score, done=True ───│                               │
```

## Reward Shaping

```
r_total  =  Σ r_step   +   R_terminal

r_step     = information_gain(action, state)
           + strategy_bonus(causal_chain_progress)
           - red_herring_penalty
           - redundancy_penalty

R_terminal = 0.40 · root_cause_service_correct
           + 0.20 · root_cause_category_correct
           + 0.15 · remediation_correct
           + 0.15 · evidence_quality
           + 0.10 · efficiency(steps_used / budget)
```

Dense per-step shaping prevents sparse-reward collapse during GRPO/PPO rollouts; terminal grading enforces diagnostic correctness.

---

## OpenEnv Contract

| OpenEnv Primitive | Implementation |
|---|---|
| `reset(task_id)` | Loads scenario fixture, emits initial `Alert` observation |
| `step(Action)` | Polymorphic Pydantic action → typed `Observation` + reward |
| `state()` | Steps remaining, services touched, evidence set |
| `Action` | Tagged union of 7 investigation primitives |
| `Observation` | `alert | logs | metrics | deploys | deps | code | grade` |
| `Rubric` | Per-step dense reward function |
| `Grader` | Terminal 0.0–1.0 scorecard with component breakdown |

## Action Space

| Action | Inputs | Returns |
|---|---|---|
| `check_status` | `service` | health, SLOs, active alerts |
| `query_logs` | `service`, `severity?`, `keyword?` | filtered log lines |
| `query_metrics` | `service`, `metric` | time-series |
| `check_deploys` | `service` | deploy history + diffs |
| `trace_dependencies` | `service` | upstream/downstream graph |
| `inspect_code` | `service`, `file_path?` | source + recent diffs |
| `submit_diagnosis` | `root_cause_service`, `category`, `remediation` | terminal grade |

## Scenarios

| Task | Budget | Services | Causal Depth |
|---|---|---|---|
| `easy_single_service_failure` | 10 | 3 | 1-hop (bad deploy) |
| `medium_cascading_dependency` | 15 | 5 | 3-hop (config → db → user-svc → gateway) |
| `hard_red_herring_chain` | 20 | 8 | 4-hop + decoys |

---


## Quickstart

```bash
# Local server
cd incident-triage-env
uv sync
uv run uvicorn server.app:app --port 8000

# Typed client rollout
python -c "
from client import IncidentTriageClient
env = IncidentTriageClient('http://localhost:8000')
obs = env.reset(task_id='medium_cascading_dependency')
obs = env.step({'type': 'check_status', 'service': 'api-gateway'})
print(obs.reward, obs.remaining_steps)
"

# Docker
docker build -t incident-triage-env .
docker run -p 8000:8000 incident-triage-env
```

## Training Integration

Plug directly into TRL GRPO rollouts — the env is a drop-in `OpenEnvClient` producing `(obs, action, reward, done)` tuples for on-policy updates. Compatible with LoRA fine-tuning on any HF-hosted base model (Qwen2.5, Llama-3, etc.).

---

**Built for the OpenEnv Hackathon** · [Hugging Face Space](https://huggingface.co/spaces/AbhishekMallick/incident-triage-env)
