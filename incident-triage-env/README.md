---
title: Incident Triage Environment
emoji: "\U0001F6A8"
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Incident Triage Environment

An OpenEnv-compliant RL environment that simulates production incident triage. An AI agent receives a production alert and must diagnose the root cause by investigating a microservices architecture — querying logs, metrics, deploys, dependencies, and source code — then submit a diagnosis with remediation.

## Why Incident Triage?

Every oncall engineer does this daily: an alert fires, you investigate across services, form hypotheses, and act. This environment captures that workflow as a structured RL task with:
- **Dense reward signals** from investigation quality (not just final accuracy)
- **Red herring resistance** — penalizes chasing misleading signals
- **Efficiency pressure** — step budgets force focused investigation
- **3 difficulty levels** from straightforward single-service failures to complex multi-hop cascades

## Action Space

| Action | Required Params | Returns |
|--------|----------------|---------|
| `query_logs` | `service`; optional: `severity`, `keyword` | Filtered log lines |
| `query_metrics` | `service`, `metric` | Time-series data |
| `check_deploys` | `service` | Recent deploy history |
| `trace_dependencies` | `service` | Upstream/downstream + health |
| `check_status` | `service` | Health, uptime, alerts |
| `inspect_code` | `service`; optional: `file_path` | Code snippets + diffs |
| `submit_diagnosis` | `root_cause_service`, `root_cause_category`, `remediation` | Episode ends, graded |

## Tasks

| Task | Difficulty | Steps | Services | Description |
|------|-----------|-------|----------|-------------|
| Single Service Failure | Easy | 10 | 3 | Bad deploy with NullPointerException |
| Cascading Dependency | Medium | 15 | 5 | Config change drops DB connections, cascades through chain |
| Multi-Signal Cascade | Hard | 20 | 8 | Memory leak + red herrings + 4-hop causal chain |

## Setup

```bash
# Install
pip install -e ".[dev,baseline]"

# Run server
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t incident-triage-env .
docker run -p 8000:8000 incident-triage-env
```

## Usage (WebSocket)

```python
import websockets, json

async with websockets.connect("ws://localhost:8000/ws") as ws:
    # Reset
    await ws.send(json.dumps({"type": "reset", "data": {"task_id": "easy_single_service_failure"}}))
    obs = json.loads(await ws.recv())

    # Step
    await ws.send(json.dumps({"type": "step", "data": {"action_type": "query_logs", "service": "payments-service"}}))
    obs = json.loads(await ws.recv())

    # Submit diagnosis
    await ws.send(json.dumps({"type": "step", "data": {
        "action_type": "submit_diagnosis",
        "root_cause_service": "payments-service",
        "root_cause_category": "bad_deploy",
        "remediation": "rollback_deploy"
    }}))
    obs = json.loads(await ws.recv())
```

## API Endpoints

- `GET /health` — Health check
- `GET /tasks` — List tasks with action schema
- `POST /grader` — Grade a completed episode: `{"episode_id": "..."}`
- `POST /baseline` — Run baseline inference (requires `GEMINI_API_KEY` or `OPENROUTER_API_KEY`)
- `WS /ws` — WebSocket for episodes

## Baseline

```bash
# Get a free API key from https://aistudio.google.com
GEMINI_API_KEY=your-key PYTHONPATH=. python scripts/baseline_inference.py
```

## Reward Function

**Per-step (trajectory):** Information gain (+0.12 direct, +0.08 causal, +0.04 contextual), investigation strategy (+0.05 following deps, +0.03 cross-ref), red herring resistance (-0.03 to -0.08).

**Terminal (grader):** Diagnosis accuracy (0.55 max), efficiency bonus (0.15 max), penalties for shotgun diagnosis, circular investigation, and remediation mismatch. Final score: 0.0–1.0.
