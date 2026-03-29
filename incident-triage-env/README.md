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

## Demo

<!-- Replace with your video link after recording -->
> **Video walkthrough:** [Coming soon — screen recording of baseline agent solving the easy task]

## Why Incident Triage?

Every oncall engineer does this daily: an alert fires, you investigate across services, form hypotheses, and act. This environment captures that workflow as a structured RL task with:
- **Dense reward signals** from investigation quality (not just final accuracy)
- **Red herring resistance** — penalizes chasing misleading signals
- **Efficiency pressure** — step budgets force focused investigation
- **3 difficulty levels** from straightforward single-service failures to complex multi-hop cascades

## How It Works (OpenEnv Alignment)

This environment implements the standard OpenEnv `step()` / `reset()` / `state()` API:

```
Agent                          Environment
  |                                |
  |-- reset(task_id) ------------->|  Load scenario, return alert
  |<-- observation (alert) --------|
  |                                |
  |-- step(query_logs, svc) ------>|  Return filtered logs + reward
  |<-- observation (logs, reward) -|
  |                                |
  |-- step(check_deploys, svc) --->|  Return deploy history + reward
  |<-- observation (deploys, reward)|
  |                                |
  |-- step(submit_diagnosis) ----->|  Grade diagnosis, end episode
  |<-- observation (score, done) --|
  |                                |
  |-- state() ------------------->|  Return episode state
  |<-- state (steps, evidence) ---|
```

| OpenEnv Concept | Our Implementation |
|----------------|-------------------|
| `reset(task_id=...)` | Loads a scenario fixture, presents the initial alert |
| `step(action)` | Agent queries logs/metrics/deploys/code, receives data + reward |
| `state()` | Steps remaining, services investigated, evidence collected |
| `Observation` | Alert + action result (text data) + steps remaining |
| `Action` | Polymorphic model with 7 action types |
| `Rubric` | Per-step rewards: information gain, strategy, red herring resistance |
| `Grader` | Terminal score 0.0-1.0: diagnosis accuracy + evidence quality + efficiency |

No external alert generation system is needed — scenarios are self-contained simulations with all logs, metrics, deploys, code, and dependency data pre-built.

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

**Root cause categories:** `bad_deploy`, `resource_exhaustion`, `dependency_failure`, `config_change`, `traffic_spike`, `data_corruption`

**Remediations:** `rollback_deploy`, `scale_up`, `restart_service`, `fix_config`, `enable_rate_limiting`, `failover_to_backup`

## Tasks

| Task | Difficulty | Steps | Services | Scenario |
|------|-----------|-------|----------|----------|
| `easy_single_service_failure` | Easy | 10 | 3 | Bad deploy causes NullPointerException in payments-service |
| `medium_cascading_dependency` | Medium | 15 | 5 | Config change drops DB connections, cascades through service chain |
| `hard_multi_signal_cascade` | Hard | 20 | 8 | Memory leak + red herrings + 4-hop causal chain |

## Setup

```bash
# Install
pip install -e ".[dev,baseline]"

# Run server
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run tests
PYTHONPATH=. pytest tests/ -v

# Docker
docker build -t incident-triage-env .
docker run -p 8000:8000 incident-triage-env
```

## Usage (WebSocket)

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset with a task
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_id": "easy_single_service_failure"}
        }))
        obs = json.loads(await ws.recv())
        print(obs["data"]["observation"]["alert"])
        # -> {"service": "payments-service", "message": "HTTP 500 rate spike...", ...}

        # Investigate: query logs
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "query_logs", "service": "payments-service", "severity": "error"}
        }))
        obs = json.loads(await ws.recv())
        print(obs["data"]["observation"]["result"]["data"])
        # -> "[2026-03-26T14:32:01Z] [ERROR] NullPointerException in PaymentProcessor..."

        # Investigate: check deploys
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "check_deploys", "service": "payments-service"}
        }))
        obs = json.loads(await ws.recv())

        # Investigate: inspect code
        await ws.send(json.dumps({
            "type": "step",
            "data": {"action_type": "inspect_code", "service": "payments-service"}
        }))
        obs = json.loads(await ws.recv())

        # Submit diagnosis
        await ws.send(json.dumps({
            "type": "step",
            "data": {
                "action_type": "submit_diagnosis",
                "root_cause_service": "payments-service",
                "root_cause_category": "bad_deploy",
                "remediation": "rollback_deploy"
            }
        }))
        obs = json.loads(await ws.recv())
        print(obs["data"]["observation"]["result"]["data"])
        # -> "Diagnosis submitted. Grader score: 0.56"
        print(obs["data"]["done"])  # -> True

asyncio.run(main())
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Environment name and description |
| `/schema` | GET | Action, observation, and state JSON schemas |
| `/tasks` | GET | List tasks with action schema |
| `/grader` | POST | Grade a completed episode: `{"episode_id": "..."}` |
| `/baseline` | POST | Run baseline inference (requires `GEMINI_API_KEY` or `OPENROUTER_API_KEY`) |
| `/reset` | POST | Reset environment with task_id |
| `/step` | POST | Take an action |
| `/state` | GET | Get current episode state |
| `/ws` | WS | WebSocket for episodes |
| `/mcp` | POST | MCP JSON-RPC endpoint |

## Baseline

```bash
# Hackathon-compliant inference (uses OpenAI Client with required env vars)
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=google/gemini-2.0-flash-exp:free \
HF_TOKEN=your-api-key \
PYTHONPATH=. python inference.py

# Or with Gemini directly
GEMINI_API_KEY=your-key PYTHONPATH=. python scripts/baseline_inference.py
```

### Baseline Scores (`stepfun/step-3.5-flash:free` via OpenRouter)

| Task | Score | Steps | Budget |
|------|-------|-------|--------|
| easy_single_service_failure | 0.67 | 4 | 10 |
| medium_cascading_dependency | 0.49 | 10 | 15 |
| hard_multi_signal_cascade | 0.43 | 6 | 20 |

## Reward Function

### Per-step (trajectory rubric)

| Signal | Reward | Description |
|--------|--------|-------------|
| Direct evidence | +0.12 | Querying the right service with the right action |
| Causal chain evidence | +0.08 | Evidence along the causal chain |
| Contextual evidence | +0.04 | Useful but not conclusive |
| Following dependencies | +0.05 | Investigating upstream/downstream of anomalous services |
| Cross-referencing | +0.03 | Different action type on a service with known anomalies |
| Ruling out | +0.02 | Checking a non-anomalous service once |
| Redundant query (3+) | -0.05 | Querying the same service too many times |
| Red herring (3+ steps) | -0.03 | Continuing to investigate a red herring after finding causal evidence |
| Red herring diagnosis | -0.08 | Submitting a red herring as root cause |

### Terminal (grader)

| Component | Max Score | Description |
|-----------|-----------|-------------|
| Root cause service | 0.20 | Exact match; 0.10 if one hop away in causal chain |
| Root cause category | 0.15 | Exact match; 0.07 if same failure family |
| Remediation | 0.10 | Exact match; 0.05 for partial credit |
| Evidence quality | 0.10 | Proportion of causal chain services investigated |
| Efficiency | 0.15 | `0.15 * sqrt(steps_remaining / step_budget)` |
| **Penalties** | | |
| Shotgun diagnosis | -0.15 | Submitting within first 3 steps |
| Circular investigation | -0.10 | Any service queried 4+ times |
| Destructive mismatch | -0.10 | Rollback when no deploy exists |

**Final score: 0.0-1.0**

## Validation

```bash
# Local file validation
openenv validate

# Runtime validation against running server
openenv validate --url http://localhost:8000
```

## Deployment to Hugging Face Spaces

```bash
# Push to HF Spaces (requires huggingface-cli login)
openenv push --repo-id <your-username>/incident-triage-env
```

Once deployed, test against the Space:
```bash
GEMINI_API_KEY=your-key python scripts/baseline_inference.py --url https://<your-username>-incident-triage-env.hf.space
```

## Project Structure

```
incident-triage-env/
├── openenv.yaml              # OpenEnv spec declaration
├── pyproject.toml            # Dependencies and entry points
├── Dockerfile                # Docker deployment
├── README.md                 # This file (HF Space metadata in frontmatter)
├── inference.py              # Hackathon-compliant inference (OpenAI Client)
├── models.py                 # Pydantic models: Action, Observation, State
├── client.py                 # OpenEnv client wrapper
├── server/
│   ├── app.py                # FastAPI app via create_app()
│   ├── environment.py        # IncidentTriageEnv (reset/step/state)
│   ├── grader.py             # InvestigationRubric + DiagnosisScorer
│   ├── scenario_loader.py    # Loads and queries scenario JSON fixtures
│   ├── episode_store.py      # In-memory episode result storage
│   └── scenarios/
│       ├── task_easy.json    # 3 services, bad deploy
│       ├── task_medium.json  # 5 services, cascading config failure
│       └── task_hard.json    # 8 services, memory leak + red herrings
├── scripts/
│   └── baseline_inference.py # LLM baseline (Gemini/OpenRouter)
└── tests/
    ├── test_models.py        # 12 tests
    ├── test_grader.py        # 9 tests
    └── test_environment.py   # 12 tests
```
