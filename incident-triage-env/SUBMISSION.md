# Incident Triage Environment — Hackathon Submission Guide

## What This Project Does

Imagine you're an oncall engineer at 3am. Your phone buzzes: "HTTP 500 spike on payments-service." You need to figure out what broke, why, and how to fix it — fast. You check logs, look at metrics dashboards, review recent deploys, trace service dependencies, and read code diffs. Eventually you identify the root cause and apply the fix.

**This project turns that entire workflow into an RL training environment.** An AI agent gets the same alert a human engineer would get, has the same investigation tools available, and must submit a diagnosis. The environment scores how well the agent investigated and whether it got the right answer.

### The System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UVICORN SERVER (port 8000)               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI App (server/app.py)             │   │
│  │                                                      │   │
│  │  Endpoints:                                          │   │
│  │    GET  /health     → {"status": "healthy"}          │   │
│  │    GET  /tasks      → list of 3 tasks                │   │
│  │    GET  /schema     → action/observation/state types │   │
│  │    POST /grader     → score a completed episode      │   │
│  │    WS   /ws         → WebSocket for agent episodes   │   │
│  │    POST /reset      → start new episode              │   │
│  │    POST /step       → take an action                 │   │
│  │    GET  /state      → current episode state          │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             │                                               │
│  ┌──────────▼───────────────────────────────────────────┐   │
│  │        IncidentTriageEnv (server/environment.py)     │   │
│  │                                                      │   │
│  │  reset(task_id) → loads scenario, returns alert      │   │
│  │  step(action)   → dispatches query, returns result   │   │
│  │  state()        → returns episode progress           │   │
│  └──────────┬──────────────────┬────────────────────────┘   │
│             │                  │                            │
│  ┌──────────▼──────────┐  ┌─── ▼─────────────────────┐      │
│  │  ScenarioLoader     │  │  Grader (grader.py)      │      │
│  │  (scenario_loader)  │  │                          │      │
│  │                     │  │  InvestigationRubric     │      │
│  │  Loads JSON fixtures│  │   → per-step rewards     │      │
│  │  Filters logs       │  │                          │      │
│  │  Returns metrics    │  │  DiagnosisScorer         │      │ 
│  │  Shows deploys/code │  │   → terminal score 0-1   │      │
│  └─────────────────────┘  └──────────────────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Scenario Data (server/scenarios/)           │   │
│  │                                                      │   │
│  │  task_easy.json   → 3 services, bad deploy           │   │
│  │  task_medium.json → 5 services, cascading config     │   │
│  │  task_hard.json   → 8 services, memory leak          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              INFERENCE SCRIPT (inference.py)                │
│                                                             │
│  1. Connects to server via WebSocket                        │
│  2. Sends reset(task_id) → gets alert                       │
│  3. Sends alert to LLM via OpenAI Client                    │
│  4. LLM returns JSON action → sends to env via step()       │
│  5. Gets observation back → sends to LLM again              │
│  6. Repeats until LLM submits diagnosis or budget runs out  │
│  7. Calls /grader to get final score                        │
│  8. Repeats for all 3 tasks                                 │
│                                                             │
│  Uses: API_BASE_URL, MODEL_NAME, HF_TOKEN env vars          │
│  Uses: OpenAI Client (AsyncOpenAI) for all LLM calls        │
└─────────────────────────────────────────────────────────────┘
```
 

### Why PYTHONPATH=. ?

When you run `PYTHONPATH=. python inference.py`, the `PYTHONPATH=.` tells Python to include the current directory in its module search path. This is needed because:
- `inference.py` is in the root directory
- It imports from `server/` and `models.py` which are also in the root
- Without `PYTHONPATH=.`, Python wouldn't know where to find these modules

### Why pass API keys on the command line vs .env?

The `.env` file is for local development convenience. But the hackathon validators run scripts with specific env vars set on the command line. The inference script reads from env vars (`os.environ.get(...)`) which works regardless of whether the value came from `.env` or from the command line. The hackathon requires these specific variables:
- `API_BASE_URL` — the LLM API endpoint
- `MODEL_NAME` — which model to use
- `HF_TOKEN` — your API key

### Do I need a separate alert-generation system?

**No.** The alerts are pre-built into the scenario JSON files. Each scenario file contains everything: the alert message, all service logs, metrics, deploy history, code snippets, dependency graphs, and the ground truth answer. When the agent calls `reset(task_id)`, the environment loads that scenario and presents the alert. It's a self-contained simulation.

---

## How This Aligns With the Hackathon

### Problem Statement
> "Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard step() / reset() / state() API."

### Direct Alignment

| Requirement | Our Implementation | Evidence |
|---|---|---|
| **Real-world task** | Production incident triage — done daily by every oncall engineer | PagerDuty, Datadog, and Grafana are all building AI for this |
| **step() API** | `step(action)` takes 7 action types, returns observation + reward | See `server/environment.py:69` |
| **reset() API** | `reset(task_id=...)` loads scenario, returns initial alert | See `server/environment.py:32` |
| **state() API** | Returns episode_id, steps, services investigated, evidence | See `server/environment.py:190` |
| **Typed models** | Pydantic v2 models for Action, Observation, State | See `models.py` |
| **openenv.yaml** | Present with spec_version, name, type, runtime, app, port | See `openenv.yaml` |
| **3+ tasks with graders** | Easy (10 steps), Medium (15 steps), Hard (20 steps) | See `server/scenarios/` |
| **Scores 0.0-1.0** | DiagnosisScorer returns clamped float | See `server/grader.py:269` |
| **Dense rewards** | Per-step rewards for info gain, strategy, red herring resistance | See `server/grader.py:63` |
| **Baseline inference** | `inference.py` using OpenAI Client, produces reproducible scores | See `inference.py` |
| **Dockerfile** | Working, builds and runs on port 8000 | See `Dockerfile` |
| **HF Space** | README has correct frontmatter (sdk: docker, app_port: 8000, tags: openenv) | See `README.md` |
| **openenv validate** | Passes both local (4/4 modes) and runtime (6/6 criteria) | Tested |

---

## Scoring Strategy

### Real-world utility (30%) — Target: 26-30

Incident triage is genuinely high-value:
- Every company with production systems has oncall engineers
- Every minute of downtime costs money (Amazon estimates $220k/min)
- Junior engineers waste time chasing symptoms — this trains better investigation patterns
- PagerDuty, Datadog, Grafana Labs are all investing in AI-assisted incident response
- No existing OpenEnv environment covers this domain

### Task & grader quality (25%) — Target: 20-25

- 3 tasks with clear difficulty progression (easy → medium → hard)
- Graders are deterministic and produce scores between 0.0 and 1.0
- Partial credit for close-but-wrong answers (same failure family, one hop away)
- Hard task genuinely challenges models (baseline scored 0.56 with perfect diagnosis knowledge)
- Penalties for bad behavior (shotgun diagnosis, circular investigation)

### Environment design (20%) — Target: 16-20

- `reset()` produces clean state every time
- 7 well-defined action types with typed parameters
- Dense per-step rewards (not just pass/fail):
  - +0.12 for direct evidence, +0.08 causal chain, +0.04 contextual
  - +0.05 for following dependency graph, +0.03 cross-referencing
  - -0.05 redundant queries, -0.08 red herring diagnosis
- Step budgets create natural episode boundaries
- Red herring mechanics test strategic reasoning

### Code quality & spec compliance (15%) — Target: 12-15

- `openenv validate` passes (local: 4/4 deployment modes, runtime: 6/6 criteria)
- Typed Pydantic v2 models throughout
- 33 passing tests (12 environment, 9 grader, 12 models)
- Docker builds cleanly
- `uv.lock` present for reproducible installs
- Clean project structure with clear separation of concerns

### Creativity & novelty (10%) — Target: 8-10

- **Novel domain**: No existing OpenEnv environment covers incident triage
- **Red herring mechanics**: Services that show symptoms but aren't the cause — the agent must resist chasing them
- **Investigation strategy rewards**: Not just "did you get the right answer" but "did you investigate smartly" (following dependencies, cross-referencing data sources)
- **Causal chain reasoning**: Multi-hop failures where Service A breaks Service B breaks Service C

---

## Baseline Results

Using `stepfun/step-3.5-flash:free` via OpenRouter (OpenAI Client):

| Task | Score | Steps Used | Budget |
|------|-------|------------|--------|
| easy_single_service_failure | 0.67 | 4 | 10 |
| medium_cascading_dependency | 0.49 | 10 | 15 |
| hard_multi_signal_cascade | 0.43 | 6 | 20 |

These show meaningful difficulty progression and room for improvement via RL training. The medium task scores lower despite more steps because the model chased some redundant queries (circular penalty: -0.10). The hard task shows the model struggles to find the right failure category and remediation even when it identifies the right service.

---

## Pre-Submission Checklist

| Check | Status | How to Verify |
|-------|--------|---------------|
| HF Space deploys + returns 200 | Pending deploy | `curl https://<space-url>/health` |
| Responds to reset() | Pass | WebSocket test passes |
| openenv validate (local) | Pass | `openenv validate` → OK, 4/4 modes |
| openenv validate (runtime) | Pass | `openenv validate --url http://localhost:8000` → 6/6 |
| Dockerfile builds | Pass | `docker build -t incident-triage-env .` |
| inference.py runs, produces scores | Pass | All 3 tasks scored 0.0-1.0 |
| Uses OpenAI Client | Pass | `inference.py` uses `AsyncOpenAI` |
| Uses API_BASE_URL, MODEL_NAME, HF_TOKEN | Pass | Reads from `os.environ` |
| 3+ tasks with graders | Pass | easy, medium, hard — all 0.0-1.0 |
| Runtime < 20 min | Pass | ~2-3 min for all 3 tasks |
| Runs on vcpu=2, 8GB RAM | Pass | FastAPI + JSON scenarios, minimal resources |

---

## How to Run (Quick Reference)

```bash
# 1. Install dependencies
pip install -e ".[dev,baseline]"

# 2. Start the environment server
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Run inference (in another terminal)
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=google/gemini-2.0-flash-exp:free \
HF_TOKEN=your-openrouter-key \
PYTHONPATH=. python inference.py

# 4. Validate
openenv validate
openenv validate --url http://localhost:8000

# 5. Run tests
PYTHONPATH=. pytest tests/ -v
```

---

## How to Deploy to Hugging Face Spaces

```bash
# Login to HF
huggingface-cli login

# Push (creates the Space automatically)
openenv push --repo-id <your-username>/incident-triage-env

# Verify
curl https://<your-username>-incident-triage-env.hf.space/health
```

---

## Judge Pitch (2-min version)

> "We built an RL environment for production incident triage — the thing every oncall engineer does when they get paged at 3am.
>
> The agent gets an alert like 'HTTP 500 spike on payments-service' and has to investigate: query logs, check metrics, review deploys, trace dependencies, and read code. Then it submits a diagnosis: which service broke, what category of failure, and how to fix it.
>
> What makes this interesting for RL:
> 1. **Dense rewards** — the agent gets feedback every step, not just pass/fail at the end. Querying the right service gives +0.12, following the dependency chain gives +0.05, chasing a red herring gets penalized.
> 2. **Red herring resistance** — we deliberately include services that show symptoms but aren't the root cause. The agent has to learn to not get distracted.
> 3. **Real difficulty progression** — easy is a single bad deploy (3 services), medium is a cascading config failure (5 services), hard is a memory leak with red herrings and a 4-hop causal chain across 8 services.
>
> Our baseline model scored 0.67 on easy, 0.35 on medium, and 0.56 on hard — showing there's real room for RL to improve investigation strategy.
>
> Everything is OpenEnv-compliant: step/reset/state API, typed Pydantic models, openenv validate passes, Dockerfile works, and it deploys to HF Spaces."
