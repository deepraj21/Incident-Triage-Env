# Incident Triage Environment — Design Spec

## Overview

An OpenEnv-compliant RL environment that simulates production incident triage. An AI agent receives a production alert and must diagnose the root cause by investigating a microservices architecture — querying logs, metrics, deploys, dependencies, and source code — then submit a diagnosis with remediation. Designed for the Meta Hackathon on OpenEnv.

**Name**: `incident-triage-env`

---

## Episode Structure

1. **`reset(task_id=...)`** — Agent receives an alert. Returns the initial observation with the alert and `result: None` (no action taken yet).
2. **`step(action)`** — Agent takes one investigative action per step. Environment returns relevant data + reward signal. Agent has a step budget (easy=10, medium=15, hard=20).
3. **`step(submit_diagnosis)`** — Agent submits root cause + remediation. Episode ends (`done=True`). Final reward scored by grader.
4. **Step budget exhaustion** — When `steps_remaining` hits 0 without a diagnosis, the environment auto-terminates (`done=True`). The agent receives only accumulated trajectory rewards (no terminal rewards, no efficiency bonus).

**Communication protocol**: All agent interaction uses **WebSocket** (`/ws`), not HTTP REST. The OpenEnv HTTP server creates a fresh environment instance per request and destroys it after — episode state cannot persist across HTTP calls. The WebSocket path maintains a persistent session with a single environment instance for the entire episode.

**Wire format (WebSocket)**:
```json
// Connect: ws://localhost:8000/ws

// Reset
{"type": "reset", "data": {"task_id": "easy_single_service_failure"}}
// → {"type": "observation", "data": {...}}

// Step
{"type": "step", "data": {"action_type": "query_logs", "service": "payments-service"}}
// → {"type": "observation", "data": {...}}

// Get state
{"type": "state"}
// → {"type": "state", "data": {...}}
```

The `reset()` method signature accepts `task_id` as a keyword argument: `def reset(self, seed=None, episode_id=None, task_id="easy_single_service_failure", **kwargs)`.

**Note**: The custom endpoints (`/tasks`, `/grader`, `/baseline`) remain HTTP REST since they are stateless lookups or trigger scripts, not part of the episode loop. The `/grader` endpoint requires an `episode_id` parameter and reads the completed episode's final state from a results store (the environment persists episode results keyed by `episode_id` before cleanup).

The step budget forces efficient investigation. An agent that queries everything exhaustively gets penalized vs. one that narrows down quickly.

---

## Action Space

A single polymorphic Pydantic model with an `action_type` discriminator (compatible with `create_app()`'s single `action_cls` parameter):

```python
class IncidentTriageAction(Action):
    action_type: Literal[
        "query_logs", "query_metrics", "check_deploys",
        "trace_dependencies", "check_status", "inspect_code",
        "submit_diagnosis"
    ]
    service: Optional[str] = None          # required for all except submit_diagnosis
    # query_logs options
    severity: Optional[Literal["error", "warn", "info", "debug"]] = None
    keyword: Optional[str] = None
    # query_metrics options
    metric: Optional[Literal["latency_p99", "error_rate", "cpu", "memory", "connections"]] = None
    # inspect_code options
    file_path: Optional[str] = None
    # submit_diagnosis fields
    root_cause_service: Optional[str] = None
    root_cause_category: Optional[RootCauseCategory] = None
    remediation: Optional[Remediation] = None
```

**Action type reference**:

| Action | Required Parameters | Returns |
|--------|-----------|---------|
| `query_logs` | `service`; optional: `severity`, `keyword` | Filtered log lines (last 30 min window) |
| `query_metrics` | `service`, `metric` | Time-series data points (text) |
| `check_deploys` | `service` | Recent deploys with timestamp, author, diff summary |
| `trace_dependencies` | `service` | Upstream/downstream services + connection health |
| `check_status` | `service` | Current health status, uptime, active alerts |
| `inspect_code` | `service`; optional: `file_path` | Source code snippets and recent code changes/diffs |
| `submit_diagnosis` | `root_cause_service`, `root_cause_category`, `remediation` | Episode ends, grader scores |

**`RootCauseCategory` enum**: `bad_deploy`, `resource_exhaustion`, `dependency_failure`, `config_change`, `traffic_spike`, `data_corruption`

**`Remediation` enum**: `rollback_deploy`, `scale_up`, `restart_service`, `fix_config`, `enable_rate_limiting`, `failover_to_backup`

Each action costs 1 step. No free actions.

---

## Observation Space

Concrete Pydantic model inheriting from OpenEnv's `Observation` base class:

```python
class AlertInfo(BaseModel):
    service: str
    message: str
    severity: Literal["critical", "warning"]
    timestamp: str

class ActionResult(BaseModel):
    action_type: str
    data: str  # formatted text (logs, metrics, code, etc.)

class IncidentTriageObservation(Observation):
    # Inherited from Observation: done (bool), reward (float|None), metadata (dict)
    alert: AlertInfo
    result: Optional[ActionResult] = None  # None on reset (no action yet)
    steps_remaining: int
    services_investigated: list[str] = []
```

**Design decisions**:
- `data` is always text (not structured JSON) — the agent must parse and reason over it, like a real engineer
- `services_investigated` gives the agent awareness of coverage
- `steps_remaining` creates urgency signal
- Alert repeated in every observation to prevent context loss
- `result` is `None` on the initial observation returned by `reset()`

---

## State

```python
class IncidentTriageState(State):
    # Inherited from State: episode_id (str), step_count (int)
    task_id: str
    steps_remaining: int
    services_investigated: list[str] = []
    actions_taken: list[dict] = []          # history of (action_type, service) pairs
    evidence_collected: dict[str, list[str]] = {}  # service -> [action_types queried]
```

---

## Reward Function

Two scoring contexts:
- **Per-step `reward`** in the observation: the incremental trajectory reward for that step (Phase 1 only)
- **Final grader score** (returned by `/grader`): Phase 2 terminal rewards only, scored 0.0–1.0

This separation means: per-step rewards guide RL training with dense signal, while the grader score evaluates final performance. They are independent.

### Phase 1: Trajectory Rewards (per-step, in observation.reward)

Each step's reward is the sum of components 1–3 below. Range per step: -0.13 to +0.17.

#### 1. Information Gain (per step, -0.05 to +0.12)

Each `(service, action_type)` pair is pre-labeled in the fixture with an evidence tier:

- +0.12 — query reveals **direct evidence** (the error log, the bad deploy)
- +0.08 — **causal chain evidence** (upstream metrics showing cascade)
- +0.04 — **contextual evidence** (relevant but not decisive, e.g., ruling out a healthy dependency)
- 0.0 — redundant query (same service + action_type already performed)
- -0.05 — **repeated redundant queries** (3rd+ total query to the same service, any action type)

#### 2. Investigation Strategy (per step, 0.0 to +0.05)

Requires tracking which services have shown anomalies. A service "has anomalies" if it is listed in the fixture's `anomalous_services` set (pre-labeled, not computed dynamically).

- +0.05 — **following dependency graph**: querying a service that is upstream/downstream of an anomalous service the agent has already investigated
- +0.03 — **cross-referencing**: checking a different action_type on a service where the agent already found anomalies
- +0.02 — **ruling out**: querying a non-anomalous service for the first time (legitimate elimination)
- 0.0 — random/unstructured jumping between unrelated services

#### 3. Red Herring Resistance (per step, -0.08 to 0.0)

Red herring services are listed in `ground_truth.red_herrings`. "Evidence points elsewhere" is defined as: the agent has already queried at least one service in the causal chain and received direct/causal-chain evidence.

- -0.08 — submitting diagnosis naming a red herring as `root_cause_service`
- -0.03 — 3rd+ step investigating a red herring service after the agent has already collected direct/causal-chain evidence from a non-red-herring service
- 0.0 — briefly checking a red herring and moving on (good debugging)

### Phase 2: Terminal Rewards (grader score, 0.0–1.0)

Computed only at episode end when `submit_diagnosis` is called. This is the score returned by `/grader`.

#### 4. Diagnosis Accuracy (0.0 to 0.55)

| Component | Full | Partial |
|-----------|------|---------|
| Correct `root_cause_service` | 0.20 | 0.10 if one hop away in causal chain |
| Correct `root_cause_category` | 0.15 | 0.07 if same failure family |
| Correct `remediation` | 0.10 | 0.05 if partially mitigating |
| Evidence quality | 0.10 | 0.10 if ≥80% of causal chain visited, 0.05 if ≥50%, 0.0 if <50% |

**Failure families** (for partial credit on `root_cause_category`):
- **Change-induced**: `bad_deploy`, `config_change`
- **Capacity-related**: `resource_exhaustion`, `traffic_spike`
- **Infrastructure**: `dependency_failure`, `data_corruption`

**Partial remediation mappings** (for 0.05 partial credit):
- `restart_service` partially mitigates `bad_deploy` (temporary fix)
- `scale_up` partially mitigates `traffic_spike` (buys time)
- `restart_service` partially mitigates `resource_exhaustion` (clears leaked memory)
- `failover_to_backup` partially mitigates `dependency_failure` (workaround)

#### 5. Efficiency (0.0 to 0.15)
```
efficiency = 0.15 * (steps_remaining / step_budget) ^ 0.5
```
Square root curve: solving in 3 vs 5 steps barely matters, but 5 vs 15 matters a lot.

### Penalties (applied to grader score)

| Penalty | Value | Trigger |
|---------|-------|---------|
| No diagnosis submitted | 0.0 grader score (only trajectory rewards were earned) | Ran out of steps |
| Shotgun diagnosis | -0.15 | Submitting within first 3 steps (insufficient investigation) |
| Circular investigation | -0.10 | Querying same service 4+ times total |
| Destructive remediation mismatch | -0.10 | Remediation contradicts the **ground truth** scenario data (e.g., `rollback_deploy` when the ground truth service has no recent deploy in the fixture) |

Final grader score: `clamp(diagnosis_accuracy + efficiency + penalties, 0.0, 1.0)`

---

## Task Definitions

### Task 1: Single Service Failure (Easy, 10 steps)

**Scenario**: `payments-service` had a bad deploy 20 minutes ago. Logs show `NullPointerException` in the new code path. Error rate jumped from 0.1% to 15% at deploy time. No other services affected.

- 3 services: `payments-service`, `database`, `api-gateway`
- Root cause is the alerted service itself
- Explicit error messages in logs
- No red herrings
- Expected path: ~4-5 steps
- Expected baseline score: 0.7–0.85

**Ground truth**:
```json
{
  "root_cause_service": "payments-service",
  "root_cause_category": "bad_deploy",
  "remediation": "rollback_deploy",
  "causal_chain": ["payments-service"],
  "red_herrings": [],
  "evidence_services": ["payments-service"],
  "anomalous_services": ["payments-service"],
  "failure_family": "change-induced"
}
```

### Task 2: Cascading Dependency Failure (Medium, 15 steps)

**Scenario**: `api-gateway` returns 500s. Logs show timeouts to `user-service`. `user-service` has connection pool exhaustion to `auth-db`. `auth-db` had a config change dropping max_connections from 200 to 20. Red herring: `cache-service` shows elevated latency (symptom, not cause).

- 5 services: `api-gateway`, `user-service`, `auth-db`, `cache-service`, `notification-service`
- Root cause is 2 hops from alert
- Requires correlating logs + metrics + config across services
- 1 red herring (`cache-service`)
- Expected path: ~8-10 steps
- Expected baseline score: 0.4–0.6

**Ground truth**:
```json
{
  "root_cause_service": "auth-db",
  "root_cause_category": "config_change",
  "remediation": "fix_config",
  "causal_chain": ["auth-db", "user-service", "api-gateway"],
  "red_herrings": ["cache-service"],
  "evidence_services": ["auth-db", "user-service", "api-gateway"],
  "anomalous_services": ["api-gateway", "user-service", "auth-db", "cache-service"],
  "failure_family": "change-induced"
}
```

### Task 3: Multi-Signal Cascading Failure with Red Herrings (Hard, 20 steps)

**Scenario**: `api-gateway` and `search-service` both alerting. `order-service` had a deploy 30 min ago (red herring — deploy is clean). Real cause: `inventory-service` memory leak in a code change 2 hours ago, manifests under load. `cache-service` evicts aggressively as `inventory-service` slows, cascading to `search-service` → `api-gateway`.

- 8 services: `api-gateway`, `search-service`, `inventory-service`, `cache-service`, `order-service`, `product-service`, `database`, `message-queue`
- Root cause is 3 hops from alert, time-delayed
- 2 red herrings (`order-service` recent deploy, `cache-service` eviction behavior)
- Requires `inspect_code` to find the memory leak
- Multiple services with real anomalies (symptoms vs. cause)
- Expected path: ~12-16 steps
- Expected baseline score: 0.15–0.35

**Ground truth**:
```json
{
  "root_cause_service": "inventory-service",
  "root_cause_category": "resource_exhaustion",
  "remediation": "restart_service",
  "causal_chain": ["inventory-service", "cache-service", "search-service", "api-gateway"],
  "red_herrings": ["order-service", "cache-service"],
  "evidence_services": ["inventory-service", "cache-service", "search-service", "api-gateway"],
  "anomalous_services": ["api-gateway", "search-service", "inventory-service", "cache-service", "order-service"],
  "failure_family": "capacity-related"
}
```

Note: `cache-service` appears in both `causal_chain` and `red_herrings`. It is a symptom-carrier (its behavior is caused by the root cause) but its own anomalies (aggressive eviction) are a red herring if the agent treats cache as the root cause.

**Red herring penalty exemption rule**: The per-step red herring penalty (-0.03 for 3+ steps) does NOT apply to services that are also in `causal_chain`. It only applies to services in `red_herrings` that are NOT in `causal_chain`. In Task 3, this means `order-service` triggers the per-step penalty but `cache-service` does not. However, submitting `cache-service` as `root_cause_service` still triggers the -0.08 diagnosis penalty since it is not the root cause.

---

## Scenario Data

Each scenario is a self-contained JSON fixture. No external databases or live systems.

**Fixture structure**:
```json
{
  "task_id": "easy_single_service_failure",
  "task_name": "Single Service Failure",
  "difficulty": "easy",
  "step_budget": 10,
  "alert": {
    "service": "payments-service",
    "message": "HTTP 500 rate spike — 15% of requests failing since 14:32 UTC",
    "severity": "critical",
    "timestamp": "2026-03-26T14:32:00Z"
  },
  "services": {
    "payments-service": {
      "dependencies": ["database"],
      "dependents": ["api-gateway"],
      "logs": [
        {"timestamp": "2026-03-26T14:32:15Z", "severity": "error", "message": "NullPointerException in PaymentProcessor.process()..."},
        ...
      ],
      "metrics": {
        "error_rate": [{"ts": "14:30", "val": 0.1}, {"ts": "14:32", "val": 15.2}, ...],
        "latency_p99": [...],
        "cpu": [...],
        "memory": [...]
      },
      "deploys": [
        {"timestamp": "2026-03-26T14:12:00Z", "author": "dev-alice", "diff_summary": "Refactored PaymentProcessor..."}
      ],
      "code_snippets": {
        "PaymentProcessor.java": "public void process(Payment p) {\n  var account = p.getAccount(); // can be null after refactor\n  account.validate(); // NPE here\n}",
        "recent_diff": "- var account = accountService.getAccount(p.getId());\n+ var account = p.getAccount();"
      },
      "status": {
        "health": "degraded",
        "uptime": "99.2%",
        "active_alerts": ["HTTP 500 spike"]
      }
    }
  },
  "evidence_labels": {
    "payments-service": {
      "query_logs": "direct_evidence",
      "query_metrics": "direct_evidence",
      "check_deploys": "direct_evidence",
      "trace_dependencies": "contextual",
      "check_status": "contextual",
      "inspect_code": "direct_evidence"
    },
    "database": {
      "query_logs": "contextual",
      "query_metrics": "contextual",
      "check_deploys": "contextual",
      "trace_dependencies": "contextual",
      "check_status": "contextual",
      "inspect_code": "contextual"
    }
  },
  "ground_truth": {
    "root_cause_service": "payments-service",
    "root_cause_category": "bad_deploy",
    "remediation": "rollback_deploy",
    "causal_chain": ["payments-service"],
    "red_herrings": [],
    "evidence_services": ["payments-service"],
    "anomalous_services": ["payments-service"],
    "failure_family": "change-induced"
  }
}
```

Key additions from review:
- `evidence_labels`: maps each `(service, action_type)` to an evidence tier for deterministic information gain scoring
- `anomalous_services`: pre-labeled set for investigation strategy reward computation
- `dependencies` + `dependents`: bidirectional dependency graph for graph-following reward

Pre-crafted (not procedurally generated) for deterministic grading.

---

## Rubric Integration

Two scoring components with different integration patterns:

**1. `InvestigationRubric(Rubric)`** — registered with the framework.
- Passed to `super().__init__(rubric=InvestigationRubric(...))` in the environment's `__init__`
- Called automatically by `self._apply_rubric(action, obs)` during `step()`
- Returns per-step trajectory reward (Phase 1) written to `observation.reward`
- Tracks trajectory state: actions taken, evidence collected, services queried
- `reset()` clears trajectory state

```python
class InvestigationRubric(Rubric):
    """Per-step trajectory rewards: information gain + strategy + red herring resistance"""
    def forward(self, action, observation) -> float: ...
    def reset(self) -> None: ...
```

**2. `DiagnosisScorer`** — a standalone scoring class, NOT a framework Rubric.
- Instantiated as a plain attribute on the environment: `self.diagnosis_scorer = DiagnosisScorer()`
- Called manually by the `/grader` endpoint handler after episode completion
- Accepts the full episode state (ground truth, agent's diagnosis, trajectory) and returns 0.0–1.0
- Stateless — does not need reset

```python
class DiagnosisScorer:
    """Terminal grader score: accuracy + efficiency + penalties. Not a framework Rubric."""
    def score(self, ground_truth: dict, diagnosis: dict, episode_state: IncidentTriageState) -> dict: ...
```

The environment stores completed episode results (final state + diagnosis) in a dict keyed by `episode_id` so the `/grader` endpoint can retrieve them after the WebSocket session closes.

---

## Custom Endpoints

### `GET /tasks`
Returns available tasks and their action schema.
```json
{
  "tasks": [
    {
      "task_id": "easy_single_service_failure",
      "task_name": "Single Service Failure",
      "difficulty": "easy",
      "step_budget": 10
    },
    ...
  ],
  "action_schema": { ... }  // JSON schema of IncidentTriageAction
}
```

### `POST /grader`
Called after an episode completes. Returns the terminal grader score for the completed episode. Requires `episode_id` to look up the stored episode result.
```json
// Request
POST /grader
{"episode_id": "abc-123"}

// Response:
{
  "task_id": "easy_single_service_failure",
  "score": 0.78,
  "breakdown": {
    "root_cause_service": 0.20,
    "root_cause_category": 0.15,
    "remediation": 0.10,
    "evidence_quality": 0.10,
    "efficiency": 0.08,
    "penalties": -0.0
  },
  "diagnosis_submitted": true
}
```

### `POST /baseline`
Triggers the baseline inference script and returns scores for all 3 tasks. Requires `OPENAI_API_KEY` env var.
```json
// Response:
{
  "scores": {
    "easy_single_service_failure": 0.82,
    "medium_cascading_dependency": 0.51,
    "hard_multi_signal_cascade": 0.23
  },
  "model": "gpt-4o",
  "total_steps_used": {"easy": 5, "medium": 9, "hard": 16}
}
```

---

## Architecture & File Structure

```
incident-triage-env/
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── README.md
├── models.py                  # Pydantic: IncidentTriageAction, IncidentTriageObservation, IncidentTriageState
├── client.py                  # EnvClient subclass
├── server/
│   ├── app.py                 # FastAPI app via create_app() + custom endpoints
│   ├── environment.py         # IncidentTriageEnv(Environment) - core logic
│   ├── grader.py              # InvestigationRubric + DiagnosisRubric
│   ├── scenario_loader.py     # Loads fixtures, filters data per query
│   └── scenarios/
│       ├── task_easy.json
│       ├── task_medium.json
│       └── task_hard.json
├── scripts/
│   └── baseline_inference.py  # OpenAI API client, runs all 3 tasks
└── tests/
    ├── test_environment.py
    ├── test_grader.py
    └── test_models.py
```

### openenv.yaml
```yaml
spec_version: 1
name: incident_triage_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

- `scenario_loader.py` — data layer. When agent calls `query_logs(service="payments", keyword="error")`, it filters the fixture's log lines and returns matches.
- `environment.py` — tracks episode state: steps taken, services investigated, evidence collected. Passes actions through scenario loader. Calls `super().__init__(rubric=InvestigationRubric(...))`.
- `grader.py` — implements `InvestigationRubric(Rubric)` (per-step, framework-integrated) and `DiagnosisScorer` (terminal, standalone class — NOT a framework Rubric).
- `app.py` — uses `create_app()` with `IncidentTriageAction`, `IncidentTriageObservation`, `IncidentTriageState` + adds custom `/baseline`, `/grader`, `/tasks` routes.

---

## Baseline Inference Script

Uses OpenAI API client (`OPENAI_API_KEY` from env vars). Connects to environment via WebSocket.

```
System prompt: "You are an oncall engineer diagnosing a production incident.
Investigate systematically by querying logs, metrics, deploys, dependencies,
service status, and code. Respond with a single JSON action object.
Available action_types: query_logs, query_metrics, check_deploys,
trace_dependencies, check_status, inspect_code, submit_diagnosis."

Loop per task:
  1. Connect ws://localhost:8000/ws
  2. Send {"type": "reset", "data": {"task_id": "..."}} → get initial observation
  3. Send observation to LLM as user message
  4. LLM responds with structured action (JSON)
  5. Send {"type": "step", "data": <action>} → get observation
  6. If done → POST /grader {"episode_id": "..."}, record score. Else → back to step 3.
  7. Close WebSocket, repeat for next task.

Output: scores for all 3 tasks.
```

---

## Deployment

- **Dockerfile**: Python 3.10 base, installs deps via pip, runs `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- **HF Space metadata** (in README.md frontmatter):
  ```yaml
  ---
  title: Incident Triage Environment
  emoji: 🚨
  colorFrom: red
  colorTo: orange
  sdk: docker
  pinned: false
  app_port: 8000
  tags:
    - openenv
  ---
  ```
- **Validation**: Must pass `openenv validate --url http://localhost:8000`

---

## Judging Alignment

| Criterion (Weight) | How we score |
|---------------------|-------------|
| Real-world utility (30%) | Incident triage is a daily task for every oncall engineer. Genuine training/eval value. |
| Task & grader quality (25%) | 3 tasks with clear difficulty progression, deterministic graders, 0.0–1.0 scores, hard task challenges frontier models. |
| Environment design (20%) | Clean reset, structured actions, 5-component reward with dense signal, step budgets for episode boundaries. |
| Code quality & spec (15%) | Full OpenEnv spec, typed Pydantic models, Rubric integration, Dockerfile, tests. |
| Creativity & novelty (10%) | Incident triage is a novel domain for OpenEnv. Red herring mechanics and investigation strategy rewards are unique. |
