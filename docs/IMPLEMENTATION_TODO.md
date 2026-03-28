# Incident Triage Environment — Implementation TODO

> **Purpose**: Feed this file to a Claude Code session to implement the environment end-to-end.
> **Design Spec**: `docs/superpowers/specs/2026-03-26-incident-triage-env-design.md`
> **Business Context**: `docs/WHAT_AND_WHY.md`
> **OpenEnv Framework**: `../OpenEnv/` (reference implementation — read but do not modify)

---

## Prerequisites

- [ ] Read the design spec thoroughly before starting
- [ ] Read the OpenEnv framework code to understand base classes:
  - `../OpenEnv/src/openenv/core/env_server/types.py` — Action, Observation, State base models
  - `../OpenEnv/src/openenv/core/env_server/interfaces.py` — Environment base class
  - `../OpenEnv/src/openenv/core/env_server/http_server.py` — create_app(), WebSocket handling
  - `../OpenEnv/src/openenv/core/rubrics/base.py` — Rubric base class
  - `../OpenEnv/src/openenv/core/rubrics/trajectory.py` — TrajectoryRubric pattern
  - `../OpenEnv/envs/coding_env/` — reference example environment
- [ ] Read the OpenEnv course reference if needed: `https://github.com/raun/openenv-course`

---

## Phase 1: Project Scaffolding

### 1.1 Initialize project structure
- [ ] Create directory structure under `OpenEnv-Hackathon/`:
  ```
  incident-triage-env/
  ├── openenv.yaml
  ├── pyproject.toml
  ├── Dockerfile
  ├── README.md
  ├── models.py
  ├── client.py
  ├── server/
  │   ├── __init__.py
  │   ├── app.py
  │   ├── environment.py
  │   ├── grader.py
  │   ├── scenario_loader.py
  │   └── scenarios/
  │       ├── task_easy.json
  │       ├── task_medium.json
  │       └── task_hard.json
  ├── scripts/
  │   └── baseline_inference.py
  └── tests/
      ├── __init__.py
      ├── test_environment.py
      ├── test_grader.py
      └── test_models.py
  ```

### 1.2 Create openenv.yaml
- [ ] Write `openenv.yaml`:
  ```yaml
  spec_version: 1
  name: incident_triage_env
  type: space
  runtime: fastapi
  app: server.app:app
  port: 8000
  ```

### 1.3 Create pyproject.toml
- [ ] Define project with dependencies:
  - `openenv` (install from `../OpenEnv/` or PyPI)
  - `fastapi`
  - `uvicorn`
  - `pydantic>=2.0`
  - `websockets`
  - `openai` (for baseline script)
  - `pytest` (dev dependency)

---

## Phase 2: Pydantic Models (`models.py`)

### 2.1 Enums
- [ ] `RootCauseCategory` enum: `bad_deploy`, `resource_exhaustion`, `dependency_failure`, `config_change`, `traffic_spike`, `data_corruption`
- [ ] `Remediation` enum: `rollback_deploy`, `scale_up`, `restart_service`, `fix_config`, `enable_rate_limiting`, `failover_to_backup`
- [ ] `ActionType` literal: `query_logs`, `query_metrics`, `check_deploys`, `trace_dependencies`, `check_status`, `inspect_code`, `submit_diagnosis`
- [ ] `EvidenceTier` enum: `direct_evidence`, `causal_chain`, `contextual`, `none`

### 2.2 Action model
- [ ] `IncidentTriageAction(Action)` — single polymorphic model with `action_type` discriminator
  - `action_type`: Literal of all 7 types
  - `service`: Optional[str] (required for all except submit_diagnosis)
  - `severity`: Optional[Literal["error", "warn", "info", "debug"]]
  - `keyword`: Optional[str]
  - `metric`: Optional[Literal["latency_p99", "error_rate", "cpu", "memory", "connections"]]
  - `file_path`: Optional[str]
  - `root_cause_service`: Optional[str]
  - `root_cause_category`: Optional[RootCauseCategory]
  - `remediation`: Optional[Remediation]
- [ ] Add a Pydantic `model_validator` to enforce: if `action_type != "submit_diagnosis"`, then `service` is required. If `action_type == "submit_diagnosis"`, then `root_cause_service`, `root_cause_category`, `remediation` are required.

### 2.3 Observation model
- [ ] `AlertInfo(BaseModel)`: service, message, severity (Literal["critical","warning"]), timestamp
- [ ] `ActionResult(BaseModel)`: action_type (str), data (str)
- [ ] `IncidentTriageObservation(Observation)`: inherits done, reward, metadata from base. Adds: alert (AlertInfo), result (Optional[ActionResult]), steps_remaining (int), services_investigated (list[str])

### 2.4 State model
- [ ] `IncidentTriageState(State)`: inherits episode_id, step_count from base. Adds: task_id (str), steps_remaining (int), services_investigated (list[str]), actions_taken (list[dict]), evidence_collected (dict[str, list[str]])

### 2.5 Validation
- [ ] Write `tests/test_models.py` — test model creation, validation, serialization
- [ ] Test that invalid actions (e.g., query_logs without service) raise validation errors
- [ ] Run tests: `pytest tests/test_models.py`

---

## Phase 3: Scenario Data (JSON Fixtures)

### 3.1 Task Easy: `server/scenarios/task_easy.json`
- [ ] Create fixture for "Single Service Failure"
- [ ] 3 services: `payments-service`, `database`, `api-gateway`
- [ ] `payments-service` logs: 15-20 log lines including NullPointerException errors after deploy time (14:32), normal logs before
- [ ] `payments-service` metrics: error_rate spikes at 14:32, latency_p99 increases, cpu/memory normal
- [ ] `payments-service` deploys: one deploy at 14:12 by dev-alice, diff shows PaymentProcessor refactor
- [ ] `payments-service` code_snippets: the buggy code + the diff showing the null-introducing change
- [ ] `payments-service` status: health=degraded, active_alerts=["HTTP 500 spike"]
- [ ] `database` and `api-gateway`: normal logs, normal metrics, no recent deploys, healthy status
- [ ] `evidence_labels`: payments-service query_logs/metrics/deploys/inspect_code = direct_evidence, others = contextual. database and api-gateway = all contextual
- [ ] `ground_truth`: root_cause_service=payments-service, category=bad_deploy, remediation=rollback_deploy, causal_chain=[payments-service], red_herrings=[], anomalous_services=[payments-service], failure_family=change-induced

### 3.2 Task Medium: `server/scenarios/task_medium.json`
- [ ] Create fixture for "Cascading Dependency Failure"
- [ ] 5 services: `api-gateway`, `user-service`, `auth-db`, `cache-service`, `notification-service`
- [ ] Dependency graph: api-gateway → user-service → auth-db; api-gateway → cache-service; user-service → notification-service
- [ ] `api-gateway` logs: HTTP 500s, timeouts calling user-service
- [ ] `user-service` logs: connection pool exhaustion, timeouts to auth-db
- [ ] `auth-db` logs: connection limit reached, config reload event; metrics show connections maxed at 20 (was 200)
- [ ] `cache-service` logs/metrics: elevated latency (red herring — caused by cascading load, not a cause)
- [ ] `notification-service`: completely normal (noise service)
- [ ] `evidence_labels`: auth-db check_deploys/query_logs = direct_evidence; user-service query_logs = causal_chain; api-gateway query_logs = causal_chain; cache-service = contextual; notification-service = contextual
- [ ] `ground_truth`: root_cause_service=auth-db, category=config_change, remediation=fix_config, causal_chain=[auth-db, user-service, api-gateway], red_herrings=[cache-service], anomalous_services=[api-gateway, user-service, auth-db, cache-service], failure_family=change-induced

### 3.3 Task Hard: `server/scenarios/task_hard.json`
- [ ] Create fixture for "Multi-Signal Cascading Failure with Red Herrings"
- [ ] 8 services: `api-gateway`, `search-service`, `inventory-service`, `cache-service`, `order-service`, `product-service`, `database`, `message-queue`
- [ ] Complex dependency graph: api-gateway → search-service → cache-service → inventory-service; api-gateway → order-service → database; inventory-service → database; search-service → product-service; order-service → message-queue
- [ ] `inventory-service` logs: increasing GC pauses, OOM warnings starting ~2hrs ago under load; code has memory leak
- [ ] `inventory-service` code_snippets: the leaky code (e.g., unbounded list growing in a loop), diff showing the change from 2hrs ago
- [ ] `cache-service` logs/metrics: aggressive eviction, high miss rate (caused by inventory-service slowness, not root cause)
- [ ] `order-service` deploys: a clean deploy 30 min ago (red herring — deploy is unrelated)
- [ ] `search-service` and `api-gateway`: timeout errors, elevated latency (symptoms)
- [ ] `product-service`, `database`, `message-queue`: mostly normal, minor noise
- [ ] `evidence_labels`: inventory-service inspect_code/query_logs/query_metrics = direct_evidence; cache-service query_metrics = causal_chain; search-service query_logs = causal_chain; api-gateway query_logs = causal_chain; order-service = contextual (red herring); others = contextual
- [ ] `ground_truth`: root_cause_service=inventory-service, category=resource_exhaustion, remediation=restart_service, causal_chain=[inventory-service, cache-service, search-service, api-gateway], red_herrings=[order-service, cache-service], anomalous_services=[api-gateway, search-service, inventory-service, cache-service, order-service], failure_family=capacity-related

### 3.4 Fixture quality check
- [ ] Ensure every service in every fixture has ALL data fields: dependencies, dependents, logs, metrics, deploys, code_snippets, status
- [ ] Ensure `evidence_labels` covers every `(service, action_type)` pair in each fixture
- [ ] Ensure log timestamps are consistent and make narrative sense
- [ ] Ensure metrics time-series shows clear anomalies at the right timestamps
- [ ] Ensure red herring data is plausible but not the root cause

---

## Phase 4: Scenario Loader (`server/scenario_loader.py`)

### 4.1 ScenarioLoader class
- [ ] `__init__()`: loads all 3 JSON fixtures from `server/scenarios/` into memory
- [ ] `get_scenario(task_id: str) -> dict`: returns the full fixture for a task
- [ ] `get_alert(task_id: str) -> AlertInfo`: returns the alert for a task
- [ ] `query_logs(task_id, service, severity=None, keyword=None) -> str`: filters log lines, formats as text
- [ ] `query_metrics(task_id, service, metric) -> str`: returns time-series data as formatted text
- [ ] `check_deploys(task_id, service) -> str`: returns deploy history as formatted text
- [ ] `trace_dependencies(task_id, service) -> str`: returns upstream/downstream with connection health as text
- [ ] `check_status(task_id, service) -> str`: returns health status as formatted text
- [ ] `inspect_code(task_id, service, file_path=None) -> str`: returns code snippets + recent diffs as text
- [ ] Validate that service exists in the scenario, return error message if not
- [ ] All return values are formatted text strings (not JSON) — the agent must parse them

### 4.2 Data formatting
- [ ] Logs: formatted as `[TIMESTAMP] [SEVERITY] message` lines, one per log entry
- [ ] Metrics: formatted as a text table with timestamp and value columns
- [ ] Deploys: formatted as `Deploy at TIMESTAMP by AUTHOR: DIFF_SUMMARY`
- [ ] Dependencies: formatted as `Upstream: [list] | Downstream: [list] | Connection health: ...`
- [ ] Code: formatted with filename headers and the code content
- [ ] Keyword filtering on logs: case-insensitive substring match on message field

---

## Phase 5: Grader (`server/grader.py`)

### 5.1 InvestigationRubric(Rubric)
- [ ] Extend OpenEnv's `Rubric` base class
- [ ] `__init__(scenario_data)`: stores the fixture's evidence_labels, ground_truth, dependency graph
- [ ] Internal state tracking: `actions_history`, `services_query_count`, `evidence_found` (which tiers collected per service), `has_causal_evidence` (bool)
- [ ] `forward(action, observation) -> float`: computes sum of:
  - Information gain: look up `evidence_labels[service][action_type]`, map to reward. Track redundancy.
  - Investigation strategy: check if action follows dependency graph, cross-references, or rules out. Uses `anomalous_services` from fixture.
  - Red herring resistance: check if service is in `red_herrings` (excluding those also in `causal_chain`), apply penalty if 3+ steps and agent has causal evidence
- [ ] `reset()`: clear all internal state

### 5.2 DiagnosisScorer
- [ ] Standalone class (NOT a Rubric subclass)
- [ ] `score(ground_truth, diagnosis, episode_state) -> dict`:
  - `root_cause_service` score: 0.20 full, 0.10 if one hop in causal_chain, 0.0 otherwise
  - `root_cause_category` score: 0.15 full, 0.07 if same failure family, 0.0 otherwise
  - `remediation` score: 0.10 full, 0.05 if in partial_remediation_mappings, 0.0 otherwise
  - `evidence_quality` score: count causal_chain services visited / total causal_chain. 0.10 if ≥80%, 0.05 if ≥50%, 0.0 otherwise
  - `efficiency` score: `0.15 * (steps_remaining / step_budget) ** 0.5`
  - Penalties: shotgun (-0.15 if step_count ≤ 3), circular (-0.10 if any service queried 4+ times), destructive_mismatch (-0.10 if remediation contradicts fixture data)
  - Returns dict with `score` (clamped 0.0–1.0) and `breakdown`

### 5.3 Failure family and partial remediation maps
- [ ] `FAILURE_FAMILIES`: dict mapping each category to its family name
  - change-induced: bad_deploy, config_change
  - capacity-related: resource_exhaustion, traffic_spike
  - infrastructure: dependency_failure, data_corruption
- [ ] `PARTIAL_REMEDIATIONS`: dict mapping (remediation, actual_category) → bool
  - (restart_service, bad_deploy) → True
  - (scale_up, traffic_spike) → True
  - (restart_service, resource_exhaustion) → True
  - (failover_to_backup, dependency_failure) → True

### 5.4 Tests
- [ ] Write `tests/test_grader.py`
- [ ] Test InvestigationRubric: direct evidence gives +0.12, redundant gives 0.0, repeated gives -0.05
- [ ] Test InvestigationRubric: following dependency graph gives +0.05
- [ ] Test InvestigationRubric: red herring penalty triggers correctly, exempts causal_chain services
- [ ] Test DiagnosisScorer: perfect diagnosis gives max score
- [ ] Test DiagnosisScorer: partial credit for close-but-wrong answers
- [ ] Test DiagnosisScorer: penalties for shotgun, circular, mismatch
- [ ] Test DiagnosisScorer: no diagnosis submitted gives 0.0
- [ ] Run tests: `pytest tests/test_grader.py`

---

## Phase 6: Environment (`server/environment.py`)

### 6.1 IncidentTriageEnv(Environment)
- [ ] Generic types: `Environment[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]`
- [ ] `__init__()`:
  - Initialize `ScenarioLoader`
  - Initialize `InvestigationRubric` (pass to `super().__init__(rubric=...)`)
  - Initialize `DiagnosisScorer` as `self.diagnosis_scorer`
  - Initialize `self._completed_episodes: dict[str, dict] = {}` for grader endpoint
- [ ] `reset(seed=None, episode_id=None, task_id="easy_single_service_failure", **kwargs) -> IncidentTriageObservation`:
  - Generate episode_id if not provided
  - Load scenario via `ScenarioLoader.get_scenario(task_id)`
  - Store current scenario, ground_truth, step_budget
  - Reset rubric via `self._reset_rubric()`
  - Initialize state: step_count=0, steps_remaining=step_budget, empty lists
  - Return initial observation: alert from scenario, result=None, steps_remaining, services_investigated=[]
- [ ] `step(action: IncidentTriageAction, ...) -> IncidentTriageObservation`:
  - Decrement steps_remaining, increment step_count
  - If `action_type == "submit_diagnosis"`:
    - Score via DiagnosisScorer
    - Store episode result in `self._completed_episodes[episode_id]`
    - Return observation with done=True
  - Else: dispatch to ScenarioLoader based on action_type
  - Track service in services_investigated (if not already)
  - Track action in evidence_collected
  - Compute per-step reward via `self._apply_rubric(action, obs)`
  - If steps_remaining == 0 and not done: auto-terminate (done=True, no terminal reward)
  - Return observation
- [ ] `state` property: return current `IncidentTriageState`

### 6.2 Episode result storage
- [ ] On episode completion (submit_diagnosis or budget exhaustion), store in `self._completed_episodes[episode_id]`:
  - task_id, ground_truth, diagnosis (if submitted), episode_state, grader_score (if submitted)
- [ ] Expose a method `get_episode_result(episode_id) -> dict` for the grader endpoint

### 6.3 Tests
- [ ] Write `tests/test_environment.py`
- [ ] Test reset: returns valid initial observation, state is clean
- [ ] Test step: query_logs returns log data, decrements steps_remaining
- [ ] Test step: submit_diagnosis ends episode (done=True)
- [ ] Test step budget exhaustion: auto-terminates
- [ ] Test state: reflects current episode accurately
- [ ] Test episode result storage: grader can retrieve after completion
- [ ] Run tests: `pytest tests/test_environment.py`

---

## Phase 7: FastAPI App (`server/app.py`)

### 7.1 Create app using create_app()
- [ ] Import `create_app` from OpenEnv
- [ ] Pass `IncidentTriageEnv`, `IncidentTriageAction`, `IncidentTriageObservation`, `IncidentTriageState`
- [ ] Verify that the base endpoints work: `/health`, `/schema`, `/reset`, `/step`, `/state`, `/ws`, `/openapi.json`

### 7.2 Custom endpoints
- [ ] `GET /tasks`: return list of tasks with task_id, task_name, difficulty, step_budget + action JSON schema
- [ ] `POST /grader`: accept `{"episode_id": "..."}`, look up completed episode, return score + breakdown
- [ ] `POST /baseline`: trigger baseline inference script, return scores for all 3 tasks. Handle missing OPENAI_API_KEY gracefully.

### 7.3 Episode result sharing
- [ ] The custom endpoints need access to the environment's `_completed_episodes`. Since `create_app()` manages the environment lifecycle, use a shared module-level dict or FastAPI dependency injection to bridge this.
- [ ] Alternative: the environment writes completed episode results to a module-level store that the endpoints can read.

### 7.4 Test
- [ ] Start server locally: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- [ ] Test `/health` returns 200
- [ ] Test `/tasks` returns 3 tasks with correct schema
- [ ] Test WebSocket flow: connect, reset, step, step, submit_diagnosis
- [ ] Test `/grader` returns score after episode completion

---

## Phase 8: Client (`client.py`)

### 8.1 IncidentTriageClient(EnvClient)
- [ ] Extend `EnvClient[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]`
- [ ] Implement `_step_payload(action)`: serialize action to dict
- [ ] Implement `_parse_result(payload)`: deserialize to StepResult[IncidentTriageObservation]
- [ ] Implement `_parse_state(payload)`: deserialize to IncidentTriageState

### 8.2 Test
- [ ] Test client can connect, reset, step, and receive typed observations
- [ ] Test sync wrapper works: `with IncidentTriageClient(...).sync() as env:`

---

## Phase 9: Baseline Inference Script (`scripts/baseline_inference.py`)

### 9.1 Implementation
- [ ] Read `OPENAI_API_KEY` from env vars
- [ ] Connect to environment via WebSocket (use the client from client.py or raw websockets)
- [ ] System prompt: instruct agent to investigate systematically and respond with JSON actions
- [ ] For each task (easy, medium, hard):
  - Reset with task_id
  - Loop: send observation to LLM → parse JSON action → step → check done
  - On done: call `/grader` endpoint, record score
  - Handle max retries for JSON parsing failures
- [ ] Print final scores for all 3 tasks
- [ ] Use `gpt-4o` or `gpt-4o-mini` as default model (configurable via env var)

### 9.2 Test
- [ ] Run against local server: `python scripts/baseline_inference.py`
- [ ] Verify it completes all 3 tasks without crashing
- [ ] Record baseline scores for README

---

## Phase 10: Docker & Deployment

### 10.1 Dockerfile
- [ ] Base: `python:3.10-slim`
- [ ] Copy project files
- [ ] Install dependencies: `pip install -e .` or `pip install -r requirements.txt`
- [ ] Expose port 8000
- [ ] CMD: `uvicorn server.app:app --host 0.0.0.0 --port 8000`

### 10.2 Test Docker
- [ ] `docker build -t incident-triage-env .`
- [ ] `docker run -p 8000:8000 incident-triage-env`
- [ ] Verify `/health` responds
- [ ] Run baseline script against Docker container
- [ ] Run `openenv validate --url http://localhost:8000` (if available)

### 10.3 Hugging Face Space
- [ ] Add HF Space metadata to README.md frontmatter:
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
- [ ] Push to HF Spaces: `openenv push --repo-id <username>/incident-triage-env`
- [ ] Verify Space deploys and responds to requests

---

## Phase 11: README & Documentation

### 11.1 README.md
- [ ] HF Space frontmatter (from 10.3)
- [ ] Environment description and motivation (from WHAT_AND_WHY.md)
- [ ] Action space definition: table of all 7 action types with parameters
- [ ] Observation space definition: model fields and types
- [ ] Task descriptions: 3 tasks with difficulty, step budget, scenario summary
- [ ] Setup instructions: pip install, docker build, docker run
- [ ] Usage instructions: WebSocket connection, reset, step, grader
- [ ] Baseline scores: table of easy/medium/hard scores
- [ ] Reward function summary

---

## Phase 12: Validation & Final Checks

### 12.1 Automated validation
- [ ] `openenv validate` passes (local file checks)
- [ ] `openenv validate --url http://localhost:8000` passes (runtime checks)
- [ ] Docker builds cleanly: `docker build -t incident-triage-env .`
- [ ] Docker runs cleanly: `docker run -p 8000:8000 incident-triage-env`
- [ ] Baseline script runs and produces scores for all 3 tasks

### 12.2 Manual validation
- [ ] HF Space deploys and returns 200 on ping
- [ ] `/reset` endpoint responds correctly
- [ ] `/tasks` returns 3 tasks
- [ ] `/grader` returns scores in 0.0–1.0 range
- [ ] `/baseline` triggers inference and returns scores
- [ ] WebSocket session works end-to-end (reset → step → ... → submit_diagnosis)
- [ ] Grader scores are deterministic (run same task twice, get same score)
- [ ] Hard task genuinely challenges frontier models (score < 0.5 expected)

### 12.3 Pre-submission checklist (from hackathon rules)
- [ ] HF Space deploys and returns 200 + responds to `reset()`
- [ ] `openenv validate` passes (openenv.yaml, typed models, step/reset/state)
- [ ] Dockerfile builds
- [ ] Baseline script runs without error and produces scores
- [ ] 3+ tasks with graders, all scores in 0.0–1.0 range
- [ ] `/baseline` endpoint works
- [ ] `/grader` endpoint works
- [ ] `/tasks` endpoint works

---

## Implementation Order (Critical Path)

```
Phase 1 (scaffolding) → Phase 2 (models) → Phase 3 (fixtures) → Phase 4 (scenario loader)
    → Phase 5 (grader) → Phase 6 (environment) → Phase 7 (app) → Phase 8 (client)
    → Phase 9 (baseline) → Phase 10 (docker) → Phase 11 (README) → Phase 12 (validation)
```

**Parallelizable**: Phase 3 (fixtures) can be written in parallel with Phase 2 (models) since fixtures are JSON. Phase 8 (client) can be done in parallel with Phase 9 (baseline). Phase 11 (README) can start after Phase 7.

**Estimated effort**: ~8-12 hours for a single developer with Claude Code assistance. The most time-intensive phases are Phase 3 (crafting realistic fixture data) and Phase 5 (grader logic).
