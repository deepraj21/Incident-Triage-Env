# Incident Triage Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete OpenEnv-compliant RL environment that simulates production incident triage — an AI agent investigates microservices by querying logs, metrics, deploys, dependencies, and code to diagnose root causes.

**Architecture:** FastAPI server using OpenEnv's `create_app()` with a single `IncidentTriageEnv(Environment)` class. Scenario data lives in pre-crafted JSON fixtures. Rewards are computed by an `InvestigationRubric(Rubric)` per step and a standalone `DiagnosisScorer` at episode end. Custom `/tasks`, `/grader`, `/baseline` endpoints are added to the FastAPI app. All agent interaction uses WebSocket (`/ws`) for state persistence.

**Tech Stack:** Python 3.10+, OpenEnv SDK (from `../OpenEnv`), FastAPI, Pydantic v2, uvicorn, Google Generative AI SDK (`google-genai`), Docker, pytest

**Spec:** `docs/superpowers/specs/2026-03-26-incident-triage-env-design.md`

**OpenEnv reference files (READ THESE FIRST):**
- `../OpenEnv/src/openenv/core/env_server/types.py` — Action, Observation, State base classes
- `../OpenEnv/src/openenv/core/env_server/interfaces.py` — Environment base class
- `../OpenEnv/src/openenv/core/env_server/http_server.py` — `create_app()`, WebSocket handler
- `../OpenEnv/src/openenv/core/rubrics/base.py` — Rubric base class
- `../OpenEnv/envs/coding_env/` — reference environment (models.py, client.py, server/)

---

## File Structure

```
incident-triage-env/
├── openenv.yaml                    # OpenEnv manifest (spec_version, name, type, runtime, app, port)
├── pyproject.toml                  # Project config with dependencies
├── Dockerfile                      # Docker build for HF Spaces deployment
├── README.md                       # HF Space frontmatter + docs
├── .env.example                    # Example env vars (GEMINI_API_KEY, OPENROUTER_API_KEY)
├── models.py                       # All Pydantic models: enums, action, observation, state
├── client.py                       # EnvClient subclass for typed client access
├── server/
│   ├── __init__.py                 # Empty
│   ├── app.py                      # FastAPI app: create_app() + custom /tasks, /grader, /baseline
│   ├── environment.py              # IncidentTriageEnv(Environment) — core logic
│   ├── grader.py                   # InvestigationRubric(Rubric) + DiagnosisScorer
│   ├── scenario_loader.py          # Loads JSON fixtures, filters/formats data per query
│   ├── episode_store.py            # Module-level dict storing completed episode results
│   └── scenarios/
│       ├── task_easy.json          # Single service failure (3 services)
│       ├── task_medium.json        # Cascading dependency failure (5 services)
│       └── task_hard.json          # Multi-signal cascade with red herrings (8 services)
├── scripts/
│   └── baseline_inference.py       # Gemini/OpenRouter API client, runs all 3 tasks via WebSocket
└── tests/
    ├── __init__.py
    ├── test_models.py              # Model creation, validation, serialization
    ├── test_scenario_loader.py     # Data filtering, formatting, edge cases
    ├── test_grader.py              # Rubric scoring, diagnosis scoring, penalties
    └── test_environment.py         # Full episode flow: reset, step, terminate
```

**Why `episode_store.py` is separate:** The `/grader` endpoint needs access to completed episode data, but `create_app()` manages environment lifecycle internally. A module-level store bridges this — the environment writes to it on episode end, the endpoint reads from it.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `incident-triage-env/openenv.yaml`
- Create: `incident-triage-env/pyproject.toml`
- Create: `incident-triage-env/.env.example`
- Create: `incident-triage-env/server/__init__.py`
- Create: `incident-triage-env/tests/__init__.py`

- [ ] **Step 1: Create the project directory**

```bash
cd /Users/abhishekmallick/Projects/hackathon/OpenEnv-Hackathon
mkdir -p incident-triage-env/server/scenarios incident-triage-env/scripts incident-triage-env/tests
```

- [ ] **Step 2: Create openenv.yaml**

```yaml
# incident-triage-env/openenv.yaml
spec_version: 1
name: incident_triage_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

- [ ] **Step 3: Create pyproject.toml**

```toml
# incident-triage-env/pyproject.toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "incident-triage-env"
version = "0.1.0"
description = "OpenEnv environment for production incident triage"
requires-python = ">=3.10"
dependencies = [
    "openenv",
    "fastapi>=0.100.0",
    "uvicorn[standard]>=0.20.0",
    "pydantic>=2.0",
    "websockets>=11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "httpx>=0.24.0",
]
baseline = [
    "google-genai>=1.0",
    "openai>=1.0",  # for OpenRouter fallback (OpenAI-compatible API)
]
```

- [ ] **Step 4: Create .env.example**

```bash
# incident-triage-env/.env.example
# Primary: Google Gemini (free tier via https://aistudio.google.com)
GEMINI_API_KEY=your-gemini-api-key-here
# Fallback: OpenRouter (free models via https://openrouter.ai)
OPENROUTER_API_KEY=your-openrouter-key-here
```

- [ ] **Step 5: Create empty __init__.py files**

```python
# incident-triage-env/server/__init__.py
# incident-triage-env/tests/__init__.py
```

- [ ] **Step 6: Verify structure**

```bash
find incident-triage-env -type f | sort
```

Expected: all files listed above exist.

- [ ] **Step 7: Commit**

```bash
cd incident-triage-env
git init
git add -A
git commit -m "chore: scaffold incident-triage-env project"
```

---

### Task 2: Pydantic Models

**Files:**
- Create: `incident-triage-env/models.py`
- Create: `incident-triage-env/tests/test_models.py`

- [ ] **Step 1: Write failing tests for models**

```python
# tests/test_models.py
import pytest
from models import (
    RootCauseCategory,
    Remediation,
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
    AlertInfo,
    ActionResult,
)


def test_query_logs_action_valid():
    action = IncidentTriageAction(
        action_type="query_logs",
        service="payments-service",
        keyword="error",
    )
    assert action.action_type == "query_logs"
    assert action.service == "payments-service"


def test_query_logs_action_missing_service_fails():
    with pytest.raises(Exception):  # ValidationError
        IncidentTriageAction(action_type="query_logs")


def test_submit_diagnosis_valid():
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    assert action.action_type == "submit_diagnosis"


def test_submit_diagnosis_missing_fields_fails():
    with pytest.raises(Exception):
        IncidentTriageAction(
            action_type="submit_diagnosis",
            root_cause_service="payments-service",
            # missing root_cause_category and remediation
        )


def test_query_metrics_requires_metric():
    with pytest.raises(Exception):
        IncidentTriageAction(
            action_type="query_metrics",
            service="payments-service",
            # missing metric
        )


def test_observation_initial():
    obs = IncidentTriageObservation(
        alert=AlertInfo(
            service="api-gateway",
            message="HTTP 500 spike",
            severity="critical",
            timestamp="2026-03-26T14:32:00Z",
        ),
        result=None,
        steps_remaining=10,
    )
    assert obs.done is False
    assert obs.reward is None
    assert obs.result is None
    assert obs.steps_remaining == 10


def test_observation_with_result():
    obs = IncidentTriageObservation(
        alert=AlertInfo(
            service="api-gateway",
            message="HTTP 500 spike",
            severity="critical",
            timestamp="2026-03-26T14:32:00Z",
        ),
        result=ActionResult(action_type="query_logs", data="[14:32:15] [ERROR] NPE..."),
        steps_remaining=9,
        services_investigated=["payments-service"],
        reward=0.12,
    )
    assert obs.result.action_type == "query_logs"
    assert obs.reward == 0.12


def test_state_model():
    state = IncidentTriageState(
        episode_id="ep-001",
        step_count=3,
        task_id="easy_single_service_failure",
        steps_remaining=7,
        services_investigated=["payments-service"],
        actions_taken=[{"action_type": "query_logs", "service": "payments-service"}],
        evidence_collected={"payments-service": ["query_logs"]},
    )
    assert state.step_count == 3
    assert state.steps_remaining == 7


def test_action_serialization_roundtrip():
    action = IncidentTriageAction(
        action_type="query_logs",
        service="payments-service",
        severity="error",
    )
    data = action.model_dump()
    restored = IncidentTriageAction(**data)
    assert restored.action_type == action.action_type
    assert restored.service == action.service


def test_enums():
    assert RootCauseCategory.BAD_DEPLOY.value == "bad_deploy"
    assert Remediation.ROLLBACK_DEPLOY.value == "rollback_deploy"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd incident-triage-env
PYTHONPATH=. pytest tests/test_models.py -v
```
Expected: ImportError — `models` module does not exist yet.

- [ ] **Step 3: Implement models.py**

```python
# models.py
"""Pydantic models for the Incident Triage Environment."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator

from openenv.core.env_server.types import Action, Observation, State


class RootCauseCategory(str, Enum):
    BAD_DEPLOY = "bad_deploy"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIG_CHANGE = "config_change"
    TRAFFIC_SPIKE = "traffic_spike"
    DATA_CORRUPTION = "data_corruption"


class Remediation(str, Enum):
    ROLLBACK_DEPLOY = "rollback_deploy"
    SCALE_UP = "scale_up"
    RESTART_SERVICE = "restart_service"
    FIX_CONFIG = "fix_config"
    ENABLE_RATE_LIMITING = "enable_rate_limiting"
    FAILOVER_TO_BACKUP = "failover_to_backup"


class EvidenceTier(str, Enum):
    DIRECT_EVIDENCE = "direct_evidence"
    CAUSAL_CHAIN = "causal_chain"
    CONTEXTUAL = "contextual"
    NONE = "none"


ActionType = Literal[
    "query_logs",
    "query_metrics",
    "check_deploys",
    "trace_dependencies",
    "check_status",
    "inspect_code",
    "submit_diagnosis",
]


class IncidentTriageAction(Action):
    """Single polymorphic action model with action_type discriminator."""

    action_type: ActionType
    service: Optional[str] = None
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

    @model_validator(mode="after")
    def validate_action_fields(self) -> "IncidentTriageAction":
        if self.action_type == "submit_diagnosis":
            missing = []
            if not self.root_cause_service:
                missing.append("root_cause_service")
            if not self.root_cause_category:
                missing.append("root_cause_category")
            if not self.remediation:
                missing.append("remediation")
            if missing:
                raise ValueError(
                    f"submit_diagnosis requires: {', '.join(missing)}"
                )
        elif self.action_type == "query_metrics":
            if not self.service:
                raise ValueError("query_metrics requires 'service'")
            if not self.metric:
                raise ValueError("query_metrics requires 'metric'")
        else:
            if not self.service:
                raise ValueError(f"{self.action_type} requires 'service'")
        return self


class AlertInfo(BaseModel):
    service: str
    message: str
    severity: Literal["critical", "warning"]
    timestamp: str


class ActionResult(BaseModel):
    action_type: str
    data: str


class IncidentTriageObservation(Observation):
    """Observation returned by step() and reset()."""

    alert: AlertInfo
    result: Optional[ActionResult] = None
    steps_remaining: int
    services_investigated: List[str] = []


class IncidentTriageState(State):
    """Internal episode state."""

    task_id: str = ""
    steps_remaining: int = 0
    services_investigated: List[str] = []
    actions_taken: List[Dict[str, Any]] = []
    evidence_collected: Dict[str, List[str]] = {}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/test_models.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add models.py tests/test_models.py
git commit -m "feat: add Pydantic models — action, observation, state, enums"
```

---

### Task 3: Easy Scenario Fixture

**Files:**
- Create: `incident-triage-env/server/scenarios/task_easy.json`

This is the most detailed fixture — study it as the template for medium and hard.

- [ ] **Step 1: Write task_easy.json**

Create `server/scenarios/task_easy.json` — a complete scenario for "Single Service Failure" with 3 services (`payments-service`, `database`, `api-gateway`). The fixture must include:

- `task_id`: `"easy_single_service_failure"`
- `task_name`: `"Single Service Failure"`
- `difficulty`: `"easy"`
- `step_budget`: `10`
- `alert`: `payments-service` HTTP 500 spike
- `services`: each with `dependencies`, `dependents`, `logs` (15-20 entries), `metrics` (4 types with 10+ data points each), `deploys`, `code_snippets`, `status`
- `evidence_labels`: every `(service, action_type)` pair mapped to a tier
- `ground_truth`: full ground truth object

Key data points:
- `payments-service` logs: normal logs before 14:30, then `NullPointerException` errors starting 14:32
- `payments-service` metrics: `error_rate` flat at 0.1% until 14:32, then jumps to 15%. `latency_p99` spikes. `cpu` and `memory` stay normal.
- `payments-service` deploys: one deploy at 14:12 by `dev-alice`: "Refactored PaymentProcessor to use direct account access"
- `payments-service` code_snippets: buggy code showing `p.getAccount()` returning null, plus the diff
- `database` and `api-gateway`: all normal data, no anomalies
- `evidence_labels`: `payments-service` → `query_logs`: `direct_evidence`, `query_metrics`: `direct_evidence`, `check_deploys`: `direct_evidence`, `inspect_code`: `direct_evidence`, `trace_dependencies`: `contextual`, `check_status`: `contextual`. All `database` and `api-gateway` pairs → `contextual`.

**The JSON should be ~200-300 lines with realistic, detailed data. Use realistic timestamps, log messages, and metric values. Make it look like real production data.**

- [ ] **Step 2: Validate the JSON parses correctly**

```bash
python -c "import json; json.load(open('server/scenarios/task_easy.json')); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add server/scenarios/task_easy.json
git commit -m "feat: add easy scenario fixture — single service failure"
```

---

### Task 4: Medium Scenario Fixture

**Files:**
- Create: `incident-triage-env/server/scenarios/task_medium.json`

- [ ] **Step 1: Write task_medium.json**

"Cascading Dependency Failure" — 5 services: `api-gateway`, `user-service`, `auth-db`, `cache-service`, `notification-service`.

Key data:
- Dependency graph: `api-gateway` → `user-service` → `auth-db`; `api-gateway` → `cache-service`; `user-service` → `notification-service`
- `api-gateway` logs: HTTP 500, timeouts to user-service. Metrics: error_rate spike, latency spike.
- `user-service` logs: connection pool exhaustion, timeout to auth-db. Metrics: connection count maxed, latency spike.
- `auth-db` logs: "max_connections limit reached", config reload event at 14:15. Metrics: connections stuck at 20 (was 200). Deploys: config change at 14:10 by `ops-bob`.
- `cache-service` (red herring): elevated latency, some cache misses. Caused by cascading load, not a root cause.
- `notification-service`: completely normal.
- `evidence_labels`: `auth-db` query_logs/check_deploys = `direct_evidence`; `user-service` query_logs = `causal_chain`; `api-gateway` query_logs = `causal_chain`; everything else = `contextual`.
- `ground_truth`: root_cause=auth-db, category=config_change, remediation=fix_config, causal_chain=[auth-db, user-service, api-gateway], red_herrings=[cache-service], anomalous_services=[api-gateway, user-service, auth-db, cache-service].

- [ ] **Step 2: Validate JSON**

```bash
python -c "import json; json.load(open('server/scenarios/task_medium.json')); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add server/scenarios/task_medium.json
git commit -m "feat: add medium scenario fixture — cascading dependency failure"
```

---

### Task 5: Hard Scenario Fixture

**Files:**
- Create: `incident-triage-env/server/scenarios/task_hard.json`

- [ ] **Step 1: Write task_hard.json**

"Multi-Signal Cascading Failure with Red Herrings" — 8 services: `api-gateway`, `search-service`, `inventory-service`, `cache-service`, `order-service`, `product-service`, `database`, `message-queue`.

Key data:
- Dependency graph: `api-gateway` → `search-service` → `cache-service` → `inventory-service`; `api-gateway` → `order-service` → `database`; `inventory-service` → `database`; `search-service` → `product-service`; `order-service` → `message-queue`
- `inventory-service` logs: GC pauses starting 2hrs ago, OOM warnings under load, growing heap. Code: memory leak in `InventoryCache.refresh()` — unbounded list. Diff shows change from 2hrs ago introducing the leak.
- `cache-service` (causal chain + red herring dual role): aggressive evictions, high miss rate. Caused by inventory-service slowness.
- `search-service`: timeouts to cache-service, elevated latency.
- `api-gateway`: HTTP 500s, timeouts.
- `order-service` (red herring): clean deploy 30min ago, no actual issues from it. Normal operation.
- `product-service`, `database`, `message-queue`: normal.
- `evidence_labels`: `inventory-service` inspect_code/query_logs/query_metrics = `direct_evidence`; `cache-service` query_metrics = `causal_chain`; `search-service` query_logs = `causal_chain`; `api-gateway` query_logs = `causal_chain`; `order-service` all = `contextual`; others = `contextual`.
- `ground_truth`: root_cause=inventory-service, category=resource_exhaustion, remediation=restart_service, causal_chain=[inventory-service, cache-service, search-service, api-gateway], red_herrings=[order-service, cache-service], anomalous_services=[api-gateway, search-service, inventory-service, cache-service, order-service].

**This fixture should be the longest (~400-500 lines) with the most realistic and detailed data. The code snippets must clearly show the memory leak if inspected carefully.**

- [ ] **Step 2: Validate JSON**

```bash
python -c "import json; json.load(open('server/scenarios/task_hard.json')); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add server/scenarios/task_hard.json
git commit -m "feat: add hard scenario fixture — multi-signal cascade with red herrings"
```

---

### Task 6: Scenario Loader

**Files:**
- Create: `incident-triage-env/server/scenario_loader.py`
- Create: `incident-triage-env/tests/test_scenario_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scenario_loader.py
import pytest
from server.scenario_loader import ScenarioLoader


@pytest.fixture
def loader():
    return ScenarioLoader()


def test_get_scenario(loader):
    scenario = loader.get_scenario("easy_single_service_failure")
    assert scenario["task_id"] == "easy_single_service_failure"
    assert scenario["step_budget"] == 10


def test_get_scenario_invalid_task(loader):
    with pytest.raises(ValueError, match="Unknown task_id"):
        loader.get_scenario("nonexistent_task")


def test_get_alert(loader):
    alert = loader.get_alert("easy_single_service_failure")
    assert alert["service"] == "payments-service"
    assert alert["severity"] == "critical"


def test_query_logs_returns_text(loader):
    result = loader.query_logs("easy_single_service_failure", "payments-service")
    assert isinstance(result, str)
    assert len(result) > 0


def test_query_logs_with_keyword_filter(loader):
    result = loader.query_logs(
        "easy_single_service_failure", "payments-service", keyword="NullPointer"
    )
    assert "NullPointer" in result


def test_query_logs_with_severity_filter(loader):
    result = loader.query_logs(
        "easy_single_service_failure", "payments-service", severity="error"
    )
    # All returned lines should be error-level
    for line in result.strip().split("\n"):
        if line.strip():
            assert "ERROR" in line.upper()


def test_query_logs_invalid_service(loader):
    result = loader.query_logs("easy_single_service_failure", "nonexistent-service")
    assert "not found" in result.lower() or "unknown" in result.lower()


def test_query_metrics(loader):
    result = loader.query_metrics(
        "easy_single_service_failure", "payments-service", "error_rate"
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_check_deploys(loader):
    result = loader.check_deploys("easy_single_service_failure", "payments-service")
    assert isinstance(result, str)
    assert "dev-alice" in result or "deploy" in result.lower()


def test_trace_dependencies(loader):
    result = loader.trace_dependencies("easy_single_service_failure", "payments-service")
    assert isinstance(result, str)
    assert "database" in result.lower() or "upstream" in result.lower() or "downstream" in result.lower()


def test_check_status(loader):
    result = loader.check_status("easy_single_service_failure", "payments-service")
    assert isinstance(result, str)
    assert "degraded" in result.lower() or "health" in result.lower()


def test_inspect_code(loader):
    result = loader.inspect_code("easy_single_service_failure", "payments-service")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_evidence_tier(loader):
    tier = loader.get_evidence_tier(
        "easy_single_service_failure", "payments-service", "query_logs"
    )
    assert tier == "direct_evidence"


def test_get_evidence_tier_contextual(loader):
    tier = loader.get_evidence_tier(
        "easy_single_service_failure", "database", "query_logs"
    )
    assert tier == "contextual"


def test_get_ground_truth(loader):
    gt = loader.get_ground_truth("easy_single_service_failure")
    assert gt["root_cause_service"] == "payments-service"
    assert gt["root_cause_category"] == "bad_deploy"


def test_list_services(loader):
    services = loader.list_services("easy_single_service_failure")
    assert "payments-service" in services
    assert len(services) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. pytest tests/test_scenario_loader.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement scenario_loader.py**

```python
# server/scenario_loader.py
"""Loads scenario fixtures and provides filtered data access per agent query."""
import json
from pathlib import Path
from typing import Optional


SCENARIOS_DIR = Path(__file__).parent / "scenarios"


class ScenarioLoader:
    def __init__(self):
        self._scenarios = {}
        for f in SCENARIOS_DIR.glob("task_*.json"):
            with open(f) as fh:
                data = json.load(fh)
                self._scenarios[data["task_id"]] = data

    def get_scenario(self, task_id: str) -> dict:
        if task_id not in self._scenarios:
            raise ValueError(f"Unknown task_id: {task_id}. Available: {list(self._scenarios.keys())}")
        return self._scenarios[task_id]

    def list_tasks(self) -> list[dict]:
        return [
            {
                "task_id": s["task_id"],
                "task_name": s["task_name"],
                "difficulty": s["difficulty"],
                "step_budget": s["step_budget"],
            }
            for s in self._scenarios.values()
        ]

    def list_services(self, task_id: str) -> list[str]:
        return list(self.get_scenario(task_id)["services"].keys())

    def get_alert(self, task_id: str) -> dict:
        return self.get_scenario(task_id)["alert"]

    def get_ground_truth(self, task_id: str) -> dict:
        return self.get_scenario(task_id)["ground_truth"]

    def get_evidence_tier(self, task_id: str, service: str, action_type: str) -> str:
        scenario = self.get_scenario(task_id)
        labels = scenario.get("evidence_labels", {})
        return labels.get(service, {}).get(action_type, "none")

    def _validate_service(self, task_id: str, service: str) -> Optional[str]:
        scenario = self.get_scenario(task_id)
        if service not in scenario["services"]:
            available = list(scenario["services"].keys())
            return f"Service '{service}' not found. Available services: {', '.join(available)}"
        return None

    def query_logs(
        self, task_id: str, service: str,
        severity: Optional[str] = None, keyword: Optional[str] = None,
    ) -> str:
        err = self._validate_service(task_id, service)
        if err:
            return err
        logs = self.get_scenario(task_id)["services"][service]["logs"]
        filtered = logs
        if severity:
            filtered = [l for l in filtered if l["severity"].lower() == severity.lower()]
        if keyword:
            filtered = [l for l in filtered if keyword.lower() in l["message"].lower()]
        if not filtered:
            return "No matching log entries found."
        lines = []
        for entry in filtered:
            lines.append(f"[{entry['timestamp']}] [{entry['severity'].upper()}] {entry['message']}")
        return "\n".join(lines)

    def query_metrics(self, task_id: str, service: str, metric: str) -> str:
        err = self._validate_service(task_id, service)
        if err:
            return err
        metrics = self.get_scenario(task_id)["services"][service]["metrics"]
        if metric not in metrics:
            return f"Metric '{metric}' not available. Available: {', '.join(metrics.keys())}"
        data_points = metrics[metric]
        lines = [f"{'Timestamp':<20} {'Value':>10}"]
        lines.append("-" * 32)
        for dp in data_points:
            lines.append(f"{dp['ts']:<20} {dp['val']:>10}")
        return "\n".join(lines)

    def check_deploys(self, task_id: str, service: str) -> str:
        err = self._validate_service(task_id, service)
        if err:
            return err
        deploys = self.get_scenario(task_id)["services"][service]["deploys"]
        if not deploys:
            return "No recent deploys."
        lines = []
        for d in deploys:
            lines.append(f"Deploy at {d['timestamp']} by {d['author']}: {d['diff_summary']}")
        return "\n".join(lines)

    def trace_dependencies(self, task_id: str, service: str) -> str:
        err = self._validate_service(task_id, service)
        if err:
            return err
        svc = self.get_scenario(task_id)["services"][service]
        deps = svc.get("dependencies", [])
        dependents = svc.get("dependents", [])
        lines = [
            f"Service: {service}",
            f"Upstream (dependencies): {', '.join(deps) if deps else 'none'}",
            f"Downstream (dependents): {', '.join(dependents) if dependents else 'none'}",
        ]
        return "\n".join(lines)

    def check_status(self, task_id: str, service: str) -> str:
        err = self._validate_service(task_id, service)
        if err:
            return err
        status = self.get_scenario(task_id)["services"][service]["status"]
        lines = [
            f"Service: {service}",
            f"Health: {status['health']}",
            f"Uptime: {status['uptime']}",
            f"Active alerts: {', '.join(status['active_alerts']) if status['active_alerts'] else 'none'}",
        ]
        return "\n".join(lines)

    def inspect_code(self, task_id: str, service: str, file_path: Optional[str] = None) -> str:
        err = self._validate_service(task_id, service)
        if err:
            return err
        snippets = self.get_scenario(task_id)["services"][service]["code_snippets"]
        if not snippets:
            return "No code snippets available for this service."
        if file_path and file_path in snippets:
            return f"--- {file_path} ---\n{snippets[file_path]}"
        lines = []
        for name, code in snippets.items():
            lines.append(f"--- {name} ---")
            lines.append(code)
            lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/test_scenario_loader.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/scenario_loader.py tests/test_scenario_loader.py
git commit -m "feat: add scenario loader with data filtering and formatting"
```

---

### Task 7: Episode Store

**Files:**
- Create: `incident-triage-env/server/episode_store.py`

- [ ] **Step 1: Implement episode_store.py**

```python
# server/episode_store.py
"""Module-level store for completed episode results.

The environment writes here on episode end; the /grader endpoint reads from here.
This bridges the gap between create_app()'s environment lifecycle and custom endpoints.
"""
from typing import Any, Dict, Optional

_completed_episodes: Dict[str, Dict[str, Any]] = {}


def store_episode(episode_id: str, result: Dict[str, Any]) -> None:
    _completed_episodes[episode_id] = result


def get_episode(episode_id: str) -> Optional[Dict[str, Any]]:
    return _completed_episodes.get(episode_id)


def list_episodes() -> list[str]:
    return list(_completed_episodes.keys())
```

- [ ] **Step 2: Commit**

```bash
git add server/episode_store.py
git commit -m "feat: add episode store for grader endpoint access"
```

---

### Task 8: Grader — InvestigationRubric & DiagnosisScorer

**Files:**
- Create: `incident-triage-env/server/grader.py`
- Create: `incident-triage-env/tests/test_grader.py`

- [ ] **Step 1: Write failing tests for grader**

```python
# tests/test_grader.py
import pytest
from models import (
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
    AlertInfo,
    ActionResult,
    RootCauseCategory,
    Remediation,
)
from server.grader import InvestigationRubric, DiagnosisScorer, FAILURE_FAMILIES


EASY_GROUND_TRUTH = {
    "root_cause_service": "payments-service",
    "root_cause_category": "bad_deploy",
    "remediation": "rollback_deploy",
    "causal_chain": ["payments-service"],
    "red_herrings": [],
    "evidence_services": ["payments-service"],
    "anomalous_services": ["payments-service"],
    "failure_family": "change-induced",
}

EASY_EVIDENCE_LABELS = {
    "payments-service": {
        "query_logs": "direct_evidence",
        "query_metrics": "direct_evidence",
        "check_deploys": "direct_evidence",
        "inspect_code": "direct_evidence",
        "trace_dependencies": "contextual",
        "check_status": "contextual",
    },
    "database": {
        "query_logs": "contextual",
        "query_metrics": "contextual",
        "check_deploys": "contextual",
        "inspect_code": "contextual",
        "trace_dependencies": "contextual",
        "check_status": "contextual",
    },
}

EASY_SERVICES = {
    "payments-service": {"dependencies": ["database"], "dependents": ["api-gateway"]},
    "database": {"dependencies": [], "dependents": ["payments-service"]},
    "api-gateway": {"dependencies": ["payments-service"], "dependents": []},
}


def _make_alert():
    return AlertInfo(
        service="payments-service", message="HTTP 500", severity="critical", timestamp="2026-03-26T14:32:00Z"
    )


def _make_obs(action_type="query_logs", steps_remaining=9, services=None):
    return IncidentTriageObservation(
        alert=_make_alert(),
        result=ActionResult(action_type=action_type, data="some data"),
        steps_remaining=steps_remaining,
        services_investigated=services or [],
    )


class TestInvestigationRubric:
    def setup_method(self):
        self.rubric = InvestigationRubric(
            evidence_labels=EASY_EVIDENCE_LABELS,
            ground_truth=EASY_GROUND_TRUTH,
            services=EASY_SERVICES,
        )

    def test_direct_evidence_reward(self):
        action = IncidentTriageAction(action_type="query_logs", service="payments-service")
        obs = _make_obs()
        reward = self.rubric.forward(action, obs)
        assert reward >= 0.12  # direct evidence + possible strategy bonus

    def test_contextual_evidence_reward(self):
        action = IncidentTriageAction(action_type="query_logs", service="database")
        obs = _make_obs()
        reward = self.rubric.forward(action, obs)
        assert 0.0 < reward <= 0.10  # contextual + ruling out bonus

    def test_redundant_query_zero_reward(self):
        action = IncidentTriageAction(action_type="query_logs", service="payments-service")
        obs = _make_obs()
        self.rubric.forward(action, obs)  # first time
        reward = self.rubric.forward(action, obs)  # redundant
        assert reward == 0.0

    def test_repeated_redundant_negative(self):
        action1 = IncidentTriageAction(action_type="query_logs", service="payments-service")
        action2 = IncidentTriageAction(action_type="query_metrics", service="payments-service", metric="error_rate")
        action3 = IncidentTriageAction(action_type="check_deploys", service="payments-service")
        obs = _make_obs()
        self.rubric.forward(action1, obs)
        self.rubric.forward(action2, obs)
        self.rubric.forward(action3, obs)
        # 4th query to same service
        action4 = IncidentTriageAction(action_type="check_status", service="payments-service")
        reward = self.rubric.forward(action4, obs)
        assert reward < 0  # penalty for repeated

    def test_reset_clears_state(self):
        action = IncidentTriageAction(action_type="query_logs", service="payments-service")
        obs = _make_obs()
        self.rubric.forward(action, obs)
        self.rubric.reset()
        reward = self.rubric.forward(action, obs)
        assert reward >= 0.12  # should be treated as first query again


class TestDiagnosisScorer:
    def setup_method(self):
        self.scorer = DiagnosisScorer()

    def test_perfect_diagnosis(self):
        diagnosis = {
            "root_cause_service": "payments-service",
            "root_cause_category": "bad_deploy",
            "remediation": "rollback_deploy",
        }
        state = IncidentTriageState(
            task_id="easy",
            step_count=5,
            steps_remaining=5,
            services_investigated=["payments-service"],
            evidence_collected={"payments-service": ["query_logs", "check_deploys"]},
        )
        result = self.scorer.score(EASY_GROUND_TRUTH, diagnosis, state, step_budget=10)
        assert result["score"] >= 0.7  # high score for perfect diagnosis

    def test_wrong_service_zero_service_score(self):
        diagnosis = {
            "root_cause_service": "database",
            "root_cause_category": "bad_deploy",
            "remediation": "rollback_deploy",
        }
        state = IncidentTriageState(
            task_id="easy", step_count=5, steps_remaining=5,
            services_investigated=["database"],
        )
        result = self.scorer.score(EASY_GROUND_TRUTH, diagnosis, state, step_budget=10)
        assert result["breakdown"]["root_cause_service"] == 0.0

    def test_partial_credit_same_family(self):
        diagnosis = {
            "root_cause_service": "payments-service",
            "root_cause_category": "config_change",  # same family as bad_deploy
            "remediation": "rollback_deploy",
        }
        state = IncidentTriageState(
            task_id="easy", step_count=5, steps_remaining=5,
            services_investigated=["payments-service"],
        )
        result = self.scorer.score(EASY_GROUND_TRUTH, diagnosis, state, step_budget=10)
        assert result["breakdown"]["root_cause_category"] == 0.07  # partial credit

    def test_shotgun_penalty(self):
        diagnosis = {
            "root_cause_service": "payments-service",
            "root_cause_category": "bad_deploy",
            "remediation": "rollback_deploy",
        }
        state = IncidentTriageState(
            task_id="easy", step_count=2, steps_remaining=8,  # submitted at step 2
            services_investigated=["payments-service"],
        )
        result = self.scorer.score(EASY_GROUND_TRUTH, diagnosis, state, step_budget=10)
        assert result["breakdown"]["penalties"] <= -0.15

    def test_no_diagnosis_returns_zero(self):
        result = self.scorer.score(EASY_GROUND_TRUTH, None, IncidentTriageState(), step_budget=10)
        assert result["score"] == 0.0

    def test_failure_families_defined(self):
        assert FAILURE_FAMILIES["bad_deploy"] == "change-induced"
        assert FAILURE_FAMILIES["config_change"] == "change-induced"
        assert FAILURE_FAMILIES["resource_exhaustion"] == "capacity-related"
        assert FAILURE_FAMILIES["traffic_spike"] == "capacity-related"
        assert FAILURE_FAMILIES["dependency_failure"] == "infrastructure"
        assert FAILURE_FAMILIES["data_corruption"] == "infrastructure"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/test_grader.py -v
```

- [ ] **Step 3: Implement grader.py**

Implement `InvestigationRubric(Rubric)` and `DiagnosisScorer` following the reward function spec in the design doc. Key implementation details:

**InvestigationRubric:**
- `__init__(evidence_labels, ground_truth, services)`: **must call `super().__init__()` first** (the Rubric base sets `_rubric_children`, `_forward_hooks`, `_forward_pre_hooks`, `last_score` via `object.__setattr__` — without this call, `__call__` will crash). Then stores scenario data, initializes tracking dicts
- Internal state: `_queried_pairs` (set of (service, action_type)), `_service_query_counts` (Counter), `_has_causal_evidence` (bool), `_investigated_anomalous` (set)
- `forward(action, observation)`: sum of information_gain + strategy + red_herring_resistance
- `reset()`: clear all internal state

**DiagnosisScorer:**
- `score(ground_truth, diagnosis, episode_state, step_budget)`: returns dict with `score` and `breakdown`
- Uses `FAILURE_FAMILIES`, `PARTIAL_REMEDIATIONS` dicts
- Implements all penalty logic

**Constants:**
- `FAILURE_FAMILIES = {"bad_deploy": "change-induced", "config_change": "change-induced", "resource_exhaustion": "capacity-related", "traffic_spike": "capacity-related", "dependency_failure": "infrastructure", "data_corruption": "infrastructure"}`
- `PARTIAL_REMEDIATIONS = {("restart_service", "bad_deploy"): True, ("scale_up", "traffic_spike"): True, ("restart_service", "resource_exhaustion"): True, ("failover_to_backup", "dependency_failure"): True}`

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/test_grader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/grader.py tests/test_grader.py
git commit -m "feat: add InvestigationRubric and DiagnosisScorer with full reward logic"
```

---

### Task 9: Environment

**Files:**
- Create: `incident-triage-env/server/environment.py`
- Create: `incident-triage-env/tests/test_environment.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_environment.py
import pytest
from models import IncidentTriageAction, RootCauseCategory, Remediation
from server.environment import IncidentTriageEnv


@pytest.fixture
def env():
    e = IncidentTriageEnv()
    return e


def test_reset_returns_observation(env):
    obs = env.reset(task_id="easy_single_service_failure")
    assert obs.alert.service == "payments-service"
    assert obs.result is None
    assert obs.steps_remaining == 10
    assert obs.done is False
    assert obs.services_investigated == []


def test_step_query_logs(env):
    env.reset(task_id="easy_single_service_failure")
    action = IncidentTriageAction(action_type="query_logs", service="payments-service")
    obs = env.step(action)
    assert obs.result is not None
    assert obs.result.action_type == "query_logs"
    assert len(obs.result.data) > 0
    assert obs.steps_remaining == 9
    assert "payments-service" in obs.services_investigated
    assert obs.reward is not None


def test_step_submit_diagnosis(env):
    env.reset(task_id="easy_single_service_failure")
    action = IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    )
    obs = env.step(action)
    assert obs.done is True


def test_step_budget_exhaustion(env):
    env.reset(task_id="easy_single_service_failure")
    # Take 10 steps without submitting
    for i in range(10):
        action = IncidentTriageAction(action_type="check_status", service="payments-service")
        obs = env.step(action)
    assert obs.done is True
    assert obs.steps_remaining == 0


def test_state_property(env):
    env.reset(task_id="easy_single_service_failure")
    action = IncidentTriageAction(action_type="query_logs", service="payments-service")
    env.step(action)
    state = env.state
    assert state.task_id == "easy_single_service_failure"
    assert state.step_count == 1
    assert state.steps_remaining == 9
    assert "payments-service" in state.services_investigated


def test_reset_clears_state(env):
    env.reset(task_id="easy_single_service_failure")
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.reset(task_id="easy_single_service_failure")
    state = env.state
    assert state.step_count == 0
    assert state.services_investigated == []


def test_episode_stored_on_completion(env):
    from server.episode_store import get_episode
    obs = env.reset(task_id="easy_single_service_failure")
    episode_id = env.state.episode_id
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.step(IncidentTriageAction(action_type="query_logs", service="payments-service"))
    env.step(IncidentTriageAction(
        action_type="submit_diagnosis",
        root_cause_service="payments-service",
        root_cause_category=RootCauseCategory.BAD_DEPLOY,
        remediation=Remediation.ROLLBACK_DEPLOY,
    ))
    result = get_episode(episode_id)
    assert result is not None
    assert result["task_id"] == "easy_single_service_failure"
    assert "grader_score" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/test_environment.py -v
```

- [ ] **Step 3: Implement environment.py**

Key implementation:
- `IncidentTriageEnv(Environment[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState])`
- `__init__`: create ScenarioLoader, DiagnosisScorer (stored as `self.diagnosis_scorer`)
- `reset(seed, episode_id, task_id, **kwargs)`: load scenario, create InvestigationRubric with scenario data, assign `self.rubric = InvestigationRubric(...)`, call `self._reset_rubric()`, init state (set `self._state.step_count = 0`, `self._state.steps_remaining = step_budget`), return initial observation
- `step(action, ...)`: **increment `self._state.step_count`** and **decrement `self._state.steps_remaining`** (the base `State` class provides `step_count` — this must be updated). Dispatch action to ScenarioLoader, track state, compute reward via `self._apply_rubric()`, handle submit_diagnosis (call DiagnosisScorer, store in episode_store), handle budget exhaustion
- `state` property: return `IncidentTriageState` from internal state

**Important**: The rubric is re-created on each `reset()` because it depends on the scenario's evidence_labels and ground_truth. Assign `self.rubric = InvestigationRubric(...)` in reset().

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/test_environment.py -v
```

- [ ] **Step 5: Commit**

```bash
git add server/environment.py tests/test_environment.py
git commit -m "feat: add IncidentTriageEnv with full episode lifecycle"
```

---

### Task 10: FastAPI App

**Files:**
- Create: `incident-triage-env/server/app.py`

- [ ] **Step 1: Implement app.py**

```python
# server/app.py
"""FastAPI application for the Incident Triage Environment."""
from models import IncidentTriageAction, IncidentTriageObservation
from server.environment import IncidentTriageEnv
from server.scenario_loader import ScenarioLoader
from server.episode_store import get_episode
from server.grader import DiagnosisScorer

from openenv.core.env_server import create_app
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
import os

# Create the OpenEnv app
app = create_app(
    IncidentTriageEnv,
    IncidentTriageAction,
    IncidentTriageObservation,
    env_name="incident_triage_env",
)

# Shared instances for custom endpoints
_loader = ScenarioLoader()
_scorer = DiagnosisScorer()


# --- Custom endpoint models ---

class GraderRequest(BaseModel):
    episode_id: str

class GraderResponse(BaseModel):
    task_id: str
    score: float
    breakdown: dict
    diagnosis_submitted: bool

class BaselineResponse(BaseModel):
    scores: dict
    model: str
    total_steps_used: dict


# --- Custom endpoints ---

@app.get("/tasks")
def get_tasks():
    tasks = _loader.list_tasks()
    schema = IncidentTriageAction.model_json_schema()
    return {"tasks": tasks, "action_schema": schema}


@app.post("/grader")
def grade_episode(req: GraderRequest):
    episode = get_episode(req.episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode {req.episode_id} not found")
    return GraderResponse(
        task_id=episode["task_id"],
        score=episode["grader_score"]["score"] if episode.get("grader_score") else 0.0,
        breakdown=episode["grader_score"]["breakdown"] if episode.get("grader_score") else {},
        diagnosis_submitted=episode.get("diagnosis_submitted", False),
    )


@app.post("/baseline")
async def run_baseline():
    # Import here to avoid requiring google-genai/openai for server startup
    from scripts.baseline_inference import get_provider, run_baseline_all_tasks
    try:
        get_provider()  # Validate that an API key is configured
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    results = await run_baseline_all_tasks(base_url="http://localhost:8000")
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Test server starts**

```bash
cd incident-triage-env
PYTHONPATH=.:../OpenEnv/src uvicorn server.app:app --host 0.0.0.0 --port 8000 &
sleep 3
curl http://localhost:8000/health
curl http://localhost:8000/tasks
kill %1
```
Expected: `/health` returns `{"status": "healthy"}`, `/tasks` returns 3 tasks.

- [ ] **Step 3: Test WebSocket flow manually**

```bash
PYTHONPATH=.:../OpenEnv/src python -c "
import asyncio, json, websockets

async def test():
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        await ws.send(json.dumps({'type': 'reset', 'data': {'task_id': 'easy_single_service_failure'}}))
        resp = json.loads(await ws.recv())
        print('Reset:', resp['type'])
        await ws.send(json.dumps({'type': 'step', 'data': {'action_type': 'query_logs', 'service': 'payments-service'}}))
        resp = json.loads(await ws.recv())
        print('Step:', resp['type'], 'done:', resp['data'].get('done'))
        await ws.send(json.dumps({'type': 'state'}))
        resp = json.loads(await ws.recv())
        print('State:', resp['type'], 'step_count:', resp['data'].get('step_count'))

asyncio.run(test())
"
```

- [ ] **Step 4: Commit**

```bash
git add server/app.py
git commit -m "feat: add FastAPI app with create_app() and custom endpoints"
```

---

### Task 11: Client

**Files:**
- Create: `incident-triage-env/client.py`

- [ ] **Step 1: Implement client.py**

```python
# client.py
"""Client for the Incident Triage Environment."""
from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import IncidentTriageAction, IncidentTriageObservation, IncidentTriageState, AlertInfo, ActionResult


class IncidentTriageClient(EnvClient[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]):

    def _step_payload(self, action: IncidentTriageAction) -> dict:
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: dict) -> StepResult[IncidentTriageObservation]:
        obs_data = payload["observation"]
        obs = IncidentTriageObservation(
            alert=AlertInfo(**obs_data["alert"]),
            result=ActionResult(**obs_data["result"]) if obs_data.get("result") else None,
            steps_remaining=obs_data["steps_remaining"],
            services_investigated=obs_data.get("services_investigated", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> IncidentTriageState:
        return IncidentTriageState(**payload)
```

- [ ] **Step 2: Commit**

```bash
git add client.py
git commit -m "feat: add IncidentTriageClient EnvClient subclass"
```

---

### Task 12: Baseline Inference Script

**Files:**
- Create: `incident-triage-env/scripts/baseline_inference.py`

- [ ] **Step 1: Implement baseline_inference.py**

The script uses a provider abstraction to support multiple free LLM APIs:

**Provider priority (first available wins):**
1. **Gemini** (`GEMINI_API_KEY`) — uses `google-genai` SDK with `response_mime_type="application/json"` for native structured output. Model: `gemini-2.0-flash`.
2. **OpenRouter** (`OPENROUTER_API_KEY`) — uses the `openai` SDK with `base_url="https://openrouter.ai/api/v1"`. Model: `qwen/qwen3.6-plus:free` or `meta-llama/llama-3-70b-instruct` (free). Use `response_format={"type": "json_object"}`.
3. Raise error if neither key is set.

**Provider abstraction:**
```python
class LLMProvider:
    """Abstract interface for LLM providers."""
    async def generate(self, messages: list[dict], system: str) -> str:
        """Send messages, return raw text response."""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError

class GeminiProvider(LLMProvider):
    """Google Gemini via google-genai SDK."""
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def generate(self, messages, system):
        # Use generate_content with response_mime_type="application/json"
        ...

class OpenRouterProvider(LLMProvider):
    """OpenRouter via OpenAI-compatible API (free models)."""
    def __init__(self, api_key: str, model: str = "qwen/qwen3.6-plus:free"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model

    async def generate(self, messages, system):
        # Use chat.completions.create with response_format={"type": "json_object"}
        ...

def get_provider() -> LLMProvider:
    if key := os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider(key)
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterProvider(key)
    raise RuntimeError("Set GEMINI_API_KEY or OPENROUTER_API_KEY")
```

The rest of the script logic is the same:
1. Connect to environment via WebSocket for each task
2. Use the provider to generate actions as JSON
3. Parse JSON actions from LLM response (handle markdown code blocks)
4. Run until episode done or max steps
5. Call `/grader` endpoint for final score
6. Print results
- Expose `run_baseline_all_tasks(base_url)` as an async function for the `/baseline` endpoint

- [ ] **Step 2: Test script runs (requires API key — get free Gemini key from https://aistudio.google.com)**

```bash
PYTHONPATH=.:../OpenEnv/src GEMINI_API_KEY=$GEMINI_API_KEY python scripts/baseline_inference.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/baseline_inference.py
git commit -m "feat: add baseline inference script using Gemini/OpenRouter (free APIs)"
```

---

### Task 13: Dockerfile

**Files:**
- Create: `incident-triage-env/Dockerfile`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install git for openenv package installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install the OpenEnv package and project dependencies
RUN pip install --no-cache-dir -e ".[baseline]"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Note: The openenv package installation depends on whether it's on PyPI or needs to be installed from the local `../OpenEnv` directory. If not on PyPI, copy the OpenEnv source into the Docker build context or install from git.

- [ ] **Step 2: Build and test Docker image**

```bash
docker build -t incident-triage-env .
docker run -p 8000:8000 incident-triage-env &
sleep 5
curl http://localhost:8000/health
curl http://localhost:8000/tasks
docker stop $(docker ps -q --filter ancestor=incident-triage-env)
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for containerized deployment"
```

---

### Task 14: README

**Files:**
- Create: `incident-triage-env/README.md`

- [ ] **Step 1: Write README with HF Space frontmatter**

Must include:
- HF Space YAML frontmatter (title, emoji, sdk: docker, app_port: 8000, tags: [openenv])
- Environment description and motivation
- Action space table (7 action types with parameters)
- Observation space definition
- Task descriptions (3 tasks with difficulty, step budget, scenario summary)
- Reward function summary
- Setup instructions (pip install, docker build/run)
- Usage instructions (WebSocket connection, reset, step, grader)
- Baseline scores (fill in after running baseline)
- API endpoints (/tasks, /grader, /baseline, /ws)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with environment docs and HF Space metadata"
```

---

### Task 15: Validation & Final Testing

**Files:** None new — this is integration testing.

- [ ] **Step 1: Run all unit tests**

```bash
PYTHONPATH=.:../OpenEnv/src pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 2: Start server and test end-to-end WebSocket flow**

```bash
PYTHONPATH=.:../OpenEnv/src uvicorn server.app:app --port 8000 &
sleep 3
```

Test each task via WebSocket: reset → step → ... → submit_diagnosis → grader.

- [ ] **Step 3: Test all custom endpoints**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/tasks
# (grader and baseline tested after WebSocket episode)
```

- [ ] **Step 4: Run baseline inference script**

```bash
PYTHONPATH=.:../OpenEnv/src GEMINI_API_KEY=$GEMINI_API_KEY python scripts/baseline_inference.py
```
Record scores, update README.

- [ ] **Step 5: Docker build and test**

```bash
docker build -t incident-triage-env .
docker run -p 8000:8000 incident-triage-env
# Test same endpoints against Docker
```

- [ ] **Step 6: Run openenv validate (if available)**

```bash
PYTHONPATH=.:../OpenEnv/src openenv validate --url http://localhost:8000
```

- [ ] **Step 7: Verify determinism — run same task twice, confirm same grader score**

Run the easy task twice with the same actions, verify the grader returns identical scores.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "chore: final validation — all tests pass, Docker works, baseline recorded (Gemini free tier)"
```

---

### Task 16: Deploy to Hugging Face Spaces

- [ ] **Step 1: Push to HF Spaces**

```bash
openenv push --repo-id <username>/incident-triage-env
```

Or manually push via `huggingface_hub` CLI.

- [ ] **Step 2: Verify Space deploys**

Check that the HF Space URL returns 200 on `/health` and responds to `/tasks`.

- [ ] **Step 3: Run baseline against deployed Space**

```bash
GEMINI_API_KEY=$GEMINI_API_KEY python scripts/baseline_inference.py --url https://<space-url>
```

- [ ] **Step 4: Pre-submission checklist**

Verify ALL of these pass:
- [ ] HF Space deploys and returns 200 + responds to `reset()`
- [ ] `openenv validate` passes
- [ ] Dockerfile builds
- [ ] Baseline script runs without error and produces scores
- [ ] 3+ tasks with graders, all scores in 0.0–1.0 range
- [ ] `/baseline` endpoint works
- [ ] `/grader` endpoint works
- [ ] `/tasks` endpoint works
