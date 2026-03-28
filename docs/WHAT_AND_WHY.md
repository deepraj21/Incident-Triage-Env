# Incident Triage Environment — What We're Building & Why

## The One-Liner

An RL training environment where AI agents learn to diagnose production incidents by investigating microservices — querying logs, metrics, deploys, code, and service dependencies — then submitting a root cause diagnosis with remediation.

## The Problem We're Solving

Every software company with production systems has oncall engineers who get paged at 3am. When an alert fires, the engineer must:

1. Read the alert and understand what's broken
2. Investigate systematically — check logs, metrics, recent deploys, service dependencies
3. Distinguish symptoms from root causes (the hardest part)
4. Identify the root cause service and failure category
5. Apply the correct remediation (rollback, scale up, fix config, etc.)

This process is:
- **Time-critical** — every minute of downtime costs money
- **Skill-intensive** — junior engineers often waste time chasing symptoms or red herrings
- **Repetitive** — the same patterns recur (bad deploys, config changes, cascading failures)
- **Perfect for RL** — there's a clear action loop, measurable outcomes, and room for the agent to develop a strategy

## What We're Building

An OpenEnv-compliant environment (`incident-triage-env`) that simulates this workflow. The environment:

- Presents the agent with a production alert (e.g., "HTTP 500 spike on api-gateway")
- Provides 7 investigation actions: query logs, query metrics, check deploys, trace dependencies, check status, inspect code, submit diagnosis
- Simulates a microservices architecture with 3–8 services, realistic log/metric data, and dependency graphs
- Includes red herrings and misleading signals that test the agent's ability to distinguish symptoms from causes
- Rewards systematic investigation, not just correct answers — following the dependency graph, cross-referencing data sources, and ruling out possibilities are all rewarded
- Ships 3 tasks with increasing difficulty:
  - **Easy**: Single service failure, obvious logs, no misdirection
  - **Medium**: Cascading failure across 3 services, 1 red herring, requires correlation
  - **Hard**: Multi-signal cascade across 4+ services, 2 red herrings, requires reading source code

## Why This Matters (Real-World Utility)

### For AI Agent Training
- **Dense reward signal**: Agents get feedback every step (not just pass/fail at the end), enabling faster learning
- **Strategy learning**: The environment rewards investigation patterns (following dependencies, cross-referencing), not just outcomes — teaching agents HOW to debug, not just WHAT the answer is
- **Difficulty progression**: Easy tasks let agents learn basic investigation; hard tasks require multi-hop reasoning and code comprehension that challenge even frontier models

### For the Industry
- **Incident response automation**: Companies like PagerDuty, Datadog, and Grafana are all building AI-assisted incident response. An RL training environment for this is directly useful.
- **Oncall training**: New engineers could practice against the environment before going on real oncall
- **Evaluation benchmark**: Teams building AI incident responders need standardized benchmarks — this fills that gap

### Compared to Existing Environments
- No existing OpenEnv environment covers incident triage/debugging
- Existing code-review and data-cleaning environments are more common submissions
- The red herring mechanics and investigation strategy rewards are novel reward design patterns

## How It Maps to OpenEnv

| OpenEnv Concept | Our Implementation |
|----------------|-------------------|
| `reset(task_id=...)` | Loads a scenario fixture, presents the initial alert |
| `step(action)` | Agent queries logs/metrics/deploys/code, receives data + reward |
| `state()` | Steps remaining, services investigated, evidence collected |
| `Observation` | Alert + action result (text data) + steps remaining |
| `Action` | Polymorphic model with 7 action types |
| `Rubric` | 5-component reward: information gain, strategy, red herring resistance, diagnosis accuracy, efficiency |
| `Grader` | Terminal score 0.0–1.0 based on diagnosis correctness + evidence quality |

## Target Hackathon Scores

| Criterion (Weight) | Target | Rationale |
|---------------------|--------|-----------|
| Real-world utility (30%) | 26–30 | Incident triage is a genuine, daily task with clear industry demand |
| Task & grader quality (25%) | 20–25 | 3 tasks, deterministic graders, meaningful difficulty progression |
| Environment design (20%) | 16–20 | Dense reward signal, clean state management, structured action space |
| Code quality & spec (15%) | 12–15 | Full spec compliance, typed models, tests, Docker |
| Creativity & novelty (10%) | 8–10 | Novel domain, red herring mechanics, strategy rewards |
| **Total** | **82–100** | |

## Tech Stack

- **Python 3.10+** with FastAPI
- **OpenEnv SDK** (`openenv` package) for Environment base class, Rubric, create_app()
- **Pydantic v2** for typed models
- **OpenAI API client** for baseline inference
- **Docker** for containerized deployment
- **Hugging Face Spaces** for hosting
