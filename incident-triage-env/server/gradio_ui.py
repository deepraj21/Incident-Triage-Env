"""Interactive Gradio UI for the Incident Triage Environment.

Provides a presentable, demo-ready interface for HuggingFace Spaces.
"""

import json
import uuid

import gradio as gr

from models import (
    IncidentTriageAction,
    RootCauseCategory,
    Remediation,
)
from server.environment import IncidentTriageEnv
from server.scenario_loader import ScenarioLoader


_env = IncidentTriageEnv()
_loader = ScenarioLoader()

TASKS = {
    "easy_single_service_failure": {
        "name": "Easy: Single Service Failure",
        "desc": "Bad deploy causes NullPointerException in payments-service",
        "services": 3, "steps": 10, "badge": "Beginner"
    },
    "medium_cascading_dependency": {
        "name": "Medium: Cascading Dependency",
        "desc": "Config change drops DB connections, cascades through service chain",
        "services": 5, "steps": 15, "badge": "Intermediate"
    },
    "hard_multi_signal_cascade": {
        "name": "Hard: Multi-Signal Cascade",
        "desc": "Memory leak + red herrings across 8 services with 4-hop causal chain",
        "services": 8, "steps": 20, "badge": "Expert"
    },
}

ACTION_TYPES = [
    "query_logs", "query_metrics", "check_deploys",
    "trace_dependencies", "check_status", "inspect_code",
    "submit_diagnosis"
]
METRICS = ["latency_p99", "error_rate", "cpu", "memory", "connections"]
SEVERITIES = ["", "error", "warn", "info", "debug"]
CATEGORIES = [c.value for c in RootCauseCategory]
REMEDIATIONS = [r.value for r in Remediation]


def get_available_services(task_id: str) -> list[str]:
    try:
        scenario = _loader.get_scenario(task_id)
        return list(scenario["services"].keys())
    except Exception:
        return []


def _render_log_table(log_data: list) -> str:
    """Render the investigation log as a Markdown table."""
    if not log_data:
        return "| Step | Action | Reward | Remaining |\n|------|--------|--------|-----------|\n| | *No steps taken yet* | | |"
    lines = ["| Step | Action | Reward | Remaining |", "|------|--------|--------|-----------|"]
    for row in log_data:
        lines.append(f"| {row[0]} | `{row[1]}` | {row[2]} | {row[3]} |")
    return "\n".join(lines)


def reset_environment(task_id: str):
    """Returns: (alert_html, service, rcs, status, log_json, log_md, result)"""
    if not task_id:
        return (
            "<p style='color:#94a3b8;text-align:center;padding:40px;'>Select a task first.</p>",
            gr.update(choices=[]),
            gr.update(choices=[]),
            "",
            "[]",
            _render_log_table([]),
            "",
        )

    obs = _env.reset(task_id=task_id, episode_id=str(uuid.uuid4()))
    alert = obs.alert
    services = get_available_services(task_id)
    task_info = TASKS.get(task_id, {})

    alert_html = f"""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:12px;padding:20px;border-left:4px solid #e74c3c;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
<span style="background:#e74c3c;color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">{alert.severity.upper()}</span>
<span style="color:#e2e8f0;font-weight:600;font-size:16px;">{alert.service}</span>
</div>
<p style="color:#cbd5e1;margin:4px 0;font-size:14px;">{alert.message}</p>
<p style="color:#64748b;margin:4px 0;font-size:12px;">Timestamp: {alert.timestamp}</p>
<div style="display:flex;gap:16px;margin-top:12px;color:#94a3b8;font-size:13px;">
<span>Budget: <strong style="color:#60a5fa;">{obs.steps_remaining} steps</strong></span>
<span>Services: <strong style="color:#60a5fa;">{task_info.get('services', '?')}</strong></span>
<span>Difficulty: <strong style="color:#fbbf24;">{task_info.get('badge', '?')}</strong></span>
</div>
</div>"""

    status = f"Episode started. You have **{obs.steps_remaining}** steps. Use the actions on the left to investigate, then submit your diagnosis."

    return (
        alert_html,
        gr.update(choices=services, value=services[0] if services else None),
        gr.update(choices=services, value=None),
        status,
        "[]",
        _render_log_table([]),
        "",
    )


def take_action(
    action_type, service, severity, keyword, metric,
    root_cause_service, root_cause_category, remediation,
    log_json_str,
):
    """Execute action. Returns: (log_json, log_md, status, result)"""
    try:
        log_data = json.loads(log_json_str) if log_json_str else []
    except (json.JSONDecodeError, TypeError):
        log_data = []

    if not action_type:
        return json.dumps(log_data), _render_log_table(log_data), "Select an action type.", ""

    try:
        action_kwargs = {"action_type": action_type}

        if action_type == "submit_diagnosis":
            if not root_cause_service or not root_cause_category or not remediation:
                return json.dumps(log_data), _render_log_table(log_data), "Fill in all diagnosis fields (service, category, remediation).", ""
            action_kwargs["root_cause_service"] = root_cause_service
            action_kwargs["root_cause_category"] = root_cause_category
            action_kwargs["remediation"] = remediation
        else:
            if not service:
                return json.dumps(log_data), _render_log_table(log_data), "Select a service to investigate.", ""
            action_kwargs["service"] = service
            if action_type == "query_logs":
                if severity:
                    action_kwargs["severity"] = severity
                if keyword:
                    action_kwargs["keyword"] = keyword
            elif action_type == "query_metrics":
                action_kwargs["metric"] = metric or "error_rate"

        action = IncidentTriageAction(**action_kwargs)
        obs = _env.step(action)

        step_num = len(log_data) + 1
        reward = obs.reward if obs.reward is not None else 0.0

        if action_type == "submit_diagnosis":
            action_desc = f"submit_diagnosis({root_cause_service}, {root_cause_category}, {remediation})"
        else:
            action_desc = f"{action_type}({service})"

        log_data = log_data + [[str(step_num), action_desc, f"{reward:+.3f}", str(obs.steps_remaining)]]

        result_data = obs.result.data if obs.result else ""

        if obs.done:
            state = _env.state
            from server.episode_store import get_episode
            episode = get_episode(state.episode_id)
            if episode and episode.get("grader_score"):
                gs = episode["grader_score"]
                score = gs["score"]
                bd = gs.get("breakdown", {})
                score_emoji = "🟢" if score >= 0.6 else "🟡" if score >= 0.3 else "🔴"
                status = (
                    f"### {score_emoji} Final Score: {score:.2f} / 1.00\n\n"
                    f"| Component | Score |\n|---|---|\n"
                    f"| Root Cause Service | {bd.get('root_cause_service', 0):.2f} |\n"
                    f"| Root Cause Category | {bd.get('root_cause_category', 0):.2f} |\n"
                    f"| Remediation | {bd.get('remediation', 0):.2f} |\n"
                    f"| Evidence Quality | {bd.get('evidence_quality', 0):.2f} |\n"
                    f"| Efficiency | {bd.get('efficiency', 0):.2f} |\n"
                    f"| Penalties | {bd.get('penalties', 0):.2f} |\n\n"
                    f"*Episode complete. Select a task and click Start Investigation to try again.*"
                )
            else:
                status = "Episode ended. Budget exhausted without submitting a diagnosis."
        else:
            status = f"Steps remaining: **{obs.steps_remaining}**. Services investigated: {', '.join(obs.services_investigated)}"

        return json.dumps(log_data), _render_log_table(log_data), status, result_data

    except Exception as e:
        return json.dumps(log_data), _render_log_table(log_data), f"**Error:** {str(e)}", ""


def build_gradio_app() -> gr.Blocks:

    with gr.Blocks(title="Incident Triage Environment") as demo:

        gr.HTML("""<div style="text-align:center;padding:20px 0 10px 0;">
<h1 style="margin:0;font-size:28px;color:#e2e8f0;">Incident Triage Environment</h1>
<p style="color:#94a3b8;margin:8px 0 0 0;font-size:15px;">
An OpenEnv RL environment for training AI agents to diagnose production incidents.<br/>
Investigate logs, metrics, deploys, and dependencies to find the root cause.</p>
</div>""")

        # Hidden textbox for log state (JSON string - avoids Gradio Dataframe/State bugs)
        log_json = gr.Textbox(value="[]", visible=False, elem_id="log-json-state")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Select Task")
                task_dropdown = gr.Dropdown(
                    choices=[(v["name"], k) for k, v in TASKS.items()],
                    label="Incident Scenario",
                    value=None,
                    info="Choose a difficulty level to start investigating",
                )

                gr.HTML("""<div style="font-size:12px;color:#94a3b8;padding:8px;background:#0f172a;border-radius:8px;margin-top:4px;">
<div style="margin-bottom:6px;"><span style="color:#22c55e;">Easy</span> — Single service, clear evidence trail (10 steps)</div>
<div style="margin-bottom:6px;"><span style="color:#fbbf24;">Medium</span> — Cascading failure across 5 services (15 steps)</div>
<div><span style="color:#ef4444;">Hard</span> — 8 services, red herrings, 4-hop chain (20 steps)</div>
</div>""")

                reset_btn = gr.Button("Start Investigation", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Investigation Actions")

                action_type = gr.Dropdown(
                    choices=ACTION_TYPES, label="Action Type", value="check_status",
                    info="What investigation step to take",
                )
                service = gr.Dropdown(choices=[], label="Service", info="Target service to investigate")

                with gr.Accordion("Query Options", open=False):
                    severity = gr.Dropdown(choices=SEVERITIES, label="Log Severity Filter", value="",
                                           info="For query_logs: filter by severity")
                    keyword = gr.Textbox(label="Log Keyword Filter", placeholder="e.g., timeout, OOM, deploy",
                                         info="For query_logs: filter by keyword")
                    metric = gr.Dropdown(choices=METRICS, label="Metric", value="error_rate",
                                         info="For query_metrics: which metric to check")

                with gr.Accordion("Submit Diagnosis", open=False):
                    root_cause_service = gr.Dropdown(choices=[], label="Root Cause Service",
                                                      info="Which service is the root cause?")
                    root_cause_category = gr.Dropdown(choices=CATEGORIES, label="Root Cause Category",
                                                       info="What type of failure?")
                    remediation_input = gr.Dropdown(choices=REMEDIATIONS, label="Remediation",
                                                     info="What fix should be applied?")

                step_btn = gr.Button("Execute Action", variant="secondary", size="lg")

            with gr.Column(scale=2):
                alert_display = gr.HTML(
                    value="""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:12px;padding:40px;text-align:center;">
<h3 style="color:#64748b;margin:0;">Select a task and click "Start Investigation" to begin</h3>
<p style="color:#475569;margin:8px 0 0 0;font-size:13px;">
You'll receive an incident alert and must investigate to find the root cause</p></div>""",
                )

                status_display = gr.Markdown("")

                gr.Markdown("### Investigation Log")
                log_display = gr.Markdown(_render_log_table([]))

                gr.Markdown("### Investigation Result")
                result_display = gr.Textbox(label="", lines=10, interactive=False)

        gr.Markdown("---")
        with gr.Accordion("About This Environment", open=True):
            gr.Markdown("""
**Incident Triage Environment** is an [OpenEnv](https://github.com/openenv)-compliant RL environment that simulates **real-world production incident triage** -- the task every oncall engineer does daily.

#### Why This Matters
Production incident response is a high-stakes, time-critical task requiring structured reasoning under uncertainty. This environment captures that workflow as a trainable RL task with dense reward signals, making it immediately useful for evaluating and training AI agents on a genuine operational challenge.

#### How to Try It (Interactive Demo)
1. **Select a task** from the dropdown (Easy/Medium/Hard) and click **Start Investigation**
2. **Investigate** using the action controls on the left -- query logs, check metrics, trace dependencies, inspect code
3. **Follow the evidence chain** upstream: symptoms appear in downstream services, but root causes are upstream
4. **Submit your diagnosis** when ready: open the "Submit Diagnosis" accordion, select the root cause service, failure category, and remediation, then set Action Type to `submit_diagnosis` and click **Execute Action**
5. **See your score** broken down by accuracy, evidence quality, and efficiency

#### Tasks (Easy to Hard)

| Task | Services | Budget | Scenario | Challenge |
|------|----------|--------|----------|-----------|
| Easy | 3 | 10 steps | Bad deploy causes NullPointerException | Single service, clear evidence |
| Medium | 5 | 15 steps | Config change cascades through dependency chain | 3-hop causal chain, 1 red herring |
| Hard | 8 | 20 steps | Memory leak with multiple red herrings | 4-hop chain, multiple red herrings |

#### Reward Design (Dense Signals)

**Per-step rewards** provide learning signal throughout the episode, not just at the end:
- **Information gain** (+0.04 to +0.12): Higher for direct evidence, lower for contextual
- **Strategy bonuses** (+0.02 to +0.05): Following dependency graphs, cross-referencing, ruling out
- **Penalties** (-0.03 to -0.08): Redundant queries, chasing red herrings

**Terminal grading** (0.0 to 1.0) scores the final diagnosis:

| Component | Max | Description |
|-----------|-----|-------------|
| Root cause service | 0.20 | Exact match (0.10 if one hop away) |
| Root cause category | 0.15 | Exact match (0.07 if same failure family) |
| Remediation | 0.10 | Exact match (0.05 partial credit) |
| Evidence quality | 0.10 | % of causal chain investigated |
| Efficiency | 0.15 | Fewer steps = higher score |
| Penalties | -0.35 max | Shotgun diagnosis, circular investigation |

#### Baseline Scores

| Task | Score | Steps Used | Budget |
|------|-------|------------|--------|
| Easy | 0.56 | 5 | 10 |
| Medium | 0.54 | 9 | 15 |
| Hard | 0.66 | 10 | 20 |

*Scores from `google/gemini-2.0-flash-001` via OpenRouter. Average: 0.59. Stronger models achieve higher scores.*

#### Technical Details
- **OpenEnv spec**: Full `step()` / `reset()` / `state()` API + WebSocket + MCP endpoint
- **7 action types**: query_logs, query_metrics, check_deploys, trace_dependencies, check_status, inspect_code, submit_diagnosis
- **6 failure categories**: bad_deploy, resource_exhaustion, dependency_failure, config_change, traffic_spike, data_corruption
- **6 remediations**: rollback_deploy, scale_up, restart_service, fix_config, enable_rate_limiting, failover_to_backup
- **33 unit tests** covering environment, grader, and models
- **Dockerized** and deployed to HuggingFace Spaces

#### API Access
```python
import websockets, json

async with websockets.connect("wss://AbhishekMallick-incident-triage-env.hf.space/ws") as ws:
    await ws.send(json.dumps({"type": "reset", "data": {"task_id": "easy_single_service_failure"}}))
    obs = json.loads(await ws.recv())
```
            """)

        # Event wiring
        reset_btn.click(
            fn=reset_environment,
            inputs=[task_dropdown],
            outputs=[alert_display, service, root_cause_service, status_display, log_json, log_display, result_display],
        )

        step_btn.click(
            fn=take_action,
            inputs=[action_type, service, severity, keyword, metric,
                    root_cause_service, root_cause_category, remediation_input, log_json],
            outputs=[log_json, log_display, status_display, result_display],
        )

    return demo
