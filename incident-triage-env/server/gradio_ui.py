"""Interactive Gradio UI for the Incident Triage Environment.

Provides a presentable, demo-ready interface for HuggingFace Spaces.
"""

import json
import uuid
from typing import Optional

import gradio as gr

from models import (
    IncidentTriageAction,
    RootCauseCategory,
    Remediation,
)
from server.environment import IncidentTriageEnv
from server.scenario_loader import ScenarioLoader


# Shared environment instance for the UI
_env = IncidentTriageEnv()
_loader = ScenarioLoader()

# Task metadata
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
    """Get list of services for a task."""
    try:
        scenario = _loader.get_scenario(task_id)
        return list(scenario["services"].keys())
    except Exception:
        return []


def reset_environment(task_id: str):
    """Reset the environment with a new task."""
    if not task_id:
        return (
            "Select a task first.",
            gr.update(choices=[]),
            gr.update(choices=[]),
            "",
            [],
            "",
            gr.update(visible=False),
        )

    obs = _env.reset(task_id=task_id, episode_id=str(uuid.uuid4()))
    alert = obs.alert

    services = get_available_services(task_id)
    task_info = TASKS.get(task_id, {})

    alert_html = f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 20px; border-left: 4px solid #e74c3c; margin-bottom: 10px;">
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="background: #e74c3c; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;">
            {alert.severity.upper()}
        </span>
        <span style="color: #e2e8f0; font-weight: 600; font-size: 16px;">{alert.service}</span>
    </div>
    <p style="color: #cbd5e1; margin: 4px 0; font-size: 14px;">{alert.message}</p>
    <p style="color: #64748b; margin: 4px 0; font-size: 12px;">Timestamp: {alert.timestamp}</p>
    <div style="display: flex; gap: 16px; margin-top: 12px; color: #94a3b8; font-size: 13px;">
        <span>Budget: <strong style="color: #60a5fa;">{obs.steps_remaining} steps</strong></span>
        <span>Services: <strong style="color: #60a5fa;">{task_info.get('services', '?')}</strong></span>
        <span>Difficulty: <strong style="color: #fbbf24;">{task_info.get('badge', '?')}</strong></span>
    </div>
</div>"""

    status_text = f"Episode started. You have {obs.steps_remaining} steps to investigate and submit a diagnosis."

    return (
        alert_html,
        gr.update(choices=services, value=services[0] if services else None),
        gr.update(choices=services, value=None),
        status_text,
        [],  # Clear investigation log
        "",  # Clear result
        gr.update(visible=True),
    )


def take_action(
    action_type: str,
    service: str,
    severity: str,
    keyword: str,
    metric: str,
    root_cause_service: str,
    root_cause_category: str,
    remediation: str,
    investigation_log: list,
):
    """Execute an action in the environment."""
    if not action_type:
        return investigation_log, "Select an action type.", "", ""

    try:
        action_kwargs = {"action_type": action_type}

        if action_type == "submit_diagnosis":
            if not root_cause_service or not root_cause_category or not remediation:
                return investigation_log, "Fill in all diagnosis fields (service, category, remediation).", "", ""
            action_kwargs["root_cause_service"] = root_cause_service
            action_kwargs["root_cause_category"] = root_cause_category
            action_kwargs["remediation"] = remediation
        else:
            if not service:
                return investigation_log, "Select a service to investigate.", "", ""
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

        # Build log entry
        step_num = len(investigation_log) + 1
        reward = obs.reward if obs.reward is not None else 0.0
        reward_color = "#22c55e" if reward > 0 else "#ef4444" if reward < 0 else "#94a3b8"

        if action_type == "submit_diagnosis":
            action_desc = f"submit_diagnosis({root_cause_service}, {root_cause_category}, {remediation})"
        else:
            action_desc = f"{action_type}({service}"
            if action_type == "query_logs" and severity:
                action_desc += f", severity={severity}"
            if action_type == "query_logs" and keyword:
                action_desc += f", keyword={keyword}"
            if action_type == "query_metrics" and metric:
                action_desc += f", metric={metric}"
            action_desc += ")"

        log_entry = [
            str(step_num),
            action_desc,
            f"{reward:+.3f}",
            str(obs.steps_remaining),
        ]
        investigation_log = investigation_log + [log_entry]

        # Format result
        result_data = ""
        if obs.result:
            result_data = obs.result.data

        # Status update
        if obs.done:
            # Get grader score
            state = _env.state
            from server.episode_store import get_episode
            episode = get_episode(state.episode_id)
            if episode and episode.get("grader_score"):
                gs = episode["grader_score"]
                score = gs["score"]
                bd = gs.get("breakdown", {})
                score_color = "#22c55e" if score >= 0.6 else "#fbbf24" if score >= 0.3 else "#ef4444"

                status = f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 20px; border: 1px solid {score_color}40;">
    <h3 style="color: {score_color}; margin: 0 0 12px 0; font-size: 24px;">
        Final Score: {score:.2f} / 1.00
    </h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
        <div style="background: #0f172a; padding: 8px; border-radius: 8px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px;">Root Cause Service</div>
            <div style="color: #e2e8f0; font-weight: 600;">{bd.get('root_cause_service', 0):.2f}</div>
        </div>
        <div style="background: #0f172a; padding: 8px; border-radius: 8px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px;">Category</div>
            <div style="color: #e2e8f0; font-weight: 600;">{bd.get('root_cause_category', 0):.2f}</div>
        </div>
        <div style="background: #0f172a; padding: 8px; border-radius: 8px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px;">Remediation</div>
            <div style="color: #e2e8f0; font-weight: 600;">{bd.get('remediation', 0):.2f}</div>
        </div>
        <div style="background: #0f172a; padding: 8px; border-radius: 8px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px;">Evidence Quality</div>
            <div style="color: #e2e8f0; font-weight: 600;">{bd.get('evidence_quality', 0):.2f}</div>
        </div>
        <div style="background: #0f172a; padding: 8px; border-radius: 8px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px;">Efficiency</div>
            <div style="color: #e2e8f0; font-weight: 600;">{bd.get('efficiency', 0):.2f}</div>
        </div>
        <div style="background: #0f172a; padding: 8px; border-radius: 8px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px;">Penalties</div>
            <div style="color: #ef4444; font-weight: 600;">{bd.get('penalties', 0):.2f}</div>
        </div>
    </div>
</div>"""
            else:
                status = "Episode ended. Budget exhausted without diagnosis."
        else:
            status = f"Steps remaining: {obs.steps_remaining}. Services investigated: {', '.join(obs.services_investigated)}"

        return investigation_log, status, result_data, ""

    except Exception as e:
        return investigation_log, f"Error: {str(e)}", "", ""


def build_gradio_app() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="Incident Triage Environment",
    ) as demo:
        # Header
        gr.HTML("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 style="margin: 0; font-size: 28px; color: #e2e8f0;">
        Incident Triage Environment
    </h1>
    <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 15px;">
        An OpenEnv RL environment for training AI agents to diagnose production incidents.
        <br/>Investigate logs, metrics, deploys, and dependencies to find the root cause.
    </p>
</div>
        """)

        # State
        investigation_log_state = gr.State([])

        with gr.Row():
            # Left panel: Task selection + Actions
            with gr.Column(scale=1):
                gr.Markdown("### Select Task")
                task_dropdown = gr.Dropdown(
                    choices=[(v["name"], k) for k, v in TASKS.items()],
                    label="Incident Scenario",
                    value=None,
                    info="Choose a difficulty level to start investigating",
                )

                # Task descriptions
                gr.HTML("""
<div style="font-size: 12px; color: #94a3b8; padding: 8px; background: #0f172a; border-radius: 8px; margin-top: 4px;">
    <div style="margin-bottom: 6px;"><span style="color: #22c55e;">Easy</span> — Single service, clear evidence trail (10 steps)</div>
    <div style="margin-bottom: 6px;"><span style="color: #fbbf24;">Medium</span> — Cascading failure across 5 services (15 steps)</div>
    <div><span style="color: #ef4444;">Hard</span> — 8 services, red herrings, 4-hop chain (20 steps)</div>
</div>
                """)

                reset_btn = gr.Button("Start Investigation", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Investigation Actions")

                action_type = gr.Dropdown(
                    choices=ACTION_TYPES,
                    label="Action Type",
                    value="check_status",
                    info="What investigation step to take",
                )

                service = gr.Dropdown(
                    choices=[],
                    label="Service",
                    info="Target service to investigate",
                )

                with gr.Accordion("Query Options", open=False):
                    severity = gr.Dropdown(
                        choices=SEVERITIES,
                        label="Log Severity Filter",
                        value="",
                        info="For query_logs: filter by severity",
                    )
                    keyword = gr.Textbox(
                        label="Log Keyword Filter",
                        placeholder="e.g., timeout, OOM, deploy",
                        info="For query_logs: filter by keyword",
                    )
                    metric = gr.Dropdown(
                        choices=METRICS,
                        label="Metric",
                        value="error_rate",
                        info="For query_metrics: which metric to check",
                    )

                with gr.Accordion("Submit Diagnosis", open=False):
                    root_cause_service = gr.Dropdown(
                        choices=[],
                        label="Root Cause Service",
                        info="Which service is the root cause?",
                    )
                    root_cause_category = gr.Dropdown(
                        choices=CATEGORIES,
                        label="Root Cause Category",
                        info="What type of failure?",
                    )
                    remediation_input = gr.Dropdown(
                        choices=REMEDIATIONS,
                        label="Remediation",
                        info="What fix should be applied?",
                    )

                step_btn = gr.Button("Execute Action", variant="secondary", size="lg")

            # Right panel: Results
            with gr.Column(scale=2):
                alert_display = gr.HTML(
                    value="""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 40px; text-align: center;">
    <h3 style="color: #64748b; margin: 0;">Select a task and click "Start Investigation" to begin</h3>
    <p style="color: #475569; margin: 8px 0 0 0; font-size: 13px;">
        You'll receive an incident alert and must investigate to find the root cause
    </p>
</div>""",
                    elem_id="alert-display",
                )

                status_display = gr.Markdown("", label="Status")

                investigation_panel = gr.Column(visible=False)
                with investigation_panel:
                    gr.Markdown("### Investigation Log")
                    log_table = gr.Dataframe(
                        headers=["Step", "Action", "Reward", "Remaining"],
                        datatype=["str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )

                    gr.Markdown("### Investigation Result")
                    result_display = gr.Textbox(
                        label="",
                        lines=12,
                        interactive=False,
                        elem_id="investigation-result",
                    )

        # About section
        with gr.Accordion("About This Environment", open=False):
            gr.Markdown("""
**Incident Triage Environment** is an [OpenEnv](https://github.com/openenv)-compliant RL environment that simulates production incident triage.

**How it works:**
1. An alert fires for a production service
2. You investigate by querying logs, metrics, deploys, dependencies, and code
3. Follow the evidence chain upstream to find the root cause
4. Submit a diagnosis with the root cause service, failure category, and remediation

**Scoring (0.0 - 1.0):**
- Root cause service identification (0.20)
- Failure category accuracy (0.15)
- Correct remediation (0.10)
- Evidence quality - how much of the causal chain you investigated (0.10)
- Efficiency - solving with fewer steps scores higher (0.15)
- Penalties for redundant queries, shotgun diagnoses, and red herring traps

**Action Space:** 7 action types (query_logs, query_metrics, check_deploys, trace_dependencies, check_status, inspect_code, submit_diagnosis)

**API:** Full OpenEnv spec with `step()`, `reset()`, `state()` + WebSocket support.
            """)

        # Wire up events
        dummy_textbox = gr.Textbox(visible=False)

        reset_btn.click(
            fn=reset_environment,
            inputs=[task_dropdown],
            outputs=[alert_display, service, root_cause_service, status_display, investigation_log_state, result_display, investigation_panel],
        )

        step_btn.click(
            fn=take_action,
            inputs=[action_type, service, severity, keyword, metric, root_cause_service, root_cause_category, remediation_input, investigation_log_state],
            outputs=[investigation_log_state, status_display, result_display, dummy_textbox],
        )

        # Update log table when investigation_log changes
        investigation_log_state.change(
            fn=lambda x: x,
            inputs=[investigation_log_state],
            outputs=[log_table],
        )

    return demo
