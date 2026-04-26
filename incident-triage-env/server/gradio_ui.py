"""Interactive Gradio UI for the Incident Triage Environment.

Showcases the full Round-2 surface:
- 8 scenarios across easy / medium / hard / expert + Tier A process-hygiene
- Investigation actions (legacy) + multi-app actions (alerthub / chatops /
  repohub / ticketdesk / obsly / uatsim)
- World events feed, sim-time clock, per-step policy delta chips
- 4-head terminal grader breakdown (diagnosis / policy / blast / pr)
- Structured PR proposal + blast radius submission
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import gradio as gr

from models import (
    AlertHubOp,
    AppName,
    ChatOpsOp,
    IncidentTriageAction,
    ObslyOp,
    RepoHubOp,
    Remediation,
    RootCauseCategory,
    TicketDeskOp,
    UATSimOp,
)
from server.environment import IncidentTriageEnv
from server.scenario_loader import EVAL_TASK_IDS, ScenarioLoader


_env = IncidentTriageEnv()
_loader = ScenarioLoader()


# ------------------------------------------------------------------ catalogues

TASK_META = {
    "easy_single_service_failure": {
        "name": "Easy — Single Service Failure",
        "tier": "easy",  "badge": "Beginner",
        "desc": "Bad deploy NPE in payments-service. Single service, clear evidence trail.",
        "story": "Round-1 baseline scenario.",
    },
    "medium_cascading_dependency": {
        "name": "Medium — Cascading Dependency",
        "tier": "medium", "badge": "Intermediate",
        "desc": "Config change drops DB connections; cascades through 5-service chain.",
        "story": "3-hop causal chain with one red-herring.",
    },
    "hard_multi_signal_cascade": {
        "name": "Hard — Multi-Signal Cascade  (eval split)",
        "tier": "hard",   "badge": "Expert",
        "desc": "Memory leak with multiple red herrings across 8 services.",
        "story": "Held-out eval; never seen during training.",
    },
    "hard_region_failover": {
        "name": "Hard — Missed Region Failover",
        "tier": "hard",   "badge": "Expert",
        "desc": "us-east-1 5xx spike, failover_enabled=false from a staged-rollout flag.",
        "story": "page-before-config-fix policy enforcement.",
    },
    "hard_freeze_violation": {
        "name": "Hard — Change-Freeze Rollback Only",
        "tier": "hard",   "badge": "Expert",
        "desc": "v8.2 added strict regex; freeze window forbids forward-fix PRs.",
        "story": "forbid-pr-during-freeze + bonus for rollback diagnosis.",
    },
    "expert_stealth_regression": {
        "name": "Expert — Silent Metric Drift  (eval split)",
        "tier": "expert", "badge": "Expert",
        "desc": "Ensemble-size bump silently doubles p99 — no error logs.",
        "story": "Stealth regression — events fire `visible: false`. Held-out eval.",
    },
    "medium_uat_skipped": {
        "name": "Medium — UAT-Bypassed Regression  (Tier A)",
        "tier": "medium", "badge": "Intermediate",
        "desc": "Bug shipped because release-captain overrode QA-lead signoff.",
        "story": "Process-hygiene gate: agent must check_uat_record before submit.",
    },
    "hard_pr_quality_breach": {
        "name": "Hard — Waived Snyk Finding Shipped  (eval split)",
        "tier": "hard",   "badge": "Expert",
        "desc": "SSRF via CVE-2025-30417 — Snyk waived, gate failed, merged anyway.",
        "story": "Tier A. Forbids forward-fix PR; rewards CI inspection. Held-out eval.",
    },
}

TIER_COLORS = {
    "easy": "#22c55e",  "medium": "#fbbf24",
    "hard": "#ef4444",  "expert": "#a78bfa",
}


# Legacy action surface (Round-1) — kept as the default tab for ease of use.
LEGACY_ACTION_TYPES = [
    "check_status", "trace_dependencies", "query_logs",
    "query_metrics", "check_deploys", "inspect_code", "submit_diagnosis",
]
METRICS = ["latency_p99", "error_rate", "cpu", "memory", "connections"]
SEVERITIES = ["", "error", "warn", "info", "debug"]
CATEGORIES = [c.value for c in RootCauseCategory]
REMEDIATIONS = [r.value for r in Remediation]

# Multi-app surface (Round-2) — agent uses these to interact with the
# enterprise-shaped tools that exist alongside the raw investigation API.
APP_OPS: dict[str, list[str]] = {
    AppName.ALERTHUB.value:  [op.value for op in AlertHubOp],
    AppName.OBSLY.value:     [op.value for op in ObslyOp],
    AppName.REPOHUB.value:   [op.value for op in RepoHubOp],
    AppName.TICKETDESK.value:[op.value for op in TicketDeskOp],
    AppName.CHATOPS.value:   [op.value for op in ChatOpsOp],
    AppName.UATSIM.value:    [op.value for op in UATSimOp],
}


# ------------------------------------------------------------------ helpers

def get_available_services(task_id: str) -> list[str]:
    try:
        return list(_loader.get_scenario(task_id)["services"].keys())
    except Exception:
        return []


def _render_task_choices() -> list[tuple[str, str]]:
    """Group tasks by tier in dropdown labels."""
    rows = []
    for task_id, meta in TASK_META.items():
        eval_tag = "  •  EVAL" if task_id in EVAL_TASK_IDS else ""
        rows.append((meta["name"] + eval_tag, task_id))
    return rows


def _render_log_table(log_data: list) -> str:
    if not log_data:
        return ("| Step | Action | Reward | Δpolicy | Remaining |\n"
                "|------|--------|--------|---------|-----------|\n"
                "| | *No steps taken yet* | | | |")
    lines = ["| Step | Action | Reward | Δpolicy | Remaining |",
             "|------|--------|--------|---------|-----------|"]
    for r in log_data:
        lines.append(f"| {r[0]} | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} |")
    return "\n".join(lines)


def _render_world_events(events: list[dict], sim_time: str | None,
                         pending: int) -> str:
    """Compact card-style HTML for the world-events feed."""
    if not events and pending == 0 and not sim_time:
        return ("<p style='color:#475569;font-size:13px;'>"
                "World clock starts when you begin an investigation.</p>")
    head = (f"<div style='color:#94a3b8;font-size:12px;margin-bottom:6px;'>"
            f"sim-time <strong style='color:#e2e8f0;'>{sim_time or '-'}</strong>  "
            f"·  pending <strong style='color:#fbbf24;'>{pending}</strong></div>")
    if not events:
        return head + ("<p style='color:#64748b;font-size:13px;'>"
                        "No new world events this step.</p>")
    cards = []
    for ev in events:
        cards.append(f"""<div style='background:#0f172a;border-left:3px solid #fbbf24;
border-radius:6px;padding:8px 12px;margin-bottom:6px;font-size:12px;'>
<div style='color:#fbbf24;font-weight:600;'>[{ev.get('event','?')}]
<span style='color:#94a3b8;font-weight:400;'>  service={ev.get('service') or '-'}</span></div>
<div style='color:#cbd5e1;margin-top:2px;'>{json.dumps(ev.get('payload', {}))}</div>
</div>""")
    return head + "".join(cards)


def _render_policy_chip(delta: float, violations: list[str]) -> str:
    if delta == 0 and not violations:
        return ""
    color = "#22c55e" if delta > 0 else "#ef4444"
    sign = "+" if delta > 0 else ""
    pieces = (f" · " + " ".join(violations)) if violations else ""
    return (f"<div style='display:inline-block;background:{color};color:white;"
            f"padding:4px 12px;border-radius:14px;font-size:12px;font-weight:600;'>"
            f"policy Δ {sign}{delta:.2f}{pieces}</div>")


def _render_grader_panel(gb: dict | None) -> str:
    """Final-step 4-head breakdown with horizontal bars."""
    if not gb:
        return ""
    score = gb.get("score", 0)
    heads = gb.get("heads", {}) or {}
    eff_w = gb.get("effective_weights", {}) or {}
    emoji = "🟢" if score >= 0.6 else "🟡" if score >= 0.3 else "🔴"

    rows = []
    head_colors = {
        "diagnosis": "#2E86AB", "policy": "#A23B72",
        "blast": "#F18F01",     "pr": "#C73E1D",
    }
    for h in ("diagnosis", "policy", "blast", "pr"):
        head = heads.get(h) or {}
        if head.get("score") is None:
            continue
        s = float(head["score"])
        w = eff_w.get(h, 0)
        c = head_colors.get(h, "#60a5fa")
        rows.append(f"""<div style='margin:6px 0;'>
<div style='display:flex;justify-content:space-between;font-size:12px;color:#cbd5e1;'>
<span><strong>{h}</strong> <span style='color:#64748b;'>(weight {w:.2f})</span></span>
<span style='color:#e2e8f0;'>{s:.3f}</span></div>
<div style='background:#1e293b;border-radius:4px;height:8px;margin-top:2px;'>
<div style='background:{c};width:{s*100:.0f}%;height:100%;border-radius:4px;'></div>
</div></div>""")
    bars = "".join(rows)
    return (f"""<div style='background:linear-gradient(135deg,#0f172a 0%,#1a1a2e 100%);
border-radius:12px;padding:16px;margin-top:8px;border:1px solid #334155;'>
<div style='font-size:18px;color:#e2e8f0;margin-bottom:8px;'>
{emoji}  <strong>Composite score: {score:.3f} / 1.000</strong></div>
{bars}
</div>""")


# ------------------------------------------------------------------ event handlers

def reset_environment(task_id: str):
    """Returns: (alert_html, service, rcs, status, log_json, log_md,
                 result, world_html, policy_html, grader_html)"""
    if not task_id:
        empty_msg = ("<p style='color:#94a3b8;text-align:center;padding:40px;'>"
                     "Select a task first.</p>")
        return (empty_msg, gr.update(choices=[]), gr.update(choices=[]),
                "", "[]", _render_log_table([]), "",
                _render_world_events([], None, 0), "", "")

    obs = _env.reset(task_id=task_id, episode_id=str(uuid.uuid4()))
    alert = obs.alert
    services = get_available_services(task_id)
    meta = TASK_META.get(task_id, {})
    tier_color = TIER_COLORS.get(meta.get("tier", ""), "#64748b")

    alert_html = f"""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
border-radius:12px;padding:20px;border-left:4px solid #e74c3c;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
<span style="background:#e74c3c;color:white;padding:2px 10px;border-radius:12px;
font-size:12px;font-weight:600;">{alert.severity.upper()}</span>
<span style="color:#e2e8f0;font-weight:600;font-size:16px;">{alert.service}</span>
<span style="background:{tier_color};color:white;padding:2px 10px;border-radius:12px;
font-size:11px;font-weight:600;">{meta.get("badge", "")}</span>
</div>
<p style="color:#cbd5e1;margin:4px 0;font-size:14px;">{alert.message}</p>
<p style="color:#64748b;margin:4px 0;font-size:12px;">
Alert timestamp: {alert.timestamp}  ·  sim-time now: {obs.sim_time or '-'}</p>
<p style="color:#94a3b8;margin:4px 0;font-size:12px;font-style:italic;">{meta.get("story", "")}</p>
<div style="display:flex;gap:16px;margin-top:12px;color:#94a3b8;font-size:13px;flex-wrap:wrap;">
<span>Budget: <strong style="color:#60a5fa;">{obs.steps_remaining} steps</strong></span>
<span>Pending world events: <strong style="color:#fbbf24;">{obs.pending_world_events}</strong></span>
</div>
</div>"""

    status = (f"Episode started.  **{obs.steps_remaining}** steps available. "
              f"Investigate via the action panel, then submit your diagnosis.")
    return (alert_html,
            gr.update(choices=services, value=services[0] if services else None),
            gr.update(choices=services, value=None),
            status, "[]", _render_log_table([]), "",
            _render_world_events(list(obs.world_events), obs.sim_time,
                                  obs.pending_world_events),
            "", "")


def _build_action(action_kwargs: dict) -> IncidentTriageAction:
    return IncidentTriageAction(**action_kwargs)


def take_legacy_action(
    action_type, service, severity, keyword, metric,
    root_cause_service, root_cause_category, remediation,
    pr_proposal_json, blast_radius_json,
    log_json_str,
):
    """Run a legacy (Round-1-shaped) action against the env."""
    try:
        log_data = json.loads(log_json_str) if log_json_str else []
    except (json.JSONDecodeError, TypeError):
        log_data = []

    if not action_type:
        return (json.dumps(log_data), _render_log_table(log_data),
                "Select an action type.", "", "", "", "")
    try:
        kw: dict[str, Any] = {"action_type": action_type}
        if action_type == "submit_diagnosis":
            if not (root_cause_service and root_cause_category and remediation):
                return (json.dumps(log_data), _render_log_table(log_data),
                        "Fill in all diagnosis fields.", "", "", "", "")
            kw.update(root_cause_service=root_cause_service,
                       root_cause_category=root_cause_category,
                       remediation=remediation)
            for fld, src in (("pr_proposal", pr_proposal_json),
                              ("blast_radius", blast_radius_json)):
                if src and src.strip():
                    try:
                        kw[fld] = json.loads(src)
                    except json.JSONDecodeError as e:
                        return (json.dumps(log_data), _render_log_table(log_data),
                                f"Invalid JSON in {fld}: {e}", "", "", "", "")
        else:
            if not service:
                return (json.dumps(log_data), _render_log_table(log_data),
                        "Select a service to investigate.", "", "", "", "")
            kw["service"] = service
            if action_type == "query_logs":
                if severity: kw["severity"] = severity
                if keyword:  kw["keyword"] = keyword
            elif action_type == "query_metrics":
                kw["metric"] = metric or "error_rate"

        return _execute(_build_action(kw), action_type, service,
                         (root_cause_service, root_cause_category, remediation),
                         log_data)
    except Exception as e:
        return (json.dumps(log_data), _render_log_table(log_data),
                f"**Error:** {e}", "", "", "", "")


def take_multi_app_action(app, op, args_json, log_json_str):
    """Run a multi-app {app, op, args} action."""
    try:
        log_data = json.loads(log_json_str) if log_json_str else []
    except (json.JSONDecodeError, TypeError):
        log_data = []
    if not app or not op:
        return (json.dumps(log_data), _render_log_table(log_data),
                "Select both app and op.", "", "", "", "")
    try:
        args = json.loads(args_json) if (args_json and args_json.strip()) else {}
    except json.JSONDecodeError as e:
        return (json.dumps(log_data), _render_log_table(log_data),
                f"Invalid args JSON: {e}", "", "", "", "")
    try:
        action = IncidentTriageAction(app=app, op=op, args=args)
        return _execute(action, op, args.get("service") or app, ("", "", ""), log_data)
    except Exception as e:
        return (json.dumps(log_data), _render_log_table(log_data),
                f"**Error:** {e}", "", "", "", "")


def _execute(action: IncidentTriageAction, action_label: str, service_label: str,
             diag_fields: tuple, log_data: list):
    obs = _env.step(action)
    step_num = len(log_data) + 1
    reward = float(obs.reward or 0.0)
    pol_delta = float(obs.policy_delta_this_step or 0.0)

    if action.action_type == "submit_diagnosis":
        rcs, rcc, rem = diag_fields
        action_desc = f"submit_diagnosis({rcs}, {rcc}, {rem})"
    elif action.app != AppName.SYSTEM and not action.action_type:
        action_desc = f"{action.app.value}.{action.op}({service_label or '-'})"
    else:
        action_desc = f"{action_label}({service_label or '-'})"

    log_data = log_data + [[
        str(step_num), action_desc, f"{reward:+.3f}",
        f"{pol_delta:+.2f}" if pol_delta else "—",
        str(obs.steps_remaining),
    ]]

    result_data = obs.result.data if obs.result else ""
    world_html = _render_world_events(
        [e.model_dump() if hasattr(e, "model_dump") else e for e in (obs.world_events or [])],
        obs.sim_time, obs.pending_world_events or 0,
    )
    policy_html = _render_policy_chip(pol_delta, list(obs.policy_violations_this_step or []))

    if obs.done:
        gb = obs.grader_breakdown
        grader_html = _render_grader_panel(gb)
        if gb:
            status = (f"### Episode complete  ·  composite score "
                      f"**{gb.get('score', 0):.3f}**\n\n"
                      "Per-head breakdown shown below. Select a task and click "
                      "**Start Investigation** to try again.")
        else:
            status = "Episode ended without a submitted diagnosis."
    else:
        grader_html = ""
        status = (f"Steps remaining: **{obs.steps_remaining}**  ·  "
                  f"sim-time: {obs.sim_time or '-'}  ·  "
                  f"investigated: {', '.join(obs.services_investigated) or 'none'}")

    return (json.dumps(log_data), _render_log_table(log_data),
            status, result_data, world_html, policy_html, grader_html)


# ------------------------------------------------------------------ layout

def build_gradio_app() -> gr.Blocks:

    with gr.Blocks(title="Incident Triage Environment") as demo:

        gr.HTML("""<div style="text-align:center;padding:20px 0 10px 0;">
<h1 style="margin:0;font-size:30px;color:#e2e8f0;">Incident Triage Environment</h1>
<p style="color:#94a3b8;margin:8px 0 0 0;font-size:15px;">
OpenEnv-compliant RL environment for production incident triage.<br/>
Multi-app enterprise surface  ·  Dynamic world clock  ·  Declarative business policies  ·  4-head composite grader</p>
</div>""")

        # Hidden state — JSON-serialised log so we don't fight Gradio's State semantics.
        log_json = gr.Textbox(value="[]", visible=False)

        with gr.Row():
            # ---------------------------- LEFT — controls
            with gr.Column(scale=1):
                gr.Markdown("### 1. Pick a scenario")
                task_dropdown = gr.Dropdown(
                    choices=_render_task_choices(),
                    label="Incident Scenario", value=None,
                    info="EVAL = held-out generalisation split, never seen during training",
                )
                gr.HTML("""<div style="font-size:11px;color:#94a3b8;padding:6px;
background:#0f172a;border-radius:6px;margin:4px 0 8px;">
<span style='color:#22c55e;'>Easy</span> single service ·
<span style='color:#fbbf24;'>Medium</span> 5 services ·
<span style='color:#ef4444;'>Hard</span> red herrings ·
<span style='color:#a78bfa;'>Expert</span> stealth signals ·
<span style='color:#94a3b8;'>Tier A</span> process-hygiene gates
</div>""")
                reset_btn = gr.Button("Start Investigation", variant="primary", size="lg")

                gr.Markdown("---\n### 2. Take an action")

                with gr.Tabs():

                    # ---- Investigation tab (legacy / Round-1)
                    with gr.Tab("Investigate"):
                        action_type = gr.Dropdown(
                            choices=LEGACY_ACTION_TYPES, value="check_status",
                            label="Action Type",
                            info="Round-1 surface: 7 verbs over the system-under-test",
                        )
                        service = gr.Dropdown(choices=[], label="Service",
                                               info="Target service to investigate")
                        with gr.Accordion("Query options", open=False):
                            severity = gr.Dropdown(choices=SEVERITIES, value="",
                                                    label="Log severity filter")
                            keyword = gr.Textbox(label="Log keyword filter",
                                                  placeholder="timeout, NPE, CVE…")
                            metric = gr.Dropdown(choices=METRICS, value="error_rate",
                                                  label="Metric (for query_metrics)")

                        with gr.Accordion("Submit diagnosis (terminal)", open=False):
                            root_cause_service = gr.Dropdown(
                                choices=[], label="Root cause service")
                            root_cause_category = gr.Dropdown(
                                choices=CATEGORIES, label="Root cause category")
                            remediation_input = gr.Dropdown(
                                choices=REMEDIATIONS, label="Remediation")
                            pr_proposal_json = gr.Textbox(
                                label="pr_proposal (JSON, optional)",
                                placeholder='{"target_repo": "...", "head_branch": "...", '
                                              '"title": "...", "summary": "...", "diff_patch": "..."}',
                                lines=4, max_lines=8)
                            blast_radius_json = gr.Textbox(
                                label="blast_radius (JSON, optional)",
                                placeholder='{"affected_services": ["..."], '
                                              '"estimated_requests_failed": 1000, '
                                              '"missed_regions": [], '
                                              '"outage_window_start": "...", "outage_window_end": "..."}',
                                lines=4, max_lines=8)

                        invest_btn = gr.Button("Execute investigation action",
                                                variant="secondary", size="lg")

                    # ---- Multi-App tab (Round-2 surface)
                    with gr.Tab("Multi-App"):
                        gr.Markdown(
                            "Six enterprise-shaped apps. Each one wraps a slice of "
                            "the world the SRE actually touches during an incident."
                        )
                        app_dropdown = gr.Dropdown(
                            choices=list(APP_OPS.keys()),
                            value=AppName.CHATOPS.value, label="App",
                            info="alerthub · obsly · repohub · ticketdesk · chatops · uatsim",
                        )
                        op_dropdown = gr.Dropdown(
                            choices=APP_OPS[AppName.CHATOPS.value],
                            value="page_oncall", label="Op",
                        )
                        args_json = gr.Textbox(
                            label="args (JSON)",
                            placeholder='{"role": "payments-lead"}',
                            lines=3,
                            info="Per-op required args are validated server-side",
                        )
                        app_btn = gr.Button("Execute multi-app action",
                                             variant="secondary", size="lg")

            # ---------------------------- RIGHT — observation & feedback
            with gr.Column(scale=2):
                alert_display = gr.HTML(
                    value="""<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
border-radius:12px;padding:40px;text-align:center;">
<h3 style="color:#64748b;margin:0;">Pick a scenario and hit "Start Investigation"</h3>
<p style="color:#475569;margin:8px 0 0;font-size:13px;">
You'll receive an alert and the world-clock starts ticking</p></div>""",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### World events  ·  sim-time")
                        world_display = gr.HTML(
                            value="<p style='color:#475569;font-size:13px;'>"
                                  "World clock starts when you begin an investigation.</p>")
                    with gr.Column(scale=1):
                        gr.Markdown("#### Per-step policy")
                        policy_display = gr.HTML(value="")
                        gr.Markdown(
                            "<small style='color:#64748b;'>Policy rules fire green for "
                            "rewarded behaviours (e.g. paging oncall before rollback) "
                            "and red for violations (e.g. forward-fix PR during freeze).</small>"
                        )

                status_display = gr.Markdown("")

                gr.Markdown("#### Investigation log")
                log_display = gr.Markdown(_render_log_table([]))

                gr.Markdown("#### Last action result")
                result_display = gr.Textbox(label="", lines=8, interactive=False)

                gr.Markdown("#### Final score breakdown")
                grader_display = gr.HTML(value="")

        # ---------------------------- About
        gr.Markdown("---")
        with gr.Accordion("About this environment", open=True):
            gr.Markdown("""
**Theme**: World Modeling — Professional Tasks (Theme 3.1) + Scaler AI Labs bonus (Multi-App Enterprise Workflows).

#### Links

| Resource | URL |
|---|---|
| Live HF Space (this env) | https://huggingface.co/spaces/AbhishekMallick/incident-triage-env |
| GitHub source | https://github.com/deepraj21/Incident-Triage-Env |
| Blog write-up (HF) | https://huggingface.co/spaces/AbhishekMallick/incident-triage-env/blob/main/BLOG.md |
| Blog write-up (Medium) | https://medium.com/@mallickabhishek97/building-an-openenv-compliant-incident-triage-environment-for-rl-what-i-learned-along-the-way-3a39b0917862 |
| Colab training notebook | https://colab.research.google.com/drive/10dHOtRzLHY3aMSc21hxQouLxTi_gXv-t#scrollTo=train |
| Trained adapter — Qwen2.5-1.5B SFT | https://huggingface.co/AbhishekMallick/incident-triage-grpo-train |
| Trained adapter — Qwen2.5-3B SFT | https://huggingface.co/AbhishekMallick/incident-triage-grpo-train-Qwen3B |
| Trained adapter — Qwen2.5-7B SFT | https://huggingface.co/AbhishekMallick/incident-triage-sft-train-Qwen2.5-7B |

#### Architecture at a glance

System architecture · episode lifecycle · app dispatcher → real-world enterprise tools:

![System architecture](https://qmplv4qr76dicsvw.public.blob.vercel-storage.com/incident-triage/incident-triage.jpeg)

![Episode lifecycle](https://qmplv4qr76dicsvw.public.blob.vercel-storage.com/incident-triage/episode-lifecycle.jpeg)

![App dispatcher → real enterprise tools](https://qmplv4qr76dicsvw.public.blob.vercel-storage.com/incident-triage/env.jpeg)

#### Why it exists
Production incident triage is the canonical SRE task: alert fires, you investigate across services,
form hypotheses under time pressure, take remediation. This env captures that full lifecycle —
plus the **process-hygiene gates** (UAT, CI quality, change-freeze) that surround the on-call work itself.

#### Grading happens per step — not just at the end
Every `step()` returns a reward. Three sources grade each action:

1. **`InvestigationRubric`** — informational value of the action (direct evidence ≫ causal-chain ≫ contextual ≫ redundant; bonuses for following the dependency chain, penalties for chasing red herrings, repeat-query damping at 3+ visits).
2. **`PolicyEngine`** — declarative business rules fire **per step** with `+` deltas for compliance (e.g. paged on-call before submitting rollback) and `−` deltas for violations (e.g. opened a forward-fix PR during change-freeze). The chip on the right shows the live delta.
3. **Terminal `CompositeScorer`** — only runs on `submit_diagnosis`. It is the **cap**, not the only signal. The 4 horizontal bars below show its head-by-head decomposition.

This is what makes the env trainable in 100 GRPO steps instead of 10,000: the reward landscape is dense and interpretable from step 1.

#### Scenarios — 8 total, 3 held-out for eval

| Task | Tier | Steps | Family | Split |
|---|---|---|---|---|
| `easy_single_service_failure` | Easy | 10 | bad_deploy NPE | train |
| `medium_cascading_dependency` | Medium | 15 | config_change cascade | train |
| `hard_multi_signal_cascade` | Hard | 20 | resource_exhaustion + red herrings | **eval** |
| `hard_region_failover` | Hard | 18 | config_change (failover_enabled=false) | train |
| `hard_freeze_violation` | Hard | 15 | bad_deploy during change-freeze | train |
| `expert_stealth_regression` | Expert | 20 | silent metric drift (no error logs) | **eval** |
| `medium_uat_skipped` (Tier A) | Medium | 14 | UAT-bypassed bad_deploy | train |
| `hard_pr_quality_breach` (Tier A) | Hard | 18 | waived Snyk finding shipped past failed CI | **eval** |

Each scenario carries: `clock` (sim-time anchor), `tags`, full services graph, `timeline` of mid-episode events, declarative `policies`, multi-app state blocks (alerthub / chatops / ticketdesk / obsly / repohub / uatsim), and ground-truth including `correct_pr` + `correct_blast_radius`. Seed variants (5 train tasks × 8 seeds) deterministically perturb red-herring ordering, timestamp jitter, and oncall lead names while preserving ground truth — yielding ~40 distinct training prompts.

#### App dispatcher — 6 enterprise apps + system

| App | Modeled after | Ops |
|---|---|---|
| `alerthub` | PagerDuty | `list_alerts` (clock-aware), `get_alert`, `ack_alert` |
| `obsly` | Datadog | `query_logs`, `query_metric`, `get_trace`, `list_dashboards`, `check_status`, `trace_dependencies` |
| `repohub` | GitHub + CI | `recent_commits`, `get_diff`, `get_file`, `list_files`, `get_blame`, `open_pr`, `ci_check`, `list_pr_history` |
| `ticketdesk` | Jira | `search_tickets`, `get_ticket`, `create_incident`, `link_pr`, `add_comment` |
| `chatops` | Slack | `post_update`, `read_channel`, `page_oncall` |
| `uatsim` (Tier A) | pre-prod gate | `list_stages`, `get_stage`, `get_signoff_status`, `check_uat_record` |
| `system` | – | `submit_diagnosis` |

`repohub.ci_check` returns a synthetic Snyk/Sonar/Raven/coverage report; `uatsim.check_uat_record` exposes whether a service deploy bypassed required UAT stages. Both are scenario-driven so policies can enforce process-hygiene rules at the operational layer.

#### Dynamic world engine

- **`WorldClock`** — deterministic step-based sim-time. Every `step()` advances by `step_seconds` (default 30 s).
- **`EventQueue`** — fires timeline events at the configured step. Five types: `new_log`, `new_alert`, `new_deploy`, `oncall_handoff`, `slo_burn`.
- Events with `visible: false` mutate the world *without* surfacing on `obs.world_events` — the agent must re-poll to discover them. This is the substrate for the stealth-regression scenario.

#### PolicyEngine — declarative DSL

Each scenario opts into rules via tags. Supported predicates:

- `args_match` — match action arguments (e.g. `remediation: rollback_deploy`)
- `require_prior` — a prior action of `{app, op}` must exist
- `require_prior_within_steps: N` — prior action must be within last N steps
- `max_occurrences: N` — no more than N times this episode
- `forbidden_if: { scenario_tag: "..." }` — tag-gated bans (e.g. forward-fix PR during `change_freeze`)
- `penalty` / `bonus` — per-step delta, signed

The same engine produces both the per-step `policy_delta_this_step` (for shaping during RL training) and the `compliance_score` head used in terminal grading.

#### 4-head composite grader

| Head | Weight | Signal |
|---|---|---|
| **diagnosis** | 0.40 | root_cause_service exact match (one-hop partial) · category exact (same-family partial) · remediation match · evidence-coverage proportion · efficiency · shotgun/circular/destructive penalties |
| **policy** | 0.20 | `1 − (violations · weight / total_attempts)` from `PolicyEngine.summary()` |
| **blast** | 0.20 | F1 on `affected_services` × 0.40 + F1 on `missed_regions` × 0.20 + log-tolerant magnitude on `estimated_requests_failed` × 0.20 + IoU on outage window × 0.20 |
| **pr** | 0.20 | exact `target_repo` × 0.30 + recall over `touched_files` × 0.35 + keyword coverage in title/summary × 0.25 + structural validity × 0.10 |

**Inapplicable heads redistribute their weight** — if a scenario has no `correct_pr` ground truth, the PR head's 0.20 rolls into diagnosis + policy + blast proportionally. Final score clamped to `(0.001, 0.999)`.

#### Training pipeline (SFT → GRPO)

```
oracle trajectories (40 = 5 tasks × 8 seeds)
        │
        ▼
TRL SFTTrainer + LoRA r=16, 3 epochs        ──── stage 1 (218 s on T4)
        │
        ▼
17.5 MB LoRA adapter
        │
        ▼
TRL GRPOTrainer + replay-and-grade reward   ──── stage 2 (~50 min on T4)
        │
        ▼
refined adapter  ⭢  HF Hub  ⭢  before/after eval on held-out split
```

Same script targets Qwen2.5-1.5B / 3B / 7B by changing the `MODEL_NAME` env var. All three trained adapter sizes are linked at the top.

#### Held-out eval results (3 scenarios × 5 seeds)

| Task | Baseline | Trained (SFT+GRPO) | Δ |
|---|---:|---:|---:|
| `hard_multi_signal_cascade` | 0.400 | 0.747 | +0.347 |
| `expert_stealth_regression` | 0.391 | 0.800 | +0.410 |
| `hard_pr_quality_breach` | 0.336 | 0.726 | +0.390 |
| **OVERALL** | **0.375** | **0.758** | **+0.382 (+102 %)** |

Per-head: diagnosis +0.20 · policy ±0 · blast +0.68 · pr +0.82.

#### Reproduce

```bash
git clone https://github.com/deepraj21/Incident-Triage-Env
cd Incident-Triage-Env
pip install -e ".[training]"

python scripts/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct \\
    --num-epochs 3 --seeds 0 1 2 3 4 5 6 7 \\
    --output-dir ./trained/sft

python scripts/eval_before_after.py --compare baseline finetuned
python scripts/plot_model.py
```

150 tests passing. Endpoints: `GET /tasks` · `WS /ws` · `POST /grader` · `POST /reset` · `POST /step` · `GET /state` · `POST /mcp`.
""")

        # --------------------------- wiring

        reset_btn.click(
            fn=reset_environment,
            inputs=[task_dropdown],
            outputs=[alert_display, service, root_cause_service, status_display,
                     log_json, log_display, result_display,
                     world_display, policy_display, grader_display],
        )

        invest_btn.click(
            fn=take_legacy_action,
            inputs=[action_type, service, severity, keyword, metric,
                    root_cause_service, root_cause_category, remediation_input,
                    pr_proposal_json, blast_radius_json, log_json],
            outputs=[log_json, log_display, status_display, result_display,
                     world_display, policy_display, grader_display],
        )

        app_btn.click(
            fn=take_multi_app_action,
            inputs=[app_dropdown, op_dropdown, args_json, log_json],
            outputs=[log_json, log_display, status_display, result_display,
                     world_display, policy_display, grader_display],
        )

        # When app changes, refresh the op dropdown to that app's ops
        def _refresh_ops(app_value: str):
            return gr.update(choices=APP_OPS.get(app_value, []),
                             value=(APP_OPS.get(app_value, []) or [None])[0])
        app_dropdown.change(_refresh_ops, inputs=[app_dropdown], outputs=[op_dropdown])

    return demo
