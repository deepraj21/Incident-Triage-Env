"""EventQueue — fires mid-episode timeline events into the running scenario.

Event shape on the wire:
    {
      "at_step": 4,
      "event": "new_log" | "slo_burn" | "new_deploy" | "oncall_handoff" | "new_alert",
      "service": "user-service",          # optional, depends on event type
      "visible": true,                    # default True; false = silent (stealth)
      "payload": { ... }
    }

Determinism: events are sorted by (at_step, original index) at construction
and each event fires at most once per episode. The queue mutates the episode-
local scenario dict (IncidentTriageEnv deep-copies on reset — see environment.py)
so static ScenarioLoader fixtures are never polluted across episodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_SUPPORTED_EVENTS = {
    "new_log",
    "slo_burn",
    "new_deploy",
    "oncall_handoff",
    "new_alert",
}


class EventQueue:
    def __init__(self, timeline: List[Dict[str, Any]]):
        # Stable sort: primary key = at_step, secondary = original order.
        indexed = list(enumerate(timeline or []))
        indexed.sort(key=lambda pair: (pair[1].get("at_step", 0), pair[0]))
        self._events: List[Dict[str, Any]] = [e for _, e in indexed]
        self._fired: set[int] = set()

    def fire_due(
        self,
        clock: Any,
        scenario: Dict[str, Any],
        state: Any,
    ) -> List[Dict[str, Any]]:
        """Apply all events whose at_step <= clock.step; return VISIBLE ones.

        Silent events (`visible: false`) mutate the world but are not returned,
        so the agent has to re-poll to discover them (stealth regressions).
        """
        visible: List[Dict[str, Any]] = []
        for i, ev in enumerate(self._events):
            if i in self._fired:
                continue
            if ev.get("at_step", 0) > clock.step:
                continue
            applied = self._apply(ev, scenario, state, clock)
            self._fired.add(i)
            # Record every event (visible or not) in state for grader + video replay.
            state.event_log.append({**applied, "fired_at_step": clock.step})
            if ev.get("visible", True):
                visible.append(applied)
        return visible

    # -- per-event-type handlers -----------------------------------------

    def _apply(
        self,
        ev: Dict[str, Any],
        scenario: Dict[str, Any],
        state: Any,
        clock: Any,
    ) -> Dict[str, Any]:
        kind = ev.get("event", "")
        svc = ev.get("service")
        payload = ev.get("payload", {}) or {}
        applied = {
            "event": kind,
            "service": svc,
            "at_step": ev.get("at_step"),
            "visible": ev.get("visible", True),
            "payload": payload,
        }

        if kind not in _SUPPORTED_EVENTS:
            applied["error"] = f"unsupported event type: {kind}"
            return applied

        services = scenario.setdefault("services", {})
        svc_bucket = services.setdefault(svc, {}) if svc else None

        if kind == "new_log" and svc_bucket is not None:
            logs = svc_bucket.setdefault("logs", [])
            log_entry = {
                "timestamp": payload.get("timestamp", clock.now_iso()),
                "severity": payload.get("severity", "info"),
                "message": payload.get("message", ""),
            }
            logs.append(log_entry)
            applied["applied_log"] = log_entry

        elif kind == "slo_burn" and svc_bucket is not None:
            status = svc_bucket.setdefault("status", {})
            status["slo_burn_rate"] = payload.get("burn_rate_x", 1.0)
            applied["slo_burn_rate"] = status["slo_burn_rate"]

        elif kind == "new_deploy" and svc_bucket is not None:
            deploys = svc_bucket.setdefault("deploys", [])
            deploy = {
                "timestamp": payload.get("timestamp", clock.now_iso()),
                "author": payload.get("author", "unknown"),
                "diff_summary": payload.get("diff_summary", ""),
                "diff_patch": payload.get("diff_patch", ""),
            }
            deploys.append(deploy)
            applied["applied_deploy"] = deploy

        elif kind == "oncall_handoff":
            chatops = scenario.setdefault("chatops", {})
            chatops["oncall_lead"] = payload.get("to", chatops.get("oncall_lead"))
            applied["handoff"] = {
                "from": payload.get("from"),
                "to": payload.get("to"),
            }

        elif kind == "new_alert":
            alerthub = scenario.setdefault("alerthub", {})
            alerts = alerthub.setdefault("alerts", [])
            alert = {
                "id": payload.get("id", f"ALERT-{len(alerts) + 2}"),
                "service": payload.get("service", svc),
                "message": payload.get("message", ""),
                "severity": payload.get("severity", "warning"),
                "fire_time": payload.get("fire_time", clock.now_iso()),
            }
            alerts.append(alert)
            applied["applied_alert"] = alert

        return applied

    # -- introspection (useful for tests and observation tension) --------

    def pending_count(self) -> int:
        return len(self._events) - len(self._fired)
