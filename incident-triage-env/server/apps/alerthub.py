"""AlertHub — PagerDuty-shaped alerting app.

Dispatcher reads from `scenario["alerthub"]["alerts"]` when present (new-schema
scenarios) and falls back to synthesizing a single alert from the Round-1
`scenario["alert"]` field so Phase 1 works end-to-end on today's fixtures.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _alert_list(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts = scenario.get("alerthub", {}).get("alerts")
    if alerts is not None:
        return list(alerts)
    # Adapter: synthesize PagerDuty-like alert list from the Round-1 flat alert.
    a = scenario.get("alert")
    if not a:
        return []
    return [
        {
            "id": a.get("id", "ALERT-1"),
            "service": a.get("service"),
            "message": a.get("message"),
            "severity": a.get("severity"),
            "fire_time": a.get("timestamp"),
        }
    ]


def dispatch(
    op: str,
    args: Dict[str, Any],
    scenario: Dict[str, Any],
    state: Any,
    clock: Optional[Any] = None,
) -> str:
    alerts = _alert_list(scenario)
    # Clock-aware filter: only surface alerts whose fire_time has elapsed.
    # This is what makes mid-episode `new_alert` events pop in at the right step.
    now_iso = clock.now_iso() if clock is not None else None
    if now_iso is not None:
        alerts = [
            a for a in alerts
            if not a.get("fire_time") or a.get("fire_time") <= now_iso
        ]

    if op == "list_alerts":
        if not alerts:
            return "No active alerts."
        summary = [
            {"id": a.get("id"), "service": a.get("service"), "severity": a.get("severity"),
             "message": a.get("message"), "fire_time": a.get("fire_time")}
            for a in alerts
        ]
        return json.dumps(summary, indent=2)

    if op == "get_alert":
        aid = args.get("alert_id")
        found = next((a for a in alerts if a.get("id") == aid), None)
        if found is None:
            return json.dumps({"error": f"alert '{aid}' not found"}, indent=2)
        return json.dumps(found, indent=2)

    if op == "ack_alert":
        aid = args.get("alert_id")
        if not any(a.get("id") == aid for a in alerts):
            return json.dumps({"error": f"cannot ack unknown alert '{aid}'"}, indent=2)
        if aid not in state.acked_alerts:
            state.acked_alerts.append(aid)
        return f"acked {aid}"

    return f"Unknown op for alerthub: {op}"
