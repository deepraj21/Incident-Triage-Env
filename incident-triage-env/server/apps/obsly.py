"""Obsly — Datadog/Grafana-shaped observability app.

Reads service telemetry (logs, metrics, status, dependency graph) from the
scenario's flat `services` map. Also exposes `get_trace` and `list_dashboards`
for richer new-schema scenarios.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from server.apps._views import (
    format_dependencies,
    format_logs,
    format_metric,
    format_status,
    unknown_service,
)


def _service_data(scenario: Dict[str, Any], svc: Optional[str]) -> Optional[Dict[str, Any]]:
    services = scenario.get("services", {})
    if not svc or svc not in services:
        return None
    return services[svc]


def dispatch(
    op: str,
    args: Dict[str, Any],
    scenario: Dict[str, Any],
    state: Any,
    clock: Optional[Any] = None,
) -> str:
    svc = args.get("service")

    if op == "query_logs":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        return format_logs(svc, data.get("logs", []), args.get("severity"), args.get("keyword"))

    if op == "query_metric":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        metric = args.get("metric", "error_rate")
        return format_metric(svc, data.get("metrics", {}), metric)

    if op == "check_status":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        return format_status(svc, data.get("status", {}))

    if op == "trace_dependencies":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        return format_dependencies(svc, scenario, data)

    if op == "get_trace":
        tid = args.get("trace_id", "")
        traces = scenario.get("obsly", {}).get("traces", {})
        trace = traces.get(tid)
        if trace is None:
            return json.dumps({"error": f"trace '{tid}' not found"}, indent=2)
        return json.dumps(trace, indent=2)

    if op == "list_dashboards":
        dashboards = scenario.get("obsly", {}).get("dashboards")
        if dashboards is None:
            # Synthesize a default dashboard from available services.
            services = list(scenario.get("services", {}).keys())
            dashboards = [{"name": "Service Health Overview", "services": services}]
        return json.dumps(dashboards, indent=2)

    return f"Unknown op for obsly: {op}"
