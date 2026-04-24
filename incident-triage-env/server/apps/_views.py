"""Shared formatting helpers for app dispatchers.

Keeps each `server/apps/<app>.py` module small by centralizing log/metric/
status/dependency rendering used across Obsly and RepoHub.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_UNIT_MAP = {
    "error_rate": "%",
    "latency_p99": "ms",
    "cpu": "%",
    "memory": "%",
    "connections": "",
}


def format_logs(
    service: str,
    logs: List[Dict[str, Any]],
    severity: Optional[str] = None,
    keyword: Optional[str] = None,
) -> str:
    filtered = logs
    if severity:
        filtered = [entry for entry in filtered if entry.get("severity") == severity]
    if keyword:
        kw = keyword.lower()
        filtered = [entry for entry in filtered if kw in entry.get("message", "").lower()]

    if not filtered:
        suffix = ""
        if severity:
            suffix += f" (severity={severity})"
        if keyword:
            suffix += f" (keyword='{keyword}')"
        return f"No log entries found for {service}" + suffix

    return "\n".join(
        f"[{e['timestamp']}] [{e['severity'].upper()}] {e['message']}" for e in filtered
    )


def format_metric(service: str, metrics: Dict[str, Any], metric: str) -> str:
    if metric not in metrics:
        available = list(metrics.keys())
        return f"No metric '{metric}' available for {service}. Available: {available}"
    points = metrics[metric]
    if not points:
        return f"No data points for {service} {metric}"
    unit = _UNIT_MAP.get(metric, "")
    lines = [f"Metric: {metric} for {service}", f"{'Time':<10} {'Value':>10}", "-" * 22]
    for point in points:
        val_str = f"{point['val']}{unit}"
        lines.append(f"{point['ts']:<10} {val_str:>10}")
    return "\n".join(lines)


def format_status(service: str, status: Dict[str, Any]) -> str:
    lines = [f"Status for {service}:"]
    lines.append(f"  Health: {status.get('health', 'unknown')}")
    lines.append(f"  Uptime: {status.get('uptime', 'N/A')}")
    alerts = status.get("active_alerts", [])
    lines.append(f"  Active alerts: {', '.join(alerts) if alerts else 'none'}")
    return "\n".join(lines)


def format_dependencies(service: str, scenario: Dict[str, Any], svc_data: Dict[str, Any]) -> str:
    deps = svc_data.get("dependencies", [])
    dependents = svc_data.get("dependents", [])
    services = scenario.get("services", {})

    lines = [f"Dependency graph for {service}:"]
    lines.append(f"  Upstream (depends on): {deps if deps else ['none']}")
    lines.append(f"  Downstream (depended on by): {dependents if dependents else ['none']}")
    lines.append("")
    lines.append("Connection health:")
    for dep in deps:
        h = services.get(dep, {}).get("status", {}).get("health", "unknown")
        lines.append(f"  {service} -> {dep}: {h}")
    for dependent in dependents:
        h = services.get(dependent, {}).get("status", {}).get("health", "unknown")
        lines.append(f"  {dependent} -> {service}: {h}")
    return "\n".join(lines)


def unknown_service(service: Optional[str], scenario: Dict[str, Any]) -> str:
    services = list(scenario.get("services", {}).keys())
    return f"Unknown service '{service}'. Available: {services}"
