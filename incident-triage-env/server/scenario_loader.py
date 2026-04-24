import json
from pathlib import Path
from typing import Optional

from models import AlertInfo
from server.scenario_variants import apply_variant

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

TASK_FILES = {
    "easy_single_service_failure": "task_easy.json",
    "medium_cascading_dependency": "task_medium.json",
    "hard_multi_signal_cascade": "task_hard.json",
    "hard_region_failover": "task_region_failover.json",
    "hard_freeze_violation": "task_freeze_violation.json",
    "expert_stealth_regression": "task_stealth_regression.json",
    # Tier A scenarios — process-hygiene failures.
    "medium_uat_skipped": "task_uat_skipped.json",
    "hard_pr_quality_breach": "task_pr_quality_breach.json",
}

# Held-out split — never used for GRPO training, reserved for before/after eval.
# Tier A's hard_pr_quality_breach is added to eval because it exercises a
# genuinely novel signal (CI gate + waived-finding detection) that training on
# the simpler UAT scenario cannot leak into.
EVAL_TASK_IDS = {
    "hard_multi_signal_cascade",
    "expert_stealth_regression",
    "hard_pr_quality_breach",
}

TRAIN_TASK_IDS = [tid for tid in TASK_FILES if tid not in EVAL_TASK_IDS]


class ScenarioLoader:
    def __init__(self):
        self._scenarios: dict[str, dict] = {}
        for task_id, filename in TASK_FILES.items():
            path = SCENARIOS_DIR / filename
            with open(path) as f:
                self._scenarios[task_id] = json.load(f)

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

    def get_scenario(self, task_id: str, seed: Optional[int] = None) -> dict:
        if task_id not in self._scenarios:
            raise ValueError(f"Unknown task_id: {task_id}. Available: {list(self._scenarios.keys())}")
        base = self._scenarios[task_id]
        if seed is None:
            return base
        return apply_variant(base, seed)

    def list_train_tasks(self) -> list[str]:
        return list(TRAIN_TASK_IDS)

    def list_eval_tasks(self) -> list[str]:
        return sorted(EVAL_TASK_IDS)

    def get_alert(self, task_id: str) -> AlertInfo:
        scenario = self.get_scenario(task_id)
        a = scenario["alert"]
        return AlertInfo(
            service=a["service"],
            message=a["message"],
            severity=a["severity"],
            timestamp=a["timestamp"],
        )

    def _get_service_data(self, task_id: str, service: str) -> dict:
        scenario = self.get_scenario(task_id)
        services = scenario["services"]
        if service not in services:
            raise ValueError(f"Unknown service '{service}'. Available: {list(services.keys())}")
        return services[service]

    def query_logs(self, task_id: str, service: str, severity: Optional[str] = None, keyword: Optional[str] = None) -> str:
        data = self._get_service_data(task_id, service)
        logs = data.get("logs", [])

        filtered = logs
        if severity:
            filtered = [l for l in filtered if l["severity"] == severity]
        if keyword:
            kw = keyword.lower()
            filtered = [l for l in filtered if kw in l["message"].lower()]

        if not filtered:
            return f"No log entries found for {service}" + (f" (severity={severity})" if severity else "") + (f" (keyword='{keyword}')" if keyword else "")

        lines = []
        for entry in filtered:
            lines.append(f"[{entry['timestamp']}] [{entry['severity'].upper()}] {entry['message']}")
        return "\n".join(lines)

    def query_metrics(self, task_id: str, service: str, metric: str) -> str:
        data = self._get_service_data(task_id, service)
        metrics = data.get("metrics", {})

        if metric not in metrics:
            return f"No metric '{metric}' available for {service}. Available: {list(metrics.keys())}"

        points = metrics[metric]
        if not points:
            return f"No data points for {service} {metric}"

        unit_map = {
            "error_rate": "%",
            "latency_p99": "ms",
            "cpu": "%",
            "memory": "%",
            "connections": "",
        }
        unit = unit_map.get(metric, "")

        lines = [f"Metric: {metric} for {service}"]
        lines.append(f"{'Time':<10} {'Value':>10}")
        lines.append("-" * 22)
        for p in points:
            val_str = f"{p['val']}{unit}"
            lines.append(f"{p['ts']:<10} {val_str:>10}")
        return "\n".join(lines)

    def check_deploys(self, task_id: str, service: str) -> str:
        data = self._get_service_data(task_id, service)
        deploys = data.get("deploys", [])

        if not deploys:
            return f"No recent deploys for {service}"

        lines = []
        for d in deploys:
            lines.append(f"Deploy at {d['timestamp']} by {d['author']}:")
            lines.append(f"  {d['diff_summary']}")
        return "\n".join(lines)

    def trace_dependencies(self, task_id: str, service: str) -> str:
        data = self._get_service_data(task_id, service)
        scenario = self.get_scenario(task_id)

        deps = data.get("dependencies", [])
        dependents = data.get("dependents", [])

        lines = [f"Dependency graph for {service}:"]
        lines.append(f"  Upstream (depends on): {deps if deps else ['none']}")
        lines.append(f"  Downstream (depended on by): {dependents if dependents else ['none']}")

        lines.append("")
        lines.append("Connection health:")
        for dep in deps:
            dep_data = scenario["services"].get(dep, {})
            health = dep_data.get("status", {}).get("health", "unknown")
            lines.append(f"  {service} -> {dep}: {health}")
        for dependent in dependents:
            dep_data = scenario["services"].get(dependent, {})
            health = dep_data.get("status", {}).get("health", "unknown")
            lines.append(f"  {dependent} -> {service}: {health}")

        return "\n".join(lines)

    def check_status(self, task_id: str, service: str) -> str:
        data = self._get_service_data(task_id, service)
        status = data.get("status", {})

        lines = [f"Status for {service}:"]
        lines.append(f"  Health: {status.get('health', 'unknown')}")
        lines.append(f"  Uptime: {status.get('uptime', 'N/A')}")
        alerts = status.get("active_alerts", [])
        if alerts:
            lines.append(f"  Active alerts: {', '.join(alerts)}")
        else:
            lines.append("  Active alerts: none")
        return "\n".join(lines)

    def inspect_code(self, task_id: str, service: str, file_path: Optional[str] = None) -> str:
        data = self._get_service_data(task_id, service)
        snippets = data.get("code_snippets", {})

        if not snippets:
            return f"No code snippets available for {service}"

        if file_path:
            if file_path in snippets:
                return f"=== {file_path} ===\n{snippets[file_path]}"
            return f"File '{file_path}' not found for {service}. Available: {list(snippets.keys())}"

        lines = []
        for name, content in snippets.items():
            lines.append(f"=== {name} ===")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).rstrip()
