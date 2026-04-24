"""RepoHub — GitHub-shaped version-control app.

Adapts Round-1 scenario fields:
  - `service.deploys[*]`      → commits / diffs / blame
  - `service.code_snippets`   → file reads and listings
New-schema scenarios may provide `scenario["repohub"]` for richer data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from server.apps._views import unknown_service


def _service_data(scenario: Dict[str, Any], svc: Optional[str]) -> Optional[Dict[str, Any]]:
    services = scenario.get("services", {})
    if not svc or svc not in services:
        return None
    return services[svc]


def _synth_sha(service: str, index: int, deploy: Dict[str, Any]) -> str:
    existing = deploy.get("sha") or deploy.get("commit_sha")
    if existing:
        return str(existing)
    # Deterministic synthetic SHA so agents can re-fetch the same commit.
    ts = deploy.get("timestamp", "")
    return f"sha-{service[:4]}-{index}-{ts[-5:].replace(':', '')}"


def dispatch(
    op: str,
    args: Dict[str, Any],
    scenario: Dict[str, Any],
    state: Any,
    clock: Optional[Any] = None,
) -> str:
    svc = args.get("service")

    if op == "recent_commits":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        deploys = data.get("deploys", [])
        if not deploys:
            return f"No recent commits for {svc}"
        lines = []
        for i, d in enumerate(deploys):
            sha = _synth_sha(svc, i, d)
            lines.append(
                f"{sha} | {d.get('timestamp', '?')} | {d.get('author', '?')}: "
                f"{d.get('diff_summary', '')}"
            )
        return "\n".join(lines)

    if op == "get_diff":
        sha = args.get("commit_sha", "")
        services = scenario.get("services", {})
        for service_name, service_data in services.items():
            for i, deploy in enumerate(service_data.get("deploys", [])):
                if _synth_sha(service_name, i, deploy) == sha:
                    return json.dumps(
                        {
                            "commit_sha": sha,
                            "service": service_name,
                            "author": deploy.get("author"),
                            "timestamp": deploy.get("timestamp"),
                            "diff_summary": deploy.get("diff_summary"),
                            "diff_patch": deploy.get("diff_patch", ""),
                        },
                        indent=2,
                    )
        return json.dumps({"error": f"commit '{sha}' not found"}, indent=2)

    if op == "get_file":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        snippets = data.get("code_snippets", {})
        path = args.get("path") or args.get("file_path")
        if path:
            if path in snippets:
                return f"=== {path} ===\n{snippets[path]}"
            return (
                f"File '{path}' not found for {svc}. "
                f"Available: {list(snippets.keys())}"
            )
        if not snippets:
            return f"No code snippets available for {svc}"
        lines = []
        for name, content in snippets.items():
            lines.append(f"=== {name} ===")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).rstrip()

    if op == "list_files":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        snippets = data.get("code_snippets", {})
        return json.dumps(list(snippets.keys()), indent=2)

    if op == "get_blame":
        data = _service_data(scenario, svc)
        if data is None:
            return unknown_service(svc, scenario)
        path = args.get("path", "")
        deploys = data.get("deploys", [])
        if not deploys:
            return f"No blame info for {svc}:{path} (no deploys recorded)"
        latest = deploys[-1]
        return (
            f"{path} last touched by {latest.get('author', '?')} "
            f"at {latest.get('timestamp', '?')}: {latest.get('diff_summary', '')}"
        )

    if op == "open_pr":
        pr = {
            "target_repo": args.get("target_repo", ""),
            "base_branch": args.get("base_branch", "main"),
            "head_branch": args.get("head_branch", ""),
            "title": args.get("title", ""),
            "summary": args.get("summary", ""),
            "diff_patch": args.get("diff_patch", ""),
            "reviewers": args.get("reviewers", []),
        }
        state.opened_prs.append(pr)
        pr_url = f"https://repohub.internal/{pr['target_repo']}/pull/{len(state.opened_prs)}"
        return json.dumps({"pr_url": pr_url, **pr}, indent=2)

    # Tier A2 — simulated CI quality gate report (Snyk / Sonar / Raven / coverage).
    # Scenarios populate `scenario["repohub"]["ci_state"][<target_repo>]`.
    if op == "ci_check":
        target_repo = args.get("target_repo", "")
        ci = (scenario.get("repohub") or {}).get("ci_state") or {}
        report = ci.get(target_repo)
        if report is None:
            # Default synthesised report — healthy repo, so the agent sees a
            # baseline and scenarios can opt-in to breakage by populating ci_state.
            report = {
                "target_repo": target_repo,
                "coverage_pct": 92.0,
                "coverage_delta_pct": 0.0,
                "snyk": {"high": 0, "medium": 0, "low": 0, "waived": []},
                "sonar": {
                    "code_smells": 3, "bugs": 0, "vulnerabilities": 0,
                    "quality_gate": "passed",
                },
                "raven": {"secrets_found": 0, "iac_misconfigs": 0},
                "gate_status": "passed",
                "note": "no scenario override — default healthy profile",
            }
        # Record evidence so a future policy / grader can check whether the
        # agent actually queried CI before acting.
        checked = state.evidence_collected.setdefault("__ci_checked__", [])
        if target_repo not in checked:
            checked.append(target_repo)
        return json.dumps(report, indent=2)

    if op == "list_pr_history":
        target_repo = args.get("target_repo", "")
        history = (scenario.get("repohub") or {}).get("pr_history") or {}
        prs = history.get(target_repo, [])
        if not prs:
            return f"No PR history found for {target_repo}."
        rows = []
        for pr in prs:
            gate = pr.get("ci_gate_status", "?")
            waived = ",".join(pr.get("waived_findings", []) or []) or "-"
            rows.append(
                f"#{pr.get('number','?')} {pr.get('title','')} "
                f"[merged={pr.get('merged', False)} gate={gate} waived={waived}] "
                f"by {pr.get('author','?')} at {pr.get('merged_at','?')}"
            )
        return "\n".join(rows)

    return f"Unknown op for repohub: {op}"
