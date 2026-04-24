"""UATSim — pre-production / UAT sign-off simulation (Tier A1).

Models the pre-deploy gate that production bugs often slip past when teams
skip UAT or waive a failing stage. Scenarios populate:

    scenario["uatsim"] = {
      "stages": [
        {"id": "smoke",      "name": "smoke-tests",     "required": true,  "status": "passed", "run_at": "..."},
        {"id": "regression", "name": "regression-suite", "required": true, "status": "skipped", "reason": "release-captain override"},
        {"id": "load",       "name": "load-test",       "required": false, "status": "passed"}
      ],
      "signoffs": [
        {"role": "qa-lead", "user": "mira",  "signed": true},
        {"role": "product", "user": "priya", "signed": false}
      ],
      "uat_records": {
        "<service-name>": {
          "passed_stages": ["smoke"],
          "skipped_stages": ["regression"],
          "signoff_complete": false,
          "deploy_shipped_anyway": true
        }
      }
    }

Ops are read-only — the incident is already live when the episode starts, so the
agent's job is *detective*: discover that UAT was bypassed and fold that into
the diagnosis / PR proposal.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def _uat_block(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return scenario.get("uatsim") or {}


def dispatch(
    op: str,
    args: Dict[str, Any],
    scenario: Dict[str, Any],
    state: Any,
    clock: Optional[Any] = None,
) -> str:
    uat = _uat_block(scenario)

    if op == "list_stages":
        stages = uat.get("stages", [])
        if not stages:
            return "No UAT stages defined for this scenario."
        # Compact single-line-per-stage view so the LLM can scan quickly.
        rows = []
        for s in stages:
            req = "REQUIRED" if s.get("required") else "optional"
            status = s.get("status", "?").upper()
            note = f" ({s.get('reason')})" if s.get("reason") else ""
            rows.append(
                f"- [{status}] {s.get('id', '?')} — {s.get('name', '')} [{req}]{note}"
            )
        return "\n".join(rows)

    if op == "get_stage":
        stage_id = args.get("stage_id")
        for s in uat.get("stages", []):
            if s.get("id") == stage_id:
                return json.dumps(s, indent=2)
        return json.dumps({"error": f"stage '{stage_id}' not found"}, indent=2)

    if op == "get_signoff_status":
        signoffs = uat.get("signoffs", [])
        if not signoffs:
            return "No signoff records for this release."
        missing = [s for s in signoffs if not s.get("signed")]
        summary = {
            "total_signoffs": len(signoffs),
            "signed": len(signoffs) - len(missing),
            "missing": [
                {"role": s.get("role"), "user": s.get("user")} for s in missing
            ],
            "complete": len(missing) == 0,
        }
        return json.dumps(summary, indent=2)

    if op == "check_uat_record":
        service = args.get("service")
        records = uat.get("uat_records", {})
        record = records.get(service)
        if record is None:
            # No explicit record → report neutral (treat as "no gate failures")
            return json.dumps({
                "service": service,
                "passed_stages": [],
                "skipped_stages": [],
                "signoff_complete": True,
                "deploy_shipped_anyway": False,
                "note": "no UAT record found — service may not require UAT",
            }, indent=2)
        payload = {"service": service, **record}
        # Mark the agent's state so graders / policies can see they checked.
        checked = state.evidence_collected.setdefault("__uat_checked__", [])
        if service not in checked:
            checked.append(service)
        return json.dumps(payload, indent=2)

    return f"Unknown op for uatsim: {op}"
