"""TicketDesk — Jira-shaped ticket tracker app."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _tickets(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(scenario.get("ticketdesk", {}).get("tickets", []))


def dispatch(
    op: str,
    args: Dict[str, Any],
    scenario: Dict[str, Any],
    state: Any,
    clock: Optional[Any] = None,
) -> str:
    tickets = _tickets(scenario)

    if op == "search_tickets":
        q = (args.get("q") or args.get("query") or "").strip().lower()
        if not q:
            summary = [{"id": t.get("id"), "title": t.get("title", "")} for t in tickets]
            return json.dumps(summary, indent=2) if summary else "No tickets"
        matches = [t for t in tickets if q in json.dumps(t).lower()]
        return json.dumps(matches, indent=2) if matches else f"No tickets match '{q}'"

    if op == "get_ticket":
        tid = args.get("ticket_id", "")
        found = next((t for t in tickets if t.get("id") == tid), None)
        if found is None:
            return json.dumps({"error": f"ticket '{tid}' not found"}, indent=2)
        return json.dumps(found, indent=2)

    if op == "create_incident":
        new_id = f"INC-{len(state.created_tickets) + 1:04d}"
        record = {
            "id": new_id,
            "title": args.get("title", ""),
            "body": args.get("body", ""),
            "severity": args.get("severity", "major"),
            "status": "open",
        }
        state.created_tickets.append(record)
        return json.dumps(record, indent=2)

    if op == "link_pr":
        return (
            f"linked PR {args.get('pr_url', '')} to ticket "
            f"{args.get('ticket_id', '')}"
        )

    if op == "add_comment":
        message = args.get("message", "")
        return f"comment added to {args.get('ticket_id', '')}: {message[:140]}"

    return f"Unknown op for ticketdesk: {op}"
