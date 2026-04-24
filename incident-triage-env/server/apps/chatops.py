"""ChatOps — Slack-shaped messaging app."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def dispatch(
    op: str,
    args: Dict[str, Any],
    scenario: Dict[str, Any],
    state: Any,
    clock: Optional[Any] = None,
) -> str:
    chatops_data = scenario.get("chatops", {})
    channels = chatops_data.get("channels", {})

    if op == "post_update":
        record = {
            "channel": args.get("channel", ""),
            "message": args.get("message", ""),
        }
        state.posted_updates.append(record)
        return f"posted to #{record['channel']}: {record['message'][:140]}"

    if op == "read_channel":
        ch = args.get("channel", "")
        messages = channels.get(ch, [])
        if not messages:
            return f"channel #{ch} is empty"
        return json.dumps(messages, indent=2)

    if op == "page_oncall":
        role = args.get("role", "")
        if not role:
            return "page_oncall requires a 'role'"
        state.paged_roles.append(role)
        lead = chatops_data.get("oncall_lead", "unknown")
        return f"paged {role} (current oncall lead: {lead})"

    return f"Unknown op for chatops: {op}"
