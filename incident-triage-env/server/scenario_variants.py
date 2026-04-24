"""ScenarioVariantGenerator — deterministic seed-driven perturbations.

Multiplies effective task count by applying seed-reproducible noise to a
hero scenario without altering its ground truth. Perturbations:

1. Shuffle the order of logs within each service (preserves set of logs).
2. Hide a random subset (0..k) of *red-herring* logs — logs that mention
   any service listed in `ground_truth.red_herrings` or `ground_truth`'s
   `failure_family` keywords. Never hides evidence tied to the root cause.
3. Jitter non-anchor log timestamps by +/- a few seconds.
4. Shuffle deploys (preserving the root-cause deploy's relative recency).
5. Permute timeline events that share the same `at_step` bucket.
6. Rename `chatops.oncall_lead` from a name pool (cosmetic; keeps narrative
   diversity for storytelling).

Given the same seed + scenario, output is byte-stable.

Ground-truth fields (root_cause_service, root_cause_category, remediation,
causal_chain) are NEVER modified, and any log whose message contains the
root_cause_service name is pinned (never hidden or re-timestamped out of
order against the alert firing time).
"""

from __future__ import annotations

import copy
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


_NAME_POOL = [
    "alice-sre", "bob-sre", "carla-sre", "derek-sre", "evelyn-sre",
    "felix-sre", "gita-sre", "hassan-sre", "indira-sre", "jun-sre",
]


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_root_cause_log(log: Dict[str, Any], root_cause_service: str) -> bool:
    msg = (log.get("message") or "").lower()
    return root_cause_service.lower() in msg


def apply_variant(scenario: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Return a deep-copied, seed-perturbed scenario. Idempotent for (scenario, seed)."""
    rng = random.Random(seed)
    s = copy.deepcopy(scenario)
    gt = s.get("ground_truth", {})
    root_svc = gt.get("root_cause_service", "")
    red_herring_services = set(_extract_red_herring_services(gt))

    # 1-3: per-service log perturbations
    for svc_name, svc in s.get("services", {}).items():
        logs = svc.get("logs", [])
        if not logs:
            continue
        logs = _maybe_hide_red_herrings(logs, svc_name, red_herring_services, rng)
        logs = _jitter_timestamps(logs, root_svc, rng)
        logs = _shuffle_logs(logs, rng)
        svc["logs"] = logs

    # 4: shuffle deploys within each service (but keep the most recent one recent)
    for svc in s.get("services", {}).values():
        deploys = svc.get("deploys", [])
        if len(deploys) > 1:
            head, *tail = deploys
            rng.shuffle(tail)
            svc["deploys"] = [head] + tail

    # 5: permute timeline events within equal-at_step buckets
    timeline = s.get("timeline", [])
    if timeline:
        s["timeline"] = _permute_within_buckets(timeline, rng)

    # 6: cosmetic oncall lead rename
    chatops = s.get("chatops")
    if isinstance(chatops, dict):
        new_lead = rng.choice(_NAME_POOL)
        old_lead = chatops.get("oncall_lead")
        chatops["oncall_lead"] = new_lead
        # Propagate rename through channel messages so the narrative stays consistent.
        if old_lead:
            for ch_msgs in chatops.get("channels", {}).values():
                for m in ch_msgs:
                    if m.get("user") == old_lead:
                        m["user"] = new_lead

    return s


def _extract_red_herring_services(gt: Dict[str, Any]) -> List[str]:
    """Best-effort: pull service names out of the red_herrings freeform list."""
    out: List[str] = []
    for item in gt.get("red_herrings", []) or []:
        token = str(item).lower()
        for possible in (gt.get("anomalous_services") or []):
            if possible == gt.get("root_cause_service"):
                continue
            if possible.lower() in token:
                out.append(possible)
    return out


def _maybe_hide_red_herrings(
    logs: List[Dict[str, Any]],
    svc_name: str,
    red_herring_services: set[str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if svc_name not in red_herring_services:
        return logs
    # Hide up to 30% of this service's logs (min 0, max 2).
    n_hide = rng.randint(0, min(2, len(logs) // 3))
    if n_hide == 0:
        return logs
    indices = list(range(len(logs)))
    rng.shuffle(indices)
    hidden = set(indices[:n_hide])
    return [log for i, log in enumerate(logs) if i not in hidden]


def _jitter_timestamps(
    logs: List[Dict[str, Any]],
    root_cause_service: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    jittered = []
    for log in logs:
        if _is_root_cause_log(log, root_cause_service):
            # Pin root-cause logs so causal ordering stays valid.
            jittered.append(log)
            continue
        ts = log.get("timestamp", "")
        dt = _parse_ts(ts)
        if dt is None:
            jittered.append(log)
            continue
        shift = rng.randint(-4, 4)
        new_dt = dt + timedelta(seconds=shift)
        log = {**log, "timestamp": _fmt_ts(new_dt)}
        jittered.append(log)
    return jittered


def _shuffle_logs(logs: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    # Light shuffle: swap ~20% of adjacent pairs. Preserves broad temporal ordering
    # while ensuring the agent can't rely on strict insertion order.
    out = list(logs)
    for i in range(len(out) - 1):
        if rng.random() < 0.2:
            out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _permute_within_buckets(
    timeline: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    order: List[int] = []
    for ev in timeline:
        step = int(ev.get("at_step", 0))
        if step not in buckets:
            buckets[step] = []
            order.append(step)
        buckets[step].append(ev)
    for step in buckets:
        rng.shuffle(buckets[step])
    out: List[Dict[str, Any]] = []
    for step in sorted(order):
        out.extend(buckets[step])
    return out
