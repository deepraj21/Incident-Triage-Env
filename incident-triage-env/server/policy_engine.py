"""PolicyEngine — declarative business-rule enforcement.

Rules are evaluated every step AFTER app dispatch. Each match produces a
reward delta (penalty or bonus) and appends to `state.policy_violations` and
`state.policy_events`.

Rule schema (all fields optional except `id` and either `penalty` or `bonus`):
    {
      "id": "require_page_before_rollback",
      "when": {
        "app": "system",                     # action.app.value
        "op": "submit_diagnosis",            # action.op
        "args_match": {"remediation": "rollback_deploy"}
      },
      "require_prior": {"app": "chatops", "op": "page_oncall"},
      "require_prior_within_steps": 5,       # only recent priors count
      "forbidden_if": {"scenario_tag": "change_freeze"},
      "max_occurrences": 2,                  # 3rd match triggers penalty
      "penalty": -0.25,                      # reward delta when violated
      "bonus":   0.10                        # reward delta when satisfied
    }

`penalty` applies on violation; `bonus` applies on satisfaction (matching
`when` *without* triggering `forbidden_if` / prior-missing / occurrence-cap
failures). Either or both can be present. If only `penalty` is set, the rule
is pure stick; if only `bonus`, pure carrot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class PolicyEngine:
    def __init__(self, rules: List[Dict[str, Any]], tags: Optional[List[str]] = None) -> None:
        self._rules: List[Dict[str, Any]] = list(rules or [])
        self._tags: set[str] = set(tags or [])

    # --------------------------------------------------------------- evaluate

    def evaluate(
        self,
        action: Any,
        state: Any,
        clock: Any = None,
    ) -> Tuple[float, List[str]]:
        """Apply all matching rules. Returns (total_delta, violations_this_step).

        `state.policy_violations` and `state.policy_events` are appended to
        in place; the returned violations list mirrors what should be surfaced
        on the current observation.
        """
        total_delta = 0.0
        violations_this_step: List[str] = []
        step_now = getattr(state, "step_count", 0)

        for rule in self._rules:
            if not self._matches_when(rule.get("when", {}), action):
                continue

            rule_id = rule.get("id", "<unnamed>")
            penalty = float(rule.get("penalty", 0.0))
            bonus = float(rule.get("bonus", 0.0))
            violated = False
            reason = ""

            # 1. forbidden_if — scenario-tag driven hard-no (e.g. change freeze).
            forbid = rule.get("forbidden_if") or {}
            if forbid.get("scenario_tag") and forbid["scenario_tag"] in self._tags:
                violated = True
                reason = f"forbidden under tag '{forbid['scenario_tag']}'"

            # 2. require_prior — a specific earlier action must exist.
            if not violated:
                req_prior = rule.get("require_prior")
                if req_prior and not self._has_prior(
                    req_prior,
                    state,
                    within_steps=rule.get("require_prior_within_steps"),
                    current_step=step_now,
                ):
                    violated = True
                    reason = (
                        f"missing prior {req_prior.get('app')}.{req_prior.get('op')}"
                    )

            # 3. max_occurrences — soft cap on repeated matches of `when`.
            cap = rule.get("max_occurrences")
            if not violated and cap is not None:
                prior_matches = self._count_prior_matches(rule.get("when", {}), state)
                # Current action already appended to state.actions_taken by env.step,
                # so prior_matches includes it. Violation when count > cap.
                if prior_matches > int(cap):
                    violated = True
                    reason = f"exceeded max_occurrences={cap} (seen={prior_matches})"

            delta = penalty if violated else bonus
            if delta == 0.0 and not violated:
                # Pure-stick rule that wasn't violated — no-op.
                continue

            total_delta += delta
            event = {
                "rule_id": rule_id,
                "violated": violated,
                "reason": reason,
                "delta": delta,
                "at_step": step_now,
                "action_app": getattr(getattr(action, "app", None), "value", None),
                "action_op": getattr(action, "op", None),
            }
            state.policy_events.append(event)
            if violated:
                state.policy_violations.append(rule_id)
                violations_this_step.append(rule_id)

        return total_delta, violations_this_step

    # --------------------------------------------------------------- summary

    def summary(self, state: Any) -> Dict[str, Any]:
        """Terminal aggregation used by Phase-5 PolicyComplianceScorer."""
        events = getattr(state, "policy_events", [])
        total_penalty = sum(e["delta"] for e in events if e["delta"] < 0)
        total_bonus = sum(e["delta"] for e in events if e["delta"] > 0)
        violation_ids = [e["rule_id"] for e in events if e["violated"]]
        # compliance_score: 1.0 if no rules fired violated; penalties scale it down.
        # Clip to [0, 1]; penalty of -0.5 cumulative maps to 0.5, -1.0+ maps to 0.0.
        compliance = max(0.0, min(1.0, 1.0 + total_penalty))
        return {
            "violations": violation_ids,
            "num_violations": len(violation_ids),
            "total_penalty": total_penalty,
            "total_bonus": total_bonus,
            "compliance_score": compliance,
        }

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _matches_when(when: Dict[str, Any], action: Any) -> bool:
        app_val = getattr(getattr(action, "app", None), "value", None)
        op_val = getattr(action, "op", None)
        if when.get("app") and when["app"] != app_val:
            return False
        if when.get("op") and when["op"] != op_val:
            return False
        for key, expected in (when.get("args_match") or {}).items():
            actual = (action.args or {}).get(key) if hasattr(action, "args") else None
            # Accept enum values as matching their `.value`.
            actual_val = getattr(actual, "value", actual)
            if actual_val != expected:
                return False
        return True

    @staticmethod
    def _has_prior(
        req: Dict[str, Any],
        state: Any,
        within_steps: Optional[int] = None,
        current_step: int = 0,
    ) -> bool:
        req_app = req.get("app")
        req_op = req.get("op")
        for past in getattr(state, "actions_taken", []) or []:
            if past.get("app") != req_app or past.get("op") != req_op:
                continue
            if past.get("at_step") == current_step:
                # Skip the current action itself — "prior" means strictly earlier.
                continue
            if within_steps is not None:
                past_step = past.get("at_step", 0)
                if current_step - past_step > int(within_steps):
                    continue
            return True
        return False

    @classmethod
    def _count_prior_matches(cls, when: Dict[str, Any], state: Any) -> int:
        count = 0
        for past in getattr(state, "actions_taken", []) or []:
            # Reconstruct a shim action-shaped object with minimal attrs.
            shim = _PastActionView(past)
            if cls._matches_when(when, shim):
                count += 1
        return count


class _PastActionView:
    """Adapter so `_matches_when` can read a past action dict uniformly."""

    __slots__ = ("app", "op", "args")

    def __init__(self, past: Dict[str, Any]) -> None:
        app_val = past.get("app", "")
        self.app = type("AppV", (), {"value": app_val})()
        self.op = past.get("op", "")
        self.args = past.get("args", {}) or {}
