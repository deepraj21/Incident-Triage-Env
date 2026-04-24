"""WorldClock — sim-time advancement for Phase 2 Dynamic World Engine.

One tick per environment step. Scenarios may declare:
    "clock": {"start": "2026-03-26T14:32:00Z", "step_seconds": 30}

If absent, we fall back to the scenario's alert timestamp (or epoch) and
30 s/step. This keeps Round-1 fixtures working unchanged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


_DEFAULT_STEP_SECONDS = 30


def _parse_iso(ts: str) -> datetime:
    # tolerate trailing Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class WorldClock:
    def __init__(self, start_iso: str, step_seconds: int = _DEFAULT_STEP_SECONDS) -> None:
        self._start = _parse_iso(start_iso)
        self._t = self._start
        self._step_seconds = step_seconds
        self._step = 0

    @classmethod
    def from_scenario(cls, scenario: Dict[str, Any]) -> "WorldClock":
        clock_cfg = scenario.get("clock") or {}
        start = clock_cfg.get("start") or scenario.get("alert", {}).get(
            "timestamp"
        ) or "1970-01-01T00:00:00Z"
        step_seconds = int(clock_cfg.get("step_seconds", _DEFAULT_STEP_SECONDS))
        return cls(start_iso=start, step_seconds=step_seconds)

    def advance(self, one_step: bool = True) -> None:
        if one_step:
            self._t += timedelta(seconds=self._step_seconds)
            self._step += 1

    def now_iso(self) -> str:
        # Normalize UTC-offset suffix back to Z for readability.
        s = self._t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return s

    @property
    def step(self) -> int:
        return self._step

    @property
    def step_seconds(self) -> int:
        return self._step_seconds
