"""Registry of per-app dispatchers used by IncidentTriageEnv.step().

Each dispatcher has signature:
    dispatch(op, args, scenario, state, clock) -> str

`clock` is `None` in Phase 1 and will be populated with a WorldClock in Phase 2.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from server.apps import alerthub, chatops, obsly, repohub, ticketdesk, uatsim

Dispatcher = Callable[..., str]

APP_DISPATCH: Dict[str, Dispatcher] = {
    "alerthub": alerthub.dispatch,
    "obsly": obsly.dispatch,
    "repohub": repohub.dispatch,
    "ticketdesk": ticketdesk.dispatch,
    "chatops": chatops.dispatch,
    "uatsim": uatsim.dispatch,
}


__all__ = [
    "APP_DISPATCH", "alerthub", "obsly", "repohub",
    "ticketdesk", "chatops", "uatsim",
]
