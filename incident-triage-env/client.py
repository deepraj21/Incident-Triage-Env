from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import (
    ActionResult,
    AlertInfo,
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
)


class IncidentTriageClient(EnvClient[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]):

    def _step_payload(self, action: IncidentTriageAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[IncidentTriageObservation]:
        obs_data = payload.get("observation", payload)

        alert_data = obs_data.get("alert", {})
        alert = AlertInfo(**alert_data)

        result = None
        result_data = obs_data.get("result")
        if result_data:
            result = ActionResult(**result_data)

        obs = IncidentTriageObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            alert=alert,
            result=result,
            steps_remaining=obs_data.get("steps_remaining", 0),
            services_investigated=obs_data.get("services_investigated", []),
        )

        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IncidentTriageState:
        return IncidentTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            steps_remaining=payload.get("steps_remaining", 0),
            services_investigated=payload.get("services_investigated", []),
            actions_taken=payload.get("actions_taken", []),
            evidence_collected=payload.get("evidence_collected", {}),
        )
