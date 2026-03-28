import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from models import (
    ActionResult,
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
)
from server.episode_store import store_episode
from server.grader import DiagnosisScorer, InvestigationRubric
from server.scenario_loader import ScenarioLoader


class IncidentTriageEnv(Environment[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._loader = ScenarioLoader()
        self._diagnosis_scorer = DiagnosisScorer()
        # Rubric will be set per-episode in reset()
        super().__init__(rubric=None)

        self._state = IncidentTriageState()
        self._current_scenario: dict = {}
        self._current_ground_truth: dict = {}
        self._step_budget: int = 0
        self._alert_info = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy_single_service_failure",
        **kwargs: Any,
    ) -> IncidentTriageObservation:
        ep_id = episode_id or str(uuid.uuid4())
        scenario = self._loader.get_scenario(task_id)
        self._current_scenario = scenario
        self._current_ground_truth = scenario["ground_truth"]
        self._step_budget = scenario["step_budget"]
        self._alert_info = self._loader.get_alert(task_id)

        # Set up rubric for this episode
        self.rubric = InvestigationRubric(scenario)
        self._reset_rubric()

        self._state = IncidentTriageState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            steps_remaining=self._step_budget,
            services_investigated=[],
            actions_taken=[],
            evidence_collected={},
        )

        return IncidentTriageObservation(
            done=False,
            reward=None,
            alert=self._alert_info,
            result=None,
            steps_remaining=self._step_budget,
            services_investigated=[],
        )

    def step(
        self,
        action: IncidentTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentTriageObservation:
        self._state.step_count += 1
        self._state.steps_remaining -= 1

        task_id = self._state.task_id

        # Track action
        action_record = {"action_type": action.action_type, "service": action.service or ""}
        self._state.actions_taken.append(action_record)

        if action.action_type == "submit_diagnosis":
            return self._handle_submit(action)

        # Dispatch to scenario loader
        service = action.service
        result_text = self._dispatch_action(task_id, action)

        # Track investigation
        if service and service not in self._state.services_investigated:
            self._state.services_investigated.append(service)
        if service:
            if service not in self._state.evidence_collected:
                self._state.evidence_collected[service] = []
            if action.action_type not in self._state.evidence_collected[service]:
                self._state.evidence_collected[service].append(action.action_type)

        action_result = ActionResult(action_type=action.action_type, data=result_text)

        obs = IncidentTriageObservation(
            done=False,
            reward=None,
            alert=self._alert_info,
            result=action_result,
            steps_remaining=self._state.steps_remaining,
            services_investigated=list(self._state.services_investigated),
        )

        # Apply rubric for per-step reward
        obs.reward = self._apply_rubric(action, obs)

        # Check budget exhaustion
        if self._state.steps_remaining <= 0:
            obs.done = True
            self._store_episode_result(diagnosis=None)

        return obs

    def _dispatch_action(self, task_id: str, action: IncidentTriageAction) -> str:
        service = action.service
        try:
            if action.action_type == "query_logs":
                return self._loader.query_logs(task_id, service, severity=action.severity, keyword=action.keyword)
            elif action.action_type == "query_metrics":
                return self._loader.query_metrics(task_id, service, action.metric or "error_rate")
            elif action.action_type == "check_deploys":
                return self._loader.check_deploys(task_id, service)
            elif action.action_type == "trace_dependencies":
                return self._loader.trace_dependencies(task_id, service)
            elif action.action_type == "check_status":
                return self._loader.check_status(task_id, service)
            elif action.action_type == "inspect_code":
                return self._loader.inspect_code(task_id, service, file_path=action.file_path)
            else:
                return f"Unknown action type: {action.action_type}"
        except ValueError as e:
            return str(e)

    def _handle_submit(self, action: IncidentTriageAction) -> IncidentTriageObservation:
        diagnosis = {
            "root_cause_service": action.root_cause_service,
            "root_cause_category": action.root_cause_category.value if action.root_cause_category else "",
            "remediation": action.remediation.value if action.remediation else "",
        }

        grader_result = self._diagnosis_scorer.score(
            self._current_ground_truth,
            diagnosis,
            self._state,
            scenario_services=self._current_scenario.get("services", {}),
        )

        obs = IncidentTriageObservation(
            done=True,
            reward=None,
            alert=self._alert_info,
            result=ActionResult(
                action_type="submit_diagnosis",
                data=f"Diagnosis submitted. Grader score: {grader_result['score']:.2f}",
            ),
            steps_remaining=self._state.steps_remaining,
            services_investigated=list(self._state.services_investigated),
        )

        # Apply rubric for submit action (red herring penalty)
        obs.reward = self._apply_rubric(action, obs)

        self._store_episode_result(diagnosis=diagnosis, grader_score=grader_result)

        return obs

    def _store_episode_result(self, diagnosis: dict | None, grader_score: dict | None = None):
        store_episode(self._state.episode_id, {
            "task_id": self._state.task_id,
            "ground_truth": self._current_ground_truth,
            "diagnosis": diagnosis,
            "episode_state": {
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "steps_remaining": self._state.steps_remaining,
                "services_investigated": list(self._state.services_investigated),
                "actions_taken": list(self._state.actions_taken),
            },
            "grader_score": grader_score,
            "diagnosis_submitted": diagnosis is not None,
        })

    @property
    def state(self) -> IncidentTriageState:
        return self._state
