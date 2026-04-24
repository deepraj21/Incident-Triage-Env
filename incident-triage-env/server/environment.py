"""IncidentTriageEnv — OpenEnv environment with multi-app action surface.

Round 2 / Phase 2:
- `step()` routes every non-SYSTEM action through `server/apps/APP_DISPATCH`.
- `WorldClock` advances sim-time one tick per `step()`, and `EventQueue`
  fires due timeline events into the *per-episode* scenario copy.
- Visible events surface on the Observation (`world_events`) so the agent can
  react to mid-episode changes; silent events still mutate the world but are
  discoverable only via re-polling (stealth-regression story).
- Legacy Round-1 actions (`{action_type, service, ...}`) are auto-migrated to
  `{app, op, args}` by `IncidentTriageAction`'s before-validator.
- PolicyEngine hook is staged for Phase 3.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from models import (
    ActionResult,
    AppName,
    IncidentTriageAction,
    IncidentTriageObservation,
    IncidentTriageState,
    WorldEvent,
)
from server.apps import APP_DISPATCH
from server.episode_store import store_episode
from server.event_queue import EventQueue
from server.grader import CompositeScorer, DiagnosisScorer, InvestigationRubric
from server.policy_engine import PolicyEngine
from server.scenario_loader import ScenarioLoader
from server.world_clock import WorldClock


class IncidentTriageEnv(
    Environment[IncidentTriageAction, IncidentTriageObservation, IncidentTriageState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._loader = ScenarioLoader()
        self._diagnosis_scorer = DiagnosisScorer()
        self._composite_scorer = CompositeScorer()
        super().__init__(rubric=None)

        self._state: IncidentTriageState = IncidentTriageState()
        self._current_scenario: dict = {}
        self._current_ground_truth: dict = {}
        self._step_budget: int = 0
        self._alert_info = None
        self._clock: Optional[WorldClock] = None
        self._event_queue: Optional[EventQueue] = None
        self._policy_engine: Optional[PolicyEngine] = None

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy_single_service_failure",
        **kwargs: Any,
    ) -> IncidentTriageObservation:
        ep_id = episode_id or str(uuid.uuid4())
        # Deep copy so EventQueue mutations never leak across episodes.
        # seed — when provided, drives scenario_variants perturbations; same
        # (task_id, seed) → byte-stable scenario, so training rollouts are
        # reproducible and hero scenarios are multiplied ~20× for free.
        scenario = copy.deepcopy(self._loader.get_scenario(task_id, seed=seed))
        self._current_scenario = scenario
        self._current_ground_truth = scenario["ground_truth"]
        self._step_budget = scenario["step_budget"]
        self._alert_info = self._loader.get_alert(task_id)

        self._clock = WorldClock.from_scenario(scenario)
        self._event_queue = EventQueue(scenario.get("timeline", []))
        self._policy_engine = PolicyEngine(
            rules=scenario.get("policies", []),
            tags=scenario.get("tags", []),
        )

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
            scenario_tags=list(scenario.get("tags", [])),
        )

        return IncidentTriageObservation(
            done=False,
            reward=None,
            alert=self._alert_info,
            result=None,
            steps_remaining=self._step_budget,
            services_investigated=[],
            sim_time=self._clock.now_iso(),
            world_events=[],
            pending_world_events=self._event_queue.pending_count(),
        )

    # ------------------------------------------------------------------ step

    def step(
        self,
        action: IncidentTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentTriageObservation:
        self._state.step_count += 1
        self._state.steps_remaining -= 1

        # Advance the world BEFORE dispatch so the agent's action observes
        # the post-tick state (new logs, new alerts, etc.).
        assert self._clock is not None and self._event_queue is not None
        self._clock.advance(one_step=True)
        visible_events = self._event_queue.fire_due(
            self._clock, self._current_scenario, self._state
        )

        legacy_type = action.action_type
        service = action.service
        self._state.actions_taken.append(
            {
                "action_type": legacy_type,
                "service": service or "",
                "app": action.app.value,
                "op": action.op,
                "args": dict(action.args),
                "at_step": self._state.step_count,
            }
        )

        if action.app == AppName.SYSTEM and action.op == "submit_diagnosis":
            return self._handle_submit(action, legacy_type, visible_events)

        result_text = self._dispatch(action)
        policy_delta, policy_violations = self._run_policy(action)

        if service and service not in self._state.services_investigated:
            self._state.services_investigated.append(service)
        if service:
            evidence_types = self._state.evidence_collected.setdefault(service, [])
            if legacy_type not in evidence_types:
                evidence_types.append(legacy_type)

        obs = IncidentTriageObservation(
            done=False,
            reward=None,
            alert=self._alert_info,
            result=ActionResult(action_type=legacy_type, data=result_text),
            steps_remaining=self._state.steps_remaining,
            services_investigated=list(self._state.services_investigated),
            sim_time=self._clock.now_iso(),
            world_events=[WorldEvent(**self._to_world_event(e)) for e in visible_events],
            pending_world_events=self._event_queue.pending_count(),
            policy_violations_this_step=policy_violations,
            policy_delta_this_step=policy_delta,
        )

        rubric_reward = self._apply_rubric(action, obs)
        obs.reward = (rubric_reward or 0.0) + policy_delta

        if self._state.steps_remaining <= 0:
            obs.done = True
            self._store_episode_result(diagnosis=None)

        return obs

    # ------------------------------------------------------------------ dispatch

    def _dispatch(self, action: IncidentTriageAction) -> str:
        handler = APP_DISPATCH.get(action.app.value)
        if handler is None:
            return f"No dispatcher registered for app '{action.app.value}'"
        try:
            return handler(
                action.op,
                dict(action.args),
                self._current_scenario,
                self._state,
                clock=self._clock,
            )
        except ValueError as exc:
            return str(exc)
        except KeyError as exc:
            return f"Missing required arg: {exc}"

    # ------------------------------------------------------------------ submit

    def _handle_submit(
        self,
        action: IncidentTriageAction,
        legacy_type: str,
        visible_events: list[dict],
    ) -> IncidentTriageObservation:
        diagnosis = {
            "root_cause_service": action.root_cause_service,
            "root_cause_category": (
                action.root_cause_category.value if action.root_cause_category else ""
            ),
            "remediation": action.remediation.value if action.remediation else "",
        }
        if action.args.get("pr_proposal") is not None:
            diagnosis["pr_proposal"] = action.args["pr_proposal"]
        if action.args.get("blast_radius") is not None:
            diagnosis["blast_radius"] = action.args["blast_radius"]

        # Policy check fires on submit too — catches rules like
        # "require page_oncall before rollback remediation".
        policy_delta, policy_violations = self._run_policy(action)

        # Phase 5: composite grader aggregates diagnosis + policy + blast + pr heads.
        policy_summary = (
            self._policy_engine.summary(self._state) if self._policy_engine else None
        )
        grader_result = self._composite_scorer.score(
            self._current_ground_truth,
            diagnosis,
            self._state,
            scenario_services=self._current_scenario.get("services", {}),
            policy_summary=policy_summary,
        )

        assert self._clock is not None and self._event_queue is not None
        obs = IncidentTriageObservation(
            done=True,
            reward=None,
            alert=self._alert_info,
            result=ActionResult(
                action_type=legacy_type,
                data=f"Diagnosis submitted. Grader score: {grader_result['score']:.2f}",
            ),
            steps_remaining=self._state.steps_remaining,
            services_investigated=list(self._state.services_investigated),
            sim_time=self._clock.now_iso(),
            world_events=[WorldEvent(**self._to_world_event(e)) for e in visible_events],
            pending_world_events=self._event_queue.pending_count(),
            policy_violations_this_step=policy_violations,
            policy_delta_this_step=policy_delta,
            grader_breakdown={
                "score": grader_result["score"],
                "heads": grader_result.get("heads", {}),
                "effective_weights": grader_result.get("effective_weights", {}),
            },
        )

        rubric_reward = self._apply_rubric(action, obs)
        obs.reward = (rubric_reward or 0.0) + policy_delta
        self._store_episode_result(diagnosis=diagnosis, grader_score=grader_result)
        return obs

    # ------------------------------------------------------------------ policy

    def _run_policy(self, action: IncidentTriageAction) -> tuple[float, list[str]]:
        if self._policy_engine is None:
            return 0.0, []
        return self._policy_engine.evaluate(action, self._state, self._clock)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _to_world_event(applied: dict) -> dict:
        return {
            "event": applied.get("event", ""),
            "service": applied.get("service"),
            "at_step": applied.get("at_step"),
            "payload": applied.get("payload", {}),
        }

    def _store_episode_result(self, diagnosis: dict | None, grader_score: dict | None = None) -> None:
        policy_summary = (
            self._policy_engine.summary(self._state)
            if self._policy_engine is not None
            else None
        )
        store_episode(
            self._state.episode_id,
            {
                "task_id": self._state.task_id,
                "ground_truth": self._current_ground_truth,
                "diagnosis": diagnosis,
                "episode_state": {
                    "episode_id": self._state.episode_id,
                    "step_count": self._state.step_count,
                    "steps_remaining": self._state.steps_remaining,
                    "services_investigated": list(self._state.services_investigated),
                    "actions_taken": list(self._state.actions_taken),
                    "event_log": list(self._state.event_log),
                    "scenario_tags": list(self._state.scenario_tags),
                    "policy_events": list(self._state.policy_events),
                    "policy_violations": list(self._state.policy_violations),
                },
                "grader_score": grader_score,
                "policy_summary": policy_summary,
                "diagnosis_submitted": diagnosis is not None,
            },
        )

    @property
    def state(self) -> IncidentTriageState:
        return self._state
