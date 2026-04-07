import math
from typing import Any

from openenv.core.rubrics.base import Rubric

from models import IncidentTriageAction, IncidentTriageObservation, RootCauseCategory, Remediation

# Evidence tier reward values
EVIDENCE_REWARDS = {
    "direct_evidence": 0.12,
    "causal_chain": 0.08,
    "contextual": 0.04,
    "none": 0.0,
}

# Failure families for partial category credit
FAILURE_FAMILIES = {
    RootCauseCategory.BAD_DEPLOY: "change-induced",
    RootCauseCategory.CONFIG_CHANGE: "change-induced",
    RootCauseCategory.RESOURCE_EXHAUSTION: "capacity-related",
    RootCauseCategory.TRAFFIC_SPIKE: "capacity-related",
    RootCauseCategory.DEPENDENCY_FAILURE: "infrastructure",
    RootCauseCategory.DATA_CORRUPTION: "infrastructure",
}

# Partial remediation mappings: (submitted_remediation, actual_category) -> partial credit
PARTIAL_REMEDIATIONS = {
    (Remediation.RESTART_SERVICE, RootCauseCategory.BAD_DEPLOY): True,
    (Remediation.SCALE_UP, RootCauseCategory.TRAFFIC_SPIKE): True,
    (Remediation.RESTART_SERVICE, RootCauseCategory.RESOURCE_EXHAUSTION): True,
    (Remediation.FAILOVER_TO_BACKUP, RootCauseCategory.DEPENDENCY_FAILURE): True,
}


class InvestigationRubric(Rubric):
    """Per-step trajectory rewards: information gain + strategy + red herring resistance."""

    def __init__(self, scenario_data: dict):
        super().__init__()
        self._evidence_labels = scenario_data.get("evidence_labels", {})
        self._ground_truth = scenario_data.get("ground_truth", {})
        self._services = scenario_data.get("services", {})
        self._anomalous_services = set(self._ground_truth.get("anomalous_services", []))
        self._causal_chain = set(self._ground_truth.get("causal_chain", []))
        self._red_herrings = set(self._ground_truth.get("red_herrings", []))
        # Red herring penalty only applies to services in red_herrings NOT in causal_chain
        self._pure_red_herrings = self._red_herrings - self._causal_chain

        # Internal tracking state
        self._queried_pairs: set[tuple[str, str]] = set()  # (service, action_type)
        self._service_query_counts: dict[str, int] = {}
        self._services_with_found_evidence: set[str] = set()  # services where we found direct/causal evidence
        self._has_causal_evidence = False
        self._investigated_anomalous: set[str] = set()

    def reset(self) -> None:
        self._queried_pairs.clear()
        self._service_query_counts.clear()
        self._services_with_found_evidence.clear()
        self._has_causal_evidence = False
        self._investigated_anomalous.clear()

    def forward(self, action: Any, observation: Any) -> float:
        if not isinstance(action, IncidentTriageAction):
            return 0.0

        if action.action_type == "submit_diagnosis":
            return self._score_submit(action)

        service = action.service
        action_type = action.action_type

        reward = 0.0

        # 1. Information Gain
        pair = (service, action_type)
        if pair in self._queried_pairs:
            # Redundant query
            reward += 0.0
        else:
            self._queried_pairs.add(pair)
            tier = self._evidence_labels.get(service, {}).get(action_type, "none")
            reward += EVIDENCE_REWARDS.get(tier, 0.0)

            if tier in ("direct_evidence", "causal_chain"):
                self._services_with_found_evidence.add(service)
                self._has_causal_evidence = True

        # Track query count per service
        self._service_query_counts[service] = self._service_query_counts.get(service, 0) + 1
        count = self._service_query_counts[service]

        # Repeated redundant queries penalty (3rd+ query to same service)
        if count >= 3 and pair in self._queried_pairs:
            # Only penalize if this specific pair was already seen (true redundancy)
            # Actually, spec says 3rd+ total query to same service, any action type
            pass
        if count >= 3:
            reward += -0.05

        # 2. Investigation Strategy
        strategy_reward = self._compute_strategy_reward(service, action_type)
        reward += strategy_reward

        # Track anomalous services investigated
        if service in self._anomalous_services:
            self._investigated_anomalous.add(service)

        # 3. Red Herring Resistance
        red_herring_penalty = self._compute_red_herring_penalty(service)
        reward += red_herring_penalty

        return reward

    def _compute_strategy_reward(self, service: str, action_type: str) -> float:
        # Following dependency graph: querying upstream/downstream of an anomalous service already investigated
        if self._investigated_anomalous:
            for investigated in self._investigated_anomalous:
                investigated_data = self._services.get(investigated, {})
                deps = set(investigated_data.get("dependencies", []))
                dependents = set(investigated_data.get("dependents", []))
                neighbors = deps | dependents
                if service in neighbors and service in self._anomalous_services:
                    return 0.05

        # Cross-referencing: different action_type on a service where agent found anomalies
        if service in self._investigated_anomalous:
            existing_types = {at for (s, at) in self._queried_pairs if s == service}
            if action_type not in existing_types:
                return 0.03

        # Ruling out: querying a non-anomalous service for the first time
        if service not in self._anomalous_services and self._service_query_counts.get(service, 0) <= 1:
            return 0.02

        return 0.0

    def _compute_red_herring_penalty(self, service: str) -> float:
        if service not in self._pure_red_herrings:
            return 0.0

        # Only penalize 3+ steps investigating a red herring after agent has causal evidence
        count = self._service_query_counts.get(service, 0)
        if count >= 3 and self._has_causal_evidence:
            return -0.03

        return 0.0

    def _score_submit(self, action: IncidentTriageAction) -> float:
        # Red herring diagnosis penalty
        if action.root_cause_service in self._pure_red_herrings:
            return -0.08
        if action.root_cause_service in self._red_herrings:
            return -0.08
        return 0.0


class DiagnosisScorer:
    """Terminal grader score: accuracy + efficiency + penalties. Not a framework Rubric."""

    def score(self, ground_truth: dict, diagnosis: dict, episode_state: Any, scenario_services: dict | None = None) -> dict:
        if not diagnosis:
            return {"score": 0.0, "breakdown": {}, "diagnosis_submitted": False}

        breakdown = {}

        # 4a. Root cause service (0.0 to 0.20)
        gt_service = ground_truth["root_cause_service"]
        submitted_service = diagnosis.get("root_cause_service", "")
        causal_chain = ground_truth.get("causal_chain", [])

        if submitted_service == gt_service:
            breakdown["root_cause_service"] = 0.20
        elif submitted_service in causal_chain:
            # Check if one hop away
            try:
                gt_idx = causal_chain.index(gt_service)
                sub_idx = causal_chain.index(submitted_service)
                if abs(gt_idx - sub_idx) == 1:
                    breakdown["root_cause_service"] = 0.10
                else:
                    breakdown["root_cause_service"] = 0.0
            except ValueError:
                breakdown["root_cause_service"] = 0.0
        else:
            breakdown["root_cause_service"] = 0.0

        # 4b. Root cause category (0.0 to 0.15)
        gt_category = RootCauseCategory(ground_truth["root_cause_category"])
        submitted_category_str = diagnosis.get("root_cause_category", "")
        try:
            submitted_category = RootCauseCategory(submitted_category_str)
        except ValueError:
            submitted_category = None

        if submitted_category == gt_category:
            breakdown["root_cause_category"] = 0.15
        elif submitted_category and FAILURE_FAMILIES.get(submitted_category) == FAILURE_FAMILIES.get(gt_category):
            breakdown["root_cause_category"] = 0.07
        else:
            breakdown["root_cause_category"] = 0.0

        # 4c. Remediation (0.0 to 0.10)
        gt_remediation = Remediation(ground_truth["remediation"])
        submitted_remediation_str = diagnosis.get("remediation", "")
        try:
            submitted_remediation = Remediation(submitted_remediation_str)
        except ValueError:
            submitted_remediation = None

        if submitted_remediation == gt_remediation:
            breakdown["remediation"] = 0.10
        elif submitted_remediation and (submitted_remediation, gt_category) in PARTIAL_REMEDIATIONS:
            breakdown["remediation"] = 0.05
        else:
            breakdown["remediation"] = 0.0

        # 4d. Evidence quality (0.0 to 0.10)
        evidence_services = set(ground_truth.get("causal_chain", []))
        investigated = set(getattr(episode_state, "services_investigated", []))
        visited_causal = evidence_services & investigated
        if len(evidence_services) > 0:
            ratio = len(visited_causal) / len(evidence_services)
        else:
            ratio = 1.0

        if ratio >= 0.8:
            breakdown["evidence_quality"] = 0.10
        elif ratio >= 0.5:
            breakdown["evidence_quality"] = 0.05
        else:
            breakdown["evidence_quality"] = 0.0

        # 5. Efficiency (0.0 to 0.15)
        step_budget = getattr(episode_state, "steps_remaining", 0) + getattr(episode_state, "step_count", 0)
        steps_remaining = getattr(episode_state, "steps_remaining", 0)
        if step_budget > 0:
            breakdown["efficiency"] = round(0.15 * math.sqrt(steps_remaining / step_budget), 4)
        else:
            breakdown["efficiency"] = 0.0

        # Penalties
        penalties = 0.0
        step_count = getattr(episode_state, "step_count", 0)

        # Shotgun diagnosis: submitting within first 3 steps
        if step_count <= 3:
            penalties -= 0.15

        # Circular investigation: any service queried 4+ times
        actions_taken = getattr(episode_state, "actions_taken", [])
        service_counts: dict[str, int] = {}
        for a in actions_taken:
            s = a.get("service", "")
            if s and a.get("action_type") != "submit_diagnosis":
                service_counts[s] = service_counts.get(s, 0) + 1
        if any(c >= 4 for c in service_counts.values()):
            penalties -= 0.10

        # Destructive remediation mismatch
        if submitted_remediation == Remediation.ROLLBACK_DEPLOY and scenario_services:
            gt_service_data = scenario_services.get(gt_service, {})
            if not gt_service_data.get("deploys"):
                penalties -= 0.10

        breakdown["penalties"] = round(penalties, 4)

        total = sum(v for k, v in breakdown.items())
        # Clamp strictly within (0, 1) — Phase 2 evaluator rejects 0.0 or 1.0 exactly.
        score = max(0.001, min(0.999, total))

        return {
            "score": round(score, 4),
            "breakdown": breakdown,
            "diagnosis_submitted": True,
        }
