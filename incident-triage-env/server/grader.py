import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

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


# =============================================================================
# Phase 5 — multi-head scorers
# =============================================================================


def _f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _window_overlap_fraction(pred_start, pred_end, gold_start, gold_end) -> float:
    """Return [0,1]: IoU of two time windows; 0 if either window invalid."""
    if None in (pred_start, pred_end, gold_start, gold_end):
        return 0.0
    if pred_end <= pred_start or gold_end <= gold_start:
        return 0.0
    latest_start = max(pred_start, gold_start)
    earliest_end = min(pred_end, gold_end)
    inter = max(0.0, (earliest_end - latest_start).total_seconds())
    union_start = min(pred_start, gold_start)
    union_end = max(pred_end, gold_end)
    union = (union_end - union_start).total_seconds()
    return inter / union if union > 0 else 0.0


class PRProposalScorer:
    """Grades an agent's structured PRProposal against scenario ground-truth."""

    def score(self, correct_pr: dict, pr_proposal: Optional[dict]) -> dict:
        if not correct_pr:
            return {"score": None, "breakdown": {}, "submitted": pr_proposal is not None,
                    "applicable": False}
        if not pr_proposal:
            return {"score": 0.0, "breakdown": {}, "submitted": False, "applicable": True}

        breakdown: dict[str, float] = {}

        # target_repo (0.30) — exact match, or substring if ground truth is a suffix
        gt_repo = (correct_pr.get("target_repo") or "").strip().lower()
        sub_repo = (pr_proposal.get("target_repo") or "").strip().lower()
        if gt_repo and gt_repo == sub_repo:
            breakdown["target_repo"] = 0.30
        elif gt_repo and gt_repo in sub_repo:
            breakdown["target_repo"] = 0.15
        else:
            breakdown["target_repo"] = 0.0

        # touched_files (0.35) — F1 of files referenced in diff_patch or summary
        gt_files = {f.lower() for f in correct_pr.get("touched_files", [])}
        haystack = " ".join([
            pr_proposal.get("diff_patch", "") or "",
            pr_proposal.get("summary", "") or "",
            pr_proposal.get("title", "") or "",
        ]).lower()
        pred_files = {f for f in gt_files if f in haystack}
        # We only measure recall here because the agent's diff may reference
        # extra files; penalising extra files would discourage honest reporting.
        recall = (len(pred_files) / len(gt_files)) if gt_files else 1.0
        breakdown["touched_files"] = round(0.35 * recall, 4)

        # keyword coverage (0.25)
        gt_keywords = [k.lower() for k in correct_pr.get("keywords", [])]
        title_summary = " ".join([
            pr_proposal.get("title", "") or "",
            pr_proposal.get("summary", "") or "",
        ]).lower()
        if gt_keywords:
            hits = sum(1 for k in gt_keywords if k in title_summary)
            breakdown["keywords"] = round(0.25 * hits / len(gt_keywords), 4)
        else:
            breakdown["keywords"] = 0.25

        # structural validity (0.10)
        struct = 0.0
        if (pr_proposal.get("title") or "").strip():
            struct += 0.03
        if (pr_proposal.get("head_branch") or "").strip():
            struct += 0.03
        if len((pr_proposal.get("summary") or "").strip()) >= 20:
            struct += 0.04
        breakdown["structural"] = round(struct, 4)

        total = sum(breakdown.values())
        return {
            "score": round(max(0.0, min(1.0, total)), 4),
            "breakdown": breakdown,
            "submitted": True,
            "applicable": True,
        }


class BlastRadiusScorer:
    """Grades the agent's BlastRadiusReport against ground-truth impact."""

    def score(self, correct_br: dict, blast_radius: Optional[dict]) -> dict:
        if not correct_br:
            return {"score": None, "breakdown": {}, "submitted": blast_radius is not None,
                    "applicable": False}
        if not blast_radius:
            return {"score": 0.0, "breakdown": {}, "submitted": False, "applicable": True}

        breakdown: dict[str, float] = {}

        # affected_services F1 (0.40)
        gt_svcs = {s.lower() for s in correct_br.get("affected_services", [])}
        sub_svcs = {s.lower() for s in blast_radius.get("affected_services", []) or []}
        breakdown["affected_services"] = round(0.40 * _f1(sub_svcs, gt_svcs), 4)

        # missed_regions F1 (0.20)
        gt_regions = {r.lower() for r in correct_br.get("missed_regions", [])}
        sub_regions = {r.lower() for r in blast_radius.get("missed_regions", []) or []}
        if not gt_regions and not sub_regions:
            breakdown["missed_regions"] = 0.20
        else:
            breakdown["missed_regions"] = round(0.20 * _f1(sub_regions, gt_regions), 4)

        # requests_failed order-of-magnitude tolerance (0.20)
        gt_req = correct_br.get("estimated_requests_failed", 0) or 0
        sub_req = blast_radius.get("estimated_requests_failed", 0) or 0
        if gt_req <= 0:
            breakdown["requests_failed"] = 0.20 if sub_req == 0 else 0.10
        elif sub_req <= 0:
            breakdown["requests_failed"] = 0.0
        else:
            diff = abs(math.log10(max(1, sub_req)) - math.log10(max(1, gt_req)))
            if diff < 0.5:
                breakdown["requests_failed"] = 0.20
            elif diff < 1.0:
                breakdown["requests_failed"] = 0.12
            elif diff < 2.0:
                breakdown["requests_failed"] = 0.05
            else:
                breakdown["requests_failed"] = 0.0

        # outage window IoU (0.20)
        iou = _window_overlap_fraction(
            _parse_iso(blast_radius.get("outage_window_start")),
            _parse_iso(blast_radius.get("outage_window_end")),
            _parse_iso(correct_br.get("outage_window_start")),
            _parse_iso(correct_br.get("outage_window_end")),
        )
        breakdown["outage_window"] = round(0.20 * iou, 4)

        total = sum(breakdown.values())
        return {
            "score": round(max(0.0, min(1.0, total)), 4),
            "breakdown": breakdown,
            "submitted": True,
            "applicable": True,
        }


class PolicyComplianceScorer:
    """Wraps PolicyEngine.summary() compliance_score into the composite schema."""

    def score(self, policy_summary: Optional[dict]) -> dict:
        if not policy_summary:
            return {"score": 1.0, "breakdown": {"compliance_score": 1.0}, "applicable": False}
        compliance = float(policy_summary.get("compliance_score", 1.0))
        return {
            "score": round(max(0.0, min(1.0, compliance)), 4),
            "breakdown": {
                "compliance_score": compliance,
                "num_violations": policy_summary.get("num_violations", 0),
                "total_penalty": policy_summary.get("total_penalty", 0.0),
                "total_bonus": policy_summary.get("total_bonus", 0.0),
            },
            "applicable": True,
        }


class CompositeScorer:
    """Aggregates diagnosis + policy + blast-radius + PR heads into a single score.

    Default weights: diagnosis 0.40, policy 0.20, blast 0.20, pr 0.20.
    When a head is inapplicable (no ground-truth for that head, or the agent
    did not submit that payload but the head is required), its weight is
    redistributed proportionally across the remaining applicable heads so the
    final score stays comparable across scenarios.
    """

    DEFAULT_WEIGHTS = {"diagnosis": 0.40, "policy": 0.20, "blast": 0.20, "pr": 0.20}

    def __init__(self, weights: Optional[dict] = None):
        self._weights = dict(weights or self.DEFAULT_WEIGHTS)
        self._diag = DiagnosisScorer()
        self._pr = PRProposalScorer()
        self._blast = BlastRadiusScorer()
        self._policy = PolicyComplianceScorer()

    def score(
        self,
        ground_truth: dict,
        diagnosis: dict,
        episode_state: Any,
        scenario_services: dict | None = None,
        policy_summary: Optional[dict] = None,
    ) -> dict:
        diag_result = self._diag.score(ground_truth, diagnosis, episode_state, scenario_services)
        pr_result = self._pr.score(
            ground_truth.get("correct_pr", {}),
            diagnosis.get("pr_proposal") if diagnosis else None,
        )
        blast_result = self._blast.score(
            ground_truth.get("correct_blast_radius", {}),
            diagnosis.get("blast_radius") if diagnosis else None,
        )
        policy_result = self._policy.score(policy_summary)

        heads = {
            "diagnosis": diag_result,
            "policy": policy_result,
            "blast": blast_result,
            "pr": pr_result,
        }

        # Decide which heads contribute. Diagnosis and policy always apply;
        # blast/pr apply only if ground-truth provides them.
        applicable = {
            "diagnosis": diag_result.get("diagnosis_submitted", False),
            "policy": policy_result.get("applicable", True),
            "blast": blast_result.get("applicable", False),
            "pr": pr_result.get("applicable", False),
        }

        # If no diagnosis was submitted, the whole score is zero (terminal failure).
        if not applicable["diagnosis"]:
            return {
                "score": 0.0,
                "heads": heads,
                "weights": dict(self._weights),
                "effective_weights": {k: 0.0 for k in self._weights},
                "diagnosis_submitted": False,
            }

        # Redistribute weights of inapplicable heads.
        active_weights = {
            k: self._weights[k] for k in self._weights if applicable.get(k, False)
        }
        total_active = sum(active_weights.values())
        if total_active <= 0:
            effective = dict(self._weights)
        else:
            effective = {k: w / total_active for k, w in active_weights.items()}

        score = 0.0
        for head_name, w in effective.items():
            head_score = heads[head_name].get("score") or 0.0
            score += w * head_score

        # Hold strict (0,1) for OpenEnv compat (inherits from diag scorer constraint).
        score = max(0.001, min(0.999, score))
        return {
            "score": round(score, 4),
            "heads": heads,
            "weights": dict(self._weights),
            "effective_weights": {k: round(v, 4) for k, v in effective.items()},
            "diagnosis_submitted": True,
        }
