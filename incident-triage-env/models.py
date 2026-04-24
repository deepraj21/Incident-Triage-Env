"""
Typed Pydantic models for the Incident Triage OpenEnv environment.

Round 2 (Phase 1) — multi-app refactor:
- IncidentTriageAction is now a tagged union over five enterprise apps
  (AlertHub, Obsly, RepoHub, TicketDesk, ChatOps) plus a SYSTEM namespace
  for submit_diagnosis.
- Action shape on the wire is `{app, op, args}` with flexible args: Dict.
- Per-op required-arg validation runs after construction.
- Round-1 flat actions (`{action_type, service, ...}`) are auto-migrated to
  the new shape by a `mode="before"` validator, so existing callers keep
  working during the Phase 1→3 transition.
- Convenience properties (.action_type, .service, .metric, ...) let the
  existing InvestigationRubric / DiagnosisScorer consume new-shape actions
  without rewrites.

Phase-5 grader models are staged here too (PRProposal, BlastRadiusReport,
FinalDiagnosis) — Phase 1 ships them so downstream phases can consume.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openenv.core.env_server import Action, Observation, State


# =============================================================================
# Diagnosis enums (shared with legacy)
# =============================================================================


class RootCauseCategory(str, Enum):
    BAD_DEPLOY = "bad_deploy"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIG_CHANGE = "config_change"
    TRAFFIC_SPIKE = "traffic_spike"
    DATA_CORRUPTION = "data_corruption"


class Remediation(str, Enum):
    ROLLBACK_DEPLOY = "rollback_deploy"
    SCALE_UP = "scale_up"
    RESTART_SERVICE = "restart_service"
    FIX_CONFIG = "fix_config"
    ENABLE_RATE_LIMITING = "enable_rate_limiting"
    FAILOVER_TO_BACKUP = "failover_to_backup"


# =============================================================================
# App + Op enums (Round 2 multi-app action surface)
# =============================================================================


class AppName(str, Enum):
    ALERTHUB = "alerthub"
    OBSLY = "obsly"
    REPOHUB = "repohub"
    TICKETDESK = "ticketdesk"
    CHATOPS = "chatops"
    SYSTEM = "system"  # reserved for submit_diagnosis


class AlertHubOp(str, Enum):
    LIST_ALERTS = "list_alerts"
    GET_ALERT = "get_alert"
    ACK_ALERT = "ack_alert"


class ObslyOp(str, Enum):
    QUERY_LOGS = "query_logs"
    QUERY_METRIC = "query_metric"
    GET_TRACE = "get_trace"
    LIST_DASHBOARDS = "list_dashboards"
    # Extended ops (superset of Round-1 surface so legacy actions map 1:1)
    CHECK_STATUS = "check_status"
    TRACE_DEPENDENCIES = "trace_dependencies"


class RepoHubOp(str, Enum):
    RECENT_COMMITS = "recent_commits"
    GET_DIFF = "get_diff"
    GET_FILE = "get_file"
    GET_BLAME = "get_blame"
    OPEN_PR = "open_pr"
    LIST_FILES = "list_files"  # free-win: needed by stealth_regression scenario


class TicketDeskOp(str, Enum):
    SEARCH_TICKETS = "search_tickets"
    GET_TICKET = "get_ticket"
    CREATE_INCIDENT = "create_incident"
    LINK_PR = "link_pr"
    ADD_COMMENT = "add_comment"  # free-win: agent can record findings


class ChatOpsOp(str, Enum):
    POST_UPDATE = "post_update"
    READ_CHANNEL = "read_channel"
    PAGE_ONCALL = "page_oncall"


class SystemOp(str, Enum):
    SUBMIT_DIAGNOSIS = "submit_diagnosis"


_ALLOWED_OPS: Dict[AppName, set] = {
    AppName.ALERTHUB: {e.value for e in AlertHubOp},
    AppName.OBSLY: {e.value for e in ObslyOp},
    AppName.REPOHUB: {e.value for e in RepoHubOp},
    AppName.TICKETDESK: {e.value for e in TicketDeskOp},
    AppName.CHATOPS: {e.value for e in ChatOpsOp},
    AppName.SYSTEM: {e.value for e in SystemOp},
}


# =============================================================================
# Structured PR + Blast-Radius payloads (carried inside submit_diagnosis args,
# and also used by repohub.open_pr)
# =============================================================================


class PRProposal(BaseModel):
    """Structured pull-request proposal produced by the agent as part of its fix."""

    target_repo: str = Field(..., description="e.g. 'fintra/payments-service'")
    base_branch: str = "main"
    head_branch: str = Field(..., description="e.g. 'fix/npe-guest-checkout'")
    title: str = Field(..., max_length=120)
    summary: str = Field("", max_length=1000, description="what & why")
    diff_patch: str = Field("", description="unified diff text")
    reviewers: List[str] = Field(default_factory=list)


class BlastRadiusReport(BaseModel):
    """Agent's reconstruction of incident impact (who/what/when was affected)."""

    affected_services: List[str] = Field(default_factory=list)
    affected_customer_segments: List[str] = Field(default_factory=list)
    outage_window_start: Optional[str] = None  # ISO-8601
    outage_window_end: Optional[str] = None  # ISO-8601
    estimated_requests_failed: int = 0
    missed_regions: List[str] = Field(default_factory=list)


class FinalDiagnosis(BaseModel):
    """Terminal payload for system.submit_diagnosis — graded by Phase 5 scorers."""

    root_cause_service: str
    root_cause_category: RootCauseCategory
    remediation: Remediation
    pr_proposal: Optional[PRProposal] = None
    blast_radius: Optional[BlastRadiusReport] = None


# =============================================================================
# Legacy (Round-1) → (app, op) migration map
# =============================================================================


_LEGACY_ACTION_TYPE_MAP: Dict[str, Tuple[AppName, str]] = {
    "query_logs": (AppName.OBSLY, "query_logs"),
    "query_metrics": (AppName.OBSLY, "query_metric"),
    "check_deploys": (AppName.REPOHUB, "recent_commits"),
    "trace_dependencies": (AppName.OBSLY, "trace_dependencies"),
    "check_status": (AppName.OBSLY, "check_status"),
    "inspect_code": (AppName.REPOHUB, "get_file"),
    "submit_diagnosis": (AppName.SYSTEM, "submit_diagnosis"),
}


# Reverse lookup for the `.action_type` convenience property — used by the
# rubric and other Round-1 consumers that key off legacy names.
_NEW_TO_LEGACY_ACTION_TYPE: Dict[Tuple[str, str], str] = {
    (app.value, op): legacy for legacy, (app, op) in _LEGACY_ACTION_TYPE_MAP.items()
}


_LEGACY_ARG_KEYS = (
    "service",
    "severity",
    "keyword",
    "metric",
    "file_path",
    "root_cause_service",
    "root_cause_category",
    "remediation",
    "pr_proposal",
    "blast_radius",
)


def _legacy_to_new_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Round-1 `{action_type, ...}` dict to `{app, op, args}`."""
    legacy_type = data.get("action_type")
    if legacy_type not in _LEGACY_ACTION_TYPE_MAP:
        raise ValueError(
            f"Unknown legacy action_type '{legacy_type}'. "
            f"Expected one of {sorted(_LEGACY_ACTION_TYPE_MAP)}."
        )
    app, op = _LEGACY_ACTION_TYPE_MAP[legacy_type]

    args: Dict[str, Any] = {}
    for key in _LEGACY_ARG_KEYS:
        value = data.get(key)
        if value is None:
            continue
        args[key] = value

    # Legacy inspect_code used `file_path`; new repohub.get_file uses `path`.
    if legacy_type == "inspect_code" and "file_path" in args:
        args["path"] = args.pop("file_path")

    return {"app": app.value, "op": op, "args": args}


# =============================================================================
# Per-op required-arg contracts (typed-core, flexible-edge)
# =============================================================================


_REQUIRED_ARGS: Dict[Tuple[str, str], List[str]] = {
    (AppName.ALERTHUB.value, "list_alerts"): [],
    (AppName.ALERTHUB.value, "get_alert"): ["alert_id"],
    (AppName.ALERTHUB.value, "ack_alert"): ["alert_id"],
    (AppName.OBSLY.value, "query_logs"): ["service"],
    (AppName.OBSLY.value, "query_metric"): ["service"],  # metric optional; defaults to error_rate
    (AppName.OBSLY.value, "get_trace"): ["trace_id"],
    (AppName.OBSLY.value, "list_dashboards"): [],
    (AppName.OBSLY.value, "check_status"): ["service"],
    (AppName.OBSLY.value, "trace_dependencies"): ["service"],
    (AppName.REPOHUB.value, "recent_commits"): ["service"],
    (AppName.REPOHUB.value, "get_diff"): ["commit_sha"],
    (AppName.REPOHUB.value, "get_file"): ["service"],  # path optional
    (AppName.REPOHUB.value, "get_blame"): ["service", "path"],
    (AppName.REPOHUB.value, "open_pr"): ["target_repo", "title", "head_branch"],
    (AppName.REPOHUB.value, "list_files"): ["service"],
    (AppName.TICKETDESK.value, "search_tickets"): [],
    (AppName.TICKETDESK.value, "get_ticket"): ["ticket_id"],
    (AppName.TICKETDESK.value, "create_incident"): ["title"],
    (AppName.TICKETDESK.value, "link_pr"): ["ticket_id", "pr_url"],
    (AppName.TICKETDESK.value, "add_comment"): ["ticket_id", "message"],
    (AppName.CHATOPS.value, "post_update"): ["channel", "message"],
    (AppName.CHATOPS.value, "read_channel"): ["channel"],
    (AppName.CHATOPS.value, "page_oncall"): ["role"],
    (AppName.SYSTEM.value, "submit_diagnosis"): [
        "root_cause_service",
        "root_cause_category",
        "remediation",
    ],
}


def _validate_op_args(app: AppName, op: str, args: Dict[str, Any]) -> None:
    missing = [
        k for k in _REQUIRED_ARGS.get((app.value, op), []) if args.get(k) in (None, "")
    ]
    if missing:
        raise ValueError(
            f"{app.value}.{op} missing required args: {', '.join(missing)}"
        )


# =============================================================================
# Main IncidentTriageAction
# =============================================================================


class IncidentTriageAction(Action):
    """Typed tagged-union action: {app, op, args}.

    The wire schema is intentionally `args: Dict[str, Any]` so the policy LLM
    can emit a flexible JSON shape. Per-op required-args are validated in
    `_validate_op_args` and individual dispatchers parse `args` into their own
    typed view (see `server/apps/*`).
    """

    model_config = ConfigDict(extra="ignore")

    app: AppName
    op: str
    args: Dict[str, Any] = Field(default_factory=dict)

    # ---- Construction hooks ----------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_shape(cls, data: Any) -> Any:
        if isinstance(data, dict) and "action_type" in data and "app" not in data:
            return _legacy_to_new_dict(data)
        return data

    @model_validator(mode="after")
    def _validate_op_for_app(self) -> "IncidentTriageAction":
        allowed = _ALLOWED_OPS.get(self.app, set())
        if self.op not in allowed:
            raise ValueError(
                f"op '{self.op}' not valid for app '{self.app.value}'. "
                f"Allowed: {sorted(allowed)}"
            )
        _validate_op_args(self.app, self.op, self.args or {})
        return self

    # ---- Legacy convenience accessors -----------------------------------
    # These let Round-1 consumers (InvestigationRubric, environment.py, tests)
    # keep working without rewriting every call site. Remove after Phase 3
    # once all downstream code has migrated to `app`/`op`/`args`.

    @property
    def action_type(self) -> str:
        return _NEW_TO_LEGACY_ACTION_TYPE.get((self.app.value, self.op), self.op)

    @property
    def service(self) -> Optional[str]:
        return self.args.get("service")

    @property
    def severity(self) -> Optional[str]:
        return self.args.get("severity")

    @property
    def keyword(self) -> Optional[str]:
        return self.args.get("keyword")

    @property
    def metric(self) -> Optional[str]:
        return self.args.get("metric")

    @property
    def file_path(self) -> Optional[str]:
        return self.args.get("path") or self.args.get("file_path")

    @property
    def root_cause_service(self) -> Optional[str]:
        return self.args.get("root_cause_service")

    @property
    def root_cause_category(self) -> Optional[RootCauseCategory]:
        v = self.args.get("root_cause_category")
        if v is None:
            return None
        if isinstance(v, RootCauseCategory):
            return v
        try:
            return RootCauseCategory(v)
        except ValueError:
            return None

    @property
    def remediation(self) -> Optional[Remediation]:
        v = self.args.get("remediation")
        if v is None:
            return None
        if isinstance(v, Remediation):
            return v
        try:
            return Remediation(v)
        except ValueError:
            return None


# =============================================================================
# Observation / State (unchanged from Round 1, with Phase 2-3 state fields
# preallocated so downstream code doesn't need another migration)
# =============================================================================


class AlertInfo(BaseModel):
    service: str
    message: str
    severity: Literal["critical", "warning"]
    timestamp: str


class ActionResult(BaseModel):
    # For Round-1 compat this is still called `action_type`; post-Phase-1 it
    # holds the legacy action-type name (or the raw `op` for new-only actions).
    action_type: str
    data: str


class WorldEvent(BaseModel):
    """Visible mid-episode world change surfaced to the agent this step."""

    event: str  # "new_log" | "slo_burn" | "new_deploy" | "oncall_handoff" | "new_alert"
    service: Optional[str] = None
    at_step: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class IncidentTriageObservation(Observation):
    alert: AlertInfo
    result: Optional[ActionResult] = None
    steps_remaining: int
    services_investigated: List[str] = Field(default_factory=list)
    # Phase 2 — Dynamic World Engine surface.
    sim_time: Optional[str] = None
    world_events: List[WorldEvent] = Field(default_factory=list)
    pending_world_events: int = 0
    # Phase 3 — per-step policy feedback for RL credit assignment.
    policy_violations_this_step: List[str] = Field(default_factory=list)
    policy_delta_this_step: float = 0.0
    # Phase 5 — per-head grader breakdown on terminal step (for storytelling heatmap).
    grader_breakdown: Optional[Dict[str, Any]] = None


class IncidentTriageState(State):
    task_id: str = ""
    steps_remaining: int = 0
    services_investigated: List[str] = Field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_collected: Dict[str, List[str]] = Field(default_factory=dict)

    # Multi-app side-effects (populated by app dispatchers; consumed by
    # PolicyEngine in Phase 3 and by terminal scorers in Phase 5).
    acked_alerts: List[str] = Field(default_factory=list)
    posted_updates: List[Dict[str, Any]] = Field(default_factory=list)
    paged_roles: List[str] = Field(default_factory=list)
    opened_prs: List[Dict[str, Any]] = Field(default_factory=list)
    created_tickets: List[Dict[str, Any]] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)

    # Phase 2 — Dynamic World Engine.
    scenario_tags: List[str] = Field(default_factory=list)
    event_log: List[Dict[str, Any]] = Field(default_factory=list)

    # Phase 3 — PolicyEngine audit trail (every rule firing this episode).
    policy_events: List[Dict[str, Any]] = Field(default_factory=list)


# Public re-exports used by callers / tests.
__all__ = [
    "ActionResult",
    "AlertInfo",
    "AppName",
    "AlertHubOp",
    "BlastRadiusReport",
    "ChatOpsOp",
    "FinalDiagnosis",
    "IncidentTriageAction",
    "IncidentTriageObservation",
    "IncidentTriageState",
    "ObslyOp",
    "PRProposal",
    "Remediation",
    "RepoHubOp",
    "RootCauseCategory",
    "SystemOp",
    "TicketDeskOp",
    "WorldEvent",
]
