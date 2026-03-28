from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from openenv.core.env_server import Action, Observation, State


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


ActionType = Literal[
    "query_logs",
    "query_metrics",
    "check_deploys",
    "trace_dependencies",
    "check_status",
    "inspect_code",
    "submit_diagnosis",
]


class IncidentTriageAction(Action):
    action_type: ActionType
    service: Optional[str] = None
    # query_logs options
    severity: Optional[Literal["error", "warn", "info", "debug"]] = None
    keyword: Optional[str] = None
    # query_metrics options
    metric: Optional[Literal["latency_p99", "error_rate", "cpu", "memory", "connections"]] = None
    # inspect_code options
    file_path: Optional[str] = None
    # submit_diagnosis fields
    root_cause_service: Optional[str] = None
    root_cause_category: Optional[RootCauseCategory] = None
    remediation: Optional[Remediation] = None

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action_type == "submit_diagnosis":
            missing = []
            if not self.root_cause_service:
                missing.append("root_cause_service")
            if not self.root_cause_category:
                missing.append("root_cause_category")
            if not self.remediation:
                missing.append("remediation")
            if missing:
                raise ValueError(
                    f"submit_diagnosis requires: {', '.join(missing)}"
                )
        else:
            if not self.service:
                raise ValueError(
                    f"action_type '{self.action_type}' requires 'service'"
                )
        return self


class AlertInfo(BaseModel):
    service: str
    message: str
    severity: Literal["critical", "warning"]
    timestamp: str


class ActionResult(BaseModel):
    action_type: str
    data: str


class IncidentTriageObservation(Observation):
    alert: AlertInfo
    result: Optional[ActionResult] = None
    steps_remaining: int
    services_investigated: List[str] = Field(default_factory=list)


class IncidentTriageState(State):
    task_id: str = ""
    steps_remaining: int = 0
    services_investigated: List[str] = Field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_collected: Dict[str, List[str]] = Field(default_factory=dict)
