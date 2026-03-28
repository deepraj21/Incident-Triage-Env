import os

from fastapi import HTTPException
from pydantic import BaseModel

from openenv.core.env_server import create_app

from models import IncidentTriageAction, IncidentTriageObservation, IncidentTriageState
from server.environment import IncidentTriageEnv
from server.episode_store import get_episode
from server.scenario_loader import ScenarioLoader

app = create_app(
    IncidentTriageEnv,
    IncidentTriageAction,
    IncidentTriageObservation,
    env_name="incident_triage_env",
)


# Custom endpoints

_loader = ScenarioLoader()


@app.get("/tasks")
async def list_tasks():
    tasks = _loader.list_tasks()
    action_schema = IncidentTriageAction.model_json_schema()
    return {"tasks": tasks, "action_schema": action_schema}


class GraderRequest(BaseModel):
    episode_id: str


class GraderResponse(BaseModel):
    task_id: str
    score: float
    breakdown: dict
    diagnosis_submitted: bool


@app.post("/grader", response_model=GraderResponse)
async def grade_episode(req: GraderRequest):
    episode = get_episode(req.episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode {req.episode_id} not found")
    return GraderResponse(
        task_id=episode["task_id"],
        score=episode["grader_score"]["score"] if episode.get("grader_score") else 0.0,
        breakdown=episode["grader_score"]["breakdown"] if episode.get("grader_score") else {},
        diagnosis_submitted=episode.get("diagnosis_submitted", False),
    )


@app.post("/baseline")
async def run_baseline():
    from scripts.baseline_inference import get_provider, run_baseline_all_tasks
    try:
        get_provider()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    results = await run_baseline_all_tasks(base_url="http://localhost:8000")
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
