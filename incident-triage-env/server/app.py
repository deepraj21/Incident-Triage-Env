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
    heads: dict = {}
    effective_weights: dict = {}
    policy_summary: dict = {}
    diagnosis_submitted: bool


@app.post("/grader", response_model=GraderResponse)
async def grade_episode(req: GraderRequest):
    episode = get_episode(req.episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"Episode {req.episode_id} not found")
    gs = episode.get("grader_score") or {}
    # Flatten heads into a breakdown dict for backwards compat with Round-1 clients.
    heads = gs.get("heads", {})
    flat_breakdown = {
        name: (h.get("score") if isinstance(h, dict) else None)
        for name, h in heads.items()
    }
    # Also fold in the diagnosis sub-breakdown if present (Round-1 shape).
    if isinstance(heads.get("diagnosis"), dict):
        flat_breakdown.update(heads["diagnosis"].get("breakdown", {}))
    return GraderResponse(
        task_id=episode["task_id"],
        score=gs.get("score", 0.0),
        breakdown=flat_breakdown,
        heads=heads,
        effective_weights=gs.get("effective_weights", {}),
        policy_summary=episode.get("policy_summary") or {},
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


# Mount Gradio UI for interactive demo
try:
    import gradio as gr
    from server.gradio_ui import build_gradio_app

    gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
except ImportError:
    # Gradio not installed — API-only mode
    pass


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
