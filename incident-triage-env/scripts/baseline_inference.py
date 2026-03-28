"""Baseline inference script for the Incident Triage Environment.

Uses free LLM APIs (Gemini primary, OpenRouter fallback) to play all 3 tasks
and report grader scores.
"""

import asyncio
import json
import os
import re
import sys
from abc import ABC, abstractmethod

import httpx
import websockets


SYSTEM_PROMPT = """You are an experienced oncall engineer diagnosing a production incident.
You will receive alerts and investigation results. Investigate systematically by querying
logs, metrics, deploys, dependencies, service status, and code.

Respond with ONLY a single JSON action object. No explanation, no markdown, just JSON.

Available action_types:
- query_logs: params: service (required), severity (optional: error/warn/info/debug), keyword (optional)
- query_metrics: params: service (required), metric (required: latency_p99/error_rate/cpu/memory/connections)
- check_deploys: params: service (required)
- trace_dependencies: params: service (required)
- check_status: params: service (required)
- inspect_code: params: service (required), file_path (optional)
- submit_diagnosis: params: root_cause_service (required), root_cause_category (required: bad_deploy/resource_exhaustion/dependency_failure/config_change/traffic_spike/data_corruption), remediation (required: rollback_deploy/scale_up/restart_service/fix_config/enable_rate_limiting/failover_to_backup)

Example actions:
{"action_type": "query_logs", "service": "payments-service", "severity": "error"}
{"action_type": "query_metrics", "service": "auth-db", "metric": "connections"}
{"action_type": "submit_diagnosis", "root_cause_service": "auth-db", "root_cause_category": "config_change", "remediation": "fix_config"}

Investigate efficiently. You have a limited step budget. Focus on following the evidence chain."""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        self._api_key = api_key
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, messages: list[dict]) -> str:
        from google import genai
        client = genai.Client(api_key=self._api_key)

        # Build contents from messages
        contents = []
        system_instruction = None
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append({"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(
            model=self._model,
            contents=contents,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
            },
        )
        return response.text


class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "stepfun/step-3.5-flash:free"):
        self._api_key = api_key
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, messages: list[dict]) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )
        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content


def get_provider() -> LLMProvider:
    if key := os.environ.get("GEMINI_API_KEY"):
        return GeminiProvider(key)
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterProvider(key)
    raise RuntimeError("Set GEMINI_API_KEY or OPENROUTER_API_KEY to run baseline inference")


def parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


async def run_task(provider: LLMProvider, base_url: str, task_id: str, max_retries: int = 2) -> dict:
    """Run a single task and return the grader score."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    async with websockets.connect(ws_url) as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
        response = json.loads(await ws.recv())
        # Response format: {"type": "observation", "data": {"observation": {...}, "reward": ..., "done": ...}}
        data = response.get("data", response)
        obs_data = data.get("observation", data)
        done = data.get("done", obs_data.get("done", False))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        episode_id = None

        # Get state to grab episode_id
        await ws.send(json.dumps({"type": "state"}))
        state_response = json.loads(await ws.recv())
        state_data = state_response.get("data", state_response)
        episode_id = state_data.get("episode_id", "unknown")

        step = 0

        while not done:
            # Format observation for the LLM
            user_msg = format_observation(obs_data, step)
            messages.append({"role": "user", "content": user_msg})

            # Get action from LLM
            action = None
            for retry in range(max_retries + 1):
                try:
                    raw = await provider.generate(messages)
                    action = parse_json_from_response(raw)
                    break
                except (ValueError, Exception) as e:
                    if retry < max_retries:
                        messages.append({"role": "user", "content": f"Your response was not valid JSON. Error: {e}. Please respond with ONLY a JSON object."})
                    else:
                        print(f"  Failed to get valid JSON after {max_retries + 1} attempts: {e}")
                        return {"task_id": task_id, "score": 0.0, "steps": step, "error": str(e)}

            # Remove metadata field if present (not part of action schema for the env)
            action.pop("metadata", None)

            # Send action to environment
            await ws.send(json.dumps({"type": "step", "data": action}))
            response = json.loads(await ws.recv())
            data = response.get("data", response)
            obs_data = data.get("observation", data)
            done = data.get("done", obs_data.get("done", False))
            step += 1

            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": json.dumps(action)})

            reward = data.get("reward", obs_data.get("reward", "N/A"))
            print(f"  Step {step}: {action.get('action_type', '?')} {action.get('service', '')} -> reward={reward}")

    # Get grader score
    async with httpx.AsyncClient() as client:
        grader_url = base_url.rstrip("/") + "/grader"
        resp = await client.post(grader_url, json={"episode_id": episode_id})
        if resp.status_code == 200:
            grader_data = resp.json()
            return {
                "task_id": task_id,
                "score": grader_data.get("score", 0.0),
                "breakdown": grader_data.get("breakdown", {}),
                "steps": step,
            }
        else:
            return {"task_id": task_id, "score": 0.0, "steps": step, "error": f"Grader returned {resp.status_code}"}


def format_observation(obs_data: dict, step: int) -> str:
    """Format an observation into a readable string for the LLM."""
    parts = []

    alert = obs_data.get("alert", {})
    if step == 0:
        parts.append(f"ALERT: [{alert.get('severity', 'unknown').upper()}] {alert.get('service', '?')}: {alert.get('message', '?')}")
        parts.append(f"Timestamp: {alert.get('timestamp', '?')}")
    else:
        parts.append(f"Alert: {alert.get('service', '?')}: {alert.get('message', '?')}")

    result = obs_data.get("result")
    if result:
        parts.append(f"\n--- Result of {result.get('action_type', '?')} ---")
        parts.append(result.get("data", ""))

    parts.append(f"\nSteps remaining: {obs_data.get('steps_remaining', '?')}")
    investigated = obs_data.get("services_investigated", [])
    if investigated:
        parts.append(f"Services investigated so far: {', '.join(investigated)}")

    return "\n".join(parts)


TASK_IDS = [
    "easy_single_service_failure",
    "medium_cascading_dependency",
    "hard_multi_signal_cascade",
]


async def run_baseline_all_tasks(base_url: str = "http://localhost:8000") -> dict:
    """Run baseline inference on all tasks. Returns results dict."""
    provider = get_provider()
    print(f"Using model: {provider.model_name}")

    scores = {}
    total_steps = {}

    for task_id in TASK_IDS:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}")

        result = run_task(provider, base_url, task_id)
        if asyncio.iscoroutine(result):
            result = await result

        scores[task_id] = result.get("score", 0.0)
        total_steps[task_id] = result.get("steps", 0)

        print(f"\nResult: score={result.get('score', 0.0):.2f}, steps={result.get('steps', 0)}")
        if "breakdown" in result:
            for k, v in result["breakdown"].items():
                print(f"  {k}: {v}")
        if "error" in result:
            print(f"  Error: {result['error']}")

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.2f} (steps: {total_steps[task_id]})")

    return {
        "scores": scores,
        "model": provider.model_name,
        "total_steps_used": total_steps,
    }


if __name__ == "__main__":
    base_url = "http://localhost:8000"
    if len(sys.argv) > 1 and sys.argv[1] == "--url":
        base_url = sys.argv[2]

    asyncio.run(run_baseline_all_tasks(base_url))
