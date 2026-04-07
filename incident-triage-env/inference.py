"""Hackathon-compliant inference script for the Incident Triage Environment.

Uses the OpenAI Client with API_BASE_URL, MODEL_NAME, and HF_TOKEN env vars
as required by the Scalar hackathon submission criteria.

STDOUT FORMAT: Emits [START], [STEP], [END] lines per hackathon spec.

Usage:
    API_BASE_URL=https://openrouter.ai/api/v1 \
    MODEL_NAME=qwen/qwen3.6-plus:free \
    HF_TOKEN=your-key \
    ENV_URL=http://localhost:8000 \
    PYTHONPATH=. python inference.py
"""

import asyncio
import json
import os
import re
import sys
from typing import List, Optional

import httpx
import websockets
from openai import AsyncOpenAI


# --- Configuration from env vars (hackathon-required) ---

API_BASE_URL = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen/qwen3.6-plus:free")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Environment server URL
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

BENCHMARK = "incident_triage_env"
MAX_RETRIES = 2
TEMPERATURE = 0.1


# --- Mandatory stdout logging (hackathon format) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# --- System prompt: teaches efficient incident triage strategy ---

SYSTEM_PROMPT = """You are an expert SRE/oncall engineer diagnosing a production incident. You must investigate systematically and efficiently within a limited step budget.

INVESTIGATION STRATEGY (follow this order):
1. START with the alerted service: check_status to see health + active alerts
2. TRACE DEPENDENCIES: trace_dependencies on the alerted service to find upstream/downstream services
3. FOLLOW THE CHAIN UPSTREAM: If a service has errors caused by an upstream dependency, ALWAYS trace_dependencies on that upstream service too. Keep going until you find the DEEPEST upstream service that is the actual root cause. The root cause is NEVER a service that is just relaying errors from upstream — it's the one where the failure ORIGINATES.
4. GATHER EVIDENCE on the deepest root cause: query_logs (severity=error), check_deploys, query_metrics
5. SUBMIT DIAGNOSIS: Once you've identified the deepest root cause, submit immediately

CRITICAL: THE ROOT CAUSE IS ALWAYS THE DEEPEST UPSTREAM SERVICE IN THE FAILURE CHAIN.
- If service A calls service B which calls service C, and C has a config error causing B to fail causing A to fail, the root cause is C — NOT A or B.
- "Connection refused" or "timeout calling X" means X (or something X depends on) is the root cause, NOT the service experiencing the timeout.
- When you see a service failing because of an upstream dependency, ALWAYS trace that upstream's dependencies too. Do not stop at the first upstream — go deeper.
- When logs mention another service name (e.g., "connection to auth-db refused"), investigate THAT service.

KEY RULES:
- Follow dependency chains UPSTREAM to find root cause (symptoms appear downstream, causes are upstream)
- A service timing out on calls to another service means the OTHER service is the problem
- Look for: recent deploys, config changes, memory leaks, connection exhaustion
- Do NOT query the same service more than 3 times (penalty for redundancy)
- Do NOT waste steps on healthy services that show no anomalies
- You MUST submit_diagnosis before running out of steps
- When in doubt, the root cause is the service furthest upstream in the dependency chain that shows errors

Root cause categories: bad_deploy, resource_exhaustion, dependency_failure, config_change, traffic_spike, data_corruption
Remediations: rollback_deploy (for bad deploys), scale_up (for traffic), restart_service (for resource exhaustion/memory leaks), fix_config (for config changes), enable_rate_limiting (for traffic spikes), failover_to_backup (for dependency failures)

MATCHING GUIDE:
- NullPointerException after a deploy -> bad_deploy -> rollback_deploy
- max_connections changed / config reload / connection pool / pool size -> config_change -> fix_config
- Memory leak / OOM / GC pressure / heap growing -> resource_exhaustion -> restart_service
- Upstream service down -> dependency_failure -> failover_to_backup
- Traffic surge / request spike -> traffic_spike -> enable_rate_limiting
- "connection refused" to a database -> check that database service for config_change or resource_exhaustion

Respond with ONLY a single JSON action object. No explanation, no markdown, just JSON.

Example actions:
{"action_type": "check_status", "service": "payments-service"}
{"action_type": "trace_dependencies", "service": "payments-service"}
{"action_type": "query_logs", "service": "payments-service", "severity": "error"}
{"action_type": "check_deploys", "service": "payments-service"}
{"action_type": "query_metrics", "service": "payments-service", "metric": "memory"}
{"action_type": "inspect_code", "service": "payments-service"}
{"action_type": "submit_diagnosis", "root_cause_service": "payments-service", "root_cause_category": "bad_deploy", "remediation": "rollback_deploy"}"""


def parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


def format_observation(obs_data: dict, step: int) -> str:
    """Format an observation into a readable string for the LLM."""
    parts = []
    alert = obs_data.get("alert", {})
    if step == 0:
        parts.append(f"INCIDENT ALERT: [{alert.get('severity', 'unknown').upper()}] {alert.get('service', '?')}: {alert.get('message', '?')}")
        parts.append(f"Timestamp: {alert.get('timestamp', '?')}")
        parts.append("\nYou are the oncall engineer. Investigate this incident systematically.")
        parts.append("Start by checking the status and dependencies of the alerted service.")
    else:
        parts.append(f"Alert context: {alert.get('service', '?')}: {alert.get('message', '?')}")

    result = obs_data.get("result")
    if result:
        parts.append(f"\n--- Result of {result.get('action_type', '?')} ---")
        parts.append(result.get("data", ""))

    steps_remaining = obs_data.get("steps_remaining", "?")
    parts.append(f"\nSteps remaining: {steps_remaining}")
    investigated = obs_data.get("services_investigated", [])
    if investigated:
        parts.append(f"Services investigated so far: {', '.join(investigated)}")

    # Urgency prompt when steps are running low
    if isinstance(steps_remaining, int) and steps_remaining <= 3:
        parts.append("\n*** URGENT: You are running low on steps! Submit your diagnosis NOW with submit_diagnosis. ***")
    elif isinstance(steps_remaining, int) and steps_remaining <= 5:
        parts.append("\n** Warning: Steps running low. Consider submitting your diagnosis soon. **")

    return "\n".join(parts)


async def run_task(client: AsyncOpenAI, base_url: str, task_id: str) -> dict:
    """Run a single task and return the grader score."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with websockets.connect(ws_url) as ws:
            # Reset
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            response = json.loads(await ws.recv())
            data = response.get("data", response)
            obs_data = data.get("observation", data)
            done = data.get("done", obs_data.get("done", False))

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Get state for episode_id
            await ws.send(json.dumps({"type": "state"}))
            state_response = json.loads(await ws.recv())
            state_data = state_response.get("data", state_response)
            episode_id = state_data.get("episode_id", "unknown")

            step = 0

            while not done:
                user_msg = format_observation(obs_data, step)
                messages.append({"role": "user", "content": user_msg})

                # Get action from LLM
                action = None
                last_error = None
                for retry in range(MAX_RETRIES + 1):
                    try:
                        completion = await client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            response_format={"type": "json_object"},
                            temperature=TEMPERATURE,
                        )
                        raw = completion.choices[0].message.content
                        action = parse_json_from_response(raw)
                        break
                    except (ValueError, Exception) as e:
                        last_error = str(e)
                        if retry < MAX_RETRIES:
                            messages.append({
                                "role": "user",
                                "content": f"Your response was not valid JSON. Error: {e}. Respond with ONLY a JSON object."
                            })

                if action is None:
                    log_end(success=False, steps=steps_taken, score=0.001, rewards=rewards)
                    return {"task_id": task_id, "score": 0.001, "steps": steps_taken, "error": last_error}

                # Clean action
                action.pop("metadata", None)
                action.pop("reasoning", None)
                action.pop("explanation", None)

                # Send action to environment
                await ws.send(json.dumps({"type": "step", "data": action}))
                response = json.loads(await ws.recv())
                data = response.get("data", response)
                obs_data = data.get("observation", data)
                done = data.get("done", obs_data.get("done", False))
                step += 1
                steps_taken = step

                reward = data.get("reward", obs_data.get("reward", 0.0))
                if reward is None:
                    reward = 0.0
                rewards.append(float(reward))

                messages.append({"role": "assistant", "content": json.dumps(action)})

                action_str = f"{action.get('action_type', '?')}({action.get('service', action.get('root_cause_service', ''))})"
                log_step(step=step, action=action_str, reward=float(reward), done=done, error=None)

        # Get grader score
        async with httpx.AsyncClient() as http_client:
            grader_url = base_url.rstrip("/") + "/grader"
            resp = await http_client.post(grader_url, json={"episode_id": episode_id})
            if resp.status_code == 200:
                grader_data = resp.json()
                score = grader_data.get("score", 0.001)
                # Clamp strictly within (0, 1) — Phase 2 rejects exact 0.0 or 1.0.
                score = max(0.001, min(0.999, float(score)))
                success = score >= 0.3
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
                return {
                    "task_id": task_id,
                    "score": score,
                    "breakdown": grader_data.get("breakdown", {}),
                    "steps": steps_taken,
                }
            else:
                log_end(success=False, steps=steps_taken, score=0.001, rewards=rewards)
                return {"task_id": task_id, "score": 0.001, "steps": steps_taken, "error": f"Grader returned {resp.status_code}"}

    except Exception as e:
        log_end(success=False, steps=steps_taken, score=0.001, rewards=rewards)
        return {"task_id": task_id, "score": 0.001, "steps": steps_taken, "error": str(e)}


TASK_IDS = [
    "easy_single_service_failure",
    "medium_cascading_dependency",
    "hard_multi_signal_cascade",
]


async def main():
    """Run baseline inference on all tasks using OpenAI Client."""
    # Validate required env vars
    # Pick the right API key based on which API we're talking to
    if "openrouter" in API_BASE_URL.lower():
        api_key = os.environ.get("OPENROUTER_API_KEY") or HF_TOKEN
    elif "huggingface" in API_BASE_URL.lower():
        api_key = HF_TOKEN or os.environ.get("API_KEY")
    else:
        api_key = HF_TOKEN or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY")

    if not api_key:
        print("ERROR: Set HF_TOKEN (or OPENROUTER_API_KEY) for LLM access.")
        print("  Example: OPENROUTER_API_KEY=your-key python inference.py")
        sys.exit(1)

    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
    )

    base_url = ENV_URL
    if len(sys.argv) > 1 and sys.argv[1] == "--url":
        base_url = sys.argv[2]

    print(f"Using model: {MODEL_NAME}")
    print(f"Using API base: {API_BASE_URL}")
    print(f"Environment URL: {base_url}")

    scores = {}
    total_steps = {}

    for task_id in TASK_IDS:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}")

        result = await run_task(client, base_url, task_id)

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
    for task_id, s in scores.items():
        print(f"  {task_id}: {s:.2f} (steps: {total_steps[task_id]})")

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Average: {avg:.2f}")

    return {
        "scores": scores,
        "model": MODEL_NAME,
        "total_steps_used": total_steps,
    }


if __name__ == "__main__":
    asyncio.run(main())
