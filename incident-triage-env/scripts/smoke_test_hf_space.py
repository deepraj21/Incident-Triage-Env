"""Smoke-test a deployed Incident Triage Space (or a local server).

Hits every public surface the agents / eval harness rely on:
    GET  /tasks                   — task list + action schema
    POST /reset  (via /ws)        — session open
    POST /step   (via /ws)        — legacy + multi-app actions work
    POST /grader                  — terminal grader returns heads + weights
    GET  /                        — Gradio UI loads (optional)

Usage:
    # local
    python scripts/smoke_test_hf_space.py --url http://localhost:8000

    # deployed HF Space (replace <you>/<space>)
    python scripts/smoke_test_hf_space.py \\
        --url https://<you>-incident-triage-env.hf.space

Exit code 0 = all good. Anything non-zero → the Space is broken; check the
HF build logs before running training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx
import websockets


def _ws_url(base: str) -> str:
    return base.replace("https://", "wss://").replace("http://", "ws://") + "/ws"


async def _ws_call(ws, payload: dict) -> dict:
    await ws.send(json.dumps(payload))
    return json.loads(await ws.recv())


async def run(base: str, verbose: bool) -> int:
    problems: list[str] = []

    def ok(label: str, detail: str = "") -> None:
        print(f"  ✓ {label}" + (f"  {detail}" if detail else ""))

    def fail(label: str, detail: str) -> None:
        problems.append(f"{label}: {detail}")
        print(f"  ✗ {label}  {detail}")

    print(f"\n=== Smoke-testing {base} ===")

    # 1) /tasks
    print("\n[1] GET /tasks")
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.get(base.rstrip("/") + "/tasks")
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        assert "tasks" in data and "action_schema" in data
        task_ids = [t["task_id"] for t in data["tasks"]]
        assert len(task_ids) >= 6, f"expected ≥6 tasks, got {len(task_ids)}"
        ok("tasks endpoint", f"{len(task_ids)} tasks")
        if verbose:
            for t in data["tasks"]:
                print(f"     - {t['task_id']}  [{t['difficulty']}]  budget={t['step_budget']}")
    except Exception as e:
        fail("GET /tasks", str(e))
        return _summary(problems)

    # 2) WebSocket reset → step (legacy) → step (multi-app) → submit
    print("\n[2] WS /ws  reset + step(legacy) + step(multi-app) + submit")
    try:
        async with websockets.connect(_ws_url(base), open_timeout=20) as ws:
            # reset
            r = await _ws_call(ws, {"type": "reset",
                                    "data": {"task_id": "easy_single_service_failure"}})
            obs = (r.get("data") or {}).get("observation") or r.get("data") or r
            alert = obs.get("alert", {})
            assert alert.get("service"), "no alert on observation"
            ok("reset", f"alert on {alert['service']}")

            # state → capture episode_id
            r = await _ws_call(ws, {"type": "state"})
            ep_id = (r.get("data") or r).get("episode_id")
            assert ep_id, "no episode_id in state"
            ok("state", f"episode_id={ep_id[:8]}…")

            # legacy action
            r = await _ws_call(ws, {"type": "step",
                                    "data": {"action_type": "check_status",
                                              "service": "payments-service"}})
            data = r.get("data", r)
            obs = data.get("observation", data)
            done_now = data.get("done", obs.get("done", False))
            assert not done_now, "unexpectedly done after 1 step"
            ok("step(legacy: check_status)",
               f"sim_time={obs.get('sim_time')} pending={obs.get('pending_world_events')}")

            # multi-app action (alerthub.list_alerts)
            r = await _ws_call(ws, {"type": "step",
                                    "data": {"app": "alerthub", "op": "list_alerts",
                                              "args": {}}})
            data = r.get("data", r)
            obs = data.get("observation", data)
            assert obs.get("result"), "no result from list_alerts"
            ok("step(multi-app: alerthub.list_alerts)",
               f"result-len={len(str(obs.get('result', {}).get('data', '')))}")

            # submit diagnosis
            r = await _ws_call(ws, {"type": "step", "data": {
                "action_type": "submit_diagnosis",
                "root_cause_service": "payments-service",
                "root_cause_category": "bad_deploy",
                "remediation": "rollback_deploy",
            }})
            data = r.get("data", r)
            obs = data.get("observation", data)
            done_now = data.get("done", obs.get("done", False))
            assert done_now, "submit did not terminate the episode"
            gb = obs.get("grader_breakdown")
            assert gb, "no grader_breakdown on terminal obs"
            ok("submit_diagnosis", f"score={gb['score']:.3f}")
    except Exception as e:
        fail("WS flow", str(e))
        return _summary(problems)

    # 3) POST /grader
    print("\n[3] POST /grader  (fetch final score by episode_id)")
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.post(base.rstrip("/") + "/grader",
                                 json={"episode_id": ep_id})
        assert r.status_code == 200, f"status={r.status_code}"
        data = r.json()
        for k in ("score", "breakdown", "heads", "effective_weights", "diagnosis_submitted"):
            assert k in data, f"missing '{k}' in grader response"
        ok("grader response shape",
           f"heads={list(data['heads'].keys())}")
    except Exception as e:
        fail("POST /grader", str(e))
        return _summary(problems)

    return _summary(problems)


def _summary(problems: list[str]) -> int:
    if problems:
        print(f"\n❌ {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n✅ all surfaces healthy — Space is ready for inference / training")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True,
                   help="base URL (e.g. https://you-incident-triage-env.hf.space)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    rc = asyncio.run(run(args.url.rstrip("/"), args.verbose))
    sys.exit(rc)


if __name__ == "__main__":
    main()
