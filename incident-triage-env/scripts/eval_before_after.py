"""Before/after eval harness — produces the 'Showing Improvement' artifact.

Runs a given model over the held-out eval split (tasks NOT in training) with
multiple seeds, captures per-head breakdowns, and writes a JSON artifact.

Usage:
    # Record baseline
    MODEL_NAME=qwen/qwen3.6-plus:free HF_TOKEN=... \\
        python scripts/eval_before_after.py --label baseline --seeds 0 1 2

    # Record fine-tuned run after Phase 7
    MODEL_NAME=your-ft-model HF_TOKEN=... \\
        python scripts/eval_before_after.py --label finetuned --seeds 0 1 2

    # Print the delta table
    python scripts/eval_before_after.py --compare baseline finetuned
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import (  # type: ignore[import-not-found]
    EVAL_TASK_IDS,
    MODEL_NAME,
    ENV_URL,
    _pick_api_key,
    run_task,
)
from openai import AsyncOpenAI


ARTIFACT_DIR = Path(__file__).resolve().parent / "eval_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _artifact_path(label: str) -> Path:
    return ARTIFACT_DIR / f"{label}.json"


async def record(label: str, seeds: list[int], url: str | None) -> dict:
    api_key = _pick_api_key()
    if not api_key:
        print("ERROR: set HF_TOKEN / OPENROUTER_API_KEY / API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(base_url=os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1"),
                          api_key=api_key)
    base_url = url or ENV_URL

    all_runs: list[dict[str, Any]] = []
    for task_id in EVAL_TASK_IDS:
        for seed in seeds:
            print(f"\n[record] task={task_id} seed={seed} model={MODEL_NAME}")
            r = await run_task(client, base_url, task_id, seed=seed)
            all_runs.append({
                "task_id": task_id,
                "seed": seed,
                "score": r["score"],
                "steps": r["steps"],
                "heads": {
                    k: (v.get("score") if isinstance(v, dict) else None)
                    for k, v in (r.get("heads") or {}).items()
                },
                "effective_weights": r.get("effective_weights", {}),
            })

    # Aggregates
    scores = [r["score"] for r in all_runs]
    per_task: dict[str, list[float]] = {}
    for r in all_runs:
        per_task.setdefault(r["task_id"], []).append(r["score"])

    artifact = {
        "label": label,
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_task_ids": list(EVAL_TASK_IDS),
        "seeds": seeds,
        "runs": all_runs,
        "summary": {
            "overall_mean": statistics.fmean(scores) if scores else 0.0,
            "overall_stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "per_task_mean": {t: statistics.fmean(v) for t, v in per_task.items()},
        },
    }

    path = _artifact_path(label)
    path.write_text(json.dumps(artifact, indent=2))
    print(f"\n[record] wrote {path}")
    print(f"[record] overall_mean={artifact['summary']['overall_mean']:.3f}")
    return artifact


def compare(before_label: str, after_label: str) -> None:
    before = json.loads(_artifact_path(before_label).read_text())
    after = json.loads(_artifact_path(after_label).read_text())

    print(f"\n=== BEFORE / AFTER — held-out eval split ===")
    print(f"before: {before['label']:<12} model={before['model']}  runs={len(before['runs'])}")
    print(f"after:  {after['label']:<12} model={after['model']}  runs={len(after['runs'])}")
    print()

    b_all = before["summary"]["overall_mean"]
    a_all = after["summary"]["overall_mean"]
    print(f"{'task':<40} {'before':>10} {'after':>10} {'delta':>10}")
    print("-" * 72)
    for task in before["eval_task_ids"]:
        b = before["summary"]["per_task_mean"].get(task, 0.0)
        a = after["summary"]["per_task_mean"].get(task, 0.0)
        sign = "+" if a >= b else ""
        print(f"{task:<40} {b:>10.3f} {a:>10.3f} {sign}{a - b:>9.3f}")
    print("-" * 72)
    sign = "+" if a_all >= b_all else ""
    print(f"{'OVERALL':<40} {b_all:>10.3f} {a_all:>10.3f} {sign}{a_all - b_all:>9.3f}")

    # Per-head deltas (averaged over all runs/tasks/seeds where the head was active)
    def _head_means(runs: list[dict]) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for r in runs:
            for h, s in (r.get("heads") or {}).items():
                if s is not None:
                    buckets.setdefault(h, []).append(float(s))
        return {h: statistics.fmean(v) for h, v in buckets.items() if v}

    b_heads = _head_means(before["runs"])
    a_heads = _head_means(after["runs"])
    all_heads = sorted(set(b_heads) | set(a_heads))
    print(f"\nper-head mean (active heads only):")
    print(f"{'head':<15} {'before':>10} {'after':>10} {'delta':>10}")
    print("-" * 47)
    for h in all_heads:
        b = b_heads.get(h, 0.0)
        a = a_heads.get(h, 0.0)
        sign = "+" if a >= b else ""
        print(f"{h:<15} {b:>10.3f} {a:>10.3f} {sign}{a - b:>9.3f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 8 — before/after eval harness")
    p.add_argument("--label", type=str, help="artifact label for this run (e.g. 'baseline')")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--url", type=str, default=None, help="override ENV_URL")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                   help="print delta table for two saved artifacts")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return
    if not args.label:
        print("ERROR: pass --label or --compare BEFORE AFTER", file=sys.stderr)
        sys.exit(2)
    asyncio.run(record(args.label, args.seeds, args.url))


if __name__ == "__main__":
    main()
