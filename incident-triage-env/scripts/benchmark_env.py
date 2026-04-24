"""Benchmark the env with an oracle-trajectory per scenario.

For every scenario we hand-author a short "oracle" trajectory that follows
the ground-truth investigation path (alert svc → trace → evidence → submit).
Running it produces the **ceiling** a well-behaved agent can reach — the
number baseline + trained models should be compared against.

Outputs:
  - docs/benchmarks/oracle_scores.csv   (per-scenario per-head breakdown)
  - stdout table

Usage:
  python scripts/benchmark_env.py
  python scripts/benchmark_env.py --seeds 0 1 2 3 4
  python scripts/benchmark_env.py --out docs/benchmarks/oracle_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import IncidentTriageAction, RootCauseCategory, Remediation  # noqa: E402
from server.environment import IncidentTriageEnv  # noqa: E402
from server.scenario_loader import TASK_FILES, ScenarioLoader  # noqa: E402


# ---------------------------------------------------------------- oracles

def _legacy(action_type: str, service: str, **extra):
    return IncidentTriageAction(action_type=action_type, service=service, **extra)


def _submit(service: str, cat: str, rem: str, *, pr=None, blast=None):
    kw = dict(
        action_type="submit_diagnosis",
        root_cause_service=service,
        root_cause_category=RootCauseCategory(cat),
        remediation=Remediation(rem),
    )
    if pr is not None:
        kw["pr_proposal"] = pr
    if blast is not None:
        kw["blast_radius"] = blast
    return IncidentTriageAction(**kw)


def oracle_for(task_id: str, loader: ScenarioLoader) -> list[IncidentTriageAction]:
    """Return a short, correct trajectory for the scenario.

    These are hand-crafted — they follow the scenario's ground_truth causal
    chain with minimum steps, include the gate checks where scenarios require
    them, and submit a fully-populated diagnosis.
    """
    s = loader.get_scenario(task_id)
    gt = s["ground_truth"]
    svc = gt["root_cause_service"]
    cat = gt["root_cause_category"]
    rem = gt["remediation"]

    pr = {
        "target_repo": gt.get("correct_pr", {}).get("target_repo", ""),
        "head_branch": "revert/" + (
            gt.get("correct_pr", {}).get("keywords", [""])[0] or "fix"
        ),
        "title": f"Rollback {svc}",
        "summary": (
            f"Revert due to {cat}. "
            + " ".join(gt.get("correct_pr", {}).get("keywords", []))
        ),
        "diff_patch": "@@ "
            + " @@\n- old\n+ new\n".join(gt.get("correct_pr", {}).get("touched_files", ["-"]))
            + " @@",
    }
    blast = dict(gt.get("correct_blast_radius", {}))

    actions: list[IncidentTriageAction] = [
        _legacy("check_status", s["alert"]["service"]),
        _legacy("trace_dependencies", s["alert"]["service"]),
        _legacy("query_logs", svc, severity="error"),
        _legacy("check_deploys", svc),
    ]

    # Gate-specific evidence (required by scenario policies).
    tags = set(s.get("tags") or [])
    if task_id == "hard_freeze_violation":
        # Scenario requires page_oncall before rollback.
        actions.append(IncidentTriageAction(
            app="chatops", op="page_oncall", args={"role": "orders-lead"}
        ))
    if task_id == "hard_region_failover":
        actions.append(IncidentTriageAction(
            app="chatops", op="page_oncall", args={"role": "platform-lead"}
        ))
    if task_id == "medium_uat_skipped":
        actions.append(IncidentTriageAction(
            app="uatsim", op="check_uat_record", args={"service": svc}
        ))
    if task_id == "hard_pr_quality_breach":
        actions.append(IncidentTriageAction(
            app="repohub", op="ci_check",
            args={"target_repo": gt["correct_pr"]["target_repo"]},
        ))

    actions.append(_submit(svc, cat, rem, pr=pr, blast=blast))
    return actions


# ---------------------------------------------------------------- runner

def run_one(task_id: str, seed: int | None = None) -> dict:
    env = IncidentTriageEnv()
    loader = env._loader
    env.reset(task_id=task_id, seed=seed, episode_id=f"oracle-{task_id}-{seed}")
    actions = oracle_for(task_id, loader)

    obs = None
    steps = 0
    for a in actions:
        obs = env.step(a)
        steps += 1
        if obs.done:
            break

    if obs is None or obs.grader_breakdown is None:
        return {
            "task_id": task_id, "seed": seed, "steps": steps, "score": 0.0,
            "diagnosis": None, "policy": None, "blast": None, "pr": None,
        }

    heads = obs.grader_breakdown.get("heads", {}) or {}
    def _h(name):
        h = heads.get(name) or {}
        return h.get("score") if isinstance(h, dict) else None

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "score": obs.grader_breakdown.get("score", 0.0),
        "diagnosis": _h("diagnosis"),
        "policy": _h("policy"),
        "blast": _h("blast"),
        "pr": _h("pr"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=[None],
                   help="seeds to vary; default: unseeded base scenarios only")
    p.add_argument("--out", type=str,
                   default="docs/benchmarks/oracle_scores.csv")
    args = p.parse_args()

    rows = []
    for task_id in TASK_FILES:
        for seed in args.seeds:
            rows.append(run_one(task_id, seed))

    # Print a compact table.
    print(f"\n{'task_id':<35} {'seed':>5} {'steps':>6} "
          f"{'score':>7} {'diag':>6} {'pol':>6} {'blast':>6} {'pr':>6}")
    print("-" * 88)
    for r in rows:
        seed = "-" if r["seed"] is None else r["seed"]
        diag = f"{r['diagnosis']:.2f}" if r['diagnosis'] is not None else "  -"
        pol = f"{r['policy']:.2f}" if r['policy'] is not None else "  -"
        blast = f"{r['blast']:.2f}" if r['blast'] is not None else "  -"
        pr = f"{r['pr']:.2f}" if r['pr'] is not None else "  -"
        print(f"{r['task_id']:<35} {str(seed):>5} {r['steps']:>6} "
              f"{r['score']:>7.3f} {diag:>6} {pol:>6} {blast:>6} {pr:>6}")

    # Write CSV.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "seed", "steps", "score",
            "diagnosis", "policy", "blast", "pr",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[benchmark_env] wrote {out_path}")

    mean_score = sum(r["score"] for r in rows) / len(rows)
    print(f"[benchmark_env] oracle mean score = {mean_score:.3f}")


if __name__ == "__main__":
    main()
