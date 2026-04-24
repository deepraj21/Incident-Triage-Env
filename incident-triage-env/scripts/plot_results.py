"""Plot helpers for Phase 8 artifacts.

Produces three PNGs into docs/benchmarks/:

  1. oracle_ceiling.png           — per-scenario oracle ceiling (bar chart)
  2. per_head_radar_<label>.png   — per-head mean radar for an eval artifact
  3. before_after_<a>_vs_<b>.png  — per-task before/after + delta bars

Usage:
  # just the oracle ceiling (always available after benchmark_env.py)
  python scripts/plot_results.py --oracle

  # radar for a single eval artifact
  python scripts/plot_results.py --radar baseline

  # before/after delta
  python scripts/plot_results.py --compare baseline finetuned
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"matplotlib not installed: {e}\n  pip install matplotlib")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "scripts" / "eval_artifacts"
PLOTS_DIR = ROOT / "docs" / "benchmarks"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- oracle

def plot_oracle_ceiling(csv_path: Path = PLOTS_DIR / "oracle_scores.csv") -> Path:
    if not csv_path.exists():
        raise SystemExit(f"oracle CSV not found at {csv_path}. "
                         "Run `python scripts/benchmark_env.py` first.")
    rows = list(csv.DictReader(csv_path.open()))
    # Collapse seeds — just take the first row per task_id for the ceiling view.
    seen, tasks, scores, diag, pol, blast, pr = set(), [], [], [], [], [], []
    for r in rows:
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        tasks.append(r["task_id"])
        scores.append(float(r["score"]))
        diag.append(float(r["diagnosis"] or 0))
        pol.append(float(r["policy"] or 0))
        blast.append(float(r["blast"] or 0))
        pr.append(float(r["pr"] or 0))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(tasks))
    # Stacked contribution per head (not exactly the composite, but visually
    # illustrative of which head carries each scenario).
    ax.bar(x, diag, label="diagnosis", color="#2E86AB")
    ax.bar(x, pol, bottom=diag, label="policy", color="#A23B72")
    ax.bar(x, blast, bottom=[a + b for a, b in zip(diag, pol)],
           label="blast", color="#F18F01")
    ax.bar(x, pr,
           bottom=[a + b + c for a, b, c in zip(diag, pol, blast)],
           label="pr", color="#C73E1D")

    ax.plot(x, scores, "ko-", label="composite score", linewidth=1.5, markersize=5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks],
                       fontsize=8, rotation=0)
    ax.set_ylabel("score")
    ax.set_ylim(0, max(4.1, max(scores) + 0.2))
    ax.set_title("Oracle ceiling per scenario (stacked per-head + composite)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "oracle_ceiling.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- radar

def _mean(vs):
    return statistics.fmean(vs) if vs else 0.0


def plot_per_head_radar(label: str) -> Path:
    artifact = ART_DIR / f"{label}.json"
    if not artifact.exists():
        raise SystemExit(f"artifact not found: {artifact}")
    data = json.loads(artifact.read_text())

    buckets = {"diagnosis": [], "policy": [], "blast": [], "pr": []}
    for r in data.get("runs", []):
        for head, score in (r.get("heads") or {}).items():
            if score is not None and head in buckets:
                buckets[head].append(float(score))
    means = {h: _mean(vs) for h, vs in buckets.items()}

    labels = list(means.keys())
    values = [means[l] for l in labels]
    # Close the radar by repeating the first point.
    angles = [n / len(labels) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#2E86AB")
    ax.fill(angles, values, alpha=0.25, color="#2E86AB")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], fontsize=8)
    ax.set_title(f"Per-head mean — {label}\n(n={len(data.get('runs', []))} runs)",
                 y=1.1)
    fig.tight_layout()
    out = PLOTS_DIR / f"per_head_radar_{label}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- before/after

def plot_before_after(before_label: str, after_label: str) -> Path:
    a = json.loads((ART_DIR / f"{before_label}.json").read_text())
    b = json.loads((ART_DIR / f"{after_label}.json").read_text())

    task_ids = a["eval_task_ids"]
    a_means = a["summary"]["per_task_mean"]
    b_means = b["summary"]["per_task_mean"]
    before_scores = [a_means.get(t, 0) for t in task_ids]
    after_scores = [b_means.get(t, 0) for t in task_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    x = range(len(task_ids))
    width = 0.38
    ax1.bar([i - width / 2 for i in x], before_scores, width,
            label=before_label, color="#999999")
    ax1.bar([i + width / 2 for i in x], after_scores, width,
            label=after_label, color="#2E86AB")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([t.replace("_", "\n") for t in task_ids], fontsize=8)
    ax1.set_ylabel("composite score")
    ax1.set_title(f"Held-out eval: {before_label} vs {after_label}")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    deltas = [bb - aa for aa, bb in zip(before_scores, after_scores)]
    colors = ["#2A9D8F" if d >= 0 else "#E76F51" for d in deltas]
    ax2.bar(list(x), deltas, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([t.replace("_", "\n") for t in task_ids], fontsize=8)
    ax2.set_ylabel("Δ score")
    ax2.set_title("Per-task delta")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / f"before_after_{before_label}_vs_{after_label}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", action="store_true",
                   help="plot the oracle ceiling from docs/benchmarks/oracle_scores.csv")
    p.add_argument("--radar", type=str, default=None,
                   help="plot per-head radar for a given eval artifact label")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                   help="plot before/after comparison for two eval artifacts")
    args = p.parse_args()

    produced = []
    if args.oracle:
        produced.append(plot_oracle_ceiling())
    if args.radar:
        produced.append(plot_per_head_radar(args.radar))
    if args.compare:
        produced.append(plot_before_after(args.compare[0], args.compare[1]))

    if not produced:
        # Default: always attempt the oracle plot (it's cheap and always useful).
        produced.append(plot_oracle_ceiling())

    print("\nPlots written:")
    for p in produced:
        print(f"  • {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
