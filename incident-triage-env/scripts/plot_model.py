"""plot_model.py — public-facing plot generator for the trained model artifacts.

Renders the full set of submission plots from on-disk CSVs / JSONs:

  1. SFT training curve              (reward_curve_sft.png)
  2. GRPO training curve             (reward_curve_grpo.png)
  3. SFT + GRPO combined panel       (training_pipeline.png)
  4. Per-head radar — baseline       (per_head_radar_baseline-hf.png)
  5. Per-head radar — finetuned      (per_head_radar_finetuned-sft.png)
  6. Before/after delta              (before_after_baseline-hf_vs_finetuned-sft.png)
  7. Oracle ceiling                  (oracle_ceiling.png)

Inputs (created upstream by training + eval pipeline):
  ./trained/sft/reward_curve.csv             ← from train_sft.py
  ./trained/grpo/reward_curve.csv            ← from train_grpo.py
  scripts/eval_artifacts/baseline-hf.json    ← from eval_before_after.py
  scripts/eval_artifacts/finetuned-sft.json  ← from eval_before_after.py
  docs/benchmarks/oracle_scores.csv          ← from benchmark_env.py

Usage:
  python scripts/plot_model.py            # render all (skips any missing inputs)
  python scripts/plot_model.py --pipeline # only the SFT+GRPO combined panel
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "scripts" / "eval_artifacts"
PLOTS_DIR = ROOT / "docs" / "benchmarks"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_curve(csv_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    rows = list(csv.DictReader(csv_path.open()))
    steps = [int(r["step"]) for r in rows]
    reward = [float(r["reward"]) if r.get("reward") not in (None, "") else None for r in rows]
    loss = [float(r["loss"]) if r.get("loss") not in (None, "") else None for r in rows]
    kl = [float(r["kl"]) if r.get("kl") not in (None, "") else None for r in rows]
    return steps, reward, loss, kl


# --------------------------------------------------------------- SFT curve

def plot_sft_curve(adapter_dir: Path = ROOT / "trained" / "sft") -> Path | None:
    src = adapter_dir / "reward_curve.csv"
    if not src.exists():
        print(f"[plot_model] skip SFT — {src} not found")
        return None
    steps, _, loss, _ = _read_curve(src)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, loss, "o-", linewidth=2, color="#E76F51", markersize=5)
    ax.set_title("SFT training — loss per step", fontsize=14)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.fill_between(steps, loss, alpha=0.15, color="#E76F51")
    fig.tight_layout()
    out = PLOTS_DIR / "reward_curve_sft.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_model] wrote {out}")
    return out


# --------------------------------------------------------------- GRPO curve

def plot_grpo_curve(adapter_dir: Path = ROOT / "trained" / "grpo") -> Path | None:
    src = adapter_dir / "reward_curve.csv"
    if not src.exists():
        print(f"[plot_model] skip GRPO — {src} not found")
        return None
    steps, reward, loss, kl = _read_curve(src)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, reward, "o-", color="#2A9D8F", linewidth=1.4, markersize=3)
    axes[0].set_title("GRPO mean reward")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, loss, "-", color="#E76F51", linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].set_title("Policy ratio loss")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("loss")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, kl, "-", color="#7E57C2", linewidth=1.5)
    axes[2].fill_between(steps, kl, alpha=0.2, color="#7E57C2")
    axes[2].set_title("KL to reference policy")
    axes[2].set_xlabel("step"); axes[2].set_ylabel("KL")
    axes[2].grid(alpha=0.3)

    fig.suptitle("GRPO refinement on top of SFT checkpoint", y=1.02, fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / "reward_curve_grpo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_model] wrote {out}")
    return out


# --------------------------------------------------------------- combined SFT+GRPO panel

def plot_pipeline(sft_dir: Path = ROOT / "trained" / "sft",
                  grpo_dir: Path = ROOT / "trained" / "grpo") -> Path | None:
    sft_csv = sft_dir / "reward_curve.csv"
    grpo_csv = grpo_dir / "reward_curve.csv"
    if not sft_csv.exists() or not grpo_csv.exists():
        print(f"[plot_model] skip pipeline — need both {sft_csv} and {grpo_csv}")
        return None
    s_steps, _, s_loss, _ = _read_curve(sft_csv)
    g_steps, g_reward, _, g_kl = _read_curve(grpo_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                              gridspec_kw={"width_ratios": [1, 1.4]})

    # Stage 1 — SFT loss
    axes[0].plot(s_steps, s_loss, "o-", color="#E76F51",
                  linewidth=2, markersize=5)
    axes[0].fill_between(s_steps, s_loss, alpha=0.18, color="#E76F51")
    axes[0].set_title("Stage 1 — Supervised fine-tuning\n(loss per step)",
                      fontsize=12)
    axes[0].set_xlabel("SFT step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    if s_loss:
        delta = (s_loss[-1] - s_loss[0]) / max(s_loss[0], 1e-6) * 100
        axes[0].annotate(
            f"Δ = {s_loss[0]:.2f} → {s_loss[-1]:.2f}\n({delta:+.0f}%)",
            xy=(s_steps[-1], s_loss[-1]),
            xytext=(0.55, 0.7), textcoords="axes fraction",
            fontsize=10, color="#444",
        )

    # Stage 2 — GRPO reward
    axes[1].plot(g_steps, g_reward, "o-", color="#2A9D8F",
                  linewidth=1.3, markersize=3)
    # smoothed trend
    if len(g_reward) > 5:
        window = max(5, len(g_reward) // 12)
        smooth = []
        for i in range(len(g_reward)):
            lo, hi = max(0, i - window // 2), min(len(g_reward), i + window // 2 + 1)
            smooth.append(sum(g_reward[lo:hi]) / (hi - lo))
        axes[1].plot(g_steps, smooth, "-", color="#1d6c63", linewidth=2.5,
                      alpha=0.85, label="moving avg")
        axes[1].legend(loc="lower right")
    axes[1].set_title("Stage 2 — GRPO refinement\n(mean reward per step)",
                      fontsize=12)
    axes[1].set_xlabel("GRPO step")
    axes[1].set_ylabel("reward")
    axes[1].grid(alpha=0.3)
    if g_reward:
        first_avg = sum(g_reward[:5]) / 5
        last_avg = sum(g_reward[-5:]) / 5
        axes[1].annotate(
            f"reward {first_avg:.2f} → {last_avg:.2f}",
            xy=(g_steps[-1], last_avg),
            xytext=(0.05, 0.85), textcoords="axes fraction",
            fontsize=10, color="#444",
        )

    fig.suptitle("Two-stage training pipeline:  SFT warm-start  →  GRPO refinement",
                  y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = PLOTS_DIR / "training_pipeline.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_model] wrote {out}")
    return out


# --------------------------------------------------------------- per-head radar

def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def plot_per_head_radar(label: str) -> Path | None:
    art = ART_DIR / f"{label}.json"
    if not art.exists():
        print(f"[plot_model] skip radar({label}) — {art} not found")
        return None
    data = json.loads(art.read_text())
    buckets = {"diagnosis": [], "policy": [], "blast": [], "pr": []}
    for r in data.get("runs", []):
        for h, s in (r.get("heads") or {}).items():
            if s is not None and h in buckets:
                buckets[h].append(float(s))
    means = {h: _mean(v) for h, v in buckets.items()}

    labels = list(means.keys())
    values = [means[l] for l in labels]
    angles = [n / len(labels) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    color = "#2E86AB" if "finetuned" in label else "#94a3b8"
    ax.plot(angles, values, "o-", linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], fontsize=8)
    ax.set_title(f"Per-head mean — {label}\n(n={len(data.get('runs', []))} runs)",
                  y=1.1, fontsize=12)
    fig.tight_layout()
    out = PLOTS_DIR / f"per_head_radar_{label}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_model] wrote {out}")
    return out


# --------------------------------------------------------------- before/after

def plot_before_after(before: str, after: str) -> Path | None:
    a_path = ART_DIR / f"{before}.json"
    b_path = ART_DIR / f"{after}.json"
    if not (a_path.exists() and b_path.exists()):
        print(f"[plot_model] skip compare — need {a_path} and {b_path}")
        return None
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    task_ids = a["eval_task_ids"]
    a_means = a["summary"]["per_task_mean"]
    b_means = b["summary"]["per_task_mean"]
    bef = [a_means.get(t, 0) for t in task_ids]
    aft = [b_means.get(t, 0) for t in task_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                     gridspec_kw={"width_ratios": [2, 1]})
    x = range(len(task_ids))
    width = 0.38
    ax1.bar([i - width / 2 for i in x], bef, width, label=before, color="#999")
    ax1.bar([i + width / 2 for i in x], aft, width, label=after, color="#2E86AB")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([t.replace("_", "\n") for t in task_ids], fontsize=8)
    ax1.set_ylabel("composite score"); ax1.set_ylim(0, 1)
    ax1.set_title(f"Held-out eval: {before} vs {after}")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    deltas = [bb - aa for aa, bb in zip(bef, aft)]
    colors = ["#2A9D8F" if d >= 0 else "#E76F51" for d in deltas]
    ax2.bar(list(x), deltas, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([t.replace("_", "\n") for t in task_ids], fontsize=8)
    ax2.set_ylabel("Δ score")
    ax2.set_title("Per-task delta")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / f"before_after_{before}_vs_{after}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_model] wrote {out}")
    return out


# --------------------------------------------------------------- oracle ceiling

def plot_oracle_ceiling() -> Path | None:
    src = PLOTS_DIR / "oracle_scores.csv"
    if not src.exists():
        print(f"[plot_model] skip oracle — {src} not found "
              f"(run scripts/benchmark_env.py first)")
        return None
    rows = list(csv.DictReader(src.open()))
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
    ax.bar(x, diag, label="diagnosis", color="#2E86AB")
    ax.bar(x, pol, bottom=diag, label="policy", color="#A23B72")
    ax.bar(x, blast, bottom=[a + b for a, b in zip(diag, pol)],
            label="blast", color="#F18F01")
    ax.bar(x, pr, bottom=[a + b + c for a, b, c in zip(diag, pol, blast)],
            label="pr", color="#C73E1D")
    ax.plot(list(x), scores, "ko-", label="composite", linewidth=1.5, markersize=5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, max(4.1, max(scores) + 0.2))
    ax.set_title("Oracle ceiling — stacked per-head + composite")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "oracle_ceiling.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot_model] wrote {out}")
    return out


# --------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", default=True,
                   help="render every plot whose inputs exist (default)")
    p.add_argument("--pipeline", action="store_true",
                   help="only the SFT+GRPO combined panel")
    p.add_argument("--before", default="baseline-hf")
    p.add_argument("--after",  default="finetuned-sft")
    args = p.parse_args()

    if args.pipeline:
        plot_pipeline()
        return

    plot_sft_curve()
    plot_grpo_curve()
    plot_pipeline()
    plot_per_head_radar(args.before)
    plot_per_head_radar(args.after)
    plot_before_after(args.before, args.after)
    plot_oracle_ceiling()


if __name__ == "__main__":
    main()
