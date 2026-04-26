# GRPO Training — Incident Triage Environment

End-to-end training guide for `scripts/train_grpo.py`. The same script runs on:

| Hardware | Model | LoRA r | Notes |
|---|---|---|---|
| **M4 Air (16 GB)** | Qwen2.5-0.5B-Instruct | 8 | Slow (~30 min/iter). Use for smoke tests only. |
| **Colab free T4 (15 GB)** | Qwen2.5-0.5B-Instruct | 8–16 | ~2 min/iter. Kills after 12 hours. |
| **Kaggle P100 (16 GB)** | Qwen2.5-1.5B-Instruct | 16 | ~1.5 min/iter. 30 hrs/week free. |
| **HF Spaces ZeroGPU (A10G)** | Qwen2.5-3B-Instruct | 16–32 | Hackathon compute — reserve for final run. |
| **HF Spaces A100** | Qwen2.5-3B-Instruct | 32 | Fastest. Only if credits allow. |

The training script uses a **single-turn GRPO formulation**: the model emits a full JSON-action trajectory in one generation; the reward function replays it against a fresh copy of the environment and returns the multi-head composite score. No env server is required during training — everything runs in-process.

---

## 1. Pre-training checklist

Before touching GPUs, verify locally:

```bash
cd incident-triage-env
pip install -e .
pip install "trl>=0.15" "peft>=0.11" "transformers>=4.44" datasets accelerate bitsandbytes

# Sanity-check the environment itself
rtk proxy python -m pytest tests/            # expect 117 passed
```

Then capture your baseline numbers so you have an honest "before":

```bash
# Start the env server in one shell
rtk proxy python -m server.app

# In another shell — record baseline on the held-out eval split
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=qwen/qwen3.6-plus:free \
OPENROUTER_API_KEY=$YOUR_KEY \
python scripts/eval_before_after.py --label baseline --seeds 0 1 2
```

Artifact lands in `scripts/eval_artifacts/baseline.json`. Do **not** skip this step — without it your "Showing Improvement" score is unverifiable.

---

## 2. Local smoke test on M4 Air

One iteration, smallest model, two completions per prompt. This only proves wiring — it will not produce a useful checkpoint.

```bash
python scripts/train_grpo.py --dry-run
```

Expect ~2–5 minutes. On success the script prints:

```
[train_grpo] dataset rows=4  (TRAIN_TASK_IDS × seeds = 4×1)
...
[train_grpo] adapter saved → ./trained/grpo-run
```

Troubleshooting:

| Symptom | Fix |
|---|---|
| `torch.backends.mps.is_available()` false | Use Colab/Kaggle; MPS is only on Apple Silicon. |
| OOM on MPS | Lower `--max-new-tokens` to 256, keep `--group-size 2`. |
| TRL import errors | `pip install -U "trl>=0.15"` — the `GRPOConfig` API stabilised in 0.15. |
| Reward is always 0.001 | The model isn't emitting JSON lines. Inspect a completion with `--logging-steps 1` and check the tokenizer chat template. |

---

## 3. Colab free-T4 run (recommended for pre-hackathon testing)

Create a new Colab notebook on a **T4 GPU** runtime. Paste these cells:

### Cell 1 — clone repo + install
```python
!git clone https://github.com/<you>/OpenEnv-Hackathon.git
%cd OpenEnv-Hackathon/incident-triage-env
!pip install -e . -q
!pip install -q "trl>=0.15" "peft>=0.11" "transformers>=4.44" datasets accelerate bitsandbytes
```

### Cell 2 — sanity-check env in-process
```python
from server.environment import IncidentTriageEnv
from models import IncidentTriageAction, RootCauseCategory, Remediation

env = IncidentTriageEnv()
env.reset(task_id="easy_single_service_failure")
obs = env.step(IncidentTriageAction(
    action_type="submit_diagnosis",
    root_cause_service="payments-service",
    root_cause_category=RootCauseCategory.BAD_DEPLOY,
    remediation=Remediation.ROLLBACK_DEPLOY,
))
print("grader:", obs.grader_breakdown)
```

### Cell 3 — run GRPO
```bash
!python scripts/train_grpo.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --lora-r 16 --lora-alpha 32 \
    --num-iters 150 \
    --group-size 4 \
    --seeds 0 1 2 3 \
    --batch-size 1 --grad-accum 4 \
    --max-new-tokens 512 \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --output-dir /content/grpo-colab
```

### Cell 4 — mount the trained adapter and eval
```bash
!pip install -q vllm
# vLLM serves the base + LoRA on :8001 in OpenAI format
!nohup python scripts/serve_trained_model.py \
    --base Qwen/Qwen2.5-0.5B-Instruct \
    --adapter /content/grpo-colab \
    --port 8001 > serve.log 2>&1 &
!sleep 30 && head serve.log
```

Then run the eval harness against the served model:

```bash
!API_BASE_URL=http://localhost:8001/v1 \
  MODEL_NAME=trained-grpo \
  HF_TOKEN=dummy \
  python scripts/eval_before_after.py --label colab-grpo-0.5B --seeds 0 1 2
!python scripts/eval_before_after.py --compare baseline colab-grpo-0.5B
```

Download `scripts/eval_artifacts/*.json` before Colab disconnects.

---

## 4. Kaggle P100 run

Similar flow. Create a new Kaggle Notebook, attach a **P100 GPU** accelerator.

```bash
!pip install -e /kaggle/working/incident-triage-env -q
!pip install -q "trl>=0.15" "peft>=0.11" "transformers>=4.44" datasets accelerate bitsandbytes

!python scripts/train_grpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --lora-r 16 \
    --num-iters 300 \
    --group-size 6 \
    --seeds 0 1 2 3 4 \
    --output-dir /kaggle/working/grpo-kaggle
```

Kaggle gives 30 hrs/week of free P100 time — more than enough for a proper run.

---

## 5. HF Spaces — hackathon day

On the 25th/26th you get HF compute credits. The full walkthrough lives in **`docs/HF_DEPLOY.md`** — this section is the 30-second version.

### 5.1 Training Space setup

The repo ships **`Dockerfile.train`** which is pre-configured to run `scripts/train_grpo.py` with all knobs as env vars.

1. Create a Space: **SDK: Docker → Hardware: A10G small** (≈$1.05/hr).
2. **Enable persistent storage** in Space settings (50 GB) — required so `/data/grpo-hf` survives restarts.
3. Push your `hf-train` branch (see `docs/HF_DEPLOY.md` §4.2 for the `dockerfile_path` patching trick).
4. Set training hyperparams in **Settings → Variables**: `MODEL_NAME`, `LORA_R`, `NUM_ITERS`, `GROUP_SIZE`, etc. — all have sensible defaults baked in the Dockerfile.

### 5.2 Serve the adapter from a second Space

While training is finishing (or after), spin up a second Space for serving:

```bash
# In the serving Space's Dockerfile
CMD ["python", "scripts/serve_trained_model.py", \
     "--base", "Qwen/Qwen2.5-3B-Instruct", \
     "--adapter", "/data/grpo-hf", \
     "--port", "7860"]
```

HF exposes it on port `7860`. Point the eval harness at `https://<you>-<space>.hf.space/v1`.

### 5.3 Record the "after" numbers

```bash
API_BASE_URL=https://<you>-<serve-space>.hf.space/v1 \
MODEL_NAME=trained-grpo \
HF_TOKEN=$HF_TOKEN \
python scripts/eval_before_after.py --label hf-grpo-3B --seeds 0 1 2 3 4

python scripts/eval_before_after.py --compare baseline hf-grpo-3B
```

Paste the delta table into `SUBMISSION.md`. This is the single artifact the "Showing Improvement" score is judged on.

---

## 6. Key hyperparameter notes

- **`--group-size` (G)**: number of completions per prompt. The learning signal is the variance of rewards within a group. G=2 gives very little signal; G=4–8 is the standard range. Larger G burns VRAM linearly.
- **`--beta`**: KL coefficient. Too low (< 0.01) → the model drifts and forgets how to format JSON. Too high (> 0.1) → no exploration. 0.04 is a safe default for this env.
- **`--learning-rate`**: 5e-6 is appropriate for LoRA on Qwen-0.5B/1.5B; drop to 3e-6 for Qwen-3B.
- **`--max-new-tokens`**: trajectories average ~300–500 tokens. 512 is enough for easy/medium; bump to 768 for expert scenarios.
- **`--seeds`**: each seed produces a different scenario variant, multiplying the effective dataset. 10 seeds × 4 train tasks = 40 distinct prompts, enough to avoid memorisation in a short run.

---

## 7. Reward signal and what to watch

The reward for each rollout is the **composite grader score** (diagnosis + policy + blast + PR heads, clamped to 0.001–0.999). Watch `rewards/mean` in the TRL logs:

- **Iters 0–20**: mean ≈ 0.01–0.05 (model spamming malformed JSON).
- **Iters 20–60**: mean climbs to ~0.15–0.30 as the model learns JSON discipline.
- **Iters 60+**: mean stabilises around ~0.40–0.60 for Qwen-3B on the train split.

If the mean plateaus below 0.05 after 40 iters, something is wrong with prompting or parsing — not the RL.

Diagnostic snippet for Colab:

```python
from scripts.train_grpo import parse_actions_from_completion, replay_and_grade

sample = open("/content/grpo-colab/checkpoint-50/rollout_sample.jsonl").readlines()[0]
row = json.loads(sample)
print("completion:", row["completion"][:400])
print("parsed:", parse_actions_from_completion(row["completion"])[:3])
print("grade:", replay_and_grade(row["task_id"], row["seed"],
                                  parse_actions_from_completion(row["completion"])))
```

---

## 8. Hackathon-day execution plan

- **Before compute credits arrive** (today/tomorrow):
  1. Run `baseline` with Phase 8 harness against your prompted model.
  2. Colab T4 Qwen-0.5B run for 150 iters — gets you a working training loop + a first delta table.
  3. Record that delta as `colab-grpo-0.5B.json`.

- **On compute day**:
  1. Push repo to HF. Build training Space on A10G with Qwen-3B.
  2. Run 500 iters (~2–4 hours).
  3. Build serving Space, eval harness records `hf-grpo-3B.json`.
  4. `--compare baseline hf-grpo-3B` goes into SUBMISSION.md + the demo video.

---

## 9. Troubleshooting

| Error | Fix |
|---|---|
| `ValueError: `num_generations` must be ≥ 2` | GRPO requires G ≥ 2. Never use --group-size 1. |
| All rewards are 0.001 | Check that `IncidentTriageAction.model_validate(raw)` accepts the generated JSON — run the diagnostic in §7. |
| Reward fn called with empty completions | TRL < 0.15 has a bug with chat-template completions; upgrade. |
| HF Space OOM on Qwen-3B | Drop `--lora-r` to 16, `--group-size` to 4, enable `--gradient-checkpointing` (already on for CUDA). |
| Adapter loads but output is garbage | Check you're using the **same base model** in `serve_trained_model.py` that you trained against. |

---

## 10. Citation / reproducibility

Every artifact in `scripts/eval_artifacts/` carries the model name, timestamp, and seed set. The entire pipeline is:

```
env → rollouts → composite reward → GRPO update → LoRA adapter
  │                                                     │
  ▼                                                     ▼
baseline.json                                       finetuned.json
  │                                                     │
  └──────────── eval_before_after --compare ────────────┘
                              │
                              ▼
                    delta table → SUBMISSION.md
```

Nothing else is stored, so the run is end-to-end reproducible from git checkout + one HF Space rebuild.
