# HF Spaces Deployment Guide

End-to-end walkthrough for deploying the Incident Triage environment — both as a **serving Space** (the env itself) and a **training Space** (GRPO on HF compute).

The repo carries two Dockerfiles:

| File | Purpose | Runs |
|---|---|---|
| `Dockerfile` | Serving Space — the env API | `uvicorn server.app:app` on port 8000 |
| `Dockerfile.train` | Training Space — GRPO run | `python scripts/train_grpo.py` |

You deploy each to a **separate** HF Space. The training Space writes its LoRA adapter to `/data`, then a serving Space (or your local eval) points at that adapter.

---

## 1. Prerequisites

- Hugging Face account with a write token: https://huggingface.co/settings/tokens
- `huggingface_hub` CLI installed locally: `pip install "huggingface_hub[cli]"`
- Authenticated: `huggingface-cli login`

Optional but recommended:

```bash
# The openenv CLI offers `openenv push` — works, but the manual git path
# below is easier to debug if something goes wrong.
pip install openenv-core
```

---

## 2. Local verification before pushing

Always smoke-test locally first — a broken Space takes 5+ minutes to surface on HF.

### 2.1 Build + run the serving image locally

```bash
cd incident-triage-env

# build
docker build -t incident-triage-env:local .

# run
docker run --rm -p 8000:8000 incident-triage-env:local
```

In another shell:

```bash
python scripts/smoke_test_hf_space.py --url http://localhost:8000
```

**Expected output**:

```
[1] GET /tasks
  ✓ tasks endpoint  8 tasks
[2] WS /ws  reset + step(legacy) + step(multi-app) + submit
  ✓ reset  alert on payments-service
  ✓ state  episode_id=...
  ✓ step(legacy: check_status)  sim_time=...
  ✓ step(multi-app: alerthub.list_alerts)  result-len=...
  ✓ submit_diagnosis  score=0.xxx
[3] POST /grader  (fetch final score by episode_id)
  ✓ grader response shape  heads=['diagnosis', 'policy', 'blast', 'pr']

✅ all surfaces healthy — Space is ready for inference / training
```

If **any** step fails, fix before pushing. The live Space runs the exact same image, so the failure will repeat.

### 2.2 Build the training image locally (sanity only — training needs GPU)

```bash
docker build -f Dockerfile.train -t incident-triage-train:local .
```

Expect a 5–8 GB image (torch + transformers + trl wheels). Don't attempt to `docker run` this without a GPU — it'll spin forever on CPU.

---

## 3. Deploy the Serving Space

### 3.1 Create the Space (once)

1. Go to https://huggingface.co/new-space.
2. Owner: your username. Name: `incident-triage-env`.
3. **SDK: Docker**.
4. Hardware: **CPU Basic** is sufficient (the env has no model — it's pure Python simulation).
5. Visibility: Public.
6. Click **Create Space**.

HF creates an empty repo at `https://huggingface.co/spaces/<you>/incident-triage-env`.

### 3.2 Push the code

```bash
# from repo root — add HF as a second remote alongside GitHub
git remote add hf https://huggingface.co/spaces/<you>/incident-triage-env

# push current branch to HF's main
git push hf <your-branch>:main
```

HF auto-builds the image. Watch progress at:
`https://huggingface.co/spaces/<you>/incident-triage-env/logs`

First build is ~2 minutes. Subsequent builds use Docker layer cache.

### 3.3 Verify the live Space

```bash
python scripts/smoke_test_hf_space.py \
    --url https://<you>-incident-triage-env.hf.space \
    --verbose
```

Same output as §2.1 — if you see the green check, the env is live.

### 3.4 Wire the eval harness at the live Space

```bash
ENV_URL=https://<you>-incident-triage-env.hf.space \
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=qwen/qwen3.6-plus:free \
OPENROUTER_API_KEY=$YOUR_KEY \
python scripts/eval_before_after.py --label baseline-hf --seeds 0 1 2
```

The resulting `scripts/eval_artifacts/baseline-hf.json` is your honest "before" number, measured against the actual deployed env.

---

## 4. Deploy the Training Space

Do this **only when you have compute credits allocated** — a GPU Space bills by the minute.

### 4.1 Create the training Space

1. https://huggingface.co/new-space → new Space.
2. Name: `incident-triage-train` (separate from the serving Space).
3. **SDK: Docker**.
4. Hardware: **A10G small** (≈$1.05/hr) is the sweet spot for Qwen-3B. ZeroGPU is fine for smoke tests but idles frequently.
5. **Enable persistent storage** in Space settings → **Persistent disk → 50GB**. This is required — without it your adapter vanishes when the Space restarts.
6. Click **Create Space**.

### 4.2 Tell HF to use `Dockerfile.train`

Add a `dockerfile_path` entry to the training Space's README frontmatter. Because this conflicts with the serving Space's README, we maintain the training README separately via a one-liner at push time.

The cleanest pattern: create a dedicated branch for the training Space.

```bash
# from the env repo root
git checkout -b hf-train

# overwrite README.md frontmatter to point HF at Dockerfile.train
python -c "
import re, pathlib
p = pathlib.Path('README.md')
txt = p.read_text()
new = re.sub(
    r'^---\\n(.*?)\\n---',
    lambda m: '---\\n' + m.group(1)
        + '\\ndockerfile_path: Dockerfile.train'
        + '\\ntitle_suffix: training\\n---',
    txt, count=1, flags=re.DOTALL)
p.write_text(new)
print('patched')
"

git commit -am "hf-train: point Space at Dockerfile.train"

# push to the training Space remote
git remote add hf-train https://huggingface.co/spaces/<you>/incident-triage-train
git push hf-train hf-train:main
```

### 4.3 Configure training hyperparameters in the Space UI

Before the build finishes, open **Settings → Variables and secrets** on the training Space and set:

| Variable | Default in Dockerfile | Notes |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Downgrade to `Qwen/Qwen2.5-1.5B-Instruct` for T4 |
| `LORA_R` | `32` | Drop to `16` under VRAM pressure |
| `NUM_ITERS` | `500` | Start with `100` for a faster first run |
| `GROUP_SIZE` | `8` | `G` in GRPO; must divide BATCH_SIZE |
| `BATCH_SIZE` | `8` | per_device_train_batch_size |
| `GRAD_ACCUM` | `4` | effective batch = BATCH_SIZE × GRAD_ACCUM |
| `LEARNING_RATE` | `3e-6` | Drop to `2e-6` if loss oscillates |
| `BETA` | `0.04` | KL coefficient |
| `SEEDS` | `"0 1 2 3 4 5 6 7 8 9"` | variant seeds → effective 40 training prompts |
| `MAX_NEW_TOKENS` | `768` | Trajectories fit in ≤512 for easy, ≤768 for expert |
| `OUTPUT_DIR` | `/data/grpo-hf` | Must be under `/data` for persistence |

### 4.4 Watch training

- Stream logs at `https://huggingface.co/spaces/<you>/incident-triage-train/logs`.
- First build: ~6 minutes (torch + trl wheels).
- Training: see `docs/TRAINING.md` §6 "Reward signal and what to watch".
- The Space auto-restarts when it finishes; to avoid paying for idle GPU after training, **pause the Space** from its settings page as soon as you see `[train_grpo] adapter saved → /data/grpo-hf`.

### 4.5 Retrieve the adapter

Option A — download via HF Hub:

```bash
# adapter artifacts sit inside the Space's persistent storage; HF exposes
# them through the /files endpoint.
huggingface-cli download \
    spaces/<you>/incident-triage-train \
    --local-dir ./trained/grpo-hf \
    --include "grpo-hf/**"
```

Option B — serve directly from a third Space (no download needed):

1. Create a new Docker Space `incident-triage-serve`.
2. Base image: same as `Dockerfile.train` (has torch + peft).
3. Use `scripts/serve_trained_model.py --base $MODEL_NAME --adapter /data/grpo-hf --port 7860`.
4. Point `ENV_URL` at this Space for eval.

### 4.6 Record the "after" numbers

```bash
# against the served trained model
API_BASE_URL=https://<you>-incident-triage-serve.hf.space/v1 \
MODEL_NAME=trained-grpo \
HF_TOKEN=$HF_TOKEN \
ENV_URL=https://<you>-incident-triage-env.hf.space \
python scripts/eval_before_after.py --label finetuned-hf --seeds 0 1 2 3 4

python scripts/eval_before_after.py --compare baseline-hf finetuned-hf

# generate the submission plot
python scripts/plot_results.py --compare baseline-hf finetuned-hf
```

---

## 5. Scaling knobs (serving Space)

The serving Space is CPU-only and small — the real bottleneck is concurrent sessions. The Dockerfile exposes two env vars:

| Variable | Effect |
|---|---|
| `WORKERS` | uvicorn worker processes. `1` is fine for a demo. `4` if you're running batched rollouts from multiple clients. |
| `MAX_CONCURRENT_ENVS` | upper bound on simultaneously-active episodes per worker. |
| `PORT` | service port. Leave at `8000` to match `app_port: 8000` in the README frontmatter. |

Set via the Space's Variables UI. **Don't** bump `WORKERS` past 4 on CPU Basic hardware — you'll OOM.

---

## 6. Common failure modes

| Symptom | Fix |
|---|---|
| HF build fails with `ModuleNotFoundError: openenv` | The build env couldn't reach GitHub. Rebuild — usually transient. |
| Space stuck in "Building" >10 min | HF queue congestion. Watch the build log; `Cancel & Retry` often unblocks. |
| `/tasks` returns 404 on the live Space | The uvicorn command didn't start. Check HF logs for a Python traceback. |
| WebSocket `/ws` returns 502 | HF's WS proxy is strict — ensure `websockets>=11` is in `pyproject.toml` (it is). |
| Training Space OOMs on Qwen-3B | Drop `LORA_R=16`, `BATCH_SIZE=4`, `GROUP_SIZE=4`. |
| Training Space finishes but `/data` is empty | Persistent disk not enabled; the `/data` mount was ephemeral. Enable it and re-run. |
| `smoke_test_hf_space.py` passes locally, fails on HF | Usually the Space is still in "Building". Wait 60s; HF flips a flag from 503 → 200 when ready. |

---

## 7. Hackathon-day checklist

- [ ] Serving Space deployed and smoke-test passes
- [ ] `baseline-hf.json` recorded against the live Space
- [ ] Training Space created with `Dockerfile.train` and persistent storage enabled
- [ ] Training run completed (watch for `adapter saved → /data/grpo-hf` in logs)
- [ ] Serving Space for the trained adapter deployed
- [ ] `finetuned-hf.json` recorded
- [ ] `scripts/eval_before_after.py --compare baseline-hf finetuned-hf` → delta table
- [ ] `scripts/plot_results.py --compare baseline-hf finetuned-hf` → PNG
- [ ] PNG + delta table pasted into `SUBMISSION.md`
- [ ] Training Space paused (stop billing)

---

## 8. What goes in SUBMISSION.md

The three artifacts judges look for:

1. **Live Space URL** — `https://<you>-incident-triage-env.hf.space` with the `/tasks` endpoint reachable.
2. **Before/after delta table** (from `--compare`) — this is the entire "Showing Improvement" score.
3. **Oracle ceiling + before/after plots** (from `scripts/plot_results.py`).

That's it. Everything else (training recipe, env design) is judged through the video + the code on the Space.
