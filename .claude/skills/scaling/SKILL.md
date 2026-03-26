---
name: openenv-scaling
description: >
  OpenEnv Scaling — WebSocket sessions, providers, and multi-node deployments. Use this skill whenever the user asks
  about: scaling an OpenEnv environment to handle many concurrent sessions; WebSocket-based scaling vs HTTP;
  using UVProvider, LocalDockerProvider, or DockerSwarmProvider; configuring WORKERS and MAX_CONCURRENT_ENVS;
  load balancing multiple containers with Envoy; SLURM / multi-node RL training infrastructure;
  HF Spaces concurrency limits; benchmark results comparing local Uvicorn, Docker, HF Spaces, and SLURM;
  or throughput and latency characteristics of OpenEnv deployments. Also trigger for questions about
  "how many concurrent envs", "batch size", "sessions per core", or "scaling experiments".
---

# OpenEnv Scaling

OpenEnv uses **WebSocket connections** (`/ws`) for all environment interactions. This is the key design choice that enables efficient scaling — one container handles many isolated sessions.

## Provider Scaling (Easiest Approach)

Use `Provider` abstractions to manage replicas automatically:

```python
from openenv.providers import UVProvider, DockerSwarmProvider, LocalDockerProvider

with EchoEnv.from_hub(
    repo_id="openenv/echo-env",
    provider=DockerSwarmProvider(),
    replicas=4,
) as env:
    result = env.reset()
    result = env.step(EchoAction(message="Hello"))
```

| Provider | Runtime |
|----------|---------|
| `LocalDockerProvider()` | Docker (default) |
| `UVProvider()` | Python / Uvicorn only |
| `DockerSwarmProvider()` | Docker Swarm |

## Why WebSocket? (Not HTTP)

HTTP: each `step()` call opens a new TCP connection (~10–50ms overhead)
WebSocket: messages are lightweight frames over a persistent connection (~0.1ms overhead)

With HTTP, maintaining session state requires cookies or IDs, meaning:
```
HTTP approach: N parallel episodes → N containers (one per session)
```

With WebSocket, **one container handles many isolated sessions**:
```python
# Single container, three isolated sessions
with MyEnv(base_url="http://localhost:8000") as env1: ...  # Session 1
with MyEnv(base_url="http://localhost:8000") as env2: ...  # Session 2
with MyEnv(base_url="http://localhost:8000") as env3: ...  # Session 3
```

The server creates a fresh environment instance per WebSocket connection, with automatic cleanup on disconnect:

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    env = MyEnvironment()  # Fresh instance per connection
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        if data["type"] == "reset":
            result = env.reset()
        elif data["type"] == "step":
            result = env.step(data["action"])
        await websocket.send_json(result)
```

**Memory efficiency:**

| Approach | Containers | Memory | Startup | Max parallel |
|----------|------------|--------|---------|--------------|
| HTTP (1 env per container) | N | N × ~100MB | N × ~5s | Container-limited |
| WebSocket (N sessions per container) | 1 | ~200MB | ~5s | `MAX_CONCURRENT_ENVS` |

## Scaling a Single Container

### Uvicorn Workers

```bash
# 8 workers × 100 sessions = up to 800 concurrent sessions
WORKERS=8 uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 8
```

For simple text game environments: **~2,000 concurrent sessions with 8 workers** before degradation.

### Docker with Scaling Variables

```bash
docker run -d -p 8000:8000 \
    -e WORKERS=8 \
    -e MAX_CONCURRENT_ENVS=400 \
    --name openenv-benchmark \
    registry.hf.space/burtenshaw-openenv-benchmark:latest
```

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions per worker |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |

### HF Spaces Configuration

Set in **Space Settings → Variables**:
- `WORKERS=4` (max 4 on free tier, 8 on CPU Upgrade)
- `MAX_CONCURRENT_ENVS=100`

| Tier | vCPU | Recommended workers | Max batch (TextArena) |
|------|------|--------------------|-----------------------|
| CPU Basic (Free) | 2 | 2 | ~128 |
| CPU Upgrade | 8 | 4–8 | ~512 |

> HF Spaces free tier caps at ~128 concurrent sessions regardless of configuration.

## Multi-Container Load Balancing

When a single container hits its limit, scale horizontally with a load balancer (e.g., Envoy):

```yaml
# envoy.yaml (abbreviated)
static_resources:
  clusters:
    - name: openenv_cluster
      lb_policy: ROUND_ROBIN
      load_assignment:
        endpoints:
          - lb_endpoints:
              - endpoint: { address: { socket_address: { address: host.docker.internal, port_value: 8001 }}}
              - endpoint: { address: { socket_address: { address: host.docker.internal, port_value: 8002 }}}
              - endpoint: { address: { socket_address: { address: host.docker.internal, port_value: 8003 }}}
              - endpoint: { address: { socket_address: { address: host.docker.internal, port_value: 8004 }}}
```

```bash
# Start Envoy
docker run -d -p 8080:8080 \
    -v $(pwd)/envoy.yaml:/etc/envoy/envoy.yaml \
    --add-host=host.docker.internal:host-gateway \
    envoyproxy/envoy:v1.28.0

# Clients connect to the load balancer
with MyEnv(base_url="http://localhost:8080") as env:
    result = env.reset()
```

## Scaling Decision Guide

| Scenario | Recommended approach |
|----------|---------------------|
| Development / testing | Single container with WebSocket sessions |
| Moderate load (< 100 concurrent) | Single container, increase `MAX_CONCURRENT_ENVS` |
| High load (100–2,000 concurrent) | Single container with more workers |
| Very high load (2,000+) | Multiple containers + load balancer |
| GPU environments | One container per GPU |

## Experiment Results

Benchmarks measured with a minimal environment using configurable `wait` time to isolate infrastructure overhead.

### Summary Table

| Infrastructure | Max Batch (WebSocket) | Cores | Batch/Core | P99 Latency | RPS |
|----------------|----------------------|-------|------------|-------------|-----|
| slurm-multi | 16,384 | 96 | 170.7 | 29.8s | 518 |
| local-uvicorn | 2,048 | 8 | 256.0 | 1.97s | 932 |
| local-docker | 2,048 | 8 | 256.0 | 2.90s | 682 |
| slurm-single | 512 | 48 | 10.7 | 1.45s | 358 |
| hf-spaces | 128 | 2 | 64.0 | 2.68s | 48 |

### Finding 1: Local has highest per-core efficiency

Local Uvicorn and Docker both achieve **256 sessions/core** — the best efficiency. Both hit 2,048 concurrent sessions before degradation.

| Batch | Success | Notes |
|-------|---------|-------|
| 512 | 100% | Perfect scaling |
| 2,048 | 96.5% | Max reliable batch |
| 4,096 | 63.8% | Connections begin failing |

### Finding 2: HF Spaces reliably handles up to 128 sessions

Free tier achieves 128 concurrent WebSocket sessions with 100% success. At 256 results become unstable. At 512 it fails completely.

> **Important:** HTTP mode (`/reset`, `/step`) does **not** work on deployed HF Spaces. Use WebSocket (`/ws`) exclusively.

### Finding 3: Multi-node SLURM reaches 16,384 sessions

SLURM multi-node (96 cores, 2 nodes, Envoy load balancer) achieves the highest absolute throughput — 100% success at 16,384 concurrent sessions — but with higher connect latency (17.5s P50) due to load balancer queuing.

### Latency Breakdown (at max load, `wait=1.0s`)

| Infrastructure | Connect P50 | Step P50 | Total P99 |
|----------------|-------------|----------|-----------|
| slurm-single | 0.26s | 1.00s | 1.33s |
| local-uvicorn | 0.58s | 1.05s | 1.95s |
| hf-spaces | 0.79s | 1.10s | 2.48s |
| local-docker | 1.38s | 1.05s | 2.90s |
| slurm-multi | 17.5s | 2.42s | 26.3s |

Step latency is consistent (~1.0s) across all setups — the benchmark is measuring infrastructure overhead accurately.

## Run Your Own Benchmark

```bash
git clone https://huggingface.co/spaces/burtenshaw/openenv-scaling
cd openenv-scaling

python tests/test_scaling.py \
    --url http://localhost:8000 \
    --requests-grid 32,128,512,2048,4096,8192,16384 \
    --wait-grid 1.0,5.0,10.0 \
    --reps 3 \
    --mode ws \
    --output-dir experiments/results/
```

## Recommendations

1. **Development / moderate workloads (<2,000 concurrent):** Single-node Uvicorn or Docker with 256 sessions/core efficiency.
2. **Demos and published environments:** HF Spaces free tier works reliably up to 128 sessions.
3. **Large-scale training (>2,000 concurrent):** Multi-node with proper load balancing, expect ~170 sessions/core.

## Resources

- [OpenEnv Scaling Experiments](https://github.com/burtenshaw/openenv-scaling)
- [Envoy Proxy Docs](https://www.envoyproxy.io/docs/envoy/latest/)
- [HF Spaces Hardware](https://huggingface.co/docs/hub/spaces-overview)
