---
name: openenv-deployment
description: >
  OpenEnv Deployment — Local, Docker, and Hugging Face Spaces. Use this skill whenever the user asks about:
  deploying an OpenEnv environment; running environments locally with Uvicorn; building and running Docker containers
  for OpenEnv; pushing environments to Hugging Face Spaces; using the openenv CLI (init, push, build);
  configuring environment variables (WORKERS, MAX_CONCURRENT_ENVS, PORT); the HF Spaces server/repository/registry
  triple; connecting a Python client to a deployed environment; WebSocket vs HTTP endpoints; sync vs async client usage;
  or the openenv.yaml manifest. Also trigger for questions about EchoEnv, from_hub(), from_docker_image(), or
  the .sync() wrapper.
---

# OpenEnv Deployment

Every deployed OpenEnv environment on Hugging Face Spaces gives you three things at once:

| Component | Access | Use as |
|-----------|--------|--------|
| **Server** | `https://<user>-<space>.hf.space` | Live agent endpoint |
| **Repository** | `pip install git+https://huggingface.co/spaces/<user>/<space>` | Typed client code |
| **Registry** | `docker pull registry.hf.space/<user>-<space>:latest` | Local container |

## Connecting to a Deployed Environment

### Async (recommended)

```python
from echo_env import EchoEnv, EchoAction

async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()
    result = await client.step(EchoAction(message="Hello"))
```

### Sync (using `.sync()` wrapper)

```python
with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello"))
```

## Available Endpoints

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/ws` | **WebSocket** | Persistent session (used by Python client) |
| `/health` | HTTP GET | Health check |
| `/reset` | HTTP POST | Reset (stateless) |
| `/step` | HTTP POST | Execute action (stateless) |
| `/state` | HTTP GET | Current state |
| `/docs` | HTTP GET | OpenAPI docs |
| `/web` | HTTP GET | Interactive web UI |

> The Python client uses `/ws` by default. HTTP endpoints are for debugging or stateless access.

```bash
curl https://openenv-echo-env.hf.space/health
# {"status": "healthy"}
```

## Install Client Package from HF Space

```bash
pip install git+https://huggingface.co/spaces/openenv/echo-env
```

This installs the typed client class, action/observation models, and any helper utilities.

## Docker Deployment

### Run from HF Spaces Registry (fastest)

```bash
docker pull registry.hf.space/openenv-echo-env:latest
docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
```

### Build Locally

```bash
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Using OpenEnv CLI
openenv build -t openenv-benchmark:latest

# Or with Docker directly
docker build -t openenv-benchmark:latest -f server/Dockerfile .
```

### Run with Environment Variables

```bash
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-env:latest
```

### Connect from Python

```python
import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    async with EchoEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(EchoAction(message="Hello"))
        print(result.observation)

asyncio.run(main())
```

### Auto-manage Docker via Client

```python
# Let the client pull and run the container automatically
client = await EchoEnv.from_hub("openenv/echo-env")
async with client:
    result = await client.reset()

# Or from a local Docker image
client = await EchoEnv.from_docker_image("my-local-image")
```

### Container Lifecycle Reference

| Method | Container | On `close()` |
|--------|-----------|--------------|
| `from_hub(repo_id)` | Auto-starts | Stops container |
| `from_hub(repo_id, use_docker=False)` | None (UV) | Stops UV server |
| `from_docker_image(image)` | Auto-starts | Stops container |
| `MyEnv(base_url=...)` | None | Disconnects only |

## Local Development with Uvicorn

```bash
# Clone and run
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark
uv sync
uv run server

# Or directly
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --reload

# Multi-worker
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

| Flag | Purpose |
|------|---------|
| `--reload` | Auto-restart on code changes |
| `--workers N` | N parallel worker processes |
| `--log-level debug` | Verbose output |

## Deploy to HF Spaces (Full Workflow)

### Step 1: Initialize

```bash
openenv init my_env
cd my_env
```

Creates the standard structure:

```
my_env/
├── server/
│   ├── app.py
│   ├── environment.py
│   └── Dockerfile
├── models.py
├── client.py
├── openenv.yaml
└── pyproject.toml
```

### Step 2: Test Locally

```bash
uv run server
curl http://localhost:8000/health
# {"status": "healthy"}
```

### Step 3: Push to HF Spaces

```bash
openenv push --repo-id username/my-env
# or private:
openenv push --repo-id username/my-env --private
```

Your environment is now live at:
- Web UI: `https://username-my-env.hf.space/web`
- API Docs: `https://username-my-env.hf.space/docs`
- Health: `https://username-my-env.hf.space/health`

### Step 4: Install the Client

```bash
uv pip install git+https://huggingface.co/spaces/username/my-env
```

### Step 5: Run Locally via Docker (Optional)

```bash
docker pull registry.hf.space/username-my-env:latest
docker run -it -p 7860:7860 --platform=linux/amd64 registry.hf.space/username-my-env:latest
```

## Configuration Reference

### `openenv.yaml` Manifest

```yaml
name: my_env
version: "1.0.0"
description: My custom environment
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 4 | Uvicorn worker processes |
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Bind address |
| `MAX_CONCURRENT_ENVS` | 100 | Max WebSocket sessions |
| `ENABLE_WEB_INTERFACE` | Auto | Enable web UI |

### Environment-Specific Variables

**TextArena:**
```bash
TEXTARENA_ENV_ID=Wordle-v0
TEXTARENA_NUM_PLAYERS=1
TEXTARENA_MAX_TURNS=6
```

**Coding Environment:**
```bash
SANDBOX_TIMEOUT=30
MAX_OUTPUT_LENGTH=10000
```

## HF Spaces Hardware Tiers

| Tier | vCPU | RAM | Cost |
|------|------|-----|------|
| CPU Basic (Free) | 2 | 16GB | Free |
| CPU Upgrade | 8 | 32GB | $0.03/hr |

## Choosing the Right Access Method

| Method | Use when | Pros | Cons |
|--------|----------|------|------|
| HF Spaces server | Quick testing, low volume | Zero setup | Latency, rate limits |
| pip from Space | Need typed classes | Type safety, IDE | Still needs a server |
| Docker locally | Dev/production, high throughput | Full control | Requires Docker |

## Resources

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Environment Hub Collection](https://huggingface.co/collections/openenv/environment-hub)
