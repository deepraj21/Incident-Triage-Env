---
name: openenv-environments
description: >
  OpenEnv Environments — Production RL Made Simple. Use this skill whenever the user asks about:
  building, wrapping, or understanding OpenEnv RL environments; the OpenEnv architecture (client/server/models pattern);
  type-safe RL observations and actions with dataclasses; integrating OpenSpiel games (Catch, Tic-Tac-Toe, Kuhn Poker,
  Cliff Walking, 2048, Blackjack); comparing policies; creating custom environment integrations;
  or the difference between OpenEnv and traditional OpenAI Gym. Also trigger for questions about the HTTPEnvClient,
  FastAPI env server, Environment ABC, reset/step/state loop, or Dockerfile patterns for RL environments.
---

# OpenEnv Environments

OpenEnv is a production-ready framework for Reinforcement Learning environments built around three principles: **type safety**, **Docker isolation**, and a **clean HTTP API**. Think of RL environments as microservices — isolated, versioned, language-agnostic, and scalable.

## Why OpenEnv Over Traditional Gym?

| Challenge | Traditional (Gym) | OpenEnv |
|-----------|-------------------|---------|
| **Type Safety** | `obs[0][3]` — unclear | `obs.info_state` — IDE-friendly |
| **Isolation** | Same process (crash risk) | Docker containers |
| **Deployment** | "Works on my machine" | Same container everywhere |
| **Scaling** | Hard to distribute | Kubernetes-ready |
| **Language** | Python only | Any language via HTTP |

## The Core RL Loop

```python
while not done:
    observation = environment.observe()
    action = policy.choose(observation)
    reward = environment.step(action)
    policy.learn(reward)
```

## The 3-Component Pattern

Every OpenEnv environment has exactly three components:

```
src/envs/your_env/
├── models.py          ← Type-safe contracts (Action, Observation, State)
├── client.py          ← What your training code imports (HTTPEnvClient)
└── server/
    ├── environment.py ← Game/simulation logic (implements Environment ABC)
    ├── app.py         ← FastAPI server (auto-created via create_fastapi_app)
    └── Dockerfile     ← Container definition
```

### Server Side (runs in Docker)

```python
from core.env_server import Environment, Action, Observation, State

class MyEnvironment(Environment):
    @abstractmethod
    def reset(self) -> Observation:
        """Start new episode"""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute action, return observation"""

    @property
    def state(self) -> State:
        """Get episode metadata"""
```

### Client Side (your training code)

```python
from core.http_env_client import HTTPEnvClient

class MyEnv(HTTPEnvClient[MyAction, MyObservation]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"action_value": action.action_value}

    def _parse_result(self, payload: dict) -> StepResult:
        return StepResult(
            observation=MyObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(...)
```

The base class handles all HTTP communication. You focus on RL.

## OpenSpiel Integration

OpenEnv ships with 6 OpenSpiel games — same client interface for all:

| Single-Player | Multi-Player |
|---------------|--------------|
| Catch | Tic-Tac-Toe |
| Cliff Walking | Kuhn Poker |
| 2048 | |
| Blackjack | |

### Using OpenSpiel Environments

```python
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import OpenSpielAction, OpenSpielObservation

env = OpenSpielEnv(base_url="http://localhost:8000")

result = env.reset()
# Returns StepResult[OpenSpielObservation] — fully typed

result = env.step(OpenSpielAction(action_id=2, game_name="catch"))
# Type checker validates this is correct

state = env.state()
# Returns OpenSpielState with episode metadata
```

### OpenSpielObservation Fields

- `info_state: List[float]` — flattened grid / game state
- `legal_actions: List[int]` — valid moves at this step
- `done: bool` — episode finished?
- `reward: Union[bool, int, float, None]`
- `current_player_id: int`
- `opponent_last_action: Optional[int]`
- `game_phase: str`

### Switching Games

Same client, different game — just change the server env var:

```bash
# Start tic-tac-toe instead of catch
OPENSPIEL_GAME=tic_tac_toe uvicorn envs.openspiel_env.server.app:app ...
```

## Creating a Custom Integration (5 Steps)

### Step 1: Define Types (`models.py`)

```python
from dataclasses import dataclass
from core.env_server import Action, Observation, State

@dataclass
class YourAction(Action):
    action_value: int

@dataclass
class YourObservation(Observation):
    state_data: List[float]
    done: bool
    reward: float

@dataclass
class YourState(State):
    episode_id: str
    step_count: int
```

### Step 2: Implement Server Environment (`server/environment.py`)

```python
from core.env_server import Environment

class YourEnvironment(Environment):
    def reset(self) -> YourObservation:
        # Initialize game state
        return YourObservation(...)

    def step(self, action: YourAction) -> YourObservation:
        # Execute action, compute reward
        return YourObservation(...)

    @property
    def state(self) -> YourState:
        return self._state
```

### Step 3: Create HTTP Client (`client.py`)

```python
from core.http_env_client import HTTPEnvClient

class YourEnv(HTTPEnvClient[YourAction, YourObservation]):
    def _step_payload(self, action: YourAction) -> dict:
        return {"action_value": action.action_value}

    def _parse_result(self, payload: dict) -> StepResult:
        return StepResult(
            observation=YourObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )
```

### Step 4: Register the Server (`server/app.py`)

```python
from core.env_server import create_fastapi_app
from .your_environment import YourEnvironment

env = YourEnvironment()
app = create_fastapi_app(env)  # OpenEnv creates all endpoints automatically
```

### Step 5: Dockerize (`server/Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Reference Examples in the Repo

- `src/envs/echo_env/` — simplest environment (great starting point)
- `src/envs/openspiel_env/` — wraps external library (6 games in one)
- `src/envs/coding_env/` — Python code execution (complex use case)

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE                                        │
│  env = OpenSpielEnv(...)     ← Import the client          │
│  result = env.reset()        ← Type-safe!                 │
│  result = env.step(action)   ← Type-safe!                 │
└─────────────────┬──────────────────────────────────────────┘
                  │  HTTP/JSON  POST /reset, POST /step, GET /state
┌─────────────────▼──────────────────────────────────────────┐
│  DOCKER CONTAINER                                          │
│  FastAPI Server                                            │
│  └─ Environment (reset, step, state)                       │
│     └─ Your Game/Simulation Logic                          │
│  Isolated • Reproducible • Secure                          │
└────────────────────────────────────────────────────────────┘
```

## Resources

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [OpenSpiel](https://github.com/google-deepmind/open_spiel)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Environment Creation Guide](src/envs/README.md)
