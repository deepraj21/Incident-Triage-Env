---
name: openenv-training
description: >
  OpenEnv Training — GRPO RL fine-tuning with TRL and Wordle. Use this skill whenever the user asks about:
  training an LLM with reinforcement learning using OpenEnv; GRPO (Group Relative Policy Optimization) with TRL;
  the Wordle environment in TextArena; defining rollout functions for GRPOTrainer; reward function design
  (correctness, green/yellow tile scores, repetition penalties); the rollout_once loop; using vLLM in colocate mode
  for generation; GRPOConfig parameters; fine-tuning Qwen or other models on text games; multi-turn RL episodes
  with an agentic environment; or the TextArena / OpenEnv integration with TRL. Also trigger for questions about
  GRPOTrainer, rollout_func, generate_rollout_completions, TextArenaEnv, or system prompts for Wordle.
---

# OpenEnv Training with GRPO

This skill covers training an LLM to play Wordle using **GRPO** (Group Relative Policy Optimization) via **TRL**, with the **TextArena environment** from OpenEnv as the RL feedback loop.

## What is This?

An **agentic RL environment** lets a model take actions, observe outcomes, and improve — similar to human trial-and-error learning. Here, the model:
1. Generates word guesses based on the game state
2. Receives color-coded feedback from the Wordle environment
3. Improves its strategy through reward signals

[Wordle](https://en.wikipedia.org/wiki/Wordle): guess a 5-letter word in 6 tries. Each letter gets: 🟩 (right place), 🟨 (wrong place), ⬛ (not in word).

## Setup

```python
# Install TRL, OpenEnv, tracking, and efficient generation
!pip install -Uq git+https://github.com/huggingface/trl.git \
    git+https://github.com/meta-pytorch/OpenEnv.git \
    trackio vllm==0.10.2 bitsandbytes

# Log in to HF Hub (to save model + track runs)
from huggingface_hub import notebook_login
notebook_login()
```

## Initialize the Environment

```python
from envs.textarena_env import TextArenaEnv

# Connect to hosted Space (for dev/testing)
env = TextArenaEnv(base_url="https://burtenshaw-textarena.hf.space")

# For production: run locally via Docker (better concurrency)
# docker run -d -p 8000:8000 registry.hf.space/burtenshaw-textarena:latest
# env = TextArenaEnv(base_url="http://localhost:8000")
```

> For high-throughput training, duplicate the Space or run locally — hosted HF Spaces have limited concurrency.

## Model and Tokenizer

```python
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"  # Small but capable; scale up if hardware allows
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

## System Prompt

Define the agent's behavior — expert Wordle strategy with a strict output format:

```python
system_prompt = """
You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

## GAME RULES
1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN: Letter is correct and in the correct position
   - YELLOW: Letter is in the word but in the wrong position
   - GRAY: Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed

## RESPONSE FORMAT
Only respond with your next guess in square brackets, e.g., [crane].

## STRATEGIC APPROACH
### Opening Strategy
- Start with words rich in common vowels (A, E, I, O, U) and consonants (R, S, T, L, N)
- Optimal starters: CRANE, SLATE, STARE, AROSE, IRATE

### Mid-Game Strategy
- Use confirmed GREEN letters in their correct positions
- Place YELLOW letters in different positions than where they appeared
- Eliminate GRAY letters from consideration
"""
```

## Rollout Function

The rollout function defines how the agent interacts with the environment during each GRPO training step. GRPOTrainer calls it automatically.

```python
def rollout_func(prompts, trainer=None):
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    correctness_rewards = []
    green_rewards = []
    yellow_rewards = []
    repetition_rewards = []

    for prompt_text in prompts:
        episode = rollout_once(
            trainer=trainer,
            env=env,
            tokenizer=tokenizer,
            dataset_prompt=prompt_text,
            system_prompt=system_prompt,
            max_turns=6,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        correctness_rewards.append(episode["correct_reward"])
        green_rewards.append(episode["green_reward"])
        yellow_rewards.append(episode["yellow_reward"])
        repetition_rewards.append(episode["repetition_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "correct_reward": correctness_rewards,
        "green_reward": green_rewards,
        "yellow_reward": yellow_rewards,
        "repetition_reward": repetition_rewards,
    }
```

## Single Episode Loop (`rollout_once`)

One full Wordle game — up to 6 turns of guess → feedback → learn:

```python
from collections import defaultdict
from envs.textarena_env import TextArenaAction
from envs.textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback
from trl.experimental.openenv import generate_rollout_completions

def rollout_once(trainer, env, tokenizer, dataset_prompt, system_prompt, max_turns):
    result = env.reset()
    observation = result.observation

    prompt_ids, completion_ids, logprobs = [], [], []
    green_scores, yellow_scores, repetition_scores, correct_scores = [], [], [], []
    guess_counts = defaultdict(int)

    for _turn in range(max_turns):
        if result.done:
            break

        # Build prompt from current game state
        user_prompt = make_user_prompt(observation.prompt or dataset_prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

        # Generate via vLLM (colocate mode)
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # Parse guess and step the environment
        guess = extract_guess(completion_text)
        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

        # Compute reward components
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)
        previous_occurrences = guess_counts[guess]
        repetition_score = scale_repetition_score(previous_occurrences, len(guess_counts))
        guess_counts[guess] += 1

        if feedback:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0
        else:
            green_score = yellow_score = 0.0

        correct_scores.append(correct_score)
        green_scores.append(green_score)
        yellow_scores.append(yellow_score)
        repetition_scores.append(repetition_score)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "correct_reward": correct_scores[-1] if correct_scores else 0.0,
        "green_reward": green_scores[-1] if green_scores else 0.0,
        "yellow_reward": yellow_scores[-1] if yellow_scores else 0.0,
        "repetition_reward": repetition_scores[-1] if repetition_scores else 0.0,
    }
```

## Helper Functions

```python
def make_user_prompt(prompt_text, messages):
    """Build structured user prompt from task description + message history."""
    history = format_history(messages)
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )

def format_history(messages):
    """Format message history with category tags."""
    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if content:
            lines.append(f"[{tag}] {content}")
    return "\n".join(lines)

def scale_repetition_score(previous_occurrences, max_occurrences):
    """Penalize repeated guesses — higher score = less repetition."""
    if max_occurrences == 0:
        return 0.0
    return (max_occurrences - previous_occurrences) / max_occurrences
```

## Reward Functions

Four separate reward signals give the trainer nuanced feedback:

```python
def reward_correct(completions, **kwargs):
    """+1 if the word was guessed correctly, else 0."""
    rewards = kwargs.get("correct_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

def reward_greens(completions, **kwargs):
    """Fraction of letters in the correct position (0–1)."""
    rewards = kwargs.get("green_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

def reward_yellows(completions, **kwargs):
    """Fraction of letters present but in wrong position (0–1)."""
    rewards = kwargs.get("yellow_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)

def reward_repetition(completions, **kwargs):
    """Penalizes repeated guesses — encourages diverse exploration."""
    rewards = kwargs.get("repetition_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)
```

## Dataset

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Play Wordle like an expert."] * 1000
})
```

## GRPO Configuration

```python
from trl import GRPOConfig

grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=64,
    per_device_train_batch_size=1,
    warmup_steps=20,
    num_generations=2,
    max_completion_length=8,
    max_prompt_length=1400,
    use_vllm=True,
    vllm_mode="colocate",              # vLLM runs in the same process
    vllm_gpu_memory_utilization=0.1,
    output_dir="wordle-grpo-Qwen3-1.7B",
    report_to="trackio",
    trackio_space_id="wordle-grpo-Qwen3-1.7B",
    logging_steps=1,
    save_steps=10,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    push_to_hub=True,
)
```

## Train

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model_name,
    processing_class=tokenizer,
    reward_funcs=[reward_correct, reward_greens, reward_yellows, reward_repetition],
    train_dataset=dataset,
    args=grpo_config,
    rollout_func=rollout_func,
)

trainer_stats = trainer.train()

# Save and push to Hub
env.close()
trainer.save_model("wordle-grpo-Qwen3-1.7B")
trainer.push_to_hub()
```

**Memory profile (A100 40GB):** ~36.7 GB peak (~93% of max), ~87 minutes for 31 steps.

## Run Inference with the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    "your-username/wordle-grpo-Qwen3-1.7B",
    dtype="auto", device_map="auto"
)

def play_wordle(env, model, tokenizer):
    result = env.reset()
    observation = result.observation
    print("Initial Prompt:\n" + observation.prompt)

    for turn in range(6):
        if result.done:
            break
        user_prompt = make_user_prompt(observation.prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=512)[0][len(inputs.input_ids[0]):]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        guess = extract_guess(generated_text)

        print(f"\nTurn {turn}: {generated_text}")
        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

try:
    play_wordle(env, fine_tuned_model, tokenizer)
finally:
    env.close()
```

## Improving Training Results

The model learns good opening strategies (CRANE → SPARE) but may still repeat guesses. To improve:
- **Longer training runs** — more episodes to explore and refine
- **Stronger repetition penalties** — increase weight of `reward_repetition`
- **Better reward shaping** — reward partial progress more aggressively
- **Larger models** — scale up from Qwen3-1.7B

## Resources

- [TRL GitHub](https://github.com/huggingface/trl)
- [TRL OpenEnv Docs](https://huggingface.co/docs/trl/main/en/openenv)
- [TRL Official Examples](https://huggingface.co/docs/trl/example_overview)
- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
