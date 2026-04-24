"""Serve a trained LoRA adapter as an OpenAI-compatible endpoint.

After training finishes in scripts/train_grpo.py, point inference.py at this
server to produce the "after" numbers for Phase 8.

Usage:
    python scripts/serve_trained_model.py \\
        --base Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter ./trained/grpo-run \\
        --port 8001

    # then in another shell:
    API_BASE_URL=http://localhost:8001/v1 \\
    MODEL_NAME=trained-grpo \\
    HF_TOKEN=dummy \\
    python scripts/eval_before_after.py --label finetuned --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"Missing deps: {e}. Install: pip install torch transformers peft vllm")
    sys.exit(1)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base model id (same as training)")
    p.add_argument("--adapter", required=True, help="LoRA adapter directory")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--merge", action="store_true",
                   help="merge LoRA into base weights before serving (faster inference)")
    return p


def main():
    args = build_argparser().parse_args()

    # Launch an OpenAI-compatible server. vLLM is the cleanest option; if not
    # installed, fall back to a minimal transformers-based FastAPI shim.
    try:
        from vllm.entrypoints.openai.api_server import run_server  # noqa: F401
        import subprocess
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.base,
            "--enable-lora",
            "--lora-modules", f"trained-grpo={args.adapter}",
            "--host", args.host,
            "--port", str(args.port),
            "--max-model-len", "4096",
        ]
        print("[serve] launching vLLM:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return
    except ImportError:
        pass

    # Fallback: transformers + FastAPI (slower but dependency-light).
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    print("[serve] vLLM not found — using transformers fallback (slower)")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    if args.merge:
        model = model.merge_and_unload()
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    app = FastAPI()

    class Msg(BaseModel):
        role: str
        content: str

    class ChatReq(BaseModel):
        model: str = "trained-grpo"
        messages: list[Msg]
        temperature: float = 0.1
        max_tokens: int = 512
        response_format: dict | None = None

    @app.post("/v1/chat/completions")
    def chat(req: ChatReq):
        prompt = tok.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=req.temperature > 0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {
            "id": "local-trained",
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
