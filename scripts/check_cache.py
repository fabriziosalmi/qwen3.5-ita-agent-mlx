"""The prompt cache gives the same next-token logits as a full prefill, on the real model.

  uv run python scripts/check_cache.py [MODEL]
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, so qita imports
from qita.harness.backend import PromptCache  # noqa: E402

model, tok = load(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B")
tools = [{"type": "function", "function": {"name": "read_text_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
msgs = [{"role": "system", "content": "Sei un assistente. Oggi è mercoledì 4 novembre 2026."},
        {"role": "user", "content": "Leggimi la lista della spesa."}]
step2 = msgs + [{"role": "assistant", "content": "", "tool_calls": [
    {"type": "function", "function": {"name": "read_text_file", "arguments": {"path": "spesa.txt"}}}]},
    {"role": "tool", "content": "latte\nuova\npane"}]


def enc(m):
    s = tok.apply_chat_template(m, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return s, tok.encode(s, add_special_tokens=False)


def full_logits(p):
    c = make_prompt_cache(model)
    out = model(mx.array([p]), cache=c)
    return out[0, -1].astype(mx.float32)


pc = PromptCache(model, prefix_dir=sys.argv[2] if len(sys.argv) > 2 else None)
for name, m in (("step1", msgs), ("step2", step2)):
    s, p = enc(m)
    pre = tok.encode(s[:s.index("<|im_start|>user")], add_special_tokens=False)
    cache, reused = pc.prepare(p, len(pre))
    got = model(mx.array([p[-1:]]), cache=cache)[0, -1].astype(mx.float32)
    ref = full_logits(p)
    d = mx.abs(got - ref).max().item()
    print(f"{name}: prompt {len(p)} tokens, reused {reused}, max |logit diff| {d:.4f}, "
          f"same argmax {mx.argmax(got).item() == mx.argmax(ref).item()}, stats {pc.stats}")
