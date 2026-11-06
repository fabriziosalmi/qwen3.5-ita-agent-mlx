"""Loss masks: only what the model generates carries loss, and it matches what inference renders."""

import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402

from qita.train.data import examples, render  # noqa: E402

try:
    TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)
except Exception:  # noqa: BLE001
    TOK = None
pytestmark = pytest.mark.skipif(TOK is None, reason="Qwen3.5-0.8B tokenizer not in the local cache")

TOOLS = [{"type": "function", "function": {"name": "read_text_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
CONV = {"tools": TOOLS, "messages": [
    {"role": "system", "content": "Sei un assistente. Oggi è mercoledì."},
    {"role": "user", "content": "Leggimi la spesa."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "read_text_file", "arguments": {"path": "spesa.txt"}}}]},
    {"role": "tool", "content": "latte\nuova SEGRETO_TOOL"},
    {"role": "assistant", "content": "Devi comprare latte e uova."},
    {"role": "user", "content": "Grazie!"},
    {"role": "assistant", "content": "Di niente!"},
]}


def decoded_targets(ex):
    runs, cur = [], []
    for t, m in zip(ex["tokens"], ex["mask"]):
        if m:
            cur.append(t)
        elif cur:
            runs.append(TOK.decode(cur))
            cur = []
    if cur:
        runs.append(TOK.decode(cur))
    return runs


def test_one_example_per_user_turn_and_only_assistant_tokens():
    exs = examples(CONV, TOK, TOK)
    assert len(exs) == 2 and all(e["aligned"] for e in exs)
    first = decoded_targets(exs[0])
    assert len(first) == 2
    assert first[0].startswith("<tool_call>") and "read_text_file" in first[0] and first[0].endswith("<|im_end|>")
    assert first[1] == "Devi comprare latte e uova.<|im_end|>"
    joined = "".join(first)
    for never in ("SEGRETO_TOOL", "Leggimi", "mercoledì", "<think>", "Read a file"):
        assert never not in joined
    assert decoded_targets(exs[1]) == ["Di niente!<|im_end|>"]


def test_context_matches_inference_prompt():
    """The tokens before the first target are exactly the prompt inference would build."""
    ex = examples(CONV, TOK, TOK)[0]
    first = ex["mask"].index(1)
    prompt = render(TOK, CONV["messages"][:2], TOOLS, generation_prompt=True)
    assert TOK.decode(ex["tokens"][:first]) == prompt
