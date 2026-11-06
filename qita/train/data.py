"""Training examples from conversations: the model's own chat template, loss only where the model speaks.

A conversation is {"messages": [...], "tools": [...]}, OpenAI-style, with assistant tool calls
in `tool_calls` and tool results as role "tool". It becomes one example per user turn: the
context is everything before that turn (rendered as the template renders history), the targets
are the assistant messages that answer it. This matches inference exactly: the template puts the
empty thinking block only on assistant messages after the last user message, and at inference
that block is part of the generation prompt, so it is never a target.

Targets are the characters the model generates: from right after the generation prompt to the
end of `<|im_end|>`. Tool results, schemas, system and user text never carry loss.
"""

from __future__ import annotations

import re

IM_END = "<|im_end|>"


def render(tok, messages: list[dict], tools: list[dict] | None, generation_prompt: bool) -> str:
    return tok.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                   add_generation_prompt=generation_prompt, enable_thinking=False)


def with_tools(conv: dict, extra: list[dict]) -> dict:
    """The conversation as if the harness had also offered `extra`: their schemas join the end of the tool
    list and their names the end of the "Strumenti disponibili" line of each intent header (as run_task
    appends a harness env). Tools already offered are left as they are."""
    have = {t["function"]["name"] for t in conv.get("tools") or []}
    new = [t for t in extra if t["function"]["name"] not in have]
    if not new:
        return conv
    names = ", ".join(t["function"]["name"] for t in new)

    def header(m: dict) -> dict:
        if m["role"] != "user" or not m["content"].startswith("[CONTESTO]"):
            return m
        text = re.sub(r"(Strumenti disponibili: )nessuno\.", rf"\g<1>{names}.", m["content"], count=1)
        text = re.sub(r"(Strumenti disponibili: [^\n]*?)\.\n", rf"\g<1>, {names}.\n", text, count=1) if text == m["content"] else text
        return {**m, "content": text}

    return {**conv, "messages": [header(m) for m in conv["messages"]], "tools": (conv.get("tools") or []) + new}


def examples(conv: dict, tok, hf_tok=None) -> list[dict]:
    """Tokenized examples with a 0/1 loss mask. hf_tok must be a fast tokenizer (for offsets)."""
    hf_tok = hf_tok or getattr(tok, "_tokenizer", tok)
    msgs, tools = conv["messages"], conv.get("tools")
    users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    out = []
    for n, u in enumerate(users):
        end = users[n + 1] if n + 1 < len(users) else len(msgs)
        part = msgs[:end]
        text = render(tok, part, tools, generation_prompt=False)
        spans = []
        for k in range(u + 1, end):
            if part[k]["role"] != "assistant":
                continue
            prefix = render(tok, part[:k], tools, generation_prompt=True)
            if not text.startswith(prefix):
                raise ValueError(f"context of assistant message {k} is not a prefix of the conversation")
            stop = text.index(IM_END, len(prefix)) + len(IM_END)
            spans.append((len(prefix), stop))
        if not spans:
            continue
        enc = hf_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        starts = {a for a, _ in offs}
        mask = [int(any(a >= s and b <= e for s, e in spans)) for a, b in offs]
        out.append({
            "tokens": ids,
            "mask": mask,
            "n_target": sum(mask),
            "aligned": all(s in starts for s, _ in spans),
            "spans": len(spans),
        })
    return out
