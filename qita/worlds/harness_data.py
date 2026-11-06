"""Training data in the intent-harness format (qita/harness/intent.py).

Two kinds of conversation, both split by the world seed like every other conversation:
  step B  the agent conversation, every user message wrapped in the harness header with the gold decision
  step A  one per user turn: the classification prompt and `INTENTO: <gold>`, nothing else

  python -m qita.worlds.harness_data --replay data/pilot/kept.jsonl --out data/harness/pilot
      replays verified plain conversations in their rebuilt worlds with the header added: the world, the
      tool results and the checks must all come out the same, or the conversation is dropped.
      Writes <out>.kept.jsonl (step B) and <out>.cls.jsonl (step A).
  python -m qita.worlds.harness_data --cls data/harness/gen_a.kept.jsonl --out data/harness/gen_a
      step A examples of conversations generated with --harness intent_gold (qita.worlds.sample).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

from ..harness.checks import score
from ..harness.intent import classify_prompt
from ..harness.runner import run_task
from .families import inject_fault, to_eval_task
from .families_store import STORE
from .families_v2 import BY_NAME, add_envs
from .store import attach_store

STORE_BY_NAME = {f.__name__: f for f in STORE}
from .world import build


class Replay:
    """A backend that gives back recorded assistant messages, tool calls included, one per model call."""

    def __init__(self, assistants: list[dict]):
        self.left = list(assistants)
        self.calls: list[dict] = []

    def complete(self, messages, tools):
        if not self.left:
            raise RuntimeError("replay asked for more assistant messages than recorded")
        m = self.left.pop(0)
        self.calls = [tc["function"] for tc in m.get("tool_calls") or []]
        text = (m["content"] or "") + "".join(f"<tool_call>\nR{k}\n</tool_call>" for k in range(len(self.calls)))
        return {"text": text, "gen_tokens": 1, "prompt_tokens": 0, "hit_max_tokens": False, "seconds": 0.0}

    def parse_tool_call(self, body, tools):
        c = self.calls[int(body.strip()[1:])]
        return {"name": c["name"], "arguments": c["arguments"]}


def finals(messages: list[dict]) -> list[str]:
    """The last assistant text of every user turn."""
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append("")
        elif m["role"] == "assistant" and not m.get("tool_calls") and out:
            out[-1] = m["content"] or ""
    return out


def classify_examples(conv: dict, tools: bool = False) -> list[dict]:
    """tools: the prompt of the "toolcls" harness flag, with the conversation's tools listed."""
    meta = conv["meta"]
    prev, out = [], []
    for i, (u, d, f) in enumerate(zip(meta["users"], meta["decisions"], finals(conv["messages"]))):
        out.append({"messages": [{"role": "user", "content": classify_prompt(u, prev, conv["tools"] if tools else None)},
                                 {"role": "assistant", "content": f"INTENTO: {d}"}],
                    "meta": {"seed": meta["seed"], "family": meta["family"], "kind": "classify", "turn": i}})
        prev += [("Utente", u), ("Assistente", f)]
    return out


async def replay(conv: dict) -> tuple[dict | None, str | None]:
    meta = conv["meta"]
    seed = meta["seed"]
    r = random.Random(seed)
    world, facts = build(seed, v=meta.get("world", 1))
    if meta["family"] in STORE_BY_NAME:
        attach_store(world, facts, seed)
    t = (BY_NAME | STORE_BY_NAME)[meta["family"]](facts, r)
    if meta.get("world", 1) >= 2:
        add_envs(t, seed)
    if meta.get("fault"):
        inject_fault(t)  # as qita.worlds.sample did: the first tool call failed once
    msgs = conv["messages"]
    sys_msg = msgs[0]["content"] if msgs[0]["role"] == "system" else None
    users = [m["content"] for m in msgs if m["role"] == "user"]
    if len(users) != len(t["turns"]):
        return None, "turns differ from the rebuilt task"
    et = to_eval_task(t, f"gen-{seed}", users)
    backend = Replay([m for m in msgs if m["role"] == "assistant"])
    traj = await run_task(et, backend, system=sys_msg, world=world, harness="intent_gold")
    if backend.left:
        return None, "replay ended early"
    if not score(et, traj)["success"]:
        return None, "checks fail in the rebuilt world"
    if traj["tools"] != conv["tools"]:
        return None, "tool schemas differ"
    if [m for m in traj["messages"] if m["role"] != "user"] != [m for m in msgs if m["role"] != "user"]:
        return None, "assistant or tool messages differ"
    return {"messages": traj["messages"], "tools": traj["tools"],
            "meta": {**meta, "harness": "intent_gold", "users": users,
                     "decisions": [tt["decision"] for tt in et["turns"]]}}, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="plain kept conversations to convert")
    ap.add_argument("--cls", help="harness conversations to take step A examples from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--toolcls", action="store_true", help="step A in the prompt of the toolcls flag (tools listed)")
    a = ap.parse_args()
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    convs, dropped = [], {}
    if a.replay:
        for line in Path(a.replay).read_text(encoding="utf-8").splitlines():
            new, why = asyncio.run(replay(json.loads(line)))
            if new:
                convs.append(new)
            else:
                dropped[why] = dropped.get(why, 0) + 1
        with open(f"{out}.kept.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in convs)
    else:
        convs = [json.loads(x) for x in Path(a.cls).read_text(encoding="utf-8").splitlines() if x.strip()]
    cls = [e for c in convs for e in classify_examples(c, a.toolcls)]
    with open(f"{out}.cls.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in cls)
    print(json.dumps({"conversations": len(convs), "dropped": dropped, "classify_examples": len(cls)}))


if __name__ == "__main__":
    main()
