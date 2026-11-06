"""Rejection sampling in generated worlds: the teacher plays the assistant, only verified runs are kept.

  python -m qita.worlds.sample --teacher Qwen/Qwen3.5-4B --start 0 --n 20 --out data/gen/smoke
writes <out>.kept.jsonl (training conversations) and <out>.log.jsonl (every task, with why it
was dropped). Seeds start at 1,000,000 so no generated world shares a seed with the tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

from ..harness.backend import MLXBackend
from ..harness.checks import score
from ..harness.runner import run_task
from .families import inject_fault, to_eval_task
from .families_store import STORE, STORE_NAMES
from .families_v2 import ALL, HELD_OUT, V2_ONLY, add_envs
from .phrasing import (SYSTEM_EXTRA_V2, TONES, clean, gate_system, gate_user, gate_user_v2, roughen, system_prompt_request,
                       user_prompt, user_prompt_v2)
from .style import POLICY_HINT, STYLE_HINT, conversation_problems, date_problems
from .store import attach_store
from .world import build

SEED_BASE = 1_000_000
ATTEMPTS = [dict(thinking=False, temp=0.0), dict(thinking=True, temp=0.0, max_tokens=4096),
            dict(thinking=False, temp=0.7), dict(thinking=True, temp=0.7, max_tokens=4096)]


def ask(writer: MLXBackend, prompt: str) -> str:
    return clean(writer.complete([{"role": "user", "content": prompt}], [])["text"])


def phrase_users(writer, t, r, tries=4, v=1):
    """v=2: length and tone drawn per sentence, and about 30% of sentences typed in a hurry (roughen)."""
    users, context = [], ""
    for tt in t["turns"]:
        for k in range(tries):
            writer.seed = r.randrange(1 << 30)
            prompt = user_prompt_v2(tt["intent"], tt["must_say"], r, context, tt["must_say_any"]) if v >= 2 else \
                user_prompt(tt["intent"], tt["must_say"], r.choice(TONES), context, tt["must_say_any"])
            text = ask(writer, prompt)
            why = (gate_user_v2 if v >= 2 else gate_user)(text, tt["must_say"], tt["must_not"], tt["must_say_any"])
            if not why:
                break
        else:
            return None, why
        if v >= 2 and r.random() < 0.3:
            rough = roughen(text, r, tt["must_say"] + tt["must_say_any"])
            if not gate_user(rough, tt["must_say"], tt["must_not"], tt["must_say_any"]):
                text = rough
        users.append(text)
        context += f"Utente: {text}\nAssistente: {tt['oracle']}\n"
    return users, None


def system_text(writer, now, r, tries=3, v=1):
    if r.random() < 0.15:
        return None, None  # some conversations have no system message at all
    prompt, must = system_prompt_request(now, r, SYSTEM_EXTRA_V2 if v >= 2 else None)
    for _ in range(tries):
        writer.seed = r.randrange(1 << 30)
        text = ask(writer, prompt)
        if not gate_system(text, must):
            return text, None
    return None, "system prompt gate"


def family(i: int, seed: int, weights: dict[str, float] | None, world: int = 1):
    """Held-out families are never drawn; v2-only families need a v2 world. Store families (a server never seen,
    qita/worlds/families_store.py) only by weight, in v2 worlds, so the unweighted order stays as it was."""
    fams = [f for f in ALL if f.__name__ not in HELD_OUT and (world >= 2 or f.__name__ not in V2_ONLY)]
    if not weights:
        return fams[i % len(fams)]
    fams = [f for f in fams + (STORE if world >= 2 else []) if weights.get(f.__name__, 0) > 0]
    return random.Random(f"family-{seed}").choices(fams, [weights[f.__name__] for f in fams])[0]


async def one(i: int, teacher: MLXBackend, writer: MLXBackend, log, kept, harness=None, fault_p=0.0,
              weights=None, world_v=1, attempts=len(ATTEMPTS)) -> tuple[bool, str]:
    seed = SEED_BASE + i
    r = random.Random(seed)
    world, facts = build(seed, v=world_v)
    fam = family(i, seed, weights, world_v)
    if fam.__name__ in STORE_NAMES:
        attach_store(world, facts, seed)  # training domains only: held-out domains are never generated
    t = fam(facts, r)
    if world_v >= 2:
        add_envs(t, seed)
    faulted = any(tt["plan"] for tt in t["turns"]) and random.Random(f"fault-{seed}").random() < fault_p
    if faulted:
        inject_fault(t)  # the first tool call fails once with a transient error
    row = {"i": i, "seed": seed, "family": fam.__name__, "category": t["category"], "fault": faulted, "world": world_v}
    sys_msg, err = system_text(writer, facts.now, r, v=world_v)
    users, why = (None, None) if err else phrase_users(writer, t, r, v=world_v)
    if users and (bad := [p for u in users for p in date_problems(u, facts.now)]):
        users, why = None, bad[0]  # the writer named a weekday that is not the weekday of its date
    if err or users is None:
        row.update(kept=False, reason=err or f"user gate: {why}")
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        return False, fam.__name__
    et = to_eval_task(t, f"gen-{seed}", users)
    if harness:
        et["context"] = sys_msg  # the header shows the visible system message, never the hidden hints
    for a, cfg in enumerate(ATTEMPTS[:attempts]):
        agent = teacher.clone(seed=r.randrange(1 << 30), **cfg)
        hint = f"{POLICY_HINT}\n{STYLE_HINT}"
        teacher_sys = f"{sys_msg}\n\n{hint}" if sys_msg else hint
        traj = await run_task(et, agent, system=teacher_sys, world=world, harness=harness)
        res = score(et, traj)
        msgs = traj["messages"][1:] if sys_msg is None else [{"role": "system", "content": sys_msg}] + traj["messages"][1:]
        style = conversation_problems(msgs, t["category"]) + [
            p for m in msgs if m["role"] == "assistant" for p in date_problems(m.get("content") or "", facts.now)]
        row.setdefault("attempts", []).append({
            "ok": res["success"], "style": style, "failed": [k for k, v in res["checks"].items() if not v],
            "calls": [(c["name"], c["arguments"]) for c in traj["calls"]], "finals": [x["final"] for x in traj["turns"]]})
        if res["success"] and not style:
            kept.write(json.dumps({"messages": msgs, "tools": traj["tools"],
                                   "meta": {"seed": seed, "family": fam.__name__, "attempt": a, **cfg,
                                            "harness": harness, "fault": faulted, "users": users, "world": world_v,
                                            "decisions": [tt["decision"] for tt in et["turns"]]}},
                                  ensure_ascii=False) + "\n")
            row.update(kept=True, attempt=a, users=users, system=sys_msg)
            break
    else:
        row.update(kept=False, reason="no attempt passed checks and style", failed=[k for k, v in res["checks"].items() if not v],
                   users=users)
    log.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    log.flush()
    kept.flush()
    return row["kept"], fam.__name__


async def main_async(a) -> None:
    teacher = MLXBackend(a.teacher, temp=0.0, max_tokens=1536, cache_reuse=a.cache_reuse, prefix_dir=a.prefix_cache)
    writer = teacher.clone(temp=0.8, thinking=False, max_tokens=300, cache=False)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_kept = time.perf_counter(), 0
    weights = json.loads(Path(a.weights).read_text()) if a.weights else None
    with open(f"{out}.kept.jsonl", "a", encoding="utf-8") as kept, open(f"{out}.log.jsonl", "a", encoding="utf-8") as log:
        for i in range(a.start, a.start + a.n):
            ok, name = await one(i, teacher, writer, log, kept, a.harness, a.fault_p, weights, a.world, a.attempts)
            n_kept += ok
            print(f"[{i}] {name:<20} {'KEPT' if ok else 'drop'}  "
                  f"kept {n_kept}/{i - a.start + 1}  {time.perf_counter() - t0:.0f}s", flush=True)


def gold_harness(value: str) -> str:
    from ..harness.runner import HARNESS
    if value == "intent_gold" or (value.startswith("intent_gold+") and HARNESS.fullmatch("intent" + value[11:])):
        return value
    raise argparse.ArgumentTypeError(f"not intent_gold with harness flags: {value}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-reuse", action="store_true", help="reuse the prompt cache between agent steps")
    ap.add_argument("--prefix-cache", default=None, help="folder of on-disk caches of system prompt + tool schemas")
    ap.add_argument("--harness", type=gold_harness, default=None,
                    help="wrap user messages in the intent-harness header with the gold decision; "
                         "harness flags may follow: intent_gold+cal2+when+calc")
    ap.add_argument("--fault-p", type=float, default=0.0, help="share of tasks whose first tool call fails once")
    ap.add_argument("--weights", default=None, help="JSON {family: weight}; families are drawn by weight, not in turn")
    ap.add_argument("--world", type=int, choices=[1, 2], default=1, help="world version (qita.worlds.world.build)")
    ap.add_argument("--attempts", type=int, default=len(ATTEMPTS), help="teacher attempts per task, in ATTEMPTS order")
    a = ap.parse_args()
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
