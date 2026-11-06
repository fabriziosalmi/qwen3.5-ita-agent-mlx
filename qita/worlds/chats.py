"""Round-6 chats without tools: the teacher answers

  doc      a request about an excerpt of a document written by people (data/seeds/documents.jsonl), two different
           requests per excerpt
The answer is in the pasted text, so the teacher does not need to know facts. What was tried and dropped:
open questions of fact, written by the teacher (a first trial) or by people (Aya, a second trial), are
trivia the 4B does not know and answers with invented facts; yes/no questions get the right sì/no and a wrong
explanation; multiple-choice questions written by the teacher carry false premises that three agreeing
answers do not catch (a third trial).

  python -m qita.worlds.chats --teacher M --kind doc --start 0 --n 50 --out data/chats/smoke_doc

Half of the conversations (--plain-share) are plain chats with no tools, no header and no system message, the
others carry the intent-harness header with the "rispondi" decision, as the agent sees it. The teacher gets
CHAT_HINT, not the agent policy: in a chat, "usa sempre gli strumenti, non inventare" turned into refusals of
harmless questions and of pasted documents (a trial run). The user turn is a person's question
or the teacher's text; the instructions in this file only reach the teacher and are never stored.
Writes <out>.kept.jsonl and <out>.log.jsonl, like qita.worlds.sample.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path

from ..harness.checks import score
from ..harness.italian import is_italian
from ..harness.runner import AGENT, run_task
from .families import task, to_eval_task, turn
from .families_v2 import TRAIN_ENVS
from .phrasing import clean
from .sample import SEED_BASE, ask, system_text
from .style import STYLE_HINT, conversation_problems
from .world import build

ROOT = Path(__file__).resolve().parents[2]
CHAT_BASE = SEED_BASE + 60_000  # above every seed of the generated tasks and of the fresh eval sets
HARNESS = "intent_gold" + AGENT[len("intent"):]
WRITE_HINT = "Se l'utente chiede di scrivere un testo (un messaggio, una mail, un avviso), scrivilo per intero."
CHAT_HINT = ("Rispondi con le tue conoscenze generali. Se l'utente incolla un testo nel messaggio, quel testo lo hai: "
             "usalo. Se non sei sicuro di un fatto preciso (un nome, una data, un luogo, un numero), dillo invece di "
             "inventarlo. Se la domanda chiede un formato preciso (solo la lettera, sì o no), rispondi esattamente in "
             "quel formato.")
KINDS = ("doc",)
THINK_TOKENS = 2048  # in a trial, a third of thinking runs never closed within 4,096 tokens and burnt the budget

DOC_TASKS = ("un riassunto in poche righe", "una spiegazione in parole semplici di cosa dice",
             "l'elenco delle date, delle scadenze o degli importi che contiene",
             "una breve mail a un collega che spieghi di cosa si tratta",
             "sapere quale ente o ufficio se ne occupa e come contattarlo")
DOC = ("Una persona incolla in una chat il documento qui sotto e chiede {what}. Scrivi la richiesta come la "
       "scriverebbe lei all'assistente, in prima persona, in una o due frasi, in italiano, senza ripetere il "
       "documento.\n\nDOCUMENTO:\n{doc}")
THIRD_PERSON = re.compile(r"\b(la persona|l'utente|la richiesta|la tua richiesta)\b|^\s*(chiedi|chiede|c'è bisogno)\b",
                          re.IGNORECASE)


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in open(ROOT / "data" / "seeds" / f"{name}.jsonl", encoding="utf-8")]


def gate(text: str, low: int, high: int) -> str | None:
    """Why a text the teacher wrote as the user is rejected, or None."""
    n = len(text.split())
    if not low <= n <= high:
        return f"{n} words outside {low}-{high}"
    if not is_italian(text):
        return "not Italian"
    return None


def user_turn(kind: str, i: int, writer, docs: list[dict]):
    """(user text, source id, why rejected or None)."""
    k, n = i % len(docs), i // len(docs)  # the n-th pass over the excerpts asks each one something else
    d = docs[k]
    text = clean(ask(writer, DOC.format(what=DOC_TASKS[(k + n) % len(DOC_TASKS)], doc=d["text"])))
    why = gate(text, 3, 50) or ("third person" if THIRD_PERSON.search(text) else None)
    return f"{text}\n\n{d['text']}", d["id"], why


async def one(i: int, kind: str, teacher, writer, docs, log, kept, plain_share: float = 0.5) -> bool:
    seed = CHAT_BASE + i
    r = random.Random(seed)
    world, facts = build(seed, v=2)
    plain = r.random() < plain_share
    sys_msg, err = (None, None) if plain else system_text(writer, facts.now, r, v=2)
    user, source, why = (None, None, err) if err else user_turn(kind, i, writer, docs)
    row = {"i": i, "seed": seed, "kind": kind, "plain": plain, "source": source}
    if why:
        row.update(kept=False, reason=why, user=user)
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        return False
    category = "summarize"
    envs = [] if plain else r.sample(TRAIN_ENVS, r.randrange(1, 4))
    et = to_eval_task(task(f"chat_{kind}", category, envs, [turn({}, [], "", {"no_tools": True})]), f"chat-{seed}", [user])
    harness = None if plain else HARNESS
    if harness:
        et["context"] = sys_msg
    hint = f"{CHAT_HINT}\n{STYLE_HINT}\n{WRITE_HINT}"
    teacher_sys = f"{sys_msg}\n\n{hint}" if sys_msg else hint
    configs = [dict(thinking=False, temp=0.0), dict(thinking=True, temp=0.0, max_tokens=THINK_TOKENS)]
    answers, keep = [], None
    for a, cfg in enumerate(configs):
        agent = teacher.clone(seed=r.randrange(1 << 30), **cfg)
        traj = await run_task(et, agent, system=teacher_sys, world=world, harness=harness)
        final = traj["turns"][0]["final"] or ""
        msgs = traj["messages"][1:] if sys_msg is None else [{"role": "system", "content": sys_msg}] + traj["messages"][1:]
        problems = conversation_problems(msgs, category) + ([] if is_italian(final) else ["not Italian"])
        ok = score(et, traj)["success"] and not problems
        answers.append({"ok": ok, "problems": problems, "final": final[:300]})
        if ok:
            keep = (msgs, traj["tools"], {**cfg, "attempt": a})
            break
    row["attempts"] = answers
    if keep:
        msgs, tools, cfg = keep
        kept.write(json.dumps({"messages": msgs, "tools": tools,
                               "meta": {"seed": seed, "family": f"chat_{kind}", **cfg, "harness": harness, "plain": plain,
                                        "source": source, "users": [user], "world": 2,
                                        "decisions": [t["decision"] for t in et["turns"]]}},
                              ensure_ascii=False) + "\n")
    row.update(kept=bool(keep), reason=None if keep else "no attempt passed", user=user)
    log.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    log.flush()
    kept.flush()
    return bool(keep)


async def main_async(a) -> None:
    from ..harness.backend import MLXBackend

    teacher = MLXBackend(a.teacher, temp=0.0, max_tokens=1536)
    writer = teacher.clone(temp=0.8, thinking=False, max_tokens=400, cache=False)
    docs = load("documents")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.perf_counter(), 0
    with open(f"{out}.kept.jsonl", "a", encoding="utf-8") as kept, open(f"{out}.log.jsonl", "a", encoding="utf-8") as log:
        for i in range(a.start, a.start + a.n):
            n += await one(i, a.kind, teacher, writer, docs, log, kept, a.plain_share)
            print(f"[{i}] {a.kind} kept {n}/{i - a.start + 1} {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--kind", choices=KINDS, default="doc")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--plain-share", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
