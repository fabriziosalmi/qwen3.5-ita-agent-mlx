"""In-distribution probe: tasks from generated worlds no training run has seen, scored by their checks.

  uv run python scripts/fresh_eval.py phrase --writer MODEL --start 9000 --n 200 --out runs/fresh_tasks.jsonl
      the writer phrases the user turns and the system message (same gates as qita.worlds.sample);
      families in turn, faults by the same rule as the training data (--fault-p).
      --world 2 --families a,b,c: v2 worlds and phrasing, the named families in turn (held-out ones allowed here).
      store families (qita/worlds/families_store.py) run on the held-out store domains unless --store train.
  uv run python scripts/fresh_eval.py run --tasks runs/fresh_tasks.jsonl --model MODEL [--adapter DIR] --out runs/X.jsonl
      the model plays the assistant with the intent harness (it classifies every turn itself), or --harness.
  uv run python scripts/fresh_eval.py compare runs/A.jsonl runs/B.jsonl
Tasks the writer could not phrase are left out of the file, so every run sees the same tasks.
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import Counter
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qita.compare import wilson  # noqa: E402
from qita.eval import harness_name  # noqa: E402
from qita.harness.backend import MLXBackend  # noqa: E402
from qita.harness.checks import score  # noqa: E402
from qita.harness.runner import run_task  # noqa: E402
from qita.worlds.families import FAMILIES, inject_fault, to_eval_task  # noqa: E402
from qita.worlds.families_store import STORE  # noqa: E402
from qita.worlds.families_v2 import BY_NAME, add_envs  # noqa: E402
from qita.worlds.store import attach_store  # noqa: E402

STORE_BY_NAME = {f.__name__: f for f in STORE}
from qita.worlds.sample import SEED_BASE, phrase_users, system_text  # noqa: E402
from qita.worlds.world import build  # noqa: E402


def phrase(a) -> None:
    writer = MLXBackend(a.writer, temp=0.8, thinking=False, max_tokens=300)
    fams = [(BY_NAME | STORE_BY_NAME)[n] for n in a.families.split(",")] if a.families else FAMILIES
    with open(a.out, "a", encoding="utf-8") as fh:
        for i in range(a.start, a.start + a.n):
            seed = SEED_BASE + i
            r = random.Random(seed)
            world, facts = build(seed, v=a.world)
            fam = fams[i % len(fams)]
            store = a.store if fam.__name__ in STORE_BY_NAME else None
            if store:
                attach_store(world, facts, seed, held_out=store == "held")
            t = fam(facts, r)
            if a.world >= 2:
                add_envs(t, seed)
            if any(tt["plan"] for tt in t["turns"]) and random.Random(f"fault-{seed}").random() < a.fault_p:
                inject_fault(t)
            sys_msg, err = system_text(writer, facts.now, r, v=a.world)
            users, why = (None, None) if err else phrase_users(writer, t, r, v=a.world)
            if users is None:
                print(f"[{i}] {fam.__name__} skipped: {err or why}", flush=True)
                continue
            fh.write(json.dumps({"seed": seed, "world": a.world, "family": fam.__name__, "system": sys_msg, "store": store,
                                 "task": to_eval_task(t, f"fresh-{seed}", users)}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}] {fam.__name__} ok", flush=True)


async def run(a) -> None:
    backend = MLXBackend(a.model, adapter=a.adapter, temp=0.0, max_tokens=1536)
    rows = [json.loads(x) for x in Path(a.tasks).read_text(encoding="utf-8").splitlines() if x.strip()]
    with open(a.out, "w", encoding="utf-8") as fh:
        for k, row in enumerate(rows):
            world, facts = build(row["seed"], v=row.get("world", 1))
            if row.get("store"):
                attach_store(world, facts, row["seed"], held_out=row["store"] == "held")
            traj = await run_task(row["task"], backend, system=row["system"], world=world, harness=a.harness)
            res = score(row["task"], traj)
            fh.write(json.dumps({"id": row["task"]["id"], "family": row["family"], "category": row["task"]["category"],
                                 "fault": bool(row["task"].get("faults")), "success": res["success"],
                                 "failed": [c for c, v in res["checks"].items() if not v],
                                 "intents": [t["intent"] for t in traj["turns"]],
                                 "decisions": [t["decision"] for t in row["task"]["turns"]],
                                 "calls": [(c["name"], c["arguments"]) for c in traj["calls"]],
                                 "finals": [t["final"] for t in traj["turns"]], "flags": traj["flags"]},
                                ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{k + 1}/{len(rows)}] {row['family']:<20} {'OK' if res['success'] else 'NO'}", flush=True)


def compare(files: list[str]) -> None:
    runs = [{r["id"]: r for r in map(json.loads, Path(f).read_text(encoding="utf-8").splitlines())} for f in files]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    fams = sorted({runs[0][i]["family"] for i in ids})
    for f, run_ in zip(files, runs):
        k = sum(run_[i]["success"] for i in ids)
        lo, hi = wilson(k, len(ids))
        fault = [i for i in ids if run_[i]["fault"]]
        intent = [(x, y) for i in ids for x, y in zip(run_[i]["intents"], run_[i]["decisions"])]
        print(f"{Path(f).stem:<28} {k}/{len(ids)} {100 * k / len(ids):.1f}% [{lo:.1f}, {hi:.1f}]  "
              f"faulted {sum(run_[i]['success'] for i in fault)}/{len(fault)}  "
              f"intent {sum(x == y for x, y in intent)}/{len(intent)}  "
              f"loops {sum(run_[i]['flags']['loop'] for i in ids)}")
    print("family".ljust(22) + "".join(Path(f).stem[:14].rjust(16) for f in files))
    for fam in fams:
        sub = [i for i in ids if runs[0][i]["family"] == fam]
        print(fam.ljust(22) + "".join(f"{sum(r[i]['success'] for i in sub)}/{len(sub)}".rjust(16) for r in runs))
    if len(runs) == 2:
        g = sum(runs[1][i]["success"] and not runs[0][i]["success"] for i in ids)
        lo_ = sum(runs[0][i]["success"] and not runs[1][i]["success"] for i in ids)
        n = g + lo_
        p = min(1.0, 2 * sum(comb(n, j) for j in range(min(g, lo_) + 1)) / 2 ** n) if n else 1.0
        print(f"paired: second gains {g}, loses {lo_}, exact McNemar p = {p:.3g}")
        print("failed checks (second):", Counter(c.split(".")[-1].split("[")[0].split(":")[0]
                                                for i in ids for c in runs[1][i]["failed"]).most_common(8))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("phrase")
    p.add_argument("--writer", required=True)
    p.add_argument("--start", type=int, default=9000)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--fault-p", type=float, default=0.2)
    p.add_argument("--world", type=int, choices=[1, 2], default=1)
    p.add_argument("--families", default=None, help="comma-separated family names, in turn (default: the v1 families)")
    p.add_argument("--store", choices=["train", "held"], default="held",
                   help="domains of the record store for store families: held = the held-out domains (a probe)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--tasks", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--harness", type=harness_name, default="intent", help="e.g. agent (qita/harness/runner.py)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")
    a = ap.parse_args()
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    if a.cmd == "phrase":
        phrase(a)
    elif a.cmd == "run":
        asyncio.run(run(a))
    else:
        compare(a.files)


if __name__ == "__main__":
    main()
