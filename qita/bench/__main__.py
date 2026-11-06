"""Run the harness bench, or compare two runs paired by task.

  python -m qita.bench run --split tune --model M --adapter A --harness intent --out results/bench/X.jsonl
  python -m qita.bench compare results/bench/A.jsonl results/bench/B.jsonl
Model runs go through scripts/jailed.sh (kernel sandbox: writes only under runs/jail/NAME).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict
from math import comb
from pathlib import Path

from ..envs.base import World
from ..eval import harness_name
from ..harness.backend import MLXBackend
from ..harness.checks import score
from ..harness.runner import run_task
from .tasks import build


async def run(a) -> None:
    backend = MLXBackend(a.model, adapter=a.adapter, temp=0.0, max_tokens=1536)
    cases = build(a.split)
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c[0]["type"] in keep or c[0]["category"] in keep]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0, ok = time.perf_counter(), 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for k, (t, home, _) in enumerate(cases):
            traj = await run_task(t, backend, world=World(home=home), harness=a.harness)
            res = score(t, traj)
            ok += res["success"]
            fh.write(json.dumps({"id": t["id"], "type": t["type"], "level": t["category"], "success": res["success"],
                                 "failed": [c for c, v in res["checks"].items() if not v],
                                 "intents": [x["intent"] for x in traj["turns"]],
                                 "harness": a.harness,
                                 "calls": [(c["name"], c["arguments"], c["error"], c["result"][:300]) for c in traj["calls"]],
                                 "finals": [x["final"] for x in traj["turns"]], "flags": traj["flags"],
                                 "seconds": traj["seconds"]}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{k + 1}/{len(cases)}] {t['id']:<32} {'OK' if res['success'] else 'NO'}  "
                  f"{ok}/{k + 1}  {time.perf_counter() - t0:.0f}s", flush=True)


def mcnemar(g: int, lo: int) -> float:
    n = g + lo
    return min(1.0, 2 * sum(comb(n, j) for j in range(min(g, lo) + 1)) / 2 ** n) if n else 1.0


def compare(files: list[str]) -> None:
    runs = [{r["id"]: r for r in map(json.loads, Path(f).read_text(encoding="utf-8").splitlines())} for f in files]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    print("".ljust(26) + "".join(Path(f).stem[:16].rjust(18) for f in files))
    for key in ("level", "type"):
        groups = defaultdict(list)
        for i in ids:
            groups[runs[0][i][key]].append(i)
        for g in sorted(groups):
            print(f"  {g:<24}" + "".join(f"{sum(r[i]['success'] for i in groups[g])}/{len(groups[g])}".rjust(18) for r in runs))
    print("  " + "total".ljust(24) + "".join(f"{sum(r[i]['success'] for i in ids)}/{len(ids)}".rjust(18) for r in runs))
    if len(runs) == 2:
        g = [i for i in ids if runs[1][i]["success"] and not runs[0][i]["success"]]
        lo = [i for i in ids if runs[0][i]["success"] and not runs[1][i]["success"]]
        print(f"paired: second gains {len(g)}, loses {len(lo)}, exact McNemar p = {mcnemar(len(g), len(lo)):.3g}")
        print("  gains:", g)
        print("  losses:", lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--split", choices=["tune", "holdout", "holdout2"], default="tune")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--harness", type=harness_name, default="intent")
    p.add_argument("--only", default="", help="comma-separated types or levels")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("files", nargs="+")
    a = ap.parse_args()
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    if a.cmd == "run":
        asyncio.run(run(a))
    else:
        compare(a.files)


if __name__ == "__main__":
    main()
