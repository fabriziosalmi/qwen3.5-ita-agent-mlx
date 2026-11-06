"""Side-by-side table of several eval result files.

  python -m qita.compare runs/base0.8B_dev_sys.jsonl runs/base2B_dev_sys.jsonl ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import tasks as T
from .envs import ROLE
from .report import load

ROOT = Path(__file__).resolve().parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def cell(rows: list[dict]) -> str:
    k, n = sum(r["success"] for r in rows), len(rows)
    return f"{k:>2}/{n:<3}" if n else "  -   "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    runs = [(Path(f).stem, load(f)) for f in a.files]
    if not all(rows for _, rows in runs):
        raise SystemExit("an empty result file: " + ", ".join(n for n, rows in runs if not rows))
    split = runs[0][1][0]["split"]
    envs = {t["id"]: t["envs"] for t in T.load(ROOT / "eval" / split, split)}
    held = {i for i, e in envs.items() if any(ROLE[x] == "test" for x in e)}
    cats = sorted({r["category"] for _, rows in runs for r in rows})
    w = max(len(n) for n, _ in runs) + 2
    print(f"{'run':<{w}} {'success':>8} {'95% CI':>13} " + " ".join(f"{c[:9]:>9}" for c in cats)
          + ("  held_env  train_env" if held else "") + "  ital  formed  loop  words")
    for name, rows in runs:
        k, n = sum(r["success"] for r in rows), len(rows)
        lo, hi = wilson(k, n)
        line = f"{name:<{w}} {100 * k / n:7.1f}% [{lo:4.1f},{hi:5.1f}] "
        line += " ".join(f"{cell([r for r in rows if r['category'] == c]):>9}" for c in cats)
        if held:
            line += f"  {cell([r for r in rows if r['id'] in held]):>8}  {cell([r for r in rows if r['id'] not in held]):>9}"
        ital = sum(r["checks"]["italian"] for r in rows)
        formed = sum(r["checks"]["well_formed"] for r in rows)
        loop = sum(not r["checks"]["no_loop"] for r in rows)
        words = sorted(len((t["final"] or "").split()) for r in rows for t in r["turns"])
        line += f"  {ital:>4}  {formed:>6}  {loop:>4}  {words[len(words) // 2] if words else 0:>5}"
        print(line)


if __name__ == "__main__":
    main()
