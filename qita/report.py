"""Summaries and trajectory dumps of eval result files.

  python -m qita.report runs/base_dev.jsonl            summary
  python -m qita.report runs/base_dev.jsonl --show ID  one trajectory, readable
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summary(rows: list[dict]) -> str:
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    lines = [f"{'category':<11} {'n':>3} {'success':>8} {'italian':>8} {'formed':>7} {'no_loop':>8}"]

    def line(name, rs):
        n = len(rs) or 1  # an empty selection prints zeros
        pct = lambda f: f"{100 * sum(1 for r in rs if f(r)) / n:7.1f}%"  # noqa: E731
        return (f"{name:<11} {n:>3} {pct(lambda r: r['success'])} {pct(lambda r: r['checks']['italian'])}"
                f"{pct(lambda r: r['checks']['well_formed'])} {pct(lambda r: r['checks']['no_loop'])}")

    for cat in sorted(by):
        lines.append(line(cat, by[cat]))
    lines.append(line("ALL", rows))
    return "\n".join(lines)


def show(row: dict) -> str:
    out = [f"== {row['id']} ({row['category']}) success={row['success']} flags={row['flags']}"]
    for m in row["messages"]:
        if m["role"] == "system":
            continue
        if m["role"] == "tool":
            out.append(f"  [tool] {m['content'][:600]}")
        elif m.get("tool_calls"):
            if m["content"]:
                out.append(f"  [assistant] {m['content']}")
            for c in m["tool_calls"]:
                out.append(f"  [call] {c['function']['name']}({json.dumps(c['function']['arguments'], ensure_ascii=False)})")
        else:
            out.append(f"  [{m['role']}] {m['content']}")
    out.append("  failed: " + ", ".join(k for k, v in row["checks"].items() if not v))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show", default="", help="task id, or 'fail' for every failed task")
    a = ap.parse_args()
    rows = load(a.file)
    if a.show:
        for r in rows:
            if r["id"] == a.show or (a.show == "fail" and not r["success"]):
                print(show(r), end="\n\n")
    else:
        print(summary(rows))


if __name__ == "__main__":
    main()
