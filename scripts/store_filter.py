"""Rules on kept store conversations that the family checks did not have when a batch was generated.

  uv run python scripts/store_filter.py runs/st7a2/st7a2.kept.jsonl ... --out runs/st7/filtered
writes <out>.kept.jsonl and <out>.drop.jsonl (with the reason). Each row's task is rebuilt from its seed, as
qita.worlds.sample built it, so the rules read the world's own tool names:
  read_first   a broad or group delete asks for a confirmation only after reading the store in that turn
               (a generated group delete asked without reading first; the families now require the read)
  no_extra     an add sets only the name and the fields the user gave (a generated add invented an author)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qita.worlds.families_store import STORE  # noqa: E402
from qita.worlds.store import attach_store  # noqa: E402
from qita.worlds.world import build  # noqa: E402

BY_NAME = {f.__name__: f for f in STORE}


def rebuild(meta: dict) -> tuple[dict, dict]:
    seed = meta["seed"]
    world, facts = build(seed, v=meta.get("world", 2))
    store = attach_store(world, facts, seed)
    return BY_NAME[meta["family"]](facts, random.Random(seed)), store


def turns(messages: list[dict]) -> list[list[dict]]:
    """The assistant messages of each user turn."""
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append([])
        elif m["role"] == "assistant" and out:
            out[-1].append(m)
    return out


def calls(msgs: list[dict]) -> list[dict]:
    return [tc["function"] for m in msgs for tc in m.get("tool_calls") or []]


def problems(row: dict) -> list[str]:
    meta = row["meta"]
    if meta["family"] not in BY_NAME:
        return []
    t, s = rebuild(meta)
    why = []
    per_turn = turns(row["messages"])
    if meta["family"] in ("store_delete_all", "store_delete_group"):
        names = {c["name"] for c in calls(per_turn[0])}
        if not names & {s["tools"]["list"], s["tools"]["search"]}:
            why.append("read_first")
    if meta["family"] in ("store_add", "store_clarify_add"):
        allowed = {k for step in t["turns"][-1]["plan"] for c in step for k in c["args"]}
        for c in calls(per_turn[-1]):
            if c["name"] == s["tools"]["add"]:
                extra = {k for k, v in (c["arguments"] or {}).items() if v not in (None, "")} - allowed
                if extra:
                    why.append(f"no_extra: {sorted(extra)}")
    return why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kept", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for f in a.kept for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    kept, drop = [], []
    for r in rows:
        why = problems(r)
        (drop if why else kept).append({**r, "why": why} if why else r)
    Path(f"{a.out}.kept.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    Path(f"{a.out}.drop.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in drop), encoding="utf-8")
    print(f"{len(rows)} rows, {len(drop)} dropped: " + json.dumps(
        {k: sum(1 for r in drop if any(w.startswith(k) for w in r["why"])) for k in ("read_first", "no_extra")}))


if __name__ == "__main__":
    main()
