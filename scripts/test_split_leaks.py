"""Test split vs training data: near-copies of test requests in the kept data and the chat pools.

Reading eval/test is a read of the test split: do it once, with the question written down first. Prints
ids, seeds and scores; the detail lists training requests only, never test text. The training data it reads
(data/harness/*.kept.jsonl) is not published with this repository.

  uv run python scripts/test_split_leaks.py .
"""
import glob
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from tests.test_data_leaks import CHAT, LIMIT, LONG, MIN_WORDS, jaccard, words  # noqa: E402
from qita.worlds import families, families_v2  # noqa: E402

test = []
for p in sorted((ROOT / "eval" / "test").glob("*.yaml")):
    for t in yaml.safe_load(p.read_text(encoding="utf-8")):
        for u in t.get("turns", []):
            test.append((t["id"], words(u if isinstance(u, str) else u["user"])))
print("test requests", len(test), "tasks", len({i for i, _ in test}))


def rule(w, d):
    return min(len(w), len(d)) >= MIN_WORDS and jaccard(w, d) >= LIMIT and {x for x in d if len(x) >= LONG} <= w


convs = []
for p in sorted(glob.glob(str(ROOT / "data" / "harness" / "*.kept.jsonl"))):
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        m = json.loads(line).get("meta", {})
        convs.append((Path(p).name, m.get("seed"), m.get("family"), words((m.get("users") or [""])[0])))
print("kept conversations", len(convs), "chat-family", sum(f in CHAT for _, _, f, _ in convs))

hits = [(n, s, f, tid, round(jaccard(w, d), 2)) for n, s, f, w in convs if f in CHAT for tid, d in test if rule(w, d)]
print("RULE hits (chat families):", len(hits))
for h in hits:
    print("  ", h)

pools = []
for mod in (families, families_v2):
    for name in ("NO_TOOL_INTENTS", "DECLINE", "CHAT", "CHAT_MORE"):
        for item in getattr(mod, name, []) or []:
            s = " ".join(str(v) for v in item.values()) if isinstance(item, dict) else str(item)
            pools.append((f"{mod.__name__.split('.')[-1]}.{name}", words(s)))
ph = [(n, tid, round(jaccard(w, d), 2)) for n, w in pools for tid, d in test if rule(w, d)]
print("pool strings", len(pools), "RULE hits:", len(ph))
for h in ph:
    print("  ", h)

info = [(n, s, f, tid, round(jaccard(w, d), 2)) for n, s, f, w in convs
        for tid, d in test if min(len(w), len(d)) >= MIN_WORDS and jaccard(w, d) >= LIMIT]
print("INFO any family, Jaccard >= 0.5 only:", len(info))
for h in sorted(info, key=lambda x: -x[4])[:15]:
    print("  ", h)

# break-check: a test request planted as a chat conversation is found (text stays in memory)
tid, d = next((i, d) for i, d in test if len(d) >= MIN_WORDS)
assert any(rule(d | {"ciao"}, dd) for _, dd in test), "break-check failed"
print("break-check ok on", tid)

print("\nINFO detail: training request, test id, Jaccard, long test words missing from it")
raw = {}
for p in sorted(glob.glob(str(ROOT / "data" / "harness" / "*.kept.jsonl"))):
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        m = json.loads(line).get("meta", {})
        raw[(Path(p).name, m.get("seed"))] = (m.get("users") or [""])[0]
dt = dict()
for i, d in test:
    dt.setdefault(i, []).append(d)
seen = set()
for n, s, f, tid, j in sorted(info, key=lambda x: -x[4]):
    u = raw[(n, s)]
    w = words(u)
    miss = min(len({x for x in d if len(x) >= LONG} - w) for d in dt[tid] if jaccard(w, d) >= LIMIT)
    key = (u, tid)
    if key in seen:
        continue
    seen.add(key)
    print(f"  {tid:22} {j:.2f} missing_long={miss}  {f}: {u}")
