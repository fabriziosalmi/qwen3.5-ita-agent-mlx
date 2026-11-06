"""No dev task reaches training: chat requests in the kept data are not near-copies of dev requests.

Tool tasks live in generated worlds with their own files and dates; requests answered without tools
carry the whole task in their wording, so those are compared with the dev requests word by word.
"""

import glob
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CHAT = {"chat_general", "chat_more", "decline_more", "no_tool_chat", "decline"}  # one request, answered without tools
MIN_WORDS = 4  # "Grazie, ok." is in every assistant's data, not a task
LIMIT = 0.5  # word-set Jaccard; the three copies found on 2026-10-29 scored 0.50-0.75
LONG = 6  # a copy also has every long word of the dev request ("ricevuta"); the same skill on another word does not


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-zàèéìòù0-9]+", s.casefold()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def dev_requests() -> list[tuple[str, set[str]]]:
    out = []
    for p in sorted((ROOT / "eval" / "dev").glob("*.yaml")):
        for t in yaml.safe_load(p.read_text(encoding="utf-8")):
            for u in t.get("turns", []):
                out.append((t["id"], words(u if isinstance(u, str) else u["user"])))
    return out


def near_copies(files: list[str], limit: float = LIMIT) -> list[tuple]:
    dev = dev_requests()
    hits = []
    for p in files:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            meta = json.loads(line).get("meta", {})
            if meta.get("family") not in CHAT:
                continue
            u = (meta.get("users") or [""])[0]
            w = words(u)
            hits += [(Path(p).name, meta.get("seed"), tid, u) for tid, d in dev
                     if min(len(w), len(d)) >= MIN_WORDS and jaccard(w, d) >= limit and {x for x in d if len(x) >= LONG} <= w]
    return hits


def test_kept_chat_requests_are_not_dev_requests():
    hits = near_copies(sorted(glob.glob(str(ROOT / "data" / "harness" / "*.kept.jsonl"))))
    assert not hits, hits


def test_break_a_dev_request_in_the_data_is_found(tmp_path):
    """Break-test: a kept chat conversation whose request is a dev request, reworded a little, is caught."""
    f = tmp_path / "x.kept.jsonl"
    f.write_text(json.dumps({"meta": {"family": "chat_general", "seed": 1, "users": ["Ciao, dimmi pure cosa sai fare."]}}) + "\n")
    assert near_copies([str(f)])
