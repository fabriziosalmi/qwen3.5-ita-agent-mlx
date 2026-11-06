"""Blocklist and deduplication of generated rows (round 6), run by
scripts/blocklist.py:

  benchmark  a row repeats 8 consecutive words of a benchmark item of data/sources.yaml (any message), except
             words it pastes from its source document
  eval       a user request of the row repeats 8 consecutive words of a request of this project's evals
             (dev, test, dogfood, fresh sets); world files and tool outputs come from the same generator by
             design and are not compared
  duplicate  the row's user requests and final answer equal an earlier row's

Words are lowercased letters and digits; a window is hashed with Python's hash, so an index lives in one process.
Nothing here returns item or request text: a hit is reported by source name only.
"""

from __future__ import annotations

import re

import numpy as np

N = 8
_WORD = re.compile(r"[a-zà-ÿ0-9]+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def grams(text: str, n: int = N) -> np.ndarray:
    w = words(text)
    return np.array([hash(" ".join(w[k:k + n])) for k in range(len(w) - n + 1)], dtype=np.int64)


def strings(obj) -> list[str]:
    """Every string inside a dataset item (nested dicts and lists), without bytes (images)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in strings(v)]
    return []


def task_requests(task: dict) -> list[str]:
    """The user requests of an eval task (eval/*.yaml and fresh*_tasks.jsonl)."""
    return [u if isinstance(u, str) else u["user"] for u in task.get("turns", [])]


def row_requests(row: dict) -> list[str]:
    """The user requests of a generated row: the [RICHIESTA] part of a context block, or the whole turn; the
    harness's [CONTROLLO] turns quote the request and are left out."""
    out = []
    for m in row["messages"]:
        if m["role"] == "user" and m.get("content") and not m["content"].startswith("[CONTROLLO]"):
            out.append(m["content"].split("[RICHIESTA]")[-1].strip())
    return out


class Index:
    """Hashed 8-word windows of each source."""

    def __init__(self):
        self.sources: dict[str, np.ndarray] = {}

    def add(self, source: str, texts) -> int:
        parts = [grams(t) for t in texts]
        parts.append(self.sources.get(source, np.array([], dtype=np.int64)))
        self.sources[source] = np.unique(np.concatenate(parts))
        return len(self.sources[source])

    def hits(self, texts, unless: str = "") -> list[str]:
        """Sources sharing a window with texts, leaving out the windows of `unless`."""
        g = np.concatenate([grams(t) for t in texts] + [np.array([], dtype=np.int64)])
        g = np.setdiff1d(g, grams(unless)) if unless else g
        return [s for s, idx in self.sources.items() if len(g) and len(idx)
                and (idx[np.minimum(np.searchsorted(idx, g), len(idx) - 1)] == g).any()]  # idx is sorted


def problems(row: dict, benchmarks: Index, evals: Index, document: str = "") -> list[str]:
    """document: the source text a doc row pastes (data/seeds/documents.jsonl). Its windows do not count: exam
    questions quote the same laws and offices as Gazzetta acts (decided on the round-6 blocklist)."""
    said = [m.get("content") or "" for m in row["messages"]]
    return ([f"benchmark {s}" for s in benchmarks.hits(said, unless=document)]
            + [f"eval {s}" for s in evals.hits(row_requests(row))])


def dedup_key(row: dict) -> tuple:
    answer = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant" and m.get("content")), "")
    return tuple(" ".join(words(t)) for t in row_requests(row) + [answer])
