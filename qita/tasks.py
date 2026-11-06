"""Loading and validation of eval task files (YAML, a list of tasks per file)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .envs import ENVS, ROLE, TEST_SPLITS

CATEGORIES = {
    "single",        # one call answers the request
    "chain",         # several calls, each depending on the previous result
    "parallel",      # several independent calls
    "no_tool",       # no tool should be called
    "clarify",       # a required detail is missing: ask, do not guess
    "recovery",      # a call fails and the model must react sensibly
    "safety",        # destructive or out-of-scope actions: confirm, or refuse
    "summarize",     # long tool output to be reported in Italian
    "choice",        # pick the right tool among similar ones
    "multi",         # 3+ turns where later turns depend on earlier ones (references, corrections, recall)
}
KEYS = {"id", "category", "envs", "turns", "checks", "system", "note"}


def load(path: str | Path, split: str) -> list[dict]:
    path = Path(path)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    tasks, ids = [], set()
    for f in files:
        for t in yaml.safe_load(f.read_text(encoding="utf-8")) or []:
            where = f"{f.name}:{t.get('id')}"
            if set(t) - KEYS:
                raise ValueError(f"{where}: unknown keys {set(t) - KEYS}")
            if t["id"] in ids:
                raise ValueError(f"{where}: duplicate id")
            ids.add(t["id"])
            if t["category"] not in CATEGORIES:
                raise ValueError(f"{where}: unknown category {t['category']}")
            for e in t["envs"]:
                if e not in ENVS:
                    raise ValueError(f"{where}: unknown env {e}")
                if ROLE[e] == "test" and split not in TEST_SPLITS:
                    raise ValueError(f"{where}: env {e} is test-only, not allowed in split {split}")
            t["turns"] = [{"user": x} if isinstance(x, str) else x for x in t["turns"]]
            t["split"] = split
            tasks.append(t)
    return tasks
