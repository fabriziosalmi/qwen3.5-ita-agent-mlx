"""Seeds for the chats without tools, from the sources in data/sources.yaml, at pinned revisions.

  uv run --with pyarrow python scripts/seeds.py
      downloads the pinned files into runs/seeds_cache/ (not tracked) and writes
      data/seeds/prompts.jsonl    questions written by people: oasst2 Italian conversation starters, Aya Italian
      data/seeds/documents.jsonl  excerpts of documents written by people: ItaIst sections, Gazzetta Ufficiale acts
Every row carries its source, license and revision. Only the question of a seed prompt is kept: its answer in
the training data is written by the teacher. Gazzetta's "Parte Seconda" (notices of private parties, with their
names) and the long Corte Costituzionale rulings are left out.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "seeds_cache"
OUT = ROOT / "data" / "seeds"
FILES = {
    "oasst2": ("OpenAssistant/oasst2", "179dd21fc55192153d94adb0e0ce8f69e222bf75",
               "data/train-00000-of-00001-88ba0162028a73fc.parquet", "apache-2.0"),
    "aya": ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
            "data/train-00000-of-00001.parquet", "apache-2.0"),
    "itaist": ("VerbACxSS/ItaIst", "b920ca209314fcb7382c39e367bb395f36d29b72", "corpus.tsv", "mit"),
    "gazzetta": ("mii-llm/gazzetta-ufficiale", "ab335f2f0bdd7bd144f51092e0c603e10068365d",
                 "data/train-00000-of-00014.parquet", "mit"),
}
EXCERPT = (800, 4000)  # characters of a document excerpt the teacher can read in one prompt
GAZZETTA_TYPES = ("Serie Generale", "Concorsi", "Contratti", "Regioni")
GAZZETTA_N, ITAIST_PER_DOC = 300, 2


def fetch(key: str) -> Path:
    repo, rev, path, _ = FILES[key]
    dest = CACHE / key / Path(path).name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}", dest)
    return dest


def row(key: str, rid: str, text: str, **extra) -> dict:
    repo, rev, _, lic = FILES[key]
    return {"id": f"{key}-{rid}", "source": repo, "revision": rev, "license": lic, "text": text.strip(), **extra}


def prompts() -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    o = pq.read_table(fetch("oasst2"), columns=["message_id", "lang", "role", "parent_id", "deleted", "text"]).to_pylist()
    out += [row("oasst2", m["message_id"], m["text"]) for m in o
            if m["lang"] == "it" and m["role"] == "prompter" and m["parent_id"] is None and not m["deleted"]]
    a = pq.read_table(fetch("aya")).to_pylist()
    out += [row("aya", str(i), r["inputs"], annotation=r["annotation_type"]) for i, r in enumerate(a)
            if r["language"] == "Italian"]
    return out


def documents() -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    csv.field_size_limit(10**9)
    sections = defaultdict(list)
    for r in csv.DictReader(open(fetch("itaist"), encoding="utf-8"), delimiter="\t"):
        if EXCERPT[0] <= len(r["content"]) <= EXCERPT[1]:
            sections[r["document_id"]].append(r)
    for doc, rs in sorted(sections.items()):
        for r in sorted(rs, key=lambda x: int(x["progress"]))[:ITAIST_PER_DOC]:
            out.append(row("itaist", f"{doc}-{r['progress']}", r["content"], title=r["title"], kind=r["document_type"]))
    g = pq.read_table(fetch("gazzetta"), columns=["type", "title", "subtitle", "url", "text"]).to_pylist()
    pool = [(i, r) for i, r in enumerate(g)
            if r["type"] in GAZZETTA_TYPES and EXCERPT[0] <= len(r["text"] or "") <= EXCERPT[1]]
    for i, r in random.Random(6).sample(pool, GAZZETTA_N):
        out.append(row("gazzetta", str(i), r["text"], title=" ".join(x for x in (r["title"], r["subtitle"]) if x),
                       kind=r["type"]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("prompts", prompts()), ("documents", documents())):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = defaultdict(int)
        for r in rows:
            by[r["source"]] += 1
        print(name, len(rows), dict(by))


if __name__ == "__main__":
    main()
