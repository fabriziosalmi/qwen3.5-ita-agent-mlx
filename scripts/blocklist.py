"""Drop generated rows that repeat a benchmark item or an eval request, and duplicates (qita/worlds/blocklist.py).

  uv run --with pyarrow python scripts/blocklist.py runs/gen6/doc/text.kept.jsonl ... --out runs/gen6/final

Downloads the Italian files of the data/sources.yaml benchmarks at pinned revisions into runs/blocklist_cache/
(not tracked). Writes <out>.kept.jsonl and <out>.drop.jsonl (seed, family, reasons) and prints counts per
source, never item or request text: ITALIC and INCLUDE are reserved evals and stay unread.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qita.worlds.blocklist import Index, dedup_key, problems, strings, task_requests  # noqa: E402

CACHE = ROOT / "runs" / "blocklist_cache"
# Italian files only; revisions read from the Hugging Face API on 2026-11-04. Pinocchio's multimodal split
# (images with their questions) is left out.
BENCHMARKS = {
    "pinocchio": ("efederici/pinocchio", "83c6a063c7dd8872e52b4d10504ddb87ed24f0e5",
                  [f"text/{k}-00000-of-00001.parquet" for k in
                   ("cultura", "diritto", "generale", "lingua_straniera", "logica", "matematica_e_scienze")]),
    "mmlu_italian": ("sapienzanlp/mmlu_italian", "a325da8f58a6417ee0beee3b7f717569621b4518",
                     ["mmlu.dev.json", "mmlu.test.json", "mmlu.validation.json"]),
    "arc_italian": ("sapienzanlp/arc_italian", "1860d09fba59b9b2576335ab472346f63c1becc8",
                    [f"arc_{k}.{s}.json" for k in ("challenge", "easy") for s in ("test", "train", "validation")]),
    "hellaswag_italian": ("sapienzanlp/hellaswag_italian", "10038db5c00a2d40acdba29a2d9e0813f1a6869d",
                          [f"hellaswag.{s}{v}.json" for s in ("train", "validation") for v in ("", ".an", ".wh")]),
    "gsm8k_italian": ("sapienzanlp/gsm8k_italian", "fd897dc0d4c5521b2e7bd1a113a121ec7c1f0dbe",
                      ["gsm8k.test.json", "gsm8k.train.json"]),
    "truthful_qa_italian": ("sapienzanlp/truthful_qa_italian", "b4b40c1dfd28c48ed5f3094de9172e56a7b222ca",
                            ["truthful_qa.validation.json"]),
    "winogrande_italian": ("sapienzanlp/winogrande_italian", "8bf955a81f360106d3391a6a1d8d0be40c3faa97",
                           ["winogrande_xl.train.json", "winogrande_xl.validation.json"]),
    "boolq_italian": ("sapienzanlp/boolq_italian", "330916be5bbe35ad22a1cdbf3d87ad7f3b2bca0f",
                      [f"boolq.{c}{s}.json" for c in ("", "corrected.") for s in ("train", "validation")]),
    "global_mmlu": ("CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993",
                    ["it/dev-00000-of-00001.parquet", "it/test-00000-of-00001.parquet"]),
    "mmmlu": ("openai/MMMLU", "325a01dc3e173cac1578df94120499aaca2e2504", ["test/mmlu_IT-IT.csv"]),
    "xcopa": ("cambridgeltl/xcopa", "042f78955ba48e6404616762fa6e05e839c3907a",
              ["it/test-00000-of-00001.parquet", "it/validation-00000-of-00001.parquet"]),
    "belebele": ("facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", ["data/ita_Latn.jsonl"]),
    "italic": ("Crisp-Unimib/ITALIC", "65397d9f4cf1d51cb43147aa341083621fa92c7a", ["data/train-00000-of-00001.parquet"]),
    "include": ("CohereLabs/include-lite-44", "a027ccc923428c0c43c295a91efcc39a1e47fa60",
                ["Italian/test-00000-of-00001.parquet"]),
}


def fetch(key: str, repo: str, rev: str, path: str) -> Path:
    dest = CACHE / key / path
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}", dest)
    return dest


def items(path: Path) -> list:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".csv":
        return list(csv.DictReader(open(path, encoding="utf-8")))
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def eval_tasks() -> dict[str, list[dict]]:
    """Every frozen task set: dev, test, dogfood and each results/<set>/<set>_tasks.jsonl (fresh sets and the
    decision sets safe1 and storeheld, which are read again and again and must not be learned)."""
    out = {}
    for split in ("dev", "test", "dogfood"):
        out[split] = [t for p in sorted((ROOT / "eval" / split).glob("*.yaml"))
                      for t in yaml.safe_load(p.read_text(encoding="utf-8"))]
    for p in sorted(glob.glob(str(ROOT / "results" / "*" / "*_tasks.jsonl"))):
        out[Path(p).stem.removesuffix("_tasks")] = [json.loads(ln)["task"] for ln in open(p, encoding="utf-8")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kept", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--against", nargs="*", default=sorted(glob.glob(str(ROOT / "data" / "harness" / "*.kept.jsonl"))),
                    help="earlier training rows a new row must not duplicate (default: data/harness/*.kept.jsonl, "
                         "the training data, not published: without it only the new rows are deduplicated)")
    a = ap.parse_args()

    bench, evals = Index(), Index()
    for key, (repo, rev, paths) in BENCHMARKS.items():
        rows = [it for p in paths for it in items(fetch(key, repo, rev, p))]
        print(f"benchmark {key}: {len(rows)} items, {bench.add(key, (s for it in rows for s in strings(it)))} windows")
    prox = [json.loads(ln) for p in sorted((ROOT / "eval" / "mmlu_prox_lite_it").glob("*.jsonl")) for ln in open(p)]
    print(f"benchmark mmlu_prox: {len(prox)} items, {bench.add('mmlu_prox', (s for it in prox for s in strings(it)))} windows")
    for split, tasks in eval_tasks().items():
        print(f"eval {split}: {len(tasks)} tasks, {evals.add(split, (u for t in tasks for u in task_requests(t)))} windows")

    seeds = ROOT / "data" / "seeds" / "documents.jsonl"  # written by scripts/seeds.py; only doc rows use it
    docs = {d["id"]: d["text"] for d in map(json.loads, open(seeds, encoding="utf-8"))} if seeds.exists() else {}
    seen = {dedup_key(json.loads(ln)) for p in a.against for ln in open(p, encoding="utf-8")}
    print(f"earlier rows: {len(seen)} distinct")
    n, why_count = 0, Counter()
    with open(f"{a.out}.kept.jsonl", "w", encoding="utf-8") as kept, open(f"{a.out}.drop.jsonl", "w", encoding="utf-8") as drop:
        for path in a.kept:
            for line in open(path, encoding="utf-8"):
                row, n = json.loads(line), n + 1
                why = problems(row, bench, evals, docs.get(row["meta"].get("source"), ""))
                key = dedup_key(row)
                if key in seen:
                    why.append("duplicate")
                seen.add(key)
                why_count.update(why)
                if why:
                    drop.write(json.dumps({"seed": row["meta"]["seed"], "family": row["meta"]["family"], "why": why}) + "\n")
                else:
                    kept.write(line)
    print(f"{n} rows, {sum(1 for _ in open(f'{a.out}.drop.jsonl'))} dropped: {dict(why_count)}")


if __name__ == "__main__":
    main()
