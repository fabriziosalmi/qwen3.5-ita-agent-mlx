"""Italian exam questions (ITALIC, INCLUDE Italian) scored by the letter the model finds most likely.

  python -m qita.mcq run --bench italic --model DIR --out runs/x.jsonl [--limit 20]
  python -m qita.mcq compare runs/A.jsonl runs/B.jsonl

Each question is one user turn (question, lettered options, an instruction to answer with the letter
only), with the model's chat template and thinking off. The answer is the option letter whose first token
has the highest logit at the start of the reply: no text is generated, so a model that would wrap its
answer in prose is not penalized for the format. Files are pinned revisions (data/sources.yaml); they never
enter seeds, prompts or training data. Rows hold the index, category, gold and predicted letter only, and
the script prints counts, never question text.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "blocklist_cache"
BENCH = {  # same pins as scripts/blocklist.py
    "italic": ("Crisp-Unimib/ITALIC", "65397d9f4cf1d51cb43147aa341083621fa92c7a",
               "data/train-00000-of-00001.parquet", "options", "category"),
    "include": ("CohereLabs/include-lite-44", "a027ccc923428c0c43c295a91efcc39a1e47fa60",
                "Italian/test-00000-of-00001.parquet", "choices", "subject"),
}
LETTERS = "ABCDE"
INSTRUCTION = "Rispondi solo con la lettera dell'opzione giusta."


def items(bench: str) -> list[dict]:
    """The bench as JSON lines; built once from the parquet file (needs pyarrow, e.g. `uv run --with pyarrow`),
    so machines without pyarrow read the JSON lines copied to them."""
    repo, rev, path, options, category = BENCH[bench]
    cached = CACHE / bench / "items.jsonl"
    if not cached.exists():
        import pyarrow.parquet as pq

        dest = CACHE / bench / path
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}", dest)
        rows = [{"question": r["question"], "options": r[options], "answer": r["answer"], "category": r[category]}
                for r in pq.read_table(dest).to_pylist()]
        cached.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return [json.loads(x) for x in cached.read_text(encoding="utf-8").splitlines()]


def prompt(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    return f"{item['question']}\n\n{opts}\n\n{INSTRUCTION}"


def pick(logits: list[float], letter_ids: list[int], n_options: int) -> str:
    """The letter, among the item's options, whose token has the highest logit."""
    best = max(range(n_options), key=lambda i: logits[letter_ids[i]])
    return LETTERS[best]


def sign_p(plus: int, minus: int) -> float:
    n, k = plus + minus, min(plus, minus)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def run(a) -> None:
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(a.model)
    letter_ids = []
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"letter {L} is {len(ids)} tokens"
        letter_ids.append(ids[0])
    out = Path(a.out)
    done = {json.loads(x)["i"] for x in out.read_text().splitlines()} if out.exists() else set()
    data = items(a.bench)[: a.limit or None]
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, it in enumerate(data):
            if i in done:
                continue
            text = tok.apply_chat_template([{"role": "user", "content": prompt(it)}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False)
            logits = model(mx.array([ids]))[0, -1].astype(mx.float32).tolist()
            pred, gold = pick(logits, letter_ids, len(it["options"])), LETTERS[it["answer"]]
            f.write(json.dumps({"i": i, "category": it["category"], "gold": gold, "pred": pred,
                                "right": pred == gold}) + "\n")
            if i % 500 == 0:
                f.flush()
                print(f"[{i}/{len(data)}] {time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    print(f"{a.bench}: {sum(r['right'] for r in rows)}/{len(rows)} in {time.time() - t0:.0f}s")


def compare(a) -> None:
    A, B = ({json.loads(x)["i"]: json.loads(x) for x in Path(p).read_text().splitlines()} for p in a.files)
    common = sorted(set(A) & set(B))
    plus = sum(B[i]["right"] and not A[i]["right"] for i in common)
    minus = sum(A[i]["right"] and not B[i]["right"] for i in common)
    print(f"{len(common)} common: A {sum(A[i]['right'] for i in common)}, B {sum(B[i]['right'] for i in common)}, "
          f"B against A +{plus}/-{minus}, sign test p = {sign_p(plus, minus):.3g}")
    by = Counter()
    for i in common:
        by[A[i]["category"], "A"] += A[i]["right"]
        by[B[i]["category"], "B"] += B[i]["right"]
        by[A[i]["category"], "n"] += 1
    for c in sorted({k[0] for k in by}):
        print(f"  {c}: A {by[c, 'A']}, B {by[c, 'B']} of {by[c, 'n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bench", choices=sorted(BENCH), required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("files", nargs=2)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
