"""MMLU-ProX-Lite in Italian, run as lm-eval's mmlu_prox_lite_it task does, with mlx-lm batch generation.

  python -m qita.mmlu_prox --model <MLX model> --out runs/x.jsonl [--limit 20] [--raw]

Same prompt as lm-eval 0.4.13 (5 chain-of-thought shots of the same category, the first five of the
validation split), greedy, at most 2048 new tokens, cut at the task's stop strings, answer taken by
the task's regex. The chat template is applied with thinking off unless --raw. One JSON line per
question; a rerun with the same --out skips the questions already written.
tests/fixtures/mmlu_prox_golden.json holds prompts recorded from lm-eval itself.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "mmlu_prox_lite_it"
LETTERS = "ABCDEFGHIJ"
QUESTION, OPTIONS, COT_START, COT_SRC = "Domanda:", "Opzioni:", "Risposta: Ragioniamo passo dopo passo.", "A: Ragioniamo passo dopo passo."
DESCRIPTION = ('Ecco una domanda a scelta multipla su {} (con risposta). Si prega di ragionare passo dopo passo e '
               'terminare la risposta con "La risposta è (X)", dove X è la lettera dell\'opzione corretta.')
SUBJECTS = {"biology": "biologia", "business": "affari", "chemistry": "chimica", "computer science": "informatica",
            "economics": "economia", "engineering": "ingegneria", "health": "salute", "history": "storia",
            "law": "diritto", "math": "matematica", "other": "altro", "philosophy": "filosofia", "physics": "fisica",
            "psychology": "psicologia"}
UNTIL = ["</s>", "Q:", "Domanda:", "<|im_end|>"]
ANSWER = re.compile(r"La risposta è \(?([ABCDEFGHIJ])\)?")
SHOTS, MAX_NEW = 5, 2048


def load(split: str) -> list[dict]:
    return [json.loads(x) for x in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def example(doc: dict, with_answer: bool) -> str:
    s = f"{QUESTION}\n{doc['question']}\n{OPTIONS}\n"
    for i, letter in enumerate(LETTERS):
        if doc[f"option_{i}"] is not None:
            s += f"{letter}. {doc[f'option_{i}']}\n"
    if with_answer:
        return s + doc["cot_content"].replace(COT_SRC, COT_START) + "\n\n"
    return s + COT_START


def shots(doc: dict, validation: list[dict]) -> list[dict]:
    return [v for v in validation if v["category"] == doc["category"]][:SHOTS]


def messages(doc: dict, validation: list[dict]) -> list[dict]:
    user = "".join(example(v, True) for v in shots(doc, validation)) + example(doc, False)
    return [{"role": "system", "content": DESCRIPTION.format(SUBJECTS[doc["category"]])},
            {"role": "user", "content": user}]


def context(doc: dict, validation: list[dict], tokenizer=None) -> str:
    """The text the model continues: raw text without a tokenizer, else the chat template with thinking off."""
    system, user = messages(doc, validation)
    if tokenizer is None:
        return system["content"] + "\n" + user["content"]
    return tokenizer.apply_chat_template([system, user], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def cut(text: str, think_end: str | None = None) -> str:
    """lm-eval's stop strings, then whatever follows an end-of-thinking tag."""
    text = text[:min([i for i in (text.find(u) for u in UNTIL) if i >= 0] + [len(text)])]
    if think_end and (i := text.find(think_end)) != -1:
        text = text[i + len(think_end):]
    return text


def extract(text: str) -> str | None:
    m = ANSWER.search(text)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--raw", action="store_true", help="no chat template, as lm-eval without --apply_chat_template")
    ap.add_argument("--cache-gb", type=float, default=2.0)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import batch_generate, load as load_model
    from mlx_lm.sample_utils import make_sampler

    mx.set_cache_limit(int(a.cache_gb * 2**30))
    model, tok = load_model(a.model)
    think_end = tok.think_end if getattr(tok, "has_thinking", False) else None
    validation, test = load("validation"), load("test")
    if a.limit:
        test = test[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["question_id"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    todo = [d for d in test if d["question_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    sampler = make_sampler(0.0)
    for start in range(0, len(todo), a.batch):
        chunk = todo[start:start + a.batch]
        prompts = [tok.encode(context(d, validation, None if a.raw else tok), add_special_tokens=a.raw) for d in chunk]
        t0 = time.perf_counter()
        r = batch_generate(model, tok, prompts=prompts, max_tokens=MAX_NEW, sampler=sampler,
                           return_token_ids=True, prefill_batch_size=a.batch, completion_batch_size=a.batch)
        dt = time.perf_counter() - t0
        with out.open("a", encoding="utf-8") as fh:
            for d, p, text, ids in zip(chunk, prompts, r.texts, r.token_ids):
                resp = cut(text, think_end)
                pred = extract(resp)
                fh.write(json.dumps({"question_id": d["question_id"], "category": d["category"], "answer": d["answer"],
                                     "pred": pred, "correct": pred == d["answer"], "prompt_tokens": len(p),
                                     "new_tokens": len(ids), "hit_cap": len(ids) >= MAX_NEW, "response": resp,
                                     "model": a.model, "chat": not a.raw}, ensure_ascii=False) + "\n")
        print(f"{start + len(chunk)}/{len(todo)} batch {dt:.1f}s new tokens {sum(len(i) for i in r.token_ids)} "
              f"peak {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
