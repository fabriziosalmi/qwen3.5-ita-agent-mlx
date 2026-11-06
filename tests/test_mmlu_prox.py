import hashlib
import json
import os
from pathlib import Path

import pytest

from qita import mmlu_prox as M

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "mmlu_prox_golden.json").read_text(encoding="utf-8"))
VAL, TEST = M.load("validation"), M.load("test")
BY_ID = {d["question_id"]: d for d in TEST}


def test_frozen_split_is_unchanged():
    digest = lambda s: hashlib.sha256((M.DATA / f"{s}.jsonl").read_bytes()).hexdigest()[:16]  # noqa: E731
    assert (digest("validation"), digest("test")) == ("8c29d607c9e97d6b", "2b521fc43e8eb5fc")
    assert len(VAL) == 70 and len(TEST) == 588
    # test rows drop missing options (None), few-shot rows print them as "N/A", as lm-eval does
    assert sum(any(d[f"option_{i}"] is None for i in range(10)) for d in TEST) == 103
    assert not any(d[f"option_{i}"] == "N/A" for d in TEST for i in range(10))
    assert any(sum(BY_ID[r["doc_id"]][f"option_{i}"] is None for i in range(10)) for r in GOLDEN["raw"])


@pytest.mark.parametrize("rec", GOLDEN["raw"], ids=lambda r: str(r["doc_id"]))
def test_raw_prompt_is_lm_eval_prompt(rec):
    assert M.context(BY_ID[rec["doc_id"]], VAL) == rec["context"]
    assert rec["until"] == M.UNTIL and rec["max_gen_toks"] == M.MAX_NEW


@pytest.mark.parametrize("rec", GOLDEN["chat"], ids=lambda r: str(r["doc_id"]))
def test_chat_messages_are_lm_eval_messages(rec):
    system, user = M.messages(BY_ID[rec["doc_id"]], VAL)
    rendered = (f"<|im_start|>system\n{system['content']}<|im_end|>\n<|im_start|>user\n{user['content']}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert rendered == rec["context"]


@pytest.mark.skipif(not os.environ.get("QITA_TOKENIZER"), reason="needs a Qwen3.5 model folder in QITA_TOKENIZER")
def test_chat_prompt_with_the_model_tokenizer():
    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(Path(os.environ["QITA_TOKENIZER"]))
    for rec in GOLDEN["chat"]:
        assert M.context(BY_ID[rec["doc_id"]], VAL, tok) == rec["context"]


def test_answer_regex_and_stops():
    assert M.extract("bla. La risposta è (B).") == "B"
    assert M.extract("La risposta è C") == "C"
    assert M.extract("La risposta è (D). Poi La risposta è (E)") == "D"
    assert M.extract("The answer is (B)") is None
    assert M.cut("ragiono. La risposta è (A).\n\nDomanda:\naltro") == "ragiono. La risposta è (A).\n\n"
    assert M.cut("<think>x</think>La risposta è (A)", "</think>") == "La risposta è (A)"
