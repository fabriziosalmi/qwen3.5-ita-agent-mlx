"""Record the prompts that lm-eval itself builds for mmlu_prox_lite_it, as a fixture for qita.mmlu_prox.

  uv run --no-project --with lm-eval==0.4.11 --with transformers python -I scripts/mmlu_prox_golden.py \
      TOKENIZER_DIR tests/fixtures/mmlu_prox_golden.json

Needs network (lm-eval downloads the dataset) and no model: a recording LM answers every request.
Two modes: raw text, and the chat template with thinking off.
"""

from __future__ import annotations

import json
import sys

import lm_eval
from lm_eval.api.model import LM
from transformers import AutoTokenizer

TASKS = ["mmlu_prox_lite_it_math", "mmlu_prox_lite_it_law", "mmlu_prox_lite_it_business", "mmlu_prox_lite_it_philosophy"]
LIMIT = 2


class Recorder(LM):
    def __init__(self, tok, chat: bool):
        super().__init__()
        self.tok, self.chat, self.seen = tok, chat, []

    @property
    def tokenizer_name(self) -> str:
        return "qwen3.5"

    def apply_chat_template(self, chat_history, add_generation_prompt=True) -> str:
        return self.tok.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
                                            continue_final_message=not add_generation_prompt, enable_thinking=False)

    def generate_until(self, requests):
        for r in requests:
            ctx, opts = r.args
            self.seen.append({"doc_id": r.doc["question_id"], "category": r.doc["category"], "context": ctx,
                              "until": opts["until"], "max_gen_toks": opts["max_gen_toks"]})
        return ["La risposta è (A)"] * len(requests)

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


def main() -> None:
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    out = {"lm_eval": lm_eval.__version__}
    for chat in (False, True):
        lm = Recorder(tok, chat)
        lm_eval.simple_evaluate(model=lm, tasks=TASKS, limit=LIMIT, apply_chat_template=chat,
                                fewshot_as_multiturn=False, random_seed=0)
        out["chat" if chat else "raw"] = sorted(lm.seen, key=lambda s: s["doc_id"])
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print({k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
