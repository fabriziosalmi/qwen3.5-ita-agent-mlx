"""Run CLM (a separate Italian grammar checker, not published with this repository) on texts. Run it with CLM's own environment, from CLM's folder:

  cd <clm folder> && .venv/bin/python <repo>/scripts/clm_check.py in.jsonl out.jsonl

in.jsonl: {"id", "text"} per line; out.jsonl adds "violations": [{rule, layer, tokens, sentence, why}].
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())  # CLM's folder
from engine import Engine  # noqa: E402

engine = Engine()
with open(sys.argv[1], encoding="utf-8") as fin, open(sys.argv[2], "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        rep = engine.analyze(row["text"])
        row["violations"] = [{"rule": v.rule_id, "layer": v.layer, "tokens": v.tokens, "sentence": v.sentence,
                              "why": v.explanation} for s in rep.sentences for v in s.violations]
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
