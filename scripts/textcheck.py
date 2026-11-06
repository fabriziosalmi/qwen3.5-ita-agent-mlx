"""Drop generated rows with an English word in an agent answer or a grammar error, after fixing "il 8 marzo"
into "l'8 marzo" (qita/worlds/textcheck.py).

  ../clm/.venv/bin/python -I scripts/textcheck.py runs/smkd/a/out.kept.jsonl --out runs/smkd/a/out.text

Runs with the environment of CLM, a separate Italian grammar checker (spaCy, morph-it) expected in ../clm and
not published with this repository; no language model is loaded.
Writes <out>.kept.jsonl and <out>.drop.jsonl (each dropped row with its reasons).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLM = ROOT.parent / "clm"
sys.path[:0] = [str(ROOT), str(CLM)]

from qita.worlds.textcheck import fix_wording, problems  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kept", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from engine import Engine

    italian = {ln.split("\t")[0].lower() for ln in gzip.open(CLM / "data" / "morph-it.txt.gz", "rt", encoding="utf-8")}
    english = {w.strip().lower() for w in open("/usr/share/dict/words", encoding="utf-8")}
    engine = Engine()

    def grammar(texts):
        return [[v.rule_id for s in rep.sentences for v in s.violations] for rep in engine.analyze_batch(texts)]

    n = dropped = fixed = 0
    with open(f"{a.out}.kept.jsonl", "w", encoding="utf-8") as kept, open(f"{a.out}.drop.jsonl", "w", encoding="utf-8") as drop:
        for path in a.kept:
            for line in open(path, encoding="utf-8"):
                row, n = json.loads(line), n + 1
                for m in row["messages"]:
                    if m["role"] == "assistant" and m.get("content") and fix_wording(m["content"]) != m["content"]:
                        m["content"], fixed = fix_wording(m["content"]), fixed + 1
                why = problems(row, italian, english, grammar)
                if why:
                    dropped += 1
                    drop.write(json.dumps({"seed": row["meta"]["seed"], "family": row["meta"]["family"], "why": why},
                                          ensure_ascii=False) + "\n")
                else:
                    kept.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{n} rows, {dropped} dropped, {fixed} answers with a wording fixed")


if __name__ == "__main__":
    main()
