"""Step A of the intent harness alone: the model's decision for every dev turn against the gold one.

  uv run python scripts/classify_eval.py MODEL [ADAPTER] > runs/classify_<tag>.json
Previous turns are shown with the oracle answers (eval-only text), so each turn is classified in context.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qita import tasks as T  # noqa: E402
from qita.harness.backend import MLXBackend  # noqa: E402
from qita.harness.decision import decision  # noqa: E402
from qita.harness.intent import classify  # noqa: E402
from tests.oracles import DEV  # noqa: E402

model = sys.argv[1]
backend = MLXBackend(model, adapter=sys.argv[2] if len(sys.argv) > 2 else None, max_tokens=64)
rows, conf = [], Counter()
for t in T.load("eval/dev", "dev"):
    finals = [s for s in DEV[t["id"]] if isinstance(s, str)]
    prev = []
    for i, tu in enumerate(t["turns"]):
        gold = decision(t, i)
        pred, raw = classify(backend, tu["user"], prev)
        conf[(gold, pred)] += 1
        rows.append({"id": t["id"], "turn": i, "category": t["category"], "gold": gold, "pred": pred, "raw": raw[:120]})
        prev += [("Utente", tu["user"]), ("Assistente", finals[i] if i < len(finals) else "")]
acc = sum(r["gold"] == r["pred"] for r in rows) / len(rows)
per = {g: f"{sum(1 for r in rows if r['gold'] == g and r['pred'] == g)}/{sum(1 for r in rows if r['gold'] == g)}"
       for g in ("rispondi", "strumenti", "chiedi", "conferma")}
print(json.dumps({"model": model, "turns": len(rows), "accuracy": round(acc, 3), "per_gold": per,
                  "confusion": {f"{g}->{p}": n for (g, p), n in sorted(conf.items())}, "rows": rows}, ensure_ascii=False))
