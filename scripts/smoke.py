"""Smoke test of the stock model: chat template with tools, thinking switch, raw tool-call output."""

import json
import sys
import time

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "leggi_file",
            "description": "Legge un file di testo e ne restituisce il contenuto.",
            "parameters": {
                "type": "object",
                "properties": {"percorso": {"type": "string", "description": "Percorso del file"}},
                "required": ["percorso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "giorni_lavorativi",
            "description": "Conta i giorni lavorativi italiani tra due date (estremi inclusi).",
            "parameters": {
                "type": "object",
                "properties": {
                    "inizio": {"type": "string", "description": "Data ISO, es. 2026-11-02"},
                    "fine": {"type": "string", "description": "Data ISO"},
                },
                "required": ["inizio", "fine"],
            },
        },
    },
]

PROMPTS = [
    "Leggimi il file note/spesa.txt",
    "Quanti giorni lavorativi ci sono dal 2 al 13 novembre 2026?",
    "Ciao! Come ti chiami?",
]

model, tok = load(MODEL)
print("tool_parser:", getattr(tok, "tool_parser", None) and tok.tool_parser.__module__)
print("tool_call_start:", tok.tool_call_start if tok.has_tool_calling else None)

msgs = [{"role": "user", "content": PROMPTS[0]}]
for think in (False, True):
    text = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=think)
    print(f"\n===== template enable_thinking={think} (tail) =====")
    print(text[-400:])

greedy = make_sampler(temp=0.0)
for p in PROMPTS:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": p}], tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    t = time.perf_counter()
    out = generate(model, tok, prompt, max_tokens=300, sampler=greedy)
    dt = time.perf_counter() - t
    print(f"\n===== {p!r} ({dt:.2f}s) =====")
    print(out)
    if tok.has_tool_calling and tok.tool_call_start in out:
        body = out.split(tok.tool_call_start, 1)[1].split("</tool_call>", 1)[0]
        try:
            print("PARSED:", json.dumps(tok.tool_parser(body, TOOLS), ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            print("PARSE ERROR:", repr(e))
