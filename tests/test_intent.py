"""Intent harness: closed label set, deterministic parsing, the header the model sees, fair use in the loop."""

import asyncio
import logging

from qita.harness.checks import score
from qita.harness.intent import INTENTS, classify_prompt, parse_intent, wrap
from qita.harness.runner import run_task
from tests.test_harness import Scripted, call

logging.getLogger("mcp").setLevel(logging.CRITICAL)


def test_parse_intent():
    assert parse_intent("INTENTO: strumenti") == "strumenti"
    assert parse_intent("<think>\nhmm INTENTO: chiedi\n</think>\n\nINTENTO: conferma") == "conferma"
    assert parse_intent("intento: Rispondi") == "rispondi"
    assert parse_intent("INTENTO: boh") == "generico"
    assert parse_intent("non lo so") == "generico"


def test_header_has_context_intent_rule_and_request():
    tools = [{"type": "function", "function": {"name": "read_text_file"}}]
    w = wrap("Leggimi la spesa", "strumenti", "Oggi è mercoledì.", tools)
    assert w.index("[CONTESTO]") < w.index("[INTENTO]") < w.index("[RISPOSTA ATTESA]") < w.index("[RICHIESTA]")
    assert "read_text_file" in w and INTENTS["strumenti"][1] in w and w.endswith("Leggimi la spesa")


def test_classify_prompt_lists_every_label_and_previous_turns():
    p = classify_prompt("spostalo alle 17", [("Utente", "cosa ho domani?"), ("Assistente", "Hai il dentista.")])
    assert all(f"- {k}:" in p for k in INTENTS) and "Hai il dentista." in p and p.rstrip().endswith("<etichetta>")


def test_loop_uses_the_models_own_intent_never_metadata():
    task = {"id": "h", "category": "single", "envs": ["files"],
            "turns": [{"user": "Cosa c'è nella lista della spesa?"}],
            "checks": {"answer": {"contains": ["latte"]}}}
    outs = ["INTENTO: strumenti", call("read_text_file", path="spesa.txt"), "Ci sono latte e uova."]
    traj = asyncio.run(run_task(task, Scripted(outs), harness="intent"))
    assert traj["turns"][0]["intent"] == "strumenti"
    user = [m for m in traj["messages"] if m["role"] == "user"][0]["content"]
    assert user.startswith("[CONTESTO]") and "[INTENTO]\nstrumenti" in user
    assert score(task, traj)["success"]


def test_labels_match_the_gold_decisions():
    from qita.harness.decision import DECISIONS
    assert set(INTENTS) == set(DECISIONS)
