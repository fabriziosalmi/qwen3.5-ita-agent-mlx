"""Gates on teacher-written text: what the user sentence and the system prompt must and must not contain."""

import datetime as dt
import random

from qita.envs.base import ROME
from qita.worlds.phrasing import clean, gate_system, gate_user, system_prompt_request, user_prompt


def test_user_gate_accepts_a_good_sentence():
    assert gate_user("Mi controlli se l'IVA della fattura 12/2027 è giusta?", ["12/2027"], ["275,00"]) == []


def test_user_gate_rejections():
    assert "missing «12/2027»" in gate_user("Mi controlli la fattura?", ["12/2027"], [])
    assert "leaks «275,00»" in gate_user("La fattura 12/2027 ha IVA 275,00, giusto?", ["12/2027"], ["275,00"])
    assert "not Italian" in gate_user("Please check the invoice 12/2027 for me", ["12/2027"], [])
    assert "sounds like an assistant" in gate_user("Ecco la fattura 12/2027 da controllare", ["12/2027"], [])
    assert "more than one line" in gate_user("Ciao.\nControlla la fattura 12/2027", ["12/2027"], [])
    assert "parentheses" in gate_user("Ok, ciao. (nessuno)", [], [])
    assert gate_user("Grazie.", [], []) == []
    assert gate_user("Cancella tutto", [], ["tutto"], ["cancellalo", "quello"])
    assert gate_user("Ok, cancellalo pure.", [], ["tutto"], ["cancellalo", "quello"]) == []


def test_clean_strips_quotes_and_thinking():
    assert clean("<think>\nragiono\n</think>\n\n«Leggimi la spesa»") == "Leggimi la spesa"


def test_system_gate_needs_the_facts():
    now = dt.datetime(2027, 3, 9, 9, 30, tzinfo=ROME)
    prompt, must = system_prompt_request(now, random.Random(0))
    assert "9 marzo 2027" in prompt and "09:30" in prompt
    good = "Sei un assistente nel terminale. Oggi è martedì 9 marzo 2027, sono le 9:30 (Europe/Rome). La home è /home/utente."
    assert gate_system(good, must) == []
    assert gate_system(good.replace("2027", "2026"), must)
    assert gate_system(good.replace("/home/utente", "la tua cartella"), must)


def test_user_prompt_carries_context():
    p = user_prompt({"azione": "rispondere con giorno e ora", "quando": "domani alle 17:00"}, ["17:00"], "diretto",
                    context="Utente: fissami il dentista\nAssistente: Per quando?")
    assert p.startswith("Conversazione finora:") and "«17:00»" in p and "domani alle 17:00" in p
    assert "(nessuno)" not in user_prompt({"azione": "ringraziare"}, [], "neutro")
    assert "una di queste espressioni: «scusa»" in user_prompt({"azione": "correggersi"}, [], "neutro", must_say_any=["scusa"])


def test_followup_turns_cannot_change_the_day_they_ask_for():
    from qita.worlds.families import mt_agenda_followup, mt_invoice_followup
    from qita.worlds.world import build
    _, facts = build(1009520, v=2)
    t3 = mt_invoice_followup(facts, random.Random(1))["turns"][2]
    gate = lambda text: gate_user(text, t3["must_say"], t3["must_not"], t3["must_say_any"])  # noqa: E731
    assert gate("Potresti programmare un promemoria in calendario per domani alle 9 per pagare la fattura?")
    assert gate("Mettimi un promemoria per lunedì prossimo alle 9, mezz'ora.")
    assert gate("Mettimi un promemoria il giorno prima della scadenza alle 9, per mezz'ora.") == []
    assert gate("Mettimi un promemoria il giorno prima della scadenza alle nove, per un'ora.")  # the check wants 30 minutes
    from qita.worlds.families import wrong_folder
    w = wrong_folder(facts, random.Random(1))["turns"][0]
    wgate = lambda text: gate_user(text, w["must_say"], w["must_not"], w["must_say_any"])  # noqa: E731
    path = w["must_say"][0]
    assert wgate(f"C'è un problema: l'app gira su {path} mentre il percorso è sbagliato.")  # never asks the port
    assert wgate(f"Su che porta gira l'app? Il file dovrebbe essere {path}.") == []
    for seed in range(1009500, 1009520):  # the last turn never restates a weekday or a month
        _, facts = build(seed, v=2)
        t3 = mt_agenda_followup(facts, random.Random(seed))["turns"][2]
        assert any(m in t3["must_not"] for m in ("novembre", "gennaio", "luglio", "marzo", "settembre", "aprile",
                                                 "maggio", "giugno", "agosto", "ottobre", "dicembre", "febbraio"))
