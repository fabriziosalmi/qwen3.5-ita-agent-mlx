"""Blocklist and deduplication of generated rows (qita/worlds/blocklist.py)."""

from qita.worlds import blocklist as bl

ITEM = "Quale dei seguenti organi esercita la funzione legislativa nella Repubblica italiana?"
REQUEST = "Segna in calendario la visita dal dentista per giovedì alle dieci e mezza."


def _row(request, answer, tool="", controllo=False):
    msgs = [{"role": "system", "content": "Oggi è lunedì."},
            {"role": "user", "content": f"[CONTESTO]\nStrumenti disponibili: fetch.\n[RICHIESTA]\n{request}"}]
    if tool:
        msgs.append({"role": "tool", "content": tool})
    if controllo:
        msgs.append({"role": "user", "content": f"[CONTROLLO] L'utente ha chiesto: «{request}»."})
    msgs.append({"role": "assistant", "content": answer})
    return {"messages": msgs, "meta": {"seed": 1, "family": "cond_free"}}


def _indexes():
    bench, evals = bl.Index(), bl.Index()
    bench.add("italic", bl.strings({"question": ITEM, "choices": ["Il Parlamento", "Il Governo"], "image": b"\x00"}))
    evals.add("test", bl.task_requests({"turns": [{"user": REQUEST, "checks": {}}]}))
    return bench, evals


def test_windows_are_eight_words_and_strings_skip_bytes():
    assert len(bl.grams("uno due tre quattro cinque sei sette otto")) == 1
    assert len(bl.grams("uno due tre quattro cinque sei sette")) == 0
    assert bl.strings({"a": "x", "b": [{"c": "y"}, b"z"], "d": 3}) == ["x", "y"]


def test_a_benchmark_span_anywhere_in_a_row_is_a_hit():
    bench, evals = _indexes()
    said = "Ti chiedo: quale dei seguenti organi esercita la funzione legislativa nella repubblica italiana?"
    assert bl.problems(_row("Rispondi a questa domanda.", said), bench, evals) == ["benchmark italic"]
    assert bl.problems(_row("Rispondi.", "Il Parlamento.", tool=ITEM), bench, evals) == ["benchmark italic"]
    assert bl.problems(_row("Rispondi.", "Esercita la funzione il Parlamento."), bench, evals) == []


def test_an_eval_request_counts_only_in_a_user_request():
    bench, evals = _indexes()
    assert bl.problems(_row(REQUEST, "Fatto."), bench, evals) == ["eval test"]
    assert bl.problems(_row("Segna la visita.", "Fatto.", tool=REQUEST), bench, evals) == []
    assert bl.row_requests(_row("Segna la visita.", "Fatto.", controllo=True)) == ["Segna la visita."]


def test_a_duplicate_has_the_same_requests_and_answer():
    a = bl.dedup_key(_row("Segna la visita.", "Ho segnato la visita."))
    assert a == bl.dedup_key(_row("segna la visita", "Ho segnato  la visita!"))
    assert a != bl.dedup_key(_row("Segna la visita.", "La visita è segnata."))


def test_words_pasted_from_the_source_document_do_not_count():
    # round-6 blocklist: Gazzetta acts and exam questions name the same laws
    bench, evals = _indexes()
    row = _row(f"Riassumi: {ITEM} Lo dice la Costituzione.", "Il Parlamento.")
    assert bl.problems(row, bench, evals) == ["benchmark italic"]
    assert bl.problems(row, bench, evals, document=f"{ITEM} Lo dice la Costituzione.") == []
    row = _row(f"Riassumi: {ITEM}", f"Il testo chiede: {ITEM.lower()}")
    assert bl.problems(row, bench, evals, document="Un altro testo, sulle elezioni comunali.") == ["benchmark italic"]


def test_eval_sets_include_the_decision_sets():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("blocklist_script", Path(__file__).parents[1] / "scripts" / "blocklist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sets = mod.eval_tasks()
    assert {"dev", "test", "fresh7", "safe1", "storeheld"} <= set(sets)
    assert len(sets["storeheld"]) == 107 and len(sets["safe1"]) == 96
