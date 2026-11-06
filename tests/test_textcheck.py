"""Post-generation text checks (qita/worlds/textcheck.py) and the days an answer names (qita.harness.checks)."""

from qita.harness.checks import check_answer, named_days
from qita.worlds import textcheck as tc

IT = {"creato", "con", "successo", "per", "il", "luglio", "alle", "ho", "letto", "usa", "ultimo", "è", "andato", "bene"}
EN = {"event", "created", "log", "python", "with"}


def _row(answer, family="cond_free", user="Fissa la telefonata alla nonna il 3 luglio."):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
            "meta": {"family": family}}


def test_an_english_word_in_an_agent_answer_is_found():
    assert tc.english_words("Event creato con successo per il 3 luglio.", set(), IT, EN) == ["event"]


def test_words_from_the_user_or_tools_names_and_loanwords_are_not_english_mistakes():
    assert tc.english_words("Ho letto il log.", set(), IT, EN) == []  # a loanword
    assert tc.english_words("Ho letto event.", {"event"}, IT, EN) == []  # the user or a tool wrote it
    assert tc.english_words("Ho letto il file con Python.", set(), IT, EN) == []  # a name mid-sentence
    assert tc.english_words("Ho letto `created` e «with».", set(), IT, EN) == []  # code and quotes


def test_problems_checks_english_in_agent_rows_only_and_keeps_the_measured_grammar_rules():
    none = lambda texts: [[] for _ in texts]  # noqa: E731
    assert tc.problems(_row("Event creato con successo."), IT, EN, none) == ["English «event»"]
    assert tc.problems(_row("Event creato con successo.", family="chat_doc"), IT, EN, none) == []
    assert tc.problems(_row("Event creato.", user="Crea un event per il 3 luglio."), IT, EN, none) == []
    fired = lambda texts: [["L1.14", "L1.31", "L0.07"] for _ in texts]  # noqa: E731
    assert tc.problems(_row("Gli backup sono andati bene."), IT, EN, fired) == [
        "grammar L1.14: 'gli' before a simple consonant"]


def test_named_days_reads_lists_ranges_and_numeric_dates():
    assert named_days("i backup del 7, 8 e 9 maggio") == {(5, 7), (5, 8), (5, 9)}
    assert named_days("dal 27 al 31 ottobre") == named_days("(27-31 ottobre)") == {(10, d) for d in range(27, 32)}
    assert named_days("il 2027-01-02, ieri 15/12, fattura 198/2027 alle 02:17") == {(1, 2), (12, 15)}


def test_an_answer_may_name_only_the_days_it_was_asked_about():
    spec = {"any": [["backup"]], "days": ["2027-05-12", "2027-05-13"]}
    assert check_answer(spec, "L'ultimo backup del 13 maggio è andato bene, quello del 12 maggio anche.")
    assert not check_answer(spec, "L'ultimo backup del 13 maggio è andato bene; quelli del 7, 8 e 9 maggio no.")


def test_a_missing_elision_before_a_month_is_fixed_and_a_copied_date_is_not():
    assert tc.fix_wording("Il 8 marzo e dal 2 al 8 giugno, del 11 aprile.") == "L'8 marzo e dal 2 all'8 giugno, dell'11 aprile."
    assert tc.fix_wording("tra il 23 marzo e il 11 aprile") == "tra il 23 marzo e l'11 aprile"
    assert tc.fix_wording("FATTURA N. 226/2026 del 11/12/2026, il 18 marzo") == "FATTURA N. 226/2026 del 11/12/2026, il 18 marzo"


def test_segnalato_for_a_calendar_entry_is_fixed_and_other_uses_are_not():
    assert tc.fix_wording("Ho segnalato la visita per martedì.") == "Ho segnato la visita per martedì."
    assert tc.fix_wording("Segnalato in calendario.") == "Segnato in calendario."
    assert tc.fix_wording("Ho segnalato al tecnico il guasto.") == "Ho segnalato al tecnico il guasto."


LOG = ("2028-09-19 02:12:18 ERRORE spazio insufficiente\n2028-09-20 02:00:01 INFO backup avviato\n"
       "2028-09-20 02:18:10 INFO backup completato: 33.386 file\n2028-09-21 02:00:01 INFO backup avviato\n"
       "2028-09-21 02:49:55 ERRORE permesso negato su /dati")


def _log(answer):
    return [{"role": "user", "content": "Come sono andati gli ultimi due backup?"},
            {"role": "tool", "content": LOG}, {"role": "assistant", "content": answer}]


def test_a_failed_backup_called_successful_is_found():
    # generated row 1055089: the last backup failed
    assert tc.backup_problems(_log("L'ultimo backup è andato a buon fine il 21 settembre 2028 (sebbene con errore di "
                                   "permesso). Il precedente, del 20 settembre, ha completato 33.386 file.")) == [
        "backup outcome: «L'ultimo backup è andato a buon fine il 21 settembre 2028 (sebbene con errore di permesso).»"]
    assert tc.backup_problems(_log("L'ultimo backup è andato a buon fine. Il precedente è fallito."))
    assert tc.backup_problems(_log("Il backup del 20 settembre è fallito."))


def test_right_outcomes_negations_and_other_days_are_not_backup_problems():
    assert tc.backup_problems(_log("L'ultimo backup del 21 settembre non è andato a buon fine: permesso negato. "
                                   "Quello del 20 settembre è andato a buon fine con 33.386 file.")) == []
    assert tc.backup_problems(_log("L'ultimo backup non è stato completato con successo, mentre il precedente sì.")) == []
    # generated row 1055107: the last two failed, the days before them did not
    msgs = _log("Gli ultimi due sono falliti. I backup precedenti (dal 17 al 19) sono andati a buon fine.")
    msgs[1]["content"] = msgs[1]["content"].replace("INFO backup completato: 33.386 file", "ERRORE disco pieno")
    assert tc.backup_problems(msgs) == []


def test_problems_checks_dates_against_the_context_and_backups_in_log_rows():
    none = lambda texts: [[] for _ in texts]  # noqa: E731
    row = _row("Ho segnato la visita per il 20 ottobre (martedì).",
               user="Calendario: oggi mercoledì 2027-10-13; domani giovedì 2027-10-14.")
    assert tc.problems(row, IT, EN, none) == ["date: «20 ottobre (martedì)» is a mercoledì"]
    # round-6 doc 1060552: the dates of a notice from another year, copied from it
    row = _row("La prova scritta è mercoledì 10 ottobre.", family="chat_doc",
               user="Calendario: oggi lunedì 2028-11-20.\nprova scritta: mercoledì 10 ottobre, ore 15")
    assert tc.problems(row, IT, EN, none) == []
    row = {"messages": _log("L'ultimo backup è andato a buon fine."), "meta": {"family": "log_latest"}}
    assert tc.problems(row, IT, EN, none) == ["backup outcome: «L'ultimo backup è andato a buon fine.»"]
