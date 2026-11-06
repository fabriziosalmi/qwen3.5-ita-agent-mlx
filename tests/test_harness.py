"""The harness itself: envs, matchers and the agent loop, with a scripted fake model."""

import asyncio
import logging

import pytest

from qita.harness.checks import match_value, score
from qita.harness.italian import is_italian
from qita.harness.runner import run_task, split_output
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call

logging.getLogger("mcp").setLevel(logging.CRITICAL)


def call(name, **args):
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


class Scripted:
    """Replays fixed assistant outputs, one per model call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def parse_tool_call(self, body, tools):
        return parse_tool_call(body, tools)

    def complete(self, messages, tools):
        text = self.outputs.pop(0) if self.outputs else "Fatto."
        return {"text": text, "prompt_tokens": 0, "gen_tokens": 1, "hit_max_tokens": False, "seconds": 0.0}


def run(task, outputs):
    task = {"category": "single", **task}
    task["turns"] = [{"user": x} if isinstance(x, str) else x for x in task["turns"]]
    traj = asyncio.run(run_task(task, Scripted(outputs)))
    return traj, score(task, traj)


def test_read_file_end_to_end():
    task = {"id": "t", "envs": ["files"], "turns": ["Cosa c'è nella lista della spesa?"],
            "checks": {"calls": [{"tool": "read_text_file", "args": {"path": {"path": "/home/utente/spesa.txt"}}}],
                       "answer": {"contains": ["latte", "caffè"]}}}
    traj, res = run(task, [call("read_text_file", path="spesa.txt"), "Nella lista ci sono latte, uova e caffè."])
    assert "latte" in traj["calls"][0]["result"]
    assert res["success"], res["checks"]


def test_wrong_answer_fails():
    task = {"id": "t", "envs": ["files"], "turns": ["Cosa c'è nella lista della spesa?"],
            "checks": {"answer": {"contains": ["latte"]}}}
    _, res = run(task, [call("read_text_file", path="spesa.txt"), "Nella lista ci sono mele e pere."])
    assert not res["success"] and not res["checks"]["answer"]


def test_english_answer_fails():
    task = {"id": "t", "envs": ["files"], "turns": ["Cosa c'è nella lista della spesa?"]}
    _, res = run(task, ["The list contains milk and eggs, and it is in your home folder."])
    assert not res["checks"]["italian"]


def test_sandbox_escape_is_an_error():
    task = {"id": "t", "envs": ["files"], "turns": ["leggi /etc/passwd"]}
    traj, _ = run(task, [call("read_text_file", path="/etc/passwd"), "Non posso."])
    assert traj["calls"][0]["error"] and "Access denied" in traj["calls"][0]["result"]
    traj, _ = run(task, [call("read_text_file", path="../../etc/passwd"), "Non posso."])
    assert traj["calls"][0]["error"]


def test_delete_changes_state_and_no_call_catches_it():
    task = {"id": "t", "envs": ["files"], "turns": ["cancella spesa.txt"],
            "checks": {"no_call": ["delete_file"], "state": [{"kind": "file_exists", "path": "spesa.txt"}]}}
    _, res = run(task, [call("delete_file", path="spesa.txt"), "Ho eliminato il file."])
    assert not res["checks"]["no_call[0]:delete_file"]
    assert not res["checks"]["state[0]:file_exists"]


def test_sandbox_is_fresh_per_task():
    task = {"id": "t", "envs": ["files"], "turns": ["x"],
            "checks": {"state": [{"kind": "file_exists", "path": "spesa.txt"}]}}
    run(task, [call("delete_file", path="spesa.txt"), "Fatto."])
    _, res = run(task, ["Fatto."])
    assert res["checks"]["state[0]:file_exists"]


def test_shell_pipe_blocked_and_date():
    task = {"id": "t", "envs": ["shell"], "turns": ["x"]}
    traj, _ = run(task, [call("esegui_comando", comando="grep -c ERRORE log/backup.log"),
                         call("esegui_comando", comando="rm -rf / ; ls"),
                         call("esegui_comando", comando="sudo ls"),
                         call("esegui_comando", comando="date"),
                         call("esegui_comando", comando="cat /home/utente/spesa.txt | wc -l"),
                         call("esegui_comando", comando="cat /etc/hosts"),
                         "Fatto."])
    r = [c["result"] for c in traj["calls"]]
    assert r[0].strip() == "3"
    assert traj["calls"][1]["error"] and "non supportato" in r[1]
    assert traj["calls"][2]["error"] and "non consentito" in r[2]
    assert "2026" in r[3]
    assert r[4].strip() == "6"
    assert traj["calls"][5]["error"]


def test_grep_no_match_is_not_an_error():
    task = {"id": "t", "envs": ["shell"], "turns": ["x"]}
    traj, _ = run(task, [call("esegui_comando", comando="grep zzz spesa.txt"), "Niente."])
    assert not traj["calls"][0]["error"]


def test_itcalc_oracles():
    task = {"id": "t", "envs": ["itcalc"], "turns": ["x"]}
    traj, _ = run(task, [call("giorni_lavorativi", inizio="2026-11-02", fine="2026-11-13"),
                         call("giorni_lavorativi", inizio="2026-12-21", fine="2026-12-31"),
                         call("calcola_iva", importo="1250", aliquota="22"),
                         call("calcola_iva", importo="1525", scorporo="true"),
                         call("valida_codice_fiscale", codice="RSSMRA85T10A562S"),
                         call("valida_codice_fiscale", codice="RSSMRA85T10A562T"),
                         call("valida_iban", iban="IT60 X054 2811 1010 0000 0123 456"),
                         call("aggiungi_giorni_lavorativi", data="2026-12-23", giorni="3"),
                         "Fatto."])
    r = [c["result"] for c in traj["calls"]]
    assert r[0].endswith(": 10")
    # 21-24 and 28-31 December; 25/12/2026 is a Friday, 26/12 a Saturday
    assert ": 8;" in r[1] and "Natale" in r[1] and "Santo Stefano" not in r[1]
    assert "IVA 22%: 275,00" in r[2] and "Totale: 1.525,00" in r[2]
    assert "Imponibile: 1.250,00" in r[3]
    assert r[4] == "Valido" and "atteso S" in r[5]
    assert r[6] == "Valido"
    assert "2026-12-29" in r[7]


def test_unknown_tool_and_malformed():
    task = {"id": "t", "envs": ["notes"], "turns": ["x"]}
    traj, res = run(task, [call("leggi_mente"), "Fatto."])
    assert traj["calls"][0]["unknown"] and traj["calls"][0]["error"]
    traj, res = run(task, ["<tool_call>\n<function=cerca_note>\n<parameter=testo>\nbici\n"])
    assert traj["flags"]["malformed"] == 1 and not res["checks"]["well_formed"]


def test_loop_flag():
    task = {"id": "t", "envs": ["notes"], "turns": ["x"]}
    traj, res = run(task, [call("cerca_note", testo="bici")] * 5)
    assert traj["flags"]["loop"] and not res["checks"]["no_loop"]


def test_multi_turn_confirmation():
    task = {"id": "t", "envs": ["notes"],
            "turns": [{"user": "elimina le note sulla casa", "checks": {"asks": True, "no_call": ["elimina_nota"]}},
                      {"user": "sì, procedi", "checks": {"calls": [{"tool": "elimina_nota", "args": {"id": 1}},
                                                                    {"tool": "elimina_nota", "args": {"id": 4}}]}}],
            "checks": {"state": [{"kind": "note_absent", "id": 1}, {"kind": "note_absent", "id": 4}]}}
    _, res = run(task, [call("cerca_note", testo="casa"), "Ho trovato le note 1 e 4. Le elimino?",
                        call("elimina_nota", id="1") + "\n" + call("elimina_nota", id="4"), "Ho eliminato le due note."])
    assert res["success"], res["checks"]


@pytest.mark.parametrize("spec,value,ok", [
    ({"path": "/home/utente/note/idee.txt"}, "~/note/idee.txt", True),
    ({"path": "/home/utente/note/idee.txt"}, "note/idee.txt", True),
    ({"path": "/home/utente/note"}, "/home/utente/note/", True),
    ({"path": "/home/utente/note/idee.txt"}, "idee.txt", False),
    ({"num": 1250}, "1250.00", True),
    ({"date": "2026-11-05"}, "2026-11-05T14:00", True),
    ({"any": ["a", {"re": "^b"}]}, "Bx", True),
    (2, "2", True),
    ("Latte", " latte ", True),
])
def test_matchers(spec, value, ok):
    assert match_value(spec, value) is ok


def test_split_output_text_after_call_is_malformed():
    _, bodies, ok = split_output(call("x") + " e poi ho finito")
    assert bodies and not ok


@pytest.mark.parametrize("text,ok", [
    ("Nella lista ci sono latte e uova.", True),
    ("Il file /home/utente/spesa.txt contiene 6 righe.", True),
    ("The file contains six lines and it is in your folder.", False),
    ("42", True),
    ("In inglese si dice «the file has been saved».", True),
    ("The file \"has been saved\" and it is in your folder.", False),
])
def test_italian(text, ok):
    assert is_italian(text) is ok


def to_text(step):
    if isinstance(step, str):
        return step
    steps = step if isinstance(step, list) else [step]
    return "\n".join(call(s["name"], **s["args"]) for s in steps)


def _oracle_cases(split):
    from qita import tasks as T
    from tests.oracles import DEV, TEST
    table = {"dev": DEV, "test": TEST}[split]
    tasks = T.load(f"eval/{split}", split)
    assert {t["id"] for t in tasks} == set(table), "every task needs exactly one oracle"
    return [(t, table[t["id"]]) for t in tasks]


@pytest.mark.parametrize("task,steps", _oracle_cases("dev") + _oracle_cases("test"),
                         ids=lambda x: x["id"] if isinstance(x, dict) else "")
def test_task_is_passable(task, steps):
    traj = asyncio.run(run_task(task, Scripted([to_text(s) for s in steps])))
    res = score(task, traj)
    assert res["success"], {k: v for k, v in res["checks"].items() if not v}


def test_split_output_drops_thinking_block():
    prose, bodies, ok = split_output("<think>\nDevo leggere il file.\n</think>\n\n" + call("read_text_file", path="spesa.txt"))
    assert ok and prose == "" and len(bodies) == 1
    prose, bodies, ok = split_output("<think>\nragiono\n</think>\n\nCiao!")
    assert ok and prose == "Ciao!" and not bodies


def test_date_text_of_eval_world():
    from qita.envs.base import EVAL_NOW
    from qita.envs.shell import date_text
    assert date_text(EVAL_NOW) == "mer  4 nov 2026 10:00:00 CET"


def test_generated_world_replaces_fixtures():
    import datetime as dt
    from qita.envs.base import ROME, World
    w = World(now=dt.datetime(2027, 3, 15, 9, 30, tzinfo=ROME),
              home={"lista.txt": "farina\nzucchero\n", "lavoro/report.md": "# Report"},
              notes=[{"id": 1, "testo": "pagare la TARI", "etichette": ["casa"], "data": "2027-03-01"}],
              events=[{"id": "ev1", "title": "Visita", "start": "2027-03-16T11:00", "end": "2027-03-16T12:00", "location": None}],
              pages={"https://a.example/x": {"title": "Pagina", "body": "testo della pagina"}})
    task = {"id": "w", "category": "single", "envs": ["files", "shell", "notes", "calendar", "web"],
            "turns": [{"user": "x"}],
            "checks": {"state": [{"kind": "file_exists", "path": "lista.txt"}, {"kind": "file_absent", "path": "spesa.txt"}]}}
    out = [call("list_directory", path="~"), call("esegui_comando", comando="date"), call("aggiungi_nota", testo="nuova"),
           call("list_events", start_date="2027-03-16", end_date="2027-03-16"), call("fetch", url="https://a.example/x"), "Fatto."]
    traj = asyncio.run(run_task(task, Scripted(out), world=w))
    r = [c["result"] for c in traj["calls"]]
    assert "lista.txt" in r[0] and "spesa.txt" not in r[0] and "[DIR] lavoro" in r[0]
    assert r[1] == "lun 15 mar 2027 09:30:00 CET"
    assert "[2027-03-15]" in r[2] and "#2" in r[2]
    assert "Visita" in r[3]
    assert "testo della pagina" in r[4]
    assert all(s["ok"] for s in traj["state"])


def test_calls_any_needs_one_of_the_alternatives():
    from qita.harness.checks import call_checks
    group = {"calls_any": [[{"tool": "read_text_file"}, {"tool": "esegui_comando", "args": {"comando": {"contains": "x.log"}}}]]}
    assert call_checks(group, [{"name": "esegui_comando", "arguments": {"comando": "cat x.log"}}], "")["calls_any[0]"]
    assert not call_checks(group, [{"name": "esegui_comando", "arguments": {"comando": "ls"}}], "")["calls_any[0]"]
    assert not call_checks(group, [], "")["calls_any[0]"]


def test_listings_do_not_show_the_host():
    import grp
    import os
    import pwd
    task = {"id": "t", "envs": ["shell"], "turns": ["Elenca la home."]}
    traj, _ = run(task, [call("esegui_comando", comando="ls -la"), "Fatto."])
    out = traj["calls"][0]["result"]
    assert "utente" in out and "@" not in out
    assert pwd.getpwuid(os.getuid()).pw_name not in out and grp.getgrgid(os.getgid()).gr_name not in out
    dates = {" ".join(line.split()[5:8]) for line in out.splitlines()[1:]}
    assert len(dates) == 1, dates  # . and .. carry the world's date like every file


def test_shell_paths_glued_to_an_option_stay_in_the_sandbox(tmp_path):
    """-f/etc/hosts, --files0-from=/etc/hosts and -o/tmp/x read or write host paths unless they are mapped."""
    out = tmp_path / "escaped.txt"
    task = {"id": "t", "envs": ["shell"], "turns": ["x"]}
    traj, _ = run(task, [call("esegui_comando", comando="grep -f/etc/hosts spesa.txt"),
                         call("esegui_comando", comando="wc --files0-from=/etc/hosts"),
                         call("esegui_comando", comando=f"sort -o{out} spesa.txt"),
                         "Fatto."])
    assert all(c["error"] for c in traj["calls"]), [c["result"] for c in traj["calls"]]
    assert not out.exists()
