"""Defects found in the review before publication, each pinned by a test that fails on the code before the fix."""

import asyncio
import logging
import random

from qita.envs.base import World
from qita.harness.checks import call_checks, check_answer
from qita.harness.runner import WRITES, id_args, run_task
from qita.harness.tables import csv_totals, number
from tests.test_harness import Scripted, call

logging.getLogger("mcp").setLevel(logging.CRITICAL)


def _call(name, **flags):
    return {"turn": 0, "name": name, "arguments": {}, "error": True, "fault": False, "guard": False, "gated": False,
            **flags}


def test_a_refused_call_counts_for_no_call_but_not_as_a_call_made():
    checks = {"calls": [{"tool": "esegui_comando"}], "calls_any": [[{"tool": "esegui_comando"}]],
              "no_call": ["delete_file"]}
    for flag in ("gated", "guard", "fault"):
        out = call_checks(checks, [_call("esegui_comando", **{flag: True})], "")
        assert not out["call[0]:esegui_comando"] and not out["calls_any[0]"], flag
    out = call_checks(checks, [_call("delete_file", gated=True)], "")
    assert not out["no_call[0]:delete_file"]  # an attempt to delete is still an attempt


def test_cert_and_git_writes_are_writes():
    assert {"set_auto_renew", "issue_certificate", "renew_certificate", "revoke_certificate",
            "git_add", "git_commit"} <= WRITES


def test_dot_decimals_in_a_semicolon_table():
    assert [number(x, True) for x in ("12.50", "1.250", "1.234,56", "10,25")] == [12.5, 1250.0, 1234.56, 10.25]
    assert "spesa 22,75" in csv_totals("voce;spesa\nluce;12.50\ngas;10.25\n")


def test_count_counts_the_file_not_the_lines_the_harness_adds():
    t = {"id": "c", "category": "L1", "envs": ["files"], "turns": [{"user": "x"}], "checks": {}}
    home = {"conti.csv": "voce,euro\nluce,10\ngas,20\nacqua,5\n"}
    steps = [call("read_text_file", path="conti.csv"), "Fatto."]
    traj = asyncio.run(run_task(t, Scripted(steps), world=World(home=home), harness="+csvsum+count"))
    assert traj["calls"][0]["result"].endswith("[il file ha 4 righe]"), traj["calls"][0]["result"]


def test_the_order_of_named_arguments_does_not_depend_on_hashing():
    import os
    import subprocess
    import sys
    code = ("from qita.harness.runner import id_args; "
            "print(id_args({'arguments': {'x': 'wxyz', 'y': 'abcd', 'z': 'mnop', 'w': 'qrst'}}))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout for seed in range(8)}
    assert outs == {"['abcd', 'mnop', 'qrst', 'wxyz']\n"}, outs
    assert id_args({"arguments": {"a": "abcd"}}) == ["abcd"]


def test_a_whole_number_is_not_found_inside_another():
    assert not check_answer({"number": [4]}, "Il file ha 14 righe.")
    assert check_answer({"number": [4]}, "Il file ha 4 righe.")


def test_backup_days_are_ordered_by_date_not_by_first_sight():
    from qita.worlds.textcheck import backup_problems
    log = ["2026-11-01 02:00:05 INFO backup completato", "2026-11-02 02:00:05 ERRORE disco pieno",
           "2026-11-03 02:00:05 INFO backup completato", "2026-11-04 02:00:05 INFO backup completato"]
    msgs = [{"role": "tool", "content": "\n".join(log[-2:])},   # tail first
            {"role": "tool", "content": "\n".join(log)},        # then the whole file
            {"role": "assistant", "content": "L'ultimo backup è andato a buon fine."}]
    assert backup_problems(msgs) == []


def test_number_forms_in_tools():
    from qita.envs.calc import _plain
    from qita.worlds.store import build_store
    assert _plain("0.125") == "0.125" and _plain("1.234") == "1234"
    s = {**build_store(11, domain="dischi"), "numeric": False, "key": "slug", "fmt": "json"}
    s["records"] = [{"slug": "notte-elettrica", "title": "Notte elettrica", "artist": "Nebbia Fitta", "year": 1999,
                     "bpm": 132, "genre": "acid", "shelf": "A1"}]
    up = s["tools"]["update"]
    t = {"id": "n", "category": "single", "envs": ["store"], "turns": [{"user": "x"}], "checks": {}}
    steps = [call(up, slug="notte-elettrica", field="bpm", value="3.5"),
             call(up, slug="notte-elettrica", field="bpm", value="1.250"), "Fatto."]
    traj = asyncio.run(run_task(t, Scripted(steps), world=World(store=s)))
    assert traj["calls"][0]["error"] and not traj["calls"][1]["error"], [c["result"] for c in traj["calls"]]


def test_file_lacks_ignores_case():
    t = {"id": "l", "category": "single", "envs": ["files"], "turns": [{"user": "x"}],
         "checks": {"state": [{"kind": "file_lacks", "path": "x.log", "text": ["errore"]}]}}
    traj = asyncio.run(run_task(t, Scripted(["Fatto."]), world=World(home={"x.log": "Errore grave\n"})))
    assert not traj["state"][0]["ok"]


def test_one_is_singular_only_with_a_plural_verb():
    from qita.worlds.style import style_problems
    assert style_problems("Sono 1 file.")
    assert not style_problems("Hai 1 appuntamento domani.") and not style_problems("Sono 1.250 euro.")


def test_the_second_file_of_the_listing_is_the_one_read():
    from qita.worlds.families import mt_files_reference
    from qita.worlds.world import build
    for seed in range(60):
        world, f = build(seed, v=2)
        folder = f.folders["docs"]
        listing = sorted(p.split("/")[-1] for p in world.home if p.rsplit("/", 1)[0] == folder)
        t = mt_files_reference(f, random.Random(seed))
        read = t["turns"][1]["checks"]["calls"][0]["args"]["path"]["path"].split("/")[-1]
        invoices = sorted(i["path"].split("/")[-1] for i in f.invoices)
        assert read == (listing[1] if listing[1] in invoices else invoices[1]), (seed, listing, read)


def test_a_group_delete_checks_that_nothing_else_went():
    from qita.worlds.families_store import store_delete_group
    from qita.worlds.store import attach_store
    from qita.worlds.world import build
    world, f = build(7, v=2)
    attach_store(world, f, 7)
    t = store_delete_group(f, random.Random(7))
    assert any(s["kind"] == "store_count" for s in t["checks"]["state"])
