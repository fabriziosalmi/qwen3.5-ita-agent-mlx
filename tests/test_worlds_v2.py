"""World v2 and its families: v1 unchanged, every oracle passes, the checks bite, held-out stays out."""

import asyncio
import hashlib
import logging
import random

import pytest

from qita.envs.itcalc import iban_ok
from qita.harness.checks import score
from qita.harness.decision import DECISIONS
from qita.harness.italian import is_italian
from qita.harness.runner import run_task
from qita.worlds.families import to_eval_task
from qita.worlds.families_v2 import ALL, HELD_OUT, ROUND6, TRAIN_ENVS, TRAIN_V2, add_envs
from qita.worlds.world import build
from tests.test_harness import Scripted
from tests.test_worlds import msg, oracle_outputs

logging.getLogger("mcp").setLevel(logging.CRITICAL)
SEEDS = range(30)


def run_v2(fam, seed, outputs_fn):
    world, facts = build(seed, v=2)
    t = add_envs(fam(facts, random.Random(seed * 1000 + 7)), seed)
    et = to_eval_task(t, f"{fam.__name__}-{seed}", ["(richiesta)"] * len(t["turns"]))
    traj = asyncio.run(run_task(et, Scripted(outputs_fn(t)), world=world))
    return t, score(et, traj)


def test_v1_worlds_are_unchanged():
    """Earlier data, replays and the v1 fresh file rebuild their worlds from the seed: v1 must not move."""
    h = hashlib.sha256()
    for seed in list(range(300)) + [1_001_000 + i for i in range(100)] + [1_009_000 + i for i in range(50)]:
        w, f = build(seed)
        h.update(repr((sorted(w.home.items()), w.notes, w.events, sorted(w.pages.items()), w.now,
                       [{k: v for k, v in i.items() if k not in ("iban_ok", "iban_shown")} for i in f.invoices],
                       f.pages, f.tax_codes)).encode())
    assert h.hexdigest()[:16] == "60556638fcca4baa"


@pytest.mark.parametrize("fam", ALL, ids=lambda f: f.__name__)
def test_oracle_passes_in_every_v2_world(fam):
    bad = []
    for seed in SEEDS:
        t, res = run_v2(fam, seed, oracle_outputs)
        if not res["success"]:
            bad.append((seed, {k: v for k, v in res["checks"].items() if not v}))
    assert not bad, bad[:3]


@pytest.mark.parametrize("fam", [f for f in ALL if f in TRAIN_V2 + ROUND6 or f.__name__ in HELD_OUT], ids=lambda f: f.__name__)
def test_v2_no_calls_fails_unless_no_tool(fam):
    """Break-test: the oracle answers with no tool call at all."""
    passed = sum(run_v2(fam, s, lambda t: [tt["oracle"] for tt in t["turns"]])[1]["success"] for s in SEEDS)
    if fam.__name__ in ("chat_general", "decline_more", "chat_more"):
        assert passed == len(SEEDS)
    elif fam.__name__ == "notes_delete_all":
        assert 0 < passed < len(SEEDS)  # only the branch that keeps the notes needs no call
    else:
        assert passed == 0, passed


def test_v2_answers_need_the_right_value():
    """Break-test: the right calls with a final answer that names nothing fails every family that checks an answer."""
    for fam in TRAIN_V2:
        for seed in range(10):
            world, facts = build(seed, v=2)
            t = fam(facts, random.Random(seed))
            if not any((tt["checks"].get("answer") or {}).get("contains") or (tt["checks"].get("answer") or {}).get("any")
                       for tt in t["turns"]) or fam.__name__ in ("chat_general", "decline_more", "chat_more"):
                continue
            outs = []
            for tt in t["turns"]:
                outs += [msg(m) for m in tt["plan"]] + ["Fatto."]
            et = to_eval_task(t, "b", ["x"] * len(t["turns"]))
            traj = asyncio.run(run_task(et, Scripted(outs), world=world))
            assert not score(et, traj)["success"], (fam.__name__, seed)


def test_v2_oracles_are_italian_and_turns_have_decisions():
    for fam in ALL:
        for seed in range(10):
            world, facts = build(seed, v=2)
            t = fam(facts, random.Random(seed))
            for tt in t["turns"]:
                assert is_italian(tt["oracle"]), (fam.__name__, tt["oracle"])
                assert not set(map(str.casefold, tt["must_say"])) & set(map(str.casefold, tt["must_not"])), fam.__name__
            for tt in to_eval_task(t, "d", ["x"] * len(t["turns"]))["turns"]:
                assert tt["decision"] in DECISIONS


def test_wrong_ibans_fail_and_shown_ones_are_in_the_file():
    seen = {True: 0, False: 0}
    for seed in range(60):
        world, facts = build(seed, v=2)
        for inv in facts.invoices:
            assert iban_ok(inv["iban_shown"]) == inv["iban_ok"]
            assert inv["iban_shown"] in world.home[inv["path"]]
            seen[inv["iban_ok"]] += 1
    assert seen[True] and seen[False]


def test_extra_envs_are_train_envs_only():
    from qita.envs import ROLE
    assert all(ROLE[e] == "train" for e in TRAIN_ENVS)
    for seed in range(50):
        world, facts = build(seed, v=2)
        t = add_envs(ALL[seed % len(ALL)](facts, random.Random(seed)), seed)
        assert all(ROLE[e] == "train" for e in t["envs"]) and len(t["envs"]) == len(set(t["envs"]))


def test_v2_files_never_copy_eval_fixtures():
    from pathlib import Path
    fixtures = {p.read_text(encoding="utf-8") for p in Path("fixtures/home").rglob("*") if p.is_file()}
    for seed in range(50):
        world, _ = build(seed, v=2)
        assert not (set(world.home.values()) & fixtures - {"JPEG\n", "PDF\n", "ZIP\n"})


def test_sampling_never_draws_held_out_families():
    """Break-test target: drop the HELD_OUT filter in qita.worlds.sample.family and this fails."""
    from qita.worlds.sample import SEED_BASE, family
    weights = {f.__name__: 1.0 for f in ALL}
    drawn = {family(i, SEED_BASE + i, weights, 2).__name__ for i in range(3000)}
    drawn |= {family(i, SEED_BASE + i, None, 2).__name__ for i in range(200)}
    assert not drawn & HELD_OUT
    assert {f.__name__ for f in TRAIN_V2} <= drawn


def test_v1_sampling_keeps_the_old_order():
    from qita.worlds.families import FAMILIES
    from qita.worlds.sample import SEED_BASE, family
    assert [family(i, SEED_BASE + i, None, 1) for i in range(60)] == [FAMILIES[i % len(FAMILIES)] for i in range(60)]
    w = {"read_shopping": 1.0, "wrong_folder": 2.5, "contact_phone": 5.0}
    assert {family(i, SEED_BASE + i, w, 1).__name__ for i in range(300)} == {"read_shopping", "wrong_folder"}


def test_roughen_keeps_the_required_words():
    from qita.worlds.phrasing import gate_user, roughen
    text = "Perché non mi dici quante cose ci sono nella lista spesa.txt? È urgente, grazie mille."
    changed = 0
    for k in range(200):
        out = roughen(text, random.Random(k), ["spesa.txt"])
        assert "spesa.txt" in out and not gate_user(out, ["spesa.txt"], [])
        changed += out != text
    assert changed > 150


def test_v2_gate_rejects_sentences_that_address_someone():
    from qita.worlds.phrasing import gate_user_v2
    for bad in ["Vittoria, chiedi il numero di telefono.", "Ciao Francesca, cosa devo fare secondo gli appunti?",
                "ehi Marco, leggi il file lista-cose.txt", "Alice"]:
        assert gate_user_v2(bad, [], []), bad
    for ok in ["Ciao, che numero ha Vittoria?", "Senti, quanto ho speso in farmacia?", "Assistente, leggi il file app.log",
               "Scusa, a che ora apre il comune di Bari lunedì?"]:
        assert not gate_user_v2(ok, [], []), ok
