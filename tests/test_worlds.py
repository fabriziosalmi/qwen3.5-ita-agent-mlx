"""Generated worlds and task families: every oracle plan passes its own checks, and the checks bite."""

import asyncio
import logging
import random

import pytest

from qita.harness.checks import score
from qita.harness.italian import is_italian
from qita.harness.runner import run_task
from qita.worlds.families import FAMILIES, to_eval_task
from qita.worlds.world import build
from tests.test_harness import Scripted, call

logging.getLogger("mcp").setLevel(logging.CRITICAL)
SEEDS = range(40)


def msg(calls):
    return "\n".join(call(c["name"], **c["args"]) for c in calls)


def run_family(fam, seed, outputs_fn):
    world, facts = build(seed)
    t = fam(facts, random.Random(seed * 1000 + 7))
    et = to_eval_task(t, f"{fam.__name__}-{seed}", ["(richiesta)"] * len(t["turns"]))
    traj = asyncio.run(run_task(et, Scripted(outputs_fn(t)), world=world))
    return t, score(et, traj)


def oracle_outputs(t):
    out = []
    for tt in t["turns"]:
        out += [msg(m) for m in tt["plan"]] + [tt["oracle"]]
    return out


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.__name__)
def test_oracle_passes_in_every_world(fam):
    bad = []
    for seed in SEEDS:
        t, res = run_family(fam, seed, oracle_outputs)
        if not res["success"]:
            bad.append((seed, {k: v for k, v in res["checks"].items() if not v}))
    assert not bad, bad[:3]


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.__name__)
def test_no_calls_fails_unless_no_tool(fam):
    """Break-test: the same final answers with no tool call at all."""
    passed = sum(run_family(fam, s, lambda t: [tt["oracle"] for tt in t["turns"]])[1]["success"] for s in SEEDS)
    if fam.__name__ in ("no_tool_chat", "decline"):
        assert passed == len(SEEDS)
    elif fam.__name__ == "bulk_delete_photos":
        # declining branch needs no call; the confirming branch must fail
        assert passed < len(SEEDS)
    else:
        assert passed == 0, passed


def test_oracle_answers_are_italian():
    for fam in FAMILIES:
        for seed in range(10):
            world, facts = build(seed)
            t = fam(facts, random.Random(seed))
            for tt in t["turns"]:
                assert is_italian(tt["oracle"]), (fam.__name__, tt["oracle"])


def test_worlds_differ_and_never_copy_eval_fixtures():
    from pathlib import Path
    fixtures = {p.read_text(encoding="utf-8") for p in Path("fixtures/home").rglob("*") if p.is_file()}
    seen = set()
    for seed in range(50):
        world, facts = build(seed)
        assert not (set(world.home.values()) & fixtures - {"JPEG\n"})
        key = (facts.now, tuple(facts.shopping["items"]))
        assert key not in seen
        seen.add(key)


def _mutate(fam, seed, mutate):
    world, facts = build(seed)
    t = fam(facts, random.Random(seed * 1000 + 7))
    mutate(t, facts)
    et = to_eval_task(t, f"m-{seed}", ["(richiesta)"] * len(t["turns"]))
    traj = asyncio.run(run_task(et, Scripted(oracle_outputs(t)), world=world))
    return score(et, traj)["success"]


def test_break_wrong_reference_fails():
    from qita.worlds.families import mt_files_reference

    def first_instead_of_second(t, facts):
        files = sorted(i["path"] for i in facts.invoices)
        for tt in t["turns"][1:]:
            for m in tt["plan"]:
                for c in m:
                    c["args"]["path"] = files[0]
    assert not any(_mutate(mt_files_reference, s, first_instead_of_second) for s in range(10))


def test_break_correction_by_duplicate_fails():
    from qita.worlds.families import mt_correction

    def create_again(t, facts):
        c = t["turns"][1]["plan"][0][0]
        t["turns"][1]["plan"] = [[{"name": "create_event", "args": {"title": "x", "start": c["args"]["start"], "end": c["args"]["end"]}}]]
    assert not any(_mutate(mt_correction, s, create_again) for s in range(10))


def test_fault_is_injected_once_and_recovery_passes():
    from qita.worlds.families import read_shopping
    world, facts = build(3)
    t = read_shopping(facts, random.Random(3))
    from qita.worlds.families import inject_fault
    inject_fault(t)
    et = to_eval_task(t, "f", ["(richiesta)"])
    plan = [m for m in t["turns"][0]["plan"] if m[0]["name"] == "read_text_file"][0]
    outs = [msg(plan), msg(plan), t["turns"][0]["oracle"]]  # first read fails, the retry succeeds
    traj = asyncio.run(run_task(et, Scripted(outs), world=world))
    assert [c["fault"] for c in traj["calls"]] == [True, False]
    assert score(et, traj)["success"]
    traj = asyncio.run(run_task(et, Scripted([msg(plan), t["turns"][0]["oracle"]]), world=world))
    assert not score(et, traj)["success"]  # giving up after the error fails


def test_every_family_turn_has_a_decision():
    from qita.harness.decision import DECISIONS
    for fam in FAMILIES:
        world, facts = build(5)
        t = fam(facts, random.Random(5))
        for tt in to_eval_task(t, "d", ["x"] * len(t["turns"]))["turns"]:
            assert tt["decision"] in DECISIONS
