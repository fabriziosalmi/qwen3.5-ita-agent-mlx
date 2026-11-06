"""Round-6 agent families (qita/worlds/families_v2.py ROUND6): acting before the condition is checked, doing half of
a two-part request, or reading the oldest lines of a log all fail their checks."""

import asyncio
import logging
import random

from qita.harness.checks import score
from qita.harness.runner import run_task
from qita.worlds.families import to_eval_task
from qita.worlds.families_v2 import cond_free, cond_office, log_latest, two_part_fee
from qita.worlds.world import build
from tests.test_harness import Scripted
from tests.test_worlds import msg

logging.getLogger("mcp").setLevel(logging.CRITICAL)


def _cases(fam, want_write, n=40):
    """(world, task) pairs whose plan writes an event (want_write) or not."""
    out = []
    for seed in range(1_060_000, 1_060_000 + n):
        world, facts = build(seed, v=2)
        t = fam(facts, random.Random(seed))
        if ("create_event" in str(t["turns"][0]["plan"])) == want_write:
            out.append((world, t))
    assert out
    return out


def _score(world, t, outputs):
    et = to_eval_task(t, "r6", ["(richiesta)"])
    return score(et, asyncio.run(run_task(et, Scripted(outputs), world=world)))["success"]


def _plan(t):
    return [msg(step) for step in t["turns"][0]["plan"]]


def test_both_branches_of_the_conditions_occur():
    for fam in (cond_office, cond_free):
        assert len(_cases(fam, True)) >= 10 and len(_cases(fam, False)) >= 10


def test_writing_when_the_condition_is_false_fails():
    for fam in (cond_office, cond_free):
        for world, t in _cases(fam, False)[:5]:
            assert _score(world, t, _plan(t) + [t["turns"][0]["oracle"]])
            book = msg([{"name": "create_event", "args": {"title": "X", "start": "2030-01-01T09:00", "end": "2030-01-01T10:00"}}])
            assert not _score(world, t, _plan(t) + [book, t["turns"][0]["oracle"]]), fam.__name__


def test_writing_before_reading_the_condition_fails():
    for world, t in _cases(cond_office, True)[:5]:
        steps = _plan(t)
        assert not _score(world, t, [steps[-1]] + steps[:-1] + [t["turns"][0]["oracle"]])  # the event first


def test_half_of_a_two_part_request_fails():
    for world, t in _cases(two_part_fee, True)[:5]:
        steps, oracle = _plan(t), t["turns"][0]["oracle"]
        assert _score(world, t, steps + [oracle])
        assert not _score(world, t, steps[:-1] + [oracle])  # the fee, no event
        assert not _score(world, t, steps + ["Fatto, ho segnato la visita."])  # the event, no fee


def test_the_newest_log_lines_are_the_answer():
    from qita.worlds.families_v2 import _day_forms
    for seed in range(1_060_000, 1_060_010):
        world, facts = build(seed, v=2)
        t = log_latest(facts, random.Random(seed))
        steps = _plan(t)
        assert _score(world, t, steps + [t["turns"][0]["oracle"]])
        first = facts.backup["runs"][0]
        wrong = f"L'ultimo backup, del {_day_forms(first['day'])[0]}, è andato bene."
        assert not _score(world, t, steps + [wrong]), seed
        last, prev = facts.backup["runs"][-1], facts.backup["runs"][-2]  # seen in generated data: the right last, a wrong previous
        say = lambda x: "completato" if x["ok"] else f"fallito ({x['error']})"  # noqa: E731
        other = first["day"]
        wrong_prev = (f"L'ultimo backup, del {_day_forms(last['day'])[0]}, è {say(last)}; quello prima, del "
                      f"{_day_forms(other)[0]}, è {say(prev)}.")
        assert not _score(world, t, steps + [wrong_prev]), seed
        right = t["turns"][0]["oracle"]  # seen in generated data: the right two runs, and a claim about an older one
        assert not _score(world, t, steps + [f"{right} Il backup del {_day_forms(other)[0]} invece no."]), seed
