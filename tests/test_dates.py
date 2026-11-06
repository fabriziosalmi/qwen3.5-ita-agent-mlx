"""The calendar check of generated conversations (qita.worlds.style.date_problems) and where it bites."""

import asyncio
import datetime as dt
import io
import json
import logging

from qita.worlds import pools as P
from qita.worlds import sample
from qita.worlds.phrasing import gate_user_v2
from qita.worlds.style import date_problems
from qita.worlds.world import build
from tests.test_chats import Teacher

logging.getLogger("mcp").setLevel(logging.CRITICAL)
NOW = dt.datetime(2027, 3, 10, 15)  # a Wednesday


def test_a_weekday_must_be_the_weekday_of_its_date():
    assert date_problems("alle 17 di lunedì 22 agosto", dt.datetime(2027, 8, 19)) == ["«lunedì 22 agosto» is a domenica"]
    assert date_problems("domenica 22 agosto", dt.datetime(2027, 8, 19)) == []
    assert date_problems("venerdì 12 marzo 2027", NOW) == [] and date_problems("venerdì 12 marzo 2026", NOW)


def test_a_weekday_after_its_date_must_be_its_weekday():
    # generated row 1055106: 8 July 2026 is a Wednesday
    today = dt.date(2026, 7, 1)
    assert date_problems("aperto l'8 luglio (giovedì) dalle 8:30", today) == ["«8 luglio (giovedì)» is a mercoledì"]
    assert date_problems("aperto l'8 luglio (mercoledì) dalle 9:00", today) == []
    assert date_problems("l'8 luglio 2027 (giovedì)", today) == []


def test_domani_must_be_tomorrow():
    assert date_problems("Ho fissato per domani 12 marzo alle 15:00.", NOW) == ["«domani 12 marzo» is not the 11"]
    assert date_problems("domani venerdì", NOW) == ["«domani venerdì» is not giovedì"]
    assert date_problems("dopodomani 12 marzo, oggi mercoledì; domani, giovedì 11 marzo", NOW) == []
    assert date_problems("oggi non posso, domani sì", NOW) == []


def test_the_writer_must_not_address_the_office():
    assert "addresses someone" in gate_user_v2("cari ufficio postale di Cagliari, segnami una visita alle 9.", [], [])
    assert "addresses someone" not in gate_user_v2("Gentile assistente, mi segni una visita alle 9?", [], [])


def _one(monkeypatch, user, answers, i=7):
    monkeypatch.setattr(sample, "system_text", lambda *a, **k: (None, None))
    monkeypatch.setattr(sample, "phrase_users", lambda *a, **k: ([user], None))
    log, kept = io.StringIO(), io.StringIO()
    ok, _ = asyncio.run(sample.one(i, Teacher(answers), Teacher([]), log, kept, weights={"chat_general": 1}, world_v=2))
    return ok, json.loads(log.getvalue().splitlines()[-1])


def test_generation_drops_a_wrong_weekday_from_the_writer_or_the_teacher(monkeypatch):
    _, facts = build(sample.SEED_BASE + 7, v=2)
    tomorrow = facts.now.date() + dt.timedelta(days=1)
    wrong = P.WEEKDAYS[(tomorrow.weekday() + 1) % 7]
    good = f"Certo: domani è {P.WEEKDAYS[tomorrow.weekday()]} {tomorrow.day} {P.MONTHS[tomorrow.month - 1]}."
    ok, row = _one(monkeypatch, "Che giorno è domani? Rispondimi in breve, grazie.", [good])
    assert ok, row
    ok, row = _one(monkeypatch, f"Che giorno è {wrong} {tomorrow.day} {P.MONTHS[tomorrow.month - 1]}? Dimmelo.", [good])
    assert not ok and "is a" in row["reason"]
    bad = f"Certo: domani è {wrong} {tomorrow.day} {P.MONTHS[tomorrow.month - 1]}."
    ok, row = _one(monkeypatch, "Che giorno è domani? Rispondimi in breve, grazie.", [bad] * 4)
    assert not ok and any("is a" in s for a in row["attempts"] for s in a["style"])
