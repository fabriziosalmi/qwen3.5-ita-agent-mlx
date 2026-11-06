"""Answer style for training data: light markdown, no emoji, and short
answers wherever possible, so that long multi-turn conversations stay inside the context.

Bold and lists only when they help, no headings, tables or rules, no emoji, a few lines.
The teacher gets STYLE_HINT in its system message while generating; the stored conversation
keeps the original system message (or none), so the student learns the style without being
told (context distillation). Every assistant message is then checked by style_problems.
"""

from __future__ import annotations

import datetime as dt
import re

from . import pools as P

STYLE_HINT = ("Stile delle risposte: rispondi nel modo più breve possibile, di solito una o due frasi, in italiano "
              "corretto. Niente emoji e niente titoli. Usa il grassetto o un elenco solo se servono davvero. "
              "Prima di usare uno strumento non spiegare cosa stai per fare.")
# Behaviour rules the teacher sees while generating (never stored in the data), so the student
# learns the safety policy (README.md) without being told.
POLICY_HINT = ("Regole: usa sempre gli strumenti per leggere dati, non inventare. Se l'utente nomina una cosa precisa "
               "da cancellare o modificare, fallo. Se chiede di cancellare più cose insieme o in modo generico "
               "(per esempio 'tutte le foto'), prima elenca cosa cancelleresti e chiedi conferma, senza cancellare. "
               "Se manca un dato necessario, come giorno e ora, chiedilo invece di indovinarlo. "
               "Se la cosa da cancellare non c'è, dillo e non proporre di cancellare altro.")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
MAX_WORDS = {"summarize": 100}
DEFAULT_MAX_WORDS = 60
MAX_PROSE_BEFORE_CALL = 25
MAX_BOLD = 3
# Agreement errors the teacher makes and the text check misses (seen in generated store data): "Ci sono 1 pianta",
# "nello salotto" (lo/nello/dello/allo/sullo/dallo only before s + consonant, z, gn, ps, pn, x, y or a vowel).
ONE_PLURAL = re.compile(r"\b(?:ci sono|sono)\s+1(?![\d.,])\b", re.IGNORECASE)  # "sono 1 file"; not "hai 1", "sono 1.250"
LO_BEFORE = re.compile(r"\b(?:nello|dello|allo|sullo|dallo)\s+(?!s[^aeiouàèéìòù\s]|z|gn|ps|pn|x|y)[b-df-hj-np-tv-z]",
                       re.IGNORECASE)


def style_problems(text: str, category: str = "", before_call: bool = False) -> list[str]:
    why = []
    words = len(text.split())
    lines = text.splitlines()
    if EMOJI.search(text):
        why.append("emoji")
    if any(re.match(r"\s*#{1,6}\s", ln) for ln in lines):
        why.append("heading")
    if any(ln.strip().startswith("|") for ln in lines):
        why.append("table")
    if any(re.fullmatch(r"\s*([-*_])\1{2,}\s*", ln) for ln in lines):
        why.append("rule")
    if ONE_PLURAL.search(text) or LO_BEFORE.search(text):
        why.append("agreement")
    if text.count("**") // 2 > MAX_BOLD:
        why.append("too much bold")
    limit = MAX_PROSE_BEFORE_CALL if before_call else MAX_WORDS.get(category, DEFAULT_MAX_WORDS)
    if words > limit:
        why.append(f"{words} words > {limit}")
    return why


def conversation_problems(messages: list[dict], category: str) -> list[str]:
    why = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        for w in style_problems(m.get("content") or "", category, before_call=bool(m.get("tool_calls"))):
            why.append(w)
    return why


_DAY = "|".join(P.WEEKDAYS)
_MONTH = "|".join(P.MONTHS)
DATED = re.compile(rf"\b({_DAY})\s+(\d{{1,2}})\s+({_MONTH})(?:\s+(\d{{4}}))?", re.IGNORECASE)
RELATIVE = re.compile(rf"\b(oggi|domani|dopodomani)\b[\s,]*(?:({_DAY})\b)?[\s,]*(?:(\d{{1,2}})\s+({_MONTH})\b)?",
                      re.IGNORECASE)
# "l'8 luglio (giovedì)": the weekday after the date (2 of 40 generated answers read by hand).
DATED_AFTER = re.compile(rf"\b(\d{{1,2}})\s+({_MONTH})(?:\s+(\d{{4}}))?\s*\(({_DAY})\)", re.IGNORECASE)
OFFSET = {"oggi": 0, "domani": 1, "dopodomani": 2}


def _nearest(day: int, month: int, year: str | None, now: dt.date) -> dt.date | None:
    """The date of "22 agosto" closest to now (or in the year given)."""
    years = [int(year)] if year else [now.year - 1, now.year, now.year + 1]
    dates = []
    for y in years:
        try:
            dates.append(dt.date(y, month, day))
        except ValueError:
            pass
    return min(dates, key=lambda d: abs((d - now).days)) if dates else None


def date_problems(text: str, now: dt.datetime | dt.date) -> list[str]:
    """A weekday that is not the weekday of its date ("lunedì 22 agosto" on a Sunday), or "domani" next to a
    day that is not tomorrow (seen in generated data): the calendar is computable, so it is checked."""
    today = now.date() if isinstance(now, dt.datetime) else now
    why = []
    named = [(m.group(0), m.group(1), m.group(2), m.group(3), m.group(4)) for m in DATED.finditer(text)]
    named += [(m.group(0), m.group(4), m.group(1), m.group(2), m.group(3)) for m in DATED_AFTER.finditer(text)]
    for said, weekday, day, month, year in named:
        d = _nearest(int(day), P.MONTHS.index(month.lower()) + 1, year, today)
        if d and P.WEEKDAYS[d.weekday()] != weekday.lower():
            why.append(f"«{said}» is a {P.WEEKDAYS[d.weekday()]}")
    for m in RELATIVE.finditer(text):
        want = today + dt.timedelta(days=OFFSET[m.group(1).lower()])
        if m.group(2) and m.group(2).lower() != P.WEEKDAYS[want.weekday()]:
            why.append(f"«{m.group(0).strip()}» is not {P.WEEKDAYS[want.weekday()]}")
        elif m.group(3) and (int(m.group(3)), P.MONTHS.index(m.group(4).lower()) + 1) != (want.day, want.month):
            why.append(f"«{m.group(0).strip()}» is not the {want.day}")
    return why
