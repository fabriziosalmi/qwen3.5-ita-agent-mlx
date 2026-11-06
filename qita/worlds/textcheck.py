"""Checks of generated conversations that need a lexicon or a parser, run after generation by
scripts/textcheck.py:

  english   an English word in an agent answer ("Event creato con successo"): a word that is not an Italian
            form (morph-it), is an English word (/usr/share/dict/words), was not written by the user or a tool
            and is not a loanword Italian uses; agent conversations only, because chats about technology,
            books or music name things in English on purpose
  grammar   a CLM rule from GRAMMAR_RULES fires on an assistant message ("Gli backup", "un ora"); the other
            CLM rules fired on correct answers (markdown bold, bare numbers, "chiuso domenica") and are not used
  dates     a weekday that is not the weekday of its date (style.date_problems), today read from the calendar
            line of the context, unless the user or a tool wrote it (a notice of another year); generation checks
            it too, this pass catches rows made with older code
  backup    a log_latest answer that calls a failed backup successful, or a successful one failed (round-6
            checkpoint: "L'ultimo backup è andato a buon fine il 21 settembre (sebbene con errore di permesso)")

The lexicons and the parser are passed in, so the tests run without them.
"""

from __future__ import annotations

import datetime as dt
import re

from qita.harness.dates import named_days

from .style import date_problems

# Measured on 517 agent answers (generated training data): "event" was the only English word flagged, with
# log, client, root and directory, which Italian uses as they are; the rest are loanwords of the same kind.
LOANWORDS = {"backup", "log", "client", "root", "directory", "server", "password", "link", "email", "mail", "online",
             "database", "browser", "desktop", "download", "upload", "account", "login", "cloud", "script", "shell",
             "file", "computer", "software", "hardware", "web", "smartphone", "app", "chat", "ok", "budget",
             "check", "report", "team", "meeting", "standard", "privacy", "spam", "bug", "default", "home"}
# Measured on 480 assistant messages (generated training data): each fired only on a real error.
GRAMMAR_RULES = {"L1.14": "'gli' before a simple consonant", "L16.02": "noun gender from the dictionary"}
_MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
# "il 8 marzo", "del 11 aprile", "dal 2 al 8 giugno": 30 times in 4,020 answers of rounds 1-5, never right; only
# before a month, so a date copied from a file ("fattura del 11/12/2026") stays as written.
ELISION = re.compile(rf"\b(il|del|al|dal|nel|sul)\s+(8|11)\b(?=(?:\s+(?:e|al|-)\s+(?:il\s+)?\d{{1,2}})?\s+(?:{_MONTHS})\b)",
                     re.IGNORECASE)
_ELIDED = {"il": "l'", "del": "dell'", "al": "all'", "dal": "dall'", "nel": "nell'", "sul": "sull'"}
# "Ho segnalato la visita" for "segnato" (2 answers, the only two in the generated data).
SEGNALATO = re.compile(r"\b([Ss])egnalat([oaie])(?=\s+(?:la|il|l'|lo|i|le|una|un|in calendario)\b)")
_SKIP = re.compile(r"`[^`]*`|https?://\S+|/\S+|«[^»]*»|“[^”]*”|\"[^\"]*\"|\*[^*]+\*|\S+@\S+|\b[\w-]+\.[a-z]{2,4}\b")
_WORD = re.compile(r"[\wÀ-ÿ']+")
TODAY = re.compile(r"\boggi \w+ (\d{4})-(\d{2})-(\d{2})\b")
LOGLINE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2}) \d{2}:\d{2}:\d{2} (?:INFO backup (completato)|(ERRORE))")
SUCCESS = re.compile(r"buon fine|con successo|andat[oi] bene|senza errori", re.IGNORECASE)
FAILURE = re.compile(r"fallit|errore|non è andato|non (?:si è |è stato )?completat", re.IGNORECASE)
_NEGATED = re.compile(r"\bnon\s+(?:[\wÀ-ÿ']+\s+){0,3}$", re.IGNORECASE)  # "non è stato completato con successo"


def fix_wording(text: str) -> str:
    """"il 8 marzo" -> "l'8 marzo", "Ho segnalato la visita" -> "Ho segnato la visita"."""
    def elide(m):
        art = _ELIDED[m.group(1).lower()]
        return (art[0].upper() + art[1:] if m.group(1)[0].isupper() else art) + m.group(2)
    return SEGNALATO.sub(r"\1egnat\2", ELISION.sub(elide, text))


def words(text: str, skip_names: bool = False) -> set[str]:
    """Lowercased words of a text, without code, paths, quotes and italics; skip_names leaves out a capitalized
    word in the middle of a sentence (a name: Python, Queen, Salt Lake City)."""
    out = set()
    for sentence in re.split(r"(?<=[.!?:;])\s+|\n+", _SKIP.sub(" ", text)):
        for k, w in enumerate(_WORD.findall(sentence)):
            if not (skip_names and k > 0 and w[:1].isupper()):
                out.add(w.strip("'").lower())
    return out


def english_words(text: str, context: set[str], italian: set[str], english: set[str]) -> list[str]:
    return sorted(w for w in words(text, skip_names=True)
                  if len(w) >= 3 and w.isalpha() and w not in italian and w in english and w not in context
                  and w not in LOANWORDS)


def backup_problems(msgs: list[dict]) -> list[str]:
    """Clauses of the final answer that give the last or the previous backup the wrong outcome. A clause is
    about a backup when it names its day, or says "ultimo" / "precedente" and has no number (a clause like "i
    backup precedenti, dal 17 al 21" is about other days)."""
    dated = {}
    for m in msgs:
        if m["role"] == "tool":
            for y, mo, d, done, err in LOGLINE.findall(m.get("content") or ""):
                dated[(int(y), int(mo), int(d))] = bool(done)
    if len(dated) < 2:
        return []
    # by date, not by first sight: a tail before a full read would put older days last
    days = [(mo, d) for _, mo, d in sorted(dated)]
    outcome = {(mo, d): ok for (_, mo, d), ok in dated.items()}
    last, prev = days[-1], days[-2]
    answer = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant" and m.get("content")), "")
    why = []
    for clause in re.split(r"(?<=[.;!?])\s+|,\s+mentre\s+", answer):
        named = named_days(clause)
        refs = named & {last, prev}
        if not refs and not re.search(r"\d", clause):
            refs = {prev} if re.search(r"precedent|prima", clause, re.IGNORECASE) else \
                   {last} if re.search(r"ultim", clause, re.IGNORECASE) else set()
        ok = {outcome[r] for r in refs}
        if len(ok) != 1:
            continue
        success = any(not _NEGATED.search(clause[:m.start()]) for m in SUCCESS.finditer(clause))
        if ok == {False} and success or ok == {True} and FAILURE.search(clause) and not success:
            why.append(f"backup outcome: «{clause.strip()}»")
    return why


def problems(row: dict, italian: set[str], english: set[str], grammar) -> list[str]:
    """Reasons to drop a kept row; grammar(texts) returns the CLM rule ids that fired on each text."""
    msgs = row["messages"]
    said = [m["content"] for m in msgs if m["role"] == "assistant" and m.get("content")]
    why = []
    if not row["meta"]["family"].startswith("chat_"):
        context = set()
        for m in msgs:
            if m["role"] != "assistant":
                context |= words(m.get("content") or "")
        why += [f"English «{w}»" for t in said for w in english_words(t, context, italian, english)]
    if m := TODAY.search(" ".join(x.get("content") or "" for x in msgs[:2])):
        today = dt.date(*map(int, m.groups()))
        given = " ".join(x.get("content") or "" for x in msgs if x["role"] != "assistant").lower()
        why += [f"date: {p}" for t in said for p in date_problems(t, today)  # a date copied from a document
                if p[1:p.index("»")].lower() not in given]
    if row["meta"]["family"] == "log_latest":
        why += backup_problems(msgs)
    for ids in grammar(said):
        why += [f"grammar {i}: {GRAMMAR_RULES[i]}" for i in ids if i in GRAMMAR_RULES]
    return why
