"""The teacher writes the Italian text around a task: user requests and system prompts.

Prompts to the teacher are instructions, not training data. What the teacher returns enters
the data only through the gates below, which check it against the task's structured facts.
"""

from __future__ import annotations

import json
import re

from ..harness.italian import is_italian
from . import pools as P

USER_PROMPT = """Scrivi la frase che un utente italiano scrive al suo assistente nel terminale.
Situazione: {situazione}
{requisiti}Scrivi come scrive una persona vera: frase breve, tono {tono}. Niente virgolette, niente parentesi, niente spiegazioni.
Rispondi solo con la frase."""
TONES = ["informale", "diretto", "gentile", "sbrigativo", "colloquiale", "neutro"]

SYSTEM_PROMPT = """Scrivi il messaggio di sistema per un assistente che lavora nel terminale di un utente e può usare strumenti.
Deve dire, con parole tue: che oggi è {data}, che sono le {ora}, fuso orario Europe/Rome, e che la cartella home dell'utente è /home/utente.
{extra}Massimo tre frasi, in italiano. Rispondi solo con il messaggio."""
SYSTEM_EXTRA = ["", "Aggiungi che deve rispondere in modo breve. ", "Aggiungi che deve chiedere conferma prima di cancellare cose. ",
                "Aggiungi che l'utente si chiama {nome}. "]


def situation(intent: dict) -> str:
    parts = [intent["azione"]]
    for k, v in intent.items():
        if k != "azione" and v not in (None, "", []):
            parts.append(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}")
    return "; ".join(parts)


def user_prompt(intent: dict, must_say: list[str], tone: str, context: str = "", must_say_any: list[str] = ()) -> str:
    req = ""
    if must_say:
        req += "La frase deve contenere esattamente questi elementi, scritti così: " + ", ".join(f"«{m}»" for m in must_say) + "\n"
    if must_say_any:
        req += "La frase deve contenere una di queste espressioni: " + ", ".join(f"«{m}»" for m in must_say_any) + "\n"
    p = USER_PROMPT.format(situazione=situation(intent), requisiti=req, tono=tone)
    return (f"Conversazione finora:\n{context}\n\n" + p) if context else p


def clean(text: str) -> str:
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    text = text.strip().strip("\"«»“”'").strip()
    return text


def gate_user(text: str, must_say: list[str], must_not: list[str], must_say_any: list[str] = ()) -> list[str]:
    """Reasons to reject a user sentence written by the teacher (empty = accepted)."""
    why = []
    low = text.casefold()
    n = len(text.split())
    if "\n" in text:
        why.append("more than one line")
    if not 1 <= n <= 60:
        why.append(f"{n} words")
    if "<tool_call>" in text or "<function=" in text:
        why.append("tool-call markup")
    if "(" in text or ")" in text:
        why.append("parentheses")
    if must_say_any and not any(m.casefold() in low for m in must_say_any):
        why.append(f"none of {must_say_any}")
    if not is_italian(text):
        why.append("not Italian")
    why += [f"missing «{m}»" for m in must_say if m.casefold() not in low]
    why += [f"leaks «{m}»" for m in must_not if m.casefold() in low]
    if low.startswith(("ecco", "certo", "ok, ecco")):
        why.append("sounds like an assistant")
    return why


def system_prompt_request(now, r, extras: list[str] | None = None) -> tuple[str, list[str]]:
    """The teacher prompt for a system message, and the strings the result must contain."""
    data = f"{P.WEEKDAYS[now.weekday()]} {now.day} {P.MONTHS[now.month - 1]} {now.year}"
    extra = r.choice(extras or SYSTEM_EXTRA).format(nome=r.choice(P.FIRST))
    must = [str(now.day), P.MONTHS[now.month - 1], str(now.year), f"{now:%H:%M}".lstrip("0") if now.hour < 10 else f"{now:%H:%M}", "/home/utente"]
    return SYSTEM_PROMPT.format(data=data, ora=f"{now:%H:%M}", extra=extra), must


def gate_system(text: str, must: list[str]) -> list[str]:
    why = [f"missing «{m}»" for m in must if m.casefold() not in text.casefold()]
    if not is_italian(text):
        why.append("not Italian")
    if len(text) > 600:
        why.append("too long")
    if "<tool_call>" in text:
        why.append("tool-call markup")
    return why


# ---------------- phrasing v2: how long, in what tone, how carefully ----------------

USER_PROMPT_V2 = """Scrivi il messaggio che un utente italiano scrive al suo assistente nel terminale.
Situazione: {situazione}
{requisiti}Il messaggio è rivolto all'assistente: non chiamarlo per nome e non rivolgerti alle persone nominate nella situazione, sono terze persone.
Messaggio {lunghezza}, tono {tono}, in italiano corretto. Niente virgolette, niente parentesi, niente spiegazioni.
Rispondi solo con il messaggio."""
TONES_V2 = TONES + ["frettoloso", "cordiale", "un po' seccato", "scherzoso", "formale"]
LENGTHS = ["breve", "breve", "di una riga", "breve, con un motivo semplice e realistico"]
SYSTEM_EXTRA_V2 = SYSTEM_EXTRA + ["Aggiungi che deve usare gli strumenti solo quando servono. ",
                                  "Aggiungi che deve rispondere in italiano semplice. ",
                                  "Aggiungi che lavora per un piccolo ufficio. ", "Aggiungi che non deve inventare dati. ",
                                  "Aggiungi che deve dare del tu all'utente. "]
ACCENTS = {"è": ["e'", "e"], "à": ["a'", "a"], "ò": ["o'", "o"], "ù": ["u'", "u"], "ì": ["i'", "i"], "é": ["e'", "e"]}


# A sentence that opens by calling someone by name ("Marta, ...", "Ciao Anna, ...") talks to a third
# person or names the assistant: the request no longer means what the task means.
VOCATIVE = re.compile(r"^(?:(?i:ciao|ehi|senti|scusa|oh)[,!]?\s+)?([A-ZÀ-Ú][a-zà-ù']+)\s*[,!]")
# "Cari ufficio postale di Cagliari, ..." (seen in generated data): the writer talks to the office, not to the assistant.
ADDRESS = re.compile(r"(?i)^(?:car[oaie]|gentil[ei]|egregi[oa]|spettabile)\s+(?!assistente\b)")
OPENERS = {"ciao", "ehi", "senti", "scusa", "allora", "ok", "okay", "grazie", "buongiorno", "buonasera", "salve",
           "perfetto", "bene", "sì", "no", "ascolta", "dai", "aspetta", "assistente", "per", "ecco"}


def gate_user_v2(text: str, must_say: list[str], must_not: list[str], must_say_any: list[str] = ()) -> list[str]:
    why = gate_user(text, must_say, must_not, must_say_any)
    m = VOCATIVE.match(text)
    if m and m.group(1).casefold() not in OPENERS:
        why.append(f"addresses «{m.group(1)}»")
    if ADDRESS.match(text):
        why.append("addresses someone")
    if len(text.split()) < 3:
        why.append("fewer than 3 words")
    return why


def user_prompt_v2(intent: dict, must_say: list[str], r, context: str = "", must_say_any: list[str] = ()) -> str:
    req = ""
    if must_say:
        req += "La frase deve contenere esattamente questi elementi, scritti così: " + ", ".join(f"«{m}»" for m in must_say) + "\n"
    if must_say_any:
        req += "La frase deve contenere una di queste espressioni: " + ", ".join(f"«{m}»" for m in must_say_any) + "\n"
    p = USER_PROMPT_V2.format(situazione=situation(intent), requisiti=req,
                              lunghezza=r.choice(LENGTHS), tono=r.choice(TONES_V2))
    return (f"Conversazione finora:\n{context}\n\n" + p) if context else p


def roughen(text: str, r, protect: list[str]) -> str:
    """Typing in a hurry: lowercase start, no final stop, accents as apostrophes, two short words fused,
    two letters swapped. Words of the protected strings (what the task needs said) are never touched."""
    keep = {w.strip(".,;:!?").casefold() for p in protect for w in p.split()}
    words = text.split(" ")
    free = [i for i, w in enumerate(words) if w.strip(".,;:!?").casefold() not in keep]
    for op in r.sample(["lower", "stop", "accents", "fuse", "swap"], r.choice([1, 2, 2, 3])):
        if op == "lower" and words[0] and 0 in free:
            words[0] = words[0][0].lower() + words[0][1:]
        elif op == "stop":
            words[-1] = words[-1].rstrip(".!") if len(free) and len(words) - 1 in free else words[-1]
        elif op == "accents":
            for i in free:
                for a, subs in ACCENTS.items():
                    words[i] = words[i].replace(a, r.choice(subs))
        elif op == "fuse":
            pairs = [i for i in free if i + 1 in free and len(words[i]) <= 4 and len(words[i + 1]) <= 5]
            if pairs:
                i = r.choice(pairs)
                words[i:i + 2] = [words[i] + words[i + 1]]
                free = [j if j < i else j - 1 for j in free if j != i + 1]
        elif op == "swap":
            long_ = [i for i in free if len(words[i]) >= 6 and words[i].isalpha()]
            if long_:
                i = r.choice(long_)
                k = r.randrange(1, len(words[i]) - 2)
                w = words[i]
                words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(words)
