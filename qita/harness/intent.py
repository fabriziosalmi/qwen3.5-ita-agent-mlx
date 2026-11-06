"""Intent harness: tell the model the context, the intent of the request and what answer we expect.

Per user turn, two steps:
  A. classify: a short call to the same model, with a closed set of four decisions (the labels follow
     qita/harness/decision.py, which also gives the gold label of eval turns) and a fixed output line
     (`INTENTO: <label>`), parsed deterministically; anything unparsable becomes `generico`.
  B. execute: the user message reaches the model wrapped in a fixed header
     [CONTESTO] (date, home, tools, one line each), [INTENTO], [RISPOSTA ATTESA] (the rule for that
     intent, from the safety and style policy in README.md), then [RICHIESTA] with the text.
The intent is always the model's own: never taken from task metadata, so the eval stays fair.
"""

from __future__ import annotations

import datetime as dt
import re

INTENTS = {
    "rispondi": ("saluti, ringraziamenti, domande generali, spiegazioni, traduzioni, oppure richieste che nessuno "
                 "strumento può fare o che non sono sicure: non servono i dati dell'utente",
                 "Non usare strumenti. Rispondi in una o due frasi; se la richiesta non si può fare o non è sicura, dillo in breve."),
    "strumenti": ("serve leggere, cercare, calcolare o fare qualcosa sui dati dell'utente (file, note, calendario, "
                  "web, terminale, certificati, git), anche riferendosi ai turni precedenti",
                  "Usa gli strumenti, non inventare dati. Se il bersaglio è nominato o chiaro dai turni precedenti, agisci. "
                  "Poi rispondi in una o due frasi con il risultato."),
    "chiedi": ("serve fare qualcosa ma manca un dato necessario: giorno, ora, importo, quale oggetto, quale dominio",
               "Non indovinare e non modificare niente. Chiedi il dato che manca con una domanda breve."),
    "conferma": ("chiede di cancellare o cambiare molte cose insieme, o in modo generico (tutte le foto, tutte le note, "
                 "la cartella intera)",
                 "Non eseguire ancora. Cerca cosa verrebbe toccato, elencalo in breve e chiedi conferma."),
}
WRITE_RULE = ("Non usare strumenti. Se ti chiede di scrivere un testo (un messaggio, una mail, un avviso), scrivilo "
              "per intero. Altrimenti rispondi in una o due frasi; se la richiesta non si può fare o non è sicura, dillo in breve.")
FALLBACK = "generico"
FALLBACK_RULE = "Usa gli strumenti solo se servono, non inventare dati, rispondi in una o due frasi."
ALWAYS = "Se uno strumento dà errore, correggi gli argomenti e riprova una volta, oppure spiega il problema."

CLASSIFY = """Classifica l'ultima richiesta dell'utente con una sola etichetta.
Etichette:
{labels}
{context}Ultima richiesta: {text}
Rispondi solo con una riga: INTENTO: <etichetta>"""


TOOLS_RULE = ("Se uno di questi strumenti può leggere, cercare o cambiare i dati di cui parla la richiesta, "
              "l'etichetta non è rispondi.")


def tool_lines(tools: list[dict], width: int = 90) -> str:
    """One line per tool: its name and the first sentence of its description, cut at width characters."""
    out = []
    for t in tools:
        f = t["function"]
        first = re.split(r"(?<=[.!?])\s|\n", (f.get("description") or "").strip(), maxsplit=1)[0]
        out.append(f"- {f['name']}" + (f": {first[:width]}" if first else ""))
    return "\n".join(out)


def classify_prompt(text: str, previous: list[tuple[str, str]], tools: list[dict] | None = None) -> str:
    """tools (harness flag "toolcls"): the connected tools and TOOLS_RULE go before the request; without them the
    prompt is the one the released models were trained on, byte for byte."""
    labels = "\n".join(f"- {k}: {d}" for k, (d, _) in INTENTS.items())
    ctx = ""
    if previous:
        ctx = "Turni precedenti:\n" + "\n".join(f"{who}: {msg[:300]}" for who, msg in previous[-4:]) + "\n"
    if tools:
        ctx = f"Strumenti collegati:\n{tool_lines(tools)}\n{TOOLS_RULE}\n" + ctx
    return CLASSIFY.format(labels=labels, context=ctx, text=text)


def parse_intent(text: str) -> str:
    m = re.search(r"intento:\s*([a-z_]+)", re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower())
    return m.group(1) if m and m.group(1) in INTENTS else FALLBACK


def context_line(system: str | None, tools: list[dict]) -> str:
    names = ", ".join(t["function"]["name"] for t in tools) or "nessuno"
    head = system.strip().replace("\n", " ") if system else "nessun contesto di sistema"
    return f"{head}\nStrumenti disponibili: {names}."


WEEKDAYS = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")


def calendar_line(now: dt.datetime, days: int = 7, v: int = 1, week: bool = False) -> str:
    """The next days with their weekday and ISO date, and the last day of the month, computed from now.

    v2 starts from today, names today and tomorrow, and says the line is a calendar, not the user's data:
    with v1 the model took the first listed day (tomorrow) for today, and the end of the month for a due date.
    week adds the Monday and Sunday of next week: the fine-tuned 2B and 0.8B read "next week" as the days after today.
    """
    day = lambda d: f"{WEEKDAYS[d.weekday()]} {d.isoformat()}"  # noqa: E731
    today = now.date()
    nxt = [today + dt.timedelta(days=i) for i in range(1, days + 1)]
    last = (today.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    if v == 1:
        return f"Prossimi giorni: {', '.join(day(d) for d in nxt)}. Ultimo giorno del mese: {day(last)}."
    line = (f"Calendario (serve solo a calcolare le date, non sono dati dell'utente): oggi {day(today)}; "
            f"domani {day(nxt[0])}; poi {', '.join(day(d) for d in nxt[1:])}; il mese corrente finisce {day(last)}.")
    if week:
        monday = today + dt.timedelta(days=7 - today.weekday())
        line = line[:-1] + f"; la settimana prossima va da {day(monday)} a {day(monday + dt.timedelta(days=6))}."
    return line


MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
WRITTEN_DATE = re.compile(rf"\b\d{{1,2}}\s+(?:{MONTHS})\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b", re.I)
RELATIVE_DAY = re.compile(r"\b(?:luned[iì]|marted[iì]|mercoled[iì]|gioved[iì]|venerd[iì]|sabato|domenica|oggi|domani|"
                          r"dopodomani|ieri|settiman\w*|prossim\w*\s+(?:\d+\s+|sette\s+)?giorni|weekend|"
                          r"fine\s+(?:del\s+)?mese|mese\s+prossimo|tra\s+\d+\s+giorni)\b", re.I)


def wants_calendar(users: list[str]) -> bool:
    """The conversation names a day by weekday or by a relative word, and the last user turn writes no date.

    With the calendar always shown, a listed day replaced a date the user wrote ("giovedì 19 novembre"
    became the Thursday listed as tomorrow) or one read from a file.
    """
    return any(RELATIVE_DAY.search(u) for u in users) and not WRITTEN_DATE.search(users[-1])


def wrap(user_text: str, intent: str, system: str | None, tools: list[dict], calendar: str | None = None,
         write: bool = False) -> str:
    """write: the "rispondi" rule asks for a requested text in full; with "una o due frasi" the fine-tuned 2B and
    0.8B declined to write a short message."""
    rule = WRITE_RULE if write and intent == "rispondi" else INTENTS[intent][1] if intent in INTENTS else FALLBACK_RULE
    context = context_line(system, tools) + (f"\n{calendar}" if calendar else "")
    return (f"[CONTESTO]\n{context}\n[INTENTO]\n{intent}\n"
            f"[RISPOSTA ATTESA]\n{rule} {ALWAYS}\n[RICHIESTA]\n{user_text}")


def classify(backend, user_text: str, previous: list[tuple[str, str]], tools: list[dict] | None = None) -> tuple[str, str]:
    """Returns (intent, raw model output). Uses the backend without tools (tools only listed in the prompt)."""
    out = backend.complete([{"role": "user", "content": classify_prompt(user_text, previous, tools)}], [])
    return parse_intent(out["text"]), out["text"]
