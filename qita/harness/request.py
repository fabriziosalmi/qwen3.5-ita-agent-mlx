"""What a user's turn asks for, read from its words: the sentences that ask for a change and the condition the
change depends on. Rules over the text, no parser: on the 135 dev and dogfood turns the small Italian spaCy
model, even with a morph-it lexicon guard, tags short imperatives ("Fissami", "Elimina", "Rinomina") as
other parts of speech or as indicatives ("Sposta", "Crea"), and attaches "Se" to the wrong word."""

from __future__ import annotations

import re

KEEP = ("mett|fiss|spost|segn|annot|aggiung|scriv|cre|rinomin|modific|cambi|prenot|inser|salv|ricord|registr|"
        "impost|corregg|sostitu|copi")
DROP = "cancell|elimin|rimuov|togli|svuot"  # a request to remove: the harness never presses for it
KEEP_IMP = ("metti|fissa|sposta|segna|annota|aggiungi|scrivi|crea|rinomina|modifica|cambia|prenota|inserisci|salva|"
            "ricorda|registra|imposta|correggi|sostituisci|copia")
DROP_IMP = "cancella|elimina|rimuovi|togli|svuota"
CLITIC = r"(?:mi|ti|ci|vi|lo|la|li|le|ne|gli|(?:me|te|ce|ve|glie)(?:lo|la|li|le|ne))"
LEAD = r"(?:(?:e|poi|anche|ora|adesso|quindi|allora|pure|per favore)\s+)*"


def change_re(imp: str, stems: str) -> re.Pattern:
    """An imperative at the start of a clause or after "e"/"poi" ("Sposta", "e aggiungi"), one with a pronoun
    attached anywhere ("La riunione spostala"), or "puoi"/"potresti" with an infinitive."""
    return re.compile(rf"(?:(?:^|[,;:]\s*|\b(?:e|poi)\s+){LEAD}(?:{imp}){CLITIC}?\b|\b(?:{imp}){CLITIC}\b"
                      rf"|\b(?:puoi|potresti|riesci\s+a)\s+(?:{stems})[aei]re?{CLITIC}?\b)", re.IGNORECASE)


CHANGE = change_re(KEEP_IMP, KEEP)
REMOVE = change_re(DROP_IMP, DROP)
CONDITION = re.compile(r"(?:^|[,;:]\s*)(?:e\s+|ma\s+)?se\s+[^,.;:?!]+", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


def changes(text: str) -> list[str]:
    """The sentences of a turn that ask for a change ("Mettimi in calendario l'anagrafe giovedì alle 14."),
    leaving out any sentence that asks to remove something."""
    return [s for s in sentences(text) if CHANGE.search(s) and not REMOVE.search(s)]


def condition(text: str) -> str | None:
    """The condition a change in this turn depends on ("Se piove"), when a sentence opens with one and also asks
    for a change; None otherwise. "Controlla se ..." asks a question and is not a condition. A bare "Se sì" comes
    with the sentence before it, the question it answers."""
    said = sentences(text)
    for i, s in enumerate(said):
        m = CONDITION.search(s)
        if m and (CHANGE.search(s[m.end():]) or REMOVE.search(s[m.end():])):
            cond = m.group(0).lstrip(",;: ").strip()
            if len(cond.split()) <= 2 and i and m.start() == 0:  # "Se sì": the question it answers comes before
                cond = f"{said[i - 1]} {cond}"
            return cond
    return None
