"""Personal notes server, Italian tool names and descriptions."""

from __future__ import annotations

from .base import Env, ToolError

SEED = [
    {"id": 1, "testo": "Chiamare l'idraulico per la caldaia che perde", "etichette": ["casa"], "data": "2026-10-30"},
    {"id": 2, "testo": "Rinnovare l'assicurazione dell'auto entro il 15 novembre", "etichette": ["auto", "scadenze"], "data": "2026-10-28"},
    {"id": 3, "testo": "Libri da leggere: Il nome della rosa, Le città invisibili, Lessico famigliare", "etichette": ["libri"], "data": "2026-10-20"},
    {"id": 4, "testo": "Codice cliente della bolletta del gas: 4471-882", "etichette": ["casa", "bollette"], "data": "2026-09-14"},
    {"id": 5, "testo": "Compleanno di Sara il 12 novembre: prenotare la trattoria", "etichette": ["compleanni"], "data": "2026-11-01"},
    {"id": 6, "testo": "Portare la bici dal meccanico, freno posteriore", "etichette": ["bici"], "data": "2026-11-02"},
]


def fmt(n: dict) -> str:
    tags = ", ".join(n["etichette"]) or "nessuna"
    return f"#{n['id']} [{n['data']}] {n['testo']} (etichette: {tags})"


class NotesEnv(Env):
    name = "notes"
    title = "note"

    def build(self) -> None:
        seed = SEED if self.ws.world.notes is None else self.ws.world.notes
        self.notes = [dict(n, etichette=list(n["etichette"])) for n in seed]
        today = self.ws.world.now.date().isoformat()
        s = self.server

        def get(nid: int) -> dict:
            for n in self.notes:
                if n["id"] == nid:
                    return n
            raise ToolError(f"Nota #{nid} non trovata")

        @s.tool()
        def aggiungi_nota(testo: str, etichette: list[str] | None = None) -> str:
            """Aggiunge una nuova nota personale, con etichette facoltative."""
            if not testo.strip():
                raise ToolError("Il testo della nota è vuoto")
            nid = max((n["id"] for n in self.notes), default=0) + 1
            n = {"id": nid, "testo": testo.strip(), "etichette": list(etichette or []), "data": today}
            self.notes.append(n)
            return f"Nota creata: {fmt(n)}"

        @s.tool()
        def cerca_note(testo: str) -> str:
            """Cerca tra le note quelle che contengono il testo indicato (anche nelle etichette)."""
            q = testo.casefold()
            hits = [fmt(n) for n in self.notes if q in n["testo"].casefold() or any(q in t for t in n["etichette"])]
            return "\n".join(hits) or "Nessuna nota trovata"

        @s.tool()
        def elenca_note(etichetta: str | None = None) -> str:
            """Elenca tutte le note, oppure solo quelle con un'etichetta."""
            ns = [n for n in self.notes if etichetta is None or etichetta.casefold() in n["etichette"]]
            return "\n".join(fmt(n) for n in ns) or "Nessuna nota"

        @s.tool()
        def modifica_nota(id: int, testo: str) -> str:
            """Sostituisce il testo di una nota esistente."""
            n = get(id)
            n["testo"] = testo.strip()
            return f"Nota aggiornata: {fmt(n)}"

        @s.tool()
        def elimina_nota(id: int) -> str:
            """Elimina definitivamente una nota."""
            n = get(id)
            self.notes.remove(n)
            return f"Nota #{id} eliminata"

    def check(self, kind: str, spec: dict) -> bool | None:
        if kind == "note_exists":
            want = [t.casefold() for t in spec["text"]]
            return any(all(w in n["testo"].casefold() for w in want) for n in self.notes)
        if kind == "note_absent":
            return all(n["id"] != spec["id"] for n in self.notes)
        if kind == "note_count":
            return len(self.notes) == spec["n"]
        return None
