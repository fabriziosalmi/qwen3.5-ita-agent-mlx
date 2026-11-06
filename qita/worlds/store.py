"""Record stores with a tool vocabulary of their own: the server the model has never seen.

Every world draws a domain (records, crates of vinyl, plants...), its records, and the shape of its server:
the noun and verbs of the tool names, their style (memo_find, findMemos, memo-find, find_memos), the names of
the key and query parameters, keys as slugs or numbers, JSON or line output, MCP annotations or none, and the
English descriptions. Plain data, no prose; the env (qita/envs/store.py) serves it.

The domains are everyday collections chosen to be unlike the servers kept for testing (no assistant memory,
no registry of repositories or hosts, no code index). HELD_OUT_DOMAINS are never
drawn for training data (qita.worlds.sample refuses them): a fresh probe on them measures transfer to a
collection and a vocabulary never trained on.
"""

from __future__ import annotations

import datetime as dt
import random
import re
import unicodedata

from . import pools as P

# op -> verbs a server may use for it. The names of a world take one verb per op.
VERBS = {
    "search": ["search", "find", "query", "lookup"],
    "list": ["list", "browse", "all", "index"],
    "get": ["get", "fetch", "show", "read"],
    "add": ["add", "create", "insert", "new"],
    "update": ["update", "edit", "set", "patch"],
    "delete": ["delete", "remove", "drop", "discard"],
}
STYLES = ["noun_verb", "verb_noun", "camel", "kebab"]
DESCRIPTIONS = {
    "search": ["Search {plural} by text. Returns every {noun} whose fields contain the words.",
               "Full-text search over the {coll}.", "Find {plural} matching a query."],
    "list": ["List all {plural} in the {coll}.", "Return every {noun}, optionally only the first N.",
             "Browse the whole {coll}."],
    "get": ["Get one {noun} by its {key}.", "Fetch a single {noun}.", "Show every field of one {noun}."],
    "add": ["Add a new {noun} to the {coll}.", "Create a {noun}. Only {name} is required.",
            "Insert a {noun} with the given fields."],
    "update": ["Change one field of a {noun}.", "Update a field of an existing {noun}: give the field and the new value.",
               "Set a field of a {noun} to a new value."],
    "delete": ["Delete a {noun} permanently.", "Remove one {noun} from the {coll}. This cannot be undone.",
               "Permanently remove a {noun}."],
}
KEY_PARAMS = ["key", "slug", "name_id", "ref"]
QUERY_PARAMS = ["query", "q", "text", "term"]
NO_ONE = "nessuno"


def slug(text: str) -> str:
    t = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def _date(r, today, lo, hi):
    return (today + dt.timedelta(days=r.randrange(lo, hi))).isoformat()


# Each domain: Italian words for the requests, English words for the server, the name field (what a user calls
# a record) with its pool, and the other fields with a value drawer. kind: int, date, text.
def _people(r):
    return r.choice([NO_ONE, NO_ONE] + P.FIRST)


DOMAINS = {
    "dischi": dict(
        it=("disco", "dischi", "m"), nouns=["record", "vinyl", "disc"], colls=["crate", "record collection"],
        name="title", pool=["Notte elettrica", "Basso continuo", "Rave in cascina", "Onde lunghe", "Acido lattico",
                            "Fabbrica vuota", "Corrente alternata", "Segnale debole", "Vento di levante",
                            "Ritmo meccanico", "Luna di ferro", "Cassa dritta", "Città sommersa", "Treno di notte",
                            "Polvere di stelle", "Ultimo binario"],
        fields={"artist": ("text", lambda r, t: r.choice(["Collettivo Ombra", "Marea Nera", "Sistema Sonoro",
                                                           "Frequenze Basse", "Officina 23", "Nebbia Fitta",
                                                           "Circuito Chiuso", "Banda del Porto"])),
                "year": ("int", lambda r, t: r.randrange(1988, 2025)),
                "bpm": ("int", lambda r, t: r.randrange(118, 176)),
                "genre": ("text", lambda r, t: r.choice(["techno", "acid", "hardcore", "house", "dub", "electro"])),
                "shelf": ("text", lambda r, t: f"{r.choice('ABCD')}{r.randrange(1, 5)}")}),
    "piante": dict(
        it=("pianta", "piante", "f"), nouns=["plant", "green"], colls=["plant list", "garden"],
        name="name", pool=["Monstera grande", "Ficus del corridoio", "Basilico", "Pothos", "Sansevieria", "Aloe",
                           "Orchidea bianca", "Limone", "Rosmarino", "Felce", "Gelsomino", "Kenzia", "Salvia",
                           "Begonia"],
        fields={"room": ("text", lambda r, t: r.choice(["salotto", "cucina", "balcone", "camera", "bagno", "studio"])),
                "water_every_days": ("int", lambda r, t: r.randrange(2, 15)),
                "last_watered": ("date", lambda r, t: _date(r, t, -14, 0)),
                "light": ("text", lambda r, t: r.choice(["sole pieno", "mezz'ombra", "ombra"]))}),
    "abbonamenti": dict(
        it=("abbonamento", "abbonamenti", "m"), nouns=["subscription", "sub", "plan"], colls=["subscriptions"],
        name="service", pool=["Palestra", "Streaming film", "Musica in streaming", "Giornale online", "Cloud foto",
                              "Trasporti urbani", "Rivista di cucina", "Assicurazione bici", "Corso di inglese",
                              "Parcheggio", "App meteo", "Car sharing"],
        fields={"price_eur": ("money", lambda r, t: f"{r.randrange(2, 60)}.{r.choice(['00', '50', '90', '99'])}"),
                "renews_on": ("date", lambda r, t: _date(r, t, 1, 330)),
                "billing": ("text", lambda r, t: r.choice(["mensile", "annuale"])),
                "payment": ("text", lambda r, t: r.choice(["carta", "bonifico", "addebito sul conto"]))}),
    "libri": dict(
        it=("libro", "libri", "m"), nouns=["book", "title"], colls=["library", "bookshelf"],
        name="title", pool=["Il giardino d'inverno", "La casa sul lago", "Storia del porto", "Le strade di sale",
                            "Il tempo dei fichi", "Un'estate a Camogli", "La biblioteca di vetro", "Il faro spento",
                            "Mappe del vento", "La sarta di Genova", "Il ponte dei sospiri", "Lettere dal fronte",
                            "Il pane e le rose", "Neve a maggio"],
        fields={"author": ("text", lambda r, t: f"{r.choice(P.FIRST)} {r.choice(P.LAST)}"),
                "lent_to": ("text", lambda r, t: _people(r)),
                "shelf": ("text", lambda r, t: f"{r.choice(['soggiorno', 'studio', 'camera'])} {r.randrange(1, 6)}"),
                "read": ("text", lambda r, t: r.choice(["sì", "no"]))}),
    "dispensa": dict(
        it=("prodotto", "prodotti", "m"), nouns=["item", "stock_item", "product"], colls=["pantry", "stock"],
        name="item", pool=["farina", "riso", "passata", "olio", "lenticchie", "caffè", "zucchero", "tonno",
                           "pasta corta", "ceci", "latte", "uova", "biscotti", "piselli surgelati"],
        fields={"qty": ("int", lambda r, t: r.randrange(1, 11)),
                "unit": ("text", lambda r, t: r.choice(["pacchi", "barattoli", "bottiglie", "scatole"])),
                "location": ("text", lambda r, t: r.choice(["dispensa", "frigo", "freezer", "cantina"])),
                "expires_on": ("date", lambda r, t: _date(r, t, -10, 300))}),
    "bici": dict(
        it=("componente", "componenti", "m"), nouns=["part", "component"], colls=["bike log", "maintenance log"],
        name="part", pool=["catena", "pastiglie dei freni", "copertone anteriore", "copertone posteriore",
                           "cassetta", "cavi del cambio", "nastro del manubrio", "pedali", "luce posteriore",
                           "camera d'aria", "sella"],
        fields={"last_km": ("int", lambda r, t: r.randrange(0, 12000)),
                "every_km": ("int", lambda r, t: r.choice([1000, 2000, 3000, 5000, 8000])),
                "bike": ("text", lambda r, t: r.choice(["corsa", "città", "mountain bike"])),
                "shop": ("text", lambda r, t: f"Ciclofficina {r.choice(P.LAST)}")}),
    "gatti": dict(
        it=("trattamento", "trattamenti", "m"), nouns=["treatment", "care_item", "vet_record"], colls=["vet records"],
        name="name", pool=[f"{a} di {c}" for a in ["vaccino", "antipulci", "sverminazione", "controllo denti"]
                           for c in ["Tigre", "Nuvola", "Pepe", "Briciola"]],
        fields={"last_date": ("date", lambda r, t: _date(r, t, -200, -1)),
                "next_due": ("date", lambda r, t: _date(r, t, 1, 200)),
                "vet": ("text", lambda r, t: f"dottoressa {r.choice(P.LAST)}"),
                "cost_eur": ("int", lambda r, t: r.randrange(15, 120))}),
    "ricette": dict(
        it=("ricetta", "ricette", "f"), nouns=["recipe", "dish"], colls=["cookbook", "recipes"],
        name="dish", pool=["pesto", "focaccia", "minestrone", "torta di riso", "farinata", "pansoti", "polpette",
                           "trofie al sugo", "frittata di zucchine", "torta di mele", "zuppa di ceci", "acciughe ripiene"],
        fields={"minutes": ("int", lambda r, t: r.choice([15, 20, 30, 40, 45, 60, 90])),
                "servings": ("int", lambda r, t: r.randrange(2, 9)),
                "course": ("text", lambda r, t: r.choice(["primo", "secondo", "dolce", "antipasto"])),
                "last_made": ("date", lambda r, t: _date(r, t, -120, 0))}),
    # held out: never in training data
    "cantina": dict(
        it=("vino", "vini", "m"), nouns=["wine", "bottle"], colls=["cellar"],
        name="wine", pool=["Rossese", "Vermentino", "Pigato", "Barbera", "Dolcetto", "Nebbiolo", "Ormeasco",
                           "Ciliegiolo", "Granaccia", "Lumassina", "Bianchetta"],
        fields={"producer": ("text", lambda r, t: f"Cantina {r.choice(P.LAST)}"),
                "year": ("int", lambda r, t: r.randrange(2012, 2025)),
                "qty": ("int", lambda r, t: r.randrange(1, 13)),
                "rack": ("text", lambda r, t: f"R{r.randrange(1, 7)}")}),
    "attrezzi": dict(
        it=("attrezzo", "attrezzi", "m"), nouns=["tool", "gear"], colls=["toolbox", "garage"],
        name="tool", pool=["trapano", "scala", "avvitatore", "sega", "livella", "metro laser", "pistola per colla",
                           "smerigliatrice", "pinza", "tenda da campeggio", "compressore"],
        fields={"location": ("text", lambda r, t: r.choice(["garage", "cantina", "ripostiglio", "soffitta"])),
                "lent_to": ("text", lambda r, t: _people(r)),
                "bought": ("int", lambda r, t: r.randrange(2005, 2026))}),
}
HELD_OUT_DOMAINS = {"cantina", "attrezzi"}
TRAIN_DOMAINS = [d for d in DOMAINS if d not in HELD_OUT_DOMAINS]


def tool_name(style: str, noun: str, verb: str, op: str) -> str:
    many = op in ("search", "list")
    plural = noun + ("" if noun.endswith("s") else "s")
    if style == "noun_verb":
        return f"{noun}_{verb}"
    if style == "verb_noun":
        return f"{verb}_{plural if many else noun}"
    if style == "camel":
        cap = "".join(w.capitalize() for w in (plural if many else noun).split("_"))
        return f"{verb}{cap}"
    return f"{noun.replace('_', '-')}-{verb}"


def build_store(seed: int, held_out: bool = False, domain: str | None = None, today: dt.date | None = None) -> dict:
    """The store of one world, from a stream of its own (the rest of the world is unchanged)."""
    r = random.Random(f"store-{seed}")
    name = domain or r.choice(sorted(HELD_OUT_DOMAINS) if held_out else TRAIN_DOMAINS)
    d = DOMAINS[name]
    today = today or dt.date(2026, 11, 4)
    noun = r.choice(d["nouns"])
    style = r.choice(STYLES)
    verbs = {op: r.choice(v) for op, v in VERBS.items()}
    if verbs["list"] == "all" and style in ("noun_verb", "kebab"):
        verbs["list"] = "list"  # "record_all" reads as a bulk action
    tools = {op: tool_name(style, noun, verbs[op], op) for op in VERBS}
    numeric = r.random() < 0.25
    key = "id" if numeric else r.choice(KEY_PARAMS)
    query = r.choice(QUERY_PARAMS)
    coll = r.choice(d["colls"])
    plural = noun.replace("_", " ") + ("" if noun.endswith("s") else "s")
    words = {"noun": noun.replace("_", " "), "plural": plural, "coll": coll, "key": key, "name": d["name"]}
    desc = {op: r.choice(v).format(**words) for op, v in DESCRIPTIONS.items()}
    n = r.randrange(5, min(11, len(d["pool"])))
    names = r.sample(d["pool"], n)
    records = []
    for i, nm in enumerate(names):
        rec = {key: i + 1 if numeric else slug(nm), d["name"]: nm}
        for f, (_kind, draw) in d["fields"].items():
            rec[f] = draw(r, today)
        records.append(rec)
    missing = r.choice([x for x in d["pool"] if x not in names] or ["Senza titolo"])
    return {"domain": name, "it": d["it"], "name": d["name"], "key": key, "query": query, "numeric": numeric,
            "fields": {f: k for f, (k, _draw) in d["fields"].items()}, "tools": tools, "desc": desc,
            "fmt": r.choice(["json", "json", "lines"]), "annotate": r.random() < 0.5, "coll": coll,
            "records": records, "missing": missing}


def attach_store(world, facts, seed: int, held_out: bool = False) -> dict:
    """Give a built world its store (same seed, own stream): the env serves world.store, families read facts.store."""
    store = build_store(seed, held_out=held_out, today=facts.now.date())
    world.store = facts.store = store
    return store


__all__ = ["DOMAINS", "HELD_OUT_DOMAINS", "TRAIN_DOMAINS", "attach_store", "build_store", "slug", "tool_name"]
