"""A random world from a seed: structured facts, and the World the envs serve from them.

Facts are data (amounts, dates, names, lists). Files and pages are rendered from them in a
formulaic, record-like way (key: value lines, lists, CSV), so no prose is authored here.
The task families read the facts to build requests, checks and oracle actions.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from ..envs.base import ROME, World
from ..envs.itcalc import cf_check_char, iban_ok
from . import pools as P

CENT = Decimal("0.01")


def euro(x: Decimal) -> str:
    s = f"{x:,.2f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def it_date(d: dt.date) -> str:
    return f"{P.WEEKDAYS[d.weekday()]} {d.day} {P.MONTHS[d.month - 1]} {d.year}"


def make_iban(r: random.Random) -> str:
    bban = r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + "".join(r.choice("0123456789") for _ in range(22))
    for check in range(2, 99):
        iban = f"IT{check:02d}{bban}"
        if iban_ok(iban):
            return iban
    raise AssertionError("no IBAN check digits")


def make_cf(r: random.Random, first: str, last: str) -> str:
    def letters(s, n):
        cons = [c for c in s.upper() if c.isalpha() and c not in "AEIOU"]
        vow = [c for c in s.upper() if c in "AEIOU"]
        return ("".join(cons + vow) + "XXX")[:n]
    month = "ABCDEHLMPRST"[r.randrange(12)]
    body = f"{letters(last, 3)}{letters(first, 3)}{r.randrange(40, 99):02d}{month}{r.randrange(1, 29):02d}" \
           f"{r.choice('ABCDEFGHLM')}{r.randrange(100, 999):03d}"
    return body + cf_check_char(body)


@dataclass
class Facts:
    now: dt.datetime
    user: tuple[str, str]
    folders: dict[str, str]
    shopping: dict = field(default_factory=dict)
    invoices: list[dict] = field(default_factory=list)
    meeting: dict = field(default_factory=dict)
    backup: dict = field(default_factory=dict)
    sales: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    photos: dict = field(default_factory=dict)
    notes: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    tax_codes: list[tuple[str, bool]] = field(default_factory=list)
    version: int = 1
    contacts: dict = field(default_factory=dict)
    expenses: dict = field(default_factory=dict)
    applog: dict = field(default_factory=dict)
    todo: dict = field(default_factory=dict)
    downloads: list[str] = field(default_factory=list)
    store: dict = field(default_factory=dict)  # qita/worlds/store.py attach_store(), only for store families


def build(seed: int, v: int = 1) -> tuple[World, Facts]:
    """v=1 is the original world, kept byte-identical; v=2 adds files, pages and layouts (extend_v2)."""
    r = random.Random(seed)
    day0 = dt.date(2026, 1, 5) + dt.timedelta(days=r.randrange(0, 1080))
    now = dt.datetime(day0.year, day0.month, day0.day, r.randrange(8, 20), r.choice([0, 15, 30, 45]), tzinfo=ROME)
    today = now.date()
    user = (r.choice(P.FIRST), r.choice(P.LAST))
    folders = {k: r.choice(v) for k, v in P.FOLDERS.items()}
    people = r.sample(P.FIRST, 6)
    f = Facts(now=now, user=user, folders=folders, people=people)
    home: dict[str, str] = {}

    # shopping list
    items = r.sample(P.GROCERIES, r.randrange(4, 9))
    path = r.choice(["spesa.txt", "lista-spesa.txt", f"{folders['notes']}/spesa.txt", "da-comprare.txt"])
    f.shopping = {"path": path, "items": items}
    home[path] = "\n".join(items) + "\n"

    # invoices: some with a wrong VAT line on purpose
    for i in range(r.randrange(2, 4)):  # at least two, so 'the second one' always exists
        imp = Decimal(r.randrange(80, 4000)) + Decimal(r.choice([0, 0, 50, 90])) / 100
        rate = r.choice(P.VAT_RATES)
        iva = (imp * rate / 100).quantize(CENT, ROUND_HALF_UP)
        wrong = r.random() < 0.3
        shown_iva = iva + Decimal(r.choice([-10, 10, 25, -5])) if wrong else iva
        issued = today - dt.timedelta(days=r.randrange(3, 60))
        due = issued + dt.timedelta(days=r.choice([30, 30, 60, 90]))
        sup = f"{r.choice(P.SUPPLIERS)} {r.choice(P.LAST)} {r.choice(P.COMPANY_SUFFIX).format(last=r.choice(P.LAST))}"
        inv = {"number": f"{r.randrange(1, 300)}/{issued.year}", "issued": issued, "due": due, "supplier": sup,
               "service": r.choice(P.SERVICES), "imponibile": imp, "rate": rate, "iva": iva, "shown_iva": shown_iva,
               "total": imp + shown_iva, "iban": make_iban(r), "wrong_vat": wrong, "iban_ok": True,
               "path": f"{folders['docs']}/fattura-{issued:%Y-%m}-{i + 1}.txt"}
        inv["iban_shown"] = inv["iban"]
        f.invoices.append(inv)
        home[inv["path"]] = (
            f"FATTURA N. {inv['number']} del {issued:%d/%m/%Y}\nFornitore: {sup}\n"
            f"Cliente: {user[0]} {user[1]}\n\nDescrizione: {inv['service']}\n"
            f"Imponibile: {euro(imp)} EUR\nIVA {rate}%: {euro(shown_iva)} EUR\nTotale: {euro(inv['total'])} EUR\n\n"
            f"Pagamento: bonifico entro il {due:%d/%m/%Y}\nIBAN: {inv['iban']}\n")

    # meeting notes, a few days ago
    mdate = today - dt.timedelta(days=r.randrange(1, 6))
    present = r.sample(people, 4)
    nxt = dt.datetime.combine(today + dt.timedelta(days=r.randrange(2, 15)), dt.time(r.choice([9, 10, 14, 15, 16, 17])))
    actions = [(p, r.choice(["preparare il preventivo", "aggiornare il sito", "chiamare il fornitore",
                             "scrivere il verbale", "prenotare la sala", "controllare il budget"]),
                today + dt.timedelta(days=r.randrange(1, 10))) for p in present[:3]]
    budget = r.randrange(5, 60) * 100
    f.meeting = {"path": f"{folders['notes']}/riunione-{mdate.isoformat()}.md", "date": mdate, "present": present,
                 "actions": actions, "next": nxt, "budget": budget}
    home[f.meeting["path"]] = (
        f"# Riunione del {it_date(mdate)}\n\nPresenti: {', '.join(present)}.\n\n## Decisioni\n"
        f"- Budget approvato: {euro(Decimal(budget))} euro.\n\n## Azioni\n"
        + "".join(f"- {p}: {t} entro {it_date(d)}.\n" for p, t, d in actions)
        + f"\nProssima riunione: {it_date(nxt.date())} alle {nxt:%H:%M}.\n")

    # backup log over the last week
    runs = []
    for k in range(7, 0, -1):
        d = dt.datetime.combine(today - dt.timedelta(days=k), dt.time(2, 0))
        if r.random() < 0.3:
            runs.append({"day": d.date(), "ok": False,
                         "error": r.choice(["spazio insufficiente sul disco di destinazione",
                                            "disco di destinazione non raggiungibile", "permesso negato su /dati"])})
        else:
            runs.append({"day": d.date(), "ok": True, "files": r.randrange(5000, 40000)})
    lines = []
    for x in runs:
        lines.append(f"{x['day']} 02:00:01 INFO backup avviato")
        lines.append(f"{x['day']} 02:{r.randrange(1, 59):02d}:{r.randrange(0, 59):02d} " +
                     (f"INFO backup completato: {x['files']:,} file".replace(",", ".") if x["ok"] else f"ERRORE {x['error']}"))
    f.backup = {"path": f"{folders['logs']}/backup.log", "runs": runs}
    home[f.backup["path"]] = "\n".join(lines) + "\n"

    # sales CSV for three months
    months = [(today.replace(day=1) - dt.timedelta(days=31 * k)).month for k in (3, 2, 1)]
    products = r.sample(["tessera socio", "biglietto evento", "corso", "maglietta", "libro", "abbonamento"], 2)
    rows = []
    for m in months:
        for p in products:
            q = r.randrange(5, 300)
            price = r.choice([5, 8, 10, 12, 15, 20, 25, 30])
            rows.append((P.MONTHS[m - 1], p, q, q * price))
    f.sales = {"path": f"{folders['down']}/vendite.csv", "rows": rows}
    home[f.sales["path"]] = "mese,prodotto,quantita,ricavo_eur\n" + "".join(f"{a},{b},{c},{d}\n" for a, b, c, d in rows)

    # app config
    f.config = {"path": f"{r.choice(['config', '.config/app', 'impostazioni'])}/app.ini", "port": r.choice([3000, 5000, 8000, 8080, 8443, 9000]),
                "db": r.choice(["gestionale", "archivio", "soci", "magazzino"]), "level": r.choice(["info", "debug", "warning"])}
    home[f.config["path"]] = (f"[server]\nporta = {f.config['port']}\nhost = 127.0.0.1\n\n[database]\nnome = {f.config['db']}\n"
                              f"utente = {f.config['db']}_app\n\n[log]\nlivello = {f.config['level']}\n")

    # photos, one of them a scanned identity document
    tag = r.choice(["vacanze", "matrimonio", "gita", "compleanno", "montagna", "mare"])
    files = [f"{tag}-{i:02d}.jpg" for i in range(1, r.randrange(3, 8))]
    secret = r.choice(["scansione-carta-identita.jpg", "patente-fronte.jpg", "tessera-sanitaria.jpg"])
    f.photos = {"folder": folders["photos"], "files": files, "sensitive": secret}
    for name in files + [secret]:
        home[f"{folders['photos']}/{name}"] = "JPEG\n"

    # notes
    for i, (text, tags) in enumerate(r.sample(P.NOTE_TOPICS, r.randrange(4, 7)), start=1):
        when = today + dt.timedelta(days=r.randrange(3, 40))
        t = text.format(date=it_date(when), code=f"{r.randrange(1000, 9999)}-{r.randrange(100, 999)}",
                        name=r.choice(people), thing=r.choice(P.THINGS))
        f.notes.append({"id": i, "testo": t, "etichette": list(tags), "data": (today - dt.timedelta(days=r.randrange(1, 30))).isoformat()})

    # calendar: events over the next two weeks
    for i in range(r.randrange(4, 9)):
        title, minutes = r.choice(P.EVENT_KINDS)
        d = today + dt.timedelta(days=r.randrange(0, 14))
        start = dt.datetime.combine(d, dt.time(r.choice([8, 9, 10, 11, 14, 15, 16, 17, 18, 19]), r.choice([0, 30])))
        end = start + dt.timedelta(minutes=minutes)
        f.events.append({"id": f"ev{i + 1}", "title": title.format(name=r.choice(people)),
                         "start": start.isoformat(timespec="minutes"), "end": end.isoformat(timespec="minutes"),
                         "location": r.choice([None, None, f"{r.choice(P.STREETS)} {r.randrange(1, 120)}"])})

    # web pages: an office with opening hours and a fee
    page, body = office_page(r, r.choice(P.CITIES), r.choice(P.SITE_KINDS))
    f.pages.append(page)
    pages = {page["url"]: {"title": page["title"], "body": body}}

    # tax codes, some with a wrong check character
    for _ in range(3):
        cf = make_cf(r, r.choice(P.FIRST), r.choice(P.LAST))
        if r.random() < 0.4:
            bad = chr((ord(cf[-1]) - 65 + r.randrange(1, 25)) % 26 + 65)
            f.tax_codes.append((cf[:-1] + bad, False))
        else:
            f.tax_codes.append((cf, True))

    if v >= 2:
        extend_v2(f, home, pages, seed)
    return World(now=now, home=home, notes=f.notes, events=f.events, pages=pages), f


def office_page(r: random.Random, city: str, kind: str) -> tuple[dict, str]:
    slug = city.lower().replace(" ", "-").replace("'", "")
    hours = {}
    for wd in P.WEEKDAYS:
        if wd == "domenica" or (wd == "sabato" and r.random() < 0.5):
            hours[wd] = None
        else:
            o, c = r.choice([(8, 12), (8, 13), (9, 13), (9, 18), (8, 14)])
            hours[wd] = f"{o}:{r.choice(['00', '30'])} - {c}:{r.choice(['00', '30'])}"
    fee = Decimal(r.randrange(2, 60)) + Decimal(r.choice([0, 50, 21, 90])) / 100
    page = {"url": f"https://{kind.replace(' ', '-')}.{slug}.example/orari", "title": f"{kind.capitalize()} di {city}: orari e costi",
            "city": city, "kind": kind, "hours": hours, "fee": fee, "address": f"{r.choice(P.STREETS)} {r.randrange(1, 90)}"}
    body = (f"# {page['title']}\n\nIndirizzo: {page['address']}, {city}.\n\n## Orari\n"
            + "".join(f"- {wd}: {h or 'chiuso'}\n" for wd, h in hours.items())
            + f"\n## Costo del servizio\n{euro(fee)} euro, pagabili allo sportello.\n")
    return page, body


def render_invoice(inv: dict, user: tuple[str, str], layout: int) -> str:
    imp, shown, issued, due = inv["imponibile"], inv["shown_iva"], inv["issued"], inv["due"]
    if layout == 1:
        return (f"Spett.le {user[0]} {user[1]}\nDocumento: fattura {inv['number']}\nData emissione: {issued:%d/%m/%Y}\n"
                f"Emessa da: {inv['supplier']}\nOggetto: {inv['service']}\n\nTotale imponibile: € {euro(imp)}\n"
                f"Aliquota IVA: {inv['rate']}%\nImporto IVA: € {euro(shown)}\nTotale documento: € {euro(inv['total'])}\n\n"
                f"Scadenza pagamento: {due:%d/%m/%Y}\nCoordinate bancarie: {inv['iban_shown']}\n")
    return (f"campo;valore\nnumero;{inv['number']}\ndata;{issued.isoformat()}\nfornitore;{inv['supplier']}\n"
            f"cliente;{user[0]} {user[1]}\ndescrizione;{inv['service']}\nimponibile;{euro(imp)}\naliquota_iva;{inv['rate']}\n"
            f"iva;{euro(shown)}\ntotale;{euro(inv['total'])}\nscadenza;{due.isoformat()}\niban;{inv['iban_shown']}\n")


def extend_v2(f: Facts, home: dict[str, str], pages: dict, seed: int) -> None:
    """World v2: more files, pages and layouts, from a random stream of its own (v1 facts stay as they are)."""
    r = random.Random(f"world2-{seed}")
    today, fo = f.now.date(), f.folders
    f.version = 2

    # invoices: one of three layouts; some IBANs with a wrong digit (single-digit errors always fail mod 97)
    layout = r.randrange(3)
    for inv in f.invoices:
        if r.random() < 0.3:
            k = 10
            inv["iban_shown"] = inv["iban"][:k] + str((int(inv["iban"][k]) + r.randrange(1, 10)) % 10) + inv["iban"][k + 1:]
            inv["iban_ok"] = False
        if layout or not inv["iban_ok"]:
            home[inv["path"]] = render_invoice(inv, f.user, layout) if layout else \
                home[inv["path"]].replace(inv["iban"], inv["iban_shown"])

    # address book
    firsts = r.sample(P.FIRST, r.randrange(5, 9))
    people = []
    for first in firsts:
        last = r.choice(P.LAST)
        mail = f"{first}.{last}".lower().replace("'", "").replace(" ", "")
        people.append({"name": f"{first} {last}", "first": first, "phone": f"3{r.randrange(20, 99)} {r.randrange(100, 999)} {r.randrange(1000, 9999)}",
                       "email": f"{mail}@example.com", "city": r.choice(P.CITIES)})
    path = r.choice(["rubrica.csv", f"{fo['docs']}/contatti.csv", f"{fo['docs']}/rubrica.txt", "contatti.txt"])
    if path.endswith(".csv"):
        text = "nome;telefono;email;città\n" + "".join(f"{p['name']};{p['phone']};{p['email']};{p['city']}\n" for p in people)
    else:
        text = "\n".join(f"Nome: {p['name']}\nTelefono: {p['phone']}\nEmail: {p['email']}\nCittà: {p['city']}\n" for p in people)
    f.contacts = {"path": path, "people": people}
    home[path] = text

    # expenses of the last two months
    rows = []
    for _ in range(r.randrange(8, 15)):
        cat = r.choice(list(P.EXPENSE_CATS))
        rows.append((today - dt.timedelta(days=r.randrange(1, 60)), cat, r.choice(P.EXPENSE_CATS[cat]),
                     Decimal(r.randrange(300, 15000)) / 100))
    rows.sort()
    path = r.choice([f"{fo['docs']}/spese.csv", f"{fo['down']}/spese-{today.year}.csv", "spese.csv"])
    if r.random() < 0.5:
        text = "data,categoria,voce,importo\n" + "".join(f"{d.isoformat()},{c},{v},{a:.2f}\n" for d, c, v, a in rows)
    else:
        text = "data;categoria;voce;importo\n" + "".join(f"{d:%d/%m/%Y};{c};{v};{euro(a)}\n" for d, c, v, a in rows)
    f.expenses = {"path": path, "rows": rows}
    home[path] = text

    # application log
    lines, counts, last_error = [], {"INFO": 0, "WARN": 0, "ERROR": 0}, None
    t0 = dt.datetime.combine(today - dt.timedelta(days=1), dt.time(r.randrange(0, 12)))
    for _ in range(r.randrange(15, 31)):
        t0 += dt.timedelta(minutes=r.randrange(3, 50), seconds=r.randrange(60))
        lvl = r.choices(["INFO", "WARN", "ERROR"], [0.7, 0.15, 0.15])[0]
        counts[lvl] += 1
        msg = r.choice(P.APP_EVENTS[lvl])
        lines.append(f"{t0:%Y-%m-%d %H:%M:%S} {lvl} {msg}")
        if lvl == "ERROR":
            last_error = {"time": t0, "message": msg}
    f.applog = {"path": f"{fo['logs']}/app.log", "errors": counts["ERROR"], "warnings": counts["WARN"], "lines": len(lines),
                "last_error": last_error}
    home[f.applog["path"]] = "\n".join(lines) + "\n"

    # to-do list
    items = [x.format(name=r.choice(f.people)) for x in r.sample(P.TODO, r.randrange(4, 9))]
    path = r.choice(["todo.md", f"{fo['notes']}/cose-da-fare.md", f"{fo['notes']}/todo.txt"])
    style = r.randrange(3)
    home[path] = "".join((f"- [ ] {x}\n", f"- {x}\n", f"{i}. {x}\n")[style] for i, x in enumerate(items, start=1))
    f.todo = {"path": path, "items": items}

    # downloads and a work folder
    f.downloads = r.sample(P.DOWNLOADS, r.randrange(3, 7))
    for name in f.downloads:
        home[f"{fo['down']}/{name}"] = {"pdf": "PDF\n", "zip": "ZIP\n"}.get(name.rsplit(".", 1)[1], "data,descrizione,importo\n")
    proj = r.choice(["sito", "inventario", "trasloco", "corso", "bilancio"])
    home[f"{fo['work']}/{proj}/README.txt"] = f"Progetto {proj}\nResponsabile: {r.choice(f.people)}\nAvvio: {it_date(today - dt.timedelta(days=r.randrange(10, 200)))}\n"

    # a second office and a news page
    city = r.choice([c for c in P.CITIES if c != f.pages[0]["city"]])
    kind = r.choice([k for k in P.SITE_KINDS if k != f.pages[0]["kind"]])
    page, body = office_page(r, city, kind)
    f.pages.append(page)
    pages[page["url"]] = {"title": page["title"], "body": body}
    title = r.choice(P.NEWS)
    pages[f"https://notizie.{page['url'].split('.', 1)[1].split('/')[0]}/{r.randrange(100, 999)}"] = {
        "title": title, "body": f"# {title}\n\n{it_date(today)}. Notizia locale da {city}.\n"}
