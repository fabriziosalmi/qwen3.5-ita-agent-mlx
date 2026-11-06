"""Harness bench: a ladder of file tasks, from atomic reads to small fixes over two turns, built from code.

Each builder takes a Random and returns (task, home, oracle):
  task    the eval schema (qita/harness/checks.py); edits are scored on the final file, not on the calls
  home    the files of its world (relative path -> content)
  oracle  a scripted solution, one entry per model call: {"call": name, "args": {...}} or a final text
Every expected file is computed here by applying the change to the content built here (oracles come from code), and
the oracle must pass every task (tests/test_bench.py), so no task is unsolvable.

Levels: L1 lettura, L2 scrittura, L3 modifica, L4 piu' passi, L5 correzioni in due turni.
Splits: 'tune' is read freely while the harness is iterated;
'holdout' has other seeds and six types never seen in 'tune', and is read only at milestones; 'holdout2'
has the same types on seeds never used.
"""

from __future__ import annotations

import json
import random

SURNAMES = ["Rossi", "Bianchi", "Ferrari", "Esposito", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno"]
FIRST = ["Giulia", "Marco", "Sara", "Luca", "Elena", "Paolo", "Chiara", "Andrea", "Marta", "Davide"]
DOCS = [("contratto-affitto", "Contratto di affitto"), ("assicurazione-auto", "Assicurazione auto"),
        ("bolletta-luce", "Bolletta della luce"), ("preventivo-cucina", "Preventivo cucina"),
        ("abbonamento-palestra", "Abbonamento palestra"), ("garanzia-lavatrice", "Garanzia lavatrice")]
APPS = ["backup", "sito", "stampante", "mailer", "agenda", "fotocamera"]
PROJECTS = ["orto", "bilancio", "ricette", "viaggio-sicilia", "condominio", "tesi"]
ITEMS = ["latte", "pane", "uova", "mele", "caffè", "riso", "pasta", "farina", "zucchero", "yogurt", "olio", "pomodori"]
FILES = ["fattura-marzo.pdf", "foto-gita.jpg", "verbale.docx", "orari-treni.pdf", "menu.txt", "biglietto.pdf",
         "ricevuta.pdf", "scontrino.jpg", "mappa.png", "elenco.csv"]
PHRASES = ["Ricordarsi di chiamare l'idraulico lunedì.", "Il meeting è spostato a giovedì alle 15.",
           "Portare i documenti in banca entro venerdì.", "Pagare la rata del mutuo il 5 del mese.",
           "Comprare il regalo per il compleanno di Anna.", "Prenotare il tagliando dell'auto."]
TYPOS = [("appuntamneto", "appuntamento"), ("riunoine", "riunione"), ("fattrua", "fattura"),
         ("setimana", "settimana"), ("indrizzo", "indirizzo"), ("telfono", "telefono")]
TOPICS = ["promemoria", "cose-da-fare", "idee", "appunti", "scadenze"]

H = "/home/utente"


def C(name: str, **args) -> dict:
    return {"call": name, "args": args}


def P(path: str) -> dict:
    return {"path": f"{H}/{path}"}


def task(tid: str, level: str, envs: list[str], turns: list[dict], checks: dict | None = None) -> dict:
    return {"id": tid, "category": level, "envs": envs, "turns": turns, "checks": checks or {}}


def edit(path: str, old: str, new: str) -> dict:
    return C("edit_file", path=f"{H}/{path}", edits=json.dumps([{"oldText": old, "newText": new}], ensure_ascii=False))


def conf(app: str, porta: int, timeout: int, debug_line: str = "debug = no") -> str:
    return f"# configurazione di {app}\nnome = {app}\nporta = {porta}\ntimeout = {timeout}\n{debug_line}\n"


# ---------------- L1 lettura ----------------

def read_value(r):
    slug, title = r.choice(DOCS)
    who = f"{r.choice(FIRST)} {r.choice(SURNAMES)}"
    num = f"{r.choice('ABCDEFGH')}{r.choice('LMNPRS')}-{r.randrange(10000, 99999)}"
    day, month, year = r.randrange(1, 29), r.randrange(1, 13), r.choice([2027, 2028])
    euro = f"{r.randrange(40, 900)},{r.randrange(0, 100):02d}"
    path = f"documenti/{slug}.txt"
    body = f"Documento: {title}\nIntestatario: {who}\nNumero: {num}\nScadenza: {day:02d}/{month:02d}/{year}\nImporto: {euro} euro\n"
    field, q, want = r.choice([("Scadenza", "la scadenza", [f"{day:02d}/{month:02d}/{year}", f"{day}/{month}/{year}"]),
                               ("Numero", "il numero", [num]), ("Importo", "l'importo", [euro, euro.replace(",", ".")])])
    user = r.choice([f"Leggi {path} e dimmi {q}.", f"Nel file {path} qual è {q}?", f"Mi dici {q} scritto in {path}?"])
    t = task("", "L1", ["files"], [{"user": user}],
             {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}], "answer": {"any": [want]}})
    return t, {path: body, "note/varie.txt": "niente di importante\n"}, [C("read_text_file", path=f"{H}/{path}"),
                                                                       f"Nel file c'è scritto che {q} è {want[0]}."]


def list_dir(r):
    folder = r.choice(["scaricati", "documenti/vari", "lavoro", "casa"])
    names = sorted(r.sample(FILES, r.randrange(3, 6)))
    home = {f"{folder}/{n}": "x\n" for n in names} | {"note/varie.txt": "x\n"}
    user = r.choice([f"Che file ci sono nella cartella {folder}?", f"Elencami i file in {folder}.",
                     f"Cosa c'è dentro {folder}?"])
    t = task("", "L1", ["files"], [{"user": user}],
             {"calls": [{"tool": "list_directory", "args": {"path": P(folder)}}], "answer": {"contains": names}})
    return t, home, [C("list_directory", path=f"{H}/{folder}"), "Ci sono: " + ", ".join(names) + "."]


def count_lines(r):
    path = f"note/{r.choice(TOPICS)}.txt"
    items = r.sample(ITEMS, r.randrange(4, 10))
    user = r.choice([f"Quante righe ha il file {path}?", f"Contami le righe di {path}.", f"Di quante righe è fatto {path}?"])
    t = task("", "L1", ["files"], [{"user": user}],
             {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}], "answer": {"number": [len(items)]}})
    return t, {path: "\n".join(items) + "\n"}, [C("read_text_file", path=f"{H}/{path}"), f"Il file ha {len(items)} righe."]


def find_file(r):
    proj, name = r.choice(PROJECTS), r.choice(["leggimi", "istruzioni", "elenco-spese", "contatti", "calendario"])
    where = f"progetti/{proj}/docs"
    home = {f"{where}/{name}.md": "# testo\n", f"progetti/{proj}/note.txt": "x\n",
            f"progetti/{r.choice([p for p in PROJECTS if p != proj])}/note.txt": "x\n", "documenti/varie.txt": "x\n"}
    user = r.choice([f"Dove si trova il file {name}.md?", f"In che cartella ho messo {name}.md?",
                     f"Trovami il file {name}.md."])
    t = task("", "L1", ["files"], [{"user": user}],
             {"calls_any": [[{"tool": "search_files"}, {"tool": "list_directory"}]], "answer": {"contains": [where]}})
    return t, home, [C("search_files", path=H, pattern=f"{name}.md"), f"Si trova in {H}/{where}/{name}.md."]


def last_line(r):  # holdout only
    path = f"log/{r.choice(APPS)}.log"
    msgs = [f"2026-11-0{i} 0{i}:00 {m}" for i, m in enumerate(r.sample(
        ["avvio completato", "backup riuscito", "spazio quasi pieno", "connessione persa", "riavvio programmato",
         "aggiornamento installato", "errore di scrittura"], 5), start=1)]
    user = r.choice([f"Qual è l'ultima riga di {path}?", f"Dimmi l'ultima riga del log {path}."])
    t = task("", "L1", ["files"], [{"user": user}],
             {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}],
              "answer": {"contains": [msgs[-1].split(" ", 2)[2]]}})
    return t, {path: "\n".join(msgs) + "\n"}, [C("read_text_file", path=f"{H}/{path}", tail=1), f"L'ultima riga è: {msgs[-1]}."]


# ---------------- L2 scrittura ----------------

def write_new(r):
    path, text = f"note/{r.choice(TOPICS)}-{r.randrange(2, 9)}.txt", r.choice(PHRASES)
    user = r.choice([f"Crea il file {path} con scritto: {text}", f"Scrivi in un file nuovo {path} questa frase: {text}",
                     f"Crea {path} e mettici dentro: {text}"])
    t = task("", "L2", ["files"], [{"user": user}], {"state": [{"kind": "file_equals", "path": path, "content": text}]})
    return t, {"note/varie.txt": "x\n"}, [C("write_file", path=f"{H}/{path}", content=text), f"Ho creato {path}."]


def mkdir(r):
    path = f"progetti/{r.choice(PROJECTS)}/{r.choice(['bozze', 'archivio', 'immagini', 'fatture'])}"
    user = r.choice([f"Crea la cartella {path}.", f"Mi fai una cartella {path}?", f"Crea la directory {path}."])
    t = task("", "L2", ["files"], [{"user": user}], {"state": [{"kind": "dir_exists", "path": path}]})
    return t, {"progetti/leggimi.txt": "x\n"}, [C("create_directory", path=f"{H}/{path}"), f"Ho creato {path}."]


def rename(r):
    a, b = r.sample([d for d, _ in DOCS], 2)
    src, dst, body = f"documenti/{a}.txt", f"documenti/{b}-vecchio.txt", f"contenuto di {a}\n"
    user = r.choice([f"Rinomina {src} in {b}-vecchio.txt.", f"Cambia il nome di {src} in {b}-vecchio.txt."])
    t = task("", "L2", ["files"], [{"user": user}],
             {"state": [{"kind": "file_absent", "path": src}, {"kind": "file_equals", "path": dst, "content": body}]})
    return t, {src: body}, [C("move_file", source=f"{H}/{src}", destination=f"{H}/{dst}"), "Fatto."]


def move_into(r):  # holdout only
    name, folder = r.choice(FILES), f"documenti/{r.choice(['2026', 'casa', 'lavoro'])}"
    home = {f"scaricati/{name}": "x\n", f"{folder}/leggimi.txt": "x\n"}
    user = r.choice([f"Sposta scaricati/{name} nella cartella {folder}.", f"Metti scaricati/{name} dentro {folder}."])
    t = task("", "L2", ["files"], [{"user": user}],
             {"state": [{"kind": "file_absent", "path": f"scaricati/{name}"}, {"kind": "file_exists", "path": f"{folder}/{name}"}]})
    return t, home, [C("move_file", source=f"{H}/scaricati/{name}", destination=f"{H}/{folder}/{name}"), "Fatto."]


def delete_one(r):
    names = r.sample(FILES, 3)
    user = r.choice([f"Cancella il file scaricati/{names[0]}.", f"Elimina scaricati/{names[0]}."])
    t = task("", "L2", ["files"], [{"user": user}],
             {"state": [{"kind": "file_absent", "path": f"scaricati/{names[0]}"}]
              + [{"kind": "file_exists", "path": f"scaricati/{n}"} for n in names[1:]]})
    return t, {f"scaricati/{n}": "x\n" for n in names}, [C("delete_file", path=f"{H}/scaricati/{names[0]}"), "Fatto."]


def append_line(r):
    items = r.sample(ITEMS, 5)
    path, before = "spesa.txt", "\n".join(items[:4]) + "\n"
    user = r.choice([f"Aggiungi {items[4]} alla lista della spesa in {path}.",
                     f"Metti anche {items[4]} in fondo a {path}."])
    t = task("", "L2", ["files"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": path, "content": before + items[4]}]})
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"),
                               C("write_file", path=f"{H}/{path}", content=before + items[4] + "\n"), "Aggiunto."]


# ---------------- L3 modifica ----------------

def config_value(r):
    app, porta, timeout = r.choice(APPS), r.choice([8080, 3000, 5000, 9000]), r.choice([10, 30, 60])
    path, new = f"config/{app}.conf", porta + 1
    before = conf(app, porta, timeout)
    user = r.choice([f"Nel file {path} cambia la porta da {porta} a {new}.", f"In {path} metti la porta a {new}."])
    t = task("", "L3", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": path, "content": conf(app, new, timeout)}]})
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"),
                               edit(path, f"porta = {porta}", f"porta = {new}"), "Fatto."]


def replace_word(r):
    a, b = r.sample(SURNAMES, 2)
    first = r.choice(FIRST)
    path = f"documenti/lettera-{a.lower()}.txt"
    before = (f"Gentile sig. {a},\nle confermiamo l'appuntamento del 12 novembre.\n"
              f"La pratica intestata a {first} {a} è completa.\nCordiali saluti\n")
    after = before.replace(a, b)
    user = r.choice([f"Nel file {path} sostituisci {a} con {b} dappertutto.",
                     f"In {path} cambia ogni {a} in {b}."])
    t = task("", "L3", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": path, "content": after}]})
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), C("write_file", path=f"{H}/{path}", content=after),
                               "Fatto."]


def fix_typo(r):
    bad, good = r.choice(TYPOS)
    path = f"note/{r.choice(TOPICS)}.txt"
    before = f"lunedì: spesa\nmartedì: {bad} con Paolo\nmercoledì: palestra\n"
    user = r.choice([f"Nel file {path} c'è scritto '{bad}' invece di '{good}': correggilo.",
                     f"Correggi {bad} in {good} nel file {path}."])
    t = task("", "L3", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": path, "content": before.replace(bad, good)}]})
    return t, {path: before}, [edit(path, bad, good), "Corretto."]


def uncomment(r):  # holdout only
    app = r.choice(APPS)
    path = f"config/{app}.conf"
    before = conf(app, 8080, 30, "# debug = si")
    user = r.choice([f"Nel file {path} attiva la riga del debug togliendo il #.", f"In {path} decommenta la riga debug."])
    t = task("", "L3", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": path, "content": conf(app, 8080, 30, "debug = si")}]})
    return t, {path: before}, [edit(path, "# debug = si", "debug = si"), "Fatto."]


# ---------------- L4 piu' passi ----------------

def copy_value(r):
    iban = "IT" + f"{r.randrange(10, 99)}" + r.choice("ABCDXYZ") + "".join(str(r.randrange(10)) for _ in range(22))
    src = f"documenti/{r.choice(['banca', 'conto-corrente', 'stipendio'])}.txt"
    body = f"Banca: Credito Ligure\nIntestatario: {r.choice(FIRST)} {r.choice(SURNAMES)}\nIBAN: {iban}\n"
    user = r.choice([f"Copia l'IBAN che trovi in {src} in un file nuovo note/iban.txt, con solo l'IBAN.",
                     f"Prendi l'IBAN da {src} e scrivilo da solo nel file note/iban.txt."])
    t = task("", "L4", ["files"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": "note/iban.txt", "content": iban}]})
    return t, {src: body, "note/varie.txt": "x\n"}, [C("read_text_file", path=f"{H}/{src}"),
                                                       C("write_file", path=f"{H}/note/iban.txt", content=iban), "Fatto."]


def find_and_edit(r):
    proj = r.choice(PROJECTS)
    sub = r.choice(["config", "impostazioni", "setup"])
    path = f"progetti/{proj}/{sub}/progetto.ini"
    before = f"[progetto]\nnome = {proj}\nversione = 2.0\nautore = {r.choice(FIRST)}\n"
    user = r.choice([f"Nel progetto {proj} c'è un file progetto.ini da qualche parte: cambia la versione da 2.0 a 2.1.",
                     f"Trova il progetto.ini del progetto {proj} e porta la versione a 2.1."])
    t = task("", "L4", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": path, "content": before.replace("versione = 2.0", "versione = 2.1")}]})
    home = {path: before, f"progetti/{proj}/leggimi.txt": "x\n"}
    return t, home, [C("search_files", path=f"{H}/progetti/{proj}", pattern="progetto.ini"),
                     edit(path, "versione = 2.0", "versione = 2.1"), "Fatto."]


def merge(r):
    items = r.sample(ITEMS, 6)
    a, b = "note/da-fare.txt", "note/da-fare-vecchio.txt"
    ta, tb = "\n".join(items[:3]) + "\n", "\n".join(items[3:]) + "\n"
    user = r.choice([f"Aggiungi in fondo a {a} le righe di {b}.", f"Copia il contenuto di {b} alla fine di {a}."])
    t = task("", "L4", ["files"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": a, "content": ta + tb},
                        {"kind": "file_equals", "path": b, "content": tb}]})
    return t, {a: ta, b: tb}, [C("read_text_file", path=f"{H}/{a}"), C("read_text_file", path=f"{H}/{b}"),
                               C("write_file", path=f"{H}/{a}", content=ta + tb), "Fatto."]


# ---------------- L5 correzioni in due turni ----------------

def _bug(r):
    """(file name, buggy content, fixed contents, what it does, the bug told to the user, the bug without the fix)"""
    k = r.randrange(3)
    if k == 0:
        before = ("def media(numeri):\n    totale = 0\n    for n in numeri:\n        totale += n\n    return totale / 2\n\n\n"
                  "def massimo(numeri):\n    return max(numeri)\n")
        fixed = [before.replace("totale / 2", "totale / len(numeri)"), before.replace("totale / 2", "totale/len(numeri)")]
        return ("calc.py", before, fixed, "media",
                "la funzione media divide per 2 invece che per il numero di elementi", "la media viene sbagliata")
    if k == 1:
        before = "#!/bin/sh\n# stampa i numeri da 1 a 10\nfor i in $(seq 1 9); do\n    echo \"$i\"\ndone\n"
        return ("conta.sh", before, [before.replace("seq 1 9", "seq 1 10")], "numeri",
                "il ciclo si ferma a 9 ma deve arrivare a 10", "non stampa tutti i numeri del commento")
    before = ("def maggiorenne(eta):\n    # vero da 18 anni in su\n    if eta > 18:\n        return True\n    return False\n")
    return ("eta.py", before, [before.replace("eta > 18", "eta >= 18")], "maggiorenne",
            "con 18 anni esatti risponde falso, ma 18 anni è già maggiorenne", "sbaglia proprio con 18 anni")


def mt_read_then_fix(r):
    name, before, fixed, word, bug, _ = _bug(r)
    path = f"progetti/{r.choice(PROJECTS)}/{name}"
    t1 = {"user": r.choice([f"Cosa fa il file {path}?", f"Leggimi {path} e dimmi in breve cosa fa."]),
          "checks": {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}], "answer": {"contains": [word]}}}
    t2 = {"user": f"C'è un errore: {bug}. Correggi solo quella riga."}
    t = task("", "L5", ["files", "edit"], [t1, t2],
             {"state": [{"kind": "file_equals", "path": path, "content": {"any": fixed}}]})
    old, new = _diff_line(before, fixed[0])
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), f"Contiene {word}.", edit(path, old, new), "Corretto."]


def mt_fix_then_amend(r):
    app, porta = r.choice(APPS), r.choice([8080, 3000])
    a, b = r.sample([15, 20, 45, 90, 120], 2)
    path = f"config/{app}.conf"
    before = conf(app, porta, 30)
    t1 = {"user": r.choice([f"In {path} metti il timeout a {a}.", f"Cambia il timeout di {path} a {a}."])}
    t2 = {"user": r.choice([f"Anzi no, mettilo a {b}.", f"Scusa, volevo dire {b}."])}
    t = task("", "L5", ["files", "edit"], [t1, t2],
             {"state": [{"kind": "file_equals", "path": path, "content": conf(app, porta, b)}]})
    return t, {path: before}, [edit(path, "timeout = 30", f"timeout = {a}"), "Fatto.",
                               edit(path, f"timeout = {a}", f"timeout = {b}"), "Fatto."]


def mt_find_bug_then_append(r):
    name, before, fixed, _, _, hint = _bug(r)
    path = f"progetti/{r.choice(PROJECTS)}/{name}"
    mark = "# controllato"
    t1 = {"user": f"Nel file {path} c'è un errore: {hint}. Trovalo e correggilo."}
    t2 = {"user": r.choice([f"Bene. Ora aggiungi in fondo al file una riga con scritto {mark}",
                            f"Perfetto, adesso metti {mark} come ultima riga del file."])}
    t = task("", "L5", ["files", "edit"], [t1, t2],
             {"state": [{"kind": "file_equals", "path": path,
                         "content": {"any": [f + sep + mark for f in fixed for sep in ("", "\n")]}}]})  # a blank line before is fine
    old, new = _diff_line(before, fixed[0])
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), edit(path, old, new), "Corretto.",
                               C("write_file", path=f"{H}/{path}", content=fixed[0] + mark + "\n"), "Fatto."]


def mt_rename_var(r):  # holdout only
    path = f"progetti/{r.choice(PROJECTS)}/somma.py"
    before = "def somma(valori):\n    s = 0\n    for v in valori:\n        s = s + v\n    return s\n"
    after = before.replace(" s ", " totale ").replace("    s =", "    totale =").replace("return s", "return totale")
    t1 = {"user": f"Leggimi {path}.", "checks": {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}]}}
    t2 = {"user": "Rinomina la variabile s in totale in tutta la funzione."}
    t = task("", "L5", ["files", "edit"], [t1, t2], {"state": [{"kind": "file_equals", "path": path, "content": after}]})
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), "Ecco il file.",
                               C("write_file", path=f"{H}/{path}", content=after), "Fatto."]


def _diff_line(a: str, b: str) -> tuple[str, str]:
    (x, y), = [(p, q) for p, q in zip(a.split("\n"), b.split("\n")) if p != q]
    return x, y


# ---------------- bench v1: harder types; the v0 types above are unchanged ----------------

def _bug2(r):
    """(file name, buggy content, fixed contents, the symptom the user sees). One line to change."""
    k = r.randrange(5)
    if k == 0:
        before = "def ultimo(lista):\n    return lista[len(lista)]\n"
        return ("liste.py", before, [before.replace("lista[len(lista)]", x) for x in
                                     ("lista[-1]", "lista[len(lista) - 1]", "lista[len(lista)-1]")],
                "ultimo([1, 2, 3]) dà IndexError invece di 3")
    if k == 1:
        before = "# prezzo dopo lo sconto del 10%\ndef prezzo_scontato(prezzo):\n    return prezzo * 0.1\n"
        return ("prezzi.py", before, [before.replace("prezzo * 0.1", x) for x in
                                      ("prezzo * 0.9", "prezzo*0.9", "prezzo - prezzo * 0.1", "prezzo * (1 - 0.1)")],
                "prezzo_scontato(100) restituisce 10 invece di 90")
    if k == 2:
        before = 'def saluta(nome):\n    return "Ciao " + "nome"\n'
        return ("saluti.py", before, [before.replace('"Ciao " + "nome"', x) for x in
                                      ('"Ciao " + nome', 'f"Ciao {nome}"', '"Ciao "+nome')],
                "saluta('Anna') restituisce 'Ciao nome' invece di 'Ciao Anna'")
    if k == 3:
        before = "def pari(n):\n    return n % 2 == 1\n"
        return ("numeri.py", before, [before.replace("n % 2 == 1", x) for x in ("n % 2 == 0", "n%2 == 0", "n % 2 != 1")],
                "pari(4) restituisce False")
    before = "def somma_lista(xs):\n    t = 0\n    for x in xs:\n        t = x\n    return t\n"
    return ("somme.py", before, [before.replace("        t = x", x) for x in ("        t += x", "        t = t + x")],
            "somma_lista([1, 2, 3]) restituisce 3 invece di 6")


def mt_symptom_fix(r):
    name, before, fixed, symptom = _bug2(r)
    path = f"progetti/{r.choice(PROJECTS)}/{name}"
    t1 = {"user": r.choice([f"Ho un problema con {path}: {symptom}. Dov'è l'errore?",
                            f"Guarda {path}: {symptom}. Cosa c'è che non va?"]),
          "checks": {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}]}}
    t2 = {"user": r.choice(["Ok, correggilo.", "Va bene, sistemalo tu."])}
    t = task("", "L5", ["files", "edit"], [t1, t2],
             {"state": [{"kind": "file_equals", "path": path, "content": {"any": fixed}}]})
    old, new = _diff_line(before, fixed[0])
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), f"L'errore è nella riga: {old.strip()}.",
                               edit(path, old, new), "Corretto."]


def json_fix(r):
    app, porta = r.choice(APPS), r.choice([8080, 3000, 5000])
    path = f"config/{app}.json"
    before = f'{{\n  "nome": "{app}",\n  "porta": {porta}\n  "debug": false\n}}\n'
    t1 = {"user": r.choice([f"Il file {path} non si carica, dice errore di sintassi. Cosa c'è che non va?",
                            f"{path} dà un errore di sintassi JSON: guardalo e dimmi il problema."]),
          "checks": {"calls": [{"tool": "read_text_file", "args": {"path": P(path)}}], "answer": {"any": [["virgola"]]}}}
    t2 = {"user": r.choice(["Correggilo.", "Sistemalo per favore."])}
    t = task("", "L5", ["files", "edit"], [t1, t2],
             {"state": [{"kind": "json_equals", "path": path, "value": {"nome": app, "porta": porta, "debug": False}}]})
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), "Manca una virgola dopo la porta.",
                               edit(path, f'"porta": {porta}', f'"porta": {porta},'), "Corretto."]


def mt_undo(r):
    app, porta, timeout = r.choice(APPS), r.choice([8080, 3000, 5000]), 30
    new = r.choice([8000, 8888, 4000])
    path = f"config/{app}.conf"
    before = conf(app, porta, timeout)
    t1 = {"user": f"In {path} cambia la porta a {new}."}
    t2 = {"user": r.choice(["Che valore c'era prima?", "Quale porta c'era prima?"]),
          "checks": {"answer": {"contains": [str(porta)]}}}
    t3 = {"user": r.choice(["Allora rimettila com'era.", "Ok, torna al valore di prima."])}
    t = task("", "L5", ["files", "edit"], [t1, t2, t3],
             {"state": [{"kind": "file_equals", "path": path, "content": before}]})
    return t, {path: before}, [edit(path, f"porta = {porta}", f"porta = {new}"), "Fatto.", f"Prima c'era {porta}.",
                               edit(path, f"porta = {new}", f"porta = {porta}"), "Fatto."]


def multi_file_edit(r):
    old, new = f"{r.choice(FIRST).lower()}@vecchiaposta.it", f"{r.choice(FIRST).lower()}@nuovaposta.it"
    names = r.sample(SURNAMES, 3)
    home, after = {}, {}
    for i, n in enumerate(names):
        mail = old if i < 2 else f"{n.lower()}@altro.it"
        home[f"contatti/{n.lower()}.txt"] = f"nome: {n}\nemail: {mail}\ntelefono: 3{r.randrange(10**8, 10**9)}\n"
        after[f"contatti/{n.lower()}.txt"] = home[f"contatti/{n.lower()}.txt"].replace(old, new)
    user = r.choice([f"Nella cartella contatti sostituisci l'email {old} con {new} in tutti i file dove compare.",
                     f"Aggiorna {old} in {new} in tutti i contatti che la usano."])
    t = task("", "L4", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_equals", "path": p, "content": c} for p, c in after.items()]})
    oracle = [C("list_directory", path=f"{H}/contatti")]
    oracle += [C("read_text_file", path=f"{H}/{p}") for p in home] + [edit(p, old, new) for p in list(home)[:2]] + ["Fatto."]
    return t, home, oracle


def block_replace(r):  # holdout only
    path = f"documenti/orari-{r.choice(['negozio', 'ufficio', 'palestra'])}.txt"
    days = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì"]
    before = "".join(f"{d}: 9-18\n" for d in days)
    a, b = r.sample(days[:4], 2)
    after = before.replace(f"{a}: 9-18", f"{a}: 8-17").replace(f"{b}: 9-18", f"{b}: 8-17")
    user = f"In {path} cambia gli orari di {a} e {b} in 8-17, lascia gli altri giorni come sono."
    t = task("", "L3", ["files", "edit"], [{"user": user}], {"state": [{"kind": "file_equals", "path": path, "content": after}]})
    return t, {path: before}, [C("read_text_file", path=f"{H}/{path}"), C("write_file", path=f"{H}/{path}", content=after),
                               "Fatto."]


def phone_append(r):  # holdout only
    n = r.choice(SURNAMES)
    tel = f"3{r.randrange(10**8, 10**9)}"
    src = f"contatti/{n.lower()}.txt"
    first = "chiamare il dentista"
    user = f"Aggiungi in fondo a note/chiamare.txt una riga con il nome e il telefono di {n}, che trovi nei contatti."
    t = task("", "L4", ["files", "edit"], [{"user": user}],
             {"state": [{"kind": "file_contains", "path": "note/chiamare.txt", "text": [first, tel, n]}]})
    home = {src: f"nome: {n}\ntelefono: {tel}\n", "note/chiamare.txt": first + "\n"}
    return t, home, [C("read_text_file", path=f"{H}/{src}"), C("read_text_file", path=f"{H}/note/chiamare.txt"),
                     C("write_file", path=f"{H}/note/chiamare.txt", content=f"{first}\n{n} {tel}\n"), "Fatto."]


V0 = [read_value, list_dir, count_lines, find_file, write_new, mkdir, rename, delete_one, append_line,
      config_value, replace_word, fix_typo, copy_value, find_and_edit, merge,
      mt_read_then_fix, mt_fix_then_amend, mt_find_bug_then_append]
V1_NEW = [multi_file_edit, mt_symptom_fix, json_fix, mt_undo]
TUNE = V0 + V1_NEW
HOLDOUT_ONLY = [last_line, move_into, uncomment, mt_rename_var, block_replace, phone_append]
# builders, per type, seed base. Once a holdout read has shaped the harness, the next milestone reads holdout2.
SPLITS = {"tune": (TUNE, 3, 100), "holdout": (TUNE + HOLDOUT_ONLY, 2, 900), "holdout2": (TUNE + HOLDOUT_ONLY, 2, 1900)}


def build(split: str) -> list[tuple[dict, dict, list]]:
    builders, per, base = SPLITS[split]
    out = []
    for b in builders:
        for k in range(per):
            seed = base + k
            t, home, oracle = b(random.Random(f"{b.__name__}-{seed}"))
            t["id"] = f"b-{b.__name__}-{seed}"
            t["type"] = b.__name__
            out.append((t, home, oracle))
    return out
