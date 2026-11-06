"""Italian deterministic utilities: working days, VAT, tax code and IBAN checks."""

from __future__ import annotations

import datetime as dt
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from .base import Env, ToolError

_CF_ODD = dict(zip("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                   [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 1, 0, 5, 7, 9, 13, 15, 17, 19, 21,
                    2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]))
_CF_RE = re.compile(r"^[A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]$")


def easter(year: int) -> dt.date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def holidays(year: int) -> dict[dt.date, str]:
    fixed = {(1, 1): "Capodanno", (1, 6): "Epifania", (4, 25): "Festa della Liberazione",
             (5, 1): "Festa del Lavoro", (6, 2): "Festa della Repubblica", (8, 15): "Ferragosto",
             (11, 1): "Ognissanti", (12, 8): "Immacolata", (12, 25): "Natale", (12, 26): "Santo Stefano"}
    out = {dt.date(year, m, d): n for (m, d), n in fixed.items()}
    out[easter(year) + dt.timedelta(days=1)] = "Lunedì dell'Angelo"
    return out


def is_working(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def cf_check_char(cf15: str) -> str:
    total = 0
    for i, ch in enumerate(cf15):
        if i % 2 == 0:
            total += _CF_ODD[ch]
        else:
            total += int(ch) if ch.isdigit() else ord(ch) - ord("A")
    return chr(total % 26 + ord("A"))


def iban_ok(iban: str) -> bool:
    s = iban[4:] + iban[:4]
    return int("".join(str(int(c, 36)) for c in s)) % 97 == 1


def euro(x: Decimal) -> str:
    s = f"{x:,.2f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError as e:
        raise ToolError(f"Data non valida: '{value}'. Usa il formato AAAA-MM-GG, es. 2026-11-04") from e


class ItCalcEnv(Env):
    name = "itcalc"
    title = "calcoli-italia"

    def build(self) -> None:
        s = self.server

        @s.tool()
        def giorni_lavorativi(inizio: str, fine: str) -> str:
            """Conta i giorni lavorativi tra due date, estremi inclusi, escludendo sabati,
            domeniche e festività nazionali italiane. Date nel formato AAAA-MM-GG."""
            a, b = parse_date(inizio), parse_date(fine)
            if b < a:
                raise ToolError("La data di fine è precedente a quella di inizio")
            n, feste, d = 0, [], a
            while d <= b:
                if is_working(d):
                    n += 1
                elif d.weekday() < 5:
                    feste.append(f"{d.isoformat()} {holidays(d.year)[d]}")
                d += dt.timedelta(days=1)
            extra = f"; festività escluse: {', '.join(feste)}" if feste else ""
            return f"Giorni lavorativi dal {a.isoformat()} al {b.isoformat()}: {n}{extra}"

        @s.tool()
        def aggiungi_giorni_lavorativi(data: str, giorni: int) -> str:
            """Restituisce la data che cade dopo N giorni lavorativi a partire da una data
            (la data di partenza non si conta)."""
            d = parse_date(data)
            if giorni < 0:
                raise ToolError("Il numero di giorni deve essere positivo")
            left = giorni
            while left > 0:
                d += dt.timedelta(days=1)
                if is_working(d):
                    left -= 1
            nomi = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
            return f"{giorni} giorni lavorativi dopo il {data}: {d.isoformat()} ({nomi[d.weekday()]})"

        @s.tool()
        def giorni_tra_date(da: str, a: str) -> str:
            """Conta i giorni di calendario tra due date (AAAA-MM-GG)."""
            x, y = parse_date(da), parse_date(a)
            return f"Tra il {x.isoformat()} e il {y.isoformat()} ci sono {(y - x).days} giorni"

        @s.tool()
        def calcola_iva(importo: float, aliquota: float = 22, scorporo: bool = False) -> str:
            """Calcola l'IVA. Con scorporo=false l'importo è l'imponibile e si aggiunge l'IVA;
            con scorporo=true l'importo è IVA inclusa e se ne ricava l'imponibile."""
            try:
                imp, al = Decimal(str(importo)), Decimal(str(aliquota))
            except InvalidOperation as e:
                raise ToolError("Importo o aliquota non numerici") from e
            if al not in (Decimal(0), Decimal(4), Decimal(5), Decimal(10), Decimal(22)):
                raise ToolError("Aliquota non prevista in Italia: usa 0, 4, 5, 10 o 22")
            cent = Decimal("0.01")
            if scorporo:
                base = (imp / (1 + al / 100)).quantize(cent, ROUND_HALF_UP)
                iva = (imp - base).quantize(cent, ROUND_HALF_UP)
                tot = imp.quantize(cent, ROUND_HALF_UP)
            else:
                base = imp.quantize(cent, ROUND_HALF_UP)
                iva = (base * al / 100).quantize(cent, ROUND_HALF_UP)
                tot = base + iva
            return f"Imponibile: {euro(base)} EUR\nIVA {al.normalize()}%: {euro(iva)} EUR\nTotale: {euro(tot)} EUR"

        @s.tool()
        def valida_codice_fiscale(codice: str) -> str:
            """Controlla formato e carattere di controllo di un codice fiscale italiano di persona fisica."""
            cf = re.sub(r"\s", "", codice).upper()
            if len(cf) != 16:
                return f"Non valido: {len(cf)} caratteri invece di 16"
            if not _CF_RE.match(cf):
                return "Non valido: formato errato"
            exp = cf_check_char(cf[:15])
            if cf[15] != exp:
                return f"Non valido: carattere di controllo {cf[15]}, atteso {exp}"
            return "Valido"

        @s.tool()
        def valida_iban(iban: str) -> str:
            """Controlla un IBAN (lunghezza per l'Italia e cifre di controllo)."""
            v = re.sub(r"\s", "", iban).upper()
            if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}", v):
                return "Non valido: formato errato"
            if v.startswith("IT") and len(v) != 27:
                return f"Non valido: un IBAN italiano ha 27 caratteri, questo ne ha {len(v)}"
            return "Valido" if iban_ok(v) else "Non valido: cifre di controllo errate"
