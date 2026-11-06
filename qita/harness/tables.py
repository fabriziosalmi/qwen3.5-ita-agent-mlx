"""Facts about a file computed by the harness, so a small model copies them instead of working them out:
the totals and the maximum of a CSV table, and the date order of a log."""

from __future__ import annotations

import csv
import datetime as dt
import io
import re

SKIP = {"id", "anno", "year", "codice", "cap", "telefono", "cellulare", "tel", "numero", "iban", "matricola",
        "codice_fiscale", "partita_iva", "piva"}  # numeric columns whose sum means nothing
MAX_GROUPS = 12  # a text column with more distinct values is not a grouping


def number(value: str, decimal_comma: bool) -> float | None:
    v = value.replace("€", "").strip()
    if " " in v or (len(v) > 1 and v[0] == "0" and v[1].isdigit()):
        return None  # "334 250 7038", "0101": a phone number or a code, not a quantity
    if decimal_comma:
        if "," in v:  # 1.234,56
            v = v.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", v):  # 1.250: a thousands dot
            v = v.replace(".", "")
        # otherwise a dot is a decimal point, as in 12.50
    try:
        return float(v)
    except ValueError:
        return None


def it_number(x: float) -> str:
    """Italian format: 10.155 and 1.234,56."""
    if abs(x - round(x)) < 1e-9:
        return f"{round(x):,}".replace(",", ".")
    return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def csv_totals(text: str) -> str | None:
    """One block with the row count, the total of each numeric column and the totals per value of each
    short text column; None when the text is not a table with a numeric column."""
    lines = [x for x in text.splitlines() if x.strip()]
    if len(lines) < 3:
        return None
    sep = ";" if lines[0].count(";") > lines[0].count(",") else ","
    rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=sep))
    head, body = [h.strip() for h in rows[0]], [r for r in rows[1:] if len(r) == len(rows[0])]
    if len(head) < 2 or len(body) < 2:
        return None
    cols = list(zip(*body))
    nums = {}
    for name, col in zip(head, cols):
        vals = [number(v, sep == ";") for v in col if v.strip()]
        if vals and all(v is not None for v in vals) and name.casefold() not in SKIP:
            nums[name] = [number(v, sep == ";") or 0.0 for v in col]
    if not nums:
        return None
    out = [f"[calcolo del harness, dal file] righe: {len(body)}. Totali: "
           + "; ".join(f"{n} {it_number(sum(v))}" for n, v in nums.items()) + "."]
    labels = [i for i, name in enumerate(head) if name not in nums]
    tops = []
    for name, v in nums.items():
        i = max(range(len(v)), key=v.__getitem__)
        tops.append(f"{name} {it_number(v[i])} (" + ", ".join(body[i][j].strip() for j in labels) + ")")
    out.append("Massimo: " + "; ".join(tops) + ".")
    for name, col in zip(head, cols):
        if name in nums:
            continue
        keys = list(dict.fromkeys(v.strip() for v in col))
        if not 2 <= len(keys) <= MAX_GROUPS or len(keys) == len(body):
            continue
        parts = []
        for k in keys:
            idx = [i for i, v in enumerate(col) if v.strip() == k]
            parts.append(f"{k}: " + ", ".join(f"{n} {it_number(sum(nums[n][i] for i in idx))}" for n in nums))
        out.append(f"Per {name}: " + "; ".join(parts) + ".")
    return "\n".join(out)


DATED = re.compile(r"^\s*\[?(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})")


def day(text: str) -> dt.date:
    if "-" in text:
        return dt.date.fromisoformat(text)
    d, m, y = (int(x) for x in text.split("/"))
    return dt.date(y, m, d)


def dated(text: str) -> str | None:
    """For a file whose lines start with a date, which way the dates run and its three newest lines; None
    when fewer than three lines start with a valid date. A model read a log from the top as if the first
    lines were the latest."""
    lines = [x for x in text.splitlines() if x.strip()]
    found = [m.group(1) for x in lines if (m := DATED.match(x))]
    if len(found) < 3:
        return None
    first, last = found[0], found[-1]
    try:
        newest_last = day(first) <= day(last)
    except ValueError:
        return None
    if newest_last:
        order, recent = "dalla più vecchia alla più recente", lines[-3:]
    else:
        order, recent = "dalla più recente alla più vecchia", lines[:3]
    return (f"[calcolo del harness, dal file] le righe vanno {order}: prima riga del {first}, ultima del {last}. "
            "Le righe più recenti:\n" + "\n".join(recent))
