"""Arithmetic and date shifts: tools the harness adds (flag "calc") to every task, never named by a task.

A small model sums a column of amounts or counts back from a due date by eye and gets it wrong;
these tools compute it. Expressions are parsed, never executed.
"""

from __future__ import annotations

import ast
import datetime as dt
import operator
import re
from decimal import ROUND_HALF_UP, Decimal, DivisionByZero, InvalidOperation

from .base import Env, ToolError
from .itcalc import euro, parse_date

WEEKDAYS = ("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")


def _plain(token: str) -> str:
    """'1.234,56' and '294,84' are Italian, '294.84' is not: every form becomes '1234.56'."""
    if re.fullmatch(r"[1-9]\d{0,2}(\.\d{3})+(,\d+)?", token):  # 0.125 is a decimal
        return token.replace(".", "").replace(",", ".")
    if token.count(",") == 1 and "." not in token:
        return token.replace(",", ".")
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return token
    raise ToolError(f"Numero non valido: '{token}'. Usa per esempio 294,84 oppure 1.234,56")


def evaluate(expression: str) -> Decimal:
    text = expression.replace("€", "").replace("×", "*").replace("÷", "/").replace("x", "*").strip()
    if not text or len(text) > 2000:
        raise ToolError("Espressione vuota o troppo lunga")
    text = _NUMBER.sub(lambda m: f"D('{_plain(m.group(0))}')", text)

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "D"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)):
            return Decimal(node.args[0].value)
        raise ToolError("Sono ammessi solo numeri, + - * / e parentesi")

    try:
        return walk(ast.parse(text, mode="eval"))
    except SyntaxError as e:
        raise ToolError(f"Espressione non valida: '{expression}'") from e
    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as e:
        raise ToolError("Divisione per zero") from e


def shown(x: Decimal) -> str:
    return str(int(x)) if x == x.to_integral_value() else euro(x.quantize(Decimal("0.01"), ROUND_HALF_UP))


class CalcEnv(Env):
    name = "calc"
    title = "calcolatrice"

    def build(self) -> None:
        s = self.server

        @s.tool()
        def calcola(espressione: str) -> str:
            """Calcola un'espressione con + - * / e parentesi, per esempio "51,23 + 68,32 + 17,75".
            Usala per sommare importi o contare, invece di fare il conto a mente."""
            return f"{espressione.strip()} = {shown(evaluate(espressione))}"

        @s.tool()
        def aggiungi_giorni(data: str, giorni: int) -> str:
            """Aggiunge (o toglie, con un numero negativo) giorni di calendario a una data AAAA-MM-GG
            e dice che giorno della settimana è, per esempio il giorno prima di una scadenza: giorni=-1."""
            d = parse_date(data) + dt.timedelta(days=int(giorni))
            return f"{data.strip()} {'+' if giorni >= 0 else '-'} {abs(int(giorni))} giorni: {WEEKDAYS[d.weekday()]} {d.isoformat()}"
