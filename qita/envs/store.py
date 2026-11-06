"""A record store whose tools change name, parameters and output from world to world (qita/worlds/store.py).

It stands for an MCP server the model has never seen: English tool names and descriptions in a vocabulary
drawn per world, Italian records. No rule of the harness is written for its tools (generic = True keeps it
out of KNOWN_ENVS in qita/harness/runner.py), so the harness treats it like any other server.
"""

from __future__ import annotations

import json
import re
import unicodedata
from inspect import Parameter, Signature

from mcp.types import ToolAnnotations

from .base import Env, ToolError

STOP = {"del", "della", "dello", "dei", "degli", "delle", "che", "per", "con", "una", "uno", "gli", "nel", "nella"}


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).casefold()) if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    """The words a search must find: 3+ letters, or any token with a digit ("R4", "A1"), stopwords out."""
    return [w for w in re.findall(r"[a-z0-9]+", fold(text))
            if (len(w) >= 3 or re.search(r"\d", w)) and w not in STOP]


def default_spec() -> dict:
    from ..worlds.store import build_store
    return build_store(0, domain="dischi")


class StoreEnv(Env):
    name = "store"
    title = "store"
    generic = True

    def build(self) -> None:
        spec = getattr(self.ws.world, "store", None) or default_spec()
        self.spec = spec
        self.records = [dict(x) for x in spec["records"]]
        key, name, kinds = spec["key"], spec["name"], spec["fields"]

        def show(x) -> str:
            if spec["fmt"] == "json":
                return json.dumps(x, ensure_ascii=False, indent=2)
            rows = x if isinstance(x, list) else [x]
            return "\n".join("; ".join(f"{k}={v}" for k, v in rec.items()) for rec in rows) or "(no records)"

        def find(k):
            for rec in self.records:
                if str(rec[key]) == str(k).strip():
                    return rec
            raise ToolError(f"not found: {key}={k}")

        def coerce(field: str, value):
            if kinds[field] == "int":
                try:
                    v = str(value).strip()
                    return int(v.replace(".", "") if re.fullmatch(r"-?[1-9]\d{0,2}(\.\d{3})+", v) else v)
                except ValueError:
                    raise ToolError(f"{field} must be a whole number, got {value!r}") from None
            return str(value).strip()

        def search(**a):
            want = words(a[spec["query"]])
            if not want:
                raise ToolError(f"{spec['query']} is empty")
            hits = [rec for rec in self.records if all(w in fold(" ".join(map(str, rec.values()))) for w in want)]
            return show(hits)

        def listing(**a):
            n = a.get("limit") or 50
            return show(self.records[:max(1, int(n))])

        def get(**a):
            return show(find(a[key]))

        def add(**a):
            nm = str(a[name]).strip()
            if any(fold(rec[name]) == fold(nm) for rec in self.records):
                raise ToolError(f"a record named {nm!r} already exists")
            from ..worlds.store import slug
            new = {key: max([rec[key] for rec in self.records] + [0]) + 1 if spec["numeric"] else slug(nm), name: nm}
            for f in kinds:
                if a.get(f) not in (None, ""):
                    new[f] = coerce(f, a[f])
            self.records.append(new)
            return show(new)

        def update(**a):
            rec = find(a[key])
            f = str(a["field"]).strip()
            if f not in kinds and f != name:
                raise ToolError(f"unknown field {f!r}; fields: {', '.join([name, *kinds])}")
            rec[f] = coerce(f, a["value"]) if f in kinds else str(a["value"]).strip()
            return show(rec)

        def delete(**a):
            rec = find(a[key])
            self.records.remove(rec)
            return json.dumps({"deleted": rec[key]}) if spec["fmt"] == "json" else f"deleted {key}={rec[key]}"

        kt = int if spec["numeric"] else str
        req = lambda n, t=str: Parameter(n, Parameter.KEYWORD_ONLY, annotation=t)  # noqa: E731
        opt = lambda n, t: Parameter(n, Parameter.KEYWORD_ONLY, annotation=t | None, default=None)  # noqa: E731
        typed = {"int": int, "date": str, "text": str, "money": str}
        sigs = {
            "search": [req(spec["query"])],
            "list": [opt("limit", int)],
            "get": [req(key, kt)],
            "add": [req(name)] + [opt(f, typed[k]) for f, k in kinds.items()],
            "update": [req(key, kt), req("field"), req("value")],
            "delete": [req(key, kt)],
        }
        fns = {"search": search, "list": listing, "get": get, "add": add, "update": update, "delete": delete}
        hints = {"search": dict(readOnlyHint=True), "list": dict(readOnlyHint=True), "get": dict(readOnlyHint=True),
                 "add": dict(readOnlyHint=False, destructiveHint=False),
                 "update": dict(readOnlyHint=False, destructiveHint=False),
                 "delete": dict(readOnlyHint=False, destructiveHint=True)}
        for op, fn in fns.items():
            def bound(_fn=fn, **a):
                return _fn(**a)
            params = sigs[op]
            bound.__signature__ = Signature(params, return_annotation=str)
            bound.__annotations__ = {p.name: p.annotation for p in params} | {"return": str}
            bound.__name__ = spec["tools"][op].replace("-", "_")
            self.server.add_tool(bound, name=spec["tools"][op], description=spec["desc"][op],
                                 annotations=ToolAnnotations(**hints[op]) if spec["annotate"] else None)

    def _named(self, nm: str):
        return [rec for rec in self.records if fold(rec[self.spec["name"]]) == fold(nm)]

    def check(self, kind: str, spec: dict) -> bool | None:
        if kind == "store_exists":
            return bool(self._named(spec["name"]))
        if kind == "store_absent":
            return not self._named(spec["name"])
        if kind == "store_count":
            return len(self.records) == spec["n"]
        if kind == "store_field":
            recs = self._named(spec["name"])
            return bool(recs) and fold(recs[0].get(spec["field"], "")) == fold(spec["value"])
        return None
