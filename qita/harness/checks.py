"""Scoring of a trajectory against the checks of its task.

Argument matchers (values in a call spec):
  scalar                  equal after strip + casefold (numbers compare as numbers)
  {path: "/home/utente/x"} same path once ~, relative and trailing / are normalized
  {any: [m1, m2]}         any of the nested matchers
  {re: "..."}             regex search, case-insensitive
  {contains: s | [s..]}   every substring present, case-insensitive
  {num: x, tol: t}        numeric within tolerance (default 0.005)
  {date: "2026-11-02"}    the value starts with this ISO date
Arguments not named in the spec are not checked.
"""

from __future__ import annotations

import re
import unicodedata

from ..envs.base import norm_path
from .dates import named_days  # noqa: F401  (check_answer's "days")
from .italian import is_italian


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def match_value(spec, value) -> bool:
    if isinstance(spec, dict):
        if "any" in spec:
            return any(match_value(s, value) for s in spec["any"])
        if value is None:
            return False
        if "path" in spec:
            return norm_path(value) == norm_path(spec["path"])
        if "re" in spec:
            return re.search(spec["re"], str(value), re.IGNORECASE) is not None
        if "contains" in spec:
            want = spec["contains"] if isinstance(spec["contains"], list) else [spec["contains"]]
            return all(norm_text(w) in norm_text(value) for w in want)
        if "num" in spec:
            try:
                return abs(float(str(value).replace(",", ".")) - float(spec["num"])) <= spec.get("tol", 0.005)
            except ValueError:
                return False
        if "date" in spec:
            return str(value).strip().startswith(spec["date"])
        raise ValueError(f"unknown matcher {spec}")
    if isinstance(spec, list):
        return isinstance(value, list) and sorted(map(norm_text, spec)) == sorted(map(norm_text, value))
    if isinstance(spec, bool) or isinstance(value, bool):
        return str(spec).casefold() == str(value).casefold()
    if isinstance(spec, (int, float)):
        try:
            return float(value) == float(spec)
        except (TypeError, ValueError):
            return False
    return norm_text(spec) == norm_text(value)


def match_call(spec: dict, call: dict) -> bool:
    if call["name"] != spec["tool"]:
        return False
    args = call["arguments"] or {}
    return all(match_value(v, args.get(k)) for k, v in (spec.get("args") or {}).items())


def check_answer(spec: dict, text: str | None) -> bool:
    t = norm_text(text or "")
    ok = all(norm_text(w) in t for w in spec.get("contains", []))
    for group in spec.get("any", []):
        group = group if isinstance(group, list) else [group]
        ok = ok and any(norm_text(w) in t for w in group)
    ok = ok and not any(norm_text(w) in t for w in spec.get("not", []))
    for n in spec.get("number", []):  # a whole number: "4" is not found in "14"
        ok = ok and re.search(rf"(?<!\d){re.escape(str(n))}(?!\d)", t) is not None
    if "days" in spec:  # every day the answer names is one of these (ISO dates)
        ok = ok and named_days(t) <= {(int(d[5:7]), int(d[8:10])) for d in spec["days"]}
    return ok


def call_checks(checks: dict, calls: list[dict], prefix: str) -> dict[str, bool]:
    out = {}
    # A call that hit an injected fault, or that the harness refused (gate, guard, named, cond, noop), did nothing:
    # it counts for no_call, never for a positive call check.
    tried, calls = calls, [c for c in calls if not (c.get("fault") or c.get("gated") or c.get("guard"))]
    if "calls" in checks:
        specs = checks["calls"]
        if checks.get("ordered"):
            i = 0
            for c in calls:
                if i < len(specs) and match_call(specs[i], c):
                    i += 1
            out[f"{prefix}calls_ordered"] = i == len(specs)
        else:
            for j, s in enumerate(specs):
                out[f"{prefix}call[{j}]:{s['tool']}"] = any(match_call(s, c) for c in calls)
    for j, group in enumerate(checks.get("calls_any", [])):
        out[f"{prefix}calls_any[{j}]"] = any(match_call(s, c) for s in group for c in calls)
    for j, s in enumerate(checks.get("no_call", [])):
        s = {"tool": s} if isinstance(s, str) else s
        out[f"{prefix}no_call[{j}]:{s['tool']}"] = not any(match_call(s, c) for c in tried)
    if checks.get("no_tools"):
        out[f"{prefix}no_tools"] = not tried
    if "max_calls" in checks:
        out[f"{prefix}max_calls"] = len(tried) <= checks["max_calls"]
    return out


def score(task: dict, traj: dict) -> dict:
    res: dict[str, bool] = {}
    finals = [t["final"] for t in traj["turns"]]
    for ti, turn in enumerate(task["turns"]):
        tc = turn.get("checks") or {}
        if not tc:
            continue
        p = f"t{ti}."
        if ti >= len(traj["turns"]):
            res[f"{p}reached"] = False
            continue
        res.update(call_checks(tc, [c for c in traj["calls"] if c["turn"] == ti], p))
        if tc.get("asks"):
            # The turn ends on a question. Looking things up first is fine; what must not
            # happen before the answer is written in no_call.
            res[f"{p}asks"] = "?" in (finals[ti] or "")
        if "answer" in tc:
            res[f"{p}answer"] = check_answer(tc["answer"], finals[ti])
    gc = task.get("checks") or {}
    res.update(call_checks(gc, traj["calls"], ""))
    if "answer" in gc:
        res["answer"] = check_answer(gc["answer"], finals[-1] if finals else None)
    for j, s in enumerate(traj.get("state", [])):
        res[f"state[{j}]:{s['kind']}"] = s["ok"]
    f = traj["flags"]
    res["all_turns_done"] = len(traj["turns"]) == len(task["turns"]) and all(fi for fi in finals)
    res["italian"] = all(is_italian(fi) for fi in finals if fi)
    res["well_formed"] = f["malformed"] == 0
    res["no_loop"] = not f["loop"] and not f["step_cap"]
    return {"checks": res, "success": all(res.values())}
