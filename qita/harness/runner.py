"""Agent loop: one task, fresh envs, real MCP calls, a full trajectory as the result."""

from __future__ import annotations

import json
import re
import tempfile
import time
import unicodedata
from contextlib import AsyncExitStack
from pathlib import Path, PurePosixPath

from mcp import Client

from ..envs import ENVS, Workspace
from ..envs.base import ToolError, norm_path
from .decision import CHANGE_VERBS, DELETE_VERBS, DESTRUCTIVE, verbs
from .request import changes, condition
from .tables import csv_totals, dated

SYSTEM = (
    "Sei un assistente che lavora nel terminale dell'utente e può usare gli strumenti collegati. "
    "Oggi è mercoledì 4 novembre 2026, sono le 10:00, fuso orario Europe/Rome. "
    "La cartella home dell'utente è /home/utente."
)
MAX_STEPS = 8  # model calls per user turn
LOOP_REPEATS = 3  # the same call this many times in one episode is a loop
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"
# Harness flags, after the base name: "intent+guard".
#   guard  write_file on an existing file not read in full in this episode is refused, so content is never lost
#   append the harness adds append_to_file (qita/envs/append.py) to every task that has the files env
#   calc   the harness adds calcola and aggiungi_giorni (qita/envs/calc.py) to every task
#   replace the harness adds replace_in_file (qita/envs/replace.py) to every task that has the files env
#   noedit with replace, edit_file is taken away, so replace_in_file is the only in-place edit
#   lint   after a full read, a write_file, edit_file or append_to_file of a .json or .py file (not replace_in_file),
#          the tool result says whether its syntax
#          is valid and, if not, where and why (parsed only, never executed)
#   count  after a full read, the tool result says how many lines the file has
#   verify a final answer that claims a change, in a turn where no tool that changes state succeeded, gets one
#          reminder that nothing changed; the model may then act or correct the answer
#   gate   when the model's own intent for the turn is "conferma" or "chiedi", tools that delete or change
#          data are refused in that turn: the user has to answer first
#   hide   in those turns the same tools are also left out of the tool list the model sees (refused if called)
#   cal    the [CONTESTO] header adds the next seven days with weekday and ISO date, and the end of the month
#   cal2   the same calendar starting from today, labelled as a calendar and not the user's data
#   named  delete_file and write_file over an existing file are refused when the user has not named that file
#          (its name, its name without extension, or every word of that name) in the conversation, nor answered a
#          turn whose reply named it: the model has to list what it would touch and ask. A second refusal in the
#          turn ends it with a question written by the harness. Independent of the intent label. Moves are
#          let through: they lose nothing, and refusing them cost tasks on dev
#   csvsum after a full read of a .csv or .tsv file, the tool result adds the row count, the total and the
#          maximum of each numeric column and the totals per value of each short text column, computed in
#          code; phone numbers and codes are not numbers
#   logtail after a full read of a file whose lines start with a date, the tool result says which way the dates
#          run and repeats the three newest lines
#   write  the "rispondi" rule asks for a requested text (message, mail) in full instead of one or two sentences
#   week   the calendar line adds the Monday and Sunday of next week
#   noop   replace_in_file with old equal to new is refused, as is an edit_file whose edits change nothing
#   cond   when the user's turn makes a change depend on a condition ("Se piove, spostalo"), a call that changes
#          state is refused until a call to another env has succeeded in that turn (the check of the condition);
#          a second refusal ends the turn with a question written by the harness
#   todo   in a "strumenti" turn that asks for a change, with no condition and nothing to remove, a final answer
#          given while no call that changes state has succeeded, and that asks the user nothing, gets one reminder
#          quoting the request; that first answer stays in the turn's final answer, before the one after the reminder
#   when   the calendar only in turns where the conversation names a day by weekday or a relative word
#          ("venerdì", "domani", "prossimi giorni") and the user's turn writes no date of its own
#   unseen tools of a server outside this repo's own envs (KNOWN_ENVS: any other MCP server) that change data, known by their MCP annotations or a
#          verb in their name (deletes, alters), count as writes for gate, hide, todo and cond; one that deletes is
#          refused unless the user named one of its text arguments, or answered a turn whose reply named it (the
#          rule of "named", for keys and ids instead of files); a second refusal ends the turn with a question
#   toolcls the intent classifier sees the connected tools (name and first sentence of the description) and the
#          rule that a request about their data is not "rispondi"
# "agent" is the default harness of the agent; flags after it add to it: "agent+count".
# The release models were measured with "intent+append+lint+replace+noedit+gate+hide+cal2" (README.md).
AGENT = "intent+append+lint+replace+noedit+gate+hide+cal2+named+csvsum+noop+logtail+write+week+todo+unseen+toolcls"
HARNESS = re.compile(r"(intent|agent)(\+(guard|append|lint|count|verify|replace|noedit|gate|hide|cal|cal2|when|calc|named|csvsum|noop|logtail|write|week|cond|todo|unseen|toolcls))*")


def expand(harness: str | None) -> str | None:
    """The harness string with "agent" replaced by the flags it stands for."""
    if harness and (harness == "agent" or harness.startswith("agent+")):
        return AGENT + harness[len("agent"):]
    return harness


GATED = DESTRUCTIVE | {"edit_file", "append_to_file", "replace_in_file"}
GATE = ("Errore: in questo turno serve prima la risposta dell'utente. Non cancellare e non modificare niente: "
        "elenca cosa verrebbe toccato e chiedi conferma, oppure chiedi il dato che manca.")
LINTED = ("read_text_file", "write_file", "edit_file", "append_to_file")
WRITES = DESTRUCTIVE | {"edit_file", "append_to_file", "replace_in_file", "create_directory", "aggiungi_nota"}
CLAIM = re.compile(r"\b(ho|abbiamo|è stat[oa]|sono stat[ie]|è|sono)\s+(già\s+|ora\s+|stat[oaie]\s+)?"
                   r"(modificat|cambiat|corrett|aggiunt|sostituit|cancellat|eliminat|spostat|rinominat|creat|scritt|"
                   r"aggiornat|ripristinat|riportat|tornat|rimess|sistemat|impostat|salvat)", re.IGNORECASE)
NOTHING_CHANGED = ("[CONTROLLO] In questo turno nessuno strumento che modifica i dati è andato a buon fine, quindi non è "
                   "cambiato niente. Se la richiesta chiede una modifica, falla adesso con lo strumento giusto; "
                   "altrimenti correggi la risposta.")
JSON_MSG = {"Expecting ',' delimiter": "manca una virgola", "Expecting ':' delimiter": "mancano i due punti",
            "Expecting value": "manca un valore (o c'è una virgola di troppo)",
            "Expecting property name enclosed in double quotes": "manca un nome tra virgolette doppie (o c'è una virgola di troppo)",
            "Unterminated string starting at": "virgolette non chiuse", "Extra data": "c'è testo dopo la fine del JSON"}


def lint(path: str, text: str) -> str | None:
    """A one-line syntax verdict for .json and .py files, in Italian; None for other files."""
    import ast
    if path.endswith(".json"):
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            return (f"[controllo sintassi] JSON non valido alla riga {e.lineno}, colonna {e.colno}: "
                    f"{JSON_MSG.get(e.msg, e.msg)}.")
        return "[controllo sintassi] JSON valido."
    if path.endswith(".py"):
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"[controllo sintassi] Python non valido alla riga {e.lineno}: {e.msg}."
        return "[controllo sintassi] sintassi Python valida."
    return None
UNNAMED = ("Errore: l'utente non ha nominato {name}. Prima di cancellare o sovrascrivere un file "
           "elenca cosa verrebbe toccato e chiedi conferma all'utente.")
ASK = {"delete_file": "Prima di cancellare {path} mi serve la tua conferma: lo cancello?",
       "write_file": "Prima di sovrascrivere {path} mi serve la tua conferma: lo sovrascrivo?"}
COND = ("Errore: la richiesta dice «{cond}». Prima di modificare qualcosa controlla con uno strumento se è vero e "
        "dillo all'utente; poi modifica solo se è vero.")
COND_ASK = "Non sono riuscito a verificare «{cond}»: vuoi che proceda lo stesso?"
TODO = ("[CONTROLLO] L'utente ha chiesto: «{ask}». In questo turno nessuna modifica è andata a buon fine. "
        "Se si può fare, fallo adesso con lo strumento giusto; se no, spiega all'utente perché.")
NOOP = "Errore: il testo nuovo è uguale a quello vecchio, quindi non cambierebbe niente. Scrivi il testo già modificato."


def target(call: dict, exists) -> str | None:
    """The existing file a call would delete or overwrite; None for any other call, or a missing file (the
    tool's own "not found" error tells the model more than a refusal)."""
    a = call["arguments"] if isinstance(call["arguments"], dict) else {}
    path = str(a.get("path", ""))
    if call["name"] in ("delete_file", "write_file") and exists is not None and exists(path):
        return path
    return None


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def named(path: str, texts: list[str]) -> bool:
    """True when one of the texts names the file: its name, its name without extension (3+ letters), or
    every word of 3+ letters in that name ("la scansione della carta d'identità" for
    carta-identita-scansione.jpg). Case and accents are ignored."""
    name = fold(PurePosixPath(norm_path(path)).name)
    stem = PurePosixPath(name).stem
    phrases = [name] + ([stem] if len(stem) >= 3 else [])
    parts = [w for w in re.split(r"[\W_]+", stem) if len(w) >= 3]
    has = lambda w, t: re.search(rf"(?<!\w){re.escape(w)}(?!\w)", t)  # noqa: E731
    return any(any(has(w, t) for w in phrases) or (parts and all(has(w, t) for w in parts))
               for t in map(fold, texts))


def noop(call: dict) -> bool:
    a = call["arguments"] if isinstance(call["arguments"], dict) else {}
    if call["name"] == "replace_in_file":
        return a.get("old") == a.get("new")
    if call["name"] == "edit_file":
        edits = a.get("edits") or []
        return bool(edits) and all(isinstance(e, dict) and e.get("oldText") == e.get("newText") for e in edits)
    return False


UNNAMED_ID = ("Errore: l'utente non ha nominato «{name}». Prima di cancellare elenca cosa verrebbe toccato e chiedi "
              "conferma all'utente.")
ASK_ID = "Prima di cancellare «{name}» mi serve la tua conferma: procedo?"


# The envs whose tools have rules written by hand above (WRITES, GATED, named...): "unseen" leaves them alone.
KNOWN_ENVS = {n for n, e in ENVS.items() if not getattr(e, "generic", False)}


def deletes(name: str, hint: dict | None = None) -> bool:
    """The tool deletes data: its annotations say so, or a verb of its name does (a read-only hint wins)."""
    hint = hint or {}
    if hint.get("readOnlyHint"):
        return False
    return bool(hint.get("destructiveHint")) or bool(verbs(name) & DELETE_VERBS)


def alters(name: str, hint: dict | None = None) -> bool:
    """The tool changes data: annotations first, then the verbs of its name."""
    hint = hint or {}
    if hint.get("readOnlyHint"):
        return False
    return deletes(name, hint) or hint.get("readOnlyHint") is False or bool(verbs(name) & CHANGE_VERBS)


def id_args(call: dict) -> list[str]:
    """The text arguments that can name what a call acts on: 3+ characters with a letter, longest first."""
    a = call["arguments"] if isinstance(call["arguments"], dict) else {}
    vals = [v.strip() for v in a.values() if isinstance(v, str)]
    return sorted({v for v in vals if len(v) >= 3 and re.search(r"[^\W\d_]", v)}, key=lambda v: (-len(v), v))


def named_id(value: str, texts: list[str]) -> bool:
    """A key or id named in the texts, by the rules of named(): the whole value or its last part after ':' or '/'
    ("repo:transformer-ita" is named by "transformer-ita")."""
    last = re.split(r"[:/]", value.rstrip(":/"))[-1]
    return any(named(v, texts) for v in {value, last} if v)


GUARD = ("Errore: il file esiste già e non l'hai letto. write_file sostituisce tutto il contenuto: "
         "leggi prima il file con read_text_file, oppure usa edit_file per cambiare solo una parte.")


def split_output(text: str) -> tuple[str, list[str], bool]:
    """Return (prose before the first call, raw call bodies, well_formed)."""
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    if TOOL_START not in text:
        return text.strip(), [], True
    prose, rest = text.split(TOOL_START, 1)
    bodies, ok = [], True
    for chunk in (TOOL_START + rest).split(TOOL_START)[1:]:
        if TOOL_END not in chunk:
            ok = False
            bodies.append(chunk.strip())
            continue
        body, after = chunk.split(TOOL_END, 1)
        bodies.append(body.strip())
        if after.strip():
            ok = False  # text after a call: the template forbids it
    return prose.strip(), bodies, ok


def result_text(res) -> tuple[str, bool]:
    parts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
    return "\n".join(parts), bool(res.is_error)


async def _connect(envs, stack: AsyncExitStack) -> tuple[dict, list[dict], dict]:
    """One MCP client per env; the tool schemas in env order, as the model sees them; the MCP annotations of each
    tool and the env that serves it (kept apart: the model never sees them)."""
    owner, tools, hints = {}, [], {}
    for env in envs:
        client = await stack.enter_async_context(Client(env.server))
        for t in (await client.list_tools()).tools:
            if t.name in owner:
                raise ValueError(f"tool name clash: {t.name}")
            owner[t.name] = client
            tools.append({"type": "function", "function": {
                "name": t.name, "description": t.description or "",
                "parameters": t.input_schema}})
            hints[t.name] = {"env": env.name}
            if getattr(t, "annotations", None) is not None:
                hints[t.name] |= t.annotations.model_dump(exclude_none=True, by_alias=True)
    return owner, tools, hints


async def env_tools(names: list[str]) -> list[dict]:
    """The schemas of these envs exactly as run_task offers them, e.g. to add a harness tool to training data."""
    with tempfile.TemporaryDirectory(prefix="qita-") as tmp:
        ws = Workspace(Path(tmp), None)
        async with AsyncExitStack() as stack:
            return (await _connect([ENVS[n](ws) for n in names], stack))[1]


async def run_task(task: dict, backend, system: str | None = SYSTEM, world=None, harness: str | None = None) -> dict:
    harness = expand(harness)
    with tempfile.TemporaryDirectory(prefix="qita-") as tmp:
        ws = Workspace(Path(tmp), world)
        names, opts = list(task["envs"]), set((harness or "").split("+")[1:])
        for flag in ("append", "replace"):
            if flag in opts and "files" in names:
                names.append(flag)
        if "noedit" in opts and "replace" in names and "edit" in names:
            names.remove("edit")
        if "calc" in opts and "calc" not in names:
            names.append("calc")
        envs = [ENVS[n](ws) for n in names]
        async with AsyncExitStack() as stack:
            owner, tools, hints = await _connect(envs, stack)
            def exists(path: str) -> bool:
                try:
                    return ws.sandbox.resolve(path).exists()
                except ToolError:
                    return False

            def peek(path: str) -> str | None:
                try:
                    p = ws.sandbox.resolve(path)
                    return p.read_text(encoding="utf-8") if p.is_file() else None
                except (ToolError, UnicodeDecodeError):
                    return None
            traj = await _episode(task, backend, tools, owner, system, harness, exists, peek, ws.world.now, hints)
            traj["state"] = _state_checks(task, envs)
            traj["tools"] = tools
            return traj


async def _episode(task: dict, backend, tools: list[dict], owner: dict, system: str | None,
                   harness: str | None = None, exists=None, peek=None, now=None, hints: dict | None = None) -> dict:
    base, *opts = (expand(harness) or "").split("+")
    harness, opts = base or None, set(opts)
    hints = hints or {}
    # "unseen": a tool of any server that changes data counts as a write, and one that deletes is guarded
    unseen = lambda name: "unseen" in opts and hints.get(name, {}).get("env") not in KNOWN_ENVS  # noqa: E731
    writes = lambda name: name in WRITES or (unseen(name) and alters(name, hints.get(name)))  # noqa: E731
    gates = lambda name: name in GATED or (unseen(name) and alters(name, hints.get(name)))  # noqa: E731
    known: set[str] = set()  # files whose whole content the model has seen (read in full, or written by it)
    system = task.get("system", system)
    messages = [{"role": "system", "content": system}] if system else []
    calls, turns, flags = [], [], {"malformed": 0, "loop": False, "step_cap": False, "max_tokens": 0}
    seen: dict[str, int] = {}
    fault = dict(task["faults"]) if task.get("faults") else None  # transient errors to recover from
    gen_tokens = prompt_tokens = reused_tokens = 0
    seconds = classify_seconds = 0.0
    every_tool = tools
    for ti, turn in enumerate(task["turns"]):
        intent, tools, turn_classify = None, every_tool, 0.0
        if harness in ("intent", "intent_gold"):
            from .intent import calendar_line, classify, wants_calendar, wrap
            if harness == "intent_gold" and turn.get("decision"):
                intent = turn["decision"]  # data generation: the header carries the correct decision
            else:
                previous = [("Utente", u["user"]) for u in task["turns"][:ti]]
                previous = [(w, m) for pair in zip(previous, [("Assistente", tt["final"] or "") for tt in turns])
                            for (w, m) in pair]
                t0 = time.perf_counter()
                intent, _ = classify(backend, turn["user"], previous, every_tool if "toolcls" in opts else None)
                turn_classify = time.perf_counter() - t0  # its own prompt: never in the agent's prompt cache
                classify_seconds += turn_classify
            # [CONTESTO] shows the visible system message: "context" overrides it when the real one carries hidden hints
            ctx = task["context"] if "context" in task else system
            if "hide" in opts and intent in ("conferma", "chiedi"):
                tools = [t for t in every_tool if not gates(t["function"]["name"])]
            cal = None if now is None else (calendar_line(now, v=2, week="week" in opts) if "cal2" in opts
                                             else calendar_line(now) if "cal" in opts else None)
            if "when" in opts and not wants_calendar([u["user"] for u in task["turns"][:ti + 1]]):
                cal = None
            messages.append({"role": "user", "content": wrap(turn["user"], intent, ctx, tools, cal, "write" in opts)})
        else:
            messages.append({"role": "user", "content": turn["user"]})
        final, n_calls, reminded, refused, ask, kept = None, 0, False, {}, None, ""
        if_ = condition(turn["user"]) if {"cond", "todo"} & opts else None
        cond = if_ if "cond" in opts else None
        wanted = changes(turn["user"]) if "todo" in opts and intent == "strumenti" and not if_ else []
        unchecked = 0
        for _step in range(MAX_STEPS):
            out = backend.complete(messages, tools)
            gen_tokens += out["gen_tokens"]
            prompt_tokens += out.get("prompt_tokens", 0)
            reused_tokens += out.get("reused_tokens", 0)
            seconds += out["seconds"]
            flags["max_tokens"] += int(out["hit_max_tokens"])
            prose, bodies, ok = split_output(out["text"])
            parsed = []
            for b in bodies:
                try:
                    parsed.append(backend.parse_tool_call(b, every_tool))  # a hidden tool is refused, not malformed
                except Exception:  # noqa: BLE001 - any parser failure is a malformed call
                    ok = False
            if not ok or (bodies and not parsed):
                flags["malformed"] += 1
                final = out["text"].strip()
                messages.append({"role": "assistant", "content": final})
                break
            if not parsed:
                final = "\n\n".join(x for x in (kept, prose) if x)  # the answer before a todo reminder stays
                messages.append({"role": "assistant", "content": prose})
                changed = any(c["turn"] == ti and writes(c["name"]) and not c["error"] for c in calls)
                if "verify" in opts and not reminded and not changed and CLAIM.search(prose or ""):
                    reminded = True
                    flags["reminders"] = flags.get("reminders", 0) + 1
                    messages.append({"role": "user", "content": NOTHING_CHANGED})
                    continue
                if wanted and not reminded and not changed and "?" not in (prose or ""):  # a question waits for the user
                    reminded = True
                    flags["reminders"] = flags.get("reminders", 0) + 1
                    messages.append({"role": "user", "content": TODO.format(ask=" ".join(wanted))})
                    kept = prose or ""
                    continue
                break
            messages.append({"role": "assistant", "content": prose, "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in parsed]})
            for c in parsed:
                key = json.dumps([c["name"], c["arguments"]], sort_keys=True, ensure_ascii=False)
                seen[key] = seen.get(key, 0) + 1
                injected = bool(fault and fault["tool"] in (None, c["name"]) and fault["times"] > 0)
                path = str(c["arguments"].get("path", "")) if isinstance(c["arguments"], dict) else ""
                guarded = ("guard" in opts and c["name"] == "write_file" and exists is not None and exists(path)
                           and norm_path(path) not in known)
                gated = bool({"gate", "hide"} & opts) and intent in ("conferma", "chiedi") and gates(c["name"])
                aim = target(c, exists) if "named" in opts else None
                unnamed = bool(aim) and not named(aim, [u["user"] for u in task["turns"][:ti + 1]]
                                                  + [t["final"] or "" for t in turns])
                ids = id_args(c) if unseen(c["name"]) and deletes(c["name"], hints.get(c["name"])) else []
                texts = [u["user"] for u in task["turns"][:ti + 1]] + [t["final"] or "" for t in turns]
                unnamed_id = bool(ids) and not any(named_id(v, texts) for v in ids)
                early = bool(cond) and writes(c["name"]) and not any(
                    x["turn"] == ti and not x["error"] and x["name"] in owner and not writes(x["name"])
                    and owner[x["name"]] is not owner.get(c["name"]) for x in calls)
                faulted = nooped = False
                if gated:
                    text, err = GATE, True
                elif early:
                    text, err = COND.format(cond=cond), True
                    unchecked += 1
                    if unchecked >= 2 and ask is None:
                        ask = COND_ASK.format(cond=cond)
                elif unnamed:
                    text, err = UNNAMED.format(name=PurePosixPath(norm_path(aim)).name), True
                    refused[norm_path(aim)] = refused.get(norm_path(aim), 0) + 1
                    if refused[norm_path(aim)] >= 2 and ask is None:
                        ask = ASK[c["name"]].format(path=norm_path(aim))
                elif unnamed_id:
                    text, err = UNNAMED_ID.format(name=ids[0]), True
                    refused[ids[0]] = refused.get(ids[0], 0) + 1
                    if refused[ids[0]] >= 2 and ask is None:
                        ask = ASK_ID.format(name=ids[0])
                elif "noop" in opts and noop(c):
                    text, err, nooped = NOOP, True, True
                elif guarded:
                    text, err = GUARD, True
                elif injected:
                    text, err, faulted = fault["error"], True, True
                    fault["times"] -= 1
                elif c["name"] not in owner:
                    text, err = f"Error: tool '{c['name']}' does not exist", True
                else:
                    res = await owner[c["name"]].call_tool(c["name"], c["arguments"])
                    text, err = result_text(res)
                raw = text  # the tool's own result, before the harness appends to it
                if ("lint" in opts and not err and peek is not None and c["name"] in LINTED
                        and not {"head", "tail"} & set(c["arguments"])):
                    content = peek(path)
                    verdict = lint(norm_path(path), content) if content is not None else None
                    if verdict:
                        text = f"{text}\n{verdict}"
                if ("csvsum" in opts and not err and peek is not None and c["name"] == "read_text_file"
                        and path.lower().endswith((".csv", ".tsv")) and not {"head", "tail"} & set(c["arguments"])):
                    content = peek(path)
                    totals = csv_totals(content) if content is not None else None
                    if totals:
                        text = f"{text}\n{totals}"
                if ("logtail" in opts and not err and peek is not None and c["name"] == "read_text_file"
                        and not {"head", "tail"} & set(c["arguments"])):
                    content = peek(path)
                    order = dated(content) if content is not None else None
                    if order:
                        text = f"{text}\n{order}"
                if ("count" in opts and not err and c["name"] == "read_text_file"
                        and not {"head", "tail"} & set(c["arguments"])):
                    n = len(raw.splitlines())
                    text = f"{text}\n[il file ha {n} righe]"
                if not err and (c["name"] == "write_file" or (c["name"] == "read_text_file" and not
                                                              {"head", "tail"} & set(c["arguments"]))):
                    known.add(norm_path(path))
                calls.append({"turn": ti, "name": c["name"], "arguments": c["arguments"], "error": err,
                              "unknown": c["name"] not in owner, "fault": faulted, "guard": guarded,
                              "gated": bool(gated or unnamed or early or unnamed_id or nooped),
                              "result": text[:2000]})
                n_calls += 1
                messages.append({"role": "tool", "content": text})
            if ask:  # refused twice (an unnamed file, an unchecked condition): the harness asks the user
                final = ask
                flags["asked"] = flags.get("asked", 0) + 1
                messages.append({"role": "assistant", "content": ask})
                break
            if max(seen.values()) >= LOOP_REPEATS:
                flags["loop"] = True
                break
        else:
            flags["step_cap"] = True
        turns.append({"final": final, "n_calls": n_calls, "intent": intent, "classify_seconds": round(turn_classify, 3)})
        if flags["loop"] or flags["step_cap"]:
            break
    return {"id": task["id"], "category": task["category"], "system": system, "messages": messages, "calls": calls,
            "turns": turns, "flags": flags, "gen_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens, "reused_tokens": reused_tokens, "seconds": round(seconds, 2),
            "classify_seconds": round(classify_seconds, 2)}


def _state_checks(task: dict, envs) -> list[dict]:
    out = []
    for spec in task.get("checks", {}).get("state", []):
        spec = dict(spec)
        kind = spec.pop("kind")
        val = None
        for env in envs:
            val = env.check(kind, spec)
            if val is not None:
                break
        if val is None:
            raise ValueError(f"{task['id']}: no env answers state check '{kind}'")
        out.append({"kind": kind, "spec": spec, "ok": bool(val)})
    return out
