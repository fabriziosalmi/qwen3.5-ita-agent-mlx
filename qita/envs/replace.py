"""Replace every occurrence of a text in a file: a tool the harness adds (flag "replace"), never named by a task.

Asks only for the old and the new text, not for the surrounding line, which a small model often rewrites wrong.
"""

from __future__ import annotations

import difflib

from .base import Env, ToolError


def reindent(text: str, old: str, new: str) -> str | None:
    """Replace the lines of 'old' when they match one place of 'text' with only the leading spaces different.

    The new lines get the indentation of the file: what the file adds in front of the first old line is added
    in front of every new line. No match, or more than one, gives None: replacing a guess is worse than failing.
    """
    lines, want = text.split("\n"), old.strip("\n").split("\n")
    if not old.strip() or len(want) > len(lines):
        return None
    key = [w.strip() for w in want]
    hits = [i for i in range(len(lines) - len(want) + 1) if [x.strip() for x in lines[i:i + len(want)]] == key]
    if len(hits) != 1:
        return None
    i = hits[0]
    have, gave = _indent(lines[i]), _indent(want[0])
    extra = have[:max(0, len(have) - len(gave))]  # what the file has in front of the first line and old does not
    repl = [extra + x if x.strip() else x for x in new.strip("\n").split("\n")]
    return "\n".join(lines[:i] + repl + lines[i + len(want):])


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


class ReplaceEnv(Env):
    name = "replace"
    title = "replace"

    def build(self) -> None:
        sb = self.ws.sandbox
        s = self.server

        @s.tool()
        def replace_in_file(path: str, old: str, new: str) -> str:
            """Replace every occurrence of the text 'old' with 'new' in a file. Use the shortest text that
            changes: a word, a value, a symbol. Returns how many occurrences changed and a diff."""
            p = sb.resolve(path)
            if not p.is_file():
                raise ToolError(f"ENOENT: no such file or directory, '{path}'")
            before = p.read_text(encoding="utf-8")
            n = before.count(old) if old else 0
            note = ""
            if n:
                after = before.replace(old, new)
            else:
                after = reindent(before, old, new)
                if after is None:
                    raise ToolError(f"Text not found in {sb.shown(p)}: {old!r}")
                n, note = 1, " Matched ignoring the indentation at the start of the lines."
            p.write_text(after, encoding="utf-8")
            shown = sb.shown(p)
            diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), shown, shown))
            return f"Replaced {n} occurrence(s).{note}\n```diff\n{diff}```"

    def check(self, kind: str, spec: dict) -> bool | None:
        return self.ws.sandbox.check(kind, spec)
