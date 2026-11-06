"""Append text at the end of a file: a tool the harness adds (flag "append"), never named by a task."""

from __future__ import annotations

from .base import Env, ToolError


class AppendEnv(Env):
    name = "append"
    title = "append"

    def build(self) -> None:
        sb = self.ws.sandbox
        s = self.server

        @s.tool()
        def append_to_file(path: str, content: str) -> str:
            """Append text at the end of a file, on new lines, keeping everything already in it.
            Creates the file if it does not exist."""
            p = sb.resolve(path)
            if not p.parent.exists():
                raise ToolError(f"ENOENT: parent directory does not exist, '{path}'")
            if p.is_dir():
                raise ToolError(f"EISDIR: illegal operation on a directory, '{path}'")
            before = p.read_text(encoding="utf-8") if p.is_file() else ""
            old, new = before.splitlines(), content.splitlines()
            repeated = bool(old) and new[:len(old)] == old and len(new) > len(old)
            if repeated:  # a small model often passes the whole file plus the new lines, as for write_file
                new = new[len(old):]
            sep = "\n" if before and not before.endswith("\n") else ""
            text = "\n".join(new) + "\n"
            p.write_text(before + sep + text, encoding="utf-8")
            note = " (the lines already in the file were not repeated)" if repeated else ""
            return f"Successfully appended {len(new)} line(s) to {sb.shown(p)}{note}"

    def check(self, kind: str, spec: dict) -> bool | None:
        return self.ws.sandbox.check(kind, spec)
