"""In-place text edits over the sandbox home, modelled on edit_file of the MCP reference filesystem server.

A separate env, so the tool lists of dev and test tasks (envs: [files]) do not change.
"""

from __future__ import annotations

import difflib

from pydantic import BaseModel

from .base import Env, ToolError


class Edit(BaseModel):
    oldText: str  # noqa: N815 - the reference server's field names
    newText: str  # noqa: N815


class EditEnv(Env):
    name = "edit"
    title = "editor"

    def build(self) -> None:
        sb = self.ws.sandbox
        s = self.server

        @s.tool()
        def edit_file(path: str, edits: list[Edit], dryRun: bool = False) -> str:  # noqa: N803
            """Make line-based edits to a text file. Each edit replaces the first exact occurrence of
            oldText with newText. Returns a git-style diff of the changes."""
            p = sb.resolve(path)
            if not p.is_file():
                raise ToolError(f"ENOENT: no such file or directory, '{path}'")
            before = p.read_text(encoding="utf-8")
            after = before
            for e in edits:
                e = e if isinstance(e, Edit) else Edit(**e)
                if not e.oldText or e.oldText not in after:
                    raise ToolError(f"Could not find exact match for edit:\n{e.oldText}")
                after = after.replace(e.oldText, e.newText, 1)
            shown = sb.shown(p)
            diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), shown, shown))
            if not dryRun:
                p.write_text(after, encoding="utf-8")
            return f"```diff\n{diff}```"

    def check(self, kind: str, spec: dict) -> bool | None:
        return self.ws.sandbox.check(kind, spec)
