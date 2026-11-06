"""Filesystem server over the sandbox home, modelled on the MCP reference filesystem server."""

from __future__ import annotations

import datetime as dt
import fnmatch
import shutil

from .base import ROME, Env, ToolError


class FilesEnv(Env):
    name = "files"
    title = "filesystem"

    def build(self) -> None:
        sb = self.ws.sandbox
        s = self.server

        def existing(path: str):
            p = sb.resolve(path)
            if not p.exists():
                raise ToolError(f"ENOENT: no such file or directory, '{path}'")
            return p

        @s.tool()
        def list_directory(path: str) -> str:
            """Get a detailed listing of all files and directories in a specified path.
            Results distinguish files and directories with [FILE] and [DIR] prefixes."""
            p = existing(path)
            if not p.is_dir():
                raise ToolError(f"ENOTDIR: not a directory, '{path}'")
            rows = [f"[{'DIR' if c.is_dir() else 'FILE'}] {c.name}" for c in sorted(p.iterdir())]
            return "\n".join(rows) or "(empty directory)"

        @s.tool()
        def read_text_file(path: str, head: int | None = None, tail: int | None = None) -> str:
            """Read the complete contents of a text file. Use 'head' or 'tail' to read only
            the first or last N lines."""
            p = existing(path)
            if p.is_dir():
                raise ToolError(f"EISDIR: illegal operation on a directory, read '{path}'")
            lines = p.read_text(encoding="utf-8").splitlines()
            if head is not None and tail is not None:
                raise ToolError("Cannot specify both head and tail parameters simultaneously")
            if head is not None:
                lines = lines[:head]
            if tail is not None:
                lines = lines[-tail:] if tail > 0 else []
            return "\n".join(lines)

        @s.tool()
        def search_files(path: str, pattern: str) -> str:
            """Recursively search for files and directories whose name matches a pattern.
            The pattern is a glob (e.g. '*.txt') or a case-insensitive substring of the name."""
            base = existing(path)
            pat = pattern.lower()
            glob = any(ch in pat for ch in "*?[")
            hits = []
            for c in sorted(base.rglob("*")):
                n = c.name.lower()
                if (fnmatch.fnmatch(n, pat) if glob else pat in n):
                    hits.append(sb.shown(c))
            return "\n".join(hits) or "No matches found"

        @s.tool()
        def get_file_info(path: str) -> str:
            """Retrieve metadata about a file or directory: size, type and modification time."""
            p = existing(path)
            st = p.stat()
            mod = dt.datetime.fromtimestamp(st.st_mtime, ROME).strftime("%Y-%m-%d %H:%M:%S")
            kind = "directory" if p.is_dir() else "file"
            return f"type: {kind}\nsize: {st.st_size} bytes\nmodified: {mod}"

        @s.tool()
        def write_file(path: str, content: str) -> str:
            """Create a new file or completely overwrite an existing file with new content."""
            p = sb.resolve(path)
            if not p.parent.exists():
                raise ToolError(f"ENOENT: parent directory does not exist, '{path}'")
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {sb.shown(p)}"

        @s.tool()
        def create_directory(path: str) -> str:
            """Create a new directory, including parent directories. Succeeds if it already exists."""
            p = sb.resolve(path)
            p.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory {sb.shown(p)}"

        @s.tool()
        def move_file(source: str, destination: str) -> str:
            """Move or rename a file or directory. Fails if the destination already exists."""
            src = existing(source)
            dst = sb.resolve(destination)
            if dst.is_dir():
                dst = dst / src.name
            if dst.exists():
                raise ToolError(f"EEXIST: destination already exists, '{destination}'")
            if not dst.parent.exists():
                raise ToolError(f"ENOENT: parent directory does not exist, '{destination}'")
            shown_src = sb.shown(src)
            shutil.move(src, dst)
            return f"Successfully moved {shown_src} to {sb.shown(dst)}"

        @s.tool()
        def delete_file(path: str) -> str:
            """Permanently delete a file. Directories are not deleted. This cannot be undone."""
            p = existing(path)
            if p.is_dir():
                raise ToolError(f"EISDIR: is a directory, '{path}'")
            p.unlink()
            return f"Successfully deleted {sb.shown(p)}"

    def check(self, kind: str, spec: dict) -> bool | None:
        return self.ws.sandbox.check(kind, spec)
