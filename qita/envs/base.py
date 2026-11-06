"""Shared pieces of the deterministic MCP environments.

An environment is built fresh for every episode. It owns an MCP server (served
in-process to the harness), its own state, and the checks that read that state
once the episode is over.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

HOME = "/home/utente"
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
ROME = ZoneInfo("Europe/Rome")
EVAL_NOW = dt.datetime(2026, 11, 4, 10, 0, tzinfo=ROME)


@dataclass
class World:
    """What the envs show. None fields fall back to the eval fixtures (fixtures/, env SEEDs).

    Eval tasks always run in the fixture world. Training data runs in generated worlds
    (qita/worlds), so no training example reuses eval content.
    """

    now: dt.datetime = EVAL_NOW
    home: dict[str, str] | None = None        # relative path -> file content
    notes: list[dict] | None = None
    events: list[dict] | None = None
    pages: dict[str, dict] | None = None      # url -> {"title", "body"}
    store: dict | None = None                 # qita/worlds/store.py build_store(): the record store env

    @property
    def mtime(self) -> int:
        """Every file gets this mtime, so listings and dates do not depend on the checkout."""
        return int(self.now.timestamp())


def norm_path(p: str) -> str:
    """A path as the model wrote it, made absolute under HOME (~, relative, trailing /)."""
    p = str(p).strip()
    if p in ("", "~", "."):
        return HOME
    if p.startswith("~/"):
        p = HOME + p[1:]
    elif not p.startswith("/"):
        p = HOME + "/" + p
    return os.path.normpath(p)


class Sandbox:
    """A private home (fixtures/home, or the world's files) that the model sees as /home/utente."""

    def __init__(self, workdir: Path, world: World):
        self.root = workdir / "home"
        if world.home is None:
            shutil.copytree(FIXTURES / "home", self.root)
        else:
            self.root.mkdir(parents=True)
            for rel, content in world.home.items():
                if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
                    raise ValueError(f"world file outside home: {rel}")
                p = self.root.joinpath(*PurePosixPath(rel).parts)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
        t = world.mtime
        for dirpath, dirnames, filenames in os.walk(self.root):
            for name in dirnames + filenames:
                os.utime(Path(dirpath) / name, (t, t))
        os.utime(self.root, (t, t))
        os.utime(self.root.parent, (t, t))  # `ls -la ~` lists the parent too

    def resolve(self, path: str) -> Path:
        """Map a path the model wrote to a real path inside the sandbox, or refuse it."""
        p = (path or "").strip()
        if p in ("", "~", "."):
            p = HOME
        elif p.startswith("~/"):
            p = HOME + p[1:]
        elif not p.startswith("/"):
            p = HOME + "/" + p
        pure = PurePosixPath(os.path.normpath(p))
        if pure != PurePosixPath(HOME) and PurePosixPath(HOME) not in pure.parents:
            raise ToolError(f"Access denied - path outside allowed directories: {path} not in {HOME}")
        return self.root.joinpath(*pure.relative_to(HOME).parts)

    def shown(self, real: Path) -> str:
        """The path as the model should see it."""
        rel = real.relative_to(self.root)
        return HOME if str(rel) == "." else f"{HOME}/{rel.as_posix()}"

    def check(self, kind: str, spec: dict) -> bool | None:
        if kind == "file_exists":
            return self.resolve(spec["path"]).is_file()
        if kind == "file_absent":
            return not self.resolve(spec["path"]).exists()
        if kind == "dir_exists":
            return self.resolve(spec["path"]).is_dir()
        if kind == "file_contains":
            p = self.resolve(spec["path"])
            if not p.is_file():
                return False
            text = p.read_text(encoding="utf-8").casefold()
            return all(t.casefold() in text for t in spec["text"])
        if kind == "file_equals":
            # exact content, every other line untouched; only line endings and the final newline are normalized
            p = self.resolve(spec["path"])
            want = spec["content"]["any"] if isinstance(spec["content"], dict) else [spec["content"]]
            return p.is_file() and _lines(p.read_text(encoding="utf-8")) in [_lines(w) for w in want]
        if kind == "json_equals":
            p = self.resolve(spec["path"])
            try:
                return p.is_file() and json.loads(p.read_text(encoding="utf-8")) == spec["value"]
            except ValueError:
                return False
        if kind == "file_lacks":
            p = self.resolve(spec["path"])
            text = p.read_text(encoding="utf-8").casefold() if p.is_file() else ""
            return p.is_file() and not any(t.casefold() in text for t in spec["text"])
        return None


def _lines(s: str) -> list[str]:
    return s.replace("\r\n", "\n").rstrip("\n").split("\n")


class Workspace:
    """Per-episode state shared by the envs of one task (the file and shell envs see one home)."""

    def __init__(self, workdir: Path, world: World | None = None):
        self.workdir = workdir
        self.world = world or World()
        self._sandbox: Sandbox | None = None

    @property
    def sandbox(self) -> Sandbox:
        if self._sandbox is None:
            self._sandbox = Sandbox(self.workdir, self.world)
        return self._sandbox


class Env:
    """Base class: subclasses register tools on self.server in build()."""

    name = "env"
    title = "env"

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.server = MCPServer(self.title)
        self.build()

    def build(self) -> None:
        raise NotImplementedError

    def check(self, kind: str, spec: dict) -> bool | None:
        """Answer one state check, or None if this env does not know the kind."""
        return None


__all__ = ["EVAL_NOW", "Env", "HOME", "ROME", "Sandbox", "ToolError", "World", "Workspace"]
