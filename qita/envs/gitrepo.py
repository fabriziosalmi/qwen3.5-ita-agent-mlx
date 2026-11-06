"""Git server over an in-memory repository (test-only env), modelled on the MCP reference git server."""

from __future__ import annotations

import hashlib

from .base import Env, ToolError, norm_path

REPO = "/home/utente/progetti/sito"
COMMITS = [  # newest first
    ("Giulia Ferri", "2026-11-03 18:12", "Aggiorna i testi della pagina chi siamo", ["contenuti/chi-siamo.md"]),
    ("Marco Repetto", "2026-11-02 11:40", "Corregge il link al modulo di iscrizione", ["contenuti/iscrizioni.md"]),
    ("Giulia Ferri", "2026-10-30 16:05", "Aggiunge la pagina eventi", ["contenuti/eventi.md", "menu.yaml"]),
    ("Marco Repetto", "2026-10-28 09:21", "Configura il tema e i colori", ["tema/config.yaml"]),
    ("Sara Costa", "2026-10-27 14:55", "Aggiunge le foto della sede", ["immagini/sede-1.jpg", "immagini/sede-2.jpg"]),
    ("Marco Repetto", "2026-10-26 10:00", "Primo commit: struttura del sito", ["README.md", "menu.yaml"]),
]
DIFF = """diff --git a/contenuti/home.md b/contenuti/home.md
--- a/contenuti/home.md
+++ b/contenuti/home.md
@@ -1,4 +1,5 @@
 # Associazione Il Faro
-Siamo un'associazione culturale nata nel 2019.
+Siamo un'associazione culturale nata nel 2019 a Genova.
+Ogni primo sabato del mese organizziamo il mercatino in piazza Matteotti.

 ## Contatti"""


def sha(i: int) -> str:
    return hashlib.sha1(f"ilfaro-{i}".encode()).hexdigest()


class GitEnv(Env):
    name = "git"
    title = "git"

    def build(self) -> None:
        self.log = [{"sha": sha(len(COMMITS) - i), "author": a, "date": d, "message": m, "files": f}
                    for i, (a, d, m, f) in enumerate(COMMITS)]
        self.modified = ["contenuti/home.md"]
        self.untracked = ["immagini/sede-3.jpg"]
        self.staged: list[str] = []
        self.branch = "main"
        s = self.server

        def repo(path: str) -> None:
            if norm_path(path) != REPO:
                raise ToolError(f"fatal: not a git repository: '{path}'")

        @s.tool()
        def git_status(repo_path: str) -> str:
            """Shows the working tree status."""
            repo(repo_path)
            out = [f"On branch {self.branch}"]
            if self.staged:
                out += ["Changes to be committed:"] + [f"\tmodified:   {f}" for f in self.staged]
            unstaged = [f for f in self.modified if f not in self.staged]
            if unstaged:
                out += ["Changes not staged for commit:"] + [f"\tmodified:   {f}" for f in unstaged]
            untracked = [f for f in self.untracked if f not in self.staged]
            if untracked:
                out += ["Untracked files:"] + [f"\t{f}" for f in untracked]
            if len(out) == 1:
                out.append("nothing to commit, working tree clean")
            return "\n".join(out)

        @s.tool()
        def git_log(repo_path: str, max_count: int = 10) -> str:
            """Shows the commit logs, newest first."""
            repo(repo_path)
            return "\n\n".join(f"commit {c['sha']}\nAuthor: {c['author']}\nDate:   {c['date']}\n\n    {c['message']}"
                               for c in self.log[: max(1, max_count)])

        @s.tool()
        def git_diff_unstaged(repo_path: str) -> str:
            """Shows changes in the working directory that are not yet staged."""
            repo(repo_path)
            return DIFF if "contenuti/home.md" in self.modified and "contenuti/home.md" not in self.staged else ""

        @s.tool()
        def git_show(repo_path: str, revision: str) -> str:
            """Shows the contents of a commit (message and changed files)."""
            repo(repo_path)
            r = revision.strip()
            idx = None
            if r in ("HEAD", "head"):
                idx = 0
            elif r.startswith("HEAD~") and r[5:].isdigit():
                idx = int(r[5:])
            else:
                for i, c in enumerate(self.log):
                    if len(r) >= 7 and c["sha"].startswith(r):
                        idx = i
            if idx is None or idx >= len(self.log):
                raise ToolError(f"fatal: bad revision '{revision}'")
            c = self.log[idx]
            return (f"commit {c['sha']}\nAuthor: {c['author']}\nDate:   {c['date']}\n\n    {c['message']}\n\n"
                    + "\n".join(f" {f}" for f in c["files"]))

        @s.tool()
        def git_add(repo_path: str, files: list[str]) -> str:
            """Adds file contents to the staging area."""
            repo(repo_path)
            known = set(self.modified) | set(self.untracked)
            for f in files:
                if f in (".", "-A", "--all"):
                    self.staged = sorted(known)
                    continue
                if f not in known:
                    raise ToolError(f"fatal: pathspec '{f}' did not match any files")
                if f not in self.staged:
                    self.staged.append(f)
            return "Files staged successfully"

        @s.tool()
        def git_commit(repo_path: str, message: str) -> str:
            """Records staged changes to the repository."""
            repo(repo_path)
            if not self.staged:
                raise ToolError("nothing added to commit")
            c = {"sha": sha(len(self.log) + 1), "author": "Utente", "date": "2026-11-04 10:00",
                 "message": message, "files": list(self.staged)}
            self.log.insert(0, c)
            self.modified = [f for f in self.modified if f not in self.staged]
            self.untracked = [f for f in self.untracked if f not in self.staged]
            self.staged = []
            return f"Changes committed successfully with hash {c['sha']}"

    def check(self, kind: str, spec: dict) -> bool | None:
        if kind == "git_committed":
            want = [w.casefold() for w in spec.get("message_contains", [])]
            new = self.log[: len(self.log) - len(COMMITS)]
            return any(all(w in c["message"].casefold() for w in want)
                       and all(f in c["files"] for f in spec.get("files", []))
                       and not any(f in c["files"] for f in spec.get("exclude_files", [])) for c in new)
        if kind == "git_commit_count":
            return len(self.log) - len(COMMITS) == spec["n"]
        return None
