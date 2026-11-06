"""Terminal server: allowlisted commands with pipes, confined to the sandbox home."""

from __future__ import annotations

import grp
import os
import pwd
import re
import shlex
import subprocess
from pathlib import PurePosixPath

from .base import HOME, Env, ToolError

ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "grep", "find", "du", "sort", "uniq", "cut", "tr",
    "file", "echo", "mkdir", "touch", "cp", "mv", "rm", "basename", "dirname", "diff", "pwd",
}
BLOCKED = {
    "sudo", "su", "curl", "wget", "ssh", "scp", "rsync", "nc", "chmod", "chown", "dd", "mkfs",
    "kill", "killall", "shutdown", "reboot", "python", "python3", "bash", "sh", "zsh", "perl",
    "ruby", "node", "osascript", "open", "crontab", "launchctl", "eval", "exec",
}
FIND_BLOCKED = {"-exec", "-execdir", "-ok", "-okdir", "-delete"}
MAX_OUT = 4000
WEEKDAYS = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]
MONTHS = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]


def host_neutral(text: str) -> str:
    """Listings as on any machine: the owner and group of the host user become utente, and the macOS
    extended-attribute marker after the permissions goes (it differs between machines)."""
    user, group = pwd.getpwuid(os.getuid()).pw_name, grp.getgrgid(os.getgid()).gr_name
    text = re.sub(rf"(?<=\s){re.escape(user)}(\s+){re.escape(group)}(?=\s)", r"utente\1utente", text)
    return re.sub(r"^([-dlcbps][rwxsStT-]{9})[@+]", r"\1", text, flags=re.MULTILINE)


def date_text(now) -> str:
    """`date` as an Italian-locale macOS prints it, e.g. 'mer  4 nov 2026 10:00:00 CET'."""
    return (f"{WEEKDAYS[now.weekday()]} {now.day:2d} {MONTHS[now.month - 1]} {now.year} "
            f"{now:%H:%M:%S} {now.tzname()}")


class ShellEnv(Env):
    name = "shell"
    title = "terminale"

    def build(self) -> None:
        sb = self.ws.sandbox
        root = str(sb.root)

        def map_path(tok: str) -> str:
            if tok == "~" or tok.startswith("~/"):
                tok = HOME + tok[1:]
            if tok.startswith("/"):
                return str(sb.resolve(tok))
            if ".." in PurePosixPath(tok).parts:
                norm = os.path.normpath(os.path.join(HOME, tok))
                return str(sb.resolve(norm))
            return tok

        def is_path(v: str) -> bool:
            return v.startswith(("/", "~")) or ".." in PurePosixPath(v).parts

        def map_arg(tok: str) -> str:
            """A path argument, also one glued to its option (--files0-from=/x, -f/x), stays in the sandbox."""
            if tok.startswith("--") and "=" in tok:
                opt, value = tok.split("=", 1)
                return f"{opt}={map_path(value)}"
            if tok.startswith("-") and not tok.startswith("--") and len(tok) > 2 and is_path(tok[2:]):
                return tok[:2] + map_path(tok[2:])
            return map_path(tok)

        def segments(command: str) -> list[list[str]]:
            lex = shlex.shlex(command, posix=True, punctuation_chars=True)
            lex.whitespace_split = True
            try:
                toks = list(lex)
            except ValueError as e:
                raise ToolError(f"errore di sintassi nel comando: {e}") from e
            if "`" in command or "$(" in command:
                raise ToolError("sostituzione di comando non supportata in questo terminale")
            segs, cur = [], []
            for t in toks:
                if t == "|":
                    segs.append(cur)
                    cur = []
                elif t and set(t) <= set(";&<>|"):
                    raise ToolError(
                        f"operatore '{t}' non supportato: un comando alla volta, solo pipe '|'. "
                        "Per scrivere un file usa lo strumento apposito."
                    )
                else:
                    cur.append(t)
            segs.append(cur)
            if any(not s for s in segs):
                raise ToolError("comando vuoto")
            return segs

        @self.server.tool()
        def esegui_comando(comando: str) -> str:
            """Esegue un comando nel terminale dell'utente e restituisce l'output.
            Ogni comando parte da /home/utente; sono supportate le pipe '|' ma non
            redirezioni, ';' o '&&'. Disponibili i comandi di base (ls, cat, head, tail,
            wc, grep, find, du, sort, uniq, cut, tr, file, echo, mkdir, touch, cp, mv, rm,
            diff, pwd, date)."""
            segs = segments(comando)
            for seg in segs:
                prog = seg[0]
                if prog == "cd":
                    raise ToolError("cd non ha effetto: ogni comando parte da /home/utente, usa i percorsi")
                if prog in BLOCKED:
                    raise ToolError(f"{prog}: comando non consentito in questo terminale")
                if prog not in ALLOWED and prog != "date":
                    raise ToolError(f"{prog}: comando non trovato")
                if prog == "find" and FIND_BLOCKED & set(seg):
                    raise ToolError("find: le azioni -exec, -ok e -delete non sono consentite")
            if len(segs) == 1 and segs[0][0] == "date":
                return date_text(self.ws.world.now)
            if any(s[0] == "date" for s in segs):
                raise ToolError("date: disponibile solo da solo, non in una pipe")
            procs, prev = [], None
            env = {"PATH": "/usr/bin:/bin", "LC_ALL": "en_US.UTF-8", "HOME": root, "TZ": "Europe/Rome"}
            try:
                for seg in segs:
                    argv = [seg[0]] + [map_arg(t) for t in seg[1:]]
                    p = subprocess.Popen(
                        argv, cwd=root, env=env, stdin=prev, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    if prev is not None:
                        prev.close()
                    prev = p.stdout
                    procs.append(p)
                out, err = procs[-1].communicate(timeout=5)
                errs = [err] + [q.stderr.read() for q in procs[:-1]]
                for q in procs[:-1]:
                    q.wait(timeout=5)
            except subprocess.TimeoutExpired as e:
                for q in procs:
                    q.kill()
                raise ToolError("tempo scaduto (5 s)") from e
            rc = procs[-1].returncode
            text = host_neutral(out.decode("utf-8", "replace").replace(root, HOME))
            etext = b"".join(errs).decode("utf-8", "replace").replace(root, HOME).strip()
            if len(text) > MAX_OUT:
                text = text[:MAX_OUT] + "\n(output troncato)"
            if rc != 0 and not text.strip():
                if not etext:
                    return f"(nessun output, codice di uscita {rc})"
                raise ToolError(f"codice di uscita {rc}: {etext}")
            if rc != 0:
                return f"{text.rstrip()}\n[codice di uscita {rc}] {etext}".strip()
            return text.rstrip() or "(nessun output)"

    def check(self, kind: str, spec: dict) -> bool | None:
        return self.ws.sandbox.check(kind, spec)
