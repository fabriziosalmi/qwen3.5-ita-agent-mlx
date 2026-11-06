"""Escape attempts from inside scripts/jailed.sh: every one must be denied, the run folder writable.

  scripts/jailed.sh probe scripts/jail_probe.py      (exit 0 only if the jail holds)
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

home = Path(os.environ["HOME"])
tmp = Path(os.environ["TMPDIR"])
must_fail = {
    "write in home": lambda: (home / "qita-jail-escape").write_text("x"),
    "write in /tmp": lambda: Path("/tmp/qita-jail-escape").write_text("x"),
    "write in the repo outside runs/jail": lambda: Path("qita-jail-escape").write_text("x"),
    "read ~/.ssh": lambda: os.listdir(home / ".ssh"),
    "read ~/Documents outside the repo": lambda: os.listdir(home / "Documents"),
    "network": lambda: socket.create_connection(("1.1.1.1", 53), timeout=3),
    "child process writes in home": lambda: subprocess.run(["touch", str(home / "qita-jail-escape2")], check=True,
                                                           capture_output=True),
}
bad = 0
for what, f in must_fail.items():
    try:
        f()
        print(f"ESCAPED  {what}")
        bad += 1
    except Exception as e:  # noqa: BLE001 - any refusal is what we want
        print(f"denied   {what} ({type(e).__name__})")
try:
    (tmp / "ok").write_text("x")
    print("allowed  write in the run folder")
except Exception as e:  # noqa: BLE001
    print(f"BROKEN   write in the run folder ({e})")
    bad += 1
sys.exit(1 if bad else 0)
