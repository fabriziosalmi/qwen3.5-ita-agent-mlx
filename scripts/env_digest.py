"""SHA-256 of every tool result and state check over all oracle trajectories (dev + test).

Run before and after a change to qita/envs: equal digests mean the eval world did not change.
"""

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("mcp").setLevel(logging.CRITICAL)

from qita import tasks as T  # noqa: E402
from qita.harness.runner import run_task  # noqa: E402
from tests.oracles import DEV, TEST  # noqa: E402
from tests.test_harness import Scripted, to_text  # noqa: E402

h = hashlib.sha256()
n = 0
for split, table in (("dev", DEV), ("test", TEST)):
    for task in T.load(f"eval/{split}", split):
        traj = asyncio.run(run_task(task, Scripted([to_text(s) for s in table[task["id"]]])))
        for c in traj["calls"]:
            h.update(json.dumps([task["id"], c["name"], c["arguments"], c["error"], c["result"]],
                                ensure_ascii=False, sort_keys=True).encode())
            n += 1
        h.update(json.dumps([task["id"], traj["state"]], sort_keys=True).encode())
print(f"{n} tool calls, digest {h.hexdigest()}")
