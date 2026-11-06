"""Run an eval split against a model and write one JSON line per task.

  python -m qita.eval --split dev --model Qwen/Qwen3.5-0.8B --out runs/base_dev.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from . import tasks as T
from .harness.backend import MLXBackend
from .harness.checks import score
from .harness.runner import SYSTEM, run_task
from .report import summary

ROOT = Path(__file__).resolve().parents[1]


async def main_async(a) -> None:
    tasks = T.load(ROOT / "eval" / a.split, a.split)
    if a.only:
        keep = set(a.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep or t["category"] in keep]
    if a.limit:
        tasks = tasks[: a.limit]
    backend = MLXBackend(a.model, temp=a.temp, max_tokens=a.max_tokens, seed=a.seed, thinking=a.thinking,
                         cache_reuse=a.cache_reuse, prefix_dir=a.prefix_cache, adapter=a.adapter)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks, 1):
            traj = await run_task(task, backend, system=None if a.system == "none" else SYSTEM, harness=a.harness)
            row = {"model": backend.name, "harness": a.harness, "split": a.split, "temp": a.temp, "seed": a.seed, "thinking": a.thinking,
                   **traj, **score(task, traj)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(row)
            bad = [k for k, v in row["checks"].items() if not v]
            print(f"[{i}/{len(tasks)}] {task['id']:<28} {'OK ' if row['success'] else 'NO '} "
                  f"{row['seconds']:5.1f}s {', '.join(bad)}", flush=True)
    print(f"\n{len(rows)} tasks in {time.perf_counter() - t0:.0f}s -> {out}\n")
    if backend.pcache is not None:
        print("prompt cache:", backend.pcache.stats)
    print(summary(rows))


def harness_name(value: str) -> str:
    """A harness from the command line, with "agent" expanded (qita/harness/runner.py)."""
    from .harness.runner import HARNESS, expand
    if not HARNESS.fullmatch(value):
        raise argparse.ArgumentTypeError(f"unknown harness {value!r}")
    return expand(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated task ids or categories")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--thinking", action="store_true", help="enable the model's thinking mode (reasoning is not scored)")
    ap.add_argument("--harness", type=harness_name, default=None,
                    help="intent: classify each user turn, then wrap it with context, intent and expected answer; "
                         "flags after it, e.g. intent+append+lint+replace+noedit; agent: the default harness of the agent "
                         "(qita/harness/runner.py AGENT), e.g. agent+cal")
    ap.add_argument("--adapter", default=None, help="folder with mlx-lm LoRA adapters (qita.train.sft output)")
    ap.add_argument("--cache-reuse", action="store_true", help="reuse the prompt cache between agent steps")
    ap.add_argument("--prefix-cache", default=None, help="folder of on-disk caches of system prompt + tool schemas")
    ap.add_argument("--system", choices=["default", "none"], default="default",
                    help="default: the harness system prompt (date, home); none: no system message")
    a = ap.parse_args()
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
