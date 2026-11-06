"""Supervised fine-tuning on verified conversations: LoRA, loss only where the model speaks.

  python -m qita.train.sft --model ~/qita-models/4B-8bit --data runs/pilot_a.kept.jsonl,runs/pilot_b.kept.jsonl \
      --out runs/sft_pilot
Runs on one Mac, or on two as a ring (scripts/ring.sh): each rank trains on its own share and
gradients are averaged every step. Rank 0 logs every step to <out>/log.jsonl, evaluates the
validation loss and saves mlx-lm adapters (<out>/adapters.safetensors + adapter_config.json),
loadable with `mlx_lm.load(model, adapter_path=out)` and `qita.eval --adapter out`, and the optimizer
state (<out>/optimizer.safetensors). `--resume N` continues a stopped run from the checkpoint
saved at step N in <out> (copy it to every rank first): same data order, the first N steps skipped.
`--init DIR` starts from the adapters in DIR instead of zero. A data file written `path@0.3` keeps
a stable 30% of its training conversations (a replay share); its validation conversations all stay.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from ..harness.runner import env_tools
from .data import examples, with_tools
from .loss import loss_targets, setup, target_index


def conv_key(conv: dict) -> str:
    """The world seed, or the content when there is none."""
    seed = conv.get("meta", {}).get("seed")
    return str(seed) if seed is not None else json.dumps(conv["messages"], sort_keys=True, ensure_ascii=False)


def in_share(key: str, frac: float) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10_000 < frac * 10_000


def is_val(conv: dict, frac: float) -> bool:
    """Stable split by the world seed (or the content when there is none)."""
    return in_share(conv_key(conv), frac)


def load_examples(paths: list[str], tok, max_len: int, val_frac: float, extra_tools: list[dict] | None = None):
    train, val, dropped, misaligned = [], [], 0, 0
    for spec in paths:
        p, _, keep = spec.partition("@")
        keep = float(keep or 1)
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            conv = with_tools(json.loads(line), extra_tools or [])
            v = is_val(conv, val_frac)
            if not v and not in_share("replay-" + conv_key(conv), keep):
                continue
            for ex in examples(conv, tok):
                if len(ex["tokens"]) > max_len:
                    dropped += 1
                    continue
                misaligned += not ex.get("aligned", True)
                (val if v else train).append(ex)
    if misaligned:  # a target span that starts inside a token loses that token's loss (qita/train/data.py)
        print(f"warning: {misaligned} examples have a target span that starts inside a token", flush=True)
    return train, val, dropped


def shard(exs: list[dict], rank: int, world: int, seed: int) -> list[dict]:
    """Same number of examples on every rank (a ring steps together), lengths dealt round-robin."""
    order = sorted(range(len(exs)), key=lambda i: len(exs[i]["tokens"]))
    mine = [exs[i] for i in order[rank::world]][: len(order) // world]
    random.Random(seed).shuffle(mine)
    return mine


def batch_of(ex: dict):
    x, m = mx.array([ex["tokens"]]), [ex["mask"]]
    idx, w = target_index(m)
    return x, mx.array(m), idx, w


def val_loss(model, val: list[dict], rank: int = 0, world: int = 1) -> float:
    """Token-weighted validation loss. Every rank evaluates its share and the totals are summed on the
    CPU: a rank idle while another evaluates would wait inside a GPU command until macOS kills it
    (measured: `GPU Timeout` on the waiting machine). The buffers of its many shapes are freed at the end:
    a run without that was killed by macOS at the validation of step 1600 twice, at 41 and 44 GB."""
    model.eval()
    tot = n = 0.0
    for ex in val[rank::world]:
        loss, k = loss_targets(model, *batch_of(ex))
        mx.eval(loss, k)
        tot += loss.item() * k.item()
        n += k.item()
    model.train()
    mx.clear_cache()
    if world > 1:
        s = mx.distributed.all_sum(mx.array([tot, n], dtype=mx.float32), stream=mx.cpu)
        tot, n = s.tolist()
    return tot / max(n, 1)


def save_adapters(model, out: Path, a, n_layers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    weights = {k: v for k, v in tree_flatten(model.trainable_parameters())}
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": n_layers,
        "lora_parameters": {"rank": a.rank, "scale": a.scale, "dropout": 0.0}}, indent=2))


def save_optimizer(opt, out: Path) -> None:
    mx.save_safetensors(str(out / "optimizer.safetensors"), dict(tree_flatten(opt.state)))


def load_optimizer(opt, path: Path) -> None:
    opt.state = tree_unflatten(list(mx.load(str(path)).items()))


def resume_lr(sched, start: int, warmup: int):
    """The original schedule shifted by `start` and ramped up again over `warmup` steps."""
    return lambda s: sched(s + start) * mx.minimum(1.0, (s + 1) / warmup)


def resume_optimizer(sched, start: int, warmup: int):
    """AdamW restarting from zero moments after `start` steps (a checkpoint without optimizer state).
    Bias correction on: without it the update is lr * (1 - 0.9^t) / sqrt(1 - 0.999^t) for a steady
    gradient, 5.6x lr at t = 30 and 3.2x at t = 100, where the original run was at 1.13x by step 1500
    (the resume at step 1500 without it: validation loss 0.178 at step 1600, 0.114 at step 1400)."""
    return optim.AdamW(learning_rate=resume_lr(sched, start, warmup), weight_decay=0.0, bias_correction=True)


def average_on_cpu(grads):
    """Gradient average with every op on the CPU. A GPU op that depends on the all-reduce waits
    inside a command buffer while the other rank is late, and macOS kills it after about 5 s
    (`GPU Timeout`; measured with a probe that stalls one rank for 90 s: it kills the compiled step,
    also with the all-reduce alone on the CPU stream, and not this one)."""
    with mx.stream(mx.cpu):
        g = nn.average_gradients(grads, communication_stream=mx.cpu)
        mx.eval(g)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="comma-separated JSONL files of kept conversations")
    ap.add_argument("--add-envs", default="", help="comma-separated harness envs (e.g. calc) whose tools every "
                    "conversation is offered, as the harness offers them at inference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--layers", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--resume", type=int, default=0, help="continue from the checkpoint saved at this step in --out")
    ap.add_argument("--init", default=None, help="start from the adapters saved in this folder")
    ap.add_argument("--cache-gb", type=float, default=2.0, help="cap on the MLX buffer cache; 0 leaves it uncapped")
    a = ap.parse_args()
    a.mode, a.freeze_embed, a.grad_ckpt, a.no_chunk = "lora", False, True, False

    from mlx_lm import load

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()
    mx.random.seed(a.seed)
    if a.cache_gb > 0:
        mx.set_cache_limit(int(a.cache_gb * 2**30))
    model, tok = load(a.model)
    n_train = setup(model, a)
    n_layers = len(model.layers) if a.layers < 0 else a.layers
    extra = asyncio.run(env_tools(a.add_envs.split(","))) if a.add_envs else []
    train, val, dropped = load_examples(a.data.split(","), tok, a.max_len, a.val_frac, extra)
    mine = shard(train, rank, world, a.seed)
    steps = len(mine) * a.epochs
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        info = {"model": a.model, "world": world, "train_examples": len(train), "val_examples": len(val),
                "dropped_over_max_len": dropped, "steps": steps, "trainable_M": round(n_train / 1e6, 1), **vars(a)}
        (out / "run.json").write_text(json.dumps(info, indent=2))
        print("RUN " + json.dumps(info), flush=True)

    sched = optim.join_schedules(
        [optim.linear_schedule(a.lr / 100, a.lr, a.warmup), optim.cosine_decay(a.lr, max(1, steps - a.warmup), a.lr / 10)],
        [a.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    if a.init and not a.resume:
        model.load_weights(str(Path(a.init) / "adapters.safetensors"), strict=False)
    if a.resume:
        model.load_weights(str(out / "adapters.safetensors"), strict=False)
        if (out / "optimizer.safetensors").exists():
            load_optimizer(opt, out / "optimizer.safetensors")
            assert opt.step.item() == a.resume, f"optimizer state is at step {opt.step.item()}, not {a.resume}"
        else:
            opt = resume_optimizer(sched, a.resume, a.warmup)
        if rank == 0:
            print(f"RESUME from step {a.resume}, optimizer state "
                  f"{'loaded' if (out / 'optimizer.safetensors').exists() else 'fresh, bias-corrected, lr ramped up again'}", flush=True)
    loss_and_grad = nn.value_and_grad(model, loss_targets)
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def grads_of(x, m, idx, w):
        return loss_and_grad(model, x, m, idx, w)

    @partial(mx.compile, inputs=state, outputs=state)
    def update(grads):
        opt.update(model, grads)

    def step(x, m, idx, w):
        """Gradients on the GPU, the average across ranks on the CPU, then the update on the GPU."""
        (loss, n), grads = grads_of(x, m, idx, w)
        if world > 1:
            mx.eval(loss, n, grads)
            grads = average_on_cpu(grads)
        update(grads)
        return loss, n

    model.train()
    log = (out / "log.jsonl").open("a") if rank == 0 else None
    if val and not a.resume:
        v0 = val_loss(model, val, rank, world)  # before any update: the reference for every later value
        if rank == 0:
            log.write(json.dumps({"step": 0, "val_loss": round(v0, 4)}) + "\n")
            print(f"step 0 val_loss {v0:.4f}", flush=True)
    t0, it = time.perf_counter(), 0
    for epoch in range(a.epochs):
        for ex in mine:
            it += 1
            if it <= a.resume:
                continue
            t = time.perf_counter()
            loss, n = step(*batch_of(ex))
            mx.eval(state, loss, n)
            if rank == 0:
                rec = {"step": it, "epoch": epoch, "loss": round(loss.item(), 4), "tokens": len(ex["tokens"]),
                       "targets": int(n.item()), "lr": opt.learning_rate.item(), "sec": round(time.perf_counter() - t, 2),
                       "active_gb": round(mx.get_active_memory() / 2**30, 2), "cache_gb": round(mx.get_cache_memory() / 2**30, 2)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if it % 10 == 0:
                    print(f"step {it}/{steps} loss {rec['loss']} lr {rec['lr']:.2e} {time.perf_counter() - t0:.0f}s", flush=True)
            if val and (it % a.eval_every == 0 or it == steps):
                v = val_loss(model, val, rank, world)
                if rank == 0:
                    log.write(json.dumps({"step": it, "val_loss": round(v, 4)}) + "\n")
                    print(f"step {it} val_loss {v:.4f}", flush=True)
            if rank == 0 and (it % a.save_every == 0 or it == steps):
                save_adapters(model, out, a, n_layers)
                save_optimizer(opt, out)
            if world > 1 and (it % a.save_every == 0 or it == steps):
                mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))  # wait for rank 0 to finish saving
    if rank == 0:
        print(f"DONE {it} steps in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
