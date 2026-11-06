"""Weighted average of two checkpoints of the same architecture (unquantized, e.g. `mlx_lm fuse --dequantize`).

  python scripts/average_weights.py A_DIR B_DIR OUT_DIR [--w 0.5]

OUT = (1 - w) * A + w * B for every tensor of every *.safetensors file, in fp32, cast back to A's dtype.
The two folders must hold the same files, keys and shapes. Every other file (config, tokenizer, template)
is copied from A. Quantize the result afterwards (`mlx_lm convert --hf-path OUT_DIR -q`).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx.core as mx


def average(a: dict, b: dict, w: float) -> dict:
    assert set(a) == set(b), f"keys differ: {sorted(set(a) ^ set(b))[:5]}"
    out = {}
    for k in a:
        assert a[k].shape == b[k].shape, f"{k}: {a[k].shape} vs {b[k].shape}"
        out[k] = ((1 - w) * a[k].astype(mx.float32) + w * b[k].astype(mx.float32)).astype(a[k].dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--w", type=float, default=0.5)
    args = ap.parse_args()
    A, B, out = Path(args.a), Path(args.b), Path(args.out)
    files = sorted(p.name for p in A.glob("*.safetensors"))
    assert files and files == sorted(p.name for p in B.glob("*.safetensors")), "safetensors files differ"
    out.mkdir(parents=True, exist_ok=True)
    for p in A.iterdir():
        if p.is_file() and p.suffix != ".safetensors":
            shutil.copy2(p, out / p.name)
    for name in files:
        avg = average(mx.load(str(A / name)), mx.load(str(B / name)), args.w)
        mx.save_safetensors(str(out / name), avg, metadata={"format": "mlx"})
        print(f"{name}: {len(avg)} tensors", flush=True)


if __name__ == "__main__":
    main()
