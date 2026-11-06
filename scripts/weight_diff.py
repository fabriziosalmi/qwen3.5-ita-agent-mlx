"""Which weights differ between two checkpoints of the same architecture, and the rank of each change.

  python scripts/weight_diff.py BASE.safetensors EDITED.safetensors

Prints, for every tensor that differs: its name and shape, how many entries changed, the rows and columns
that hold them, and the top singular values of the difference (a rank-one edit has one value far above the
rest). Reads the files only; no model is run.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np


def main() -> None:
    a, b = mx.load(sys.argv[1]), mx.load(sys.argv[2])
    print(f"tensors: {len(a)} and {len(b)}; only in one: {sorted(set(a) ^ set(b))}")
    total = changed = 0
    for name in sorted(set(a) & set(b)):
        x, y = np.array(a[name].astype(mx.float32)), np.array(b[name].astype(mx.float32))
        total += x.size
        if x.shape != y.shape:
            print(f"{name}: shape {x.shape} vs {y.shape}")
            continue
        d = y - x
        nz = int(np.count_nonzero(d))
        if not nz:
            continue
        changed += nz
        line = f"{name} {x.shape}: {nz} changed"
        if d.ndim == 2:
            rows, cols = np.nonzero(np.abs(d).sum(1))[0], np.nonzero(np.abs(d).sum(0))[0]
            s = np.linalg.svd(d[np.ix_(rows, cols)], compute_uv=False)
            line += f", {len(rows)} rows x {len(cols)} cols, singular values {np.round(s[:4], 4).tolist()}"
            line += f", |d|/|w| {np.linalg.norm(d) / np.linalg.norm(x):.4f}"
        print(line)
    print(f"changed {changed} of {total} weights ({100 * changed / total:.4f}%)")


if __name__ == "__main__":
    main()
