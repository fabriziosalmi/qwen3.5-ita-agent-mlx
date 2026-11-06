"""loss_targets equals loss_fn (value and LoRA gradients) on the real model.

  uv run python scripts/check_loss.py
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, so qita imports
from qita.train.gdn_chunk import patch_qwen3_5  # noqa: E402
from qita.train.loss import loss_fn, loss_targets, target_index  # noqa: E402

model, tok = load("Qwen/Qwen3.5-0.8B")
model.set_dtype(mx.float32)
patch_qwen3_5(64)
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": 8, "scale": 20.0, "dropout": 0.0})
mx.random.seed(0)
model.update(tree_unflatten([(k, mx.random.normal(v.shape) * 1e-4) for k, v in tree_flatten(model.trainable_parameters())
                             if k.endswith("lora_b")]))
ids = tok.encode("Oggi è mercoledì e devo leggere il file della spesa, poi segnare la riunione. " * 12)[:300]
x = mx.array([ids, ids[::-1]])
L = len(ids)
m = [[0] * (L // 3) + [1] * (L // 3) + [0] * (L - 2 * (L // 3)), [1] * (L // 4) + [0] * (L - 2 * (L // 4)) + [1] * (L // 4)]
assert all(len(r) == L for r in m), L
idx, w = target_index(m)
model.train()
(l0, n0), g0 = nn.value_and_grad(model, loss_fn)(model, x, mx.array(m))
(l1, n1), g1 = nn.value_and_grad(model, loss_targets)(model, x, mx.array(m), idx, w)
g0, g1 = dict(tree_flatten(g0)), dict(tree_flatten(g1))
rows = sorted(((mx.linalg.norm(g1[k] - g0[k]) / mx.maximum(mx.linalg.norm(g0[k]), 1e-12)).item(), mx.linalg.norm(g0[k]).item(), k) for k in g0)
tot = (sum(mx.linalg.norm(g1[k] - g0[k]).item() ** 2 for k in g0) / sum(mx.linalg.norm(g0[k]).item() ** 2 for k in g0)) ** 0.5
print(f"loss full {l0.item():.6f} targets {l1.item():.6f}; tokens {n0.item()} {n1.item()}")
print(f"relative grad diff per tensor: median {rows[len(rows) // 2][0]:.2e}, max {rows[-1][0]:.2e}; whole gradient {tot:.2e}")
print("worst:", [(f"{r:.1e}", f"norm {n:.1e}", k) for r, n, k in rows[-3:]])
print("largest-norm tensors:", [(f"{r:.1e}", f"norm {n:.1e}") for r, n, k in sorted(rows, key=lambda x: -x[1])[:3]])
