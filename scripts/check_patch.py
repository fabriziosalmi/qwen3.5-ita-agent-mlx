"""The chunked training path gives the same loss and LoRA gradients as mlx-lm's own, on the real model.

  uv run python scripts/check_patch.py [MODEL] [fp32]      (QITA_T: sequence length, default 256)
"""

import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, so qita imports
from qita.train.gdn_chunk import patch_qwen3_5  # noqa: E402
from qita.train.loss import loss_fn  # noqa: E402

T = int(os.environ.get("QITA_T", "256"))
args = [a for a in sys.argv[1:] if a != "fp32"]
model, tok = load(args[0] if args else "Qwen/Qwen3.5-0.8B")
if "fp32" in sys.argv[1:]:
    model.set_dtype(mx.float32)
model.freeze()
linear_to_lora_layers(model, len(model.layers), {"rank": 8, "scale": 20.0, "dropout": 0.0})
# give the LoRA B matrices non-zero values, so the adapters change the forward and get real gradients
mx.random.seed(0)
model.update(tree_unflatten([(k, (mx.random.normal(v.shape) * 1e-4).astype(v.dtype))
                             for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]))
text = ("Oggi è mercoledì e devo leggere il file della spesa, poi segnare in calendario la riunione. " * 20)
ids = tok.encode(text)[:T]
x = mx.array([ids])
m = mx.ones_like(x)
model.train()
lg = nn.value_and_grad(model, loss_fn)


def run():
    (loss, _), grads = lg(model, x, m)
    flat = dict(tree_flatten(grads))
    mx.eval(loss, flat)
    return loss.item(), flat


def rel_err(g, ref):
    r = sorted((mx.linalg.norm((g[k].astype(mx.float32) - ref[k].astype(mx.float32))) /
                mx.maximum(mx.linalg.norm(ref[k].astype(mx.float32)), 1e-12)).item() for k in ref)
    return f"median {r[len(r) // 2]:.2e}, p90 {r[int(len(r) * 0.9)]:.2e}, max {r[-1]:.2e}"


l0, g0 = run()
patch_qwen3_5(64)
l1, g1 = run()
print(f"model {args[0] if args else '0.8B'} {'fp32' if 'fp32' in sys.argv else 'native'}: loss original {l0:.6f} chunked {l1:.6f}; "
      f"chunked vs original grads: {rel_err(g1, g0)} over {len(g0)} tensors")
if "fp32" not in sys.argv[1:]:
    model.set_dtype(mx.float32)
    lr, gr = run()
    print(f"chunked fp32 loss {lr:.6f}; original bf16 vs chunked fp32: {rel_err(g0, gr)}; chunked bf16 vs chunked fp32: {rel_err(g1, gr)}")
