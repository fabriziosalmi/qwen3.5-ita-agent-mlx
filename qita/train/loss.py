"""Masked loss for Qwen3.5 in MLX: loss only on the model's own tokens, and the LoRA setup of qita.train.sft."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def target_index(m: list[list[int]], bucket: int = 128) -> tuple[mx.array, mx.array]:
    """Flat positions (in the shifted [B, T-1] grid) that carry loss, padded to a multiple of bucket with weight 0."""
    flat = [i * (len(row) - 1) + j for i, row in enumerate(m) for j, v in enumerate(row[1:]) if v]
    n = max(bucket, (len(flat) + bucket - 1) // bucket * bucket)
    w = [1.0] * len(flat) + [0.0] * (n - len(flat))
    return mx.array(flat + [0] * (n - len(flat))), mx.array(w)


def loss_targets(model, x, m, idx, w):
    """Same value as loss_fn, but the vocabulary projection runs only at target positions."""
    lm = model.language_model
    h = lm.model(x[:, :-1])
    h = h.reshape(-1, h.shape[-1])[idx]
    logits = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    ce = nn.losses.cross_entropy(logits, x[:, 1:].reshape(-1)[idx]) * w
    n = w.sum()
    return ce.astype(mx.float32).sum() / n, n


def loss_fn(model, x, m, idx=None, w=None):
    """The reference: full logits, masked cross-entropy (scripts/check_loss.py compares the two)."""
    logits = model(x[:, :-1])
    mask = m[:, 1:]
    ce = nn.losses.cross_entropy(logits, x[:, 1:]) * mask
    n = mask.sum()
    return ce.astype(mx.float32).sum() / n, n


def setup(model, a) -> int:
    """LoRA on the last a.layers layers (all when negative), the chunked gated delta rule unless a.no_chunk,
    gradient checkpointing when a.grad_ckpt; returns the number of trainable parameters."""
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from .gdn_chunk import patch_qwen3_5

    if not a.no_chunk:
        patch_qwen3_5(a.chunk)
    layers = model.layers
    model.freeze()
    n_layers = len(layers) if a.layers < 0 else a.layers
    linear_to_lora_layers(model, n_layers, {"rank": a.rank, "scale": a.scale, "dropout": 0.0})
    if a.grad_ckpt:
        grad_checkpoint(layers[0])
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
