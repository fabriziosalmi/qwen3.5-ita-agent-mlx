"""Chunkwise gated delta rule for training Qwen3.5 in MLX.

mlx-lm runs the linear-attention layers in training through `gated_delta_ops`, a Python loop
over tokens whose backward keeps a [Dv, Dk] state per token per layer (1 MB at Dk = Dv = 128):
at 1,024 tokens the 0.8B model does not fit in 16 GB, and a step takes about 40 s.

This is the chunkwise form (Yang et al., Gated Delta Networks; the reference is
`torch_chunk_gated_delta_rule` in transformers): parallel inside chunks of C tokens with a UT
transform, sequential only across chunks, made of plain ops so autodiff gives the backward.
The unit lower-triangular inverse is built by block recursion (as LAPACK trtri): diagonal
blocks of size b are inverted, then merged pairwise with X21 = -X22 L21 X11. Every
intermediate is a true sub-inverse, so it stays bounded. The power series
(I + N)(I + N^2)(I + N^4)... is exact in algebra but cancels catastrophically in float32 when
keys are nearly parallel (measured: error 1.7e16 at chunk 64 in tests/test_gdn_chunk.py).
MLX's solve_triangular runs only on the CPU and has no vjp.

`patch_qwen3_5()` swaps it into mlx-lm's Qwen3.5 model for training only (the inference
path, with its Metal kernel, is untouched).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def unit_lower_inverse(strict):
    """(I + strict)^-1 for a strictly lower-triangular [..., C, C] matrix, C a power of two."""
    C = strict.shape[-1]
    lead = strict.shape[:-2]
    x = mx.ones((*lead, C, 1, 1), dtype=strict.dtype)       # inverses of the 1x1 diagonal blocks
    b = 1
    while b < C:
        nsb = C // (2 * b)
        blocks = strict.reshape(*lead, nsb, 2 * b, nsb, 2 * b)
        diag = mx.diagonal(blocks, axis1=-4, axis2=-2)          # [..., 2b, 2b, nsb]
        diag = mx.moveaxis(diag, -1, -3)                        # [..., nsb, 2b, 2b]
        l21 = diag[..., b:, :b]
        x11, x22 = x[..., 0::2, :, :], x[..., 1::2, :, :]
        x21 = -(x22 @ l21 @ x11)
        top = mx.concatenate([x11, mx.zeros_like(x11)], axis=-1)
        bottom = mx.concatenate([x21, x22], axis=-1)
        x = mx.concatenate([top, bottom], axis=-2)
        b *= 2
    return x.reshape(*lead, C, C)


def chunk_gated_delta(q, k, v, log_g, beta, state=None, chunk: int = 64):
    """Same contract as mlx-lm's gated_delta_ops, with the decay given in log space.

    q, k: [B, T, Hk, Dk] (q already carries the readout scale), v: [B, T, Hv, Dv],
    log_g, beta: [B, T, Hv], state: [B, Hv, Dv, Dk] or None.
    Returns y: [B, T, Hv, Dv] (dtype of q) and the final state [B, Hv, Dv, Dk] (float32).
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if (rep := Hv // Hk) > 1:
        q = mx.repeat(q, rep, -2)
        k = mx.repeat(k, rep, -2)
    out_dtype = q.dtype
    f = mx.float32
    # [B, H, T, D] in float32
    q, k, v = (x.astype(f).transpose(0, 2, 1, 3) for x in (q, k, v))
    log_g, beta = (x.astype(f).transpose(0, 2, 1) for x in (log_g, beta))
    pad = (-T) % chunk
    if pad:
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v))
        log_g, beta = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (log_g, beta))
    n = (T + pad) // chunk
    H = Hv

    def chunks(x):
        return x.reshape(B, H, n, chunk, x.shape[-1])

    q, k, v = chunks(q), chunks(k), chunks(v)
    log_g = log_g.reshape(B, H, n, chunk)
    beta = beta.reshape(B, H, n, chunk)

    cum = mx.cumsum(log_g, axis=-1)                          # [B, H, n, C]
    diff = cum[..., :, None] - cum[..., None, :]             # i - j
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    pairwise = mx.exp(mx.where(lower, diff, -math.inf))      # decay from j to i, 0 above the diagonal

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    a = (k_beta @ k.swapaxes(-1, -2)) * pairwise
    strict = mx.tril(mx.ones((chunk, chunk), dtype=f), -1)
    inv = unit_lower_inverse(a * strict)                     # (I + strictly lower part of a)^-1

    u = inv @ v_beta                                         # new values
    w = inv @ (k_beta * mx.exp(cum)[..., None])              # decayed keys that read the old state
    qd = q * mx.exp(cum)[..., None]
    kd = k * mx.exp(cum[..., -1:] - cum)[..., None]
    intra = (q @ k.swapaxes(-1, -2)) * pairwise
    chunk_decay = mx.exp(cum[..., -1])[..., None, None]      # [B, H, n, 1, 1]

    s = mx.zeros((B, H, Dk, Dv), dtype=f) if state is None else state.astype(f).swapaxes(-1, -2)
    outs = []
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ s
        outs.append(qd[:, :, i] @ s + intra[:, :, i] @ v_new)
        s = s * chunk_decay[:, :, i] + kd[:, :, i].swapaxes(-1, -2) @ v_new
    y = mx.stack(outs, axis=2).reshape(B, H, n * chunk, Dv)[:, :, :T]
    return y.transpose(0, 2, 1, 3).astype(out_dtype), s.swapaxes(-1, -2)


def log_decay(A_log, a, dt_bias):
    """log of mlx-lm's compute_g, computed directly: -exp(A_log) * softplus(a + dt_bias)."""
    return -mx.exp(A_log.astype(mx.float32)) * nn.softplus((a + dt_bias).astype(mx.float32))


def patch_qwen3_5(chunk: int = 64) -> None:
    from mlx_lm.models import qwen3_5

    original = qwen3_5.gated_delta_update
    if getattr(original, "_qita_chunked", False):
        return

    def update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
               lower_bound=None, allow_neg_eigval=False):
        if use_kernel or mask is not None or lower_bound is not None or allow_neg_eigval:
            return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                            lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        return chunk_gated_delta(q, k, v, log_decay(A_log, a, dt_bias), mx.sigmoid(b), state, chunk)

    update._qita_chunked = True
    qwen3_5.gated_delta_update = update
