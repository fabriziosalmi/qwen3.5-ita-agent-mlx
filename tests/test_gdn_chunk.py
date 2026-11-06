"""The chunkwise gated delta rule equals mlx-lm's token-by-token reference: outputs, state, gradients."""

import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_ops

from qita.train.gdn_chunk import chunk_gated_delta


def inputs(B=2, T=150, Hk=4, Hv=4, D=32, seed=0, with_state=False):
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, D))
    k = mx.random.normal((B, T, Hk, D))
    # unit-norm keys and scaled queries, as normalize_qk gives the real model
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * D ** -0.5
    v = mx.random.normal((B, T, Hv, D))
    log_g = -mx.random.uniform(0.0, 0.5, (B, T, Hv))
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv))
    state = mx.random.normal((B, Hv, D, D)) * 0.1 if with_state else None
    return q, k, v, log_g, beta, state


def reference(q, k, v, log_g, beta, state):
    return gated_delta_ops(q, k, v, mx.exp(log_g), beta, state)


@pytest.mark.parametrize("T,chunk,with_state,hv", [(150, 64, False, 4), (64, 64, True, 4), (37, 16, True, 8), (200, 32, False, 4)])
def test_forward_matches_reference(T, chunk, with_state, hv):
    q, k, v, lg, b, s = inputs(T=T, Hv=hv, with_state=with_state)
    y0, s0 = reference(q, k, v, lg, b, s)
    y1, s1 = chunk_gated_delta(q, k, v, lg, b, s, chunk)
    assert y1.shape == y0.shape and s1.shape == s0.shape
    assert mx.allclose(y1, y0, atol=1e-4, rtol=1e-4).item(), mx.abs(y1 - y0).max().item()
    assert mx.allclose(s1, s0, atol=1e-4, rtol=1e-4).item(), mx.abs(s1 - s0).max().item()


def test_gradients_match_reference():
    q, k, v, lg, b, s = inputs(T=100, with_state=True)
    mx.random.seed(1)
    r = mx.random.normal((2, 100, 4, 32))

    def f(impl):
        def g(q, k, v, lg, b, s):
            y, st = impl(q, k, v, lg, b, s)
            return (y * r).sum() + st.sum()
        return mx.grad(g, argnums=(0, 1, 2, 3, 4, 5))

    g0 = f(reference)(q, k, v, lg, b, s)
    g1 = f(lambda *x: chunk_gated_delta(*x, chunk=32))(q, k, v, lg, b, s)
    for name, a0, a1 in zip("q k v log_g beta state".split(), g0, g1):
        err = mx.abs(a1 - a0).max().item()
        scale = mx.abs(a0).max().item()
        assert err <= 1e-3 * max(1.0, scale), (name, err, scale)


def test_break_wrong_decay_is_caught():
    """Break-test: dropping the decay inside chunks must be detected by the comparison."""
    q, k, v, lg, b, s = inputs(T=80)
    y0, _ = reference(q, k, v, lg, b, s)
    y1, _ = chunk_gated_delta(q, k, v, lg * 0, b, s, 64)
    assert not mx.allclose(y1, y0, atol=1e-4, rtol=1e-4).item()


def stress_inputs(T=128, D=32, cos=0.95, seed=3):
    """Nearly parallel keys, beta near 1, decay near 1: high powers of N matter here."""
    mx.random.seed(seed)
    base = mx.random.normal((1, 1, 2, D))
    k = base + mx.random.normal((1, T, 2, D)) * (1 - cos)
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    q = mx.random.normal((1, T, 2, D))
    q = q / mx.linalg.norm(q, axis=-1, keepdims=True) * D ** -0.5
    v = mx.random.normal((1, T, 2, D))
    log_g = -mx.random.uniform(0.0, 0.02, (1, T, 2))
    beta = mx.random.uniform(0.8, 1.0, (1, T, 2))
    return q, k, v, log_g, beta, None


@pytest.mark.parametrize("chunk", [16, 64])
def test_forward_matches_reference_correlated_keys(chunk):
    q, k, v, lg, b, s = stress_inputs()
    y0, s0 = reference(q, k, v, lg, b, s)
    y1, s1 = chunk_gated_delta(q, k, v, lg, b, s, chunk)
    err = mx.abs(y1 - y0).max().item()
    assert err <= 1e-3 * max(1.0, mx.abs(y0).max().item()), err


def test_unit_lower_inverse_is_an_inverse():
    from qita.train.gdn_chunk import unit_lower_inverse
    mx.random.seed(5)
    s = mx.tril(mx.random.uniform(0.8, 0.95, (3, 64, 64)), -1)   # the hard case: large, same-sign entries
    x = unit_lower_inverse(s)
    eye = mx.eye(64)
    err = mx.abs((eye + s) @ x - eye).max().item()
    assert err < 1e-4, err


def test_gradients_match_reference_correlated_keys():
    q, k, v, lg, b, s = stress_inputs(T=96)
    mx.random.seed(2)
    r = mx.random.normal((1, 96, 2, 32))

    def f(impl):
        return mx.grad(lambda q, k, v, lg, b: (impl(q, k, v, lg, b, None)[0] * r).sum(), argnums=(0, 1, 2, 3, 4))

    g0 = f(reference)(q, k, v, lg, b)
    g1 = f(lambda *x: chunk_gated_delta(*x, chunk=64))(q, k, v, lg, b)
    for name, a0, a1 in zip("q k v log_g beta".split(), g0, g1):
        err, scale = mx.abs(a1 - a0).max().item(), mx.abs(a0).max().item()
        assert err <= 1e-3 * max(1.0, scale), (name, err, scale)
