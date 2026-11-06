"""Weighted average of two checkpoints (scripts/average_weights.py)."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

spec = importlib.util.spec_from_file_location("average_weights", Path(__file__).parents[1] / "scripts" / "average_weights.py")
aw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aw)


def test_the_average_is_weighted_and_keeps_the_dtype():
    a = {"w": mx.array([0.0, 2.0], dtype=mx.bfloat16)}
    b = {"w": mx.array([4.0, 6.0], dtype=mx.bfloat16)}
    assert aw.average(a, b, 0.5)["w"].tolist() == [2.0, 4.0]
    assert aw.average(a, b, 0.25)["w"].tolist() == [1.0, 3.0]
    assert aw.average(a, b, 0.5)["w"].dtype == mx.bfloat16


def test_checkpoints_with_other_keys_or_shapes_are_refused():
    a = {"w": mx.zeros((2,))}
    with pytest.raises(AssertionError):
        aw.average(a, {"v": mx.zeros((2,))}, 0.5)
    with pytest.raises(AssertionError):
        aw.average(a, {"w": mx.zeros((3,))}, 0.5)
