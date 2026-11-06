"""On-disk prefix cache stays under its cap, dropping the least recently used files first."""

import os
import time

from qita.harness.backend import evict_lru


def make(folder, name, size, age):
    f = folder / f"{name}.safetensors"
    f.write_bytes(b"0" * size)
    t = time.time() - age
    os.utime(f, (t, t))
    return f


def test_evicts_oldest_until_under_cap(tmp_path):
    old, mid, new = make(tmp_path, "old", 100, 300), make(tmp_path, "mid", 100, 200), make(tmp_path, "new", 100, 100)
    assert evict_lru(tmp_path, 250) == 1
    assert not old.exists() and mid.exists() and new.exists()


def test_never_evicts_the_file_just_written(tmp_path):
    big = make(tmp_path, "big", 500, 1000)
    assert evict_lru(tmp_path, 100, keep=big) == 0 and big.exists()


def test_under_cap_nothing_happens(tmp_path):
    make(tmp_path, "a", 10, 10)
    assert evict_lru(tmp_path, 1000) == 0
