"""SFT data plumbing: stable validation split and equal, length-balanced shards per rank."""

from qita.train.sft import is_val, shard


def test_val_split_is_stable_and_about_the_fraction():
    convs = [{"meta": {"seed": 1_000_000 + i}, "messages": []} for i in range(4000)]
    a = [is_val(c, 0.05) for c in convs]
    assert a == [is_val(c, 0.05) for c in convs]
    assert 120 <= sum(a) <= 280


def test_shards_are_equal_disjoint_and_balanced():
    exs = [{"tokens": [0] * (i % 97 + 1), "id": i} for i in range(101)]
    s0, s1 = shard(exs, 0, 2, 0), shard(exs, 1, 2, 0)
    assert len(s0) == len(s1) == 50
    assert not ({e["id"] for e in s0} & {e["id"] for e in s1})
    l0, l1 = sum(len(e["tokens"]) for e in s0), sum(len(e["tokens"]) for e in s1)
    assert abs(l0 - l1) / max(l0, l1) < 0.05
    assert shard(exs, 0, 1, 0) and len(shard(exs, 0, 1, 0)) == 101


def test_optimizer_state_round_trips_so_a_resumed_run_continues_the_same_updates(tmp_path):
    """Break-test target: a load_optimizer that drops the moments makes the next update differ."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from qita.train.sft import load_optimizer, save_optimizer

    def run(model, opt, steps):
        for k in range(steps):
            x = mx.array([[1.0, 2.0 + k]])
            _, g = nn.value_and_grad(model, lambda m: (m(x) ** 2).sum())(model)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state)

    mx.random.seed(0)
    a = nn.Linear(2, 2)
    oa = optim.AdamW(learning_rate=optim.cosine_decay(1e-2, 20), weight_decay=0.0)
    run(a, oa, 3)
    a.save_weights(str(tmp_path / "w.safetensors"))
    save_optimizer(oa, tmp_path)
    b = nn.Linear(2, 2)
    b.load_weights(str(tmp_path / "w.safetensors"))
    ob = optim.AdamW(learning_rate=optim.cosine_decay(1e-2, 20), weight_decay=0.0)
    load_optimizer(ob, tmp_path / "optimizer.safetensors")
    assert ob.step.item() == 3
    run(a, oa, 2)
    run(b, ob, 2)
    assert mx.allclose(a.weight, b.weight, atol=1e-7).item() and mx.allclose(a.bias, b.bias, atol=1e-7).item()


def test_resume_lr_is_the_shifted_schedule_after_a_new_ramp():
    """Break-test target: without the shift by `start` the resumed run would replay the warmup peak."""
    import mlx.optimizers as optim

    from qita.train.sft import resume_lr
    sched = optim.join_schedules([optim.linear_schedule(2e-7, 2e-5, 30), optim.cosine_decay(2e-5, 2504, 2e-6)], [30])
    r = resume_lr(sched, 1500, 30)
    assert r(0).item() < sched(1500).item() / 10
    for s in (29, 100, 1000):
        assert abs(r(s).item() - sched(1500 + s).item()) < 1e-12


def test_a_fresh_resume_steps_at_about_lr():
    """Break-test target: without bias correction the zero moments make the update 3-6x lr for
    hundreds of steps."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from qita.train.sft import resume_optimizer
    sched = optim.join_schedules([optim.linear_schedule(2e-7, 2e-5, 30), optim.cosine_decay(2e-5, 2504, 2e-6)], [30])
    lin = nn.Linear(1, 1, bias=False)
    opt = resume_optimizer(sched, 1500, 30)
    g = {"weight": mx.array([[0.3]])}
    for s in range(200):
        before = lin.weight.item()
        opt.update(lin, g)
        mx.eval(lin.parameters(), opt.state)
        if s >= 30:
            ratio = abs(lin.weight.item() - before) / sched(1500 + s).item()
            assert 0.9 < ratio < 1.1, (s, ratio)


def test_a_replay_share_keeps_a_stable_part_of_training_and_all_of_validation(tmp_path, monkeypatch):
    import json
    import qita.train.sft as sft
    monkeypatch.setattr(sft, "examples", lambda conv, tok: [{"tokens": [conv["meta"]["seed"]], "mask": [1]}])
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps({"messages": [], "meta": {"seed": s}}) + "\n" for s in range(2000)))
    train, val, _ = sft.load_examples([str(path)], None, 10, 0.05)
    t30, v30, _ = sft.load_examples([f"{path}@0.3"], None, 10, 0.05)
    assert v30 == val
    assert 0.25 < len(t30) / len(train) < 0.35
    assert t30 == sft.load_examples([f"{path}@0.3"], None, 10, 0.05)[0]
    assert {e["tokens"][0] for e in t30} <= {e["tokens"][0] for e in train}


def test_validation_frees_the_buffers_of_its_shapes():
    """Break-test: without mx.clear_cache() in val_loss the cache keeps the freed buffers."""
    from types import SimpleNamespace

    import mlx.core as mx
    import mlx.nn as nn

    from qita.train.sft import val_loss

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 16)

        def __call__(self, x):
            return self.embed_tokens(x)

    class Lang(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.args = SimpleNamespace(tie_word_embeddings=True)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Lang()

    model = Model()
    val = [{"tokens": [i % 64 for i in range(n)], "mask": [0] + [1] * (n - 1)} for n in (40, 300, 900, 2000)]
    mx.set_cache_limit(2**30)
    v = val_loss(model, val)
    assert v > 0
    assert mx.get_cache_memory() < 2**20  # only the last two scalars, freed on return
