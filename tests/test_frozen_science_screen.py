import numpy as np
from collections import Counter

from src.analysis.frozen_science_screen import (
    anchor,
    assigned,
    count_kernel,
    family_map,
    gram,
    residual,
    shuffle_trails,
    slot_vectors,
)


def test_family_controls_preserve_training_counts_and_ignore_development():
    rng = np.random.default_rng(43)
    ids = rng.integers(0, 20, (18, 4, 2, 15))
    alpha = rng.normal(size=(18, 4, 15))
    tr = np.arange(12)
    ids[-1] = 20
    real, random = family_map(ids, alpha, tr, 4, 21)
    for j in range(2):
        assert Counter(real[j, ids[tr, :, j]].ravel()) == Counter(random[j, ids[tr, :, j]].ravel())
    assert (assigned(ids, real)[-1] == 4).all()
    alpha[12:] = 9000
    np.testing.assert_array_equal(real, family_map(ids, alpha, tr, 4, 21)[0])
    null = shuffle_trails(ids)
    for a, b in zip(ids.reshape(-1, 15), null.reshape(-1, 15)):
        assert (a[0], a[-1]) == (b[0], b[-1])
        assert Counter(zip(a[:-1], a[1:])) == Counter(zip(b[:-1], b[1:]))
    np.testing.assert_allclose(count_kernel(ids, tr, False), count_kernel(null, tr, False))


def test_grouped_anchor_and_donor_routing():
    rng = np.random.default_rng(17)
    people = np.repeat(np.arange(10), 3)
    x = rng.normal(size=(30, 4, 2, 15, 2))
    y = x[:, 0, 0, 1]
    tr, ev = np.arange(24), np.arange(24, 30)
    oof, base, _ = anchor(x, y, people, tr, ev, [1.0])
    assert oof.shape == (24, 2) and np.isfinite(base).all()
    perturbed = y.copy()
    perturbed[ev] += 100
    for a, b in zip((oof, base), anchor(x, perturbed, people, tr, ev, [1.0])[:2]):
        np.testing.assert_allclose(a, b)
    k = gram(slot_vectors(x, tr))
    donor = np.arange(30).reshape(-1, 3)[:, [2, 0, 1]].ravel()
    r = y[tr] - oof
    fixed = residual(k, k, r, tr, ev, donor)
    assert not np.allclose(fixed, residual(k, k, r, tr, ev, donor, True, False))
    assert not np.allclose(fixed, residual(k, k, r, tr, ev, donor, False, True))


def test_two_reader_smoke_with_native_geometry(tmp_path, monkeypatch):
    from pathlib import Path
    import pandas as pd
    import pytest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
    import run_frozen_science_screen as suite

    suite.check_contract()
    rng = np.random.default_rng(43)
    n = 24
    m = pd.DataFrame(
        dict(subject=np.repeat(np.arange(12), 2), donor=np.arange(n).reshape(-1, 2)[:, ::-1].ravel())
    )
    tr, dev = np.arange(16), np.arange(16, n)
    arrays = dict(
        hb=rng.normal(size=(n, 4, 300, 2)),
        target=rng.normal(size=(n, 4, 3, 2)),
        xy=rng.normal(size=(n, 2, 4, 2)),
    )
    x = slot_vectors(rng.normal(size=(n, 4, 2, 15, 5)), tr)
    with pytest.raises(ValueError, match="sites"):
        slot_vectors(x[:, :3], tr)
    monkeypatch.setattr(suite, "panel", lambda: (m, tr, dev))
    monkeypatch.setattr(suite, "OUT", tmp_path)
    monkeypatch.setattr(suite, "array", arrays.__getitem__)
    monkeypatch.setattr(suite.parent, "TR", dict(suite.parent.TR, residual_steps=1))
    np.savez(tmp_path / "anchor.npz", oof=np.zeros((len(tr), 24)), base=np.zeros((len(dev), 24)))
    table = suite.score_sources("synthetic", gram(x), x)
    assert set(table.reader) == {"linear", "slot"}
    assert set(table.arm) == {"anchor", "capacity", "true", "wrong"}
    assert set(table.subject) == set(m.subject[dev])
    assert np.isfinite(table.nmse).all()
