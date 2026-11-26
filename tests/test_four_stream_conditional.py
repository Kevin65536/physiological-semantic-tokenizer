import numpy as np
from src.analysis.four_stream_bridge import grouped_folds, select_ridge


def test_training_person_selection_is_blind_to_development_targets():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(24, 5))
    y = x[:, :2] + rng.normal(scale=0.15, size=(24, 2))
    who = np.repeat(np.arange(8), 3)
    train = np.arange(18)
    dev = np.arange(18, 24)
    for a, b in grouped_folds(who, train):
        assert not set(who[a]) & set(who[b])
        assert set(np.r_[a, b]) == set(train)
    grid = [0.01, 0.1, 1.0]
    baseline = select_ridge(x, y, who, train, dev, grid)
    y[dev] += 100
    actual = select_ridge(x, y, who, train, dev, grid)
    for a, b in zip(baseline, actual):
        np.testing.assert_allclose(a, b)


def test_all_hb_path_and_local_interaction_before_pooling():
    import torch
    from src.foundation.conditional_slot_readers import AllHb, HbIncrement, LocalInteraction

    g = torch.Generator().manual_seed(8)
    x = torch.randn(3, 4, 2, 15, 128, generator=g)
    xy = torch.randn(3, 4, 2, generator=g)
    model = AllHb(8)
    other = x.clone()
    other[:, 3] += 2
    assert not torch.equal(model(x, xy)[:, 0], model(other, xy)[:, 0])
    r = HbIncrement(8)
    torch.testing.assert_close(r(x, xy, other, xy), torch.zeros(3, 4, 3, 2))
    inter = LocalInteraction(4, 2)
    torch.testing.assert_close(inter(x, xy), torch.zeros(3, 4, 3, 2))
    torch.nn.init.ones_(inter.output.weight)
    shifted = x.clone()
    shifted[:, :, 1] = shifted[:, :, 1].roll(4, dims=2)
    assert not torch.allclose(inter(x, xy), inter(shifted, xy))


def test_nested_training_replays_with_guarded_targets():
    import torch
    from src.inference.conditional_slot_training import run_conditional

    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(5)
    t = dict(
        v={m: torch.randn(24, 4, 2, 15, 128, generator=g) for m in ("eeg", "hb")},
        xy=torch.randn(24, 2, 4, 2, generator=g),
        y=torch.randn(24, 4, 3, 2, generator=g),
        donor=np.arange(24).reshape(-1, 3)[:, [1, 2, 0]].ravel(),
        people=np.repeat(np.arange(8), 3),
        mod="hb",
        kind="A",
        curve=[],
    )
    cfg = dict(
        width=4,
        rank=2,
        batch=3,
        lr=0.001,
        anchor_step_grid=[1, 2],
        residual_steps=2,
        residual_l2=0.1,
        folds=3,
    )
    tr = np.arange(18)
    dv = np.arange(18, 24)
    a = run_conditional(t, 17, tr, dv, cfg)
    t["y"][dv] += 123
    b = run_conditional(t, 17, tr, dv, cfg)
    for k in a[0]:
        torch.testing.assert_close(a[0][k], b[0][k])
        torch.testing.assert_close(a[5][k][0], b[5][k][0])


def test_cached_event_keeps_filter_and_raw_record_boundary():
    from src.data.four_stream_future import event_observations, sample_limits

    g = np.random.default_rng(9)
    e = g.normal(size=(11000, 4))
    h = g.normal(size=(670, 4, 2))
    ev = event_observations(e, h, 30000.0, 45000.0, cached_hb=True)
    lo, hi, _ = sample_limits(45000.0, 10)
    expected = (h[lo:hi] - h[lo : lo + 50].mean(0)).transpose(1, 0, 2).reshape(4, 15, 40)
    np.testing.assert_allclose(ev["hb_wave"], expected, atol=1e-6)
    assert ev["raw_stop"][1] == len(h) / 10 - 45 and ev["raw_stop"][0] == 4


def test_joint_null_preserves_counts_and_excludes_overlapping_windows():
    import sys
    from pathlib import Path
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
    from analyze_four_stream_transfer import temporal_null

    frame = pd.DataFrame(
        [
            dict(
                dataset="d",
                subject="s",
                record="r",
                condition="c",
                position="site",
                start_time=t,
                s=np.repeat(i, 15),
                o=np.repeat(i, 15),
            )
            for i, t in enumerate([0.0, 30.0])
        ]
    )
    result, coverage = temporal_null(frame, "hb")
    assert coverage[0]["legal_shifts"] == 1
    table = pd.DataFrame(result)
    for kind, total in [("occupancy", 30), ("transition", 28)]:
        sub = table.loc[table.kind.eq(kind)]
        assert sub.observed.sum() == sub.expected.sum() == total
    assert table.query("kind=='occupancy' and cell==0").observed.iat[0] == 15
    assert table.query("kind=='occupancy' and cell==0").expected.iat[0] == 0
    result, coverage = temporal_null(frame.assign(start_time=[0, 29]), "hb")
    assert not result and coverage[0]["legal_shifts"] == 0
