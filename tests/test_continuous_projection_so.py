from pathlib import Path
import numpy as np
import yaml
from src.data.temporal_projection_so import continuous_indices, continuous_projection

ORDER = np.arange(2, 90)


def test_rank_closure_gap_and_phase():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(3261, 2))
    mask = np.ones_like(x, bool)
    mask[1400:1650, 0] = False
    y = continuous_projection(x, mask, ORDER)
    np.testing.assert_allclose(continuous_projection(y, mask, ORDER), y, atol=1e-12)
    z = x.copy()
    z[1400:, 0] = 1e5
    np.testing.assert_allclose(continuous_projection(z, mask, ORDER)[:1400], y[:1400], atol=1e-12)
    assert not y[~mask].any() and abs(np.sum(y * (x - y))) < 1e-10
    for n in [3, 250, 400, 18000]:
        ix = continuous_indices(n, ORDER)
        assert len(ix) <= n * 64 // 400 and 0 not in ix
        f = np.arange(n) * 100 / n
        assert set(np.flatnonzero((f >= 4) & (f < 13))) <= set(ix)
    t = np.arange(18000) / 200
    for hz in [4, 8, 10, 11, 11.25, 11.5, 12.5, 12.95]:
        s = np.sin(2 * np.pi * hz * t[:, None] + np.arange(4)[None, :])
        y = continuous_projection(s, np.ones_like(s, bool), ORDER)
        assert (np.sum(y * y, 0) / np.sum(s * s, 0) > 0.95).all()


def test_contract():
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / 'experiments/configs/physiology_semantic_tokenizer/continuous_projection_so_v2.yaml'
        ).read_text()
    )
    assert '1000 warmup plus 9000 hard' in cfg['formal'] and 'only when target gate passes' in cfg['stopping']


def test_projection_does_not_read_regression_or_excluded_support():
    from experiments.scripts.check_continuous_so_targets import admissible

    row = dict(
        inputs=dict(
            train=[dict(start_s=0, positions=[1])],
            score=[dict(start_s=30, positions=[0])],
            regression=[dict(start_s=60, positions=[0, 1])],
        ),
        qc_spans=[(1, 5000, 6000)],
    )
    mask = np.ones((18000, 2), bool)
    valid = admissible(row, mask)
    assert valid[:, 0].sum() == 6000 and valid[:, 1].sum() == 5000 and not valid[12000:].any()
    rng = np.random.default_rng(7)
    s = rng.normal(size=valid.shape)
    ref = continuous_projection(s, valid, ORDER)
    s[~valid] = 1e20
    np.testing.assert_array_equal(continuous_projection(s, valid, ORDER), ref)


def test_frozen_input_scale_changes_only_variance_coordinates():
    from src.data.dual_component_tokenizer import base_tokenizer_scales

    spec = {'target_variance_floor': 0.01, 'difference_variance_floor_fraction': 0.001}
    fields = ['x', 's', 'o', 'dx', 'ds', 'do']
    row = dict(
        split='train',
        dataset='public',
        record='r',
        subject='a',
        modality='eeg',
        positions=['P3'],
        moments={k: dict(count=[[4]], sum=[[0]], second=[[8]]) for k in fields},
    )
    orig = base_tokenizer_scales([row], spec)
    orig['positions']['eeg']['public']['P3']['scale'] = 4.0
    ref = base_tokenizer_scales([row], spec, input_scales=orig)['positions']['eeg']['public']['P3']
    assert ref['scale'] == 4.0 and ref['target_variance']['s'] == [0.125]
