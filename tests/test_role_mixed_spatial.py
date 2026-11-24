import numpy as np
import pytest

from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.role_mixed_information import (
    RoleInformationPanel,
    mix_by_role,
    select_fixed_mask,
)
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit
from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary
from src.tokenizers.spatial_local_gain import LocalEnergyGainDecoder
from src.tokenizers.role_mixed_spatial import (
    FixedRoleMixedDecoder,
    load_role_mixed_decoder,
)


def test_role_stitching_is_exact_under_original_subject_record_crop_hierarchy():
    rng = np.random.default_rng(736)
    rows = [
        dict(dataset=f"d{i // 4}", subject=f"s{i // 2}", record=f"r{i}", parent=f"p{i}")
        for i in range(8)
    ]
    examples = np.array(
        [(i, start, p) for i in range(8) for start in [0, 30] for p in range(1 + i % 4)]
    )
    n = len(examples)
    role = np.arange(n) % 5
    low, high = rng.uniform(0.1, 2, size=(2, n, 2, 3))
    high[:, 0] = low[:, 0]
    high[7] = low[7] = np.nan
    bank = dict(examples=examples, fixed_band=role, valid=np.ones((n, 15), bool))
    masses = rng.uniform(0.7, 1, size=(2, n))
    panel = RoleInformationPanel(high, low, masses[0], masses[1], bank, rows)
    for mask in range(32):
        expected = base_training_aggregate(
            {"nmse": mix_by_role(high, low, role, mask)}, examples, rows
        )["metrics"]["nmse"]
        actual = panel.evaluate(mask)
        np.testing.assert_allclose(
            actual["hierarchy"]["macro"], expected["macro"], rtol=1e-14, atol=1e-14
        )
        for a, b in zip(actual["hierarchy"]["by_subject"], expected["by_subject"]):
            assert (a["dataset"], a["subject"]) == (b["dataset"], b["subject"])
            np.testing.assert_allclose(a["value"], b["value"], rtol=1e-14, atol=1e-14)
    assert (
        select_fixed_mask(
            {0: {"passed": True}, 1: {"passed": True}, 24: {"passed": True}}
        )
        == 1
    )
    assert select_fixed_mask({0: {"passed": False}}) is None
    perturbed = high.copy()
    perturbed[:, 0] += 1e-6
    with pytest.raises(AssertionError):
        RoleInformationPanel(perturbed, low, masses[0], masses[1], bank, rows)


def test_reindexed_role_dictionary_conditional_tables_match_actual_mixed_fitting():
    rng = np.random.default_rng(838)
    role = np.repeat(np.arange(5), 200)
    background = role * 4 + np.arange(len(role)) % 4
    high, low = rng.integers(1, 8, len(role)), rng.integers(1, 4, len(role))
    target = rng.normal(size=(len(role), 10)) + role[:, None] * 3
    fit = np.arange(len(role)) % 3 != 0
    fields = {"theta": (0, 1), "alpha": (1, 2), "RMS": (2, 10)}

    def read(ids, capacity):
        model = SparsePatternAudit(5 * capacity, 20, [], 10, fields)
        words = role * capacity + ids
        model.fit(words[fit], background[fit], target[fit])
        return model.read(words[~fit], background[~fit])[2]

    y_high, y_low = read(high, 8), read(low, 4)
    for mask in [0, 1, 14, 31]:
        mixed = mix_by_role(high, low, role, mask)
        expected = mix_by_role(y_high, y_low, role[~fit], mask)
        np.testing.assert_allclose(read(mixed, 8), expected, atol=1e-14, rtol=1e-14)


def mixed_fixture():
    rng = np.random.default_rng(436)
    count = 10
    times = np.arange(400) / 200
    models = []
    for size in [8, 4]:
        atoms, centers, weights = {}, {}, {}
        for branch in "SO":
            for role, frequency in enumerate([2.0, 6.0, 10.0, 20.0, 36.0]):
                atoms[branch, role] = (
                    np.sin(
                        2 * np.pi * frequency * times[None] + np.arange(size)[:, None]
                    )
                    * np.arange(size)[:, None]
                    / size
                )
                centers[branch, role] = np.zeros((size - 1, 4))
                weights["fixture", branch, role] = rng.uniform(0.2, 1, size=(2, count))
        meta = dict(
            schema="spatial_waveform_dictionary_v1",
            codebook_size=size,
            positions={"fixture": [str(j) for j in range(count)]},
            output_projection_hz=[1, 45],
        )
        models.append(
            SpatialWaveformDictionary(
                meta,
                {"S": np.full(3001, 0.6), "O": np.full(3001, 0.4)},
                centers,
                atoms,
                weights,
            )
        )
    high_raw, low = models
    state = dict(
        mean=np.zeros(32 + count),
        scale=np.ones(32 + count),
        coefficient=np.zeros((4, 32 + count)),
        intercept=np.zeros(4),
    )
    settings = dict(
        residual_gain_bounds=[0.5, 2.0],
        total_gain_bounds=[0.25, 4.0],
        smoothing_sigma_s=0.25,
    )
    high = LocalEnergyGainDecoder(
        high_raw,
        {"fixture": state},
        {"fixture": [1.2] * 10},
        settings,
        0.25,
        dict(high_raw.metadata, schema="spatial_AS_local_gain_v1"),
    )
    return (
        high,
        low,
        {b: rng.integers(1, 8, (1, count, 15), dtype=np.int32) for b in "SO"},
        {b: rng.integers(1, 4, (1, count, 15), dtype=np.int32) for b in "SO"},
    )


def test_exact_pure_endpoints_and_one_emitted_stream_only():
    high, low, high_ids, low_ids = mixed_fixture()
    for mask, mode, ids, reference in [
        (31, "raw", high_ids, high.source),
        (31, "gain", high_ids, high),
        (0, "raw", low_ids, low),
    ]:
        model = FixedRoleMixedDecoder(high, low, {"S": mask, "O": mask}, mode=mode)
        for branch, value in reference.decode_ids(ids, "fixture").items():
            np.testing.assert_array_equal(
                value, model.decode_ids(ids, "fixture")[branch]
            )
    model = FixedRoleMixedDecoder(high, low, {"S": 1, "O": 25}, mode="gain")
    emitted = model.emit_ids(high_ids, low_ids, "fixture")
    changed_high, changed_low = {}, {}
    for branch in "SO":
        role = model.raw.roles("fixture", branch)
        used = ((model.masks[branch] >> role) & 1).astype(bool)
        changed_high[branch] = high_ids[branch].copy()
        changed_high[branch][:, ~used] = 7
        changed_low[branch] = low_ids[branch].copy()
        changed_low[branch][:, used] = 3
    repeated = model.emit_ids(changed_high, changed_low, "fixture")
    for branch in "SO":
        np.testing.assert_array_equal(emitted[branch], repeated[branch])
    wrong = {b: v.copy() for b, v in emitted.items()}
    wrong["S"][:, 1] = 7
    with pytest.raises(ValueError, match="role dictionary"):
        model.decode_ids(wrong, "fixture")
    rate = model.rate("fixture")
    assert rate["extra_integers"] == 0 and rate["fixed_maximum_width_bits_each_ID"] == 3


def test_frozen_mixed_manifest_replays_from_only_integer_arrays(tmp_path):
    import hashlib

    high, low, high_ids, low_ids = mixed_fixture()
    raw_path, high_path, low_path = (
        tmp_path / "high_raw.npz",
        tmp_path / "high_gain.npz",
        tmp_path / "low.npz",
    )
    high.source.save(raw_path)
    low.save(low_path)
    high.metadata.update(
        source_decoder=str(raw_path),
        source_decoder_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    )
    high.save(high_path)
    model = FixedRoleMixedDecoder(high, low, {"S": 9, "O": 1}, mode="gain")
    file = tmp_path / "mixed.npz"
    model.save(file, high_path=high_path, low_path=low_path)
    loaded = load_role_mixed_decoder(file)
    ids = model.emit_ids(high_ids, low_ids, "fixture")
    for branch, value in model.decode_ids(ids, "fixture").items():
        np.testing.assert_array_equal(value, loaded.decode_ids(ids, "fixture")[branch])
