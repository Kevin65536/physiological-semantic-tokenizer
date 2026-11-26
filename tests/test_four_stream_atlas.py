import numpy as np
import pandas as pd
import pytest

from src.analysis.four_stream_atlas import (
    check_partitions,
    hb_member_slopes,
    paired_readout,
    spatial_counts,
    summarize_sites,
)


def test_array_site_join_and_valid_opportunities():
    panel = pd.DataFrame(
        [
            dict(
                dataset="d",
                split="validation",
                subject="01",
                record="a",
                parent="w",
                start=0,
                eeg_start=5.0,
                hb_start=11.0,
                status="complete",
            )
        ]
    )
    positions = pd.concat(
        [
            panel.assign(mod=mod, position=f"site{i}", x=i, y=0, Q4=0.0, event=-1, condition="", observed=i)
            for mod in ("eeg", "hb")
            for i in range(4)
        ],
        ignore_index=True,
    )
    arrays = {}
    for mod in ("eeg", "hb"):
        arrays[mod + "_y"] = np.broadcast_to(np.arange(4)[None, :, None], (1, 4, 15))
        for b in "SO":
            f = np.zeros((1, 4, 3, 4))
            f[..., "SO".index(b)] = 1
            arrays[mod + b] = f
    counts = spatial_counts(panel, positions, arrays)
    _, nodes = summarize_sites(counts)
    assert set(nodes.layer) == {"EEG-S", "EEG-O", "HB-S", "HB-O"}
    assert nodes.groupby(["position", "layer"]).occupancy.sum().eq(1).all()
    assert nodes.opportunities.eq(15).all()
    assert nodes.ci_low.isna().all()
    with pytest.raises(AssertionError):
        spatial_counts(panel, positions.iloc[::-1], arrays)
    arrays["hbO"][0, 0, 0] = [0.3, 0.7, 0, 0]
    with pytest.raises(ValueError, match="counts"):
        spatial_counts(panel, positions, arrays)


def test_split_is_subject_and_dataset_specific():
    rows = pd.DataFrame(
        dict(dataset=["a", "b"], subject=["01", "01"], record=["a", "b"], split=["train", "validation"])
    )
    check_partitions(rows)
    with pytest.raises(ValueError, match="crosses"):
        check_partitions(rows.assign(dataset="a"))
    with pytest.raises(ValueError, match="split"):
        check_partitions(rows.assign(split="test"))


def test_selected_member_colour_and_raw_support():
    t = np.arange(20) / 10
    wave = np.column_stack([2 * t, -3 * t]).ravel()
    waves = {"low_0_validation_member": np.stack([wave, wave / 4, wave * 0.75])[None]}
    row = dict(
        dataset="d",
        subject="1",
        record="a",
        group="low",
        word=0,
        split="validation",
        kind="member",
        pair=0,
        patch=14,
        crop_s=30,
        x_attribute=2.0,
        branch_attribute=1.5,
    )
    frame = pd.DataFrame([row, dict(row, group="heldout_failure")])
    result = hb_member_slopes(frame, waves, [0], "O")
    assert result.patch_start_in_parent_s.tolist() == [58, 58]
    np.testing.assert_allclose(result.HbO_slope, [2, 1.5])
    np.testing.assert_allclose(result.HbR_slope, [-3, -2.25])
    with pytest.raises(ValueError, match="manifest"):
        hb_member_slopes(frame.assign(pair=1), waves, [0], "O")


def test_readout_pairs_within_people_then_equal_weights_datasets():
    frame = pd.DataFrame(
        [
            dict(dataset=ds, subject=p, target="X", method=m, mse=v)
            for ds, p, gain in [("a", "01", 1), ("a", "02", -1), ("b", "01", 2)]
            for m, v in [("S", 4), ("SO", 4 - gain)]
        ]
    )
    _, result = paired_readout(frame, "hb")
    assert result["difference"] == 1 and result["relative_gain"] == 0.25
    assert result["people"] == 3 and result["improved"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        paired_readout(frame.iloc[:-1], "hb")
