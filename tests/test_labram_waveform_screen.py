"""Synthetic pairing and scale checks for the bounded observation-codec screen."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
import run_labram_waveform_screen as screen


def test_pairing_respects_split_site_and_absolute_clock():
    rows = []
    for split, onset in [
        ("train", 0),
        ("validation", 0),
        ("validation", 2000),
        ("validation", 6000),
    ]:
        for mod, position in screen.C["positions"].items():
            rows.append(
                dict(
                    dataset=screen.C["dataset"],
                    split=split,
                    modality=mod,
                    subject="P",
                    record="R",
                    parent=str(onset),
                    native_rate_hz=200,
                    native_start_sample=onset,
                    positions=[position],
                    inputs=dict(score=[dict(start_s=0, positions=[0])]),
                )
            )
    pairs = screen.panel(rows, "validation")
    assert [p["eeg"][0] for p in pairs] == [2, 6]
    assert [p["hb"][0] for p in pairs] == [3, 7]
    rows[7]["inputs"]["score"][0]["positions"] = []
    assert len(screen.panel(rows, "validation")) == 1


def test_score_uses_frozen_color_variance():
    x = np.array([[[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]])
    nmse, corr = screen.score(x, x + 2, np.array([[2.0, 8.0]]))
    np.testing.assert_allclose(nmse, [[2.0, 0.5]])
    np.testing.assert_allclose(corr, [[1.0, 1.0]])
    assert np.isnan(screen.score(x, np.zeros_like(x), np.ones((1, 2)))[1]).all()
