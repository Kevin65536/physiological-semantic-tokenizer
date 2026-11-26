"""Synthetic checks for frozen D128 coordinates, component attributes and evidence indexing."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from analyze_aligned_token_diagnostics import attrs
from audit_aligned_tokenizer import CONTRACT


def test_hb_joint_slope_color_order_and_time_units():
    t = np.arange(20) / 10
    phases = np.stack([2 * t + 1, 3 - 0.5 * t], axis=-1).reshape(1, 40)
    names, values = attrs(phases, "hb")
    value = dict(zip(names, values[0]))
    assert value["HbO_slope"] == pytest.approx(2)
    assert value["HbR_slope"] == pytest.approx(-0.5)
    assert value["peak_lag"] == pytest.approx(-1.9)
    assert value["color_cov"] < 0


def test_alpha_power_respects_rms_and_amplitude_scaling():
    t = np.arange(400) / 200
    x = np.sin(2 * np.pi * 10 * t)
    names, values = attrs(np.stack((x, 2 * x)), "eeg")
    c = {name: values[:, i] for i, name in enumerate(names)}
    assert c["alpha_power"][1] == pytest.approx(4 * c["alpha_power"][0])
    assert c["alpha_frac"][0] > 0.999
    np.testing.assert_allclose(c["alpha_frac"][0], c["alpha_frac"][1])
    assert c["halfpower_width"].min() >= 0.5


def test_frozen_audit_matrix_and_tensor_contract():
    cfg = yaml.safe_load(CONTRACT.read_text())
    assert cfg["shape"]["eeg"] == [15, 400] and cfg["shape"]["hb"] == [15, 40]
    assert cfg["models"] == {m: dict(d=128, K=k, seeds=[17]) for m, k in [("eeg", 8192), ("hb", 1024)]}
    assert cfg["scope"]["test"] == cfg["scope"]["confirmation"] == "forbidden"
    assert not cfg["scope"]["tokenizer_refit"]
    table = pd.read_csv(CONTRACT.with_name("aligned_token_audit_catalog.tsv"), sep="\t")
    assert table.ID.tolist() == [f"A{i:02}" for i in range(1, 65)]
    assert set(table.loc[table["状态"] == "需联合模型", "ID"]) == {"A48", "A49", "A51", "A52", "A53", "A63"}
    assert not table.isna().any().any()
