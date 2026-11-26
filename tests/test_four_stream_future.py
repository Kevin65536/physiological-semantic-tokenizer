import numpy as np
import pytest
from src.data.four_stream_future import event_observations, lowpass_split, sample_limits


def test_native_clock_cutoff_and_future_invariance():
    rng = np.random.default_rng(1126)
    e = rng.normal(size=(12000, 4))
    h = rng.normal(size=(800, 4, 2))
    em, hm = 30001.0, 45137.0
    base = event_observations(e, h, em, hm)
    assert (base["raw_stop"] <= 4).all()
    assert np.allclose(base["raw_stop"] - base["encoder_start"], 30)
    assert (base["hb_target_start"] >= [6, 8, 10]).all()
    _, ec, _ = sample_limits(em, 200)
    _, hc, _ = sample_limits(hm, 10)
    e[ec:] = 0
    h[hc:] += [1, -2]
    alt = event_observations(e, h, em, hm)
    for k in ("hb_history", "eeg_wave", "hb_wave", "eeg_alpha"):
        np.testing.assert_array_equal(base[k], alt[k])
    np.testing.assert_allclose(
        alt["hb_target"] - base["hb_target"], np.broadcast_to([1, -2], (4, 3, 2)), atol=1e-6
    )
    assert not np.allclose(alt["eeg_target"], base["eeg_target"])
    for mod, freq in [("eeg", 12), ("hb", 0.1)]:
        x = base[mod + "_wave"]
        a, b = lowpass_split(x, mod, freq)
        np.testing.assert_allclose(a + b, x, atol=1e-6)
    with pytest.raises(ValueError, match="incomplete"):
        event_observations(e, h, 3000, 1000)


def test_all_fits_own_budget_and_no_development_scaling():
    import pandas as pd
    import torch
    from src.inference.four_stream_training import fit_reader, predictions, training_tensors

    rng = np.random.default_rng(123)
    meta = pd.DataFrame(
        dict(
            split=["train"] * 6 + ["validation"] * 2,
            subject=["a"] * 2 + ["b"] * 4 + ["c"] * 2,
            donor=[1, 0, 3, 2, 5, 4, 7, 6],
        )
    )
    data = {"xy": rng.normal(size=(8, 2, 4, 2))}
    for mod in ("eeg", "hb"):
        data[mod + "_target"] = rng.normal(size=(8, 4, 3, 2) if mod == "hb" else (8, 4, 3))
        data[mod + "_ids"] = rng.integers(0, 7, size=(8, 2, 4, 15))
        data[mod + "_alt"] = rng.integers(0, 7, size=(8, 2, 4, 15))
        for b in "SO":
            data[mod + b + "_book"] = rng.normal(size=(7, 128)).astype("float32")
    t, normalizers = training_tensors(data, meta)
    for mod in ("eeg", "hb"):
        data[mod + "_target"][-2:] += 14
    assert training_tensors(data, meta)[1] == normalizers
    cfg = dict(width=4, rank=2, steps=2, batch=3, lr=0.001)
    counts = {}
    base = None
    for arm in [
        "A_base",
        "A_H",
        "A_HE",
        "A_wrong",
        "A_generic",
        "eeg_S",
        "eeg_O",
        "hb_add",
        "eeg_interaction",
        "eeg_wrong",
        "eeg_generic",
        "hb_alt_add",
        "hb_alt_interaction",
    ]:
        model, loss = fit_reader(arm, 17, t, cfg, base)
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert len(loss) == 2 and np.isfinite(loss).all()
        with torch.no_grad():
            if arm == "A_base":
                base = predictions(model, arm, t, np.arange(8))
            y = predictions(model, arm, t, t["dev"], base)
            assert y.shape == (2, 4, 6 if arm.startswith(("A_", "hb")) else 3)
    assert counts["A_H"] == counts["A_HE"] == counts["A_wrong"] == counts["A_generic"]
    assert counts["eeg_interaction"] == counts["eeg_generic"]
