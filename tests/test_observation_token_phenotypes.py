"""In-memory positive and falsifying controls; no paths, datasets or fitting."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.analysis.observation_token_phenotypes import run_phenotypes
from src.tokenizers.factorized_observation import _properties


def _alpha(narrow, fraction, beta_background):
    # Fix rectangular 8–13 fraction and RMS *exactly*. Different remaining
    # routing bands can distinguish IDs even on the matched validation panel.
    f = np.arange(201)/2
    a = 1/np.sqrt(np.maximum(f, .5)); a[0] = 0
    a *= (1 + 1.5*((f >= 19) & (f <= 23))) if beta_background else (1 + (f < 6))
    a[f >= 45] = .025
    band = (f >= 8) & (f < 13); a[band] = 0
    w = np.full(10, .04) if narrow else np.ones(10); w[4] = 1  # 10 Hz
    rest = 2*(a@a) - a[-1]**2
    a[band] = (w/np.linalg.norm(w))*np.sqrt(rest*fraction/(1-fraction)/2)
    # Linear phase: deterministic (no random null periodogram lottery).
    x = np.fft.irfft(a*(-1.)**np.arange(201), n=400)
    return (x/x.std())[None, :]


def _high(which):
    f = np.arange(201)/2
    if which == "broad":
        a = f >= 45
    elif which == "both":
        a = (f == 50) | (f == 60)
    else:
        a = f == which
    x = np.fft.irfft(a*(-1.)**np.arange(201), n=400)
    return (x/x.std())[None, :]


def _hb(p):
    t = (np.arange(20)-9.5)/10
    if p in (0, 1):
        return np.array([t, t if p == 1 else -t])
    return np.repeat([[2.], [-1.]], 20, axis=1)  # signed levels alone, no trend


def _synthetic(mods=("eeg", "hb"), *, reverse=False):
    frames, waves, codecs = {}, {}, {}
    for m in mods:
        for split in ("train", "validation"):
            val = split == "validation"; frames_m, x = [], []
            if m == "eeg":
                ids = [0]*5 + [1]*5 + [2]*5
            elif val and reverse:
                ids = [0, 1]*7+[0]  # current word unchanged, next class reverses H2
            else:
                ids = [0]*6+[1]*6+[2]*3
            for d in ("lib-a", "lib-b"):
                for s in ("a", "b"):  # same subject strings across different datasets
                    for j, p in enumerate(ids):
                        row = dict(dataset=d, subject=f"{split}-{s}", record="r", parent="w", start_s=0,
                                   position="site", patch=j, pattern=p, blind=(j % 3))
                        if m == "eeg" and p in (0, 1):
                            narrow = bool((p == 0) != (val and reverse))
                            wave = _alpha(narrow, .45 if val or p == 0 else .24, p == 1)
                        elif m == "eeg":
                            wave = _high([50, 60, "both", "broad", 80][j-10])
                        else:
                            wave = _hb(p)
                        frames_m.append(row); x.append(wave)
            key = m, split
            x = np.asarray(x); info = pd.DataFrame(frames_m)
            direct = _properties(x, m)
            for name, col in zip(("alpha", "beta") if m == "eeg" else ("so", "sr"), (2, 3)):
                info[name] = direct[:, col]
            info["rms"] = ((x-x.mean(-1, keepdims=True))**2).mean((1, 2))**.5
            frames[key], waves[key] = info, x
            if not val:
                centers = np.stack([direct[info.pattern == p].mean(0) for p in range(3)]+[direct[0]])
                if m == "eeg":
                    centers[-1] = [0, 0, 1, 0, 0, 0]  # tempting but no TRAIN support
                codecs[m] = SimpleNamespace(pattern_centers_=centers, pattern_counts_=np.bincount(info.pattern, minlength=4))
    return frames, waves, codecs


def _row(out, m, p, split="validation"):
    return next(r for r in out["support"] if (r["modality"], r["pattern_id"], r["split"]) == (m, p, split))


def test_matched_narrow_peak_and_future_color_direction_are_auditable():
    out = run_phenotypes(*_synthetic())
    json.dumps(out, allow_nan=False)
    assert {k: v["pattern_id"] for k, v in out["representatives"].items()} == {
        "eeg_alpha": 0, "eeg_high": 2, "hb_opposed": 0, "hb_common": 1}
    assert len(out["support"]) == 2*2*4  # not a long per-frequency or per-patch table
    alpha, broad = _row(out, "eeg", 0), _row(out, "eeg", 1)
    assert alpha["direct_q"]["alpha"] == pytest.approx(broad["direct_q"]["alpha"])
    assert alpha["direct_q"]["rms"] == pytest.approx(broad["direct_q"]["rms"])
    assert alpha["peak_hits"] == alpha["n"] == 20 and alpha["hit_peak_hz_q"] == [10.]*3
    assert broad["peak_hits"] == 0 < broad["max_excess_db_q"][1]  # broad excess is NOT a narrow peak
    assert _row(out, "eeg", 3)["construction_label"] == "unlabelled_sparse_train"
    assert alpha["by_dataset"]["lib-a"] == dict(n=10, subjects=2, subjects_with_two=2, positions={"site": 10})
    hf = _row(out, "eeg", 2)
    assert hf["high_classes"] == {"little_high_power": 0, "line50_like": 4, "line60_like": 4,
                                  "both_lines_like": 4, "diffuse_like": 4, "other_or_mixed": 4}
    assert len(hf["subject_profiles"]) == 4
    assert all(sum(p["high_classes"][kind] for p in hf["subject_profiles"]) == count
               for kind, count in hf["high_classes"].items())
    assert alpha["hit_prominence_db_q"][1] >= 3.
    assert _row(out, "hb", 0)["interior_classes"]["opposed_o_up"] == 24
    assert _row(out, "hb", 1)["interior_classes"]["common_up"] == 24
    assert _row(out, "hb", 2)["interior_classes"]["unresolved"] == 12
    for key, truth in (("eeg_alpha_peak", 1.), ("hb_next_opposed", 5/6)):
        stat = out["comparisons"][key]
        assert stat["status"] == "direction_observed" and stat["delta"] == pytest.approx(truth)
        assert stat["ci95"] == pytest.approx([truth]*2)
        people = [s for s in out["subjects"] if s["comparison"] == key]
        assert len(people) == stat["support"]["qualified_subjects"] == 4
        assert all(s["qualified"] and np.allclose(s["mean_abs_within_cell_gaps"], 0, atol=1e-12) for s in people)
        for hit, rate in [("matched_target_hits", "target_rate"), ("matched_reference_hits", "reference_rate")]:
            p = pd.DataFrame(people)
            rebuild = (p[hit]/p.pair_mass).groupby(p.dataset).mean().mean()
            assert stat[rate] == pytest.approx(rebuild)


def test_train_choices_frozen_even_when_both_predictions_reverse():
    good = run_phenotypes(*_synthetic())
    bad = run_phenotypes(*_synthetic(reverse=True))
    assert good["representatives"] == bad["representatives"]
    assert [r for r in good["support"] if r["split"] == "train"] == [r for r in bad["support"] if r["split"] == "train"]
    assert all(c["status"] == "direction_not_observed" and c["delta"] == pytest.approx(-1.)
               for c in bad["comparisons"].values())


@pytest.mark.parametrize("separate", ["alpha", "rms", "position", "levels"])
def test_no_extrapolation_across_nuisance_or_native_position(separate):
    m = "hb" if separate == "levels" else "eeg"
    frames, waves, codecs = _synthetic([m]); key = m, "validation"
    # One person retains exact support; the other three are outside common
    # strata. A perfectly positive tiny subset may NOT license a positive name.
    f, x = frames[key], waves[key]
    change = (f.pattern == 1) & ~((f.dataset == "lib-a") & (f.subject == "validation-a"))
    if separate == "position":
        f.loc[change, "position"] = "other_site"
    elif separate == "levels":
        x[change, 0] += 1.2  # only current HbO level separates; slope is unchanged
    elif separate == "alpha":
        x[change] = _alpha(False, .30, True)
    else:
        x[change] *= 2.5
    frames[key] = f.drop(columns=[c for c in ("alpha", "beta", "rms") if c in f])
    result = run_phenotypes(frames, waves, codecs)
    stat = next(iter(result["comparisons"].values()))
    assert stat["status"] == "insufficient_common_support" and stat["delta"] is stat["target_rate"] is None
    assert stat["overlap_only"]["delta"] > 0 and stat["support"]["qualified_subjects"] == 1
    assert stat["support"]["by_dataset"]["lib-b"]["target_n"] > 0
    assert stat["support"]["by_dataset"]["lib-b"]["pair_mass"] == 0


def test_hb_endpoint_spikes_are_not_interior_trends_or_future_evidence():
    frames, waves, codecs = _synthetic(["hb"]); key = "hb", "validation"
    f, x = frames[key], waves[key]
    ix = f.pattern == 0; x[ix] = 0
    x[ix, 0, 0] = -5; x[ix, 0, -1] = 5; x[ix, 1, 0] = 5; x[ix, 1, -1] = -5
    frames[key] = f.drop(columns=["so", "sr", "rms"])
    result = run_phenotypes(frames, waves, codecs)
    opposed = _row(result, "hb", 0)
    assert opposed["ols_both_center_signs"] == opposed["interior_classes"]["unresolved"] == opposed["n"]
    assert opposed["next_opposed_hits"] == 0
    assert result["comparisons"]["hb_next_opposed"]["status"] == "direction_not_observed"


def test_next_patch_never_wraps_or_bridges_missing_patch_or_series():
    frames, waves, codecs = _synthetic(["hb"]); key = "hb", "validation"
    # Only patch 12 -> 13 and 13 -> 14 exist in the SAME physical series. No
    # 14 -> 0, 0 -> 2, or adjacency from a different record, site or crop.
    base = dict(dataset="lib-a", subject="v", record="r", parent="w", start_s=0, position="site")
    spec = [(12, 0, {}), (13, 0, {}), (14, 1, {}), (0, 0, {"start_s": 30}), (2, 0, {"start_s": 30}),
            (1, 0, {"start_s": 30, "position": "far"}), (1, 0, {"start_s": 30, "parent": "other"}),
            (1, 0, {"start_s": 30, "record": "other"})]
    frames[key] = pd.DataFrame([dict(base, **dict(a, patch=j, pattern=p)) for j, p, a in spec]).iloc[::-1]
    waves[key] = np.stack([_hb(p) for _, p, _ in spec][::-1])
    result = run_phenotypes(frames, waves, codecs)
    assert _row(result, "hb", 0)["next_evaluable"] == 2 and _row(result, "hb", 0)["next_opposed_hits"] == 1
    assert _row(result, "hb", 1)["next_evaluable"] == 0
    assert result["comparisons"]["hb_next_opposed"]["delta"] is None


def test_unsupported_words_are_not_chosen_and_constant_eeg_is_not_a_peak():
    frames, waves, codecs = _synthetic(["eeg"])
    for split in ("train", "validation"):
        key = "eeg", split
        use = (frames[key].dataset == "lib-a") & frames[key].subject.str.endswith("a")
        frames[key] = frames[key].loc[use].drop(columns=["alpha", "beta", "rms"])
        waves[key] = waves[key][use]
    waves["eeg", "validation"][:] = .37
    result = run_phenotypes(frames, waves, codecs)
    assert result["representatives"]["eeg_alpha"]["pattern_id"] is None
    assert result["comparisons"]["eeg_alpha_peak"]["status"] == "no_train_representative"
    assert _row(result, "eeg", 0)["peak_evaluable"] == 0
    assert _row(result, "eeg", 0)["max_excess_db_q"] is None
    json.dumps(result, allow_nan=False)


def test_invalid_keys_rejected_before_access_and_misalignment_detected():
    class Guard(dict):
        def __getitem__(self, key):
            raise AssertionError("must reject invalid split keys before array/frame access")
    frames, waves, codecs = _synthetic(["eeg"]); key = "eeg", "validation"
    with pytest.raises(ValueError, match="train/validation"):
        run_phenotypes(Guard(frames), Guard({**waves, ("eeg", "test"): object()}), codecs)
    invalid = {**frames, key: frames[key].assign(patch=15)}
    with pytest.raises(ValueError, match="patch must"):
        run_phenotypes(invalid, waves, codecs)
    with pytest.raises(ValueError, match="duplicate"):
        run_phenotypes({**frames, key: frames[key].assign(patch=0)}, waves, codecs)
    with pytest.raises(ValueError, match="crosses"):
        run_phenotypes({**frames, key: frames[key].assign(subject=frames[key].subject.str.replace("validation", "train"))}, waves, codecs)
    with pytest.raises(ValueError, match="disagrees"):
        run_phenotypes(frames, {**waves, key: waves[key][::-1]}, codecs)
