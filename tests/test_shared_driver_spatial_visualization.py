"""Synthetic export, scientific identities and bitmap delivery contracts."""
from __future__ import annotations

from copy import deepcopy
import json

import fitz
import numpy as np
import pytest

from experiments.scripts import render_shared_driver_spatial as render


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = render.create_demo_bundle(tmp_path_factory.mktemp("spatial-viz") / "bundle")
    manifest, arrays = render.load_bundle(path)
    return path, manifest, arrays


def test_demo_has_explicit_generator_truth_and_disjoint_scoring(bundle):
    _, manifest, arrays = bundle
    assert manifest["kind"] == "synthetic_demo"
    assert "mm" not in manifest["units"]["geometry"]
    assert arrays["hb_observation"].shape[1:] == (4, 2)
    assert not np.any(arrays["visible_hb"] & arrays["score_hb"])
    # A whole paired channel is hidden, not one chromophore with its
    # algebraically equivalent information left in another field.
    assert np.any(np.all(arrays["score_hb"], axis=(0, 2)))
    np.testing.assert_array_equal(arrays["visible_hb"][:, :, 0],
                                  arrays["visible_hb"][:, :, 1])
    assert "prediction_hb" not in arrays  # A generator is not a fitted predictor.
    assert not np.allclose(arrays["hb_k1"], arrays["hb_k2"])


def test_duplicate_cores_are_an_exact_observational_counterexample(bundle):
    _, _, arrays = bundle
    np.testing.assert_allclose(arrays["equivalence_total_a"],
                               arrays["equivalence_total_b"], rtol=0, atol=1e-12)
    assert np.max(np.abs(arrays["equivalence_components_a"]
                         - arrays["equivalence_components_b"])) > 1e-5
    for suffix in ("a", "b"):
        np.testing.assert_allclose(arrays[f"equivalence_components_{suffix}"].sum(axis=2),
                                   arrays[f"equivalence_total_{suffix}"], atol=1e-12)


def test_measured_export_does_not_inherit_synthetic_truth(bundle):
    _, original_manifest, original_arrays = bundle
    manifest = deepcopy(original_manifest)
    manifest["kind"] = "measured_export"
    manifest.pop("demo_status", None)
    manifest.pop("spatial_permutation", None)
    arrays = {k: v.copy() for k, v in original_arrays.items() if k in render.REQUIRED_ARRAYS}
    # An actual K=1 export remains renderable without inventing a second mode.
    manifest["names"]["modes"] = ["固定局部模式"]
    manifest["W"] = [[1.0] for _ in manifest["names"]["hb"]]
    render.validate_bundle(manifest, arrays)
    arrays["hb_truth"] = original_arrays["hb_truth"]
    with pytest.raises(ValueError):
        render.validate_bundle(manifest, arrays)


def test_invalid_predictions_cannot_bridge_missing_intervals(bundle):
    _, manifest, original = bundle
    arrays = {k: v.copy() for k, v in original.items()}
    arrays["prediction_hb"] = arrays["hb_truth"].copy()
    arrays["prediction_valid"] = np.ones_like(arrays["visible_hb"])
    arrays["prediction_valid"][10:20, 0] = False
    with pytest.raises(ValueError):
        render.validate_bundle(manifest, arrays)
    arrays["prediction_hb"][~arrays["prediction_valid"]] = np.nan
    render.validate_bundle(manifest, arrays)


@pytest.mark.parametrize("mutation", ["clock", "overlap", "pair_mask", "shape", "visible_nan"])
def test_reject_ambiguous_or_invalid_scientific_support(bundle, mutation):
    _, manifest, original = bundle
    arrays = {k: v.copy() for k, v in original.items()}
    if mutation == "clock":
        arrays["time"][1] = arrays["time"][0]
    elif mutation == "overlap":
        arrays["score_hb"][arrays["visible_hb"]] = True
    elif mutation == "pair_mask":
        arrays["visible_hb"][0, 0, 1] = not arrays["visible_hb"][0, 0, 0]
    elif mutation == "shape":
        arrays["hb_observation"] = arrays["hb_observation"][:, :-1]
    else:
        arrays["hb_observation"][arrays["visible_hb"]] = np.nan
    with pytest.raises(ValueError):
        render.validate_bundle(deepcopy(manifest), arrays)


def test_load_is_explicit_and_does_not_traverse_to_other_artifacts(bundle, tmp_path):
    _, manifest, _ = bundle
    changed = deepcopy(manifest)
    changed["arrays_file"] = "../outside.npz"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        render.load_bundle(path)


def test_retained_exports_cannot_be_overwritten(bundle, tmp_path):
    path, _, _ = bundle
    output = tmp_path / "retained"
    output.mkdir()
    sentinel = output / "retained.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        render.render_bundle(path, output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_bitmap_atlas_keeps_readable_text_and_synthetic_identity(bundle, tmp_path):
    path, _, _ = bundle
    output = tmp_path / "export"
    render.render_bundle(path, output)
    assert len(list((output / "figures").glob("*.png"))) == 6
    with fitz.open(output / "FIGURE_ATLAS.pdf") as pdf:
        assert len(pdf) == 6
        for page in pdf:
            text = page.get_text()
            assert "合成" in text
            assert len(text.strip()) > 30
            images = page.get_images(full=True)
            assert images
            # Every complete figure is a bitmap, at the declared final size.
            for info in page.get_image_info():
                bbox = fitz.Rect(info["bbox"])
                dpi = info["width"] * 72 / bbox.width
                assert 199 <= dpi <= 301
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit


@pytest.fixture(scope="module")
def whole_head_bundle(tmp_path_factory):
    path = render.create_whole_head_demo_bundle(tmp_path_factory.mktemp("whole-head-viz") / "bundle")
    manifest, arrays = render.load_bundle(path)
    return path, manifest, arrays


def test_whole_head_uses_public_layout_but_explicit_synthetic_signals(whole_head_bundle):
    _, manifest, arrays = whole_head_bundle
    assert manifest["schema"] == "shared_driver_spatial_visualization_v2"
    assert manifest["view"] == "whole_head"
    assert manifest["kind"] == "synthetic_demo"
    assert len(manifest["names"]["eeg"]) == 64
    assert len(manifest["names"]["hb"]) == 51
    assert "AFZ" not in manifest["names"]["eeg"]
    assert manifest["head"]["reference"]["name"] == "AFZ"
    assert manifest["head"]["x_positive"] == "figure_right"
    assert manifest["head"]["y_positive"] == "anterior"
    xy = np.asarray(manifest["geometry"]["eeg_xy"])
    fp1, fp2 = [manifest["names"]["eeg"].index(name) for name in ("FP1", "FP2")]
    assert xy[fp1, 0] < 0 < xy[fp2, 0]
    assert xy[fp1, 1] > 0 and xy[fp2, 1] > 0
    np.testing.assert_allclose(arrays["hb_observation"].sum(axis=-1), arrays["hbt"], atol=1e-12)


def test_common_cause_correlation_does_not_create_intervention_effect(whole_head_bundle):
    _, manifest, arrays = whole_head_bundle
    nodes = manifest["example_nodes"]
    a, b, c, d = [nodes[name]["selected_index"] for name in "ABCD"]
    ah, bh = nodes["A"]["hb_index"], nodes["B"]["hb_index"]
    assert arrays["corr_hbt"][ah, bh] > .5
    assert abs(arrays["partial_hbt"][ah, bh]) < abs(arrays["corr_hbt"][ah, bh])
    assert arrays["known_direct"][a, b] == 0
    np.testing.assert_allclose(arrays["do_response"][a, :, b], 0., atol=1e-12)
    assert arrays["known_direct"][a, c] > 0
    assert arrays["known_direct"][c, d] < 0
    assert arrays["known_direct"][a, d] == 0
    assert arrays["known_total_effect"][a, d] < 0  # An indirect effect survives.


def test_intervention_arrays_match_structural_replay(whole_head_bundle):
    _, _, arrays = whole_head_bundle
    selected = arrays["selected_indices"]
    transition = arrays["known_A"]
    # A pulse is applied to the source state; future responses, not the
    # instantaneous pulse, are exported in row-source order.
    state = np.zeros(len(transition))
    state[selected[0]] = 1.
    expected = []
    for _ in arrays["do_time"]:
        state = transition @ state
        expected.append(state[selected])
    np.testing.assert_allclose(arrays["do_response"][0], expected, atol=1e-12)
    np.testing.assert_allclose(arrays["known_total_effect"],
                               arrays["do_response"].mean(axis=1), atol=1e-12)


def test_pairwise_correlation_keeps_actual_missing_support():
    signals = np.array([[0., 0., 1.], [1., 2., 1.], [2., 4., 1.],
                        [3., np.nan, 1.], [4., 8., 1.]])
    correlation, count = render.correlation_matrix(signals, min_count=3)
    assert count[0, 1] == 4
    assert correlation[0, 1] == pytest.approx(1.)
    assert np.isnan(correlation[0, 2])  # Constant is undefined, not zero.


def test_heatmap_never_extrapolates_or_bridges_unmeasured_gap():
    xy = np.array([[-.75, -.1], [-.55, -.1], [-.75, .1], [-.55, .1],
                   [.55, -.1], [.75, -.1], [.55, .1], [.75, .1]])
    head = dict(center_xy=[0., 0.], radius=1.)
    field = render.interpolate_head_field(xy, np.arange(8.), head, grid_size=101)
    x, y, z = field["x"], field["y"], field["z"]
    assert np.isnan(z[(x*x + y*y) > 1.]).all()
    assert np.isnan(z[np.abs(x) < .3]).all()
    assert np.isfinite(z).any()
    sparse = render.interpolate_head_field(xy[:2], np.arange(2.), head, grid_size=21)
    assert sparse["method"] == "points_only"
    assert not np.isfinite(sparse["z"]).any()


def test_measured_label_cannot_inherit_generator_causal_truth(whole_head_bundle):
    _, original, arrays = whole_head_bundle
    manifest = deepcopy(original)
    manifest["kind"] = "measured_export"
    with pytest.raises(ValueError):
        render.validate_bundle(manifest, arrays)


@pytest.mark.parametrize("mutation", ["crossmodal_range", "counts", "paired_order"])
def test_relation_exports_reject_inconsistent_values_and_identity(whole_head_bundle, mutation):
    _, original, original_arrays = whole_head_bundle
    manifest = deepcopy(original)
    arrays = {key: value.copy() for key, value in original_arrays.items()}
    if mutation == "crossmodal_range":
        arrays["crossmodal_corr_eeg_hbt_zero"][0, 0] = 1.5
    elif mutation == "counts":
        arrays["corr_hbt_count"][0, 1] = -1
    else:
        manifest["relation_specs"]["partial_hbt"]["node_indices"].reverse()
    with pytest.raises(ValueError):
        render.validate_bundle(manifest, arrays)


def test_measured_minimal_head_does_not_invent_demo_relations(whole_head_bundle):
    _, original, original_arrays = whole_head_bundle
    manifest = deepcopy(original)
    manifest["kind"] = "measured_export"
    manifest["title"] = "Temporary non-measured fixture for measured-export rendering"
    manifest["units"].update(hb="fixture Hb unit", eeg="fixture EEG unit")
    manifest["relation_specs"] = {}
    manifest["head"].pop("reference")
    manifest["names"]["hb"] = manifest["names"]["hb"][:4]
    manifest["names"]["eeg"] = manifest["names"]["eeg"][:5]
    for field, count in (("hb_xy", 4), ("eeg_xy", 5), ("hb_pairs", 4)):
        manifest["geometry"][field] = manifest["geometry"][field][:count]
    manifest["geometry"]["neighbour_edges"] = []
    arrays = {key: value.copy() for key, value in original_arrays.items()
              if key in render.REQUIRED_ARRAYS}
    for field in ("hb_observation", "visible_hb", "score_hb"):
        arrays[field] = arrays[field][:, :4]
    for field in ("eeg_observation", "visible_eeg", "score_eeg"):
        arrays[field] = arrays[field][:, :5]
    render.validate_bundle(manifest, arrays)
    plt = render._plotting()
    for _, _, function, explanation in render._whole_specs(manifest, arrays):
        fig, metadata = function(plt, manifest, arrays)
        texts = " ".join(artist.get_text() for artist in fig.findobj()
                         if hasattr(artist, "get_text"))
        assert "64 路 EEG" not in texts
        assert "51 个 Hb" not in texts
        assert "合成相对 Hb 单位" not in texts
        assert "控制已知共同原因" not in explanation
        assert "known_direct" not in metadata.get("source_arrays", [])
        plt.close(fig)


def test_full_head_atlas_has_all_nine_maps_and_bitmap_text_contract(whole_head_bundle, tmp_path):
    path, _, _ = whole_head_bundle
    output = tmp_path / "full-head-export"
    render.render_bundle(path, output)
    assert len(list((output / "figures").glob("*.png"))) == 9
    with fitz.open(output / "FIGURE_ATLAS.pdf") as pdf:
        assert len(pdf) == 9
        for page in pdf:
            assert "合成" in page.get_text()
            assert len(page.get_text().strip()) > 30
            assert page.get_images(full=True)
            for info in page.get_image_info():
                dpi = info["width"] * 72 / fitz.Rect(info["bbox"]).width
                assert 199 <= dpi <= 301
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["page_count"] == 9
