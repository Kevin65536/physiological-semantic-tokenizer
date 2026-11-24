"""Atomic virtual-carrier IDs must preserve layout and exclude continuous decode inputs."""

import numpy as np
import pytest

from src.tokenizers.anchored_carrier_dictionary import (
    AnchoredCarrierDictionary,
    assignment_features,
    feature_coordinate,
)


class Source:
    metadata = {"positions": {"fixture": list("abcde")}}

    @staticmethod
    def roles(dataset):
        return np.arange(5)

    @staticmethod
    def continuous_source(values):
        return {"S": values * 0.5, "O": values * 0.5, "sum": values}

    @staticmethod
    def analyze(prediction, dataset, observed):
        return {b: prediction[b].copy() for b in ("S", "O")}

    @staticmethod
    def synthesize(carriers, dataset):
        return dict(carriers, sum=carriers["S"] + carriers["O"])


def fixture():
    time = np.arange(400) / 200
    waves = np.stack(
        [np.sin(2 * np.pi * f * time) for f in [2.0, 6.0, 10.0, 20.0, 35.0]]
    )
    books = {}
    for b in ("S", "O"):
        for band in range(5):
            atom = waves[band : band + 1]
            feature = assignment_features(atom, atom, band)
            books[b, band] = dict(
                mean=np.zeros(feature.shape[1]),
                weight=np.ones(feature.shape[1]),
                centers=feature,
                atoms=np.vstack((np.zeros(400), atom)),
                counts=np.array([0, 10]),
                iterations=1,
            )
    meta = dict(schema="anchored_carrier_dictionary_v1", codebook_size=2)
    return AnchoredCarrierDictionary(Source(), books, meta), waves


def test_integer_decode_excludes_source_analysis_and_preserves_missing_slot():
    model, waves = fixture()
    values = np.broadcast_to((2 * waves)[None, :, None], (1, 5, 15, 400))
    observed = np.array([[True, True, False, True, True]])
    ids = model.encode(values, "fixture", observed)
    for branch in ("S", "O"):
        np.testing.assert_array_equal(ids[branch][0, 2], 0)
    expected = model.decode_ids(ids, "fixture")
    model.source.continuous_source = lambda *_: (_ for _ in ()).throw(
        AssertionError("continuous path used")
    )
    model.source.analyze = lambda *_: (_ for _ in ()).throw(
        AssertionError("analysis path used")
    )
    repeated = model.decode_ids(ids, "fixture")
    for branch in ("S", "O", "sum"):
        np.testing.assert_array_equal(repeated[branch], expected[branch])
        np.testing.assert_array_equal(repeated[branch][0, 2], 0)
    np.testing.assert_array_equal(repeated["sum"], repeated["S"] + repeated["O"])
    with pytest.raises(ValueError, match="integer"):
        model.decode_ids({k: v.astype(float) for k, v in ids.items()}, "fixture")


def test_physical_attributes_are_part_of_joint_assignment_coordinate():
    carrier = np.zeros((2, 400))
    physical = carrier.copy()
    physical[1] = np.sin(2 * np.pi * 10 * np.arange(400) / 200)
    features = assignment_features(carrier, physical, 0)
    np.testing.assert_array_equal(features[0, :-3], features[1, :-3])
    assert features[1, -2] > features[0, -2]
    _, phase = feature_coordinate(features, 0.0)
    _, joint = feature_coordinate(features, 0.5)
    np.testing.assert_array_equal(phase[-3:], 0)
    assert np.all(joint[-3:] > 0)


def test_saved_books_reload_with_frozen_source_and_reject_source_mutation(tmp_path):
    from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier
    from src.tokenizers.anchored_carrier_dictionary import load_model
    from src.tokenizers.spatial_waveform_dictionary import BANDS

    model, waves = fixture()
    analysis, synthesis = {}, {}
    for branch in ("S", "O"):
        for band in range(5):
            analysis["fixture", branch, band] = np.eye(5)[:, band : band + 1]
            synthesis["fixture", branch, band] = np.eye(5)[band : band + 1]
    metadata = dict(
        schema="anchored_spatial_carrier_v1",
        positions={"fixture": list("abcde")},
        bands_hz=[list(v) for v in BANDS],
    )
    source = AnchoredSpatialCarrier(
        metadata, {b: np.full(3001, 0.5) for b in "SO"}, analysis, synthesis
    )
    model.source = source
    path, source_path = tmp_path / "model.npz", tmp_path / "source.npz"
    source.save(source_path)
    model.save(path, source_path)
    ids = {b: np.ones((1, 5, 15), np.int32) for b in "SO"}
    expected = model.decode_ids(ids, "fixture")
    restored = load_model(path)
    restored.source.continuous_source = lambda *_: (_ for _ in ()).throw(
        AssertionError("X used by decoder")
    )
    restored.source.analyze = lambda *_: (_ for _ in ()).throw(
        AssertionError("continuous carrier used by decoder")
    )
    for branch, value in restored.decode_ids(ids, "fixture").items():
        np.testing.assert_array_equal(value, expected[branch])
    source_path.write_bytes(b"changed frozen source")
    with pytest.raises(ValueError, match="identity"):
        load_model(path)
