"""Synthetic truth, paired interventions, split identity and input isolation."""
import itertools

import numpy as np
import pytest

from src.inference.semantic_response_synthetic import (
    ALIAS_SCENARIOS, BASIC_SCENARIOS, DEFAULT_SEED, DT, MECHANISM_SCENARIOS,
    OOD_SCENARIOS, PARTITION_SIZES, SCENARIOS, SOURCE_EPS, STEPS,
    generate_response_case, iter_response_cases, mechanism_case_index,
    observation_view, reference_operator, response_case_index, source_coordinates,
)
from src.inference.semantic_response_ssm import fixed_driver_response
from src.inference.shared_driver_modes import temporal_mode_basis
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def _case(scenario="baseline", family="restricted", **kwargs):
    return generate_response_case(DEFAULT_SEED, scenario, family, base_id=17, **kwargs)


def test_continuous_source_coordinates_have_explicit_reference_and_amplitude_gauge():
    time = np.arange(STEPS)
    source = .07 + .025*np.cos(time*.04)
    coordinates = source_coordinates(source)
    centered = source-np.mean(source[:20])
    amplitude = np.sqrt(np.mean(centered**2))
    np.testing.assert_array_equal(coordinates["a_absolute"], source)
    np.testing.assert_allclose(coordinates["source_shape"], centered/amplitude)
    assert abs(np.mean(coordinates["source_shape"][:20])) < 1e-14
    assert np.sqrt(np.mean(coordinates["source_shape"]**2)) == pytest.approx(1.)
    assert coordinates["log_amplitude"] == pytest.approx(np.log(amplitude))
    shifted = source_coordinates(source+4.)
    np.testing.assert_allclose(shifted["source_shape"], coordinates["source_shape"], atol=5e-14)
    zero = source_coordinates(np.zeros(STEPS))
    assert not zero["amplitude_supported"]
    assert zero["log_amplitude"] == np.log(SOURCE_EPS)
    np.testing.assert_array_equal(zero["source_shape"], 0.)


def test_generation_is_deterministic_and_has_continuous_truth_and_separate_inputs():
    case, repeated = _case(family="randomized"), _case(family="randomized")
    for role in ("control", "intervention"):
        record = case[role]
        assert record["eeg"].shape == (STEPS, 30)
        assert record["hb"].shape == (STEPS, 2)
        assert record["states"].shape == (STEPS, 6)
        assert record["a"].shape == record["b"].shape == record["source_shape"].shape == (STEPS,)
        assert record["source_mask"].all() and record["state_mask"].all()
        assert record["success"] and record["valid"] and record["support"]
        for key in ("eeg", "hb", "a", "b", "states", "source_shape", "loading"):
            np.testing.assert_array_equal(record[key], repeated[role][key])
        assert record["metadata"]["model_conditional"]
        assert not record["metadata"]["encoder_label_use"]
        view = observation_view(record)
        assert set(view) == {"eeg", "hb", "time_s", "eeg_mask", "hb_mask", "observation_mask"}
        assert all(name not in view for name in ("a", "u", "states", "metadata", "source_shape"))
        view["eeg"][0, 0] = 123.
        assert record["eeg"][0, 0] != 123.
        np.testing.assert_allclose(record["loading"].T@record["loading"], np.eye(2), atol=2e-15)


def test_partition_index_uses_base_id_before_interventions_and_no_source_overlap():
    all_ids = []
    for partition, size in PARTITION_SIZES.items():
        index = response_case_index(partition, scenarios=BASIC_SCENARIOS)
        assert len(index) == size*len(BASIC_SCENARIOS)
        assert len({item["base_id"] for item in index}) == size
        assert all(item["partition"] == partition for item in index)
        assert all(not any(isinstance(value, np.ndarray) for value in item.values()) for item in index)
        small = list(iter_response_cases(partition, scenarios=("baseline", "feature_gain_only"), indices=[0]))
        assert small[0]["metadata"]["base_id"] == small[1]["metadata"]["base_id"]
        np.testing.assert_array_equal(small[0]["control"]["eeg"], small[1]["control"]["eeg"])
        all_ids.append(small[0]["metadata"]["base_id"])
    assert len(set(all_ids)) == len(PARTITION_SIZES)
    train = next(iter_response_cases("train", indices=[0]))
    evaluation = next(iter_response_cases("evaluation", indices=[0]))
    assert not np.array_equal(train["control"]["a"], evaluation["control"]["a"])
    assert not np.array_equal(train["control"]["eeg_noise"], evaluation["control"]["eeg_noise"])


def test_restricted_randomized_families_preserve_base_sources_for_ablation():
    restricted, randomized = _case(), _case(family="randomized")
    for key in ("a", "b", "source_shape", "log_amplitude"):
        np.testing.assert_array_equal(restricted["control"][key], randomized["control"][key])
    assert restricted["metadata"]["base_id"] == randomized["metadata"]["base_id"]
    assert randomized["control"]["metadata"]["feature_coordinate_gain"] != 1.
    assert randomized["control"]["metadata"]["component_amplitude"] > 0.
    assert not np.array_equal(restricted["control"]["hb"], randomized["control"]["hb"])


@pytest.mark.parametrize("scenario", ("feature_gain_only", "Hb_gain_only", "common_Hb",
    "tau_change", "initial_change", "input_lowpass", "viscoelastic_outflow",
    "independent_Hb_drive", "no_coupling", "rotated_spectral_mixture", "alternative_hrf"))
def test_nuisance_and_mechanism_interventions_keep_stipulated_source_target(scenario):
    case = _case(scenario)
    control, intervention = case["control"], case["intervention"]
    for key in ("a", "b", "source_shape", "log_amplitude", "time_s"):
        np.testing.assert_array_equal(control[key], intervention[key])
    assert case["metadata"]["causal_interventions"]
    assert control["success"] and intervention["success"]
    if scenario != "rotated_spectral_mixture" and scenario != "feature_gain_only":
        np.testing.assert_array_equal(control["eeg"], intervention["eeg"])
    if scenario == "feature_gain_only":
        np.testing.assert_array_equal(control["states"], intervention["states"])
        np.testing.assert_array_equal(control["hb"], intervention["hb"])
        assert not np.array_equal(control["eeg"], intervention["eeg"])
    if scenario in ("Hb_gain_only", "common_Hb"):
        np.testing.assert_array_equal(control["states"], intervention["states"])
        np.testing.assert_array_equal(control["physical_hb"], intervention["physical_hb"])


def test_amplitude_shape_changes_are_not_observation_gain_changes():
    amplitude = _case("true_a_amplitude")
    control, intervention = amplitude["control"], amplitude["intervention"]
    np.testing.assert_allclose(intervention["source_shape"], control["source_shape"], atol=1e-14)
    assert intervention["log_amplitude"]-control["log_amplitude"] == pytest.approx(np.log(2.))
    assert intervention["metadata"]["feature_coordinate_gain"] == control["metadata"]["feature_coordinate_gain"]
    assert not np.array_equal(intervention["hb"], control["hb"])
    shape = _case("true_a_shape")
    assert shape["intervention"]["log_amplitude"] == pytest.approx(shape["control"]["log_amplitude"])
    assert not np.allclose(shape["intervention"]["source_shape"], shape["control"]["source_shape"])


def test_declared_sixteen_second_duration_preserves_reference_relative_amplitude():
    record = _case(duration=16., amplitude=.025)["control"]
    assert record["metadata"]["duration_s"] == 16.
    assert record["amplitude"] == pytest.approx(.025)
    assert record["physical_hb"].shape == (120, 2)
    assert record["success"]


@pytest.mark.parametrize("scenario", ALIAS_SCENARIOS)
def test_exact_aliases_have_identical_observations_and_different_true_amplitude(scenario):
    case = _case(scenario, family="randomized")
    control, intervention = case["control"], case["intervention"]
    np.testing.assert_array_equal(intervention["eeg"], control["eeg"])
    np.testing.assert_array_equal(intervention["b"], control["b"])
    np.testing.assert_array_equal(intervention["eeg_noise"], control["eeg_noise"])
    np.testing.assert_allclose(intervention["source_shape"], control["source_shape"], atol=1e-14)
    assert intervention["log_amplitude"]-control["log_amplitude"] == pytest.approx(np.log(2.))
    assert not np.array_equal(intervention["physical_hb"], control["physical_hb"])
    gain = intervention["metadata"]["feature_coordinate_gain"]
    np.testing.assert_allclose(gain*intervention["a"],
        control["metadata"]["feature_coordinate_gain"]*control["a"], atol=0., rtol=0.)
    if scenario == "exact_joint_alias":
        np.testing.assert_array_equal(intervention["hb"], control["hb"])
        assert intervention["metadata"]["constructed_ambiguity"]
        assert intervention["metadata"]["observational_alias_modalities"] == ["eeg", "hb"]
        np.testing.assert_allclose(intervention["hb"],
            intervention["metadata"]["hb_coordinate_gain"]*intervention["physical_hb"]
            + intervention["observation_component"]+intervention["hb_noise"], atol=2e-18)
    else:
        assert not np.array_equal(intervention["hb"], control["hb"])
        assert not intervention["metadata"]["constructed_ambiguity"]


def test_physical_prefix_is_generated_before_observation_reference_subtraction():
    identity = _case("initial_change", hb_operator=np.eye(STEPS))
    referenced = _case("initial_change")
    for role in ("control", "intervention"):
        physical, processed = identity[role], referenced[role]
        np.testing.assert_array_equal(physical["states"], processed["states"])
        np.testing.assert_array_equal(physical["canonical_hb"], processed["canonical_hb"])
        np.testing.assert_allclose(processed["physical_hb"],
            reference_operator()@physical["canonical_hb"], atol=0., rtol=0.)
        np.testing.assert_allclose(processed["physical_hb"][:20].mean(axis=0), 0., atol=1e-16)
    assert not np.array_equal(identity["intervention"]["states"][0], identity["control"]["states"][0])
    assert np.linalg.norm(referenced["intervention"]["canonical_hb"][:20]) > 0.


def test_standard_case_agrees_with_physical_owner_and_distinguishes_driver_from_lagged_input():
    control = _case()["control"]
    expected = fixed_driver_response(control["a"], BalloonParameters(), DT,
        hb_operator=reference_operator())
    np.testing.assert_array_equal(control["states"], expected["states"])
    np.testing.assert_array_equal(control["physical_hb"], expected["hb_prediction"])
    np.testing.assert_array_equal(control["u"], control["a"])
    delayed = _case("input_lowpass")["intervention"]
    assert delayed["metadata"]["tau_n"] == 1.5
    assert delayed["u"][0] == 0.
    assert not np.allclose(delayed["u"], delayed["a"])
    viscous = _case("viscoelastic_outflow")["intervention"]
    assert viscous["metadata"]["tau_v"] == 2.


def test_common_Hb_component_uses_variable_direction_and_declared_correlation():
    directions = []
    for base_id in (1, 2, 3):
        record = generate_response_case(DEFAULT_SEED, "correlated_Hb_component", base_id=base_id)["intervention"]
        direction = np.asarray(record["metadata"]["component_direction"])
        process = record["observation_component"]@direction
        process /= np.sqrt(np.mean(process**2))
        assert np.mean(process*record["source_shape"]) == pytest.approx(.85, abs=1e-14)
        assert np.linalg.norm(direction) == pytest.approx(1.)
        directions.append(direction)
    assert not np.allclose(directions, np.broadcast_to(directions[0], (3, 2)))
    assert not np.allclose(directions, np.tile([1/np.sqrt(2), 1/np.sqrt(2)], (3, 1)))


def test_outside_generator_drives_have_nonDCT_energy_and_independent_forward_no_fake_states():
    basis = temporal_mode_basis(STEPS, 24)
    base = _case()["control"]["source_shape"]
    np.testing.assert_allclose(basis@(basis.T@base/STEPS), base, atol=5e-15)
    for scenario in ("non_dct_impulse", "ou_drive"):
        source = _case(scenario)["intervention"]["source_shape"]
        assert np.sqrt(np.mean((source-basis@(basis.T@source/STEPS))**2)) > .1
    alternative = _case("alternative_hrf")["intervention"]
    assert alternative["metadata"]["truth_family"] == "illustrative_double_gamma"
    assert not alternative["metadata"]["state_truth_available"]
    assert not alternative["state_mask"].any()
    assert np.isnan(alternative["states"]).all()
    assert alternative["eeg_mask"].all() and alternative["hb_mask"].all()


def test_zero_and_independent_Hb_coupling_do_not_erase_true_source_targets():
    for scenario in ("no_coupling", "independent_Hb_drive"):
        record = _case(scenario)["intervention"]
        assert record["amplitude"] > SOURCE_EPS
        assert record["source_mask"].all()
        assert record["metadata"]["source_identifiability"] == "a_not_identified_by_Hb_under_true_route"
        assert not np.array_equal(record["driver"], record["a"])
        if scenario == "no_coupling":
            np.testing.assert_array_equal(record["driver"], 0.)
    coupled = _case("coupling_change")["intervention"]
    assert coupled["metadata"]["truth_gamma"] == .7
    np.testing.assert_array_equal(coupled["driver"], coupled["a"]+.7*coupled["b"])


def test_mechanism_index_is_balanced_bounded_and_distinct_across_training_evaluation():
    index = mechanism_case_index()
    assert len(index) == 8*3*len(MECHANISM_SCENARIOS)
    assert {spec["amplitude"] for spec in index} == {.0125, .025, .05}
    assert len({spec["base_id"] for spec in index}) == 24
    assert all(spec["partition"] == "evaluation" for spec in index)
    assert all(3. <= spec["duration"] <= 7. for spec in index)
    train_spec, evaluation_spec = mechanism_case_index(partition="train")[0], index[0]
    train, evaluation = generate_response_case(**train_spec), generate_response_case(**evaluation_spec)
    assert train["metadata"]["base_id"] != evaluation["metadata"]["base_id"]
    assert not np.array_equal(train["control"]["a"], evaluation["control"]["a"])
    assert tuple(itertools.islice(iter_response_cases("evaluation", indices=[]), 1)) == ()


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(seed=True), dict(partition="test"),
    dict(family="unknown"), dict(scenario="unknown"), dict(base_id=999),
    dict(amplitude=0.), dict(duration=.1), dict(hb_operator=np.eye(3)),
    dict(overrides={"allow_measured": True}), dict(overrides={"tau_n": -1.}),
    dict(overrides={"component_correlation": 1.}), dict(overrides={"initial_state": [0, 0, 1, 1, 1]})])
def test_invalid_contract_is_rejected(kwargs):
    arguments = dict(seed=DEFAULT_SEED)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        generate_response_case(**arguments)


def test_index_rejects_unknown_duplicate_and_outside_partition_cases():
    for kwargs in (dict(indices=[0, 0]), dict(indices=[512]), dict(indices=[True]),
                   dict(scenarios=[]), dict(scenarios=["baseline", "baseline"]),
                   dict(scenarios=["nonsense"])):
        with pytest.raises(ValueError):
            response_case_index("train", **kwargs)
    assert set(BASIC_SCENARIOS + OOD_SCENARIOS + ALIAS_SCENARIOS) <= set(SCENARIOS)
