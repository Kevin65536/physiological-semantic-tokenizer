"""Known-truth/metamorphic checks for the synthetic visual guide; no data reads."""
import numpy as np
import pytest

from experiments.scripts import illustrate_model_compatibility as guide


@pytest.fixture(scope="module")
def examples():
    return guide.prepare_examples()


def test_spatial_lower_bound_distinguishes_rotation_from_added_rank(examples):
    legal, rotated, extra = examples["spatial"]
    assert legal["D"] < 1e-25
    assert rotated["D"] == pytest.approx(np.sin(np.pi/6)**2)
    assert rotated["rank_D"] < 1e-25
    assert extra["D"] == pytest.approx(.25/1.25)
    assert extra["rank_D"] == pytest.approx(extra["D"])
    for y, score in zip(examples["spatial_inputs"], examples["spatial"]):
        assert np.sum(y*y) == pytest.approx(np.sum(score["fitted"]**2)+np.sum(score["residual"]**2))
        assert np.max(abs(score["residual"] @ examples["mixing"])) < 1e-14


def test_linear_closures_match_analytic_dynamics_and_scale_invariance(examples):
    derivatives = examples["derivatives"]
    total = derivatives[:, :, 1] + derivatives[:, :, 2]
    r = derivatives[:, :, 2]
    h = 2*r[1] + r[0] - guide.RHO*(2*guide.XI*total[1] + guide.ZETA*total[0])
    assert max(abs(h)) < 1e-14
    x = guide.RHO*derivatives[:, :, 1]-(1-guide.RHO)*derivatives[:, :, 2]
    left = 2*x[3] + (1+.64*2)*x[2] + (.64+.32*2)*x[1]+.32*x[0]
    assert max(abs(left-guide.PARAMETERS.fixed.Q0*guide.B*derivatives[0, :, 0])) < 1e-14
    assert examples["hb_good_result"]["D"] < 1e-12
    assert examples["cross_good"]["D"] < 1e-12
    scaled = examples["hb_bad"].copy()
    scaled[:, 1:] *= 3.7
    result = guide.hb_closure(scaled, examples["t"], examples["bank"])
    assert result["D"] == pytest.approx(examples["hb_bad_result"]["D"], rel=1e-10)


def test_legal_same_phase_relaxation_needs_initial_transient(examples):
    y = examples["relaxation"]
    assert np.corrcoef(y[:, 1], y[:, 2])[0, 1] == pytest.approx(1.)
    assert examples["relaxation_good"]["D"] < 1e-12
    assert examples["relaxation_wrong"]["D"] > 1
    expected_c0 = .04*guide.RHO*guide.B*(1/guide.PARAMETERS.fixed.alpha-1)
    assert examples["relaxation_good"]["c0"] == pytest.approx(expected_c0, rel=1e-8)


def test_shifted_eeg_only_changes_cross_modal_check(examples):
    shifted_hb = guide.hb_closure(examples["eeg_bad"], examples["t"], examples["bank"])
    assert shifted_hb["D"] == examples["hb_good_result"]["D"]
    assert examples["cross_bad"]["D"] > .5
    assert examples["cross_bad"]["gain"] > 0
    assert examples["hb_bad_result"]["D"] > .2


def test_feasible_sets_recover_known_intersection_and_expose_weak_excitation(examples):
    interior, conflict, edge, weak = examples["profiles"]
    tau = examples["taus"]
    assert interior["accepted"][np.argmin(abs(tau-2))]
    assert not conflict["accepted"].any()
    assert np.min(conflict["hb"]) < 1e-12
    assert np.min(conflict["cross"]) < 1e-12
    assert edge["accepted"][np.argmin(abs(tau-7.9))]
    assert edge["maximum_inward_fraction"] <= .1+1e-12
    assert weak["accepted"].all()


def test_nonlinear_hb_only_reconstruction_recovers_hidden_flow(examples):
    reconstructed = examples["nl_best"][1]
    # Hidden states are only a test oracle; the reconstruction takes O and R.
    truth = examples["nonlinear"][1]["states"]
    assert np.max(abs(reconstructed["flow"]-truth[:, 2])) < 2e-7
    assert np.max(abs(reconstructed["volume"]-truth[:, 3])) < 1e-10
    assert reconstructed["D"] < 1e-12
    assert examples["nl_best"][2]["D"] > .4
    assert examples["nl_linear"][1] > 100*examples["nl_linear"][0]
    assert examples["nonlinear_refinement_error"] < 1e-9
    assert all(np.isfinite(scores).all() for scores in examples["nl_profiles"])


def test_invalid_absolute_coordinates_and_zero_information_are_not_passes(examples):
    bad = examples["y"].copy()
    bad[:, 2] = -2.
    assert not guide.nonlinear_closure(examples["t"], bad, 0.)["valid"]
    zero = np.zeros_like(bad)
    assert guide.hb_closure(zero, examples["t"], examples["bank"])["D"] is None
    assert guide.cross_closure(zero, examples["bank"])["D"] is None


def test_known_white_noise_residual_has_expected_degrees_of_freedom(examples):
    ratio = np.mean(examples["null_q"])/examples["noise_degrees"]
    assert ratio == pytest.approx(1., abs=.01)
    assert examples["noisy_added_Q"]/examples["noise_degrees"] > 10
