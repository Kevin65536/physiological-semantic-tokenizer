"""Paired, synthetic continuous-source observations for semantic-response research.

This is a model-conditional identification benchmark, not a physiological data
generator validated against humans. ``a`` is the stipulated common neural drive;
``b`` is a fixed alpha/beta EEG feature contrast and normally has no vascular
effect. EEG values are effective baseline-relative log-power coordinates, not
raw voltages. An unknown ``feature_coordinate_gain`` multiplies the common
log-power readout; it must not be interpreted as raw EEG voltage gain.

Each base identity has one deterministic partition and random stream. All of
its interventions, including deliberately observationally equivalent pairs,
remain in that partition. Source trajectories exist at all 120 time points
before any tokenizer pooling. Use :func:`observation_view` to construct encoder
inputs: source values, mechanisms, gains and ambiguity labels are truth only.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from collections.abc import Iterable, Iterator, Mapping

import numpy as np
from scipy.special import gamma as gamma_function

from .semantic_response_ssm import fixed_driver_response
from .shared_driver_modes import spectral_loadings, temporal_mode_basis
from .t3a_balloon_robust_ssm import BalloonParameters


SCHEMA = "semantic_response_synthetic_v1"
DEFAULT_SEED = 20261002
STEPS = 120
DT = .25
REFERENCE_STEPS = 20
SOURCE_EPS = 1e-12
PARTITION_SIZES = {"train": 512, "selection": 128, "evaluation": 256}
PARTITION_CODES = {"train": 1, "selection": 2, "evaluation": 3}
BASIC_SCENARIOS = (
    "baseline", "true_a_amplitude", "true_a_shape", "feature_gain_only",
    "Hb_gain_only", "common_Hb", "tau_change", "initial_change",
)
OOD_SCENARIOS = (
    "non_dct_impulse", "ou_drive", "time_varying_feature_gain",
    "rotated_spectral_mixture", "correlated_Hb_component", "input_lowpass",
    "viscoelastic_outflow", "independent_Hb_drive", "no_coupling",
    "coupling_change", "alternative_hrf",
)
ALIAS_SCENARIOS = ("exact_eeg_alias", "exact_joint_alias")
MECHANISM_SCENARIOS = (
    "standard", "nonrest", "input_lowpass", "viscoelastic_outflow",
    "common_Hb", "no_coupling",
)
SCENARIOS = BASIC_SCENARIOS + OOD_SCENARIOS + ALIAS_SCENARIOS + ("standard", "nonrest")
_OVERRIDE_NAMES = frozenset((
    "feature_coordinate_gain", "contrast_coordinate_gain", "hb_coordinate_gain",
    "tau", "kappa", "tau_n", "tau_v", "initial_state", "noise_eeg_sd",
    "noise_hb_sd", "component_amplitude", "component_correlation", "alias_scale",
))


def reference_operator(steps: int = STEPS, reference_steps: int = REFERENCE_STEPS) -> np.ndarray:
    """Linear reference subtraction, preserving the physical prefix in truth."""
    if (isinstance(steps, bool) or not isinstance(steps, (int, np.integer))
            or isinstance(reference_steps, bool)
            or not isinstance(reference_steps, (int, np.integer))
            or not 1 <= reference_steps <= steps):
        raise ValueError("integer steps and 1 <= reference_steps <= steps required")
    result = np.eye(steps)
    result[:, :reference_steps] -= 1. / reference_steps
    return result


def source_coordinates(a: np.ndarray, *, reference_steps: int = REFERENCE_STEPS,
                       eps: float = SOURCE_EPS) -> dict:
    """Reference-centered RMS shape and natural log reference-relative amplitude.

    ``a_absolute`` is the supplied physical drive. Amplitude means RMS of
    ``a - mean(a[:reference_steps])`` over the entire continuous window, rather
    than RMS of the absolute drive or a token-pooled sequence. The denominator
    and log use ``max(rms, eps)``; zero sources stay zero and carry explicit
    ``amplitude_supported=False``.
    """
    values = np.asarray(a, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or isinstance(reference_steps, bool)
            or not isinstance(reference_steps, (int, np.integer))
            or not 1 <= reference_steps <= len(values)
            or not np.isfinite(eps) or eps <= 0):
        raise ValueError("finite source [T], valid reference_steps and positive eps required")
    reference = float(np.mean(values[:reference_steps]))
    centered = values - reference
    amplitude = float(np.sqrt(np.mean(centered**2)))
    scale = max(amplitude, eps)
    return dict(a_absolute=values.copy(), a_reference_centered=centered,
                source_shape=centered / scale, amplitude=amplitude,
                log_amplitude=float(np.log(scale)), reference_value=reference,
                amplitude_supported=amplitude > eps, source_eps=float(eps))


def _rng(seed: int, partition: str, base_id: str, stream: int) -> np.random.Generator:
    identity = hashlib.blake2s(base_id.encode("utf-8"), digest_size=8).digest()
    words = np.frombuffer(identity, dtype="<u4").astype(np.uint32).tolist()
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), PARTITION_CODES[partition], *words, int(stream)]))


def _unit_shape(values: np.ndarray) -> np.ndarray:
    coordinates = source_coordinates(values)
    if not coordinates["amplitude_supported"]:
        raise ValueError("shape must have nonzero reference-relative amplitude")
    return coordinates["source_shape"]


def _dct_shape(rng: np.random.Generator, duration: float) -> np.ndarray:
    """Slow continuous pulse family represented exactly by twelve DCT columns."""
    time = np.arange(STEPS) * DT
    onset = rng.uniform(7., 10.)
    pulse = np.exp(-.5*((time-onset)/duration)**2)
    pulse -= rng.uniform(.1, .35)*np.exp(-.5*((time-onset-2*duration)/(1.3*duration))**2)
    basis = temporal_mode_basis(STEPS, 12)
    coefficients = basis.T @ pulse / STEPS
    coefficients[1:5] += rng.normal(0., .025, 4)
    return _unit_shape(basis @ coefficients)


def _colored_process(rng: np.random.Generator, phi: float = .96) -> np.ndarray:
    values = np.empty(STEPS)
    values[0] = rng.normal()
    innovations = rng.normal(size=STEPS-1)
    for t, innovation in enumerate(innovations, start=1):
        values[t] = phi*values[t-1] + np.sqrt(1-phi**2)*innovation
    return _unit_shape(values)


def _component_process(independent: np.ndarray, a: np.ndarray, correlation: float) -> np.ndarray:
    """Declared correlation is achieved in the reference-centered RMS metric."""
    source = _unit_shape(a)
    independent = independent - source*np.mean(independent*source)
    independent = _unit_shape(independent)
    return correlation*source + np.sqrt(1-correlation**2)*independent


def _alternative_response(driver: np.ndarray, dt: float) -> np.ndarray:
    """Independent illustrative double-gamma response, outside the Balloon family."""
    time = np.arange(STEPS)*dt
    def kernel(shape, scale):
        return time**(shape-1)*np.exp(-time/scale)/(gamma_function(shape)*scale**shape)
    hbo = kernel(6., .8)-.22*kernel(12., 1.)
    hbr = -.3*kernel(7., .8)+.06*kernel(14., 1.)
    return np.column_stack((np.convolve(driver, hbo)[:STEPS],
                            np.convolve(driver, hbr)[:STEPS]))*dt


def _validate_positive(name, value, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite {'nonnegative' if allow_zero else 'positive'}")


def _record(a, b, settings, *, parameters, hb_operator, loading, noise_eeg,
            noise_hb, component_process, component_direction, identity, scenario,
            family, partition, seed, duration, driver_override=None,
            alternative=False, feature_gain_trajectory=None):
    p = replace(parameters, free=replace(parameters.free,
        tau=float(settings["tau"]), kappa=float(settings["kappa"])))
    initial = np.asarray(settings["initial_state"], dtype=float)
    tau_n, tau_v = float(settings["tau_n"]), float(settings["tau_v"])
    truth_gamma = float(settings.get("truth_gamma", 0.))
    driver = a + truth_gamma*b if driver_override is None else np.asarray(driver_override, float)
    if alternative:
        canonical_hb = _alternative_response(driver, DT)
        physical_hb = hb_operator @ canonical_hb
        states = np.full((STEPS, 6), np.nan)
        u, outflow = driver.copy(), np.full(STEPS, np.nan)
        physical_check = {"valid": True, "applicable": False,
                          "reason": "illustrative_double_gamma_has_no_Balloon_states"}
    else:
        forward = fixed_driver_response(driver, p, DT, initial_state=initial,
            tau_n=tau_n, tau_v=tau_v, hb_operator=hb_operator, derivative=False)
        canonical_hb, physical_hb = forward["canonical_hb"], forward["hb_prediction"]
        states, u, outflow = forward["states"], forward["vascular_input"], forward["outflow"]
        physical_check = forward["physical_check"]
    common = _component_process(component_process, a, settings["component_correlation"])
    # This component is already defined in processed Hb coordinates. Applying
    # hb_operator again would change its declared generative meaning.
    component = (settings["component_amplitude"] * common[:, None]
                 * component_direction[None, :])
    gain = (np.full(STEPS, settings["feature_coordinate_gain"])
            if feature_gain_trajectory is None else feature_gain_trajectory)
    canonical_eeg = ((gain*a)[:, None]*loading[:, 0]
                     + (settings["contrast_coordinate_gain"]*b)[:, None]*loading[:, 1])
    eeg_noise = reference_operator() @ (noise_eeg*settings["noise_eeg_sd"])
    hb_noise = hb_operator @ (noise_hb*settings["noise_hb_sd"])
    eeg = reference_operator() @ canonical_eeg + eeg_noise
    hb = settings["hb_coordinate_gain"]*physical_hb + component + hb_noise
    coordinates = source_coordinates(a)
    metadata = dict(schema=SCHEMA, base_id=identity, partition=partition,
        seed=int(seed), scenario=scenario, family=family,
        truth_family="illustrative_double_gamma" if alternative else "Balloon_inlet_balance",
        model_conditional=True, source_interpretation="stipulated_common_neural_drive",
        truth_gamma=truth_gamma, tau_n=tau_n, tau_v=tau_v, parameters=asdict(p),
        initial_state=initial.tolist(), feature_coordinate_gain=float(settings["feature_coordinate_gain"]),
        feature_coordinate_gain_definition="common_effective_log_power_coordinate_multiplier",
        contrast_coordinate_gain=float(settings["contrast_coordinate_gain"]),
        hb_coordinate_gain=float(settings["hb_coordinate_gain"]),
        component_amplitude=float(settings["component_amplitude"]),
        component_correlation=float(settings["component_correlation"]),
        component_direction=component_direction.tolist(), component_coordinate="processed_Hb",
        noise_eeg_sd=float(settings["noise_eeg_sd"]), noise_hb_sd=float(settings["noise_hb_sd"]),
        duration_s=float(duration), dt_s=DT, reference_steps=REFERENCE_STEPS,
        source_eps=SOURCE_EPS, source_amplitude_definition="RMS_after_first20_reference_subtraction",
        amplitude_supported=coordinates["amplitude_supported"],
        state_truth_available=not alternative, true_hb_driver_route=settings.get("driver_route", "a"),
        source_identifiability="model_and_calibration_conditional",
        encoder_label_use=False, constructed_ambiguity=False,
        observational_alias_modalities=[], physical_check=physical_check)
    eeg_mask, hb_mask = np.isfinite(eeg), np.isfinite(hb)
    valid = bool(eeg_mask.all() and hb_mask.all() and physical_check["valid"])
    return dict(eeg=eeg, hb=hb, a=coordinates["a_absolute"], b=b.copy(),
        a_reference_centered=coordinates["a_reference_centered"],
        source_shape=coordinates["source_shape"], amplitude=coordinates["amplitude"],
        log_amplitude=coordinates["log_amplitude"], source_reference=coordinates["reference_value"],
        time_s=np.arange(STEPS)*DT, states=states, u=u, driver=driver.copy(), outflow=outflow,
        canonical_eeg=canonical_eeg, canonical_hb=canonical_hb, physical_hb=physical_hb,
        observation_component=component, eeg_noise=eeg_noise, hb_noise=hb_noise,
        loading=loading.copy(), feature_gain_trajectory=gain.copy(),
        eeg_mask=eeg_mask, hb_mask=hb_mask,
        observation_mask=np.column_stack((eeg_mask, hb_mask)),
        source_mask=np.isfinite(coordinates["source_shape"]), state_mask=np.isfinite(states),
        success=valid, valid=valid, support=valid, metadata=metadata)


def generate_response_case(seed: int, scenario: str = "baseline", family: str = "randomized", *,
        partition: str = "evaluation", base_id: int | str | None = None,
        parameters: BalloonParameters | None = None, amplitude: float | None = None,
        duration: float | None = None, hb_operator: np.ndarray | None = None,
        overrides: Mapping | None = None) -> dict:
    """Generate one control/intervention pair without reading measured data.

    The control is identical across scenarios for a given seed/base/family and
    partition. Restricted/randomized families share ``a`` and ``b``; only
    nuisance settings differ. ``overrides`` changes control nuisance settings
    before the intervention and is recorded in truth. It cannot add features.
    Exact alias cases reuse observed arrays after the explicitly declared
    algebraic intervention to guarantee bitwise equality, not approximate
    matching obscured by floating-point evaluation order.
    """
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
            or partition not in PARTITION_SIZES or family not in ("restricted", "randomized")
            or scenario not in SCENARIOS):
        raise ValueError("valid nonnegative integer seed, partition, family and scenario required")
    if isinstance(base_id, bool) or (base_id is not None and not isinstance(base_id, (int, str, np.integer))):
        raise ValueError("base_id must be an integer or string")
    if isinstance(base_id, (int, np.integer)) and not 0 <= int(base_id) < PARTITION_SIZES[partition]:
        raise ValueError("integer base_id exceeds its fixed partition size")
    identity = f"{SCHEMA}/{partition}/{str(seed) if base_id is None else str(base_id)}"
    parameters = BalloonParameters() if parameters is None else parameters
    parameters.validate()
    operator = reference_operator() if hb_operator is None else np.asarray(hb_operator, float)
    if operator.shape != (STEPS, STEPS) or not np.isfinite(operator).all():
        raise ValueError("hb_operator must be finite [120,120]")
    source_rng = _rng(seed, partition, identity, 1)
    nuisance_rng = _rng(seed, partition, identity, 2)
    duration = float(source_rng.uniform(3., 7.) if duration is None else duration)
    amplitude = float(np.exp(source_rng.uniform(np.log(.0125), np.log(.05)))
                      if amplitude is None else amplitude)
    _validate_positive("duration", duration)
    _validate_positive("amplitude", amplitude)
    if not .002 <= amplitude <= .15 or not 1. <= duration <= 20.:
        raise ValueError("amplitude [.002,.15] and duration [1,20] bound this benchmark")
    a = amplitude*_dct_shape(source_rng, duration)
    b = float(source_rng.uniform(.0125, .05))*_dct_shape(source_rng, float(source_rng.uniform(2., 6.)))
    settings = dict(feature_coordinate_gain=1., contrast_coordinate_gain=1., hb_coordinate_gain=1.,
        tau=parameters.free.tau, kappa=parameters.free.kappa, tau_n=0., tau_v=0.,
        initial_state=np.r_[0., np.ones(4)], noise_eeg_sd=.0015, noise_hb_sd=.0006,
        component_amplitude=0., component_correlation=0., truth_gamma=0., driver_route="a")
    if family == "randomized":
        settings.update(feature_coordinate_gain=float(np.exp(nuisance_rng.uniform(np.log(.65), np.log(1.55)))),
            contrast_coordinate_gain=float(nuisance_rng.uniform(.7, 1.4)),
            hb_coordinate_gain=float(nuisance_rng.uniform(.7, 1.4)),
            tau=float(nuisance_rng.uniform(1.4, 3.2)), kappa=float(nuisance_rng.uniform(.45, .9)),
            initial_state=np.r_[nuisance_rng.normal(0., .004), np.exp(nuisance_rng.normal(0., .01, 4))],
            noise_eeg_sd=float(nuisance_rng.uniform(.001, .003)),
            noise_hb_sd=float(nuisance_rng.uniform(.0004, .0012)),
            component_amplitude=float(nuisance_rng.uniform(0., .003)),
            component_correlation=float(nuisance_rng.choice([-.25, 0., .25])))
    if overrides is not None:
        unknown = set(overrides)-_OVERRIDE_NAMES
        if unknown:
            raise ValueError(f"unknown overrides: {sorted(unknown)}")
        settings.update({key: value for key, value in overrides.items() if key != "alias_scale"})
    for name in ("feature_coordinate_gain", "contrast_coordinate_gain", "hb_coordinate_gain", "tau", "kappa"):
        _validate_positive(name, settings[name])
    for name in ("tau_n", "tau_v", "noise_eeg_sd", "noise_hb_sd", "component_amplitude"):
        _validate_positive(name, settings[name], allow_zero=True)
    if not np.isfinite(settings["component_correlation"]) or not abs(settings["component_correlation"]) < 1:
        raise ValueError("component_correlation must be finite strictly between -1 and 1")
    initial = np.asarray(settings["initial_state"], float)
    if initial.shape != (5,) or not np.isfinite(initial).all() or np.any(initial[1:] <= 0):
        raise ValueError("initial_state must be finite [s,f,v,p,q] with positive compartments")
    settings["initial_state"] = initial.copy()
    observation_rng = _rng(seed, partition, identity, 3)
    noise_eeg = observation_rng.normal(size=(STEPS, 30))
    noise_hb = observation_rng.normal(size=(STEPS, 2))
    common = _colored_process(_rng(seed, partition, identity, 4))
    angle = float(_rng(seed, partition, identity, 5).uniform(-np.pi, np.pi))
    direction = np.array([np.cos(angle), np.sin(angle)])
    loading = spectral_loadings()
    kwargs = dict(parameters=parameters, hb_operator=operator, noise_eeg=noise_eeg,
        noise_hb=noise_hb, component_process=common, component_direction=direction,
        identity=identity, scenario=scenario, family=family, partition=partition,
        seed=seed, duration=duration)
    control = _record(a, b, settings, loading=loading, **kwargs)
    changed = dict(settings, initial_state=initial.copy())
    a_changed, b_changed, changed_loading = a.copy(), b.copy(), loading.copy()
    driver_override, gain_trajectory, alternative = None, None, False
    causal_names = []
    intervention_rng = _rng(seed, partition, identity, 6)
    if scenario in ("true_a_amplitude", *ALIAS_SCENARIOS):
        factor = float((overrides or {}).get("alias_scale", 2.))
        _validate_positive("alias_scale", factor)
        if factor == 1:
            raise ValueError("amplitude/alias intervention requires a factor other than one")
        a_changed *= factor
        causal_names.append("a_amplitude")
        if scenario in ALIAS_SCENARIOS:
            changed["feature_coordinate_gain"] /= factor
            causal_names.append("inverse_common_feature_coordinate_gain")
    elif scenario == "true_a_shape":
        a_changed = amplitude*_dct_shape(intervention_rng, float(intervention_rng.uniform(2., 8.)))
        causal_names.append("a_shape_at_fixed_reference_RMS")
    elif scenario == "feature_gain_only":
        changed["feature_coordinate_gain"] *= 1.8
        causal_names.append("common_feature_coordinate_gain")
    elif scenario == "Hb_gain_only":
        changed["hb_coordinate_gain"] *= 1.7
        causal_names.append("Hb_mean_coordinate_gain")
    elif scenario in ("common_Hb", "correlated_Hb_component"):
        changed["component_amplitude"] += .008
        if scenario == "correlated_Hb_component":
            changed["component_correlation"] = .85
        causal_names.append("additive_processed_Hb_component")
    elif scenario == "tau_change":
        changed["tau"] *= 1.6
        causal_names.append("compartment_transit_tau")
    elif scenario in ("initial_change", "nonrest"):
        changed["initial_state"] = initial*np.exp(np.array([0., .035, -.025, .025, -.035]))
        changed["initial_state"][0] += .015
        causal_names.append("physical_initial_state")
    elif scenario == "non_dct_impulse":
        impulses = np.zeros(STEPS)
        impulses[[29, 43, 64, 83]] = [1., -.2, .65, -.3]
        a_changed = amplitude*_unit_shape(impulses)
        causal_names.append("a_outside_slow_DCT_impulses")
    elif scenario == "ou_drive":
        a_changed = amplitude*_colored_process(intervention_rng, .88)
        causal_names.append("a_outside_slow_DCT_OU")
    elif scenario == "time_varying_feature_gain":
        gain_trajectory = changed["feature_coordinate_gain"]*(1.+.5*np.sin(np.arange(STEPS)*DT*.28))
        causal_names.append("time_varying_common_feature_coordinate_gain")
    elif scenario == "rotated_spectral_mixture":
        rotation = np.array([[np.cos(.55), -np.sin(.55)], [np.sin(.55), np.cos(.55)]])
        changed_loading = loading @ rotation
        causal_names.append("spectral_loading_rotation")
    elif scenario == "input_lowpass":
        changed["tau_n"] = 1.5
        causal_names.append("vascular_input_lowpass_tau_n")
    elif scenario == "viscoelastic_outflow":
        changed["tau_v"] = 2.
        causal_names.append("viscoelastic_outflow_tau_v")
    elif scenario == "independent_Hb_drive":
        driver_override = amplitude*_dct_shape(intervention_rng, float(intervention_rng.uniform(2., 8.)))
        changed["driver_route"] = "independent_from_a_and_b"
        causal_names.append("independent_Hb_drive")
    elif scenario == "no_coupling":
        driver_override = np.zeros_like(a)
        changed["driver_route"] = "zero_neural_to_Hb_coupling"
        causal_names.append("zero_neural_to_Hb_coupling")
    elif scenario == "coupling_change":
        changed["truth_gamma"] = .7
        changed["driver_route"] = "a_plus_gamma_b"
        causal_names.append("EEG_contrast_to_Hb_coupling")
    elif scenario == "alternative_hrf":
        alternative = True
        causal_names.append("independent_illustrative_double_gamma_forward")
    intervention = _record(a_changed, b_changed, changed, loading=changed_loading,
        driver_override=driver_override, alternative=alternative,
        feature_gain_trajectory=gain_trajectory, **kwargs)
    if scenario in ALIAS_SCENARIOS:
        intervention["eeg"] = control["eeg"].copy()
        intervention["metadata"]["observational_alias_modalities"] = ["eeg"]
        intervention["metadata"]["source_identifiability"] = "EEG_amplitude_nonidentifiable"
    if scenario == "exact_joint_alias":
        # A deliberately compensating observational component constructs the
        # joint alias. It is retained separately from random sensor noise.
        intervention["observation_component"] += control["hb"]-intervention["hb"]
        intervention["hb"] = control["hb"].copy()
        intervention["metadata"].update(constructed_ambiguity=True,
            observational_alias_modalities=["eeg", "hb"],
            source_identifiability="joint_amplitude_nonidentifiable_constructed_alias",
            component_construction="exact_Hb_compensation_not_ordinary_noise")
        causal_names.append("exact_Hb_observation_compensation")
    if scenario in ("independent_Hb_drive", "no_coupling"):
        intervention["metadata"]["source_identifiability"] = "a_not_identified_by_Hb_under_true_route"
    for name, record in (("control", control), ("intervention", intervention)):
        record["metadata"].update(pair_role=name, causal_interventions=[] if name == "control" else causal_names)
    return dict(control=control, intervention=intervention,
        metadata=dict(schema=SCHEMA, base_id=identity, partition=partition, seed=int(seed),
            scenario=scenario, family=family, causal_interventions=causal_names,
            paired_noise=True, paired_source_identity=True, label_use=False,
            success=control["success"] and intervention["success"],
            source_truth_precedes_tokenization=True))


def observation_view(record: Mapping) -> dict:
    """Only observed inputs and support masks; never mechanism or source truth."""
    return {name: np.asarray(record[name]).copy() for name in
            ("eeg", "hb", "time_s", "eeg_mask", "hb_mask", "observation_mask")}


def response_case_index(partition: str, family: str = "randomized", *,
        scenarios: Iterable[str] = ("baseline",), seed: int = DEFAULT_SEED,
        indices: Iterable[int] | None = None) -> list[dict]:
    """Return small generation specs; no observations or source arrays allocated."""
    if partition not in PARTITION_SIZES or family not in ("restricted", "randomized"):
        raise ValueError("unknown partition or family")
    scenarios = tuple(scenarios)
    if not scenarios or any(s not in SCENARIOS for s in scenarios) or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be nonempty, unique and known")
    indices = range(PARTITION_SIZES[partition]) if indices is None else tuple(indices)
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
           or not 0 <= int(i) < PARTITION_SIZES[partition] for i in indices):
        raise ValueError("indices must lie within the fixed partition")
    if len(set(indices)) != len(indices):
        raise ValueError("base indices must be unique")
    return [dict(seed=int(seed), partition=partition, family=family,
                 base_id=int(index), scenario=scenario)
            for index in indices for scenario in scenarios]


def iter_response_cases(partition: str, family: str = "randomized", *,
        scenarios: Iterable[str] = ("baseline",), seed: int = DEFAULT_SEED,
        indices: Iterable[int] | None = None) -> Iterator[dict]:
    """Yield independent paired records without allocating a complete dataset."""
    for spec in response_case_index(partition, family, scenarios=scenarios, seed=seed, indices=indices):
        yield generate_response_case(**spec)


def mechanism_case_index(*, replicas: int = 8, amplitudes: Iterable[float] = (.0125, .025, .05),
        scenarios: Iterable[str] = MECHANISM_SCENARIOS, seed: int = DEFAULT_SEED,
        partition: str = "evaluation", family: str = "restricted") -> list[dict]:
    """Balanced bounded mechanism specs, with distinct training/evaluation streams.

    Defaults contain eight independent source replicas times three amplitudes
    for every mechanism. Siblings share identity/noise; each amplitude/duration
    has its own identity, and no evaluation source seed is a training seed.
    """
    if (isinstance(replicas, bool) or not isinstance(replicas, (int, np.integer))
            or not 1 <= replicas <= 32 or partition not in PARTITION_SIZES
            or family not in ("restricted", "randomized")):
        raise ValueError("valid partition/family and 1 <= replicas <= 32 required")
    amplitudes, scenarios = tuple(amplitudes), tuple(scenarios)
    if (not amplitudes or any(not np.isfinite(a) or not .002 <= a <= .15 for a in amplitudes)
            or len(set(amplitudes)) != len(amplitudes)
            or not scenarios or any(s not in SCENARIOS for s in scenarios)
            or len(set(scenarios)) != len(scenarios)):
        raise ValueError("unique bounded amplitudes and known unique scenarios required")
    return [dict(seed=int(seed), partition=partition, family=family,
        base_id=f"mechanism/r{replica:02d}/a{amplitude:.8g}",
        scenario=scenario, amplitude=float(amplitude), duration=float(3.+replica % 5))
        for replica in range(replicas) for amplitude in amplitudes for scenario in scenarios]
