# Experiment sequencing

_Owning protocol: `PST-DISCOVERY-v1` · state: planned · 2026-08-26_

This is the single current experiment-design owner for the next
physiology-semantic tokenizer generation. The plan is limited to physical-teacher
qualification, source/observation tokenization, and coupling-prior retention. It
does not use downstream task performance as a training or selection endpoint.

The bounded SSM entry points are indexed in
[`experiments/README.md`](../experiments/README.md). The synthetic P0 launcher
remains the qualification path; a separate synthetic-only `T3c` composite
`T-P2` screen tests the gain/time directions but is decision-ineligible. The
measured reconstruction/null, fit-only identifiability, and fit-only
three-session LOSO launchers are nonprotected development diagnostics with
their own executable contracts. An array-free `T3c` admission gate checks
whether hierarchical composite fitting may start. Reconstruction fits on
subjects 01--18 and applies frozen objects to subjects 19--23.
Identifiability and LOSO use only subjects 01--18 and load no 19--23 arrays;
all three measured diagnostics keep subjects 24--29 closed. Their outputs are
exploratory; they are not clean truth, teacher qualification, or a
physical-teacher claim.
Protected data and every other protected surface remain closed.

The measured diagnostic does not replace the synthetic qualification gates or
authorize tokenizer promotion. Any future measured confirmation or physical
teacher qualification still requires the unresolved margins, primary
estimand, calibration, and compute decisions below to be frozen in a separate
contract.

## Step5 observation adaptation diagnostic

The requested follow-up to the September 7 Step5 results is owned by
[`step5_observation_diagnostic_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml)
and [`evaluate_step5_observation_diagnostic.py`](../experiments/evaluate_step5_observation_diagnostic.py).
It retains the six-state model, GWZ support and Step5B negative decision. The
user's request is for this bounded observation diagnostic; no prior run status
or historical authorization is used to open a new qualification campaign.

The measured panel fixes subjects 01/09/18 (first/middle/last of the Step5
development inventory) before its results. It uses only the original eight
training MA trials in each of sessions 01/03/05. Original trial positions 4/9
are excluded before slicing and processing. Native files contain full sessions;
that storage fact does not make their other trials diagnostic inputs. The
existing metadata validator verifies record/event/clock contracts before native
access. Subjects 19–29 are outside the loader's scope. Old Step5 code, frozen
configurations, completed runs and failures remain unchanged.

The synthetic bridge uses 24 independent matched-model trials and 24 independent
noise-estimation trials, with truth W=0. It compares original model coordinates,
baseline subtraction, the fNIRS filter operator, a controlled 4→10→4 polyphase
resampling round trip, a common HbO/HbR amplitude factor, training noise-scale
estimation, and their combination. The filter uses the existing 0.01–0.2 Hz
third-order implementation on the controlled 4 Hz coordinate. This isolates
operators; it does not claim to simulate native raw EEG, optical motion/MBLL,
or the full native 10 Hz filter distribution. Both W=0 and W=−0.5 are fixed
diagnostic settings. No parameters are fitted. Known driver/model-clean truth
and processed-clean truth are reported separately; nominal intervals retain
the existing observation model, deliberately exposing operator mismatch.

Training W curves cover the entire original support at 17 points, separately
refitting EEG-only, fNIRS-only and joint filters. They are likelihood curves,
not resolved parameter posteriors. Chronological joint-density increments are
summed over baseline, task and nominal recovery without resetting at segment
boundaries; marginal predictive scores do not replace the joint likelihood.
Fixed-setting innovations report bias, within-trial autocorrelation and PSD.
The first original training trial provides predeclared 13/17-order endpoint
checks. EEG-only W invariance is also checked against the likelihood owner.

The only alternative EEG coordinate is positive F3 8–13 Hz log block power,
with a training-fitted scale in the same reference gauge. It is a prespecified
left-frontal diagnostic, not validated anatomical correspondence to whichever
fNIRS pair is selected. There is no post-result channel/band/sign search.

The low-capacity control predicts a center-masked modality from visible own
endpoints and a same-session training task template, then adds six fixed lags
of the other modality. It uses four outer and three inner folds within the
original training inventory; every inner/outer fit repeats pair selection,
projection and scaling. Inner trial folds choose ridge strength. EEG→fNIRS
uses preceding EEG; fNIRS→EEG uses later fNIRS and is explicitly offline.
Independent-training-trial pairing and a half-trial circular shift change only
the other modality at validation. Template construction excludes the example's
own trial. Scores are negative MSE in outer-training variance units, with
subject-cluster summaries. Three clusters support descriptive localization,
not a new confirmation or teacher admission claim. Missing/failing registered
cases are retained, never replaced. Comprehensive UQ and tokenizer promotion
remain outside this diagnostic.

## Step5A0 inference consistency diagnostic

[`step5a_inference_consistency_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5a_inference_consistency_v1.yaml)
owns the small synthetic localization panel requested in `ssm_next.md`.
[`evaluate_step5a_inference_consistency.py`](../experiments/evaluate_step5a_inference_consistency.py)
implements it without modifying Step 1–4 code, configurations, evidence, or
negative decisions. This is a diagnostic before Step5A1 teacher checks, not a
new qualification gate or a measured-data campaign.

The primary endpoints are forward/derivative equivalence and inference
consistency. G and W have truncated-normal priors in relative log gain and
log frequency; Z has a uniform prior in physical damping ratio. Other
coordinates stay at the reference value. The runner integrates densities in
these coordinates with trapezoidal weights and reports prior/posterior mode
separation, normalized boundary distance, boundary mass, interval width, and
quadrature refinement. W is the negative of Step 4's log-time coordinate.

- `oracle_r_known` conditions on a deterministic, prescribed driver and rest
  hemodynamic initial state, with zero state diffusion. Independent numerical
  integration settings are compared, and the likelihood is the product of
  conditional Student-t observation densities.
- `matched_model_calibration` draws the parameter from its declared prior,
  the transformed initial state from the configured zero-mean Gaussian, and
  all six transition innovations independently. Its transition law is
  `z[t+1] = RK4(z[t]) + epsilon`, with covariance
  `dt * diag(process_std**2)`. This is the discrete model approximated by the
  fitter; it does not assert exact continuous-time SDE simulation.
- The production filter and smoother are checked in a linear Gaussian
  specialization against exact Kalman filtering/RTS. A bootstrap particle
  filter then estimates the joint likelihood for short matched prefixes.
  Independent likelihood estimates are averaged on the likelihood scale.
  Particle budgets, independent-run splits, grid refinement, effective sample
  size, and surviving ancestors determine whether a reference comparison is
  numerically resolved. An unresolved reference remains inconclusive.
- `misspecification_stress_test` calls the retained Step 4 truth generator
  with shorter records, preserving its pulse, external driver, deterministic
  hemodynamics, and Student-t noise. It is not labelled SBC. The old failure
  establishes that the old complete pipeline failed its registered gates;
  it did not isolate identifiability, model mismatch, and inference error.

The existing EKF marginal score is named `predictive_score` in this diagnostic
and produces a **generalized posterior**. Only the direct oracle likelihood
and joint particle likelihood use `parameter_log_likelihood`. Increasing grid
resolution alone does not validate the former as a likelihood.

Secondary diagnostics compare U0 fixed, true-parameter conditional inference,
and the score-optimal one-parameter fit against true `r` and clean EEG/HbO/HbR.
State-posterior variance excludes observation noise; noisy masked-observation
intervals are reported separately. These are conditional Gaussian-moment
intervals, without parameter-uncertainty propagation. Coverage is aggregated
by independent replicate before descriptive bootstrap; the small panel cannot
establish SBC or teacher qualification. The no-observation prior is a software
negative control. Shared-information pairing nulls, parameter-integrated UQ,
U3, and the registered larger calibration panel belong to later experiments.

Masked fNIRS values are removed before estimation, including parameter refits.
There is no normalization or data-derived noise estimation in this synthetic
panel. Numerical failure stops the run and retains its failure record; seeds
are not redrawn and thresholds are not relaxed after seeing results. Outputs
use a fresh directory under the configured experiment root. No measured or
protected data, tokenizer target, independent-modality ownership, or coupling
contract changes are included. Current execution and next action remain in
the research-state registry.

## Full Step5 staged continuation

The user-requested continuation is governed by
[`step5_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml)
and [`evaluate_step5.py`](../experiments/evaluate_step5.py). Missing numerical
details in `ssm_next.md` are frozen in that configuration before the relevant
stage is evaluated. Stage results are reported separately, with numerical,
parameter, state, and shared-information conclusions distinguished.

**Step5A0** replaces IRLS observation curvature with joint Student-t
Gauss-Hermite integration and Gaussian moment matching in a separate
[`joint inference module`](../src/inference/t3a_balloon_joint_ssm.py). The
immutable T3a dynamics and transition Jacobian remain the forward owner.
The likelihood integrates the observation-active transformed coordinates;
the other states are handled by Gaussian conditional regression. EKF dynamics
and Gaussian closure remain approximations. The new matched calibration panel
uses fresh independent parameter/noise draws and compares oracle, legacy
marginal-score, and joint-likelihood parameter distributions. Numerical
refinement changes grids, not observations or statistical thresholds.

Particle likelihood precision and path ancestry are separate diagnostics.
Likelihood-scale independent estimates, grid/budget checks, and log-likelihood
Monte Carlo error govern the likelihood reference. Surviving ancestor fraction
governs whether those same particles may support path smoothing. The prior
combined check is also reported; a path failure is retained and cannot be
described as a fully resolved particle smoother.

**Step5A1** tests U0 and the minimal G/W/Z candidates. A parameter distribution
is fitted to an independent synthetic training trial; the state targets and
masked scores use newly generated held-out trials. For the stress branch,
both training and held-out observations use the retained Step 4 generator.
The parameter distribution is held fixed across correct-pair and null inputs;
it cannot carry information about the held-out pair. This is a frozen-training
parameter mixture of conditional state posteriors, not a claim that parameter
weights have been updated to the full joint posterior using the held-out
record. Its uncertainty must pass the declared state coverage checks.

Variance separates the mean conditional state variance from variance across
parameter means; only noisy-observation prediction adds Student-t noise.
Posterior-CDF quadrature is refined for teacher mean/variance stability.
U0 additionally receives the predeclared GWZ prior-quantile sensitivity panel.
Cross-parameter driver stability and known-truth recovery are both reported.
U3 uses a full-support tensor grid with explicit refinement, posterior
correlation/ridge, and boundary-mass diagnostics; it is not eligible for
selection. Parameter intervals are compared with configured material changes,
not a point-identification or “any equivalent alternative fails” rule.

Shared information requires paired improvement over own history, own history
plus an independently trained task-time template, independent pairing, and
circular-shift controls in both center-masked directions. Whole-modality
missingness is reported as a separate diagnostic. Synthetic clusters are
independent parameter/trial replicates; measured clusters are subjects. No
timepoint-level binomial confidence claim is used for trajectory coverage.
Every fixed case identity must receive either its complete result or a retained
failure record. Generation, prior-support or inference exceptions are not
redrawn, filtered out, or repaired by changing frozen clipping/step-size rules.
Partial successful-case summaries are descriptive and cannot grant teacher
qualification when the registered experiment is incomplete. Independent
same-seed numerical diagnostics may explain a failure without reclassifying it.

**Step5B** is restricted to the explicitly listed development subjects and
sessions, with the configured eight training and two held-out trials per
session. It targets a new trial in an existing subject/session. U0 and a
synthetically qualified minimal one-parameter candidate are evaluated; if no
free candidate qualifies, a qualified U0 may be evaluated alone. Every
data-dependent transform and parameter fit uses the training inventory.
Input masking precedes information-propagating transforms, using distinct
input and target-scoring processing where required. Loader source facts and
the concrete mask-processing implementation are checked before array access.
Subjects 19–23 are not a new confirmation cohort and are not included in this
version; protected subjects 24–29 remain closed.

The measured implementation's additional numerical choices were first frozen in
[`step5b_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5b_v1.yaml),
which pins the synthetic base configuration and reuses the immutable
three-session metadata validator for factual record/event/clock checks. That
older diagnostic's authorization fields are not reused as authorization.
The explicit Step5 request and the admitted candidate's own complete synthetic
panel govern progression. A failed independent G/Z candidate does not
invalidate a complete W panel; U0's registered cross-scenario sensitivity does
require the full scenario inventory.

[`step5b_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5b_v2.yaml)
adds same-support local posterior refinement after the retained uniform-grid
failure: old endpoints and tail nodes remain, intervals within the configured
log-density drop are bisected, and the CDF threshold is unchanged. It also
restricts fNIRS channel eligibility using training-only positive native
intensities before channel selection. A selected held-out pair with invalid
support still fails; held-out observations cannot select a replacement pair.
Same-configuration output recovery can reuse prepared inputs, resolved curves
and case outcomes with input digests and unchanged scientific functions. It
adds no independent statistical replicates and preserves prior failure records.

The new native-trial path consumes the cache's `native_input_fnirs` and the
existing native EEG reader. It does not use the globally standardized or
filtered canonical arrays to construct masked inputs. EEG log block power and
trial-local OD/motion/filter/MBLL fNIRS transforms precede a frozen per-subject
projection. Channel selection uses training signal/first-difference ratios;
PCA and projection scales use no task labels. HbO and HbR share a single
training scale, preserving their relative amplitudes. The amplitude gauge is
defined by the fixed reference model covariance; P0/Q0 and the state structure
are not added as free parameters. Event-relative output times and both native
clock anchors remain in the trial inventory.

The first-difference estimator uses the exact median absolute difference of
two unit Student-t draws. Its software check uses retained, known synthetic
noise. The measured scale is the larger of its training estimate and the
frozen synthetic noise scale. This is an initial scale rule, not a calibration
claim about colored measured noise. Center-mask scores compare the qualified
candidate with same-modality visible context (including future context in the
fixed-interval operator), a training task-time template, independent training
pairing and circular shifts. Subject-cluster intervals and leave-one-subject-out
means govern shared-information evidence. Noisy-observation Gaussian-moment
coverage is diagnostic; measured data provide no latent-state coverage truth.

**Comprehensive UQ** follows core teacher qualification. It reports modality
and subject variation, conditional-state/parameter/noisy-observation variance,
mask effects, and calibration with the known-subject new-trial unit explicitly
stated. Leave-one-training-trial-out calibration is descriptive, not a claim
of standard split-conformal finite-sample coverage. Precision weighting is
tested only when the configured clustered risk-ranking criterion is met;
uniform weighting is otherwise retained. Exports preserve the same-modality
observation-space teacher mean, variance, and masks; `r` is a diagnostic
export. No tokenizer training or target redesign is included.

A failed numerical or scientific prerequisite is retained and prevents the
dependent measured/UQ action. “Stage evaluated” does not imply qualification,
and a skipped dependent stage is reported as not run, never as a successful
experiment. Current execution remains solely in the research-state registry.

## Fixed question and decision target

The intended final object is the smallest tokenizer that jointly satisfies:

1. `source` retains the same-modality slice of a qualified offline joint
   EEG+HbO+HbR teacher trajectory;
2. `observation` retains modality-specific measured information;
3. quantization, if admitted, does not materially degrade either function;
4. an optional coupling prior improves a held-out cross-modal proper score over
   the observation/source-history baseline without harming the first three
   properties.

The physical teacher's primary task is a physiology-constrained decomposition,
not pointwise observation copying or EEG-only cross-modal translation. From
aligned noisy EEG, HbO, and HbR observations it estimates an operational shared
neural driver `r(t)`, named hemodynamic states, an observation-space posterior,
and modality-specific or systemic nuisance/residual components. The shared
driver, physiological states, and separated noise are estimates rather than
ground truth and become decision-eligible only after the physical,
identifiability, corruption, null, and calibration checks below.

Observation-space reconstruction remains useful for posterior predictive
checking and for locating failure, but it is not the primary definition of a
good physical teacher. A candidate is not rejected solely because its point
MSE/NRMSE, R², or PCC is worse than persistence or a time-shift control.
Non-finite trajectories, physical-boundary violations, non-identifiable
physiological claims, systematic posterior-predictive failure, or failed
uncertainty calibration still reject the candidate.

The nine forward principles in
[`METHOD_RATIONALE.md`](METHOD_RATIONALE.md#frozen-theory-and-architecture-contract-unimplemented)
and the data/mask/split rules in [`DATA_CONTRACT.md`](DATA_CONTRACT.md) remain
fixed. This protocol selects implementations inside those boundaries; it does
not redefine them.

"Optimal" is deliberately lexicographic rather than a weighted total score:

1. qualify the teacher on physical identity, identifiability, robustness, null,
   and calibration gates;
2. pass every required source and observation fidelity gate;
3. pass uncertainty, stability, and codebook-health floors;
4. among passers, use the lowest token rate and simplest model;
5. use held-out proper score only as the final tie-breaker.

A strong result in one modality cannot compensate for failure in the other. If
continuous representations pass but VQ fails, the result is "no discrete
tokenizer admitted", not a forced codebook. If coupling fails, a qualified
coupling-free tokenizer may still be retained.

## Experiment flow

![PST-DISCOVERY-v1 staged experiment plan](physiology_semantic_tokenizer/figures/pst_discovery_v1_experiment_plan.svg)

[Editable figure source](physiology_semantic_tokenizer/architecture/pst_discovery_v1_experiment_plan.json) ·
[standalone SVG](physiology_semantic_tokenizer/figures/pst_discovery_v1_experiment_plan.svg) ·
[visual style reference](physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)

The upper spine is the only promotion path. The three detailed panels expose the
teacher, tokenizer, and coupling candidate ladders; the bottom lane contains
diagnostic children that are explicitly not decision-eligible.

The bottom diagnostic lane can explain a failure or motivate `v2`; it is not an
in-run retry and cannot be pooled into the `v1` promotion estimate.

## Common estimand and statistical contract

### Unit of inference

The biological unit is the subject. Scores are first aggregated over valid
coordinates, time points, patches, trials, and records within subject and then
averaged with equal subject weight. Channels, teacher coordinates, code IDs,
windows, and seeds are not independent biological replicates. Three paired seeds
are optimization/stability repeats and are reported separately from subject
uncertainty.

For a candidate `C` and baseline `B`, every main contrast is oriented so higher
is better:

```text
delta_s = subject_score_s(C) - subject_score_s(B)
Delta   = mean_s(delta_s)
```

Intervals use subject-cluster bootstrap or an exact subject-block test when the
subject count is too small for a stable bootstrap. Missing required support is
`INVALID`; the denominator is not silently reduced.

### Partitions and leakage control

- All channel selection, normalization, teacher parameters, uncertainty
  calibration, target projections, encoder/checkpoint selection, codebooks, and
  coupling maps are fit inside the authorized fit partition.
- Subject plus record/trial/video dependencies are grouped across every split.
- Historical subjects 01--18 and 19--23 have already influenced prior method
  development and cannot become a genuinely fresh confirmation cohort merely by
  relabeling them.
- The new nonprotected confirmation inventory is unresolved and must be frozen
  before measured execution. Protected subjects 24--29 remain closed.
- Task/condition annotations may define nuisance controls or matched nulls, but
  they are not prediction targets, architecture-selection endpoints, or losses
  in this protocol.

### Decision states and multiplicity

Every gate returns exactly one of `PASS`, `FAIL`, `INCONCLUSIVE`, or `INVALID`.
Confidence intervals crossing zero or a non-inferiority margin are
`INCONCLUSIVE`; technical interruption or incomplete support is `INVALID`.

Each stage has one named primary contrast. Lag, horizon, chromophore, channel,
and candidate families are either descriptive or controlled with a predeclared
max-statistic/closed-testing procedure. No best patch, lag, channel set, seed, or
checkpoint may be chosen after confirmation data are viewed.

The practical margins `delta_T`, `delta_S`, `delta_O`, and `delta_H` are
intentionally unresolved here: each must be estimated from synthetic recovery,
measurement repeatability, or fit-only technical repeats and then frozen before
the corresponding confirmation run. For the teacher, `delta_T` governs shared
driver/state robustness and calibration, not superiority of pointwise
observation reconstruction. A percentage chosen after seeing held-out results
is not an admissible margin.

## P0: software and synthetic qualification

P0 is mandatory before measured data. One synthetic generator must emit known
`r(t)`, extended Balloon states `s/f/v/p/q`, the true parameters and operators,
and clean EEG/HbO/HbR trajectories under an explicit
`p/q -> HbT/HbO/HbR` concentration map (plus the recorded optical operator when
the input coordinate is optical density) and known EEG-to-fNIRS delay. It then
injects heteroscedastic noise and controlled modality-specific or systemic artifacts--spikes, drift,
steps, bursts or high-frequency contamination, and dropout--while retaining the
clean reference, nuisance component, artifact mask, and severity. Full-input,
masked/held-out, and missing-modality replays must use the same generator. The
smallest runnable check must demonstrate:

- continuous target construction before patching/tokenization;
- exact canonical-key joins and distinct measurement, teacher, uncertainty,
  observation-residual, token, and lag masks;
- no cross-modal read before either main tokenizer emits its representation;
  the offline joint teacher is the declared fit-fold-only exception and emits
  detached modality-specific targets;
- the resting equilibrium, positive physiological states, stable integration,
  valid oxygen extraction, Balloon-compartment inflow/outflow, total-Hb and
  deoxy-Hb balances, and the explicit hemodynamic/optical observation map; these
  checks
  do not turn the model into a full oxygen-diffusion or CMRO2 model;
- prior-predictive support plus simulation-based calibration, profile-likelihood
  or equivalent identifiability checks, and multi-start sensitivity for every
  parameter allowed to vary;
- recovery of known `r(t)` and named physiological states, attenuation of
  injected artifacts, and failure on independent/time-shift/pairing/spatial
  nulls; observation-space MSE and correlation remain descriptive;
- residual agreement with injected corruption on artifact support and absence
  of systematic clean-signal removal off that support;
- calibrated predictive intervals, with uncertainty increasing under stronger
  corruption, masking, or missing input;
- branch and coupling gradient allowlists;
- config/target/summary serialization, atomic publication, and an explicit
  incomplete-run state.

P0 remains the qualification path. The separately registered measured
reconstruction/null diagnostic may run only on its nonprotected development
split and remains decision-ineligible; it cannot open protected data or promote
a teacher. Protected evaluation requires a separate explicit request.

## T: physical-teacher selection

### Selection principle and candidate range

The comparison is a staged ladder, not a Cartesian model search. Controls and
mechanism references cannot become the physical teacher merely by winning a
reconstruction metric. The first promotion candidate is the smallest robust
nonlinear Balloon model with explicit observation operators.

| ID | Candidate | Frozen question | Role |
| --- | --- | --- | --- |
| `T0-native` | measured coordinates with persistence, time-shift, and fit-fold smoothing controls | How much apparent recovery requires no latent physiology? | predictive control; never promoted |
| `T1-self` | independent EEG and fNIRS linear LDS/RTS models | How much smoothing and uncertainty calibration is available without a shared state? | single-modality attribution control |
| `T2a-croce-pf` | paper-faithful Croce-2017 nonlinear particle-filter mechanism | Which published Croce behaviours reproduce under the same synthetic contract? | fixed mechanism reference; not the default teacher |
| `T2b-adaptive-legacy` | current bounded adaptive Croce-like RTS/AR implementation | Which current results survive the new physical and identifiability tests? | historical regression baseline; never relabelled as exact Croce |
| `T3a-balloon-robust` | constrained nonlinear `r/s/f/v/p/q` extended Balloon state model, explicit EEG and fNIRS optical observations, masks, and fixed-degree-of-freedom Student-t observation noise | Can the model recover an identifiable shared drive and plausible physiological states while isolating corruption? | **primary promotion candidate** |
| `T3b-systemic` | `T3a` plus one low-dimensional fNIRS systemic/extracerebral nuisance factor | Does a frozen residual/PPC failure specifically improve without absorbing `r(t)`? | conditional extension after its predeclared `T-P3`/`T-G4` trigger |
| `T3c-hierarchical` | partial pooling of only parameters already identifiable in `T3a` | Does cross-subject pooling improve stability without prior domination? | conditional extension only after `T-P2` identifiability and a frozen cross-subject stability failure |
| `T4-dcm-lite` | two-stage EEG neural-state to Balloon/optical fNIRS model; fNIRS cannot retroactively rewrite the EEG neural state | Does a more conventional directed interpretation support the same physiology? | interpretability reference, not a joint-teacher promotion arm |
| `T5-spatial` | local geometry extension of the simplest `T3` model passing `T-G0`--`T-G4` | Is additional local spatial support necessary after physiology qualifies? | final conditional refinement at `T-G5` |

The executable synthetic P0 panel is intentionally smaller:
`T0-native`, `T1-self`, `T2b-adaptive-legacy`, and
`T3a-balloon-robust`. `T2a-croce-pf` and `T4-dcm-lite` remain frozen design
references until a contract-faithful adapter exists; they must not appear as
tested or unavailable rows manufactured from `NaN`. This P0 qualifies the
primary candidate and its current controls, not the later `T-P5` comparison.

Gamma-HRF/delay controls, Factorial/SLDS noise branches, switching regimes,
heteroscedastic process models, Gaussian-process dynamics, and full neural-mass
models remain diagnostics. They are not part of the first promotion ladder.
`T3a` does not simultaneously add switching, hierarchy, spatial structure, and
multiple nuisance factors.

`T5-spatial` starts from one HbO/HbR pair plus six nearest EEG channels, then
tests two and at most four local fNIRS pairs with at most twelve EEG channels,
subject to actual channel support. It reuses the existing adjacency/geometry
owners. A geometry-aware linear observation operator and covariance are tested
before any graph neural network. Template geometry supports adjacency and
qualitative topology only, not exact cross-modal distance or co-registration.

Channel sets are selected on fit data without labels. Added channels are
retained only if they improve a frozen posterior-predictive or proper-score
endpoint, survive channel-drop and geometry-permutation nulls, and do not
degrade calibration or state stability. Otherwise the smaller local model wins.
An all-scalp model is not part of this generation.

### Physiological state and parameter contract

The initial `T3a` continuous-time core follows the normalized Balloon dynamics
of [Friston et al. (2000)](https://www.fil.ion.ucl.ac.uk/spm/doc/papers/karl_nonlinear.pdf)
and the total-Hb/optics extension of
[Tak et al. (2015)](https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf).
Those papers define the model family; their fitted prior means are not treated
as universal human measurement ranges.

```text
ds/dt       = beta * r - kappa * s - gamma * (f - 1)
df/dt       = s
f_out       = v^(1/alpha)
tau * dv/dt = f - f_out
tau * dp/dt = f - f_out * p / v
tau * dq/dt = f * E(f, E0) / E0 - f_out * q / v
E(f, E0)    = 1 - (1 - E0)^(1/f)

domain: f > 0; 0 < E0 < 1; 0 < E(f, E0) < 1
rest:   r = s = 0; f = v = p = q = 1
```

Here `r` is the shared neural state in the fixed EEG loading/variance gauge; it
is not measured firing. `beta` is a dimensionless effective neural-to-vascular
gain in that gauge, not a molecular efficacy constant.
`s = df/dt` is the vasoactive signal; `f` is inflow normalized to rest; `v` is
normalized venous Balloon volume; and `p/q` are the normalized total-Hb/deoxy-Hb
model coordinates of that compartment. With time in seconds, `f/v/p/q` are
dimensionless, `s` has units s^-1, `r` and `gamma` have units s^-2, `beta` is
dimensionless, `kappa` has units s^-1, and `tau` has units s. `tau` is the resting transit constant
`V0/F0` of the modeled venous Balloon, not whole-region or whole-brain mean
transit time. `alpha` is its dimensionless outflow-volume exponent. A numeric
prior from another state/time scaling is usable only after its unit conversion
is recorded; copying a published coefficient labelled only as a "rate" into
this parameterization is a `T-P0` failure.

This initial model fixes Tak et al.'s viscoelastic time constant `tau_v` to
zero, so `f_out = v^(1/alpha)`. It is therefore the smallest explicit
total-Hb extension needed for `T3a`, not a claim to reproduce the paper's full
viscoelastic model. A nonzero `tau_v` is admitted only as a later one-parameter
extension after the initial state and parameter contract is identifiable.

The fNIRS forward model must be explicit rather than learned through arbitrary
signed gains:

```text
delta_HbT = P0 * (p - 1)
delta_HbR = Q0 * (q - 1)
delta_HbO = delta_HbT - delta_HbR
```

`P0` and `Q0` are positive baseline scales. If the declared measurement
coordinate is raw optical density, the above concentrations additionally pass
through the recorded wavelength-specific extinction, sensitivity/pathlength,
and cortical-mixing operator. If the coordinate is a released HbO/HbR export,
that optical-density transform is not applied a second time; its recorded
preprocessing/normalization transform is part of the observation operator.
Without those baselines and the recorded optical lineage, `p/q` remain
dimensionless model coordinates and cannot be relabelled as absolute Hb
concentrations. EEG has its own declared observation operator.

The parameter contract separates three kinds of restriction:

- **Hard mathematical/physical boundaries:** `kappa`, `gamma`, `tau`, and
  `alpha` are positive; `0 < E0 < 1`; `f`, `v`, `p`, `q`, and `f_out` remain
  positive; `E(f,E0)` remains in `(0,1)` at every step; `P0 > 0`, `Q0 > 0`,
  and the mapped absolute HbT/HbR/HbO values remain nonnegative with HbR not
  exceeding HbT. The resting equilibrium, compartment balances, units,
  finite integration, and stability checks must pass. These are validity
  conditions, not fitted medical ranges.
- **Neural-drive gauge:** set the baseline of `r` to zero, fix its sign so a
  positive drive increases `s`, normalize its scale by one predeclared
  fit-fold rule, and fix one EEG observation loading. The conventional
  `epsilon` factor is absorbed into `r`; no separate neural-efficacy parameter
  is fitted or reported as measured physiology.
- **Soft source-backed priors:** every numeric prior and plausible-response
  interval must record its units, compartment, species/population and challenge
  condition, primary source, and prior parameterization in the executable
  contract. A posterior pressed against a bound or unchanged from its prior is
  not evidence that the parameter was measured.
- **Measured exploratory release ladder:** retain `P0/Q0`, EEG loading, driver
  scale, noise, and Student-t degrees of freedom as fit-cohort gauges. Compare
  the fixed model first, then the single-parameter `beta`, `kappa`, and `tau`
  fits, then `beta+kappa+tau`, followed by one-at-a-time additions of `gamma`
  and `alpha`. Release `E0` only as a final strong-prior diagnostic because the
  current standardized fNIRS coordinate cannot establish absolute OEF. Only the
  fixed model and the three single-parameter fits are recommendation-eligible;
  `M2`--`M5` are retained only to diagnose compensation. Each subject shares one
  parameter vector across independently reset trials. A later stage cannot be
  retained merely for reconstruction gain when its posterior is boundary-bound,
  prior-dominated, or compensatory. `p` has no separate free dynamic parameter
  in `T3a`.

Names must not overstate what the equations identify. `kappa` and `gamma` are
lumped model coefficients, not direct molecular vasodilation rates; `E0` is the
resting oxygen extraction fraction, not an oxygen dissociation rate. Without
absolute flow/volume and optical calibration, the experiment cannot claim
absolute OEF, CMRO2, or an oxygen dissociation rate. Such quantities remain
outside the result vocabulary even when the latent trajectory looks plausible.

### Teacher test sequence

| Stage | Test items | Promotion consequence |
| --- | --- | --- |
| `T-P0 semantics/physics` | state names, equations, units, gauge, observation map, equilibrium, positivity, finite/stable integration, and parameter-source ledger | any violation is `FAIL` before fitting |
| `T-P1 prior predictive` | draw prior trajectories across the frozen design; check plausible amplitudes/delays, boundary contact, solver failures, and prior sensitivity | unsupported priors or implausible mass dynamics block the candidate |
| `T-P2 identifiability` | simulation-based calibration using the declared EKF-Laplace posterior-CDF approximation, rank/coverage diagnostics, fixed-other-parameter objective slices as the initial posterior-geometry check, multi-start recovery, and parameter/state confounding | non-identifiable parameters are fixed/removed; stable `r` alone earns only state-level status; exact posterior SBC is required if the Laplace approximation itself fails calibration |
| `T-P3 known-truth corruption` | recover `r/s/f/v/p/q`, separate known artifacts/nuisance, preserve clean off-artifact morphology, vary severity/masks/missing modalities, and run independent/time-shift/pairing/spatial nulls | qualifies shared-state and noise-separation claims; point reconstruction metrics remain descriptive |
| `T-P4 measured development` | posterior-predictive checks, residual temporal/spectral structure, modality ablations, leave-one-trial/subject-out stability, and prior-to-posterior movement | permitted only after an executable measured-data contract; no protected access |
| `T-P5 comparison/spatial` | compare the simplest surviving models by predictive score, calibration, complexity, perturbation stability, and spatial/channel nulls | select the smallest fully qualified teacher; otherwise stop |

The current authorized P0 software/synthetic scope covers `T-P0` through
`T-P3` and the known-clean synthetic portion of `T-G4`. Final `T-G4`, `T-P4`,
`T-G5`, and `T-P5` require the later executable measured-data contract; this
plan does not open measured or protected data.

The synthetic `T-G4` screen uses Student-t interval/proper-score calibration
plus lag-one autocorrelation and normalized-spectrum errors of the posterior
mean. Those two reconstruction-shape diagnostics are not full posterior-
predictive simulations and are not labelled PPC in the executable output.

### Teacher outputs and uncertainty convention

All candidates publish common observation and diagnostic fields:

```text
trajectory_mean
aleatoric_variance
epistemic_variance
total_variance = aleatoric_variance + epistemic_variance
observation_values
observation_residual = observation_values - trajectory_mean
nuisance_mean / nuisance_variance, when the candidate declares a nuisance state
named masks and coordinate/channel identities
fit, model/config, parameter, and calibration identities
```

Physiological candidates additionally publish, with explicit state names:

```text
shared_driver_mean / shared_driver_variance
physiological_state_mean / physiological_state_variance
parameter_posterior_summary
parameter_identifiability_status
physical_check_status
```

`shared_driver_mean` is the operational `r(t)` estimate.
`physiological_state_mean` contains only states actually present and identified
in the fitted model. `trajectory_mean` is an observation-space posterior
prediction, not a clean-ground-truth claim. `observation_residual` may be
described as separated noise/artifact only to the extent supported by `T-P3`;
otherwise it remains an unassigned observation residual.

The contract uses **variance**, not an ambiguous `uncertainty` scalar. The
legacy adapter mixes variance-like summaries while the current loss divides by
that field without a log-variance term; therefore its uncertainty-weighting
switch is not admitted evidence for this generation.

Calibration is fit-fold-only and frozen before application. Primary uncertainty
endpoints are predictive log score and CRPS on known-clean synthetic coordinates
and prespecified masked real coordinates. 50/80/95% interval coverage and width,
standardized residuals, PIT, and risk-versus-uncertainty monotonicity are
required diagnostics. Same-point joint-posterior coverage is descriptive
because the observation was consumed by the smoother. Aleatoric and epistemic
components remain separate in the artifact and report.

### Teacher gate

| Gate | Required evidence |
| --- | --- |
| `T-G0 physical contract` | lineage, folds, masks, state/operator identity, units, sign/gauge, equilibrium, positivity, finite/stable integration, no label use, and no protected dereference |
| `T-G1 prior/synthetic validity` | source-frozen priors have plausible prior-predictive support; synthetic `r/s/f/v/p/q` and observations are generated without extraction, boundary, compartment-balance, or optical-map failure across the frozen design |
| `T-G2 identifiability` | SBC/coverage, posterior geometry or profile checks, and multi-start recovery support every reported state/parameter; prior-dominated or mutually confounded quantities cannot receive physiological labels |
| `T-G3 shared-state/noise adequacy` | `r(t)` and admitted states remain within frozen perturbation limits; known artifacts enter nuisance/residual rather than the physiological state; off-artifact leakage stays below its frozen bound; independent/time-shift/pairing/spatial null inputs do not yield a qualified shared state |
| `T-G4 calibration/PPC` | predictive log score, CRPS, interval calibration, uncertainty-risk monotonicity, and prespecified temporal/spectral posterior-predictive checks pass; MSE/NRMSE/R²/PCC and same-point reconstruction are descriptive only |
| `T-G5 measured/spatial stability` | measured modality ablations and subject/fold/seed/channel perturbations preserve the admitted claims; any spatial gain survives channel and geometry nulls without worse calibration |

Only the simplest `T3` candidate passing all applicable gates becomes the
frozen training-target producer. It is privileged, label-blind, fit-fold-only,
and training-only; it is never a tokenizer inference input or ground truth.
EEG-only and fNIRS-only reruns are attribution and missing-modality ablations,
not requirements that EEG reconstruct omitted HbO/HbR or vice versa.

Qualification has three explicit outcomes. Passing the full gate yields a
physical teacher. A robust `r(t)` with non-identifiable physiological parameters
is a state-only diagnostic and cannot support parameter-level interpretation.
A model that only smooths observations remains a baseline. If no `T3` candidate
passes `T-G0`--`T-G4`, source-tokenizer development stops; reconstruction work
may continue only as a diagnostic.

## B/Q: source and observation tokenizer

### Functional implementation

The first implementation uses one simple modality-local temporal stem with two
heads:

```text
X_m -> stem_m -> source latent      -> teacher-trajectory decoder
             -> observation latent -> measured-signal decoder
```

EEG and fNIRS stems never read the other modality. Source and observation are
functional roles, not an assertion of statistical independence, and they need
not start as four physically separate encoders. A separate stem is considered
only if the shared-stem gradient audit demonstrates reproducible interference.

The observation target is the measured/masked modality coordinate. It is not
defined as `raw - source` unless a later diagnostic first establishes compatible
units and an identifiable additive decomposition. This avoids repeating the old
power-versus-voltage and single-decoder ambiguity.

### Candidate sequence

| ID | Change from previous row | Question |
| --- | --- | --- |
| `B0-O` | continuous observation-only autoencoder | What reconstruction is available without teacher semantics? |
| `B1-SO` | add continuous source head and frozen teacher supervision | Can both functional roles pass before discretization? |
| `Q-S` | quantize source only with the existing EMA-VQ family | Are physiological source patterns discretizable without losing semantics? |
| `Q-O` | quantize observation only after `Q-S` passes and only if a fully discrete interface is required | Can measured information also survive the bottleneck? |

The initial temporal grid and latent width use the smallest existing setting that
can express the continuous targets. If it fails, width doubles only until the
continuous gate passes. Patch duration is screened on the continuous model,
starting from the existing 2 s grid and testing 1 s only when temporal averaging
is the diagnosed failure; 0.5 s is a later diagnostic, not a default row.

The VQ family is EMA-VQ first. Codebook size starts at the retained K128
reference. If support is persistently redundant, K64 is the only next reduction;
larger K or another quantizer family is considered only when a healthy K128 loses
required information. There is no simultaneous K x D x quantizer search.

### Loss ladder

The default `B1-SO` objective contains only:

```text
L = L_observation_reconstruction + L_source_trajectory
```

`Q-S/Q-O` add only the corresponding VQ commitment/update term. No prototype,
context, balance, independence, cross-masking, or coupling loss is enabled by
default.

Additional terms are one-factor diagnostics with a named trigger:

| Trigger | Single allowed diagnostic | Promotion condition |
| --- | --- | --- |
| calibrated teacher uncertainty passes `T-G4` | uniform source loss vs clipped, normalized precision weighting | improves source score/calibration without worse observation fidelity or effective support |
| actual code collapse under a passing continuous model | existing straight-through balance loss | restores health without exceeding source/observation non-inferiority margins |
| continuous semantics pass but hard-token semantics fail | isolated prototype/topology loss | improves hard retention without codebook redundancy or gradient conflict |
| a valid local sequence endpoint fails while local targets pass | isolated context loss | improves the frozen sequence endpoint without future leakage |

The old multi-entry loss bundle is not restored. Every new entrance has its own
coordinates, masks, weight, ablation, and gradient audit.

### Tokenizer endpoints and gates

| Gate | Required evidence |
| --- | --- |
| `B-G0 support` | train loss and evaluation use the same declared target/mask population; subject/trial/patch coverage is explicit |
| `B-G1 observation` | held-out masked measurement log score/NRMSE is non-inferior to `B0-O`; EEG spectral and fNIRS HbO/HbR morphology are secondary fidelity checks |
| `B-G2 source` | continuous source latent/decoder retains the frozen teacher trajectory beyond history and target-permutation baselines, for every required modality/coordinate |
| `B-G3 attribution` | source-only, observation-only, and full interventions show that source gain is not supplied by an observation/residual bypass; cross-decoding is reported, not forced to zero |
| `Q-G1 retention` | expected embedding, posterior, and hard ID are each compared with the continuous upper bound; hard-token source and observation losses stay within frozen margins |
| `Q-G2 health` | active/effective support, dead/revival history, minimum per-code support, usage concentration, participation rank, near-duplicates, and subject/seed stability pass as guardrails |

Codebook utilization is not itself a semantic endpoint. Among gate-passing
models, the lowest bitrate wins; a higher occupancy count cannot rescue worse
reconstruction or teacher retention. Continuous latents, expected embeddings,
posteriors, hard IDs, and codebook embeddings are all exported so hard IDs never
become the entire representation record.

## C: coupling-prior return

### Frozen evaluation target

Coupling is tested only after the marginal tokenizer is frozen. The primary
representation-level estimand is the subject-equal held-out proper-score
increment for measured fNIRS observation residual:

```text
q0(Y_F(t+h) | H_F_observation, H_F_source, phase/time/systemic controls)
q1(Y_F(t+h) | H_F_observation, H_F_source,
                 H_E_source, phase/time/systemic controls)

Delta_coupling = score(q1) - score(q0)
```

The evaluator is low-capacity and cross-fitted. Positive lag means EEG precedes
the fNIRS endpoint. One primary horizon or integrated horizon score is frozen
before confirmation; individual lag curves are descriptive and family-wise
controlled. Full-window tokens can support only an offline association label. A
prospective/delayed-prediction claim additionally requires strict receptive-field
cutoff tests.

Required nulls preserve the relevant marginals and dependence structure:

- whole-window circular shift with tokens and masks shifted together;
- same-subject/condition nonoverlapping trial derangement;
- independent-window pairing;
- lag reversal/negative-lag control;
- spatial adjacency permutation for a spatial-prior diagnostic.

NMI, co-occurrence heatmaps, row entropy, and same numeric IDs are descriptive
only. The teacher's latent flow is an upper-bound diagnostic, not the primary
coupling target.

### Minimal prior ladder

| ID | Tokenizer gradient | Purpose |
| --- | --- | --- |
| `C0` | none; fit a lag-balanced, marginal-residualized `q0/q1` after tokenizer freeze | establish whether the representation contains incremental information at all |
| `C1-source` | a small fit-selected weight reaches only the EEG source path; fNIRS target/history, both observation paths, teacher, and baseline are detached | test whether the one-term shaper preserves coupling-relevant source information |
| `C2-uncertainty` | same as `C1`, with clipped normalized confidence weights | optional only after `T-G4`; unweighted results remain co-primary sensitivity |

If `C0` does not beat `q0` and all registered nulls, no coupling loss reaches the
tokenizer. `C1/C2` are admitted only when `Delta_coupling` improves and all
observation/source fidelity and codebook-health gates remain non-inferior to the
coupling-free tokenizer.

Only the historical lag-balanced conditional pair likelihood is eligible to
return initially, because its training target matches the evaluation contrast.
The former lag-focus entropy, joint-entropy, codebook-neighbor JS, local/context
residual maps, and multi-term coupling bundle remain diagnostics. They may make a
coupling tensor look concentrated without improving held-out information and are
not reintroduced together. Best and final checkpoints, per-loss gradient norms,
reconstruction-versus-coupling cosine conflict, and assignment health are all
reported.

## Side-path experiments without workflow sprawl

A side path is a diagnostic child of a main run, not a new project track. It
shares the parent's data/split/teacher/code identities and lives at:

```text
experiments/runs/physiology_semantic_tokenizer/tokenizer_discovery_v1/
  <immutable-run-id>/
    resolved_config.yaml
    summary.json
    metrics.csv
    figures/
    diagnostics/
      <probe-id>/
```

Each diagnostic records `parent_run_id`, `scope`, `hypothesis`, `estimand_id`,
`operator/null`, `status`, and `decision_eligibility=false`. It may use
`synthetic`, `diagnostic`, `null`, or `development` scope. It cannot change the
parent summary, reuse a protected unlock, or promote a candidate. A diagnostic
that motivates a new main hypothesis requires a new contract version before
fresh confirmation data are viewed.

`research_state/registry.json` records only suite/program state transitions. It
does not gain one record per probe, seed, channel arm, or gate. The retained
result index is updated only when a conclusion and its minimum provenance package
are frozen.

## Code ownership for later implementation

No scaffolding is created by this design. When implementation starts, reuse the
existing owners:

| Responsibility | Owner |
| --- | --- |
| continuous teacher and family adapter | `src/teachers/` |
| target artifact, masks, joins, and provenance | `src/data/` |
| modality-local source/observation tokenizer | `src/tokenizers/` |
| reconstruction, semantic, VQ, and optional coupling objectives | `src/losses/` |
| proper scores, calibration, retention, and codebook health | `src/metrics/` and `src/analysis/` |
| one orchestration/analysis entry | `experiments/scripts/` |
| reviewed executable contract, when ready | `experiments/configs/physiology_semantic_tokenizer/` |

Do not reactivate or rename an E0--E2/R-series YAML, archived source/observation
runner, or old coupling suite. There is no need for a manager, plugin layer,
parallel results root, or separate authorization file.

## Unresolved before measured qualification or confirmation

The synthetic P0 contract and the bounded measured diagnostic contract are
executable. The following values remain unresolved for measured qualification
or confirmation and do not change the diagnostic's exploratory status:

1. the exact nonprotected dataset and subject/record split providing a genuinely
   fresh confirmation set beyond the registered development diagnostic;
2. the source-frozen soft priors, fixed versus free parameter list, parameter
   identifiability/SBC criteria, and numerical `r(t)` or physiological-state
   perturbation limits for `T-G1`--`T-G3`;
3. numeric `delta_S`, `delta_O`, and `delta_H` margins plus the single primary
   coupling horizon or integrated horizon definition and its
   family-wise null procedure;
4. maximum training steps/checkpoint rule and the measured-run compute budget;
5. the measured-data continuous target schema/version implementing the named shared
   driver, physiological states, nuisance/residual, trajectory, parameter
   summary, identifiability, physical-check, and variance fields above;
6. any measured-data corruption/masking schedule and clean-reference definition.

The `T3a-balloon-robust` P0 implementation, frozen synthetic generator,
corruption/null schedule, common output tables, gates, and Chinese renderer now
live in
[`t3a_balloon_robust_p0.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml),
[`evaluate_t3a_balloon_robust_p0.py`](../experiments/evaluate_t3a_balloon_robust_p0.py),
and
[`render_t3a_balloon_robust_p0.py`](../experiments/scripts/render_t3a_balloon_robust_p0.py).
The bounded measured reconstruction/null diagnostic is registered in
[`t3_measured_reconstruction_null_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml)
and
[`evaluate_t3_measured_reconstruction_null.py`](../experiments/evaluate_t3_measured_reconstruction_null.py).
It uses the canonical measured loader with `raw_with_ocular_artifact`, the
01--18 fit / 19--23 population pure-apply split, and declared independent,
pairing, and time-shift nulls. The measured non-circular time-shift comparison
scores the paired and shifted targets only on their common finite support; its
100-point support is not pooled with the 200-point independent/pairing nulls.
Its result is a nonprotected exploratory
diagnostic and is not a Croce/Balloon qualification, clean-ground-truth claim,
or protected evaluation. `T3b`, `T3c`, and `T5` enter only after their declared
triggers.

The plan's second-step fit-only identifiability suite is registered separately
in
[`t3_identifiability_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3_identifiability_v1.yaml)
and
[`evaluate_t3_identifiability.py`](../experiments/evaluate_t3_identifiability.py).
It freezes likelihood-only M2 (`beta`, `kappa`, `tau`) diagnostics at 16
transformed-space starts, a true one-parameter profile that reoptimizes both
companion parameters and latent states, 25% transformed-bound expansion, and
a six-raw-parameter conditional forward sensitivity SVD. One noisy
known-truth clean-scenario synthetic case must complete before the loader is
called. The measured arm fits its observation gauge and M0 selection score on
01--18 only, then analyzes the low/median/high representatives' eight fit
trials. The shared loader constructs canonical dataset-index metadata and
window references, but it never loads arrays or materializes window samples
for 19--23 validation and 24--29 protected subjects. This suite is exploratory
and cannot change qualification, promotion, or protected-data state.

The plan's third-step three-session LOSO diagnostic is registered in
[`t3_multisession_loso_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml)
and
[`evaluate_t3_multisession_loso.py`](../experiments/evaluate_t3_multisession_loso.py).
It uses only subjects 01--18, MA trials, and cache records
`session_01/03/05`; each fold fits two complete sessions and applies frozen
objects to the third. The common safe window is `[-5,+25) s`, with fNIRS
masked from task onset and the primary score restricted to the 15-second
nominal recovery envelope `[+10,+25) s`. Because event durations are absent,
the endpoint is not labelled an exact annotated rest period. Only effective
`kappa` varies: the two training-session estimates define a zero-sum log
session deviation and a geometric subject center for held-out apply. All other
physiological parameters remain fixed. This diagnostic cannot load 19--29
arrays or alter qualification, promotion, or protected-data state.

The plan's fourth-step hierarchy begins with the array-free admission contract
[`t3c_hierarchical_composite_admission_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml)
and
[`evaluate_t3c_hierarchical_composite_admission.py`](../experiments/evaluate_t3c_hierarchical_composite_admission.py).
It freezes the analytic `G_f/T_f/zeta_f/T_v` coordinate and a diagonal
one/two-dimensional Normal hierarchy, then checks the frozen Step 2/3 evidence
before any new measured metadata or array access. At the 2026-09-03 v3
admission snapshot the result was `BLOCKED_PREREQUISITE`: `T-P2`, a common
gauge, a prospective fixed endpoint, composite SBC/profile/multistart evidence,
and a pre-measured practical margin were not yet available. Consequently no
measured hierarchical fit was registered or authorized by this entry.

The follow-up synthetic `T-P2` composite screen is registered in
[`t3c_composite_synthetic_t2_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml)
and
[`evaluate_t3c_composite_synthetic_t2.py`](../experiments/evaluate_t3c_composite_synthetic_t2.py).
Its formal run uses 60 independent known-truth replicates for each
one-dimensional direction (`C1_G`, `C1_T`), independently reset training and
held-out trials, and a fitter boundary containing noisy training observations
but no realized truth, driver, generation seed, or held-out array. Both C1
directions failed their registered gates, so the run decision is
`BLOCKED_C1_COMPOSITE_IDENTIFIABILITY` and `C2_GT` was not run. The detailed
result is retained in the
[`T-P2` report](analysis/20260903_T3C_COMPOSITE_SYNTHETIC_TP2_REPORT.md).
This synthetic evidence is not qualification evidence and does not authorize
measured hierarchical fitting.

## Historical lifecycle boundary

The following table remains a lifecycle overlay for the superseded flow. It does
not rewrite dated evidence; linked reports remain the evidence owners.

| Historical item | Lifecycle | Evidence or retained plan | Retained use |
| --- | --- | --- | --- |
| E0--E2 and R0--R2 generations | **stopped** | [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) and [`20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md) | Historical results and failure boundaries only |
| SSM reliability screen | **stopped** | [`20260819 SSM reconstruction reliability results`](analysis/20260819_SSM_RECONSTRUCTION_RELIABILITY_RESULTS.md) | Exploratory reliability evidence only |
| Continuous-latent screen | **stopped** | [`20260819 continuous shared/private latent results`](analysis/20260819_CONTINUOUS_SHARED_PRIVATE_LATENT_RESULTS.md) | Exploratory latent evidence only |
| LC-SPVQ optimization and QC | **stopped** | Dated LC-SPVQ reports under [`analysis/`](analysis/) | Negative/undetermined evidence only |
| Token Atlas Core (T0) | **stopped** | [`TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) | Development-only retained result |
| Protected comparison campaign and P0 degradation | **stopped** | [`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md) and [`PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md) | Retained comparison evidence only |
| Croce legacy solver and audits | **stopped** | [`CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Historical qualification/audit evidence only |
| Comparison P1/P2 and unexecuted follow-up | **abandoned** | [`PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md) | Unstarted comparison candidates only |
| D1B, future R/VQ, LC full development, observation/source map, Atlas Statistical/Full, and Croce follow-ons | **abandoned** | Dated plans and candidate snapshots indexed in [`README.md`](README.md) | Non-runnable historical candidates only |

Neither `stopped` nor `abandoned` evidence authorizes or determines a row in
`PST-DISCOVERY-v1`. Historical plans preserve their original wording for
reproducibility.
