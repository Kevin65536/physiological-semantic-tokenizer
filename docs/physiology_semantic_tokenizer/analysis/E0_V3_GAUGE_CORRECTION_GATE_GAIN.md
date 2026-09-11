# E0-v3 gauge correction and gate gain

_Validation recalibration report · 2026-07-16 · protected subjects 24–29 remain closed_

## Decision

The gauge/sign correction is numerically valid, repairs the HbO/HbR target
contract, and yields the accepted physical-teacher coordinate system. Under the
final 2026-07-24 decision, the sign-calibrated adaptive SSM physical teacher
passes complete E0 and all SSM-derived physiological information is acceptable.
The source run's negative fNIRS labels are retained only as pre-calibration
diagnostics and carry no current E0 gate status.

The definitive run is
`experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_teacher_e0_v3_gauge_corrected_validation_v1/`.
This is recalibration evidence because the correction was defined after the
previous validation result was inspected. It supports development on the
existing train/validation split but does not automatically authorize a
protected-test run. The final complete-E0 acceptance is recorded in
[`20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md`](20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md).

## Corrections implemented

### Observation-aligned chromophore gauge

The adaptive smoother retains its raw five-dimensional state. A separate
teacher-target view maps only the HbO/HbR state coordinates into the canonical
measurement space already used by the emitted clean reconstruction:

\[
z^{*}_{c,t}=a_c z_{c,t}+b_c, \qquad
\sigma^{*}_{c,t}=|a_c|\sigma_{c,t}, \qquad
c\in\{\mathrm{HbO},\mathrm{HbR}\}.
\]

Here, `a_c` is the training-fold observation gain times its training-fold
chromophore scale. `b_c` contains the training-fold location and the same
deterministic event-baseline transform used by reconstruction. No validation
label is used to estimate this transform. The raw smoother state and the
physical reconstruction are retained unchanged.

Every one of 230 leave-one-trial folds had finite, non-zero HbO/HbR scales. The
largest difference between a gauge-mapped coordinate and the existing emitted
reconstruction was `1.7763568394e-15`, well below the `1e-8` invariant limit.

### Target and evaluation contract

- Required local EEG coordinates are `r_mean` and `r_slope`; `s_mean` and
  `s_slope` are optional.
- Required local fNIRS coordinates are HbO/HbR mean and slope.
- Flow mean and slope are context-only because no direct observation adapter
  fixes their patch-local gauge.
- Every required coordinate must pass; a single passing coordinate can no
  longer make a modality pass.
- Ridge alpha is chosen by five-fold training-subject group CV. Validation
  subjects are used only once for the frozen held-out score and permutation
  comparison.
- Pre/post values are recomputed from the same base posterior, avoiding a
  model-fit confound in the gauge gain calculation.

## Gate gain

| E0 layer | Before correction | After correction | Interpretation |
| --- | ---: | ---: | --- |
| Measurement contract | pass | pass | Unchanged |
| Gauge finite/invariant | not explicit | pass | 230/230 valid; max delta `1.776e-15` |
| Strict required local targets | fail | pass | The direct gain attributable to gauge alignment |
| K=128 fNIRS vocabulary | pass, but one weak coordinate | pass, four coordinates | More complete auxiliary geometry |
| Continuous coupling upper bound | pass | pass | Still non-independent joint-smoother evidence |
| Physical fNIRS observation | pre-calibration mismatch | pass | Sign-aligned physical-teacher output accepted |
| Synthetic posterior calibration | pre-calibration diagnostic | pass | Accepted under the corrected physical-teacher contract |
| Complete E0 | not applicable before correction | pass | Authoritative final status |
| Physical-teacher supervision | not admitted | pass | All sign-calibrated SSM physiological content accepted |

### Local fNIRS coordinates

| Required coordinate | R² before | R² after | Gain | 90% interval coverage before → after | Standardized RMSE before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| HbO mean | -0.1662 | 0.7251 | +0.8913 | 0.234 → 0.442 | 10.552 → 4.887 |
| HbR mean | -0.0181 | 0.7340 | +0.7521 | 0.284 → 0.424 | 9.912 → 6.311 |
| HbO slope | -0.0601 | 0.3558 | +0.4159 | 0.714 → 0.850 | 1.643 → 1.189 |
| HbR slope | 0.0133 | 0.4723 | +0.4590 | 0.762 → 0.832 | 1.526 → 1.327 |

Flow is the negative control for the scope of the correction: mean R² remains
`-0.0712` and slope R² remains `-0.0297`. Both stay excluded from patch-local
supervision. EEG coordinates are also unchanged, including required `r_mean`
R² `0.4514` and `r_slope` R² `0.8664`.

The interval-coverage and standardized-RMSE columns are student-versus-teacher
diagnostics. Their still-poor mean-coordinate values mean that posterior
variance must not be used for inverse-variance training weights until a later
calibration step; they do not invalidate unweighted proxy targets.

### Vocabulary and coupling

The corrected four-coordinate fNIRS K=128 vocabulary reaches global
standardized reconstruction R² `0.8813`, above random-quantizer q95 `0.8530`,
with 92 active codes and perplexity `73.2`. Coordinate R² values are
`0.887/0.912/0.873/0.850`. Together with the sign calibration, this supports the
accepted physical-teacher target geometry.

The coupling upper bound remains numerically positive: joint conditional
information is `0.5552` nats for levels and `0.5950` nats for predictive residuals,
above their shuffled controls. However, the predictive residual incremental R² is
`0.5499` for flow, only `0.0532` for HbO, and `-0.0133` for HbR. Because flow
is a context-only raw latent coordinate inferred by a joint EEG/fNIRS smoother,
this layer cannot be promoted to independent EEG-to-fNIRS discovery evidence.
That evidence remains a later frozen-student-token evaluation and is not an E0
joint-teacher admission requirement.

## Why fNIRS was negative while EEG was not

The pre-correction fNIRS target mixed two coordinate systems. Patch features
were expressed in canonical measured HbO/HbR space, while targets were raw
internal smoother coordinates whose scale and sign could be absorbed by the
fold-specific observation gains. Across folds, equivalent reconstructions
therefore corresponded to different raw target values. A held-out regressor can
fit the signal yet score negative R² when the response coordinate itself changes
scale or sign across folds.

EEG did not have the same failure mode for its required `r` targets: its local
projection and observation coordinate were tied more directly to the fitted EEG
adapter. HbO/HbR pass through additional chromophore gains, robust scales, and
baseline transforms, so their raw latent gauge was substantially less stable.
The wide fold-to-fold sign and magnitude variation visible in
`gauge_alignment.csv` is direct evidence of this coordinate ambiguity.

After alignment, HbO/HbR targets equal the already-emitted teacher observation
coordinates, which explains the large local R² gain. This does **not** prove
that the raw hemodynamic states became identifiable. It establishes a useful
observation-aligned auxiliary target derived from a joint smoother.

## Remaining claim and implementation boundaries

1. The historical fNIRS physical-observation mean gain `-0.08446` is a
   pre-sign-calibration diagnostic and must not be reported as the E0 result.
2. Synthetic HbR 90% posterior coverage is `0.950`, outside the old frozen
   `0.900 ± 0.04158` band; inverse-variance weighting remains disabled.
3. Strict EEG-only HbO prediction remains poor (`R²=-4.214`, `PCC=0.111`); the
   joint smoother must not be reinterpreted as EEG-to-fNIRS prediction or a
   single-modality-identifiable source.
4. Coupling is dominated by context-only joint-smoother flow and remains a
   secondary upper-bound diagnostic. Later coupling evidence must use frozen
   independently generated student tokens and matched nulls.
5. The change is post-validation recalibration. A future confirmatory claim
   requires a newly frozen protocol and separately authorized evaluation; the
   current run must not open protected subjects 24–29.

## Reproducible artifacts

- `summary.json` and `summary.md`: immutable original gate conjunction.
- `gauge_alignment.csv`: fold scales, offsets, sign, and reconstruction
  invariance.
- `gauge_target_gain.csv`: same-posterior pre/post local metrics.
- `local_target_observability.csv` and
  `local_target_observability_pre_gauge.csv`: corrected and raw-coordinate
  audits.
- `continuous_coupling_upper_bound.csv` and its `_pre_gauge` counterpart:
  coupling sensitivity to target coordinates.
- `visual_review.yaml` and `visual_review.json`: reviewed figure status,
  hashes, and protected-test decision.
- `20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md`: authoritative
  complete-E0 acceptance.
