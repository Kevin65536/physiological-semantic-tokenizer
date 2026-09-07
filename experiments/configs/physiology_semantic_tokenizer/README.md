# Physiology-semantic tokenizer configurations

This directory contains the active synthetic T3a P0 contract, synthetic
T3c composite T-P2 and Step5A consistency diagnostics, three bounded nonprotected measured diagnostics, one
array-free T3c admission contract, and retained contracts from the stopped
R-series. Generated variants, local tuning files, run outputs and abandoned
preregistries are intentionally excluded from Git.

> **Historical configuration snapshot (2026-09-03): one synthetic qualification contract, one
> synthetic composite identifiability screen, three measured exploratory
> diagnostics, and one array-free admission gate.**
> `t3a_balloon_robust_p0.yaml` remains the
> synthetic qualification path. `t3_measured_reconstruction_null_v1.yaml`,
> `t3_identifiability_v1.yaml`, and `t3_multisession_loso_v1.yaml` can read only
> their declared nonprotected development scope and are decision-ineligible.
> `t3c_hierarchical_composite_admission_v1.yaml` cannot read measured arrays
> and currently blocks the measured hierarchy.
> `t3c_composite_synthetic_t2_v1.yaml` is synthetic-only; its formal result
> blocks both C1 directions and leaves C2 unrun. No entry can open protected
> data. The retained E0–E2 and R-series configurations remain stopped replay
> surfaces.

## Retained historical boundary

The E0–E2 and failed post-R2 development YAMLs are historical and have been
moved out of this config surface. They are stopped records; new work must not
infer a forward contract from them. The R-series stopped at
`do_not_enter_r2_p`; no VQ or token co-occurrence continuation is authorized by
these records.

The theory/architecture principles are frozen in
[`METHOD_RATIONALE.md`](../../../docs/METHOD_RATIONALE.md). The
observation–source v2 map is a pre-freeze implementation snapshot only; no
executable tokenizer-training YAML exists yet. The
historical YAML/runtime surface is not a template for a new candidate. Do not
reuse an old YAML by changing its experiment ID, target, or output path. The
replaceable design note is
[`observation_source_exploration_v2.json`](../../../docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json).

No architecture or matrix is fixed by that note; it does not override the
current frozen principles. Any implementation proposed for a development
comparison must first pass synthetic software, target/teacher,
tensor-shape, split, and null checks. Protected cohorts require the owning
protocol plus separate, explicit user authorization and are never opened by a
config flag.

The version-controlled configuration surface is deliberately limited to:

| File | Purpose | Status |
| --- | --- | --- |
| `t3a_balloon_robust_p0.yaml` | Synthetic physics, identifiability, corruption, null, calibration and visualization contract | **Active synthetic P0**; measured/protected inputs disabled |
| `t3_measured_reconstruction_null_v1.yaml` | Real EEG/fNIRS center-mask reconstruction and independent/pairing/time-shift null diagnostic | **Active measured development diagnostic**; subjects 01–23 only, protected 24–29 closed, not qualification evidence |
| `t3_identifiability_v1.yaml` | Known-truth synthetic plus fit-only M2 multistart/profile/bound/SVD diagnostic | **Active measured development diagnostic**; gauges/selects on 01–18, analyzes three fit representatives, loads no subject 19–29 arrays/window samples, not qualification evidence |
| `t3_multisession_loso_v1.yaml` | Fit-only three-session LOSO with a shared effective-κ center, zero-sum training-session deviations, and target-masked nominal recovery scoring | **Active measured development diagnostic**; subjects 01–18 and records 01/03/05 only, loads no subject 19–29 arrays, not qualification evidence |
| `t3c_hierarchical_composite_admission_v1.yaml` | Array-free gain/time composite map, shrinkage smoke, and Step 2/3 prerequisite audit | **Active admission gate**; current result `BLOCKED_PREREQUISITE`, measured hierarchy not started |
| `t3c_composite_synthetic_t2_v1.yaml` | Known-truth C1 gain/time SBC, profile, multistart, confounding, SVD, and held-out screen | **Complete synthetic negative**; both C1 directions failed, C2 not run, measured hierarchy remains blocked |
| `step5a_inference_consistency_v1.yaml` | Matched discrete-model calibration diagnostics, known-driver likelihood, short particle reference, and mismatch stress | Synthetic localization only; [protocol](../../../docs/EXPERIMENT_PLAN.md#step5a0-inference-consistency-diagnostic), no teacher admission |
| `step5_v1.yaml` | Joint-likelihood calibration, independent held-out minimal teachers, state uncertainty, sensitivity and cross-modal nulls | [Staged protocol](../../../docs/EXPERIMENT_PLAN.md#full-step5-staged-continuation); measured/UQ depend on core teacher qualification; no protected access |
| `step5b_v1.yaml` / `step5b_v2.yaml` | Native trial masking, train-only projection and noise, W parameter mixture, subject-cluster nulls; v2 retains support while resolving boundary mass and training channel eligibility | Pins the synthetic contract; subjects 01–18 only, new trials within existing sessions; v1 failures remain historical evidence |
| `r0p_raw_lag_baseline.yaml` | Preregistered raw EEG–fNIRS lag benchmark | **Stopped**; completed, primary result negative |
| `r1p_population_frozen_teacher.yaml` | Fit on subjects 01–18 and pure-apply on 19–23 | **Stopped**; completed, structural audit passed |
| `r1p_teacher_qualification_registry.json` | Frozen G1–G6 gate definitions | **Stopped**; formal-v3 did not qualify |
| `r1p_teacher_perturbation_registry.json` | Three finite train-only G4 stress bundles | **Stopped**; completed |
| `r2d_continuous_observability.yaml` | One-seed development continuous observability | **Stopped**; completed, bilateral endpoint failed |
| `token_physiology_atlas.yaml` | Versioned descriptive analysis contract for an already trained tokenizer | **Stopped** development-only replay; does not authorize a new VQ or coupling experiment |

Two matching evidence contracts live under
`docs/physiology_semantic_tokenizer/architecture/`:
`r0p_raw_lag_baseline_preregistry.json` and
`r1p_prevalidation_seal.json`. They are required to replay the corresponding
formal scripts and must not be edited retrospectively.

## Retired execution contract

The following requirements describe the stopped R-series replay surface. They do
not authorize a new run or define the clean flow's future contract. Any replacement
must carry its own parser test, shape and split assertions, synthetic path, and
output namespace below:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<run>/
```

The retained R-series snapshot used subjects 01–18 as the development-fit cohort,
subjects 19–23 as development pure-apply, and subjects 24–29 as protected. A
config flag could not relax that boundary. Teacher-supervised replay requires the
exact registry and seal identities; teacher-free replay sets all teacher-derived
loss weights to zero.

The consolidated methods, results, interpretation and stop decision are in
[`20260728_R_SERIES_EXPERIMENT_REPORT.md`](../../../docs/physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).

_Last updated: 2026-09-07_
