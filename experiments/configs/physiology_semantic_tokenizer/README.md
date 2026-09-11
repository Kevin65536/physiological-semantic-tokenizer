# Physiology-semantic tokenizer configurations

This directory contains reviewed SSM, T3 and Step5 diagnostic contracts and
retained R-series contracts. The
[entrypoint/config/test map](../../README.md#entrypoint-config-and-test-map)
owns their command and test correspondence. Execution and scientific verdicts
come from the [project registry view](../../../docs/PROJECT_STATUS.md).
Generated variants, local tuning files, run outputs and abandoned preregistries
are intentionally excluded from Git.

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

| File | Purpose | Contract / reading boundary |
| --- | --- | --- |
| `ssm_overnight_v3.yaml` | Observation-layer and missing-feature comparisons with independent adaptation panels | [Observation protocol](../../../docs/EXPERIMENT_PLAN.md#观测合同修复后的下一轮实验设计2026-09-10); uses `docs/EXPERIMENT_PLAN.md` as its design owner |
| `ssm_overnight_v1.yaml` / `ssm_overnight_v2.yaml` | Retained N1–N6 / N1–N7 diagnostic panels | [Retained v2 protocol](../../../docs/EXPERIMENT_PLAN.md#bounded-overnight-ssm-diagnostics-retained-v2-contract); their literal `plan: ssm_next.md` records the removed short-term note and is not a live documentation link |
| `t3a_balloon_robust_p0.yaml` | Synthetic physics, identifiability, corruption, null, calibration and visualization contract | Synthetic qualification contract; measured/protected inputs disabled |
| `t3_measured_reconstruction_null_v1.yaml` | Real EEG/fNIRS center-mask reconstruction and independent/pairing/time-shift null diagnostic | Exploratory subjects 01–23; protected 24–29 closed |
| `t3_identifiability_v1.yaml` | Known-truth synthetic plus fit-only M2 multistart/profile/bound/SVD diagnostic | Fit-only subjects 01–18; no subject 19–29 arrays/window samples |
| `t3_multisession_loso_v1.yaml` | Fit-only three-session LOSO with a shared effective-κ center, zero-sum training-session deviations, and target-masked nominal recovery scoring | Fit-only subjects 01–18 and records 01/03/05; no subject 19–29 arrays |
| `t3c_hierarchical_composite_admission_v1.yaml` | Array-free gain/time composite map, shrinkage smoke, and Step 2/3 prerequisite audit | Array-free prerequisite checks; no measured hierarchy launcher |
| `t3c_composite_synthetic_t2_v1.yaml` | Known-truth C1 gain/time SBC, profile, multistart, confounding, SVD, and held-out screen | Synthetic-only composite screen; [dated result interpretation](../../../docs/analysis/20260903_T3C_COMPOSITE_SYNTHETIC_TP2_REPORT.md) |
| `step5a_inference_consistency_v1.yaml` | Matched discrete-model calibration diagnostics, known-driver likelihood, short particle reference, and mismatch stress | Synthetic localization only; [protocol](../../../docs/EXPERIMENT_PLAN.md#step5a0-inference-consistency-diagnostic), no teacher admission |
| `step5_v1.yaml` | Joint-likelihood calibration, independent held-out minimal teachers, state uncertainty, sensitivity and cross-modal nulls | [Staged protocol](../../../docs/EXPERIMENT_PLAN.md#full-step5-staged-continuation); measured/UQ depend on core teacher qualification; no protected access |
| `step5b_v1.yaml` / `step5b_v2.yaml` | Native trial masking, train-only projection and noise, W parameter mixture, subject-cluster nulls; v2 retains support while resolving boundary mass and training channel eligibility | Pins the synthetic contract; subjects 01–18 only, new trials within existing sessions; v1 failures remain historical evidence |
| `step5_observation_diagnostic_v1.yaml` | Observation coordinates, joint/single-modality likelihood curves, bridge and nested linear controls | [Observation diagnostic protocol](../../../docs/EXPERIMENT_PLAN.md#step5-observation-adaptation-diagnostic-retained-v1-contract) |
| `step5_observation_repair_v1.yaml` | Scale invariance, temporal reference and paired replay | [Repair v1 protocol](../../../docs/EXPERIMENT_PLAN.md#step5-observation-contract-repair-and-regression-retained-v1) |
| `step5_observation_repair_v2.yaml` | Flow-domain diagnosis and mask-specific feature bridge | [Mask-bridge protocol](../../../docs/EXPERIMENT_PLAN.md#step5-flow-domain-and-mask-specific-observation-experiment) |
| `r0p_raw_lag_baseline.yaml` | Preregistered raw EEG–fNIRS lag benchmark | Retained R0-P preregistration |
| `r1p_population_frozen_teacher.yaml` | Fit on subjects 01–18 and pure-apply on 19–23 | Retained R1-P population fit/apply contract |
| `r1p_teacher_qualification_registry.json` | Frozen G1–G6 gate definitions | Retained R1-P G1–G6 qualification definitions |
| `r1p_teacher_perturbation_registry.json` | Three finite train-only G4 stress bundles | Retained R1-P perturbation panel |
| `r2d_continuous_observability.yaml` | One-seed development continuous observability | Retained R2-D observability contract |
| `token_physiology_atlas.yaml` | Versioned descriptive analysis contract for an already trained tokenizer | Retained Atlas replay; no new VQ or coupling experiment |

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

Short-term planning notes are not configuration owners. Keep frozen YAML content
unchanged; use the stable protocol sections above when navigating retained
versions. New contracts name a maintained protocol rather than a scratch file.
