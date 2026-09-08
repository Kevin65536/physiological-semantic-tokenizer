# Retained result index

_Evidence surface updated 2026-09-08; payload-pruning record retained from
2026-07-30_

This index identifies the experiment material kept for routine reading,
comparison, and paper review. A retained result is defined by its
conclusion and provenance package, not by keeping every intermediate tensor or
checkpoint.

This file owns retention and evidence locations, not current project state. Status
terms in its tables describe the retained artifact at its recorded snapshot. Use the
generated [`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) for current execution and
scientific verdicts.

Some run and checkpoint paths below point to local Git-ignored artifacts that remain
useful in this workspace but are not part of the tracked paper record. For manuscript
Methods/Results, start with the linked human-readable reports and campaign tables;
use local payloads only when a report explicitly requires them.

## Main-method evidence

| Stage | Retained authority / artifact | Why it remains |
| --- | --- | --- |
| Step5 observation contract repair and regression | [`20260908_observation_repair_v1/summary.md`](runs/physiology_semantic_tokenizer/step5/20260908_observation_repair_v1/summary.md), [numerical compatibility and refined replay](runs/physiology_semantic_tokenizer/step5/20260908_observation_repair_v1/verification_review.json), [contract](configs/physiology_semantic_tokenizer/step5_observation_repair_v1.yaml) | log-domain extraction repair, known-scale invariance on reused and independent synthetic inputs, explicitly approximate temporal reference, identical-input failure continuation and same-fold fixed SSM/linear control; numerical corrections work, measured teacher remains unqualified |
| Step5 observation adaptation diagnostic | [`20260908_observation_diagnostic_v1/summary.md`](runs/physiology_semantic_tokenizer/step5/20260908_observation_diagnostic_v1/summary.md), [numerical review](runs/physiology_semantic_tokenizer/step5/20260908_observation_diagnostic_v1/diagnostic_review.json), [contract](configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml) | controlled preprocessing bridge, original-training-only modality curves/residuals, nested-CV lag control and retained saturation failures; exploratory, no teacher admission |
| E0 final teacher revalidation (stopped) | [`20260723_adaptive_teacher_e0_v3_line_clean_v4_revalidation_v1`](runs/physiology_semantic_tokenizer/e0_teacher_validity/20260723_adaptive_teacher_e0_v3_line_clean_v4_revalidation_v1/summary.md) | retained development teacher surface and claim boundary |
| E1 K128 health (stopped) | [`20260722_e1_health_coupling_visual_report_v1`](runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/summary.json) and retained multi-seed summaries | software/occupancy reference |
| E2 weight calibration (stopped) | [`20260723_e2_v4_training_gradient_weight_calibration_v1`](runs/physiology_semantic_tokenizer/e2_weight_calibration/20260723_e2_v4_training_gradient_weight_calibration_v1/analysis/summary.md) | explains frozen semantic-objective scale |
| E2 final decision (stopped) | [`20260723_e2_v4_semantic_objective_suite_v1/decision`](runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/decision/summary.md) | no semantic row admitted; T0 retained |
| R0-P (stopped) | [`20260728_r1p_raw_lag_formal_v1`](runs/physiology_semantic_tokenizer/r0p_raw_lag_baseline/20260728_r1p_raw_lag_formal_v1/summary.json) | registered raw-lag negative result |
| R1-D (stopped) | [`20260728_e0_v3_reanalysis_v1`](runs/physiology_semantic_tokenizer/r1d_teacher_geometry/20260728_e0_v3_reanalysis_v1/summary.json) | exploratory correction geometry |
| R1-P structure (stopped) | [`20260728_r1p_bundle_qualification_structure_v2`](runs/physiology_semantic_tokenizer/r1p_structural_audit/20260728_r1p_bundle_qualification_structure_v2/AUDIT.md) | confirms formal bundle integrity |
| R1-P formal panel (stopped) | [`20260728_r1p_population_frozen_formal_v3`](runs/physiology_semantic_tokenizer/r1p_teacher_qualification/20260728_r1p_population_frozen_formal_v3/panel_summary.json) | G2 failure and final qualification decision |
| R1-P post-formal (stopped) | [`20260728_r1p_formal_v3_postformal_v1`](runs/physiology_semantic_tokenizer/r1p_cross_session_hemodynamic_adaptation/20260728_r1p_formal_v3_postformal_v1/summary.json) | failure interpretation without gate revision |
| D1B (abandoned) | [`20260728_d1b_train_only_grid_v1`](runs/physiology_semantic_tokenizer/r1p_d1b_train_only_hyperparameter_seal/20260728_d1b_train_only_grid_v1/summary.json) | incomplete train-only evidence; validation remained undetermined and the lane is not continued |
| R2-D formal (stopped) | [`20260728_r2d_cj_seed20260728_formal_v1`](runs/physiology_semantic_tokenizer/r2_continuous_observability/20260728_r2d_cj_seed20260728_formal_v1/summary.json) | bilateral observability failure |
| R2-D statistical audit (stopped) | [`20260728_r1d_cj_seed20260728_v2_stat_audit`](runs/physiology_semantic_tokenizer/r2_continuous_observability_analysis/20260728_r1d_cj_seed20260728_v2_stat_audit/diagnostic_summary.json) | uncertainty and diagnostic reference |
| T3 Step 2 identifiability (complete exploratory negative) | [`detailed report`](../docs/analysis/20260902_T3_IDENTIFIABILITY_STEP2_REPORT.md) and local [`v3 summary`](runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.md) | fit-only beta/kappa/tau practical-identifiability endpoint unsupported; no qualification or promotion |
| T3 Step 3 three-session LOSO (complete exploratory negative) | [`detailed report`](../docs/analysis/20260902_T3_MULTISESSION_LOSO_STEP3_REPORT.md) and local [`v2 summary`](runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.md) | fit-only effective-kappa candidate worsened held-out recovery NLL versus fixed M0; parameter screen failed; no trait, qualification, or promotion claim |
| T3c Step 4 hierarchical admission (complete blocked gate) | [`detailed report`](../docs/analysis/20260903_T3C_HIERARCHICAL_STEP4_ADMISSION_REPORT.md) and local [`v3 summary`](runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/summary.md) | array-free admission found 6/8 prerequisites unmet; measured partial pooling not started; no trait, qualification, or promotion claim |
| T3c Step 4 composite synthetic T-P2 (complete negative) | [`detailed report`](../docs/analysis/20260903_T3C_COMPOSITE_SYNTHETIC_TP2_REPORT.md) and local [`v1 summary`](runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/summary.md) | both C1 gain/time directions failed; C2 not run; synthetic identifiability evidence only, measured hierarchy remains blocked |
| Step5A0 inference consistency (complete localization) | Local [`panel report`](runs/physiology_semantic_tokenizer/step5a_inference_consistency/20260907_step5a_localization_v1/summary.md), [`same-data oracle refinement`](runs/physiology_semantic_tokenizer/step5a_inference_consistency/20260907_step5a_oracle_quadrature_refinement_v1/summary.json), and [`reference CDF figure`](runs/physiology_semantic_tokenizer/step5a_inference_consistency/20260907_step5a_localization_v1/reference_cdf.pdf) | parameterization/Kalman checks passed; resolved G/Z short references reveal score/posterior differences; W reference unresolved; no calibration or teacher admission claim |
| Step5 staged continuation | Local [`detailed stage review`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/stage_review.md), [`measured report`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/summary.md), and [`UQ prerequisite decision`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/uq_stage_report.md) | A0 limited calibration passed; A1 admitted W only; B did not qualify a measured teacher; comprehensive UQ was not executed |

### Step5A0 evidence snapshot — 2026-09-07

The Git evidence package retains the stage summaries, manifests, resolved
configurations, frozen runner/inference snapshots, failure diagnostics and
key figures in their original run paths. Per-case arrays, prepared measured
inputs and full numerical grids remain local generated payloads; references
to those payloads in manifests identify the local audit/replay surface.

The staged continuation adds a separate
[`Step5A0 joint-inference/calibration report`](runs/physiology_semantic_tokenizer/step5/20260907_a0_calibration_v1/summary.md)
and [`particle-reference comparison`](runs/physiology_semantic_tokenizer/step5/20260907_a0_joint_reference_v1/summary.json).
Its 180 fresh independent cases passed the frozen numerical and parameter
calibration screens. G/W/Z parameter 95% coverage was `93.33% / 95.00% / 95.00%`,
versus `81.67% / 88.33% / 80.00%` for the legacy marginal score on the same data.
The new joint-reference CDF differences were `0.00669 / 0.00908 / 0.01700`.
W's likelihood precision passed while its ancestry-diversity check remained
failed; no reference smoothed-state claim is made. Z's rank KS p was `0.04851`,
above the frozen `0.01` threshold but still a residual-calibration diagnostic.
This qualifies only the bounded synthetic inference screen. State teacher,
parameter-mixture, null, and measured qualifications are separate stages.

The [`A1 candidate-scope review`](runs/physiology_semantic_tokenizer/step5/20260907_a1_candidate_scope_review_v2/summary.md)
accounts for 240 registered cases: 231 complete and nine retained failures.
U1_W completed its own 60-case panel and passed state recovery, uncertainty,
stability and all eight matched-model shared-information increments. Its
driver NRMSE/correlation were `0.4640 / 0.8835`; r/clean EEG/HbO/HbR coverage
was `94.61% / 94.61% / 95.01% / 94.93%`. G/Z panels retained five/four failures;
U0 failed cross-parameter sensitivity. U3's three diagnostic grids remained
unresolved and were never selection-eligible. The review corrects the initial
aggregation's propagation of independent G/Z incompleteness to W, without
altering case metrics, thresholds or seeds. Prior snapshots remain retained.

The [`B output recovery`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/summary.md)
evaluated the admitted W candidate and fixed baseline on subjects 01–18,
sessions 01/03/05, with eight training and two held-out trials per session.
Masking precedes native information-propagating transforms; projections,
channel selection, noise scales and parameter weights use training only.
U1_W completed 106/108 held-out trials. Both candidates encountered physical
domain exceptions on subject_07/session_03 and subject_14/session_01; these
were retained. The 16 complete-subject summaries are descriptive and do not
replace the incomplete registered cohort.

For center EEG, W's joint-minus-own-context log score was `-0.9896`
(`95% CI [-1.1910, -0.7815]`). Center fNIRS improved slightly over own context
(`+0.0689 [0.0235, 0.1125]`) but lost to the training task-template control
(`-2.5864 [-3.4106, -1.7699]`) and did not beat pairing or shift nulls.
EEG/HbO/HbR NRMSE was `1.9554 / 1.1928 / 2.5521`; noisy-observation 95%
coverage was `74.87% / 26.95% / 36.46%`. These are not latent-state coverage
claims. Whole-modality predictions also lost to the all-missing prior.

All 18 W posterior grids resolved at CDF difference at most `0.01980`;
the three preselected 13/17-order checks differed by at most `0.000629`.
Nevertheless all posterior mass was effectively in the lower boundary band
(W means `-0.49973` to `-0.49849`). The 17 completed mixture-resolution checks
were numerically stable, but one designated trial failed; removing the
boundary band changed driver NRMSE by up to `0.2247`, exceeding `0.20`.
No individual physiological-parameter interpretation is granted.

The retained [`v1`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_v1/manifest.json)
and [`v2`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_v2/manifest.json)
record the coarse-grid/native-channel and diagnostic JSON-write failures.
V2 kept the full support while locally resolving posterior mass and selecting
only training-valid fNIRS pairs. Recovery reused 102 case outcomes with
digests and recomputed only subject_01's six missing outcomes; three retained
training-likelihood check points matched exactly. These continuations add
zero independent statistical replicates and do not erase earlier failures.

The [`synthetic/measured null figure`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/shared_information_diagnostic.pdf)
and [`full review`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/stage_review.md)
separate inference calibration from measured shared-information failure.
With no qualified measured core teacher, comprehensive UQ, ICC, conformal,
precision weighting and training exports were not executed. No replication
subjects 19–23 or protected subjects 24–29 signal arrays were read. The original
Step 1–4 implementation and evidence remain unchanged.

The following paragraphs retain the initial small-panel snapshot, before the
staged continuation above.

The [versioned configuration](configs/physiology_semantic_tokenizer/step5a_inference_consistency_v1.yaml)
ran four independent cases per G/W/Z axis, with separate known-driver oracle,
matched discrete-model generation, and Step 4 mismatch stress branches. The
12-case panel completed in 204 seconds with three workers. No measured or
protected data were read. Original Step 1–4 evidence and decisions were retained.

Forward trajectories agree exactly in the checked cases; state Jacobian and
GWZ chain-rule errors are below `1e-11`. In the linear Gaussian specialization,
the production smoother agrees with exact Kalman/RTS to `3.01e-14` in means and
`3.74e-16` in covariance. Its marginal score nevertheless differs from the
joint log likelihood by `-0.192924`. This is a check of the distinct statistical
objects, not evidence that the nonlinear approximation is calibrated.

For six-second nonlinear reference cases, G and Z pass the frozen particle
precision checks, while their EKF-score and particle-likelihood posterior CDFs
differ by up to `0.08418` and `0.09715`. W remains
`INCONCLUSIVE_REFERENCE_PRECISION`: maximum log-likelihood SE `0.36554` exceeds
`0.20`, and surviving ancestor fraction `0.002686` is below `0.005`. Its apparent
CDF agreement cannot validate the approximation.

The original oracle grids (65 versus 129 points) passed the CDF refinement
threshold in only 1/12 cases. A separate retained numerical refinement used
129 versus 257 points on **the same parameter draws and data**, without changing
the `0.02` threshold: all 12 then passed; worst CDF differences were
`0.01351 / 0.00874 / 0.01587` for G/W/Z. Oracle mean biases remained
`0.00571 / 0.02214 / 0.00306`; parameter coverage was `4/4, 3/4, 4/4`.
This adds zero independent calibration cases and does not replace the original
coarse-grid failure record. The main run retains its exact runner snapshot;
the refinement has its own source and configuration identities.

The matched score-based posterior biases were `+0.10002 / +0.02091 / -0.10834`;
the stress biases were `+0.00370 / +0.07585 / -0.08689`. With four replicates per
axis these are localization observations, not estimates of systematic bias.
The state reports separately compare U0 fixed, true parameters, and fitted
parameters against true `r` and clean EEG/HbO/HbR, including masked-center
coverage and replicate bootstrap intervals. Their intervals condition on
parameters and are Gaussian-moment approximations, without parameter UQ.
The evidence does not isolate a single cause for Step 4's failures or qualify
a teacher. Full 60-repeat calibration, W reference repair, parameter-integrated
UQ, shared-information nulls, and Step5A1/5B were not executed in this snapshot.

All E/R entries above are stopped historical evidence surfaces except D1B, which is
an abandoned incomplete lane. They can be read or replayed where their exact
artifacts remain, but they do not define a current queue.

The compact decision snapshot is
[`06_EXPERIMENT_LOG.md`](../docs/physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md);
the integrated report is
[`20260728_R_SERIES_EXPERIMENT_REPORT.md`](../docs/physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).
The R1-P source/config/test package remains together as a dated result surface.

## Token Atlas

The frozen 2026-07-30 E2 T0 Core artifact remains retained:

[`token_physiology_atlas_standard_loader_core_20260730`](runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/runs/t0_seed20260719/analysis/token_physiology_atlas_standard_loader_core_20260730/)

It contains the manifest, summaries, 12 figures with sidecars/alt text, source
tables, compact assignments, measurement cache, and sequence counts. It is a
stopped development artifact and did not open protected data. The unexecuted
Statistical-tier continuation in this lineage is abandoned; a new flow must define
its own question and evidence contract.

## Comparison methods

| Method | Retained surface | Policy |
| --- | --- | --- |
| STA-Net | complete [`20260727` formal run](../comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/) including 140 retained formal checkpoints and [`aggregate`](../comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/aggregate/paper_table.md) | **stopped** method-native historical reference |
| Joint protected campaign | tracked [`42-cell result report`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md), dated local (Git-ignored) 540-job status, unblind record, aggregate, and traceability manifest | **stopped**; 22 ready-with-note, 12 rejected, 2 overlap-only, 6 unsupported; run payload remains ignored |
| EFRM v2 | entire [`efrm_lodo_full_target_fivefold_v2`](../comparative_methods/EFRM-PyTorch/runs/formal/efrm_lodo_full_target_fivefold_v2/) run/protocol/status plus method runs and caches | **stopped**; retained as frozen campaign evidence |
| BIOT / CBraMod / REVE | source-fidelity and alignment evidence plus frozen public/protected campaign identities | **stopped**; protected payload remains ignored |
| BrainFusion / NormWear | reimplementation/adaptation evidence plus frozen public/protected campaign identities | **stopped**; labels and track boundaries remain mandatory |
| EFRM v1 | aggregate/status and lightweight tables/figures | **stopped** different-estimand context |
| UMAP | code, configs, README, design, and [historical UMAP note](../comparative_methods/EXPERIMENT_PLAN.md#umap) | **abandoned** diagnostic candidate; no old run directory retained |

The STA-Net checkpoints outside the latest formal run were older tuning,
smoke, personalized, or earlier formal payloads. Their configs, manifests,
metrics, aggregates, logs, and figures remain.

## Croce validation

The `croce_validation` design documents, scripts, reports, manifests, figures,
and the expensive retained `cache/croce_local/highwl_v2/` surface are **stopped**
evidence. Historical archive NPZ payloads are rebuildable from the retained
manifests/configuration and measured data, but no regeneration is implied. The
redesigned Synthetic Phase 1 and Real Phase 2 were not run and are **abandoned**;
they do not form a current dependency lane. A future clean flow must establish a
new owner, protocol, and evidence identity.

## Historical source/observation generation

The pre-2026-07 physiology-semantic archive retains its README/inventory,
resolved configs, manifests, metric logs, summaries, CSV/JSON tables, figures,
and reports. Two X3 causal-exchange checkpoints are retained because this is
the most frequently referenced audited control:

- `archive/pre_forward_implementation_20260824/snapshot/experiments/archive/pre_physiology_semantic_20260701/runs/tokenizer_cross_modal_exchange/20260626_173718_causal_cross_adapter_v1/tokenizer_interventions/k128_dim128_x3_causal_exchange_seed20260651/checkpoints/best_model.pt`
- `archive/pre_forward_implementation_20260824/snapshot/experiments/archive/pre_physiology_semantic_20260701/runs/tokenizer_cross_modal_exchange/20260626_173718_causal_cross_adapter_v1/tokenizer_interventions/k128_dim128_x3_causal_exchange_seed20260652/checkpoints/best_model.pt`

All other archived `.pt` and `.npz` payloads were removed. Their conclusions
remain readable and comparable, but exact checkpoint/array replay is no longer
locally available without rebuilding from retained code/configuration and raw
data.

## 2026-07-30 pruning record

The cleanup was frozen after checking all local processes. The only active
project compute was EFRM LODO v2; none of the targets below intersected an
open file. The Atlas and EFRM paths were outside every deletion scope.

| Removed payload | Files | Bytes |
| --- | ---: | ---: |
| `experiments/archive/**/*.npz` | 7,264 | 510,838,674,895 |
| `experiments/archive/**/*.pt`, except the two X3 controls | 423 | 75,775,750,670 |
| `experiments/runs/.../software_smoke/**/*.pt` | 46 | 874,468,394 |
| STA-Net `.pt` outside the retained 20260727 formal run | 2,369 | 94,131,041,391 |
| `croce_validation/archive/**/*.npz` | 2,343 | 8,702,201,778 |
| **Total binary payload** | **12,445** | **690,322,137,128** |

That is approximately `690.3 GB` decimal (`642.9 GiB`). Generated
`__pycache__` and `.pytest_cache` directories were also removed separately and
are excluded from this byte total.

This deletion is not recoverable from the local working tree. Git-tracked
source/docs were not removed by the payload cleanup; ignored binary payloads
would require a backup or experiment rebuild.
