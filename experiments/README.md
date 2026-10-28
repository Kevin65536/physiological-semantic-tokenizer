# Experiment workspace

Commands and configuration for data preparation, SSM diagnostics, conditional
semantic/tokenizer experiments, reporting, and retained replay.
This page owns the entrypoint/config/test map. Execution and scientific verdicts
come from the [project registry view](../docs/PROJECT_STATUS.md).

## Task navigation

| Task | Start here |
| --- | --- |
| Prepare inputs or audit measurement semantics | [Data preparation and observation checks](#data-preparation-and-observation-checks) |
| Inspect shared dynamics, reconstruction or parameter reliability | [Shared-driver reconstruction and reliability](#shared-driver-reconstruction-and-reliability) |
| Diagnose conditional Hb constraints and observation effects | [Hb conditional diagnostics](#hb-conditional-diagnostics) |
| Inspect semantic representations and tokenizer probes | [Semantic and tokenizer development](#semantic-and-tokenizer-development) |
| Export figures, reports or parameter explanations | [Visualization and report exports](#visualization-and-report-exports) |
| Replay T3, Step5 or overnight diagnostics | [Recorded diagnostic entries](#t3-step5-and-overnight-diagnostic-replay) |
| Find retained evidence | [Result index](RESULTS_INDEX.md) and [run-directory contract](runs/README.md) |
| Compare downstream task-label performance | [Comparison methods](../comparative_methods/README.md), [new rebenchmark design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1) and [retained comparison protocol](../docs/comparisons/PROTOCOL.md) |

The SSM suites evaluate observation reconstruction, reliability and conditional
representations. Downstream task-label benchmarks have a separate comparison
owner and evaluation contract.

## Directory roles

| Path | Role |
| --- | --- |
| Root `*.py` | existing diagnostic, audit, and cache-building commands at their recorded paths |
| [`configs/physiology_semantic_tokenizer/`](configs/physiology_semantic_tokenizer/README.md) | reviewed diagnostic and retained replay contracts |
| `scripts/` | all new executable entrypoints, plus existing replay, state, and figure tools |
| [`runs/`](runs/README.md) | retained generated evidence; future run root only after registration |
| `archive/` | local superseded generations; Git-ignored and never default-discovered |
| [`RESULTS_INDEX.md`](RESULTS_INDEX.md) | retained-result map and pruning record |

Comparison methods own their code, configs, runs, and caches below
`comparative_methods/<method>/`. Croce validation owns
`croce_validation/`. Do not create a second generic results root.

Add new main-project commands in `scripts/`. Fix existing root-level commands in
place; recorded paths and snapshots remain stable. Shared computations belong
in the closest `src/` owner, with data and replay roots passed explicitly.

## Entrypoint, config and test map

### Cached Hb input migration

New measured Hb preparation uses the already bandpassed v5/none cache through
`src.data.unified_physiology.load_cached_fnirs_record`. The input, units, support,
offline masking semantics and historical boundary are owned by
[the data contract](../docs/DATA_CONTRACT.md#新-hb-建模输入统一使用带通缓存--2026-10-27).
The following cached-input contracts retain their **pre-Tak model/metric**
identity. Their CLIs now require an explicit `--config`; they are historical
replay/control entrypoints, not implicit choices for the research baseline.
The [Tak entry](#shared-driver-reconstruction-and-reliability) uses the same
unified cache reader. A new Tak spatial/continuity/teacher/closure extension
must version its state and target contract rather than only replace a config path.

| Retained entry | Explicit pre-Tak cache-input contract | Preparation checks |
| --- | --- | --- |
| `scripts/evaluate_ssm_output_support.py` | [`ssm_output_support_cached_v2.yaml`](configs/physiology_semantic_tokenizer/ssm_output_support_cached_v2.yaml); REFED: [`ssm_output_support_refed_cached_v3.yaml`](configs/physiology_semantic_tokenizer/ssm_output_support_refed_cached_v3.yaml) | `test_cached_fnirs_input.py`, `test_ssm_output_support.py` |
| `scripts/evaluate_shared_driver_spatial.py` | [`shared_driver_spatial_cached_v2.yaml`](configs/physiology_semantic_tokenizer/shared_driver_spatial_cached_v2.yaml) | `test_cached_fnirs_input.py`, `test_shared_driver_spatial_data.py`, `test_shared_driver_spatial_runner.py` |
| `scripts/run_functional_tokenizer.py` | [`functional_tokenizer_cached_v2.yaml`](configs/physiology_semantic_tokenizer/functional_tokenizer_cached_v2.yaml) | `test_cached_fnirs_input.py`, `test_functional_tokenizer.py` |
| `scripts/evaluate_model_compatibility.py` | [`model_compatibility_cached_v2.yaml`](configs/physiology_semantic_tokenizer/model_compatibility_cached_v2.yaml) | `test_cached_fnirs_input.py`, `test_model_compatibility_runner.py`, `test_model_compatibility_report.py` |

These contracts require fresh preparation and training coordinates; their presence
does not record a measured campaign. The cached census and its report do not mix in
old native-parent fit panels. Downstream volume/decay, feasibility, information and
token probes still reference explicit retained parents; replay preserves those
parents and does not turn their arrays into filtered evidence. Migrating those
results requires newly fitted cached parents, not a path substitution.

The native reader remains for the Hb prevalence/native-vs-cache audit, historical
raw-to-prepared curve verification, and historical window-local **bandpassed**
semantics/parameter-stability experiments. The old unfiltered 120-second context
producer is retired from current execution; retained result reading is supported.
Use its frozen source for exact historical regeneration. The versioned entries
below retain the contracts used to generate their dated results.

Configs below are relative to `configs/physiology_semantic_tokenizer/`; tests
are relative to `../tests/`. The grouped tables map software surfaces, not
execution state or permission to launch. Detailed usage follows in the linked
sections.

### Data preparation and observation checks

Measurement caches, coordinate and scale audits, native Hb summaries and optical preprocessing.

| Diagnostic / usage | Entry | Config | Targeted tests |
| --- | --- | --- | --- |
| General public measurement cache; explicit versioned output root, signal → event → geometry builders under a durable supervisor | [`build_clean_eeg_fnirs_cache.py`](build_clean_eeg_fnirs_cache.py), [`build_clean_event_index.py`](build_clean_event_index.py), [`build_clean_channel_geometry.py`](build_clean_channel_geometry.py) | [Data contract](../docs/DATA_CONTRACT.md#无运动校正的通用缓存--2026-09-28); CLI `--processing-schema physiology_measurement_alignment_v5 --motion-method none`; MNE is an explicit optical comparison in a separate root. Metadata-only REFED layout ([paper extraction](scripts/extract_refed_paper_layout.py)): `--datasets refed --refed-geometry paper_layout_v1 --output-dir <fresh_geometry_root>`, consumed via `geometry_root`; [planar geometry contract](../docs/DATA_CONTRACT.md#refed-paper-layout-registration--2026-10-13) | `test_clean_cache_builder.py`, `test_clean_physiology_cache.py`, `test_homer2_measurement_v5.py`, `test_homer2_measurement_v4.py`, `test_unified_physiology.py`, `test_event_alignment.py`, `test_channel_geometry.py`, `test_physiology_semantic_local.py` |
| Observation coordinate stability, native EOG events, hardware donor contrasts and matched synthetic mechanisms; `--stage check`, supervised `pilot` / `all`, bounded `refinement`, retained-array `verify` | [`scripts/evaluate_observation_semantics.py`](scripts/evaluate_observation_semantics.py); [`PPT/PDF renderer`](scripts/render_observation_semantics_report.py) | [`v1`](configs/physiology_semantic_tokenizer/observation_semantics_v1.yaml), [`C/D v2`](configs/physiology_semantic_tokenizer/observation_semantics_v2.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#观测物理量事件锚点与竞争机制2026-10-09) | `test_observation_semantics.py` |
| Measurement precision and unit revision; fixed v3 stage 1/2 | [`evaluate_ssm_overnight_diagnostics.py`](evaluate_ssm_overnight_diagnostics.py) | [`ssm_measurement_alignment_v3.yaml`](configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml) | `test_ssm_overnight_diagnostics.py`, `test_step5_observation_diagnostic.py`, `test_step5.py`, `test_homer2_preprocessing.py` |
| Four-dataset amplitude audit and alignment repair reports (`--stage alignment-report --output-dir <new versioned export>`) | [`scripts/analyze_dataset_scaling.py`](scripts/analyze_dataset_scaling.py) | [`dataset_scaling_report_v1.yaml`](configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml); alignment report reads the retained [measurement v3](configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml) evidence | [`test_dataset_scaling_report.py`](../tests/test_dataset_scaling_report.py) |
| Public native Hb direction prevalence; `--prevalence --stage synthetic`, then supervised `--stage measured --workers N`; optional `--prevalence-pilot`; `--stage report` creates a versioned bitmap export | [`scripts/analyze_hbo_hbr_relationship.py`](scripts/analyze_hbo_hbr_relationship.py); native fNIRS reader in `src/data/unified_physiology.py` | [`hbo_hbr_prevalence_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_prevalence_v1.yaml): four public datasets, native/released Hb versus v5 none, subject-weighted census, trend and optical-coefficient sensitivity; no model fitting | `test_hbo_hbr_relationship.py`, `test_unified_physiology.py`, `test_clean_physiology_cache.py`; paired-mask statistics, native identity/support, subject weighting, synthetic exclusion |
| Read-only optical motion pipeline audit; metadata `--check-only`, then separate synthetic/measured phases on the exact retained public windows; no model fits | [`scripts/analyze_fnirs_motion_correction.py`](scripts/analyze_fnirs_motion_correction.py) | [`optical motion audit`](configs/physiology_semantic_tokenizer/shared_driver_optical_motion_audit_v1.yaml) | `test_fnirs_motion_correction_audit.py`, `test_homer2_preprocessing.py`, `test_homer2_measurement_v4.py` |

### Shared-driver reconstruction and reliability

The default shared-driver model is the [Tak baseline](../docs/ARCHITECTURE.md#tak-hemodynamic-baseline).
With no historical mode selected, the existing runner resolves
[`shared_driver_tak_10hz_v2.yaml`](configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml)
and starts with its array-free `check` stage. For example:

```bash
.venv/bin/python experiments/scripts/evaluate_shared_driver_reconstruction.py \
  --check-only --run-dir <fresh_run>
```

The current path recomputes 100 ms EEG power from registered native voltage and
uses the same 10 Hz model grid as cached Hb. Preparation refits development-only
coordinates and rejects old 4 Hz arrays; physical-time rules are owned by the
[data contract](../docs/DATA_CONTRACT.md#新-eeg-代理目标统一为-10-hz--2026-10-27).
`check` is synthetic software validation; it does not start measured work.

The complete public-inventory four-axis Tak diagnostic uses
[`evaluate_tak_mismatch.py`](scripts/evaluate_tak_mismatch.py) and
[`tak_mismatch_v1.yaml`](configs/physiology_semantic_tokenizer/tak_mismatch_v1.yaml).
Its [design](../docs/EXPERIMENT_PLAN.md#tak-四轴全公开数据失配分析2026-10-28)
owns subject-heldout coordinates, equal-capacity spatial selection and the
fixed/shared-parameter comparison. `check` and `synthetic` precede metadata-only
`plan`, supervised `prepare` and `run`, then `summarize`. Record preparation,
window fits and shared-parameter groups retain resumable terminal evidence.
After both parameter modes are terminal, `audit-status` writes a fresh
`terminal_status_audit.json`, separating numerical failure from unavailable
support and execution errors without changing any fit or summary.
The read-only [`render_tak_mismatch.py`](scripts/render_tak_mismatch.py) exports
dataset distributions and preselected time curves after both parameter modes
are terminal, including signed residuals and instantaneous spatial error
differences on their retained scoring supports. Revisions use fresh export
directories. Tests: `test_tak_mismatch.py`, `test_tak_mismatch_data.py`,
`test_tak_mismatch_runner.py`, `test_tak_mismatch_current_dof.py`,
`test_tak_mismatch_report.py`; independent replay equivalence is covered by
`test_shared_driver_replay_equivalence.py`.

The bounded sensor-proxy suite uses
[`evaluate_shared_driver_sensor_proxy.py`](scripts/evaluate_shared_driver_sensor_proxy.py)
with [`shared_driver_sensor_proxy_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_sensor_proxy_v1.yaml).
It compares the original six-channel PC1, five band gains, equal-capacity spatial
means and fixed ring/remote/random controls under one fixed Tak model. Its
[protocol](../docs/EXPERIMENT_PLAN.md#传感器空间代理的固定-tak-首轮检验2026-10-27)
separates EEG-only whole-Hb prediction, joint reconstruction and the fixed
Hb-only reference. Use `--stage check` for the configuration contract,
`--stage synthetic --output <fresh_synthetic_run>` for the numerical pilot,
then metadata-only `--stage plan`;
`--stage run --output <fresh_run> --workers <measured_capacity>` executes
preparation, subject-grouped screening and the bounded nonlinear panel under
a durable supervisor. Preparation reads registered native EEG and cached Hb;
it does not reuse historical 4 Hz fitted coordinates or open protected evidence.
Per-task terminal records retain numerical failures and unavailable controls.
After completion, `--stage analyze --output <completed_run> --analysis-output
<completed_run>/<fresh_analysis_version>` exports the joint/Hb-only residual
comparison on identical successful windows and the prediction pairing contrasts.
This reads retained arrays only and preserves the original run summaries.
Tests: `test_shared_driver_sensor_proxy_data.py`, `test_low_capacity_eeg_proxy.py`,
`test_shared_driver_sensor_proxy_runner.py`, `test_sensor_proxy_contract_audit.py`.
The read-only Chinese report entry is
[`render_shared_driver_sensor_proxy.py`](scripts/render_shared_driver_sensor_proxy.py):
`--run <completed_run> --output <fresh_child_export>`. It verifies retained
arrays and produces a searchable PDF/HTML report plus a complete waveform atlas,
with PNG figures and retained LaTeX sources. Configuration explanations, model
selection, failures and common-support residuals stay distinct. Report-specific
checks are in `test_sensor_proxy_report.py`.

The retained 4 Hz Tak comparison is selected explicitly with `--tak-comparison
--tak-stage <check|plan|synthetic|prepare|pilot|measured|sensitivity|summarize|verify|audit>
--run-dir <fresh_run> --config
experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_v1.yaml` using
[`shared_driver_tak_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_tak_v1.yaml).
It compares the retained inlet-balance core with deterministic reduced Tak/SPM
hemodynamics, matched initial priors, parameter estimation and declared
sensitivity controls on newly prepared cached Hb coordinates. The
[protocol](../docs/EXPERIMENT_PLAN.md#tak-原始血流核心的缓存实测对照2026-10-27)
owns scope and interpretation. Tests: `test_shared_driver_tak.py`,
`test_shared_driver_tak_experiment.py`, `test_shared_driver_parameter_stability.py`;
the current clock, native preparation and loss weights use `test_eeg_proxy_10hz.py`.
After verification and the independent retained-array audit, render with
`scripts/render_shared_driver_reconstruction.py --tak-comparison --run <run>
--output <fresh_export>` for the searchable PDF and complete bitmap waveform atlas.

The same renderer's `--tak-parameter-individuality --run <completed_tak_v1_run>
--output <fresh_export>` reads only retained task JSON to compare the final
κ/τ estimates at log-prior SD 0.125, 0.25 and 0.5. It exports subject plots,
independent-record repeats, multistart spans, matched reconstruction comparisons,
source-linked CSVs and a bitmap-figure PDF. Shared parameter groups and fixed
reference values never count as independent stable repeats; failed fits remain
in the planned denominator. This is descriptive reporting, with no signal reads
or refits. Tests: `test_tak_parameter_report.py`.

The new disjoint-data repeatability experiment uses the same evaluation entry
with `--tak-repeatability --repeat-stage
<check|plan|synthetic|prepare|pilot|run|summarize|analyze>` and
[`shared_driver_tak_repeatability_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_tak_repeatability_v1.yaml).
It preserves the original 24 evaluation subjects and prefrontal montage, prepares
new 10 Hz coordinates from the original development subjects, and independently
reestimates constant κ/τ from metadata-selected nonoverlapping 90 s subsets.
Native record/support membership, same-condition and supplementary REFED videos,
all planned windows and failures remain explicit. Preparation and fitting require
the recorded durable supervisor; `analyze` reads completed retained results only
and writes a fresh child export (`--repeat-analysis-output` optional).
Render with `render_shared_driver_reconstruction.py --tak-repeatability
--run <completed_run> --output <fresh_export>`; `--repeat-analysis` can select
an explicit analysis version. Tests: `test_tak_repeatability.py` and the existing
Tak/10 Hz suites. [Protocol](../docs/EXPERIMENT_PLAN.md#tak-参数的更多不重叠数据重复估计2026-10-28).

Automatic fNIRS morphology scoring reads the retained targets and predictions
without refitting or changing the parent evidence:

```bash
.venv/bin/python experiments/scripts/evaluate_shared_driver_reconstruction.py \
  --config <completed_tak_run>/resolved_config.json \
  --tak-stage morphology --run-dir <completed_tak_run> \
  --morphology-output <completed_tak_run>/morphology_v1
```

The export currently accepts the retained `shared_driver_tak_v1` contract;
pass its saved configuration explicitly rather than a newer fitting default.
It uses the complete waveform-atlas scope: every planned 90 s identity,
`joint_full` and `joint_blocked`, and `legacy_free`, `legacy_free_fit`, `tak_fit`.
It requires a fresh export directory below the completed run. The parent
configuration, task identities and public cached-input contract are checked
before reading the saved arrays; no raw-data, cache-production or fitting path
is invoked. Original qualification and failure fields remain attached to each
result. The hidden condition scores its actual disconnected hidden Hb blocks;
metrics never bridge those blocks or substitute full-window scores.

The numerical definitions and engineering defaults are owned by
[`src/metrics/fnirs_morphology.py`](../src/metrics/fnirs_morphology.py).
They report separate dimensions, without a weighted total or an automatic
physiological qualification:

- trajectory and local detrended shape correlation, with slope-direction loss;
- matched local peak/trough timing and half-prominence width;
- missed and additional extrema, precision, recall and matching denominators;
- measured versus reconstructed HbO/HbR covariation on common support.

Amplitude ratio and training-scale bias/error are separate. Independent
normalization is restricted to temporary single-component shape scoring; it
does not alter observations, predictions, the paired-Hb coordinates or fitting.
`within_segment_r` pools values after centering each continuous support block;
the ordinary full-curve correlation is unavailable for disconnected support.
Local shape correlations remove a linear trend within each complete window.
Slope-direction loss is the fraction of reference slope-squared energy whose
predicted direction is zero or opposite; paired-Hb direction loss instead
weights by the absolute product of the two reference slopes. Lower is better.
Event precision/recall count same-type, order-preserving one-to-one matches;
timing and width errors describe only matched events, alongside their counts.
Positive timing error means the reconstructed event is later. Widths whose
controlling prominence base is at a support boundary are censored; segment-local
widths remain diagnostic detail and do not enter complete-width error summaries.
Low dynamics, missing support, flat predictions, nonfinite values and censored
widths remain explicit. Unavailable quantities are JSON `null`/empty CSV cells,
with their status and coverage counts; they are not filled with good scores.
Short hidden blocks may have no complete local-correlation window or peak width.
The saved settings are numerical definitions, not calibrated biological or
acceptance thresholds; optional `--morphology-settings <json>` overrides are
validated and recorded in each fresh export.

The CSV provides scalar metrics with canonical sample, model, condition and
component keys for joins; JSON retains segment, local-window and event detail.
Use subject-level paired summaries when combining these with other endpoints;
overlapping windows and extrema are not independent subjects. Tests:
[`test_fnirs_morphology.py`](../tests/test_fnirs_morphology.py),
[`test_fnirs_morphology_export.py`](../tests/test_fnirs_morphology_export.py).

The bounded post-10.20 filter sensitivity uses
`scripts/evaluate_shared_driver_reconstruction.py --filter-retest --filter-stage
<check|plan|prepare|pilot|run|summarize|verify> --run-dir <fresh_run>` with
[`shared_driver_filter_retest_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_filter_retest_v1.yaml).
It selects examples by retained original-fit median error before new fits, uses
only original/slow/no-decay dynamics without additional components or inputs,
and compares retained arrays with the unified cache using development-only
branch coordinates. Fitting runs under user systemd; reports retain waveform,
amplitude, offset, numerical, conditional-state and compatibility diagnostics.
Selection uses the owning `evidence_v2` fit corrections. Render verified results
with `scripts/render_shared_driver_reconstruction.py --filter-retest --run
<completed_run> --output <fresh_export>`; the PDF embeds bitmap figures.
Tests: `test_fnirs_filter_retest.py`, `test_shared_driver_decay_experiment.py`.
See the [bounded design](../docs/EXPERIMENT_PLAN.md#无额外成分的滤波敏感性样例复测2026-10-27).

Representation lower bounds and conditional joint feasibility use
[`evaluate_source_observability.py --feasibility`](scripts/evaluate_source_observability.py),
the [`feasibility contract`](configs/physiology_semantic_tokenizer/shared_driver_feasibility_v1.yaml),
and its [design](../docs/EXPERIMENT_PLAN.md#当前共享驱动模型的表达边界与联合可行性2026-10-22).
Run `--stage check`, synthetic controls, then supervised bounds, vascular and
frontier stages; parameter profiles require a verified joint witness. Targeted
tests are `test_shared_driver_feasibility.py`, `test_hb_vascular_feasibility.py`
and `test_shared_driver_feasibility_runner.py`. The only measured input is the
explicit retained public spatial parent; no new raw-data or protected reader.

Continuous 30/90 s Hb compatibility and independent common-volume diagnostics use
the historical output-direction comparison below. The direct follow-up on the
unforced `p-v` decay uses the same runner with `--decay-diagnostic --decay-stage
<check|pilot|synthetic|measured|summarize|replay-migrate>` and
[`shared_driver_decay_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_decay_v1.yaml).
It changes the p-state equation while retaining the original observation mapping;
targeted tests are `test_shared_driver_decay.py` and
`test_shared_driver_decay_experiment.py`. See the
[direct dynamics protocol](../docs/EXPERIMENT_PLAN.md#pv-无源衰减的直接动力学检查2026-10-23).
The replay-only migration preserves the original fits and writes `evidence_v2`;
render its retained results with `--decay-diagnostic --run <run>
--decay-evidence <run>/evidence_v2 --output <fresh_export>`.

The retained output-direction comparison uses
[`scripts/evaluate_shared_driver_reconstruction.py`](scripts/evaluate_shared_driver_reconstruction.py)
`--volume-drive-diagnostic --volume-stage <pilot|synthetic|linear|measured|summarize>`
with [`shared_driver_volume_drive_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_volume_drive_v1.yaml).
Run `--check-only` and `tests/test_shared_driver_volume_drive.py` first; substantial
stages use a durable supervisor. The [protocol](../docs/EXPERIMENT_PLAN.md#单一血流驱动的-hb-相容性与-3090-秒持续波动2026-10-22)
restricts reads to the two named public output-support parents and distinguishes
reconstruction, blocked interpolation and source attribution.
The existing renderer accepts `--volume-drive-diagnostic --run <completed_run>
--output <fresh_export>` for the Chinese bitmap PDF/HTML report and figure atlas.

Shared dynamics, conditional reconstruction, input/geometry controls and parameter or state reliability.

The [four independent fit/recovery diagnostics](../docs/EXPERIMENT_PLAN.md#拟合观测信息与生理恢复的四套独立诊断2026-10-21)
separate data observability, spatial information, cross-modal increments and mechanism
attribution. Each owns its contract and run; their technical reports consume retained
results rather than defining an additional state surface.

| Diagnostic / usage | Entry | Config | Targeted tests |
| --- | --- | --- | --- |
| Joint physiological recovery, data/prior spectra, nuisance compensation and nonlinear profiles; synthetic truth plus explicit public retained evidence | [`scripts/evaluate_source_observability.py`](scripts/evaluate_source_observability.py) | [`source_observability_v1.yaml`](configs/physiology_semantic_tokenizer/source_observability_v1.yaml) | `test_source_observability.py` |
| Nested spatial observation information, geometry and same-readout dynamic controls; new synthetic controls and retained public arrays | [`scripts/evaluate_spatial_information.py`](scripts/evaluate_spatial_information.py) | [`spatial_information_v1.yaml`](configs/physiology_semantic_tokenizer/spatial_information_v1.yaml) | `test_spatial_information.py` |
| Fixed-weight EEG pairing/time interventions and independently generated shared-signal dose controls | [`scripts/evaluate_crossmodal_information.py`](scripts/evaluate_crossmodal_information.py) | [`crossmodal_information_v1.yaml`](configs/physiology_semantic_tokenizer/crossmodal_information_v1.yaml), [`conditional positive control`](configs/physiology_semantic_tokenizer/crossmodal_information_positive_v1.yaml) | `test_crossmodal_information.py` |
| Paired true-parameter, observation-gain and common-component interventions with nonlinear refits and explicit oracle controls | [`scripts/evaluate_mechanism_attribution.py`](scripts/evaluate_mechanism_attribution.py) | [`mechanism_attribution_v1.yaml`](configs/physiology_semantic_tokenizer/mechanism_attribution_v1.yaml) | `test_mechanism_attribution.py` |
| REFED extension of the ten public SSM diagnostics; explicit paper geometry, video grouping and retained synthetic provenance; shared/disjoint-optode experiment excluded | Existing entries below; base `scripts/evaluate_shared_driver_reconstruction.py --physiology-semantics`; four-dataset response statistics `evaluate_semantic_response_discovery.py --stage four-dataset-summary`; PPT/PDF `render_shared_driver_reconstruction.py --refed-comparison --comparison-config <report_contract> --output <fresh_export>` | [`shared_driver_physiology_semantics_refed_v2.yaml`](configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_refed_v2.yaml), corresponding `*_refed_v2.yaml` and [`report evidence map`](configs/physiology_semantic_tokenizer/refed_four_dataset_report_v1.yaml); [scope](../docs/EXPERIMENT_PLAN.md#refed-公开实验补齐与四数据集比较2026-10-13) | Corresponding suites below; geometry/video fixtures in `test_shared_driver_physiology_semantics.py` and `test_shared_driver_parameter_stability.py`; pooled-statistics fixture in `test_semantic_response_discovery_runner.py` |
| Shared-driver reconstruction, parameter profiles, frozen replay and constrained nonlinear fitting; `--check-only`, then fresh `--run-dir` under a durable supervisor; replay uses `--replay-of <exact parent>`; fitting uses `--nonlinear-fit`, `--gain-prior-fit`, `--fixed-roi-tau-fit`, `--fixed-roi-flow-fit` or `--fixed-roi-initial-fit`, `--fixed-roi-optical-fit` or `--conditional-optical-gain-fit`; training-only numerical comparison uses `--conditional-step-control`, with separate `--phase synthetic/measured` calls | [`scripts/evaluate_shared_driver_reconstruction.py`](scripts/evaluate_shared_driver_reconstruction.py), [`bitmap report renderer`](scripts/render_shared_driver_reconstruction.py) | [`linear screen`](configs/physiology_semantic_tokenizer/shared_driver_reconstruction_v1.yaml), [`nonlinear replay`](configs/physiology_semantic_tokenizer/shared_driver_nonlinear_replay_v1.yaml), [`nonlinear fit`](configs/physiology_semantic_tokenizer/shared_driver_nonlinear_fit_v1.yaml), [`gain and driver prior`](configs/physiology_semantic_tokenizer/shared_driver_gain_prior_v1.yaml), [`training budget continuation`](configs/physiology_semantic_tokenizer/shared_driver_gain_prior_budget_v1.yaml), [`fixed ROI and shared tau`](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml), [`fixed ROI log-flow sensitivity`](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_flow_v1.yaml), [`initial Hb-volume constraint`](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_initial_v1.yaml), [`optical observation contrast`](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_optical_v1.yaml), [`conditional optical mapping and gain`](configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml), [`same-objective step control`](configs/physiology_semantic_tokenizer/shared_driver_conditional_step_control_v1.yaml), [`paired solver panel`](configs/physiology_semantic_tokenizer/shared_driver_conditional_step_control_v2.yaml); conditional diagnostics, no teacher qualification | `test_shared_driver_reconstruction.py`, `test_shared_driver_nonlinear_fit.py`, `test_shared_driver_observation.py`, `test_shared_driver_flow_regularization.py`, `test_shared_driver_initial_constraint.py`, `test_shared_driver_initial_experiment.py`, `test_shared_driver_optical_experiment.py`, `test_shared_driver_conditional_gain_experiment.py`, `test_shared_driver_step_control.py`, `test_shared_driver_step_control_experiment.py`, `test_shared_driver_positive_hb.py`, `test_shared_driver_flow_experiment.py`, `test_shared_driver_experiment.py` |
| Fixed-structure cross-dataset physiology semantics; `--physiology-semantics --check-only`, then supervised `--semantics-stage` phases with frozen phase sources; native QC, synthetic recovery, parameter sharing, completion, profiles, `diagnostics`, `integration_audit`, then terminal `summarize` | [`scripts/evaluate_shared_driver_reconstruction.py`](scripts/evaluate_shared_driver_reconstruction.py); optional arithmetic backend `src/inference/shared_driver_rk4.py`, original Python replay retained | [`shared_driver_physiology_semantics_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml); [design and execution boundaries](../docs/EXPERIMENT_PLAN.md#固定结构的跨数据集生理语义检验2026-09-28-设计) | `test_shared_driver_physiology_semantics.py`, `test_shared_driver_compiled.py`, `test_shared_driver_nonlinear_fit.py`, `test_shared_driver_flow_regularization.py`, `test_shared_driver_step_control.py`, `test_unified_physiology.py` |
| Local 2/4-Hb and 6/12-EEG spatial observations, K1/K2 independent conserved dynamics, frozen calibration and geometry/pairing controls; `check`, `dryrun`, `synthetic-smoke`, then explicit supervised stages | [`scripts/evaluate_shared_driver_spatial.py`](scripts/evaluate_shared_driver_spatial.py) | [`shared_driver_spatial_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_spatial_v1.yaml); [design and read boundary](../docs/EXPERIMENT_PLAN.md#局部多通道观测与空间动态维数2026-10-16); metadata-selected public Single-Trial only, no protected reads or automatic measured continuation | `test_shared_driver_spatial.py`, `test_shared_driver_spatial_data.py`, `test_spatial_observation_baselines.py`, `test_shared_driver_spatial_runner.py` |
| Regional broadband/spectral shared states; synthetic checks, training-frozen vascular readout, hidden-feature and pairing controls under a durable supervisor | [`scripts/evaluate_shared_driver_modes.py`](scripts/evaluate_shared_driver_modes.py); [bitmap report export](scripts/render_shared_driver_modes.py) | [`shared_driver_modes_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_modes_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#区域与谱模式共享驱动检验2026-10-02) | `test_shared_driver_modes.py`, `test_shared_driver_modes_runner.py` |
| Paired 4/10 Hz EEG proxy reconstruction with fixed Hb targets and driver-grid control; check, synthetic, supervised prepare/pilot/run | [`scripts/evaluate_eeg_proxy_rate.py`](scripts/evaluate_eeg_proxy_rate.py) | [`eeg_proxy_rate_v1.yaml`](configs/physiology_semantic_tokenizer/eeg_proxy_rate_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#eeg-功率代理采样率与实测重建2026-10-10) | `test_eeg_proxy_rate.py` |
| Finite response selection, cross-fitted independent spatial means and confounding profiles; `--check-only`, then supervised `--stage pilot` and `--stage run` | [`scripts/evaluate_ssm_strengthening.py`](scripts/evaluate_ssm_strengthening.py) | [`ssm_strengthening_v1.yaml`](configs/physiology_semantic_tokenizer/ssm_strengthening_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#有限响应适配与独立观测约束2026-10-03) | `test_ssm_strengthening.py`, `test_ssm_strengthening_runner.py` |
| Triggered synthetic same-observation state propagation; `--check-only`, supervised `--stage pilot` and `--stage run` | [`scripts/evaluate_ssm_state_continuity.py`](scripts/evaluate_ssm_state_continuity.py) | [`ssm_state_continuity_v1.yaml`](configs/physiology_semantic_tokenizer/ssm_state_continuity_v1.yaml); same strengthening protocol | `test_ssm_state_continuity.py`, `test_shared_driver_reconstruction.py` |
| Parameter stability, common-interior profiles, continuous records, working covariance and development-selected sharing; metadata `--stage plan`, with named supervised stages and source/invocation snapshots retained in the owning run | [`scripts/evaluate_shared_driver_parameter_stability.py`](scripts/evaluate_shared_driver_parameter_stability.py); [`PPT/PDF renderer`](scripts/render_shared_driver_parameter_stability.py) | [`shared_driver_parameter_stability_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_parameter_stability_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#参数稳定性共同内点与拟合改进2026-10-10) | `test_shared_driver_parameter_stability.py`, `test_shared_driver_compiled.py`, `test_shared_driver_physiology_semantics.py` |
| Output-function profiles, matched-prior continuous states, native Hb masking and paired-EEG power calibration; metadata `--stage plan`, then supervised pilot/prepare/calibrate/synthetic/measured/profiles/power/summary | [`scripts/evaluate_ssm_output_support.py`](scripts/evaluate_ssm_output_support.py); [`Markdown renderer`](scripts/render_ssm_output_support.py) | [`ssm_output_support_v1.yaml`](configs/physiology_semantic_tokenizer/ssm_output_support_v1.yaml); [`power calibration v2`](configs/physiology_semantic_tokenizer/ssm_output_power_calibration_v2.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#ssm-输出支持范围连续状态与跨模态约束2026-10-10) | `test_ssm_output_support.py`, `test_shared_driver_compiled.py`, `test_ssm_state_continuity.py` |
| Four-arm conditional teacher robustness; `--teacher-robustness --check-only`, then supervised `--robustness-stage` pilot, synthetic_calibration, synthetic, calibration, measured, profiles, summarize | [`scripts/evaluate_shared_driver_reconstruction.py`](scripts/evaluate_shared_driver_reconstruction.py); PPT export: existing renderer `--teacher-robustness` | [`shared_driver_teacher_robustness_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_teacher_robustness_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#观测失配与驱动先验的四臂检验2026-10-01) | `test_shared_driver_teacher_robustness.py`, `test_shared_driver_nonlinear_fit.py`, `test_shared_driver_physiology_semantics.py` |
| Frequency-band and soft geometry structure screen; bitmap report | [`evaluate_ssm_overnight_diagnostics.py`](evaluate_ssm_overnight_diagnostics.py), [`render_ssm_band_geometry_report.py`](scripts/render_ssm_band_geometry_report.py) | [`ssm_band_geometry_v1.yaml`](configs/physiology_semantic_tokenizer/ssm_band_geometry_v1.yaml) | `test_ssm_band_geometry.py`, `test_ssm_overnight_diagnostics.py` |
| Retained Hb waveform mechanism diagnosis; `--waveform-diagnostic`, separate synthetic/measured phases; same fold-3 training/validation objects, fixed difficult identity, no new raw inputs | [`scripts/evaluate_shared_driver_reconstruction.py`](scripts/evaluate_shared_driver_reconstruction.py), report via [`scripts/render_shared_driver_reconstruction.py`](scripts/render_shared_driver_reconstruction.py) `--waveform-diagnostic` and optional `--volume-fraction-run`; offline interactive explanation: `--fitting-explainer --run <completed_waveform_run> --output <fresh_export>` | [`waveform diagnostic`](configs/physiology_semantic_tokenizer/shared_driver_waveform_diagnostic_v1.yaml), [`extra volume loading followup`](configs/physiology_semantic_tokenizer/shared_driver_waveform_volume_fraction_v1.yaml); linear parameter/component screens and nonlinear penalty ablations, exploratory reconstruction only | `test_shared_driver_waveform_diagnostic.py`, `test_shared_driver_fitting_explainer.py`, `test_shared_driver_reconstruction.py`, `test_shared_driver_nonlinear_fit.py` |
| Retained lower-bound example: paired tau/driver-curvature and EEG/flow/initial-prior controls; same `--waveform-diagnostic` entry, separate `--phase synthetic`, measured `--pilot`, then measured continuation | Existing [`runner`](scripts/evaluate_shared_driver_reconstruction.py) and [`renderer`](scripts/render_shared_driver_reconstruction.py) `--waveform-diagnostic` | [`tau/curvature diagnostic`](configs/physiology_semantic_tokenizer/shared_driver_tau_curvature_v1.yaml); exact public MNE TDDR subject_01 fold-3 validation trajectories only; fixed tau stress grid, full-window conditional refits, no new parameter selection or raw reads | `test_shared_driver_waveform_diagnostic.py`, `test_shared_driver_compiled.py` |

### Hb conditional diagnostics

Hb constraints, observation mappings and counterfactual checks share the relationship-analysis entry below.

The versioned whole-inventory compatibility suite uses
[`evaluate_model_compatibility.py`](scripts/evaluate_model_compatibility.py)
with [`model_compatibility_v1.yaml`](configs/physiology_semantic_tokenizer/model_compatibility_v1.yaml).
Run metadata `--stage plan` and array-free `--stage synthetic` first, then supervised
`pilot`, `census`, `summarize` and `panels --panel-stage <pilot-linear|pilot-nonlinear|linear|nonlinear|space>`.
The [design](../docs/EXPERIMENT_PLAN.md#五项模型相容性指标的总体分布与典型曲线2026-10-23)
distinguishes native Hb inventory from the smaller frozen observation panels.
[`render_model_compatibility.py --run <completed_run> --output <fresh_export>`](scripts/render_model_compatibility.py)
exports the distributions and exact selected-window curves as Chinese bitmap PDF/HTML.
Tests: `test_model_compatibility.py`, `test_model_compatibility_runner.py`,
`test_model_compatibility_panels.py`, `test_model_compatibility_report.py`.

| Diagnostic / usage | Entry | Config | Targeted tests |
| --- | --- | --- | --- |
| Descriptive HbO/HbR morphology, weak dynamic constraints and residual terms | [`scripts/analyze_hbo_hbr_relationship.py`](scripts/analyze_hbo_hbr_relationship.py) (`--check-only`, optional `--dynamics --nonlinear-check`; measurement audit uses `--dynamics` or `--terms` with `--output-dir <fresh data_quality_audit child>`) | Subject/record scope from `dataset_scaling_report_v1.yaml`, retained v4 measurement cache; reference dynamics from `step5_v1.yaml`. `--terms` reports level/velocity/acceleration distributions and signed residual-energy allocation. No SSM fitting or qualification. | `test_hbo_hbr_relationship.py`; built-in analytic transfer, independent ODE, fault-identity, derivative/cancellation and optional nonlinear controls |
| Cross-window shared Hb constraint adaptation | Same entry, `--adapt --check-only`, then `--adapt --output-dir <fresh audit child>` under a durable supervisor; `--resume` continues saved fits. Completed evidence exports use `--adapt-report` or `--adapt-smooth-report` with its output directory. | [`hbo_hbr_adaptation_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_adaptation_v1.yaml): blocked fit/evaluation, subject/channel sharing, primary/expanded parameter boxes; conditional correction semantics, no unique physiological attribution | `test_hbo_hbr_relationship.py`; synthetic heldout adaptation, exact compressed projection, split, identifiability and constrained repair checks |
| Hb parameter-boundary mechanisms | Same entry, `--boundary --check-only`, then `--boundary --output-dir <fresh audit child>` under a durable supervisor; `--resume` continues task rows; `--boundary-report` exports completed evidence | [`hbo_hbr_boundary_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_boundary_v1.yaml): retained adaptation arrays only, effective-coordinate profiles, independent optimizer check, smooth/initial-state controls and known-truth observation perturbations; signed combinations are diagnostics, not physiological replacements | `test_hbo_hbr_relationship.py`; effective transfer/projection equivalence, limiting operators, smooth minimum and built-in synthetic recovery |
| Hb observation-layer and metric separation | Same entry, `--observation --check-only`, then `--observation --output-dir <fresh audit child>` under a durable supervisor; `--resume` retains task rows; `--observation-report` exports completed evidence | [`hbo_hbr_observation_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_observation_v1.yaml): retained curves, shared relative gain, training-frozen covariance proxies, independent synthetic noise calibration and exact gain aliases; no raw/protected reads or unique physiological-parameter claim | `test_hbo_hbr_relationship.py`; gain equivalence, generalized minimum correction, data-independent synthetic/shape checks |
| Training-frozen Hb observation gain and masked prediction | Same Hb relationship entry, `--predictive-separation --check-only`, then `--stage synthetic` and `--stage measured` with the same `--output-dir`, under a durable supervisor; `--stage report` regenerates the report | [`hbo_hbr_predictive_separation_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_predictive_separation_v1.yaml): inherited public panel, train-anchored relative gain, fixed E0/alpha, shared tau/eta, feature-masked HbR completion; exploratory and distinct from calibrated C/D | `test_hbo_hbr_predictive_separation.py`; independent analytic response, gain recovery, hidden-value isolation, training coordinate invariance and configuration serialization |
| Independent observation information and nonlinear counterfactuals | [`scripts/evaluate_hbo_hbr_calibration.py`](scripts/evaluate_hbo_hbr_calibration.py): `--check-only`, `--inventory-only`, bounded `--pilot N`, then full run under a durable supervisor with `--output-dir <fresh audit child>`; `--resume` retains atomic panel results | [`hbo_hbr_calibration_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v1.yaml): A metadata inventory, B conditional nonlinear recovery with independent optical/noise records; [A–D dependency contract](../docs/EXPERIMENT_PLAN.md#独立观测信息与非线性双向反事实2026-09-18), [v2 attribution contract](configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v2.yaml) adds mean/noise factorials, oracle ablations, continuous context and profiles; actual-file audit uses `scripts/audit_hbo_hbr_calibration_sources.py`, bitmap report uses `scripts/render_hbo_hbr_calibration_report.py`; no teacher qualification | `test_hbo_hbr_calibration.py`; nonlinear sensitivities, optical/processing closure, paired counterfactuals, independent calibration, no-truth-start recovery array-free inventory, fixed-nuisance fitting, continuous common support and exact eta/matrix equivalence |

### Semantic and tokenizer development

The independent Tak-component/residual generation uses
[`scripts/run_dual_component_tokenizer.py`](scripts/run_dual_component_tokenizer.py)
and [`dual_component_tokenizer_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_tokenizer_v1.yaml).
Run `--stage check`, then `--stage synthetic --output <fresh_run>`; `--stage all`
under a durable supervisor prepares new 10 Hz targets, trains local independent
S/O vocabularies and their controls, and executes the frozen-vocabulary masked
pilot and analyses. `--stage benchmark` measures GPU throughput before choosing
concurrency. The [design](../docs/EXPERIMENT_PLAN.md#完整-tak-输出与残差的双离散词表2026-10-27)
and [data contract](../docs/DATA_CONTRACT.md#independent-tak-componentresidual-targets--2026-10-27)
define the public input and interpretation boundaries. Tests are
`test_dual_component_tokenizer.py`, `test_dual_component_tokenizer_data.py`,
`test_dual_component_tokenizer_runner.py` and
`test_dual_component_tokenizer_analysis.py`. The retained functional generation
below keeps its original model, target and evidence identity.

Rapid waveform/codebook refinement uses the same runner with
`--config configs/physiology_semantic_tokenizer/dual_component_refinement_v1.yaml
--stage refine --output <fresh_run>` under a durable supervisor. It freezes
the retained target producer and uses training/validation subjects only. The
finite controls compare continuous-checkpoint VQ recovery, an observation-space
frontend, declared wider context, and waveform-space one/two-level quantization.
All bitrate and context differences are explicit. `test_dual_component_refinement.py`
covers support closure, teacher-input exclusion, fixed budgets, hard decoding
and separate health/fidelity decisions.
The same entry accepts the versioned
[`v2`](configs/physiology_semantic_tokenizer/dual_component_refinement_v2.yaml) and
[`v3`](configs/physiology_semantic_tokenizer/dual_component_refinement_v3.yaml)
contracts for same-modality spectral/temporal frontends and finite seed repeats.
All preprocessing fits use training subjects; repeated seeds retain the same
validation set and the original quality thresholds.

Response semantics, component attribution and conditional continuous or discrete representations.

| Diagnostic / usage | Entry | Config | Targeted tests |
| --- | --- | --- | --- |
| EEG-only response dynamics, attenuation/spatial nulls and independent continuous semantic–observation prototype | `scripts/evaluate_semantic_response_discovery.py`; `scripts/train_semantic_response_tokenizer.py` | [`semantic_response_discovery_v1.yaml`](configs/physiology_semantic_tokenizer/semantic_response_discovery_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#响应动力学与连续语义原型2026-10-02) | `test_semantic_response_ssm.py`, `test_semantic_response_synthetic.py`, `test_semantic_response_discovery_runner.py`, `test_semantic_response_tokenizer.py` |
| Structured-component attribution, direction/spatial completion, known-truth counterfactuals and finite typed-token prototype; `--config <contract> --component-attribution --check-only`, then supervised `--attribution-stage` phases | [`scripts/evaluate_shared_driver_reconstruction.py`](scripts/evaluate_shared_driver_reconstruction.py); existing report renderer `--component-attribution` | [`shared_driver_attribution_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_attribution_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#结构性成分归属与分级语义检验2026-10-02) | `test_shared_driver_attribution.py`, `test_shared_driver_teacher_robustness.py` |
| TK0–TK5 conditional functional tokenizer; `--stage plan`, `synthetic`, supervised `pilot` / `all`; resume the same immutable run; strict integer evaluation `--stage hard-probe --arm QT --seed 17` | [`scripts/run_functional_tokenizer.py`](scripts/run_functional_tokenizer.py); flags `--output`, `--project-root`, `--workers`, `--devices`, `--gpu-workers` | [`functional_tokenizer_v1.yaml`](configs/physiology_semantic_tokenizer/functional_tokenizer_v1.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#条件功能监督与-hard-token-开发矩阵2026-10-10); public native measurements, independent S/O, frozen conditional teacher, hard VQ, grouped probes/nulls; no physical qualification | `test_functional_tokenizer.py`, `test_physiology_semantic_quantizer.py`, `test_ssm_output_support.py` |

Frozen semantic and relationship probes reuse the TK entry above with
`--stage relation-probe --parent-run <completed_parent> --output <fresh_run>`.
The [versioned contract](configs/physiology_semantic_tokenizer/functional_relation_probe_v1.yaml)
and [design](../docs/EXPERIMENT_PLAN.md#冻结-token-的语义读出与时空关系探针2026-10-25)
bind the retained public exports, subject splits and descriptive interpretation;
`test_functional_relation_probe.py` covers the new reader and scoring boundaries.
Run `--relation-pilot --workers 1` under user systemd before choosing concurrency
for the fixed nine-task panel. This path does not train a tokenizer or overwrite
the completed parent. The same entry's `--stage shape-probe` tests a local
conditional-waveform tokenizer with separate shape IDs and continuous scales;
its [contract](configs/physiology_semantic_tokenizer/functional_shape_probe_v1.yaml)
and `test_functional_shape_probe.py` bind the current-patch input and next-patch
Hb scoring boundary. `--shape-pilot` measures one task before the six-task run.
The final `--stage shape-relations --shape-parent-run <completed_shape_run>`
uses the [pairing contract](configs/physiology_semantic_tokenizer/functional_shape_relations_v1.yaml)
and the same shape tests to compare true history with matched temporal/regional
donors, refitting only the readout. It preserves both completed parent runs.

### Visualization and report exports

Explicit run/bundle exports and labeled synthetic illustrations; each entry retains its own input boundary.

| Diagnostic / usage | Entry | Config | Targeted tests |
| --- | --- | --- | --- |
| Four independent Chinese technical reports; `--content <report_content.json> --output <fresh_export>`; PDF/HTML, bitmap figures/formulas, selectable tables and export checks | [`scripts/render_source_diagnostics_report.py`](scripts/render_source_diagnostics_report.py); mechanism plots from [`scripts/render_mechanism_attribution.py`](scripts/render_mechanism_attribution.py) | `source_diagnostics_report_v1` content schema; explicit completed suite evidence, no inference or new data access | `test_source_diagnostics_report.py`, `test_mechanism_attribution.py` |
| Five model-compatibility quantities, step-by-step synthetic visual guide; `--output experiments/runs/physiology_semantic_tokenizer/model_compatibility_guide/<fresh_version>`; PNG, Chinese PDF/HTML, formula sources and auditable arrays | [`scripts/illustrate_model_compatibility.py`](scripts/illustrate_model_compatibility.py), using the existing technical report renderer | Renderer-owned `model_compatibility_illustration_v1`; current Balloon rest Jacobian and nonlinear forward solver, no dataset/run reads; descriptive normalized closure energies and explicitly illustrative tolerance, not a measured-data test | `test_model_compatibility_illustration.py`, `test_source_diagnostics_report.py` |
| Spatial visualization: `--demo` retains local forward examples; `--whole-head-demo` uses the public REFED drawing layout with synthetic signals; `--bundle <manifest.json>` renders an explicit export. Completed experimental reports use `--experiment-report --measured-run <completed_public_run> --synthetic-run <completed_synthetic_run>` for PPT, PDF and companion technical documents. All modes require `--output <fresh_export>`. | [`scripts/render_shared_driver_spatial.py`](scripts/render_shared_driver_spatial.py) | Renderer-owned `shared_driver_spatial_visualization_v1/v2` JSON/NPZ contracts. Full-head sensor/optode maps, coverage-limited heatmaps, Pearson and controlled associations, conditional lag-prediction maps, and separately labelled generator intervention examples; complete matrices retained alongside display-only edge selection. PNG figures and searchable bitmap PDF. Public layout metadata are not individual cortical registration; demo signals and causal mechanisms are synthetic. Visualization capacity does not enlarge experiment scope. | `test_shared_driver_spatial_visualization.py`, `test_shared_driver_spatial_report.py` |
| Offline physiology parameter animation; `--parameter-explorer --run <completed_conditional_optical_gain_run> --output <fresh_export>`; optional `--boundary-run <completed_fixed_roi_optical_run>` | Existing [`renderer`](scripts/render_shared_driver_reconstruction.py) and [HTML template](scripts/shared_driver_parameter_explorer.html); ManimGL environment: `manimgl`, `imageio-ffmpeg`, LaTeX/dvisvgm and EGL | Retained public fold-3 AF7Fp1 targets; deterministic low/middle/high-error selection across distinct subjects; forward-only τ/κ/effective-gain sweeps with fixed fitted driver and initial states; optional MNE TDDR shared-τ lower-bound example with retained optimizer trace, separate scales and excluded out-of-bound nodes; no refitting or new raw reads; the exported `data.json` can also supply separately authored ManimGL animations | `test_shared_driver_fitting_explainer.py`, `test_shared_driver_compiled.py` |
| [Overnight report](#overnight-observation-diagnostics) | [`scripts/render_ssm_overnight_report.py`](scripts/render_ssm_overnight_report.py) | explicit completed run and fresh report directory | [`test_ssm_overnight_diagnostics.py`](../tests/test_ssm_overnight_diagnostics.py) |

### T3, Step5 and overnight diagnostic replay

Recorded diagnostic generations with detailed usage below; each retains its original split, gate and evidence identity.

| Diagnostic / usage | Entry | Config | Targeted tests |
| --- | --- | --- | --- |
| [Overnight observation diagnostics](#overnight-observation-diagnostics) | [`evaluate_ssm_overnight_diagnostics.py`](evaluate_ssm_overnight_diagnostics.py) | [`ssm_overnight_v3.yaml`](configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml); retained [`v1`](configs/physiology_semantic_tokenizer/ssm_overnight_v1.yaml), [`v2`](configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml) | [`test_ssm_overnight_diagnostics.py`](../tests/test_ssm_overnight_diagnostics.py) |
| [Synthetic P0](#synthetic-p0) | [`evaluate_t3a_balloon_robust_p0.py`](evaluate_t3a_balloon_robust_p0.py) | [`t3a_balloon_robust_p0.yaml`](configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml) | [`test_t3a_balloon_robust_p0.py`](../tests/test_t3a_balloon_robust_p0.py), [`test_t3a_balloon_robust_ssm.py`](../tests/test_t3a_balloon_robust_ssm.py) |
| [Measured reconstruction/null](#measured-reconstruction-and-null) | [`evaluate_t3_measured_reconstruction_null.py`](evaluate_t3_measured_reconstruction_null.py) | [`t3_measured_reconstruction_null_v1.yaml`](configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml) | [`test_t3_measured_reconstruction_null.py`](../tests/test_t3_measured_reconstruction_null.py) |
| [Identifiability](#fit-only-identifiability) | [`evaluate_t3_identifiability.py`](evaluate_t3_identifiability.py) | [`t3_identifiability_v1.yaml`](configs/physiology_semantic_tokenizer/t3_identifiability_v1.yaml) | [`test_t3_identifiability.py`](../tests/test_t3_identifiability.py) |
| [Multi-session LOSO](#fit-only-multi-session-loso) | [`evaluate_t3_multisession_loso.py`](evaluate_t3_multisession_loso.py) | [`t3_multisession_loso_v1.yaml`](configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml) | [`test_t3_multisession_loso.py`](../tests/test_t3_multisession_loso.py) |
| [T3c admission](#t3c-admission) | [`evaluate_t3c_hierarchical_composite_admission.py`](evaluate_t3c_hierarchical_composite_admission.py) | [`t3c_hierarchical_composite_admission_v1.yaml`](configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml) | [`test_t3c_hierarchical_admission.py`](../tests/test_t3c_hierarchical_admission.py) |
| [T3c synthetic composite](#t3c-synthetic-composite) | [`evaluate_t3c_composite_synthetic_t2.py`](evaluate_t3c_composite_synthetic_t2.py) | [`t3c_composite_synthetic_t2_v1.yaml`](configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml) | [`test_t3c_composite_synthetic_t2.py`](../tests/test_t3c_composite_synthetic_t2.py) |
| [Step5A localization](#step5a-inference-localization) | [`evaluate_step5a_inference_consistency.py`](evaluate_step5a_inference_consistency.py) | [`step5a_inference_consistency_v1.yaml`](configs/physiology_semantic_tokenizer/step5a_inference_consistency_v1.yaml) | [`test_step5a_inference_consistency.py`](../tests/test_step5a_inference_consistency.py) |
| [Step5 stages](#step5-stages) | [`evaluate_step5.py`](evaluate_step5.py) | [`step5_v1.yaml`](configs/physiology_semantic_tokenizer/step5_v1.yaml), [`step5b_v2.yaml`](configs/physiology_semantic_tokenizer/step5b_v2.yaml); retained [`step5b_v1.yaml`](configs/physiology_semantic_tokenizer/step5b_v1.yaml) | [`test_step5.py`](../tests/test_step5.py), [`test_t3a_balloon_joint_ssm.py`](../tests/test_t3a_balloon_joint_ssm.py) |
| [Observation diagnostic](#step5-observation-diagnostic-and-repair) | [`evaluate_step5_observation_diagnostic.py`](evaluate_step5_observation_diagnostic.py) | [`step5_observation_diagnostic_v1.yaml`](configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml) | [`test_step5_observation_diagnostic.py`](../tests/test_step5_observation_diagnostic.py) |
| [Observation repair](#step5-observation-diagnostic-and-repair) | [`evaluate_step5_observation_repair.py`](evaluate_step5_observation_repair.py) | [`step5_observation_repair_v1.yaml`](configs/physiology_semantic_tokenizer/step5_observation_repair_v1.yaml), [`v2`](configs/physiology_semantic_tokenizer/step5_observation_repair_v2.yaml) | [`test_step5_observation_repair.py`](../tests/test_step5_observation_repair.py) |

For data/cache utilities and retained R-series commands, use the
[source map](../src/README.md), [test map](../tests/README.md) and
[retained result index](RESULTS_INDEX.md). Their presence is not a new run queue.

## Recorded state

This workspace does not maintain a second status summary. Query the generated
[`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) or run:

```bash
.venv/bin/python experiments/scripts/project_state.py show --format agent
```

Live-process checks must be performed immediately before touching generated run
or cache directories.

## Retained T3, Step5 and observation entries

The following commands retain their dated inlet-balance inference and data
contracts. New shared-driver model work uses the Tak entry above; old teacher,
stochastic and closure interfaces are not automatically Tak models.

The following sections explain recorded command interfaces. Their lifecycle is
owned by the registry view linked above.

### Overnight observation diagnostics

The observation-contract diagnostic follows the [v3 plan](../docs/EXPERIMENT_PLAN.md)
and [v3 configuration](configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml).
Use the same overnight entry with `--config .../ssm_overnight_v3.yaml` for
`--check-only` and `--prepare`. V3 preparation uses synthetic pilots and a
signal-free identity inventory; native preparation is a queued stage after the
Gaussian mean precheck. Freeze and launch the snapshot as described below.
The run's automatic report includes stage gates, fixed failure denominators,
feature-missing comparisons and independent adaptation panels.
For a completed v3 controller, the existing report renderer accepts
`--run-dir <v3_run> --output-dir <v3_run>/<new_report_version>` without a previous
run. It exports Chinese Markdown/HTML/PDF, figures, planned-fit failure
attribution and missing-support metrics reconstructed from saved trajectories.

The bounded overnight N1–N7 diagnostic follows the
[retained v2 protocol](../docs/EXPERIMENT_PLAN.md#bounded-overnight-ssm-diagnostics-retained-v2-contract)
and [`ssm_overnight_v2.yaml`](configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml).
Its entry is [`evaluate_ssm_overnight_diagnostics.py`](evaluate_ssm_overnight_diagnostics.py).
Use `--check-only`, then `--prepare --run-dir <fresh_run>`, then `--freeze` for
the same run. Launch the saved `source_snapshot/experiments/` entry with `--run`
under a persistent service, setting `SSM_PROJECT_ROOT` to this repository. The
registered artifact root is `runs/physiology_semantic_tokenizer/ssm_overnight/`.
The run manifest and fixed task/status tables own execution; its
`OVERNIGHT_REPORT.md` summarizes all seven families without granting teacher
qualification. The native measured O2 branch is explicitly unavailable while
the existing helper does not expose its required pre-linear noise boundary.

Render a separate Chinese report with complete candidate panels, embedded-image
HTML, Markdown, a PDF report, PNG/SVG figures and a PDF figure atlas using
[`scripts/render_ssm_overnight_report.py`](scripts/render_ssm_overnight_report.py):
`--run-dir <completed_N1-N7_run> --previous-run-dir <retained_N1-N6_run>
--output-dir <completed_N1-N7_run>/<report_version>`.
This reads retained result tables and checks fixed identities; it does not rerun
models or replace either run's frozen evidence or automatic report.
The [retained visual report and publication scope](RESULTS_INDEX.md#overnight-evidence-snapshot--2026-09-10)
provide the reader-facing evidence entry. Rebuilding figures requires the local
per-task evidence, which is excluded from the published package.
Install the Python dependencies in `../requirements.txt` and the system Noto
CJK font (`fonts-noto-cjk` on Debian/Ubuntu) to reproduce the Chinese PDF figures.

The measurement revision uses the same `--prepare`, `--freeze`, frozen `--run`
sequence with `ssm_measurement_alignment_v3.yaml`. Before preparation, build its
scoped native inputs with `build_clean_eeg_fnirs_cache.py --ssm-training-config
experiments/configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml`.
This writes a fresh native-only cache; it does not perform continuous-record
processing over excluded SSM trials. Generic public measurement caches use the
signal builder's `--output-dir`, the event builder's matching
`--output-dir <cache>/event_index`, and the geometry builder's
`--output-dir <cache>/channel_geometry`.

The A–E measurement follow-up uses the same overnight entry and
[`ssm_measurement_retest_v1.yaml`](configs/physiology_semantic_tokenizer/ssm_measurement_retest_v1.yaml).
It references completed native feature/fold evidence explicitly, adds fixed W,
single-chromophore, loading and W-curve diagnostics, and conditionally screens
gain/Q. Its scope and interpretation are owned by the
[2026-09-17 protocol](../docs/EXPERIMENT_PLAN.md#新测量链固定模型w-曲线与配对增量重测2026-09-17).
Export its completed evidence with
[`render_ssm_measurement_retest_report.py`](scripts/render_ssm_measurement_retest_report.py)
using `--run-dir <completed_run> --output-dir <fresh_versioned_export>`;
both delivered PDFs embed PNG figures and retain selectable text.

### Default measurement cache

The signal/event/geometry builders share `DEFAULT_CLEAN_CACHE_ROOT`, whose
processing and version boundary are defined in the [data contract](../docs/DATA_CONTRACT.md#无运动校正的通用缓存--2026-09-28):

```bash
.venv/bin/python experiments/build_clean_event_index.py
.venv/bin/python experiments/build_clean_channel_geometry.py
# Full signal builds run under a durable supervisor; choose workers after a
# CPU/memory/IO check, and use a fresh unit name for each build attempt.
systemd-run --user --unit=physiology-cache-v5 --property=Type=exec \
  --property=WorkingDirectory="$PWD" --property=RemainAfterExit=yes \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  "$PWD/.venv/bin/python" -u "$PWD/experiments/build_clean_eeg_fnirs_cache.py" \
  --processing-schema physiology_measurement_alignment_v5 --motion-method none --workers 16
```

Verify `loginctl show-user "$USER" -p Linger` reports `Linger=yes` before using
the user service for disconnect-safe work. Monitor it with `systemctl --user
status physiology-cache-v5` and `journalctl --user -u physiology-cache-v5`.
The worker count is an example, not a resource limit; constrain it by measured
throughput and memory, and retain launch/resource details with the owning run.

The signal builder persists float64 EEG and HbO/HbR as memory-mapped record arrays;
all 29 published Single-Trial subjects are eligible under the current data contract.
Use explicit subject/record
limits for a smoke build. These commands prepare data, not models or evaluations.
The local-view factory defaults to float64 measurement inputs without cross-dataset
amplitude normalization. Each SSM/tokenizer consumer owns its training-fitted
scaling and computation dtype.
Old experiment configs retain their historical cache identities.

### Synthetic P0

The synthetic qualification entry is
[`t3a_balloon_robust_p0.yaml`](configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml).
It uses synthetic data only and exercises `T0-native`, `T1-self`,
`T2b-adaptive-legacy`, and the primary `T3a-balloon-robust` candidate. Run a
small software diagnostic and render its Chinese figures with:

```bash
.venv/bin/python experiments/evaluate_t3a_balloon_robust_p0.py --smoke --output-dir experiments/runs/physiology_semantic_tokenizer/t3a_balloon_robust_p0/20260827_smoke_v1
.venv/bin/python experiments/scripts/render_t3a_balloon_robust_p0.py --run-dir experiments/runs/physiology_semantic_tokenizer/t3a_balloon_robust_p0/20260827_smoke_v1
```

`--smoke` can never qualify the model. A formal synthetic P0 uses the same
entry without `--smoke` and a fresh run directory. The executable panel does
not yet claim the `T2a-croce-pf` or `T4-dcm-lite` design references were tested.

### Measured reconstruction and null

The nonprotected measured-development diagnostic is
[`t3_measured_reconstruction_null_v1.yaml`](configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml)
with entrypoint
[`evaluate_t3_measured_reconstruction_null.py`](evaluate_t3_measured_reconstruction_null.py).
It reads the canonical measured EEG/fNIRS loader with
`raw_with_ocular_artifact`, fits all data-dependent objects on subjects 01--18,
and applies them without refitting to subjects 19--23. Subjects 24--29 remain
closed. The experiment compares observation reconstruction and independent,
pairing, and time-shift nulls; its outputs are exploratory measured evidence,
not clean ground truth, model qualification, or a physical-teacher claim.
It cannot open protected data or authorize tokenizer/VQ promotion.

Run the complete registered diagnostic into a fresh workspace directory with:

```bash
.venv/bin/python experiments/evaluate_t3_measured_reconstruction_null.py \
  --run-dir experiments/runs/physiology_semantic_tokenizer/t3_measured_reconstruction_null/<fresh_run_id>
```

The primary reconstruction rows are the four-second center blocks withheld
from the target modality, with NRMSE normalized by the observed SD on that same
valid block. The EEG coordinate is the existing 10 Hz fit-fold log-power/PCA
proxy rather than the native 200 Hz voltage waveform; HbO/HbR are canonical
standardized coordinates rather than absolute concentrations. Same-point joint
smoothing is retained only as a posterior-fit description. Null results report
EEG-only pairing specificity separately from donor leakage into the T2b/T3a
joint shared state.

### Fit-only identifiability

The second-step fit-only identifiability diagnostic is registered in
[`t3_identifiability_v1.yaml`](configs/physiology_semantic_tokenizer/t3_identifiability_v1.yaml)
with entrypoint
[`evaluate_t3_identifiability.py`](evaluate_t3_identifiability.py). It first
runs one known-truth synthetic case. Only after that completes does it fit the
01--18 observation gauge, select fixed-M0 low/median/high residual subjects,
and run 16 transformed-space starts, true profile likelihood, expanded-bound
refits, and conditional forward sensitivity SVD on their eight fit trials.
The shared loader still constructs canonical dataset-index metadata and window
references, but no arrays are loaded and no window samples are materialized for
subjects 19--29. The output is exploratory and cannot qualify a teacher or
authorize tokenizer promotion.

```bash
.venv/bin/python experiments/evaluate_t3_identifiability.py \
  --run-dir experiments/runs/physiology_semantic_tokenizer/t3_identifiability/<fresh_run_id>
```

### Fit-only multi-session LOSO

The third-step fit-only multi-session diagnostic is registered in
[`t3_multisession_loso_v1.yaml`](configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml)
with entrypoint
[`evaluate_t3_multisession_loso.py`](evaluate_t3_multisession_loso.py). It uses
only subjects 01--18 and cache records `session_01/03/05`, fits two complete
sessions per fold, and freezes all fit-dependent objects before target-masked
apply to the third session. The common window is `[-5,+25) s`; the primary
score uses the nominal recovery envelope `[+10,+25) s`. Event durations are
not indexed, so this is not labelled an exact annotated rest period. The run
is exploratory, decision-ineligible, and cannot open subjects 19--29.

```bash
.venv/bin/python experiments/evaluate_t3_multisession_loso.py \
  --config experiments/configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml \
  --run-dir experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/<fresh_run_id>
```

### T3c admission

The fourth-step `T3c` entry is an array-free admission gate, not a
measured hierarchy launcher. It checks frozen Step 2/3 evidence, the analytic
gain/time coordinate, and the Normal–Normal shrinkage primitive. If any
prerequisite is absent it records `BLOCKED_PREREQUISITE` before measured
metadata or arrays are read.

```bash
.venv/bin/python experiments/evaluate_t3c_hierarchical_composite_admission.py \
  --config experiments/configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml \
  --run-dir experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/<fresh_run_id>
```

### T3c synthetic composite

The fourth-step known-truth composite `T-P2` screen is registered separately in
[`t3c_composite_synthetic_t2_v1.yaml`](configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml)
with entrypoint
[`evaluate_t3c_composite_synthetic_t2.py`](evaluate_t3c_composite_synthetic_t2.py).
It runs 60 independent replicates for each one-dimensional gain/time direction,
passes only noisy training observations to the fitter, and opens no measured,
validation, or protected data. The formal v1 result is
`BLOCKED_C1_COMPOSITE_IDENTIFIABILITY`: both C1 directions failed, so C2 remains
`NOT_RUN_C1_GATE_NOT_MET` and measured hierarchy remains blocked. The manifest
owns run state; the retained
[`detailed report`](../docs/analysis/20260903_T3C_COMPOSITE_SYNTHETIC_TP2_REPORT.md)
owns the human-readable interpretation.

```bash
.venv/bin/python experiments/evaluate_t3c_composite_synthetic_t2.py \
  --config experiments/configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml \
  --run-dir experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/<fresh_run_id>
```

### Step5A inference localization

The synthetic inference-localization entry is
[`evaluate_step5a_inference_consistency.py`](evaluate_step5a_inference_consistency.py),
with the versioned
[`Step5A configuration`](configs/physiology_semantic_tokenizer/step5a_inference_consistency_v1.yaml).
The [owning protocol](../docs/EXPERIMENT_PLAN.md#step5a0-inference-consistency-diagnostic)
defines matched generation, known-driver likelihood, particle reference,
separate mismatch stress tests, and conditional state-coverage diagnostics.
Run the software checks and then the bounded panel in a fresh directory:

```bash
.venv/bin/python experiments/evaluate_step5a_inference_consistency.py --check-only
.venv/bin/python experiments/evaluate_step5a_inference_consistency.py \
  --run-dir experiments/runs/physiology_semantic_tokenizer/step5a_inference_consistency/<fresh_run_id> \
  --workers 3
```

The entry writes resolved configuration, source identities, per-case scores and
truth arrays, reference precision diagnostics, and JSON/Markdown summaries.
It cannot confer teacher qualification or open measured/protected data.
An explicit completed panel can receive a deterministic oracle grid check via
`--oracle-refinement-of <original_run_dir> --run-dir <fresh_run_dir>`.
It preserves the same cases and thresholds and adds no independent replicates.

### Step5 stages

The full staged Step5 entry is
[`evaluate_step5.py`](evaluate_step5.py), using
[`step5_v1.yaml`](configs/physiology_semantic_tokenizer/step5_v1.yaml) and the
[staged protocol](../docs/EXPERIMENT_PLAN.md#full-step5-staged-continuation).
It separates joint parameter likelihood, held-out teacher qualification and
subsequent measured/UQ prerequisites. Reference inputs are the explicitly
retained localization run named in the configuration. Each experiment stage
requires a fresh direct child of the configured Step5 artifact root:

```bash
.venv/bin/python experiments/evaluate_step5.py --stage reference --run-dir <fresh_reference_dir>
.venv/bin/python experiments/evaluate_step5.py --stage a0 \
  --reference-run <reference_dir> --run-dir <fresh_calibration_dir>
.venv/bin/python experiments/evaluate_step5.py --stage a1 \
  --calibration-run <calibration_dir> --run-dir <fresh_teacher_dir> --workers 48
.venv/bin/python experiments/evaluate_step5.py --stage review \
  --teacher-run <teacher_dir> --run-dir <fresh_candidate_scope_review_dir>
.venv/bin/python experiments/evaluate_step5.py --stage b \
  --teacher-run <candidate_scope_review_dir> --run-dir <fresh_measured_dir> \
  --measured-config experiments/configs/physiology_semantic_tokenizer/step5b_v2.yaml
.venv/bin/python experiments/evaluate_step5.py --stage report --run-dir <completed_unqualified_measured_dir>
```

The review command applies candidate-specific case completeness to existing
A1 outcomes, without new inference. The B command checks that candidate's
synthetic qualification before metadata or native-array access. Its optional
`--reuse-run` accepts only the recorded v1-to-v2 refinement or an identical
configuration with unchanged scientific functions and hashed retained outputs.
The report command adds stage diagnostics and an explicit unexecuted UQ decision
when measured core qualification fails; it reads saved results only. Case
exceptions retain their identities and tracebacks. Successful subsets cannot
replace a candidate's complete registered experiment.

### Step5 observation diagnostic and repair

The observation diagnostic uses
[`evaluate_step5_observation_diagnostic.py`](evaluate_step5_observation_diagnostic.py);
the repair and mask-bridge comparisons use
[`evaluate_step5_observation_repair.py`](evaluate_step5_observation_repair.py).
Choose the versioned config in the map above and consult the owning
[diagnostic](../docs/EXPERIMENT_PLAN.md#step5-observation-adaptation-diagnostic-retained-v1-contract),
[repair v1](../docs/EXPERIMENT_PLAN.md#step5-observation-contract-repair-and-regression-retained-v1),
or [mask-bridge](../docs/EXPERIMENT_PLAN.md#step5-flow-domain-and-mask-specific-observation-experiment)
section before using the CLI. Source and native/replay data roots remain separate
when an overnight command runs from its frozen snapshot.

## Frozen method boundary and implementation candidates

The theory/architecture principles are retained in
[`METHOD_RATIONALE.md`](../docs/METHOD_RATIONALE.md). The v2 exploration is an
abandoned, not-yet-implemented pre-freeze candidate map. It is
recorded in the [design note](../docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json)
and its [framework diagram](../docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg).
No YAML or measured-data run is authorized by those artifacts.

Except for the entries above, the existing physiology-semantic
YAML/runtime surface is stopped historical and replay-only; do not clone or
reinterpret it as a new contract. An implementation inside the frozen boundary must first
pass synthetic software, target/teacher, tensor-shape, split, and null checks.
The measured diagnostic remains nonprotected and decision-ineligible; protected
data requires the owning protocol and a separate, explicit user authorization
for that measured action.

## Run contract

Any future registered physiology-semantic output uses:

```text
experiments/runs/physiology_semantic_tokenizer/<suite>/<immutable-run>/
```

A result is auditable only when its resolved config, declared split/cache/schema
identity, non-hash runtime versions, completion status,
primary endpoint, summary/table, and necessary prediction or figure source
data are present. Suite summaries do not override run records; file/data hashes
are not part of this contract.

Generated payloads remain ignored by Git. Lightweight decision summaries and
evidence indexes may be force-tracked intentionally. Historical
analysis always names an exact archive path; active tools never recurse through
archives.

Launch boundaries are in [`../docs/EXPERIMENT_PLAN.md`](../docs/EXPERIMENT_PLAN.md);
test conventions are in [`../tests/README.md`](../tests/README.md).
