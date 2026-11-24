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
| Compare downstream task-label performance | [Comparison methods](../comparative_methods/README.md), [new rebenchmark design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1), [retained comparison protocol](../docs/comparisons/PROTOCOL.md), and [dated comparison report renderer](scripts/render_downstream_comparison_report.py) (`--summary`, `--paper-values`, fresh `--output`; report only, no fitting) |
| Assemble the interpretability presentation | [Presentation renderer](scripts/render_interpretability_presentation.py) (`--output <fresh_versioned_export>`): explicit retained figures and summaries, editable PPTX, bitmap-figure PDF and per-slide source audit; no dataset loading or fitting |

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

**Four-bit word stability.** [Nightly CPU suite](scripts/run_observation_word_stability.py) with its [fixed development contract](configs/physiology_semantic_tokenizer/observation_word_stability_v1.yaml) runs `--stage preflight`, then supervised `--stage screen` at the same fresh `--output` below the observation suite. Completed refit cells can be resumed. [Synthetic and contract tests](../tests/test_observation_word_stability.py) cover the shared draw, split boundary, integer interface and paired verdict.

The [Hb coordinate simulation](scripts/run_hb_word_coordinate_simulation.py) isolates rank transformations of levels and slopes against known generated slopes. Pass a fresh `--output` ending in `_hb_coordinate_sim_vN` inside the observation suite. [Contract](configs/physiology_semantic_tokenizer/hb_word_coordinate_simulation_v1.yaml) · [synthetic tests](../tests/test_hb_word_coordinate_simulation.py).

**Matched detail allocation.** [`run_observation_detail_allocation.py`](scripts/run_observation_detail_allocation.py) (`--stage synthetic`, then `screen`) replays the [fixed parent panel and budgets](configs/physiology_semantic_tokenizer/observation_detail_allocation_v1.yaml); [allocation tests](../tests/test_adaptive_observation_detail.py) and [replay tests](../tests/test_observation_allocation_screen.py). Its report separates the same 4-bit observation words from the train-only allocation of the remaining integer bits. The [final EEG gain/shape ablation](scripts/run_observation_gain_shape.py) ([contract](configs/physiology_semantic_tokenizer/observation_gain_shape_v1.yaml), [tests](../tests/test_observation_gain_shape_screen.py)) charges explicitly for the transmitted DC/AC levels at the same three full budgets. The [bundle exporter](scripts/export_observation_baseline.py) makes a portable CPU-only consumer from the frozen selected models, not a training launcher.

**Independent observation-pattern baseline.** [`run_observation_factorization.py`](scripts/run_observation_factorization.py) with the [fixed versioned config](configs/physiology_semantic_tokenizer/factorized_observation_v1.yaml) runs `--stage synthetic`, `smoke`, then `screen` against the same fresh `--output` within `runs/physiology_semantic_tokenizer/observation_factorization/`. It freezes training dictionaries, retains paired rate–distortion and background-controlled attribute tables plus a genuine event readout; it does not modify the stopped or current dual-component launchers. Tests: [`codec`](../tests/test_factorized_observation.py), [`panel`](../tests/test_observation_tokenizer_panel.py), [`legacy replay`](../tests/test_observation_legacy_control.py), [`evaluation`](../tests/test_observation_factor_screen.py), [`task`](../tests/test_observation_task_probe.py).

[Frozen-word physiology probes](scripts/run_observation_relations.py) ([contract](configs/physiology_semantic_tokenizer/observation_relations_v1.yaml)): run `--stage preflight`, then `--stage screen`, both with `--output` pointing to the same fresh `..._relations_vN` run in the observation suite. Checks include native clocks and patch identity, same-modality word order against exact within-window permutations, delayed EEG-to-Hb controls and real-event occupancy. No tokenizer fit. Tests: `tests/test_observation_relations_screen.py` and the four `tests/test_observation_token_{phenotypes,dynamics,coupling,events}.py` modules.

The [Chinese report and figures](runs/physiology_semantic_tokenizer/observation_factorization/20261120_relations_v2/REPORT.md), all 32 words and the [subject-level verification](runs/physiology_semantic_tokenizer/observation_factorization/20261120_relations_v2/verification.json) can be checked without data caches. The [renderer](scripts/render_observation_relations.py) takes `--run` pointing to a copy of that completed run under the same suite and recalculates the figures and decision from the retained tables. Data-free preflight uses the four hash-checked parent dictionaries. A new **measured** run additionally requires the original G0 and public development event caches plus the parent `20261120_development_v3/task_probe.json` (SHA-256 `2060df9dfd28a67b937cb34b5b0c4e698850ff556e246c27886a836aefdc5e7d`) for an exact event-identity comparison. Do not substitute the new run’s own event list for this independent reference. Raw waveforms and historical per-trial predictions are not redistributed here.

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
| Four-dataset downstream input freeze; metadata `--downstream-stage plan`, supervised `pilot`/`build`, retained-input `verify`; corrected Visual native channels, raw-voltage record preprocessing and unchanged cached Hb | [`build_clean_eeg_fnirs_cache.py`](build_clean_eeg_fnirs_cache.py) with `--downstream-config <contract>`; consumer: [`DownstreamInputDataset`](../src/data/downstream_input_cache.py) | [`downstream_input_v1.yaml`](configs/physiology_semantic_tokenizer/downstream_input_v1.yaml); [input/split/label contract](../docs/DATA_CONTRACT.md#four-dataset-downstream-input-cache--2026-11-04) | `test_visual_channel_identity.py`, `test_downstream_input_cache.py`, `test_unified_physiology.py`, `test_clean_cache_builder.py`, `test_cached_fnirs_input.py` |
| Visual-only measurement v6 / downstream v2 migration; same `--downstream-stage <plan|pilot|build|verify>`; compact usable Hb columns and frozen subject normalization | Existing [`build_clean_eeg_fnirs_cache.py`](build_clean_eeg_fnirs_cache.py) and [`downstream_input_cache.py`](../src/data/downstream_input_cache.py); completed-cache report: [`analyze_tak_cache_versions.py`](scripts/analyze_tak_cache_versions.py) `--visual-cache-review <v2_cache> --export <fresh_cache_child>` | [`downstream_input_v2.yaml`](configs/physiology_semantic_tokenizer/downstream_input_v2.yaml); [measurement and normalization coordinates](../docs/DATA_CONTRACT.md#visual-measurement-v6--downstream-input-v2--2026-11-10); other datasets and sample identities preserved; no SSM refits | `test_visual_cache_migration.py`, `test_downstream_input_cache.py`, `test_cached_fnirs_input.py`, `test_unified_physiology.py`, `test_clean_cache_builder.py`, `test_visual_channel_identity.py` |
| General public measurement cache; explicit versioned output root, signal → event → geometry builders under a durable supervisor | [`build_clean_eeg_fnirs_cache.py`](build_clean_eeg_fnirs_cache.py), [`build_clean_event_index.py`](build_clean_event_index.py), [`build_clean_channel_geometry.py`](build_clean_channel_geometry.py) | [Data contract](../docs/DATA_CONTRACT.md#无运动校正的通用缓存--2026-09-28); CLI `--processing-schema physiology_measurement_alignment_v5 --motion-method none`; MNE is an explicit optical comparison in a separate root. Metadata-only REFED layout ([paper extraction](scripts/extract_refed_paper_layout.py)): `--datasets refed --refed-geometry paper_layout_v1 --output-dir <fresh_geometry_root>`, consumed via `geometry_root`; [planar geometry contract](../docs/DATA_CONTRACT.md#refed-paper-layout-registration--2026-10-13) | `test_clean_cache_builder.py`, `test_clean_physiology_cache.py`, `test_homer2_measurement_v5.py`, `test_homer2_measurement_v4.py`, `test_unified_physiology.py`, `test_event_alignment.py`, `test_channel_geometry.py`, `test_physiology_semantic_local.py` |
| Observation coordinate stability, native EOG events, hardware donor contrasts and matched synthetic mechanisms; `--stage check`, supervised `pilot` / `all`, bounded `refinement`, retained-array `verify` | [`scripts/evaluate_observation_semantics.py`](scripts/evaluate_observation_semantics.py); [`PPT/PDF renderer`](scripts/render_observation_semantics_report.py) | [`v1`](configs/physiology_semantic_tokenizer/observation_semantics_v1.yaml), [`C/D v2`](configs/physiology_semantic_tokenizer/observation_semantics_v2.yaml); [protocol](../docs/EXPERIMENT_PLAN.md#观测物理量事件锚点与竞争机制2026-10-09) | `test_observation_semantics.py` |
| Measurement precision and unit revision; fixed v3 stage 1/2 | [`evaluate_ssm_overnight_diagnostics.py`](evaluate_ssm_overnight_diagnostics.py) | [`ssm_measurement_alignment_v3.yaml`](configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml) | `test_ssm_overnight_diagnostics.py`, `test_step5_observation_diagnostic.py`, `test_step5.py`, `test_homer2_preprocessing.py` |
| Four-dataset amplitude audit and alignment repair reports (`--stage alignment-report --output-dir <new versioned export>`) | [`scripts/analyze_dataset_scaling.py`](scripts/analyze_dataset_scaling.py) | [`dataset_scaling_report_v1.yaml`](configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml); alignment report reads the retained [measurement v3](configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml) evidence | [`test_dataset_scaling_report.py`](../tests/test_dataset_scaling_report.py) |
| Public native Hb direction prevalence; `--prevalence --stage synthetic`, then supervised `--stage measured --workers N`; optional `--prevalence-pilot`; `--stage report` creates a versioned bitmap export | [`scripts/analyze_hbo_hbr_relationship.py`](scripts/analyze_hbo_hbr_relationship.py); native fNIRS reader in `src/data/unified_physiology.py` | [`hbo_hbr_prevalence_v1.yaml`](configs/physiology_semantic_tokenizer/hbo_hbr_prevalence_v1.yaml): four public datasets, native/released Hb versus v5 none, subject-weighted census, trend and optical-coefficient sensitivity; no model fitting | `test_hbo_hbr_relationship.py`, `test_unified_physiology.py`, `test_clean_physiology_cache.py`; paired-mask statistics, native identity/support, subject weighting, synthetic exclusion |
| Read-only optical motion pipeline audit; metadata `--check-only`, then separate synthetic/measured phases on the exact retained public windows; no model fits | [`scripts/analyze_fnirs_motion_correction.py`](scripts/analyze_fnirs_motion_correction.py) | [`optical motion audit`](configs/physiology_semantic_tokenizer/shared_driver_optical_motion_audit_v1.yaml) | `test_fnirs_motion_correction_audit.py`, `test_homer2_preprocessing.py`, `test_homer2_measurement_v4.py` |

The downstream-only cache has an explicit consumer; it does not change the SSM
input root or the retained benchmark loaders:

```python
from src.data.downstream_input_cache import DownstreamInputDataset

samples = DownstreamInputDataset(
    "data/cache/physiology_downstream_v1", split="train", task="motor_imagery"
)
sample = samples[0]  # EEG/Hb, masks, clocks, channel identities and task target
```

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
directories. The metadata-only `plot-plan` and supervised `plot-run` stages use
the bounded, coverage-first panel at `plot_priority_v1/plan.json`; the original
group fits and evidence identities are unchanged. The renderer's explicit
`--priority-panel` mode can export pending/failed coverage and matched results
before the full campaign finishes, to a fresh directory on each invocation.
Tests: `test_tak_mismatch.py`, `test_tak_mismatch_data.py`,
`test_tak_mismatch_runner.py`, `test_tak_mismatch_current_dof.py`,
`test_tak_mismatch_report.py`; independent replay equivalence is covered by
`test_shared_driver_replay_equivalence.py`.

The Visual channel repair uses the same entry with
[`tak_mismatch_visual_corrected_v2.yaml`](configs/physiology_semantic_tokenizer/tak_mismatch_visual_corrected_v2.yaml).
It freezes the publisher workbook's native EEG identity (including A64/FCz),
joins the existing geometry by electrode name, and recomputes native EEG power,
training-fold coordinates, spatial selection and both parameter modes for all
admitted Visual records and Hb pairs. The retained v5 cache supplies only its
unchanged Hb arrays, source identities and clock metadata; its EEG names/arrays
are not relabelled or consumed. The new prepared arrays own the corrected input
identity. Other datasets' retained results remain separate evidence sources.
Tests: `test_visual_channel_identity.py`, `test_tak_mismatch_data.py` and
`test_tak_mismatch_runner.py` check mapping, retained-Hb provenance and rejection
of channel drift before signal access.
`fixed-run` completes only the full fixed stage; `plot-plan` / `plot-run` then
prioritize the unchanged three-pair shared-parameter panel. The corrected
[`fixed analysis contract`](configs/physiology_semantic_tokenizer/tak_mismatch_visual_physiology_v2.yaml)
uses the existing `physiology-*` stages. `physiology-four-dataset`, with an
explicit retained `--reference-run`, replaces only Visual in the four-dataset
descriptive synthesis and requires a completed common shared-parameter panel.
`render_tak_mismatch.py --four-dataset-analysis <analysis> --run <corrected-run>
--output <fresh-export>` produces the corresponding bitmap PDF, HTML and atlas.
For a fixed-parameter census report, add `--fixed-only` to
`physiology-four-dataset` and omit `--panel-export`. This reads only the completed
fixed scalar/shape/spatial owners, replaces Visual, and examines all-subject
regional contrasts, record-map stability and repeated high-mismatch locations.
Top-quarter overlap and per-location persistence keep their missing-record
denominators; they do not assume neighbouring channels are independent. The same
renderer accepts this fixed-only summary and does not resume shared-parameter fits.
The all-subject disagreement comparison uses this same stage with
`--fixed-only --disagreement-config <contract> --pattern-output <fresh_child> --workers 4`.
[`tak_disagreement_comparison_v1.yaml`](configs/physiology_semantic_tokenizer/tak_disagreement_comparison_v1.yaml)
matches retained windows before comparing Hb reconstruction residuals, independent
drive Hb replay gaps, their common-amplitude ratios and EEG readout disagreement.
It preserves failed support, strict window-coverage maps, paired subject effects,
hotspot persistence, prediction baselines and an explicit relaxed-coverage sensitivity.
[`v2`](configs/physiology_semantic_tokenizer/tak_disagreement_comparison_v2.yaml)
reads only the completed v1 maps to finish the relaxed-coverage paired comparisons;
it preserves v1 results and independently reproduces the retained subject statistics.
The existing `--four-dataset-analysis` renderer accepts the resulting summary.
Tests: `test_tak_disagreement_comparison.py`, `test_tak_mismatch_spatial.py`,
`test_tak_mismatch_runner.py` and `test_tak_mismatch_report.py`.

The fixed-pattern report's scalp maps reuse the vertex-centred azimuthal-equidistant display projection,
with MNE full-head extrapolation and a dashed measured-node convex hull. REFED
keeps its registered planar layout. Exported display coordinates are separate
from the unchanged statistical locations and values.

For the concise Hb-only atlas, the same renderer accepts
`--four-dataset-analysis <completed_fixed_analysis> --hb-atlas-design --output <fresh_export>`.
It plots the original D1 population maps, one overall D1 point per subject,
and each selected subject's per-location probability of entering the highest
D1 quarter across eligible records, from the retained `hotspot_persistence.csv`.
Both topography sets share the same head coordinates; population maps use
individual dataset color bars, while subject examples share a 0–1 probability
scale and retain the existing caption. The subject scatter shows
compact horizontal jitter, a median marker and the interquartile range. Support
counts and hotspot rings are omitted. The export retains per-record consistency checks,
all selection candidates, retained values and probability denominators alongside the figures.

The fixed-result physiological exploration uses the same evaluator's
`physiology-extract`, `physiology-analyze`, `physiology-ocular`,
`physiology-counterfactual`, and `physiology-shapes` stages with
[`tak_mismatch_fixed_physiology_v1.yaml`](configs/physiology_semantic_tokenizer/tak_mismatch_fixed_physiology_v1.yaml).
Full-denominator scalar statistics and the prespecified array/EOG panel remain
separate. It changes no parent fit; EOG regression sensitivity keeps the Hb-only
readout and PCA/scale frozen and includes a mean/SD-matched proxy control.
`render_tak_mismatch.py --fixed-analysis <analysis> --output <fresh-export>`
consumes the explicit completed analysis manifest. The same renderer's
`--revise-report <previous-export> --visual-channel-audit <audit> --output <fresh-export>`
applies the confirmed Visual interpretation boundary while retaining the
previous report, numerical tables and bitmap figures. Tests:
`test_tak_mismatch_analysis.py`, `test_tak_mismatch_fixed_physiology.py`,
`test_tak_mismatch_physiology.py`, `test_tak_mismatch_ocular.py`, and the existing
report tests.

The evaluator's read-only `physiology-topography` stage consumes those completed
scalar tables and the retained registry geometry, writing a fresh
`--spatial-output <fixed-analysis/spatial_patterns_version>` child. It keeps
every record/channel denominator and ranks illustrative subjects with
`--spatial-examples 3`. The same renderer's `--spatial-analysis <completed-analysis>
--output <fresh-export> --workers 2` creates MNE-style maps, a bitmap PDF report
and a complete per-record/metric PNG gallery. Spatial contrast, neighbouring
agreement and early/late consistency select cases; they are not an independent
confirmation. Partial time support stays visible, while Visual EEG-dependent
maps remain excluded from spatial interpretation under the owning impact audit.
Add `--spatial-style-preview` to this renderer for two display-only old/new
comparisons: vertex-centred azimuthal equidistant coordinates (equator at 90% of
the head radius), linear full-head fill, and a dashed measured-node convex hull.
The preview reads the retained plan geometry and subject-map scalars, preserves
their values and selection, and writes two PNGs plus coordinate/provenance tables
to a fresh export. It does not update spatial statistics or existing reports;
the normal spatial export retains its original projection and local support.
For a mechanism-oriented review of retained scalars, the evaluator's
`physiology-patterns --spatial-analysis <completed-spatial-analysis>
--panel-export <completed-shared-snapshot> --pattern-output <fixed-analysis/mechanism_patterns_version>`
keeps complete-region subject denominators and one parameter estimate per shared
group/view, including failed terminal values. The renderer's
`--mechanism-analysis <completed-pattern-analysis> --output <fresh-export>`
creates population/paired regional plots, angular full-head fixed maps, record
stability matrices, parameter/failure plots and a bitmap PDF/HTML report.
Shared-parameter results remain three sampled positions; they are not filled
into a full-head parameter map. These stages add no fits or significance tests.
Tests: `test_tak_mismatch_spatial.py`, the report/runner suites and the registered
geometry projection check in `test_tak_topography.py`.

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

The matched τ/β and within-subject task extension uses the same evaluation entry
with `--tak-task-parameters --task-parameter-stage
<check|plan|synthetic|prepare|pilot|run|summarize|analyze>` and
[`shared_driver_tak_task_parameters_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_tak_task_parameters_v1.yaml).
It freezes the parent development coordinates, reuses parent tasks with explicit
source identity, and adds metadata-admitted MI/MA and WG/n-back/DSR supports.
Synthetic unchanged-parameter and known-shift controls precede new signal reads.
Task labels only enter the postfit within-subject leave-support-out readout;
missing supports and numerical failures retain the requested denominator.
Tests: `test_tak_task_parameters.py` and the Tak/repeatability suites.
The read-only `analyze` stage writes a fresh child (`--task-parameter-analysis-output`).
Render the completed analysis with `render_shared_driver_reconstruction.py
--tak-task-parameters --run <completed_run> --output <fresh_export>`;
`--task-parameter-analysis` selects an explicit analysis version. The PNG/HTML
collection and bitmap PDF preserve full coverage, paired task contrasts,
held-out-support discrimination, null controls and all parameter estimates.
The same renderer's `--tak-parameter-candidates --run <completed_run>
--output <fresh_export>` makes uniform-marker, linear-axis six-point panels.
It keeps all primary estimates and adds the three lowest numeric supplementary
REFED video IDs per subject. The display inventory retains source task paths
and conditions; this presentation subset does not redefine the original
same-condition statistical analysis.

The separately versioned full reconstruction cache uses this same entry with
`--tak-reconstruction-cache --cache-stage
<check|plan|synthetic|pilot|produce|verify> --run-dir <fresh_run>`. The default
[`v3 contract`](configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v3.yaml)
uses normalized 1:4:4 observation weights and cached-solution starts with explicit
standard fallback. It reuses verified input features and independently
fits PC1 and positive scales independently per dataset/subject, frozen across
all that subject's records and sites. The population coordinate supplies only
reference scoring SDs. Its pilot compares standard CPU, warm CPU and warm CUDA
using production coordinates, including large-PCA-change training subjects.
Successful warm tasks omit the uncomputed fixed baseline; consumers use free
predictions and validity flags. The earlier individual-PCA
[`v2 contract`](configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v2.yaml)
remains explicit and equal-weight. The retained
[`v1 contract`](configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v1.yaml)
remains available through explicit `--config` and preserves shared PCA; the
versions require separate output namespaces and cannot reuse each other's fits.

The explicit [v4 Visual migration contract](configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v4.yaml)
uses the same cache entry and stages. It refits only each Visual record's usable
positions from measurement v6, using the frozen downstream-v2 Hb scale once and
retaining v3 EEG coordinates. Other datasets' exact v3 tasks/features are linked
with their original provenance. Native pair IDs remain stable while a separate
compact index addresses selected Hb columns. The plan excludes rejected pairs
before any solver call; final verification checks target/scale identity and
rejects extra task files. Synthetic sparse-index, feature-replay and warm-start
tests are in `test_tak_reconstruction_cache.py`.

The [tokenizer cache audit](configs/physiology_semantic_tokenizer/tokenizer_cache_audit_v2.yaml)
uses `analyze_tak_cache_versions.py --tokenizer-cache-audit --stage
<plan|pilot|run|join-check|review>` on the completed measurement-v6,
downstream-v2 and reconstruction-v4 roots. It checks every record/sample/task,
keeps failure and undefined denominators, describes input/latent/S–O statistics,
compares Visual on the same retained locations, and checks adjacent-window Hb
agreement after restoring window references. `join-check` independently compares
loaded Hb samples with prepared targets using absolute integer record indices;
naive relative-offset and reported-clock rounding discrepancies remain explicit.
Input spectral panels require at least 30 seconds of supported data; native
short sample lengths remain in the input census. `review` adds robust amplitude
distributions and the complete low-variance-window list without changing masks.
Tests: `test_tak_cache_version_analysis.py`; synthetic checks cover cancellation,
unit invariance, subject aggregation, reference restoration and half-grid joins.

Long-window full-electrode EEG and paired-Hb S/O targets use the existing
`run_dual_component_tokenizer.py --config
experiments/configs/physiology_semantic_tokenizer/long_so_supervision_v2.yaml
--stage <check|so-plan|so-reference|so-pilot|so-produce|so-verify>` entry.
The [contract](configs/physiology_semantic_tokenizer/long_so_supervision_v2.yaml)
retains exact completed-v4 supports, combines identical-support Visual Probes,
uses training-only voltage references and up to three rank-revealing proxy rows,
and preserves all Hb positions with explicit source-failure masks. Production
includes an independent final verification. Substantial stages require the
owning user-systemd service. The earlier all-proxy v1 pilot remains failed
evidence and owns the reused training-reference arrays. Tests:
`test_so_supervision_cache.py`, `test_dual_component_power_cap.py`.

The [full unified-grid S/O contract](configs/physiology_semantic_tokenizer/long_so_supervision_v3.yaml)
uses the same long-window command with `--stage so-produce`. It owns training
reference replay, the representative training pilot, resumable regional fits,
window synthesis and independent verification in `data/cache/physiology_so_supervision_v3`.
Use the original supervised launch script in that output root to resume the
same generation. [Targeted tests](../tests/test_unified_so_production.py)
cover split membership, short-parent clocks and qualified proxy selection.

The [unified EEG S/O pilot](scripts/evaluate_unified_eeg_so_pilot.py) runs
`synthetic`, `fit`, `targets`, then `report` with the
[eight-subject paired contract](configs/physiology_semantic_tokenizer/unified_eeg_so_pilot_v1.yaml).
It compares the retained native production baseline to 1–45 Hz/200 Hz EEG
with the same Hb, individual PC, scales, three proxy sites, and known additions.
[Paired clock and metric checks](../tests/test_unified_eeg_so_pilot.py) use synthetic voltage.

The read-only [S/O synthesis audit](scripts/audit_so_supervision_synthesis.py)
uses [its population/mechanism contract](configs/physiology_semantic_tokenizer/so_supervision_synthesis_audit_v1.yaml)
with `--stage <plan|pilot|population|mechanism|summarize|render>` and the
[subject-wide position comparison](configs/physiology_semantic_tokenizer/so_supervision_position_audit_v1.yaml)
with `--stage <position-plan|position|position-summary>`. It reads the completed
public cache only: all retained windows contribute cross-position conditional
power-reference comparisons; metadata-selected windows isolate carrier
overlap-add from recombination/refiltering, with a constant-reference control.
Single-position voltage outputs are explicitly diagnostic reconstructions,
not separate electrode targets already present in the SSM cache. Population
and subject sampling have separate denominators. No SSM/reference refits or
cache mutations occur. Substantial stages use user systemd, bounded workers
and single numerical threads. Tests: `test_so_supervision_synthesis_audit.py`.
The [matched temporal control](configs/physiology_semantic_tokenizer/so_supervision_temporal_audit_v1.yaml)
uses `--stage temporal` on the same mechanism windows. It fixes each selected
proxy to its mean over the original reference frames, explicitly verifying
unchanged mean log power before synthesis; this separates temporal variation
from differences in average reference level.
The [spectral/overlap extension](configs/physiology_semantic_tokenizer/so_supervision_spectral_overlap_v1.yaml)
uses `--stage <spectrum-plan|spectrum|overlap-pilot|overlap|spectrum-summary|spectrum-render>`
and writes a separate `spectral_overlap_v1/` evidence child. It reuses the same
subject and mechanism identities. Native-rate Welch spectra compare original,
single-position and joint S/O voltage using separate band-retention and
within-signal shape denominators. Exact binned outer products verify the Jensen
mixture identity; the boxcar control reuses the same projected blocks. Matched
one-second spectra isolate synthesis changes; these have a different estimator
and sample denominator from the eight-second subject spectra. No cache repair
or new SSM/reference fitting occurs. Tests remain in
`test_so_supervision_synthesis_audit.py`.

Read-only v1/v3 descriptive comparisons use
[`analyze_tak_cache_versions.py`](scripts/analyze_tak_cache_versions.py) with
`--v1 <completed_v1> --v3 <completed_v3> --output <fresh_v3_child> --stage
<plan|pilot|run|summarize|report>`. The frozen analysis configuration owns the full
metadata census, outcome-independent spectral panel, common-success pairing,
and record-then-subject aggregation. `report` writes a fresh bitmap/HTML export
with robust amplitude and spectral summaries. Pilot/run use a durable supervisor;
no raw inputs, labels, or refits are read or generated. Synthetic frequency,
pairing and aggregation checks live in `test_tak_cache_version_analysis.py`.

The same owner provides `--fnirs-audit --run <completed_v3> --features
<retained_so_feature_run> --comparison <completed_v1_v3_audit> --output
<fresh_v3_child> --stage <plan|pilot|run|summarize|report>` for the Hb amplitude,
morphology and residual-cancellation census. It reuses retained waveform
metadata and the fixed spectral panel, independently checks residual algebra
on one saved task per subject, and joins Visual's native channel-reference
metadata. Pilot/run use a durable supervisor; reports retain subject-level
rates, local-amplitude strata, undefined metrics and numerical failures.
`--stage examples --example-export <fresh_audit_child>` adds a bounded real
HbO/HbR waveform comparison in the saved common subject Hb coordinate, with
source-file unit/header checks and replay against retained prepared inputs.
`--stage source-audit --source-export <fresh_audit_child>` joins the completed
cache's Visual header exclusions and REFED acquisition quality table, replays
the two previously illustrated subjects' retained calibration moments, and
checks common-unit invariance. It changes no production mask, scale or fit;
REFED's unresolved flag polarity remains explicit.
`--stage author-check --source-export <author_review_child>` exercises pinned,
reviewed REFED author function definitions on temporary synthetic MAT files,
checking channel preservation, amplitude linearity, HbT selection and the
actual training baseline/normalization path. It runs no training or real data.
`--stage <refed-plan|refed-pilot|refed-run|refed-report> --source-export
<fresh_audit_child>` audits the published subject-by-channel REFED flags across
all retained records. It preserves failure/undefined denominators, computes
paired subject statistics and all-channel spectra, checks duration/channel
composition sensitivity, and exports same-window flag0/flag1 waveforms in the
saved normalization. Pilot/run use a durable supervisor; neither flag is
reclassified as valid/invalid by this analysis.
`--stage <state-plan|state-pilot|state-run|state-report|state-readout>
--source-export <fresh_audit_child>` extends the retained REFED audit to
tau/beta distributions, physical-state summaries and input-amplitude ranges.
Subject-level sign-flip tests and Holm correction preserve repeated-record
dependence. The requested raw/filter display reads only the three previously
selected video variables through the unified reader and checks the low-band
replay against the retained cache. Pilot/run use a durable supervisor; no SSM
fits or production scales change. `state-readout` adds versioned explanatory
zooms from the first export's saved numeric arrays.
The same analysis owner accepts `--amplitude-comparison --run <completed_v3>
--visual-run <v4_with_completed_verified_preparation> --audit <fnirs_audit>
--export <fresh_audit_child> --stage <plan|pilot|run|report>`. It compares all
finite admitted prepared inputs in frozen subject coordinates, preserves
missing-input and failed-fit denominators, and joins retained REFED beta/S/O
statistics. Visual uses its repaired usable-channel inputs; running v4 fit
results are not read. Weighted amplitude distributions, same-window channel
spreads, 90-second sensitivity and posthoc waveform-gain diagnostics are
descriptive. Pilot/run use a durable supervisor; no model is refitted.
`--stage <visual-plan|visual-pilot|visual-run|visual-report> --visual-export
<fresh_audit_child>` performs the Visual-only usable-channel Hb normalization
comparison. It counts native EEG and per-record usable Hb locations, renders
MNE scalp layouts, and compares retained waveforms under old and usable-only
common Hb scales. Pilot/run use a durable supervisor; EEG PCA/scales and all
SSM fits remain frozen. The excluded-channel amplitude fixture is in the same
targeted test module.

The bounded [three-subject normalization comparison](configs/physiology_semantic_tokenizer/shared_driver_normalization_comparison_v1.yaml)
uses the same entry with `--normalization-comparison --normalization-stage
<check|prepare|pilot|run|summarize> --run-dir <fresh_run>`. It reads retained v1
features/tasks for five named Simultaneous development subjects only, selects
three subjects and three nonoverlapping n-back/POOz windows each by observed
amplitude, then compares individual PCA and local scaling factorially. Component
weights and the original objective SDs stay fixed. `pilot` replays three cached
baselines and benchmarks three local-scale fits; `run` resumes the 27 changed
fits. Both fitting stages require a durable supervisor. The existing
`test_tak_reconstruction_cache.py` covers selection, factorial controls and
metrics; `summarize` writes CSV/JSON and bitmap waveform figures with HTML.

The [amplitude frequency audit](configs/physiology_semantic_tokenizer/shared_driver_amplitude_frequency_v1.yaml)
uses `--amplitude-frequency --frequency-stage <check|scan|summarize> --run-dir
<fresh_run>`. It reads planned train/validation task metadata from the existing
four-dataset cache before a frozen cutoff, with bounded task-array verification.
The supervised scan preserves missing and failed denominators, both cached
parameter arms, nonoverlapping/all-window counts and subject-level rates.
Thresholds are engineering diagnostics and do not establish causal attribution.
The existing cache test file covers inclusive boundaries, missing/failed counts
and rejecting test-split access before task reads; no model is refitted.

The [weight/SO comparison](configs/physiology_semantic_tokenizer/shared_driver_weight_so_v1.yaml)
uses the same entry with `--weight-so-comparison --weight-so-stage
<check|plan|prepare|pilot|fit|targets|summarize|review> --run-dir <fresh_run>`.
Native preparation and fitting/target production run under a durable supervisor.
`targets --pilot` benchmarks the equal-weight target for the first validation
case of each dataset; normal `targets` resumes the full fixed matrix.
The shared voltage reference/projection owner supports native 200/500/1000 Hz;
the default 200 Hz path retains its numerical definition. Targeted tests are
`test_tak_reconstruction_cache.py` and `test_dual_component_power_cap.py`.
The study keeps normalization and τ/β fitting fixed and normalizes weight sums
to three; it generates a configuration recommendation, not a new full cache.
`review` adds Hb waveform diagnostics and a versioned report that distinguishes
the relative screening result from the production recommendation; original
numerical results and the first report remain retained.
The [paired τ/β S/O diagnostic](configs/physiology_semantic_tokenizer/shared_driver_tau_beta_so_v1.yaml)
uses `--tau-beta-so-comparison --tau-beta-so-stage <prepare|targets|summarize>`.
It reuses both qualified prediction arms from the completed weight study,
reconstructs fixed-parameter targets and freezes baseline predictions in a
separate ocular mechanism control. No new raw reads or nonlinear fits occur.
`targets --pilot` measures four dataset cases before the supervised full matrix;
the same two targeted test files cover arm routing and the retained split boundary.
The [cached-window ocular confirmation](configs/physiology_semantic_tokenizer/shared_driver_cache_ocular_confirmation_v1.yaml)
uses `--cache-ocular-confirmation --cache-ocular-stage <plan|pilot|run|summarize>`.
It freezes a metadata-selected development batch, replays native features against
the source cache, injects known EOG-derived voltage at three local RMS levels,
actually refits every condition, then constructs and plots before/after S/O.
The supervised pilot measures four native readers and eight fits; the full run
resumes those exact tasks. The same cache and power-cap test files cover selection,
split exclusion, amplitude/support invariants, numerical fits and projections.
`summarize --cache-ocular-export <fresh_run_child>` creates a versioned figure
and HTML export, requiring exact reproduction of the retained scientific summary;
it leaves the original report, numeric evidence and fitted cache unchanged.
The [ocular modality-weight comparison](configs/physiology_semantic_tokenizer/shared_driver_ocular_modality_weight_v1.yaml)
uses `--ocular-modality-weight --modality-weight-stage <plan|pilot|screen|analyze|verify|report|parameters>`.
It reads only its six-subject retained development panel, verifies equal-weight
reuse, and compares direct SSM responses separately from voltage power-cap targets.
The supervised pilot runs four weighted synthetic fits and eight measured fits;
screening preserves the declared stopping rule before any candidate continuation.
Tests in `test_tak_reconstruction_cache.py` cover frozen-scale metrics and enforcing
fidelity guards even when Hb disturbance improves. Every subject receives a
four-configuration waveform panel and a model-versus-target change panel.
`parameters` reads retained fits to export paired τ/β sensitivity: common qualified
ROI support for cross-weight summaries, plus all qualified 1:4:4 subject values.
It preserves failed locations, distinguishes signed and absolute relative changes,
and writes a fresh parameter review without refitting or changing the source results.
The [individual-PCA 1:4:4 warm-start comparison](configs/physiology_semantic_tokenizer/shared_driver_warmstart_pca_weight_v1.yaml)
uses `--warmstart-comparison --warmstart-stage <plan|synthetic|prepare|pilot|run|report>`.
It reconstructs selected training subjects' native EEG features, checks retained
input equivalence, and compares standard optimization with old-cache starts under
the same new coordinates and weights. Timing separates preparation, calibration
and fitting; repeated pairs use counterbalanced order. Tests in
`test_tak_reconstruction_cache.py` cover explicit-start routing, unchanged weighted
observations and the pre-read training boundary. Production cache defaults remain
unchanged by this diagnostic. `report --warmstart-export <fresh_export>`
re-exports the same numerical comparison into a fresh report directory.
The [S/O physiology feature audit](configs/physiology_semantic_tokenizer/shared_driver_so_physiology_features_v1.yaml)
uses `--so-physiology-features --feature-stage <plan|pilot|run|report>`.
It replays retained 1:4:4 native targets, compares spectra and traditional
features, and exports separate training-scaled t-SNE/PCA views. It performs no
new SSM fits or raw dataset reads. Use `report --feature-export <fresh_export>`
for a revised figure export while preserving retained numerical results.
`src/analysis/physiological_patch_features.py`
owns the native audit feature definitions, separate from its legacy Atlas API;
`test_physiological_patch_features.py` checks physical-unit invariance, known
spectra, undefined features and cross-power cancellation, and
`test_dual_component_power_cap.py` covers target construction.
The source cache fitting arms explicitly fit τ/β at log-prior SD 0.125. Fits use 90 s,
or the actual shorter REFED support, and the corrected Visual native identities.
`produce` resumes matching terminal work, completes preparation/calibration,
fits the fixed reference and τ/β, then independently verifies the saved arrays.
Substantial stages require the recorded durable supervisor; `check` and `plan`
read no signals. The run's plan, coordinates, per-task arrays/status, summary
and verification provide the reusable cache interface, including failures and
downstream sample joins. Tests: `test_tak_reconstruction_cache.py`,
`test_tak_mismatch_data.py`, `test_eeg_proxy_10hz.py` and the Tak solver suites.
For a measured resource-only continuation, the existing `--workers` argument
may exceed the initial resource count retained in the frozen configuration;
the runner checks the process affinity and inherited cgroup CPU quota. Keep a
new execution-source snapshot and launch/resource record, drain the previous
controller, and reuse matching terminal tasks and frozen calibration. This
does not change the scientific configuration or restart completed fits.
The optional accelerated implementation is checked with `--cache-stage
accelerator-check --workers 4 --cache-gpu-devices 0,1
--cache-validation-output <fresh_child_of_run>`. It refits only retained
synthetic/pilot cases plus slow and nonconverged cases into that child, while
the producer can keep running. The child's protocol owns the comparison
thresholds and inventory. Production selects the validated implementation with
`--cache-algebra structured --cache-linear-backend cuda --cache-gpu-devices 0,1
--cache-validation-output <validated_child>` and requires a matching execution
source. One GPU process per device serves bounded batches of float64 normal
equations through per-worker shared buffers; CPU workers retain dynamics,
constraints, line search and priors. A later transport-only validation may use
`--cache-cpu-reference <completed-validation>` after checking the unchanged CPU
numerical functions; GPU cases are fitted again into a fresh validation child.
Retained CPU timings and newly measured GPU timings are labelled separately.
`--cache-gpu-workers N` assigns only N of the production workers to CUDA; the
remainder use the same validated structured CPU solve. Resource-only routing
changes may reuse retained comparisons when numerical source, GPU transport and
fitting functions are unchanged, with explicit source-equivalence and routing
test evidence in the matching validation child. Measure aggregate throughput
and GPU wait time to choose this split.
GPU infrastructure failures stop execution instead of being recorded as model
failures. Completed targets keep their original backend identity and are reused.
[Data boundary](../docs/DATA_CONTRACT.md#subject-calibrated-tak-reconstruction-cache--2026-11-04).

For a minimal retained-curve export, use
`render_shared_driver_reconstruction.py --best-subject-reconstruction --run <cache_run>
--output <fresh_export>`. It selects the qualified 90-second fit with the lowest
mean EEG/HbO/HbR local NRMSE in a timestamped terminal-record snapshot, checks
the selected arrays, and exports a three-row PNG and bitmap PDF. The candidate
table preserves failures and short-record exclusions. This is a descriptive
best-case joint reconstruction, not a population estimate or a new fit.

The spatial τ/β case study uses this evaluation entry with `--tak-topography
--topography-stage <check|plan|synthetic|screen|prepare|pilot|run|analyze>` and
[`shared_driver_tak_topography_v1.yaml`](configs/physiology_semantic_tokenizer/shared_driver_tak_topography_v1.yaml).
It freezes a complete metadata candidate pool, selects illustrative subjects
using repeated cached-Hb spatial patterns, then fits all available sites with
one development-only coordinate per dataset. Source geometry, missing coverage,
all selection scores and numerical failures remain explicit. Tests:
`test_tak_topography.py` plus the Tak/10 Hz suites.
The `analyze` stage verifies retained prediction scores before writing a fresh
direct-child analysis; `--topography-analysis-output` selects its versioned path.
The existing renderer's `--tak-topography --run <completed_run>
--output <fresh_export>` creates absolute and subject-centered sensor maps,
independent-repeat comparisons, reconstruction quality and synthetic controls.
It shows observed positions without spatial interpolation and keeps one color
scale per dataset and parameter across subjects and repeats. The HTML and
searchable bitmap PDF atlas link directly to the owning analysis tables.
For a display-only contour preview from retained partial results, use the same
renderer with `--tak-topography-style-preview --run <run> --output <fresh_export>`.
It selects the first non-Visual subject with two complete qualified maps,
uses range-preserving local display smoothing inside measured coverage, and
starts no fits. The frozen primary map contract and source estimates remain
unchanged; the preview records its separate interpolation/display convention.
Add `--topography-examples 3` to export three distinct subjects on the common
first-window convention, retaining all subjects with at least 90% available
positions in this stopped run. All panels share parameter color scales;
unfinished positions retain explicit crosses and blank neighborhoods. The
single-subject preview keeps its original two-repeat computation, with the
aggregation label omitted from the figure and retained in the companion text.

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

The [TK-BASE-v1 G0 design](../docs/EXPERIMENT_PLAN.md#tk-base-v1) uses
[`dual_component_base_g0_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_base_g0_v1.yaml)
through the same `scripts/run_dual_component_tokenizer.py`: `--stage check`,
`--stage synthetic`, metadata-only `--stage base-plan`, then supervised
`--stage base-prepare`; `--stage base-verify` rechecks the prepared interface.
G0 prepares the full original train/validation population from the completed S/O
v2 cache, with 200 Hz EEG and 10 Hz paired Hb. Original pilot regression intervals
are excluded from every overlapping training/scale input and separately resampled.
Each exact physical crop has one parent owner; scale/scoring blocks do not overlap.
Per-position training moments weight records within subjects, then subjects equally;
validation receives frozen scales. Missing training position scales remain unavailable.
Sampling weights datasets, subjects and native records equally; codebook initialization
covers every eligible training record. G0 stops after data/interface verification;
it does not train models. `test_dual_component_base_g0.py` covers these boundaries,
support accounting, masked targets, sampling, scales, resumption and integer replay.
The library data owner also provides `base_tokenizer_training_data` for completed
G0 boundary validation and `base_tokenizer_examples` for scoring-index enumeration.
The historical runner retains its original private names as import aliases.

The [G1 five-model contract](configs/physiology_semantic_tokenizer/dual_component_base_g1_v1.yaml)
consumes this exact completed G0 run through the same runner's `check`, `synthetic`,
metadata-only `base-train-plan`, and supervised `base-train` stages; controller-owned
`base-train-one` runs the declared children, and `base-train-report` summarizes them.
Three modality/architecture warmups supply matched waveform continuous/K128 pairs
and the EEG dense-long K128 reference. All five use seed 17 and 10,000 total updates,
including 1,000 warmup updates. The learning-rate clock stays constant through step
4,000, then follows a fixed cosine to step 10,000; 4,000/7,000/10,000 checkpoints use
the complete G0 train-score, validation and regression panels. Integer reconstruction,
per-color metrics, code health and training-fitted single-ID shape/X prototypes are
retained with their actual ID sequences and fixed examples. Selection uses only the
complete validation NMSE; regression is reported separately. Failed jobs remain in
the five-model denominator. `test_dual_component_base_g1.py` covers boundary checks,
masked gradients, equal-level aggregation, the bounded budget rule, mmap equivalence,
matched warmup, exact interrupted continuation, and all five synthetic interfaces.

For unified 200 Hz S/O supervision, the versioned
[G0 preparation contract](configs/physiology_semantic_tokenizer/dual_component_base_g0_v2.yaml)
uses the v3 target grid directly and maps the five native pilot holdouts to that
grid before excluding overlapping crops. The paired
[G1 training contract](configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml)
uses this G0 population, training scales and masks with the same five-model
matrix, stages and targeted tests described above.

For the matched 10,000-step waveform comparison, run
[`compare_base_so_versions.py`](scripts/compare_base_so_versions.py) with
`--output <fresh_child_of_unified_G1_run>`. It joins retained validation inputs
by dataset, subject, record, parent window, crop start and position name. It
checks single-ID prototype errors against the G0 validation targets, estimates
the zero-shape baseline and compares assignments on mutually valid patches.
The HTML export embeds reconstruction examples and training/validation token shapes.

The [joint acceptance contract](configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml)
binds waveform reconstruction, local dynamics, single-ID information, image review
and integer replay to the same frozen four-branch candidate. The metadata-only
[`prepare_joint_tokenizer_panel.py`](scripts/prepare_joint_tokenizer_panel.py)
freezes G0 index triples for training prototypes, development screening, full
development and post-selection subject review in one `panels.npz`. The retained
G0 train/validation boundary and frozen training scales govern every consumer.

[`run_joint_waveform_dictionary.py`](scripts/run_joint_waveform_dictionary.py)
uses the [single-atom dictionary contract](configs/physiology_semantic_tokenizer/dual_component_joint_dictionary_v1.yaml).
Run `--synthetic` with a fresh `--output`, then fit under a durable supervisor
with the same `--config` and output. Training fits an X-only frequency predictor
and two-second waveform atoms on train samples. Each branch emits one integer
per patch and frozen lookup restores the waveform. The companion
[`run_joint_contextual_tokenizer.py`](scripts/run_joint_contextual_tokenizer.py)
uses the [contextual dictionary contract](configs/physiology_semantic_tokenizer/dual_component_contextual_v1.yaml)
with `--modality eeg|hb` and `--output`. It fits one-level latent dictionaries and
aligns decoders using the fixed historical continuous checkpoint and train samples.
Both entries preserve thirty-second observations, 200 Hz EEG, paired-color 10 Hz
Hb, and independent S/O outputs. The dictionary's shape, resume identity and
integer consumer tests are in
[`test_joint_waveform_dictionary.py`](../tests/test_joint_waveform_dictionary.py).

Training and development diagnostics keep their own versioned contracts and supervised
output children below the same joint-baseline run.

| Diagnostic | Entry and contract | Interface and checks |
| --- | --- | --- |
| Spectral assignment | [`run_joint_spectral_dictionary.py`](scripts/run_joint_spectral_dictionary.py), [`dual_component_joint_spectral_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_joint_spectral_v1.yaml); `--config --output` | Single two-second EEG atom with training-fitted energy calibration; [`test_spectral_waveform_dictionary.py`](../tests/test_spectral_waveform_dictionary.py) |
| Spatial predictability | [`diagnose_tokenizer_spatial_predictability.py`](scripts/diagnose_tokenizer_spatial_predictability.py), [`dual_component_spatial_diagnostic_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_spatial_diagnostic_v1.yaml); `--synthetic`, then supervised `--config --output` | Training-subject folds for local X, other-electrode X and privileged same-branch predictors; the entry's synthetic stage checks the spatial controls |
| Hb local shape | [`run_joint_hb_shape_tokenizer.py`](scripts/run_joint_hb_shape_tokenizer.py), [`dual_component_joint_hb_shape_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_joint_hb_shape_v1.yaml); `--candidate continuous|vq1024 --mode benchmark|train --output` | Matched v3 continuations of the retained G4 paired-color architecture |
| Short atomic support | [`run_joint_temporal_diagnostic.py`](scripts/run_joint_temporal_diagnostic.py), [`dual_component_joint_temporal_diagnostic_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_joint_temporal_diagnostic_v1.yaml); `--config --output` | EEG 0.5-second and 0.125-second diagnostic atoms, with their actual integer counts and rates; final two-second interface eligibility is false |
| Spatial integer decoder | [`run_joint_spatial_dictionary.py`](scripts/run_joint_spatial_dictionary.py), [`dual_component_spatial_dictionary_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_spatial_dictionary_v1.yaml); `--synthetic`, then `--config --output` | Native-position band roles, one two-second ID per branch and a training-fitted spatial readout; `--evaluate-only --audit-output` replays the saved integers |
| Frozen cross-branch readout | [`fit_joint_cross_branch_readout.py`](scripts/fit_joint_cross_branch_readout.py), [`2×2 contract`](configs/physiology_semantic_tokenizer/dual_component_cross_branch_readout_v1.yaml) and [`diagonal control`](configs/physiology_semantic_tokenizer/dual_component_cross_branch_readout_control_v1.yaml); `--synthetic`, then supervised `--config --output`; `--render-only` | Train-subject cross-fitted energy calibration of frozen spatial integer streams, matched own-branch gains and ID shift/masking controls; synthetic coefficient, identity-map and integer replay checks |
| Frozen branch-role pairing | [`run_joint_spatial_cross_role.py`](scripts/run_joint_spatial_cross_role.py), [`dual_component_spatial_cross_role_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_spatial_cross_role_v1.yaml); `--synthetic`, then supervised `--config --output`; `--render-only` | Frozen S70/O25 K512 books with equal-capacity joint spatial ridge under same-role and fixed O-offset layouts; synthetic role-specific atom mapping, missing ID and checkpoint replay checks |
| Full-sample EEG decoder | [`run_joint_fullband_eeg.py`](scripts/run_joint_fullband_eeg.py), [`K1024`](configs/physiology_semantic_tokenizer/dual_component_joint_neural_eeg_v1.yaml), [`K8192`](configs/physiology_semantic_tokenizer/dual_component_joint_neural_eeg_k8192_v1.yaml) and [`commitment`](configs/physiology_semantic_tokenizer/dual_component_joint_neural_eeg_commit_v1.yaml) contracts; `--mode prepare|benchmark|train --candidate continuous|vq1024|vq8192 --output` | Full 400-sample reconstruction with local and adjacent-token residuals; fixed-panel code use and latent geometry via [`audit_fullband_code_use.py`](scripts/audit_fullband_code_use.py); [`test_fullband_eeg_tokenizer.py`](../tests/test_fullband_eeg_tokenizer.py) |
| V3 EEG/Hb observation quality census | [Audit entry](scripts/audit_tokenizer_observation_quality.py) and [public development contract](configs/physiology_semantic_tokenizer/tokenizer_observation_qc_census_v1.yaml); stages `synthetic`, `pilot`, `census` | Full G0 non-overlapping score grid, original observation masks and frozen scales; [frequency tables, flagged identities and measured waveforms](runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261124_observation_qc_census_v1/index.html) |

The contextual entry also accepts
[`dual_component_contextual_semantic_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_contextual_semantic_v1.yaml)
for Hb assignment from training-normalized paired-color level, centered shape,
slope and curvature features. It retains a single ID per two-second patch.
The spectral entry's
[`semantic v2 contract`](configs/physiology_semantic_tokenizer/dual_component_joint_spectral_semantic_v2.yaml)
uses `--codebook-size 16|64` to diagnose a smaller attribute-based vocabulary.
The Hb local-shape entry's `--mode synthesis --synthesis-config` consumes the
fixed source milestones in the [`2k`](configs/physiology_semantic_tokenizer/dual_component_joint_hb_synthesis_v1.yaml)
and [`4k`](configs/physiology_semantic_tokenizer/dual_component_joint_hb_synthesis_v2.yaml)
contracts. Its frozen 0.4 Hz decoder synthesis writes a fresh versioned checkpoint.
[`refit_joint_hb_decoder.py`](scripts/refit_joint_hb_decoder.py) with the
[`frozen-ID decoder contract`](configs/physiology_semantic_tokenizer/dual_component_joint_hb_decoder_refit_v1.yaml)
uses `--arm bounded|unrestricted --config --output` for matched training-only
decoder updates while preserving the observation encoder and integer vocabulary.

Independent review uses
[`evaluate_joint_tokenizer.py`](scripts/evaluate_joint_tokenizer.py) with
`--panel`, `--modality`, `--role`, `--kind`, `--checkpoint` and fresh `--output`.
It reloads saved integers through the frozen checkpoint and recomputes scores.
[`review_joint_tokenizer.py`](scripts/review_joint_tokenizer.py) takes the candidate,
matched K128 and continuous audit directories plus a fresh output JSON path, and
applies the reconstruction limits after checking identical example triples.
The pattern entry
[`audit_joint_tokenizer_patterns.py`](scripts/audit_joint_tokenizer_patterns.py)
with `--checkpoint`, `--modality`, `--panels`, `--panel` and fresh `--output`, and
[`render_joint_tokenizer_waveforms.py`](scripts/render_joint_tokenizer_waveforms.py)
for retained fixed-panel waveform arrays. Training-fitted background and
background-plus-ID readouts share the frozen panel. Reconstruction metric tests
are in [`test_joint_tokenizer_metrics.py`](../tests/test_joint_tokenizer_metrics.py),
pattern controls in [`test_joint_tokenizer_patterns.py`](../tests/test_joint_tokenizer_patterns.py),
and saved contextual decoding in
[`test_contextual_latent_dictionary.py`](../tests/test_contextual_latent_dictionary.py).
The [full-QC codebook t-SNE atlas](scripts/render_full_qc_codebook_tsne.py)
uses `geometry`, `profiles`, then `render` with the same fresh `--output` directory.
It projects each assignment codebook separately and colors words by observed
two-second development patches on the frozen pattern panels.
The [readout-variance audit](scripts/audit_joint_tokenizer_readout_variance.py)
fits and selects shrinkage on separate training subjects from a retained pattern
audit (`--source --output`). The
[target-input dictionary oracle](scripts/audit_joint_dictionary_target_oracle.py)
uses `--run --output` for exhaustive supervised nearest-atom error on the frozen
screen panel.

[`apply_joint_tokenizer.py`](scripts/apply_joint_tokenizer.py) provides
`encode|decode --bundle --input --output` for observed X and four integer sequences
in the frozen G0 signal coordinates. The retained
[four-branch replay verification](runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1/four_branch_replay_v1/verification.json)
uses the rejected EEG/Hb K8192 local dictionary control pair. Its sibling
`bundle.json` and saved arrays provide a reproducible interface example with fixed
training scales, dataset identities and position identities.
The `band_exchange_eeg` bundle kind uses the same complete-montage input and
native-unit output interface as `spatial`, with a self-contained frozen folded
band-word model. The [research-pair exporter](scripts/export_band_exchange_reference.py)
retains its training-scope disclosure and verifies an independently extracted copy.

The spatial audit uses [the development evaluator](scripts/audit_spatial_development.py)
to group positions by physical crop and replay complete montages. The
[single-ID audit](scripts/audit_spatial_tokenizer_patterns.py) identifies each atom
by branch and fixed frequency band. Its `--attributes` stage reads the saved
train and screen IDs. The [natural-context comparison](scripts/audit_montage_natural_context.py)
uses real adjacent windows and common observed electrodes. The [member atlas exporter](scripts/export_joint_member_atlas.py) displays train and validation members, support and occupancy for all codes. The [paired audit](scripts/compare_joint_tokenizer_audits.py) reapplies the frozen numerical rules to matching evaluation records. The [distortion audit](scripts/audit_spatial_tokenizer_distortion.py) accepts a full development result with `--matched-hard` for the Wiener and projection controls, `--matched-roles` to compare continuous and integer inputs to the same spatial readout, or `--band-limited-floor` to bound the best possible 30-second band-limited waveform under the original score masks.

| Follow-up computation | Entry | Contract or check |
| --- | --- | --- |
| Decoder-aware EEG integer assignment and train-only calibration | [Assignment](scripts/run_joint_spatial_analysis_synthesis.py), [gain fitting](scripts/calibrate_spatial_analysis_synthesis.py) | [Assignment contract](configs/physiology_semantic_tokenizer/dual_component_spatial_analysis_synthesis_v1.yaml), [gain contract](configs/physiology_semantic_tokenizer/dual_component_spatial_as_gain_v1.yaml) |
| Multielectrode end-to-end learning and frozen-integer phase diagnostics | [Montage entry](scripts/run_joint_montage_tokenizer.py): `prepare/train/evaluate/audit/summarize`, then `phase-readout`, `phase-review` or train-gated `spatial-phase` | [K512 joint contract](configs/physiology_semantic_tokenizer/dual_component_montage_joint_v1.yaml), [K2048 global analysis](configs/physiology_semantic_tokenizer/dual_component_montage_global_k2048_v2.yaml), [local complex phase readout](configs/physiology_semantic_tokenizer/dual_component_montage_phase_readout_v1.yaml), [cross-electrode train-CV diagnostic](configs/physiology_semantic_tokenizer/dual_component_montage_cross_phase_v1.yaml); [hard-forward, gradients, masked quadratics and replay checks](../tests/test_montage_joint_tokenizer.py) |
| Hb branch assignment and joint reconstruction | [Prototype training](scripts/run_joint_hb_prototype_tokenizer.py), [cross-decoding](scripts/audit_hb_tokenizer_crossover.py), [joint decoder](scripts/run_joint_component_decoder.py) | [Prototype contract](configs/physiology_semantic_tokenizer/dual_component_joint_hb_prototype_v1.yaml), [decoder checks](../tests/test_joint_component_decoder.py) |
| Actual AS decisions and categorical sample support | [Training-subject audit](scripts/run_actual_id_coarsening.py) | [Coarse-ID contract](configs/physiology_semantic_tokenizer/dual_component_actual_id_coarsening_v1.yaml), [fixed sample-size comparison](configs/physiology_semantic_tokenizer/dual_component_actual_id_sample_support_v1.yaml), [integer-only representative decode checks](../tests/test_coarsened_spatial_dictionary.py), [held-subject sample checks](../tests/test_actual_id_sample_support.py) |
| EEG single-word information by native frequency role | [Held training-subject audit](scripts/audit_eeg_id_role_information.py) | [Fixed background-relative readout and within-background shuffled control](configs/physiology_semantic_tokenizer/dual_component_eeg_id_role_readout_v1.yaml), [absolute-loss comparison](configs/physiology_semantic_tokenizer/dual_component_eeg_id_role_absolute_v1.yaml), [pooling and unseen-word test](../tests/test_eeg_id_role_readout.py) |
| Anchored continuous EEG coordinates and matched historical references | [Train-subject fitting](scripts/run_anchored_spatial_carrier.py), [historical replay on the training panel](scripts/audit_anchored_training_references.py) | [Frozen design](configs/physiology_semantic_tokenizer/dual_component_anchored_carrier_v1.yaml), [alignment, support and serialization tests](../tests/test_anchored_spatial_carrier.py) |
| Native EEG component transfer from observed voltage | [Dataset and sensor transfer comparison](scripts/fit_eeg_component_transfer.py) | [Fixed source-separation contract](configs/physiology_semantic_tokenizer/dual_component_eeg_component_transfer_v1.yaml), [phase and conserved sum tests](../tests/test_eeg_component_transfer.py) |
| Actual AS integers with held-subject EEG source calibration | [Matched two-source search](scripts/run_transient_spatial_search.py) | [Frozen within-codebook contrasts](configs/physiology_semantic_tokenizer/dual_component_transient_as_v1.yaml), [component allocation from the integer-decoded total](configs/physiology_semantic_tokenizer/dual_component_integer_sum_source_v1.yaml), [initialization, integer search and independent decoding checks](../tests/test_spatial_analysis_synthesis.py) |
| X conditional bounded EEG source gains | [Same source-comparison runner](scripts/fit_eeg_component_transfer.py), `--synthetic` and `--render` | [Fixed transient-gain contract](configs/physiology_semantic_tokenizer/dual_component_eeg_transient_gain_v1.yaml), [synthetic transient, missing-array and frozen-source tests](../tests/test_eeg_transient_gain.py); training-only continuous diagnostic before integer assignment |
| Actual EEG integers from matched transient source folds | [Hard integer assignment](scripts/run_transient_spatial_search.py), [same-source raw single-ID information, `--review` and `--render`](scripts/audit_transient_actual_words.py) | [Hard comparison](configs/physiology_semantic_tokenizer/dual_component_transient_as_v1.yaml), [frozen word-information scope](configs/physiology_semantic_tokenizer/dual_component_transient_word_information_v1.yaml), [source ancestry, selected-family and original-readout tests](../tests/test_transient_actual_words.py) |
| Actual frequency-tagged EEG word members, same source fold | [All-code atlas, independent `--review`, offline `--browser`, exact `--coverage`](scripts/export_band_exchange_word_atlas.py) | [Train-only 400-crop atlas rules](configs/physiology_semantic_tokenizer/dual_component_band_exchange_word_atlas_v1.yaml), [whole paid ID, people support and true-member template tests](../tests/test_band_exchange_word_atlas.py) |
| Low K512 EEG integers predict hard or direct full teacher atoms | [Four-fold training, payload-only `--review`, `--direct-config` and isolated `--guarded-api`](scripts/run_eeg_integer_distillation.py) | [Fixed sequence student](configs/physiology_semantic_tokenizer/dual_component_eeg_integer_distillation_v1.yaml), [same-payload full-atom synthesis and projected-high floor](configs/physiology_semantic_tokenizer/dual_component_eeg_integer_distillation_direct_v1.yaml), [categorical fitting, full-band routing and replay tests](../tests/test_eeg_integer_distillation.py) |
| In-band events in the actual integer-decoded EEG neighborhood | [Frozen gallery diagnostic and independent `--review`](scripts/audit_integer_neighbor_voltage.py) | [Seventeen-case comparison and conditional outer-fold fitting rule](configs/physiology_semantic_tokenizer/dual_component_integer_neighbor_voltage_v1.yaml), [held-patch exclusion, band support and primary-control tests](../tests/test_integer_neighbor_voltage.py) |
| Whole-array union of frozen EEG integer vocabularies | [Waveform/support necessary bounds and exact endpoint display](scripts/audit_adaptive_array_vocabulary.py), `--review` for dual certificates, `--render` for retained training examples | [Training-only route, disjoint integer ranges and original support bounds](configs/physiology_semantic_tokenizer/dual_component_adaptive_array_v1.yaml) |
| Same-source EEG vocabulary relabeling and conditional K512 shortlist | [Reversible and limited-dictionary training audit](scripts/audit_transient_actual_words.py) | [Full-order contract](configs/physiology_semantic_tokenizer/dual_component_predictive_word_order_v1.yaml), [limited-order contract](configs/physiology_semantic_tokenizer/dual_component_conditional_shortlist_v1.yaml), [inverse and recovered-state checks](../tests/test_predictive_word_order.py) |
| Sparse continuous capacity at EEG event slots selected from X | [Capacity diagnostic and actual waveform gallery](scripts/run_band_exchange_event_slots.py) | [Budget, source and stopping contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_event_slots_v1.yaml), [physical propagation and exact-quota checks](../tests/test_band_event_slots.py) |
| Fixed-event EEG residual-VQ rate diagnostic | [Fit and decode 1/2/4 total layers, tuple support and saved-byte replay](scripts/run_band_event_rvq.py) | [Bounded multi-ID contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_event_rvq_v1.yaml), [zero-word and full-tuple decoder checks](../tests/test_band_event_rvq.py) |
| Finite waveform and amplitude-stratified EEG event words | [Fit, encode, score, actual-word readout and 14-bit replay](scripts/run_band_exchange_event_words.py) | [Single-word event contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_event_words_v1.yaml), [self-contained event and amplitude-tier checks](../tests/test_band_event_dictionary.py) |
| Per-patch native EEG frequency assignment and a single paid band-and-waveform word | [Source covariance and role exchanges](scripts/run_learned_spatial_roles.py), [prepare folded waveforms](scripts/prepare_band_exchange_words.py), [fit, encode and score atomic words](scripts/run_adaptive_band_dictionary.py) | [Role comparison](configs/physiology_semantic_tokenizer/dual_component_adaptive_band_allocation_v1.yaml), [exchange objective](configs/physiology_semantic_tokenizer/dual_component_adaptive_band_exchange_v1.yaml), [integer contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_words_v1.yaml) |
| Independent transport and physical replay of frequency-tagged EEG integers | [True 13-bit pack/unpack, frozen decode and original-X checks](scripts/audit_band_exchange_replay.py) | [Original hard-word contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_words_v1.yaml), [transport and payload-only decode tests](../tests/test_band_exchange_replay.py) |
| Portable frozen band-word EEG and Hb111 research pair | [Export, independent extraction and native-unit CLI comparison](scripts/export_band_exchange_reference.py), [four-branch inference](scripts/apply_joint_tokenizer.py) | [Frozen EEG contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_words_v1.yaml), [portable model and four-branch shape/unit checks](../tests/test_band_exchange_portable.py) |
| Native EEG waveform residuals fitted to frozen tagged integers | [Fit and replay local supervised correction tables](scripts/run_band_word_residual.py) | [Same-payload training contract](configs/physiology_semantic_tokenizer/dual_component_band_word_residual_v1.yaml), [categorical waveform and time-edge checks](../tests/test_band_exchange_words.py) |
| Complementary native S/O frequency sampling | [Paired spatial covariance, anchor reliability and frozen-word comparison](scripts/run_cross_branch_sampling.py) | [Same-source two-layout comparison](configs/physiology_semantic_tokenizer/dual_component_cross_branch_sampling_v1.yaml), [quota, cross-branch sign and independent marginal checks](../tests/test_cross_branch_sampling.py) |
| Independent cross-branch sampling transport and waveform review | [Actual 13-bit transport, payload-only decode and original masked score recomputation](scripts/audit_cross_branch_sampling.py) | [Frozen source and original train panel](configs/physiology_semantic_tokenizer/dual_component_cross_branch_sampling_v1.yaml), [independent bit transport and denominator tests](../tests/test_cross_branch_replay.py) |
| Conditional EEG residual words at a fixed 14-bit rate | [Pilot-conditioned encoding, independent replay and gallery](scripts/run_cross_branch_innovations.py) | [Matched capacity](configs/physiology_semantic_tokenizer/dual_component_cross_branch_innovations_capacity_v2.yaml), [independent training seed](configs/physiology_semantic_tokenizer/dual_component_cross_branch_innovations_seed223_v2.yaml), [conditional-word tests](../tests/test_cross_branch_innovations.py) |
| X-driven native EEG role geometry diagnosis | [One-position whole-context band change and physical waveform review](scripts/run_dynamic_band_roles.py) | [Train-only continuous information contract](configs/physiology_semantic_tokenizer/dual_component_dynamic_band_roles_v1.yaml), [geometry and one-position tests](../tests/test_dynamic_band_roles.py) |
| Frozen montage word search: independent EEG replay and physical waveform review | [Integer-only replay, original identities and hard-event figures](scripts/audit_montage_word_search.py), using the run-local `visual_review_v1/audit_config.yaml` | [Montage word-search contract](configs/physiology_semantic_tokenizer/dual_component_montage_word_search_v1.yaml), [independent lookup and transport tests](../tests/test_montage_word_search_audit.py) |
| Fixed EEG residual words across contexts and people | [Natural-context audit](scripts/audit_band_word_natural_context.py), [complete member atlas](scripts/export_band_exchange_word_atlas.py) | [Context contract](configs/physiology_semantic_tokenizer/dual_component_innovation_natural_context_v1.yaml), [true-member contract](configs/physiology_semantic_tokenizer/dual_component_innovation_word_atlas_v1.yaml) |
| EEG spectral readout and paired Hb allocation | [EEG magnitude/phase diagnostic](scripts/run_innovation_spectral_readout.py), [Hb visible-array and integer controls](scripts/run_hb_array_source.py) | [Frozen EEG frequency readout](configs/physiology_semantic_tokenizer/dual_component_innovation_spectral_readout_v1.yaml), [Hb train-array contract](configs/physiology_semantic_tokenizer/dual_component_hb_array_source_v1.yaml) |
| Nonlinear cross-person Hb event source at the frozen 111 total | [Local linear, local nonlinear and broad-array nonlinear event regression with frozen replay](scripts/run_hb_event_source.py) | [Original train-fold event contract](configs/physiology_semantic_tokenizer/dual_component_hb_event_source_v1.yaml), [visible input, event projection, quiet and fit-only tests](../tests/test_hb_event_source.py) |
| Train-only cross-spectral montage synthesis of unchanged EEG words | [Benchmark, four-fold fitting, integer-only replay and original event figures](scripts/run_band_exchange_synthesis.py) | [Frozen five-group, matched same-electrode and continuous controls](configs/physiology_semantic_tokenizer/dual_component_band_exchange_synthesis_v1.yaml), [complex phase, native mask support and held-person exclusion tests](../tests/test_band_exchange_synthesis.py) |
| Atomic distributed signed EEG measurements and local attribute bits | [Dimension-matched benchmark, four-fold fit, saved-integer replay and true waveform review](scripts/run_distributed_measurement.py) | [Train-only K1024 joint-word and same-rate controls](configs/physiology_semantic_tokenizer/dual_component_distributed_measurement_v1.yaml), [rank-one operators, complete ID alphabet and held-person exclusion tests](../tests/test_distributed_measurement.py) |
| Independent random-seed replication of paid EEG waveform words | [Original folded dictionary training](scripts/run_adaptive_band_dictionary.py), [fit-person ID permutation matching and held template transfer](scripts/audit_band_exchange_seed.py) | [Independent seed 211 contract](configs/physiology_semantic_tokenizer/dual_component_band_exchange_words_seed211_v1.yaml), [permutation direction and held-member tests](../tests/test_band_exchange_seed_alignment.py) |
| Paid frequency-tagged EEG word visual review | [Integer-only waveform replay and predeclared hard-case figures](scripts/render_band_exchange_word_review.py) | [Common folded books, exact ID interface and original train scope](configs/physiology_semantic_tokenizer/dual_component_band_exchange_words_v1.yaml), [word decode tests](../tests/test_band_exchange_words.py) |
| Paid EEG words under natural 30-second context changes | [Same-physical-patch integer and conditional-attribute replay](scripts/audit_band_word_natural_context.py), `--render` for retained train examples | [Four-fold frozen-source, true adjacent-two-second and fixed-background contract](configs/physiology_semantic_tokenizer/dual_component_band_word_natural_context_v1.yaml), [natural-observation and integer-interface checks](../tests/test_spatial_pattern_as.py), [word decode tests](../tests/test_band_exchange_words.py) |
| Cross-branch sampling actual single-word audit | [Independent actual-ID background readout, control replay and code support](scripts/audit_cross_branch_words.py) | [Same native-array sampling contract](configs/physiology_semantic_tokenizer/dual_component_cross_branch_sampling_v1.yaml), [single-ID namespace and person-fold tests](../tests/test_cross_branch_word_review.py) |
| Continuous per-patch EEG frequency exchanges, independent review | [Physical covariance, quota, voltage-SSE audit and train galleries](scripts/audit_adaptive_band_exchange.py) | [Frozen exchange protocol](configs/physiology_semantic_tokenizer/dual_component_adaptive_band_exchange_v1.yaml), [quota and physical objective tests](../tests/test_adaptive_band_allocation.py) |
| Exhaustive saved-AS training-position support and disjoint-person raw-ID replication | [Full support and exact source/readout checks](scripts/run_actual_id_full_support.py), `--review`, `--replication-review`, `--render` | [Frozen existing-IDs-only scope, original held panel and raw-mean gates](configs/physiology_semantic_tokenizer/dual_component_actual_id_full_support_v1.yaml), [existing support invariants](../tests/test_actual_id_sample_support.py) |
| Hard source-attribute groups and atomic EEG IDs | [Paired new-source dictionary and true native readout](scripts/run_stratified_source_atoms.py) | [Frozen K512 two-source contract](configs/physiology_semantic_tokenizer/dual_component_stratified_source_atoms_v1.yaml), [hard routing, atomic replay and train-boundary tests](../tests/test_stratified_source_dictionary.py) |
| Current-patch forward atomic gain diagnostic | [Frozen actual K512 parents and forward gain feasibility](scripts/audit_forward_atomic_gain.py) | [Train-only continuous-first contract](configs/physiology_semantic_tokenizer/dual_component_forward_gain_v1.yaml), [gain1 endpoint and local projection tests](../tests/test_forward_gain_dictionary.py) |
| Anatomically local EEG increment for Hb allocation | [Strong frozen Hb reference with matched global/local/cyclic EEG](scripts/diagnose_hb_local_eeg_source.py) | [Original-panel train-only contract](configs/physiology_semantic_tokenizer/dual_component_hb_local_eeg_v1.yaml), [fixed geometry, feature support and paired allocation tests](../tests/test_hb_local_eeg_source.py) |
| Fixed-sum and allocation error on frozen four-branch waveforms | [Paired integer replay and exact loss decomposition](scripts/audit_fixed_sum_component_error.py) | [Registered fixed-pair comparison](configs/physiology_semantic_tokenizer/dual_component_fixed_sum_error_v1.yaml), [analytic sharp-floor test](../tests/test_fixed_sum_error.py) |
| Single-ID quantization of anchored EEG carriers | [Subject-fold training and full integer replay](scripts/run_anchored_carrier_dictionary.py) | [Matched carrier/attribute distance contract](configs/physiology_semantic_tokenizer/dual_component_anchored_dictionary_v1.yaml), [dictionary and integer decoder tests](../tests/test_anchored_carrier_dictionary.py) |
| EEG gain updated from preceding decoded integers | [Matched fixed/adaptive subject-fold training and independent `--review` replay](scripts/run_backward_adaptive_carrier.py) | [Bounded train-only contract](configs/physiology_semantic_tokenizer/dual_component_backward_adaptive_carrier_v1.yaml), [state causality, reset and integer replay tests](../tests/test_backward_adaptive_carrier.py) |
| Training-learned balanced electrode-band roles | [Role fitting and subject cross-validation](scripts/run_learned_spatial_roles.py) | [Capacity and CV contract](configs/physiology_semantic_tokenizer/dual_component_learned_roles_v1.yaml), [role, mask and integer reload checks](../tests/test_learned_spatial_roles.py) |
| Compact frequency-specific spatial coordinates before quantization | [Train-subject reachability](scripts/run_frequency_spatial_carrier.py) | [Frequency, capacity and stopping contract](configs/physiology_semantic_tokenizer/dual_component_frequency_spatial_v1.yaml), [complex phase, missing-slot and replay checks](../tests/test_frequency_spatial_carrier.py) |
| Matched transient-source temporal-coordinate dimension curve | [Same native-array runner](scripts/run_frequency_spatial_carrier.py) | [Train-only support and effective-dimension contract](configs/physiology_semantic_tokenizer/dual_component_empirical_temporal_rate_v2.yaml), [short-window DC, output-band and frozen-replay checks](../tests/test_frequency_spatial_carrier.py) |
| Fixed staggered real-time samples before atomic quantization | [Same native-array train-subject runner](scripts/run_frequency_spatial_carrier.py) | [Sampling, covariance and stopping contract](configs/physiology_semantic_tokenizer/dual_component_staggered_time_carrier_v1.yaml), [actual-sample alias, missing-slot and replay checks](../tests/test_staggered_time_carrier.py) |
| Frozen actual AS integers under natural 30-second context changes | [Native-array pattern audit](scripts/audit_spatial_tokenizer_patterns.py) | [Same-physical-piece, observation-mask and frozen-template contract](configs/physiology_semantic_tokenizer/dual_component_spatial_as_natural_context_v1.yaml), [native-ID and true added-two-second checks](../tests/test_spatial_pattern_as.py) |
| Closed-loop spatial EEG innovation dictionaries | [Matched direct and predicted single-ID coding](scripts/run_spatial_predictive_dictionary.py), `--review` and `--pca-diagnostic` | [Train-only closed-loop tree, real decoder prefix and resource contract](configs/physiology_semantic_tokenizer/dual_component_spatial_predictive_v1.yaml), [prefix, missing-site and saved-model checks](../tests/test_spatial_predictive_dictionary.py) |
| Fixed native-role K8192/K512 dictionaries with one emitted integer | [Same-source information and nested selection](scripts/audit_transient_actual_words.py), [actual integer waveform](scripts/run_transient_spatial_search.py) | [Role information contract](configs/physiology_semantic_tokenizer/dual_component_role_mixed_as_v2.yaml), [bounded waveform falsification](configs/physiology_semantic_tokenizer/dual_component_role_mixed_waveform_v1.yaml), [role stitching and integer-only endpoints](../tests/test_role_mixed_spatial.py) |
| Historical Hb parent from unchanged semantic integers | [Train-subject label distillation](scripts/run_hb_parent_distillation.py) | [Frozen-ID and HbO-only sum contract](configs/physiology_semantic_tokenizer/dual_component_hb_parent_distillation_v1.yaml), [categorical-context and exact-O/quiet checks](../tests/test_hb_parent_distillation.py) |
| Real HbO sum residual from unchanged semantic integers | [Full-pattern training-subject readout](scripts/run_hb_native_id_readout.py) | [Additive ID, sparse-pair and waveform-control contract](configs/physiology_semantic_tokenizer/dual_component_hb_native_id_readout_v1.yaml), [support, edge and physical-coordinate checks](../tests/test_hb_native_id_readout.py) |
| Hb allocation from paired integers at a fixed total | [Train-subject fitting and integer replay](scripts/run_hb_native_id_readout.py), [signed-gain feasibility diagnostic](scripts/audit_hb_pair_gain_feasibility.py) | [Fixed-sum head](configs/physiology_semantic_tokenizer/dual_component_hb_pair_allocation_v1.yaml), [fold-specific gain contract](configs/physiology_semantic_tokenizer/dual_component_hb_pair_signed_gain_v1.yaml), [allocation checks](../tests/test_hb_pair_allocation.py), [gain feasibility tests](../tests/test_hb_pair_gain_feasibility.py) |
| EEG local energy from the transmitted montage IDs | [Train-subject calibration and replay](scripts/run_spatial_as_local_gain.py), [actual assignment audit](scripts/audit_spatial_tokenizer_patterns.py) | [Frozen local calibration](configs/physiology_semantic_tokenizer/dual_component_spatial_as_local_gain_v1.yaml), [decoder test](../tests/test_spatial_local_gain.py), [full-montage assignment test](../tests/test_spatial_pattern_as.py) |
| Hb component and sum corrections | [Component calibration](scripts/calibrate_joint_hb_output.py), [sum residual cross-validation](scripts/fit_hb_x_residual.py) | [Conserved energy fit](configs/physiology_semantic_tokenizer/joint_hb_component_transfer_gain_v2.yaml), [residual fit](configs/physiology_semantic_tokenizer/joint_hb_x_residual_v1.yaml), [integer residual checks](../tests/test_hb_residual_readout.py) |
| Hb low-rank native-array integer readout | [Train-subject comparison](scripts/run_hb_neighbor_readout.py) | [Neighbor contract](configs/physiology_semantic_tokenizer/dual_component_hb_neighbor_readout_v1.yaml), [missing/quiet and incremental predictor checks](../tests/test_hb_neighbor_readout.py) |
| Hb frozen G1 atomic O child vocabulary | [Train, review and admitted evaluation](scripts/run_hb_parent_leaf.py) | [Initial contract](configs/physiology_semantic_tokenizer/dual_component_hb_parent_leaf_v1.yaml), [matched direction contract](configs/physiology_semantic_tokenizer/dual_component_hb_parent_leaf_v2.yaml), [fixed model fit-size contract](configs/physiology_semantic_tokenizer/dual_component_hb_parent_leaf_fit1024_v1.yaml), [emitted-output consistency repair](configs/physiology_semantic_tokenizer/dual_component_hb_parent_leaf_emitted_gain_v1.yaml), [atomic parent/sum checks](../tests/test_hb_parent_leaf.py) |
| Hb low-amplitude observations | [Input ambiguity audit](scripts/audit_hb_supervision_ambiguity.py), [silent integer pair](scripts/run_hb_silent_atom.py) | [Collision tests](../tests/test_supervision_ambiguity.py), [silence and ordinary replay](../tests/test_silent_observation_tokenizer.py) |
| EEG diagnostic bounds and exposure verification | [Affine oracle](scripts/audit_eeg_affine_oracle.py), [saved-panel review](scripts/audit_joint_panel_exposure.py) | [Gate regression](../tests/test_joint_tokenizer_decision.py), [inference input contract](../tests/test_joint_tokenizer_bundle.py) |

The [dense continuation/G2/G4 contract](configs/physiology_semantic_tokenizer/dual_component_base_g124_v1.yaml)
uses the same `check`, `synthetic`, `base-train-plan`, `base-train` and report stages.
It preserves the original dense checkpoint/optimizer/RNG at step 7,000 in a fresh
run, and references the completed 4,000-step evaluation without rewriting it.
Its dependency-aware controller runs this continuation alongside the new EEG-32
and Hb structured-decoder warmups, then launches their paired continuous/K128 arms.
All runs stop at 10,000 total steps. At the user's request, this version disables
forward/replay comparison in synthetic and measured evaluation; actual integer
decoding remains the scoring path. Hb local shapes read the current code alone,
while per-color context corrections are bounded at 25% of its norm. Zero local
shape forces zero correction; the original difference loss acts on patch joins.
`test_dual_component_base_followup.py` covers these structural constraints,
the cancelled comparison, unchanged migrated continuation and the parallel schedule.

The [G3 contract](configs/physiology_semantic_tokenizer/dual_component_base_g3_v1.yaml)
uses the same `check`, `synthetic`, `base-train-plan`, and supervised `base-train`
entrypoints. It validates the audited G2 dimension and closed G1 budget, then reuses
the exact G1 EEG waveform-64 warmup and initialization identities. One seed-17 K128
candidate makes 9,000 new updates. Total steps 1,001–2,000 blend the actual training
decoder input from continuous to hard-quantized latents; EMA and commitment stay
unchanged, and all evaluation remains integer-only without replay comparison.
Transition probes, fixed milestones, optimizer/RNG state and the 2,000-step checkpoint
are retained. Two independent device pilots choose one GPU for the single candidate.
An explicit `--devices cuda:1` pins both the resource pilot and the formal candidate
to GPU1; the requested device is retained in the run plan and no other GPU is piloted.
`test_dual_component_base_g3.py` checks the main forward path, matched VQ controls,
hard-only evaluation, immutable external warmup and exact mid-transition recovery.

The [D0/D1/H1 contract](configs/physiology_semantic_tokenizer/dual_component_base_diagnostics_v1.yaml)
uses the same runner. The `base-d0` stage reads the matched G1 and G4 final evaluations,
training moments, IDs and checkpoints. It exports REFED subject/position statistics,
zero and training-mean controls, native-unit Hb errors by dataset, the signed S/O error
cross term, G4 cap frequencies, and held-out ID examples and shape readouts.
The readouts compare training-fitted pooled and hierarchically weighted ID templates
against zero and dataset/position mean shapes on common valid support.
The figures show deterministic samples of up to 64 complete validation patches per ID.
The `base-train` stage runs two continuous target-input diagnostics and two
decoder-only Hb fits. The exact source checkpoint and G0 identities govern H1.
Use the allowlisted GPU UUIDs in the contract for both the resource pilot and training.
The [targeted tests](../tests/test_dual_component_base_diagnostics.py) cover branch
isolation, fixed assignments and EMA, matched decoder initialization, and shape controls.

The [E1 contract](configs/physiology_semantic_tokenizer/dual_component_base_e1_v1.yaml)
uses `base-train` for one continuous EEG waveform model with eight internal positions
per two-second anchor. It compares the matched G1 four-position baseline at 10,000
steps, using full validation NMSE and REFED high-band power ratios. The resource pilot
uses the two permitted GPUs and selects one for the fresh warmup and continuation.
[Targeted tests](../tests/test_dual_component_base_e1.py) cover the time grid, kernel
sizes, matched support and the two-band comparison.

The [phase A contract](configs/physiology_semantic_tokenizer/dual_component_base_pattern_a_v1.yaml)
uses `check`, `synthetic`, then supervised `base-pattern-audit` on G1 EEG, G1 Hb and
G4 Hb. It extends the retained D0 galleries with all-code support, conditional
variances, held-out attribute readouts and real overlapping validation contexts.
The [phase B contract](configs/physiology_semantic_tokenizer/dual_component_base_matrix_v1.yaml)
uses `check`, `synthetic`, `base-train-plan` and supervised `base-train`.
The H-TQ and E-Q continuations import the original matched warmups. E-TC and
E-TQ share one new EEG target-input warmup. Both target-input encoders see their
own branch target during fitting, dictionary initialization and scoring.
[Phase A tests](../tests/test_dual_component_base_pattern_a.py) and
[phase B tests](../tests/test_dual_component_base_matrix.py) exercise these paths
on development fixtures. For the retained November 20 runs,
[`render_base_pattern_matrix_review.py`](scripts/render_base_pattern_matrix_review.py)
accepts `--output <fresh-export>` and rebuilds the paired validation review,
searchable PDF with PNG figures and detailed workbook from the final evaluations.
Use the same entry with `--phase-c --output <fresh-export>` for component/color
contrasts and the paired single-ID versus sequence shape readouts.
The [D/E contract](configs/physiology_semantic_tokenizer/dual_component_base_band_de_v1.yaml)
uses `check`, `synthetic` and supervised `base-band-de` in the same training runner.
It freezes training-record band scales, benchmarks both permitted devices and
compares each eligible continuation against its matched baseline. The
[loss and stopping tests](../tests/test_dual_component_band_de.py) cover phase,
masking, scale boundaries and the joint screening rule. Once terminal,
the same review exporter accepts `--phase-e --output <fresh-export>` for
paired intervals, retained checkpoint trajectories and the decision report.
For each measured X-K128 comparison, `base-band-atlas --seed <seed> --candidate
baseline|balanced` applies the frozen A readouts to that actual checkpoint
before final export; it reuses the G0 reader and the same real-patch gallery.

The [TK-STATE-v1 training/acceptance design](../docs/EXPERIMENT_PLAN.md#tk-state-v1)
starts with the bounded [state pilot contract](configs/physiology_semantic_tokenizer/dual_component_state_v1.yaml)
in `scripts/run_dual_component_tokenizer.py`: `--stage check`, `--stage synthetic`,
metadata-only `--stage state-plan`, then supervised `--stage state-pilot`.
`state-prepare`, `state-benchmark` and controller-owned `state-one` expose the
same stages. The completed long-window S/O v2 reader supplies a metadata-selected
training-only panel. EEG X/S/O share one 200 Hz antialias conversion; Hb retains
10 Hz. The three equal-exposure controls are continuous S/O, K128 S/O and K128
raw-waveform quantization. `test_dual_component_state.py` covers source boundaries,
resampling/closure/masks, time-block isolation, frozen training scales, raw-only
loss, integer replay and synthetic state/null controls. Pilot within-subject
input holdout is separate from the later A–F acceptance and protected evaluation.

For a completed pilot, `--stage state-visualize --output <completed-run>
--analysis-output <fresh-direct-child> --device cuda:0` exports token patterns,
complete code-use distributions, training-fitted PCA and two-seed t-SNE through
the same runner and `src/analysis/dual_component_tokenizer.py`. Pattern panels
use fully scored anchors from nonoverlapping inputs; occupancy uses every legal
training crop. Conditional teacher components, full integer-sequence decodes
and observed patches are labelled separately. All codebooks include the raw
quantization control; its branches are named codebook 1/2. The local HTML index
links PNG figures, numerical coordinates and a bitmap-only PDF figure atlas.
This freezes model weights and uses only the retained pilot training population.

The same completed run supports `--stage state-reconstruction` with a fresh
`--analysis-output`: paired S/O targets and integer-sequence reconstructions,
their sum against observed X, and the independent continuous S/O control.
Each dataset/modality supplies fixed P10/P50/P90 total-error examples from
within-subject holdout; REFED short parents and the retained Visual Hb error
concentration are explicit supplements. Full 30-second curves, masks, IDs,
latents and frozen-scale metrics accompany the PNG, HTML and bitmap PDF.
The export checks exact predicted additivity, integer replay, unchanged model
state and agreement with retained per-position evaluations before rendering.

The [reconstruction refinement](../docs/EXPERIMENT_PLAN.md#tk-state-reconstruction-v1)
uses [`dual_component_state_refinement_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_state_refinement_v1.yaml)
through the same runner: `--stage check`, `--stage synthetic --output <fresh-run>`,
then supervised `--stage state-refine --output <same-run>`. Controller-owned
`state-refine-one` trains each fixed job. The parent pilot arrays and masks are
read-only; dense extended-budget and sampled-waveform paths share the 30-second,
200 Hz EEG / 10 Hz Hb interface. Continuous/K128/K512 expose reconstruction and
quantization effects separately. `test_dual_component_state.py` also covers
waveform-grid shapes, integer replay, branch isolation, gradients, adjacent-mask
loss and exact warmup/continuation budgets.
Frozen re-evaluation and the fixed-example before/after atlas use
`--stage state-refine-report --output <terminal-run> --analysis-output <fresh-child>`.
Training failures remain in their original records; this stage never updates weights.

The versioned bounded iteration contracts `dual_component_state_iteration_r1_v1.yaml`
through `dual_component_state_iteration_r3_v1.yaml` use the same `check`, `synthetic`,
supervised `state-refine` and controller-owned `state-refine-one` stages. Their
explicit task lists determine the job count; retained waveform warmups supply
initialization without changing the pilot or refinement contracts. `state-diagnose`
exports frozen dense/waveform K128/K512 patterns, masked level/shape decomposition,
2/6-second slopes and full health to a fresh `--analysis-output`. The bounded
campaign index and independent retained-array audit use `state-iteration-summary`.
`test_dual_component_state.py` covers the new schema, masked shape objective,
frozen training monitors, synthetic dynamics and baseline-inclusive plotting limits.
For a chromophore-separated reading atlas, the analysis owner's
`render_state_hb_pattern_detail(candidate_export, parent_export, fresh_export)`
reads retained diagnostic arrays, verifies paired sample/target identity and saves
its reproduction command. Code selection uses only complete training patches.

The matched Hb local-shape sequence uses
[`dual_component_hb_id_shape_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_hb_id_shape_v1.yaml)
with the existing runner: `--stage check`, `--stage synthetic --output <fresh-run>`,
then supervised `--stage hb-shape-sequence --output <same-run>`. Its controller
first analyzes the retained waveform K128, continuous and R2 references, measures
worker throughput, and schedules the declared channel and shape-loss contrasts.
The contract owns conditional E2/repeat selection and the training budget.
Analysis keeps both colors separate and compares reconstruction, real ID groups,
training-fitted ID/sequence readouts and same-patch context stability. Targeted
coverage is in `test_dual_component_state.py`, including channel coefficients,
input ablation, single-ID-only gradients, shared/fresh warmups and readout leakage.

Completed masked-state numerical results can be independently audited and plotted
with `scripts/analyze_masked_state_results.py --run <completed-run> --output <fresh-export>`.
It reads retained numerical tables only, checks hierarchical subject aggregation,
and produces PNG figures, a local HTML analysis and a bitmap-figure PDF with selectable text.

The frozen-token masked-state suite uses the same runner with
[`dual_component_masked_state_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_masked_state_v1.yaml):
`--stage check`, `--stage synthetic --output <fresh-run>`, metadata-only
`--stage masked-state-plan`, then supervised `--stage masked-state`.
The controller owns export/throughput trials, checkpoint-resumable jobs and
final subject-level analysis. `masked-state-export` and `masked-state-one` are
controller children; `masked-state-report` reads completed job results.
The 30-second tokenizer remains frozen while its 2-second S/O IDs are jointly
masked. Public train/validation subjects retain their original partition;
test/protected arrays are excluded before reads. The model owner is
`src/foundation/dual_component_masked.py`; source adaptation is in
`src/data/dual_component_tokenizer.py`. `test_dual_component_masked_state.py`
covers hidden-ID isolation, temporal/spatial/modality interactions, matched
mask exposure, read boundaries, support deduplication, training-only priors,
subject aggregation and a temporary-data train/evaluate/resume integration.

The 30-second global latent health pilot uses the same dual-component runner
with [`dual_component_latent_health_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_latent_health_v1.yaml).
Run `--stage check`, `--stage synthetic --output <fresh-run>`, then supervised
`--stage latent-health --output <same-run>`. Its metadata plan, retained Hb
preparation, resource benchmark, continuous warmups and finite K128/K512 jobs
share that run; `latent-health-plan`, `latent-health-prepare` and
`latent-health-benchmark` expose the same stages for inspection. HbO/HbR are
jointly reconstructed, with one latent ID per 2-second patch and independent
S/O decoders over all 15 same-branch time patches. Only retained public
train/validation Hb targets enter measured training. EEG uses synthetic
interface checks pending the independent supervision decision. The
[design](../docs/EXPERIMENT_PLAN.md#30-秒全局交互与潜在词表健康度起点2026-11-05)
owns interpretation; `test_dual_component_latent_health.py` covers partition
rejection, global support, branch isolation, integer replay, subject metrics,
dataset-specific collapse and exact continuation budgets.

The native-voltage target feasibility pilot uses the existing dual-component
runner with [`dual_component_voltage_teacher_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_voltage_teacher_v1.yaml):
`--stage check`, `--stage voltage-teacher-plan`, then supervised
`--stage voltage-teacher-pilot` / `--stage voltage-teacher` and
`--stage voltage-teacher-report --analysis-output <fresh-export>`.
It compares retained S/O labels to an EOG-covariance Wiener estimate with and
without a newly calibrated Tak power prior, on a fixed public development
panel. Synthetic known-voltage recovery, paired-label identities and estimator
responses to added artifacts/neural bursts are distinct endpoints. No tokenizer
is trained. The [design](../docs/EXPERIMENT_PLAN.md#eeg-电压监督目标的有限可行性试验2026-11-05)
owns the interpretation; `test_dual_component_voltage_teacher.py` covers the
native transform, covariance, pairing and pre-read partition boundaries.

The next voltage separation experiment uses the same runner with
[`dual_component_voltage_separation_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_voltage_separation_v1.yaml):
`--stage check`, supervised `--stage synthetic`, metadata-only
`--stage separation-plan`, supervised `--stage separation-pilot` / `--stage separation`,
and `--stage separation-report --analysis-output <fresh-export>`.
Its frozen EEG-only frontend compares state-guided spatial projection and
neighbor prediction against empirical Hb guidance, EEG guidance, phase-shuffled
guidance, PCA and random directions. Independent waveform dictionaries test hard
decoding at declared rates. The [data contract](../docs/DATA_CONTRACT.md#eeg-spatialtemporal-voltage-separation--2026-11-05)
owns its retained public boundary; `test_dual_component_voltage_separation.py`
covers projection, temporal support, perturbation scaling and actual O decoding.

The bounded follow-up uses the same runner with
[`dual_component_voltage_power_cap_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_v1.yaml):
`--stage check`, `--stage synthetic`, metadata-only `--stage power-cap-plan`,
then supervised `--stage power-cap`. It reuses one named ocular-perturbation
window and six training windows to compare frozen statistical, frozen physical
and refitted physical power references, including neural-like controls. Every
carrier, including the exact voltage complement, receives the local constraint;
post-synthesis violations remain diagnostics. `test_dual_component_power_cap.py`
covers the exact projection, its non-invariance counterexample, training-only
calibration, full-carrier reconstruction and the bounded input contract.
The separately frozen
[`Hb-weighted contract`](configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v1.yaml)
uses the same stages with explicit EEG/HbO/HbR objective multipliers. It changes
neither reporting scales nor final voltage-target rules, and performs a weighted
90-second synthetic fit before the measured paired fits.
The [second weight setting](configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v3.yaml)
and its [stable-projection continuation](configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v4.yaml)
retain the same bounded panel. A continuation checks the scientific contract and
weights before reusing accepted fits or initializing an unfinished fit from its
retained trajectory. Direct SVD in channel-normalized, whitened coordinates
avoids squaring the condition number; it does not alter the reference or cap.
The [result index](RESULTS_INDEX.md#eeg-spatialtemporal-voltage-separation--2026-11-05)
links the paired comparisons and preserves unsuccessful weight/solver attempts.
The [fixed four-subject examples contract](configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_examples_v1.yaml)
uses `--stage power-cap-plan`, then supervised `--stage power-cap`; it reuses
the accepted reference and known Fp1 addition, and fits one preselected window
per other validation subject. `--stage power-cap-report --report-output <fresh>`
exports the original and new examples with total O and paired ΔO comparisons.
Tests cover metadata-only selection, the fixed scope/weights and failed-pair
denominators. The reusable setting rationale is owned by
[METHOD_RATIONALE](../docs/METHOD_RATIONALE.md#hb-prioritized-joint-fitting-for-voltage-targets--2026-11-06).
The [two-case posterior contract](configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_posterior_v1.yaml)
uses the same stages and report entry. Its data owner adds the geometry-selected
POOz/occipital ROI to a separate 17-electrode view, calibrates only from the same
six training windows, verifies unchanged regional-fit reuse, and includes the
direct O1/O2 perturbation in the occipital joint fit. It preserves the previous
front-facing examples and uses the same waveform, weights and voltage cap.
The read-only `--stage power-cap-method-report` uses the examples contract,
`--output <completed_examples>`, `--posterior-run <completed_posterior>`, and
`--report-output <fresh_export>` to explain the retained voltage computation.
It replays the named B/G pairs against retained arrays, exports intermediate
waveforms and a Chinese PDF/HTML report, and does not refit SSMs or read raw/test
signals. Run with at most two single-thread workers under the usual supervisor;
include the installed `pdflatex` directory in that service's `PATH`. Report-only
revisions can reuse the authored content and bitmap figures through
`render_source_diagnostics_report.py`, with a fresh output directory. Numerical
replay checks are covered by `test_dual_component_power_cap.py`.
The same examples contract supports `--stage power-cap-normalization` with
`--output <completed_examples> --posterior-run <completed_posterior>
--report-output <fresh_export>`. This retained-array audit compares the original
training calibration and named examples before/after coordinate scaling,
training-SD standardization and objective weighting. It exports distributions,
within/between-window variance accounting and bitmap plots without SSM refits
or raw/test reads; its arithmetic tests use synthetic fixtures in the same suite.

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

Electrode-voltage targets and Hb boundary comparisons use the same runner with
[`dual_component_voltage_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_voltage_v1.yaml):
`--stage check`, `--stage synthetic`, then supervised `--stage voltage` under a
fresh run. The parent Tak outputs remain fixed; native EEG is revalidated through
the existing registry reader, corrected once per unique electrode, and used as
the new voltage target. `test_dual_component_voltage.py` covers native proxy replay,
zero-correction identity, overlapping ROI constraints, independent overlap decoding,
signed boundary metrics, fixed budgets and the final public-test read order.
The decoder-only [`voltage v2`](configs/physiology_semantic_tokenizer/dual_component_voltage_v2.yaml)
uses the same `voltage` stage to consume the completed v1 run. It verifies equal
data/training contracts and reuses completed artifacts; only fixed same-branch
100 ms synthesis and the validation-frozen selection change.

EEG-O recovery uses the same entry with
[`dual_component_eeg_o_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_eeg_o_v1.yaml):
`--stage check`, `--stage synthetic`, then supervised `--stage eeg-o` under a
fresh run. The finite development-only comparison uses observed band carriers,
bounded gain prediction, a training-patch memorization check and a labelled O
compression reference. Full-window filtering, gain context and independent
integer decoding remain distinct. `test_dual_component_eeg_o.py` covers the
pre-read split boundary, EEG-only features, analytic gains, gradient/context
checks, subject aggregation, fixed budgets and complete fixture evaluation.
The [reference-aware v2](configs/physiology_semantic_tokenizer/dual_component_eeg_o_v2.yaml)
uses the same stages and read-only v1 band features. It adds the producer's
observed EEG proxy and fixed training projection, with the original 90 s EEG
reference/filter support explicit. Its two gain-context candidates predict
three shared correction directions; Hb and teacher-output fields stay excluded.

Frozen S-token lag inspection uses
[`scripts/analyze_dual_component_token_lags.py`](scripts/analyze_dual_component_token_lags.py)
with [`dual_component_token_lags_v1.yaml`](configs/physiology_semantic_tokenizer/dual_component_token_lags_v1.yaml).
It replays the completed voltage model on public validation/test subjects, joins
native trial/ROI identities, and compares 2 s lag-cell counts against whole-trial
pairing permutations within subject, record, condition and session. Run its
synthetic tests, `--stage benchmark`, then supervised `--stage all`; no model is
fitted. `test_dual_component_token_lags.py` covers lag direction, context edges,
matched-null expectations, task-locked negative controls and synthetic delays.
After completion, `--stage report` renders the statistical summaries and
`--stage heatmaps` exports the EEG/fNIRS, EEG/lag and expected-Hb-rank triptychs.
Codeword order and the 24-word overview selection use training evidence only;
full 128-word views retain the vocabulary. Group/individual and overview/full
views declare their separate shared color scales. The lag heatmap shows
descriptive divergence from matched random pairings, not significance.

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

The E0-v2 [visual audit renderer](https://github.com/Kevin65536/PID-MCM/blob/bb38ed236604be7c891fcd617723411852057c55/experiments/scripts/visualize_e0_v2_audit.py),
[review registrar](https://github.com/Kevin65536/PID-MCM/blob/bb38ed236604be7c891fcd617723411852057c55/experiments/scripts/finalize_e0_v2_visual_review.py),
and formal-v3 [best-variance plotter](https://github.com/Kevin65536/PID-MCM/blob/bb38ed236604be7c891fcd617723411852057c55/experiments/scripts/plot_best_variance_full_trajectories.py)
are retired from the working tree. These fixed Git revisions preserve their
historical source: the E0-v2 tools write back audit records, and the plotter
reloads measurements and fits a model before writing into its dated run.
They are not current report-export entrypoints. Retained reports and regression
tests remain in place; current exporters use fresh versioned output directories.

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
The output directory must be new or empty. Both PDFs embed PNG figures exported
at 300 dpi, with searchable body text, tables and captions; SVG files are retained
only as editable figure sources.
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
