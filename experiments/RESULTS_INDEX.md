# Retained result index

_Evidence surface updated 2026-10-25; payload-pruning record retained from
2026-07-30_

This index identifies the experiment material kept for routine reading,
comparison, and paper review. A retained result is defined by its
conclusion and provenance package, not by keeping every intermediate tensor or
checkpoint.

This file owns retention and evidence locations, not current project state. Status
terms in its tables describe the retained artifact at its recorded snapshot. Use the
generated [`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md) for current execution and
scientific verdicts. Its owning record is
[`research_state/registry.json`](../research_state/registry.json); experiment designs
and access boundaries remain in the [experiment plan](../docs/EXPERIMENT_PLAN.md)
and [data contract](../docs/DATA_CONTRACT.md).

## Find results by question

Suite names below are relative to `runs/physiology_semantic_tokenizer/`.
Follow an evidence entry to its exact run, owning summary, endpoint tables and
report links. The [entrypoint/config/test map](README.md#entrypoint-config-and-test-map)
provides the corresponding implementation route; [run-directory guidance](runs/README.md)
explains the path hierarchy and reading order.

| Research question | Suite root | Evidence entry |
| --- | --- | --- |
| Which information survives hard tokens, and do time/region relations or a shape-plus-scale architecture help? | `functional_tokenizer/` | [Frozen relationships and local shape tokens](#frozen-relationships-and-local-shape-tokens--2026-10-25) |
| How do model-compatibility scores vary across the registered public inventory and frozen observation panels? | `model_compatibility/` | [Population distributions and typical curves](#model-compatibility-distributions-and-typical-curves--2026-10-23) |
| Can the frozen observation family meet separate EEG/Hb tolerances, and which failures remain numerical? | `source_observability/` | [Shared-driver conditional feasibility](#shared-driver-conditional-feasibility--2026-10-22) |
| Does the missing independent common-volume direction restrict sustained 30/90 s Hb reconstruction? | `shared_driver_reconstruction/` | [30/90 s volume-drive diagnostic](#continuous-hb-compatibility-and-independent-volume-direction--2026-10-22) |
| Do local channels, geometry and independent dynamics contribute separately? | `shared_driver_spatial/` | [Complete spatial experiment](#local-spatial-complete-experiment--2026-10-20), [initial software evidence](#local-spatial-software-and-metadata--2026-10-16) |
| How do the diagnostic families compare across datasets? | Multiple suites, resolved by the report evidence map | [REFED and four-dataset comparison](#refed-backfill-and-four-dataset-comparison--2026-10-13) |
| What does a fixed shared driver reconstruct, and how do observation and solver choices affect it? | `shared_driver_reconstruction/` | [Fixed-structure semantics](#fixed-structure-physiology-semantics--2026-09-28), [EEG proxy rates](#eeg-proxy-sampling-rate-comparison--2026-10-10), [reconstruction diagnostics](#reconstruction-and-measurement-diagnostics) |
| Are fitted parameters repeatable and supported away from boundaries? | `shared_driver_parameter_stability/` | [Parameter stability and common interior](#parameter-stability-and-common-interior--2026-10-10) |
| What output support and continuous-state constraints do the observations provide? | `ssm_output_support/` | [Output support and continuous states](#ssm-output-support-and-continuous-states--2026-10-10) |
| Which observation semantics survive mechanism and hardware controls? | `observation_semantics/` | [Observation semantics](#observation-semantics--git-retention-2026-10-10) |
| What do response dynamics and continuous representations recover? | `semantic_response_discovery/` | [Response dynamics and continuous semantics](#response-dynamics-and-continuous-semantics--2026-10-02-run) |
| Do finite responses and independent observations improve hidden-feature prediction? | `ssm_strengthening/` | [Finite response and independent observations](#finite-response-and-independent-observations--2026-10-03) |
| Do regional and spectral coordinates add shared information? | `shared_driver_modes/` | [Regional spectral shared drivers](#regional-spectral-shared-drivers--2026-10-02) |
| What can the fitted components be attributed to? | `shared_driver_attribution/` | [Structured-component attribution](#structured-component-attribution--2026-10-02) |
| How sensitive is the conditional teacher to observation mismatch and driver priors? | `shared_driver_teacher_robustness/` | [Conditional teacher robustness](#conditional-teacher-robustness--2026-10-01) |
| What do native Hb pairs and measurement processing show? | `data_quality_audit/` | [Native Hb direction prevalence](#native-hb-direction-prevalence--2026-09-28), [measurement diagnostics](#reconstruction-and-measurement-diagnostics) |
| What did earlier identifiability, inference and temporal diagnostics establish? | `t3_*`, `t3c_*`, `step5/`, `step5a_inference_consistency/`, `ssm_overnight/` | [T3 and Step5 evidence](#t3-identifiability-and-step5-inference), [overnight snapshot](#overnight-evidence-snapshot--2026-09-10), [Step5A0 snapshot](#step5a0-evidence-snapshot--2026-09-07) |
| What is retained from the original teacher and tokenizer qualification? | `e0_*`, `e1_*`, `e2_*`, `r0*`, `r1*`, `r2*` | [E0–E2 and R-series evidence](#e0e2-and-r-series-qualification-evidence), [Token Atlas](#token-atlas) |

[Comparison-method evidence](#comparison-methods) and [Croce validation](#croce-validation)
keep their separate package owners. [Retention and migration history](#retention-and-migration-history)
records storage changes and dated cleanup separately from the result-reading routes.

Some run and checkpoint paths below point to local Git-ignored artifacts that remain
useful in this workspace but are not part of the tracked paper record. For manuscript
Methods/Results, start with the linked human-readable reports and campaign tables;
use local payloads only when a report explicitly requires them.

## Frozen relationships and local shape tokens — 2026-10-25

The completed [TK-FUNCTIONAL parent summary](runs/physiology_semantic_tokenizer/functional_tokenizer/20261011_v1/summary.json)
owns its eight-arm training, hidden-Hb comparisons and mechanism controls. Its
24 failed native trials remain in the original 5,008-trial denominator. New
probes use only the shared retained subset and frozen public subject split.

| Probe | Contract | Numeric evidence |
| --- | --- | --- |
| Frozen QT/Q0/CT patch semantics and temporal/regional feature comparisons | [Relation contract](configs/physiology_semantic_tokenizer/functional_relation_probe_v1.yaml) | [Per-subject summary](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_relation_v1/summary.json), [paired analysis](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_relation_v1/analysis.json) |
| New local shape IDs, continuous scales and current/past-token Hb prediction; SSM vs matched smoothing | [Shape contract](configs/physiology_semantic_tokenizer/functional_shape_probe_v1.yaml) | [Per-subject summary](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_shape_v1/summary.json), [paired analysis](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_shape_v1/analysis.json) |
| Correct temporal and regional pairing vs condition-matched donors, using frozen shape exports | [Pairing contract](configs/physiology_semantic_tokenizer/functional_shape_relations_v1.yaml) | [Per-subject summary](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_shape_relations_v1/summary.json), [paired analysis](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_shape_relations_v1/analysis.json) |

Retain each run's resolved configuration, source archive, launch/resource record,
atomic tasks and summary. Git retains these records, paired analyses, reports,
figures and the small numeric shape encoders with their dictionaries. The public
parent's coordinate/identity manifest, resolved contract, terminal summaries and
the cited QT mechanism records preserve the split, failure and interpretation
provenance. Dense hard-ID/scale exports, continuous references, parent signal
caches and old training checkpoints remain local; the new shape exports can be
rebuilt from the retained numeric encoders and the declared parent cache. Pilot
records retain throughput evidence without duplicating their model exports.
The [combined interpretation](runs/physiology_semantic_tokenizer/functional_tokenizer/20261025_shape_v1/REPORT.md)
is a communication view of these sources. Same-input waveform readability,
offline relationship classification and actual next-patch observed-Hb prediction
are distinct endpoints; none establishes physiological source recovery.

## Model-compatibility distributions and typical curves — 2026-10-23

The [versioned contract](configs/physiology_semantic_tokenizer/model_compatibility_v1.yaml)
and [design](../docs/EXPERIMENT_PLAN.md#五项模型相容性指标的总体分布与典型曲线2026-10-23)
separate the native public Hb inventory from the retained frozen output-support
and spatial panels. The numerical owners are the
[census summary](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/summary.json),
its record/subject tables and per-record profiles, and the
[panel summary](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/panels/summary.json)
with task JSON/NPZ and common-parameter tables. Keep the resolved configuration,
explicit identities, source snapshots, synthetic controls,
[supervised launches](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/launch.json),
[resource measurements](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/resources_resolved.json),
and [verification](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/verification.json).

The Chinese [report](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/report_v4/TECHNICAL_REPORT.pdf)
and [bitmap figure atlas](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/report_v4/FIGURE_ATLAS.pdf)
show full distributions, subject weighting and explicitly selected waveform examples.
The [spatial stability migration](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/derived_spatial_stability_v1/migration.json)
owns corrected raw spatial projections, and the
[dynamic-energy migration](runs/physiology_semantic_tokenizer/model_compatibility/20261023_v1/panels/dynamic_diagnostics_v2_migration.json)
owns corrected constant-nuisance diagnostics; original arrays, failed checks and
earlier exports remain retained. The cross-modal distance uses the actual discrete
observation operators; it is not the continuous third-order weak-form value.
Nonlinear values are legal conditional reconstruction witnesses under a finite
optimization budget, not converged optima. Wide common-parameter sets do not
identify physiology, and unscorable windows remain in the denominator record.

Git retains the census rows/profiles, panel task records and arrays, selected
examples, correction overlays, source snapshots, verification and launch logs,
and the final v4 PDF/HTML, bitmap figures and formula sources. The earlier
[synthetic visual guide](runs/physiology_semantic_tokenizer/model_compatibility_guide/20261023_v7/report/TECHNICAL_REPORT.pdf)
is retained with its arrays and provenance as a separate educational example.
Superseded report exports, rendered page previews and duplicate ZIP bundles
remain local; no retained experiment evidence is deleted by this selection.

## Shared-driver conditional feasibility — 2026-10-22

The [versioned contract](configs/physiology_semantic_tokenizer/shared_driver_feasibility_v1.yaml)
and [design](../docs/EXPERIMENT_PLAN.md#当前共享驱动模型的表达边界与联合可行性2026-10-22)
use the explicit retained public spatial parent. Evidence is owned by the
[run summary](runs/physiology_semantic_tokenizer/source_observability/20261022_feasibility_v1/summary.json),
its fixed stage manifests and task JSON/NPZ arrays. Preserve the resolved config,
source snapshot, [supervised launches](runs/physiology_semantic_tokenizer/source_observability/20261022_feasibility_v1/launch.json),
[resource choice](runs/physiology_semantic_tokenizer/source_observability/20261022_feasibility_v1/resources_resolved.json),
and [independent verification](runs/physiology_semantic_tokenizer/source_observability/20261022_feasibility_v1/verification.json).
The Chinese [PDF](../docs/report/20261022_shared_driver_feasibility_v1/TECHNICAL_REPORT.pdf)
and [HTML](../docs/report/20261022_shared_driver_feasibility_v1/TECHNICAL_REPORT.html)
are communication exports of this evidence.

Conservative float64 representation bounds, numerical minimum-norm residuals,
single-position vascular witnesses and hard-EEG-tolerance searches have distinct
interpretations. A legal low-error trajectory need not be an optimizer optimum
or a physiologically identified solution. The report retains numerical failures,
the separate strict joint criterion, conditional non-entry of parameter profiles,
and the mismatch controls that pass observation tolerances despite driver error.

## Direct unforced p−v decay diagnostic — 2026-10-23

The [six-arm contract](configs/physiology_semantic_tokenizer/shared_driver_decay_v1.yaml)
changes only the p−v relaxation rate and its actual state-equation input, with
the retained public identities fixed by the
[plan](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/plan.json).
The numerical owner is the versioned
[corrected summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/evidence_v2/summary.json),
with [paired endpoints](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/evidence_v2/paired_contrasts.json)
and [fit metrics](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/evidence_v2/metrics.csv).
The [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/report_v4/REPORT.pdf)
and [bitmap figure atlas](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/report_v4/FIGURES.pdf)
summarize the retained arrays; [independent verification](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/evidence_v2/independent_verification.json)
checks their scores, dynamics, physical constraints and paired aggregation.
Earlier exports remain retained; v4 includes concise example identities,
visible fixed parameters and corrected comparison labels without changing results.
The [migration record](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/evidence_v2/migration.json)
separates physically infeasible fixed-upstream counterfactuals from valid original
fits. It preserves the original task records and fitted arrays, with no refits.
Keep both source snapshots, all original task JSON/NPZ and replay overlays,
resolved configuration, launch records, and
[resource measurements](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261023_decay_v1/resources.json).
Git retains both source snapshots and the per-export renderer sources in
`source_snapshot.tar.gz`; extract it in this run directory to restore their
recorded paths. The archive preserves source bytes and excludes compiled caches.
Fit failures, counterfactual failures, state recovery and observation errors have
separate denominators. The older output-component experiment below remains a
distinct retained result.

## Continuous Hb compatibility and independent volume direction — 2026-10-22

The versioned [contract](configs/physiology_semantic_tokenizer/shared_driver_volume_drive_v1.yaml)
reads the exact public continuous-record parents named in its
[plan](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261022_volume_drive_v1/plan.json).
Retain the [owning summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261022_volume_drive_v1/summary.json),
[paired endpoints](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261022_volume_drive_v1/metrics.csv),
[primary-source literature audit](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261022_volume_drive_v1/literature.json),
[Chinese bitmap report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261022_volume_drive_v1/report_v2/REPORT.pdf),
and [independent score/observation checks](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261022_volume_drive_v1/report_v2/verification.json).
The original run's resolved configuration, source snapshot, launch records,
resource choices, complete synthetic/linear/measured task JSON and trajectory NPZ
remain the audit package. Git retains the byte-preserving `source_snapshot.tar.gz`;
extract it in this run directory to restore the recorded source paths and patch.
Report v1 is the retained first export; v2 corrects one
unsupported superscript glyph without changing numerical evidence.

Full-window reconstruction, Hb-only compatibility, blocked interpolation,
same-center shape comparisons and equal-capacity exchange controls have distinct
endpoints. A fitted common component does not identify its anatomical source;
linear minimum residuals may require nonphysical state excursions. This package
does not qualify a physiological teacher or replace the original measured data.

## Four independent fit/recovery diagnostics — 2026-10-21

The [owning design](../docs/EXPERIMENT_PLAN.md#拟合观测信息与生理恢复的四套独立诊断2026-10-21)
separates four scientific questions. Each suite retains its own fixed task plan,
resolved configuration, source snapshot, supervisor/resource records, complete
failures, endpoint tables and independent numerical verification. Current execution
and interpretation remain in the project registry.

| Question / suite | Owning numeric evidence | Independent technical report |
| --- | --- | --- |
| Data, priors and physiological recovery / `source_observability` | [Summary](runs/physiology_semantic_tokenizer/source_observability/20261021_v1/summary.json), [verification](runs/physiology_semantic_tokenizer/source_observability/20261021_v1/verification.json) | [PDF](../docs/report/20261021_source_observability_v2/TECHNICAL_REPORT.pdf), [HTML](../docs/report/20261021_source_observability_v2/TECHNICAL_REPORT.html) |
| Channels, geometry and dynamic dimension / `spatial_information` | [Summary](runs/physiology_semantic_tokenizer/spatial_information/20261021_v1/summary.json), [verification](runs/physiology_semantic_tokenizer/spatial_information/20261021_v1/verification.json) | [PDF](../docs/report/20261021_spatial_information_v3/TECHNICAL_REPORT.pdf), [HTML](../docs/report/20261021_spatial_information_v3/TECHNICAL_REPORT.html) |
| Conditional cross-modal information / `crossmodal_information` | [Summary](runs/physiology_semantic_tokenizer/crossmodal_information/20261021_v1/summary.json), [verification](runs/physiology_semantic_tokenizer/crossmodal_information/20261021_v1/verification.json); separately versioned matched-generator positive control under `20261021_positive_v1/` | [PDF](../docs/report/20261021_crossmodal_information_v3/TECHNICAL_REPORT.pdf), [HTML](../docs/report/20261021_crossmodal_information_v3/TECHNICAL_REPORT.html) |
| Physiological change versus observation mismatch / `mechanism_attribution` | [Summary](runs/physiology_semantic_tokenizer/mechanism_attribution/20261021_v1/summary.json), [verification](runs/physiology_semantic_tokenizer/mechanism_attribution/20261021_v1/verification.json) | [PDF](../docs/report/20261021_mechanism_attribution_v2/TECHNICAL_REPORT.pdf), [HTML](../docs/report/20261021_mechanism_attribution_v2/TECHNICAL_REPORT.html) |

The reports retain portable content JSON, searchable body text and tables, PNG
figures/formulas, LaTeX source, figure atlases and delivery checks. Retained measured
parents keep their original identity; no historical evidence is rewritten. Compact
plans, task/endpoint tables, summaries, verification scripts/results and source
snapshots are retained in Git. Each run's `terminal_records.zip` losslessly retains
the original terminal JSON (and mechanism task CSV) files with their relative paths;
the archive was checked byte-for-byte against the local records. The spatial suite's
large `plan.json`, `information_results.json` and `profile_results.json` are retained
as lossless `.json.gz` copies; their working originals remain unchanged. A checkout
can decompress these copies beside the archives before an explicit evidence replay.
Large per-task arrays and intermediate previews stay
local at the named run/export paths. Source-level replays require the explicit parent
arrays recorded by each contract; default tests use synthetic fixtures instead.
Resource pilots, including the mechanism suite's initial CLI startup rejection, stay
at their original versioned paths and do not count as formal scientific evidence.

## Local spatial complete experiment — 2026-10-20

The frozen `SSM-LOCAL-SPATIAL-v1` contract was executed on its registered public
Single-Trial panel, following the separate learned-calibration synthetic controls.

- [Complete synthetic summary](runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_synthetic_v1/summary.json)
  links the six independently calibrated scenario summaries, plans, training/selection
  records and terminal cells. Keep their known-truth diagnostics, resolved configuration,
  source snapshot, launch/resources records and independent `verification.json`.
- [Complete measured summary](runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2/summary.json)
  retains numerical failures and unavailable wrong-EEG controls separately. Keep the
  immutable plan, coordinates, calibration/selection records, per-cell scores and arrays,
  [independent verification](runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2/verification.json),
  [supervised launch](runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2/launch.json),
  resources, execution history and source snapshot.
- `20261020_measured_v1` is resource-pilot evidence, not the final measured experiment.
  Its initial base-interpreter preparation failures and corrected repository-environment
  throughput measurements remain intact; final v2 preparation did not reuse those files.
- Communication export:
  [36-page PPT](../docs/report/20261020_local_spatial_v5/LOCAL_SPATIAL_REPORT.pptx),
  [slide PDF](../docs/report/20261020_local_spatial_v5/LOCAL_SPATIAL_REPORT.pdf),
  [39-page technical document](../docs/report/20261020_local_spatial_v5/TECHNICAL_REPORT.pdf)
  and [complete delivery bundle](../docs/report/20261020_local_spatial_v5/LOCAL_SPATIAL_DELIVERY.zip).
  The export retains per-page evidence links, full CSV tables, LaTeX source, PNG figures,
  and [numeric/layout/bitmap checks](../docs/report/20261020_local_spatial_v5/delivery_checks.json).
  Earlier report-export attempts remain local with their original identities.

Git retains the final plans, calibration/selection records, summaries, source snapshots,
execution and verification records, complete measured-cell CSV, report files and source
assets, plus the resource pilot's failure records. Prepared signal arrays, per-cell
JSON/NPZ payloads, preview images and the duplicate delivery ZIP remain in their existing
local directories; array-level verification requires those retained local payloads.

## Local spatial software and metadata — 2026-10-16

The [versioned contract](configs/physiology_semantic_tokenizer/shared_driver_spatial_v1.yaml)
and [design](../docs/EXPERIMENT_PLAN.md#局部多通道观测与空间动态维数2026-10-16)
define the local multi-channel suite. Retain these bounded implementation checks:

Git retains the final v3 JSON evidence and source snapshots listed below. Rebuildable
synthetic NPZ arrays, runtime locks and superseded v1/v2 directories remain local.

- [Synthetic smoke v3](runs/physiology_semantic_tokenizer/shared_driver_spatial/20261016_implementation_smoke_v3/summary.json):
  per-scenario known-readout checks, with scoring scales frozen from separate training
  seeds. Keep the resolved contract, source snapshot, execution record, scenario
  coordinates and complete cell diagnostics. This is not learned-calibration,
  measured-data or physiological qualification evidence.
- [Metadata plan v3](runs/physiology_semantic_tokenizer/shared_driver_spatial/20261016_metadata_plan_v3/plan.json):
  subject splits, frozen sensor identities, target-optode exclusions, window support,
  wrong-EEG availability and planned cells. Signal-array readers were disabled during
  this audit; no measured fit was performed.
- Preserve `20261016_implementation_smoke_v1` as superseded debugging evidence:
  its scales used evaluation statistics and must not support hidden-target claims.
  Preserve the v1 metadata plan and both v2 runs with their original source identities.
  The v3 checks reproduce v2 numerics and metadata while completing the read-dependency
  source snapshots and per-stage execution history.

## REFED backfill and four-dataset comparison — 2026-10-13

The [versioned protocol](../docs/EXPERIMENT_PLAN.md#refed-公开实验补齐与四数据集比较2026-10-13)
and [report evidence map](configs/physiology_semantic_tokenizer/refed_four_dataset_report_v1.yaml)
give the exact retained and REFED roots for the ten diagnostic families. Each
family keeps its own resolved contract, terminal task records, score tables,
failures, verification and source/supervisor identity. Shared/disjoint-optode
experiments are excluded. The base
[run](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261013_refed_physiology_semantics_v3/summary.json)
also retains planar-neighbour sensitivity and the separately versioned
[four-dataset transfer summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261013_refed_physiology_semantics_v3/four_dataset_transfer_v1/summary.json).
The response owner's
[four-dataset statistics](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261013_refed_v2/four_dataset_comparison_v1/summary.json)
recompute dataset-equal effects and the registered nine-test family from the
disjoint old/new cell tables; they do not average previously reported intervals.

Retain fitted/profile arrays needed for numerical replay, native-record and
split identities, prepared representative windows, calibration inputs and the
explicit inherited-synthetic references. Frozen synthetic encoder checkpoints
remain under their original response run; public REFED probes retain that
encoder identity. Preparation caches are rebuildable and remain local. The new
artifacts stay local under the existing ignored run/report roots; reviewed
executable configs and report source are part of the code reading surface.
No historical evidence or failed attempt is removed or overwritten.

The [54-page Chinese PPT](../docs/report/20261013_refed_four_dataset_v1/REFED_FOUR_DATASET_REPORT.pptx),
[PDF](../docs/report/20261013_refed_four_dataset_v1/REFED_FOUR_DATASET_REPORT.pdf)
and [bitmap figure atlas](../docs/report/20261013_refed_four_dataset_v1/FIGURE_ATLAS.pdf)
are versioned communication exports. Keep their figure CSVs, LaTeX/PNG formula
sources, slide-to-evidence map, renderer snapshot and
[delivery checks](../docs/report/20261013_refed_four_dataset_v1/delivery_checks.json).
The report reads the final parameter
[summary](runs/physiology_semantic_tokenizer/shared_driver_parameter_stability/20261013_refed_v2/summary.json),
including the selected-sharing synthetic calibration and the preserved
summary-only software correction; it does not replace any run's fact source.

## Parameter stability and common interior — 2026-10-10

The [versioned contract](configs/physiology_semantic_tokenizer/shared_driver_parameter_stability_v1.yaml)
and [protocol](../docs/EXPERIMENT_PLAN.md#参数稳定性共同内点与拟合改进2026-10-10)
define the E1–E6 suite. The retained
[owning summary](runs/physiology_semantic_tokenizer/shared_driver_parameter_stability/20261010_v1/summary.json)
links the frozen metadata pairs, development coordinates/calibration, method
selection, measured and synthetic results, full profile nodes, and numerical
audits. `pair_metrics.csv` and `parameter_estimates.csv` retain the registered
denominators. Original fit files and the failed numerical audit are preserved;
the summary records the qualification exclusions without changing that evidence.

The versioned [Chinese PPT](../docs/report/20261010_parameter_stability_v2/PARAMETER_STABILITY.pptx)
and [PDF](../docs/report/20261010_parameter_stability_v2/PARAMETER_STABILITY.pdf)
retain PNG figures, slide provenance, renderer source and export checks. These
communication assets and the selected evidence are explicitly tracked at their
original paths despite the default artifact ignore rules. Retain terminal task
JSON, full profile nodes, original fitted trajectories for numerical replay,
cross-fit residuals, calibration inputs, source snapshots, supervisor logs and
software recovery records. The three report examples also retain prepared
targets so the delivered figures can be regenerated. Other prepared targets are
rebuildable caches and stay local; array-only tests do not require run artifacts.
The earlier presentation export remains local at its original versioned path.

## SSM output support and continuous states — 2026-10-10

The [contract](configs/physiology_semantic_tokenizer/ssm_output_support_v1.yaml)
and [protocol](../docs/EXPERIMENT_PLAN.md#ssm-输出支持范围连续状态与跨模态约束2026-10-10)
define output-function profiles, native continuous-record comparisons with
matched boundary priors, and paired-EEG controls. Read the
[Chinese technical report](../docs/report/20261010_ssm_output_support_v1.md)
alongside the [owning summary](runs/physiology_semantic_tokenizer/ssm_output_support/20261010_v1/summary.json)
and [verification](runs/physiology_semantic_tokenizer/ssm_output_support/20261010_v1/verification.json).
Retain the resolved contracts, metadata plan, training coordinates, source
snapshots, supervisor/resource records, terminal task JSON, comparison tables,
and fitted/profile arrays needed to audit decomposition closure, hidden scores
and alternative solutions. The initial power calibration remains intact; its
separately versioned [background refinement](runs/physiology_semantic_tokenizer/ssm_output_support/20261010_v1/power_calibration_v2/summary.json)
retains the empirical training bank, moment checks, terminal tasks and paired
results. `report_analysis_v1/` holds the report's derived tables and PNG figures.
This package is explicitly retained in Git at its original paths. The tracked
arrays include every standard full-observation measured fit for report/state
audits, the prespecified records across all fitted conditions, the numerical
failure, all output-profile vectors, and seed-zero synthetic examples. The
development prepared records needed to regenerate the empirical background
and the prespecified profile records are also retained. Other fitted arrays,
evaluation preparation caches, pilot
arrays and power-example arrays remain local; repeating the full numerical
array audit requires those local payloads or regenerating them from the frozen
sources and declared native datasets. All terminal task records and complete
metric/profile/comparison tables are tracked, including unsuccessful outcomes.
No previous evidence package is replaced or deleted.

## EEG proxy sampling-rate comparison — 2026-10-10

The [versioned contract](configs/physiology_semantic_tokenizer/eeg_proxy_rate_v1.yaml)
and [protocol](../docs/EXPERIMENT_PLAN.md#eeg-功率代理采样率与实测重建2026-10-10)
define the native EEG rate comparison, fixed Hb target and driver-grid control.
The Git-retained package is
[`20261010_eeg_proxy_rate_v1`](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261010_eeg_proxy_rate_v1/summary.json):
`metrics.csv` and `paired_comparisons.csv` hold the per-fit and subject-block
comparisons; `preparation_summary.json` audits native 4 Hz replay;
`verification.json` checks arrays, metrics and the original baseline replay.
It includes the resolved contract, exact window/coordinate plan, source snapshot,
supervisor/resource records, synthetic checks, all terminal fit JSON/NPZ files,
the compact prepared proxy/target arrays needed to audit their scores, and the
final PNG overview/examples in `figures_v2/`. The runtime lock and superseded
run-root PNG export remain local. Native-record regeneration and historical
baseline replay still require the explicitly referenced datasets and parent
evidence. No old parent evidence is changed. The common
4 Hz EEG compatibility score and each method's own-rate EEG fit are distinct
endpoints; the former is not an inverse log-power aggregation.

## Observation semantics — Git retention 2026-10-10

The observation-semantics evidence is retained in Git at
[`observation_semantics/20261009_v2`](runs/physiology_semantic_tokenizer/observation_semantics/20261009_v2/summary.json)
and its bounded
[`C/D refinement`](runs/physiology_semantic_tokenizer/observation_semantics/20261009_refinement_v1/summary.json).
Each keeps the resolved contract, frozen source (excluding compiled caches),
supervisor launch, per-unit result/provenance JSON, complete metric and comparison
tables, fitted readouts and verification. Semantic-stability range/mask arrays
and the saved mechanism and event/hardware example arrays are also retained. The
[`Chinese PPT/PDF export`](../docs/report/20261009_observation_semantics_v3/observation_semantics_20261009.pptx)
and its adjacent `slide_sources.json`, validation and 23 source PNG figures trace
the report back to those owners. The superseded
[`pilot conclusion`](runs/physiology_semantic_tokenizer/observation_semantics/20261009_v1/pilot_conclusion.json)
and its contract/launch/provenance records are retained without changing their
bytes. Prepared native input NPZ caches, compiled caches, rendered page/contact
sheet previews and earlier presentation attempts remain local; full verification
and report regeneration require the prepared caches and the explicitly referenced
parent evidence. Current execution and interpretation are recorded only in the
research-state registry.

## Finite response and independent observations — 2026-10-03

The [strengthening contract](configs/physiology_semantic_tokenizer/ssm_strengthening_v1.yaml)
and [owning protocol](../docs/EXPERIMENT_PLAN.md#有限响应适配与独立观测约束2026-10-03)
define response selection, cross-fitted spatial residual readouts, mechanism
controls and the development factorial. Use the
[versioned evaluation summary](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/summary.json),
[paired endpoints](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/paired_hidden_metrics.csv),
[decision](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/decision.json)
and [verification](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/verification.json).
The [correction record](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v3/correction.json)
explains why raw-observation ridge predictions must not inherit SSM fit failures.
This version reuses all other fitted results and their original arrays; it does
not represent another training campaign.

Keep the [original fit run](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v2/run_manifest.json)
with its frozen sources, input hashes, training/selection identities, calibration
objects, terminal task records, decomposition/profile arrays, resource pilot and
supervisor logs. The v3 array paths depend explicitly on v2. Retain the
[prelaunch failure](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_v1/run_manifest.json)
and its validation/source record; that version launched no scientific fits.
Response curves, spatial maps, mechanism attribution, null comparisons,
confounding profiles and risk/coverage tables accompany the v3 summary.

The separately versioned [continuous-state contract](configs/physiology_semantic_tokenizer/ssm_state_continuity_v1.yaml)
has its own [summary](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/summary.json),
[state comparisons](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/state_continuity.csv),
[restricted profiles](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/confounding_profiles.csv)
and [verification](runs/physiology_semantic_tokenizer/ssm_strengthening/20261003_s3_v1/verification.json).
Retain its continuous truth/fit arrays, training scales, source identity,
validation and supervised launch. The phase-one decision's S3 field describes
that run's scope; this separate synthetic S3 run is indexed here and in the
registry. It uses future context for offline smoothing, with no continuous
measured-data confirmation. The reused public panel remains development evidence;
neither run authorizes fresh/protected evaluation or physical-teacher promotion.

## Response dynamics and continuous semantics — 2026-10-02 run

The [response contract](configs/physiology_semantic_tokenizer/semantic_response_discovery_v1.yaml)
and [owning protocol](../docs/EXPERIMENT_PLAN.md#响应动力学与连续语义原型2026-10-02)
define the EEG-only response comparisons, synthetic counterfactuals, independent
modality encoders and frozen public readouts. The
[terminal summary](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/summary.json)
and [verification](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/verification.json)
index execution and denominators. Retain the resolved contract, metadata split,
per-cell records, calibration choices, supervised launch/resource records, failed
training-resume logs, and immutable preparation/training/analysis source snapshots.

Use the [dataset-equal paired comparisons](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/dataset_equal_comparisons.csv)
and [primary null family](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/primary_null_specificity.csv)
for measured SSM endpoints. The
[synthetic counterfactual table](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/synthetic_counterfactual_response.csv)
preserves paired mechanism responses. The
[tokenizer summary](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/tokenizer/summary.json)
contains every fit, synthetic evaluation and public-probe record; the
[scenario/variant table](runs/physiology_semantic_tokenizer/semantic_response_discovery/20261002_v1/tokenizer/scenario_variant_seed_summary.csv)
keeps intervention and control separate and labels optimization seeds as repeats
of the same evaluation identities. Per-model `paired_changes.csv` retains false
and true semantic changes. Public-probe `records.csv`, `selection.json` and
`record.json` retain the common-context, O→O+S and equal-capacity comparisons.

The dated [detailed presentation](../docs/report/20261003_semantic_response_discovery_v1/SSM_SEMANTIC_RESPONSE_REPORT_v2.pptx)
and [bitmap-figure PDF](../docs/report/20261003_semantic_response_discovery_v1/SSM_SEMANTIC_RESPONSE_REPORT_v2.pdf)
connect the discussion, frozen design, aggregate results, native waveforms and
conditional component plots. The [export validation](../docs/report/20261003_semantic_response_discovery_v1/export_validation.json)
and per-slide source manifest accompany the communication export; the run records
remain the scientific evidence owner.

Keep selected encoder checkpoints for frozen representation reuse and probe
reproduction. Versioned `features/batch_v2` adds reversible training-gain
provenance without overwriting the earlier feature batches. Reconstructed units
refer to parent prepared coordinates, not raw native waveforms. This is exposed
public-cohort development evidence; synthetic recovery does not confer measured
physiological semantics, teacher qualification or VQ promotion.

## Regional spectral shared drivers — 2026-10-02

The [mode-driver contract](configs/physiology_semantic_tokenizer/shared_driver_modes_v1.yaml)
and [owning protocol](../docs/EXPERIMENT_PLAN.md#区域与谱模式共享驱动检验2026-10-02)
define fixed regional broadband/spectral coordinates, the training-selected
vascular readout, and equal-temporal-capacity reconstruction and hidden-feature
comparisons. The [terminal summary](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/summary.json)
indexes the numerical evidence. Retain the resolved contract, separate pilot and
formal source snapshots, supervised launch/resource/completion records, fixed
window and calibration identities, all cell records including failures and
unavailable matched donors, and full-observation decomposition arrays.

Use the [common-success paired comparisons](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/measured_paired.csv)
and [pairing nulls](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/pairing_nulls.csv)
for gains; successful-arm marginal means are descriptive and can have different
denominators. [Synthetic recovery](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/synthetic_summary.csv)
separates reference-centered modes, vascular input, and observation mismatch.
The [verification](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/verification.json)
checks identities, failure denominators, numerical metrics, and decompositions.
The [Chinese bitmap-figure PDF](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/report_v2/REPORT.pdf)
is a communication export with a retained renderer source and
[export validation](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/report_v2/render_validation.json).
The Git review package retains the metric and comparison tables, calibration
identities, launch/configuration and verification records, and the reviewed PDF.
The [source identity map](runs/physiology_semantic_tokenizer/shared_driver_modes/20261002_v1/source_snapshot_identity_git_export_v1.json)
resolves each pilot/formal source file to its recorded Git base or an unchanged
retained snapshot file. Full prediction/state arrays, prepared signals and page
previews remain local; original cell metrics and failure denominators are in the
retained CSVs. Parent prepared arrays and historical results retain their original
identities.

## Structured-component attribution — 2026-10-02

The [four-group contract](configs/physiology_semantic_tokenizer/shared_driver_attribution_v1.yaml)
and [owning protocol](../docs/EXPERIMENT_PLAN.md#结构性成分归属与分级语义检验2026-10-02)
define the attribution audit, equal-capacity direction and independent-region
completion, known-truth counterfactuals, EOG anchor check, and finite synthetic
typed-token prototype. The
[terminal summary](runs/physiology_semantic_tokenizer/shared_driver_attribution/20261002_v1/summary.json)
indexes their numerical endpoint tables. Retain the resolved contract, original
fitting source snapshot, supervised launch/resource records, frozen calibration
identities, all task records and failures, decomposition arrays, prototype
selection evidence, and the separate post-run verification/report source snapshot.
The [verification](runs/physiology_semantic_tokenizer/shared_driver_attribution/20261002_v1/verification.json)
checks complete denominators, exact parent prediction replay, decomposition,
training/evaluation separation, spatial target exclusion and EOG source identity.
Spatial and null support are reported separately; original parent evidence stays
at its existing paths.

The [Chinese PPT](../docs/report/20261002_ssm_component_attribution_v3/SSM_COMPONENT_ATTRIBUTION.pptx)
and [bitmap-figure PDF](../docs/report/20261002_ssm_component_attribution_v3/SSM_COMPONENT_ATTRIBUTION.pdf)
are communication exports with per-slide sources and
[export validation](../docs/report/20261002_ssm_component_attribution_v3/export_validation.json).
Earlier exports retain their layout-revision evidence. Source semantics remain
unresolved; the synthetic continuous-token prototype does not establish measured
teacher, tokenizer or VQ qualification.

The Git review package retains the endpoint CSVs (including failed fits), plans,
stage manifests, training coordinates/calibration choices, prototype selection
records, software/evidence checks, supervised launch logs, and exact fitting and
report source snapshots. The
[source bundle](runs/physiology_semantic_tokenizer/shared_driver_attribution/20261002_v1/source_bundle_git_v1.tar.gz)
preserves the original snapshot, report sources, launch patch and request bytes;
extracting it in the run directory restores their recorded paths. The delivered PPT embeds its
figure bitmaps; the companion PDF and both exports' source/validation records
are tracked. Original task arrays, derived auxiliary signals, prototype weights,
generated training batches, and earlier presentation binaries remain local.
Earlier export validation records retain the layout failures. This selection
does not delete or rewrite the local experiment evidence. The
[commit checks](runs/physiology_semantic_tokenizer/shared_driver_attribution/20261002_v1/git_review_20261003_v1/verification_v2.json)
record tests and contract validation against an isolated staged checkout.

## Conditional teacher robustness — 2026-10-01

The [four-arm contract](configs/physiology_semantic_tokenizer/shared_driver_teacher_robustness_v1.yaml)
and [owning protocol](../docs/EXPERIMENT_PLAN.md#观测失配与驱动先验的四臂检验2026-10-01)
define the fixed-H0 observation/prior comparison. Retain the
[run summary](runs/physiology_semantic_tokenizer/shared_driver_teacher_robustness/20261001_v1/summary.json),
resolved configuration, source snapshot, supervised launch and resource records,
calibration identities, all task records (including numerical failures), paired
endpoint tables, and teacher sensitivity/decomposition sidecars. The
[verification](runs/physiology_semantic_tokenizer/shared_driver_teacher_robustness/20261001_v1/verification.json)
checks record completeness, training separation, wrong-subject identities and
prediction decomposition. The public parent arrays and frozen coordinates remain
at their original 2026-09-28 evidence paths.

The [PPT](../docs/report/20261001_ssm_teacher_robustness_v3/SSM_TEACHER_ROBUSTNESS.pptx)
and [companion PDF](../docs/report/20261001_ssm_teacher_robustness_v3/SSM_TEACHER_ROBUSTNESS.pdf)
are communication exports; per-slide sources and export validation accompany
them. Earlier v1/v2 exports retain their font/layout failure records and are not
the delivered version. These artifacts do not confer tokenizer or physiological
teacher qualification.

## Main-method evidence

### Fixed-structure physiology semantics — 2026-09-28

The [terminal summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/summary.json)
owns this campaign's numerical conclusions. The requested
[Chinese presentation](../docs/report/20260928_ssm_physiology_semantics_v3/SSM_PHYSIOLOGY_SEMANTICS.pptx)
distinguishes matched synthetic recovery, conditional measured reconstruction,
parameter repeatability, hidden-feature completion and observation-source limits.
Its [executable contract](configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml)
keeps the existing dynamics and single driver without an independent slow common
component. Retain phase source snapshots, native QC and geometry exclusions,
training coordinates and splits, all parameter fits and failure denominators,
paired sharing comparisons, nulls, integration replays/refits and negative outcomes.

The [software verification](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/verification.json)
and [presentation verification](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/presentation_verification_v3.json)
record distinct checks. Preserve the interrupted hierarchy launch, failed technical
attempts, aggregation corrections, and earlier presentation exports with their
revision evidence. Presentation v3 is the reviewed delivery; LibreOffice PDF
previews are internal layout checks, not an additional technical report.

The Git review package retains the existing summaries, comparison tables,
completed and interrupted phase records, analysis corrections, software checks,
and reviewed presentation at their original paths. Start with the terminal
summary above, then follow these evidence routes:

| Review question | Retained evidence |
| --- | --- |
| Matched synthetic recovery and observation mismatch | [State recovery](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/synthetic_state_recovery.csv), [stress and hierarchy summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/synthetic_diagnostics_summary.json) |
| Reconstruction and paired parameter-sharing comparisons | [Common-window full denominators](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/reconstruction_common_later_windows.csv), [paired comparisons](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/sharing_paired_comparisons.csv), [regional and QC summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/reconstruction_summary.json) |
| Individual-parameter stability and failed fits | [Training parameters](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/training_parameters.csv), [repeat pairs](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/repeatability_pairs.csv), [task-stratified ICC](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/repeatability_icc.csv) |
| Missing-feature completion and pairing nulls | [Completion summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/completion_summary.csv), [null comparisons](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/completion_null_comparisons.csv) |
| Integration precision | [Full-observation and masked replay summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/integration_audit_summary.json) |
| Exact source and environment used by each phase | [Frozen source identity](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/source_snapshot_identity_git_export_v1.json), [environment](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/environment_snapshot.json), [resolved configuration](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1/resolved_config.yaml) |

The source identity maps each phase's recorded file set to the existing Git
commit or a retained differing source version. Identical differing versions are
kept once at an original snapshot path; the map gives the replacement path for
each phase. This is provenance for the dated experiment, not an active launcher
or a new scientific result. The original evidence bytes and verdicts are unchanged.
Native/derived arrays, caches, complete reconstruction/prediction tables and
earlier presentation binaries remain local. Figure bitmaps are embedded in the
tracked presentation. Historical-focus references to earlier runs remain local
where their separately indexed evidence has not been released.

### Native Hb direction prevalence — 2026-09-28

The [completed census](runs/physiology_semantic_tokenizer/data_quality_audit/20260928_hbo_hbr_prevalence_v1/manifest.json)
retains native/released Hb pair statistics, record provenance, subject summaries,
the executable contract, source snapshots and supervised launch logs. The
[report](runs/physiology_semantic_tokenizer/data_quality_audit/20260928_hbo_hbr_prevalence_v1/report_v3/REPORT.pdf)
explains the physiology, processing sensitivity, subject/channel variation and
limits of the retained-model association. Its main numerical source is
[summary.csv](runs/physiology_semantic_tokenizer/data_quality_audit/20260928_hbo_hbr_prevalence_v1/summary.csv).
Only `model_cases_v2.csv` / `model_association_v2.csv` are valid auxiliary model
tables; the [revision record](runs/physiology_semantic_tokenizer/data_quality_audit/20260928_hbo_hbr_prevalence_v1/analysis_revision_v2.json)
preserves the correction of synthetic cells mistakenly included in the first
auxiliary aggregation. Native census statistics were unaffected. Earlier failed
launch/export evidence stays retained; report v3 is the reviewed delivery.

Shared-driver results using the retained V3 optical motion branch remain dated
fits to those processed targets. The [same-window motion audit](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_optical_motion_audit_v1/report_v1/REPORT.pdf)
identifies algorithm-induced slow drift; those scores cannot establish recovered
physiology. The optical observation contrast retains separate targets and scales.

### Reconstruction and measurement diagnostics

| Stage | Retained authority / artifact | Why it remains |
| --- | --- | --- |
| Shared nonlinear gain and driver-amplitude prior | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_v1/manifest.json), [summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_v1/summary.json), [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_v1/report_v1/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_v1/report_v1/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_gain_prior_v1.yaml) | Preserve frozen source, supervised launches, synthetic truth, all shared-parameter starts, training failures and validation denominators, four-arm paired comparisons, state excursions and visual checks. Signal-SD ridge bias, optimization-budget limits and changing Hb locations prevent a physiological subject-parameter claim. |
| Shared-gain optimization-budget warm restart | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_budget_v1/manifest.json), [summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_budget_v1/summary.json), [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_budget_v1/report_v1/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_gain_prior_budget_v1/report_v1/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_gain_prior_budget_v1.yaml) | Preserve parent lineage, frozen source, supervised launches, objective-continuity checks, optimization traces and call accounting, all failed starts, and paired identity transitions. Inherited synthetic evidence is not an independent replication; added validation coverage does not improve previously successful predictions or establish physiological parameter validity. |
| Optical motion processing audit | [Summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_optical_motion_audit_v1/measured_summary.json), [independent analysis](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_optical_motion_audit_v1/analysis/independent_analysis.json), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_optical_motion_audit_v1/report_v1/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_optical_motion_audit_v1.yaml) | Preserve all 72 input identities, source provenance, 216 reproduction checks, all three pipeline traces, synthetic trend controls and fixed-identity figures. Algorithm intervention explains the retained slow-decline component; neither no-motion nor MNE is physiological truth. Units remain relative Hb. |
| Conditional optical observation map and shared gain | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_v1/manifest.json), [pilot](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_pilot_v1/manifest.json), [synthetic bitmap report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_v1/report_synthetic_v3/REPORT.pdf), [full report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_v1/report_v1/REPORT.pdf), [all-curve atlas](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_v1/report_v1/FULL_CURVE_ATLAS.pdf), [independent audit](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_conditional_optical_gain_v1/preparation_audit/final_verification.json), [contract](configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml) | Preserve exact within-pipeline parent targets and training SD, inherited baseline failures, new paired synthetic truth, engineering gain-recovery screen, actual beta/reference/relative gain, state/driver errors and conditional calibration assumptions. Low reconstruction error alone does not validate latent trajectory recovery. |
| Paired same-objective solver step control | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_conditional_step_control_panel_v2/manifest.json), [pilot](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_conditional_step_control_pilot_v2/manifest.json), [contract](configs/physiology_semantic_tokenizer/shared_driver_conditional_step_control_v2.yaml), [audit](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_conditional_step_control_panel_v2/report_v1/audit.json), [bitmap report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_conditional_step_control_panel_v2/report_v1/REPORT.pdf), [PPT](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_conditional_step_control_panel_v2/report_v1/SOLVER_RESULTS.pptx) | Retain both fresh arms, historical failures, exact prepared/training target and SD identities, all start traces and convergence reasons, CPU/wall timing, forward budgets, integration checks, source snapshot and supervisor/resource records. Training numerics only; no validation or physiology qualification. |
| Hb waveform mechanism and extra volume loading diagnostics | [Mechanism run](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_waveform_diagnostic_v1/manifest.json), [volume loading run](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_waveform_volume_fraction_v1/manifest.json), [bitmap report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_waveform_diagnostic_v1/report_v2/REPORT.pdf), [waveform figure](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_waveform_diagnostic_v1/report_v2/figures/waveform_mechanisms.png), [numerical failure recheck](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_waveform_diagnostic_v1/integration_failure_recheck_v1/summary.json) | Preserve the named difficult identity, exact parent targets and training SD, training-only parameter/loading selection, equal-dimension mechanism contrasts, original failed outcomes and versioned failure detail. Added-component reconstruction does not establish independent prediction or a unique physiological source. |
| Lower-bound tau versus driver-curvature diagnostic | [Run](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261010_tau_curvature_v1/manifest.json), [owning summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261010_tau_curvature_v1/summary.json), [paired metrics](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261010_tau_curvature_v1/metrics.csv), [waveforms](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261010_tau_curvature_v1/figures_v4/focus_waveforms.png), [all controls](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261010_tau_curvature_v1/figures_v4/constraint_comparison.png), [contract](configs/physiology_semantic_tokenizer/shared_driver_tau_curvature_v1.yaml) | Exact public MNE TDDR subject_01 fold-3 validation evidence; preserve fixed-target tau/curvature contrasts, EEG/flow/initial-prior controls, synthetic checks, failed iterations and linear out-of-domain screens. Git retains all terminal JSON/NPZ, source snapshots, logs, resource/audit records and `figures_v4`, plus the six original parent files required by this reader (manifest, config, named preparation metadata, training selection, cell result and trajectories). Earlier figure exports and the runtime lock remain local. Full-window conditional reconstruction in one subject does not establish physiological truth or population generalization. |
| Optical observation reconstruction contrast | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_optical_v1/manifest.json), [bitmap report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_optical_v1/report_v1/REPORT.pdf), [all curves](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_optical_v1/report_v1/FULL_CURVE_ATLAS.pdf), [synthetic optical controls](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_optical_v1/optical_controls/results.json), [contract](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_optical_v1.yaml) | Preserve both pipelines and all identities, training-only Hb scales, unchanged EEG, complete new measured failures, exact nonindependent synthetic inheritance, parameter/state diagnostics, and known-response loss controls. Different targets and SDs do not support a same-task NRMSE ranking. |
| Initial total-Hb / volume constraint | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_initial_v1/manifest.json), [synthetic bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_initial_v1/synthetic_report_v1/REPORT.pdf), [pilot](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_initial_pilot_v1/manifest.json), [contract](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_initial_v1.yaml) | Preserve frozen source, all fresh paired synthetic initial-ratio conditions, reduced-coordinate initialization, parameter/state recovery, and resource evidence. The planned old-target measured phase was not executed after the optical artifact finding. The equality is a model restriction, not an observed baseline or physiological normal-range assertion. |
| Fixed ROI log-flow sensitivity | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_flow_v1/manifest.json), [full report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_flow_v1/report_v2/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_flow_v1/report_v2/REPORT.pdf), [all 72 curves](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_flow_v1/report_v2/FULL_CURVE_ATLAS.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_flow_v1.yaml) | Preserve immutable pilot/formal source snapshots, exact parent prepared and baseline inheritance including failures, distinct new/inherited denominators, all weight arms, tau/driver/flow recovery, and the stationary baseline null check. Flow weights are engineering anchors, not empirical physiological ranges; inherited evidence is not an independent replication. |
| Fixed Hb location and shared nonlinear tau | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_tau_v1/manifest.json), [summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_tau_v1/summary.json), [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_tau_v1/report_v3/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_fixed_roi_tau_v1/report_v3/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml) | Preserve frozen source, supervised pilot and phase launches, fixed-AF7Fp1 preparation with unchanged EEG, paired synthetic truth, every training start and validation identity, and tau information/state diagnostics. Historical changing-channel scores are not same-target paired baselines; the montage and amplitude gauge are not physiological calibration. |
| Constrained nonlinear shared-driver reconstruction | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_fit_v1/manifest.json), [summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_fit_v1/summary.json), [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_fit_v1/report_v3/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_fit_v1/report_v3/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_nonlinear_fit_v1.yaml) | Preserve frozen source, launch and resource records, both phases, every start/failure, same-identity comparisons, observation-map metadata and state excursions. Retain the earlier report export. Low full-observation error is conditional on convergence and does not qualify hidden recovery or physiological parameters. |
| Original nonlinear dynamics with frozen shared driver | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_replay_v1/manifest.json), [summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_replay_v1/summary.json), [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_replay_v1/report_v2/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_nonlinear_replay_v1/report_v2/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_nonlinear_replay_v1.yaml) | Preserve exact parent linear fits, frozen-driver nonlinear trajectories, both integration resolutions, complete domain-failure denominator and same-success-subset comparisons. This is deterministic replay, not nonlinear refitting or parameter qualification. |
| Shared-driver reconstruction and physiological-parameter freedom | [Manifest](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_linear_screen_v1/manifest.json), [summary](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_linear_screen_v1/summary.json), [Chinese report](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_linear_screen_v1/report_v2/REPORT.md), [bitmap PDF](runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260926_linear_screen_v1/report_v2/REPORT.pdf), [contract](configs/physiology_semantic_tokenizer/shared_driver_reconstruction_v1.yaml) | Preserve frozen measurement folds/scales, synthetic truth, tau profiles, single-driver trajectories, unconstrained numerical residual diagnostics, state-domain failures and SVD rank sensitivity. Low linear Hb residual is not nonlinear physiology or teacher qualification. |
| Multi-band and geometry structural screen | [Corrected report, 2026-09-18](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/analysis_20260918_v2/REPORT.md), [corrected PDF](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/analysis_20260918_v2/REPORT.pdf), [annotated analysis](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/analysis_20260918_v2/analysis.json). [2026-09-18 geometry selection correction audit](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/geometry_selection_audit_20260918_v1.json); the original report's geometry-selection interpretation is withdrawn pending corrected predictions. [Chinese report](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/analysis_20260917_v1/REPORT.md), [PDF](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/analysis_20260917_v1/REPORT.pdf), [analysis tables](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/analysis_20260917_v1/analysis.json), [main summary](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v2/summary.json), [template noise correction](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_template_correction_v1/summary.json), [retained preparation failure](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_band_geometry_v1/manifest.json), [contract](configs/physiology_semantic_tokenizer/ssm_band_geometry_v1.yaml) | Paired local scalar/multichannel, merged/split bands and true/permuted geometry; preserve independent synthetic panels, complete measured identity denominators, same-feature linear controls and numerical/domain failures. Corrected template controls explicitly link to retained originals; primary endpoints unchanged. No structure promotion or teacher qualification. Generated evidence remains local. |
| Independent observation information and nonlinear counterfactuals | [Run manifest](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_calibration_v1/manifest.json), [record calibration table](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_calibration_v1/calibration_information.csv), [launch and resume](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_calibration_v1/launch.json), [pilot evidence](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_calibration_pilot_v1/summary.json), [pilot export recovery](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_calibration_pilot_v1/export_recovery.json), [contract](configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v1.yaml) | Retain metadata provenance, immutable source snapshot, independent synthetic calibration identities, paired physiology/observation counterfactuals, per-panel fits, approximate coverage and complete failure denominators. Conditional deterministic recovery only; no measured calibration or teacher qualification follows automatically. |
| Calibration attribution and oracle localization | [Full Chinese report](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_calibration_v2/report_v3/REPORT.md), [PDF](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_calibration_v2/report_v3/REPORT.pdf), [bitmap figure atlas](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_calibration_v2/report_v3/FIGURES.pdf), [analysis](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_calibration_v2/report_v3/analysis.json), [actual source audit](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_calibration_v2/source_audit.json), [contract](configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v2.yaml) | Retain paired factorials, complete failure denominators, common-center continuous support, projected information, profile resolution and basin checks as separate immutable runs, source snapshots, actual-file identities and all report exports. Synthetic recovery remains conditional; missing independent measured constraints keep calibrated C/D unexecuted. |
| New-measurement A–E retest | [Chinese report](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1/analysis_20260917_v1/REPORT.md), [PDF](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1/analysis_20260917_v1/REPORT.pdf), [analysis tables](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1/analysis_20260917_v1/analysis.json), [summary](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1/summary.json), [aggregation recovery](runs/physiology_semantic_tokenizer/ssm_overnight/20260917_measurement_retest_v1/aggregation_recovery.json), [contract](configs/physiology_semantic_tokenizer/ssm_measurement_retest_v1.yaml) | Fixed W, explicit relative Hb loading, chromophore attribution, W curves and bidirectional controls. Preserve the complete task denominator, numerical/physical failures and conditional nonexecution of gain/Q; no teacher qualification. Generated evidence remains local. |
| Measurement alignment revision | [Detailed Chinese alignment report](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_report_v2/REPORT.md), [PDF](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_report_v2/REPORT.pdf), [self-contained HTML](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_report_v2/REPORT.html), [bitmap figure atlas](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_report_v2/FIGURES.pdf), [Execution and fixed-denominator summary](runs/physiology_semantic_tokenizer/ssm_overnight/20260916_measurement_alignment_v3/summary.json), [generated report](runs/physiology_semantic_tokenizer/ssm_overnight/20260916_measurement_alignment_v3/OVERNIGHT_REPORT.md), [contract](configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml), [public loader smoke](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_v3/public_loader_smoke.json), [independent MBLL check](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_v3/mbll_independent_check.json), [Visual CH6 audit](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_dataset_alignment_v3/visual_ch6_quality.json) | Versioned unit/precision/support repair; retain source snapshot, task identities, original-input manifests, numerical/physical failures and first failed public join smoke. These interfaces do not grant teacher or tokenizer qualification. All generated payload remains local. |
| Four-dataset scaling audit and bounded SSM diagnostic | [Chinese report](runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/REPORT.md), [PDF](runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/REPORT.pdf), [self-contained HTML](runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/REPORT.html), [summary](runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/report_summary.json), [contract](configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml) | Actual raw/current amplitude and Hb-pair ratio audit, known-truth scaling controls, and fixed-fold pointwise measured sensitivity; distinct from v3 temporal qualification and tokenizer training. Generated report/evidence package remains local. |
| Training-frozen observation gain and real Hb prediction | [Report](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_predictive_separation_v1/REPORT.md), [manifest](runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_predictive_separation_v1/manifest.json), [contract](configs/physiology_semantic_tokenizer/hbo_hbr_predictive_separation_v1.yaml) | Retain blocked identities, synthetic controls, frozen training gains and physiology, hidden-HbR per-window errors, supervisor and source snapshots. Reference-anchored gain is not independent calibration; no C/D or teacher qualification. |
| Hb observation-layer and metric separation | [Report](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_observation_v1/REPORT.md), [manifest](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_observation_v1/manifest.json), [contract](configs/physiology_semantic_tokenizer/hbo_hbr_observation_v1.yaml) | Retain shared-gain profiles, paired metric/repair costs, exact gain aliases, synthetic independent-noise calibration and total-whitening negative control. Conditional effective-parameter recovery does not establish unique measured physiology or qualify the six-state model. |
| Hb-pair dynamic constraints and shared parameter adaptation | [Fixed-parameter dynamics](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_dynamics_v1/REPORT.md), [term decomposition](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_terms_v1/REPORT.md), [adaptation report with smooth-repair sensitivity](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_adaptation_v1/REPORT_v2.md), [adaptation manifest](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_adaptation_v1/manifest.json), [adaptation contract](configs/physiology_semantic_tokenizer/hbo_hbr_adaptation_v1.yaml), [boundary-mechanism report](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_boundary_v1/REPORT.md), [boundary manifest](runs/physiology_semantic_tokenizer/data_quality_audit/20260918_hbo_hbr_boundary_v1/manifest.json), [boundary contract](configs/physiology_semantic_tokenizer/hbo_hbr_boundary_v1.yaml) | Retain window/split identities, heldout metrics, parameter equivalence and bounds, conditional corrections, profiles, synthetic perturbation controls and source snapshots. Original discrete projections and v1 exports remain evidence; smooth v2 export identifies quadrature artifacts. Descriptive necessary-constraint diagnostics, not a complete stochastic-model rejection or unique physiological attribution. |
| SSM observation-contract v3 diagnostics | [Chinese visual report](runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/analysis_20260911_v2/REPORT.md), [PDF](runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/analysis_20260911_v2/REPORT.pdf), HTML export (local), [summary](runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/summary.json), [continuation identity](runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/continuation.json), [contract](configs/physiology_semantic_tokenizer/ssm_overnight_v3.yaml) | Temporal mean/covariance ablations, independent gain/Q panels, measured failures and technical continuation; synthetic recovery evidence does not establish measured teacher qualification |
| SSM overnight N1–N7 diagnostics | [Chinese visual report](runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/analysis_20260910_v2/REPORT.md), [full PDF](runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/analysis_20260910_v2/REPORT.pdf), HTML export (local), [vector figure atlas](runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/analysis_20260910_v2/FIGURES.pdf), [contract](configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml) | Complete task denominators, failure evidence, synthetic/measurement boundaries and G/W ablations; exploratory evidence without teacher qualification |
| SSM overnight N1–N6 predecessor | [retained automatic report](runs/physiology_semantic_tokenizer/ssm_overnight/20260909_overnight_v2_verified/OVERNIGHT_REPORT.md), [manifest](runs/physiology_semantic_tokenizer/ssm_overnight/20260909_overnight_v2_verified/manifest.json), [contract](configs/physiology_semantic_tokenizer/ssm_overnight_v1.yaml) | Original evidence and numerical comparison source for the N7 extension; repeated inputs/seeds are not independent replications |
| Step5 flow-domain and mask-specific observation repair | [`20260909_observation_repair_v2_verified/summary.md`](runs/physiology_semantic_tokenizer/step5/20260909_observation_repair_v2_verified/summary.md), [update histories and drift roots](runs/physiology_semantic_tokenizer/step5/20260909_observation_repair_v2_verified/replay.json), [contract](configs/physiology_semantic_tokenizer/step5_observation_repair_v2.yaml) | matrix-exponential drift-domain classification and visible-only interpolation operators; synthetic full/center/whole-modality comparison; nonlinear temporal qualification remains a prerequisite to a new measured comparison |
| Step5 observation contract repair and regression (retained v1) | [`20260908_observation_repair_v1/summary.md`](runs/physiology_semantic_tokenizer/step5/20260908_observation_repair_v1/summary.md), [numerical compatibility and refined replay](runs/physiology_semantic_tokenizer/step5/20260908_observation_repair_v1/verification_review.json), [contract](configs/physiology_semantic_tokenizer/step5_observation_repair_v1.yaml) | log-domain extraction repair, known-scale invariance on reused and independent synthetic inputs, explicitly approximate temporal reference, identical-input failure continuation and same-fold fixed SSM/linear control; numerical corrections work, measured teacher remains unqualified |
| Step5 observation adaptation diagnostic | [`20260908_observation_diagnostic_v1/summary.md`](runs/physiology_semantic_tokenizer/step5/20260908_observation_diagnostic_v1/summary.md), [numerical review](runs/physiology_semantic_tokenizer/step5/20260908_observation_diagnostic_v1/diagnostic_review.json), [contract](configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml) | controlled preprocessing bridge, original-training-only modality curves/residuals, nested-CV lag control and retained saturation failures; exploratory, no teacher admission |

### E0–E2 and R-series qualification evidence

| Stage | Retained authority / artifact | Why it remains |
| --- | --- | --- |
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

### T3 identifiability and Step5 inference

| Stage | Retained authority / artifact | Why it remains |
| --- | --- | --- |
| T3 Step 2 identifiability (complete exploratory negative) | [`detailed report`](../docs/analysis/20260902_T3_IDENTIFIABILITY_STEP2_REPORT.md) and local [`v3 summary`](runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.md) | fit-only beta/kappa/tau practical-identifiability endpoint unsupported; no qualification or promotion |
| T3 Step 3 three-session LOSO (complete exploratory negative) | [`detailed report`](../docs/analysis/20260902_T3_MULTISESSION_LOSO_STEP3_REPORT.md) and local [`v2 summary`](runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.md) | fit-only effective-kappa candidate worsened held-out recovery NLL versus fixed M0; parameter screen failed; no trait, qualification, or promotion claim |
| T3c Step 4 hierarchical admission (complete blocked gate) | [`detailed report`](../docs/analysis/20260903_T3C_HIERARCHICAL_STEP4_ADMISSION_REPORT.md) and local [`v3 summary`](runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/summary.md) | array-free admission found 6/8 prerequisites unmet; measured partial pooling not started; no trait, qualification, or promotion claim |
| T3c Step 4 composite synthetic T-P2 (complete negative) | [`detailed report`](../docs/analysis/20260903_T3C_COMPOSITE_SYNTHETIC_TP2_REPORT.md) and local [`v1 summary`](runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/summary.md) | both C1 gain/time directions failed; C2 not run; synthetic identifiability evidence only, measured hierarchy remains blocked |
| Step5A0 inference consistency (complete localization) | Local [`panel report`](runs/physiology_semantic_tokenizer/step5a_inference_consistency/20260907_step5a_localization_v1/summary.md), [`same-data oracle refinement`](runs/physiology_semantic_tokenizer/step5a_inference_consistency/20260907_step5a_oracle_quadrature_refinement_v1/summary.json), and [`reference CDF figure`](runs/physiology_semantic_tokenizer/step5a_inference_consistency/20260907_step5a_localization_v1/reference_cdf.pdf) | parameterization/Kalman checks passed; resolved G/Z short references reveal score/posterior differences; W reference unresolved; no calibration or teacher admission claim |
| Step5 staged continuation | Local [`detailed stage review`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/stage_review.md), [`measured report`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/summary.md), and [`UQ prerequisite decision`](runs/physiology_semantic_tokenizer/step5/20260907_b_measured_output_recovery_v3/uq_stage_report.md) | A0 limited calibration passed; A1 admitted W only; B did not qualify a measured teacher; comprehensive UQ was not executed |

The v3 package also retains its fixed task table, terminal ledger, split identities,
numerical audits, report metric tables and renderer identity. Per-trial trajectories,
prepared arrays, frozen source directories and detailed controller residual/replay
tables remain local under the named run. The interrupted predecessor's evidence is
retained separately; the continuation binding records inherited results without
replacing their original identities.

### Overnight evidence snapshot — 2026-09-10

The Git package retains Markdown/PDF visual reports, their bitmap figures,
aggregate metric views, run manifests, resolved configurations, software pilots,
candidate-rule summaries and source identities. The Markdown reports display
their figures on GitHub; the N7 report-linked figure atlas also remains tracked.
Self-contained HTML, vector originals, the duplicate v3 atlas and detailed
per-fit/truth tables are local after the
[2026-10-03 retention update](#git-retention-update--2026-10-03).
The frozen reports and their scientific conclusions are unchanged.

Per-task/per-trial tables, their compressed copies, task/status ledgers,
selection records, native inputs and prepared arrays remain local. They are not
required to read the published results. Rebuilding figures with the
[read-only renderer](README.md#active-ssm-entries) requires the retained local
run evidence; a cloud checkout is a result-reading package, not a per-output
audit or native-data replay package. Frozen manifests can name these local
artifacts without making them part of the published surface.

### Step5A0 evidence snapshot — 2026-09-07

The Git evidence package retains the stage summaries, manifests, resolved
configurations, failure diagnostics, report-linked sources and key figures in
their original run paths. Unreferenced frozen runner/inference copies and
superseded diagnostic grids are local after the
[2026-10-03 retention update](#git-retention-update--2026-10-03).
Per-case arrays, prepared measured inputs and full numerical grids also remain
local; manifest references identify that audit/replay surface.

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

The independent [downstream rebenchmark v1](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1)
retains its [frozen inventory, splits and job manifest](../comparative_methods/runs/downstream_rebench_v1/20261016_v2/),
[resource measurements](../comparative_methods/runs/downstream_rebench_v1/20261016_v2/resource_summary.json),
source snapshot and service record in the shared owning run. Method-specific final
fits remain under `<method>/runs/downstream_rebench_v1/20261016_v2/`; the final
`aggregate/` consumes that exact planned matrix. Keep the original smoke failures,
corrected REFED smoke and `input_audit_v2/` multimodal coverage audit together.
The superseded `20261016_v1/` dispatch retains its initial baselines and interrupted
work. The revised run preserves its first failed BrainFusion smoke and uses
`source_v2/` after the constant-channel fix; no dated source or failure is replaced.
These new public development outputs do not replace the dated evidence below.

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

## Retention and migration history

These dated records explain which payloads are tracked, local, rebuildable or
historically removed. They preserve the scope and identities of the original
storage actions; they do not authorize another cleanup or define a new run queue.

### Git retention update — 2026-10-08

The response-discovery and strengthening suites retain their executable contracts,
implementation and tests together with the following selected evidence in Git.
All newly registered evidence paths and the final response report's 57 source PNGs
are included. Existing dated evidence files are retained without modification.

| Suite | Retained in Git | Retained locally for audit/replay |
| --- | --- | --- |
| Response discovery, 2026-10-02 | Resolved contract, metadata split and plan, calibration, source snapshots/identities, launch/completion and failure logs, aggregate SSM comparisons/nulls, tokenizer summaries and paired changes, public-probe records/selection, final v2 PPT/PDF and figure provenance | Per-cell SSM tables/records/arrays, synthetic tokenizer per-example records, prepared/feature/mode caches, selected encoder weights and resumable optimizer checkpoints, pilot payloads and earlier report exports |
| Strengthening, 2026-10-03 v1/v2/v3 | Prelaunch failure record, original and correction source/launch identities, input/split/calibration records, original summaries/decisions, v3 measured and aggregate synthetic tables, response/spatial/profile/risk comparisons, correction record and verification | Per-task training/evaluation/profile payloads and full synthetic metric tables; v3 still refers to the retained local v2 arrays |
| Continuous-state S3, 2026-10-03 | Frozen contract/source, scales, pilot/launch/completion records, all terminal task JSON, state/profile/paired tables and verification | Continuous truth and fitted trajectory arrays and compiled caches |

A checkout supports review of the reported comparisons and failure denominators;
replaying individual fits or regenerating all report panels requires the local
payloads listed above. These payloads remain unchanged on disk.

### Git retention update — 2026-10-03

This update removes 165 historical run files (130.8 MiB of uncompressed
tracked content) from the Git reading surface. Their original local files remain
unchanged, and their previous tracked copies remain available at commit
`87402efbeb3dd6ebd6104d0f9a6629b696c5971d`. The later reports and consolidated
stage reviews in the result sections own the reading routes; this changes
storage, not numerical results or scientific verdicts.

| Historical group | Retained in Git | Retained locally for audit/replay |
| --- | --- | --- |
| Step5A / Step5, 2026-09-07–09 | Stage summaries and reviews, manifests, resolved contracts, failure diagnostics, registered observation-repair evidence and report-linked sources/images | Unreferenced runner/inference copies, unresolved U3 numerical grids and redundant standalone figures |
| Overnight N1–N7 / observation v3, 2026-09-09–11 | Markdown/PDF reports, their bitmap figures, the report-linked N7 atlas, aggregate comparisons, manifests, source/continuation identities and verification records | Self-contained HTML, vector originals, the duplicate v3 atlas, task/status ledgers and detailed per-fit/truth tables |
| Hb calibration v1/v2 and pilots, 2026-09-18/24 | Final reports, figures, summaries, paired effects, failure tables, manifests and source audit | Pilot metric tables, compressed task records, source copies and detailed parameter/profile records |

All previously tracked registry evidence paths and all images embedded by the
retained Markdown reports remain tracked. The frozen v3 report's four appendix
links to `fit_status.csv`, `synthetic_state_truth.csv`,
`missing_support_truth.csv` and `processed_visible_truth.csv` now require the
local evidence or the historical Git copies. Historical report bytes are unchanged.
The full-tree Git history is retained; this pruning reduces a future checkout's
files, not an existing clone's object history. Recent shared-driver evidence,
protected comparison packages and sealed R-series material keep their boundaries.

Generated runs are ignored as one directory. Add any newly selected evidence by
explicit file path with `git add -f -- <file>` and describe its reading route here;
adding another run does not require another `.gitignore` exception list.

### Measurement cache migration — 2026-09-16

The [migration inventory](runs/physiology_semantic_tokenizer/data_quality_audit/20260916_measurement_cache_migration_v4/migration.json)
records authorized cleanup, retained metadata, cache validation and before/after
allocated disk usage. That migration selected the v4 measurement cache;
model-specific normalization belongs to SSM/tokenizer consumers. Retired general
v1 and public smoke arrays are rebuildable; preserve their manifests, event and
geometry metadata in this migration package before deleting those directories.
The v3 SSM native cache and historical teacher sidecars remain retained inputs.
This is a data preparation migration, not a new model result or protected evaluation.

The subsequent [public-subject extension](runs/physiology_semantic_tokenizer/data_quality_audit/20260917_all_subject_cache_v4/manifest.json)
records retirement of the inherited subject embargo and the independently supervised
cache expansion. Its [validation and 30-second inventory](runs/physiology_semantic_tokenizer/data_quality_audit/20260917_all_subject_cache_v4/validation.json)
verify new signal/event/geometry coverage and unchanged existing arrays/events.
Compressed pre-extension indices and manifests preserve the earlier cache inventory;
the 2026-09-16 migration evidence remains unchanged.

### No-motion cache migration — 2026-09-28

The [migration record](runs/physiology_semantic_tokenizer/data_quality_audit/20260928_motion_cache_v5/migration.json)
retains the decision sources, exact supervised commands, worker/IO measurements,
source patch and test results. The [validation](runs/physiology_semantic_tokenizer/data_quality_audit/20260928_motion_cache_v5/validation.json)
compares the new cache with the previous record inventory, EEG arrays, units,
support, events, geometry and actual unified/sequence loading. Preserve both
cache identities and this evidence; this migration does not authorize deletion
of the old cache. Method choice and processing boundaries are owned by
[DATA_CONTRACT](../docs/DATA_CONTRACT.md#无运动校正的通用缓存--2026-09-28).

### Historical source/observation generation

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

### 2026-07-30 pruning record

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
