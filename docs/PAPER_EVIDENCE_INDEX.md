# Paper evidence index

This is the reader-facing route from the project tree to manuscript material. The
retained theory/architecture claim boundary is owned by the method rationale and
data contract below; the 2026-08-22 observation–source design note is an
abandoned implementation snapshot. The registry and generated status pages are
useful lifecycle navigation, but they are not substitutes for the dated reports
and contracts listed here.

## Methods

Use these documents to describe what was designed, measured, and scientifically
specified:

- [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md): research question, claim levels,
  method rationale, and scientific boundaries.
- [Tak hemodynamic baseline](ARCHITECTURE.md#tak-hemodynamic-baseline): current
  equations, independent states, observation scope and executable settings.
  Attribute the blood-flow model to Tak/SPM; the project retains its EEG
  driver/Hb observation inference. Historical inlet-balance results keep their
  original Methods identity.
- [`DATA_CONTRACT.md`](DATA_CONTRACT.md) and
  [`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md): units, modality semantics,
  masks, joins, normalization, and split rules.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) and the retained runtime description in
  [`physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json`](physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json):
  the retained-tokenizer sections describe the stopped E2-compatible implementation and data-flow structure, for replay and
  historical description only. It must be kept separate from the abandoned
  design note below.
- [`physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json`](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json):
  abandoned pre-freeze candidate map for continuous teachers, observation/source branches,
  token hierarchies, and an earlier optional-grammar projection. It is not a
  Methods contract and does not override the retained endpoint/proper-score/null
  kernel.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): current planned clean-slate
  sequence plus the lifecycle overlay for the stopped historical flow; it does
  not authorize measured or protected execution.
- [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md),
  [`comparative_methods/README.md`](../comparative_methods/README.md), and
  [`comparative_methods/ASSET_STATUS.md`](../comparative_methods/ASSET_STATUS.md):
  comparison protocol, method identity, and source/weight preparation.
- [`../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md):
  Croce legacy solver/audit record (stopped), plus the unstarted Synthetic Phase 1
  and Real Phase 2 candidates (abandoned). This is a separate qualification lane;
  its legacy particle-filter status does not qualify a future teacher. The
  accepted E0 development baseline is recorded in
  [`E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md`](physiology_semantic_tokenizer/analysis/E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md).

## Results

Use the dated report or retained table that directly contains each result. Use
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) only to locate its recorded lifecycle;
historical “current” or “next” wording in a report is local to that snapshot.

- [Tak versus inlet-balance comparison](../experiments/RESULTS_INDEX.md#tak-hemodynamic-migration--2026-10-27):
  implementation checks, synthetic controls, cached measured comparisons,
  numerical failures and parameter sensitivity. Adoption as the research
  baseline does not relabel the recorded reconstruction or physiological claims.

- [`physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md):
  compact E0–E2/R0–R2 decision snapshot.
- [`physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md):
  full R-series methods, numerical results, and gate interpretation.
- [`analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md`](analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md):
  v1 2026-08-21 fit-selection QC, including historical LC-SPVQ controls,
  three-seed continuous SSM screen, leakage controls, and the fail-closed decision.
  It is exploratory development evidence: the screen used no VQ, protected cohorts
  remained closed, and its condition×time baseline is a secondary residual oracle,
  not a basis for a final VQ admission claim. The report's “v2” and “v3” labels
  refer only to LC-SPVQ mask-contract generations; they do not name a forward
  method generation.
- [`analysis/TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md):
  retained development-only T0 Atlas result and its support limits.
- [`comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md):
  tracked 42-cell comparison results and cell-level acceptance states.
- [`comparisons/METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md):
  rules for deciding which comparison cells can enter tables.
- [`comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md):
  exploratory degradation analysis; it does not provide a single causal result.
- [`../experiments/RESULTS_INDEX.md`](../experiments/RESULTS_INDEX.md):
  retained artifact locations and links to reports. Some payload paths there are
  local Git-ignored artifacts; the linked tracked reports remain the paper-facing
  record.

## Discussion

Build interpretation and limitations from:

- the evidence-supported and unsupported claim tables in
  [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md);
- the R-series interpretation and failure boundary in the
  [`R-series report`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md);
- the v1 observation/LC-SPVQ QC boundary in the
  [`2026-08-21 QC report`](analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md):
  the recorded EEG-target/encoder mismatch and teacher non-superiority remain
  unresolved; they leave several observation/source candidates untested. Those
  candidates are abandoned snapshots and do not select a successor architecture.
- the support/stability and non-coupling boundaries in the
  [`Token Atlas report`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md);
- the track labels, overlap conditions, rejected values, and unsupported cells in
  the [`comparison result report`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)
  and [`metric acceptance rules`](comparisons/METRIC_ACCEPTANCE.md).

Keep engineering completion, descriptive findings, and failed or undetermined
scientific gates separate in the manuscript.

## Figures

Use the tracked runtime figure to describe the stopped implementation. Candidate
figures are abandoned pre-freeze snapshots and are not current Methods evidence.

- [`physiology_semantic_runtime_overview.svg`](physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg)
  with its editable Draw.io source in the
  [`architecture` directory](physiology_semantic_tokenizer/architecture/) is the
  stopped E2-compatible runtime presentation view (historical/replay-only).
- [`physiology_semantic_architecture.svg`](physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)
  is a detailed exploratory observation–source candidate view, not Methods or a
  frozen architecture target.
- [`observation_source_exploration_v2.svg`](physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg),
  its tracked design note
  [`observation_source_exploration_v2.json`](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json)
  preserve replaceable implementation candidates from before the retained
  boundary.
  Their optional-grammar wording is superseded by `METHOD_RATIONALE.md`. The
  schematic contains **no measured values** and is not a Methods target.
- [`project_workflow_progress.svg`](project_workflow_progress.svg) is the optional
  historical four-lane snapshot dated 2026-08-16. Its completed nodes are now
  **stopped**, and its planned/blocked nodes are **abandoned**; labels such as
  `Current`, `NEXT`, or `blocked` are snapshot-local, not current instructions.

## Not paper evidence

- Local Git-ignored older manuscript trees may contain obsolete project names,
  live-run text, or presentation figures. They are not Methods or Results sources;
  use the tracked runtime, method boundary, and dated reports above.
- Local Git-ignored `docs/paper_pdf/` and `docs/report/` trees are
  communication/reference assets; use the tracked reports above for retained claims.
- [`../data/DATASETS_DESCRIPTION.md`](../data/DATASETS_DESCRIPTION.md) is only a
  compatibility pointer to the authoritative dataset description.
