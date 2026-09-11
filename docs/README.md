# Documentation map

_Authority index. Registered execution and scientific verdicts are generated from the
machine-readable research-state registry. The method-rationale and architecture
documents are retained claim-boundary and implementation records. The planned
`PST-DISCOVERY-v1` sequence is owned by
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md); the 2026-08-22 [tracked design note](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json)
remains an abandoned candidate snapshot. Paper-facing evidence routes are listed
in [`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md)._

## Authority documents

| Question | Authority |
| --- | --- |
| What lifecycle is recorded for each track? | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) (generated) |
| Where is the planned clean-slate experiment sequence? | [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) (`PST-DISCOVERY-v1`; no measured run authorized) |
| What rationale and evidence support the scientific conclusion? | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) |
| Where should paper Methods, Results, Discussion, and Figures take material from? | [`PAPER_EVIDENCE_INDEX.md`](PAPER_EVIDENCE_INDEX.md) |
| What data, masks, joins, geometry, and splits are valid? | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) |
| What are the dataset-native facts and original sources? | [`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md) |
| What code/runtime exists today? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| What theory and architecture principles are frozen? | [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md#frozen-theory-and-architecture-contract-unimplemented) |
| Where is the abandoned pre-freeze candidate snapshot? | [tracked design note](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json) · [candidate snapshot figure](physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg) |
| What did the 2026-08-21 v1 QC actually measure? | [`analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md`](analysis/SSM_OBSERVATION_AND_COUPLING_QC_RESULTS_20260821.md) |
| What happened in E0–E2 and R0–R2? | [`physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) |
| What is the full R-series evidence? | [`20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md) |
| What is the comparison contract? | [`comparisons/PROTOCOL.md`](comparisons/PROTOCOL.md) |
| Which comparisons stopped, and which follow-ups were abandoned? | [`PROJECT_STATUS.md#对比实验`](PROJECT_STATUS.md#对比实验) (generated) |
| What is the short comparison-method summary? | [`comparisons/STATUS.md`](comparisons/STATUS.md) |
| What are the completed protected-comparison results? | [`comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md) |
| Which method sources and weights are prepared? | [`../comparative_methods/ASSET_STATUS.md`](../comparative_methods/ASSET_STATUS.md) |
| Which values can enter a final table? | [`comparisons/METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md) |
| What did the retained Token Atlas result show? | [`analysis/TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |
| Where are retained result artifacts? | [`../experiments/RESULTS_INDEX.md`](../experiments/RESULTS_INDEX.md) |

## Frozen and historical evidence

[`physiology_semantic_tokenizer/README.md`](physiology_semantic_tokenizer/README.md)
indexes the retained E0/E1/E2/R decision and evidence reports. Those generations,
the SSM/continuous-latent and LC-SPVQ development, and Atlas Core are stopped
evidence. Superseded design, code, config, test, and run generations are in one
local Git-ignored archive and are not active instructions.

The 2026-08-22 theory note is retained as a method-boundary source. Its candidate
branches are abandoned implementation snapshots, not runtime or scientific
results; its optional-grammar wording does not override the retained
endpoint/proper-score/null evidence kernel. The lifecycle table in
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) marks E0–E2/R0–R2, SSM reliability,
continuous latent, LC optimization/QC, and Atlas Core as **stopped**; it marks
D1B, future R/VQ, LC full development, observation-source candidates, and Atlas
Statistical/Full as **abandoned**. No sequence follows from those records.

Within the 2026-08-21 QC report, “v2” and “v3” label historical LC-SPVQ
mask-contract generations only; they do not name a current method generation.

Other evidence layers (historical unless their owning contract says otherwise):

- [`METHOD_RATIONALE.md`](METHOD_RATIONALE.md) is the retained 2026-08-22 v1/v2
  claim-boundary and architecture-rationale owner; it explains what evidence can
  support a claim but does not define an active experiment sequence;
- [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) is a
  dated E0–E2/R0–R2 decision snapshot. Any “current” wording inside it is local to
  that snapshot and does not replace the generated project status;
- [`architecture_changelog/`](architecture_changelog/INDEX.md): architecture
  and data-contract decisions;
- [`project_changelog/`](project_changelog/INDEX.md): repository and
  operational changes;
- `physiology_semantic_tokenizer/analysis/`: dated result reports;
- `reliable_survey/`, `references/`, and `notes/`: literature/background, not
  implementation authority;
- ignored manuscript/report/PDF trees: communication or reference assets. In
  particular, `docs/paper/科学通报/` is a local communication asset; its figures and
  manuscript drafts do not replace the tracked runtime, exploration note, or dated
  evidence reports.

Accordingly, PID or partial-information-decomposition language in notes,
references, archives, or local manuscript drafts is background or historical
material. The retained claim boundary treats PID only as a replaceable later
pretraining exploration—not as a core contribution, method identity, or a frozen
component.

## Lightweight lifecycle

- Registered execution state and scientific verdicts are read from
  [`../research_state/registry.json`](../research_state/registry.json) and its
  generated [`PROJECT_STATUS.md`](PROJECT_STATUS.md). Hand-written documents
  explain methods, evidence, and interpretation.
- The lifecycle overlay and current planned sequence are owned by
  [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md); do not hand-maintain parallel stage
  state or next-action lists outside the registry-generated views.
- For a future registry update, edit its owning item and run `validate`, then
  `render`. File hashes are not part of the project-state contract.
- Protocol and data-access constraints stay in their owning contracts. A passing
  software test does not override a failed scientific gate.
- Preregistrations, manifests, and result reports are dated snapshots; if a
  narrative needs correction, point to a newer dated note rather than rewriting
  the history.
- Archived code, configs, tests, and results require an explicit path and do not
  participate in default discovery.
