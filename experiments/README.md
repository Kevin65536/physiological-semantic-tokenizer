# Experiment workspace

Commands and configuration for SSM diagnostics, reporting, and retained replay.
This page owns the entrypoint/config/test map. Execution and scientific verdicts
come from the [project registry view](../docs/PROJECT_STATUS.md).

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

Configs below are relative to `configs/physiology_semantic_tokenizer/`; tests
are relative to `../tests/`. This table maps software surfaces, not execution
state or permission to launch. Detailed usage follows in the linked sections.

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
| [Overnight report](#overnight-observation-diagnostics) | [`scripts/render_ssm_overnight_report.py`](scripts/render_ssm_overnight_report.py) | explicit completed run and fresh report directory | [`test_ssm_overnight_diagnostics.py`](../tests/test_ssm_overnight_diagnostics.py) |

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

## Active SSM entries

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

Launch and evidence conventions are in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
The owning plan is [`../docs/EXPERIMENT_PLAN.md`](../docs/EXPERIMENT_PLAN.md).
