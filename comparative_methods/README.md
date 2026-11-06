# Comparison methods

> **Retained 2026 campaigns — STOPPED.** The recorded comparison campaigns and
> method deliveries are complete and retained as read-only evidence. Any
> unexecuted legacy queue, launch path, or follow-up is **ABANDONED**.
> The separately versioned [downstream rebenchmark design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1)
> does not restart those campaigns or modify their frozen evidence.

This tree contains comparison-method integrations, compact protocol/evidence files,
and local generated assets. It also retains the frozen 2026 public matrices and joint
protected campaign at their original paths so their hashes remain auditable. Those
historical control files are not the workflow for ordinary new development.

Current execution and scientific verdicts come only from the generated
[`PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md#对比实验). Do not infer current work,
authorization, or next actions from a method README, launch YAML, candidate, run
directory, or signed campaign artifact.

## Start here

| Reading task | Owning entry |
| --- | --- |
| Design the next downstream benchmark | [`DOWNSTREAM-REBENCH-v2`](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v2): complete metrics, subject folds and Visual/REFED repair; [v1 design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1) retains its execution identity |
| Run the versioned downstream benchmark | [Execution entry](#downstream-rebenchmark-v1), [reviewed contract](downstream_rebench_v1.yaml) |
| Find implementation and local tests | [Directory map](#directory-map), then the selected method's README and source manifest |
| Read retained formal performance | [Dated joint result table](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md); [retained result locations](../experiments/RESULTS_INDEX.md#comparison-methods) |
| Explain historical low scores | [`performance_analysis/`](performance_analysis/README.md): exploratory analyses, separate from formal performance |
| Check registered lifecycle | [Generated comparison status](../docs/PROJECT_STATUS.md#对比实验) |

Read only what the task needs:

1. [`docs/comparisons/PROTOCOL.md`](../docs/comparisons/PROTOCOL.md) — shared data,
   split, task, metric, and protected-evaluation contract for the retained campaign;
2. [`docs/comparisons/METRIC_ACCEPTANCE.md`](../docs/comparisons/METRIC_ACCEPTANCE.md)
   — rules for admitting a numeric result;
3. for historical adapter semantics, [`ADAPTER_ALIGNMENT_GATES_V2.md`](ADAPTER_ALIGNMENT_GATES_V2.md),
   its [`machine contract`](adapter_alignment_gate_contract_v2.yaml), and
   [`metric targets`](comparison_metric_targets_v1.yaml);
4. the target method's `README.md`, `sources/method_manifest.yaml`, retained config,
   implementation, and tests;
5. for the completed 2026 results only, the dated
   [`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md).

Local ignored release candidates, authorization records, ORR files, predictions,
and job statuses are optional historical evidence. They are never prerequisites for
understanding the repository or starting public-surface code work.

## Downstream rebenchmark v1

The [frozen classification revision](downstream_frozen_classification_v1.yaml)
uses the existing `performance_analysis/audit_downstream_inputs_v1.py` with
`--stage plan|fit|aggregate`. It reuses original fixed-validation evidence for
CBraMod, STA-Net, EFRM and BrainFusion, adds MtReNIRS readouts, and derives the
explicit REFED arousal binary task without changing retained regression labels.
The [current design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v2) owns its
scope; these are development results, with no encoder fine-tuning or original
test access. Tests: `test_downstream_input_benchmark.py`,
`test_downstream_input_audit.py`, `test_downstream_rebench_eeg_models.py`, and
BrainFusion's method-local `tests/test_rebench_v1.py`.

The [STA-Net optimization contract](downstream_sta_optimization_v1.yaml) uses
the same analysis entry with `--stage sta-plan|sta-fit|sta-screen-aggregate|sta-aggregate`.
Its opt-in `sta_net_pytorch/source_aligned_v2.py` corrects the pinned source
semantics while retaining `model.py` and historical checkpoints. It uses the
original public development population, explicit source-duration/short-window
adaptations and supervised classification from random initialization. Targeted
checks are STA-Net's `tests/test_source_aligned_v2.py`, `tests/test_rebench_v1.py`,
and the common `tests/test_downstream_input_audit.py`.

The [REFED source-aligned revision](downstream_sta_refed_source_v3.yaml) uses
`source_aligned_v3.py` to distinguish TF2.10 fused EEG and unfused Hb BatchNorm
moving variances. Its method-local `tests/test_source_aligned_v3.py` covers those
conventions and inference-preserving parameter conversion. The separate
[normalization diagnostic](downstream_sta_normalization_v1.yaml) and
[REFED normalization diagnostic](downstream_sta_refed_normalization_v1.yaml)
use `--stage sta-normalization-plan|sta-normalize|sta-normalization-aggregate`
to reestimate inference statistics from training inputs with frozen parameters.
They retain the parent candidate and seed choices and do not perform gradient updates.

The [design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1) and
[executable contract](downstream_rebench_v1.yaml) define the new public v5 comparison.
It has its own `rebench_v1.py` adapters and does not resume retained v2 campaigns.

| Owner | Contract / targeted tests |
| --- | --- |
| [`src/data/downstream_benchmark.py`](../src/data/downstream_benchmark.py): inventory, subject splits, native masks and label-blind source windows | [`test_downstream_benchmark.py`](../tests/test_downstream_benchmark.py) |
| [`downstream_rebench_worker_v1.py`](downstream_rebench_worker_v1.py): train-only transforms, inner selection, fresh final fit and recovery | [`test_downstream_rebench_worker_v1.py`](../tests/test_downstream_rebench_worker_v1.py); method-local `tests/test_rebench_v1.py` |
| [`run_downstream_rebench_v1.py`](run_downstream_rebench_v1.py): build frozen jobs, bound resources, resume and dispatch dependencies | [`test_downstream_rebench_scheduler.py`](../tests/test_downstream_rebench_scheduler.py) |
| [`aggregate_downstream_rebench_v1.py`](aggregate_downstream_rebench_v1.py): planned denominators, subject/seed aggregation and admission | [`test_aggregate_downstream_rebench_v1.py`](../tests/test_aggregate_downstream_rebench_v1.py); metrics in [`src/metrics/downstream_benchmark.py`](../src/metrics/downstream_benchmark.py) |
| [`rebench_baselines_v1.py`](rebench_baselines_v1.py), [`rebench_eeg_common_v1.py`](rebench_eeg_common_v1.py): signal baselines and shared EEG adaptation | [`test_downstream_rebench_baselines.py`](../tests/test_downstream_rebench_baselines.py), [`test_downstream_rebench_eeg_models.py`](../tests/test_downstream_rebench_eeg_models.py) |

The separate [MA adaptation audit](ma_adaptation_audit_v1.yaml) uses event-relative
`[-5,25)` seconds, the same 16 EEG channels and 72 Hb channels across the applicable
methods, and one frozen development subject split. Its
[analysis entry](performance_analysis/audit_ma_adaptation_v1.py) reuses this worker
and scheduler; [targeted tests](../tests/test_ma_adaptation_audit.py) cover the
versioned window, sample migration, masks and adapter shapes. The explicit
`ma_30s_diagnostic_v1` profile does not change the retained v1 defaults or imply
30-second support for every task. This diagnostic does not establish a final
paper ranking or extend historical protected-data access.

The [Visual investigation](visual_investigation_v1.yaml) uses
[`investigate_visual_v1.py`](performance_analysis/investigate_visual_v1.py) to audit
native label/trigger lineage and run fixed development controls for class balance,
label dimensions, time support and CBraMod readout. Its
[tests](../tests/test_visual_investigation.py) cover label semantics and split/window
boundaries. It writes only to its analysis run; inclusion in a presentation report
requires the user's later decision.

The [frozen-input diagnostic](downstream_input_audit_v1.yaml) uses
[`audit_downstream_inputs_v1.py`](performance_analysis/audit_downstream_inputs_v1.py)
to audit retained public production predictions and test the corrected downstream
cache on its existing train/validation partitions. The new
`FrozenInputBenchmarkDataset` in the data owner joins Visual probe views into one
event, orders real Hb pairs, and supplies a common EEG panel to every method.
Native-channel capacity, legacy Visual electrode selection, time support,
readout, scaling, short supervision and stacking diagnostics remain separate
arms. REFED retains its actual 30-point masked target sequence. These are
development-selected results, not final test values or automatic replacements
for the running frozen benchmark. The [design boundary](../docs/EXPERIMENT_PLAN.md#frozen-downstream-input-audit)
and `test_downstream_input_benchmark.py` / `test_downstream_input_audit.py` cover
the versioned consumer and diagnostic; existing method suites cover adapters.
Its [bounded follow-up](downstream_input_followup_v1.yaml) checks the discovered
BrainFusion HRF time-grid mismatch with paired refits, and the actual 30-step
STA-Net REFED head. The explicit `physical_common_grid_v1` profile samples the
kernel at the common Hb period; `retained_sample_index_v1` preserves the old
control. Old snapshots and protected method packages are unchanged.
The [STA-Net completion contract](downstream_sta_completion_v1.yaml) adds WG,
n-back and DSR using the same frozen diagnostic source, retained training and
validation inputs, and short-training selection rule. Its explicit job manifest
contains only three task smokes and their corresponding fits; input references
point to the original prepared owner without copying or changing those arrays.

The same entry executes the [iterative repair design](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v2)
through versioned contracts indexed in that design. `--stage probe-plan` resolves each
version's retained development population and jobs; pass the original
`--project-root` and frozen `--source-root` explicitly. The existing scheduler
runs those jobs. `probe-verify` independently checks saved fold predictions,
ACC/F1, dynamic CCC, native errors, subject SD and null baselines. Independent
record controls and stimulus templates retain separate interpretations; they
cannot qualify cross-subject signal performance. The targeted suite remains
[`test_downstream_input_audit.py`](../tests/test_downstream_input_audit.py).
Native Hb/EEG replay, encoder train/evaluation mode, stimulus-student and
individual-calibration diagnostics keep their own information conditions.
The [EFRM temporal transfer probe](downstream_repair_probe_v11.yaml) fits input
scales separately within each development fold. The
[encoder learning-rate comparison](downstream_repair_probe_v15.yaml),
[BIOT physical montage subset comparison](downstream_repair_probe_v16.yaml), and
[Visual kernel regularization grid](downstream_repair_probe_v17.yaml) use the same
entry and preserve their fixed input and development-fold identities. The
[REFED record-objective probe](downstream_repair_probe_v18.yaml) compares a
within-record CCC term with the same record-batched MSE control, retaining
constant target coordinates in the error objective. The
[paired encoder continuation](downstream_repair_probe_v19.yaml) starts from the
same record-objective head and contrasts trainable with frozen encoders after a
longest-record gradient/memory smoke. The
[Visual class-weight comparison](downstream_repair_probe_v20.yaml) removes
inverse training-class weights while preserving equal participant mass, features,
kernels, folds and all four-class endpoints. The
[REFED modality comparison](downstream_repair_probe_v21.yaml) retains the record
objective and training budget while testing EEG and Hb separately against the
retained joint-input control. The
[fixed-objective solver replay](downstream_convergence_probe_v1.yaml) retains
the earlier split and cannot be compared directly with these folds.
`--stage probe-report --evidence-runs <explicit completed run paths>` exports
the retained summaries as HTML/CSV; original summaries remain the numeric owners.

The [pure fNIRS probe](downstream_fnirs_probe_v1.yaml) reuses these probe stages
for six classification tasks on the same retained development population.
[`MtReNIRS/rebench_v1.py`](MtReNIRS/rebench_v1.py) owns the frozen official
encoder adaptation; the existing signal-baseline owner supplies Hb statistics
and block covariance/tangent features. It reads no EEG signals. Its scope,
short-window adaptation and accepted pretraining overlap are recorded in the
[design](../docs/EXPERIMENT_PLAN.md#纯-fnirs-方法开发测试2026-11-05).
[`test_downstream_fnirs_probe.py`](../tests/test_downstream_fnirs_probe.py)
covers masks, geometry, training-only fitting and independent binary/multiclass
aggregation without local evidence dependencies.
The block-Riemannian arm is a bounded tangent-space/logistic variant of the
[fNIRS Riemannian comparison](https://doi.org/10.1117/1.NPh.12.4.045002), using
pyRiemann training-fold references; it does not reproduce that paper's complete
kernel search or replace its original evaluation protocol.

Build a new run only after its inventory, splits, multimodal input audit and resource
smoke are retained. Use the same controller's `--build-config`, `--run-root`,
`--source-root` and `--smoke-summary` options. The generated `jobs.json` owns the
initial resolved DAG; model outputs remain under each method's existing `runs/` root.
Run `--manifest <jobs.json>` through the recorded persistent service and use
`--manifest <jobs.json> --status` to reconnect. Preserve the frozen source and job
identity when resuming. Per-job failures block dependent training; the final aggregate
still records all planned units and failures. Runtime status is in `scheduler_status.json`;
registered lifecycle is in the project registry, not this README.

For a requested reduction containing only never-started jobs, detach the
controller while preserving its workers, then use this same entrypoint with
`--manifest <current.json> --remove-pending-method <method> --revised-manifest
<fresh.json>`. The operation holds the controller lock, preserves the original
manifest and a dated state snapshot, and updates the final aggregate's input and
dependencies. Restart the service with the revised manifest; the normal recovery
path adopts existing workers without launching duplicates. The current manifest
path is recorded in `scheduler_status.json`; a retained initial config does not
override an explicit scope reduction.

## Directory map

| Path | Role | Default treatment |
| --- | --- | --- |
| `BIOT/`, `CBraMod/`, `REVE/` | EEG-only method owners | legacy method-local code/evidence; no current queue |
| `NormWear/`, `EFRM-PyTorch/`, `BrainFusion-NVC-CSP-Stacking/` | multimodal/adapted method owners | legacy method-local code/evidence; no current queue |
| `STA-Net-PyTorch/` | independent retained implementation and frozen result | change only for an explicit STA-Net task |
| `performance_analysis/` | cross-method descriptive analysis | analysis only; never an execution authority |
| `single_modal_eeg/` | early shared BIOT/CBraMod/REVE runner | compatibility only; no new features |
| `STA-Net/`, `EFRM-A-*/`, `*/upstream/` | local upstream source mirrors | ignored, read only when source fidelity requires it |
| `*/checkpoints/`, `*/runs/`, root `runs/` | model and run payloads | generated/ignored; do not scan by default |
| `*/evidence/`, root `evidence/` | compact summaries plus local protected evidence | read a named final summary, not the whole tree |
| root campaign `*.py` files | frozen 2026 joint control plane | historical/read-only unless a new versioned campaign is requested |

The physical paths above are retained because configs, reports, tests, and source
hashes refer to them. Future cleanup must use a versioned migration; adding wrappers
or moving the frozen v2 files merely for appearance would increase complexity.

## Method-local shape

New comparison work belongs to one method owner:

```text
comparative_methods/<method>/
├── README.md             # identity, supported surface, local entrypoints
├── sources/              # upstream revision and source-fidelity record
├── configs/              # data/protocol/runtime configuration
├── adapters/ or package/ # method-specific implementation
├── tests/                # adapter and method-specific tests
├── evidence/             # compact reviewed summaries only
├── checkpoints/          # local binaries; ignored
├── runs/                 # local generated outputs; ignored
└── upstream/             # local pinned checkout; ignored
```

Do not add a second status file, generic results root, method-local authorization
service, or another top-level runner. Reusable public path/task/metric invariants
belong in one small shared helper; model and training differences remain local.
Methods must not import implementation details from another method.

## Public and protected work

Public-surface development is ordinary work and creates no authorization artifact.
Protected evaluation is a separate transition owned by one protocol/controller; a
prior GO never carries forward. The existing v2 flags remain historical schema and
must not be copied into new work.

Scheduling, runtime attestation, and schema rules belong to the owning
protocol/controller, not to method-local authorization copies.

## Evidence and tests

For retained adapter coverage, prefer `evidence/alignment_v2/summary_final.json`;
for public training completion, use the reviewed matrix completion summary. Neither
is a final test score. Read the dated joint result table for final performance,
and follow its explicit evidence links when a deeper audit is needed.
Prefer these final summaries over intermediate `summary.json`, pilot, or partial files.
The intermediate files remain only where frozen configs still hash or consume them.

The repository's default `pytest.ini` does not collect method-local test directories.
Run the owner suite explicitly after changing a method, for example:

```bash
.venv/bin/python -m pytest -q comparative_methods/BIOT/tests
```

Run different method suites in separate pytest processes because some historical
packages use overlapping top-level module names. The joint protected evidence suite
depends on local sealed artifacts and is not an orientation check; use a temporary,
non-authorizing fixture where possible and never alter a workspace signing record
merely to make it pass.

## Historical material and local assets

[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) and
[`ROUND_ARTIFACTS_20260812.md`](ROUND_ARTIFACTS_20260812.md) are dated historical
snapshots, not current queues or permission records. Source and weight provenance is
indexed by [`ASSET_STATUS.md`](ASSET_STATUS.md).

Git tracks controlled source, contracts, tests, manifests, compact summaries, and
the dated human-readable result report. Upstream checkouts, weights, feature caches,
large runs, protected predictions, and named signing records stay local. Do not add
new `.gitignore` allowlist entries for source code; generated files must use the
existing owner-local ignored paths.
