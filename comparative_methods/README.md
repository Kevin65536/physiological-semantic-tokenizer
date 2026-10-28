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
| Design the next downstream benchmark | [`DOWNSTREAM-REBENCH-v1`](../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1): data version, input support, fitting budget, tracks and evaluation design |
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
