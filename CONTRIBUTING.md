# Development and experiment guide

_Development conventions; experiment design and registered state have separate owners._

This is the operational guide for code, tests, experiment launches, results,
and documentation. Registered execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md); the clean-slate experiment
entry and lifecycle overlay live in [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

## Repository boundaries

```text
src/                         reusable library code
tests/                       unit, integration, contract, and evidence tests
experiments/configs/         reviewed executable experiment contracts
experiments/scripts/         all new launch, evaluation, analysis, and utility entrypoints
experiments/*.py             existing commands retained at their recorded paths
experiments/runs/            active generated results (ignored by Git)
experiments/archive/         local superseded generations (ignored by Git)
comparative_methods/         isolated comparison-method implementations
croce_validation/            physical-model validation and derived caches
docs/                        active contracts, status, evidence, and history
data/                        original datasets and derived caches (ignored)
```

Reusable classes and functions belong in the closest existing `src/` package.
Add new main-project executable workflows under `experiments/scripts/`; isolated
comparison and Croce work keeps its existing owner. Existing `experiments/*.py`
commands remain at their recorded paths and receive local fixes there. Do not add
a second launcher or a forwarding wrapper to make an old command match the new
placement rule. The [experiment map](experiments/README.md#entrypoint-config-and-test-map)
links each diagnostic to its config and targeted tests.

When multiple callers need the same computation, import it from its library
owner rather than another runner's private functions. Shared SSM observation
transforms, noise estimates and linear baselines live in
`src/inference/observation_baselines.py`. Pass data/replay roots explicitly at
read boundaries; importing a runner must not overwrite another module's roots.
Historical R-series dependencies bound by a seal remain in place; changing
those requires a versioned migration.
Any future versioned physiology-semantic runs write below
`experiments/runs/physiology_semantic_tokenizer/<suite>/<run>/`; comparison
methods write only below their owning package. Archive discovery is always
explicit: active tools do not recursively search or import
`experiments/archive/`.

The R1-P qualification surface is a dated, stopped evidence package. Keep its model,
scripts, registries, configuration, and tests together when revisiting that
historical result; ordinary cleanup should not silently mix it with a new
experiment generation.

## Environment

Use the repository environment:

```bash
source .venv/bin/activate
python -m pytest --collect-only -q
```

`requirements.txt` is a CUDA-capable environment snapshot, not a minimal
library specification. Do not assume system Python has the required packages.

## Data contract

Before using a dataset:

1. read [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md);
2. read the original dataset documentation named there;
3. use the central registry and unified loader rather than a new ad hoc parser;
4. preserve dataset-native units, task semantics, masks, channel identities,
   timing anchors, and source provenance;
5. fit normalization or target transforms on the permitted training partition
   only;
6. keep the train/validation/test split and any protected-sample boundary
   declared by the owning protocol.

Raw datasets are reference inputs. Derived caches should carry a schema/version,
source identity, transformation record, and a small manifest. A cached artifact
mask is audit metadata, not an automatic signal-validity mask.

## Configuration and launch

Reviewed YAML/JSON contracts live below `experiments/configs/`. A new
experiment needs:

- an unambiguous experiment ID and output namespace;
- resolved tensor, split, target, mask, and seed assertions;
- primary/secondary/diagnostic endpoint labels;
- explicit null, baseline, stopping rule, and protected-data boundary;
- parser and shape tests;
- dry-run or synthetic execution before measured data access.

The observation–source v2 material is an abandoned design note, not an
executable configuration or fixed architecture. Do not repurpose an E0–E2 or
R-series YAML for it. A future candidate would need its own versioned
software, tensor-shape, split, and null checks on synthetic data first. A
protected measured run additionally requires its owning protocol and separate
explicit user authorization.

SSM diagnostic implementations and commands are indexed in `experiments/README.md`.
The next tokenizer generation has no training launcher. Archived E0–E2 launchers
are stopped/abandoned evidence; a future tokenizer launcher needs its own
versioned implementation/config contract and synthetic checks.

## Evidence ladder

Verify work in this order:

1. unit tests;
2. integration and contract tests;
3. dry run;
4. smoke run;
5. short formal run, if the protocol defines one;
6. full public/development run;
7. one-time protected evaluation, only when the owning protocol gate and the
   explicitly user-approved scope are both satisfied.

A run is evidence only when its resolved configuration, completion status,
summary, and declared endpoint are present. A suite summary cannot
override an individual run record. Failed, negative, aborted, and
scientifically undetermined outcomes remain part of the record.

## Result retention

Keep the smallest package that preserves the scientific conclusion and normal
comparison use:

- resolved configuration, split/registry identities, summaries,
  tables, figures, and decision records;
- a checkpoint only when it is still needed for an active run, a recurring
  analysis, a consumer interface, or an irreplaceable reference;
- raw predictions or arrays only when the reported result cannot be audited or
  regenerated from retained material at acceptable cost.

Routine smoke checkpoints, older tuning checkpoints, duplicated token
exports, and rebuildable caches do not belong in the long-term result surface.
The retained evidence map is
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md).

Never clean a directory used by a live process. Query the unified project state and
inspect the owning process/controller immediately before cleanup; dated prose is not
a live-process detector.

## Tests

The active test taxonomy is documented in [`tests/README.md`](tests/README.md).
Run targeted tests after a local change, then:

```bash
python -m pytest --collect-only -q
git diff --check
```

Some formal-artifact tests intentionally depend on local data or sealed
evidence. Do not weaken or mark a sealed test merely to make a clean checkout
green; introduce a separate small fixture or revise the seal explicitly.

## Documentation

[`docs/README.md`](docs/README.md) is the only documentation authority map.
Registered execution and scientific verdicts live in
[`research_state/registry.json`](research_state/registry.json). Do not copy
historical state, job counts, or next actions into hand-written README files.

When a future registered experiment changes state, edit its corresponding item in
`research_state/registry.json`, then run:

```bash
.venv/bin/python experiments/scripts/project_state.py validate
.venv/bin/python experiments/scripts/project_state.py render
```

Dated reports preserve what was known at the time. Current state uses stable paths,
versions, and dates; file hashes are not a project requirement.

Generated reports, manuscript exports, and literature PDFs are communication
or reference assets, not active implementation authority.
