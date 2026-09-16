# Code and configuration archive isolation

_Project operations record; not a model architecture change._

_Date: 2026-07-02 · Phase: Phase 3 preparation · Git: `0ad233a..HEAD` · Status: Merged_

_Links: [source authority map](../../src/README.md) · [repository layout](../../README.md#repository-layout)_

---

## 🎯 Motivation

The documentation and result archives were isolated on 2026-07-01, but the executable roots still presented the superseded source/observation model, coupling losses, suite launchers, configs, and tests as default implementation context. A new contributor or code-search tool could therefore select an old model type or run an old test suite while following the new design documents.

## 🔀 Architecture delta

### Before

```mermaid
flowchart LR
    accTitle: Mixed executable context before code archival
    accDescr: Active package imports registered the old source observation tokenizer, while old launchers, configs, and tests shared default roots with the pending redesign.

    tokenizer_import["📦 Import src.tokenizers"] --> legacy_model["🔁 Old source/observation model"]
    script_root["🛠️ Active script root"] --> legacy_suites["🧪 Old coupling suites"]
    config_root["📋 Active config root"] --> legacy_configs["🗂️ Old config families"]
    pytest["✅ Default pytest"] --> legacy_tests["🔁 Old-contract tests"]

    classDef mixed fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    class tokenizer_import,legacy_model,script_root,legacy_suites,config_root,legacy_configs,pytest,legacy_tests mixed
```

### After

```mermaid
flowchart LR
    accTitle: Explicit active and compatibility code boundaries
    accDescr: Default imports, launchers, configs, and tests expose only reusable or target-facing surfaces; historical execution opts into dated compatibility namespaces.

    active["🔧 Active implementation"] --> reusable["🧱 Reusable primitives"]
    archived_scripts["🗂️ Dated scripts"] --> compatibility["📦 Dated compatibility package"]
    archived_configs["🗂️ Dated configs"] --> archived_scripts
    archived_tests["🗂️ Dated tests"] --> compatibility
    active -.->|no dependency| compatibility

    classDef active fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef reusable fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef archive fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#1f2937

    class active active
    class reusable reusable
    class archived_scripts,compatibility,archived_configs,archived_tests archive
```

## 🧱 Component changes

| Component | Change | Result |
| --- | --- | --- |
| `src/compatibility/pre_physiology_semantic_20260701/` | Added/moved | Contains old tokenizer, coupling losses/fusion, visualizations, classifiers, and ELP prototype |
| `src/tokenizers/__init__.py` | Changed | No default registration of old source/observation aliases |
| `experiments/scripts/archive/pre_physiology_semantic_20260701/` | Added/moved | Contains old training, suite, audit, export, probe, and downstream entrypoints |
| `experiments/configs/archive/pre_physiology_semantic_20260701/` | Added/moved | Contains all old root config families and their base file |
| `tests/archive/pre_physiology_semantic_20260701/` | Added/moved | Contains old-contract regression tests |
| `pytest.ini` | Added | Excludes archived tests from default collection |
| `experiments/scripts/launch_training_nohup.sh` | Replaced | Rejects legacy tasks and keeps the target task reserved until gates pass |

## 🛡️ Dependency and execution rules

- Active Python modules must not import `src.compatibility`.
- Historical checkpoint loading calls `register_legacy_tokenizers()` explicitly.
- New configs resolve only from `experiments/configs/physiology_semantic_tokenizer/`.
- The active launcher cannot dispatch pre-redesign tasks.
- Archived tests run only by exact path with archive collection explicitly enabled.

## ✅ Validation

- active default registry excludes both old tokenizer aliases;
- explicit compatibility registration restores both aliases;
- active launcher rejects `source-observation-tokenizer` with exit status 2;
- default collection finds 46 active tests and does not recurse into `tests/archive/`;
- 17 representative compatibility tests pass by exact archived path;
- archived source/observation base config still resolves through the shared loader.

## ↩️ Rollback considerations

Rollback requires restoring model registration, script/config roots, default test collection, and documentation links together. Restoring only a launcher or config directory would recreate a mixed authority state and is not a valid partial rollback.

_Last updated: 2026-07-02_
