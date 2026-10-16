<a id="retained-and-future-generated-results"></a>

# Generated experiment results

Main-project runs use the existing suite/run hierarchy:

```text
experiments/runs/
└── physiology_semantic_tokenizer/
    └── <suite>/
        └── <dated_versioned_run>/
```

Existing suite roots include `shared_driver_reconstruction/`,
`semantic_response_discovery/` and `data_quality_audit/`. A suite groups related
diagnostics; each dated run preserves its resolved contract and provenance.
Separately versioned analyses, continuations and report exports can sit below a
run; their owning manifests identify inherited evidence and original sources.

Read results in this order:

1. Use the [result index](../RESULTS_INDEX.md#find-results-by-question) to find the
   scientific question, suite and exact evidence path. The linked protocol and
   executable contract define its scope.
2. Inspect the individual run manifest, resolved configuration and completion or
   failure records before interpreting an aggregate summary. Read the owning
   summary and endpoint tables together, including planned denominators,
   unsuccessful fits and paired-comparison rules.
3. Use the linked reports for explanation and figures, tracing numerical claims
   back to those run records. Detailed arrays and replay inputs may remain local;
   the index distinguishes them from the tracked result-reading package.

Retain negative results, failed or interrupted attempts, lightweight conclusions
and provenance at their original versioned paths. A continuation must preserve
the identities of reused evidence. Directory presence does not establish
scientific promotion or authorize data access; distinguish smoke, development,
formal and protected evidence using the owning contract.

Current execution and verdicts belong to the
[research-state registry](../../research_state/registry.json) and its generated
[project status](../../docs/PROJECT_STATUS.md). Superseded generations have an
explicit local archive path; do not recursively mix `experiments/archive/` with
this root. [Comparison-method results](../../comparative_methods/) keep their
own package directories outside this tree.
