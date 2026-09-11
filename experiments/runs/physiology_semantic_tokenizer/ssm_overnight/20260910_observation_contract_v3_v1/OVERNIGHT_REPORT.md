# Observation-contract diagnostic v3

Feature-layer missingness; original 72 training identities only. O2 is Gaussian MAP. Teacher qualification: none. MAP intervals: NOT_ESTIMATED.

Per-cell result.json and trajectory files own the evidence. Tables below retain the complete planned denominators.

| Stage | Completed cells / planned | Actual / planned fits | Status counts |
|---|---:|---:|---|
| S1 | 3601 / 3601 | 3600 / 3600 | {"completed": 3601} |
| S2 | 214 / 292 | 2268 / 3024 | {"completed": 214, "failed_contract": 9, "data_unavailable": 68, "not_started_budget": 1} |
| S3_synthetic | 0 / 1717 | 0 / 19008 | {"not_started_budget": 1717} |
| S3_measured | 0 / 265 | 0 / 3168 | {"not_started_budget": 265} |
| S4_synthetic | 0 / 1221 | 0 / 14400 | {"not_started_budget": 1221} |
| S4_measured | 0 / 385 | 0 / 4608 | {"not_started_budget": 385} |

## Gates

```json
{
  "v3_S1_gate": {
    "status": "completed",
    "passed": true,
    "completed": 24,
    "expected": 24,
    "mean_nrmse": {
      "r": 0.4842879153068124,
      "clean_EEG": 0.4842879153068124,
      "clean_HbO": 0.1486210528861541,
      "clean_HbR": 0.16291038947375336
    },
    "mean_r_correlation": 0.8709836882269576,
    "engineering_passed": true,
    "interpretation": "Gaussian mean precheck for bounded feature-missing diagnostics; no Student-t or teacher qualification",
    "rows": [],
    "task_id": "v3_S1_gate",
    "family": "S1",
    "kind": "v3_gate",
    "elapsed_seconds": 0.33379300695378333,
    "peak_rss_bytes": 684650496,
    "completed_at": "2026-09-10T15:44:20.115455+00:00"
  },
  "v3_gain_screen": {
    "status": "not_started_budget",
    "rows": []
  },
  "v3_process_screen": {
    "status": "not_started_budget",
    "rows": []
  }
}
```

## Measured comparisons

Only complete 72-trial and (where applicable) 12-selection-fold panels define full risk. Common-subset values are descriptive.

| Rule | Reference | Complete common / 72 | Candidate B | Baseline B | Relative improvement |
|---|---|---:|---:|---:|---:|
| S2/O0 | S2/O0 | 21 | 3.64785 | 3.64785 | 0 |
| S2/O1 | S2/O0 | 2 | 0.831134 | 4.8513 | 0.828678 |
| S2/O2 | S2/O0 | 0 | undefined | undefined | undefined |
| S3_measured/O2 | S2/O2 | 0 | undefined | undefined | undefined |
| S4_measured/O0 | S2/O0 | 0 | undefined | undefined | undefined |
| S4_measured/O2 | S2/O2 | 0 | undefined | undefined | undefined |

Detailed evidence: [solver/truth comparison](S1/solver_comparison.csv), [per-mode denominators](mode_status.csv), [failures](failure_attribution.csv), [full residuals](full_fit_residuals.csv), [process replay](process_replay.csv).

[Independent synthetic baseline / selected / oracle panels](synthetic_adaptation.csv).

Original native-mask results have a different input contract and cannot be used as a direct effect-size baseline. Four-panel synthetic intervals are approximate; incomplete or inconclusive screens do not establish harmless adaptation.

![Combined-processing driver recovery](S1/driver_recovery.png)
