# Test suite map

`pytest.ini` collects the active `tests/` surface only. Superseded architecture
tests are in the local ignored archive and are not importable or default-collected.
Method-local comparison suites must be run explicitly.

The [entrypoint/config/test map](../experiments/README.md#entrypoint-config-and-test-map)
owns the correspondence between executable diagnostics and their tests. Use it
to choose a targeted suite; the table below explains the library boundaries.
Existing sealed test paths remain fixed. New tests use descriptive subsystem
names such as `test_step5_*` or `test_ssm_*` in this directory.

The `sealed_evidence` marker excludes checks that require ignored campaign
artifacts. Run those only with the exact evidence restored:

```bash
.venv/bin/python -m pytest -q -m sealed_evidence tests/test_protected_campaign_v1.py
```

| Area | Representative files | Contract |
| --- | --- | --- |
| Data/preprocessing | `test_unified_physiology.py`, `test_event_alignment.py` | dataset, timing, mask, and cache identity |
| Balloon inference | `test_t3a_balloon_robust_ssm.py`, `test_t3a_balloon_joint_ssm.py` | forward dynamics, joint likelihood, masking, temporal operators and posterior moments |
| T3 diagnostics | `test_t3a_balloon_robust_p0.py`, `test_t3_measured_reconstruction_null.py`, `test_t3_identifiability.py`, `test_t3_multisession_loso.py` | synthetic recovery, observation nulls, parameter identifiability and fit-only LOSO |
| T3c composite | `test_t3c_hierarchical_admission.py`, `test_t3c_composite_synthetic_t2.py` | array-free admission and known-truth composite screening |
| Step5 | `test_step5a_inference_consistency.py`, `test_step5.py`, `test_step5_observation_diagnostic.py`, `test_step5_observation_repair.py` | likelihood calibration, shared observation baselines, fold isolation and observation/mask regression |
| Overnight SSM | `test_ssm_overnight_diagnostics.py` | fixed task identities, observation contracts, snapshot/data-root separation, scheduling and report readers |
| Trajectory metrics | `test_trajectory_reliability.py` | reconstruction and predictive reliability; residual-field reader compatibility is exercised by the overnight report tests |
| Retained E2/T0 | `test_physiology_semantic_*`, `test_token_physiology*.py` | frozen tokenizer and Atlas replay |
| R0/R1/R2 (stopped) | `test_r0p_*`, `test_build_r1*`, `test_qualify_r1p_*`, `test_r2d_*` | regression coverage for stopped sealed preregistration, no-leakage, and negative-result records |
| Croce/solver (stopped) | `test_croce_*`, `test_benchmark_solver_optimizations.py` | regression coverage for the stopped physical-model/cache implementation |
| Infrastructure | `test_archive_isolation.py`, `test_project_state.py` | archive boundary and unified state |
| Figures | `test_physiology_semantic_architecture_svg.py` | source/provenance consistency |

These rows describe software regression coverage, not an experiment
queue or authorization to launch a new lane. Ordinary tests use temporary,
non-authorizing fixtures. Never alter sealed records or weaken a test to make a
clean checkout green.
