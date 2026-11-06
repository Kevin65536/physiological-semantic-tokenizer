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
| STA-Net source correction | method-local `STA-Net-PyTorch/tests/test_source_aligned_v2.py`, `tests/test_source_aligned_v3.py`, `tests/test_rebench_v1.py`; `test_downstream_input_audit.py` | independent Keras population-normalization/Pearson equations, fused/unfused moving variance, training-only inference-statistics calibration with frozen parameters, pre-projection positional encoding, channels-last flattening, exact input support, finite masked gradients, known-signal learning, fixed development scope and subject-after-seed aggregation |
| Frozen downstream classification | `test_downstream_input_benchmark.py`, `test_downstream_input_audit.py`, `test_downstream_rebench_eeg_models.py`; BrainFusion method-local `tests/test_rebench_v1.py` | masked REFED window-mean midpoint labels, immutable regression targets, exact prediction identity joins, training-only frozen readout, explicit 30 s / 2 s BrainFusion adaptations and subject OOF isolation |
| Pure fNIRS comparison | `test_downstream_fnirs_probe.py`, `test_downstream_input_audit.py` | official MtReNIRS adapter masks, registered template geometry, short-window support, training-only Riemannian references and complete subject-fold aggregation |
| Global latent vocabulary health | `test_dual_component_latent_health.py` | 30-second same-location global context, paired Hb colors, one hard ID per 2-second patch/branch, train-only initialization, per-dataset health, integer decoder isolation and fixed budgets |
| EEG voltage teacher feasibility | `test_dual_component_voltage_teacher.py` | full-spectrum inverse, full-rank covariance, zero-noise preservation, polarity, exact paired targets, and training/development read boundaries |
| EEG spatial/temporal voltage separation | `test_dual_component_voltage_separation.py` | frozen projection identities, unit-DC neighbor prediction, signed state controls, EOG read boundary, known perturbations and actual independent O transfer |
| Conditional voltage power reference | `test_dual_component_power_cap.py` | stable weighted second-moment projection, non-invariance counterexample, training-only power completion, complete carrier synthesis, paired S/O error identities, objective-only modality weights, same-objective continuation, metadata-only example/posterior scope, posterior perturbation visibility and complete failed-pair denominators |
| Frozen S-token lag inspection | `test_dual_component_token_lags.py` | signed lag, native window boundaries, full-context support, whole-trial stratified null, task-locked negative control and delayed synthetic association |
| Data/preprocessing | `test_unified_physiology.py`, `test_event_alignment.py` | dataset, timing, mask, and cache identity |
| Independent S/residual tokenizer | `test_dual_component_tokenizer.py`, `test_dual_component_tokenizer_data.py`, `test_dual_component_tokenizer_runner.py`, `test_dual_component_tokenizer_analysis.py`, `test_dual_component_refinement.py`, `test_dual_component_voltage.py`, `test_dual_component_eeg_o.py` | full Tak targets, training-only coordinates, local hard IDs, independent decoders, fixed residuals, atomic resume, joint masking, native-voltage proxy correction, overlapping ROI constraints, fixed synthesis, signed boundary-neighborhood errors, and development-only band-gain O estimation |
| Tak baseline and parameter diagnostics | `test_shared_driver_tak.py`, `test_shared_driver_tak_experiment.py`, `test_eeg_proxy_10hz.py`, `test_tak_repeatability.py`, `test_tak_task_parameters.py`, `test_tak_topography.py` | default Tak/10 Hz dispatch, native power preparation, physical-time masks and loss weights, four-state/p=v invariant, SPM-equation replay, discrete tangents, disjoint-subset reestimation, paired parameter controls, support-level task holdouts, full-denominator abstentions, retained-array identity, spatial geometry/masks, selection invariance and level-versus-map-shape checks |
| Sensor proxy interventions | `test_shared_driver_sensor_proxy_data.py`, `test_low_capacity_eeg_proxy.py`, `test_shared_driver_sensor_proxy_runner.py`, `test_sensor_proxy_contract_audit.py` | fixed Tak conditional prediction, fold-local coordinates, PC1 nesting, matched channel sets, whole-Hb masking, donor support and retained failures |
| Fixed Tak spatial mismatch | `test_tak_mismatch_spatial.py`, `test_tak_disagreement_comparison.py`, `test_tak_mismatch_runner.py`, `test_tak_mismatch_report.py` | record/site squared-error and matched-ratio aggregation, failed-window matching before aggregation, paired record identities, strict/relaxed coverage, early/late support, exploratory spatial selection, Visual identity boundary and interpolation without invented extrema or gap filling |
| Historical Balloon inference | `test_t3a_balloon_robust_ssm.py`, `test_t3a_balloon_joint_ssm.py` | retained inlet-balance forward dynamics, joint likelihood, masking, temporal operators and posterior moments |
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
