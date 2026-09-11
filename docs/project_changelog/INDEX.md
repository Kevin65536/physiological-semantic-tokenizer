# Project Operations Changelog

This directory preserves changes to repository organization and operational policy. It is intentionally separate from [`architecture_changelog/`](../architecture_changelog/INDEX.md), which is limited to model structure and scientific data-contract changes.

Included here:

- storage and result-directory normalization;
- documentation authority and archive moves;
- code/config/test namespace isolation;
- launcher and repository-maintenance policy.
- experiment baseline freezes and archive handoffs that do not change model structure.

## Records

| Date | Title | Scope |
| --- | --- | --- |
| 2026-05-11 | [Phase 1 Gate1 Baseline Lock and Archive](2026-05-11_phase1_gate1_baseline_lock.md) | Experiment baseline freeze and archive handoff |
| 2026-06-04 | [Storage Layout Normalization](2026-06-04_storage_layout_normalization.md) | Cache/run/result namespaces and archival layout |
| 2026-07-01 | [Documentation and Run Archive Isolation](2026-07-01_document_and_run_archive_isolation.md) | Documentation authority and historical run isolation |
| 2026-07-02 | [Code and Configuration Archive Isolation](2026-07-02_code_and_config_archive_isolation.md) | Compatibility package and script/config/test isolation |
| 2026-07-03 | [SVG Architecture Visualization System](2026-07-03_svg_architecture_visualization_system.md) | Maintained current diagram and plan-specific change overlays |
| 2026-07-06 | [Shared-state Reconstruction-bound Diagnostic](2026-07-06_shared_state_reconstruction_bound.md) | Non-gate bound analysis, result provenance, and architecture-plan overlay |
| 2026-07-06 | [Cross-dataset Shared Neural State Diagnostic](2026-07-06_cross_dataset_shared_neural_state.md) | Four-dataset delayed predictive-residual experiment, adapters, and visual evidence |
| 2026-07-10 | [Four-dataset Unified Quality Loader](2026-07-10_four_dataset_unified_quality_loader.md) | Correct dataset scope, unified loading contract, and quality-report verification |
| 2026-07-14 | [Single-Trial EEG Artifact-cleaning Candidate](2026-07-14_single_trial_eeg_artifact_candidate.md) | Raw/clean loader branch, adaptive artifact QC, full audit, and admission decision |
| 2026-07-14 | [Single-Trial EEG Artifact-cleaning v3 Admission](2026-07-14_single_trial_eeg_artifact_final_admission.md) | Controlled-artifact sham validation, versioned cache, and default-branch admission |
| 2026-07-25 | [Disable EEG Artifact-mask Authority](2026-07-25_disable_eeg_artifact_mask_authority.md) | All-false compatibility mask, boundary-only validity, and retirement of artifact-gated patch exclusion |
