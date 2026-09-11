# Cross-dataset shared neural state diagnostic

- **Date:** 2026-07-06
- **Scope:** four-dataset experimental adapters, lagged shared-state analysis, and visualization
- **Scientific gate impact:** historical diagnostic only; current sign-calibrated E0 status is pass

## Change

Added a two-subject-per-dataset diagnostic for Single-Trial, REFED, Simultaneous EEG&NIRS, and Visual Cognitive Motivation. The workflow constructs one-second EEG spectral and fNIRS mean/slope features, removes modality self-history plus trial phase/condition, and fits a three-dimensional delayed CCA state over 0–10 second EEG-leading lags.

The formal run emits reciprocal cross-subject folds, trial/video block bootstrap intervals, alignment nulls, raw dataset inventory, adapter quality summaries, and publication-style SVG/PNG figures. Synthetic-truth tests verify that the estimator recovers a known five-second shared driver.

## Result boundary

The primary five-second cross-inferable fraction was non-positive in both directions for every dataset and is conservatively reported as `0%`. Joint-input state ceilings are reported separately and range from `0.62%` to `3.97%` of balanced predictive residual. These are standardized feature-space reconstruction fractions, not population estimates, waveform information fractions, or E0 evidence.

## Artifacts

- `experiments/evaluate_cross_dataset_shared_neural_state.py`
- `experiments/configs/physiology_semantic_tokenizer/cross_dataset_shared_neural_state.yaml`
- `tests/test_cross_dataset_shared_neural_state.py`
- `docs/physiology_semantic_tokenizer/archive/diagnostics/10_CROSS_DATASET_SHARED_NEURAL_STATE_DIAGNOSTIC.md`
- `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/`

## Validation

- synthetic delayed-shared-state recovery;
- small-unit EEG spectral feature regression;
- existing fNIRS standardization and E0-D1 tests;
- visual review of summary, lag-profile, and variance-attribution figures.
