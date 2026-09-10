# SSM overnight diagnostics

Run: `20260909_overnight_v2_verified`. Execution: **completed**. Teacher qualification: **none**.

72 unique original training trials: subjects 01/09/18, sessions 01/03/05; original MA positions 4/9 excluded before preprocessing.
Every outer fold refits all learned objects using 6 training / 2 evaluation trials per session. Three inner folds only use outer training.
The N5 all-training common gauge is descriptive and never enters outer candidate scoring.

| Family | Completed cells / fixed cells | Failed/unavailable/budget cells | Actual / planned solves |
|---|---:|---:|---:|
| N1 | 588/588 | 0 | 3600/3600 |
| N2 | 3168/3169 | 1 | 3192/3192 |
| N3 | 759/780 | 21 | 3024/4032 |
| N4 | 261/282 | 21 | 2754/3258 |
| N5 | 1137/1158 | 21 | 8520/8856 |
| N6 | 1045/1074 | 29 | 4794/5046 |

A completed cell can contain failed individual fits. See `failure_attribution.csv` and per-trial tables; no successful-only denominator is used for candidate admission.

| Rule | Complete outer trials / 72 | Endpoint | Relative improvement | Priority review |
|---|---:|---|---:|---|
| N3 | 51/72 | B | NOT_ESTIMATED | False |
| N4 | 52/72 | fnirs | NOT_ESTIMATED | False |
| N5 | 53/72 | B | NOT_ESTIMATED | False |
| N6 | 53/72 | B | NOT_ESTIMATED | False |

O2 nonlinear-Gaussian W=0/full mean precheck: 24/24 complete; mean NRMSE {'r': 0.5263342964423754, 'clean_EEG': 0.5263342964423754, 'clean_HbO': 0.18150958750761062, 'clean_HbR': 0.2073634738638309}; r correlation 0.8476758094322516.
O2 native measured branch: NOT_IMPLEMENTED. The current helper combines nonlinear EEG power/optics/motion transforms without an exposed pre-linear noise layer.
MAP intervals and marginal likelihood: NOT_ESTIMATED. No pointwise variance is borrowed. Shared native-noise cross-operator prediction intervals are NOT_ESTIMATED.
EEG voltage reconstruction: NOT_SUPPORTED after log-power/PCA. HbO/HbR retain order, shared scale and HbT algebra; no sign flip was selected.
N4 local coordinates use the nearest up to three EEG locations to the fold-fixed fNIRS pair in compatible native montage metadata. If unavailable, F3 is explicitly labelled anatomical_mapping_unverified. EOG is train-fitted; artificial contamination uses an un-injected measured reference, not clean truth.
N5 bootstrap: 200 trial resamples conditional on one fixed within-subject gauge; not population ICC or parameter posterior coverage.
N6 mean-driver replay is not a private-information proportion. Nonlinear posterior means need not obey deterministic closure.

Detailed evidence: `task_table.csv`, `case_status.jsonl`, `candidate_table.csv`, `direction_decision_table.csv`, `failure_attribution.csv`, `N1/`–`N6/` and compact `cells/*/*.npz`.
No optional cross-factor combination, additional frequency band, whole-session generalization extension, or tokenizer training was selected from these outer scores.
Next: review complete single-factor rules and their null/truth checks. Incomplete rules remain diagnostic; retain every failed identity before deciding a new version.

Completed diagnostic comparisons (each denominator is fixed before inference):

| N2 law / processing / full W=0 | Solver | Trials | r NRMSE | EEG / HbO / HbR NRMSE |
|---|---|---:|---:|---|
| linearized_gaussian / model | O0 | 24/24 | 0.5785 | 0.5785 / 0.1879 / 0.2172 |
| linearized_gaussian / model | O1 | 24/24 | 0.5596 | 0.5596 / 0.1853 / 0.2031 |
| linearized_gaussian / model | O2 | 24/24 | 0.5668 | 0.5668 / 0.1876 / 0.2182 |
| linearized_gaussian / combined | O0 | 24/24 | 0.8469 | 0.8469 / 0.9514 / 0.8847 |
| linearized_gaussian / combined | O1 | 24/24 | 0.5771 | 0.5771 / 0.2831 / 0.3007 |
| linearized_gaussian / combined | O2 | 24/24 | 0.5837 | 0.5837 / 0.2846 / 0.3104 |
| nonlinear_gaussian / model | O0 | 24/24 | 0.5326 | 0.5326 / 0.1887 / 0.2136 |
| nonlinear_gaussian / model | O1 | 24/24 | 0.5364 | 0.5364 / 0.1829 / 0.2167 |
| nonlinear_gaussian / model | O2 | 24/24 | 0.5263 | 0.5263 / 0.1815 / 0.2074 |
| nonlinear_gaussian / combined | O0 | 24/24 | 0.8552 | 0.8552 / 1.0065 / 0.9415 |
| nonlinear_gaussian / combined | O1 | 24/24 | 0.5492 | 0.5492 / 0.2381 / 0.2883 |
| nonlinear_gaussian / combined | O2 | 24/24 | 0.5397 | 0.5397 / 0.2321 / 0.2768 |
| nonlinear_student_t / model | O0 | 24/24 | 0.5154 | 0.5154 / 0.1380 / 0.1713 |
| nonlinear_student_t / model | O1 | 24/24 | 0.5448 | 0.5448 / 0.1565 / 0.2023 |
| nonlinear_student_t / model | O2 | 24/24 | 0.5311 | 0.5311 / 0.1571 / 0.1816 |
| nonlinear_student_t / combined | O0 | 24/24 | 0.8060 | 0.8060 / 0.8724 / 0.8765 |
| nonlinear_student_t / combined | O1 | 24/24 | 0.5487 | 0.5487 / 0.2179 / 0.2772 |
| nonlinear_student_t / combined | O2 | 24/24 | 0.5389 | 0.5389 / 0.2193 / 0.2641 |

N6 mean-driver replay: 64/72 complete, mean gap / training SD = [0.0, 0.18077572097462402, 0.46075957093004516].
A replay gap alone is not a physical violation. Matched and mismatch synthetic replay values remain in each N1/N6 synthetic row.

| N5 subject / session | W curve values | Grid maximum W | Boundary | Half-sample W |
|---|---:|---:|---|---|
| subject_01 / session_01 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_01 / session_03 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_01 / session_05 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_09 / session_01 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_09 / session_03 | 124/136 | NOT_ESTIMATED | NOT_ESTIMATED | NOT_ESTIMATED |
| subject_09 / session_05 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_18 / session_01 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_18 / session_03 | 136/136 | -0.5 | True | [-0.5, -0.5] |
| subject_18 / session_05 | 136/136 | -0.5 | True | [-0.5, -0.5] |
