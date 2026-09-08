# Observation contract repair and regression

This is engineering regression and a bounded training-only diagnostic. No teacher qualification is granted.

## Known coordinate scaling

| Cohort | Completed / expected | Invariance passed | Max density error | Max parameter CDF error |
|---|---:|---:|---:|---:|
| reused | 24/24 | 24 | 6.82e-13 | 9.13e-14 |
| independent | 24/24 | 24 | 9.09e-13 | 1.17e-13 |

| Cohort / branch | Δ log L(−0.5 − 0) [95% descriptive CI] | Slow preference | r coverage | HbO coverage | HbR coverage |
|---|---:|---:|---:|---:|---:|
| reused / original | -15.2339 [-19.6936, -11.0245] | 2/24 | 95.174% | 92.569% | 91.736% |
| reused / observation_only | 1.7406 [0.6685, 2.8133] | 18/24 | 87.674% | 28.125% | 30.764% |
| reused / synchronized | -15.2339 [-19.6936, -11.0245] | 2/24 | 95.174% | 92.569% | 91.736% |
| independent / original | -18.3493 [-23.8102, -13.5262] | 0/24 | 95.208% | 95.938% | 93.924% |
| independent / observation_only | 2.0406 [0.8665, 3.1799] | 19/24 | 86.076% | 30.833% | 35.208% |
| independent / synchronized | -18.3493 [-23.8102, -13.5262] | 0/24 | 95.208% | 95.938% | 93.924% |

The deliberately mismatched branch scales observations alone. Synchronized scaling transforms the operator and noise once; canonical latent states, parameter posterior and density correction are tested. Reused trials are regression inputs, not independent validation.

## Temporal reference

The batch reference is exact only for a rest-linearized Gaussian path and Gaussian observation noise on the retained SVD subspace. Nonlinear Student-t inputs use a moment approximation. This is not a transformed independent Student-t likelihood and does not inherit A0 qualification.

| Generating law / processing / branch | Complete fits | Retained rank | r / HbO / HbR canonical coverage | Δ log L(−0.5 − 0) |
|---|---:|---:|---|---:|
| linearized_gaussian / model__synchronized | 48/48 | [192] | 95.117% / 92.969% / 95.508% | -6.7232 |
| linearized_gaussian / baseline__observation_only | 48/48 | [192] | 87.174% / 43.620% / 54.362% | -2.9687 |
| linearized_gaussian / baseline__synchronized | 48/48 | [189] | 94.661% / 92.513% / 92.839% | -6.2903 |
| linearized_gaussian / fnirs_filter__observation_only | 48/48 | [192] | 81.641% / 24.479% / 31.380% | -1.6965 |
| linearized_gaussian / fnirs_filter__synchronized | 48/48 | [178] | 95.117% / 93.750% / 95.247% | -6.6379 |
| linearized_gaussian / combined__observation_only | 48/48 | [192] | 74.674% / 27.669% / 37.891% | 5.1996 |
| linearized_gaussian / combined__synchronized | 48/48 | [177] | 94.922% / 93.490% / 94.141% | -6.3011 |
| nonlinear_student_t / model__synchronized | 48/48 | [192] | 94.727% / 92.773% / 91.471% | -5.7717 |
| nonlinear_student_t / baseline__observation_only | 48/48 | [192] | 85.612% / 29.883% / 40.755% | -1.9221 |
| nonlinear_student_t / baseline__synchronized | 48/48 | [189] | 93.880% / 91.341% / 86.719% | -5.228 |
| nonlinear_student_t / fnirs_filter__observation_only | 48/48 | [192] | 80.924% / 27.539% / 35.417% | -0.8273 |
| nonlinear_student_t / fnirs_filter__synchronized | 48/48 | [178] | 94.596% / 95.052% / 91.927% | -5.5686 |
| nonlinear_student_t / combined__observation_only | 48/48 | [192] | 79.167% / 29.036% / 37.435% | 5.8066 |
| nonlinear_student_t / combined__synchronized | 48/48 | [177] | 94.271% / 93.099% / 88.281% | -5.2056 |

Rank-tolerance sensitivity: maximum transformed-mean change 0.00600944; maximum change in the W likelihood difference 0.0330596. Absolute log densities on different retained subspaces are not directly comparable. Full ranks, discarded singular values, canonical and processed coverage, and approximation discrepancies are retained in JSON.

## Identical-input numerical continuation

Original failed tasks restored: 2/42; these represent 21 unique input/curve tasks because fNIRS-only coordinates were duplicated. No independent trial was added.

Saturation occurred in 3 unique subject/trial identities; f<0.01 occurred in 3. Minimum internal evaluated flow: 1.3168967017204317e-215. These are internal RK4 evaluation diagnostics, not counts of independent failures or a physical qualification.

| Curve | Completed | W maximum (complete curves only) | Δ log L(−0.5 − 0) |
|---|---:|---:|---:|
| subject_01__broadband_pca__fNIRS_only | 0/17 | None | None |
| subject_01__broadband_pca__EEG_only | 2/2 | None | 0.0 |
| subject_01__broadband_pca__joint | 17/17 | -0.5 | 999.9423695076493 |
| subject_01__local_F3_alpha__EEG_only | 2/2 | None | 0.0 |
| subject_01__local_F3_alpha__joint | 17/17 | -0.5 | 776.0117097362672 |
| subject_09__broadband_pca__EEG_only | 2/2 | None | 0.0 |
| subject_09__broadband_pca__fNIRS_only | 14/17 | None | None |
| subject_09__broadband_pca__joint | 17/17 | -0.5 | 1261.9121827335343 |
| subject_09__local_F3_alpha__EEG_only | 2/2 | None | 0.0 |
| subject_09__local_F3_alpha__joint | 17/17 | -0.5 | 1071.5844988690164 |
| subject_18__broadband_pca__EEG_only | 2/2 | None | 0.0 |
| subject_18__broadband_pca__fNIRS_only | 17/17 | -0.5 | 93.86112278654082 |
| subject_18__broadband_pca__joint | 17/17 | -0.5 | 859.1517110314671 |
| subject_18__local_F3_alpha__EEG_only | 2/2 | None | 0.0 |
| subject_18__local_F3_alpha__joint | 17/17 | -0.5 | 616.1062115187442 |

## Same-fold fixed SSM and linear control

All dynamics are fixed at W=0. Inputs, outer folds, native masks, target coordinates, training variance normalization and null donors match. No extra measurement-gain candidate is selected. Scores are negative normalized MSE, so a positive increment favors the correctly paired fixed SSM.

| Target / contrast | Complete subjects | Increment [95% descriptive CI] |
|---|---:|---:|
| EEG / own_context | 3/3 | -4.288765 [-6.660364, -3.071063] |
| EEG / linear_basic | 3/3 | -4.495305 [-6.709398, -3.145127] |
| EEG / linear_joint | 3/3 | -4.487543 [-6.696620, -3.158194] |
| EEG / independent_pairing | 3/3 | 0.254806 [0.090571, 0.428315] |
| EEG / circular_shift | 3/3 | -2.753209 [-4.694187, -1.514857] |
| fNIRS / own_context | 2/3 | 0.467215 [0.307019, 0.627411] |
| fNIRS / linear_basic | 2/3 | -3.903186 [-4.443647, -3.362725] |
| fNIRS / linear_joint | 2/3 | -3.902026 [-4.447861, -3.356191] |
| fNIRS / independent_pairing | 2/3 | 0.029731 [-0.116304, 0.175766] |
| fNIRS / circular_shift | 2/3 | 0.197848 [0.042774, 0.352922] |

Retained failed jobs/curves/conditions: 21. Missing cases do not become successful-case qualification evidence.

Original trial positions 4/9 and subjects 19–29 are not processed. The temporal reference has not replaced measured pointwise inference. Persistent bias, extreme flow, W pressure and failure to beat pairing controls remain separate questions from numerical/coordinate correctness. No protected campaign, comprehensive UQ, support expansion, tokenizer training or promotion is performed.
