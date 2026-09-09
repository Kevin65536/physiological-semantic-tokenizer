# Flow-domain and mask-specific observation experiment

The six-state drift, oxygen extraction, noise, W support and measurement gain are unchanged. Failures are classified and retained; no state is projected or redrawn. No new measured comparison or teacher qualification is claimed.

## Three original training trials

| Subject / training index | W | Outcome | Transition | First zero after update (s) | Event-relative zero (s) |
|---|---:|---|---:|---:|---:|
| subject_01 / 4 | 0.0 | flow_domain_exit | 108 | 0.04419948174634456 | 21.794199481746343 |
| subject_01 / 4 | -0.5 | flow_domain_exit | 112 | 2.3383774670599078e-05 | 22.75002338377467 |
| subject_01 / 7 | 0.0 | flow_domain_exit | 54 | 0.0002861214310373596 | 8.250286121431037 |
| subject_01 / 7 | -0.5 | flow_domain_exit | 58 | 0.12995447630611534 | 9.379954476306116 |
| subject_09 / 12 | 0.0 | completed | None | None | None |
| subject_09 / 12 | -0.5 | flow_domain_exit | 34 | 0.0402886779529628 | 3.290288677952963 |

Indices identify the original prepared training inventory, not native trial positions. Two W evaluations of one trial are not independent trials. Subject_09/trial_12 at W=0 already completed in the old replay; its completion is not a rescue. Full pre/post observation means, covariance matrices, and modality visibility for the last four updates are retained in replay.json.

## Mask-specific linear bridge

| Window | Mask | Old conservative EEG / HbO / HbR outputs | New outputs |
|---:|---|---|---|
| 64 | full | [64 64 64] | [64 64 64] |
| 64 | center_EEG | [48 64 64] | [48 64 64] |
| 64 | center_fNIRS | [64  0  0] | [64 48 48] |
| 64 | whole_EEG | [ 0 64 64] | [ 0 64 64] |
| 64 | whole_fNIRS | [64  0  0] | [64  0  0] |
| 120 | full | [120 120 120] | [120 120 120] |
| 120 | center_EEG | [104 120 120] | [104 120 120] |
| 120 | center_fNIRS | [120   0   0] | [120 104 104] |
| 120 | whole_EEG | [  0 120 120] | [  0 120 120] |
| 120 | whole_fNIRS | [120   0   0] | [120   0   0] |

The actual visible-only interpolation → processing → output-selection matrix transforms both means and the complete time-noise covariance. Hidden-center interpolants are not observations. Impulse construction and hidden-value interventions are checked independently. This does not validate raw EEG power, optical conversion or motion suppression.

## Same-input synthetic comparison at truth W=0

Each generating law has 24 independent trials. W=0 and W=−0.5 fits share each trial. The pointwise branch uses the existing nonlinear Student-t filter; the mask-specific branch remains a resting-state Gaussian moment reference, including on nonlinear Student-t data. Coverage below is canonical clean truth over all times; hidden-time metrics are separately retained in summary.json. Completed subsets cannot qualify an incomplete panel.

| Law / processing / mask / inference | W=0 trials | r / HbO / HbR coverage | HbR bias | Mean log L(−0.5)−log L(0) |
|---|---:|---|---:|---:|
| linearized_gaussian__model__full__pointwise_student_t | 24/24 | 94.60% / 91.93% / 91.02% | 0.00048 | -4.27079 |
| linearized_gaussian__model__full__mask_specific_gaussian_reference | 24/24 | 95.96% / 94.92% / 95.25% | 0.00030 | -3.95362 |
| linearized_gaussian__model__center_EEG__pointwise_student_t | 24/24 | 95.18% / 91.80% / 91.54% | 0.00045 | -2.93126 |
| linearized_gaussian__model__center_EEG__mask_specific_gaussian_reference | 24/24 | 95.57% / 94.79% / 95.38% | 0.00027 | -2.51967 |
| linearized_gaussian__model__center_fNIRS__pointwise_student_t | 24/24 | 94.66% / 92.64% / 92.19% | 0.00028 | -3.53779 |
| linearized_gaussian__model__center_fNIRS__mask_specific_gaussian_reference | 24/24 | 95.38% / 95.83% / 95.12% | 0.00014 | -3.0675 |
| linearized_gaussian__model__whole_EEG__pointwise_student_t | 24/24 | 96.03% / 92.32% / 91.73% | 0.00055 | -1.2653 |
| linearized_gaussian__model__whole_EEG__mask_specific_gaussian_reference | 24/24 | 96.16% / 94.79% / 95.51% | 0.00031 | -1.00569 |
| linearized_gaussian__model__whole_fNIRS__pointwise_student_t | 24/24 | 93.49% / 92.90% / 91.28% | 0.00085 | W unidentifiable (EEG only) |
| linearized_gaussian__model__whole_fNIRS__mask_specific_gaussian_reference | 24/24 | 95.12% / 96.48% / 95.12% | -0.00022 | W unidentifiable (EEG only) |
| linearized_gaussian__combined__full__pointwise_student_t | 24/24 | 79.10% / 23.70% / 32.94% | 0.00233 | 4.5779 |
| linearized_gaussian__combined__full__mask_specific_gaussian_reference | 24/24 | 95.70% / 94.40% / 96.09% | 0.00141 | -3.73043 |
| linearized_gaussian__combined__center_EEG__pointwise_student_t | 24/24 | 79.75% / 23.96% / 33.01% | 0.00231 | 2.76477 |
| linearized_gaussian__combined__center_EEG__mask_specific_gaussian_reference | 24/24 | 95.90% / 94.53% / 95.96% | 0.00139 | -2.50755 |
| linearized_gaussian__combined__center_fNIRS__pointwise_student_t | 24/24 | 79.69% / 26.63% / 34.77% | 0.00239 | 4.15563 |
| linearized_gaussian__combined__center_fNIRS__mask_specific_gaussian_reference | 24/24 | 95.38% / 95.31% / 96.35% | 0.00126 | -2.6229 |
| linearized_gaussian__combined__whole_EEG__pointwise_student_t | 24/24 | 85.16% / 24.22% / 33.33% | 0.00227 | 1.99178 |
| linearized_gaussian__combined__whole_EEG__mask_specific_gaussian_reference | 24/24 | 96.16% / 95.18% / 96.09% | 0.00130 | -1.05976 |
| linearized_gaussian__combined__whole_fNIRS__pointwise_student_t | 24/24 | 66.21% / 40.56% / 44.73% | 0.00920 | W unidentifiable (EEG only) |
| linearized_gaussian__combined__whole_fNIRS__mask_specific_gaussian_reference | 24/24 | 96.22% / 97.33% / 97.40% | 0.00216 | W unidentifiable (EEG only) |
| nonlinear_student_t__model__full__pointwise_student_t | 24/24 | 94.08% / 94.53% / 93.42% | 0.00014 | -5.97789 |
| nonlinear_student_t__model__full__mask_specific_gaussian_reference | 24/24 | 94.08% / 93.62% / 90.17% | 0.00025 | -5.21782 |
| nonlinear_student_t__model__center_EEG__pointwise_student_t | 24/24 | 94.53% / 94.86% / 93.68% | 0.00017 | -4.6481 |
| nonlinear_student_t__model__center_EEG__mask_specific_gaussian_reference | 24/24 | 93.95% / 93.23% / 90.10% | 0.00028 | -4.20494 |
| nonlinear_student_t__model__center_fNIRS__pointwise_student_t | 24/24 | 94.34% / 95.77% / 93.42% | 0.00030 | -4.25468 |
| nonlinear_student_t__model__center_fNIRS__mask_specific_gaussian_reference | 24/24 | 94.01% / 91.86% / 89.19% | 0.00014 | -3.28273 |
| nonlinear_student_t__model__whole_EEG__pointwise_student_t | 24/24 | 93.88% / 94.60% / 94.40% | 0.00026 | -1.94101 |
| nonlinear_student_t__model__whole_EEG__mask_specific_gaussian_reference | 24/24 | 93.36% / 92.71% / 90.30% | 0.00028 | -2.04317 |
| nonlinear_student_t__model__whole_fNIRS__pointwise_student_t | 24/24 | 94.73% / 98.05% / 97.66% | 0.00032 | W unidentifiable (EEG only) |
| nonlinear_student_t__model__whole_fNIRS__mask_specific_gaussian_reference | 24/24 | 95.05% / 96.29% / 95.44% | -0.00108 | W unidentifiable (EEG only) |
| nonlinear_student_t__combined__full__pointwise_student_t | 24/24 | 73.70% / 24.48% / 32.94% | -0.00321 | 4.55774 |
| nonlinear_student_t__combined__full__mask_specific_gaussian_reference | 24/24 | 94.14% / 96.68% / 94.60% | 0.00101 | -4.97123 |
| nonlinear_student_t__combined__center_EEG__pointwise_student_t | 24/24 | 73.31% / 24.22% / 32.62% | -0.00324 | 3.13254 |
| nonlinear_student_t__combined__center_EEG__mask_specific_gaussian_reference | 24/24 | 93.62% / 97.01% / 94.99% | 0.00104 | -3.88889 |
| nonlinear_student_t__combined__center_fNIRS__pointwise_student_t | 24/24 | 73.31% / 27.08% / 35.48% | -0.00315 | 4.84586 |
| nonlinear_student_t__combined__center_fNIRS__mask_specific_gaussian_reference | 24/24 | 94.01% / 96.81% / 93.10% | 0.00092 | -2.98614 |
| nonlinear_student_t__combined__whole_EEG__pointwise_student_t | 24/24 | 78.19% / 25.46% / 32.94% | -0.00330 | 1.86325 |
| nonlinear_student_t__combined__whole_EEG__mask_specific_gaussian_reference | 24/24 | 93.82% / 96.61% / 95.38% | 0.00095 | -2.01072 |
| nonlinear_student_t__combined__whole_fNIRS__pointwise_student_t | 24/24 | 69.79% / 46.74% / 49.28% | -0.00451 | W unidentifiable (EEG only) |
| nonlinear_student_t__combined__whole_fNIRS__mask_specific_gaussian_reference | 24/24 | 93.82% / 93.95% / 93.23% | -0.00401 | W unidentifiable (EEG only) |

Likelihood differences compare W only within the same mask/operator and retained support. EEG-only W is checked as invariant; floating-point signs near zero are not slow-W preferences. Absolute pointwise and temporal densities are not comparable. SVD rank, discarded directions, support residuals and tolerance sensitivity are retained; numerical rank truncation is an approximation.

## Remaining prerequisite

No nonlinear Student-t temporal inference has been validated under the new mask contract. The Gaussian moment reference cannot supply that prerequisite.

Therefore the requested new measured same-fold linear/pairing/shift comparison is not executed. Three old training failures were replayed for classification only. Subjects 19–29 and original held-out trial positions remain closed; old A0 evidence and Gaussian-reference coverage are not substituted for nonlinear temporal qualification.
