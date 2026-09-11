# E0 pre-sign-calibration diagnostic analysis

_Historical analysis of the diagnostics that preceded sign calibration, 2026-07-09; status corrected 2026-07-24_

---

> **Historical-scope notice (updated 2026-07-24):** This report records the
> pre-sign-calibration Croce E0-v1/v2 and E0-D1–D5 diagnostics as they stood on
> 2026-07-09. It does not represent the current E0 status. The sign-calibrated
> adaptive SSM physical teacher passes complete E0 and its physiological
> information, including fNIRS, is fully acceptable. See
> [`20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md`](20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md).

> **Comparative-method audit notice (2026-07-17):** Sections 4.3 and the
> priority-2/priority-7 EFRM and STA-Net recommendations below are preserved as
> historical proposals. The later checkout audit found that neither source tree
> is integrated with the unified loader, downstream target contract, shared
> subject splits, regression metrics, or active artifact schema. The earlier
> “one training run” and “low implementation impact” estimates are superseded by
> [`comparisons/PROTOCOL.md`](../../comparisons/PROTOCOL.md).
> Both methods remain candidates; availability of source code does not admit
> either as an E0 control or downstream SOTA baseline.

## Executive Summary

At the time, two Croce validation attempts and five follow-up diagnostics
motivated replacing the sign-ambiguous coordinate contract. That historical
finding led to the adaptive SSM and gauge/sign calibration. It must not be read
as a rejection of the current physical teacher, which now passes complete E0.

The project deep-research report, *Deep Research on Physiology-Semantic
Tokenization for EEG–fNIRS Coupling*, independently reached the same conclusion
and provided a literature-grounded framework for revision. The PDF is not part
of the current checkout; this document preserves the report's conclusions
alongside the project's own diagnostic evidence.

---

## 1. What the E0 gate actually requires

E0 is not "the teacher must perfectly reconstruct fNIRS." The
[E0-v2 information contract](../../architecture_changelog/2026-07-03_e0_v2_teacher_information_contract.md)
evaluates six independent layers:

| Layer | Meaning | E0-v2 result |
|---|---|---|
| Measurement adapter | Units/scale contract auditable and reversible | **PASS** |
| Local target observability | Each coordinate identifiable from its declared modality patch | **PASS** (3/4 EEG, 6/6 fNIRS) |
| Posterior uncertainty calibration | Teacher variance covers true error at declared rate | Historical negative label |
| Finite-vocabulary transmissibility | 128 prototypes can represent admitted target geometry | **PASS** (EEG R²=0.918, fNIRS R²=0.949) |
| Physical observation prediction | Semantic-only decoder reconstructs clean observation in canonical space | Historical pre-calibration negative label |
| Continuous coupling upper bound | EEG state history adds information beyond fNIRS history | **PASS** (0.17 nats) |

Under the old coordinate contract, two layers received negative labels. Those
labels were superseded by sign calibration and are not current E0 results.

---

## 2. Detailed pre-calibration diagnostic evidence

### 2.1 E0-v1 (2026-07-03): First formal validation

| Metric | EEG | fNIRS |
|---|---|---|
| Normalized predictive gain | +0.756 | **-1.504** |
| 95% bootstrap interval | [0.691, 0.805] | [-2.384, -0.663] |
| Positive subjects | 5/5 | **0/5** |

The pre-calibration fNIRS clean-waveform endpoint did not outperform its
history baseline under the old coordinate contract.

### 2.2 E0-v2 (2026-07-03): Layered validation

**Historical fNIRS physical-observation diagnostic:**
- Clean MSE: 2.193 vs. history baseline: 0.834
- Mean gain: -1.359 (teacher is worse than simple history)
- 0/5 validation subjects positive

**Historical posterior-uncertainty diagnostic:**
- Even after synthetic-truth variance scaling, three hemodynamic coordinates (delta_f, delta_hbo, delta_hb) remained outside the sample-size-derived 95% coverage band
- Real-data student errors were much larger than teacher posterior SD
- Teacher is overconfident about wrong predictions

### 2.3 E0-D1 (2026-07-06): Shared-state reconstruction bound

This diagnostic tested whether the paired observations themselves support one low-dimensional state describing both modalities, **without using the Croce solver at all**.

Key finding at k=5 (matching Croce state dimension):

| Model | EEG descriptor R² | fNIRS descriptor R² | Loading balance |
|---|---|---|---|
| Validation-oracle joint PCA | 0.893 | 0.931 | 0.041 |
| Train-fitted joint PCA | 0.835 | 0.880 | — |
| **CCA shared state** | **0.098** | **-0.222** | — |
| Separate modality PCA | 0.880 | 0.965 | — |

The decisive result: when axes are forced to be genuinely cross-modally shared (CCA), validation canonical correlation collapses to **0.090 (waveform) and 0.004 (descriptor)**. The nominally "joint" PCA components are just allocating axes to different modalities, not discovering shared structure.

**Conclusion:** The data do not support a five-dimensional, same-patch, linear shared representation that generalizes across subjects. This is a data-level constraint, not a Croce-solver issue.

### 2.4 E0-D2 (2026-07-06): Cross-dataset delayed predictive residual

Four datasets (Single-Trial, REFED, Simultaneous, Visual), two subjects each, 5-second EEG-leading lag, with self-history/trial-phase/condition removed:

| Dataset | Cross-inferable EEG→fNIRS | Cross-inferable fNIRS→EEG | Joint ceiling |
|---|---|---|---|
| Single-Trial | **0%** | **0%** | 3.97% |
| REFED | **0%** | **0%** | 0.62% |
| Simultaneous | **0%** | **0%** | 1.23% |
| Visual | **0%** | **0%** | 2.56% |

No dataset produced a positive cross-inferable shared fraction. The joint ceiling (which can see the target modality) reaches only 0.6-4% of predictive residual variance. This means that even with both modalities, the shared structure is very small.

### 2.5 E0-D3/D4/D5 (2026-07-08): Lin 2024 inspired diagnostics

Three variants of Lin-style EEG→fNIRS HRF modeling, from Croce-cache to raw continuous records to a separate Simultaneous EEG&NIRS dataset:

| Diagnostic | Best EEG→fNIRS R² (cross-validated) | fNIRS self-persistence R² |
|---|---|---|
| E0-D3: Croce cache, subject-specific | -0.024 to -0.152 | 0.997 |
| E0-D4: Raw Single-Trial session, TRTD | -1.889 | 0.997 |
| E0-D5: Raw Simultaneous session, TRTD | -0.616 | 0.992 |

Even the **in-sample upper bound** (same trials for fit and eval) reached only R²=0.022 (D4) and R²=0.004 (D5). The fNIRS self-persistence baseline consistently reaches R²≈0.997.

**D5 conclusion at the time:** The Simultaneous dataset already stores fNIRS
as oxy/deoxy concentration (mmol/L), so approximate optical-to-HbO conversion
did not explain the old negative label. The later adaptive SSM and sign
calibration supersede that one-dimensional-driver contract.

---

## 3. Root cause analysis

### 3.1 Primary: The shared state assumption is too strong

The Croce model assumes five latent variables (s, delta_f, delta_hbo, delta_hb, r) simultaneously explain EEG and fNIRS through fixed observation equations. The diagnostics converge on a single finding: **this assumption is violated in the current data**. The evidence:

1. CCA components don't generalize (validation r≈0)
2. Joint PCA allocates axes to different modalities, not shared dimensions
3. Cross-inferable shared fraction is 0% across four datasets
4. Even the in-sample upper bound for EEG→fNIRS prediction is near zero

This does not mean "no neurovascular coupling exists." It means a five-dimensional, same-patch, linear shared state that is independently inferable from either modality is too ambitious.

### 3.2 Secondary: fNIRS observation model is misspecified

The teacher identified state coordinates above permutation nulls, while the
pre-calibration clean-fNIRS score was strongly negative in the raw internal
coordinate system. The later sign-calibrated contract supersedes this
interpretation. At the time, this suggested:

- The Croce state-space dynamics may capture some latent structure
- But the forward observation mapping (state → clean fNIRS waveform) is wrong
- The teacher's posterior uncertainty is miscalibrated — it's confident about wrong predictions

### 3.3 Secondary: Rigid fixed-parameter dynamics

The Croce solver uses fixed dynamics parameters. The literature (adaptive HRF work, DCM model comparison, Lin 2024 subject-specific HRF) consistently shows that hemodynamic response varies by brain region, task, and subject. A fixed low-order state-space model cannot represent subject-dependent, condition-dependent, or region-dependent delays.

### 3.4 Secondary: fNIRS signal is dominated by private components

Across all diagnostics, fNIRS self-persistence reaches R²≈0.997. Five private PCA components achieve R²=0.99993. The shared EEG→fNIRS component is tiny relative to fNIRS's own temporal continuity. Most predictable fNIRS structure is carried by modality-private slow trajectory, baseline drift, and vascular components.

### 3.5 Secondary: Timescale mismatch is not adequately modeled

EEG responds at millisecond scale; fNIRS at seconds. The cross-dataset diagnostic found 0% cross-inferable shared fraction at same-time patches, but the continuous coupling upper bound found positive delayed information (0.17 nats). This means the useful bridge is delayed EEG history → fNIRS predictive residual, not same-patch shared state.

### 3.6 Tertiary: Teacher misspecification propagates through multiple loss entry points

As the deep research paper notes, the current architecture has multiple teacher contact points with different risk profiles:

| Loss | Contact point | Risk if teacher is wrong |
|---|---|---|
| State loss | Continuous semantic latent | **Low** — residual branch can compensate |
| Prototype loss | Codebook prototype → state head | **High** — bakes wrong semantics into vocabulary |
| Masked-state loss | Quantized expected embedding → context | **High** — bakes wrong temporal grammar |
| Reconstruction loss | Semantic + residual → waveform | **Low** — information preservation only |
| VQ commitment loss | Continuous → discrete latent | **Medium** — propagates prototype bias back |

The current `private_weight=0.0` means the residual branch has no explicit shaping objective, making it a free absorber of whatever the semantic branch fails to encode. Good reconstruction with weak teacher pressure can result in generic semantic tokens and information-rich residuals — exactly the opposite of the design intent.

---

## 4. Comparison with existing literature

### 4.1 What the deep research paper recommends

The deep research paper identified the same pre-calibration asymmetry and
recommended five changes:

1. **Redefine E0** as "prove a restricted shared teacher subspace exists" rather than "prove the whole five-dimensional shared state is real"
2. **Replace fixed hemodynamic mapping** with hierarchical subject/condition-specific lag/HRF components
3. **Staged teacher trust policy**: allow teacher supervision only through continuous state head first; block prototype/context teacher losses until teacher passes calibration
4. **Redefine the bridge variable** as EEG-derived neural envelope (frequency-aware), not raw phase-consistent latent
5. **Add a non-mechanistic ceiling model** (seq2seq EEG→fNIRS predictor) to test whether coupling signal is absent vs. model is too rigid

### 4.2 Alignment with close-route methods

| Method family | Key insight for this project |
|---|---|
| Lin 2024 (subject-specific NVC) | Shared neural drive should be estimated jointly with subject-specific hemodynamic parameters |
| EFRM 2025 (shared+private EEG-fNIRS) | Shared component can have its own training signal; doesn't need to explain entire observation |
| Rosa/DCM (Bayesian NVC comparison) | Compare alternative mechanistic models rather than committing to one |
| Sirpal (multimodal autoencoder) | EEG→fNIRS ceiling model answers "is there a bridge at all?" |
| LaBraM/NeuroRVQ (biosignal tokenization) | Multi-scale/frequency-aware structure helps tokenizer allocate capacity to physiology |

### 4.3 What has been proposed but not yet tested in this project

The EFRM and STA-Net implementations have been added to `comparative_methods/` but have not been evaluated as teacher-family controls or ceiling models for E0.

---

## 5. Recommended improvements (priority-ordered)

### Priority 1: Narrow the teacher contract (architectural)

**What to change:**
- Replace the five-dimensional same-patch shared state with a three-component model:
  - **Shared neural driver** `r_shared(t)`: narrow, only coordinates with cross-subject evidence
  - **Hemodynamic transition state** `H(t)`: delayed, receives `r_shared` history
  - **Modality-private observation states**: EEG-private and fNIRS-private
- Only `r_shared` and its delayed hemodynamic projection supervise shared token identity
- All other state coordinates become modality-private targets

**Evidence:** E0-D1 shows separate modality models achieve R²=0.880/0.965 while shared CCA achieves R²=0.098/-0.222. The data support rich private states and a narrow shared driver.

**Implementation impact:** Moderate. Changes to `physical_state_teacher.py` (target splitting), `physiology_semantic.py` (loss routing), and config. Compatible with existing tokenizer architecture.

### Priority 2: Build a non-mechanistic ceiling model (diagnostic)

**What to change:**
- Implement a flexible EEG→fNIRS prediction model (e.g., EFRM-style shared+private autoencoder, or LSTM/CNN seq2seq) using the existing EFRM code in `comparative_methods/`
- Train it as a **diagnostic tool only** on the existing paired data
- Measure: "How much fNIRS variance is predictable from EEG when model rigidity is not the bottleneck?"
- If ceiling is high → problem is mechanistic teacher rigidity → redesign teacher
- If ceiling is low → coupling signal itself is narrow → accept limited shared supervision

**Evidence:** Sirpal et al. showed LSTM/CNN autoencoder can predict fNIRS from EEG in resting-state. EFRM explicitly learns shared representations without biophysical constraints. This test costs one training run and resolves the "absent signal vs. wrong model" ambiguity.

**Implementation impact:** Low. EFRM code already exists in `comparative_methods/`. Requires only a training config and evaluation script.

### Priority 3: Implement staged teacher trust policy (training)

**What to change:**
- Phase 1: Enable only `state_weight` (continuous latent supervision). Disable `prototype_weight` and `masked_state_weight`. Train to convergence.
- Phase 2: After continuous state head passes held-out calibration, enable `prototype_weight` with reduced coefficient.
- Phase 3: After prototype-state decoding passes held-out checks, enable `masked_state_weight`.
- Throughout: keep `private_weight > 0` (currently 0.0) to explicitly shape the residual branch.

**Evidence:** The deep research paper identifies prototype and masked-state losses as the highest-risk teacher entry points. Staged trust prevents baking teacher bias into discrete vocabulary before the teacher is validated.

**Implementation impact:** Low. All weights are already configurable. Requires only a training schedule and phase-gating logic.

### Priority 4: Add teacher-family comparison (evaluation)

**What to change:**
- Implement at least two alternative teacher candidates beyond Croce:
  1. **Data-driven multi-view teacher**: CCA/PCA-based, no biophysical dynamics
  2. **Physics-regularized hybrid**: Croce dynamics with learned observation model
- Evaluate all candidates on the same E0-v2 protocol
- A shared-state claim requires convergence across families or a mechanistic explanation for disagreement

**Evidence:** Rosa et al. and DCM literature show Bayesian model comparison is standard practice for neurovascular coupling models. The E0-D1 and E0-D2 diagnostics already use data-driven references.

**Implementation impact:** Moderate. Requires implementing alternative teacher adapters. The evaluation protocol (E0-v2) already exists.

### Priority 5: Add frequency-aware EEG preprocessing (signal processing)

**What to change:**
- Replace raw EEG patch input with band-specific features (delta, theta, alpha, beta, gamma envelopes)
- Or add a multi-scale patch pathway that preserves spectral structure
- The shared driver should be an EEG-derived neural envelope, not a raw-phase latent

**Evidence:** McLinden et al. show phase-amplitude coupling between fNIRS low-frequency and EEG high-frequency. Sirpal et al. show gamma-range EEG is most predictive of fNIRS. NeuroRVQ and BandVQ show multi-scale tokenization improves biosignal representation.

**Implementation impact:** Moderate. Changes encoder input and architecture. Compatible with existing patch grid.

### Priority 6: Strengthen fNIRS private-state modeling (architectural)

**What to change:**
- Add explicit per-channel baseline/drift terms `b_{s,c}(t)` estimated per subject, session, and channel
- Model slow vascular/components as a separate latent variable, not forced into shared driver
- Allow fNIRS residual dimension to be larger than currently planned (32 → 64 or more)

**Evidence:** fNIRS self-persistence reaches R²≈0.997. The dominant fNIRS structure is private temporal continuity plus channel-specific drift. Forcing this into a shared driver degrades the teacher.

**Implementation impact:** Low. Primarily configuration changes (residual dimension, baseline estimation in teacher adapter).

### Priority 7: Evaluate EFRM and STA-Net as comparative baselines

**What to change:**
- Complete the integration of EFRM and STA-Net from `comparative_methods/`
- Run them on the same paired data as the current teacher
- Report their EEG→fNIRS prediction performance alongside Croce teacher metrics

**Evidence:** These methods are already in the repository but unevaluated. EFRM's shared+private architecture mirrors the project's own design philosophy.

**Implementation impact:** Moderate. Requires completing the integration and writing evaluation configs.

---

## 6. What should NOT change

Based on the analysis, these aspects of the current design are well-supported and should be preserved:

1. **Independent EEG/fNIRS tokenizer inference** — the diagnostics confirm modalities carry largely private information
2. **Discrete semantic tokens + continuous residual** — this decomposition is correct; the issue is what supervises the semantic branch, not the architecture
3. **Delayed sequence-to-distribution coupling** — the continuous coupling upper bound is positive (0.17 nats), supporting delayed rather than same-time correspondence
4. **Uncertainty-weighted supervision** — the historical calibration diagnostic motivated improving variance estimates
5. **The core narrative** — "EEG token sequences provide incremental information about future fNIRS token distributions" remains viable
6. **Soft posterior and codebook embedding as first-class outputs** — the legacy postmortem evidence strongly supports this

---

## 7. Proposed revised E0 protocol (E0-v3)

### Admission criteria (before protected test opens):

1. **Measurement adapter**: Unchanged from E0-v2 (already passed)
2. **Narrowed shared target**: Only coordinates with held-out cross-modal evidence above time-shift null admitted as shared
3. **Private target observability**: Each modality's private state identifiable from its own patch (already passed for most coordinates)
4. **Teacher-family convergence**: At least two teacher families agree on admitted shared subspace, OR mechanistic explanation for disagreement is documented
5. **Non-mechanistic ceiling**: seq2seq/EFRM ceiling model establishes upper bound on EEG→fNIRS predictability
6. **Calibrated uncertainty**: Narrowed teacher passes posterior coverage check on admitted coordinates
7. **Continuous coupling upper bound**: Unchanged but computed on admitted shared coordinates only

### Blocking conditions (unchanged from E0-v2):
- Sign-calibrated physical-state supervision is authorized by the final complete-E0 decision
- Protected test opens only once after protocol freeze
- Visual review is co-equal with numerical checks

---

## 8. Implementation sequence

```
Week 1-2: Priority 2 (ceiling model) + Priority 4 (teacher-family comparison)
         → Resolves "absent signal vs. wrong model" ambiguity
         
Week 2-3: Priority 1 (narrow teacher contract) + Priority 5 (frequency-aware EEG)
         → Implements the architectural revision
         
Week 3-4: Priority 3 (staged trust) + Priority 6 (fNIRS private state)
         → Safer training and better fNIRS modeling
         
Week 4:   Priority 7 (EFRM/STA-Net baselines)
         → Comparative evidence for paper
         
Week 5:   E0-v3 formal validation
         → Freeze protocol, open protected test once
```

---

## 9. Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Ceiling model also shows near-zero EEG→fNIRS predictability | Medium | Accept that coupling signal is narrow; design tokenizer for private-state semantics with coupling as secondary claim |
| Sign-calibrated teacher drifts under future data changes | Medium | Re-run the calibrated contract and retain provenance |
| Subject-specific HRF doesn't generalize | High (already shown) | Use population HRF with random effects; don't require per-subject fitting |
| Architecture changes break existing software validation | Low | All losses are independently weightable; config changes are backward-compatible |

---

## References

Key project documents:
- [Target Architecture](../02_TARGET_ARCHITECTURE.md)
- [Consolidated method rationale](../../METHOD_RATIONALE.md)
- [Implementation Validation Plan](../04_IMPLEMENTATION_VALIDATION_PLAN.md)
- [Experiment Design](../05_EXPERIMENT_DESIGN.md)
- [Experiment Log](../06_EXPERIMENT_LOG.md)
- [Consolidated method rationale](../../METHOD_RATIONALE.md)

Diagnostic archives:
- [Shared-State Reconstruction Bound](../archive/diagnostics/09_SHARED_STATE_RECONSTRUCTION_BOUND.md)
- [Cross-Dataset Shared Neural State](../archive/diagnostics/10_CROSS_DATASET_SHARED_NEURAL_STATE_DIAGNOSTIC.md)
- [Lin 2024 Subject-Specific NVC](../archive/diagnostics/20260708_lin2024_subject_specific_nvc_diagnostic.md)
- [Lin 2024 Raw Session TRTD](../archive/diagnostics/20260708_lin2024_raw_session_trtd_diagnostic.md)
- [Lin 2024 Simultaneous Raw TRTD](../archive/diagnostics/20260708_lin2024_simultaneous_raw_trtd_diagnostic.md)

External:
- *Deep Research on Physiology-Semantic Tokenization for EEG–fNIRS Coupling*
  (project deep-research report; PDF not present in this checkout)
- Croce et al. (2017) — Bayesian sequential Monte Carlo for EEG-fNIRS
- Lin et al. (2024) — Subject-specific EEG-fNIRS NVC by task-related tensor decomposition
- EFRM (2025) — Multimodal EEG-fNIRS representation learning
- Rosa et al. (2011) — Bayesian comparison of neurovascular coupling models
- Sirpal et al. — Multimodal autoencoder predicts fNIRS from EEG
- LaBraM (2024) — Large Brain Model for EEG
- NeuroRVQ (2025) — Multi-scale EEG tokenization
