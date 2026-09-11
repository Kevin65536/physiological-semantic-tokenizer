# EEG–fNIRS data contract

_Active measured-data, alignment, mask, and cache rules; v1 historical target
boundary plus an exploratory continuous-target interface, 2026-08-22_

This document is the active data entrypoint. Dataset-native formats, tasks,
units, and original-document locations remain in the reference catalog
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md); the full dated
implementation audit remains in
[`physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md`](physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md).

## Version boundary

The measured-data loader, canonical identity, masks, splits, and protected
boundaries below are hard-frozen for the forward method generation. The
continuous-target interface is **exploratory and unimplemented**: only the rule
"preserve timestamps and construct the continuous trajectory before patching or
tokenization" is frozen. Sampling rate, target coordinates, filters, and target
dimension remain replaceable through a versioned implementation contract. This
does not relabel the current patch target, Croce cache, or any v1 checkpoint.
Documentation changes do not authorize measured or protected-data access.

## Registered measured datasets

| Registry ID | Main task surface | EEG | fNIRS | Important boundary |
| --- | --- | --- | --- | --- |
| `eeg_fnirs_single_trial` | motor imagery; mental arithmetic | 200 Hz analysis view | released dual-wavelength intensity, canonical 10 Hz view | 29 subjects; use native trial/session identity |
| `refed` | continuous valence/arousal video response | released 64-channel EEG | released HbO/HbR | targets are time-aligned continuous sequences, not video-class labels |
| `visual_cognitive_motivation` | RR/RF/FF/FR | released EEG | released HbO/HbR | reject `unknown`; S06 Part1 remains excluded under current evidence |
| `simultaneous_eeg_nirs` | n-back; word generation; DSR | 28 scalp channels after the admitted EOG-auxiliary branch | released HbO/HbR | DSR labels are EEG-native Go/No-go; fNIRS is synchronized context |

`croce_local_cache` is derived teacher evidence, not a fifth measured dataset.
Every analysis must name the measured dataset, task, record/session, subject,
condition/event, branch, and window/patch identity it consumes.

Read the original dataset documentation listed in
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md) before changing a loader
or interpreting units, task labels, timing, or licensing.

## Canonical sample identity

The join key is composed from stable dataset-native identity, not array order:

```text
dataset_id
subject_id
record_id / session_id
task_id
condition or event_id
window_start / window_end
modality branch
```

Teacher targets, raw views, trajectories, labels, masks, geometry, split
registries, and exported tokens must round-trip this identity exactly. A join
that succeeds only because two arrays currently share an ordering is invalid.

## Target lineage and versioned continuous-target candidate

### v1 patch target (historical)

The implemented v1 observation screen is retained as historical evidence. Its
former producer path, `src/data/ssm_observation_targets.py`, is now archived; its
provenance schema is `ssm_modality_observation_teacher_v1`. It extracts an array shaped
[sample, token, feature] from ten 2 s positions. EEG features are selected
channel-by-band patch log-power; fNIRS samples inside each patch are flattened
into the feature axis. A full-rank observation-space AR smoother then operates
on the ten positions. This is not a continuous 10 Hz target and does not satisfy
the exploratory interface below. Existing v1 results retain their original
schema and provenance records.

### Observation–source candidate target (exploratory; no reader yet)

The draft target schema is observation_source_candidate_target_v1. A
producer may not publish this schema until the synthetic contract tests,
fold-fitting audit, and source/provenance manifest exist. For a 20 s window,
the continuous axis has T=200 points at 10 Hz; other durations must record T
and the exact timestamps rather than assuming 200.

Using this schema does not select a teacher, observation/source split, token
hierarchy, grammar, or downstream decomposition. An experiment may use a
different versioned interface when its candidate needs different fields.

Each modality m publishes one record with these fields:

| Field | Shape / type | Contract |
| --- | --- | --- |
| canonical identity | exact join-key object | dataset, subject, record/session, task/event, window, and modality branch; no array-order joins |
| time_s | [T] float64 | common record-relative time grid; native anchor and modality-clock offsets remain in provenance |
| observation_values | [T,C_m] float32 | measured target coordinate consumed by the teacher; EEG is channel×band envelope, fNIRS is HbO/HbR model coordinate |
| trajectory_mean | [T,C_m] float32 | posterior/teacher dynamic trajectory $\widetilde O_m(t)$; emitted only where teacher support is valid |
| trajectory_uncertainty | [T,C_m] float32 | non-negative predictive standard deviation, or a declared diagonal/covariance representation; never an unlabelled scalar |
| observation_residual | [T,C_m] float32 | observation_values - trajectory_mean on the observation-residual-valid support; no implicit zero fill |
| coordinate_names / component_roles | [C_m] strings | stable order, channel identity, band or HbO/HbR role |
| masks | named boolean arrays | measured, teacher, uncertainty, observation residual, and any token/lag masks below |
| teacher_mode | enum | native_baseline, self, or privileged_joint; native_baseline records the identity comparison arm and is not a dynamic-teacher claim |
| fit provenance | manifest object | fit fold, parameter/config identity, target/code version, source identity, and label-use=false |

### Residual and noise quantities

Code, serialized outputs, and figure labels use the following names according
to the quantity being computed:

| Name | Definition and interpretation |
| --- | --- |
| `observation_residual` / 观测残差 | `observation_values - trajectory_mean`, in the same observation coordinates; the observed signal minus its estimated clean trajectory |
| `predictive_residual` / 一步预测残差 | Observation minus the one-step prediction before assimilating that observation |
| `state_transition_residual` / 状态转移残差 | `z[t] - F(z[t-1])`, in the declared state coordinates, where `F` is the deterministic transition |
| `process_noise_increment` / 过程噪声增量 | A random increment drawn from the declared process-noise law during generation |

Observation residuals may represent measurement noise or artifacts to the extent
supported by teacher validation. A fitted state transition residual measures
departure from the dynamics; its coordinate space and interpretation stay
distinct from observation residuals. Standardized fields identify their scale,
including whether it is predictive SD, training SD, or process-noise SD.

### Observation coordinates

EEG observation construction is defined at the continuous coordinate level:

$$
e_{c,b}(t)=\log\left(\left|\mathcal H(B_b*x_c)(t)\right|^2+\epsilon\right),
\qquad
O_E(t)=e_{c,b}(t)-\overline e_{c,b}^{baseline}.
$$

This draft schema uses baseline-relative envelope/ERD--ERS; absolute log energy
may be retained as an explicitly named auxiliary coordinate. Channel order and
band order are preserved (for example, six channels by alpha/beta/low-gamma
gives C_E=18). The envelope is formed from the 200 Hz EEG view and then
aligned/downsampled to 10 Hz. A frequency-aware, amplitude-preserving stem is
an architecture choice. None of this paragraph freezes 10 Hz, the filterbank,
the coordinate family, or `C_E=18` as method identity.

This draft schema uses continuous HbO/HbR at 10 Hz after the
declared native transformation and fit-fold model scaling. For a 20 s window,
the unified model coordinate is [B,200,2] (or [T,2] for one record); this is
not a claim about the native raw array shape or unit. The teacher is applied
to the [T,2] trajectory first; only then may a tokenizer create
patches, fine tokens, coarse meta-tokens, or lag endpoints. A target producer
must not flatten each 2 s patch and call the resulting feature sequence
continuous.

### Teacher modes and Croce provenance

The self teacher fits each modality using only that modality and no task
labels. The privileged_joint candidate may fit aligned EEG and fNIRS together
within the fit partition and then emit modality-specific slices, but it is an
offline training/ablation target and is never an inference input. The accepted
adaptive Croce/Balloon implementation and its E0 development-supervision
decision are linked from [METHOD_RATIONALE](METHOD_RATIONALE.md); its later
population-frozen R1-P physical qualification failed. The legacy
croce_validation particle-filter lane is an independent, inconclusive audit
track, not evidence that the v2 target exists.

Croce parameter bounds, candidate version, gauge/sign convention, state
dimension, and output coordinate must be recorded in the fit manifest. A
Croce-derived sidecar is not allowed to replace the measured fNIRS branch or
to be described as ground truth, a causal estimator, or a unique physical
parameterization.

### Continuous-target masks

The named masks are distinct even when their arrays happen to be equal:

| Mask | Shape | Meaning |
| --- | --- | --- |
| observation_valid_mask | [T,C_m] | real recorded support after alignment; derived from the measured branch |
| teacher_valid_mask | [T,C_m] | the frozen teacher emitted a finite trajectory at this point |
| uncertainty_valid_mask | [T,C_m] | uncertainty is finite, non-negative, and calibrated under the declared convention |
| trajectory_valid_mask | [T,C_m] | observation/teacher target can be used for state loss; typically teacher_valid_mask intersected with observation support |
| observation_residual_valid_mask | [T,C_m] | observation_valid_mask intersected with trajectory and uncertainty support |
| token_valid_mask | [N_token] | support after an explicitly declared aggregation from the continuous target; no padding is observed data |
| endpoint_aligned_lag_mask | [N_source,N_target] | source endpoint t and target endpoint t+tau are both valid under the declared lag; same-position shortcut masks are forbidden |
| causal_valid_mask | [T] | only required for a strict-cutoff/future estimand; a full-window offline teacher cannot be labelled causal |

Losses and metrics consume the mask belonging to their tensor. Missing,
unsupported, padded, and zero values are never interchangeable. Observation
residual is undefined where either observation or trajectory support is absent;
it is not silently replaced by zero.

### Fit-fold rules

For every outer fold, fit only on the authorized fit-parameter partition:
channel selection, EEG envelope/scaling parameters, fNIRS model-coordinate
normalizers, baseline templates if learned, teacher dynamics and Q/R/H/A (or
Croce parameters), uncertainty calibration, target projections, fine/coarse
aggregation, codebooks, and grammar parameters. Record the fold identifier,
subject/record inventory, source and software versions, and parameter/config IDs.
Apply the frozen objects to validation and held-out rows without refitting.
Task labels are excluded from teacher fitting; if a downstream private adapter
uses labels, that use is recorded separately and does not alter the
label-blind state vocabulary. Subject, trial, and record dependencies cannot
cross a fold.

For a main coupling claim, the endpoint-aligned estimand, tested increment,
baseline, proper-score endpoint, and null operators are the frozen evidence
kernel. Their task-specific choices, along with the selected target, estimator,
split, thresholds, and stopping rule, are preregistered before held-out access.
The exact grammar network may change before preregistration. A learned
grammar/map may be selected on fit/selection data, but held-out proper-score
increments and the declared null comparisons are the evidence surface.

## Signal branches

The loader distinguishes measurement provenance from model coordinates:

- `raw_*`: dataset-native/released measured arrays and their source metadata;
- `homer2_aligned_fnirs`: a consistent HbO/HbR modeling branch with explicit
  component and transform provenance;
- `simultaneous_eeg_eog_clean_v1`: 28 scalp EEG channels with HEOG/VEOG used
  as auxiliary detection inputs, never as model channels;
- `single_trial_line_clean_v4`: the admitted Single-Trial line-clean branch;
- teacher/trajectory sidecars: privileged or derived targets joined to a
  measured raw view, never substituted for that view.

The Single-Trial v2/v3 artifact-removal branches are historical. The v4
decision removed cached artifact detections as a validity authority. New runs
use real recorded support from `valid_mask`; artifact annotations may be
reported as diagnostics but cannot silently zero samples.

## fNIRS measurement coordinate

Native sources are not falsely described as one physical unit:

- Single-Trial enters from released optical intensities and requires the
  recorded optical-density/MBLL transformation to form HbO/HbR.
- REFED, Visual, and Simultaneous enter from released chromophore exports.
- Subsequent robust centering/scaling creates a dimensionless model
  coordinate and must retain the native source/transform record.

HbO and HbR roles remain explicit. A linear model-space normalization does not
make the upstream physical measurements identical. High-wavelength-only
Croce caches are historical derived supervision and are not the default
measured fNIRS input.

For every continuous target and every derived cache, retain this native fNIRS
provenance tuple before any fold scaling:

| Provenance field | Required value |
| --- | --- |
| source family | Single-Trial optical intensity, or released chromophore export for REFED/Visual/Simultaneous |
| source path/record | stable dataset-native identifier and source location |
| native sampling rate and units | recorded rate and unit (for example optical intensity, optical density, concentration, or dimensionless export); never inferred from the canonical rate |
| transformation contract | optical-density/MBLL or released-HbO/HbR lineage, component roles, filter/resampling steps, and software/schema version |
| model coordinate | fold-fitted centering/scaling and the resulting coordinate name; this does not erase the native source |
| channel identity | optode/channel label, HbO/HbR role, geometry version, and missingness |

An fNIRS teacher may consume the fold-fitted model coordinate, but its manifest
must point back to this native tuple. Optical highWL/lowWL Croce values and
HbO/HbR concentration coordinates cannot be concatenated or called one unit
without an explicit transform and provenance record.

## Time and event alignment

- Window timestamps are expressed in a common record-relative coordinate with
  the dataset-native anchor preserved.
- EEG and fNIRS support must overlap the requested window; missing support is
  not filled and called observed data.
- DSR formal labels come from released EEG codes 16/32. The current event
  registry retains 8,980 Go/No-go windows from 25 admitted subjects and
  excludes VP005 for continuous clock drift.
- Visual timing follows the documented appearance-to-disappearance semantics;
  every-third-row heuristics are forbidden.
- REFED continuous targets use a `[2,T]` valence/arousal sequence and a
  coordinate-wise `target_valid_mask`. A padded value is not a measured
  target.

Offline full-window analysis and strict-cutoff future prediction are different
estimands. A full-window encoder may not be described as causal or prospective.

## Masks

Keep these concepts separate:

| Mask | Meaning |
| --- | --- |
| `valid_mask` | actual recorded signal support |
| channel/component mask | measured channel or HbO/HbR pair availability |
| `target_valid_mask` | observed target support |
| padding mask | batching or fixed-length padding |
| artifact/QC annotation | diagnostic metadata unless a frozen protocol explicitly promotes it |

Zero, missing, censored, excluded, and padded values are not interchangeable.
Every loss and metric must consume the mask that belongs to its tensor.

`cross masking` is not a named mask or frozen mechanism in this contract. It
cannot be introduced by reusing zero, missingness, censoring, exclusion, or
padding. Before the term is used as an architecture component, a versioned
information-intervention contract must define what is intervened on and how;
until then it remains undefined and unfrozen.

## Geometry

Channel labels and geometry sidecars are versioned inputs. Template/projection
coordinates can support adjacency and qualitative spatial structure, but they
cannot support exact distance or co-registration claims. Geometry missingness
must remain visible; copying or mirroring a channel to fill a model input is
not allowed in the primary shared benchmark.

## Splits and protected data

- Fit normalizers, adapters, target scalers, hyperparameters, and model
  selection only on the partition authorized by the owning protocol.
- Group by subject and by any record/trial/video dependency that could cross a
  split.
- Keep sample-random, within-subject, and strict cross-subject protocols
  separately labeled.
- R-series subjects 24–29 remain closed.
- Each comparison protocol controls its own protected boundary. Completed
  STA-Net and historical EFRM v1 evaluations do not authorize EFRM LODO v2;
  the v2 protected folds remain closed until its explicit unlock path passes.

The presence of an index file is not permission to dereference its protected
arrays.

## Cache contract

Every derived cache records:

- schema and branch version;
- source paths/identifiers;
- transformation and software identity;
- sample/join-key inventory;
- shapes, sampling rates, units/coordinates, channel roles, and masks;
- success, exclusion, and failure counts;
- creation time and atomic completion marker.

Any cache using the exploratory continuous-target interface must additionally
record its schema, teacher_mode, time grid, trajectory/uncertainty/
observation-residual field versions, every named mask, fit-fold and
parameter/config IDs, native fNIRS provenance tuple, and whether the target was
self or privileged joint. A cache without these fields remains a v1/legacy
sidecar and cannot be joined under this candidate interface.

Raw data is immutable. Rebuildable caches may be removed after their manifest,
summary, and retained-result status are recorded. Never clean
`data/cache/physiology_semantic_clean_v1/` while the active EFRM protocol is
using it.

## Current audit state

The post-DSR unified audit traversed all 22,952 then-admitted windows and
confirmed finite loading and stable Simultaneous channel signatures. It also
preserved explicit warnings and blockers rather than turning a successful
loader pass into scientific validation. The exact counts remain a dated cache
snapshot; formal protocols must record a fresh inventory and versioned identities.

The active data layer is ready for the implemented STA-Net and EFRM adapters.
That readiness establishes a software/data contract only. It does not qualify
a physical teacher, authorize SD-SVQ/VQ experiments, or validate physiological
coupling.

## Validation entrypoints

Representative checks:

```bash
.venv/bin/python -m pytest -q \
  tests/test_dataset_registry.py \
  tests/test_unified_physiology.py \
  tests/test_event_alignment.py \
  tests/test_fnirs_standardization.py \
  tests/test_channel_geometry.py \
  tests/test_channel_adjacency.py
```

The full real-data audit and visualization commands remain documented in the
dated audit. Their generated reports are evidence artifacts, not an additional
source of data-contract authority.
