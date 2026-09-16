# EEG–fNIRS data contract

_Active measured-data, alignment, mask, and cache rules; v1 historical target
boundary plus an exploratory continuous-target interface, 2026-08-22_

This document is the active data entrypoint. Dataset-native formats, tasks,
units, and original-document locations remain in the reference catalog
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md); the full dated
implementation audit remains in
[`physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md`](physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md).

## Version boundary

The v1 measured-data loader/cache remains frozen historical evidence. The
2026-09-16 timing repair below versions its reader and producers as v2; it does
not relabel v1 artifacts, change splits/protected boundaries, or authorize a
measured cache rebuild. The
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


### Timing repair v2 — 2026-09-16

The timing implementation owner is `src/data/event_alignment.py`; both the
unified reader and legacy Simultaneous multimodal reader use its pairing and
admission rules. The v2 changes are:

- Physical drift is fitted on actual EEG milliseconds **within each offset
  segment**, before checking dispersion. Global slopes across concatenated
  sessions remain diagnostics and are explicitly marked as including jumps.
- A single missing marker is matched with label consistency and original
  source indices. Ambiguous pairing, nonfinite/nonmonotonic times, label
  disagreement, and larger count differences are rejected rather than truncated.
- Events retain conservative support on each modality clock. Without exact
  append boundaries, the interval between anchors surrounding an offset jump
  is unverified; windows entering it are excluded, including pre-event offsets.
  This can reduce the admitted inventory. It is not a reconstruction of the
  unpublished session boundary, and does not change full-record filtering into
  a causal preprocessing method.
- Returned windows record actual sample-grid start indices/times and rounding
  errors as well as requested event anchors.
- Visual fNIRS markers and signals use the same native CSV `Time` coordinate.
  The v2 signal producer verifies Oxy/Deoxy clocks and marks, retains native
  samples/times, and interpolates onto the regular grid **before** filtering.
  Invalid rows, nonmonotonic clocks and gaps over 1.5 nominal sample periods
  are rejected; no row deletion or silent Oxy/Deoxy truncation is permitted.
  Event labels are joined using the original fNIRS marker index after skips.
- REFED retains the published 1 Hz annotation rate rather than stretching it
  to the rounded fNIRS duration. EEG/fNIRS duration disagreement larger than
  one native fNIRS sample excludes the segment. Hardware offset and jitter are
  unknown (`null`), with the publisher's shared origin and label first-sample
  phase recorded as assumptions. This repair cannot recover absent triggers.

The earlier counts below describe the retained **v1** inventory, not a v2
inventory. No measured v2 cache has been built as part of this repair.

- Window timestamps are expressed in a common record-relative coordinate with
  the dataset-native anchor preserved.
- EEG and fNIRS support must overlap the requested window; missing support is
  not filled and called observed data.
- DSR formal labels come from released EEG codes 16/32. The retained v1 event
  registry records 8,980 Go/No-go windows from 25 admitted subjects and
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


The cache schema owner is `src/data/clean_physiology_cache.py`.
`clean_eeg_fnirs_cache_v1`, including the retained
`data/cache/physiology_semantic_clean_v1/` and its v1 event index, is
**deprecated for current consumption**. Its files remain unchanged as
historical evidence. The current reader rejects v1/missing schemas before
reading event payloads or signal arrays; there is no automatic rebuild or
silent fallback to the old cache. Historical sealed replay must use its
retained historical implementation, not reinterpret old arrays as v2.

New producers use `clean_eeg_fnirs_cache_v2` and
`physiology_event_alignment_v2`; the window and REFED sequence contracts are
`unified_physiology_window_v2` and `refed_continuous_va_sequence_v2`.
The timing-repair namespace is `data/cache/physiology_semantic_clean_v2/`;
the new measurement signal producer defaults to `data/cache/physiology_semantic_clean_v3/`.
Its event-index output must explicitly use that same cache root.
`--overwrite` cannot upgrade an existing v1 cache/index in place. Existing
configured v1 paths now fail explicitly until a separately requested versioned
rebuild and configuration migration; updating a schema string is not a rebuild.

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

## Retained v1 audit state

The post-DSR unified audit traversed all 22,952 then-admitted windows and
confirmed finite loading and stable Simultaneous channel signatures. It also
preserved explicit warnings and blockers rather than turning a successful
loader pass into scientific validation. The exact counts remain a dated cache
snapshot; formal protocols must record a fresh inventory and versioned identities.

The historical v1 audit found its data layer ready for the then-implemented
STA-Net and EFRM adapters. That historical readiness establishes a software/data contract only. It does not qualify
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

## Default measurement-cache migration — 2026-09-16

The active default is now `data/cache/physiology_semantic_clean_v4/`, with
`physiology_measurement_alignment_v3`, the existing v2 cache container/event
identity, and `measurement_npy_record_v1` storage. EEG and paired HbO/HbR are
produced once as float64 `.npy` arrays; the reader memory-maps complete records
and copies only the requested windows. It does not re-run EEG cleaning during
training. Native arrays remain in their original datasets; redundant optical,
MAD-normalized and absorbance branches are not duplicated in this cache.

`UnifiedPhysiologyWindowDataset`, REFED sequences, local views and their configured
factory default to float64 measurement coordinates. The user explicitly removed
cross-dataset amplitude normalization from the generic data interface on
2026-09-16: no automatic population scaler or per-channel MAD is applied. SSM
and tokenizer consumers own their model-specific, training-fitted transformations
and computation dtype. The paired adapter remains an optional library component,
not an automatic loader stage. Explicit `legacy_robust` remains for historical
interfaces; frozen experiment configurations and result identities are not migrated.

The user authorized this cache migration, cleanup of superseded rebuildable
caches, and offline full-record two-sided filtering on 2026-09-16. This accepts
its temporal-context limitation for this data preparation, not a causal claim or
an evaluation unlock. Single-Trial subjects 24–29 remain unread; its public
cache inventory covers 01–23. Other datasets use their published records.
No SSM fit or tokenizer training is launched by this migration.

Signal, event and geometry builders share the v4 root. Build completion is
atomic at the signal manifest; readers reject an incomplete build or missing
v2 event index. Retired general v1 cache arrays may be removed under this
explicit cleanup, retaining their manifests/index/geometry and removal inventory
in the migration evidence linked from `experiments/RESULTS_INDEX.md`. This
supersedes the earlier blanket v1 file-retention wording below for rebuildable
arrays only. Frozen SSM native inputs and completed campaign evidence stay in place.

## 测量坐标实现边界（2026-09-16）

新处理身份为 `physiology_measurement_alignment_v3`；缓存容器结构仍为
`clean_eeg_fnirs_cache_v2`，二者不可混用。旧缓存不得逆标准化或原地重标成新版本。
原始采集单位及未核实项仍由 `DATASETS_DESCRIPTION.md` 持有。

- `UnifiedPhysiologyWindowDataset(output_coordinate="measurement")` 只接受新 producer，
  输出 float64 测量坐标及单位证据、参考、几何、色团角色、真实逐通道支持。
  它不做逐通道 MAD；当前默认切换见上一节；历史调用可显式使用 `legacy_robust`。连续 record 清理为非因果操作，
  不能用于含有隔离 trial 的训练边界；此类面板必须先按身份裁窗再处理。
- 光强、OD、已发布色团和 Abs 的入口阶段分开；已知色团单位只换算一次。
  新光学路径将 MBLL 放在带通前，与公开的线性算子次序一致；旧近似系数仍只产生
  相对 Hb。只有显式提供消光系数单位、来源、光程和对数底才输出物理浓度。
- `measurement_baseline` 只用声明的真实共同支持，支持不足返回明确状态；
  事件前窗口不自动等同静息。`physiology_paired_measurement_adapter_v2` 要求显式
  baseline 和训练记录身份，HbO/HbR 共享正尺度，可冻结、序列化和逆变换。
  该 adapter 为消费者可选组件，不在通用 loader/factory 内自动应用。
- 有证据的 EEG 电位先换成 µV，SSM 特征为 `log(P / 1 µV²)`，物理功率下限
  `1e-12 µV²`；未核定电位单位不得进入此物理功率分支。
- 原生缺失在非线性前施加；插值不增加真实支持。双向 IIR 的严格有效支持按整个
  滤波依赖范围保守传播；设备运动/异常注记与有效性 mask 分开。
- `channel_valid_mask` 是测量分支消费者必须使用的逐通道支持；单独的时间窗口 mask
  不能替代它。没有 teacher 资格时不产生 teacher 目标，现有 tokenizer 未由此启动训练。
  现有 factory/local-view 可显式传入 `output_coordinate: measurement`，保留 float64、
  选择后的单位/几何/通道支持，并由全部所选通道的真实支持构造 patch mask；该分支
  拒绝旧 teacher sidecar。各 SSM/tokenizer 消费者自行定义训练组变换及计算 dtype，
  不可把此测量接口直接当作已验收的新训练配置。

限定 SSM 面板复用现有 cache builder 的 `--ssm-training-config`：仅保存原生 float64
光强和 v2 事件身份，原生存储与连续记录清理缓存分开。唯一信号读边界
`load_training_subject` 在读取前核验原训练身份、允许的对齐类型和完整窗口支持，
再裁取训练窗口。新配置及执行流程见 `experiments/README.md`；运行状态和结果只由
registry 及对应 run 持有。

## 跨数据集训练前一致性复核（2026-09-15）

_以下是当日审查与设计记录；当前新增实现边界见上一节，历史结论不追改。_

本节重申统一化原则，区分源码现状和建议；不修改上文冻结的 loader、cache、split，
也不建立新实验状态。采集设备、原始参考、单位证据和原作者说明由
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md#采集方式与单位证据复核2026-09-15)
统一持有。本次范围是文档/源码审查与合成软件检查，没有新的实测数据训练或总体统计。

**目标：在进入模型之前消除可确认的单位、时钟、排列和数值尺度差异，保留真实的
脑区覆盖、任务动力学及模态关系。数值分布相近只证明数值可比较，不证明物理标定一致。**

### 1. 当前通用训练入口已经统一了什么

当前入口为 registry/factory → `UnifiedPhysiologyWindowDataset`；fNIRS 来自
`build_clean_eeg_fnirs_cache.py` 所建的 `homer2_aligned_fnirs`，EEG 来自原始
record 或已声明的清理缓存。实现 owner 是
[`unified_physiology.py`](../src/data/unified_physiology.py)、
[`homer2_preprocessing.py`](../src/data/homer2_preprocessing.py)。

| 环节 | 源码现状 | 一致性的边界 |
| --- | --- | --- |
| 时间/频带 | EEG 200 Hz、1–45 Hz；fNIRS 10 Hz、0.01–0.2 Hz；连续 record 处理后按事件裁窗，polyphase 重采样 | 两模态共享秒级时间坐标，各自保留采样率；采样点数一致不能替代事件/时钟对齐 |
| fNIRS 成分 | Single-Trial 经 OD、TDDR-like 导数抑制、带通、近似 MBLL；其余从已发布 HbO/HbR 起步，做后转换处理 | `HOMER2-aligned` 是可用步骤的近似对齐，不是完整 HOMER2 复现，也没有让原设备与光程相同 |
| 数值尺度 | 两模态均在整段 record 上逐通道 median/MAD；退化时用 IQR/std，再退化用 1 | 输出 `robust_standard_deviation`，消除常量偏置和增益；不会保留跨通道/色团的幅度比 |
| EEG 参考/伪迹 | 保留原参考；Single-Trial v4 与 Simultaneous EOG-only 分支不同；Visual 使用连续 EDF；REFED 使用发布 MAT | 相同 EEG 通道名仍可能表示不同参考下的电位差；相同频带不意味着相同伪迹处理 |
| 身份/空间/缺失 | 已有 join key、channel names、component roles、geometry 和 window support mask | 模板坐标不等于个体定位；插值后的有限值也不等于真实测量 |

缓存字段还需按 producer 解读：`native_input_fnirs` 保留原输入数值；
`raw_native_fnirs` 实际已经经过 `standardize_fnirs_record` 的去趋势/缩放。
默认 loader 读取的是另一条 `homer2_aligned_fnirs` 分支，不可把这些数组重复串联处理，
也不可把 `raw_native_fnirs` 的名称当作“未处理原始数据”的证据。

**SSM 有另一个明确的消费入口。** 当前 Step5/overnight 从限定原训练 trial 的
native EEG 和 `native_input_fnirs` 构造特征，不经过此处的 record/channel MAD。
它已有折内 PCA、HbO/HbR 共同尺度和时序噪声传播。下面第 2 节的通用 loader
问题不能直接当作 SSM 失败归因；SSM 的现状与修订见第 5 节。两者应共享测量
语义与来源合同，按各自观测量派生输入，不能把同一标准化数组强行用于两个模型。

### 2. 本次发现的实质差距

1. **单位证据还没有全部闭合。** REFED EEG 的 `V` 是当前代码常量；Visual/REFED
   色团导出缺少明确物理单位。不得根据幅值猜测 µV、µmol/L 或 mmol·mm。
   Single-Trial optical V 与 EEG V 属于不同测量量。
2. **Single-Trial MBLL 只能提供带假设的相对坐标。** 当前函数使用仓库近似消光系数表、
   默认源探距 3 cm、路径因子 6，并以 `-ln(I/I0)` 构造 OD。系数来源/单位、对数底、
   波长依赖光程与输出单位尚不足以支持绝对浓度解释。只在元数据中写下默认值不能
   完成标定；应以原转换软件和已知浓度合成正例核对。参考
   [MNE 的 MBLL 接口](https://mne.tools/stable/generated/mne.preprocessing.nirs.beer_lambert_law.html)。
3. **当前 normalization 的作用域是 record，不是 fit fold。**
   `_robust_standardize` 每次从待处理 record 估计位置/尺度；它保持重叠 crop 的数值一致，
   但不是训练折拟合后冻结的 scaler。完整独立测试 record 的无标签归一化不自动构成
   跨被试标签泄漏；它属于使用该 record 统计量的预处理。若任务是严格未来预测、
   同记录时间切分，或遮挡重建，则整段统计量、双向滤波和插值可能使用不可见区间。
   上文 fit-fold/strict-cutoff 合同不能仅靠现有 loader 名称宣称已实现。
4. **逐通道标准化改变生理幅度关系。** 对同相合成对 `(2 sin(t), 0.5 sin(t))`，
   当前函数将峰峰幅度比从 **4:1 变为 1:1**；现有
   [`PhysiologyMeasurementAdapter`](../src/data/physiology_measurement_adapter.py)
   使用一个共享尺度时保留 **4:1**。这是可复现的数值性质，不是实测效益。
   该 adapter 的 `fit` 已能复用，但默认 loader 未调用它；默认 baseline 仍取全 record，
且 `fit` 由调用者提供训练记录，并不自行验证 split。
5. **有限值检查不能代替测量支持检查。** 当前 `_interpolate_nonfinite` 会把全 NaN
   通道置零，window mask 主要表示裁窗是否越出 record；原始逐通道/逐时刻缺测需要
   保留并传播，不能把补值用于无条件 target loss。这与伪迹标记是否具有排除权限是
   两个问题，不需要重新赋予 artifact cache 有效性权力。
6. **统一输入还要与 checkpoint 的训练分布区分。** 当前无量纲 MAD 输出不能直接
   宣称等价于外部模型的 µV/100、P95 或 z-score 输入。共享 benchmark 可固定同一
   数据支持域；原方法复现仍须记录它自己的 reference、filter 和 scaler 差异。

### 3. 开源 foundation model 如何处理异质性

下表依据原论文与作者仓库，核查日期为 2026-09-15。描述的是各方法的训练方案，
不是本项目已停止对比 campaign 的得分，也不是这些方法已验证 EEG–fNIRS 物理一致性的证据。

| 方法与来源 | 进入模型前的处理 | 模型内处理/训练目标 | 本项目可以借鉴的部分与限制 |
| --- | --- | --- | --- |
| LaBraM：[论文 §3.2](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)，[作者代码](https://github.com/935963004/LaBraM) | 0.1–75 Hz，50 Hz notch，200 Hz；以 0.1 mV（100 µV）为单位做固定缩放 | 单通道时间 patch、通道/时间 embedding；先训练频谱量化 tokenizer，再预测被遮挡 token；频谱 target 还另做 sample z-score | 固定物理缩放保留输入幅度关系；前提是 EEG 单位已查明。频谱 target 的归一化不能混称为原始波形归一化 |
| CBraMod：[论文 §2](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbbd6d915cb90be21c1254a82d45cedd-Paper-Conference.pdf)，[作者代码](https://github.com/wjq-learning/CBraMod) | TUEG 预训练：0.3–75 Hz、60 Hz notch、200 Hz、30 s 片段；剔除任一点绝对幅值超过 100 µV 的片段，再除以 100 µV | 时域/频域 patch 表征，空间/时间注意力分开，条件位置编码，masked waveform reconstruction | 物理尺度和预训练质量筛选减少输入负担；阈值针对其语料，不能直接用于无量纲 fNIRS 或照搬到本项目的 artifact mask |
| BIOT：[论文 §2.1](https://arxiv.org/pdf/2305.10351)，[作者代码](https://github.com/ycq091044/BIOT) | 统一采样率；每个通道除以绝对幅值第 95 百分位 | 每通道独立分段、频谱 token；通道/相对时间 embedding，缺失片段可不生成 token；跨数据集预训练 | 对设备增益较稳健，变通道/变长度无需伪造通道；逐通道分位数缩放仍丢失幅度比例。论文的重采样描述不代替本项目抗混叠要求 |
| REVE：[论文 §3.1](https://proceedings.neurips.cc/paper_files/paper/2025/file/20a917f77773ac0fa8bea2bdd6606b66-Paper-Conference.pdf)，[作者预处理说明](https://github.com/elouayas/reve_eeg/blob/main/preprocessing/README.md) | 92 数据集、约 6 万小时；200 Hz、0.5–99.5 Hz；使用 recording-session 统计量的 z-score，超过 15 SD 截幅；用已知位置或标准标签推得位置 | 3D 空间坐标加时间的位置编码、空间位置扰动、时空 block masking 与波形重建 | 可借鉴显式几何和按记录稳定统计；不能把论文的 session normalization 等同于本项目的 train-only 规则。位置编码也不自动消除参考差异 |
| NormWear：[论文 §2.2–2.3](https://arxiv.org/html/2412.09758v1)，[作者预处理源码（固定版本）](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/blob/07517fcb13def8c89cb586128359cec02f86ec8d/modules/signal_preprocess.py) | 所查 helper 做差分离群点处理、插值/重采样、去趋势、平滑、平均绝对幅值缩放；论文将原序列及一/二阶差分转为 CWT scalogram | 通道级 token 与通道间注意力、masked reconstruction，适配多种 wearable 信号 | 可借鉴按模态选时频尺度；原训练模态不包含 fNIRS。helper 中滤波调用被注释，不能仅凭函数/参数名声称完整带通已执行；固定 scale 索引也不等于相同 Hz |

这些方案共同支持“先做可解释的基础预处理，再用位置/通道信息容纳剩余差异”。
它们没有证明必须将所有数据集做成完全相同的幅度分布，也没有替我们解决光学单位、
光程或血红蛋白成分混淆。对本项目，优先考虑单位与参考审计、单次尺度转换、真实 mask，
而不是先增加新的域适配网络。

### 4. 建议的下一版输入方案（尚未实现）

以下是下一版实现的候选规范；改变冻结行为时另立版本，并复用现有 data owner，
不原地改写旧缓存或已完成实验。其核心顺序是：

```text
原始 record + 单位/参考/设备/时钟证据
  → 物理可解释转换 + 真实测量支持 mask
  → 连续信号清理、重参考、抗混叠重采样
  → 在声明的时间支持上定 baseline，以训练分区拟合尺度
  → 一次缩放，保留幅度/转换 sidecar
  → 按真实秒数裁窗，附 channel/geometry/component mask
  → 模态各自的 patch 或时频输入
```

**EEG：先统一电位单位与参考，再决定缩放。** 已核实的 V、mV、µV 转为 µV
（分别乘 10^6、10^3、1）。缺失或来源不明的单位记录为未知，禁止根据幅值自动赋单位。
下一版可将有效 scalp channels 的 common-average reference 作为候选；须排除 EOG、
触发和非 scalp 辅助通道，保留原参考与重参考通道集合，并在逐通道非等比例缩放之前完成。
不同电极覆盖下的平均参考仍不同，不承诺完全等价；无法可靠重参考的记录显式保留原参考。
EEG 的 200 Hz/1–45 Hz 可作为现有基线，是否保留更低频 ERP 或更高频信号由任务支持
决定，不直接把 REVE 的带宽覆盖到旧结果。

**fNIRS：统一成分语义，分级保留单位可信度。** 已知 `mmol/L` 转为
`µmol/L = µM` 乘 1000；若已知 M 则乘 10^6。这表示已发布的浓度变化，并非恢复绝对
血红蛋白浓度。Single-Trial 先核对 OD/MBLL 的完整量纲链，再决定能否赋物理单位；
未核定前仍标为近似相对 HbO/HbR。Visual/REFED 的已导出 HbO/HbR 直接保留色团语义，
单位标为未报告；Abs 单独保留，不能当 HbO 或补成一对 760/850 nm 光强。
未知单位不会阻止相对坐标表征研究，但阻止跨数据集绝对浓度比较。

**数值缩放只做一次，并保留成对幅度。** 建议比较下式与当前逐通道 MAD 基线：

\[
z_{r,c,k}(t)=\frac{y_{r,c,k}(t)-b_{r,c,k}(t)}{s_g^{\mathrm{fit}}},
\qquad k\in\{\mathrm{HbO},\mathrm{HbR}\}.
\]

`y` 是声明单位/相对测量语义下的清理后信号；`b` 是规定支持域内的 baseline；
`g` 是单位和采集/转换规则相容的一组记录。同组 HbO/HbR 使用相同正尺度，尺度仅从
训练分区的真实有效值拟合，并以被试/记录均衡的采样估计，避免长记录独占统计量。
物理单位已知且可比时优先共享尺度；单位未核定的设备导出允许独立训练组尺度，
但不能把原始数值跨单位直接混池。现有 `PhysiologyMeasurementAdapter` 提供基本
fit/transform/inverse 接口，需补足真实 mask、拟合作用域及 baseline 参数的验证。

随后按用户要求完成的
[实测可视化与缩放诊断报告](../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/REPORT.md)
确认了当前末级逐通道缩放的 Hb 比例改变，也记录了简单共同训练尺度候选的严重尾部。
因此共同尺度只作为保留幅度关系的接口约束；其数值适用性还须结合测量质量、噪声和
完整分母验证，不能直接替换 scaler 后宣称可训练或 SSM 已改善。

对于完全未见且单位未知的新设备组，不能从测试集合重新拟合组尺度再宣称严格 zero-shot。
固定已有变换时应报告为尺度未知的域外输入；如采用独立无标签校准段，必须另列
calibrated transfer 设定。对外部 checkpoint 则使用其原生输入缩放；需要时从保留的
清理后测量坐标派生，避免在 MAD 输出上再除 100 来冒充 µV/100。

**baseline、漂移与滤波要服从采集范式。** MI/MA、认知 task 的休息段、REFED 的
video baseline 和 Visual 的事件前支持不是同一种采集条件。先核实各自时间锚点与
可用长度；无足够 baseline 时显式记录，不能拿任务期均值冒充静息基线。连续滤波后裁窗
有利于减少短窗边缘伪影，但训练/验证时间切分必须先隔离；严格未来预测使用因果算子。
fNIRS 0.01–0.2 Hz 是当前分析频带，短至 20 s 的窗口不足以稳定估计 0.01 Hz 慢成分，
应使用被允许的长上下文，并报告滤波支持/边缘。漂移消除不能凭整窗线性拟合自动决定，
否则会同时移除长时任务响应。事件同步校正只处理仪器时钟，不抹去生理性血流动力学延迟。

**保留空间覆盖与支持域。** 各模态保留各自通道数和采样率；使用已知 channel names、
geometry 来源/可信度、HbO/HbR role 和真实 mask 批处理。无通道不复制填补，补值不作为
真实 target；filter/resampling 对缺测的影响也需传播。保留 `channel_location/scale`、
baseline、单位证据、参考变换、光学假设和 fit inventory，加入现有 provenance/manifest，
不另建第二套配置。可参考 [SNIRF 的 measurementList/dataUnit 等字段](https://fnirs.github.io/snirf/)，
无需为此次复核迁移文件格式。缩放逆变换只能回到清理后的坐标，不能逆转滤波或运动校正。

### 5. 将状态空间拟合纳入统一化合同（2026-09-15 修订建议）

以下为候选方案，尚未实施或用于新实测拟合。实验次序、固定面板、比较分母和
验收由 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md#数据统一化与-ssm-拟合的联合修订2026-09-15)
持有；本节只定义数据与模型的接口。已有冻结配置、缓存和 run 不原地修改。

#### 5.1 核实的 SSM 路径与问题归属

实现入口是 [`load_training_subject`](../experiments/evaluate_step5_observation_diagnostic.py)
→ [`preprocess_native_trial / fit_measured_projection`](../experiments/evaluate_step5.py)
→ [`v3_prepare_projection / v3_view`](../experiments/evaluate_ssm_overnight_diagnostics.py)。
它从 200 Hz EEG 提取 1–45 Hz 宽带 log-power，经训练折 PCA 得到一个 EEG 代理量；
发布视图的 10 Hz 双波长信号经 OD、运动处理、近似 MBLL 后得到 HbO/HbR，选择一个训练折
光学位置；最终为 4 Hz 的三个观测量。这个 EEG 代理量不是原始电位，也不是已经标定的
神经驱动真值。当前 30 s 窗口内做处理，不能套用通用 loader 的“连续 record 后裁窗”描述。

| 层面 | 已实现/已观察到的事实 | 修订针对的问题 |
| --- | --- | --- |
| 尺度拟合 | PCA、尺度与通道选择仅用训练折；HbO/HbR 共用正尺度 | 保留这些约束；当前信号 MAD 被匹配到固定参考模型的先验预测 SD，仍属于模型相关的幅度映射，不能称为物理单位标定 |
| 时间观测 | v3 已将 baseline、滤波、重采样及插值用于均值与噪声因子；MBLL 保留色团交叉协方差 | 继续闭合实际特征与算子重组、目标与输入的同源噪声；不能退回滤波后逐点独立噪声 |
| 特征噪声 | 训练 first-difference MAD 与合成模型噪声下限取最大值 | 保存下限前的估计与触发率；当前保留投影中的 EEG 最终噪声均等于合成下限，不能把最终值当作独立测得的传感器噪声 |
| 精度 | native optical 缓存与 Homer 输出有 float32；预线性特征/算子主要为 float64 | 数学上可交换的 MBLL、滤波和基线，在不同舍入路径中不保证满足重组容差；仅提升基线精度未解决全部失败 |
| 模型适配 | 共同观测增益只改 Hb 均值；过程噪声另作单因素诊断 | 历史子集预测改善伴随合成状态恢复退化/不确定，不能用增益或 Q 自动吸收测量链失配 |

依据为 [N1–N7 保留分析](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/analysis_20260910_v2/REPORT.md)、
[v3 保留分析](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/analysis_20260911_v2/REPORT.md)、
[精度审计](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/precision_audit_v1.json)
和同一 run 的 `prepared/subject_*_E0.json` 折内元数据。这些来源继续持有结果；
本节不建立新的实验状态表。尺度映射可能混淆幅度与生理参数是待检验机制，
不是全部物理路径或 MAP 收敛失败的已证实原因。

#### 5.2 一个测量来源，明确三个坐标层

1. **测量坐标：** 保存有证据的 EEG 电位、光强/OD 或已发布的 ΔHbO/ΔHbR，以及
   原始单位、参考、转换假设、时钟和真实支持。未知量纲保留“相对/未知”。
2. **特征坐标：** 非线性特征构造后的 EEG log-power、运动处理后的 OD 或色团特征。
   从现有 producer 中保留 float64 边界；记录特征定义和噪声所在层。SSM 使用此边界，
   foundation model 波形输入则从适合它的测量坐标派生，不从 SSM 三变量反推多通道。
3. **计算坐标：** 给推断器或网络作已知缩放/白化。训练折拟合一次并冻结；同时保存
   变换、模型侧对应映射与输出逆变换。网络可在最终导出时使用 float32，SSM 保留
   float64。已有 float32 缓存转回 float64 不会恢复丢失的精度。

这三个层次复用已有 cache/provenance 和观测 spec，不新增三套 loader 或状态清单。
先在现有 SSM 特征入口闭合合同，再让通用 loader 消费相同的单位/支持信息。

#### 5.3 分开单位换算、计算缩放、观测增益和状态先验

设 `u` 为特征坐标观测，`h_theta(x)` 已映射到同一特征坐标；`A_M` 为指定 mask 下的
已知线性处理，`b_fit` 为另行拟合并冻结的偏置，`D` 为已知可逆计算缩放。要求：

\[
y'=D(A_Mu-b_{fit}),\qquad
h'(x)=D(A_Mh_\theta(x)-b_{fit}),\qquad
L'=DA_ML_0,\quad R'=L'L'^T.
\]

`L_0` 是特征噪声因子，`R_0=L_0L_0^T`；观测雅可比同乘 `D A_M`。
若 baseline 已编入 `A_M`，不再用 `b_fit` 重复扣除。不同采样时钟的均值和噪声
分别使用已声明的映射；不能把 4 Hz 模型插值误当作 10 Hz 原生噪声生成。
仅换观测单位时，潜状态、动力学、过程 Q、初态先验以及 P0/Q0 不变。
潜状态本身换坐标是另一种模型变换，需要同步变更动力学和先验，不混入本次预处理。
相同坐标变换同步作用于预测误差和冻结训练 SD 时，NRMSE 也应不变；不能通过单独
调整某模态的数据幅度或评分分母来定义“缩放收益”。

| 操作 | 应同步改变 | 不应据此声称 |
| --- | --- | --- |
| 在电位/浓度坐标上已知 V→µV、mmol/L→µM，或固定除以常数 | 数据、观测均值/雅可比、观测噪声 SD；协方差按常数平方变换，评分单位也转换 | 单纯换单位改善了潜状态恢复 |
| 训练统计量决定的数值缩放 | 同上；冻结统计量及其训练身份，候选间保持一致 | 把每个 trial 拉到相同幅度即完成设备标定 |
| 未知共同 fNIRS 观测增益 `a_N` | 只改变 HbO/HbR 观测均值及其雅可比；已有代码保持噪声不变 | 增益是单位转换，或改善拟合证明神经血管耦合增强 |
| 过程噪声 STD 倍率 | Q 方差按倍率平方变化；保持数据坐标与观测噪声固定 | 增大 Q 后标准化转移残差变小代表物理失配减少 |

当前 `reference_observation_gauge` 从固定参考 Q、初态协方差与时间跨度计算 SD，
再令 `factor = SD_reference / MAD_training`。第一步保留该旧映射作为对照并记录其
逆映射，检验同步换坐标的等价性；不得只把数据放大后仍调用原来的观测方程。
后续科学候选将**模型无关的测量缩放**与**特征到潜状态的观测 loading**分开持有。
有物理标定时用标定映射；仅有相对单位时，在声明的训练校准组固定一个参考桥，
对新 trial、参数候选和 Q 倍率不重新匹配先验预测 SD，明确其仍是相对坐标研究。
先固定 `a_N=1`；只有独立合成检查支持时才检验训练内的一个共同增益。

不能简单删除旧 factor 后把 µM 数组直接交给原来的 P0/Q0。当前 P0/Q0 是模型
血红蛋白基准，并非每个被试的实测绝对浓度；观测 loading 未标定时，绝对生理参数
可辨识性仍未成立。不同波长的光程/系数误差也不一定能由一个共同增益修复。

#### 5.4 EEG 与 fNIRS 的具体特征修订

**EEG：** 在功率计算之前解析电位单位和参考，已知单位统一为 µV；功率明确为
µV²，再计算无量纲 `log(P/P_ref)`；不能把已经取过 log 的 EEG 特征再乘 10^6 来换电位单位。
`P_ref` 是声明单位的固定参考功率；若采用事件前 log-power 均值扣除，则对应相对
事件前几何平均功率的对数比。功率 floor 也必须带单位。
当前 `log(max(P, 1e-12))` 的常数不能在 V² 与 µV² 输入间原样复用，否则截断所对应的
物理功率相差 10^12；远离 floor 时常量对数偏移可被基线消掉，这不证明 floor 无影响。
当前 `preprocess_native_trial` 不接收电位单位参数，后续需在其输入边界完成单位解析
或传入明确的单位合同。记录 floor/无效功率比例；不使用整窗 z-score 抹平任务期能量变化。

最小修订先固定现有 1–45 Hz、PCA 及其符号规则，仅拆清单位和尺度。PCA 首成分
最大方差、载荷和为正不保证它是正向神经驱动；保留训练载荷、解释方差与空间支持。
后续空间/频带假说单独设计，不在精度修复时同时搜索更好拟合的通道和频带。

**fNIRS：** Single-Trial 在原生光学层做有效强度判断与运动处理，核对 OD 的对数底、
消光系数单位、实际波长、源探距和路径因子。用已知浓度变化合成输入以及明确设置相同
假设的 [MNE MBLL 实现](https://mne.tools/stable/generated/mne.preprocessing.nirs.beer_lambert_law.html)
作独立对照；不能仅因函数名相同就认为输出单位相同。已发布色团数据从其真实导出层
进入，不能重复 MBLL，也不能反推假造原始波长噪声。

同一个光学位置的 HbO/HbR 使用相同正比例缩放，保持相对幅度、极性及
`ΔHbT=ΔHbO+ΔHbR`。基线分别从相同时间支持计算。不分别调到单位方差、不单独
放大 HbR 来降低它的 NRMSE，也不强制 HbO/HbR 反相关；负的浓度变化可以是真实响应。
物理正性检查施加于 SSM 的绝对状态约束，不能把负 ΔHb 直接裁零。

当 `ΔHb=M ΔOD` 时，波长噪声协方差通过 `M R_OD M^T` 传播。v3 已实现独立波长
噪声的特例；下一版保留交叉项，并以训练证据检验波长独立假设。只有色团导出的
Visual/REFED/Simultaneous 在色团特征层估计相应噪声，保留“上游转换已发生”的限制。
REFED 的 HbT 可作同单位下的成分一致性诊断，但不能假设它与 HbO/HbR 是第三份
独立测量而重复计算似然；Abs 继续独立标记语义和单位。

#### 5.5 baseline、时间算子与精度

- 事件前 baseline 是有噪声的观测。令权重 `w` 仅使用允许的可见支持且和为 1，
  则 `C=I-1 w^T` 同时作用于目标、预测均值与噪声。保留由此产生的时间相关及秩损失；
  不把去基线后的首段均值为零当作潜状态/血流确实处于静息的证据。
- 当前第一步仍用 30 s、4 Hz 和 5 s baseline，以隔离修订效果。0.01 Hz 对应约
  100 s 周期，30 s 窗内低频基线与边缘效应需作敏感性检查。将来使用连续长上下文时，
  先由 scope 确定可读支持，再拟合/滤波；不能为了减少边缘效应越过被排除的 trial。
  `sosfiltfilt` 是前后向滤波，不能用于声称严格未来预测的路径，见
  [SciPy 接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)。
- 将非线性特征边界之后的线性次序固定为一个实际 producer 和可重放的算子；从特征、
  MBLL、滤波/重采样、baseline、尺度到噪声因子用 float64。先测定误差来自哪一次舍入，
  再比较与旧 float32 输出的差；不放宽旧阈值来宣布旧重组失败通过。
- 算子导致协方差低秩时在有效子空间做 SVD 白化。保留 rank、舍弃奇异值和支持残差，
  不靠对角 jitter 把本来丢失的 baseline 信息补回来。完整秩的单位变换报告密度的
  Jacobian 修正；低秩时间观测使用相应支持上的密度，不能照搬全维行列式。
- 当前 v3 的 mask 位于非线性特征之后、线性处理之前，只回答“特征缺失”问题。
  原始传感器缺失需在滤波、功率、OD/运动处理之前遮挡并另做干预检查。两种模式显式
  分列；隐藏值扰动不能改变输入特征、训练对象或预测。插值不增加独立测量数。
- 评分目标与输入来自同一有噪声记录时，除 `R_ii`、`R_tt` 外保留 `R_ti`。
  同源条件预测和 clean-map 残差分别评分；“能复现带噪目标”不能替代潜状态恢复。
  当前 v3 已有这套条件预测，修订必须保留，不能只统一均值而丢掉它。

#### 5.6 噪声、空间选择与拟合稳定性的接口

训练 first-difference MAD 是声明特征层的初始噪声估计；平滑、功率、log 和运动处理
后不保证独立同分布。保存 `estimate_before_floor`、floor 的单位/来源、最终值与触发率，
使用现有折内 metadata；当前只保存最终值，不能从保留结果恢复未经截断的估计。
区分传感器/特征噪声、有限精度下限和模型失配，不能用同一个合成数值下限兼任三者。

第一步冻结现有 R 的统计假设，只做一致变换和可审计记录。下一步单独比较在允许的
训练 baseline/特征支持上估计的噪声，用已知噪声合成与训练外残差检查其校准；事件前
静息段也含生理变化，不能自动当作纯传感器噪声。任何新下限都须说明单位和依据，
不因拟合失败临时设值。初始只保留现有波长小矩阵与已知
时间算子，不直接估计整个 360×360 自由协方差。新 R、观测增益与 Q 不联合搜索。

Student-t 的 `scale` 不等于标准差；自由度 `nu>2` 时方差为
`scale² * nu/(nu-2)`。转 Gaussian 或传播协方差时显式转换；保留噪声族标签，
不能用 Gaussian 合成通过替代 Student-t 资格。
噪声白化后的残差要与该算子产生的有效子空间参考比较；轨迹拟合后的残差也受
拟合自由度影响，不能把全时间轴相关为零机械地当作合格阈值。

当前光学位置按训练信号 MAD / 差分 MAD 选择；这是平滑度相关评分，不是皮层来源
或空间对应的保证。最小数值修订保持同一训练选择以实现配对比较。后续再以原几何、
真实测量支持和独立质量指标限定候选脑区；选择不接触验证误差，不自动引入新的空间
网格。同一六状态局部模型只用于空间支持相容的 EEG 代理和 Hb 对；四数据集的脑区
覆盖不相同时，统一单位不能替代空间观测模型。

`MAP_evaluation_budget`、非法试探路径、最终物理状态失败、输入重组失败和依赖失败
分别记录。已知缩放闭合后，检查白化残差、加权雅可比的奇异值/条件数、梯度与终止原因，
将数值预条件与生理参数估计分开；不把统一幅度视为求解器必然收敛的保证，也不为获得
有限输出放宽正性约束。确定性回放另外核对非零 driver 的插值/积分误差，不能全归入 Q。

#### 5.7 每个数据集的接入动作与迁移边界

采集设备、单位证据与采样率仍由 `DATASETS_DESCRIPTION.md` 持有。下一版按其证据
执行以下差异化接入，再汇入上述同一合同：

| 数据集 | 必做接入动作 | SSM 可以使用的解释 |
| --- | --- | --- |
| Single-Trial | 保留双波长正强度/缺失信息；核对完整 MBLL 量纲链，审计原生与派生缓存精度；EEG 单位从原头读取 | 标定闭合前为相对 Hb 特征；先在现有固定开发面板验证合同 |
| Simultaneous | 从已发布 oxy/deoxy 起步，已知 mmol/L 转 µM；保留 EEG 原参考和 EOG 身份；避免重复光学转换 | 已发布浓度变化有单位，不等于已知 P0/Q0 或恢复原生光学噪声 |
| Visual | EEG 按 EDF 的物理单位/增益解码；Oxy/Deoxy 保留导出语义及未知单位；使用其实际事件前支持 | 独立相对单位组；不能因仪器波长已知就假定光强或 µM |
| REFED | 审计 EEG 代码常量与原证据；显式拆开六种信号类型；保留原参考、视频/标签时钟与 baseline 定义 | Hb 与 Abs 分开；单位未闭合前不与已知浓度做绝对幅值混池 |

将 `unit/evidence`、`entry_stage`、`reference/geometry`、`clock/support`、
`feature_definition/dtype`、`baseline/operator`、`fit identities/scale/inverse` 和
`noise layer/factor/floor` 放入既有 provenance 与 projection metadata；补足字段，
不另立 manifest owner。未知新设备若要拟合尺度或 loading，明确声明独立训练/无标签
校准支持；使用测试全集统计量后不能称严格 zero-shot。

### 6. “减轻模型压力”的验收方式

当前研究的首要验收是 SSM 的**输入/观测等价性、合成状态恢复、完整分母的隐藏观测
预测和配对增量**，具体消融由实验设计 owner 规定。已知单位重表达只应改善数值条件，
潜状态与参数推断应保持不变；改变特征或噪声模型后才检验科学效益。
RMS/偏差同时报告清理后测量坐标和冻结训练 SD 归一化值，HbR 低方差显式标记。
不扩大评分分母来美化弱信号，也不按每个候选重新缩放目标。

未来另行开展 foundation-model 训练比较时，可在相同 split、真实支持、架构与预算
下，分别比较冻结 loader、测量/支持修复和成对训练尺度，记录梯度/非有限值、训练效率
及获准的验证指标；它不替代本协议的物理 teacher 资格，也不把下游任务标签用于当前
SSM 选择。若修复改变有效支持，主比较限制在共同真实支持，同时报告支持变化。
数据集/被试均衡采样与 loss 加权另作独立控制，不和处理修订同时改变。

dataset-ID probe 只能诊断残留设备指纹：任务/脑区不匹配也可预测数据集身份。
不能以所有 PSD 相同或 probe 降到随机为成功标准。现有开源方法说明可用已知尺度、
位置和通道信息降低输入差异，但不能据此跳过 SSM 的均值、协方差与状态恢复检查。
