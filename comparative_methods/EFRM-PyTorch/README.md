# EFRM synchronous-data reproduction

> **STATUS: STOPPED（legacy method record）。** EFRM LODO v2, public delivery and its
> later joint-campaign cell evaluation are complete. This workspace is retained for
> evidence and replay only; unexecuted full-finetune/transfer extensions are
> **ABANDONED**, and no launch or protected follow-up here is current.

_Project-level execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Protocol and implementation
details here describe the frozen EFRM evidence layer._

This directory is the isolated project adaptation of EFRM for the four
synchronously acquired EEG-fNIRS datasets in this repository. The untouched
upstream checkout remains in
`../EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/` at revision
`a62bf3d4c092ac3022b6c0bad90ec3993d5a5720`.

The reporting name is **`efrm_sync_200_10_variable_channel_v1`**. It is a
source-faithful retraining of the EFRM objectives, not a claim of numerical
reproduction: the original checkpoint and processed pretraining arrays were
not released, all project data are paired, the sampling rates are changed to
the repository-wide 200/10 Hz contract, and channels are not duplicated.

## Versioned rebenchmark interface

The separately versioned [rebenchmark adapter](rebench_v1.py) implements the
[downstream rebenchmark contract](../../docs/EXPERIMENT_PLAN.md#downstream-rebench-v1).
It reuses the EFRM library architecture and three pretraining objectives, with
explicit 200/10 Hz measurement-v5 panels and recorded-support masks. Its
`create_pretraining_model`, `pretraining_loss` / `pretrain_step`, and checkpoint
save/resume functions are consumed by the common durable worker; historical
launchers and v1 weights are not relabeled or resumed. Downstream creation
requires a newly completed fresh refit excluding the entire target dataset.
REFED exposes actual time-token features and a separate input-support mask for
pointwise linear readout, rather than projecting one pooled vector into a
trajectory. [Synthetic tests](tests/test_rebench_v1.py) exercise this version
without reading retained datasets or runs. The remaining sections document the
frozen historical implementation.

## Frozen v2 performance-testing protocol

The completed v2 experiment followed
[`sources/20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md`](sources/20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md)
and its machine contract
[`sources/lodo_full_target_fivefold_v2.yaml`](sources/lodo_full_target_fivefold_v2.yaml).
The frozen design pretrained four leave-one-dataset-out checkpoints and performed
downstream five-fold evaluation on every eligible subject in the complete
target dataset. Inner validation selects the downstream training epoch, after
which the head is reinitialized and refitted on the complete outer-training
partition before the frozen formal evaluation.

Final paper-table numbers additionally follow the lightweight
[final-number acceptance rules](../../docs/comparisons/METRIC_ACCEPTANCE.md)
and [machine-readable targets](../comparison_metric_targets_v1.yaml). They
judge only the resulting value and do not rewrite the frozen LODO training
estimand. Linear probing remains a representation track; full fine-tuning is a
separate end-to-end track.

The completed source/target dual-protocol v1 remains historical development
evidence. It is not the protocol for future EFRM-versus-mainline ranking and
its checkpoint must not be reused in v2.

The v2 protocol was materialized on 2026-07-27. All four Stage-A
target-excluded selection jobs and all four Stage-B full non-target refits
completed on 2026-08-03. Inspect the retained status with:

```bash
.venv/bin/python comparative_methods/EFRM-PyTorch/run_lodo_pretraining.py status
```

The comparison-aligned historical public downstream phase is implemented by
[`run_downstream_public_v2.py`](run_downstream_public_v2.py). It unions only
the five public manifests, verifies and freezes the checkpoint that excluded
the target dataset, caches full-public paired embeddings, and trains only a
fresh LayerNorm/Dropout/Linear probe. REFED retains its input and target masks
and fits target scaling on the allowed training membership only. Historical
smoke command (do not launch new work):

```bash
.venv/bin/python comparative_methods/EFRM-PyTorch/run_downstream_public_v2.py \
  --config comparative_methods/EFRM-PyTorch/configs/downstream_public_v2.yaml \
  --task motor_imagery --outer-fold 0 --seed 17 --smoke \
  --output-dir comparative_methods/EFRM-PyTorch/runs/formal/efrm_lodo_full_target_fivefold_v2/downstream_public_v2/smoke/motor_imagery_outer0_seed17
```

The non-self-authorizing public matrix completed 105/105 serial jobs with no
failures or retries. Its retained completion and A0-A8 summaries are
[`matrix_completion_summary.json`](evidence/public_development_v2/matrix_completion_summary.json)
and [`summary_final.json`](evidence/alignment_v2/summary_final.json). For
audit-only re-materialization of the same historical candidate definition, use:

```bash
.venv/bin/python comparative_methods/EFRM-PyTorch/build_downstream_public_matrix_v2.py \
  --output comparative_methods/EFRM-PyTorch/runs/formal/efrm_lodo_full_target_fivefold_v2/protocol/downstream_public_job_matrix_candidate_v2.json
```

Neither replay command can open measured evaluation data. Every retained public
result is marked `table_admissible=false`; final cell verdicts come from the tracked
joint-campaign result report.

## Frozen scientific and data contract

### What the source pretraining corpus actually aligns

The paper's nominal 1,247.5 hours are not 1,247.5 hours of synchronized
multimodal data. Its reported composition is 868 hours / 766 participants of
EEG-only data, 364 hours / 123 participants of fNIRS-only data, and only 15.5
hours / 29 participants of paired EEG-fNIRS data. The released pretraining
loop draws all three loader types each step: the EEG-only branch supplies EEG
masked reconstruction, the fNIRS-only branch supplies fNIRS masked
reconstruction, and only the paired branch supplies the symmetric CLIP loss.
Shorter loaders are cycled, so this is objective-level joint training, not a
claim that unpaired recordings were aligned across people or experiments.

Our corpus has no comparable single-modality expansion. Every admitted sample
is a genuinely synchronized pair, and that one pair supplies both masked
reconstruction losses and the CLIP loss. This preserves EFRM's three
objectives while changing the data regime from `mixed_unpaired_plus_paired`
to the explicitly named `synchronized_in_domain_only` track.

### fNIRS entrance

EFRM consumes the `homer2_aligned_fnirs` branch exposed by
`UnifiedPhysiologyWindowDataset`. Every admitted sample has explicit HbO/HbR
component roles and is transformed to a dimensionless, full-record robust
coordinate before cropping. This is an acceptable EFRM input because the
model reconstructs and aligns a consistent two-component coordinate.

The claim boundary is important. The original physical measurements are not
all connected by a purely linear transform: Single-Trial intensity is mapped
through `-log` optical density and MBLL, whereas REFED, Visual, and
Simultaneous enter from released chromophore exports. Record-level centering
and scaling after component construction are affine and preserve each native
measurement provenance; they do not establish identical physical units.

### Sampling and patch geometry

- EEG: 200 Hz, 8-second pretraining window, 50-sample patch (0.25 seconds).
- fNIRS: 10 Hz, 8-second pretraining window, 20-sample patch (2 seconds).
- The temporal token grids remain 32 EEG patches and 4 fNIRS patches, exactly
  matching the physical patch durations and token counts of EFRM at 128/16 Hz.
- All pretraining examples are real synchronized pairs. The same admitted pair
  supplies EEG MAE, fNIRS MAE, and symmetric EFRM/CLIP alignment losses.
- Development tensors are materialized once at a deterministic common-valid
  8-second crop after the public split boundary is resolved. This method-local
  cache contains no protected samples and avoids repeatedly decoding REFED MAT
  containers or Visual EDF files. Batches are grouped by the real measured channel inventory and draw windows
  round-robin across records. Dataset sampling is balanced before selecting
  deterministic epoch-dependent crops. This keeps tensors stackable without
  copying channels while preventing every CLIP negative from being an adjacent
  or overlapping window in the same record. Within-record negatives are still
  measured and reported when they occur.

### Variable-channel policy

All measured channel slots are retained. HbO and HbR are paired by spatial
channel name, and a location is valid only when both components are present
and pass QC. Bad channels and invalid time support are zeroed and represented by masks; they
are excluded from reconstruction loss, self-attention, and global pooling. Batches contain one
record inventory, while sinusoidal position embeddings are generated for the
current channel/time grid. No channel is copied, mirrored, or assigned
fabricated geometry.

This is the primary shared-benchmark variant. An upstream-style channel
duplication experiment may be added only as a separately named sensitivity
analysis and may never replace it.

## Model and loss contract

The default architecture retains the two independent EFRM ViT-base MAEs:
embedding 768, 12 encoder blocks, 12 heads, 512-dimensional decoder, 8 decoder
blocks, 16 decoder heads, and 50% masking. It contains approximately 223M
parameters before downstream heads.

For a synchronized batch `(eeg, fnirs)`:

```text
L = L_EEG_masked_reconstruction
  + L_fNIRS_masked_reconstruction
  + L_symmetric_pair_retrieval
```

The contrastive term preserves the released implementation's fixed `0.1`
similarity multiplier as the primary source-code setting. Raw cosine
similarities are exported separately so this unusual scale cannot obscure the
diagnostic. A conventional learned/divisive temperature is a named ablation,
not the primary EFRM run.

### Alignment failure-warning assessment

The epoch-8 interruption was the first **serious but scope-limited warning**;
it is not the current state. The unchanged public-development run subsequently
completed 84 epochs with best checkpoint epoch 69. Re-evaluation of best and
latest checkpoints still found chance top-1 retrieval in both directions,
positive-vs-negative AUC below `0.5`, negative positive-pair margins, and
near-one-dimensional validation embeddings. Aggregated validation CLIP
evidence likewise did not improve, so the warning is now a completed
public-development finding rather than a planned epoch-20 decision.

The fixed source multiplier makes scalar CLIP loss intrinsically insensitive.
For 32 pairs, all logits are bounded to `[-0.1, 0.1]`: random CE is
`log(32) = 3.4657`, while even the idealized geometric limit with every
positive cosine at `+1` and every negative at `-1` is about `3.2726`.
Retrieval, positive/negative separation, permutation evidence, and embedding
geometry therefore carry more diagnostic weight than the raw loss curve.

This result is recorded as a failure of source-faithful exact-window alignment
on the synchronized public-development track. It does **not** establish that
the datasets contain no EEG-fNIRS relationship: slow, delayed, task-shared
coupling need not make one fNIRS window uniquely retrievable among nearby or
physiologically similar negatives. A learned/divisive-temperature experiment
must start from scratch under a separate ablation name and may never replace
the source-faithful baseline. Full evidence and numerical boundaries are in
[the completed alignment analysis](../../docs/physiology_semantic_tokenizer/analysis/20260724_EFRM_RETRAINING_AND_ALIGNMENT_COMPARISON.md).

## Leakage boundary and evaluation matrix

Development uses the public subject-grouped train/validation manifests already
frozen for STA-Net. A single development checkpoint is pretrained on public
training subjects from all four datasets; validation subjects are used only
for self-supervised stopping and downstream model selection. Reserved test
subjects are never loaded.

The stopped v2 formal path did not pretrain inside every outer fold. Instead,
it trains one checkpoint per target dataset using all subjects from the other
three datasets. The target dataset was excluded in its entirety from
representation pretraining, so the same frozen checkpoint can be reused across
all five downstream folds without making the result transductive.

A method-neutral full-target fold registry is mandatory. EFRM, the project
mainline, STA-Net, and any other directly ranked method must consume identical
eligible subjects, samples, outer test folds, labels, modalities, and
endpoints. A global checkpoint pretrained on the target dataset remains
a `transductive_diagnostic` and cannot enter the inductive primary table.

The downstream matrix matches STA-Net:

| Task | Target | EFRM input support | Reporting name |
| --- | --- | --- | --- |
| Motor imagery | LMI/RMI | paired 8 s | `efrm_sync_classification` |
| Mental arithmetic | MA/BL | paired 8 s | `efrm_sync_classification` |
| Word generation | WG/BL | paired 8 s | `efrm_sync_classification` |
| N-back | 0/2/3-back | paired 8 s | `efrm_sync_classification` |
| DSR | Go/No-go | 2 s EEG plus synchronized 2 s fNIRS context | `efrm_sync_dsr_context_adapter` |
| Visual motivation | RR/RF/FF/FR | paired 8 s | `efrm_sync_classification` |
| REFED | 20-point valence/arousal sequence | paired 20 s | `efrm_sync_regression_adapter` |

The stopped protocol required frozen-backbone linear probing for all seven
tasks; its unexecuted full-fine-tuning extension is abandoned for this legacy
flow.
Classification selects an epoch by inner-validation macro-F1, then refits a
fresh head on all outer-training subjects for that epoch count. REFED follows
the same procedure with masked scaled RMSE. Protected results are opened only
through the frozen protocol's explicit unlock path.

The earlier public-development tuning plan followed the STA-Net policy: seed
42, 12 Optuna trials per task and transfer mode, 2/8/20/40/100-epoch rungs,
and best validation checkpoint through each rung. It remains development
evidence. The stopped v2 protocol used downstream seeds 17, 42, and 73. Seed
results are averaged inside each outer fold before fold aggregation, and seed
dispersion is reported separately.

### Stopped LODO full-target performance protocol

The normative protocol and its machine-readable contract are:

- [`20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md`](sources/20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md)
- [`lodo_full_target_fivefold_v2.yaml`](sources/lodo_full_target_fivefold_v2.yaml)

Four target-excluded checkpoints serve the four datasets. The seven-task
primary matrix is strict cross-subject five-fold frozen-backbone linear
probing on the complete target dataset. Its reporting name is
`efrm_lodo_strict_cross_subject_5fold_v2`. A complete sample-random matrix is
reported separately as
`efrm_lodo_sample_random_5fold_secondary_v2` and cannot support new-subject
claims.

Every fold performs inner epoch selection followed by a fresh full-outer
refit. Primary uncertainty includes fold mean and sample SD, pooled
out-of-fold metrics, separately reported seed dispersion, and a 10,000-draw
subject-cluster bootstrap interval. Direct ranking is allowed only against
methods using the exact shared fold registry.

### Historical completed resource-bounded protocol

The immutable v1 protocol, machine-readable contract, and final results are:

- [`20260725_RESOURCE_BOUNDED_DUAL_PROTOCOL_FREEZE.md`](sources/20260725_RESOURCE_BOUNDED_DUAL_PROTOCOL_FREEZE.md)
- [`resource_bounded_dual_protocol_v1.yaml`](sources/resource_bounded_dual_protocol_v1.yaml)
- [`20260727_RESOURCE_BOUNDED_DUAL_PROTOCOL_RESULTS.md`](sources/20260727_RESOURCE_BOUNDED_DUAL_PROTOCOL_RESULTS.md)

Dataset-level subjects are deterministically divided into a source cohort and
a disjoint target cohort. The source cohort supplies one source-only
pretraining run. The target cohort supplies two independent five-fold
downstream registries: subject-disjoint strict cross-subject folds and direct
sample-random folds that deliberately permit participant and acquisition
dependencies to cross partitions. The required linear-probe grid is 7 tasks ×
2 protocols × 5 outer folds at fixed seed 42. Its primary uncertainty is
sample SD across the five target outer folds (`ddof=1`), not seed SD.

These EFRM aggregates estimate source-to-target transfer and are not directly
ranked against the current full-dataset STA-Net five-fold aggregate. A direct
STA-Net comparison requires a matched rerun on the exact EFRM target cohort
and folds. The existing development checkpoint and any checkpoint exposed to
target samples remain transductive diagnostics only.

The frozen matrix is complete: all 70 linear-probe jobs and protected
evaluations passed exact five-fold coverage checks. The final result report
records fold means, sample SDs, uncertainty intervals, protocol sensitivity,
and the claim boundary. The main finding is weak task-dependent transfer:
sample-random improves six of seven primary endpoints, most clearly for visual
motivation, while DSR macro-F1 is the exception and strict motor imagery,
N-back, visual, and REFED remain near chance or weak in absolute terms.

These aggregates are retained to explain why the source/target allocation was
replaced. They must be labeled
`historical_resource_bounded_source_to_target_transfer`, may not be mixed with
v2 folds, and may not be selected task-by-task according to which protocol
produces the higher score.

### Historical public-development transfer runner（ABANDONED extensions）

The launch examples in this section are retained for historical replay only; do
not start the former transfer queue or its unexecuted full-finetune branch.

`train_downstream.py` executes one public train/validation task without opening
protected indices. It supports `linear_probe` and `full_finetune`, EEG-only,
fNIRS-only, and paired input, pretrained or scratch initialization,
classification and masked REFED regression, resumable checkpoints, validation
predictions, calibration metrics, and per-subject metric rows. The MAE decoders
remain frozen because downstream inference uses only the two encoders.

Both current shared split schemas are accepted: index-addressed
`sta_net_split_registry_v2` and the older subject-only
`sta_net_subject_split_v1`. The runner rejects paths below a `protected`
directory and verifies that every downstream validation subject was held out by
the selected pretraining boundary.

Historically, the detached queue launcher ran task matrices sequentially on each
assigned GPU; this former queue is abandoned and must not be restarted:

```bash
.venv/bin/python comparative_methods/EFRM-PyTorch/launch_downstream.py start \
  --run-id 20260724_efrm_public_transfer_pilot_v1 \
  --pretrained-checkpoint comparative_methods/EFRM-PyTorch/runs/pretraining/\
20260722_efrm_sync_dev_v5/checkpoints/best.pt \
  --split-root comparative_methods/STA-Net-PyTorch/runs/tuning/\
20260722_sta_net_hpo_v2_checkpoint_objective_100ep \
  --gpus 0 1 \
  --transfer-modes linear_probe full_finetune \
  --modalities paired \
  --initializations pretrained
```

Inspect detached workers with:

```bash
.venv/bin/python comparative_methods/EFRM-PyTorch/launch_downstream.py status \
  20260724_efrm_public_transfer_pilot_v1
```

These runs are `public_development_pilot` evidence. They do not replace
target-excluded LODO pretraining, the full-outer refit, frozen selection
rules, scratch/modality controls, or the explicit protected-test unlock
required for formal comparison.

## CLIP alignment evidence and physiological-coupling contrast

Every validation evaluation exports sample IDs, dataset/subject/record/task,
window times, both 768-dimensional embeddings, raw cosine similarity, scaled
logits, and the exact positive-pair mask. The visualization suite generates:

1. an EEG-by-fNIRS similarity matrix with diagonal positives explicitly boxed;
2. positive, all-negative, within-record, and hard-negative distributions;
3. EEG-to-fNIRS and fNIRS-to-EEG positive rank, top-k retrieval, and MRR;
4. a paired embedding projection with a connector for every true pair;
5. a side-by-side contrast panel: EFRM's symmetric window-retrieval matrix
   versus the project model's directional, lag-conditioned EEG-to-fNIRS
   coupling evidence.

The figure caption must state that an EFRM diagonal is defined by acquisition
co-occurrence and does not identify direction, hemodynamic delay, or a
physiological mechanism. Figure source arrays, pair metadata, SVG, 300-DPI PNG,
and metric JSON are retained together.

## Historical execution stages and gates（STOPPED）

1. **Protocol/preflight:** hash upstream, clean cache, event index, geometry,
   task contracts, public splits, channel inventories, eligible synchronized
   duration, and all exclusions.
2. **Correctness smoke:** tiny model forward/backward on real paired samples;
   prove patch counts, positive diagonal, reconstruction masks, deterministic
   crops, and finite gradients.
3. **Architecture smoke:** full ViT-base on one 4090 with bf16, activation
   checkpointing, and automatic microbatch search. Preserve a true 32-pair
   contrastive matrix with two-pass gradient caching when the model must be
   recomputed in smaller chunks. Ordinary gradient accumulation does **not**
   enlarge CLIP's negative set and is therefore not presented as equivalent.
4. **Development pretraining:** up to 100 epochs, AdamW `1e-4`, betas
   `(0.9,0.95)`, weight decay `0.01`, cosine decay, gradient norm 5, early stop
   patience 15 after at least 20 epochs. Save latest/best and resumable sampler
   state after every epoch.
5. **Transfer and visualization:** run seven-task linear-probe/full-finetune
   public development studies and generate validation-only reports.
6. **Formal evaluation:** freeze configs and rerun pretraining per protected
   outer fold before the explicit shared-protocol test unlock.

Historical implementation status (2026-07-23): the independent model, unified
adapter, seven-task heads, public-split boundary, CLIP evidence exporter,
resumable ViT-base pretrainer, and post-hoc analysis tool are implemented.
The first development run reached eight complete epochs and then stopped
midway through epoch 9. Its durable checkpoint is epoch 8; it must not be
reported as a completed 100-epoch or early-stopped run.

### Parameter and memory audit

The synchronized implementation has 221,459,034 trainable parameters:
110,735,922 in the EEG MAE and 110,723,112 in the fNIRS MAE. The official
default source model has 222,780,000 total parameters, of which 221,466,720
are trainable; most of the total-count difference is its fixed positional
tables. FP32 parameters occupy 0.825 GiB. FP32 parameters, gradients, and the
two AdamW moment tensors require approximately 3.30 GiB before activations;
a weights-only checkpoint is approximately 0.825 GiB and a checkpoint with
Adam state approximately 2.48 GiB before serialization overhead.

Measured on an RTX 4090 with a true 32-pair contrastive matrix:

| Recompute chunk | Activation checkpointing | Peak allocated | Peak reserved | Train+validation smoke |
| ---: | --- | ---: | ---: | ---: |
| 8 | on | 4.20 GiB | 4.55 GiB | 35.8 s |
| 32 | on | 4.19 GiB | 4.38 GiB | 29.1 s |
| 32 | off | 20.25 GiB | 20.62 GiB | 20.8 s |
| 16 | off | 11.89 GiB | 12.55 GiB | 21.9 s |
| 8 | off, EEG + worst-case REFED | 9.12 GiB | 9.56 GiB | 28.0 s / 2 train + 1 validation batches |

The historical development default was batch 32, exact two-pass gradient cache,
recompute chunk 8, BF16 autocast, and no activation checkpointing. This left
substantial margin beside the then-active STA-Net jobs without paying the slower
checkpoint-recompute path. Batched attention masks exclude invalid channel/time
tokens as keys and values without falling back to per-sample transformer loops.

## Artifact contract

Runs live only below `runs/<protocol_id>/...` and follow the common comparison
layout: protocol/method/adapter/split manifests, resolved config, environment,
checkpoints, metrics, predictions, `figures/`, `figure_data/`, hashes, status,
and a self-contained summary. Vendor source files are never modified.

The pretraining entry point is `train_pretrain.py`. It writes resumable
latest/best checkpoints, epoch/step JSONL, the resolved config, split-boundary
hashes, CUDA peak memory, and validation-only CLIP evidence. Its two-pass
gradient cache is regression-tested against the full contrastive-batch
gradient before GPU training.

### Historical detached launch and process audit（ABANDONED）

Long training must not be started by invoking `train_pretrain.py` directly
from a Codex PTY, SSH terminal, or other temporary execution cell. Use the
detached supervisor:

```bash
.venv/bin/python \
  comparative_methods/EFRM-PyTorch/launch_pretrain_detached.py \
  --run-id 20260722_efrm_sync_dev_v5 \
  --device cuda:1 \
  --chunk-size 8 \
  --num-workers 0 \
  --resume
```

The launcher creates a new POSIX session, connects stdin to `/dev/null`,
redirects both output streams to a durable file, and returns only after the
supervisor has recorded its state. Closing the invoking terminal therefore
does not deliver its hangup signal to training. It records the supervisor PID,
training PID, session/process-group IDs, exact command, log path, exit code,
termination signal, and terminal timestamps below
`runs/launcher/<run_id>/`. It also changes the run manifest/status to
`completed` or `failed` when the child exits.

Inspect a run without depending on the original terminal:

```bash
.venv/bin/python \
  comparative_methods/EFRM-PyTorch/launch_pretrain_detached.py \
  status 20260722_efrm_sync_dev_v5

tail -F comparative_methods/EFRM-PyTorch/runs/launcher/\
20260722_efrm_sync_dev_v5/logs/*.log
```

The launcher rejects an existing run unless `--resume` is explicit, rejects a
resume without `checkpoints/latest.pt`, and refuses a second live supervisor
for the same run ID. Launcher state is operational metadata and remains under
the ignored `runs/` tree.

### Post-hoc analysis

Generate a read-only public train/validation audit with:

```bash
.venv/bin/python comparative_methods/EFRM-PyTorch/analyze_pretraining.py \
  --run-dir comparative_methods/EFRM-PyTorch/runs/pretraining/<run_id>
```

The command writes `analysis/REPORT.md`, machine-readable
`analysis_metrics.json`, an epoch table, and 300-dpi PNG plus editable SVG
figures for training dynamics, optimizer behavior, loss decomposition,
positive/negative CLIP separation, bidirectional retrieval, and embedding
geometry. A stale `running` marker is classified separately from a completed
run. The analyzer refuses artifacts that report an opened protected test.

The historical trainer exported one final validation batch per completed epoch and
overwrites the previous evidence. Consequently, the post-hoc tool labels saved
retrieval results as batch diagnostics rather than full-validation estimates.
Any future formal run belongs to a clean, separately versioned experiment flow;
the legacy launch path here is not an instruction to resume.
