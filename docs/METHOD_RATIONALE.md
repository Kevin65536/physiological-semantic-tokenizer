# Method rationale and claim boundary

_Consolidated from the legacy postmortem, theoretical foundations, and
architecture-return review; v1 historical boundary and frozen forward method
principles, 2026-08-24_

## Version boundary and status

This document has two deliberately different jobs. **v1** names the
implemented E2/SSM_OBSERVATION/LC-SPVQ development surfaces and their audited
negative results. The **forward contract** freezes only the theory and
architecture principles listed below; the concrete implementation remains
unimplemented and open where the contract says it is open. This documentation
freeze does not authorize a measured run, open a protected split, or relabel a
v1 result.

The v1 code, checkpoints, manifests, and reports remain reproducible history.
They must not be silently upgraded by changing their names in prose. The
forward-looking claim boundary is therefore:

```text
v1 implementation/negative evidence   ->  historical fact
forward theory/architecture principles ->  frozen below
implementation candidates              ->  replaceable until versioned
future held-out coupling claim          ->  unavailable until implementation is
                                           synthetic-tested and its evaluation
                                           contract is preregistered
```

Partial information decomposition (PID) is only a replaceable direction for
later pretraining development. It is **not** the core contribution, method
identity, architecture invariant, or a frozen objective, and a future method
may omit it entirely. If a PID-style probe is eventually selected, only that
concrete estimator, objective, and evaluation protocol may be versioned for the
corresponding experiment; the present exploration freezes none of them.

No measured or protected data was read for this documentation update.

## Research question

The project asks whether separately observed EEG and fNIRS can support useful
representations of a shared physiological process without writing one modality
directly into the other. This is stricter than showing that a multimodal model
can reconstruct signals, classify tasks, or produce visually structured token
co-occurrence.

Three propositions must remain separate:

1. a coordinate can be constructed from joint EEG–fNIRS data;
2. each modality can independently predict that coordinate;
3. the coordinate is physically adequate and supports a reproducible
   incremental cross-modal claim.

The first two do not prove the third.

## Comparative positioning and novelty boundary

The candidate exploration changes possible estimands and evidence interfaces; it does not, by
itself, establish priority or superiority over earlier multimodal methods.
[DMSL](https://pubmed.ncbi.nlm.nih.gov/41632672/) combines multimodal attention,
reconstructed-modality graph processing, consistency/disparity constraints,
and adaptive fusion for EEG--fNIRS classification. [FOCAL](https://proceedings.neurips.cc/paper_files/paper/2023/file/93e98ddf39a9beb0a97fbbe56a986c80-Paper-Conference.pdf)
uses a factorized orthogonal latent space with modal matching, private-view
invariance, and temporal structure; [FactorCL](https://proceedings.neurips.cc/paper_files/paper/2023/file/6818dcc65fdf3cbd4b05770fb957803e-Paper-Conference.pdf)
uses mutual-information bounds and multimodal augmentation assumptions to
target task-relevant shared and unique information. [EEGPT](https://papers.nips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
motivates high-signal, semantically informed EEG representation targets rather
than treating a noisy raw waveform as the only self-supervision target.

Available candidates include continuous trajectory, uncertainty, and residual
targets; modality-specific observation/source inference and vocabularies; a
fine-to-coarse endpoint grammar; and a preregistered held-out
proper-score/null evidence surface. These are hypotheses about input ownership,
estimands, and claim contracts, not a frozen bundle. Direct performance or novelty claims require a
same-track comparison with matched inputs, splits, pretraining exposure, and
training budget; none is supplied by this documentation update.
Invoking PID or naming operational information components is not itself a
novelty claim.

## What the earlier tokenizer taught us

The pre-physiology-semantic generation used source/residual targets, four
quantizers, coupling or exchange modules, and hard token IDs. Its strongest
audited control was the X3 causal-exchange run. The evidence remains useful as
a failure surface, not as an active architecture.

| Observation | Direct evidence | Method lesson |
| --- | --- | --- |
| Hard IDs lost usable structure | LOSO CCA: continuous `0.1483`, soft `0.1589`, hard one-hot `0.0584`, quantized embedding `0.0602` | Export posteriors, embeddings, and continuous latents; do not treat hard ID as the whole representation |
| Global dependence was not task-local | Global held-out NLL gain `0.3044`; mental arithmetic `-0.0189`, motor imagery `-0.0149`; WG interval crossed zero | Compare against task/phase/history marginals and report local uncertainty |
| Strong exchange contaminated the test | EEG context entered the fNIRS source encoder before quantization | Inference paths must remain modality-specific when testing cross-modal information |
| High utilization did not imply semantic geometry | Effective ranks included EEG source `12.28`, fNIRS source `7.20`, fNIRS observation `3.51` | Audit rank, stability, support, and phenotype rather than occupancy alone |
| Downstream performance was confounded | Source identity balanced accuracy `0.6476`; fine task label `0.2851` | Dataset/style information cannot be presented as physiological semantics |

Reconstruction, codebook utilization, a coupling heatmap, and a downstream
score are therefore engineering or descriptive observations unless an
explicit scientific gate binds them to the intended claim.

## E0–E2 generation

The physiology-semantic generation introduced a physical-teacher coordinate
and a corrected fixed `K=128` quantizer.

- E0 admitted a sign-calibrated adaptive teacher for development supervision
  only. It did not establish teacher identifiability or ground truth.
- E1 established a healthy three-seed K128 software/occupancy surface.
- E2 completed nine runs. Weak state/prototype/context/coupling objectives did
  not improve a preregistered semantic row, so the final decision retained T0.

This result rules against the tested weak multi-entry objective route. It does
not prove that no EEG–fNIRS relationship exists.

## Why the SD-SVQ return was tested

The 2026-07-25 Shared-Driver Semantic VQ proposal deliberately removed same-ID
semantics, shared codebooks, pre-VQ exchange, and a mandatory foundation-model
consumer. Its intended core was:

- raw-only modality-specific full-window encoders;
- independent `K=128,D=64` codebooks;
- a training-only complete joint-driver proxy trajectory;
- frozen exports followed by an independent evaluator.

The full-window scope was always offline/bidirectional. A future raw-fNIRS
prediction claim required a separate completed-window cutoff experiment.

The original frozen architecture, implementation, migration, and experiment
documents remain in
[`physiology_semantic_tokenizer/`](physiology_semantic_tokenizer/README.md) as
the preregistered 2026-07-25 generation. They are no longer active work
instructions.

## What R0–R2 established

The R series tested the prerequisite for quantization rather than assuming it.

| Gate | Result | Interpretation |
| --- | --- | --- |
| R0-P raw alpha–HbO lag | validation AUC `-0.02202`, CI `[-0.06685, 0.04020]`, `p=0.8224`; no 30-family FWER discovery | the registered low-dimensional raw-lag effect did not replicate |
| R2-D continuous observability | EEG ΔR² `0.031296`, CI `[-0.002166, 0.069625]`; fNIRS `-0.023018`, CI `[-0.035806, -0.007696]` | bilateral full-trajectory criterion failed |
| R1-P teacher physical consistency | HbO gain `0.234535`, but only `3/5` subjects exceeded the frozen threshold; HbR passed `4/5` | a non-degenerate, decodable coordinate still failed physical qualification |
| R1-P jointness/observability/nulls | G3–G6 passed | positive necessary properties did not compensate for G2 |
| D1B validation | serializer stopped before endpoint calculation and atomic publication | scientifically undetermined, not pass or fail |

The snapshot decision is:

```text
promotion_eligible = false
next_action = do_not_enter_r2_p
protected_subjects_24_29 = closed
```

Detailed methods and numbers remain in the
[`R-series report`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).

## What the 2026-08 v1 screen did and did not establish

The latest exploratory screen is a v1 implementation audit, not a v2
qualification. The corrected endpoint-aligned lag mask removed the historical
same-position shortcut; the old LC-SPVQ checkpoint still showed no stable
sample-dependent interaction increment (motor-imagery private-only macro-F1
was already `0.7273`, while the historical word-generation combined head was
worse). The interaction-logit variation was on the order of `10^-6`, and the
shuffle control was essentially unchanged.

The v1 observation target also did not have the geometry claimed for a
continuous teacher. A 20 s window was split into ten 2 s patches; fNIRS patch
samples were flattened into features and the full-rank observation AR smoother
then operated on the ten patch positions. It therefore saw approximately
`[B, 10, 40]` for the two-channel fNIRS branch, not a `[B, 200, 2]` 10 Hz
trajectory. EEG used absolute patch log-band power with a generic
full-patch-convolution/LayerNorm stem. These are historical v1 contracts.

The three-seed v1 SSM screen found motor-imagery EEG delta-R² from `-0.0554`
to `-0.0702` across NATIVE/SSM modes while fNIRS remained about `0.4737`;
word-generation EEG was only `0.0285`--`0.0347` while fNIRS was
`0.4251`--`0.4252`. SSM-SELF did not beat NATIVE, SSM-JOINT provided no useful
upper-bound gain, and low-weight XPRED did not provide a stable repair. The
longer one-seed motor-imagery probe made the EEG endpoint worse (`-0.2610`,
with fNIRS `0.8505`) and is not a multi-seed full-budget estimate. The
condition-by-time mean was a useful secondary trial-residual baseline, but it
was too strong to serve as the only admission gate. The v1 private decoder
reconstructed an observation residual rather than complete raw/high-rate
modality information; in NATIVE that residual is identically zero.

These facts rule out the tested v1 implementation as evidence for a learned
coupling mechanism or SSM superiority. They do **not** rule out separately
testing a modality-specific dynamic target, an observation--source split,
an independent vocabulary, or a lagged grammar. The v1 gate consequently
deferred K16/q0/q1 work; it did not select or preserve a successor architecture.

## Frozen theory and architecture contract (unimplemented)

Only the nine rows below define the forward freeze. They freeze scientific
principles, functional semantics, and evidence contracts rather than a concrete
network. No implementation or measured evaluation has started.

| Design object | Frozen boundary | Explicitly open boundary |
| --- | --- | --- |
| Data source, canonical identity, masks, splits, and protected boundary | Hard-frozen within the method generation. Joins use canonical identity rather than array order; missing, zero, and padding remain distinct; fitting occurs only on the authorized partition. | A correction requires a versioned contract rather than an in-place reinterpretation. |
| EEG/fNIRS input ownership | Every main path that carries a coupling claim satisfies $Z_E=f_E(X_E)$ and $Z_F=f_F(X_F)$. Before each tokenizer produces its representation, it cannot read the other modality. | Network structure remains replaceable. A privileged joint teacher is a separately declared training-only ablation, not a main-path input. |
| Continuous target before patch/tokenization | Timestamps and continuous trajectories are preserved; the continuous target is constructed before patching or tokenization. | Sampling rate, EEG coordinates, filterbank, target dimension, patch length, and stride remain replaceable. |
| Teacher epistemic boundary | A teacher is not ground truth. It is label-blind, fit-fold-only, and carries provenance, support, and uncertainty. | NATIVE, LDS, neural SSM, Croce/RTS, and other teacher families remain replaceable. |
| Codebook semantics | If VQ is used, EEG and fNIRS use independent codebook namespaces; equal numeric IDs never imply equal cross-modal semantics. | Whether VQ is used, plus $K$, $D$, quantizer family, and parameters remain replaceable. |
| Observation/source functions | `observation` preserves modality-specific measured information; `source` represents the continuous teacher target from the same modality. Their claimed roles are falsified when their preregistered incremental endpoints fail the corresponding baseline/null contract. | Encoder, decoder, loss, and latent dimension remain replaceable; the functions need not be separate physical modules. |
| Endpoint-aligned lag grammar, proper score, and nulls | The main-method evidence kernel fixes the endpoint-aligned estimand, the increment being tested, its baseline, its proper-score endpoint, and its null operators before held-out access. Source contribution is tested beyond the observation baseline; grammar interaction is tested beyond observation plus source. | The exact grammar network and estimator implementation remain replaceable before preregistration. |
| Fine-to-coarse hierarchy | No scientific claim depends on a fine-to-coarse hierarchy. | The hierarchy, capacities, aggregation, and whether it is used remain replaceable. |
| Cross masking | It is not frozen and cannot be claimed as a defined component until a versioned contract specifies it as an information intervention. | No architecture may use the name alone as a frozen mechanism. |

The proper-score endpoint and null operators are task-specific but stable once
declared for that task. Learned grammar maps remain training artifacts;
held-out proper-score increments and their declared null comparisons are the
coupling evidence. Cross masking is outside this freeze until its intervention
contract exists.

## Interpretation hierarchy

Use the narrowest applicable description:

1. **engineering token** — a stable discrete interface with software and
   occupancy checks;
2. **descriptive physiological token** — a supported token–measurement
   phenotype that is stable under the declared development analysis;
3. **coupling-relevant token** — a frozen representation that improves a
   preregistered cross-modal endpoint over appropriate marginal, timing, and
   subject-level nulls;
4. **mechanistic or causal token** — requires intervention or identification
   evidence not present in this project.

The completed development-only E2 T0 Atlas can support level 1 and carefully
bounded level-2 descriptions under its observed support/stability limits. The
R-series stop decision blocks promotion to levels 3–4.

## If the main method is restarted

No next SD-SVQ run is supported by this snapshot. A restart would require:

- a genuinely new independent holdout;
- a newly frozen estimator, null family, threshold, and stopping contract;
- synthetic end-to-end tests for dtype, serialization, cache-to-summary, and
  atomic publication before measured access;
- explicit competition among phase/history/systemic and physiological
  explanations;
- a target such as held-out predictive residual beyond task phase and fNIRS history,
  rather than visual token co-occurrence.

Only a newly qualified continuous target may reopen the decision about VQ.

## Evidence-supported statements

This evidence supports:

- the tested K128 quantizer can be software-healthy;
- the E2 weak auxiliary teacher objectives did not improve the registered
  semantic endpoint;
- the population-frozen R1-P coordinate was non-degenerate and bilaterally
  decodable but failed its full physical qualification;
- the R2-D bilateral continuous prerequisite failed;
- SD-SVQ/R2-P/R3–R7 are blocked under the frozen generation.

## Claims not supported by this evidence

This evidence does not support:

- calling the shared driver physiological ground truth;
- treating teacher reconstruction as coupling discovery;
- assigning cross-modal meaning to equal token IDs;
- selecting an attractive patch or lag after viewing the full family;
- opening protected data to rescue a failed public/development gate.
