# Software architecture

_Software surfaces and package ownership. Tak is the shared-driver hemodynamic
baseline; retained tokenizer runtimes keep their recorded model identities. Registered execution and
scientific verdicts are generated in [`PROJECT_STATUS.md`](PROJECT_STATUS.md)._

## Repository layers

The repository deliberately retains some frozen paths because configs, tests, reports,
and SHA-bound evidence refer to them. Physical presence does not make every subtree
active. Use these three layers:

| Layer | Paths | Default rule |
| --- | --- | --- |
| Development code | `src/`, `tests/`, indexed `experiments/*.py`, `experiments/scripts/` and reviewed configs | read and change through the owning package; runnable code does not imply qualification |
| Local/generated | `data/`, `runs/`, `cache/`, `checkpoints/`, `upstream/`, `.tmp/` | do not recursively discover; write only to an existing owner root |
| Frozen/history | dated reports, the local ignored archive, completed comparison campaign files | explicit-path, read-only unless a versioned migration is the task |

A bulk move of frozen comparison code would invalidate recorded paths and result
references, so future cleanup must create a new version instead of layering shims
over the old one.

## Runtime scope and version boundary

The current shared-driver path is:

```text
src/data/                              registry, native loaders, masks and preprocessing
src/inference/observation_baselines.py   shared transforms, noise and linear controls
src/inference/shared_driver_reconstruction.py  Tak forward/replay and joint parameter fitting
src/inference/shared_driver_rk4.py       reduced Tak RK4 and exact discrete tangents
experiments/                           existing diagnostic entrypoints and reviewed configs
experiments/scripts/                   reporting tools and all new executable entrypoints
```

Use the [source map](../src/README.md) for package responsibilities and the
[entrypoint/config/test map](../experiments/README.md#entrypoint-config-and-test-map)
for commands. Shared computations belong in `src/`; data/replay roots are passed
at their read boundaries rather than assigned to another runner's globals.
Recorded command paths and frozen source snapshots retain their identities.

## Tak hemodynamic baseline

The model choice is explained in
[`METHOD_RATIONALE.md`](METHOD_RATIONALE.md#tak-baseline-for-subsequent-research--2026-10-27).
The reviewed executable settings are the `parameters`, `solver`, `fitting` and
`arms[name == model.primary]` entries of
[`shared_driver_tak_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_v1.yaml).
That dated comparison contract is immutable; a later experiment gives its own
data/split/output contract and versions any deliberate change to these settings.

The deterministic vascular states are $(s,f,v,q)$, with $p\equiv v$:

$$
\begin{aligned}
\dot s &= r-\kappa s-\gamma(f-1), & \dot f &= s,\\
\dot v &= \frac{f-v^{1/\alpha}}{\tau+\tau_v}, &
f_{\rm out} &= v^{1/\alpha}+\tau_v\dot v,\\
\dot q &= \frac{fE(f)/E_0-f_{\rm out}q/v}{\tau}, &
E(f) &= 1-(1-E_0)^{1/f}.
\end{aligned}
$$

Here $r$ is the EEG-constrained driver; $s$ is the flow-inducing signal;
$f,v,q$ are flow, venous volume and deoxy-Hb relative to rest. Transit time
$\tau$ and viscoelastic time $\tau_v$ are in seconds. In the declared time
coordinate, $s$ has units of inverse seconds, $\kappa$ inverse seconds and
$\gamma$ inverse seconds squared; the driver has the effective units required
by $\dot s$. The EEG readout fixes its engineering scale. $\alpha$ and $E_0$
are dimensionless. The q equation retains its own $\tau$ denominator.

The physical initial states $(s_0,f_0,v_0,q_0)$ are optimized through
$(s_0,\log f_0,\log v_0,\log q_0)$; p has no independent initial coordinate,
process innovation or noise. Positivity and $P_0v\ge Q_0q$ are checked at RK
stages. The primary setting estimates $\kappa,\tau$ with the declared log
prior, holding $\gamma,\tau_v,\alpha,E_0$ and observation gains fixed. Volume
is penalized once. Shared parameter priors are counted once per native
record/session/channel group; separate records keep separate initial states.

Canonical Hb changes are $H_O=P_0(v-1)-Q_0(q-1)$ and $H_R=Q_0(q-1)$.
The existing observation operator then bins/references the prediction. New
measured targets use the [unified cached Hb contract](DATA_CONTRACT.md#新-hb-建模输入统一使用带通缓存--2026-10-27);
unknown physical units and uncalibrated optical scales remain explicit.

`nonlinear_driver_forward` and `replay_nonlinear_driver` default to Tak.
`fit_nonlinear_shared_parameters` is the fitting entry for one or multiple
records, including fixed-parameter fits with an empty parameter list. Its
default is the four-initial-state Tak model. For compatibility, returned
trajectories use `[r,s,f,v,p,q]`; this is a six-column view with **derived p**,
not six independent states. A four-entry Tak initializer is accepted directly.

The old scalar/component fitters, linear design, decay interventions,
`t3a_balloon_robust_ssm`, `t3a_balloon_joint_ssm`, `balloon_trajectory_map`,
multi-shooting/spatial interfaces and conditional tokenizer targets retain
their original inlet-balance contracts. Historical calls to the common
forward/replay/joint fitter explicitly select that model. The old T3 module
also supplies parameter/observation containers; using those containers does
not select its dynamics or independent process noise. New Tak extensions
must preserve the reduced-state invariant rather than reuse an old p−v
metric, linearization or teacher schema as if it already applied to Tak.

Sources: [Tak et al. (2015), equations 3–6](https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf),
[SPM dynamics](https://github.com/spm/spm12/blob/main/toolbox/dcm_fnirs/spm_fx_fnirs.m),
[SPM initial states and priors](https://github.com/spm/spm12/blob/main/toolbox/dcm_fnirs/spm_dcm_fnirs_priors.m).

## Retained tokenizer scope

The E2-compatible `PhysiologySemanticTokenizer` (v1) is a retained runtime.
The separately versioned `functional_tokenizer.py` implements the later
conditional-supervision screen with its recorded pre-Tak teacher; it is not
automatically a Tak-based tokenizer. R0-P, R1-D/R1-P, D1B, and R2-D added
diagnostic and qualification components around it. The v1 implementation,
checkpoints, and negative screen are **stopped historical evidence** and are
replay-only; no new tokenizer runtime generation is active. Their execution outcomes
and scientific interpretation are recorded in the owning reports rather than
inferred from file presence.

The theory and architecture boundary is retained in
[`METHOD_RATIONALE.md`](METHOD_RATIONALE.md). It is not runnable code, an
admission result, or permission to access measured/protected data. No concrete
tokenizer implementation is registered in the clean-slate flow. SSM diagnostics
implement bounded precursor questions under `EXPERIMENT_PLAN.md`; a future
tokenizer implementation requires a versioned software contract and synthetic checks, while an
independent evaluation would separately preregister its estimator and
task-specific evidence settings.

Specifically, partial information decomposition (PID) is at most a replaceable
probe during later pretraining development. It is not a core contribution, a runtime
requirement, or an item frozen by this candidate map.

The software consequences of the owning freeze are:

| Design object | Architecture boundary |
| --- | --- |
| Data identity, masks, splits, protected boundary | Hard-frozen through `DATA_CONTRACT.md`; no array-order joins or missing/zero/padding conflation |
| Input ownership | Main coupling paths implement `Z_E=f_E(X_E)` and `Z_F=f_F(X_F)`; neither tokenizer reads the other modality before producing its representation |
| Continuous target order | Preserve timestamps and construct a continuous trajectory before patch/tokenization; rate, coordinates, filters, and dimensions stay open |
| Teacher | Label-blind, fit-fold-only, provenance/support/uncertainty-carrying, and never ground truth; teacher family stays open |
| Codebook | If VQ is used, modality namespaces are independent and equal IDs have no shared semantics; VQ, `K`, and `D` stay open |
| Observation/source | Functional roles and falsification endpoints are fixed; encoder, decoder, loss, latent dimension, and physical module split stay open |
| Lag grammar/evidence | Endpoint-aligned increment, baseline, proper score, and null operators form the method kernel; grammar network stays open before preregistration |
| Fine-to-coarse | Optional capacity/stability/readability mechanism, not a scientific invariant |
| Cross masking | Undefined and unfrozen until a versioned information-intervention contract exists |

The machine-readable retained-runtime authority is
[`physiology_semantic_architecture.json`](physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json).
The review-oriented **quick overview / paper-figure candidate** is
[`physiology_semantic_runtime_overview.svg`](physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg),
with its editable
[`drawio` source](physiology_semantic_tokenizer/architecture/physiology_semantic_runtime_overview.drawio).
It is a retained historical presentation projection of the stopped E2 runtime,
not a timestamped registry view, second source of truth, or scientific-admission
figure.
The detailed
[`candidate architecture`](physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)
and its
[`Draw.io source`](physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.drawio)
are exploratory visual projections, not a runtime or frozen target. Draw.io owns
their visual layout; the JSON and registry remain the implementation and state
authorities.

![Quick runtime overview (presentation draft)](physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg)

## Abandoned observation–source candidate snapshot (unimplemented)

The earlier exploration projection is retained separately from the stopped v1
runtime and the method-rationale contract. It is an **abandoned** candidate map,
not a future target:

- [`observation_source_exploration_v2.json`](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json)
  is the text-diffable pre-freeze candidate note.
- [`observation_source_exploration_v2.drawio`](physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.drawio)
  owns the editable visual layout and shared project figure style.
- [`observation_source_exploration_v2.svg`](physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg)
  is the exported framework figure.

The JSON owns only the snapshot content; this Mermaid view is a compact reader
aid. Neither overrides the frozen boundary above or defines an implementation:

```mermaid
flowchart LR
    eeg0["EEG native waveform<br/>200 Hz"] --> eeg1["frequency-aware,<br/>amplitude-preserving envelope"]
    nir0["fNIRS native branch<br/>HbO/HbR provenance"] --> nir1["continuous 10 Hz trajectory"]
    eeg1 --> eeg10["EEG aligned target<br/>10 Hz, e.g. 18 coordinates"]
    eeg10 -. "candidate" .-> es["modality-specific low-rank self teacher<br/>LDS/neural-SSM family; label-blind, fit-fold only"]
    nir1 -. "candidate" .-> fs["modality-specific low-rank self teacher<br/>LDS/neural-SSM family; label-blind, fit-fold only"]
    eeg10 -. "optional" .-> jt["privileged joint Croce candidate<br/>fit-only / ablation"]
    nir1 -. "optional" .-> jt
    es --> eo["EEG teacher output<br/>trajectory + uncertainty + observation residual"]
    fs --> fo["fNIRS teacher output<br/>trajectory + uncertainty + observation residual"]
    jt -. "offline target only;<br/>never inference input" .-> eo
    jt -. "offline target only;<br/>never inference input" .-> fo
    eo --> se["EEG source path S_E"]
    fo --> sf["fNIRS source path S_F"]
    eeg0 --> oe["EEG observation path O_E^res"]
    nir0 --> of["fNIRS observation path O_F^res"]
    eo -. "optional residual target" .-> oe
    fo -. "optional residual target" .-> of
    se --> qe["independent Q_E<br/>fine tokens Z_E^f"] --> ae["A_E<br/>coarse Z_E^c"]
    sf --> qf["independent Q_F<br/>fine tokens Z_F^f"] --> af["A_F<br/>coarse Z_F^c"]
    ae -. "optional" .-> gram["endpoint-aligned grammar<br/>P(Z_F^c(t+tau)|Z_E^c(t), history, condition)"]
    af -. "optional" .-> gram
    oe -.-> probe["optional conditional-contribution probe<br/>development diagnostic only"]
    of -.-> probe
    se -.-> probe
    sf -.-> probe
    gram -.-> probe
    gram --> eval["select estimand + preregister<br/>held-out proper scores + nulls"]
```

The graph records abandoned implementation candidates for historical comparison;
it does not keep a runtime selection available. Teacher family, codebooks,
fine-to-coarse mapping, and exact modules remain replaceable only in a future
versioned contract. Its historical `optional grammar` label is superseded: the
retained method kernel fixes endpoint alignment, the tested increment, baseline,
proper score, and null operators while leaving the grammar network open.

If a grammar is tested, its fit/selection map is a learned artifact. Coupling
evidence is available only after a final estimator and evaluation protocol are
preregistered and applied to fresh held-out rows. A training map,
reconstruction score, occupancy plot, or decomposition label is not evidence.

## Stopped E2/v1 historical dataflow

```mermaid
flowchart LR
    loader["Unified measured local view<br/>20 s · 10 patches"]
    eeg["EEG B×6×4000"] --> ep["patch-local encoder"]
    fnirs["HbO/HbR B×2×200"] --> fp["patch-local encoder"]
    loader --> eeg
    loader --> fnirs
    ep --> es["semantic D64"] --> eq["independent EMA VQ K128"]
    ep --> er["continuous residual D64"]
    fp --> fs["semantic D64"] --> fq["independent EMA VQ K128"]
    fp --> fr["continuous residual D32"]
    eq --> recon["raw reconstruction"]
    er --> recon
    fq --> recon
    fr --> recon
    sidecar["E2 target sidecar<br/>(training only; no semantic row admitted)"] --> routed["training-only objective probes"]
    eq --> routed
    fq --> routed
    eq --> export["IDs, posterior, vectors, residual, masks, provenance"]
    fq --> export
```

- Inputs are measured modality-specific views; subject/task/nuisance/teacher
  metadata are not encoder inputs.
- EEG and fNIRS have independent codebooks. Equal numeric IDs have no shared
  semantics.
- The teacher is privileged training/diagnostic evidence, not inference input
  and not physiological ground truth.
- The full-window teacher and patch-local token have different receptive
  fields; E2's weak routed objectives did not resolve that mismatch.
- Artifact annotations are diagnostics. Real recorded support in `valid_mask`
  remains the current validity authority.

## Retained package ownership

| Package / entrypoint | Retained responsibility |
| --- | --- |
| `src/data/registry.py`, `factory.py`, `unified_physiology.py` | central measured-data registry and loader |
| `src/data/physiology_semantic_*` | E0–E2 local views and targets |
| `src/data/shared_driver_*` | independent raw view/teacher joins for R-series work |
| `src/tokenizers/physiology_semantic_tokenizer.py` | stopped E2 tokenizer; replay-only |
| `src/tokenizers/ema_vector_quantizer.py` | corrected fixed-K128 VQ |
| `src/tokenizers/shared_driver_semantic_vq.py` | R2 diagnostic model component; not promoted runtime |
| `src/inference/adaptive_neurovascular_ssm.py` | Croce/Balloon-inspired adaptive five-state RTS joint candidate; E0 offline development supervision accepted, R1-P population-frozen qualification rejected; not a qualified future teacher |
| `src/teachers/physical_state_teacher.py` | frozen teacher output adapter and consumer boundary |
| `src/analysis/token_*` and `physiological_patch_features.py` | stopped Token Physiology Atlas |

Executable training, qualification, evaluation, and rendering workflows live
under `experiments/`. Comparison methods remain isolated below
`comparative_methods/`.

## Scientific boundary

Code presence is not experiment state and does not imply scientific support.
The v1 runtime and its negative results remain historical facts; the v2 JSON
and SVG are exploratory design artifacts only. Query the unified project status
before using a runnable component. A future method generation requires a
versioned implementation contract; independent evaluation then requires a new
holdout and a preregistered estimator/null/threshold contract. No architecture
edit opens a protected boundary or authorizes a measured campaign.
