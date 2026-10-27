# Source-code map

Reusable code is organized by responsibility. SSM diagnostics have executable
implementations. The independent conditional-supervision tokenizer screen has a
versioned runtime in `experiments/scripts/run_functional_tokenizer.py`; the
physical-teacher qualification path remains separate.
Execution and scientific verdicts come from the
[project registry view](../docs/PROJECT_STATUS.md), not this directory map.

| Package | Responsibility and reading entry |
| --- | --- |
| `data/` | registry, unified loaders, preprocessing, alignment, masks, geometry, and retained target contracts |
| `inference/` | Current Tak forward/replay and fitting: `shared_driver_reconstruction.py`, with reduced RK4/tangents in `shared_driver_rk4.py`; `observation_baselines.py` owns shared observation transforms. T3/Step5, scalar/component, spatial, continuity and compatibility paths retain explicit historical model contracts |
| `teachers/` | `functional_teacher.py`: retained pre-Tak conditional SSM target adapter; `physical_state_teacher.py`: retained training-teacher adapter; neither is automatically relabeled by the baseline change |
| `tokenizers/` | `functional_tokenizer.py`: independent S/O continuous/hard-VQ screen; corrected EMA VQ and stopped E2/R2 replay |
| `analysis/` | `functional_tokenizer.py`: frozen grouped probes and integer-only reader; stopped Token Physiology Atlas |
| `losses/` | losses required by the stopped E2 runtime |
| `metrics/` | trajectory reliability and the in-memory residual-field schema used by diagnostic readers and reports |
| `foundation/` | frozen-token consumer interface |
| `visualization/` | Token Physiology Atlas figures |
| `utils/` | shared I/O, logging, checkpoints, metrics comparison, and registry validation/rendering |

The retained E2 public tokenizer surface has one owner:
`PhysiologySemanticTokenizer`. Generic tokenizers and the failed continuous,
lag-conditioned, and observation-SSM generations were moved to the local ignored
archive; packages do not import that archive. The E2 surface is stopped and
replay-only.

For new shared-driver work, start with the [Tak baseline](../docs/ARCHITECTURE.md#tak-hemodynamic-baseline)
and `fit_nonlinear_shared_parameters` in `inference/shared_driver_reconstruction.py`, then use the
[entrypoint/config/test map](../experiments/README.md#entrypoint-config-and-test-map)
to find the diagnostic that exercises it. The observation-baseline module owns
array transforms, robust noise estimation, and linear controls shared by Step5
and overnight diagnostics. It does not load recordings or select data splits.

The target/loader and raw-view/teacher modules remain separate to enforce identity
and leakage boundaries. The abandoned observation–source design snapshot is
historical context; it does not select a future tokenizer implementation.

The R1-P prevalidation seal fixes exact source, script, config, registry, and test
paths/hashes. The retained T0 exporter/consumer path is also left in place. See
[`../tests/README.md`](../tests/README.md) before reorganizing either surface.

New executable workflows belong under `experiments/scripts/` or the isolated
package that owns them; existing commands keep their recorded paths. Registered state is generated in
[`../docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md); implemented code does not
imply scientific support or data authorization.
