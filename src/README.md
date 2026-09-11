# Source-code map

Reusable code is organized by responsibility. SSM diagnostics have executable
implementations; the next tokenizer generation has no training runtime yet.
Execution and scientific verdicts come from the
[project registry view](../docs/PROJECT_STATUS.md), not this directory map.

| Package | Responsibility and reading entry |
| --- | --- |
| `data/` | registry, unified loaders, preprocessing, alignment, masks, geometry, and retained target contracts |
| `inference/` | Balloon forward/robust inference (`t3a_balloon_robust_ssm.py`), joint and temporal observation inference (`t3a_balloon_joint_ssm.py`), trajectory MAP (`balloon_trajectory_map.py`), and shared observation baselines (`observation_baselines.py`); older adaptive/SMC models remain retained references |
| `teachers/` | `physical_state_teacher.py`: frozen training-teacher output adapter; separate from raw loaders and inference solvers |
| `tokenizers/` | stopped E2 tokenizer, corrected EMA VQ, and retained R2 diagnostic component (replay-only) |
| `analysis/` | stopped Token Physiology Atlas |
| `losses/` | losses required by the stopped E2 runtime |
| `metrics/` | trajectory reliability and the in-memory residual-field schema used by diagnostic readers and reports |
| `foundation/` | frozen-token consumer interface |
| `visualization/` | Token Physiology Atlas figures |
| `utils/` | shared I/O, logging, checkpoints, metrics comparison, and registry validation/rendering |

The retained public tokenizer surface has one owner:
`PhysiologySemanticTokenizer`. Generic tokenizers and the failed continuous,
lag-conditioned, and observation-SSM generations were moved to the local ignored
archive; packages do not import that archive. The E2 surface is stopped and
replay-only.

For SSM work, start with the relevant solver in `inference/`, then use the
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
