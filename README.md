# EEG–fNIRS physiology-semantic tokenization

<!-- project-state:begin -->
## Current research status

_Generated from `research_state/registry.json`; do not edit this block._

- **主方法**（Tak双离散词表预训练与解释性分析）— 运行中 / 尚未判定：CUDA训练步数与显存字节预算已分离，新增11项回归检查，69项相关测试及全仓测试收集通过，套件提交2baaab15。新v3源码复用合同及生产代码不变的v2目标、训练尺度与完整失败分母，超预算benchmark checkpoint不进入新训练。三个GPU试跑均精确100步；单任务1034、双任务1706区域窗口/秒（含启动开销），选择双并发。systemd服务已进入正式3000步训练，实际进度预算正确；GPU0利用率短采样97%。尚无完整套件科学收益结论。
- **Token Atlas**（Atlas Statistical tier）— 已废弃（未完成且不再开展） / 尚未判定：Statistical tier 尚未运行；本旧分析支路废弃，不再开展。
- **对比实验**（统一测量v5下游任务性能重新获取）— 运行中 / 尚未判定：按用户2026-10-27明确要求，从当前队列移除NormWear全部120个从未启动的任务（30个LP、90个FT），其历史结果和原始988节点清单保留。当前版本化矩阵包含868节点：855个下游评价、12个EFRM预训练及最终汇总；其他方法的工作进程、任务合同、候选、数据和输出路径保持不变。控制器通过现有恢复入口接管全部6个在途训练进程，PID身份与尝试次数均未改变；最终汇总使用缩减后的清单，不将移出任务记为训练失败或缺失结果。27项调度测试及独立systemd进程接管测试通过，持久服务继续运行。
- **Croce 验证**（新版 Synthetic Phase 1）— 已废弃（未完成且不再开展） / 尚未判定：新版 Synthetic Phase 1 尚未开始；本旧验证流废弃，不再开展。

### Next steps
- **主方法** — 继续现有v3固定预算GPU训练，自动接续选解敏感性和冻结词表掩码分析；计算速度和预计完成时间以run中的实测及工程估计为准，保留全部失败，不复用超预算模型。
- **对比实验** — 按照控制器记录的当前版本化任务清单继续执行其余方法，保留全部计划分母和失败；完整轨道终态后核验被试/种子等权统计、目标覆盖、REVE重叠及方法准入。NormWear不再属于本次执行范围，不因历史配置仍列有该方法而重新派发。

See the [generated project status](docs/PROJECT_STATUS.md) for lifecycle states and evidence links.
<!-- project-state:end -->

Start with the [generated project status](docs/PROJECT_STATUS.md),
[experiment sequencing](docs/EXPERIMENT_PLAN.md), and
[documentation map](docs/README.md). For manuscript work, use the
[paper evidence index](docs/PAPER_EVIDENCE_INDEX.md).

## Main entrypoints

| Need | Document |
| --- | --- |
| Which model should new hemodynamic research use? | [Tak baseline and implementation](docs/ARCHITECTURE.md#tak-hemodynamic-baseline) |
| What is stopped, abandoned, or scientifically resolved? | [Generated project status](docs/PROJECT_STATUS.md) |
| Where is the clean-slate experiment entry? | [Experiment sequencing](docs/EXPERIMENT_PLAN.md) |
| What does the evidence permit us to claim? | [Method rationale](docs/METHOD_RATIONALE.md) |
| What data/mask/split contract is active? | [Data contract](docs/DATA_CONTRACT.md) |
| What code is currently runnable? | [Architecture](docs/ARCHITECTURE.md) |
| Where are retained experiment commands and outputs indexed? | [Experiment workspace](experiments/README.md) |
| What results were retained? | [Results index](experiments/RESULTS_INDEX.md) |
| Which sources should feed the manuscript? | [Paper evidence index](docs/PAPER_EVIDENCE_INDEX.md) |
| What contract governed the stopped comparisons? | [Comparison protocol](docs/comparisons/PROTOCOL.md) |
| Which comparison sources and weights are prepared? | [Comparison asset status](comparative_methods/ASSET_STATUS.md) |
| Where is the stopped Token Atlas evidence? | [Token Physiology Atlas](docs/analysis/TOKEN_PHYSIOLOGY_ATLAS.md) |

## Repository layout

```text
src/                    reusable data, inference, teacher, and tokenizer modules
tests/                  default software and shared-contract tests
experiments/            SSM diagnostics, reviewed configs, tools, and retained evidence
comparative_methods/    stopped method owners plus frozen comparison history
croce_validation/       stopped physical-model validation and derived caches
docs/                   authority map, concise contracts, and dated history
research_state/         single registry for execution and scientific verdicts
data/                   immutable measured inputs and owner-local derived caches
```

Start code changes with the [source map](src/README.md) and the
[entrypoint/config/test map](experiments/README.md#entrypoint-config-and-test-map).
New executable workflows go in `experiments/scripts/`; existing root-level
commands retain their paths. Local upstream references live in the ignored
`reference_repository/` and method-local `upstream/` directories.

Generated payloads are ignored except for explicitly retained evidence packages.
Active tools do not recursively search
archives, upstream mirrors, caches, checkpoints, or run trees, and comparison
packages write only to their own run roots. Frozen paths stay in place when reports
depend on them; directory cleanup is versioned rather than hidden behind
compatibility layers.

## Environment and checks

Use the repository environment; system Python is not assumed to be complete.

```bash
source .venv/bin/activate
python -m pytest --collect-only -q
```

SSM synthetic and observation diagnostics are implemented; the next tokenizer
generation has no training launcher yet. A new implementation must own
its versioned config, synthetic checks, split contract, and output namespace rather
than repurposing an E0–E2 entrypoint.

Superseded code, configs, tests, plans, and local runs were moved to one
Git-ignored archive generation. Frozen evidence paths and the retained surfaces in
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md) remain in place;
no directory used by a live process may be cleaned.
