# EEG–fNIRS physiology-semantic tokenization

<!-- project-state:begin -->
## Current research status

_Generated from `research_state/registry.json`; do not edit this block._

- **主方法**（Tak与10 Hz EEG代理基线下的后续研究）— 未开始 / 尚未判定：后续研究采用Tak确定性四血管状态、p=v及非静息初态，并统一10 Hz EEG功率代理、共享驱动和观测网格。新目标从原生EEG计算100 ms功率，Hb读取既有带通缓存；PCA与尺度只由开发集重新拟合。数据SSE以dt/0.25秒保持每秒工程权重，参考、遮挡及连续正则保持物理时间定义。代码迁移和软件/合成检查完成，未启动新的实测campaign；既有模型与4 Hz证据保留原身份和负结果。
- **Token Atlas**（Atlas Statistical tier）— 已废弃（未完成且不再开展） / 尚未判定：Statistical tier 尚未运行；本旧分析支路废弃，不再开展。
- **对比实验**（统一测量v5下游任务性能重新获取）— 运行中 / 尚未判定：完整988节点计划已从20261016_v2/source_v2冻结快照启动：975个下游评价、12个EFRM全目标数据集排除的三种子预训练及最终汇总。保持5外折×3内折、LP/FT/从头监督和三种信号基线；MI/MA各60部分fNIRS支持窗与REFED480部分支持窗保留。184项综合测试及后续62项缺失输入/恢复/调度专项检查通过；NormWear与BrainFusion真实缺失支持smoke、REFED空标签分支均通过。BrainFusion按内部OOF训练分区独立填补，未定义NVC使用明确零特征约定，旧核及失败证据保留。user systemd服务downstream-rebench-20261016-v2启用且linger开启，40线程/180GiB调度预算，双GPU优先预训练与LP；用户选择完整执行、不设总时限。初版保留20个已完成EEG简单基线、5个中断任务及旧快照；尚无完整方法比较结论。
- **Croce 验证**（新版 Synthetic Phase 1）— 已废弃（未完成且不再开展） / 尚未判定：新版 Synthetic Phase 1 尚未开始；本旧验证流废弃，不再开展。

### Next steps
- **主方法** — 新研究从Tak与10 Hz代理合同出发，按版本声明数据/端点/输出及开发集变换。空间、连续状态和teacher扩展需同时适配Tak状态与10 Hz目标；以独立记录检验观测映射、条件参数和生理资格。
- **对比实验** — 通过同一冻结任务矩阵和持久服务监控、恢复，保持所有计划分母和失败；完整轨道终态后核验被试/种子等权统计、REFED训练均值误差、输入与目标覆盖、REVE重叠及方法准入。预估数月至一年以上，待持续epoch吞吐细化；不因低外折分数改候选或扩大数据范围。

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
