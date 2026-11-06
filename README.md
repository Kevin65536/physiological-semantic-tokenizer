# EEG–fNIRS physiology-semantic tokenization

<!-- project-state:begin -->
## Current research status

_Generated from `research_state/registry.json`; do not edit this block._

- **主方法**（原生电压双词表与Hb连续性实验）— 已完成 / 证据不足：电极级条件电压路线已实际应用并完成训练测试：4984/5008试次、每试次16个唯一覆盖电极；24个原teacher失败保留，新增修正失败与SSM拟合均为0。8次训练各严格5000步，EEG及Hb拼接基线各3种子。固定100ms同分支解码合成仅按验证结果选定并复用模型；公开测试总Hb NMSE由0.01668/0.01029降至0.01579/0.00964，0.1秒边界变化误差降低86.4%/86.4%，但0.5秒跨度仅降低3.1%/2.8%，较宽patch偏移仍未解决。原始电压总NMSE为0.00268/0.62492，同码率直接量化为0.000925/0.54853；Simultaneous连续模型为0.01505，硬量化受少数被试电压均值超出码字范围主导。EEG修正残差相关仅0.063/0.050，仍未充分学会。码本健康检查通过不代表波形全面保真。97项主套件检查及新增边界邻域回归通过，独立Hb回放差异0；目标float32相对闭合误差低于6.3e-8。此为既有公开测试的探索性结果，未开展新的掩码预训练或生理真值确认。
- **Token Atlas**（Atlas Statistical tier）— 已废弃（未完成且不再开展） / 尚未判定：Statistical tier 尚未运行；本旧分析支路废弃，不再开展。
- **对比实验**（统一测量v5下游任务性能重新获取）— 已中止 / 证据不足：用户要求停止本机下游序列并归档。原调度器收到SIGTERM后记录interrupted并退出，5个工作进程全部停止；正式服务与3个历史smoke服务均已禁用自动启动。868项计划保留772完成、5中断、91待执行；计划分母不删除、不将中断算完成。停止时复用既有聚合入口核验保留公开预测：855个评价任务中760有效，83格中46格TABLE_READY_WITH_NOTE、30格REJECTED_VALUE、7格INVALID_VALUE。原位保留完整模型/预测/失败证据，紧凑证据包已逐文件回读核对；未读取原始或受保护数据、未新增拟合。旧Visual输入及REFED负结果边界继续保留，完整矩阵未完成，不能称最终全量基准。
- **Croce 验证**（新版 Synthetic Phase 1）— 已废弃（未完成且不再开展） / 尚未判定：新版 Synthetic Phase 1 尚未开始；本旧验证流废弃，不再开展。

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
