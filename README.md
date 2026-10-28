# EEG–fNIRS physiology-semantic tokenization

<!-- project-state:begin -->
## Current research status

_Generated from `research_state/registry.json`; do not edit this block._

- **主方法**（Tak双离散词表波形保真与解释性分析）— 已完成 / 证据不足：三轮有限快速迭代及选中方案三个种子复核完成，保持单模态输入、完整双模态Tak的固定S/O标签和独立整数解码。仅用32名训练、11名验证被试，保留3907/3925试次、11721/11775窗口；测试不参与选择。选中6秒观测前端与两层波形VQ，码率由每模态每2秒14增至28bit。全部训练码字使用128/128，验证至少125/128，有效码字至少93/128，全部健康检查通过。相对旧seed17模型，三种子均值的EEG总NMSE由0.53174/0.52348降至0.16032/0.17614（69.8%/66.4%），Hb由0.04247/0.81274降至0.00726/0.49677（82.9%/38.9%）。EEG残差只改善8.9%/2.2%，各分支均改善30%的整体要求未通过；频带扩展降低训练误差但验证泛化不足。71项针对测试、全仓收集及整数回载/独立指标复算通过，数值差0；同验证被试的种子重复不是独立数据确认，尚无新掩码实验或全面S/O可靠性结论。
- **Token Atlas**（Atlas Statistical tier）— 已废弃（未完成且不再开展） / 尚未判定：Statistical tier 尚未运行；本旧分析支路废弃，不再开展。
- **对比实验**（统一测量v5下游任务性能重新获取）— 运行中 / 尚未判定：按用户2026-10-27明确要求，从当前队列移除NormWear全部120个从未启动的任务（30个LP、90个FT），其历史结果和原始988节点清单保留。当前版本化矩阵包含868节点：855个下游评价、12个EFRM预训练及最终汇总；其他方法的工作进程、任务合同、候选、数据和输出路径保持不变。控制器通过现有恢复入口接管全部6个在途训练进程，PID身份与尝试次数均未改变；最终汇总使用缩减后的清单，不将移出任务记为训练失败或缺失结果。27项调度测试及独立systemd进程接管测试通过，持久服务继续运行。
- **Croce 验证**（新版 Synthetic Phase 1）— 已废弃（未完成且不再开展） / 尚未判定：新版 Synthetic Phase 1 尚未开始；本旧验证流废弃，不再开展。

### Next steps
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
