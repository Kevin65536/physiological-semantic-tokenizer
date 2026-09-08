# EEG–fNIRS physiology-semantic tokenization

<!-- project-state:begin -->
## Current research status

_Generated from `research_state/registry.json`; do not edit this block._

- **主方法**（PST-DISCOVERY-v1 tokenizer 发现流程）— 未开始 / 尚未判定：旧合同下Step5A0联合似然与A1的有限合成正证据保留；观测尺度合同与氧提取数值修复通过回归，短窗时序参考已有工程正证据。实测极端流量、持续HbR偏差及双向配对增量问题仍未解决，Step5B teacher负结论保留，全面UQ和tokenizer未晋级。
- **Token Atlas**（Atlas Statistical tier）— 已废弃（未完成且不再开展） / 尚未判定：Statistical tier 尚未运行；本旧分析支路废弃，不再开展。
- **对比实验**（六方法联合正式 campaign）— 已停止（此前已完成） / 混合结论：540/540 jobs 完成且无技术失败；42 个 cell 中 22 个可带注释报告、12 个数值被拒、2 个仅 overlap track、6 个不适用。
- **Croce 验证**（新版 Synthetic Phase 1）— 已废弃（未完成且不再开展） / 尚未判定：新版 Synthetic Phase 1 尚未开始；本旧验证流废弃，不再开展。

### Next steps
- **主方法** — 先验证保留时间相关误差的非线性Student-t观测参考或更靠近处理前的推断坐标，解决HbR欠覆盖和真实低流量塌缩；冻结后仍在原训练边界复核固定/至多一个测量增益候选与配对对照，不扩大GWZ、开发面板或teacher资格，protected 24–29保持关闭。

See the [generated project status](docs/PROJECT_STATUS.md) for lifecycle states and evidence links.
<!-- project-state:end -->

Start with the [generated project status](docs/PROJECT_STATUS.md),
[experiment sequencing](docs/EXPERIMENT_PLAN.md), and
[documentation map](docs/README.md). For manuscript work, use the
[paper evidence index](docs/PAPER_EVIDENCE_INDEX.md).

## Main entrypoints

| Need | Document |
| --- | --- |
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
| How should code and experiments be changed? | [Contributor guide](CONTRIBUTING.md) |

## Repository layout

```text
src/                    active reusable library code
tests/                  default software and shared-contract tests
experiments/            stopped configs/workflows plus retained local evidence
comparative_methods/    stopped method owners plus frozen comparison history
croce_validation/       stopped physical-model validation and derived caches
docs/                   authority map, concise contracts, and dated history
data/                   immutable measured inputs and owner-local derived caches
```

Generated payloads are ignored by Git. Active tools do not recursively search
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

There is no forward-method training launcher yet. A new implementation must own
its versioned config, synthetic checks, split contract, and output namespace rather
than repurposing an E0–E2 entrypoint.

Superseded code, configs, tests, plans, and local runs were moved to one
Git-ignored archive generation. Frozen evidence paths and the retained surfaces in
[`experiments/RESULTS_INDEX.md`](experiments/RESULTS_INDEX.md) remain in place;
no directory used by a live process may be cleaned.
