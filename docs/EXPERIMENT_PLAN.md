# Experiment sequencing

_Design owner for independently versioned experiment protocols. Registered
execution and scientific verdicts live in `research_state/registry.json`._

后续血流动力学研究从
[Tak 基线](ARCHITECTURE.md#tak-hemodynamic-baseline)出发；选择依据由
[方法说明](METHOD_RATIONALE.md#tak-baseline-for-subsequent-research--2026-10-27)持有。
未显式选择历史模式时，共享驱动入口使用
[`Tak 10 Hz v2`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_10hz_v2.yaml)
合同；后续 EEG 代理、驱动及观测网格按
[10 Hz 数据约定](DATA_CONTRACT.md#新-eeg-代理目标统一为-10-hz--2026-10-27)
统一，准备数组和开发集变换重新生成。本次仅修订代码与后续约定，既有采样率
比较不重跑，原结论不改写。旧 T3/Step5、空间、
连续状态和 tokenizer 设计保留其原始模型身份，下面的历史顺序不构成默认
继续执行旧方程的队列。后续新实验需声明自己的数据、端点及输出版本；本次
基线修订不重新执行已完成实测矩阵，也不修改冻结配置或保护边界。
以下按日期保留的设计中，“当前模型”“本轮”等措辞指对应日期的版本，
不覆盖上述后续研究基线。

The 2026-10-27 [cache-input correction](DATA_CONTRACT.md#新-hb-建模输入统一使用带通缓存--2026-10-27)
supersedes native unfiltered Hb preparation for new work. The dated designs below
retain the observation coordinates used for their evidence; the
[entrypoint map](../experiments/README.md#cached-hb-input-migration) identifies the
new cached contracts. This code/input migration does not register a new measured
campaign or reinterpret previously completed results.

This document owns experiment design. The retained `PST-DISCOVERY-v1` path
concerns physical-teacher qualification, source/observation tokenization and
coupling-prior retention. The separately versioned `TK-FUNCTIONAL-v1` screen
below tests conditional supervision without claiming physical qualification.
Neither path uses task labels to train the tokenizer.

The [2026-11-06 normalization redesign](#ssm-so-normalization-redesign-20261106)
specifies a proposed successor for SSM calibration and voltage S/O targets.
It separates numerical coordinates, observation gains, error weighting and
supervision scales; existing executable contracts retain their recorded meaning.

The bounded SSM entry points are indexed in
[`experiments/README.md`](../experiments/README.md). The synthetic P0 launcher
remains the qualification path; a separate synthetic-only `T3c` composite
`T-P2` screen tests the gain/time directions but is decision-ineligible. The
measured reconstruction/null, fit-only identifiability, and fit-only
three-session LOSO launchers are nonprotected development diagnostics with
their own executable contracts. An array-free `T3c` admission gate checks
whether hierarchical composite fitting may start. Reconstruction fits on
subjects 01--18 and applies frozen objects to subjects 19--23.
Identifiability and LOSO use only subjects 01--18 and load no 19--23 arrays;
these retained diagnostic contracts excluded subjects 24--29. Their outputs are
exploratory; they are not clean truth, teacher qualification, or a
physical-teacher claim.
For current and future public-data preparation, the
[2026-09-17 public-subject policy](DATA_CONTRACT.md#current-public-subject-policy--2026-09-17)
retires the inherited Single-Trial subject embargo. References below to closed
subjects describe the bounded retained experiment contracts, not a project-wide
restriction on public cache construction or future experiment split selection.
Separate comparison protocols retain their own declared evaluation boundaries.

The measured diagnostic does not replace the synthetic qualification gates or
authorize tokenizer promotion. Any future measured confirmation or physical
teacher qualification still requires the unresolved margins, primary
estimand, calibration, and compute decisions below to be frozen in a separate
contract.

## Reading guide

Section order is not an execution queue. Use the [registry view](PROJECT_STATUS.md)
for state and next actions, the
[entrypoint/config/test map](../experiments/README.md#entrypoint-config-and-test-map)
for implemented commands, and the [result index](../experiments/RESULTS_INDEX.md#find-results-by-question)
for evidence. A design section does not establish that an executable contract or
measured result exists.

| Reading task | Sections |
| --- | --- |
| Arrange tokenizer training and check physiological-state acceptance item by item | [TK-STATE-v1：训练与验收安排](#tk-state-v1) |
| Test the added value of physical structure beyond task accuracy | [Physical-model contribution and competing explanations](#物理模型独特贡献与非任务证据设计2026-11-04) |
| Design EEG voltage separation and falsifiable source-token evidence | [Voltage separation with independent intervention anchors](#eeg-电压域分解与独立干预验证设计2026-11-03) |
| Explain reconstruction versus physiological recovery | [Four independent diagnostics](#拟合观测信息与生理恢复的四套独立诊断2026-10-21) |
| Reacquire complete downstream metrics and repair Visual/REFED | [DOWNSTREAM-REBENCH-v2](#downstream-rebench-v2); [retained v1 design](#downstream-rebench-v1) and historical comparison contracts remain separate |
| Local spatial structure and independent dynamics | [Spatial observation design](#局部多通道观测与空间动态维数2026-10-16) |
| Dataset extensions and conditional tokenizer probes | [REFED extension](#refed-公开实验补齐与四数据集比较2026-10-13), [TK-FUNCTIONAL](#条件功能监督与-hard-token-开发矩阵2026-10-10) |
| Output support, parameter stability and measurement semantics | [Output support](#ssm-输出支持范围连续状态与跨模态约束2026-10-10), [parameter stability](#参数稳定性共同内点与拟合改进2026-10-10), [observation semantics](#观测物理量事件锚点与竞争机制2026-10-09) |
| Response, regional and component semantics | [Response design](#响应动力学与连续语义原型2026-10-02), [regional modes](#区域与谱模式共享驱动检验2026-10-02), [attribution](#结构性成分归属与分级语义检验2026-10-02) |
| Fixed-structure physiology and shared-driver reconstruction | [2026-09-28 design](#固定结构的跨数据集生理语义检验2026-09-28-设计), [reconstruction screen](#共享驱动重建与参数自由度筛查2026-09-26) |
| Dataset processing and SSM fit revision | [Joint revision](#数据统一化与-ssm-拟合的联合修订2026-09-15) |
| Observation-contract design | [Observation repair follow-up](#观测合同修复后的下一轮实验设计2026-09-10) |
| Retained overnight and observation protocols | [Overnight v2](#bounded-overnight-ssm-diagnostics-retained-v2-contract), [flow/mask bridge](#step5-flow-domain-and-mask-specific-observation-experiment), [repair v1](#step5-observation-contract-repair-and-regression-retained-v1), [observation diagnostic](#step5-observation-adaptation-diagnostic-retained-v1-contract) |
| Synthetic calibration and staged qualification | [Step5A0 localization](#step5a0-inference-consistency-diagnostic), [Step5 stages](#full-step5-staged-continuation) |
| Original qualification design | [Question](#fixed-question-and-decision-target), [flow](#experiment-flow), [estimand and statistics](#common-estimand-and-statistical-contract) |
| Qualification and tokenizer design | [P0](#p0-software-and-synthetic-qualification), [teacher selection](#t-physical-teacher-selection), [tokenizer](#bq-source-and-observation-tokenizer), [coupling prior](#c-coupling-prior-return) |
| Implementation and remaining qualification decisions | [Code ownership](#code-ownership-for-later-implementation), [unresolved qualification decisions](#unresolved-before-measured-qualification-or-confirmation) |
| Historical boundaries | [Side paths](#side-path-experiments-without-workflow-sprawl), [lifecycle boundary](#historical-lifecycle-boundary) |

<a id="tk-state-v1"></a>
## TK-STATE-v1：Tokenizer 训练与生理状态验收安排（2026-11-12）

本节落实本次用户方案，作为下一代 tokenizer 的训练、选择和验收设计。
**保留 30 秒可见上下文、每位置每分支每 2 秒输出一个 token；核心目标是形成
可迁移的情境化生理状态族，并取得至少一项可复现的时间组织或空间表型。**
S/O 功能分工与同码率对照用于检验物理参照的贡献，跨模态响应组织为最高
挑战目标。200 Hz EEG 电压波形保真、Hb 波形保真、整数重放和码本健康构成工程基础。

**用户确定的频率约束：所有数据集的 EEG 监督统一使用 200 Hz。** EEG 的
X、S、O 经同一抗混叠、同相位的重采样算子进入训练与评分，模型输入和输出
均为 200 Hz；30 秒为 6,000 点，每个 2 秒 patch 为 400 点。源缓存继续保留
各自原生采样率及来源身份，转换发生在 tokenizer 的版本化数据适配中。

本节是设计与检查依据；执行状态和科学判定仍由
[`research_state/registry.json`](../research_state/registry.json)持有，数值由实际
运行记录持有。下列方框是待核对的交付要求，不另行维护一份运行进度表。
本次工作落实文档；实测训练、冻结评价和保护评价按各自后续执行范围开展。
历史试验的配置、结果和失败记录保留原身份。

### 1. 对象、信息支持与可复用入口

每个 token 以一个 2 秒片段为时间锚点，由同一位置的 30 秒观测共同决定。
EEG 每电极、fNIRS 每原生位置分别赋码，HbO/HbR 联合输入；四类词表为
EEG-S、EEG-O、Hb-S、Hb-O。两个模态独立前向，同一分支的解码器读取
该位置全部 15 个整数 ID。一个状态族可以包含多个承担相位、极性或细节的
ID；状态持续时间由实际序列估计，2 秒赋码步长的定位精度另行核验。

沿用如下计算关系，其中 M 表示 EEG 或 Hb，b 表示 S 或 O，i 为时间槽位：

\[
z^{M,b}_i=Q^{M,b}(E^{M,b}_i(X^M_{1:15})),\qquad
\widehat Y^{M,b}_{1:15}=D^{M,b}(z^{M,b}_{1:15}),\qquad
\widehat X^M=\widehat S^M+\widehat O^M.
\]

模型看到本模态观测 X；缓存 S/O 只作监督与评分。任务标签、另一模态、
当前窗口的 teacher 参数及连续潜变量均不进入整数消费者。固定训练尺度、
采样率和位置身份属于声明的接口元数据；逐窗均值、幅度及 teacher 估计不能
作为解码 sidecar。30 秒是网络可见支持，上游滤波、参考与个体无标签校准的
真实支持另列，基础结果按离线条件状态解释。

本节“原始观测”指缓存中分解前的 X，继承数据合同中的预处理。整数重建
在该缓存及冻结训练归一化坐标中验收；物理单位的展示换算单列其参考和
尺度来源，不能把逐窗逆变换所需的信息暗中加入 token 解码接口。

| 责任 | 已核对的 owner 与本轮使用方式 |
| --- | --- |
| 数据与条件目标 | [长窗 S/O 数据合同](DATA_CONTRACT.md#full-long-window-so-supervision-cache--2026-11-11)及 [long_so_supervision_v2.yaml](../experiments/configs/physiology_semantic_tokenizer/long_so_supervision_v2.yaml)；使用完成并核验的 `data/cache/physiology_so_supervision_v2/`，不重新拟合 teacher |
| 读取接口 | [SOSupervisionDataset](../src/data/so_supervision_cache.py)；显式指定 split，在任何数组读取前核对允许身份和分区。该类可读取多个公开 split，实验调用方须实施本轮边界 |
| 时序模型与整数解码 | [GlobalLatentComponentTokenizer](../src/tokenizers/dual_component_tokenizer.py)；复用 30 秒编码、独立 S/O 解码及 `decode_ids` |
| 训练与分析入口 | 扩展现有 [run_dual_component_tokenizer.py](../experiments/scripts/run_dual_component_tokenizer.py) 中最接近的训练/评价函数，新增版本合同；不覆盖历史 latent-health 合同或另建转发 launcher |
| 回归与新合同测试 | [入口/配置/测试索引](../experiments/README.md#semantic-and-tokenizer-development)；重点复用 `test_dual_component_latent_health.py`、`test_so_supervision_cache.py`，补足新适配、split、评分与合成控制测试 |
| 新运行输出 | `experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/<fresh_tk_state_run>/`；解析配置、身份计划、训练、冻结对象和 A–F 结果同属一个运行 |

现有 latent-health 实测合同限定于 Hb 的历史训练/验证目标，EEG 为合成接口；
它可提供实现起点。新四库长窗目标、统一 200 Hz EEG 适配和 A–F 验收须在新合同中
实现，拟使用新配置 `experiments/configs/physiology_semantic_tokenizer/dual_component_state_v1.yaml`
（P0 实施时创建）。缓存已接受的失败窗口与合成后功率诊断按原记录保留，使用有效性掩码
训练，不把已结项的缓存修复重新列为本轮前置任务。

### 2. 先固定人口、窗口与训练/选择/评价边界

- [ ] **P0.1 来源与分母。** 从完成 manifest 固定数据集、被试、记录、父窗、
  原生位置与绝对样本索引。计划数、可用数、源失败、输入失败、训练失败和
  分析不可用分别记录；失败 teacher 位置仍可有有效 X，不能用零值补成 S/O。
  主模型与普通量化的主比较使用相同 X/S/O 可用面板；仅 X 可用的扩展单列。
- [ ] **P0.2 划分。** 继承 cache 的 train/validation/test 被试身份，同一被试的
  所有记录和重叠父窗保持同一 split。train 拟合权重、码本、归一化、状态族
  和读出系数；validation 选择容量、checkpoint、有限超参数及实际效应容差。
  冻结后再评价 test。缓存生产时允许公开 test 的无标签个体适配，并不自动
  开放 tokenizer 的 test 评价；沿用其校准支持并在结果中说明。
- [ ] **P0.3 确认集是否仍独立。** 读取评价数组前，根据既有使用记录核对候选
  被试是否曾参与本轮选择或相关目标导向的调参。已用于选择的集合继续作为
  development；若无未用的确认被试，报告开发结论，并将独立复现留待新的
  明确面板。不能通过重命名 split 恢复独立性。另有协议保护的数据遵守该
  协议和该次明确授权，使用一个拥有边界的读取控制入口。
- [ ] **P0.4 坐标与掩码。** EEG 保留电压单位，在统一 200 Hz 视图上训练和评分；
  Hb 保留 10 Hz、双色团和一次 subject scale。原生单位未知时沿用未知标记。
  输入有效、teacher 有效、source 评分三个 mask 分开传递，保留最后未评分
  bin。所有臂在同一真实支持上评分，变换只拟合训练分区。
- [ ] **P0.5 30 秒切片。** 在每个父窗内按 2 秒网格选完整 30 秒输入，使用
  绝对整数索引对齐模态，保留实际短 REFED 父窗。少于 30 秒或含不支持区间
  的输入记为不可用；不跨记录、拼接点或父目标边界补齐。训练按数据集、
  被试、记录、父窗、位置分层抽样，避免长记录或电极多的数据集支配优化。
- [ ] **P0.6 统一 200 Hz 接口。** 按 cache metadata 核验源速率；已是 200 Hz
  的 EEG 保持不变，其余 EEG 的 X/S/O 使用同一固定线性抗混叠算子转换到
  200 Hz，保持 `X=S+O`、原有时间锚点和通道身份。统一输入/输出投影及
  400 点 EEG patch；所有对照复用同一转换。Hb patch 仍为 20 点×2 色团。
  记录源/目标速率、滤波核、边界处理和 mask 支持；分开的训练/留出时间块
  独立重采样，避免滤波跨越其边界。检查 200 Hz 恒等路径、其他源速率的
  抗混叠、时钟/shape、S/O 闭合及整数重放；保真结论对应统一后的 200 Hz
  波形。原生缓存保留，不为此重算 teacher 或覆盖既有数组。
- [ ] **P0.7 去重与条件目标。** 同一原生 EEG 被两个 Visual Probe 引用时按
  cache 既有身份去重。不同父窗的重叠区保留各自条件 S/O，训练采样权重
  避免重复放大；原始轨迹验收中同一物理时间只计一次。评分方案预先固定：
  保真以父窗内不重叠完整 30 秒块为基本单位，尾段与失败另记；时序分析使用
  移动窗口中央槽位，重叠父窗按最早可完整覆盖该输入的父窗唯一归属。

### 3. 固定训练对照与有限预算

第一轮以已有全局时序架构检验目标是否已经产生状态组织。逐位置模型中的
空间拓扑来自多个位置状态的联合分析；只有后述空间交互版本才检验单个 token
是否利用整个观测构型作判断。

| 臂 | 输入、监督与离散出口 | 主要回答 |
| --- | --- | --- |
| **Q-SO 主模型** | 30 秒本模态 X；两支分别监督缓存 S/O，独立整数解码后相加 | 物理条件参照是否组织了有限容量中的状态信息 |
| **Q-X 普通量化** | 同一 X、双码本、两条同尺寸独立解码路径；仅监督两输出之和重建 X | 同码率、同上下文、同训练预算的波形压缩能达到什么程度 |
| **C-SO 连续学生** | 同一 X、同 S/O 目标和主干，移除硬量化瓶颈；连续潜变量仅作参照 | 当前学生架构的目标可学习程度，以及 Q-SO 增加的量化损失 |

Q-X 两路仅称码本 1/2，以合并表示和总重建与 Q-SO 比较；不将任一路事后
命名为 S。C-SO 的误差是已拟合连续参照，不能直接当作理论不可约误差。
S/O 目标中依赖另一模态和长窗的部分，通过它的留出误差与量化增量分开描述。

- [ ] **P1.1 容量。** 初始固定每分支 K∈{128,512}、种子 {17,29,43}。
  每模态每位置每 2 秒传两个 ID，名义容量为
  `log2(K_S)+log2(K_O)` bit，即两档各 14/18 bit，折合 7/9 bit/s；
  四词表不意味着全空间每 2 秒仅四个 ID。另报实际熵和序列存储量。
- [ ] **P1.2 矩阵。** 每模态为 Q-SO×2 容量、Q-X×2 容量、C-SO×1，
  各 3 个种子；两模态共 **30 个训练终态**。这是计划模型数，人口和窗口数
  由 P0 身份计划持有。共享预热的成本单列，不重复记成独立科学重复。
- [ ] **P1.3 训练目标。** Q-SO/C-SO 对 S/O、HbO/HbR 分别以冻结训练目标
  方差归一化 MSE 后等权，方差下限在合成/训练校准中固定；Q-X 对 X 使用
  同一定义的训练观测尺度。Q-SO/Q-X 加相同量化 commitment 项。总重建的
  评分统一使用训练 X 尺度，不能将各自训练 loss 直接比较为保真优劣。
- [ ] **P1.4 等预算。** 同种子、同模态共享身份抽样顺序与训练曝光量；
  Q-SO 与 C-SO 可共享同目标连续预热，Q-X 用自己的 X 目标等步数预热。
  码本只从训练 latent 初始化与复活，评价时冻结。学习率、有效 batch、
  warmup、总更新数、checkpoint 间隔及允许恢复次数在 pilot 后写入解析
  合同，全部候选采用相同曝光上限；记录参数量、更新数、样本量和设备时间。
- [ ] **P1.5 有界 pilot。** 每模态在元数据确定的训练小面板上完成 C-SO、
  K128 Q-SO、K128 Q-X 的 seed17 smoke，不参与最终表。先测固定 100 个
  更新的吞吐/显存，再完成一个预定曝光周期，检查梯度、非有限值及损失下降。
  资源和数值失败修正通过后冻结正式预算；后续仅恢复同一任务，不按验收结果
  追加训练。先检验合成已知状态和 task-locked 阴性对照可被正确识别。
- [ ] **P1.6 训练监测。** 保存 S/O 与总 X 的学习曲线、每库/每支用码、有效
  码数、最大占用、复活历史、失败与整数回放。健康检查用于诊断塌缩；状态
  不必等频，历史健康试验的 95% 活跃码比例等阈值不直接充当本轮生理验收阈值。
- [ ] **P1.7 候选选择。** 每臂先按 validation 对应重建目标选择 checkpoint；
  两种容量在三个种子汇总后比较，先满足保真护栏，再按 A 的预定原始观测
  读出误差在各模态内选择共同比较容量；在冻结的等效容差内优先 K128。Q-SO/Q-X 的
  主比较必须同 K；另一档为码率敏感性分析。所有种子保留，不挑最好种子。

### 4. 冻结状态族、读出与评分方式

状态族定义先于独立评价。在训练记录中统计每个 ID 的原始观测描述分布，
并列缓存 S/O 与整数重建的同名描述；只有第一层支持真实观测中的状态解释。
EEG 描述包含固定频带功率、振荡包络及其持续/增强/消退特征；Hb 包含双色团
水平、趋势、曲率、联合变化和恢复过程。各描述的单位、滤波、时间支持、
参考和训练标准化必须逐项写入解析合同，慢过程描述使用声明的较长支持。

- [ ] **P2.1 状态族。** 以训练 ID 条件描述构造固定聚类映射；先固定一个
  算法和有限族数候选（4/8/16），在 train 内按被试交叉验证选取复现性与
  支持度满足要求的最小族数。最低被试支持和未知 ID 的处理在此冻结。
  稀有码保留原 ID，不能为了通过而反复合并；低支持族标为未定。
  各种子独立建族，再仅用训练描述匹配族身份，留出数据不重新对齐。
- [ ] **P2.2 简单读出。** 主读出为当前锚点两个 ID 的 one-hot 加固定线性
  ridge，Q-SO 与 Q-X 输入维度和调参网格一致；训练拟合、validation 选
  正则。单支、单族及固定量化嵌入读出作解释。完整 15-ID 读出与先解码
  再算特征单列为序列信息保留，不能替代当前 ID/状态族的可读性验收。
- [ ] **P2.3 避免循环。** 另列训练中直接监督的量和独立验收量。状态发现
  以冻结族在留出原始轨迹、独立事件和未直接监督的时空结构中复现为依据。
  后续若增加锚点头，其直接监督的摘要移到约束完成检查，不再作为独立发现。
- [ ] **P2.4 共同比较。** 原始观测的固定传统特征、仅局部 2 秒描述、位置/
  槽位/整体幅度/任务相位分别构成低容量解释基线。事件标签只供冻结后的
  读出或匹配对照，不训练 tokenizer、码本或状态族。主结果保持各模态和
  数据集可分辨，空间比较只用真实可比的覆盖位置与几何。

**指标与统计统一如下。** 对波形 Y，先在固定有效样本上计算
`NMSE = mean((Y_hat−Y)^2) / sigma_train(Y)^2`，NRMSE 为其平方根；
X、S、O 的分母各自固定，另给原单位 RMSE、相关、均值偏差和幅度误差。
低方差分母下限和无效相关在训练阶段确定，并报数量。用于跨模型保真护栏的
总 X 一律使用相同 `sigma_train(X)`。EEG 分开给 200 Hz 电压波形与谱/包络，HbO/HbR
分开报告；总和准确不能掩盖 S/O 相互抵消。S/O 功率分析保留交叉项。

窗口/位置先在记录内汇总，记录在被试内等权，种子再在被试内汇总；最终
数据集内被试等权，四库宏平均为各库等权平均。主检验在共同可用配对上计算，
同时列全计划分母、覆盖率及失败差，不能用成功子集替换完整验收。种子、
重叠窗口、时间点和通道均不增加独立被试数。主误差改善统一定义为
`Delta = error(Q-X) − error(Q-SO)`，正值表示主模型改善。

区间默认在数据集内整被试配对重采样 2,000 次，再等权合并数据集，报告
冻结模型条件下的 95% 百分位区间；每种子的方向和效应另列。需要显著性
判断时采用预先固定的被试级配对检验，校正后显著性水平为 0.05。
EEG/Hb 的 A 主比较作为一个检验族
作 Holm 校正；B–F 各自的多描述、多状态、多位置比较在冻结时定义检验族
并校正，未登记的搜索列为探索。置信区间跨零记为方向未确定。
优越性同时核对预定实际效应和校正检验；X 保真非劣性用
`error(Q-SO)−error(Q-X)` 的区间上界与冻结非劣容差比较，具体误差尺度在
P3 固定。“未检出差异”不替代非劣性判定。

### 5. A–F 验收卡：每项均须保留对照、完整分母与失败

**A．离散状态的生理含义能否迁移（主终点）。**

- [ ] 训练定义 ID/族→观测描述分布与简单读出，冻结后在未见被试/记录评分。
  主量为当前两个 ID 对预定原始轨迹描述的标准化 MSE，分别给 EEG、Hb。
  EEG 首选中央锚点的包络水平及前后变化；Hb 首选双色团水平与前后趋势；
  具体支持与各维等权规则在 P2 冻结，避免评价后挑选最好的频带或色团。
- [ ] 比较 Q-SO、同 K Q-X、C-SO 和训练均值/传统特征基线，并列原始观测、
  teacher、整数重建三个层次；记录族内离散度、族间差异及跨被试/种子对应。
- [ ] 通过要求：两个模态各至少一个有足够被试支持的状态族在留出原始观测
  中复现预定描述差异、简单读出优于均值基线；至少一个模态的预定 A 主比较
  对 Q-X 达到冻结的实际效应门槛且校正检验成立。另一模态保持保真并报告
  实际结果，不将“至少一个”扩写成两个模态均优越。

**B．30 秒上下文是否增加有依据的状态区分（核心）。**

- [ ] 在自然连续记录中，按训练冻结的幅度、频谱/斜率匹配规则寻找局部
  2 秒描述相近、较长轨迹不同的片段；预先规定 caliper、供体资格和不足
  匹配的处理。先验证自然匹配，再把固定中央片段的人工上下文替换作为诊断。
- [ ] 比较“局部描述＋幅度/位置/相位”等混杂量与“再加冻结状态族”的低容量
  读出，对持续、转折或恢复描述计算留出误差增益；匹配和读出跨被试复现。
  采用记录内分块或同任务阶段的整段供体对照，保留自相关与窗口重叠。
- [ ] 通过要求：EEG、Hb 各至少一种预定情境区分达到冻结的增益与复现门槛，
  且超出槽位、整体幅度和任务时间表的解释。ID 改变比例只描述上下文依赖，
  不单独决定成败。需要重训 2 秒模型时作为后续独立消融，不替换本轮主输入。

**C．同一过程在不同裁窗下是否有相容解释（核心稳健性）。**

- [ ] 在同一个 90 秒父目标窗内移动 30 秒输入，使同一真实片段覆盖边缘、
  中央和不同槽位；短父窗仅用实际可覆盖位置，并报告可比较分母。先固定
  teacher 身份，避免把独立父窗条件目标的差异计入 tokenizer 不稳定。
- [ ] 核对精确 ID 一致率、状态族一致率、描述分布距离、S/O 重建差异及
  边界偏移（秒）；对同一物理片段等权，按槽位/距边缘距离分层。已知合成
  状态边界提供定位误差，实测独立事件提供事件对齐；两种参照分开。
- [ ] 通过要求：族含义和重建在冻结容差内相容，主要边界跟随实际过程，
  相比保留时间结构的槽位对照具有一致优势。允许新上下文系统修正判断；
  精确 ID 不变并非硬条件。时序主分析固定中央第 8 个槽位（从 1 计数），
  每移动 2 秒取一个标签，边缘缺少完整上下文的时间另计覆盖损失。

**D．S/O 是否具有可重复的功能分工（核心贡献）。**

- [ ] 固定至少一种已知加性扰动与一种应保留的变化，分别评价稳定性和
  敏感性。使用固定训练尺度设定扰动幅度，保存原输入、增量、位置、时间
  支持与随机种子；天然事件/EOG 只在既有允许支持内提供独立依据。
- [ ] 冻结 tokenizer、teacher 和所有尺度，对原输入/扰动输入分别编码。
  计算 `Delta S_hat`、`Delta O_hat` 对已知增量的有符号投影和归一化误差，
  并检验 `Delta X_hat` 恢复与未扰动结构保持。模型直接估计两支，不能把
  人工定义 `O'=O+扰动` 的恒等式当作成功；需改 teacher 的诊断另行版本化。
- [ ] 对独立事件或耦合合成变化核验相应结构没有被统一压低；O 中的稳定
  生理相关状态作为正面候选保留。人工频段注入按已知信号变化解释，生成器
  真值、缓存条件估计和实测目标各有名称。
- [ ] 通过要求：至少一种预定变化上两路的差异达到冻结效应并跨被试/种子
  重复，同时满足有意义变化的保留护栏。Q-X 以合并表示的扰动响应为参照；
  单一路的“更稳定”不能替代双向功能检查。

**E．码族是否具有稳定的解码作用（功能核验）。**

- [ ] 在同 split、同模态/位置和背景相容的真实序列间选供体，只替换一个
  ID 或预定短片段。匹配当前总幅度和外围背景，不把待检验的状态变化本身
  匹配掉；匹配规则在训练上固定。对所有臂使用相同供体资格，缺供体记缺失。
- [ ] 用 `decode_ids` 重放，以冻结训练尺度标准化描述 phi，计算贡献矩阵
  `C[i,j]=mean(||phi(Yhat_i(Z))−phi(Yhat_i(Z[j←donor]))||²)`；同时保存
  有符号描述变化、总量、时间扩散和跨背景方差。无替换、同 ID、同族与异族
  供体为对照，验证只改 S 的 ID 不改变 O 的解码，反之亦然。
- [ ] 通过要求：预定族替换的描述方向和作用范围在留出背景重复，并与 A/B
  的族解释相容；允许稳定的远程作用，不要求贡献矩阵完全对角。报告的是
  表示对解码输出的功能作用，身体系统干预仍由独立生理证据判断。

**F．状态组织能否在独立原始记录中复现（主要发现目标）。**

- [ ] 训练中提出有限候选：空间占据图、连续驻留时长、进入/退出/回返路径、
  短状态序列或独立事件关联。每模态最多登记一个主要时序候选和一个主要
  空间候选；在评价前固定方向、位置、事件、统计量、对照和候选总数。
- [ ] 使用唯一物理时间轨迹，驻留时长为连续 2 秒格数乘 2 秒；遇缺失或
  父窗切换即断开并标记删失，不计跨断点转移。转换率以有效驻留时间为分母，
  占据率以有效时间为分母。空间分析保留原生覆盖和位置，跨库只比较明确
  对应的区域；逐位置模型的结果表述为状态配置，而非空间联合推断。
- [ ] 通过要求：至少一个预定时序/空间模式在独立原始记录及独立被试中
  复现，达到冻结效应门槛并通过相应多重比较控制、幅度/任务相位/裁窗对照。
  同被试留出记录仅支持跨记录重复；报告另给跨被试证据。模式只存在于
  teacher 或 decoder 时，结论留在物理参照表示层，不计入原始观测表型发现。

### 6. 冻结后挑战：状态条件的跨模态响应组织

- [ ] **G1 同窗关系。** 在真实同步且几何可对应的位置，训练确定 EEG 状态族
  与 Hb 响应阶段/转换路径的候选，冻结 lag 网格、状态条件、位置与多重比较。
  对照用同任务阶段的整记录/整试次配对替换或合法长块移位，保留单模态
  自相关、窗口重叠和任务结构；换配对后重新完成相同 token 化/读出步骤。
  避免逐 token 独立洗牌。双向编码的 lag 峰按状态组织解释，并用原始轨迹
  与已知时延合成检查定位偏移。
- [ ] **G2 窗口外预测。** 可用冻结 30 秒 token 的小型读出预测整段输入结束
  之后的 Hb/EEG 目标，与本模态历史、传统特征、Q-X 和 C-SO 同支持比较。
  开始前核对滤波、参考和无标签校准的完整依赖；输入预处理若使用了目标
  时段，只能报告离线读出，严格预测须另有只使用历史的版本化预处理。
- [ ] **G3 判读。** 被试级独立复现、条件对照和实际增益共同支持跨模态结果；
  同窗共现峰不独立给出生理时延。G 的阴性或不确定结果保留，不抵消已经
  成立的 A–F 状态发现；本轮无需大型预训练或另建动力学生成模型。

### 7. 验收阈值冻结与最小修订规则

在 **P3 冻结检查**中，把下表中的每一项变成解析合同里的具体值及统计方法。
效应门槛由训练/validation 的测量尺度、重测/合成误差和可接受保真损失确定，
保留依据；不能在查看确认集后修改。这里不把尚未校准的数字写成生理标准。
若确认集规模无法支持门槛判定，提前登记为精度受限，仍完整报告效应与区间。

| 检查项 | 冻结内容 | 最终判定 |
| --- | --- | --- |
| 整数出口 | 同设备 hard forward 与 ID replay 的绝对/相对数值容差、跨设备容差；由合成误差校准 | 全部允许接口通过，S/O 隔离且无连续旁路 |
| 绝对与相对保真 | EEG、HbO、HbR 的 X/S/O 最大误差，Q-SO 对 Q-X 的 X 非劣容差；相同训练分母 | 绝对标准与相对护栏同时满足；按数据集/模态展示，不靠宏平均掩盖失败 |
| 核心状态 A/B | 每模态状态族最小被试支持、原始描述读出及情境增益门槛、跨种子复现要求 | 两模态均有可迁移的情境状态族；预定 A 主比较至少一项优于同码率 Q-X |
| 稳健性与功能 C/D/E | 裁窗相容容差、边界误差、扰动分工及保留容差、供体方向一致性 | 相应核心检查通过；不能用保真改善抵消功能或相容性失败 |
| 发现 F | 至多四个候选的预期方向、实际效应、检验族、原始观测支持与复现人口 | 至少一个预定时空模式在独立原始记录/被试成立 |
| 覆盖与失败 | 最低有效独立被试/记录数、各臂允许失败差及不可用规则 | 报全分母；不足时记未确定或失败，不能按成功子集宣告全部通过 |

**主验收结论**同时要求：整数与保真成立；EEG/Hb 各有情境状态族；S/O
至少一种稳定功能分工；至少一项独立时空复现；预定指标优于同码率普通量化。
A–F 完成检查与 A–F 科学通过分别记录。部分通过时明确已支持的层级和失败项，
例如“保真成立、状态族迁移未确定”，而不是把工程完成标作科学验收完成。

仅在 validation 诊断指出具体缺口时启用以下修订；每次只改变一个因素，
原基线和失败保留，修订后重新冻结再进入确认评价。

| 触发证据 | 最小动作 | 必须配套的比较与停止规则 |
| --- | --- | --- |
| C-SO 已有较大留出误差 | 定位目标尺度、均值/幅度和本模态可实现性；报告连续参照 | 先结束该配置结论，不能靠扩大码本掩盖输入/目标困难；改变目标另立版本 |
| C-SO 可用而硬量化保真失败 | 在已定 K128/K512 中判断码率损失和用码问题 | 完成既定容量矩阵即停止，不自动扩展码率搜索 |
| 波形通过，单 ID 的 A/B/E 解释性不足 | 只加 `rhat_i=g(e[z_i])` 的低容量锚点头，预测少量当前 S/O 摘要；仅读当前 ID 嵌入 | 保留全局解码、原生电压和原词表；原 Q-X 保留，另给等容量的原始 X 摘要头对照，明确两种监督来源；训练摘要不再计为独立发现，最多一个冻结权重方案 |
| 需要检验空间联合判定，且已有可比几何和空间候选 | 仅增加一次模态内空间交互，保留四词表、15 个时间槽和两模态独立前向 | 与逐位置版做单变量比较，Q-X 同样适配；新增参数/资源单列，不同时引入全局状态码本或脑源映射 |
| 确认评价未通过 | 完整报告原冻结候选及失败原因 | 本轮结束；改阈值、改族或改模型属于下一版本，已查看评价集按其使用历史处理 |

### 8. 执行顺序、检查出口与交付

| 顺序 | 开始条件与工作 | 本阶段交付/结束条件 |
| --- | --- | --- |
| **P0 合同与接口** | 核对第 2 节；在既有 runner 实现新版本合同和统一 200 Hz EEG 接口 | 身份/split/mask/时钟/无旁路测试及合成阳性、阴性对照通过；完整输入与失败计划 |
| **P1 训练基线** | 软件检查、dry-run、有限 smoke 和资源 pilot；冻结正式训练预算 | 固定 30 个模型均有终态，完整学习曲线、C-SO 对照、整数回放及每库保真/健康 |
| **P2 开发验收** | 仅 train/validation 建族、拟合简单读出并执行 A–F 开发分析 | 按预定容量规则选候选；决定是否触发一次局部修订；失败原因可定位 |
| **P3 冻结** | 固定模型、三个种子、映射、读出、阈值、统计、候选和评价身份 | 一份解析合同与冻结对象清单，无未填写的主要容差；确认集使用历史核对完成 |
| **P4 独立验收** | 满足对应数据读取范围后，对冻结对象一次性执行 A–F | 逐项判断、被试配对表、全分母及原始/teacher/重建对照；达到或未达到主目标均结束 |
| **P5 挑战与汇总** | 按预先登记且数据支持的 G1/G2 执行，不反调 A–F | 分层科学结论、与 Q-X/C-SO 的差异、负结果、适用范围与后续用途 |

- [ ] **运行前资源。** 检查 CPU affinity/quota、物理核、内存、GPU、存储与
  竞争作业，实测单/多 worker 吞吐再确定并发、在途任务和数值线程。
  所有正式训练在持久 supervisor 下执行，验证 user-systemd linger 与
  启动 shell 退出后的存活；命令、环境、日志、进度、预计完成时间与恢复点
  写入同一运行。每组实验核验后在对话中汇报数值、基线差异和结论。
- [ ] **证据包。** 沿用运行内解析配置与 `plan.json` 保存完整身份和预定
  比较；选择规则及冻结对象由同一 `selection.json` 持有。逐任务记录保存
  全部种子/失败；`summary.json` 汇总声明端点并链接 A–G 的逐窗、逐记录、
  逐被试表。冻结映射、最小审计数组及必要 checkpoint 与该运行一起保留。
- [ ] **图表。** 至少交付：三臂保真与量化损失；A 的原始/teacher/重建族
  描述及被试配对差；B 的匹配轨迹；C 的同片段跨槽位曲线；D 的扰动/保留
  双向响应；E 的贡献矩阵与有符号作用；F 的原始记录时空复现。
  G 有数据时增加真实配对与结构保持对照。例图按预定规则选择并给分母。
- [ ] **报告与代码验证。** 报告逐项解释问题、输入支持、指标分母、比较方向、
  独立单元及失败；若导出 PDF，完整图以 PNG 嵌入，正文和表格保持可检索。
  运行直接相关测试、`pytest --collect-only -q`、`git diff --check`；登记
  状态变化时更新 registry 后执行 `project_state.py validate` 和 `render`。

后续首个实施单元为 P0/P1 的合同、读取适配、合成检查与有界 pilot。阶段
推进依据本节检查出口和实际运行证据，不由文档中的方框或历史通过状态替代。

### 首批有界训练 pilot

首批执行由 [dual_component_state_v1.yaml](../experiments/configs/physiology_semantic_tokenizer/dual_component_state_v1.yaml)
持有，沿用既有 runner 的 `state-*` 阶段。仅选择各库字典序第一名训练被试的
第一个完整 90 秒原生记录窗，并增加同一 REFED 被试的第一个短窗；选择只读
元数据，失败保留。全部原生位置进入计划，三臂使用相同有效 X/S/O 位置。
按 2 秒网格留下最后一个完整 30 秒块，其前缀用于训练及尺度估计；这是同一
训练被试、同一离线父目标下的输入留出检查。EEG X/S/O 在两个支持块内分别
转换为 200 Hz，保留滤波支持与无效掩码，Hb 保持 10 Hz。

每模态以 seed17 比较连续 S/O、K128 S/O 和 K128 原始波形量化。分别进行
100 步 S/O 与 X 预热，再各继续 400 步；连续与离散 S/O 共享同一预热，三个
最终模型的总曝光均为 500 步。先在持久监督器中比较每 GPU 1/2 个 worker
的 100 步吞吐，按实际速度与显存选择并发，再完成六个 pilot 模型。保留
学习曲线、梯度、整数回放、分库 S/O/X 误差、短窗支持、用码及全部失败。
本批结束于工程可行性和正式预算建议，不自动扩展到三种子完整训练或 A–F 验收。

<a id="tk-state-reconstruction-v1"></a>
### 重建改良与历史核对：TK-STATE-RECON-v1（2026-11-13）

本轮依用户要求，将 EEG 与 fNIRS 的可见重建改良设为直接工程目标，同时保留
码本健康检查。执行合同为
[`dual_component_state_refinement_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_state_refinement_v1.yaml)。
固定 30 秒可见上下文、每位置每分支每 2 秒一个整数 ID、EEG 200 Hz、Hb 10 Hz；
解码器只读取本分支整数序列，不接收当前窗均值、幅度或原始 X 的旁路。

**已核对的历史经验。** 下表的不同代际保留原目标、尺度和码率，数值用于选择
本轮设计，跨代际绝对误差不充当同条件优劣比较。

| 保留证据 | 核对结果 | 本轮采用的结论 |
| --- | --- | --- |
| [E1 健康审计](physiology_semantic_tokenizer/analysis/20260722_E1_TOKENIZER_OPTIMIZATION_HEALTH_AND_COUPLING_REPORT.md) | 有效复活、训练初始化和停止复活后的保留提高了占用；高占用仍伴随低谱参与维数 | 记录完整用码分布与复活历史，保留足够长的无复活训练段 |
| [2026-10-28 三轮 refinement](../experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261028_refine_v3/refinement_summary.json) | 上下文和两层波形 VQ 改善总波形，但 EEG 目标是代理量，且每模态每 2 秒为 28 bit；分支门未全通过 | 总重建与 S/O 分支分别核验，所有离散容量均计入码率 |
| [2026-10-29 电压合同](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v2.yaml)及[结果](../experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261029_voltage_v2/voltage_summary.json) | EEG 每 0.1 秒每分支 4 层 K128，合计 560 bit/s，是当前双支 K128 的 80 倍；Hb 固定合成改善 0.1 秒接缝，对 0.5 秒偏移改善有限 | 不用高码率历史重建承诺当前低码率表现；将整条采样网格作为解码对象，并检查宽尺度波形而非只看接缝 |
| [2026-11-05 全局潜在量化](../experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261105_latent_health_v1/summary.json) | 3,000 步 Hb 实验中，K512 相对 K128 的两库总 NMSE 降低约 36.6%/13.7%；均值漂移仍造成连续模型误差 | 同时保留连续、K128、K512，分开判断学生拟合与量化损失 |
| [当前 500 步 pilot](../experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261112_state_pilot_v1/summary.json)与[逐窗重建](../experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261112_state_pilot_v1/reconstruction_curves_20261113_v1/index.html) | 训练监测仍在下降；连续学生已存在 EEG 高频幅度不足、Hb 分支偏差和 Visual 大幅漂移 | 先增加固定曝光并改良波形路径，保留高误差位置和原评分分母 |

**有限矩阵。** 沿用完成 pilot 的四名训练被试、五个父窗及其训练/留出时间块，
不扩展数据人口或重拟合 teacher。已查看的同被试留出块明确作为 development。
每模态两组结构各训练连续、K128、K512，seed17，共 12 个终态；每组共享
1,000 步连续预热，再各继续 3,000 步。末态统一评价，不按留出曲线选 checkpoint。

- **dense_long：** 原始 64 宽度、32 维 latent 与 S/O 等权损失，延长优化预算；
  复活窗口延至第 1,500 个量化更新，随后固定码字维护策略至训练结束。
- **waveform：** 宽度 128、latent 64；在完整采样网格做时间卷积编码，按 15 个
  锚点进入全局交互；本分支解码特征经可学习上采样与卷积生成完整 30 秒曲线。
  不逐 patch 独立拼接输出。目标为原 S/O 等权项，加 0.5 倍 X 误差及 0.1 倍
  S/O/X 相邻样本差分误差。差分方差仅由原训练前缀估计，mask 要求相邻两点均有效。
  容量、波形路径与损失作为一个改良组合检验；不把组合结果归因于单个组件。
- 两组各自比较连续到硬量化的损失；两种结构在同 K 下比较。K128/K512 的
  双分支名义码率分别为 7/9 bit/s；K512 的收益同时标明增加的容量。

**验收与逐项交付。** 本轮工程目标预定为两模态四库等权总 X NMSE 均较
500 步 Q-SO 降低至少 20%，且对应 S/O 宏平均误差各不恶化超过 5%。同时给出
与等曝光 dense_long 的同 K 比较。完整面板是主表，Visual 高误差位置不剔除；
分位数仅补充描述。HbO/HbR、记录和数据集分别保留，通道与窗口不充当独立被试。
若目标未达则保留失败，不按结果追加训练预算或改写门槛。

- [ ] 确认父运行完成、200 Hz 配对数组/尺度/掩码一致，任何数组读取前限定 train。
- [ ] 合成整数重放、分支隔离、全采样网格 shape、反向梯度、差分 mask、固定预算与恢复测试通过。
- [ ] user-systemd 持久执行；先实测每 GPU 1/2 worker，按吞吐与显存选择并发。
- [ ] 全部固定任务完成，保留失败及每 500 步固定训练探针的 S/O/X 误差和复活计数。
- [ ] 全允许训练滑窗及完整留出集按库/支计算活跃码、有效码和最大占用；健康阈值见合同，末半程无复活。
- [ ] 分开汇报 S/O/X 的冻结尺度 NMSE、相关、幅度、差分与有符号边界误差。
- [ ] 固定复用上一图集的 27 个身份，展示改良前后及连续参照，保留完整曲线与 ID。
- [ ] 报告主目标、同 K 比较、码率代价、全部负结果及下一步；这些同被试开发结果用于工程选择。

### Hb 单 ID 联合形态：TK-HB-ID-SHAPE-v1（2026-11-13）

本轮保持 30 秒 encoder 与主 decoder、每 2 秒一个 K128 ID，聚焦 Hb-S 的
真实局部形态。执行与条件分支由
[`dual_component_hb_id_shape_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_hb_id_shape_v1.yaml)
持有。E0 精确复用原始 waveform 基线；E1 去除 HbR 的全部波形监督但保留
HbO 原系数；E2 按 E1 的共同 HbO 指标触发，并将 HbR 输入固定为零。
E3/E4 使用相同的 Hb-S 中心化形态目标、训练尺度和权重，分别约束主 decoder
与单 ID 线性读出。E3/E4 共享原始连续预热，新增形态损失从硬量化阶段启用。

统一分析先核对原始基线、R2 和连续参照，再评价逐色团重建、真实同码字片段、
训练拟合的单 ID／序列读出和同一物理片段的跨窗口稳定性。首轮仅使用原训练
被试面板的固定开发块；晋级候选按合同进行完整多种子复验，之后一次验证未参与
结构选择的训练分区被试，使用冻结模型和原面板训练尺度。EEG 复用现有模型，
S/O 缓存与 teacher 不变。工程晋级阈值、平局规则及停止条件均在配置中预先固定；
运行结果和判定进入原有研究状态注册表，不改写历史迭代证据。

### 有界动态与量化迭代：TK-STATE-DYNAMICS-v1（2026-11-13）

本轮是原固定矩阵后的独立有界开发任务。三个版本化执行合同
[`R1`](../experiments/configs/physiology_semantic_tokenizer/dual_component_state_iteration_r1_v1.yaml)、
[`R2`](../experiments/configs/physiology_semantic_tokenizer/dual_component_state_iteration_r2_v1.yaml)、
[`R3`](../experiments/configs/physiology_semantic_tokenizer/dual_component_state_iteration_r3_v1.yaml)
分别冻结任务列表、单项假设、初始化、预算和晋级规则；历史合同与权重保持原身份。
沿用四名训练被试、五父窗及原27个开发样例，K128、30秒上下文、每支2秒单ID、
200 Hz EEG/10 Hz Hb和原目标/掩码/尺度均保持。原连续预热共享来源与新硬量化
曝光分开记录。最多三轮修改、六候选与两个仅供晋级者使用的确认作业。

冻结诊断把每个2秒patch的误差正交分为水平项与去均值形态项；2秒/6秒OLS
斜率使用真实秒数，近零动态阈值仅由原训练前缀确定。去均值用于诊断或明确的
损失投影，不改变输入幅度。完整波形、逐库/色团指标、停止复活前后的完整健康、
真实赋码分布与实际图像检查共同决定接受/拒绝。独立连续模型与临时连续旁路
诊断分别记录。状态解释检查仅在工程晋级后执行。运行结果和负结果通过
[保留证据索引](../experiments/RESULTS_INDEX.md)及 registry 读取。

## 四库下游测量输入冻结（2026-11-04）

用户确认保留现有90秒SSM拟合规则，并要求修复Visual、更新加载器和构建四库新版
下游缓存。SSM继续从自己的原生EEG及保留Hb路径准备输入，不读取本次短窗缓存。
本次只生产测量输入、样本索引及被试划分，不重新拟合SSM或训练tokenizer/词表。
切窗、标签、掩码、处理分支和版本边界统一由
[数据合同](DATA_CONTRACT.md#four-dataset-downstream-input-cache--2026-11-04)及
[`downstream_input_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/downstream_input_v1.yaml)
持有；旧benchmark及其已发布窗口协议不原地改写。

常规事件任务使用共同30秒上下文，标签指向登记的目标试次/block；Visual保留目标
图片锚点，后续图片只是上下文。DSR保留短刺激判断，REFED使用全视频连续窗口及
动态valence/arousal序列。由于切分规则、输入长度和监督目标不同，**DSR与REFED
的任务难度很可能与其他任务显著不同**；这是设计层面的预期与可比性限制，不能写成
已经通过统计检验证实的难度差异。后续按任务分列指标和完整分母，不用跨任务分数
差异直接归因于tokenizer能力，也不把不同Probe或重叠窗口算作独立被试。

先冻结源身份和按被试的划分，保留已有两库名单，再添加REFED/Visual；代码不得因
数据集排序改变旧被试归属。使用修正的Visual原生通道映射及FCz，保留其他模态的
原始身份。连续记录只缓存一次，样本以时间索引读取，未获支持的区间和通道通过
明确掩码或排除记录保留。大型生产由持久服务执行，先测吞吐与内存再设置并发，
完成后核验所有登记数组、样本身份、时间支持和标签。

## Tak 四轴全公开数据失配分析（2026-10-28）

`SSM-TAK-MISMATCH-v1` 的执行合同为
[`tak_mismatch_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/tak_mismatch_v1.yaml)，
入口为 `evaluate_tak_mismatch.py`。这次用户要求四个登记数据集的指标分布和典型时间曲线；
清单来自中央 v5/none 缓存登记，不沿用历史有限面板，也不访问独立比较协议的保护材料。
每个原生 Hb 记录按不重叠 30 秒网格保留全部通道对，跨已知拼接支持的窗口记录为不可用；
能在中央声明时钟支持内对齐的窗口附加 EEG，其余仍保留 Hb-only 分析资格。
EEG 功率由原生电位在 90 秒上下文计算；不足 90 秒的 REFED 视频使用其完整、至少
30 秒的真实支持并记录实际长度。尾部、缺失、时钟、几何和数值失败均进入分母。

四轴分别为 Hb-only 条件拟合误差 `D1` 与 10/20/40 秒弱平衡 `D_bal`、
Hb 驱动解释 EEG 的 `D_EH` 与局部可检测驱动差 `D_obs`、独立 EEG-only 的
近邻减替代邻域 Hb NMSE `Delta_spatial`，以及联合三观察等权残差 `D4`。
误差使用外训练被试冻结的观察 SD；输入 HbO/HbR 保持共同正尺度。
弱核只在完整连续支持内计算。`D_obs` 明示预设有色/跨色团相关噪声情景，
剔除允许初态、参数和参考所删除的驱动常数方向；同时报告秩、非线性回放差和情景敏感性，
不把该噪声假设称为实测噪声校准或生理可识别性。

五个被试外折使每名被试评价一次；PC1、尺度和空间选择只使用外训练被试。
空间候选均为六通道五频带 PC1，共用原近邻的正缩放因子；内层被试留出使用固定 Tak
切线模型筛选，外层再用完整非线性 EEG-only 比较，因此收益针对预选替代方案。
保留零预测、训练模板和有合法非重叠支持的错配 EEG 对照。
固定参考参数为主；配套 Hb-only 与 joint 在原生连续 Hb 支持和通道对内共享
`kappa,tau`，各窗独立驱动与四初态，参数先验每组一次。三起点仅允许同可见数据的
固定解热启动。空间独立预测始终使用固定参考参数。
自由参数解处的 `D_obs_free` 逐窗投影 `kappa,tau` 切向量，允许每窗独立的局部参数
补偿，因此是共享参数解处的逐窗放松诊断，不是整个共享组堆叠 Jacobian 的可检测差。
它与固定参数 `D_obs` 的投影空间不同，数值下降不单独解释为组共享模型改善；
直接观测误差的两口径比较以共同成功支持上的 `D1/D4` 为主。

分布按数据集分别报告中位数、IQR、P90/P95、覆盖与失败；主汇总被试等权，
时间分布对同一记录时刻的多通道重复使用反权重，区间以整被试重采样。
典型曲线按各库固定 `D1` 的中位、P90 和最高有效值选择，尽量来自不同被试，
不参与调参。完整终态、数值数组和训练坐标由版本化 run 持有；报告只作只读导出。
这组指标检验带通缓存坐标下的模型相容性，四轴不能相加，也不能证明唯一生理恢复。

2026-10-29 按用户“每名被试每项任务优先产出绘图结果”的要求，增加同一数值合同下的
有界调度清单 `20261028_v1/plot_priority_v1/plan.json`。它保留全部允许被试、原生记录及
连续 Hb 支持，每条记录按登记几何的最远点采样固定三个分散通道对，不按拟合结果选取。
REFED 每个视频单列，其他数据按原生任务分列；Visual 的 Part 和 Probe 均保留。
先交错执行每个被试×任务的一个完整组，再补充其余支持段和通道；组内窗口、两种视图、
三起点、参数先验、训练坐标和原拟合代码不变。完成项直接复用，原全量证据不覆盖。
不存在完整 Hb 支持的组和已排除记录仍在面板清单中列明。
`plot-plan` 冻结清单，`plot-run` 执行该面板；运行进度和持久服务由该清单相邻记录持有。
`render_tak_mismatch.py --priority-panel` 只读导出阶段覆盖图、固定参数四轴图、共享参数
配对图及时间曲线。每个点为被试×任务单元，配对误差仅用共同有效窗口：先组内平均
平方误差、再对组等权平均并开方。不同任务点仍有被试内依赖；不作独立样本显著性推断。
未完成、失败和支持不足不填零；阶段成功子集不能替代全量结论，面板完成也不把原全量
campaign 标为完成。每次导出使用新的版本路径。

固定参数的生理失配探索由
[`tak_mismatch_fixed_physiology_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/tak_mismatch_fixed_physiology_v1.yaml)
及同一 `evaluate_tak_mismatch.py` 的 `physiology-*` 阶段持有，证据写入父 run 的
`fixed_physiology_v1/`。用户于2026-10-29要求跨被试、任务、重复记录的失配分布及生理解释；
这是一组看过早期结果后的探索分析，不是独立确认性试验。主标量保留完整固定分母，
数组形态与眼电分析使用已冻结三通道面板，不重新拟合SSM或改变父证据。
主任务比较为D1、D_EH和共同初态血流重放差D_drive_H；被试内配对、整被试bootstrap
及双侧sign-flip检验，全部数据集×任务对×主指标统一BH校正，诊断指标单列校正族。
跨原生记录重复性和记录内早晚一致性分开；Visual同步Probe不当重复，REFED不同视频
不当同条件重测。Simultaneous固定任务顺序带来的时间/疲劳混杂仍保留。

Hb残差分析区分幅度、偏置、去均值残差以及标准化共同/相反方向；这些是条件模型残差
的统计特征，不自动识别生理来源。眼电锚点仅使用中央登记的原生HEOG/VEOG；按同窗
先聚合通道对、再记录和被试汇总，并比较零时滞与非循环±5秒。预处理敏感性只复用
已有EOG回归，再使用原冻结PCA/尺度重算EEG代理；原Hb-only读出保持不变。均值/SD
匹配对照不读取Hb，用于区分幅度变化和额外时间结构变化。此操作改变了EEG目标坐标，
不能描述为原EEG预测准确率提升或眼源因果占比。报告通过现有可解释性报告目录导出，
所有可用、失败与缺失分母及具体假设继续由分析合同和保留表格持有。

### 固定模型偏离的通道地形图（2026-10-29）

按用户要求，基于上面的完整固定参数标量，为每名被试的各原生记录、各Hb通道对
汇总空间偏离，并从每个数据集挑选三名明显模式的展示病例。新分析由同一入口的
`physiology-topography` 阶段和独立 `spatial_patterns_*` 子目录持有；父拟合、冻结
配置和数值不改写，不读取新原始信号。跨模态指标仍定位在Hb测量点及其六通道EEG
代理，不能称为单个EEG电极偏离。Visual只对已核准的Hb-only端点作空间解释。

选例主指标固定为D1。记录分数为通道P90−P10相对中位数的反差、截断到0–1的
Moran邻近一致性、截断到0–1的早晚Spearman相关三者乘积；被试分数取合格记录
分数中位数，再乘合格记录比例。早晚按原记录计划时间等分，丢弃奇数中心窗，
每半至少两个有效窗口。合格记录要求至少8个且不少于80%的计划位置具有共同支持，
各半达到80%计划窗口覆盖；这些阈值只用于选例，全部可计算的部分支持仍在地图中
显示并注明覆盖。早晚一致性也参与挑选，因此不是独立验证。所有候选与失败保留。

整记录RMS先合并平方量，残差/零预测比先分别合并匹配分子和分母；被试图按原记录
等权。Visual同步Probe合图而不算两次记录，REFED不同视频不当同条件重测。
通道与外训练折冻结尺度保留，不把标准化地形当绝对浓度地图。保留登记几何的头顶
投影和测量节点；MNE线性插值限制于有邻近测量支持的小三角形，大间隙及缺失留白，
边缘或近共线节点仍以有色点显示原值。地图、中心化图和跨记录相似性均为探索性
描述，不能把有意挑选的清晰病例推广为人群比例或皮层生理来源。

### Visual 原生 EEG 身份修复与四库报告（2026-11-04）

按本次用户明确要求，`SSM-TAK-MISMATCH-VISUAL-v2` 使用
[`tak_mismatch_visual_corrected_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/tak_mismatch_visual_corrected_v2.yaml)
重做全部允许 Visual 记录、Hb 通道对的固定参数拟合；组共享 κ/τ 先完成原有
三通道覆盖面板，供四库同口径比较，随后补齐全部24个位置。按本次用户选择，
允许在全量运行途中只读抽取已终态结果；每次报告注明实际范围并版本化导出，
不将中途面板标为全通道完成。
原生 EEG 的 A* 身份只认发布者 `Raw_EEG_channel_reference.xls`，包括 A64/FCz；
空间几何继续按电极名称与已登记模板连接。保留 v5 缓存只提供原 Hb 数组、
原始来源及时钟元数据，不读取或重标其错误 EEG 数组。新运行的 prepared 数组
与通道身份共同形成独立版本，不依赖下游任务新建的30秒输入缓存。

Hb 30秒目标网格、至多90秒原生EEG滤波支持、5秒参考、被试外折、冻结参数、
先验、数值预算和评分口径均沿用原合同。通道修复后重新计算全部候选EEG特征、
训练折内PC1/尺度与空间选择；不把旧投影套在新电极上。固定Hb-only结果与
旧版本的配对一致性作为核验，旧结果和错误身份的负面证据保留。

四库报告使用更正后的Visual与其他三库已有证据。固定参数空间/幅度/形态端点
保留各库完整分母；共享参数跨库比较以各库都有的既定三通道面板作配对，Visual
新增全部通道共享参数结果另列，不将不同覆盖范围的汇总直接当作方法差异。
优先检验四库共同可计算的Hb相容性、Hb与EEG代理的差异、空间异质性及记录内
稳定性；同条件重测和眼电机制只在存在相应原生记录/独立参照的数据集判读。
所有模式为探索性模型失配特征，不把小残差或参数变化直接称为生理真值恢复。

## 全量个体校准 τ/β 重建缓存（2026-11-04）

**后续版本选择（2026-11-06）：** 用户指定下一轮使用每被试个体 PCA。
[`v2 合同`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v2.yaml)
改为每个数据集内、每名被试仅用自身有效无标签观测拟合一个 PC1 及尺度，
在其所有记录与位置间冻结；完整范围与评分分母由
[数据合同](DATA_CONTRACT.md#individual-pca-successor--2026-11-06)规定。
下一轮使用新版本输出，保留 v1 的公共 PCA 缓存；本节下述公共 PCA 设计属于 v1。

用户明确要求在分析个体校准后开始全量缓存，并要求不足90秒的REFED也参与拟合。
[`shared_driver_tak_subject_cache_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v1.yaml)
固定四个公开数据集、原生EEG与v5/none Hb输入、10 Hz网格、τ/β自由参数及两者
log先验SD=0.125。其余Tak参数、驱动/四初态正则、三启动点、数据项每秒权重与
4/8子步积分检查沿用基础合同。新版本不恢复已停止的空间选例，也不修改旧证据。

个体校准适用于离线条件重建，但逐被试PCA会改变同名代理的频带组合，不利于共享
表示。因此每库仅用下游训练被试拟合公共PCA方向，各被试使用自身无标签观测拟合
正EEG尺度、共同正Hb尺度与成分SD，并在所有记录和位置间固定。完整校准支持和
逆变换保留；验证/测试被试的个体校准明确属于适配，不宣称严格零样本跨被试输入。
这不是对哪种校准更优的确认性检验，也不将β解释为独立标定的神经血管耦合。

在中央连续支持内作90秒拟合、60秒步长及完整尾段窗口。REFED不足90秒时使用
实际完整时长，其相对先验影响与长窗可能不同，质量按时长区分。短窗不补齐，不跨
间隙或视频拼接。所有合法Hb位置都进入计划；未知几何、同步拒绝、无效通道、
准备失败、校准失败和求解失败各自保留，不按拟合质量筛选人群。

先进行固定PCA/尺度逆变换、身份与时钟测试及90秒/短时长合成求解，再在每库首个
元数据可用训练记录和首个短REFED训练记录上作先导。先导的临时坐标不进入正式
缓存；仅同身份的未缩放特征可复用。先导用于数值和资源验证，不用于调参数或换
样例。正式生产完成全量准备、冻结校准、固定参数基线和τ/β拟合，最后独立核对
每份数组、状态观测关系、残差闭合、指标及完整分母。

主终点为完整计划上的可复用条件重建及失败状态；次要比较是与同驱动/初态自由度
固定基线的配对重建误差，先窗口、记录、被试等权汇总平方误差，最后开方。零假设
为新增自由度不保证各观测分量变好。相关、幅度、偏置、局部误差、零预测基线和
数值验证同时保留；成功收敛不等于良好波形或生理真值。全量完成后停止，不自动
训练tokenizer或打开其他协议的保护评价。入口、存储与输入边界见
[数据合同](DATA_CONTRACT.md#subject-calibrated-tak-reconstruction-cache--2026-11-04)。

同日按用户要求开展数值实现优化：以等价的参考均值减法和初态坐标映射代替
稠密矩阵操作，并将float64正规矩阵与Cholesky块交给共享GPU进程。动力学、
先验、边界、线搜索、停止条件及冻结校准保持原定义。先用保留合成、四库先导、
短REFED、慢窗口和未收敛窗口进行完整拟合比较；用例和数值阈值由运行内
版本化验证子目录的`protocol.json`持有。验证输出写入独立子目录，
不覆盖正式结果；通过同源码核验后仅续接尚未完成的任务，保留原后端结果和失败。
只调整共享内存传输时，先核对CPU数值函数未变，再复用其已完成对照；GPU用例
重新拟合。不同时间测得的耗时明确标注，生产吞吐另行实测。
只改变CPU/GPU worker分配时，维持单一控制器和任务队列；核对数值库、拟合
函数及GPU传输实现未变，并测试混合分配后，可以明确复用既有完整拟合比较。
根据总吞吐、GPU等待时间和并发资源选择分工，不把单个线性块的提速当作全流程提速。

## τ/β空间分布的探索性选例（2026-10-29）

按用户要求，以空间特征清晰度优先选择展示被试，不绑定原前额面板的24名被试。
[空间合同](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_topography_v1.yaml)
从注册公开数据中排除原开发被试后建立候选池；先用两个90秒片段的Hb动态RMS
空间对比度及重复相关筛每库三名，再对全部有效Hb位置拟合固定参数和τ/β两臂。
全部候选分数、排除和入选后的失败保留。该选择针对有特征的解释样例，不能作为
人群确认性结论；观测的空间特征不预先等同于参数特征。

新空间坐标仅由原开发被试的全部测量位置拟合，每库共用一个PC1、一个Hb正尺度
与一套观测SD，避免逐通道归一化吸收β差异。模型、先验宽度、90秒窗口与10Hz
合同沿用父实验，旧前额数值不被新空间坐标重标。两个片段分别拟合，比较同一
被试空间模式是否复现，再比较不同被试的中心化空间形状与绝对水平。

位置来自已登记模板；Visual的两Probe共同投影，REFED使用已核验论文布局。
只绘制有测量支持的节点，重复模板位置合并展示并保留通道成员，不向未覆盖区
外推，也不把头皮投影称为个体MRI或精确皮层定位。合成平坦与结构化参数图先于
实测筛选执行；分析仅描述相关、对比度和误差，通道与重复片段不充当独立被试。

## τ/β组合与同被试跨任务参数关联（2026-10-28）

按用户要求，以[版本化合同](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_task_parameters_v1.yaml)
比较 τ/β 与 κ/τ 的同片段拟合、同条件重复分散及任务关联。父实验的开发集
坐标和先验宽度冻结；复用结果明确保留原身份，不计为新增重复。
Single-Trial 的 MI/MA、Simultaneous 的 WG/n-back/DSR 用同一被试的完整
连续支持作留出单位；每个支持内两个不重叠片段的 log 参数先平均，训练支持
形成任务质心。标签仅用于元数据分组及拟合后的读出。中央时钟合同拒绝的
记录不读取信号，缺失和拟合失败造成的弃判计入完整预期分母。

合成对照分别固定生理参数、只改变 τ、只改变 β，并改变任务驱动；噪声实现
在三种情景间保持一致。已知变化恢复不足或伪变化保留，不据此调先验或换样例。
主任务读出采用被试内/支持序号内标签置换及四项 Holm 校正；差值和区间按
整名被试配对重采样。结果只能支持模型坐标下的任务关联，不能排除记录顺序、
状态及预处理混杂，也不构成生理参数可识别性证明。Visual 仅用于同任务参数
组合比较，REFED 的情绪条件作为补充。报告备选依据完整分母、配对证据和
可读性选择；完整图和失败作为配套证据保留。

## Tak 参数的更多不重叠数据重复估计（2026-10-28）

用户要求为已展示的每名被试增加参数估计结果，验证稳定性，并选择“同任务/条件
为主，其他条件单独补充”。`SSM-TAK-REPEATABILITY-v1` 使用
[`shared_driver_tak_repeatability_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_repeatability_v1.yaml)，
沿用共享驱动入口的 `--tak-repeatability --repeat-stage`。身份限定为已完成
Tak-v1 的24名公开被试及原前额 Hb 对、六个 EEG 通道，不涉及其他协议的保护材料。
新数据遵循当前10 Hz EEG代理与v5/none Hb缓存合同；开发身份复用原父计划，PCA、
共同Hb正尺度和观察SD重新仅由开发被试拟合，旧4 Hz结果保持原身份。

每个90秒片段内 κ/τ 均为常数。对同一原生记录/连续支持/Hb通道的不同、不重叠
数据子集分别重估这对常数，以检验同一参数假设下的估计一致性。各子集不硬绑
拟合参数，各自估计驱动和四初态；重复估计不解释为真实参数随时间改变，也不
充当额外独立被试。图表区分同支持内子集重复、不同原生记录以及不同条件。

Single-Trial 使用三个MI session、Simultaneous 使用WG的三个中央声明连续支持，
各取两个片段；Visual 在原Probe1的可用记录间均衡取六段。候选只依据时间与
中央元数据选择：先按两种时钟均不重叠的顺序保留，再均匀选取声明数量。
REFED以三条同悲伤条件、长度足够的视频各自前90秒为主，其余长度不少于90秒
的视频单列跨情绪补充。短记录、未知条件、跨拼接支持和准备失败均保留；不依据
拟合误差、参数接近程度或成功与否替换样例。

共同log先验SD固定为0.125；κ/τ参考、其他物理参数、三启动点、驱动/初态先验、
训练尺度误差权重和数值收敛要求沿用10 Hz基础合同。对照臂固定κ=0.64、τ=2，
保留相同驱动和初态自由度。完整计划分母、失败、多起点和4/8子步差均保留。
先执行三个已知生成参数的90秒Tak非静息合成数值检查，再执行公开实测准备和
拟合。大型准备/拟合由持久user systemd监督，经吞吐与峰值内存先导确定并发。

主终点为按被试和原生支持分层的log参数分布及重复离散度；零假设为被试内
重估分散不小于被试间分散。配套报告共同成功身份上的固定参数/自由参数双Hb
重建误差。先在原生支持内汇总，再按被试等权；窗口不作为独立被试。仅作描述性
判断，不新增生理资格阈值或将先验收缩当作真值恢复。完成冻结清单后停止，负结果
同样进入最终图像；本次授权不扩展到其他被试、保护评价或tokenizer训练。

## Tak 原始血流核心的缓存实测对照（2026-10-27）

用户要求恢复忠于 Tak 原始设计的方程，并全面比较实测差异。
`SSM-TAK-v1` 由
[`shared_driver_tak_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_v1.yaml)
定义，沿用 `evaluate_shared_driver_reconstruction.py --tak-comparison`。它恢复
Tak/SPM 的血流核心：容量分母为 `tau+tau_v`，动态流出量为
`v^(1/alpha)+tau_v*dv/dt`，脱氧 Hb 方程仍使用独立 `tau`。
直接消去独立 p 状态，令 `p=v`，无独立血管过程创新。旧 inlet-balance
核心保留用于历史调用及同口径对照，不改写旧运行或冻结源码。

输入为既有公开 output-support 两父运行的全部评价身份，四库各六被试、每人
两个不重叠 90 秒片段及其中间 30 秒。父运行只提供登记身份和保留 EEG 特征；
Hb 通过统一 reader 读取 v5/none 带通缓存，重新按原开发被试确定 PCA 与共同
Hb 正尺度。不得沿用旧 Hb 训练尺度，不读取原始 Hb 或其他协议的保护材料。
完整记录预处理先于遮挡；窗口内 ODE 是处理后坐标的条件模型，隐藏块只解释为
离线特征补全，不解释为原生传感器预测。所有样本按事先固定身份纳入，保留准备失败。

八臂分别为旧自由初态、旧绑定初态、Tak 参考参数／自由四初态、Tak 固定静息
血管初态，以及前述旧自由、旧绑定、Tak 四初态各自估计 kappa/tau，最后为
Tak 固定静息血管初态与原始四参数先验。两种绑定模型均使用四初态工程惩罚
`100*[s0²+(f0-1)²+(v0-1)²+(q0-1)²]`，避免重复惩罚容量。主臂固定
gamma=0.32、tau_v=2 秒，以 log-SD=0.25 估计 kappa/tau；文献先验的
kappa/tau/gamma log-SD 为 exp(-3)，tau_v 为 1。工程目标的观测权重和
Gauss–Newton 推断保持现有定义，所以文献先验臂也不称为完整 Tak DCM 复现：
未恢复任务驱动的神经连接、光密度观测或变分推断，EEG 约束的驱动仍可估计。
P0/Q0 保持现有观测坐标约定，不用论文浓度值替代未经标定的尺度。

生理参数仅在同一被试、原生记录／连续 session 支持和 Hb 通道内共享；不跨
会话绑定。原文件拼接时 session 支持也进入分组键。不同片段各估初态，不假造
缺失间隔的状态延续。参数先验每组加一次，固定参数启动解后使用三个声明起点，
按收敛后的完整目标选解。30/90 秒和联合完整、Hb-only、联合 Hb 分块隐藏的
矩阵均执行。90 秒联合完整另检查 log-SD=0.125/0.5、tau_v=1/4 秒和
初态权重50/200；这些是工程敏感性，不能称生理正常范围。

独立合成包含 Tak 静息、Tak 非静息和旧自由初差三种生成器，每种三个固定种子。
真值只用于评分，不进入初始化。独立 SPM 对数 ODE、零黏弹性新旧等价、状态与
参数 Jacobian、隐藏值隔离及物理域测试先于实测。4/8 RK4 子步差异用冻结训练
SD 归一化，阈值0.005；预算耗尽或未通过梯度判据均保留失败。完成合成后再
准备实测；所有较大拟合均由持久 user systemd 监督，pilot 后按实测吞吐选择并发。

主终点为同身份 Hb NRMSE 的旧模型减 Tak 主臂。数据集、窗口长度与可见条件
分别统计；先时间／色团 NMSE，再记录被试内均值、被试等权均值，最后开方。
2000次整被试配对 bootstrap 给出探索性逐项区间，不作多重检验后的显著性声明。
报告完整计划数、成功数及每项配对分母，同时给出 EEG/HbO/HbR、相关、中心化
幅比、有符号增益、偏置、峰时间差、零预测和可见 Hb 插值基线。状态范围、
初态、驱动、参数边界、多起点差异、先验代价和积分加密分别列出。低误差、
硬绑定或窄先验产生的稳定性不能独立证明生理真值或个体可识别性。

## 无额外成分的滤波敏感性样例复测（2026-10-27）

本次用户要求仅检查 10.20 之后、没有额外成分的实验，用少量样例判断原解释是否
改变。执行合同为
[`shared_driver_filter_retest_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_filter_retest_v1.yaml)。
复用 10.23 无源衰减诊断的 original、slow、no_decay 三臂和无额外项相容性指标；
不纳入 volume/exchange、任何持续外加输入或含额外项 teacher 的 tokenizer。

各库在旧 original 的 90 秒联合完整误差有效记录中选择最接近中位数者，先冻结
样例，再读取新缓存或拟合。保留全部候选及旧失败分母。这是解释性样例选择，
不是总体确认或调参。早于 10.20 的父运行只提供登记身份、准备数组及训练身份，
不作为本次复测的旧模型结论。

未滤波对照只读保留数组；新 Hb 只读统一 v5/none 缓存。两个分支使用相同、同时
有支持的原开发记录重新确定各自 PCA 和共同 Hb 正尺度，评价样例不参与。保留
30/90 秒相同中心、原固定响应参数、求解预算与起点规则。只有目标、算子、尺度
均匹配时复用旧拟合，其余按同预算重拟合；所有失败保留。完整记录预处理先于
遮挡，隐藏段是离线处理特征插值，不能称原始传感器的未来预测。

除各分支训练 SD NRMSE 外，比较共同缓存训练尺度下的误差、缓存单位 RMSE、
相关系数、中心化幅比、有符号投影增益、偏移和峰时间差，以及残差、合法性、
收敛、积分精度、条件状态范围与衰减保留。低振幅零预测基线单列，防止把低误差
误判为波形恢复。相容性保留同参数联合节点与完整分母；窗口内状态没有生理真值，
更低误差不授予恢复资格。逐样例判断“保持、削弱、出现反例、仍不能判断”，不把
四个不同数据集的样例混池作显著性或总体翻转声明。

## 五项模型相容性指标的总体分布与典型曲线（2026-10-23）

用户明确要求实现完整指标套件、获取实测结果并分析总体分布和典型样例。
`MODEL-COMPATIBILITY-v1` 的执行合同由
[`model_compatibility_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/model_compatibility_v1.yaml)
持有，数值由 `src/inference/model_compatibility.py` 持有，主入口为
`experiments/scripts/evaluate_model_compatibility.py`。冻结模型面板的独立读取与执行
由 `evaluate_model_compatibility_panels.py` 承担；证据写入
`experiments/runs/physiology_semantic_tokenizer/model_compatibility/<version>/`。
这是公开开发材料的描述性相容性分析，不接入比较协议保护数据，不授予生理资格。

**普查与面板分母分开。** 普查范围是 v5 通用缓存登记且符合现行数据合同的
四库公开记录，沿用 Visual S06 Part1 的明确排除。登记库存不是原发布材料的所有
可能分支，例如 REFED 只含登记视频，不额外纳入 baseline 文件。原生 Hb 由统一
`load_native_fnirs_record` 读取，不自行解析或把缓存当原始绝对浓度。主分析不新增
滤波，避免把 v5 全记录双向滤波后的瞬态当成未滤波指数。按原生时间分别划分
30/90 秒不重叠完整窗，丢弃尾部，保留跨拼接边界、缺支持、低激励与失败状态。
单库的 Hb 单位及两色团相对标度保持不变，不将各色团分别 z-score 后相加。

Hb 弱闭合剖面同时消去常数偏置与允许的初态指数项。归一化分母由该窗口中心化
动态量确定，对全部候选参数固定；报告固定参数与剖面最优值、完整 τ 剖面和
动态能量。此量可能超过一，不是异常成分占比，也不是噪声白化统计量。单窗
参数剖面是条件描述，不是训练后预测。合成控制包含合法初态、共同缩放/偏置、
相对色团变化、独立成分和不同噪声水平，先验证数值及已知响应，再读取实测信号。

具有冻结 EEG 特征、训练尺度及确切时间算子的四库 output-support 评价面板
读取两个既有公共父：`ssm_output_support/20261010_v1` 与 `20261013_refed_v2`。
保留连续 90 秒与同一记录中部 30 秒，参考与末端插值支持遵循父入口。每个
相同 τ/κ 节点计算 Hb-only 与 EEG+Hb 离散设计投影，保留自由驱动和独立初态，
按同一节点的各模态误差形成共同容许网格。工程容限敏感性、秩与低振幅基线
同时报告；训练 SD 不是传感器噪声，网格范围不是置信区间，空网格不是连续域
无解证明。不得把各项分别最优值拼成共同参数解。

固定空间检查仅应用于有核验几何和冻结 loading 的 Single-Trial 空间父面板
`shared_driver_spatial/20261020_measured_v2` 的评价身份及 K1/K2 读出。保留完整
观测算子、增益、评分支持，比较固定子空间和自由秩残差。原生能量比例直接投影
到固定空间子空间，放松时间限制以避免病态潜曲线反演；实际时间算子下的训练
尺度保守下界另列，不把有限精度潜曲线回放残差冒称精确下界。其他普查记录没有
对应冻结空间映射时标为不适用，不临时拟合新 loading 冒充原模型。

非线性层在 output-support 面板的同一选定参数节点，复用
`fit_vascular_flow_relaxation` 生成合法血管轨迹，再经过原观测算子评分。
记为条件非线性前向重建距离 `D_Hb,NL,recon`，与教学版绝对坐标反演残差区分。
保留独立 v/p/q 初态、数值域、双精度积分复核与优化状态；这是已找到误差上界，
不是全局最优或生理不可能性的证明。未具备坐标的全库存直接非线性指标标不适用；
不将空间面板的独立单通道拟合拼成多核联合结论。

统计保留全部计划和状态分母，分别呈现窗口分布与先通道/窗口到记录、记录到
被试、再被试等权的分布。区间以被试整簇重采样，四数据集分开；不把重叠观测
或多个通道当独立被试。典型曲线依据预先声明的固定参数误差，在每库选择接近
P10、中位和 P90 的可用窗口，尽量来自不同被试，另展示失败或共同集为空样例。
这种选择只用于解释，不用于调参；原生身份、时间支持、分母与选择规则一并保留。

数值测试和合成检查之后，以 user systemd（验证 lingering）运行实测 pilot。
记录 CPU/内存/存储、竞争任务、吞吐和峰值内存，再选 bounded 并发完成有限任务
矩阵；保留失败与可恢复任务记录，不在 Codex 前台会话里启动长期 producer。
执行状态由 registry 持有，指标与分布由本轮运行的 summary/逐条结果持有。

## 当前共享驱动模型的表达边界与联合可行性（2026-10-22）

`SSM-FEASIBILITY-v1` 根据用户提供的详细分析，将表示限制、Hb 内部相容性、
跨模态冲突和参数信息分开检验。可执行合同为
[`shared_driver_feasibility_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_feasibility_v1.yaml)，
复用 `evaluate_source_observability.py --feasibility`；新证据写入既有
`source_observability/` 下的版本运行。执行状态和科学裁决由 registry 持有，
各阶段数值及终态由本轮运行记录持有，不重写 10 月 20/21 日证据。

输入限于已完成公开空间运行 `shared_driver_spatial/20261020_measured_v2`
列明的评价身份、prepared 数组、训练尺度、冻结读出及完整拟合；不重读原始数据、
不触及比较协议受保护评价，也不重选 EEG 权重。沿用六名被试各六窗、K1/K2
主面板，全部观测可见，评分支持为 30 s 窗内的 5–30 s。这是已使用公开材料的
条件重分析，不是新的确认性评价；完整重建也不是隐藏通道预测。

1. **表示下界。** EEG 用实际时间算子作用于原 16 维时间基，并保留冻结空间
   读出；对评分支持上的固定训练 SD 加权矩阵求最小二乘投影。Hb 松弛为每核
   任意 HbO/HbR 时间曲线，保留空间混合、增益及实际时间算子。分别报告三模态
   数值投影残差、秩和保守下界。数值最小范数解可能截断极小奇异方向；用于
   排除的证书把这些方向保留在更大的观测子空间中，避免截断抬高下界造成误判。
   同时校验评分支持和观察算子。保守下界不冒称原预测空间的精确最短距离。
   Hb 的这个下界只检验观察映射，不能冒充血管动力学的下界。
2. **Hb 内部见证。** 在预定目标光学位置，直接以正流量驱动当前 inlet-balance
   血管方程，保留独立的 v/p/q 初态和完整观测算子。冻结八个 tau/alpha/E0
   工程敏感性节点，含原参数及改变低频符号的宽 alpha 节点；参数和流量范围
   均不命名为正常生理范围。逐节点优化 4 Hz log-flow，检查两色团误差和 4/8
   子步一致性。找到合法低误差解是存在性见证；有限节点、有限流量范围内未找到
   解仍为未决。当前实现的 p−v 衰减关系不泛化为所有 Balloon 方程的限制。
3. **联合容限。** 直接约束 EEG NRMSE，而不是选择一个 EEG 权重。预先固定
   0.25/0.5/0.75/1.0/1.5 的 EEG 容限，最小化两色团等权训练 SD 标准化均方
   残差；低于 EEG 表示下界的节点直接排除。对固定生理参数与释放 tau/kappa
   比较，保留同一目标下的固定解和更窄容限解，保证见证集合嵌套。结果称已找到
   可行解的包络，不称全局最优边界。alpha/E0 敏感性属于上一阶段，不混入
   原固定 alpha/E0 类的联合结论。
4. **参数信息的条件入口。** 仅当 EEG、HbO、HbR 各自 NRMSE≤0.5，且物理和
   积分检查通过，才在固定 tau/kappa 网格重新拟合驱动与初态，报告已发现的
   容许参数节点。它们不是置信区间；没有找到远端节点不证明参数已被识别。
   个体迁移比较另须满足已知真值恢复及任务、区域、校准可比的独立记录条件；
   任何前置条件未满足都明确报告未进入，不用稳定点估计替代证据。

主标准是三模态分别小于 0.5 个训练 SD，完整报告其余容限的覆盖。训练 SD
不当成传感器噪声；并列零预测、中心化形态误差、相关和幅比，防止低振幅曲线
靠小幅预测被解释为波形恢复。统计先时间/特征均方，再记录内、被试内平均，
最后被试等权开方；描述性区间整块重采样被试。完整保留表示排除、合法见证、
数值未决、失败及条件未进入的分母，不把六窗当六名独立受试者。

独立合成包含非静息且 p0≠v0 的匹配正例、固定噪声正例、已知 EEG/Hb 空间
正交残差及分模态可行但驱动不同的冲突控制，真值不作普通初始化。先通过
软件和合成检查，再在持久 user systemd 服务中计时、选并发并执行完整有限矩阵。
优化收敛状态与可行性状态分开；预算耗尽但经细积分核验满足要求的解仍可证明
存在性，预算耗尽而未找到解不能证明无解。

## p−v 无源衰减的直接动力学检查（2026-10-23）

用户明确指出上一轮额外观测方向对照没有直接隔离持续衰减问题。本轮
`SSM-DECAY-v1` 由
[`shared_driver_decay_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_decay_v1.yaml)
定义，复用 `evaluate_shared_driver_reconstruction.py --decay-diagnostic`。
它直接检查 `delta=p-v` 的无源衰减，不把所有 Hb 或神经驱动称为衰减量。
旧报告和旧结果保留原身份；旧额外共同/交换观测项不进入本轮主比较。

唯一动力学干预为 `p_dot=v_dot-lambda*a(v)*(p-v)+u_delta(t)`，其中
`a(v)=v^(1/alpha-1)/tau`。保持其他状态方程、Hb 观测映射、固定生理参数、
训练尺度与掩码不变。三种衰减倍数 1、0.1、0 与无/有持续输入交叉为六臂。
输入直接进入 p 状态方程，30/90 s 各含DC常数项加6/18个余弦模式，最高0.1 Hz，
单位为归一化状态/秒；不同衰减臂不能重新缩放输入基来匹配输出幅度。
这只是总Hb与血容量失配的形式性动力学松弛，不直接代表已识别的真实血容量室。

无输入时所有非负衰减倍数都保持 delta 的符号；正倍数只改变初差消失速度，
零倍数给出常数，不能自行产生重复振荡。原生理驱动仍可持续激励 v、q 与 Hb。
固定原拟合的驱动和全部初态、只改变衰减倍数的反事实必须保持 s/f/v/q 与 HbR
一致；与六臂完整重拟合分开报告。前5秒参考可使已衰减初差留下恒定观测偏移，
须分列原状态衰减、参考后偏移和晚期波动，不能把偏移视为持续动态。

实测读取上一轮相同两个公开 output-support 父运行的精确计划、prepared数组
和冻结训练坐标；每数据集6被试各2条原生连续90秒记录，与中间30秒配对。
末端插值bin排除、逐长度前5秒参考、三种联合完整/Hb-only完整/联合分块遮挡
条件沿用原合同。无新原始读取或其他协议受保护评价。分段误差、共同中段去均值
残差和衰减保留率分别计算；30秒独立初态重置不冒充90秒的连续传播。

独立合成覆盖零初差匹配模型、原衰减初态、慢衰减初态和持续输入四种真值；
同seed固定上游驱动和噪声。观测拟合与 delta/输入恢复分别评分，合成真值不进入
拟合初值。先通过独立ODE、离散切线、零输入复现、隐藏值隔离与物理域检查，
再通过持久监督pilot测量并发资源，完成固定合成与实测矩阵；完整保留失败。

主比较分别为无输入时改变衰减速度、同衰减下增加持续输入。报告数据项和各先验
代价、初差保留比例及95%损失时间、早/中/晚Hb误差、输入RMS、驱动及其他状态
补偿，以及4/8子步一致性。按记录配对、被试内平均、被试等权汇总，2000次
bootstrap以被试为整块，区间为探索性逐项区间。取消衰减后无改善，可能因为常数
模式被参考算子消去；输入改善可能只是容量收益。均不能据此单独确认真实生理
衰减率、血容量来源或跨记录预测能力。

## 单一血流驱动的 Hb 相容性与 30/90 秒持续波动（2026-10-22）

用户要求调查独立拟合初态仍缺少持续独立血容量激励的局限，并在当前公开实测
数据上比较 30 s 与 90 s。可执行合同为
[`shared_driver_volume_drive_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_volume_drive_v1.yaml)，
复用 `evaluate_shared_driver_reconstruction.py --volume-drive-diagnostic`。
读取两个合同列明的已完成 output-support 父运行的 plan、prepared 和训练坐标；
不重建原始缓存，不访问其他比较协议。每条原生连续 90 s 记录与其中间 30 s
配对，保留父运行的全部评价身份、原生通道和记录依赖。训练 PCA、共同 Hb 缩放
和分项训练 SD 冻结；每个长度仅重设可见前 5 s 参考。模型算子逐行检查裁窗外
依赖，末端插值 bin 不进入拟合或评分。

当前确定性动力学中 `p-v` 是无外源的衰减模式；独立初态不增加持续输入。
全矩阵线性筛查比较绑定/独立初态及额外共同/交换方向，并保留每窗口
tau/alpha/E0 工程网格内的最佳无正则残差，作为对原模型有利的表达能力诊断。
这个有限线性子空间的下界不是非线性全局不可能性证明。

非线性主比较固定原生理参数，重新拟合逐时点驱动和五个独立初态；额外共同项
与等自由度氧交换项使用相同物理截止频率（0.1 Hz）、训练 SD 加权列范数及系数
先验。共同项可以实现为独立固定饱和度血容量室的连续状态及其持续输入，不修改
原 Balloon 室的方程；它也可能吸收浅表或观测失配，不能据拟合命名其真实来源。
30/90 s 各为 6/18 个余弦系数，按时长保持自由度密度。联合完整、Hb-only 完整
和联合分块遮挡三个条件分别检验 EEG 妥协、Hb 内部相容性和离线插值收益；
隐藏 Hb 不进入拟合或初值，参考区保持可见，不能称未来预测。

先独立软件及合成检查，再在 user systemd 下计时 pilot、完成固定合成矩阵和
实测配对矩阵。4/8 子步复核、未收敛和缺失记录完整保留。主终点为训练 SD
归一的双 Hb 误差，另列单 Hb、共同中间区间、尾段、局部 SD 及固定频带误差。
先时间/色团均方，再记录内配对、被试内平均、被试等权开方；区间整块重采样
被试，同一被试的两条记录不当作独立样本。10% 相对 RMSE 改善仅是本轮描述性
重要程度标准；所有结果为已使用公开开发材料的探索性重分析。

## 拟合、观测信息与生理恢复的四套独立诊断（2026-10-21）

本轮用户明确要求把此前讨论的四个问题分别建立为独立实验套件，执行实验、整理
证据、交付四篇详细技术报告并作本地 Git 提交。四套各有配置、入口、运行目录和
科学结论；下面是设计 owner，执行状态仍由 registry 及其生成视图持有。实验使用
新生成的已知真值合成数据及明确列出的保留公开开发数据，不接入比较协议的受保护
评价，不修改历史配置、数组或结果。公开数据已经有研究使用历史，不能称新确认集。

| 独立套件 | 配置与入口 | 主问题与独立交付 |
| --- | --- | --- |
| `SOURCE-OBSERVABILITY-v1` | `source_observability_v1.yaml`；`evaluate_source_observability.py` | 数据与先验各约束哪些驱动、初态和生理参数；非名义参数的已知真值恢复、联合重拟合与剖面 |
| `SPATIAL-INFORMATION-v1` | `spatial_information_v1.yaml`；`evaluate_spatial_information.py` | 更多通道、正确几何及独立动态分别增加了什么信息；弱方向、新观测和容量匹配对照 |
| `CROSSMODAL-INFORMATION-v1` | `crossmodal_information_v1.yaml`；`evaluate_crossmodal_information.py` | 真实 EEG 是否提供超出 Hb 供体与条件模板的独立信息；配对破坏及共享信号剂量校准 |
| `MECHANISM-ATTRIBUTION-v1` | `mechanism_attribution_v1.yaml`；`evaluate_mechanism_attribution.py` | 生理变化与观测失配如何互相补偿；已知反事实中的参数、驱动和共同项归属 |

配置位于 `experiments/configs/physiology_semantic_tokenizer/`，入口位于
`experiments/scripts/`；四个 artifact root 分别为既有
`experiments/runs/physiology_semantic_tokenizer/` 下的 `source_observability/`、
`spatial_information/`、`crossmodal_information/` 和 `mechanism_attribution/`。
每个根中的新日期运行保存自己的 resolved config、固定任务矩阵、源码身份、
监督服务、完整终态分母、逐任务数值与摘要。报告输出到 `docs/report/` 下四个
独立版本目录；报告汇总 owning evidence，不另作实验状态源。

### 共同科学问题与数据边界

“良好拟合”指声明的可见或隐藏观测误差；“恢复”指已知生成器参数、驱动、状态
和成分的误差，两者分别评分。实测潜变量没有对应真值，因此不能用实测拟合
授予恢复资格，也不能从有限模型中的负结果推导生理恢复普遍不可能。四篇报告均
区分数据的非唯一性、低信息量、模型/观测失配、先验偏置及数值未收敛。

直接保留数据入口为 `shared_driver_spatial/20261020_measured_v2` 和
`shared_driver_spatial/20261020_synthetic_v1` 的计划、训练坐标、冻结校准、
prepared/fit 数组与完整任务表；补充旧输出支持、归属和观测语义证据只读取各配置
明确列出的公开结果路径。训练、选择、评价身份沿原计划继承，新的合成生成身份
彼此分开。读取前检查 parent schema、数据集、计划身份与允许的相对路径；不递归
搜索历史根或从未列明的原始记录补齐缺失材料。失配 EEG 的不可用记录保留，不
改变条件定义来制造供体。没有建立读取接口的 ECG/呼吸标为当前不可用，不等同
于原数据集未采集这些信号；EOG 不命名为浅表血流真值。

### 一：观测约束与生理参数恢复

新合成使用非名义 tau/kappa 与非静息初态，固定训练确定的尺度，分别比较 Hb-only
和联合 EEG/Hb、数据项和不同预先固定的先验、多初值以及有明确额外信息的 oracle。
实际释放 tau/kappa、驱动时间系数和五个初态；冻结参数的零误差不能记为估计恢复。
每个剖面固定一个参数，重新优化其余实际未知量；必要时保留二维节点及多个初值。
报告训练尺度观测误差、driver/state 误差、参数相对/对数误差、边界命中、成功数和
优化终止原因。离散网格最低值是该搜索内的最佳见证，不是全局最优证明。

可观测性分析使用实际观测雅可比，明示参数尺度和观测尺度，分别分析纯数据与
加入先验的矩阵；初态或共同项作为 nuisance 联合消元/重拟合。弱奇异方向通过
有限扰动和非线性回放核验。工程容差内的替代解是非唯一性见证，不是置信区间；
先验稳定性不等于数据识别，固定生理参数的历史实测只提供条件敏感性证据。

### 二：空间观测是否打破旧歧义

同一拟合点、同一坐标和目标支持上比较嵌套 Hb/EEG 面板，保留 raw、按声明尺度
加权及原工程目标三种雅可比度量。原目标按可见数量归一，增加通道并不自动增加
模态总权重，不能把其谱直接称 Fisher 信息。检查旧弱方向在新增通道上的响应，
并在预定代表单元上做非线性扰动验证。

新合成在同一驱动/初态/噪声下比较正确、置换及相同秩的非物理读出，固定 EEG
权重以免选到零权重后将相同模型误读为几何无效。K2 自由模型与同一读出下驱动、
初态严格相等的模型配对；生成器含可分双源、相同双源与失配。隐藏预测、真实
动态恢复、弱方向和失败分母分别解释，不以可见拟合下降授予更多生理源。

### 三：跨模态独立信息与检出能力

在同一 Hb 供体与目标上新增固定 EEG 权重的真实配对、同条件错配以及非环绕
时间移位实验；不同移位比较使用相同有效支持，丢失端点不补零冒充观测。Hb-only、
原始空间 ridge 和 EEG+Hb ridge 使用相同输入边界；selected-weight 历史结果只作
策略参照。主终点是整对隐藏 Hb 的被试配对误差差异，原生目标和参考仅评分，不能
进入归一化、特征或拟合。弱方向变化只作为与隐藏预测并列的诊断。

独立合成以训练记录背景和已知共享动态构造预定信号剂量，固定各剂量间噪声，
分别报告零注入假阳性与各剂量的检出比例及其生成单元数。有限重复的经验检出率
不能称总体功效保证；阴性结论只在已校准的动态、背景、观测和模型范围内解释。

**条件独立信息的定位补充。** 原注入改变共享动态强度，但供体也可能看见同一
动态，不能预设它必定增加 EEG 的条件增量。独立版本
`crossmodal_information_positive_v1.yaml` 在同套件的新运行中设置有利的已知
读出、模型匹配合成控制：目标仅看第二动态，供体分别覆盖两个动态或仅第一动态；
移除供体第二动态时以已知增益保持第一动态贡献不变。正确、错配 EEG 与 Hb-only
使用相同目标及支持，零/非零剂量固定同一噪声和初态。训练生成身份确定共同尺度
与增益，校准和评价生成流独立；有限矩阵、既有求解预算和失败处理在补充运行前
冻结。此控制检验分析链在确有目标独有信息时的行为，不能替代实测背景的功效，
也不覆写原阴性或按其结果修改实测权重。报告并列区分共享动态与条件独立增量。

### 四：真实参数变化与观测失配的归属

新合成固定驱动、初态与噪声作配对反事实，独立改变 tau、共同 Hb 增益和额外
共同成分，保留组合条件。比较固定/释放 tau、无/有有限共同项以及知晓真实失配
或驱动的 oracle；每个 tau 节点重新拟合实际未知量。并列呈现带噪观测拟合、无噪
物理 Hb、driver、tau 与共同成分恢复，并统计“拟合改善而恢复恶化”的配对现象。
共同项对观测失配的吸收不能命名为真实浅表来源；实测归属证据保留其来源未决状态。

### 执行、统计与报告验收

先运行针对性软件、导数、隐藏污染、形状与已知真值检查，再用独立 pilot 测量
耗时/内存并确定并发。较大阶段在 user systemd 下运行、确认 linger，限制数值库
线程与在途任务，不干扰已有下游作业。各套固定矩阵独立完成，不按评价结果临时
扩大参数网格或挑选较优生成身份；数值失败、域失败和不可用单元全部保留。

实测先窗口/记录配对、再被试内汇总，bootstrap 整块重采样被试；合成以独立生成
seed/identity 重采样，同一 seed 的反事实和多个窗口保持成组。各套预先指定主比较，
其他结果标探索性；区间条件于冻结训练坐标和当前生成/评价面板。成功子集与完整
失败分母并列，不用成功子集的好拟合掩盖困难单元。

四篇报告各自说明研究问题、数据/真值类型、可见支持、生成与推断路径、公式符号、
尺度与指标、计划/可用/独立分母、基线差异、数值失败、机制解释及未解决边界。
每个图表逐项可追溯到本套结果；图与公式以高分辨率 PNG 嵌入 PDF，正文、表格和
图注可搜索。检查 PDF 图像对象、文字边界及实际尺寸可读性；交付时在对话中逐套
汇报主要数值和结论，并本地提交代码、合同、精简证据和四篇报告。

<a id="downstream-rebench-v1"></a>

## 下游任务性能重新获取（2026-10-16）

`DOWNSTREAM-REBENCH-v1` 是独立版本的设计方案，目标是在统一测量输入、训练支持和
评价口径下重新获取对比方法的下游性能。本节是设计 owner。最初请求限于设计；
2026-10-16 后续请求明确授权本版公开数据 smoke、两批完整训练及自动汇总，
并选择持续运行、不设总时限。核定后的可执行合同为
[`downstream_rebench_v1.yaml`](../comparative_methods/downstream_rebench_v1.yaml)，
实现和恢复入口见[比较方法说明](../comparative_methods/README.md#downstream-rebenchmark-v1)。
旧 [比较协议](comparisons/PROTOCOL.md)、[最终结果](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)
和方法内冻结文件保留原身份；本方案不恢复旧 queue，也不改写历史数值。

<a id="frozen-downstream-input-audit"></a>

### 冻结下游缓存的适配与短程筛查（2026-11-05）

用户要求检查仍在运行的公开下游生产结果，并先以短测试确定可用于后续正式生产的
加载和适配方法。独立合同为
[`downstream_input_audit_v1.yaml`](../comparative_methods/downstream_input_audit_v1.yaml)，
复用现有比较调度器，写入比较方法所属的 analysis run。旧生产源码、配置、预测及
未完成分母保留；修正输入不会自动迁移旧训练或把旧 Visual EEG 分数变成有效通道结果。

本轮输入、任务目标和被试分区完全沿用新冻结缓存的 `input_plan.json`。只读取训练与
验证信号，测试分区只核对身份和数量，不参与模型选择。Visual 两个探头共享的 EEG
和标签合并为一个事件，Hb 以真实探头及色团身份拼接，缺失探头保留 mask；所有方法
共用同一事件分母和正确的 16 EEG 通道面板。完整原生通道容量另列；只替换旧流程选错
物理电极的控制保持事件、标签、切分、时间及 Hb 不变。DSR 仍为 2 秒，REFED 仍为
真实 1 Hz、30 点、逐坐标带 mask 的连续目标，其余任务保持事件 `[-5,25)` 秒。

固定诊断比较整体统计、事件局部输入、分段统计、事件相关电位、类别平衡、训练尺度
与样本内无标签尺度，以及冻结编码器的平均／保留空间／有序读出。少量任务再执行
固定预算的短程监督训练，并检查 BrainFusion 的已拟合组件与 OOF stacking。
用户随后要求快速补齐 STA-Net 的 WG、n-back、DSR；
[`downstream_sta_completion_v1.yaml`](../comparative_methods/downstream_sta_completion_v1.yaml)
沿用此处已保留的训练／验证数组、同一 STA-Net 冻结源码、种子、训练尺度和
最多 30 轮／至少 15 轮／耐心值 10 的选择规则。每个任务先做实际形状 smoke，
再并行完成短程监督拟合；DSR 保留 2 秒输入，另外两项保留 30 秒。
结果写入独立补齐运行，不改写原输入或旧结果；ACC、F1 与被试间 SD 由同一
保存预测复算后进入新版表格，不以旧生产口径补格。
EFRM 复用已完成且排除整个目标数据集的 v5 预训练，明确标为旧输入域向新缓存迁移，
不声称已按新缓存重新预训练。高维线性读出可在训练行空间求解，保留等价的 L2 惩罚，
并以数值等价测试核验；验证输入不参与基底拟合。

切分影响只作开发性敏感性检查：旧外折 0 内折 0 的身份与新训练／验证池取交集，
所有尺度与读出重新由该控制的训练子集拟合。由于比较被试和训练支持不同，分数差
不能称为配对增益，也不据此挑选一个更有利的切分。新划分始终由缓存 owner 决定。
缓存已排除的时钟／标签支持及没有合规任务样本的被试均显式记录，不作为模型失败
重新删样本。分类使用原有被试等权 macro-F1 与精确有限样本 train-prior/majority B0；
REFED 保留 record→subject CCC 和原生单位误差。所有候选、负结果、失败及收敛限制
均保留。验证集选出的数值只支持后续生产方案，不能直接进入最终测试排名。

同日代码审计确认 BrainFusion 的保留配置把 `hrf_tr=1 s, oversampling=1` 的核直接
应用于 10 Hz 共同信号网格；CPU/GPU 一致性检查未覆盖这个物理时间错误。
独立的 [`downstream_input_followup_v1.yaml`](../comparative_methods/downstream_input_followup_v1.yaml)
因此保留原诊断作为对照，仅以 0.1 秒 HRF 核步长重新拟合五个支持任务，固定其余输入、
划分、种子及 CSP/OOF/stacking 参数。新版输入的 BrainFusion 默认按共同物理网格生成
HRF，旧样本索引行为仅作显式保留控制。该补充还以同一短预算检验 STA-Net 对 REFED
真实 30 点目标的适配，先做形状、mask 和短程 smoke；不把分类 embedding 复制成轨迹。
补充由同一入口与调度器在原运行位置的独立 phase 执行，不修改已冻结的首轮任务或结果。

### 纯 fNIRS 方法开发测试（2026-11-05）

用户决定保留 CBraMod、STA-Net、EFRM、BrainFusion，并要求直接测试纯 fNIRS
候选；明确接受 MtReNIRS 的预训练语料重叠。独立合同为
[`downstream_fnirs_probe_v1.yaml`](../comparative_methods/downstream_fnirs_probe_v1.yaml)，
复用现有 `audit_downstream_inputs_v1.py` 的 probe 入口、持久调度和独立复算。
仅使用已保留的原训练／验证人口，在六个分类任务上按原规则构造三个被试开发折；
不读取原测试信号，不改变正在运行的旧生产矩阵。各任务全部样本和失败均保留。

纯 Hb 对照包括六段均值／SD／斜率与线性、RBF 读出，以及 HbO/HbR 分块
Ledoit–Wolf 协方差、训练折拟合的黎曼切空间和逻辑回归。分块协方差在单窗内
按色团的共同 RMS 缩放，缺失值零权重，收缩与小对角项保证正定；不把填充值当
新观测。MtReNIRS 使用固定官方权重、128 维 token 及有序／均值线性读出，
只读取 Hb 信号；模板几何可以使用已登记的 EEG 位置名称建立坐标系，但不输入
EEG 信号。所有人口特征尺度、黎曼参考和分类器仅在对应训练折拟合。

共同 Hb 保留 10 Hz、0.01–0.2 Hz；MtReNIRS 采用当前单窗内归一化和 ±6 截断，
显式区别于上游 0.01–0.5 Hz 与整记录归一化。30 秒输入保留三个时段，DSR 只用
实际两秒后补零至十秒，不增加上下文。REFED 尚无经过验证的逐秒回归适配，列为
本轮不支持。Simultaneous 语料重叠只作来源标记；该开发比较不作无重叠迁移声明。
指标为被试等权 ACC、固定类别 macro-F1、被试 SD 及同折无信息基线；有限候选
全部报告，开发候选中的最高值不替代最终五折评价。原始结果和进度由注册表所链接
的单一 run 持有，不在方法 README 维护另一份状态。

<a id="downstream-rebench-v2"></a>

### DOWNSTREAM-REBENCH-v2：完整指标与 Visual／REFED 修复（2026-11-05）

#### 当前快速分类修订：冻结表示与固定验证集

用户随后将方法组合调整为 **CBraMod、STA-Net、EFRM、BrainFusion、MtReNIRS**，
明确选择复用开发缓存、只补分类读出、无编码器微调，并将 REFED 改为二分类。
执行合同为 [`downstream_frozen_classification_v1.yaml`](../comparative_methods/downstream_frozen_classification_v1.yaml)。
本修订优先于下方保留的全体被试五折／回归修复设计；不把快速开发结果称为已完成五折。

REFED 每个原生 30 秒窗只对有效 arousal 点求均值；原生值 **≤128 为低类，>128
为高类**，对应 `(arousal-1)/254` 在 0.5 处分割，等号归低类。保留中性窗和部分
尾窗，不按分数选择阈值、视频或被试；无有效 arousal 的窗显式保留为无标签。
原连续标签和回归结果不变。原论文提醒中性点不能自然划入低／高，因此此处是
用户指定的新操作性分类定义，不能解释为成功恢复动态情绪轨迹。

所有方法使用原固定训练／验证被试和同一评分身份。已有 STA-Net 监督模型是在
原训练人群拟合的，不能将它用于覆盖训练人群的新交叉验证折。六项原分类任务
复用无未解决收敛问题的匹配 16 电极 LP、STA-Net 和物理 HRF 网格 BrainFusion
预测；已有 LP 候选按原验证 macro-F1 选择同一候选的 ACC/F1/SD，明确保留选择偏差。
MtReNIRS 复用冻结特征，补固定 α=1 的 ridge 分类读出。REFED 使用冻结 CBraMod、
目标数据集排除的既有 EFRM、官方 MtReNIRS 表示，以及已有回归 STA-Net 的冻结隐藏
表示，均只拟合相同固定分类读出；STA-Net 原验证回归选 epoch 的影响明确标注。
BrainFusion 在训练被试内重新拟合其监督 CSP/NVC/stacking 分类器，并显式验证
REFED 30 秒及 DSR 2 秒适配，不增加时间上下文或神经编码器微调。
REFED 的 MtReNIRS 位置适配按登记论文平面图选择最近的命名 EEG 电极，再使用该
名称的模型模板坐标；这是粗略邻域映射，不是个体三维配准，完整映射随特征保留。

主指标为被试等权 ACC 和固定词表 macro-F1，各自报告被试间样本 SD（`ddof=1`）。
保留同训练支持的多数类 ACC 和多数类／训练先验随机预测较强者的 F1 基线。
所有新增读出固定单一配置，无新的验证集网格搜索；原测试信号、目标及预测均不读取。
代码、标签、掩码及 synthetic 检查后，用既有持久调度器完成缺项和独立预测复算。

#### STA-Net 源实现核对与有界优化（2026-11-06）

用户随后要求进一步优化 STA-Net，并检查低分是否来自不充分的复现。该请求授权
STA-Net 在同一公开开发支持上从随机初始化重新做监督分类；不微调其他预训练
模型，也不开启原测试。执行合同为
[`downstream_sta_optimization_v1.yaml`](../comparative_methods/downstream_sta_optimization_v1.yaml)。

逐项对照作者固定版本 `b6db8bb5eb2f6491a13f0938880ee70e32162ee7` 的 `sta.py`、
`run_sta_net.py` 和其声明的 Keras/TensorFlow 2.10 语义。新增显式选择的
`source_aligned_v2`，保留原 `model.py` 与旧结果：位置编码在 512 维时滞特征上
加入后再投影，采用对应的 HeUniform／三维注意力核 Glorot 初始化、channels-last
展开、BatchNorm 的 epsilon/移动统计和加法 epsilon 相关损失。组件数值核对、
常量输入梯度、标签与掩码、时间支持及已知分类信号学习先于实测拟合。

主候选保留作者 FGSA/EGTA、EEG 辅助头与可学习融合权重。普通事件在共同窗口内
读取事件 `[0,3)` 秒 EEG 与 `[0,13)` 秒 Hb；DSR 仅实际 `[0,2)` 秒，两模态均不
延长；REFED 保留完整 30 秒输入并直接训练固定 arousal 二分类头。Visual 同时
保留全窗口配对控制，REFED 同时保留原实现的直接二分类控制，区别任务训练和
源实现修正。三个困难任务另设对齐损失权重 0.1／训练类别平衡候选，不用新标签、
刺激身份或更多输入换取分数。所有候选与失败均保留，不把联合训练的内部支路
当作独立单模态基线。

训练采用真实 batch=32、FP32、Adam 的作者默认超参数；保留有限轮数和梯度上限
作为明确的工程差异。seed17 的固定验证 macro-F1 只用于开发候选和 epoch 选择，
DSR、Visual、REFED 随后固定所选候选重做 seed42/73，不按复核种子重选或只报最好
种子。最终先算每种子／每被试 ACC、macro-F1，再对种子取均值，最后给被试均值及
样本 SD；种子 SD 单列。旧参照只有一个种子，差异为开发描述，不称确认性显著改善。

完整作者 TensorFlow 数值轨迹、作者被试内 session 协议、原始预制 NPZ 与 300 轮
训练不在本快速执行范围；代码核对通过与跨被试性能提高分别判断。仍低于基线
或未收敛时如实报告，不用绝对论文分数替代当前协议验收。

#### 保留的五折回归修复设计

本版响应用户要求：解释不同方法落后强简单基线的原因与频次，修复 Visual 的近随机
表现及 REFED 的无效回归，并最终交付完整 ACC、F1 和明确单位的标准差。用户已选择
**按被试五折评价**。本节持有设计与执行顺序；数值准入及两项修复的验收条件统一由
[指标准入规则第 7 节](comparisons/METRIC_ACCEPTANCE.md#downstream-v2-repair-acceptance)
持有。本节先形成设计；用户随后明确要求通过 probe 快速迭代直至获得可用结果。
开发执行由既有诊断入口和逐轮版本合同承接，运行状态由注册表持有；不能把启动
probe 或完成软件修改写成已经取得可用的最终指标。

首轮执行合同为
[`downstream_repair_probe_v1.yaml`](../comparative_methods/downstream_repair_probe_v1.yaml)，
仅复用已保留的原训练／验证数组，三开发折覆盖 Visual 12 名、REFED 25 名被试。
每个固定候选输出全部留出被试预测，神经读出使用固定轮数、不按留出折选择最佳
epoch；视频时间模板明确为带刺激身份的诊断，不能冒充信号模型的修复证据。
第二轮合同为
[`downstream_repair_probe_v2.yaml`](../comparative_methods/downstream_repair_probe_v2.yaml)，
在同一开发人口、同一三折和 30 秒输入内检验共同平均参考、窗口内尺度、事件前
协方差参考及整段轨迹读出；被试内独立 Part／视频留出仅作可学性控制，单独保存
适用分母与缺失记录。两轮结果及独立复算入口由结果索引持有，不替代最终五折指标。
原生 Hb 预处理查验由 [r3 合同](../comparative_methods/downstream_repair_probe_v3.yaml)
持有，仅对同一开发人口和窗口调用中央原生读取器；先重放缓存的处理链，再固定
比较上限 0.7 Hz 及保留慢变化的低通版本，不覆盖缓存或扩大测试信号访问。
[r4 合同](../comparative_methods/downstream_repair_probe_v4.yaml)仅修正该查验中 Visual
的原生 Time 网格重放，保留 r3 失败；REFED 不重复拟合。
[r5 合同](../comparative_methods/downstream_repair_probe_v5.yaml)检验真实 CBraMod
事件编码器 FT、属性辅助损失及 REFED 的短轮数强正则化；训练轮数和候选在
拟合前固定，不能依据留出被试挑 checkpoint。
后续保留 [r6](../comparative_methods/downstream_repair_probe_v6.yaml) 的小批量优化失败，
以 [r7](../comparative_methods/downstream_repair_probe_v7.yaml) 单独核查编码器 Dropout
造成的训练／评价差异；参数是否真正更新和留出成绩分别核验。
[共享刺激轨迹学生对照](../comparative_methods/downstream_repair_probe_v8.yaml)只在训练
人口构造教师轨迹，推断不输入留出视频身份或绝对时刻，并保留 mask-only 控制。
[个体无标签校准诊断](../comparative_methods/downstream_repair_probe_v9.yaml)单列额外
信息条件：每名开发被试仅用自身全部保留输入估计特征均值／SD，不使用其标签，
后续人口尺度仍只拟合训练被试。这是离线个体校准，具有超出单个 30 秒窗口的
无标签上下文；不能将该条件的成绩替换原严格单窗主表或据此宣称原任务已修复。
[原生 EEG 频带诊断](../comparative_methods/downstream_repair_probe_v10.yaml)在同一
开发被试、同一物理电极和窗口上重放 1–45 Hz，并比较 0.1／0.01–45 Hz；三条件
使用相同电位特征和模型，保留眼动等原生伪迹。更低高通会改变全记录离线滤波的
原始时间支持，不能称实时预测或纯皮层信息恢复；原缓存不改写。
[EFRM 时序表示诊断](../comparative_methods/downstream_repair_probe_v11.yaml)使用已完成的
全 REFED 排除预训练权重，每个开发折单独拟合训练被试的输入尺度，再提取对应
30 个时刻的冻结表示，比较线性岭回归与固定 5／15 轮时序读出。它是旧 v5 输入域
权重向当前输入的迁移对照，不是新输入重新预训练或编码器 FT；不复用旧划分的人口尺度。
[EFRM 编码器适配诊断](../comparative_methods/downstream_repair_probe_v12.yaml)从同折固定
5 轮时序头出发，以冻结的训练侧输入／特征尺度继续训练两轮；更新 EEG 与 Hb
编码器，保持预训练解码器冻结。保留梯度小样本检查、两路编码器参数变化和完整
留出预测，区别于仅训练时序头；它仍不替代计划中的新输入预训练。
[Visual 紧凑事件读出](../comparative_methods/downstream_repair_probe_v13.yaml)针对原任务
决定时刻在 9 秒内变化的条件，比较原始 EEG 与冻结 CBraMod 每秒表示上的小容量
共享时空卷积。两者仅取目标事件 0–12 秒、保留六个有序平均／最大池化区间，
窗口内归一化仍限于原 30 秒；人口尺度只用当前训练被试。该对照检验时刻表示与
读出容量，不把冻结编码器读出称作 FT，也不把两项二元辅助损失替代原四分类目标。
[额部低频投影诊断](../comparative_methods/downstream_repair_probe_v14.yaml)保持 r13 的
模型、训练轮数、事件和目标不变，只在各自原 30 秒窗口内，以 7 Hz 低通后的
Fp1/Fp2 拟合带固定微小岭惩罚的通道投影并相减，随后使用相同归一化。保留逐窗、
逐通道的移除量；不删事件、不更改评分 mask。额部电极也包含神经活动，因此该
代理不等于实测 EOG，投影后的波形不能称为已核验的纯神经真值。
[EFRM 更新幅度对照](../comparative_methods/downstream_repair_probe_v15.yaml)针对 r12
训练损失下降很小的现象，保持同折初始化、两轮预算、头部学习率及输入／特征尺度
不变，只将编码器学习率从原来的 1e-6 分别提高到 1e-5 和 1e-4；先检验较大值的
小样本梯度与数值稳定性。训练拟合是否改善与留出性能是否改善分别报告。
[BIOT 源导联子集对照](../comparative_methods/downstream_repair_probe_v16.yaml)只用于
能够精确构造源导联的 Visual 面板；在同一 16 个物理电极预算内，先计算八条
额－顶方向的原生电位差，再拟合当前训练被试的均值／SD，调用官方编码器的
`n_channel_offset=8` 对应 PREST 第 8–15 号槽。与原单电极按序迁移使用同样的窗口、
被试折和两项线性头正则化对照；没有近邻电极替代，也没有扩大通道预算。这是源
16 导联中的精确八条子集，不是完整源蒙太奇复现；该对照同时改变参考与槽位语义，
两者的单独贡献不能由这个比较直接区分。
[Visual 核正则化对照](../comparative_methods/downstream_repair_probe_v17.yaml)针对已有
核分类器训练拟合强、留出成绩弱的现象，复用 r1 事件电位／协方差及 r2 相对频带
特征，只比较更小的 C 与更宽的 RBF 核。各候选、全部折和原被试等权／类别权重
均保留；开发网格最高值不作为独立确认，也不替代命名参照的修复要求。
[REFED 记录内动态损失对照](../comparative_methods/downstream_repair_probe_v18.yaml)固定
原始多模态特征与同折 EFRM 冻结时序表示的 15 轮读出，比较 MSE 加权重为
0／0.01／0.05 的记录内 CCC 损失。各批次以完整被试－视频记录为单元，零权重
对照保持相同批次组织；不跨视频计算相关。真实恒定坐标仅从动态项排除，继续
参与 MSE 和全部误差评分。该组只更新读出，不将其写成编码器 FT。
[REFED 记录目标微调对照](../comparative_methods/downstream_repair_probe_v19.yaml)从 r18
同折、固定 15 轮、动态损失权重 0.05 的 EFRM 头初始化，继续相同的完整记录损失
两轮；比较两路编码器共同更新与编码器保持冻结的配对控制。头部学习率、批次、
尺度和目标支持一致，完整记录之间累积梯度，不将视频拆开计算 CCC。最长开发
记录先作梯度／显存检查；冻结对照必须核验编码器参数零变化。
[Visual 类别权重对照](../comparative_methods/downstream_repair_probe_v20.yaml)补齐原计划中
类别平衡与原损失的比较：复用 r1 事件电位／协方差和 r2 相对频带核分类器的
特征、C、核宽度与被试折，只移除训练类别逆频率权重，继续保持每名训练被试
的总权重相同。F1、ACC、四类召回和全部事件分母不变；不使用留出类别先验。
[REFED 单模态时序对照](../comparative_methods/downstream_repair_probe_v21.yaml)保持 r18
原信号读出的记录损失权重 0.05、15 轮训练、网络宽度与被试折，分别只输入
EEG 频带或 Hb 水平／斜率特征，与保留的联合输入结果比较。该消融检查联合
输入是否损害泛化，不增加时间上下文，也不改变目标、评分或全部记录支持。
另由 [固定目标数值复算合同](../comparative_methods/downstream_convergence_probe_v1.yaml)
重放原选中的 BIOT／REVE／EFRM 线性头：保持原 C、类别权重、特征与原训练／验证
划分，使用 Newton 求解、解析梯度及从该解启动的 L-BFGS 核对，不能将其分数
直接与三个开发折作差或称重新选型后的最终性能。

#### 依据：问题出现得有多普遍

以下计数来自同一新缓存、同一验证被试及正确 16 电极面板上的
[逐方法差距分析](../comparative_methods/runs/performance_analysis/20261105_downstream_input_audit_v1/final_plan_analysis_v1.json)
和[逐任务表](../comparative_methods/runs/performance_analysis/20261105_downstream_input_audit_v1/final_plan_gap_cases_v1.csv)。
每个方法／轨道取本轮已保留适配候选中的最高验证 F1，与同任务最强的正确面板信号
基线比较；排除旧电极、原生全通道及旧切分控制。EEG 方法另外与 EEG-only 强基线
核对，频次结论相同。“低于”仅指点估计小于，不等于显著差异。未做的任务不计入
分母；不同任务不是用于检验方法优劣的独立重复。
方法与信号基线的候选数量也不相同，在同一个开发集上分别取最高值会带来选择偏差；
因此这些频次用于定位当前配置的问题，最终差距必须在内折选型后的外折上重新计算。

| 方法与轨道 | 低于强基线／已测分类任务 | 落后至少 5 个百分点 | 不高于 B0 | 最优线性头有收敛警告 |
| --- | ---: | ---: | ---: | ---: |
| BIOT LP | 6/6 | 5/6 | 2/6 | 6/6 |
| CBraMod LP（已含有序读出） | 4/6 | 1/6 | 1/6 | 0/6 |
| REVE LP | 6/6 | 5/6 | 1/6 | 2/6 |
| EFRM LP | 6/6 | 5/6 | 0/6 | 1/6 |
| CBraMod 有序短程 FT | 2/3 | 0/3 | 0/3 | 0/3 |
| STA-Net 短程监督 | 2/3 | 2/3 | 1/3 | 0/3 |
| BrainFusion，物理 HRF 修正后 | 5/5 | 4/5 | 2/5 | 0/5 |

原 CBraMod 平均头短程 FT 为 3/3 落后，三个差距均超过 5 个百分点；仅作已诊断的
控制，不混入有序头主行。CBraMod LP 落后的任务为 MI、WG、DSR、Visual，其中只有
DSR 差距超过 5 个百分点。以上均经开发集选型，不能当最终测试排名，也不能把短程
FT 和充分训练的端到端性能等同。

**已证实的因素与待检验解释分开处理。**

| 证据 | 当前判断 | 下一步对应动作 |
| --- | --- | --- |
| 固定输入、尺度和 C=1，CBraMod 保留顺序使五项任务增加 10.9–21.5 个百分点 | 读出适配确实低估了部分表示；大头训练达到 1.0 也提示过拟合 | 保留位置／事件信息，限制头部容量；读出在内折选择 |
| BIOT 六个最优 LP 头均触及迭代上限，REVE/EFRM 也有此问题 | 数值求解未闭合，不能先判断表示无用 | 先完成相同目标的收敛修复及独立求解对照，再作方法评价 |
| BIOT 使用原生电极顺序迁移原预训练通道槽；REVE 使用全局注意力池化；EFRM 为旧 v5 域向新 raw 输入迁移 | 导联语义、事件读出、预训练域是具体候选解释，尚无因果归因 | 各方法逐项做有限对照，不把 CBraMod 的结论直接推广给它们 |
| 强简单基线显式保留事件后电位或六段统计 | 任务特定时间先验有效；也可能利用提示／决策反应 | 核查刺激、决策与持续任务期的输入支持；方法获得相同信息，解释限于任务解码 |
| Visual 的有序短 FT 训练 0.9564、验证 0.2573 | 大训练／验证差距，通道纠错与增加容量不能单独解决 | 事件局部表达、类平衡、分层目标及跨被试校准分别检验 |
| REFED 效价的逐记录 Pearson：EFRM 0.0487，STA-Net 0.0276 | 动态关联也弱，不只是 CCC 对均值／幅度失配的惩罚 | 保留时刻表示，修复损失／评价对应关系，检查跨被试与时序信息 |

按 [Visual 原始说明](<../data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/readme.txt>)，
四类是“编码时是否想记住 × 随后是否记住”的联合行为标签，不能视为四种即时刺激。
由现有四类预测拆出的诊断显示：CBraMod 有序 FT 的意愿 F1=0.5182
（B0=0.4969），随后记忆 F1=0.4819（B0=0.4992）；平均头甚至把所有事件都预测为
随后记住。二元分量只用于定位问题，最终目标仍为原四分类。

REFED 的 STA-Net 现有实现先对主干时间维平均，再由全局表示输出 60 个数；它不是
常数复制，但仍缺少与 30 个目标时刻对应的局部读出。现有 worker 按 CCC 选轮次，
并不约束原生单位误差优于训练均值。损失又按有效时点加权，而评价按记录、被试
等权，这些差异均需单独检查。两个唤醒记录的真实目标恒定，Pearson 未定义，已保留
在 90 个验证记录的分母中，没有删除。

[REFED 原论文 §4.2、附录 E](https://papers.neurips.cc/paper_files/paper/2025/file/2bf0ed7c35d9d84128e7f7c72ab76402-Paper-Datasets_and_Benchmarks_Track.pdf)
使用被试内的视频三折，标签从 [1,255] 变到 [0,1]，并非本版的跨被试评价。
其信号特征和频带也不同；不能用量纲转换后的较小 MAE 或被试内结果宣称跨被试修复。

#### 共同数据、五折与方法矩阵

1. **输入身份。** 继续使用 `physiology_downstream_v1/input_plan.json` 的原生样本、
   支持和排除理由，不回写缓存的 60/20/20 分区。新实验在自己的 `splits.json` 中
   将同一批公共被试分配到五个外折；所有 session、record、Probe、相邻窗口及派生
   标签随被试移动。同一数据集的不同任务共用被试折号，DSR 的有效被试覆盖在
   元数据建折时均衡，不能因某个方法表现不好重新排折。
2. **范围。** 2026-11-05 元数据中，MI/MA 各 29 名被试，WG/n-back 各 26 名，
   DSR 17 名，Visual 16 名，REFED 32 名。Visual 的五折将覆盖全部 3,784 个合并
   事件，REFED 覆盖 1,888 个原生 30 秒窗；最终解析合同以 cache owner 的身份集合
   逐项断言，不能只检查数量。这里仅检查了测试部分的元数据，没有读取测试信号。
3. **选择与最终拟合。** 外折只产生一次锁定配置的折外预测。每个外训练池内再分
   三个被试内折，训练尺度、监督目标尺度、类别权重、特征基底、读出、超参数与
   训练轮数都只在该池内确定。内折指标按被试等权合并，不平均大小不同的 fold
   均值。最终从初始化重新拟合整个外训练池；轮数取选中候选三个内折最佳轮数的
   中位数上取整，不在外折标签上 early-stop、改阈值或调参。
4. **身份与重复。** 新 split seed 固定为 `20261105`，只生成一次。选择 seed=17；
   随机训练的最终 seeds 固定为 `[17,42,73]`，包括随机森林／预训练带来的随机性。
   官方固定编码器的确定性 LP 不复制相同拟合来制造 seed SD。REVE 的 MI/MA
   保留已知预训练语料重叠标记；公共数据曾被研究过，不称全新的未触碰确认集。
5. **信息预算。** 主比较继续使用共同正确 16 电极及各任务真实 Hb 面板；同任务
   所有方法的起点、可见时长、mask 和评价样本一致。方法内部可作已声明的事件
   读出或无标签归一化。原生全通道、BIOT 所需原生双极导联重构等容量／源兼容
   控制单列，不能换入共同 16 通道主表。不能用不匹配真实电极的预训练槽名冒充
   导联对齐，也不通过补造未测电极满足方法要求。

| 方法族 | 轨道 | 分类任务 | REFED | 计划最终单元 |
| --- | --- | --- | --- | ---: |
| 强信号基线 | EEG、Hb、双模态分别 LP；有限特征／核候选在内折选择 | 六项 | 时序回归 | 21 |
| BIOT、CBraMod、REVE | 各自 LP 与 FT 分表 | 六项 | 不伪造不支持的序列头 | 36 |
| EFRM | target-dataset-excluded 预训练后的 LP、FT | 六项 | 两轨均有真实时刻读出 | 14 |
| STA-Net | 从头监督的明确任务适配 | 六项 | 时间维保留版本 | 7 |
| BrainFusion | HRF 修正后的监督版本 | MI、MA、WG、n-back、Visual | 不支持 | 5 |

核心矩阵共 83 个单元；NormWear 不因复制旧 YAML 而恢复。按五外折及上述种子规则，
确定性 39 单元各一次／折，随机 44 单元各三次／折，即 **855 份最终评价产物**。
EFRM 的四个目标数据集排除版本 × 三个种子，另有 12 份预训练产物；内折训练、
数值复核和诊断不冒充额外独立评价样本。全部作业由解析后的单一 DAG 持有分母。

#### 阶段 A：先闭合指标和数值合同

- 在现有 `src/metrics/downstream_benchmark.py` 增加逐被试 ACC，保留原 macro-F1；
  当前诊断的 pooled ACC 只能作旁证，不能直接与被试等权 F1 共用误差条。
  在现有 aggregator 支持从 v2 解析合同读取 REFED 的 30 点形状；v1 的固定
  20 点重放保持严格不变。先用不同样本量被试、缺失类别、重复 Probe、部分尾窗和
  已知预测的小 fixture 验证所有均值与标准差，再做真实数据拟合。
- 线性头沿用同一标准化、L2 目标、C 与截距语义，先提高数值精度／迭代上限，
  检查梯度、目标下降及无收敛警告。必要时使用训练行空间等价求解或第二求解器；
  不把改变正则化、验证拟合基底或删除难样本叫作收敛修复。固定 C 的独立高精度
  概率／目标值对照通过后，才执行新的性能选择。所有未收敛候选及其原预测保留。
- 以训练小批过拟合、已知类别／已知连续轨迹的合成正控、输入／标签置乱、局部
  时间脉冲与完整整数索引回放检查监督链。真实模型必须对输入改变有响应；形状
  正确、loss 有限和 GPU 使用正常不能代替这一检查。

#### 阶段 B：面向各方法的有限适配修复

| 对象 | 固定比较与修复动作 | 保留的反例／解释边界 |
| --- | --- | --- |
| 强简单基线 | 保留 ERP、六段统计、EEG 频带对数功率／DE、Hb 水平及斜率；线性与小网格 RBF 读出；各模态独立选型 | 不能仅保留弱整窗平均；不得只给基线较少传感器或较短标签支持 |
| BIOT | 先收敛；核对官方预处理、通道槽语义与实际导联；比较原均值与固定事件分段读出，保持编码器身份 | 无法实现原导联时明确为电极迁移；源兼容双极面板另列，不把它的成绩填进共同面板 |
| CBraMod | 原生有序 token 线性头与受正则约束的分段头；FT 保留原实现 MLP 及训练尺度／样本内尺度有限对照 | 不以训练 1.0 为可用性证据；添加 ERP 特征的混合模型必须另命名 |
| REVE | 保留官方电极位置，比较 CLS-query pooling 与事件／分段 token 读出；只在 FT 解冻对应查询／主干 | 不能假设注意力池化已保留所需时刻；MI/MA 的语料重叠不撤销 |
| EFRM | 新输入坐标上重新做整库目标排除预训练，分类保留时段表示，回归保留原时刻 token；LP/FT 共用相同对应种子预训练 | 旧 v5 权重只作迁移控制，不代替本版预训练；源验证与目标内折选择分别持有 |
| STA-Net | 保留空间／耦合模块；REFED 去掉先全时间平均再展开输出的任务适配，采用与目标时刻相连的读出 | 这是新的序列适配，不称原论文原封不动复现；原全局头作为负对照 |
| BrainFusion | 保持 0.1 秒 HRF 网格，检查 EEG/Hb/NVC 各分支、OOF 预测及元分类器尺度；比较原自由权重与训练内约束的 stacking | 不依据外折标签翻转符号；单组件／约束变体另有身份，不覆盖原算法结果 |

本阶段先在已使用的开发分区上定位软件与表示问题，然后冻结每个方法的有限候选
集合。正式内折对所有候选使用相同选择规则；开发结果不能提前替代其折外评价。
不要求每个有效模型都胜过最强简单基线，但每个低分都必须区分数值失败、未适配、
过拟合与在已声明预算内的有效负结果。

#### 阶段 C：Visual 的专门修复

1. 复用原发布标签／事件核对和通道审计，逐项确认刺激开始、3 秒图片结束、随后
   决策及后测记忆结果的连接；Probe 只合并一次。决策类别、后测答案、图片 ID、
   时间戳编码和事件序号不作为模型标签捷径输入。
2. 在相同可见窗口中明确目标事件位置。固定比较全局读出与目标事件分段读出：
   EEG 的 0–3 秒编码、3–12 秒决策分别表示，Hb 保留声明的延迟支持及 mask；
   所有分段都从同一缓存窗口获得，不新增未来上下文。对重叠相邻刺激的影响作
   条件上下文诊断，不把整段 30 秒当作一个类别持续成立。
3. 固定比较训练类频率决定的平衡损失与原损失，以及四类联合头和“四类主头＋
   两个行为属性辅助头”（辅助损失权重只比较 0 与 0.5）。四类 F1/ACC 始终为主
   结果；两个属性分别报告召回，定位意愿或记忆哪个分量仍失败，不能用较容易的
   二分类结果替换原四分类。
4. 训练内按被试均衡采样；比较训练尺度与无标签样本内尺度，不使用外折被试的
   标签、真实类别先验或测试标签校准阈值。保留全部类别、被试、Part 和失败。
5. 以训练被试中可用的独立 Part 留出作正控，另做严格跨被试检查。若 Part 内可学、
   跨被试无效，重点转向泛化；若两者都无效，先排查目标时间和监督链。只有一个
   可用 Part 的被试保留为此诊断不可用，不把随机相邻窗切分当成独立正控。

修复后以 **CBraMod 的固定事件读出 FT** 为预指定命名方法参照，以独立的强信号
基线作复核；其余方法完成相同五折评价。不能在看过折外结果后改选最有利的方法
来通过任务级验收；替换参照属于下一版本。

#### 阶段 D：REFED 的专门修复

1. 逐坐标核对视频时钟、1 Hz 目标、[1,255] 标签范围、有效尾窗和输入／目标索引；
   以已知时间移动的合成信号验证秒级对应。训练使用统一且可逆的标签坐标，所有
   结果回到原生标注单位；量纲缩放不会被当作误差改善。
2. 先建立真正的逐秒强基线：各 EEG 通道的频带功率／DE、HbO/HbR/HbT 的局部水平、
   斜率及训练内选择的有限时间滞后，比较 ridge、RBF-SVR 和小型时序读出。
   对照原论文时明确当前 EEG 截止 45 Hz、Hb 截止 0.2 Hz，不能从现有数组恢复
   已被滤掉的更高频段。只有前述检查指向带宽损失时，才准备独立版本的原生
   带宽对照；不得给现有低通数组重复滤波后称已恢复信息。
3. EFRM 与 STA-Net 每个目标时刻读对应时间特征和已声明的上下文。固定比较当前
   全局头／逐点线性头与局部时序头，分别验证时间置换、局部干预和局部时间支持。
   不把 30 个可不同的输出数等同于已经学到动态。
4. 对齐训练与评价：先以坐标→记录→被试的权重训练 MSE；另一个有限候选增加
   记录内动态一致性项。CCC 项若使用，必须在同一完整记录的有效坐标上计算，
   不能把无关视频混成一条轨迹；恒定目标显式处理。轮次／候选选择同时检查
   CCC 与原生单位误差，不能只追求一个微小正 CCC 却使 MAE 劣于均值预测。
5. 将误差拆成记录内变化、均值偏置和幅度失配；分别报告 valence、arousal。
   增设只用训练被试标签构造的视频／时间模板作为刺激依赖诊断，视频身份不
   输入主模型。真实信号与同视频错配信号的比较用于检验是否得到信号增量。
6. 仅在训练被试内复查按视频整段留出的正控，区分数据／实现不可学与跨被试
   迁移困难。被试内成绩、带目标标签的个体校准及视频模板都不能替代本版
   严格跨被试回归结果。无标签输入校准若使用，支持范围和算法在外评估前固定。

修复以 **保留时刻的 EFRM FT** 为预指定命名参照，独立的强时序信号回归作为复核；
STA-Net 保留为另一命名方法。两坐标都需满足任务级验收，不能以效价平均收益掩盖
唤醒完全无效。

#### 阶段 E：冻结候选并完成全矩阵

- 优化器、学习率、权重衰减优先取各方法现有核验过的源配置。数值修复与结构／
  尺度筛查完成后，每个随机方法／任务／轨道最多保留六组优化候选；线性 C／
  ridge 强度沿用已有六点网格，类平衡为训练内选择。所有候选在执行前解析到
  合同，不按外折分数追加搜索。
- 随机训练的内折采用同一个 200 轮优化器时程；六候选先到 15 轮，再保留三个到
  50 轮、两个到完整预算，后段最少 50 轮、patience=30。晋级按三个内折的被试
  等权验证指标，记录所有淘汰者及训练暴露。续训不重置优化器、随机状态或 cosine
  时程；先做恢复等价测试。最终拟合重放该完整时程的选定前缀。
- EFRM 预训练仅使用整个目标数据集以外的公共输入，新坐标、新时长及来源清单
  先冻结；源验证按源被试隔离。选好预训练时长后重新拟合源训练支持，不看目标
  外折标签。一个目标排除版本可以供该数据集不同任务及外折使用，不重复生产。
- Visual／REFED 的开发修复证据未达到准入规则中的预检要求时，先完成该问题的
  固定修复面板，不盲目派发全部昂贵 FT。按用户后续要求，开发 probe 持续迭代，
  每轮先冻结有限候选、具体输入和停止点，完成后依据新增诊断决定下一版本；
  优先数值／时序／损失，再检查有证据支持的域、参考或带宽问题。全部尝试保留，
  不把反复使用的开发集当作独立确认。仍不达标时明确“修复目标未达成”，不能
  删除 Visual／REFED 或在打开最终折外结果后继续追逐显著性。
- 一旦进入锁定的五外折评价，完成所有计划方法／种子，不因分数低删节点。仅
  对可定位的实现或运行错误作版本化修复；受影响的完整配对单元重做，保留原
  失败。有效但不利的数值保持可见；不作 best-fold／best-seed 替换。

#### 最终 ACC、F1 与标准差的计算合同

分类对每名折外被试 s、训练种子 k 计算 `a_sk=正确事件数/有效事件数`，以及固定
类别词表的 `f_sk=各类别F1的算术平均`。同一被试所有事件先合并，再计算 F1；先
对种子取均值得到 `a_s,f_s`，然后对被试等权得到主表均值。不是 pooled ACC，
也不是先平均窗口 F1、大小不同的 fold 均值或预测投票后的 ensemble 分数。

- 主表并列给出 **ACC mean±SD_subject、macro-F1 mean±SD_subject**；SD 使用
  `ddof=1`，独立单位是被试。附表另列每种子的被试均值及 `SD_seed`、每折结果和
  被试区间。确定性 LP 的 seed 数为 1、`SD_seed=not_applicable_deterministic`，
  不复制预测制造三次重复。主表所有方法仍有同口径的被试间 SD。
- REFED 不强行补分类 ACC/F1。分别给 valence、arousal 及双坐标汇总的 CCC、
  MAE、RMSE、各自 SD_subject 与 SD_seed；按有效时点→坐标→记录→被试汇总。
  Pearson 及时间差指标作诊断。常量目标的未定义统计量单独列出，MAE/RMSE 仍
  覆盖全部目标；冻结共同的统计适用分母，不能按方法删除未定义或失败记录。
- B0 按每个外训练折计算：ACC 为训练多数类在外折的实际表现，macro-F1 为
  训练多数类和训练先验随机预测精确期望的较强者；REFED 使用外训练目标均值的
  实际外折误差。强信号参考也只能由相同内折选择，再在外折评分。
- 不确定性整体重采样被试，所有记录、窗口、seed 及各方法的配对结果随被试
  一起移动；5,000 次、seed=20261105。它条件于已拟合的外折模型；seed SD 不
  当作新被试，fold SD 不当作总体采样误差。任务级修复检验及多重比较规则见
  唯一准入 owner，不从 83 个单元中事后挑显著结果。

最终表每格同时保留计划／完成的 fold、seed、subject、event／record 数、ACC/F1
或回归指标、两类 SD、B0、强基线差、数值收敛状态、预训练范围及既有准入状态。
一张完整结果表覆盖所有 83 格，失败和拒绝值保留真实值及原因；正常性能排名只
使用符合既有准入规则的同轨道行。不能只交付最高分摘要。

#### 实现顺序、资源与交付

复用现有 data owner、worker、调度器、聚合器及各方法的 `rebench_v1.py`，添加
明确 v2 profile/schema 分支；旧源码快照、配置、签署材料及结果不动。新解析合同
使用 `DOWNSTREAM-REBENCH-v2`，结果位于既有比较产物根下的
`comparative_methods/runs/downstream_rebench_v2/<run>/`；不增加转发 launcher 或
第二套 manager。v2 的可执行配置在实现及 synthetic 检查时生成，当前不能把 v1
YAML 换个输出目录直接运行。

先指标／形状／信息边界测试，再已知信号 synthetic、实际输入 smoke、收敛与
两任务修复面板，最后五折完整矩阵。固定编码器的特征提取与 CPU 读出在同一 DAG
中分阶段，提取完成释放 GPU；缓存键包含训练身份和实际尺度，不能跨不相容内折
复用特征。检查断点续训与最终参数选择后再扩大并发。

所有较大阶段使用独立 user systemd／既有持久调度器。启动前核对 CPU quota、
物理核、RAM、GPU、NVMe 及旧生产占用；实测不同并发的每秒样本／训练步和峰值
内存，按吞吐选择并发，限制 BLAS 线程和在途任务。先用各方法代表任务测量单位
成本，解析全部内折、预训练和外折工作量后给出预计 GPU 小时及完成时间；不以
“855 个节点”掩盖节点内部的多次训练。不中断其他任务，不因会话中断重复生产。

每组结束在对话中给出完整分母、相对强基线结果、修复是否成立及下一步。最终交付
全表 CSV/JSON、逐被试／逐种子／逐折表、保存预测与独立 ACC/F1/SD 复算、Visual
混淆及属性诊断、REFED 原生误差与动态分解、运行资源与失败记录，以及可读报告。
指标汇总完成与 Visual／REFED 修复成功分别判断，不能用前者替代后者。

### MA 单任务适配查验：共同 30 秒输入（2026-10-29）

用户明确要求在单一任务上检查 CBraMod 读出/尺度、STA-Net 及 BrainFusion 的
适配，同时固定每个样本的输入，并确认窗口为事件前 5 秒至后 25 秒。
独立诊断合同为 [`ma_adaptation_audit_v1.yaml`](../comparative_methods/ma_adaptation_audit_v1.yaml)。
该诊断从原 MA 外折 0 的内折 0 提取训练/开发被试身份，在信号读取前重新核验
30 秒时钟支持和相邻事件；沿用中央 loader，不复用旧 8 秒样本 ID，也不读取
该外折评价被试。所有方法共享同一 16 通道 EEG 面板，双模态方法再共享相同
72 通道 Hb；缺失支持、排除和失败统一保留，不能按方法删样本。

这是单划分、单种子、有限预算的开发性机制检查。简单信号/覆盖率/标签置换对照、
CBraMod 的尺度与有序 token 读出、STA-Net 的 BatchNorm/dropout 单因素对照及
BrainFusion 已拟合组件诊断，均在共同输入上进行。模型内部无标签变换可以不同，
不得增加时间或传感器支持。开发集选择的最佳 epoch 不是未查看测试成绩；结果
不能直接与历史 8 秒五折均值相减或进入最终方法排名。现有完整重测继续使用其
冻结源码、合同和输出，本诊断由同一类调度器在独立运行目录执行。

### Visual 四分类失败调查（2026-10-29）

用户要求调查 Visual 未超过无信息参照的问题，并保留稍后判断是否纳入汇报的决定。
[`visual_investigation_v1.yaml`](../comparative_methods/visual_investigation_v1.yaml)
固定本次开发性调查：全部当前事件只作原生类别 Excel、EEG 注释和 Hb mark 的元数据
追溯；信号只读原 Visual 外折 0 内折 0 的训练与开发被试，不读取该外折评价信号或
历史 protected 预测。所有方法使用共同 16 EEG 通道及双探头 96 Hb 通道；8 秒输入
是共同 12 秒数组的原样前缀，不新增滤波、参考或方法特异样本筛选。

冻结诊断包括固定正则的信号特征与类别平衡、动机/后续记忆两个标签维度、8/12 秒
支持、仅原训练被试的按时间留后且间隔一个事件的对照，以及 CBraMod 平均/有序
token 读出和小样本训练可拟合性正控。四分类仍是原任务；二分类只定位信息缺失。
神经模型使用单种子、固定预算和开发集选 epoch，不构成新的确认性评价。按时间
对照继续保留离线全记录预处理，不宣称原始信号隔离的实时泛化。结果保留在比较
方法既有 analysis run 下，本次不导出至 `docs/reports/interpretability_analysis`。

调查中确认原始通道 XLS 与 CED 行号不等价后，另冻结
`corrected_panel_v1/resolved_config.yaml` 的同事件轻量对照：保持样本、时间、标签、
划分与 Hb 不变，仅依据原生通道表重新选择固定的 16 个真实 EEG 电极，并重算
信号特征及冻结 CBraMod 线性探测。旧缓存、原诊断和源快照保留，修正视图不添加
固定面板外的电极，也不冒充生产缓存修复或完整微调重测。输入合同发现及其使用
边界由 [`DATA_CONTRACT.md`](DATA_CONTRACT.md#visual-eeg-channel-identity-finding--2026-10-29)
持有；全部数值与解释保留在本次调查运行中。

### 问题、比较层级和交付范围

分别回答：冻结表示中有多少可供简单读出使用的任务信息；方法经过合理的下游监督
适配后表现如何；主方法的收益能否在相同输入和监督预算下保留。冻结线性探测（LP）、
全参数微调（FT）和从头监督训练分别呈现，EEG-only 与 EEG–fNIRS 分表；预训练
来源、目标语料重叠和训练期额外信息在每行明确标注。不同轨道不形成总体排名。

方案分两批交付：先取得 LP、简单信号基线和原生监督模型，再取得预训练模型的 FT
结果。若只完成第一批，结论限于冻结表示和既定监督模型；要声称充分下游适配后的
最终比较，第二批也是必要工作。两批都使用同一冻结库存、外折和评分规则。主分析
使用全部允许的训练标签；few-shot、跨 session、长血流上下文和另一套 tokenizer
输入不在主矩阵中同时展开。

本版是已有开发历史的公开数据 benchmark。重新划分、换随机种子或换名称不会
产生新的未查看确认集。旧分数只提供历史背景，不作为新版的性能下界；新版分数
也不证明生理耦合、原论文数值复现或方法理论上限。

### 统一测量与任务合同

复用 [数据合同](DATA_CONTRACT.md#cache-contract) 中的当前统一 loader/registry，
固定 `physiology_semantic_clean_v5`、`physiology_measurement_alignment_v5`、
`motion_method=none`、measurement 坐标与 event alignment v2；实施时以具体 manifest
核定身份，不跟随运行期间的默认值漂移。原数据说明由
[`DATASETS_DESCRIPTION.md`](DATASETS_DESCRIPTION.md) 指向。不得把旧 v1 特征缓存或
checkpoint 的输入身份直接改名为新版，也不得恢复通用逐通道 MAD。模型特定的群体
尺度、PCA、特征选择和目标变换只在相应训练分区拟合。

| 数据集／任务 | 编码窗口与目标 | 主评价支持 |
| --- | --- | --- |
| Single-Trial：MI、MA | 事件后 `[0,8)` 秒；LMI/RMI、MA/BL | 原生 trial，保留 session 和事件身份 |
| Simultaneous：WG、n-back | 同步 `[0,8)` 秒；WG/BL、0/2/3-back | 原生事件及其 record/session 依赖 |
| Simultaneous：DSR | 同步 `[0,2)` 秒；EEG-native Go/No-go | fNIRS 仅为同步 context，不将 block marker 当事件级响应或类别 |
| Visual：认知动机 | 同步 `[0,8)` 秒；RR/RF/FF/FR | 保留事件角色、探头和记录身份；不把区域副本当独立样本 |
| REFED：valence/arousal | 20 秒窗，原生 1 Hz `[2,T]` 标签与逐坐标 mask | 按原生视频构造不重叠评分段，保留末端部分输入及目标支持 |

**编码窗口不是全部信息支持。** 当前缓存含连续记录双侧预处理；每格同时记录
编码窗、参考前缀、重采样／滤波上下文及实际处理记录范围。主表明确为离线任务，
不宣称严格只看 8 秒或实时预测。相同比较组从相同 canonical arrays、真实通道
身份集合、时间锚点和 recorded-support mask 开始；内部 patch、池化和已声明的
无标签固定变换可以保持方法原生，但不能暗中增加时间或传感器信息。

EEG-only 主组固定每任务同一组 16 个真实通道，面板按元数据而非下游分数确定。
双模态主组按任务固定同一真实 EEG/HbO/HbR 通道清单；不能接收该面板的方法在看
评价分数前记录不支持，不能复制通道、删掉困难被试或改成自己的较容易支持集。
完整原生容量的单模态结果和长血流上下文如需研究，另定义 profile。同步 8 秒预算
可能限制 NVC/血流方法，故本版只评价该预算下的性能，不将其称为这些方法的能力上限。

### 方法、来源与适用性

| Owner／方法 | 第一批 | 第二批及命名边界 |
| --- | --- | --- |
| BIOT、CBraMod、REVE | 官方预训练 frozen LP，六个分类任务逐格验证 | FT 单独成表；REFED 需真实时序输出和 mask 能力，不把分类 embedding 平铺成序列 |
| NormWear | `NormWear EEG-fNIRS adapted` frozen LP | 相同适配的 FT；原预训练无 fNIRS，不能写成原论文 fNIRS 复现；REFED 未通过适配前不填数 |
| EFRM | 目标数据集整体排除的预训练与 frozen LP，含 REFED 序列任务 | FT 保持同一 target-excluded 条件；旧 v1 输入上训练的权重如复用，只能另标旧域迁移 |
| BrainFusion | 独立监督重实现，训练内完成 CSP/NVC/stacking 选择 | 最终在全部外层训练样本重新拟合；DSR/REFED 在新适配被验证前保持 unsupported |
| STA-Net-PyTorch | 在本版相同窗口、通道和分母上重新训练的监督适配 | 旧 3 秒 EEG／13 秒 fNIRS 结果继续仅作 native-context 参考；新版本须验证形状和 source deviations |
| 简单基线 | majority／train-prior、训练均值；EEG 频带功率＋线性模型、Hb 统计特征＋线性模型、双模态特征拼接 | REFED 使用同时间支持的训练拟合 ridge；与复杂模型共用分区、标签和评分支持 |
| 本项目候选 | 仅在相同 profile 的新适配和外折训练完成后进入对应 LP 表 | 可训练模型进入相应监督／FT 表；条件目标、原生输入、训练曝光分别声明 |

REVE 的 Single-Trial MI/MA 仍为已知 target-corpus-overlap 轨道；其他官方预训练
语料的未排除重叠也要如实标记。EFRM 的 LODO、外部官方预训练和项目内预训练不
称为相同数据曝光。按 source manifest 固定权重、输出层及偏离项；预训练预算独立
计量，不能隐藏在下游训练成本里。跨数据集可核定为同一自然人的记录仍按人隔离；
无法核定的跨语料身份重叠作为限制保留。

现有 [`TK-FUNCTIONAL-v1`](#条件功能监督与-hard-token-开发矩阵2026-10-10) 使用原生 EEG、
20 秒学生窗口、额外 Hb 参考前缀和 90 秒条件教师，其单次划分与 balanced-accuracy
主指标也不同，不能直接接入本版。纳入候选时须按本版重新适配、分折训练和评分；
教师、目标尺度、码本及学生都不能使用外层评价被试。训练期更长的教师上下文必须
披露，并保留相同架构、训练曝光的无教师／替代监督对照，不能将额外信息收益全部
归因于 token 设计。本方案不自动展开第二套 TK20 完整基线矩阵。

### 被试划分、选择和最终重新拟合

默认每数据集 **5 个外折、外层训练集合内 3 个内折**；划分种子 `20261016`，最终
随机训练种子 `[17,42,73]`。同数据集各任务共享被试外折，全部 session、record、
trial、video 和区域副本随被试分组。先用许可元数据核定共同库存、类别和通道支持，
再读数组。Visual 等较小集合若无法满足内折类别覆盖，在任何评价前统一修订该
数据集的内折设计，不给某方法单独改折或丢样本。

内折选择超参数、训练轮数和阈值；所有有学习状态的预处理及表示训练同样服从
内折边界。最终用选定设置在**完整外层训练集合**重新拟合，再预测该外折评价被试。
BrainFusion 的全部组件和 stacking 内部 OOF 也适用此规则；最终拟合样本身份须
与同组其他方法一致。不得继续保留“一个方法仅 train，其余 train+validation”
的差别。确定性方法无需人为重复三个相同结果；存在随机训练的种子全部保留，不选
最好 seed。表示预训练种子和读出种子分开记录，共享同一 backbone 的多个 head
种子不能冒充多次独立预训练。

### 搜索预算与收敛判据

以下是实施时核验的设计默认值，不能视为所有模型已充分训练的保证：

- LP 使用固定表示、训练拟合标准化和真正的线性读出。分类
  `C ∈ {0.001,0.01,0.1,1,10,100}`；序列 ridge 的
  `alpha ∈ {0.001,0.01,0.1,1,10,100}`。可训练非线性 pooling/LayerNorm
  若保留，必须另标 adapter/head 轨道，不能仍叫统一线性探测。
- 监督神经模型和 FT 每个 method×task×track 最多 12 个预先列出的候选配置，
  三内折、调参种子 17。候选包括来自固定上游实现的优化器、backbone/head 学习率、
  weight decay 和 clipping；不对所有方法强制同一学习率。BrainFusion 等非神经
  方法使用同上限的有限训练内候选，完整记录其内部拟合成本。
- 神经下游默认最多 200 epochs，至少 50 epochs 后允许 patience 30 停止；选择
  指标与本版任务主指标一致。配置按合并三个内折验证被试后的被试等权分数选择，平局依次取训练成本
  较低者及冻结顺序靠前者。最终 refit 使用选定配置三个内折最佳 epoch 的中位数
  （向上取整），外折评价不参与早停。
- pilot 在开发支持上核对吞吐、内存和训练／内折曲线。若持续触及预算上限且仍在
  改善，先修订并冻结该类模型的预算，再开展外折评价；不能看过外折低分后有选择地
  追加搜索。未证明收敛的结果保留预算限制，不称“合理高水平”。

### 评分、基线、不确定性与失败

分类主指标为**被试等权 macro-F1**：每位被试汇总全部合规事件预测，按任务完整
类别集合计算；先平均该被试的训练种子，再平均被试。区域副本如存在，先按冻结
规则合为一次事件预测。伴随报告 balanced accuracy、accuracy、各类召回和校准。
旧 fold-mean macro-F1 只作单独标注的兼容描述，不替换新主指标，也不能把旧数值
与新主指标相减。缺失类别按固定完整类别集合和 `zero_division=0` 计算，不能按
该方法的预测类别缩小分母。

REFED 先在原生 record 内拼接不重复的有效评分段，对 valence/arousal 分别计算
masked CCC、MAE/RMSE，再 record→subject 等权汇总，两个坐标的算术平均为主 CCC。
不足有效点或 CCC 分母为零时保留 undefined 及覆盖缺口；原生标签坐标上的误差
与训练缩放坐标区分。Pearson/Spearman 遇常数目标或预测标记 undefined，禁止
继续使用旧辅助指标中的强制 0。记录输入 mask 和 target mask 各自作用，不能用
输入完整性代替目标有效性。

无信息基线使用同支持的训练 majority、train-prior random 期望和训练目标均值；
分类 macro-F1 的 B0 取前两者较强者，沿用同一被试聚合顺序，不能一律写成 `1/K`。
CCC 的无正向关联参照为 0，同时保留训练均值预测的实际 MAE/RMSE。简单信号基线
提供独立于无信息基线的能力参照，不能只比较复杂方法彼此的低分。

默认 2,000 次被试块 bootstrap（种子 `20261016`），两侧 95% percentile 区间；一次抽样保留
被试的全部记录、seed 和各方法结果，方法差使用相同抽样。区间条件于已训练模型，
不覆盖重新训练的全部变异；窗口、种子、五个外折均不是独立生物重复。预先指定
主方法相对同轨简单信号基线的逐任务比较：先计算每位被试种子平均后的配对差
`d_s`，统计量为其被试均值；以种子 `20261016` 进行 10,000 次被试级独立正负号
翻转，双侧 `p=(1+次数(|T_b|≥|T_obs|))/10001`。该检验要求零假设下被试差的符号
可交换，保持条件于已训练模型的开发性解释。按模态／训练轨道分别定义检验族，
族内跨任务对上述原始 p 值使用 Holm 校正；主方法缺席时只报告描述性比较。
其他事后两两比较标为探索性，
不以区间是否重叠代替配对差检验，不将分类和回归合成平均排行榜。

每格同时报告计划与完成的 fold×seed、被试、事件／有效时间点、排除及失败。
unsupported 无数值，运行失败不记成零分，失败后不缩小分母发布正常排名。保留全部
有效低分和负结果；完整观测表与“正常方法性能”的准入判断分开。正式数字仍按
[`METRIC_ACCEPTANCE.md`](comparisons/METRIC_ACCEPTANCE.md) 逐格给出 B0、target、
相容参考 H／source 的适用性、关系判断、uncertainty、decision 和具体脚注。
旧 `B0+0.02` fallback 只作最低线，不能代替充分训练或接近能力上界的证据；被拒
任务不得在保留同方法高分的同时静默删除。

### 实施顺序、资源与产物归属

| 阶段 | 工作与可审阅产物 | 结束条件 |
| --- | --- | --- |
| 软件和合同准备 | 统一元数据库存／被试折、每格支持声明、版本化适配与指标 fixture | train-only fitting、mask/时钟、固定类别、常数相关、同支持 refit、失败分母测试及 synthetic/dry-run 通过 |
| 公开开发 pilot | MI、n-back、REFED 代表二类／多类／序列路径；资源与收敛试测 | 只用开发支持定候选、预算和适配；无最终外折分数用于选择 |
| 第一批正式重获取 | 所有支持任务的 LP、简单基线和原生监督模型 | 全部计划单元终态、保留预测／指标和完整失败；只作对应轨道结论 |
| 第二批适配性能 | 官方预训练及 EFRM 的 FT，同一库存和外折 | 不根据第一批外折低分改选择规则；与 LP 独立成表，完成后才形成适配性能比较 |
| 汇总和验收 | 被试配对统计、每格准入、来源／预算／失败脚注及版本化图表 | 表值可追溯到单次运行，所有不支持、失败和拒绝格可见 |

两批候选空间、报告规则及拟比较的方法必须在第一批外折结果揭示前冻结。pilot
使用独立列明的开发身份；若这些身份后来进入外折评价，整体结论继续标为开发性，
不能宣称未受 pilot 决策影响。软件缺陷若要求更改已冻结输入或评分，保留旧版本
及失败记录，在新版本中统一重测受影响单元，不能只修复有利的一格。

正式运行前检查 CPU 配额／亲和性、内存、GPU、存储及竞争进程；以 pilot 吞吐和
峰值内存确定并发、线程和预估时长。用已约定的持久 supervisor 运行，记录命令、
环境、源版本、日志、进度和可恢复输出；不因历史队列串行而闲置可并行资源。
每个随机训练支持单元默认有 5 外折×3 最终种子；内折搜索、预训练及确定性基线
成本另计。任务／方法支持表和试测成本核定前不填写虚假的固定总 job 数或 GPU 小时。

沿用现有 owner 和 artifact root，实施时在其中增加版本化 run，不建立平行
`results/`、wrapper 或逐方法授权层：

```text
comparative_methods/<method>/runs/downstream_rebench_v1/<run>/
comparative_methods/runs/downstream_rebench_v1/<run>/          # 共享合同、路径索引与联合汇总
experiments/runs/physiology_semantic_tokenizer/downstream_rebench/<run>/  # 主方法
```

各方法保存其拟合／预测／指标事实；联合汇总只消费这些带身份的终态产物，不维护
第二份手写状态。公共统计放在既有 library owner，方法差异留在各自 package；历史
sealed 脚本与配置不原地修改。实际结果以后由
[`RESULTS_INDEX.md`](../experiments/RESULTS_INDEX.md#comparison-methods) 链接，注册执行
状态仍只在 registry 更新。

最初整理和设计的授权已由上述本版执行请求接续，覆盖声明的 pilot、
两批训练和汇总，同范围内不逐命令重复确认；历史 protected 数据不因本版执行解锁，
如涉及受保护读取仍须满足唯一 owning protocol/controller 及其独立 gate。当前
公开被试政策不等于允许重读旧 protected prediction 或把它用于调参。

本版执行冻结 timing v2 库存和被试外／内折、各方法支持面板、预训练来源与重叠、
候选参数及 smoke 资源预算。EFRM 使用非目标数据集的无标签 8 秒源窗，整个目标
数据集排除；Stage A 在源被试间选择，Stage B 从新初值在全部允许源窗重新拟合。
普通神经拟合仅保留单个滚动恢复 checkpoint、轻量阶段记录及最终模型，避免每个
内折候选重复长期保存优化器。具体源窗、未使用支持、资源测量、失败及源码快照
保存在[本版运行目录](../comparative_methods/runs/downstream_rebench_v1/20261016_v2/)；
首版派发及缺失通道 smoke 的失败记录保留在相邻 `20261016_v1/` 和新版 smoke 目录。
主方法尚无本版实现，不包含在已实现的比较矩阵中。当前执行状态仍以 registry 为准。
2026-10-27 用户要求移出 NormWear；该执行范围缩减由 registry 所链接的版本化
任务清单记录，原始配置与历史证据保留，不因旧配置仍列有该方法而重新派发。

## 局部多通道观测与空间动态维数（2026-10-16）

`SSM-LOCAL-SPATIAL-v1` 的数值合同由
[`shared_driver_spatial_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_spatial_v1.yaml)
持有，入口为 [`evaluate_shared_driver_spatial.py`](../experiments/scripts/evaluate_shared_driver_spatial.py)。
2026-10-16 的首次交付只含设计、实现及软件／小规模合成验证；后续运行记录与
科学判读以 registry 及其生成视图为准，不从本节的设计描述推断执行状态。
它是独立版本的公开开发诊断，不继承历史 T5 的资格、受保护授权或已撤回的几何结论。

**问题与最小结构。** 区分增加通道、使用正确空间关系和增加动态维数三种收益。
在一个前额粗邻域内保留四个真实 HbO/HbR 对和十二个唯一 EEG 通道；两个 Hb 对、
六个 EEG 是同一冻结面板的嵌套子集。这里的 Hb 数包含目标对，整对隐藏时分别有
一个或三个可见供体。先用一个共同驱动和一套守恒 Balloon 状态（K1），再比较
两条驱动、两套独立传播且共享参数的状态（K2）。两套 Hb 输出在传播后按固定空间
权重混合，不把两个驱动先平均后送入同一个 core；相同驱动和初态的 K2 应复现 K1。
K1 的多个标量观测读出增加约束，不能独立生成不同峰形。没有额外 Hb 时间基底、
每通道自由初态、自由网络边或逐窗生理参数搜索。

**数据与几何。** v1 实测读取面仅支持已登记的公开 Single-Trial，复用统一原生
reader 和 geometry owner。先完成 metadata inventory、被试级 train/selection/evaluation
划分及通道冻结，再访问信号；评价人群仍是已被研究的公开开发人群。目标按几何选取，
供体与目标不共享已核验光源／探测器，且位于预定 45 度模板角邻域内。供体之间
可以共享光极，其硬件邻接必须保留；不能将这些供体当作独立组织观测。
该角度只是工程选取条件，不是个体皮层或生理距离。支持不足保留 excluded/unavailable，
不退到全头、补造坐标或放宽硬件身份。各原生时钟上的窗口都不重叠。其他三个数据集
需要各自版本化的几何／硬件读取合同，本版本不得借旧 prepared 文件绕过该边界。

**观测与隐藏。** 30 秒窗口形成 120 个 0.25 秒位置，EEG 保留通道×五频带的
基线相对 log-power；Hb 使用原生样本的互不交叠 bin 均值，不在隐藏区间之间滤波
或插值实测 Hb。两模态保留前 5 秒参考算子。`heldout_pair` 隐藏目标 HbO/HbR
全部时间，包括参考段；目标自身光学／Hb 参考仅定义评分坐标，不得进入推断输入、
加载系数或当前窗统计。模型对自身轨迹应用相同固定参考算子。`center_pair` 仅隐藏
预定中心区间，参考段可见；`full` 是可见条件重建。目标原生强度及参考段的替换测试
必须保持全部可见输入和预测不变。完整 EEG 可见，所有任务均为离线补全而非因果预测。

**训练坐标和有限校准。** 每个相容 montage 用 train 被试拟合最大面板的一套坐标，
各模型和子面板仅切片，不能重新归一化或重选核宽。EEG 每频带跨通道共用训练 RMS
尺度，保留空间载荷的幅度差异；不能逐通道标准化后仍用未变换的固定空间核。
Hb 使用共同正尺度以保留两色团
及通道间幅比；各评分 SD 来自同一训练坐标。模型固定 beta=1、Hb 有效增益=1，
输出与归一化目标处于声明的条件坐标，不把原始 MBLL 单位当作绝对物理标定。
训练 Hb-only 条件拟合提供驱动参考，然后拟合
`L[channel,band,k] = spatial_kernel[channel,k] * band_coefficient[band,k]`。
空间核固定，仅五个带符号频带系数／模式在训练中估计；回归的 ridge 乘以训练
Gram 的平均对角线，保持驱动单位变换下的预测不变。驱动先验和加载列不再重新定标。
这些校准驱动不是神经真值；缺乏激励或秩退化时保留校准失败。

**固定矩阵。** 主面板比较 K1 的真实、均匀及置换几何，以及 K2 的真实和置换几何。
K2 均匀混合的秩塌缩仅作为合成反例，不冒充等容量主对照。置换保持信号数组、
目标身份、通道数和权重分布，只改变几何与通道的对应；每个候选重新按相同训练
流程校准。数量消融限 K1 真实几何的 2 Hb／12 EEG 与 4 Hb／6 EEG。EEG 权重
0、0.1、1 只由独立 selection 被试上的隐藏 Hb 误差选择，平局优先较小权重；
主面板的整对／中心隐藏任务另保留固定权重 1 的结果。K2 还复用完全相同的冻结
读出，比较两条驱动及初态强制相同的限制模型：等价单核的 EEG loading 为两列之和，
Hb 权重为 1，曲率／初态／流量先验权重均乘 2。该固定权重 1 对照保持原 K2 目标
的严格限制形式，隔离解除轨迹和初态相等约束的增量；不同读出家族的普通 K2/K1
比较不能独自解释为纯动态维数收益。列和抵消保留退化，不重新定标或校准。
当前窗仅拟合驱动时间系数及每个 core 的五个初态。
数据项各模态使用 `T / visible_scalar_count` 乘标准化 SSE，再乘 EEG 权重，避免
通道数量或隐藏比例改变模态与先验的相对总权重；该量是工程目标，不是校准似然。

**对照、评分与失败。** 相同可见供体下比较正确 EEG、Hb-only 和任务、条件、
montage 相容的非重叠错配 EEG；找不到供体保留 unavailable，不以不匹配对象代替。
训练模板、原始供体空间 ridge 和 EEG＋供体 ridge 独立于 SSM 成功状态。
整对隐藏时 ridge 获得相同输入；中心隐藏另保留可见自身上下文插值对照。
主指标为目标对在固定评分支持上的训练 SD NRMSE，另列 HbO、HbR、相关、幅度误差、
积分精度及完整失败分母。窗口内先计算归一化均方误差，再在被试内汇总；配对差及
bootstrap 使用被试作为独立单位，成功子集和含数值失败惩罚的结果分别保留。
缺失错配供体或输入的单元进入完整计划分母，但方法收益只在双方共同可用的身份上
计算；无可用配对时为不可估计，不以缺失对照的惩罚制造正面收益。
真实几何的收益、EEG 在空间 Hb 之外的增量和 K2 的动态收益分别判断，不合成一个
自动生理资格标签。新增动态只改善可见拟合时，不升级来源或参数解释。

**软件、合成及执行。** 合成保留单源、可分双源、相同双源、无信息 EEG、几何失配
及同向共同 Hb 干扰；训练、选择、评价的随机种子独立。每个生成身份保留两条独立
记录以构造同条件错配 EEG；各机制条件独立训练校准、选择并分别汇总，不把负对照
混入同一校准分布。合成数据直接在声明的模型坐标生成，
尺度变换取恒等、评分 SD 仅由训练生成数据确定，避免再次调整 Hb 幅度却沿用原
物理真值。原生数据的训练尺度流程单独通过 fixture 验证。检查核心嵌套、解析导数、
隐藏污染、零 EEG 权重、读前边界、积分加密和失败预算。`synthetic-smoke` 明确使用
已知读出，仅验证模型与执行链；完整 `synthetic` 另走训练校准／选择，二者不能
互相替代。`check` 与 `dryrun` 不读取实测数据，`plan` 只访问元数据；prepare、
calibrate、measured 通过显式阶段执行，不由默认命令接续。较大阶段须遵守已有
持久监督及资源试测规范，保留固定配置、源代码身份、命令、进度与失败，重连原运行
而非覆盖旧结果。当前执行和科学判读只由 registry 及其生成视图持有。

## REFED 公开实验补齐与四数据集比较（2026-10-13）

本轮按用户审核后的补齐要求执行十组既有诊断的 REFED 扩展；**不执行共享／不共享
光极实验**，不扩展 TK-FUNCTIONAL 分类任务、EOG 锚点或独立短间距／外周验证。
各方法的 `*_refed_v2.yaml` 是新运行合同，保留原模型、候选网格、评分方向、失败
处理与比较臂。历史配置、划分、运行和负结果保持原身份；新生成物写入各方法原有
artifact root 的新版本子目录。研究状态仍只由项目 registry 持有。

基础入口复用 `evaluate_shared_driver_reconstruction.py --physiology-semantics`，
[基础合同](../experiments/configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_refed_v2.yaml)
显式消费 `refed_paper_layout_2d_v1` sidecar。区域分别取距离 FP1、C3、O1 最近的
真实 Hb 对及六个相邻 EEG 电极；位置只是粗粒度前额、运动与后部模板标签，不是
个体三维定位或等同的皮层覆盖。记录第六／第七邻域间距，并在固定 Hb、固定方法及
训练划分下交换边界邻居，单独重拟合训练 PCA，检查几何选择对 Hb 结果的影响。
此工程对照不是生物位置置信区间，亦不据拟合好坏选择布局。

公共 REFED 的全部已登记视频按原生支持构造互不重叠的 30 秒窗口。完整视频是
记录分组；同情绪视频依原生编号交替进入两个重复组，视频内部的全部窗口留在同组。
复用旧 metadata 已登记的被试折与六名 QC 分层诊断被试。诊断面板按 metadata 取
同情绪、不同重复组的两个合格视频，各取末两个完整窗口，共 24 个目标窗口；
三种空间区域是同一记录的相关观测，不充当独立被试。全部选择均在新拟合前固定。
总体重建／参数共享用所有合格区域窗口，H3 仍只用首三个窗口拟合并在后续窗口评价。

参数稳定性与输出支持的连续输入取两个同情绪视频各自最初 90 秒，原生记录不足
90 秒或不满足实际支持则保留为不可用，绝不拼接不同视频。其重复性含刺激内容
变化，只称跨视频条件稳定性，不称同刺激、跨 session 或跨日重测。错配 EEG 与
目标同被试、同情绪、同区域，但来自另一个完整视频；报告说明内容和瞬时情绪轨迹
未精确匹配。标签只用于划分及匹配，不进入 SSM 驱动或 teacher 拟合。训练坐标、
响应选择、背景校准和监督探针仍只在各合同指定的训练／选择被试上估计。

依赖顺序为：基础准备与共享驱动；四臂稳健性；成分归属与区域谱模式；响应／空间
强化；观测语义 A；参数稳定性；连续输出支持及 REFED 背景功效校准。响应动力学
及冻结连续编码器公开探针、EEG 4/10 Hz 比较使用相同版本的基础输入独立接续。
不因补齐数据集自动触发旧协议未通过的维度扩展。与数据集无关且数值合同不变的
已完成合成证据通过显式来源引用复用，不重复计为新生成身份；新增软件／几何／
视频划分先通过 fixture 与合成检查，再监督先导和实测。

四数据集展示读取三个旧数据集的保留结果及 REFED 新结果，按各端点原有层级重新
汇总，保留失败分母与共同成功配对数。改变来源集合的跨数据集时间参数集成重新
拟合目标驱动并评价；新增 REFED 不改变原三个数据集独立估计的训练对象。合并
统计重新计算数据集等权效应、被试块区间及对应多重比较，不平均已汇总的区间。
绝对单位与硬件同步未知项保留，不能把各自训练 SD 归一化后的误差解释为统一物理
幅值。新 PPT/PDF 使用可核对的位图图表与 LaTeX 公式，不覆盖既有日期报告。

## 冻结 token 的语义读出与时空关系探针（2026-10-25）

`TK-RELATION-PROBE-v1` 复用已完成 `TK-FUNCTIONAL-v1` 的公开缓存和冻结
QT/Q0/CT 出口，数值合同由
[`functional_relation_probe_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/functional_relation_probe_v1.yaml)
持有。入口沿用 `run_functional_tokenizer.py --stage relation-probe`，结果写入
同一套件的独立版本目录，不改写父运行，不重读原始数据或拟合 teacher。
这是公开开发复分析；父运行的失败 trial 继续保留在分母中，读出结果只对应
共同成功子集，不能据此恢复父运行未成立的全分母结论。

第一组用严格整数 ID 与冻结码本（QT/Q0），以及独立的连续 CT 参照，读出
2 秒 patch 的 EEG 特征、`X=0.35 HbO−0.65 HbR` 和 `T=HbO+HbR` 均值。
实测与模型条件预测分别评分。另在相同的内部八个 patch 上比较当前 token
与前／当前／后 token 对下一 patch 减当前 patch 的变化读出。每个 token
已具有 20 秒双向上下文，邻域又包含目标时刻，故此量是离线变化可读性，
不是未来预测、局部两秒推断或生理状态恢复。各目标的归一化尺度仅由训练
被试确定；误差先 patch、区域到 trial，再记录到被试，种子在被试内汇总。

第二组比较三个区域各自的 token 均值、SD 及完整硬 ID 占用量，与加入
时间乘积／差分／端点差、区域同步及双向 lag-1 乘积的表示。固定 32 维
训练 PCA 和验证集选择的线性 probe，评分左右手 MI 与 n-back 负荷。
各方案独立拟合 PCA，因此这是相同维数预算的读出比较，不是严格嵌套的
条件信息检验。关系扰动在同数据集／被试／记录／条件／split 内，用不同
原生 trial 逐相位重组时间关系或逐区域换整段关系，保留真实边际块并重拟合
probe。所有方案使用相同 donor 可用子集；这种组合只是描述性扰动对照，
不是随机化显著性检验、可部署预测器或生理干预。

训练、选择和评分继承父运行的被试划分。标签只用于 probe 和条件匹配对照，
不进入 tokenizer 拟合。既有隐藏 Hb 的上下文／配对对照与合成机制反例单独
解释，不能以同模态读出或分类增益替代跨模态收益、来源识别与真实生理校准。
任务按固定九个 arm×seed 单元结束，不根据评价结果调参。先经临时合成检查，
再在 user systemd 下测量单任务吞吐，依 CPU、内存与竞争任务选择并发。

### 局部形状 ID 与连续尺度的架构探针

`TK-SHAPE-PROBE-v1` 在用户明确允许调整架构后，以
[`functional_shape_probe_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/functional_shape_probe_v1.yaml)
定义一个独立小候选。每模态、每区域的编码器只读取当前 2 秒观测 patch，
区域间共享参数。训练 ridge 将观测映射到同 patch 的 SSM 条件波形；等容量
对照改用父运行训练集确定的平滑目标。teacher 只作训练／选择目标，推断不读
teacher。两支分别是条件估计与观测减条件估计，不命名为生理源与噪声。

每支保存各通道均值和跨通道共同的正 RMS 幅度，对去均值、单位幅度形状
用训练集拟合的 16 项码本离散化。共同幅度保持 HbO/HbR 相对比例；近乎平坦
的 patch 显式处理。分别评价仅 ID、仅连续尺度、二者合并与未量化连续表示，
记录完整连续尺度的储存成本，不将混合表示称为与旧硬 ID 等码率。

主终点不读回表示自带的均值，而是预测下一 2 秒实测 X/T 均值减当前均值。
对当前位置 1–8（从零计数），比较仅本区当前、本区前／当前、三区前／当前
信息，各区域独立拟合读出；所有候选保持相同目标支持。直接观测 ridge 和
保持上一 patch 不变同时作为基线。Hb 采用父运行不跨 bin 的原生平均及此前
可见参考，输入止于当前 patch 末端；未核验上游滤波支持的 EEG 特征不进入
此预测终点。结果仍是处理后 Hb 观测预测，不是生理来源或因果验证。

两个目标臂各三个种子、固定码本及预算，一次完成。沿用父运行的被试划分和
失败分母；统计先 patch／区域到 trial、记录到被试，训练种子在被试内汇总。
SSM 与平滑的配对差异决定是否保留条件目标，不因候选名称预设其生理优势。

随后以独立
[`TK-SHAPE-RELATIONS-v1`](../experiments/configs/physiology_semantic_tokenizer/functional_shape_relations_v1.yaml)
合同做本轮最后两项冻结关系对照，不改写上述结果或重训编码器：本区历史中
仅将前一 patch 换成同相位供体；三区历史中保持目标区前／当前 patch 真实，
其余两区共同取自同一供体 trial 的完整序列。供体限于同数据集／被试／记录／
条件／split 的不同原生 trial，保留区域身份、事件相位及供体内部关系。
SSM hybrid、smooth hybrid 和直接观测使用同一可用子集，真实／扰动方案
分别重新拟合读出。主量为供体扰动误差减真实配对误差；这是条件匹配的
描述性配对检查，不是随机化因果检验。两项完成即结束本轮探针。

## 条件功能监督与 hard token 开发矩阵（2026-10-10）

`TK-FUNCTIONAL-v1` 按用户本次指引建立独立公开开发线。数值合同由
[`functional_tokenizer_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/functional_tokenizer_v1.yaml)
持有，统一入口为 `experiments/scripts/run_functional_tokenizer.py`。当前条件目标
可以用于工程蒸馏，无须先通过旧生理资格流程；结果也不反向授予 teacher 资格。
旧 E/R、连续语义原型和已完成 SSM 证据保持原身份及负结果。

从统一索引枚举 Single-Trial、Simultaneous 全部可用公开被试，按数据集内固定
种子划分 train/validation/test，再读数组、拟合变换及生成缓存。主任务为左右手
MI 和 n-back 负荷，MA/BL、WG/BL 为次要任务；Visual、DSR、完整外折、集合监督
和 O 量化为条件扩展，本批不自动启动。新划分不把已有开发历史的公共数据变成
全新确认集。状态只在 `research_state/registry.json` 维护。

每个原生事件的 teacher 读取连续 `[-30,+60) s`，学生编码 `[0,+20) s`；Hb
预处理另使用可见 `[-30,-25) s` 参考前缀，不能称仅有 20 s 的完整输入支持。预设
前额、左/右运动区域各取一个真实 Hb 通道对和六个邻近 EEG 通道，保留模板几何
与源身份。学生重建选定原生 200 Hz EEG 电位和原生 Hb 的 4 Hz 不重叠 bin；
这是明确的原生 EEG 支路，不使用项目伪迹清理，也不是全头皮或原始 10 Hz Hb
重建。Hb 与 Single-Trial OD 参考只用最初可见 5 s，不跨缺失区间滤波。完整
上下文不足则记录排除；区域副本、重叠 teacher 上下文和全部记录按被试分组。

teacher 复用六状态 RK4、固定响应参数、`chain_matched` 连续传播/边界先验及
D-fixed 额外项。PCA、Hb 共同正尺度、目标尺度及平滑带宽匹配只用训练被试。
主目标是 SSM EEG 特征预测和 `X=0.35 HbO−0.65 HbR`，wide 增加 T；完整剩余项
是同坐标的 `u−SSM prediction`，不是共同项系数，更不能从原始 EEG 电位相减。
失败及分母保留，全部臂只在同一完整、收敛 trial 子集上作匹配训练。

八臂共用容量与曝光预算，S/O 编码路径独立，C-stop 只切断重建到 S 的梯度。
无任务、耦合、独立性或去标签训练损失；shuffle 同数据集同区域跨训练被试，
不读取类别。QT 的 teacher 监督和重建均在 hard-forward 路径。每次保留检查点
同步导出 pre/soft/O 和单独的整数 ID/冻结码本/身份文件；主 hard-S 评价通过
`--stage hard-probe`，不打开连续导出。S+O 与纯 ID 的性能、实际码率分别报告。
连续臂的后处理 k-means 只在训练集拟合。

TK0 检查观测、SSM、完整剩余项、直接统计和平滑目标的任务可读性及模态独立的
目标可预测性；后者不是生理正确性。TK1–3 使用固定容量线性 probe 与同一原生
trial/block 标签子集，验证标签计入预算；区域先合并，记录内计算后被试等权，
种子在被试内汇总。跨 session 是次要协议：外层未见被试的前、中、末原生
session 分别用于 probe 训练、选择、评分。差值方向、工程阈值由 YAML 冻结，
探索性区间不宣称多重比较后的统计显著性，成功子集不能掩盖失败。

TK4 在同一生成身份、驱动和噪声下比较真实变化、D/off-D 扰动、竞争归属及
相同观测/不同真值的 alias；合成 EEG 为固定载波压力测试，不代表真实来源。
小型训练面板的 T 替代解须在原目标容差内整段可行；CT/C-wide 用同初始化与
数据顺序配对重训，没有见证则保留不可用，不逐时刻拼目标。TK5 汇总条件波形
原型并用训练身份对齐 ID；隐藏 Hb 在任何操作前遮蔽，真实 EEG、简单 EEG、同
被试错记录/原生 session、同条件错配和非回绕移位均用共同支持。未来 EEG
可见的结果只能称离线补全，共现不进入训练或承担未经支持的耦合论据。

先通过合成/软件检查及资源先导，再以源代码快照、user systemd 和 linger 启动。
CPU 与 GPU 并发、线程及在途任务受限，独立 TK0/机制准备与 GPU 训练并行；
原子任务结果及优化器/RNG 检查点可恢复，命令、资源、失败和进度保留于 run。

## SSM 输出支持范围、连续状态与跨模态约束（2026-10-10）

`SSM-OUTPUT-SUPPORT-v1` 执行用户本次附文的三项实验；数值合同由
[`ssm_output_support_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_output_support_v1.yaml)
持有，入口为 `experiments/scripts/evaluate_ssm_output_support.py`。这是一轮公开开发
诊断，保留已知六状态 SSM、固定参考响应参数和已有 D-fixed 的方向/容量，不访问
其他协议的受保护 split，不启动 tokenizer，不将当前评价身份称为历史未查看的确认数据。
原参数稳定性运行及历史 S3 证据保持各自身份。

从指定配对元数据中先冻结三个联合数据集的原开发校准身份及另六名评价身份，每名
选择一个具有校准支持的前额优先 ROI 和两条同被试、同任务、相同条件起点计数、
相同通道、互不重叠的 90 s 原生连续记录。配对记录用于错 EEG 对照；它保留条件
计数但不保证事件顺序或相位完全匹配，不能解释为严格任务模板匹配的条件独立检验。
同一记录的两个块不能计为两名被试。中央 30 s 为评分区间。

本轮原生 Hb 遮蔽要求采用新的、显式版本化观测算子：每个 250 ms bin 只平均该 bin
实际原生样本，减去可见最初 5 s 参考，不使用跨隐藏区间的双向 Hb 带通。相同原生
时间戳上的模型线性插值、bin 平均及参考算子也作用于 SSM 与额外项。Single-Trial
的 OD 参考只用可见最初 5 s；没有独立绝对 Hb 校准。EEG 从完整原生 90 s 提取五频带
功率，经仅用开发被试拟合的 PCA 和尺度形成一维特征。HbO/HbR 使用共同正尺度。
该算子与旧 S3 的块内带通不同，不能跨版本直接比较 NRMSE；三臂内部的可见数据、
算子、额外项、响应参数和驱动正则完全相同。剩余项只在真实可见观测处定义。

主对照为自由重置初态与连续传播：`free` 和 `chain_matched` 在同样三个时点对同样
物理状态施加同一先验，只有状态是否来自传播不同；`chain_first` 单列仅首端先验
的实际连续模型。分窗扰动只移动血管重置/对应先验时点，额外项基底和驱动正则
边界保持原位置；另将可见 Hb 未来上下文从 90 s 缩至 75 s，仍保留完整 EEG 和
潜在轨迹，明确这只改变 Hb 上下文。此任务是离线补全，不是未来预测。

实验一直接分析中央 SSM Hb、T、X、v−1、p−v、f−1 和驱动。复用离散 RK4 解析
切向量，对每个输出寻找自身最不利的局部数据/总目标方向；再固定该输出轨迹的
有符号投影，重新优化全部实际未知量，包括额外项系数。分别在纯数据目标及加入
原正则的工程目标下寻找 5% 容差内的合法解见证。有限剖面不是完整容许集合、
校准置信区间或后验。固定 τ、有效增益和 κ 所造成的零变化明确归为设定限制，
不据此声称数据识别。p−v 在连续模型中衰减也不能自动归因于观测的信息量。

实验二按分窗、Hb 上下文、中央 Hb/HbR 遮蔽分别检查输出变化和隐藏误差。
实验三在相同连续模型下比较真实 EEG、上述错记录 EEG、仅 Hb 上下文；同时用
训练记录的幅度、AR(1) 时间相关和 HbO/HbR 创新协方差构造已知 SSM 动态阳性校准。
背景保留实测挑战尺度但不命名为纯仪器噪声；为使注入量的来源已知，生成背景的
EEG–Hb 创新交叉协方差置零。合成真值仅用于评分及幅度定义，不能送入推断器。

所有任务经单元/合同测试和独立合成先导后，在用户 systemd supervisor 下运行。
结果保留计划分母、数值/任务失败、终止记录与源代码快照；失败评分使用冻结惩罚，
并并列成功子集结果。先按记录配对、再被试内平均、最后被试等权；合成按生成种子
重采样。区间是固定配置下的块 bootstrap 描述区间，不作多重比较后的确认性声明。
三项实验的结论逐输出报告；需要新确认数据的后续冻结复验不由本轮开发结果替代。

**功效背景核验后的版本化修订。** 初版 AR(1) 校准的全部运行结果保留。背景矩核验
发现它低估部分数据集 HbR 波动且未保留 HbO/HbR 相关结构，因此不能用其阳性表现
排除实测中的共享动态。补充合同
[`ssm_output_power_calibration_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_output_power_calibration_v2.yaml)
仅复用同一开发校准记录：保留逐时间点均值，以独立标准高斯权重重组中心化的完整
训练轨迹，HbO/HbR 共用一组权重，EEG 使用独立权重。这在期望上保持训练均值、
Hb 协方差、多滞后乘积及功率谱；背景仍包含未辨明来源，且受有限训练记录张成空间
限制。已知 SSM 注入走原观测算子，拟合策略完全不变。错配 EEG 同时更换驱动与
独立 EEG 背景，保留相同冻结时间均值；新增零注入阴性对照，并列最初六个生成
身份的样本量匹配敏感性，不按结果选择种子。输出保存在原运行下的
`power_calibration_v2/`，不覆盖初版校准，也不重新读取原始测量数据。

## 参数稳定性、共同内点与拟合改进（2026-10-10）

`SSM-PARAMETER-STABILITY-v1` 按本次用户提供的 E1–E6 指引执行公开实测诊断。
数值合同由
[`shared_driver_parameter_stability_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_parameter_stability_v1.yaml)
持有；入口为 `experiments/scripts/evaluate_shared_driver_parameter_stability.py`。
保留六状态、单驱动及原参数搜索范围，依次比较固定参数、单独释放 τ/β、联合
τ/β，再依据开发信息与合成校准决定是否释放 κ。固定参数或完全共享参数是
端点对照，不因重复误差为零而获得个体参数资格。

开发身份只复用原 `diagnostic_plan.json` 的三联合数据集名单；其余身份为本轮
冻结评价，不称为历史未查看的确认队列。原 outer-fold 坐标不复用；新 PCA、
Hb 共同尺度、工作误差协方差和参数共享对象仅由开发身份拟合。元数据先冻结
同任务、条件标记计数比例、ROI/通道、时长一致的配对。不同原生记录为主评价，
不能确认 native session 的拼接段以及同记录不重叠块均为辅助评价；Visual
不同 Probe 不能配为重测。条件占比按原生标记计数定义，不将缺失的任务持续时间
补写为实测。统计按数据集和坐标分别计算，重采样单位是被试。

两记录各取真实连续 90 s；三段 30 s 的原生局部处理算子组成块对角算子。
C0a 每段有自由初态及先验；C0b 仍可重置，但仅最初初态有先验；C1 只保留
真实起点的五个 ODE 初态；驱动仍按原网格自由更新。C1 不跨原分段边界增加驱动二阶平滑。各记录独立估计，再固定
源记录参数、重拟合目的记录驱动和初态，称为“参数转移后的条件重建”。
共享参数的联合拟合只检查相容性，不进入重测 ICC。

E1/E2 保留原目标及纯观测目标两套剖面，每个固定参数节点均重拟合驱动与初态，
双向 warm start 并保留数值失败、路径依赖与预算。P2/P3 按整个参数向量检查
共同容许集合。容差以冻结 P0 目标为分母，并逐记录、逐模态限制 NRMSE 恶化；
有限搜索找到节点可提供存在性证据，未找到不证明不存在，网格跨度不当作置信区间。
信息矩阵分别保留固定补偿量的条件信息、消去补偿量后的数据项信息和加入原
正则后的曲率；不称为校准 Fisher 信息或生理后验。

W1 在开发记录内进行三块时间交叉拟合：每次隐藏各原 30 s 块的连续 10 s、
三个模态同时隐藏，使用这些留出残差估计 AR(1) 与三模态工作协方差。误差
精度按平均对角线归一化；它包含模型失配，不等同仪器噪声。E5 的共享中心
由开发记录的 pooled 目标估计，尺度是工作正则尺度，不把截断参数直方图当作
生理正态分布。强度选择使用只在源记录侧建立的共享对象，对目的记录作条件
重建；最终总体对象再由全部开发记录冻结。每个参数层次最多保留两项单因素
改进及其组合，评价结果不再参与选择。

合成校准使用多真值、零/小/大个体差异和独立重复，同时给出真实驱动、初态
已知的 oracle。生成域失败与优化失败分开保留在计划分母，成功子集的误差分位数
须显式注明。实测没有个体生理参数真值；最佳数值拟合、重复性、原始观测保真
和个体差异恢复分别判断，不能互相替代。本轮不访问其他协议的 protected split，
不授予 teacher/tokenizer 生理资格。

## EEG 功率代理采样率与实测重建（2026-10-10）

`SSM-EEG-PROXY-RATE-v1` 按用户要求在原三数据集固定公开开发面板上比较
4 Hz 与原生 EEG 重新计算的 10 Hz 功率代理。数值、身份范围与停止规则由
[`eeg_proxy_rate_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/eeg_proxy_rate_v1.yaml)
持有，入口为 `experiments/scripts/evaluate_eeg_proxy_rate.py`。先做数组级测试及
已知生成器的合成检查，再通过持久 supervisor 重算原窗口的 EEG；4 Hz 特征须
复现保留 parent 数组，Hb、外折 PCA、加载与训练 SD 原样继承，不新拟合评价尺度。

A4 为原 4 Hz EEG／4 Hz 驱动网格；B4_grid10 保留相同 EEG、仅加密驱动网格；
C10 使用 100 ms 非重叠均方功率的 10 Hz EEG／10 Hz 驱动网格。所有方法的 Hb
输入、目标及滤波支持仍为原 4 Hz 坐标。B/C 的 Hb 均值算子与自由度相同；EEG
误差按 `4 / EEG采样率` 加权，保持每单位时间的模态权重，曲率和 flow 先验继续
使用已有时间积分定义。固定 H0 参数，无额外观测成分，零驱动／静息初态同起点。
这是工程加权重建，不将平滑后时间点视为独立噪声似然。

主要端点是完整观测拟合下相同 HbO/HbR 目标的冻结训练 SD NRMSE；同时报告
对原 4 Hz EEG 目标的兼容重建、各自采样率 EEG 误差及完全隐藏 Hb 的预测。
共同 EEG 评分使用潜在读出映回原 4 Hz 处理算子，不声称是 log-power/PCA 的
精确逆聚合；不同采样率自身目标的误差不能单独用于优劣排名。沿用功率 bin
左端时间戳，因此 100/250 ms 窗口的支持差异属于本次特征变更，不解读成新的
生理延迟。按窗口配对、被试内平均 NMSE、被试等权后开方，并给出条件于冻结
坐标的被试块 bootstrap 区间。保留完整失败分母、状态范围及耗时；无测试调参，
不将原 QC 选择面板称为新确认，不开启其他协议的 protected split。

## 观测物理量、事件锚点与竞争机制（2026-10-09）

`OBSERVATION-SEMANTICS-v1` 执行本次用户提供的 A–E 计划。数值、划分和停止规则由
[`observation_semantics_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/observation_semantics_v1.yaml)
持有，入口为 `experiments/scripts/evaluate_observation_semantics.py`；先运行 synthetic
软件检查，再在持久 supervisor 下运行 pilot 和各阶段。输出使用独立的
`observation_semantics/<run>/`，旧证据只读。该套件是公开开发诊断，不改变正式
SSM/tokenizer，不开启其他比较协议的 protected split，不构成 teacher 晋级。

- **A：语义稳定性。** 复用明确指定的成分归属六窗口 profile 和 72 窗口敏感性数组，
  在保留 HbO/HbR 幅度关系后计算 HbT 与 Xρ。分别交付完整预测、物理项、额外项的
  sampled lower/upper、训练坐标 SD 中的范围和有效性掩码。5% 工程目标 profile
  与改变测量/先验/加载设置的敏感性集合分开；单一解不能被判为稳定。这里只评估
  保留的有限候选，并复用强化实验的 BC/HselectedC 响应选择及 initial/response
  tangent-guided profile。后者仍是固定响应参数的成分位移再拟合，不能被叙述为
  连续参数的完整 profile；未探测方向和原生连续初态竞争仍未被排除。
- **B：事件级实测关联。** Single-Trial 的 session_00 和 Simultaneous 的 cnt_wg，
  从冻结 parent 的合法时间身份按双时钟不重叠、时间先后确定每记录至多八窗。
  奇数编号被试训练，偶数编号被试评价；尺度、事件阈值、回归和条件时间模板都只用
  训练被试。使用统一 reader 返回的 signed HEOG/VEOG 和六路原生前额 EEG 电位。
  VEOG 幅度/HEOG 斜率越阈事件是检测候选，不冒充人工标注眨眼/眼跳。
  比较真实 EOG、同条件错训练被试和非环绕八秒时移；评分共同支持为 [8,21) 秒。
  HbT/Xρ 与原始正光强的 OD 一步差分是独立目标，不以拟合的 SSM 残差为真值。
  Hb 端点使用冻结的即时及延迟事件 FIR 特征。电位预测相减后再走原有五频带功率链，
  报告功率交叉项，不能将 log-power 变化直接命名为眼电幅度。
- **C：硬件与空间对照。** 仅在有经核实 source/detector 身份的 Single-Trial 上，
  逐目标通道比较一个共享 optode donor、一个模板距离尽量匹配的非共享 donor、
  一个更远的非共享 donor。三者均为两路 Hb 输入的相同容量 ridge，冻结于训练被试；
  评价目标 Hb 完全隐藏。共享硬件与组织重叠仍可能混杂；距离匹配剩余差异必须报告。
  Simultaneous 现有几何表的 source/detector 缺失明确标为不可执行，不能由邻近关系补写。
- **D：匹配观测的竞争机制。** 用现有非线性六状态 SSM 生成 τ、Hb 增益、额外 HbT
  和时移，按预定 RMS/谱/波形距离选择最相似的非零干预。比较无锚点、正确锚点、
  错配锚点及推理缺失锚点。锚点是合成的独立增益标定、时间标定与额外 HbT 观测，
  没有 τ 或机制标签输入。训练身份仅定尺度，评价身份只生成、匹配和评分；没有评价调参。
  本阶段向所有候选提供相同已知驱动/初态，是条件 oracle 能力上界，不能替代实测
  teacher 保护证据。报告错误类别、物理/成分误差、τ 变化被虚假分配到额外项的
  有符号投影，以及固定似然比候选集合的实际覆盖率；该集合不是已校准后验。
- **E：独立机制验证。** 需要另外的局部短间距与同步血压/CO₂ 等数据；缺少这些
  文件时记录未执行，不以双波长导出的 HbT、模板几何或合成锚点代替。

所有结果先在窗口/目标上配对，再在被试内平均，最后被试等权；D 以生成身份为块。
区间为冻结模型的 2,000 次块 bootstrap percentile 区间，单侧 sign-flip 与 Holm
仅在声明的主要检验族内使用。计划数、可用数和失败原因同时保留。报告说明原生单位、
训练 SD、离线处理支持和误差方向，PPT 与 PDF 均使用 PNG 图形；正文及表格保留文本。

**C/D 的有界补充合同。** 首轮保留结果揭示两项设计限制：最近共享 donor 的非共享
对照距离普遍更远；已知驱动/初态和白噪声的合成面板具有分类天花板。
[`observation_semantics_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/observation_semantics_v2.yaml)
因此只增加两项敏感性诊断，不改写首轮结果、不重新读取原生数据、不重做 B：C 从全部
合法共享/非共享 donor 配对中按纯几何最小化距离差，并在预定 0.02 模板头单位的
caliper 内给出主要对照，全通道为次要描述；这仍是同一公开开发人群，不称新确认。
D 固定 stationary AR(1) ρ=0.95、相同边际噪声 SD 和新的训练/评价生成身份，采用
与生成器完全一致的协方差白化似然，并据此匹配波形。补充检验一次完成全部单元，
不根据新结果调整噪声或锚点。其目的为检验有效时间信息量和距离混杂，不是获得
预定正结果。局部短间距或独立生理来源的实测验证仍需 E 的新增观测。

## 有限响应适配与独立观测约束（2026-10-03）

`SSM-STRENGTHEN-v1` 执行用户提供的“有限生理响应适配＋独立观测约束＋成分混淆
检验”方案。可执行数值与预算由
[`ssm_strengthening_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_strengthening_v1.yaml)
持有；唯一入口为
[`evaluate_ssm_strengthening.py`](../experiments/scripts/evaluate_ssm_strengthening.py)。
启动范围为 S0、S1、冻结空间预测的 S2、S4，以及开发面板上的最小二乘二比较。
S3 的扩大、S2 的目标软修正、短因果读出和 S5 的新实测确认有条件地保留。原
response-discovery、区域谱模式与归属实验保持原身份。

S0 固定原逐点共享驱动、五个独立血管初态、H0、训练坐标、处理算子、先验和
求解预算。B0 无结构项，BC 使用原 D-fixed，原 D-trained 仅作第二参照。独立
条件均值通过平移可见观测进入原精确系数消元求解器，不引入新生理状态。先做
纯合成隐藏值隔离与已知单位变换检查，再核验输入、校准对象及历史核心源码的
SHA256，并重放预定身份的原预测。缺失材料记为 `missing_input`，不重新生成后
冒充原输入；新计算的本地输入哈希不虚称历史封印。

S1 的九点 tau/kappa 库在训练内层选择身份上按隐藏 Hb 评分。每个候选同时评价
有、无结构项，先被试内平均，再被试等权；失败使用固定惩罚并保留状态。Htau、
Hkappa 和联合族的选择表及统一脉冲输入响应曲线同时保留。每个数据集、目标区域、
外折内的候选冻结到评价窗口及供体，不逐测试窗口选参，不把近等价响应强行解释为
参数恢复。合成选择单位同样是训练总体，不是评价场景或单窗真值。

S2 拼接每个同步、不同 Hb 对供体的 HbO/HbR 残差，保留固定身份顺序。供体用自身
可见 EEG/Hb 做当前窗口拟合；目标训练残差来自完全隐藏 Hb 的 EEG-only 预测。
训练被试分为三组，适配响应只由另外两组的 B0 隐藏评分选择，再在留出组生成
残差。交叉生成的残差训练秩一／秩二 ridge；训练 RMS 尺度、四模式低频支持、秩与
正则均在训练／内层选择中冻结。无逐目标窗口自由系数或截距；秩二增加观测读出
容量，不能称等容量生理源比较。原平均供体、训练模板、空间 ridge、任务条件匹配
错被试和非环绕移位保留；移位只在共同支持上比较。无合格供体或训练身份组合
时保留失败，不以目标信息或补零替代。
原始供体空间 ridge 只依赖观测和冻结读出，不继承供体或目标 SSM 的数值失败。
对已完成运行的此类基线修正使用同一入口的 `--stage correct-ridge --source-run`
写入新的版本：逐位核对原目标与尺度，仅重新评价 ridge，保留原拟合、数组和失败
记录，并在 `correction.json` 中说明哪些行改变；不得原地覆盖原运行。

旧面板继续条件于继承的外折 PCA、尺度和结构项校准，不称整个处理链的完全嵌套
交叉验证。外折评价被试不得进入任何目标／供体区域的坐标、响应或读出训练。
合成训练、选择和评价生成流互斥；四个固定响应总体下，共享神经变化、共同有色／
相关成分、仅全局成分、目标特有变化、供体污染，以及真实幅度与观测增益组合
共享可审计反事实身份。真实响应、真实结构成分和联合 oracle 仅作合成定位。
输出的固定训练尺度误差与真实变化的有符号投影分列；近零真值变化的投影未定义。

S4 区分驱动、初态、log(tau)/log(kappa) 切向与结构基底的数据空间重叠，以及加入
原先验后的重叠。响应切向仅诊断，评价参数仍冻结。固定共同项的指定变化后重拟合
物理轨迹并计入原系数惩罚，保留隐藏误差与分解轨迹；5% 工程目标差不是置信区间。
局部初态／驱动切向 RMS 之比不称解释方差。独立空间均值没有目标拟合方向，其
稳定首先来自独立条件约束，不能自动宣称由目标数据识别。风险指标为完全隐藏 Hb
时 H0 与训练选择响应的物理 Hb 差异，阈值只取选择流中位数；新合成流报告覆盖率、
实际误差和“错误放心”比例，保留所有同分样本，不用测试真值重选阈值或称校准后验。

实测隐藏 HbO、隐藏 HbR、中心 Hb、完整目标 Hb 对分别报告，完整拟合仅辅助。
窗口／区域先在被试内汇总；被试整块 bootstrap 条件于冻结训练器与坐标，不把
重复窗口当成独立被试，不宣称多端点整体显著。失败与实际可用配对数均保留。
F00=BC、F10=H-selected+C、F01=H0+冻结空间均值、F11=H-selected+冻结空间均值；
标签复用同一拟合，不增加独立样本或虚增求解次数。parent cohort 已参与过研究，
未找到未经决策使用的新身份前，不执行或宣称 S5 独立确认。连续状态实验须先
验证连续生成器和原生时间支持，不能拼接现有经过独立处理的窗口。

初态混淆触发后的小规模 S3 由
[`ssm_state_continuity_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_state_continuity_v1.yaml)
与 [`evaluate_ssm_state_continuity.py`](../experiments/scripts/evaluate_ssm_state_continuity.py)
执行，只使用独立的连续 90 秒合成生成流。C-free 将相同观测分成三个独立拟合，
C-chain 只在起点估计五个血管初态，之后由原动力学传播；两边均有 360 点驱动、
每段四个共同项系数和相同分块观测算子。删除跨段二阶差分行，不另加驱动或共同项
连续性先验。每个估计初态使用相同原权重，因此 C-free 有三份初态惩罚，C-chain
只有一份，须同时报告这一差别；不强制静息或 p=v。原目标单窗拟合复用 C-free
中段结果，不能重复计算为独立样本。共同的中段评分区间有完整、中心 Hb 遮挡和
全中段 Hb 遮挡，前后文可见，因此是处理后特征的离线平滑，不是未来预测。

合成观察算子在每个 30 秒块内重新参考和滤波，是为保持比较相同而声明的人工
分块处理；生成器的血管状态本身不重置，不据此宣称原生连续实测处理已经验证。
训练种子只确定固定尺度，pilot 与评价种子独立。rest/nonrest 指整个 90 秒区间
起点，不能把中段起点视为静息。针对 C-free 中段初态最强重叠的共同项方向，
两臂施加同样固定尺度的变化后重拟合，保留全部 90 秒目标代价和中段隐藏误差。
这是固定全部共同项系数的受限 profile；能证明找到的替代解存在，不能证明没有
其他混淆方向。以独立生成身份给出配对区间，不把同种子条件或目标单窗别名当作
额外独立样本。未通过隐藏预测、真值保护和失败率条件时，不扩大到连续实测。

入口先 `--check-only`，再在持久 user systemd 下 `--stage pilot` 测量并发吞吐和
峰值内存，最后 `--stage run`。固定任务清单、单线程数值库、有界在途任务、超时、
域失败、不收敛与终态文件一并保留；终态失败不自动重跑替换。资源和结果在本轮
版本化 run 中，注册状态由 registry 及生成视图发布。不增加 tokenizer、来源命名
或比较协议受保护评估的授权。

## 响应动力学与连续语义原型（2026-10-02）

`SSM-SEMANTIC-RESPONSE-DISCOVERY-v1` 在用户要求的原始 semantic–observation
叙事内，联合检验 SSM 改进和有限 tokenizer 原型。执行数值由
[`semantic_response_discovery_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/semantic_response_discovery_v1.yaml)
持有；状态由 registry 持有。前轮 modes 和 attribution 证据保持原身份。

本轮固定 EEG-only 的两条连续模式，再以相同 Hb 前缀校准响应初态及共同观测项，
不让隐藏 Hb 反向重估驱动。这是新的推断制度，不把历史联合拟合视作同条件对照。
比较宽频驱动、训练选择谱混合、使用同一 gamma 的纯衰减、独立选择纯增益、
零驱动及三个预定随机正交谱方向。主比较为谱混合对匹配衰减和独立增益，
从而区分谱信息与单纯减弱错误血管输入。

动力学竞争保留固定 H0 参数，分别增加输入低通或静脉流出松弛，不在单一臂
同时自由拟合二者。后者采用 `dv/dt=(f-v^(1/alpha))/(tau+tau_v)`，相同新出流
进入 p、q 的质量平衡；零扩展须数值回归原核。形式依据
[Buxton 等，2004，式 11](https://fmri.ucsd.edu/tliu/pdf/buxton04_model.pdf)，
本数据能否区分机制仍待检验。主初态绑定 p=v，使用 log 坐标 proper prior；
rest 和旧自由初态仅作固定消融。由相同方程可得 p−v 是衰减暂态，不能将其
自由度解释为新增持续生理过程。零 Hb 前缀只使用先验，禁止隐蔽初态或共同项拟合。

训练均值、任务条件模板、Hb 自身前缀、完整 EEG ridge、其他区域 Hb ridge、
其他 Hb 加 EEG 的即时／稳定双级滤波响应构成对照。空间项在首次读边界排除目标
Hb，模板可用任务标签但标签不进入语义 teacher。超出自身历史及空间观测的 EEG
增量才是跨模态信息候选。选择仅用选择被试的主前缀，冻结后复用于零前缀；
评价 CRPS、Gaussian NLL、训练尺度 NRMSE 及校准覆盖，不将工程损失解释为后验。
任务条件匹配错被试、非环绕移位、其他区域 EEG 和训练配对置乱保留相同评分支持。

数据只来自明确 parent prepared 数组，按元数据种子一次冻结被试分组，重拟合
训练尺度，不继承旧 outer PCA/SD。具体身份由本轮 `metadata_plan_proposal.json`
持有。同步三区域随原生窗口及被试分组，Visual 使用 Probe1 并去除重复 EEG 窗口。
此前整个 cohort 已分析，因此新评价身份仅减少本轮直接适配，不构成独立确认。
parent 非因果处理保留，所有后缀端点称离线处理后特征补全。

合成选择与评价种子分离；标准、非 rest、输入滞后、出流松弛、共同 Hb 和无耦合
真值在多个幅值与持续时间下交叉拟合，报告机制混淆与状态功能恢复。另建立按基础
轨迹分组的 paired counterfactual 数据，用受限／随机化生成器训练独立 EEG、Hb
编码器。连续语义目标为参考后共同驱动的单位 RMS 形状及 log 幅值，幅值分布明确
条件于生成假设；observation 分支重建完整本模态，不只重建 teacher 残差。
语义监督臂阻断重建到语义头的梯度，每个解码器只使用自己模态的语义均值和
observation 坐标。相同总容量的 observation-only 对照允许全部均值坐标学习重建。
四臂、三种子、固定训练上限及选择早停后，在未见动力学／时间基底、增益、谱旋转、
相关共同项及无耦合场景评价；精确相同观测但不同真实幅值的 alias 不能从语义评分
中删除，须报告覆盖、宽度与自信错误。各模态推理不读取另一模态或生成真值。

最后冻结编码器在本轮公开分组导出连续及 patch 表征，以训练线性探针比较 source
加入完整 observation 后的条件 Hb 增量，同时对比等容量 observation-only。
目标 Hb 编码器与 observation 表征只读该制度可见前缀，零前缀不用目标 Hb token，
禁止读待评分后缀；其他区域 Hb 及 EEG 可用全窗。合成完整 Hb 恢复与此条件推断
分表。同一冻结监督模型的 O→O+S 与对独立等容量 observation-only 的比较分别报告。
`best_stable` 只在输入低通、出流松弛中按数据集×区域的选择前缀选取，冻结增益后
比较，同分选输入低通。2% NRMSE 及区间/null 条件仅判断工程响应是否值得继续；
本轮语义指标为描述性结果，不作 teacher 晋级判定。即使重建改善也不自动训练 VQ
或宣称细胞、代谢、E/I、唯一神经来源。

实测读取前明确 primary null family：40 点前缀的三个 null 臂乘三类 null，共九项。
每项差值为训练 SD 标准化 Hb NRMSE 的 null−real，正值支持真实配对；时移只与
`real_shift_support` 比较。先被试内平均窗口和区域，再数据集内被试等权，最后
三数据集等权。固定这些权重，按有效被试整块翻转差值符号，完整枚举所有符号配置
计算单侧 p（含原始符号，不加 Monte Carlo 修正），九项统一 Holm；某臂三类 null
均通过才满足此特异性条件。该检验假设给定冻结训练器后被试差值在零假设下对称，
枚举精确不等于无条件随机化检验；只作开发性判断。保留缺失/失败分母和实际人数，
不凭共同成功子集单独宣布通过。零前缀、各数据集/区域及其他主端点的效应区间
保持描述性，九项校正不扩张为对 tokenizer 或 gamma 对照的整体显著性声明。
先软件与合成检查，再读取公开数组；所有较大生成、拟合及训练均使用经资源预试的
持久 supervisor。完整保留失败与负结果，不根据评价结果追加调参。

## 区域与谱模式共享驱动检验（2026-10-02）

`SSM-SHARED-DRIVER-MODES-v1` 回应用户对更细空间、生理结构的 `r(t)` 及实际数据
测试的要求。数值合同由
[`shared_driver_modes_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_modes_v1.yaml)
持有，新入口为
[`evaluate_shared_driver_modes.py`](../experiments/scripts/evaluate_shared_driver_modes.py)。
这是公开开发诊断；旧归属、稳健性与参数实验的配置和结果保留原身份。

每个已登记区域采用两个固定坐标：宽频同向 log-power 模式 `a(t)`，以及 alpha/beta
相对其余已测频带的谱对比模式 `b(t)`。空间单位保留 parent 的前额、运动和后部
Hb 锚点及六个 EEG 邻居，分别估计区域轨迹，不宣称恢复精确皮层源或网络连接。
EEG 读出矩阵的列、范数和符号预先固定；血管输入为
`u_H=(a+gamma*b)/sqrt(1+gamma^2)`，固定读出范数以免权重选择仅改变隐含驱动先验，
通过既有
非线性 Balloon 核生成 HbO/HbR。不同群体读出的研究动机可参考
[Hermes 等，2017](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2001461)，
但本实验的固定谱方向和权重是待检验假设，不是由该文标定的生理参数。
`a/b` 是相对包络模式，不是绝对神经活动、同步率、代谢量或 E/I。

比较固定 PCA 方向的一维重拟合、相同多通道多频带观测的一维模型、两维且
`gamma=0`、训练选择 `gamma` 的两维模型，以及独立驱动对照。独立对照保留两维
EEG 状态并另加一条 Hb 驱动，明确多一条轨迹。PCA 参照在原 30 维特征中以固定
PC 回投拟合，保证隐藏通道不通过 PCA 泄漏；它不是历史逐点驱动 M0 的数值复现。
所有臂使用相同的低阶余弦时间支持和稳定 AR(1) 创新约束，固定 H0 生理参数，
保留相同四维共同 Hb 观测项。EEG 和 Hb 生成均值分别经过原时间算子；每模态损失
按坐标数平均，避免增加 EEG 维度自动增加其权重。24 个余弦模式包含 DC，在
30 s 窗内最高包络频率约 0.383 Hz。首 5 s 参考使 EEG 不约束 DC，故状态恢复与
语义解释以参考后的轨迹为主，原坐标和 DC 单列。初态、物理域和数值失败显式保留。

只读取 2026-09-28 public parent 的明确 prepared 数组及身份元数据，复用其中
PCA 前的六通道五频带特征；不重新处理原始信号。原预选 72 窗、18 被试作为身份
锚点，加入其同记录、同原生窗口的另外两个已准备区域。三个区域不是三个独立
被试或试次，统计先在被试内汇总。REFED 缺少跨模态共定位，不进入该局部实验。
训练及内层选择沿原被试不重叠的身份，其他区域通过同步身份映射；尺度、共同项
约束和 `gamma` 只由允许的训练部分确定。继承的 parent PCA/Hb 坐标是共同条件，
不能把内层选择称为整个预处理流程重新嵌套的无偏验证。

主要结果分列完整重建、隐藏 EEG 通道、局部隐藏 Hb 和完全隐藏 Hb 的补全。
真实配对与任务匹配错被试、非环绕移位、同窗跨区域 EEG 交换比较；移位与真实
配对使用相同目标评分支持。自身上下文、训练模板和训练 ridge 保留为参照。
这些是处理后特征的离线补全，不是原始连续信号的因果预测。已知真值的独立合成
种子先检查宽频变化、谱模式变化、二者变化、共同 Hb、独立 Hb、谱方向失配和
时间偏差，分别评价两个状态、血管输入和串扰，不仅报告观测重建。

先通过软件、形状、隐藏值隔离和合成运行，再进入实测。使用固定任务清单与预算，
完整保留失败；不根据外折结果调整模型。若两维只改善 EEG 拟合，而隐藏 Hb 或
真实配对不优于对照，则只保留区域谱模式重建用途，不升级为共享生理 teacher。
若 Hb 结构性残差保持不变，不以继续增加状态维数追逐该误差。本诊断不训练
tokenizer，不打开任何比较协议的受保护评估。执行状态和科学判读由 registry
及其生成视图持有，运行、资源、源码与结果在版本化 run 中保留。

## 结构性成分归属与分级语义检验（2026-10-02）

`SSM-COMPONENT-ATTRIBUTION-v1` 执行用户本次提供的四组实验要求，合同为
[`shared_driver_attribution_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_attribution_v1.yaml)，
沿用现有 runner 的 `--component-attribution` 入口。上一轮四臂和更早证据保持只读。
本次用户要求覆盖新公共开发诊断与有限合成原型；不涉及比较协议的受保护评估。

1. **归属审计。** 复用上一轮 18 人、72 窗的目标与分解，检查重建收益与记录级
   native Hb 同步、窗口级同步、HbT/HbX 能量、质量负担、共同项强度及单条 Hb
   补全损害的关系。HbT=HbO+HbR，HbX=0.35HbO−0.65HbR，只作诊断坐标，不重复
   加入损失、不称血氧饱和度。重跑原 12 项扰动以保存物理 Hb 与共同项的变化、
   形状和 loading；既有 mask 继续视为工程筛查。原预选六窗的 Jacobian 分析区分
   H0 自由坐标与额外生理参数切向，沿最大重叠方向固定共同项偏移后重拟合物理
   部分，保留共同项惩罚；5% 目标差不是置信区间，也不施加硬正交。
2. **方向与独立通道。** M0、固定同向、训练选择同向、交换样方向共用四个余弦
   模式和相同 Hb loading 范数。方向网格只在原训练内被试不重叠的选择记录上
   按隐藏 Hb 评分；尺度由训练残差估计，先验强度沿用 parent 训练值。旧 outer
   PCA/SD 保持固定，故内层选择条件于该坐标，并非重新拟合整个嵌套预处理。
   使用同记录同原生窗口、不同 Hb 对的其他区域残差估计一个共同标量，再以训练
   冻结的 [0,2] loading 预测完全隐藏的目标 Hb 对。训练模板、EEG-only M0、
   空间 ridge、任务匹配错被试和非环绕 12 s 移位一并保留。目标被试不得进入
   任一区域的坐标或 loading 训练。AR(1) 与零、保持、余弦外推比较只回答离线
   处理后 Hb 后缀补全；窗口局部算子和参考不支持原生连续跨窗状态或因果预测主张。
3. **真实变化的反事实。** 独立训练种子选择强度/方向后，8 重复×2 频谱×2 初态
   检验原反事实及方向错配、交换成分、kappa 改变、未建模的快慢驱动响应。保留
   驱动、物理响应、额外成分真值，分别计算变化误差及观测变化在物理、额外、
   残差中的有符号投影份额。tau/kappa 真值参数 oracle 只作合成定位，不扩大实测
   自由参数。动力学失配由 Hb 接收 0.65r(t)+0.35r(t−2s)、EEG 保持原驱动构造，
   标明这是竞争生成机制，不把它强行命名为观测噪声。
4. **来源锚定与有限原型。** 几何与原文说明先区分独立空间关系和来源测量。
   Simultaneous 原文所有源探距均为 30 mm；通过统一 native reader 对原 24 窗
   及 parent 校准训练记录读取 HEOG/VEOG，以固定 0/1/2 s envelope 和训练 ridge
   预测条件共同项及隐藏 Hb；与零、训练均值、匹配错被试比较。EOG 是眼动相关
   辅助观测，不能据此命名自主生理或浅表血流。Single-Trial 文档中的 ECG/呼吸
   采集描述不等于当前 loader 已提供这些测量，Visual 模板坐标不提供精确源探距。
   来源未定继续显式保留，不为完成实验虚构外部锚点。

有限 tokenizer 使用单独的连续 typed-token 原型，不复用封存 R/VQ 训练流程。
EEG-only 编码器预测参考后的 12 个驱动片段均值，Hb-only 编码器预测结构性 HbT
的 4 个余弦投影；各模态独立 observation 编码器保留重建信息。Hb decoder 使用
成对、停止梯度的语义条件，属于 privileged 重建。训练损失不反向改变语义编码器，
也不强迫语义、额外成分和任务相互独立。对照为同容量重建训练及训练内线性探针。
256/64/128 个互斥生成种子分别用于训练、选择和评价，3 个优化种子；分列见过和
未见过的干预。目标真值来自合成生成器，不是未经准入的实测拟合轨迹；不训练 VQ，
不发布实测 tokenizer 资格。token 保留类型、来源状态、坐标和支持信息。

数据读取限定于已核验公共 parent prepared 数组、原面板、训练身份及同步区域；
新增原始读取仅限上述 EOG 记录并走中央 registry/统一 reader。先完成软件和合成
执行，再进入公共实测阶段。阶段、任务、分母与失败保留在独立 attribution run；
固定预算执行全部预定单元，不据外折结果调参。计算采用 user systemd、单线程
数值库、实测吞吐决定并发。最终中文 PPT 写入 `docs/report/` 的新版本目录，
PNG 图像与可编辑文字/表格分开；状态和判读仍由 registry 发布。

## 观测失配与驱动先验的四臂检验（2026-10-01）

`SSM-TEACHER-ROBUSTNESS-v1` 执行用户本次提供的计划，合同为
[`shared_driver_teacher_robustness_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_teacher_robustness_v1.yaml)。
沿用现有 runner 的 `--teacher-robustness` 入口，六状态方程、H0、训练坐标、
预处理与 2026-09-28 QC 预选的 72 窗口/18 被试面板均固定。新观测项属于本轮
明确请求，不修改下节历史设计或已完成证据。只读该公共 parent 的明确 prepared
数组、cohort、坐标和元数据，不进入任何比较协议的 protected 数据。

四臂为 M0、M-observation、M-prior、M-combined。观测项沿用 volume 方向，
固定 HbR 比例 0.35、4 个余弦模式，系数带 proper 二次约束；对每条轨迹精确
消元，分别保存物理预测、额外成分、完整预测。该项不命名为头皮或脑血容量。
先验沿用驱动幅度接口，覆盖曲率罚不约束的常数和线性方向；实测尺度来自
外折训练 H0 潜驱动的 RMS，额外成分尺度来自这些拟合的 Hb 残差系数。
这些是工程先验，均不称观测噪声或校准后验。每外折另外保留被试不重叠的
训练内选择窗口，分别选择观测系数尺度倍率和驱动权重，组合臂复用两者；
不根据外折成绩更换候选或继续调参。已冻结 parent PCA/SD 为所有臂共同条件。

合成先用独立 seed 流选强度，再执行 8 重复×2 频谱×2 初态、13 个配对反事实、
3 个可见性条件与四臂。观测失配包括基底内/外共同成分、与驱动相关共同成分、
EEG/Hb 增益、噪声、1/3 s 时移；真实变化包括驱动幅度/形状与 tau 变化。
另用 1e-6 增益作近零耦合对照，不放宽核心严格正增益域。既报告完整潜驱动
误差，也报告首 5 s 参考后的驱动误差，避免把离线观测算子的零空间隐藏掉。

实测对同窗口执行 full 与六种特征遮挡，比较训练模板、自身上下文、own/cross
ridge；错误配对来自外折训练中的不同被试且 task/condition 匹配。非环绕
12 s 时移与真实配对共用可见支持。原坐标 RMSE（Hb 为 parent 处理后原单位，
EEG 为 log-power PCA）、训练 SD NRMSE、窗口局部 SD NRMSE 和相关性分列。
缺失任务属于离线处理后的特征补全，不是原传感器预测；full 会重拟合驱动
与初态，不是独立预测。时钟审计复用原生事件锚点并检查公共处理算子，不根据
外折曲线逐窗选时延；没有新增硬件时钟测量。

敏感性覆盖 ±10% 观测校准、±1 样本时移、先验强度、固定共同方向变化与
训练记录 bootstrap。teacher sidecar 保留目标、逐点敏感性标记、坐标、
参数策略和未解释残差；缺失条件稳定性单列。0.25 个训练潜驱动尺度的阈值
是预设工程筛查，不是概率或语义资格。预选每数据集两名被试做参数—轨迹
profile，每格重拟合驱动、初态和允许的观测项；5% 近等价目标差不是置信区间。
跨记录/分块只比较可比较的轨迹功能及敏感性，不把不同任务的瞬时轨迹当重测真值。

所有已登记单元在固定预算下执行，失败保留分母，不根据结果扩大预算。先
synthetic/software，再读取上述公共面板；由 user systemd、单线程数值库、
受限 in-flight 队列执行。结果进入独立的 `shared_driver_teacher_robustness`
命名空间，最终 PPT 写入 `docs/report/`；本轮不训练 tokenizer，不自动发布
个体生理参数，不启动下一轮 H1/H2 或生理状态扩展。

## 固定结构的跨数据集生理语义检验（2026-09-28 设计）

设计标识：`SSM-PHYSIOLOGY-SEMANTICS-v1`。本节回应用户在复核当前能力后提出的
新设计要求：暂不增加独立慢共同成分，先检查疑似问题观测，再比较参数共享方式、
轨迹拟合、被试内重复性和现有观测层的分离边界。用户随后明确要求启动本轮全面实验，
并将最终汇报保存为 `docs/report/` 下的 PPT，替代 technical report。
可执行合同为 [shared_driver_physiology_semantics_v1.yaml](../experiments/configs/physiology_semantic_tokenizer/shared_driver_physiology_semantics_v1.yaml)，
执行状态与结论仍只由 registry 及其生成视图发布。已有波形成分实验保留为历史
诊断，其建议不构成本轮增加状态或观测成分的依据。

**设计与执行的关系。** 下文保留设计时的候选范围；实际执行由上述冻结配置、
阶段源码与运行记录确定。H2s 使用由参数搜索区间确定的固定 log 收缩尺度，
未执行内折选择收缩强度或一标准误自动选择层级；因此 H0–H3 的外折比较是
描述性策略比较，不是已验证的最优参数分配器。H1/H2 的记录轮转采样与 H3 的
短校准块预算不同，主比较限定共同的非重叠后续窗口。H1g 在合成层级实验中直接
拟合，在实测迁移子面板中以来源数据集等权 log-τ 集成诊断，没有混池未标定 β。

实际准备使用原生时间与窗口局部算子，避免全记录离线处理跨越训练支持；这也
使本轮目标不同于历史整记录缓存。REFED 保留 Hb 审计，未补造 EEG–Hb 共同几何。
连续上下文子面板使用同一连续支持、公共参考和未带通 Hb，属于另一个诊断坐标；
缺失和前缀任务是处理后特征补全。所有 bootstrap 区间条件于冻结的坐标和参数，
没有重新拟合外层标定。数值审计保留原 4 子步结果，并增加全量冻结潜轨迹的
8 子步重放及预选隐藏 EEG 子面板的 8 子步重新拟合；不据此修改生理结构或先验。
这些执行边界也须出现在最终 PPT 中，运行结果通过[保留证据索引](../experiments/RESULTS_INDEX.md)定位。

### 检验对象与三个独立问题

保持 `src/inference/t3a_balloon_robust_ssm.py` 的六状态动力学，以及
`src/inference/shared_driver_reconstruction.py` 的单一逐时点驱动、五个血管初态、
无逐时点血管创新的重建结构。沿用当前 inlet-balance 的 p 方程，不顺便替换为文献
另一版本；不加入慢共同成分、静脉黏弹性、自由观测混合矩阵或新的残差过程。
参数拟合接口的必要扩展与状态方程修改分开。三个问题各自出结论：

1. **取值方式：** 固定、数据集共享、被试共享、记录共享和 trial 独立，哪种参数
   分配能在未参与选参的目标上保持拟合，同时避免不必要的自由度？
2. **参数语义：** 在已声明观测规范下，哪些参数可恢复、可重复、能保留真实差异；
   哪些只是补偿尺度、初态、噪声、输入代理或数据问题的有效参数？
3. **观测分离：** 哪些可观测组合、相对变化和跨模态增量受数据约束；哪些成分
   只能作为条件模型输出，现有观测不足以唯一分离？

联合全观测重建继续是主要拟合端点；遮挡补全、参数恢复和生理来源归因独立报告。
训练冻结参数后的验证 trial 仍需拟合自身驱动和初态，必须标为重建，不能标为预测。
当前信号 SD 工程目标不是校准似然，不能凭其 Hessian 或目标差给出生理置信区间。

### D0：数据身份与高度同步观测的独立审查

范围为中央 registry 中四个公共数据集的全部合同允许被试/记录；确切身份、可用
事件、缺失和排除原因由准备阶段的唯一 manifest 生成，不继承旧三被试小面板。
数据与单位只认 [DATA_CONTRACT](DATA_CONTRACT.md) 及其原始说明入口。主处理采用
v5 none；保留原生/发布层与处理层的对应关系，不将 none 称为无伪迹真值。
Visual S06 Part1 继续按原合同排除。比较 campaign 的 sealed 材料不属于本设计。

执行时的窗口内参数共享/遮挡测试从统一原生 reader 构造局部特征，避免整记录缓存的
零相位处理跨越训练/验证边界：保留当前 Hb 的 none、带通、重采样及共同参考算子；
EEG 采用各原生窗口的 1–45 Hz 分带 log-power。首 5 s 仅作为观测参考，连续任务中
不称为静息或刺激前基线。此局部处理分支与整记录 v5 缓存有不同边界条件，不能把
两者误差变化都归于模型；v5 保留为独立原生/处理层 QC 对照。遮挡端点限定为
处理后特征层，不能宣称原始传感器预测。合成同时检查直接观测与上述处理观测，
扩大实测参数自由度采用后者的恢复筛查。可选 Numba 仅替换同一 RK4 算法的算术，
每个选中拟合仍经原 Python 核心独立重放；默认历史后端和状态方程保留。
全量验证的各参数法共用零驱动、物理静息初态起点；多起点用于参数训练和固定平衡
子面板，避免把验证端不同求解投入与参数分配的收益混在一起。

| 数据集 | 主要覆盖与独立重复单元 | 可审查的测量层与解释限制 |
| --- | --- | --- |
| Single-Trial | 全部公共被试，MI/MA 及其真实 session，全部合格 trial；用原事件和 registry 核验 session 编号映射 | 正光强、双波长 OD、相对 MBLL、v5 Hb；个体光程与绝对基线未标定 |
| Simultaneous | n-back、DSR、WG，按原 session/block 边界分组，不把拼接容器当一个独立 session | 已发布 oxy/deoxy 及有证据的浓度单位；不重复 MBLL，也不假定有原始光强接口 |
| Visual | 完整连续任务与 Part/Probe 身份；密集事件的重叠响应保留 | EDF 与发布 Oxy/Deoxy；Hb 单位未核定；不能把每个 12 s 事件当独立静息起始的 30 s trial |
| REFED | 全部合格视频及对应 baseline；video 是记录/刺激重复单元 | HbO/HbR/HbT 与三路 Abs 分开；HbT 可核对代数闭合，不能作为第三份独立测量重复加权 |

审查先于 SSM 误差和参数选择，保存 `dataset/subject/record/block/trial/channel/
native_time/support/processing` 身份。仅读取原文和 metadata 不算信号 QC 已完成。
审查包含：

- 原生通道/波长/色团顺序、符号、单位、重复读取或复制列、时间戳与双模态事件映射；
  光强非正、饱和/平顶、掉线、突跳和不连续支持；检查局部与多通道同步是否共同出现。
- 原生/发布层、去线性趋势、1 s 差分、v5 处理层的双 Hb 相关、滞后及谱相干。
  `r>0.5`、`r>0.8` 是描述层；另列 `r<-0.5` 和中间组，不将反相关直接当作质量证据。
  近乎完全正/负线性关系都检查复制、秩与变换，但不能仅凭相关阈值认定损坏。
- 有原始 optical 输入时，在正确波长、对数底和系数下重现转换，并做已知响应及
  光学系数敏感性；只有发布 Hb 时明确上游不可核查。Hb→OD→Hb 的代数闭环不算
  独立验证；不能用更容易拟合 SSM 的转换规则作选择标准。
- EOG、设备坏道记录和真正可访问的辅助生理记录只作独立证据；Single-Trial 文档
  提到 ECG/呼吸，仍须核验实际文件、通道和时钟，不能假设这些信号已可用。

每个窗口分别记录**确定的支持/身份错误、质量疑点、相关方向**，不合成一个按模型
残差排序的好坏分数。主表保留全部有合法测量支持的窗口；另列有/无独立质量疑点及
相关方向的交叉分层。确定错误由数据 owner 定位，修复必须新版本，并在相同身份上
给出修复前后对照；无法修复的缺失保留分母和理由。单纯高度同步不自动删除、降权、
修改 `valid_mask` 或改变参数自由度。来源未明的高同步组保留诊断结果，但不承担
唯一生理归因。本轮不为改善这部分拟合增加成分。

### D1：空间、时间、尺度与划分

**空间。** 主分析为局部 EEG 代理与一个真实 Hb 对。按版本化 geometry 和原文的
对应关系，预先列出前额、运动、后部三个区域中各数据集实际支持的锚点；每区主锚点
只按通道身份、真实支持和与目标误差无关的规则确定，跨折固定。没有可信对应的区域
不进入局部共享来源主张，可进入 Hb-only 观测审查。其他合格通道用于预定空间复核，
不按最低 NRMSE 替换主锚点。不同区域分别报告，不能把不同被试选到的不同通道混成
一个被试参数；模板位置不作为个体皮层配准。

**张量和目标。** 基本输入仍为 `[trial,120,3]`、4 Hz、30 s，三列为现有
1–45 Hz log-power/PCA EEG 特征与成对 HbO/HbR；保留原生支持、时间算子、训练 SD、
观察坐标及物理单位证据。动态状态为 `[trial,120,6]`，中心遮挡为连续16点。
Hb 两列共用观测尺度，分项评分 SD 独立记录；不按每 trial 单独 z-score。
短事件/视频的 30 s 窗口仍属于其原连续记录，不改称新独立 trial。真实连续上下文
的 30/60/120 s 诊断在共同中心30 s评分，分别为120/240/480点；不拼接不同 trial。

**划分先于依赖数据的处理。** 跨记录/被试比较以整个 record/session/video 成组，
同一连续块的相邻/重叠窗口不能穿越外折。记录内适配比较另有明确分块身份；
需要记录内分块时，先切分原生时间支持再局部构造特征及时间算子；
两侧边缘按训练前冻结的滤波支持规则舍弃。用扰动另一折原始值不影响本折输入的测试
检查隔离，不能假设一个短间隔足以消除双向 IIR 的依赖。现有全记录离线 v5 缓存可作
完整重建/描述审查；若跨分块滤波，不能直接承载独立分块预测结论。

固定种子 `20260928` 生成身份划分，synthetic truth/noise、优化起点、donor、bootstrap
使用独立随机流。所有候选复用同一身份、单位/尺度、可见支持、任务权重及失败分母：

| 检验 | 外层留出 | 它回答什么 |
| --- | --- | --- |
| 已知被试的新记录 | 有真实重复 session 时留一 session；否则至少三个连续独立块轮换，REFED 按 video 成组 | 被试参数能否迁移到新记录；分块稳定性不冒充跨 session/跨日重测 |
| 未见被试 | 每数据集按被试固定随机五折，内折只含外层训练被试 | 数据集共享参数对新人是否够用；同刺激泛化与新刺激泛化分列 |
| 新被试有限校准 | 同一未见被试保留预先指定的最早合格独立块作校准，其余记录评价；各方法校准支持一致 | 少量个体数据的价值；这是个体适配，不是 zero-shot |
| 未见数据集 | 四次留一数据集；仅比较空间与特征语义相容的端点 | 动力学参数能否迁移；未知单位/观测映射无法冻结时，严格零样本幅度比较记为不可评估，单列观测校准迁移 |
| trial 内部诊断 | 固定前10 s作局部校准，后20 s Hb 不可见而 EEG 可见；与相同可见支持的共享参数比较 | trial 特异参数是否增加跨模态补全信息；不称严格未来预测 |

内层选择参数集、共享层级及正则；外层只评价。任务标签仅用于原任务分层、保持
split/donor 可比和解释结果，不作为驱动输入或按任务拟合不同生理参数。REFED 的
相同视频跨被试重复：新被试/同刺激与新被试/留出视频双重分组分别汇报。
密集 Visual 事件按连续块抽样与重采样，不能把重叠响应当作独立统计样本。

### P：参数取值方式的主对照

先固定观测映射、幅度规范和正则，再比较生理参数层级；不同时调观测增益、过程噪声
和全部生理参数。初态和驱动本来就是 trial 局部变量，不计作个体生理参数自由度。
主比较沿用初态权重100、驱动曲率0.01、驱动幅度0、log-flow权重1及其log尺度ln2；
这些是工程惩罚。子面板另对flow权重0.1/1/4作预定敏感性，不能从外折成绩选择权重。

| 量 | 本轮默认处理 | 允许检验的变化及解释 |
| --- | --- | --- |
| `r(t), s0, f0, v0, p0, q0` | 每个连续窗口推断，沿用合法物理域与初态合同 | 审查初态与驱动 DC 补偿；连续记录的跨窗一致性单列，不把基线扣除等同生理静息 |
| `tau` | 参考2 s；首先研究数据集/被试/记录共享 | 以秒报告模型条件估计，取值区间0.5–8 s及起点1/2/4 s是工程搜索设置 |
| `beta` / 相对增益 `g` | 现有观测规范下参考增益，先单独拟合 | 只称有效耦合；相对范围0.1–10、起点0.5/1/2继承诊断语义，不是人体正常范围 |
| `kappa` | 主比较固定0.64 | τ、β辨识问题定位后，单独检验0.2–1.5及起点0.32/0.64/1.28；需要补足当前共享参数接口 |
| `gamma, alpha, E0` | 主比较均固定0.32 | 仅做预定敏感性/剖面；不根据困难 trial 拟合值宣称测得自身调节、血管弹性或氧提取 |
| `P0,Q0`、EEG loading、DPF/光学敏感度 | 采用有来源的测量映射或显式条件参考，冻结后比较 | 不将71 µM/DPF6套给所有数据集；相对单位不获得绝对浓度语义 |
| 时间算子、观测尺度、噪声假设 | 全部从允许的训练/校准支持确定 | β跨记录比较必须固定或可验证地重表达观测规范；不把尺度变化混入被试差异 |

参考常数来自当前实现，不代表任何被试的真值。生理方程、光学映射和当前实现相对
文献的差异分别留证；[Tak 等的原始模型](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401444/)
提供参数作用的理论参照，不提供这四个数据集的个体标定。

对**同一可识别参数集**比较以下分配方式；跨单位、EEG规范和空间支持不相容的 β
不进入全局共享，时间参数也只能在合格区域中比较：

| 方式 | 参数如何取得 | 对照意义 |
| --- | --- | --- |
| H0 固定参考 | 不从本次观测估计生理参数 | 所有适配的共同基线 |
| H1g 跨数据集×相容区域共享 | 仅从训练数据估计共同时间参数，各数据集等权；不混池未标定β | 区分参考常数不合适与确需数据集差异，提供迁移基线 |
| H1 数据集×区域共享 | 仅用训练被试/记录，组间等权估计 | 是否需要设备/范式相关的有效参数；差异不自动称人群生理差异 |
| H2 被试×区域共享 | 用该被试的训练记录估计，向新记录冻结应用 | 本轮个体化的主要候选 |
| H2s 部分汇聚 | log参数向训练被试形成的数据集中心收缩，强度仅由内折选择 | 少 trial 的稳定性；同时保留不汇聚结果，防止把先验压缩当可识别 |
| H3 被试×记录×区域共享 | 只用该记录校准块估计，向独立后续块冻结应用 | 是否确有记录差异，或只是在吸收观测/接触变化 |
| H4 每 trial 独立 | 全窗拟合只作自由度诊断；有独立校准支持才做上述 trial 内部补全比较 | 查明最乐观拟合和参数游走；不能用全窗重估参数的成绩战胜冻结参数预测 |

参数集按 `固定 → 仅tau / 仅beta → tau+beta → 必要时增加kappa` 检验，不能把
改变参数数量和改变共享层级的收益合在一起。只有合成与训练端剖面支持联合辨识，
联合参数集才进入实测主比较；否则继续单参并报告无法分开的组合。gamma/alpha/E0
的敏感性使用现有波形诊断网格及版本身份，仅在固定平衡子面板评估，不扩为全组合
搜索。触界保留，禁止看到外折结果后扩大范围或缩窄先验。

设计时 H2s 是待实现的参数估计候选，不能用旧 T3c admission 或旧运行代替其验证。当时
共享非线性入口仅支持 tau 或 beta 单参；tau+beta、kappa、部分汇聚和四数据集
prepare 尚需局部实现与合成测试。这些接口工作均保持现有状态与观测成分数。

### S：已知真值与可辨识性测试

在实测拟合前，用独立生成/拟合分辨率及固定随机流覆盖三类问题；不以旧合成结果
继承替代本设计的检查。匹配方程使用当前核心，可另用高精度积分作数值参照，不把
高精度参照当作另一物理真值。

- **匹配恢复。** tau=1/2/4 s、g=0.5/1/2、slow/mixed驱动、每条件8个独立噪声
  重复，按单参/联合顺序运行；噪声与初态跨真值配对。保留18训练/6验证结构，
  固定和独立非静息初态均检查，生成8子步、拟合4子步、独立8子步核查。
- **层级恢复。** 分别生成全体不变、数据集差异、被试差异、记录差异和trial差异；
  同时生成“生理参数不变、只有观测幅度/相对Hb增益/噪声变化”的负对照。
  每机制8个独立虚拟队列，每队列4组×4被试×3记录×6窗口；只对通过上一项的
  最小参数集比较共享层级，不与全部噪声、网格、上下文作笛卡尔积。
- **失配压力。** 只在观测中注入已知幅度的共同/独立有色扰动、漂移、突跳、通道
  交换和时钟偏移；拟合器不获得新增成分。检查扰动被写入参数、初态、驱动还是残差。
  分别改变真实EEG–Hb配对和驱动频谱；增加30/60/120 s真实连续上下文，避免把已知
  驱动频谱、短窗零基线或极低噪声当作恢复能力。

报告参数偏差、真实排序/差异保留、原坐标driver NRMSE、各状态RMSE、失败/触界、
先验敏感性；去均值/幅度对齐的误差只作附表，不替代原坐标恢复。合成能恢复增益
而driver DC不能恢复时，分别判读，不赋予完整状态资格。

每个待解释参数做训练端多起点和 nuisance 重新优化的参数剖面；先固定候选参数值，
再重拟合其余参数、驱动和初态，不能用固定其他量的切片冒充剖面。记录数据项与
工程惩罚项，以及低误差替代解的参数/driver差异。当前目标只称“目标剖面”；只有
独立验证噪声模型和完整似然后才采用似然比置信阈值。采用剖面区分结构/实际不可辨识
的依据见 [Raue 等原始研究](https://pubmed.ncbi.nlm.nih.gov/19505944/)。
局部投影雅可比/SVD用于定位弱方向，不替代全局辨识、重复性或区间覆盖验证。

### F：拟合与独立补全的共同评价

主实测面板覆盖全部合格被试/记录，固定区域分别汇总；固定/仅tau/仅beta在通过S后
以H0–H3最小对照进入全量。通过S的联合参数、kappa、多起点剖面、额外通道、上下文
和H4诊断使用固定平衡子面板，不因子面板正结果自动扩大全量参数自由度：
每数据集按D0质量疑点负担低/中/高三层、每层两被试随机选取，不足则全取；选择不看
SSM误差。既有S09困难身份另列历史复核，不进入随机子面板的代表性结论。

| 端点角色 | 指标与比较 |
| --- | --- |
| 主要：观测重建 | EEG/HbO/HbR各自训练SD归一化NRMSE；全计划分母上“成功且三项<0.5”比例；同时报告完成率及条件误差 |
| 主要：参数分配 | 在相同可见支持、观测规范和参数集上比较H0–H3外折达标率/误差与自由度；H4全观测单列 |
| 次要：波形 | 各分量相关、振幅/峰谷深度偏差、低频功率与残差结构，NRMSE中位/P90/最差例；峰谷规则由训练/合成确定 |
| 次要：缺失恢复 | 中心4 s EEG、中心4 s双Hb、单独HbO/HbR，以及整模态缺失；同模态可见上下文、训练模板和同特征ridge为基线 |
| 诊断：状态/数值 | 完整轨迹及RK中间态合法性、最小flow、初态/状态偏离、惩罚贡献、梯度终止原因、积分差和所有失败 |

完整观测时不能用目标本身作为“重建基线”；采用固定参数SSM及明确自由度的同特征
线性模型。中心缺失时同模态插值、含另一模态的ridge和无另一模态ridge使用相同
输入/目标支持。整模态缺失时不允许该模态的插值、初态估计或缩放读取隐藏值。
参数、PCA、尺度、正则、初态初始化及模板均执行隐藏值干预测试。
若只能在处理后特征层保证隔离，结论仅为该层补全；未来/原始传感器预测另需合同，
不由本实验自动获得。

配对对照包括真实EEG–Hb、同任务且记录条件相容的trial donor，以及大于预定相关尺度的
时间移位；donor来自允许训练库且保持幅度/支持相容，不能跨折取目标或循环绕回造成
虚假事件连续。时间移位大小与可用支持由合成/训练冻结，缺少合格donor标为不可评估。
空间对照只在有多个真实相容锚点的记录进行。要求报告真配对相对自身上下文、线性
跨模态基线和null的增量；联合重建变好但null无差异，只支持拟合灵活性。

每个数据集先按window→record/session→subject等权汇总，再给数据集等权描述。
保留质量疑点×相关方向×区域分层；跨数据集汇总不能掩盖单一数据集失败。
配对区间采用被试聚类bootstrap，记录内以连续块为单位；用于新刺激泛化时视频也
成组。相关方向与误差的关系先在被试内比较，再合并，避免再次被S09占比混杂。
重叠外折与多ROI不当作独立样本；预声明多组比较用Holm校正，其他分析标探索性。
未校准的工程SD目标不报预测NLL/CRPS/覆盖已通过；这些端点只有噪声与推断资格
独立验证后才有概率解释。

### R：被试参数稳定性及取值方式的判读

同一被试的独立session/不重叠块分别估参；重叠训练折仅反映训练扰动，不能当作
重测。β的重复性要求同一观测规范：可用外层训练被试或独立第三校准块确定共同
PCA/尺度/光学映射后，冻结到A/B；没有此支持且不能精确重表达时不计算跨规范β
生理稳定性。共享校准的有限样本不确定性在外层重采样中保留。

同时报告：log参数的被试内变化、被试间方差、记录/区域效应、多起点范围、边界率、
绝对一致性ICC及被试聚类区间、重复估计差值图。ICC只在相同任务支持/区域/观测规范
且有真实独立重复时计算，并明确是块间、session间还是跨日；不能只用高相关或
收缩后的小方差主张稳定。部分汇聚与无汇聚并列，检查“所有人回到先验中心”的负例。

以下数值是本设计建议冻结的工程判读标准，不是正常生理范围：

- 拟合支持：各数据集主要面板至少95%数值成功，至少80%的计划窗口三项NRMSE均
  <0.5；报告实际比例及区间，疑点组仍保留，不通过者缩小支持主张而非删除记录。
- 参数恢复：拟赋予个体语义的参数在匹配合成上相对误差中位≤10%、P90≤25%，
  非零真实差异方向保留≥90%；另报失配压力下退化，不以匹配恢复保证真实数据语义。
- 稳定性：同规范独立重复的参数对称相对差异 `2*abs(A-B)/(A+B)` 中位≤20%，绝对一致性ICC的95%区间下界
  ≥0.60，且剖面不是平坦/触界/纯先验限定。区间过宽或重复不足为证据不足。
- 取值层级：内折按全分母达标率的一标准误规则优先较简单层级，外折一次评价；
  报告共同成功身份误差和新增失败。若个体化只降低全窗残差却不改善独立记录表现，
  不支持将个体估计作为稳定生理特征。H3仅在同预算独立块优于H2时保留其工程价值。

这些标准不能单独授予绝对生理参数资格；没有独立尺度/来源证据时，最终名称仍为
“条件时间参数”或“有效耦合”。trial级参数若只有全观测拟合收益，建议使用共享
生理参数配trial驱动/初态；只有trial校准后独立补全也稳定改善，才讨论局部参数。

### O：现有观测层能分离什么

当前核心在光学/时间变换之前的观测关系为 `EEG=a*r+b`、
`HbR=Q0*(q-1)`、`HbO=P0*(p-1)-Q0*(q-1)`。本轮固定 `b=0`，没有独立拟合偏置项。
若幅度映射已知且可逆，瞬时观测
直接约束r、p、q的三个组合；s、f、v没有独立直接测量，依赖跨时刻动力学约束。
瞬时矩阵秩3不等于整个动态系统不可观测，也不能证明六状态已可辨识，需要下面的
时序/参数检验。未知尺度、处理算子的低秩方向和观测混合会进一步限制实际恢复。

| 待检验内容 | 不增加成分的实验 | 最多允许的结论 |
| --- | --- | --- |
| 双Hb的两个观测方向 | 相同单位/公共尺度下分析HbT=HbO+HbR及HbO−HbR，检查秩、噪声相关与原生转换 | 两个观测坐标/可用信号方向；线性重表达不增加独立测量，差分不直接等于氧代谢 |
| 驱动可见部分与血管响应 | EEG-only、Hb-only、联合；真实配对与donor/移位；合成已知r、f/v/p/q | 模型条件下的相对驱动/状态及跨模态预测增量；逐个量决定支持程度 |
| 初态与持续驱动 | 非静息合成、目标剖面、30/60/120 s共同中心评价、同记录跨窗估计对照 | 能否减少驱动DC与初态补偿；基线算子消失的常数方向仍不能凭拟合恢复 |
| 观测尺度与β | 已知单位变换同时作用于数据/均值/噪声及评分；冻结观测映射后做增益敏感性，查近等价解 | 规范不变性及有效组合；固定EEG loading不能替代光学/神经幅度标定 |
| 模型可解释部分与剩余部分 | 保留 y−yhat 的时间、频谱、空间及QC关联，合成扰动检查泄漏 | 条件模型解释量与残差；残差不自动等于噪声、伪迹或私有生理来源 |
| 皮层/头皮、局部/全身、流量/代谢 | 审查原数据中实际可用且独立的空间/短距离/外周记录，冻结参数后做外部关联 | 现有EEG代理和一对Hb不足以自动授予唯一来源解释；无独立记录则标未分离 |

如几个不同参数/状态解能产生近等价观测，保留整组替代解和不可辨识方向；不要挑
生理数值最顺眼的解。原生高度同步的来源审查与该项分离能力评估保持独立：存在
同步不会证明观测损坏，也不会证明模型已提取共同生理源。皮层与浅表混杂需要独立
依据的理由可参考 [Kirilina 等的多模态实证](https://pmc.ncbi.nlm.nih.gov/articles/PMC3348501/)，
该文不构成本项目已经找到浅表来源的证据。

### 实现、资源与交付边界

顺序为D0元数据/原生来源核查与synthetic软件准备，随后完成D1划分/观测合同、S恢复，
再做四数据集小规模资源pilot、全量P/F以及固定子面板R/O。D0审查可以与不读实测的
synthetic开发并行；某个参数集辨识失败，只终止该参数集的解释性分支，保留固定及
较小参数集的拟合对照。确定测量错误阻止依赖该错误输入的生理判断，不阻止其他
合格数据继续。pilot不以最低SSM残差选被试或参数。

扩展现有 `evaluate_shared_driver_reconstruction.py` 和其 `src/inference/` owner；
数据审查复用 `analyze_hbo_hbr_relationship.py`、统一loader及事件/geometry owner。
不新增平行launcher。实现将本节明确数值转入本节开头链接的版本化可执行合同，
本节保留理由与解释边界并链接该合同，不维护第二份可执行数值。输出仍位于
`experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/<versioned_run>/`。
新增默认测试用synthetic/临时fixture覆盖多数据集身份、split/遮挡隔离、参数导数、
单位不变性、缩放重表达、层级选择负例和失败传播，不依赖本地实测数组。

正式计算前按已有运行规范检查CPU配额/物理核、内存、I/O与竞争任务，测量不同
窗口长度/参数臂的吞吐后确定并发；数值库线程和在途任务有界，采用持久supervisor。
固定参数/单参全量优先；昂贵剖面、联合参数、H4和空间敏感性只用预声明子面板。
冻结任务身份与预算后不按结果追加重试/网格；失败、预算耗尽及未执行条件完整保留。
预计成本由pilot和实际身份表计算；本轮执行依据本节开头记录的用户明确启动要求，
不从历史运行或此设计推断对其他 campaign 的授权。

交付需能逐数据集、被试、区域和trial回答：推荐何种参数共享层级、与固定基线的
拟合/补全差值、参数能否重复及其观测规范、疑点数据造成的影响，以及每个状态/参数
只能达到哪一级解释。保留所有trial数值及可定位曲线，正文展示预定中位/P90/最差
和同步分层样例。运行manifest与registry各依现有职责持有事实；每组结果在对话中
主动汇报。最终交付为 `docs/report/` 下的 PPTX，图表使用 PNG，正文与表格可编辑；
内部 PDF 仅用于版面预览并遵循位图插图要求，不交付 technical report 或平行状态文档。

## 独立观测信息与非线性双向反事实（2026-09-18）

用户按同日附件要求执行 A–D 的依赖顺序。本轮入口为
[`evaluate_hbo_hbr_calibration.py`](../experiments/scripts/evaluate_hbo_hbr_calibration.py)，
执行合同为 [`hbo_hbr_calibration_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v1.yaml)。
它与旧线性增益/AR1 试验保持不同身份，不修改旧结果或已冻结配置。

- **A：独立信息盘点。** 以旧 Hb 适配试验的逐记录身份为范围，使用当前中央缓存索引、
  registry 和原始说明，输出已知处理/单位、光学入口、波长/几何、增益/混合、DPF、
  绝对生理基线、噪声、辅助记录和连续支持的可固定/可估计/区间/未知表。
  此阶段不读实测信号数组；未在所查资料找到证据不等于证明数据不存在。
  原始说明中 Single-Trial ECG/呼吸的线索必须与已验证的消费者接口分开。
- **B：首轮条件性恢复筛查。** 使用现有六状态非线性 drift，在受控的未知频带内
  r(t) 强迫下生成五个血流状态；完整经过合成强度、OD、现有 MBLL、10 Hz 采样、
  现有 0.01–0.2 Hz 带通、2 Hz polyphase 重采样和前 5 s 基线算子。
  加性误差在发布 Hb 层进入；没有加入血流过程扩散或原始光强噪声。
  均值与完整时序/双 Hb 协方差同步传播，低秩基线方向不以 jitter 补回。
  这不是任意 r 的随机 SSM 恢复或原生仪器噪声验证。
- 比较固定单位观测、同目标自由观测矩阵、真实观测 oracle、独立已知标准与噪声
  记录估计后冻结四种方法。两个额外拟合只检查独立校准最大不确定方向的正负扰动。
  所有拟合从固定的非真值起点开始，且拟合全部五个血流初态。P0=1、alpha/E0、
  kappa/gamma 与 driver 频带作为明确的有利条件固定，只拟合 tau/eta 和 r 系数；
  报告 lambda/a/k 不代表它们是三个独立自由参数，更不代表四个生理量唯一可辨识。
- 每个独立重复在 tau=1/2/4 s 下复用同一个真实 r，跨白噪声、独立/共同慢成分、
  相对 gain=2、波长依赖 DPF 混合及非平衡初态进行配对。DPF 0.8/1.2 是合成
  敏感性设定，不是任何实测记录的可信区间。独立标准和噪声记录不使用目标 Hb。
  首轮固定 8 次重复；不按结果重抽、调阈值或自动增加重复。
- 主端点是有效组合与不做后验对齐的 r 恢复；并列报告真实 tau 差异保留、HbO/HbR
  分项误差、校准敏感性、局部 Gaussian 区间覆盖、秩亏/触界和全分母失败。
  区间传播光学校准不确定性，但把估计噪声协方差条件性冻结；其有限样本不确定性
  没有额外积分，覆盖率需实证复核。不得仅凭触界减少或重建改善宣布通过。

**C** 需 A 建立可用于目标记录的独立校准依据，且 B 出现正恢复证据后才构造实测候选。
保持原开发身份，比较原坐标偏离、冻结误差评分和实际曲线修正；匹配随机参照分布，
并在已验证真实连续支持上比较 30/60/120 s，不拼接 trial。跨记录验证先固定预处理和
校准支持。**D** 需 C 正证据及相应非线性推断资格；再比较模态/中心遮挡、配对/
移位/错配和同特征线性对照，并审查等价观测解释下 r 的尺度、符号和时间稳定性。
本轮 A/B 不能替代这些条件，也不自动启动完整 teacher、tokenizer 或受保护评价。

所有运行使用独立持久 supervisor、冻结源码和配置、逐面板原子结果与唯一续跑锁。
执行状态由 registry 和 run manifest 持有；附加重复与实测候选根据保存的筛查证据决定。

## 观测校准归因与 oracle 定位补充（2026-09-24）

复用上述 v1 面板与已完成拟合，合同为
[`hbo_hbr_calibration_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_calibration_v2.yaml)，
入口仍为 `evaluate_hbo_hbr_calibration.py`。仅均值与仅噪声两臂补齐交叉比较；
另做无噪声闭环、已知初态/生理参数、慢噪声减弱、正确协方差下自由矩阵及固定 eta 诊断。
连续时长从同一120秒轨迹裁出30/60/120秒上下文，仅在共同45–75秒评价驱动。
局部投影信息与 tau/驱动系数剖面互补，不能用 Hessian 秩替代全局可辨识性。

剖面分辨率补充与发现局部解后的有限复核各保留独立版本合同、源码与输出：
[`profile resolution`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_calibration_profile_resolution_v1.yaml)、
[`basin check`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_calibration_basin_check_v1.yaml)、
[`profile basin`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_calibration_profile_basin_v1.yaml)。
原结果只读；复核起点来自已保存的目标拟合解，不能以真值初始化或覆盖旧 oracle。

`audit_hbo_hbr_calibration_sources.py` 将 Single-Trial 已有9条公开记录落实到原始双波长、
实际 MAT 通道、单位、ZIP 成员及事件时间，复用中央索引和既有读取器；不改通用数据幅度。
只有查得独立约束，才构造辅助观测候选并开展 C 的留出验证；文档声称存在但实际文件未找到的
ECG/呼吸不能替代数据，DPF 合成压力范围不能转成经验区间。D 仍依赖 C 的正证据。
汇报由 `render_hbo_hbr_calibration_report.py` 读取保存结果生成，图在 PDF 中均为位图。

## 训练冻结观测增益的探索性实测对照（2026-09-24）

按用户明确选择，使用既有四数据集 Hb 适配面板开展独立的探索性对照，合同为
[`hbo_hbr_predictive_separation_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_predictive_separation_v1.yaml)。
它不替代上节要求独立标定的 C/D，不修改其进入条件。入口复用
`analyze_hbo_hbr_relationship.py --predictive-separation`，先合成后实测，均由持久 supervisor 执行。

使用已有线性化双 Hb 必要关系，允许两个初态瞬变和常数偏置，固定 E0/alpha、拟合 tau/eta。
观测候选先在训练窗口、参考生理参数下估计一个跨窗口共享的 HbR 相对增益，再冻结增益拟合
生理参数；它是依赖参考模型的观测假设，不是独立光学校准。保留固定、只调生理、只调观测、
联合拟合，以及线性上下文和 HbO 移位对照，不搜索额外噪声自由度。

沿用原早晚分块和中间间隔，逆转父缓存逐窗口归一化后采用单一训练尺度与训练 HbR 评分方差。
留出窗口的 HbR 中间 10 秒在局部初态/偏置拟合前隐藏；HbO 全窗可见。主端点是相同隐藏目标的
标准化 MSE，按窗口后被试/折等权，另报告参数边界、参考增益偏置及完整失败分母。
上游离线全记录处理保持原身份，因此只检验处理后特征补全，不声称原始传感器遮挡、未来预测、
EEG–fNIRS 共享驱动恢复、唯一生理解释或 teacher 资格。

## 共享驱动重建与参数自由度筛查（2026-09-26）

按用户本轮要求，联合输入后的共享状态重建是主验收：EEG/HbO/HbR 分项
NRMSE < 0.5，分母固定为训练折 SD；遮挡预测独立检验。入口
[`evaluate_shared_driver_reconstruction.py`](../experiments/scripts/evaluate_shared_driver_reconstruction.py)
与 [`shared_driver_reconstruction_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_reconstruction_v1.yaml)
先检验现有 Balloon 的静息线性化必要条件，不把它称为完整非线性 SSM。

仅一个逐时点共享驱动和五个初态自由度，血流状态无逐时点创新；固定观测规范、
kappa/gamma/alpha/E0/P0/Q0，比较固定 tau、跨训练试次共享 tau、来源 log-prior
形状的 tau 正则。该正则施加于训练 SD 加权的工程目标，不是已校准后验，也不是
人群生理范围。另保留不加正则、在同目标上遍历 tau 的线性残差下界；它不参与模型选择。
聚合 NRMSE 是被试/会话等权 NMSE 的平方根，另列逐 trial NRMSE 平均。
三分量平均 NMSE 的最小值仍大于 0.25，才提示该网格/保留数值子空间无法同时达到三个
NRMSE < 0.5；单项超标不足以作此推断。保留 1e-10/1e-12/1e-14 SVD 截断敏感性；
截断后的最小残差不是任意连续驱动或完整非线性模型的数学不可能性证明。

先运行12组合成条件性恢复与软件检查，再读取既有 measurement-retest 的精确
prepared 折对象：3被试×4折，每折18训练、6验证，共72个唯一验证身份。
不重读原始数据，不改变训练尺度/选通道，不覆盖历史结果，不启动独立校准 C/D。
完整、中心特征缺失、整模态缺失及移位控制均用原均值时间算子；误差不作为噪声
似然或独立原始传感器预测。还报告同模态可见端点插值、状态正性与线性小信号适用性、
参数剖面和跨折稳定性。合成入口只检查计算和条件性线性重建，不授予生理恢复资格；
真值驱动/参数与失效分母仍完整报告。图按固定基线的中位和最差样例展示。

同入口的 `--replay-of` 按独立
[`shared_driver_nonlinear_replay_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_nonlinear_replay_v1.yaml)
固定该轮已保存驱动、五个物理初态及 tau，通过原非线性 Balloon 方程重放完整输入轨迹。
先合成后实测，每步4/8子步配对核查；初态或任何中间态非法均保留为失败，不裁剪、
不重拟合。分列原线性误差、非线性误差和二者差异；成功子集不能覆盖完整失败分母。
这只定位线性化影响，不是非线性优化结果、被试生理参数确认或 teacher 资格。

后续同入口的 `--nonlinear-fit` 使用独立
[`shared_driver_nonlinear_fit_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_nonlinear_fit_v1.yaml)。
原非线性方程中只优化共享驱动和五个初态，所有 RK 中间状态满足硬物理域；
用精确离散敏感度、阻尼 Gauss–Newton 和非法试探拒绝代替逐点血流状态修正。
先按较细8子步生成 slow/mixed 驱动合成真值，再以4子步拟合并核验8子步预测。
实测仍是上述72身份，原尺度、特征遮挡和18/6划分冻结。

为分开 EEG 过平滑与初态补偿，固定初态惩罚100，对固定 tau=2 比较驱动曲率惩罚
100/0.01；另在0.01下比较训练线性剖面选择的共享 tau 及来源先验 tau。四臂均预声明，
不按验证结果挑选一个“获胜”设置；tau 在非线性验证拟合时冻结，不能称为已完成
非线性参数估计。每轨迹预设合法线性初态与静息两起点，只在梯度收敛解间比较同一
目标；预算耗尽、域失败或积分分辨率失败均保留完整分母。低误差且数学有效仍需审查
初态补偿、状态幅度、被试参数可重复性及合成恢复，不能单凭曲线达标宣告生理资格。

后续增益与驱动幅度对照复用入口 `--gain-prior-fit`，合同为
[`shared_driver_gain_prior_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_gain_prior_v1.yaml)。
同一实测 prepared 范围、尺度与划分不变；固定 tau=2、EEG loading=1、曲率惩罚0.01和
初态惩罚100，用2×2四臂分开固定/训练共享有效神经血管增益与无/有驱动幅度惩罚。
有幅度惩罚时残差为 `sqrt(dt) * r / sigma_r`，sigma_r 固定为该训练折EEG评分标准差
除以EEG loading。它是所有模式共用的软工程先验，不把前5秒观测零均值当作生理静息。

训练增益用18条训练轨迹联合优化，固定三个起点、正增益搜索范围及每组仅一次的log先验；
所有起点进入终态后在收敛解中按训练目标选择，再冻结增益拟合验证驱动与初态。
无收敛共享参数则相应验证轨迹记为训练失败，不删除身份或退回固定参数。
训练起点独立并行，验证仍检查4/8子步与完整失败分母。先用固定tau=2、增益0.5/1/2、
slow/mixed驱动的新合成真值检验增益恢复，再单独启动同范围实测阶段。

有效增益依赖未经独立标定的EEG/Hb幅度规范；它不能直接称为被试神经血管效能。
同时报告局部数据信息与正则信息、起点一致性、触界、初态和整段状态幅度、隐藏EEG振荡。
四臂完整对照，不依据验证分数再挑选先验强度或更改评分分母；Hb配对随折变化的
既有混杂仍须保留，不能借增益自由度宣称已经取得纯粹的被试生理差异。

预算诊断继续使用 `--gain-prior-fit`，独立合同为
[`shared_driver_gain_prior_budget_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_gain_prior_budget_v1.yaml)。
从已完成的 gain/prior v1 中，仅对预算耗尽且没有物理域或导数拒绝的35个实测训练起点，
以其18条训练轨迹的最后有效解热启动；每起点另给3600次试次评估和90次迭代。
目标函数、尺度、参数边界、先验和收敛阈值均不变，记录首目标连续性、逐步目标、梯度、
参数和累计评估数。阻尼会重置，因此这是热启动预算扩展，不是完全复现不中断的优化路径。
其余37个实测起点保留父运行的终态，包括15个有域拒绝的失败；仍只在所有起点终态后
从收敛解中选共享参数。不能将近似相同的目标值或较小梯度替代原成功条件。

先通过不读取实测数据的合成热启动测试，再以分阶段入口核验并继承父运行的合成证据；
继承不算独立重复实验。prepared对象及固定增益A/B的实测验证结果原样继承，训练增益
C/D的实测验证全部按新选参重算，仍保留原72身份及训练失败分母。只访问父运行的精确
保留产物，不重读原始数据，不改变Hb位置，也不在同一对照中改变先验或修复域边缘失败。

固定空间位置与共享非线性τ对照继续复用入口 `--fixed-roi-tau-fit`，独立合同为
[`shared_driver_fixed_roi_tau_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_tau_v1.yaml)。
使用同一父运行保留的三被试完整特征及12折投影，按注册顺序预先固定 AF7Fp1；不按验证误差
选位置，不读取原始数组。先逐折核验身份和训练可用性，再保留原EEG特征、投影及评分SD不变，
对固定Hb通道用18条训练的公共MAD和父观测规范重建尺度，Hb分项评分SD也只来自训练。
旧通道的噪声对象不沿用；本目标使用信号SD工程加权，并非噪声似然。所有三臂在同一新目标上
重算；历史变通道结果仅作背景参照，不能将不同Hb目标的误差差值当作同目标配对改进。

固定β=1、曲率0.01、初态惩罚100、驱动幅度惩罚0，比较固定τ=2、共享训练τ无参数先验、
共享训练τ加中心2秒/log SD ln2的工程先验。τ搜索0.5–8秒并以1/2/4秒三起点训练；该范围和
先验不声称人体正常分布。每个起点仅使用18条训练轨迹，所有起点终态后才选择收敛解，随后
冻结τ执行full、中心EEG及中心Hb验证，仍保留72个唯一验证身份和所有失败。保存迭代轨迹、
多起点目标/参数差、边界、纯数据与正则贡献的信息诊断，不能将局部曲率称为置信区间。

先做静息无驱动的信息负对照软件测试，再运行12组合成τ=1/2/4×slow/mixed×2重复。
同一频谱/重复跨τ复用driver、初态与标准化噪声，8子步生成、4子步拟合并独立8子步核验；
保持18/6划分和实测相同的正则，报告τ、driver与血流状态恢复，先验压向2秒不算真实差异恢复。
相同光极位置不代表个体皮层配准，折训练集相互重叠也不构成独立被试重复证据；τ不直接
进入s/f方程，不能靠τ自由度宣称已解决血流幅度或观测标定问题。

固定位置的血流软约束敏感性实验复用入口 `--fixed-roi-flow-fit`，合同为
[`shared_driver_fixed_roi_flow_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_flow_v1.yaml)。
仅访问已完成固定ROI τ实验的精确保留对象，原样继承prepared目标、训练SD、18/6划分、
合成真值及全部身份；固定τ和无先验训练τ的零权重结果（包括失败）作为父证据继承，
不算新重复实验。新计算固定τ2/权重1，以及共享训练τ/权重0.1、1、4四臂。
每个输出时刻（含初态）加入残差 `sqrt(weight*dt)*log(f)/ln(2)`，训练与验证权重一致。
这是工程软锚，不裁剪轨迹，也不把ln2解释为正常人群范围或校准的不确定性。

固定β、驱动和初态惩罚、τ搜索范围、三起点、预算、收敛阈值与4/8子步检查均不变，
新训练臂不加τ先验。每臂内部等待三起点终态后选择收敛解，不能跨权重比较目标值，
也不按验证表现选择一个权重后只报告胜者。每阶段108个新训练起点；1296条验证记录
中432条继承、864条新计算。先完整合成终态再实测，合成阶段不得读取实测数组。

并列报告全72分母、成功子集误差、共同成功身份配对及新救回/新失败身份。固定τ的0对1
隔离软约束作用；权重1的固定与训练τ比较检验τ自由度；训练τ的0/0.1/1/4检验敏感性。
合成中检查τ、driver直流与振荡成分、flow和初态恢复偏差，同时保留最小flow、极端偏离
持续时间、域拒绝、最终梯度、积分差、多起点与局部信息。flow对τ的直接导数为零，
但软约束可经driver/初态和Hb耦合间接改变τ；数学正性、低NRMSE或较少域拒绝均不能
单独证明被试参数已具有生理解释。

初态约束机制对照复用入口 `--fixed-roi-initial-fit`，独立合同为
[`shared_driver_fixed_roi_initial_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_initial_v1.yaml)。
四臂为固定τ2/共享训练τ分别配自由初态或 `p0=v0`，均使用预设中间工程权重1；
不重新优化血流权重，也不将该选择称为生理先验或验证集胜者。绑定移除一个初态坐标，
保留原有v、p两项物理初态惩罚；线性初始化和非线性求解均在约束坐标中重解，
不能把自由解的p值事后投影为v。当前方程中该约束保持p=v，从而移除初态差值衰减项，
不改变动力学方程或光学观测尺度，也不声称其他初态必须静息。

合成τ1/2/4×slow/mixed×2重复×初态 `log(p0/v0)=-0.05/0/0.05` 共36组。
跨τ及初态条件复用driver、其余初态和标准化噪声随机抽样；比例是模型错设压力参数，
不是人体正常范围。四方法均新计算，每组18训练/6验证，先检查匹配真值的参数恢复，
再检查错设绑定是否将误差转移到τ、driver或其他初态。保留所有失败、真实初态、
训练信息、多起点差异及4/8子步积分诊断；局部信息仍不作为置信区间。

实测只继承固定ROI flow父运行精确prepared对象和权重1的两个自由初态基线，包含全部
训练起点与验证失败；新计算两条绑定臂。共享τ仍仅由18条训练轨迹估计后冻结，使用
原三被试12折、全部72验证身份及full/隐藏EEG/隐藏Hb口径。合成阶段不得读取实测数组，
完整合成终态与软件检查先于实测阶段；继承不算独立重复。主要比较同目标同身份的
重建误差、覆盖与τ变化，单列初态和血流范围；低误差、绑定成立或较少优化失败均不能
消除相对Hb幅度、EEG方向和个体光程尚未标定的限制。

生命周期修订：上段保留最初实测设计以说明冻结配置；后续同窗口光学运动核查发现
旧目标存在算法引入的慢下降，因此该初态版本止于合成终态，未执行的旧目标实测阶段
不再启动。数值manifest和冻结配置不改写；当前执行状态与结果以registry为准。

光学运动处理的只读管道核查使用
[`analyze_fnirs_motion_correction.py`](../experiments/scripts/analyze_fnirs_motion_correction.py)，合同为
[`shared_driver_optical_motion_audit_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_optical_motion_audit_v1.yaml)。
依现行公共数据接口政策，仅追溯同三被试、同72个保留窗口的九个原生光学缓存容器；
不重新运行SSM、不选择模型、不构建缓存、不读取原始EEG/MAT或执行保护评价。
统一index先核验所有record、事件与窗口身份，再读取两个指定光学key；按AF7Fp1实际
low/high波长标签匹配，且文件身份必须与保留来源一致。容器打开不增加输出窗口，
只转换这72段各300点的10Hz光学输入。

在相同OD基线、相对MBLL、滤波和重采样下，比较无运动处理、当前导数抑制及官方MNE
TDDR三路径。首先在零信号、常量、正负非对称周期、真实线性趋势与噪声上检验软件；
实测阶段要求当前路径与保留OD、Hb特征、4Hz输出的216项逐一复现检查通过后，
才可归因管道差异。完整保存72窗口、三路径、复现失败及有符号漂移/斜率/滤波后变化，
不将无处理或MNE当作真实生理曲线，也不把工程相对Hb改称μM。该核查与冻结模型实验
分开留证，不能用处理路径变化来重写旧NRMSE或宣称生理参数已校准。

光学观测版本对照继续复用模型入口 `--fixed-roi-optical-fit`，合同为
[`shared_driver_fixed_roi_optical_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_fixed_roi_optical_v1.yaml)。
只读取已完成光学审计的无运动处理/MNE原生相对Hb及flow父运行精确prepared对象；
不再次读取native容器，不扩大72身份或改变EEG、ROI、划分、MBLL、滤波和窗长。
先核验72身份和216项旧路径重现，再将两个新Hb路径分别以18条训练计算公共MAD与分项SD，
父观测loading保持冻结；保存未缩放Hb、比例、原尺度及训练索引。新旧目标及SD不同，
不能把跨路径NRMSE差值称为同目标性能改善；各路径内固定/训练τ可作同身份比较。

两路径均只比较固定τ2与无τ先验的训练共享τ，保持自由五初态和flow权重1、原三起点、
预算、域约束、曲率/初态惩罚及4/8子步检查，不同时优化极性、光学校准或初态绑定。
每个pipeline×被试×折单独训练，τ在验证冻结；联合重建为主，中心特征遮挡独立检验，
不将已经离线处理后的特征遮挡称为原始传感器预测。按路径单列全部失败、所有72验证身份、
τ多起点与跨折变化、flow和其他状态幅度，以及固定顺序曲线图，不能只报告较好管道。

先做新入口的fresh合成pilot；正式合成阶段精确继承flow权重1的12组两臂全部证据和失败，
明确不是独立重复。另用无运动、已知慢响应、噪声、孤立尖峰与永久阶跃的配对光学控制，
检查新增漂移、响应损失和上下文影响。MNE可能削弱真实慢变化，无校正也可能残留运动；
两者均不是ground truth。库中v4 schema必须显式指定none/MNE，v1/v3保留原数值行为；
现有缓存不原地升级，模型本轮使用经过逐项一致性核验的已保留光学审计数组。

条件性光学幅度与有效血管增益对照复用入口 `--conditional-optical-gain-fit`，合同为
[`shared_driver_conditional_optical_gain_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_optical_gain_v1.yaml)。
父级为已完成的光学版本对照；两管道合计24个prepared对象中，观测target、训练SD、
EEG、mask和18/6划分原样保留，不再读取native容器。同一管道内比较三臂：
继承的单位Hb映射/β1/τ2（含全部失败），条件性Hb映射/固定β参考，以及相同映射/训练共享β。
只用18条训练选择β，在6条验证冻结；τ固定2秒，五个初态自由，保持原惩罚、预算、
全RK阶段域检查和4/8子步精度标准。正性Hb初态新坐标在本对照关闭，仍用独立log坐标。

条件映射为 `B = k inv(Eold) [ln(10) × 1e−6 × Edecadic × 71]`，
`k`是父级训练MAD系数；760/850 nm与HbO/HbR顺序固定。B只进入预测均值算子
`L @ kron(I, blockdiag(1,B))`，不再次变换target或SD。消光系数取
[Prahl原始表](https://omlc.org/spectra/hemoglobin/summary.html)；
[Tak模型](https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf)中的71 µM和
HbR比例0.35仅是皮层源基线假设，不是本被试的静脉腔内浓度或正常范围。
此处额外假设源敏感度为1、光程3 cm×DPF6，未复现Tak的空间敏感度及pial权重。
两光程相同才可相消；不将条件性敏感度差称为已证实的标定错误。

β参考值为 `1/sqrt(det(B))`，是工程幅度参考；训练β的起点为参考值的0.5/1/2倍，
边界为0.1–10倍，无参数先验。β进入血管驱动方程，实际β与相对参考的g分开保存。
固定EEG loading并不能赋予EEG代理物理单位，因此β仍与光学灵敏度及driver规范混杂。
第一、二臂同时改变映射和β参考，也改变物理初态与flow惩罚的相对作用；只有第二、三臂
直接检验增加共享β自由度的贡献。不能把第一、二臂误差差值单独归因于光学校准。

先做fresh合成pilot，再做12组fresh合成：真g为0.5/1/2，slow/mixed各两重复，
预声明k40；各g配对driver、初态及标准噪声，B在加噪与生成目标之前施加。
合成三臂全部重拟合，不继承旧真值族结果。除重建外，检查真g恢复、多起点、边界、
driver/状态误差与初态贡献；具体恢复阈值及未通过时的停止规则由可执行合同持有。
完整合成检查后才进入同72身份的实测对照。每管道报告全部失败分母、同身份误差、
训练增益的跨折变化、状态幅度和固定顺序全曲线；联合重建为主，遮挡预测独立保留。
这组实验只能判断条件性观测/增益假设，不能建立个体绝对生理标定或teacher资格。

条件增益的数值步控制诊断复用 `--conditional-step-control`，合同为
[`shared_driver_conditional_step_control_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_step_control_v1.yaml)。
仅改变共享参数求解器的可选步长控制：当实际下降与完整Gauss–Newton预测下降之比低于0.25，
额外尝试一次受限二次插值，仍须满足原下降与物理域条件。默认Armijo路径保持原样；
目标、坐标、起点、3600次试次评估/90次迭代预算、梯度阈值和积分检查均不改变。

先在已保留的12组合成真值上比较全部36起点，要求没有相对父级的收敛退步，且通过原参数恢复检查。
之后只比较四个预声明受影响实测训练组的全部12起点，包含原成功对照；父级失败原样保留。
主要端点为同预算的收敛率，另比较目标、β、调用数及插值诊断。仅复制父级prepared和训练证据，
不读取新原始数据，不运行验证拟合，不由这组训练诊断改写旧重建成绩或宣称生理参数已识别。

2026-09-28 的多数据求解器验证扩展为同入口的
[`shared_driver_conditional_step_control_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_conditional_step_control_v2.yaml)。
在同一冻结代码与机器上从相同起点重跑 Armijo 与二次插值两臂，先做12组合成数据的36对起点，
通过两臂原恢复检查且无逐起点收敛退步后，再覆盖全部3被试×2光学处理×4划分的72对实测起点。
这是既有单一公开数据集的多被试检验，不视为跨独立数据集验证。目标、积分器、阈值及预算不变。
主端点为同预算收敛且不丢失任一原成功起点；性能端点预定为实测总优化forward调用至少下降10%。
另记录进程CPU时间、墙钟时间、目标与参数/轨迹一致性；固定种子交错提交两臂，以减小机器负载的时间偏差。
每次运行冻结源代码、命令和资源证据，由持久systemd服务执行；失败与未通过端点照实保留。

HbO/HbR 同向普遍性审计（2026-09-28）复用 `analyze_hbo_hbr_relationship.py --prevalence`，
合同为 [`hbo_hbr_prevalence_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/hbo_hbr_prevalence_v1.yaml)。
按中央 v5 索引的公共记录身份，从统一原生 fNIRS 入口读取未作项目时间处理的发布 Hb，
或 Single-Trial 正光强到相对 MBLL 的最少转换。主要端点为不重叠 30 s 通道窗口中
Pearson r>0.5 的被试等权比例；同时报告反向、弱关系、缺失、去线性趋势、1 s 差分、
v5 none 带通及公开光谱系数敏感性。Single-Trial 事件窗口与条件标签为次要分层，
不混入连续窗口分母。严格保留原生时钟与共同支持，排除合同中的 Visual S06 Part1；
跨模态几何不作为同一光学通道内描述统计的准入条件。先合成／fixture 检查，再监督试跑、
全量统计；不训练模型，不使用比较实验的受保护材料。已有 SSM 面板只作辅助关联，
显式排除合成单元、保留被试混杂，不能从相关方向唯一识别生理来源。

波形机制诊断（2026-09-28）复用入口 `--waveform-diagnostic`，合同为
[`shared_driver_waveform_diagnostic_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_waveform_diagnostic_v1.yaml)。
针对已记录的 S09/o3/trial3，先运行匹配动力学、错设κ/τ、静脉黏弹性、
共同血容量和等总Hb氧交换六个合成控制；再读取条件光学增益父运行中三个被试
outer=3的两管道精确prepared对象。保持18/6划分、目标、训练SD、观测算子和固定ROI。
线性无正则SVD筛查逐个改变β、κ、γ、τ、α、E0或静脉黏弹性时间，仅按18条训练
残差选择网格值；6条验证仅评估条件重建，不把逐验证窗口重拟合driver称为预测。
另比较joint与Hb-only，检查SVD截断敏感性。线性解保留状态偏离，不能替代非线性可行性。

静脉黏弹性仅在诊断线性化中用 `f_out=v^(1/alpha)+tau_v*dv/dt`，不改写原非线性核心。
新增的共同血容量与氧交换观测方向分别为 `[P0-Q0,Q0]` 和 `[-Q0,Q0]`，
各配相同六个预声明慢余弦基，再经原光学/时间算子；该表达能力对照没有独立输入，
不作为头皮来源或CMRO2异常的独特归因。两管道困难身份另做原目标、去曲率、
去初态惩罚、去flow惩罚、全去惩罚、Hb-only全去惩罚六臂非线性检验，保持参数冻结、
物理域、梯度阈值和4/8子步检查。原成功与失败证据只读，未启动新原始数据生产、
保护评价、完整新SSM训练或teacher资格评定。

固定新增成分比例后仍有残差时，版本化小范围后续合同
[`shared_driver_waveform_volume_fraction_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_waveform_volume_fraction_v1.yaml)
沿用相同入口、六组对象、六个慢基和18/6划分，只用训练目标选择新增成分的
HbR/HbT loading（0.05/0.1/0.2/0.35/0.5）。先通过0.1/0.2/0.35三种已知比例合成恢复，
再冻结训练选择的比例做逐窗重建；不改变Balloon的Q0或E0。该参数不是独立测得的
血氧饱和度，不能在没有新观测的情况下确定头皮、动脉或静脉来源。此后续属于
看过第一轮结果后的探索性诊断，不追溯声称预注册或独立确认。

### 贴边样例的驱动曲率与转运时间诊断（2026-10-10）

[`shared_driver_tau_curvature_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tau_curvature_v1.yaml)
持有这次小范围探索的完整合同，复用 `--waveform-diagnostic` 入口。对象是用户指出的
MNE TDDR、subject_01、outer=3、AF7Fp1 的保留验证窗口；只读取已完成公开光学运行
中的精确目标、重建及状态数组。原训练尺度、目标处理、固定生理参数和样例身份不变。
固定 τ 网格与曲率惩罚交叉，每个节点重新拟合逐时点驱动及五个初态；辅助对照分开
移除 EEG 目标、血流和初态惩罚。这里不重新选择训练参数，不把验证窗口全目标拟合
称为预测，也不把下界外的工程敏感性范围称为正常生理范围。

先检验小 τ 下的 Python/编译后端及步长一致性、已知驱动恢复和曲率抑制控制，再在
持久 supervisor 下完成计时 pilot 和有界并行诊断。报告同配对分母的 Hb NRMSE、
目标峰谷误差、EEG 代价、驱动粗糙度及全部失败；无正则线性单/双 Hb 筛查只用于
表达能力解释，不替代物理可行的非线性结果。具体执行及数值由新的运行证据持有，
不覆盖原始贴边参数和历史结论。

## 数据统一化与 SSM 拟合的联合修订（2026-09-15）

本节响应对数据缩放与 SSM 拟合的复核，定义后续候选方案的验证顺序。
单位、特征、baseline、尺度与噪声接口唯一由
[`DATA_CONTRACT.md` 第 5 节](DATA_CONTRACT.md#5-将状态空间拟合纳入统一化合同2026-09-15-修订建议)
持有。这里不改变下文保留的 v3 设计、冻结配置、数据身份或结果。
本节定义后续联合修订；本次新增的有界实测与合成缩放诊断见
[数据统一化诊断报告](../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/REPORT.md)，
该诊断不替代下列完整主协议，也未启动 protected evaluation。

### 依据与待检验解释

[v3 保留报告](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260911_observation_contract_v3_continuation_v1/analysis_20260911_v2/REPORT.md)
支持“已知时间处理必须同时进入均值与协方差”，但没有规则通过完整实测改进筛查。
[N1–N7 保留报告](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/analysis_20260910_v2/REPORT.md)
中的增益/Q 成功子集改善也没有建立无害的状态恢复改进。
精度审计、折内投影、逐模式完成性继续从原 run 读取，不复制结果表。

本次区分四个问题，分别验证：

1. **工程等价性：** 单位或计算缩放改变后，数据、观测均值、R 和目标是否仍描述同一模型。
2. **测量假设：** 先验预测幅度映射、噪声下限及特征选择是否在吸收采集/模型差异。
3. **推断稳定性：** 算术重组通过后，求解预算与合法物理路径是否仍失败。
4. **科学有效性：** 拟合观测更好是否同时保留合成潜状态和正确配对增量。

前三项不能互相替代；低 NRMSE、有限输出或所有 task 进入终态均不足以证明第四项。
本协议继续以物理 teacher、观测与耦合问题为范围，不用下游任务准确率选择预处理。

### 修订次序与最小对照

| 次序 | 只改变什么 | 对照与通过条件 |
| --- | --- | --- |
| 0：软件合同 | 明确单位/噪声层、保留 float64 特征与单一目标 producer；同步计算缩放；记录下限前后数值 | 既有单位重表达、线性 Gaussian 独立解、mask 干预、噪声传播与精度检查通过；原有错误缩放负对照仍应失败 |
| 1：数值修订基线 | 在旧统计模型下修复精度和算子重组；保持现有 PCA、Hb 对、幅度桥、R/Q 和 G/W/Z | 旧输出保留为历史对照；新 producer 与自身算子闭合。只改变单位/预条件时潜状态与参数推断应等价，不能把数值误差当科学收益 |
| 2：测量尺度候选 | 用数据合同定义的模型无关缩放与显式观测 loading，替代将训练折幅度匹配到模型先验预测 SD 的做法 | 先做独立合成，在训练校准组冻结参考桥后 apply；不同参数候选共享同一目标、尺度和 scoring SD；R 同步换单位，Q 不动 |
| 3：特征噪声候选 | 尺度固定后，单独检验训练特征噪声估计与有依据的下限 | 对照次序 2；检验已知噪声恢复、有效子空间残差与预测校准，不同时搜索增益或 Q |
| 4：固定参数实测验证 | 对已满足上述软件/合成要求的完整候选，复用 v3 的限定面板、同折线性对照与 14 模式 | 固定分母、共同身份比较；区分两方向 joint 结果与全模式完整结果；输入失败和求解失败保留 |
| 5：条件性模型适配 | 固定输入后分别检查一个共同 Hb 均值增益、或既有 Q STD 三点；不联合搜索 | 沿用独立合成适配检查，增益不重标噪声，Q 不触发重新缩放数据；不确定/不完整不作合格结论 |
| 6：跨数据集扩展 | 接入相同数据合同下的其他采集/单位组，再检验留组泛化 | 先完成单位/空间/噪声层的接口检查；每组完整性单列，独立校准与严格 zero-shot 分列；具体面板另行设计 |

次序 0/1 是工程修复，次序 2/3 才改变统计假设。旧桥已经在训练折拟合并冻结；
次序 2 要检验的是幅度参照取自模型先验的影响。
第一版候选不扩展 EEG 频带、空间搜索、增益上界、生理参数网格、状态维数或面板。
更改 EEG 参考、长上下文滤波、脑区选择等会改变真实观测，应独立比较，不能夹在精度
修复里归功于缩放。先锁定上述最小比较，再考虑这些数据集层面的后续差异。

### 软件与合成验证

- **单位压力检查：** 同一观测以 V/µV 或 mmol/L/µM 等价表示，经同步观测变换后，
  比较潜状态、参数后验/目标函数差和逆变换预测；加入只改数据、不改均值/R 的负对照。
  同时覆盖弱 HbR、退化 baseline、不同采样时钟及缺失模式。大动态范围检查要区分
  单位换算和 SVD 截断的数值影响，不能要求错误秩下的结果“近似相同”。
- **特征链重组：** 使用独立合成 EEG 与双波长输入，包含低功率、非正/缺失光强、
  色团幅度不等、漂移和运动扰动。检查从原生值到 float64 特征、已知线性算子与评分
  目标的闭合；物理单位合成正例与旧近似 MBLL 行为分别报告。
- **噪声与 mask：** 检查 baseline/滤波/重采样的完整协方差、色团交叉项、输入与带噪
  target 的交叉项、秩与支持残差。分别做原生缺失和特征缺失干预，拒绝用补值增加
  有效样本数。特征噪声估计同时报告下限前偏差与下限触发率。
- **已知状态：** 沿用线性 Gaussian、非线性 Gaussian、Student-t 三种生成规律和
  独立种子，保留正确/错误配对及均值/协方差消融。报告 r、clean EEG/HbO/HbR、
  合法路径、观测恢复；MAP 不报告未估计的后验区间或边际似然。
- **适配代价：** 使用下文独立 18/6、四 panel 的合成选择/assessment 设计；每次只
  适配一个轴。沿用 `+0.02` NRMSE 非实质退化的探索容差和单侧上界检查，明确其小样本
  不确定性；不把同一 trial 的多个 mask 视为独立重复，也不增加样本直到通过。

沿用既有工程容差：目标均值重组 `1e-6` 个冻结训练 SD、雅可比 `1e-5`、隐藏干预
`1e-10`、密度单位检查 `1e-8`，以及 SVD `1e-10` 主容差与 `1e-8/1e-12` 敏感性。
这些量必须在同一声明坐标下计算；换单位时其绝对尺度和低方差判定同步变换。
新版本不能要求旧 float32 数组逐位等于新 float64 特征；旧差异保留并量化，
新实现与新算子必须自行闭合。未完成测量量纲标定时不赋绝对浓度资格。

### 把拟合效果作为数据处理验收，而非调参反馈环

后续实测第一面板沿用下文已定义的 Single-Trial 原训练身份、嵌套折、30 s 窗口和
14 模式；本次不启动它，不扩展到额外被试或数据集测量数组。
所有候选在同一训练身份上拟合变换，在外折只 apply。目标以保留的测量/特征坐标
为共同参照，不随候选的尺度、gain、R 或 Q 改变；若特征定义本身改变，则另定义
共同可观察目标，不能直接比较各自坐标的 loss。

每组后续实验须并列给出：

| 验收方面 | 必报量及解释 |
| --- | --- |
| 输入一致性 | 单位证据/转换误差、时钟误差、真实支持、量化/重组误差、Hb 对幅度关系、floor 触发率；缺失证据不靠幅值猜测补齐 |
| 完整性与成本 | 预定 trial/选择折/模式分母，准备失败、求解失败、物理失败与依赖失败；固定预算下耗时、迭代数与峰值内存 |
| 预测与残差 | 两方向隐藏目标的物理或声明相对坐标 RMSE/偏差、冻结训练 SD 的 NRMSE、HbR 残差尾部；clean-map 和同源带噪预测分列 |
| 共享性 | 同折自身上下文/任务模板/跨模态线性对照，正确配对对四类 null 的增量；完整风险要求下文固定全部身份与选择折完整 |
| 状态与参数 | 合成状态恢复、真值可用且推断支持时的校准；固定尺度后参数曲线/边界率与加权雅可比诊断，不把贴边称为可辨识 |
| 动力学代价 | 绝对转移残差及按各自 Q 标准化的残差、非零 driver 确定性回放闭合、非法路径与初态影响 |

继续沿用下文的 B 权重、层级等权聚合、至少 10% 相对风险改善、各模态最多 0.05
NRMSE 退化及配对增量要求，作为改变统计假设后的探索筛查；已知单位重表达应以
等价性为正确结果，不需要达到 10% 改善。低训练 SD 未定义分数保留为空；候选与
基线不同成功子集不排名，候选失败不以降低分母或加大 epsilon 隐去。

仅预测改善而状态恢复或配对增量变差，判为重建补偿；单位等价性通过但仍有物理/收敛
失败，转向相应求解或模型问题。只有完整诊断支持时，才把新处理版本作为后续模型
训练候选。跨数据集不强求相同 PSD、幅度直方图或随机水平的数据集身份识别率。

### 实现归属与可审阅交付

| 现有 owner | 后续最小改动 |
| --- | --- |
| `src/data/unified_physiology.py` 与现有数据 reader | 解析单位证据、真实支持与参考；保留可供各模型派生的测量坐标 |
| `src/data/homer2_preprocessing.py` / `preprocess_native_trial` | 新版本 float64 特征边界、明确非线性次序、单位化功率 floor 与可重放线性处理 |
| `fit_measured_projection` / `v3_prepare_projection` | 训练尺度与观测 loading 分离、单一目标生成、记录 fit 身份及噪声下限前后值 |
| `BalloonObservationSpec` / `TrajectoryObservationSpec` / 既有推断器 | 复用已知坐标变换、噪声因子、有效子空间与同源 target 条件预测，不新增 adapter/manager 层 |
| 既有 overnight 入口与报告 renderer | 同输入消融、失败固定分母、精度/成本/状态恢复汇报 |

先完成上述局部实现与 targeted synthetic tests，再形成新版本配置、输入 identity、
展开后的拟合次数和 synthetic pilot 成本。新 measured run 的准备与实际启动分开；
旧配置、source snapshot、缓存与失败结果保留原身份。结果仍写入既有 run root，
状态仍由 registry 持有；本设计不创建授权文件或额外控制器。

### 测量修订的首轮执行合同（2026-09-16）

本轮按用户要求执行上述软件修订及限定开发验证。新配置
`experiments/configs/physiology_semantic_tokenizer/ssm_measurement_alignment_v3.yaml`
复用 v3 控制器的 stage 1/2、既有独立合成种子和原训练身份；新处理身份为
`physiology_measurement_alignment_v3`。它先验证数值修订基线：固定原 PCA/Hb 选择、
参考 loading、R/Q 与评分规则，仅修复单位证据、精度和算子次序。独立合成门槛仍由
原 v3 合同持有；未通过或实测不完整时不得晋级跨数据集 SSM、gain/Q 或 tokenizer。
首轮不同时改变噪声下限或搜索 gain/Q。成对尺度和模型无关数值尺度的接口已可供
独立候选使用，但该接口的存在不等同完成统计假设比较。

输入使用新命名空间的 native-only 缓存，信号只在原允许窗口内预处理；旧 run、
配置、失败与缓存保留。展开任务表、pilot 成本、实际执行和科学结论分别由新 run
和 registry 持有，不在本设计维护状态副本。

### 新测量链固定模型、W 曲线与配对增量重测（2026-09-17）

本轮按用户提供的 A–E 实验文本执行，配置为
[`ssm_measurement_retest_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_measurement_retest_v1.yaml)，
复用现有 overnight 入口、持久控制器及结果根目录。它是新的探索诊断，不覆盖
9 月 11/16 日的配置、结果或资格判定。原 72 个身份、4×3 嵌套折、30 s/4 Hz、
特征层缺失和 14 模式保持不变；这不是全项目公共被试访问政策。

- A：只读复用 9 月 16 日已经合格的合成桥与已闭合的 float64 原生特征/折内投影；
  对 9 月 11 日相同身份审计幅度、HbT、baseline、边缘、时间锚点、支持和 floor。
  新增每种生成规律 24 个独立种子、五种缺失模式的 Gaussian MAP 检查；Student-t
  生成行是失配压力诊断，不是 Student-t 时序推断资格。错误单独放大 HbR 的负对照
  只用于合成，不能描述成历史 SSM 的输入。
- B/D：主推断为完整时间均值与相关噪声的 O2。两种观测假设均比较 W=0/-0.5 的
  72×14 面板，另加 HbO-only/HbR-only 两种归因模式。原统计模型的 W=0 结果及同折
  线性 basic/joint/null 直接引用已完成证据，源身份先冻结再读取。
- 新假设在训练 pooled Hb MAD 坐标中固定共同 loading `a_N=1`，不改变 EEG 驱动
  尺度或符号。为保持所有比较的目标/R/评分 SD 完全一致，实际在旧评分坐标中用
  `fnirs_gain=reference_gauge.fnirs_common` 表达它。这是相对单位的观测假设，
  不是已知单位换算、设备标定或简单字段改名。R、Q、P0/Q0、PCA 与 Hb 选择不适配。
- C：W 在原支持内取 -0.5/-0.25/0/0.25/0.5，两种映射均测 full/fNIRS-only/EEG-only。
  O1 的 rest-linearized Gaussian 边际似然单列为近似模型的参考；O2 为轨迹 MAP，
  单列观测项与状态先验项，不能当作边际似然或区间。EEG-only 的精确线性参考须
  对 W 不变。等价解以同一 trial、同一模式下 MAP 代价相差不超过 1% 定义，报告
  r 的相关性及相对幅度差；物理失败和不收敛点不参与最优点排名。
- E：只有原统计模型 W=0 的 full/两个 center 模式全部 216 行有效且 Hb clean-map
  NRMSE 均值大于 0.5，才进入适配诊断。此条件将数值失败与可解释残差分开。
  共同 gain 与 Q STD 轴分别使用既有网格，固定候选逐一报告，不作验证集选择或
  联合搜索。每个候选先在 24 个 Gaussian 独立种子上比较 full/两种 center 模式，
  每个种子内平均，r/clean Hb NRMSE 差的单侧 95% t 上界均不超过 +0.02 后，才运行
  该候选的完整实测 72×14。Q 协方差按 STD 倍率平方变化。这一有界固定候选筛查
  不代替保留 v3 的四 panel 嵌套适配资格；条件未满足时 E 明确记为未启动。

全部新拟合先做工程回归和合成 pilot，再冻结任务表/源码。最多 48 个单线程 worker，
根据物理核、cgroup quota、内存和 pilot 记录限额，8 h 硬上限；systemd 用户服务及
lingering 保证断线不终止。完整面板风险只在原定分母全部有效时定义；共同子集、
两方向 null 增量、失败分类、驱动回放及候选的状态恢复分别报告。不启动 tokenizer，
不授予 physical teacher、Student-t 校准或保护数据评估资格。


### 分频带与空间软约束测试（2026-09-17）

用户在新测量链重测后要求提交该组证据并开展所附结构方案。执行合同为
[`ssm_band_geometry_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_band_geometry_v1.yaml)，
复用 overnight 的冻结来源、任务表、持久服务和 Gaussian 轨迹 MAP；不启动新的生理参数网格。
本轮是固定候选的开发测试，不以外折结果反复筛选或修改表示。

- 复用 subject_01/09/18、session_01/03/05 的 72 个原训练身份、4 外折和 3 内折。
  保留事件前 5 s、总长 30 s、4 Hz 及原生窗口隔离，源位置 4/9 不参与。
- B_ref 保留此前的新测量宽频 PCA、目标 Hb 对、共享 Hb 缩放、R 的光学分量和全部
  生理参数/Q/初态。候选依次为局部六通道 PCA、局部多观测、三个功率之和取 log、
  alpha/beta/low-gamma 分开观测，以及固定分频带上的真实/置换几何软约束。
  几何从 v4 的公开 montage metadata 读取，信号仍经原生 trial 窗口；不混入全记录滤波缓存。
- 本轮载荷估计明确采用**训练宽频 PCA 代理驱动的有符号回归**，不是神经驱动真值，
  也不是缺失目标端到端学习或联合 EM。基础 ridge=0.01；真实/置换几何额外惩罚
  偏离每频带几何方向的分量，三个强度 0/0.1/1 只由 3 个内折的每维平均载荷重建
  MSE 选择。内折重新拟合缩放、PCA、Hb 锚点和邻域；并列时保留零惩罚。
  缺失预测训练、多区域状态和学习型初始化不属于本轮。
- 多观测保留完整通道/频带协方差，训练差分 MAD 边际尺度加固定 0.1 相关矩阵收缩。
  每维合成噪声 floor 与 B_ref 同源，明确是特征近似而非测得传感器噪声。
  对固定载荷，Gaussian 充分统计量严格保留关于同一 r 的目标差与梯度；所有时间均值、
  噪声和同源目标交叉项继续经原观测算子。此运算不删除候选的多维载荷/协方差证据。
- 合成先运行 8 情形 × 4 独立 panel，每 panel 18 训练 + 6 评价；六必需模式及
  整段 fNIRS 缺失的错配/15 s 移位。共享 Gaussian 的 24 个完整模式必须全部成功且
  r 平均 NRMSE≤0.65、相关≥0.8，才读取该候选的开发面板。压力情形独立报告。
  最终保留还需四 panel 的六状态/clean Hb 恢复 +0.02 NRMSE 非劣结果；不将入口通过当作资格。
- 开发面板六模式共同固定，另保留 fNIRS 中心自身上下文/模板/错配/移位和整模态
  模板/错配/移位。缺失在非线性特征后、插值及滤波基线之前施加。
  整模态缺失时不输入真实 fNIRS baseline。EEG 反向结果使用固定 B_ref 审计坐标，
  各候选原生 EEG 训练误差不跨维数排名。
- 同特征滞后 ridge 对照在内折选择 lag 集合 [0]、[0,2,4]、[0,2,4,6] s 及
  ridge 0.01/0.1/1，选择目标为两个缺失模式等权的 Hb 归一 MSE；内折特征也独立拟合。
  自身插值、训练时序模板及其组合分别保留，错配 donor 仅来自同被试同 session 训练 MA。
- 主端点为中心 fNIRS 和整段 fNIRS 缺失风险各占一半，再对 HbO/HbR 标准化 MSE
  等权；trial→session→subject 等权汇总。完整风险固定 72 身份×2 模式，缺失时未定义，
  同时显示共同成功子集和失败计数。10% 风险改善、单色团退化≤0.05、至少两被试改善、
  正确配对优势、合成恢复与全模式完整性共同决定是否保留。

整段 fNIRS 缺失时，EEG 训练模板平均 N 个同 session 训练 trial，均值与噪声必须
同步传播：噪声因子除以 √N。技术修正使用现有入口的
`--prepare --template-correction-from <completed original run>`，建立新的独立 run，
仅补跑 7 候选 × 72 身份的 `EEG_only_template`，绑定原训练数组和配置并继承原八小时
预算时钟。原始逐拟合证据保持只读；修正不改变主端点、超参数选择、其余 mask 或生理参数。
报告通过显式 `--template-correction <completed correction run>` 逐身份引用修正记录。


## 观测合同修复后的下一轮实验设计（2026-09-10）

本节依据用户提供的 N1–N7 复核指引，以及提交
`fa972804f0fc453a47933a7db6305a7d8196462f` 的代码与保留结果制定。
本节只规划后续工作；实现、运行和科学结论仍由各自的代码、run 记录和
`research_state/registry.json` 持有，不因计划写入而发生状态迁移。
N1–N7 的冻结配置、逐 trial 证据和旧判定保留。

### 中心问题、优先级与边界

中心问题是：**在六状态共享 SSM 不变时，使输入、观测方程和评价目标具有一致的
时间、尺度与误差定义，能否同时改善观测预测和正确配对增量？**

依据为[原始运行报告](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/OVERNIGHT_REPORT.md)、
[候选表](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/candidate_table.csv)
和[阅读版分析](../experiments/runs/physiology_semantic_tokenizer/ssm_overnight/20260910_overnight_n7_v1/analysis_20260910_v2/REPORT.md)。
N2 支持优先检查时间处理；N5 的共同有效子集风险改善约 26.30%，支持保留一个
观测增益假说；N6 的约 12.99% 改善需检验是否来自状态转移残差补偿。
这些历史比例只用于提出假说，不能作为新实验的预期效应或跨候选排名。

固定六个状态及其生理方程，主比较固定 `G=W=0`，Z 保持参考阻尼比
`kappa_ref/(2*sqrt(gamma_ref))`（Z 不是以零为参考的坐标）、原始其他生理参数、
初态先验和过程噪声。先完成观测修复，再分别检验一个共同 fNIRS 增益和受限的
过程噪声调整。不扩大 G/W 网格、增益上界、EEG 频带/空间搜索或数据面板；
全输入目标函数贴边只作诊断，不作为选参依据。

Gaussian 时序推断沿用夜间套件的独立探索定位。它可以在自己的特征合同及合成
检查完成后开展受限实测诊断；不能满足下文旧 Step5 repair v2 明确要求的
非线性 Student-t 时序资格。MAP 路径不能称为后验均值；其区间、边际似然和
实测覆盖率继续标为 `NOT_ESTIMATED`，不借用 O0/O1 的方差。
本轮不作 teacher、全面 UQ 或 tokenizer 晋级。

### 固定数据与对照设计

| 项目 | 下一轮设计 |
| --- | --- |
| 数据身份 | Single-Trial MA，subject_01/09/18 × session_01/03/05 × 每 session 8 个原训练 trial，共 72 个唯一身份 |
| 排除范围 | 原 MA 位置 4、9（代码中的零基位置）在切片和预处理前排除；subjects 19–29 的信号不进入本轮 |
| 时间与目标 | 保留事件相对 `[-5,+25) s`、4 Hz 评价网格、120 点、前 5 s 基线；中心缺失为 4 s / 16 点；处理前时钟逐模态保存 |
| 外折 | 每被试 4 折，沿用训练 ordinal 模 4；每 session 6 个训练、2 个评价，三 session 合计 18/6；共 12 个选择折 |
| 内折 | 每外折的 18 个训练 trial 内再分 3 折，每 session 4/2，三 session 合计 12/6；外折目标不参与任何选择 |
| 折内拟合 | EEG PCA、通道/光学 pair、gauge、噪声估计、模板、ridge 参数和增益/过程噪声选择只用对应训练折；各 solver 共用冻结对象 |
| 外折模式 | 沿用 `OUTER_MASKS` 的 14 种：full、两种单模态、all-missing，以及 EEG/fNIRS 中心缺失各自的 joint、own、template、pairing、shift |
| 线性对照 | 复用同折 own-context/task-template ridge 与含另一模态的 ridge，沿用 lag/alpha 支持；它们与 SSM 共用目标、尺度、可见支持和 donor 身份 |

仍由 `load_training_subject` 和现有 identity/scope validator 负责原生入口。
文件按 session 存储与实际获准切片的 trial 范围分开记录，不新增直接解析器。
三名被试的重复使用只提供开发诊断；不称为独立确认或人群泛化。

### 阶段 0：先把观测层和评价口径变成可检验合同

在现有预处理函数中暴露中间值，明确：

`原生输入 → 非线性特征处理 → 特征观测层 → 已知线性时间处理 → 评分目标`。

EEG 的带通后功率及 log 属于特征构造；fNIRS 的 OD 和数据依赖运动处理也在
这条分界之前。当前 fNIRS 实现先做时间滤波、后做固定 MBLL 转换；若将浓度层
提前暴露，必须先用 fixture 证明固定线性转换重排的数值等价，不直接改流程顺序。
PCA 中心项作为已知仿射偏置处理，训练 gauge 只应用一次。
同时核对波长配对、HbO/HbR 顺序、符号、共同缩放及 `HbT=HbO+HbR`。

每个视图显式保存输入/输出时钟、单位、基线权重、滤波端点、重采样顺序、
可见输入/输出 mask、算子及训练噪声来源。在确实线性的特征层使用

\[
y_i=A_M h(x)+b_M+\epsilon_i,\qquad
R_{ii}=A_M R_{\mathrm{pre}} A_M^\top.
\]

`R_pre` 定义在该特征层，不能把处理后差分估计的噪声冒充处理前白噪声。
优先保留固定估计规则和最少参数，不搜索任意协方差；但原生重采样或已知处理
带来的相关项必须传播。现有 O1/O2 只接受三个处理前方差，接口暴露本身不能
证明这些方差足够；发现必须保留的块相关时，在现有推断 owner 内局部支持它。
剩余的特征噪声 Gaussian/独立假设明确标为近似，并用合成压力与训练残差检查。

**遮挡位置是本阶段的必决项。**代码中的原生插值后功率/运动处理一般不能写成
对完整特征的同一个线性 `A_M`。因此首个可执行时间机制实验采用明确标识的
**特征层缺失**：隐藏值在后续插值、滤波和基线处理前移除，各 solver 接收完全
相同的新输入；它检验特征缺失恢复。完整特征目标须与原 helper 数值对齐。
这个实验不代表原生传感器缺失恢复，不能把它的误差直接接到旧 N1/N5 的趋势上。
原生预处理前遮挡保留为单列的旧合同对照和隐藏值干预检查；只有建立其自身的
非线性输入/目标关系后，才可声称时序修复改善了原生缺失恢复。
若某模态不能建立声明的特征边界，相关比较记录缺口，不用处理后的 NaN 冒充
已经验证的原生遮挡。

对同源带噪评分目标 `y_t=B h(x)+b_t+epsilon_t`，同时保存
`R_ti=B R_pre A_M^T` 和 `R_tt=B R_pre B^T`。
Gaussian 点预测使用对应的条件噪声修正；O2 可给定 MAP 路径作条件点预测，
但不将其标为积分后的后验预测均值。另存 `B h(x)+b_t`，用于完整输入的
clean-map/观测残差诊断。没有共同特征噪声定义的原生视图不套用上述公式。
评分器固定消费哪一种输出，避免把噪声条件预测和 clean-map 残差混作一项。

软件验收使用公开数值或临时 fixture：完整输入重组一致；64/120 点上下文保留；
全缺失没有伪观测；固定折中验证/评价 trial 的隐藏特征扰动不改变可见输入、
拟合对象、选参及预测；原生隐藏扰动不影响其单列遮挡视图；已知尺度协变、
解析导数和非法路径拒绝均成立。
沿用工程阈值：线性均值 `1e-6`、导数 `1e-5`、隐藏干预 `1e-10`、密度单位
`1e-8`；SVD 主容差 `1e-10`，检查 `1e-8/1e-12` 的秩与结果敏感性。
不通过时定位该接口，继续与其无关的统计/合成准备。

评价器同时修正两点：所有 trial 级指标及 null 变化统一采用
trial → session 等权 → subject 等权；合成固定配置压力测试与适配算法测试
分列。临时 fixture 必须覆盖 session 有效数不等、null 单独失败、共同基线失败
和 train/test 干预，不能只检查全成功面板。

### 阶段 1：用同一推断器分离时间均值和噪声传播的作用

合成主面板为三种生成规律（线性 Gaussian、非线性 Gaussian、非线性 Student-t），
每种 24 个独立 trial、120 点；64 点只保留已有短窗回归。每个 trial 的原坐标与
组合处理共用同一 clean 状态和噪声 realization；每个条件跑 full、center EEG、
center fNIRS、whole EEG、whole fNIRS。固定 W=0、增益 1、原始过程噪声。

对每份输入比较 O0、O1、O2，并在 **同一个 Gaussian 非线性 MAP** 中增加两个
消融：逐点均值/对角噪声、正确时间均值/对角噪声。完整 O2 使用正确时间均值和
完整相关噪声；两个消融共用处理后边际噪声尺度。这样可以区分均值算子和非对角
噪声的贡献；O0 与 O2 的差仍同时包含噪声族、推断近似的变化，不能作纯单因素解释。
消融只在时钟/支持明确的合成坐标实施，不从故意失配的密度比较生理参数。

独立真值的主要指标是 `r` 与 clean EEG/HbO/HbR NRMSE、相关、HbR bias，
分别报告全时间、隐藏时间及处理后的可见 clean 目标。补充求解失败、合法路径、
秩敏感性、同源带噪条件预测与运行成本；O1/O0 已实现的区间按各自生成规律
报告覆盖与宽度，O2 区间保持未估计。坏配对压力必须同时列出误差和区间变化。

进入受限实测探索的工程预检保留 N2 的均值门槛：非线性 Gaussian、W=0、full
原坐标的 24/24 完成，四个真值平均 NRMSE 均不超过 0.65，`r` 平均相关不低于
0.80；此外中心/整模态缺失均须通过其接口、支持和隐藏干预检查。
组合处理相对逐点 Gaussian 消融的收益单独报告，不把没有收益改写为工程成功。
Student-t 压力失败限制适用范围，不使 Gaussian 结果继承 Student-t 资格。

### 阶段 2：固定参数，在同一 72-trial 面板比较观测修复

预先指定 O2 为合法路径的主候选，O1 为较简单的参考，O0 为旧推断参考。
三者在阶段 0 定义的新特征缺失合同下，共用每折输入、目标、projection、
噪声来源、mask 和对照；不根据外折误差挑 solver。原 native-mask O0 的保留结果
只标为旧合同背景。主参数始终固定 W=0、gain=1、原过程噪声。

每个 solver 按固定身份完成 72 × 14 个模式拟合，线性对照同折准备。
首先报告两个方向的中心缺失预测、完整输入 EEG/HbO/HbR 残差和正确配对增量，
再看总风险。特别检查 HbR 的训练尺度、残差 bias、p50/p90/p95 和白化残差
相关；p50 从小分母造成的放大与绝对坐标偏差分别展示。

复核旧失败身份时同时保存旧/新输入身份。只有数组、mask 和配置相同才称
“同输入失败复现”；换了特征合同只能称“同 trial 的新合同比较”。
全输入 W=0/−0.5 的目标差可作边界压力旁证，但不选 W；O2 的 MAP 目标值
不是边际似然，也不与 O0/O1 的似然数值直接相减。

### 阶段 3：检验从训练数据适配一个观测增益

只用阶段 2 预先指定并完成自身检查的 O2，固定 G/W/Z、噪声估计和过程参数。
候选沿用 `a_N ∈ {0.5,0.75,1,1.5,2}`，共同乘 HbO/HbR 的均值映射；它不是
坐标 gauge，也不同时改变噪声。禁止每次候选重新标准化，把 gain 吸收进尺度。
`a_N=1` 是新合同下同 solver 的基线，不依赖旧 O0 的成功身份。

先在合成训练集执行完整的折内拟合/选参，再评价独立生成的合成 trial：

| 真值条件 | 要回答的问题 |
| --- | --- |
| Gaussian，a=1，G=W=0 | 何时应保持基线；错误适配频率及相对固定 a=1 的代价 |
| Gaussian，a=0.75 或 1.5 | 适配方向与 held-out 恢复收益；与固定 a=1、已知真值增益 oracle 比较 |
| Gaussian，a=1，G=±0.3 | 观测增益是否代偿生理增益；不能据此宣布生理参数可辨识 |
| Gaussian，a=1，W=±0.25 | 时间变化是否被错误地吸收到幅度标定 |
| Gaussian，a=1，(G,W)=(-0.3,-0.25)/(0.3,0.25) | 联合生理变化下的适配局限，不开展 G/W 拟合 |
| Student-t，a=1；破坏配对，a=1 | 尾部/共享关系失配下的风险、错误自信和适配代价 |

共 11 个条件；每条件先固定 4 个独立合成 panel，每 panel 模拟一个被试的
三 session，每 session 6 个训练、2 个 assessment trial，即 18/6，内折仍为
12/6。一个 panel 对应一次独立外折实验，不循环使用 assessment 进行开发。
训练/assessment 的完整潜在路径、初态、过程噪声增量及观测噪声独立；条件间可共用
随机数作配对，但不增加独立样本数。4 panel × 24 trial = 每条件 96 个唯一
trial；不把候选、mask、重采样或选择频率计作独立重复。
开发用种子与这些 panel 分离，启动前在配置中固定种子映射。
破坏配对条件以 EEG 对应 driver 为 r 真值，Hb clean 真值绑定其 donor，
不假装错误配对的两模态仍有一条共同真值路径。

该预算是机制筛查。按 panel 计算配对差和 Monte Carlo 不确定性；4 个 panel
不能支撑强校准结论。新设计的实际退化容差预设为真值 NRMSE `+0.02`，同时
报告旧 `0` 容差的敏感性。按 panel 的配对差计算近似单侧 95% t 上界（df=3），
并列出全部四个 panel 点；上界未排除 `+0.02`、或该小样本近似不可靠时均为
“不确定”，不得以均值略低于容差宣布无害，也不临时追加样本直到通过。
这一新容差是后续诊断的设计选择，不修改旧 synthetic screen 的负结果。

合成结果没有显示明确的实质退化、接口检查完整后，再计划实测 12 个内折选择
与 72 个外折评价。每个候选的 36 个内折中心缺失拟合须完整；每模态 NRMSE
相对新基线不得退化超过 0.05，在可接受集合中最小化下述 B。
风险并列先选 a=1，再选较小 `|log(a)|`，最后用固定数值顺序。
外折只 apply 内折选出的规则。报告边界选择率、HbR 完整残差和配对增量；
重复选到 2 不触发扩大上界。合成结论不确定时，实测结果也只能定位机制，
不能把它列为已验证的共享 teacher 改进。

### 阶段 4：检查状态转移残差与合法路径，增益不与过程噪声联合搜索

沿用阶段 2 的修复合同、固定 a=1。只比较血流相关五状态过程噪声**标准差**
倍率 `{0.5,1,2}`，`r` 过程噪声和初态先验固定；方差倍率为 `{0.25,1,4}`。
使用相同的内外折选择规则、独立合成适配检查和 14 模式外折评价，不做
gain × Q 的网格。该阶段以自身 q=1 基线为参照，不能跨阶段成功子集排名。

合成沿用阶段 3 的 18/6 独立 panel 结构，每条件固定 4 个 panel；真值 Q 标准差
倍率设为 0.5/1/2，另用 q=1 的 Student-t 和错误配对条件，共五条件。比较
固定 q=1、折内所选 q 和真值 q oracle，并沿用 `+0.02` 真值退化检查。
在同一新特征缺失输入上，为 O0 复算这个三点 Q 诊断，才能比较观测修复前后
所需倍率；不能把新缺失合同的选择率直接与旧 N6 选择率相减。

分开记录确定性转移域退出、filter 更新后非法、后验均值物理检查、MAP 无合法
收敛解、数值失败和 null/选择依赖失败。允许优化器拒绝非法试探步并回溯，但
规则和尝试预算预先固定；不裁剪 flow、不跳过最后观测、不删除困难 trial。
旧四次基线内折失败及保留的五个越界案例依 owner 表按身份定位，不按新排名
重新挑选“典型案例”。

比较修复前后所需的 Q 倍率、各状态转移残差的绝对大小及按各自 Q 标准化的大小，
防止仅因分母增大就声称状态转移残差减小。复用 `driver_replay`，对实测和匹配/失配
合成都计算同样的回放差、失败率和初态处理；先用无过程噪声的确定性合成检查
回放闭合。MAP 路径与后验均值驱动回放分列，后者包含非线性均值效应。
回放差只与相应合成分布比较，不解释成私有信息比例。

若时间修复后 q=1 已稳定、增大 Q 无额外配对收益，保留原 Q；若较大 Q 仍改善
观测预测但损害合成 r 或配对增量，只支持重建补偿。只有在合成状态恢复与
实测共享性同时支持时，才讨论下一版过程模型假说；本轮不增加状态或生理自由度。

### 统一评分、失败分母和结果判读

令 `E_m` 为每 trial 中心隐藏目标的 MSE 除以对应训练折方差，主风险为

\[
B=0.5E_{EEG}+0.25E_{HbO}+0.25E_{HbR},\qquad
\Delta_{null,m}=E_{null,m}-E_{joint,m}.
\]

先在每 trial 内得到数值，再 session 等权、subject 等权；聚合模态 NRMSE
取聚合 NMSE 的平方根。候选与基线的差在完全相同身份上计算，再按同一层级
聚合；残差分位数则先在 trial 内求分位数。训练 SD 非有限或 ≤`1e-8` 时记
`low_training_variance` 和未定义分数，不用 epsilon 造出有限风险。
每个量写明目标、mask、权重和单位。

本轮 SSM `template` 延续 N1 的“另一模态替换为同 session 训练平均”，
不同于 ridge 的“自身上下文＋目标模板”；两个对照名称和列分开。
`pairing` donor 只取同 session 的训练 trial，`shift` 沿用半 trial 循环移位，
各 solver 共享 donor/位移与评分支持。null 增量统一为训练方差归一化的 NMSE
差，不能直接串联旧 Step5B 的其他单位结果。

每个规则同时报告：12 个选择折的完整性；72 个正确 joint 预测的分方向成功率；
72 个身份的每模式/每 null 状态；14 模式完整 trial 数；共同完整子集风险及
各自失败原因。完整主风险要求 72 个身份和（适配时）12 个选择折都完整。
任一 null 失败不抹除该 trial 已有的 joint 结果，但后者只进入明确标记的单模式
描述表。完整风险无法计算就留空，不把成功子集外推为全体。

旧 O0 不完整不使新 solver 的固定参数实验或自身适配规则自动失败。
若新 solver 的自身基线内折仍不完整，则该选择折未定义；不静默回退到另一
solver/更大 Q，也不删除失败后重选。失败率本身按固定分母比较。

后续优先验证的筛查沿用至少 10% 的相对 B 改善、每模态 NRMSE 退化不超过
0.05、至少 2/3 被试改善，并要求四类 null 增量按统一权重均不降低；另列各
方向正确配对增量是否为正。均值降低但配对无增益只称观测重建改善。
不完整分母或合成不确定不支持完整规则合格；明确退化则保留负结果。
这些是探索性的继续研究条件，不是 teacher admission。

### 执行顺序、预算与交付

| 顺序 | 交付与完成条件 |
| --- | --- |
| 0 | 一个特征/噪声/mask 合同；目标重组、隐藏干预和不等分母统计检查；明确当前仍未建模的原生非线性部分 |
| 1 | 同一合成 realization 的均值/协方差消融表、全模式失败表；结论区分工程正确性、状态恢复和噪声族适用性 |
| 2 | 固定参数 72-trial 的两方向风险、HbR 残差、null 增量和完整分母；解决时间修复是否有效 |
| 3 | 独立合成训练→assessment 的增益适配结果，然后同折实测增益诊断；解决 N5 的收益能否在新合同下保留 |
| 4 | Q 单因素及匹配合成回放；解决下游创新是否仍在补偿失配 |
| 条件性后续 | 只有完整诊断支持继续时，另行规划三 session 留一整 session 泛化；所有拟合仅用另两 session，仍只用原训练身份。当前核心任务表不含该扩展 |

实现时复用现有 owner：`preprocess_native_trial` / `homer2_preprocessing.py` 的
特征拆分，`TrajectoryObservationSpec` 与 O1/O2 的算子/推断，夜间入口的
`equal_subject_mean`、选择器、任务表和单一有界调度器，以及已有报告 renderer。
在 `experiments/configs/physiology_semantic_tokenizer/` 新增版本化
`ssm_overnight_v3.yaml`，不原地修改 v1/v2 合同；
新 run 使用既有 `experiments/runs/physiology_semantic_tokenizer/ssm_overnight/`
下的新身份，拒绝覆盖旧证据。不新增 manager、授权布尔值或第二份状态清单。

未来启动前，先用 synthetic pilot 得到每 solver 的耗时/峰值内存并展开实际
任务数与拟合次数；阶段 1 的五种推断/消融最多 3,600 次拟合，阶段 2 的三个
solver 共 3,024 次模式拟合，均另计线性对照和工程检查。
阶段 3 按 11 × 4 个 panel 展开：五候选的内折中心拟合为 7,920 次，所选规则
外部 assessment 的 14 模式为 3,696 次，固定基线/oracle 及未缓存拟合另列。
不以 task cell 数充当实际求解数。

拟采用整个核心运行合计 8 小时、最多 16 workers、可用内存 60% 的上限，沿用
普通 cell 900 s、MAP cell 1800 s、每个 start 最多 200 次评估；pilot、固定
多起点和重试均计入总预算。先完成阶段 0–2，再按依赖推进 3、4；预算预计不能
容纳完整下一阶段时不启动其一部分来做排名，记录未启动并报告所需追加预算。
这些是待实现和实测授权时确认的计划上限，不继承旧后台运行的授权或资源状态。

旧 v1/v2 的 `--prepare` 包含原生读取和实测 pilot，不能当作无数据 dry-run。
v3 的 `--prepare` 只做无信号身份清单、软件检查和分离种子的 synthetic pilot；
冻结配置、任务表与 source snapshot 后由同一调度器推进。
实测准备依赖阶段 1 的明确资格结果；启动仍须满足用户级授权边界。
后续实现与执行记录由 `research_state/registry.json` 及其所链接的 run 持有。

每阶段产物沿用 run 内 resolved config、固定 task/status 表和逐 trial 结果。
报告从这些表派生，至少包括风险/失败分母、残差尾部、配对增量、增益选择与
合成 oracle 差、状态转移残差及回放图。每完成一组实验，都在对话中主动报告关键
数值、基线比较、结论、失败/不确定性和下一步；日志或文件链接不替代该汇报。

## Bounded overnight SSM diagnostics (retained v2 contract)

The September 9 overnight implementation is a separate, decision-ineligible
exploration with executable contract
[`ssm_overnight_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml)
and entry [`evaluate_ssm_overnight_diagnostics.py`](../experiments/evaluate_ssm_overnight_diagnostics.py).
The v1/v2 YAML `plan` field preserves the name of the removed short-term note;
this section is the maintained reader entry. The frozen configurations and saved
source snapshots retain their original identities.
It retains the preceding experiment's negative results and does not satisfy its
nonlinear Student-t measured-comparison prerequisite. Its O2 branch explicitly
uses correlated Gaussian noise and a nonlinear six-state batch MAP; the native
measured O2 branch remains unavailable unless the native feature/noise boundary
is exposed. Other families retain the existing pointwise inference as an
exploratory baseline and run independently of O2. The v2 contract adds N7: independent G/W fitting with W-only and G-only inner-fold controls, while observation gain, Z, tau and the other physiological/noise settings remain fixed. The earlier v1 configuration and prepared run remain retained.

The suite's training-only native entry is the existing observation-diagnostic
loader, with optional retention of admitted EEG/EOG windows for fold-fitted
artifact regression. Subject/session/trial checks precede slicing and
preprocessing. The versioned contract owns its fixed candidate panels, nested
folds, seeds, budgets and failure policy. `--prepare` performs software checks,
scope verification, fit-fold preparation and family pilots before writing the
single task table. `--freeze` captures source and input identities. `--run`
requires that frozen source and uses one bounded worker pool. Each fit is saved
atomically; final reporting joins the fixed task table, including unavailable,
failed, timed-out and unstarted cells. No protected unlock or tokenizer stage is
part of this suite.

## Step5 flow-domain and mask-specific observation experiment

The September 9 follow-up is owned by
[`step5_observation_repair_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5_observation_repair_v2.yaml)
and the existing [`evaluate_step5_observation_repair.py`](../experiments/evaluate_step5_observation_repair.py)
entry. It closes oxygen-extraction investigation and retains known-scale
invariance as regression only. It does not change the physiological drift,
Student-t noise, state priors, measurement gain or W support.

Before each deterministic transition, the dynamics owner checks whether flow
remains positive throughout the interval. A sufficient matrix-exponential
displacement bound handles ordinary states; otherwise analytic extrema
isolation and bracketed matrix-exponential roots locate the minimum and first
zero. This includes crossings followed by recovery before the next sample.
The process-noise increment remains after the deterministic transition. A `flow_domain_exit`
retains the last four joint-filter observation updates: predicted and filtered
means/covariances and modality visibility. No failed case is projected, removed,
redrawn, or rescued by changing substeps. Only the previously recorded training
indices subject_01:4/7 and subject_09:12 are replayed, in fNIRS-only coordinates
at fixed W=0/−0.5 using the original prepared inputs and training noise scales.
The existing training-identity validator owns access; native files and original
held-out trials are not processed.

The controlled feature bridge compiles `S_out P I_M S_in`: visible-input
selection, linear interpolation on the input clock with constant endpoint
extension, the existing processing sequence, then explicit output selection.
Full missing modalities emit no observations; partially observed modalities
need two visible inputs. Hidden center interpolants are not observations. The
same compiled matrix transforms the model mean and complete noise covariance.
The v1 conservative operator remains available for reproducibility. Regression
must independently match the actual visible-interpolation pipeline, preserve
context at 64/120 time points, and make hidden-value interventions irrelevant.
This is a linear feature-coordinate result, not validation of native EEG power,
optical conversion or motion suppression.

Use 24 independent trials per generating law (linearized Gaussian and nonlinear
Student-t), 64 time points at 4 Hz, full/center EEG/center fNIRS/whole EEG/whole
fNIRS masks, unprocessed and combined processing, and fixed W=0/−0.5. The old
pointwise Student-t filter and mask-specific Gaussian temporal reference receive
identical processed observations. Report canonical all-time and hidden-time
clean recovery, processed visible clean recovery, HbR bias, domain exits and W
likelihood differences. Count independent trials separately from fits and masks;
retain all failures and numerical rank sensitivity. Diagnostic coverage bounds
are not new teacher admission criteria. In particular, a Gaussian path reference
on Student-t data cannot inherit nonlinear A0 calibration.

The new measured same-fold linear/pairing/shift comparison is conditional on
validated nonlinear Student-t temporal inference under this mask contract. The
current Gaussian reference cannot fulfill that prerequisite, regardless of
synthetic coverage. This version therefore executes the bounded failure replay
and synthetic comparison; its measured-comparison entry rejects before reading
data. The later comparison retains the original subjects 01/09/18, sessions
01/03/05, training-only outer folds, fixed W and same-fold controls. Protected
24–29, all subjects 19–29, and original trial positions 4/9 remain closed.

## Step5 observation contract repair and regression (retained v1)

The September 8 review is implemented by
[`step5_observation_repair_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5_observation_repair_v1.yaml)
and [`evaluate_step5_observation_repair.py`](../experiments/evaluate_step5_observation_repair.py).
This version changes the existing numerical/observation owners locally; the
previous diagnostic configuration, snapshots, data identities and failed outcomes
remain immutable. The six physiological states, GWZ support, driver gauge and
tokenizer target are unchanged. Results cannot grant teacher qualification.

P0 validates the oxygen-extraction domain with finite negative `log(1-E)`,
including finite positive flows for which binary64 rounds E to one. The flux
and derivatives retain stable evaluation, including a high-flow series where
subtraction would cancel. Nonfinite/illegal inputs and genuine integration
failure still raise; no clipping, redraw or default state is permitted. Replays
record saturation separately from extreme flow (descriptive f<0.01). Counts of
RK4 evaluations, curve tasks, and unique trials must not be interchanged.

P1 defines a known invertible diagonal coordinate transform in the observation
specification. Canonical means, Jacobians and noise transform together; the
Student-t density includes the absolute scale normalization only for visible
coordinates. Latent states and P0/Q0 retain canonical model units. This known
transform is separate from both the old training-MAD gauge and an unknown
measurement gain; the existing measured `fnirs_factor` is not multiplied into
the model map again. The three branches are original, deliberately observation-
only scaling, and synchronized scaling. Use the original 24 bridge seeds for
regression, then 24 new independent seeds, with the unchanged 17-point W support.
Check fixed-W and parameter-mixture state moments, canonical clean moments,
posterior CDFs and density Jacobians. These are invariance checks, not evidence
of parameter identifiability or measured gain recovery.

P2 adds a **short-window reference**, not a production Student-t solver. Its
explicit finite-window operator records processing order, baseline weights,
filter/resampling boundaries, clocks, units and masks. Mean and full temporal
noise covariance transform jointly. Missing inputs invalidate every output
depending on them. The reference linearizes the six-state dynamics and
observation at rest and uses independent Gaussian noise *before* processing,
with the old Student-t marginal variance. Thus it is exact only for that new
linearized Gaussian contract; applying it to nonlinear Student-t bridge data is
an explicitly labeled moment approximation. It cannot inherit A0 calibration.

Compare baseline-only and filter-only before the combined order (4→10→4
resampling, filter, baseline, known scale), on 64-sample windows at 4 Hz. Each
law has 24 trials, W=0 truth, and only fixed W=0/−0.5 evaluations. Retain canonical
state/clean truth and processed-clean truth separately. SVD handles singular
baseline covariance and ill-conditioned filter directions in the operator's
observable subspace; report retained rank, discarded singular values, support
residual and sensitivity to the declared rank tolerance. Information discarded
by irreversible or numerically truncated processing is not recovered by a
Jacobian correction. Gaussian-law coverage is separate from the nonlinear
Student-t approximation discrepancy; neither is a new teacher admission.

P3 uses the same subjects 01/09/18, sessions 01/03/05 and 72 original training
trials as the preceding diagnostic. The existing training-only loader and
metadata validator remain the sole native entry. Trial positions 4/9 are excluded
before slicing and preprocessing; subjects 19–29 remain outside the scope.
Numerical continuation reads the identical prepared training arrays, records
input/identity hashes and retains per-trial failures. Identical fNIRS-only curves
shared by the two EEG coordinates are computed once and explicitly aliased;
EEG-only W is checked as an invariant and never reported as an estimate.

The measured comparison fixes all dynamics at W=0, uses broadband PCA as already
declared, and adds no measurement-gain candidate in this version. Fixed SSM and
the existing nested ridge control share the same four outer folds, projections,
raw-input masks, targets, normalizers and pairing/shift donors. Report normalized
negative MSE and paired increments over own-context SSM, the training task-
template/context linear baseline, independent pairing and half-trial shift.
Noise estimates use outer-training inputs only. Missing cases remain failures;
three-subject intervals are descriptive. The temporal reference is not installed
into measured pointwise inference until its approximation is adequately
validated. Continued measured bias or negative pairing increments therefore
remain open observation/shared-information failures, regardless of numerical
repair success. No wider campaign, comprehensive UQ or tokenizer promotion is
included in this request.

## Step5 observation adaptation diagnostic (retained v1 contract)

The requested follow-up to the September 7 Step5 results is owned by
[`step5_observation_diagnostic_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml)
and [`evaluate_step5_observation_diagnostic.py`](../experiments/evaluate_step5_observation_diagnostic.py).
It retains the six-state model, GWZ support and Step5B negative decision. The
user's request is for this bounded observation diagnostic; no prior run status
or historical authorization is used to open a new qualification campaign.

The measured panel fixes subjects 01/09/18 (first/middle/last of the Step5
development inventory) before its results. It uses only the original eight
training MA trials in each of sessions 01/03/05. Original trial positions 4/9
are excluded before slicing and processing. Native files contain full sessions;
that storage fact does not make their other trials diagnostic inputs. The
existing metadata validator verifies record/event/clock contracts before native
access. Subjects 19–29 are outside the loader's scope. Old Step5 code, frozen
configurations, completed runs and failures remain unchanged.

The synthetic bridge uses 24 independent matched-model trials and 24 independent
noise-estimation trials, with truth W=0. It compares original model coordinates,
baseline subtraction, the fNIRS filter operator, a controlled 4→10→4 polyphase
resampling round trip, a common HbO/HbR amplitude factor, training noise-scale
estimation, and their combination. The filter uses the existing 0.01–0.2 Hz
third-order implementation on the controlled 4 Hz coordinate. This isolates
operators; it does not claim to simulate native raw EEG, optical motion/MBLL,
or the full native 10 Hz filter distribution. Both W=0 and W=−0.5 are fixed
diagnostic settings. No parameters are fitted. Known driver/model-clean truth
and processed-clean truth are reported separately; nominal intervals retain
the existing observation model, deliberately exposing operator mismatch.

Training W curves cover the entire original support at 17 points, separately
refitting EEG-only, fNIRS-only and joint filters. They are likelihood curves,
not resolved parameter posteriors. Chronological joint-density increments are
summed over baseline, task and nominal recovery without resetting at segment
boundaries; marginal predictive scores do not replace the joint likelihood.
Fixed-setting one-step predictive residuals report bias, within-trial autocorrelation and PSD.
The first original training trial provides predeclared 13/17-order endpoint
checks. EEG-only W invariance is also checked against the likelihood owner.

The only alternative EEG coordinate is positive F3 8–13 Hz log block power,
with a training-fitted scale in the same reference gauge. It is a prespecified
left-frontal diagnostic, not validated anatomical correspondence to whichever
fNIRS pair is selected. There is no post-result channel/band/sign search.

The low-capacity control predicts a center-masked modality from visible own
endpoints and a same-session training task template, then adds six fixed lags
of the other modality. It uses four outer and three inner folds within the
original training inventory; every inner/outer fit repeats pair selection,
projection and scaling. Inner trial folds choose ridge strength. EEG→fNIRS
uses preceding EEG; fNIRS→EEG uses later fNIRS and is explicitly offline.
Independent-training-trial pairing and a half-trial circular shift change only
the other modality at validation. Template construction excludes the example's
own trial. Scores are negative MSE in outer-training variance units, with
subject-cluster summaries. Three clusters support descriptive localization,
not a new confirmation or teacher admission claim. Missing/failing registered
cases are retained, never replaced. Comprehensive UQ and tokenizer promotion
remain outside this diagnostic.

## Step5A0 inference consistency diagnostic

[`step5a_inference_consistency_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5a_inference_consistency_v1.yaml)
owns the small synthetic localization panel described in this section.
[`evaluate_step5a_inference_consistency.py`](../experiments/evaluate_step5a_inference_consistency.py)
implements it without modifying Step 1–4 code, configurations, evidence, or
negative decisions. This is a diagnostic before Step5A1 teacher checks, not a
new qualification gate or a measured-data campaign.

The primary endpoints are forward/derivative equivalence and inference
consistency. G and W have truncated-normal priors in relative log gain and
log frequency; Z has a uniform prior in physical damping ratio. Other
coordinates stay at the reference value. The runner integrates densities in
these coordinates with trapezoidal weights and reports prior/posterior mode
separation, normalized boundary distance, boundary mass, interval width, and
quadrature refinement. W is the negative of Step 4's log-time coordinate.

- `oracle_r_known` conditions on a deterministic, prescribed driver and rest
  hemodynamic initial state, with zero state diffusion. Independent numerical
  integration settings are compared, and the likelihood is the product of
  conditional Student-t observation densities.
- `matched_model_calibration` draws the parameter from its declared prior,
  the transformed initial state from the configured zero-mean Gaussian, and
  all six process-noise increments independently. Its transition law is
  `z[t+1] = RK4(z[t]) + epsilon`, with covariance
  `dt * diag(process_std**2)`. This is the discrete model approximated by the
  fitter; it does not assert exact continuous-time SDE simulation.
- The production filter and smoother are checked in a linear Gaussian
  specialization against exact Kalman filtering/RTS. A bootstrap particle
  filter then estimates the joint likelihood for short matched prefixes.
  Independent likelihood estimates are averaged on the likelihood scale.
  Particle budgets, independent-run splits, grid refinement, effective sample
  size, and surviving ancestors determine whether a reference comparison is
  numerically resolved. An unresolved reference remains inconclusive.
- `misspecification_stress_test` calls the retained Step 4 truth generator
  with shorter records, preserving its pulse, external driver, deterministic
  hemodynamics, and Student-t noise. It is not labelled SBC. The old failure
  establishes that the old complete pipeline failed its registered gates;
  it did not isolate identifiability, model mismatch, and inference error.

The existing EKF marginal score is named `predictive_score` in this diagnostic
and produces a **generalized posterior**. Only the direct oracle likelihood
and joint particle likelihood use `parameter_log_likelihood`. Increasing grid
resolution alone does not validate the former as a likelihood.

Secondary diagnostics compare U0 fixed, true-parameter conditional inference,
and the score-optimal one-parameter fit against true `r` and clean EEG/HbO/HbR.
State-posterior variance excludes observation noise; noisy masked-observation
intervals are reported separately. These are conditional Gaussian-moment
intervals, without parameter-uncertainty propagation. Coverage is aggregated
by independent replicate before descriptive bootstrap; the small panel cannot
establish SBC or teacher qualification. The no-observation prior is a software
negative control. Shared-information pairing nulls, parameter-integrated UQ,
U3, and the registered larger calibration panel belong to later experiments.

Masked fNIRS values are removed before estimation, including parameter refits.
There is no normalization or data-derived noise estimation in this synthetic
panel. Numerical failure stops the run and retains its failure record; seeds
are not redrawn and thresholds are not relaxed after seeing results. Outputs
use a fresh directory under the configured experiment root. No measured or
protected data, tokenizer target, independent-modality ownership, or coupling
contract changes are included. Current execution and next action remain in
the research-state registry.

## Full Step5 staged continuation

The user-requested continuation is governed by
[`step5_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml)
and [`evaluate_step5.py`](../experiments/evaluate_step5.py). Numerical
details are frozen in that configuration before the relevant stage is evaluated.
Stage results are reported separately, with numerical,
parameter, state, and shared-information conclusions distinguished.

**Step5A0** replaces IRLS observation curvature with joint Student-t
Gauss-Hermite integration and Gaussian moment matching in a separate
[`joint inference module`](../src/inference/t3a_balloon_joint_ssm.py). The
immutable T3a dynamics and transition Jacobian remain the forward owner.
The likelihood integrates the observation-active transformed coordinates;
the other states are handled by Gaussian conditional regression. EKF dynamics
and Gaussian closure remain approximations. The new matched calibration panel
uses fresh independent parameter/noise draws and compares oracle, legacy
marginal-score, and joint-likelihood parameter distributions. Numerical
refinement changes grids, not observations or statistical thresholds.

Particle likelihood precision and path ancestry are separate diagnostics.
Likelihood-scale independent estimates, grid/budget checks, and log-likelihood
Monte Carlo error govern the likelihood reference. Surviving ancestor fraction
governs whether those same particles may support path smoothing. The prior
combined check is also reported; a path failure is retained and cannot be
described as a fully resolved particle smoother.

**Step5A1** tests U0 and the minimal G/W/Z candidates. A parameter distribution
is fitted to an independent synthetic training trial; the state targets and
masked scores use newly generated held-out trials. For the stress branch,
both training and held-out observations use the retained Step 4 generator.
The parameter distribution is held fixed across correct-pair and null inputs;
it cannot carry information about the held-out pair. This is a frozen-training
parameter mixture of conditional state posteriors, not a claim that parameter
weights have been updated to the full joint posterior using the held-out
record. Its uncertainty must pass the declared state coverage checks.

Variance separates the mean conditional state variance from variance across
parameter means; only noisy-observation prediction adds Student-t noise.
Posterior-CDF quadrature is refined for teacher mean/variance stability.
U0 additionally receives the predeclared GWZ prior-quantile sensitivity panel.
Cross-parameter driver stability and known-truth recovery are both reported.
U3 uses a full-support tensor grid with explicit refinement, posterior
correlation/ridge, and boundary-mass diagnostics; it is not eligible for
selection. Parameter intervals are compared with configured material changes,
not a point-identification or “any equivalent alternative fails” rule.

Shared information requires paired improvement over own history, own history
plus an independently trained task-time template, independent pairing, and
circular-shift controls in both center-masked directions. Whole-modality
missingness is reported as a separate diagnostic. Synthetic clusters are
independent parameter/trial replicates; measured clusters are subjects. No
timepoint-level binomial confidence claim is used for trajectory coverage.
Every fixed case identity must receive either its complete result or a retained
failure record. Generation, prior-support or inference exceptions are not
redrawn, filtered out, or repaired by changing frozen clipping/step-size rules.
Partial successful-case summaries are descriptive and cannot grant teacher
qualification when the registered experiment is incomplete. Independent
same-seed numerical diagnostics may explain a failure without reclassifying it.

**Step5B** is restricted to the explicitly listed development subjects and
sessions, with the configured eight training and two held-out trials per
session. It targets a new trial in an existing subject/session. U0 and a
synthetically qualified minimal one-parameter candidate are evaluated; if no
free candidate qualifies, a qualified U0 may be evaluated alone. Every
data-dependent transform and parameter fit uses the training inventory.
Input masking precedes information-propagating transforms, using distinct
input and target-scoring processing where required. Loader source facts and
the concrete mask-processing implementation are checked before array access.
Subjects 19–23 are not a new confirmation cohort and are not included in this
version; protected subjects 24–29 remain closed.

The measured implementation's additional numerical choices were first frozen in
[`step5b_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5b_v1.yaml),
which pins the synthetic base configuration and reuses the immutable
three-session metadata validator for factual record/event/clock checks. That
older diagnostic's authorization fields are not reused as authorization.
The explicit Step5 request and the admitted candidate's own complete synthetic
panel govern progression. A failed independent G/Z candidate does not
invalidate a complete W panel; U0's registered cross-scenario sensitivity does
require the full scenario inventory.

[`step5b_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/step5b_v2.yaml)
adds same-support local posterior refinement after the retained uniform-grid
failure: old endpoints and tail nodes remain, intervals within the configured
log-density drop are bisected, and the CDF threshold is unchanged. It also
restricts fNIRS channel eligibility using training-only positive native
intensities before channel selection. A selected held-out pair with invalid
support still fails; held-out observations cannot select a replacement pair.
Same-configuration output recovery can reuse prepared inputs, resolved curves
and case outcomes with input digests and unchanged scientific functions. It
adds no independent statistical replicates and preserves prior failure records.

The new native-trial path consumes the cache's `native_input_fnirs` and the
existing native EEG reader. It does not use the globally standardized or
filtered canonical arrays to construct masked inputs. EEG log block power and
trial-local OD/motion/filter/MBLL fNIRS transforms precede a frozen per-subject
projection. Channel selection uses training signal/first-difference ratios;
PCA and projection scales use no task labels. HbO and HbR share a single
training scale, preserving their relative amplitudes. The amplitude gauge is
defined by the fixed reference model covariance; P0/Q0 and the state structure
are not added as free parameters. Event-relative output times and both native
clock anchors remain in the trial inventory.

The first-difference estimator uses the exact median absolute difference of
two unit Student-t draws. Its software check uses retained, known synthetic
noise. The measured scale is the larger of its training estimate and the
frozen synthetic noise scale. This is an initial scale rule, not a calibration
claim about colored measured noise. Center-mask scores compare the qualified
candidate with same-modality visible context (including future context in the
fixed-interval operator), a training task-time template, independent training
pairing and circular shifts. Subject-cluster intervals and leave-one-subject-out
means govern shared-information evidence. Noisy-observation Gaussian-moment
coverage is diagnostic; measured data provide no latent-state coverage truth.

**Comprehensive UQ** follows core teacher qualification. It reports modality
and subject variation, conditional-state/parameter/noisy-observation variance,
mask effects, and calibration with the known-subject new-trial unit explicitly
stated. Leave-one-training-trial-out calibration is descriptive, not a claim
of standard split-conformal finite-sample coverage. Precision weighting is
tested only when the configured clustered risk-ranking criterion is met;
uniform weighting is otherwise retained. Exports preserve the same-modality
observation-space teacher mean, variance, and masks; `r` is a diagnostic
export. No tokenizer training or target redesign is included.

A failed numerical or scientific prerequisite is retained and prevents the
dependent measured/UQ action. “Stage evaluated” does not imply qualification,
and a skipped dependent stage is reported as not run, never as a successful
experiment. Current execution remains solely in the research-state registry.

## Fixed question and decision target

The intended final object is the smallest tokenizer that jointly satisfies:

1. `source` retains the same-modality slice of a qualified offline joint
   EEG+HbO+HbR teacher trajectory;
2. `observation` retains modality-specific measured information;
3. quantization, if admitted, does not materially degrade either function;
4. an optional coupling prior improves a held-out cross-modal proper score over
   the observation/source-history baseline without harming the first three
   properties.

The physical teacher's primary task is a physiology-constrained decomposition,
not pointwise observation copying or EEG-only cross-modal translation. From
aligned noisy EEG, HbO, and HbR observations it estimates an operational shared
neural driver `r(t)`, named hemodynamic states, an observation-space posterior,
and modality-specific or systemic nuisance/residual components. The shared
driver, physiological states, and separated noise are estimates rather than
ground truth and become decision-eligible only after the physical,
identifiability, corruption, null, and calibration checks below.

Observation-space reconstruction remains useful for posterior predictive
checking and for locating failure, but it is not the primary definition of a
good physical teacher. A candidate is not rejected solely because its point
MSE/NRMSE, R², or PCC is worse than persistence or a time-shift control.
Non-finite trajectories, physical-boundary violations, non-identifiable
physiological claims, systematic posterior-predictive failure, or failed
uncertainty calibration still reject the candidate.

The nine forward principles in
[`METHOD_RATIONALE.md`](METHOD_RATIONALE.md#frozen-theory-and-architecture-contract-unimplemented)
and the data/mask/split rules in [`DATA_CONTRACT.md`](DATA_CONTRACT.md) remain
fixed. This protocol selects implementations inside those boundaries; it does
not redefine them.

"Optimal" is deliberately lexicographic rather than a weighted total score:

1. qualify the teacher on physical identity, identifiability, robustness, null,
   and calibration gates;
2. pass every required source and observation fidelity gate;
3. pass uncertainty, stability, and codebook-health floors;
4. among passers, use the lowest token rate and simplest model;
5. use held-out proper score only as the final tie-breaker.

A strong result in one modality cannot compensate for failure in the other. If
continuous representations pass but VQ fails, the result is "no discrete
tokenizer admitted", not a forced codebook. If coupling fails, a qualified
coupling-free tokenizer may still be retained.

## Experiment flow

![PST-DISCOVERY-v1 staged experiment plan](physiology_semantic_tokenizer/figures/pst_discovery_v1_experiment_plan.svg)

[Editable figure source](physiology_semantic_tokenizer/architecture/pst_discovery_v1_experiment_plan.json) ·
[standalone SVG](physiology_semantic_tokenizer/figures/pst_discovery_v1_experiment_plan.svg) ·
[visual style reference](physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg)

The upper spine is the only promotion path. The three detailed panels expose the
teacher, tokenizer, and coupling candidate ladders; the bottom lane contains
diagnostic children that are explicitly not decision-eligible.

The bottom diagnostic lane can explain a failure or motivate `v2`; it is not an
in-run retry and cannot be pooled into the `v1` promotion estimate.

## Common estimand and statistical contract

### Unit of inference

The biological unit is the subject. Scores are first aggregated over valid
coordinates, time points, patches, trials, and records within subject and then
averaged with equal subject weight. Channels, teacher coordinates, code IDs,
windows, and seeds are not independent biological replicates. Three paired seeds
are optimization/stability repeats and are reported separately from subject
uncertainty.

For a candidate `C` and baseline `B`, every main contrast is oriented so higher
is better:

```text
delta_s = subject_score_s(C) - subject_score_s(B)
Delta   = mean_s(delta_s)
```

Intervals use subject-cluster bootstrap or an exact subject-block test when the
subject count is too small for a stable bootstrap. Missing required support is
`INVALID`; the denominator is not silently reduced.

### Partitions and leakage control

- All channel selection, normalization, teacher parameters, uncertainty
  calibration, target projections, encoder/checkpoint selection, codebooks, and
  coupling maps are fit inside the authorized fit partition.
- Subject plus record/trial/video dependencies are grouped across every split.
- Historical subjects 01--18 and 19--23 have already influenced prior method
  development and cannot become a genuinely fresh confirmation cohort merely by
  relabeling them.
- The new nonprotected confirmation inventory is unresolved and must be frozen
  before measured execution. Protected subjects 24--29 remain closed.
- Task/condition annotations may define nuisance controls or matched nulls, but
  they are not prediction targets, architecture-selection endpoints, or losses
  in this protocol.

### Decision states and multiplicity

Every gate returns exactly one of `PASS`, `FAIL`, `INCONCLUSIVE`, or `INVALID`.
Confidence intervals crossing zero or a non-inferiority margin are
`INCONCLUSIVE`; technical interruption or incomplete support is `INVALID`.

Each stage has one named primary contrast. Lag, horizon, chromophore, channel,
and candidate families are either descriptive or controlled with a predeclared
max-statistic/closed-testing procedure. No best patch, lag, channel set, seed, or
checkpoint may be chosen after confirmation data are viewed.

The practical margins `delta_T`, `delta_S`, `delta_O`, and `delta_H` are
intentionally unresolved here: each must be estimated from synthetic recovery,
measurement repeatability, or fit-only technical repeats and then frozen before
the corresponding confirmation run. For the teacher, `delta_T` governs shared
driver/state robustness and calibration, not superiority of pointwise
observation reconstruction. A percentage chosen after seeing held-out results
is not an admissible margin.

## P0: software and synthetic qualification

P0 is mandatory before measured data. One synthetic generator must emit known
`r(t)`, extended Balloon states `s/f/v/p/q`, the true parameters and operators,
and clean EEG/HbO/HbR trajectories under an explicit
`p/q -> HbT/HbO/HbR` concentration map (plus the recorded optical operator when
the input coordinate is optical density) and known EEG-to-fNIRS delay. It then
injects heteroscedastic noise and controlled modality-specific or systemic artifacts--spikes, drift,
steps, bursts or high-frequency contamination, and dropout--while retaining the
clean reference, nuisance component, artifact mask, and severity. Full-input,
masked/held-out, and missing-modality replays must use the same generator. The
smallest runnable check must demonstrate:

- continuous target construction before patching/tokenization;
- exact canonical-key joins and distinct measurement, teacher, uncertainty,
  observation-residual, token, and lag masks;
- no cross-modal read before either main tokenizer emits its representation;
  the offline joint teacher is the declared fit-fold-only exception and emits
  detached modality-specific targets;
- the resting equilibrium, positive physiological states, stable integration,
  valid oxygen extraction, Balloon-compartment inflow/outflow, total-Hb and
  deoxy-Hb balances, and the explicit hemodynamic/optical observation map; these
  checks
  do not turn the model into a full oxygen-diffusion or CMRO2 model;
- prior-predictive support plus simulation-based calibration, profile-likelihood
  or equivalent identifiability checks, and multi-start sensitivity for every
  parameter allowed to vary;
- recovery of known `r(t)` and named physiological states, attenuation of
  injected artifacts, and failure on independent/time-shift/pairing/spatial
  nulls; observation-space MSE and correlation remain descriptive;
- residual agreement with injected corruption on artifact support and absence
  of systematic clean-signal removal off that support;
- calibrated predictive intervals, with uncertainty increasing under stronger
  corruption, masking, or missing input;
- branch and coupling gradient allowlists;
- config/target/summary serialization, atomic publication, and an explicit
  incomplete-run state.

P0 remains the qualification path. The separately registered measured
reconstruction/null diagnostic may run only on its nonprotected development
split and remains decision-ineligible; it cannot open protected data or promote
a teacher. Protected evaluation requires a separate explicit request.

## T: physical-teacher selection

### Selection principle and candidate range

The comparison is a staged ladder, not a Cartesian model search. Controls and
mechanism references cannot become the physical teacher merely by winning a
reconstruction metric. The first promotion candidate is the smallest robust
nonlinear Balloon model with explicit observation operators.

| ID | Candidate | Frozen question | Role |
| --- | --- | --- | --- |
| `T0-native` | measured coordinates with persistence, time-shift, and fit-fold smoothing controls | How much apparent recovery requires no latent physiology? | predictive control; never promoted |
| `T1-self` | independent EEG and fNIRS linear LDS/RTS models | How much smoothing and uncertainty calibration is available without a shared state? | single-modality attribution control |
| `T2a-croce-pf` | paper-faithful Croce-2017 nonlinear particle-filter mechanism | Which published Croce behaviours reproduce under the same synthetic contract? | fixed mechanism reference; not the default teacher |
| `T2b-adaptive-legacy` | current bounded adaptive Croce-like RTS/AR implementation | Which current results survive the new physical and identifiability tests? | historical regression baseline; never relabelled as exact Croce |
| `T3a-balloon-robust` | constrained nonlinear `r/s/f/v/p/q` extended Balloon state model, explicit EEG and fNIRS optical observations, masks, and fixed-degree-of-freedom Student-t observation noise | Can the model recover an identifiable shared drive and plausible physiological states while isolating corruption? | **primary promotion candidate** |
| `T3b-systemic` | `T3a` plus one low-dimensional fNIRS systemic/extracerebral nuisance factor | Does a frozen residual/PPC failure specifically improve without absorbing `r(t)`? | conditional extension after its predeclared `T-P3`/`T-G4` trigger |
| `T3c-hierarchical` | partial pooling of only parameters already identifiable in `T3a` | Does cross-subject pooling improve stability without prior domination? | conditional extension only after `T-P2` identifiability and a frozen cross-subject stability failure |
| `T4-dcm-lite` | two-stage EEG neural-state to Balloon/optical fNIRS model; fNIRS cannot retroactively rewrite the EEG neural state | Does a more conventional directed interpretation support the same physiology? | interpretability reference, not a joint-teacher promotion arm |
| `T5-spatial` | local geometry extension of the simplest `T3` model passing `T-G0`--`T-G4` | Is additional local spatial support necessary after physiology qualifies? | final conditional refinement at `T-G5` |

The executable synthetic P0 panel is intentionally smaller:
`T0-native`, `T1-self`, `T2b-adaptive-legacy`, and
`T3a-balloon-robust`. `T2a-croce-pf` and `T4-dcm-lite` remain frozen design
references until a contract-faithful adapter exists; they must not appear as
tested or unavailable rows manufactured from `NaN`. This P0 qualifies the
primary candidate and its current controls, not the later `T-P5` comparison.

Gamma-HRF/delay controls, Factorial/SLDS noise branches, switching regimes,
heteroscedastic process models, Gaussian-process dynamics, and full neural-mass
models remain diagnostics. They are not part of the first promotion ladder.
`T3a` does not simultaneously add switching, hierarchy, spatial structure, and
multiple nuisance factors.

`T5-spatial` starts from one HbO/HbR pair plus six nearest EEG channels, then
tests two and at most four local fNIRS pairs with at most twelve EEG channels,
subject to actual channel support. It reuses the existing adjacency/geometry
owners. A geometry-aware linear observation operator and covariance are tested
before any graph neural network. Template geometry supports adjacency and
qualitative topology only, not exact cross-modal distance or co-registration.

Channel sets are selected on fit data without labels. Added channels are
retained only if they improve a frozen posterior-predictive or proper-score
endpoint, survive channel-drop and geometry-permutation nulls, and do not
degrade calibration or state stability. Otherwise the smaller local model wins.
An all-scalp model is not part of this generation.

### Physiological state and parameter contract

The initial `T3a` continuous-time core follows the normalized Balloon dynamics
of [Friston et al. (2000)](https://www.fil.ion.ucl.ac.uk/spm/doc/papers/karl_nonlinear.pdf)
and the total-Hb/optics extension of
[Tak et al. (2015)](https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf).
Those papers define the model family; their fitted prior means are not treated
as universal human measurement ranges.

```text
ds/dt       = beta * r - kappa * s - gamma * (f - 1)
df/dt       = s
f_out       = v^(1/alpha)
tau * dv/dt = f - f_out
tau * dp/dt = f - f_out * p / v
tau * dq/dt = f * E(f, E0) / E0 - f_out * q / v
E(f, E0)    = 1 - (1 - E0)^(1/f)

domain: f > 0; 0 < E0 < 1; 0 < E(f, E0) < 1
rest:   r = s = 0; f = v = p = q = 1
```

Here `r` is the shared neural state in the fixed EEG loading/variance gauge; it
is not measured firing. `beta` is a dimensionless effective neural-to-vascular
gain in that gauge, not a molecular efficacy constant.
`s = df/dt` is the vasoactive signal; `f` is inflow normalized to rest; `v` is
normalized venous Balloon volume; and `p/q` are the normalized total-Hb/deoxy-Hb
model coordinates of that compartment. With time in seconds, `f/v/p/q` are
dimensionless, `s` has units s^-1, `r` and `gamma` have units s^-2, `beta` is
dimensionless, `kappa` has units s^-1, and `tau` has units s. `tau` is the resting transit constant
`V0/F0` of the modeled venous Balloon, not whole-region or whole-brain mean
transit time. `alpha` is its dimensionless outflow-volume exponent. A numeric
prior from another state/time scaling is usable only after its unit conversion
is recorded; copying a published coefficient labelled only as a "rate" into
this parameterization is a `T-P0` failure.

This initial model fixes Tak et al.'s viscoelastic time constant `tau_v` to
zero, so `f_out = v^(1/alpha)`. It is therefore the smallest explicit
total-Hb extension needed for `T3a`, not a claim to reproduce the paper's full
viscoelastic model. A nonzero `tau_v` is admitted only as a later one-parameter
extension after the initial state and parameter contract is identifiable.

The fNIRS forward model must be explicit rather than learned through arbitrary
signed gains:

```text
delta_HbT = P0 * (p - 1)
delta_HbR = Q0 * (q - 1)
delta_HbO = delta_HbT - delta_HbR
```

`P0` and `Q0` are positive baseline scales. If the declared measurement
coordinate is raw optical density, the above concentrations additionally pass
through the recorded wavelength-specific extinction, sensitivity/pathlength,
and cortical-mixing operator. If the coordinate is a released HbO/HbR export,
that optical-density transform is not applied a second time; its recorded
preprocessing/normalization transform is part of the observation operator.
Without those baselines and the recorded optical lineage, `p/q` remain
dimensionless model coordinates and cannot be relabelled as absolute Hb
concentrations. EEG has its own declared observation operator.

The parameter contract separates three kinds of restriction:

- **Hard mathematical/physical boundaries:** `kappa`, `gamma`, `tau`, and
  `alpha` are positive; `0 < E0 < 1`; `f`, `v`, `p`, `q`, and `f_out` remain
  positive; `E(f,E0)` remains in `(0,1)` at every step; `P0 > 0`, `Q0 > 0`,
  and the mapped absolute HbT/HbR/HbO values remain nonnegative with HbR not
  exceeding HbT. The resting equilibrium, compartment balances, units,
  finite integration, and stability checks must pass. These are validity
  conditions, not fitted medical ranges.
- **Neural-drive gauge:** set the baseline of `r` to zero, fix its sign so a
  positive drive increases `s`, normalize its scale by one predeclared
  fit-fold rule, and fix one EEG observation loading. The conventional
  `epsilon` factor is absorbed into `r`; no separate neural-efficacy parameter
  is fitted or reported as measured physiology.
- **Soft source-backed priors:** every numeric prior and plausible-response
  interval must record its units, compartment, species/population and challenge
  condition, primary source, and prior parameterization in the executable
  contract. A posterior pressed against a bound or unchanged from its prior is
  not evidence that the parameter was measured.
- **Measured exploratory release ladder:** retain `P0/Q0`, EEG loading, driver
  scale, noise, and Student-t degrees of freedom as fit-cohort gauges. Compare
  the fixed model first, then the single-parameter `beta`, `kappa`, and `tau`
  fits, then `beta+kappa+tau`, followed by one-at-a-time additions of `gamma`
  and `alpha`. Release `E0` only as a final strong-prior diagnostic because the
  current standardized fNIRS coordinate cannot establish absolute OEF. Only the
  fixed model and the three single-parameter fits are recommendation-eligible;
  `M2`--`M5` are retained only to diagnose compensation. Each subject shares one
  parameter vector across independently reset trials. A later stage cannot be
  retained merely for reconstruction gain when its posterior is boundary-bound,
  prior-dominated, or compensatory. `p` has no separate free dynamic parameter
  in `T3a`.

Names must not overstate what the equations identify. `kappa` and `gamma` are
lumped model coefficients, not direct molecular vasodilation rates; `E0` is the
resting oxygen extraction fraction, not an oxygen dissociation rate. Without
absolute flow/volume and optical calibration, the experiment cannot claim
absolute OEF, CMRO2, or an oxygen dissociation rate. Such quantities remain
outside the result vocabulary even when the latent trajectory looks plausible.

### Teacher test sequence

| Stage | Test items | Promotion consequence |
| --- | --- | --- |
| `T-P0 semantics/physics` | state names, equations, units, gauge, observation map, equilibrium, positivity, finite/stable integration, and parameter-source ledger | any violation is `FAIL` before fitting |
| `T-P1 prior predictive` | draw prior trajectories across the frozen design; check plausible amplitudes/delays, boundary contact, solver failures, and prior sensitivity | unsupported priors or implausible mass dynamics block the candidate |
| `T-P2 identifiability` | simulation-based calibration using the declared EKF-Laplace posterior-CDF approximation, rank/coverage diagnostics, fixed-other-parameter objective slices as the initial posterior-geometry check, multi-start recovery, and parameter/state confounding | non-identifiable parameters are fixed/removed; stable `r` alone earns only state-level status; exact posterior SBC is required if the Laplace approximation itself fails calibration |
| `T-P3 known-truth corruption` | recover `r/s/f/v/p/q`, separate known artifacts/nuisance, preserve clean off-artifact morphology, vary severity/masks/missing modalities, and run independent/time-shift/pairing/spatial nulls | qualifies shared-state and noise-separation claims; point reconstruction metrics remain descriptive |
| `T-P4 measured development` | posterior-predictive checks, residual temporal/spectral structure, modality ablations, leave-one-trial/subject-out stability, and prior-to-posterior movement | permitted only after an executable measured-data contract; no protected access |
| `T-P5 comparison/spatial` | compare the simplest surviving models by predictive score, calibration, complexity, perturbation stability, and spatial/channel nulls | select the smallest fully qualified teacher; otherwise stop |

The current authorized P0 software/synthetic scope covers `T-P0` through
`T-P3` and the known-clean synthetic portion of `T-G4`. Final `T-G4`, `T-P4`,
`T-G5`, and `T-P5` require the later executable measured-data contract; this
plan does not open measured or protected data.

The synthetic `T-G4` screen uses Student-t interval/proper-score calibration
plus lag-one autocorrelation and normalized-spectrum errors of the posterior
mean. Those two reconstruction-shape diagnostics are not full posterior-
predictive simulations and are not labelled PPC in the executable output.

### Teacher outputs and uncertainty convention

All candidates publish common observation and diagnostic fields:

```text
trajectory_mean
aleatoric_variance
epistemic_variance
total_variance = aleatoric_variance + epistemic_variance
observation_values
observation_residual = observation_values - trajectory_mean
nuisance_mean / nuisance_variance, when the candidate declares a nuisance state
named masks and coordinate/channel identities
fit, model/config, parameter, and calibration identities
```

Physiological candidates additionally publish, with explicit state names:

```text
shared_driver_mean / shared_driver_variance
physiological_state_mean / physiological_state_variance
parameter_posterior_summary
parameter_identifiability_status
physical_check_status
```

Residual and noise names follow the [data contract](DATA_CONTRACT.md#residual-and-noise-quantities).

`shared_driver_mean` is the operational `r(t)` estimate.
`physiological_state_mean` contains only states actually present and identified
in the fitted model. `trajectory_mean` is an observation-space posterior
prediction, not a clean-ground-truth claim. `observation_residual` may be
described as separated noise/artifact only to the extent supported by `T-P3`;
otherwise it remains an unassigned observation residual.

The contract uses **variance**, not an ambiguous `uncertainty` scalar. The
legacy adapter mixes variance-like summaries while the current loss divides by
that field without a log-variance term; therefore its uncertainty-weighting
switch is not admitted evidence for this generation.

Calibration is fit-fold-only and frozen before application. Primary uncertainty
endpoints are predictive log score and CRPS on known-clean synthetic coordinates
and prespecified masked real coordinates. 50/80/95% interval coverage and width,
standardized residuals, PIT, and risk-versus-uncertainty monotonicity are
required diagnostics. Same-point joint-posterior coverage is descriptive
because the observation was consumed by the smoother. Aleatoric and epistemic
components remain separate in the artifact and report.

### Teacher gate

| Gate | Required evidence |
| --- | --- |
| `T-G0 physical contract` | lineage, folds, masks, state/operator identity, units, sign/gauge, equilibrium, positivity, finite/stable integration, no label use, and no protected dereference |
| `T-G1 prior/synthetic validity` | source-frozen priors have plausible prior-predictive support; synthetic `r/s/f/v/p/q` and observations are generated without extraction, boundary, compartment-balance, or optical-map failure across the frozen design |
| `T-G2 identifiability` | SBC/coverage, posterior geometry or profile checks, and multi-start recovery support every reported state/parameter; prior-dominated or mutually confounded quantities cannot receive physiological labels |
| `T-G3 shared-state/noise adequacy` | `r(t)` and admitted states remain within frozen perturbation limits; known artifacts enter nuisance/residual rather than the physiological state; off-artifact leakage stays below its frozen bound; independent/time-shift/pairing/spatial null inputs do not yield a qualified shared state |
| `T-G4 calibration/PPC` | predictive log score, CRPS, interval calibration, uncertainty-risk monotonicity, and prespecified temporal/spectral posterior-predictive checks pass; MSE/NRMSE/R²/PCC and same-point reconstruction are descriptive only |
| `T-G5 measured/spatial stability` | measured modality ablations and subject/fold/seed/channel perturbations preserve the admitted claims; any spatial gain survives channel and geometry nulls without worse calibration |

Only the simplest `T3` candidate passing all applicable gates becomes the
frozen training-target producer. It is privileged, label-blind, fit-fold-only,
and training-only; it is never a tokenizer inference input or ground truth.
EEG-only and fNIRS-only reruns are attribution and missing-modality ablations,
not requirements that EEG reconstruct omitted HbO/HbR or vice versa.

Qualification has three explicit outcomes. Passing the full gate yields a
physical teacher. A robust `r(t)` with non-identifiable physiological parameters
is a state-only diagnostic and cannot support parameter-level interpretation.
A model that only smooths observations remains a baseline. If no `T3` candidate
passes `T-G0`--`T-G4`, source-tokenizer development stops; reconstruction work
may continue only as a diagnostic.

## B/Q: source and observation tokenizer

### Functional implementation

The first implementation uses one simple modality-local temporal stem with two
heads:

```text
X_m -> stem_m -> source latent      -> teacher-trajectory decoder
             -> observation latent -> measured-signal decoder
```

EEG and fNIRS stems never read the other modality. Source and observation are
functional roles, not an assertion of statistical independence, and they need
not start as four physically separate encoders. A separate stem is considered
only if the shared-stem gradient audit demonstrates reproducible interference.

The observation target is the measured/masked modality coordinate. It is not
defined as `raw - source` unless a later diagnostic first establishes compatible
units and an identifiable additive decomposition. This avoids repeating the old
power-versus-voltage and single-decoder ambiguity.

### Candidate sequence

| ID | Change from previous row | Question |
| --- | --- | --- |
| `B0-O` | continuous observation-only autoencoder | What reconstruction is available without teacher semantics? |
| `B1-SO` | add continuous source head and frozen teacher supervision | Can both functional roles pass before discretization? |
| `Q-S` | quantize source only with the existing EMA-VQ family | Are physiological source patterns discretizable without losing semantics? |
| `Q-O` | quantize observation only after `Q-S` passes and only if a fully discrete interface is required | Can measured information also survive the bottleneck? |

The initial temporal grid and latent width use the smallest existing setting that
can express the continuous targets. If it fails, width doubles only until the
continuous gate passes. Patch duration is screened on the continuous model,
starting from the existing 2 s grid and testing 1 s only when temporal averaging
is the diagnosed failure; 0.5 s is a later diagnostic, not a default row.

The VQ family is EMA-VQ first. Codebook size starts at the retained K128
reference. If support is persistently redundant, K64 is the only next reduction;
larger K or another quantizer family is considered only when a healthy K128 loses
required information. There is no simultaneous K x D x quantizer search.

### Loss ladder

The default `B1-SO` objective contains only:

```text
L = L_observation_reconstruction + L_source_trajectory
```

`Q-S/Q-O` add only the corresponding VQ commitment/update term. No prototype,
context, balance, independence, cross-masking, or coupling loss is enabled by
default.

Additional terms are one-factor diagnostics with a named trigger:

| Trigger | Single allowed diagnostic | Promotion condition |
| --- | --- | --- |
| calibrated teacher uncertainty passes `T-G4` | uniform source loss vs clipped, normalized precision weighting | improves source score/calibration without worse observation fidelity or effective support |
| actual code collapse under a passing continuous model | existing straight-through balance loss | restores health without exceeding source/observation non-inferiority margins |
| continuous semantics pass but hard-token semantics fail | isolated prototype/topology loss | improves hard retention without codebook redundancy or gradient conflict |
| a valid local sequence endpoint fails while local targets pass | isolated context loss | improves the frozen sequence endpoint without future leakage |

The old multi-entry loss bundle is not restored. Every new entrance has its own
coordinates, masks, weight, ablation, and gradient audit.

### Tokenizer endpoints and gates

| Gate | Required evidence |
| --- | --- |
| `B-G0 support` | train loss and evaluation use the same declared target/mask population; subject/trial/patch coverage is explicit |
| `B-G1 observation` | held-out masked measurement log score/NRMSE is non-inferior to `B0-O`; EEG spectral and fNIRS HbO/HbR morphology are secondary fidelity checks |
| `B-G2 source` | continuous source latent/decoder retains the frozen teacher trajectory beyond history and target-permutation baselines, for every required modality/coordinate |
| `B-G3 attribution` | source-only, observation-only, and full interventions show that source gain is not supplied by an observation/residual bypass; cross-decoding is reported, not forced to zero |
| `Q-G1 retention` | expected embedding, posterior, and hard ID are each compared with the continuous upper bound; hard-token source and observation losses stay within frozen margins |
| `Q-G2 health` | active/effective support, dead/revival history, minimum per-code support, usage concentration, participation rank, near-duplicates, and subject/seed stability pass as guardrails |

Codebook utilization is not itself a semantic endpoint. Among gate-passing
models, the lowest bitrate wins; a higher occupancy count cannot rescue worse
reconstruction or teacher retention. Continuous latents, expected embeddings,
posteriors, hard IDs, and codebook embeddings are all exported so hard IDs never
become the entire representation record.

## C: coupling-prior return

### Frozen evaluation target

Coupling is tested only after the marginal tokenizer is frozen. The primary
representation-level estimand is the subject-equal held-out proper-score
increment for measured fNIRS observation residual:

```text
q0(Y_F(t+h) | H_F_observation, H_F_source, phase/time/systemic controls)
q1(Y_F(t+h) | H_F_observation, H_F_source,
                 H_E_source, phase/time/systemic controls)

Delta_coupling = score(q1) - score(q0)
```

The evaluator is low-capacity and cross-fitted. Positive lag means EEG precedes
the fNIRS endpoint. One primary horizon or integrated horizon score is frozen
before confirmation; individual lag curves are descriptive and family-wise
controlled. Full-window tokens can support only an offline association label. A
prospective/delayed-prediction claim additionally requires strict receptive-field
cutoff tests.

Required nulls preserve the relevant marginals and dependence structure:

- whole-window circular shift with tokens and masks shifted together;
- same-subject/condition nonoverlapping trial derangement;
- independent-window pairing;
- lag reversal/negative-lag control;
- spatial adjacency permutation for a spatial-prior diagnostic.

NMI, co-occurrence heatmaps, row entropy, and same numeric IDs are descriptive
only. The teacher's latent flow is an upper-bound diagnostic, not the primary
coupling target.

### Minimal prior ladder

| ID | Tokenizer gradient | Purpose |
| --- | --- | --- |
| `C0` | none; fit a lag-balanced, marginal-residualized `q0/q1` after tokenizer freeze | establish whether the representation contains incremental information at all |
| `C1-source` | a small fit-selected weight reaches only the EEG source path; fNIRS target/history, both observation paths, teacher, and baseline are detached | test whether the one-term shaper preserves coupling-relevant source information |
| `C2-uncertainty` | same as `C1`, with clipped normalized confidence weights | optional only after `T-G4`; unweighted results remain co-primary sensitivity |

If `C0` does not beat `q0` and all registered nulls, no coupling loss reaches the
tokenizer. `C1/C2` are admitted only when `Delta_coupling` improves and all
observation/source fidelity and codebook-health gates remain non-inferior to the
coupling-free tokenizer.

Only the historical lag-balanced conditional pair likelihood is eligible to
return initially, because its training target matches the evaluation contrast.
The former lag-focus entropy, joint-entropy, codebook-neighbor JS, local/context
residual maps, and multi-term coupling bundle remain diagnostics. They may make a
coupling tensor look concentrated without improving held-out information and are
not reintroduced together. Best and final checkpoints, per-loss gradient norms,
reconstruction-versus-coupling cosine conflict, and assignment health are all
reported.

## Side-path experiments without workflow sprawl

A side path is a diagnostic child of a main run, not a new project track. It
shares the parent's data/split/teacher/code identities and lives at:

```text
experiments/runs/physiology_semantic_tokenizer/tokenizer_discovery_v1/
  <immutable-run-id>/
    resolved_config.yaml
    summary.json
    metrics.csv
    figures/
    diagnostics/
      <probe-id>/
```

Each diagnostic records `parent_run_id`, `scope`, `hypothesis`, `estimand_id`,
`operator/null`, `status`, and `decision_eligibility=false`. It may use
`synthetic`, `diagnostic`, `null`, or `development` scope. It cannot change the
parent summary, reuse a protected unlock, or promote a candidate. A diagnostic
that motivates a new main hypothesis requires a new contract version before
fresh confirmation data are viewed.

`research_state/registry.json` records only suite/program state transitions. It
does not gain one record per probe, seed, channel arm, or gate. The retained
result index is updated only when a conclusion and its minimum provenance package
are frozen.

## Code ownership for later implementation

No scaffolding is created by this design. When implementation starts, reuse the
existing owners:

| Responsibility | Owner |
| --- | --- |
| continuous teacher and family adapter | `src/teachers/` |
| target artifact, masks, joins, and provenance | `src/data/` |
| modality-local source/observation tokenizer | `src/tokenizers/` |
| reconstruction, semantic, VQ, and optional coupling objectives | `src/losses/` |
| proper scores, calibration, retention, and codebook health | `src/metrics/` and `src/analysis/` |
| new orchestration/analysis entry | `experiments/scripts/`; existing commands retain their recorded paths |
| reviewed executable contract, when ready | `experiments/configs/physiology_semantic_tokenizer/` |

Do not reactivate or rename an E0--E2/R-series YAML, archived source/observation
runner, or old coupling suite. There is no need for a manager, plugin layer,
parallel results root, or separate authorization file.

## Unresolved before measured qualification or confirmation

The synthetic P0 contract and the bounded measured diagnostic contract are
executable. The following values remain unresolved for measured qualification
or confirmation and do not change the diagnostic's exploratory status:

1. the exact nonprotected dataset and subject/record split providing a genuinely
   fresh confirmation set beyond the registered development diagnostic;
2. the source-frozen soft priors, fixed versus free parameter list, parameter
   identifiability/SBC criteria, and numerical `r(t)` or physiological-state
   perturbation limits for `T-G1`--`T-G3`;
3. numeric `delta_S`, `delta_O`, and `delta_H` margins plus the single primary
   coupling horizon or integrated horizon definition and its
   family-wise null procedure;
4. maximum training steps/checkpoint rule and the measured-run compute budget;
5. the measured-data continuous target schema/version implementing the named shared
   driver, physiological states, nuisance/residual, trajectory, parameter
   summary, identifiability, physical-check, and variance fields above;
6. any measured-data corruption/masking schedule and clean-reference definition.

The `T3a-balloon-robust` P0 implementation, frozen synthetic generator,
corruption/null schedule, common output tables, gates, and Chinese renderer now
live in
[`t3a_balloon_robust_p0.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3a_balloon_robust_p0.yaml),
[`evaluate_t3a_balloon_robust_p0.py`](../experiments/evaluate_t3a_balloon_robust_p0.py),
and
[`render_t3a_balloon_robust_p0.py`](../experiments/scripts/render_t3a_balloon_robust_p0.py).
The bounded measured reconstruction/null diagnostic is registered in
[`t3_measured_reconstruction_null_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3_measured_reconstruction_null_v1.yaml)
and
[`evaluate_t3_measured_reconstruction_null.py`](../experiments/evaluate_t3_measured_reconstruction_null.py).
It uses the canonical measured loader with `raw_with_ocular_artifact`, the
01--18 fit / 19--23 population pure-apply split, and declared independent,
pairing, and time-shift nulls. The measured non-circular time-shift comparison
scores the paired and shifted targets only on their common finite support; its
100-point support is not pooled with the 200-point independent/pairing nulls.
Its result is a nonprotected exploratory
diagnostic and is not a Croce/Balloon qualification, clean-ground-truth claim,
or protected evaluation. `T3b`, `T3c`, and `T5` enter only after their declared
triggers.

The plan's second-step fit-only identifiability suite is registered separately
in
[`t3_identifiability_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3_identifiability_v1.yaml)
and
[`evaluate_t3_identifiability.py`](../experiments/evaluate_t3_identifiability.py).
It freezes likelihood-only M2 (`beta`, `kappa`, `tau`) diagnostics at 16
transformed-space starts, a true one-parameter profile that reoptimizes both
companion parameters and latent states, 25% transformed-bound expansion, and
a six-raw-parameter conditional forward sensitivity SVD. One noisy
known-truth clean-scenario synthetic case must complete before the loader is
called. The measured arm fits its observation gauge and M0 selection score on
01--18 only, then analyzes the low/median/high representatives' eight fit
trials. The shared loader constructs canonical dataset-index metadata and
window references, but it never loads arrays or materializes window samples
for 19--23 validation and 24--29 protected subjects. This suite is exploratory
and cannot change qualification, promotion, or protected-data state.

The plan's third-step three-session LOSO diagnostic is registered in
[`t3_multisession_loso_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml)
and
[`evaluate_t3_multisession_loso.py`](../experiments/evaluate_t3_multisession_loso.py).
It uses only subjects 01--18, MA trials, and cache records
`session_01/03/05`; each fold fits two complete sessions and applies frozen
objects to the third. The common safe window is `[-5,+25) s`, with fNIRS
masked from task onset and the primary score restricted to the 15-second
nominal recovery envelope `[+10,+25) s`. Because event durations are absent,
the endpoint is not labelled an exact annotated rest period. Only effective
`kappa` varies: the two training-session estimates define a zero-sum log
session deviation and a geometric subject center for held-out apply. All other
physiological parameters remain fixed. This diagnostic cannot load 19--29
arrays or alter qualification, promotion, or protected-data state.

The plan's fourth-step hierarchy begins with the array-free admission contract
[`t3c_hierarchical_composite_admission_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml)
and
[`evaluate_t3c_hierarchical_composite_admission.py`](../experiments/evaluate_t3c_hierarchical_composite_admission.py).
It freezes the analytic `G_f/T_f/zeta_f/T_v` coordinate and a diagonal
one/two-dimensional Normal hierarchy, then checks the frozen Step 2/3 evidence
before any new measured metadata or array access. At the 2026-09-03 v3
admission snapshot the result was `BLOCKED_PREREQUISITE`: `T-P2`, a common
gauge, a prospective fixed endpoint, composite SBC/profile/multistart evidence,
and a pre-measured practical margin were not yet available. Consequently no
measured hierarchical fit was registered or authorized by this entry.

The follow-up synthetic `T-P2` composite screen is registered in
[`t3c_composite_synthetic_t2_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml)
and
[`evaluate_t3c_composite_synthetic_t2.py`](../experiments/evaluate_t3c_composite_synthetic_t2.py).
Its formal run uses 60 independent known-truth replicates for each
one-dimensional direction (`C1_G`, `C1_T`), independently reset training and
held-out trials, and a fitter boundary containing noisy training observations
but no realized truth, driver, generation seed, or held-out array. Both C1
directions failed their registered gates, so the run decision is
`BLOCKED_C1_COMPOSITE_IDENTIFIABILITY` and `C2_GT` was not run. The detailed
result is retained in the
[`T-P2` report](analysis/20260903_T3C_COMPOSITE_SYNTHETIC_TP2_REPORT.md).
This synthetic evidence is not qualification evidence and does not authorize
measured hierarchical fitting.

## Historical lifecycle boundary

The following table remains a lifecycle overlay for the superseded flow. It does
not rewrite dated evidence; linked reports remain the evidence owners.

| Historical item | Lifecycle | Evidence or retained plan | Retained use |
| --- | --- | --- | --- |
| E0--E2 and R0--R2 generations | **stopped** | [`06_EXPERIMENT_LOG.md`](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) and [`20260728_R_SERIES_EXPERIMENT_REPORT.md`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md) | Historical results and failure boundaries only |
| SSM reliability screen | **stopped** | [`20260819 SSM reconstruction reliability results`](analysis/20260819_SSM_RECONSTRUCTION_RELIABILITY_RESULTS.md) | Exploratory reliability evidence only |
| Continuous-latent screen | **stopped** | [`20260819 continuous shared/private latent results`](analysis/20260819_CONTINUOUS_SHARED_PRIVATE_LATENT_RESULTS.md) | Exploratory latent evidence only |
| LC-SPVQ optimization and QC | **stopped** | Dated LC-SPVQ reports under [`analysis/`](analysis/) | Negative/undetermined evidence only |
| Token Atlas Core (T0) | **stopped** | [`TOKEN_PHYSIOLOGY_ATLAS.md`](analysis/TOKEN_PHYSIOLOGY_ATLAS.md) | Development-only retained result |
| Protected comparison campaign and P0 degradation | **stopped** | [`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md) and [`PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md) | Retained comparison evidence only |
| Croce legacy solver and audits | **stopped** | [`CROCE2017_REAL_DATA_VALIDATION_PLAN.md`](../croce_validation/CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Historical qualification/audit evidence only |
| Comparison P1/P2 and unexecuted follow-up | **abandoned** | [`PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md`](comparisons/PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md) | Unstarted comparison candidates only |
| D1B, future R/VQ, LC full development, observation/source map, Atlas Statistical/Full, and Croce follow-ons | **abandoned** | Dated plans and candidate snapshots indexed in [`README.md`](README.md) | Non-runnable historical candidates only |

Neither `stopped` nor `abandoned` evidence authorizes or determines a row in
`PST-DISCOVERY-v1`. Historical plans preserve their original wording for
reproducibility.

## 传感器空间代理的固定 Tak 首轮检验（2026-10-27）

本版落实用户提供的传感器空间实验建议，仅包含层 0–3；数值设置、有限候选矩阵
与停止规则由
[`shared_driver_sensor_proxy_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_sensor_proxy_v1.yaml)
持有，入口为
[`evaluate_shared_driver_sensor_proxy.py`](../experiments/scripts/evaluate_shared_driver_sensor_proxy.py)。
它使用固定 Tak 血流核心和 10 Hz EEG 功率代理，独立于旧空间实验的 inlet-balance
多通道 SSM。旧空间校准曾允许并选中零 EEG 权重，因而相同结果不能说明空间代理
等价；本版空间比较固定共同正 EEG 权重，零权重仅作为单独对照。执行状态和实测
结论仍由 research-state registry 与 owning run 持有，本节不记录进度或结果。

### 范围、身份与观测合同

首轮只使用公开 Single-Trial、一个目标 HbO/HbR 对，以及旧空间父运行
`shared_driver_spatial/20261020_measured_v2/plan.json` 的 metadata 身份；不读取
其历史预测、校准值或 prepared 数组作为新模型输入。目标为该面板中心 FpzFp1
通道对，身份以 plan 的 channel ID 核验，不用裸数组下标跨数据源连接。训练、
开发选择、评价按被试隔离，保留原训练被试全部计划窗口；开发窗口按固定身份顺序
缩减，评价窗口全部保留。完整计划中的缺失、准备失败和数值失败都留下记录，
不按 Hb 拟合残差、响应幅度或候选表现删窗。精确 subject/record/event、目标时间、
90 秒滤波支持、通道列表、错配供体和划分写入本版 plan，先冻结身份再读测量信号。
错配供体允许从同一已登记记录的全部相容事件 metadata 中额外选择；只增加供体
EEG 特征，不增加目标、Hb 读取或非线性拟合，仍先冻结供体再读测量。
这些公开被试此前已经用于开发，评价属于冻结选择后的公开开发重分析，不能称为
从未查看的最终确认集；其他协议的 protected split 仍关闭。

EEG 从中央登记原生电位读取，保持原参考配置，在同一 90 秒连续支持上按
当前 10 Hz owner 的五频带、双向带通、100 毫秒均方功率和 log 变换处理。
30 秒目标身份不变：优先置于 90 秒支持中央，记录边界处将整个支持向记录内部
平移，并保存真实 crop offset；记录不足 90 秒、目标不被包含或支持不完整时失败，
不补零或拼接其他记录。这是本版明确的边界处理，不冒称每个窗口均与旧 Tak
严格中段裁剪相同。裁剪后按目标前 5 秒重新参考。所有代理使用相同滤波支持与
时间算子；不能插值旧 4 Hz 特征或按候选改变滤波。

Hb 只由统一 reader 读取 v5/none 全记录 0.01–0.2 Hz 带通缓存，取原目标支持、
配对通道、掩码与 100 毫秒位置，不重作带通或读取原始 Hb。共同正缩放保留 HbO/HbR
幅比，模型轨迹使用相同的时间参考算子。隐藏目标 Hb 的参考值仅定义评分坐标，
不能成为预测输入。最后一箱按当前 10 Hz 合同排除；主评分支持与参考段排除由
YAML 固定。完整记录离线预处理先于隐藏，所以本版检验的是处理后特征的跨模态
预测，不是原生传感器隔离、实时预测或完整记录滤波 ODE 的精确似然。

### 一个物理核心与三种不同用途

所有代理进入同一个标量驱动、同一 Tak 状态传播及固定 Hb 观测关系。物理参数
采用当前 Tak 参考值，整个代理比较均冻结；不将 `tak_fit` 的逐窗参数重估混入
本版。独立血管状态为 s、f、v、q，严格 p=v，无逐采样血管状态修补、额外 Hb
曲线、每色团独立驱动或不同读出后的重新定标。驱动曲率、四初态、log-flow
先验、观测权重、积分精度和优化预算对所有候选相同。

| 条件 | 当前窗口可见量 | 用途 |
| --- | --- | --- |
| Hb-only 自由驱动参考 | 完整目标 HbO/HbR，无 EEG | 固定物理模型在声明先验与预算下的最佳可得条件拟合 |
| EEG→Hb 预测 | 完整 EEG，目标 HbO/HbR 全时段隐藏，无其他 Hb | 代理晋级的主要证据 |
| 联合重建 | 完整 EEG 与目标 HbO/HbR | 描述加入 EEG 约束后拟合妥协及残差变化 |

Hb-only 每个目标身份只计算并保留一份，后续代理不改写它。它保留纯 Hb 数据误差、
每项先验代价、完整目标、收敛状态和积分检查；有限预算单起点的正则化解不能当作
全局物理不可达下界。三种条件保留相同四初态自由度与先验，但预测的驱动和初态
只能由 EEG 与先验估计；不能从 Hb-only、联合重建或目标 Hb 统计量初始化。
零 EEG 权重的全 Hb 隐藏解只由先验决定，是独立无 EEG 预测基线；零权重联合条件
对应 Hb-only 参考。共同正权重并不意味着 EEG 已有用，仍须报告相对零权重的差异。

### 层 0–3 的有限代理矩阵

层 0 在本版坐标中重建最近六通道 PC1；PCA 的训练协方差、符号和 EEG 尺度锚点
保留，后续候选不能各自再强制归一化输出。设现有参考期处理后的特征为 X、PC loading 为 p、
冻结 EEG 尺度为 s，则按频带分解的 u_b=s·Σ_c p_cb X_cb 满足
原代理 z0=Σ_b u_b。层 1 只学习五个全训练共享的频带系数 a_b，读出
z1=Σ_b a_b u_b，ridge 以 a_b=1 为中心，必须在全一系数时数值复现原代理。
训练中心只用于估计协方差和 loading；实际读出沿用参考期坐标，不另减训练均值
或增加 offset，必须与现有 `fit_feature_coordinate` 的应用语义一致。

层 2 使用另一种明确标识的模型族：先在通道内计算现有功率特征，再对选定通道
逐频带取均值，最后由五个系数合成标量代理。最近六通道与“第六近邻换成第七近邻”
具有相同形式容量；这不是 PC1 的嵌套子模型。层 3 比较最近 3、6、12 通道均值；
内六与外六的共同混合系数只允许 0、0.25、0.5，五个频带共享，端点复用最近六、
十二通道结果。0.25 作为唯一额外环带候选预先进入廉价筛选，不在评价集逐窗、
逐被试或逐频带选择。远端六通道扩展及三个固定随机六通道扩展保留同样的十二
通道总数、聚合方式和五系数容量；它们是空间解释对照，不自动支持局部关系。
随机外圈从原内六之外取不放回唯一通道，种子及实际 ID 在 plan 冻结。

均值族相同的通道数与参数数不保证相同统计信息或严格嵌套；原 PC1、频带增益修正、
均值六通道都保留。空间候选分别相对 PC1 与均值六通道报告，区分原读出改善和空间
范围改变；若近邻、远端和随机扩展相近，只能描述为扩展传感器信息范围有用。
预测系数不是电极位置的生理贡献，模板近邻不是个体皮层来源定位。

### 训练、筛选及冻结后的非线性核查

训练被试内部使用三折分组验证和三个预定 ridge 候选；每折重新拟合 PCA、EEG
尺度、共同 Hb 缩放、评分 SD 和所有学习读出，不能先用完整训练集 PCA 再做内部
验证。ridge 的数值解释是训练 Gram 平均对角线的无量纲倍数；PC1 增益以全一
向量为中心，均值读出以零向量为中心。所有代理参数跨记录共享，两个 Hb 分量
由同一标量读出经物理模型产生。训练选择后重拟整个训练集，开发和评价对象冻结。

廉价物理筛选从实际 Tak 前向模型的静息 Jacobian 构造，复用 EEG/Hb 时间参考
算子、初态规则、可见掩码与先验。它先由 EEG 条件线性反演得到驱动和初态，再映射
到 Hb；不能把处理后代理直接当未参考物理驱动卷积，也不能复用明确标为历史
inlet-balance 的旧线性设计矩阵。筛选属于静息线性近似，最终非线性比较不因此
成为已验证线性模型。训练验证指标的权重按被试等权，时间点和重复窗口不算独立样本。

原 PC1、训练选择的同六通道读出、训练选择的空间候选至多三者进入开发非线性核查。
空间廉价筛选没有稳定收益时保留预定十二通道均值作诊断，不称胜者。开发集只选择
一个与 PC1 不同的候选进入评价；若无人满足晋级条件，可按预定最小误差及固定
复杂度顺序保留一个诊断比较，但显式标记未晋级。评价结果不得反向替换候选、
重选正则或改变阈值。重复候选去重，不能以剩余预算增加其他模型。

上述至多三个代理另有小型 FIR-ridge 经验对照：仅 EEG 输入，以固定时延基直接
预测两个 Hb 分量；独立色团系数只属于经验诊断，不进入物理 SSM。FIR 的时延由 YAML 固定；目标前历史在参考期相对 EEG 坐标中明确采用零值假设，
对预测设计施加与 Hb 相同的前 5 秒参考算子。该边界假设不能扩展成缺失实测值
的补零。边界、参考算子和可评分支持保留与物理预测相同的输入限制与
被试划分。经验预测改善而物理预测不改善提示当前连接或观测结构可能限制信息利用，
不能单凭这一差异指定新机制。FIR 不用于增大非线性候选数。

### 错配、晋级、统计与失败

错误配对使用同一被试、任务、条件及 montage 相容、实际 EEG 滤波支持不重叠的
另一个相容窗口；供体池限同一已登记记录中的 metadata 事件，可包含不作为目标的
额外供体。供体选择只依赖 metadata，并对所有代理固定一致；未找到供体
记为 unavailable，不改用不相容条件，也不将该对照的缺失惩罚成巨大误差以制造收益。
训练用正确配对学习的代理在错配条件保持冻结。预测包括真实 EEG、错误 EEG 及
无 EEG 基线；另保留训练模板，避免把共同任务时间结构误认为样本特异 EEG 信息。

每窗先对固定评分时间和两个色团计算训练 SD 归一化平方误差，再在记录内、被试内
依次等权平均，最后被试等权；主比较使用 NMSE，展示 NRMSE 时只在最终均方后开方。
相对改善定义为 1−候选被试均值 NMSE／基线被试均值 NMSE，不能将 NRMSE 改善
冒充 NMSE 的百分比。基线分母为零或不可估时保留 undefined。跨候选使用相同目标
身份、训练尺度和评分支持，不做每窗重新归一化。HbO/HbR 分开误差、相关和幅度
诊断为次要结果，联合重建误差不能替代全 Hb 隐藏预测收益。

工程晋级要求隐藏 Hb 相对 PC1 的 NMSE 改善至少 3%，至少三分之二可配对被试
方向改善，真实 EEG 优于相容错配和无 EEG 基线，且数值失败率不高于 PC1；真实与错配
的晋级对比只使用双方数值成功并通过积分检查的相同目标，不能用错配求解失败的
惩罚制造信息收益。有效独立配对被试不足三个时判为不可估；错配可配对目标还须
覆盖该阶段至少一半计划
目标，否则错配判据不可估并不晋级。额外供体数量不增加独立被试数或目标分母。
空间改善同时报告相对均值六通道结果，局部
空间主张还须查看远端和随机扩展对照。3% 是预先规定的工作阈值，不是已测效果或
显著性水平。评价仅检查冻结候选的收益是否保留，不启动新的选择循环。

对全部条件一起进行整被试配对 bootstrap，报告描述性 95% 区间和被试方向计数；
不把窗口 bootstrap 当作独立证据，不据逐项区间作多重校正后的显著性声明。
区间跨零是统计不确定，不能称等价；小被试面板也不能据负结果证明 EEG 没有信息。
保留完整计划数、输入可用数、错配可用数、数值收敛数、积分检查通过数、每项配对
被试数，以及有限惩罚的全计划汇总与共同成功子集结果。数值失败惩罚不能掩盖真实
失败率；缺失输入或错配不形成收益配对。停止条件是有限计划单元全部 terminal，
失败不通过换样本、放宽容差或扩大参数搜索消失。

### 残差、预算与解释边界

每个评分身份保存原 PC1、冻结候选和 Hb-only 的完整同坐标预测／残差、掩码、
训练 SD、纯数据代价、先验代价和数值状态，预测残差与联合残差分别标识。残差
诊断限双 Hb 方向／幅比与零时差关系、窗口支持允许的自相关与频谱，以及对已有 EEG
和任务 metadata 的描述性关系。本版不估计跨色团相位；短窗不能支持超出时长分辨能力的低频结论；任务
标签不进入物理读出训练。单目标 Hb 对不足以估计跨 Hb 位置共同变化，本版明确
不估计该项；没有已有运动或外周输入时也不补造对应分析。稳定残差可能来自模型
结构、观测坐标、先验、噪声或数值限制，不直接指定神经或系统性来源。

非线性预算包含开发三候选、评价 PC1 加一个冻结候选，各自的联合、全 Hb 隐藏
真实 EEG、全 Hb 隐藏错误 EEG，以及每个目标的一次 Hb-only 和无 EEG预测。
配置记录所有逻辑单元上限，重复候选和不可用错配各有可审计状态；不能沿用附件中
未计错配与零基线的示例总量。统一单起点，不用多起点的名义暗中扩大预算。
先通过隐藏污染、精确 PC1 嵌套、环带端点、fold 隔离、错配不相容、Tak 导数、
数值失败与积分加密等针对性软件／合成检查，再进行本次已请求的公开范围运行。
较大阶段使用持久监督并根据资源与实测吞吐确定并发；失败产物及完成摘要保留。

首轮不包含电压层局部平均参考、全头优化、多驱动、额外 Hb 成分或物理结构交叉
实验。任何后续分支须基于本版结果提出独立具体设计，不能由当前脚本自动继续。

## 完整 Tak 输出与残差的双离散词表（2026-10-27）

本版按用户给定方案建立独立的 SSM 输出／完整残差词表，不延续旧 functional
网络的 X/T 主监督和联合解码。可执行数值、有限矩阵与停止条件由
[`dual_component_tokenizer_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_tokenizer_v1.yaml)
持有，入口是
[`run_dual_component_tokenizer.py`](../experiments/scripts/run_dual_component_tokenizer.py)。
新版本仅继承完整公开 functional inventory 的身份、几何与被试划分，并与当前中央
metadata 核验；重新准备 10 Hz EEG 代理、缓存 Hb、训练坐标和 Tak 目标。
旧准备数组、teacher 成功子集和训练 checkpoint 不成为新目标。

teacher 有意采用固定参考 Tak 核心，区别于逐窗估计 κ/τ 的 `tak_fit` 对照：
完整 90 秒内只拟合驱动和四个血管初态。S 为观测算子后的完整 EEG 功率代理、
HbO/HbR 预测，O 固定为相同观测坐标中的 u−S。二者相加严格回到 u；它们是
条件模型分解，不是已恢复的生理真值。EEG encoder 读取 2 秒原生电压，不能将
电压与功率代理直接相减。Hb encoder 使用同长缓存观测和明确的可见参考前缀。
每个模态的 S/O encoder、硬量化器及 decoder 独立，decoder 只读取自身整数 ID。
主损失为两个分支的独立误差与 VQ 项，总重建只作诊断。

阶段 A 比较分量压缩参照、连续 student、双硬 student、训练粗糙度匹配平滑分解、
以及同总码率的普通观测 tokenizer。最后一项只监督总观测，其两个码流不具有
SSM／残差含义。所有方法使用相同匹配试次和训练预算，按模态及 HbO/HbR 分别
报告分支与总重建误差、码字占用、独立被试／记录覆盖和码字波形。数据数值除以
冻结训练观测 SD，不逐窗重新归一化，也不加入未计码率的连续幅值 sidecar。

词典工程筛查只使用 validation：与训练分量均值基线比较分支误差，并与连续
student 比较硬量化增加的误差；具体阈值在 YAML 中预先给出。未通过时仍完成
小型下游诊断，但不能称词表已验证或自动扩大模型规模。近等价 teacher 分析在
预定训练身份上改变合法初态后重新优化，要求同一完整目标近等价、数值收敛、
物理域合法和积分加密一致；没有见证时记 unavailable。配对 reference/alternative
student 使用相同初始化、批次次序和短预算。该分析仅涵盖有限优化选解敏感性，
不称为 posterior 或所有合法 teacher 的不确定性。

阶段 B 冻结 tokenizer、码本、decoder、尺度及观测算子，小型 Transformer 只读
整数 token。SSM 分解、平滑分解与普通观测三路线使用相同预算、上下文和有效
遮挡范围。主任务联合遮挡同一片段的 S/O，比较连续时间块及模态块；单分支遮挡
单列诊断。每条序列只有一个 ROI，不通过同窗其他 ROI 暴露重叠 EEG 通道。
用户接受既有带通预处理，本阶段按处理后观测坐标的离线补全执行，不将带通
影响设为启动门槛，也不将未测量的泄漏幅度、原生样本隔离或前瞻能力作为结论。

不同词表的 masked CE 不直接排名：报告各自训练边际／元数据基线到上下文模型
的 CE 增量，并以同一 u 空间的解码预测误差作可比终点。概率波形先逐码字解码，
再以预测概率加权。相同容量和预算的自身模态上下文模型检查跨模态净增益；
冻结模型的来源分支／模态／时间替换检查实际使用敏感性。错配与时间移位仅作
预定控制，不将注意力图等同贡献，也不以两个边际方差直接构造总预测区间。

分析保留 tokenizer 解码与 masked 预测两层误差、真实观测与分解目标的追溯图、
上下文贡献和原型稳定性。汇总顺序为窗口内、试次／记录内、被试内，最后被试等权，
按数据集分开报告；配对区间整被试重采样，区域、窗口和种子不充当独立被试。
完整计划数、可用输入、数值失败、积分失败和共同成功分母均保留。运行由持久
监督器管理，先测 CPU/GPU 吞吐及峰值资源，再安排有界并行；有限任务全部终态后
停止，不因负结果换样本、放宽数值标准或扩展搜索。

## 双词表波形保真与码本健康快速迭代（2026-10-28）

用户要求在保留完整 S/O 目标的前提下实际改善波形和码本健康。首轮有限设置由
[`dual_component_refinement_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_refinement_v1.yaml)
持有，使用同一双词表 runner 的 `refine` / `refine-one` 阶段。只继承已完成 v3
的训练和验证身份、固定 Tak 目标及训练观测尺度，不选择测试被试数组，不重新
拟合 teacher。旧结果和 checkpoint 保留原身份。

先复现旧硬词表，再比较连续 checkpoint 预热与有码字复活的 EMA-VQ、同模态
观测坐标前端、扩大局部上下文以及形态差分监督。EEG 观测前端输入是 SSM 实际
使用的处理后功率代理 u，不是 SSM 预测或残差；Hb 使用相同观测坐标。上下文
变化逐项记录，较长支持须在后续掩码中联合遮住所有重叠 token，不能直接移用
原来的局部遮挡规则。

波形空间字典仅拟合训练集的连续 student 预测，各分支独立用整数查表解码。
一层、两层量化分别计入完整码率，不保留逐片段连续均值或幅度旁路。直接编码
teacher 分量只作量化容量参照，不进入可部署 student 的选择。码本健康与 S/O
独立波形保真分别判定；使用率不能抵消波形误差。阈值和预算在首次执行前固定，
不随结果放宽；通过者按固定额外种子复核。

验证样例只依赖旧模型选择：每库首个身份、中等误差和 P90 误差窗口，尽量来自
不同被试。图中展示 teacher 目标、旧硬重建和新硬重建，独立于下游掩码模型。
可见输入、上下文秒数、完整码率、归一化与样例身份均保留。每组结果在对话中
汇报；未达标时保留失败与具体瓶颈，再决定下一项局部迭代，不启动无界搜索。

后续有限迭代保持用户明确要求的单模态输入边界：
[`v2`](../experiments/configs/physiology_semantic_tokenizer/dual_component_refinement_v2.yaml)
比较原有六通道五频带 EEG 信息、局部时序编码、已知区域/时间元数据和延长训练；
[`v3`](../experiments/configs/physiology_semantic_tokenizer/dual_component_refinement_v3.yaml)
比较观测前端加元数据及较短训练的正则化时序编码，并复核选中方案的两个额外种子。
不以不同种子当作独立被试，也不将验证集选型后的重跑称为独立数据确认。
即使结果未通过全部预定阈值，也保留这组优化稳定性复核；通过条件不变。

## 30 秒全局交互与潜在词表健康度起点（2026-11-05）

按本次设计讨论，后续 tokenizer 的验证起点采用 **30 秒序列、2 秒 patch、
全局时间交互、潜在表示硬量化**。每个电极或光学位置、每个 S/O 分支、每个
patch 只输出一个整数 ID；fNIRS 同一位置的 HbO/HbR 联合输入与重建，不再按
色团拆分词表。S/O 各有独立编码器、码本和解码器；解码器只读同一分支的
15 个硬 ID，输出两条色团波形。两分支均读取实测观测，teacher 的 S/O 仅为
监督目标。重建损失与结果分别检查 S/O × HbO/HbR，不用总误差或码字使用率
掩盖单分支、单色团失败。该起点是待验证的工程选择，不是已通过的最终架构。

这里的全局交互首先限定为同一位置、同一模态内的 15 个时间 patch，编码与
解码均可利用完整序列；不加入跨模态或跨 S/O 解码。token 每 2 秒产生一次，
不等于仅依赖那 2 秒观测。后续时序、掩码及方向分析必须计入完整输入支持；
不能直接遮住一个 ID 后把其他已读取该片段的 ID 当作无泄漏上下文。已有离线
滤波与参考支持另行保留。联合双色团词表允许学习幅比、符号和时序关系，不
强制 HbO/HbR 负相关，也不保证解码后满足完整物理动力学。

EEG 电压 S/O 监督的设计选择由下文的独立目标可行性试验持有；本节不提前
选定其候选，也不将历史修正电压换名为新目标。首轮实测健康检查只使用已经
冻结的 Hb S/O。EEG 的 200 Hz、2 秒 patch 接口先以合成信号验证，不能把这项
软件证据称为 EEG 实测码本健康或新版监督验证。

首轮合同为
[`dual_component_latent_health_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_latent_health_v1.yaml)，
复用同一双分量 runner。只读取已完成公开父运行中训练和验证被试的三 ROI
身份、保留的 90 秒 Tak 目标与训练观测尺度，截取事件 [-5,25) 秒形成 30 秒
序列；不读取测试，不重新拟合 Tak。父运行失败与本次新增读取/数值失败均保留。
物理模型条件分解继续作为标签，不称生理真值。

固定比较连续模型、单层 K128 和单层 K512。三个种子分别做连续预热，再从
同一种子的同一预热参数开始等暴露继续训练；最终连续与硬模型均为 3000 步。
训练潜变量初始化码本，采用现有 EMA 与有限码字复活，最后 500 个量化训练步
停止复活。每分支每 patch 的名义码率分别为 7/9 bit；不携带连续幅度、均值或
原始波形旁路。训练、码本初始化与尺度均不使用验证标签统计；验证只作本轮
固定配置的描述性评价，不按结果继续扩展候选。

主要检查冻结模型在完整训练/验证人口中的占用、有效码数和最大码概率，按
两库及 S/O 分别判断；阈值继承已有健康筛查。独立被试/记录覆盖、潜变量与
码本参与秩、近重复、复活历史及固定中央 patch 的上下文替换作为诊断。
同时报告连续到硬量化的误差增量，S/O/总重建按 HbO/HbR 分列，逐记录、逐被试
等权汇总；窗口和种子不充当独立被试。本轮不做词表生理资格判定或掩码预训练。
先软件/合成检查，再通过持久监督实测读取及 GPU 吞吐，按资源竞争和实测结果
确定有界并发，完成固定矩阵后停止。执行状态和数值结果分别由注册表和原运行
摘要持有，不在本节复制。

## 电极级条件电压目标与 Hb 跨 patch 连续性（2026-10-29）

按用户本次明确要求，全面应用保留 Tak 代理引导的原始 EEG 幅度修正，并开展
tokenizer 训练测试；既有 SSM 参数、求解器与拟合结果保持冻结。
[`dual_component_voltage_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v1.yaml)
持有有限预算、划分、修正强度回退序列、码率和选择规则。同一双词表入口新增
`voltage` / `voltage-one` 阶段，先软件与合成检查，再生产完整公开开发目标、
实测并发、训练固定候选、验证选型和种子复核，最后评价冻结选择及匹配基线。
保留原始计划与 teacher 失败；不因修正失败更换试次或重新拟合 teacher。

同一试次三个 ROI 的投影联合约束唯一电极集合；在既有 90 秒原始支持上计算
对数功率的最小改动，平滑增益只作用于目标 20 秒及两侧过渡。修正后重新执行
原功率、参考与 PCA 算子，报告实际代理误差、增益裁切及回退强度。电压 S/O
表示条件修正信号及其原始电压差，不冒称 SSM 直接生成的电压或真实神经／噪声。

EEG 使用原生电压输入、独立 S/O 分支和 100 ms 输出，比较每分支两／四层
128 码字，明确为每电极 280／560 bit/s。Hb 保持每 2 秒 28 bit，比较独立
2 秒拼接、带有符号边界差分监督的拼接，以及 4 秒输出／2 秒步幅的固定重叠
解码与重叠一致性训练。解码只读同分支硬 ID，不能读取实测边界或连续均值。
直接观测同码率量化、零电压修正和解码后平滑为分列基线。

选择只用验证集：各库 HbO/HbR 及联合 S/O/总 NMSE 相对拼接基线最多增加5%，
在可用健康候选中选择边界误差最低者，预定目标为至少降低30%。该条件是工程
筛查，不是统计非劣效检验。选中 Hb、拼接基线和 EEG 各补两个种子；公开测试
只评价冻结选择和这些基线。指标按案例、记录、被试逐级汇总，种子在被试内
平均，配对区间以完整被试重采样；无多重比较显著性主张。此次不启动新的掩码
预训练，不将 tokenizer 重建改善表述为下游或跨模态增益。

[`解码合成 v2`](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_v2.yaml)
保留 v1 全部训练和负结果，只把 v1 已列出的固定 100 ms 高斯解码平滑纳入部署
选择；各库、各色团和 S/O/总误差容限不变。该选择仅使用开发验证结果，在 v1
已开始公开测试目标生产、尚未开始模型测试评分时冻结，并保留当时的源阶段。
因此不声称在首次测试信号读取前完成新的方案登记，也不作新确认性结论。
v2 等待 v1 全部完成后只读复用目标、模型、整数词表和既定测试评价，重新汇总
所选解码方式；不新增 SSM 拟合或 tokenizer 训练。固定合成仅线性组合同分支
码字，sigma 为一个 10 Hz 样本、支持截断为四倍 sigma；每个码字的输出支持
两端各增加0.4秒，码率不变，后续掩码需计入该输出支持。

## EEG-O 条件电压残差重建改进（2026-10-29）

[`dual_component_eeg_o_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_eeg_o_v1.yaml)
持有本轮有限预算与选择规则，同一入口使用 `eeg-o` / `eeg-o-one`。仅复用电压 v2
保留的训练、验证目标；既有 teacher、S/O 定义和训练观测尺度冻结，本轮不读取
测试分区。先做固定训练 patch 的原结构 O 分支过拟合检查，以及训练 O 真目标的
同码率直接压缩参照，分别定位学习能力与量化容量。后者不作为可部署候选。

候选以原生观测的五频带电压和对数功率估计限幅、平滑的增益，再在编码端生成 O。
比较单电极短增益上下文、单电极长增益上下文和同模态多电极长上下文；不读取 Hb
或 teacher 侧信息。所有候选的零相位频带滤波均使用保留的完整20秒输入，所以
增益网络的1.9／5.9秒上下文不等于 token 的原始输入支持，后续掩码须覆盖整段。
损失使用冻结 O 方差归一化的波形误差与有符号差分误差。连续预测再由训练预测
拟合四层128码字，O保持每电极280 bit/s；解码仅整数查表求和，不携带原始载波
或连续增益。原 S 分支冻结，其与新 O 相加的误差单独作为诊断，不替代 O 保真。

主比较按每库验证被试等权的 O MSE／零预测 MSE，选取最差数据集比值最低的
候选，并完成固定两个额外种子复核。相关系数、RMS比、有符号投影、量化增量、
边界误差和码字使用率共同报告；全部失败与负结果保留。阶段结果、运行进度和
科学判断由统一研究状态及原运行证据持有，此处不维护另一份结果表。

[`参考信息 v2`](../experiments/configs/physiology_semantic_tokenizer/dual_component_eeg_o_v2.yaml)
在同一训练/验证身份与 O 目标上只读复用 v1 频带缓存和容量诊断，补充原生成器
保存的 EEG 观测代理及固定训练投影。输入只取 `original_proxy`、`projection`；
不取同文件中的 Hb、teacher 目标或修正后代理。该观测代理继承原90秒 EEG
预处理及5秒参考，模型输入范围据此扩大并明确标注。两个固定候选分别使用短
及整段20秒增益上下文，估计三个 ROI 修正方向，经训练投影伪逆映射到各电极
频带增益，保持原限幅及100ms平滑。仍只以 O 波形及差分监督，码率、验证选择
口径和两个额外种子复核不变；不将额外参考信息的收益称为等输入模型增益。

## EEG 电压域分解与独立干预验证设计（2026-11-03）

本节完成方法设计，候选标识为 `TK-EEG-WAVE-SEPARATION-v1`；尚未形成可执行
配置，不记录实验执行状态或科学结果。本次只核对代码、合同与登记摘要，没有
读取实测数组、启动训练或变更既有目标。后续实现使用独立版本，保留原电压
修正路线作为对照，不改写已完成的目标、配置和证据。

### 现有路径与可检验主张

`src/data/dual_component_tokenizer.py::voltage_proxy_correction` 构造的目标为

\[
S_{\rm old}=X+\sum_b(e^{g_b}-1)B_bX,\qquad O_{\rm old}=X-S_{\rm old}.
\]

\(X\) 是原生电压，\(B_b\) 为五频带滤波，\(g_b\) 是由冻结 Tak 代理目标与
观测代理之差得到的平滑、限幅对数幅度增益。三个 ROI 将每个时间箱的 16×5
个特征压到至多三维；伪逆先求最小范数 log-power 改变量，后续平滑、限幅及
载波合成不保证最终电压改动最小或精确达到目标。接受修正只要求重新提取的
代理误差下降，不验证神经来源。`voltage_train` 的两分支实际都读取原始电压，
S/O 是监督目标；其后对训练预测分别量化。现有整数解码没有原始载波或连续
增益旁路，载波混淆发生在目标构造中。

\(X\) 与 \(-X\) 的平方功率完全相同，电压极性却相反，故功率/Hb 不能唯一
决定电压相位。未耦合 Hb 的活动也可能是正常神经活动，任务相关伪迹也可能与
Hb 同步。因此，本设计分别检验**净化、神经波形保留、Tak 的额外价值**。
S 是待验证的先验支持传感器域电压分量；O 是补充残差，可能含未建模神经活动、
系统性成分及模型失配。不能把 O 定义成全部非神经信息，也不宣称 S 是脑内
电流源。“原始 EEG 形式”指保留电压单位、电极、参考、采样轴和正负波形；
估计出的 S 不称为未经修改的原始实测值。

### 主结构与输入合同

\[
\begin{aligned}
S_v&=F_\theta(X),& O_v&=X-S_v,\\
I_S&=Q_S(S_v),& I_O&=Q_O(O_v),\\
\widehat S&=D_S(I_S),&\widehat O&=D_O(I_O),
&\widehat X&=\widehat S+\widehat O.
\end{aligned}
\]

多电极波形提取器 F 属于编码端，只读 EEG 和已有电极/数据集身份，不读 Hb、
EOG、任务、被试 ID 或逐样本 teacher 目标。首版复用当前两个数据集的 16 个
唯一电极，以 2 秒、200 Hz 上下文输出中央 100 ms 的全电极波形，按 100 ms
步幅拼接。为完整生成事件相对时间 [-30,60) 秒的 S，本设计要求读取
[-31,61) 秒原生 EEG，两端各1秒仅作上下文；Phi仍只在中央90秒输出上计算。
92秒原始可见支持与旧合同不同，后续可执行版本须显式登记该支持并经既有
reader核验；缺失边缘者记为输入失败，不补零或复制实测值。所有基线采用
相同支持，完整计划分母保留。增加的多电极信息也须在对照中匹配。

source 量化器实际接收 S 的电压，observation 量化器接收残差电压，功率不
成为替代输入。冻结 F 后直接复用 `WaveformDictionary`，避免再训练一对
可以改变归属的学生。每电极、每分支四层、每层128码字，每100 ms为28 bit，
即每分支280 bit/s、S+O合计560 bit/s/电极。解码只查自身整数 ID；可使用
冻结训练尺度恢复电压单位，不允许逐窗均值、幅值、载波或增益侧通道。

连续闭合 \(S_v+O_v=X\) 是恒等式，不是分离证据。硬解码 O 不得写成
\(X-\widehat S\)。语义由编码端学到，VQ 负责保留，不宣称离散化自身发现
物理来源。连续与硬 ID 必须分别验收。

首版不增加硬约束 \(S=Lq\)：模板 leadfield 若覆盖全部可观测电极空间，
该约束不能排除伪迹；强行降秩又可能删除合法神经活动。当前缺少个体头模型，
不为获得“物理”命名增加这一层。空间结构用于多电极预测及跨混合条件验证，
展示有身份依据的电极场，不将其画成已定位的皮层来源。

### 分阶段训练：先确定电压语义，再检验先验收益

先训练无 Hb 监督的 F0。完全合成数据为 X=N+A，N 是已知生成神经电压，A
是已知加性污染。N 包含有/无 Hb 耦合的振荡、事件瞬态、相位变化和多电极
混合；包含 A=0 正控制。半合成用实测基底 B 加入已知 A，只把注入变化当
真值，不把 B 称为干净神经电压；另注入已知神经变化 U，检验保留性：

\[
\begin{aligned}
\mathcal L_0={}&\mathbb E_{N,A}\|F(N+A)-N\|_{\sigma_X}^2\\
&+\mathbb E_{B,A}\|F(B+A)-F(B)\|_{\sigma_X}^2\\
&+\mathbb E_{B,U}\|F(B+U)-F(B)-U\|_{\sigma_X}^2.
\end{aligned}
\]

范数指按每数据集/电极冻结训练观测 RMS 缩放后的有效样本均方；三类样本
分开等权平均。绝对神经目标及无污染正控制排除恒零解，污染约束排除原样
输出解。评价的基底、供体、被试/记录、生成种子及至少一种生成机制与训练
隔离。观测完全相同却要求不同来源的案例属于不可识别边界，不能强配互相
矛盾的标签，要求网络同时恢复。

然后固定 F0，在训练被试上拟合低容量 EEG→驱动读出 g。每 ROI 对原有六
电极的同频带参考后 log-power 等权平均，复用五频带、100 ms功率支持及
5秒参考，尺度仅拟合训练数据。为定义零/极低输出的log功率，每数据集/电极/
频带冻结 epsilon=训练原始正功率中位数乘10的负8次方，使用log(max(P,epsilon))；
epsilon的单位是原生电压单位平方，记录触发比例，不把触发位置默默丢弃。
训练参考功率全零或缺失则该坐标准备失败；本数值约定只属于新版本，不能
回写旧目标。每数据集/ROI只有五个权重，无截距：

\[
r_r(t)=\sum_{b=1}^{5}w_{r,b}z_{r,b}(F_0(X);t),\qquad
\widehat H=\mathcal H_{\mathrm{Tak},\theta_0}(r).
\]

H 是既有缓存坐标中的 HbO/HbR，r 是当前 Tak 核心接受的驱动坐标。
g 不读时间/任务标签，不含可调滞后、逐窗增益或自由驱动序列。Tak 参数、
Hb 观测变换和尺度固定；五个权重以 L2 正则拟合，在训练内被试划分选择
\(10^{-4},10^{-2},1\) 三个无量纲惩罚。旧联合拟合驱动不是 g 的真值。

主设置在中央90秒的首端（事件前30秒）固定静息四状态 \((s,f,v,q)=(0,1,1,1)\)，实现保持
p=v，用事件前30秒 EEG传播，评分中央20秒。所有方法共享初态、时钟和
参考，不根据当前目标窗 Hb 重新拟合状态。冷启动偏差作为限制保留；非法
状态和积分失败使用现有检查，不裁切或删去失败后只汇总成功者。

最后冻结 g/Tak，各候选从相同 F0 开始，只调整 F：

\[
\mathcal L_\lambda=\mathcal L_0+
\lambda\operatorname{MSE}_{\sigma_H}
\left[\mathcal H_{\mathrm{Tak},\theta_0}\big(g(\Phi(F(X)))\big),H\right],
\quad\lambda\in\{0,0.01,0.1,1\}.
\]

Phi 为上述五频带特征，sigma_H 为训练 Hb SD；辅助损失按数据集/ROI/色团
分层等权。两个训练被试折使用另一折拟合、冻结的 g 计算辅助损失；最终
验证/测试读出在全训练 F0 输出上拟合一次并冻结。不交替优化 F/g 追逐 Hb，
不读取评价被试的 Hb 来适配。该损失约束血流解释能力，不强制所有神经活动
都有 Hb 对应，也不强制 Phi(S) 等于旧 teacher。它仍可能导致过度清理，
必须通过未耦合神经正控制，不能因写入 Tak 方程而免检。

梯度路径复用 `shared_driver_rk4.py::tak_forward` 的驱动 Jacobian，与固定
Hb观测变换组成向量-Jacobian积，再连接线性滤波、分箱平方、log和F的导数；
此桥接尚未实现，必须用方向有限差分和独立回放核验，不能将detach后的NumPy
输出冒作可微损失。g从零权重的合法静息轨迹开始拟合，非法提议不接纳。
F更新若产生非法轨迹，回滚参数及优化器状态，在同一批次按步长1、1/2、1/4、
1/8重试；全部失败则该候选终止并保留原因，不跳过样本、删掉Hb项或填零梯度。
评价失败同样保留原分母。

F 的局部上下文与 Hb 辅助读出的完整支持分别记录：92秒原生输入产生中央
90秒S，再作90秒零相位滤波及参考，因此主评价是离线处理后信号预测，不称实时/前瞻。
未来 completion 要在完整处理链之前遮挡，100 ms token 掩码不能替代
对更长原始支持的隐藏。冻结 F 后只在训练 S/O 输出拟合字典，不再用总
重建损失反向移动分解边界。

### 必要的干预与对照

| 改变的量 | 固定的量 | S 应有响应 | O 应有响应 |
| --- | --- | --- | --- |
| 已知加性污染 A | 神经电压、Hb | 基本不变 | 增加 A |
| 未耦合 Hb 的神经变化 U | 污染、耦合神经分量、Hb | 增加 U | 基本不变 |
| 耦合神经包络及正向生成的 Hb | 污染、其他神经分量 | 保留真实变化 | 不吸收该神经变化 |
| 全电压极性翻转 | 功率与相应 Hb | 正确翻转 | 对应翻转 |
| 独立系统性 Hb | 全部 EEG、冻结模型 | 完全不变 | 完全不变 |
| 任务共同驱动的污染与 Hb | 神经真值 | 不因 Hb 相关保留污染 | 恢复污染变化 |

神经正控制与污染反控制至少有一组匹配频带、RMS、持续时间和空间平滑度，
避免凭振幅阈值通过。Hb相关污染由共同任务/系统性驱动产生 EEG 包络和滞后
Hb，在评价中改变耦合强度、方向和滞后。用非 Tak 血流核检验模型失配，不
仅测试训练模型生成的真值。削顶、断连和接触阻抗乘性变化不是加性污染，
单列分布外压力测试，不承诺恢复已丢失的信息。

实测只用真实登记 EOG 等独立参照及事件/低污染区间，不把额区 EEG 当 EOG，
也不把回归清理结果当神经真值。没有可靠 EMG 的数据不作实测肌电净化结论。
初始范围保持已有 Single-Trial/Simultaneous 两库及中央身份/通道合同。

最小对照为 raw、S=0、旧增益修正及学生、同结构/同多电极输入的 lambda=0、
匹配平滑程度的简单清理、同总码率直接观测 VQ。同步 EOG 回归另列额外传感器
参照，旧单电极学生另列上下文差异。普通观测 VQ 只评价其完整重建，不能给其
任意两个码流强加 S/O 语义。先验贡献必须与同数据同结构的无 Tak 版本比较。

冻结表示后另拟合训练期低容量 Hb 评价读出，与训练辅助读出的复现误差分列。
在独立被试比较 raw、无 Tak、旧增益法、source-only、observation-only、
S+O及同码率观测 token；基线包含可用 Hb 历史和任务/时间控制，给所有方法
相同信息。任务仅进入该诊断基线，不进入 tokenizer。加入同任务/被试/会话
内跨试次错配、时间移位对照，保持有效支持。若声称耦合增量，继续遵守现有
proper-score/null 合同；同窗联合拟合后的误差或相关不提供该证据。

### 指标与有限选择规则

分支 B 取 S 或 O，定义已知干预下的响应
\(\Delta B_\delta=B(X+\delta)-B(X)\)。连续输出与硬 ID 分开计算：

\[
\begin{array}{ll}
L_{A\to S}=\|\Delta S_A\|^2/\|A\|^2,&
E_{A\to O}=\|\Delta O_A-A\|^2/\|A\|^2,\\
E_{U\to S}=\|\Delta S_U-U\|^2/\|U\|^2,&
L_{U\to O}=\|\Delta O_U\|^2/\|U\|^2.
\end{array}
\]

分母为同一有效支持的已知干预电压平方和；零注入单独评价，不除零或逐输出
重归一化。连续严格互补分解中两项污染指标代数相同、神经两项亦相同，不能
算成四份独立证据；硬量化后分别计算可定位分支失真。

完全合成另外报告 S 对 N、O 对 A 的绝对 MSE，以及净化收益
\(\operatorname{MSE}(X,N)-\operatorname{MSE}(\widehat S,N)\)，排除固定偏移。
连续互补分解中 O-A=-(S-N)，两项绝对MSE也相同，并列只是帮助读图，不是
两份独立证据；量化后分别验算。
同时报告有符号相关、幅值比、峰时误差、频谱/事件波形保留和总重建误差。
S/O 不必正交，不用二者平方和计算“来源能量百分比”。图中的
\(\langle\Delta B_\delta,\delta\rangle/\|\delta\|^2\) 是可超出0至1的
有符号投影，必须与剩余波形误差并列。

以下为预先建议的工程筛查数字，不是生理界限或实验结果：

1. F0须先具有净化资格：在预定可识别加性污染的SNR 0、5、10 dB条件下，
   每类污染、每种强度的独立实例均值MSE(S,N)/MSE(X,N)≤0.8，污染进入S
   的平方响应比均值≤0.25、P90≤0.5；未耦合神经恢复误差E(U→S)均值≤0.1、
   P90≤0.25。无污染source NRMSE中位数≤0.05、P90≤0.10，分母为N的RMS。
   F0未通过则停止词表阶段；不能以保留F0为由接受恒等映射。SNR −5、−10 dB、
   非加性污染和观测等价反例预先列为压力/识别边界，不要求通过上述门槛，
   但全部显示，结论不得扩展到未通过的条件。
2. 相对 F0，每库、每类干预的神经恢复 NMSE 及两个方向的泄漏/恢复误差
   的均值及P90分别不得超过 \(M(F_0)+0.05M(F_0)+0.002\)，M分别代表该
   汇总统计量，且仍须通过第1项绝对门槛。绝对容差避免近零比值失稳。
   数值失败不能通过删除样本豁免。
3. 满足保留性后，非零 lambda 的冻结 g/Tak 验证 Hb 误差须在每库比 F0
   至少降低2%。多个通过时选最小 lambda；无通过者保留 F0，并明确未证实
   Tak 辅助收益。这是开发规则，最终先验贡献仍由独立评价和配对区间判断。
4. 固定码率后硬 ID 重新通过第1项全部绝对门槛和干预检查；相对选中连续 F
   的均值/P90增量不超过上述同一容差。越界只支持连续分解，不以总误差或码本健康替代；连续方案均
   失败则停止词表阶段，不扩大无界搜索。

基底、供体、被试/记录、生成种子严格分组，实测按窗口→记录→被试汇总，
最后被试等权、各库分列、整被试配对重采样；重复窗口、污染版本、训练种子
不是独立被试。完整计划分母、失败和成功子集并列。现有公开 test 已有探索，
不能包装成新确认集；新确认需要明确的未参与选择评价面板，不由本节扩大
数据访问范围。

若 N1+A1=N2+A2 而 N1≠N2，同一个 EEG-only F 不可能同时恢复两个目标，
两目标平均平方误差下界为 \(\|N_1-N_2\|^2/4\)（按所用均方归一化）。
该反例用于展示识别边界，不要求模型完成不可能的分离。支持的结论限定于
声明的混合假设和已验证机制，不能升级为无条件神经真值恢复。

### 三张主图与一张解释图

**主图一：神经/污染交叉混合与 token 交换。** 构造
\(X_{ij}=N_i+A_j\)，i,j各取1或2。两行是两段已知神经电压，两列是两种
污染；每格同轴显示输入、N真值、连续S和硬IDS，下面显示A真值和硬IDO。
相邻面板画污染交换应产生的S零变化，以及神经交换应产生的S真实差值。
包含无Hb耦合的神经变化条件；半合成基底不能冒充纯神经真值。

再从 X11 取 S ID、X22 取 O ID，直接解码相加，与 N1+A2=X12 比较；
原始波形不进入输出。分别显示所取S对N1、O对A2的误差，再给交换重建误差
和直接编码X12的误差，区分组合失真与普通量化损失。交换合成可能因两分支
偏移抵消而良好，必须与分支真值误差及干预响应共同检验词表归属。

**主图二：污染路由与神经保留的响应曲线。** 四列分别为普通污染、Hb相关
污染、耦合神经变化、未耦合神经变化。横轴已知干预幅度，纵轴分别为S/O
有符号响应系数和波形误差；标出理想0/1响应与全部基线。首版污染压力取
SNR −10、−5、0、5、10 dB和无污染，使用固定基底RMS，注明工程压力范围。
若模型只把所有波形缩小，神经保留列将直接暴露失败。

**主图三：Tak 收益是否以神经失真为代价。** 并列展示净化收益、未耦合
神经恢复误差、污染通过量、独立Hb预测增量。每点是独立生成实例或被试
汇总，同一单元配对连接；按数据集/未见机制分面，连续与硬ID分列。零线、
区间、失败数和完整分母可见。只有Hb改善不能抵消神经删除或污染保留。

**解释图：相同功率不能确定电压。** X与−X的电压相反，完整算子的功率
代理相同，说明极性与快相位来自 EEG；Tak约束的是神经血管观测相容性。

样例在评价前按固定基线误差选择中位、P90和首个失败，尽量不同被试；不能
按候选收益选漂亮样例。N为黑虚线、S为蓝实线、O为橙点划线、输入为灰线，
同时用线型区分，坐标、单位、参考和色标固定。报告/PPT图和LaTeX公式按
最终尺寸200–300 dpi PNG嵌入，正文保持可选择；本节未生成实验结果图。

### 实施落点与依据

数据身份、干预、尺度及支持放在 `src/data/dual_component_tokenizer.py`；
F与现有波形字典放在 `src/tokenizers/dual_component_tokenizer.py`；差分
指标与被试汇总放在 `src/analysis/dual_component_tokenizer.py`。新版本接入
既有双分量 runner，不新增平行 launcher。旧S/O标签不作为新分解真值。
先完成符号翻转、已知混合、幅值缩放、单模态输入隔离、整数独立解码和全链
支持的synthetic检查，再按新版本合同安排后续实测。

[Tak原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC4401444/)提供隐神经
状态到fNIRS的生成模型，本设计不把血流核心扩称为头皮电压逆模型。
[EEGdenoiseNet原始研究](https://arxiv.org/abs/2009.11662)提供EEG与EOG/EMG
组合的去噪基准思路；本设计借鉴成对注入，不自动接入新数据或把参考片段
视为绝对生理真值。[Kirilina等原始研究](https://pubmed.ncbi.nlm.nih.gov/22426347/)
展示任务诱发的系统性/头皮血流伪迹，支持设置Hb相关污染反控制，不能把
任务锁定的跨模态相关性单独用作神经来源证明。

## 物理模型独特贡献与非任务证据设计（2026-11-04）

本节将“物理约束天然产生更好的特征”改为可被否定的研究假设，补充上一节
电压分解方案的归因设计，不记录新的运行或结果。上一节 Tak 对无 Hb 辅助
F0 的比较及 lambda 选择只作为工程筛查；单凭它不能区分额外 Hb 监督、
一般平滑/动态约束和 Tak 特有结构。本节收紧物理贡献的判读，不回写历史证据。

### 三个层次分别判断

| 待验证层次 | 必要证据 | 不能自动推出 |
| --- | --- | --- |
| 模型相容 | 物理域、数值积分、方程及训练约束通过 | 真实来源、独立预测优势 |
| 表示/预测效用 | 未参与选择的测量或干预上，胜过同信息同预算替代，且保留所声明神经信息 | 该模型是真实生成机制 |
| 生理/机制解释 | 竞争模型给出不同的未见响应，独立观测支持所选机制；被解释状态/参数有识别依据 | 仅凭参数命名或重建好获得因果解释 |

物理模型限制可选解并提供归纳偏置，优势可能是有限样本下更容易读出有用
信息、抑制指定污染、节省码率或获得可验证预测，不必体现为任务分类提升。
冻结模型条件下，EEG-only token是原始EEG的确定性函数，不能创造原始观测
之外的新测量信息；“更易读出”不能写成“总信息增加”。没有任何一项优势
由加入方程本身保证，适用范围和负结果同样是结论。

### 最小四臂：主比较保持 Hb 监督不变

| 臂 | 共同电压分解/干预监督之外的变化 | 用途 |
| --- | --- | --- |
| A：F0 | 无 Hb 辅助 | 总体参照，暴露加入辅助项的整体作用 |
| B：普通正则 | 无 Hb，加入平滑/粗糙度正则，训练内选择强度 | 检查收益能否由一般正则解释 |
| C：经验动态 | 同 Hb 辅助，使用训练拟合的 FIR 或低阶稳定线性状态空间读出 | 排除额外监督、延迟和动态记忆的替代解释 |
| P：Tak | 同 Hb 辅助，使用 Tak 读出 | 主要检验 P 相对 C 的结构收益 |

C 的 FIR/稳定 SSM 在内层训练/验证中按相同预算选择，不能事后选取最弱者。
四臂属于初筛：只胜线性C可能来自一般非线性。若要归因到物理特有结构，
必须把同信息、合理调优的小型非物理非线性读出纳入C，并实际比较稳定的
机制近邻；不能在未完成这些比较时授予物理特异性。
例如 Tak 线性化/去除黏弹性后的版本可检验具体动态是否必要，但线性化模型
继承了 Tak 局部结构，不属于完全独立的经验模型。

实测中近邻是竞争假设，不能预称“错误物理”。合成时才可明确哪项被设错。
改变全局符号、驱动尺度或状态坐标而能被 g 完全吸收的变换不是有效反控制；
须核验重拟合允许参数后仍存在不同的可观测响应。明显失稳或未调优的错误
模型也不是合格对手。上述臂的相邻差存在交互，不能相加为各项贡献百分比。

各臂统一原始可见支持、训练被试、Hb目标、波形F、分支码率、尺度、初态信息
与验证预算。C/P采用同一流程：固定F0拟合辅助读出，冻结读出，再从F0调整F。
匹配可训练参数数量之外，记录动态阶数、响应自由度、平滑程度、目标初始
拟合质量、作用于F的梯度量级、收敛及总费用；相同lambda不表示约束同强。
完整费用含teacher/读出拟合、积分和搜索。等预算比较支持工程效率；某臂未
收敛不能直接解释为该机制错误，也不为凑算力执行无意义运算。

### 核心实验一：预测模型未看过的测量

冻结表示，在被试分离的评价集隐藏预定Hb端点，采用相同信息边界和实际测量
评分。各表示使用共同、非Tak专属、相同容量/调参预算的训练期评价读出，
它与各自辅助教师分开；包含raw、S、O和S+O。另列模型自己的原生预测，
不能以“更易被Tak解码”替代共同评价结果。

比较超出同模态可见历史、任务/时间模板以及同信息经验跨模态模型的增量，
并用相同任务/会话条件下的配对破坏和时间移位排除共同事件结构。隐藏目标
不能进入PCA、尺度、参考、状态拟合、预处理或teacher；若只能保证处理后
目标隔离，则如实限为该坐标的离线补全。独立目标区间的原始隔离或未来预测
需要另核验完整滤波支持，不能由隐藏几个token自动获得。

主端点先采用固定训练尺度下、各库分列的被试配对测量误差，主比较为P对C。
若已建立预测分布，可评价NLL/CRPS及覆盖率和区间宽度；目前的工程MSE、
优化Hessian、多种子离散或有限备选轨迹不能直接称为校准概率分布。
O仍可能保留可预测的正常神经或观测信息，不把“O不能预测Hb”设为成功条件。
Hb可预测性改善必须与下一组的神经保留/污染抑制并列，不能互相抵消。

### 核心实验二：预测正确的变化，而非只画相容的曲线

复用上一节已知污染和神经干预矩阵，至少包含：加污染、加不耦合Hb的神经
变化、污染与Hb共同变化。冻结模型，直接评分真实波形差与S/O解码变化，
报告污染进入S、神经进入O及绝对波形误差；连续和硬ID分开。核心问题是P
是否比C更选择性地保留/排除信息，而非是否都满足各自训练约束。

训练/评价隔离基底、供体和生成机制。自家Tak方程生成、自家Tak反演属于
软件/匹配模型检查；更精细或独立积分只排除数值错误，不等于独立机制验证。
至少加入不同神经混合及不同血流生成机制，实测以独立传感器/预定事件为
外部参照。数字注入证明指定混合条件下的响应，不称人体因果干预；实测自然
事件的关联也不直接等同随机干预。

如需解释真实参数改变与观测失配，单列已知真值条件：保持生理状态改变观测
增益，与保持观测改变血流时间尺度，检查变化被归入驱动、状态、参数还是
残差。固定了某个参数的模型没有该参数的估计恢复结果。当前EEG-only F在
单独改变Hb时不变属于结构保证，可作输入隔离检查，不能算Tak独特优势。

### 核心实验三：动态外推与失效边界

优先使用现有Tak确实描述的动态：固定机制、参数、主设置首态和观测，改变
训练未见的驱动持续时间、频谱及响应幅度，检验冻结模型对新响应的误差和
神经保留。核心测试从未见EEG波形估计驱动、预测未见Hb，不提供真驱动或
真实参数；至少在一种非Tak生成机制和独立实测条件下重复。提供真驱动/
参数的同Tak外推仅作匹配模型与积分检查。共同评价器的表示效用与模型原生
读出的动态外推分列，不能互相替代。

初态变化另列：未提供初态信息且仍用静息首态时，是未知初态/冷启动压力，
不能直接归咎动力学方程。若允许适配，所有方法只能从相同可见前缀估计，
规则在训练期确定；给定真初态另列条件上界。随后逐项引入机制失配：额外
系统性Hb、耦合关系改变、不同血流核。
配对条件固定非目标因素、噪声实现和评价支持；分别画P相对C的收益随偏离
程度变化的曲线。不能预设假设已失效后物理模型仍须胜出。

该组区分有用范围、先验偏置和求解失败。若有独立校准的失配检测/区间机制，
再检验误差上升时是否及时报出不适用及其漏报/误报率；没有这种机制时，
只能报告性能退化，不能宣称模型自动知道自己不确定。

已知单位变换和正确逆变换是所有方法应过的检查，不是物理贡献。参考变换
要求解码波形按观测关系等变，不要求原始token ID不变；跨训练种子的码号
也不直接比较。缺通道等变换会丢失信息，不能要求完全不变。只有存在经过
核验的观测算子时，才追加“冻结机制、仅更换已知算子预测新测量”的条件实验。
当前Tak血流核心不提供EEG leadfield，不能据此承诺新脑电montage的源迁移。

### 补充优势：数据/码率效率与解释资格

核心实验后，按需要只展开一个资源轴。样本效率使用嵌套、被试级的Hb监督
数量，保持F0训练曝光和最终评价集不变；码率效率使用固定表示下的总S+O
比特数，包含所有层和任何实际传输侧信息。画达到相同隐藏测量误差、神经
保留和污染泄漏容限所需的被试数/码率。仅把S压得更简单而损失正常活动，
是率失真取舍，不是无代价优势。低数据时P好、大数据时C追平，可支持样本
效率，不必据此否定实用价值，也不能升级为机制真实。

参数解释需独立资格。示例模型 y=abx 中，数据只识别ab；真值(2,0.5)与
强对称正则稳定返回的(1,1)给出相同观测。重复性好、先验更窄或接近文献值，
都不能证明a、b恢复正确。应检查无先验数据项的剖面（其他未知量重新优化）、
先验敏感性、已知真值偏差、真实差异/排序保持和区间覆盖。有合法似然后才
称likelihood profile，否则沿用已有“目标剖面”命名。全固定Tak参数不能
成为个体生理参数恢复证据；参数不可识别也不自动否定可识别预测的价值。

### 优先比较、图形及结论边界

不合成一个任意“物理优势总分”。预先选择P对C的一个主要独立测量端点，
神经保留及污染归属为不可被预测收益抵消的条件；其他轴按声明主张分别
判读。若优势只在资源轴出现，结论限于效率。选择使用训练/验证，评价集
不用于选最优对照、指标、例图或偏离范围；多组确认性比较预先定义校正，
其余标探索。实测被试、合成独立生成实例为统计单元，保留失败与完整分母。
未达到预定实际收益或区间仍不确定时写“未证实优势”，不写严格等效；若
同监督替代更好且无已声明的其他收益，停用Tak项，不以物理命名抵消负结果。

建议四张主图：第一张四臂的被试配对效应，突出P对C；第二张已知干预的
真实/估计变化及分支误差；第三张机制偏离程度与相对收益，明确零线和失败；
第四张样本或码率学习曲线。参数解释另用数据项/先验项剖面与真值覆盖图，
不以聚类、注意力或更光滑的波形替代。图的单位、分母、连续/硬ID、完整
分母及来源按上一节和项目报告规则标明。

判读顺序：只胜过A，支持辅助方案整体有用；与C无可分辨差异，尚未证明
Tak特有收益；胜过C但与机制近邻相当，支持一类动态结构；独立预测和干预
响应均改善，支持Tak辅助方案相对已检验经验模型的效用。物理特有结构归因
还须实际完成非物理非线性及可区分机制近邻的比较；具体生理参数/因果机制
仍须额外识别证据。只在匹配模拟器上成功、Hb改善但神经保留恶化、先验越强越稳定，
均不足以跨越这些层次。

本设计复用既有[四套独立诊断](#拟合观测信息与生理恢复的四套独立诊断2026-10-21)
的预测、可识别性和归属思想；那些保留结果各有旧模型与数据身份，不能直接
改标为本候选的证据。新比较接入现有owner和版本化合同，本次不执行实验。
参数/预测分开判断可参考[Gutenkunst等原始研究](https://doi.org/10.1371/journal.pcbi.0030189)，
数据识别可参考[Raue等原始研究](https://doi.org/10.1093/bioinformatics/btp358)，
预测分布评价可参考[Gneiting与Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。

## EEG 空间与时间双尺度分离试验（2026-11-05）

本次按用户要求检验一个具体目标：S 减少主要伪迹与细尺度污染，O 同时承载
移出的主要结构和真实输入中已有的细密波动。版本合同为
[`dual_component_voltage_separation_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_separation_v1.yaml)，
沿用同一双分量 runner、原生电压读取边界和上一轮已核验的公开窗口。
运行与结果由研究注册表及运行内 summary 记录；历史目标和已完成运行保持原身份。

六名训练被试各自的首窗拟合全带协方差、原始五频带功率坐标、固定邻域预测器、
电压尺度与波形词表，末窗检验空间方向关联的符号和稳定性。五名验证被试的
两窗只作评价。重新拟合的 Hb-only Tak 状态作为慢引导，EEG 观测行全部隐藏；
失败 ROI 保留并屏蔽，不用联合拟合补齐。物理、经验 Hb 时序预测、EEG 自身
慢代理和共同相位打乱引导共享协方差、时间支持和空间算法，另有 PCA 与随机
方向对照。经验模型在首窗训练被试内留一选择 ridge，物理核心参数固定。

空间前端采用状态加权频带协方差相对同一全带协方差的广义特征方向，合并并
控制总秩。固定回投后，另比较对称邻域预测：每个成分使用前后 1–4 个采样点，
不读取当前值，系数和为一。S 为原投影与预测的固定混合，O 严格定义为原电压
减 S，因而同时包含空间补充和时间预测残差。该残差可能含神经信号，不能直接
命名为噪声真值。部署前端只读同一蒙太奇的 EEG，最长需要 20 ms 未来支持。

秩 4/8/12 与预测残差移出比例 0/0.25/0.5/0.75/1 的有限网格由独立预定
合成扰动选择，神经保留误差在目标中双倍加权。空间方向稳定性使用训练末窗；
扰动选择本身因前端线性而可直接计算，无需重复叠加天然背景。另保留固定
秩 8、比例 0.5 的容量匹配比较，不按验证图形是否明显调参。

在完整验证 EEG 背景中分别加入眼电形态、25–70 Hz 独立细噪声、二者组合及
10/35/55 Hz 神经样变化；幅度为首窗训练电压 RMS 的 0.1/0.25/0.5 倍，评分
为原窗口 [30,50) 秒。真实 EOG 只供独立诊断及跨验证被试的眼电供体，不进入
成分训练、选择或编码。三种已知生成器真值场景检验空间可分、细噪声共享神经
方向及伪迹与状态相关的失效条件；此处生成器状态是额外真值机制对照，并非
实测 Tak 状态或方法上界。窗口均方误差先在被试内、再在被试间等权，最后开方；
有符号干预投影与能量占比分开解释。配对区间以被试为重采样单位，属于冻结
配置下的探索性区间，未作多重比较校正。

在冻结前端后复用 128 项残差波形词表，每 100 ms、每电极、每分支输出四个
硬 ID，两个解码器只读取自己的 ID 和固定训练尺度。原始电压压缩对照使用
八级以匹配两支总码率；另保留物理 O 八级容量诊断，明确其码率更高。目标、
连续前端与硬重建分别展示，尤其直接核对 O 的细频段相关、误差及块边界失真，
避免把量化伪影当成成功分离的噪声。所有测试、失败与预定窗口完整保留。
这是一项有限公开开发实验，不扩展测试分区、不启动下游 benchmark，也不据
视觉反差认定天然神经真值或正式监督资格。

### 局部功率参考的单窗配对诊断（2026-11-05）

用户给出的条件功率参考方案按
[`dual_component_voltage_power_cap_v1.yaml`](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_v1.yaml)
作有限诊断。只复用上一轮点名的验证窗口及其眼电注入，训练统计使用原六名
训练被试首窗。三条联合 Tak 重建代理经冻结 log-power 均值、协方差和投影
提升为完整通道/频带参考；绝对基线、跨通道相关及代理频带之外的余项参考
同样冻结。没有当前窗口零空间功率、原始局部均值或带外电压的无条件旁路。

局部加权最小改动投影等价于白化后奇异值截断，保证所处理的局部载波块
二阶矩上界。它不保证来源可识别或扰动不变性：未越界的眼电可完整通过，
联合 SSM 的扰动响应也可抬高参考。原五频带及精确电压余项分别处理，非负
Hann 重叠合成后重新提取并记录功率越界，不把局部定理直接扩大为合成信号
的全局保证。容差主值及敏感性值在看本轮结果前固定，不按验证图形选择。

比较原目标、固定统计参考、冻结原 SSM 参考和重拟合 SSM 参考；配对神经样
变化独立重拟合以检验保留。眼电条件下，冻结参考的电压响应与参考变化导致
的附加响应分别列出。主要指标为已知眼电增量归一化的 S 变化；因 O=x-S，
O 与已知眼电的变化误差与该指标相同，不记为第二份独立证据。神经样增量
保留误差、原基底波形改变、失败及最终功率越界同时呈现。单窗结果不作
人群推断，不训练 tokenizer，不更改全量缓存的 τ/β 及个体校准条件。

按用户后续要求，[Hb 高权重版本](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v1.yaml)
将 EEG/HbO/HbR 标准化平方误差的数据项乘数设为 0.01/100/100；观测、
训练尺度、物理参数、驱动与初态正则及电压约束算法保持原定义。权重仅进入
求解目标，展示 NRMSE 和积分核验仍用原冻结尺度。原始、眼电和神经样干预
均重新拟合，不复用等权联合结果。此为同窗事后诊断，不视为独立确认；
不将代理稳定等同于最终 S 电压稳定。

[同目标延长求解版本](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v2.yaml)
只增大数值预算，保留原收敛与积分验收阈值。
[第二档权重](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v3.yaml)
将数据项乘数改为 0.1/10/10，仍只使用同窗配对及既定神经样对照，属于明确
记录的事后探索，不以成功子集替换前档的失败分母。
[数值续算版本](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_hb_weighted_v4.yaml)
校验相同科学合同后引用已验收拟合；局部投影改用通道归一化、Cholesky
白化及直接奇异值截断，以避免白化协方差的条件数平方。该实现与原加权投影
数学等价，不裁剪参考功率、不改变上限，最终重叠合成仍须重新检验功率界。
各档结果、数值失败和中断证据由既有结果索引链接，不扩展到全量缓存。

用户随后将这组权重选为[可复用的电压目标开发设置](METHOD_RATIONALE.md#hb-prioritized-joint-fitting-for-voltage-targets--2026-11-06)，
并要求另外四名被试各一个说明性样例。
[有限扩展合同](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_examples_v1.yaml)
在读取前选取 VP005、VP014、VP017、VP019 的原验证面板首窗，不按结果换窗。
同一已知 Fp1 眼电波形与绝对幅度、冻结参考和训练坐标用于所有样例，只运行
主容差及原始/眼电配对，不增加权重搜索。每窗三个基底 ROI 和一个受影响的
前额 ROI 独立求解；未收敛时仅允许一次同目标延长，原阈值及全部尝试保留。
展示完整 O 与注入波形，并同时展示 ΔO，区分背景残差与新增扰动的响应。
所有计划样例及失败均列出，挑图只用于展示；本扩展不替代既有神经保留对照。

按用户随后明确要求，[枕区补充](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_power_cap_posterior_v1.yaml)
固定为原 VP002 窗口的 O1 与 VP005 首窗的 O2，直接加入同一条已知眼电，
末行仅比较 ΔO 与注入。为使扰动后的枕区 EEG 实际进入联合拟合，在保留前额、
左运动区的同时，以 POOz Hb 和六个最近 EEG 电极建立枕区 ROI。相同六名
训练被试首窗拟合新的枕区坐标及 17 电极参考；验证输入不参与这些拟合。
原两个 ROI 的观测与权重逐点核验后复用，新增四次枕区原始/污染求解，失败
仅作一次同目标续算。范围为两个预定说明性样例，不新增选窗或权重搜索。
由于电极集合和训练参考也变化，本批与前额批之间的差异不能只归因于脑区。

## EEG 电压监督目标的有限可行性试验（2026-11-05）

本次按用户明确请求，测试附带的“物理功率先验＋独立伪迹协方差＋配对加噪”
目标构造。它是目标可行性实验，不启动 tokenizer/VQ 训练，不覆盖或重标历史
S/O 监督。[版本合同](../experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_teacher_v1.yaml)
经既有双分量 runner 执行；运行状态由研究注册表记录。

**范围。** 只用 Simultaneous 真实 HEOG/VEOG 及其对应的 200 Hz 原生 EEG。
从保留公开划分的六名训练被试、五名验证被试中，各选 n-back 第一段和最后一段
完整三 ROI 90 秒窗口；选择先于信号读取，两个窗口不重叠。完整分母包含读取、
拟合及生成失败。评分为窗口内 [30,50) 秒，滤波和教师可见完整 90 秒，参考为
[0,5) 秒；Hb 沿用保留缓存及实际采样时刻。重叠 ROI 合并为唯一电极集。
对照旧目标逐样本核验与原运行 float32 监督一致。测试分区及受保护材料不读取。

**估计器。** 第一版设伪迹条件均值为零。训练被试的 EEG/EOG 0.5–15 Hz
统计估计一个固定、带 ridge 的 EOG→EEG 混合；其局部复数谱外积提供噪声协方差。
训练中 EOG 活动较低的帧估计信号空间相关矩阵，20% 对角收缩保留满秩背景。
1 秒 Hann 窗、250 ms 步幅的 STFT 覆盖完整 0–100 Hz 并可逆；这里的谱估计
步幅不改变 200 Hz 标签采样率。每个频率的信号功率为平滑观测功率减估计噪声
功率并使用训练下限，随后按 K_S(K_S+K_N)^{-1} 处理复数电压。

另在已有五频带及 100 ms 均方支持上计算噪声扣除功率，减相同五秒 log-power
参考，重新拟合训练 PCA 和 EEG/Hb 尺度。固定 Tak 核心每窗重新估计驱动与初态。
物理先验以有权重的小矩阵融合只修正对应统计方向；其余信号方向继续存在。
物理误差尺度取训练各被试首窗 Hb-only 预测 EEG 的均方误差与训练代理方差
两者较大值；这避免仅凭当前联合拟合残差给先验过高权重。该值是保守工程
正则强度，不是独立真值或校准后验方差。物理失败单独计数并显式退回无先验。

**有限对照。** 原始 EEG、原实际 S 标签、EOG 均值回归、无物理 Wiener、
物理 Wiener、循环平移 45 秒的物理先验。旧标签保留其历史训练人口和坐标；
不能把它与新目标之差单独归因于物理模型。物理增量只在相同噪声估计、同一
新坐标和同一面板的有/无先验之间判断。零噪声证据、极性翻转、满频谱重建与
标签闭合由数值测试验证。

**已知干预。** 在验证窗口加入其他验证被试的实测 EOG 波形，经固定且相对
训练混合存在 30% 通道差异的空间映射形成已知眼电；供体不参与训练校准。
同时加入无 EOG 参考的 25–70 Hz 干扰作为覆盖范围负对照，以及应保留的 10 Hz
神经样 burst。三种幅度固定为训练电压 RMS 的 0.5、1、2 倍；完整数值数组
用于评分，不为绘图逐帧/逐窗重新定标。天然背景 EEG 及 Hb 保持不变。
眼电和肌电的配对标签按定义保持 S、令 O 增加扰动；该恒等式不是估计器成功。
另固定原物理预测、重新计算电压估计，测量 ΔS 相对已知干预的误差、以及
ΔS/ΔO 对干预的有符号投影。噪声应进入 O，神经样变化应进入 S；投影不是
能量占比，两路不要求正交。该诊断不包含对污染输入重新拟合 Tak 的响应。

**真值与统计。** 12 个独立合成评价种子提供隐藏干净电压，六个训练种子拟合
噪声模型；其中真值 log-power 先验明确标为 oracle 上界，不冒称实际 Tak。
实测部分只有注入变化已知，天然 S 不以“干净 EEG”命名。主指标先平均时间/
通道平方误差，再平均同被试窗口、最后等权平均被试并开方。干预误差分母为
已知干预 RMS；自然波形改变量使用固定逐通道训练 RMS。物理配对差另在被试
层计算 NRMSE 差，以 2000 次整被试重采样给探索性 95% 百分位区间，不宣称
多重校正显著性，不用重复窗口扩充独立样本量。完整分母与失败率先于成功
子集的误差呈现。

**交付与停止。** 固定面板、三个扰动和三个强度全部终结后停止，不根据验证
结果扩大搜索。报告同时呈现原始电压、物理 EEG 功率代理及 Hb 重建、旧与新
S/O 电压、配对标签、独立重估变化、无先验和错配先验结果。三个重点样例按
原始 EEG–EOG 相关性的 P10/P50/P90 选取不同被试，只用于说明。另保留全窗口
图集、数值数组、逐窗/被试表及失败记录。PDF 图为位图，正文可检索；先完成
软件/合成检查，再在持久监督下运行公开面板。可行性结论分别判断电压合同、
天然伪迹依据、干预保留、物理增量与未覆盖干扰；不推出 token 泛化或来源可识别。

<a id="ssm-so-normalization-redesign-20261106"></a>
## SSM 拟合与原始信号 S/O 监督的归一化重规划（2026-11-06）

用户随后明确要求先执行[三被试缓存控制实验](../experiments/configs/physiology_semantic_tokenizer/shared_driver_normalization_comparison_v1.yaml)。
该有限实验只复用已有缓存：PCA 因子采用已实现的个体内跨位置方向，尺度因子
只将原个体跨位置 SD 改为同被试 POOz SD，沿用原估计器和离线适配支持。
四组固定显式权重 1/1/1、原目标函数 SD、τ/β 拟合、先验和求解器，不提前
加入下文的稳健尺度、观测增益或误差模型。五名验证候选按输入幅度选三人，
各选三个不重叠窗口；原基线独立复现、失败完整保留。它检验两个具体因素，
不等同于实施下面整个重规划，也不生成新电压 S/O 标签。执行与结果由注册表
及运行输出持有。

本节按用户要求给出下一版本的设计，尚未实现或启动实测拟合。拟议诊断标识为
`SSM-NORM-SO-v1`；正式执行前在既有 configs/runner 内版本化，不覆盖已完成
S/O 标签、Hb 加权结果、缓存 v1 或已实现的个体 PCA 缓存 v2 配置。个体 PCA
仍是下一代要求，本节增加的校准支持、观测增益与噪声模型需要独立的后继合同。
当前实现与数据边界仍由 DATA_CONTRACT、源码和冻结运行配置持有。

### 设计依据与需要区分的四种量

问题证据引用[已有归一化审计](../experiments/RESULTS_INDEX.md)：冻结总体尺度
不能消除局部窗口的 Hb/EEG 相对幅度差异；原电压重建还使用训练人口的绝对
功率基线。审计未完成尺度、拟合权重、观测增益的因果分离。本节不把幅度差
直接解释为非神经信号，也不把标准化后方差相等当作生理耦合证据。

| 层次 | 新设计中的职责 | 估计支持与冻结范围 |
| --- | --- | --- |
| 测量单位与时间参考 | 保留 EEG 电压、已核验 Hb 单位及相对单位组；记录基线和滤波算子 | 中央 loader 和原数据合同；不加入通用逐通道 z-score |
| 数值预条件尺度 D | 改善求解数值条件；数据、预测、误差协方差同步变换 | 被试自己的校准段，按 ROI/Hb 通道对估计；应用窗口内冻结 |
| 观测增益与误差协方差 | 分别表达稳定的有效响应幅度与预测误差可靠性 | 校准段和训练被试的超参数选择；评价窗口不自由重标增益/噪声尺度 |
| 监督训练尺度 | 方便网络学习，保留原信号与 S/O 的线性闭合 | 仅训练被试的原测量拟合；X、S、O 共用同一尺度和固定单位 |

必须先验证坐标不变性。令原观测为 y、模型预测为 f、误差协方差为 R，D
为可逆正对角矩阵。使用 z=D^-1 y、f_z=D^-1 f、R_z=D^-1 R D^-T 时，

\[
(y-f)^T R^{-1}(y-f)=(z-f_z)^T R_z^{-1}(z-f_z).
\]

因此，仅更改数值标准化不能改变同一模型的物理拟合结果。更改状态坐标时还
必须变换动力学、初态、过程噪声和先验；本设计优先保持状态坐标不变。估计
R 时使用完整似然中的 log-determinant，不能靠无限增大误差尺度降低平方项。
已有代码中的输入增益同时定义了观测映射，不能当作纯数值预条件器。

### 个体校准、PCA 与幅度参考

按 dataset × subject 建立独立校准身份，并在个体内按 ROI 建立 PCA/尺度，
明确这比缓存 v2 的“个体内跨位置共用一个 PCA”更细。保持六个几何近邻 ×
五频带和原时间支持；不先逐 EEG 通道/频带强制单位方差。PCA 只读取该被试
校准段 EEG，符号采用最大绝对载荷为正；不用目标 Hb 选择 PCA 符号或方向。
共享电极身份在不同 ROI 中严格一致。个体主成分是统计坐标，不直接横向解释
PC1 绝对值为相同神经源强度。

校准数据与拟合评价窗口分开，禁止重叠窗口重复计数同一原始样本。初始诊断
可使用既有开发面板的两个不重叠 90 秒窗口，做首窗校准/末窗评价及反向交叉
拟合；它是离线个体适配，两个方向不是独立被试。预处理仍是已声明的整记录
离线滤波，不能称为严格未来预测。现有 A–G 仅用于回放展示，原首窗成为校准
数据时不能再计为独立验证结果。全量扩展先从 metadata 固定每个个体的校准
记录/连续块及样本支持；校准不足保留为不可用，不暗中借用其他被试的 PCA。
超参数和算法选择只使用训练/验证边界允许的支持，测试分区不用于本设计选择。

保留当前自然对数功率特征：

\[
b_{w,c,b}=\operatorname{mean}_{t\in B_w}\log P_{w,c,b}(t),\quad
\ell_{w,c,b}(t)=\log P_{w,c,b}(t)-b_{w,c,b},\quad
p_{s,j}(t)=w_{s,j}^{T}\ell_w(t).
\]

w 是窗口下标时写在 b/ell 中，w_{s,j} 是被试 s、ROI j 的单位 PCA 载荷；
c 为电极，b 为频带，B_w 为窗口前 5 秒，P 是 100 ms 均方电压（µV²）。
前 5 秒是计算参考，不自动称为生理静息。参考基线、功率下限和掩码必须保存；
基线受污染时标记不可用或按预定方案整体重定义参考，不只修改逆变换一端。
Hb 使用同一前 5 秒的线性均值参考，HbO/HbR 不分别做幅度单位化。

数值尺度的首选候选是个体校准块内中心化后的稳健幅度：对等长、非重叠的
校准块 k 计算 q_k=1.4826 median|u-median(u)|，再取各块 q_k 的中位数。
这只用于估计幅度，不把应用窗口重新中心化。记录退化、排除和全部块数；
有效块不足或尺度退化时判不可用，不用任意 1 伪装正常尺度。首轮与块内 SD
的中位数作敏感性比较，MAD 不是脑信号真实标准差的无条件估计。

设所得尺度为 a_E、a_O、a_R，采用

\[
a_H=\sqrt{(a_O^2+a_R^2)/2},\qquad D=\mathrm{diag}(a_E,a_H,a_H).
\]

共同 Hb 尺度保留色团幅度比；去除块间均值对尺度估计的贡献，不消除记录
本身的有意义偏移。MAD、分块和 ROI 粒度是待验证的本项目选择；文献依据
支持稳健个体参考与光学异质性，不规定此精确组合。[R1,R2,R6]

### SSM：固定 EEG 尺度锚点，单独校准 Hb 有效增益

模型在未标准化的代理/Hb 测量坐标中定义：

\[
p=c_E\,\mathcal R r+\epsilon_E,\qquad
\mathbf h=c_H\,\mathcal O_H H_\theta[r]+Z\eta+\boldsymbol\epsilon_H.
\]

r 为原 Tak 驱动，mathcal R 表示其前 5 秒参考算子；H_theta[r] 是 Tak
状态产生的二维 Hb 读出，mathcal O_H 包含匹配的采样、滤波及基线参考；
Z 是实测辅助回归量，eta 为系数。c_E 固定为 a_E/0.025，使驱动沿用已有
工程尺度，0.025 不解释为生理常数。c_H>0 为该被试/ROI 的共同有效 Hb
读出增益，初始化 a_H/0.025，在独立校准段估计后冻结。对数增益的收缩先验
由训练被试内部验证确定；它不是 DPF 的物理估计。HbO、HbR 共用 c_H。

第一版固定 neurovascular_gain=1 及现有血容量/氧提取读出常数，避免同时
放开驱动幅度、神经血管增益和光学增益。其他动力学先采用固定 Tak 参照，
通过之后才单独比较 τ 自由度。每个应用窗口只估计驱动和初态，不能用该窗
Hb 幅度重新拟合 c_H 以掩盖失配。增益与动力学仍需做灵敏度、剖面和跨块
复现检查；不可识别时保留失败，不作为可靠监督。此为本项目对已有生成模型
中“生理状态—光学读出—观测误差”分离的简化适配。[R2,R3]

对于已带通的 Hb，模型比较对象必须经过等价观测算子；不能只滤数据而把
未滤模型输出直接拟合。先合成核验完整处理支持、边界和前 5 秒参考的影响。
有限窗口若缺少复现滤波边界所需上下文，明确扩大已允许的上下文或限制评分
支持，不能在短窗口上再次滤缓存假装匹配。基线参考导致的相关性/秩约束进入
误差传播，求解使用独立有效支持。[R4]

### 噪声权重：从校准预测误差出发，替换固定 Hb 百倍偏好

不再把信号的总 SD 当作噪声 SD，也不把 0.1/10/10 当成新默认。先在校准
支持内做连续块遮挡的交叉预测，估计冻结的工作误差协方差；隐藏块必须考虑
滤波时间支持，不使用随机散点遮挡。误差包含未建模变化，命名为预测误差
模型，不宣称纯仪器噪声。校准不足时记录不可用。

误差模型第一版分为 EEG 标量与 HbO/HbR 二维块，不预先白化掉跨模态耦合。
校准误差的 AR 算子 F 处理时间相关，Hb 二维协方差 R_H 处理色团残差相关；
用收缩估计保持可逆。相同 F 必须作用于观测和预测残差两端。AR 阶数用校准
块内验证选择，并在未用于估计的块上检查残差相关。[R4,R7]

主候选使用白化创新的 Student-t 负对数似然：

\[
e_{m,t}=F_m[y_m-f_m]_t,\quad
L=\sum_{m,t}\left\{\tfrac12\log|R_m|
+\tfrac{\nu+d_m}{2}\log\left(1+
\tfrac{e_{m,t}^{T}R_m^{-1}e_{m,t}}{\nu}\right)\right\}+L_{prior}.
\]

m 为 EEG 或二维 Hb，d_m=1 或 2；nu 首轮固定为 5（工程候选），R_m 是
t 分布的尺度矩阵，协方差为 nu/(nu-2) R_m。另保留 Gaussian 创新对照。
序列起始项按相同约定处理；若拟合 F，其变换行列式与初始创新项也必须一致。
原驱动/初态先验保留物理坐标定义，不能在新损失下直接假定旧系数仍有相同
相对强度；用合成与训练内验证校准后冻结。评价窗口不重新估计 R_m；稳健
似然可以降低单点异常影响，但不能把持续失配解释为成功。

Barker 等的直接依据是 AR 预白化与 IRLS 在 fNIRS GLM 中的应用；这里的
Student-t 非线性 SSM 是本项目的待验证推广，不是该研究已验证的算法。[R4]

若数据确有短距离通道/外周生理辅助量，可在 Z 中加入并保持校准/评价边界。
没有这些测量时，首轮令 Z 为空；全头 Hb PC 或均值仅作独立敏感性对照，
不默认删除，因为它可能含共同神经活动。乘法尺度处理不了加性浅表血流。
这一步的依据是浅表生理干扰和短距离回归研究。[R3,R5,R8]

### 从 SSM 代理回到电压：同一坐标、匹配基线与最小修正

先把 SSM 输出逆数值变换为 p_hat；绝不能把单位 SD、0.025 模型坐标或
白化创新直接送入电压反演。A_s 将多个 ROI 的个体 PCA 载荷按真实电极顺序
嵌入全部通道 × 五频带，满足 p_obs=A_s ell_obs。若采用归一化代理，则 A_s
也必须带同一缩放；本设计主路径统一在未缩放代理上完成反演。

以“对观测 log-power 做最小修正”为主候选：

\[
K_s=\Sigma_s A_s^T(A_s\Sigma_s A_s^T)^{-1},\qquad
\widehat\ell=\ell_{obs}+K_s(\widehat p-A_s\ell_{obs}).
\]

Sigma_s 从独立个体校准段估计并收缩，定义修正的统计度量，不是已知干净
信号协方差。A_s 满行秩时 A_s K_s=I；必须检查奇异值和条件数，秩不足则
按预定规则只处理独立代理方向并报告丢失维数。此式保留 (I-K_s A_s)ell_obs，
即代理未约束的分量；p_hat=p_obs 时严格不改 log-power。它不同于旧的
mu+K(p_hat-A mu)，后者将未约束部分替换为人口模板。这是本项目提出的
最小变化估计器，是否保留了伪迹由已知干预单独验证。

绝对功率使用与特征定义完全匹配的当前参考：

\[
\widehat P_{c,b}(t)=\exp[\widehat\ell_{c,b}(t)+b_{w,c,b}].
\]

b_w 是上述已存的当前窗口 log-power 基线，不再无条件使用人口训练基线。
若仍使用旧的 b_train，功率会多乘 exp(b_train-b_w)，电压尺度对应平方根；
这与 SSM 是否拟合成功无关。参考污染必须显式检测/做基线污染干预，不能用
受污染基线宣称得到干净功率。[R1] 的基线研究支持明确区分参考规则，不直接
验证本项目电压反演。

现有电压最小变化投影/重叠合成可作为实现起点，但功率支持必须匹配：SSM
的 100 ms 功率先在线性功率域聚合到投影的 1 秒支持；不能把均值 log-power
指数化冒充 1 秒算术均方。窗口端点、反射和重叠规则与实际电压块一致。
校准相关矩阵与时变对角功率构成的正定矩阵只提供工程约束，不是假定的
生理真值协方差。若使用协方差上界，p_hat=p_obs 仅保证功率反演恒等，
空间相关矩阵仍可能改变电压；因此保留“无代理修正”的同参考投影对照，并
单列它的电压改变，不能把两者混称恒等映射。

不再用 Hb 代理去设定没有对应观测的第六频带/偏置补项的功率上限；该补项
在此分支原样保留，并明确它只是未修改成分。其他频段保留原电压的相位/
载波并作最小变化估计，定义 S 为估计电压、O=X-S。功率代理本身不能恢复
唯一相位。最终在 µV 中核验 S+O=X，重新提取 S 的代理与频带功率，并检验
合成后的约束；局部投影通过不能代替最终输出核验。ASR 提供了独立参考、
稳健协方差和回到通道空间的先例，不证明当前 S 是干净神经信号。[R6,R7]

Hb 分支同样先回到原测量单位：在去基线测量 h 上定义
S_H=c_H mathcal O_H H_theta[r]，O_H=h-S_H；辅助项 Z eta 归入 O_H，
完整预测另存为 S_H+Z eta。需要恢复带参考的缓存信号时，保存的基线只加回
S_H 一次，记录这是闭合约定；该基线不是可测绝对静息 Hb 浓度。原始信号
在此指已声明的原生 EEG/缓存 Hb 支持，不指被滤除频率或光学原始强度。

### 监督网络与评价：同一线性尺度作用于 X、S、O

标签以 µV 和合法 Hb 测量单位保存。网络尺度只从训练被试原 X 估计，按
被试等权汇总均方后开方；EEG 每个固定电极身份一个 q_c，Hb 每个合法单位组/
位置一个共同 q_H。同一路径的原信号与两分支统一除以同一正尺度：

\[
X'=X/q,\quad S'=S/q,\quad O'=O/q,\quad X'=S'+O'.
\]

不分别用 S、O 自身 SD 归一化，不按窗口重新定标，不分别减三个不同均值。
如需减去中心 m，必须明确 X'=(X-m)/q、S'=(S-m)/q、O'=O/q。默认仅缩放。
网络 decoder 使用已冻结训练尺度恢复单位；不新增被试/窗口增益连续 sidecar。
SSM 的个体适配尺度仅用于产生物理单位标签，不自动成为 hard-token 解码条件。
损失先在同一被试内汇总再跨被试平均，避免记录多的个体主导训练。

评价同时报告：物理 RMSE、共同训练尺度 NRMSE、局部 SD NRMSE、相关、
均值偏差与预测/观测 SD 比。局部 SD 只用于明确标注的诊断分母，不能把
尺度设成一样后只报告标准化误差。S/O 的闭合、神经样变化保留、伪迹变化
归属、无先验及错配 Hb 对照分别评分。天然 S/O 为模型条件估计；合成信号
及已知注入变化提供对应范围内的真值。

### 有限验证顺序与实施边界

1. 软件/合成：单位变换不变性；PCA/尺度来源独立；冻结变换的正逆一致；
   增益、先验与协方差同步；HbO/HbR 比例保留；相同代理的 log-power 恒等；
   匹配基线往返；100 ms→1 s 线性功率聚合；S+O 闭合及最终重分析。
2. 合成因子分离：分别改变单位、共同 Hb 测量增益、真实驱动幅度、加性
   Hb 生理波动、校准污染和参考污染。单位改变应完全不变；稳定测量增益
   由 c_H 吸收；真实驱动改变应保留；加性干扰不能仅靠缩放解释。包含神经
   与伪迹独立/相关、HbO/HbR 不等比例增益的失配情形，检查共同增益边界。
3. 后续实测诊断限定于已有开发被试面板，全部读权限/身份先按原合同核对。
   六名训练被试内部选有限超参数；五名验证被试独立个体校准后评价，两个
   交叉方向先在被试内汇总。现有 F/G 是解释性反例，不用它们挑最佳设置。
4. 核心对照分步加入：旧坐标/旧权重；只做严格坐标变换（应复现旧解）；
   个体 ROI 校准及 c_H；固定信号 SD 下取消 Hb 特殊权重；交叉预测误差
   协方差；时间白化与稳健创新。不能将 PCA、尺度、权重和反演一起改变后
   把全部收益称为归一化收益。可用辅助量的回归单列，不加入隐含自由度。
5. SSM 冻结后，电压反演分别比较旧人口基线、仅匹配基线、再加最小修正，
   并保留同参考无代理修正/错配代理。已知眼电、神经样变化及参考污染都要
   测量；抑制更多电压不是单独的成功指标。
6. 完整分母保留校准不足、不可识别、数值失败、最终功率越界和质量失败。
   数值收敛不自动生成可靠 S/O 标签。可靠性范围由训练内留出预测和已知
   干预固定，不能临时按验证结果挑阈值；同时给出未筛选结果和可用标签覆盖率。
   配对改善按被试汇总，说明 EEG 误差、Hb 误差与神经保留的取舍；完成后
   才决定是否扩展四数据集全量缓存。该规划本身不启动此类运行。

实现继续使用 `ssm_prepared.py` 的坐标 owner、既有 SSM 求解器与
`dual_component_tokenizer.py` 的电压参考/投影 owner；元数据保存原单位、
校准身份、原始支持、PCA、电压基线、D、c_E/c_H、误差模型、监督尺度和
逆变换。只在既有版本配置/缓存元数据内扩展，不建立平行 normalization manager。

### 原始研究依据与适用范围

- **[R1] Grandchamp & Delorme, 2011.** *Single-Trial Normalization for
  Event-Related Spectral Decomposition Reduces Sensitivity to Noisy Trials*.
  [Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2011.00236/full).
  比较功率基线与单试次归一化的统计影响，支持明确记录加性/乘性/对数参考。
  本项目保留原 log-power 规则并修正其逆变换，不把文中的 ERSP 结论直接
  当成 SSM 或电压 S/O 已验证。
- **[R2] Chiarelli et al., 2019.** *Differential pathlength factor in continuous
  wave functional near-infrared spectroscopy: reducing hemoglobin’s cross talk
  in high-density recordings*. [Neurophotonics](https://pmc.ncbi.nlm.nih.gov/articles/PMC6689143/).
  光程差异涉及个体/位置；平均 DPF 误差改变幅度，光谱依赖误差还导致色团
  串扰。支持检查个体观测增益，也说明共同标量不能修复所有测量差异。
- **[R3] Tak et al., 2015.** *Dynamic causal modelling for functional
  near-infrared spectroscopy*.
  [NeuroImage，作者全文](https://www.fil.ion.ucl.ac.uk/~wpenny/publications/tak-penny15.pdf).
  原式 y=g(theta)+X beta+e 分开生成模型、混杂项与误差协方差，并对生理/
  光学参数和误差精度作贝叶斯估计。支持本方案的分层职责；共同 Hb 标量、
  EEG 锚点和冻结校准是本项目简化设计，不是照搬原文参数化。
- **[R4] Barker, Aarabi & Huppert, 2013.** *Autoregressive model based algorithm
  for correcting motion and serially correlated errors in fNIRS*.
  [Biomedical Optics Express 4:1366–1379](https://doi.org/10.1364/BOE.4.001366).
  使用 AR 预白化及 IRLS 处理序列相关和运动异常。支持区分信号幅度与误差
  权重；非线性 Student-t 扩展和预测误差交叉校准需由本项目单独检验。
- **[R5] Kirilina et al., 2012.** *The physiological origin of task-evoked
  systemic artefacts in functional near infrared spectroscopy*.
  [NeuroImage 61:70–81](https://pubmed.ncbi.nlm.nih.gov/22426347/).
  联合光学、MRI 和外周生理测量识别浅表血流干扰，支持把加性生理成分与
  乘性尺度区分，不能把 Hb 幅度统一当作去除干扰。
- **[R6] Mullen et al., 2015.** *Real-Time Neuroimaging and Cognitive Monitoring
  Using Wearable Dry EEG*.
  [IEEE TBME，作者全文](https://sccn.ucsd.edu/~scott/pdf/Mullen_IEEE15.pdf).
  ASR 使用低伪迹校准参考、稳健协方差及通道回投影。支持独立参考的原则；
  不提供当前 Hb 条件电压上限的生理真实性或所选阈值依据。
- **[R7] Ledoit & Wolf, 2004.** *A well-conditioned estimator for
  large-dimensional covariance matrices*.
  [Journal of Multivariate Analysis，作者全文](https://www.ledoit.net/Well-conditioned2004.pdf).
  支持有限样本下的协方差收缩与稳定求逆。时间相关样本不直接视为独立样本，
  收缩强度在分块校准中验证，不从该论文推导固定 10% 的最优性。
- **[R8] Gagnon et al., 2014.** *Further improvement in reducing superficial
  contamination in NIRS using double short separation measurements*.
  [NeuroImage 85:127–135](https://pubmed.ncbi.nlm.nih.gov/23403181/).
  支持使用有空间对应关系的短距离观测解释浅表干扰；没有该类通道的数据
  不能凭算法补造同等证据。


## 四数据集优化权重与 S/O 已知干预对照（2026-11-07）

用户要求用多组数据比较 EEG/HbO/HbR 优化权重下的抗噪与形态保留，并据此
建议未来缓存配置。[执行合同](../experiments/configs/physiology_semantic_tokenizer/shared_driver_weight_so_v1.yaml)
固定四库各两名验证被试、各三名训练功率参考被试，选样按元数据完成，不按
重建效果替换窗口。原生电压保持 200/500/1000 Hz 及原单位，未知单位不改称
微伏；显示统一到 200 Hz，评分仍用原生数组。

七组比例为 EEG:HbO:HbR 的 1:1:1、4:1:1、1:2:2、1:4:4、1:4:1、1:1:4、
1:100:100，均将总和归一到 3，控制数据项相对先验的整体强度。PCA、个体
尺度、τ/β 模型、先验、起点、求解预算和数值验收保持原缓存设置；各扰动
独立重新优化驱动、初态和 τ/β。三个位置的输入由中央边界核对，等权基线
与原缓存配对复现后才执行完整矩阵。

原始、眼电、肌电、HbO/HbR 平滑扰动和 10/35 Hz 神经样变化形成固定矩阵。
已知变化的支持为 [30,50) 秒；电压强度取训练参考电极 RMS，另报告其相对
当前窗口 RMS 的倍数，不将这一工程强度称为自然神经事件幅度。神经样探针
保持 Hb 不变，不作为真实耦合神经源真值。抗噪指标要求 delta S 小，神经样
保留指标要求 delta S 接近已知添加量；因 O=X-S，两路变化不是独立成功证据。
原始窗口也不是干净真值，相关、导数、RMS 和频带功率共同描述形态。

电压 S/O 沿用既有局部功率上限与训练基线，不混入归一化重规划；检查最终
重分析与闭合。失败不替换为原始信号或零目标。先被试内汇总、再数据集内
等权、最后四库等权，完整分母与成功子集明确分开。预设相对筛查要求每库
噪声误差至少下降 20%，神经样误差增量不超过 0.05、基线 EEG 相关下降不
超过 0.02、冻结训练尺度 NRMSE 增量不超过 0.05，并保留完整配对支持。
该相对筛查没有保证绝对保留能力，也没有对 Hb 原始波形保真设硬门槛；因此
生产配置建议还需明确呈现 Hb 的相关、幅度及局部误差取舍，而不能仅据通过
相对筛查重生成全量缓存。结果与应用建议分别由运行 summary 和 review
持有，执行状态由研究注册表持有。本轮不启动全量缓存或 tokenizer 生产。

### τ/β 固定与可调的配对归因对照（2026-11-07）

[执行合同](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tau_beta_so_v1.yaml)
复用上组八名验证被试、1:1:1 和 4:1:1 的全部保留输入、训练参考与干预。
每个来源任务已经同时保存固定 τ=2、β=1 和可调 τ/β 的独立驱动/初态拟合。
本轮分别核对两分支自己的资格与预测，重建固定参数 S/O，并与已保留可调
分支配对；不读取新原始数据或重新运行非线性拟合。主终点为原生采样率下
眼电引起的 S 增量范数/已知扰动范数，比较八个被试配对差，再等权汇总数据集。
原零假设为可调参数不减少此泄漏。其他噪声、神经样探针及原波形为次要终点。

机制对照仅对原始/眼电条件冻结各自基线 SSM 预测，保留相同污染电压和功率
约束。完整重拟合与冻结参考的 S 增量差描述预测更新的贡献；其波形分解在
统一 200 Hz 显示坐标计算并单独标注，不替代原生采样率主终点。所有条件
保持 PCA、尺度、参考、权重总和、输入扰动与 S/O 算法不变；历史跨版本
结果只做方法差异审计。结果由新运行 summary 持有，失败保留，不扩样调参。

### 已完成缓存的眼电添加与重拟合确认（2026-11-07）

[执行合同](../experiments/configs/physiology_semantic_tokenizer/shared_driver_cache_ocular_confirmation_v1.yaml)
按用户要求，从完成的 v1 缓存选择此前权重实验之外的全部验证被试，每人取
首末两个可用记录的中间完整 90 秒窗口。选择只读元数据，不按拟合表现替换。
冻结原缓存等权、τ/β 先验、PCA、尺度及原训练功率参考。通过中央 reader 读取
这些精确窗口的原生电压，并复现缓存特征后，加入保留实测 VEOG 派生波形。
空间分布为预设额前较强模式，幅度为各原窗口选定通道整体 RMS 的 0.5/1/2 倍；
它是明确的工程注入，不当作被试真实眼电传播矩阵。三档支持均为 [30,50) 秒，
边缘一秒平滑到零，Hb 保持相同。重新计算所有频带及代理，逐条件实际重拟合
状态空间，再构建 S/O；未注入条件另与原缓存预测配对复现。

原生采样率计算 S 泄漏、O 增量相关/有符号增益、S 前后形态及相对变化、
Hb 泄漏与支持外变化。预先声明 S 泄漏≤0.30、O 增量相关≥0.90 为描述性
工程核对线，不作为经文献验证的生理阈值。逐强度保留全部窗口和失败分母，
先窗内、被试内、数据集内再四库等权；完整图谱显示所有前后波形和失败。
允许完成当前有限公开验证批次，不读取 test/protected、不重新生成全量缓存。

### 眼电跨模态传递下的拟合权重序列（2026-11-09）

本节设计经用户后续明确要求执行，固化为
[`SSM-OCULAR-MODALITY-WEIGHT-v1`](../experiments/configs/physiology_semantic_tokenizer/shared_driver_ocular_modality_weight_v1.yaml)。
问题是：τ/β 允许拟合时，相对提高 EEG 或
Hb 权重，怎样影响眼电对 SSM 代理、HbO/HbR-S 及最终 EEG-S 的传递。旧权重
实验的综合噪声分数混合 EEG 和 Hb 扰动，本序列把两个传播方向分别报告。

**有限面板与不变量。** 复用
`20261107_cache_ocular_confirmation_v1` 的已保留输入、三位置、原生采样率、
训练功率参考和三档眼电。取 Single-Trial subject_02/subject_14、Simultaneous
VP014/VP017、REFED 9、Visual S08，共六名被试。其两个保留窗口的三个基线
ROI 均数值合格；VP005 已知不收敛基线继续作为来源失败记录，不混入这组
完整配对的主面板。第一记录用于定向筛查，第二记录用于同被试跨记录复核。
这是已观察公开开发数据上的机制消融和配置比较，不是独立未见被试评价。
Visual 只有一名被试，按该实际支持报告。

固定数据集级 PCA、个体尺度、目标函数 SD、滤波、初态与驱动正则、训练统计
和功率上限 cap=1；本轮不混入个体 PCA、尺度校准或 S/O 算法修改。所有主
条件独立重拟合驱动、初态、τ、β，保留原 log 参数先验 SD=0.125、边界、
三个起点及固定参数热启动；τ/β 每窗每 ROI 为常数，不作为时变曲线估计。
实测眼电条件只改 EEG，Hb 输入逐点相同；相同窗口的所有权重共享同一注入。

**权重面板。** 顺序统一为 EEG/HbO/HbR；三个平方误差权重总和归一到 3，
保持数据项相对正则的总系数一致。HbO/HbR 之间保持等权，不增加单色团偏好。

| 比例 | 实际平方误差乘数 | 作用 |
| --- | --- | --- |
| 4:1:1 | 2, 0.5, 0.5 | EEG 偏重对照 |
| 1:1:1 | 1, 1, 1 | 当前缓存基线 |
| 1:2:2 | 0.6, 1.2, 1.2 | 温和 Hb 偏重 |
| 1:4:4 | 1/3, 4/3, 4/3 | 较强 Hb 偏重 |

不用不同权重各自的加权目标函数值给候选排名。所有评价均回到同一冻结 SD
及测量坐标，逐个模态报告；权重总和一致也不等同于实际残差贡献相同。

**序列一：直接检查 SSM。** 六个第一记录窗口，各比较未注入和 1 倍原窗口
RMS 眼电，三 ROI、四权重，形成 144 个 ROI 条件。等权的 36 个条件已有，
先完成软件/张量检查和合成 dry-run，再核对等权配置及基线复现；同源结果
可复用，其余新增 108 个。HbO/HbR 的预测增量 RMS/原冻结
分量 SD 为主终点，主支持取 [30,85) 秒，以覆盖注入之后的血流响应；另列
[30,50) 和 [50,85) 的有符号均值、正/负峰，定位 HbO 上抬及延迟效应。
EEG 端主比较预测代理增量 RMS/冻结 EEG 代理 SD，尚未经过电压功率约束。
预测代理增量/观测代理增量的范数比作为传递诊断；观测增量 RMS 不超过冻结
代理 SD 的 1e-6 时标为分母不足，保留主指标而不把该比值设为零或参与平均。
记录 τ、β 注入前后数值与模型实际血管输入 βr 的变化，检验是否由参数补偿
维持 Hb 拟合。基线各模态 NRMSE/相关、[5,25) 秒支持外变化及所有失败同时报告。

**序列二：分离电压构建的贡献。** 对序列一同一批预测构建 EEG/Hb S/O。
EEG 同时构建“污染条件重拟合代理提供参考”和“冻结该权重注入前代理参考”
两条路径。后一条件仍使用相同污染电压、训练统计及 cap，不增加 SSM 拟合，
也不把主实验的 τ/β 改为固定。比较 ΔS 全量、冻结参考下的 ΔS 及二者差，
分别展示 SSM 代理层与电压目标层的敏感性。Hb-O 的变化等于 −ΔHb-S，
作为解释图展示，不重复计为独立成功证据。

**序列三：有限复核。** 筛查后最多保留一个候选与等权基线。用六名被试的
第二记录重复未注入/1 倍条件；在六个第一记录补看 0.5/2 倍强度。等权条件
全部复用，新候选分别增加 36、36 个 ROI 条件。不在第二记录上继续调权重。
如没有候选满足下述约束，停止候选扩展并报告权衡，不自动加大到 Hb 百倍权重。

**序列四：方向性及保留核对。** 为判断建议能否用于一般缓存，而不只是保护
“Hb 干净、EEG 被污染”的情况，候选与等权在六个第一记录各增加 HbO-only、
HbR-only 平滑干扰：仅预先由眼电空间权重确定的最前部 ROI，幅度为对应冻结
被试 SD，EEG 与另一色团不变。每条件仅重拟合改变的 ROI，其余基线复用，
共新增 24 个 ROI 条件。分别评价 Hb 扰动保留与向 EEG 代理/电压的传播，不与
眼电分数平均为一个“综合抗噪”分数。

同时做最小的模型内阳性核对：使用两组合法已知 τ/β 和快/慢两种驱动包络，
从同一前向模型生成配套 EEG 代理与 Hb 响应；每对保持相同初态及同一噪声实现，
仅加入已知驱动变化。四个合成配对、两权重、前后条件共 16 个拟合单元。
比较恢复的驱动增量与生成器真值，检查权重变化是否仅靠削弱 EEG 响应换取
稳定。本项只检验 SSM 层，不能冒称恢复了真实 EEG 电压或生理真值；不再把
“只加 10/35 Hz 电压而 Hb 不变”的旧探针当作耦合神经事件保留证据。

**预设比较和停止规则。** 筛查的工程候选要求：相对等权，分层平均 HbO-S
扰动 RMS 至少下降 20%，HbR-S 扰动不增加，冻结尺度下 EEG 代理预测变化不增加；
未注入时 EEG/HbO/HbR 的固定尺度 NRMSE 增量各不超过 0.05，EEG-S 与原输入
相关下降不超过 0.02，无新增数值失败。HbO 改善方向需在每个有支持的数据集
一致。若多个候选满足，优先选择距离等权最近的比例。EEG 代理扰动若也下降
至少 20%，才额外标注 SSM 双模态抗扰动改善；否则只标注 Hb 保护改善。
这些是预先提出的工程容差，不是生理标准或显著性结论。

第二记录和强度复核保留相同护栏。反向 Hb 干扰与耦合合成的归一化恢复误差
若比等权增加超过 0.05，则该候选不能推荐为通用缓存配置，可报告为特定噪声
条件下的取舍。所有计划条件和失败保留，不以成功子集代替分母。先 ROI/窗口
在被试内汇总，再数据集内被试等权、最后四库等权；另列每 ROI 值和最大变化，
避免空间平均掩盖局部抬升。六名被试而非时间点/ROI 是独立统计单元；只给配对
差和分布，不据此作人群优越性判断。未注入观测用于保真护栏，不作为干净真值。

**预计工作量与交付。** 若进入全部复核，约 204 个新增实测 ROI 条件及 16 个
合成拟合单元；计数中的每个实测单元包含原求解器内部的固定参数热启动。
已保留的等权数据与数值结果先验算同源后复用，不新读原始数据，不启动全量
缓存。输出分层统计及全体面板：代理观测/预测前后、HbO/HbR 实测与 S/O 前后、
各分量增量、τ/β 参数变化、EEG 目标冻结参考对照。时间支持、纵轴与归一化
分母固定，所有候选与等权配对。执行时仍按既有入口、检查和持久监督器合同
实现。执行状态由研究注册表持有，筛查结果与是否触发后续序列由运行的
`summary.json` 和 `selection.json` 持有。每名被试至少导出四种配置的原始/
注入后 EEG、SSM 三路输出、EEG-S、EEG-O 对比，并在对话汇报数值与读图结论。

## S/O traditional physiology feature audit — 2026-11-09

The [versioned feature contract](../experiments/configs/physiology_semantic_tokenizer/shared_driver_so_physiology_features_v1.yaml)
reuses retained 1:4:4 free-τ/β predictions from the two weight experiments. Its
metadata panel comprises fourteen validation subjects (four/four/three/three
across Single/Sim/REFED/Visual), plus twelve retained training windows used only
for feature scales. The six recent subjects provide a separate ocular panel;
the older injection protocol is not pooled. No new fit or raw/protected read is
part of this audit. Source model settings, reference arrays and weights must
match before pooling. Native EEG S/O must reproduce the saved display target;
Hb S/O use the retained arrays exactly.

The EEG comparison uses the same 200 Hz, 1–45 Hz analysis view for X/S/O,
three nonoverlapping 20-second segments, and mean Welch PSD with four-second
Hann windows. Unfiltered native RMS, kurtosis and spectra remain separate
checks. Hb uses the original 10 Hz signal over 5–85 seconds without additional
filtering. Its 0.0125 Hz spectral resolution supports a primary 0.0375–0.2 Hz
comparison; lower frequencies have fewer than three cycles and are descriptive.
RMS, Fisher excess kurtosis, skewness, Hjorth statistics, spectral shape/band
powers and HbO–HbR correlation are paired to the original measured signal.
The spectral cross term is retained: additive S/O are not an orthogonal power
partition. Original measurements are the reference, not clean neural truth.

Within each subject, paired channel/segment statistics use the median; subjects
are then averaged within dataset, followed by equal dataset averaging. Full
point tables, subject values and invalid-feature counts accompany these means.
Separate EEG/HbO/HbR shape and shape-plus-amplitude embeddings use feature
scales fitted only on training originals, with equal dataset/subject weights.
PCA is also fitted on training originals. t-SNE jointly embeds the selected
analysis points; two perplexities and three seeds are retained without choosing
a favorable map. The six-subject intervention uses joint before/after coordinates,
deduplicating unchanged Hb inputs. Standardized original-space paired distances
and local-neighborhood trustworthiness accompany the plots; cluster separation
is not a physiological acceptance criterion. Complete the fixed feature grid
and report its results without expanding the panel or rebuilding the cache.

## Individual PCA and 1:4:4 cache warm-start comparison — 2026-11-09

The [bounded comparison contract](../experiments/configs/physiology_semantic_tokenizer/shared_driver_warmstart_pca_weight_v1.yaml)
compares standard optimization against a single cached-solution start under the
same new individual-PCA, positive-scale and normalized 1:4:4 objective. Select
one metadata-first training subject per dataset; fit the first full record/window
at first/middle/last available native pair, plus the same REFED subject's first
short record. The fixed panel contains fifteen window/pair units from four
subjects, without replacing failures. All admitted records/windows/sites of each
selected subject enter that individual's PCA and scales.

The cold input route re-extracts native EEG with the central reader and reads the
existing Hb measurement cache without another filter. The warm input route reuses
v1 prepared features. Compare all features/masks, independently estimated PCA
and scales, then the new three-component targets before fitting. This diagnostic
allows only these training subjects and excludes test/protected arrays before
reads. Raw processing and calibration time are reported separately from solver
time; common preparation is not credited as a numerical optimization speedup.

Cold optimization retains the fixed-parameter nuisance fit and three free τ/β
starts. Warm optimization takes the old free driver, physical initial state and
τ/β as one numerical guess and optimizes all variables under the new objective,
without a fixed prefit or automatic fallback. It does not assert that the old
and new PCA coordinates represent the same latent physiology. Both paths use
CPU float64 structured observation algebra, identical priors/bounds/stopping,
one numerical thread and the same final-bin exclusion. Each pair runs twice in
one worker with counterbalanced order after an untimed synthetic JIT warmup.

Report qualification, total and free-stage time, new objective, evaluations,
parameter changes and prediction RMS differences divided by the fixed new
subject SD, with every failed/unavailable unit retained. Objective relative
change uses (warm−cold)/abs(cold); engineering checks use 1e−6 for objective,
1e−3 subject SD for prediction RMS and 1e−3 relative parameter change. These are
comparison diagnostics, not physiological tolerances. The fixed panel ends with
paired tables and three-modality waveforms. It does not launch the full new cache
or switch any downstream experiment source. Registered lifecycle remains owned
by the research-state registry; numerical outcomes remain in the run artifacts.

## Full individual-PCA 1:4:4 cache generation — 2026-11-09

The [v3 cache contract](../experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_subject_cache_v3.yaml)
implements the user's full-cache launch after the bounded warm-start comparison.
Reuse the verified v1 prepared inputs and immutable time/identity plan; recompute
all individual coordinates. Produce free τ/β reconstructions under sum-three
1:4:4 weights, with cached-solution initialization and explicit standard fallback.
No old fit becomes a new terminal outcome. Public unlabelled teacher adaptation
uses the existing data contract and excludes protected material.

Software/synthetic checks precede source-array access. The training-only pilot
uses first full records, short REFED and each dataset's training subject with the
largest signed-PC direction change. Compare standard CPU, cached-start CPU and
cached-start CUDA at production coordinates. Preserve all statuses; subjects
whose warm result is worse than the standard objective by more than 1e−6 use the
standard production route. CPU/CUDA checks require matching qualification and
1e−4 prediction/parameter tolerances on jointly qualified cases. Pilot results
remain diagnostic rather than silently reused production outcomes.

Full execution uses a frozen source snapshot, bounded user-systemd supervision,
CPU/CUDA allocation grounded in the retained resource measurements, resumable
atomic tasks and final population verification. Warm failures and fallback
outcomes remain auditable. Completion requires all planned identities terminal
and independent arrays/objective checks. A pending or numerically failed task is
not a usable target. The registry owns lifecycle; the run's plan, coordinates,
execution/progress, task records and final summary own generated evidence.
