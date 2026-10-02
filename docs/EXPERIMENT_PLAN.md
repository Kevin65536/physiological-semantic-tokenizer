# Experiment sequencing

_Owning protocol: `PST-DISCOVERY-v1` · state: planned · 2026-08-26_

This is the single current experiment-design owner for the next
physiology-semantic tokenizer generation. The plan is limited to physical-teacher
qualification, source/observation tokenization, and coupling-prior retention. It
does not use downstream task performance as a training or selection endpoint.

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

## Reading guide

This file owns experiment design. It contains bounded diagnostic protocols and
the broader tokenizer design; section order is not an execution queue. Use the
[registry view](PROJECT_STATUS.md) for state and next actions, and the
[entrypoint/config/test map](../experiments/README.md#entrypoint-config-and-test-map)
to navigate implemented commands. Short-term scratch plans are not dependencies.

| Reading task | Sections |
| --- | --- |
| Fixed-structure physiology semantics and parameter sharing | [2026-09-28 design](#固定结构的跨数据集生理语义检验2026-09-28-设计) |
| Dataset processing and SSM fit revision | [Joint revision](#数据统一化与-ssm-拟合的联合修订2026-09-15) |
| Observation-contract design | [Observation repair follow-up](#观测合同修复后的下一轮实验设计2026-09-10) |
| Retained overnight and observation protocols | [Overnight v2](#bounded-overnight-ssm-diagnostics-retained-v2-contract), [flow/mask bridge](#step5-flow-domain-and-mask-specific-observation-experiment), [repair v1](#step5-observation-contract-repair-and-regression-retained-v1), [observation diagnostic](#step5-observation-adaptation-diagnostic-retained-v1-contract) |
| Synthetic calibration and staged qualification | [Step5A0 localization](#step5a0-inference-consistency-diagnostic), [Step5 stages](#full-step5-staged-continuation) |
| Overall research design | [Question](#fixed-question-and-decision-target), [flow](#experiment-flow), [estimand and statistics](#common-estimand-and-statistical-contract) |
| Qualification and tokenizer design | [P0](#p0-software-and-synthetic-qualification), [teacher selection](#t-physical-teacher-selection), [tokenizer](#bq-source-and-observation-tokenizer), [coupling prior](#c-coupling-prior-return) |
| Implementation and remaining design decisions | [Code ownership](#code-ownership-for-later-implementation), [unresolved qualification decisions](#unresolved-before-measured-qualification-or-confirmation) |
| Historical boundaries | [Side paths](#side-path-experiments-without-workflow-sprawl), [lifecycle boundary](#historical-lifecycle-boundary) |

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
