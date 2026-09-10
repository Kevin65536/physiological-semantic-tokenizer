# PST 状态空间 teacher：无人值守并行诊断与改进实验方案

版本：2026-09-10 / overnight_v2（新增 N7，保留 v1 准备记录与配置）

代码基线：Kevin65536/physiological-semantic-tokenizer @ 6174f3092d7f52a8b89da35607b7bc0c03469a9d。

文档性质：当前夜间诊断的实验合同；运行状态与结果由 run manifest 和固定 task_table 持有。执行入口为 experiments/evaluate_ssm_overnight_diagnostics.py，当前配置为 experiments/configs/physiology_semantic_tokenizer/ssm_overnight_v2.yaml；v1 配置与旧准备记录保留。

## 0. 本轮目标与取舍

本轮不是一次新的完整 physical-teacher 资格评审，而是一个七路并行的辨因与候选筛选实验。期望输出两类结果：第一，判断当前失败主要来自观测处理、模态权重、空间特征、参数补偿、过程模型还是近似推断；第二，找出值得继续验证的低复杂度 teacher 候选。

必须保留单一六状态共享 Balloon 结构、GWZ 参数支持域、固定的 tau/alpha/E0/P0/Q0 与原有 teacher 导出语义。不增加 private latent、额外 systemic 状态、switching、多脑区或新的全脑前向模型；不训练 tokenizer。下面的观测增益、噪声敏感性和 EEG 特征替换是明确的新探索分支，不回写旧运行。

允许接受原始生理参数不能被精确恢复；不允许靠重新归一化、任意扩大区间、静默裁剪状态或删除失败案例制造成功。当前代码的时序参考仍是静息线性化 Gaussian 参考，不是已验证的非线性 Student-t 时序推断器。[R1–R4]

## 1. 总体并行图

公共准备 P0 → N1 / N2 / N3 / N4 / N5 / N6 / N7 同时进入队列。

| 编号 | 核心问题 | 操作 | 执行性质 |
|---|---|---|---|
| N1 | 共享状态究竟向谁妥协、代价多大？ | 完整输入、单模态、中心遮挡与 null 的统一观测残差及状态影响审计 | 主要复用 |
| N2 | 时间处理失配，还是近似推断本身有问题？ | 相同输入下，逐点、线性时序参考、非线性相关 Gaussian 批量 MAP 对比 | 前两项复用，第三项新实现；不能阻塞其他实验 |
| N3 | 模态观测噪声权重是否失衡？ | 只改变 fNIRS 的共同观测噪声尺度 | 主要复用，直接改进候选 |
| N4 | 全局 EEG 坐标、眼动或频带/符号不匹配？ | 固定的一小组 EEG 清理与空间/频带投影 | 小改动，直接改进候选 |
| N5 | W 边界是否在代偿观测幅度？跨 session 是否可重复？ | W–观测增益二维诊断、一维增益候选、共同坐标下 session 重采样 | 小改动，诊断加直接改进候选 |
| N6 | Hb 拟合是否主要绕过 r 依赖过程创新？流量越界从哪里来？ | r 驱动前向回放、分组过程噪声敏感性、同输入越界追踪 | 主要复用，诊断加直接改进候选 |
| N7 | 增益与时间尺度锁定是否限制预测？ | G–W 独立拟合，W-only/G-only 消融及观测增益对照 | 新增生理自由度探索，不授予参数或 teacher 资格 |

同族实验应共享同一个已冻结基线。新实现未完成、局部合成检查不通过或个别数据缺失，只影响对应实验分支，不中止整晚任务。

## 2. 公共数据、评价与防泄漏合同

### 2.1 数据边界

实测信号仍只用 subject_01、subject_09、subject_18，session_01/03/05，每 session 原有 8 个训练 MA trial，共 72 个唯一 trial。原始 MA trial_position 为 4、9 的 trial 在切片及预处理前排除；不读取 subjects 19–29 的信号，不重新使用旧 Step5B 留出 trial。[R1–R2]

特别注意：旧 replay 的 prepared training index=4 与原始 MA trial_position=4 不是一回事。必须以 sample_id / event_index / 原始 trial_position 核对，不能误删合法失败案例或误开旧留出数据。[R4]

已有 18 被试 summary/JSON 可做不读取信号的描述性汇总，但不能冒充新 18 被试实验。今晚 3 被试的 session 结果只能支持方法定位，不能支持人群 ICC、临床 trait 或稳定亚型结论。

### 2.2 两种数据用途严格分开

**候选评分通道：** 保留现有 4 外折 × 3 内折。每 session 的 8 个原训练 trial 排序后以存活序号 modulo 4 指派外折；每外折每 session 6 训练、2 评价。内折仅在外训练 trial 内重新划分。所有 PCA、EOG 回归、fNIRS pair、尺度、噪声与参数选择都在对应训练部分重新拟合。[R2]

**共同坐标描述通道：** 仅 N5 的 session 可重复性和 N7 的 G–W 曲面诊断允许用全部 24 个原训练 trial 建立同一被试的共同投影和尺度，再分别拟合三个 session。因为目的是比较同一个操作性坐标，不是声称新 session 泛化。该结果不得进入候选外折评分。

缓存可以共享原生允许窗口和分折清单，不能共享基于全部 24 个 trial 拟合的 PCA 给外折评价使用。

### 2.3 固定比较对象

初轮多数单因素候选固定 W=0，G=0，Z=参考值；旧 W=-0.5 单列为第二基线。N3–N6 非基线候选先不与 W 搜索笛卡尔相乘；N7 是明确例外，仅独立放开 G 与 W，不与观测增益、噪声或 EEG 候选相乘。候选经内折选择后，外折只评价该选择规则的结果，并保留所有失败。

各候选默认使用相同的 fNIRS 目标通道、目标处理、训练标准差与 trial 评价位置。改变 EEG 特征的 N4，以不变 fNIRS 目标的恢复和配对增量为跨坐标主比较；不同频带 EEG 自身 NRMSE 不能直接作为整体 teacher 优劣排名。

输入在滤波、功率、重采样等传播信息的步骤之前遮挡；被插值的隐藏点不是新观测。新算子必须通过“任意替换隐藏原值，最终可见输入与预测不变”的干预测试。[R3]

### 2.4 公共合成数据库

在候选拟合前固定 discovery 与 assessment 两套不同种子，每种条件各 16 个独立 trial，至少包括：

1. 现有匹配非线性六状态离散过程 + Student-t 观测噪声。
2. 同一 clean truth 经实际受控时间算子与已知尺度处理。
3. 固定 clean truth 叠加低频漂移、跨 HbO/HbR 相关误差或稀疏异常；强度只用预设值/训练数据统计量确定。
4. trial 特异跨模态关系被破坏的 negative control：共享任务模板可以保留，随机 trial 成分独立配对。

N2 另有线性 Gaussian 与非线性 Gaussian 匹配生成条件，详见该节。N4 的眼动实验有独立原生 EEG 人工污染检查；不能把缺少 EEG 空间生成机制的三坐标合成数据用于证明 PCA 清理有效。

生成器拿到 truth 不代表拟合器可以拿到 truth。已知参数、clean r、clean 观测和实现噪声只传给评分器。无效生成轨迹保留，不能补抽到全部成功。匹配生成结果与失配 stress 分开；固定参数小面板不是完整 SBC。[W1]

### 2.5 必报指标

**完整输入拟合差：** 每个模态的 bias、RMSE/训练 SD、MAE/训练 SD、|残差|/训练 SD 的 p50/p90/p95。原生单位、当前观测坐标单位和标准化单位三者命名分开。不能从 log-power PCA 反推出原生 200 Hz EEG 电位差；不可逆部分明确记为不支持。

**遮挡恢复：** 中心 4 秒为主，整模态缺失为次；按 trial 计算，再 session、subject 等权汇总。不能用更大的训练标准差人为降低 NRMSE。

**跨模态增量：** joint 对 own-context、own-context+task-template、independent pairing、circular shift。另有同折线性 basic / joint 基线。跨不同目标、不同保留子空间或不同处理坐标，不比较绝对 log likelihood。

**不确定性：** clean 区间、带噪观测预测区间、参数区间分列；同时报告覆盖率和平均区间宽度。没有可靠区间的 MAP 候选填 NOT_ESTIMATED，不能借用旧 solver 方差。

**神经共享性：** joint 与 EEG-only/fNIRS-only 的 r 差异；打乱另一模态后的 r 改变量必须与预测质量一起看。状态“变化大”本身不是发现共享信息。

**合成去噪：** r 与 clean EEG/HbO/HbR 的误差，且对同一目标与原带噪坐标、简单同模态平滑、同模态推断比较。不能只报告输出更平滑或高频功率减少。

**残差结构：** 一步预测标准化创新的 bias/RMS、trial 内 ACF、低频功率比例，HbO/HbR 残差互相关。平滑残差与一步预测创新分表。

**物理及数值：** 有限性、f>0、氧提取域、绝对 Hb 约束、首次失败位置、求解器和优化收敛状态、运行时间、内存峰值与完整样本分母。

三个被试的 bootstrap 区间只作描述。时间点、多个遮挡、重复拟合与 bootstrap 重抽都不是新的独立被试。

## 3. N1：共享状态的折中代价与模态影响

### 3.1 假设

当前“共享”可能主要由 EEG 驱动，fNIRS 只是通过后续状态勉强配合；也可能 HbR 的持续偏差主要属于观测失配。这一实验先将现象量化，不调整模型。

### 3.2 固定矩阵

对 72 个原训练 trial 按外折训练标定后评价 W=0 与 -0.5。至少包括 full-joint、EEG-only、fNIRS-only、all-missing、center-EEG、center-fNIRS；两种中心遮挡补齐四种旧对照。使用明确相同的目标坐标与评分支持。

定义单模态代价：

C_m = D_m(y_m, yhat_m_joint) - D_m(y_m, yhat_m_own)。

这里 D_m 是同一观测坐标下的标准化均方误差。C_m 可以为正也可以为负，不能用绝对值抹掉谁获益、谁付出代价。full-fit C_m 与 masked C_m 分开。

另报告 joint r 对两种单模态 r 的标准化差异、相关、幅度比；若单模态推断越界，保留失败，不填零值。

### 3.3 额外关键检查：HbO/HbR 双坐标自洽性

报告处理后 HbT=HbO+HbR 的误差与残差，核对原生波长配对、浓度顺序、符号、基线和共同缩放。只作单位/代数合同检查，不因某个符号让结果更好就自动翻转 fNIRS。若找到实际单位或顺序 bug，另建修复版本并保留原失败。

### 3.4 输出与判读

输出 compromise_by_subject_session.csv、full_fit_residuals.csv、masked_control_scores.csv、innovation_structure.csv 和同一时间轴的观测/预测/区间图。

若 EEG full-fit 好而 HbR 始终有定向残差，应优先解释 HbR 观测映射和噪声，而不是把妥协命名为去噪。若另一模态改变 r 却使预测变差，属于有害影响而不是有效共享。若 full-fit 好、遮挡很差，优先怀疑追噪声或模型缺乏外推能力。

本实验不依赖 N2 完成。

## 4. N2：观测时间算子与非线性推断的分离实验

### 4.1 假设与候选

当前修复已能显式编译遮挡感知的时间算子，但 Gaussian 参考同时线性化了动力学；不能把它的改善全归因于一个因素。[R3]

| 候选 | 动力学 | 观测误差与推断 |
|---|---|---|
| O0 | 现有非线性 | 现有逐点 Student-t 联合 filter/RTS，旧基线 |
| O1 | 静息线性化 | 已有完整时间算子 + Gaussian 精确短窗参考 |
| O2 | 原非线性六状态 | 新增完整时间算子 + 相关 Gaussian 观测误差，批量 MAP；可选局部 Laplace 诊断 |

O2 是本轮有意探索的观测噪声简化，不冒充非线性 Student-t 时序推断。O1 不是 O2。旧 measured 入口的 Student-t 前提与负结论保持不变；新 Gaussian 分支用独立入口、独立实验名、自己的合成检查，只作探索。

### 4.2 模型合同

可控线性特征桥接下：

y_in = A_M h(z) + A_M epsilon,
R_in = A_M R_pre A_M^T。

A_M 是真实的 visible-selection → interpolation → processing → output-selection；两个矩阵的 mask、处理顺序和边界必须相同。R_pre 是处理之前的噪声，不能把处理后差分估出来的噪声再当作 R_pre 重复过滤。

在原生实测上，应明确把推断坐标放在已声明的 EEG feature / Hb concentration 层，并仅对后续确实线性的算子建模。raw EEG 平方取 log、OD、运动抑制并不会因此自动获得生成模型验证。若现有 native helper 不能暴露这一分界，O2 的实测分支记为 NOT_IMPLEMENTED；完成合成和已有参考，不绕过分界。

使用完整非线性转移 F_theta，与原固定 Q 和初始状态先验，优化：

J(z)=0.5||z0-mu0||²_(Sigma0^-1)
 +0.5 sum_t ||z[t+1]-F_theta(z[t])||²_(Q^-1)
 +0.5||y_in-A_M h(z)||²_(R_in^+)。

同时记录定义密度所需的归一化项和可观测子空间；MAP 的最小 J 不等于边际似然，不用于 W 后验或宣称“W 似然已恢复”。有可信的 Laplace evidence 时也只能标记 approximation。

在可观测 SVD 子空间求解，保留丢弃秩和容差敏感性。不能把稠密时间算子的 Jacobian 错写成稀疏；状态转移部分可利用稀疏结构。两固定初值为静息解和现有线性参考的可行解；不可行初值须报告，不裁剪成可行。限制 max_nfev=200，使用已存在解析导数或经过检查的 Jacobian。优化器的不可行试探步可以拒绝并记录，但不能裁剪状态；须区分一次非法试探与最终无可行收敛解，后者才是案例失败。[W3]

### 4.3 合成核心矩阵

线性 Gaussian、非线性 Gaussian、非线性 Student-t stress 三种生成规律，各 24 个独立 trial；64 点、4 Hz；model / combined 两种处理；full、center-EEG、center-fNIRS 三种主 mask；W 固定 0 与 -0.5。每种规律前 8 个固定身份额外检查 whole-EEG、whole-fNIRS。新种子与既有 24-trial 桥接分开。

O0/O1 可立即执行；O2 单独实现。O2 的线性特例 MAP 应和 O1 均值相合，相对误差阈值 1e-6；导数相对误差 1e-5；隐藏值干预测试 1e-10；密度坐标变换误差 1e-8。以上是本轮工程阈值，不是生理标准。

O2 的非线性 Gaussian 匹配规律检查均值误差、求解一致性、失败率。为使无人值守分支判断明确，本轮仅为进入受限实测探索设置均值预检：W=0、full 输入的 24 个固定案例全部完成；r/clean EEG/clean HbO/clean HbR 的平均 NRMSE 各不超过 0.65，r 平均相关不低于 0.80，所有前述工程检查通过。该阈值是探索分支的保守预检，不是 teacher 资格或统计校准保证；失败则只停止 O2 实测分支。Student-t 规律只作失配压力测试。24 次固定 W 不是参数 SBC，也不足以授予正式后验校准资格。

### 4.4 带噪预测的重要区别

输入 y_in 与评分目标 y_target 经过不同算子，但可能包含相同原生噪声。预测带噪目标时要保留 R_target,input=A_target R_pre A_in^T；不能简单使用 clean posterior variance+独立噪声。clean teacher 预测不加入观测噪声。

若 O2 尚未实现正确预测密度，只比较同一目标上的均值误差；区间与 log-score 标为未估计，不复用旧 pointwise scorer。

### 4.5 判读

O1/O2 对同步算子均改善，而 O0 失败：观测合同问题有支持。O2 比 O1 明显改善：非线性需要保留。O2 比 O0 少越界且同输入误差更好：局部 filter 更新/闭合误差值得优先修改，但不能据此证明整个生成过程正不变。仅 Gaussian 匹配规律成功、Student-t stress 失败：噪声简化仅适合有限情景，不能广泛替代原 teacher。

O2 的实测探索只在自己的数据层分界、软件检查和合成均值验证完成后入队；其他六族不等待。

## 5. N3：观测噪声与模态折中权重

### 5.1 单因素候选

先固定 W=0、EEG 特征与全部生理参数，只将 HbO/HbR 两个观测噪声标准差共同乘 c_N：

c_N ∈ {0.5, 1, 2, 4}。

不要分别调 HbO 和 HbR，不改变均值映射、不改变目标尺度。标准差乘 c_N，方差乘 c_N²。现有训练噪声估计规则给出基值；倍率是新实验显式敏感性，不重新标记成已知真实噪声。

每个外折在内折选择一个倍率；外折评价该选择规则。保留所有倍率的内折表现、参数选择频率、三模态风险和失败率。固定 W=-0.5 的重复只进入扩展队列，不与所有参数组合展开。

### 5.2 选择指标

在相同目标坐标使用模态平衡的中心遮挡风险：

B=0.5 NMSE_EEG+0.25 NMSE_HbO+0.25 NMSE_HbR。

这是候选评价指标，不是新增训练 loss。先检查完整性与各模态退化，再比较 B。必须看正确配对相对 null 的增量，不能选择只会忽略 fNIRS 或使区间更宽的候选。

### 5.3 诊断补充

在外训练 trial 中报告一阶差分尺度、创新尺度和时间相关性；只用训练 trial 的分块重采样描述噪声尺度不确定性，不将 ACF 0.99 的残差解释为许多独立观测。

### 5.4 判读

放松 fNIRS 权重后 r 更稳、两方向外折风险改善，且不损失配对增量：相对置信度确有问题。区间覆盖改善但点预测/真值恢复不改善：只有不确定性膨胀。最优 c_N=4 仍强烈 HbR 偏差与高自相关：不要继续扩大到 8、16，转向 N2/N5。

## 6. N4：空间、伪迹与 EEG 坐标

### 6.1 预先固定的六个分支

| 编号 | EEG 输入/投影 | 目的 |
|---|---|---|
| E0 | 现有全通道 1–45 Hz log-power PCA1 | 对照 |
| E1 | 训练拟合的 EOG 回归清理后，同样的全通道特征 | 隔离眼动影响 |
| E2 | 排除额极/额部的固定通道子集，再做同样 PCA | 伪迹敏感性负担检查，不称正确脑区 |
| E3 | 与固定 fNIRS pair 对应的局部 EEG，1–45 Hz | 隔离全局/局部空间混合 |
| E4 | 相同局部 EEG，8–13 Hz，正 log-power | 频带检查 |
| E5 | 与 E4 完全相同，只反转该特征相对基线的符号 | 检验功率特征与固定正耦合约定不匹配 |

不更改正值 beta、EEG loading 或固定状态符号约定；E5 是观测特征定义对照，不能解释为改变神经效能。beta 频带可进入预定义扩展队列，但不在首轮扩成通道×频带×符号全面搜索。

E1 仅当当前允许原生源确有可用 EOG 且能在训练/遮挡合同中安全读取时运行；否则记 NOT_AVAILABLE。EOG 回归是已有常规方法，系数可以在训练数据拟合后应用，但不能用外折目标估计系数。[W2]

局部对应先依据实际 montage/optode 元数据固定最近的 1–3 个 EEG 位置或已有明确的 pair 映射，不能用跨模态相关最大化选位置。元数据不支持时，只保留已声明 F3 对照，并标记 anatomical_mapping_unverified；不能把 F3 自动说成所有被试的正确局部对应。E2 的通道名单在读取响应评分前冻结。

### 6.2 同一 fNIRS 目标原则

外折训练先用基线规则固定一个 fNIRS pair，各 EEG 分支共用这个 pair。fNIRS 的预测目标、评分标准差和同模态上下文都不变。

跨 EEG 定义的主终点是 EEG→固定 fNIRS 的中心遮挡误差、相对自身/任务模板的增量、正确配对对独立配对/移位的增量。不同 EEG 特征上的 EEG NRMSE、r 幅度和相关性不能无条件横向比较。

### 6.3 人工伪迹检查

选择已允许原训练 EEG 窗口，先冻结原窗口作为未注入参考，再在输入副本叠加训练来源的 EOG 波形，或固定形状的 blink-like 脉冲。对额部与后部两种冻结空间权重注入，强度为原训练 EEG SD 的 0.5 和 1.0 倍；fNIRS 不改。

该实验检验估计 r/teacher 对新增伪迹的敏感性，不把未注入实测窗口称为干净神经真值。只有有真实 EOG 参考的分支能评价 EOG 回归；没有 EOG 时，只能评价对人工污染的特征鲁棒性。

### 6.4 判读

E1 优于 E0、局部配对增量也改善：眼动清理值得保留。E3/E4 优于 E0 且超过两种 null：空间/频带对应更有希望。只有 E2 的 full-fit 更漂亮：不足以证明后部更适合前部 fNIRS。E5 优于 E4：优先重审特征符号和观测映射，而不是增加血流参数。

## 7. N5：W–观测增益补偿与 session 重复性

### 7.1 增益和单位变换必须分开

新增一个无生理解释的、正的 fNIRS 共同观测增益 a_N：

h_N_new(z)=a_N h_N_old(z)，a_N ∈ {0.5, 0.75, 1, 1.5, 2}。

观测数据、目标标准差、P0/Q0 与观测噪声 s_N 不随 a_N 重新缩放；均值和状态 Jacobian 的 fNIRS 行必须一致乘 a_N，clean 轨迹方差按映射变化。

这不是已知单位变换。禁止使用 coordinate_scale 来伪装未知增益，因为已知单位变换会同时改变数据和噪声，从而检验的是不变性而不是测量失配。[R3]

### 7.2 两类任务

**诊断任务：** 每被试三个 session 共用一套描述性投影和噪声坐标。计算 W 的 17 点一维曲线；另计算 9 点 W × 5 点 a_N 的二维曲面。W 范围仍是 [-0.5,0.5]，其他生理参数全固定。按 trial 缓存 log likelihood，再组合成 session 曲线，无需为 bootstrap 反复求解。

**改进候选：** 固定 W=0，只在内折选择 a_N，外折进行同目标评估。W+a_N 同时自由拟合仅作补偿诊断，不进入首轮 teacher 排名。

### 7.3 session 与技术重复

每个 session 有 8 个 trial。固定相同坐标，分别估计 W 曲线；对 trial 进行 200 次 bootstrap（使用缓存逐 trial 曲线）；再按预定义奇偶存活序号分成两个四-trial 半样本。输出估计差异、曲线平坦度、边界状态与近优参数对应的 teacher 差异。

该 bootstrap 条件于冻结的 gauge；不包含 gauge 估计误差。其区别必须与旧不同折 gauge 下的变动分开。部分网格失败时，保留缺失形状，不能删点后宣布完整 posterior。近优支持只作诊断，不把粗网格 Delta log L 自动转为精确 95% 区间。

如资源充足，追加真整-session 留出：只在两个源 session 拟合全部对象，应用到第三 session。预测可比较，跨折 W 数字仍不能视作同一 gauge 的直接生理重复。

### 7.4 判读

允许 a_N 后 W 的边界压力明显减弱，且一维 a_N 候选改善外折风险：先改观测幅度标定。a_N 与 W 存在长斜谷而 teacher 不敏感：接受参数不可辨识，固定一个方向。不同参数的近优 teacher 也大幅变化：状态本身仍不确定。session 差异不超过同 session 半样本/重采样变化：没有足够证据讨论稳定个体差异。

## 8. N6：过程创新、r 驱动闭合与流量域

### 8.1 为什么是本质问题

当前六个变换状态都有过程扩散；因此 HbO/HbR 拟合可以同时使用共享 r 和下游状态创新。仅看 joint 观测重建并不能证明全部重建来自 r。[R5]

### 8.2 r 驱动前向回放

对 N1 的 full-joint 结果，冻结 r(t)、该 trial 起始血流状态与所有参数；以 r(t) 的固定分段线性插值作为驱动，独立积分五个血流状态，关闭下游新增过程创新，不逐时刻重置到 smoother 状态。经过同一观测算子输出 HbO/HbR。

比较 joint clean prediction 与 r-only forward replay prediction；同时比较它们与观测/合成真值的误差。输出 replay_gap / 训练 SD，以及变换状态一步偏离 F(z_t) 的分组标准化幅度。

非线性下 F(E[z]) 不等于 E[F(z])，且过程噪声本来就允许非零创新。因此不能把均值回放差直接解释成“私有信息比例”或物理违规。要与匹配合成数据的相同统计量对照；有可靠联合路径样本时才追加路径级检查，不能用独立边际采样冒充完整后验路径。

### 8.3 两条独立的过程噪声敏感性

固定 W=0、观测尺度与全部生理参数：

- 血流相关五状态过程标准差 sigma_H 共同乘 {0.25,1,2}，sigma_r 不变。
- r 的过程标准差 sigma_r 乘 {0.5,1,2}，sigma_H 不变。

合计五个唯一配置，不展开两个倍率的 3×3 网格；方差按倍率平方变化。零过程噪声只用于确定性回放，不直接塞进依赖可逆 Q 的原 filter。

候选必须同样进行中心遮挡、配对 null、已知 clean truth 与物理失败评估；不能以 full-fit residual 下降为唯一选择条件。

### 8.4 同输入失败追踪

首先复用三个已记录 training identities，在 W=0/-0.5 上回放；该清单可先读旧 prepared 文件，不读取新数据。分离“首次有风险的观测更新”和“首次 f=0 过零时刻”。

在完全相同状态与输入前缀上，做跳过最后一次观测更新、仅 EEG 更新、仅 fNIRS 更新的局部反事实诊断；它们不是可部署候选或新独立 trial。N2 的 O2 已可运行时，对相同完整观测做独立批量求解比较。

不得裁剪 f、改变 substeps 后覆写旧失败、删除产生失败的 W 点，或将失败计数藏到成功子集分母中。数值步长敏感性只作解释，不能自动重新分类旧结果。

### 8.5 判读

降低 sigma_H 后 full-fit 小幅变差但遮挡、r 真值和配对增量改善：旧模型可能过多使用下游创新。增大 sigma_H 只使 Hb 更贴观测，r 与 EEG-only 几乎相同且 null 无增量：是更自由的重建，不是更好的共享 teacher。改变 sigma_r 改善脉冲真值恢复、越界减少：驱动先验值得修改。独立积分也在相同状态和 r 下过零：不只是 RK4 步长问题；若批量可行解明显更优而 filter 越界，优先研究更新/闭合，而非立即改生理方程。

## 9. N7：G–W 增益与时间尺度解耦

### 9.1 问题、参数与历史依据

新增 G=log[(beta/gamma)/(beta*/gamma*)]，与 W 独立拟合。固定参考阻尼比 Z=kappa*/(2 sqrt(gamma*))，映射由既有 multi_parameter_model 持有：

beta=beta* exp(G+2W)，gamma=gamma* exp(2W)，kappa=kappa* exp(W)。

保持 tau/alpha/E0/P0/Q0、EEG loading、观测噪声和过程噪声固定。当前 G=0 会锁住 beta/gamma；本实验检验解除该锁定是否改善遮挡预测，不能把 G 自动解释为个体生理效能。

历史桥接中，仅 HbO/HbR 共同乘 0.5，就使偏好 W=-0.5 的案例从 2/24 增至 18/24。A0 的 G 覆盖为 93.33%，但 A1 的 G 仅完成 55/60，生成与推断域失败仍保留；旧 kappa 跨 session 候选的 delta NLL 为 +5.288570，未优于固定模型。N7 是新的受限探索，不改写这些负结果。

### 9.2 三条冻结选择规则

候选网格 G={-0.6,-0.3,0,0.3,0.6}，W={-0.5,-0.25,0,0.25,0.5}，共 25 点；支持域仍为 G=[-0.6,0.6]、W=[-0.5,0.5]。所有组合接受现有物理及数值域检查，不扩边、不裁剪。

- GW：内折在全部 25 点选择，外折评价冻结选择。
- W-only：同一批内折结果的 G=0 五点子集。
- G-only：同一批内折结果的 W=0 五点子集。

三条规则共用每个内折的投影、目标、训练尺度与每点推断结果；G=W=0 复用 N1。三条规则各自保留 12 个选择折及 72 个外折 trial，按公共 B 风险、单模态退化、配对 null、合成真值和失败分母评分。W-only→GW 是新增自由度的主要消融，GW 的 10% 风险改善和不退化筛选以同折 W-only 为参考；固定 W=0 的 G-only 与 N5 a_N-only 是同自由度对照。N7 不同时放开 G 与 a_N；不同规则的外折分数不用于再次选择后重评。

### 9.3 合成与二维描述

除公共两套种子、六条件、每条件 16 trial 的面板外，N7 另有 true_g、true_w、true_gw、measurement_gain 四种生成条件，每套种子每条件 16 个独立 trial。预定偶/奇重复分别取 G=-0.3/+0.3、W=-0.25/+0.25、观测增益=0.75/1.5；非激活方向固定参考。每个真值输入共用 25 个 G–W 固定候选和 N5 四个非基线观测增益对照，G 与观测增益始终分开变化。观测增益只改变生成均值，原噪声实现不变。

真值只供生成器与评分器；拟合器收到预定候选参数及带噪输入。报告 r/clean EEG/HbO/HbR 恢复、区间、full 与整模态缺失及全部失败。额外面板是固定候选响应曲面诊断，不是参数恢复资格或 SBC，不以知晓真值选择实测候选。

共同坐标描述通道另计算 5 点 G × 9 点 W 的逐 trial 联合似然曲面，G=0 复用 N5 对应 W 点。按 session 输出完整性、两轴边界、delta log L<=2 的近优网格及近优 teacher/r 差异；该阈值仅为描述，不产生 95% 参数区间。N5/N7 的共同 gauge 不进入外折评分。

### 9.4 调度与判读

N7 先做全网格映射/导数/固定量软件检查和一个独立 G/W 合成试跑。核心配额 16000 次模型求解，优先三条内折规则、外折比较和匹配合成；二维加密及额外真值/失配面板进入扩展队列。全局仍为一个最多 16 worker 的池、8 小时硬截止，N1–N7 按族轮转。未完成点和失败点保留。

GW 相对 W-only 改善遮挡且保留配对增量、真值质量，才支持继续增加 G；G-only 与观测增益效果近似时，只支持有效幅度适配，不能定位为生理来源。仅训练似然变好、W 离开边界或参数沿补偿谷移动均不单独算成功。

输出 N7/rule_comparison.csv、gw_session_surfaces.csv、synthetic_response_surfaces.csv、逐 trial 指标和 compact 轨迹，纳入统一总报告。实现复用既有参数 owner、选择器、评分器和单一调度器。

## 10. 今晚候选筛选：不是新 teacher 资格

所有新阈值只用于明日复核优先级，不能回写旧 PASS/FAIL。

初步优先候选需要：核心评估身份完整；相同目标下的 B 风险至少下降 10%；任一中心目标 NRMSE 不增加超过 0.05 个训练 SD；至少 2/3 被试方向一致；配对增量不出现新的明显退化；合成 r/clean 真值恢复不因追噪声恶化。N4 跨 EEG 特征比较仅使用共同 fNIRS 风险，不使用跨定义的 B。

没有完整新候选时仍报告最佳诊断方向。失败率下降可作为工程进展单列；不能把缺失难例后的条件均值下降当成完整队列改善。基线与候选的共同成功子集分数只能作补充，必须同时显示固定全部 trial 的完成分母。

不要仅凭覆盖接近 95%、W 不再贴边或 r 曲线更平滑挑选模型。提高覆盖若仅靠区间变宽，不算 teacher 均值质量改善。

最多对两项结构上互补的单因素改进做后续组合，例如“局部 EEG + 观测噪声倍率”。组合选择只能用内折结果，外折在组合选择冻结后一次评分；不能用同一批外折结果挑前两名再重评并称其独立验证。没有剩余预算则只输出组合建议。

## 11. 无人值守实现与调度

### 11.1 最小实现边界

使用现有 experiments/evaluate_ssm_overnight_diagnostics.py + 版本化配置 + 按族组织的小函数。保留现有科学实现 owner，不建立新的实验管理平台。N2 的新 solver 可放一个独立模块，其他族调用既有函数。

可复用的现有文件包括：

- experiments/evaluate_step5.py：原生 trial 准备、投影、mixture 与评分；注意旧 scorer 只适用于其观测合同。
- experiments/evaluate_step5_observation_diagnostic.py：训练限定 loader、局部 EEG、同折线性模型与创新诊断。
- experiments/evaluate_step5_observation_repair.py：bridge、冻结输入 replay 和新 mask 算子回归。
- src/inference/t3a_balloon_joint_ssm.py：联合 filter、TrajectoryObservationSpec、线性时序参考。
- src/inference/t3a_balloon_robust_ssm.py：六状态方程、Jacobians、flow-domain owner。

实现阶段可以分工，计算开始前由一个协调者集成并冻结代码快照；禁止多个 agent 在正在运行的同一核心文件上同时修改。

### 11.2 执行前预检

验证允许的 72 个 trial 身份；验证外折/内折无交叉；运行现有相关测试和新分支最小线性/隐藏值/单位检查；每族试跑一个低成本 cell，记录 CPU/RAM；完成 task_table 后冻结。

预检失败仅隔离该族/该候选。对私有或保护数据的边界失败则停止对应数据入口，不能换 loader 绕开。不得升级现有环境依赖来解决今晚任务，先检查本机安装 API。

### 11.3 资源预算而非运行时承诺

全局一个 worker 池，BLAS/OMP/MKL 线程均为 1。默认：

P=min(16, max(1, available_cpu-2), floor(0.60×available_RAM / pilot_peak_job_RAM))。

RAM 不足一个 job 时标记资源不足，不用 max(1,...) 强行启动。N2 使用其自己的高内存估计，调度时按内存令牌限制并发。GPU 不作为今晚方案的必要依赖；是否可用不决定启动。

硬截止默认 8 小时、可在启动配置中修改；这是作业终止预算，不是预计完成时长。普通 cell 超时 900 秒，非线性批量 cell 1800 秒；新非线性优化每起点最多 200 次目标评估。所有求值调用还记录实际数量，避免 max_nfev 未包含数值 Jacobian 调用时超出真实预算。[W3]

### 11.4 公平队列

第一层：每族的软件预检、一个基线、一个关键对比。
第二层：N1 完整报告；N3/N4/N5 一维改进候选；N6 主敏感性；N7 三条冻结规则；N2 已就绪分支。
第三层：二维曲面加密、整 session 留出、额外频带、W=-0.5 敏感性、更多合成重复。

同一层按实验族轮转，不让 N2 开发/大矩阵或 N5 参数网格耗尽全部资源。到硬截止时不启动新 cell，终止仍运行的 cell 并保存已完成检查点，立即汇总现有状态；中断项标记 timeout。所有 core cell 的队列顺序固定，不按早期结果挑好看的案例先算。

建议核心计算配额按模型求解数量限制：N1 2500、N2 3500、N3 3500、N4 4500、N5 5000、N6 3500、N7 16000。额度是上限，不是必须消耗；任务生成器必须在启动前列出实际数量，超额部分进入扩展队列。不同 solver 成本不同，另受超时和内存限制。

### 11.5 故障与恢复

每个 cell 都写一行状态：completed、data_unavailable、not_implemented、failed_domain、failed_numerical、failed_contract、timeout、not_started_budget。成功案例即时原子写出，不等整族结束。后续聚合从 task_table 左连接结果，不从成功文件列表猜分母。

不自动扩 W/增益/噪声范围，不修改阈值，不为失败换 seed，不换 fNIRS 通道逃避失败，不把 NaN 填成零，不启用 tokenizer 训练。已知确定性失败不循环重试；同输入数值验证另建诊断 cell，保留原结果。

### 11.6 必须留下的产物

一个 run 目录，包含 resolved_config.yaml、task_table.csv、case_status.jsonl、source snapshot、fold/scope inventory，以及 N1–N7 的逐 trial 指标和 summary。最后统一输出 OVERNIGHT_REPORT.md、candidate_table.csv、direction_decision_table.csv、failure_attribution.csv。

保存每个候选的必要 compact 轨迹（r、clean mean、可用 variance、观测 mask、实际处理后的输入和评分目标）；不得只保存图或聚合数字。旧 native 原始数组不重复提交 Git，compact 数值图表、失败记录和配置按现有保留规则管理，不自动对外发布。

## 12. 明日修改方向决策表

| 关键结果 | 优先修改方向 | 不能据此声称 |
|---|---|---|
| N2 同步时间算子后均值和覆盖同时改善 | 修正观测时间算子、噪声协方差和预测条件化 | 实测 neural clean truth 已得到验证 |
| N3 有限噪声倍率改善共同目标和正确配对增量 | 调整模态相对置信度 | 真实噪声被精确测量 |
| N4 局部/EOG 清理改善固定 fNIRS 目标，超过 null | 修正输入特征与空间对应 | 后部 EEG 普遍优于前部 |
| N5 仅 a_N 就解除 W 压力并改善外折结果 | 修改幅度标定，暂不增加生理自由度 | a_N 是个体生理增益 |
| N5 参数沿谷变化但 teacher 稳定 | 接受参数不可辨识，固定/边缘化参数 | 原始参数已恢复 |
| N6 较小下游过程噪声改善遮挡与 r 真值 | 减少下游过程创新对重建的补偿 | replay gap 是 PID/private 信息比例 |
| N6 相同状态与驱动的独立积分仍过零 | 检查状态过程支持/动力学适用范围 | 被试真实脑血流归零 |
| 简单线性 joint 也不超过任务模板和 pairing | 限定当前坐标的可识别共同成分，再检查任务/时间尺度 | 原始数据绝无跨模态神经关系 |
| N7 GW 超过 W-only，且优于观测增益替代并保留配对/真值质量 | 继续验证 G–W 独立适配 | G 已成为可解释的个体生理参数 |
| 所有改进只改善 full-fit，不能改善遮挡/真值/null | 暂停增加参数，重新审视观测与共享假设 | teacher 去噪已成功 |

## 13. 给执行 agent 的最终任务描述

按本方案维护一个有界诊断 suite。先冻结共同数据边界、分折、baseline 和任务清单，再独立并行运行 N1–N7。优先保证每一族都有可解释结果，不把整体串行依赖于新非线性时序 solver。不要重构 tokenizer，不增加 latent 状态，不更改旧实验判定。明确区分旧基线、机制诊断、新观测族探索和 teacher 资格。每个失败保留固定身份；每个候选都报告对 EEG、HbO、HbR 各自付出的代价。达到预算后停止扩展并自动生成总报告，无需等待用户确认下一小步。

## 资料依据

[R1] 仓库固定快照：experiments/configs/physiology_semantic_tokenizer/step5_observation_repair_v2.yaml。
[R2] 同快照：experiments/configs/physiology_semantic_tokenizer/step5_observation_diagnostic_v1.yaml。
[R3] 同快照：src/inference/t3a_balloon_joint_ssm.py，TrajectoryObservationSpec 与 smooth_balloon_trajectory_reference。
[R4] 同快照：experiments/runs/physiology_semantic_tokenizer/step5/20260909_observation_repair_v2_verified/summary.md。
[R5] 同快照：src/inference/t3a_balloon_robust_ssm.py，过程扩散、固定参数与流量域约束。
[W1] Stan User's Guide，Simulation-Based Calibration Checking；访问于 2026-09-09。
[W2] MNE 官方文档，Repairing artifacts with regression / EOGRegression；访问于 2026-09-09。
[W3] SciPy 官方文档，scipy.optimize.least_squares；访问于 2026-09-09。本地实际版本在预检中记录，不默认升级。
