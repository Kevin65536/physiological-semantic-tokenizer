# R 系列 EEG–fNIRS 共享驱动实验总结

_实验日期：2026-07-28｜报告整理：2026-07-29｜当前决策：停止晋级，不进入 R2-P/VQ_

## 摘要

本轮实验检验了一个比“EEG 与 fNIRS 是否相关”更严格的问题：从两种模态的原始窗口中，能否得到一个跨被试稳定、可由两侧观测、并具有足够物理重建能力的共享驱动坐标，进而为离散语义 token 提供可信监督。实验依次覆盖原始信号滞后基线、开发期连续可观测性、population-frozen teacher 资格门，以及资格失败后的有限诊断。

最终证据不支持继续进入 token 化。原始 alpha–HbO 滞后关联没有在验证被试中复现；单种子连续模型未达到双模态同时为正的标准；population-frozen teacher 虽然通过了 jointness、有限扰动稳定性、双侧 raw-only 可观测性和时间错位 null，但 HbO 物理重建只在 3/5 名验证被试中超过冻结阈值，因此正式资格失败。后续诊断显示，主要问题不是 correction 数值退化，而是跨被试血流动力学异质性、任务相位基线过强，以及 HbO/HbR 的非对称补偿。D1B 验证又因 NumPy 布尔值序列化缺陷在 endpoint 计算前终止，所以其科学结果是“未判定”，而不是通过或失败。

这组结果给出的核心启发是：目标坐标“可解码”并不等于它已经获得生理资格；局部漂亮的时滞或 patch 结果也不能替代预注册的全轨迹和多重比较判定。现阶段最可靠的结论是保留连续模型与数据合同代码，停止 R2-P、VQ 和 token 共现解释。

## 实验边界与共同合同

所有 R 系列实验都使用 `eeg_fnirs_single_trial` 的 mental-arithmetic 数据。开发集由 subjects 01–18 构成，验证集由 subjects 19–23 构成，subjects 24–29 保持保护状态且 measured arrays 从未解引用。R1-P 数据共有 23 名开发被试、3 个 session（01/03/05）、2 个 condition（BL/MA）和每个 cell 10 个 trial，共 1,380 个窗口，其中 1,080 个用于拟合，300 个只允许 pure apply。

推断的生物学单位始终是 subject。模型先在被试内累计误差或关联统计量，再对被试等权汇总；区间使用 subject bootstrap，null 使用冻结的 trial pairing、subject block permutation 或 whole-patch circular shift。任何开发期正结果都不能打开保护集，也不能单独授权 VQ。R2-D 还存在 checkpoint 由同一验证集选择的条件性，因此只属于探索性可行性证据。

| 阶段 | 主要问题 | 结果 | 决策 |
| --- | --- | --- | --- |
| R0-P | 原始 EEG bandpower 是否先于 fNIRS 形成稳定滞后关联 | 预注册 alpha–HbO 方向未复现，30 个诊断均未过 FWER | raw association 阴性 |
| R1-D | 逐被试 joint teacher correction 是否存在可解释几何 | correction RMS 约为 train SD 的 0.526，8.5 s 有探索性峰 | 仅生成假设 |
| R2-D | 两种模态能否独立预测完整共享轨迹 | EEG 边界性为正，fNIRS 为负，双侧标准失败 | 不授权 VQ |
| R1-P | population-frozen teacher 是否通过六门全合取资格 | G2 失败；G3–G6 通过；G1 为 dtype 合同无效 | 不进入 R2-P |
| D1B | 个体动态适配能否修复跨被试物理重建 | train-only 呈 chromophore 非对称；validation 在 endpoint 前中止 | 科学结果未判定 |

## R0-P：原始信号滞后基线

R0-P 将每个 20 s 窗口划为十个 2 s patch。EEG 特征是六通道等权的 theta、alpha 和 beta 对数带功率，fNIRS 特征是 HbO/HbR 均值；正向时滞为 2–10 s。主终点预注册为 negative alpha–HbO Fisher-z 曲线在 2–10 s 上的归一化 AUC。10,000 次 cell 内 fNIRS whole-trial permutation 保留 fNIRS 时间结构和 HbO/HbR 依赖，但打破 EEG–fNIRS trial 配对；另用 10,000 次 subject bootstrap 给出区间。

训练集 AUC 为 0.00537，95% CI [−0.01534, 0.02672]，单侧置换 \(p=0.3319\)，18 名被试中 9 名为预注册方向。验证集 AUC 为 −0.02202，95% CI [−0.06685, 0.04020]，\(p=0.8224\)，仅 1/5 名被试为预注册方向。对 3 个频带、2 个 chromophore 和 5 个 lag 形成的 30 项诊断族进行 max-stat FWER 校正后没有结果通过 0.05。

这不是对所有神经血管耦合形式的否定，因为分析仅覆盖同 trial、离线、低维特征的滞后关联。它确实建立了一个重要底线：后续 token 曲线若只表现为可见峰值，而不能超过同一 subject-equal raw baseline、配对 null 和时序 null，就不能获得新的生理解释。

## R1-D 与 R2-D：开发期连续可观测性

R1-D 的逐被试 teacher geometry 显示 joint 与 EEG-only 坐标之间存在非零 correction，correction RMS 约为训练坐标标准差的 0.526。历史 8.5 s 位置出现约 +0.130 的探索性峰，但这一 teacher 使用开发期逐被试拟合，并不是 population-frozen 模型，因此该峰只能用于提出后续定位假设。

R2-D 使用两个相互独立的 raw-window encoder 和同一 trajectory decoder，分别从 EEG 或 fNIRS 预测完整 \(r_J\) 轨迹。它只运行了一个种子，并以 subjects 19–23 汇报相对于 train-only condition-by-relative-time phase baseline 的 subject-equal \(\Delta R^2\)。EEG 为 0.031296，95% CI [−0.002166, 0.069625]，3/5 名被试为正；fNIRS 为 −0.023018，95% CI [−0.035806, −0.007696]，1/5 为正；两模态等权结果为 0.004139，95% CI [−0.014133, 0.023801]。双侧连续可观测性因此失败。

在检查十个 patch 后，8 s patch 的 pointwise 结果较好：EEG 为 0.0930，未校正 95% CI [0.0378, 0.1481]，5/5 为正；fNIRS 为 0.0370，未校正 95% CI [0.0046, 0.0852]，4/5 为正。但把两种模态与十个 patch 作为 20-cell family 后，同时置信带在该位置均跨越零：EEG [−0.0164, 0.2024]，fNIRS [−0.0724, 0.1465]。因此 8 s 只能是定位线索，不能改写全轨迹主终点。

## R1-P：population-frozen teacher 正式资格

R1-P 先在 subjects 01–18 上拟合一个参数 bundle 和一个共同 scalar gauge，再对 subjects 19–23 零拟合 pure apply。结构审计确认 1,380 个 sample key、raw-view 与 trajectory registry 的顺序、session/condition/event index、mask 和形状完全对齐；validation fit、validation normalization 和 protected dereference 计数均为零。所有行共享同一 parameter bundle 和 train-only gauge。

正式资格采用六门全合取规则。文件中的 G1 与 G2 为 `FAIL`，G3–G6 为 `PASS`。审计后，G1 被分类为实现合同无效而不是科学失败：builder 把 sidecar 写成 float32，qualification 却用 float64 replay 与 `1e-10` 绝对容差比较。将 replay cast 到 float32 后，\(r_J/r_E\) 共 552,000 个点均 bitwise 相同，最大 ULP 为 0。该事实不能追认 formal-v3 通过，因为文件状态不可事后修改，而且 G2 仍独立失败。

G2 的 HbO subject-equal gain 为 0.234535，95% CI [0.072574, 0.433652]，虽然区间下界高于冻结阈值 0.069875，但只有 3/5 名被试超过阈值，未达到 4/5 consistency；HbR 为 0.266190，95% CI [0.103207, 0.401147]，4/5 通过。具体地，subjects 20 和 23 的 HbO 不足，subject 20 的 HbR 也不足。群体均值的正迹象因此不能代替跨被试一致性。

其余四门提供了有限但真实的积极证据。G3 中 HbO/HbR jointness gain 分别为 0.504361 和 0.519989，均为 5/5，correction RMS ratio 中位数为 0.511822，说明 fNIRS update 没有被忽略。G4 的三个预注册 train-only perturbation 的 subject-median CCC 分别为 0.99994、0.90934 和 0.90762，均达到 5/5。G5 的 EEG 与 fNIRS raw-only \(\Delta R^2\) 分别为 0.131446 和 0.138637，95% CI 分别为 [0.102743, 0.166048] 和 [0.091124, 0.193357]，均为 5/5。G6 的 heldout level 与 within-patch first-difference gain 分别为 0.517423 和 0.513631，均超过冻结的 circular-shift null。

这些通过项说明候选坐标不是纯 EEG 坐标、零 correction 或任意平滑错位曲线。然而 \(r_J\) 本身来自 joint teacher，raw-only 可解码仍可能包含任务相位、共享 teacher circularity、系统生理或双向窗口上下文。它们不能弥补 G2 所揭示的跨被试物理充分性不足。正式决策保持 `promotion_eligible=false` 和 `do_not_enter_r2_p`。

## Post-formal 诊断与 D1B

失败分解表明低表现主要集中在 subject 20 的 HbO/HbR 和 subject 23 的 HbO。这些行的 joint SSE 确实偏高，而不是由极小的 baseline denominator 人为放大。对 2 chromophores × 2 metrics × 2 conditions × 10 patches 构成的 80-cell family 计算同时区间后，所有区间均跨越零，说明不能从个别 patch 反向选择一个稳定失败位置。

correction geometry 也不支持固定群体相位机制。验证集 8.5 s 的 signed correction 为 −0.039398，200 点同时区间 [−0.304021, 0.225225]；全局绝对峰转移到 −1.2 s、幅度约 +0.141。把训练期 phase template 转移到验证集时，相对于零基线的 \(\Delta R^2\) 为 −0.02130，95% CI [−0.05803, 0.01172]，2/5 为正；相对于 subject mean 为 −0.08939，95% CI [−0.14862, −0.03016]，0/5 为正。correction 有广泛能量，但没有稳定共同的 signed phase。

跨 session affine 适配提供了机制线索而非确认性结论。HbO 的 pointwise improvement 为 +0.033657，95% CI [0.003679, 0.069540]，4/5 为正，但同时区间 [−0.017176, 0.084490] 跨零；HbR 为 −0.017307，仅 1/5 为正。这个非对称方向促成 D1B 的有限个体适配搜索。

D1B train-only 使用 subjects 01–18 的 nested leave-one-subject-out 选择 shrinkage。0.1 在 16/18 个 outer folds 被选中，0.3 在 2/18 被选中。最终 descriptive endpoints 为：HbO population-base −0.000723，同时 95% CI [−0.095298, 0.093851]，10/18 为正；HbR population-base 0.213111，[0.118536, 0.307685]，14/18 为正；HbO dynamic increment 0.109474，[0.014900, 0.204048]，15/18 为正；HbR dynamic increment −0.000187，[−0.094761, 0.094388]，7/18 为正。`tau0` 在 54 个 held-session folds 中有 37 次达到 5 s 上界，说明它更像模型失配补偿，而不是已识别的生理 transit time。

唯一一次 D1B validation v2 measured-access attempt 覆盖 subjects 19–23 的 300 个窗口、125 个候选、44 个 donor derangements，以及各 10,000 次 pairing、shift 和 bootstrap transforms。进程完成 125/125 个 score-cache chunks，写出约 14.28 MB staging evidence 后，在写 `null_transform_seal.json` 时退出：`numpy.all` 产生的 `numpy.bool_` 未被 JSON serializer 接受。故障发生在 observed endpoint、null、同时区间和 gate evaluation 之前，最终目录没有原子发布。其科学状态严格记为“未判定”；已有 cache 不被事后读取成新结论，也不修复后重跑。

## 结果带来的启发

第一，弱且不复现的 raw lag baseline 说明同 trial 的宽泛时滞相关不是可靠语义目标。后续工作如果恢复，应该优先预测相对于任务相位、fNIRS 自身历史和系统性协变量的 held-out predictive residual，而不是优化 token 共现曲线。

第二，R1-P 同时出现 G3–G6 通过与 G2 失败，清楚地区分了“数学上非退化且可解码的共享坐标”和“跨被试物理上合格的 teacher”。可观测性是必要条件，但不是构念效度的充分条件。

第三，主要失败呈现 subject 与 chromophore 异质性。HbR scale correction 与 HbO dynamic compensation 可以在训练期分别改善不同终点，却没有形成一个双侧共同机制。下一代模型需要把这种异质性作为显式竞争假设，并用独立数据判定，而不能事后删被试、改阈值或只报告较好的 chromophore。

第四，8–8.5 s 的局部模式在 pointwise 图上可见，却在 family-level uncertainty 下消失。时间定位和 patch 搜索必须和多重比较一起设计；先看全曲线再挑峰值不能作为 confirmatory evidence。

第五，float32 replay 和 `numpy.bool_` 序列化事件说明，数值精度、JSON 边界和原子发布都是实验合同的一部分。未来任何确认性 runner 都应在打开 measured data 前，用纯合成 fixture 覆盖 dtype-aware replay、完整 payload serialization、cache-to-summary 路径和最终原子 rename。

## 决策与保留范围

本轮保留 R0-P、R1-P 和 R2-D 的数据合同、核心模型、必要配置、冻结注册表、主 runner/auditor 及其回归测试。重复的阶段报告、post-formal 一次性分析器、D1B 预访问治理栈、source snapshots 和未执行的 R2-P 草案不进入版本库；它们的科学信息已经在本文集中保留。运行产物继续位于被 Git 忽略的 `experiments/runs/` 和本地 cache 中。

若未来重新启动研究，应把新的独立 holdout、重新冻结的 estimator/null/threshold 合同和端到端发布测试作为前置条件。既有 subjects 19–23 不应再被描述为首次确认性验证，subjects 24–29 继续保持关闭。在这些条件满足前，不训练 VQ，不制作或解释 token co-occurrence，也不启动 D2 或 R2-P。
