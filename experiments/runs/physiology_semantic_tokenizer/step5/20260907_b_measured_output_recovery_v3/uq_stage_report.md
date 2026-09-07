# 第四阶段：全面 UQ 未执行

**Step5B 未获得合格实测核心 teacher，因此按 ssm_next.md 的阶段顺序停止。全面 UQ 的统计结论未被检验。**

前置失败：complete_registered_subjects_and_trials, shared_information, not_driven_by_one_subject, masked_observation_accuracy, parameter_mixture_resolution, boundary_insensitive_driver。详见 [Step5B 报告](summary.md)。
没有用跨被试/跨模态森林图、ICC、conformal 或 precision weighting 绕过核心资格；这些分析未运行。

A1/B 已保存的状态方差、参数均值间方差、观测噪声与遮挡覆盖属于资格诊断。它们不是合格 teacher 的全面 UQ，也不证明真实潜在轨迹覆盖。
clean teacher 方差为 E[Var(h(x)|D,phi)] + Var(E[h(x)|D,phi])；仅带噪观测预测再加 Student-t 观测方差。状态后验方差条件于固定过程模型，不包含全部模型失配。

若未来核心资格成立，校准对象应继续明确为已知被试和 session 中的新 trial。普通 cross-fit 不被宣称具有标准 split-conformal 的有限样本保证；风险排序用于判断是否可测试精度加权，不代替 teacher 资格。

本轮未创建可供训练的合格 teacher 导出，未训练 tokenizer。监督目标仍是同模态观测空间 clean teacher 切片，r 仅作共享过程诊断；默认统一权重。

subjects 19–23 未启动 replication；subjects 24–29 未读取。停止来自实测资格结果，不是等待额外授权。
