# Step5B 阶段报告：已知被试与 session 中的新 trial

注册 18 被试 × 6 留出 trial = 108；完成 100，失败 8；具有完整六条留出结果的被试 15。

训练范围为 subjects 01–18、sessions 01/03/05，每 session 8 train / 2 heldout。此处不是整 session 留出，也不是新被试泛化。
输入和评分目标使用不同原始 trial 处理路径，隐藏值在滤波、功率和重采样前移除。PCA、通道选择、共同 HbO/HbR 比例、噪声尺度及参数后验仅使用训练 trial；留出 trial 不更新参数权重。
own_history 对照使用同模态的未遮挡上下文，包括未来上下文；所有结果属于 fixed-interval 遮挡重建，不是因果预测。HbO/HbR 是显式近似 MBLL 后的共同缩放坐标，不是个体绝对生理浓度。

| 候选 | measured 核心资格 | 未通过检查 |
|---|---|---|
| U0_FIXED | False | complete_registered_subjects_and_trials, physical, shared_information, not_driven_by_one_subject, masked_observation_accuracy, synthetic_teacher_eligible |
| U1_W | False | complete_registered_subjects_and_trials, shared_information, not_driven_by_one_subject, masked_observation_accuracy, training_posterior_resolution, parameter_mixture_resolution, boundary_insensitive_driver |

| 候选 | 遮挡 | 对照 | log-score 增量 | subject bootstrap 95% CI | 最小留一被试均值 |
|---|---|---|---:|---|---:|
| U0_FIXED | center_EEG | own_history | -0.936048 | [-1.130193, -0.743458] | -0.9883863548332136 |
| U0_FIXED | center_EEG | own_history_and_task | -0.957596 | [-1.152533, -0.762087] | -1.003738992944257 |
| U0_FIXED | center_EEG | independent_pairing | 0.011206 | [-0.173088, 0.179069] | -0.03789416833467297 |
| U0_FIXED | center_EEG | circular_shift | -0.435518 | [-0.612292, -0.255673] | -0.4776305937065138 |
| U0_FIXED | center_fNIRS | own_history | 0.074078 | [-0.009886, 0.155672] | 0.05168166938502404 |
| U0_FIXED | center_fNIRS | own_history_and_task | -2.979114 | [-3.832020, -2.104806] | -3.175666939218292 |
| U0_FIXED | center_fNIRS | independent_pairing | -0.003601 | [-0.049568, 0.056874] | -0.025603758560344945 |
| U0_FIXED | center_fNIRS | circular_shift | 0.019447 | [-0.069274, 0.107077] | -0.00906343689201286 |
| U1_W | center_EEG | own_history | -0.946048 | [-1.137706, -0.746858] | -1.0032787280551598 |
| U1_W | center_EEG | own_history_and_task | -0.967596 | [-1.157389, -0.765964] | -1.0186313661662036 |
| U1_W | center_EEG | independent_pairing | 0.017459 | [-0.186991, 0.209708] | -0.029309534125334064 |
| U1_W | center_EEG | circular_shift | -0.210157 | [-0.399908, -0.025294] | -0.25915746580455895 |
| U1_W | center_fNIRS | own_history | 0.070772 | [0.022701, 0.115525] | 0.061457173625188474 |
| U1_W | center_fNIRS | own_history_and_task | -2.687187 | [-3.487466, -1.865180] | -2.874199871763802 |
| U1_W | center_fNIRS | independent_pairing | 0.004761 | [-0.024505, 0.039102] | -0.006548297962040062 |
| U1_W | center_fNIRS | circular_shift | -0.002630 | [-0.043979, 0.040717] | -0.011336200278474021 |

| 候选 | 遮挡目标 | NRMSE（训练坐标SD单位） | 带噪观测95%覆盖 |
|---|---|---:|---:|
| U0_FIXED | center_EEG/EEG | 1.9114 | 0.7681 |
| U0_FIXED | center_fNIRS/HbO | 1.4166 | 0.2451 |
| U0_FIXED | center_fNIRS/HbR | 2.4447 | 0.3847 |
| U1_W | center_EEG/EEG | 1.9361 | 0.7646 |
| U1_W | center_fNIRS/HbO | 1.1706 | 0.2778 |
| U1_W | center_fNIRS/HbR | 2.6159 | 0.3417 |

覆盖率使用 Gaussian-moment 95% 预测区间，仅为带噪观测诊断；没有真实 r 或 clean trajectory，不能声称潜在状态覆盖已验证。先按 trial，再按 subject 汇总，未把自相关时间点当独立样本。
完整 posterior 边界/网格、参数混合加密、去边界敏感性、整模态缺失及配对方差变化保留在 posterior/case/summary JSON。固定模型仅作基线，不因 measured 某个分数较好而跳过其 synthetic 敏感性失败。

可进入全面 UQ 的核心 teacher：**无**。

subjects 19–23 没有作为新确认样本使用；subjects 24–29 未读取。没有 tokenizer 训练、外部发布或 protected evaluation。
