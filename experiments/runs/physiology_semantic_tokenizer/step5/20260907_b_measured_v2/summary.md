# Step5B 阶段报告：已知被试与 session 中的新 trial

注册 18 被试 × 6 留出 trial = 108；主候选完成 100，失败 8；主候选具有完整六条留出结果的被试 15。

训练范围为 subjects 01–18、sessions 01/03/05，每 session 8 train / 2 heldout。此处不是整 session 留出，也不是新被试泛化。
输入和评分目标使用不同原始 trial 处理路径，隐藏值在滤波、功率和重采样前移除。PCA、通道选择、共同 HbO/HbR 比例、噪声尺度及参数后验仅使用训练 trial；留出 trial 不更新参数权重。
own_history 对照使用同模态的未遮挡上下文，包括未来上下文；所有结果属于 fixed-interval 遮挡重建，不是因果预测。HbO/HbR 是显式近似 MBLL 后的共同缩放坐标，不是个体绝对生理浓度。

| 候选 | measured 核心资格 | 未通过检查 |
|---|---|---|
| U0_FIXED | False | complete_registered_subjects_and_trials, physical, shared_information, not_driven_by_one_subject, masked_observation_accuracy, synthetic_teacher_eligible |
| U1_W | False | complete_registered_subjects_and_trials, shared_information, not_driven_by_one_subject, masked_observation_accuracy, parameter_mixture_resolution, boundary_insensitive_driver |

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
| U1_W | center_EEG | own_history | -0.945536 | [-1.137059, -0.746298] | -1.0027428376046272 |
| U1_W | center_EEG | own_history_and_task | -0.967085 | [-1.156963, -0.765724] | -1.0180954757156704 |
| U1_W | center_EEG | independent_pairing | 0.017541 | [-0.186927, 0.209807] | -0.029205897885713362 |
| U1_W | center_EEG | circular_shift | -0.209358 | [-0.399012, -0.024722] | -0.2583160391220635 |
| U1_W | center_fNIRS | own_history | 0.070882 | [0.022813, 0.115640] | 0.06162182406820531 |
| U1_W | center_fNIRS | own_history_and_task | -2.686687 | [-3.487173, -1.864971] | -2.873681181894574 |
| U1_W | center_fNIRS | independent_pairing | 0.004727 | [-0.024500, 0.039029] | -0.006560633753096958 |
| U1_W | center_fNIRS | circular_shift | -0.002755 | [-0.044061, 0.040423] | -0.011420028815975895 |

| 候选 | 遮挡目标 | NRMSE（训练坐标SD单位） | 带噪观测95%覆盖 |
|---|---|---:|---:|
| U0_FIXED | center_EEG/EEG | 1.9114 | 0.7681 |
| U0_FIXED | center_fNIRS/HbO | 1.4166 | 0.2451 |
| U0_FIXED | center_fNIRS/HbR | 2.4447 | 0.3847 |
| U1_W | center_EEG/EEG | 1.9358 | 0.7646 |
| U1_W | center_fNIRS/HbO | 1.1701 | 0.2785 |
| U1_W | center_fNIRS/HbR | 2.6163 | 0.3410 |

覆盖率使用 Gaussian-moment 95% 预测区间，仅为带噪观测诊断；没有真实 r 或 clean trajectory，不能声称潜在状态覆盖已验证。先按 trial，再按 subject 汇总，未把自相关时间点当独立样本。
完整 posterior 边界/网格、参数混合加密、去边界敏感性、整模态缺失及配对方差变化保留在 posterior/case/summary JSON。固定模型仅作基线，不因 measured 某个分数较好而跳过其 synthetic 敏感性失败。

可进入全面 UQ 的核心 teacher：**无**。

subjects 19–23 没有作为新确认样本使用；subjects 24–29 未读取。没有 tokenizer 训练、外部发布或 protected evaluation。
