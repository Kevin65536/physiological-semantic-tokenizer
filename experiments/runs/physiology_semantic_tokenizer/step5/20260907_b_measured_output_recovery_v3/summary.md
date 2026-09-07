# Step5B 阶段报告：已知被试与 session 中的新 trial

注册 18 被试 × 6 留出 trial = 108；主候选完成 106，失败 2；主候选具有完整六条留出结果的被试 16。

训练范围为 subjects 01–18、sessions 01/03/05，每 session 8 train / 2 heldout。此处不是整 session 留出，也不是新被试泛化。
输入和评分目标使用不同原始 trial 处理路径，隐藏值在滤波、功率和重采样前移除。PCA、通道选择、共同 HbO/HbR 比例、噪声尺度及参数后验仅使用训练 trial；留出 trial 不更新参数权重。
own_history 对照使用同模态的未遮挡上下文，包括未来上下文；所有结果属于 fixed-interval 遮挡重建，不是因果预测。HbO/HbR 是显式近似 MBLL 后的共同缩放坐标，不是个体绝对生理浓度。

| 候选 | measured 核心资格 | 未通过检查 |
|---|---|---|
| U0_FIXED | False | complete_registered_subjects_and_trials, physical, shared_information, not_driven_by_one_subject, masked_observation_accuracy, synthetic_teacher_eligible |
| U1_W | False | complete_registered_subjects_and_trials, shared_information, not_driven_by_one_subject, masked_observation_accuracy, parameter_mixture_resolution, boundary_insensitive_driver |

| 候选 | 遮挡 | 对照 | log-score 增量 | subject bootstrap 95% CI | 最小留一被试均值 |
|---|---|---|---:|---|---:|
| U0_FIXED | center_EEG | own_history | -0.979517 | [-1.177254, -0.771405] | -1.0312646641573016 |
| U0_FIXED | center_EEG | own_history_and_task | -1.011423 | [-1.227518, -0.795067] | -1.0580786728233087 |
| U0_FIXED | center_EEG | independent_pairing | 0.010287 | [-0.157566, 0.170979] | -0.03560030141744709 |
| U0_FIXED | center_EEG | circular_shift | -0.466831 | [-0.647104, -0.285658] | -0.5082238938074668 |
| U0_FIXED | center_fNIRS | own_history | 0.071510 | [-0.002809, 0.149652] | 0.05043519103152936 |
| U0_FIXED | center_fNIRS | own_history_and_task | -2.905587 | [-3.755609, -2.051252] | -3.084134742558855 |
| U0_FIXED | center_fNIRS | independent_pairing | -0.003460 | [-0.046130, 0.052068] | -0.02398601367074518 |
| U0_FIXED | center_fNIRS | circular_shift | 0.022340 | [-0.061163, 0.105459] | -0.004077367029058835 |
| U1_W | center_EEG | own_history | -0.989627 | [-1.190985, -0.781467] | -1.0459593854684588 |
| U1_W | center_EEG | own_history_and_task | -1.021534 | [-1.238651, -0.805950] | -1.0727733941344657 |
| U1_W | center_EEG | independent_pairing | 0.015133 | [-0.170623, 0.193754] | -0.028657694330202862 |
| U1_W | center_EEG | circular_shift | -0.230278 | [-0.405444, -0.053843] | -0.27736651568461623 |
| U1_W | center_fNIRS | own_history | 0.068871 | [0.023504, 0.112521] | 0.06009455517416757 |
| U1_W | center_fNIRS | own_history_and_task | -2.586422 | [-3.410601, -1.769865] | -2.7542658478270834 |
| U1_W | center_fNIRS | independent_pairing | 0.003454 | [-0.023721, 0.034859] | -0.007166584841086644 |
| U1_W | center_fNIRS | circular_shift | 0.000630 | [-0.039335, 0.040172] | -0.007232047107436037 |

| 候选 | 遮挡目标 | NRMSE（训练坐标SD单位） | 带噪观测95%覆盖 |
|---|---|---:|---:|
| U0_FIXED | center_EEG/EEG | 1.9291 | 0.7533 |
| U0_FIXED | center_fNIRS/HbO | 1.4354 | 0.2363 |
| U0_FIXED | center_fNIRS/HbR | 2.3776 | 0.4115 |
| U1_W | center_EEG/EEG | 1.9554 | 0.7487 |
| U1_W | center_fNIRS/HbO | 1.1928 | 0.2695 |
| U1_W | center_fNIRS/HbR | 2.5521 | 0.3646 |

覆盖率使用 Gaussian-moment 95% 预测区间，仅为带噪观测诊断；没有真实 r 或 clean trajectory，不能声称潜在状态覆盖已验证。先按 trial，再按 subject 汇总，未把自相关时间点当独立样本。
完整 posterior 边界/网格、参数混合加密、去边界敏感性、整模态缺失及配对方差变化保留在 posterior/case/summary JSON。固定模型仅作基线，不因 measured 某个分数较好而跳过其 synthetic 敏感性失败。

可进入全面 UQ 的核心 teacher：**无**。

subjects 19–23 没有作为新确认样本使用；subjects 24–29 未读取。没有 tokenizer 训练、外部发布或 protected evaluation。
