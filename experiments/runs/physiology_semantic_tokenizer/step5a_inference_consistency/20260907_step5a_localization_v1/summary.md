# Step5A_inference_consistency — synthetic localization

这是 Step5A0 小样本定位实验，不是 60 次 SBC，也不授予参数解释或 teacher 资格。
matched 是显式离散 RK4 + Gaussian innovation 模型；stress 原样调用 Step 4 生成函数，缩短记录并保留脉冲/确定性血流失配。
oracle 先验定义在 g/w/物理 zeta 上；网格使用梯形积分。EKF 曲线仅称 predictive_score，参数分布为 generalized posterior。
状态区间为条件于参数的 Gaussian moment 近似；clean 区间不加入观测噪声。coverage 先按独立 replicate 汇总，bootstrap 仅作描述。

| Check | Result |
|---|---|
| parameterization | {"transition_max_abs": 0.0, "state_jacobian_max_abs": 9.829914660031136e-12, "parameter_chain_rule_max_abs": 1.4952980920932306e-12, "observation_max_abs": 0.0, "passed": true} |
| linear_gaussian | {"mean_max_abs": 3.00454106039183e-14, "covariance_max_abs": 3.7383290907300193e-16, "exact_joint_log_likelihood": 105.10030328358295, "predictive_score": 104.90737923864314, "score_minus_joint": -0.19292404493980086, "passed": true} |

| Axis / law | Bias | Parameter 95% coverage | Max grid CDF Δ |
|---|---:|---:|---:|
| G / oracle_r_known | 0.00571 | 1.000 | 0.04463 |
| G / matched_model_calibration | 0.10002 | 1.000 | 0.00838 |
| G / misspecification_stress_test | 0.00370 | 0.750 | 0.00816 |
| W / oracle_r_known | 0.02214 | 0.750 | 0.03340 |
| W / matched_model_calibration | 0.02091 | 1.000 | 0.00663 |
| W / misspecification_stress_test | 0.07585 | 0.500 | 0.01242 |
| Z / oracle_r_known | 0.00306 | 1.000 | 0.05216 |
| Z / matched_model_calibration | -0.10834 | 0.750 | 0.00394 |
| Z / misspecification_stress_test | -0.08689 | 1.000 | 0.01491 |

| Axis | Reference precision | Max logL SE | Budget / split / grid CDF Δ | EKF–PF CDF Δ |
|---|---|---:|---|---:|
| G | RESOLVED_SHORT_CASE | 0.1093 | 0.0046 / 0.0064 / 0.0120 | 0.0842 |
| W | INCONCLUSIVE_REFERENCE_PRECISION | 0.3655 | 0.0070 / 0.0063 / 0.0148 | 0.0560 |
| Z | RESOLVED_SHORT_CASE | 0.0950 | 0.0167 / 0.0149 / 0.0048 | 0.0972 |

| Axis / law / estimator | r coverage | EEG coverage | HbO coverage | HbR coverage |
|---|---:|---:|---:|---:|
| G / matched_model_calibration / U0_FIXED | 0.910 | 0.910 | 0.879 | 0.930 |
| G / matched_model_calibration / true_parameters | 0.906 | 0.906 | 0.906 | 0.945 |
| G / matched_model_calibration / score_mode | 0.895 | 0.895 | 0.898 | 0.941 |
| G / misspecification_stress_test / U0_FIXED | 0.922 | 0.922 | 0.875 | 0.922 |
| G / misspecification_stress_test / true_parameters | 0.930 | 0.930 | 0.883 | 0.922 |
| G / misspecification_stress_test / score_mode | 0.922 | 0.922 | 0.867 | 0.926 |
| W / matched_model_calibration / U0_FIXED | 0.945 | 0.945 | 0.957 | 0.891 |
| W / matched_model_calibration / true_parameters | 0.957 | 0.957 | 0.969 | 0.879 |
| W / matched_model_calibration / score_mode | 0.941 | 0.941 | 0.977 | 0.871 |
| W / misspecification_stress_test / U0_FIXED | 0.957 | 0.957 | 0.902 | 0.934 |
| W / misspecification_stress_test / true_parameters | 0.949 | 0.949 | 0.918 | 0.949 |
| W / misspecification_stress_test / score_mode | 0.926 | 0.926 | 0.930 | 0.965 |
| Z / matched_model_calibration / U0_FIXED | 0.949 | 0.949 | 0.957 | 0.941 |
| Z / matched_model_calibration / true_parameters | 0.965 | 0.965 | 0.961 | 0.953 |
| Z / matched_model_calibration / score_mode | 0.953 | 0.953 | 0.957 | 0.969 |
| Z / misspecification_stress_test / U0_FIXED | 0.914 | 0.914 | 0.918 | 0.973 |
| Z / misspecification_stress_test / true_parameters | 0.938 | 0.938 | 0.957 | 0.996 |
| Z / misspecification_stress_test / score_mode | 0.922 | 0.922 | 0.961 | 1.000 |

完整误差、遮挡覆盖、边界质量、区间宽度、rank 和参考曲线见 summary.json 与 case_*.json；原始合成数组见 case_*.npz。
reference 未通过精度检查的案例只能记为未确定；通过也只支持相应短案例，不外推到长序列或完整 SBC。
未运行 measured/protected、tokenizer、60-repeat calibration、Step5A1/5B 或 U3 联合参数拟合。
