# Step5A1 阶段报告：最小状态 teacher 资格

训练参数分布与留出 trial 的随机过程相互独立；正确配对、自身历史、任务模板和配对/时间位移 null 使用相同冻结训练参数分布。
matched 用于同模型状态校准；stress 使用 Step 4 脉冲与确定性血流生成合同，其 coverage 为失配诊断，不称 SBC。
U0 使用固定参数并接受 G/W/Z 的预设先验分位数敏感性检查；U1 使用训练后参数分布传播。U3 仅诊断，不进入候选选择。

注册案例 240，成功 231，失败 9。失败种子与异常完整保留；以下数值仅描述成功案例，不能以成功子集授予资格。

| 候选 | 合格 | 未通过检查 |
|---|---|---|
| U0_FIXED | False | complete_registered_cases, fixed_parameter_sensitivity |
| U1_G | False | complete_registered_cases |
| U1_W | True | 无 |
| U1_Z | False | complete_registered_cases |

| 候选 | matched r NRMSE | r corr | r 95%覆盖 | EEG | HbO | HbR |
|---|---:|---:|---:|---:|---:|---:|
| U0_FIXED | 0.4547 | 0.8893 | 0.9477 | 0.9477 | 0.9495 | 0.9566 |
| U1_G | 0.4607 | 0.8872 | 0.9480 | 0.9480 | 0.9358 | 0.9369 |
| U1_W | 0.4640 | 0.8835 | 0.9461 | 0.9461 | 0.9501 | 0.9493 |
| U1_Z | 0.4599 | 0.8865 | 0.9515 | 0.9515 | 0.9490 | 0.9475 |

| 候选 | 遮挡目标 | 对照 | 配对 log-score 增量 | replicate bootstrap 95% CI |
|---|---|---|---:|---|
| U0_FIXED | center_EEG | own_history | 0.176770 | [0.129603, 0.226272] |
| U0_FIXED | center_EEG | own_history_and_task | 0.220795 | [0.171805, 0.270760] |
| U0_FIXED | center_EEG | independent_pairing | 0.638932 | [0.497142, 0.790342] |
| U0_FIXED | center_EEG | circular_shift | 0.570114 | [0.465146, 0.679247] |
| U0_FIXED | center_fNIRS | own_history | 0.057533 | [0.032591, 0.084551] |
| U0_FIXED | center_fNIRS | own_history_and_task | 0.149337 | [0.113035, 0.186876] |
| U0_FIXED | center_fNIRS | independent_pairing | 0.279998 | [0.177331, 0.411795] |
| U0_FIXED | center_fNIRS | circular_shift | 0.275971 | [0.171768, 0.392928] |
| U1_G | center_EEG | own_history | 0.124293 | [0.082077, 0.168854] |
| U1_G | center_EEG | own_history_and_task | 0.197756 | [0.145195, 0.252518] |
| U1_G | center_EEG | independent_pairing | 0.630531 | [0.476060, 0.799340] |
| U1_G | center_EEG | circular_shift | 0.453596 | [0.330796, 0.587612] |
| U1_G | center_fNIRS | own_history | 0.030875 | [0.003867, 0.057326] |
| U1_G | center_fNIRS | own_history_and_task | 0.096156 | [0.060729, 0.128106] |
| U1_G | center_fNIRS | independent_pairing | 0.238074 | [0.148468, 0.338396] |
| U1_G | center_fNIRS | circular_shift | 0.165113 | [0.107127, 0.226115] |
| U1_W | center_EEG | own_history | 0.171386 | [0.108625, 0.249444] |
| U1_W | center_EEG | own_history_and_task | 0.241827 | [0.171080, 0.317971] |
| U1_W | center_EEG | independent_pairing | 0.663470 | [0.475360, 0.868095] |
| U1_W | center_EEG | circular_shift | 0.506086 | [0.386007, 0.637901] |
| U1_W | center_fNIRS | own_history | 0.036551 | [0.012937, 0.060926] |
| U1_W | center_fNIRS | own_history_and_task | 0.118991 | [0.087715, 0.152643] |
| U1_W | center_fNIRS | independent_pairing | 0.270356 | [0.183694, 0.370160] |
| U1_W | center_fNIRS | circular_shift | 0.220034 | [0.148201, 0.301382] |
| U1_Z | center_EEG | own_history | 0.099141 | [0.049350, 0.154710] |
| U1_Z | center_EEG | own_history_and_task | 0.206986 | [0.138354, 0.283572] |
| U1_Z | center_EEG | independent_pairing | 0.649712 | [0.474920, 0.846082] |
| U1_Z | center_EEG | circular_shift | 0.343046 | [0.244884, 0.448636] |
| U1_Z | center_fNIRS | own_history | 0.042722 | [0.016564, 0.068684] |
| U1_Z | center_fNIRS | own_history_and_task | 0.118274 | [0.083070, 0.154294] |
| U1_Z | center_fNIRS | independent_pairing | 0.210687 | [0.128597, 0.310925] |
| U1_Z | center_fNIRS | circular_shift | 0.301593 | [0.201055, 0.414704] |

可进入后续最小 measured 检验的候选：**U1_W**。
所有区间先按独立 replicate 汇总；均值、方差分解、whole-modality 诊断、stress 指标和 U3 网格/相关性/边界质量详见 summary.json 与 case/u3 原始结果。
参数恢复、状态恢复及不确定性风险排序是不同结论。上述状态资格不授予个体生理参数解释，也不自动验证精度加权。
若无候选合格，Step5B 与全面 UQ 按冻结顺序记为未执行，不能用本阶段完成状态代替 scientific pass。

完整性按候选自身所需的注册面板核对；U1_W 的 60 个 W 案例全部完成。G/Z 的失败保留，不将它们错误地传播为 W 的失败。原聚合快照保留，所有指标、阈值和随机种子不变。
