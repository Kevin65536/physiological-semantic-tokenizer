# 独立观测信息与非线性双向反事实筛查

本轮 A 只读取上一轮身份与当前缓存元数据；B 使用非线性六状态核心的受迫确定性子模型。alpha/E0/kappa 固定，拟合 tau、eta、未知频带内 r 系数及全部五个血流初态。这是有利条件下的恢复筛查，不是完整随机 SSM 或 teacher 资格。

完整分母：168 个面板、1008 次拟合（含校准敏感性），997 次收敛。

| condition | method | total | completed | valid_intervals | combination_relative_error_median | driver_nrmse_median | tau_95_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dpf_mixing | fixed | 24 | 24 | 13 | 0.442 | 0.3327 | 0 |
| dpf_mixing | independent | 24 | 24 | 21 | 0.2271 | 0.1744 | 0.8571 |
| dpf_mixing | independent_minus | 24 | 24 | 21 | 0.2308 | 0.1693 | 0.8571 |
| dpf_mixing | independent_plus | 24 | 24 | 21 | 0.2276 | 0.1768 | 0.8571 |
| dpf_mixing | joint_target | 24 | 24 | 0 | 0.6281 | 0.6737 | None |
| dpf_mixing | oracle | 24 | 24 | 19 | 0.2216 | 0.1842 | 0.8947 |
| dpf_mixing_nonrest | fixed | 24 | 24 | 14 | 0.4675 | 0.3388 | 0.1429 |
| dpf_mixing_nonrest | independent | 24 | 24 | 24 | 0.2073 | 0.1634 | 0.7917 |
| dpf_mixing_nonrest | independent_minus | 24 | 24 | 24 | 0.2107 | 0.1637 | 0.7917 |
| dpf_mixing_nonrest | independent_plus | 24 | 24 | 24 | 0.2064 | 0.1631 | 0.7917 |
| dpf_mixing_nonrest | joint_target | 24 | 22 | 0 | 0.6688 | 0.7404 | None |
| dpf_mixing_nonrest | oracle | 24 | 24 | 23 | 0.2262 | 0.175 | 0.8261 |
| gain2_independent_slow | fixed | 24 | 24 | 4 | 0.5665 | 0.3132 | 0 |
| gain2_independent_slow | independent | 24 | 24 | 24 | 0.1143 | 0.1332 | 0.8333 |
| gain2_independent_slow | independent_minus | 24 | 24 | 24 | 0.1149 | 0.1343 | 0.8333 |
| gain2_independent_slow | independent_plus | 24 | 24 | 24 | 0.1137 | 0.1334 | 0.8333 |
| gain2_independent_slow | joint_target | 24 | 23 | 0 | 0.6611 | 0.7085 | None |
| gain2_independent_slow | oracle | 24 | 24 | 24 | 0.1337 | 0.1344 | 0.875 |
| gain2_white | fixed | 24 | 24 | 0 | 0.6486 | 0.1722 | None |
| gain2_white | independent | 24 | 24 | 24 | 0.01761 | 0.02801 | 1 |
| gain2_white | independent_minus | 24 | 24 | 24 | 0.01911 | 0.02833 | 1 |
| gain2_white | independent_plus | 24 | 24 | 24 | 0.01726 | 0.02718 | 1 |
| gain2_white | joint_target | 24 | 22 | 0 | 0.2635 | 0.5796 | None |
| gain2_white | oracle | 24 | 24 | 24 | 0.01778 | 0.02342 | 1 |
| independent_slow | fixed | 24 | 24 | 11 | 0.4336 | 0.3223 | 0.2727 |
| independent_slow | independent | 24 | 24 | 22 | 0.192 | 0.1722 | 0.9091 |
| independent_slow | independent_minus | 24 | 24 | 22 | 0.1867 | 0.1713 | 0.9545 |
| independent_slow | independent_plus | 24 | 24 | 22 | 0.1895 | 0.1744 | 0.9091 |
| independent_slow | joint_target | 24 | 24 | 0 | 0.6169 | 0.7019 | None |
| independent_slow | oracle | 24 | 24 | 22 | 0.1981 | 0.196 | 0.8636 |
| shared_slow | fixed | 24 | 24 | 12 | 0.5595 | 0.5046 | 0 |
| shared_slow | independent | 24 | 24 | 22 | 0.1694 | 0.1055 | 0.9091 |
| shared_slow | independent_minus | 24 | 24 | 23 | 0.1615 | 0.1033 | 0.913 |
| shared_slow | independent_plus | 24 | 24 | 22 | 0.1753 | 0.1031 | 0.9091 |
| shared_slow | joint_target | 24 | 23 | 0 | 0.6344 | 0.7078 | None |
| shared_slow | oracle | 24 | 24 | 23 | 0.1673 | 0.1113 | 0.9565 |
| white | fixed | 24 | 24 | 24 | 0.03135 | 0.02755 | 1 |
| white | independent | 24 | 24 | 24 | 0.03339 | 0.03338 | 1 |
| white | independent_minus | 24 | 24 | 24 | 0.03401 | 0.03495 | 1 |
| white | independent_plus | 24 | 24 | 24 | 0.03634 | 0.03401 | 1 |
| white | joint_target | 24 | 19 | 0 | 0.4419 | 0.5727 | None |
| white | oracle | 24 | 24 | 24 | 0.03135 | 0.02755 | 1 |

组合误差为 lambda/a/k 的相对 RMS；r NRMSE 不做逐条符号、幅度或时移对齐。局部 Gaussian 区间在秩亏或触界时记不可用；覆盖率只在有效区间上计算，同时保留完整分母。光学标准和噪声记录与目标独立；慢成分在加性 Hb 层生成，滤波/重采样和基线算子同步传播到均值及协方差。DPF 0.8/1.2 是机制敏感性设定，没有被当成实测 DPF 置信区间。

[逐记录校准表](calibration_information.csv)、[逐次拟合](recovery_metrics.csv)、[真实生理变化](physiology_counterfactual.csv)、[观测变化](observation_counterfactual.csv)。

A 尚未建立可用于本次实测候选的独立标定；Single-Trial ECG/呼吸的发布说明提供后续线索。C、D 按依赖条件保留未启动。未生成 PDF，未检查 WPS。
