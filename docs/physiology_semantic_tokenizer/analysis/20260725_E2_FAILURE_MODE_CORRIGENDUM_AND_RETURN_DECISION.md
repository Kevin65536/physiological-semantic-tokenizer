# E2 失败模式口径更正与架构回归决定

_日期：2026-07-25；作用：补充而不改写 2026-07-24 E2 冻结报告_

---

## 📋 决定

E2 的 development decision `no_semantic_row_admitted_retain_T0` 保持不变：在 E2 的准确训练合同下，T1/T2 teacher 入口没有改善预注册 hard-token semantic endpoint。

但 E2 不能继续被表述为“已经充分检验 physical teacher 或 shared-driver tokenization”。本更正区分三个被历史报告混用的支持总体，并记录 E2 之后 validity policy 的变化。

## 🔢 三个不同分母

| 总体 | Validation 数量 | 含义 |
| --- | ---: | --- |
| E2 loader 总体 | 300 windows / 3,000 patches | 五个 validation subjects，每人 60 个 MA namespace windows |
| E0 teacher sidecar | 50 windows / 500 patches | 仅 `session_01/MA`，每人 10 trials；占总 patch `16.67%` |
| 历史 EEG frozen probe | 178 / 500 teacher patches | 旧 artifact invalidity policy 下的 probe support |

因此，EEG validation target patch 少首先是 **teacher sidecar 只覆盖 E2 validation 总体的六分之一**；`178/500` 又是该 teacher-covered 子集在旧 signal mask 下的 frozen-probe 支持。

## ⚠️ 训练 loss 与 frozen probe 的 mask 不一致

E2 的 state/prototype teacher loss 使用 entry target mask。target-present sample 的这些 mask 为 true，loss 没有再与 signal/token-valid mask 相交。因此：

- frozen EEG probe 使用了 178 个旧 policy 下有效的 target patches；
- teacher semantic loss 实际使用了 500 个 target patches；
- 历史综合报告把 `178/500` 描述为 semantic supervision 的有效支持，是口径错误；
- 这属于训练/评估支持总体不一致，不能称为“无软件/契约问题”。

原始 runs、metrics 和 `no_semantic_row_admitted` 决定不变，但所有后续引用必须附带这一限制。

## 🧼 E2 后的数据 policy 变化

commit `6d6c648` 撤销 EEG artifact mask 的 invalidity authority。当前合同中：

- `artifact_mask` 恒为 false annotation；
- `analysis_valid_mask == valid_mask`；
- token validity 只由 boundary/finite measurement contract 决定；
- artifact diagnostics 只用于 QC 分层和敏感性分析。

因此旧 E2 与新 R 系列使用不同 validity policy。旧 T0 不能作为 R3 的匹配 baseline；新计划必须在当前 policy 下重跑 `N0`，且 manifest 记录 `artifact_mask_policy`。

## 🔍 E2 检验与未检验的命题

E2 检验了：

- raw reconstruction 为主要目标；
- teacher 权重降至 `0.005` 的弱辅助入口；
- EEG `r_mean/r_slope` 与可选 `s` 摘要；
- fNIRS HbO/HbR 自身状态摘要；
- local/prototype 多入口；
- patch-local encoder；
- 固定 K128 量化器。

E2 没有检验：

- 完整 joint driver \(r^J\) 轨迹作为唯一主要 semantic target；
- 量化前 modality-only full-window temporal encoder；
- 两侧共享 driver decoder；
- 100% target-covered cohort；
- joint \(r^J\) 相对 EEG-only、shuffled-joint 和 smooth pseudo-target 的特异性；
- 冻结双向 token 的离线时延条件关联；
- 满足严格 receptive-field cutoff 后，对窗外未来原始 fNIRS predictive residual 的增量信息。

所以 E2 的合适结论是：

> 弱权重、摘要化、多入口的 teacher supervision 不足以改变 E2 的 hard-token semantic geometry；该结果推动改变 estimand、receptive field、coverage 和主损失，而不是缩小 K 或放弃 shared-driver 假设。

## 🎯 回归决定

新的 SD-SVQ 计划：

1. tokenizer 输入只含本模态 raw physiology 与 valid mask；
2. 两侧独立 `K=128,D=64`；
3. 量化前使用 modality-only full-window encoder；
4. 以完整 \(r^J\) 轨迹为 primary semantic target；
5. 删除首轮 discrete nuisance/effect token、多入口 loss 和 coupling shaper；
6. 在 VQ 前先通过双侧 continuous observability；
7. token 冻结后先用 teacher-independent raw fNIRS endpoint 检验离线关联；只有严格 cutoff 模式才检验窗外未来预测，独立确认留给 protected R7。

该计划是新实验代际 R0–R7，不是 E2 权重补跑。

## 🔗 相关记录

- [E2 综合报告](20260724_E2_COMPREHENSIVE_REVIEW.md)
- [artifact validity policy](../../project_changelog/2026-07-25_disable_eeg_artifact_mask_authority.md)
- [目标架构](../02_TARGET_ARCHITECTURE.md)
- [R 系列实验设计](../05_EXPERIMENT_DESIGN.md)
