# 生理信号 tokenizer 的原文、官方实现与发布权重核验

核验日期为 2026-11-23。本文件是本轮离散基线探索的文献证据快照。项目实现、数据合同和实验结果继续由各自 owning source 管理。

三项参照的名义码率差异很大。LaBraM 每通道每秒提供 1 个 K8192 ID，BioCodec 每约 48 ms 提供 6 个 K256 ID，NeuroRVQ EEG 每秒提供 32 个 K8192 ID。后两项的波形质量对应高得多的整数预算。NeuroRVQ 官方波形恢复还使用输入窗口的频谱统计量，比较原单位重建时需要单独审查这条路径。

## 来源及版本

| 工作 | 读取的原始论文 | 固定的官方代码版本 | 本次核验材料 |
| --- | --- | --- | --- |
| LaBraM | ICLR 2024 proceedings 版本，§2.2、§3.1、附录 C 表 3/4 | `935963004/LaBraM@c431221e6cfd23dbfa9950e0180682fb322b0548`，与核验时远端 HEAD 一致 | README、tokenizer runner、model、EMA quantizer、engine，以及仓库发布的 `checkpoints/vqnsp.pth` |
| BioCodec | `arXiv:2510.09095v1`，2025-10-10，§2/3、附录 B/G | `klean2050/BioCodec-release@575bf6ea7ae14a34754b217d9b940597a16951bd`，与核验时远端 HEAD 一致 | EEG YAML、model、SEANet、RVQ、loss、trainer、loader，以及仓库发布的 `biocodec_ckpt.pt` |
| NeuroRVQ | `arXiv:2510.13068v5`，2026-11-07，§3–5、附录 C/J | `KonstantinosBarmpas/NeuroRVQ@d2061c7b1b29cf2616af0be38fb0a842b00a54cf`，核验时远端 HEAD | EEG YAML、model、RVQ、配置构建、preprocessing 和 tokenizer inference example |

原始入口分别见 [LaBraM 论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)、[LaBraM 官方代码](https://github.com/935963004/LaBraM/tree/c431221e6cfd23dbfa9950e0180682fb322b0548)、[BioCodec 论文](https://arxiv.org/html/2510.09095v1)、[BioCodec 官方发布](https://github.com/klean2050/BioCodec-release/tree/575bf6ea7ae14a34754b217d9b940597a16951bd)、[NeuroRVQ 论文 v5](https://arxiv.org/pdf/2510.13068v5) 和 [NeuroRVQ 官方代码](https://github.com/KonstantinosBarmpas/NeuroRVQ/tree/d2061c7b1b29cf2616af0be38fb0a842b00a54cf)。

BioCodec 论文中的 `usc-sail/BioCodec` 在本次访问中返回 404。作者主页和作者参与的 BrainCodes 发布均指向上述 `klean2050/BioCodec-release`，该仓库 README 自述为论文官方实现。本地 `reference_repository/NeuroRVQ` 仍是 `14bd0ce6a4cea627ab3cf1e7862fb31a5172d691`，故以下最新实现核验通过固定远端源码完成。本地 reference repositories 的工作树另有改动，LaBraM 代码核验使用提交内容与固定远端源码。来源见 [作者主页](https://klean2050.github.io/)、[BrainCodes 官方 README](https://github.com/microsoft/BrainCodes) 和 [BioCodec 发布 README](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/README.md)。

## 结构与整数预算

以下码率均按每个原始通道计算，为固定宽度 ID 的理论净荷。文件整数 dtype、封装开销和熵编码另行计算。

| 核验项目 | LaBraM tokenizer | BioCodec EEG tokenizer | NeuroRVQ EEG tokenizer |
| --- | --- | --- | --- |
| 输入与上下文 | 200 Hz EEG，1 s/200 点 patch，上限约 256 个通道时间 patch。典型 64 通道 4 s 或 32 通道 8 s | 250 Hz 单通道，训练片长 5 s/1250 点，卷积及 LSTM 支持可变长度 | 200 Hz EEG，1 s/200 点 patch，默认最多 256 个通道时间 patch，示例据通道数截取 context |
| 主干宽度/深度 | Transformer hidden 200，encoder 12 层、decoder 3 层、10 heads、MLP 800 | SEANet base filters 32，卷积宽度到 256，2 层 LSTM，输出 latent 128，镜像 decoder | 4 个多尺度卷积分支，各分支经共享 12 层 encoder/3 层 decoder，hidden 200、10 heads、MLP ratio 4 |
| 量化前维度 | encoder 200，经 `200→200→64` 投影进入实际量化空间 | encoder latent 128。每个 RVQ 层各有 `128→16` 与 `16→128` 投影，残差在 128 维中递推 | 每分支 encoder 200，经 `200→200→128` 投影进入量化空间 |
| 码本向量维度 | 64，runner/class 的裸默认值另为 32 | 16，外层 `dimension=128` 与实际码向量维度分别定义 | 128 |
| K 和量化级数 | K8192，单级归一化 EMA VQ，k-means 初始化 | 6 级 RVQ，每级 K256，EMA、k-means 初始化 | 4 个分支各 8 级 RVQ，每级 K8192，归一化 EMA VQ、k-means 初始化 |
| 每时间位置的 ID 数 | 1 ID/通道/s | 6 ID/通道/12 点，即约 48 ms | 32 ID/通道/s |
| 名义码率 | `log2(8192)/1 = 13 bit/s` | `(250/12)×6×8 = 1000 bit/s`。5 s 有 105 个 latent 位置时为 `105×6×8/5 = 1008 bit/s` | `4×8×13/1 = 416 bit/s` |

结构和默认值逐项核对 [LaBraM model](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/modeling_vqnsp.py#L178)、[LaBraM runner](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/run_vqnsp_training.py#L33)、[BioCodec model](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/model.py#L150)、[SEANet](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/modules/seanet.py#L120)、[BioCodec quantizer](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/quantization/quantizer.py#L45)、[NeuroRVQ EEG 配置](https://github.com/KonstantinosBarmpas/NeuroRVQ/blob/d2061c7b1b29cf2616af0be38fb0a842b00a54cf/flags/NeuroRVQ_EEG_v1.yml) 和 [NeuroRVQ EEG model](https://github.com/KonstantinosBarmpas/NeuroRVQ/blob/d2061c7b1b29cf2616af0be38fb0a842b00a54cf/NeuroRVQ_EEG/NeuroRVQ.py#L390)。

本项目每分支每位置每 2 s 一个 ID 时，K128、K1024、K8192 分别为 3.5、5、6.5 bit/s。若某种 RVQ 诊断把多个 ID 打包成一个整数，仍须按组合词表和真实总比特计数。这里的码向量 16/64/128 维均独立于 ID 的 `log2(K)` 比特数。

## 训练目标、尺度和解码条件

| 工作 | 原文/官方实现中的重建目标与尺度 |
| --- | --- |
| LaBraM | 原始 EEG 以 0.1 mV 为固定单位，官方 engine 对 µV 输入除以 100。每个样本的 FFT 幅度和相位分别在全部通道、patch、频点上 z-score，decoder 预测这两个标准化频谱，二者 MSE 加 VQ commitment。原文的主要用途是为后续 masked code prediction 提供监督。若需恢复原单位电压，需明确额外幅度/相位统计量的来源。 |
| BioCodec | 原文输入按 session 去 DC、去线性趋势、z-score，250 Hz 单通道训练。论文列出 waveform L1 与多尺度频谱项，权重 0.2 和 1.0、commitment 0.25。公开非 AMP trainer 实际使用 Huber 时域项（beta 0.7），频谱为 32/64/128/256 点 STFT 上 log-magnitude 与 sin/cos phase 的 L1+MSE。loader 将已标准化样本裁到 ±10。EEG YAML `normalize=False`，可避免 model 内部逐片 RMS side information。`_get_optimized_model()` 裸默认为 True，此路径 encode 会返回连续 scale 并由 decode 乘回，调用者必须显式审查。 |
| NeuroRVQ | 预测每个 1 s patch 的标准化 log-FFT 幅度与 sin/cos phase，使用 log-amplitude MSE、圆周 phase cosine 项及单位圆 penalty 0.1、时域项、RVQ commitment。官方 RVQ 另加入权重 0.4 的局部辅助 MSE。示例预处理为约 0.5–44.5 Hz 三阶 Butterworth、适用 notch、±500 裁剪、200 Hz。官方 `forward` 从输入计算 `amp_mean/amp_std`，用它们逆标准化预测 log-amplitude，随后对输入和重建分别 z-score，再返回波形。该输出接口评价的是逐窗标准化形态，整数恢复原幅度需要另行实现与复验。 |

代码证据见 [LaBraM engine 的固定缩放](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/engine_for_vqnsp.py#L58)、[LaBraM target normalization](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/modeling_vqnsp.py#L143)、[BioCodec objective](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/objective.py)、[BioCodec trainer](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/trainer_eeg.py#L60)、[BioCodec scale 路径](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/model.py#L82)、[NeuroRVQ waveform 路径](https://github.com/KonstantinosBarmpas/NeuroRVQ/blob/d2061c7b1b29cf2616af0be38fb0a842b00a54cf/NeuroRVQ_EEG/NeuroRVQ.py#L573) 和 [NeuroRVQ preprocessing](https://github.com/KonstantinosBarmpas/NeuroRVQ/blob/d2061c7b1b29cf2616af0be38fb0a842b00a54cf/preprocessing/preprocessing_eeg_example.py)。

## 论文值、代码默认值、实际启动证据

| 工作/阶段 | 论文使用值 | 固定代码与命令解析值 | 实际运行证据 |
| --- | --- | --- | --- |
| LaBraM tokenizer | AdamW，beta=(0.9,0.99)，WD 1e-4，global batch 1024，peak LR 5e-5/min 1e-5，cosine，100 epochs，warmup 10，data stride 200 点。论文使用 8×A800 | parser batch 64、code dim 32、warmup 5。README 命令覆盖为每卡 batch128、code dim64、warmup10、8 workers。命令省略 LR，parser 5e-5，脚本再乘 `global_batch/128`，故该命令峰值为 4e-4 | 官方 `vqnsp.pth` 的 args 为 batch128、world_size8、LR4e-4、code dim64、100 epochs。保留 epoch99，optimizer step880970。WD decay group1e-4，beta=(0.9,0.99)。与 README 解析一致，和论文表的 peak LR 有差异 |
| BioCodec EEG tokenizer | Adam，beta=(0.5,0.9)，WD1e-2，batch512，LR1e-4，cosine，20%warmup，至多20 epochs，4×A100。附录 B 选择 checkpoint 为46K steps，验证集100万单通道样本 | EEG YAML batch512、max_epoch20、LR1e-4、warmup5 epochs、seed79911092、AMP False。Adam 构造未传 WD，默认0。warmup 为25%。trainer 先对含 commitment 的总 loss backward，再对 commitment 单独 backward，梯度累计需按代码核对 | 发布权重 `epoch=2`，各 optimizer state 的 step19530，initial_lr1e-4、beta=(0.5,0.9)、WD0。scheduler `max_iter=195300`、`warmup_iter=48825`、`last_epoch=19530`。该发布权重的预算与论文46K checkpoint描述存在差异 |
| NeuroRVQ EEG tokenizer | AdamW，beta=(0.9,0.999)，WD1e-4，batch256，LR5e-5/min1e-5，cosine，100 epochs，warmup10，clip3，layer scale0.001，4×V100 | YAML 明确结构与下游 fine-tuning 配置。当前官方仓库提供推理路径，tokenizer 训练 runner 未公开。构造8级/分支，使用固定 code_dim128 | 本次读取的发布面无 tokenizer 训练日志和完整 launch record。表 C 的训练预算属于论文报告值，推理 YAML 的 `*_finetuning` 字段属于后续阶段 |

LaBraM 的命令、默认值与学习率解析分别见 [README tokenizer 命令](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/README.md#2-train-the-neural-tokenizer)、[runner defaults](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/run_vqnsp_training.py#L33) 和 [LR 线性缩放](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/run_vqnsp_training.py#L248)。BioCodec 的默认值与 optimizer 见 [EEG YAML](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/configs/eeg_config.yaml)、[trainer optimizer/scheduler](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec/trainer_eeg.py#L272) 和 [发布权重](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec_ckpt.pt)。NeuroRVQ 训练值见 [原文 v5 附录 C](https://arxiv.org/pdf/2510.13068v5)。

LaBraM 发布 checkpoint 的本地文件 Git blob 为 `4ba4be5cc403143f8370bf9c57b6b10b4c2b3727`，与官方固定提交中的 blob 一致，SHA256 为 `d8f14b4232f06a5c37b6386ab021c270dd4bfa12bfecdd60372dc8ac7b711101`。BioCodec 发布 checkpoint SHA256 为 `b8b5c864bd92242a2aa70c414ab1abe23231772dc2f2e39a8f509fdded728af1`。二者使用 `torch.load(..., weights_only=True)` 读取。LaBraM 仅为 argparse Namespace、NumPy scalar/dtype 加入精确安全类型映射。上述摘要保留关键元数据以便复核。LaBraM args 中仍有图像/VQKD遗留字段，本文只采用与实际模型 shape、optimizer 和运行预算相互核对的字段。权重来源见 [LaBraM 发布 tokenizer](https://github.com/935963004/LaBraM/blob/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints/vqnsp.pth) 和 [BioCodec 发布 tokenizer](https://github.com/klean2050/BioCodec-release/blob/575bf6ea7ae14a34754b217d9b940597a16951bd/biocodec_ckpt.pt)。

## 后续预训练的独立预算

LaBraM 的 masked EEG pretraining 使用 frozen tokenizer targets，论文表4为 batch512、AdamW beta=(0.9,0.98)、WD0.05、LR5e-4、50 epochs、warmup5、mask0.5。Base/Large/Huge hidden 分别200/400/800，12/24/48层。这些宽度和训练预算属于下游 foundation model。BioCodec 原文冻结 codec，再以 RVQ codes 训练时间与通道 Transformer 和任务头，各任务调 LR/batch并训练20epochs。NeuroRVQ-FM 原文附录J为 batch64、LR5e-4、50epochs、warmup5、WD0.05、mask0.5，区别于 tokenizer 的 batch256、LR5e-5、100epochs。来源见 [LaBraM 原文表4](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)、[BioCodec 原文 §2.3/3.2](https://arxiv.org/html/2510.09095v1#S2.SS3)、[NeuroRVQ v5 附录J](https://arxiv.org/pdf/2510.13068v5)。

## 对本轮探索范围的依据

以下是根据三项参照和本项目接口作出的实验设计推断。

- K128 到 K1024/K8192 的单 ID 扩容有文献数量级依据，仍须测量训练样本是否足以支持码占有与跨被试解释。K8192 每2秒仅6.5bit/s，容量仍远低于上述 RVQ 波形系统。
- hidden width、量化输入维度、码本向量维度应分别配置与记录。可以探索 hidden128–256、量化向量32–128，在匹配连续上界中先定位编码与量化误差，再决定是否扩大网络。
- 波形主损失与频谱/局部动态损失可组合。频谱项的设计应由当前1–45Hz输入支持、分支失真和定量校准决定。原文归一化频谱结果为这种诊断提供依据，项目最终评分需要原信号坐标。
- BioCodec 与 NeuroRVQ 的多级/高码率设置可以提供容量诊断。最终单ID接口需独立满足本轮四分支、原幅度、波形动态与局部单码信息标准。
- 训练预算按 seen examples、optimizer steps、global batch和资源时间共同比较。LaBraM 发布tokenizer接近88万步，BioCodec论文46K与发布19.53K分别保留身份。单用10000步或epoch数字无法建立匹配预算。

这些参照支持扩大容量和诊断预算，同时要求单独验证原幅度解码与整数重放。项目验收阈值由本地匹配参考和固定失真校准登记。
