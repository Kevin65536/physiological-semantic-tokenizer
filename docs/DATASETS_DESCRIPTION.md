# 数据集说明文档

跨数据集进入模型前的可执行单位、基线、漂移、尺度、mask 与 split 规范见
[`DATA_CONTRACT.md`](DATA_CONTRACT.md)。该合同保留各数据集的原始测量语义，
不把电压、吸光度和浓度伪装成同一物理单位。

### 采集方式与单位证据复核（2026-09-15）

本节记录原始说明文件、原作者论文和当前 loader 源码的来源复核。
随后进行的有界实测统计见
[数据统一化诊断报告](../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/20260915_dataset_scaling_report_v1/REPORT.md)；
下文原有的代表性幅值仍是历史检查记录，不能用于推定单位。
统一化建议与实现差距统一放在
[`DATA_CONTRACT.md`](DATA_CONTRACT.md#跨数据集训练前一致性复核2026-09-15)。

| 数据集 | EEG 采集、参考与文件单位 | fNIRS 采集与测量语义 | 对训练前处理的直接影响 |
| --- | --- | --- | --- |
| Single-Trial | BrainAmp，30 scalp channels，linked-mastoid reference；采集 1000 Hz，项目使用 200 Hz MATLAB 导出。loader 读取 `yUnit`，缺失时默认 `uV`；默认值不是单位证据 | NIRScout，采集 12.5 Hz、发布分析视图 10 Hz；36 对 760/850 nm 通道，已记录 `yUnit=V`。这里的 V 是光电探测电压，与 EEG 电位不是同一测量量 | 从光强比构造 OD，再按明确的 MBLL 假设得到相对 HbO/HbR；保留原 EOG 辅助测量与事件同步信息 |
| REFED | ESI Neuroscan，64 channels，10–10 布局、AFz reference，采集与本地 README 均为 1000 Hz。当前 `_refed_eeg` 硬编码 `native_unit="V"`，已核查原始说明未给出对应 MAT 数值单位，故仍需单位证据 | Shimadzu LABNIRS，780/805/830 nm，51 channels，名义源探距 30 mm，47.62 Hz；六种导出量并存，HbO/HbR 与 Abs 必须分开，浓度导出的物理单位仍未核定 | 选 HbO/HbR 作为当前共同成分；保留 baseline/video 区别和动态标签时间轴；不能用 loader 的 V 常量证明单位已经核实 |
| Visual | Nihon Kohden Neurofax EEG-1100，论文描述 32 electrodes、500 Hz；当前 loader 读取连续 EDF 并按每通道 physical/digital extrema 解码，单位取 EDF header。采集参考方式尚未由本次资料核定 | Hitachi ETG-7100，10 Hz；Oxy/Deoxy CSV，设备头为 695/830 nm，当前文件未明确物理浓度单位；两侧 Probe 具有不同位置 | 当前路径使用连续原始 EDF；发布的去眼跳 MAT epochs 是另一条处理支路。设备标为 raw export 不等于 CSV 存的是光强 |
| Simultaneous | BrainAmp，采集 1000 Hz，TP9 reference、TP10 ground；MATLAB 分析视图 200 Hz、28 scalp channels + HEOG/VEOG。loader 读取 `yUnit`，缺失时默认 `uV` | NIRScout，采集 10.4 Hz、MATLAB 视图 10 Hz，36 channels、30 mm；`oxy/deoxy` 为浓度变化，既有字段检查为 `mmol/L`。原始格式说明另记载 `.wl1/.wl2=760/850 nm` | 当前 MATLAB 分支已完成色团转换，不能重复做 MBLL；原厂格式文档描述的光强文件不等于当前分支已经加载了这些文件 |

证据入口：Single-Trial 的 [原始 HTML](<../data/EEG+NIRS Single-Trial/Open access dataset for simultaneous EEG and NIRS Brain-Computer Interfaces (BCIs).html>)；
REFED [原始 README](../data/REFED-dataset/README.md) 与
[原作者论文 §3.2](https://papers.neurips.cc/paper_files/paper/2025/file/2bf0ed7c35d9d84128e7f7c72ab76402-Paper-Datasets_and_Benchmarks_Track.pdf)；
Visual [原始说明](<../data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/readme.txt>) 与
[原作者论文 §4.3–4.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10964074/)；
Simultaneous [MATLAB 说明](<../data/Simultaneous EEG&NIRS/Dataset description_MATLAB.pdf>)、
[原厂格式说明](<../data/Simultaneous EEG&NIRS/Dataset description_BrainVision and NIRx.pdf>)。
REFED 论文附录概览出现 EEG 200 Hz，与 §3.2 和发布 README 的 1000 Hz 不一致；
本项目当前导入路径按后两者使用 1000 Hz，不能混写采集率与分析率。

#### 单位证据执行复核（2026-09-16）

对上述缺口再次检查原始说明、REFED 原论文全文及
[作者数据读取代码](https://github.com/REFED-dataset/REFED-codes/blob/main/load_REFED.py)、
Visual 原论文 §3.2/§4.3–4.5（[全文 XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10964074/fullTextXML)）。
**REFED EEG、REFED Hb 和 Visual Hb 均仍未找到足以证明发布数组物理单位的声明**。
设备型号、典型幅值、论文称 concentration 或 raw 均不替代数组单位证据；也不把
设备可能使用的浓度×光程单位擅自当成 µM。

2026-11-10 的[原论文、设备手册及发布者代码复核](DATA_CONTRACT.md#visual--refed-source-quality-finding--2026-11-10)
补充了 Hitachi Hb 显示量 `mMmm` 的设备层证据，并确认 Visual 的
`Exception Ch` 是排除通道信息。数据集文件的单位身份保持不变；通道有效性及其对
已有共同幅度尺度的影响见该复核，不以高幅度或拟合优度代替测量有效性。

reader 已去除 REFED 的硬编码 V 声明及 MATLAB EEG 缺失 `yUnit` 时的 µV 默认值。
每条记录保留原单位与证据字段；Visual 保留每通道 EDF 单位及增益解码来源。
有头字段证据的 EEG 可转 µV，有 `yUnit=mmol/L` 证据的发布 Hb 可乘 1000 转 µM；
缺字段或未知单位保持各自相对组。Single-Trial 光电 V 独立于电位 V，近似 MBLL
仍为相对 Hb。这里的“未知”是核查结果，不能记成已完成物理标定。

> **重要提示**：使用任何数据集前，请务必先阅读本文档以及对应数据集目录中的原始说明文件。

---

## 数据集总览

| 数据集名称 | 包含的模态 | 被试数量 | 刺激素材类型 | 任务类型 | 采样率 | 标签类型 | 说明文件 |
|-----------|-----------|---------|-------------|---------|--------|---------|---------|
| EEG+NIRS Single-Trial | EEG (30ch) + fNIRS (72ch: 36 lowWL + 36 highWL) + EOG, ECG, 呼吸 | 29 | 视觉指令 (箭头/数字) | Motor Imagery (左右手), Mental Arithmetic | EEG: 200Hz, fNIRS: 10Hz | Left/Right MI, MA/Baseline | `.html` 文档 |
| REFED-dataset | EEG (64ch) + fNIRS (51ch, 6信号类型) | 32 | 情绪视频 (15个) | 情绪诱发 | EEG: 1000Hz, fNIRS: 47.62Hz | 实时动态 Valence + Arousal | `README.md` |
| Visual Cognitive Motivation | EEG + fNIRS (Oxy/Deoxy CSV 导出, 共享位置) | 16 | 场景图片 (250个) | 视觉认知动机决策 | EEG: 500Hz, fNIRS: 10Hz | RF/RR/FF/FR (记忆动机) | `readme.txt` |
| Simultaneous EEG&NIRS | EEG + fNIRS (MATLAB 导出为 oxy/deoxy) | 26 | 认知任务 | N-back、DSR、word generation | EEG: 200Hz, fNIRS: 10Hz (MATLAB导出) | 认知负荷、Go/No-go、词生成/基线 | PDF 文档 |

---

## fNIRS 原始单位与幅值核对

下表基于原始说明文件和代表性原始文件的直接检查，用于回答两个问题：
1. 当前数据目录里保存的是 HbO/HbR，还是 highWL/lowWL / Abs 波长信号。
2. 不同数据集的绝对幅值是否处于同一量级，能否直接混合解释。

| 数据集 | 当前保存形式 | 原始单位/标注来源 | 代表性幅值（单被试全部原始 fNIRS 文件） | 稳健结论 |
|--------|-------------|------------------|----------------------------------------|---------|
| EEG+NIRS Single-Trial | 72通道，36个空间位置的 lowWL/highWL 成对保存，`wavelengths=[760,850]` | 样例 `cnt.mat` 内 `yUnit='V'`, `signal='NIRS (low wavelength, high wavelength)'` | lowWL: median=0.210, P01-P99=0.009-0.817, max=0.878; highWL: median=0.321, P01-P99=0.014-1.087, max=1.216 | 当前保存的是波长通道/光强样式，不是已转换的 HbO/HbR 浓度 |
| REFED-dataset | 同一张量内混合 6 类信号: HbO, HbR, HbT, Abs780, Abs805, Abs830 | README 明确给出 6 类信号，但未提供统一单位；原始 `.mat` 也未附单一 `yUnit` | HbO/HbR 约在 `[-5, 5]`; HbT 约在 `[-2, 3]`; Abs 信号约在 `[0.6, 4.5]` | 不能假定存在单一原始单位；缓存前必须先显式选择 signal type |
| Visual Cognitive Motivation | 以 Oxy/Deoxy CSV 分文件保存，CSV 头仍保留 695/830nm 设备信息 | `readme.txt` 说明为 oxyhemoglobin / deoxyhemoglobin，同时说明文件是 Hitachi ETG-7100 raw export；CSV 未显式写单位 | Oxy: median=0.029, P01-P99=-26.410-54.135, max=61.670; Deoxy: median=-0.034, P01-P99=-13.624-29.934, max=33.646 | 保存语义已是 Oxy/Deoxy，不是 highWL/lowWL；但单位未显式标注，且绝对幅值与 mmol/L 数据集不在同一量级 |
| Simultaneous EEG&NIRS | MATLAB 文件按 `cnt_{task}.oxy` / `cnt_{task}.deoxy` 保存 | 样例 `cnt_nback.mat` 内 `yUnit='mmol/L'`, `signal='NIRS (oxy, deoxy)'` | oxy: median≈0, P01-P99=-0.0085-0.0098, max=0.0747; deoxy: median≈0, P01-P99=-0.0045-0.0050, max=0.0423 | 当前保存已经是 HbO/HbR 浓度，且单位明确为 mmol/L |

**说明**:
- 幅值统计分别取单个代表被试的全部原始 fNIRS 文件: Single-Trial `subject 01`, REFED `subject 1`, Visual `S01`, Simultaneous `VP001`。
- 这里的幅值只用于判断量级和语义是否一致，不代表所有被试的完整总体分布。
- 单位与幅值联合判断后，可将四个数据集分成三类: `波长/optical-domain 通道`、`混合信号类型`、`已导出为 Oxy/Deoxy 浓度语义`。

## 光强可用性与波段对照（历史 optical-cache 方案）

以下保留旧 optical-cache 方案的来源辨析，不是当前训练输入指令。
当前统一 loader 使用 `homer2_aligned_fnirs` 的 HbO/HbR 分支。
将 HbO/HbR 前向投影为假定波长的光学量只能产生模型派生量，不能恢复原始测量；
缺少消光系数、路径长度、基线和单位时，也不能据此构造物理可比的 optical cache。

以下表格只回答统一 optical measurement space 所需的两个问题：当前仓库里是否已经有直接可用的 optical-domain 通道，以及这些通道的波段是否与 EEG+NIRS Single-Trial 的 `760/850 nm` 基准一致。

| 数据集 | 当前文件里是否有直接可用的 optical-domain 通道 | 当前可见的 optical-domain 形式 | 相对 Single-Trial `760/850 nm` 的差异 | 进入统一 optical cache 前的要求 |
|--------|--------------------------------------------|------------------------------|-------------------------------------|----------------------------------|
| EEG+NIRS Single-Trial | 是 | `lowWL/highWL`, `760/850 nm`, `V` | 基准，不存在差异 | 可直接进入统一 optical cache |
| REFED-dataset | 是 | `Abs780/Abs805/Abs830` 三路 optical-domain 通道 | 波段不同，且是三波段而不是 `760/850` 二波段 | 必须先显式选择/投影到统一的两通道 optical contract |
| Visual Cognitive Motivation | 否 | 当前文件只给 `Oxy/Deoxy`; CSV 头保留 `Wave[nm]=695,830` 设备元数据 | 仪器波段与 `760/850` 不同，但当前导出里没有直接 optical-domain 通道 | 不能直接进入统一 optical cache；若要对齐，需拿到上游 optical export 或做显式前向投影 |
| Simultaneous EEG&NIRS | 否 | 当前 MATLAB 导出只给 `oxy/deoxy`, `mmol/L` | 已检查导出字段里未暴露 optical-domain 波段，无法和 `760/850` 直接比较 | 不能直接进入统一 optical cache；若要对齐，需做显式前向投影 |

---

## 详细描述

### 1. EEG+NIRS Single-Trial (TU Berlin)

**目录**: `data/EEG+NIRS Single-Trial/`

**说明文件**: `Open access dataset for simultaneous EEG and NIRS Brain-Computer Interfaces (BCIs).html`

#### 基本信息
- **来源**: TU Berlin Machine Learning Group
- **被试**: 29人 (健康成人)
- **数据格式**: MATLAB (.mat)

#### 模态详情

| 模态 | 通道数 | 采样率 | 覆盖区域 | 数据格式 |
|------|-------|--------|---------|---------|
| EEG | 30 | 200 Hz (原1000Hz下采样) | 全脑 (10-5系统) | `.mat` |
| fNIRS | 72 (36 lowWL + 36 highWL) | 10 Hz (原12.5Hz下采样) | 前额、运动区、视觉区 | `.mat` |
| EOG | 4 | 1000 Hz | 眼电 | 包含在EEG文件 |
| ECG | 2 | 1000 Hz | 心电 | 包含在EEG文件 |
| 呼吸 | 1 | 1000 Hz | 胸带 | 包含在EEG文件 |

#### 实验范式

**Dataset A - Motor Imagery (运动想象)**
- 任务: 左手/右手握拳想象
- 试次结构: 2s指令 + 10s任务 + 15-17s休息
- 每session: 20次重复 (每类10次)
- 共3个session

**Dataset B - Mental Arithmetic (心算)**
- 任务: 连续减法 vs 休息基线
- 试次结构: 同上
- 共3个session

**Dataset C - Motion Artifacts**
- 用于运动伪迹研究

#### 标签类型
- Motor Imagery: `marker 16 = left`, `marker 32 = right`
- Mental Arithmetic: `marker 16 = MA`, `marker 32 = baseline`

#### fNIRS 通道语义
- 原始 `cnt.mat` 里的 fNIRS 通道按空间位置成对出现，总共 72 通道。
- 样例字段明确给出 `signal = NIRS (low wavelength, high wavelength)`，`yUnit = V`，`wavelengths = [760, 850]`。
- 通道名中的 `highWL` / `lowWL` 应视为波长通道标签，而不是已经转换完成的 HbO/HbR 浓度对。
- 因此在统一缓存前，这个数据集应先标记为“波长通道输入”，不能直接和 mmol/L 的浓度型数据按相同语义对齐。
- 历史 Croce source/observation cache 曾采用 optical measurement space；该派生监督方案不再定义当前 measured-data loader。当前 Single-Trial 经显式 OD/MBLL 形成近似 HbO/HbR 分支，其他数据集保留已发布的色团导出；转换细节与限制见统一数据合同。

#### 适用场景
✅ Motor Imagery BCI  
✅ EEG-fNIRS融合研究  
✅ 多模态脑-机接口  
✅ 运动伪迹分析

---

### 2. REFED-dataset (Real-time Dynamic Labeled)

**目录**: `data/REFED-dataset/`

**说明文件**: `README.md`

#### 基本信息
- **来源**: NeurIPS 2025 Datasets Track (CC BY-NC-SA 4.0)
- **被试**: 32人 (18-34岁, 21男11女)
- **数据格式**: MATLAB (.mat)

#### 模态详情

| 模态 | 通道数 | 采样率 | 信号类型 | 数据格式 |
|------|-------|--------|---------|---------|
| EEG | 64 | 1000 Hz | 标准10-10命名（含 M1/M2、CB1/CB2） | `.mat` (channel × time) |
| fNIRS | 51 | 47.62 Hz | HbO, HbR, HbT, Abs 780/805/830nm | `.mat` (signal_type × channel × time) |

EEG 原始文件未提供逐被试数字化坐标，但 `EEG_channels.csv` 与官方通道分布图
给出固定 64-channel layout。旧 `standard_montage_v1` sidecar 使用版本化标准
montage 推导 within-EEG adjacency；它不是个体实测位置或 EEG-fNIRS 共配准。

2026-10-13 增加独立的 `refed_paper_layout_v1`：论文 Figure 2 与发布表的 EEG
身份及光学连线对应后，直接从联合布局提取两模态位置，支持示意图平面中的粗粒度
邻接配准。AFz 仅作为参考位置；原生 XYZ 不据此认定为毫米或转换为个体三维坐标。
坐标定义、版本入口与审计证据由 [数据合同](DATA_CONTRACT.md#refed-paper-layout-registration--2026-10-13) 持有。

#### fNIRS 原始保存语义与单位
- README 明确说明原始 fNIRS 张量同时包含 `HbO`, `HbR`, `HbT`, `Abs 780 nm`, `Abs 805 nm`, `Abs 830 nm` 六类信号。
- 由于同一张量同时混合浓度类信号和吸光度类信号，原始保存形式不存在单一统一单位；后续缓存必须显式记录所选 `signal_type`。
- 从幅值上看，HbO/HbR 大致在 `[-5, 5]`，而 Abs 三路主要落在 `[0.6, 4.5]`，也支持其语义并不相同。

#### 实验范式
- **刺激**: 15个情绪诱发视频 (60-170秒不等)
- **情绪类型**:
  - HVHA (High Valence, High Arousal): 快乐
  - HVLA (High Valence, Low Arousal): 放松
  - LVHA (Low Valence, High Arousal): 恐惧
  - LVLA (Low Valence, Low Arousal): 悲伤
  - MVMA (Medium): 中性

#### 标签类型
- **实时动态标注**: 被试使用摇杆实时标注 Valence 和 Arousal
- **SAM评分**: 每个视频后的主观情绪评分
- **PANAS**: 实验前后的正负情绪评估

#### 数据结构
```
REFED-dataset/
├── data/{subject_id}/
│   ├── EEG_baselines.mat    # 基线期EEG
│   ├── EEG_videos.mat       # 视频观看期EEG
│   ├── fNIRS_baselines.mat  # 基线期fNIRS
│   └── fNIRS_videos.mat     # 视频观看期fNIRS
└── annotations/{id}_label.mat  # 实时标注 (2 × time)
```

#### 适用场景
✅ 情感脑-机接口  
✅ 动态情绪识别  
✅ 神经血管耦合研究  
✅ 多模态情感计算

---

### 3. Visual Cognitive Motivation Study

**目录**: `data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/`

**说明文件**: `readme.txt`

#### 基本信息
- **来源**: Kyushu University (Data in Brief)
- **被试**: 16人 (健康成人)
- **数据格式**: EEG - EDF (原始) + MATLAB (预处理), fNIRS - Hitachi 原始格式

#### 模态详情

| 模态 | 设备 | 采样率 | 位置系统 | 格式 |
|------|------|--------|---------|------|
| EEG | Nihon Kohden Neurofax EEG-1100 | 500 Hz | 国际10-20系统 | `.edf` / `.mat` |
| fNIRS | Hitachi NIRS ETG-7100 | 10 Hz | 与EEG共享位置 | Oxy/Deoxy CSV 原始导出 |

#### 实验范式
- **刺激**: 250个不重复场景图片
- **试次结构**: 3s刺激呈现 + 9s决策期 (共12s/试次)
- **任务**: 决定是否想记住呈现的刺激
- **验证**: 实验后进行500张图片的再认测试

#### 标签类型
根据认知实验决策和再认测试结果组合:

| 标签 | 实验期决策 | 再认测试结果 | 含义 |
|------|-----------|-------------|------|
| RR | 想记住 | 记住了 | 高动机+成功记忆 |
| RF | 想记住 | 忘记了 | 高动机+记忆失败 |
| FR | 不想记住 | 记住了 | 低动机+意外记忆 |
| FF | 不想记住 | 忘记了 | 低动机+正常遗忘 |

#### Trigger说明
- **EEG (DC9/DC09通道)**: 1=刺激出现, 2=刺激消失, 3=被试响应
- **fNIRS (Mark)**: 1=刺激出现, 2=刺激消失, 3=被试响应

#### fNIRS 原始保存语义与单位
- 原始说明文件写明记录的是 `oxyhemoglobin` 和 `deoxyhemoglobin`，目录中的文件也按 `*_Oxy.csv` / `*_Deoxy.csv` 分开保存。
- 但同一说明文件同时指出这些 CSV 仍是 Hitachi ETG-7100 的 raw export without further processing；CSV 头部保留了 `Wave[nm]=695,830` 等设备元数据。
- 因此该数据集的当前保存语义应视为 Oxy/Deoxy 导出值，而不是 highWL/lowWL；不过原始文件未显式标出统一浓度单位，绝对幅值也明显大于 mmol/L 数据集，不能跨数据集直接按绝对数值对齐。

#### fNIRS 空间几何契约

- 112 个原始 fNIRS CSV 均声明 `Mode,4x4`，每个 Probe 含 CH1–CH24；`Graphical_recording_head_model.pdf` 给出双侧 optode 与 EEG 电极的相对头皮布局。
- `fNIRS_to_EEG_channel_reference.xlsx` 每侧给出 14 个 channel-to-EEG anchors，剩余 10 个为 `-`；锚点坐标来自 `Location.ced`，未标注通道按共享 optode 的 24-channel graph 做 harmonic interpolation。
- Probe1/Probe2 各生成 24 个完整 graphical-template channel coordinates 和 52 条无向邻接边；统一 loader 按 record 中的 probe 后缀选择对应侧 geometry。
- 这些坐标不是逐被试 digitization，也不提供真实 source/detector 三维点。它们只允许用于通道邻接、图模型、可视化和 coarse EEG-fNIRS alignment，不能用于精确光程、MBLL replay 或 exact co-registration。

#### Probe、Part 与通道身份

| 文件标识 | 实际含义 | 使用规则 |
| --- | --- | --- |
| `Probe1` | 左半球 fNIRS 探头阵列，名义位置 CH1–CH24 | 通道身份必须包含 Probe，例如 `Probe1:CH13`。 |
| `Probe2` | 右半球 fNIRS 探头阵列，名义位置 CH1–CH24 | 同编号与 Probe1 不同位置，例如 CH13 分别对应 FC3、FC4；Probe2 工作簿中的 `FP4` 误记由已有几何校正规则处理。 |
| `Part1` / `Part2` | 同一实验因中途休息等原因形成的时间分段 | 与左右半球无关；保留具体 Part 的文件、时间轴和事件身份。 |
| `Oxy` / `Deoxy` | 同一位置的 HbO / HbR 导出序列 | 两种色团是同一测量位置的两条序列，不是两个独立空间通道。 |

同一 Part 的两个 Probe 同时记录，共享同一份 EEG 和实验事件。一段完整双侧记录通常
对应 `Probe1_Oxy`、`Probe1_Deoxy`、`Probe2_Oxy`、`Probe2_Deoxy` 四个 CSV。
两侧通道排除标记分别读取；不能因为编号相同就共享有效性，也不能把重复使用的 EEG
计算为两份独立记录。依据为原论文 §3 与发布者
[`fNIRS_to_EEG_channel_reference.xlsx`](<../data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/fNIRS_to_EEG_channel_reference.xlsx>)。

#### 已核实的数据问题与处理影响（2026-11-10 证据快照）

本节汇总 **2026-11-10 对保留 v3 重建输入的核查**，不是滚动维护的实验状态。
精确记录身份、失败分母和数值以表末所链接的 JSON/CSV 为准；执行边界及版本迁移规则
由 [`DATA_CONTRACT.md`](DATA_CONTRACT.md#visual--refed-source-quality-finding--2026-11-10)
负责。本次已完成来源核查、通道统计和只改变 Hb 校准集合的对照，**尚未完成新生产缓存
及新 S/O 重拟合**。旧数据、尺度和结果均保留原身份。

**1. EEG 的论文电极数、发布通道和旧缓存身份需要区分。**

论文描述32个电极；经发布者 `Raw_EEG_channel_reference.xls`、EDF 和既有准备数组核对，
当前原生读取路径有31个正确映射且有数据支持的 EEG 通道，包括 `A64 → FCz`。
触发 `DC9/DC09` 和参考不计入这31条 EEG 信号。应以原始通道工作簿确定放大器编号，
以 `Location.ced` 的电极名称连接坐标；CED 的 `Number` 不是放大器通道ID。
历史 v5 EEG 路径曾混淆这两个编号并遗漏 FCz，已记录于
[Visual EEG 身份核查](DATA_CONTRACT.md#visual-eeg-channel-identity-finding--2026-10-29)。
本节的31通道结论对应已修正的原生 EEG 路径，不追溯重标旧数组；Hb缓存的设备排除问题
是另一个独立问题。fNIRS通道被排除不意味着同位置 EEG 自动失效。

**2. `Exception Ch` 是设备通道排除信息，不能只当普通注释。**

日立 *Topo Signal Processing* v2.08 手册（Q1J-FS0241-7，2006；
[原厂文档公开镜像](https://studylib.net/doc/25894667/manual)）§6.3 p.11说明该字段用于
排除测量通道，§6.8.1 p.25给出编码：**0为有效，1为排除**。这与某一时刻的运动标注
不同；设备排除状态本身也不说明是何种接触、布局或采集故障，不能自行补写原因。

此次核查范围为16名被试、27个 EEG 会话、54个 Probe 记录的108个 Oxy/Deoxy 文件。
54个记录均含排除通道；同一记录的两种色团标记一致。共542个“记录×测量位置”被
标记排除，却仍进入旧 v3 计划的可用通道集合。它不是542个独立物理传感器或被试。
原目录112个CSV与本次纳入108个CSV的统计范围不同；既有 S06 Part1 排除边界保持不变。

S06 Part2 的两个 Probe 均排除 CH2、5、6、9、15、17、19、20、22、24。
这个10通道名单在54个记录中的36个成立，其余记录有增减，**不能硬编码为整个数据集
通用名单**。工作簿中“有 EEG 对应标签的14个通道／标为 `-` 的10个通道”是位置映射
分组，不是通道有效性的替代定义；即使二者在某些记录中吻合，也必须使用逐记录标记。

核查时的实现链如下：

1. `src/data/event_alignment.py::read_visual_fnirs_csv` 保存 `Exception Ch` 等文件头，
   但声明 `annotation_only_not_signal_validity`。
2. `src/data/homer2_preprocessing.py` 的处理后有效性来自原数据有限性，未接入设备排除。
3. `experiments/build_clean_eeg_fnirs_cache.py::build_measurement_record` 将该有效性写入
   `fnirs_supported_channels`。
4. 既有 `_tak_cache_chunks` 和 `src/data/ssm_prepared.py::fit_feature_coordinate`
   据此将数值有限的设备排除通道纳入被试校准；几何插值提供坐标，不提供测量有效性。

**3. 实际可用数量应按空间位置、色团序列及独立会话分别报告。**

以下“可用”表示设备标记允许且满足既有数据支持检查，不等于已证明没有全部生理伪迹。
EEG支持检查覆盖各记录参与校准的全部保存窗口。

| 对象 | 本次保留记录中的数量 |
| --- | --- |
| 同一 EEG 会话 | 31个 EEG 通道；两个 Probe 共用，不能重复计成62个。 |
| 每侧 fNIRS Probe | 名义24个位置；可用12–15个位置。 |
| 双侧 fNIRS 合计 | 名义48个位置；可用25–29个，排除19–23个。 |
| 双侧可用 HbO/HbR 序列 | 50–58条，最常见为28个位置对应56条序列。 |
| 27个会话的双侧可用位置数分布 | 25个：1会话；26个：2会话；27个：3会话；28个：13会话；29个：8会话。 |

MNE图使用注册的CED／设备布局模板，鼻尖朝上、图左为左半球。绿色圆点标可用，红叉
标排除；频率图表示被试内记录排除率再对16人等权平均，不是脑激活图。部分位置由既有
几何插值获得，不能据此推定真实光程或逐被试皮层位置。

**4. 已发布 Hb 的单位语义与通道质量问题应分开处理。**

本地[Phukhachee等原论文](<paper_pdf/Phukhachee 等 - 2024 - A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults.pdf>)
§3、§4.5说明文件为设备导出的 Oxy/Deoxy，作者未进一步处理fNIRS；“raw export”
不表示数组是原始光强。日立手册§6.5 p.14将Hb显示纵轴标为 `mMmm`，指浓度×光程；
A/D图另以V为单位。该设备层证据仍不提供本数据集各通道的实际光程或额外导出换算，
所以保留既有 `relative:visual_cognitive_motivation` 坐标，不直接赋予µM单位。
也没有证据支持对这些已导出的Hb再次做OD/MBLL，或直接除以 `Analog Gain`、
`Digital Gain`；这些增益字段保留为采集元数据。

本文前面的历史幅值表包含当时检查范围内的原始通道，没有按本次设备排除标记重算。
因此它可用于辨认文件量级，不能作为有效通道的正常生理幅度参考，也不能凭幅值判断单位。

**5. 设备排除通道显著抬高了旧共同幅度分母。**

沿用既有每被试共同Hb归一化：

\[
g_s=\frac{0.025}{\sqrt{\{\operatorname{Var}_{\mathcal C_s}(HbO)
 +\operatorname{Var}_{\mathcal C_s}(HbR)\}/2}},\qquad
z_{s,w,c,k}(t)=g_s\,[x_{s,w,c,k}(t)-\bar x_{s,w,c,k,[0,5\mathrm{s})}].
\]

其中，\(\mathcal C_s\) 为该被试的校准时间×位置样本集合；\(w,c,k\) 分别是窗口、
位置和色团，\(x\) 是既有实测Hb缓存。旧集合包含所有准备路径判为有效的通道；新对照
进一步排除 `Exception Ch=1` 的位置。两者均保留原窗口及重叠权重、10 Hz采样、
既有全记录0.01–0.2 Hz处理和前5秒参考；同一被试所有记录、位置、HbO/HbR共用一个
因子，不按当前窗口或单个通道重新标准化。被试尺度来自既有合同允许的自身无标签数据，
包括已声明的验证／测试被试适配，不是严格零样本归一化。

16名被试的旧因子均从保存输入精确重现。设备排除通道贡献旧共同方差的
**98.69%–99.97%，中位数99.85%**；仅用可用通道后的因子为旧因子的
**6.68–44.33倍，中位数23.05倍**。方差贡献按原全体均值计算，包含组间均值偏离，
两组贡献之和等于原方差。可用集合在旧尺度下的共同SD为0.000564–0.003740，
在新尺度下按定义为0.025；这个目标值是归一化定义，不是拟合成绩。

S06的完整校准重放可具体说明影响：

| 校准集合 | 共同Hb SD（既有相对坐标） | 缩放因子 | 占原共同方差 |
| --- | ---: | ---: | ---: |
| 旧：全部通道 | 2.554036 | 0.00978843 | 100% |
| 仅 `Exception Ch=0` | 0.128789 | 0.19411542 | 0.1613% |
| `Exception Ch=1` 组，诊断统计 | 3.953144 | 0.00632408 | 99.8387% |

对同一 `S06_Part2_Probe1 / w00005 / CH12` 可用记录，新旧因子相差19.83倍。
其HbO波动SD由0.000609变为0.012084，HbR由0.000168变为0.003337，均为模型坐标。
同窗口排除的CH6在参考后的缓存坐标中，HbO/HbR SD为1.808049/1.650199，
可用CH12仅为0.062251/0.017191。所展示波形是既有处理后的实测输入，不是模型生成信号。

新旧归一化是同一波形乘以被试内共同正数，形态相关性为1；绝对功率乘以该数的平方，
相对谱形、谱重心和HbO/HbR相关性不变。若只统一换单位且相应变换校准因子，常数会
抵消；因此“补乘1000”不能解决排除通道混入校准的问题。

**6. 旧 S/O 结果的解释需要结合设备标记更新。**

旧v3模型中Hb的S直接取预测、O取X−S。按每记录真实标记重新分组，已有数值合格且
相关性可定义的拟合中，严重放大率如下；严重定义为SD(S)/SD(X)≥2，先记录内平均，
再记录及被试等权，重复窗口不作为独立被试。

| 设备标记组 | HbO严重放大率 | HbR严重放大率 | HbO平均corr(S,X) |
| --- | ---: | ---: | ---: |
| 0：可用 | 89.33% | 70.79% | 0.276 |
| 1：设备排除 | 2.78% | 0.65% | 0.890 |

这些数值描述旧拟合，不是修复后的成绩。高幅度、拟合较好的设备排除组不能作为正常
生理参考；小幅度可用组也不能仅因与模型不符被判为噪声。上述结果确认了旧尺度对可用
信号的压低，但归一化后S/O能改善多少仍须配对重拟合。仅将旧X、S、O统一乘以新因子，
SD(S)/SD(X)与相关性不变；S06 CH12的旧HbO/HbR幅度比4.0446/3.9357在这种操作下
原样保留。本次对照固定EEG PCA及EEG尺度，隔离Hb归一化变化，不代替完整的新校准流程。

**7. 后续实现应遵守的处理范围。**

按数据合同进行版本化迁移：逐记录核对Oxy/Deoxy标记与通道身份，将设备排除和既有
数值支持合成同一个有效性掩码，并在校准、PCA所用样本集合及拟合前生效。被排除的
位置保留身份和排除原因，但不以零值参与方差、损失或评价分母。保留共同Hb尺度公式、
两种色团的相对幅度与原时间支持；无需通过猜测单位或逐窗口自归一化掩盖数据边界问题。
旧缓存、尺度、拟合及负结果保留；新版本独立记录实际分母、配对重拟合和失败处理。
S06 Part1等已存在的记录排除不因这次通道修复而自动解除。

**复核证据与图表入口：**

| 内容 | 唯一数值来源／图表 |
| --- | --- |
| 原论文、设备手册、作者代码与实现链核查 | [来源复核报告](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/source_review_20261110_v1/report.html) |
| 每份原始文件头及排除名单 | [visual_headers.json](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/source_review_20261110_v1/visual_headers.json) |
| 按设备标记分组的旧S/O统计 | [dataset_source_flag_groups.csv](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/source_review_20261110_v1/dataset_source_flag_groups.csv) |
| 逐Probe掩码、名称与通道数量 | [record_counts.json](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/record_counts.json) |
| 27个会话的双侧数量 | [session_counts.csv](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/session_counts.csv) |
| 16人新旧因子、方差贡献和归一化SD | [subject_normalization.csv](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/subject_normalization.csv) |
| MNE位置、排除频率、实测与归一化波形 | [完整图表报告](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/report_v2/report.html) |
| 示例身份与幅度核验 | [example_selection.json](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/example_selection.json) · [example_metrics.csv](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/example_metrics.csv) |
| 检查与完成摘要 | [verification.json](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/verification.json) · [summary.json](../experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20261109_subject_pca_hb4_cache_v3/fnirs_cancellation_20261110_v1/visual_valid_normalization_20261110_v1/summary.json) |

#### 适用场景
✅ 记忆编码研究  
✅ 认知动机与注意力  
✅ 同步EEG-fNIRS记录方法学  
✅ 事件相关范式

#### 后续输入缓存迁移的证据入口

上方证据快照之后的生产迁移由
[`downstream_input_v2.yaml`](../experiments/configs/physiology_semantic_tokenizer/downstream_input_v2.yaml)
及[对应数据合同](DATA_CONTRACT.md#visual-measurement-v6--downstream-input-v2--2026-11-10)定义。
底层测量v6保存按原始CH身份压缩后的可用测量值；下游输入v2的Visual Hb明确应用
仅用可用通道计算的被试共同因子，归一化信息与逆变换保留。其他数据集、样本划分及
标签保持原合同；旧SSM结果不随输入缓存迁移而改变。

[新缓存统计与通道分布报告](../data/cache/physiology_downstream_v2/analysis_20261110_v1/report.html)
提供实际加载后的幅度统计、固定窗口频谱、各被试稳定/变化通道集合和Jaccard比较。
最新完成状态及校验分母由[下游验证记录](../data/cache/physiology_downstream_v2/verification.json)
和项目注册状态负责，本节不维护第二份当前计数。

---

### 4. Simultaneous EEG&NIRS (Cognitive Tasks)

**目录**: `data/Simultaneous EEG&NIRS/`

**说明文件**: 
- `Dataset description_BrainVision and NIRx.pdf`
- `Dataset description_MATLAB.pdf`
- `brain_image_data_classification/README.md`

#### 基本信息
- **来源**: Nature Scientific Data (sdata20183)
- **被试**: 26人 (健康成人)
- **数据格式**: BrainVision (.vhdr, .vmrk, .dat) 或 MATLAB

#### 模态详情

| 模态 | 设备 | 采样率 | 通道数 |
|------|------|--------|-------|
| EEG | BrainVision | 200 Hz MATLAB导出 | 30（28 scalp EEG + HEOG + VEOG） |
| fNIRS | NIRx | 10.4 Hz 原始采集, 10 Hz MATLAB导出 | 36个空间位置的 oxy/deoxy |

#### 实验范式
- **任务类型**: N-back、DSR（discrimination/selection/response）、word generation
- **认知负荷**: 不同难度等级
- **DSR**: O/Go（EEG code 16）与 X/No-go（code 32）刺激；发布文件为每人
  360 个 stimulus markers。论文正文写 180，当前协议保留并报告这一差异。

#### 数据结构
```
Simultaneous EEG&NIRS/
├── VP001-EEG/  # 被试1的EEG数据
├── VP001-NIRS/ # 被试1的fNIRS数据
├── VP002-EEG/
├── VP002-NIRS/
... (共26个被试)
└── brain_image_data_classification/  # 分类代码示例
```

#### fNIRS 原始保存语义与单位
- MATLAB 版原始文件 `cnt_{task}.mat` 直接以 `oxy` 和 `deoxy` 两个字段保存 fNIRS。
- 样例文件中 `yUnit = mmol/L`，`signal = NIRS (oxy, deoxy)`，说明当前目录下的 MATLAB 数据已经是 HbO/HbR 浓度表示。
- 其幅值集中在约 `10^-3` 到 `10^-2 mmol/L` 的量级，和 Single-Trial 的 `V` 量级、Visual 的 ETG-7100 原始导出量级明显不同。

#### 适用场景
✅ 认知负荷评估  
✅ 工作记忆研究  
✅ EEG-fNIRS同步采集方法学

#### 当前统一 EEG/DSR 契约

- 标准 loader 使用 `simultaneous_eeg_eog_clean_v1`：HEOG/VEOG 只作 robust
  nuisance regression 参考，输出中排除，最终为 28 个 scalp EEG channels；
- 该分支不附带坏道插值或 muscle-band attenuation，避免把眼动修复扩张成
  未单独论证的 EEG 重处理；
- DSR 的 fNIRS marker 只提供 block-level clock anchor。Go/No-go label 来自 EEG，
  fNIRS 仅作为同步血流动力学上下文；没有对齐 anchor 的 block 不生成样本；
- 保留的 v1 alignment inventory 为 25 人/8,980 个 DSR events，VP005 因 continuous
  drift 保持隔离。当前 v4 缓存由 v2 时钟规则重新裁定窗口，不能沿用这一历史计数。

---

## 数据使用注意事项

### 通用要求
1. **阅读原始文档**: 使用任何数据集前必须阅读对应目录中的说明文件
2. **引用要求**: 使用数据集请按要求引用原始论文
3. **许可证**: 注意各数据集的使用许可 (如 CC BY-NC-SA)
4. **预处理**: 了解数据是原始还是预处理过的

### 各数据集特殊注意事项

| 数据集 | 特殊注意事项 |
|--------|-------------|
| EEG+NIRS Single-Trial | 数据已下采样; 使用前需要BBCI Toolbox; fNIRS 当前保存为 `lowWL/highWL` 波长通道，不要直接按 HbO/HbR 解释 |
| REFED | 需同意非商业使用; fNIRS有6种信号类型且不共享单一单位，必须分别处理 |
| Visual Cognitive Motivation | EEG预处理数据已去除眼动伪迹epoch; fNIRS 为 Oxy/Deoxy 原始导出，绝对幅值不可直接与 mmol/L 数据集比较 |
| Simultaneous EEG&NIRS | MATLAB fNIRS 为 `oxy/deoxy`、`mmol/L`；EEG 输出前用 HEOG/VEOG 修复并排除；DSR symbol label 是 EEG-native，fNIRS 仅有 block anchor |

---

## 快速参考

### 按任务类型选择

| 任务类型 | 推荐数据集 |
|---------|-----------|
| Motor Imagery BCI | EEG+NIRS Single-Trial |
| 情感识别 | REFED-dataset |
| 认知记忆 | Visual Cognitive Motivation |
| 认知负荷 | Simultaneous EEG&NIRS |

### 按采样率选择

| 需求 | 数据集 | EEG采样率 | fNIRS采样率 |
|-----|--------|----------|------------|
| 高时间分辨率EEG | REFED | 1000 Hz | 47.62 Hz |
| 标准BCI采样 | EEG+NIRS Single-Trial | 200 Hz | 10 Hz |

---

*目录初版日期: 2026-05-29；采集方式与单位复核见本文开头的日期标记。*
