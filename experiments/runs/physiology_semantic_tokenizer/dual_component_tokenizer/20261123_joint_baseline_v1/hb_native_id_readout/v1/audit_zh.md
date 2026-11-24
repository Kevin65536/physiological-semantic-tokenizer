# 冻结Hb整数的原生X读出审计

日期为2026-11-23。事实源为本目录summary.json，完整原始数值见training_crossfit.json、screen/summary_v3.json、development/summary.json及各metrics.npz。

结论为保留新的可重放开发参照，联合目标尚未达标。新模型通过476固定筛查及57578开发窗的原数值检查，保持既有单码含义，但重复的分量幅度、事件及时序失败，以及完整开发出现的严重X尾部错误，阻止视觉验收。没有删库、删被试或删困难窗。

## 冻结模型和比较

输入仍是30秒X，Hb为10Hz并联合表示HbO/HbR，每位置每支每2秒一个ID。当前S129/O257编码器和两套码本完全冻结，新增0个整数。主模型使用当前、前一及后一2秒片段的冻结S/O两色解码波形各3个固定DCT系数，再加常数项，形成37个预测量，预测实测X-HbO减当前X-HbO的3个平滑系数。系数为37×3=111个float64标量，训练折RMS标准化已吸收入权重。合成的残差经过固定0.4Hz滤波，再加到原X-HbO。O两色、HbR及完整quiet pair逐值保持，S-HbO为新X减原O。解码只依赖已发整数和冻结模型。

使用固定15360个pattern_train位置窗、60名train被试、4折留人。新头排除对应训练被试，原tokenizer和G0训练尺度历史上已使用这些被试。本次没有把这称为整个模型的独立被试测试。拟合使用完整2秒目标片段，评分保留所有原mask、部分尾窗和困难窗。

6个候选在读取screen前冻结。相同真实残差、平滑基、折及ridge下比较波形匹配对照、当前类别加稀疏S×O、再加邻时类别，两档ridge为512和4096。最终由训练证据选择波形对照ridge512，111参数；类别方案没有超过该对照。

|候选|训练X-HbO NMSE|相对误差下降|训练准入|
|---|---:|---:|---|
|decoded_waveform__ridge512|0.01119251|15.08%|通过|
|decoded_waveform__ridge4096|0.01124729|14.66%|未通过|
|current_IDs__ridge512|0.01228590|6.78%|通过|
|current_IDs__ridge4096|0.01291477|2.01%|通过|
|context_IDs__ridge512|0.01185337|10.07%|通过|
|context_IDs__ridge4096|0.01266718|3.89%|通过|

获选头60/60名训练被试改善，四库在四折内都改善。旧944窗、32系数残差和本次完整15360窗、111系数对照同时改变了训练支持和表示，不能把其差异只归因于某个单独因素。

## 同一冻结头的结果

|面板|原X-HbO|新X-HbO|变化|
|---|---:|---:|---:|
|训练留人|0.01318004|0.01119251|+15.08%|
|476筛查|0.01182591|0.00949271|+19.73%|
|57578开发|0.03543146|0.03431556|+3.15%|

筛查和开发均覆盖同一16名开发被试，原G1历史上使用过全部validation被试。本次保留的4名预留被试及test没有访问，完整开发不称为新独立测试。

|完整开发分量|HbO NMSE|HbR NMSE|
|---|---:|---:|
|X|0.03431556|0.04432334|
|S|0.08719269|0.11723620|
|O|0.50418859|0.35496118|

NMSE为原mask上每窗MSE除以G0冻结的位置/分量/色团训练目标方差；NRMSE为其平方根。原单位RMSE和偏置使用冻结训练输入尺度恢复。位置→物理窗→记录→被试→数据集逐级等权，四库再等权。完整指标、各库、各被试、分布及非有限项均保留，见per_subject_metrics.csv与distribution_quantiles.json。分布单位是位置窗，不作为独立被试数。

完整开发REFED的X-HbO从0.0959931变为0.0984095，恶化2.52%；另外三库改善。该负项完整保留，未据validation重新选头或减小强度。

## 整数与词义证据

15360训练窗及476筛查窗的S/O整数与原强pattern模型逐项相同，训练背景X/S/O/mask对原G0银行全数相等。460自然换窗的原/移位ID也逐项相同。完整57578窗共863670个S和同数O整数，与旧源逐项相同；所有O行与HbR列指标数组逐值相同。

因ID、真实目标、mask、背景及训练支持不变，继承同一筛查面板上的background、single_ID和background+ID读出、模板、成员图、占有率及自然换窗证据。原integer_sequence_decoder一行不继承，新序列的波形/动态指标已单独实算。继承关系、源路径及hash见pattern_inheritance.json、pattern_inheritance_scope.json和development/inheritance_exact.json。

统一apply_joint_tokenizer入口新增hb_native_id_readout kind，真实新checkpoint经过独立CLI整数解码。新四支bundle保留旧EEG整数/固定元数据，Hb为476个筛查窗，ids.npz仅含整数及固定身份。voltage_and_Hb.npz的物理坐标往返误差不超过8.88e−16，NMSE复算差不超过2.22e−15；EEG、O和HbR在物理坐标逐值相同。

Hb每30秒每位置发15个S和15个O整数。声明S129/O257的逻辑定宽载荷为8.5 bit/s，当前NPZ的实际序列化成本、int64数组成本及经验熵另列integer_rate_and_usage.json，未实现bitpack。新增头不增加码率。旧源metadata中的3.5/4.0为加quiet前名义值，保留其历史身份。

## 实际视觉结果

已逐一打开筛查17套/34图和完整开发16套/32图，包括每库固定首窗、固定随机、历史P90、历史最差和候选最差。每图同时显示两色原幅度或独立去片均值的形态、30秒、14–16秒及13.5–14.5秒边界，目标与预测共坐标，没有展示平滑、独立缩放两条曲线或删除失败尾部。demeaned的片均值移除只用于形态诊断。

X的大趋势在许多记录恢复较好，但跨独立记录反复出现S/O相反的基线错误、峰谷幅度丢失和局部斜率/事件消失。筛查已能看到这些问题。完整开发又出现严重X错误，例如REFED subject17/video5/CH40/+60s，双颜色X原单位RMSE约0.252且相关为负；Single-Trial subject03/session04/C5FC5/+30s的X-HbO RMSE约0.115并明显压低大峰。原始图片和逐例决定在各role/visual_audit.json。

因此数值检查通过与单码含义保持，均不能代替本次未通过的视觉验收。固定O原值的这条读出路径无法消除O已有的幅度/事件失败；本轮不继续微调。

## 复现、资源与保留

最终配置为experiments/configs/physiology_semantic_tokenizer/dual_component_hb_native_id_readout_v1.yaml；训练入口为experiments/scripts/run_hb_native_id_readout.py；统一推理入口为experiments/scripts/apply_joint_tokenizer.py，四支bundle位于four_branch_replay_v1/bundle.json。native_readout.pt及current.pt/model.pt冻结依赖、每折头、训练整数/解码银行、完整指标、命令、源码身份和失败日志均保留。

已记录监督CPU合计776.183秒，另有本次小型汇总导出，低于2700秒预算。每服务16GiB、CPU配额200%、Torch2线程/数值库1线程，观察到内存current最大约2.74GiB。没有使用GPU或抢占无关任务。

最初screen导出漏填modality和integer_replay结构，另一次图审写路径变量遮蔽失败；保留了失败日志、原摘要及版本化恢复，未改变模型或指标。10项相关测试通过，完整测试收集成功，git diff --check通过。

本轮没有触发确认性4人、独立source训练种子或test读取，因为联合视觉条件尚未满足。最终状态为“开发参照已冻结、联合目标尚未达标”。
