#!/usr/bin/env python3
"""Read retained N1--N7 evidence and write a separate, reproducible report.

No inference, native-data loading, or modification of run evidence occurs here.
The fixed task table, terminal ledger and family result tables remain the owners.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import fitz
import numpy as np
import pandas as pd
import markdown

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.metrics.trajectory_reliability import canonical_residual_fields

FAMILIES = [f'N{i}' for i in range(1, 8)]
MODS = ['EEG', 'HbO', 'HbR']
TARGETS = ['r', 'clean_EEG', 'clean_HbO', 'clean_HbR']
SUBJECTS = ['subject_01', 'subject_09', 'subject_18']
SESSIONS = ['session_01', 'session_03', 'session_05']
COLORS = ['#176B87', '#CE793A', '#7557A5', '#248A74', '#B64A50']
STATE_COLORS = {'completed': '#278975', 'failed_contract': '#DAA34A',
                'failed_domain': '#C14E54', 'failed_numerical': '#8064A2',
                'not_implemented': '#8B929D', 'timeout': '#3E5868', 'not_started_budget': '#BBC3CC'}
MASKS = ['full', 'center_EEG', 'center_fNIRS', 'whole_EEG', 'whole_fNIRS']
CONDITIONS = ['matched_student_t', 'combined', 'drift', 'correlated_hb', 'outliers', 'independent_pairing']
CONDITION_NAMES = ['匹配 t', '时间处理', '低频漂移', 'Hb 相关误差', '稀疏异常', '独立配对']
OUTER_MASKS = ['full', 'EEG_only', 'fNIRS_only', 'all_missing', 'center_EEG',
              'center_EEG_own', 'center_EEG_template', 'center_EEG_pairing', 'center_EEG_shift',
              'center_fNIRS', 'center_fNIRS_own', 'center_fNIRS_template', 'center_fNIRS_pairing', 'center_fNIRS_shift']
CORE_FIGURES = ['01_completion', '05_n1_null_linear', '08_n1_tails',
                'reader_n2_comparison', 'n2_solver_checks', 'n3_inner',
                'n4_artifact', 'n5_inner', 'n5_w_curves', 'n6_inner',
                'n6_replay', 'reader_rules_summary', 'reader_n7_selection',
                'n7_synthetic_ablation']
KEEP = {'task_id', 'kind', 'rule', 'candidate', 'row_id', 'mode', 'status', 'metrics', 'center_metrics',
        'parameter_log_likelihood', 'physical_checks', 'physical_checks_at_transformed_mean', 'trajectory_path',
        'trial', 'subject', 'session', 'sample_id', 'outer', 'predictive_residual_structure', 'smoothing_residual_structure',
        'truth', 'reference_truth', 'r_driver_replay', 'hidden_truth', 'noisy_prediction', 'starts',
        'retained_rank', 'observed_coordinates', 'support_residual_norm', 'converged_start_max_path_difference',
        'rank_sensitivity', 'spatial', 'strength_training_sd', 'r_change_rms', 'teacher_change_nrmse',
        'first_difference_scale', 'effective_noise_scale', 'trial_difference_structure', 'bootstrap_scale_quantiles',
        'replay_gap_nrmse', 'replay_vs_observed', 'joint_vs_observed', 'standardized_transition_rms',
        'grouped_transition_rms', 'transition_domain_failures', 'complete_transition_denominator', 'nmse', 'increment',
        'hbt', 'error', 'flow_domain_exit', 'processed_visible_rmse'}


def decode(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def read_json(path):
    return canonical_residual_fields(json.loads(path.read_text()))


def read_csv(path, keep=None):
    csv.field_size_limit(20_000_000)
    with path.open() as stream:
        rows = []
        for row in csv.DictReader(stream):
            fields = canonical_residual_fields(row)
            rows.append(canonical_residual_fields({k: decode(v) for k, v in fields.items()
                                                   if keep is None or k in keep}))
        return rows


def nested(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return np.nan
        obj = obj.get(key)
    return float(obj) if isinstance(obj, (int, float)) else np.nan


def equal_mean(rows, function):
    """Documented trial -> session -> subject equal weighting, finite subset."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = function(row)
        if np.isfinite(value):
            groups[row['subject']][row['session']].append(value)
    return float(np.mean([np.mean([np.mean(v) for v in sessions.values()])
                          for sessions in groups.values()])) if groups else np.nan


def mean_metric(rows, field, target, metric='nrmse'):
    values = [nested(r, field, target, metric) for r in rows if r.get('status') == 'completed']
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def short_subject(subject):
    return 'S' + subject[-2:]


def rule_label(row):
    return row['family'] + ('/' + row['rule'] if row['family'] == 'N7' else '')


def candidate_label(candidate):
    name = candidate.get('id', 'baseline') if isinstance(candidate, dict) else str(candidate)
    return {'baseline': '基线', 'process_h2_r1': 'h×2', 'process_h0.25_r1': 'h×0.25',
            'process_h1_r0.5': 'r×0.5', 'process_h1_r2': 'r×2'}.get(name, name)


def finite_number(value, digits=3):
    return f'{value:.{digits}f}' if value is not None and np.isfinite(value) else '未估计'


def table(headers, rows):
    return '\n'.join(['|'+'|'.join(map(str, headers))+'|', '|'+'|'.join(['---']*len(headers))+'|'] +
                     ['|'+'|'.join(str(v).replace('|', '/') for v in row)+'|' for row in rows])


def save_table(out, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/(name+'.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def heat(ax, values, rows, cols, title='', cmap='YlGnBu', center=False, annotate=True,
         vmin=None, vmax=None, fmt='.2f', colorbar=True):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    palette = plt.get_cmap(('RdBu' if cmap == 'RdBu' else 'RdBu_r') if center else cmap).copy()
    palette.set_bad('#E3E6E9')
    kwargs = {}
    if center:
        bound = max(float(np.max(np.abs(finite))) if finite.size else 1., 1e-12)
        kwargs['norm'] = TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound)
    else:
        kwargs.update(vmin=vmin, vmax=vmax)
    image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=palette, **kwargs)
    ax.set_xticks(range(len(cols)), cols, rotation=35 if len(cols) > 6 else 0,
                  ha='right' if len(cols) > 6 else 'center')
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, loc='left', fontweight='bold', pad=10)
    if annotate:
        for i in range(len(rows)):
            for j in range(len(cols)):
                value = values[i, j]
                text = format(value, fmt) if np.isfinite(value) else '—'
                color = 'white' if np.isfinite(value) and (image.norm(value) > .73 or center and image.norm(value) < .2) else '#243444'
                ax.text(j, i, text, ha='center', va='center', color=color,
                        fontsize=7 if len(rows) > 18 or len(cols) > 9 else 8)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=.025, pad=.025)
    return image


def setup_style():
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'font.size': 10, 'axes.titlesize': 11,
                         'figure.titlesize': 15, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'savefig.dpi': 170,
                         'svg.fonttype': 'path', 'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def describe_figure(name, title, original, fig):
    """Keep each figure's content, reading instructions and finding explicit."""
    reading = '按子图标题选择比较条件，再在同一色标内比较；误差越小越好，灰格是缺失或不适用。'
    conclusions = {
        '01_completion': '运行完成不等于评价完整。候选只有51–53/72个完整外折，没有规则通过本轮筛选。',
        '02_failures': '三处基线选择折缺口传播到全部规则；跨族相同缺口不能算作独立失败。',
        '03_n1_residuals': '完整输入仍有模态残差与session差异；W=-0.5增加有效拟合数，但不足以建立状态恢复资格。',
        '04_n1_masks': '增加联合输入没有在所有目标、所有遮挡方式下稳定占优；不同成功分母限制直接比较。',
        '05_n1_null_linear': 'EEG的正确联合预测平均逊于自身上下文；现有联合状态尚无稳定的跨模态预测优势。',
        '06_n1_compromise': '联合输入明显改变r，却没有相应稳定的遮挡预测增益，状态改变本身不足以证明共享信息。',
        '07_n1_acf': '一步预测残差与平滑残差含有时间相关结构，逐点独立误差近似未消除时间处理失配。',
        '08_n1_tails': '残差尾部明显高于中位数；单个平均误差无法概括不同模态的极端偏差。',
        '09_n1_influence': 'r变化较大时仍可出现零增益或负增益；对另一模态敏感不等于预测获益。',
        '10_n1_hbt': 'HbO/HbR相加后仍存在session相关偏差；代数关系正确不能替代观测恢复检查。',
        'n2_uncertainty': '均值改善没有自动带来校准资格；O2缺少区间，不能由O0/O1区间代替。',
        'n2_shared_noisy': 'O1确有共享噪声条件区间；已见目标的覆盖1和宽度0不是完美的未来观测预测。',
        'n2_solver_checks': '固定初值和秩容差检查支持工程一致性；这些检查没有给出MAP后验或teacher资格。',
        'n3_inner': '全部9个有效选择折均选倍率1；放大或缩小共同fNIRS噪声未形成可用改进。',
        'n4_inner': 'E2被选5次、E4被选1次、基线3次；外折fNIRS风险仅改善0.034%，没有稳定收益。',
        'n5_inner': '8/9个有效折选择更大的观测增益；实测子集风险下降26.30%，但合成r误差增加0.103。',
        'n6_inner': '8折选择血流过程噪声×2，1折选择r过程噪声×0.5；12.99%的子集改善伴随真值恢复代价。',
        'n7_inner': 'GW规则9折中5折保留基线；没有有效折同时选中非零G与非零W。',
        'n3_noise': '训练噪声估计与模型实际尺度并非同一个量；仅调整共同倍率没有解决候选筛选问题。',
        'n4_artifact': 'E2抑制额部注入主要来自排除额部通道；90/90拟合完成没有证明神经clean真值恢复。',
        'n5_w_curves': '8条完整曲线及其重采样均触W=-0.5边界；重复触边不能解释为参数可辨识。',
        'n5_surfaces': '只有3/9个二维曲面完整；一维W边界现象不能用来宣称观测增益与生理时间已被分离。',
        'n7_surfaces': '4/9个完整曲面均最大于G=0.6、W=-0.5的双边界，其余5个不完整；未得到内部稳定最优点。',
        'n6_replay': '64个成功回放的EEG/HbO/HbR平均闭合差为0/0.181/0.461；Hb回放不能完全复现联合均值。',
        'n6_transition': '联合均值与确定性r驱动回放存在闭合差；状态转移残差可参与拟合，但这里不能量化私有信息比例。',
        'n6_failure_traces': '5/6案例复现越界；跳过最后更新仅修复其中2例的下一步，最后一条观测不是唯一原因。',
        'rules_comparison': 'N5/N6较大的实测子集收益伴随合成代价；N7/GW对W-only仅改善1.51%，全部规则仍未合格。',
        'rules_modality_costs': '综合风险下降并未保证各模态及配对增量同时改善；N5/N6的收益不能视为无代价改进。',
        'n7_synthetic_ablation': 'GW比W-only略好，却仍差于固定基线；新增G的局部收益不足以改变本轮否定结论。',
        'synthetic_references': '联合模型的优势取决于目标与失配条件，不能将单一匹配条件的优势推广到全部压力场景。',
        'reader_n2_comparison': 'combined条件下O1已取得大部分改善；O2进一步收益较小，主要正面证据指向观测时间与噪声合同。',
        'reader_rules_summary': 'N5/N6实测子集分别改善26.30%/12.99%，同时产生合成代价；N7改善1.51%，仍低于10%门槛。',
        'reader_n7_selection': '实测选择没有同时放开G和W；描述性曲面却重复触双边界，两者都未支持稳定生理参数适配。',
    }
    if name == '01_completion':
        reading = '左侧按N1–N7读任务完成比例；右侧蓝色为14种输入/对照均有效的trial，灰色为72个固定身份中的缺口。'
    elif name.endswith('_inner'):
        original = '全部预先固定候选在三被试×四个外折中的内折风险。候选只使用对应训练边界；基线失败的选择折保留为空，不另挑成功子集。'
        reading = '行是被试与外折，列是候选；风险变化小于0更好，星号为通过模态退化约束后选中的候选。灰格不参与补选。'
    elif name in ('n5_surfaces', 'n7_surfaces'):
        reading = '横轴W，纵轴为观测增益或G；颜色为本session相对最大log L，越接近0越好。只在完整曲面标星，不能跨子图比较色深。'
        if name == 'n7_surfaces':
            original = original.split('N5 的增益')[0]
    elif name == 'n5_w_curves':
        reading = '九格对应三被试×三session，横轴W，纵轴为相对最大log L；星号是完整曲线的最大值，虚线是下降2的参考线。'
    elif name in ('05_n1_null_linear', 'rules_modality_costs'):
        reading = '先看增量符号：对照−正确联合NMSE大于0才是联合获益；误差差值大于0则表示退化。各子图标题注明不同口径。'
    elif name == 'n6_replay':
        reading = '三子图对应EEG/HbO/HbR，横轴为被试，每点是一条成功回放；黑线为被试均值，纵轴越大表示闭合差越大。'
    elif name == 'n6_failure_traces':
        reading = '横轴为事件相对时间；两条曲线比较观测更新前后未来一个步长内的最小流量，跌破红色零线表示流量越界。'
    elif name == '07_n1_acf':
        reading = '横轴为滞后秒数，纵轴为ACF，颜色区分被试；上排为一步预测残差，下排为平滑残差，远离零线表示剩余时间相关。'
    elif name == '03_n1_residuals':
        reading = '行是被试/session，列是模态；上下两排分别为W=0/−0.5，三列面板依次为RMSE、带符号bias、MAE。误差越低越好，bias越接近0越好。'
    elif name == '04_n1_masks':
        reading = '左右区分W；上排比较完整窗输入模式，下排只比较中心遮挡目标。行是对照，列是模态；同一色标下NRMSE越小越好，括号列出有效分母。'
    elif name == '06_n1_compromise':
        reading = '左/中散点按固定trial身份排列，纵轴为联合与单模态的r差，颜色区分W；右侧热图为联合−单模态NMSE，正值表示完整输入拟合代价。'
    elif name == '08_n1_tails':
        reading = '三个面板是EEG/HbO/HbR，行区分W，列是trial内绝对残差的p50/p90/p95；同一面板中色深和数值越大，表示相应分位的误差越大。'
    elif name == '09_n1_influence':
        reading = '横轴为对照引起的r改变，纵轴为对照−正确联合NMSE，颜色区分对照；零线以上才是联合获益，右下方表示状态变了但预测变差。'
    elif name == '10_n1_hbt':
        reading = '行对应被试/session，列区分W；左侧bias可正可负，接近0更好，右侧RMSE越小越好。只在当前Hb观测坐标内解读。'
    elif name in ('n2_uncertainty', 'n2_shared_noisy'):
        reading = '先按标题区分生成规律与输入条件，再对照覆盖率和宽度：95%覆盖需接近0.95，区间宽度需同时查看。已见目标的覆盖1、宽度0是条件化结果。' if name == 'n2_shared_noisy' else '每行是一种W/时间处理/输入模式；左右分别看clean覆盖与宽度，列区分solver和目标。覆盖需与0.95比较，宽区间带来的高覆盖不等于精确恢复。'
    elif name == 'n3_noise':
        reading = '上排按训练折比较重采样区间、差分点估计和实际噪声尺度，纵轴为对数尺度；下排按被试看一阶差分ACF，非零滞后不接近0表示仍有时间结构。'
    elif name == 'n4_artifact':
        reading = '行是六个EEG分支，列是额部/后部注入及强度，三面板是恢复目标；数值越大表示人工伪迹引起的teacher改变越大，不能把小改变量直接解释为更准确。'
    elif name == 'n6_transition':
        reading = '左侧按被试比较不同状态组的状态转移残差；右侧按六种assessment条件比较EEG/HbO/HbR回放差。两侧归一化分母不同，只在各面板内比较大小。'
    elif name == 'synthetic_references':
        reading = '上下排是独立discovery/assessment种子流，列面板区分恢复目标；每格按六种失配条件比较原带噪、自身平滑和联合基线，NRMSE越低越好。'
    elif name == 'n2_solver_checks':
        reading = '左右分布图的横轴是差异的log10，越靠左表示差异越小；中图计数为初值状态，不能当作独立trial分母。'
    elif name == 'reader_n2_comparison':
        reading = '左为匹配model，右为combined时间处理；每组三色分别为O0/O1/O2，柱高为同一24个trial的平均NRMSE，越低越好。'
    elif name in ('reader_rules_summary', 'rules_comparison'):
        reading = '左侧风险下降百分比越大越好，虚线为10%；右侧合成ΔNRMSE为正表示退化。每行使用自身共同有效子集，不能跨行排名。'
    elif name == 'reader_n7_selection':
        reading = '左为9个有效选择折的G/W选择次数，0是从未选中；右分别显示描述性session曲面完整性及完整曲面的最大值位置。'
    elif name == 'n7_synthetic_ablation':
        reading = '两行仅改变参照，分别减固定基线和减W-only；列为恢复目标，负ΔNRMSE才是GW改善，色标以0为中心。'
    elif name.startswith('n2_') and name.endswith(('means', 'hidden')):
        reading = '行按输入mask分组，每组三行为O0/O1/O2；列为r/EEG/HbO/HbR；上下为W=0/−0.5，左右为model/combined。NRMSE越小越好。'
        if 'student_t' in name:
            conclusions[name] = '匹配Student-t的完整输入均值中O0优于Gaussian近似；Gaussian MAP的改善并不跨噪声规律普遍成立。' if name.endswith('means') else '隐藏区间仍随输入模式和噪声失配而变化；均值图中的优势不能自动外推到所有隐藏目标。'
        elif 'nonlinear_gaussian' in name:
            conclusions[name] = 'combined完整输入下r/HbO/HbR由O0的0.855/1.007/0.941降至O2的0.540/0.232/0.277；O1已实现大部分改善。' if name.endswith('means') else '隐藏区间须单独判断，缺失整模态时不等于完整输入拟合；该面板没有提供实测teacher资格。'
        else:
            conclusions[name] = '线性匹配参考为区分观测时间处理与非线性近似提供对照，不能把线性参考本身当作非线性Student-t资格。'
    elif name.startswith('n7_truth_'):
        reading = '行是29个固定候选，列是真G、真W、联合扰动及观测增益的正/负方向；每列8个trial，四子图按恢复目标区分，误差越低越好。'
    elif name.startswith('n7_likelihood_'):
        reading = '行是候选，列是真值扰动方向；三个子图区分输入模式。Δlog L大于0表示优于固定基线，但不同输入模式的绝对值不可比较。'
        conclusions[name] = '仅EEG输入下候选似然差均小于1e-8，无法靠该似然分辨G/W；出现其他方向响应也不等于参数恢复。'
    elif name.startswith('synthetic_') and name.endswith('_uq'):
        reading = '行是候选，列是六种合成条件；从上到下依次读覆盖、宽度、完成数。覆盖须与95%目标及相应宽度共同判断。'
        conclusions[name] = '候选完成数不等于区间校准，覆盖与宽度随失配条件变化；本面板不新增UQ资格。'
    elif name.startswith('synthetic_') and name != 'synthetic_references':
        reading = '行是候选×输入模式（全/E/N），列是六种合成条件，四子图是r/EEG/HbO/HbR；每格16个trial，NRMSE越低越好。'
    if name not in conclusions:
        arrays = [np.asarray(im.get_array(), float) for ax in fig.axes for im in ax.images]
        finite = arrays[0][np.isfinite(arrays[0])] if arrays else np.array([])
        assert finite.size, ('A figure needs a supported conclusion', name)
        conclusions[name] = f'本图首个恢复目标的平均NRMSE跨展示条件为{finite.min():.3f}–{finite.max():.3f}，结果依赖输入与扰动方向；不能把其中最小格推广为整套实验优势。'
    return dict(title=title, content=original, reading=reading,
                conclusion=conclusions[name], caption=f'内容：{original} 读法：{reading} 结论：{conclusions[name]}')


def figure(out, figures, name, title, caption, draw, figsize=(13, 8), *, description=None):
    if name in CORE_FIGURES:
        figsize = (8.5, min(5.2, max(2.7, figsize[1]*8.5/figsize[0])))
    fig = plt.figure(figsize=figsize, layout='constrained')
    draw(fig)
    if name in CORE_FIGURES:
        for ax in fig.axes:
            ax.title.set_fontsize(8.5)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(labelsize=7.5)
    # Captions are document text, not tiny labels burned into the image.
    fig.get_layout_engine().set(rect=(0, 0, 1, 1))
    description = description or describe_figure(name, title, caption, fig)
    for extension in ('png', 'svg'):
        fig.savefig(out/'figures'/f'{name}.{extension}', bbox_inches='tight')
    # The document compositor uses the SVG glyph outlines to avoid CFF/Type42
    # embedding corruption and adds searchable captions outside each image.
    plt.close(fig)
    figures[name] = description
    print('figure', name, flush=True)


def load_evidence(run, previous):
    tasks = read_csv(run/'task_table.csv')
    tasks = [t['payload'] for t in tasks]
    by_id = {t['id']: t for t in tasks}
    assert len(tasks) == len(by_id)
    ledger = [json.loads(line) for line in (run/'case_status.jsonl').read_text().splitlines() if line.strip()]
    registered = [r for r in ledger if r['task_id'] in by_id]
    counts = Counter(r['task_id'] for r in registered)
    assert set(counts) == set(by_id), 'Missing terminal identities'
    assert set(counts.values()) == {1}, 'Duplicate terminal identities'
    summaries = {f: read_json(run/f/'summary.json') for f in FAMILIES}
    for f in FAMILIES:
        observed = Counter(r['status'] for r in registered if r['family'] == f)
        assert dict(observed) == summaries[f]['status_counts'], (f, observed)
    identity = read_json(run/'frozen_input_identity.json')
    checked = {}
    for name in ['task_table.csv', 'resolved_config.yaml', 'scope_inventory.json', 'fold_inventory.json', 'preflight.json']:
        checked[name] = hashlib.sha256((run/name).read_bytes()).hexdigest() == identity[name]
    assert all(checked.values()), 'Frozen metadata changed'
    scope = read_json(run/'scope_inventory.json')
    assert set(scope) == set(SUBJECTS)
    assert sum(len(v) for v in scope.values()) == 72
    data = {}
    for f in FAMILIES:
        data[f] = read_csv(run/f/'trial_metrics.csv', KEEP)
        for row in data[f]:
            task = by_id[row['task_id']]
            row['_task'] = task
            row['_candidate'] = task.get('candidate')
        print('loaded', f, len(data[f]), flush=True)
    selections = {}
    traces = []
    for t in tasks:
        if t['kind'] == 'select':
            selections[t['id']] = read_json(run/'cells'/t['id']/'result.json')
        elif t['kind'] == 'failure_trace':
            traces.append((t, read_json(run/'cells'/t['id']/'result.json')['rows'][0]))
    scientific_equal = {}
    numeric_differences = {}
    for f in FAMILIES[:-1]:
        fields = ['task_id', 'row_id', 'status', 'metrics', 'center_metrics', 'truth', 'hidden_truth', 'parameter_log_likelihood']
        a = read_csv(previous/f/'trial_metrics.csv', set(fields))
        b = [{k: r.get(k) for k in fields if k in r} for r in data[f]]
        # Column additions with only empty values have no scientific content.
        def signature(rows):
            return hashlib.sha256(json.dumps([{k: r.get(k) for k in fields} for r in rows], sort_keys=True).encode()).hexdigest()
        scientific_equal[f] = signature(a) == signature(b)
        if not scientific_equal[f]:
            maximum = 0.
            structure = 0
            def compare(x, y):
                nonlocal maximum, structure
                if isinstance(x, dict) and isinstance(y, dict):
                    structure += set(x) != set(y)
                    for key in set(x) & set(y):compare(x[key], y[key])
                elif isinstance(x, list) and isinstance(y, list):
                    structure += len(x) != len(y)
                    for v,w in zip(x,y):compare(v,w)
                elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
                    if np.isfinite(x) and np.isfinite(y):maximum=max(maximum,abs(x-y))
                elif x != y:
                    structure += 1
            compare([{k:r.get(k) for k in fields} for r in a], [{k:r.get(k) for k in fields} for r in b])
            numeric_differences[f] = dict(maximum_absolute_numeric_difference=maximum, nonnumeric_mismatches=structure)
    audit = dict(primary_run=run.name, previous_run=previous.name, frozen_metadata_checks=checked,
                 registered_tasks=len(tasks), terminal_tasks=len(registered),
                 terminal_states=dict(Counter(r['status'] for r in registered)),
                 unique_training_trials=72, repeated_N1_to_N6_scientific_fields_equal=scientific_equal,
                 repeated_run_numeric_differences=numeric_differences,
                 classification_note='task status differs from row fit status; reused fits and dependent failures are not independent trials')
    audit['result_table_sha256']={f+'/trial_metrics.csv':hashlib.sha256((run/f/'trial_metrics.csv').read_bytes()).hexdigest() for f in FAMILIES}
    return tasks, by_id, registered, summaries, data, selections, traces, audit


def outer_scores(data, selections):
    scored = defaultdict(list)
    for family, rows in data.items():
        grouped = defaultdict(dict)
        for row in rows:
            if row['kind'] in ('outer', 'selected_outer'):
                grouped[row['task_id']][row['mode']] = row
        for identifier, modes in grouped.items():
            task = next(iter(modes.values()))['_task']
            if not all(m in modes and modes[m]['status'] == 'completed' for m in OUTER_MASKS):
                continue
            nmse = np.array([nested(modes['center_EEG'], 'center_metrics', 'EEG', 'nmse'),
                             *[nested(modes['center_fNIRS'], 'center_metrics', m, 'nmse') for m in MODS[1:]]])
            if not np.isfinite(nmse).all():
                continue
            label = f"N1/W={task['candidate']['w']:g}" if family == 'N1' else rule_label(dict(family=family, rule=task.get('rule')))
            full = modes['full']
            record = dict(subject=task['subject'], session=full['session'], sample_id=full['sample_id'],
                          outer=task['outer'], trial=task['trial'], nmse=nmse, B=float(nmse@[.5,.25,.25]),
                          fnirs=float(np.mean(nmse[1:])))
            for control in ('own', 'template', 'pairing', 'shift'):
                null = np.array([nested(modes['center_EEG_'+control], 'center_metrics', 'EEG', 'nmse'),
                                 *[nested(modes['center_fNIRS_'+control], 'center_metrics', m, 'nmse') for m in MODS[1:]]])
                record[control] = null-nmse
            scored[label].append(record)
    return scored


def plot_completion(out, figs, summaries, ledger, rules, selections, tasks, data, scored):
    def draw(fig):
        axes = fig.subplots(1, 2, width_ratios=[1.1, 1])
        left = np.zeros(7)
        for state, color in STATE_COLORS.items():
            x = np.array([summaries[f]['status_counts'].get(state, 0)/summaries[f]['expected_cells']*100 for f in FAMILIES])
            if x.sum():
                axes[0].barh(FAMILIES, x, left=left, color=color, label=state)
                left += x
        for i, f in enumerate(FAMILIES):
            s = summaries[f]
            axes[0].text(101, i, f"{s['status_counts'].get('completed',0):,}/{s['expected_cells']:,}", va='center', fontsize=9)
        axes[0].set(xlim=(0, 124), xlabel='任务 cell 成功占比（%）', title='所有固定 cell 均已到终态')
        axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=7)
        labels = ['N1/W=0', 'N1/W=-0.5']+[rule_label(r) for r in rules]
        complete = [len(scored.get(n, [])) for n in labels]
        axes[1].barh(labels, complete, color=COLORS[0], label='14 种输入/对照均有效')
        axes[1].barh(labels, 72-np.array(complete), left=complete, color='#D9DEE3', label='缺口')
        for i, n in enumerate(complete):axes[1].text(n/2, i, f'{n}/72', va='center', ha='center', color='white', fontsize=9)
        axes[1].set(xlabel='固定外折 trial 数', xlim=(0, 72), title='科学评价完整性')
        axes[1].legend(fontsize=8, loc='lower right')
        for ax in axes:ax.invert_yaxis()
    figure(out, figs, '01_completion', '运行已结束，候选外折评价仍不完整',
           '左：固定任务终态。右：每个 trial 的 14 种输入及对照全部有效才计入完整外折；不能用 cell 完成数替代。', draw, (14, 7))
    def draw_fail(fig):
        axes = fig.subplots(1, 2, width_ratios=[1, 1.7])
        states = ['failed_domain', 'failed_numerical', 'failed_contract']
        counts = [[sum(r.get('status') == s for r in data[f]) for s in states] for f in FAMILIES]
        heat(axes[0], counts, FAMILIES, ['物理域', '数值', '合同'], '保存的结果行失败数（含诊断行与复用）', fmt='.0f')
        panel = [t for t in tasks if t['kind']=='select']
        labels = [rule_label(r) for r in rules]
        values = np.full((len(labels), 12), np.nan)
        for t in panel:
            label = rule_label(dict(family=t['family'], rule=t.get('rule')))
            i = labels.index(label); j=SUBJECTS.index(t['subject'])*4+t['outer']
            values[i,j] = selections[t['id']]['status']=='completed'
        heat(axes[1], values, labels, [f'S{s[-2:]}·{k}' for s in SUBJECTS for k in range(4)],
             '选择折：1=已定义，0=基线内折失败', cmap='YlGn', vmin=0, vmax=1, fmt='.0f', colorbar=False)
    figure(out, figs, '02_failures', '失败来源与跨族依赖传播',
           '同一基线在 S09 的外折 0/3、S18 的外折 2 不完整；每条规则因此损失 3 个选择折及 18 个外折任务。行失败数含复用，不是独立失败事件数。', draw_fail, (15, 6))


def plot_n1(out, figs, run, data, scored):
    rows = [r for r in data['N1'] if r['kind']=='outer']
    def full_draw(fig):
        axes = fig.subplots(2, 3)
        for i,w in enumerate([0.,-.5]):
            panel = [r for r in rows if r['_task']['candidate']['w']==w and r['mode']=='full' and r['status']=='completed']
            for j,(metric,title) in enumerate([('nrmse','RMSE / 训练 SD'),('bias_sd','有符号 bias / 训练 SD'),('mae_sd','MAE / 训练 SD')]):
                grid = [[equal_mean([r for r in panel if r['subject']==s and r['session']==q],lambda r:nested(r,'metrics',m,metric))
                         for m in MODS] for s in SUBJECTS for q in SESSIONS]
                heat(axes[i,j],grid,[f'S{s[-2:]}·{q[-2:]}' for s in SUBJECTS for q in SESSIONS],MODS,
                     f'W={w:g} · {title}',center=metric=='bias_sd')
    figure(out,figs,'03_n1_residuals','N1｜完整输入仍存在显著模态残差',
           '每个格为该 session 有效 trial 的均值。误差符号为预测−目标；跨 trial 先计算后聚合。完整输入 W=0 有效 69/72，W=-0.5 有效 71/72。',full_draw,(14,10))
    def masks_draw(fig):
        axes=fig.subplots(2,2)
        for j,w in enumerate([0.,-.5]):
            panel=[r for r in rows if r['_task']['candidate']['w']==w]
            modes=['full','EEG_only','fNIRS_only','all_missing']
            values=[];labels=[]
            for mode in modes:
                rs=[r for r in panel if r['mode']==mode and r['status']=='completed']
                values.append([np.sqrt(equal_mean(rs,lambda r:nested(r,'metrics',m,'nmse'))) for m in MODS])
                labels.append(f'{mode} ({len(rs)}/72)')
            heat(axes[0,j],values,labels,MODS,f'W={w:g} · 全窗 NRMSE')
            values=[];labels=[]
            for suffix,name in [('', '正确联合'),('_own','自身上下文'),('_template','自身+任务模板'),('_pairing','独立配对'),('_shift','环形移位')]:
                v=[];counts=[]
                for m in MODS:
                    mode=('center_EEG' if m=='EEG' else 'center_fNIRS')+suffix
                    rs=[r for r in panel if r['mode']==mode and r['status']=='completed']
                    v.append(np.sqrt(equal_mean(rs,lambda r:nested(r,'center_metrics',m,'nmse'))));counts.append(len(rs))
                values.append(v);labels.append(f'{name} ({counts[0]}/{counts[1]}/72)')
            heat(axes[1,j],values,labels,MODS,f'W={w:g} · 中心 4 秒目标 NRMSE')
    figure(out,figs,'04_n1_masks','N1｜整模态缺失、中心遮挡与四种对照',
           '等权 trial→session→subject 后对 NMSE 开方。上图各模态都评分；下图只评分被遮挡目标。各行分母不同，仅作条件描述；配对增量见下一图。下图括号为 EEG/fNIRS 有效 trial 数，分母均为 72。',masks_draw,(14,10))
    def null_draw(fig):
        axes=fig.subplots(1,3)
        for j,label in enumerate(['N1/W=0','N1/W=-0.5']):
            rs=scored[label]
            values=[[equal_mean(rs,lambda r:r[control][i]) for i in range(3)] for control in ['own','template','pairing','shift']]
            heat(axes[j],values,['自身','任务模板','独立配对','移位'],MODS,f'{label} · n={len(rs)}/72',center=True,cmap='RdBu')
        values=[]; labels=[]
        linear=[r for r in data['N1'] if r['kind']=='linear']
        for modality in ['EEG','fNIRS']:
            rs=[r for r in linear if r['_task']['modality']==modality and r['status']=='completed']
            for kind in ['basic','joint','pairing','shift']:
                labels.append(modality+' / '+kind)
                values.append([np.sqrt(equal_mean(rs,lambda r:nested(r,'nmse',kind)))])
        heat(axes[2],values,labels,['NRMSE'],'同折线性基线 · 每方向 72/72')
    figure(out,figs,'05_n1_null_linear','N1｜配对增量与线性基线',
           '前两图：对照 NMSE−正确联合 NMSE，正值才表示正确联合更好；使用每个 W 下全部模式共同有效的 trial。线性 fNIRS 指标是两种 Hb 的平均 NMSE 再开方；不与 EEG 坐标混为一个终点。',null_draw,(16,6))
    comp=read_csv(run/'N1/compromise_by_subject_session.csv')
    influence=read_csv(run/'N1/modality_influence.csv')
    def compromise_draw(fig):
        axes=fig.subplots(1,3)
        for j,own in enumerate(['EEG_only','fNIRS_only']):
            for k,w in enumerate([0.,-.5]):
                panel=[r for r in comp if r['own']==own and r['w']==w and r['status']=='completed']
                v=[r['r_standardized_rms_difference'] for r in panel]
                x=np.arange(len(v));axes[j].scatter(x,v,s=15,alpha=.6,color=COLORS[k],label=f'W={w:g} n={len(v)}/72')
            axes[j].set(title='joint vs '+own,ylabel='r RMS 差 / EEG 训练 SD',xlabel='按固定身份顺序排列的有效 trial');axes[j].legend(fontsize=8)
        groups=['EEG','HbO','HbR']; vals=[]
        for w in [0.,-.5]:
            vals.append([equal_mean([r for r in comp if r['w']==w and r['own']==('EEG_only' if m=='EEG' else 'fNIRS_only') and r['status']=='completed'],lambda r:nested(r,'signed_full_fit_cost',m)) for m in groups])
        heat(axes[2],vals,['W=0','W=-0.5'],groups,'joint−单模态：完整输入 NMSE 代价',center=True)
    figure(out,figs,'06_n1_compromise','N1｜共享状态改变与模态妥协代价',
           'r 改变量大不等于共享信息增加，必须结合上一图的遮挡和配对结果。散点展示全部有效身份；颜色按 W 分组。',compromise_draw,(16,5.5))
    def acf_draw(fig):
        axes=fig.subplots(2,3)
        panel=[r for r in rows if r['_task']['candidate']['w']==0 and r['mode']=='full' and r['status']=='completed']
        for i,field in enumerate(['predictive_residual_structure','smoothing_residual_structure']):
            for j,m in enumerate(MODS):
                for k,s in enumerate(SUBJECTS):
                    rs=[r for r in panel if r['subject']==s]
                    arrays=[r[field][m]['acf'] for r in rs if isinstance(r.get(field),dict) and m in r[field]]
                    if arrays:
                        axes[i,j].plot(np.arange(len(arrays[0]))/4,np.mean(arrays,axis=0),color=COLORS[k],label=short_subject(s))
                axes[i,j].axhline(0,color='#AEB7C0',lw=.7);axes[i,j].set(title=m+(' · 一步预测残差' if i==0 else ' · 平滑残差'),xlabel='滞后（秒）',ylabel='ACF',ylim=(-.4,1.05));axes[i,j].legend(fontsize=8)
    figure(out,figs,'07_n1_acf','N1｜预测残差和平滑残差的时间相关结构',
           'W=0、完整输入有效 trial；各被试内作描述性均值。不能把这些相关时间点当作独立重复，亦不能由低残差直接推出正确的概率模型。',acf_draw,(14,8))
    def quantile_draw(fig):
        axes=fig.subplots(1,3)
        for j,m in enumerate(MODS):
            values=[]
            for w in [0.,-.5]:
                rs=[r for r in rows if r['_task']['candidate']['w']==w and r['mode']=='full' and r['status']=='completed']
                values.append([equal_mean(rs,lambda r:r['metrics'][m]['absolute_residual_sd_quantiles'][i]) for i in range(3)])
            heat(axes[j],values,['W=0','W=-0.5'],['p50','p90','p95'],m+' · |残差|/训练 SD')
    figure(out,figs,'08_n1_tails','N1｜绝对残差的中位数与尾部',
           '先取每个 trial 内的残差分位数，再按 session、subject 等权汇总；不是把所有时间点混合后的总体分位数。',quantile_draw,(13,4))
    def influence_draw(fig):
        axes=fig.subplots(2,3)
        for i,w in enumerate([0.,-.5]):
            for j,m in enumerate(MODS):
                for k,control in enumerate(['own','template','pairing','shift']):
                    rs=[r for r in influence if r['w']==w and r['target']==('EEG' if m=='EEG' else 'fNIRS') and r['control']==control and r['status']=='completed']
                    axes[i,j].scatter([r['r_change_normalized_rms'] for r in rs],
                                      [r['prediction_nmse_increment_of_joint'][m] for r in rs],
                                      label=f'{control} n={len(rs)}/72',color=COLORS[k],s=14,alpha=.5)
                axes[i,j].axhline(0,color='#98A5AD',ls='--');axes[i,j].set(title=f'W={w:g} · {m}',xlabel='r RMS 改变 / EEG训练SD',ylabel='对照−正确联合 NMSE');axes[i,j].legend(fontsize=6)
    figure(out,figs,'09_n1_influence','N1｜改变共享状态是否带来预测增量',
           '每个点为一个固定外折 trial 与一种对照的配对结果；纵轴正值表示正确联合预测更好。横向改变大而纵向增量不足，不能解释为共享神经信息的证据。所有有效身份保留，不删除极端值。',influence_draw,(16,10))
    def hbt_draw(fig):
        axes=fig.subplots(1,2)
        for j,(metric,label) in enumerate([('bias_coordinate','HbT bias'),('rmse_coordinate','HbT RMSE')]):
            vals=[]
            for s in SUBJECTS:
                for q in SESSIONS:
                    vals.append([equal_mean([r for r in rows if r['subject']==s and r['session']==q and r['mode']=='full' and r['_task']['candidate']['w']==w and r['status']=='completed'],lambda r:nested(r,'hbt',metric)) for w in [0.,-.5]])
            heat(axes[j],vals,[f'S{s[-2:]}·{q[-2:]}' for s in SUBJECTS for q in SESSIONS],['W=0','W=-0.5'],label,center=metric=='bias_coordinate',fmt='.3f')
    figure(out,figs,'10_n1_hbt','N1｜HbT 代数与当前坐标残差',
           'HbT按HbO+HbR逐点相加，采用相同的Hb坐标尺度。图示当前观测坐标单位，不能直接当作绝对浓度；仅含完整输入有效trial。',hbt_draw,(11,7))


def plot_n2(out,figs,data):
    rows=data['N2']
    assert set(r['_task']['mode'] for r in rows) == set(MASKS)
    expected_groups=Counter((r['_task']['law'],r['_task']['variant'],r['_task']['w'],r['_task']['mode'],r['_task']['solver']) for r in rows)
    assert all(n==(8 if key[3].startswith('whole_') else 24) for key,n in expected_groups.items())
    laws=['linearized_gaussian','nonlinear_gaussian','nonlinear_student_t']
    law_names=['线性 Gaussian','非线性 Gaussian','非线性 Student-t stress']
    labels=[f'{m} / {s}' for m in MASKS for s in ['O0','O1','O2']]
    for law,lname in zip(laws,law_names):
        for field,suffix,description in [('truth','means','全轨迹 clean NRMSE'),('hidden_truth','hidden','隐藏区间 clean NRMSE')]:
            def draw(fig,law=law,field=field):
                axes=fig.subplots(2,2)
                for i,w in enumerate([0.,-.5]):
                    for j,variant in enumerate(['model','combined']):
                        values=[]
                        for mask in MASKS:
                            for solver in ['O0','O1','O2']:
                                panel=[r for r in rows if r['_task']['law']==law and r['_task']['variant']==variant and r['_task']['w']==w and r['_task']['mode']==mask and r['_task']['solver']==solver]
                                values.append([mean_metric(panel,field,t) for t in TARGETS])
                        heat(axes[i,j],values,labels,['r','EEG','HbO','HbR'],f'{variant} · W={w:g}')
            figure(out,figs,f'n2_{law}_{suffix}',f'N2｜{lname}：{description}',
                   '每个 full/中心遮挡格含 24 个独立合成 trial；整模态缺失格含固定的前 8 个 trial。各格全部完成。O0=逐点近似，O1=静息线性时序，O2=非线性相关 Gaussian MAP。隐藏区间表的 full 行不适用；指标按各自评分区间的 truth SD 归一化。',draw,(15,14))
    def uq_draw(fig):
        axes=fig.subplots(3,2)
        for i,law in enumerate(laws):
            groups=[(w,v,m) for w in [0.,-.5] for v in ['model','combined'] for m in MASKS]
            labels=[f'{w:g}/{v}/{m}' for w,v,m in groups]
            for j,(metric,title) in enumerate([('coverage95','clean 95% 覆盖'),('mean_interval_width','clean 区间平均宽度（坐标单位）')]):
                values=[]
                for w,v,m in groups:
                    vals=[]
                    for solver in ['O0','O1']:
                        panel=[r for r in rows if r['_task']['law']==law and r['_task']['variant']==v and r['_task']['w']==w and r['_task']['mode']==m and r['_task']['solver']==solver]
                        vals.extend(mean_metric(panel,'truth',t,metric) for t in ['r','clean_HbO','clean_HbR'])
                    values.append(vals)
                heat(axes[i,j],values,labels,['O0 r','O0 HbO','O0 HbR','O1 r','O1 HbO','O1 HbR'],law_names[i]+' · '+title,
                     vmin=0 if metric=='coverage95' else None,vmax=1 if metric=='coverage95' else None,fmt='.2f' if metric=='coverage95' else '.3f')
    figure(out,figs,'n2_uncertainty','N2｜覆盖率必须与区间宽度共同解读',
           '完整轨迹 clean 区间的条件近似评价；EEG 与 r 在该模型下数值相同，图中只列 r。O2 未估计区间、参数后验或边际似然；O1另有共享原生噪声的带噪目标条件区间，见下一图。',uq_draw,(18,23))
    def noisy_draw(fig):
        axes=fig.subplots(3,3)
        for i,law in enumerate(laws):
            groups=[(w,v,m) for w in [0.,-.5] for v in ['model','combined'] for m in MASKS]
            labels=[f'{w:g}/{v}/{m}' for w,v,m in groups]
            for j,(metric,title) in enumerate([('coverage95','全轨迹带噪覆盖'),('hidden_coverage95','隐藏时段带噪覆盖'),('mean_interval_width','全轨迹平均宽度')]):
                vals=[]
                for w,v,m in groups:
                    panel=[r for r in rows if r['_task']['law']==law and r['_task']['w']==w and r['_task']['variant']==v and r['_task']['mode']==m and r['_task']['solver']=='O1']
                    arrays=[r['noisy_prediction'].get(metric) for r in panel if r.get('noisy_prediction') and r['noisy_prediction'].get(metric) is not None]
                    vals.append(np.mean(arrays,axis=0) if arrays else [np.nan]*3)
                heat(axes[i,j],vals,labels,MODS,law_names[i]+' · '+title,vmin=0,vmax=1 if 'coverage' in metric else None,fmt='.2f' if 'coverage' in metric else '.3f')
    figure(out,figs,'n2_shared_noisy','N2｜O1 已实现的共享原生噪声条件预测',
           'O1在R_target,input中保留同一原生噪声；O0/O2对应带噪区间未估计。条件已给定的观测坐标可以出现覆盖1、宽度0，这是对已见随机变量的条件化，不是新噪声预测或完美校准。隐藏覆盖在整个被遮挡时段计算，图中的可见模态仍可能是已给定值。',noisy_draw,(20,22))
    def solver_draw(fig):
        axes=fig.subplots(1,3)
        panels=[r for r in rows if r['_task']['solver']=='O2']
        gaps=[r['converged_start_max_path_difference'] for r in panels if r.get('converged_start_max_path_difference') is not None]
        axes[0].hist(np.log10(np.maximum(gaps,1e-16)),bins=25,color=COLORS[0]);axes[0].set(xlabel='log10(两收敛初值的最大状态路径差)',ylabel='cell 数',title=f'可比较初值 {len(gaps)}/{len(panels)}')
        states=Counter(s['status'] for r in panels for s in (r.get('starts') or []))
        axes[1].bar(list(states),list(states.values()),color=COLORS[:len(states)]);axes[1].tick_params(axis='x',rotation=25);axes[1].set(title='O2 两个固定初值状态',ylabel='初值数')
        checks=[s for r in rows for s in (r.get('rank_sensitivity') or [])]
        valid=[r.get('maximum_state_mean_difference') for r in checks if r.get('status')=='completed' and r.get('maximum_state_mean_difference') is not None]
        axes[2].hist(np.log10(np.maximum(valid,1e-16)),bins=20,color=COLORS[2]);axes[2].set(title=f'秩容差对照 {len(valid)}/{len(checks)}',xlabel='log10(最大均值差)',ylabel='对照数')
    figure(out,figs,'n2_solver_checks','N2｜多初值与秩容差的工程核查',
           '初值失败、不可行试探和最终案例失败分别保留。多初值接近仅支持这些固定案例的求解一致性；不等同于后验校准。',solver_draw,(16,5))


def plot_inner(out,figs,tasks,selections):
    for family in ['N3','N4','N5','N6','N7']:
        panel=[t for t in tasks if t['family']==family and t['kind']=='select' and (family!='N7' or t['rule']=='GW')]
        names=list(selections[panel[0]['id']]['candidates'])
        values=[];labels=[];choices=[]
        for t in panel:
            record=selections[t['id']];base=record['candidates']['baseline']['risk'];line=[]
            endpoint='fnirs' if family=='N4' else 'B'
            for name in names:
                risk=record['candidates'][name]['risk']
                line.append(100*(risk[endpoint]/base[endpoint]-1) if base and risk else np.nan)
            values.append(line);labels.append(f"S{t['subject'][-2:]}·折{t['outer']}")
            choices.append(record.get('selected',{}).get('id'))
        def draw(fig,values=values,labels=labels,names=names,choices=choices):
            ax=fig.subplots()
            heat(ax,values,labels,[candidate_label(n) for n in names],'相对同折基线风险变化（%；负值更好）',center=True,annotate=len(names)<10)
            for i,name in enumerate(choices):
                if name in names:ax.scatter(names.index(name),i,marker='*',s=120,facecolor='#F1C453',edgecolor='#303B45',linewidth=.7)
        figure(out,figs,f'{family.lower()}_inner',f'{family}｜全部冻结候选的内折风险',
               '星号为同时满足模态退化约束后选中的候选；风险最低不一定入选。空白保留拟合或基线缺口，不补选。N4 只以固定 fNIRS 终点跨 EEG 坐标比较；N7 此图展示 GW 的共享候选网格，W-only/G-only 是其同折子集。',draw,(17 if family=='N7' else 12,7))


def plot_noise_artifact(out,figs,data):
    rows=[r for r in data['N3'] if r['kind']=='noise_diagnostic']
    def noise_draw(fig):
        axes=fig.subplots(2,3)
        for j,m in enumerate(MODS):
            x=np.arange(len(rows));point=np.array([r['first_difference_scale'][j] for r in rows]);quant=np.array([r['bootstrap_scale_quantiles'] for r in rows])
            axes[0,j].errorbar(x,quant[:,1,j],yerr=[quant[:,1,j]-quant[:,0,j],quant[:,2,j]-quant[:,1,j]],fmt='o',color=COLORS[0],label='训练分块重采样 2.5–97.5%')
            axes[0,j].scatter(x,point,marker='x',color=COLORS[1],label='一阶差分点估计')
            axes[0,j].plot(x,[r['effective_noise_scale'][j] for r in rows],color=COLORS[2],label='实际模型噪声尺度')
            axes[0,j].set_yscale('log');axes[0,j].set_xticks(x,[f"{r['subject'][-2:]}·{r['outer']}" for r in rows],rotation=60);axes[0,j].set(title=m,ylabel='当前观测坐标单位（对数轴）');axes[0,j].legend(fontsize=6)
            for k,s in enumerate(SUBJECTS):
                arrays=[v[m]['acf'] for r in rows if r['subject']==s for v in r['trial_difference_structure']]
                axes[1,j].plot(np.arange(len(arrays[0]))/4,np.mean(arrays,axis=0),color=COLORS[k],label=short_subject(s))
            axes[1,j].set(xlabel='滞后（秒）',ylabel='一阶差分 ACF');axes[1,j].legend(fontsize=8)
    figure(out,figs,'n3_noise','N3｜训练噪声尺度及相关性',
           '12 个训练折；每折 200 次 trial 内移动块重采样，块长 8 点。重采样区间是噪声描述区间，不是生理参数区间。实际噪声尺度还受既有下限规则约束。',noise_draw,(16,9))
    artifacts=[r for r in data['N4'] if r['kind']=='artifact' and r.get('spatial')]
    def artifact_draw(fig):
        axes=fig.subplots(1,3)
        groups=[(s,a) for s in ['frontal','posterior'] for a in [.5,1.]]
        labels=[f'{s} ×{a:g}' for s,a in groups]
        for j,m in enumerate(MODS):
            values=[]
            for eeg in [f'E{k}' for k in range(6)]:
                values.append([np.mean([r['teacher_change_nrmse'][j] for r in artifacts if r['_task']['eeg']==eeg and r['spatial']==s and r['strength_training_sd']==a and r.get('teacher_change_nrmse')]) for s,a in groups])
            heat(axes[j],values,[f'E{k}' for k in range(6)],labels,m+' teacher 改变量 / 训练 SD')
    figure(out,figs,'n4_artifact','N4｜全部人工伪迹条件的敏感性',
           '3 被试×6 EEG 分支×（1 未注入参考+4 注入条件）共 90/90 拟合有效。每格为 3 被试均值。未注入实测窗口不是 clean 神经真值；E2 对仅额部注入接近零主要来自排除相应通道的构造。',artifact_draw,(16,6))


def plot_surfaces(out,figs,run):
    sessions=read_csv(run/'N5/session_reproducibility.csv')
    def curves_draw(fig):
        axes=fig.subplots(3,3)
        for r,ax in zip(sessions,axes.flat):
            if r.get('log_likelihood'):
                values=np.array(r['log_likelihood']);w=np.array(r['w_grid'])
                ax.plot(w,values-values.max(),'-o',ms=3,color=COLORS[0]);ax.axhline(-2,color=COLORS[1],ls='--',lw=.8)
                ax.scatter(r['estimate_w'],0,marker='*',s=100,color=COLORS[1]);ax.set_ylim(min(values-values.max())*1.08,20)
            else:ax.text(.5,.5,'曲线不完整\n不报告 W 最大值',ha='center',va='center',transform=ax.transAxes)
            ax.set(title=f"S{r['subject'][-2:]}·{r['session'][-2:]} ({r['completed_grid_values']}/{r['expected_grid_values']})",xlabel='W',ylabel='log L−完整网格最大值')
    figure(out,figs,'n5_w_curves','N5｜共同坐标下的 W 曲线与 session 重复性',
           '8/9 条一维曲线完整，8 条均在 W=-0.5 达到边界最大值；200 次固定 gauge 重采样和奇偶半样本均回到该边界。重复触边不是可辨识性或人群稳定性。S09/session03 保留 124/136 个有效值。',curves_draw,(14,11))
    for family,filename in [('N5','session_reproducibility.csv'),('N7','gw_session_surfaces.csv')]:
        records=read_csv(run/family/filename)
        def surface_draw(fig,records=records,family=family):
            axes=fig.subplots(3,3)
            for r,ax in zip(records,axes.flat):
                surface=r['surface'];grid=surface['grid'];scores=surface['summed_log_likelihood']
                ws=sorted(set(g[0] for g in grid));qs=sorted(set(g[1] for g in grid));values=np.full((len(qs),len(ws)),np.nan)
                finite=[v for v in scores if v is not None];maximum=max(finite) if finite else 0.
                for (w,q),v in zip(grid,scores):
                    if v is not None:values[qs.index(q),ws.index(w)]=v-maximum
                heat(ax,values,[f'{q:g}' for q in qs],[f'{w:g}' for w in ws],
                     f"S{r['subject'][-2:]}·{r['session'][-2:]} · {surface['completed_grid_values']}/{surface['expected_grid_values']}",annotate=False,cmap='viridis')
                ax.set_xlabel('W');ax.set_ylabel('观测增益 a_N' if family=='N5' else 'G')
                if surface['complete']:
                    k=np.nanargmax(values);i,j=np.unravel_index(k,values.shape);ax.scatter(j,i,marker='*',s=120,color='#F1C453',edgecolor='white',lw=.5)
        figure(out,figs,f'{family.lower()}_surfaces',f'{family}｜'+('W–观测增益' if family=='N5' else 'G–W')+' 共同坐标似然曲面',
               '每个有效格都要求同一 session 的 8 个 trial 全部有效；缺一即灰色，不对成功子集求和。颜色为相对该图可见最大值的 Δlog L，刻度按 session 独立；仅完整曲面标星。N5 的增益=1 行额外有 17 点细网格，其他增益行仅 9 点，交错灰格属于未设计。',surface_draw,(15,12))


def plot_replay_trace(out,figs,data,traces):
    replay=[r for r in data['N6'] if r['kind']=='replay' and r['status']=='completed']
    def replay_draw(fig):
        axes=fig.subplots(1,3)
        for j,m in enumerate(MODS):
            for i,s in enumerate(SUBJECTS):
                values=[r['replay_gap_nrmse'][j] for r in replay if r['subject']==s]
                axes[j].scatter(np.full(len(values),i)+np.linspace(-.13,.13,len(values)),values,s=16,alpha=.65,color=COLORS[i])
                if values:axes[j].plot([i-.17,i+.17],[np.mean(values)]*2,color='#253443',lw=2)
            axes[j].set_xticks(range(3),[short_subject(s) for s in SUBJECTS]);axes[j].set(title=m,ylabel='确定性 r 驱动回放−联合均值 RMS / 训练 SD')
    figure(out,figs,'n6_replay','N6｜确定性 r 驱动回放的闭合差',
           '72 个固定身份中 64 个回放完成，3 个基线全输入不可用，5 个积分失败。黑线为被试有效 trial 均值。总体成功子集 EEG/HbO/HbR 平均差为 0/0.181/0.461；非线性后验均值不必满足确定性闭合，因此该差不是私有信息比例。',replay_draw,(15,5))
    def transition_draw(fig):
        axes=fig.subplots(1,2)
        keys=sorted({k for r in replay for k in (r.get('grouped_transition_rms') or {})})
        values=[[np.mean([r['grouped_transition_rms'][k] for r in replay if r['subject']==s and r.get('grouped_transition_rms') and k in r['grouped_transition_rms']]) for k in keys] for s in SUBJECTS]
        heat(axes[0],values,[short_subject(s) for s in SUBJECTS],keys,'分组状态转移残差 RMS / 固定过程尺度')
        synthetic=[r for r in data['N1'] if r['kind']=='synthetic' and r['mode']=='full' and r['_task']['candidate']['id']=='baseline' and r['_task']['stream']=='assessment']
        vals=[]
        for condition in CONDITIONS:
            rs=[r for r in synthetic if r['_task']['condition']==condition and (r.get('r_driver_replay') or {}).get('status')=='completed']
            vals.append([np.mean([r['r_driver_replay']['gap_nrmse'][j] for r in rs]) if rs else np.nan for j in range(3)])
        heat(axes[1],vals,CONDITION_NAMES,MODS,'合成基线回放差 / 生成器原clean SD')
    figure(out,figs,'n6_transition','N6｜状态转移残差和合成回放参照',
           '左：有效实测回放的分组转移诊断；右：独立 assessment 六条件，每条件最多 16 trial。实测和合成的归一化分母不同，不能直接把数值相除解释为比例。',transition_draw,(15,6))
    def trace_draw(fig):
        axes=fig.subplots(2,3)
        for (task,row),ax in zip(traces,axes.flat):
            trace=row.get('retained_update_risk_trace',[])
            for key,color in [('before',COLORS[0]),('after',COLORS[1])]:
                x=[v['time_index']/4-5 for v in trace];y=[v[key]['minimum_flow'] for v in trace]
                ax.plot(x,y,'-o',ms=4,color=color,label='观测更新前' if key=='before' else '观测更新后')
            ax.axhline(0,color='#BB4852',ls='--');ax.set(title=f"S{task['subject'][-2:]}·旧索引{task['trial']}·W={task['w']:g}",xlabel='事件相对时间（秒）',ylabel='下一 dt 内最小流量 f')
            if trace:ax.legend(fontsize=7)
            else:ax.text(.5,.5,'未发生 flow exit',ha='center',va='center',transform=ax.transAxes)
    figure(out,figs,'n6_failure_traces','N6｜全部六个固定越界复现案例',
           '5/6 复现 flow-domain exit，1/6 完成。图示 owner 保留的最后 4 次更新，横坐标为原事件时间；只追踪所保留时间窗，不反推更早风险。跳过最后更新仅在 2 个失败案例中使下一步恢复正流量。',trace_draw,(15,10))


def plot_rules(out,figs,rules,scored):
    def draw(fig):
        axes=fig.subplots(1,3,width_ratios=[1.3,1,1.3])
        labels=[rule_label(r) for r in rules]
        changes=[100*(1-r['common_success_only_risk']/r['common_success_only_baseline_risk']) for r in rules]
        axes[0].barh(labels,changes,color=[COLORS[0] if v>=0 else COLORS[1] for v in changes]);axes[0].axvline(10,color='#AD4554',ls='--',label='预设 10% 门槛')
        for i,v in enumerate(changes):axes[0].text(max(v,0)+.4,i,f'{v:+.2f}%',va='center',fontsize=9)
        axes[0].set(xlabel='共同有效子集风险下降（%）',title='不构成完整规则排名',xlim=(-3,34));axes[0].invert_yaxis();axes[0].legend(fontsize=8)
        synth=[r for r in rules if isinstance(r.get('synthetic_assessment'),dict) and r['synthetic_assessment'].get('mean_nrmse_change') is not None]
        heat(axes[1],[r['synthetic_assessment']['mean_nrmse_change'] for r in synth],[rule_label(r) for r in synth],['r','EEG','HbO','HbR'],'独立匹配合成 ΔNRMSE',center=True)
        nr=next(r for r in rules if r['family']=='N7' and r['rule']=='GW')
        common={r['sample_id']:r for r in scored['N7/W_only']};rows=[r for r in scored['N7/GW'] if r['sample_id'] in common]
        for i,s in enumerate(SUBJECTS):
            subset=[r for r in rows if r['subject']==s];x=[common[r['sample_id']]['B'] for r in subset];y=[r['B'] for r in subset]
            axes[2].scatter(x,y,label=f'{short_subject(s)} n={len(subset)}',alpha=.7,s=24,color=COLORS[i])
        limit=max([r['B'] for r in rows]+[common[r['sample_id']]['B'] for r in rows])*1.05
        axes[2].plot([0,limit],[0,limit],ls='--',color='#8D969E');axes[2].set(xlabel='同折 W-only B',ylabel='GW B',title=f'GW vs W-only：共同 n={len(rows)}/72');axes[2].legend(fontsize=8)
    figure(out,figs,'rules_comparison','候选结果｜实测子集改善与合成代价',
           '左：每行仅限其自身共同有效身份，N4 用固定 fNIRS 风险，N7/GW 参考 W-only，其他参考 W=0；禁止跨行直接排名。中：按 9 个已定义选择折的频率加权，来自同一组 16 个独立 assessment trial，144 是加权比较数，不是 144 个独立 trial。中图各规则统一参考 N1 固定基线，GW 的实测主消融仍以 W-only 为准。',draw,(18,6))
    def costs_draw(fig):
        axes=fig.subplots(1,2,width_ratios=[1,2.1])
        costs=[];nulls=[];labels=[]
        for rule in rules:
            label=rule_label(rule)
            reference='N7/W_only' if label=='N7/GW' else 'N1/W=0'
            by_id={r['sample_id']:r for r in scored[reference]}
            panel=[r for r in scored[label] if r['sample_id'] in by_id]
            baseline=[by_id[r['sample_id']] for r in panel]
            delta=[np.sqrt(equal_mean(panel,lambda r:r['nmse'][i]))-np.sqrt(equal_mean(baseline,lambda r:r['nmse'][i])) for i in range(3)]
            change=[value for null in ['own','template','pairing','shift'] for value in rule['null_increment_changes'][null]]
            if rule['family']=='N4':
                delta[0]=np.nan
                for k in [0,3,6,9]:change[k]=np.nan
            costs.append(delta);nulls.append(change);labels.append(f'{label} n={len(panel)}/72')
        heat(axes[0],costs,labels,MODS,'实测 ΔNRMSE：正值为退化',center=True,fmt='.3f')
        heat(axes[1],nulls,labels,[c+'/'+m for c in ['own','template','pairing','shift'] for m in MODS],
             '配对增量变化：负值为损失',center=True,cmap='RdBu',fmt='.3f')
    figure(out,figs,'rules_modality_costs','候选结果｜EEG、HbO、HbR各自的代价',
           '均为各规则自身共同有效子集。左按trial→session→subject等权后开方；右采用owner候选表中共同trial等权的增量变化。N4的EEG坐标已改变，对应跨坐标差留空。GW参考同折W-only，其余参考W=0；不能据不同子集横向排名。',costs_draw,(20,7))
    def gw_truth_draw(fig):
        gw=next(r for r in rules if r['family']=='N7' and r['rule']=='GW')
        w=next(r for r in rules if r['family']=='N7' and r['rule']=='W_only')
        g=np.array(gw['synthetic_assessment']['mean_nrmse_change'])
        ref=np.array(w['synthetic_assessment']['mean_nrmse_change'])
        ax=fig.subplots()
        heat(ax,[g,g-ref],['GW − 固定基线','GW − 同折 W-only'],['r','EEG','HbO','HbR'],
             '独立匹配合成：ΔNRMSE（负值更好）',center=True,fmt='.5f')
    figure(out,figs,'n7_synthetic_ablation','N7｜合成消融的两个参照必须区分',
           'owner原合成筛选统一参考固定基线；GW和W-only均使用同一9个有效选择折、同一16个assessment生成trial。两条规则相减得到同输入消融：GW相对W-only有小幅改善，但仍差于固定基线。不能把144个频率加权比较当成144个独立trial。',gw_truth_draw,(13,4.5))


def plot_synthetic(out,figs,data):
    exported=[]
    for family in ['N1','N3','N5','N6','N7']:
        rows=[r for r in data[family] if r['kind']=='synthetic' and r['_task']['condition'] in CONDITIONS]
        names=list(dict.fromkeys(r['_task']['candidate']['id'] for r in rows))
        for stream in ['discovery','assessment']:
            panel=[r for r in rows if r['_task']['stream']==stream]
            # A complete facet for every fixed candidate, mode, condition and target.
            groups=[(name,mode) for name in names for mode in ['full','EEG_only','fNIRS_only']]
            for name,mode in groups:
                for condition in CONDITIONS:
                    rs=[r for r in panel if r['_task']['candidate']['id']==name and r['mode']==mode and r['_task']['condition']==condition]
                    for target in TARGETS:
                        exported.append(dict(family=family,stream=stream,candidate=name,mode=mode,condition=condition,target=target,
                                             expected=16,completed=sum(r['status']=='completed' for r in rs),
                                             nrmse=mean_metric(rs,'truth',target),coverage95=mean_metric(rs,'truth',target,'coverage95'),
                                             interval_width=mean_metric(rs,'truth',target,'mean_interval_width')))
            # Split the large N7 grid so labels and all candidate values remain readable.
            chunks=[names[i:i+8] for i in range(0,len(names),8)]
            for ci,chunk in enumerate(chunks):
                def draw(fig,chunk=chunk,panel=panel):
                    axes=fig.subplots(2,2)
                    labels=[candidate_label(n)+'/'+{'full':'全','EEG_only':'E','fNIRS_only':'N'}[mode] for n in chunk for mode in ['full','EEG_only','fNIRS_only']]
                    for target,ax in zip(TARGETS,axes.flat):
                        vals=[]
                        for name in chunk:
                            for mode in ['full','EEG_only','fNIRS_only']:
                                vals.append([mean_metric([r for r in panel if r['_task']['candidate']['id']==name and r['mode']==mode and r['_task']['condition']==condition],'truth',target) for condition in CONDITIONS])
                        heat(ax,vals,labels,CONDITION_NAMES,target+' NRMSE')
                figure(out,figs,f'synthetic_{family}_{stream}_{ci+1}',f'{family}｜{stream} 全部合成候选与单模态消融'+(f'（{ci+1}/{len(chunks)}）' if len(chunks)>1 else ''),
                       '每格预定 16 个独立 trial；图中为有效拟合的平均 NRMSE，失败率及其固定分母在配套 synthetic_metrics.csv、完成率图保留。E=仅 EEG，N=仅 fNIRS。独立配对条件的 r 真值只对应 EEG 驱动，禁止赋予两个模态共同生理真值。',draw,(16,max(7,2.3+len(chunk)*.88)))
    save_table(out,'synthetic_metrics',exported)
    # Coverage, width, failures and raw/smoothing controls for the baseline at every condition.
    baseline=[r for r in data['N1'] if r['kind']=='synthetic' and r['_task']['candidate']['id']=='baseline']
    def reference_draw(fig):
        axes=fig.subplots(2,3)
        for i,stream in enumerate(['discovery','assessment']):
            for j,target in enumerate(['r','clean_HbO','clean_HbR']):
                panel=[r for r in baseline if r['_task']['stream']==stream and r['mode']=='full']
                vals=[]
                for condition in CONDITIONS:
                    rs=[r for r in panel if r['_task']['condition']==condition]
                    vals.append([np.mean([nested(r,'reference_truth',method,target,'nrmse') for r in rs]) for method in ['raw_noisy','own_smoothing']]+[mean_metric(rs,'truth',target)])
                heat(axes[i,j],vals,CONDITION_NAMES,['原带噪','自身平滑','联合基线'],f'{stream} · {target}')
    figure(out,figs,'synthetic_references','公共合成｜联合模型与原带噪、自身平滑的比较',
           '每条件 16 个独立 trial。自身平滑为预设 Gaussian 平滑。r 的原带噪/平滑参照使用 EEG 坐标；clean EEG 与 r 数值相同。',reference_draw,(16,10))
    frame=pd.DataFrame(exported)
    for stream in ['discovery','assessment']:
        def uq_draw(fig,stream=stream):
            axes=fig.subplots(3,3)
            subset=frame[(frame.stream==stream)&(frame['mode']=='full')]
            identifiers=list(dict.fromkeys(zip(subset.family,subset.candidate)))
            for i,(metric,title) in enumerate([('coverage95','clean 95% 覆盖'),('interval_width','clean 区间宽度'),('completed','完成 / 16')]):
                for j,target in enumerate(['r','clean_HbO','clean_HbR']):
                    vals=[]
                    for fam,name in identifiers:
                        vals.append([float(subset[(subset.family==fam)&(subset.candidate==name)&(subset.condition==c)&(subset.target==target)][metric].iloc[0]) for c in CONDITIONS])
                    heat(axes[i,j],vals,[f'{fam}/{candidate_label(name)}' for fam,name in identifiers],CONDITION_NAMES,target+' · '+title,
                         annotate=False,vmin=0 if metric in ('coverage95','completed') else None,vmax=1 if metric=='coverage95' else 16 if metric=='completed' else None)
        figure(out,figs,f'synthetic_{stream}_uq',f'公共合成｜{stream} 全候选覆盖、宽度和分母',
               '完整输入的所有固定候选；对应逐格数值和三种输入模式均见 synthetic_metrics.csv。宽度为各自当前观测坐标单位，不能跨不同物理映射直接排序；灰色缺值不按零覆盖处理。',uq_draw,(20,25))


def plot_n7_truth(out,figs,run,data):
    conditions=['true_g','true_w','true_gw','measurement_gain']
    rows=[]
    for source in data['N7']:
        task=source['_task']
        if task['kind']=='synthetic' and task['condition'] in conditions:
            rows.append(dict(source,stream=task['stream'],condition=task['condition'],replicate=task['replicate'],candidate=task['candidate']))
    exported=[]
    for stream in ['discovery','assessment']:
        for cond in conditions:
            panel=[r for r in rows if r['stream']==stream and r['condition']==cond]
            names=list(dict.fromkeys(r['candidate']['id'] for r in panel))
            for name in names:
                for sign in [0,1]:
                    for mode in ['full','EEG_only','fNIRS_only']:
                        rs=[r for r in panel if r['candidate']['id']==name and r['replicate']%2==sign and r['mode']==mode]
                        assert len(rs)==8
                        record=dict(stream=stream,condition=cond,parity=sign,candidate=name,mode=mode,expected=8,completed=sum(r['status']=='completed' for r in rs))
                        for t in TARGETS:
                            for metric in ['nrmse','coverage95','mean_interval_width']:
                                record[t+'_'+metric]=mean_metric(rs,'truth',t,metric)
                        lls=[r['parameter_log_likelihood'] for r in rs if r['status']=='completed' and r.get('parameter_log_likelihood') is not None]
                        record['mean_log_likelihood']=float(np.mean(lls)) if len(lls)==8 else None
                        exported.append(record)
        def draw(fig,stream=stream,mode='full'):
            axes=fig.subplots(2,2)
            subset=[r for r in exported if r['stream']==stream and r['mode']==mode];names=list(dict.fromkeys(r['candidate'] for r in subset));cols=[(c,k) for c in conditions for k in [0,1]]
            for target,ax in zip(TARGETS,axes.flat):
                vals=[[next((r[target+'_nrmse'] for r in subset if r['candidate']==n and r['condition']==c and r['parity']==k),np.nan) for c,k in cols] for n in names]
                heat(ax,vals,[candidate_label(n) for n in names],[c.replace('measurement_gain','a_N')+('−' if k==0 else '+') for c,k in cols],target+' NRMSE',annotate=False)
        figure(out,figs,f'n7_truth_{stream}',f'N7｜{stream} 独立真值方向与观测增益替代',
               '每列 8 个独立 trial，负/正两列保留真实方向而不相互抵消。true_g=±0.3、true_w=±0.25、true_gw 同号组合；a_N=0.75/1.5。25 个 GW 候选（含基线）与 4 个独立观测增益对照共用同一输入。该图是固定候选响应，不用真值选择再宣称参数恢复。',draw,(17,19))
        for mode in ['EEG_only','fNIRS_only']:
            figure(out,figs,f'n7_truth_{stream}_{mode}',f'N7｜{stream} 真值方向：{mode}',
                   '与对应完整输入图使用相同的生成身份和29个固定候选；每列8个独立trial，只保留标题指定的输入模态。误差、覆盖、区间宽度和失败分母均可在n7_truth_metrics.csv按模式复核。',lambda fig,mode=mode:draw(fig,mode=mode),(17,19))
        def likelihood_draw(fig,stream=stream):
            axes=fig.subplots(1,3)
            cols=[(c,k) for c in conditions for k in [0,1]]
            for ax,mode in zip(axes,['full','EEG_only','fNIRS_only']):
                subset=[r for r in exported if r['stream']==stream and r['mode']==mode];names=list(dict.fromkeys(r['candidate'] for r in subset))
                lookup={(r['candidate'],r['condition'],r['parity']):r['mean_log_likelihood'] for r in subset}
                vals=[]
                for name in names:
                    line=[]
                    for c,k in cols:
                        x=lookup[name,c,k];ref=lookup['baseline',c,k]
                        line.append(x-ref if x is not None and ref is not None else np.nan)
                    vals.append(line)
                heat(ax,vals,[candidate_label(n) for n in names],[c.replace('measurement_gain','a_N')+('−' if k==0 else '+') for c,k in cols],mode+' · Δlog L',center=True,cmap='RdBu',annotate=False)
                if np.nanmax(np.abs(vals)) < 1e-8:
                    ax.text(.5,.03,'所有差异均小于 1e-8',ha='center',transform=ax.transAxes,
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
        figure(out,figs,f'n7_likelihood_{stream}',f'N7｜{stream} 真值方向的似然响应',
               '同一输入、同一模式内，相对G=W=0、a_N=1基线的平均log L差。每格必须8/8有效才着色；不同输入模式的可见维度不同，不比较绝对似然。似然响应并不构成参数后验、区间或恢复资格。',likelihood_draw,(22,17))
    save_table(out,'n7_truth_metrics',exported)


def plot_reader_summaries(out, figs, data, rules, selections, tasks, run):
    def n2_draw(fig):
        axes = fig.subplots(1, 2, sharey=True)
        for ax, variant in zip(axes, ['model', 'combined']):
            for i, solver in enumerate(['O0', 'O1', 'O2']):
                rows = [r for r in data['N2'] if r['_task']['law'] == 'nonlinear_gaussian'
                        and r['_task']['variant'] == variant and r['_task']['w'] == 0
                        and r['_task']['solver'] == solver and r['_task']['mode'] == 'full']
                assert len(rows) == 24 and all(r['status'] == 'completed' for r in rows)
                values = [mean_metric(rows, 'truth', t) for t in ['r', 'clean_HbO', 'clean_HbR']]
                bars = ax.bar(np.arange(3)+(i-1)*.24, values, .24, color=COLORS[i], label=solver)
                ax.bar_label(bars, fmt='%.3f', fontsize=7, rotation=90, padding=3)
            ax.set(xticks=np.arange(3), xticklabels=['r / EEG', 'HbO', 'HbR'],
                   title=variant+' · W=0 · n=24', ylabel='平均 NRMSE', ylim=(0, 1.25))
            ax.legend(ncol=3, fontsize=7, loc='upper left')
    figure(out, figs, 'reader_n2_comparison', 'N2｜观测时间处理的一致性带来主要改善',
           '非线性Gaussian生成、W=0、完整输入，同一24个独立合成trial。O0为逐点近似，O1为线性时序参考，O2为非线性相关Gaussian MAP；r与clean EEG误差相同。', n2_draw, (8.5, 3.4))

    def rules_draw(fig):
        axes = fig.subplots(1, 2, width_ratios=[1, 1.25])
        labels = [rule_label(r) for r in rules]
        values = [100*(1-r['common_success_only_risk']/r['common_success_only_baseline_risk']) for r in rules]
        axes[0].barh(labels, values, color=COLORS[0])
        for i, v in enumerate(values):
            axes[0].text(max(v, 0)+.4, i, f'{v:+.2f}', va='center', fontsize=7.5)
        axes[0].axvline(10, color=COLORS[1], ls='--')
        axes[0].set(xlim=(-3, 35), xlabel='共同有效子集风险下降（%）', title='实测：完整外折仅51–53/72')
        axes[0].invert_yaxis()
        panels = [r for r in rules if (r.get('synthetic_assessment') or {}).get('mean_nrmse_change') is not None]
        heat(axes[1], [r['synthetic_assessment']['mean_nrmse_change'] for r in panels],
             [rule_label(r) for r in panels], ['r', 'EEG', 'HbO', 'HbR'],
             '合成：相对固定基线 ΔNRMSE', center=True, fmt='.3f')
    figure(out, figs, 'reader_rules_summary', 'N3–N7｜实测改善与合成代价需同时判断',
           'N4实测使用固定fNIRS风险，GW实测参照同折W-only，其余参照W=0。合成均参照固定基线，来自16个assessment trial按9个选择折频率加权；不是144个独立trial。', rules_draw, (8.5, 3.8))

    def selection_draw(fig):
        axes = fig.subplots(1, 2, width_ratios=[1.3, 1])
        panel = [t for t in tasks if t['kind'] == 'select' and t['family'] == 'N7' and t['rule'] == 'GW']
        candidates = [selections[t['id']]['selected'] for t in panel if selections[t['id']]['status'] == 'completed']
        gs = [-.6, -.3, 0., .3, .6]; ws = [-.5, -.25, 0., .25, .5]
        values = [[sum(c['g'] == g and c['w'] == w for c in candidates) for w in ws] for g in gs]
        assert sum(map(sum, values)) == 9
        heat(axes[0], values, gs, ws, 'GW内折选择次数：9/12折有效', fmt='.0f', vmin=0, vmax=5)
        axes[0].set(xlabel='W', ylabel='G')
        records = read_csv(run/'N7/gw_session_surfaces.csv')
        complete = sum(r['surface']['complete'] for r in records)
        axes[1].barh(['全部session曲面', '完整曲面的最大值'], [complete, complete], color=COLORS[0])
        axes[1].barh(['全部session曲面', '完整曲面的最大值'], [9-complete, 0], left=[complete, complete], color='#D9DEE3')
        axes[1].text(2, 0, '4/9完整', ha='center', va='center', color='white', fontsize=8)
        axes[1].text(2, 1, '4/4双边界', ha='center', va='center', color='white', fontsize=8)
        axes[1].set(xlim=(0, 9), title='共同坐标诊断（非选择规则）', xlabel='session数；双边界 G=0.6 / W=−0.5')
        axes[1].invert_yaxis()
    figure(out, figs, 'reader_n7_selection', 'N7｜选择频率与描述性边界诊断',
           '左侧统计已定义GW规则的9个选择折；右侧统计三被试×三session的共同坐标曲面。两侧是不同分析层次，不能用右侧曲面最优点替代左侧训练折选择。', selection_draw, (8.5, 3.6))


def write_report(out,run,previous,figs,audit,summaries,rules,data,traces,scored):
    primary=read_json(run/'manifest.json');old=read_json(previous/'manifest.json')
    def localtime(s):return datetime.fromisoformat(s).astimezone().strftime('%Y-%m-%d %H:%M:%S')
    def seconds(m):return (datetime.fromisoformat(m['finished_at'])-datetime.fromisoformat(m['started_at'])).total_seconds()
    def image_block(name):
        if name not in CORE_FIGURES:
            return ''
        info=figs[name]
        number=CORE_FIGURES.index(name)+1
        return f"\n\n<!-- figure:{name} -->\n\n![{info['title']}](figures/{name}.png)\n\n**图 {number}：{info['title']}。**\n\n**内容：** {info['content']}\n\n**读法：** {info['reading']}\n\n**结论：** {info['conclusion']}\n\n<!-- /figure -->\n"
    def many(names):
        core=''.join(image_block(n) for n in names)
        details=[n for n in names if n not in CORE_FIGURES]
        if details:
            core+='\n\n详细图：'+ ' · '.join(f"[{figs[n]['label']}](#detail-{n})" for n in details)+'（可展开附录；完整图册也保留对应图注）。\n'
        return core
    complete=sum(s['status_counts'].get('completed',0) for s in summaries.values())
    actual=sum(s['actual_model_solves'] for s in summaries.values());planned=sum(s['planned_model_solves'] for s in summaries.values())
    gw=next(r for r in rules if r['family']=='N7' and r['rule']=='GW')
    ratio=100*(1-gw['common_success_only_risk']/gw['common_success_only_baseline_risk'])
    report=f'''# SSM 夜间 N1–N7 实验报告

报告日期：2026-09-10（北京时间）。本报告覆盖已冻结的 N1–N7 诊断合同及其公共准备、合成对照、失败追踪；不将更早的项目实验算作本轮新证据。

## 主要结论

两次运行均已完成固定任务队列。含 N7 的主运行有 **{audit['registered_tasks']:,} 个终态 cell，其中 {complete:,} 个 cell 标记 completed**；这不代表所有拟合成功。全部候选规则仅有 9/12 个选择折有效、51–53/72 个完整外折 trial，**没有规则达到本轮优先验证条件，也没有新增 teacher、参数可辨识或 UQ 资格**。

最清楚的正面证据来自 N2：在非线性 Gaussian + combined 的 24 个合成 trial、W=0、完整输入下，O2 相对 O0 的 r/HbO/HbR NRMSE 从 **0.855/1.007/0.941 降至 0.540/0.232/0.277**；O1 已取得大部分改善。这支持优先修正时间处理与噪声传播的一致性。O2 的实测数据分界尚未暴露，实测分支仍未实现；MAP 区间、参数后验和边际似然未估计。

新增 N7 中，GW 对同折 W-only 的共同有效子集风险从 **{gw['common_success_only_baseline_risk']:.6f} 降至 {gw['common_success_only_risk']:.6f}（{ratio:.2f}%）**，远低于预设 10%；分母为 53/72。4/9 个完整 G–W session 曲面全部落在 **G=0.6、W=-0.5** 的双边界，另外 5 个曲面不完整。尚无充分理由将 G–W 独立适配升级为新的 teacher 默认规则。

## 1. 完成情况、证据来源和评价口径

主证据：[`{run.name}`](../manifest.json)。旧运行：[`{previous.name}`](../../{previous.name}/manifest.json)。执行以两份 manifest 和固定任务/终态表为准，科学指标以各族 `trial_metrics.csv`、`summary.json` 和对应 cell 结果为准。本报告是这些保留证据的分析视图，不改写原始报告、冻结输入或 source snapshot。

'''
    report+=table(['运行','北京时间开始','结束','耗时','固定 cell','状态'],[
        ['原 N1–N6',localtime(old['started_at']),localtime(old['finished_at']),f'{seconds(old)/60:.2f} 分钟',old['task_count'],old['execution']],
        ['含 N7 主运行',localtime(primary['started_at']),localtime(primary['finished_at']),f'{seconds(primary)/60:.2f} 分钟',primary['task_count'],primary['execution']]])
    report+='\n\n两次都是 16-worker 的独立后台运行，均以 `all_registered_cells_terminal` 结束；没有用满 8 小时，没有超时或预算未启动 cell。重复部分采用同一批 trial/种子，**不得把两次 N1–N6 叠加为独立重复**。逐行核查身份、状态、残差、truth误差及似然：N1/N2/N3/N4/N6完全一致；N5身份与状态一致，数值最大绝对差5.82e-9（似然最大差1.67e-11），只见数值精度层面的差异。本报告统一采用含N7的主运行。\n\n'
    report+=table(['族','completed / 固定 cell','cell 非成功终态','记账实际 / 计划求解'],[[f,f"{s['status_counts'].get('completed',0):,}/{s['expected_cells']:,}",json.dumps({k:v for k,v in s['status_counts'].items() if k!='completed'},ensure_ascii=False),f"{s['actual_model_solves']:,}/{s['planned_model_solves']:,}"] for f,s in summaries.items()])
    report+=f'\n\n总计实际/计划求解记账为 {actual:,}/{planned:,}。计划含复用基线的名义拟合；actual 不计零成本复用，且某些工程对照/诊断不等于独立科学 trial。结果行还包含复用和汇总，三种分母不可混用。manifest 的 `completed_cells` 是最后一次心跳值（{primary.get("completed_cells")}），不是最终全部终态数；本报告按固定 task_table 对齐终态 ledger，{audit["terminal_tasks"]:,}/{audit["registered_tasks"]:,} 身份齐全、无重复，七族统计一致。\n\n'
    report+='实测仅 3 被试（01/09/18）×3 session（01/03/05）×8 个原训练 MA trial，共 **72 个唯一 trial**；原 MA 位置 4/9 在预处理前排除。4 外折×3 内折，每个 session 外折为 6 训练、2 评价；所有投影、EOG 回归、尺度与候选选择在对应训练折拟合。中心遮挡为 4 秒，预处理前遮挡。N5/N7 共同坐标只做描述，不能进入外折评分。本次报告生成没有读取原生信号、没有重跑模型、没有开放保护数据。\n\n'
    report+='主风险为 B=0.5·NMSE_EEG+0.25·NMSE_HbO+0.25·NMSE_HbR；N4 跨坐标主终点为固定 fNIRS 风险。测量图按 trial→session→subject 等权；图注另有说明的分布或成功子集仅作描述。合成 NRMSE 按对应 truth SD，实测按训练 SD；两者不可直接混比。灰格表示缺失/不适用，不按零填补。只有 3 被试，不报告人群置信区间、ICC 或显著性结论。\n'
    report+=many(['01_completion','02_failures'])
    report+='\n四个基线内折拟合未通过物理约束，分布在 S09 外折0（35/36有效）、S09 外折3（34/36）、S18 外折2（35/36）。每条选择规则的这 3 折均未定义，依赖的 18 个外折任务保留为 failed_contract；这些是相同失败源的传播，不是 7 组独立模型失败。外折内部的额外物理失败又进一步减少完整 trial 数。\n\n'
    report+='## 2. N1：共享状态、残差和对照\n\nN1 完成所有 588 个 cell，但保留 19 个失败结果行。固定 W=0 的 14 模式完整外折为 62/72；基线自身已不满足全分母比较要求。完整输入、整模态缺失、中心遮挡、自身/模板/配对/移位、同折线性基线、残差尾部、r 影响与时间相关性均在下图覆盖。原生 200 Hz EEG 电位无法由 log-power PCA 逆变换，本报告不把当前坐标差称为原生电位重建误差。\n'
    report+=many(['03_n1_residuals','04_n1_masks','05_n1_null_linear','06_n1_compromise','07_n1_acf','08_n1_tails','09_n1_influence','10_n1_hbt'])
    report+='\n## 3. N2：时间处理和推断近似\n\n3,168 个合成 solver cell 全部完成，1 个 native measured O2 cell 明确未实现。三种规律×两种处理×两个 W×三种 solver 使用相同输入；主 mask 每组 24 个 trial，整模态缺失每组 8 个固定 trial。\n\n'
    report+=table(['非线性 Gaussian / W=0 / full','O0 r/HbO/HbR','O1 r/HbO/HbR','O2 r/HbO/HbR'],[
        ['model','0.533 / 0.189 / 0.214','0.536 / 0.183 / 0.217','0.526 / 0.182 / 0.207'],
        ['combined','0.855 / 1.007 / 0.941','0.549 / 0.238 / 0.288','0.540 / 0.232 / 0.277']])
    report+='\n\ncombined 下 O2 相对 O0 的 r/HbO/HbR 误差降幅约为 36.9%/76.9%/70.6%；相对 O1 则仅约 1.7%/2.5%/4.0%。在未经过 combined 的匹配 model 条件，三者差异小。Student-t 的 model 条件下 O0 对三种目标的均值误差低于 O1/O2，说明 Gaussian MAP 不能泛化替代原 Student-t 近似。下列图表完整保留两种 W、全部输入模式及其条件。\n'
    report+=many(['reader_n2_comparison']+[f'n2_{law}_{suffix}' for law in ['linearized_gaussian','nonlinear_gaussian','nonlinear_student_t'] for suffix in ['means','hidden']])
    report+='\nO2 匹配非线性 Gaussian 均值预检为 24/24，r NRMSE 0.526、r 平均相关 0.848；通过探索预检不等于 teacher 或 UQ 资格。工程误差：线性均值对照 8.92e-13、导数 6.64e-12、隐藏干预 0、密度单位变换 1.42e-14，均低于冻结工程阈值。\n'
    report+='\n原自动摘要将共享原生噪声的带噪区间概括为未估计；逐trial记录和冻结实现显示**O1已实现该条件Gaussian计算，O0/O2未实现**。本报告依照细粒度owner结果区分这三者。O1可见目标的零条件方差与覆盖1来自已观测条件化，不能当作新观测预测能力。\n'
    report+=many(['n2_uncertainty','n2_shared_noisy','n2_solver_checks'])
    report+='\n## 4. N3：fNIRS 观测噪声权重\n\n噪声共同倍率 0.5/1/2/4；9 个已定义选择折全部保留基线倍率 1。外折 51/72，共同有效子集风险与同一基线完全相同（B=5.952917）。本轮没有得到仅改变共同噪声权重的改进规则。训练一阶差分尺度与实际采用噪声尺度、相关性和 200 次分块重采样如下；不可单凭放宽区间将较好覆盖称为均值预测改善。\n'
    report+=many(['n3_inner','n3_noise'])
    report+='\n## 5. N4：EEG 空间、眼动和频带/符号\n\n六条冻结分支：E0 全通道1–45Hz，E1训练EOG回归，E2排除额部，E3局部1–45Hz，E4局部8–13Hz正特征，E5对E4反号。9 个有效选择折选中 E2 五次、E4 一次、基线三次；外折 52/72，与基线共同 51/72。共同子集固定 fNIRS 风险 7.229241→7.226792，仅约0.034%改善，且配对/移位增量有退化。人工伪迹90/90完成，但没有空间 clean truth 生成器，不能据此证明神经真值恢复。\n'
    report+=many(['n4_inner','n4_artifact'])
    report+='\n## 6. N5：观测增益补偿和 session\n\n增益选择为 a_N=2 七次、1.5 一次、基线一次；外折53/72，共同基线子集51/72。描述性 B 从5.952917降到4.387177（26.30%），但独立匹配合成 r/EEG/HbO/HbR NRMSE 分别恶化0.102884/0.102884/0.026386/0.014590，且部分配对增量下降。因此不能只凭实测子集改善采用该规则。8/9条完整W曲线均触W=-0.5边界；只有3/9个完整W–增益曲面。\n'
    report+=many(['n5_inner','n5_w_curves','n5_surfaces'])
    report+='\n## 7. N6：状态转移残差、回放和流量越界\n\n9 个有效选择折中8次选择血流相关过程噪声×2，1次选择r过程噪声×0.5；外折53/72，共同子集风险下降12.99%，但匹配合成r/EEG/HbO误差增加，HbR小幅降低，仍未通过规则筛选。确定性r回放64/72完成；它是闭合诊断，不能当作共享/私有信息占比。\n'
    report+=many(['n6_inner','n6_replay','n6_transition','n6_failure_traces'])
    report+='\n'+table(['被试/旧训练索引','W','复现状态','首次f=0事件时间(s)','跳过最后更新后下一步正流量'],[
        [f"{t['subject']}/{t['trial']}",t['w'],r['status'],finite_number(r.get('first_zero_event_time_s'),6),
         next((str(v['drift']['in_domain']) for v in r.get('local_counterfactuals',[]) if v['kind']=='skip_last_update'),'不适用')]
        for t,r in traces])+'\n\n这些旧 prepared training index 与原 MA trial_position 不是同一个编号。5 个越界案例中仅2个可由跳过最后一次更新避免紧接的越界，另外3个之前状态已使下一步不安全；不能把所有失败归于最后一条观测。\n'
    report+='\n## 8. N7：独立 G–W、单方向消融与观测增益替代\n\n25个固定GW候选仅放开G和W，Z、tau、其余生理/噪声设置固定。映射为 `β=β_ref·exp(G+2W)`、`γ=γ_ref·exp(2W)`、`κ=κ_ref·exp(W)`；G=0原本锁定β/γ。GW/W-only/G-only共用同折投影与输入，分别冻结选择规则；G与观测增益从未同时放开。9,547个cell全部终态，其中9,484个completed、63个依赖合同失败；completed cell内还有必须保留的失败拟合。\n\n'
    report+=table(['规则','有效选择折','完整外折','实测参考','共同n','参考风险→规则风险','共同子集改善'],[
        [rule_label(r),f"{r['completed_selection_folds']}/12",f"{r['completed_outer_trials']}/72",r['comparison_reference'],r['common_success_trials'],f"{r['common_success_only_baseline_risk']:.6f} → {r['common_success_only_risk']:.6f}",f"{100*(1-r['common_success_only_risk']/r['common_success_only_baseline_risk']):.2f}%"]
        for r in rules if r['family']=='N7'])
    report+='\n\nGW 的选择为 G=0.3/W=0 两次，G=0/W=-0.5一次，G=0/W=-0.25一次，其余五次为基线；没有任何已定义选择折选中G与W同时非零。W-only对应−0.5两次、−0.25一次、基线六次；G-only对应G=0.3两次、基线七次。新增G没有形成强的外折证据。GW相对W-only的1.51%改善来自成功子集，完整72-trial结果未估计。\n'
    report+='\n合成主消融另需分清参照：owner筛选表对GW和W-only都与固定基线比较。按相同9折选择频率、同一16个assessment生成trial比较，GW相对W-only的r/EEG/HbO/HbR NRMSE分别下降0.005639/0.005639/0.003188/0.001405；幅度小，且两者均劣于固定基线。这一局部正面结果不弥补外折缺口和10%风险门槛。\n'
    report+=many(['reader_rules_summary','reader_n7_selection','n7_inner','rules_comparison','rules_modality_costs','n7_synthetic_ablation','n7_surfaces']+
                 [n for n in figs if n.startswith(('n7_truth_','n7_likelihood_'))])
    report+='\n4个完整G–W曲面全部取G=0.6/W=-0.5，近优网格只有该单个边界点；其teacher/r差为0是粗网格仅含一个近优点的直接结果，不是参数不确定性为0。额外true_g、true_w、true_gw、measurement_gain共128个独立生成trial（两种子流×4条件×16），各自比较25个GW和4个独立观测增益候选。响应图保留正负方向，未用truth重选实测规则，也不构成SBC。\n'
    report+='\n## 9. 全部公共合成结果与完整可视化附录\n\n两条独立种子流、六条件、每条件16个独立trial。匹配与失配压力条件分开显示。N1/N3/N5/N6/N7所有冻结候选、三种输入模式的误差均在以下分面图覆盖；N4单独使用原生伪迹检查，不冒用三坐标合成作为空间真值。每格有效数、覆盖率和宽度详见 [synthetic_metrics.csv](synthetic_metrics.csv)，新增真值方向详见 [n7_truth_metrics.csv](n7_truth_metrics.csv)。这些CSV是图表的可追溯派生视图，run中的逐trial表仍是证据owner。\n'
    report+=many(['synthetic_references']+[n for n in figs if n.startswith('synthetic_') and n not in ['synthetic_references']])
    report+='\n## 10. 结论与下一步建议\n\n本轮首先定位到时间处理/噪声传播合同问题；N2 的正面合成效应远大于新增G自由度的外折效应。优先推进既有O2实测特征层分界和正确噪声传播的实现，并针对已保留的物理失败身份检查均值/协方差近似与更新行为。\n\n保持现有所有负结果和分母；后续完整评价需要版本化解决基线失败，不能删除失败后再次排名。N3暂不扩大噪声倍率；N4没有足够的跨模态增量；N5/N6的实测子集改善需同时解决合成真值代价；N7暂不扩大GW网格，也不将边界重复解释为个人生理参数。任何新实测运行按其范围和合同另行定义，本次未启动额外实验。\n\n本次产物验证：固定任务与终态逐一对齐；冻结的task/config/scope/fold/preflight哈希匹配；候选完整trial与owner表一致；每幅图均从保存结果生成；HTML图片内嵌；SVG及PDF图册可导出。未把MAP缺失的区间填为零，未借用其他solver的方差。\n\n## 证据与复现入口\n\n'
    script_link=os.path.relpath(Path(__file__).resolve(),out)
    report+=table(['内容','入口'],[['原自动报告','[OVERNIGHT_REPORT.md](../OVERNIGHT_REPORT.md)'],['运行记录','[manifest.json](../manifest.json)'],['候选规则汇总','[candidate_table.csv](../candidate_table.csv)'],['源代码身份','[source_snapshot_identity.json](../source_snapshot_identity.json)'],['报告核查','[report_validation.json](report_validation.json)'],['PDF正文与完整图册','[REPORT.pdf](REPORT.pdf) / [FIGURES.pdf](FIGURES.pdf)'],['生成脚本',f'[render_ssm_overnight_report.py]({script_link})']])
    report+='\n\n云端提供实验结论、汇总数值和可视化；逐任务表、压缩逐任务表、任务状态流水与原生/准备数组仅保留在本地，不作为阅读报告的前提。图表重建需要本地保留证据，云端包不提供逐项输出审计。\n'
    report+='\n\n## 阅读详细图形\n\n正文保留14幅核心图，其余条件、候选和输入模式放在下方可展开的详细图形附录中。PDF正文采用连续排版；完整矢量图及各自caption见 [FIGURES.pdf](FIGURES.pdf)。图册按正文图1–14、附图A1起排序，正文图片可点击打开对应图册页。\n'
    # Replace references to former all-inline plots with the current reading path.
    report=report.replace('下列图表完整保留两种 W、全部输入模式及其条件。','详细图形附录完整保留两种W、全部输入模式及其条件。')
    report=report.replace('均在下图覆盖。','均在正文及详细图形附录覆盖。')
    report=report.replace('均在以下分面图覆盖；','均在详细图形附录的分面图覆盖；')
    narrative=report
    report+='\n\n## 详细图形附录\n\n'
    for name, info in figs.items():
        if name in CORE_FIGURES:
            continue
        report+=(f'\n<details id="detail-{name}">\n<summary>{info["label"]} · {html.escape(info["title"])}</summary>\n\n'
                 f'<figure><img src="figures/{name}.png" alt="{html.escape(info["title"])}">'
                 f'{caption_html(info)}</figure>\n</details>\n')
    (out/'REPORT.md').write_text(report)
    processor=markdown.Markdown(extensions=['tables','fenced_code','toc'])
    body=processor.convert(report)
    body='<details class="toc"><summary>报告目录</summary>'+processor.toc+'</details>'+body
    for name in CORE_FIGURES:
        info=figs[name]
        replacement=(f'<figure class="core-figure" id="figure-{name}">'
                     f'<img src="figures/{name}.png" alt="{html.escape(info["title"])}">'
                     f'{caption_html(info)}</figure>')
        body=re.sub(r'<!-- figure:'+re.escape(name)+r' -->.*?<!-- /figure -->',lambda m:replacement,body,flags=re.S)
    for name in figs:
        uri='data:image/png;base64,'+base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        body=body.replace(f'src="figures/{name}.png"',f'src="{uri}"')
    style='''body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#243444;background:#f2f5f7;margin:0;line-height:1.7}main{max-width:880px;background:white;margin:24px auto;padding:36px 48px}h1{font-size:30px;color:#164f65}h2{font-size:22px;border-top:1px solid #dbe5eb;padding-top:22px;margin-top:32px}p{font-size:15px}figure{margin:20px 0 26px;border:1px solid #dbe5eb;padding:14px;border-radius:4px}figure img{display:block;max-width:100%;height:auto;margin:0 auto}.core-figure img{max-height:450px}figcaption{margin-top:12px;padding-top:10px;border-top:1px solid #dbe5eb}figcaption p{font-size:13px;margin:5px 0;line-height:1.65}figcaption .figure-title{font-size:14px;font-weight:bold;color:#164f65}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.55;margin:18px 0;display:block;overflow-x:auto}td,th{padding:7px 9px;border:1px solid #d9e3ea;text-align:left}th{background:#edf3f6}a{color:#176b87}code{background:#edf3f6;padding:2px 4px;font-size:.85em;overflow-wrap:anywhere}details{border:1px solid #dbe5eb;margin:12px 0;padding:10px 14px}summary{cursor:pointer;font-weight:bold;color:#164f65}details figure{border:0;padding:0}@media(max-width:700px){main{padding:20px;margin:0}figure{padding:8px}h1{font-size:25px}}@media print{body{background:white}main{margin:0;padding:0;max-width:none}.toc{display:none}figure{break-inside:avoid}h2{break-after:avoid}table{display:table;font-size:9px}p{font-size:10px}}'''
    (out/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SSM N1–N7 实验报告</title><style>'+style+'</style><main>'+body+'</main></html>')
    compose_report_pdf(out, narrative, figs, audit)


def caption_html(info):
    return ('<figcaption><p class="figure-title">'+html.escape(info['label']+'：'+info['title'])+'</p>'+
            ''.join('<p><b>'+label+'：</b>'+html.escape(info[key])+'</p>'
                    for label,key in [('内容','content'),('读法','reading'),('结论','conclusion')])+'</figcaption>')


PDF_CSS = '''body{font-family:sans-serif;font-size:9.2pt;line-height:1.5;color:#243444;margin:0}
h1{font-size:22pt;color:#164f65;margin:0 0 12pt}h2{font-size:14pt;color:#164f65;margin:16pt 0 6pt;page-break-after:avoid}
p{margin:5pt 0}table{font-size:7.5pt;margin:8pt 0;border-collapse:collapse}td,th{padding:3pt;border:0.5pt solid #d8e1e8}th{background:#edf3f6}code{font-size:8pt}a{color:#176b87}
figcaption{font-size:8.3pt;line-height:1.4;text-align:left}figcaption p{margin:2pt 0;text-align:left}.figure-title{font-weight:bold;color:#164f65;font-size:9pt}'''


def compose_report_pdf(out, narrative, figs, audit, *, core_figures=None, label='SSM N1-N7'):
    """Flow text continuously; reserve a measured block for each image+caption."""
    core_figures = CORE_FIGURES if core_figures is None else core_figures
    media=fitz.paper_rect('a4');left=36;right=media.width-36;top=34;bottom=media.height-34
    output=io.BytesIO();writer=fitz.DocumentWriter(output)
    device=writer.begin_page(media);page_number=0;y=top
    positions=[]
    def new_page():
        nonlocal device,page_number,y
        writer.end_page();device=writer.begin_page(media);page_number+=1;y=top
        if page_number>80:raise RuntimeError('Report layout did not converge')
    def flow(content):
        nonlocal y
        if not content.strip():return
        markup=markdown.markdown(content,extensions=['tables','fenced_code'])
        for block in re.split(r'(<table>.*?</table>)',markup,flags=re.S):
            if not block.strip():continue
            if block.startswith('<table>'):
                probe=fitz.Story(block,user_css=PDF_CSS)
                more,filled=probe.place(fitz.Rect(left,top,right,bottom))
                if not more and filled[3]-top+8>bottom-y:new_page()
            story=fitz.Story(block,user_css=PDF_CSS)
            while True:
                if bottom-y<55:new_page()
                more,filled=story.place(fitz.Rect(left,y,right,bottom))
                story.draw(device)
                y=max(y,float(filled[3]))+5
                if not more:break
                new_page()
    pattern=r'<!-- figure:([^ ]+) -->.*?<!-- /figure -->'
    cursor=0
    for match in re.finditer(pattern,narrative,re.S):
        flow(narrative[cursor:match.start()]);cursor=match.end()
        name=match.group(1);info=figs[name]
        with fitz.open(out/'figures'/f'{name}.svg') as svg:
            aspect=svg[0].rect.height/svg[0].rect.width
        height=min(285,(right-left)*aspect)
        width=height/aspect
        cap=caption_html(info)
        probe=fitz.Story(cap,user_css=PDF_CSS)
        more,filled=probe.place(fitz.Rect(0,0,right-left,500))
        assert not more, ('Caption is too tall',name)
        caption_height=filled[3]+4
        total=height+caption_height+15
        assert total<bottom-top
        if bottom-y<total:new_page()
        rect=fitz.Rect((media.width-width)/2,y,(media.width+width)/2,y+height)
        caption_rect=fitz.Rect(left,y+height+7,right,y+height+7+caption_height)
        story=fitz.Story(cap,user_css=PDF_CSS)
        more,filled=story.place(caption_rect)
        assert not more,(name,caption_rect)
        story.draw(device)
        positions.append(dict(figure=name,report_page=page_number+1,image_rect=list(rect),caption_rect=list(caption_rect)))
        y+=total
    flow(narrative[cursor:]);writer.end_page();writer.close()
    pdf=fitz.open('pdf',output.getvalue())
    atlas_order=core_figures+[n for n in figs if n not in core_figures]
    for pos in positions:
        name=pos['figure'];p=pdf[pos['report_page']-1]
        with fitz.open(out/'figures'/f'{name}.svg') as svg:
            with fitz.open('pdf',svg.convert_to_pdf()) as source:
                p.show_pdf_page(fitz.Rect(pos['image_rect']),source,0)
        p.insert_link({'kind':fitz.LINK_GOTOR,'from':fitz.Rect(pos['image_rect']),'file':'FIGURES.pdf','page':atlas_order.index(name)})
    footer = 'SSM N1-N7 | 2026-09-10' if label == 'SSM N1-N7' else label
    title_label = 'SSM N1–N7' if label == 'SSM N1-N7' else label
    for i,p in enumerate(pdf):
        p.insert_text((left,media.height-17),f'{footer} | {i+1}/{len(pdf)}',fontsize=7,color=(.4,.46,.5))
    pdf.set_metadata({'title':title_label+' 实验报告 · 阅读版','author':'SSM experiment report'})
    audit['report_pdf_pages']=len(pdf);audit['report_figure_pages']=positions
    audit['core_figure_count']=len(positions)
    audit['pdf_continuous_a4_layout']=True
    audit['maximum_image_fraction_of_page']=max(fitz.Rect(p['image_rect']).height/media.height for p in positions)
    assert all(fitz.Rect(p['caption_rect']).y1<bottom for p in positions)
    assert {p['figure'] for p in positions}==set(core_figures)
    pdf.save(out/'REPORT.pdf',garbage=4,deflate=True);pdf.close()


def write_captioned_atlas(out, figs, audit, *, core_figures=None, label='SSM N1-N7'):
    book=fitz.open()
    core_figures = CORE_FIGURES if core_figures is None else core_figures
    order=core_figures+[n for n in figs if n not in core_figures]
    positions=[]
    for name in order:
        info=figs[name]
        with fitz.open(out/'figures'/f'{name}.svg') as svg:
            raw=fitz.open('pdf',svg.convert_to_pdf())
        width=max(700,raw[0].rect.width)
        height=raw[0].rect.height*width/raw[0].rect.width
        cap=caption_html(info)
        css=PDF_CSS+'figcaption{font-size:11pt;line-height:1.45}.figure-title{font-size:13pt}'
        probe=fitz.Story(cap,user_css=css)
        more,filled=probe.place(fitz.Rect(0,0,width-48,600))
        assert not more
        caption_height=filled[3]+14
        page=book.new_page(width=width,height=height+caption_height+48)
        page.show_pdf_page(fitz.Rect(0,0,width,height),raw,0);raw.close()
        # Story's own PDF retains searchable Chinese caption text.
        def rectfn(number,filled):
            area=fitz.Rect(0,0,width-48,caption_height)
            return area,area,None
        document=fitz.Story(cap,user_css=css).write_with_links(rectfn)
        assert len(document)==1,(name,len(document))
        page.show_pdf_page(fitz.Rect(24,height+12,width-24,height+12+caption_height),document,0);document.close()
        page.insert_text((24,page.rect.height-12),f'{label} | {len(book)}/{len(figs)}',fontsize=8,color=(.4,.46,.5))
        positions.append(dict(figure=name,label=info['label'],page=len(book)))
    title_label = 'SSM N1–N7' if label == 'SSM N1-N7' else label
    book.set_metadata({'title':title_label+' 完整图册与图注','author':'SSM experiment report'})
    book.save(out/'FIGURES.pdf',garbage=4,deflate=True);book.close()
    audit['atlas_figure_pages']=positions


def v3_hidden_support_metrics(truth, estimate, observed, variance=None):
    """Use each declared missing coordinate, normalized by full-trial truth SD.

    The retained fitter's hidden_truth field always used a 16-point center
    window. This separately named diagnostic covers whole-modality missingness
    without changing those frozen fields or their original normalization.
    """
    truth, estimate, observed = np.asarray(truth), np.asarray(estimate), np.asarray(observed, dtype=bool)
    if truth.shape != estimate.shape or truth.shape != (len(observed), 4) or observed.shape[1] != 3:
        raise ValueError('truth/estimate/observed shape mismatch')
    hidden = np.column_stack((~observed.all(axis=1), ~observed))
    result = {}
    for j, name in enumerate(TARGETS):
        mask = hidden[:, j]
        denominator = float(np.std(truth[:, j]))
        row = dict(hidden_samples=int(mask.sum()), full_trial_truth_sd=denominator if np.isfinite(denominator) else None)
        if mask.any() and denominator > 1e-12:
            error = estimate[mask, j]-truth[mask, j]
            row.update(nrmse=float(np.sqrt(np.mean(error**2))/denominator),
                bias_coordinate=float(error.mean()), bias_full_truth_sd=float(error.mean()/denominator))
            if variance is not None:
                spread = np.asarray(variance)[mask, j]
                if np.isfinite(spread).all() and np.all(spread >= 0):
                    half = 1.95996398454*np.sqrt(spread)
                    row.update(coverage95=float(np.mean(abs(error) <= half)), mean_interval_width=float(np.mean(2*half)))
        result[name] = row
    return result


def v3_failure_reason(task, result, row, results):
    if row is None:
        failed = [d for d in task.get('dependencies', []) if results[d]['status'] != 'completed']
        return 'dependency_failure' if failed else result['status']
    if row['status'] == 'completed':
        return 'completed'
    if row.get('failure_stage'):
        return row['failure_stage']
    if any(s.get('convergence_reason') == 'evaluation_budget' for s in row.get('starts', [])):
        return 'MAP_evaluation_budget'
    return row.get('failure_stage') or row['status']


def render_v3(run, out, previous=None):
    """A terminal-run report derived from the same fixed task/row evidence."""
    manifest = read_json(run/'manifest.json')
    if manifest['execution'] not in ('completed', 'stopped', 'stopped_budget'):
        raise ValueError('v3 detailed report requires a terminal controller')
    if out == run or not out.is_relative_to(run) or (out.exists() and any(out.iterdir())):
        raise ValueError('v3 report requires a fresh directory inside the run')
    tasks = [r['payload'] for r in read_csv(run/'task_table.csv')]
    by_id = {t['id']: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('duplicate registered task identity')
    results, hashes = {}, {}
    for task in tasks:
        path = run/'cells'/task['id']/'result.json'
        payload = path.read_bytes()
        results[task['id']] = canonical_residual_fields(json.loads(payload))
        hashes[str(path.relative_to(run))] = hashlib.sha256(payload).hexdigest()
    ledger = [json.loads(line) for line in (run/'case_status.jsonl').read_text().splitlines() if line.strip()]
    registered = [r for r in ledger if r['task_id'] in by_id]
    counts = Counter(r['task_id'] for r in registered)
    assert set(counts) == set(by_id) and set(counts.values()) == {1}, 'terminal ledger identities differ'
    assert all(r['status'] == results[r['task_id']]['status'] for r in registered), 'terminal status differs from owning cell'
    frozen = read_json(run/'frozen_input_identity.json')
    assert all(hashlib.sha256((run/p).read_bytes()).hexdigest() == digest for p, digest in frozen.items())
    scope = read_json(run/'scope_inventory.json')
    assert set(scope) == set(SUBJECTS) and sum(map(len, scope.values())) == 72
    assert all(r['original_ma_trial_position'] not in (4, 9) for values in scope.values() for r in values)
    summary = read_json(run/'summary.json')
    for family, stats in summary['families'].items():
        states = Counter(results[t['id']]['status'] for t in tasks if t['family'] == family)
        assert dict(states) == stats['status_counts'], 'summary is stale'
    out.mkdir(parents=True)
    (out/'figures').mkdir()
    (out/'renderer.py').write_bytes(Path(__file__).read_bytes())
    setup_style()
    fit_status, fit_rows, linear, costs = [], [], [], []
    for task in tasks:
        result = results[task['id']]
        if task['kind'] == 'v3_linear':
            linear.extend(dict(r, modality=task['modality']) for r in result.get('rows', []))
        rows = {r.get('row_id'): r for r in result.get('rows', [])}
        if task.get('row_ids') and result['status'] == 'completed':
            assert set(task['row_ids']) == set(rows), 'completed cell misses predeclared fit rows'
        if task.get('row_ids'):
            reused = bool(result.get('reused_task'))
            starts = [] if reused else [s for r in rows.values() for s in r.get('starts', [])]
            costs.append(dict(task_id=task['id'], family=task['family'], solver=task.get('solver'),
                fit_calls_lower_bound=0 if reused else max(result.get('actual_solves', 0), len(rows)),
                optimizer_starts=len(starts), optimizer_evaluations=sum(s.get('evaluations', 0) for s in starts),
                interrupted_call_possible=result['status'] == 'timeout', reused_task=result.get('reused_task'),
                elapsed_seconds=result.get('elapsed_seconds'), peak_rss_bytes=result.get('peak_rss_bytes')))
        for row_id in task.get('row_ids', []):
            row = rows.get(row_id)
            fit_status.append(dict(task_id=task['id'], row_id=row_id, family=task['family'], solver=task.get('solver'),
                role=task.get('role'), mode=row.get('mode') if row else row_id.removeprefix(str(task.get('trials', [''])[0])+'__'),
                source_status=row['status'] if row else result['status'],
                attribution=v3_failure_reason(task, result, row, results),
                reused_task=result.get('reused_task'), subject=task.get('subject')))
            if row is not None:
                fit_rows.append(dict(row, _task=task))
    assert len(fit_status) == sum(t['planned_solves'] for t in tasks), 'planned fit denominator changed'
    save_table(out, 'fit_status', fit_status)
    save_table(out, 'computation_cost', costs)
    audit = dict(run=run.name, execution=manifest['execution'], registered_tasks=len(tasks), terminal_tasks=len(registered),
        last_heartbeat_completed_cells=manifest.get('completed_cells'),
        heartbeat_count_matches_terminal_ledger=manifest.get('completed_cells') == len(registered),
        planned_fit_denominator=len(fit_status), terminal_ledger_matches_cells=True,
        fixed_metadata_sha256_verified=True, original_training_identities=72,
        result_sha256=hashes, renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        classification='rule rows include baseline reuse; missing preparation dependencies are not unavailable raw data')
    if (run/'continuation.json').exists():
        continuation = read_json(run/'continuation.json')
        retained = continuation['retained_result_sha256']
        assert all(hashes['cells/'+identifier+'/result.json'] == digest for identifier,digest in retained.items())
        assert all(hashlib.sha256((run/name).read_bytes()).hexdigest() == digest
                   for name,digest in continuation.get('retained_audit_sha256', {}).items())
        audit['retained_terminal_results_sha256_verified'] = len(retained)
        audit['original_campaign_budget_start_preserved'] = manifest['budget_started_at'] == continuation['budget_started_at']
        assert audit['original_campaign_budget_start_preserved']
    audit.update(fit_calls_lower_bound=sum(r['fit_calls_lower_bound'] for r in costs),
        optimizer_starts=sum(r['optimizer_starts'] for r in costs),
        optimizer_evaluations=sum(r['optimizer_evaluations'] for r in costs),
        interrupted_calls_possible=sum(r['interrupted_call_possible'] for r in costs))
    historical_mapping = []
    if previous is not None:
        failure_path = previous/'failure_attribution.csv'
        old_failures = [r for r in read_csv(failure_path) if
            (r['family'] == 'N1' and '__inner__' in r['task_id']) or
            (r['family'] == 'N6' and 'failure_trace' in r['task_id'])]
        audit['historical_failure_source'] = str(failure_path)
        audit['historical_failure_source_sha256'] = hashlib.sha256(failure_path.read_bytes()).hexdigest()
        audit['historical_failure_identities'] = len(old_failures)
        for old in old_failures:
            mode = old['row_id'].split('__', 1)[1] if old['family'] == 'N1' else 'full'
            for solver in ('O0', 'O1', 'O2'):
                matches = [r for r in fit_rows if r['_task']['family'] == 'S2' and r['solver'] == solver and
                           r['sample_id'] == old['sample_id'] and r['mode'] == mode]
                if len(matches) > 1:
                    raise ValueError('historical trial maps to multiple fixed outer rows')
                new = matches[0] if matches else None
                historical_mapping.append(dict(old_task_id=old['task_id'], old_row_id=old['row_id'],
                    sample_id=old['sample_id'], old_status=old['status'], new_solver=solver, new_mode=mode,
                    new_task_id=new['_task']['id'] if new else None,
                    new_status=new['status'] if new else 'preparation_dependency_unavailable',
                    new_failure_stage=new.get('failure_stage') if new else None,
                    input_relation='same trial, new feature contract and outer-fold context; not a same-input failure reproduction'))
        save_table(out, 'historical_failure_identity_mapping', historical_mapping)
    figures = {}
    def emit(name, title, content, reading, conclusion, draw, size=(11, 6)):
        desc = dict(title=title, content=content, reading=reading, conclusion=conclusion,
                    caption=' '.join([content, reading, conclusion]))
        figure(out, figures, name, title, content, draw, figsize=size, description=desc)
    measured_rules = [('S2', 'O0'), ('S2', 'O1'), ('S2', 'O2'), ('S3_measured', 'O2'), ('S4_measured', 'O0'), ('S4_measured', 'O2')]
    mode_counts = []
    for family, solver in measured_rules:
        rows = [r for r in fit_status if r['family'] == family and r['solver'] == solver and r['role'] != 'inner']
        mode_counts.append([sum(r['mode'] == mode and r['source_status'] == 'completed' for r in rows) for mode in OUTER_MASKS])
    emit('v3_completion', '实测模式完成数与固定分母',
        '每格分母均为72个预定身份；14模式完整性另由规则表判断。',
        '数字是成功路径数，灰色或零不等于零误差。输入准备、依赖及求解失败全部留在分母。',
        '任一规则存在缺口时，成功子集不能替代完整72-trial结果。',
        lambda f: heat(f.subplots(), mode_counts, [a+'/'+b for a,b in measured_rules], OUTER_MASKS,
                       vmin=0, vmax=72, fmt='.0f'), size=(13, 5))
    temporal = read_csv(run/'S1/solver_comparison.csv')
    for target in ('r', 'clean_HbR'):
        def draw(f, target=target):
            for ax, law in zip(f.subplots(1, 3, sharey=True), ['linearized_gaussian', 'nonlinear_gaussian', 'nonlinear_student_t']):
                for color, solver in zip(COLORS, ['O0', 'O1', 'O2', 'O2_pointwise', 'O2_mean_only']):
                    rows = [r for r in temporal if r['law'] == law and r['variant'] == 'combined' and r['solver'] == solver]
                    lookup = {r['mode']: r for r in rows}
                    ax.plot(range(5), [lookup[m].get(target+'_nrmse', np.nan) for m in MASKS], 'o-', color=color, label=solver)
                ax.set_xticks(range(5), ['全输入', 'EEG中心', 'fNIRS中心', '整段EEG', '整段fNIRS'], rotation=30)
                ax.set_title(law.replace('_', ' '), fontsize=9)
                ax.set_ylabel('平均真值 NRMSE');ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        emit('v3_temporal_'+target, '时间均值与相关噪声消融：'+target,
            '三种独立生成规律；每个条件24个120点trial，同trial的输入、真值与噪声实现共享。',
            '在同一Gaussian MAP中比较逐点、只修正时间均值和完整O2；O0/O1是额外参考。误差越低越好。',
            '组合处理收益须与缺失模式一起判断；Gaussian均值结果不提供Student-t或MAP区间资格。', draw, size=(12, 4))
    # Recompute explicitly named missing-support metrics from saved trajectories.
    # No optimizer or native reader is used by this report.
    snapshot = run/'source_snapshot'
    sys.path.insert(0, str(snapshot))
    from experiments import evaluate_ssm_overnight_diagnostics as frozen_suite
    if not Path(frozen_suite.__file__).resolve().is_relative_to(snapshot):
        raise ValueError('operator reconstruction requires the frozen run module in a fresh renderer process')
    cfg, dc, _, _, _ = frozen_suite.load_config(run/'resolved_config.yaml')
    operators = {variant: frozen_suite.repair.trajectory_operator(120, variant, dc) for variant in ('model', 'combined')}
    native = frozen_suite.v3_native_operators(120)
    support_rows, visible_rows = [], []
    computed = {}
    for row in fit_rows:
        task = row['_task']
        assessment = task['family'].endswith('_synthetic') and task.get('role') in ('baseline', 'selected', 'oracle')
        if (task['family'] != 'S1' and not assessment) or row['status'] != 'completed':
            continue
        path = (run/row['trajectory_path']).resolve()
        if not path.is_relative_to(run):
            raise ValueError('trajectory path leaves its owning run')
        variant = task.get('variant', 'native_feature')
        if (path, variant) not in computed:
            with np.load(path, allow_pickle=False) as archive:
                truth = archive['truth']
                estimate = np.column_stack((archive['r'], archive['canonical_clean_mean']))
                metrics = v3_hidden_support_metrics(truth, estimate, archive['observation_mask'],
                    archive['canonical_variance'] if 'canonical_variance' in archive.files else None)
                biases = {name: float(np.mean(estimate[:,j]-truth[:,j])) if np.isfinite(truth[:,j]).all() else None
                          for j,name in enumerate(TARGETS)}
                processed_truth = (operators[variant].apply(truth[:, 1:]) if variant != 'native_feature' else
                    np.column_stack((native['eeg']@truth[:, 1], native['fnirs']@native['native_interpolation']@truth[:, 2:])))
                prediction = archive['clean_mean']
                observed = archive['observation_mask']
                visible_metrics = {}
                for j, name in enumerate(MODS):
                    mask = observed[:, j]
                    denominator = float(np.std(processed_truth[:, j]))
                    value = dict(visible_samples=int(mask.sum()),
                        full_processed_truth_sd=denominator if np.isfinite(denominator) else None)
                    if mask.any() and denominator > 1e-12:
                        error = prediction[mask, j]-processed_truth[mask, j]
                        value.update(nrmse=float(np.sqrt(np.mean(error**2))/denominator), bias_coordinate=float(error.mean()))
                    visible_metrics[name] = value
                computed[(path, variant)] = (metrics, visible_metrics, biases)
        metrics, visible_metrics, biases = computed[(path, variant)]
        identity = dict(task_id=task['id'], family=task['family'],
            law=task.get('law', (task.get('condition') or {}).get('law')), variant=variant, mode=row['mode'],
            solver=task['solver'], replicate=task.get('replicate'), role=task.get('role'),
            condition=task.get('condition'), panel=task.get('panel'), sample_id=row.get('sample_id'))
        support_rows.append(dict(identity, metrics=metrics, full_time_canonical_bias_coordinate=biases))
        visible_rows.append(dict(identity, metrics=visible_metrics))
    save_table(out, 'missing_support_truth', support_rows)
    save_table(out, 'processed_visible_truth', visible_rows)
    audit['distinct_trajectory_files_for_support_metrics'] = len(computed)
    synthetic_truth = []
    for row in fit_rows:
        task = row['_task']
        if task['family'] == 'S1' or (task['family'].endswith('_synthetic') and task.get('role') in ('baseline','selected','oracle')):
            synthetic_truth.append(dict(task_id=task['id'], axis=task.get('axis'), condition=task.get('condition'),
                law=task.get('law'), variant=task.get('variant'), panel=task.get('panel'), role=task.get('role'),
                solver=task['solver'], mode=row['mode'], status=row['status'],
                full_time_truth=row.get('truth'), uncertainty='NOT_ESTIMATED' if task['solver'].startswith('O2') else 'conditional_approximation'))
    save_table(out, 'synthetic_state_truth', synthetic_truth)
    def draw_support(f):
        for ax, target in zip(f.subplots(1, 2), ['r', 'clean_HbR']):
            for color, solver in zip(COLORS, ['O0', 'O1', 'O2']):
                values = []
                for mode in MASKS[1:]:
                    rr = [r for r in support_rows if r['law'] == 'nonlinear_gaussian' and r['variant'] == 'combined' and r['solver'] == solver and r['mode'] == mode]
                    vv = [nested(r, 'metrics', target, 'nrmse') for r in rr]
                    values.append(float(np.nanmean(vv)) if np.isfinite(vv).any() else np.nan)
                ax.plot(range(4), values, 'o-', color=color, label=solver)
            ax.set_xticks(range(4), ['EEG中心', 'fNIRS中心', '整段EEG', '整段fNIRS'], rotation=25)
            ax.set_title(target);ax.set_ylabel('隐藏支持误差 / 完整trial真值SD');ax.grid(alpha=.2)
        ax.legend()
    emit('v3_missing_support', '按实际缺失支持检查状态恢复',
        '使用保存的观测mask重新计算缺失坐标；r采用任一模态缺失的时间并集。',
        '中心缺失为16点，整模态缺失为120点；分母统一为完整trial真值SD。无缺失的目标不绘制数值。',
        '该表与冻结行中固定16点窗口的hidden_truth字段分开，避免把整模态缺失误作中心窗口恢复。', draw_support, size=(10, 4))
    # Independent panels, never candidate frequencies, define Monte Carlo n.
    panels = read_csv(run/'synthetic_adaptation.csv')
    selection_summaries = []
    for axis in ('gain', 'process'):
        gate = summary['gates'].get('v3_'+axis+'_screen', {})
        conditions = [r['condition'] for r in gate.get('conditions', [])]
        if not conditions:
            continue
        def draw_adaptation(f, axis=axis, conditions=conditions):
            for ax, target_index in zip(f.subplots(1, 2), [0, 3]):
                for index, condition in enumerate(conditions):
                    rr = [r for r in panels if r['axis'] == axis and r['condition'] == condition and r['solver'] == 'O2']
                    differences = [r['mean_nrmse']['selected'][target_index]-r['mean_nrmse']['baseline'][target_index]
                        for r in rr if r['completed']['selected'] == r['completed']['baseline'] == 6]
                    for j, value in enumerate(differences):
                        ax.scatter(value, index+(j-1.5)*.08, color=COLORS[0], s=20)
                    if len(differences) == 4:
                        mean = float(np.mean(differences));error = 2.3533634348*np.std(differences, ddof=1)/2
                        ax.errorbar(mean, index, xerr=error, fmt='s', color='#222222', markersize=4, capsize=3)
                    ax.text(.99, index, f'{len(differences)}/4', ha='right', va='center', transform=ax.get_yaxis_transform(), fontsize=8)
                ax.axvline(0, color='#888888', linewidth=1);ax.axvline(.02, color=COLORS[1], linestyle='--')
                ax.set_yticks(range(len(conditions)), conditions);ax.invert_yaxis();ax.grid(axis='x', alpha=.2)
                ax.set_title(TARGETS[target_index]);ax.set_xlabel('所选规则 − 自身固定基线 NRMSE')
        emit('v3_adaptation_'+axis, '独立合成适配：'+axis,
            '每条件4个独立panel；每panel训练18、assessment 6，展示O2相对自身基线的变化。',
            '圆点是完整panel配对差，黑线给出df=3的两侧显示范围（端点各对应单侧95%界），虚线为+0.02容差。右侧是完整panel数。',
            '当前门槛判定：'+str(gate.get('verdict', gate.get('status')))+'；不完整或不确定不建立无害适配结论。',
            draw_adaptation, size=(12, max(4, len(conditions)*.4+1)))
        selections = {}
        for task in tasks:
            if task.get('axis') == axis and task['family'].endswith('_synthetic') and task.get('selection'):
                selection_id = task['selection']
                selection = results[selection_id]
                selections[selection_id] = dict(condition=task['condition']['id'], panel=task['panel'],
                    solver=task['solver'], status=selection['status'], selected=selection.get('selected'))
        save_table(out, axis+'_panel_selections', list(selections.values()))
        for condition in conditions:
            for solver in ('O0', 'O2') if axis == 'process' else ('O2',):
                rr = [r for r in selections.values() if r['condition'] == condition and r['solver'] == solver]
                key = 'gain' if axis == 'gain' else 'sigma_h'
                selection_summaries.append(dict(axis=axis, condition=condition, solver=solver, expected=4,
                    completed=sum(r['status'] == 'completed' for r in rr),
                    counts=dict(Counter(str(r['selected'][key]) for r in rr if r['status'] == 'completed'))))
        def draw_oracle(f, axis=axis, conditions=conditions, selections=selections):
            axes = f.subplots(1, 3)
            for ax, target_index in zip(axes[:2], [0, 3]):
                for j, condition in enumerate(conditions):
                    rr = [r for r in panels if r['axis'] == axis and r['condition'] == condition and r['solver'] == 'O2']
                    for k, role in enumerate(['baseline', 'selected']):
                        dd = [r['mean_nrmse'][role][target_index]-r['mean_nrmse']['oracle'][target_index]
                            for r in rr if r['completed'][role] == r['completed']['oracle'] == 6]
                        ax.scatter(dd, np.full(len(dd), j+(k-.5)*.18), color=COLORS[k], s=17,
                            label=role+' − oracle' if j == 0 else None)
                ax.axvline(0, color='#888888', linewidth=1);ax.set_yticks(range(len(conditions)), conditions)
                ax.invert_yaxis();ax.set_title(TARGETS[target_index]);ax.grid(axis='x', alpha=.2)
                ax.set_xlabel('配对panel的 NRMSE 差');ax.legend(fontsize=7)
            values = [.5, .75, 1, 1.5, 2] if axis == 'gain' else [.5, 1, 2]
            key = 'gain' if axis == 'gain' else 'sigma_h'
            counts = [[sum(r['condition'] == condition and r['solver'] == 'O2' and
                r['status'] == 'completed' and r['selected'][key] == value for r in selections.values())
                for value in values] for condition in conditions]
            heat(axes[2], counts, conditions, [str(v) for v in values], vmin=0, vmax=4, fmt='.0f', title='独立panel所选倍率（每行分母4）')
        emit('v3_oracle_selection_'+axis, '真值轴oracle与实际选择：'+axis,
            '左两图逐panel比较固定基线、所选规则与真值轴oracle；右图每个独立选择只计一次。',
            '负差表示该panel中误差低于oracle；oracle只知道被扫描的一个轴。选择数不足4时保留缺口。',
            '候选选择率与assessment误差分别读取，不能用六个assessment trial把一次选择计成六次重复。',
            draw_oracle, size=(15, max(4, len(conditions)*.4+1)))
    full = [r for r in fit_rows if r['_task']['family'] == 'S2' and r.get('mode') == 'full' and r['status'] == 'completed']
    whiten_cache, whiten_rows = {}, []
    for row in full:
        task = row['_task']; key = (task['subject'], task['outer'])
        if key not in whiten_cache:
            info, arrays = frozen_suite.load_projection(str(run), *key, None, 'E0')
            prefix = frozen_suite.projection_path(run, *key, None, 'E0')
            if hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest() != info['array_sha256']:
                raise ValueError('saved fold changed before residual whitening')
            identities = read_json(run/'prepared'/f'{task["subject"]}.json')['trials']
            view = frozen_suite.v3_view(cfg, arrays, info, identities, row['trial'], 'full')
            u, singular, _ = np.linalg.svd(view['noise_factor'], full_matrices=False)
            keep = singular > cfg['observation']['rank_rtol']*singular[0]
            u, singular = u[:, keep], singular[keep]
            whitening = (u/singular)@u.T
            projector = u@u.T
            # The symmetric whitening preserves the output clock. Baseline
            # constraints leave a projector, not a full-rank identity covariance.
            projected = whitening@view['noise_factor']
            error = float(np.max(np.abs(projected@projected.T-projector)))
            whiten_cache[key] = whitening, projector, int(keep.sum()), error
        whitening, projector, rank, error = whiten_cache[key]
        with np.load(run/row['trajectory_path'], allow_pickle=False) as saved:
            residual = saved['target']-saved['clean_mean']
        transformed = (whitening@residual.ravel()).reshape(residual.shape)
        lagged = {}
        for j, modality in enumerate(MODS):
            indexes = np.arange(j, residual.size, 3)
            def correlation(values):
                return float(np.corrcoef(values[:-1], values[1:])[0, 1]) if min(np.std(values[:-1]), np.std(values[1:])) > 1e-12 else None
            pairs = projector[indexes[:-1], indexes[1:]]/np.sqrt(projector[indexes[:-1], indexes[:-1]]*projector[indexes[1:], indexes[1:]])
            lagged[modality] = dict(raw_lag1=correlation(residual[:, j]), symmetric_whitened_lag1=correlation(transformed[:, j]),
                prefitted_noise_pair_correlation=float(np.mean(pairs)))
        whiten_rows.append(dict(task_id=task['id'], subject=row['subject'], session=row['session'], sample_id=row['sample_id'],
            solver=row['solver'], retained_rank=rank, projected_covariance_max_error=error, correlations=lagged,
            interpretation='symmetric whitening on retained noise support; prefitted reference excludes effects of fitting the trajectory'))
    save_table(out, 'full_residual_whitening', whiten_rows)
    audit['residual_whitening_covariance_max_error'] = max((r['projected_covariance_max_error'] for r in whiten_rows), default=None)
    def draw_whitening(f):
        values, labels = [], []
        for solver in ('O0', 'O1', 'O2'):
            rr = [r for r in whiten_rows if r['solver'] == solver]
            for modality in MODS:
                values.append([equal_mean(rr, lambda r, name=name: nested(r, 'correlations', modality, name))
                    for name in ('raw_lag1', 'symmetric_whitened_lag1', 'prefitted_noise_pair_correlation')])
                labels.append(f'{solver}/{modality} ({len(rr)}/72)')
        heat(f.subplots(), values, labels, ['原残差lag1', '对称白化后lag1', '拟合前噪声参考'], vmin=-1, vmax=1, fmt='.2f')
    emit('v3_residual_whitening', '共同噪声合同下的残差时间相关',
        '按冻结训练噪声做对称子空间白化，保留120点输出时钟；基线约束使参考协方差为投影矩阵。',
        '各solver仅汇总成功full行。第三列是噪声合同的相邻坐标相关参考，未扣除轨迹拟合带来的影响。',
        '该诊断描述剩余时间结构，不是白噪声检验p值；不同完成分母仍限制solver比较。', draw_whitening, size=(9, 6))
    def draw_residuals(f):
        for ax, metric, title in zip(f.subplots(1, 2), ['bias_sd', 'rmse_coordinate'], ['HbR bias / 训练SD', 'HbR绝对坐标RMSE']):
            values, counts = [], []
            for solver in ('O0', 'O1', 'O2'):
                rr = [r for r in full if r['solver'] == solver]
                values.append(equal_mean(rr, lambda r: nested(r, 'clean_map_residual', 'HbR', metric)))
                counts.append(len(rr))
            ax.bar(range(3), values, color=COLORS[:3]);ax.set_xticks(range(3), [f'{s}\n{n}/72' for s,n in zip(['O0','O1','O2'], counts)])
            ax.set_title(title);ax.axhline(0, color='#555555', linewidth=.7)
    emit('v3_residuals', '完整输入HbR残差的尺度与偏差',
        '使用clean-map残差，按trial→session→subject等权；标签保留各solver的完整输入成功分母。',
        '左图归一化偏差，右图绝对坐标RMSE。不同成功身份集合仅作条件描述，不作候选排名。',
        '同源带噪条件预测可以复现已见目标，仍须单独检查clean-map残差与训练尺度。', draw_residuals, size=(9, 4))
    def draw_tails(f):
        values, labels = [], []
        for solver in ('O0', 'O1', 'O2'):
            rr = [r for r in full if r['solver'] == solver]
            for modality in MODS:
                row = [equal_mean(rr, lambda r, q=q: r['clean_map_residual'][modality]['absolute_residual_sd_quantiles'][q]
                    if r.get('clean_map_residual', {}).get(modality) is not None else np.nan) for q in range(3)]
                values.append(row);labels.append(f'{solver}/{modality} ({len(rr)}/72)')
        heat(f.subplots(), values, labels, ['p50', 'p90', 'p95'], title='逐trial绝对残差 / 训练SD 分位数的等权平均')
    emit('v3_residual_tails', '完整输入残差尾部',
        '先在各trial内求分位数，再session和subject等权；各solver仅使用成功full行。',
        '比较同一行的p50、p90、p95；不同完成分母不能直接解释为求解器优劣。',
        '归一化尾部需与绝对坐标偏差共同阅读，缺失行没有被填为零。', draw_tails, size=(8, 6))
    null_rows = []
    for solver in ('O0', 'O1', 'O2'):
        by_trial = defaultdict(dict)
        for row in fit_rows:
            if row['_task']['family'] == 'S2' and row['solver'] == solver:
                by_trial[row.get('sample_id')][row['mode']] = row
        for direction, modalities in [('EEG', ['EEG']), ('fNIRS', ['HbO', 'HbR'])]:
            for null in ('own', 'template', 'pairing', 'shift'):
                for modality in modalities:
                    pairs = []
                    for modes in by_trial.values():
                        joint_row, null_row = modes.get('center_'+direction), modes.get('center_'+direction+'_'+null)
                        if joint_row and null_row and joint_row['status'] == null_row['status'] == 'completed':
                            left = nested(null_row, 'center_metrics', modality, 'nmse')
                            right = nested(joint_row, 'center_metrics', modality, 'nmse')
                            if np.isfinite(left) and np.isfinite(right):
                                pairs.append(dict(subject=joint_row['subject'], session=joint_row['session'], delta=left-right))
                    null_rows.append(dict(solver=solver, modality=modality, null=null, expected_trials=72,
                        paired_completed=len(pairs), mean_nmse_increment=equal_mean(pairs, lambda r: r['delta'])))
    save_table(out, 'mode_specific_null_increments', null_rows)
    center_rows = []
    for solver in ('O0', 'O1', 'O2'):
        for modality in MODS:
            mode = 'center_EEG' if modality == 'EEG' else 'center_fNIRS'
            rr = [r for r in fit_rows if r['_task']['family'] == 'S2' and r['solver'] == solver and
                  r['mode'] == mode and r['status'] == 'completed']
            value = equal_mean(rr, lambda r: nested(r, 'center_metrics', modality, 'nmse'))
            center_rows.append(dict(solver=solver, modality=modality, expected_trials=72, completed=len(rr),
                nmse=value, nrmse=float(np.sqrt(value)) if np.isfinite(value) else None,
                weighting='trial -> session equal -> subject equal; solver-specific successful subset'))
    save_table(out, 'center_prediction', center_rows)
    def draw_null(f):
        values, labels = [], []
        for solver in ('O0', 'O1', 'O2'):
            for modality in MODS:
                rr = [r for r in null_rows if r['solver'] == solver and r['modality'] == modality]
                values.append([r['mean_nmse_increment'] for r in rr])
                labels.append(solver+'/'+modality+' ['+','.join(str(r['paired_completed']) for r in rr)+']/72')
        heat(f.subplots(), values, labels, ['own', 'template', 'pairing', 'shift'], fmt='.2g',
             title='null − joint：训练方差归一化NMSE差')
    emit('v3_pairing_increments', '正确配对相对四类对照的单模式增量',
        '每列只配对该joint和null均成功的身份，再按session和subject等权；括号按列列出成功数。',
        '正值表示正确joint预测误差更小；每列固定分母72，成功子集可能不同。',
        '单模式增量保留可用结果，但不能补齐完整14模式主风险或支持跨solver排名。', draw_null, size=(11, 6))
    linear_rows = []
    for direction in ('EEG', 'fNIRS'):
        rr = [r for r in linear if r['modality'] == direction and r['status'] == 'completed']
        linear_rows.append(dict(direction=direction, completed=len(rr), expected=72,
            mean_nmse={kind: equal_mean(rr, lambda r, kind=kind: r['nmse'][kind]) for kind in ('basic','joint','pairing','shift')},
            increments={kind: equal_mean(rr, lambda r, kind=kind: r['increment'][kind]) for kind in ('basic','pairing','shift')}))
    save_table(out, 'linear_controls', linear_rows)
    replay_rows = [r for r in fit_rows if r.get('mode') == 'full' and isinstance(r.get('replay'), dict)]
    save_table(out, 'replay_summary', [dict(task_id=r['_task']['id'], subject=r.get('subject'), solver=r['solver'],
        role=r['_task'].get('role'), condition=r['_task'].get('condition'), replay=r['replay'],
        absolute_state_transition_residuals=r.get('state_transition_residual_rms'), standardized_state_transition_residuals=r.get('standardized_state_transition_residual_rms')) for r in replay_rows])
    def draw_replay(f):
        groups = [('实测固定O0', 'S2', 'O0', None), ('实测固定O2', 'S2', 'O2', None),
            ('合成匹配O0', 'S4_synthetic', 'O0', 'matched'), ('合成匹配O2', 'S4_synthetic', 'O2', 'matched'),
            ('合成错配O2', 'S4_synthetic', 'O2', 'independent_pairing')]
        for ax, metric, title in zip(f.subplots(1, 3), ['state_transition_residual_rms', 'standardized_state_transition_residual_rms', 'replay'],
                ['六状态转移残差绝对RMS', '六状态转移残差 / 各自Q标准差', '回放HbR差 / 训练SD']):
            vv, counts = [], []
            for label, family, solver, condition in groups:
                rr = [r for r in replay_rows if r['_task']['family'] == family and r['solver'] == solver and
                    (condition is None or (r['_task'].get('condition') or {}).get('id') == condition) and
                    r['_task'].get('role') in ('baseline', 'fixed') and r['status'] == 'completed']
                if metric == 'replay':
                    good = [r for r in rr if r['replay'].get('status') == 'completed']
                    vv.append(equal_mean(good, lambda r: r['replay']['gap_nrmse_training_sd'][2]))
                else:
                    good = [r for r in rr if isinstance(r.get(metric), list)]
                    vv.append([equal_mean(good, lambda r, j=j: r[metric][j]) for j in range(6)])
                counts.append(len(good))
            if metric == 'replay':
                ax.bar(range(len(groups)), vv, color=COLORS);ax.set_xticks(range(len(groups)),
                    [label+'\n'+str(n) for (label,*_),n in zip(groups,counts)], rotation=35, ha='right', fontsize=8)
                ax.set_title(title, fontsize=9)
            else:
                heat(ax, vv, [label+f' ({n})' for (label,*_),n in zip(groups,counts)], ['r','s','log f','log v','log p','log q'], title=title, fmt='.2g')
    emit('v3_process_replay', '状态转移残差与独立r驱动回放',
        '固定自身Q=1的完成子集；初态取同一拟合初态，回放不再加入过程噪声。数字是有效行数。',
        '依次比较绝对状态转移残差、按各自Q归一化的状态转移残差和回放HbR差；O0为后验均值，O2为MAP。',
        '回放差含有初态、动力学与非线性均值效应；不能解释为私有信息比例。', draw_replay, size=(12, 4))
    precision_path = run/'precision_audit_v1.json'
    if precision_path.exists():
        precision = read_json(precision_path)['rows']
        def draw_precision(f):
            ax = f.subplots()
            ax.plot(range(48), [max(r['legacy_error_training_sd']) for r in precision], 'o', markersize=3, label='冻结完整目标对齐误差')
            ax.plot(range(48), [max(r['promoted_baseline_arithmetic_error_training_sd']) for r in precision], '.', label='仅提升基线运算精度的审计')
            ax.axhline(1e-6, color=COLORS[1], linestyle='--', label='预定阈值')
            ax.set_xlabel('48个固定外/内投影折');ax.set_ylabel('最大绝对误差 / 训练SD');ax.legend(fontsize=8)
        emit('v3_precision', '完整目标重组的有限精度检查',
            '读取同一运行的算术审计；不涉及重新拟合或改变原始资格。',
            '高于虚线即超过预定1e-6阈值；第二组点只是单项精度变更的定位结果。',
            '只提升基线运算精度仍不能满足全部折的目标对齐，冻结失败保持有效。', draw_precision, size=(10, 4))
    replay_audit_paths = [run/'deterministic_replay_nonzero_driver_v1.json', run/'deterministic_replay_model_drift_audit_v1.json']
    if all(path.exists() for path in replay_audit_paths):
        replay_audits = [read_json(path)['rows'] for path in replay_audit_paths]
        def draw_replay_audit(f):
            ax = f.subplots()
            for values, label in zip(replay_audits, ['冻结的分段线性driver', '模型区间衰减的独立审计']):
                ax.semilogy(range(len(values)), [max(r['max_absolute_transformed_error']) for r in values], 'o-', label=label)
            ax.axhline(1e-5, color=COLORS[1], linestyle='--', label='确定性检查阈值')
            ax.set_xticks(range(4), [str(r['initial_r']) for r in replay_audits[0]])
            ax.set_xlabel('非零初始driver');ax.set_ylabel('120点最大变换状态误差');ax.legend(fontsize=8)
        emit('v3_replay_closure_audit', '非零driver下的确定性回放闭合',
            '同一零过程噪声轨迹，以独立DOP853核对不同采样间driver处理。',
            '按同一初始driver比较误差与1e-5阈值；修正审计不改变本轮冻结Q选择及回放结果。',
            '零driver通过不能外推到动态driver；原回放包含插值误差，需要与生理/状态转移残差分开。', draw_replay_audit, size=(10, 4))
    for i, info in enumerate(figures.values(), 1):
        info['label'] = f'图 {i}'
    qualifying_rules = [r['rule'] for r in summary['comparisons'] if r['measured_priority_screen_passed']]
    headline = ('本轮没有规则满足完整实测改进筛查。合成时间处理收益与实测推断稳定性必须分开判断。'
                if not qualifying_rules else '满足完整实测继续研究筛查的规则：'+', '.join(qualifying_rules)+'；仍不构成teacher资格。')
    s1_gate = summary['gates']['v3_S1_gate']
    numerical_notes = []
    if not audit['heartbeat_count_matches_terminal_ledger']:
        numerical_notes.append(f'冻结控制器结束时未刷新末次心跳计数（{manifest["completed_cells"]}）；'
            f'逐任务终态台账已核验为 {len(registered)}/{len(tasks)}，本报告使用台账计数。')
    rank_path = run/'rank_sensitivity_development_v1/summary.json'
    if rank_path.exists():
        rank_audit = read_json(rank_path)
        completed = sum(r['status'] == 'completed' for r in rank_audit['rows'])
        changes = [v for r in rank_audit['rows'] for v in r.get('rms_change_full_truth_sd', [])]
        audit['rank_development_summary_sha256'] = hashlib.sha256(rank_path.read_bytes()).hexdigest()
        numerical_notes.append(f'分离开发种子的SVD容差检查完成 {completed}/{rank_audit["fit_calls"]} 次拟合；'
            f'在1e-8/1e-10/1e-12下，相对主容差的最大轨迹RMS变化为 {max(changes):.6f} 个完整trial真值SD。'
            '它是数值敏感性检查，不增加主面板的独立重复数。'
            '[完整数值记录](../rank_sensitivity_development_v1/summary.json)。')
    combined = {r['solver']: r for r in temporal if r['law'] == 'nonlinear_gaussian' and
                r['variant'] == 'combined' and r['mode'] == 'full'}
    narrative = ['# SSM 观测合同 v3 实验报告', '',
        headline, '',
        f'运行 `{run.name}`；控制器状态 `{manifest["execution"]}`。证据来自冻结任务表和逐cell结果，报告不重跑拟合。', '',
        f'原坐标非线性Gaussian全输入预检完成 {s1_gate["completed"]}/{s1_gate["expected"]}：r NRMSE '
        f'{finite_number(s1_gate["mean_nrmse"]["r"])}，r相关 {finite_number(s1_gate["mean_r_correlation"])}。'
        '这只支持Gaussian均值预检，不提供Student-t或MAP区间资格。', '',
        table(['同一Gaussian MAP、组合处理全输入', 'r NRMSE', 'clean HbR NRMSE'],
            [[label, finite_number(combined[key]['r_nrmse']), finite_number(combined[key]['clean_HbR_nrmse'])]
             for key,label in [('O2_pointwise','逐点均值/对角噪声'), ('O2_mean_only','时间均值/对角噪声'), ('O2','时间均值/相关噪声')]]), '',
        *[line for paragraph in numerical_notes for line in (paragraph, '')],
        '## 数据边界与总体完成情况', '',
        '三名被试、三个session、72个原训练身份；原MA位置4/9及其他被试信号排除。缺失施加于非线性特征构造之后。MAP区间与teacher资格均未估计。', '',
        table(['阶段', 'completed cell / 预定', '主任务求解调用 / 最大规则行'], [[k,
            f'{v["status_counts"].get("completed",0)} / {v["expected_cells"]}',
            f'{v["actual_model_solves"]} / {v["planned_model_solves"]}'] for k,v in summary['families'].items()]), '',
        'completed cell可以包含失败拟合；复用自身基线的规则行不重复计作求解调用。软件pilot另列于原运行证据。', '',
        '## 主要判定与风险', '',
        table(['中心隐藏目标', '成功身份 / 72', 'NRMSE（自身成功子集）'],
            [[r['solver']+'/'+r['modality'], r['completed'], finite_number(r['nrmse'])] for r in center_rows]), '',
        table(['规则','共同完整身份 / 72','候选B','基线B','完整分母'], [[r['rule'],r['complete_common_trials'],
            finite_number(r['candidate_B']),finite_number(r['baseline_B']),r['full_denominators']] for r in summary['comparisons']]), '',
        '完整主风险要求72个身份及适配时12个选择折都完整。共同子集结果只描述对应身份，不填补失败分母；没有完整分母就不判为规则改进。', '',
        '## 合成资格与适配门槛', '']
    for name, gate in summary['gates'].items():
        narrative += [f'- `{name}`：状态 `{gate.get("status")}`，继续条件 `{gate.get("passed")}`，判定 `{gate.get("verdict", "Gaussian均值预检")}`。']
    narrative += ['', 'Q倍率的独立panel选择次数（每行固定分母4）：', '',
        table(['真值条件','solver','0.5','1','2','选择完成'],
            [[r['condition'],r['solver'],r['counts'].get('0.5',0),r['counts'].get('1.0',0),
              r['counts'].get('2.0',0),r['completed']] for r in selection_summaries if r['axis']=='process'])]
    narrative += ['', 'oracle仅知道所检验增益或Q轴的真值，其他G/W等生理参数仍固定。四个独立panel决定Monte Carlo样本量，候选频率和mask不是额外独立重复。', '', '## 图表与解释', '']
    if manifest.get('continued_from'):
        narrative.insert(6, '原控制器因派生报告的空值处理异常中断；本目录继承已完成结果并在原八小时起点下续跑，未重做完成拟合。'
                         '继承身份和原失败见 `continuation.json`。\n')
    for name, info in figures.items():
        narrative += [f'<!-- figure:{name} -->', f'![{info["title"]}](figures/{name}.png)',
            f'**{info["label"]}：{info["title"]}**  {info["caption"]}', '<!-- /figure -->', '']
    reasons = Counter(r['attribution'] for r in fit_status if r['source_status'] != 'completed')
    narrative += ['## 失败归因与下一步', '', table(['拟合行归因','预定行数'], sorted(reasons.items())), '',
        '输入重组、选择依赖、物理路径和求解收敛分别保留。首先处理未满足的数值/观测接口，再依据独立合成适配结果决定后续诊断；本报告不放宽冻结阈值或扩大候选网格。', '',
        '缺失支持表从保存的mask计算：整模态为120点、中心为16点，使用完整trial真值SD。它与原始固定中心窗口hidden_truth字段分开。', '',
        '[逐预定拟合状态](fit_status.csv) · [全时间真值及区间](synthetic_state_truth.csv) · [实际缺失支持真值指标](missing_support_truth.csv) · [处理后可见clean真值](processed_visible_truth.csv) · [中心预测](center_prediction.csv) · [逐模式null增量](mode_specific_null_increments.csv) · [残差白化](full_residual_whitening.csv) · [同折线性对照](linear_controls.csv) · [校验记录](report_validation.json)', '']
    if historical_mapping:
        narrative += ['旧四个内折失败及五个越界案例均按原owner身份映射到新合同结果；新输入及折内对象不同，不能称为同输入失败复现。'
            '详见[旧失败身份映射](historical_failure_identity_mapping.csv)。', '']
    narrative = '\n'.join(narrative)
    (out/'REPORT.md').write_text(narrative)
    body = markdown.markdown(narrative, extensions=['tables', 'fenced_code'])
    for name, info in figures.items():
        uri = 'data:image/png;base64,'+base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        replacement = f'<figure><img src="{uri}" alt="{html.escape(info["title"])}">{caption_html(info)}</figure>'
        body = re.sub(r'<!-- figure:'+re.escape(name)+r' -->.*?<!-- /figure -->', lambda m: replacement, body, flags=re.S)
    style = 'body{font-family:"Noto Sans CJK SC",sans-serif;max-width:1000px;margin:32px auto;line-height:1.7;color:#243444;padding:20px}img{max-width:100%}figure{margin:28px 0}td,th{border:1px solid #d8e1e8;padding:6px}table{border-collapse:collapse}figcaption{font-size:14px}'
    (out/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SSM 观测合同 v3</title><style>'+style+'</style>'+body+'</html>')
    write_captioned_atlas(out, figures, audit, core_figures=list(figures), label='SSM observation v3')
    compose_report_pdf(out, narrative, figures, audit, core_figures=list(figures), label='SSM observation v3')
    assert (out/'REPORT.html').read_text().count('src="data:image/png;base64,') == len(figures)
    assert all(hashlib.sha256((run/path).read_bytes()).hexdigest() == digest for path,digest in hashes.items())
    assert hashlib.sha256((out/'renderer.py').read_bytes()).hexdigest() == audit['renderer_sha256']
    audit.update(figures=figures, figure_count=len(figures), html_embedded_images=len(figures), cell_results_unchanged=True)
    (out/'report_validation.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('result_sha256','figures')}, ensure_ascii=False, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--previous-run-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run=args.run_dir.resolve();out=args.output_dir.resolve()
    if read_json(run/'manifest.json').get('schema') == 'ssm_overnight_v3':
        render_v3(run, out, args.previous_run_dir.resolve() if args.previous_run_dir else None)
        return
    if args.previous_run_dir is None:
        parser.error('--previous-run-dir is required for the retained N1-N7 report')
    previous=args.previous_run_dir.resolve()
    assert out != run and out != previous, 'Report must not replace retained evidence'
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    setup_style()
    tasks,by_id,ledger,summaries,data,selections,traces,audit=load_evidence(run,previous)
    rules=read_csv(run/'candidate_table.csv');scored=outer_scores(data,selections)
    for r in rules:
        assert len(scored[rule_label(r)])==r['completed_outer_trials']
    audit['candidate_outer_denominators_match']=True
    figs={}
    plot_completion(out,figs,summaries,ledger,rules,selections,tasks,data,scored)
    plot_n1(out,figs,run,data,scored)
    plot_n2(out,figs,data)
    plot_inner(out,figs,tasks,selections)
    plot_noise_artifact(out,figs,data)
    plot_surfaces(out,figs,run)
    plot_replay_trace(out,figs,data,traces)
    plot_rules(out,figs,rules,scored)
    plot_synthetic(out,figs,data)
    plot_n7_truth(out,figs,run,data)
    plot_reader_summaries(out,figs,data,rules,selections,tasks,run)
    detail_index=0
    for name,info in figs.items():
        if name in CORE_FIGURES:
            info['label']=f'图 {CORE_FIGURES.index(name)+1}'
        else:
            detail_index+=1;info['label']=f'附图 A{detail_index}'
    write_captioned_atlas(out,figs,audit)
    assert all(all(info.get(key) for key in ['content','reading','conclusion']) for info in figs.values())
    audit['figure_count']=len(figs);audit['figures']=figs
    audit['every_figure_has_content_reading_conclusion']=True
    audit['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_report(out,run,previous,figs,audit,summaries,rules,data,traces,scored)
    for name,identity in audit['result_table_sha256'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==identity, 'Result table changed while rendering'
    report=(out/'REPORT.md').read_text()
    links=re.findall(r'\]\(([^)]+)\)',report)
    unresolved=[link for link in links if not link.startswith(('http:','https:','#')) and not (out/link.split('#')[0]).exists()]
    # The audit is saved immediately after this check.
    unresolved=[link for link in unresolved if link!='report_validation.json']
    assert not unresolved, unresolved
    assert (out/'REPORT.html').read_text().count('src="data:image/png;base64,')==len(figs)
    audit['linked_files_exist']=True
    audit['html_embedded_figure_count']=len(figs)
    audit['result_tables_unchanged_during_render']=True
    (out/'report_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='figures'},ensure_ascii=False,indent=2))
    print('REPORT',out/'REPORT.html',flush=True)


if __name__=='__main__':
    main()
