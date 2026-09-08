#!/usr/bin/env python3
"""Render retained observation diagnostics and localize their numerical failures.

Adds derived review artifacts; never rewrites original case outcomes or sources.
Failure replay uses already prepared training arrays, with unchanged state and
observation updates. Substep checks are post-result numerical diagnostics only.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import evaluate_step5_observation_diagnostic as diagnostic

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def replay_failure(y, config, w, substeps=2):
    config = copy.deepcopy(config)
    config['model']['rk4_substeps'] = substeps
    core = diagnostic.step5.core
    extraction, transition = core._extraction, core.rk4_transition_with_jacobian
    captured = dict(transition_index=0, minimum_extraction_flow=None)

    def traced_extraction(flow, e0):
        old = captured['minimum_extraction_flow']
        captured['minimum_extraction_flow'] = float(flow) if old is None else min(old, float(flow))
        try:
            return extraction(flow, e0)
        except Exception:
            exponent = float(np.log1p(-e0)/flow)
            captured['extraction_failure'] = dict(flow=float(flow), E0=e0,
                log_one_minus_E=exponent, one_minus_E=float(np.exp(exponent)),
                computed_E=float(-np.expm1(exponent)),
                interpretation='finite positive flow but E rounds to one' if flow > 0 and -np.expm1(exponent) == 1. else 'other numerical/physical support error')
            raise

    def traced_transition(state, parameters, numerics):
        captured['transition_index'] += 1
        captured['pretransition_transformed_state'] = state.copy()
        return transition(state, parameters, numerics)

    try:
        with patch.object(core, '_extraction', traced_extraction), patch.object(core, 'rk4_transition_with_jacobian', traced_transition):
            trace = diagnostic.filter_trace(y, config, w, 13)
        return dict(execution='completed', substeps=substeps, parameter_log_likelihood=float(trace['increments'].sum()),
                    diagnostic=captured)
    except Exception as exc:
        return dict(execution='failed', substeps=substeps, error=repr(exc), diagnostic=captured,
                    event_relative_time_s=captured['transition_index']*config['model']['dt']-5.)


def failure_review(run_dir, cfg, base, summary):
    rows = []
    for subject in cfg['subjects']:
        detail = json.loads((run_dir/f'prepared_{subject}.json').read_text())
        with np.load(run_dir/f'prepared_{subject}.npz', allow_pickle=False) as archive:
            observations = archive['broadband_pca']
        if len(observations) != 24 or len(detail['trials']) != 24:
            raise ValueError('prepared training inventory changed')
        if any(r['subject'] != subject or r['original_ma_trial_position'] in (4, 9) for r in detail['trials']):
            raise ValueError('prepared input crosses original training boundary')
        config = copy.deepcopy(base)
        config['model']['observation_scale'] = summary['preparation'][subject]['broadband_pca_noise_scale']
        for w in cfg['curve']['fixed_w']:
            failures = []
            for i, y in enumerate(observations):
                masked = y.copy()
                masked[:, 0] = np.nan
                result = replay_failure(masked, config, w)
                if result['execution'] == 'failed':
                    failures.append(dict(trial_index=i, trial=detail['trials'][i], replay=result))
            first = None
            if failures:
                i = failures[0]['trial_index']
                y = observations[i].copy()
                y[:, 0] = np.nan
                first = dict(trial_index=i, substep_replays=[replay_failure(y, config, w, n) for n in (8, 32)])
            rows.append(dict(subject=subject, w=w, registered_trials=24, failed_trials=len(failures),
                             failures=failures, first_failure_substep_check=first))
    return dict(scope='post-result numerical replay of original training inputs; failures retain their original outcome',
                modality='fNIRS_only', coordinate='broadband_pca; fNIRS values identical across both coordinates', rows=rows)


def audited_curves(cfg, summary):
    result = {}
    for subject in cfg['subjects']:
        result[subject] = {}
        for coordinate in cfg['coordinates']:
            result[subject][coordinate] = {}
            for modality in cfg['curve']['modalities']:
                value = summary['curves'].get(subject, {}).get(coordinate, {}).get(modality)
                rows = [] if value is None else value['rows']
                span = float(np.ptp([r['parameter_log_likelihood'] for r in rows])) if rows else None
                complete = len(rows) == cfg['curve']['grid_points']
                result[subject][coordinate][modality] = dict(completed=len(rows), expected=cfg['curve']['grid_points'],
                    log_likelihood_span=span, interpretation='incomplete_curve' if not complete else ('flat_no_W_information' if span < 1e-8 else 'complete_diagnostic_curve'),
                    likelihood_maximum_w=(max(rows, key=lambda r: r['parameter_log_likelihood'])['w'] if complete and span >= 1e-8 else None))
    return result


def quadrature_review(summary):
    rows = []
    q = summary['quadrature_checks']
    for key, value in q.items():
        if not key.endswith('__17'):
            continue
        low = q[key[:-2]+'13']
        if value['execution'] == low['execution'] == 'completed':
            difference = value['result']['parameter_log_likelihood']-low['result']['parameter_log_likelihood']
            rows.append(dict(case=key.removesuffix('__17'), log_likelihood_17_minus_13=difference))
    return dict(expected=36, completed=len(rows), rows=rows,
                max_absolute_log_likelihood_difference=max(abs(r['log_likelihood_17_minus_13']) for r in rows),
                scope='first original training trial, fixed W endpoints; not full-curve or Gaussian-closure validation')


def save_figure(fig, run_dir, name):
    fig.savefig(run_dir/f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(run_dir/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def render_figures(run_dir, cfg, summary, review):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
    labels = ['Model', 'Baseline', 'fNIRS filter', 'Resampling', 'Common scale', 'Noise rule', 'Combined']
    bridge = [summary['bridge'][k] for k in cfg['bridge']['variants']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    for j, target in enumerate(('r', 'clean_HbO', 'clean_HbR')):
        axes[0].plot(range(7), [v['fixed_w']['0.0']['model_truth'][target]['coverage95'] for v in bridge], 'o-', label=target)
    axes[0].axhline(.95, ls='--', color='gray', lw=1)
    axes[0].set(title='Known model truth coverage (W=0)', ylabel='Nominal 95% coverage', ylim=(0, 1.03))
    axes[0].legend(fontsize=8)
    for target in ('clean_HbO', 'clean_HbR'):
        axes[1].plot(range(7), [v['fixed_w']['0.0']['processed_truth'][target]['coverage95'] for v in bridge], 'o-', label=target)
    axes[1].axhline(.95, ls='--', color='gray', lw=1)
    axes[1].set(title='Processed clean truth coverage', ylim=(0, 1.03))
    axes[1].legend(fontsize=8)
    means = np.array([v['boundary_minus_truth_ll']['mean'] for v in bridge])
    intervals = np.array([v['boundary_minus_truth_ll']['ci95'] for v in bridge])
    axes[2].errorbar(range(7), means, yerr=np.stack((means-intervals[:, 0], intervals[:, 1]-means)), fmt='o', capsize=4, color='#b75038')
    axes[2].axhline(0, color='gray', lw=1)
    axes[2].set(title='Preference for slower W', ylabel='log L(W=-0.5) - log L(W=0)')
    for ax in axes:
        ax.set_xticks(range(7), labels, rotation=40, ha='right')
    fig.suptitle('Controlled preprocessing bridge · 24 independent replicates · fixed W only', fontsize=15)
    save_figure(fig, run_dir, 'bridge_diagnostic')

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    colors = {'broadband_pca': '#286b91', 'local_F3_alpha': '#bf6532'}
    grid = np.linspace(*cfg['curve']['bounds'], cfg['curve']['grid_points'])
    for i, subject in enumerate(cfg['subjects']):
        for j, modality in enumerate(cfg['curve']['modalities']):
            ax = axes[i, j]
            for coordinate in cfg['coordinates']:
                value = summary['curves'].get(subject, {}).get(coordinate, {}).get(modality, {})
                values = {r['w']: r['parameter_log_likelihood'] for r in value.get('rows', [])}
                ys = np.array([values.get(w, np.nan) for w in grid])
                if values:
                    ys -= max(values.values())
                ax.plot(grid, ys, 'o-', ms=3, label=coordinate, color=colors[coordinate])
            incomplete = review['curves'][subject]['broadband_pca'][modality]['completed']
            ax.set(title=f'{subject} · {modality}\n{incomplete}/17 complete grid points', xlabel='W', ylabel='log L - maximum of valid points')
            if modality == 'EEG_only':
                ax.set_ylim(-1, 1)
                ax.text(.5, .3, 'Flat: EEG likelihood has no W information', ha='center', transform=ax.transAxes, fontsize=8)
            elif incomplete < 17:
                ax.text(.02, .08, 'Incomplete: no parameter optimum claim', transform=ax.transAxes, color='#9f3525', fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle('Original training trials · separate modality refits · no marginal-score substitution', fontsize=14)
    save_figure(fig, run_dir, 'w_likelihood_curves')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for j, subject in enumerate(cfg['subjects']):
        row = next(r for r in summary['curves'][subject]['broadband_pca']['joint']['rows'] if r['w'] == -.5)
        for label, v in row['standardized_innovations'].items():
            axes[0, j].plot(v['lag_seconds'], v['acf'], label=label)
            freq, psd = np.array(v['frequency_hz']), np.array(v['mean_psd'])
            axes[1, j].semilogy(freq[1:], psd[1:], label=label)
        axes[0, j].axhline(0, color='gray', lw=1)
        axes[0, j].set(title=f'{subject}: within-trial innovation ACF', xlabel='Lag (seconds)', ylim=(-.4, 1.05))
        axes[1, j].set(title='Standardized innovation spectrum', xlabel='Frequency (Hz)', ylabel='Mean PSD')
        axes[0, j].legend(fontsize=8)
    fig.suptitle('Broadband joint filter at W=-0.5 · one-step predictive residuals · trial resets respected', fontsize=14)
    save_figure(fig, run_dir, 'innovation_diagnostic')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for j, modality in enumerate(('EEG', 'fNIRS')):
        ax = axes[j]
        for shift, coordinate in zip((-.1, .1), cfg['coordinates']):
            rows = summary['linear'][coordinate][modality]['increments']
            controls = ('basic', 'independent_pairing', 'circular_shift')
            means = np.array([rows[k]['mean'] for k in controls])
            ci = np.array([rows[k]['ci95'] for k in controls])
            ax.errorbar(np.arange(3)+shift, means, yerr=np.stack((means-ci[:, 0], ci[:, 1]-means)),
                        fmt='o', capsize=4, label=coordinate, color=colors[coordinate])
        ax.axhline(0, color='gray', lw=1)
        ax.set_xticks(range(3), ['Own context + task', 'Independent pair', 'Half-trial shift'])
        ax.set(title=f'Masked {modality}', ylabel='Joint improvement in normalized negative MSE')
        ax.legend(fontsize=8)
    fig.suptitle('Low-capacity lag control · nested CV in 72 original training trials\nDescriptive 95% bootstrap intervals from only 3 subject clusters', fontsize=14)
    save_figure(fig, run_dir, 'linear_lag_diagnostic')


def number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def report(run_dir, cfg, summary, review):
    lines = ['# Step5 实测观测适配诊断（2026-09-08）', '',
        '**结论：在受限面板内，预处理与原观测模型的不一致已能复现慢 W 偏好和状态欠覆盖；实测的持续 HbR 偏差仍在，预定义局部 EEG 替换及低容量滞后对照没有提供稳定的配对增量。应先修正并验证观测合同，暂不扩大 GWZ 搜索或开展全面 UQ。**', '',
        '这是探索性诊断，不能把桥接机制直接归因为全部实测失败的唯一原因，也不证明不存在其他非线性跨模态关系。', '',
        '## 范围与完成情况', '',
        '- 模型保持六状态共享 Balloon、GWZ 原支持域与既有联合 Student-t 推断。旧 Step5 代码、结果、失败及 teacher 负判定保留。',
        '- 预选 subjects 01/09/18，sessions 01/03/05：72 个原训练 MA trial。原试次位置 4/9 在切片和预处理前排除；本次未处理或评分 Step5B 原留出 trial。',
        '- 原始 EEG 文件含同被试多个完整 session，fNIRS 缓存以完整记录存储；读取文件后只处理已声明训练窗口。未读取 subjects 19–29 文件。',
        '- 24 组合成桥接 × 7 处理分支 × 2 个固定 W = 336 次推断全部有完整结果。',
        '- 306 个实测模态曲线点中 264 个完成、42 个失败；72 个积分阶数检查任务与48 个嵌套 CV 折任务全部完成。42 个失败全部来自 fNIRS-only，两个 EEG 坐标重复使用相同 fNIRS，不能算作42个独立失败 trial。',
        '- 本次没有新被试/新 session 泛化检验、teacher 授权、tokenizer 训练、参数扩边、全面 UQ 或外部发布。', '',
        '## 1. 受控预处理合成桥接', '',
        '真值固定 W=0；同一组合成观测分别经过各处理。每个重复另有独立训练 trial 估计噪声。正的 Δlog L 表示 W=−0.5 优于 W=0。覆盖率首先对未变换模型真值计算，W 均固定为0。', '',
        '| 处理 | r覆盖95% | clean HbO覆盖95% | clean HbR覆盖95% | Δlog L均值 [95% CI] | 偏好−0.5 |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    names = dict(zip(cfg['bridge']['variants'], ['原模型坐标', '基线扣除', 'fNIRS滤波', '重采样往返', 'HbO/HbR共同×0.5', '训练噪声规则', '组合处理']))
    for variant, row in summary['bridge'].items():
        m = row['fixed_w']['0.0']['model_truth']
        ll = row['boundary_minus_truth_ll']
        lines.append(f"| {names[variant]} | {m['r']['coverage95']:.2%} | {m['clean_HbO']['coverage95']:.2%} | {m['clean_HbR']['coverage95']:.2%} | {ll['mean']:+.3f} [{ll['ci95'][0]:+.3f}, {ll['ci95'][1]:+.3f}] | {round(row['boundary_preference_fraction']*24)}/24 |")
    lines += ['', '基线扣除、滤波各自会破坏部分真值恢复，但在这个面板中，二者单独并未把总体似然偏好推到−0.5。共同幅度变化已使平均偏好转正，组合处理将偏好−0.5的比例从2/24提高到23/24。因此，**相对 EEG 的 fNIRS 幅度/观测映射，至少是一条无需改变血流动力学即可复现慢 W 偏好的机制**。这里的固定×0.5是受控诊断，不是从实测结果选出的校正系数。', '',
        '模型 clean truth 与处理后的 clean truth 不能混用。对处理后真值，基线扣除分支 HbO/HbR 覆盖为'
        f"{summary['bridge']['baseline']['fixed_w']['0.0']['processed_truth']['clean_HbO']['coverage95']:.2%}/"
        f"{summary['bridge']['baseline']['fixed_w']['0.0']['processed_truth']['clean_HbR']['coverage95']:.2%}；滤波分支为"
        f"{summary['bridge']['fnirs_filter']['fixed_w']['0.0']['processed_truth']['clean_HbO']['coverage95']:.2%}/"
        f"{summary['bridge']['fnirs_filter']['fixed_w']['0.0']['processed_truth']['clean_HbR']['coverage95']:.2%}。共同缩放分支处理后覆盖接近99.5%，却不代表模型潜状态正确；原模型真值覆盖仅约28%/31%。",
        '', '桥接仅隔离可控线性处理：滤波使用既有0.01–0.2 Hz三阶算子在4 Hz模型坐标执行，重采样为4→10→4往返；没有模拟原始宽频 EEG、完整10 Hz采集、OD/运动校正/MBLL。这些结果支持机制定位，不是完整原始实测生成模型的验证。24个固定W真值重复也不是参数先验SBC。', '',
        '![合成桥接](bridge_diagnostic.png)', '', '## 2. 谁在推动 W 下界', '',
        'EEG-only曲线在浮点误差范围内完全平坦：W不进入EEG的独立驱动分布，不能从EEG-only的数值并列最大点推断W。联合曲线在两种EEG坐标、三个被试上均偏好原范围下界。下表为全部24个训练trial的联合密度差；分段保留历史滤波上下文，不能解释为三个独立似然。', '',
        '| 被试 | EEG坐标 | 全部 Δlog L | 基线 | 任务 | 恢复 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for subject in cfg['subjects']:
        for coordinate in cfg['coordinates']:
            row = summary['curves'][subject][coordinate]['joint']
            seg = row['segment_boundary_minus_zero_ll']
            lines.append(f"| {subject} | {coordinate} | {row['boundary_minus_zero_ll']:+.2f} | {seg['baseline']:+.2f} | {seg['task']:+.2f} | {seg['recovery']:+.2f} |")
    lines += ['', '慢W收益主要来自任务和恢复段，基线段贡献均为负。局部F3 alpha坐标降低了部分似然差，但没有改变偏好方向；不能宣称它已经建立神经血管对应关系。训练选择的fNIRS对分别是AF3Fp1、C5CP5、AF7Fp1，F3与全部这些局部观测的解剖对应尚未验证。', '',
        'fNIRS-only：subject_01的17个点全部遇到至少一个trial失败；subject_09的最低4个W点失败，其余13点完成；subject_18的17点全部完成，独立fNIRS也偏好−0.5（Δlog L=+93.861）。因此**fNIRS自身可以推动慢响应，但当前失败曲线使我们无法把所有人的边界偏好唯一分解为单模态来源或联合冲突**。不能把subject_09剩余曲线的−0.25称为完整支持域的最优点。', '',
        f"预先指定第一训练trial上的36组13/17阶积分比较全部完成，最大绝对log-likelihood差为{review['quadrature']['max_absolute_log_likelihood_difference']:.6f}。这支持这些短例的积分稳定，不验证所有trial、完整W后验分辨率或高斯闭合近似。", '',
        '![W曲线](w_likelihood_curves.png)', '', '## 3. 创新残差揭示持续偏差与时间相关', '',
        '以下是宽频PCA、联合滤波、W=−0.5的一步预测标准化残差。ACF在trial内去均值计算，不跨独立重置拼接。低频比例为非零频率至0.2 Hz的PSD占比。', '',
        '| 被试 | 通道 | 均值 | RMS | 0.25s ACF | 低频功率比例 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for subject in cfg['subjects']:
        r = next(r for r in summary['curves'][subject]['broadband_pca']['joint']['rows'] if r['w'] == -.5)
        for label, value in r['standardized_innovations'].items():
            lines.append(f"| {subject} | {label} | {value['mean']:+.3f} | {value['rms']:.3f} | {value['acf'][1]:.3f} | {value['low_frequency_power_fraction']:.1%} |")
    lines += ['', 'HbR出现明显负偏差、较大标准化误差和近乎连续的低频残差；换成F3 alpha或固定W=0并未消除。这支持优先检查符号/幅度、基线算子和有色误差合同。残差结构本身不能在这些机制之间给出唯一因果归因，放大独立噪声也不能自动恢复配对特异信息。', '',
        '![创新残差](innovation_diagnostic.png)', '', '## 4. 低容量跨模态滞后对照', '',
        '每个坐标和方向均有72个原训练trial的外折预测，四外折/三内折按session内训练trial序号固定分配；PCA、fNIRS通道、尺度和岭强度都在相应训练折拟合。基础模型含目标自身可见端点与独立训练任务模板；联合模型再加六个固定滞后。配对和移位null只在外折改变另一模态。EEG重建可用未来fNIRS，所以是离线遮挡重建。', '',
        '数值为联合相对各对照的负MSE增量，按外折训练方差归一化；正值有利。区间以三个被试为cluster，仅作描述性提示，与旧Step5B的log-score数值不可直接比较。', '',
        '| 坐标 | 目标 | 对照 | 增量均值 [95% CI] |', '| --- | --- | --- | --- |']
    for coordinate in cfg['coordinates']:
        for modality in ('EEG', 'fNIRS'):
            for control, v in summary['linear'][coordinate][modality]['increments'].items():
                lines.append(f"| {coordinate} | {modality} | {control} | {v['mean']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] |")
    lines += ['', '两种坐标都没有在两个方向同时、稳定优于自身上下文＋任务模板、错配与移位。宽频EEG方向虽然优于半trial移位，仍未优于基础/错配对照；宽频fNIRS方向甚至不如错配。不能挑单个正null差作为共享信息证据，也不能由三个被试的负结果断言所有坐标均无跨模态信息。', '',
        '![线性滞后对照](linear_lag_diagnostic.png)', '', '## 5. 保留的数值域失败及同输入复核', '',
        '本节是看到fNIRS-only失败后，对相同已准备训练输入做的定位；未用于替换原曲线或改变判定。先在W=0/−0.5逐trial重放，再对每个有失败的被试/设定取第一个失败trial比较2/8/32子步。', '',
        '| 被试 | W | 失败trial/24 | 首次失败标识 | 2子步失败时间 |', '| --- | ---: | ---: | --- | ---: |']
    for r in review['numerical_failures']['rows']:
        first = r['failures'][0] if r['failures'] else None
        label = '—' if first is None else f"{first['trial']['session']}/MA位置{first['trial']['original_ma_trial_position']}"
        when = '—' if first is None else f"{first['replay']['event_relative_time_s']:.2f}s"
        lines.append(f"| {r['subject']} | {r['w']:.1f} | {r['failed_trials']}/24 | {label} | {when} |")
    lines += ['', '具体流量、`log(1−E)`、浮点E值、失败前状态及8/32子步结果保留在 [diagnostic_review.json](diagnostic_review.json)。严格E<1检查的触发点与上游推断为何走到极小流量必须分开解释；数值重放不构成真实生理异常证据。', '',
        '## 决策与可复现入口', '',
        '本轮已完成指定诊断及失败记录。证据支持下一步先建立与baseline/filter/相对幅度一致的观测前向算子及噪声合同，在相同真值桥接上验证；随后才在训练边界内复核固定/单参数SSM的配对增量。当前不扩大GWZ范围、不把内部参数或区间膨胀当修复、不晋级teacher和全面UQ。局部F3替换未给出继续沿该坐标搜索的积极依据。', '',
        '运行：', '', '```bash',
        '.venv/bin/python experiments/evaluate_step5_observation_diagnostic.py \\\n+  --run-dir experiments/runs/physiology_semantic_tokenizer/step5/<fresh-run> --stage all',
        '.venv/bin/python experiments/scripts/review_step5_observation_diagnostic.py \\\n+  --run-dir experiments/runs/physiology_semantic_tokenizer/step5/<fresh-run>',
        '```', '',
        '冻结配置见 [resolved_config.yaml](resolved_config.yaml)，原运行源码见 [runner_snapshot.py](runner_snapshot.py)，版本/源文件SHA见 [manifest.json](manifest.json)。原 [summary.json](summary.json) 与所有case JSON保留；本复核将平坦曲线的浮点并列最大标记为“无W信息”，将不完整曲线明确标记为“无完整最优点主张”，不改动原似然值。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg, base, _, _ = diagnostic.load_config(run_dir/'resolved_config.yaml')
    if run_dir.parent != (ROOT/base['output_root']).resolve():
        raise ValueError('review is restricted to the existing Step5 root')
    manifest = json.loads((run_dir/'manifest.json').read_text())
    if manifest['execution'] != 'completed' or manifest['stage'] != 'all' or manifest['subjects'] != cfg['subjects']:
        raise ValueError('a completed diagnostic run in the exact subject scope is required')
    for source, expected in manifest['source_sha256'].items():
        if diagnostic.digest(ROOT/source) != expected:
            raise ValueError(f'replay source differs from frozen run: {source}')
    summary = json.loads((run_dir/'summary.json').read_text())
    review_path = run_dir/'diagnostic_review.json'
    if review_path.exists():
        raise ValueError('review already exists; do not overwrite retained numerical evidence')
    review = dict(source_summary_sha256=diagnostic.digest(run_dir/'summary.json'),
        reviewer_sha256=diagnostic.digest(__file__), curves=audited_curves(cfg, summary),
        quadrature=quadrature_review(summary), numerical_failures=failure_review(run_dir, cfg, base, summary))
    diagnostic.step5.write_json(review_path, review)
    render_figures(run_dir, cfg, summary, review)
    (run_dir/'summary.md').write_text(report(run_dir, cfg, summary, review))
    print(run_dir/'summary.md')


if __name__ == '__main__':
    main()
