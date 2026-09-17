#!/usr/bin/env python3
"""Read the frozen A--E task table; export analysis and bitmap-figure reports."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

MODES = ['full', 'EEG_only', 'fNIRS_only', 'all_missing', 'center_EEG', 'center_EEG_own',
         'center_EEG_template', 'center_EEG_pairing', 'center_EEG_shift', 'center_fNIRS',
         'center_fNIRS_own', 'center_fNIRS_template', 'center_fNIRS_pairing', 'center_fNIRS_shift']
MODS = ['EEG', 'HbO', 'HbR']


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def mean(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def equal_subject(rows, field):
    sessions = defaultdict(list)
    for row in rows:
        if row.get(field) is not None and np.isfinite(row[field]):
            sessions[(row['subject'], row['session'])].append(row[field])
    subjects = defaultdict(list)
    for (subject, _), values in sessions.items():
        subjects[subject].append(mean(values))
    return mean([mean(values) for values in subjects.values()])


def metric(row, field, coordinate, key='nmse'):
    return (row.get(field, {}).get(coordinate) or {}).get(key)


def score(row, direction):
    cols = ['EEG'] if direction == 'EEG' else ['HbO', 'HbR']
    values = [metric(row, 'center_metrics', c) for c in cols]
    return mean(values) if all(v is not None for v in values) else None


def analyze(run):
    csv.field_size_limit(20_000_000)
    with (run/'task_table.csv').open() as stream:
        tasks = [json.loads(r['payload']) for r in csv.DictReader(stream)]
    rows, failures, cell_failures, starts = [], [], [], []
    projection_cache = {}
    for task in tasks:
        result = read(run/'cells'/task['id']/'result.json')
        if result['status'] != 'completed':
            cell_failures.append(dict(task_id=task['id'], family=task['family'], status=result['status'],
                planned_solves=task['planned_solves'], reason=result.get('reason', result.get('error'))))
        for row in result.get('rows', []):
            row = dict(row, family=task['family'], task_id=task['id'], mapping=task.get('mapping'),
                       w=task.get('candidate', {}).get('w'), outer=task.get('outer'), axis=task.get('axis'),
                       modality=task.get('modality'))
            if row.get('solver') == 'O2' and row.get('objective') is not None and row.get('whitened_residual_norm') is not None:
                row.setdefault('observation_cost', row['whitened_residual_norm']**2/2)
                row.setdefault('state_prior_cost', row['objective']-row['observation_cost'])
            if row.get('subject') and row['outer'] is not None and task['family'] in ('B', 'C'):
                key = (row['subject'], row['outer'])
                if key not in projection_cache:
                    projection_cache[key] = read(run/'prepared'/f'{key[0]}_o{key[1]}_E0.json')['projection']
                row['inverse_hb_factor'] = 1/projection_cache[key]['fnirs_factor']
            rows.append(row)
            for attempt in row.get('starts', []):
                starts.append(dict(task_id=task['id'], family=task['family'], mapping=row['mapping'], w=row['w'],
                    mode=row.get('mode'), fit_status=row['status'], start=attempt['start'], status=attempt['status'],
                    reason=attempt.get('convergence_reason'), evaluations=attempt.get('evaluations'),
                    gradient_inf=attempt.get('last_evaluated_gradient_inf_norm'),
                    initial_flow_domain_exit='flow_domain_exit' in attempt.get('error', ''),
                    rejected_domain_steps=sum(x.get('kind') == 'domain_trial_step' for x in attempt.get('rejected_steps', []))))
            if row['status'] != 'completed':
                categories = [s.get('failure_category', s.get('status')) for s in row.get('starts', [])]
                failures.append(dict(task_id=task['id'], mode=row.get('mode'), status=row['status'],
                    mapping=row['mapping'], w=row['w'], solver=row.get('solver'), categories=categories,
                    physical_stage=row.get('failure_stage'), error=row.get('error')))
    measured = [r for r in rows if r['family'] in ('B', 'C')]
    groups = defaultdict(list)
    for row in measured:
        groups[(row['mapping'], row['w'], row['solver'])].append(row)
    mode_table, panel_table, increments, replay, modality_conflicts = [], [], [], [], []
    for (mapping, w, solver), panel in groups.items():
        label = dict(mapping=mapping, w=w, solver=solver)
        by_sample = defaultdict(dict)
        for row in panel:
            if row['mode'] in by_sample[row['sample_id']]:
                raise ValueError('duplicate scored identity/mode')
            by_sample[row['sample_id']][row['mode']] = row
        for mode in dict.fromkeys(r['mode'] for r in panel):
            found = [r for r in panel if r['mode'] == mode]
            valid = [r for r in found if r['status'] == 'completed']
            record = dict(**label, mode=mode, expected=72, completed=len(valid),
                          status_counts=dict(Counter(r['status'] for r in found)))
            for field in ('metrics', 'center_metrics', 'clean_map_residual'):
                for coord in MODS:
                    for key in ('nrmse', 'rmse_coordinate', 'bias_coordinate'):
                        record[f'{field}_{coord}_{key}'] = equal_subject(
                            [dict(r, value=metric(r, field, coord, key)) for r in valid], 'value')
                    if coord != 'EEG':
                        for key in ('rmse_coordinate', 'bias_coordinate'):
                            record[f'{field}_{coord}_{key}_relative_hb'] = equal_subject(
                                [dict(r, value=metric(r, field, coord, key)*r['inverse_hb_factor'])
                                 for r in valid if metric(r, field, coord, key) is not None], 'value')
            mode_table.append(record)
        risks = []
        for identity, entries in by_sample.items():
            if all(m in entries and entries[m]['status'] == 'completed' for m in MODES):
                a, b = score(entries['center_EEG'], 'EEG'), score(entries['center_fNIRS'], 'fNIRS')
                if a is not None and b is not None:
                    risks.append(dict(subject=entries['full']['subject'], session=entries['full']['session'], value=.5*(a+b)))
        declared_modes = {m for t in tasks if t.get('mapping') == mapping and t.get('solver') == solver
                          and t.get('candidate', {}).get('w') == w for m in t.get('modes', [])}
        if set(MODES).issubset(declared_modes):
            panel_table.append(dict(**label, complete_14_mode_trials=len(risks), expected=72,
                                   full_B=equal_subject(risks, 'value') if len(risks) == 72 else None,
                                   descriptive_complete_subset_B=equal_subject(risks, 'value')))
        common = []
        separate_valid = 0
        for entries in by_sample.values():
            if all(entries.get(m, {}).get('status') == 'completed' for m in ('HbO_only', 'HbR_only')):
                separate_valid += 1
            if all(entries.get(m, {}).get('status') == 'completed' for m in ('full', 'fNIRS_only')):
                joint_row, self_row = entries['full'], entries['fNIRS_only']
                a = mean([metric(joint_row, 'clean_map_residual', c) for c in ('HbO', 'HbR')])
                b = mean([metric(self_row, 'clean_map_residual', c) for c in ('HbO', 'HbR')])
                if a is not None and b is not None:
                    common.append(dict(subject=joint_row['subject'], session=joint_row['session'], joint=a, fnirs=b, difference=a-b))
        modality_conflicts.append(dict(**label, separate_chromophores_valid=separate_valid,
            joint_fnirs_common=len(common), expected=72, joint_clean_hb_nmse=equal_subject(common, 'joint'),
            fnirs_only_clean_hb_nmse=equal_subject(common, 'fnirs'),
            adding_eeg_nmse_change=equal_subject(common, 'difference')))
        for direction in ('EEG', 'fNIRS'):
            for control in ('own', 'template', 'pairing', 'shift'):
                pairs = []
                for entries in by_sample.values():
                    a = entries.get('center_'+direction)
                    b = entries.get('center_'+direction+'_'+control)
                    if a and b and a['status'] == b['status'] == 'completed':
                        x, y = score(a, direction), score(b, direction)
                        if x is not None and y is not None:
                            pairs.append(dict(subject=a['subject'], session=a['session'], joint=x, control=y, increment=y-x))
                if pairs:
                    increments.append(dict(**label, direction=direction, control=control, common=len(pairs), expected=72,
                        joint_nmse=equal_subject(pairs, 'joint'), control_nmse=equal_subject(pairs, 'control'),
                        increment=equal_subject(pairs, 'increment'), full_panel=len(pairs) == 72))
        for row in panel:
            if row['mode'] == 'full':
                replay.append(dict(**label, sample_id=row['sample_id'], status=row['status'],
                    replay=row.get('replay'), transition_rms=row.get('state_transition_residual_rms'),
                    standardized_rms=row.get('standardized_state_transition_residual_rms')))
    curves = []
    grouped = defaultdict(list)
    for row in measured:
        if row['mode'] in ('full', 'fNIRS_only', 'EEG_only'):
            grouped[(row['mapping'], row['solver'], row['subject'], row['session'], row['mode'])].append(row)
    stability = []
    for (mapping, solver, subject, session, mode), panel in grouped.items():
        by_trial = defaultdict(dict)
        for row in panel:
            by_trial[row['sample_id']][row['w']] = row
        complete = {k: v for k, v in by_trial.items() if len(v) == 5 and all(
            r['status'] == 'completed' and r.get('parameter_log_likelihood' if solver == 'O1' else 'objective') is not None
            for r in v.values())}
        if solver not in ('O1', 'O2'):
            continue
        for w in [-.5, -.25, 0., .25, .5]:
            selected = [v[w] for v in complete.values()]
            curves.append(dict(mapping=mapping, solver=solver, subject=subject, session=session, mode=mode, w=w,
                common_complete_curves=len(complete), expected_trials=8,
                score=mean([r['parameter_log_likelihood'] if solver == 'O1' else -r['objective'] for r in selected]),
                observation_cost=mean([r.get('observation_cost') for r in selected]),
                state_prior_cost=mean([r.get('state_prior_cost') for r in selected])))
        if solver == 'O2':
            for identity, candidates in by_trial.items():
                valid = [r for r in candidates.values() if r['status'] == 'completed' and r.get('objective') is not None]
                if not valid:
                    continue
                best = min(valid, key=lambda r: r['objective'])
                equivalent = [r for r in valid if r is not best and r['objective']-best['objective'] <= .01*(1+abs(best['objective']))]
                with np.load(run/best['trajectory_path']) as arrays:
                    reference = arrays['r'].copy()
                for alternative in equivalent:
                    with np.load(run/alternative['trajectory_path']) as arrays:
                        other = arrays['r'].copy()
                    sd = np.std(reference)
                    stability.append(dict(mapping=mapping, subject=subject, session=session, sample_id=identity, mode=mode,
                        best_w=best['w'], alternative_w=alternative['w'], score_gap=alternative['objective']-best['objective'],
                        r_correlation=float(np.corrcoef(reference, other)[0, 1]) if sd > 1e-12 and np.std(other) > 1e-12 else None,
                        r_difference_reference_sd=float(np.sqrt(np.mean((reference-other)**2))/sd) if sd > 1e-12 else None))
    synthetic = []
    for law in ('linearized_gaussian', 'nonlinear_gaussian', 'nonlinear_student_t'):
        for mapping in ('prior_gauge', 'unit_loading', 'wrong_chromophore'):
            for mode in ('full', 'center_EEG', 'center_fNIRS', 'whole_EEG', 'whole_fNIRS'):
                panel = [r for r in rows if r['family'] == 'A' and r.get('law') == law and r.get('mapping') == mapping and r.get('mode') == mode]
                valid = [r for r in panel if r['status'] == 'completed']
                record = dict(law=law, mapping=mapping, mode=mode, completed=len(valid), expected=24)
                for field in ('truth', 'hidden_truth'):
                    for coord in ('r', 'clean_HbO', 'clean_HbR'):
                        for key in ('nrmse', 'correlation'):
                            record[f'{field}_{coord}_{key}'] = mean([metric(r, field, coord, key) for r in valid])
                synthetic.append(record)
    linear = [r for r in rows if r['family'] == 'D']
    linear_summary = []
    for direction in ('EEG', 'fNIRS'):
        selected = [r for r in linear if r['modality'] == direction]
        linear_summary.append(dict(direction=direction, completed=len(selected), expected=72,
            **{key+'_nmse': equal_subject([dict(r, value=r['nmse'][key]) for r in selected], 'value')
               for key in ('basic', 'joint', 'pairing', 'shift')}))
    invariance = []
    curve_groups = defaultdict(list)
    for row in curves:
        if row['mode'] == 'EEG_only' and row['solver'] == 'O1' and row['score'] is not None:
            curve_groups[(row['mapping'], row['subject'], row['session'])].append(row['score'])
    for key, values in curve_groups.items():
        invariance.append(dict(mapping=key[0], subject=key[1], session=key[2], spread=max(values)-min(values)))
    preference_groups = defaultdict(list)
    for row in curves:
        if row['score'] is not None:
            preference_groups[(row['mapping'], row['solver'], row['subject'], row['session'], row['mode'])].append(row)
    preferences = []
    for (mapping, solver, subject, session, mode), panel in preference_groups.items():
        spread = max(r['score'] for r in panel)-min(r['score'] for r in panel)
        best = max(panel, key=lambda r: r['score'])
        preferences.append(dict(mapping=mapping, solver=solver, subject=subject, session=session, mode=mode,
            common=best['common_complete_curves'], expected=8, spread=spread,
            optimum_w=best['w'] if spread > 1e-8 else None,
            boundary=best['w'] in (-.5, .5) if spread > 1e-8 else None))
    stability_summary = []
    for mapping in ('prior_gauge', 'unit_loading'):
        for mode in ('full', 'fNIRS_only', 'EEG_only'):
            panel = [r for r in stability if r['mapping'] == mapping and r['mode'] == mode]
            correlations = [r['r_correlation'] for r in panel if r['r_correlation'] is not None]
            differences = [r['r_difference_reference_sd'] for r in panel if r['r_difference_reference_sd'] is not None]
            stability_summary.append(dict(mapping=mapping, mode=mode, pairs=len(panel),
                median_r_correlation=float(np.median(correlations)) if correlations else None,
                median_r_difference_reference_sd=float(np.median(differences)) if differences else None))
    cfg = yaml.safe_load((run/'resolved_config.yaml').read_text())
    root = Path(read(run/'manifest.json')['project_root'])
    historical = root/cfg['measurement_retest']['historical_run']
    selected_inputs = []
    for subject in cfg['subjects']:
        with np.load(historical/'prepared'/f'{subject}.npz') as old, np.load(run/'prepared'/f'{subject}.npz') as new:
            for outer in range(4):
                info = read(run/'prepared'/f'{subject}_o{outer}_E0.json')
                projection = info['projection']
                pair = projection['fnirs_pair']
                old_projection_path = historical/'prepared'/f'{subject}_o{outer}_E0.json'
                old_projection = read(old_projection_path)['projection'] if old_projection_path.exists() else None
                for trial in info['validation']:
                    a, b = [arrays['target_fnirs'][trial, :, pair].copy() for arrays in (old, new)]
                    a -= a[:20].mean(axis=0); b -= b[:20].mean(axis=0)
                    error = (b-a)*projection['fnirs_factor']/np.asarray(info['normalization_sd'])[1:]
                    selected_inputs.append(dict(subject=subject, outer=outer, trial=trial, pair=pair,
                        historical_projection_available=old_projection is not None,
                        same_historical_pair=old_projection['fnirs_pair'] == pair if old_projection else None,
                        max_abs_change_new_training_sd=float(np.max(abs(error))),
                        rms_change_new_training_sd=float(np.sqrt(np.mean(error**2))),
                        old_hbo_hbr_rms_ratio=float(np.sqrt(np.mean(a[:,0]**2)/np.mean(a[:,1]**2))),
                        new_hbo_hbr_rms_ratio=float(np.sqrt(np.mean(b[:,0]**2)/np.mean(b[:,1]**2)))))
    return dict(modes=mode_table, panels=panel_table, paired_increments=increments, curves=curves,
        state_stability=stability, synthetic=synthetic, failures=failures, cell_failures=cell_failures, replay=replay,
        optimizer_starts=starts,
        modality_conflicts=modality_conflicts,
        linear=linear_summary, eeg_w_invariance=invariance, selected_input_attribution=selected_inputs,
        w_preferences=preferences, state_stability_summary=stability_summary,
        adaptation_trigger=read(run/'cells/retest_adaptation_trigger/result.json'))


def render(run, out):
    if out.exists():
        raise FileExistsError('Use a fresh versioned report export')
    if read(run/'manifest.json')['execution'] != 'completed':
        raise ValueError('Report requires the terminal experiment evidence')
    out.mkdir(parents=True)
    (out/'figures').mkdir()
    data = analyze(run)
    dump(out/'analysis.json', data)
    for key, value in data.items():
        if isinstance(value, list) and value:
            pd.DataFrame(value).to_csv(out/(key+'.csv'), index=False)
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
    figures = []
    def save(name, title, fig):
        fig.tight_layout()
        fig.savefig(out/'figures'/f'{name}.png', dpi=240, bbox_inches='tight')
        plt.close(fig)
        figures.append((name, title))
    selected = [r for r in data['modes'] if r['solver'] == 'O2' and r['w'] in (0., -.5)]
    labels = list(dict.fromkeys((r['mapping'], r['w']) for r in selected))
    modes = [*MODES, 'HbO_only', 'HbR_only']
    matrix = np.array([[next((r['completed'] for r in selected if (r['mapping'], r['w']) == lab and r['mode'] == mode), np.nan)
                        for mode in modes] for lab in labels])
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, vmin=0, vmax=72, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(modes)), modes, rotation=40, ha='right')
    ax.set_yticks(range(len(labels)), [f'{a}, W={b:g}' for a, b in labels])
    for i in range(len(labels)):
        for j in range(len(modes)):
            ax.text(j, i, f'{matrix[i,j]:g}', ha='center', va='center', color='white' if matrix[i,j] > 45 else 'black')
    fig.colorbar(im, ax=ax, label='有效 trial / 72')
    save('completion', '固定参数有效性：失败保留在 72 个身份的分母中', fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, mode in zip(axes, ['full', 'center_EEG', 'center_fNIRS']):
        found = [r for r in data['synthetic'] if r['law'] == 'nonlinear_gaussian' and r['mode'] == mode]
        ax.bar(range(len(found)), [r['truth_r_nrmse'] if r['truth_r_nrmse'] is not None else np.nan for r in found])
        ax.set_xticks(range(len(found)), [r['mapping'] for r in found], rotation=25, ha='right')
        ax.set_title(mode); ax.set_ylabel('r NRMSE')
    save('synthetic', '已知真值 r 恢复；各映射的生成信噪比不同，不能当作纯坐标改善', fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, solver in enumerate(['O1', 'O2']):
        for j, mode in enumerate(['EEG_only', 'fNIRS_only', 'full']):
            ax = axes[i, j]
            for mapping in ['prior_gauge', 'unit_loading']:
                group = defaultdict(list)
                for r in data['curves']:
                    if r['mapping'] == mapping and r['solver'] == solver and r['mode'] == mode and r['score'] is not None:
                        group[(r['subject'], r['session'])].append(r)
                for k, rows in enumerate(group.values()):
                    top = max(r['score'] for r in rows)
                    ax.plot([r['w'] for r in rows], [r['score']-top for r in rows], marker='.', alpha=.55,
                            color='#176B87' if mapping == 'prior_gauge' else '#CE793A', label=mapping if k == 0 else None)
            ax.set_title(f'{solver}: {mode}'); ax.set_xlabel('W'); ax.set_ylabel('相对最高分（同 session 完整曲线子集）')
            if mode == 'EEG_only':
                bound = max([1e-3, *[float(np.max(abs(line.get_ydata())))*1.1 for line in ax.lines]])
                ax.set_ylim(-bound, bound)
                ax.set_ylabel('分数差（理论不含 W 信息）')
            if ax.lines:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, '无完整有效曲线', ha='center', transform=ax.transAxes)
    save('w_curves', 'O1 边际似然与 O2 轨迹 MAP 分列；缺失曲线不补齐', fig)
    def table(rows, columns):
        if not rows:
            return '无满足比较条件的有效结果。\n'
        headings = dict(mapping='映射', w='W', solver='推断', mode='输入模式', law='生成规律',
            completed='有效数', expected='计划数', complete_14_mode_trials='14 模式完整数',
            full_B='完整 B', descriptive_complete_subset_B='子集 B', truth_r_nrmse='r NRMSE',
            truth_r_correlation='r 相关', truth_clean_HbO_nrmse='HbO NRMSE', truth_clean_HbR_nrmse='HbR NRMSE',
            variant='处理', ranks='前后秩', objective_difference='目标函数差',
            clean_map_residual_HbO_nrmse='HbO 残差', clean_map_residual_HbR_nrmse='HbR 残差',
            separate_chromophores_valid='分别有效数', joint_fnirs_common='共同有效数',
            joint_clean_hb_nmse='联合 Hb 误差', fnirs_only_clean_hb_nmse='单模态 Hb 误差', adding_eeg_nmse_change='加入 EEG 的变化',
            subject='被试', session='Session', common='共同数', optimum_w='最大值 W', boundary='边界',
            pairs='等价对数', median_r_correlation='r 相关中位数', median_r_difference_reference_sd='r 差/SD 中位数',
            direction='目标', control='对照', joint_nmse='联合 NMSE', control_nmse='对照 NMSE', increment='增量',
            basic_nmse='自身 NMSE', pairing_nmse='错配 NMSE', shift_nmse='移位 NMSE',
            passed='触发', reason='原因', status='状态', initial_flow_domain_exit='初态流量出域', count='数量')
        labels = dict(prior_gauge='先验幅度桥', unit_loading='共同 loading=1', wrong_chromophore='错误 HbR 缩放',
            linearized_gaussian='线性 Gaussian', nonlinear_gaussian='非线性 Gaussian', nonlinear_student_t='Student-t 失配')
        def fmt(v):
            if isinstance(v, str):
                return labels.get(v, v)
            return '未定义' if v is None else f'{v:.5g}' if isinstance(v, float) else str(v)
        return '\n'.join(['|'+'|'.join(headings.get(c, c) for c in columns)+'|', '|'+'|'.join(['---']*len(columns))+'|']+
                         ['|'+'|'.join(fmt(r.get(c)) for c in columns)+'|' for r in rows])+'\n'
    attribution = read(run/'input_attribution.json')
    floor_eeg = mean([f['noise']['evidence']['eeg']['trigger_fraction'] for f in attribution['folds']])
    floor_od = mean([f['noise']['evidence']['optical_density']['trigger_fraction'] for f in attribution['folds']])
    unit_gain = attribution['folds'][0]['projection']['observation_loading']['fnirs_common']
    report = '# 新测量链 SSM 重测：固定动力学、W 曲线与联合信息\n\n'
    report += ('本报告依据冻结任务表和逐拟合结果生成。实测为原 3 被试 × 3 session × 8 trial 的 72 个开发身份，'
        '所有变换与噪声由训练折冻结；14 模式主面板外增加两个色团单独观测诊断。'
        '主推断 O2 保留时间均值、相关噪声及输入—带噪目标交叉协方差。'
        '本轮没有 teacher 资格、后验区间或 tokenizer 训练。\n\n')
    execution = read(run/'summary.json')
    actual = sum(v['actual_solves'] for v in execution['families'].values())
    fixed_full = [r for r in selected if r['mode'] == 'full']
    full_valid = sum(r['completed'] for r in fixed_full)
    report += (f'本轮新增 {actual} 次拟合，另复用新测量链已完成的 W=0 三推断器对照及同折线性参考。'
        f'两种映射 × 两个固定 W 的完整联合 O2 拟合，有效 {full_valid}/{72*len(fixed_full)} 个 trial。'
        '优化失败或最终物理失败不计作有效拟合。\n\n')
    if full_valid == 0:
        report += ('**核心结论：新测量链及此次显式 loading 假设没有使固定动力学的联合非线性推断达到可用条件。'
            '这不能单独证明六状态模型不成立；优化预算和物理约束失败仍需区分。'
            '本轮不支持 W 已可辨识、联合 teacher 已有效或应默认开放 gain/Q。**\n\n')
    report += '## A：输入归因和已知真值\n\n'
    report += (f'新旧输入按全部 {len(attribution["trials"])} 个相同身份比较。12 个外折中 EEG 合成噪声 floor 触发比例为 '
        f'{floor_eeg:.1%}，波长噪声 floor 平均触发比例为 {floor_od:.1%}；这些最终值不能称为独立测得的传感器噪声。'
        'HbO/HbR 仍是近似 MBLL 的相对单位。完整幅度、HbT、baseline、边缘及锚点见上级 input_attribution.json。\n\n')
    selected_input_max = max(r['max_abs_change_new_training_sd'] for r in data['selected_input_attribution'])
    report += (f'以新折内模型选定的 Hb 对比较，在 72 个外折 trial 上的新旧最大逐点差为 '
               f'{selected_input_max:.6g} 个新训练 SD。全通道极值另含未选择的通道，不作为实际 SSM 输入变化的替代指标。\n\n')
    report += ('历史折投影缺失仍在 selected_input_attribution.csv 标记，不能把已有旧特征等同于历史拟合已完成；'
               '本轮主要统计比较使用同一份新测量折内投影。\n\n')
    report += table([r for r in data['synthetic'] if r['mode'] == 'full'],
                    ['law', 'mapping', 'completed', 'truth_r_nrmse', 'truth_r_correlation', 'truth_clean_HbO_nrmse', 'truth_clean_HbR_nrmse'])
    report += '\n错误单独放大 HbR 的条件是合成负对照，并非最近 SSM 实际输入链。Student-t 行使用 Gaussian MAP，属于生成失配压力检查。\n\n'
    rank_audit = read(run/'coordinate_rank_audit.json')
    report += table(rank_audit, ['variant', 'ranks', 'objective_difference'])
    report += '\n强滤波低秩条件下，仅更换坐标会改变相对 SVD 阈值保留的秩；因此本轮候选在同一旧计算坐标表达，固定目标、R 和评分 SD。\n\n'
    report += '## B：固定参数与色团可解释性\n\n'
    report += ('unit_loading 是在训练 pooled Hb MAD 坐标中固定共同 loading=1 的新相对观测假设；'
        f'在冻结旧评分坐标等价表达为 Hb 均值 gain={unit_gain:.7g}。它不代表物理浓度标定。'
        '原统计模型 W=0、原 O0/O1 和同折线性对照引用 9 月 16 日已完成证据，避免重复拟合；新增行保留独立运行身份。\n\n')
    report += table([r for r in data['panels'] if r['w'] in (0., -.5)],
                    ['mapping', 'w', 'solver', 'complete_14_mode_trials', 'full_B', 'descriptive_complete_subset_B'])
    report += '\n主风险 B=0.5 EEG NMSE+0.25 HbO NMSE+0.25 HbR NMSE；仅完整 72×14 时定义。各自成功子集不能用于规则排名。\n\n'
    report += 'O0 评分为 processed clean-map；O1/O2 为同源条件带噪目标预测，评分语义也不同，不能将表中跨推断器的子集 B 直接排名。\n\n'
    report += table([r for r in selected if r['mode'] in ('full', 'fNIRS_only', 'HbO_only', 'HbR_only')],
                    ['mapping', 'w', 'mode', 'completed', 'clean_map_residual_HbO_nrmse', 'clean_map_residual_HbR_nrmse'])
    report += '\n加入 EEG 的影响只在共同有效身份上比较（正值表示加入 EEG 后 Hb clean-map 误差更高）：\n\n'
    report += table([r for r in data['modality_conflicts'] if r['solver'] == 'O2' and r['w'] in (0., -.5)],
                    ['mapping', 'w', 'separate_chromophores_valid', 'joint_fnirs_common', 'joint_clean_hb_nmse',
                     'fnirs_only_clean_hb_nmse', 'adding_eeg_nmse_change'])
    report += '\n## C：同坐标 W 曲线与 r 稳定性\n\n'
    spread = max((r['spread'] for r in data['eeg_w_invariance']), default=None)
    report += f'EEG-only 精确线性参考的 session 平均似然在 W 网格上的最大变化为 {spread}。O1 曲线仅属于静息点线性化模型，O2 是轨迹 MAP，均不得当作已获得生理参数可信区间。\n\n'
    report += table([r for r in data['w_preferences'] if r['mode'] != 'EEG_only'],
                    ['mapping', 'solver', 'subject', 'session', 'mode', 'common', 'optimum_w', 'boundary'])
    report += '\n同一 trial 内 1% MAP 代价等价阈值下的 r 稳定性：\n\n'
    report += table(data['state_stability_summary'], ['mapping', 'mode', 'pairs', 'median_r_correlation', 'median_r_difference_reference_sd'])
    report += '\nEEG-only 对 W 的不变性是结构检查，不能代替联合或 fNIRS-only 潜状态稳定性的证据。等价对不足或完整曲线缺失时不能得出实用可辨识性已改善。\n\n'
    report += '\n## D：双向正确配对增量\n\n正值表示正确配对 NMSE 更低。每项只比较成对共同有效身份；不足 72 时为描述性结果。\n\n'
    report += table([r for r in data['paired_increments'] if r['solver'] == 'O2'],
                    ['mapping', 'w', 'direction', 'control', 'common', 'joint_nmse', 'control_nmse', 'increment'])
    report += '\n同折线性参考：\n\n'+table(data['linear'], ['direction', 'completed', 'basic_nmse', 'joint_nmse', 'pairing_nmse', 'shift_nmse'])
    report += '\n## E：适配触发与独立失败分类\n\n'+table([data['adaptation_trigger']], ['passed', 'completed', 'expected', 'reason'])
    start_counts = Counter((r['status'], r['reason'], r['initial_flow_domain_exit']) for r in data['optimizer_starts'])
    report += '\n优化起点诊断（一个拟合可以包含两个起点，不能与拟合行数相加）：\n\n'
    report += table([dict(status=k[0], reason=k[1], initial_flow_domain_exit=k[2], count=v) for k, v in start_counts.items()],
                    ['status', 'reason', 'initial_flow_domain_exit', 'count'])
    failures = Counter(r['status'] for r in data['failures'])
    report += '\n失败行（含复用的历史 W=0 对照）：'+str(dict(failures))+'。详见 failures.csv；MAP 评估预算、初始化流量出域和最终物理约束失败分别保留。\n\n'
    report += ('过程回放和状态转移残差见 replay.csv。观测拟合好或条件带噪插补改善不能自动转成 r 恢复、样本特异耦合或 physical teacher 结论。'
        '主分析是特征层遮挡；原生非线性前缺失只做干预回归，不据此声称整套实测缺失评估覆盖真实传感器丢失。\n\n')
    for name, title in figures:
        report += f'## {title}\n\n![{title}](figures/{name}.png)\n\n'
    report += '## 证据与复现\n\n完整表格见 [analysis.json](analysis.json)。配置、源码、任务表、日志和资源记录位于上级运行目录。图以 240 dpi PNG 嵌入 PDF，正文与表格可搜索；未在 WPS 中检查滚动/缩放。\n'
    (out/'REPORT.md').write_text(report)
    body = markdown.markdown(report, extensions=['tables'])
    body = body.replace('<img ', '<img width="510" ')
    css = 'body{font-family:sans-serif;font-size:9pt;line-height:1.45}h1{font-size:20pt}h2{font-size:13pt}table{font-size:6.5pt;border-collapse:collapse}td,th{border:0.5pt solid #cbd5df;padding:3pt}img{max-width:100%}p{margin:6pt 0}'
    story = fitz.Story(body, user_css=css, archive=fitz.Archive(str(out)))
    def rectfn(number, filled):
        if number > 70:
            raise RuntimeError('PDF layout did not terminate')
        media = fitz.paper_rect('a4')
        return media, media+(36, 32, -36, -32), None
    pdf = story.write_with_links(rectfn)
    pdf.save(out/'REPORT.pdf', garbage=4, deflate=True)
    images = sum(len(p.get_images()) for p in pdf)
    text_chars = sum(len(p.get_text()) for p in pdf)
    pages = len(pdf)
    pdf.close()
    atlas = fitz.open()
    for name, title in figures:
        page = atlas.new_page(width=842, height=595)
        page.insert_image(fitz.Rect(25, 50, 817, 565), filename=str(out/'figures'/f'{name}.png'), keep_proportion=True)
        page.insert_htmlbox(fitz.Rect(25, 12, 817, 46), html.escape(title))
    atlas.save(out/'FIGURES.pdf', garbage=4, deflate=True)
    atlas.close()
    for name, _ in figures:
        uri = 'data:image/png;base64,'+base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
        body = body.replace(f'figures/{name}.png', uri)
    (out/'REPORT.html').write_text('<html lang="zh-CN"><meta charset="utf-8"><style>'+css+'</style>'+body+'</html>')
    audit = dict(pages=pages, figure_image_objects=images, expected_figures=len(figures), searchable_text_characters=text_chars,
                 bitmap_dpi=240, wps_checked=False, source_task_table=str(run/'task_table.csv'))
    assert images == len(figures) and text_chars > 1000, audit
    dump(out/'report_validation.json', audit)
    shutil.copyfile(__file__, out/'render_source.py')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    render(args.run_dir.resolve(), args.output_dir.resolve())
