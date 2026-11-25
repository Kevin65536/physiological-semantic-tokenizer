"""Render the paired teacher and formal integer-decoding evidence as a searchable PDF."""

import argparse
from pathlib import Path
import sys
import fitz
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scripts.check_continuous_so_targets import (
    OUT,
    PRIOR,
    ORDER,
    old,
    projection,
    complete_blocks,
    project_blocks,
)
from experiments.scripts.summarize_continuous_so import SHORT, FORMAL
from experiments.scripts.render_temporal_projection_so import DS, table, p
from experiments.scripts.render_projected_so_members import CSS

COL = ['#344f66', '#ba673b', '#147768']
LABEL = {
    'short_v3': 'v3',
    'short_projected_v3': '逐块 P(S₀)',
    'short_projected_x': '逐块 P(X)',
    'continuous': '连续投影',
    'v3': 'v3',
    'block': '逐块 S',
    'direct': '连续投影 X',
}


def saved(fig, file):
    fig.savefig(file, dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return file


def target_figure(folder):
    synth = pd.read_csv(OUT / 'synthetic.csv')
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), layout='constrained')
    ax = axes[0, 0]
    freq = [10, 11, 11.25, 11.5, 12.5]
    x = np.arange(len(freq))
    for j, (arm, k) in enumerate([('block', 1), ('continuous', 2)]):
        g = synth[synth.arm == arm].groupby('hz').energy.min()
        ax.bar(x + (j - 0.5) * 0.34, g.loc[freq], 0.34, label=arm, color=COL[k], hatch='//' if j == 0 else '')
    ax.set(
        xticks=x,
        xticklabels=freq,
        xlabel='Sine frequency (Hz)',
        ylabel='Minimum energy retained',
        ylim=(0, 1.11),
    )
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    df = pd.read_csv(OUT / 'target.csv')
    for i, (_, g) in enumerate(df.groupby('i')):
        ax.plot(
            [0, 1, 2],
            g.set_index('arm').loc[['v3', 'block', 'continuous'], 'boundary_ratio'],
            '-o',
            ms=3,
            alpha=0.7,
        )
    ax.axhline(1, ls=':', color='black')
    ax.set(
        xticks=[0, 1, 2], xticklabels=['v3', 'Block', 'Continuous'], ylabel='Boundary / all difference RMS'
    )
    case = old.read(PRIOR / 'plan.json')['cases'][0]
    site = next(j for j, n in enumerate(case['valid']) if case['channels'][n].upper() == 'AFP1')
    with np.load(old.PARENT / 'targets' / str(case['i']) / 'baseline.npz') as f:
        raw = f['v3'][:-20, site : site + 1]
    b, sl = complete_blocks(raw, case['source']['start_sample'])
    c = projection(raw)[sl]
    y = project_blocks(b, ORDER[:32])
    t = np.arange(len(y)) / 200
    for ax, half in zip(axes[1], [1.8, 0.16]):
        cut = abs(t - 40) < half
        for w, co, ls, lab in zip([raw[sl], y, c], COL, ['-', '--', ':'], ['v3', 'Block', 'Continuous']):
            ax.plot(t[cut] - 40, w[cut, 0], color=co, ls=ls, lw=1.4, label=lab)
        ax.axvline(0, color='#888', lw=0.8)
        ax.set(xlabel='Seconds from original block boundary', ylabel='S target (µV)')
    axes[1, 0].legend(fontsize=8)
    return saved(fig, folder / 'target.png')


def decode_figure(folder, seed, case):
    fig, axes = plt.subplots(3, 2, figsize=(9, 5.4), layout='constrained', sharex=True)
    for j, arm in enumerate(['v3', 'continuous']):
        with np.load(OUT / 'evaluation' / f'formal_{arm}_{seed}' / (case + '.npz')) as d:
            site = str(d['site'])
            person = str(d['subject'])
            scale = float(d['scale'])
            ex = d['example']
            valid = d['mask'].ravel()
            time = np.arange(len(valid)) / 200
            for i, (k, b) in enumerate(zip('xso', ['sum', 'S', 'O'])):
                ax = axes[i, j]
                y = d[k].ravel() * scale
                yh = d['decoded_' + b].ravel() * scale
                cut = (time >= 12) & (time < 16) & valid
                ax.plot(
                    time[cut], y[cut], color=COL[0], lw=1, label='Observed X' if k == 'x' else 'Branch target'
                )
                ax.plot(time[cut], yh[cut], color=COL[2], ls='--', lw=1, label='Integer decode')
                ax.set_ylabel(k.upper() + ' (µV)')
                if i == 0:
                    ax.set_title(f'{arm} · {person} {site} · seed {seed}')
                    ax.legend(fontsize=8)
                if i == 2:
                    ax.set_xlabel('Time in 30 s crop (s)')
    for aa in axes:
        low = min(ax.get_ylim()[0] for ax in aa)
        high = max(ax.get_ylim()[1] for ax in aa)
        for ax in aa:
            ax.set_ylim(low, high)
    return saved(fig, folder / f'{case}_{seed}.png')


def members_figure(folder, seed, short=False):
    source = PRIOR / 'vq_screen'
    x = np.load(source / 'x.npy', mmap_mode='r').reshape(-1, 15, 400)
    arms = ['v3', 'projected_v3', 'projected_x'] if short else ['v3', 'continuous']
    fig, axes = plt.subplots(2, len(arms), figsize=(9, 4.7), layout='constrained')
    for j, arm in enumerate(arms):
        tag = 'short_' + arm if short else f'formal_{arm}_{seed}'
        dest = OUT / 'members' / tag
        sel = pd.read_csv(dest / 'selection.csv').set_index(['branch', 'endpoint'])
        v = pd.read_csv(dest / 'validation_members.csv.gz', index_col='row', dtype={'subject': str})
        matches = pd.read_csv(dest / 'matches.csv')
        for i, (b, endpoint) in enumerate([('S', 'alpha'), ('O', 'shape')]):
            ax = axes[i, j]
            g = matches[(matches.branch == b) & (matches.endpoint == endpoint)]
            token = int(sel.loc[(b, endpoint), 'token'])
            if not len(g):
                ax.set_title(f'{arm} · {b} {token}')
                ax.set_axis_off()
                ax.text(.5,.5,'No matched development patches',ha='center',va='center',transform=ax.transAxes)
                continue
            examples = v.loc[g.member_row].drop_duplicates('person').iloc[:3]
            for ix, r in examples.iterrows():
                wave = x[r.example, r.block]
                centered = wave - wave.mean()
                if i == 0:
                    power = abs(np.fft.rfft(centered * np.hanning(400))) ** 2
                    f = np.arange(2, 91) * 0.5
                    y = power[2:91] / power[2:91].sum() * 100
                else:
                    f = np.arange(400) / 200
                    y = centered / centered.std()
                ax.plot(f, y, label=f'{DS[r.dataset]} {r.subject} {r.site}', lw=0.95)
            ax.set_title(f'{arm} · {b} {token} · seed {seed}', fontsize=11)
            ax.set(
                xlabel='Frequency (Hz)' if i == 0 else 'Time (s)',
                ylabel='% power / bin' if i == 0 else 'Observed X / patch SD',
            )
            ax.legend(fontsize=7)
            if i == 0:
                ax.axvspan(8, 13, alpha=0.15, color=COL[1])
    return saved(fig, folder / f'members_{seed}_{len(arms)}arms.png')


def render(folder):
    folder.mkdir(parents=True, exist_ok=False)
    doc = fitz.open()

    def page(title, body, fig=None, after=''):
        pg = doc.new_page(width=720, height=840)

        def box(y, bottom, body):
            spare, _ = pg.insert_htmlbox(
                fitz.Rect(30, y, 690, bottom), '<div lang="zh-Hans">' + body + '</div>', css=CSS, scale_low=1
            )
            if spare < 0:
                raise ValueError((title, y, bottom))
            return bottom - spare

        y = box(19, 806, '<small>EEG 时间投影迁移试验　 /　公开开发面板</small><h1>' + title + '</h1>' + body)
        if fig:
            pix = fitz.Pixmap(str(fig))
            h = 650 * pix.height / pix.width
            pg.insert_image(fitz.Rect(35, y + 3, 685, y + 3 + h), filename=str(fig))
            y += h + 10
        if after:
            box(y, 806, after)
        box(
            812,
            838,
            f'<small>EEG-SO-CONTINUOUS-PROJECTION-v2　·　2026 年 11 月 25 日　·　第 {pg.number+1} 页</small>',
        )

    r = old.read(OUT / 'summary.json')
    rank = old.read(OUT / 'cache_checks.json')
    person_scores = pd.read_csv(OUT / 'people.csv')
    gate_names = dict(
        common_X='共同 X 误差',
        S_nmse='S 误差',
        O_nmse='O 误差',
        S_corr='S 波形相关',
        O_corr='O 波形相关',
        S_shift='S 移窗一致率',
        S_alpha_effect='S 的 α 富集',
        S_coverage='S 的 α 候选覆盖人数',
    )
    paired_rows = []
    for seed in [17, 43]:
        z = person_scores.query('seed==@seed and metric=="nmse"').pivot(
            index=['dataset', 'subject'], columns='arm', values='X'
        )
        change = (z.continuous / z.v3 - 1) * 100
        paired_rows.append(
            [
                seed,
                f'{(change<0).sum()} / {len(change)}',
                f'{change.median():+.2f}%',
                f'{change.min():+.2f}% 至 {change.max():+.2f}%',
            ]
        )
    m = pd.read_csv(OUT / 'macro.csv').set_index(['seed', 'arm', 'metric'])
    target = pd.read_csv(OUT / 'target.csv')
    it = pd.read_csv(OUT / 'intervention.csv').query('variant=="ocular1"')
    cap = pd.read_csv(OUT / 'cap.csv').query('variant=="baseline"')
    edge = (
        pd.read_csv(OUT / 'parent_edge_sensitivity.csv')
        .groupby(['region', 'i'])
        .difference_over_full_S.mean()
        .groupby('region')
        .median()
    )
    mr = pd.read_csv(OUT / 'member_comparison.csv').set_index(['model', 'branch', 'endpoint'])
    shift = (
        pd.read_csv(OUT / 'natural_shift_people.csv')
        .groupby(['model', 'branch', 'dataset'])
        .rate.mean()
        .groupby(['model', 'branch'])
        .mean()
    )
    verdict = '两种子均达到迁移条件' if r['eligible_for_migration'] else '正式预算复核后保留 v3'
    page(
        verdict,
        p(
            '本轮只更换 EEG 的 S 监督目标。S₀ 来自既有 v3 教师；候选保留其连续时段中的指定频率，O 仍等于实测 X 减 S。模型编码器只看 X，每个电极的 30 秒窗输出 15 对 S/O 整数。实际部署的生产目标仍为 v3。'
        )
        + table(
            ['种子', '目标', 'X NMSE', 'S NMSE', 'O NMSE', 'S 移窗同码率'],
            [
                [
                    s,
                    LABEL[a],
                    *[f'{v:.4f}' for v in m.loc[(s, a, 'nmse'), list('XSO')]],
                    f'{shift[f"formal_{a}_{s}","S"]:.1%}',
                ]
                for s in [17, 43]
                for a in ['v3', 'continuous']
            ],
        )
        + p(
            '每个种子使用完整正式 C 的 60 名训练被试、相同抽样顺序、批量 64、1,000 步连续预热和 9,000 步硬量化训练。旧目标复用同种子终点，并在全部开发例上复算核对。16 名开发被试提供 53,476 个电极裁窗；移窗和成员证据单独在原 8 人面板计算。'
        )
        + p(
            '共同 X 的目标和分母跨臂一致。分支 NMSE 除以各自目标在同数据集、电极的训练方差，同时报告有符号波形相关及数据集内原单位 RMSE。各裁窗内按有效点计分，再依次等权平均电极、同记录的裁窗、同被试记录、同库被试和四个数据集。'
        )
        + table(['种子', 'X 误差下降人数', '逐人变化中位数', '逐人变化范围'], paired_rows)
        + table(
            ['种子', '未达到的迁移条件'],
            [
                [
                    s,
                    '、'.join(
                        gate_names[g['gate']] for g in r['checks'] if g['seed'] == s and not g['passed']
                    )
                    or '全部达到',
                ]
                for s in [17, 43]
            ],
        )
        + p(
            '判定使用事前固定的两种子条件。共同 X NMSE 须下降；S/O 本分支 NMSE 最多上升 5%，相关最多下降 0.01；S 的自然移窗一致率、α 成员效应及覆盖人数须维持。下表和图谱同时保留 O 的代价。这些比较使用已反复研究的开发面板，未作独立确认性评估。'
        ),
    )
    page(
        '补齐 α，消除原两秒格点处的异常跳变',
        p(
            '在每个父记录、电极的连续合格时段上，用正交第二型离散余弦变换（DCT-II）投影一次，再取模型裁窗。跨度 N 个采样点允许至多 ⌊64N/400⌋ 个实系数。先保留 3.75–13.25 Hz，再按既有训练排序填充外围频率。有效缺口、排除区间和父记录边缘切断变换。'
        )
        + p(
            f'在正式支持的 {rank["eligible_channel_spans"]:,} 个电极有效段上，实际平均每 400 点 {rank["effective_real_coefficients_per_400"]:.2f} 个系数，4–13 Hz 内缺失应保留频点的段数为 {rank["spans_missing_core_bins"]}。例如 17,980 点有 2,876 个实系数。每个 2 秒片段依赖同一有效时段的其他部分；其局部向量没有单独的 64 维约束。标签是离线的。余弦基在真实端点采用偶延拓。'
        ),
        target_figure(folder),
        p(
            '左上按 8 个相位取最小保留能量。右上每条线是一人，纵轴为两秒格点处一阶差分 RMS 除以全段一阶差分 RMS。下排为预定首例 Single-Trial 01、AFp1，在同一内部格点的两种时间刻度。原位比较显示了波形变化；逐人的绝对边界差分和分母见 target.csv。'
        ),
    )
    rows = []
    for a in ['v3', 'block', 'continuous', 'direct']:
        z = target[target.arm == a].median(numeric_only=True)
        c = cap[cap.arm == a].median(numeric_only=True)
        rows.append(
            [
                LABEL[a],
                f'{z.s_energy:.3f}',
                f'{z.theta_power:.3f}',
                f'{z.alpha_power:.3f}',
                f'{z.beta_power:.3f}',
                f'{z.gamma_power:.3f}',
                f'{z.boundary_ratio:.3f}',
            ]
        )
    page(
        '目标保留、眼电和物理约束分开评价',
        table(['监督目标', 'S₀ 能量比', 'θ 功率比', 'α 功率比', 'β 功率比', 'γ 功率比', '边界比'], rows)
        + p(
            '表值为 8 名被试的中位数。先在电极上与同段旧 S 比较；频带功率通过原滤波器组和一秒包络计算。覆盖的 θ/α 能量和包络相对于原教师保留，β、γ 归属仍有变化。闭合误差和每个被试的分子分母分别见 target.csv、intervention.csv。'
        )
        + table(
            ['目标', '眼电注入泄漏', '自然眼电投影比', '最终 cap P90', '超限帧×频带比例'],
            [
                [
                    LABEL[a],
                    f'{it[it.arm==a].leak.median():.3f}',
                    f'{target[target.arm==a].eog_ratio.median():.3f}',
                    f'{cap[cap.arm==a].p90.median():.3f}',
                    f'{cap[cap.arm==a].above_one.median():.1%}',
                ]
                for a in ['v3', 'block', 'continuous', 'direct']
            ],
        )
        + p(
            '泄漏是注入活动区间内 S 变化 RMS / 已知注入 RMS。自然眼电用 4 人独立 HEOG、VEOG 的额区投影，与原 X 的投影范数作比。物理 cap 按最终多通道波形在每帧、每频带的最大广义特征值评价，报告相对 v3 门槛和绝对超限。'
        )
        + p(
            '对连续投影，记录两端各一秒的改变量 RMS 相对当地旧 S 的被试中位数为 %.3f，内部为 %.3f。该指标描述目标改变量；真实记录端点没有跨端实测连续性真值。'
            % (
                target.query('arm=="continuous"').edge_change_from_v3.median(),
                target.query('arm=="continuous"').interior_change_from_v3.median(),
            )
        )
        + p(
            f'另将同一份旧 S 的有效父段截短为两种 60 秒支持再独立投影。与完整父段对应时刻相比，新端点内侧 2 秒的相对 RMS 差为 {edge["first_2s_from_new_edge"]:.1%}，距新端点至少 10 秒为 {edge["interior_at_least_10s"]:.1%}（先平均两种截法，再取 8 人中位数）。实际构造先固定父段，随后裁窗。'
        ),
    )

    def member_table(tags):
        rows = []
        for tag in tags:
            lab = LABEL.get(tag, tag.replace('formal_', ''))
            for b, k in [('S', 'alpha'), ('O', 'shape')]:
                z = mr.loc[tag, b, k]
                rows.append(
                    [
                        lab,
                        b + str(int(z.token)),
                        ('α 占比' if k == 'alpha' else '波形相关'),
                        ('无配对' if not z.matched else (f'{z.difference*100:+.2f} pp' if k == 'alpha' else f'{z.difference:+.3f}')),
                        f'{int(z.matched)} / {int(z.distinct_times)} / {int(z.people)}',
                        f'{shift[tag,b]:.2%}',
                    ]
                )
        return table(['模型', '词元', '真实 X 属性', '成员减对照', '配对段 / 时段 / 人', '移窗一致率'], rows)

    page(
        '相同检索下，旧 v3 和直接投影也有可重复的词',
        p(
            '用原有的三个 500 步模型重发整数。67,710 个训练电极片段选词，12,414 个开发电极片段独立检索。同人同电极匹配 RMS 最近的其他编号，允许最大幅度比 1.4。效应先在人内平均，再对有配对的人等权；重复电极按同一人同一时刻去重计时段。'
        )
        + member_table(SHORT)
        + p(
            'α 候选按训练成员平均 8–13 Hz 功率占比选择，形态候选按保留符号和起点的逐段单位波形均值长度选择。开发相关评分使用冻结训练模板。自然移窗是同一个两秒片段在相差两秒的 30 秒上下文中的整数相等率，按人、库等权。'
        )
        + p(
            '逐块 S 的 α 差值为 +5.33 个百分点，旧 v3 为 +2.74，直接投影 X 为 +2.08；三者均覆盖 8 人。O 波形相关差反而以旧 v3 较大，而且覆盖更多人。成员检索方法本身可以在所有模型中找到有结构的例子。完整六项候选、两组搭配、额区眼电和投影背景残差见 member_comparison.csv。'
        ),
    )
    page(
        '相同规则检索出的三种短训码本成员',
        p(
            '每列一份 500 步码本，种子均为 17。列顺序依次为 v3、逐块投影旧 S、逐块投影 X。每个面板按匹配表次序，从至多三位不同被试各取第一条可用片段，样例不按效果挑选。'
        ),
        members_figure(folder, 17, short=True),
        p(
            '上排为训练选出的 S 词在开发成员实测 EEG 上的功率谱，各点为该频点占 1–45 Hz 功率的百分数。下排为 O 形态词成员的实测 EEG，按每段标准差缩放。各面板纵轴独立；全体成员与对照的配对差及人数见前页。'
        ),
    )
    for seed, case in [(17, 'median'), (43, 'difficult')]:
        native = pd.read_csv(OUT / 'dataset.csv').query('metric=="rmse_native" and seed==@seed')
        page(
            f'种子 {seed} 的整数解码波形与原单位误差',
            p(
                '沿用正式容量试验预先固定的'
                + ('中等难度' if case == 'median' else '困难')
                + '窗，显示第 12–16 秒。两列为同一电极和物理时段。第一行实线为共同 X，第二、三行为各自监督目标，虚线为仅由整数序列解出的电压。每行两列共用纵轴。'
            ),
            decode_figure(folder, seed, case),
            table(
                ['库', 'X RMSE 旧 → 新', 'S RMSE 旧 → 新', 'O RMSE 旧 → 新'],
                [
                    [
                        DS[ds],
                        *[
                            f'{g.set_index("arm").loc["v3",k]:.3g} → {g.set_index("arm").loc["continuous",k]:.3g}'
                            for k in 'XSO'
                        ],
                    ]
                    for ds, g in native.groupby('dataset')
                ],
            )
            + p('误差来自该库全部正式开发记录。REFED 使用释放文件的相对单位，其余三库使用 µV。'),
        )
        page(
            f'种子 {seed} 的真实成员与自然移窗',
            member_table([f'formal_v3_{seed}', f'formal_continuous_{seed}']),
            members_figure(folder, seed),
            p(
                '上排为 S 的 α 候选在实测 X 中的相对功率谱，下排为 O 的形态候选在实测 X 中的带符号波形。每幅图按匹配表次序取最先出现的至多 3 人各一段。阴影为 8–13 Hz；波形去均值并除以本段标准差，各面板纵轴独立。'
            )
            + p(
                '波形相似、频谱偏好和监督目标的生理来源是不同问题。第二个种子独立选词，编号只在各自码本内有意义。表中比较相同检索规则下的效应与覆盖人数。'
                + ('修正目标的 O 6132 在 3 人的 5 个时段内显示较强波形效应；S 的 α 富集为 +1.24 个百分点。' if seed==17 else '修正目标的 O 5036 在开发面板命中 0 次，因此没有可估算的波形效应。S 的 α 差值为 −3.91 个百分点，种子 17 的正向偏好没有复现。')
            ),
        )
    doc.subset_fonts()
    doc.save(folder / '连续投影修正与双种子迁移测试.pdf', garbage=4, deflate=True)
    doc.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
