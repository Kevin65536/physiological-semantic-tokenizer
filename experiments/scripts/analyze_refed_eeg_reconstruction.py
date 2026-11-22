"""Explain the between-dataset voltage NMSE gap on the frozen phase-B EEG controls."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import time

import run_dual_component_tokenizer as runner
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import torch

from src.analysis.dual_component_tokenizer import base_training_aggregate

BASE = runner.SOURCE / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer'
MATRIX = BASE / '20261120_input_quantization_b_v1'
EDGES = [0, 1, 4, 8, 13, 30, 45, 49, 51, 101]
BANDS = ['sub1', '1_4', '4_8', '8_13', '13_30', '30_45', '45_49', '49_51', '51_100']


def split_energy(y, mask):
    """Exact rectangular-window Parseval partition in the original scoring norm."""
    count = mask.sum(1)
    assert np.all(count > 0)
    mean = np.where(mask, y, 0.).sum(1) / count
    centered = np.where(mask, y - mean[:, None], 0.)
    power = abs(np.fft.rfft(centered))**2 / (count[:, None] * y.shape[1])
    power[:, 1:-1] *= 2
    freq = np.fft.rfftfreq(y.shape[1], 1 / 200)
    bands = np.column_stack([power[:, (freq >= lo) & (freq < hi)].sum(1)
                             for lo, hi in zip(EDGES, EDGES[1:])])
    np.testing.assert_allclose(bands.sum(1) + mean**2, np.where(mask, y*y, 0).sum(1)/count,
                               rtol=1e-10, atol=1e-13)
    return mean**2, bands


def check_spectrum():
    t = np.arange(6000)/200
    x = (2 + np.cos(2*np.pi*10*t) + 2*np.sin(2*np.pi*50*t))[None]
    mask = np.ones(x.shape, bool)
    dc, bands = split_energy(x, mask)
    np.testing.assert_allclose(dc, [4])
    np.testing.assert_allclose(bands[0], [0,0,0,.5,0,0,0,2,0], atol=1e-12)
    mask[:, :11] = False
    split_energy(x, mask)


def replay_block(bounds):
    start, end = bounds
    cache = OrderedDict()
    blocks = []
    with torch.inference_mode():
        for i in range(start, end, 32):
            examples = EXAMPLES[i:min(i+32, end)]
            batch = runner._base_training_batch(ROWS, examples, SCALES, 'cpu', cache, CONFIG)
            x = batch['x'].numpy().reshape(len(examples), -1).astype(float)
            mask = batch['mask'].numpy().reshape(x.shape)
            p = MODEL(batch['x'])['sum'].numpy().reshape(x.shape).astype(float)
            var = batch['vx'][:, 0, 0].numpy().astype(float)
            mu2, xband = split_energy(x, mask)
            bias2, errband = split_energy(p-x, mask)
            ac = xband.sum(1)
            mse = bias2 + errband.sum(1)
            patch_x = np.where(mask, x, 0.).reshape(len(x), 15, 400)
            patch_n = mask.reshape(patch_x.shape).sum(2).clip(1)
            level = (patch_x.sum(2)**2 / patch_n).sum(1)/mask.sum(1)
            shape = (ac + mu2 - level).clip(0)
            features = dict(nmse=mse/var, valid_fraction=mask.mean(1), ac_train=ac/var,
                            dc_train=mu2/var, patch_shape_train=shape/var,
                            local_nmse=np.divide(mse, ac, out=np.full_like(ac,np.nan), where=ac>0),
                            err_dc=bias2/var)
            for k, name in enumerate(BANDS):
                features['x_'+name] = xband[:,k]/var
                features['err_'+name] = errband[:,k]/var
            blocks.append(pd.DataFrame(features))
    return pd.concat(blocks, ignore_index=True)


def report(out, stats, reference, frame):
    metrics = stats['metrics']
    for level in ('dataset', 'subject'):
        if level == 'dataset':
            table = pd.DataFrame({key: value['by_dataset'] for key,value in metrics.items()})
            table.index.name = 'dataset'
        else:
            table = pd.DataFrame({key: {(r['dataset'],r['subject']):r['value'] for r in value['by_subject']}
                                  for key,value in metrics.items()})
            table.index.names = ['dataset','subject']
        table.to_csv(out/(level+'.csv'))
    other = ['eeg_fnirs_single_trial','simultaneous_eeg_nirs','visual_cognitive_motivation']
    data = pd.read_csv(out/'dataset.csv').set_index('dataset')
    ref, baseline = data.loc['refed'], data.loc[other].mean()
    keys = ['err_dc'] + ['err_'+b for b in BANDS]
    gap = ref[keys]-baseline[keys]
    np.testing.assert_allclose(gap.sum(), ref.nmse-baseline.nmse)
    pd.DataFrame(dict(refed=ref[keys],other_three_equal=baseline[keys],difference=gap,
                      fraction_of_nmse_gap=gap/gap.sum())).to_csv(out/'gap_by_band.csv')
    subjects = pd.read_csv(out/'subject.csv')
    tests = []
    for dataset in other:
        a = subjects.loc[subjects.dataset=='refed','nmse']
        b = subjects.loc[subjects.dataset==dataset,'nmse']
        tests.append(dict(comparison=dataset, n_refed=len(a), n_other=len(b),
                          p=mannwhitneyu(a,b,alternative='two-sided',method='exact').pvalue))
    tests.sort(key=lambda r:r['p'])
    adjusted=0.
    for i, item in enumerate(tests):
        adjusted=max(adjusted,min(1.,item['p']*(len(tests)-i)))
        item['holm_p']=adjusted
    # The retained endpoint was evaluated on CUDA. Measure CPU replay deviation
    # separately from the exact within-replay Parseval identity.
    errors = np.abs(EXPECTED-frame.nmse.to_numpy())
    frozen_subjects = {(r['dataset'],r['subject']):r['value'][0][0]
                       for r in reference['metrics']['nmse']['by_subject']}
    for row in subjects.itertuples():
        np.testing.assert_allclose(row.nmse, frozen_subjects[row.dataset,str(row.subject)],
                                   atol=.0005, rtol=.001)
    names = [ROWS[i]['positions'][p].upper() for i, _, p in EXAMPLES]
    positions = {}
    for (i, _, _), name in zip(EXAMPLES, names):
        positions.setdefault(ROWS[i]['dataset'], set()).add(name)
    common = set.intersection(*positions.values())
    take = np.array([name in common for name in names])
    common_stats = base_training_aggregate({'nmse': EXPECTED[take]}, np.asarray(EXAMPLES)[take], ROWS)
    runner.write_json(out/'common_electrodes.json',dict(positions=sorted(common), **common_stats))
    runner.write_json(out/'verification.json',dict(
        model='phase B X continuous, E1 waveform8, seed 17, 10000 steps', split='validation',
        checkpoint=str(JOB/'checkpoint_step_10000.pt'), score=str(JOB/'evaluation/step_10000/validation.json'),
        counts={k:stats[k] for k in ('inputs','windows','records','subjects')},
        tests=tests, nonfinite={k:v['nonfinite_entries'] for k,v in metrics.items()},
        max_input_nmse_difference=float(errors.max()), p99_input_nmse_difference=float(np.quantile(errors,.99)),
        max_subject_nmse_difference=max(abs(r.nmse-frozen_subjects[r.dataset,str(r.subject)]) for r in subjects.itertuples()),
        method='Per admitted input: 30 s at 200 Hz. Remove valid-sample mean, zero invalid centered samples, one-sided rectangular FFT, normalize by N times valid count and frozen training X variance. Add DC separately. Average positions, crops, records, subjects. Positive sub1 excludes DC. local_nmse divides each input MSE by its own valid-sample variance. err_ columns sum to replayed X NMSE. Relative band fractions from dataset totals are energy-weighted within this hierarchy. Statistical tests are exploratory exact two-sided independent-subject rank tests on X NMSE, with Holm correction for the three comparisons.'))



def summarize_frequency_gap(out):
    """Account for the gap by swapping measured factors, averaging all six orders."""
    from itertools import permutations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    datasets = pd.read_csv(out/'dataset.csv').set_index('dataset')
    x = datasets[['x_'+b for b in BANDS]].to_numpy()
    error = datasets[['err_'+b for b in BANDS]].to_numpy()
    total = x.sum(1)
    share = x / total[:, None]
    relative_error = error / x
    factors = list(zip(total, share, relative_error))
    ref = datasets.index.get_loc('refed')
    rows = []
    for j, dataset in enumerate(datasets.index):
        if j == ref:
            continue
        contributions = np.zeros(3)
        for order in permutations(range(3)):
            levels = list(factors[j])
            for k in order:
                before = levels[0]*np.dot(levels[1], levels[2])
                levels[k] = factors[ref][k]
                after = levels[0]*np.dot(levels[1], levels[2])
                contributions[k] += (after-before)/6
        dc = datasets.err_dc.iloc[ref]-datasets.err_dc.iloc[j]
        gap = datasets.nmse.iloc[ref]-datasets.nmse.iloc[j]
        np.testing.assert_allclose(contributions.sum()+dc,gap)
        rows.append(dict(comparator=dataset, gap=gap, ac_in_training_units=contributions[0],
                         spectral_composition=contributions[1], within_band_error_ratio=contributions[2],
                         squared_mean_error=dc))
    pd.DataFrame(rows).to_csv(out/'gap_factorization.csv',index=False)

    names = dict(eeg_fnirs_single_trial='Single-Trial', refed='REFED',
                 simultaneous_eeg_nirs='Simultaneous', visual_cognitive_motivation='Visual')
    order = list(names)
    cols = ['#49758f','#ad452f','#298174','#8a6896']
    band_labels = ['<1','1–4','4–8','8–13','13–30','30–45','45–49','49–51','51–100']
    plt.rcParams.update({'font.family':['Noto Sans CJK JP', 'DejaVu Sans'], 'axes.unicode_minus':False, 'axes.spines.top':False,
                         'axes.spines.right':False,'font.size':10})
    fig, axes = plt.subplots(1, 3, figsize=(14.6,4.3),layout='constrained',
                             gridspec_kw={'width_ratios':[1.3,1.05,1.15]})
    for dataset, color, marker in zip(order,cols,['s','o','^','D']):
        i=datasets.index.get_loc(dataset)
        axes[0].plot(band_labels,100*share[i],label=names[dataset],c=color,marker=marker,ms=4)
    axes[0].set(title='实测 EEG 电压的频带构成',xlabel='频带（Hz）',
                ylabel='占窗内波动总能量的百分比（%）')
    axes[0].tick_params(axis='x',rotation=45)
    axes[0].legend(frameon=False,fontsize=8)
    subjects=pd.read_csv(out/'subject.csv')
    for i,(dataset,color) in enumerate(zip(order,cols)):
        y=subjects.loc[subjects.dataset==dataset,'nmse'].sort_values().to_numpy()
        axes[1].scatter(i+np.linspace(-.10,.10,len(y)),y,color=color,s=23)
        axes[1].plot([i-.23,i+.23],[y.mean()]*2,color='black')
    axes[1].set(xticks=range(4),xticklabels=[names[d] for d in order],title='验证被试的总电压重建误差',
                ylabel='归一化均方误差（点为被试，横线为库均值）')
    axes[1].tick_params(axis='x',rotation=35)
    gap=pd.read_csv(out/'gap_by_band.csv',index_col=0).fraction_of_nmse_gap*100
    axes[2].barh(['窗均值']+band_labels,gap,color='#ad452f')
    axes[2].invert_yaxis()
    axes[2].set(title='REFED 相对其余三库等权平均的误差增量',xlabel='各频带占总误差差距的百分比（%）',
                ylabel='频带（Hz）')
    fig.savefig(out/'comparison.png',dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    check_spectrum()
    torch.set_num_threads(1)
    plan=runner.read_json(MATRIX/'plan.json')
    JOB=runner.SOURCE/plan['matrix_controls']['EEG_X_continuous']
    run=JOB.parents[1]
    CONFIG=runner.read_json(run/'resolved_config.json')['config']
    manifest,SCALES,_,_=runner._base_training_data(CONFIG,runner.SOURCE)
    ROWS=manifest['rows']
    EXAMPLES=runner._base_training_examples(ROWS,'eeg','validation')
    state=runner.read_json(JOB/'summary.json')
    checkpoint=torch.load(JOB/'checkpoint_step_10000.pt',map_location='cpu',weights_only=False)
    assert state['status']=='completed' and checkpoint['step']==state['completed_step']==10000
    assert checkpoint['signature']==state['signature']
    assert runner.identity(manifest)==plan['manifest_identity']==state['signature']['input']
    assert runner.identity(SCALES)==plan['scales_identity']==state['signature']['scales']
    MODEL=runner._base_training_model(CONFIG,state['signature']['task']).eval()
    MODEL.load_state_dict(checkpoint['model_state'])
    folder=JOB/'evaluation/step_10000'
    reference=runner.read_json(folder/'validation.json')
    with np.load(folder/'validation.npz') as saved:
        np.testing.assert_array_equal(saved['examples'], EXAMPLES)
        EXPECTED=saved['nmse'][:,0,0]
    args.output.mkdir(parents=True,exist_ok=True)
    chunks=[(i,min(i+768,len(EXAMPLES))) for i in range(0,len(EXAMPLES),768)]
    parts=args.output/'blocks'
    parts.mkdir(exist_ok=True)
    frames=[]
    begin=time.monotonic()
    pending=[pair for pair in chunks if not (parts/f'{pair[0]}.npz').exists()]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('fork')) as pool:
        for (start,end),frame in zip(pending,pool.map(replay_block,pending)):
            runner.write_npz(parts/f'{start}.npz', **{k:v.to_numpy() for k,v in frame.items()})
            print(f'{end}/{len(EXAMPLES)} inputs, {time.monotonic()-begin:.1f} s',flush=True)
    for start, end in chunks:
        with np.load(parts/f'{start}.npz') as saved:
            frames.append(pd.DataFrame(dict(saved)))
        assert len(frames[-1]) == end-start
    frame=pd.concat(frames,ignore_index=True)
    stats=base_training_aggregate({k:v.to_numpy() for k,v in frame.items()},EXAMPLES,ROWS)
    report(args.output,stats,reference,frame)
    summarize_frequency_gap(args.output)
