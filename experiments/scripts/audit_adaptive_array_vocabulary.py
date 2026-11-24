"""Necessary waveform/support bounds for a whole-array union of two real vocabularies."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_training_data

METRICS = ['nmse', 'rms_gain', 'band_4_8_abs_db', 'band_8_13_abs_db',
           'band_4_8_db', 'band_8_13_db', 'local_rms_correlation', 'correlation']


def write(path, value):
    def scalar(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=scalar, ensure_ascii=False)+'\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--review', action='store_true')
    args = parser.parse_args()
    config = ROOT / args.config
    cfg = yaml.safe_load(config.read_text())
    if cfg['scope']['splits'] != ['train']:
        raise ValueError('registered training-only table diagnostic required')
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.render:
        render_examples(cfg,out)
        return
    if args.review:
        independent_review(out)
        return
    reference = ROOT / cfg['source']['integer_sum_run']
    word_root = ROOT / cfg['source']['word_audit']
    pop = json.loads((word_root/'population.json').read_text())
    cases = pop['cases']
    with np.load(word_root/'training_attributes.npz', allow_pickle=False) as z:
        examples = z['examples'].copy()
        valid_count = z['valid'].sum(1)
    rule = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, _, _, _ = base_tokenizer_training_data(rule, ROOT)
    rows = manifest['rows']
    if len(cases) != 400 or len(examples) != 17040 or len(pop['subjects']) != 60:
        raise ValueError('original 400/17040/60 support required')
    for c in cases:
        row = rows[c['crop']['row']]
        if (row['split'],row['dataset'],row['subject']) != ('train',c['dataset'],c['subject']):
            raise ValueError('train subject and crop identities changed')
        sl = slice(c['first'], c['first']+c['count'])
        expected = [(c['crop']['row'],c['crop']['start_s'],p) for p in c['crop']['positions']]
        np.testing.assert_array_equal(examples[sl], expected)
    source_hashes = {str(config):sha(config), str(word_root/'population.json'):sha(word_root/'population.json'),
                     str(word_root/'training_attributes.npz'):sha(word_root/'training_attributes.npz')}
    lookup = {tuple(ex):j for j,ex in enumerate(examples)}
    per_position = {}
    for family in ['semantic','high']:
        parts = {m:np.zeros((len(examples),3,1)) for m in METRICS}
        seen = np.zeros(len(examples),bool)
        for dataset in sorted({c['dataset'] for c in cases}):
            path = reference/f'{family}_{dataset}.npz'
            source_hashes[str(path)] = sha(path)
            with np.load(path,allow_pickle=False) as z:
                index = [lookup[tuple(e)] for e in z['examples']]
                if seen[index].any():
                    raise ValueError('repeated metric identity')
                seen[index]=True
                for m in METRICS:
                    values=z['transient_source__'+m]
                    if not np.isfinite(values).all():
                        raise ValueError('finite matched affine endpoints required')
                    parts[m][index]=values
        if not seen.all():
            raise ValueError('missing original score position')
        per_position[family]=parts
    corpus_subjects=defaultdict(set)
    subject_records=defaultdict(set)
    record_crops=defaultdict(list)
    for i,c in enumerate(cases):
        row=rows[c['crop']['row']]
        corpus_subjects[c['dataset']].add(c['subject'])
        subject_records[c['dataset'],c['subject']].add(row['record'])
        record_crops[c['dataset'],c['subject'],row['record']].append(i)
    weights=np.zeros(400)
    for key,indices in record_crops.items():
        dataset,subject,_=key
        weights[indices]=1/(4*len(corpus_subjects[dataset])*len(subject_records[dataset,subject])*len(indices))
    np.testing.assert_allclose(weights.sum(),1,rtol=1e-14)
    metrics={}
    for family,parts in per_position.items():
        metrics[family]={}
        aggregate=base_training_aggregate(parts,examples,rows)['metrics']
        for m,v in parts.items():
            means=np.array([v[c['first']:c['first']+c['count']].mean(0) for c in cases])
            np.testing.assert_allclose(np.einsum('n,nij->ij',weights,means),aggregate[m]['macro'],rtol=2e-14,atol=1e-14)
            metrics[family][m]=means[...,0]
    support={}
    for family in ['semantic','high']:
        path=word_root/f'{family}_transient_readout_errors.npz'
        source_hashes[str(path)] = sha(path)
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['examples'],examples)
            support[family]=np.array([[np.sum(z[b+'_mass'][c['first']:c['first']+c['count']]*valid_count[c['first']:c['first']+c['count']]) for b in 'SO'] for c in cases])/valid_count.sum()
    support_weight=np.array([valid_count[c['first']:c['first']+c['count']].sum() for c in cases])/valid_count.sum()
    A=[];B=[];names=[]
    def upper(metric,component,limit,label,mask=None):
        w=weights if mask is None else weights*np.asarray(mask)*4
        lo=w*metrics['semantic'][metric][:,component]
        delta=w*(metrics['high'][metric][:,component]-metrics['semantic'][metric][:,component])
        A.append(delta);B.append(limit-lo.sum());names.append(label)
    def lower(metric,component,limit,label):
        w=weights
        lo=w*metrics['semantic'][metric][:,component]
        delta=w*(metrics['high'][metric][:,component]-metrics['semantic'][metric][:,component])
        A.append(-delta);B.append(lo.sum()-limit);names.append(label)
    high={m:np.einsum('n,nj->j',weights,v) for m,v in metrics['high'].items()}
    for component,label in enumerate(['X','S','O']):
        upper('nmse',component,high['nmse'][component]*([1.05,1.10,1.10][component]),label+'_NMSE')
        for dataset in sorted(corpus_subjects):
            mask=np.array([c['dataset']==dataset for c in cases])
            ref=float(np.sum(weights[mask]*4*metrics['high']['nmse'][mask,component]))
            upper('nmse',component,1.10*ref,dataset+'_'+label+'_NMSE',mask)
        lower('rms_gain',component,.75,label+'_RMS_lower')
        upper('rms_gain',component,1.25,label+'_RMS_upper')
        for band in ['4_8','8_13']:
            upper('band_'+band+'_abs_db',component,3.0103,label+'_'+band+'_absolute_dB')
            upper('band_'+band+'_db',component,3.0103,label+'_'+band+'_positive_dB')
            lower('band_'+band+'_db',component,-3.0103,label+'_'+band+'_negative_dB')
        lower('local_rms_correlation',component,.5,label+'_local_RMS')
        lower('correlation',component,0,label+'_correlation')
    A=np.array(A);B=np.array(B)
    bounds=[(0,1)]*400
    def solve(objective, a=A, b=B):
        return linprog(objective,A_ub=a,b_ub=b,bounds=bounds,method='highs')
    only=solve(support_weight)
    # Complete mixing is a relaxation. The optimum cannot exceed actual routed-ID support.
    # Append t <= support_S and t <= support_O and maximize the common upper bound.
    sup_lo=support['semantic'].sum(0)
    sup_delta=support['high']-support['semantic']
    joint_A=np.concatenate([np.column_stack([A,np.zeros(len(A))]),np.column_stack([-sup_delta.T,np.ones(2)])])
    joint_B=np.r_[B,sup_lo]
    opt=linprog(np.r_[np.zeros(400),-1],A_ub=joint_A,b_ub=joint_B,bounds=bounds+[(0,1)],method='highs')
    joint=linprog(np.zeros(400),A_ub=np.concatenate([A,-sup_delta.T]),b_ub=np.r_[B,sup_lo-.90],bounds=bounds,method='highs')
    parity=[]
    for fraction in np.linspace(0,1,41):
        a=np.concatenate([A,support_weight[None]])
        b=np.r_[B,fraction]
        result=linprog(np.r_[np.zeros(400),-1],A_ub=np.concatenate([np.column_stack([a,np.zeros(len(a))]),np.column_stack([-sup_delta.T,np.ones(2)])]),b_ub=np.r_[b,sup_lo],bounds=bounds+[(0,1)],method='highs')
        parity.append(dict(maximum_high_patch_fraction=float(fraction),success=result.success,max_min_branch_support_upper=float(result.x[-1]) if result.success else None))
    solution=opt.x[:400] if opt.success else np.ones(400)
    aggregate_selected={m:np.einsum('n,nj->j',weights,metrics['semantic'][m]+solution[:,None]*(metrics['high'][m]-metrics['semantic'][m])) for m in METRICS}
    active=[dict(name=names[j],slack=float(B[j]-A[j]@solution)) for j in range(len(names)) if abs(B[j]-A[j]@solution)<1e-7]
    summary=dict(status='necessary_relaxation_completed',scope='train only, 400 crops, 17040 positions, 60 people',waveform_only_feasible=bool(only.success),minimum_high_patch_fraction_for_waveform=float(only.fun) if only.success else None,
                 joint_necessary_conditions_feasible=bool(joint.success),joint_solver_status=int(joint.status),joint_solver_message=joint.message,
                 maximum_common_branch_support_upper=float(opt.x[-1]) if opt.success else None,minimum_required_support=.9,
                 relaxed_selected_metrics=aggregate_selected,relaxed_high_patch_fraction=float(support_weight@solution),relaxed_high_crop_fraction=float(solution.mean()),active_waveform_constraints=active,
                 proof='Each actual deterministic whole-array selector is included in the [0,1] convex crop relaxation. For union vocabulary IDs with disjoint family names, restricting fit crops never increases a code count or subject count, so the corresponding full-family support bounds above each mixed support. LP infeasibility rules out jointly satisfying only these necessary waveform and support requirements for the two frozen decoder families; semantic error gains have not been assumed.',
                 original_high_metrics=high,source_files=source_hashes,CPU_seconds=time.process_time(),validation_reads=0,confirmation_reads=0,test_reads=0,
                 decision='stop_union_array_family_if_infeasible_else_evaluate_predeclared_X_gate')
    np.savez_compressed(out/'relaxed_bound_arrays.npz',weights=weights,A=A,B=B,support_semantic=support['semantic'],support_high=support['high'],support_weight=support_weight,relaxed_high=solution,**{f+'__'+m:v for f,p in metrics.items() for m,v in p.items()})
    write(out/'summary.json',summary)
    write(out/'support_frontier.json',parity)
    write(out/'constraint_names.json',names)
    write(out/'population.json',pop)
    print(json.dumps({k:v for k,v in summary.items() if k in ['waveform_only_feasible','minimum_high_patch_fraction_for_waveform','joint_necessary_conditions_feasible','maximum_common_branch_support_upper','active_waveform_constraints','CPU_seconds']},ensure_ascii=False))


def independent_review(out):
    """Re-solve retained relaxations with another HiGHS method and save dual bounds."""
    with np.load(out/'relaxed_bound_arrays.npz',allow_pickle=False) as saved:
        z={k:saved[k].copy() for k in saved.files}
    names=json.loads((out/'constraint_names.json').read_text())
    matrix,bound=z['A'],z['B']
    support=z['support_semantic'].sum(0)
    delta=z['support_high']-z['support_semantic']
    modes=[('macro_X_only',[i for i,n in enumerate(names) if n=='X_NMSE']),
           ('macro_X_S_O',[i for i,n in enumerate(names) if n in ['X_NMSE','S_NMSE','O_NMSE']]),
           ('macro_and_dataset_NMSE',[i for i,n in enumerate(names) if n.endswith('_NMSE')]),
           ('all_original_dynamics',list(range(len(names))))]
    reviews=[]
    for mode,keep in modes:
        a=np.concatenate([np.column_stack([matrix[keep],np.zeros(len(keep))]),
                          np.column_stack([-delta.T,np.ones(2)])])
        b=np.r_[bound[keep],support]
        objective=np.r_[np.zeros(400),-1.]
        result=linprog(objective,A_ub=a,b_ub=b,bounds=[(0,1)]*401,method='highs-ipm')
        if not result.success:
            raise ValueError('independent relaxed optimizer failed')
        y=result.ineqlin.marginals
        lower=result.lower.marginals
        upper=result.upper.marginals
        dual=b@y+upper.sum()
        stationarity=float(np.abs(objective-a.T@y-lower-upper).max())
        primal_error=float(max(0,(a@result.x-b).max()))
        gap=float(abs(result.fun-dual))
        if max(stationarity,primal_error,gap)>1e-10 or np.any(y>1e-10) or np.any(upper>1e-10) or np.any(lower<-1e-10):
            raise ValueError('dual feasibility and objective certificate did not reproduce')
        reviews.append(dict(mode=mode,constraints=len(keep),support_upper=float(result.x[-1]),
                            dual_upper=float(-dual),max_primal_violation=primal_error,
                            stationarity_max=stationarity,duality_gap=gap,method='highs-ipm'))
        np.savez_compressed(out/(mode+'_dual.npz'),inequality_matrix=a,inequality_bound=b,objective=objective,
                            primal=result.x,inequality_multipliers=y,lower_multipliers=lower,upper_multipliers=upper)
    summary=json.loads((out/'summary.json').read_text())
    np.testing.assert_allclose(reviews[-1]['support_upper'],summary['maximum_common_branch_support_upper'],atol=1e-12,rtol=1e-12)
    weights=z['weights']
    choice=np.mean(z['high__nmse'][:,1:],axis=1)<np.mean(z['semantic__nmse'][:,1:],axis=1)
    metric={m:np.einsum('n,nj->j',weights,np.where(choice[:,None],z['high__'+m],z['semantic__'+m])) for m in METRICS}
    oracle=dict(choice='whole-array true-target minimum S/O NMSE per crop, 1=high',high_crop_count=int(choice.sum()),
                high_crop_fraction=float(choice.mean()),high_patch_mass=float(z['support_weight']@choice),
                waveform_metrics=metric,pure_support_upper=support+delta.T@choice)
    np.savez_compressed(out/'true_target_oracle_choices.npz',high=choice)
    write(out/'independent_lp_review.json',reviews)
    write(out/'true_target_oracle.json',oracle)
    print('Independent dual certificates agree; weakest-constraint support upper',reviews[0]['support_upper'])


def render_examples(cfg, out):
    """Replay existing predeclared train gallery cases without opening new signals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.tokenizers.eeg_transient_gain import load_model as load_source
    from src.tokenizers.eeg_component_transfer import IntegerSumSourceDecoder
    from src.tokenizers.spatial_waveform_dictionary import load_model as load_decoder

    reference=ROOT/cfg['source']['integer_sum_run']
    hardcfg=yaml.safe_load((ROOT/cfg['source']['hard_contract']).read_text())
    run=ROOT/hardcfg['data']['run']
    display=reference/'training_examples'
    cases=json.loads((display/'waveform_examples.json').read_text())['examples']
    population=json.loads((out/'population.json').read_text())['cases']
    bycrop={(c['crop']['row'],c['crop']['start_s']):c for c in population}
    root=ROOT/cfg['source']['word_audit']
    positions=json.loads((root/'population.json').read_text())['native_positions']
    ids={}
    for dataset in positions:
        for path in sorted((reference.parent/'hard_as_v1/semantic'/dataset).glob('ids_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                for j,key in enumerate(z['keys']):
                    ids[tuple(map(int,key))]={b:z['transient_'+b][j:j+1].copy() for b in 'SO'}
    semantic=load_decoder(run/hardcfg['families']['semantic']['decoder'])
    high=load_decoder(run/hardcfg['families']['high']['decoder'])
    source={}
    records=[]
    waves=np.load(display/'waves.npz',allow_pickle=False)
    outdir=out/'waveforms';outdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for i,case in enumerate(cases):
        if not ({'fixed_identity_first','candidate_max'} & set(case['selection'])):
            continue
        key=case['row'],case['start_s'];native=bycrop[key]
        fold=native['fold'];dataset=case['dataset'];source_key=dataset,fold
        if source_key not in source:
            source[source_key]=load_source(run/hardcfg['data']['source_model_run']/f'{dataset}_fold{fold}_transient.npz')
        decoder=IntegerSumSourceDecoder(semantic,source[source_key])
        sent=ids[key[0],key[1],fold]
        # These are the complete emitted sequences for the low endpoint.
        values=decoder.decode_ids(sent,dataset)
        site={v.upper():j for j,v in enumerate(positions[dataset])}[case['position_name'].upper()]
        target={b:waves[k][i].reshape(6000)*case['scale'] for b,k in [('X','x'),('S','s'),('O','o')]}
        pred_high={b:waves[k][i].reshape(6000)*case['scale'] for b,k in [('X','prediction_sum'),('S','prediction_S'),('O','prediction_O')]}
        pred_low={b:values[k][0,site].reshape(6000)*case['scale'] for b,k in [('X','sum'),('S','S'),('O','O')]}
        time=np.arange(6000)/200
        fig,axes=plt.subplots(3,3,figsize=(15,8.2),layout='constrained')
        for row,b in enumerate(['X','S','O']):
            y=target[b]
            lower=min(y.min(),pred_high[b].min(),pred_low[b].min())
            upper=max(y.max(),pred_high[b].max(),pred_low[b].max())
            margin=(upper-lower)*.08
            for col,(left,right) in enumerate([(0,30),(14,16),(13.5,14.5)]):
                ax=axes[row,col]
                ax.plot(time,y,color='#1c4666',lw=.7,label='Target')
                ax.plot(time,pred_high[b],color='#d77935',lw=.75,ls='--',label='High union endpoint')
                ax.plot(time,pred_low[b],color='#2e937c',lw=.75,ls=':',label='Low union endpoint')
                for t in np.arange(0,32,2):ax.axvline(t,color='#bbc2c8',lw=.5,ls=':')
                ax.fill_between(time,lower-margin,upper+margin,where=~waves['mask'][i].reshape(6000).astype(bool),color='#d3d3d3',alpha=.5)
                ax.set(xlim=(left,right),ylim=(lower-margin,upper+margin),xlabel='Time within visible 30 s (s)')
                if col==0:ax.set_ylabel(b+' / '+case['source_unit'])
                if row==0:ax.set_title(['Full 30 s','Fixed 2 s patch','Fixed token boundary'][col])
        axes[0,0].legend(fontsize=8)
        title=f"Whole-array union endpoints | {dataset} | {case['subject']} | {case['record']} | {case['position_name']} | crop +{case['start_s']}s"
        fig.suptitle(title,fontsize=11)
        path=outdir/f'{i:03d}_{dataset}_{case["subject"]}_{case["position_name"]}.png'
        fig.savefig(path,dpi=220,facecolor='white');plt.close(fig)
        records.append(dict(index=i,identity=case,complete_low_integer_shape={b:list(v.shape) for b,v in sent.items()},figure=str(path.relative_to(out)),no_original_signal_read=True))
    frontier=json.loads((out/'support_frontier.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    reviewed=json.loads((out/'independent_lp_review.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.8),layout='constrained')
    valid=[r for r in frontier if r['success']]
    axes[0].plot([r['maximum_high_patch_fraction'] for r in valid],[100*r['max_min_branch_support_upper'] for r in valid],color='#2e7884',lw=2,marker='o',ms=3)
    axes[0].axhline(90,color='#ad4b3b',ls='--',label='Required 90%')
    axes[0].set(xlabel='Maximum high-family patch fraction',ylabel='Optimistic common support bound (%)',ylim=(55,100),title='All waveform and dynamics requirements')
    axes[0].legend()
    names=['Macro X only','Macro X/S/O','Dataset NMSE','All dynamics']
    axes[1].barh(names,[100*r['support_upper'] for r in reviewed],color=['#79a8b0','#79a8b0','#4a8893','#2e7884'])
    for j,row in enumerate(reviewed):axes[1].text(100*row['support_upper']+.6,j,f'{100*row["support_upper"]:.2f}%',va='center')
    axes[1].axvline(90,color='#ad4b3b',ls='--');axes[1].set(xlim=(0,100),xlabel='Optimistic common support bound (%)',title='Even the weakest total-voltage limit fails support')
    fig.savefig(out/'support_waveform_bound.png',dpi=240,facecolor='white');plt.close(fig)
    write(out/'figure_manifest.json',dict(status='rendered_pending_actual_image_review',figures=records,bound_figure='support_waveform_bound.png',candidate_exists=False))
    print('Rendered',len(records),'retained examples and one bound figure')


if __name__=='__main__':
    main()
