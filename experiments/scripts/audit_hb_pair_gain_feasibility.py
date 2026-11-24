"""Train-only oracle geometry for signed scaling of an already frozen Hb delta."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import yaml
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/scripts'))
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.analysis.tokenizer_shape_diagnostics import training_weights
from run_hb_native_id_readout import digest, write, fold_assignment


def squared_curve(base, direction, target, valid, variance):
    count = valid.sum(1)
    mean = lambda a: np.where(valid, a, 0.).sum(1) / np.maximum(count, 1)
    error = base - target
    return np.stack([mean(error**2), 2*mean(error*direction), mean(direction**2)], -1) / variance[..., None]


def evaluate(coefficients, gain):
    return coefficients[..., 0] + gain*coefficients[..., 1] + gain*gain*coefficients[..., 2]


def quadratic_improvement_interval(q):
    """Exact open interval on [-1,1] where a convex quadratic beats q(0)."""
    _, b, a = map(float, q)
    if a <= 0:
        return [-1., 0.] if b > 0 else ([0., 1.] if b < 0 else [])
    root = -b/a
    left, right = max(-1., min(0., root)), min(1., max(0., root))
    return [left, right] if right > left else []


def intersect(left, right):
    return [[max(a,c), min(b,d)] for a,b in left for c,d in right if min(b,d)>=max(a,c)]


def quadratic_below(q, threshold):
    c,b,a = np.asarray(q,float) - [threshold,0,0]
    if a < 1e-25:
        if abs(b)<1e-25:return [[-1.,1.]] if c<=0 else []
        root=-c/b
        return intersect([[-1.,1.]],[[-np.inf,root] if b>0 else [root,np.inf]])
    disc=b*b-4*a*c
    if disc<0:return []
    roots=sorted([(-b-np.sqrt(disc))/(2*a),(-b+np.sqrt(disc))/(2*a)])
    return intersect([[-1.,1.]],[roots])


def rms_intervals(energy_coefficients, weights, lower, upper):
    curve=lambda z:float(weights@np.sqrt(np.maximum(evaluate(energy_coefficients,z),0)))
    minimum=minimize_scalar(curve,bounds=(-1.,1.),method='bounded',options={'xatol':1e-13}).x
    candidates=[-1.,1.,float(minimum)]
    for threshold in (lower,upper):
        for a,b in [(-1.,minimum),(minimum,1.)]:
            if (curve(a)-threshold)*(curve(b)-threshold)<0:
                candidates.append(float(brentq(lambda z:curve(z)-threshold,a,b,xtol=1e-14)))
    candidates=sorted(set(candidates));intervals=[]
    for a,b in zip(candidates[:-1],candidates[1:]):
        if lower<=curve((a+b)/2)<=upper:intervals.append([a,b])
    return intervals,curve


def tail_necessary_intervals(coefficients, threshold, quantile):
    """Exact optimistic intervals from the lower order statistic of NumPy quantile.

    Linear-interpolated quantile >= lower order statistic. This condition is
    necessary, so emptiness provides a certificate without a gain grid.
    """
    required=int(np.floor((len(coefficients)-1)*quantile))+1
    events={-1.:0,1.:0}
    for q in coefficients:
        for lo,hi in quadratic_below(q,threshold):
            if hi>lo:
                events[lo]=events.get(lo,0)+1;events[hi]=events.get(hi,0)-1
    points=sorted(events);count=0;result=[]
    for a,b in zip(points[:-1],points[1:]):
        count+=events[a]
        if count>=required:
            if result and result[-1][1]==a:result[-1][1]=b
            else:result.append([a,b])
    return result


def aggregate_curve(value, examples, rows):
    finite=np.isfinite(value).all(-1)
    weights=training_weights(examples[finite],rows)
    return np.einsum('n,nk->k',weights,value[finite])


def correlation_moments(p,d,y,valid):
    mean=lambda z:np.divide(np.where(valid,z,0.).sum(1),valid.sum(1),out=np.full(len(y),np.nan),where=valid.sum(1)>0)
    pc,dc,yc=[z-mean(z)[:,None] for z in (p,d,y)]
    return np.stack([mean(yc*pc),mean(yc*dc),mean(yc*yc),mean(pc*pc),2*mean(pc*dc),mean(dc*dc)],-1)


def correlations(moments,gains):
    gains=np.asarray(gains)[:,None]
    denominator=moments[:,2]*(moments[:,3]+gains*moments[:,4]+gains*gains*moments[:,5])
    return np.divide(moments[:,0]+gains*moments[:,1],np.sqrt(np.maximum(denominator,0)),out=np.full(denominator.shape,np.nan),where=denominator>1e-20)


def hierarchical_mean(values,examples,rows):
    finite=np.isfinite(values)
    if np.all(finite==finite[0]):
        use=finite[0];w=training_weights(examples[use],rows)
        return values[:,use]@w
    return np.array([v[np.isfinite(v)]@training_weights(examples[np.isfinite(v)],rows) for v in values])


def oracle_search(out,source,old_cfg,examples,rows,subjects,folds,original,delta,bank,coeff,scales):
    from run_hb_native_id_readout import audit,pair_decision,pair_peak_error
    n=len(examples);valid=bank['mask'].reshape(n,300,2)
    with np.load(ROOT/old_cfg['source']['bank']) as f:
        fullbank={k:f[k].copy() for k in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo')}
        g1={b:f['original_'+b].astype(float) for b in ('S','O','sum')}
    for k in ('vx','vs','vo','dvx','dvs','dvo'):
        fullbank[k]=np.broadcast_to(np.tile(fullbank[k],(1,20))[:,None,:],fullbank['x'].shape)
    truth={b:fullbank[b.lower()].reshape(n,300,2).astype(float) for b in 'SO'}
    with np.load(source/'diagnostic_predictions.npz') as f:total=f['source_sum'].copy()
    baseline={b:original[b].reshape(n,15,40) for b in 'SO'}|{'sum':total}
    source_summary=json.loads((source/'source111_CV_summary.json').read_text());g1_summary=json.loads((source/'G1_summary.json').read_text())
    with np.load(source/'source111_CV_metrics.npz') as f:source_metrics={k:f[k].copy() for k in f.files if k!='examples'}
    zero=np.zeros_like(delta)
    gm=[]
    for b,k in [('sum','x'),('S','s'),('O','o')]:
        target=fullbank[k].reshape(n,300,2).astype(float);variance=fullbank['v'+k][:,0,:2]
        gm.append(squared_curve(g1[b].reshape(n,300,2),zero,target,valid,variance)[...,0])
    g1metrics={'nmse':np.stack(gm,1)}
    full=valid.reshape(n,15,20,2).all(2)
    t=(np.arange(20)-9.5)/10;quad=t*t-np.mean(t*t)
    corr={}
    for field,basis in [('slope',t/(t@t)),('curvature',2*quad/(quad@quad))]:
        value=[]
        for b,sign in [('S',-1),('O',1)]:
            p,d,y=[np.einsum('nptc,t->npc',v.reshape(n,15,20,2),basis) for v in (original[b],sign*delta,truth[b])]
            value.append(np.stack([correlation_moments(p[:,:,c],d[:,:,c],y[:,:,c],full[:,:,c]) for c in range(2)],1))
        corr[field]=np.stack(value,1)
        for gain,name in [(0,'source111_CV_metrics.npz'),(1,'window__ridge4096_metrics.npz')]:
            calculated=np.stack([np.stack([correlations(corr[field][:,b,c], [gain])[0] for c in range(2)],1) for b in range(2)],1)
            with np.load(source/name) as f:np.testing.assert_allclose(calculated,f[field+'_correlation'][:,1:],rtol=2e-10,atol=2e-10,equal_nan=True)
    constraints_doc=json.loads((out/'necessary_constraints.json').read_text())
    grids={};selected={};gain_matrix=np.zeros((n,2));rule=old_cfg['selection']
    for dataset,records in constraints_doc['datasets'].items():
        use=np.flatnonzero(subjects[:,0]==dataset);selected[dataset]=[]
        for color,record in enumerate(records):
            q=np.asarray(record['joint_coefficients']);domain=record['necessary_domain']
            gains=np.unique(np.r_[np.linspace(-1,1,4001),0.,record['best_optimistic_gain']])
            gains=gains[np.array([any(a<=g<=b for a,b in domain) for g in gains])]
            accepted=np.ones(len(gains),bool);details={}
            for start in range(0,len(gains),64):
                take=slice(start,start+64);gs=gains[take]
                good=np.ones(len(gs),bool)
                for branch,b in enumerate('SO'):
                    values=evaluate(coeff[use,branch,color],gs[:,None])
                    for quantile in rule['difficulty_quantiles']:
                        threshold=record['constraints'][b+f'_tail_{quantile}_necessary']['threshold']
                        good&=np.quantile(values,quantile,axis=1)<=threshold+1e-15
                    for field in ('slope','curvature'):
                        means=hierarchical_mean(correlations(corr[field][use,branch,color],gs),examples[use],rows)
                        threshold=source_summary['metrics'][field+'_correlation']['by_dataset'][dataset][branch+1][color]-rule['slope_curvature_correlation_drop_max']
                        good&=means>=threshold
                accepted[take]=good
            allowed=gains[accepted]
            best=float(allowed[np.argmin(evaluate(q,allowed))]) if len(allowed) else None
            if best is not None:gain_matrix[use,color]=best
            selected[dataset].append(best)
            grids[dataset+'_'+str(color)]=dict(color=record['color'],grid_points=len(gains),accepted_points=int(accepted.sum()),
                accepted_ranges_sampled=sampled_ranges(gains,accepted),best=best,grid_step=.0005,
                interpretation='point search only; empty grid alone is not an infeasibility proof')
    write(out/'oracle_gain_grid.json',dict(status='diagnostic_only_not_gain_fitting',selected=selected,grids=grids,validation_reads=0))
    print('oracle_point',json.dumps(selected),flush=True)
    if any(v is None for values in selected.values() for v in values):return
    prediction={b:(original[b]+sign*delta*gain_matrix[:,None]).reshape(n,15,40) for b,sign in [('S',-1),('O',1)]}|{'sum':total.copy()}
    measured,summary=audit(prediction,fullbank,examples,rows,scales)
    peaks=pair_peak_error(prediction,fullbank,examples,rows);base_peaks=pair_peak_error(baseline,fullbank,examples,rows)
    decision=pair_decision(summary,source_summary,g1_summary,measured['metrics'],source_metrics,g1metrics,examples,rows,folds,old_cfg,peaks,base_peaks)
    write(out/'oracle_point_original_acceptance.json',dict(status='oracle_diagnostic_only_not_fitted_candidate',gains=selected,decision=decision,summary=summary))
    print('oracle_eligible',decision['eligible'],'failed',[k for k,v in decision['checks'].items() if not v],flush=True)


def sampled_ranges(values,accepted):
    ranges=[];start=None
    for i,ok in enumerate(accepted):
        if ok and start is None:start=i
        if start is not None and (not ok or i==len(accepted)-1):
            end=i if ok else i-1;ranges.append([float(values[start]),float(values[end])]);start=None
    return ranges


def render_geometry(out,gain_run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geometry=json.loads((out/'quadratic_geometry.json').read_text())
    oracle=json.loads((out/'oracle_gain_grid.json').read_text())
    fits=json.loads((gain_run/'gain_fit_scope.json').read_text())['folds']
    fig,axes=plt.subplots(4,2,figsize=(13,14),layout='constrained');grid=np.linspace(-1,1,1001)
    for row,(dataset,entry) in enumerate(geometry['datasets'].items()):
        for color in range(2):
            ax=axes[row,color];q=np.asarray(entry['joint_color_coefficients'])[color]
            ax.plot(grid,100*(1-evaluate(q,grid)/q[0]),color='#247db3',lw=1.3,label='allfold color joint gain')
            for fold in entry['folds']:
                fq=np.asarray(fold['joint_color_coefficients'])[color]
                ax.plot(grid,100*(1-evaluate(fq,grid)/fq[0]),color='.6',lw=.6,alpha=.7)
                gain=fits[fold['fold']]['gains']['dataset'][dataset]['gain'][color]
                ax.scatter(gain,100*(1-evaluate(fq,gain)/fq[0]),marker='x',color='#c44e52',s=35,label='fit-only gain on its held fold' if fold['fold']==0 else None)
            for a,b in oracle['grids'][dataset+'_'+str(color)]['accepted_ranges_sampled']:
                ax.axvspan(a,b,color='#55a868',alpha=.15)
            point=oracle['selected'][dataset][color]
            ax.scatter(point,100*(1-evaluate(q,point)/q[0]),marker='*',s=90,color='black',label='oracle diagnostic point')
            ax.axhline(0,color='black',ls=':',lw=.7);ax.axvline(0,color='black',ls=':',lw=.7);ax.axvline(1,color='#d98732',ls='--',lw=.7)
            ax.set(xlim=(-1,1),title=dataset+' / '+['HbO','HbR'][color],xlabel='signed multiplier of frozen delta',ylabel='joint S/O error reduction (%)')
            ax.title.set_fontsize(10);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Frozen Hb delta: original15360 train windows,60 subjects,4 outer folds\nGreen: sampled local waveform-constraint domains; gray: individual-fold color loss curves\nZero retains frozen111; one is the rejected original delta. Oracle uses held truth only for feasibility.',fontsize=12)
    fig.savefig(out/'signed_gain_geometry_v1.png',dpi=180);plt.close(fig)


def constraints(out,source,old_cfg,examples,rows,subjects,folds,original,delta,bank,coeff):
    from src.analysis.dual_component_tokenizer import base_training_aggregate
    from run_hb_native_id_readout import audit,pair_decision,pair_peak_error
    rule=old_cfg['selection'];n=len(examples);valid=bank['mask'].reshape(n,300,2)
    count=valid.sum(1);mean=lambda a:np.where(valid,a,0.).sum(1)/np.maximum(count,1)
    source_summary=json.loads((source/'source111_CV_summary.json').read_text())
    g1_summary=json.loads((source/'G1_summary.json').read_text())
    with np.load(source/'source111_CV_metrics.npz') as f:source_metrics={k:f[k].copy() for k in f.files if k!='examples'}
    # Only the native training bank is opened. No screen or development arrays.
    with np.load(ROOT/old_cfg['source']['bank']) as f:
        truth={b:f[b.lower()].reshape(n,300,2).astype(np.float64) for b in 'SO'}
        g1={b:f['original_'+b].reshape(n,300,2).astype(np.float64) for b in 'SO'}
        dv={b:f['dv'+b.lower()].copy() for b in 'SO'}
        x=f['x'].copy();vx=f['vx'].copy();dvx=f['dvx'].copy()
    zero=np.zeros_like(delta)
    g1nmse=np.stack([squared_curve(g1[b],zero,truth[b],valid,bank['v'+b.lower()])[...,0] for b in 'SO'],1)
    peak=np.argmax(np.where(valid,abs(truth['S']),-np.inf),axis=1)
    ii=np.arange(n)[:,None];cc=np.arange(2)[None,:]
    se=(original['S']-truth['S'])[ii,peak,cc];sd=-delta[ii,peak,cc]
    peak_q=np.stack([se**2,2*se*sd,sd**2],-1)/bank['vs'][...,None]
    energy=[];boundary=[]
    edges=np.arange(1,15)*20;ev=valid[:,edges]&valid[:,edges-1];ec=ev.sum(1)
    for b,sign in [('S',-1),('O',1)]:
        p,d,y=original[b],sign*delta,truth[b]
        denominator=mean(y*y)
        en=np.stack([mean(p*p),2*mean(p*d),mean(d*d)],-1)
        energy.append(np.divide(en,denominator[...,None],out=np.full_like(en,np.nan),where=denominator[...,None]>0))
        error=(p-y)[:,edges]-(p-y)[:,edges-1];change=d[:,edges]-d[:,edges-1]
        moment=lambda a:np.divide(np.where(ev,a,0.).sum(1),ec,out=np.full(ec.shape,np.nan),where=ec>0)
        boundary.append(np.stack([moment(error**2),2*moment(error*change),moment(change**2)],-1)/dv[b][...,None])
    energy=np.stack(energy,1);boundary=np.stack(boundary,1)
    np.testing.assert_allclose(coeff[...,0],source_metrics['nmse'][:,1:],rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(np.sqrt(energy[...,0]),source_metrics['rms_gain'][:,1:],rtol=2e-13,atol=2e-13,equal_nan=True)
    np.testing.assert_allclose(boundary[...,0],source_metrics['boundary_jump_difference_nmse'][:,1:],rtol=2e-12,atol=2e-12,equal_nan=True)
    datasets=sorted(set(subjects[:,0]));output={};gain_matrix=np.zeros((n,2));optimistic_joint=[]
    for dataset in datasets:
        use=np.flatnonzero(subjects[:,0]==dataset);record=[]
        source_nmse=np.asarray(source_summary['metrics']['nmse']['by_dataset'][dataset])[1:]
        g1_nmse=np.asarray(g1_summary['metrics']['nmse']['by_dataset'][dataset])[1:]
        for color in range(2):
            domain=[[-1.,1.]];details={}
            q=aggregate_curve(coeff[use,:,color].sum(1),examples[use],rows)
            for branch,b in enumerate('SO'):
                component=aggregate_curve(coeff[use,branch,color],examples[use],rows)
                cap=min(rule['dataset_component_color_ratio_to_current_max']*source_nmse[branch,color],rule['dataset_component_color_ratio_to_G1_max']*g1_nmse[branch,color])
                allow=quadratic_below(component,cap);details[b+'_component_NMSE']=allow;domain=intersect(domain,allow)
                en=energy[use,branch,color];finite=np.isfinite(en).all(-1);w=training_weights(examples[use[finite]],rows)
                old=float(w@np.sqrt(en[finite,0]));lo,hi=rule['overall_RMS_interval']
                if not lo<=old<=hi:lo,hi=1-abs(old-1)-.01,1+abs(old-1)+.01
                allow,curve=rms_intervals(en[finite],w,lo,hi);details[b+'_RMS']=dict(source=old,lower=lo,upper=hi,intervals=allow)
                domain=intersect(domain,allow)
                boundary_q=aggregate_curve(boundary[use,branch,color],examples[use],rows)
                allow=quadratic_below(boundary_q,max(4.,1.05*boundary_q[0]));details[b+'_boundary']=allow;domain=intersect(domain,allow)
                for quantile in rule['difficulty_quantiles']:
                    old=np.quantile(coeff[use,branch,color,0],quantile);ref=np.quantile(g1nmse[use,branch,color],quantile)
                    cap=min(rule['difficulty_source_ratio_max']*old,max(rule['difficulty_G1_ratio_max']*ref,old))
                    allow=tail_necessary_intervals(coeff[use,branch,color],cap,quantile)
                    details[b+f'_tail_{quantile}_necessary']=dict(threshold=float(cap),intervals=allow)
                    domain=intersect(domain,allow)
            pq=aggregate_curve(peak_q[use,color],examples[use],rows)
            allow=quadratic_below(pq,rule['S_peak_abs_amplitude_error_ratio_max']*pq[0]+1e-15)
            details['S_true_peak']=allow;domain=intersect(domain,allow)
            candidates=[float(np.clip(-q[1]/(2*q[2]),a,b)) for a,b in domain]
            best=min(candidates,key=lambda z:evaluate(q,z)) if candidates else None
            value=float(evaluate(q,best)) if best is not None else float('inf')
            if best is not None:gain_matrix[use,color]=best
            record.append(dict(color=['HbO','HbR'][color],constraints=details,necessary_domain=domain,
                best_optimistic_gain=best,source_joint=float(q[0]),best_optimistic_joint=value,relative_gain=1-value/q[0],joint_coefficients=q))
            optimistic_joint.append((q[0],value))
        output[dataset]=record
    baseline=sum(v[0] for v in optimistic_joint);optimistic=sum(v[1] for v in optimistic_joint)
    write(out/'necessary_constraints.json',dict(status='oracle_only_necessary_not_full_admission',datasets=output,
        maximum_possible_macro_joint_relative_gain=1-optimistic/baseline,
        required_macro_joint_relative_gain=rule['minimum_joint_relative_improvement'],
        exact_constraints=['dataset_component_NMSE','dataset_RMS','true_S_peak','boundary'],
        optimistic_constraints=['tails_lower_order_statistic_necessary_only'],
        not_yet_applied=['slope_curvature_correlation','macro_component_caps','macro_RMS','subject_and_fold_repeat','macro_tails'],
        validation_reads=0,CPU_seconds=time.process_time()))
    print('necessary_domains',json.dumps({d:[dict(color=v['color'],domain=v['necessary_domain'],best=v['best_optimistic_gain'],gain=v['relative_gain']) for v in entries] for d,entries in output.items()}),flush=True)
    print('optimistic_macro_gain',1-optimistic/baseline,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--constraints', action='store_true')
    parser.add_argument('--oracle-search', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--gain-run',type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['schema'] != 'hb_pair_gain_feasibility_v1' or cfg['data']['allowed_splits'] != ['train']:
        raise ValueError('train-only signed-gain diagnostic contract')
    if args.render_only:
        if args.gain_run is None:raise ValueError('completed gain run required')
        return render_geometry(args.output,args.gain_run)
    resource.setrlimit(resource.RLIMIT_CPU, (cfg['resources']['CPU_seconds'], cfg['resources']['CPU_seconds']+1))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'quadratic_geometry.json').exists() and not (args.constraints or args.oracle_search):
        raise FileExistsError('preserve completed diagnostic')
    source = ROOT/cfg['source']['pair_run']
    old_cfg = json.loads((source/'resolved_config.json').read_text())['config']
    manifest, scales, _, _ = base_tokenizer_training_data(old_cfg, ROOT)
    rows = manifest['rows']
    prediction_path = source/'diagnostic_predictions.npz'
    bank_path = ROOT/old_cfg['source']['bank']
    if digest(bank_path) != old_cfg['source']['bank_sha256']:
        raise ValueError('frozen train bank SHA')
    with np.load(prediction_path) as f:
        ex = f['examples'].copy()
        if len(ex) != 15360 or any(rows[int(i)]['split'] != 'train' for i, _, _ in ex):
            raise ValueError('exact train triples before waveform arrays')
        original = {b: f['source_'+b].reshape(len(ex), 300, 2).copy() for b in 'SO'}
        delta = f['O'].reshape(len(ex), 300, 2) - original['O']
        np.testing.assert_allclose(f['S'].reshape(len(ex), 300, 2), original['S']-delta, atol=2e-15, rtol=1e-13)
    with np.load(bank_path) as f:
        np.testing.assert_array_equal(f['examples'], ex)
        bank = {k: f[k].copy() for k in ('s','o','mask','vs','vo','dvs','dvo')}
    valid = bank['mask'].reshape(len(ex), 300, 2)
    coeff = np.stack([squared_curve(original[b], sign*delta, bank[b.lower()].reshape(len(ex), 300, 2), valid, bank['v'+b.lower()])
                      for b, sign in [('S',-1),('O',1)]], 1)
    with np.load(source/'source111_CV_metrics.npz') as f:
        np.testing.assert_array_equal(f['examples'], ex)
        np.testing.assert_allclose(coeff[...,0], f['nmse'][:,1:], rtol=2e-13, atol=2e-13)
    with np.load(source/'window__ridge4096_metrics.npz') as f:
        np.testing.assert_array_equal(f['examples'], ex)
        np.testing.assert_allclose(evaluate(coeff,1), f['nmse'][:,1:], rtol=2e-13, atol=2e-13)
    folds = np.load(ROOT/old_cfg['source']['native_run']/'training_folds.npy')
    np.testing.assert_array_equal(folds, fold_assignment(ex, rows, old_cfg))
    subjects = np.array([(rows[int(i)]['dataset'],rows[int(i)]['subject']) for i,_,_ in ex])
    assert len(set(map(tuple,subjects))) == 60
    if args.oracle_search:
        return oracle_search(out,source,old_cfg,ex,rows,subjects,folds,original,delta,bank,coeff,scales)
    if args.constraints:
        return constraints(out,source,old_cfg,ex,rows,subjects,folds,original,delta,bank,coeff)
    datasets = sorted(set(subjects[:,0]))
    result = {}
    for dataset in datasets:
        use = np.flatnonzero(subjects[:,0] == dataset)
        weights = training_weights(ex[use],rows)
        q = np.einsum('n,nbck->bck',weights,coeff[use])
        joint = q.sum(0)
        item = dict(component_coefficients=q, joint_color_coefficients=joint,
                    color_oracle_optimum=np.clip(-joint[:,1]/(2*joint[:,2]),-1,1),
                    color_improvement_intervals=[quadratic_improvement_interval(v) for v in joint],folds=[])
        for fold in range(4):
            take = use[folds[use] == fold]
            w = training_weights(ex[take],rows)
            fq = np.einsum('n,nbck->bck',w,coeff[take]).sum(0)
            item['folds'].append(dict(fold=fold,joint_color_coefficients=fq,
                color_improvement_intervals=[quadratic_improvement_interval(v) for v in fq],
                joint_gain_at_one=float(1-evaluate(fq,1).sum()/fq[:,0].sum())))
        result[dataset] = item
    np.savez_compressed(out/'quadratic_coefficients.npz',examples=ex,fold=folds,subjects=subjects,NMSE_coefficients=coeff)
    write(out/'quadratic_geometry.json',dict(status='oracle_geometry_only_not_fitted_model',coefficient_order=['constant','linear','quadratic'],
        datasets=result,scope='same15360_60train_4fold',source_prediction_SHA256=digest(prediction_path),source_bank_SHA256=digest(bank_path),
        source_exposure=old_cfg['training']['exposure'],validation_reads=0,confirmation_reads=0,test_reads=0,CPU_seconds=time.process_time()))
    print(json.dumps({d:dict(opt=np.asarray(v['color_oracle_optimum']).tolist(),gain_one=[f['joint_gain_at_one'] for f in v['folds']]) for d,v in result.items()}),flush=True)


if __name__ == '__main__':
    main()
