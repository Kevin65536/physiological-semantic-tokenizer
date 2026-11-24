"""Falsify current-patch forward gains before paying for composite atomic IDs."""
import argparse
from collections import OrderedDict,defaultdict
from concurrent.futures import ProcessPoolExecutor
import fcntl,hashlib,json,multiprocessing,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/scripts'))
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from src.tokenizers.forward_gain_dictionary import projection_gains,decode_scaled_atoms
from src.tokenizers.spatial_waveform_dictionary import load_model
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_joint_eeg,base_tokenizer_batch
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,base_training_aggregate
from evaluate_joint_tokenizer import input_scales,write
STATE={}
ARMS=('gain1','X_continuous_gain','target_continuous_gain')


def validate(c):
    if (c['schema']!='forward_gain_atomic_diagnostic_v1' or c['source']['family']!='semantic'
        or c['data']['allowed_splits']!=['train'] or any(c['data'][s]!='forbidden' for s in ['validation_access','confirmation_access','test_access'])
        or c['resources']['workers']>4 or c['resources']['maximum_threads']>4):
        raise ValueError('frozen train-only forward atomic gain contract required')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget():
    unit=os.environ.get('TOKENIZER_FORWARD_GAIN_UNIT')
    used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    used+=float(os.environ.get('TOKENIZER_FORWARD_GAIN_PRIOR_CPU','0'))
    if used>=STATE['cfg']['resources']['stop_CPU_seconds']:raise RuntimeError('bounded forward-gain CPU budget reached')
    return used


def dataset_worker(dataset):
    with threadpool_limits(1):
        cfg,hard,run,rows,scales,folds,out,gallery=(STATE[k] for k in ['cfg','hard','run','rows','scales','folds','out','gallery'])
        root=run/cfg['source']['hard_run'];spec=hard['families']['semantic'];model=load_model(run/spec['decoder']);names=model.metadata['positions'][dataset]
        if model.codebook_size!=512 or model.readout or model.metadata.get('joint_spatial_inputs'):raise ValueError('frozen actual semantic separable K512 required')
        native={n.upper():i for i,n in enumerate(names)};sources={f:load_source(run/hard['data']['source_model_run']/f'{dataset}_fold{f}_transient.npz') for f in range(4)}
        people=set(folds[dataset])
        for fold,source in sources.items():
            if (source.metadata['held_fold']!=fold or set(source.metadata['fit_subjects'])!={s for s in people if folds[dataset][s]!=fold}
                or source.metadata['input_identity']!=model.metadata['input_identity'] or source.metadata['scales_identity']!=model.metadata['scales_identity']):raise ValueError('fold source ancestry before arrays')
        dest=out/dataset;dest.mkdir(exist_ok=True);cache=OrderedDict();plots={};plot_meta=[];gain_parts=defaultdict(list);allblocks=[];identities=[]
        for file in sorted((root/'semantic'/dataset).glob('ids_*.npz')):
            budget();suffix=file.stem.split('_',1)[1];block=dest/f'block_{suffix}.npz';allblocks.append(block)
            if block.exists():raise FileExistsError('preserve prior block; do not restart a completed diagnostic')
            parts=defaultdict(list);examples=[]
            with np.load(file) as ids,np.load(file.with_name('block_'+suffix+'.npz')) as original:
                if ids['identity'].tolist()!=[sha(run/spec['encoder']),sha(run/spec['decoder'])]:raise ValueError('frozen actual parent model identity')
                np.testing.assert_array_equal(ids['keys'],original['keys'])
                for j,key in enumerate(ids['keys']):
                    row,start,fold=map(int,key);r=rows[row]
                    if (r['split'],r['modality'],r['dataset'])!=('train','eeg',dataset) or folds[dataset][r['subject']]!=fold:raise ValueError('same held original train crop before raw reads')
                    ex=original['examples'][(original['examples'][:,0]==row)&(original['examples'][:,1]==start)]
                    code={b:ids['transient_'+b][j:j+1].copy() for b in 'SO'}
                    baseline=model.decode_ids(code,dataset);one={b:np.ones_like(v,float) for b,v in code.items()};endpoint=decode_scaled_atoms(model,code,one,dataset)
                    for b in baseline:np.testing.assert_array_equal(baseline[b],endpoint[b])
                    fields,seen,_=base_tokenizer_joint_eeg(rows,dict(row=row,start_s=start,positions=[]),scales,names,cache,allowed_splits=('train',))
                    x=fields['x'].T.reshape(1,len(names),15,400);physical=sources[fold].predict_continuous(x,dataset,seen[None]);gains=projection_gains(physical,code,model,dataset)
                    np.testing.assert_array_equal(seen,np.any(code['S'][0]!=0,axis=1));fresh=decode_scaled_atoms(load_model(run/spec['decoder']),code,gains,dataset)
                    xoutput=decode_scaled_atoms(model,code,gains,dataset)
                    for b in fresh:np.testing.assert_array_equal(fresh[b],xoutput[b])
                    batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=12);n=np.array([native[rows[i]['positions'][p].upper()] for i,_,p in ex]);factors=input_scales(rows,ex,scales)
                    true={b:np.zeros_like(physical[b]) for b in 'SO'};complete=np.zeros(code['S'].shape,bool)
                    for b in 'SO':true[b][0,n]=batch[b.lower()]
                    complete[0,n]=batch['mask'].all(-1);oracle=projection_gains(true,code,model,dataset,nonnegative=False,complete=complete)
                    predictions=dict(gain1=baseline,X_continuous_gain=xoutput,target_continuous_gain=decode_scaled_atoms(model,code,oracle,dataset))
                    for arm,prediction in predictions.items():
                        metric=joint_reconstruction_metrics({b:v[0,n] for b,v in prediction.items()},batch,1,factors)
                        if metric['support']['nonfinite_prediction_points'].any():raise ValueError('nonfinite forward-gain emission')
                        if arm=='gain1':
                            take=(original['examples'][:,0]==row)&(original['examples'][:,1]==start)
                            for k,v in metric['metrics'].items():np.testing.assert_array_equal(v,original['transient__'+k][take])
                        for k,v in metric['metrics'].items():parts[arm+'__'+k].append(v)
                    gain_parts['examples'].append(ex)
                    for b in 'SO':gain_parts['X_'+b].append(gains[b][0,n]);gain_parts['target_'+b].append(oracle[b][0,n])
                    for index,item in enumerate(gallery):
                        if item['dataset']!=dataset or (item['row'],item['start_s'])!=(row,start):continue
                        a=np.flatnonzero(ex[:,2]==item['position'])
                        if len(a)!=1:raise ValueError('original gallery physical position missing')
                        a=int(a[0]);label=f'gallery_{index:02d}';plot_meta.append(dict(index=index,label=label,**item));arrays={k:batch[k][a] for k in ['x','s','o','mask']}
                        for arm,p in predictions.items():
                            for b in ['S','O','sum']:arrays[arm+'_'+b]=p[b][0,n[a]]
                        for b in 'SO':arrays['X_gain_'+b]=gains[b][0,n[a]];arrays['target_gain_'+b]=oracle[b][0,n[a]]
                        plots.update({label+'__'+k:v for k,v in arrays.items()})
                    examples.extend(ex);identities.append(dict(row=row,start_s=start,fold=fold,subject=r['subject'],source_f_excludes_subject=True,gain1_output_exact=True,saved_gain_reloaded_model_output_exact=True))
            np.savez_compressed(block,examples=examples,**{k:np.concatenate(v) for k,v in parts.items()});write(dest/'progress.json',dict(last_block=suffix,scored=sum(len(v) for v in gain_parts['examples']),CPU_seconds=budget()))
        joined=defaultdict(list)
        for p in allblocks:
            with np.load(p) as f:
                for k in f:joined[k].append(f[k])
        np.savez_compressed(out/f'{dataset}_metrics.npz',**{k:np.concatenate(v) for k,v in joined.items()});np.savez_compressed(dest/'projection_gains.npz',**{k:np.concatenate(v) for k,v in gain_parts.items()});np.savez_compressed(dest/'gallery_arrays.npz',**plots);write(dest/'gallery_meta.json',plot_meta);write(dest/'replay_endpoints.json',identities)
        write(dest/'summary.json',dict(status='complete',arrays=len(identities),position_windows=sum(len(v) for v in gain_parts['examples']),CPU_seconds=time.process_time(),validation_reads=0));return dataset


def merge(cfg,hard,run,rows,out):
    data=defaultdict(list)
    for d in sorted(STATE['folds']):
        with np.load(out/f'{d}_metrics.npz') as f:
            for k in f:data[k].append(f[k])
    data={k:np.concatenate(v) for k,v in data.items()};ex=data.pop('examples')
    if len(ex)!=17040 or len(set(map(tuple,ex)))!=17040:raise ValueError('full original scoring population required')
    summaries={a:base_training_aggregate({k.split('__',1)[1]:v for k,v in data.items() if k.startswith(a+'__')},ex,rows) for a in ARMS}
    ref=json.loads((run/cfg['source']['hard_run']/'summary.json').read_text())['families']['high']['arms']['transient']['metrics'];rule=cfg['admission']
    for a,v in summaries.items():
        m=v['metrics'];nmse=np.asarray(m['nmse']['macro']);rn=np.asarray(ref['nmse']['macro']);gain=np.asarray(m['rms_gain']['macro']);lo,hi=rule['RMS_gain']
        checks=dict(X_NMSE=bool(np.all(nmse[0]<=rule['X_NMSE_max_ratio']*rn[0])),S_O_NMSE=bool(np.all(nmse[1:]<=rule['S_O_NMSE_max_ratio']*rn[1:])),dataset=all(np.all(np.asarray(x)<=rule['each_dataset_component_NMSE_max_ratio']*np.asarray(ref['nmse']['by_dataset'][d])) for d,x in m['nmse']['by_dataset'].items()),amplitude=bool(np.all((gain>=lo)&(gain<=hi))),bands=all(np.all(abs(np.asarray(m[k]['macro']))<=rule['theta_alpha_abs_signed_and_mean_abs_db_max']) for k in ['band_4_8_db','band_8_13_db','band_4_8_abs_db','band_8_13_abs_db']),envelope=bool(np.all(np.asarray(m['local_rms_correlation']['macro'])>=rule['local_RMS_correlation_min'])),correlation=bool(np.all(np.asarray(m['correlation']['macro'])>0)))
        v.update(waveform_gates=checks,waveform_pass=all(checks.values()))
    write(out/'summary.json',dict(status='complete_continuous_diagnostic_pending_actual_images',arms=summaries,high_reference=ref,eligible_for_discrete=bool(summaries['X_continuous_gain']['waveform_pass']),ordinary_validation_reads=0,confirmation_reads=0,test_reads=0,position_windows=17040,arrays=400,subjects=60,CPU_seconds=budget(),oracle_scope=cfg['diagnostic']['oracle_scope']))
    print(json.dumps({a:dict(nmse=v['metrics']['nmse']['macro'],rms=v['metrics']['rms_gain']['macro'],checks=v['waveform_gates']) for a,v in summaries.items()}),flush=True)


def replay_worker(dataset):
    """Saved IDs and saved diagnostic gains finish decoding before scoring reads."""
    with threadpool_limits(1):
        cfg,hard,run,rows,scales,out=(STATE[k] for k in ['cfg','hard','run','rows','scales','out'])
        model=load_model(run/hard['families']['semantic']['decoder']);names=model.metadata['positions'][dataset];native={n.upper():i for i,n in enumerate(names)};cache=OrderedDict();maximum=0.;crops=0;positions=0
        with np.load(out/dataset/'projection_gains.npz') as f:
            gain_examples=f['examples'];saved={a:{b:f[('X_' if a=='X_continuous_gain' else 'target_')+b].copy() for b in 'SO'} for a in ['X_continuous_gain','target_continuous_gain']}
        for path in sorted((run/cfg['source']['hard_run']/'semantic'/dataset).glob('ids_*.npz')):
            budget();suffix=path.stem.split('_',1)[1]
            with np.load(path) as ids,np.load(out/dataset/f'block_{suffix}.npz') as expected:
                for j,key in enumerate(ids['keys']):
                    row,start,fold=map(int,key);take=(gain_examples[:,0]==row)&(gain_examples[:,1]==start);ex=gain_examples[take];sites=np.array([native[rows[i]['positions'][p].upper()] for i,_,p in ex]);code={b:ids['transient_'+b][j:j+1] for b in 'SO'}
                    emitted={'gain1':model.decode_ids(code,dataset)}
                    for a in ['X_continuous_gain','target_continuous_gain']:
                        gain={b:np.ones(code[b].shape) for b in 'SO'}
                        for b in 'SO':gain[b][0,sites]=saved[a][b][take]
                        emitted[a]=decode_scaled_atoms(model,code,gain,dataset)
                    # Only the scorer below reads X/S/O/masks; the decoder has finished.
                    batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=12);scale=input_scales(rows,ex,scales);where=(expected['examples'][:,0]==row)&(expected['examples'][:,1]==start);np.testing.assert_array_equal(ex,expected['examples'][where])
                    for a,p in emitted.items():
                        metrics=joint_reconstruction_metrics({b:v[0,sites] for b,v in p.items()},batch,1,scale)['metrics']
                        for k,v in metrics.items():
                            reference=expected[a+'__'+k][where];np.testing.assert_array_equal(v,reference);finite=np.isfinite(v)&np.isfinite(reference)
                            if finite.any():maximum=max(maximum,float(abs(v[finite]-reference[finite]).max()))
                    crops+=1;positions+=len(ex)
        result=dict(status='passed',arrays=crops,position_windows=positions,max_metric_difference=maximum,encoder_calls=0,source_calls=0,X_reads_at_decode=0,gain1='saved_atomic_integers_only',continuous_gains='saved_float64_diagnostic_gain_stream_not_an_integer_candidate');write(out/dataset/'independent_replay.json',result);return dataset,result


def render(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'waveforms';folder.mkdir(exist_ok=True);manifest=[]
    hard=yaml.safe_load((ROOT/cfg['source']['hard_contract']).read_text());run=ROOT/hard['data']['run']
    original=json.loads((run/cfg['source']['gallery']).with_name('neighbor_voltage_diagnostic.json').read_text())['cases']
    original_patch={v['index']:v['worst_complete_patch'] for v in original}
    for d in sorted(p.name for p in out.iterdir() if p.is_dir() and (p/'gallery_meta.json').exists()):
        for item in json.loads((out/d/'gallery_meta.json').read_text()):
            with np.load(out/d/'gallery_arrays.npz') as f:arrays={k.split('__',1)[1]:f[k].reshape(-1) for k in f if k.startswith(item['label']+'__')}
            mask=arrays['mask'].astype(bool);scale=item['scale'];patch=original_patch[item['index']]
            if not mask.reshape(15,400)[patch].all():raise ValueError('original complete difficult patch support changed')
            t=np.arange(6000)/200;edge=min(max(2*patch,2),28);spans=[(0,30),(2*patch,2*patch+2),(edge-.25,edge+.25)]
            fig,axes=plt.subplots(3,3,figsize=(17,10),layout='constrained')
            for row,(b,target) in enumerate([('sum','x'),('S','s'),('O','o')]):
                for col,(lo,hi) in enumerate(spans):
                    ax=axes[row,col];ax.plot(t,np.where(mask,arrays[target]*scale,np.nan),c='black',lw=.85,label='target')
                    for a,c,style in [('gain1','#b27400','-'),('X_continuous_gain','#0072b2','-'),('target_continuous_gain','#b8518c',':')]:ax.plot(t,arrays[a+'_'+b]*scale,c=c,ls=style,lw=.7,label=a)
                    for boundary in range(0,31,2):
                        if lo<=boundary<=hi:ax.axvline(boundary,c='.6',ls=':',lw=.5)
                    ax.fill_between(t,0,1,where=~mask,transform=ax.get_xaxis_transform(),color='.8',alpha=.4);ax.set_xlim(lo,hi);ax.set_xlabel('time (s)');ax.set_ylabel(b+' ('+item['source_unit']+')');ax.grid(alpha=.2)
                    if row==0:ax.set_title(['complete30s','original frozen difficult2s patch','fixed nearest token boundary'][col])
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4);fig.suptitle(f'Original gallery{item["index"]}: {d}/{item["subject"]}/{item["position_name"]} +{item["start_s"]}s\nK512 fixed atomic parent; continuous current-patch gains are diagnostic side information, not integer candidate; no smoothing')
            path=folder/(item['label']+'_'+d+'.png');fig.savefig(path,dpi=170);plt.close(fig);manifest.append(dict(**item,path=str(path),zoom_patch=patch))
    if len(manifest)!=17:raise ValueError('retain every original gallery example')
    write(out/'figure_manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--synthetic',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--replay',action='store_true');a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());validate(cfg);out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.render:render(cfg,out);return
    if a.synthetic:
        from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary
        rng=np.random.default_rng(3);m=SpatialWaveformDictionary(dict(codebook_size=2,positions={'fixture':list('abcde')},output_projection_hz=[1,45]),{}, {},{(b,k):np.r_[np.zeros((1,400)),rng.normal(size=(1,400))] for b in 'SO' for k in range(5)},{('fixture',b,k):rng.normal(size=(1,5)) for b in 'SO' for k in range(5)})
        ids={b:np.ones((1,5,15),int) for b in 'SO'};g={b:np.ones_like(v,float) for b,v in ids.items()}
        for b,v in m.decode_ids(ids,'fixture').items():np.testing.assert_array_equal(v,decode_scaled_atoms(m,ids,g,'fixture')[b])
        write(out/'synthetic.json',dict(status='pass',measured_reads=0,gain1_exact=True));return
    if not (out/'synthetic.json').exists():raise ValueError('synthetic before original data')
    lock=open(out/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'summary.json').exists() and not a.replay:raise FileExistsError('preserve completed gain diagnostic')
    hard=yaml.safe_load((ROOT/cfg['source']['hard_contract']).read_text());run=ROOT/hard['data']['run'];basis=yaml.safe_load((ROOT/hard['data']['basis_config']).read_text())['source'];manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/basis['g0_config']).read_text()),ROOT);folds=json.loads((run/hard['data']['folds']).read_text());gallery=json.loads((run/cfg['source']['gallery']).read_text())['examples']
    if len(gallery)!=17 or hard['data']['allowed_splits']!=['train'] or sum(len(p) for p in folds.values())!=60:raise ValueError('original17 gallery and60train people required before arrays')
    STATE.update(cfg=cfg,hard=hard,run=run,rows=manifest['rows'],scales=scales,folds=folds,out=out,gallery=gallery)
    if a.replay:
        if not (out/'summary.json').exists():raise ValueError('complete initial diagnostic before independent replay')
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:result=dict(pool.map(replay_worker,sorted(folds)))
        write(out/'independent_replay.json',dict(status='passed',datasets=result,position_windows=sum(v['position_windows'] for v in result.values()),encoder_calls=0,source_calls=0,CPU_seconds=budget()));return
    write(out/'resolved_config.json',cfg)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(dataset_worker,sorted(folds)))
    merge(cfg,hard,run,manifest['rows'],out)


if __name__=='__main__':main()
