"""Versioned public 10 Hz targets for independent S/residual tokenizers.

The parent supplies identities only. Native EEG and the v5/none Hb cache are
read again after checking that identity and split. The teacher is privileged
supervision; student inputs contain only a two-second modality patch. Hb keeps
the declared [-30,-25] s reference and its full-record cache preprocessing.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .functional_tokenizer import inventory, identity_digest
from .ssm_prepared import fit_feature_coordinate, EEG_PROXY_INPUT_SCHEMA
from .unified_physiology import (
    CACHED_FNIRS_INPUT_SCHEMA, load_cached_fnirs_record, load_native_eeg_record,
)
from src.inference.observation_baselines import eeg_band_power, disjoint_native_bin_operator


PARENT = 'experiments/runs/physiology_semantic_tokenizer/functional_tokenizer/20261011_v1'
DATASETS = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
DT = .1
STEPS = 900
STUDENT = slice(300, 500)


def latent_health_plan(parent, spec):
    """Metadata-only admission of retained public train/validation Hb targets."""
    parent = Path(parent)
    if (spec['allowed_splits'] != ['train', 'validation'] or spec['test_access'] != 'forbidden'
            or spec['measured_modality'] != 'hb' or spec['parent_steps'] != [250, 550]):
        raise ValueError('registered Hb development-only 30-second scope required before arrays')
    resolved = json.loads((parent / 'resolved_config.json').read_text())['config']
    state = json.loads((parent / 'controller.json').read_text())
    manifest = json.loads((parent / 'cache/manifest.json').read_text())
    source = json.loads((parent / 'inventory.json').read_text())
    if (resolved['data']['protected_data'] != 'forbidden' or state['status'] != 'completed'
            or manifest['schema'] != 'dual_component_training_cache_v1'
            or manifest['target_units'] != 'training_observation_SD'
            or resolved['teacher']['arm'] != 'tak_fixed'
            or resolved['tensor']['teacher_steps'] != 900):
        raise ValueError('completed fixed-Tak public parent with frozen scales required before arrays')
    planned = [r for r in source['rows'] if r['split'] in spec['allowed_splits']]
    retained = {r['id']: r for r in manifest['rows'] if r['split'] in spec['allowed_splits']}
    subjects, groups = {}, defaultdict(list)
    for row in planned:
        if row['dataset'] not in DATASETS:
            raise ValueError('unregistered dataset in development inventory')
        key = (row['dataset'], row['subject'])
        if key in subjects and subjects[key] != row['split']:
            raise ValueError('subject appears in multiple partitions')
        subjects[key] = row['split']
        if row['id'] in retained and retained[row['id']] != row:
            raise ValueError('parent inventory and retained row identity disagree')
        groups[row['trial_id']].append(row)
    trials, failures = [], []
    for trial, group in groups.items():
        if len(group) == 3 and all(r['id'] in retained for r in group):
            trials.append(dict(trial_id=trial, rows=group))
        else:
            failures.append(dict(trial_id=trial, status='parent_unavailable', planned_windows=len(group)))
    return dict(schema='dual_component_latent_health_plan_v1', parent=str(parent.resolve()),
        parent_identity=manifest['identity'], planned_trials=len(groups), planned_windows=len(planned),
        trials=trials, failures=failures, observation_training_sd=manifest['observation_training_sd'],
        allowed_splits=spec['allowed_splits'], test_access='forbidden', protected_data='forbidden',
        measured_arrays_read=0, identity=identity_digest(planned),
        event_support_s=spec['event_support_s'], parent_reference_event_s=[-30, -25],
        target='retained_fixed_Tak_conditional_output_and_residual_not_physiological_truth')


def latent_health_hb_patch(row, arrays, scale, spec):
    """Extract observed paired Hb and frozen labels without fitting or centering."""
    if row['split'] not in ('train', 'validation') or row['dataset'] not in DATASETS:
        raise ValueError('test/protected or unknown rows cannot enter latent health data')
    if spec['parent_steps'] != [250, 550] or spec['patches'] != 15 or spec['input_dim'] != 40:
        raise ValueError('exact 30-second paired Hb patch contract required')
    if str(np.asarray(arrays['identity'])) != row['id'] or str(np.asarray(arrays['teacher_arm'])) != 'tak_fixed':
        raise ValueError('frozen teacher identity or arm mismatch')
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('positive frozen training observation scales required')
    values = {}
    for key, field in (('u', 'u'), ('S', 'physical'), ('O', 'remaining')):
        source = np.asarray(arrays[field], dtype=float)
        if source.shape != (900, 3):
            raise ValueError('retained 90-second observation-coordinate trajectory required')
        value = source[250:550, 1:] / scale[1:]
        if not np.isfinite(value).all():
            raise ValueError('nonfinite Hb support is a retained failure, not an imputed sample')
        values[key] = value.reshape(15, 40).astype('float32')
    if not np.allclose(values['u'], values['S'] + values['O'], rtol=2e-6, atol=2e-6):
        raise ValueError('frozen S/O decomposition does not close in observation coordinates')
    return values


def voltage_teacher_plan(cfg, retained):
    """Select a finite public development panel using identities, before arrays."""
    spec = cfg['voltage_teacher']
    if (spec['dataset'] != 'simultaneous_eeg_nirs' or spec['record'] != 'cnt_nback'
            or spec['allowed_splits'] != ['train', 'validation']
            or spec['test_access'] != 'forbidden' or spec['windows_per_subject'] != 2
            or spec['sample_rate_hz'] != 200 or spec['seconds'] != 90
            or spec['score_seconds'] != [30, 50]
            or cfg['data']['protected_data'] != 'forbidden'):
        raise ValueError('bounded native-voltage public development contract required')
    selected, excluded = [], []
    for split in spec['allowed_splits']:
        subjects = spec[split + '_subjects']
        if not subjects or not set(subjects) <= set(retained['subjects'][spec['dataset']][split]):
            raise ValueError('subjects must retain the parent development partition')
        for subject in subjects:
            groups = defaultdict(list)
            for row in retained['rows']:
                if (row['dataset'], row['subject'], row['record']) == (spec['dataset'], subject, spec['record']):
                    if row['split'] != split:
                        raise ValueError('subject crosses the frozen split')
                    groups[row['trial_id']].append(row)
            complete = [sorted(g, key=lambda r: r['coordinate']) for g in groups.values()
                        if len(g) == 3 and len({r['coordinate'] for r in g}) == 3
                        and len({r['eeg_start_s'] for r in g}) == 1]
            complete.sort(key=lambda g: (g[0]['eeg_start_s'], g[0]['trial_id']))
            if len(complete) < 2 or complete[-1][0]['eeg_start_s'] - complete[0][0]['eeg_start_s'] < 90:
                excluded.append(dict(subject=subject, split=split, reason='two_disjoint_windows_unavailable'))
                continue
            for slot, group in enumerate((complete[0], complete[-1])):
                selected.append(dict(id=group[0]['trial_id'], subject=subject, split=split,
                    record=spec['record'], dataset=spec['dataset'], slot=slot, rows=group,
                    candidate_trials=len(complete)))
    return dict(schema='voltage_teacher_panel_v1', rows=selected, excluded=excluded,
        planned_subjects=sum(len(spec[k + '_subjects']) for k in spec['allowed_splits']),
        planned_windows=2 * sum(len(spec[k + '_subjects']) for k in spec['allowed_splits']),
        selection=spec['selection'], protected_data='forbidden', measured_arrays_read=0)


def voltage_separation_plan(project_root, cfg):
    """Validate the exact completed public panel before opening any arrays."""
    spec=cfg['separation'];root=Path(project_root).resolve()
    expected='experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261105_voltage_teacher_v2'
    if (spec['parent_run']!=expected or spec['allowed_splits']!=['train','validation']
            or spec['test_access']!='forbidden' or spec['protected_data']!='forbidden'
            or spec['dataset']!='simultaneous_eeg_nirs' or spec['channels']!=16
            or spec['sample_rate_hz']!=200 or spec['seconds']!=90):
        raise ValueError('exact completed native-voltage public development panel required')
    parent=root/expected
    resolved=json.loads((parent/'resolved_config.json').read_text())
    original=json.loads((parent/'plan.json').read_text())
    summary=json.loads((parent/'summary.json').read_text())
    verified=json.loads((parent/'verification.json').read_text())
    if (resolved['config']['schema']!=spec['parent_schema'] or summary['status']!='completed'
            or verified['status']!='passed' or original['protected_data']!='forbidden'):
        raise ValueError('completed and verified public parent required')
    inventory_parent=root/resolved['config']['voltage_teacher']['parent_run']
    canonical=voltage_teacher_plan(resolved['config'],json.loads((inventory_parent/'inventory.json').read_text()))
    if original!=canonical:raise ValueError('retained development identities changed')
    rows=[]
    for row in original['rows']:
        if row['split'] not in spec['allowed_splits']:raise ValueError('test row cannot enter separation')
        phase=('fit' if row['slot']==0 else 'select') if row['split']=='train' else 'validation'
        rows.append(dict(row,phase=phase))
    return dict(schema='voltage_separation_panel_v1',parent=str(parent),rows=rows,
        protected_data='forbidden',test_access='forbidden',
        fit_ids=[r['id'] for r in rows if r['phase']=='fit'],
        selection_ids=[r['id'] for r in rows if r['phase']=='select'],
        validation_ids=[r['id'] for r in rows if r['phase']=='validation'],
        parent_summary=str(parent/'summary.json'),measured_arrays_read=0)


def voltage_power_cap_example_cases(panel, original_case, subjects):
    """Select one existing validation slot per other subject, without arrays."""
    if (panel.get('schema')!='voltage_separation_panel_v1'
            or panel.get('protected_data')!='forbidden' or panel.get('test_access')!='forbidden'
            or len(subjects)!=4 or len(set(subjects))!=4):
        raise ValueError('four distinct subjects in the verified public panel required')
    originals=[r for r in panel['rows'] if r['id']==original_case]
    if len(originals)!=1 or originals[0]['split']!='validation':
        raise ValueError('original validation case required')
    if originals[0]['subject'] in subjects:
        raise ValueError('examples must be the other validation subjects')
    result=[]
    for subject in subjects:
        candidates=[r for r in panel['rows'] if r['subject']==subject and r['slot']==0]
        if (len(candidates)!=1 or candidates[0]['split']!='validation'
                or candidates[0]['phase']!='validation'
                or candidates[0]['dataset']!='simultaneous_eeg_nirs'
                or candidates[0]['record']!='cnt_nback'):
            raise ValueError('one retained first validation n-back slot per named subject required')
        result.append(candidates[0])
    return result



def voltage_power_cap_posterior_plan(project_root,cfg):
    """Extend two admitted windows to one metadata-defined occipital ROI."""
    from .functional_tokenizer import regional_anchors
    from .unified_physiology import ChannelGeometryIndex
    from copy import deepcopy
    root=Path(project_root);spec=cfg['posterior'];panel=voltage_separation_plan(root,cfg)
    examples=root/spec['reference_examples']
    previous=json.loads((examples/'summary.json').read_text())
    previous_plan=json.loads((examples/'plan.json').read_text())
    if (previous['counts']['completed_pairs']!=5 or previous['counts']['failed_pairs']
            or json.loads((examples/'verification.json').read_text())['status']!='passed'
            or previous_plan['panel']!=panel
            or previous['observation_weights']!=[.1,10.,10.]):
        raise ValueError('verified five-example reference and unchanged original public panel required')
    selected={r['case_id']:r['injection_channel'] for r in spec['cases']}
    if len(selected)!=2:raise ValueError('two distinct predeclared posterior examples required')
    originals=[e for e in panel['rows'] if e['phase']=='fit' or e['id'] in selected]
    if (len(originals)!=8 or sum(e['phase']=='fit' for e in originals)!=6
            or any(e['split']!='validation' for e in originals if e['id'] in selected)):
        raise ValueError('six train-only calibration windows and two validation windows required')
    geometry=ChannelGeometryIndex(root/cfg['data']['cache_root']);entries=[]
    references={r['case_id']:r['run'] for r in previous['rows']}
    for source_case in originals:
        _,record=_registered_record(root,cfg,source_case['rows'])
        anchor=regional_anchors(record,geometry,dict(regions=['occipital'],
            region_vectors=[spec['occipital_direction']]))[0]
        if (anchor['hb_channel']!=spec['occipital_hb_channel']
                or anchor['eeg_channels']!=spec['occipital_eeg_channels']):
            raise ValueError('registered posterior geometry differs from the fixed anatomical contract')
        rows=deepcopy(source_case['rows'][:2])
        if [r['anchor']['region'] for r in rows]!=['left_motor','prefrontal']:
            raise ValueError('unchanged left motor and frontal anchors required')
        row=deepcopy(source_case['rows'][2]);row.update(id=source_case['id']+'__occipital',
            coordinate='simultaneous_eeg_nirs__occipital',anchor=anchor)
        rows.append(row)
        names={n for r in rows for n in r['anchor']['eeg_channels']}
        if len(names)!=17 or not {'O1','O2'}<=names:
            raise ValueError('registered 17-electrode posterior view required before arrays')
        entries.append(dict(source_case,rows=rows,source_case=source_case,
            injection_channel=selected.get(source_case['id']),
            original_physical_run=references.get(source_case['id'])))
    return dict(schema='voltage_power_cap_posterior_plan_v1',panel=panel,rows=entries,
        cases=[e for e in entries if e['injection_channel']],
        training=[e for e in entries if e['phase']=='fit'],
        reference_examples=str(examples.resolve()),reference_run=previous_plan['reference_run'],
        parent=previous_plan['parent'],test_access='forbidden',protected_data='forbidden',
        channels=17,regions=spec['regions'],measured_arrays_read=0,
        selection='original VP002/O1 and first additional VP005/O2, fixed before posterior arrays')


def prepare_voltage_power_cap_posterior(project_root,cfg,plan,entry):
    """Native posterior voltage and registered cached Hb; no new parser."""
    if (plan.get('schema')!='voltage_power_cap_posterior_plan_v1'
            or plan.get('protected_data')!='forbidden' or plan.get('test_access')!='forbidden'
            or entry not in plan['rows'] or entry['split'] not in ('train','validation')
            or entry['source_case'] not in plan['panel']['rows']):
        raise ValueError('declared posterior development identity required before array access')
    source=entry['source_case']
    root,record=_registered_record(project_root,cfg,source['rows'])
    # Cached reader validates v5/none, native units and bandpass before arrays.
    cached=load_cached_fnirs_record(root,record)
    native=load_native_eeg_record(root,record)
    if (native.sample_rate_hz!=200 or native.native_unit not in ('uV','µV','μV')
            or not native.unit_evidence):
        raise ValueError('native 200 Hz microvolt EEG required')
    old,old_meta=read_voltage_separation_input(plan['panel'],source)
    lookup={n:i for i,n in enumerate(native.channel_names)}
    names=sorted({n for r in entry['rows'] for n in r['anchor']['eeg_channels']},key=lookup.get)
    start=round(source['rows'][0]['eeg_start_s']*200)
    x=np.asarray(native.values[start:start+18000,[lookup[n] for n in names]],float)
    if x.shape!=(18000,17) or not np.isfinite(x).all():
        raise ValueError('complete 17-channel 90-second posterior voltage required')
    old_names=list(old['channels']);common=[n for n in names if n in old_names]
    if not np.array_equal(x[:,[names.index(n) for n in common]],old['x'][:,[old_names.index(n) for n in common]]):
        raise ValueError('native posterior reader does not reproduce retained overlapping channels')
    row=entry['rows'][2];pair=row['anchor']['hb_pair']
    if cached['channel_names'][2*pair]!=row['anchor']['hb_channel']+'_HbO':
        raise ValueError('posterior Hb channel does not match registered geometry')
    first,last=row['hb_start_s'],row['hb_start_s']+90
    tolerance=8*np.spacing(max(abs(first),abs(last),1.))
    take=(cached['time_s']>=first-tolerance)&(cached['time_s']<last-tolerance)
    times=cached['time_s'][take]-first;times[abs(times)<=tolerance]=0.
    h=cached['values'][take,pair]
    if not np.isfinite(h).all():raise ValueError('posterior Hb support is missing')
    ops=disjoint_native_bin_operator(times)
    hb=ops['bins']@h;hb-=hb[:50].mean(0)
    if not np.array_equal(times,old['hb_native_time'][:,0]):
        raise ValueError('new posterior and retained Hb clocks differ')
    values=dict(x=x,hb=np.stack([old['hb'][:,0],old['hb'][:,1],hb],axis=1),
        hb_native_time=np.stack([old['hb_native_time'][:,0],old['hb_native_time'][:,1],times],axis=1),
        roi_indices=np.array([[names.index(n) for n in r['anchor']['eeg_channels']] for r in entry['rows']]),
        channels=np.array(names))
    values['features_baseline']=separation_covariances(x,cfg['separation'])['features']
    metadata=dict(case_id=entry['id'],split=entry['split'],source_case_id=source['id'],
        native_source=str(native.source_path),native_start_sample=start,eeg_unit=native.native_unit,
        eeg_reference=native.reference,sample_rate_hz=200,channels=names,
        posterior_anchor=row['anchor'],posterior_hb_provenance=cached['provenance'],
        original_overlapping_voltage_exact=True,hb_clock_exact=True,calibration_eligible=entry['phase']=='fit')
    return values,metadata


def read_voltage_separation_input(plan, entry, *, include_eog=False):
    """One retained read boundary; EOG is available only for validation scoring."""
    if (plan.get('schema')!='voltage_separation_panel_v1' or plan.get('protected_data')!='forbidden'
            or entry not in plan['rows'] or entry['split'] not in ('train','validation')
            or (include_eog and entry['split']!='validation')):
        raise ValueError('declared development identity; EOG is validation-only')
    parent=Path(plan['parent'])
    original=json.loads((parent/'plan.json').read_text())
    matches=[r for r in original['rows'] if r['id']==entry['id']]
    if len(matches)!=1 or any(entry[k]!=v for k,v in matches[0].items()):
        raise ValueError('exact parent identity required before array access')
    metadata=json.loads((parent/'prepared'/f"{entry['id']}.json").read_text())
    if metadata.get('status')!='completed' or not metadata.get('legacy_supervision_float32_exact'):
        raise ValueError('completed verified native-voltage window required')
    keys=['x','hb','hb_native_time','roi_indices','channels','legacy_s']
    if include_eog:keys.append('eog')
    with np.load(parent/'prepared'/f"{entry['id']}.npz",allow_pickle=False) as archive:
        arrays={k:archive[k] for k in keys}
    if (arrays['x'].shape!=(18000,16) or arrays['hb'].shape!=(900,3,2)
            or arrays['roi_indices'].shape!=(3,6) or metadata['eeg_unit'] not in ('uV','µV','μV')
            or not np.isfinite(arrays['x']).all() or not np.isfinite(arrays['hb']).all()):
        raise ValueError('complete native 200 Hz microvolt EEG and cached 10 Hz Hb required')
    return arrays,metadata


def separation_covariances(x, spec):
    """Centred 2-second statistics; original voltage itself is not centred."""
    from scipy.signal import butter,sosfiltfilt
    x=np.asarray(x,float);length=int(spec['covariance_seconds']*200)
    if x.ndim!=2 or x.shape[0]%length or not np.isfinite(x).all():
        raise ValueError('finite complete native-voltage epochs required')
    def covariance(v):
        blocks=v.reshape(-1,length,x.shape[1]);blocks=blocks-blocks.mean(axis=1,keepdims=True)
        return np.einsum('ntc,ntd->ncd',blocks,blocks)/length
    carriers=[sosfiltfilt(butter(4,band,fs=200,btype='bandpass',output='sos'),x,axis=0) for band in BANDS]
    power=np.stack([np.mean(v.reshape(-1,20,x.shape[1])**2,axis=1) for v in carriers],axis=-1)
    log=np.log(np.maximum(power,1e-20));log-=log[:50].mean(axis=0)
    return dict(full=covariance(x),band=np.stack([covariance(v) for v in carriers],axis=1),
        features=log,time_s=(np.arange(len(x)//length)+.5)*length/200)


def separation_state_epochs(z, spec):
    """Shared slow guidance transform, never a transform of the voltage target."""
    from scipy.signal import butter,sosfiltfilt
    value=np.asarray(z,float)
    if value.shape!=(900,3):raise ValueError('three 10 Hz whole-window guidance trajectories required')
    out=np.full((45,3),np.nan)
    for roi in range(3):
        if np.isfinite(value[:,roi]).all():
            slow=sosfiltfilt(butter(2,spec['guidance_lowpass_hz'],fs=10,output='sos'),value[:,roi])
            out[:,roi]=slow.reshape(45,20).mean(axis=1)
    return out


def phase_shuffled_states(z, seed):
    """Common phase preserves cross-ROI spectra while breaking EEG alignment."""
    rng=np.random.default_rng(seed);value=np.asarray(z,float);out=value.copy()
    good=np.isfinite(value).all(axis=0)
    if good.any():
        spectrum=np.fft.rfft(value[:,good],axis=0)
        phases=np.exp(1j*rng.uniform(-np.pi,np.pi,len(spectrum)));phases[0]=1
        if len(value)%2==0:phases[-1]=1
        out[:,good]=np.fft.irfft(spectrum*phases[:,None],n=len(value),axis=0)
    return out


def fit_state_guided_subspace(fit_items, select_items, rank, method, spec):
    """Shared full-band covariance and training-only SPoC direction selection."""
    if (not fit_items or not select_items or any(i['phase']!='fit' for i in fit_items)
            or any(i['phase']!='select' for i in select_items)):
        raise ValueError('separate training fit and selection windows required')
    c=fit_items[0]['cov']['full'].shape[-1]
    if not 0<rank<c:raise ValueError('strictly reduced native-channel rank required')
    def support(item):
        lo,hi=spec['covariance_support_seconds'];t=item['cov']['time_s']
        return (t>=lo)&(t<hi)
    raw=np.mean(np.concatenate([i['cov']['full'][support(i)] for i in fit_items]),axis=0)
    shrink=spec['covariance_shrinkage']
    c0=(1-shrink)*raw+shrink*np.diag(np.diag(raw))
    values,vectors=np.linalg.eigh(c0);floor=max(values.max()*1e-10,1e-12)
    values=np.maximum(values,floor)
    square=(vectors*np.sqrt(values))@vectors.T
    inverse=(vectors/np.sqrt(values))@vectors.T
    candidates=[]
    if method=='pca':
        q=vectors[:,::-1][:,:rank];selection=[]
    elif method=='random':
        q=np.linalg.qr(np.random.default_rng(spec['seed']).normal(size=(c,c)))[0][:,:rank];selection=[]
    else:
        for roi in range(3):
            for band in range(5):
                matrices=[]
                for items in (fit_items,select_items):
                    chunks=[];zs=[]
                    for item in items:
                        z=item['states'][method][:,roi];use=support(item)&np.isfinite(z)
                        if use.sum()<4:continue
                        y=z[use]-np.mean(z[use]);sd=float(np.std(y))
                        if sd<1e-12:continue
                        chunks.append(item['cov']['band'][use,band]);zs.append(y/sd)
                    if not chunks:matrices.append(None);continue
                    cov=np.concatenate(chunks);z=np.concatenate(zs)
                    matrices.append(np.einsum('n,ncd->cd',z,cov-cov.mean(axis=0))/len(z))
                if any(m is None for m in matrices):continue
                eigen,dirs=np.linalg.eigh(inverse@matrices[0]@inverse)
                hold=np.einsum('ci,cd,di->i',dirs,inverse@matrices[1]@inverse,dirs)
                for k in range(c):
                    score=min(abs(float(eigen[k])),abs(float(hold[k]))) if eigen[k]*hold[k]>0 else 0.
                    candidates.append(dict(vector=dirs[:,k],score=score,roi=roi,band=band,
                        fit_covariance=float(eigen[k]),selection_covariance=float(hold[k])))
        basis=[];selection=[]
        for candidate in sorted(candidates,key=lambda r:-r['score']):
            if candidate['score']<=0:continue
            v=candidate['vector'].copy()
            for previous in basis:v-=previous*(previous@v)
            norm=np.linalg.norm(v)
            if norm<.15:continue
            basis.append(v/norm);selection.append({k:v for k,v in candidate.items() if k!='vector'})
            if len(basis)==rank:break
        if len(basis)!=rank:raise ValueError(f'only {len(basis)} stable independent directions for requested rank {rank}')
        q=np.stack(basis,axis=1)
    w=inverse@q;b=square@q;p=b@w.T
    if np.max(abs(p@p-p))>1e-9 or np.max(abs(w.T@b-np.eye(rank)))>1e-9:
        raise AssertionError('native-space projection identities failed')
    return dict(W=w,B=b,P=p,C0=c0,rank=np.asarray(rank),
        selection=selection,condition_number=float(values.max()/values.min()),
        idempotence_error=float(np.max(abs(p@p-p))),operator_norm=float(np.linalg.norm(p,2)))


def fit_neighbor_predictor(fit_voltages, subspace, spec):
    """Small symmetric blind-spot predictor, unit DC gain, frozen after fit."""
    lags=tuple(spec['predictor_lags_samples']);margin=max(lags);rank=subspace['W'].shape[1]
    gram=np.zeros((rank,len(lags),len(lags)));rhs=np.zeros((rank,len(lags)));count=0
    for x in fit_voltages:
        y=np.asarray(x,float)@subspace['W']
        y=y-y.mean(axis=0)  # Statistics only; unit DC gain preserves voltage offsets at application.
        centre=y[margin:-margin]
        pairs=np.stack([.5*(y[margin-l:len(y)-margin-l]+y[margin+l:len(y)-margin+l]) for l in lags],axis=-1)
        gram+=np.einsum('nrl,nrk->rlk',pairs,pairs);rhs+=np.einsum('nrl,nr->rl',pairs,centre);count+=len(centre)
    weights=[]
    for g,b in zip(gram/count,rhs/count):
        regularizer=spec['predictor_ridge']*max(float(np.trace(g))/len(lags),1e-12)
        kkt=np.zeros((len(lags)+1,len(lags)+1));kkt[:-1,:-1]=g+regularizer*np.eye(len(lags))
        kkt[-1,:-1]=1;kkt[:-1,-1]=1
        weights.append(np.linalg.solve(kkt,np.r_[b,1.])[:-1])
    return np.asarray(weights)


def apply_voltage_separation(x, model, spec, *, innovation_fraction=None):
    """EEG-only frozen frontend; both outputs retain native samples and units."""
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=model['W'].shape[0] or not np.isfinite(x).all():
        raise ValueError('finite same-montage EEG required')
    eta=float(model.get('innovation_fraction',0.) if innovation_fraction is None else innovation_fraction)
    if not 0<=eta<=1:raise ValueError('innovation fraction outside the declared range')
    y=x@model['W'];prediction=y.copy();lags=spec['predictor_lags_samples'];margin=max(lags)
    prediction[margin:-margin]=sum(.5*(y[margin-l:len(y)-margin-l]+y[margin+l:len(y)-margin+l])*model['predictor'][:,j]
        for j,l in enumerate(lags))
    spatial=y@model['B'].T
    temporal=eta*(y-prediction)@model['B'].T
    s=spatial-temporal;o=x-s
    return dict(s=s,o=o,o_spatial=x-spatial,o_temporal=temporal,
        closure_max_abs=float(np.max(abs(s+o-x))))


def separation_probe(channels, seed, kind, spec, *, rms=1., donor_eog=None):
    """Independent known perturbations; no learned-subspace alignment."""
    from scipy.signal import butter,sosfiltfilt
    rng=np.random.default_rng(seed);t=np.arange(18000)/200;c=len(channels)
    anterior=np.array([1. if n.startswith('Fp') else .65 if n.startswith(('AF','F1','F2')) else .22 if n.startswith('FC') else .08 for n in channels])
    anterior*=rng.uniform(.8,1.2,c)
    if kind=='ocular':
        if donor_eog is None:
            carrier=sum(rng.choice([-1.,1.])*np.exp(-.5*((t-onset)/rng.uniform(.1,.3))**2)
                for onset in (33.,37.,42.,46.))
        else:
            e=np.asarray(donor_eog,float)
            carrier=np.roll(sosfiltfilt(butter(4,[.5,15],fs=200,btype='bandpass',output='sos'),e[:,1]),1400)
            carrier*=np.exp(-.5*((t-40)/4)**2)
        value=carrier[:,None]*anterior
    elif kind=='fine':
        value=sosfiltfilt(butter(4,spec['fine_noise_band_hz'],fs=200,btype='bandpass',output='sos'),rng.normal(size=(len(t),c)),axis=0)
        value*=np.exp(-.5*((t-40)/3)**2)[:,None]
    elif kind.startswith('neural'):
        hz=float(kind.removeprefix('neural'))
        topology=np.array([1. if n in ('C3','Cz','C4','CP1','CP2') else .3 for n in channels])
        # Calibration and evaluation vary the field before any method is fitted.
        topology*=rng.uniform(.7,1.3,c)
        value=(np.sin(2*np.pi*hz*t+rng.uniform(0,2*np.pi))*np.exp(-.5*((t-40)/2)**2))[:,None]*topology
    else:raise ValueError('unknown prespecified separation probe')
    scale=float(np.sqrt(np.mean(value[6000:10000]**2)))
    if scale<=1e-15:raise ValueError('degenerate intervention')
    return value*(float(rms)/scale)


def separation_synthetic_record(seed, slot, scenario):
    """Known-voltage mechanism controls, including confounding and overlap."""
    from scipy.signal import butter,sosfiltfilt
    if scenario not in ('separable','overlapping_fine_noise','state_correlated_artifact'):
        raise ValueError('unknown frozen synthetic scenario')
    rng=np.random.default_rng(int(seed)+10000*int(slot));n=18000;t=np.arange(n)/200
    # One stable sensor montage; seeds vary signals, not the hidden ground-truth basis.
    mixing=np.random.default_rng(71).normal(size=(16,16))
    mixing[:,0]=np.r_[1.,.7,.5,.35,.2,.15,.1,.1,.1,.1,1.,.2,.15,.1,.1,.1]
    basis=np.linalg.qr(mixing)[0];neural=basis[:,2:8]
    state=np.column_stack([np.sin(2*np.pi*f*t+rng.uniform(0,6))+.25*np.sin(2*np.pi*.11*t+rng.uniform(0,6)) for f in (.035,.055,.075)])
    carriers=[]
    for j,(hz,amp) in enumerate(zip((3.,6.,10.,18.,35.,55.),(10.,15.,30.,18.,10.,8.))):
        carrier=amp*np.exp(.4*state[:,j%3])*np.sin(2*np.pi*hz*t+rng.uniform(0,6))
        carriers.append(carrier)
    clean=np.stack(carriers,axis=1)@neural.T
    slow=sum(rng.uniform(70,130)*np.exp(-.5*((t-onset)/rng.uniform(.12,.3))**2) for onset in rng.uniform(5,85,18))
    if scenario=='state_correlated_artifact':
        slow=70*np.exp(.85*state[:,0])*np.sin(2*np.pi*2*t)
    major=slow[:,None]*basis[:,0]
    dimension=6 if scenario=='overlapping_fine_noise' else 8
    fine_source=sosfiltfilt(butter(3,[25,70],fs=200,btype='bandpass',output='sos'),rng.normal(size=(n,dimension)),axis=0)
    fine_source*=5./np.sqrt(np.mean(fine_source**2))
    fine=fine_source@(neural if scenario=='overlapping_fine_noise' else basis[:,8:]).T
    return dict(x=clean+major+fine,clean=clean,major=major,fine=fine,
        state=state.reshape(900,20,3).mean(axis=1),true_neural_basis=neural,
        channels=np.array(['Fp1','AFF5h','AFz','F1','FC5','FC1','C3','Cz','CP5','CP1','Fp2','FC2','FC6','C4','CP2','CP6']))


def prepare_voltage_teacher_record(root, cfg, entries, coordinates, read_parent):
    """Reuse the registered native reader and exact retained physical baselines."""
    rows = [r for entry in entries for r in entry['rows']]
    if any(r['split'] not in ('train', 'validation') or r['dataset'] != 'simultaneous_eeg_nirs' for r in rows):
        raise ValueError('only the frozen public development panel may be read')
    root, record = _registered_record(root, cfg, rows)
    native = load_native_eeg_record(root, record)
    if (native.sample_rate_hz != 200 or native.native_unit not in ('uV', 'µV', 'μV')
            or not native.unit_evidence or native.auxiliary_values is None
            or set(native.auxiliary_channel_names) != {'HEOG', 'VEOG'}):
        raise ValueError('native microvolt EEG and genuine HEOG/VEOG required')
    lookup = {n: i for i, n in enumerate(native.channel_names)}
    for entry in entries:
        group = entry['rows']
        names = sorted({n for r in group for n in r['anchor']['eeg_channels']}, key=lookup.get)
        first = round(group[0]['eeg_start_s'] * 200)
        x = np.asarray(native.values[first:first + 18000, [lookup[n] for n in names]], float)
        eog = np.asarray(native.auxiliary_values[first:first + 18000], float)
        if x.shape != (18000, len(names)) or eog.shape != (18000, 2):
            raise ValueError('complete 90-second native support required')
        a = np.zeros((3, len(names) * 5)); physical, original, hb = [], [], []
        old_hb, roi_indices, hb_native_time = [], [], []
        for i, row in enumerate(group):
            co = coordinates[row['coordinate']]
            indices = [names.index(n) for n in row['anchor']['eeg_channels']]
            roi_indices.append(indices)
            loading = np.asarray(co['pc']).reshape(6, 5) * co['eeg_factor']
            for j, c in enumerate(indices):
                a[i, c * 5:(c + 1) * 5] = loading[j]
            old = read_parent('teacher', row)
            prepared = read_parent('prepared', row)
            if str(old['identity']) != row['id'] or str(prepared['identity']) != row['id']:
                raise ValueError('retained observation/teacher identity mismatch')
            physical.append(old['physical'][:, 0]); original.append(old['u'][:, 0])
            hb.append(prepared['hb']); old_hb.append(old['physical'][:, 1:] / co['hb_factor'])
            hb_native_time.append(prepared['native_time'])
        physical = np.stack(physical, axis=1)
        legacy = voltage_proxy_correction(x, a, physical)
        replay = float(np.max(np.abs(legacy['observed'] - np.stack(original, axis=1))))
        if replay > 1e-6:
            raise ValueError('retained raw-voltage proxy replay failed')
        retained_voltage = read_parent('voltage_prepared', dict(id=entry['id']))
        if (not np.array_equal(retained_voltage['eeg_u'], x[6000:10000].T.astype('float32'))
                or not np.array_equal(retained_voltage['eeg_s'], legacy['corrected'][6000:10000].T.astype('float32'))):
            raise ValueError('current S/O label replay differs from the retained voltage target')
        payload = dict(x=x, eog=eog, legacy_s=legacy['corrected'], legacy_projection=a,
            legacy_proxy=legacy['observed'], legacy_physical=physical,
            hb=np.stack(hb, axis=1), legacy_hb=np.stack(old_hb, axis=1),
            hb_native_time=np.stack(hb_native_time, axis=1),
            roi_indices=np.asarray(roi_indices), channels=np.asarray(names))
        metadata = dict(entry=entry, eeg_unit=native.native_unit, reference=native.reference,
            native_source=str(native.source_path), native_start_sample=first,
            sample_rate_hz=200, support_seconds=90, score_relative_seconds=[30, 50],
            eog_names=list(native.auxiliary_channel_names), channels=names,
            native_proxy_replay_max_abs=replay, legacy_diagnostic=legacy['diagnostic'])
        metadata['legacy_supervision_float32_exact'] = True
        yield entry, payload, metadata


def voltage_power_cap_bank(x, *, sample_rate_hz=200):
    """Original five Butterworth carriers plus their exact voltage complement.

    The complement includes DC and frequencies outside the proxy bands. It is
    constrained too; neither a local mean nor an unmodelled band bypasses S.
    These overlapping carriers are not an orthogonal energy decomposition.
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, float)
    if (sample_rate_hz not in (200,500,1000) or x.ndim != 2
            or len(x) < sample_rate_hz or not np.isfinite(x).all()):
        raise ValueError('finite supported native-rate multichannel voltage required')
    bands = [sosfiltfilt(butter(4, band, fs=sample_rate_hz, btype='bandpass', output='sos'), x, axis=0)
             for band in BANDS]
    return np.stack([*bands, x - np.sum(bands, axis=0)], axis=1)


def voltage_power_cap_frames(bank, *, window=200, hop=20, sample_rate_hz=200):
    """One-second second-moment frames, 100 ms hop, reflected endpoints.

    No frame mean is removed. Reflection is solely a declared finite-window
    edge convention; reported interventions are in the interior [30,50) s.
    """
    bank = np.asarray(bank, float)
    if (bank.ndim != 3 or sample_rate_hz not in (200,500,1000)
            or window != sample_rate_hz or hop != sample_rate_hz//10 or len(bank) % hop
            or not np.isfinite(bank).all()):
        raise ValueError('finite bank and declared one-second/100 ms frame clock required')
    padded = np.pad(bank, ((window//2, window//2), (0, 0), (0, 0)), mode='reflect')
    frames = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)[::hop]
    # [frame, band, channel, local sample], read-only view; time is the centre.
    return frames, np.arange(len(frames)) * hop / sample_rate_hz


def fit_voltage_power_cap_reference(training, projection, *, shrinkage=.1, sample_rate_hz=200):
    """Frozen train-only log-power completion and six second-moment references.

    This is a signal-statistics reference, not estimated clean covariance.
    Each item is one already-admitted training window with native voltage x.
    """
    if (not training or any(item.get('split') != 'train' for item in training)
            or not 0 < shrinkage <= 1):
        raise ValueError('training windows only and positive covariance shrinkage required')
    a = np.asarray(projection, float); c = np.shape(training[0]['x'])[1]
    if (a.shape != (3, c*5) or not np.isfinite(a).all()
            or np.linalg.matrix_rank(a) != 3):
        raise ValueError('full-row-rank three-ROI channel-major projection required')
    features, baselines, correlations, complements = [], [], [], []
    for item in training:
        x = np.asarray(item['x'], float)
        if x.shape != (90*sample_rate_hz, c):
            raise ValueError('each reference window must retain complete 90 s support')
        bank = voltage_power_cap_bank(x,sample_rate_hz=sample_rate_hz)
        power = np.mean(bank[:, :5].reshape(900, sample_rate_hz//10, 5, c)**2, axis=1).transpose(0, 2, 1)
        log = np.log(np.maximum(power, 1e-20)); baseline = log[:50].mean(axis=0)
        features.append((log - baseline)[150:750].reshape(600, -1))
        baselines.append(baseline.reshape(-1))
        frames, time_s = voltage_power_cap_frames(bank,window=sample_rate_hz,hop=sample_rate_hz//10,sample_rate_hz=sample_rate_hz)
        frames = frames[(time_s >= 15) & (time_s < 75)]
        second = np.einsum('nbcl,nbdl->nbcd', frames, frames) / sample_rate_hz
        diagonal = np.diagonal(second, axis1=-2, axis2=-1)
        if np.any(diagonal <= 0):
            raise ValueError('zero training channel/band power cannot define a reference')
        correlations.append(np.mean(second / np.sqrt(diagonal[..., :, None] * diagonal[..., None, :]), axis=0))
        complements.append(diagonal[:, -1])
    ell = np.concatenate(features); mu = ell.mean(axis=0)
    raw = np.cov(ell, rowvar=False, bias=True)
    floor = max(float(np.diag(raw).max()) * 1e-8, 1e-12)
    covariance = (1-shrinkage)*raw + shrinkage*np.diag(np.maximum(np.diag(raw), floor))
    av = a @ covariance
    lift = np.linalg.solve(av @ a.T, av).T
    if not np.allclose(a @ lift, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError('log-power conditional completion lost its equality constraint')
    correlation = (1-shrinkage)*np.mean(correlations, axis=0) + shrinkage*np.eye(c)[None]
    return dict(projection=a, mean=mu, covariance=covariance, lift=lift,
        baseline_log_power=np.median(baselines, axis=0), correlation=correlation,
        complement_power=np.median(np.concatenate(complements), axis=0),
        shrinkage=np.array(shrinkage), sample_rate_hz=np.array(sample_rate_hz), training_windows=np.array(len(training)),
        training_feature_rows=np.array(len(ell)))


def voltage_power_cap_reference(model, proxy, time_s):
    """Complete the reference from frozen statistics; never from current EEG.

    proxy=None is the matched constant-statistics control. Input proxies keep
    their existing 10 Hz, first-five-second-reference coordinates.
    """
    a, mu, lift = (np.asarray(model[k], float) for k in ('projection', 'mean', 'lift'))
    time_s = np.asarray(time_s, float); c = a.shape[1]//5
    if proxy is None:
        ell = np.broadcast_to(mu, (len(time_s), len(mu)))
    else:
        p = np.asarray(proxy, float)
        if p.shape != (900, 3) or not np.isfinite(p).all():
            raise ValueError('three complete finite 10 Hz reconstructed proxy trajectories required')
        aligned = np.stack([np.interp(time_s, np.arange(900)/10, p[:, j]) for j in range(3)], axis=1)
        ell = mu + (aligned - a @ mu) @ lift.T
        if not np.allclose(ell @ a.T, aligned, rtol=1e-10, atol=1e-11):
            raise ValueError('completed reference does not reproduce the proxy coordinate')
    with np.errstate(over='raise', invalid='raise'):
        power = np.exp(ell + model['baseline_log_power']).reshape(-1, c, 5).transpose(0, 2, 1)
    power = np.concatenate((power, np.broadcast_to(model['complement_power'], (len(time_s), 1, c))), axis=1)
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError('invalid reference power; no clipping of extrapolated physical references')
    return power


def project_voltage_second_moment(x, reference, cap):
    """Exact weighted projection onto S S' / L <= cap * reference.

    x is [...,channel,sample]. The formula is singular-value clipping in
    reference-whitened coordinates, not an artifact/source classifier.
    """
    x, reference = np.asarray(x, float), np.asarray(reference, float)
    if (reference.shape != x.shape[:-1] + (x.shape[-2],) or not np.isfinite(x).all()
            or not np.isfinite(reference).all() or not np.isfinite(cap) or cap <= 0
            or not np.allclose(reference, np.swapaxes(reference, -1, -2), rtol=1e-11, atol=1e-12)):
        raise ValueError('finite matched symmetric reference and positive power cap required')
    diagonal = np.diagonal(reference, axis1=-2, axis2=-1)
    if np.any(diagonal <= 0):
        raise ValueError('strictly positive reference covariance required')
    # Separate channel units before factoring C=D R D. Any exact whitening
    # gives the same weighted projection; symmetric square roots are not needed.
    # SVD of the whitened voltage avoids squaring its condition number in XX'.
    scale = np.sqrt(diagonal)
    correlation = reference / scale[..., :, None] / scale[..., None, :]
    try:
        root = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError('strictly positive reference covariance required') from exc
    whitened = np.linalg.solve(root, x / scale[..., :, None])
    left, singular, right = np.linalg.svd(whitened, full_matrices=False)
    limit = np.sqrt(cap*x.shape[-1])
    clipped = np.minimum(singular, limit)
    transformed = (left * clipped[..., None, :]) @ right
    result = scale[..., :, None] * (root @ transformed)
    # The unmodified feasible branch is exactly identity, including its mean.
    feasible = singular[..., 0] <= limit
    result = np.where(feasible[..., None, None], x, result)
    bounded = transformed @ np.swapaxes(transformed, -1, -2) / x.shape[-1]
    after = np.linalg.eigvalsh(bounded)[..., -1]
    gain = np.ones_like(singular)
    np.divide(limit, singular, out=gain, where=singular > limit)
    return result, dict(max_input_eigenvalue=singular[..., 0]**2/x.shape[-1],
        max_output_eigenvalue=after, clipped_directions=np.sum(singular > limit, axis=-1),
        minimum_gain=gain.min(axis=-1))


def voltage_power_cap_target(x, model, proxy, *, cap=1., audit_final=True):
    """Cap every local carrier and overlap-add; exact residual voltage O=x-S.

    The proven cap applies to individual local carrier blocks. Refiltering the
    final sum can change those blocks; its violations are measured, never hidden
    or promoted to a global bound. No mean or out-of-band voltage is added back.
    """
    rate=int(model.get('sample_rate_hz',200));hop=rate//10
    x = np.asarray(x, float); bank = voltage_power_cap_bank(x,sample_rate_hz=rate)
    frames, time_s = voltage_power_cap_frames(bank,window=rate,hop=hop,sample_rate_hz=rate)
    power = voltage_power_cap_reference(model, proxy, time_s)
    d = np.sqrt(power)
    covariance = d[..., :, None] * model['correlation'][None] * d[..., None, :]
    estimated, diagnostic = project_voltage_second_moment(frames, covariance, cap)
    if np.max(diagnostic['max_output_eigenvalue']) > cap*(1+1e-8):
        raise AssertionError('local spectral cap was not satisfied')
    window = np.hanning(rate+1)[:-1]
    accumulated = np.zeros((len(x)+rate, x.shape[1])); weight = np.zeros(len(x)+rate)
    for k, block in enumerate(estimated):
        start = k*hop
        accumulated[start:start+rate] += block.sum(axis=0).T * window[:, None]
        weight[start:start+rate] += window
    section = slice(rate//2, rate//2+len(x))
    s = accumulated[section]/weight[section, None]
    o = x-s
    post = np.empty((0, 6))
    if audit_final:
        final_frames, _ = voltage_power_cap_frames(voltage_power_cap_bank(s,sample_rate_hz=rate),
            window=rate,hop=hop,sample_rate_hz=rate)
        _, check = project_voltage_second_moment(final_frames, covariance, cap)
        post = check['max_input_eigenvalue']/cap
    return dict(s=s, o=o, reference_power=power, frame_time_s=time_s,
        local_input_ratio=diagnostic['max_input_eigenvalue']/cap,
        local_output_ratio=diagnostic['max_output_eigenvalue']/cap,
        clipped_directions=diagnostic['clipped_directions'],
        minimum_gain=diagnostic['minimum_gain'], final_reanalysis_ratio=post,
        closure_max_abs=np.array(np.max(abs(s+o-x))))


def voltage_stft(x, spec):
    """All Fourier bins, zero-extended Hann analysis with exact overlap-add."""
    from scipy.signal import stft
    value = np.asarray(x, float)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('finite time-by-channel voltage required')
    n = spec['stft']['samples']; hop = spec['stft']['hop']
    f, t, z = stft(value, fs=200, nperseg=n, noverlap=n-hop, window='hann',
        boundary='zeros', padded=True, axis=0)
    return f, t, z.transpose(2, 0, 1)


def voltage_istft(z, length, spec):
    from scipy.signal import istft
    n = spec['stft']['samples']; hop = spec['stft']['hop']
    _, x = istft(np.asarray(z).transpose(2, 1, 0), fs=200, nperseg=n,
        noverlap=n-hop, window='hann', boundary=True)
    return x.T[:length]


def voltage_eog_reference(eog, spec):
    from scipy.signal import butter, sosfiltfilt
    value = np.asarray(eog, float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError('two finite measured EOG references required')
    return sosfiltfilt(butter(4, spec['eog_band_hz'], fs=200, btype='bandpass', output='sos'), value, axis=0)


def fit_voltage_noise_model(training, spec):
    """Estimate one transfer and full-rank spectral background on train only.

    EOG regression supplies a noise *covariance*, with zero conditional mean
    in the Wiener arm. It is an uncertain nuisance model, never clean truth.
    """
    from scipy.signal import butter, sosfiltfilt
    if not training or any(a.get('split') != 'train' for a in training):
        raise ValueError('noise calibration may use training subjects only')
    c = np.shape(training[0]['x'])[1]
    ee = np.zeros((2, 2)); ex = np.zeros((2, c))
    filtered = []
    for item in training:
        e = voltage_eog_reference(item['eog'], spec)
        x = np.asarray(item['x'], float)
        if x.shape != (18000, c):
            raise ValueError('matching native EEG channels and support required')
        low = sosfiltfilt(butter(4, spec['eog_band_hz'], fs=200, btype='bandpass', output='sos'), x, axis=0)
        ee += e.T @ e; ex += e.T @ low
        filtered.append((x, e))
    b = np.linalg.solve(ee + spec['regression_ridge'] * np.diag(np.diag(ee)), ex)
    reference_frames, residuals, powers = [], [], []
    for x, e in filtered:
        _, t, ze = voltage_stft(e, spec)
        energy = np.sum(abs(ze)**2, axis=(1, 2))
        residual = x - e @ b
        _, _, zr = voltage_stft(residual, spec)
        reference_frames.append(energy); residuals.append(zr)
        powers.append(eeg_band_power(residual, bands=BANDS, sample_rate=200, target_rate=10))
    threshold = float(np.quantile(np.concatenate(reference_frames), spec['low_eog_quantile']))
    cov = np.zeros((residuals[0].shape[1], c, c), complex); count = 0
    for energy, z in zip(reference_frames, residuals):
        low = energy <= threshold
        cov += np.einsum('tfc,tfd->fcd', z[low], z[low].conj()); count += low.sum()
    if count < c:
        raise ValueError('insufficient low-EOG training frames')
    cov /= count
    diagonal = np.maximum(np.real(np.diagonal(cov, axis1=-2, axis2=-1)), 1e-14)
    correlation = cov / np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    shrink = spec['signal_covariance_shrinkage']
    correlation = (1-shrink)*correlation + shrink*np.eye(c)[None]
    return dict(transfer=b, correlation=correlation,
        spectral_floor=np.maximum(diagonal*spec['power_floor_fraction'], 1e-12),
        band_floor=np.maximum(np.median(np.concatenate(powers), axis=0)*spec['power_floor_fraction'], 1e-12),
        voltage_scale=np.sqrt(np.mean(np.concatenate([x*x for x, _ in filtered]), axis=0)),
        low_eog_threshold=np.asarray(threshold), low_eog_training_frames=np.asarray(count))


def voltage_power_evidence(x, eog, model, spec):
    """Return noise-subtracted 100 ms band power in the new reference gauge."""
    noise = voltage_eog_reference(eog, spec) @ model['transfer']
    px = eeg_band_power(x, bands=BANDS, sample_rate=200, target_rate=10)
    pn = eeg_band_power(noise, bands=BANDS, sample_rate=200, target_rate=10)
    pe = np.maximum(px-pn, model['band_floor'])
    log = np.log(pe); reference = log[:50].mean(axis=0)
    ell = (log-reference).reshape(len(log), -1)
    rho = np.clip(pn / np.maximum(px, model['band_floor']), 0, 1).reshape(len(log), -1)
    return dict(ell=ell, power=pe, noise=noise, noise_fraction=rho,
        floor_fraction=float(np.mean(px-pn <= model['band_floor'])), reference=reference)


def fuse_voltage_power(ell, noise_fraction, projection, physical, variance, spec):
    """Precision-weighted low-rank log-power correction; no voltage synthesis."""
    a = np.asarray(projection, float); target = np.asarray(physical, float)
    value = np.asarray(ell, float); variance = np.asarray(variance, float)
    if (a.ndim != 2 or a.shape[1] != value.shape[1] or target.shape != (len(value), len(a))
            or variance.shape != (len(a),) or np.any(variance <= 0)
            or not all(np.isfinite(v).all() for v in (a, target, value, variance))):
        raise ValueError('finite same-gauge physical target, projection and positive variance required')
    settings = spec['log_power_variance']
    sigma = settings['minimum'] + settings['noise_fraction_multiplier']*np.asarray(noise_fraction)
    matrix = np.einsum('rc,tc,sc->trs', a, sigma, a) + np.diag(variance)[None]
    correction = sigma * (np.linalg.solve(matrix, (target-value @ a.T)[..., None])[..., 0] @ a)
    bound = spec['maximum_log_power_change']
    return np.clip(correction, -bound, bound), float(np.mean(abs(correction) > bound))


def voltage_wiener_target(x, eog, model, spec, *, projection=None, physical=None,
                          physical_variance=None):
    """Full-spectrum multichannel voltage posterior and exact residual label."""
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(x, float)
    evidence = voltage_power_evidence(x, eog, model, spec)
    f, t, z = voltage_stft(x, spec)
    _, _, zn = voltage_stft(evidence['noise'], spec)
    width = spec['spectral_smoothing_frames']
    kn = gaussian_filter1d(zn[..., :, None]*zn[..., None, :].conj(), width, axis=0, mode='nearest')
    observed = gaussian_filter1d(abs(z)**2, width, axis=0, mode='nearest')
    ps = np.maximum(observed - np.real(np.diagonal(kn, axis1=-2, axis2=-1)), model['spectral_floor'])
    delta = np.zeros_like(evidence['ell']); clipped = 0.
    if physical is not None:
        delta, clipped = fuse_voltage_power(evidence['ell'], evidence['noise_fraction'],
            projection, physical, physical_variance, spec)
        changes = delta.reshape(len(delta), x.shape[1], 5)
        for band, (lo, hi) in enumerate(BANDS):
            bins = (f >= lo) & (f < hi if band < 4 else f <= hi)
            for channel in range(x.shape[1]):
                factor = np.exp(np.interp(t, (np.arange(len(delta))+.5)/10, changes[:, channel, band]))
                ps[:, bins, channel] *= factor[:, None]
    d = np.sqrt(ps)
    ks = d[..., :, None] * model['correlation'][None] * d[..., None, :]
    # A positive-definite signal covariance retains all sensor directions.
    estimated = (ks @ np.linalg.solve(ks+kn, z[..., None]))[..., 0]
    s = voltage_istft(estimated, len(x), spec)
    o = x-s
    return dict(s=s, o=o, ell=evidence['ell'], delta=delta,
        noise=evidence['noise'], noise_fraction=evidence['noise_fraction'],
        power_floor_fraction=evidence['floor_fraction'], clipped_fraction=clipped,
        closure_max_abs=float(np.max(abs(s+o-x))))


def paired_voltage_targets(x, s, disturbance):
    """The augmentation contract, deliberately independent of teacher fitting."""
    x, s, n = (np.asarray(v, float) for v in (x, s, disturbance))
    if x.shape != s.shape or x.shape != n.shape or not all(np.isfinite(v).all() for v in (x, s, n)):
        raise ValueError('finite identical native voltage shapes required')
    return dict(x=x+n, s=s.copy(), o=x-s+n)


def voltage_teacher_intervention(x, eog, donor, model, spec, kind, amplitude, seed):
    """Known added perturbations; no validation fit and no clean-EEG claim."""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    t = np.arange(len(x))/200; channels = x.shape[1]
    envelope = np.exp(-.5*((t-40.)/2.)**2)
    added_reference = np.zeros_like(eog)
    if kind == 'ocular':
        added_reference = np.roll(voltage_eog_reference(donor, spec), 1400, axis=0)*envelope[:, None]
        mismatch = 1 + .3*np.sin(np.arange(channels)*1.7+.4)
        n = (added_reference @ model['transfer'])*mismatch
    elif kind == 'unreferenced_muscle':
        carrier = sosfiltfilt(butter(4, [25, 70], fs=200, btype='bandpass', output='sos'),
            rng.normal(size=x.shape), axis=0)
        n = carrier*envelope[:, None]
    elif kind == 'neural_alpha':
        topology = np.cos(np.linspace(-1, 1, channels)*np.pi/2)
        n = (np.sin(2*np.pi*10*t+.3)*envelope)[:, None]*topology
    else:
        raise ValueError('unknown prespecified perturbation')
    score = slice(6000, 10000)
    scale = amplitude*np.sqrt(np.mean(np.asarray(model['voltage_scale'])**2)) / np.sqrt(np.mean(n[score]**2))
    return n*scale, np.asarray(eog)+added_reference*scale


def load_eeg_residual_development(parent, *, datasets=DATASETS, channels=16, length=4000):
    """Validate the retained development boundary before opening EEG arrays.

    Hb, teacher trajectories, test caches and raw datasets are not inputs to
    this consumer. S and O are fixed supervised targets, never encoder inputs.
    """
    folder = Path(parent) / 'voltage_cache'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (manifest.get('schema') != 'dual_component_voltage_cache_v1'
            or manifest.get('protected_data') != 'forbidden' or manifest.get('final', False)):
        raise ValueError('retained public development voltage cache required')
    rows = manifest['rows']['eeg']
    if (not rows or {r['dataset'] for r in rows} != set(datasets)
            or {r['split'] for r in rows} != {'train', 'validation'}
            or len({r['id'] for r in rows}) != len(rows)):
        raise ValueError('unique train/validation EEG identities required; test is forbidden')
    subjects = defaultdict(set)
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        subjects[(row['dataset'], row['subject'])].add(row['split'])
        grouped[row['trial_id']].append(i)
        if manifest['coordinates']['eeg'][row['coordinate_index']] != row['coordinate']:
            raise ValueError('coordinate identity mismatch')
    if any(len(splits) != 1 for splits in subjects.values()):
        raise ValueError('subject crosses the development split')
    trials = []
    for trial, indices in sorted(grouped.items()):
        group = [rows[i] for i in indices]
        if (len(group) != channels or len({r['channel'] for r in group}) != channels
                or len({(r['dataset'], r['subject'], r['record'], r['split']) for r in group}) != 1):
            raise ValueError('one complete unique-electrode trial required')
        trials.append({**{k: group[0][k] for k in ('dataset', 'subject', 'record', 'split')},
                       'trial_id': trial, 'indices': indices})
    expected = (len(rows), length, 1)
    for key in ('eeg_u', 'eeg_s', 'eeg_o'):
        if manifest['arrays'][key] != key + '.npy':
            raise ValueError('unexpected retained EEG array path')
        if 'shapes' in manifest and tuple(manifest['shapes'][key]) != expected:
            raise ValueError('retained voltage shape contract mismatch')
    arrays = {key: np.load(folder / manifest['arrays'][key], mmap_mode='r', allow_pickle=False)
              for key in ('eeg_u', 'eeg_s', 'eeg_o')}
    if any(value.shape != expected or value.dtype != np.float32 for value in arrays.values()):
        raise ValueError('float32 retained EEG target shapes required')
    return manifest, arrays, trials


def eeg_residual_band_features(observed, *, sample_rate=200, hop=20):
    """Observed-only five-band features on the caller's complete support.

    Zero-phase IIR filtering reads the full supplied window. A shorter gain
    network context does not shorten this raw-input support. No target or
    other modality is accepted by this function.
    """
    from scipy.signal import butter, sosfiltfilt
    value = np.asarray(observed, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] % hop or sample_rate != 200 or not np.isfinite(value).all():
        raise ValueError('finite [electrode-case,time] 200 Hz observations required')
    bands = np.stack([sosfiltfilt(butter(4, limits, btype='bandpass', fs=sample_rate, output='sos'),
                                  value, axis=1) for limits in BANDS], axis=-1).astype('float32')
    power = np.mean(bands.astype('float64').reshape(len(value), -1, hop, 5) ** 2, axis=2)
    return bands, np.log(np.maximum(power, 1e-12)).astype('float32')


def load_retained_eeg_projection(path, metadata, expected_rows):
    """Read only the observed EEG proxy and its frozen training projection.

    The native producer separately replay-checked this observed proxy. Its
    original 90 s EEG filtering and five-second reference remain part of the
    input support; the adjacent teacher/Hb fields in the NPZ are not read.
    """
    replay_error=float(metadata.get('diagnostic', {}).get('native_proxy_replay_max_abs', float('inf')))
    if (not expected_rows or {r['split'] for r in expected_rows} - {'train', 'validation'}
            or len({r['trial_id'] for r in expected_rows}) != 1
            or metadata.get('trial_id') != expected_rows[0]['trial_id']
            or metadata.get('eeg_channels') != [r['channel'] for r in expected_rows]
            or not np.isfinite(replay_error) or replay_error > 1e-6):
        raise ValueError('verified matching development EEG proxy identity required')
    with np.load(path, allow_pickle=False) as archive:
        observed = np.asarray(archive['original_proxy'], dtype=float)
        projection = np.asarray(archive['projection'], dtype=float)
    if (observed.shape != (200, 3) or projection.shape != (3, len(expected_rows) * 5)
            or not np.isfinite(observed).all() or not np.isfinite(projection).all()):
        raise ValueError('finite observed EEG projection shapes required')
    row_scale = np.linalg.norm(projection, axis=1)
    if (row_scale < 1e-12).any():raise ValueError('degenerate observed EEG projection')
    return (observed / row_scale).astype('float32'), np.linalg.pinv(projection / row_scale[:, None], rcond=1e-10).astype('float32')


def voltage_proxy_correction(voltage, projection, target, *, strength=1.,
                             max_log_gain=.7, backtracking=(1., .5, .25, .125)):
    """Condition native voltage on frozen, possibly overlapping ROI proxies.

    ``projection`` maps channel-major five-band referenced log power to each
    ROI's original SSM coordinate. All ROIs use one correction per electrode.
    Only the central 20 s and a 2 s cosine margin change; the 5 s reference is
    retained. Backtracking uses the *recomputed* proxy, never another SSM fit.
    This is a conditional voltage correction, not an inverse SSM observation.
    """
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(voltage, float); a = np.asarray(projection, float)
    target = np.asarray(target, float)
    if (x.ndim != 2 or x.shape[0] != 18000 or a.ndim != 2
            or a.shape[1] != x.shape[1] * 5 or target.shape != (900, len(a))
            or not all(np.isfinite(v).all() for v in (x,a,target))
            or not 0 <= strength <= 1 or max_log_gain <= 0):
        raise ValueError('finite 90 s/200 Hz voltage, frozen ROI projection and 10 Hz targets required')
    def features(value):
        p = eeg_band_power(value, bands=BANDS, sample_rate=200, target_rate=10)
        if np.any(p <= 0): raise ValueError('nonpositive native band power')
        z = np.log(p).reshape(900,-1)
        return z-z[:50].mean(0)
    original = features(x); observed = original @ a.T
    row_scale = np.linalg.norm(a,axis=1)
    if np.any(row_scale < 1e-12): raise ValueError('degenerate proxy projection')
    normal = a/row_scale[:,None]
    difference = (target-observed)/row_scale
    delta = difference @ np.linalg.pinv(normal, rcond=1e-10).T
    feature_constraint_error = float(np.max(np.abs(delta @ a.T-(target-observed))))
    delta = gaussian_filter1d(delta.reshape(900,x.shape[1],5),sigma=1.,axis=0,mode='nearest')
    # Smooth application margins are outside the scored [0,20) event interval.
    clock = np.arange(900)*.1-30.; envelope=np.zeros(900)
    envelope[(clock>=0)&(clock<20)]=1.
    left=(clock>=-2)&(clock<0);right=(clock>=20)&(clock<22)
    envelope[left]=.5-.5*np.cos(np.pi*(clock[left]+2)/2)
    envelope[right]=.5+.5*np.cos(np.pi*(clock[right]-20)/2)
    native_clock=np.arange(18000)/200.; bin_clock=(np.arange(900)+.5)/10
    band_signal=np.stack([sosfiltfilt(butter(4,[low,high],btype='bandpass',fs=200,output='sos'),x,axis=0)
        for low,high in BANDS],axis=-1)
    before=float(np.mean(((observed[STUDENT]-target[STUDENT])/row_scale)**2))
    attempted=[]; best=None
    if strength == 0 or before < 1e-16:
        return dict(corrected=x.copy(),remaining=np.zeros_like(x),observed=observed,
            corrected_proxy=observed.copy(),diagnostic=dict(status='identity',strength=0.,before=before,after=before,
                feature_constraint_error=feature_constraint_error,attempted=[],maximum_log_gain=0.))
    for fraction in backtracking:
        alpha=float(strength*fraction)
        unconstrained=.5*alpha*delta*envelope[:,None,None]
        log_gain=np.clip(unconstrained,-max_log_gain,max_log_gain)
        gain=np.empty_like(band_signal)
        for c in range(x.shape[1]):
            for b in range(5):
                gain[:,c,b]=np.exp(np.interp(native_clock,bin_clock,log_gain[:,c,b],left=0.,right=0.))
        corrected=x+np.sum((gain-1)*band_signal,axis=-1)
        actual=features(corrected) @ a.T
        after=float(np.mean(((actual[STUDENT]-target[STUDENT])/row_scale)**2))
        record=dict(strength=alpha,after=after,relative_proxy_reduction=1-after/before,
            clipped_fraction=float(np.mean(np.abs(unconstrained)>max_log_gain)),maximum_log_gain=float(np.max(np.abs(log_gain))))
        attempted.append(record)
        if after < before and np.isfinite(corrected).all():
            best=dict(corrected=corrected,remaining=x-corrected,observed=observed,corrected_proxy=actual,
                diagnostic=dict(record,status='corrected',before=before,feature_constraint_error=feature_constraint_error,
                    attempted=attempted,projection_rank=int(np.linalg.matrix_rank(normal))))
            break
    if best is None:
        # Explicit non-admission, not a silently relabelled zero correction.
        return dict(corrected=x.copy(),remaining=np.zeros_like(x),observed=observed,
            corrected_proxy=observed.copy(),diagnostic=dict(status='no_improving_correction',strength=0.,
                before=before,after=before,attempted=attempted,feature_constraint_error=feature_constraint_error))
    return best


def prepare_voltage_record(project_root, cfg, rows, coordinates, read_teacher,
                           *, strength=1., max_log_gain=.7,backtracking=(1.,.5,.25,.125)):
    """Yield unique-electrode corrected targets using only admitted native data."""
    root,record=_registered_record(project_root,cfg,rows)
    native=load_native_eeg_record(root,record)
    if native.sample_rate_hz != 200 or native.native_unit not in ('uV','µV','μV') or not native.unit_evidence:
        raise ValueError('verified native 200 Hz microvolt EEG required')
    channel_lookup={name.upper():i for i,name in enumerate(native.channel_names)}
    by_trial=defaultdict(list)
    for row in rows:by_trial[row['trial_id']].append(row)
    for trial,group in by_trial.items():
        try:
            if len(group)!=3 or len({r['eeg_start_s'] for r in group})!=1:
                raise ValueError('complete aligned three-ROI trial required')
            group=sorted(group,key=lambda r:r['coordinate'])
            names=sorted({n for r in group for n in r['anchor']['eeg_channels']},key=lambda n:channel_lookup[n.upper()])
            start=round(group[0]['eeg_start_s']*200)
            x=np.asarray(native.values[start:start+18000,[channel_lookup[n.upper()] for n in names]],float)
            if x.shape!=(18000,len(names)):raise ValueError('complete 90 s raw support required')
            a=np.zeros((len(group),len(names)*5));target=[];original=[];hb=[];hb_s=[]
            for i,row in enumerate(group):
                coordinate=coordinates[row['coordinate']]
                loading=np.asarray(coordinate['pc']).reshape(6,5)*coordinate['eeg_factor']
                for j,name in enumerate(row['anchor']['eeg_channels']):
                    c=names.index(name);a[i,c*5:(c+1)*5]=loading[j]
                teacher=read_teacher(row)
                if str(teacher['identity'])!=row['id']:raise ValueError('retained teacher identity mismatch')
                target.append(teacher['physical'][:,0]);original.append(teacher['u'][:,0])
                hb.append(teacher['u'][STUDENT,1:]);hb_s.append(teacher['physical'][STUDENT,1:])
            target=np.stack(target,axis=1);original=np.stack(original,axis=1)
            result=voltage_proxy_correction(x,a,target,strength=strength,max_log_gain=max_log_gain,backtracking=backtracking)
            lineage_error=float(np.max(np.abs(result['observed']-original)))
            if lineage_error>1e-6:raise ValueError(f'native proxy replay mismatch: {lineage_error}')
            if result['diagnostic']['status']=='no_improving_correction':
                yield trial,None,dict(error='no_improving_correction',diagnostic=result['diagnostic']);continue
            eeg=x[6000:10000].T.astype('float32');s=result['corrected'][6000:10000].T.astype('float32')
            payload=dict(eeg_u=eeg,eeg_s=s,eeg_o=eeg-s,hb_u=np.asarray(hb,dtype='float32'),
                hb_s=np.asarray(hb_s,dtype='float32'),projection=a,
                original_proxy=result['observed'][STUDENT],target_proxy=target[STUDENT],
                corrected_proxy=result['corrected_proxy'][STUDENT])
            payload['hb_o']=payload['hb_u']-payload['hb_s']
            metadata=dict(trial_id=trial,rows=group,eeg_channels=names,
                native_unit=native.native_unit,native_source=str(native.source_path),
                reference='unchanged dataset-native reference; no additional re-referencing',
                diagnostic=dict(result['diagnostic'],native_proxy_replay_max_abs=lineage_error),
                interpretation='S=SSM-guided conditional voltage correction; O=voltage correction residual; no voltage physiological-truth claim')
            yield trial,payload,metadata
        except Exception as exc:
            yield trial,None,dict(error=f'{type(exc).__name__}: {exc}')


def _json(value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value, default=convert, sort_keys=True, ensure_ascii=False)


def _check_scope(project_root, cfg):
    """Resolve the one admitted public parent before any measured array read."""
    root = Path(project_root).resolve()
    data = cfg['data']
    tensor = cfg['tensor']
    parent = (root / cfg['parent_run']).resolve()
    if (parent != (root / PARENT).resolve()
            or tuple(data['datasets']) != DATASETS
            or data['protected_data'] != 'forbidden'
            or data.get('labels_as_training_inputs') is not False
            or data.get('fnirs_input') != CACHED_FNIRS_INPUT_SCHEMA
            or (root / data['cache_root']).resolve() != (root / 'data/cache/physiology_semantic_clean_v5').resolve()
            or cfg['seed'] != 20261010
            or data['train_fraction'] != .6 or data['validation_fraction'] != .2):
        raise ValueError('dual tokenizer requires its exact public parent, v5/none cache and frozen subject split')
    expected = dict(teacher_seconds=90, teacher_steps=900, dt_s=.1,
        student_steps=[300, 500], student_seconds=20, patch_seconds=2,
        patches=10, eeg_rate_hz=200)
    if (any(tensor.get(key) != value for key, value in expected.items())
            or tuple(map(tuple, tensor.get('eeg_bands_hz', ()))) != BANDS):
        raise ValueError('90 s native-recomputed 10 Hz teacher and local two-second input contract required')
    return root, parent


def metadata_inventory(project_root, cfg):
    """Check the full 5,008-trial metadata plan without opening signal arrays."""
    root, parent = _check_scope(project_root, cfg)
    retained = json.loads((parent / 'inventory.json').read_text())
    current = inventory(root, cfg)
    if retained['rows'] != current['rows'] or retained['subjects'] != current['subjects']:
        raise ValueError('current registry identities/geometry/splits differ from frozen parent inventory')
    return dict(current, schema='dual_component_tokenizer_inventory_v1',
        parent_run=str(parent), parent_inventory_schema=retained['schema'],
        planned_trials=len({r['trial_id'] for r in current['rows']}),
        parent_teacher_success_filter=False, measured_arrays_read=0,
        information_scope='local_modality_patch_with_fixed_Hb_reference; processed_coordinate_completion')


def _registered_record(project_root, cfg, rows):
    from .clean_physiology_cache import CleanPhysiologyCacheIndex
    from .registry import REGISTERED_DATASETS
    root, parent = _check_scope(project_root, cfg)
    admitted = {r['id']: r for r in json.loads((parent / 'inventory.json').read_text())['rows']}
    if (not rows or len({r['join_key'] for r in rows}) != 1
            or any(admitted.get(r['id']) != r for r in rows)
            or any(r['split'] not in ('train', 'validation', 'test') for r in rows)):
        raise ValueError('exact parent public identities and subject partitions required before array access')
    index = CleanPhysiologyCacheIndex(root / cfg['data']['cache_root'])
    matches = [r for r in index.records if r.join_key == rows[0]['join_key']]
    if len(matches) != 1:
        raise ValueError('unique central registry record required')
    record = matches[0]
    if any(r['source_identity'] != record.manifest['source_files'] for r in rows):
        raise ValueError('registered source identity changed')
    admitted_root = (root / REGISTERED_DATASETS[record.dataset_id].default_root).resolve()
    for source in record.manifest['source_files']:
        if not (root / source['path']).resolve().is_relative_to(admitted_root):
            raise ValueError('source is outside the registered public dataset root')
    return root, record


def prepare_cached_record(project_root, cfg, rows):
    """Yield (row, arrays, error), reading a native record only once per group."""
    root, record = _registered_record(project_root, cfg, rows)
    # Both identity and the cache contract are checked before either reader.
    from .clean_physiology_cache import require_measurement_producer
    require_measurement_producer(record.manifest)
    bandpass = record.manifest.get('homer2_aligned_contract', {}).get('quality', {}).get('bandpass', {})
    if (record.manifest.get('processing_schema') != 'physiology_measurement_alignment_v5'
            or record.manifest.get('motion_method') != 'none'
            or bandpass.get('status') != 'applied'
            or (bandpass.get('low_hz'), bandpass.get('high_hz')) != (.01, .2)):
        raise ValueError('current v5/none bandpassed measurement cache required before reading EEG')
    native = load_native_eeg_record(root, record)
    cached = load_cached_fnirs_record(root, record)
    if (native.sample_rate_hz != 200 or native.native_unit not in ('uV', 'µV', 'μV')
            or not native.unit_evidence
            or list(native.channel_names) != record.manifest['measurement']['eeg_channel_names']):
        raise ValueError('verified 200 Hz native EEG units and exact channels required')
    channels = {name.upper(): i for i, name in enumerate(native.channel_names)}
    for row in rows:
        try:
            start = round(row['eeg_start_s'] * 200)
            selected = [channels[n.upper()] for n in row['anchor']['eeg_channels']]
            e = np.asarray(native.values[start:start + 18000, selected], float)
            pair = int(row['anchor']['hb_pair'])
            if cached['channel_names'][2 * pair] != row['anchor']['hb_channel'] + '_HbO':
                raise ValueError('registered Hb channel identity mismatch')
            hb_start, hb_end = row['hb_start_s'], row['hb_start_s'] + 90
            # Marker milliseconds and the cache sample grid can represent the
            # same endpoint one ULP apart (108.80000000000001 vs 108.8).
            # Admit only that floating-point roundoff, at both half-open
            # boundaries. Do not enlarge a genuine window or invent support.
            clock_tolerance = 8 * np.spacing(max(abs(hb_start), abs(hb_end), 1.))
            take = ((cached['time_s'] >= hb_start - clock_tolerance)
                & (cached['time_s'] < hb_end - clock_tolerance))
            times = cached['time_s'][take] - hb_start
            times[np.abs(times) <= clock_tolerance] = 0.
            h = cached['values'][take, pair]
            if (start < 0 or e.shape != (18000, 6) or not np.isfinite(e).all()
                    or not np.isfinite(h).all()):
                raise ValueError('complete finite native EEG and supported cached Hb required')
            power = eeg_band_power(e, bands=BANDS, sample_rate=200, target_rate=10)
            if power.shape != (STEPS, 6, 5) or np.any(power <= 0) or not np.isfinite(power).all():
                raise ValueError('native-recomputed 10 Hz five-band power required')
            features = np.log(power).reshape(STEPS, 30)
            features -= features[:50].mean(axis=0)
            # Means use disjoint cached 100 ms bins, then the fixed reference.
            groups = np.floor(np.maximum(times, 0.) / DT + 1e-10).astype(int)
            counts = np.bincount(groups, minlength=STEPS)
            if len(counts) != STEPS or np.any(counts == 0):
                raise ValueError('every 100 ms bin requires recorded cached Hb support')
            hb = np.zeros((STEPS, 2), float)
            np.add.at(hb, groups, h)
            hb /= counts[:, None]
            hb -= hb[:50].mean(axis=0)
            provenance = dict(canonical_identity=row, native_EEG_source=str(native.source_path),
                eeg_channels=row['anchor']['eeg_channels'], eeg_unit=native.native_unit,
                eeg_unit_evidence=native.unit_evidence, eeg_rate_hz=200,
                native_start_sample=start, start_rounding_s=start / 200 - row['eeg_start_s'],
                teacher_EEG_filter_support_event_s=[-30, 60], teacher_proxy_rate_hz=10,
                teacher_bands_hz=BANDS, teacher_reference_event_s=[-30, -25],
                student_EEG_support_event_s=[0, 20], student_patch_seconds=2,
                student_Hb_reference_event_s=[-30, -25], hb=cached['provenance'],
                hb_clock_roundoff_tolerance_s=clock_tolerance,
                inference='offline_processed_coordinate_completion_not_raw_sample_isolation')
            payload = dict(eeg=e[6000:10000].astype('float32'), features=features, hb=hb,
                native_time=times, time_s=np.arange(STEPS) * DT,
                eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA), eeg_proxy_rate_hz=np.array(10.),
                fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA),
                identity=np.array(row['id']), trial_id=np.array(row['trial_id']),
                provenance=np.array(_json(provenance)))
            yield row, payload, None
        except (ValueError, KeyError, IndexError, FloatingPointError) as exc:
            yield row, None, str(exc)


def _check_prepared(arrays):
    if (str(np.asarray(arrays.get('eeg_proxy_input', ''))) != EEG_PROXY_INPUT_SCHEMA
            or float(np.asarray(arrays.get('eeg_proxy_rate_hz', np.nan))) != 10.
            or str(np.asarray(arrays.get('fnirs_input', ''))) != CACHED_FNIRS_INPUT_SCHEMA
            or np.shape(arrays.get('features')) != (STEPS, 30)
            or np.shape(arrays.get('hb')) != (STEPS, 2)
            or np.shape(arrays.get('eeg')) != (4000, 6)
            or not np.array_equal(arrays.get('time_s'), np.arange(STEPS) * DT)):
        raise ValueError('versioned 90 s 10 Hz prepared features and 20 s native EEG required')
    if any(not np.isfinite(arrays[k]).all() for k in ('features', 'hb', 'eeg')):
        raise ValueError('finite complete prepared support required')


def fit_training_coordinates(rows, read_arrays):
    """Fit region/dataset PCA, common Hb scale and native-input SD on train only."""
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'train': groups[row['coordinate']].append(row)
    coordinates = {}
    for key, group in groups.items():
        chunks = []
        count, total, squared = 0, 0., 0.
        for row in group:
            a = read_arrays(row)
            _check_prepared(a)
            chunks.append((a['features'], a['hb']))
            e = np.asarray(a['eeg'], float)
            total += float(e.sum()); squared += float(np.square(e).sum()); count += e.size
        co = fit_feature_coordinate(chunks)
        co.update(eeg_mean=total / count,
            eeg_scale=max(float(np.sqrt(max(squared / count - (total / count) ** 2, 0))), 1e-8),
            training_ids=[r['id'] for r in group],
            training_subjects=sorted({r['subject'] for r in group}),
            eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA, eeg_proxy_rate_hz=10., dt_s=DT,
            fnirs_input=CACHED_FNIRS_INPUT_SCHEMA)
        coordinates[key] = co
    if {r['coordinate'] for r in rows} != set(coordinates):
        raise ValueError('each evaluation coordinate requires training-subject support')
    return coordinates


def produce_teacher(arrays, coordinate, tak_cfg, *, arm_name='tak_fixed', warm_start=None,
                    eeg_visible=True, observation_weights=None):
    """Fit the declared Tak arm; return observation-space u, S and full residual.

    Fixed-reference Tak is the default engineering teacher. ``tak_fit`` may be
    requested explicitly in the versioned run, retaining per-window privileged
    kappa/tau inference. Neither variant is physiological ground truth.
    Explicit observation weights multiply the standardized EEG/HbO/HbR SSE;
    saved observations and integration/quality scales retain the original gauge.
    """
    from src.inference.shared_driver_reconstruction import (
        fit_nonlinear_shared_parameters, nonlinear_driver_forward,
    )
    from src.inference.t3a_balloon_robust_ssm import (
        BalloonParameters, BalloonFixedParameters, BalloonFreeParameters,
    )
    _check_prepared(arrays)
    if (tak_cfg.get('schema') != 'shared_driver_tak_10hz_v2' or tak_cfg['tensor']['dt_s'] != DT
            or coordinate.get('eeg_proxy_input') != EEG_PROXY_INPUT_SCHEMA
            or arm_name not in ('tak_fixed', 'tak_fit')):
        raise ValueError('current Tak 10 Hz target contract and matching train coordinate required')
    arm = next(a for a in tak_cfg['arms'] if a['name'] == arm_name)
    if arm['hemodynamics'] != 'tak' or arm['initial'] != 'volume_tied' or arm['tau_v'] != 2:
        raise ValueError('Tak with four vascular initial coordinates and fixed tau_v=2 required')
    p, solver, fitting = tak_cfg['parameters'], tak_cfg['solver'], tak_cfg['fitting']
    parameters = BalloonParameters(
        fixed=BalloonFixedParameters(**{k: p[k] for k in
            ('gamma', 'alpha', 'E0', 'P0', 'Q0', 'neurovascular_gain')}),
        free=BalloonFreeParameters(tau=p['tau'], kappa=p['kappa']))
    u = np.column_stack((arrays['features'] @ np.asarray(coordinate['pc']) * coordinate['eeg_factor'],
        arrays['hb'] * coordinate['hb_factor']))
    ops = disjoint_native_bin_operator(arrays['native_time'], steps=STEPS, dt=DT, reference_steps=50)
    operator = np.zeros((3 * STEPS, 3 * STEPS))
    for k, op in enumerate((ops['eeg'], ops['model'], ops['model'])):
        operator[k::3, k::3] = op
    mask = np.ones_like(u, bool); mask[-1] = False
    if not eeg_visible:
        mask[:, 0] = False
    weights = np.ones(3) if observation_weights is None else np.asarray(observation_weights, float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('three finite positive EEG/HbO/HbR objective weights required')
    original_sd = np.asarray(coordinate['sd'])
    objective_sd = original_sd if observation_weights is None else original_sd / np.sqrt(weights)
    names = arm['fit']
    values = [fitting['reference'][name] for name in names]
    starts = [dict(parameter_values=np.asarray(values))]
    if names:
        starts = [dict(parameter_values=np.asarray(v)) for v in fitting['starts_kappa_tau']]
    if warm_start is not None:
        if (np.shape(warm_start.get('driver')) != (STEPS,)
                or np.shape(warm_start.get('initial_state')) not in ((4,), (5,))):
            raise ValueError('one whole-trajectory driver and physical initial state required')
        for start in starts:
            start.update(driver=np.asarray(warm_start['driver'])[None],
                initial_state=np.asarray(warm_start['initial_state'])[None])
    kwargs = dict(parameter_names=names, parameter_bounds=[fitting['bounds'][n] for n in names],
        sd=objective_sd, mean_operator=operator, visible=mask[None], starts=starts,
        penalty=solver['penalty'], initial_penalty=solver['initial_penalty'],
        flow_prior_weight=solver['flow_prior_weight'], flow_prior_log_sd=solver['flow_prior_log_sd'],
        substeps=solver['substeps'], max_iterations=solver['max_iterations'],
        max_evaluations_per_trial=solver['max_evaluations_per_trial'],
        gradient_tolerance=solver['gradient_tolerance'], numerical_backend='numba',
        hemodynamics='tak', viscoelastic_s=2., initial_state_mode='volume_tied',
        initial_coordinate_weights=solver['initial_weights_tied'],
        data_weight=DT / solver['observation_reference_dt_s'])
    if names:
        kwargs.update(parameter_prior_log_mean=np.log(values),
            parameter_prior_precision=1 / np.square(arm['log_sd']))
    fit = fit_nonlinear_shared_parameters(u[None], parameters, DT, **kwargs)
    status = {k: fit.get(k) for k in ('status', 'converged', 'objective', 'evaluations',
        'projected_scaled_gradient_inf_norm', 'objective_components', 'parameter_names',
        'parameter_values', 'kappa', 'tau', 'gamma', 'viscoelastic_s', 'starts', 'selected_start')}
    status.update(teacher_arm=arm_name, observation_weights=weights,
        interpretation='privileged_conditional_model_output_not_physiological_truth')
    if 'prediction' not in fit:
        status['accepted'] = False
        return None, status
    physical = fit['prediction'][0]
    refined_parameters = BalloonParameters(fixed=parameters.fixed,
        free=BalloonFreeParameters(tau=fit['tau'], kappa=fit['kappa']))
    fine = nonlinear_driver_forward(fit['driver'][0], fit['initial_state'][0], refined_parameters, DT,
        substeps=solver['validation_substeps'], derivative=False, numerical_backend='numba',
        hemodynamics='tak', viscoelastic_s=2.)
    refined = (operator @ fine['canonical_prediction'].ravel()).reshape(u.shape)
    integration_error = float(np.max(np.abs((refined - physical)[:-1] / coordinate['sd'])))
    status.update(integration_max_training_sd=integration_error,
        accepted=bool(fit['converged'] and integration_error <= solver['maximum_integration_difference_training_sd']))
    payload = dict(u=u, physical=physical, remaining=u - physical, visible=mask,
        states=fit['states'][0], driver=fit['driver'][0], initial_state=fit['initial_state'][0],
        time_s=np.arange(STEPS) * DT,
        identity=np.asarray(arrays['identity']), teacher_arm=np.array(arm_name),
        objective=np.array(fit['objective']),
        eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA), fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA))
    if not np.allclose(payload['u'], payload['physical'] + payload['remaining'], rtol=0, atol=1e-12):
        raise AssertionError('teacher observation = physical + full residual identity failed')
    return payload, status


def produce_alternative_teacher(arrays, coordinate, tak_cfg, reference, settings):
    """Search the fixed small set of legal whole-trajectory Tak restarts.

    Offsets apply to the four initial coordinates [s, log(f), log(v), log(q)];
    p is derived from v. The first coordinate is additive in the declared s
    units. Driver and all four initial coordinates are then jointly refitted.
    A witness is one distinct converged complete solution, never timewise mixing.
    """
    if str(np.asarray(reference.get('teacher_arm', ''))) != 'tak_fixed':
        raise ValueError('alternative target requires the same fixed-reference Tak teacher')
    initial = np.asarray(reference['initial_state'], float)
    if initial.shape != (5,) or initial[2] != initial[3] or np.any(initial[1:] <= 0):
        raise ValueError('legal four-coordinate Tak initial state required')
    ref_objective = float(np.asarray(reference['objective']))
    maximum = float(settings['maximum_objective_fraction'])
    if maximum < 0 or not np.isfinite(ref_objective):
        raise ValueError('finite reference objective and nonnegative feasibility tolerance required')
    witnesses, eligible = [], []
    for offset in settings['offsets']:
        shifted = initial.copy()
        shifted[0] += float(offset)
        shifted[1:] *= np.exp(float(offset))
        shifted[3] = shifted[2]
        record = dict(offset=float(offset), transformed_coordinates=['s', 'log_f', 'log_v', 'log_q'])
        try:
            candidate, status = produce_teacher(arrays, coordinate, tak_cfg,
                warm_start=dict(driver=np.asarray(reference['driver']), initial_state=shifted))
            record.update(status)
            if candidate is not None:
                fraction = (float(candidate['objective']) - ref_objective) / max(abs(ref_objective), 1e-12)
                difference = float(np.sqrt(np.mean(np.square(
                    (candidate['physical'] - reference['physical']) / np.asarray(coordinate['sd'])))))
                distinct = difference > 1e-5
                record.update(objective_fraction=fraction, physical_RMS_difference_training_sd=difference,
                    distinct=distinct, eligible=bool(status['accepted'] and fraction <= maximum and distinct))
                if record['eligible']: eligible.append((float(candidate['objective']), candidate, record))
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            record.update(accepted=False, eligible=False, status='failed_restart', reason=str(exc))
        witnesses.append(record)
    status = dict(status='unavailable', accepted=False, witnesses=witnesses,
        interpretation='finite_same_model_whole_trajectory_restart_witness_not_posterior_or_truth')
    if not eligible: return None, status
    _, chosen, chosen_status = min(eligible, key=lambda item: item[0])
    status.update(status='completed', accepted=True, selected_offset=chosen_status['offset'],
        objective_fraction=chosen_status['objective_fraction'],
        physical_RMS_difference_training_sd=chosen_status['physical_RMS_difference_training_sd'])
    return chosen, status


def synthetic_teacher_check(tak_cfg, seed=20261010, *, observation_weights=None):
    """One actual 90 s fixed-Tak fit, without measured files or model mocks."""
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    from src.inference.t3a_balloon_robust_ssm import BalloonParameters
    started = time.monotonic()
    t = np.arange(STEPS) * DT
    rng = np.random.default_rng(seed)
    driver = .015 * np.sin(2 * np.pi * t / 17) + .008 * np.cos(2 * np.pi * t / 7)
    forward = nonlinear_driver_forward(driver, np.r_[0., np.ones(4)], BalloonParameters(), DT,
        substeps=8, derivative=False, numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
    u = forward['canonical_prediction'].copy()
    u -= u[:50].mean(0)
    arrays = dict(eeg=rng.normal(size=(4000, 6)).astype('float32'),
        features=np.pad(u[:, 0, None], ((0, 0), (0, 29))), hb=u[:, 1:],
        native_time=t, time_s=t, eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA),
        eeg_proxy_rate_hz=np.array(10.), fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA),
        identity=np.array('synthetic_fixed_Tak_90s'))
    co = dict(pc=np.r_[1., np.zeros(29)], eeg_factor=1., hb_factor=1.,
        sd=np.array([.025, .025, .025]), eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA)
    targets, status = produce_teacher(arrays, co, tak_cfg, observation_weights=observation_weights)
    if targets is None or not status['accepted']:
        raise AssertionError('actual synthetic fixed-Tak teacher did not pass numerical acceptance')
    closure = float(np.max(np.abs(targets['u'] - targets['physical'] - targets['remaining'])))
    if closure > 1e-12: raise AssertionError('synthetic teacher decomposition did not close')
    return dict(status='completed', measured_arrays_read=0, steps=STEPS, dt_s=DT,
        teacher_status=status, decomposition_max_error=closure,
        physical_NRMSE_training_sd=np.sqrt(np.mean(np.square(
            (targets['physical'] - targets['u']) / co['sd']), axis=0)),
        elapsed_seconds=time.monotonic() - started,
        implementation='actual_numba_Tak_four_and_eight_substeps_with_fixed_reference_parameters')


def pack_training_cache(cfg, rows, coordinates, read_prepared, read_teacher, output):
    """Write mmap-friendly patch arrays on complete matched three-region trials.

    Readers receive one metadata row. A missing/failed teacher returns None or
    raises FileNotFoundError. The runner must finish all teacher task records
    before calling this function; no success-subset selection occurs here.
    """
    bytrial = defaultdict(list)
    for row in rows: bytrial[row['trial_id']].append(row)
    retained, failures = [], []
    for trial_id, group in sorted(bytrial.items()):
        success = True
        for row in group:
            try: teacher = read_teacher(row)
            except FileNotFoundError: teacher = None
            if teacher is None:
                success = False
        if len(group) != 3 or not success:
            failures.append(trial_id)
        else:
            retained.extend(group)
    if not retained: raise ValueError('no complete matched Tak-teacher trial population')
    # Observation SD is fit on retained training subjects, shared by both
    # component targets. No per-patch normalization or target centering.
    moments = {key: [0, np.zeros(3), np.zeros(3)] for key in coordinates}
    for row in retained:
        if row['split'] != 'train': continue
        u = np.asarray(read_teacher(row)['u'][STUDENT], float)
        n, total, squared = moments[row['coordinate']]
        moments[row['coordinate']] = [n + len(u), total + u.sum(0), squared + np.square(u).sum(0)]
    scales = {}
    for key, (n, total, squared) in moments.items():
        if not n: raise ValueError('retained training support required for every coordinate')
        scales[key] = np.maximum(np.sqrt(np.maximum(squared / n - (total / n) ** 2, 0)), 1e-8)
    shapes = dict(eeg_input=(len(retained), 10, 2400), hb_input=(len(retained), 10, 40))
    for modality, dimension in (('eeg', 20), ('hb', 40)):
        for component in ('u', 's', 'e'): shapes[modality + '_' + component] = (len(retained), 10, dimension)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    maps = {k: np.lib.format.open_memmap(out / (k + '.tmp.npy'), mode='w+', dtype='float32', shape=v)
        for k, v in shapes.items()}
    for i, row in enumerate(retained):
        a, teacher = read_prepared(row), read_teacher(row)
        _check_prepared(a)
        co, scale = coordinates[row['coordinate']], scales[row['coordinate']]
        if str(np.asarray(a['identity'])) != row['id'] or str(np.asarray(teacher['identity'])) != row['id']:
            raise ValueError('canonical prepared/teacher identity mismatch')
        u, s, e = (np.asarray(teacher[k][STUDENT], float) for k in ('u', 'physical', 'remaining'))
        if (u.shape != (200, 3) or not all(np.isfinite(v).all() for v in (u, s, e))
                or not np.allclose(u, s + e, rtol=0, atol=1e-10)):
            raise ValueError('finite same-coordinate teacher decomposition required')
        maps['eeg_input'][i] = ((a['eeg'] - co['eeg_mean']) / co['eeg_scale']).reshape(10, 2400)
        maps['hb_input'][i] = (a['hb'][STUDENT] * co['hb_factor'] / scale[1:]).reshape(10, 40)
        for key, value in (('u', u), ('s', s), ('e', e)):
            scaled = value / scale
            maps['eeg_' + key][i] = scaled[:, 0].reshape(10, 20)
            maps['hb_' + key][i] = scaled[:, 1:].reshape(10, 40)
    for key, value in maps.items():
        value.flush()
        (out / (key + '.tmp.npy')).replace(out / (key + '.npy'))
    manifest = dict(schema='dual_component_training_cache_v1', rows=retained,
        identity=identity_digest(retained), shapes=shapes,
        arrays={key: key + '.npy' for key in shapes}, dtype='float32',
        planned_windows=len(rows), retained_windows=len(retained),
        planned_trials=len(bytrial), retained_trials=len(retained) // 3, failed_trials=failures,
        observation_training_sd=scales, scale_fit='retained_training_subjects_only_no_target_centering',
        target_scaling='divide_by_frozen_observation_training_sd_no_centering',
        target_units='training_observation_SD',
        target_coordinates=['EEG_training_PCA_log_bandpower', 'HbO', 'HbR'],
        decomposition='observation = physical_Tak + full_fixed_teacher_remaining',
        dt_s=DT, student_seconds=20, patch_seconds=2, patches=10,
        teacher_seconds=90, teacher_event_support_s=[-30, 60],
        eeg_input='native_200Hz_voltage_local_2s_no_teacher_features',
        hb_input='cached_v5_none_local_2s_with_fixed_reference_event_minus30_to_minus25',
        hb_preprocessing='full_record_offline_0.01_0.2Hz_before_masking',
        masking_claim='processed_coordinate_completion', labels_as_training_inputs=False,
        parent_run=cfg['parent_run'], frozen_subject_split=True,
        independent_subjects={ds: {split: len({r['subject'] for r in retained
            if r['dataset'] == ds and r['split'] == split})
            for split in ('train', 'validation', 'test')} for ds in DATASETS},
        failures='all branches share complete three-region trial support; all planned failures retained')
    path = out / 'manifest.json'; tmp = path.with_suffix('.tmp')
    tmp.write_text(_json(manifest) + '\n'); tmp.replace(path)
    return manifest
