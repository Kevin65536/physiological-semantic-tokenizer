"""Fit paired frequency responses to frozen, paid EEG reconstructions on training people."""
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import argparse
import fcntl
import json
import os
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.band_exchange_words import load_fold_books
from src.tokenizers.cross_branch_innovations import decode_words
from evaluate_joint_tokenizer import write


def coefficients(gram, cross, fraction):
    """Complex Hermitian regression with a trace-scaled identity prior per output/bin."""
    identity = np.eye(2)
    full = np.zeros_like(cross)
    positive = np.zeros_like(cross)
    for b in range(2):
        covariance = gram[b]
        ridge = np.maximum(fraction * np.trace(covariance, axis1=-2, axis2=-1).real / 2, 1e-20)
        rhs = cross[b] + ridge[:, None] * identity[b]
        full[b] = np.linalg.solve(covariance + ridge[:, None, None] * identity, rhs[..., None])[..., 0]
        positive[b, :, b] = np.maximum(0, rhs[:, b].real / (covariance[:, b, b].real + ridge))
    return positive, full


def solve_dataset(dataset, cfg, out):
    run = ROOT / cfg['run']
    source = run / cfg['source']
    folder = out / dataset
    folder.mkdir(exist_ok=True)
    population = json.loads((run / cfg['population']).read_text())
    g0 = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest['rows']
    if (manifest['cfg_identity'], manifest['scales_identity']) != (population['cfg_identity'], population['scales_identity']):
        raise ValueError('fixed input/scale identity mismatch')
    cases = [c for c in population['cases'] if c['dataset'] == dataset]
    for f in range(4):
        dest = folder/f'fold_{f}_metrics.npz'
        if dest.exists():
            continue
        old = {b: load_fold_books(run/cfg['pilot_books'], f, b) for b in 'SO'}
        new = {b: load_fold_books(source/'books', f, b) for b in 'SO'}
        with np.load(source/dataset/f'fold_{f}_state.npz') as disk:
            state = dict(disk)
        with np.load(source/dataset/f'fold_{f}_ids.npz') as disk:
            ex, fold = disk['examples'], disk['fold']
            paid = {b: disk['innovation_'+b] for b in 'SO'}
        identities = [rows[i] for i, _, _ in ex]
        if any(row['split'] != 'train' or row['dataset'] != dataset for row in identities):
            raise ValueError('train-only source population before any signal read')
        groups = np.array([row['subject'] for row in identities])
        if set(groups[fold==f]) & set(groups[fold!=f]):
            raise ValueError('fit and held people overlap')
        weights = np.zeros(len(ex));weights[fold!=f] = training_weights(ex[fold!=f], rows)
        gram = np.zeros((2,201,2,2), complex);cross = np.zeros((2,201,2), complex)
        measures = defaultdict(list); cache = OrderedDict()
        for fit in (True, False):
            if not fit:
                gain, joint = coefficients(gram, cross, cfg['fit']['ridge_fraction_of_trace'])
                head = folder/f'fold_{f}_head.npz'
                np.savez_compressed(head, gain=gain, joint=joint, fit_people=np.unique(groups[fold!=f]), held_people=np.unique(groups[fold==f]))
                with np.load(head) as saved:
                    for current, previous in [(saved['gain'],gain),(saved['joint'],joint)]:
                        np.testing.assert_array_equal(current, previous)
                    gain, joint = saved['gain'], saved['joint']
            offset = 0
            for case in cases:
                take = np.arange(offset,offset+case['count']);offset += len(take)
                if fit != (f != case['fold']):
                    continue
                assert np.all(fold[take] == case['fold'])
                native = np.asarray(case['native']); order = take[native.argsort()]
                sent = {b: paid[b][order] for b in 'SO'}
                waves = decode_words(sent, old, new, state, state, 'innovation', False)
                x = np.stack([waves[b][native] for b in 'SO'], axis=-1)
                spectrum = np.fft.rfft(x, axis=2)
                batch = base_tokenizer_batch(rows, ex[take], scales, array_cache=cache, cache_limit=8)
                mask = batch['mask'].astype(bool)
                target = np.stack([batch[b] for b in 'so'], axis=-1)
                target_fft = np.fft.rfft(np.where(mask[...,None],target,0.),axis=2)
                if fit:
                    whole = mask.all(axis=2)
                    for b,key in enumerate(['vs','vo']):
                        w = weights[take,None]*whole/np.maximum(mask.sum((1,2)),1)[:,None]/batch[key][:,0,0,None]
                        gram[b] += np.einsum('apfi,ap,apfj->fij',spectrum.conj(),w,spectrum,optimize=True)
                        cross[b] += np.einsum('apfi,ap,apf->fi',spectrum.conj(),w,target_fft[...,b],optimize=True)
                    continue
                measures['examples'].append(ex[take])
                factor = np.array([scales['positions']['eeg'][dataset][rows[i]['positions'][p]]['scale'] for i,_,p in ex[take]])
                def wavefields():
                    yield 'frozen',x
                    for name,coef in [('positive_gain',gain),('complex_pair',joint)]:
                        pred = np.einsum('apfi,bfi->apfb',spectrum,coef,optimize=True)
                        yield name,np.fft.irfft(pred,n=400,axis=2)
                    for name,mag,phase in [('oracle_magnitude',target_fft,spectrum),('oracle_phase',spectrum,target_fft)]:
                        yield name,np.fft.irfft(abs(mag)*np.exp(1j*np.angle(phase)),n=400,axis=2)
                for label,values in wavefields():
                    pred = dict(S=values[...,0],O=values[...,1],sum=values.sum(-1))
                    score = joint_reconstruction_metrics(pred,batch,1,factor)
                    for k,v in score['metrics'].items():
                        measures[label+'__'+k].append(v)
        tmp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,**{k:np.concatenate(v) for k,v in measures.items()});os.replace(tmp,dest)
        print(dataset,'fold',f,'complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if cfg['data']!={'fit':'original_train_fold_other_people','score':'original_train_fold_held_people','validation':'forbidden','confirmation':'forbidden','test':'forbidden'}:raise ValueError('training-only contract required')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.synthetic:
        g=np.broadcast_to(np.eye(2),(2,201,2,2)).copy(); rhs=np.zeros((2,201,2),complex);rhs[0,:,0]=rhs[1,:,1]=1
        for estimator in coefficients(g,rhs,.05):np.testing.assert_allclose(estimator,rhs)
        write(out/'synthetic.json',{'identity_filter_recovered':True});return
    if not (out/'synthetic.json').is_file():raise RuntimeError('synthetic identity required')
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=ROOT/cfg['run'];pop=json.loads((run/cfg['population']).read_text());datasets=sorted(pop['native_positions'])
    write(out/'resolved_config.json',cfg)
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(partial(solve_dataset,cfg=cfg,out=out),datasets))
    arrays=defaultdict(list)
    for ds in datasets:
        for fold in range(4):
            with np.load(out/ds/f'fold_{fold}_metrics.npz') as z:
                for k in z.files:arrays[k].append(z[k])
    ex=np.concatenate(arrays.pop('examples'));assert len(ex)==17040
    rows=base_tokenizer_training_data(yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()),ROOT)[0]['rows']
    stats={arm:base_training_aggregate({k.split('__',1)[1]:np.concatenate(v) for k,v in arrays.items() if k.startswith(arm+'__')},ex,rows) for arm in cfg['controls']}
    expected=json.loads((run/cfg['source']/'summary.json').read_text())['arms']['innovation_mean']['summary']['metrics']
    for k in expected:np.testing.assert_allclose(stats['frozen']['metrics'][k]['macro'],expected[k]['macro'],rtol=1e-8,atol=1e-12,equal_nan=True)
    write(out/'summary.json',dict(arms=stats,source_reproduced=True,arrays=400,people=60,positions=len(ex),validation_reads=0,test_reads=0,oracle='paired target FFT computed with original invalid points set to zero; diagnostic only'))


if __name__=='__main__':
    main()
