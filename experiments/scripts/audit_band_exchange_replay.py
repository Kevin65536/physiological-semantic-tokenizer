"""Independent 13-bit transport, frozen decoding and train-only physical provenance."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tokenizers.band_exchange_words import decode_native, encode_native
from src.tokenizers.eeg_transient_gain import load_model

STATE = {}
PRIORITY = ('exchange_balanced_energy', 'exchange_balanced_allocated')


def write(path, value):
    def scalar(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=scalar) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def pack_ids(values):
    a = np.asarray(values)
    if a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a > 5115):
        raise ValueError('paid alphabet is integer 0..5115')
    a = a.astype(np.uint16, copy=False).reshape(-1)
    bits = ((a[:, None] >> np.arange(13, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bits.reshape(-1), bitorder='little').tobytes()


def unpack_ids(payload, shape):
    if any(not isinstance(n, (int, np.integer)) or n < 0 for n in shape):
        raise ValueError('nonnegative fixed dimensions required')
    count = int(np.prod(shape, dtype=np.int64))
    if len(payload) != (13 * count + 7) // 8:
        raise ValueError('incorrect 13-bit payload length')
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='little')
    if np.any(bits[13 * count:]): raise ValueError('nonzero trailing padding')
    values = np.sum(bits[:13 * count].reshape(count, 13).astype(np.uint16)
                    << np.arange(13, dtype=np.uint16), axis=1, dtype=np.uint16)
    if np.any(values > 5115): raise ValueError('unused 13-bit word received')
    return values.reshape(shape)


def books(fold):
    result = {}
    for b in 'SO':
        result[b] = []
        for role in range(5):
            with np.load(STATE['parent'] / 'books' / f'fold_{fold}_{b}_{role}.npz', allow_pickle=False) as z:
                meta = json.loads(str(z['metadata']))
                if (meta['fold'], meta['branch'], meta['role']) != (fold, b, role):
                    raise ValueError('dictionary identity changed')
                result[b].append({k: z[k].copy() for k in z if k != 'metadata'})
    return result


def frozen(dataset, fold):
    run = STATE['run']; people = STATE['contract']['datasets'][dataset]['folds'][str(fold)]
    for name in [str(STATE['parent']/'books'/f'fold_{fold}_{b}_{role}.npz') for b in 'SO' for role in range(5)] + [
            str(run/'component_source_transfer/transient_gain_v1'/f'{dataset}_fold{fold}_transient.npz'),
            str(run/'component_source_transfer/adaptive_band_v1'/dataset/f'fold_{fold}_model.npz')]:
        if sha(name) != STATE['contract']['frozen_files'][name]: raise ValueError('frozen decoder asset changed')
    source = load_model(run / 'component_source_transfer/transient_gain_v1' / f'{dataset}_fold{fold}_transient.npz')
    if set(source.metadata['fit_subjects']) != set(people['fit_subjects']):
        raise ValueError('source fit population changed')
    if source.metadata['held_fold'] != fold or source.metadata['positions'][dataset] != STATE['contract']['datasets'][dataset]['names']:
        raise ValueError('source fold/layout identity changed')
    with np.load(run / 'component_source_transfer/adaptive_band_v1' / dataset / f'fold_{fold}_model.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['names'], STATE['contract']['datasets'][dataset]['names'])
        if set(z['fit_people']) != set(people['fit_subjects']): raise ValueError('covariance fit population changed')
        cov = {b: z[b].copy() for b in 'SO'}
    return books(fold), cov, source


def budget():
    unit = os.environ.get('BAND_REPLAY_UNIT')
    if unit:
        import subprocess
        used = int(subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'CPUUsageNSec', '--value'], text=True)) / 1e9
        used += float(os.environ.get('BAND_REPLAY_PRIOR_CPU', '0'))
        if used > 880: raise RuntimeError('15 CPU-minute independent audit budget exhausted')


def pack_dataset(dataset):
    here = STATE['out'] / dataset; here.mkdir(exist_ok=True)
    cases = STATE['contract']['datasets'][dataset]['cases']; total = []
    for fold in range(4):
        budget(); all_indices = []; offset = 0
        for c in cases:
            if c['fold'] == fold: all_indices.extend(range(offset, offset+c['count']))
            offset += c['count']
        all_indices = np.array(all_indices)
        input_path = STATE['parent'] / dataset / f'fold_{fold}_ids.npz'
        with np.load(input_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['examples'], STATE['contract']['datasets'][dataset]['examples'])
            for layout in ('fixed', 'exchange'):
                for mode in ('wave', 'balanced'):
                    for b in 'SO':
                        name = f'fold_{fold}_{layout}_{mode}_{b}'
                        values = z[f'{layout}_{mode}_{b}'][all_indices]
                        payload = pack_ids(values)
                        path = here / (name + '.13bit'); path.write_bytes(payload)
                        np.testing.assert_array_equal(unpack_ids(path.read_bytes(), values.shape), values)
                        total.append(dict(file=path.name, shape=list(values.shape), count=values.size,
                                          paid_bits=13*values.size, bytes=len(payload), sha256=sha(path),
                                          max_ID=int(values.max()), source_ids_sha256=sha(input_path)))
    write(here / 'transport.json', total)
    return dataset


def decode_dataset(dataset):
    """No raw cache, target, X, prepared latent or encoder feature is opened here."""
    here = STATE['out'] / dataset; desc = STATE['contract']['datasets'][dataset]
    transport = {v['file']: v for v in json.loads((here / 'transport.json').read_text())}
    arms = STATE['arms']; checks = []
    for fold in range(4):
        budget(); book, cov, source = frozen(dataset, fold)
        streams = {}
        for layout in ('fixed', 'exchange'):
            for mode in ('wave', 'balanced'):
                for b in 'SO':
                    name = f'fold_{fold}_{layout}_{mode}_{b}.13bit'; record = transport[name]
                    if sha(here/name) != record['sha256']: raise ValueError('payload changed')
                    streams[f'{layout}_{mode}_{b}'] = unpack_ids((here/name).read_bytes(), record['shape'])
        predictions = {a: {b: [] for b in ('sum', 'S', 'O')} for a in arms}
        offset = 0
        for c in desc['cases']:
            if c['fold'] != fold: continue
            take = np.arange(offset, offset+c['count']); offset += c['count']
            native = np.asarray(c['native']); np.testing.assert_array_equal(np.sort(native), np.arange(len(desc['names'])))
            native_order = take[native.argsort()]
            for arm in arms:
                layout, mode, method = arm.split('_'); tag = f'{layout}_{mode}'
                sent = {b: streams[tag+'_'+b][native_order] for b in 'SO'}
                full = decode_native(sent, dataset, book, cov, source, mode, method)
                for b in ('sum', 'S', 'O'): predictions[arm][b].append(full[b][native])
        for arm in arms:
            # These arrays are independent decoder outputs, never inputs to decode_native.
            np.savez(here/f'fold_{fold}_{arm}_prediction.npz',
                                **{b: np.concatenate(v) for b, v in predictions[arm].items()})
        checks.append(dict(fold=fold, held_positions=offset, arms=arms,
                           decode_inputs=['13bit payload', 'frozen atom books', 'frozen covariance',
                                          'fixed dataset native order', 'frozen source_f for allocated decoded total only'],
                           raw_X_reads=0, target_reads=0, prepared_array_reads=0, encoder_calls=0))
    write(here/'decode.json', checks)
    return dataset


def training_data():
    from src.data.dual_component_tokenizer import base_tokenizer_training_data
    cfg = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    if (manifest['cfg_identity'], manifest['scales_identity']) != (STATE['contract']['input_identity'], STATE['contract']['scales_identity']):
        raise ValueError('G0 identity changed')
    rows = manifest['rows']
    for d in STATE['contract']['datasets'].values():
        for c in d['cases']:
            row = rows[c['crop']['row']]
            if (row['split'], row['modality'], row['subject']) != ('train', 'eeg', c['subject']):
                raise ValueError('train boundary before any physical array reads')
    return rows, scales


def score_dataset(dataset):
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
    rows, scales = training_data(); here = STATE['out']/dataset
    cache = OrderedDict(); checks = []
    for fold in range(4):
        budget(); parent_path = STATE['parent']/dataset/f'fold_{fold}_metrics.npz'
        with np.load(parent_path, allow_pickle=False) as parent:
            ex = parent['examples']
            retained = here/f'fold_{fold}_rescored.npz'
            if retained.exists():
                # A completed independently scored fold is immutable evidence. Reuse only
                # after every requested metric is compared anew with the producer.
                with np.load(retained,allow_pickle=False) as previous:
                    np.testing.assert_array_equal(previous['examples'],ex)
                    for arm in STATE['arms']:
                        names=[k for k in parent if k.startswith(arm+'__')]
                        if not names:raise ValueError('no requested parent metrics')
                        for name in names:np.testing.assert_array_equal(previous[name],parent[name])
                        checks.append(dict(fold=fold,arm=arm,positions=len(ex),metrics=len(names),
                            retained_independent_fold_rechecked=True,every_metric_array_equal=True,max_abs_difference=0))
                continue
            batch = base_tokenizer_batch(rows, ex, scales, array_cache=cache, cache_limit=8)
            factors = np.array([scales['positions']['eeg'][rows[i]['dataset']][rows[i]['positions'][p]]['scale'] for i,t,p in ex])
            result = {}
            for arm in STATE['arms']:
                with np.load(here/f'fold_{fold}_{arm}_prediction.npz', allow_pickle=False) as z:
                    predicted = {b: z[b].copy() for b in ('sum','S','O')}
                scored = joint_reconstruction_metrics(predicted, batch, 1, factors)
                for name, values in scored['metrics'].items():
                    expected = parent[arm+'__'+name]
                    np.testing.assert_array_equal(values, expected, err_msg=f'{dataset}/{fold}/{arm}/{name}')
                    result[arm+'__'+name] = values
                checks.append(dict(fold=fold,arm=arm,positions=len(ex),metrics=len(scored['metrics']),
                                   every_metric_array_equal=True,max_abs_difference=0))
            np.savez_compressed(here/f'fold_{fold}_rescored.npz', examples=ex, **result)
    write(here/'metric_comparison.json', checks)
    return dataset


def reencode_dataset(dataset):
    from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg
    rows, scales = training_data(); here=STATE['out']/dataset; checks=[]
    desc=STATE['contract']['datasets'][dataset]; names=desc['names']
    for fold in range(4):
        budget(); book,cov,source=frozen(dataset,fold)
        c=next(c for c in desc['cases'] if c['fold']==fold)
        fields,seen,_=base_tokenizer_joint_eeg(rows,dict(c['crop'],positions=[]),scales,names,allowed_splits=('train',))
        x=fields['x'].T.reshape(len(names),15,400).astype(np.float32)
        weights={}
        table={n.upper():v for n,v in scales['positions']['eeg'][dataset].items()}
        for b in 'SO':
            weights[b]=np.array([1/float(np.ravel(table[n.upper()]['target_variance'][b.lower()])[0]) for n in names])
        offset=0
        for prior in desc['cases']:
            if prior is c:break
            offset+=prior['count']
        native=np.asarray(c['native']); take=np.arange(offset,offset+c['count'])
        original=STATE['parent']/dataset/f'fold_{fold}_ids.npz'
        with np.load(original,allow_pickle=False) as z:
            for layout in ('fixed','exchange'):
                for mode in ('wave','balanced'):
                    sent=encode_native(x,seen,dataset,source,book,cov,weights,layout,mode)
                    for b in 'SO':
                        expected=z[f'{layout}_{mode}_{b}'][take]
                        np.testing.assert_array_equal(sent[b][native],expected)
                    checks.append(dict(fold=fold,layout=layout,mode=mode,subject=c['subject'],row=c['crop']['row'],
                                       start_s=c['crop']['start_s'],positions=names,observed=int(seen.sum()),
                                       input_X_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                       all_S_O_IDs_exact=True,IDs=2*len(names)*15))
    write(here/'physical_reencode.json',checks)
    return dataset


def setup(args):
    contract_path=args.output/'audit_contract.json'
    if not contract_path.exists():
        if args.stage not in ('pack','all'):raise ValueError('pack must freeze contract first')
        config=yaml.safe_load(args.config.read_text()); run=ROOT/config['source']['joint_run']
        pop_path=run/config['source']['same_source_bank']/'population.json';pop=json.loads(pop_path.read_text())
        datasets={}
        for d,names in sorted(pop['native_positions'].items()):
            cases=[c for c in pop['cases'] if c['dataset']==d]
            examples=[(c['crop']['row'],c['crop']['start_s'],p) for c in cases for p in c['crop']['positions']]
            datasets[d]=dict(names=names,cases=cases,examples=examples,folds={str(f):dict(
                fit_subjects=sorted({c['subject'] for c in cases if c['fold']!=f}),
                held_subjects=sorted({c['subject'] for c in cases if c['fold']==f})) for f in range(4)})
        if len(pop['cases'])!=400 or sum(len(v['examples']) for v in datasets.values())!=17040:raise ValueError('original population changed')
        frozen_files=list((args.parent/'books').glob('fold_*_*.npz'))
        for d in datasets:
            for f in range(4):
                frozen_files += [run/'component_source_transfer/transient_gain_v1'/f'{d}_fold{f}_transient.npz',
                                 run/'component_source_transfer/adaptive_band_v1'/d/f'fold_{f}_model.npz']
        if len(frozen_files)!=72:raise ValueError('40 dictionaries and 32 source/covariance states required')
        write(contract_path,dict(schema='band_exchange_independent_replay_v1',run=str(run),parent=str(args.parent.resolve()),
            input_identity=pop['cfg_identity'],scales_identity=pop['scales_identity'],population_sha256=sha(pop_path),
            frozen_files={str(p.resolve()):sha(p) for p in frozen_files},
            source_config_sha256=sha(args.config),datasets=datasets,bitorder='little',bits_per_ID=13,
            alphabet_size=5116,priority_arms=PRIORITY,CPU_minutes=15,validation_reads=0,confirmation_reads=0,test_reads=0))
    contract=json.loads(contract_path.read_text())
    arms=list(PRIORITY)
    if args.all_arms:
        arms += [f'{l}_{m}_{s}' for l in ('fixed','exchange') for m in ('wave','balanced')
                 for s in (('mean','energy','allocated') if m=='balanced' else ('mean','energy'))
                 if f'{l}_{m}_{s}' not in arms]
    STATE.update(contract=contract,out=args.output,parent=Path(contract['parent']),run=Path(contract['run']),arms=arms)
    if args.stage in ('pack','reencode','all'):training_data()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_band_exchange_words_v1.yaml')
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('pack','decode','score','reencode','summarize','all'),default='all')
    parser.add_argument('--all-arms',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);setup(args)
    stages=['pack','decode','score','reencode','summarize'] if args.stage=='all' else [args.stage]
    for stage in stages:
        start=time.monotonic()
        if stage=='summarize':
            from src.analysis.dual_component_tokenizer import base_training_aggregate
            rows,_=training_data();values=defaultdict(list);examples=[]
            for d in STATE['contract']['datasets']:
                for f in range(4):
                    with np.load(args.output/d/f'fold_{f}_rescored.npz',allow_pickle=False) as z:
                        examples.append(z['examples']);
                        for key in z:
                            if key!='examples':values[key].append(z[key])
            ex=np.concatenate(examples);results={}
            for arm in STATE['arms']:
                arrays={k.split('__',1)[1]:np.concatenate(v) for k,v in values.items() if k.startswith(arm+'__')}
                results[arm]=base_training_aggregate(arrays,ex,rows)
            write(args.output/'summary.json',dict(status='complete',all_native_crops=400,all_positions=17040,
                all_subjects=60,arms=results,every_metric_array_equal=True,max_abs_difference=0,
                physical_reencode_crops=16,physical_reencode_all_four_ID_assignments=True,
                logical_IDs_per_arm=17040*15*2,logical_bits_per_arm=17040*15*2*13,
                nominal_bits_per_second_per_branch_native_position=6.5,
                validation_reads=0,confirmation_reads=0,test_reads=0))
        else:
            function={'pack':pack_dataset,'decode':decode_dataset,'score':score_dataset,'reencode':reencode_dataset}[stage]
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:
                list(pool.map(function,STATE['contract']['datasets']))
        write(args.output/f'{stage}_complete.json',dict(stage=stage,wall_seconds=time.monotonic()-start))


if __name__=='__main__':main()
