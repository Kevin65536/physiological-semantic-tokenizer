"""Retrospective on public, frozen November 23 case selections and saved scores."""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.window_exclusions import VERSION

BASE = ROOT / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer'
RUN = BASE / '20261123_joint_baseline_v1'
TRANSFER = RUN / 'component_source_transfer'
CENSUS = BASE / '20261124_observation_qc_census_v1'
DATASETS = ['eeg_fnirs_single_trial', 'refed', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation']
EEG = ['nmse', 'rms_gain', 'band_4_8_db', 'band_4_8_abs_db', 'band_8_13_db',
       'band_8_13_abs_db', 'correlation', 'local_rms_correlation']
HB = ['nmse', 'rms_gain', 'correlation']


def read(path):
    return json.loads(path.read_text())


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def inventory(rows, identity):
    patterns = [
        'waveforms/*/manifest.json', 'waveforms/*/*/manifest.json', 'waveforms/*/*/*/manifest.json',
        'development_audit/spatial/waveforms_v1/manifest.json',
        'development_audit/as_local_gain_v1/spatial/waveforms_v1/manifest.json',
        'hb_native_id_readout/v1/development/waveform_examples_v2.json',
        'hb_native_id_readout/v1/screen/visual_audit.json',
        'hb_native_id_readout/v1/development/paired_worst_degradation.json',
        'anchored_dictionary/v1/full_training_review_v2/hard_waveforms_v2/examples.json',
        '*/*/waveforms/manifest.json', 'component_source_transfer/*/visual_review_v1/figure_manifest.json',
        'component_source_transfer/*/visual_review_v1/manifest.json',
        'component_source_transfer/*/visual_review_v1/selected_examples.json',
        'component_source_transfer/*/review_v1/selected_examples.json',
        'component_source_transfer/*/review_v1/examples.json',
        'component_source_transfer/*/training_examples/waveform_examples.json',
        'distributed_measurement_v1/selected_examples.json',
        'stratified_source_atoms/v1/visual_review.json',
        'frequency_spatial/v1/difficult_figure_manifest_v2.json',
        'frequency_spatial/temporal_rate_v2/figure_manifest.json',
        'spatial_predictive/v1/figure_manifest_v2.json','multimodal_hb/local_geometry_v1/figure_manifest.json',
        'hb_neighbor_readout/v1/held_figure_manifest.json','hb_parent_distillation/v1/figure_manifest.json',
        'hb_parent_leaf/*/training_review_v2/figure_manifest.json',
        'component_source_transfer/adaptive_array_v1/figure_manifest.json',
        'component_source_transfer/forward_gain_v1/figure_manifest.json',
        'development_audit/hb/vq_source/waveform_examples.json',
        'candidate_audit/fullband/eeg/vq1024/step_02048/hard/waveform_examples.json',
        'candidate_audit/vq_source/hb/k4096/hard/waveform_examples.json',
        'spatial/band_dictionary_v2/audit_retry_v1/waveform_examples.json',
    ]
    grouped = {}
    counts = {}
    for path in sorted({p for pat in patterns for p in RUN.glob(pat)}):
        if any(k.startswith(('legacy_v3_', 'inspection_previews')) for k in path.parts):
            continue
        doc = read(path)
        if isinstance(doc, list):
            items = doc
        elif 'triple' in doc:
            items = [dict(doc, selection=['worst_paired_degradation'])]
        else:
            items = next((doc[k] for k in ['examples', 'cases', 'waveforms', 'figures', 'records']
                          if isinstance(doc.get(k), list)), [])
        src = str(path.relative_to(RUN))
        marked = set()
        index_map = {}
        if path.parts[-4] == 'hb_parent_leaf':
            with np.load(path.parent/'saved_integer_ids.npz') as saved:
                index_map = dict(zip(saved['training_panel_index'],saved['examples'],strict=True))
        for item in items:
            if not isinstance(item, dict):
                continue
            v = item['identity'] if isinstance(item.get('identity'),dict) else item
            tags = []
            for k in ('selection', 'audit_selection', 'selector', 'label', 'kind', 'rule', 'rules'):
                val = v.get(k)
                if isinstance(val, str):
                    tags.append(val)
                elif isinstance(val, list):
                    tags.extend(val)
            if src == 'stratified_source_atoms/v1/visual_review.json':
                # The producer records the selection rule in the figure basename.
                filename = Path(v['path']).stem
                for rule in ('original_highest_montage_S_NMSE','new_hard_S_NMSE'):
                    if filename.endswith('_'+rule):
                        tags.append(rule)
            text = ','.join(tags).lower()
            worst = bool(re.search(r'max|worst|hard|difficult|highest|largest', text))
            p90 = 'p90' in text
            p95 = 'p95' in text
            if not (worst or p90 or p95):
                continue
            if 'triple' in v or 'example' in v:
                ex = v.get('triple', v.get('example'))
            elif isinstance(v.get('identity'),list):
                ex = v['identity']
            elif index_map:
                ex = index_map[v['index']]
            elif 'crop' in v:
                crop = v['crop']
                pos = v.get('native_site', v.get('position'))
                ex = [crop['row'], crop['start_s'], rows[crop['row']]['positions'].index(pos)]
            else:
                ex = [v[k] for k in ['row','start_s','position']]
            i,t,j = map(int,ex)
            row = rows[i]
            assert row['dataset'] == v.get('dataset',row['dataset']) and row['subject'] == v.get('subject',row['subject'])
            if 'start_s' in v:
                assert t == v['start_s']
            name = v.get('position_name', v.get('site', v.get('native_site', v.get('native_position'))))
            if name is None and isinstance(v.get('position'), str):
                name = v['position']
            if name is not None:
                assert row['positions'][j] == name, (path, ex, name)
            if 'parent' in v:
                assert row['parent'] == v['parent']
            key, hits = identity(i,t,j)
            marked.add(key)
            info = grouped.setdefault(key, dict(worst=False,p90=False,p95=False,hits=hits,sources=set(),tags=set()))
            info['sources'].add(src)
            info['tags'].update(tags)
            info['worst'] |= worst
            info['p90'] |= p90
            info['p95'] |= p95
        if marked:
            counts[src] = [len(marked), sum(bool(grouped[k]['hits']) for k in marked)]
    out = []
    for (d,split,s,r,m,pos,start),info in sorted(grouped.items()):
        out.append(dict(dataset=d,split=split,subject=s,record=r,modality=m,position=pos,
                        start_200=start,stop_200=start+6000,worst=int(info['worst']),p90=int(info['p90']),p95=int(info['p95']),
                        flags='|'.join(info['hits']),selection='|'.join(sorted(info['tags'])),
                        evidence='|'.join(sorted(info['sources']))))
    return out, counts


def main(output):
    manifest = read(BASE/'20261122_unified_base_g0_v1/cache/manifest.json')
    assert manifest['allowed_splits'] == ['train','validation']
    rows = [{k:r[k] for k in ('dataset','split','subject','record','modality','positions','parent',
                               'native_start_sample','native_rate_hz')} for r in manifest['rows']]
    del manifest
    markers = defaultdict(list)
    with (ROOT/'src/data/assets'/f'{VERSION}.csv').open() as f:
        for r in csv.DictReader(f):
            markers[tuple(r[k] for k in ['dataset','split','subject','record','modality','position'])].append(
                (int(r['start_200']),int(r['stop_200']),r['reason']))
    dev = read(RUN/'development_audit/as_local_gain_v1/spatial/summary.json')
    admitted = {(v['dataset'],v['subject']) for v in dev['metrics']['nmse']['by_subject']}
    assert len(admitted)==16

    @lru_cache(maxsize=None)
    def identity(i,t,j):
        row=rows[i]
        assert row['split']=='train' or (row['split']=='validation' and (row['dataset'],row['subject']) in admitted)
        stem = tuple(row[k] for k in ('dataset','split','subject','record','modality'))+(row['positions'][j],)
        start = round(row['native_start_sample'] * 200 / row['native_rate_hz']) + 200*t
        hits = tuple(sorted({reason for a,b,reason in markers.get(stem,()) if start < b and start+6000 > a}))
        return stem+(start,),hits

    cases, sources = inventory(rows,identity)
    ranked = {tuple(v[k] for k in ['dataset','split','subject','record','modality','position','start_200']) for v in cases}
    totals = []
    def collect(panel, arm, files, prefix, keys, reference=None, words=False):
        indices, pieces = [], defaultdict(list)
        for path in files:
            with np.load(path,allow_pickle=False) as z:
                indices.append(z['examples'])
                for k in keys:
                    pieces[k].append(z[prefix+k])
        ex=np.concatenate(indices)
        assert len(ex)==len(set(map(tuple,ex)))
        flags=np.array([bool(identity(*map(int,v))[1]) for v in ex])
        parts={k:np.concatenate(v) for k,v in pieces.items()}
        no_ranked = np.array([identity(*map(int,v))[0] not in ranked for v in ex])
        for label, mask in [('all',np.ones(len(ex),bool)),('clean',~flags),
                            ('clean_without_ranked_examples',~flags & no_ranked)]:
            metrics={k:v[mask] for k,v in parts.items()}
            if 'nmse' not in metrics:
                metrics['nmse']=next(iter(metrics.values()))
            result=base_training_aggregate(metrics,ex[mask],rows)
            measure=result['metrics']
            for k in keys:
                for group, value in [('macro',measure[k]['macro']),*measure[k]['by_dataset'].items()]:
                    val=np.array(value)
                    if reference is not None and label=='all' and group=='macro':
                        target=reference[k]['macro']
                        np.testing.assert_allclose(val,target,rtol=1e-8,atol=1e-10)
                    if words:
                        val=1-val[1]/val[0]
                    totals.append(dict(panel=panel,arm=arm,support=label,dataset=group,metric=k,
                                       positions=len(ex[mask]) if group=='macro' else sum(rows[i]['dataset']==group for i in ex[mask,0]),
                                       order=('theta_gain,alpha_gain,RMS_gain' if words else
                                              'X_HbO,X_HbR,S_HbO,S_HbR,O_HbO,O_HbR' if val.size==6 else 'X,S,O'),
                                       values=json.dumps(val.ravel().tolist())))
    mdir=TRANSFER/'montage_word_search_v1'
    ref=read(mdir/'summary.json')['arms']
    for arm in ['frozen','semantic64','waveform']:
        collect('EEG training holdouts',arm,[mdir/d/f'held_fold{f}_scores.npz' for d in DATASETS for f in range(4)],
                arm+'__',EEG,ref[arm]['metrics'])
        files=[mdir/'single_word_v1'/f'fold_{f}_metrics.npz' for f in range(4)]
        for branch in 'SO':
            collect('EEG training holdouts single words',arm+'_'+branch,files,arm+'_'+branch+'_',['word'],words=True)
    for branch in 'SO':
        collect('EEG training holdouts single words','high_transient_source_'+branch,
                [TRANSFER/'hard_as_v1/local_information_v1/high_transient_readout_errors.npz'],
                branch+'_',['error'],words=True)
    for book,arm in [('cross_branch_innovations_capacity_v2','direct_mean'),
                     ('cross_branch_innovations_seed223_v2','innovation_mean')]:
        folder=TRANSFER/book
        files=[folder/d/f'fold_{f}_metrics.npz' for d in DATASETS for f in range(4)]
        collect('EEG training holdouts',book+'_'+arm,files,arm+'__',EEG,
                read(folder/'summary.json')['arms'][arm]['summary']['metrics'])
    event=TRANSFER/'native_event_replacement_v2'
    collect('EEG training holdouts','waveform_event',[event/d/f'held_fold{f}_scores.npz' for d in DATASETS for f in range(4)],
            'waveform_event__',EEG,read(event/'summary.json')['arms']['waveform_event']['metrics'])
    prior=TRANSFER/'integer_sum_source_v1'
    collect('EEG training holdouts','high_transient_source',[prior/f'high_{d}.npz' for d in DATASETS],
            'transient_source__',EEG,read(prior/'summary.json')['arms']['high_transient_source']['metrics'])
    hb=TRANSFER/'hb_nonlinear_event_source_v1'
    for arm in ['baseline111','nonlinear_array']:
        collect('Hb training holdouts',arm,[hb/(arm+'_metrics.npz')],'',HB,read(hb/'summary.json')['arms'][arm]['metrics'])
    for m,folder in [('eeg',RUN/'development_audit/as_local_gain_v1/spatial'),('hb',RUN/'hb_native_id_readout/v1/development')]:
        for arm,file,refpath,keys in [
            ('K128',RUN/'development_audit/historical_primary'/f'{m}_vq128_metrics.npz',
             RUN/'development_audit/historical_primary'/f'{m}_vq128_summary.json',['nmse']),
            ('AS' if m=='eeg' else 'Hb111',folder/'metrics.npz',folder/'summary.json',EEG if m=='eeg' else HB)]:
            collect(m+' development',arm,[file],'',keys,read(refpath)['metrics'])
    # Check the reconstruction and single-word joins against independent retained summaries.
    macro = {(r['arm'],r['support'],r['metric']):r for r in totals if r['dataset']=='macro'}
    q = read(TRANSFER/'dynamic_band_roles_v1/visual_review_v1/p7_ssm_tokenizer_qc_v1/case_qc.json')['sensitivity']
    np.testing.assert_allclose(json.loads(macro['frozen','clean','nmse']['values']),q['omit_flag'])
    assert macro['frozen','clean','nmse']['positions'] == q['total']-q['flagged']
    word_ref = read(mdir/'single_word_v1/summary.json')['arms']
    for arm in ['frozen','semantic64','waveform']:
        for b in 'SO':
            np.testing.assert_allclose(json.loads(macro[arm+'_'+b,'all','word']['values']),
                                       word_ref[arm][b]['word']['gain'])
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'hard_cases.csv',cases)
    write_csv(output/'frozen_scores.csv',totals)
    write_csv(output/'case_sources.csv',[dict(manifest=p,ranked_cases=n,flagged=h) for p,(n,h) in sources.items()])
    print('Unique worst or high-quantile cases:',Counter((v['modality'],v['split'],bool(v['worst']),bool(v['flags'])) for v in cases))
    print('Matched sources:',len(sources))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
