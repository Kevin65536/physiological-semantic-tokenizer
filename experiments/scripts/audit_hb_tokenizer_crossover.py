"""Measure whether the reconstruction gain depends on coordinated errors in frozen Hb branches."""
import argparse
from collections import OrderedDict, defaultdict
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from evaluate_joint_tokenizer import SOURCE, input_scales, load_candidate, predict
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data


def run(output):
    run_dir = SOURCE / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1'
    legacy_run = run_dir.parent/'20261122_unified_base_g1_v1'
    config = yaml.safe_load((SOURCE/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(config, SOURCE)
    rows = manifest['rows']
    with np.load(run_dir/'panels.npz', allow_pickle=False) as panels:
        panel = panels['hb_screen']
    if any(rows[i]['split'] != 'validation' or rows[i]['modality'] != 'hb' for i, _, _ in panel):
        raise ValueError('expected development Hb observations')
    sources = dict(base=('legacy',run_dir/'baseline_audit/hb/vq128'),
                   aux=('contextual',run_dir/'candidate_audit/hb_prototype/auxiliary/step_02000/hard'))
    records = {}
    for name, (kind, folder) in sources.items():
        summary = json.loads((folder/'summary.json').read_text())
        args = SimpleNamespace(kind=kind, modality='hb', continuous=False, checkpoint=Path(summary['checkpoint']),
                               legacy_run=legacy_run)
        with np.load(folder/'integer_ids.npz',allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['examples'], panel)
            ids = {b: archive[b].copy() for b in ('S','O')}
        records[name] = args, load_candidate(args, manifest, scales), ids
    pairs = list(product(records, repeat=2))
    all_metrics = {pair:defaultdict(list) for pair in pairs}
    support = defaultdict(list)
    cache = OrderedDict()
    for start in range(0,len(panel),32):
        take = panel[start:start+32]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=32)
        waves = {}
        for name,(args,model,ids) in records.items():
            packets = {b:values[start:start+len(take)] for b,values in ids.items()}
            waves[name], _ = predict(model, batch, args, packets)
        for pair in pairs:
            s, o = waves[pair[0]]['S'], waves[pair[1]]['O']
            predicted = {'S':s,'O':o,'sum':s+o}
            audited = joint_reconstruction_metrics(predicted,batch,2,input_scales(rows,take,scales))
            for key,value in audited['metrics'].items():
                all_metrics[pair][key].append(value)
            mask = batch['mask'].reshape(-1,300,2)
            err_s = (s.astype(float)-batch['s']).reshape(-1,300,2)
            err_o = (o.astype(float)-batch['o']).reshape(-1,300,2)
            variance = batch['vx'][:,0,:2]
            terms = [np.where(mask,x,0.).sum(1)/mask.sum(1)/variance
                     for x in (err_s**2,err_o**2,2*err_s*err_o)]
            np.testing.assert_allclose(sum(terms),audited['metrics']['nmse'][:,0],rtol=1e-4,atol=2e-6)
            all_metrics[pair]['sum_error_terms'].append(np.stack(terms,axis=1))
            if pair==pairs[0]:
                for key,value in audited['support'].items():
                    support[key].append(value)
    compact = {}
    combined_support={k:np.concatenate(v) for k,v in support.items()}
    for pair in pairs:
        label='_'.join(pair)
        metrics={k:np.concatenate(v) for k,v in all_metrics[pair].items()}
        summary=joint_reconstruction_summary(metrics,combined_support,panel.tolist(),rows)
        (output/(label+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
        np.savez_compressed(output/(label+'_metrics.npz'),examples=panel,**metrics)
        compact[label]=dict(nmse=summary['metrics']['nmse']['macro'],
                            terms=summary['metrics']['sum_error_terms']['macro'],
                            macro=summary['selection_score'])
    (output/'summary.json').write_text(json.dumps(dict(status='completed',sources={k:str(v[0].checkpoint) for k,v in records.items()},
        roles='first label supplies S, second supplies O',integer_count='15 per branch/position/window',
        denominator='per-color G0 X variance for all three sum_error_terms fields',
        term_order=['mean eS squared','mean eO squared','twice mean eS*eO'],
        panel='476 Hb development screen position windows; all same IDs as source audit',
        results=compact),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.output/'summary.json').exists():
        raise FileExistsError('completed crossover already exists')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    run(args.output)
