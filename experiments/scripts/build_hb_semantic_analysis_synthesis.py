#!/usr/bin/env python3
"""Fit semantic O-code neighborhoods and choose search work using fixed training examples."""
import argparse
from collections import OrderedDict,defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import SOURCE,identity,write_json
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.tokenizers.contextual_latent_dictionary import load_model
from src.tokenizers.hb_semantic_analysis_synthesis import HbSemanticAnalysisSynthesis


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config['schema'] != 'hb_semantic_analysis_synthesis_config_v1':
        raise ValueError('versioned train-only analysis-synthesis contract required')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        raise FileExistsError('completed frozen search retained')
    torch.set_num_threads(config['resources']['torch_threads'])
    manifest,scales,_,_ = base_tokenizer_training_data(config,SOURCE)
    rows = manifest['rows']
    source_path = SOURCE/config['source_checkpoint']
    source = load_model(source_path)
    if source.metadata['signature']['input'] != identity(manifest):
        raise ValueError('source G0 identity drift')
    pattern = SOURCE/config['training_pattern']
    report = json.loads((pattern/'summary.json').read_text())
    if Path(report['checkpoint']).resolve() != source_path.resolve():
        raise ValueError('training prototype source checkpoint mismatch')
    fields = report['attributes']
    variance = report['branches']['O']['training_attribute_variance']
    atlas = json.loads((pattern/'O_atlas.json').read_text())
    with np.load(pattern/'O_training_prototypes.npz') as stored:
        mean = stored['conditional_mean'].copy()
        slices = dict(zip(fields,map(tuple,stored['attribute_slices'])))
        counts = stored['count'].copy()
    groups = ['HbO_centered','HbR_centered','HbO_slope','HbR_slope','HbO_level','HbR_level']
    distances = {}
    for name in groups:
        start,stop = slices[name]
        delta = mean[:,None,start:stop]-mean[None,:,start:stop]
        distances[name] = np.mean(delta*delta,axis=-1)/variance[name]
    size = len(mean)
    candidates = np.zeros((size,max(config['search_budgets'])),dtype=np.int64)
    supported = ~np.asarray(atlas['low_training_support'],bool)
    for code in range(size):
        allowed = supported.copy()
        for name in groups[:4]:
            allowed &= distances[name][code]<=config['maximum_normalized_semantic_drift']
        for name in groups[4:]:
            allowed &= distances[name][code]<=config['maximum_normalized_level_distance']
        allowed &= np.isfinite(mean).all(axis=1)
        allowed[code] = False
        rank = sum(distances[name][code] for name in groups)
        order = np.flatnonzero(allowed)
        order = order[np.argsort(rank[order],kind='stable')]
        codes = np.r_[code,order[:candidates.shape[1]-1]]
        candidates[code] = np.pad(codes,(0,candidates.shape[1]-len(codes)),constant_values=code)
    x_values = defaultdict(list)
    for dataset,positions in scales['positions']['hb'].items():
        for position,record in positions.items():
            x_values[dataset].append(record['target_variance']['x'])
    x_variance = np.mean([np.mean(values,axis=0) for values in x_values.values()],axis=0)
    shape_variance = np.asarray([variance['HbO_centered'],variance['HbR_centered']])
    slope_variance = np.asarray([variance['HbO_slope'],variance['HbR_slope']])
    o_change_variance = np.asarray([variance['HbO_wave'],variance['HbR_wave']])
    panel = SOURCE/config['panel']
    panel_record = json.loads((panel.parent/'panel.json').read_text())
    if hashlib.sha256(panel.read_bytes()).hexdigest() != panel_record['panel_sha256']:
        raise ValueError('training panel identity drift')
    with np.load(panel) as stored:
        examples = stored['hb_pattern_train'].copy()
    by_subject = defaultdict(list)
    for example in examples:
        row = rows[example[0]]
        if row['split'] != 'train':
            raise ValueError('search-budget training boundary before arrays')
        by_subject[row['dataset'],row['subject']].append(example)
    rng = np.random.default_rng(config['pilot_seed'])
    pilot = np.asarray([example for key,group in sorted(by_subject.items())
        for example in np.asarray(group)[rng.choice(len(group),min(config['pilot_per_subject'],len(group)),replace=False)]])
    signature = dict(config=identity(config),input=identity(manifest),scales=identity(scales),
        original_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        training_prototypes_sha256=hashlib.sha256((pattern/'O_training_prototypes.npz').read_bytes()).hexdigest(),
        training_report_sha256=hashlib.sha256((pattern/'summary.json').read_bytes()).hexdigest(),
        training_atlas_sha256=hashlib.sha256((pattern/'O_atlas.json').read_bytes()).hexdigest(),
        panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest())
    metadata = dict(source.metadata,schema='hb_semantic_analysis_synthesis_v1',signature=signature,
        source_file='source.pt',source_checkpoint=str(source_path),source_sha256='',
        maximum_normalized_semantic_drift=config['maximum_normalized_semantic_drift'],
        coordinate_passes=1,neighbor_budget=1,
        analysis_input='30s observed X plus deterministic source integer reconstruction',
        decoding='source S/O integer sequences only',search_neighborhood='training_conditional_shape_slope_level',
        continuous_diagnostic='source prequantization path without integer search')
    cache = OrderedDict()
    results = []
    settings = [(budget,penalty) for budget in config['search_budgets']
        for penalty in (config.get('change_penalties',[0.]) if budget>1 else [0.])]
    for budget,penalty in settings:
        metadata['neighbor_budget'] = budget
        metadata['search_change_penalty'] = penalty
        model = HbSemanticAnalysisSynthesis(source,dict(metadata),candidates,x_variance,shape_variance,slope_variance,o_change_variance)
        errors,baseline_errors,o_errors,baseline_o_errors = [],[],[],[]
        began = time.monotonic()
        for offset in range(0,len(pilot),config['resources']['batch_size']):
            take = pilot[offset:offset+config['resources']['batch_size']]
            batch = base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=128)
            before = source(batch['x'])
            after = model(batch['x'])
            if not np.array_equal(before['ids']['S'],after['ids']['S']) or not np.array_equal(before['S'],after['S']):
                raise AssertionError('search changed frozen S mapping')
            errors.append(model.error(after['sum'],batch['x']))
            baseline_errors.append(model.error(before['sum'],batch['x']))
            mask = batch['mask'].reshape(-1,300,2)
            def o_error(value):
                error = ((value-batch['o'])**2/batch['vo']).reshape(-1,300,2)
                return np.where(mask,error,0.).sum(1)/np.maximum(mask.sum(1),1)
            o_errors.append(o_error(after['O']))
            baseline_o_errors.append(o_error(before['O']))
            write_json(out/'progress.json',dict(stage='train_search_budget',budget=budget,penalty=penalty,completed=offset+len(take),total=len(pilot)))
        def aggregate(values):
            return float(base_training_aggregate({'nmse':np.concatenate(values)[:,None]},pilot.tolist(),rows)['metrics']['nmse']['macro'][0])
        current,base = aggregate(errors),aggregate(baseline_errors)
        def aggregate_color(values):
            return np.asarray(base_training_aggregate({'nmse':np.concatenate(values)},pilot.tolist(),rows)['metrics']['nmse']['macro'])
        o_error,base_o_error = aggregate_color(o_errors),aggregate_color(baseline_o_errors)
        eligible = bool((o_error<=config.get('training_O_nmse_ratio_max',float('inf'))*base_o_error+1e-12).all())
        results.append(dict(neighbors=budget,change_penalty=penalty,observed_X_error=current,base_observed_X_error=base,
            O_nmse=o_error,base_O_nmse=base_o_error,eligible=eligible,
            improvement=base-current,seconds=time.monotonic()-began))
    eligible_results = [value for value in results if value['eligible']]
    if not eligible_results:
        raise ValueError('no search setting met the fixed training O guard')
    best = max(value['improvement'] for value in eligible_results)
    chosen = min((value for value in eligible_results if value['improvement']>=config['gain_fraction_of_best']*best),
        key=lambda value:(value['neighbors'],-value['change_penalty']))
    metadata['neighbor_budget'] = chosen['neighbors']
    metadata['search_change_penalty'] = chosen['change_penalty']
    saved = torch.load(source_path,map_location='cpu',weights_only=False)
    compact = {key:saved[key] for key in ('metadata','model_config','model_state','signature','step')}
    torch.save(compact,out/'source.pt')
    metadata['source_sha256'] = hashlib.sha256((out/'source.pt').read_bytes()).hexdigest()
    torch.save(dict(metadata=metadata,neighbors=candidates,x_variance=x_variance,
        shape_variance=shape_variance,slope_variance=slope_variance,o_change_variance=o_change_variance),out/'model.pt')
    write_json(out/'resolved_config.json',config)
    write_json(out/'summary.json',dict(status='completed',signature=signature,chosen_neighbor_budget=chosen['neighbors'],
        chosen_change_penalty=chosen['change_penalty'],
        training_pilot_examples=pilot,training_budget_results=results,original_source_sha256=signature['original_source_sha256'],
        frozen_S_mapping='IDs_and_waveforms_exact',new_training_updates=0,validation_arrays_read=0,test_arrays_read=0,
        fixed_width_bits_per_second_per_branch_position={'S':3.5,'O':4.},
        inference_package_bytes=(out/'source.pt').stat().st_size+(out/'model.pt').stat().st_size))


if __name__=='__main__':
    main()
