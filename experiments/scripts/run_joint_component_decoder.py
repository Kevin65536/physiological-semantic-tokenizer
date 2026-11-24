"""Fit paired Hb decoders from frozen G4 S/O integers on a fixed training draw."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import SOURCE, _state_refinement_loss, identity, owner_lock
from refit_joint_hb_decoder import continuity_loss
from run_joint_waveform_dictionary import preserve_record, write_json
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch, base_tokenizer_training_data, base_tokenizer_sample, base_tokenizer_sampling_index,
)
from src.tokenizers.contextual_latent_dictionary import load_model as load_source_model
from src.tokenizers.joint_component_decoder import JointComponentDecoder, load_model


def save(path, payload):
    temporary = path.with_suffix('.pending.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def source(config, manifest, scales):
    path = SOURCE/config['source']['checkpoint']
    retained = torch.load(path, map_location='cpu', weights_only=False)
    source_cfg = yaml.safe_load((SOURCE/config['source']['config']).read_text())
    if (retained['step'] != config['source']['step'] or retained['signature']['config'] != identity(source_cfg)
            or retained['signature']['input'] != identity(manifest) or retained['signature']['scales'] != identity(scales)
            or retained['metadata']['candidate'] != 'vq1024' or not retained['model_config']['hb_local_shape']):
        raise ValueError('exact frozen G4-2k source and G0 identity required')
    return retained, path


def initialized(source_record, routing):
    model = JointComponentDecoder(source_record['model_config'], routing=routing,
                                  codebook_size=source_record['metadata']['codebook_size'])
    missing = model.load_state_dict(source_record['model_state'], strict=False)
    allowed = ('embedding_mean', 'embedding_scale')
    if missing.unexpected_keys or any(key not in allowed and '.cross_from_latent.' not in key and
                                     '.free_output.' not in key for key in missing.missing_keys):
        raise ValueError('source encoder or decoder architecture mismatch')
    return model.eval()


def frozen_state(model):
    return {key: value.detach().clone() for key,value in model.state_dict().items()
            if key == 'position' or any('.'+name+'.' in key for name in ('input','encoder','to_latent','quantizer'))}


def prepare(config, output, manifest, scales):
    source_record, path = source(config, manifest, scales)
    signature = dict(config=identity(config), input=identity(manifest), scales=identity(scales),
                     source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if (output/'initial_model.pt').exists():
        if torch.load(output/'initial_model.pt',weights_only=False)['metadata']['signature'] != signature:
            raise ValueError('retained initialization changed')
        if (output/'training_bank.npz').exists() and (output/'preparation.json').exists():
            return
    model = initialized(source_record, 'joint')
    reference = load_source_model(path)
    rows = manifest['rows']
    examples = base_tokenizer_sample(np.random.default_rng(config['preparation']['seed']),
        base_tokenizer_sampling_index(rows,'hb'),config['preparation']['contexts'])
    if any(rows[i]['split'] != 'train' for i,_,_ in examples):
        raise ValueError('train-only decoder fitting support required before arrays')
    cache = OrderedDict()
    parts = {key:[] for key in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo','ids_S','ids_O')}
    started = time.monotonic()
    for offset in range(0,len(examples),64):
        take = examples[offset:offset+64]
        batch = base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
        with torch.inference_mode():
            codes = model.encode_ids(torch.as_tensor(batch['x'],dtype=torch.float32))
            replay = model.decode_ids(codes)
        original_ids = reference.encode(batch['x'])
        original_prediction = reference.decode_ids(original_ids)
        for branch in ('S','O'):
            np.testing.assert_array_equal(codes[branch].numpy(),original_ids[branch])
        for branch in ('S','O','sum'):
            np.testing.assert_array_equal(replay[branch].numpy(),original_prediction[branch])
        for key in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo'):
            value = batch[key]
            if key in ('s','o'):
                value = np.where(batch['mask'],value,0.)
            parts[key].append(value.astype(bool if key == 'mask' else np.float32))
        for branch in ('S','O'):
            parts['ids_'+branch].append(codes[branch].numpy().astype(np.int32))
        write_json(output/'preparation_progress.json',dict(completed=offset+len(take),planned=len(examples)))
    arrays = {key:np.concatenate(value) for key,value in parts.items()}
    means, deviations, modal = [], [], {}
    for branch in ('S','O'):
        code = arrays['ids_'+branch]
        with torch.inference_mode():
            embedded = model.branches[branch]['quantizer'].get_embedding(torch.as_tensor(code,dtype=torch.long)).numpy()
        means.append(embedded.mean((0,1)))
        deviations.append(np.maximum(embedded.std((0,1)),config['preparation']['embedding_scale_floor']))
        modal[branch] = int(np.bincount(code.reshape(-1),minlength=1024).argmax())
    model.embedding_mean.copy_(torch.as_tensor(np.stack(means)))
    model.embedding_scale.copy_(torch.as_tensor(np.stack(deviations)))
    metadata = dict(schema='joint_component_decoder_v1', modality='hb', routing='joint', codebook_size=1024,
        signature=signature,cfg_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'],
        source_checkpoint=str(path),source_step=2000,source_integer_encoder='frozen_G4',
        context_seconds=30,patch_seconds=2,integers_per_branch_position_context=15,
        fixed_width_bits_per_branch_position_second=5.,train_modal_ids=modal,
        dependency=dict(S=['S_ID_sequence','O_ID_sequence'],O=['S_ID_sequence','O_ID_sequence'],X=['S_ID_sequence','O_ID_sequence']))
    np.savez_compressed(output/'training_bank.pending.npz',examples=np.asarray(examples,dtype=np.int32),**arrays)
    (output/'training_bank.pending.npz').replace(output/'training_bank.npz')
    save(output/'initial_model.pt',dict(metadata=metadata,model_config=source_record['model_config'],model_state=model.state_dict(),step=0))
    write_json(output/'preparation.json',dict(status='completed',signature=signature,training_contexts=len(examples),
        modal_ids=modal,source_IDs_exact=True,source_initial_predictions_exact=True,
        fitting_split='train',validation_arrays_read=0,test_arrays_read=0,seconds=time.monotonic()-started))


def torch_batch(bank, indices):
    result = {}
    for key in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo'):
        value = bank[key][indices]
        if key.startswith('dv'):
            value = value[:,0,:2]
        result[key] = torch.as_tensor(value,dtype=torch.bool if key == 'mask' else torch.float32)
    return result


def objective(model, codes, batch, config):
    result = model.decode_ids(codes)
    result['commitment'] = result['S'].sum()*0.
    loss, terms = _state_refinement_loss(result,batch,config['loss'])
    boundary = continuity_loss(result,batch)
    return loss+config['loss']['boundary_weight']*boundary,dict(terms,boundary=boundary)


def benchmark(config, output):
    initial = torch.load(output/'initial_model.pt',map_location='cpu',weights_only=False)
    with np.load(output/'training_bank.npz') as stored:
        indices = np.arange(config['training']['batch_size'])
        bank = {key:stored[key][indices].copy() for key in stored if key != 'examples'}
    batch = torch_batch(bank,indices)
    codes = {branch:torch.as_tensor(bank['ids_'+branch],dtype=torch.long) for branch in ('S','O')}
    results = []
    for threads in config['resources']['benchmark_threads']:
        torch.set_num_threads(threads)
        model = JointComponentDecoder(initial['model_config'],routing='joint',codebook_size=1024)
        model.load_state_dict(initial['model_state'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002)
        model.train()
        began = time.monotonic()
        for _ in range(config['resources']['benchmark_steps']):
            loss,_ = objective(model,codes,batch,config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        seconds = time.monotonic()-began
        results.append(dict(torch_threads=threads,seconds=seconds,
            steps_per_second=config['resources']['benchmark_steps']/seconds,
            process_cumulative_peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    selected = max(results,key=lambda result:result['steps_per_second'])['torch_threads']
    write_json(output/'benchmark.json',dict(status='completed',results=results,selected_threads=selected,
        parallel_workers=2,reason='two independent equal-budget decoder arms; GPU remains occupied by unrelated work'))


def train(config, output, arm, manifest, scales):
    source(config,manifest,scales)
    initial = torch.load(output/'initial_model.pt',map_location='cpu',weights_only=False)
    if (initial['metadata']['signature']['config'] != identity(config) or
            initial['metadata']['signature']['input'] != identity(manifest)):
        raise ValueError('prepared training model identity changed')
    job = output/arm
    job.mkdir(exist_ok=True)
    signature = dict(initial['metadata']['signature'],arm=arm,runtime_torch_threads=torch.get_num_threads())
    if (job/'summary.json').exists():
        if json.loads((job/'summary.json').read_text())['signature'] != signature:
            raise ValueError('completed arm identity changed')
        return
    torch.manual_seed(config['training']['seed'])
    model = JointComponentDecoder(initial['model_config'],routing=arm,codebook_size=1024)
    model.load_state_dict(initial['model_state'])
    fixed = frozen_state(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    metadata = dict(initial['metadata'],routing=arm,signature=signature,
        dependency=dict(S=['S_ID_sequence'] if arm=='branch_only' else ['S_ID_sequence','O_ID_sequence'],
                        O=['O_ID_sequence'] if arm=='branch_only' else ['S_ID_sequence','O_ID_sequence'],
                        X=['S_ID_sequence','O_ID_sequence']),
        parameter_count_inference=sum(parameter.numel() for parameter in model.parameters()),
        trainable_decoder_parameters=sum(parameter.numel() for parameter in trainable))
    with np.load(output/'training_bank.npz') as stored:
        bank = {key:stored[key].copy() for key in stored if key != 'examples'}
    count = len(bank['x'])
    rng = np.random.default_rng(config['training']['seed'])
    checkpoint = job/'checkpoint.pt'
    start,history = 0,[]
    if checkpoint.exists():
        retained = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if retained['signature'] != signature:
            raise ValueError('decoder continuation changed')
        model.load_state_dict(retained['model_state'])
        optimizer.load_state_dict(retained['optimizer_state'])
        rng.bit_generator.state = retained['numpy_rng']
        torch.set_rng_state(retained['torch_rng'])
        start,history = retained['step'],retained['history']
    monitor_indices = np.random.default_rng(8031).integers(0,count,128)
    monitor = torch_batch(bank,monitor_indices)
    monitor_codes = {branch:torch.as_tensor(bank['ids_'+branch][monitor_indices],dtype=torch.long) for branch in ('S','O')}
    began = time.monotonic()
    for step in range(start,config['training']['steps']):
        indices = rng.integers(0,count,config['training']['batch_size'])
        batch = torch_batch(bank,indices)
        codes = {branch:torch.as_tensor(bank['ids_'+branch][indices],dtype=torch.long) for branch in ('S','O')}
        phase = (step+1)/config['training']['steps']
        rate = config['training']['final_learning_rate']+(config['training']['learning_rate']-config['training']['final_learning_rate'])*(1+np.cos(np.pi*phase))/2
        for group in optimizer.param_groups:
            group['lr'] = float(rate)
        model.train()
        loss,terms = objective(model,codes,batch,config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable,config['training']['gradient_clip'])
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite frozen-integer decoder fit')
        optimizer.step()
        if (step+1)%config['training']['checkpoint_every']==0:
            model.eval()
            with torch.inference_mode():
                value,monitor_terms = objective(model,monitor_codes,monitor,config)
                prediction = model.decode_ids(monitor_codes)
                branch_replay = 0.
                if arm == 'branch_only':
                    for branch in ('S','O'):
                        branch_replay = max(branch_replay,float((model.decode_branch_ids(branch,monitor_codes[branch])-prediction[branch]).abs().max()))
            if branch_replay != 0:
                raise AssertionError('branch-only replay changed when other branch was omitted')
            for key,value_in_source in fixed.items():
                if not torch.equal(value_in_source,model.state_dict()[key]):
                    raise AssertionError('frozen encoder/codebook changed')
            history.append(dict(step=step+1,loss=float(loss.detach()),monitor_loss=float(value),
                monitor={key:float(item) for key,item in monitor_terms.items()},branch_only_replay_max_abs=branch_replay,
                seconds=time.monotonic()-began))
            payload = dict(signature=signature,metadata=metadata,model_config=initial['model_config'],model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),step=step+1,history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            save(checkpoint,payload)
            if step+1 in config['training']['milestones']:
                save(job/f'checkpoint_step_{step+1:05d}.pt',payload)
            write_json(job/'progress.json',dict(status='training',step=step+1,budget=config['training']['steps'],history=history))
    save(job/'model.pt',dict(signature=signature,metadata=metadata,model_config=initial['model_config'],model_state=model.state_dict(),step=config['training']['steps']))
    write_json(job/'summary.json',dict(status='completed',signature=signature,history=history,
        frozen_encoder_and_codebook='exact_equal_to_G4_source',validation_arrays_read=0,test_arrays_read=0,
        parameter_count=metadata['parameter_count_inference'],trainable_decoder_parameters=metadata['trainable_decoder_parameters'],
        seconds=time.monotonic()-began))



def ablate(config, output, checkpoint, audit_source, manifest, scales):
    from argparse import Namespace
    from collections import defaultdict
    from evaluate_joint_tokenizer import load_source, input_scales
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary

    arguments = Namespace(panel=SOURCE/config['evaluation']['panel'], modality='hb', role='screen')
    rows, checked_scales, examples, checked_manifest = load_source(arguments)
    if identity(checked_manifest) != identity(manifest) or identity(checked_scales) != identity(scales):
        raise ValueError('ablation G0 panel identity mismatch')
    model = load_model(checkpoint)
    if model.metadata['signature']['config'] != identity(config):
        raise ValueError('ablation model configuration mismatch')
    with np.load(audit_source/'integer_ids.npz') as stored:
        np.testing.assert_array_equal(stored['examples'],examples)
        actual = {name:stored[name].copy() for name in ('S','O')}
    original, original_path = source(config,manifest,scales)
    source_model = load_source_model(original_path)
    with np.load(audit_source/'metrics.npz') as stored:
        np.testing.assert_array_equal(stored['examples'],examples)
        original_nmse = stored['nmse'].copy()
    donors = np.empty(len(examples),dtype=np.int64)
    donor_records = []
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples}):
        group = [n for n,(i,_,_) in enumerate(examples) if rows[i]['dataset'] == dataset]
        group.sort(key=lambda n:(rows[examples[n,0]]['subject'],rows[examples[n,0]]['record'],*examples[n]))
        record_keys = [(rows[examples[n,0]]['subject'],rows[examples[n,0]]['record']) for n in group]
        candidates = list(range(len(group)//2,len(group)))+list(range(1,len(group)//2))
        chosen = None
        for offset in candidates:
            shifted = np.roll(np.asarray(group),offset)
            if all(record_keys[j] != (rows[examples[k,0]]['subject'],rows[examples[k,0]]['record'])
                   for j,k in enumerate(shifted)):
                chosen = shifted
                break
        if chosen is None:
            raise ValueError('no complete cross-record permutation for registered dataset panel')
        donors[group] = chosen
    for n,donor in enumerate(donors):
        recipient_row, donor_row = rows[examples[n,0]], rows[examples[donor,0]]
        donor_records.append(dict(recipient=n,donor=int(donor),dataset=recipient_row['dataset'],
            recipient_subject=recipient_row['subject'],recipient_record=recipient_row['record'],
            donor_subject=donor_row['subject'],donor_record=donor_row['record'],
            same_position=recipient_row['positions'][examples[n,2]] == donor_row['positions'][examples[donor,2]]))
    modes = dict(normal={name:value.copy() for name,value in actual.items()})
    for name in ('S','O'):
        masked = {key:value.copy() for key,value in actual.items()}
        masked[name].fill(model.metadata['train_modal_ids'][name])
        modes[name+'_train_modal'] = masked
        shuffled = {key:value.copy() for key,value in actual.items()}
        shuffled[name] = actual[name][donors]
        modes[name+'_record_permuted'] = shuffled
    destination = audit_source/'ablations'
    destination.mkdir(exist_ok=True)
    if (destination/'summary.json').exists():
        raise FileExistsError('retained ablation already completed')
    payload = {mode+'_'+branch:ids[branch] for mode,ids in modes.items() for branch in ('S','O')}
    np.savez_compressed(destination/'integer_ids.npz',examples=examples,**payload)
    with np.load(destination/'integer_ids.npz') as stored:
        np.testing.assert_array_equal(stored['examples'],examples)
        modes = {mode:{branch:stored[mode+'_'+branch].copy() for branch in ('S','O')} for mode in modes}
    model = load_model(checkpoint)
    metrics = {mode:defaultdict(list) for mode in modes}
    support = {mode:defaultdict(list) for mode in modes}
    cross_effect = {mode:0. for mode in modes if mode != 'normal'}
    branch_replay = 0.
    cache = OrderedDict()
    for offset in range(0,len(examples),32):
        take = examples[offset:offset+32]
        batch = base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=32)
        reference_ids, current_ids = source_model.encode(batch['x']), model.encode(batch['x'])
        for branch in ('S','O'):
            np.testing.assert_array_equal(reference_ids[branch],actual[branch][offset:offset+len(take)])
            np.testing.assert_array_equal(current_ids[branch],reference_ids[branch])
        normal_prediction = None
        for mode,ids in modes.items():
            codes = {branch:value[offset:offset+len(take)] for branch,value in ids.items()}
            prediction = model.decode_ids(codes)
            if mode == 'normal':
                normal_prediction = prediction
                if model.metadata['routing'] == 'branch_only':
                    for branch in ('S','O'):
                        repeated = model.decode_branch_ids(branch,codes[branch])
                        branch_replay = max(branch_replay,float(np.max(abs(repeated-prediction[branch]))))
            else:
                unaffected_branch = 'O' if mode.startswith('S_') else 'S'
                difference = float(np.max(abs(prediction[unaffected_branch]-normal_prediction[unaffected_branch])))
                cross_effect[mode] = max(cross_effect[mode],difference)
                if model.metadata['routing'] == 'branch_only' and difference != 0:
                    raise AssertionError('branch-only output changed under foreign-ID intervention')
            audited = joint_reconstruction_metrics(prediction,batch,2,input_scales(rows,take,scales))
            for key,value in audited['metrics'].items():
                metrics[mode][key].append(value)
            for key,value in audited['support'].items():
                support[mode][key].append(value)
        write_json(destination/'progress.json',dict(completed=offset+len(take),planned=len(examples)))
    if branch_replay != 0:
        raise AssertionError('branch-only independent replay differs')
    summaries = {}
    for mode in modes:
        values = {key:np.concatenate(parts) for key,parts in metrics[mode].items()}
        counts = {key:np.concatenate(parts) for key,parts in support[mode].items()}
        if mode == 'normal':
            np.testing.assert_array_equal(values['nmse'],original_nmse)
        np.savez_compressed(destination/(mode+'_metrics.npz'),examples=examples,**values,
            **{'support_'+key:value for key,value in counts.items()})
        summaries[mode] = joint_reconstruction_summary(values,counts,examples.tolist(),rows)
    original_macro = np.asarray(summaries['normal']['metrics']['nmse']['macro'])
    write_json(destination/'summary.json',dict(status='completed',checkpoint=str(checkpoint),routing=model.metadata['routing'],
        source_IDs_exact_to_G4=True,branch_only_replay_max_abs=branch_replay,
        dependency=model.metadata['dependency'],integers_per_position_30s=30,
        fixed_width_bits_per_position_second=10.,source_pattern='unchanged encoder and quantizer',
        intervention=dict(training_modal_ids=model.metadata['train_modal_ids'],donor_records=donor_records,
            meaning='Replace one real branch integer sequence while retaining the other real sequence and target window'),
        foreign_ID_max_abs_change=cross_effect,
        nmse_macro_delta_vs_normal={mode:(np.asarray(value['metrics']['nmse']['macro'])-original_macro).tolist()
                                    for mode,value in summaries.items()},modes=summaries))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','benchmark','train','ablate'],required=True)
    parser.add_argument('--arm',choices=['branch_only','joint'])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--audit-source',type=Path)
    args = parser.parse_args()
    if args.stage == 'ablate' and (args.checkpoint is None or args.audit_source is None):
        parser.error('--checkpoint and --audit-source required for ablate')
    if args.stage == 'train' and args.arm is None:
        parser.error('--arm required for train')
    config = yaml.safe_load(args.config.read_text())
    if config['schema'] != 'dual_component_joint_cross_decoder_v2':
        raise ValueError('versioned frozen-integer decoder contract required')
    args.output.mkdir(exist_ok=True,parents=True)
    preserve_record(args.output/'resolved_config.json',config)
    threads = (json.loads((args.output/'benchmark.json').read_text())['selected_threads'] if args.stage=='train'
               else (2 if args.stage=='ablate' else config['resources']['preparation_threads']))
    torch.set_num_threads(threads)
    manifest,scales,_,_ = base_tokenizer_training_data(config,SOURCE)
    lock_output = args.output if args.stage != 'train' else args.output/args.arm
    with owner_lock(lock_output):
        try:
            if args.stage=='prepare':
                prepare(config,args.output,manifest,scales)
            elif args.stage=='benchmark':
                benchmark(config,args.output)
            elif args.stage=='ablate':
                ablate(config,args.output,args.checkpoint,args.audit_source,manifest,scales)
            else:
                train(config,args.output,args.arm,manifest,scales)
        except BaseException as error:
            write_json(lock_output/'failure.json',dict(type=type(error).__name__,reason=str(error)))
            raise


if __name__ == '__main__':
    main()
