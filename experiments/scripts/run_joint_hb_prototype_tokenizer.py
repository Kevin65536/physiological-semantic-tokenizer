#!/usr/bin/env python3
"""Train matched Hb K128 continuations with single-ID shape/slope supervision."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import (SOURCE,_base_training_batch,_state_refinement_loss,
    identity,write_json,owner_lock)
from src.data.dual_component_tokenizer import (base_tokenizer_training_data,
    base_tokenizer_sample,base_tokenizer_sampling_index)
from src.tokenizers.contextual_latent_dictionary import PrototypeSupervisedHbTokenizer


def save(path,payload):
    temporary = path.with_suffix('.pending.pt')
    torch.save(payload,temporary)
    temporary.replace(path)


def source_model(config,manifest,scales):
    source_cfg = yaml.safe_load((SOURCE/config['source']['config']).read_text())
    path = SOURCE/config['source']['checkpoint']
    saved = torch.load(path,map_location='cpu',weights_only=False)
    if (saved['step'] != 10000 or saved['signature']['config'] != identity(source_cfg)
            or saved['signature']['input'] != identity(manifest) or saved['signature']['scales'] != identity(scales)
            or saved['signature']['task'] != dict(modality='hb',architecture='waveform',candidate='vq128')):
        raise ValueError('original v3 hard G1 source required')
    setting = dict(source_cfg['model'],**source_cfg['architectures']['waveform'],
        input_dim=40,channels=2,sample_strides=[5])
    branch_sizes = config.get('branch_codebook_sizes')
    if branch_sizes is not None:
        setting['branch_codebook_sizes'] = branch_sizes
    model = PrototypeSupervisedHbTokenizer(setting,codebook_size=128,hard=True).eval()
    replace_o = branch_sizes is not None and branch_sizes['O'] != 128
    source_state = {key:value for key,value in saved['model_state'].items()
                    if not (replace_o and key.startswith('branches.O.quantizer.'))}
    loaded = model.load_state_dict(source_state,strict=False)
    if loaded.unexpected_keys or any(not (key.startswith('prototype_heads.') or
            replace_o and key.startswith('branches.O.quantizer.')) for key in loaded.missing_keys):
        raise ValueError('prototype source architecture drift')
    signature = dict(config=identity(config),input=identity(manifest),scales=identity(scales),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    initializer = config['initialization'].get('checkpoint')
    if initializer is not None:
        signature['initializer_sha256'] = hashlib.sha256((SOURCE/initializer).read_bytes()).hexdigest()
    return model,setting,signature


def prepare(config,out,manifest,scales):
    model,setting,signature = source_model(config,manifest,scales)
    destination = out/'initialization.pt'
    if destination.exists():
        retained = torch.load(destination,map_location='cpu',weights_only=False)
        if retained['signature'] != signature:
            raise ValueError('prototype initialization identity changed')
        return
    initializer = config['initialization'].get('checkpoint')
    if initializer is not None:
        original = torch.load(SOURCE/initializer,map_location='cpu',weights_only=False)
        for key in ('input','scales','source_sha256'):
            if original['signature'][key] != signature[key]:
                raise ValueError('reused prototype initialization source drift')
        expected_setting = dict(original['model_config'])
        replace_o = config.get('branch_codebook_sizes',{}).get('O',128) != 128
        if 'branch_codebook_sizes' in config:
            expected_setting['branch_codebook_sizes'] = config['branch_codebook_sizes']
        if expected_setting != setting:
            raise ValueError('reused prototype initialization architecture drift')
        expected = model.state_dict()
        for key,value in original['model_state'].items():
            if (not key.startswith('prototype_heads.') and not (replace_o and key.startswith('branches.O.quantizer.'))
                    and not torch.equal(expected[key],value)):
                raise ValueError('reused initialization changed the original tokenizer')
        reused = {key:value for key,value in original['model_state'].items()
                  if not (replace_o and key.startswith('branches.O.quantizer.'))}
        loaded = model.load_state_dict(reused,strict=False)
        if loaded.unexpected_keys or any(not key.startswith('branches.O.quantizer.') for key in loaded.missing_keys):
            raise ValueError('reused prototype tensor mismatch')
        dictionary = initialize_o_dictionary(model,config,manifest['rows'],scales) if replace_o else None
        auxiliary = original['auxiliary_scales']
        weights,evidence = calibrate_weights(model,config,manifest['rows'],scales,auxiliary)
        metadata = dict(original['metadata'],signature=signature,frozen_branches=config['training'].get('frozen_branches',[]))
        if 'branch_codebook_sizes' in config:
            metadata.update(codebook_size=max(config['branch_codebook_sizes'].values()),
                branch_codebook_sizes=config['branch_codebook_sizes'],
                bits_per_branch_position_per_second={b:float(np.log2(k)/2) for b,k in config['branch_codebook_sizes'].items()})
        save(destination,dict(signature=signature,metadata=metadata,model_config=setting,model_state=model.state_dict(),
            auxiliary_scales=auxiliary,calibrated_auxiliary_weights=weights))
        write_json(out/'initialization.json',dict(status='completed',signature=signature,
            source_initialization=str(SOURCE/initializer),dictionary=dictionary,calibration=evidence,calibrated_auxiliary_weights=weights))
        return
    rows = manifest['rows']
    groups = base_tokenizer_sampling_index(rows,'hb')
    examples = base_tokenizer_sample(np.random.default_rng(config['initialization']['seed']),groups,
        config['initialization']['contexts'])
    cache = OrderedDict()
    width = int(setting['latent_dim'])+1
    xx = {key:np.zeros((2,width,width)) for key in ('S','O')}
    xy = {key:np.zeros((2,width,20)) for key in ('S','O')}
    moments = {key:np.zeros((2,2)) for key in ('S','O')}
    counts = np.zeros(2,dtype=np.int64)
    clock = (np.arange(20)-9.5)/10
    tick = time.monotonic()
    torch.set_num_threads(config['resources']['torch_threads'])
    for offset in range(0,len(examples),64):
        take = examples[offset:offset+64]
        if any(rows[i]['split'] != 'train' for i,_,_ in take):
            raise ValueError('prototype initialization training boundary')
        batch = _base_training_batch(rows,take,scales,'cpu',cache,config)
        complete = batch['mask'].numpy().reshape(-1,20,2).all(1)
        counts += complete.sum(0)
        with torch.inference_mode():
            latent = model.encode_continuous(batch['x'])
        for branch,target in [('S','s'),('O','o')]:
            quantizer = model.branches[branch]['quantizer']
            ids = quantizer._assignment_logits(latent[branch].reshape(-1,setting['latent_dim'])).argmax(-1)
            feature = quantizer.get_embedding(ids).numpy().astype(float)
            feature = np.c_[feature,np.ones(len(feature))]
            y = batch[target].numpy().reshape(-1,20,2).astype(float)
            centered = y-y.mean(1,keepdims=True)
            slope = np.einsum('ntc,t->nc',centered,clock)/np.sum(clock**2)
            for color in range(2):
                valid = complete[:,color]
                x = feature[valid]
                z = centered[valid,:,color]
                xx[branch][color] += x.T@x
                xy[branch][color] += x.T@z
                moments[branch][0,color] += np.sum(z**2)
                moments[branch][1,color] += np.sum(slope[valid,color]**2)
        if offset%512==0:
            write_json(out/'preparation_progress.json',dict(completed=offset+len(take),planned=len(examples)))
    auxiliary = {}
    for branch in ('S','O'):
        if (counts==0).any():
            raise ValueError('empty paired training prototype support')
        coefficients = []
        for color in range(2):
            matrix = xx[branch][color]
            penalty = np.eye(width)*config['initialization']['ridge_fraction']*np.trace(matrix[:-1,:-1])/(width-1)
            penalty[-1,-1] = 0
            coefficients.append(np.linalg.solve(matrix+penalty,xy[branch][color]))
        fitted = np.stack(coefficients,axis=-1).reshape(width,40)
        with torch.no_grad():
            model.prototype_heads[branch].weight.copy_(torch.as_tensor(fitted[:-1].T,dtype=torch.float32))
            model.prototype_heads[branch].bias.copy_(torch.as_tensor(fitted[-1],dtype=torch.float32))
        auxiliary[branch+'_shape_mean_square'] = np.maximum(moments[branch][0]/(counts*20),config['initialization']['scale_floor'])
        auxiliary[branch+'_slope_mean_square'] = np.maximum(moments[branch][1]/counts,config['initialization']['scale_floor'])
    metadata = dict(schema='contextual_latent_dictionary_v1',modality='hb',codebook_size=128,
        decoder_architecture='prototype_supervised_hb_v1',signature=signature,
        cfg_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'],
        input='X_only',context_seconds=30,patch_seconds=2,quantization_levels=1,
        integers_per_branch_position_context=15,bits_per_branch_position_per_second=3.5,
        source=str(SOURCE/config['source']['checkpoint']),source_step=10000,
        source_historical_validation_exposure='all_original_validation_subjects',
        parameter_count_inference=sum(p.numel() for p in model.parameters()))
    save(destination,dict(signature=signature,metadata=metadata,model_config=setting,
        model_state=model.state_dict(),auxiliary_scales=auxiliary))
    write_json(out/'initialization.json',dict(status='completed',signature=signature,examples=examples,
        complete_patches_per_color=counts,auxiliary_scales=auxiliary,
        normalization='training mean-square centered shape and slope; input-SD coordinates and per-second slopes',
        seconds=time.monotonic()-tick))


def initialize_o_dictionary(model,config,rows,scales):
    spec = config['initialization']
    torch.set_num_threads(config['resources']['torch_threads'])
    torch.manual_seed(spec['new_codebook_seed'])
    groups = base_tokenizer_sampling_index(rows,'hb')
    examples = base_tokenizer_sample(np.random.default_rng(spec['new_codebook_seed']),groups,spec['new_codebook_contexts'])
    cache = OrderedDict()
    latent_parts,target_parts,mask_parts = [],[],[]
    model.eval()
    for offset in range(0,len(examples),64):
        take = examples[offset:offset+64]
        if any(rows[i]['split'] != 'train' for i,_,_ in take):
            raise ValueError('O dictionary training boundary')
        batch = _base_training_batch(rows,take,scales,'cpu',cache,config)
        with torch.inference_mode():
            latent = model._encode_branch('O',batch['x'])
        latent_parts.append(latent.reshape(-1,latent.shape[-1]))
        target = batch['o'].numpy().reshape(-1,20,2).astype(float)
        target_parts.append(target-target.mean(1,keepdims=True))
        mask_parts.append(batch['mask'].numpy().reshape(-1,20,2).all(1))
    latent = torch.cat(latent_parts)
    quantizer = model.branches['O']['quantizer']
    quantizer.train()
    with torch.no_grad():
        quantizer(latent)
    quantizer.eval()
    ids = quantizer._assignment_logits(latent).argmax(-1)
    feature = quantizer.get_embedding(ids).numpy().astype(float)
    feature = np.c_[feature,np.ones(len(feature))]
    target = np.concatenate(target_parts)
    complete = np.concatenate(mask_parts)
    coefficients = []
    for color in range(2):
        x = feature[complete[:,color]]
        y = target[complete[:,color],:,color]
        matrix = x.T@x
        penalty = np.eye(matrix.shape[0])*spec['ridge_fraction']*np.trace(matrix[:-1,:-1])/(matrix.shape[0]-1)
        penalty[-1,-1] = 0
        coefficients.append(np.linalg.solve(matrix+penalty,x.T@y))
    fitted = np.stack(coefficients,axis=-1).reshape(feature.shape[1],40)
    with torch.no_grad():
        model.prototype_heads['O'].weight.copy_(torch.as_tensor(fitted[:-1].T,dtype=torch.float32))
        model.prototype_heads['O'].bias.copy_(torch.as_tensor(fitted[-1],dtype=torch.float32))
    return dict(examples=examples,contexts=len(examples),patches=len(latent),
        codebook_size=quantizer.codebook_size,quantizer_update_count=int(quantizer.update_count),
        initialization='training_X_prequantization_latent_kmeans_then_O_prototype_ridge',
        S_vocabulary='original_128_exactly_preserved',mean_square_scales='reused_frozen_training_scales')


def auxiliary_loss(result,batch,scales,targets=None):
    mask = batch['mask'].reshape(len(batch['x']),15,20,2).all(2)
    clock = (torch.arange(20,dtype=batch['x'].dtype)-9.5)/10
    shape_terms,slope_terms = [],[]
    for branch,target in [('S','s'),('O','o')]:
        if targets is not None and branch not in targets:
            continue
        colors = [0,1] if targets is None else targets[branch]
        predicted = result[branch+'_prototype'].reshape(len(mask),15,20,2)
        y = batch[target].reshape(predicted.shape)
        centered = y-y.mean(2,keepdim=True)
        variance = predicted.new_tensor(scales[branch+'_shape_mean_square'])
        error = (predicted-centered).square()/variance
        score = torch.where(mask[:,:,None,:],error,0.).sum((1,2))/(mask.sum(1)*20).clamp_min(1)
        shape_terms.append(score[:,colors].mean())
        predicted_slope = torch.einsum('bptc,t->bpc',predicted,clock)/clock.square().sum()
        true_slope = torch.einsum('bptc,t->bpc',centered,clock)/clock.square().sum()
        variance = predicted.new_tensor(scales[branch+'_slope_mean_square'])
        error = (predicted_slope-true_slope).square()/variance
        score = torch.where(mask,error,0.).sum(1)/mask.sum(1).clamp_min(1)
        slope_terms.append(score[:,colors].mean())
    return torch.stack(shape_terms).mean(),torch.stack(slope_terms).mean()


def freeze_branches(model,config):
    frozen = config['training'].get('frozen_branches',[])
    if frozen == ['S']:
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('branches.O.','prototype_heads.O.')))
    for branch in frozen:
        model.branches[branch].requires_grad_(False)
        model.prototype_heads[branch].requires_grad_(False)
    return frozen


def calibrate_weights(model,config,rows,scales,auxiliary):
    spec = config['gradient_calibration']
    torch.set_num_threads(config['resources']['torch_threads'])
    frozen = freeze_branches(model,config)
    baseline_state = {key:value.clone() for key,value in model.state_dict().items()}
    parameters = [parameter for name in ('input','encoder','to_latent')
                  for parameter in model.branches['O'][name].parameters()]
    rng = np.random.default_rng(spec['seed'])
    groups = base_tokenizer_sampling_index(rows,'hb')
    cache = OrderedDict()
    squares = dict(main=0.,shape=0.,slope=0.)
    examples = []
    for _ in range(spec['batches']):
        take = base_tokenizer_sample(rng,groups,spec['batch_size'])
        examples.extend(take)
        if any(rows[i]['split'] != 'train' for i,_,_ in take):
            raise ValueError('gradient calibration training boundary')
        batch = _base_training_batch(rows,take,scales,'cpu',cache,config)
        model.train()
        for branch in model.branches.values():
            branch['quantizer'].eval()
        for branch in frozen:
            model.branches[branch].eval()
        result = model(batch['x'])
        main,_ = _state_refinement_loss(result,batch,config['loss'])
        shape,slope = auxiliary_loss(result,batch,auxiliary,config.get('auxiliary_targets'))
        for name,value in [('main',main),('shape',shape),('slope',slope)]:
            gradient = torch.autograd.grad(value,parameters,retain_graph=name!='slope')
            squares[name] += sum(float(item.detach().square().sum()) for item in gradient)
    for key,value in baseline_state.items():
        if not torch.equal(value,model.state_dict()[key]):
            raise AssertionError('gradient calibration changed tokenizer state')
    weights = {name:min(config['auxiliary']['auxiliary'][name],
        spec['encoder_gradient_fraction']/2*np.sqrt(squares['main']/max(squares[name],1e-20)))
        for name in ('shape','slope')}
    evidence = dict(examples=examples,gradient_squared_norm_sums=squares,
        encoder_parameters='O input/encoder/to_latent',encoder_gradient_fraction=spec['encoder_gradient_fraction'],
        fixed_weights=weights,state_after_calibration='exact_equal_to_source')
    return weights,evidence


def train(config,out,arm,manifest,scales):
    rows = manifest['rows']
    retained = torch.load(out/'initialization.pt',map_location='cpu',weights_only=False)
    _,_,initial_signature = source_model(config,manifest,scales)
    if retained['signature'] != initial_signature:
        raise ValueError('prototype training initialization drift')
    signature = dict(initial_signature,arm=arm)
    job = out/arm
    job.mkdir(exist_ok=True)
    if (job/'summary.json').exists():
        if json.loads((job/'summary.json').read_text())['signature'] != signature:
            raise ValueError('completed prototype continuation drift')
        return
    torch.manual_seed(config['training']['seed'])
    torch.set_num_threads(config['resources']['torch_threads'])
    model = PrototypeSupervisedHbTokenizer(retained['model_config'],codebook_size=128,hard=True)
    model.load_state_dict(retained['model_state'])
    auxiliary = retained['auxiliary_scales']
    frozen = freeze_branches(model,config)
    source_reference = deepcopy(model).eval() if frozen else None
    frozen_state = {key:value.clone() for key,value in model.state_dict().items()
                    if frozen and not key.startswith(('branches.O.','prototype_heads.O.'))}
    weights = (retained['calibrated_auxiliary_weights'] if arm=='auxiliary' and 'calibrated_auxiliary_weights' in retained
               else config['auxiliary'][arm])
    metadata = dict(retained['metadata'],signature=signature,seed=config['training']['seed'],
        auxiliary_weights=weights,frozen_branches=frozen,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'])
    groups = base_tokenizer_sampling_index(rows,'hb')
    cache = OrderedDict()
    rng = np.random.default_rng(config['training']['seed'])
    monitor_examples = base_tokenizer_sample(np.random.default_rng(8031),groups,64)
    monitor = _base_training_batch(rows,monitor_examples,scales,'cpu',cache,config)
    checkpoint = job/'checkpoint.pt'
    start,history = 0,[]
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('prototype continuation identity drift')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        start,history = saved['step'],saved['history']
    began = time.monotonic()
    training = config['training']
    for step in range(start,training['steps']):
        take = base_tokenizer_sample(rng,groups,training['batch_size'])
        batch = _base_training_batch(rows,take,scales,'cpu',cache,config)
        model.train()
        for branch in frozen:
            model.branches[branch].eval()
            model.prototype_heads[branch].eval()
        result = model(batch['x'])
        loss,_ = _state_refinement_loss(result,batch,config['loss'])
        shape,slope = auxiliary_loss(result,batch,auxiliary,config.get('auxiliary_targets'))
        loss = loss+weights['shape']*shape+weights['slope']*slope
        phase = (step+1)/training['steps']
        rate = training['final_learning_rate']+(training['learning_rate']-training['final_learning_rate'])*(1+np.cos(np.pi*phase))/2
        for group in optimizer.param_groups:
            group['lr'] = float(rate)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite prototype continuation')
        optimizer.step()
        if (step+1)%training['checkpoint_every']==0:
            model.eval()
            with torch.inference_mode():
                result = model(monitor['x'])
                value,terms = _state_refinement_loss(result,monitor,config['loss'])
                shape,slope = auxiliary_loss(result,monitor,auxiliary,config.get('auxiliary_targets'))
                replay = model.decode_ids(result['ids'])
                error = max(float((result[key]-replay[key]).abs().max()) for key in ('S','O','S_prototype','O_prototype'))
            frozen_s_error = None
            if frozen:
                for key,expected in frozen_state.items():
                    if not torch.equal(expected,model.state_dict()[key]):
                        raise AssertionError('frozen S/shared state changed')
                with torch.inference_mode():
                    latent = source_reference._encode_branch('S',monitor['x'])
                    quantizer = source_reference.branches['S']['quantizer']
                    ids = quantizer._assignment_logits(latent.reshape(-1,latent.shape[-1])).argmax(-1).reshape(latent.shape[:2])
                    expected = source_reference._decode_component('S',quantizer.get_embedding(ids))
                if not torch.equal(result['ids']['S'],ids):
                    raise AssertionError('frozen S integer sequence changed')
                frozen_s_error = float((result['S']-expected).abs().max())
                if frozen_s_error != 0:
                    raise AssertionError('frozen S waveform changed')
            if error != 0:
                raise AssertionError('prototype or full integer replay differs')
            history.append(dict(step=step+1,loss=float(loss.detach()),monitor_loss=float(value),
                monitor={key:float(val) for key,val in terms.items()},shape_loss=float(shape),
                slope_loss=float(slope),integer_replay_max_abs=error,frozen_s_max_abs=frozen_s_error,seconds=time.monotonic()-began))
            payload = dict(signature=signature,metadata=metadata,model_config=retained['model_config'],
                model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),auxiliary_scales=auxiliary,
                step=step+1,history=history,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            save(checkpoint,payload)
            if step+1 in training['milestones']:
                save(job/f'checkpoint_step_{step+1:05d}.pt',payload)
            write_json(job/'progress.json',dict(status='training',step=step+1,budget=training['steps'],history=history))
    write_json(job/'summary.json',dict(status='completed',signature=signature,history=history,
        validation_arrays_read=0,test_arrays_read=0,seconds=time.monotonic()-began))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','train'],required=True)
    parser.add_argument('--arm',choices=['control','auxiliary'])
    args = parser.parse_args()
    if args.stage=='train' and args.arm is None:
        parser.error('--arm is required for training')
    config = yaml.safe_load(args.config.read_text())
    if config['schema'] not in ('dual_component_joint_hb_prototype_v1','dual_component_joint_hb_o_prototype_v1','dual_component_joint_hb_o_prototype_v2') or config['training']['steps'] < 1:
        raise ValueError('versioned prototype continuation configuration required')
    if config['schema'].startswith('dual_component_joint_hb_o_prototype_v') and (
            config['training'].get('frozen_branches') != ['S'] or config.get('auxiliary_targets') != {'O':[0]}):
        raise ValueError('frozen S and O-HbO-only auxiliary supervision required')
    args.output.mkdir(parents=True,exist_ok=True)
    record = args.output/'resolved_config.json'
    if record.exists() and json.loads(record.read_text()) != config:
        raise ValueError('prototype configuration changed')
    if not record.exists():
        write_json(record,config)
    manifest,scales,_,_ = base_tokenizer_training_data(config,SOURCE)
    lock = args.output if args.stage=='prepare' else args.output/args.arm
    with owner_lock(lock):
        try:
            if args.stage=='prepare':
                prepare(config,args.output,manifest,scales)
            else:
                train(config,args.output,args.arm,manifest,scales)
        except BaseException as error:
            write_json(lock/'failure.json',dict(type=type(error).__name__,reason=str(error)))
            raise


if __name__=='__main__':
    main()
