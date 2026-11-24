#!/usr/bin/env python3
"""Matched decoder refits with the G4 observation encoder and integer vocabulary frozen."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from run_dual_component_tokenizer import (
    SOURCE, _base_training_batch, _state_refinement_loss, identity, write_json, owner_lock,
)
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.data.dual_component_tokenizer import base_tokenizer_sampling_index, base_tokenizer_sample
from src.tokenizers.contextual_latent_dictionary import UnrestrictedHbContextDecoder
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer


def save(path, payload):
    temporary = path.with_suffix('.pending.pt')
    torch.save(payload, temporary)
    temporary.replace(path)


def continuity_loss(result, batch):
    boundary = torch.arange(1,15)*20
    mask = batch['mask'].reshape(len(batch['x']),300,2)
    valid = mask[:,boundary]&mask[:,boundary-1]
    values = []
    for branch, target in [('S','s'),('O','o'),('sum','x')]:
        error = (result[branch]-batch[target]).reshape(len(mask),300,2)
        jump = error[:,boundary]-error[:,boundary-1]
        normalized = jump.square()/batch['dv'+target][:,None,:]
        score = torch.where(valid,normalized,0.).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)
        values.append(score.mean())
    return torch.stack(values).mean()


def prediction(model, values):
    with torch.no_grad():
        latent = model.encode_continuous(values)
        ids = {key: model.branches[key]['quantizer']._assignment_logits(value.reshape(-1,value.shape[-1])).argmax(-1).reshape(value.shape[:2])
               for key,value in latent.items()}
    result = model.decode_ids(ids)
    return dict(result,ids=ids,commitment=values.new_zeros(()))


def run(args, config):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    resolved = out/'resolved_config.json'
    if resolved.exists() and json.loads(resolved.read_text()) != config:
        raise ValueError('decoder refit configuration changed')
    write_json(resolved,config)
    manifest,scales,_,_ = base_tokenizer_training_data(config,SOURCE)
    rows = manifest['rows']
    path = SOURCE/config['source']['checkpoint']
    source = torch.load(path,map_location='cpu',weights_only=False)
    source_cfg = yaml.safe_load((SOURCE/config['source']['config']).read_text())
    if (source['signature']['input'] != identity(manifest) or source['signature']['scales'] != identity(scales)
            or source['signature']['config'] != identity(source_cfg) or source['step'] != config['source']['step']
            or source['metadata']['candidate'] != 'vq1024' or not source['model_config']['hb_local_shape']):
        raise ValueError('fixed G4 source or G0 identity drift')
    signature = dict(config=identity(config),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input=identity(manifest),scales=identity(scales),arm=args.arm)
    if (out/'summary.json').exists():
        if json.loads((out/'summary.json').read_text())['signature'] != signature:
            raise ValueError('completed decoder refit identity changed')
        return
    torch.manual_seed(config['training']['seed'])
    torch.set_num_threads(config['resources']['torch_threads'])
    model_class = GlobalLatentComponentTokenizer if args.arm == 'bounded' else UnrestrictedHbContextDecoder
    model = model_class(source['model_config'],codebook_size=1024,hard=True).eval()
    model.load_state_dict(source['model_state'])
    frozen = {}
    for branch_name,branch in model.branches.items():
        for name in ('input','encoder','to_latent','quantizer'):
            branch[name].requires_grad_(False)
            for key,value in branch[name].state_dict().items():
                frozen[(branch_name,name,key)] = value.clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    metadata = dict(source['metadata'],signature=signature,decoder_architecture=(
        'bounded_hb_context_v1' if args.arm=='bounded' else 'unrestricted_hb_context_v1'),
        frozen_indices_source=str(path),frozen_indices_source_step=source['step'],
        source_sha256=signature['source_sha256'],seed=config['training']['seed'],
        parameter_count_inference=sum(p.numel() for p in model.parameters()),
        trainable_decoder_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    groups = base_tokenizer_sampling_index(rows,'hb')
    cache = OrderedDict()
    rng = np.random.default_rng(config['training']['seed'])
    monitor_examples = base_tokenizer_sample(np.random.default_rng(8031),groups,64)
    monitor = _base_training_batch(rows,monitor_examples,scales,'cpu',cache,config)
    checkpoint = out/'checkpoint.pt'
    start,history = 0,[]
    if checkpoint.exists():
        retained = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if retained['signature'] != signature:
            raise ValueError('decoder continuation identity changed')
        model.load_state_dict(retained['model_state'])
        optimizer.load_state_dict(retained['optimizer_state'])
        rng.bit_generator.state = retained['numpy_rng']
        start,history = retained['step'],retained['history']
    began = time.monotonic()
    training = config['training']
    for step in range(start,training['steps']):
        take = base_tokenizer_sample(rng,groups,training['batch_size'])
        batch = _base_training_batch(rows,take,scales,'cpu',cache,config)
        model.eval()
        result = prediction(model,batch['x'])
        loss,terms = _state_refinement_loss(result,batch,config['loss'])
        boundary = continuity_loss(result,batch)
        loss = loss+training['continuity_weight']*boundary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite frozen-index decoder training')
        optimizer.step()
        if (step+1)%training['checkpoint_every']==0:
            with torch.inference_mode():
                result = prediction(model,monitor['x'])
                monitored,values = _state_refinement_loss(result,monitor,config['loss'])
                jump = continuity_loss(result,monitor)
                replay = model.decode_ids(result['ids'])
                replay_error = max(float((replay[key]-result[key]).abs().max()) for key in ('S','O'))
            if replay_error != 0:
                raise AssertionError('frozen integer replay differs')
            history.append(dict(step=step+1,loss=float(loss.detach()),monitor_loss=float(monitored),
                monitor={key:float(value) for key,value in values.items()},boundary_loss=float(jump),
                replay_max_abs=replay_error,seconds=time.monotonic()-began))
            payload = dict(signature=signature,metadata=metadata,model_config=source['model_config'],
                model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),step=step+1,
                numpy_rng=rng.bit_generator.state,history=history)
            save(checkpoint,payload)
            if step+1 in training['milestones']:
                save(out/f'checkpoint_step_{step+1:05d}.pt',payload)
            write_json(out/'progress.json',dict(status='training',step=step+1,budget=training['steps'],history=history))
    for (branch,name,key),value in frozen.items():
        if not torch.equal(value,model.branches[branch][name].state_dict()[key]):
            raise AssertionError('frozen integer assignment state changed')
    save(out/'model.pt',dict(signature=signature,metadata=metadata,model_config=source['model_config'],
        model_state=model.state_dict(),step=training['steps']))
    write_json(out/'summary.json',dict(status='completed',signature=signature,history=history,
        frozen_assignment_state='exact_equal_to_source',validation_arrays_read=0,test_arrays_read=0,
        seconds=time.monotonic()-began))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=['bounded','unrestricted'],required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    with owner_lock(args.output):
        try:
            run(args,cfg)
        except BaseException as error:
            write_json(args.output/'failure.json',dict(type=type(error).__name__,reason=str(error)))
            raise


if __name__=='__main__':
    main()
