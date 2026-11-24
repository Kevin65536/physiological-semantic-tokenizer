"""Launch the retained research interfaces with newly fitted weights on QC-filtered public G0."""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CFG = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_full_qc_retrain_v1.yaml').read_text())
OUT = ROOT/CFG['run']
OLD = ROOT/CFG['historical_run']
CONFIGS = ROOT/'experiments/configs/physiology_semantic_tokenizer'


def read_cfg(name):
    return yaml.safe_load((CONFIGS/name).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def configuration():
    cfg = yaml.safe_load((ROOT/CFG['g1']).read_text())
    subjects = json.loads((OLD/'development_audit/as_local_gain_v1/spatial/summary.json').read_text())['metrics']['nmse']['by_subject']
    admitted = sorted([v['dataset'], v['subject']] for v in subjects)
    reserved = json.loads((OLD/'panel.json').read_text())['confirmation_subjects']
    if len(admitted) != 16 or any(reserved[d] == s for d,s in admitted):
        raise ValueError('ordinary development identity drift')
    cfg['data']['validation_subjects'] = admitted
    cfg['experiment_id'] = CFG['experiment_id']+'-Hb-G1'
    cfg['evaluation']['steps'] = [10000]
    cfg['evaluation']['roles'] = ['train','validation']
    cfg['resources']['gpu_memory_limit_gib'] = 4
    return cfg


def hb_base():
    from run_dual_component_tokenizer import train_base_tokenizer
    cfg = configuration()
    folder = OUT/'hb'
    folder.mkdir(parents=True,exist_ok=True)
    config_path = folder/'base.yaml'
    text = yaml.safe_dump(cfg,sort_keys=False)
    if config_path.exists() and config_path.read_text()!=text:
        raise ValueError('Hb base contract changed')
    config_path.write_text(text)
    for candidate in ['warmup','vq128']:
        task = dict(modality='hb',architecture='waveform',candidate=candidate)
        train_base_tokenizer(cfg,ROOT,folder,task,os.environ.get('HB_DEVICE','cuda:0'))



def hb_final():
    from collections import OrderedDict
    from types import SimpleNamespace
    import torch
    from run_dual_component_tokenizer import identity
    import run_joint_hb_prototype_tokenizer as proto
    from src.data.dual_component_tokenizer import base_tokenizer_examples, base_tokenizer_batch, base_tokenizer_training_data
    from src.tokenizers.contextual_latent_dictionary import load_model as load_prototype
    from src.tokenizers.silent_observation_tokenizer import load_model as load_silent
    from src.tokenizers.hb_native_id_readout import fit_head, validate_ids, smooth_basis, load_model as load_final
    from src.utils.io import file_sha256
    from run_hb_silent_atom import amplitudes, fit_models, training_weights
    from calibrate_joint_hb_output import transfer_fit, project_training_energy, export_transfer_arms

    cfg=configuration()
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    rows=manifest['rows']; folder=OUT/'hb'
    torch.set_num_threads(4)
    source=dict(config=str(folder/'base.yaml'),checkpoint=str(folder/'models/hb__waveform__vq128/checkpoint_step_10000.pt'),
                step=10000,validation_exposure='ordinary_development_16_only')
    for name,sub in [('dual_component_joint_hb_prototype_v1.yaml','prototype_init'),
                     ('dual_component_joint_hb_o_prototype_v2.yaml','prototype_256')]:
        c=read_cfg(name)
        c.update(data=cfg['data'],source=source)
        dest=folder/sub
        dest.mkdir(exist_ok=True)
        if sub=='prototype_256':
            c['initialization']['checkpoint']=str(folder/'prototype_init/initialization.pt')
        dump(dest/'resolved_config.json',c)
        proto.prepare(c,dest,manifest,scales)
        if sub=='prototype_256':
            proto.train(c,dest,'auxiliary',manifest,scales)
    checkpoint=folder/'prototype_256/auxiliary/checkpoint_step_02000.pt'
    model=load_prototype(checkpoint)
    quiet=folder/'quiet'
    quiet.mkdir(exist_ok=True)
    c=read_cfg('dual_component_hb_silent_atom_v1.yaml')
    train=np.asarray(base_tokenizer_examples(rows,'hb','train'),np.int32)
    assert all(rows[i]['split']=='train' for i in np.unique(train[:,0]))
    amp=amplitudes(rows,train,scales,quiet,'train',c)
    g0=ROOT/cfg['data']['run']
    signature=dict(source_sha256=file_sha256(checkpoint),manifest_sha256=file_sha256(g0/'cache/manifest.json'),
                   scales_sha256=file_sha256(g0/'scales.json'),filtered_manifest=identity(manifest))
    fitted_quiet=fit_models(c,quiet,rows,scales,train,amp,model,signature)
    gain=folder/'gain';energy=folder/'energy'
    for name,target in [('joint_hb_component_transfer_gain_v1.yaml',gain),('joint_hb_component_transfer_gain_v2.yaml',energy)]:
        c=read_cfg(name)
        c.update(data=cfg['data'],source_checkpoint=str(checkpoint),quiet_training_fit=str(quiet/'fit.json'))
        target.mkdir(exist_ok=True)
        if (target/'summary.json').exists():continue
        args=SimpleNamespace(output=target,synthetic=False)
        if target==gain:
            transfer_fit(args,c)
        elif json.loads((gain/'summary.json').read_text())['status']=='completed':
            c['reuse_training_fit']=str(gain)
            project_training_energy(args,c)
        else:
            # Preserve the infeasible grid. An identity transfer gives a replayable
            # uncalibrated control while the final integer head can still be fitted.
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            signature=dict(input=identity(manifest),source_sha256=file_sha256(checkpoint),quiet_training_fit_sha256=file_sha256(quiet/'fit.json'))
            export_transfer_arms(target,saved,signature,checkpoint,fitted_quiet,quiet/'fit.json',
                                 {'HbO_only':[1.,1.]},'identity_control_after_infeasible_training_grid')
            dump(target/'summary.json',dict(status='training_constraints_infeasible_identity_control',gain=[1.,1.]))
    model_dir=energy/'HbO_only'
    silent_path=model_dir/'training_tail.pt'
    source_model=load_silent(silent_path)
    # Recompute the original train-person design on the eligible rows only.
    # The historical panel can supply row identities, never IDs or predictions.
    legal=set(map(tuple,train))
    with np.load(OLD/'panels.npz') as panel:
        examples=np.asarray([ex for ex in panel['hb_pattern_train'] if tuple(ex) in legal],int)
    if len({(rows[i]['dataset'],rows[i]['subject']) for i in examples[:,0]})!=60:
        raise ValueError('missing training subject in Hb readout panel')
    cache=OrderedDict();parts={k:[] for k in ['si','oi','S','O','sum','complete','response','variance']}
    for offset in range(0,len(examples),128):
        take=examples[offset:offset+128]
        batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=128)
        ids=source_model.encode(batch['x'])
        pred=source_model.decode_ids(ids)
        residual=(batch['x']-pred['sum']).reshape(-1,15,20,2)[:,:,:,0]
        for k,v in dict(si=ids['S'],oi=ids['O'],S=pred['S'],O=pred['O'],sum=pred['sum'],
                        complete=batch['mask'].reshape(-1,15,20,2).all((2,3)),
                        response=np.einsum('npt,tk->npk',residual,smooth_basis())/20,
                        variance=batch['vx'][:,0,0]).items():
            parts[k].append(v)
    bank={k:np.concatenate(v) for k,v in parts.items()}
    ids=dict(S=bank['si'],O=bank['oi']);_,quiet_windows=validate_ids(ids)
    ok=bank['complete']&~quiet_windows[:,None]
    raw=training_weights(examples,rows)[:,None]*ok/15
    weights=raw*(ok.sum()/raw.sum())/bank['variance'][:,None]
    prediction={b:bank[b] for b in ('S','O','sum')}
    head=fit_head(ids,prediction,bank['response'],weights,bank['complete'],
                  np.array([rows[i]['subject'] for i in examples[:,0]]),'decoded_waveform',512,{})
    dest=model_dir/'native_readout.pt'
    proto.save(dest,dict(metadata=dict(schema='hb_native_id_readout_v1',source_file=silent_path.name,
                                      source_sha256=file_sha256(silent_path)),fitted=head))
    replay=load_final(dest)
    x=base_tokenizer_batch(rows,examples[:8],scales)['x']
    words=replay.encode(x)
    for b in 'SO':
        np.testing.assert_array_equal(words[b],ids[b][:8])
    output=replay.decode_ids(words)
    assert head['scalar_coefficients']==111 and all(np.isfinite(output[b]).all() for b in ('S','O','sum'))
    dump(folder/'fit_complete.json',dict(status='completed',model=str(dest),head_parameters=111,
          readout_training_positions=len(examples),quiet_transfer=json.loads((energy/'summary.json').read_text())['status']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['hb-base','hb-final'])
    args=parser.parse_args()
    (hb_base if args.stage=='hb-base' else hb_final)()
