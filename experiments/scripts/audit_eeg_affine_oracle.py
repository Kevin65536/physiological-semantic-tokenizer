"""Target-assisted lower bound for windowwise amplitude/offset changes to frozen EEG waveforms."""
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data


def lower_bound(local_variance, denominator, correlation):
    # Undefined r can include numerical degeneracy; zero is always an optimistic bound.
    recognized = np.isfinite(correlation)
    return np.where(recognized, local_variance / denominator * np.maximum(0., 1-correlation**2), 0.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    r=args.run
    cfg=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _=base_tokenizer_training_data(cfg,ROOT)
    rows=manifest['rows']
    with np.load(r/'panels.npz') as panels:
        examples=panels['eeg_screen']
    if any(rows[i]['split']!='validation' or rows[i]['modality']!='eeg' for i,_,_ in examples):
        raise ValueError('invalid screen support')
    local_variance=[]; denominator=[]; cache=OrderedDict()
    for off in range(0,len(examples),64):
        batch=base_tokenizer_batch(rows,examples[off:off+64],scales,array_cache=cache,cache_limit=32)
        mask=batch['mask'].reshape(-1,6000)
        valid=mask.sum(1)
        if not valid.all():raise ValueError('window without valid observations')
        var=[]; den=[]
        for key in ('x','s','o'):
            y=batch[key].reshape(-1,6000).astype(float)
            mu=np.where(mask,y,0.).sum(1)/valid
            var.append(np.where(mask,(y-mu[:,None])**2,0.).sum(1)/valid)
            den.append(batch['v'+key][:,0,0].astype('float32'))
        local_variance.append(np.stack(var,axis=1)[:,:,None])
        denominator.append(np.stack(den,axis=1)[:,:,None])
    var=np.concatenate(local_variance);den=np.concatenate(denominator)
    records={}
    paths={'phase_projection':'spatial/projection_v1/spatial_metrics.npz',
           'phase_AS_energy':'spatial_analysis_synthesis/gain_v1/energy_constrained/spatial_metrics.npz',
           'semantic_phase_k512':'spatial/joint_phase_semantic_70_v1/spatial_metrics.npz',
           'semantic_phase_k2048':'spatial/joint_phase_semantic_70_k2048_v1/spatial_metrics.npz',
           'semantic_AS_k2048':'spatial_analysis_synthesis/joint70_k2048_v1/spatial_metrics.npz',
           'montage_k2048_step1500':'montage_joint_v1/global_analysis_k2048_v2/joint/screen_step_1500/spatial_metrics.npz',
           'montage_k2048_step6000':'montage_joint_v1/global_analysis_k2048_v2/joint/screen_step_6000/spatial_metrics.npz'}
    for name,path in paths.items():
        with np.load(r/path) as data:
            np.testing.assert_array_equal(examples,data['examples'])
            corr=data['correlation']; floor=lower_bound(var,den,corr);actual=data['nmse']
            if np.any(floor>actual+5e-6):raise AssertionError('oracle exceeds original MSE')
        means=base_training_aggregate({'nmse':floor,'original':actual},examples,rows)['metrics']
        records[name]=dict(source=path,lower_bound=means['nmse'],original=means['original'],
                           undefined_correlations=int((~np.isfinite(corr)).sum()))
    args.output.write_text(json.dumps(dict(status='completed',role='frozen_eeg_screen',planned=len(examples),
        endpoint='Var_window(target)/G0_frozen_train_variance times (1-r_squared); arbitrary gain and offset per window/component, using target. X fitted separately as a generous diagnostic.',
        hierarchy='G0 position/crop/record/subject/dataset',records=records),indent=2)+'\n')

if __name__=='__main__':
    main()
