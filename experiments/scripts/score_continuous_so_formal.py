"""Score retained and corrected final-C on the identical formal development support."""

from collections import OrderedDict, defaultdict
from pathlib import Path
import argparse
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_continuous_so_formal import formal, overlay, OUT, CFG, ROOT, read_json, write_json
from compare_continuous_so_members import inspect_checkpoint
from run_dual_component_tokenizer import _base_training_model, identity, owner_lock
from src.analysis.dual_component_tokenizer import base_training_metrics, base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_batch
from evaluate_joint_tokenizer import input_scales


def network(arm, seed):
    cfg = formal.training_cfg(seed)
    if arm == 'v3':
        folder = formal.folder(seed, 'C')
        calibration = formal.OUT / 'calibration.json'
    else:
        folder = OUT / f'seed_{seed}' / 'C'
        cfg['experiment_id'] = CFG['experiment_id']
        calibration = OUT / 'calibration.json'
    st = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    summary = read_json(folder / 'summary.json')
    assert summary['status'] == 'completed' and st['signature'] == summary['signature']
    assert st['step'] == summary['completed_step'] == 10000 and st['signature']['config'] == identity(cfg)
    obj = dict(
        location='single_id_spectrum', weight=formal.PLAN['weight'], calibration=read_json(calibration)
    )
    assert st['signature']['spectral_objective'] == identity(obj) and st['signature']['task'] == formal.TASK
    model = _base_training_model(cfg, formal.TASK, single_id_spectrum=True)
    model.load_state_dict(st['model_state'])
    return model.eval().to('cuda:0')


@torch.inference_mode()
def score(arm, seed):
    tag = 'formal_' + arm + '_' + str(seed)
    dest = OUT / 'evaluation' / tag
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'summary.json').exists():
        return read_json(dest / 'summary.json')
    model = network(arm, seed)
    if not (dest / 'metrics.json').exists():
        formal.capacity.prepare_panel(recorded_root=formal.PLAN['recorded_root'])
        panel = formal.capacity.PANEL
        rows, scales = panel['rows'], panel['scales']
        if arm != 'v3':
            manifest, scales, _, _ = overlay()
            rows = manifest['rows']
        cache = OrderedDict()
        all_ex = []
        all_metrics = defaultdict(list)
        patches = 0
        fixed = read_json(formal.capacity.OUT / 'review_20261124_v1/tables/wave_eeg.json')
        fixed_lookup = {
            (e['row'], e['start_s'], e['position']): name
            for name, e in zip(['median', 'difficult'], fixed, strict=True)
        }
        for ds in formal.capacity.DATASETS:
            ex = panel['examples']['eeg', 'validation', ds]
            with np.load(formal.folder(seed, 'C') / 'evaluation' / f'{ds}_validation.npz') as ref:
                np.testing.assert_array_equal(ref['examples'], ex)
                valid = ref['valid']
                baseline_nmse = ref['nmse']
            for a in range(0, len(ex), 64):
                batch = ex[a : a + 64]
                raw = base_tokenizer_batch(rows, batch, scales, array_cache=cache)
                np.testing.assert_array_equal(raw['mask'].all(-1), valid[a : a + len(batch)])
                patches += int(raw['mask'].all(-1).sum())
                result = model(torch.as_tensor(raw['x'], device='cuda:0'))
                pred = model.decode_ids(result['ids'])
                for k in ['sum', 'S', 'O']:
                    torch.testing.assert_close(pred[k], result[k], rtol=0, atol=0)
                pred = {k: v.cpu().numpy() for k, v in pred.items() if k in ['sum', 'S', 'O']}
                met = base_training_metrics(pred, raw, 1)
                if arm == 'v3':
                    np.testing.assert_allclose(
                        met['nmse'], baseline_nmse[a : a + len(batch)], rtol=1e-5, atol=1e-6
                    )
                for k in ['nmse', 'correlation', 'rms_gain', 'rmse']:
                    all_metrics[k].append(met[k])
                native = input_scales(rows, batch, scales)
                all_metrics['rmse_native'].append(met['rmse'] * native[:, None, None])
                all_ex.extend(batch)
                for j, (r, start, site) in enumerate(batch):
                    key = int(r), int(start), rows[r]['positions'][site]
                    if key in fixed_lookup:
                        np.savez_compressed(
                            dest / (fixed_lookup[key] + '.npz'),
                            example=batch[j],
                            site=key[2],
                            dataset=ds,
                            subject=rows[r]['subject'],
                            mask=raw['mask'][j],
                            scale=native[j],
                            **{k: raw[k][j] for k in 'xso'},
                            **{'decoded_' + k: v[j] for k, v in pred.items()},
                        )
        assert len(all_ex) == 53476 and all(
            (dest / (name + '.npz')).is_file() for name in fixed_lookup.values()
        )
        agg = base_training_aggregate({k: np.concatenate(v) for k, v in all_metrics.items()}, all_ex, rows)
        assert agg['subjects'] == 16
        agg['metrics']['rmse_native'].pop('macro')  # REFED has a separate release unit
        write_json(
            dest / 'metrics.json', dict(status='completed', seed=seed, arm=arm, valid_patches=patches, **agg)
        )
    inspect_checkpoint(tag, arm, network=model, step=10000, seed=seed)
    write_json(dest / 'summary.json', dict(status='completed', arm=arm, seed=seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['v3', 'continuous'])
    parser.add_argument('seed', type=int, choices=[17, 43])
    args = parser.parse_args()
    torch.set_num_threads(1)
    with owner_lock(OUT, f'score_{args.arm}_{args.seed}'):
        score(args.arm, args.seed)
