"""Apply the same frozen-member protocol to each checkpoint and compare physical shifts."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scripts import (
    audit_projected_so_members as members,
    projected_so_alpha_specificity as specificity,
)
from experiments.scripts.check_continuous_so_targets import OUT, PRIOR, CFG
from src.data.so_supervision_cache import write_json


def inspect_checkpoint(tag, arm, network=None, step=500, seed=17):
    dest = OUT / 'members' / tag
    config = dict(
        members.CFG,
        experiment_id=CFG['experiment_id'],
        output=str(dest.relative_to(members.ROOT)),
        checkpoint=dict(arm=arm, step=step, seed=seed),
    )
    members.run('cuda:0', output=dest, arm=arm, network=network, config=config)
    specificity.run(dest, config)
    if network is None:
        network = members.model(True, 'cuda:0').eval()
        state = torch.load(members.SOURCE / arm / 'resume.pt', map_location='cpu', weights_only=False)
        network.load_state_dict(state['net'])
    shift_scores(network, dest)
    frame = pd.read_csv(dest / 'validation_members.csv.gz', index_col='row')
    matches = pd.read_csv(dest / 'matches.csv')
    for key, g in matches.groupby(['branch', 'endpoint']):
        j = frame.loc[g.member_row]
        matches.loc[g.index, 'distinct_times'] = len(j.drop_duplicates(['person', 'start', 'block']))
    counts = matches.groupby(['branch', 'endpoint']).distinct_times.first()
    table = pd.read_csv(dest / 'contrasts.csv').join(counts, on=['branch', 'endpoint'])
    table.distinct_times = table.distinct_times.fillna(0).astype(int)
    table.to_csv(dest / 'retrieval.csv', index=False)


@torch.inference_mode()
def shift_scores(net, dest):
    src = members.SOURCE
    meta = pd.read_csv(src / 'examples.csv', dtype={'subject': str})
    x = np.load(src / 'x.npy', mmap_mode='r').reshape(-1, 15, 400)
    mask = np.load(src / 'mask.npy')
    join = ['dataset', 'subject', 'parent', 'site']

    def side(role, start):
        return meta[(meta.role == role) & (meta.start == start)].reset_index().set_index(join)['index']

    indices = pd.concat([side('validation', 0), side('shift', 2)], axis=1, keys=['a', 'b'])
    assert len(indices) == 305 and indices.notna().all().all()
    out = []
    for _, g in indices.groupby(level=['dataset', 'subject']):
        a = g.a.to_numpy()
        b = g.b.to_numpy()
        first = net(torch.as_tensor(x[a], device='cuda:0'))['ids']
        second = net(torch.as_tensor(x[b], device='cuda:0'))['ids']
        valid = mask[a, 1:] & mask[b, :-1]
        np.testing.assert_array_equal(x[a, 1:][valid], x[b, :-1][valid])
        for k in 'SO':
            eq = first[k][:, 1:].cpu().numpy() == second[k][:, :-1].cpu().numpy()
            out.append(
                dict(
                    dataset=g.index[0][0],
                    subject=g.index[0][1],
                    branch=k,
                    pairs=int(valid.sum()),
                    matches=int(eq[valid].sum()),
                )
            )
    pd.DataFrame(out).to_csv(dest / 'shift.csv', index=False)


if __name__ == '__main__':
    arm = sys.argv[1]
    if arm not in ['v3', 'projected_x', 'projected_v3']:
        raise ValueError(arm)
    torch.set_num_threads(1)
    inspect_checkpoint('short_' + arm, arm)
