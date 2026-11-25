from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.temporal_projection_so import BLOCK, complete_blocks, project_blocks, rank_frequencies


def test_projection_rank_phase_and_absolute_grid():
    origin = 199
    t = (origin+np.arange(4800))/200
    x = np.stack([np.sin(2*np.pi*10*t+.47),np.cos(2*np.pi*19*t-.19)],axis=1)
    b,s = complete_blocks(x,origin)
    y = project_blocks(b,[20])
    np.testing.assert_allclose(y[:,0],x[s,0],atol=2e-13)
    np.testing.assert_allclose(y[:,1],0,atol=2e-13)
    b2,s2 = complete_blocks(x[133:3900],origin+133)
    k=133+s2.start-s.start
    np.testing.assert_allclose(project_blocks(b2,[20]),y[k:k+len(b2)*BLOCK],atol=1e-13)
    rng=np.random.default_rng(17)
    u=rng.normal(size=b.shape)
    v=project_blocks(u,[8,10,19])
    np.testing.assert_allclose(project_blocks(v.reshape(u.shape),[8,10,19]),v,atol=1e-13)
    assert abs(np.sum(v*(u.reshape(v.shape)-v)))<1e-10
    assert np.linalg.norm(v)<=np.linalg.norm(u)
    for invalid in ([0],[3,3],[201]):
        with pytest.raises(ValueError):
            project_blocks(b,invalid)


def test_nested_spaces_and_contract():
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/temporal_projection_so_v1.yaml').read_text())
    order=rank_frequencies(np.ones(201))
    assert set(order)==set(range(2,90)) and len(order)==88
    assert order[:8]==[2,8,16,26,9,17,27,60]
    assert c['candidate_max_pairs'] in c['pair_counts'] and c['arms']==['v3','projected_v3','projected_x']


def test_partial_blocks_and_context_share_labels_and_masks():
    from experiments.scripts.train_temporal_projection_vq import pack
    rng=np.random.default_rng(17);x=rng.normal(size=(6600,2));old=x/3
    mask=np.ones_like(x,bool);mask[1300,0]=False
    first=pack(x[:6000],old[:6000],mask[:6000],35,[8,10,19])
    next_=pack(x[400:6400],old[400:6400],mask[400:6400],435,[8,10,19])
    joint=first['mask'][:,1:] & next_['mask'][:,:-1]
    assert joint.any() and joint.sum(axis=1)[0]<joint.sum(axis=1)[1]
    for k in ('v3','projected_v3','projected_x'):
        a=first[k].reshape(2,15,400)[:,1:][joint]
        b=next_[k].reshape(2,15,400)[:,:-1][joint]
        np.testing.assert_array_equal(a,b)


def test_spatial_cap_matches_direct_whitened_svd():
    from experiments.scripts.evaluate_temporal_projection_so import physics
    from src.data.dual_component_tokenizer import voltage_power_cap_bank,voltage_power_cap_frames
    rng=np.random.default_rng(17)
    x=rng.normal(size=(2000,2))
    correlation=np.tile([[1.,.2],[.2,1.]],(6,1,1))
    projection=np.eye(10)[:3]
    model=dict(projection=projection,mean=np.zeros(10),lift=projection.T,
               baseline_log_power=np.zeros(10),complement_power=np.ones(2),
               correlation=correlation,proxy_steps=100)
    ratio=physics(x,model,np.zeros((100,3)),[0,1],0)
    frames,t=voltage_power_cap_frames(voltage_power_cap_bank(x))
    z=np.linalg.solve(np.linalg.cholesky(correlation[3]),frames[t==1][0,3])
    np.testing.assert_allclose(ratio[0,3],np.linalg.svd(z,compute_uv=False)[0]**2/200)
