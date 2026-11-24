import numpy as np
import pytest
from src.tokenizers.forward_gain_dictionary import projection_gains,decode_scaled_atoms
from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary


def fixture():
    rng=np.random.default_rng(8)
    meta=dict(codebook_size=4,positions={'fixture':list('abcde')},output_projection_hz=[1,45])
    atoms={(b,k):np.r_[np.zeros((1,400)),rng.normal(size=(3,400))] for b in 'SO' for k in range(5)}
    weights={('fixture',b,k):rng.normal(size=(1,5)) for b in 'SO' for k in range(5)}
    return SpatialWaveformDictionary(meta,{}, {},atoms,weights)


def test_gain1_exact_endpoint_and_linearity_with_missing():
    model=fixture();ids={b:np.ones((1,5,15),int) for b in 'SO'};ids['S'][:,0]=0
    gains={b:np.ones_like(v,float) for b,v in ids.items()}
    for b,v in model.decode_ids(ids,'fixture').items():np.testing.assert_array_equal(v,decode_scaled_atoms(model,ids,gains,'fixture')[b])
    twice=decode_scaled_atoms(model,ids,{b:2*v for b,v in gains.items()},'fixture')
    for b,v in model.decode_ids(ids,'fixture').items():np.testing.assert_array_equal(2*v,twice[b])


def test_projection_uses_current_local_source_and_unconstrained_oracle_is_distinct():
    model=fixture();ids={b:np.ones((1,5,15),int) for b in 'SO'}
    source={b:np.stack([model.atoms[b,k][1] for k in range(5)])[None,:,None].repeat(15,2)*3 for b in 'SO'}
    gains=projection_gains(source,ids,model,'fixture')
    for g in gains.values():np.testing.assert_allclose(g,3)
    negative={b:-v for b,v in source.items()}
    for g in projection_gains(negative,ids,model,'fixture').values():np.testing.assert_array_equal(g,0)
    for g in projection_gains(negative,ids,model,'fixture',nonnegative=False).values():np.testing.assert_allclose(g,-3)
    with pytest.raises(ValueError):decode_scaled_atoms(model,{b:v.astype(float) for b,v in ids.items()},gains,'fixture')


def test_contract_rejects_other_splits_before_reads():
    from experiments.scripts.audit_forward_atomic_gain import validate
    import yaml
    from pathlib import Path
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_forward_gain_v1.yaml').read_text());validate(cfg)
    cfg['data']['allowed_splits'].append('validation')
    with pytest.raises(ValueError):validate(cfg)
