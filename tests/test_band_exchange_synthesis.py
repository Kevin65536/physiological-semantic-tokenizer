import numpy as np
import pytest

from src.tokenizers.band_exchange_synthesis import (
    FREQUENCY_GROUPS, apply_synthesis, fit_grouped_heads, frequency_indices,
)


def test_complex_cross_spectrum_recovers_foreign_electrode_while_local_control_cannot():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(2000, 4)) + 1j*rng.normal(size=(2000, 4))
    truth = np.zeros((4,4),complex)
    truth[0,1] = .7+.4j
    truth[1,2] = -.3+.8j
    truth[2,3] = .6-.2j
    truth[3,0] = .4+.3j
    y=x@truth.T
    gram=np.tile((x.conj().T@x/len(x))[None,None],(len(FREQUENCY_GROUPS),1,1,1))
    cross=np.tile((x.conj().T@y/len(x)).T[None],(len(FREQUENCY_GROUPS),1,1))
    fitted=fit_grouped_heads(gram,cross,np.zeros(4,int),2,1e-8)
    np.testing.assert_allclose(fitted['montage'][0],truth,atol=2e-8)
    held=rng.normal(size=(100,4))+1j*rng.normal(size=(100,4))
    assert np.mean(abs(held@fitted['montage'][0].T-held@truth.T)**2)<1e-12
    assert np.mean(abs(held@fitted['same_site'][0].T-held@truth.T)**2)>.1


def test_fixed_synthesis_uses_both_streams_preserves_grid_sum_and_declared_support():
    t=np.arange(6000)/200
    s=np.stack((np.sin(2*np.pi*6*t),np.cos(2*np.pi*6*t))).reshape(2,15,400)
    o=.5*s
    coefficient=np.zeros((5,4,4),complex)
    coefficient[1,0,1]=1j
    coefficient[1,2,2]=1
    result=apply_synthesis(dict(S=s,O=o),coefficient)
    np.testing.assert_allclose(result['S'][0].ravel(),-np.sin(2*np.pi*6*t),atol=1e-12)
    np.testing.assert_array_equal(result['sum'],result['S']+result['O'])
    assert result['S'].shape==(2,15,400)
    assert 1350 in frequency_indices(6000)[-1]  # the original base retains exactly45Hz
    with pytest.raises(ValueError):
        apply_synthesis(dict(S=s,O=o[:1]),coefficient)


def test_zero_spectral_energy_never_amplifies_floating_roundoff():
    gram=np.zeros((5,1,4,4),complex)
    cross=np.ones((5,4,4),complex)
    result=fit_grouped_heads(gram,cross,np.zeros(4,int),2)
    assert all(np.array_equal(value,np.zeros_like(value)) for value in result.values())


def test_fit_uses_only_outer_fit_people_and_retains_output_mask_groups(monkeypatch):
    from experiments.scripts import run_band_exchange_synthesis as runner

    rng = np.random.default_rng(107)
    blocks = []
    for i, fold in enumerate((1, 2, 0)):
        tracks = {b: rng.normal(size=(2, 15, 400)) for b in "SO"}
        mask = np.ones((2, 15, 400), bool)
        mask[1, 0] = False
        blocks.append(dict(case=dict(fold=fold, subject=f"s{i}", crop=dict(row=i)),
                           native=np.arange(2), hard=tracks, continuous=tracks,
                           batch=dict(s=tracks["S"].copy(), o=tracks["O"].copy(), mask=mask)))
    monkeypatch.setattr(runner, "STATE", dict(rows=[dict(record="r")] * 3,
                                             cfg=dict(model=dict(ridge_fraction=.05))))
    first, support = runner.fit_heads(blocks, 0)
    blocks[-1]["batch"]["s"][:] = 1e12
    blocks[-1]["batch"]["o"][:] = -1e12
    second, repeated = runner.fit_heads(blocks, 0)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert support == repeated
    assert support["fit_people"] == ["s0", "s1"]
    assert support["support_groups"] == 2
    assert support["partial_target_patches_excluded_from_fit_only"] == 2
