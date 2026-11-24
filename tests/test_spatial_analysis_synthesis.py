"""Coordinate search uses observed X and preserves the frozen integer decoder."""
from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('analysis_synthesis_runner', ROOT/'experiments/scripts/run_joint_spatial_analysis_synthesis.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_synthetic_search_changes_integers_and_reduces_actual_decoder_proxy():
    result = runner.synthetic()
    assert result['accepted_updates'] > 0
    assert result['proxy_after'] < result['proxy_before']
    assert result['integer_count_per_branch_position'] == 15
    assert result['deterministic_encoder']


def test_observation_source_is_used_for_initialization_and_search_with_decoder_unchanged(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from src.tokenizers.spatial_analysis_synthesis import AnalysisSynthesisSpatialEncoder

    captured = {}
    def remember(source, settings, variances):
        model = AnalysisSynthesisSpatialEncoder(source,settings,variances)
        captured.update(source=source, model=model, settings=settings, variances=variances)
        return model
    monkeypatch.setattr(runner, 'AnalysisSynthesisSpatialEncoder', remember)
    runner.synthetic()
    source = captured['source']
    values = np.stack([np.tile(source.atoms['S',band][2],(15,1)) for band in range(5)])[None]
    seen = np.ones((1,5),bool)
    seen[0,-1]=False
    for multiplier in [1.,0.]:
        def estimate(x, dataset, observed):
            assert dataset=='fixture' and np.array_equal(observed,seen)
            return {'S':x*multiplier,'O':x*(1-multiplier),'sum':x.copy()}
        model=AnalysisSynthesisSpatialEncoder(source,captured['settings'],captured['variances'],component_source=SimpleNamespace(metadata=source.metadata,predict_continuous=estimate))
        codes=model.encode(values,'fixture',seen)
        if multiplier==1.:
            ref=captured['model'].encode(values,'fixture',seen)
            for key in 'SO':
                np.testing.assert_array_equal(codes[key],ref[key])
        else:
            assert np.any(codes['S']!=ref['S'])
        for key in 'SO':
            np.testing.assert_array_equal(codes[key][0,-1],np.zeros(15))
        for key,v in model.decode_ids(codes,'fixture').items():
            np.testing.assert_array_equal(v,source.decode_ids(codes,'fixture')[key])


def test_persisted_calibrated_manifest_requires_frozen_source_identity(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    from types import SimpleNamespace
    import src.tokenizers.spatial_analysis_synthesis as owner
    raw=tmp_path/'atoms.npz';raw.write_bytes(b'fixture')
    source=tmp_path/'component.npz';source.write_bytes(b'changed')
    manifest=tmp_path/'encoder.json'
    manifest.write_text(json.dumps(dict(schema='spatial_analysis_synthesis_transient_v1',source=raw.name,
        source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),component_source=source.name,component_source_sha256='bad')))
    monkeypatch.setattr(owner,'load_model',lambda path: SimpleNamespace(metadata={}))
    with pytest.raises(ValueError,match='frozen component source changed'):
        owner.load_encoder(manifest)
