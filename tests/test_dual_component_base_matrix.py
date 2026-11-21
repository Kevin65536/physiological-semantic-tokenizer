from collections import OrderedDict

import numpy as np
import torch

from test_dual_component_base_g1 import ROOT, prepared, runner


def test_target_hard_training_initialization_and_integer_scoring(tmp_path, monkeypatch):
    cfg, out = prepared(tmp_path)
    cfg['architectures']['waveform']['target_input'] = True
    cfg['evaluation']['integer_replay_check'] = 'disabled_by_user'
    manifest, scales, audit, _ = runner._base_training_data(cfg, tmp_path)
    task = dict(modality='hb', architecture='waveform', candidate='vq128')
    model = runner._base_training_model(cfg, task)
    monkeypatch.setattr(model, 'encode_continuous', lambda *args: (_ for _ in ()).throw(AssertionError('X input')))
    examples = audit['hb']['initialization']
    runner._base_initialize(model, manifest['rows'], examples, scales, 'cpu', OrderedDict(), cfg)
    batch = runner._base_training_batch(manifest['rows'], examples, scales, 'cpu', OrderedDict(), cfg)
    model.eval()
    with torch.no_grad():
        pred = runner._base_forward(model, batch)
        alternate = runner._base_forward(model, dict(batch, x=-batch['x'], o=-batch['o']))
        torch.testing.assert_close(pred['ids']['S'], alternate['ids']['S'])
        torch.testing.assert_close(pred['S'], model.decode_ids(pred['ids'])['S'])
    runner.evaluate_base_training(cfg,model,manifest['rows'],scales,'hb','cpu',out,OrderedDict(),limit=8)
    assert np.load(out/'validation.npz')['ids_S'].shape[1] == 15


def test_fixed_matrix_contract():
    cfg = runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_matrix_v1.yaml')
    assert len(cfg['matrix']['controls']) == len(cfg['tasks']) == 4
    assert cfg['training']['total_steps'] == 10000
    assert cfg['architectures']['target8'] == dict(cfg['architectures']['waveform8'],target_input=True)
