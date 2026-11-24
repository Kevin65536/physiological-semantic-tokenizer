from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
import run_single_vq_dynamic_factorial as suite


def test_both_continuous_objectives_share_warmup_and_full_budget(tmp_path):
    from test_dual_component_base_g1 import prepared
    from run_dual_component_tokenizer import train_base_tokenizer
    from src.data.dual_component_tokenizer import base_tokenizer_training_data

    suite.check()
    cfg, root = prepared(tmp_path)
    cfg['evaluation']['steps'] = []
    task = suite.TASK
    train_base_tokenizer(cfg, tmp_path, root, dict(task, candidate='warmup'), 'cpu')
    manifest, *_ = base_tokenizer_training_data(cfg, tmp_path)
    calibration = {r['dataset']: dict(mean=[0] * 4, sd=[1] * 4) for r in manifest['rows']}
    states = []
    for arm in suite.ARMS:
        obj = (
            dict(location='decoded_spectrum', weight=0.01, calibration=calibration)
            if arm.endswith('dynamic')
            else None
        )
        result = train_base_tokenizer(
            cfg, tmp_path, root, task, 'cpu', destination=root / arm, spectral_objective=obj
        )
        assert result['completed_step'] == 8
        assert ('spectral_attributes' in result['history'][-1]['monitor']) == (obj is not None)
        states.append(torch.load(root / arm / 'checkpoint.pt', weights_only=False))
    assert states[0]['numpy_rng'] == states[1]['numpy_rng']
    assert any(not torch.equal(v, states[1]['model_state'][k]) for k, v in states[0]['model_state'].items())


def test_contrast_pairs_people_and_gives_each_dataset_equal_weight():
    import numpy as np
    from score_single_vq_dynamic_factorial import CONTRASTS, paired_means

    # Three people from a large dataset and one from a second. Shared person effects cancel.
    people = np.array([1.0, 2.0, 3.0, 4.0])
    data = np.repeat(people[:, None], 4, axis=1)
    data[:, 1] -= [2, 2, 2, 4]
    data[:, 3] -= [1, 1, 1, 2]
    point, draws = paired_means(data, np.array(['a', 'a', 'a', 'b']), draws=64)
    weights = CONTRASTS['hard_minus_continuous_benefit']
    np.testing.assert_allclose(point @ weights, -1.5)
    np.testing.assert_allclose(draws @ weights, -1.5)
