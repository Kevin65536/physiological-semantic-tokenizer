from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
import run_single_vq_spectral_location as suite
from src.analysis.dual_component_tokenizer import eeg_spectral_attributes, eeg_spectrum_loss


def test_spectral_contract_and_sine_scaling():
    suite.check()
    t = torch.arange(400, dtype=torch.float64) / 200
    wave = (torch.sin(2*torch.pi*6*t) + 2*torch.sin(2*torch.pi*10*t)).reshape(1, 1, 400)
    value = eeg_spectral_attributes(wave, torch.ones(1, dtype=t.dtype))
    torch.testing.assert_close(value.exp()[0, 0, :2], torch.tensor([.5, 2.], dtype=t.dtype), atol=.001, rtol=.001)
    torch.testing.assert_close(value[0, 0, 2:], torch.tensor([.2, .8], dtype=t.dtype), atol=.001, rtol=.001)
    scaled = eeg_spectral_attributes(wave, torch.tensor([2.], dtype=t.dtype))
    torch.testing.assert_close(scaled[..., :2] - value[..., :2], torch.full_like(value[..., :2], 4.).log())
    torch.testing.assert_close(scaled[..., 2:], value[..., 2:])


@pytest.mark.parametrize('location', ['decoded_spectrum', 'single_id_spectrum'])
def test_attachment_and_incomplete_patch_gradients(location):
    wave = torch.randn(1, 2, 400)
    batch = dict(s=wave.clone(), mask=torch.ones_like(wave, dtype=torch.bool), input_scale=torch.ones(1),
                 feature_mean=torch.zeros(1, 1, 4), feature_sd=torch.ones(1, 1, 4))
    batch['mask'][0, 1, -1] = False
    batch['s'][0, 1, -1] = float('nan')
    decoded = torch.randn_like(wave, requires_grad=True)
    head = torch.randn(1, 2, 4, requires_grad=True)
    eeg_spectrum_loss(dict(S=decoded, S_spectrum=head), batch, location).backward()
    active, inactive = (decoded, head) if location == 'decoded_spectrum' else (head, decoded)
    assert active.grad[0, 0].abs().sum() > 0
    assert not active.grad[0, 1].any() and inactive.grad is None


def test_shared_warmup_new_head_and_fixed_window_scoring(tmp_path):
    import numpy as np
    from test_dual_component_base_g1 import prepared
    from score_single_vq_spectral_location import extract, supported_ids
    from run_dual_component_tokenizer import _base_training_model, train_base_tokenizer
    from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_examples
    cfg, run = prepared(tmp_path)
    cfg['capacity'] = dict(sizes=[8], uniform_count_fraction=.1)
    cfg['evaluation']['steps'] = []
    warm = dict(modality='eeg', architecture='waveform', candidate='warmup')
    train_base_tokenizer(cfg, tmp_path, run, warm, 'cpu')
    data, scales, _, _ = base_tokenizer_training_data(cfg, tmp_path)
    rows = data['rows']
    calibration = {r['dataset']:dict(mean=[0]*4, sd=[1]*4) for r in rows}
    task = dict(warm, candidate='vq8')
    for arm in ('B','C'):
        objective = dict(location=suite.PLAN['arms'][arm], weight=.01, calibration=calibration)
        dest = run/arm
        result = train_base_tokenizer(cfg, tmp_path, run, task, 'cpu', destination=dest, spectral_objective=objective)
        assert result['completed_step']==8 and result['history'][-1]['monitor']['spectral_attributes'] >= 0
        model = _base_training_model(cfg, task, single_id_spectrum=(arm=='C')).eval()
        model.load_state_dict(torch.load(dest/'checkpoint.pt', weights_only=False)['model_state'])
        ex = np.array(base_tokenizer_examples(rows, 'eeg', 'validation')[:2])
        row,start,pos = ex[0]
        ds = rows[row]['dataset']
        sites = {(r['dataset'],p):i for r in rows for i,p in enumerate(r['positions'])}
        with torch.no_grad():
            extract(model, ex, ds, 'validation', rows, scales, calibration, sites,
                    {(row,start,rows[row]['positions'][pos]):'difficult'}, dest)
        with np.load(dest/f'{ds}_validation.npz') as z:
            assert z['id'].shape == (z['valid'].sum(),)
            assert z['nmse'].shape == (len(ex),3,1) and np.isfinite(z['target']).all()
        assert (dest/'difficult_waveforms.npz').is_file()
    ids = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(supported_ids(ids, p, 3, dict(min_train_patches=3,min_train_people=2)),
                                  [True,False,False])


def test_replication_requires_gain_and_each_reconstruction_guardrail(tmp_path):
    from score_single_vq_spectral_location import compare
    from run_dual_component_tokenizer import write_json
    def metric(value):
        return dict(macro=value, by_subject=[dict(dataset='d',subject=str(i),value=value) for i in range(16)])
    def store(arm, gain, nmse, corr):
        metrics = dict(nmse=metric([[v] for v in nmse]), correlation=metric([[v] for v in corr]),
                       conditional_id=metric([1-gain]*4), background=metric([1]*4))
        write_json(tmp_path/'seed_17'/arm/'evaluation/summary.json', dict(metrics=metrics, mean_id_gain=gain,
                   selection_score=sum(nmse)/3, id_gain=[gain]*4, supported_fraction_vocabulary=.2,
                   supported_development_tokens=30))
    store('A', .01, [1,1,1], [.8]*3)
    store('B', .05, [1.055,.96,.96], [.8]*3)
    store('C', .03, [1,1,1], [.8]*3)
    compare_all = lambda: compare(tmp_path, 17, ['A','B','C'], suite.PLAN)
    assert compare_all()['selected'] == 'C'
    store('C', .1, [1,1,1], [.77,.8,.8])
    assert compare_all()['selected'] is None
    store('B', .05, [1,1,1], [.8]*3)
    store('C', .1, [1,1,1], [.8]*3)
    report = compare_all()
    assert report['selected'] == 'B' and len(report['contrasts']['C-minus-B']['subjects']) == 16
