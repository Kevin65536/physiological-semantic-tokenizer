"""Independent equations and support checks for the opt-in source correction."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sta_net_pytorch.source_aligned_v2 import SourceBatchNorm3d, SourceEgta, source_flatten, source_pearson

adapter = importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_batchnorm_matches_documented_keras_population_update_and_inference():
    x = np.arange(2 * 3 * 2 * 2 * 3, dtype=np.float32).reshape(2, 3, 2, 2, 3) / 7
    layer = SourceBatchNorm3d(3)
    output = layer(torch.tensor(x)).detach().numpy()
    mean, variance = x.mean((0, 2, 3, 4)), x.var((0, 2, 3, 4))
    expected = (x - mean[None, :, None, None, None]) / np.sqrt(variance[None, :, None, None, None] + .001)
    np.testing.assert_allclose(output, expected, atol=1e-6)
    np.testing.assert_allclose(layer.running_mean, mean * .01, atol=1e-8)
    np.testing.assert_allclose(layer.running_var, .99 + variance * .01, atol=1e-7)
    layer.eval()
    expected = (x - (mean * .01)[None, :, None, None, None]) / np.sqrt((.99 + variance * .01)[None, :, None, None, None] + .001)
    np.testing.assert_allclose(layer(torch.tensor(x)).detach().numpy(), expected, atol=2e-6)


def test_pearson_additive_epsilon_matches_source_and_constant_gradients_are_finite():
    x = torch.tensor([[0., 1e-4, 2e-4], [1., 1., 1.]], requires_grad=True)
    y = torch.tensor([[0., 2e-4, 3e-4], [0., 0., 0.]], requires_grad=True)
    a, b = x.detach().numpy(), y.detach().numpy()
    expected = np.abs(((a-a.mean(1, keepdims=True))*(b-b.mean(1, keepdims=True))).mean(1) / (a.std(1)*b.std(1)+1e-6)).mean()
    value = source_pearson(x, y)
    np.testing.assert_allclose(value.detach(), expected, atol=1e-8)
    value.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


def test_positional_embedding_is_in_unprojected_source_channels_last_coordinates():
    layer = SourceEgta(8, 4, 2, 2, 0., 3)
    assert layer.position.shape == (1, 3, 8)
    eeg = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2, 1)
    fnirs = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 2, 2, 1)
    expected = np.transpose(fnirs.numpy(), (0, 1, 3, 4, 5, 2)).reshape(2, 3, 8)
    np.testing.assert_array_equal(source_flatten(fnirs), expected)
    captured = []
    hook = layer.key_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    layer(eeg, fnirs); hook.remove()
    np.testing.assert_allclose(captured[0], expected + layer.position.detach().numpy())
    assert layer.position.detach().abs().max() <= np.sqrt(6 / 3)


def profile(task="visual"):
    return dict(input_window_schema="downstream_input_v1", duration_s=30, window_start_s=-5,
        sampling_rate_hz=10, fnirs_sampling_rate_hz=2, num_classes=4,
        eeg_geometry=[[-1,-1],[-1,1],[1,-1],[1,1]],
        fnirs_geometry=[[-.8,-1],[-.8,1],[.8,-1],[.8,1]],
        embedding_dim=8, attention_heads=2, attention_key_dim=4, dropout=0.,
        sta_implementation_version="source_aligned_v2", sta_input_support="event_3s_eeg_13s_hb")


def test_event_support_excludes_surrounding_samples_and_keeps_native_shapes():
    model = adapter.create_model("visual", profile(), "cpu", False)
    eeg, hb = torch.randn(2,4,300), torch.randn(2,8,60)
    first = model(eeg,hb)
    eeg[:,:,:50] = 1000; eeg[:,:,80:] = -1000
    hb[:,:,:10] = 1000; hb[:,:,36:] = -1000
    torch.testing.assert_close(first, model(eeg,hb), rtol=0, atol=0)
    assert len(model.segment_starts) == 11
    assert model.spec["encoded_eeg_seconds"] == 3 and model.spec["encoded_fnirs_seconds"] == 13
    torch.testing.assert_close(first.exp().sum(1), torch.ones(2))


def test_binary_refed_source_model_is_trained_for_two_classes_with_tail_masks():
    p = {**profile(), "window_start_s":0, "num_classes":2, "sta_input_support":"full_admitted_window",
         "classification_adaptation":"refed_arousal_window_mean_midpoint_v1"}
    model = adapter.create_model("refed_arousal_binary", p, "cpu")
    eeg, hb = torch.randn(2,4,300), torch.randn(2,8,60)
    em, hm = torch.ones_like(eeg,dtype=torch.bool), torch.ones_like(hb,dtype=torch.bool)
    em[0,:,-40:] = False; hm[0,:,-8:] = False
    eeg[~em] = float("nan"); hb[~hm] = float("nan")
    losses = model.supervised_loss(eeg,hb,torch.tensor([0,1]),eeg_mask=em,fnirs_mask=hm)
    losses["total"].backward()
    assert model.backbone.fusion_head.out_features == 2
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError,match="explicit"):
        adapter.create_model("refed_arousal_binary", {**p,"classification_adaptation":"wrong"}, "cpu")


def test_source_model_learns_a_small_known_classification_signal():
    torch.manual_seed(32)
    p = {**profile(), "num_classes":2}
    model = adapter.create_model("motor_imagery",p,"cpu")
    y = torch.arange(8) % 2
    eeg, hb = torch.randn(8,4,300)*.1, torch.randn(8,8,60)*.1
    eeg[:,0,50:80] += (2*y-1)[:,None]*2
    hb[:,0,10:36] += (2*y-1)[:,None]*2
    optimizer = torch.optim.Adam(model.parameters(),lr=.005)
    losses=[]
    for _ in range(40):
        optimizer.zero_grad(); loss=model.supervised_loss(eeg,hb,y)["total"]
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * .6
    assert (model(eeg,hb).argmax(1) == y).float().mean() >= .875
