"""TF2.10 fused variance and train-only layerwise statistics, no data assets."""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sta_net_pytorch.source_aligned_v3 import SourceBatchNorm3dV3, calibrate_training_batchnorm, convert_retained_parameters
from sta_net_pytorch.model import STANet, STANetConfig

adapter=importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("fused",[False,True])
def test_actual_fused_and_unfused_running_variance_conventions(fused):
    values=np.arange(144,dtype=np.float32).reshape(2,3,2,4,3)/7
    layer=SourceBatchNorm3dV3(3,fused_variance=fused)
    actual=layer(torch.tensor(values)).detach().numpy()
    mean=values.mean((0,2,3,4));population=values.var((0,2,3,4))
    moving=values.var((0,2,3,4),ddof=int(fused))
    np.testing.assert_allclose(actual,(values-mean[None,:,None,None,None])/np.sqrt(population[None,:,None,None,None]+.001),atol=2e-6)
    np.testing.assert_allclose(layer.running_var,.99+.01*moving,atol=2e-7)


def test_layerwise_calibration_uses_only_supplied_training_inputs_and_changes_no_weights():
    p=dict(input_window_schema="downstream_input_v1",duration_s=2,window_start_s=0,num_classes=2,
        sampling_rate_hz=10,fnirs_sampling_rate_hz=2,
        eeg_geometry=[[-1,-1],[-1,1],[1,-1],[1,1]],fnirs_geometry=[[-.8,-1],[-.8,1],[.8,-1],[.8,1]],
        embedding_dim=8,attention_heads=2,attention_key_dim=4,dropout=.5,sta_implementation_version="source_aligned_v3")
    torch.manual_seed(31)
    model=adapter.create_model("dsr",p,"cpu",False)
    original=copy.deepcopy(model)
    eeg,hb=torch.randn(9,4,20),torch.randn(9,8,4)
    train=torch.arange(6);calls=[]
    def batches():
        for ids in train.split(3):
            calls.extend(ids.tolist())
            yield dict(eeg=eeg[ids],fnirs=hb[ids])
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    state=calibrate_training_batchnorm(model,batches)
    assert len(calls)==12 and set(calls)==set(range(6))
    assert len(state["layers"])==6 and state["parameter_updates"]==0
    for key,value in model.named_parameters():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    eeg[6:]=1e6;hb[6:]=float("nan")
    second=calibrate_training_batchnorm(original,batches)
    assert state==second
    for key,value in model.state_dict().items():torch.testing.assert_close(value,original.state_dict()[key],rtol=0,atol=0)
    assert torch.isfinite(model(eeg[:2],hb[:2])).all()
    model.requires_grad_(True)
    with pytest.raises(ValueError,match="frozen"):
        calibrate_training_batchnorm(model,batches)


@pytest.mark.parametrize("classes",[2,4])
def test_retained_classifier_has_an_equivalent_source_parameterization(classes):
    torch.manual_seed(42)
    old=STANet(STANetConfig(task_type="classification",output_dim=classes,embedding_dim=8,
                          attention_heads=2,attention_key_dim=4,max_lags=3,dropout=.5)).eval()
    for layer in old.modules():
        if isinstance(layer,torch.nn.BatchNorm3d):
            layer.running_var.uniform_(1e-4,2.)
            layer.running_mean.uniform_(-.5,.5)
    new,audit=convert_retained_parameters(old)
    assert audit["key_projection_rank"]==8 and audit["positional_residual_float64"]<1e-12
    eeg,hb=torch.randn(3,1,16,16,60),torch.randn(3,3,2,16,16,6)
    with torch.no_grad():
        before,after=old(eeg,hb),new(eeg,hb)
    for key in ["prediction","eeg_prediction","fusion_prediction","fnirs_prediction","fusion_weights","lag_attention"]:
        torch.testing.assert_close(before[key],after[key],rtol=1e-5,atol=2e-6)
