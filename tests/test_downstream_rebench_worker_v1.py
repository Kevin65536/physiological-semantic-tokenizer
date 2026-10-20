"""Temporary, non-authorizing fixtures for the new common training owner."""
import importlib.util
from pathlib import Path
import json

import numpy as np
import pytest
import torch
from torch import nn
from sklearn.metrics import f1_score
from src.metrics.downstream_benchmark import classification_baselines, concordance, regression_train_mean_baseline, score_predictions

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("downstream_worker_test", ROOT / "comparative_methods/downstream_rebench_worker_v1.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


@pytest.fixture
def job(tmp_path):
    method = tmp_path / "rebench_v1.py"
    method.write_text('''import torch
from torch import nn
class Encoder(nn.Module):
    feature_dim=3
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(2,3)
        self.dropout=nn.Dropout(.2)
    def forward(self,eeg,**kwargs):
        return self.dropout(self.projection(eeg.mean(-1)))
def create_model(task,profile,device,trainable):
    model=Encoder().to(device)
    model.requires_grad_(trainable)
    return model
''')
    return dict(schema=worker.JOB_SCHEMA,job_id="fixture",kind="smoke",method="biot",task="motor_imagery",track="ft",
                synthetic_fixture=True,output_dir=str(tmp_path/"run"),batch_size=4,microbatch_size=2,
                smoke_samples=8,smoke_batches=2,num_threads=1,seed=17,
                method_profile=dict(module_path=str(method),channel_names=["C3","C4"],input_samples=20,num_classes=2))


def test_smoke_optimizer_resources_and_completed_are_immutable(job):
    result=worker.run_job(job,"cpu")
    assert result["status"]=="completed",result
    assert result["optimizer_steps"]==2
    assert result["changed_parameter_tensors"]>0
    output=Path(job["output_dir"])
    assert result["resources"]["rss_peak_bytes"]>0
    assert result["resources"]["wall_seconds"]>0
    assert (output/"resource_usage.json").exists()
    probabilities=np.load(output/"predictions.npz")
    assert probabilities["probability"].shape==(8,2)
    np.testing.assert_allclose(probabilities["probability"].sum(-1),1.,atol=1e-6)
    np.testing.assert_array_equal(probabilities["class_ids"][probabilities["probability"].argmax(-1)],probabilities["prediction"])
    assert probabilities["score_kind"].item()=="probability"
    retained=(output/"result.json").read_bytes()
    timestamp=(output/"checkpoint.pt").stat().st_mtime_ns
    assert worker.run_job(job,"cpu")==result
    assert (output/"result.json").read_bytes()==retained
    assert (output/"checkpoint.pt").stat().st_mtime_ns==timestamp
    with pytest.raises(ValueError,match="different job"):
        worker.run_job({**job,"seed":42},"cpu")


def test_failed_oom_retains_explicit_evidence_and_no_batch_shrink(job):
    path=Path(job["method_profile"]["module_path"])
    path.write_text(path.read_text().replace('return self.dropout(self.projection(eeg.mean(-1)))','raise torch.cuda.OutOfMemoryError("temporary fixture OOM")'))
    result=worker.run_job(job,"cpu")
    assert result["status"]=="failed"
    assert result["cuda_oom"] is True
    assert json.loads((Path(job["output_dir"])/"job.json").read_text())["microbatch_size"]==2
    assert (Path(job["output_dir"])/"failure_001.json").exists()
    assert worker.run_job(job,"cpu")["status"]=="failed"
    assert (Path(job["output_dir"])/"failure_002.json").exists()


def test_input_scaler_and_affine_head_fit_training_only(job):
    data=worker.SyntheticDataset(job["method_profile"],job["task"],8)
    data.samples[0]["eeg"][0,0]=1e9
    data.samples[0]["channel_valid_mask"]["eeg"][0,0]=False
    scaler=worker.fit_input_scaler(data)
    assert abs(scaler["eeg"]["mean"][0][0])<1
    train=dict(features=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]]),target=np.array([0,0,1,1]),
               mask=np.ones(4,bool),subject_id=np.array(["a"]*4),sample_id=np.array(["a0","a1","a2","a3"]),record_id=np.array(["r"]*4))
    evaluation={**train,"features":np.array([[1e6,100.]]*4)}
    arrays,state=worker.fit_linear(train,evaluation,job,{"C":1.})
    np.testing.assert_allclose(state["feature_scaler"].mean_,[0,0])
    assert arrays["prediction"].tolist()==[1]*4
    assert arrays["probability"].shape==(4,2)
    np.testing.assert_allclose(arrays["probability"].sum(-1),1.,atol=1e-12)
    assert arrays["score_kind"].item()=="probability"
    assert state["head"].coef_.shape==(1,2)


def test_refed_record_then_subject_weighting_and_undefined_coverage():
    target=np.zeros((4,2,20));prediction=np.zeros_like(target)
    for index in range(4):
        target[index]=np.arange(20)
        prediction[index]=target[index] if index<3 else -target[index]
    subjects=["s"]*4;records=["long"]*3+["short"]
    metrics=score_predictions(target,prediction,np.ones_like(target,bool),subjects,"refed_regression",0,records)
    bad=concordance(target[3,0],prediction[3,0])
    assert metrics["primary"]==pytest.approx((1+bad)/2)
    assert metrics["coverage_complete"]
    target[3]=0;prediction[3]=0
    metrics=score_predictions(target,prediction,np.ones_like(target,bool),subjects,"refed_regression",0,records)
    assert metrics["primary"] is None
    assert metrics["diagnostic_defined_primary"]==1
    assert not metrics["decision_eligible"]
    assert metrics["subjects"][0]["defined_record_count"]==1


def test_temporal_lp_preserves_valid_target_with_missing_input_features(job):
    rng=np.random.default_rng(42)
    train=dict(features=rng.normal(size=(4,20,3)),target=rng.normal(size=(4,2,20)),mask=np.ones((4,2,20),bool),
               feature_valid_mask=np.ones((4,20),bool),subject_id=np.array(["a"]*4),sample_id=np.array(["a0","a1","a2","a3"]),record_id=np.array(["r"]*4))
    train["features"][0]=0;train["feature_valid_mask"][0]=False
    refed_job={**job,"task":"refed_regression"}
    arrays,state=worker.fit_linear(train,train,refed_job,{"alpha":1})
    assert arrays["prediction"].shape==(4,2,20)
    assert state["head"][0].n_features_in_==3
    np.testing.assert_allclose(state["target_scaler"]["mean"][0],train["target"][:,0].mean())


def test_sigterm_checkpoint_resumes_same_rng_optimizer_and_epoch(job,tmp_path):
    module=worker.load_method(job)
    dataset=worker.SyntheticDataset(job["method_profile"],job["task"],8)
    fit_job={**job,"training":dict(max_epochs=2,min_epochs=1,patience=1)}
    resumed=tmp_path/"resume";resumed.mkdir()
    worker._STOP.set()
    with pytest.raises(worker.InterruptedRun):
        worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",resumed,"fit",{"lr":.001},epochs=2)
    initial=torch.load(resumed/"active_training.pt",weights_only=False)
    assert initial["epoch"]==0 and "optimizer" in initial and "rng" in initial
    worker._STOP.clear()
    model,_,details=worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",resumed,"fit",{"lr":.001},epochs=2)
    clean=tmp_path/"clean";clean.mkdir()
    reference,_,_=worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",clean,"fit",{"lr":.001},epochs=2)
    assert details["epochs_completed"]==2
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,reference.state_dict()[key],rtol=0,atol=0)


def test_split_rejects_subject_and_index_overlap():
    inventory=dict(samples=[dict(subject_id="s1"),dict(subject_id="s1"),dict(subject_id="s2")])
    with pytest.raises(ValueError,match="subject leakage"):
        worker._assert_disjoint(inventory,[0],[1])
    with pytest.raises(ValueError,match="indices overlap"):
        worker._assert_disjoint(inventory,[0],[0])
    worker._assert_disjoint(inventory,[0],[2])


def test_invalid_neural_scope_and_old_evidence_paths_rejected(job):
    with pytest.raises(ValueError,match="at most 12"):
        worker.validate_job({**job,"search":{"candidates":[{}]*13}})
    with pytest.raises(ValueError,match="historical/protected"):
        worker.validate_job({**job,"inventory_path":"/runs/protected_campaign_v1/inventory.json"})


def test_prior_random_baseline_matches_exact_enumeration():
    import itertools
    y=np.array([0,0,1]);prior=np.array([.75,.25])
    expected=0.
    for prediction in itertools.product(range(2),repeat=len(y)):
        expected+=np.prod(prior[list(prediction)])*f1_score(y,prediction,labels=[0,1],average="macro",zero_division=0)
    arrays=dict(target=y,mask=np.ones(3,bool),subject_id=np.array(["s"]*3))
    baseline=classification_baselines([0,0,0,1],arrays,2)
    assert baseline["prior_random_exact_expected_subject_equal_macro_f1"]==pytest.approx(expected)


def test_selection_reuse_checks_scope_but_permits_final_seed_change(job,tmp_path):
    retained=tmp_path/"selection.json"
    retained.write_text(json.dumps(dict(scope_identity=worker._selection_scope(job),completed=True,candidates=[{"candidate":{"lr":.001}}])))
    assert worker._reused_selection({**job,"seed":73,"reuse_selection_from":str(retained)})
    with pytest.raises(ValueError,match="another data/model"):
        worker._reused_selection({**job,"task":"dsr","reuse_selection_from":str(retained)})
    with pytest.raises(ValueError,match="synthetic exceptions"):
        worker.validate_job({**job,"kind":"pretrain"})


def test_pretraining_excludes_whole_dataset_and_refits_fresh_stage_b(job,tmp_path,monkeypatch):
    from types import SimpleNamespace
    datasets=["target","source-a","source-b","source-c"]
    samples=[dict(dataset_id=dataset,subject_id=f"{dataset}-s{subject}",task_id=dataset,sample_id=f"{dataset}-{subject}-{window}")
             for dataset in datasets for subject in range(4) for window in range(2)]
    inventory=dict(inventory_kind="pretraining",samples=samples)
    inventory_path=tmp_path/"source_inventory.json";inventory_path.write_text(json.dumps(inventory))
    created=[];saved=[];seen=[]
    class PretrainModel(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.randn(()));self.model_config={"fixture":True};self.pretraining_steps=0
    def create(profile,device,task):
        model=PretrainModel();created.append(float(model.weight.detach()));return model
    def step(model,eeg,fnirs,optimizer,**kwargs):
        optimizer.zero_grad();loss=(model.weight-1).square();loss.backward();optimizer.step();model.pretraining_steps+=1
        return dict(loss=float(loss.detach()))
    def save(model,path,provenance,**kwargs):
        saved.append(provenance);torch.save(dict(provenance=provenance,model=model.state_dict()),path)
    def panels(inv,indices,current_job):
        assert all(inv["samples"][index]["dataset_id"]!="target" for index in indices)
        seen.append(set(indices));result={}
        for source in datasets[1:]:
            selected=[index for index in indices if inv["samples"][index]["dataset_id"]==source]
            if not selected:continue
            profile={**job["method_profile"],"fnirs_channel_names":["L_HbO","L_HbR"],"fnirs_samples":4}
            data=worker.SyntheticDataset(profile,"motor_imagery",len(selected))
            for sample,index in zip(data.samples,selected):
                sample.pop("target");sample.pop("target_valid_mask")
                sample["sample_id"]=inv["samples"][index]["sample_id"]
                sample["subject_id"]=inv["samples"][index]["subject_id"]
            result[source]=(data,profile)
        return result
    monkeypatch.setattr(worker,"_pretrain_panels",panels)
    module=SimpleNamespace(TASK_DATASETS=dict(zip(["motor_imagery","wg","visual","refed_regression"],datasets)),
        INPUT_IDENTITY={"fixture":True},PRETRAINING_BUDGET=dict(max_epochs=100,min_epochs=20,patience=15,learning_rate=.001,
        weight_decay=.01),create_pretraining_model=create,pretrain_step=step,
        pretraining_loss=lambda model,*args,**kwargs: model.weight*0+1.,save_pretraining_checkpoint=save)
    output=tmp_path/"pretrain";output.mkdir()
    pretrain_job={**job,"kind":"pretrain","batch_size":32,"pretrain_inventory_path":str(inventory_path)}
    worker._STOP.clear()
    result=worker.pretrain(pretrain_job,module,"cpu",output)
    assert result["source_dataset_ids"]==datasets[1:]
    assert created[0]==created[1]  # same seed, independently fresh initialization
    assert saved[-1]["stage"]=="stage_b" and saved[-1]["status"]=="completed_refit"
    source_indices={index for index,row in enumerate(samples) if row["dataset_id"]!="target"}
    assert seen[-1]==source_indices
    assert not (seen[0]&seen[1]) and seen[0]|seen[1]==source_indices
    assert (output/"pretrained_checkpoint.pt").exists()


def test_nested_linear_evaluation_refits_outer_training_and_preserves_denominator(job,tmp_path,monkeypatch):
    from types import SimpleNamespace
    samples=[dict(subject_id=f"s{subject}",sample_id=f"s{subject}-{window}",record_id=f"r{subject}",
                  task_id="motor_imagery",target=window%2)
             for subject in range(8) for window in range(4)]
    inventory=dict(samples=samples,tasks={"motor_imagery":dict(duration_s=.1,profiles={"eeg_only":job["method_profile"]})})
    train=list(range(24));evaluation=list(range(24,32));inner=[]
    for index in range(3):
        held=list(range(index*8,(index+1)*8))
        inner.append(dict(train_indices=sorted(set(train)-set(held)),eval_indices=held))
    splits=dict(tasks={"motor_imagery":dict(folds=[dict(train_indices=train,eval_indices=evaluation,inner_folds=inner)])})
    inventory_path=tmp_path/"inventory.json";inventory_path.write_text(json.dumps(inventory))
    splits_path=tmp_path/"splits.json";splits_path.write_text(json.dumps(splits))
    def dataset(inv,indices,current_job):
        data=worker.SyntheticDataset(job["method_profile"],"motor_imagery",len(indices))
        data.indices=indices
        for sample,index in zip(data.samples,indices):
            row=samples[index]
            sample.update({key:row[key] for key in ("subject_id","sample_id","record_id")})
            sample["target"]=np.int64(row["target"])
            sample["eeg"][:]=2*row["target"]-1
        return data
    class Encoder(nn.Module):
        feature_dim=2
        def forward(self,eeg,**kwargs):return eeg.mean(-1)
    module=SimpleNamespace(create_model=lambda *args,**kwargs:Encoder())
    monkeypatch.setattr(worker,"_dataset",dataset)
    output=tmp_path/"evaluate";output.mkdir()
    eval_job={**job,"kind":"evaluate","track":"lp","inventory_path":str(inventory_path),"splits_path":str(splits_path),
              "outer_fold":0,"search":{"candidates":[{"C":.1},{"C":1.}]}}
    result=worker.evaluate(eval_job,module,"cpu",output)
    assert result["metrics"]["primary"]==1
    assert result["train_sample_count"]==24 and result["eval_sample_count"]==8
    assert all(row["inner_subject_count"]==6 for row in result["selection"])
    checkpoint=torch.load(output/"checkpoint.pt",weights_only=False)
    assert set(checkpoint["fit_sample_ids"])=={samples[index]["sample_id"] for index in train}
    predictions=np.load(output/"predictions.npz")
    assert len(predictions["target"])==8
    assert set(predictions["sample_id"])=={samples[index]["sample_id"] for index in evaluation}
    assert json.loads((output/"selection.json").read_text())["completed"]


def test_input_audit_retains_partial_samples_without_scoring(job,tmp_path,monkeypatch):
    inventory=dict(samples=[dict(task_id=job["task"]) for _ in range(4)])
    path=tmp_path/"audit_inventory.json";path.write_text(json.dumps(inventory))
    dataset=worker.SyntheticDataset(job["method_profile"],job["task"],4)
    dataset.samples[0]["channel_valid_mask"]["eeg"][0,-1]=False
    dataset.samples[0]["eeg"][0,-1]=np.nan
    monkeypatch.setattr(worker,"_dataset",lambda *args:dataset)
    result=worker.run_job({**job,"kind":"input_audit","inventory_path":str(path),"output_dir":str(tmp_path/"audit")},"cpu")
    assert result["status"]=="completed"
    assert result["audit_pass"]
    assert result["sample_count"]==4 and result["full_support_sample_count"]==3
    assert result["nonfinite_valid_point_count"]==0
    assert "metrics" not in result


@pytest.mark.parametrize("optimizer_name",["adam","adamw"])
def test_source_optimizer_options_match_explicit_torch_updates(optimizer_name):
    candidate=dict(optimizer=optimizer_name,lr=.003,weight_decay=.05,betas=[.92,.999],eps=1e-9,lr_schedule="constant")
    value=nn.Parameter(torch.tensor([.8,-.3]));reference=nn.Parameter(value.detach().clone())
    optimizer,scheduler,resolved=worker._neural_optimizer([value],candidate,updates_per_epoch=2,horizon_epochs=200)
    cls=torch.optim.Adam if optimizer_name=="adam" else torch.optim.AdamW
    original=cls([reference],lr=.003,weight_decay=.05,betas=(.92,.999),eps=1e-9)
    for _ in range(3):
        value.grad=torch.tensor([.1,-.2]);reference.grad=value.grad.clone()
        optimizer.step();scheduler.step();original.step()
    torch.testing.assert_close(value,reference,rtol=0,atol=0)
    assert resolved["betas"]==[.92,.999] and resolved["eps"]==1e-9
    assert len(optimizer.param_groups)==1


def test_warmup_and_cosine_follow_full_budget_update_prefix():
    parameter=nn.Parameter(torch.tensor(1.))
    optimizer,scheduler,resolved=worker._neural_optimizer([parameter],dict(lr=1.,lr_schedule="cosine",warmup_epochs=2),
                                                        updates_per_epoch=2,horizon_epochs=4)
    rates=[]
    for _ in range(8):
        rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad=torch.ones_like(parameter);optimizer.step();scheduler.step()
    np.testing.assert_allclose(rates,[.25,.5,.75,1,1,(1+np.sqrt(.5))/2,.5,(1-np.sqrt(.5))/2],atol=1e-12)
    assert resolved["horizon_epochs"]==4 and resolved["warmup_updates"]==4
    assert "no horizon compression" in resolved["refit_schedule_policy"]


def test_completed_epoch_resume_replays_optimizer_and_cosine_exactly(job,tmp_path,monkeypatch):
    module=worker.load_method(job);dataset=worker.SyntheticDataset(job["method_profile"],job["task"],8)
    fit_job={**job,"training":dict(max_epochs=4,min_epochs=1,patience=1)}
    candidate=dict(optimizer="adamw",lr=.003,weight_decay=.05,betas=[.92,.999],eps=1e-9,lr_schedule="cosine",warmup_epochs=1)
    resumed=tmp_path/"scheduled_resume";resumed.mkdir()
    original_json=worker._atomic_json
    def stop_after_first_epoch(path,value):
        original_json(path,value)
        if value.get("epoch")==1:worker._STOP.set()
    monkeypatch.setattr(worker,"_atomic_json",stop_after_first_epoch)
    worker._STOP.clear()
    with pytest.raises(worker.InterruptedRun):
        worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",resumed,"fit",candidate,epochs=4)
    checkpoint=torch.load(resumed/"active_training.pt",weights_only=False)
    assert checkpoint["epoch"]==1 and checkpoint["scheduler"]["last_epoch"]==2
    monkeypatch.setattr(worker,"_atomic_json",original_json);worker._STOP.clear()
    model,_,details=worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",resumed,"fit",candidate,epochs=4)
    clean=tmp_path/"scheduled_clean";clean.mkdir()
    reference,_,expected=worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",clean,"fit",candidate,epochs=4)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,reference.state_dict()[key],rtol=0,atol=0)
    assert [row["learning_rate_start"] for row in details["history"]]==[row["learning_rate_start"] for row in expected["history"]]
    assert details["history"][-1]["optimizer_updates_completed"]==8


def test_masked_gradient_accumulation_matches_valid_point_full_batch(job):
    profile=job["method_profile"];dataset=worker.SyntheticDataset(profile,"refed_regression",3)
    dataset.samples[0]["target_valid_mask"][:]=False
    dataset.samples[1]["target_valid_mask"][:,2:]=False
    scaler=worker.TargetScaler(True).fit(dataset);input_scaler=worker.fit_input_scaler(dataset)
    class Encoder(nn.Module):
        feature_dim=2
        def forward(self,eeg,**kwargs):return eeg.mean(-1)
    worker._seed(17)
    full=worker.DownstreamModel(Encoder(),output_kind="features",regression=True,classes=0)
    micro=worker.DownstreamModel(Encoder(),output_kind="features",regression=True,classes=0)
    micro.load_state_dict(full.state_dict())
    left=torch.optim.SGD(full.parameters(),lr=.1,weight_decay=.05)
    right=torch.optim.SGD(micro.parameters(),lr=.1,weight_decay=.05)
    target_job={**job,"task":"refed_regression"};worker._STOP.clear()
    combined=worker._supervised_step(full,[worker._collate(dataset.samples)],left,scaler,input_scaler,target_job,"cpu",{})
    separate=worker._supervised_step(micro,[worker._collate([sample]) for sample in dataset.samples],right,scaler,input_scaler,target_job,"cpu",{})
    assert combined["valid_target_point_count"]==separate["valid_target_point_count"]==44
    assert separate["zero_target_batch_count"]==1 and separate["zero_target_sample_count"]==1
    assert combined["loss"]==pytest.approx(separate["loss"],rel=1e-6)
    for key,value in full.state_dict().items():torch.testing.assert_close(value,micro.state_dict()[key],rtol=1e-6,atol=1e-7)
    before={key:value.detach().clone() for key,value in micro.state_dict().items()}
    empty=worker._supervised_step(micro,[worker._collate([dataset.samples[0]])],right,scaler,input_scaler,target_job,"cpu",{})
    assert empty["optimizer_steps"]==0 and empty["loss"] is None
    for key,value in micro.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)


def test_batch_one_smoke_skips_empty_targets_counts_real_updates_and_retains_predictions(job,tmp_path,monkeypatch):
    profile=job["method_profile"];dataset=worker.SyntheticDataset(profile,"refed_regression",4)
    dataset.samples[0]["target_valid_mask"][:]=False;dataset.samples[1]["target_valid_mask"][:]=False
    dataset.samples[2]["target_valid_mask"][:,2:]=False
    path=tmp_path/"masked_inventory.json"
    path.write_text(json.dumps(dict(tasks={"refed_regression":dict(duration_s=.1,profiles={"eeg_only":profile})})))
    monkeypatch.setattr(worker,"_dataset",lambda *args:dataset)
    monkeypatch.setattr(worker,"_loader",lambda data,current_job,**kwargs:torch.utils.data.DataLoader(data,batch_size=1,collate_fn=worker._collate,shuffle=False))
    masked_job={**job,"kind":"smoke","task":"refed_regression","batch_size":1,"microbatch_size":1,
                "inventory_path":str(path),"sample_indices":[0,1,2,3],"output_dir":str(tmp_path/"masked_smoke")}
    result=worker.run_job(masked_job,"cpu")
    assert result["status"]=="completed",result
    assert result["optimizer_steps"]==2 and result["zero_target_batch_count"]==2
    assert result["zero_target_effective_batch_count"]==2 and result["zero_target_sample_count"]==2
    assert result["valid_target_point_count"]==44
    assert result["sample_count"]==4 and result["processed_training_samples"]==4
    predictions=np.load(Path(masked_job["output_dir"])/"predictions.npz")
    assert predictions["prediction"].shape==(4,2,20)
    np.testing.assert_array_equal(predictions["mask"],np.stack([sample["target_valid_mask"] for sample in dataset]))


def test_native_auxiliary_alignment_uses_sample_mean_while_supervision_uses_valid_points(job):
    from types import SimpleNamespace
    profile=job["method_profile"];dataset=worker.SyntheticDataset(profile,"refed_regression",3)
    dataset.samples[0]["target_valid_mask"][:]=False
    dataset.samples[1]["target_valid_mask"][:,2:]=False
    scaler=worker.TargetScaler(True).fit(dataset);input_scaler=worker.fit_input_scaler(dataset)
    class NativeEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.projection=nn.Linear(2,40)
            self.objective=SimpleNamespace(main_weight=1.,eeg_aux_weight=.5,alignment_weight=2.)
        def supervised_loss(self,eeg,fnirs,target,target_valid_mask,**kwargs):
            values=self.projection(eeg.mean(-1)).reshape(-1,2,20)
            main=(values-target).square()[target_valid_mask].mean()
            auxiliary=(values*.5-target).square()[target_valid_mask].mean()
            alignment=values.square().mean()
            return dict(main=main,eeg_aux=auxiliary,alignment=alignment,total=main+.5*auxiliary+2*alignment)
    worker._seed(17)
    full=worker.DownstreamModel(NativeEncoder(),output_kind="predictions",regression=True,classes=0)
    micro=worker.DownstreamModel(NativeEncoder(),output_kind="predictions",regression=True,classes=0)
    micro.load_state_dict(full.state_dict());worker._STOP.clear()
    candidate={"gradient_clip_norm":1e6};target_job={**job,"task":"refed_regression"}
    a=worker._supervised_step(full,[worker._collate(dataset.samples)],torch.optim.SGD(full.parameters(),lr=.1),
                              scaler,input_scaler,target_job,"cpu",candidate)
    b=worker._supervised_step(micro,[worker._collate([sample]) for sample in dataset.samples],torch.optim.SGD(micro.parameters(),lr=.1),
                              scaler,input_scaler,target_job,"cpu",candidate)
    assert a["loss"]==pytest.approx(b["loss"],rel=1e-6)
    for key,value in full.state_dict().items():torch.testing.assert_close(value,micro.state_dict()[key],rtol=1e-6,atol=1e-7)


@pytest.fixture
def neural_evaluation_fixture(job,tmp_path,monkeypatch):
    samples=[dict(subject_id=f"s{subject}",sample_id=f"s{subject}-{window}",record_id=f"r{subject}",
                  task_id="motor_imagery",target=window%2)
             for subject in range(4) for window in range(4)]
    inventory=dict(samples=samples,tasks={"motor_imagery":dict(duration_s=.1,profiles={"eeg_only":job["method_profile"]})})
    train=list(range(12));inner=[]
    for index in range(3):
        held=list(range(index*4,(index+1)*4))
        inner.append(dict(train_indices=sorted(set(train)-set(held)),eval_indices=held))
    splits=dict(tasks={"motor_imagery":dict(folds=[dict(train_indices=train,eval_indices=list(range(12,16)),inner_folds=inner)])})
    inventory_path=tmp_path/"neural_inventory.json";inventory_path.write_text(json.dumps(inventory))
    splits_path=tmp_path/"neural_splits.json";splits_path.write_text(json.dumps(splits))
    def dataset(inv,indices,current_job):
        data=worker.SyntheticDataset(job["method_profile"],"motor_imagery",len(indices));data.indices=list(indices)
        for sample,index in zip(data.samples,indices):
            row=samples[index]
            sample.update({key:row[key] for key in ("subject_id","sample_id","record_id")})
            sample["target"]=np.int64(row["target"]);sample["eeg"][:]=2*row["target"]-1
        return data
    monkeypatch.setattr(worker,"_dataset",dataset)
    module=worker.load_method(job)
    output=tmp_path/"neural_evaluate";output.mkdir()
    eval_job={**job,"track":"ft","inventory_path":str(inventory_path),"splits_path":str(splits_path),"outer_fold":0,
              "output_dir":str(output),"training":{"max_epochs":1,"min_epochs":1,"patience":1},
              "search":{"candidates":[{"lr":.001}],"source":"synthetic fixture"}}
    worker._STOP.clear()
    return eval_job,module,output,inventory,inner


def test_rolling_state_cross_stage_resume_skips_committed_stages(neural_evaluation_fixture,monkeypatch):
    eval_job,module,output,_,_=neural_evaluation_fixture
    original_finish=worker._finish_stage
    def stop_after_commit(directory,stage,*args,**kwargs):
        value=original_finish(directory,stage,*args,**kwargs)
        if stage=="candidate_00_inner_0":worker._STOP.set()
        return value
    monkeypatch.setattr(worker,"_finish_stage",stop_after_commit)
    with pytest.raises(worker.InterruptedRun):worker.evaluate(eval_job,module,"cpu",output)
    first_record=output/"candidate_00_inner_0.json";retained=first_record.read_bytes()
    assert torch.load(output/"active_training.pt",weights_only=False)["stage"]=="candidate_00_inner_0"
    monkeypatch.setattr(worker,"_finish_stage",original_finish);worker._STOP.clear()
    original_fit=worker.fit_neural;stages=[]
    def tracked_fit(*args,**kwargs):
        stages.append(args[7]);return original_fit(*args,**kwargs)
    monkeypatch.setattr(worker,"fit_neural",tracked_fit)
    result=worker.evaluate(eval_job,module,"cpu",output)
    assert stages==["candidate_00_inner_1","candidate_00_inner_2","refit"]
    assert first_record.read_bytes()==retained
    assert sorted(path.name for path in output.glob("*.pt"))==["active_training.pt","checkpoint.pt"]
    assert torch.load(output/"active_training.pt",weights_only=False)["stage"]=="refit"
    timestamp=(output/"checkpoint.pt").stat().st_mtime_ns
    def no_retraining(*args,**kwargs):raise AssertionError("completed stage was retrained")
    monkeypatch.setattr(worker,"fit_neural",no_retraining)
    repeated=worker.evaluate(eval_job,module,"cpu",output)
    assert repeated["metrics"]==result["metrics"]
    assert (output/"checkpoint.pt").stat().st_mtime_ns==timestamp


def test_completed_training_without_stage_commit_resumes_prediction_without_training(neural_evaluation_fixture,monkeypatch):
    eval_job,module,output,_,inner=neural_evaluation_fixture
    original_finish=worker._finish_stage
    def interrupt_before_commit(directory,stage,*args,**kwargs):
        if stage=="candidate_00_inner_0":raise worker.InterruptedRun("fixture interruption before stage commit")
        return original_finish(directory,stage,*args,**kwargs)
    monkeypatch.setattr(worker,"_finish_stage",interrupt_before_commit)
    with pytest.raises(worker.InterruptedRun):worker.evaluate(eval_job,module,"cpu",output)
    assert not (output/"candidate_00_inner_0.json").exists()
    active=torch.load(output/"active_training.pt",weights_only=False)
    assert active["epoch"]==1 and active["stage"]=="candidate_00_inner_0"
    original_groups=worker._training_groups;forbidden=set(inner[0]["train_indices"])
    def prevent_completed_epoch(data,*args,**kwargs):
        assert set(data.indices)!=forbidden,"completed epoch must resume directly to prediction"
        yield from original_groups(data,*args,**kwargs)
    monkeypatch.setattr(worker,"_finish_stage",original_finish)
    monkeypatch.setattr(worker,"_training_groups",prevent_completed_epoch)
    result=worker.evaluate(eval_job,module,"cpu",output)
    assert result["eval_sample_count"]==4
    assert json.loads((output/"candidate_00_inner_0.json").read_text())["training"]["epochs_completed"]==1
    assert len(list(output.glob("*.pt")))==2


def test_rolling_replacement_requires_matching_commit_and_job_identity(job,tmp_path):
    module=worker.load_method(job);dataset=worker.SyntheticDataset(job["method_profile"],job["task"],8)
    fit_job={**job,"training":{"max_epochs":1,"min_epochs":1,"patience":1}}
    output=tmp_path/"rolling_guard";output.mkdir();worker._STOP.clear()
    _,_,detail=worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",output,"first",{"lr":.001},epochs=1)
    retained=(output/"active_training.pt").read_bytes()
    with pytest.raises(ValueError,match="without matching completed record"):
        worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",output,"second",{"lr":.001},epochs=1)
    assert (output/"active_training.pt").read_bytes()==retained
    worker._finish_stage(output,"first",fit_job,detail["stage_identity"],training=detail,metrics={"primary":1.},
                         checkpoint_identity=detail["checkpoint_identity"])
    changed=json.loads((output/"first.json").read_text());changed["checkpoint_identity"]="different-state"
    (output/"first.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="without matching completed record"):
        worker.fit_neural(module,job["method_profile"],dataset,None,fit_job,"cpu",output,"second",{"lr":.001},epochs=1)
    with pytest.raises(ValueError,match="another job"):
        worker.fit_neural(module,job["method_profile"],dataset,None,{**fit_job,"job_id":"different"},"cpu",output,"second",{"lr":.001},epochs=1)
    assert (output/"active_training.pt").read_bytes()==retained


def test_explicit_multimodal_profile_takes_priority_over_signal_eeg_default():
    assert worker._modality(dict(method="signal_linear",modality_profile="multimodal"))=="multimodal"
    assert worker._modality(dict(method="signal_linear"))=="eeg_only"


def test_refed_training_mean_uses_training_masks_and_native_record_subject_errors(job):
    dataset=worker.SyntheticDataset(job["method_profile"],"refed_regression",2)
    for sample in dataset:
        sample["target"][0]=1.;sample["target"][1]=10.
    dataset.samples[0]["target"][0,0]=1e9;dataset.samples[0]["target_valid_mask"][0,0]=False
    fitted=worker.TargetScaler(True).fit(dataset).statistics()
    assert fitted==dict(train_mean=[1.,10.],train_valid_point_counts=[39,40])
    target=np.zeros((5,2,20));target[:3,0]=np.tile([1.,3.],10);target[:3,1]=12.
    target[3,0]=5.;target[3,1]=10.;target[4,0]=1.;target[4,1]=10.
    mask=np.ones_like(target,bool);mask[3,0,-1]=False;target[3,0,-1]=1e9
    arrays=dict(target=target,mask=mask,subject_id=np.array(["a"]*4+["b"]),record_id=np.array(["long"]*3+["short","one"]))
    result=regression_train_mean_baseline(fitted["train_mean"],fitted["train_valid_point_counts"],arrays)
    assert result["available"] and result["coverage_complete"] and result["b0"]==0
    assert result["mae"]==pytest.approx(.875)
    assert result["rmse"]==pytest.approx((((np.sqrt(2)+2)/2+2)/2)/2)
    assert result["subjects"][0]["records"][1]["valid_point_counts"]==[19,20]
    np.testing.assert_array_equal(result["train_mean"],[1,10])
    arrays["mask"][4]=False
    incomplete=regression_train_mean_baseline(fitted["train_mean"],fitted["train_valid_point_counts"],arrays)
    assert not incomplete["coverage_complete"] and incomplete["mae"] is None


def test_estimator_retains_raw_decisions_without_probability_claim():
    from types import SimpleNamespace
    values=dict(eeg=np.zeros((2,2,20)),fnirs=np.zeros((2,2,2)),target=np.array([0,1]),mask=np.ones(2,bool),
                subject_id=np.array(["a","b"]),sample_id=np.array(["a0","b0"]),record_id=np.array(["r0","r1"]))
    values.update(eeg_mask=np.ones_like(values["eeg"],dtype=bool),fnirs_mask=np.ones_like(values["fnirs"],dtype=bool))
    decisions=np.array([[4.,-4.],[-2.,2.]])
    estimator=SimpleNamespace(classes_=np.array([0,1]),predict=lambda *args,**kwargs:np.array([0,1]),decision_function=lambda *args,**kwargs:decisions)
    arrays=worker._estimator_predict(estimator,values)
    assert "probability" not in arrays
    np.testing.assert_array_equal(arrays["decision_score"],decisions)
    assert arrays["score_kind"].item()=="not_probability"


def test_normwear_receives_original_masks_and_training_mean_zero_filled_inputs(job):
    profile={**job["method_profile"],"fnirs_channel_names":["L_HbO","L_HbR"],"fnirs_samples":2}
    dataset=worker.SyntheticDataset(profile,"motor_imagery",2)
    dataset.samples[0]["channel_valid_mask"]["fnirs"][0]=False
    dataset.samples[0]["fnirs"][0]=np.nan
    scaler=worker.fit_input_scaler(dataset)
    batch=worker._batch(worker._collate([dataset[0]]),"cpu",scaler)
    seen={}
    def adapter(eeg,fnirs,**kwargs):
        seen.update(kwargs)
        assert torch.equal(fnirs[~kwargs["fnirs_mask"]],torch.zeros_like(fnirs[~kwargs["fnirs_mask"]]))
        assert torch.isfinite(eeg).all() and torch.isfinite(fnirs).all()
        return torch.zeros((len(eeg),3))
    worker._method_forward(adapter,batch,{**job,"method":"normwear"})
    assert set(seen)=={"eeg_mask","fnirs_mask"}
    assert not seen["fnirs_mask"][0,0].any()


def test_refed_final_evaluation_records_training_only_mean_errors(neural_evaluation_fixture,monkeypatch):
    eval_job,module,output,inventory,_=neural_evaluation_fixture
    profile=inventory["tasks"]["motor_imagery"]["profiles"]["eeg_only"]
    inventory["tasks"]={"refed_regression":dict(duration_s=20.,profiles={"eeg_only":profile})}
    for row in inventory["samples"]:row["task_id"]="refed_regression";row["target"]=None
    Path(eval_job["inventory_path"]).write_text(json.dumps(inventory))
    splits=json.loads(Path(eval_job["splits_path"]).read_text())
    splits["tasks"]={"refed_regression":splits["tasks"]["motor_imagery"]}
    Path(eval_job["splits_path"]).write_text(json.dumps(splits))
    def data(inv,indices,current_job):
        dataset=worker.SyntheticDataset(profile,"refed_regression",len(indices));dataset.indices=list(indices)
        for sample,index in zip(dataset.samples,indices):
            row=inventory["samples"][index]
            for key in ("subject_id","sample_id","record_id"):sample[key]=row[key]
            sample["target"][0]=np.linspace(0,1,20)+index*.1
            sample["target"][1]=np.linspace(1,0,20)-index*.2
            if index==0:sample["target_valid_mask"][:]=False
            if index==1:sample["target_valid_mask"][:,10:]=False
        return dataset
    monkeypatch.setattr(worker,"_dataset",data)
    regression_job={**eval_job,"task":"refed_regression"}
    result=worker.evaluate(regression_job,module,"cpu",output)
    expected=worker.TargetScaler(True).fit(data(inventory,list(range(12)),regression_job)).statistics()
    baseline=result["baselines"]
    assert baseline["available"] and baseline["coverage_complete"]
    assert baseline["train_valid_point_counts"]==[210,210]
    np.testing.assert_allclose(baseline["train_mean"],expected["train_mean"])
    assert baseline["mae"]>0 and baseline["rmse"]>0 and baseline["b0"]==0
    assert baseline["sample_count"]==4 and baseline["subject_count"]==1


def test_brainfusion_passes_raw_partial_measurements_and_masks_to_fit_and_prediction(job,tmp_path):
    from types import SimpleNamespace
    profile={**job["method_profile"],"fnirs_channel_names":["L_HbO","L_HbR"],"fnirs_samples":2}
    dataset=worker.SyntheticDataset(profile,"motor_imagery",4)
    dataset.samples[0]["channel_valid_mask"]["fnirs"][0]=False
    dataset.samples[0]["fnirs"][0]=np.nan
    dataset.samples[1]["channel_valid_mask"]["eeg"][1,-2:]=False
    dataset.samples[1]["eeg"][1,-2:]=777.
    values=worker._collect_native(dataset)
    assert len(values["target"])==4
    assert np.isnan(values["fnirs"][0,0]).all() and not values["fnirs_mask"][0,0].any()
    assert (values["eeg"][1,1,-2:]==777).all() and not values["eeg_mask"][1,1,-2:].any()
    calls=[]
    class Estimator:
        classes_=np.array([0,1])
        def record(self,name,eeg,fnirs,eeg_mask,fnirs_mask):
            np.testing.assert_equal(eeg,values["eeg"]);np.testing.assert_equal(fnirs,values["fnirs"])
            np.testing.assert_array_equal(eeg_mask,values["eeg_mask"]);np.testing.assert_array_equal(fnirs_mask,values["fnirs_mask"])
            calls.append(name)
        def fit(self,eeg,fnirs,target,*,subjects,sample_ids,eeg_mask,fnirs_mask):
            self.record("fit",eeg,fnirs,eeg_mask,fnirs_mask)
            assert subjects==values["subject_id"].tolist() and sample_ids==values["sample_id"].tolist()
        def predict(self,eeg,fnirs,*,eeg_mask,fnirs_mask):
            self.record("predict",eeg,fnirs,eeg_mask,fnirs_mask);return values["target"]
        def decision_function(self,eeg,fnirs,*,eeg_mask,fnirs_mask):
            self.record("decision",eeg,fnirs,eeg_mask,fnirs_mask);return np.tile([-1.,1.],(4,1))
        def save(self,directory):Path(directory).mkdir(parents=True,exist_ok=True)
        def audit_state(self):return {"fixture":True}
    module=SimpleNamespace(create_estimator=lambda *args,**kwargs:Estimator())
    estimator=worker._fit_estimator(module,profile,{},values,17,"cpu",tmp_path/"fit")
    predictions=worker._estimator_predict(estimator,values)
    assert calls==["fit","predict","decision"] and len(predictions["prediction"])==4
    smoke_output=tmp_path/"smoke";smoke_output.mkdir()
    result=worker.smoke_estimator({**job,"method":"brainfusion"},module,dataset,profile,smoke_output,"synthetic")
    assert result["sample_count"]==4 and calls[-3:]==["fit","predict","decision"]
    dataset.samples[2]["eeg"][0,0]=np.nan
    with pytest.raises(ValueError,match="nonfinite valid BrainFusion"):
        worker._collect_native(dataset)
