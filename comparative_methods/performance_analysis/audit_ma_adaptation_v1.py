#!/usr/bin/env python3
"""One-task, identical-window adaptation audit; no historical test artifacts.

The frozen subject inner fold defines train/development identities before any
signal read. One shared array package is the input to every arm. The existing
rebenchmark scheduler and neural fitter own supervision and recovery.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT))
from src.data.downstream_benchmark import DownstreamBenchmarkDataset, canonical_sample_id
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.event_alignment import window_within_alignment_support
from src.metrics.downstream_benchmark import score_predictions, classification_baselines
from comparative_methods.downstream_rebench_worker_v1 import fit_input_scaler, fit_neural, ResourceMonitor
from comparative_methods.rebench_baselines_v1 import SignalFeatures
from comparative_methods.run_downstream_rebench_v1 import atomic_json

TASK = "mental_arithmetic"


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    atomic_json(Path(path), serial(value))


def resolve_plan(inventory, splits, config, events):
    """Metadata-only migration; do not reuse 8 s sample indices or identities."""
    if config["task"] != TASK or config["input"]["duration_s"] != 30 or config["input"]["window_start_s"] != -5:
        raise ValueError("this audit fixes MA [-5,25)")
    fold = splits["tasks"][TASK]["folds"][config["outer_fold"]]
    inner = fold["inner_folds"][config["inner_fold"]]
    train, validation = set(inner["train_subjects"]), set(inner["validation_subjects"])
    if train & validation or (train | validation) & set(fold["eval_subjects"]):
        raise ValueError("subject split overlap")
    selected = train | validation
    new = deepcopy(inventory)
    rows, excluded, neighbors = [], [], []
    event_groups = {}
    for (key, _), event in events.items():
        event_groups.setdefault(key, []).append(float(event["eeg_time_ms"]))
    for row in inventory["samples"]:
        if row["task_id"] != TASK or row["subject_id"] not in selected:
            continue
        reason = None
        for ref in row["event_refs"]:
            event = events[(ref["join_key"], ref["event_index"])]
            if not window_within_alignment_support(event, -5, 30):
                reason = "unverified_alignment_support_for_30s"
        if reason:
            excluded.append(dict(parent_sample_id=row["sample_id"], reason=reason))
            continue
        migrated = deepcopy(row)
        migrated["parent_sample_id"] = row["sample_id"]
        migrated.update(window_start_s=-5., window_duration_s=30.)
        migrated["sample_id"] = canonical_sample_id(row["dataset_id"], row["subject_id"], row["record_id"], TASK, row["event_id"], -5, 30)
        for ref in migrated["event_refs"]:
            shapes = inventory["records"][ref["join_key"]]["array_shapes"]
            ref["recorded_sample_counts"] = {}
            for modality, rate in (("eeg", 200), ("fnirs", 10)):
                start = round((ref[f"{modality}_time_ms"] / 1000 - 5) * rate)
                ref["recorded_sample_counts"][modality] = max(0, min(start + 30 * rate, shapes[modality][0]) - max(start, 0))
            anchor = ref["eeg_time_ms"]
            others = [t for t in event_groups[ref["join_key"]] if anchor - 5000 <= t < anchor + 25000 and t != anchor]
            if others:
                neighbors.append(dict(sample_id=migrated["sample_id"], other_event_offsets_s=[(t-anchor)/1000 for t in others]))
        rows.append(migrated)
    if neighbors:
        raise ValueError("MA [-5,25) contains another task onset; revise label contract before signals")
    new["samples"] = rows
    spec = deepcopy(inventory["tasks"][TASK])
    eeg = spec["profiles"]["eeg_only"]
    multi = spec["profiles"]["multimodal"]
    multi["channel_names"] = deepcopy(eeg["channel_names"])
    multi["eeg_geometry"] = deepcopy(eeg["eeg_geometry"])
    multi["geometry_provenance"]["eeg"] = deepcopy(eeg["geometry_provenance"]["eeg"])
    for profile in spec["profiles"].values():
        profile.update(duration_s=30., window_start_s=-5., input_samples=6000, fnirs_samples=300,
                       input_window_schema="ma_30s_diagnostic_v1", task=TASK)
    spec.update(duration_s=30., sample_count=len(rows), subject_count=len(selected),
                class_counts={label:sum(r["label"] == label for r in rows) for label in spec["classes"]})
    spec.pop("support_summary", None)
    new["tasks"] = {TASK:spec}
    keys = {ref["join_key"] for row in rows for ref in row["event_refs"]}
    new["records"] = {k:v for k,v in new["records"].items() if k in keys}
    new["exclusions"] = excluded
    new["window_profile"] = "ma_30s_diagnostic_v1"
    new["information_support"] = dict(encoding="event-relative [-5,25) seconds", preprocessing="retained offline full-record v5 preprocessing", label="central MA or BL event; no other task onset inside window")
    partition = np.asarray(["train" if r["subject_id"] in train else "validation" for r in rows])
    for name, subjects in (("train",train),("validation",validation)):
        actual = {r["subject_id"] for r,p in zip(rows,partition) if p == name}
        if actual != subjects:
            raise ValueError("30s support drops a planned subject")
        for subject in subjects:
            if {r["target"] for r in rows if r["subject_id"] == subject} != {0,1}:
                raise ValueError("30s support lacks a class")
    plan = dict(schema="ma_30s_audit_plan_v1", task=TASK, window_s=[-5,25],
                planned_count=len(rows)+len(excluded), retained_count=len(rows), exclusions=excluded,
                train_subjects=sorted(train), validation_subjects=sorted(validation),
                outer_subjects_not_read=fold["eval_subjects"], sample_ids=[r["sample_id"] for r in rows],
                train_count=int((partition=="train").sum()), validation_count=int((partition=="validation").sum()),
                other_task_onsets_inside_window=neighbors, profile=multi,
                masks="common recorded support; no method-specific sample removal",
                scope="single development split; not fresh confirmation or paper ranking")
    return new, plan, partition


def prepare(config, project, run):
    inventory = json.loads((project/config["inventory"]).read_text())
    splits = json.loads((project/config["splits"]).read_text())
    index = CleanPhysiologyCacheIndex(inventory["cache_root"])
    events = {(e["join_key"],int(e["event_index"])):e for e in index.events}
    migrated, plan, partition = resolve_plan(inventory,splits,config,events)
    write(run/"plan.json",plan)
    write(run/"inventory.json",migrated)
    # The complete support plan above is persisted before the first signal read.
    dataset = DownstreamBenchmarkDataset(migrated, task_id=TASK, modality_profile="multimodal")
    eeg=[]; hb=[]; em=[]; hm=[]; units=set()
    for i in range(len(dataset)):
        row=dataset[i]
        eeg.append(np.asarray(row["eeg"],dtype=np.float32));hb.append(np.asarray(row["fnirs"],dtype=np.float32))
        em.append(row["channel_valid_mask"]["eeg"]);hm.append(row["channel_valid_mask"]["fnirs"])
        units.add(migrated["records"][migrated["samples"][i]["event_refs"][0]["join_key"]]["eeg_unit"])
    if units != {"uV"}:
        raise ValueError("physical scale audit requires verified microvolt input")
    arrays = dict(eeg=np.stack(eeg), fnirs=np.stack(hb), eeg_mask=np.stack(em), fnirs_mask=np.stack(hm),
                  target=np.asarray([r["target"] for r in migrated["samples"]]),
                  subject_id=np.asarray([r["subject_id"] for r in migrated["samples"]]),
                  record_id=np.asarray([r["record_id"] for r in migrated["samples"]]),
                  sample_id=np.asarray(plan["sample_ids"]), partition=partition)
    for modality in ("eeg","fnirs"):
        if not np.isfinite(arrays[modality][arrays[f"{modality}_mask"]]).all():
            raise ValueError("nonfinite recorded input")
    if not arrays["eeg_mask"].all():
        write(run/"input_failure.json",dict(reason="incomplete_common_EEG_support",count=int((~arrays["eeg_mask"].all((1,2))).sum())))
        raise ValueError("incomplete EEG support cannot be silently filled for pretrained adapters")
    np.savez(run/"shared_arrays.npz",**arrays)
    train=MemoryDataset(arrays,"train")
    scaler=fit_input_scaler(train)
    write(run/"input_scaler.json",scaler)
    hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
    partial=~arrays["fnirs_mask"].all((1,2))
    write(run/"input_audit.json",dict(array_hashes=hashes,shapes={k:list(v.shape) for k,v in arrays.items()},eeg_unit="uV",
          partial_fnirs_windows=int(partial.sum()),partial_fnirs_by_class={str(k):int((partial & (arrays["target"]==k)).sum()) for k in (0,1)},
          train_count=len(train),validation_count=len(arrays["target"])-len(train),same_package_for_all_arms=True))
    return dict(status="completed",plan=str(run/"plan.json"),input_audit=str(run/"input_audit.json"))


class MemoryDataset:
    def __init__(self, arrays, partition):
        self.arrays=arrays
        self.indices=np.flatnonzero(arrays["partition"]==partition).tolist()
    def __len__(self):return len(self.indices)
    def __getitem__(self, index):
        i=self.indices[index];a=self.arrays
        return dict(eeg=a["eeg"][i],fnirs=a["fnirs"][i],target=a["target"][i],target_valid_mask=np.bool_(True),
                    sample_id=str(a["sample_id"][i]),subject_id=str(a["subject_id"][i]),record_id=str(a["record_id"][i]),
                    valid_mask={m:a[f"{m}_mask"][i].any(0) for m in ("eeg","fnirs")},
                    channel_valid_mask={m:a[f"{m}_mask"][i] for m in ("eeg","fnirs")})


def batches(arrays, indices, scaler, device, batch_size=4):
    for start in range(0,len(indices),batch_size):
        rows=np.asarray(indices[start:start+batch_size]);out={}
        for m in ("eeg","fnirs"):
            values=torch.as_tensor(arrays[m][rows],device=device)
            mask=torch.as_tensor(arrays[f"{m}_mask"][rows],device=device)
            parameters=scaler[m]
            values=(values-torch.tensor(parameters["mean"],device=device))/torch.tensor(parameters["scale"],device=device)
            out[m]=torch.where(mask,values,0.)
        out["channel_valid_mask"]={m:torch.as_tensor(arrays[f"{m}_mask"][rows],device=device) for m in ("eeg","fnirs")}
        yield rows,out


def scores(arrays, indices, prediction):
    rows=np.asarray(indices);y=arrays["target"][rows]
    result=score_predictions(y,np.asarray(prediction),np.ones(len(rows),bool),arrays["subject_id"][rows],TASK,2)
    result.update(accuracy=float(accuracy_score(y,prediction)),confusion_matrix=confusion_matrix(y,prediction,labels=[0,1]).tolist(),
                  recall=recall_score(y,prediction,labels=[0,1],average=None,zero_division=0).tolist(),
                  prediction_counts=np.bincount(prediction,minlength=2).tolist())
    return result


def linear_result(x,arrays,output,name,config,shuffle=False):
    tr=np.flatnonzero(arrays["partition"]=="train");va=np.flatnonzero(arrays["partition"]=="validation")
    scale=StandardScaler().fit(x[tr]);y=arrays["target"][tr].copy()
    if shuffle:np.random.default_rng(config["seed"]).shuffle(y)
    model=LogisticRegression(C=config["linear"]["C"],max_iter=3000,solver="lbfgs").fit(scale.transform(x[tr]),y)
    prediction=model.predict(scale.transform(x))
    np.savez(output/f"{name}_predictions.npz",sample_id=arrays["sample_id"],prediction=prediction,target=arrays["target"])
    return dict(train=scores(arrays,tr,prediction[tr]),validation=scores(arrays,va,prediction[va]),feature_dim=x.shape[1],
                nonconstant_training_features=int((np.std(x[tr],axis=0)>1e-8).sum()),iterations=int(model.n_iter_.max()),
                training_labels_permuted=shuffle)


def run_baselines(config,arrays,profile,scaler,output,device):
    results={};features={}
    for modality in ("eeg","fnirs","multimodal"):
        model=SignalFeatures(TASK,{**profile,"baseline_modality":modality}).to(device)
        xs=[]
        with torch.no_grad():
            for rows,b in batches(arrays,list(range(len(arrays["target"]))),scaler,device,32):
                xs.append(model(b["eeg"],b["fnirs"] if modality!="eeg" else None,
                                eeg_mask=b["channel_valid_mask"]["eeg"],fnirs_mask=b["channel_valid_mask"]["fnirs"]).cpu().numpy())
        features[modality]=np.concatenate(xs)
        results[modality]=linear_result(features[modality],arrays,output,modality,config)
    coverage=np.concatenate([arrays[f"{m}_mask"].mean(-1) for m in ("eeg","fnirs")],axis=1)
    results["coverage_only"]=linear_result(coverage,arrays,output,"coverage_only",config)
    results["permuted_eeg_labels"]=linear_result(features["eeg"],arrays,output,"permuted_eeg_labels",config,True)
    tr=arrays["partition"]=="train";va=~tr
    results["B0"]=classification_baselines(arrays["target"][tr],dict(target=arrays["target"][va],mask=np.ones(va.sum(),bool),subject_id=arrays["subject_id"][va]),2)
    return results


def run_cbramod_lp(config,arrays,profile,scaler,output,device):
    from comparative_methods.CBraMod.rebench_v1 import create_audit_model
    results={}
    for scale in ("train_sd","physical_100uv"):
        p={**profile,"audit_readout":"flatten","audit_scale":scale,"audit_training_eeg_scaler":scaler["eeg"],"eeg_unit":"uV"}
        model=create_audit_model(TASK,p,device).eval();xs=[]
        with torch.no_grad():
            for rows,b in batches(arrays,list(range(len(arrays["target"]))),scaler,device):
                xs.append(model(b["eeg"],eeg_mask=b["channel_valid_mask"]["eeg"]).cpu().numpy())
        flat=np.concatenate(xs);mean=flat.reshape(len(flat),16,30,200).mean((1,2))
        for readout,x in (("mean",mean),("flatten",flat)):
            name=f"{scale}_{readout}"
            results[name]=linear_result(x,arrays,output,name,config)
        del model,flat,mean,xs
        if device.startswith("cuda"):torch.cuda.empty_cache()
    return results


def run_neural(arm,config,arrays,profile,scaler,output,device,smoke=False):
    is_cb=arm.startswith("cbramod")
    if is_cb:
        from comparative_methods.CBraMod.rebench_v1 import create_audit_model
        p={**profile,"audit_readout":"flatten" if arm.endswith("source_head") else "mean",
           "audit_head":"source_mlp" if arm.endswith("source_head") else "linear",
           "audit_training_eeg_scaler":scaler["eeg"],"eeg_unit":"uV"}
        module=SimpleNamespace(create_model=create_audit_model,OUTPUT_KIND="predictions")
    else:
        sta=importlib.import_module("comparative_methods.STA-Net-PyTorch.rebench_v1")
        p={**profile,"dropout":.3 if arm=="sta_dropout03" else .5}
        def factory(task,profile,device,trainable):
            model=sta.create_model(task,profile,device,trainable)
            if arm=="sta_source_bn":
                for layer in model.modules():
                    if isinstance(layer,nn.BatchNorm3d):layer.eps=.001;layer.momentum=.01
            return model
        module=SimpleNamespace(create_model=factory,OUTPUT_KIND="predictions")
    train=MemoryDataset(arrays,"train");validation=MemoryDataset(arrays,"validation")
    budget=config["training"]
    if smoke:
        # Explicit performance smoke only; these are not scientific results.
        train.indices=train.indices[:128];validation.indices=validation.indices[:64]
    job=dict(job_id=arm,kind="diagnostic",task=TASK,method="cbramod" if is_cb else "sta_net",seed=config["seed"],
             modality_profile="eeg_only" if is_cb else "multimodal",batch_size=32,microbatch_size=budget["microbatch_size"],
             gradient_accumulation=32//budget["microbatch_size"],precision="bf16",data_workers=0,
             training=dict(max_epochs=2 if smoke else budget["max_epochs"],min_epochs=2 if smoke else budget["min_epochs"],patience=budget["patience"]))
    model,_,training=fit_neural(module,p,train,validation,job,device,output,arm,budget["cbramod" if is_cb else "sta_net"])
    model.eval();indices=train.indices+validation.indices;pred=[];lags=[];gates=[]
    with torch.no_grad():
        for rows,b in batches(arrays,indices,scaler,device,budget["microbatch_size"]):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                logits=model(b,job)
                if not is_cb:
                    detail=model.encoder(b["eeg"],b["fnirs"],eeg_mask=b["channel_valid_mask"]["eeg"],fnirs_mask=b["channel_valid_mask"]["fnirs"],return_details=True)
                    lags.append(detail["lag_attention"].float().cpu().numpy());gates.append(detail["fusion_weights"].float().cpu().numpy())
            pred.extend(logits.argmax(-1).cpu().tolist())
    pred=np.asarray(pred);n=len(train)
    result=dict(train=scores(arrays,train.indices,pred[:n]),validation=scores(arrays,validation.indices,pred[n:]),training=training,
                parameter_count=sum(v.numel() for v in model.parameters()),profile=p,
                scope="throughput_smoke_not_result" if smoke else "single_split_development_diagnostic")
    if lags:
        attention=np.concatenate(lags)[n:];weights=np.concatenate(gates)[n:]
        result["attention_diagnostic"]=dict(lag_count=attention.shape[1],lag_start_event_seconds=list(range(-5,23)),
            mean_lag_attention=attention.mean(0),mean_entropy=float(-(attention*np.log(attention.clip(1e-12))).sum(1).mean()),
            maximum_entropy=float(np.log(attention.shape[1])),mean_fusion_weights=weights.mean(0))
    np.savez(output/"predictions.npz",sample_id=arrays["sample_id"][indices],target=arrays["target"][indices],prediction=pred)
    return result


def run_brainfusion(config,arrays,profile,output,device):
    module=importlib.import_module("comparative_methods.BrainFusion-NVC-CSP-Stacking.rebench_v1")
    tr=np.flatnonzero(arrays["partition"]=="train");va=np.flatnonzero(arrays["partition"]=="validation")
    model=module.create_estimator(profile,seed=config["seed"],device=device)
    def values(rows):return [torch.from_numpy(arrays[m][rows]) for m in ("eeg","fnirs")]
    def masks(rows):return dict(eeg_mask=torch.from_numpy(arrays["eeg_mask"][rows]),fnirs_mask=torch.from_numpy(arrays["fnirs_mask"][rows]))
    model.fit(*values(tr),arrays["target"][tr],subjects=arrays["subject_id"][tr],sample_ids=arrays["sample_id"][tr],**masks(tr))
    result={}
    for name,rows in (("train",tr),("validation",va)):
        prediction=model.predict(*values(rows),**masks(rows))
        result[name]=scores(arrays,rows,prediction)
        # Same fitted components, observed validation support; no validation fit.
        inputs=model._prediction_inputs(*values(rows),**masks(rows))
        features=model.pipeline.features.transform(*inputs)
        component={}
        for view in module.VIEW_ORDER:
            fitted=model.pipeline.stacking
            x=features[view].cpu().numpy()
            component[view]=scores(arrays,rows,fitted.base_estimators_[view].predict(fitted.view_scalers_[view].transform(x)))
        result[f"{name}_components"]=component
        np.savez(output/f"{name}_predictions.npz",sample_id=arrays["sample_id"][rows],prediction=prediction,target=arrays["target"][rows])
    result["fit_audit"]=model.audit_state()
    result["prediction_degeneracy"]=model.last_prediction_degeneracy_
    model.save(output/"checkpoint")
    return result


def aggregate(config,run):
    results={};failures=[]
    for arm in config["arms"]:
        path=run/arm/"result.json"
        if path.exists():
            value=json.loads(path.read_text())
            if value.get("status")=="completed":results[arm]=value
            else:failures.append(dict(arm=arm,result=value))
        else:failures.append(dict(arm=arm,reason="missing_result"))
    return dict(status="completed" if not failures else "complete_with_failures",scope=config["scope"],
                plan=str(run/"plan.json"),input_audit=str(run/"input_audit.json"),results=results,failures=failures,
                interpretation="Single development split/seed. Validation selected epochs; no statistical superiority or final paper ranking.")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,default=SOURCE_ROOT);parser.add_argument("--arm",required=True)
    parser.add_argument("--device",default="cpu");parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text());run=args.run.resolve()
    if args.arm not in ["prepare","aggregate",*config["arms"]]:raise ValueError("unknown audit arm")
    output=run/("smoke_"+args.arm if args.smoke else args.arm);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config["resources"]["cpu_threads_per_job"])
    torch.manual_seed(config["seed"]);np.random.seed(config["seed"])
    if (output/"result.json").exists() and json.loads((output/"result.json").read_text()).get("status")=="completed":return
    write(output/"status.json",dict(status="running",arm=args.arm,pid=os.getpid(),start_unix=time.time()))
    try:
        with ResourceMonitor(args.device) as monitor:
            if args.arm=="prepare":result=prepare(config,args.project_root,run)
            elif args.arm=="aggregate":result=aggregate(config,run)
            else:
                with np.load(run/"shared_arrays.npz",allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
                profile=json.loads((run/"plan.json").read_text())["profile"]
                profile["method_root"]=str(args.project_root/"comparative_methods/CBraMod")
                scaler=json.loads((run/"input_scaler.json").read_text())
                if args.arm=="baselines":result=run_baselines(config,arrays,profile,scaler,output,args.device)
                elif args.arm=="cbramod_lp":result=run_cbramod_lp(config,arrays,profile,scaler,output,args.device)
                elif args.arm=="brainfusion":result=run_brainfusion(config,arrays,profile,output,args.device)
                else:result=run_neural(args.arm,config,arrays,profile,scaler,output,args.device,args.smoke)
            resource=monitor.snapshot()
        write(output/"resource_usage.json",resource)
        write(output/"result.json",dict(status="completed",arm=args.arm,result=result,resources=resource))
        write(output/"status.json",dict(status="completed",arm=args.arm))
        if args.arm=="aggregate":write(run/"summary.json",result)
    except Exception as error:
        write(output/"status.json",dict(status="failed",arm=args.arm,error=repr(error)))
        raise


if __name__=="__main__":main()
