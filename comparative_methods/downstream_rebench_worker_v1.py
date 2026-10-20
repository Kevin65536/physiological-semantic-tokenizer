#!/usr/bin/env python3
"""One resumable, measured-resource worker for downstream rebenchmark v1.

The scheduler owns authorization and job placement. This worker owns nested
selection, training-only fitted transforms, refitting, predictions and failures.
It imports no historical campaign controller or runner.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.downstream_benchmark import classification_baselines, concordance, regression_train_mean_baseline, score_predictions
JOB_SCHEMA = "downstream_rebench_job_v1"
RESULT_SCHEMA = "downstream_rebench_result_v1"
SUPERVISION_REDUCTION = {
    "supervised_terms": "mean over actual valid target points in each effective batch",
    "native_alignment_terms": "mean over windows with at least one valid target point",
    "zero_target_windows": "retained in inventory, prediction and target coverage; omitted from supervised forward/backward",
    "zero_target_effective_batches": "counted without optimizer, weight-decay or scheduler update",
}
METHOD_DIRS = {"biot": "BIOT", "cbramod": "CBraMod", "labram": "LaBraM", "reve": "REVE",
               "normwear": "NormWear", "efrm": "EFRM-PyTorch",
               "sta_net": "STA-Net-PyTorch", "stanet": "STA-Net-PyTorch", "sta": "STA-Net-PyTorch", "brainfusion": "BrainFusion-NVC-CSP-Stacking"}
_STOP = threading.Event()


class InterruptedRun(RuntimeError):
    pass


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng() -> dict[str, Any]:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _parameter_samples(model):
    return {key:value.detach().reshape(-1)[::max(1,value.numel()//32)][:32].cpu().clone()
            for key,value in model.named_parameters() if value.requires_grad}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def validate_job(job: Mapping[str, Any]) -> None:
    if job.get("schema") != JOB_SCHEMA:
        raise ValueError(f"job schema must be {JOB_SCHEMA}")
    for name in ("job_id", "kind", "method", "task", "output_dir"):
        if not job.get(name):
            raise ValueError(f"missing job field {name}")
    if job["kind"] not in {"smoke", "evaluate", "pretrain", "input_audit"}:
        raise ValueError("kind must be smoke, evaluate, pretrain or input_audit")
    if job.get("track", "lp") not in {"lp", "ft", "supervised"}:
        raise ValueError("unknown downstream track")
    if job["kind"] in {"evaluate","pretrain"} and (job.get("synthetic_fixture") or
        any(job.get("method_profile",{}).get(key) for key in ("synthetic_random_encoder","synthetic_fixture"))):
        raise ValueError("synthetic exceptions are restricted to smoke jobs")
    # These are prohibited historical evidence readers, not an authorization gate.
    for name in ("inventory_path", "splits_path"):
        path = str(job.get(name, "")).lower()
        if any(part in path for part in ("/archive/", "protected_campaign", "/sealed/")):
            raise ValueError(f"historical/protected evidence path prohibited: {name}")
    if job["kind"] == "evaluate" and not all(job.get(key) for key in ("inventory_path", "splits_path")):
        raise ValueError("evaluation requires inventory_path and splits_path")
    if int(job.get("batch_size", 32)) < 1 or int(job.get("microbatch_size", job.get("batch_size", 32))) < 1:
        raise ValueError("batch sizes must be positive")
    if len(job.get("search", {}).get("candidates", [])) > 12:
        raise ValueError("at most 12 tuning candidates are permitted")


def load_method(job: Mapping[str, Any]):
    profile = job.get("method_profile", {})
    default_path = ROOT / "comparative_methods" / METHOD_DIRS.get(str(job["method"]).lower(), str(job["method"])) / "rebench_v1.py"
    if job["method"] == "signal_linear":default_path = ROOT / "comparative_methods" / "rebench_baselines_v1.py"
    path = Path(profile.get("module_path", default_path))
    if not path.is_absolute():
        path = ROOT / path
    if path.name not in {"rebench_v1.py", "rebench_baselines_v1.py"} and not job.get("synthetic_fixture", False):
        raise ValueError("methods must use their versioned rebench_v1.py owner")
    specification = importlib.util.spec_from_file_location("rebench_method_" + _signature(str(path))[:12], path)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot import method: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ResourceMonitor:
    """RSS includes resident children; CUDA peaks are the selected allocator's."""
    def __init__(self, device: str):
        self.device = device
        self.started = time.monotonic()
        self.process = psutil.Process()
        self.rss_peak = 0
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def __enter__(self):
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread.start()
        return self

    def _poll(self):
        while not self.finished.is_set():
            try:
                processes = [self.process, *self.process.children(recursive=True)]
                self.rss_peak = max(self.rss_peak, sum(proc.memory_info().rss for proc in processes if proc.is_running()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self.finished.wait(0.1)

    def snapshot(self) -> dict[str, Any]:
        output = dict(wall_seconds=time.monotonic() - self.started,
                      cpu_seconds=sum(self.process.cpu_times()[:2]),
                      rss_peak_bytes=max(self.rss_peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                      device=self.device, cpu_affinity=self.process.cpu_affinity(),
                      torch_num_threads=torch.get_num_threads())
        if self.device.startswith("cuda"):
            output.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                          cuda_device_name=torch.cuda.get_device_name(self.device))
        return output

    def __exit__(self, *_args):
        self.finished.set()
        self.thread.join(timeout=1)


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("eeg", "fnirs", "target", "target_valid_mask"):
        if key not in samples[0]:continue
        output[key] = torch.as_tensor(np.stack([sample[key] for sample in samples]))
    for key in ("valid_mask", "channel_valid_mask"):
        output[key] = {modality: torch.as_tensor(np.stack([sample[key][modality] for sample in samples]), dtype=torch.bool)
                       for modality in ("eeg", "fnirs")}
    for key in ("sample_id", "subject_id", "record_id"):
        output[key] = [str(sample.get(key, "")) for sample in samples]
    return output


def fit_input_scaler(dataset: Dataset) -> dict[str, Any]:
    """Training-only, channel-wise measured mean/SD on recorded support."""
    moments = {}
    for sample in dataset:
        for modality in ("eeg", "fnirs"):
            values = np.asarray(sample[modality], dtype=np.float64)
            valid = np.asarray(sample["channel_valid_mask"][modality], dtype=bool)
            if not np.isfinite(values[valid]).all():
                raise ValueError("nonfinite valid scaler training measurement")
            if modality not in moments:
                moments[modality] = [np.zeros(values.shape[0]) for _ in range(3)]
            total, square, count = moments[modality]
            total += np.where(valid, values, 0).sum(axis=-1)
            square += np.where(valid, values * values, 0).sum(axis=-1)
            count += valid.sum(axis=-1)
    result = {}
    for modality, (total, square, count) in moments.items():
        if np.any(count == 0):
            raise ValueError(f"training panel contains unsupported {modality} channel")
        mean = total / np.maximum(count, 1)
        scale = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean ** 2, 1e-12))
        result[modality] = dict(mean=mean[:, None].tolist(), scale=scale[:, None].tolist(),
                                valid_sample_count=count.tolist())
    return result


def _batch(batch: dict[str, Any], device: str, input_scaler=None) -> dict[str, Any]:
    result = dict(batch)
    for modality in ("eeg", "fnirs"):
        values = batch[modality].to(device=device, dtype=torch.float32)
        support = batch["channel_valid_mask"][modality].to(device=device)
        if support.shape != values.shape:
            raise ValueError(f"{modality} support must preserve channel/sample axes")
        if not torch.isfinite(values[support]).all():
            raise ValueError(f"nonfinite valid {modality} measurement")
        if input_scaler is not None and values.shape[1]:
            parameters = input_scaler[modality]
            values = (values - torch.as_tensor(parameters["mean"], device=device, dtype=values.dtype)) / torch.as_tensor(parameters["scale"], device=device, dtype=values.dtype)
        result[modality] = torch.where(support, values, 0.0)
    if "target" in batch:
        result["target"] = batch["target"].to(device)
        result["target_valid_mask"] = batch["target_valid_mask"].to(device=device, dtype=torch.bool)
    result["channel_valid_mask"] = {key: value.to(device) for key, value in batch["channel_valid_mask"].items()}
    return result


def _loader(dataset: Dataset, job: Mapping[str, Any], *, shuffle=False, epoch=0) -> DataLoader:
    generator = torch.Generator().manual_seed(int(job.get("seed", 42)) + epoch)
    return DataLoader(dataset, batch_size=int(job.get("microbatch_size", job.get("batch_size", 32))),
                      shuffle=shuffle, num_workers=int(job.get("data_workers", 0)),
                      collate_fn=_collate, generator=generator, drop_last=False)


def _modality(job: Mapping[str, Any]) -> str:
    if job.get("modality_profile") is not None:return str(job["modality_profile"])
    if job["method"]=="signal_linear" and job.get("method_profile",{}).get("baseline_modality","eeg")=="eeg":
        return "eeg_only"
    return str(job.get("modality_profile", "eeg_only" if str(job["method"]).lower() in {"biot", "cbramod", "labram", "reve"} else "multimodal"))


def _autocast(job,device):
    if str(device).startswith("cuda") and job.get("precision","bf16")=="bf16":
        return torch.autocast(device_type="cuda",dtype=torch.bfloat16)
    return nullcontext()


def _dataset(inventory: dict[str, Any], indices: Sequence[int], job: Mapping[str, Any]):
    from src.data.downstream_benchmark import DownstreamBenchmarkDataset
    dataset = DownstreamBenchmarkDataset(inventory, indices, task_id=job["task"], modality_profile=_modality(job))
    if len(dataset) != len(indices):
        raise ValueError("split includes indices from another task")
    return dataset


def _profile(inventory: dict[str, Any] | None, job: Mapping[str, Any]) -> dict[str, Any]:
    profile = {} if inventory is None else dict(inventory["tasks"][job["task"]]["profiles"][_modality(job)])
    if inventory is not None:
        profile["duration_s"] = inventory["tasks"][job["task"]]["duration_s"]
    profile.update(job.get("method_profile", {}))
    profile["task"] = job["task"]
    return profile


def _forward(model: nn.Module, batch: dict[str, Any], job: Mapping[str, Any]) -> torch.Tensor:
    with _autocast(job,batch["eeg"].device):
        return _method_forward(model,batch,job)


def _method_forward(model, batch, job):
    arguments = dict(eeg_mask=batch["channel_valid_mask"]["eeg"])
    if _modality(job) == "eeg_only":
        return model(batch["eeg"], **arguments)
    return model(batch["eeg"], batch["fnirs"], fnirs_mask=batch["channel_valid_mask"]["fnirs"], **arguments)


class TargetScaler:
    def __init__(self, regression: bool):
        self.regression = regression
        self.mean = np.zeros((2, 1), dtype=np.float32)
        self.scale = np.ones((2, 1), dtype=np.float32)
        self.training_mean=np.zeros(2,dtype=np.float64)
        self.valid_point_counts=np.zeros(2,dtype=np.int64)

    def fit(self, dataset: Dataset):
        if self.regression:
            sums = np.zeros(2); squares = np.zeros(2); counts = np.zeros(2)
            for sample in dataset:
                values = np.asarray(sample["target"], dtype=float)
                valid = np.asarray(sample["target_valid_mask"], dtype=bool) & np.isfinite(values)
                sums += np.where(valid, values, 0).sum(axis=-1)
                squares += np.where(valid, values ** 2, 0).sum(axis=-1)
                counts += valid.sum(axis=-1)
            if np.any(counts < 2):
                raise ValueError("insufficient valid training targets")
            self.training_mean=sums/counts;self.valid_point_counts=counts.astype(np.int64)
            self.mean = (sums / counts).astype(np.float32)[:, None]
            self.scale = np.sqrt(np.maximum(squares / counts - self.mean[:, 0] ** 2, 1e-12)).astype(np.float32)[:, None]
        return self

    def transform(self, target: torch.Tensor) -> torch.Tensor:
        return (target.float() - torch.as_tensor(self.mean, device=target.device)) / torch.as_tensor(self.scale, device=target.device) if self.regression else target.long()

    def inverse(self, target: np.ndarray) -> np.ndarray:
        return target * self.scale + self.mean if self.regression else target

    def state(self):
        return dict(regression=self.regression,mean=self.mean,scale=self.scale,
                    training_mean=self.training_mean,valid_point_counts=self.valid_point_counts)

    def statistics(self):
        return dict(train_mean=self.training_mean.tolist(),train_valid_point_counts=self.valid_point_counts.tolist())

    @classmethod
    def from_state(cls,state):
        scaler=cls(bool(state["regression"]))
        scaler.mean=np.asarray(state["mean"]);scaler.scale=np.asarray(state["scale"])
        scaler.training_mean=np.asarray(state.get("training_mean",scaler.mean[:,0]),dtype=np.float64)
        scaler.valid_point_counts=np.asarray(state.get("valid_point_counts",[0,0]),dtype=np.int64)
        return scaler


class DownstreamModel(nn.Module):
    def __init__(self, encoder: nn.Module, *, output_kind: str, regression: bool, classes: int):
        super().__init__()
        self.encoder = encoder
        self.regression = regression
        self.output_kind = output_kind
        self.head: nn.Module | None = None
        if output_kind != "predictions":
            self.head = nn.Linear(int(encoder.feature_dim), 2 if regression and getattr(encoder, "sequence_features", False) else (40 if regression else classes))

    def forward(self, batch, job):
        with _autocast(job,batch["eeg"].device):
            return self._forward(batch,job)

    def _forward(self,batch,job):
        features = _forward(self.encoder, batch, job)
        if self.head is None:
            return features
        # EFRM's fixed one-second REFED features retain time instead of pooling.
        if features.ndim == 3:
            if not self.regression or features.shape[1] != 20:
                raise ValueError("unexpected temporal feature shape")
            if self.head.out_features != 2:
                raise ValueError("temporal feature model must declare sequence_features=True")
            return self.head(features).transpose(1, 2)
        values = self.head(features)
        return values.reshape(-1, 2, 20) if self.regression else values


def _loss(prediction: torch.Tensor, batch: dict[str, Any], scaler: TargetScaler) -> torch.Tensor:
    target = scaler.transform(batch["target"])
    valid = batch["target_valid_mask"]
    if not valid.any():
        raise ValueError("training batch has no valid target")
    if scaler.regression:
        if prediction.shape != target.shape:
            raise ValueError(f"regression shape mismatch: {prediction.shape} versus {target.shape}")
        return ((prediction - torch.where(valid, target, 0.0)) ** 2)[valid].mean()
    return nn.functional.cross_entropy(prediction[valid], target[valid])


def _training_loss(model: DownstreamModel, batch, scaler, job, *, target_weight=1., alignment_weight=1.):
    if hasattr(model.encoder, "supervised_loss"):
        with _autocast(job,batch["eeg"].device):
            terms=model.encoder.supervised_loss(
                batch["eeg"], batch["fnirs"], scaler.transform(batch["target"]),
                target_valid_mask=batch["target_valid_mask"],
                eeg_mask=batch["channel_valid_mask"]["eeg"],
                fnirs_mask=batch["channel_valid_mask"]["fnirs"],
            )
            objective=getattr(model.encoder,"objective",None)
            if objective is not None and all(key in terms for key in ("main","eeg_aux","alignment")):
                supervised=objective.main_weight*terms["main"]+objective.eeg_aux_weight*terms["eeg_aux"]
                return target_weight*supervised+alignment_weight*objective.alignment_weight*terms["alignment"]
            return target_weight*terms["total"]
    return target_weight*_loss(model(batch, job), batch, scaler)


def _training_groups(dataset,job,epoch):
    effective=int(job.get("batch_size",32));micro=int(job.get("microbatch_size",effective))
    accumulation=int(job.get("gradient_accumulation",math.ceil(effective/micro)))
    if effective!=micro*accumulation:raise ValueError("batch_size must equal microbatch_size * gradient_accumulation")
    group=[]
    for raw in _loader(dataset,job,shuffle=True,epoch=epoch):
        group.append(raw)
        if len(group)==accumulation:
            yield group;group=[]
    if group:yield group


def _target_rows(raw):
    mask=raw["target_valid_mask"]
    return mask.reshape(mask.shape[0],-1).any(-1)


def _select_batch(raw,selected):
    """Select supervised rows for training; full inventory remains in prediction."""
    return {key:({name:array[selected] for name,array in value.items()} if isinstance(value,dict)
                 else value[selected] if isinstance(value,torch.Tensor)
                 else [item for item,keep in zip(value,selected.tolist()) if keep])
            for key,value in raw.items()}


def _supervised_step(model,group,optimizer,scaler,input_scaler,job,device,candidate):
    target_count=sum(int(raw["target_valid_mask"].sum()) for raw in group)
    eligible_count=sum(int(_target_rows(raw).sum()) for raw in group)
    sample_count=sum(len(raw["sample_id"]) for raw in group)
    empty_batches=sum(not raw["target_valid_mask"].any().item() for raw in group)
    result=dict(optimizer_steps=0,valid_target_point_count=target_count,supervised_sample_count=eligible_count,
                processed_sample_count=sample_count,zero_target_sample_count=sample_count-eligible_count,
                zero_target_batch_count=empty_batches,loss=None)
    optimizer.zero_grad(set_to_none=True)
    if not target_count:return result
    losses=[]
    for raw in group:
        rows=_target_rows(raw)
        if not rows.any():continue
        selected=_select_batch(raw,rows)
        batch=_batch(selected,device,input_scaler)
        valid_count=int(batch["target_valid_mask"].sum())
        loss=_training_loss(model,batch,scaler,job,target_weight=valid_count/target_count,
                            alignment_weight=len(selected["sample_id"])/eligible_count)
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite supervised loss")
        loss.backward();losses.append(float(loss.detach()))
        if _STOP.is_set():raise InterruptedRun("SIGTERM: resume the last completed epoch checkpoint")
    torch.nn.utils.clip_grad_norm_(model.parameters(),float(candidate.get("gradient_clip_norm",candidate.get("grad_clip_norm",5.0))))
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    result.update(optimizer_steps=1,loss=float(sum(losses)))
    return result


def _predict(model: DownstreamModel, dataset: Dataset, scaler: TargetScaler,
             job: Mapping[str, Any], device: str) -> dict[str, Any]:
    model.eval()
    predictions=[]; targets=[]; masks=[]; subjects=[]; sample_ids=[]; records=[];probabilities=[]
    with torch.no_grad():
        for raw in _loader(dataset, job):
            batch = _batch(raw, device, job.get("_input_scaler", getattr(model, "input_scaler", None)))
            scores=model(batch,job).float()
            values = scores.cpu().numpy()
            if not scaler.regression:probabilities.append(torch.softmax(scores,dim=-1).cpu().numpy())
            predictions.append(scaler.inverse(values) if scaler.regression else values.argmax(axis=-1))
            targets.append(raw["target"].numpy()); masks.append(raw["target_valid_mask"].numpy())
            subjects.extend(raw["subject_id"]); sample_ids.extend(raw["sample_id"]); records.extend(raw["record_id"])
    arrays=dict(prediction=np.concatenate(predictions), target=np.concatenate(targets), mask=np.concatenate(masks),
                subject_id=np.asarray(subjects), sample_id=np.asarray(sample_ids), record_id=np.asarray(records))
    if probabilities:
        arrays.update(probability=np.concatenate(probabilities),class_ids=np.arange(probabilities[0].shape[-1]),score_kind=np.asarray("probability"))
    return arrays


def _metrics(arrays, job, profile):
    return score_predictions(arrays["target"], arrays["prediction"], arrays["mask"], arrays["subject_id"],
                             job["task"], int(profile.get("num_classes", 2)), arrays.get("record_id"))


def _training_job_identity(job):
    return job.get("_job_identity") or _signature({key:value for key,value in job.items() if not key.startswith("_")})


def _stage_identity(job,stage,candidate,profile,train_indices,eval_indices,epochs=None):
    return _signature(dict(job_identity=_training_job_identity(job),stage=stage,candidate=candidate,
                           profile=profile,train_indices=train_indices,eval_indices=eval_indices,epochs=epochs,
                           seed=job.get("seed"),supervision_reduction=SUPERVISION_REDUCTION))


def _completed_stage(output,stage,job,identity):
    path=output/(stage+".json")
    if not path.exists():return None
    retained=_json(path)
    if retained.get("status")!="completed" or retained.get("stage")!=stage or retained.get("job_identity")!=_training_job_identity(job) or retained.get("stage_identity")!=identity:
        raise ValueError(f"completed stage contract changed: {stage}")
    return retained


def _finish_stage(output,stage,job,identity,**details):
    retained=_completed_stage(output,stage,job,identity)
    if retained is not None:return retained
    value=dict(status="completed",stage=stage,job_identity=_training_job_identity(job),stage_identity=identity,**details)
    _atomic_json(output/(stage+".json"),value)
    return value


def _active_checkpoint(output,stage,job):
    """A stage may replace only its own state or a committed predecessor."""
    path=output/"active_training.pt"
    if not path.exists():return path,None
    retained=torch.load(path,map_location="cpu",weights_only=False)
    if retained.get("job_identity")!=_training_job_identity(job):
        raise ValueError("active training checkpoint belongs to another job")
    if retained.get("stage")==stage:return path,retained
    previous=retained.get("stage")
    if not previous:raise ValueError("active training checkpoint has no owning stage")
    record=_completed_stage(output,previous,job,retained.get("stage_identity"))
    if record is None or record.get("checkpoint_identity")!=retained.get("identity"):
        raise ValueError(f"cannot replace active stage without matching completed record: {previous}")
    return path,None


def _neural_optimizer(parameters, candidate, *, updates_per_epoch, horizon_epochs):
    """One parameter group; source-declared Adam variant and update schedule."""
    name=str(candidate.get("optimizer","adamw")).lower()
    if name not in {"adam","adamw"}:raise ValueError("optimizer must be adam or adamw")
    learning_rate=float(candidate.get("learning_rate",candidate.get("lr",1e-4)))
    if "learning_rate" in candidate and "lr" in candidate and candidate["learning_rate"]!=candidate["lr"]:
        raise ValueError("conflicting learning_rate and lr")
    weight_decay=float(candidate.get("weight_decay",.01));epsilon=float(candidate.get("eps",1e-8))
    betas=tuple(float(value) for value in candidate.get("betas",(.9,.999)))
    if len(betas)!=2 or not all(math.isfinite(value) and 0<=value<1 for value in betas):
        raise ValueError("betas must contain two finite values in [0,1)")
    if not math.isfinite(learning_rate) or learning_rate<=0 or not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError("learning rate and eps must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay<0:raise ValueError("weight_decay must be finite and nonnegative")
    schedule=str(candidate.get("lr_schedule","constant"))
    if schedule not in {"constant","cosine"}:raise ValueError("lr_schedule must be constant or cosine")
    warmup=candidate.get("warmup_epochs",0)
    if not isinstance(warmup,(int,float)) or not math.isfinite(warmup) or int(warmup)!=warmup or warmup<0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    warmup=int(warmup)
    if updates_per_epoch<1 or horizon_epochs<1 or warmup>horizon_epochs:
        raise ValueError("invalid update schedule horizon")
    optimizer_class=torch.optim.Adam if name=="adam" else torch.optim.AdamW
    optimizer=optimizer_class(parameters,lr=learning_rate,weight_decay=weight_decay,betas=betas,eps=epsilon)
    total_updates=horizon_epochs*updates_per_epoch;warmup_updates=warmup*updates_per_epoch
    def multiplier(update):
        if update<warmup_updates:return (update+1)/warmup_updates
        if schedule=="constant":return 1.
        progress=min(1.,max(0.,(update-warmup_updates)/max(1,total_updates-warmup_updates)))
        return .5*(1.+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    resolved=dict(optimizer=name,learning_rate=learning_rate,weight_decay=weight_decay,betas=list(betas),eps=epsilon,
                  lr_schedule=schedule,warmup_epochs=warmup,updates_per_epoch=updates_per_epoch,horizon_epochs=horizon_epochs,
                  total_schedule_updates=total_updates,warmup_updates=warmup_updates,
                  refit_schedule_policy="replay the inner-budget schedule prefix using this partition's actual updates per epoch; no horizon compression")
    return optimizer,scheduler,resolved


def fit_neural(module, profile: dict[str, Any], train: Dataset, evaluation: Dataset | None,
               job: Mapping[str, Any], device: str, output: Path, stage: str,
               candidate: Mapping[str, Any], *, epochs: int | None = None) -> tuple[DownstreamModel, TargetScaler, dict[str, Any]]:
    phase_started=time.monotonic()
    job={**job,"_job_identity":_training_job_identity(job)}
    path,retained_checkpoint=_active_checkpoint(output,stage,job)
    regression = job["task"] == "refed_regression"
    _seed(int(job.get("seed", 42)))
    encoder = module.create_model(job["task"], profile, device, trainable=True)
    output_kind = str(getattr(module, "OUTPUT_KIND", "features"))
    model = DownstreamModel(encoder, output_kind=output_kind, regression=regression,
                            classes=int(profile.get("num_classes", 2))).to(device)
    scaler = TargetScaler(regression).fit(train) if retained_checkpoint is None else TargetScaler.from_state(retained_checkpoint["target_scaler"])
    input_scaler = fit_input_scaler(train) if retained_checkpoint is None else retained_checkpoint["input_scaler"]
    job = {**job, "_input_scaler": input_scaler}
    budget = job.get("training", {})
    maximum = int(epochs if epochs is not None else budget.get("max_epochs", 200))
    minimum = int(budget.get("min_epochs", 50))
    patience = int(budget.get("patience", 30))
    if job["kind"] == "evaluate" and epochs is None and (maximum != 200 or minimum != 50 or patience != 30):
        raise ValueError("formal neural selection budget is max200/min50/patience30")
    effective = int(job.get("batch_size", 32))
    micro = int(job.get("microbatch_size", effective))
    accumulation = int(job.get("gradient_accumulation", math.ceil(effective / micro)))
    if effective != micro * accumulation:
        raise ValueError("batch_size must equal microbatch_size * gradient_accumulation")
    updates_per_epoch=math.ceil(math.ceil(len(train)/micro)/accumulation)
    optimizer,scheduler,optimizer_spec=_neural_optimizer(model.parameters(),candidate,
        updates_per_epoch=updates_per_epoch,horizon_epochs=int(budget.get("max_epochs",200)))
    stage_contract_identity=job.get("_stage_identity") or _stage_identity(job,stage,candidate,profile,getattr(train,"indices",None),getattr(evaluation,"indices",None),epochs)
    checkpoint_owner=dict(stage=stage,job_identity=_training_job_identity(job),stage_identity=stage_contract_identity)
    identity = _signature(dict(candidate=candidate, indices=getattr(train,"indices",None),
                               seed=job.get("seed"), profile=profile, epochs=epochs,optimizer_spec=optimizer_spec,
                               supervision_reduction=SUPERVISION_REDUCTION,**checkpoint_owner))
    start=0; best=-math.inf; best_epoch=0; stale=0; best_state=None; history=[]
    if retained_checkpoint is not None:
        checkpoint = retained_checkpoint
        if checkpoint["identity"] != identity:
            raise ValueError(f"checkpoint contract changed: {stage}")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start=checkpoint["epoch"]; best=checkpoint["best"]; best_epoch=checkpoint["best_epoch"]
        stale=checkpoint["stale"]; best_state=checkpoint["best_state"]; history=checkpoint["history"]
        scaler=TargetScaler.from_state(checkpoint["target_scaler"]); _restore_rng(checkpoint["rng"])
        del checkpoint,retained_checkpoint
    else:
        _atomic_torch(path, dict(**checkpoint_owner,identity=identity, epoch=0, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),optimizer_spec=optimizer_spec,
                                rng=_rng(), best=best,best_epoch=0,stale=0,best_state=None,
                                target_scaler=scaler.state(),input_scaler=input_scaler,history=[]))
    model.input_scaler = input_scaler
    if evaluation is not None and start>=minimum and stale>=patience:maximum=start
    for epoch in range(start, maximum):
        epoch_started=time.monotonic();optimizer_steps=0
        learning_rate_start=float(optimizer.param_groups[0]["lr"]);learning_rate_last=learning_rate_start
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses=[];loss_weights=[];zero_target_batches=0;zero_target_samples=0;zero_target_groups=0
        valid_target_points=0;supervised_samples=0
        for group in _training_groups(train,job,epoch):
            learning_rate_last=float(optimizer.param_groups[0]["lr"])
            step=_supervised_step(model,group,optimizer,scaler,input_scaler,job,device,candidate)
            optimizer_steps+=step["optimizer_steps"]
            zero_target_batches+=step["zero_target_batch_count"];zero_target_samples+=step["zero_target_sample_count"]
            valid_target_points+=step["valid_target_point_count"];supervised_samples+=step["supervised_sample_count"]
            if step["optimizer_steps"]:
                scheduler.step();losses.append(step["loss"]);loss_weights.append(step["valid_target_point_count"])
            else:zero_target_groups+=1
            if _STOP.is_set():raise InterruptedRun("SIGTERM: resume the last completed epoch checkpoint")
        if not losses:raise ValueError("training partition has no supervised target points")
        epoch_loss=float(np.average(losses,weights=loss_weights))
        if device.startswith("cuda"):torch.cuda.synchronize(device)
        training_seconds=time.monotonic()-epoch_started
        validation_started=time.monotonic()
        validation = None if evaluation is None else _metrics(_predict(model,evaluation,scaler,job,device),job,profile)["primary"]
        validation_seconds=time.monotonic()-validation_started
        if evaluation is not None and validation is None:
            raise ValueError("validation endpoint has incomplete or undefined record coverage")
        if validation is None or validation > best:
            best = float(validation if validation is not None else -epoch_loss); best_epoch=epoch+1; stale=0
            best_state={key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            stale += 1
        history.append(dict(epoch=epoch+1, train_loss=epoch_loss, validation_primary=validation,
                            valid_target_point_count=valid_target_points,supervised_sample_count=supervised_samples,
                            zero_target_batch_count=zero_target_batches,zero_target_sample_count=zero_target_samples,
                            zero_target_effective_batch_count=zero_target_groups,
                            training_seconds=training_seconds,validation_seconds=validation_seconds,
                            training_samples_per_second=len(train)/training_seconds,optimizer_steps=optimizer_steps,
                            learning_rate_start=learning_rate_start,learning_rate_last=learning_rate_last,
                            optimizer_updates_completed=int(scheduler.last_epoch)))
        _atomic_torch(path, dict(**checkpoint_owner,identity=identity, epoch=epoch+1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),optimizer_spec=optimizer_spec,
                                rng=_rng(), best=best,best_epoch=best_epoch,stale=stale,best_state=best_state,
                                target_scaler=scaler.state(),input_scaler=input_scaler, history=history))
        _atomic_json(output/"status.json",dict(status="running",stage=stage,epoch=epoch+1,max_epochs=maximum,
                                             job_id=job["job_id"],updated_unix=time.time()))
        if evaluation is not None and epoch+1 >= minimum and stale >= patience:
            break
    if best_state is None:
        raise ValueError("training produced no completed epoch")
    model.load_state_dict(best_state)
    return model, scaler, dict(best_epoch=best_epoch,epochs_completed=len(history),history=history,
                              input_scaler=input_scaler,optimizer_spec=optimizer_spec,
                              supervision_reduction=SUPERVISION_REDUCTION,
                              checkpoint_identity=identity,stage_identity=stage_contract_identity,
                              target_statistics=scaler.statistics() if regression else None,
                              phase_wall_seconds=time.monotonic()-phase_started)


def _features(module, profile, dataset, job, device, input_scaler):
    started=time.monotonic()
    _seed(int(job.get("seed",42)))
    encoder=module.create_model(job["task"],profile,device,trainable=False)
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("LP encoder must be frozen by create_model(trainable=False)")
    encoder.eval(); features=[]; targets=[]; masks=[]; subjects=[]; ids=[]; records=[]; feature_masks=[]
    with torch.no_grad():
        for raw in _loader(dataset,job):
            batch=_batch(raw,device,input_scaler)
            values=_forward(encoder,batch,job).float().cpu().numpy()
            features.append(values)
            support=getattr(encoder,"sequence_feature_mask",None)
            if callable(support) and values.ndim==3:
                support=support(batch["eeg"],batch["fnirs"],eeg_mask=batch["channel_valid_mask"]["eeg"],fnirs_mask=batch["channel_valid_mask"]["fnirs"])
            else:
                support=None
            feature_masks.append(np.ones(values.shape[:-1],dtype=bool) if support is None else support.detach().cpu().numpy())
            targets.append(raw["target"].numpy());masks.append(raw["target_valid_mask"].numpy())
            subjects.extend(raw["subject_id"]);ids.extend(raw["sample_id"]);records.extend(raw["record_id"])
    return dict(features=np.concatenate(features),target=np.concatenate(targets),mask=np.concatenate(masks),
                subject_id=np.asarray(subjects),sample_id=np.asarray(ids),record_id=np.asarray(records),
                feature_valid_mask=np.concatenate(feature_masks),extraction_wall_seconds=time.monotonic()-started)


def fit_linear(train: dict[str,Any], evaluation: dict[str,Any], job: Mapping[str,Any],
               candidate: Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    """Actual affine head; all scales and penalties see training rows only."""
    regression=job["task"]=="refed_regression"
    x=train["features"]; z=evaluation["features"]
    temporal=x.ndim==3
    if temporal and (not regression or x.shape[1]!=20):
        raise ValueError("unexpected temporal LP features")
    rows=x.reshape(-1,x.shape[-1])
    feature_support=np.asarray(train.get("feature_valid_mask",np.ones(x.shape[:-1],dtype=bool))).reshape(-1)
    if not feature_support.any():raise ValueError("no valid training features")
    feature_scaler=StandardScaler().fit(rows[feature_support])
    xs=feature_scaler.transform(x.reshape(-1,x.shape[-1]));zs=feature_scaler.transform(z.reshape(-1,z.shape[-1]))
    if not regression:
        mask=train["mask"].astype(bool)
        head=LogisticRegression(C=float(candidate.get("C",1.0)),max_iter=int(candidate.get("max_iter",3000)),
                                random_state=int(job.get("seed",42)),solver="lbfgs")
        head.fit(xs[mask],train["target"][mask])
        prediction=head.predict(zs); target_scaler=None
    else:
        target=train["target"].transpose(0,2,1).reshape(-1,2) if temporal else train["target"].reshape(len(x),-1)
        valid=train["mask"].transpose(0,2,1).reshape(-1,2) if temporal else train["mask"].reshape(len(x),-1)
        heads=[];means=[];scales=[];columns=[]
        for dimension in range(target.shape[1]):
            support=valid[:,dimension].astype(bool)&np.isfinite(target[:,dimension])
            if support.sum()<2:
                raise ValueError("insufficient valid training labels for linear output")
            values=target[support,dimension];mean=float(values.mean());scale=max(float(values.std()),1e-6)
            head=Ridge(alpha=float(candidate.get("alpha",1.0)))
            head.fit(xs[support],(values-mean)/scale)
            columns.append(head.predict(zs)*scale+mean);heads.append(head);means.append(mean);scales.append(scale)
        prediction=np.stack(columns,axis=-1)
        prediction=prediction.reshape(len(z),20,2).transpose(0,2,1) if temporal else prediction.reshape(len(z),2,20)
        head=heads;target_scaler=dict(mean=means,scale=scales)
    arrays={key:evaluation[key] for key in ("target","mask","subject_id","sample_id","record_id")}
    arrays["prediction"]=prediction
    if not regression:
        arrays.update(probability=head.predict_proba(zs),class_ids=head.classes_,score_kind=np.asarray("probability"))
    return arrays,dict(head=head,feature_scaler=feature_scaler,target_scaler=target_scaler,
                       fit_sample_ids=train["sample_id"].tolist(),temporal_features=temporal)


def _assert_disjoint(inventory: dict[str,Any], train: Sequence[int], evaluation: Sequence[int]) -> None:
    if len(set(train))!=len(train) or len(set(evaluation))!=len(evaluation):
        raise ValueError("duplicate sample indices inside split")
    if set(train)&set(evaluation):
        raise ValueError("train/evaluation indices overlap")
    left={inventory["samples"][int(i)]["subject_id"] for i in train}
    right={inventory["samples"][int(i)]["subject_id"] for i in evaluation}
    if left&right:
        raise ValueError("subject leakage across split")


def _candidates(job, module):
    search=job.get("search",{})
    candidates=search.get("candidates")
    if candidates is None:
        if job.get("track","lp")=="lp":
            candidates=[dict(alpha=value) if job["task"]=="refed_regression" else dict(C=value)
                        for value in (0.001,0.01,0.1,1.0,10.0,100.0)]
        else:
            candidates=getattr(module,"DEFAULT_SEARCH",None)
            if candidates is None:
                raise ValueError("FT/supervised search requires explicit source-based candidate ranges")
    if not candidates or len(candidates)>12:
        raise ValueError("candidate count must be between 1 and 12")
    if job.get("track","lp")!="lp" and not search.get("source") and not getattr(module,"SEARCH_SOURCE",None):
        raise ValueError("neural search must identify its method-source range")
    if hasattr(module,"create_estimator"):
        allowed={"nvc_pair_count","components_per_class","csp_regularization","meta_svm_c"}
    elif job.get("track","lp")=="lp":
        allowed={"C","max_iter"} if job["task"]!="refed_regression" else {"alpha"}
    else:
        allowed={"learning_rate","lr","weight_decay","gradient_clip_norm","grad_clip_norm",
                 "optimizer","betas","eps","lr_schedule","warmup_epochs"}
    for candidate in candidates:
        if set(candidate)-allowed:raise ValueError(f"unconsumed candidate parameters: {sorted(set(candidate)-allowed)}")
    return list(candidates)


def _selection_scope(job):
    return _signature({key:job.get(key) for key in ("method","task","track","outer_fold","inventory_path", "splits_path",
                       "modality_profile","method_profile","search","training","precision","batch_size","microbatch_size","gradient_accumulation")})


def _reused_selection(job):
    path=job.get("reuse_selection_from")
    if not path:return []
    path=Path(path)
    if path.is_dir():path=path/"selection.json"
    retained=_json(path)
    if retained.get("scope_identity")!=_selection_scope(job):
        raise ValueError("reused selection belongs to another data/model/outer-fold/search contract")
    if not retained.get("completed",False):raise ValueError("cannot reuse partial hyperparameter selection")
    return retained["candidates"]


def evaluate(job, module, device, output):
    job={**job,"_job_identity":_training_job_identity(job)}
    inventory=_json(job["inventory_path"]);splits=_json(job["splits_path"])
    profile=_profile(inventory,job)
    fold=splits["tasks"][job["task"]]["folds"][int(job["outer_fold"])]
    train_indices=fold["train_indices"];eval_indices=fold["eval_indices"]
    _assert_disjoint(inventory,train_indices,eval_indices)
    inner=fold["inner_folds"]
    if len(inner)!=3:
        raise ValueError("exactly three inner folds required")
    for group in inner:
        _assert_disjoint(inventory,group["train_indices"],group["eval_indices"])
        if (set(group["train_indices"])|set(group["eval_indices"]))!=set(train_indices):
            raise ValueError("inner train/evaluation must partition all outer-training samples")
    validation_indices=[index for group in inner for index in group["eval_indices"]]
    if len(validation_indices)!=len(set(validation_indices)) or set(validation_indices)!=set(train_indices):
        raise ValueError("inner evaluation folds must cover outer-training samples exactly once")
    candidates=_candidates(job,module);selection=_reused_selection(job);linear=job.get("track","lp")=="lp"
    local_selection=output/"selection.json"
    if not selection and local_selection.exists():
        retained=_json(local_selection)
        if retained.get("job_identity")!=_training_job_identity(job) or retained.get("scope_identity")!=_selection_scope(job):
            raise ValueError("local selection belongs to another job contract")
        if retained.get("completed"):selection=retained["candidates"]
    tuning_job={**job,"seed":17}
    if hasattr(module,"create_estimator"):
        return evaluate_estimator(job,module,device,output,inventory,fold,profile,candidates)
    # External pretrained weights, including dataset-excluded EFRM, are fixed.
    # Feature extraction has no fitted target-task quantities and can be shared.
    feature_cache={}
    if linear and not selection:
        for number,group in enumerate(inner):
            training=_dataset(inventory,group["train_indices"],job)
            input_scaler=fit_input_scaler(training)
            feature_cache[number]=(_features(module,profile,training,tuning_job,device,input_scaler),
                                   _features(module,profile,_dataset(inventory,group["eval_indices"],job),tuning_job,device,input_scaler))
    for number,candidate in enumerate(candidates if not selection else []):
        scores=[];epochs=[];subject_scores=[]
        for index,group in enumerate(inner):
            stage=f"candidate_{number:02d}_inner_{index}"
            stage_scope=_stage_identity(tuning_job,stage,candidate,profile,group["train_indices"],group["eval_indices"])
            retained=_completed_stage(output,stage,tuning_job,stage_scope)
            if retained is not None:
                metric=retained["metrics"];detail=retained["training"]
            else:
                if linear:
                    arrays,_=fit_linear(*feature_cache[index],tuning_job,candidate);detail={}
                else:
                    train=_dataset(inventory,group["train_indices"],job);validation=_dataset(inventory,group["eval_indices"],job)
                    stage_job={**tuning_job,"_stage_identity":stage_scope}
                    model,scaler,detail=fit_neural(module,profile,train,validation,stage_job,device,output,stage,candidate)
                    arrays=_predict(model,validation,scaler,job,device)
                    del model
                    if device.startswith("cuda"):torch.cuda.empty_cache()
                metric=_metrics(arrays,job,profile)
            scores.append(metric["primary"])
            if metric["primary"] is None:raise ValueError("inner-fold primary endpoint is undefined")
            if not linear:epochs.append(detail["best_epoch"])
            subject_scores.extend(row["score"] for row in metric["subjects"] if row["score"] is not None)
            _finish_stage(output,stage,tuning_job,stage_scope,candidate=candidate,metrics=metric,training=detail,
                          checkpoint_identity=detail.get("checkpoint_identity"))
            if _STOP.is_set():raise InterruptedRun("stop requested")
        selection.append(dict(candidate=candidate,inner_scores=scores,mean_primary=float(np.mean(subject_scores)),
                              inner_subject_count=len(subject_scores),tuning_seed=17,best_epochs=epochs))
        _atomic_json(output/"selection.json",dict(candidates=selection,selection_uses="outer-training subjects only",
                     job_identity=_training_job_identity(job),scope_identity=_selection_scope(job),completed=len(selection)==len(candidates)))
    _atomic_json(output/"selection.json",dict(candidates=selection,selection_uses="outer-training subjects only",
                 job_identity=_training_job_identity(job),scope_identity=_selection_scope(job),completed=True,reused_from=job.get("reuse_selection_from")))
    best=max(range(len(selection)),key=lambda index:selection[index]["mean_primary"])
    chosen=selection[best];training=_dataset(inventory,train_indices,job);evaluation=_dataset(inventory,eval_indices,job)
    if linear:
        input_scaler=fit_input_scaler(training)
        train_features=_features(module,profile,training,job,device,input_scaler);eval_features=_features(module,profile,evaluation,job,device,input_scaler)
        fit_started=time.monotonic()
        arrays,checkpoint=fit_linear(train_features,eval_features,job,chosen["candidate"])
        linear_fit_seconds=time.monotonic()-fit_started
        _atomic_torch(output/"checkpoint.pt",dict(kind="frozen_encoder_linear_head",**checkpoint,
                     profile=profile,input_scaler=input_scaler,selected_candidate=chosen["candidate"]))
        refit=dict(fit_sample_count=len(training),encoder_trainable=False,linear_fit_seconds=linear_fit_seconds,
                   training_feature_seconds=train_features["extraction_wall_seconds"],
                   evaluation_feature_seconds=eval_features["extraction_wall_seconds"])
    else:
        fixed_epochs=max(1,math.ceil(float(np.median(chosen["best_epochs"]))))
        refit_scope=_stage_identity(job,"refit",chosen["candidate"],profile,train_indices,eval_indices,fixed_epochs)
        retained=_completed_stage(output,"refit",job,refit_scope)
        if retained is not None:
            if not (output/"checkpoint.pt").exists() or not (output/"predictions.npz").exists():
                raise ValueError("completed refit is missing its final checkpoint or predictions")
            with np.load(output/"predictions.npz",allow_pickle=False) as archive:
                arrays={key:archive[key].copy() for key in archive.files}
            refit=retained["training"]
        else:
            model,scaler,refit=fit_neural(module,profile,training,None,{**job,"_stage_identity":refit_scope},device,output,"refit",chosen["candidate"],epochs=fixed_epochs)
            arrays=_predict(model,evaluation,scaler,job,device)
            _atomic_torch(output/"checkpoint.pt",dict(kind="neural_full_outer_refit",model=model.state_dict(),target_scaler=scaler.state(),
                         profile=profile,input_scaler=model.input_scaler,selected_candidate=chosen["candidate"],epochs=fixed_epochs))
            _atomic_npz(output/"predictions.npz",**arrays)
            _finish_stage(output,"refit",job,refit_scope,candidate=chosen["candidate"],metrics=_metrics(arrays,job,profile),
                          training=refit,checkpoint_identity=refit["checkpoint_identity"])
    if linear:_atomic_npz(output/"predictions.npz",**arrays)
    if job["task"]=="refed_regression":
        statistics=refit.get("target_statistics") or TargetScaler(True).fit(training).statistics()
        baselines=regression_train_mean_baseline(statistics["train_mean"],statistics["train_valid_point_counts"],arrays)
    else:
        baselines=classification_baselines([inventory["samples"][index]["target"] for index in train_indices],arrays,int(profile["num_classes"]))
    return dict(metrics=_metrics(arrays,job,profile),baselines=baselines,selected_candidate=chosen["candidate"],
                selection=selection,training=refit,outer_fold=int(job["outer_fold"]),seed=int(job.get("seed",42)),
                train_sample_count=len(training),eval_sample_count=len(evaluation),
                refit_epoch_rule="ceiling median of three selected-candidate inner best epochs" if not linear else None)


def evaluate_estimator(job,module,device,output,inventory,fold,profile,candidates):
    selection=_reused_selection(job)
    for number,candidate in enumerate(candidates if not selection else []):
        scores=[];subject_scores=[]
        for index,group in enumerate(fold["inner_folds"]):
            train=_collect_native(_dataset(inventory,group["train_indices"],job))
            validation=_collect_native(_dataset(inventory,group["eval_indices"],job))
            stage=output/f"candidate_{number:02d}_inner_{index}"
            estimator=_fit_estimator(module,profile,candidate,train,17,device,stage)
            arrays=_estimator_predict(estimator,validation)
            metric=_metrics(arrays,job,profile)
            subject_scores.extend(row["score"] for row in metric["subjects"])
            scores.append(metric["primary"])
            _atomic_json(stage.with_suffix(".json"),dict(metrics=metric,candidate=candidate,audit=estimator.audit_state()))
            if _STOP.is_set():raise InterruptedRun("estimator fit checkpoint saved after stop")
        selection.append(dict(candidate=candidate,inner_scores=scores,mean_primary=float(np.mean(subject_scores)),tuning_seed=17))
        _atomic_json(output/"selection.json",dict(candidates=selection,scope_identity=_selection_scope(job),completed=len(selection)==len(candidates)))
    _atomic_json(output/"selection.json",dict(candidates=selection,scope_identity=_selection_scope(job),completed=True,
                 reused_from=job.get("reuse_selection_from")))
    chosen=max(selection,key=lambda row:row["mean_primary"])
    train=_collect_native(_dataset(inventory,fold["train_indices"],job))
    evaluation=_collect_native(_dataset(inventory,fold["eval_indices"],job))
    estimator=_fit_estimator(module,profile,chosen["candidate"],train,int(job.get("seed",42)),device,output/"checkpoint")
    arrays=_estimator_predict(estimator,evaluation)
    _atomic_npz(output/"predictions.npz",**arrays)
    return dict(metrics=_metrics(arrays,job,profile),baselines=classification_baselines(train["target"],arrays,int(profile["num_classes"])),
                selected_candidate=chosen["candidate"],selection=selection,
                seed=int(job.get("seed",42)),outer_fold=int(job["outer_fold"]),
                train_sample_count=len(train["target"]),eval_sample_count=len(evaluation["target"]),
                audit=estimator.audit_state())


def _collect_native(dataset):
    values={key:[] for key in ("eeg","fnirs","eeg_mask","fnirs_mask","target","mask","subject_id","sample_id","record_id")}
    for sample in dataset:
        for modality in ("eeg","fnirs"):
            support=np.asarray(sample["channel_valid_mask"][modality],dtype=bool)
            array=np.asarray(sample[modality],dtype=np.float64)
            if support.shape!=array.shape:
                raise ValueError("BrainFusion native support must preserve channel/sample axes")
            if not np.isfinite(array[support]).all():
                raise ValueError("nonfinite valid BrainFusion native measurement")
            values[modality].append(array)
            values[modality+"_mask"].append(support)
        values["target"].append(sample["target"]);values["mask"].append(sample["target_valid_mask"])
        for key in ("subject_id","sample_id","record_id"):values[key].append(sample[key])
    return {key:np.asarray(value) for key,value in values.items()}


def _fit_estimator(module,profile,candidate,train,seed,device,destination):
    resolved={**profile,**candidate}
    estimator=module.create_estimator(resolved,seed=seed,device=device)
    if (destination/"rebench_fit_completed.json").exists():
        if not hasattr(type(estimator),"load"):raise ValueError("estimator does not implement its retained state reader")
        return type(estimator).load(destination,profile=resolved,seed=seed,device=device)
    estimator.fit(train["eeg"],train["fnirs"],train["target"],subjects=train["subject_id"].tolist(),sample_ids=train["sample_id"].tolist(),
                  eeg_mask=train["eeg_mask"],fnirs_mask=train["fnirs_mask"])
    estimator.save(destination)
    _atomic_json(destination/"rebench_fit_completed.json",dict(seed=seed,candidate=candidate,
                 fit_sample_ids=train["sample_id"].tolist()))
    return estimator


def _estimator_predict(estimator,values):
    masks=dict(eeg_mask=values["eeg_mask"],fnirs_mask=values["fnirs_mask"])
    arrays=dict(prediction=estimator.predict(values["eeg"],values["fnirs"],**masks),
                **{key:values[key] for key in ("target","mask","subject_id","sample_id","record_id")})
    arrays["class_ids"]=np.asarray(estimator.classes_)
    if hasattr(estimator,"predict_proba"):
        arrays.update(probability=estimator.predict_proba(values["eeg"],values["fnirs"],**masks),score_kind=np.asarray("probability"))
    else:
        arrays.update(decision_score=estimator.decision_function(values["eeg"],values["fnirs"],**masks),score_kind=np.asarray("not_probability"))
    return arrays


class SyntheticDataset(Dataset):
    def __init__(self, profile, task, count=8, seed=42):
        rng=np.random.default_rng(seed);self.samples=[]
        for index in range(count):
            eeg=rng.normal(size=(len(profile["channel_names"]),int(profile["input_samples"]))).astype(np.float32)
            fnirs=rng.normal(size=(len(profile.get("fnirs_channel_names",[])),int(profile.get("fnirs_samples",0)))).astype(np.float32)
            target=rng.normal(size=(2,20)).astype(np.float32) if task=="refed_regression" else np.int64(index%int(profile.get("num_classes",2)))
            self.samples.append(dict(eeg=eeg,fnirs=fnirs,target=target,target_valid_mask=np.ones_like(target,dtype=bool),
                valid_mask=dict(eeg=np.ones(eeg.shape[-1],dtype=bool),fnirs=np.ones(fnirs.shape[-1],dtype=bool)),
                channel_valid_mask=dict(eeg=np.ones_like(eeg,dtype=bool),fnirs=np.ones_like(fnirs,dtype=bool)),
                sample_id=f"synthetic-{index}",subject_id=f"synthetic-subject-{index//2}",record_id="synthetic"))
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return self.samples[index]


def smoke(job,module,device,output):
    inventory=_json(job["inventory_path"]) if job.get("inventory_path") else None
    profile=_profile(inventory,job)
    if inventory is None:
        if not job.get("synthetic_fixture",False):
            raise ValueError("synthetic smoke requires synthetic_fixture=True")
        dataset=SyntheticDataset(profile,job["task"],int(job.get("smoke_samples",8)),int(job.get("seed",42)))
        scope="synthetic"
    else:
        indices=job.get("sample_indices")
        if indices is None:
            raise ValueError("public-development smoke requires explicit sample_indices")
        dataset=_dataset(inventory,indices,job);scope="public_development"
    _seed(int(job.get("seed",42)))
    if hasattr(module,"create_estimator"):
        return smoke_estimator(job,module,dataset,profile,output,scope)
    if job.get("smoke_objective")=="pretrain":
        return smoke_pretraining(job,module,dataset,profile,device,output,scope)
    encoder=module.create_model(job["task"],profile,device,trainable=True)
    regression=job["task"]=="refed_regression"
    model=DownstreamModel(encoder,output_kind=getattr(module,"OUTPUT_KIND","features"),regression=regression,
                          classes=int(profile.get("num_classes",2))).to(device)
    scaler=TargetScaler(regression).fit(dataset)
    input_scaler=fit_input_scaler(dataset);model.input_scaler=input_scaler
    candidate=job.get("smoke_candidate",dict(learning_rate=float(job.get("smoke_learning_rate",1e-4))))
    updates_per_epoch=math.ceil(len(dataset)/int(job.get("batch_size",32)))
    optimizer,scheduler,optimizer_spec=_neural_optimizer(model.parameters(),candidate,updates_per_epoch=updates_per_epoch,
                                                        horizon_epochs=int(job.get("training",{}).get("max_epochs",200)))
    before=_parameter_samples(model)
    requested_steps=int(job.get("smoke_optimizer_steps",job.get("smoke_batches",2)))
    losses=[];started=time.monotonic();model.train();processed_samples=0;valid_target_points=0
    zero_target_batches=0;zero_target_samples=0;zero_target_groups=0;supervised_samples=0
    for epoch in range(int(job.get("smoke_epochs",max(1,requested_steps)))):
        for group in _training_groups(dataset,job,epoch):
            step=_supervised_step(model,group,optimizer,scaler,input_scaler,job,device,candidate)
            processed_samples+=step["processed_sample_count"];valid_target_points+=step["valid_target_point_count"]
            zero_target_batches+=step["zero_target_batch_count"];zero_target_samples+=step["zero_target_sample_count"]
            supervised_samples+=step["supervised_sample_count"]
            if step["optimizer_steps"]:scheduler.step();losses.append(step["loss"])
            else:zero_target_groups+=1
            if len(losses)>=requested_steps:break
        if len(losses)>=requested_steps:break
    if len(losses)<requested_steps:
        raise ValueError(f"smoke achieved {len(losses)} of {requested_steps} requested supervised optimizer steps")
    if device.startswith("cuda"):torch.cuda.synchronize(device)
    training_seconds=time.monotonic()-started
    after=_parameter_samples(model)
    changed=sum(not torch.equal(before[key],after[key]) for key in before)
    if changed==0:raise ValueError("smoke optimizer changed no trainable parameter")
    prediction_started=time.monotonic()
    arrays=_predict(model,dataset,scaler,job,device);prediction_seconds=time.monotonic()-prediction_started
    _atomic_npz(output/"predictions.npz",**arrays)
    _atomic_torch(output/"checkpoint.pt",dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                  optimizer_spec=optimizer_spec,rng=_rng(),target_scaler=scaler.state(),input_scaler=input_scaler))
    return dict(scope=scope,forward_backward_optimizer=True,optimizer_steps=len(losses),changed_parameter_tensors=changed,
                losses=losses,sample_count=len(dataset),requested_optimizer_steps=requested_steps,optimizer_spec=optimizer_spec,
                supervision_reduction=SUPERVISION_REDUCTION,
                zero_target_batch_count=zero_target_batches,zero_target_sample_count=zero_target_samples,
                zero_target_effective_batch_count=zero_target_groups,valid_target_point_count=valid_target_points,
                supervised_training_samples=supervised_samples,
                training_seconds=training_seconds,prediction_seconds=prediction_seconds,
                processed_training_samples=processed_samples,sample_throughput=processed_samples/training_seconds)


def smoke_estimator(job,module,dataset,profile,output,scope):
    values=_collect_native(dataset)
    estimator=_fit_estimator(module,{**profile,"task":job["task"]},{},values,int(job.get("seed",42)),"cpu",output/"checkpoint")
    arrays=_estimator_predict(estimator,values)
    _atomic_npz(output/"predictions.npz",**arrays)
    return dict(scope=scope,fit_predict_completed=True,sample_count=len(dataset),audit=estimator.audit_state())


def smoke_pretraining(job,module,dataset,profile,device,output,scope):
    model=module.create_pretraining_model(profile,device,task=job["task"])
    scaler=fit_input_scaler(dataset)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.95))
    before=_parameter_samples(model)
    batches=DataLoader(dataset,batch_size=int(job.get("batch_size",32)),collate_fn=_collate,shuffle=False)
    history=[];started=time.monotonic();processed=0
    for index,raw in enumerate(batches):
        raw.pop("target",None);raw.pop("target_valid_mask",None)
        batch=_batch(raw,device,scaler)
        metrics=module.pretrain_step(model,batch["eeg"],batch["fnirs"],optimizer,profile=profile,
                    microbatch_size=int(job.get("microbatch_size",4)),
                    amp_dtype=torch.bfloat16 if device.startswith("cuda") else None,
                    eeg_mask=batch["channel_valid_mask"]["eeg"],fnirs_mask=batch["channel_valid_mask"]["fnirs"])
        history.append(metrics)
        processed+=len(raw["sample_id"])
        if index+1>=int(job.get("smoke_batches",2)):break
    if device.startswith("cuda"):torch.cuda.synchronize(device)
    training_seconds=time.monotonic()-started
    after=_parameter_samples(model)
    changed=sum(not torch.equal(before[key],after[key]) for key in before)
    if not changed:raise ValueError("pretraining smoke changed no trainable parameter")
    _atomic_torch(output/"checkpoint.pt",dict(kind="smoke_only_not_a_pretrained_encoder",model=model.state_dict(),
                  optimizer=optimizer.state_dict(),rng=_rng(),input_scaler=scaler))
    return dict(scope=scope,objective="pretraining",forward_backward_optimizer=True,optimizer_steps=len(history),
                history=history,changed_parameter_tensors=changed,sample_count=len(dataset),
                training_seconds=training_seconds,processed_training_samples=processed,sample_throughput=processed/training_seconds,
                effective_batch_size=int(job.get("batch_size",32)),microbatch_size=int(job.get("microbatch_size",4)))


def pretrain(job,module,device,output):
    inventory=_json(job["pretrain_inventory_path"])
    if inventory.get("inventory_kind")!="pretraining":
        raise ValueError("EFRM pretraining requires the label-blind eight-second source inventory")
    excluded=module.TASK_DATASETS[job["task"]]
    sources=sorted(set(module.TASK_DATASETS.values())-{excluded})
    membership={};validation_subjects={}
    for dataset_id in sources:
        subjects=sorted({row["subject_id"] for row in inventory["samples"] if row["dataset_id"]==dataset_id})
        if len(subjects)<2:raise ValueError("each non-target dataset needs source train and validation subjects")
        permutation=np.random.default_rng(20261016).permutation(subjects)
        count=max(1,int(round(0.2*len(subjects))))
        validation_subjects[dataset_id]=set(permutation[:count])
        membership[dataset_id]=dict(train_subjects=sorted(set(subjects)-validation_subjects[dataset_id]),
                                    validation_subjects=sorted(validation_subjects[dataset_id]))
    source_rows=[index for index,row in enumerate(inventory["samples"]) if row["dataset_id"] in sources]
    if not source_rows:raise ValueError("no admitted non-target pretraining windows")
    identity=_signature(dict(samples=[inventory["samples"][index]["sample_id"] for index in source_rows],
                             source_partition=membership,excluded_target_dataset=excluded))
    _atomic_json(output/"source_membership.json",dict(excluded_target_dataset=excluded,source_dataset_ids=sources,
                 partition_seed=20261016,membership=membership,source_membership_identity=identity,
                 sample_ids=[inventory["samples"][index]["sample_id"] for index in source_rows]))
    budget=dict(module.PRETRAINING_BUDGET);budget.update(job.get("pretraining",{}))
    if (int(budget["max_epochs"]),int(budget["min_epochs"]),int(budget["patience"]))!=(100,20,15):
        raise ValueError("EFRM formal source budget is max100/min20/patience15")
    base_provenance=dict(input_identity=module.INPUT_IDENTITY,excluded_target_dataset=excluded,
                        source_dataset_ids=sources,source_membership_identity=identity,
                        initialization="fresh_random",seed=int(job.get("seed",42)))
    stage_a=_pretrain_stage(job,module,device,output,inventory,source_rows,validation_subjects,budget,
                            base_provenance,"stage_a",None)
    # Stage B is deliberately a separate fresh initialization, never A weights.
    stage_b=_pretrain_stage(job,module,device,output,inventory,source_rows,validation_subjects,budget,
                            base_provenance,"stage_b",int(stage_a["best_epoch"]))
    return dict(excluded_target_dataset=excluded,source_dataset_ids=sources,seed=int(job.get("seed",42)),
                source_membership_identity=identity,stage_a=stage_a,stage_b=stage_b,
                pretrained_checkpoint=str(output/"pretrained_checkpoint.pt"))


def _pretrain_panels(inventory,indices,job):
    from src.data.downstream_benchmark import DownstreamBenchmarkDataset
    panels={}
    for task in sorted({inventory["samples"][index]["task_id"] for index in indices}):
        selected=[index for index in indices if inventory["samples"][index]["task_id"]==task]
        if selected:
            profile=_profile(inventory,{**job,"task":task,"modality_profile":"multimodal"})
            if float(profile["duration_s"])!=8 or int(profile["input_samples"])!=1600 or int(profile["fnirs_samples"])!=80:
                raise ValueError("pretraining panels must preserve the registered eight-second native support")
            panels[task]=(DownstreamBenchmarkDataset(inventory,selected,task_id=task,modality_profile="multimodal"),profile)
    return panels


def _pretrain_batches(dataset,job,epoch,*,shuffle):
    effective=int(job.get("batch_size",32))
    if effective!=32:raise ValueError("EFRM pretraining effective batch size is 32")
    count=len(dataset)
    if count<2:raise ValueError("pretraining panel has fewer than two admitted pairs")
    order=np.random.default_rng(int(job.get("seed",42))+epoch).permutation(count) if shuffle else np.arange(count)
    # Distribute remainders so a final singleton is never dropped or duplicated.
    groups=[group.tolist() for group in np.array_split(order,math.ceil(count/effective))]
    return DataLoader(dataset,batch_sampler=groups,collate_fn=_collate,num_workers=int(job.get("data_workers",0)))


def _pretrain_stage(job,module,device,output,inventory,source_rows,validation_subjects,budget,base,stage,fixed_epochs):
    phase_started=time.monotonic()
    training=[index for index in source_rows if stage=="stage_b" or
              inventory["samples"][index]["subject_id"] not in validation_subjects[inventory["samples"][index]["dataset_id"]]]
    evaluation=[] if stage=="stage_b" else [index for index in source_rows if index not in set(training)]
    panels=_pretrain_panels(inventory,training,job);validation=_pretrain_panels(inventory,evaluation,job) if evaluation else {}
    scalers={task:fit_input_scaler(dataset) for task,(dataset,_profile_) in panels.items()}
    first_profile=next(iter(panels.values()))[1]
    _seed(int(job.get("seed",42)))
    model=module.create_pretraining_model(first_profile,device,task=job["task"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(budget["learning_rate"]),
                               weight_decay=float(budget["weight_decay"]),betas=tuple(budget.get("adam_betas",(.9,.95))))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(budget["max_epochs"]))
    maximum=int(fixed_epochs if fixed_epochs is not None else budget["max_epochs"])
    start=0;best=math.inf;best_epoch=0;stale=0;history=[]
    checkpoint_path=output/(stage+"_resume.pt")
    stage_identity=_signature(dict(base=base,stage=stage,indices=training,budget=budget,model_config=model.model_config,fixed_epochs=fixed_epochs))
    if checkpoint_path.exists():
        checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if checkpoint["identity"]!=stage_identity:raise ValueError("pretraining resume source/stage/config changed")
        model.load_state_dict(checkpoint["model"]);optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"]);model.pretraining_steps=checkpoint["pretraining_steps"]
        start=checkpoint["epoch"];best=checkpoint["best"];best_epoch=checkpoint["best_epoch"]
        stale=checkpoint["stale"];history=checkpoint["history"];scalers=checkpoint["scalers"]
        _restore_rng(checkpoint["rng"])
    def save_resume(epoch):
        _atomic_torch(checkpoint_path,dict(identity=stage_identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                      scheduler=scheduler.state_dict(),pretraining_steps=model.pretraining_steps,epoch=epoch,best=best,
                      best_epoch=best_epoch,stale=stale,history=history,scalers=scalers,rng=_rng()))
    if start==0:save_resume(0)
    if stage=="stage_a" and start>=int(budget["min_epochs"]) and stale>=int(budget["patience"]):maximum=start
    for epoch in range(start,maximum):
        losses=[];counts=[];epoch_started=time.monotonic()
        for task in np.random.default_rng(int(job.get("seed",42))+epoch).permutation(list(panels)):
            dataset,profile=panels[task]
            for raw in _pretrain_batches(dataset,job,epoch,shuffle=True):
                batch=_batch(raw,device,scalers[task])
                if "target" in batch:raise ValueError("labels reached the self-supervised pretraining boundary")
                values=module.pretrain_step(model,batch["eeg"],batch["fnirs"],optimizer,profile=profile,
                    microbatch_size=int(job.get("microbatch_size",4)),
                    amp_dtype=torch.bfloat16 if device.startswith("cuda") else None,
                    gradient_clip_norm=float(budget.get("gradient_clip_norm",5.0)),
                    eeg_mask=batch["channel_valid_mask"]["eeg"],fnirs_mask=batch["channel_valid_mask"]["fnirs"])
                losses.append(float(values["loss"]));counts.append(len(raw["sample_id"]))
                if _STOP.is_set():raise InterruptedRun("pretraining resumes from last completed epoch")
        if device.startswith("cuda"):torch.cuda.synchronize(device)
        training_seconds=time.monotonic()-epoch_started
        validation_loss=None;validation_started=time.monotonic()
        if validation:
            val_losses=[];val_counts=[];model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device).index] if device.startswith("cuda") else []):
                torch.manual_seed(20261016)
                for task,(dataset,profile) in validation.items():
                    for raw in _pretrain_batches(dataset,job,0,shuffle=False):
                        batch=_batch(raw,device,scalers[task])
                        with _autocast(job,device):
                            value=module.pretraining_loss(model,batch["eeg"],batch["fnirs"],profile=profile,
                                eeg_mask=batch["channel_valid_mask"]["eeg"],fnirs_mask=batch["channel_valid_mask"]["fnirs"])
                        if not torch.isfinite(value):raise FloatingPointError("nonfinite source validation loss")
                        val_losses.append(float(value));val_counts.append(len(raw["sample_id"]))
            validation_loss=float(np.average(val_losses,weights=val_counts))
        if validation_loss is None or validation_loss<best:
            best=validation_loss if validation_loss is not None else float(np.average(losses,weights=counts))
            best_epoch=epoch+1;stale=0
        else:stale+=1
        history.append(dict(epoch=epoch+1,training_loss=float(np.average(losses,weights=counts)),validation_loss=validation_loss,
                            batch_pair_counts=counts,learning_rate=float(optimizer.param_groups[0]["lr"]),
                            training_seconds=training_seconds,validation_seconds=time.monotonic()-validation_started,
                            training_samples_per_second=sum(counts)/training_seconds))
        scheduler.step();save_resume(epoch+1)
        _atomic_json(output/"status.json",dict(status="running",job_id=job["job_id"],stage=stage,epoch=epoch+1,max_epochs=maximum))
        if stage=="stage_a" and epoch+1>=int(budget["min_epochs"]) and stale>=int(budget["patience"]):break
    if not history:raise ValueError("no completed source epoch")
    provenance={**base,"stage":stage,"status":"completed_refit" if stage=="stage_b" else "completed_selection",
                "epochs_completed":len(history),"selected_epochs":best_epoch,"input_scalers":scalers,
                "scaler_fit_sample_ids":[inventory["samples"][index]["sample_id"] for index in training]}
    module.save_pretraining_checkpoint(model,output/("pretrained_checkpoint.pt" if stage=="stage_b" else "stage_a_checkpoint.pt"),
                                      provenance,optimizer=optimizer,scheduler=scheduler)
    result=dict(best_epoch=best_epoch,epochs_completed=len(history),history=history,training_sample_count=len(training),
                validation_sample_count=len(evaluation),initialization="fresh_random",phase_wall_seconds=time.monotonic()-phase_started,
                selection_policy="minimum epochs delays stopping; earlier best epoch remains eligible; fresh refit uses selected best epoch")
    _atomic_json(output/(stage+".json"),result)
    del model,optimizer,scheduler
    if device.startswith("cuda"):torch.cuda.empty_cache()
    return result


def input_audit(job,_module,device,output):
    """Read every requested native input; preserve partial support explicitly."""
    inventory=_json(job["inventory_path"])
    indices=job.get("sample_indices")
    if indices is None:
        indices=[index for index,row in enumerate(inventory["samples"]) if row["task_id"]==job["task"]]
    dataset=_dataset(inventory,indices,job)
    rows=[];nonfinite_valid=0;shape_failures=0;totals={}
    for sample in dataset:
        row=dict(sample_id=sample["sample_id"],subject_id=sample["subject_id"],record_id=sample["record_id"])
        for modality in ("eeg","fnirs"):
            values=np.asarray(sample[modality]);support=np.asarray(sample["channel_valid_mask"][modality],dtype=bool)
            if support.shape!=values.shape:
                shape_failures+=1;row[modality]=dict(shape_valid=False);continue
            bad=int((~np.isfinite(values)&support).sum());nonfinite_valid+=bad
            valid=int(support.sum());size=int(support.size)
            row[modality]=dict(shape_valid=True,valid_point_count=valid,total_point_count=size,
                               full_support=valid==size,nonfinite_valid_count=bad,
                               unsupported_channel_count=int((support.sum(axis=-1)==0).sum()))
            total=totals.setdefault(modality,dict(valid_point_count=0,total_point_count=0,full_support_sample_count=0))
            total["valid_point_count"]+=valid;total["total_point_count"]+=size
            total["full_support_sample_count"]+=int(valid==size)
        if "target_valid_mask" in sample:
            mask=np.asarray(sample["target_valid_mask"],dtype=bool)
            row["target"]=dict(valid_point_count=int(mask.sum()),total_point_count=int(mask.size))
        rows.append(row)
        if _STOP.is_set():raise InterruptedRun("input audit interrupted; no training was started")
    _atomic_json(output/"input_support.json",dict(task=job["task"],sample_count=len(rows),samples=rows))
    return dict(audit_pass=nonfinite_valid==0 and shape_failures==0,sample_count=len(rows),
                modalities=totals,nonfinite_valid_point_count=nonfinite_valid,shape_failure_count=shape_failures,
                full_support_sample_count=sum(all(row[key].get("full_support",False) for key in ("eeg","fnirs")) for row in rows),
                label_use="target support count only; no fitted transforms or model evaluation")


def run_job(job: dict[str,Any], device: str) -> dict[str,Any]:
    validate_job(job)
    output=Path(job["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    with (output/"worker.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_job_locked(job,device,output)


def _run_job_locked(job,device,output):
    result_path=output/"result.json"
    identity=_signature(job)
    if result_path.exists():
        retained=_json(result_path)
        if retained.get("status")=="completed":
            if retained.get("job_identity")!=identity:raise ValueError("successful output belongs to a different job")
            return retained
    retained_job=output/"job.json"
    if retained_job.exists() and _signature(_json(retained_job))!=identity:
        raise ValueError("output directory belongs to another job contract")
    _atomic_json(retained_job,job)
    _STOP.clear()
    for number in (signal.SIGTERM,signal.SIGINT):
        signal.signal(number,lambda *_args:_STOP.set())
    torch.set_num_threads(int(job.get("num_threads",1)))
    status=dict(status="running",job_id=job["job_id"],pid=os.getpid(),device=device,started_unix=time.time())
    _atomic_json(output/"status.json",status)
    with ResourceMonitor(device) as monitor:
        try:
            module=None if job["kind"]=="input_audit" else load_method(job)
            details={"evaluate":evaluate,"smoke":smoke,"pretrain":pretrain,"input_audit":input_audit}[job["kind"]](job,module,device,output)
            if device.startswith("cuda"):torch.cuda.synchronize(device)
            result=dict(schema=RESULT_SCHEMA,status="completed",job_identity=identity,job_id=job["job_id"],
                        method=job["method"],task=job["task"],track=job.get("track","lp"),**details)
        except BaseException as error:
            status_value="interrupted" if isinstance(error,InterruptedRun) else "failed"
            result=dict(schema=RESULT_SCHEMA,status=status_value,job_identity=identity,job_id=job["job_id"],
                        error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                        cuda_oom=isinstance(error,torch.cuda.OutOfMemoryError))
        resources=monitor.snapshot()
    _atomic_json(output/"resource_usage.json",resources)
    result["resources"]=resources
    if result["status"]!="completed":
        previous=list(output.glob("failure_*.json"))
        _atomic_json(output/f"failure_{len(previous)+1:03d}.json",result)
    _atomic_json(result_path,result)
    _atomic_json(output/"status.json",dict(status,status=result["status"],finished_unix=time.time()))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",required=True,type=Path)
    parser.add_argument("--device",required=True)
    arguments=parser.parse_args(argv)
    result=run_job(_json(arguments.job),arguments.device)
    print(json.dumps(dict(status=result["status"],job_id=result["job_id"],error=result.get("error"))))
    return 0 if result["status"]=="completed" else (75 if result["status"]=="interrupted" else 1)


if __name__=="__main__":
    raise SystemExit(main())
