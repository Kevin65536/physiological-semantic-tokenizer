#!/usr/bin/env python3
"""Durable resource-bounded DAG scheduler for downstream rebenchmark v1.

The immutable jobs.json is the execution contract. This controller owns only
scheduling, logs and resource observations; workers own scientific results and
checkpoints. No model or data modules are imported, including for --status.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

import psutil

MIB = 1024 ** 2
GIB = 1024 ** 3
TERMINAL = {"completed", "failed", "blocked", "interrupted"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: Any, name: str, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("schema") != "downstream_rebench_schedule_v1":
        raise ValueError("unsupported schedule schema")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("run_id is required")
    run_root = Path(manifest["run_root"])
    if not run_root.is_absolute():
        raise ValueError("run_root must be absolute")
    limits = manifest["limits"]
    _number(limits["cpu_threads"], "limits.cpu_threads", 1)
    _number(limits["ram_gib"], "limits.ram_gib", 0.001)
    if int(limits["cpu_threads"]) != limits["cpu_threads"]:
        raise ValueError("cpu_threads must be an integer")
    ids = limits.get("gpu_ids", [])
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("gpu_ids must contain distinct nonnegative integers")
    per_device = limits.get("max_gpu_jobs_per_device", 2)
    if type(per_device) is not int or per_device < 1:
        raise ValueError("max_gpu_jobs_per_device must be a positive integer")
    for gpu, reserve in limits.get("gpu_reserve_mib", {}).items():
        if str(gpu) not in {str(x) for x in ids}:
            raise ValueError(f"reserve for unknown GPU {gpu}")
        _number(reserve, f"GPU {gpu} reserve")
    jobs = manifest["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list")
    known: dict[str, dict[str, Any]] = {}
    outputs: set[str] = set()
    for job in jobs:
        job_id = job.get("id", "")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_id):
            raise ValueError(f"invalid job id {job_id!r}")
        if job_id in known:
            raise ValueError(f"duplicate job id {job_id}")
        known[job_id] = job
        command = job.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x or "\0" in x for x in command):
            raise ValueError(f"{job_id}: command must be a nonempty argv list")
        for field in ("cwd", "output_dir"):
            if not isinstance(job.get(field), str) or not Path(job[field]).is_absolute():
                raise ValueError(f"{job_id}: {field} must be absolute")
        if not Path(job["cwd"]).is_dir():
            raise ValueError(f"{job_id}: cwd does not exist")
        output = str(Path(job["output_dir"]).resolve())
        if output in outputs:
            raise ValueError("jobs must have distinct output_dir paths")
        outputs.add(output)
        resources = job["resources"]
        if type(resources.get("gpu")) is not bool:
            raise ValueError(f"{job_id}: resources.gpu must be boolean")
        cpu = _number(resources["cpu_threads"], f"{job_id}.cpu_threads", 1)
        ram = _number(resources["ram_gib"], f"{job_id}.ram_gib", 0.001)
        _number(resources.get("gpu_memory_mib", 0), f"{job_id}.gpu_memory_mib")
        if cpu != int(cpu) or cpu > limits["cpu_threads"] or ram > limits["ram_gib"]:
            raise ValueError(f"{job_id}: resources exceed controller limits")
        if resources["gpu"] and not ids:
            raise ValueError(f"{job_id}: GPU requested without configured GPUs")
        if resources["gpu"] and resources.get("gpu_memory_mib", 0) <= 0:
            raise ValueError(f"{job_id}: GPU jobs require a positive memory reservation")
        _number(job.get("timeout_seconds", 86400), f"{job_id}.timeout_seconds", 0.001)
        if not isinstance(job.get("dependencies", []), list):
            raise ValueError(f"{job_id}: dependencies must be a list")
        if "kind" in job and (not isinstance(job["kind"], str) or not job["kind"]):
            raise ValueError(f"{job_id}: kind must be a nonempty string")
        policy = job.get("dependency_policy", "completed")
        if policy not in {"completed", "terminal"}:
            raise ValueError(f"{job_id}: dependency_policy must be completed or terminal")
        if policy == "terminal" and job.get("kind") != "aggregate":
            raise ValueError(f"{job_id}: terminal dependency_policy is only allowed for aggregate jobs")
        if job.get("kind") == "aggregate" and resources["gpu"]:
            raise ValueError(f"{job_id}: aggregate jobs must use CPU resources")
    for job_id, job in known.items():
        deps = job.get("dependencies", [])
        if len(set(deps)) != len(deps) or any(x not in known or x == job_id for x in deps):
            raise ValueError(f"{job_id}: invalid or duplicate dependency")
        profile = job.get("profile_from")
        if profile is not None and profile not in deps:
            raise ValueError(f"{job_id}: profile_from must be an explicit dependency")
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(job_id: str) -> None:
        if job_id in visiting:
            raise ValueError("dependency cycle")
        if job_id in visited:
            return
        visiting.add(job_id)
        for dep in known[job_id].get("dependencies", []):
            visit(dep)
        visiting.remove(job_id)
        visited.add(job_id)
    for job_id in known:
        visit(job_id)
    return manifest, hashlib.sha256(payload).hexdigest()


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        if tail[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": tail[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def same_process(identity: dict[str, Any] | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


def gpu_snapshot() -> dict[int, dict[str, Any]]:
    """Observe physical GPUs and process allocations; missing observations fail closed."""
    try:
        answer = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        devices = {}
        for line in answer.stdout.splitlines():
            index, total, free = [int(float(x.strip())) for x in line.split(",")]
            devices[index] = {"total_mib": total, "free_mib": free, "process_mib": {}, "process_query_ok": False}
        answer = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        mapping = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        uuids = {line.split(",")[1].strip(): int(line.split(",")[0]) for line in mapping.stdout.splitlines()}
        for item in devices.values():
            item["process_query_ok"] = True
        for line in answer.stdout.splitlines():
            uuid, pid, used = [x.strip() for x in line.split(",")]
            if uuid in uuids and used not in {"[N/A]", "N/A"}:
                devices[uuids[uuid]]["process_mib"][int(pid)] = int(float(used))
        return devices
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return locals().get("devices", {})


def process_resources(identity: dict[str, Any] | None) -> tuple[int, list[int]]:
    if not same_process(identity):
        return 0, []
    try:
        parent = psutil.Process(identity["pid"])
        processes = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return 0, []
    rss, pids = 0, []
    for process in processes:
        try:
            rss += process.memory_info().rss
            pids.append(process.pid)
        except psutil.Error:
            pass
    return rss, pids


def measured_gpu_peak(output: Path) -> float | None:
    values: list[float] = []
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item > 0:
                    lowered = key.lower()
                    if ("peak" in lowered or "max" in lowered) and any(x in lowered for x in ("vram", "cuda", "gpu_memory")):
                        if "mib" in lowered or "_mb" in lowered:
                            values.append(float(item))
                        elif "gib" in lowered or "_gb" in lowered:
                            values.append(float(item) * 1024)
                        else:
                            values.append(float(item) / MIB)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    for name in ("resource_usage.json", "result.json", "summary.json", "scheduler_job_status.json"):
        data = read_json(output / name)
        if data:
            collect(data)
    return max(values) if values else None


def worker_terminal(output: Path) -> str | None:
    for name in ("status.json", "result.json"):
        value = read_json(output / name)
        state = value.get("status", value.get("state")) if value else None
        if state in {"completed", "failed", "interrupted"}:
            return state
    return None


def _source_constant(path: Path, name: str) -> Any:
    """Read a literal configuration constant without importing model code."""
    for statement in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError(f"{path}: missing literal {name}")


def build_schedule(config_path: Path, run_root: Path, source_root: Path,
                   smoke_summary: Path | None = None) -> dict[str, Any]:
    """Build immutable worker contracts from frozen metadata; never open signals."""
    import yaml

    config_path, run_root, source_root = config_path.resolve(), run_root.resolve(), source_root.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "downstream_rebench_execution_v1":
        raise ValueError("unsupported execution configuration schema")
    if config_path.parent.name != "comparative_methods":
        raise ValueError("build config must live in the owning comparative_methods directory")
    artifact = Path(config["artifact_root"])
    if artifact.is_absolute() or ".." in artifact.parts or tuple(run_root.parent.parts[-len(artifact.parts):]) != artifact.parts:
        raise ValueError("run_root must use the configured artifact_root/<run> namespace")
    project_root = run_root.parents[len(artifact.parts)]
    smoke_summary = (smoke_summary or run_root / "resources.json").resolve()
    reviewed = read_json(smoke_summary)
    if not reviewed:
        raise ValueError(f"reviewed --smoke-summary is required: {smoke_summary}")
    resources_by_method = reviewed.get("methods", reviewed)
    inventory_path, splits_path = run_root / "inventory.json", run_root / "splits.json"
    pretrain_inventory_path = run_root / "pretraining_inventory.json"
    inventory, splits, pretrain_inventory = (read_json(path) for path in (inventory_path, splits_path, pretrain_inventory_path))
    if not inventory or inventory.get("schema") != "downstream_rebench_inventory_v1":
        raise ValueError("frozen inventory.json is required")
    if not splits or splits.get("schema") != "downstream_rebench_subject_folds_v1":
        raise ValueError("frozen splits.json is required")
    if not pretrain_inventory or pretrain_inventory.get("inventory_kind") != "pretraining":
        raise ValueError("label-blind frozen pretraining_inventory.json is required")
    tasks = list(config["data"]["tasks"])
    outer_count, inner_count = int(config["split"]["outer_folds"]), int(config["split"]["inner_folds"])
    if (splits.get("outer_fold_count"), splits.get("inner_fold_count"), splits.get("seed")) != (outer_count, inner_count, config["split"]["seed"]):
        raise ValueError("frozen split counts/seed differ from the execution configuration")
    audit_dir = Path(config["data"].get("input_audit_dir", "input_audit_v2" if (run_root / "input_audit_v2").is_dir() else "input_audit_v1"))
    if not audit_dir.is_absolute():
        audit_dir = run_root / audit_dir
    audits, input_support = {}, {}
    for task in tasks:
        metadata = inventory["tasks"][task]
        folds = splits["tasks"][task]["folds"]
        if len(folds) != outer_count or any(len(fold["inner_folds"]) != inner_count for fold in folds):
            raise ValueError(f"{task}: incomplete frozen outer/inner fold grid")
        audit_path = audit_dir / task / "result.json"
        audit = read_json(audit_path)
        if (not audit or audit.get("status") != "completed" or audit.get("audit_pass") is not True
                or audit.get("sample_count") != metadata["sample_count"]
                or audit.get("shape_failure_count") != 0 or audit.get("nonfinite_valid_point_count") != 0):
            raise ValueError(f"{task}: completed passing input audit for the whole frozen inventory is required")
        if any(audit.get("modalities", {}).get(modality, {}).get("total_point_count", 0) <= 0 for modality in ("eeg", "fnirs")):
            raise ValueError(f"{task}: input audit must inspect nonempty EEG and fNIRS inputs")
        supported = audit.get("full_support_sample_count")
        if type(supported) is not int or not 0 <= supported <= metadata["sample_count"]:
            raise ValueError(f"{task}: full/partial input support counts are required")
        if supported < metadata["sample_count"] and config["data"]["missing_inputs"] != "retain_masks_no_method_specific_deletion":
            raise ValueError(f"{task}: partial inputs require the declared mask-retention contract")
        audits[task] = str(audit_path)
        input_support[task] = {"audit_path": str(audit_path), "sample_count": metadata["sample_count"],
                               "full_support_sample_count": supported, "partial_support_sample_count": metadata["sample_count"] - supported,
                               "modalities": audit["modalities"], "policy": config["data"]["missing_inputs"], "dropped_sample_count": 0}
    worker_path = source_root / "comparative_methods/downstream_rebench_worker_v1.py"
    aggregate_path = source_root / "comparative_methods/aggregate_downstream_rebench_v1.py"
    if not worker_path.is_file() or not aggregate_path.is_file():
        raise ValueError("frozen source_root must contain the worker and aggregate entrypoints")
    method_dirs = _source_constant(worker_path, "METHOD_DIRS")
    deterministic = list(config["split"]["deterministic_seeds"])
    final_seeds = list(config["split"]["final_seeds"])
    tuning_seed = int(config["split"]["tuning_seed"])
    if deterministic != [tuning_seed] or final_seeds[0] != tuning_seed:
        raise ValueError("deterministic/tuning seed must be the first final seed")
    if list(config["pretraining"]["seeds"]) != final_seeds:
        raise ValueError("EFRM pretraining must have its own backbone for every final seed")
    nodes: list[dict[str, Any]] = []
    contracts: dict[Path, dict[str, Any]] = {}
    method_paths: dict[str, Path] = {}
    timeout = 365 * 24 * 3600
    training = {"max_epochs": config["training"]["max_epochs"],
                "min_epochs": config["training"]["min_epochs_before_early_stop"], "patience": config["training"]["patience"]}
    if tuple(training.values()) != (200, 50, 30) or int(config["training"]["effective_batch_size"]) != 32:
        raise ValueError("formal downstream budget is effective batch32/max200/min50/patience30")

    def resource_spec(method: str, *, pretraining: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        measured = reviewed.get("pretraining", resources_by_method.get("efrm_pretrain")) if pretraining else None
        measured = measured or resources_by_method.get(method)
        if not isinstance(measured, dict):
            raise ValueError(f"{method}: reviewed smoke resources are missing")
        spec = {"gpu": measured["gpu_memory_mib"] > 0, "gpu_memory_mib": measured["gpu_memory_mib"],
                "ram_gib": measured["ram_gib"], "cpu_threads": measured["cpu_threads"]}
        for key in ("gpu_memory_mib", "ram_gib", "cpu_threads", "microbatch_size"):
            _number(measured[key], f"{method}.{key}", 0 if key == "gpu_memory_mib" else 1 if key in {"cpu_threads", "microbatch_size"} else .001)
        if measured["microbatch_size"] != int(measured["microbatch_size"]):
            raise ValueError(f"{method}: microbatch_size must be an integer")
        return spec, measured

    def module_path(method: str) -> Path:
        if method not in method_paths:
            relative = config["methods"][method].get("source")
            path = source_root / (relative or f"comparative_methods/{method_dirs[method]}/rebench_v1.py")
            if not path.is_file():
                raise ValueError(f"missing frozen method source: {path}")
            method_paths[method] = path
        return method_paths[method]

    def owner_output(method: str) -> Path:
        return project_root / "comparative_methods" / method_dirs[method] / "runs/downstream_rebench_v1" / run_root.name

    def add(job: dict[str, Any], resources: dict[str, Any], dependencies: list[str]) -> None:
        path = run_root / "jobs" / f"{job['job_id']}.json"
        contracts[path] = job
        nodes.append({"id": job["job_id"], "kind": job["kind"],
                      "timing_kind": f"{job['method']}_{job['track']}_{job['task']}_{job['kind']}",
                      "job_path": str(path), "method": job["method"], "task": job["task"], "track": job["track"],
                      "seed": job["seed"], **({"outer_fold": job["outer_fold"]} if "outer_fold" in job else {}),
                      "command": [sys.executable, str(worker_path), "--job", str(path), "--device", "{device}"],
                      "cwd": str(source_root), "output_dir": job["output_dir"], "dependencies": dependencies,
                      "resources": resources, "timeout_seconds": timeout})

    def base_job(method: str, task: str, track: str, seed: int, job_id: str,
                 output: Path, micro: int, resources: dict[str, Any]) -> dict[str, Any]:
        if not 1 <= micro <= 32:
            raise ValueError(f"{method}: microbatch_size must be between 1 and effective batch32")
        return {"schema": "downstream_rebench_job_v1", "job_id": job_id, "kind": "evaluate",
                "method": method, "task": task, "track": track, "seed": seed,
                "inventory_path": str(inventory_path), "splits_path": str(splits_path), "output_dir": str(output),
                "batch_size": 32, "microbatch_size": micro, "gradient_accumulation": math.ceil(32 / micro),
                "num_threads": resources["cpu_threads"], "data_workers": 0, "precision": "bf16",
                "input_support_policy": config["data"]["missing_inputs"], "sample_filtering": "none_whole_frozen_inventory",
                "modality_profile": config["methods"][method].get("profile", "multimodal"),
                "method_profile": {"module_path": str(module_path(method))}, "training": dict(training)}

    def search_spec(method: str, task: str, track: str) -> dict[str, Any]:
        if track == "lp":
            key = "alpha" if task == "refed_regression" else "C"
            grid = config["training"]["ridge_alpha" if key == "alpha" else "linear_C"]
            return {"source": str(config_path), "candidates": [{key: value} for value in grid]}
        method_config = config["methods"][method]
        if method_config.get("search") == "method_local_DEFAULT_SEARCH":
            return {"source": str(module_path(method)), "candidates": list(_source_constant(module_path(method), "DEFAULT_SEARCH"))}
        specification = method_config["ft_search"]
        fixed = {key: value for key, value in specification.items() if key not in {"learning_rates", "weight_decays", "source", "deviation"}}
        return {"source": str(source_root / specification["source"]), "deviation": specification.get("deviation"),
                "candidates": [{**fixed, "learning_rate": rate, "weight_decay": decay}
                               for rate in specification["learning_rates"] for decay in specification["weight_decays"]]}

    # Construct contracts first; dispatch order is fixed below for GPU utilization.
    baseline = config["methods"]["signal_linear"]
    baseline_resources, baseline_measured = resource_spec("signal_linear")
    for modality in baseline["modalities"]:
        for task in tasks:
            for fold in range(outer_count):
                for seed in deterministic:
                    job_id = f"signal_linear-{modality}-lp-{task}-outer{fold}-seed{seed}"
                    output = run_root / "baselines" / modality / task / f"outer{fold}" / f"seed{seed}"
                    job = base_job("signal_linear", task, "lp", seed, job_id, output, int(baseline_measured["microbatch_size"]), baseline_resources)
                    job.update(outer_fold=fold, search=search_spec("signal_linear", task, "lp"),
                               modality_profile="eeg_only" if modality == "eeg" else "multimodal")
                    job["method_profile"]["baseline_modality"] = modality
                    add(job, dict(baseline_resources), [])

    pretrain_ids: dict[tuple[str, int], str] = {}
    pretrain_outputs: dict[tuple[str, int], Path] = {}
    representatives = [task for task in ("motor_imagery", "wg", "visual", "refed_regression") if task in tasks]
    datasets = {inventory["tasks"][task]["dataset_id"] for task in tasks}
    if {inventory["tasks"][task]["dataset_id"] for task in representatives} != datasets or len(datasets) != 4:
        raise ValueError("pretraining requires one representative alias for each of the four target datasets")
    pretraining_resources, pretraining_measured = resource_spec("efrm", pretraining=True)
    for task in representatives:
        dataset = inventory["tasks"][task]["dataset_id"]
        if pretrain_inventory["tasks"][task]["dataset_id"] != dataset:
            raise ValueError("pretraining and downstream dataset identities differ")
        for seed in config["pretraining"]["seeds"]:
            job_id = f"efrm-pretrain-exclude_{dataset}-seed{seed}"
            output = owner_output("efrm") / "pretrain" / f"exclude_{dataset}" / f"seed{seed}"
            job = base_job("efrm", task, "ft", seed, job_id, output, int(pretraining_measured["microbatch_size"]), pretraining_resources)
            job.update(kind="pretrain", pretrain_inventory_path=str(pretrain_inventory_path), pretraining=dict(config["pretraining"]))
            job.pop("training")
            pretrain_ids[(dataset, seed)], pretrain_outputs[(dataset, seed)] = job_id, output
            add(job, dict(pretraining_resources), [])

    unsupported = []
    for method, method_config in config["methods"].items():
        if method == "signal_linear":
            continue
        supported = tasks if method_config["tasks"] == "all" else [task for task in tasks if task != "refed_regression"] if method_config["tasks"] == "classification" else method_config["tasks"]
        unsupported.extend({"method": method, "task": task, "tracks": method_config["tracks"],
                            "reason": "continuous output adapter is not validated" if task == "refed_regression" else "event-level task adapter is not validated"}
                           for task in tasks if task not in supported)
    # LP and supervised work precede FT; dependencies remain explicit for EFRM.
    for track in ("lp", "supervised", "ft"):
        for method, method_config in config["methods"].items():
            if method == "signal_linear" or track not in method_config["tracks"]:
                continue
            supported = tasks if method_config["tasks"] == "all" else [task for task in tasks if task != "refed_regression"] if method_config["tasks"] == "classification" else method_config["tasks"]
            resources, measured = resource_spec(method)
            seeds = deterministic if track == "lp" and method != "efrm" else final_seeds
            micro = int(measured.get("lp_microbatch_size", 2 if method == "normwear" else 32)) if track == "lp" else int(measured["microbatch_size"])
            if micro < 1 or micro > 32:
                raise ValueError(f"{method}: microbatch_size must be between 1 and effective batch32")
            for task in supported:
                for fold in range(outer_count):
                    for seed in seeds:
                        job_id = f"{method}-{track}-{task}-outer{fold}-seed{seed}"
                        output = owner_output(method) / track / task / f"outer{fold}" / f"seed{seed}"
                        job = base_job(method, task, track, seed, job_id, output, micro, resources)
                        job.update(outer_fold=fold, search=search_spec(method, task, track))
                        dependencies = []
                        if method == "efrm":
                            key = (inventory["tasks"][task]["dataset_id"], seed)
                            dependencies.append(pretrain_ids[key])
                            job["method_profile"]["pretrained_checkpoint"] = str(pretrain_outputs[key] / "pretrained_checkpoint.pt")
                        elif seed != tuning_seed:
                            dependencies.append(f"{method}-{track}-{task}-outer{fold}-seed{tuning_seed}")
                            job["reuse_selection_from"] = str(output.parent / f"seed{tuning_seed}")
                        if method == "reve":
                            overlap = task in method_config.get("overlap_tasks", [])
                            job["target_corpus_overlap"] = "confirmed_Shin2017A_overlap" if overlap else "none_in_published_exhaustive_list"
                            job["reporting_track"] = "open_world_pretrained_with_target_corpus_overlap" if overlap else "official_pretrained"
                        add(job, dict(resources), dependencies)

    # Start long GPU dependency chains before CPU references consume the thread
    # budget. Stable sorting preserves seed17 before its selection consumers.
    nodes.sort(key=lambda node: 0 if node["kind"] == "pretrain" else
               1 if node["track"] == "lp" and node["resources"]["gpu"] else
               2 if node["method"] == "signal_linear" else
               3 if node["track"] == "supervised" else 4)
    nodes.append({"id": "aggregate", "kind": "aggregate", "dependency_policy": "terminal",
                  "command": [sys.executable, str(aggregate_path), "--schedule", str(run_root / "jobs.json"), "--output", str(run_root / "aggregate")],
                  "cwd": str(source_root), "output_dir": str(run_root / "aggregate"),
                  "dependencies": [node["id"] for node in nodes],
                  "resources": {"gpu": False, "gpu_memory_mib": 0, "ram_gib": 8, "cpu_threads": 2}, "timeout_seconds": timeout})
    settings = config["resources"]
    manifest = {"schema": "downstream_rebench_schedule_v1", "run_id": f"{config['experiment_id']}-{run_root.name}", "run_root": str(run_root),
                "source_root": str(source_root), "config_path": str(config_path), "smoke_summary_path": str(smoke_summary),
                "inventory_path": str(inventory_path), "splits_path": str(splits_path), "input_audits": audits, "input_support": input_support,
                "unsupported_cells": unsupported,
                "pretrained_overlap": {"reve": {"tasks": config["methods"].get("reve", {}).get("overlap_tasks", []), "status": "confirmed_Shin2017A_overlap"}},
                "limits": {"cpu_threads": settings["cpu_thread_budget"], "ram_gib": settings["ram_budget_gib"], "gpu_ids": settings["gpus"],
                           "gpu_reserve_mib": {str(key): value for key, value in settings["gpu_reserve_mib"].items()},
                           "max_gpu_jobs_per_device": settings.get("max_gpu_jobs_per_device", 2)}, "jobs": nodes}
    manifest_path = run_root / "jobs.json"
    for node in nodes:
        request = node["resources"]
        if (request["cpu_threads"] != int(request["cpu_threads"]) or request["cpu_threads"] > manifest["limits"]["cpu_threads"]
                or request["ram_gib"] > manifest["limits"]["ram_gib"] or (request["gpu"] and request["gpu_memory_mib"] <= 0)):
            raise ValueError(f"{node['id']}: reviewed resources do not fit the controller limits")
    planned = {**contracts, manifest_path: manifest}
    for path, value in planned.items():
        if path.exists() and read_json(path) != value:
            raise ValueError(f"refusing to overwrite a different retained contract: {path}")
    for path, value in planned.items():
        if not path.exists():
            atomic_json(path, value)
    load_manifest(manifest_path)
    return manifest


class Scheduler:
    def __init__(self, manifest_path: Path, *, poll_seconds: float = 2, stop_grace_seconds: float = 30):
        self.manifest_path = manifest_path.resolve()
        self.manifest, self.manifest_hash = load_manifest(self.manifest_path)
        self.root = Path(self.manifest["run_root"])
        self.jobs = {job["id"]: job for job in self.manifest["jobs"]}
        self.limits = self.manifest["limits"]
        self.poll_seconds = poll_seconds
        self.stop_grace_seconds = stop_grace_seconds
        self.active: dict[str, dict[str, Any]] = {}
        self.stop_signal: int | None = None
        self.records: dict[str, dict[str, Any]] = {}
        self.started_at = utc_now()
        self.started_epoch = time.time()
        self.prior_elapsed = 0.0
        self.max_active = 0
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.lock_stream: Any = None

    def event(self, event: str, **fields: Any) -> None:
        with (self.root / "scheduler_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": utc_now(), "event": event, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_stream = (self.root / "scheduler.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_stream.close()
            self.lock_stream = None
            raise RuntimeError("another scheduler already holds this run's controller lock")

    def restore(self) -> None:
        previous = read_json(self.root / "scheduler_status.json")
        if previous:
            if previous.get("manifest_sha256") != self.manifest_hash:
                raise ValueError("manifest changed since this run began; use a new run_root")
            self.prior_elapsed = float(previous.get("elapsed_seconds", 0))
            self.started_at = previous["started_at"]
            self.max_active = int(previous.get("max_active_jobs", 0))
        for job_id, job in self.jobs.items():
            record = dict((previous or {}).get("jobs", {}).get(job_id, {}))
            if not record:
                record = {"id": job_id, "status": "pending", "attempt": 0}
            if record["status"] == "running":
                if same_process(record.get("process_identity")):
                    self.active[job_id] = {"process": None, "logs": [], "record": record}
                    self.event("job_adopted", job_id=job_id, pid=record["process_identity"]["pid"])
                else:
                    outcome = worker_terminal(Path(job["output_dir"]))
                    if outcome in {"completed", "failed"}:
                        record.update(status=outcome, finished_at=utc_now(), reason="recovered_worker_terminal_evidence")
                        self.event("job_recovered", job_id=job_id, status=outcome)
                    else:
                        record.update(status="pending", reason="resume_after_interruption")
                        self.event("job_resumed", job_id=job_id, previous_status="running")
            elif record["status"] == "interrupted":
                # A live process after an unclean stop is adopted before any retry.
                if same_process(record.get("process_identity")):
                    record["status"] = "running"
                    self.active[job_id] = {"process": None, "logs": [], "record": record}
                    self.event("job_adopted", job_id=job_id, pid=record["process_identity"]["pid"])
                else:
                    record.update(status="pending", reason="resume_after_interruption")
                    self.event("job_resumed", job_id=job_id, previous_status="interrupted")
            self.records[job_id] = record

    def request(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        request = dict(job["resources"])
        if request["gpu"] and job.get("profile_from"):
            source = Path(self.jobs[job["profile_from"]]["output_dir"])
            peak = measured_gpu_peak(source)
            if peak is None:
                raise ValueError(f"profile {job['profile_from']} has no positive measured GPU memory peak")
            # Never reduce an explicit static safety budget after a small smoke.
            request["gpu_memory_mib"] = max(float(request["gpu_memory_mib"]), math.ceil(peak * 1.3 + 512))
            request["profile_peak_mib"] = peak
        return request

    def observe(self) -> None:
        self.snapshots = gpu_snapshot() if self.limits.get("gpu_ids") else {}
        for job_id, item in self.active.items():
            record = item["record"]
            rss, pids = process_resources(record.get("process_identity"))
            record["rss_bytes"] = rss
            record["peak_rss_bytes"] = max(record.get("peak_rss_bytes", 0), rss)
            record["pids"] = pids
            gpu = record.get("gpu_id")
            allocated = sum(self.snapshots.get(gpu, {}).get("process_mib", {}).get(pid, 0) for pid in pids)
            record["current_gpu_memory_mib"] = allocated
            record["peak_vram_bytes"] = max(record.get("peak_vram_bytes", 0), allocated * MIB)
            record["elapsed_seconds"] = max(0, time.time() - record["started_epoch"])

    def fit(self, request: dict[str, Any]) -> tuple[bool, int | None, str]:
        cpu = 0.0
        reserved_ram = 0.0
        unallocated_ram = 0.0
        for item in self.active.values():
            record = item["record"]
            resources = record["reserved_resources"]
            cpu += resources["cpu_threads"]
            rss_gib = record.get("rss_bytes", 0) / GIB
            reserved_ram += max(resources["ram_gib"], rss_gib)
            unallocated_ram += max(0, resources["ram_gib"] - rss_gib)
        if cpu + request["cpu_threads"] > self.limits["cpu_threads"]:
            return False, None, "cpu_threads"
        if reserved_ram + request["ram_gib"] > self.limits["ram_gib"]:
            return False, None, "reserved_ram"
        if request["ram_gib"] + unallocated_ram > psutil.virtual_memory().available / GIB:
            return False, None, "available_ram"
        if not request["gpu"]:
            return True, None, "fit"
        candidates = []
        for gpu in self.limits["gpu_ids"]:
            observed = self.snapshots.get(gpu)
            if not observed:
                continue
            active = [x["record"] for x in self.active.values() if x["record"].get("gpu_id") == gpu]
            if len(active) >= self.limits.get("max_gpu_jobs_per_device", 2):
                continue
            reserve = self.limits.get("gpu_reserve_mib", {}).get(str(gpu), 0)
            pending_allocation = sum(max(0, x["reserved_resources"]["gpu_memory_mib"] - x.get("current_gpu_memory_mib", 0)) for x in active)
            available = observed["free_mib"] - reserve - pending_allocation
            if available >= request["gpu_memory_mib"]:
                candidates.append((available, gpu))
        if candidates:
            # Prefer the most available physical device without changing any other process.
            return True, max(candidates)[1], "fit"
        return False, None, "available_gpu_memory_or_slots"

    def launch(self, job_id: str, resources: dict[str, Any], gpu: int | None) -> None:
        job = self.jobs[job_id]
        output = Path(job["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        device = f"cuda:{gpu}" if gpu is not None else "cpu"
        command = [arg.replace("{device}", device) for arg in job["command"]]
        record = self.records[job_id]
        stdout = (output / "scheduler_stdout.log").open("ab", buffering=0)
        stderr = (output / "scheduler_stderr.log").open("ab", buffering=0)
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(resources["cpu_threads"])
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(command, cwd=job["cwd"], env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        except Exception as error:
            stdout.close()
            stderr.close()
            record.update(status="failed", reason=f"launch_error: {error}", finished_at=utc_now())
            self.event("job_failed", job_id=job_id, reason=record["reason"])
            return
        record.update(status="running", attempt=record.get("attempt", 0) + 1,
                      started_at=utc_now(), started_epoch=time.time(), process_identity=process_identity(process.pid),
                      pid=process.pid, gpu_id=gpu, device=device, reserved_resources=resources, command=command,
                      returncode=None, reason=None, peak_rss_bytes=0, peak_vram_bytes=0,
                      rss_bytes=0, current_gpu_memory_mib=0)
        self.active[job_id] = {"process": process, "logs": [stdout, stderr], "record": record}
        self.max_active = max(self.max_active, len(self.active))
        self.event("job_started", job_id=job_id, attempt=record["attempt"], pid=process.pid, device=device, resources=resources)
        self.write_job(job_id)
        # Persist each launch before a later GPU query or another launch.
        self.persist("running")

    def write_job(self, job_id: str) -> None:
        atomic_json(Path(self.jobs[job_id]["output_dir"]) / "scheduler_job_status.json", self.records[job_id])

    def finish(self, job_id: str, status: str, *, returncode: int | None = None, reason: str | None = None) -> None:
        item = self.active.pop(job_id)
        for stream in item["logs"]:
            stream.close()
        record = item["record"]
        record.update(status=status, finished_at=utc_now(), returncode=returncode, reason=reason,
                      elapsed_seconds=max(0, time.time() - record["started_epoch"]), rss_bytes=0,
                      current_gpu_memory_mib=0)
        self.write_job(job_id)
        self.event("job_finished", job_id=job_id, status=status, returncode=returncode, reason=reason,
                   elapsed_seconds=record["elapsed_seconds"])

    def terminate(self, job_id: str) -> None:
        item = self.active[job_id]
        identity = item["record"].get("process_identity")
        if same_process(identity):
            try:
                os.killpg(identity["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass

    def reap(self) -> None:
        for job_id, item in list(self.active.items()):
            record = item["record"]
            process = item["process"]
            returncode = process.poll() if process is not None else None
            alive = returncode is None and same_process(record.get("process_identity"))
            if alive and time.time() - record["started_epoch"] > self.jobs[job_id].get("timeout_seconds", 86400):
                self.terminate(job_id)
                record.setdefault("timeout_started_epoch", time.time())
                if time.time() - record["timeout_started_epoch"] >= self.stop_grace_seconds:
                    try:
                        os.killpg(record["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                continue
            if alive:
                continue
            if process is not None:
                returncode = process.wait()
            worker_state = worker_terminal(Path(self.jobs[job_id]["output_dir"]))
            if record.get("timeout_started_epoch"):
                self.finish(job_id, "failed", returncode=returncode, reason="timeout")
            elif (returncode == 0 and worker_state not in {"failed", "interrupted"}) or (process is None and worker_state == "completed"):
                self.finish(job_id, "completed", returncode=returncode)
            elif returncode == 75 or (process is None and worker_state == "interrupted"):
                self.finish(job_id, "interrupted", returncode=returncode, reason="worker_interrupted_resume_on_controller_restart")
            elif process is None and worker_state is None:
                self.finish(job_id, "interrupted", reason="orphan_exited_without_terminal_evidence")
            else:
                self.finish(job_id, "failed", returncode=returncode, reason="worker_failed")

    def eta(self) -> float | None:
        samples: dict[str, list[float]] = collections.defaultdict(list)
        for job_id, record in self.records.items():
            if record["status"] == "completed":
                kind = self.jobs[job_id].get("timing_kind", self.jobs[job_id].get("kind", job_id))
                samples[kind].append(record.get("elapsed_seconds", 0))
        remaining = []
        for job_id, record in self.records.items():
            if record["status"] not in {"pending", "running"}:
                continue
            values = samples[self.jobs[job_id].get("timing_kind", self.jobs[job_id].get("kind", job_id))]
            if len(values) < 2:
                return None
            remaining.append(max(0, sum(values) / len(values) - (record.get("elapsed_seconds", 0) if record["status"] == "running" else 0)))
        return sum(remaining) / max(1, self.max_active) if remaining else 0.0

    def persist(self, status: str) -> None:
        counts = dict(collections.Counter(x["status"] for x in self.records.values()))
        progress = {"total": len(self.jobs), "counts": counts,
                    "terminal": sum(counts.get(x, 0) for x in TERMINAL),
                    "completed_fraction": counts.get("completed", 0) / len(self.jobs)}
        value = {"schema": "downstream_rebench_scheduler_status_v1", "run_id": self.manifest["run_id"],
                 "status": status, "manifest_path": str(self.manifest_path), "manifest_sha256": self.manifest_hash,
                 "controller_identity": process_identity(os.getpid()), "started_at": self.started_at,
                 "updated_at": utc_now(), "elapsed_seconds": self.prior_elapsed + time.time() - self.started_epoch,
                 "eta_seconds": self.eta(), "eta_basis": "at_least_two_completed_jobs_per_remaining_kind; max_observed_concurrency",
                 "progress": progress, "max_active_jobs": self.max_active,
                 "gpu_observation": self.snapshots, "jobs": self.records}
        atomic_json(self.root / "scheduler_status.json", value)
        atomic_json(self.root / "progress.json", {key: value[key] for key in ("run_id", "status", "updated_at", "elapsed_seconds", "eta_seconds", "progress", "max_active_jobs", "gpu_observation")})
        for job_id in self.active:
            self.write_job(job_id)

    def stop(self) -> None:
        for job_id in list(self.active):
            self.terminate(job_id)
        deadline = time.monotonic() + self.stop_grace_seconds
        while self.active and time.monotonic() < deadline:
            for job_id, item in list(self.active.items()):
                process = item["process"]
                if process is not None:
                    process.poll()
                if not same_process(item["record"].get("process_identity")):
                    if process is not None:
                        process.wait()
                    terminal = worker_terminal(Path(self.jobs[job_id]["output_dir"]))
                    if terminal == "completed":
                        self.finish(job_id, "completed", returncode=0, reason="completed_during_shutdown")
                    elif terminal == "failed":
                        self.finish(job_id, "failed", returncode=process.returncode if process is not None else None,
                                    reason="worker_failed_during_shutdown")
                    else:
                        self.finish(job_id, "interrupted", reason="controller_signal")
            if self.active:
                self.observe()
                self.persist("stopping")
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        for job_id, item in list(self.active.items()):
            if same_process(item["record"].get("process_identity")):
                try:
                    os.killpg(item["record"]["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if item["process"] is not None:
                item["process"].wait()
            self.finish(job_id, "interrupted", reason="controller_signal_grace_expired")
        self.persist("interrupted")

    def run(self) -> int:
        self.acquire()
        previous_handlers = {}
        try:
            self.restore()
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, lambda number, frame: setattr(self, "stop_signal", number))
            self.event("controller_started", pid=os.getpid(), manifest_sha256=self.manifest_hash)
            while not self.stop_signal:
                self.observe()
                self.reap()
                for job_id, record in self.records.items():
                    if self.stop_signal:
                        break
                    if record["status"] != "pending":
                        continue
                    dependencies = self.jobs[job_id].get("dependencies", [])
                    policy = self.jobs[job_id].get("dependency_policy", "completed")
                    failed = [dep for dep in dependencies if self.records[dep]["status"] in TERMINAL - {"completed"}] if policy == "completed" else []
                    if failed:
                        record.update(status="blocked", reason="dependency_not_completed", blocked_by=failed, finished_at=utc_now())
                        self.event("job_blocked", job_id=job_id, blocked_by=failed)
                        self.write_job(job_id)
                        continue
                    allowed = {"completed", "failed", "blocked"} if policy == "terminal" else {"completed"}
                    if not all(self.records[dep]["status"] in allowed for dep in dependencies):
                        record["waiting_for"] = "dependencies"
                        continue
                    try:
                        request = self.request(job_id)
                    except ValueError as error:
                        record.update(status="failed", reason=str(error), finished_at=utc_now())
                        self.event("job_failed", job_id=job_id, reason=str(error))
                        self.write_job(job_id)
                        continue
                    fits, gpu, reason = self.fit(request)
                    record["waiting_for"] = None if fits else reason
                    if fits:
                        self.launch(job_id, request, gpu)
                terminal = all(x["status"] in TERMINAL for x in self.records.values())
                if terminal:
                    status = "completed" if all(x["status"] == "completed" for x in self.records.values()) else "failed"
                    self.persist(status)
                    self.event("controller_finished", status=status)
                    return 0 if status == "completed" else 1
                self.persist("running")
                time.sleep(self.poll_seconds)
            self.event("controller_stopping", signal=self.stop_signal)
            self.stop()
            return 128 + self.stop_signal
        except BaseException:
            if self.active:
                self.stop()
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if self.lock_stream is not None:
                self.lock_stream.close()
                self.lock_stream = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--build-config", type=Path, help="build immutable job contracts from frozen metadata without running them")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--status", action="store_true", help="read controller status without importing workers or acquiring its lock")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--stop-grace-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.build_config:
            if not args.run_root or not args.source_root or args.status:
                raise ValueError("build mode requires --run-root and --source-root and cannot use --status")
            manifest = build_schedule(args.build_config, args.run_root, args.source_root, args.smoke_summary)
            print(json.dumps({"manifest": str(args.run_root.resolve() / "jobs.json"),
                              "job_counts": dict(collections.Counter(job["kind"] for job in manifest["jobs"])),
                              "unsupported_cell_count": len(manifest["unsupported_cells"])}))
            return 0
        if args.status:
            manifest, manifest_hash = load_manifest(args.manifest)
            status = read_json(Path(manifest["run_root"]) / "scheduler_status.json")
            if not status:
                print(json.dumps({"status": "not_started", "run_id": manifest["run_id"]}))
                return 0
            if status.get("manifest_sha256") != manifest_hash:
                raise ValueError("manifest differs from retained controller state")
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        if args.poll_seconds <= 0 or args.stop_grace_seconds < 0:
            raise ValueError("poll must be positive and stop grace nonnegative")
        return Scheduler(args.manifest, poll_seconds=args.poll_seconds, stop_grace_seconds=args.stop_grace_seconds).run()
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as error:
        print(f"scheduler: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
