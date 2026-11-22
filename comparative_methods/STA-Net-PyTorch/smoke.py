#!/usr/bin/env python3
"""Run correctness-only smoke jobs for the PyTorch STA-Net task variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from sta_net_pytorch import (
    STANetUnifiedTaskDataset,
    collate_sta_net,
    get_sta_net_task_spec,
    task_contract_sha256,
)
from sta_net_pytorch import STANet, STANetConfig, STANetObjective


RUN_SCHEMA = "sta_net_pytorch_smoke_v1"
UPSTREAM_URL = "https://github.com/MutianLiu-SHU/STA-Net"
UPSTREAM_REVISION = "b6db8bb5eb2f6491a13f0938880ee70e32162ee7"
PAPER_DOI = "10.1016/j.inffus.2025.103023"


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist() if value.numel() != 1 else value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _resolve_device(value: str) -> torch.device:
    requested = value
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for STA-Net smoke but unavailable")
    return torch.device(requested)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _select_smoke_indices(
    dataset: STANetUnifiedTaskDataset,
    *,
    train_subject_count: int,
    samples_per_split: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    metadata = [dataset.lightweight_metadata(index) for index in range(len(dataset))]
    subjects = sorted({row["subject"] for row in metadata})
    if len(subjects) < train_subject_count + 1:
        raise RuntimeError(f"Task {dataset.spec.key} lacks enough subjects for disjoint smoke train/validation")
    train_subjects = subjects[:train_subject_count]
    validation_subjects = [subjects[train_subject_count]]

    def select(subject_pool: Sequence[str]) -> list[int]:
        eligible = [index for index, row in enumerate(metadata) if row["subject"] in subject_pool]
        if dataset.spec.task_type == "classification":
            chosen = []
            for class_name in dataset.spec.class_names:
                match = next((index for index in eligible if metadata[index]["condition"] == class_name), None)
                if match is not None:
                    chosen.append(match)
            for index in eligible:
                if len(chosen) >= samples_per_split:
                    break
                if index not in chosen:
                    chosen.append(index)
            return chosen[: max(samples_per_split, len(dataset.spec.class_names))]
        return eligible[:samples_per_split]

    train_indices = select(train_subjects)
    validation_indices = select(validation_subjects)
    if not train_indices or not validation_indices:
        raise RuntimeError(f"Smoke split for {dataset.spec.key} selected an empty partition")
    split = {
        "schema": "sta_net_smoke_split_v1",
        "formal_benchmark_split": False,
        "protected_test_opened": False,
        "selection": "first lexicographic subjects and deterministic class coverage; correctness only",
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "train_dataset_indices": train_indices,
        "validation_dataset_indices": validation_indices,
        "train_metadata": [metadata[index] for index in train_indices],
        "validation_metadata": [metadata[index] for index in validation_indices],
    }
    return train_indices, validation_indices, split


def _prediction_metrics(task_type: str, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if task_type == "classification":
        predicted = prediction.argmax(dim=-1)
        return {"accuracy": float((predicted == target).float().mean().detach())}
    weights = mask.to(dtype=prediction.dtype)
    error = prediction - target
    denominator = weights.sum().clamp_min(1.0)
    return {
        "masked_mae_native_coordinate": float((error.abs() * weights).sum().detach() / denominator),
        "masked_rmse_native_coordinate": float(
            torch.sqrt((error.square() * weights).sum().detach() / denominator)
        ),
        "target_valid_fraction": float(weights.mean().detach()),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
    }


def _run_task(
    *,
    task_key: str,
    config: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    task_start = time.perf_counter()
    spec = get_sta_net_task_spec(task_key)
    task_dir = output_root / task_key
    for relative in ("checkpoints", "metrics", "predictions"):
        (task_dir / relative).mkdir(parents=True, exist_ok=True)

    data_config = config.get("data", {})
    smoke_config = config.get("smoke", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    dataset = STANetUnifiedTaskDataset(spec, cache_root=str(data_config.get("cache_root", DEFAULT_CLEAN_CACHE_ROOT)))
    train_indices, validation_indices, split_manifest = _select_smoke_indices(
        dataset,
        train_subject_count=int(smoke_config.get("train_subject_count", 2)),
        samples_per_split=int(smoke_config.get("samples_per_split", 2)),
    )
    _write_json(task_dir / "split_manifest.json", split_manifest)

    train_batch = collate_sta_net([dataset[index] for index in train_indices])
    validation_batch = collate_sta_net([dataset[index] for index in validation_indices])
    adapter_manifest = dataset.adapter.manifest()
    adapter_manifest.update(
        {
            "observed_train_eeg_shape": list(train_batch["eeg"].shape),
            "observed_train_fnirs_shape": list(train_batch["fnirs"].shape),
            "excluded_label_counts": dataset.excluded_label_counts,
        }
    )
    _write_json(task_dir / "adapter_manifest.json", adapter_manifest)

    resolved_model = STANetConfig(
        task_type=spec.task_type,
        output_dim=spec.output_dim,
        sequence_length=spec.target_length,
        dropout=float(model_config.get("dropout", 0.5)),
        embedding_dim=int(model_config.get("embedding_dim", 256)),
        attention_heads=int(model_config.get("attention_heads", 10)),
        attention_key_dim=int(model_config.get("attention_key_dim", 256)),
        max_lags=spec.fnirs_lag_count,
    )
    model = STANet(resolved_model).to(device)
    criterion = STANetObjective(
        spec.task_type,
        main_weight=float(config.get("loss", {}).get("main_weight", 1.0)),
        eeg_aux_weight=float(config.get("loss", {}).get("eeg_aux_weight", 1.0)),
        alignment_weight=float(config.get("loss", {}).get("alignment_weight", 1.0)),
        regression_loss=str(config.get("loss", {}).get("regression_loss", "smooth_l1")),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("lr", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    train_batch = _move(train_batch, device)
    validation_batch = _move(validation_batch, device)
    amp_enabled = bool(training_config.get("amp", False)) and device.type == "cuda"
    amp_context = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else nullcontext()
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_context():
        train_outputs = model(train_batch["eeg"], train_batch["fnirs"])
        train_losses = criterion(
            train_outputs,
            train_batch["target"],
            train_batch["target_valid_mask"] if spec.task_type == "regression" else None,
        )
    train_losses["total"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config.get("grad_clip_norm", 1.0)))
    optimizer.step()
    train_metrics = _prediction_metrics(
        spec.task_type,
        train_outputs["prediction"],
        train_batch["target"],
        train_batch["target_valid_mask"],
    )
    _append_jsonl(
        task_dir / "metrics" / "train.jsonl",
        {
            "optimizer_step": 1,
            "losses": train_losses,
            "metrics": train_metrics,
            "gradient_norm_before_clip": gradient_norm,
        },
    )

    model.eval()
    with torch.no_grad(), amp_context():
        validation_outputs = model(validation_batch["eeg"], validation_batch["fnirs"])
        validation_losses = criterion(
            validation_outputs,
            validation_batch["target"],
            validation_batch["target_valid_mask"] if spec.task_type == "regression" else None,
        )
    validation_metrics = _prediction_metrics(
        spec.task_type,
        validation_outputs["prediction"],
        validation_batch["target"],
        validation_batch["target_valid_mask"],
    )
    _append_jsonl(
        task_dir / "metrics" / "validation.jsonl",
        {"optimizer_step": 1, "losses": validation_losses, "metrics": validation_metrics},
    )

    np.savez_compressed(
        task_dir / "predictions" / "validation.npz",
        sample_id=np.asarray(validation_batch["sample_id"], dtype=str),
        subject=np.asarray(validation_batch["subject"], dtype=str),
        prediction=validation_outputs["prediction"].detach().float().cpu().numpy(),
        target=validation_batch["target"].detach().cpu().numpy(),
        target_valid_mask=validation_batch["target_valid_mask"].detach().cpu().numpy(),
        lag_attention=validation_outputs["lag_attention"].detach().float().cpu().numpy(),
    )
    checkpoint_path = task_dir / "checkpoints" / "smoke_model.pt"
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "task": asdict(spec),
            "model_config": asdict(resolved_model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": 1,
            "scientific_status": "correctness_only_smoke",
        },
        checkpoint_path,
    )

    finite = all(torch.isfinite(value).all().item() for value in train_losses.values()) and all(
        torch.isfinite(value).all().item() for value in validation_losses.values()
    )
    summary = {
        "schema": RUN_SCHEMA,
        "task_key": task_key,
        "namespace": spec.namespace,
        "task_type": spec.task_type,
        "variant_scope": spec.scientific_scope,
        "status": "smoke_passed" if finite else "smoke_failed_nonfinite",
        "optimizer_steps": 1,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_batch_size": len(train_indices),
        "validation_batch_size": len(validation_indices),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "gradient_norm_before_clip": gradient_norm,
        "elapsed_seconds": time.perf_counter() - task_start,
        "protected_test_opened": False,
        "claim_boundary": "software correctness only; no comparative performance or SOTA claim",
    }
    _write_json(task_dir / "summary.json", summary)
    (task_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# PyTorch STA-Net smoke: {task_key}",
                "",
                f"- Status: `{summary['status']}`",
                f"- Variant: `{spec.scientific_scope}`",
                f"- Task: `{spec.namespace}` (`{spec.task_type}`)",
                f"- Parameters: {summary['parameter_count']:,}",
                f"- Optimizer steps: {summary['optimizer_steps']}",
                f"- Protected test opened: `{summary['protected_test_opened']}`",
                "",
                "This is a correctness-only smoke run. It is not a formal performance result.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def run(args: argparse.Namespace) -> Path:
    started_at = datetime.now(timezone.utc).isoformat()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != RUN_SCHEMA:
        raise ValueError(f"Expected config schema {RUN_SCHEMA}")
    seed = int(config.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(args.device or str(config.get("training", {}).get("device", "auto")))
    task_keys = args.tasks or list(config.get("tasks", []))
    if not task_keys:
        raise ValueError("At least one STA-Net smoke task is required")
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else METHOD_ROOT
        / "runs"
        / "smoke"
        / "sta_net_pytorch_smoke_v1"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_yaml(output_root / "resolved_config.yaml", {**config, "tasks": task_keys, "resolved_device": str(device)})
    method_manifest = {
        "schema": "comparative_method_manifest_v1",
        "method_id": "sta_net_pytorch",
        "paper_doi": PAPER_DOI,
        "upstream_url": UPSTREAM_URL,
        "upstream_revision": UPSTREAM_REVISION,
        "local_upstream_revision": _git("rev-parse", "HEAD", cwd=REPO_ROOT / "comparative_methods" / "STA-Net"),
        "implementation": "independent PyTorch structural reimplementation",
        "tensorflow_used": False,
        "license_status": "upstream repository exposes no license file; redistribution remains blocked",
        "deviations": [
            "configurable binary/multiclass heads",
            "mask-aware continuous sequence regression head",
            "unified-loader geometry and validity adapter",
            "subject-disjoint correctness smoke instead of upstream subject/session performance protocol",
        ],
        "implementation_sha256": {
            "model": _sha256(METHOD_ROOT / "sta_net_pytorch" / "model.py"),
            "adapter": _sha256(METHOD_ROOT / "sta_net_pytorch" / "data.py"),
            "runner": _sha256(Path(__file__).resolve()),
        },
    }
    _write_yaml(output_root / "method_manifest.yaml", method_manifest)
    _write_json(output_root / "environment.json", _environment(device))
    _write_yaml(
        output_root / "decision_protocol.yaml",
        {
            "schema": RUN_SCHEMA,
            "purpose": "classification and regression implementation smoke",
            "selection_metric": None,
            "protected_test_policy": "not opened",
            "formal_performance_claims": "prohibited",
        },
    )
    _write_json(
        output_root / "metric_registry.json",
        {
            "classification_diagnostic": ["loss", "accuracy"],
            "regression_diagnostic": ["masked_smooth_l1", "masked_mae_native_coordinate", "masked_rmse_native_coordinate"],
            "alignment_diagnostic": ["fgsa1", "fgsa2", "egta"],
            "primary_scientific_endpoint": None,
        },
    )
    _write_json(
        output_root / "evidence_calibration.json",
        {
            "evidence_level": "software_correctness_smoke",
            "supports": ["finite forward", "finite backward", "optimizer step", "real unified-loader input"],
            "does_not_support": ["comparative superiority", "source numerical reproduction", "generalization", "SOTA"],
            "protected_test_opened": False,
        },
    )

    start = time.perf_counter()
    summaries = [
        _run_task(task_key=task_key, config=config, output_root=output_root, device=device)
        for task_key in task_keys
    ]
    status = "smoke_passed" if all(summary["status"] == "smoke_passed" for summary in summaries) else "smoke_failed"
    manifest = {
        "schema": RUN_SCHEMA,
        "status": status,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "command": [sys.executable, *sys.argv],
        "repository_commit": _git("rev-parse", "HEAD"),
        "repository_dirty": bool(_git("status", "--porcelain")),
        "task_contract_sha256": {key: task_contract_sha256(get_sta_net_task_spec(key)) for key in task_keys},
        "tasks": summaries,
        "artifacts": {},
        "protected_test_opened": False,
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps({"status": status, "output_dir": str(output_root), "tasks": task_keys}, indent=2))
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+", choices=("motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual", "refed_regression"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
