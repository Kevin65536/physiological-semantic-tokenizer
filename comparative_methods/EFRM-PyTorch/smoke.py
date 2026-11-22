#!/usr/bin/env python3
"""Real-data EFRM adapter plus tiny-model correctness smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

METHOD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = METHOD_ROOT.parents[1]
for path in (REPO_ROOT, METHOD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from efrm_pytorch.data import EFRMSyncPretrainDataset, collate_efrm_pairs
from efrm_pytorch.model import EFRMSyncModel
from efrm_pytorch.visualization import export_alignment_evidence, render_alignment_report


def _find_same_record_pair(dataset: EFRMSyncPretrainDataset) -> list[dict]:
    first_by_record: dict[str, dict] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        if not sample["admitted"]:
            continue
        previous = first_by_record.get(sample["join_key"])
        if previous is not None and previous["sample_id"] != sample["sample_id"]:
            return [previous, sample]
        first_by_record[sample["join_key"]] = sample
    raise RuntimeError("no record contains two fully admitted synchronized samples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=DEFAULT_CLEAN_CACHE_ROOT)
    parser.add_argument("--dataset", default="eeg_fnirs_single_trial")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = EFRMSyncPretrainDataset(cache_root=args.cache_root, dataset_ids=(args.dataset,))
    batch = collate_efrm_pairs(_find_same_record_pair(dataset))
    device = torch.device(args.device)
    model = EFRMSyncModel(
        embed_dim=32, depth=1, num_heads=4,
        decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=2.0,
    ).to(device)
    tensors = {
        key: batch[key].to(device)
        for key in ("eeg", "fnirs", "eeg_patch_valid", "fnirs_patch_valid")
    }
    result = model(**tensors)
    result["loss"].backward()
    metadata = [
        {key: batch[key][index] for key in (
            "sample_id", "dataset_id", "subject", "record_id", "join_key",
            "task_namespace", "condition", "crop_start_s", "duration_s",
        )}
        for index in range(len(batch["sample_id"]))
    ]
    evidence = export_alignment_evidence(
        output / "figure_data",
        eeg_embeddings=result["eeg_embedding"],
        fnirs_embeddings=result["fnirs_embedding"],
        metadata=metadata,
    )
    metrics = render_alignment_report(evidence, output)
    status = {
        "schema": "efrm_sync_smoke_v1",
        "status": "smoke_passed",
        "dataset": args.dataset,
        "device": str(device),
        "eeg_shape": list(batch["eeg"].shape),
        "fnirs_shape": list(batch["fnirs"].shape),
        "total_loss": float(result["loss"].detach().cpu()),
        "retrieval": metrics,
        "protected_test_opened": False,
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
