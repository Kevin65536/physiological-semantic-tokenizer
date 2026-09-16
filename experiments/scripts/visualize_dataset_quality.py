#!/usr/bin/env python
"""Unified dataset quality visualisation tool.

Produces a self-contained HTML report with waveform previews, post-unification
amplitude distributions, channel geometry, unit/provenance metadata, label and
timing checks for the four original datasets.  Croce caches are derived
source/observation supervision targets and are intentionally excluded.

Typical usage::

    # Audit all registered datasets
    python experiments/scripts/visualize_dataset_quality.py --all

    # Audit a single dataset
    python experiments/scripts/visualize_dataset_quality.py --dataset eeg_fnirs_single_trial

    # Use a specific subject for signal extraction
    python experiments/scripts/visualize_dataset_quality.py --all --subject-id 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_quality_report import DatasetQualityReporter
from src.data.registry import (
    get_dataset_registration,
)
from src.data.unified_physiology import DEFAULT_UNIFIED_WINDOW_DURATION_S, RAW_DATASET_IDS

DEFAULT_OUTPUT_BASE = (
    PROJECT_ROOT / 'experiments' / 'runs' / 'physiology_semantic_tokenizer' / 'data_quality_audit'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a unified dataset quality visualisation report.',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all',
        action='store_true',
        default=True,
        help='Audit all registered datasets (default).',
    )
    group.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Audit a single dataset by id or alias.',
    )
    parser.add_argument(
        '--subject-id',
        type=int,
        default=1,
        help='Subject to use for signal extraction (default: 1; ignored for planned loaders).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='Override output directory (default: timestamped dir under data_quality_audit/).',
    )
    parser.add_argument(
        '--embed-images',
        action='store_true',
        default=True,
        help='Embed PNGs as base64 in HTML (default).',
    )
    parser.add_argument(
        '--no-embed-images',
        action='store_false',
        dest='embed_images',
        help='Reference PNGs as external files instead of embedding.',
    )
    parser.add_argument(
        '--max-channels',
        type=int,
        default=8,
        help='Maximum channels per modality in waveform plots (default: 8).',
    )
    parser.add_argument(
        '--cache-root',
        default=str(PROJECT_ROOT / 'data/cache/physiology_semantic_clean_v4'),
        help='Canonical clean-cache root containing signal, event, and geometry sidecars.',
    )
    parser.add_argument(
        '--samples-per-dataset',
        type=int,
        default=4,
        help='Number of aligned windows used for each amplitude audit.',
    )
    parser.add_argument(
        '--window-duration-s',
        type=float,
        default=DEFAULT_UNIFIED_WINDOW_DURATION_S,
        help=(
            'Aligned observation-context duration in seconds '
            f'(default: {DEFAULT_UNIFIED_WINDOW_DURATION_S:g}).'
        ),
    )
    parser.add_argument(
        '--eeg-signal-branch',
        choices=(
            'raw_with_ocular_artifact',
            'single_trial_eeg_artifact_clean_v2',
            'single_trial_eeg_artifact_clean_v4',
        ),
        default='single_trial_eeg_artifact_clean_v4',
        help='Single-Trial EEG branch to audit; v3 is the admitted registry default.',
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all registered datasets and exit.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --list-datasets
    if args.list_datasets:
        print('[DatasetQuality] Four original datasets:')
        for dataset_id in RAW_DATASET_IDS:
            reg = get_dataset_registration(dataset_id)
            print(f'  {reg.dataset_id:40s}  {reg.display_name:45s}  [{reg.loader_status}]')
        print('  croce_local_cache: derived supervision target (excluded)')
        return

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = DEFAULT_OUTPUT_BASE / timestamp

    print(f'[DatasetQuality] Output directory: {output_dir}')

    # Determine dataset list
    if args.dataset:
        resolved = get_dataset_registration(args.dataset).dataset_id
        if resolved not in RAW_DATASET_IDS:
            raise SystemExit(
                f'{resolved!r} is not one of the four original datasets; Croce caches are derived targets.'
            )
        dataset_ids: List[str] = [resolved]
        print(f'[DatasetQuality] Auditing single dataset: {dataset_ids[0]}')
    else:
        dataset_ids = list(RAW_DATASET_IDS)
        print(f'[DatasetQuality] Auditing all {len(dataset_ids)} original datasets')

    # Run
    reporter = DatasetQualityReporter(
        output_dir=output_dir,
        cache_root=Path(args.cache_root),
        embed_images=args.embed_images,
        max_channels=args.max_channels,
        samples_per_dataset=args.samples_per_dataset,
        window_duration_s=args.window_duration_s,
        eeg_signal_branch=args.eeg_signal_branch,
    )

    html_path, md_path = reporter.build_full_report(
        dataset_ids,
        subject_id=args.subject_id,
    )

    print(f'[DatasetQuality] Done.')
    print(f'[DatasetQuality]   HTML: {html_path}')
    print(f'[DatasetQuality]   MD:   {md_path}')
    print(f'[DatasetQuality]   Figs: {output_dir / "figures"}')


if __name__ == '__main__':
    main()
