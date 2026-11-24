"""Apply the preregistered componentwise reconstruction tests on identical G0 examples."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml


def score(candidate, baseline, continuous, contract):
    spec = contract['reconstruction']
    modality = candidate['modality']
    if modality not in ('eeg', 'hb'):
        raise ValueError('unknown modality')
    for ref in (baseline, continuous):
        if (candidate['role'], modality) != (ref['role'], ref['modality']):
            raise ValueError('different panel roles or modalities')
    metrics = candidate['metrics']
    shape = (3, 1 if modality == 'eeg' else 2)
    checks = {}

    def macro(source, key):
        value = np.asarray(source['metrics'][key]['macro'], dtype=float)
        if value.shape != shape:
            raise ValueError(f'{key} is outside the X/S/O × color contract')
        return value

    def compare(key, x, bound, comparator):
        bound = np.broadcast_to(bound, x.shape)
        result = np.isfinite(x) & np.isfinite(bound) & comparator(x, bound)
        checks[key] = dict(passed=bool(result.all()), values=np.where(np.isfinite(x), x, None).tolist(),
                           limits=np.where(np.isfinite(bound), bound, None).tolist(),
                           component_color_pass=result.tolist())

    old = macro(baseline, 'nmse')
    upper = macro(continuous, 'nmse')
    limit = old - .5 * np.maximum(0, old - upper) if modality == 'eeg' else 1.05 * old
    compare('nmse', macro(candidate, 'nmse'), limit, np.less_equal)
    dataset_values = metrics['nmse']['by_dataset']
    if len(dataset_values) != 4 or dataset_values.keys() != baseline['metrics']['nmse']['by_dataset'].keys():
        raise ValueError('four matched datasets required')
    for dataset, values in dataset_values.items():
        limit = spec['dataset_component_max_vs_baseline_ratio'] * np.asarray(baseline['metrics']['nmse']['by_dataset'][dataset])
        compare('nmse/' + dataset, np.asarray(values), limit, np.less_equal)
    for name, bound, op in [('gain_min', spec['rms_gain_interval'][0], np.greater_equal),
                            ('gain_max', spec['rms_gain_interval'][1], np.less_equal)]:
        compare(name, macro(candidate, 'rms_gain'), bound, op)
    compare('positive_correlation', macro(candidate, 'correlation'), 0., np.greater)
    if modality == 'eeg':
        band_limit = spec['eeg']['component_band_abs_db_max']
        for lo, hi in spec['eeg']['bands_for_gate_hz']:
            for suffix in ('db', 'abs_db'):
                key = f'band_{lo}_{hi}_{suffix}'
                compare(key, abs(macro(candidate, key)), band_limit, np.less_equal)
        compare('local_rms_correlation', macro(candidate, 'local_rms_correlation'),
                spec['eeg']['local_rms_correlation_min'], np.greater_equal)
    else:
        for name, threshold, op in [('slope_correlation', 'slope_correlation_min', np.greater_equal),
                                    ('curvature_correlation', 'curvature_correlation_min_exclusive', np.greater),
                                    ('boundary_jump_difference_nmse', 'boundary_jump_difference_nmse_max', np.less_equal)]:
            compare(name, macro(candidate, name), spec['hb'][threshold], op)
    hard_replay = candidate['integer_replay']['status'] == 'passed'
    complete = candidate['support']['nonfinite_prediction_item_color_cells'] == 0
    return dict(modality=modality, role=candidate['role'], checkpoint=candidate['checkpoint'],
                replay_passed=hard_replay, finite_predictions=complete, checks=checks,
                reconstruction_passed=bool(hard_replay and complete and all(v['passed'] for v in checks.values())),
                failed=[k for k, v in checks.items() if not v['passed']])


def paired_subject_improvement(candidate, reference):
    """Estimate uncertainty across subjects conditional on a fixed development panel."""
    def index(summary):
        return {(item['dataset'], item['subject']): np.mean(item['value'])
                for item in summary['metrics']['nmse']['by_subject']}
    current, baseline = index(candidate), index(reference)
    if current.keys() != baseline.keys():
        raise ValueError('paired subject populations differ')
    deltas = {k: float(baseline[k] - current[k]) for k in sorted(current)}
    groups = {dataset: np.array([v for k, v in deltas.items() if k[0] == dataset])
              for dataset in sorted({k[0] for k in deltas})}
    if not all(np.isfinite(v).all() for v in groups.values()):
        return dict(status='nonfinite_subjects', planned=len(deltas))
    rng = np.random.default_rng(20261123)
    estimates = np.stack([rng.choice(v, size=(5000, len(v)), replace=True).mean(1)
                          for v in groups.values()]).mean(0)
    return dict(contrast='historical K128 minus candidate NMSE; positive favors candidate',
                unit='subject; stratified within dataset; each dataset equal',
                conditions='checkpoint and reused panel fixed; exploratory intervals, no multiplicity correction',
                dataset_mean={k: float(v.mean()) for k, v in groups.items()},
                improved_subjects={k: int((v > 0).sum()) for k, v in groups.items()},
                independent_subjects={k: len(v) for k, v in groups.items()},
                paired_differences=[dict(dataset=d, subject=t, difference=v) for (d,t),v in deltas.items()],
                macro_difference=float(np.mean([v.mean() for v in groups.values()])),
                percentile_95_interval=np.quantile(estimates, [.025,.975]).tolist(), draws=5000)


def integer_rate(checkpoint, ids_path):
    """Fixed-width payload, empirical marginal entropy, and actual exported bytes."""
    if checkpoint.suffix == '.npz':
        with np.load(checkpoint, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays['metadata']))
            size = int(metadata['codebook_size'])
            parameters = sum(arrays[k].size * (2 if arrays[k].dtype.kind == 'c' else 1)
                             for k in arrays if arrays[k].dtype.kind in 'fc')
        model_size = dict(fitted_real_scalar_count=int(parameters))
    else:
        import torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'model_config' in saved:
            size = int(saved['model_config']['codebook_size'])
        elif 'codebook_size' in saved.get('metadata', {}):
            size = int(saved['metadata']['codebook_size'])
        else:
            size = int(saved['signature']['task']['candidate'].removeprefix('vq'))
        def scalar_count(value):
            if isinstance(value, dict):
                return sum(scalar_count(child) for child in value.values())
            if isinstance(value, torch.Tensor):
                return value.numel()
            if isinstance(value, np.ndarray):
                return value.size
            return 0
        model_size = dict(state_scalars_including_buffers=sum(scalar_count(saved.get(k, {}))
                          for k in ('model_state', 'reference_state', 'assignment')))
        source_file = saved.get('metadata', {}).get('source_file')
        package_bytes = checkpoint.stat().st_size
        if source_file is not None:
            import hashlib
            source_path = (checkpoint.parent / source_file).resolve()
            if (source_path.parent != checkpoint.parent.resolve() or
                    hashlib.sha256(source_path.read_bytes()).hexdigest() != saved['metadata']['source_sha256']):
                raise ValueError('frozen inference source identity mismatch')
            source_state = torch.load(source_path, map_location='cpu', weights_only=False)
            model_size['state_scalars_including_buffers'] += sum(scalar_count(source_state.get(k, {}))
                for k in ('model_state', 'reference_state', 'assignment'))
            model_size['fixed_search_table_scalars'] = sum(scalar_count(saved.get(k, {}))
                for k in ('neighbors', 'x_variance', 'shape_variance', 'slope_variance'))
            package_bytes += source_path.stat().st_size
        summary_path = checkpoint.parent / 'summary.json'
        if summary_path.is_file():
            training = json.loads(summary_path.read_text())
            if 'parameters' in training:
                model_size['network_parameters'] = training['parameters']
        if 'metadata' in saved:
            model_size.update({key: value for key, value in saved['metadata'].items()
                               if 'parameter' in key})
    branch_sizes = saved.get('metadata',{}).get('branch_codebook_sizes',{}) if checkpoint.suffix != '.npz' else {}
    with np.load(ids_path, allow_pickle=False) as stored, ZipFile(ids_path) as archive:
        branches = {}
        for key in ('S', 'O'):
            branch_size = int(branch_sizes.get(key,size))
            ids = stored[key]
            if ids.ndim != 2 or ids.shape[1] != 15 or ids.dtype.kind not in 'iu':
                raise ValueError('expected fifteen integer IDs per position window')
            if ids.size == 0 or ids.min() < 0 or ids.max() >= branch_size:
                raise ValueError('integer outside checkpoint vocabulary')
            counts = np.bincount(ids.reshape(-1).astype(np.int64), minlength=branch_size)
            p = counts[counts > 0] / ids.size
            seconds = len(ids) * 30
            branches[key] = dict(integer_count=ids.size, active_codes=len(p), vocabulary=branch_size,
                                 marginal_entropy_bits_per_second=float(-np.sum(p*np.log2(p)) / 2),
                                 fixed_width_bits_per_second=int(np.ceil(np.log2(branch_size))) / 2,
                                 exported_dtype=ids.dtype.name, raw_array_bits_per_second=ids.itemsize * 4,
                                 zip_member_payload_bits_per_second=8*archive.getinfo(key+'.npy').compress_size/seconds)
    return dict(branches=branches, checkpoint_bytes=checkpoint.stat().st_size,
                inference_package_bytes=checkpoint.stat().st_size if checkpoint.suffix == '.npz' else package_bytes, model_size=model_size,
                archive_bytes_including_example_indices=ids_path.stat().st_size,
                scope='per branch and native position; two branch payloads per modality; entropy is a marginal estimate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'baseline', 'continuous', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--spatial-arm', choices=['spatial', 'independent'],
                        help='select one arm of a multisite screen audit')
    args = parser.parse_args()
    contract_path = Path(__file__).resolve().parents[1] / 'configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml'
    contract = yaml.safe_load(contract_path.read_text())
    paths = [args.candidate, args.baseline, args.continuous]
    summary_files = [p/('summary_v2.json' if (p/'summary_v2.json').exists() else 'summary.json') for p in paths]
    if args.spatial_arm:
        summary_files[0] = args.candidate / 'screen_summary.json'
    summaries = [json.loads(p.read_text()) for p in summary_files]
    metric_file = args.candidate / 'metrics.npz'
    if args.spatial_arm:
        record = summaries[0]
        replay_pass = (record['same_integer_sequences'] and record['integer_replay_max_abs'] == 0
                       and record['standalone_metric_replay'] == 'exact_from_saved_ID_file_after_model_reload')
        summaries[0] = dict(record['summaries'][args.spatial_arm], role=record['role'], modality='eeg',
                            checkpoint=record['checkpoint'], integer_replay={'status': 'passed' if replay_pass else 'unverified'})
        metric_file = args.candidate / (args.spatial_arm + '_metrics.npz')
    with np.load(metric_file) as items:
        examples = items['examples']
    for path in paths[1:]:
        with np.load(path/'metrics.npz') as items:
            np.testing.assert_array_equal(examples, items['examples'])
    result = score(*summaries, contract)
    result['paired_subject_improvement'] = paired_subject_improvement(*summaries[:2])
    if not args.spatial_arm and (args.candidate/'integer_ids.npz').exists():
        result['rate_and_size'] = integer_rate(Path(result['checkpoint']), args.candidate/'integer_ids.npz')
    result.update(contract=str(contract_path), summaries=list(map(str, summary_files)),
                  count=len(examples), component_order=['X', 'S', 'O'])
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
