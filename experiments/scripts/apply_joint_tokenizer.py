"""Encode 30-second observed X or decode its four integer sequences in G0 signal units."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def checked_path(bundle, spec):
    path = (bundle.parent / spec['path']).resolve()
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
            raise ValueError(f'frozen file changed: {path}')
    return path


def load_branch_pair(kind, path, variant=None):
    if kind == 'montage_word_search_eeg':
        from src.tokenizers.montage_word_search_inference import load_model as load_search
        return load_search(path, variant=variant)
    if kind == 'dictionary':
        from src.tokenizers.supervised_waveform_dictionary import load_model
    elif kind == 'spectral':
        from src.tokenizers.spectral_waveform_dictionary import load_model
    elif kind == 'spatial':
        from src.tokenizers.spatial_waveform_dictionary import load_model
    elif kind == 'fullband':
        from src.tokenizers.fullband_eeg_tokenizer import load_model
    elif kind == 'contextual':
        from src.tokenizers.contextual_latent_dictionary import load_model
    elif kind == 'hb_native_id_readout':
        from src.tokenizers.hb_native_id_readout import load_model
    elif kind == 'cross_branch_innovation_eeg':
        from src.tokenizers.cross_branch_innovations import load_model
    elif kind == 'band_exchange_eeg':
        from src.tokenizers.band_exchange_words import load_model
    else:
        raise ValueError(f'unsupported frozen model type: {kind}')
    return load_model(path)


def normalize_observations(values, scales):
    """Match G0's per-position division in the observation's numeric precision."""
    precision = np.result_type(values.dtype, np.float32)
    factors = np.asarray(scales, dtype=precision)
    column_shape = (len(factors),) + (1,) * (values.ndim - 1)
    return (values / factors.reshape(column_shape)).astype(np.float32)


def spatial_eeg(model, source, mode, scales):
    """One fixed montage per crop, with only identifiers forwarded to the decoder."""
    ds, pos, crop = (source['eeg_' + k] for k in ('dataset', 'position', 'crop'))
    if ds.ndim != 1 or ds.shape != pos.shape or ds.shape != crop.shape or not len(ds):
        raise ValueError('expected dataset, native position and crop identity for every row')
    result = {'eeg_dataset': ds, 'eeg_position': pos, 'eeg_crop': crop}
    if mode == 'encode':
        x = source['eeg_x']
        observed = source['eeg_observed']
        if x.shape != (len(ds), 6000, 1) or observed.shape != ds.shape or observed.dtype != bool:
            raise ValueError('expected a complete montage and its observation flags')
        result.update({f'eeg_{b}': np.empty((len(ds), 15), np.int32) for b in ('S', 'O')})
    else:
        ids = {b: source['eeg_' + b] for b in ('S', 'O')}
        if any(v.shape != (len(ds), 15) or v.dtype.kind not in 'iu' for v in ids.values()):
            raise ValueError('expected both complete integer arrays')
        result.update({f'eeg_{b}': np.empty((len(ds), 6000, 1)) for b in ('S', 'O', 'sum')})
    for dataset in np.unique(ds):
        layout = model.metadata['positions'][str(dataset)]
        dataset_scales = {k.upper(): v['scale'] for k, v in scales[str(dataset)].items()}
        for group in np.unique(crop[ds == dataset]):
            rows = np.flatnonzero((ds == dataset) & (crop == group))
            table = {str(pos[i]).upper(): i for i in rows}
            if len(rows) != len(layout) or set(table) != {name.upper() for name in layout}:
                raise ValueError('every fixed montage site must occur exactly once per crop')
            indices = np.array([table[name.upper()] for name in layout])
            factor = np.array([dataset_scales[name.upper()] for name in layout])
            if not np.all(np.isfinite(factor) & (factor > 0)):
                raise ValueError('invalid frozen montage scale')
            if mode == 'encode':
                available = observed[indices]
                waves = np.where(available[:, None], x[indices, :, 0], 0.)
                if not np.isfinite(waves).all():
                    raise ValueError('observed voltage must be finite')
                waves = normalize_observations(waves, factor).reshape(1, len(layout), 15, 400)
                encoded = model.encode(waves, str(dataset), available[None])
                for b in ('S', 'O'):
                    result['eeg_' + b][indices] = encoded[b][0]
            else:
                decoded = model.decode_ids({b: v[indices][None] for b, v in ids.items()}, str(dataset))
                for b in ('S', 'O', 'sum'):
                    result['eeg_' + b][indices, :, 0] = decoded[b].reshape(len(layout), 6000) * factor[:, None]
    return result


def apply(bundle, mode, source):
    spec = json.loads(bundle.read_text())
    if spec['schema'] != 'joint_tokenizer_inference_v1':
        raise ValueError('unsupported inference bundle')
    scales = json.loads(checked_path(bundle, spec['scales']).read_text())['positions']
    output = {}
    required = {m + '_' + k for m in ('eeg', 'hb')
                for k in (('dataset', 'position', 'x') if mode == 'encode' else ('dataset', 'position', 'S', 'O'))}
    if spec['eeg']['kind'] in ('spatial', 'band_exchange_eeg', 'cross_branch_innovation_eeg', 'montage_word_search_eeg'):
        required.add('eeg_crop')
        if mode == 'encode':
            required.add('eeg_observed')
    if set(source) != required:
        raise ValueError(f'expected exactly these arrays: {sorted(required)}')
    for modality, samples, colors in [('eeg', 6000, 1), ('hb', 300, 2)]:
        item = spec[modality]
        checkpoint = checked_path(bundle, item)
        if mode == 'encode' and item['kind'] == 'spatial' and 'encoder' in item:
            from src.tokenizers.spatial_analysis_synthesis import load_encoder
            model = load_encoder(checked_path(bundle, item['encoder']))
        else:
            model = load_branch_pair(item['kind'], checkpoint, item.get('encoder_variant'))
        if modality == 'eeg' and item['kind'] in ('spatial', 'band_exchange_eeg', 'cross_branch_innovation_eeg', 'montage_word_search_eeg'):
            output.update(spatial_eeg(model, source, mode, scales['eeg']))
            continue
        if model.metadata['modality'] != modality:
            raise ValueError('modality and checkpoint differ')
        dataset, position = (source[modality + '_' + k] for k in ('dataset', 'position'))
        if dataset.ndim != 1 or dataset.shape != position.shape or not len(dataset):
            raise ValueError('supply one native dataset and position per 30-second observation')
        factor = np.array([scales[modality][str(d)][str(p)]['scale'] for d, p in zip(dataset, position)])
        if not np.all(np.isfinite(factor) & (factor > 0)):
            raise ValueError('unknown or invalid frozen G0 scale')
        output[modality + '_dataset'] = dataset
        output[modality + '_position'] = position
        if mode == 'encode':
            x = source[modality + '_x']
            if x.shape != (len(factor), samples, colors) or not np.isfinite(x).all():
                raise ValueError('expected complete finite [positions, time, colors] observations')
            x = normalize_observations(x, factor).reshape(len(x), 15, -1)
            encoded = model.encode(x)
            for branch in ('S', 'O'):
                ids = np.asarray(encoded[branch])
                if ids.shape != (len(x), 15) or ids.dtype.kind not in 'iu':
                    raise ValueError('candidate did not produce 15 atomic integers per branch')
                output[modality + '_' + branch] = ids
        else:
            ids = {b: source[modality + '_' + b] for b in ('S', 'O')}
            sizes = model.metadata.get('branch_codebook_sizes', {})
            for branch, values in ids.items():
                size = int(sizes.get(branch, model.metadata['codebook_size']))
                if (values.shape != (len(factor), 15) or values.dtype.kind not in 'iu'
                        or np.any(values < 0) or np.any(values >= size)):
                    raise ValueError('expected valid frozen code indices [positions, 15]')
            decoded = model.decode_ids(ids)
            for key in ('S', 'O', 'sum'):
                values = np.asarray(decoded[key]).reshape(len(factor), samples, colors)
                output[modality + '_' + key] = values * factor[:, None, None]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Input and output are NPZ archives. Each modality can have a different number of rows.
Encode input:
  eeg_x [n,6000,1] and hb_x [m,300,2] contain observations in the retained G0
  signal units (HbO, HbR in that order). Provide {eeg,hb}_dataset and
  {eeg,hb}_position string vectors, one entry per row. A spatial EEG bundle
  also requires eeg_crop (group identity) and eeg_observed (boolean vector).
  Include the entire frozen electrode layout for each spatial crop.
Encode output / decode input:
  eeg_S, eeg_O, hb_S, hb_O are integer arrays with 15 columns per position.
  Dataset, position and spatial crop identities accompany those arrays.
Decode output:
  {eeg,hb}_{S,O,sum} have the same shape and units as their observed inputs.
A bundle resolves its pinned checkpoint and training-scale paths relative to
the bundle file. The output path must be new.""")
    parser.add_argument('mode', choices=['encode', 'decode'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        output = apply(args.bundle, args.mode, archive)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, **output)


if __name__ == '__main__':
    main()
