"""A reserved paired integer atom for train-defined quiet Hb observations.

The source tokenizer handles every ordinary integer sequence. A complete
reserved pair selects fixed zero-sum Hb levels across all fifteen patches.
The observation-dependent decision is represented inside those integers.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch


class SilentObservationTokenizer:
    def __init__(self, source, metadata, level):
        self.source = source
        self.metadata = metadata
        self.codebook_size = int(metadata['codebook_size'])
        self.base_codebook_size = int(metadata['base_codebook_size'])
        self.silent_id = self.base_codebook_size
        self.base_sizes = metadata.get('base_codebook_sizes',dict(S=self.base_codebook_size,O=self.base_codebook_size))
        self.branch_sizes = metadata.get('branch_codebook_sizes',dict(S=self.codebook_size,O=self.codebook_size))
        self.silent_ids = {branch:int(self.base_sizes[branch]) for branch in ('S','O')}
        self.level = np.asarray(level, dtype=np.dtype(metadata['output_dtype']))
        if (metadata['modality'] != 'hb' or self.codebook_size != self.base_codebook_size + 1
                or self.level.shape != (2,) or not np.isfinite(self.level).all()
                or not 0 <= float(metadata['quiet_threshold_input_sd'])
                or any(int(self.branch_sizes[b])!=int(self.base_sizes[b])+1 for b in ('S','O'))):
            raise ValueError('finite joint-Hb atom and one additional integer required')

    def select_ids(self, values, ordinary_ids, ordinary_prediction=None):
        values = np.asarray(values)
        if values.ndim != 3 or values.shape[1:] != (15,40) or not np.isfinite(values).all():
            raise ValueError('complete finite observed Hb X [batch,15,40] required')
        if set(ordinary_ids) != {'S','O'}:
            raise ValueError('source S/O integers required')
        ids = {branch: np.asarray(ordinary_ids[branch]).copy() for branch in ('S','O')}
        if any(value.shape != values.shape[:2] or value.dtype.kind not in 'iu'
               or (value < 0).any() or (value >= self.base_sizes[branch]).any() for branch,value in ids.items()):
            raise ValueError('in-range ordinary source IDs required')
        energy = np.square(values.astype(np.float64)).mean(axis=(1,2))
        low = np.sqrt(energy) <= float(self.metadata['quiet_threshold_input_sd'])
        if low.any():
            prediction = self.source.decode_ids(ids) if ordinary_prediction is None else ordinary_prediction
            error = np.square(np.asarray(prediction['sum'],float) - values).mean(axis=(1,2))
            quiet = low & (energy < error)
            for branch in ('S','O'):
                ids[branch][quiet] = self.silent_ids[branch]
        return ids

    def encode(self, values):
        return self.select_ids(values,self.source.encode(values))

    def decode_ids(self, ids):
        if set(ids) != {'S','O'}:
            raise ValueError('both branch integer sequences required')
        arrays = {branch: np.asarray(ids[branch]) for branch in ('S','O')}
        shape = arrays['S'].shape
        if (len(shape) != 2 or shape[1] != 15 or any(value.shape != shape or value.dtype.kind not in 'iu'
                or (value < 0).any() or (value >= self.branch_sizes[branch]).any() for branch,value in arrays.items())):
            raise ValueError('matched in-range [batch,15] integer arrays required')
        quiet = (arrays['S'] == self.silent_ids['S']).all(axis=1) & (arrays['O'] == self.silent_ids['O']).all(axis=1)
        any_reserved = (arrays['S'] == self.silent_ids['S']).any(axis=1) | (arrays['O'] == self.silent_ids['O']).any(axis=1)
        if not np.array_equal(quiet,any_reserved):
            raise ValueError('reserved observation atom requires a complete S/O pair for the whole window')
        safe = {branch: np.where(quiet[:,None],0,value) for branch,value in arrays.items()}
        result = self.source.decode_ids(safe)
        if quiet.any():
            result = {key: value.copy() for key,value in result.items() if key in ('S','O','sum')}
            result['S'][quiet] = np.tile(self.level,20)[None,None,:]
            result['O'][quiet] = -np.tile(self.level,20)[None,None,:]
            result['sum'][quiet] = 0
        return result

    def __call__(self,values):
        ids = self.encode(values)
        return dict(self.decode_ids(ids),ids=ids)


def load_model(path):
    from .contextual_latent_dictionary import load_model as load_source
    path = Path(path)
    saved = torch.load(path,map_location='cpu',weights_only=False)
    metadata = saved['metadata']
    if metadata['schema'] != 'silent_observation_tokenizer_v1':
        raise ValueError('reserved-observation checkpoint schema required')
    source_path = (path.parent / metadata['source_file']).resolve()
    if source_path.parent != path.parent.resolve():
        raise ValueError('frozen source must accompany the atom checkpoint')
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != metadata['source_sha256']:
        raise ValueError('frozen source checkpoint changed')
    source = load_source(source_path)
    source_sizes = source.metadata.get('branch_codebook_sizes',dict(S=source.metadata['codebook_size'],O=source.metadata['codebook_size']))
    expected_sizes = metadata.get('base_codebook_sizes',dict(S=metadata['base_codebook_size'],O=metadata['base_codebook_size']))
    if source.metadata['modality'] != 'hb' or source_sizes != expected_sizes:
        raise ValueError('matching frozen Hb vocabulary required')
    return SilentObservationTokenizer(source,metadata,saved['level'])
