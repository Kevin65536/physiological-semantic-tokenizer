"""Tiny inference interface for the exported, trusted local observation baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


class ObservationBaseline:
    """No fitting. Each item is a complete 2 s site, not a 30 s context or a source label.

    `x` must already be in the named G0 preprocessed, training-position-scaled
    coordinate. Never feed raw intensity or silently scale/center on a test item.
    The caller keeps subject/record/site/time/validity metadata and skips missing
    items, rather than inventing tokens for a missing channel or scoring edge.
    """
    def __init__(self, directory):
        root=Path(directory).resolve(strict=True)
        self.manifest=json.loads((root/'manifest.json').read_text())
        if self.manifest.get('schema')!='observation_baseline_bundle_v1' or set(self.manifest['modalities'])!={'eeg','hb'}:
            raise ValueError('invalid observation baseline bundle')
        def checked(rel):
            path=(root/rel).resolve(strict=True)
            if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=self.manifest['sha256'][rel]:
                raise ValueError('bundle integrity check failed')
            return path
        # joblib is pickle-based: hashes verify internal consistency, NOT an untrusted publisher.
        self._routers={}; self._models={}
        for modality, m in self.manifest['modalities'].items():
            self._routers[modality]=joblib.load(checked(m['router'])); self._models[modality]=joblib.load(checked(m['model']))
            if (self._routers[modality].modality!=modality or self._routers[modality].n_patterns!=16 or
                    self._models[modality].rate('conditional')['bits_per_item']!=m['total_bits_per_patch']):
                raise ValueError('manifest and frozen codec disagree')
        self._scales=json.loads(checked('training_x_scales.json').read_text())

    def scale_g0(self, x, modality, dataset, position):
        """Unscaled *G0 X*, not raw data. No centering; Hb uses one training SD for both colors."""
        if modality not in self._models: raise ValueError('unknown modality')
        value=self._scales[modality][dataset][position]['scale']
        return np.asarray(x,dtype=np.float32)/np.float32(value)

    def encode(self, x, modality, *, detail=False):
        if modality not in self._models: raise ValueError('unknown modality')
        return (self._models[modality].encode(x,mode='conditional') if detail else
                self._routers[modality].encode(x,mode='prototype'))

    def decode(self, codes, modality, *, detail=False):
        """For detail=False this is a training prototype, NOT a phase-accurate waveform."""
        if modality not in self._models: raise ValueError('unknown modality')
        return (self._models[modality].decode(codes,mode='conditional') if detail else
                self._routers[modality].decode(codes,mode='prototype'))

    def pack(self, codes, modality, *, detail=False):
        if modality not in self._models: raise ValueError('unknown modality')
        return (self._models[modality].pack(codes,mode='conditional') if detail else
                self._routers[modality].pack(codes,mode='prototype'))

    def unpack(self, payload, n_items, modality, *, detail=False):
        if modality not in self._models: raise ValueError('unknown modality')
        return (self._models[modality].unpack(payload,n_items,mode='conditional') if detail else
                self._routers[modality].unpack(payload,n_items,mode='prototype'))
