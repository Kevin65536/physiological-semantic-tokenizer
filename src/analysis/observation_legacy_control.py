"""Frozen, observation-only controls for the matched G0 development panel.

The caller owns input identity, masks and training-scale normalization. This
adapter opens no signal/teacher arrays, does not fit a scale, and cannot accept
S/O targets, site metadata, or a caller-selected checkpoint. Its hard decoder
consumes two sequences of 15 integers: 14 nominal bits per two-second patch,
per location. The continuous reference is not a bitrate-matched codec.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock

import numpy as np
import torch
import yaml

from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer


_CONFIGS = Path('experiments/configs/physiology_semantic_tokenizer')
_RUNS = Path('experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer')
_G0 = str(_RUNS / '20261117_base_g0_v2')
# Existing G0 manifest/scale identities, as recorded by all three source plans.
_G0_INPUT = '37319917430f3aa11e3eaf39'
_G0_SCALES = 'd8b6634e55fa73bbc1243650'
# These are the owning runner's config identities, not checkpoint filenames supplied by users.
_G1 = ('20261117_base_g1_v2', 'dual_component_base_g1_v1', 'ab4ef6e8fc1367876a7272ad')
_SOURCES = {
    ('eeg', False): ('20261120_input_quantization_b_v1', 'dual_component_base_matrix_v1', 'a7c8203824a2a8391f586127'),
    ('eeg', True): ('20261119_e1_waveform8_v1', 'dual_component_base_e1_v1', '0e526c3f0014549549170ace'),
    ('hb', False): _G1,
    ('hb', True): _G1,
}
_GEOMETRY = {'eeg': (1, 400), 'hb': (2, 20)}  # channels, samples per two seconds
_STEP = 10_000
_CPU_LOCK = RLock()


@contextmanager
def _bounded_cpu():
    # torch thread count is process-global; restore the caller's setting even on failure.
    with _CPU_LOCK:
        previous = torch.get_num_threads()
        try:
            if previous > 2:
                torch.set_num_threads(2)
            yield
        finally:
            if previous > 2:
                torch.set_num_threads(previous)


def _fixed_file(root, relative):
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError(f'expected fixed regular source file (no redirection): {relative}')
    return path


def _json(root, relative):
    return json.loads(_fixed_file(root, relative).read_text(encoding='utf-8'))


@dataclass(frozen=True)
class LegacyObservationControl:
    modality: str
    continuous: bool
    provenance: dict
    _model: GlobalLatentComponentTokenizer

    def reconstruct(self, windows, *, return_ids=False, batch_size=64):
        """Return float32 X in the caller's [N,15,C,T] G0 X-scale coordinates.

        Supply complete, finite, already normalized encoder windows, HbO then
        HbR. Retain any *scoring* mask upstream; an encoder window must not be
        zeroed using a target mask. No local centering or rescaling takes place.
        If return_ids=True, return (X, {'S': uint8[N,15], 'O': uint8[N,15]})
        for hard controls, or (X, None) for the continuous controls.
        """
        c, t = _GEOMETRY[self.modality]
        if (not isinstance(windows, np.ndarray) or np.ma.isMaskedArray(windows)
                or windows.dtype.kind != 'f' or windows.ndim != 4
                or windows.shape[1:] != (15, c, t) or not len(windows)):
            raise ValueError(f'expected nonempty floating G0 X [N,15,{c},{t}]')
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        size = min(int(batch_size), 64)
        x_pred = np.empty(windows.shape, dtype=np.float32)
        ids = {b: np.empty((len(windows), 15), dtype=np.uint8) for b in ('S', 'O')} if return_ids and not self.continuous else None
        with _bounded_cpu(), torch.inference_mode():
            for start in range(0, len(windows), size):
                sl = slice(start, start + size)
                # The retained G0 owner flattens time, then color within each patch.
                xt = windows[sl].transpose(0, 1, 3, 2).astype(np.float32, order='C', copy=True)
                x = torch.from_numpy(xt.reshape(len(xt), 15, t*c))
                if self.continuous:
                    result = self._model(x, quantize=False)
                else:
                    # Public owner: encode_continuous(X) -> hard IDs -> decode_ids(IDs).
                    # Unlike a straight-through latent, no X side channel reaches the decoder.
                    result = self._model.forward_frozen_ids(x)
                    if ids is not None:
                        for b in ids:
                            ids[b][sl] = result['ids'][b].numpy()
                pred = result['sum']
                if not torch.isfinite(pred).all():
                    raise ValueError('nonfinite reconstructed X from frozen control')
                x_pred[sl] = pred.reshape(len(xt), 15, t, c).permute(0, 1, 3, 2).numpy()
        return (x_pred, ids) if return_ids else x_pred


def load_legacy_observation_control(project_root, modality, *, continuous=False):
    """Load one allowlisted, completed CPU checkpoint once for streaming batches."""
    if modality not in _GEOMETRY or not isinstance(continuous, bool):
        raise ValueError("modality must be 'eeg' or 'hb' and continuous must be boolean")
    root = Path(project_root).resolve(strict=True)
    run_name, schema, cfg_id = _SOURCES[modality, continuous]
    run = _RUNS / run_name
    contract = _CONFIGS / f'{schema}.yaml'
    cfg = yaml.safe_load(_fixed_file(root, contract).read_text(encoding='utf-8'))
    resolved = _json(root, run / 'resolved_config.json')
    if (sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:24] != cfg_id
            or resolved.get('config') != cfg or resolved.get('project_root') != str(root)):
        raise ValueError('legacy source contract/recorded root mismatch')
    arch = 'waveform8' if modality == 'eeg' else 'waveform'
    candidate = 'continuous' if continuous else 'vq128'
    task = dict(modality=modality, architecture=arch, candidate=candidate)
    task_id = f'{modality}__{arch}__{candidate}'
    job = run / 'models' / task_id
    plan = _json(root, run / 'plan.json')
    if (plan.get('G0_run') != _G0 or cfg['data']['run'] != _G0
            or plan.get('manifest_identity') != _G0_INPUT or plan.get('scales_identity') != _G0_SCALES
            or task not in plan['tasks'] or task not in cfg['tasks']):
        raise ValueError('legacy task must match the pinned G0 input and scale identities')
    setting = {**cfg['model'], **cfg['architectures'][arch]}
    if (cfg['training']['seed'] != 17 or cfg['training']['total_steps'] != _STEP
            or setting['patches'] != 15 or setting['codebook_size'] != 128 or not setting['waveform_io']
            or setting.get('target_input') or setting.get('frozen_source') or setting.get('input_color_weights')):
        raise ValueError('only the declared X-only waveform controls are allowed')
    signature = dict(config=cfg_id, input=_G0_INPUT, scales=_G0_SCALES, task=task, benchmark=False)
    record = _json(root, job / 'summary.json')
    if (record.get('status') != 'completed' or record.get('completed_step') != _STEP
            or record.get('task_id') != task_id or record.get('signature') != signature):
        raise ValueError('legacy source is not the exact completed 10000-step task')
    checkpoint = _fixed_file(root, job / 'checkpoint_step_10000.pt')
    # No general-pickle fallback. Ignore optimizer/RNG; never resume or refit.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('signature') != signature or saved.get('step') != _STEP
            or saved.get('decoder_quantization_blend', 1.) != 1.):
        raise ValueError('legacy checkpoint task/step/decoder identity mismatch')
    c, t = _GEOMETRY[modality]
    setting.update(input_dim=t*c, channels=c)
    setting.setdefault('sample_strides', [5] if modality == 'hb' else [5, 2, 5])
    with _bounded_cpu(), torch.random.fork_rng(devices=[]):
        model = GlobalLatentComponentTokenizer(setting, codebook_size=128, hard=not continuous)
        model.load_state_dict(saved['model_state'], strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
            raise ValueError('nonfinite legacy model weights')
        if not continuous and not all(bool(b['quantizer'].initialized) for b in model.branches.values()):
            raise ValueError('uninitialized frozen hard codebook; refitting is forbidden')
        model.requires_grad_(False).eval()
    provenance = dict(checkpoint=str(checkpoint.relative_to(root)), contract=str(contract), step=_STEP,
                      signature=signature, seed=17, input='G0 training-scale X only',
                      nominal_bits_per_two_seconds_per_location=None if continuous else 14)
    return LegacyObservationControl(modality, continuous, provenance, model)


def legacy_observation_reconstruct(project_root, modality, windows, *, continuous=False, return_ids=False, batch_size=64):
    """One-shot X-only replay; use load_legacy_observation_control() for repeated batches."""
    control = load_legacy_observation_control(project_root, modality, continuous=continuous)
    return control.reconstruct(windows, return_ids=return_ids, batch_size=batch_size)
