"""
Visualization and signal analysis helpers for EEG and fNIRS segments.

This module supports two common workflows:
1. Plot raw and processed segments directly from numpy arrays or torch tensors.
2. Extract raw windows from existing dataset objects and generate waveform,
    power spectrum, differential entropy, and synchronized multimodal plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, welch

from .eeg_fnirs_dataset import EEGfNIRSDataset, MultiModalEEGfNIRSDataset


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


EEG_FREQUENCY_BANDS: Mapping[str, Tuple[float, float]] = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0),
}

FNIRS_FREQUENCY_BANDS: Mapping[str, Tuple[float, float]] = {
    'very_low': (0.01, 0.05),
    'low': (0.05, 0.10),
    'mid': (0.10, 0.20),
    'task': (0.20, 0.40),
}


@dataclass
class SegmentSnapshot:
    """Container for a single signal segment and its metadata."""

    signal: np.ndarray
    sample_rate: float
    channel_names: List[str]
    modality: str
    subject_id: Optional[int] = None
    session_idx: Optional[int] = None
    trial_idx: Optional[int] = None
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _to_numpy_2d(signal: ArrayLike, channel_first: Optional[bool] = None) -> np.ndarray:
    if isinstance(signal, torch.Tensor):
        array = signal.detach().cpu().numpy()
    else:
        array = np.asarray(signal)

    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {array.shape}")

    if channel_first is None:
        channel_first = array.shape[0] <= array.shape[1]

    if not channel_first:
        array = array.T

    return np.asarray(array, dtype=np.float64)


def _normalize_per_channel(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + eps
    return (signal - mean) / std


def _slice_and_pad_window(
    cnt: np.ndarray,
    start: int,
    end: int,
    target_samples: int,
) -> np.ndarray:
    if start < 0:
        start = 0
        end = target_samples
    if end > cnt.shape[0]:
        end = cnt.shape[0]
        start = max(0, end - target_samples)

    window = cnt[start:end, :]
    if window.shape[0] < target_samples:
        pad_size = target_samples - window.shape[0]
        window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')

    return window.T.astype(np.float64, copy=False)


def _select_channel_indices(
    total_channels: int,
    channel_indices: Optional[Sequence[int]],
    max_channels: int,
) -> List[int]:
    if channel_indices is None:
        return list(range(min(total_channels, max_channels)))

    selected = []
    for index in channel_indices:
        if index < 0 or index >= total_channels:
            raise IndexError(f"Channel index {index} is out of bounds for {total_channels} channels")
        selected.append(int(index))
    return selected[:max_channels]


def _default_frequency_bands(modality: str) -> Dict[str, Tuple[float, float]]:
    normalized = modality.lower()
    if normalized == 'eeg':
        return dict(EEG_FREQUENCY_BANDS)
    if normalized in {'fnirs', 'nirs'}:
        return dict(FNIRS_FREQUENCY_BANDS)
    raise ValueError(f"Unsupported modality: {modality}")


def _safe_filter_band(
    signal: np.ndarray,
    sample_rate: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = sample_rate * 0.5
    if nyquist <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    low = max(0.0, float(low))
    high = min(float(high), nyquist * 0.99)

    if high <= 0:
        return signal.copy()

    if low <= 0 and high >= nyquist * 0.99:
        return signal.copy()
    if low <= 0:
        sos = butter(order, high / nyquist, btype='lowpass', output='sos')
    elif high >= nyquist * 0.99:
        sos = butter(order, low / nyquist, btype='highpass', output='sos')
    elif low >= high:
        return signal.copy()
    else:
        sos = butter(order, [low / nyquist, high / nyquist], btype='bandpass', output='sos')

    try:
        return sosfiltfilt(sos, signal, axis=1)
    except ValueError:
        return signal.copy()


def compute_power_spectrum(
    signal: ArrayLike,
    sample_rate: float,
    channel_first: Optional[bool] = None,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch power spectral density for each channel."""
    signal_2d = _to_numpy_2d(signal, channel_first=channel_first)

    if nperseg is None:
        nperseg = min(signal_2d.shape[1], max(64, int(sample_rate * 2)))
    nperseg = max(8, min(signal_2d.shape[1], nperseg))

    frequencies, psd = welch(signal_2d, fs=sample_rate, axis=1, nperseg=nperseg)
    return frequencies, psd


def compute_differential_entropy(
    signal: ArrayLike,
    sample_rate: float,
    bands: Mapping[str, Tuple[float, float]],
    channel_first: Optional[bool] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute Gaussian differential entropy for each channel and frequency band."""
    signal_2d = _to_numpy_2d(signal, channel_first=channel_first)
    entropies = []

    for low, high in bands.values():
        band_signal = _safe_filter_band(signal_2d, sample_rate, low, high)
        variance = np.var(band_signal, axis=1) + eps
        entropies.append(0.5 * np.log(2.0 * np.pi * np.e * variance))

    return np.stack(entropies, axis=1)


def resolve_preprocessing_config(
    preprocessing: Optional[Mapping[str, Any]],
    modality: str,
) -> Dict[str, Any]:
    """Resolve the effective temporal filtering settings for a modality."""
    config = dict(preprocessing or {})
    normalized_modality = modality.lower()
    resolved: Dict[str, Any] = {}

    if normalized_modality == 'eeg':
        bandpass = config.get('bandpass')
        if isinstance(bandpass, (list, tuple)) and len(bandpass) == 2:
            resolved['bandpass'] = (float(bandpass[0]), float(bandpass[1]))
        else:
            if 'highpass' in config:
                resolved['highpass'] = float(config['highpass'])
            if 'lowpass' in config:
                resolved['lowpass'] = float(config['lowpass'])
    elif normalized_modality in {'fnirs', 'nirs'}:
        if 'highpass' in config:
            resolved['highpass'] = float(config['highpass'])
        if 'lowpass' in config:
            resolved['lowpass'] = float(config['lowpass'])

        if 'highpass' not in resolved and 'lowpass' not in resolved:
            bandpass = config.get('bandpass')
            if isinstance(bandpass, (list, tuple)) and len(bandpass) == 2:
                resolved['bandpass'] = (float(bandpass[0]), float(bandpass[1]))
    else:
        raise ValueError(f'Unsupported modality: {modality}')

    if 'resample_rate' in config:
        resolved['resample_rate'] = float(config['resample_rate'])

    return resolved


def summarize_preprocessing_config(
    preprocessing: Optional[Mapping[str, Any]],
    modality: str,
) -> Dict[str, Any]:
    """Return a concise human-readable summary of effective filtering settings."""
    resolved = resolve_preprocessing_config(preprocessing, modality)
    summary = {'modality': modality.lower()}

    if 'bandpass' in resolved:
        summary['filter'] = {
            'type': 'bandpass',
            'low_hz': resolved['bandpass'][0],
            'high_hz': resolved['bandpass'][1],
        }
    elif 'highpass' in resolved and 'lowpass' in resolved:
        summary['filter'] = {
            'type': 'bandpass',
            'low_hz': resolved['highpass'],
            'high_hz': resolved['lowpass'],
        }
    elif 'lowpass' in resolved:
        summary['filter'] = {
            'type': 'lowpass',
            'high_hz': resolved['lowpass'],
        }
    elif 'highpass' in resolved:
        summary['filter'] = {
            'type': 'highpass',
            'low_hz': resolved['highpass'],
        }
    else:
        summary['filter'] = {'type': 'none'}

    if 'resample_rate' in resolved:
        summary['resample_rate_hz'] = resolved['resample_rate']

    return summary


def apply_preprocessing_config(
    signal: ArrayLike,
    sample_rate: float,
    modality: str,
    preprocessing: Optional[Mapping[str, Any]],
    channel_first: Optional[bool] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Apply the effective config-driven temporal filtering to a signal segment."""
    signal_2d = _to_numpy_2d(signal, channel_first=channel_first)
    resolved = resolve_preprocessing_config(preprocessing, modality)
    processed = signal_2d.copy()

    if 'bandpass' in resolved:
        low, high = resolved['bandpass']
        processed = _safe_filter_band(processed, sample_rate, low, high)
    elif 'highpass' in resolved and 'lowpass' in resolved:
        processed = _safe_filter_band(processed, sample_rate, resolved['highpass'], resolved['lowpass'])
    elif 'lowpass' in resolved:
        processed = _safe_filter_band(processed, sample_rate, 0.0, resolved['lowpass'])
    elif 'highpass' in resolved:
        processed = _safe_filter_band(processed, sample_rate, resolved['highpass'], sample_rate * 0.5)

    if normalize:
        processed = _normalize_per_channel(processed)

    return processed


def extract_dataset_segment(
    dataset: EEGfNIRSDataset,
    idx: int,
    normalize: Optional[bool] = None,
) -> SegmentSnapshot:
    """Extract a raw or normalized segment from EEGfNIRSDataset."""
    trial = dataset.trials[idx]
    cnt_list, _, info = dataset._get_subject_data(trial.subject_id)
    cnt = cnt_list[trial.session_idx]

    if dataset.modality == 'eeg':
        start = trial.eeg_start_sample
        end = trial.eeg_end_sample
    else:
        start = trial.nirs_start_sample
        end = trial.nirs_end_sample

    window = _slice_and_pad_window(cnt, start, end, dataset.window_samples)
    apply_normalization = normalize if normalize is not None else dataset.normalize
    if apply_normalization:
        window = _normalize_per_channel(window)

    return SegmentSnapshot(
        signal=window,
        sample_rate=float(info['fs']),
        channel_names=list(info['clab']),
        modality=dataset.modality,
        subject_id=trial.subject_id,
        session_idx=trial.session_idx,
        trial_idx=trial.trial_idx,
        label=trial.label,
        metadata={
            'task_type': trial.task_type,
            'onset_time_ms': trial.onset_time_ms,
            'window_offset_ms': dataset.window_offset_ms,
        },
    )


def extract_multimodal_dataset_segment(
    dataset: MultiModalEEGfNIRSDataset,
    idx: int,
    normalize: Optional[bool] = None,
) -> Dict[str, SegmentSnapshot]:
    """Extract aligned EEG and fNIRS segments from MultiModalEEGfNIRSDataset."""
    trial = dataset.trials[idx]
    apply_normalization = normalize if normalize is not None else dataset.normalize

    eeg_cnt_list, _, eeg_info = dataset._get_eeg_data(trial.subject_id)
    eeg_cnt = eeg_cnt_list[trial.session_idx]
    eeg_samples = int(dataset.window_duration_s * eeg_info['fs'])
    eeg_window = _slice_and_pad_window(
        eeg_cnt,
        trial.eeg_start_sample,
        trial.eeg_end_sample,
        eeg_samples,
    )
    if apply_normalization:
        eeg_window = _normalize_per_channel(eeg_window)

    fnirs_cnt_list, _, fnirs_info = dataset._get_nirs_data(trial.subject_id)
    fnirs_cnt = fnirs_cnt_list[trial.session_idx]
    fnirs_samples = int(dataset.window_duration_s * fnirs_info['fs'])
    fnirs_window = _slice_and_pad_window(
        fnirs_cnt,
        trial.nirs_start_sample,
        trial.nirs_end_sample,
        fnirs_samples,
    )
    if apply_normalization:
        fnirs_window = _normalize_per_channel(fnirs_window)

    common_metadata = {
        'task_type': trial.task_type,
        'onset_time_ms': trial.onset_time_ms,
        'window_duration_s': dataset.window_duration_s,
        'window_offset_ms': dataset.window_offset_ms,
    }

    return {
        'eeg': SegmentSnapshot(
            signal=eeg_window,
            sample_rate=float(eeg_info['fs']),
            channel_names=list(eeg_info['clab']),
            modality='eeg',
            subject_id=trial.subject_id,
            session_idx=trial.session_idx,
            trial_idx=trial.trial_idx,
            label=trial.label,
            metadata=dict(common_metadata),
        ),
        'fnirs': SegmentSnapshot(
            signal=fnirs_window,
            sample_rate=float(fnirs_info['fs']),
            channel_names=list(fnirs_info['clab']),
            modality='fnirs',
            subject_id=trial.subject_id,
            session_idx=trial.session_idx,
            trial_idx=trial.trial_idx,
            label=trial.label,
            metadata=dict(common_metadata),
        ),
    }


class SignalSegmentVisualizer:
    """Visualization toolkit for EEG and fNIRS signal segments."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        figsize_base: Tuple[float, float] = (12.0, 8.0),
        dpi: int = 150,
        style: str = 'seaborn-v0_8-whitegrid',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize_base = figsize_base
        self.dpi = dpi
        self.generated_figures: List[Path] = []

        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use('seaborn-whitegrid')
            except OSError:
                pass

        self.colors = {
            'raw': '#2E86AB',
            'processed': '#A23B72',
            'accent': '#F18F01',
            'success': '#2ECC71',
            'warning': '#F39C12',
            'light': '#95A5A6',
        }

    def _save_or_return(
        self,
        fig: plt.Figure,
        save: bool,
        filename: str,
    ) -> Optional[plt.Figure]:
        if save:
            path = self.output_dir / filename
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            self.generated_figures.append(path)
            plt.close(fig)
            return None
        return fig

    def _build_segment_title(
        self,
        modality: str,
        metadata: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
    ) -> str:
        if title:
            return title
        if not metadata:
            return f"{modality.upper()} signal segment"

        subject_id = metadata.get('subject_id')
        session_idx = metadata.get('session_idx')
        trial_idx = metadata.get('trial_idx')
        pieces = [modality.upper()]
        if subject_id is not None:
            pieces.append(f"subject {subject_id}")
        if session_idx is not None:
            pieces.append(f"session {session_idx}")
        if trial_idx is not None:
            pieces.append(f"trial {trial_idx}")
        return ' | '.join(pieces)

    def plot_waveform_comparison(
        self,
        original: ArrayLike,
        sample_rate: float,
        processed: Optional[ArrayLike] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_indices: Optional[Sequence[int]] = None,
        max_channels: int = 6,
        channel_first: Optional[bool] = None,
        modality: str = 'eeg',
        metadata: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        save: bool = True,
        filename: str = 'waveform_comparison.png',
    ) -> Optional[plt.Figure]:
        original_2d = _to_numpy_2d(original, channel_first=channel_first)
        processed_2d = None if processed is None else _to_numpy_2d(processed, channel_first=channel_first)

        if processed_2d is not None and processed_2d.shape != original_2d.shape:
            raise ValueError('original and processed must have identical shapes')

        selected = _select_channel_indices(original_2d.shape[0], channel_indices, max_channels)
        time_axis = np.arange(original_2d.shape[1]) / sample_rate
        ncols = 2 if processed_2d is not None else 1

        fig, axes = plt.subplots(
            len(selected),
            ncols,
            figsize=(self.figsize_base[0], max(2.8 * len(selected), self.figsize_base[1])),
            sharex=True,
        )
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 0:
            axes = axes.reshape(1, 1)
        elif axes.ndim == 1:
            axes = axes.reshape(len(selected), ncols)

        names = list(channel_names) if channel_names is not None else [f'ch_{i}' for i in range(original_2d.shape[0])]

        for row, channel_idx in enumerate(selected):
            channel_label = names[channel_idx] if channel_idx < len(names) else f'ch_{channel_idx}'
            raw_ax = axes[row, 0]
            raw_ax.plot(time_axis, original_2d[channel_idx], color=self.colors['raw'], linewidth=1.2)
            raw_ax.set_ylabel(channel_label)
            raw_ax.grid(True, alpha=0.25)
            if row == 0:
                raw_ax.set_title('Original')

            if processed_2d is not None:
                proc_ax = axes[row, 1]
                proc_ax.plot(time_axis, processed_2d[channel_idx], color=self.colors['processed'], linewidth=1.2)
                proc_ax.grid(True, alpha=0.25)
                if row == 0:
                    proc_ax.set_title('After preprocess')

        axes[-1, 0].set_xlabel('Time (s)')
        if processed_2d is not None:
            axes[-1, 1].set_xlabel('Time (s)')

        fig.suptitle(self._build_segment_title(modality, metadata, title), fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
        return self._save_or_return(fig, save=save, filename=filename)

    def plot_power_spectrum(
        self,
        original: ArrayLike,
        sample_rate: float,
        processed: Optional[ArrayLike] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_indices: Optional[Sequence[int]] = None,
        max_channels: int = 6,
        channel_first: Optional[bool] = None,
        modality: str = 'eeg',
        metadata: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        max_frequency: Optional[float] = None,
        save: bool = True,
        filename: str = 'power_spectrum.png',
    ) -> Optional[plt.Figure]:
        original_2d = _to_numpy_2d(original, channel_first=channel_first)
        processed_2d = None if processed is None else _to_numpy_2d(processed, channel_first=channel_first)

        selected = _select_channel_indices(original_2d.shape[0], channel_indices, max_channels)
        frequencies, original_psd = compute_power_spectrum(original_2d[selected], sample_rate, channel_first=True)
        processed_psd = None
        if processed_2d is not None:
            _, processed_psd = compute_power_spectrum(processed_2d[selected], sample_rate, channel_first=True)

        if max_frequency is None:
            max_frequency = 45.0 if modality.lower() == 'eeg' else min(0.5, sample_rate * 0.5)
        freq_mask = frequencies <= max_frequency

        names = list(channel_names) if channel_names is not None else [f'ch_{i}' for i in range(original_2d.shape[0])]
        fig, axes = plt.subplots(
            len(selected),
            1,
            figsize=(self.figsize_base[0], max(2.5 * len(selected), self.figsize_base[1] * 0.8)),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        for row, channel_idx in enumerate(selected):
            label = names[channel_idx] if channel_idx < len(names) else f'ch_{channel_idx}'
            axes[row].plot(
                frequencies[freq_mask],
                original_psd[row, freq_mask],
                color=self.colors['raw'],
                linewidth=1.6,
                label='Original',
            )
            if processed_psd is not None:
                axes[row].plot(
                    frequencies[freq_mask],
                    processed_psd[row, freq_mask],
                    color=self.colors['processed'],
                    linewidth=1.6,
                    linestyle='--',
                    label='After preprocess',
                )
            axes[row].set_ylabel(label)
            axes[row].set_yscale('log')
            axes[row].grid(True, alpha=0.25)
            if row == 0:
                axes[row].legend(loc='upper right')

        axes[-1].set_xlabel('Frequency (Hz)')
        fig.suptitle(self._build_segment_title(modality, metadata, title) + ' | Power spectrum', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._save_or_return(fig, save=save, filename=filename)

    def plot_differential_entropy(
        self,
        original: ArrayLike,
        sample_rate: float,
        processed: Optional[ArrayLike] = None,
        bands: Optional[Mapping[str, Tuple[float, float]]] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_indices: Optional[Sequence[int]] = None,
        max_channels: int = 12,
        channel_first: Optional[bool] = None,
        modality: str = 'eeg',
        metadata: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        save: bool = True,
        filename: str = 'differential_entropy.png',
    ) -> Optional[plt.Figure]:
        original_2d = _to_numpy_2d(original, channel_first=channel_first)
        processed_2d = None if processed is None else _to_numpy_2d(processed, channel_first=channel_first)
        selected = _select_channel_indices(original_2d.shape[0], channel_indices, max_channels)

        band_map = dict(bands) if bands is not None else _default_frequency_bands(modality)
        band_names = list(band_map.keys())
        names = list(channel_names) if channel_names is not None else [f'ch_{i}' for i in range(original_2d.shape[0])]
        selected_names = [names[idx] if idx < len(names) else f'ch_{idx}' for idx in selected]

        original_de = compute_differential_entropy(
            original_2d[selected],
            sample_rate=sample_rate,
            bands=band_map,
            channel_first=True,
        )
        processed_de = None
        if processed_2d is not None:
            processed_de = compute_differential_entropy(
                processed_2d[selected],
                sample_rate=sample_rate,
                bands=band_map,
                channel_first=True,
            )

        ncols = 2 if processed_de is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(self.figsize_base[0], self.figsize_base[1] * 0.8))
        axes = np.atleast_1d(axes)

        image = axes[0].imshow(original_de, aspect='auto', cmap='viridis')
        axes[0].set_title('Original')
        axes[0].set_xticks(range(len(band_names)))
        axes[0].set_xticklabels(band_names, rotation=30, ha='right')
        axes[0].set_yticks(range(len(selected_names)))
        axes[0].set_yticklabels(selected_names)
        axes[0].set_xlabel('Band')
        axes[0].set_ylabel('Channel')
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

        if processed_de is not None:
            image = axes[1].imshow(processed_de, aspect='auto', cmap='magma')
            axes[1].set_title('After preprocess')
            axes[1].set_xticks(range(len(band_names)))
            axes[1].set_xticklabels(band_names, rotation=30, ha='right')
            axes[1].set_yticks(range(len(selected_names)))
            axes[1].set_yticklabels(selected_names)
            axes[1].set_xlabel('Band')
            axes[1].set_ylabel('Channel')
            fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(self._build_segment_title(modality, metadata, title) + ' | Differential entropy', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return self._save_or_return(fig, save=save, filename=filename)

    def plot_segment_overview(
        self,
        original: ArrayLike,
        sample_rate: float,
        processed: Optional[ArrayLike] = None,
        bands: Optional[Mapping[str, Tuple[float, float]]] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_indices: Optional[Sequence[int]] = None,
        max_channels: int = 6,
        channel_first: Optional[bool] = None,
        modality: str = 'eeg',
        metadata: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        save: bool = True,
        filename: str = 'segment_overview.png',
    ) -> Optional[plt.Figure]:
        original_2d = _to_numpy_2d(original, channel_first=channel_first)
        processed_2d = None if processed is None else _to_numpy_2d(processed, channel_first=channel_first)
        selected = _select_channel_indices(original_2d.shape[0], channel_indices, max_channels)

        names = list(channel_names) if channel_names is not None else [f'ch_{i}' for i in range(original_2d.shape[0])]
        selected_names = [names[idx] if idx < len(names) else f'ch_{idx}' for idx in selected]
        time_axis = np.arange(original_2d.shape[1]) / sample_rate
        frequencies, original_psd = compute_power_spectrum(original_2d[selected], sample_rate, channel_first=True)
        processed_psd = None
        if processed_2d is not None:
            _, processed_psd = compute_power_spectrum(processed_2d[selected], sample_rate, channel_first=True)

        band_map = dict(bands) if bands is not None else _default_frequency_bands(modality)
        original_de = compute_differential_entropy(original_2d[selected], sample_rate, band_map, channel_first=True)
        processed_de = None
        if processed_2d is not None:
            processed_de = compute_differential_entropy(processed_2d[selected], sample_rate, band_map, channel_first=True)

        fig = plt.figure(figsize=(self.figsize_base[0] * 1.2, self.figsize_base[1] * 1.1))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.25)

        waveform_ax = fig.add_subplot(grid[0, 0])
        focus_idx = selected[0]
        waveform_ax.plot(time_axis, original_2d[focus_idx], color=self.colors['raw'], linewidth=1.5, label='Original')
        if processed_2d is not None:
            waveform_ax.plot(
                time_axis,
                processed_2d[focus_idx],
                color=self.colors['processed'],
                linewidth=1.5,
                linestyle='--',
                label='After preprocess',
            )
        waveform_ax.set_title(f'Waveform ({selected_names[0]})')
        waveform_ax.set_xlabel('Time (s)')
        waveform_ax.grid(True, alpha=0.25)
        waveform_ax.legend(loc='upper right')

        spectrum_ax = fig.add_subplot(grid[0, 1])
        spectrum_ax.plot(
            frequencies,
            np.mean(original_psd, axis=0),
            color=self.colors['raw'],
            linewidth=1.8,
            label='Original mean PSD',
        )
        if processed_psd is not None:
            spectrum_ax.plot(
                frequencies,
                np.mean(processed_psd, axis=0),
                color=self.colors['processed'],
                linewidth=1.8,
                linestyle='--',
                label='Processed mean PSD',
            )
        spectrum_ax.set_title('Mean power spectrum')
        spectrum_ax.set_xlabel('Frequency (Hz)')
        spectrum_ax.set_yscale('log')
        spectrum_ax.grid(True, alpha=0.25)
        spectrum_ax.legend(loc='upper right')

        de_ax = fig.add_subplot(grid[1, 0])
        de_image = de_ax.imshow(original_de, aspect='auto', cmap='viridis')
        de_ax.set_title('Original differential entropy')
        de_ax.set_xticks(range(len(band_map)))
        de_ax.set_xticklabels(list(band_map.keys()), rotation=30, ha='right')
        de_ax.set_yticks(range(len(selected_names)))
        de_ax.set_yticklabels(selected_names)
        fig.colorbar(de_image, ax=de_ax, fraction=0.046, pad=0.04)

        meta_ax = fig.add_subplot(grid[1, 1])
        if processed_de is not None:
            meta_image = meta_ax.imshow(processed_de, aspect='auto', cmap='magma')
            meta_ax.set_title('Processed differential entropy')
            meta_ax.set_xticks(range(len(band_map)))
            meta_ax.set_xticklabels(list(band_map.keys()), rotation=30, ha='right')
            meta_ax.set_yticks(range(len(selected_names)))
            meta_ax.set_yticklabels(selected_names)
            fig.colorbar(meta_image, ax=meta_ax, fraction=0.046, pad=0.04)
        else:
            meta_ax.axis('off')
            lines = [
                f'Modality: {modality}',
                f'Sample rate: {sample_rate:.3f} Hz',
                f'Selected channels: {", ".join(selected_names)}',
            ]
            if metadata:
                for key, value in metadata.items():
                    lines.append(f'{key}: {value}')
            meta_ax.text(0.02, 0.98, '\n'.join(lines), va='top', ha='left', fontsize=11)

        fig.suptitle(self._build_segment_title(modality, metadata, title), fontsize=14, fontweight='bold')
        fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
        return self._save_or_return(fig, save=save, filename=filename)

    def plot_synchronized_modalities(
        self,
        eeg_original: ArrayLike,
        eeg_sample_rate: float,
        fnirs_original: ArrayLike,
        fnirs_sample_rate: float,
        eeg_processed: Optional[ArrayLike] = None,
        fnirs_processed: Optional[ArrayLike] = None,
        eeg_channel_names: Optional[Sequence[str]] = None,
        fnirs_channel_names: Optional[Sequence[str]] = None,
        eeg_channel_idx: int = 0,
        fnirs_channel_idx: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        save: bool = True,
        filename: str = 'synchronized_modalities.png',
    ) -> Optional[plt.Figure]:
        """Plot EEG and fNIRS segments on aligned time axes for the same trial."""
        eeg_original_2d = _to_numpy_2d(eeg_original, channel_first=True)
        fnirs_original_2d = _to_numpy_2d(fnirs_original, channel_first=True)
        eeg_processed_2d = None if eeg_processed is None else _to_numpy_2d(eeg_processed, channel_first=True)
        fnirs_processed_2d = None if fnirs_processed is None else _to_numpy_2d(fnirs_processed, channel_first=True)

        if eeg_channel_idx < 0 or eeg_channel_idx >= eeg_original_2d.shape[0]:
            raise IndexError(f'EEG channel index {eeg_channel_idx} is out of bounds')
        if fnirs_channel_idx < 0 or fnirs_channel_idx >= fnirs_original_2d.shape[0]:
            raise IndexError(f'fNIRS channel index {fnirs_channel_idx} is out of bounds')

        eeg_time = np.arange(eeg_original_2d.shape[1]) / eeg_sample_rate
        fnirs_time = np.arange(fnirs_original_2d.shape[1]) / fnirs_sample_rate
        duration = max(eeg_time[-1] if len(eeg_time) > 0 else 0.0, fnirs_time[-1] if len(fnirs_time) > 0 else 0.0)

        eeg_names = list(eeg_channel_names) if eeg_channel_names is not None else [f'eeg_{i}' for i in range(eeg_original_2d.shape[0])]
        fnirs_names = list(fnirs_channel_names) if fnirs_channel_names is not None else [f'fnirs_{i}' for i in range(fnirs_original_2d.shape[0])]
        eeg_label = eeg_names[eeg_channel_idx] if eeg_channel_idx < len(eeg_names) else f'eeg_{eeg_channel_idx}'
        fnirs_label = fnirs_names[fnirs_channel_idx] if fnirs_channel_idx < len(fnirs_names) else f'fnirs_{fnirs_channel_idx}'

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize_base[0] * 1.1, self.figsize_base[1] * 1.1), sharex=True)

        axes[0].plot(eeg_time, eeg_original_2d[eeg_channel_idx], color=self.colors['raw'], linewidth=1.2, label='Original')
        if eeg_processed_2d is not None:
            axes[0].plot(eeg_time, eeg_processed_2d[eeg_channel_idx], color=self.colors['processed'], linewidth=1.2, linestyle='--', label='Filtered')
        axes[0].set_ylabel(eeg_label)
        axes[0].set_title('EEG')
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc='upper right')

        axes[1].plot(fnirs_time, fnirs_original_2d[fnirs_channel_idx], color=self.colors['raw'], linewidth=1.2, label='Original')
        if fnirs_processed_2d is not None:
            axes[1].plot(fnirs_time, fnirs_processed_2d[fnirs_channel_idx], color=self.colors['processed'], linewidth=1.2, linestyle='--', label='Filtered')
        axes[1].set_ylabel(fnirs_label)
        axes[1].set_title('fNIRS')
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc='upper right')

        eeg_overlay_source = eeg_processed_2d if eeg_processed_2d is not None else eeg_original_2d
        fnirs_overlay_source = fnirs_processed_2d if fnirs_processed_2d is not None else fnirs_original_2d
        eeg_overlay = _normalize_per_channel(eeg_overlay_source[[eeg_channel_idx]])[0]
        fnirs_overlay = _normalize_per_channel(fnirs_overlay_source[[fnirs_channel_idx]])[0]

        axes[2].plot(eeg_time, eeg_overlay, color=self.colors['primary'] if 'primary' in self.colors else self.colors['raw'], linewidth=1.5, label=f'EEG filtered ({eeg_label})')
        axes[2].plot(fnirs_time, fnirs_overlay, color=self.colors['success'], linewidth=1.5, label=f'fNIRS filtered ({fnirs_label})')
        axes[2].set_ylabel('z-score for display')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Aligned filtered signals')
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc='upper right')
        axes[2].set_xlim(0.0, duration)

        plot_title = title or 'Synchronized EEG and fNIRS segments'
        if metadata and not title:
            subject_id = metadata.get('subject_id')
            session_idx = metadata.get('session_idx')
            trial_idx = metadata.get('trial_idx')
            pieces = ['EEG/fNIRS synchronized view']
            if subject_id is not None:
                pieces.append(f'subject {subject_id}')
            if session_idx is not None:
                pieces.append(f'session {session_idx}')
            if trial_idx is not None:
                pieces.append(f'trial {trial_idx}')
            plot_title = ' | '.join(pieces)

        fig.suptitle(plot_title, fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        return self._save_or_return(fig, save=save, filename=filename)

    def get_generated_figures(self) -> List[Path]:
        return list(self.generated_figures)


def _snapshot_metadata(snapshot: SegmentSnapshot) -> Dict[str, Any]:
    metadata = dict(snapshot.metadata)
    metadata.update(
        {
            'subject_id': snapshot.subject_id,
            'session_idx': snapshot.session_idx,
            'trial_idx': snapshot.trial_idx,
            'label': snapshot.label,
        }
    )
    return metadata


def _build_file_prefix(snapshot: SegmentSnapshot) -> str:
    subject = 'na' if snapshot.subject_id is None else str(snapshot.subject_id)
    session = 'na' if snapshot.session_idx is None else str(snapshot.session_idx)
    trial = 'na' if snapshot.trial_idx is None else str(snapshot.trial_idx)
    return f"{snapshot.modality}_subject{subject}_session{session}_trial{trial}"


def visualize_dataset_sample(
    dataset: EEGfNIRSDataset,
    idx: int,
    output_dir: Union[str, Path],
    channel_indices: Optional[Sequence[int]] = None,
    max_channels: int = 6,
) -> Dict[str, Path]:
    """Generate waveform, PSD, DE, and overview figures for a single-modality sample."""
    raw_snapshot = extract_dataset_segment(dataset, idx, normalize=False)
    processed_snapshot = extract_dataset_segment(dataset, idx, normalize=True)

    visualizer = SignalSegmentVisualizer(output_dir)
    metadata = _snapshot_metadata(raw_snapshot)
    prefix = _build_file_prefix(raw_snapshot)

    visualizer.plot_waveform_comparison(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_snapshot.signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        max_channels=max_channels,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_waveform.png',
    )
    visualizer.plot_power_spectrum(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_snapshot.signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        max_channels=max_channels,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_psd.png',
    )
    visualizer.plot_differential_entropy(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_snapshot.signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_de.png',
    )
    visualizer.plot_segment_overview(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_snapshot.signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        max_channels=max_channels,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_overview.png',
    )

    figures = visualizer.get_generated_figures()
    return {path.stem: path for path in figures}


def visualize_filtered_dataset_sample(
    dataset: EEGfNIRSDataset,
    idx: int,
    preprocessing: Optional[Mapping[str, Any]],
    output_dir: Union[str, Path],
    channel_indices: Optional[Sequence[int]] = None,
    max_channels: int = 6,
) -> Dict[str, Path]:
    """Generate single-modality figures using raw vs real filtered signals."""
    raw_snapshot = extract_dataset_segment(dataset, idx, normalize=False)
    processed_signal = apply_preprocessing_config(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        modality=raw_snapshot.modality,
        preprocessing=preprocessing,
        channel_first=True,
        normalize=False,
    )

    visualizer = SignalSegmentVisualizer(output_dir)
    metadata = _snapshot_metadata(raw_snapshot)
    metadata['effective_preprocessing'] = summarize_preprocessing_config(preprocessing, raw_snapshot.modality)
    prefix = _build_file_prefix(raw_snapshot)

    visualizer.plot_waveform_comparison(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        max_channels=max_channels,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_waveform.png',
    )
    visualizer.plot_power_spectrum(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        max_channels=max_channels,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_psd.png',
    )
    visualizer.plot_differential_entropy(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_de.png',
    )
    visualizer.plot_segment_overview(
        raw_snapshot.signal,
        sample_rate=raw_snapshot.sample_rate,
        processed=processed_signal,
        channel_names=raw_snapshot.channel_names,
        channel_indices=channel_indices,
        max_channels=max_channels,
        modality=raw_snapshot.modality,
        metadata=metadata,
        filename=f'{prefix}_overview.png',
    )

    figures = visualizer.get_generated_figures()
    return {path.stem: path for path in figures}


def visualize_synchronized_filtered_sample(
    dataset: MultiModalEEGfNIRSDataset,
    idx: int,
    eeg_preprocessing: Optional[Mapping[str, Any]],
    fnirs_preprocessing: Optional[Mapping[str, Any]],
    output_dir: Union[str, Path],
    eeg_channel_idx: int = 0,
    fnirs_channel_idx: int = 0,
) -> Dict[str, Path]:
    """Generate aligned EEG/fNIRS plots with real filtered signals."""
    snapshots = extract_multimodal_dataset_segment(dataset, idx, normalize=False)
    eeg_snapshot = snapshots['eeg']
    fnirs_snapshot = snapshots['fnirs']

    eeg_processed = apply_preprocessing_config(
        eeg_snapshot.signal,
        sample_rate=eeg_snapshot.sample_rate,
        modality='eeg',
        preprocessing=eeg_preprocessing,
        channel_first=True,
        normalize=False,
    )
    fnirs_processed = apply_preprocessing_config(
        fnirs_snapshot.signal,
        sample_rate=fnirs_snapshot.sample_rate,
        modality='fnirs',
        preprocessing=fnirs_preprocessing,
        channel_first=True,
        normalize=False,
    )

    visualizer = SignalSegmentVisualizer(output_dir)
    metadata = _snapshot_metadata(eeg_snapshot)
    metadata['eeg_preprocessing'] = summarize_preprocessing_config(eeg_preprocessing, 'eeg')
    metadata['fnirs_preprocessing'] = summarize_preprocessing_config(fnirs_preprocessing, 'fnirs')
    prefix = f"sync_subject{eeg_snapshot.subject_id}_session{eeg_snapshot.session_idx}_trial{eeg_snapshot.trial_idx}"

    visualizer.plot_synchronized_modalities(
        eeg_original=eeg_snapshot.signal,
        eeg_sample_rate=eeg_snapshot.sample_rate,
        eeg_processed=eeg_processed,
        eeg_channel_names=eeg_snapshot.channel_names,
        eeg_channel_idx=eeg_channel_idx,
        fnirs_original=fnirs_snapshot.signal,
        fnirs_sample_rate=fnirs_snapshot.sample_rate,
        fnirs_processed=fnirs_processed,
        fnirs_channel_names=fnirs_snapshot.channel_names,
        fnirs_channel_idx=fnirs_channel_idx,
        metadata=metadata,
        filename=f'{prefix}_aligned.png',
    )

    figures = visualizer.get_generated_figures()
    return {path.stem: path for path in figures}


def visualize_multimodal_dataset_sample(
    dataset: MultiModalEEGfNIRSDataset,
    idx: int,
    output_dir: Union[str, Path],
    eeg_channel_indices: Optional[Sequence[int]] = None,
    fnirs_channel_indices: Optional[Sequence[int]] = None,
    max_channels: int = 6,
) -> Dict[str, Dict[str, Path]]:
    """Generate visualization suites for aligned EEG and fNIRS segments."""
    raw_segments = extract_multimodal_dataset_segment(dataset, idx, normalize=False)
    processed_segments = extract_multimodal_dataset_segment(dataset, idx, normalize=True)

    generated: Dict[str, Dict[str, Path]] = {}
    channel_map = {
        'eeg': eeg_channel_indices,
        'fnirs': fnirs_channel_indices,
    }

    for modality, raw_snapshot in raw_segments.items():
        visualizer = SignalSegmentVisualizer(Path(output_dir) / modality)
        processed_snapshot = processed_segments[modality]
        metadata = _snapshot_metadata(raw_snapshot)
        prefix = _build_file_prefix(raw_snapshot)
        indices = channel_map[modality]

        visualizer.plot_waveform_comparison(
            raw_snapshot.signal,
            sample_rate=raw_snapshot.sample_rate,
            processed=processed_snapshot.signal,
            channel_names=raw_snapshot.channel_names,
            channel_indices=indices,
            max_channels=max_channels,
            modality=modality,
            metadata=metadata,
            filename=f'{prefix}_waveform.png',
        )
        visualizer.plot_power_spectrum(
            raw_snapshot.signal,
            sample_rate=raw_snapshot.sample_rate,
            processed=processed_snapshot.signal,
            channel_names=raw_snapshot.channel_names,
            channel_indices=indices,
            max_channels=max_channels,
            modality=modality,
            metadata=metadata,
            filename=f'{prefix}_psd.png',
        )
        visualizer.plot_differential_entropy(
            raw_snapshot.signal,
            sample_rate=raw_snapshot.sample_rate,
            processed=processed_snapshot.signal,
            channel_names=raw_snapshot.channel_names,
            channel_indices=indices,
            modality=modality,
            metadata=metadata,
            filename=f'{prefix}_de.png',
        )
        visualizer.plot_segment_overview(
            raw_snapshot.signal,
            sample_rate=raw_snapshot.sample_rate,
            processed=processed_snapshot.signal,
            channel_names=raw_snapshot.channel_names,
            channel_indices=indices,
            max_channels=max_channels,
            modality=modality,
            metadata=metadata,
            filename=f'{prefix}_overview.png',
        )

        generated[modality] = {path.stem: path for path in visualizer.get_generated_figures()}

    return generated


__all__ = [
    'SegmentSnapshot',
    'SignalSegmentVisualizer',
    'EEG_FREQUENCY_BANDS',
    'FNIRS_FREQUENCY_BANDS',
    'resolve_preprocessing_config',
    'summarize_preprocessing_config',
    'apply_preprocessing_config',
    'compute_power_spectrum',
    'compute_differential_entropy',
    'extract_dataset_segment',
    'extract_multimodal_dataset_segment',
    'visualize_dataset_sample',
    'visualize_filtered_dataset_sample',
    'visualize_multimodal_dataset_sample',
    'visualize_synchronized_filtered_sample',
]