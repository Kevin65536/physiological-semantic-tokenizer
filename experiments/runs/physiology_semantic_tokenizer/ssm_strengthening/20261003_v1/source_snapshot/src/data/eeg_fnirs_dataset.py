"""
EEG+NIRS Single-Trial Dataset Loader

This module implements data loading for the TU Berlin EEG+NIRS BCI dataset.
Dataset URL: https://doc.ml.tu-berlin.de/hBCI/

Key findings from data exploration (2026-01-14):
- EEG and NIRS markers have a fixed offset (~51-55s) due to different recording start times
- Event intervals match well between modalities (mean diff ~0ms, std ~40ms)
- Labels are identical between modalities for corresponding events
- Session 0,2,4: Motor Imagery (LMI/RMI), Session 1,3,5: Mental Arithmetic (MA/BL)

Channel Structure:
- EEG: 32 channels total (30 EEG + 2 EOG: 'VEOG', 'HEOG')
- fNIRS: 72 channels total, arranged as 36 paired optical tracks
    - Raw channel labels in this dataset use `highWL` / `lowWL` suffixes
    - The Single-Trial files do not convert those tracks into HbO / HbR concentration channels
    - Some other datasets expose paired branches as `_O` / `_R`
    - Current implementation supports filtering one branch of the pair via the legacy `hbo_only` / `hbr_only` flags
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Union
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt
import torch
from torch.utils.data import Dataset, DataLoader

from .fnirs_standardization import (
    get_fnirs_measurement_contract,
    standardization_config_from_mapping,
    standardize_fnirs_record,
)


def uses_record_level_fnirs_standardization(preprocessing: Optional[dict]) -> bool:
    return standardization_config_from_mapping(preprocessing) is not None


# =============================================================================
# Channel Filtering Utilities
# =============================================================================

def get_eeg_channel_mask(channel_names: List[str], exclude_eog: bool = True) -> np.ndarray:
    """
    Get a boolean mask for EEG channels, optionally excluding EOG channels.
    
    Args:
        channel_names: List of channel names
        exclude_eog: If True, exclude EOG channels (EOGv, EOGh, VEOG, HEOG, etc.)
        
    Returns:
        Boolean mask array where True = include channel
    """
    eog_patterns = ['EOG', 'eog', 'VEOG', 'HEOG']
    mask = np.ones(len(channel_names), dtype=bool)
    
    if exclude_eog:
        for i, name in enumerate(channel_names):
            for pattern in eog_patterns:
                if pattern in name:
                    mask[i] = False
                    break
    
    return mask


def get_fnirs_channel_mask(
    channel_names: List[str], 
    hbo_only: bool = True,
    hbr_only: bool = False
) -> np.ndarray:
    """
    Get a boolean mask for paired fNIRS channels.
    
    The Single-Trial dataset exposes raw `highWL` / `lowWL` optical tracks.
    Other datasets in this project may instead use `_O` / `_R` suffixes for a
    paired branch. The flag names are kept for backward compatibility with
    existing call sites, but they should be read as branch selectors rather than
    proof that every dataset already stores HbO / HbR concentrations.
    
    Note: Some datasets may use '_O' and '_R' suffixes instead.
    
    Args:
        channel_names: List of channel names
        hbo_only: Legacy selector for the `highWL` / `_O` branch
        hbr_only: Legacy selector for the `lowWL` / `_R` branch
        
    Returns:
        Boolean mask array where True = include channel
    """
    if hbo_only and hbr_only:
        raise ValueError("Cannot set both hbo_only and hbr_only to True")
    
    mask = np.ones(len(channel_names), dtype=bool)
    
    if hbo_only:
        # Keep the `highWL` / `_O` branch.
        for i, name in enumerate(channel_names):
            if 'highWL' in name or name.endswith('_O'):
                mask[i] = True
            elif 'lowWL' in name or name.endswith('_R'):
                mask[i] = False
            # If neither pattern matches, keep the channel (conservative)
    elif hbr_only:
        # Keep the `lowWL` / `_R` branch.
        for i, name in enumerate(channel_names):
            if 'lowWL' in name or name.endswith('_R'):
                mask[i] = True
            elif 'highWL' in name or name.endswith('_O'):
                mask[i] = False
            # If neither pattern matches, keep the channel (conservative)
    
    return mask


def resolve_preprocessing_config(preprocessing: Optional[dict], modality: str) -> Dict[str, float]:
    """Resolve effective temporal filtering settings for a modality."""
    config = dict(preprocessing or {})
    normalized_modality = modality.lower()
    resolved: Dict[str, float] = {}

    if normalized_modality == 'eeg':
        bandpass = config.get('bandpass')
        if isinstance(bandpass, (list, tuple)) and len(bandpass) == 2:
            resolved['low_hz'] = float(bandpass[0])
            resolved['high_hz'] = float(bandpass[1])
        else:
            if 'highpass' in config:
                resolved['low_hz'] = float(config['highpass'])
            if 'lowpass' in config:
                resolved['high_hz'] = float(config['lowpass'])
    elif normalized_modality == 'fnirs':
        if 'highpass' in config:
            resolved['low_hz'] = float(config['highpass'])
        if 'lowpass' in config:
            resolved['high_hz'] = float(config['lowpass'])
        elif 'low_hz' not in resolved:
            bandpass = config.get('bandpass')
            if isinstance(bandpass, (list, tuple)) and len(bandpass) == 2:
                resolved['low_hz'] = float(bandpass[0])
                resolved['high_hz'] = float(bandpass[1])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return resolved


def apply_temporal_filter(
    signal: np.ndarray,
    sample_rate: float,
    modality: str,
    preprocessing: Optional[dict],
    order: int = 4,
) -> np.ndarray:
    """Apply config-driven temporal filtering to continuous data.

    Args:
        signal: Continuous data with shape (n_samples, n_channels)
    """
    working_dtype = np.float32
    measurement_cfg = standardization_config_from_mapping(preprocessing) if modality.lower() == 'fnirs' else None
    if measurement_cfg is not None:
        contract = get_fnirs_measurement_contract(
            str(measurement_cfg['dataset_id']),
            str(measurement_cfg['signal_key']),
        )
        signal = standardize_fnirs_record(
            signal,
            sample_rate_hz=float(sample_rate),
            contract=contract,
            baseline_rule=str(measurement_cfg.get('baseline_rule', 'robust_linear')),
            trend_blocks=int(measurement_cfg.get('trend_blocks', 20)),
        ).values
    resolved = resolve_preprocessing_config(preprocessing, modality)
    if not resolved:
        return signal.astype(working_dtype, copy=True)

    filtered = signal.astype(working_dtype, copy=True)
    nyquist = sample_rate * 0.5
    low_hz = max(0.0, float(resolved.get('low_hz', 0.0)))
    high_hz = min(float(resolved.get('high_hz', nyquist * 0.99)), nyquist * 0.99)

    if high_hz <= 0:
        return filtered
    if low_hz <= 0 and high_hz >= nyquist * 0.99:
        return filtered

    if low_hz <= 0:
        sos = butter(order, high_hz / nyquist, btype='lowpass', output='sos')
    elif high_hz >= nyquist * 0.99:
        sos = butter(order, low_hz / nyquist, btype='highpass', output='sos')
    elif low_hz >= high_hz:
        return filtered
    else:
        sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype='bandpass', output='sos')

    sos = sos.astype(working_dtype, copy=False)

    try:
        if filtered.ndim == 1:
            return sosfiltfilt(sos, filtered, axis=0).astype(working_dtype, copy=False)

        filtered_output = np.empty_like(filtered)
        for channel_idx in range(filtered.shape[1]):
            filtered_output[:, channel_idx] = sosfiltfilt(sos, filtered[:, channel_idx], axis=0)
        return filtered_output
    except ValueError:
        return filtered


def normalize_window(
    window: np.ndarray,
    mode: str,
    session_mean: Optional[np.ndarray] = None,
    session_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalize channel-first window using either window or session statistics."""
    if mode == 'none':
        return window

    if mode == 'window':
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True) + 1e-8
        return (window - mean) / std

    if mode == 'session':
        if session_mean is None or session_std is None:
            raise ValueError('Session statistics are required for session normalization')
        return (window - session_mean[:, None]) / (session_std[:, None] + 1e-8)

    raise ValueError(f'Unsupported normalization mode: {mode}')


@dataclass
class TrialInfo:
    """Information about a single trial."""
    subject_id: int
    session_idx: int
    trial_idx: int
    label: int  # 0 or 1 (left/right for MI, arithmetic/baseline for MA)
    task_type: str  # 'motor_imagery' or 'mental_arithmetic'
    eeg_start_sample: int
    eeg_end_sample: int
    nirs_start_sample: int
    nirs_end_sample: int
    onset_time_ms: float  # Event onset time in ms


@dataclass  
class SyncInfo:
    """Synchronization information between EEG and NIRS."""
    offset_ms: float  # NIRS time - EEG time at first event
    interval_diff_mean_ms: float
    interval_diff_std_ms: float
    interval_diff_max_ms: float
    labels_match: bool


def load_mat_struct(filepath: str) -> dict:
    """Load a .mat file with struct_as_record=False for easier access."""
    return sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)


class BBCIDataLoader:
    """
    Loader for BBCI Toolbox format data (EEG+NIRS Single-Trial dataset).
    
    Data structure:
    - cnt: continuous data, shape (n_samples, n_channels)
    - mrk: markers with time (ms), y (one-hot labels), className
    - mnt: montage with channel locations
    
    Sessions:
    - 0, 2, 4: Motor Imagery (left/right hand)
    - 1, 3, 5: Mental Arithmetic (arithmetic/baseline)
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic', 'both'] = 'motor_imagery',
        modality: Literal['eeg', 'fnirs', 'both'] = 'both',
        use_artifact_data: bool = True,  # Use data with ocular artifacts (recommended)
    ):
        """
        Initialize the data loader.
        
        Args:
            data_root: Path to 'EEG+NIRS Single-Trial' directory
            subject_ids: List of subject IDs (1-29), None for all
            task: Which task to load
            modality: Which modality to load
            use_artifact_data: If True, use 'with occular artifact' folder for EEG
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids or list(range(1, 30))
        self.task = task
        self.modality = modality
        self.use_artifact_data = use_artifact_data
        
        # Session indices for each task
        self.mi_sessions = [0, 2, 4]  # Motor Imagery
        self.ma_sessions = [1, 3, 5]  # Mental Arithmetic
        
        # Validate paths
        self._validate_paths()
        
    def _validate_paths(self):
        """Check that data directories exist."""
        eeg_dir = self.data_root / 'EEG_01-29'
        nirs_dir = self.data_root / 'NIRS_01-29'
        
        if self.modality in ['eeg', 'both'] and not eeg_dir.exists():
            raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")
        if self.modality in ['fnirs', 'both'] and not nirs_dir.exists():
            raise FileNotFoundError(f"NIRS directory not found: {nirs_dir}")
            
    def _get_subject_dir(self, subject_id: int, modality: str) -> Path:
        """Get the directory for a specific subject and modality."""
        subdir = 'EEG_01-29' if modality == 'eeg' else 'NIRS_01-29'
        subject_dir = self.data_root / subdir / f'subject {subject_id:02d}'
        
        if modality == 'eeg' and self.use_artifact_data:
            artifact_dir = subject_dir / 'with occular artifact'
            if artifact_dir.exists():
                return artifact_dir
        return subject_dir
    
    def load_subject_data(
        self, 
        subject_id: int,
        modality: str
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load all session data for a subject.
        
        Returns:
            cnt_list: List of 6 continuous data arrays, shape (n_samples, n_channels)
            mrk_list: List of 6 marker dicts with 'time', 'y', 'className'
            info: Dict with 'fs', 'clab' (channel labels)
        """
        subject_dir = self._get_subject_dir(subject_id, modality)
        
        cnt_data = load_mat_struct(str(subject_dir / 'cnt.mat'))
        mrk_data = load_mat_struct(str(subject_dir / 'mrk.mat'))
        
        cnt_list = []
        mrk_list = []
        
        for i in range(6):
            cnt_struct = cnt_data['cnt'][i]
            mrk_struct = mrk_data['mrk'][i]
            
            cnt_list.append(cnt_struct.x)
            mrk_list.append({
                'time': mrk_struct.time,
                'y': mrk_struct.y,
                'className': mrk_struct.className,
            })
        
        # Get metadata from first session
        info = {
            'fs': cnt_data['cnt'][0].fs,
            'clab': cnt_data['cnt'][0].clab,
        }
        
        return cnt_list, mrk_list, info
    
    def check_synchronization(
        self, 
        subject_id: int,
        session_idx: int = 0
    ) -> SyncInfo:
        """
        Check synchronization between EEG and NIRS markers for a session.
        
        Returns:
            SyncInfo with offset and alignment statistics
        """
        _, eeg_mrk, _ = self.load_subject_data(subject_id, 'eeg')
        _, nirs_mrk, _ = self.load_subject_data(subject_id, 'fnirs')
        
        eeg_time = eeg_mrk[session_idx]['time']
        nirs_time = nirs_mrk[session_idx]['time']
        
        # Check offset at first event
        offset_ms = float(nirs_time[0] - eeg_time[0])
        
        # Check interval alignment
        eeg_intervals = np.diff(eeg_time)
        nirs_intervals = np.diff(nirs_time)
        interval_diff = nirs_intervals - eeg_intervals
        
        # Check label matching
        eeg_labels = np.argmax(eeg_mrk[session_idx]['y'], axis=0)
        nirs_labels = np.argmax(nirs_mrk[session_idx]['y'], axis=0)
        labels_match = bool(np.all(eeg_labels == nirs_labels))
        
        return SyncInfo(
            offset_ms=offset_ms,
            interval_diff_mean_ms=float(np.mean(interval_diff)),
            interval_diff_std_ms=float(np.std(interval_diff)),
            interval_diff_max_ms=float(np.max(np.abs(interval_diff))),
            labels_match=labels_match,
        )
    
    def check_all_synchronization(self) -> Dict[int, List[SyncInfo]]:
        """Check synchronization for all subjects and sessions."""
        results = {}
        for subject_id in self.subject_ids:
            results[subject_id] = []
            for session_idx in range(6):
                try:
                    sync_info = self.check_synchronization(subject_id, session_idx)
                    results[subject_id].append(sync_info)
                except Exception as e:
                    print(f"Error checking subject {subject_id} session {session_idx}: {e}")
                    results[subject_id].append(None)
        return results
    

class EEGfNIRSDataset(Dataset):
    """
    PyTorch Dataset for EEG+NIRS Single-Trial data.
    
    Extracts fixed-length windows around event markers for training tokenizers.
    
    Channel Filtering:
    - EEG: Can exclude EOG channels (EOGv, EOGh) via exclude_eog parameter
    - fNIRS: Can filter one branch of each paired optical track via the legacy hbo_only/hbr_only parameters
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
        modality: Literal['eeg', 'fnirs'] = 'eeg',
        window_samples: int = 512,  # 2.56s @ 200Hz for EEG, ~51s @ 10Hz for NIRS
        window_offset_ms: float = 0,  # Offset from event onset (can be negative for pre-stimulus)
        normalize: bool = True,
        normalization_mode: Literal['window', 'session', 'none'] = 'window',
        preprocessing: Optional[dict] = None,
        use_artifact_data: bool = True,
        # Channel filtering options
        exclude_eog: bool = True,  # For EEG: exclude EOG channels
        hbo_only: bool = True,     # Legacy fNIRS selector for the `highWL` / `_O` branch
        hbr_only: bool = False,    # Legacy fNIRS selector for the `lowWL` / `_R` branch
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to 'EEG+NIRS Single-Trial' directory
            subject_ids: List of subject IDs (1-29), None for all
            task: Which task to use
            modality: 'eeg' or 'fnirs'
            window_samples: Number of samples per window
            window_offset_ms: Offset from event onset in milliseconds
            normalize: Whether to z-score normalize each window
            use_artifact_data: If True, use 'with occular artifact' folder for EEG
            exclude_eog: If True, exclude EOG channels from EEG data (default: True)
            hbo_only: Legacy selector for the `highWL` / `_O` fNIRS branch (default: True)
            hbr_only: Legacy selector for the `lowWL` / `_R` fNIRS branch (default: False)
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids or list(range(1, 30))
        self.task = task
        self.modality = modality
        self.window_samples = window_samples
        self.window_offset_ms = window_offset_ms
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.preprocessing = dict(preprocessing or {})
        self.use_artifact_data = use_artifact_data
        
        # Channel filtering options
        self.exclude_eog = exclude_eog
        self.hbo_only = hbo_only
        self.hbr_only = hbr_only
        
        # Loader for data access
        self.loader = BBCIDataLoader(
            data_root=data_root,
            subject_ids=self.subject_ids,
            task=task,
            modality=modality,
            use_artifact_data=use_artifact_data,
        )
        
        # Session indices for task
        if task == 'motor_imagery':
            self.session_indices = [0, 2, 4]
        else:
            self.session_indices = [1, 3, 5]
        
        # Build trial index
        self.trials: List[TrialInfo] = []
        self._build_trial_index()
        
        # Cache for loaded data
        self._raw_data_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._processed_data_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._session_stats_cache: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        
        # Cache for channel mask (computed once per subject)
        self._channel_mask_cache: Dict[int, np.ndarray] = {}
        self._filtered_channel_names_cache: Dict[int, List[str]] = {}
        
    def _build_trial_index(self):
        """Build an index of all available trials."""
        for subject_id in self.subject_ids:
            try:
                _, mrk_list, info = self.loader.load_subject_data(subject_id, self.modality)
                fs = info['fs']
                
                for session_idx in self.session_indices:
                    mrk = mrk_list[session_idx]
                    n_trials = len(mrk['time'])
                    labels = np.argmax(mrk['y'], axis=0)
                    
                    for trial_idx in range(n_trials):
                        onset_time_ms = mrk['time'][trial_idx]
                        
                        # Calculate sample indices
                        offset_samples = int(self.window_offset_ms * fs / 1000)
                        start_sample = int(onset_time_ms * fs / 1000) + offset_samples
                        end_sample = start_sample + self.window_samples
                        
                        trial = TrialInfo(
                            subject_id=subject_id,
                            session_idx=session_idx,
                            trial_idx=trial_idx,
                            label=int(labels[trial_idx]),
                            task_type=self.task,
                            eeg_start_sample=start_sample if self.modality == 'eeg' else 0,
                            eeg_end_sample=end_sample if self.modality == 'eeg' else 0,
                            nirs_start_sample=start_sample if self.modality == 'fnirs' else 0,
                            nirs_end_sample=end_sample if self.modality == 'fnirs' else 0,
                            onset_time_ms=onset_time_ms,
                        )
                        self.trials.append(trial)
                        
            except Exception as e:
                print(f"Warning: Could not load subject {subject_id}: {e}")
                
    def _cache_subject_data(self, subject_id: int):
        """Load, filter, and cache both raw-filtered and temporally processed session data."""
        if subject_id not in self._raw_data_cache:
            cnt_list, mrk_list, info = self.loader.load_subject_data(subject_id, self.modality)
            
            # Get channel mask for filtering
            channel_names = list(info['clab'])
            if self.modality == 'eeg':
                channel_mask = get_eeg_channel_mask(channel_names, exclude_eog=self.exclude_eog)
            else:  # fnirs
                channel_mask = get_fnirs_channel_mask(
                    channel_names, 
                    hbo_only=self.hbo_only, 
                    hbr_only=self.hbr_only
                )
            
            # Apply channel filtering to all sessions
            raw_filtered_cnt_list = []
            processed_cnt_list = []
            session_stats = []
            fs = float(info['fs'])
            for cnt in cnt_list:
                # cnt shape: (n_samples, n_channels) -> filter channels
                filtered_cnt = cnt[:, channel_mask].astype(np.float32, copy=False)
                raw_filtered_cnt_list.append(filtered_cnt)

                processed_cnt = apply_temporal_filter(
                    filtered_cnt,
                    sample_rate=fs,
                    modality=self.modality,
                    preprocessing=self.preprocessing,
                )
                processed_cnt_list.append(processed_cnt)

                channel_mean = processed_cnt.mean(axis=0)
                channel_std = processed_cnt.std(axis=0) + 1e-8
                session_stats.append((channel_mean, channel_std))
            
            # Update channel info
            filtered_info = info.copy()
            filtered_channel_names = [name for i, name in enumerate(channel_names) if channel_mask[i]]
            filtered_info['clab'] = filtered_channel_names
            filtered_info['original_clab'] = channel_names
            filtered_info['channel_mask'] = channel_mask
            
            self._raw_data_cache[subject_id] = (raw_filtered_cnt_list, mrk_list, filtered_info)
            self._processed_data_cache[subject_id] = (processed_cnt_list, mrk_list, filtered_info)
            self._session_stats_cache[subject_id] = session_stats
            self._channel_mask_cache[subject_id] = channel_mask
            self._filtered_channel_names_cache[subject_id] = filtered_channel_names

    def _get_subject_data(
        self,
        subject_id: int,
        processed: bool = True,
    ) -> Tuple[List[np.ndarray], List[dict], dict]:
        """Get cached subject data after channel filtering and optional temporal preprocessing."""
        self._cache_subject_data(subject_id)
        if processed:
            return self._processed_data_cache[subject_id]
        return self._raw_data_cache[subject_id]

    def _get_session_statistics(self, subject_id: int, session_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._cache_subject_data(subject_id)
        return self._session_stats_cache[subject_id][session_idx]

    def get_session_continuous_data(
        self,
        subject_id: int,
        session_idx: int,
        processed: bool = True,
        normalized: bool = False,
    ) -> np.ndarray:
        """Get continuous session data as channel-first array for visualization or analysis."""
        cnt_list, _, _ = self._get_subject_data(subject_id, processed=processed)
        session = cnt_list[session_idx].T.copy()

        if (
            normalized
            and self.normalize
            and self.normalization_mode == 'session'
            and not (self.modality == 'fnirs' and uses_record_level_fnirs_standardization(self.preprocessing))
        ):
            session_mean, session_std = self._get_session_statistics(subject_id, session_idx)
            session = normalize_window(session, 'session', session_mean, session_std)

        return session

    def get_session_markers(self, subject_id: int, session_idx: int) -> dict:
        """Get raw marker dict for a given session."""
        _, mrk_list, _ = self._get_subject_data(subject_id, processed=False)
        return mrk_list[session_idx]

    def get_session_trial_regions(
        self,
        subject_id: int,
        session_idx: int,
        window_duration_s: float,
        offset_ms: float = 0.0,
    ) -> List[Dict[str, Union[int, float, str]]]:
        """Return time regions that would be extracted as trial windows on a full session timeline."""
        markers = self.get_session_markers(subject_id, session_idx)
        labels = np.argmax(markers['y'], axis=0)
        class_names = markers.get('className')
        regions = []

        for trial_idx, onset_ms in enumerate(np.asarray(markers['time'], dtype=float)):
            label = int(labels[trial_idx])
            if class_names is not None and len(class_names) > label:
                label_name = str(class_names[label])
            else:
                label_name = str(label)

            start_s = (float(onset_ms) + offset_ms) / 1000.0
            end_s = start_s + float(window_duration_s)
            regions.append(
                {
                    'trial_idx': trial_idx,
                    'label': label,
                    'label_name': label_name,
                    'start_s': start_s,
                    'end_s': end_s,
                    'onset_s': float(onset_ms) / 1000.0,
                }
            )

        return regions
    
    def __len__(self) -> int:
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trial.
        
        Returns:
            Dict with:
                - 'data': tensor of shape (n_channels, window_samples)
                - 'label': tensor of shape () with class index
                - 'subject_id': int
                - 'session_idx': int
                - 'trial_idx': int
        """
        trial = self.trials[idx]
        cnt_list, _, _ = self._get_subject_data(trial.subject_id, processed=True)
        
        # Get continuous data for this session
        cnt = cnt_list[trial.session_idx]  # shape: (n_samples, n_channels)
        
        # Extract window
        if self.modality == 'eeg':
            start = trial.eeg_start_sample
            end = trial.eeg_end_sample
        else:
            start = trial.nirs_start_sample
            end = trial.nirs_end_sample
        
        # Handle boundary cases
        if start < 0:
            start = 0
            end = self.window_samples
        if end > cnt.shape[0]:
            end = cnt.shape[0]
            start = max(0, end - self.window_samples)
            
        window = cnt[start:end, :]  # shape: (window_samples, n_channels)
        
        # Pad if necessary
        if window.shape[0] < self.window_samples:
            pad_size = self.window_samples - window.shape[0]
            window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')
        
        # Transpose to (n_channels, window_samples)
        window = window.T
        
        # Normalize
        normalization_mode = self.normalization_mode if self.normalize else 'none'
        if self.modality == 'fnirs' and uses_record_level_fnirs_standardization(self.preprocessing):
            normalization_mode = 'none'
        if normalization_mode == 'session':
            session_mean, session_std = self._get_session_statistics(trial.subject_id, trial.session_idx)
            window = normalize_window(window, 'session', session_mean, session_std)
        elif normalization_mode == 'window':
            window = normalize_window(window, 'window')
        
        return {
            'data': torch.from_numpy(window).float(),
            'label': torch.tensor(trial.label, dtype=torch.long),
            'subject_id': trial.subject_id,
            'session_idx': trial.session_idx,
            'trial_idx': trial.trial_idx,
        }
    
    def get_channel_names(self) -> List[str]:
        """Get the channel names for the current modality (after filtering)."""
        if len(self.trials) == 0:
            return []
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_subject_data(first_subject, processed=True)
        return list(info['clab'])
    
    def get_num_channels(self) -> int:
        """Get the number of channels (after filtering)."""
        return len(self.get_channel_names())
    
    def get_original_channel_names(self) -> List[str]:
        """Get the original channel names before filtering."""
        if len(self.trials) == 0:
            return []
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_subject_data(first_subject, processed=True)
        return list(info.get('original_clab', info['clab']))
    
    def get_sample_rate(self) -> float:
        """Get the sampling rate for the current modality."""
        if len(self.trials) == 0:
            return 0.0
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_subject_data(first_subject, processed=True)
        return float(info['fs'])


class MultiModalEEGfNIRSDataset(Dataset):
    """
    PyTorch Dataset for synchronized EEG+NIRS data.
    
    Returns aligned windows from both modalities for the same trial.
    Note: Due to different sampling rates, window durations are matched, not sample counts.
    
    Channel Filtering:
    - EEG: Can exclude EOG channels (EOGv, EOGh) via exclude_eog parameter
    - fNIRS: Can filter one branch of each paired optical track via the legacy hbo_only/hbr_only parameters
    """
    
    def __init__(
        self,
        data_root: str,
        subject_ids: Optional[List[int]] = None,
        task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
        window_duration_s: float = 2.5,  # Window duration in seconds
        window_offset_ms: float = 0,
        normalize: bool = True,
        normalization_mode: Literal['window', 'session', 'none'] = 'window',
        eeg_preprocessing: Optional[dict] = None,
        fnirs_preprocessing: Optional[dict] = None,
        use_artifact_data: bool = True,
        # Channel filtering options
        exclude_eog: bool = True,  # For EEG: exclude EOG channels
        hbo_only: bool = True,     # Legacy fNIRS selector for the `highWL` / `_O` branch
        hbr_only: bool = False,    # Legacy fNIRS selector for the `lowWL` / `_R` branch
    ):
        """
        Initialize the multimodal dataset.
        
        Args:
            data_root: Path to 'EEG+NIRS Single-Trial' directory
            subject_ids: List of subject IDs (1-29), None for all
            task: Which task to use
            window_duration_s: Window duration in seconds (applied to both modalities)
            window_offset_ms: Offset from event onset in milliseconds
            normalize: Whether to z-score normalize each window
            use_artifact_data: If True, use 'with occular artifact' folder for EEG
            exclude_eog: If True, exclude EOG channels from EEG data (default: True)
            hbo_only: Legacy selector for the `highWL` / `_O` fNIRS branch (default: True)
            hbr_only: Legacy selector for the `lowWL` / `_R` fNIRS branch (default: False)
        """
        self.data_root = Path(data_root)
        self.subject_ids = subject_ids or list(range(1, 30))
        self.task = task
        self.window_duration_s = window_duration_s
        self.window_offset_ms = window_offset_ms
        self.normalize = normalize
        self.normalization_mode = normalization_mode
        self.eeg_preprocessing = dict(eeg_preprocessing or {})
        self.fnirs_preprocessing = dict(fnirs_preprocessing or {})
        
        # Channel filtering options
        self.exclude_eog = exclude_eog
        self.hbo_only = hbo_only
        self.hbr_only = hbr_only
        
        # Loaders for both modalities
        self.eeg_loader = BBCIDataLoader(
            data_root=data_root,
            subject_ids=self.subject_ids,
            task=task,
            modality='eeg',
            use_artifact_data=use_artifact_data,
        )
        self.nirs_loader = BBCIDataLoader(
            data_root=data_root,
            subject_ids=self.subject_ids,
            task=task,
            modality='fnirs',
            use_artifact_data=use_artifact_data,
        )
        
        # Session indices for task
        if task == 'motor_imagery':
            self.session_indices = [0, 2, 4]
        else:
            self.session_indices = [1, 3, 5]
        
        # Build trial index
        self.trials: List[TrialInfo] = []
        self._build_trial_index()
        
        # Cache for loaded data
        self._eeg_raw_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._eeg_processed_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._eeg_session_stats_cache: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self._nirs_raw_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._nirs_processed_cache: Dict[int, Tuple[List[np.ndarray], List[dict], dict]] = {}
        self._nirs_session_stats_cache: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        
    def _build_trial_index(self):
        """Build an index of all available trials with both modalities."""
        for subject_id in self.subject_ids:
            try:
                # Load both modalities to verify synchronization
                _, eeg_mrk_list, eeg_info = self.eeg_loader.load_subject_data(subject_id, 'eeg')
                _, nirs_mrk_list, nirs_info = self.nirs_loader.load_subject_data(subject_id, 'fnirs')
                
                eeg_fs = eeg_info['fs']
                nirs_fs = nirs_info['fs']
                
                for session_idx in self.session_indices:
                    eeg_mrk = eeg_mrk_list[session_idx]
                    nirs_mrk = nirs_mrk_list[session_idx]
                    
                    n_trials = len(eeg_mrk['time'])
                    eeg_labels = np.argmax(eeg_mrk['y'], axis=0)
                    nirs_labels = np.argmax(nirs_mrk['y'], axis=0)
                    
                    # Verify labels match
                    if not np.all(eeg_labels == nirs_labels):
                        print(f"Warning: Label mismatch for subject {subject_id} session {session_idx}")
                        continue
                    
                    for trial_idx in range(n_trials):
                        eeg_onset_ms = eeg_mrk['time'][trial_idx]
                        nirs_onset_ms = nirs_mrk['time'][trial_idx]
                        
                        # Calculate EEG sample indices
                        eeg_offset_samples = int(self.window_offset_ms * eeg_fs / 1000)
                        eeg_window_samples = int(self.window_duration_s * eeg_fs)
                        eeg_start = int(eeg_onset_ms * eeg_fs / 1000) + eeg_offset_samples
                        eeg_end = eeg_start + eeg_window_samples
                        
                        # Calculate NIRS sample indices
                        nirs_offset_samples = int(self.window_offset_ms * nirs_fs / 1000)
                        nirs_window_samples = int(self.window_duration_s * nirs_fs)
                        nirs_start = int(nirs_onset_ms * nirs_fs / 1000) + nirs_offset_samples
                        nirs_end = nirs_start + nirs_window_samples
                        
                        trial = TrialInfo(
                            subject_id=subject_id,
                            session_idx=session_idx,
                            trial_idx=trial_idx,
                            label=int(eeg_labels[trial_idx]),
                            task_type=self.task,
                            eeg_start_sample=eeg_start,
                            eeg_end_sample=eeg_end,
                            nirs_start_sample=nirs_start,
                            nirs_end_sample=nirs_end,
                            onset_time_ms=eeg_onset_ms,
                        )
                        self.trials.append(trial)
                        
            except Exception as e:
                print(f"Warning: Could not load subject {subject_id}: {e}")
    
    def _cache_eeg_data(self, subject_id: int):
        if subject_id not in self._eeg_processed_cache:
            cnt_list, mrk_list, info = self.eeg_loader.load_subject_data(subject_id, 'eeg')
            
            channel_names = list(info['clab'])
            channel_mask = get_eeg_channel_mask(channel_names, exclude_eog=self.exclude_eog)
            
            raw_filtered_cnt_list = []
            processed_cnt_list = []
            session_stats = []
            fs = float(info['fs'])
            for cnt in cnt_list:
                filtered_cnt = cnt[:, channel_mask]
                raw_filtered_cnt_list.append(filtered_cnt.astype(np.float32, copy=False))

                processed_cnt = apply_temporal_filter(
                    filtered_cnt,
                    sample_rate=fs,
                    modality='eeg',
                    preprocessing=self.eeg_preprocessing,
                )
                processed_cnt_list.append(processed_cnt)

                channel_mean = processed_cnt.mean(axis=0)
                channel_std = processed_cnt.std(axis=0) + 1e-8
                session_stats.append((channel_mean, channel_std))
            
            filtered_info = info.copy()
            filtered_channel_names = [name for i, name in enumerate(channel_names) if channel_mask[i]]
            filtered_info['clab'] = filtered_channel_names
            filtered_info['original_clab'] = channel_names
            filtered_info['channel_mask'] = channel_mask
            
            self._eeg_raw_cache[subject_id] = (raw_filtered_cnt_list, mrk_list, filtered_info)
            self._eeg_processed_cache[subject_id] = (processed_cnt_list, mrk_list, filtered_info)
            self._eeg_session_stats_cache[subject_id] = session_stats

    def _get_eeg_data(
        self,
        subject_id: int,
        processed: bool = True,
    ) -> Tuple[List[np.ndarray], List[dict], dict]:
        """Get cached EEG data after channel filtering and optional temporal preprocessing."""
        self._cache_eeg_data(subject_id)
        if processed:
            return self._eeg_processed_cache[subject_id]
        return self._eeg_raw_cache[subject_id]
    
    def _cache_nirs_data(self, subject_id: int):
        if subject_id not in self._nirs_processed_cache:
            cnt_list, mrk_list, info = self.nirs_loader.load_subject_data(subject_id, 'fnirs')
            
            channel_names = list(info['clab'])
            channel_mask = get_fnirs_channel_mask(
                channel_names, 
                hbo_only=self.hbo_only, 
                hbr_only=self.hbr_only
            )
            
            raw_filtered_cnt_list = []
            processed_cnt_list = []
            session_stats = []
            fs = float(info['fs'])
            for cnt in cnt_list:
                filtered_cnt = cnt[:, channel_mask]
                raw_filtered_cnt_list.append(filtered_cnt.astype(np.float32, copy=False))

                processed_cnt = apply_temporal_filter(
                    filtered_cnt,
                    sample_rate=fs,
                    modality='fnirs',
                    preprocessing=self.fnirs_preprocessing,
                )
                processed_cnt_list.append(processed_cnt)

                channel_mean = processed_cnt.mean(axis=0)
                channel_std = processed_cnt.std(axis=0) + 1e-8
                session_stats.append((channel_mean, channel_std))
            
            filtered_info = info.copy()
            filtered_channel_names = [name for i, name in enumerate(channel_names) if channel_mask[i]]
            filtered_info['clab'] = filtered_channel_names
            filtered_info['original_clab'] = channel_names
            filtered_info['channel_mask'] = channel_mask
            
            self._nirs_raw_cache[subject_id] = (raw_filtered_cnt_list, mrk_list, filtered_info)
            self._nirs_processed_cache[subject_id] = (processed_cnt_list, mrk_list, filtered_info)
            self._nirs_session_stats_cache[subject_id] = session_stats

    def _get_nirs_data(
        self,
        subject_id: int,
        processed: bool = True,
    ) -> Tuple[List[np.ndarray], List[dict], dict]:
        """Get cached NIRS data after channel filtering and optional temporal preprocessing."""
        self._cache_nirs_data(subject_id)
        if processed:
            return self._nirs_processed_cache[subject_id]
        return self._nirs_raw_cache[subject_id]

    def _get_eeg_session_statistics(self, subject_id: int, session_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._cache_eeg_data(subject_id)
        return self._eeg_session_stats_cache[subject_id][session_idx]

    def _get_nirs_session_statistics(self, subject_id: int, session_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._cache_nirs_data(subject_id)
        return self._nirs_session_stats_cache[subject_id][session_idx]
    
    def _extract_window(
        self, 
        cnt: np.ndarray, 
        start: int, 
        end: int, 
        target_samples: int
    ) -> np.ndarray:
        """Extract and pad a window from continuous data."""
        # Handle boundary cases
        if start < 0:
            start = 0
            end = target_samples
        if end > cnt.shape[0]:
            end = cnt.shape[0]
            start = max(0, end - target_samples)
            
        window = cnt[start:end, :]  # shape: (window_samples, n_channels)
        
        # Pad if necessary
        if window.shape[0] < target_samples:
            pad_size = target_samples - window.shape[0]
            window = np.pad(window, ((0, pad_size), (0, 0)), mode='constant')
        
        # Transpose to (n_channels, window_samples)
        window = window.T
        
        return window
    
    def __len__(self) -> int:
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trial with both modalities.
        
        Returns:
            Dict with:
                - 'eeg': tensor of shape (n_eeg_channels, eeg_window_samples)
                - 'fnirs': tensor of shape (n_nirs_channels, nirs_window_samples)
                - 'label': tensor of shape () with class index
                - 'subject_id': int
                - 'session_idx': int
                - 'trial_idx': int
        """
        trial = self.trials[idx]
        
        # Get EEG data
        eeg_cnt_list, _, eeg_info = self._get_eeg_data(trial.subject_id, processed=True)
        eeg_cnt = eeg_cnt_list[trial.session_idx]
        eeg_window_samples = int(self.window_duration_s * eeg_info['fs'])
        eeg_window = self._extract_window(
            eeg_cnt, trial.eeg_start_sample, trial.eeg_end_sample, eeg_window_samples
        )
        eeg_norm_mode = self.normalization_mode if self.normalize else 'none'
        if eeg_norm_mode == 'session':
            eeg_mean, eeg_std = self._get_eeg_session_statistics(trial.subject_id, trial.session_idx)
            eeg_window = normalize_window(eeg_window, 'session', eeg_mean, eeg_std)
        elif eeg_norm_mode == 'window':
            eeg_window = normalize_window(eeg_window, 'window')
        
        # Get NIRS data
        nirs_cnt_list, _, nirs_info = self._get_nirs_data(trial.subject_id, processed=True)
        nirs_cnt = nirs_cnt_list[trial.session_idx]
        nirs_window_samples = int(self.window_duration_s * nirs_info['fs'])
        nirs_window = self._extract_window(
            nirs_cnt, trial.nirs_start_sample, trial.nirs_end_sample, nirs_window_samples
        )
        nirs_norm_mode = self.normalization_mode if self.normalize else 'none'
        if uses_record_level_fnirs_standardization(self.fnirs_preprocessing):
            nirs_norm_mode = 'none'
        if nirs_norm_mode == 'session':
            nirs_mean, nirs_std = self._get_nirs_session_statistics(trial.subject_id, trial.session_idx)
            nirs_window = normalize_window(nirs_window, 'session', nirs_mean, nirs_std)
        elif nirs_norm_mode == 'window':
            nirs_window = normalize_window(nirs_window, 'window')
        
        return {
            'eeg': torch.from_numpy(eeg_window).float(),
            'fnirs': torch.from_numpy(nirs_window).float(),
            'label': torch.tensor(trial.label, dtype=torch.long),
            'subject_id': trial.subject_id,
            'session_idx': trial.session_idx,
            'trial_idx': trial.trial_idx,
        }
    
    def get_eeg_channel_names(self) -> List[str]:
        """Get the EEG channel names (after filtering)."""
        if len(self.trials) == 0:
            return []
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_eeg_data(first_subject, processed=True)
        return list(info['clab'])
    
    def get_fnirs_channel_names(self) -> List[str]:
        """Get the fNIRS channel names (after filtering)."""
        if len(self.trials) == 0:
            return []
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_nirs_data(first_subject, processed=True)
        return list(info['clab'])
    
    def get_num_eeg_channels(self) -> int:
        """Get the number of EEG channels (after filtering)."""
        return len(self.get_eeg_channel_names())
    
    def get_num_fnirs_channels(self) -> int:
        """Get the number of fNIRS channels (after filtering)."""
        return len(self.get_fnirs_channel_names())
    
    def get_eeg_sample_rate(self) -> float:
        """Get the EEG sampling rate."""
        if len(self.trials) == 0:
            return 0.0
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_eeg_data(first_subject, processed=True)
        return float(info['fs'])
    
    def get_fnirs_sample_rate(self) -> float:
        """Get the fNIRS sampling rate."""
        if len(self.trials) == 0:
            return 0.0
        first_subject = self.trials[0].subject_id
        _, _, info = self._get_nirs_data(first_subject, processed=True)
        return float(info['fs'])


def create_dataloaders(
    data_root: str,
    modality: Literal['eeg', 'fnirs', 'both'] = 'eeg',
    task: Literal['motor_imagery', 'mental_arithmetic'] = 'motor_imagery',
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    test_subjects: Optional[List[int]] = None,
    window_samples: int = 512,
    window_duration_s: float = 2.5,
    batch_size: int = 32,
    num_workers: int = 0,
    dataloader_cfg: Optional[Dict[str, Union[int, bool]]] = None,
    exclude_eog: bool = True,
    hbo_only: bool = True,
    hbr_only: bool = False,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders with subject-based splits.
    
    Args:
        data_root: Path to 'EEG+NIRS Single-Trial' directory
        modality: 'eeg', 'fnirs', or 'both' for multimodal
        task: 'motor_imagery' or 'mental_arithmetic'
        train_subjects: Subject IDs for training (default: 1-20)
        val_subjects: Subject IDs for validation (default: 21-25)
        test_subjects: Subject IDs for testing (default: 26-29)
        window_samples: Window size for single-modality (ignored if modality='both')
        window_duration_s: Window duration for multimodal
        batch_size: Batch size
        num_workers: DataLoader workers
        dataloader_cfg: Optional DataLoader keyword overrides
        exclude_eog: If True, exclude EOG channels from EEG data (default: True)
        hbo_only: Legacy selector for the `highWL` / `_O` fNIRS branch (default: True)
        hbr_only: Legacy selector for the `lowWL` / `_R` fNIRS branch (default: False)
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    # Default subject splits
    if train_subjects is None:
        train_subjects = list(range(1, 21))  # 1-20
    if val_subjects is None:
        val_subjects = list(range(21, 26))  # 21-25
    if test_subjects is None:
        test_subjects = list(range(26, 30))  # 26-29

    dataloader_cfg = dict(dataloader_cfg or {})
    
    dataloaders = {}
    
    for split, subject_ids in [
        ('train', train_subjects),
        ('val', val_subjects),
        ('test', test_subjects)
    ]:
        if modality == 'both':
            dataset = MultiModalEEGfNIRSDataset(
                data_root=data_root,
                subject_ids=subject_ids,
                task=task,
                window_duration_s=window_duration_s,
                exclude_eog=exclude_eog,
                hbo_only=hbo_only,
                hbr_only=hbr_only,
                **kwargs
            )
        else:
            dataset = EEGfNIRSDataset(
                data_root=data_root,
                subject_ids=subject_ids,
                task=task,
                modality=modality,
                window_samples=window_samples,
                exclude_eog=exclude_eog,
                hbo_only=hbo_only,
                hbr_only=hbr_only,
                **kwargs
            )
        
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': bool(dataloader_cfg.get('pin_memory', True)),
        }
        if num_workers > 0 and bool(dataloader_cfg.get('persistent_workers', False)):
            loader_kwargs['persistent_workers'] = True
        if num_workers > 0 and dataloader_cfg.get('prefetch_factor') is not None:
            loader_kwargs['prefetch_factor'] = int(dataloader_cfg['prefetch_factor'])

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            **loader_kwargs,
        )
    
    return dataloaders


if __name__ == '__main__':
    # Test the data loader
    import sys
    
    data_root = 'data/EEG+NIRS Single-Trial'
    
    print("=" * 60)
    print("Testing BBCIDataLoader")
    print("=" * 60)
    
    loader = BBCIDataLoader(data_root, subject_ids=[1, 2])
    
    # Check synchronization
    print("\nChecking synchronization for Subject 1...")
    sync_info = loader.check_synchronization(1, 0)
    print(f"  Offset: {sync_info.offset_ms:.1f} ms")
    print(f"  Interval diff: mean={sync_info.interval_diff_mean_ms:.1f}, std={sync_info.interval_diff_std_ms:.1f}, max={sync_info.interval_diff_max_ms:.1f} ms")
    print(f"  Labels match: {sync_info.labels_match}")
    
    # Show raw channel info
    print("\n" + "=" * 60)
    print("Testing Channel Filtering")
    print("=" * 60)
    
    cnt_list, _, info = loader.load_subject_data(1, 'eeg')
    eeg_channels = list(info['clab'])
    print(f"\nOriginal EEG channels ({len(eeg_channels)}): {eeg_channels}")
    eeg_mask = get_eeg_channel_mask(eeg_channels, exclude_eog=True)
    eeg_filtered = [ch for i, ch in enumerate(eeg_channels) if eeg_mask[i]]
    print(f"Filtered EEG channels ({len(eeg_filtered)}, exclude_eog=True): {eeg_filtered}")
    
    cnt_list, _, info = loader.load_subject_data(1, 'fnirs')
    fnirs_channels = list(info['clab'])
    print(f"\nOriginal fNIRS channels ({len(fnirs_channels)}): {fnirs_channels[:10]}... (showing first 10)")
    fnirs_mask = get_fnirs_channel_mask(fnirs_channels, hbo_only=True)
    fnirs_filtered = [ch for i, ch in enumerate(fnirs_channels) if fnirs_mask[i]]
    print(f"Filtered fNIRS channels ({len(fnirs_filtered)}, hbo_only=True): {fnirs_filtered[:10]}... (showing first 10)")
    
    print("\n" + "=" * 60)
    print("Testing EEGfNIRSDataset (single modality) with filtering")
    print("=" * 60)
    
    # Test EEG with EOG exclusion
    dataset = EEGfNIRSDataset(
        data_root=data_root,
        subject_ids=[1, 2],
        task='motor_imagery',
        modality='eeg',
        window_samples=512,
        exclude_eog=True,  # Default: exclude EOG
    )
    
    print(f"\nEEG Dataset (exclude_eog=True):")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Sample rate: {dataset.get_sample_rate()} Hz")
    print(f"  Channels ({dataset.get_num_channels()}): {dataset.get_channel_names()}")
    
    sample = dataset[0]
    print(f"  Sample data shape: {sample['data'].shape}")
    print(f"  Sample label: {sample['label']}")
    
    # Test fNIRS with HbO only
    dataset_fnirs = EEGfNIRSDataset(
        data_root=data_root,
        subject_ids=[1, 2],
        task='motor_imagery',
        modality='fnirs',
        window_samples=25,  # ~2.5s @ 10Hz
        hbo_only=True,  # Default: HbO only
    )
    
    print(f"\nfNIRS Dataset (hbo_only=True):")
    print(f"  Dataset size: {len(dataset_fnirs)}")
    print(f"  Sample rate: {dataset_fnirs.get_sample_rate()} Hz")
    print(f"  Channels ({dataset_fnirs.get_num_channels()}): {dataset_fnirs.get_channel_names()[:5]}...")
    
    sample_fnirs = dataset_fnirs[0]
    print(f"  Sample data shape: {sample_fnirs['data'].shape}")
    
    print("\n" + "=" * 60)
    print("Testing MultiModalEEGfNIRSDataset with filtering")
    print("=" * 60)
    
    mm_dataset = MultiModalEEGfNIRSDataset(
        data_root=data_root,
        subject_ids=[1, 2],
        task='motor_imagery',
        window_duration_s=2.5,
        exclude_eog=True,
        hbo_only=True,
    )
    
    print(f"Dataset size: {len(mm_dataset)}")
    print(f"EEG channels ({mm_dataset.get_num_eeg_channels()}): {mm_dataset.get_eeg_channel_names()}")
    print(f"fNIRS channels ({mm_dataset.get_num_fnirs_channels()}): {mm_dataset.get_fnirs_channel_names()[:5]}...")
    
    mm_sample = mm_dataset[0]
    print(f"EEG shape: {mm_sample['eeg'].shape}")
    print(f"fNIRS shape: {mm_sample['fnirs'].shape}")
    print(f"Label: {mm_sample['label']}")
    
    print("\n" + "=" * 60)
    print("Testing create_dataloaders")
    print("=" * 60)
    
    dataloaders = create_dataloaders(
        data_root=data_root,
        modality='eeg',
        task='motor_imagery',
        train_subjects=[1, 2],
        val_subjects=[3],
        test_subjects=[4],
        window_samples=512,
        batch_size=8,
        exclude_eog=True,
    )
    
    for split, dl in dataloaders.items():
        print(f"{split}: {len(dl.dataset)} samples, {len(dl)} batches")
    
    # Test a batch
    batch = next(iter(dataloaders['train']))
    print(f"Batch data shape: {batch['data'].shape}")
    print(f"Batch labels: {batch['label']}")
    
    print("\n✓ All tests passed!")
