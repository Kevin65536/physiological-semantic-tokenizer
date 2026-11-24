"""Deterministic integer search through a frozen linear spatial EEG decoder."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.fft import rfft

from .spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary, band_filter, load_model, spectral_features


class AnalysisSynthesisSpatialEncoder(SpatialWaveformDictionary):
    """Search local atom alternatives against observation-only decoder targets."""

    def __init__(self, source, settings, variances, *, component_source=None):
        if source.readout or source.metadata.get('output_projection_hz') != [1, 45]:
            raise ValueError('frozen separable spatial decoder with 1–45 Hz projection required')
        metadata = deepcopy(source.metadata)
        metadata['encoder_rule'] = 'observation_only_analysis_by_synthesis_v1'
        metadata['search_settings'] = settings
        super().__init__(metadata, source.transfer, source.centers, source.atoms, source.weights, feature_weights=source.feature_weights)
        self.component_source = component_source
        self.settings = settings
        self.variances = variances
        self.atom_spectra = {}
        self.atom_envelopes = {}
        self.frequency_indices = {}
        self.phases = {}
        self.search_history = []
        frequency = np.fft.rfftfreq(6000, 1 / 200)
        for band, (lower, upper) in enumerate(BANDS):
            indices = np.flatnonzero((frequency >= lower) & (frequency < upper))
            self.frequency_indices[band] = indices
            self.phases[band] = np.exp(-2j * np.pi * np.arange(15)[:, None] * 400 * indices[None] / 6000)

    def _book(self, branch, band):
        key = branch, band
        if key not in self.atom_spectra:
            atoms = np.asarray(self.atoms[key], dtype=np.float64)
            spectrum = rfft(atoms, n=6000, axis=1, norm='ortho', workers=self.settings['fft_workers'])
            self.atom_spectra[key] = spectrum[:, self.frequency_indices[band]].copy()
            filtered = band_filter(atoms, band)
            self.atom_envelopes[key] = np.sqrt(np.mean(filtered.reshape(len(atoms), 8, 50) ** 2, axis=-1))
        return self.atom_spectra[key], self.atom_envelopes[key]

    def _candidates(self, prediction, branch, band):
        features = self.assignment_features(prediction, branch, band)
        centers = self.centers[branch, band]
        distances = np.maximum(0., np.square(features).sum(1)[:, None] + np.square(centers).sum(1)[None]
                               - 2 * features @ centers.T)
        count = min(self.settings['top_m'], len(centers))
        candidates = np.argpartition(distances, count - 1, axis=1)[:, :count]
        chosen_distance = np.take_along_axis(distances, candidates, axis=1)
        order = np.argsort(chosen_distance, axis=1, kind='stable')
        candidates = np.take_along_axis(candidates, order, axis=1)
        chosen_distance = np.take_along_axis(chosen_distance, order, axis=1)
        _, envelopes = self._book(branch, band)
        predicted_envelope = np.sqrt(np.mean(band_filter(prediction, band).reshape(len(prediction), 8, 50) ** 2, axis=-1))
        if self.feature_weights:
            # Gain guards are in physical carrier coordinates; the shortlist uses joint features.
            physical_centers = spectral_features(self.atoms[branch, band][1:], band)
            energy = np.square(spectral_features(prediction, band)).sum(1)
            center_energy = np.square(physical_centers[candidates]).sum(-1)
            radius_scale = np.square(features[:, :-3]).sum(1)
        else:
            energy = np.square(features).sum(1)
            center_energy = np.square(centers[candidates]).sum(-1)
            radius_scale = energy
        ratio = center_energy / np.maximum(energy[:, None], 1e-12)
        low, high = self.settings['energy_ratio']
        envelope_error = np.mean((envelopes[candidates + 1] - predicted_envelope[:, None]) ** 2, axis=-1)
        envelope_error /= np.maximum(np.mean(predicted_envelope ** 2, axis=1)[:, None], 1e-12)
        radius = np.maximum(self.settings['distance_radius_factor'] * chosen_distance[:, :1],
                            self.settings['relative_band_distance_floor'] * radius_scale[:, None])
        admissible = ((ratio >= low) & (ratio <= high)
                      & (envelope_error <= self.settings['envelope_nrmse_squared_max'])
                      & (chosen_distance <= radius + 1e-10))
        return candidates + 1, chosen_distance, admissible

    def _weights(self, dataset, observed):
        result = {}
        for branch in ('S', 'O', 'sum'):
            values = np.asarray(self.variances[dataset][branch], dtype=np.float64)
            if values.shape != observed.shape or np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError('fixed positive training variances required per native electrode')
            result[branch] = observed / values
        result['sum'] *= self.settings['sum_weight']
        return result

    @staticmethod
    def _wave_loss(result, target, weights):
        return sum(float(np.sum(np.square(result[branch] - target[branch]) * weights[branch][None, :, None, None]))
                   for branch in ('S', 'O', 'sum'))

    def encode_positions(self, values, dataset, native_indices, observed=None):
        if self.component_source is not None:
            raise ValueError('a calibrated montage search requires full-array encode')
        return super().encode_positions(values, dataset, native_indices, observed)

    def encode(self, values, dataset, observed):
        values = np.asarray(values, dtype=np.float64)
        observed = np.asarray(observed)
        if len(values) != 1:
            raise ValueError('stateful coordinate search uses one independent 30-second array at a time')
        started = time.monotonic()
        if self.component_source is None:
            ids = super().encode(values, dataset, observed)
            target = self.predict_continuous(values)
        else:
            native = self.metadata['positions'][dataset]
            if (self.component_source.metadata['positions'].get(dataset) != native
                    or values.shape != (1, len(native), 15, 400)
                    or observed.dtype != bool or observed.shape != values.shape[:2]):
                raise ValueError('calibrated source requires the same complete native layout')
            target = self.component_source.predict_continuous(values, dataset, observed)
            indices = np.arange(len(native))[None, :]
            ids = self._assign_positions(target, dataset, indices, observed)
        original_ids = {branch: value.copy() for branch, value in ids.items()}
        target['sum'] = values
        weights = self._weights(dataset, observed[0])
        initial = self.decode_ids(ids, dataset)
        initial_loss = self._wave_loss(initial, target, weights)
        spectra = {branch: np.fft.rfft((initial[branch] - target[branch]).reshape(len(observed[0]), 6000),
                                      axis=1, norm='ortho').T for branch in ('S', 'O', 'sum')}
        roles = self.roles(dataset)
        updates = 0
        proposed = 0
        admissible_count = 0
        proxy_changes = []
        local_drift = 0.
        for branch in self.settings['branches']:
            for band in range(len(BANDS)):
                positions = np.flatnonzero((roles == band) & observed[0])
                if not len(positions):
                    continue
                all_positions = np.flatnonzero(roles == band)
                selected_prediction = target[branch][0, positions].reshape(-1, 400)
                candidates, distances, admissible = self._candidates(selected_prediction, branch, band)
                candidates = candidates.reshape(len(positions), 15, -1)
                distances = distances.reshape(candidates.shape)
                admissible = admissible.reshape(candidates.shape)
                proposed += admissible.size
                admissible_count += int(admissible.sum())
                atom_spectra, _ = self._book(branch, band)
                frequency = self.frequency_indices[band]
                branch_residual = spectra[branch][frequency].copy()
                sum_residual = spectra['sum'][frequency].copy()
                matrix = self.weights[dataset, branch, band]
                for sweep in range(self.settings['sweeps']):
                    for local_position, position in enumerate(positions):
                        row = int(np.flatnonzero(all_positions == position)[0])
                        influence = np.asarray(matrix[row], dtype=np.float64)
                        weighted_branch = weights[branch] * influence
                        weighted_sum = weights['sum'] * influence
                        curvature = float(np.sum((weights[branch] + weights['sum']) * influence ** 2))
                        for patch in range(15):
                            current = int(ids[branch][0, position, patch])
                            options = candidates[local_position, patch]
                            keep = admissible[local_position, patch] & (options != current)
                            if not keep.any():
                                continue
                            options = options[keep]
                            delta = (atom_spectra[options] - atom_spectra[current]) * self.phases[band][patch]
                            gradient = branch_residual @ weighted_branch + sum_residual @ weighted_sum
                            proxy_delta = 4 * np.real(delta @ gradient.conj()) + 2 * curvature * np.sum(abs(delta) ** 2, axis=1)
                            current_feature = self.centers[branch, band][current - 1]
                            feature = self.assignment_features(target[branch][0, position, patch][None], branch, band)[0]
                            current_distance = float(np.sum((current_feature - feature) ** 2))
                            option_distance = np.sum((self.centers[branch, band][options - 1] - feature) ** 2, axis=1)
                            semantic_delta = (self.settings['semantic_weight'] * weights[branch][position]
                                              * (option_distance - current_distance))
                            total_delta = proxy_delta + semantic_delta
                            eligible = (proxy_delta < -1e-10) & (total_delta < -1e-10)
                            if not eligible.any():
                                continue
                            choice = int(np.argmin(np.where(eligible, total_delta, np.inf)))
                            change = delta[choice]
                            ids[branch][0, position, patch] = int(options[choice])
                            branch_residual += change[:, None] * influence[None]
                            sum_residual += change[:, None] * influence[None]
                            local_drift += float(semantic_delta[choice])
                            proxy_changes.append(float(proxy_delta[choice]))
                            updates += 1
                spectra[branch][frequency] = branch_residual
                spectra['sum'][frequency] = sum_residual
        final = self.decode_ids(ids, dataset)
        final_loss = self._wave_loss(final, target, weights)
        predicted_loss = initial_loss + sum(proxy_changes)
        np.testing.assert_allclose(final_loss, predicted_loss, rtol=1e-9, atol=1e-7)
        if final_loss > initial_loss + 1e-7:
            raise AssertionError('actual frozen-decoder observation proxy loss increased')
        record = dict(dataset=dataset, seconds=time.monotonic() - started,
                      proxy_loss_before=initial_loss, proxy_loss_after=final_loss,
                      proxy_relative_reduction=(initial_loss - final_loss) / max(initial_loss, 1e-12),
                      accepted_updates=updates, admissible_options=admissible_count, proposed_options=proposed,
                      semantic_penalty_change=local_drift,
                      changed_ids={branch: int(np.sum(ids[branch] != original_ids[branch])) for branch in ('S', 'O')},
                      actual_proxy_recalculation='matches accumulated exact Fourier coordinate differences',
                      input=('X and observation availability only; targets are fixed train-fitted Wiener estimates'
                             if self.component_source is None else 'X and observation availability only; frozen transient-conditioned source estimates'))
        self.search_history.append(record)
        return ids


def load_encoder(bundle, *, component_source=None):
    bundle = Path(bundle)
    metadata = json.loads(bundle.read_text())
    schema = metadata['schema']
    if schema not in ('spatial_analysis_synthesis_encoder_v1', 'spatial_analysis_synthesis_transient_v1'):
        raise ValueError('versioned encoder bundle required')
    source = (bundle.parent / metadata['source']).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['source_sha256']:
        raise ValueError('frozen decoder source changed')
    spatial = load_model(source)
    if schema == 'spatial_analysis_synthesis_transient_v1':
        from .eeg_transient_gain import load_model as load_transient_source
        if component_source is not None:
            raise ValueError('calibrated bundle already specifies its own frozen component source')
        path = (bundle.parent / metadata['component_source']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['component_source_sha256']:
            raise ValueError('frozen component source changed')
        component_source = load_transient_source(path)
        for key in ('input_identity', 'scales_identity', 'positions'):
            if component_source.metadata[key] != spatial.metadata[key]:
                raise ValueError(f'component source and spatial encoder disagree on {key}')
    return AnalysisSynthesisSpatialEncoder(spatial, metadata['settings'], metadata['variances'], component_source=component_source)
