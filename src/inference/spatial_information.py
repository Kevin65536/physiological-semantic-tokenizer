"""Conditional spatial information in one frozen parameter coordinate system.

These functions assess supplied local derivatives. Training SD is a coordinate
scale, not a noise covariance estimate. They never read data or fit scales.
"""
from __future__ import annotations

import numpy as np


def panel_mask(steps, eeg_features, hb_pairs, *, observed_eeg, observed_hb,
               hidden_pair=0):
    """Select channel prefixes on the full panel, keeping one pair hidden."""
    if (steps < 1 or eeg_features < 1 or hb_pairs < 1
            or not 0 <= observed_eeg <= eeg_features
            or not 1 <= observed_hb <= hb_pairs
            or not 0 <= hidden_pair < hb_pairs):
        raise ValueError('invalid panel dimensions')
    mask = np.zeros((steps, eeg_features+2*hb_pairs), bool)
    mask[:, :observed_eeg] = True
    mask[:, eeg_features:eeg_features+2*observed_hb] = True
    mask[:, eeg_features+2*hidden_pair:eeg_features+2*hidden_pair+2] = False
    return mask


def legacy_objective_weights(mask, eeg_features, *, eeg_weight=1.):
    """Existing T/n_visible weights, explicitly separate from added rows."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2 or not 0 < eeg_features < mask.shape[1]:
        raise ValueError('boolean full-panel mask and valid EEG count required')
    if not np.isfinite(eeg_weight) or eeg_weight < 0:
        raise ValueError('eeg_weight must be finite nonnegative')
    t = len(mask)
    ne, nh = mask[:, :eeg_features].sum(), mask[:, eeg_features:].sum()
    return np.r_[np.full(eeg_features, np.sqrt(eeg_weight*t/ne) if ne else 0.),
                 np.full(mask.shape[1]-eeg_features, np.sqrt(t/nh) if nh else 0.)]


def weak_direction_lift(old_jacobian, full_jacobian, *, directions=5, rcond=1e-8):
    """Project added rows on OLD weak directions; do not rotate to new vectors.

    Caller supplies the old observations and the complete nested observation
    set in the same unknown coordinates and row scaling. The difference of
    Grams must be positive semidefinite, otherwise this is not an added-row
    experiment. Small roundoff negatives are reported and tolerated.
    """
    old, full = np.asarray(old_jacobian, float), np.asarray(full_jacobian, float)
    if (old.ndim != 2 or full.ndim != 2 or old.shape[1] != full.shape[1]
            or not old.shape[1] or not np.isfinite(old).all() or not np.isfinite(full).all()
            or not 1 <= directions <= old.shape[1] or not 0 < rcond < 1):
        raise ValueError('finite common-coordinate Jacobians and direction count required')
    _, singular, right = np.linalg.svd(old, full_matrices=False)
    if len(singular) < old.shape[1]:
        _, singular_full, right_full = np.linalg.svd(old, full_matrices=True)
        singular = np.r_[singular_full, np.zeros(old.shape[1]-len(singular_full))]
        right = right_full
    delta = full.T@full-old.T@old
    minimum = float(np.linalg.eigvalsh(delta)[0])
    tolerance = 1e-9*max(float(np.linalg.norm(full.T@full, 2)), 1.)
    if minimum < -tolerance:
        raise ValueError('full Jacobian must add rows without changing old row scaling')
    vectors = right[-directions:].T
    old_energy = np.sum((old@vectors)**2, axis=0)
    new_energy = np.sum((full@vectors)**2, axis=0)
    added = new_energy-old_energy
    threshold = (singular[0]*rcond)**2 if len(singular) else 0.
    return dict(old_directions=vectors, old_singular_values=singular[-directions:],
                old_direction_energy=old_energy, full_direction_energy=new_energy,
                added_direction_energy=added,
                energy_gain_ratio=np.divide(new_energy, old_energy,
                    out=np.full_like(new_energy, np.nan), where=old_energy > threshold),
                newly_supported_old_directions=int(np.sum((old_energy <= threshold) & (new_energy > threshold))),
                minimum_added_gram_eigenvalue=minimum, roundoff_tolerance=tolerance)


def generic_equal_rank_readout(eeg_loadings, hb_loadings, *, seed=0):
    """Arbitrary sensor assignment with exact loading singular spectra.

    EEG left-orthogonal mixing and Hb row permutation retain each matrix's
    singular values, rank, Frobenius norm and parameter count. Hb convexity is
    preserved. Signals and scoring targets must remain unchanged: this is an
    operator mismatch control, not a coordinate transform of the data. All Hb
    rows, including the hidden target's operator, may be reassigned.
    """
    eeg, hb = np.asarray(eeg_loadings, float), np.asarray(hb_loadings, float)
    if (eeg.ndim != 2 or hb.ndim != 2 or eeg.shape[1] != hb.shape[1]
            or not np.isfinite(eeg).all() or not np.isfinite(hb).all()
            or np.any(hb < 0) or not np.allclose(hb.sum(axis=1), 1.)
            or np.linalg.matrix_rank(eeg) != eeg.shape[1]
            or np.linalg.matrix_rank(hb) != hb.shape[1]):
        raise ValueError('finite full-rank readouts with convex Hb rows required')
    rng = np.random.default_rng(seed)
    # Householder is dense orthogonal without a costly F-by-F QR.
    normal = rng.normal(size=len(eeg)); normal /= np.linalg.norm(normal)
    mixed = eeg-2*normal[:, None]*(normal@eeg)[None, :]
    permutation = rng.permutation(len(hb))
    if np.array_equal(permutation, np.arange(len(hb))) and len(hb) > 1:
        permutation = np.roll(permutation, 1)
    shuffled = hb[permutation]
    return dict(eeg_loadings=mixed, hb_loadings=shuffled,
                eeg_householder_normal=normal, hb_permutation=permutation,
                eeg_singular_values=np.linalg.svd(eeg, compute_uv=False),
                hb_singular_values=np.linalg.svd(hb, compute_uv=False),
                interpretation='unchanged_signal_operator_mismatch_equal_loading_spectra')
