"""
Croce et al. 2017 — Bayesian Sequential Monte Carlo for joint EEG-fNIRS.

Implements the state-space model:
    s_k = A · s_{k-1} + w_k          w_k ~ N(0, Q)
    y^E_k = H^E · s_k + v^E_k       v^E_k ~ N(0, R^E)
    y^F_k = H^F · (HRF * s)_k + v^F_k  v^F_k ~ N(0, R^F)

with a bootstrap particle filter that jointly estimates the latent neural state
from simultaneous EEG and fNIRS observations.

Reference:
    Croce, P., Zappasodi, F., Marzetti, L., Merla, A., & Chiarelli, A. M. (2017).
    Exploiting neurovascular coupling: a Bayesian sequential Monte Carlo approach
    applied to simulated EEG-fNIRS data.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.linalg import solve_discrete_lyapunov


def _systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Systematic resampling — lower variance than multinomial."""
    N = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # avoid floating point drift
    generator = rng if rng is not None else np.random
    u0 = generator.uniform(0, 1.0 / N)
    positions = u0 + np.arange(N) / N
    indices = np.searchsorted(cumsum, positions)
    return indices


def _effective_sample_size(weights: np.ndarray) -> float:
    """Normalized ESS ∈ [1/N, N]."""
    w = np.asarray(weights, dtype=np.float64)
    if np.sum(w) <= 0:
        return 0.0
    w_norm = w / np.sum(w)
    return 1.0 / max(float(np.sum(w_norm ** 2)), 1e-16)


def double_gamma_hrf(
    fs_hz: float,
    duration_s: float = 24.0,
    peak_delay_s: float = 6.0,
    peak_dispersion_s: float = 1.0,
    undershoot_delay_s: float = 16.0,
    undershoot_dispersion_s: float = 1.5,
    peak_scale: float = 1.0,
    undershoot_scale: float = 0.25,
) -> np.ndarray:
    """Build a double-gamma hemodynamic response kernel."""
    steps = max(int(round(duration_s * fs_hz)), 1)
    t = np.arange(steps, dtype=np.float64) / fs_hz

    def _gamma_pdf(time_axis, delay, dispersion):
        delay = max(float(delay), 1e-3)
        dispersion = max(float(dispersion), 1e-3)
        concentration = max(delay / dispersion, 1e-3) + 1.0
        safe = np.maximum(time_axis, 1e-6)
        return np.exp(
            (concentration - 1.0) * np.log(safe)
            - safe / dispersion
            - math.lgamma(concentration)
            - concentration * np.log(dispersion)
        )

    peak = _gamma_pdf(t, peak_delay_s, peak_dispersion_s)
    undershoot = _gamma_pdf(t, undershoot_delay_s, undershoot_dispersion_s)
    kernel = peak_scale * peak - undershoot_scale * undershoot
    denom = np.sum(np.abs(kernel))
    return (kernel / max(denom, 1e-8)).astype(np.float32)


@dataclass
class SMCFilterResult:
    """Output of one SMC filtering pass."""
    state_mean: np.ndarray      # [T, K] filtered posterior mean
    state_std: np.ndarray       # [T, K] filtered posterior std
    state_particles: np.ndarray # [T, N, K] all particle trajectories
    weights_history: np.ndarray # [T, N] weights at each step
    ess_history: np.ndarray     # [T] effective sample size
    log_likelihood: float       # total log-likelihood

    # Reconstructed clean observations
    eeg_reconstructed: np.ndarray    # [T, E]
    fnirs_reconstructed: np.ndarray  # [T, F]

    # Single-modality baselines (for comparison)
    eeg_only_state_mean: Optional[np.ndarray] = None   # [T, K]
    fnirs_only_state_mean: Optional[np.ndarray] = None # [T, K]


class NeurovascularSMCFilter:
    """Bootstrap particle filter for the Croce et al. neurovascular coupling model.

    The model links EEG and fNIRS through a shared latent neural state s(t):
      - EEG observes s(t) instantaneously (plus noise)
      - fNIRS observes HRF * s(t) (convolved hemodynamic response, plus noise)

    The filter estimates the posterior p(s_k | y^E_{1:k}, y^F_{1:k}) sequentially.
    """

    def __init__(
        self,
        hrf_kernel: np.ndarray,
        state_transition_matrix: np.ndarray,
        process_noise_cov: np.ndarray,
        eeg_forward: np.ndarray,
        fnirs_forward: np.ndarray,
        eeg_noise_cov: np.ndarray,
        fnirs_noise_cov: np.ndarray,
        n_particles: int = 500,
        resample_threshold: float = 0.5,
        seed: int = 42,
    ):
        """
        Args:
            hrf_kernel: HRF impulse response [L] at observation rate
            state_transition_matrix: A [K, K] for AR(1) state dynamics
            process_noise_cov: Q [K, K] process noise covariance
            eeg_forward: H^E [E, K] EEG forward/observation matrix
            fnirs_forward: H^F [F, K] fNIRS forward/observation matrix
            eeg_noise_cov: R^E [E, E] EEG observation noise covariance
            fnirs_noise_cov: R^F [F, F] fNIRS observation noise covariance
            n_particles: number of particles
            resample_threshold: resample when ESS/N < this fraction
        """
        self.hrf_kernel = np.asarray(hrf_kernel, dtype=np.float32)
        self.hrf_len = len(self.hrf_kernel)

        self.A = np.asarray(state_transition_matrix, dtype=np.float64)
        self.Q = np.asarray(process_noise_cov, dtype=np.float64)
        self.Q_chol = np.linalg.cholesky(self.Q)

        self.H_eeg = np.asarray(eeg_forward, dtype=np.float64)
        self.H_fnirs = np.asarray(fnirs_forward, dtype=np.float64)

        self.R_eeg = np.asarray(eeg_noise_cov, dtype=np.float64)
        self.R_fnirs = np.asarray(fnirs_noise_cov, dtype=np.float64)

        # Precompute precision matrices for Gaussian likelihood
        self.R_eeg_inv = np.linalg.inv(self.R_eeg)
        self.R_fnirs_inv = np.linalg.inv(self.R_fnirs)
        _, self.R_eeg_logdet = np.linalg.slogdet(self.R_eeg)
        _, self.R_fnirs_logdet = np.linalg.slogdet(self.R_fnirs)

        self.E_dim = self.H_eeg.shape[0]
        self.F_dim = self.H_fnirs.shape[0]
        self.K_dim = self.A.shape[0]

        self.N = n_particles
        self.resample_threshold = resample_threshold
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gaussian_log_likelihood(
        self, y: np.ndarray, H: np.ndarray, s: np.ndarray,
        R_inv: np.ndarray, R_logdet: float,
    ) -> float:
        """Compute log p(y | s) = log N(y; H·s, R) for a single particle."""
        residual = y - H @ s
        # Mahalanobis distance
        mahal = residual @ (R_inv @ residual)
        D = len(y)
        return -0.5 * (D * np.log(2 * np.pi) + R_logdet + mahal)

    def _convolve_hrf(self, state_history: np.ndarray) -> np.ndarray:
        """Convolve state history buffer [K, L] with HRF [L] -> [K]."""
        return state_history @ self.hrf_kernel

    def _predict(self, particles: np.ndarray) -> np.ndarray:
        """Sample from state transition: s_k ~ N(A·s_{k-1}, Q)."""
        N, K = particles.shape
        noise = self.rng.randn(N, K) @ self.Q_chol.T
        return particles @ self.A.T + noise

    def _resample_particles(
        self, particles: np.ndarray, weights: np.ndarray,
        history_buffers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Systematic resample with small post-resample jitter."""
        idx = _systematic_resample(particles, weights, self.rng)
        new_particles = np.ascontiguousarray(particles[idx])
        new_buffers = np.ascontiguousarray(history_buffers[idx])

        # Add tiny jitter to maintain particle diversity (Rougier & Briers 2010)
        jitter_scale = np.sqrt(np.diag(self.Q)) * 0.1 / np.sqrt(self.N)
        jitter = self.rng.randn(self.N, self.K_dim) * jitter_scale[np.newaxis, :]
        new_particles += jitter.astype(new_particles.dtype)

        return new_particles, new_buffers, np.ones(self.N) / self.N

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        eeg_observations: np.ndarray,
        fnirs_observations: np.ndarray,
        init_state_mean: Optional[np.ndarray] = None,
        init_state_cov: Optional[np.ndarray] = None,
        return_particles: bool = True,
    ) -> SMCFilterResult:
        """Run the bootstrap particle filter on a sequence.

        Args:
            eeg_observations: [T, E] EEG observations at each time step
            fnirs_observations: [T, F] fNIRS observations at each time step
            init_state_mean: [K] initial state mean (default: zeros)
            init_state_cov: [K, K] initial state covariance (default: stationary cov)
            return_particles: whether to store full particle trajectories

        Returns:
            SMCFilterResult with filtered state estimates and reconstructed signals
        """
        T = len(eeg_observations)
        assert len(fnirs_observations) == T, "EEG and fNIRS must have same time length"

        # --- initialise particles ---
        if init_state_mean is None:
            init_state_mean = np.zeros(self.K_dim, dtype=np.float64)
        if init_state_cov is None:
            # Stationary covariance: solve P = A P A^T + Q.  Returning Q
            # itself strongly under-disperses persistent states at every
            # independently filtered event boundary.
            try:
                init_state_cov = solve_discrete_lyapunov(self.A, self.Q)
            except Exception:
                init_state_cov = self.Q.copy()
            init_state_cov = (init_state_cov + init_state_cov.T) * 0.5
            init_state_cov += np.eye(self.K_dim, dtype=np.float64) * 1e-9

        init_chol = np.linalg.cholesky(init_state_cov)
        particles = (
            init_state_mean[np.newaxis, :]
            + self.rng.randn(self.N, self.K_dim) @ init_chol.T
        )

        # Each particle maintains a history buffer [K, L] for HRF convolution
        history_buffers = np.zeros((self.N, self.K_dim, self.hrf_len), dtype=np.float64)
        weights = np.ones(self.N, dtype=np.float64) / self.N

        # --- output storage ---
        state_mean = np.zeros((T, self.K_dim), dtype=np.float64)
        state_std = np.zeros((T, self.K_dim), dtype=np.float64)
        state_particles = (
            np.zeros((T, self.N, self.K_dim), dtype=np.float64)
            if return_particles else None
        )
        weights_history = np.zeros((T, self.N), dtype=np.float64)
        ess_history = np.zeros(T, dtype=np.float64)
        eeg_reconstructed = np.zeros((T, self.E_dim), dtype=np.float64)
        fnirs_reconstructed = np.zeros((T, self.F_dim), dtype=np.float64)
        total_log_lik = 0.0

        for t in range(T):
            # ---- predict ----
            particles = self._predict(particles)

            # Shift history buffers, insert current state
            if self.K_dim == 1:
                history_buffers[:, 0, 1:] = history_buffers[:, 0, :-1]
                history_buffers[:, 0, 0] = particles[:, 0]
            else:
                history_buffers[:, :, 1:] = history_buffers[:, :, :-1]
                history_buffers[:, :, 0] = particles

            # ---- update weights ----
            # Carry the previous posterior unless resampling reset it.
            log_weights = np.log(np.maximum(weights, 1e-300))

            # EEG likelihood
            eeg_obs_t = np.asarray(eeg_observations[t], dtype=np.float64)
            for i in range(self.N):
                log_weights[i] += self._gaussian_log_likelihood(
                    eeg_obs_t, self.H_eeg, particles[i],
                    self.R_eeg_inv, self.R_eeg_logdet,
                )

            # fNIRS likelihood (uses HRF-convolved state history)
            fnirs_obs_t = np.asarray(fnirs_observations[t], dtype=np.float64)
            for i in range(self.N):
                hrf_state = self._convolve_hrf(history_buffers[i])
                log_weights[i] += self._gaussian_log_likelihood(
                    fnirs_obs_t, self.H_fnirs, hrf_state,
                    self.R_fnirs_inv, self.R_fnirs_logdet,
                )

            # Numerically stable weight normalisation (log-sum-exp)
            max_lw = np.max(log_weights)
            weights = np.exp(log_weights - max_lw)
            weights /= np.sum(weights)
            total_log_lik += max_lw + np.log(np.sum(np.exp(log_weights - max_lw)))

            # ---- estimate ----
            state_mean[t] = np.average(particles, axis=0, weights=weights)
            diff = particles - state_mean[t]
            state_std[t] = np.sqrt(
                np.average(diff ** 2, axis=0, weights=weights)
            )

            # Reconstruct clean observations
            eeg_reconstructed[t] = self.H_eeg @ state_mean[t]
            hrf_mean_state = self._convolve_hrf(
                np.average(history_buffers, axis=0, weights=weights)
            )
            fnirs_reconstructed[t] = self.H_fnirs @ hrf_mean_state

            if return_particles:
                state_particles[t] = particles
            weights_history[t] = weights
            ess_history[t] = _effective_sample_size(weights)

            # ---- resample ----
            if ess_history[t] < self.N * self.resample_threshold:
                particles, history_buffers, weights = self._resample_particles(
                    particles, weights, history_buffers,
                )

        # --- separate EEG-only and fNIRS-only baseline runs ---
        eeg_only_mean = self._run_single_modality_filter(
            eeg_observations, modality='eeg', init_state_mean=init_state_mean,
            init_state_cov=init_state_cov,
        )
        fnirs_only_mean = self._run_single_modality_filter(
            fnirs_observations, modality='fnirs', init_state_mean=init_state_mean,
            init_state_cov=init_state_cov,
        )

        return SMCFilterResult(
            state_mean=state_mean,
            state_std=state_std,
            state_particles=state_particles,
            weights_history=weights_history,
            ess_history=ess_history,
            log_likelihood=total_log_lik,
            eeg_reconstructed=eeg_reconstructed,
            fnirs_reconstructed=fnirs_reconstructed,
            eeg_only_state_mean=eeg_only_mean,
            fnirs_only_state_mean=fnirs_only_mean,
        )

    def _run_single_modality_filter(
        self,
        observations: np.ndarray,
        modality: str,
        init_state_mean: np.ndarray,
        init_state_cov: np.ndarray,
    ) -> np.ndarray:
        """Run filter with only one modality's likelihood, for baseline comparison."""
        T = len(observations)
        init_chol = np.linalg.cholesky(init_state_cov)
        particles = (
            init_state_mean[np.newaxis, :]
            + self.rng.randn(self.N, self.K_dim) @ init_chol.T
        )
        history_buffers = np.zeros((self.N, self.K_dim, self.hrf_len), dtype=np.float64)
        weights = np.ones(self.N, dtype=np.float64) / self.N
        state_mean = np.zeros((T, self.K_dim), dtype=np.float64)

        for t in range(T):
            particles = self._predict(particles)
            if self.K_dim == 1:
                history_buffers[:, 0, 1:] = history_buffers[:, 0, :-1]
                history_buffers[:, 0, 0] = particles[:, 0]
            else:
                history_buffers[:, :, 1:] = history_buffers[:, :, :-1]
                history_buffers[:, :, 0] = particles

            log_weights = np.log(np.maximum(weights, 1e-300))
            obs_t = np.asarray(observations[t], dtype=np.float64)

            if modality == 'eeg':
                for i in range(self.N):
                    log_weights[i] += self._gaussian_log_likelihood(
                        obs_t, self.H_eeg, particles[i],
                        self.R_eeg_inv, self.R_eeg_logdet,
                    )
            else:
                for i in range(self.N):
                    hrf_state = self._convolve_hrf(history_buffers[i])
                    log_weights[i] += self._gaussian_log_likelihood(
                        obs_t, self.H_fnirs, hrf_state,
                        self.R_fnirs_inv, self.R_fnirs_logdet,
                    )

            max_lw = np.max(log_weights)
            weights = np.exp(log_weights - max_lw)
            weights /= np.sum(weights)

            state_mean[t] = np.average(particles, axis=0, weights=weights)
            ess = _effective_sample_size(weights)

            if ess < self.N * self.resample_threshold:
                idx = _systematic_resample(particles, weights, self.rng)
                particles = np.ascontiguousarray(particles[idx])
                history_buffers = np.ascontiguousarray(history_buffers[idx])
                jitter_scale = np.sqrt(np.diag(self.Q)) * 0.1 / np.sqrt(self.N)
                jitter = self.rng.randn(self.N, self.K_dim) * jitter_scale
                particles += jitter.astype(particles.dtype)
                weights = np.ones(self.N, dtype=np.float64) / self.N

        return state_mean


# ------------------------------------------------------------------
# Convenience: EEG dimensionality reduction
# ------------------------------------------------------------------


def reduce_eeg_to_pc1(
    eeg_power: np.ndarray,
    fs_hz: float = 10.0,
    lowpass_cutoff_hz: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reduce multi-channel EEG power to first PC, split into slow+fast components.

    Returns both the raw PC1 (for observation) and the lowpass-filtered slow
    component (for state dynamics). The fast fluctuation variance is used as
    EEG observation noise.

    Args:
        eeg_power: [T, E] EEG power (already z-scored per channel)
        fs_hz: sampling rate
        lowpass_cutoff_hz: cutoff for extracting slow neural driver

    Returns:
        eeg_pc1_raw: [T, 1] raw PC1 (full-band, for filter observation)
        eeg_pc1_slow: [T, 1] slow PC1 (lowpass, for state proxy / AR fitting)
        loading: [E] PC loading weights (for reconstruction)
        pc1_std: standard deviation of slow PC1 for un-normalisation
    """
    from scipy.signal import butter, sosfiltfilt

    T, E = eeg_power.shape
    demeaned = eeg_power - eeg_power.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(demeaned, full_matrices=False)
        pc1_raw = U[:, 0] * S[0]
        loading = Vt[0].copy()
    except np.linalg.LinAlgError:
        pc1_raw = demeaned.mean(axis=1)
        loading = np.ones(E, dtype=np.float64)

    # Normalise raw PC1 to unit variance
    pc1_raw_std = float(np.std(pc1_raw)) + 1e-8
    pc1_norm = pc1_raw / pc1_raw_std

    # Lowpass to extract slow hemodynamic-relevant component
    nyquist = fs_hz * 0.5
    cutoff = min(lowpass_cutoff_hz, nyquist * 0.9)
    sos = butter(4, cutoff / nyquist, btype='lowpass', output='sos')
    pc1_slow = sosfiltfilt(sos, pc1_norm)

    # The slow component defines the state scale for AR fitting
    pc1_std = float(np.std(pc1_slow)) + 1e-8
    pc1_slow_norm = pc1_slow / pc1_std

    return (
        pc1_norm[:, np.newaxis].astype(np.float64),           # raw for observation
        pc1_slow_norm[:, np.newaxis].astype(np.float64),      # slow for state proxy
        loading.astype(np.float64),
        pc1_std,
    )


def reconstruct_eeg_from_pc1(
    state_estimate: np.ndarray,
    loading: np.ndarray,
    pc1_std: float,
    eeg_mean: np.ndarray,
    eeg_std: np.ndarray,
) -> np.ndarray:
    """Reconstruct full-channel EEG power from scalar state estimate.

    Args:
        state_estimate: [T] or [T, 1] filtered state (normalised space)
        loading: [E] PC1 loading weights
        pc1_std: std of raw PC1 before normalisation
        eeg_mean: [E] original EEG mean per channel
        eeg_std: [E] original EEG std per channel

    Returns:
        [T, E] EEG power in original units
    """
    state = np.asarray(state_estimate, dtype=np.float64).reshape(-1)
    pc1_reconstructed = state * pc1_std
    eeg_norm = pc1_reconstructed[:, np.newaxis] @ loading[np.newaxis, :]
    return np.asarray(eeg_norm * eeg_std[np.newaxis, :] + eeg_mean[np.newaxis, :],
                      dtype=np.float64)


# ------------------------------------------------------------------
# Convenience: build a model from data
# ------------------------------------------------------------------

def estimate_ar1_params(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit AR(1) model to a univariate signal: s_k = α·s_{k-1} + w_k.

    Args:
        signal: [T] or [T, 1] time series (should be zero-mean, unit-variance)

    Returns:
        alpha: AR coefficient
        q: process noise variance (scalar, for normalised signal)
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x_lag = x[:-1]
    x_cur = x[1:]
    # Yule-Walker / OLS estimate
    alpha = np.dot(x_cur, x_lag) / max(np.dot(x_lag, x_lag), 1e-8)
    alpha = np.clip(alpha, 0.90, 0.999)  # slow neural dynamics: high temporal continuity
    residuals = x_cur - alpha * x_lag
    q = float(np.var(residuals))
    q = max(q, 1e-6)
    return np.array([[alpha]]), np.array([[q]])


@dataclass
class NormalizationParams:
    """Per-channel normalisation parameters for observations."""
    eeg_mean: np.ndarray   # [E]
    eeg_std: np.ndarray    # [E]
    fnirs_mean: np.ndarray # [F]
    fnirs_std: np.ndarray  # [F]


def normalize_observations(
    eeg_power: np.ndarray, fnirs_signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, NormalizationParams]:
    """Z-score per-channel to put EEG and fNIRS on comparable scales."""
    eeg_mean = eeg_power.mean(axis=0, keepdims=True)
    eeg_std = eeg_power.std(axis=0, keepdims=True) + 1e-8
    fnirs_mean = fnirs_signal.mean(axis=0, keepdims=True)
    fnirs_std = fnirs_signal.std(axis=0, keepdims=True) + 1e-8

    eeg_norm = (eeg_power - eeg_mean) / eeg_std
    fnirs_norm = (fnirs_signal - fnirs_mean) / fnirs_std

    params = NormalizationParams(
        eeg_mean=eeg_mean.squeeze(0),
        eeg_std=eeg_std.squeeze(0),
        fnirs_mean=fnirs_mean.squeeze(0),
        fnirs_std=fnirs_std.squeeze(0),
    )
    return np.asarray(eeg_norm, dtype=np.float64), np.asarray(fnirs_norm, dtype=np.float64), params


def denormalize_observations(
    eeg_norm: np.ndarray, fnirs_norm: np.ndarray, params: NormalizationParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse the z-score normalisation."""
    eeg_raw = eeg_norm * params.eeg_std[np.newaxis, :] + params.eeg_mean[np.newaxis, :]
    fnirs_raw = fnirs_norm * params.fnirs_std[np.newaxis, :] + params.fnirs_mean[np.newaxis, :]
    return eeg_raw, fnirs_raw


def build_model_from_data(
    eeg_obs: np.ndarray,
    eeg_slow_proxy: np.ndarray,
    fnirs_signal: np.ndarray,
    hrf_kernel: np.ndarray,
    n_particles: int = 500,
    seed: int = 42,
) -> NeurovascularSMCFilter:
    """Build a NeurovascularSMCFilter with parameters estimated from data.

    Args:
        eeg_obs: [T, E_dim] raw EEG observation (e.g., full-band PC1, z-scored)
        eeg_slow_proxy: [T, E_dim] slow EEG component for AR fitting (lowpass PC1)
        fnirs_signal: [T, F] fNIRS signal, z-scored per channel
        hrf_kernel: [L] HRF kernel
        n_particles: number of particles
        seed: random seed

    Returns:
        configured NeurovascularSMCFilter
    """
    T, E_dim = eeg_obs.shape
    F = fnirs_signal.shape[1]

    eeg_pc_slow = eeg_slow_proxy[:, 0]  # [T], slow component

    # --- state dynamics (fit AR(1) to slow neural driver) ---
    A, Q = estimate_ar1_params(eeg_pc_slow)
    H_eeg = np.ones((E_dim, 1), dtype=np.float64)

    # EEG observation noise: variance of fast fluctuations (raw - slow)
    eeg_fast = eeg_obs[:, 0] - eeg_pc_slow
    eeg_noise_var = max(float(np.var(eeg_fast)), 0.05)
    R_eeg = np.diag([eeg_noise_var])

    # --- fNIRS forward model ---
    H_fnirs = np.ones((F, 1), dtype=np.float64)
    hrf_raw = np.convolve(eeg_pc_slow, hrf_kernel, mode='full')[:T]
    for f in range(F):
        y = fnirs_signal[:, f].astype(np.float64)
        y = y - y.mean()
        denom = np.dot(hrf_raw, hrf_raw)
        if denom > 1e-8:
            H_fnirs[f, 0] = float(np.dot(y, hrf_raw) / denom)
        else:
            H_fnirs[f, 0] = 0.0

    fnirs_residuals = fnirs_signal - hrf_raw[:, np.newaxis] @ H_fnirs.T
    R_fnirs = np.diag(np.clip(np.var(fnirs_residuals, axis=0), 1e-4, None))

    return NeurovascularSMCFilter(
        hrf_kernel=hrf_kernel,
        state_transition_matrix=A,
        process_noise_cov=Q,
        eeg_forward=H_eeg,
        fnirs_forward=H_fnirs,
        eeg_noise_cov=R_eeg,
        fnirs_noise_cov=R_fnirs,
        n_particles=n_particles,
        seed=seed,
    )
