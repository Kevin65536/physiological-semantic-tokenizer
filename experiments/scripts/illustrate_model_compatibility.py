#!/usr/bin/env python3
"""Synthetic, reproducible visual explanations of five model-compatibility checks.

This communication workflow reads no datasets or retained runs. Linear examples
use the owning Balloon rest Jacobian; nonlinear examples use its existing forward
solver. It is not a measured-data evaluator or a calibrated hypothesis test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import t3a_balloon_robust_ssm as core
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward


PARAMETERS = core.BalloonParameters(free=core.BalloonFreeParameters(kappa=.64, tau=2.))
RHO = PARAMETERS.fixed.Q0 / PARAMETERS.fixed.P0
B = -(1 - PARAMETERS.fixed.E0) * np.log1p(-PARAMETERS.fixed.E0) / PARAMETERS.fixed.E0
XI, ZETA = 1 - B, 1 - B / PARAMETERS.fixed.alpha
COLORS = {"HbO": "#c95743", "HbR": "#287bb5", "model": "#187969",
          "error": "#ac3970", "EEG": "#6856a6", "gray": "#71808e"}
SCHEMA = "model_compatibility_illustration_v1"


def linear_example(t, tau=2., frequencies=(1/30, 1/12, 1/7.5)):
    """Exact periodic small-signal solution, including analytic derivatives.

    There is no zero-state start: the periodic steady-state initial condition is
    part of the example. The derivative of order zero is the actual observation.
    """
    parameters = replace(PARAMETERS, free=replace(PARAMETERS.free, tau=tau))
    jac = core.balloon_rhs_jacobian(np.zeros(6), parameters)
    obs = core.observation_jacobian(np.zeros(6), parameters)
    a, b = jac[1:, 1:], jac[1:, 0]
    derivatives = np.zeros((4, len(t), 3))
    state = np.zeros((len(t), 5))
    for frequency, amplitude, phase in zip(frequencies, (.014, .009, .006), (.2, 1.1, -.4)):
        omega = 2 * np.pi * frequency
        response = np.linalg.solve(1j * omega * np.eye(5) - a, b)
        oscillation = amplitude * np.exp(1j * (omega * t + phase))
        state += np.real(oscillation[:, None] * response)
        loading = obs[:, 1:] @ response + obs[:, 0]
        for order in range(4):
            derivatives[order] += np.real((1j * omega)**order * oscillation[:, None] * loading)
    return derivatives, state


def weak_bank(t, widths=(4., 8.), spacing=2.):
    """Compact smooth test functions; all integrals use fixed Simpson weights.

    width is the half-support in seconds. phi and its first two derivatives
    vanish at each support boundary. Overlapping functions are NOT independent.
    """
    dt = float(t[1] - t[0])
    if len(t) % 2 != 1 or not np.allclose(np.diff(t), dt):
        raise ValueError("weak quadrature needs an odd, equally spaced time grid")
    weights = np.ones(len(t))
    weights[1:-1:2], weights[2:-1:2] = 4., 2.
    weights *= dt / 3
    poly = Polynomial([1, 0, -4, 0, 6, 0, -4, 0, 1])
    bank, positions = [], []
    for width in widths:
        for center in np.arange(t[0] + width, t[-1] - width + 1e-8, spacing):
            u = (t - center) / width
            support = np.abs(u) < 1
            row = []
            for order in range(4):
                phi = np.zeros(len(t))
                phi[support] = poly.deriv(order)(u[support]) / (width**order * width * 256/315)
                row.append(phi * weights)
            bank.append(row)
            positions.append((center, width))
    return np.transpose(bank, (1, 0, 2)), np.asarray(positions)


def hb_closure(y, t, bank, tau=2., *, initial_transient=True):
    """First-order Hb closure, with an independently profiled legal transient.

    D here is a descriptive normalized energy, not noise-whitened chi-square.
    Its whole-record denominator is fixed throughout each parameter profile.
    """
    total, deoxy = y[:, 1] + y[:, 2], y[:, 2]
    total_m, deoxy_m = bank[:2] @ total, bank[:2] @ deoxy
    left = deoxy_m[0] - tau * deoxy_m[1]
    right = RHO * (ZETA * total_m[0] - XI * tau * total_m[1])
    transient = bank[0] @ np.exp(-(t - t[0]) / tau)
    c0 = float((left - right) @ transient / (transient @ transient)) if initial_transient else 0.
    right = right + c0 * transient
    residual = left - right
    denominator = float(deoxy_m[0] @ deoxy_m[0] + RHO**2 * (total_m[0] @ total_m[0]))
    return dict(D=float(residual @ residual / denominator) if denominator > 1e-24 else None,
                c0=c0, left=left, right=right, residual=residual, denominator=denominator)


def cross_closure(y, bank, tau=2., kappa=.64):
    """Weak third-order EEG--Hb closure with a nonnegative composite gain."""
    contrast = RHO * y[:, 1] - (1 - RHO) * y[:, 2]
    moment = bank @ contrast
    eeg = bank[0] @ y[:, 0]
    gamma = PARAMETERS.fixed.gamma
    left = (-tau * moment[3] + (1 + kappa * tau) * moment[2]
            - (kappa + gamma * tau) * moment[1] + gamma * moment[0])
    if eeg @ eeg <= 1e-24:
        return dict(D=None, gain=None, left=left, right=np.zeros_like(left), residual=left)
    gain = max(0., float(eeg @ left / (eeg @ eeg)))
    right = gain * eeg
    residual = left - right
    denominator = float(gamma**2 * (moment[0] @ moment[0]))
    return dict(D=float(residual @ residual / denominator) if denominator > 1e-24 else None,
                gain=gain, left=left, right=right, residual=residual, denominator=denominator)


def spatial_scores(y, mixing):
    """y has stacked [HbO time; HbR time] rows; mixing is [channel, core]."""
    projection = mixing @ np.linalg.pinv(mixing)
    fitted = y @ projection
    residual = y - fitted
    singular = np.linalg.svd(y, compute_uv=False)
    energy = float(np.sum(y*y))
    return dict(D=float(np.sum(residual**2)/energy),
                rank_D=float(np.sum(singular[mixing.shape[1]:]**2)/energy),
                fitted=fitted, residual=residual, energy=energy,
                residual_energy=float(np.sum(residual**2)))


def nonlinear_closure(t, y, d0, *, tau=2.):
    """Reconstruct using Hb observations only, no generator states or EEG.

    This noiseless teaching example has known absolute model coordinates and g=1.
    Cubic-spline derivatives are justified only for these densely sampled smooth
    synthetic signals; measured data require a declared smoothing/noise model.
    """
    p = 1 + (y[:, 1] + y[:, 2]) / PARAMETERS.fixed.P0
    q = 1 + y[:, 2] / PARAMETERS.fixed.Q0
    p_spline, q_spline = CubicSpline(t, p), CubicSpline(t, q)
    exponent = 1 / PARAMETERS.fixed.alpha - 1
    if np.min(p) <= 0 or np.min(q) <= 0 or np.any(PARAMETERS.fixed.Q0*q > PARAMETERS.fixed.P0*p):
        return dict(valid=False, reason="invalid absolute Hb states")

    def rhs(time, d):
        volume = float(p_spline(time)) - d[0]
        if volume <= 0:
            raise FloatingPointError("invalid reconstructed volume")
        return [-volume**exponent * d[0] / tau]

    try:
        solution = solve_ivp(rhs, (t[0], t[-1]), [d0], t_eval=t, rtol=2e-10, atol=2e-12)
    except FloatingPointError:
        return dict(valid=False, reason="invalid reconstructed volume")
    if not solution.success:
        return dict(valid=False, reason=solution.message)
    d = solution.y[0]
    volume = p - d
    if np.min(volume) <= 0:
        return dict(valid=False, reason="invalid reconstructed volume")
    loss = volume**exponent
    flow = tau * p_spline(t, 1) + loss * p
    if np.min(flow) <= 0:
        return dict(valid=False, reason="invalid reconstructed flow")
    right = flow * (-np.expm1(np.log1p(-PARAMETERS.fixed.E0)/flow)) / PARAMETERS.fixed.E0
    left = tau * q_spline(t, 1) + loss * q
    mask = (t >= t[0] + 1) & (t <= t[-1] - 1)
    return dict(valid=True, p=p, q=q, volume=volume, flow=flow, left=left, right=right,
                residual=left-right, mask=mask, mse=float(np.mean((left[mask]-right[mask])**2)))


def prepare_examples():
    t = np.linspace(0, 60, 2401)
    bank, positions = weak_bank(t)
    derivatives, state = linear_example(t)
    y = derivatives[0]
    hb_bad = y.copy()
    hb_bad[:, 2] += .8 * np.std(y[:, 2]) * np.sin(2*np.pi*t/10 + .3)
    eeg_bad = y.copy()
    eeg_bad[:, 0] = linear_example(t + 5)[0][0, :, 0]
    hb_good_result, hb_bad_result = [hb_closure(z, t, bank) for z in (y, hb_bad)]
    cross_good, cross_bad = [cross_closure(z, bank) for z in (y, eeg_bad)]

    # A lawful initial-content relaxation with exactly +1 Hb correlation.
    relaxation = np.zeros_like(y)
    amplitude = .04 * np.exp(-t/2)
    relaxation[:, 1] = (PARAMETERS.fixed.P0-PARAMETERS.fixed.Q0) * amplitude
    relaxation[:, 2] = PARAMETERS.fixed.Q0 * amplitude
    relaxation_good = hb_closure(relaxation, t, bank)
    relaxation_wrong = hb_closure(relaxation, t, bank, initial_transient=False)

    # Spatial controls use the same time coordinate, downsampled to 4 Hz.
    spatial_t = t[::10]
    mixing = np.array([1., .7, .4])
    mixing /= np.linalg.norm(mixing)
    orthogonal = np.array([-.7, 1., 0.])
    orthogonal /= np.linalg.norm(orthogonal)
    coefficient = np.r_[y[::10, 1], y[::10, 2]]
    base_space = coefficient[:, None] * mixing
    rotated = coefficient[:, None] * (np.cos(np.pi/6)*mixing + np.sin(np.pi/6)*orthogonal)
    extra = np.sin(np.arange(len(coefficient)) * .12)
    extra -= coefficient * (coefficient @ extra) / (coefficient @ coefficient)
    extra *= .5 * np.linalg.norm(coefficient) / np.linalg.norm(extra)
    added = base_space + extra[:, None] * orthogonal
    spatial_inputs = [base_space, rotated, added]
    spatial = [spatial_scores(z, mixing[:, None]) for z in spatial_inputs]

    # Actual profiles, not drawn acceptance bands. Only tau varies; all other
    # time parameters are fixed. c0 and the composite cross gain are refitted.
    taus = np.linspace(.5, 8., 151)
    conflict = linear_example(t, tau=5.)[0]
    x = RHO*conflict[:, :, 1] - (1-RHO)*conflict[:, :, 2]
    gamma, kappa = PARAMETERS.fixed.gamma, PARAMETERS.free.kappa
    conflict[0, :, 0] = (x[3] + (1+kappa)*x[2] + (kappa+gamma)*x[1] + gamma*x[0]) / (PARAMETERS.fixed.Q0*B)
    boundary = linear_example(t, tau=7.9)[0][0]
    jac = core.balloon_rhs_jacobian(np.zeros(6), PARAMETERS)
    constant_state = np.linalg.solve(-jac[1:, 1:], jac[1:, 0]*.01)
    obs = core.observation_jacobian(np.zeros(6), PARAMETERS)
    weak_y = np.broadcast_to(obs[:, 1:] @ constant_state + obs[:, 0]*.01, y.shape).copy()
    profile_inputs = [y, conflict[0], boundary, weak_y]
    profiles = []
    tolerance = .01  # Educational energy tolerance, never an empirical cutoff.
    for sample in profile_inputs:
        dh = np.array([hb_closure(sample, t, bank, tau=float(tau))["D"] for tau in taus])
        de = np.array([cross_closure(sample, bank, tau=float(tau))["D"] for tau in taus])
        accepted = (dh <= tolerance) & (de <= tolerance)
        feasible = taus[accepted]
        inward = np.minimum(feasible-.5, 8-feasible)/7.5
        profiles.append(dict(hb=dh, cross=de, accepted=accepted,
                             feasible_count=int(accepted.sum()),
                             feasible_min=float(feasible.min()) if len(feasible) else None,
                             feasible_max=float(feasible.max()) if len(feasible) else None,
                             maximum_inward_fraction=float(inward.max()) if len(inward) else None))

    # Full nonlinear examples: change only driver amplitude; all physical
    # initials fixed. A separate Hb redistribution leaves total Hb unchanged.
    pulse = np.exp(-.5*((t-12)/3)**2) + .65*np.exp(-.5*((t-34)/4)**2)
    nonlinear = []
    for scale in (.012, .22):
        result = nonlinear_driver_forward(scale*pulse, np.array([0., 1., 1., 1., 1.]),
                                          PARAMETERS, t[1]-t[0], substeps=2, derivative=False, hemodynamics='inlet_balance', viscoelastic_s=0., numerical_backend='python')
        nonlinear.append(result)
    nonlinear_y = [z["canonical_prediction"] for z in nonlinear]
    corrupt = nonlinear_y[1].copy()
    dq = .04*np.sin(2*np.pi*t/10) * np.sin(np.pi*t/60)**4
    corrupt[:, 2] += PARAMETERS.fixed.Q0*dq
    corrupt[:, 1] -= PARAMETERS.fixed.Q0*dq
    nonlinear_y.append(corrupt)
    d0_grid = np.linspace(-.12, .12, 49)
    reference = nonlinear_closure(t, nonlinear_y[1], 0.)
    nl_scale_squared = float(np.mean((reference["right"][reference["mask"]]-1)**2))
    nl_profiles, nl_best = [], []
    for sample in nonlinear_y:
        results = [nonlinear_closure(t, sample, float(d0)) for d0 in d0_grid]
        score = np.array([z["mse"]/nl_scale_squared if z["valid"] else np.nan for z in results])
        best_index = int(np.nanargmin(score))
        nl_best.append({**results[best_index], "d0": float(d0_grid[best_index]), "D": float(score[best_index])})
        nl_profiles.append(score)
    nl_linear = [hb_closure(z, t, bank)["D"] for z in nonlinear_y]
    refined = nonlinear_driver_forward(.22*pulse, np.array([0., 1., 1., 1., 1.]),
                                       PARAMETERS, t[1]-t[0], substeps=4, derivative=False, hemodynamics='inlet_balance', viscoelastic_s=0., numerical_backend='python')
    numerical_error = float(np.max(np.abs(refined["canonical_prediction"]-nonlinear_y[1])))

    rng = np.random.default_rng(20261023)
    noise_sigma = float(np.sqrt(np.mean(base_space**2))*.15)
    null_q, null_d = [], []
    for _ in range(500):
        score = spatial_scores(base_space+rng.normal(0, noise_sigma, base_space.shape), mixing[:, None])
        null_q.append(score["residual_energy"]/noise_sigma**2)
        null_d.append(score["D"])
    noisy_added = spatial_scores(added+rng.normal(0, noise_sigma, base_space.shape), mixing[:, None])
    degrees = base_space.shape[0]*(base_space.shape[1]-1)

    return dict(t=t, bank=bank, positions=positions, y=y, derivatives=derivatives, state=state,
                hb_bad=hb_bad, eeg_bad=eeg_bad, hb_good_result=hb_good_result, hb_bad_result=hb_bad_result,
                cross_good=cross_good, cross_bad=cross_bad, relaxation=relaxation,
                relaxation_good=relaxation_good, relaxation_wrong=relaxation_wrong,
                spatial_t=spatial_t, mixing=mixing, spatial_inputs=spatial_inputs, spatial=spatial,
                taus=taus, profiles=profiles, profile_inputs=profile_inputs, tolerance=tolerance,
                nonlinear_y=nonlinear_y, nonlinear=nonlinear, nl_best=nl_best, nl_profiles=nl_profiles,
                nl_linear=nl_linear, d0_grid=d0_grid, nl_scale_squared=nl_scale_squared,
                nonlinear_refinement_error=numerical_error, noise_sigma=noise_sigma,
                null_q=np.array(null_q), null_d=np.array(null_d), noise_degrees=degrees,
                noisy_added_D=noisy_added["D"], noisy_added_Q=noisy_added["residual_energy"]/noise_sigma**2)


def numerical_summary(data):
    return {
        "schema": SCHEMA, "kind": "synthetic_educational_examples", "measured_records": 0,
        "linear_parameters": asdict(PARAMETERS), "B": B, "xi": XI, "zeta": ZETA,
        "linear_periods_seconds": [30, 12, 7.5], "duration_seconds": 60,
        "quadrature_step_seconds": .025, "space_sample_rate_hz": 4,
        "profile_tau_range_seconds": [.5, 8.], "profile_tau_step_seconds": .05,
        "profile_other_time_parameters": "fixed; c0 and nonnegative composite gain are profiled separately",
        "illustrative_tolerance_energy": data["tolerance"], "noise_seed": 20261023,
        "noise_realizations": len(data["null_q"]), "statistical_independent_subjects": 0,
        "space": [{k: z[k] for k in ("D", "rank_D", "energy", "residual_energy")} for z in data["spatial"]],
        "hb": {"legal_D": data["hb_good_result"]["D"], "relative_HbR_perturbation_D": data["hb_bad_result"]["D"],
               "legal_relaxation_D": data["relaxation_good"]["D"],
               "wrong_zero_initial_transient_D": data["relaxation_wrong"]["D"],
               "relaxation_c0": data["relaxation_good"]["c0"]},
        "cross": {"legal_D": data["cross_good"]["D"], "eeg_advanced_5s_D": data["cross_bad"]["D"],
                  "legal_composite_gain": data["cross_good"]["gain"], "shifted_composite_gain": data["cross_bad"]["gain"]},
        "parameter_slices": [{k: z[k] for k in ("feasible_count", "feasible_min", "feasible_max", "maximum_inward_fraction")}
                             for z in data["profiles"]],
        "nonlinear": {"linear_hb_D": data["nl_linear"], "D": [z["D"] for z in data["nl_best"]],
                      "best_d0": [z["d0"] for z in data["nl_best"]],
                      "d0_grid": [-.12, .12, .005], "tau_fixed_seconds": 2.,
                      "score_support_seconds": [1, 59], "normalizer_squared": data["nl_scale_squared"],
                      "invalid_grid_nodes": [int(np.count_nonzero(~np.isfinite(z))) for z in data["nl_profiles"]],
                      "rk4_refinement_max_absolute_error": data["nonlinear_refinement_error"]},
        "space_noise": {"sigma": data["noise_sigma"], "residual_degrees_of_freedom": data["noise_degrees"],
                        "null_mean_Q_over_df": float(np.mean(data["null_q"])/data["noise_degrees"]),
                        "null_D_95_percentile": float(np.quantile(data["null_d"], .95)),
                        "perturbed_D": data["noisy_added_D"],
                        "perturbed_Q_over_df": data["noisy_added_Q"]/data["noise_degrees"]},
    }


def draw_figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font_path).get_name(),
                         "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .18, "lines.linewidth": 1.8,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    figures = output / "figures"
    figures.mkdir()
    paths = {}

    def save(fig, name):
        path = figures / f"{name}.png"
        fig.savefig(path, dpi=250)
        plt.close(fig)
        paths[name] = str(path.relative_to(output))

    def zero(ax):
        ax.axhline(0, color=COLORS["gray"], lw=.8)

    def time_axes(ax, ylabel="模型 Hb 坐标"):
        ax.set(xlabel="时间（秒）", ylabel=ylabel)
        zero(ax)

    t, y, positions = data["t"], data["y"], data["positions"]
    short = positions[:, 1] == 4
    center = positions[short, 0]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    cards = [
        (r"$D_{\mathrm{space}}$", "空间方向", "多通道 Hb → 投影 → 垂直残差", "空间读出能容纳多少观测能量？"),
        (r"$D_{\mathrm{Hb}}$", "Hb 内部动力学", "HbO + HbR → 动态闭合 → 残差", "不看 EEG，Hb 自身能否共同产生？"),
        (r"$D_{\mathrm{EEG-Hb}}$", "跨模态动力学", "氧合对比 + EEG 特征 → 弱积分", "允许调幅后，时间结构是否一致？"),
        (r"$\mathcal{C}_{\theta}$", "共同参数集合", "候选参数 → 各项通过范围 → 交集", "存在、贴边、很宽，还是没有交集？"),
        (r"$D_{\mathrm{Hb,NL}}$", "完整非线性复核", "绝对 Hb 坐标 → 重建流量 → 再闭合", "线性异常是否只是合法非线性？"),
    ]
    for i, (symbol, title, route, question) in enumerate(cards):
        yy = 5.05-i*1.02
        patch = FancyBboxPatch((.1, yy), 9.8, .85, boxstyle="round,pad=.05,rounding_size=.09",
                               facecolor="#edf4f8" if i != 3 else "#e8f4ed", edgecolor="#bdced8")
        ax.add_patch(patch)
        ax.text(.35, yy+.54, f"0{i+1}", fontsize=15, color=COLORS["model"], weight="bold")
        ax.text(1.25, yy+.51, symbol, fontsize=16)
        ax.text(3.5, yy+.55, title, fontsize=12, weight="bold")
        ax.text(3.5, yy+.14, question, fontsize=9.5, color="#43566a")
    ax.text(.1, .16, "五项互补；先报告每项残差、适用条件与检验力，再解释联合结果。", fontsize=10.5)
    save(fig, "01_five_questions")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), layout="constrained")
    m = data["mixing"]
    n = np.array([-.7, 1., 0.]); n /= np.linalg.norm(n)
    count = len(data["spatial_t"])
    observed = data["spatial_inputs"][1][:count]
    along, outside = observed@m, observed@n
    axes[0, 0].plot(along[::5], outside[::5], "o", ms=3, color=COLORS["HbR"], label="观测：一个点是一时刻")
    axes[0, 0].axhline(0, color=COLORS["model"], label="固定空间方向")
    for index in (25, 60, 80, 125, 180):
        axes[0, 0].plot([along[index]]*2, [0, outside[index]], color=COLORS["error"], alpha=.55)
    axes[0, 0].set(title="A. 旋转空间方向 30°", xlabel="沿固定方向的坐标", ylabel="与固定方向垂直的坐标")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    ax = axes[0, 1]
    score = data["spatial"][1]
    ax.plot(data["spatial_t"], observed[:, 1], color=COLORS["HbR"], label="通道 2：观测")
    ax.plot(data["spatial_t"], score["fitted"][:count, 1], "--", color=COLORS["model"], label="空间投影")
    ax.plot(data["spatial_t"], score["residual"][:count, 1], ":", color=COLORS["error"], label="观测 - 投影")
    ax.set_title("B. 投影后还剩什么波形？")
    time_axes(ax); ax.legend(fontsize=8)
    labels = ["合法方向", "方向旋转", "额外维度"]
    scores = np.array([z["D"] for z in data["spatial"]])
    axes[1, 0].bar(labels, 100*(1-scores), color=COLORS["model"], label="投影能量")
    axes[1, 0].bar(labels, 100*scores, bottom=100*(1-scores), color=COLORS["error"], label="垂直残差能量")
    for i, value in enumerate(scores):
        label_y = 50 if value < 1e-10 else 100-50*value
        axes[1, 0].text(i, label_y, f"残差\n{100*value:.0f}%", ha="center", va="center", color="white", fontsize=10)
    axes[1, 0].set(title="C. 先平方求和，再除以总能量", ylabel="占该例总观测能量（%）", ylim=(0, 122))
    axes[1, 0].legend(loc="upper right", ncol=2, fontsize=8)
    x = np.arange(3)
    axes[1, 1].bar(x-.18, 100*scores, .36, color=COLORS["error"], label="固定方向残差")
    axes[1, 1].bar(x+.18, [100*z["rank_D"] for z in data["spatial"]], .36, color=COLORS["HbR"], label="自由秩 1 残差")
    axes[1, 1].set(xticks=x, xticklabels=labels, title="D. 方向不对，还是维度不够？", ylabel="残差能量（%）", ylim=(0, 34))
    axes[1, 1].legend(fontsize=8)
    save(fig, "02_space_projection")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), layout="constrained")
    axes[0, 0].plot(t, y[:, 1], color=COLORS["HbO"], label="HbO：保持不变")
    axes[0, 0].plot(t, y[:, 2], color=COLORS["HbR"], label="HbR：合法")
    axes[0, 0].plot(t, data["hb_bad"][:, 2], "--", color=COLORS["error"], label="HbR：另加一个成分")
    axes[0, 0].set_title("A. 输入两条 Hb 曲线")
    axes[0, 0].legend(fontsize=8); time_axes(axes[0, 0])
    for sample, style, color, label in ((y, "-", COLORS["model"], "合法总 Hb"),
                                      (data["hb_bad"], "--", COLORS["error"], "扰动后总 Hb")):
        axes[0, 1].plot(t, sample[:, 1]+sample[:, 2], style, color=color, label=label)
    axes[0, 1].set_title("B. 相加得到总 Hb")
    time_axes(axes[0, 1]); axes[0, 1].legend(fontsize=8)
    bad = data["hb_bad_result"]
    axes[1, 0].plot(center, bad["left"][short], "o-", ms=3, color=COLORS["HbR"], label="闭合左侧")
    axes[1, 0].plot(center, bad["right"][short], "s--", ms=3, color=COLORS["model"], label="右侧 + 最优初态项")
    axes[1, 0].set(title="C. 对同一局部区间积分", xlabel="积分窗中心（秒）", ylabel="局部加权模型 Hb")
    axes[1, 0].legend(fontsize=8); zero(axes[1, 0])
    for result, color, style, label in ((data["hb_good_result"], COLORS["model"], "-", "合法"),
                                      (bad, COLORS["error"], "--", "HbR 扰动")):
        axes[1, 1].plot(center, result["residual"][short], style, color=color, label=label)
    axes[1, 1].set(title=f"D. 扰动后的 D = {100*bad['D']:.2f}%", xlabel="积分窗中心（秒）", ylabel="闭合残差：左侧 - 右侧")
    zero(axes[1, 1]); axes[1, 1].legend(fontsize=8)
    save(fig, "03_hb_closure")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5), layout="constrained")
    use = t <= 14
    for channel, label in ((1, "HbO"), (2, "HbR")):
        axes[0, 0].plot(t[use], data["relaxation"][use, channel], color=COLORS[label], label=label)
    axes[0, 0].set_title("A. 完全同步，但方程严格合法")
    time_axes(axes[0, 0]); axes[0, 0].legend()
    axes[0, 0].text(.45, .78, "相关系数 = +1", transform=axes[0, 0].transAxes, fontsize=10)
    transient = .04*RHO*B*(1/PARAMETERS.fixed.alpha-1)*np.exp(-t/2)
    fitted = data["relaxation_good"]["c0"]*np.exp(-t/2)
    axes[0, 1].plot(t[use], transient[use], color=COLORS["HbR"], label="由 Hb 得到的括号项")
    axes[0, 1].plot(t[use], fitted[use], "--", color=COLORS["model"], label="允许的指数初态项")
    axes[0, 1].set_title("B. 应扣掉合法初态瞬态")
    time_axes(axes[0, 1]); axes[0, 1].legend(fontsize=8)
    frequencies = np.geomspace(.001, 2, 400)
    ratios = []
    for tau in data["taus"]:
        iw = 2j*np.pi*frequencies*tau
        ratios.append(RHO*(ZETA+iw*XI)/(1-RHO*ZETA+iw*(1-RHO*XI)))
    ratios = np.array(ratios)
    ratio = ratios[np.argmin(abs(data["taus"]-2))]
    for ax, values, line, ylabel, title in (
        (axes[1, 0], abs(ratios), abs(ratio), "HbR / HbO 幅度比", "C. 稳态幅度也受到约束"),
        (axes[1, 1], np.angle(ratios, deg=True), np.angle(ratio, deg=True), "HbR 相对 HbO 相位（度）", "D. 低频反相，高频趋向同相")):
        ax.fill_between(frequencies, values.min(axis=0), values.max(axis=0), alpha=.2, color=COLORS["HbR"], label="候选参数包络")
        ax.plot(frequencies, line, color=COLORS["HbR"], label="固定 τ = 2 秒")
        ax.set(xscale="log", xlabel="模型正弦输入频率（Hz）", ylabel=ylabel, title=title)
        ax.legend(fontsize=8)
    save(fig, "04_legal_synchrony_and_spectrum")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), layout="constrained")
    axes[0, 0].plot(t, y[:, 0], color=COLORS["EEG"], label="正确配对的 EEG 特征")
    axes[0, 0].plot(t, data["eeg_bad"][:, 0], "--", color=COLORS["gray"], label="仅将 EEG 提前 5 秒")
    axes[0, 0].set_title("A. 改变跨模态时间对应")
    time_axes(axes[0, 0], "EEG 特征模型坐标"); axes[0, 0].legend(fontsize=8)
    contrast = RHO*y[:, 1]-(1-RHO)*y[:, 2]
    axes[0, 1].plot(t, contrast, color=COLORS["model"], label="两例完全相同的氧合对比")
    axes[0, 1].set_title("B. Hb 曲线和内部闭合均不变")
    time_axes(axes[0, 1]); axes[0, 1].legend(fontsize=8)
    for result, marker, color, label in ((data["cross_good"], "o", COLORS["model"], "正确配对"),
                                        (data["cross_bad"], "x", COLORS["EEG"], "时间错配；重新拟合幅度")):
        axes[1, 0].scatter(result["left"], result["right"], marker=marker, color=color, s=17, label=label)
    limit = max(abs(data["cross_good"]["left"]).max(), abs(data["cross_bad"]["right"]).max())*1.1
    axes[1, 0].plot([-limit, limit], [-limit, limit], "--", color=COLORS["gray"], lw=1, label="两侧相等")
    axes[1, 0].set(title="C. 每点是一条局部弱形式约束", xlabel="由 Hb 算出的左侧", ylabel="复合增益 × EEG 积分")
    axes[1, 0].legend(fontsize=7.5)
    for result, color, style, label in ((data["cross_good"], COLORS["model"], "-", "正确配对"),
                                      (data["cross_bad"], COLORS["EEG"], "--", "EEG 提前")):
        axes[1, 1].plot(center, result["residual"][short], style, color=color, label=label)
    axes[1, 1].set(title=f"D. 错配 D = {100*data['cross_bad']['D']:.2f}%", xlabel="积分窗中心（秒）", ylabel="闭合残差（模型 Hb / 秒²）")
    zero(axes[1, 1]); axes[1, 1].legend(fontsize=8)
    save(fig, "05_eeg_hb_closure")

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), layout="constrained")
    index = int(np.flatnonzero((positions[:, 0] == 24) & (positions[:, 1] == 4))[0])
    use = (t >= 16) & (t <= 32)
    axes[0].plot(t[use], contrast[use], color=COLORS["model"], label="Hb 氧合对比 X")
    axes[0].set(xlabel="时间（秒）", ylabel="模型 Hb 坐标", title="A. 取中心 24 秒、总宽 8 秒的一段曲线")
    twin = axes[0].twinx()
    u = (t-24)/4
    phi = np.zeros(len(t)); support = abs(u) < 1
    phi[support] = (1-u[support]**2)**4/(4*256/315)
    twin.fill_between(t[use], phi[use], color="#b7c9dd", alpha=.4, label="权函数 φ：积分为 1")
    twin.set_ylabel("权函数（1 / 秒）", color=COLORS["gray"])
    axes[0].legend(loc="upper left"); twin.legend(loc="upper right")
    moment = data["bank"][:, index] @ contrast
    tau, kappa, gamma = 2., .64, .32
    first = np.array([-tau*moment[3], (1+kappa*tau)*moment[2], -(kappa+gamma*tau)*moment[1], gamma*moment[0]])
    labels = [r"$-\tau\int\phi'''X$", r"$(1+\kappa\tau)\int\phi''X$", r"$-(\kappa+\gamma\tau)\int\phi'X$", r"$\gamma\int\phi X$", r"$-c\int\phi e$", "五项相加"]
    for offset, sample, result, color, label in ((-.18, y, data["cross_good"], COLORS["model"], "正确配对"),
                                               (.18, data["eeg_bad"], data["cross_bad"], COLORS["EEG"], "EEG 提前 5 秒")):
        terms = np.r_[first, -result["gain"] * (data["bank"][0, index] @ sample[:, 0])]
        axes[1].bar(np.arange(6)+offset, np.r_[terms, sum(terms)], .36, color=color, label=label)
    axes[1].set(xticks=np.arange(6), xticklabels=labels, ylabel="各项贡献（模型 Hb / 秒²）", title="B. 导数作用于已知权函数；把五项相加得到残差")
    axes[1].tick_params(axis="x", labelsize=9, rotation=12)
    zero(axes[1]); axes[1].legend(fontsize=8)
    save(fig, "06_weak_integral_arithmetic")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), layout="constrained")
    titles = ["A. 共同内点", "B. 各自通过，但交集为空", "C. 可行点都靠近上界", "D. 没有时间激励，范围很宽"]
    for ax, profile, title in zip(axes.flat, data["profiles"], titles):
        ax.plot(data["taus"], np.maximum(100*profile["hb"], 1e-8), color=COLORS["HbR"], label="Hb 内部闭合")
        ax.plot(data["taus"], np.maximum(100*profile["cross"], 1e-8), "--", color=COLORS["EEG"], label="EEG–Hb 闭合")
        ax.axhline(100*data["tolerance"], color=COLORS["error"], ls=":", lw=1.3, label="教学容差 1%")
        if profile["feasible_count"]:
            ax.axvspan(profile["feasible_min"], profile["feasible_max"], alpha=.16, color=COLORS["model"], label="共同通过范围")
            text_label = f"共同 τ：{profile['feasible_min']:.2f}–{profile['feasible_max']:.2f} 秒"
        else:
            text_label = "151 个候选 τ：没有共同通过点"
        ax.text(.04, .12, text_label, transform=ax.transAxes, va="bottom", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
        ax.set(title=title, xlabel="候选 τ（秒）", ylabel="归一化残差能量 D（%）", yscale="log", ylim=(4e-9, 1e3), xlim=(.5, 8))
    axes[0, 0].legend(fontsize=7.5, loc="center right")
    save(fig, "07_common_parameter_sets")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.1), layout="constrained")
    clean, bad = data["nonlinear_y"][1:]
    for i, label in ((1, "HbO"), (2, "HbR")):
        axes[0, 0].plot(t, clean[:, i], color=COLORS[label], label=f"{label}：合法")
        axes[0, 0].plot(t, bad[:, i], "--", color=COLORS[label], alpha=.8, label=f"{label}：重分配扰动")
    axes[0, 0].set_title("A. 总 Hb 不变，改变氧合组成")
    time_axes(axes[0, 0]); axes[0, 0].legend(fontsize=7.5)
    result = data["nl_best"][2]
    for key, label, color, style in (("p", "总 Hb 状态 p", COLORS["gray"], "-"),
                                    ("q", "脱氧 Hb 状态 q", COLORS["HbR"], "--"),
                                    ("flow", "由 Hb 重建的血流 f", COLORS["model"], "-")):
        axes[0, 1].plot(t, result[key], style, color=color, label=label)
    axes[0, 1].set(title="B. 用绝对坐标和初态重建流量", xlabel="时间（秒）", ylabel="相对基线状态（无量纲）")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(t, result["left"], color=COLORS["error"], label="扰动：脱氧 Hb 变化 + 流出")
    axes[1, 0].plot(t, result["right"], "--", color=COLORS["model"], label="由流量给出的脱氧 Hb 流入")
    axes[1, 0].set(title="C. 非线性守恒式两侧应该相等", xlabel="时间（秒）", ylabel="相对基线通量（无量纲）")
    axes[1, 0].legend(fontsize=7.5)
    axes[1, 1].plot(t, data["nl_best"][1]["residual"], color=COLORS["model"], label="合法大响应")
    axes[1, 1].plot(t, result["residual"], "--", color=COLORS["error"], label="重分配扰动")
    axes[1, 1].set(title=f"D. 非线性扰动残差 D = {100*result['D']:.2f}%", xlabel="时间（秒）", ylabel="脱氧 Hb 守恒残差")
    zero(axes[1, 1]); axes[1, 1].legend(fontsize=8)
    save(fig, "08_nonlinear_reconstruction")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.5), layout="constrained")
    labels = ["合法\n小响应", "合法\n大响应", "大响应\n加扰动"]
    axes[0].bar(np.arange(3), 100*np.asarray(data["nl_linear"]), color=[COLORS["gray"], COLORS["model"], COLORS["error"]])
    axes[0].set(xticks=np.arange(3), xticklabels=labels, title="A. 线性 Hb 闭合", ylabel="线性 D（%）")
    for i, value in enumerate(data["nl_linear"]):
        axes[0].text(i, 100*value+.04, f"{100*value:.3f}", ha="center", fontsize=8.5)
    axes[0].set_ylim(0, max(data["nl_linear"])*120)
    nl_values = [z["D"] for z in data["nl_best"]]
    axes[1].bar(np.arange(3), 100*np.array(nl_values), color=[COLORS["gray"], COLORS["model"], COLORS["error"]])
    axes[1].set(xticks=np.arange(3), xticklabels=labels, title="B. 完整非线性闭合", ylabel="非线性 D（%）", ylim=(0, 67))
    for i, value in enumerate(nl_values):
        axes[1].text(i, 100*value+1.5, "近于 0" if value < 1e-8 else f"{100*value:.2f}", ha="center", fontsize=8.5)
    for values, color, style, label in ((data["nl_profiles"][1], COLORS["model"], "-", "合法大响应"),
                                      (data["nl_profiles"][2], COLORS["error"], "--", "重分配扰动")):
        axes[2].plot(data["d0_grid"], 100*values, style, color=color, label=label)
    axes[2].set(title="C. 允许改变初态差", xlabel="初始 p - v", ylabel="非线性 D（%）")
    axes[2].legend(fontsize=7.5)
    save(fig, "09_nonlinearity_vs_mismatch")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), layout="constrained")
    axes[0].hist(100*data["null_d"], bins=24, color=COLORS["HbR"], alpha=.8)
    percentile = 100*np.quantile(data["null_d"], .95)
    axes[0].axvline(percentile, color=COLORS["error"], ls="--", label=f"经验 95% 分位：{percentile:.2f}%")
    axes[0].set(title="A. 合法模型加噪声也有空间残差", xlabel="空间残差能量（%）", ylabel="噪声实现次数")
    axes[0].legend(fontsize=8)
    means = [np.mean(data["null_q"])/data["noise_degrees"], data["noisy_added_Q"]/data["noise_degrees"]]
    axes[1].bar(["合法 + 噪声", "额外维度 + 同级噪声"], means, color=[COLORS["model"], COLORS["error"]])
    axes[1].axhline(1, color=COLORS["gray"], ls="--", label="已知白噪声下期望为 1")
    for i, value in enumerate(means):
        axes[1].text(i, value+.35, f"{value:.2f}", ha="center")
    axes[1].set(title="B. 除以噪声方差后再解释", ylabel="白化残差 Q / 残差自由度", ylim=(0, 20))
    axes[1].tick_params(axis="x", labelsize=8.5); axes[1].legend(fontsize=7.5)
    save(fig, "10_noise_and_testability")
    return paths


def report_content(data, figures, summary):
    """Visible explanations are authored alongside the exact numeric examples."""
    sections = []

    def paragraph(text):
        return {"type": "paragraph", "text": text}

    def equation(latex, explanation, source="附件公式；已核对当前 Balloon 核心及本例数值实现"):
        return {"type": "equation", "latex": latex, "explanation": explanation, "source": source}

    def figure(name, caption):
        return {"type": "figure", "path": figures[name], "caption": caption,
                "source": "本次合成示例：summary.json、arrays.npz；illustrate_model_compatibility.py"}

    def table(columns, rows, widths=None, caption=""):
        result = {"type": "table", "columns": columns, "rows": rows, "caption": caption}
        if widths:
            result["widths"] = widths
        return result

    def section(title, blocks, page_break=True):
        sections.append({"title": title, "page_break": page_break, "blocks": blocks})

    section("从五个问题开始读图", [figure("01_five_questions",
        "前三项从空间、Hb 内部与跨模态时间结构三个方向查找冲突；第四项要求同一组参数共同成立；第五项复核线性近似的适用性。它们不合成为一个万能分数。")], False)

    section("先约定：图上的数据、单位和分数分别是什么", [
        paragraph("所有曲线都是本次生成的合成信号。主线性例采用当前核心在静息点的 Jacobian，精确计算三个正弦输入的周期稳态响应；周期为 30、12、7.5 秒，驱动幅度为 0.014、0.009、0.006。初始状态取周期稳态，未假设观察从静息起步。非线性例直接调用现有确定性前向求解器。没有读取实测数据、保护数据或历史运行数组。"),
        table(["量", "本图解中的含义与单位"], [
            ["EEG 特征 e", "驱动的模型读出，增益为 1、偏置为 0；不是原始 EEG 电压，也不是 μV。"],
            ["HbO 与 HbR", "记作 O、R；合成模型 Hb 坐标中的变化量。本例 P0=1、Q0=0.35、共同增益为 1，不能将数值直接写成实测 μM。"],
            ["总 Hb 与氧合对比", "T=O+R；X=0.35O-0.65R。X 是线性对比，不是血氧饱和度。"],
            ["时间与参数", "时间以秒计；τ 为秒，κ 为每秒，γ 为每平方秒。固定 κ=0.64、γ=0.32、α=E0=0.32。候选 τ 为 0.5–8 秒，是声明域。"],
            ["空间矩阵", "3 通道、1 核；固定方向与 (1,0.7,0.4) 成比例，按欧氏长度归一化。HbO、HbR 使用同一矩阵。"],
            ["评分支持", "线性例观察 0–60 秒，以 0.025 秒网格作弱积分；空间例用 4 Hz、241 个点。非线性评分用 1–59 秒，去掉两端各 1 秒。"],
        ], [1, 3.4]),
        paragraph("图上的 D_space 采用附件的原始能量比例。附件的三个动力学量还需要噪声范数与容差才能完成统计定义；本图解明确使用带“示例”上标的归一化残差能量，公式如下各节所示。它们是可复算的教学量，不是已经校准的显著性检验。共同参数图只检查 τ 的一维切片。"),
        paragraph("所有残差都按“左侧减右侧”定义，再平方求和。图中百分数等于 D×100；0.01 显示为 1%。动力学 D 可能超过 100%，不能解释为非生理成分比例。每条剖面的分母固定，不随候选参数、时间点或局部积分窗重新归一化。"),
        paragraph("为把计算过程看清，本例没有去基线、滤波、缺失掩码或过程噪声。当前实测空间协议使用 30 秒、4 Hz 窗口；这些教学曲线不替代该观测合同。只有第十图单独加入已知独立高斯观测噪声。"),
    ])

    space = summary["space"]
    section("量一的计算：空间投影与能量分解", [
        equation(r"\begin{aligned}P_M&=MM^+,\quad \widehat Y=YP_M,\quad E=Y-\widehat Y,\\"
                 r"D_{\mathrm{space}}&=\frac{\|E\|_F^2}{\|Y\|_F^2},\qquad "
                 r"D_{\mathrm{rank},K}=\frac{\sum_{\ell>K}\sigma_\ell^2}{\sum_\ell\sigma_\ell^2}.\end{aligned}",
                 "Y 将 HbO 的时间×通道矩阵与 HbR 矩阵上下拼接；本例形状为 482×3。M 是 3×1 固定读出，伪逆给出正交投影。平方和覆盖两个色团、全部时间和通道。自由秩残差允许重新选择空间方向，只保留 K 个奇异值。"),
        paragraph("计算时先把每个时刻的三通道观测看作一个向量，再投影到 M 指定的方向。投影向量和残差向量垂直，所以二者的平方能量恰好相加为总能量。第一步允许核心时间轨迹完全自由，不需要先拟合动力学。"),
        paragraph("上式的简单右投影使用本例无缺失、通道共用时间算子的条件。若有缺失支持或通道特异性处理，必须将掩码和完整处理算子纳入观测设计后再投影；不能把缺失值补零后照搬这个矩阵公式。"),
        paragraph("下一页比较三个已知例：原空间方向、只旋转空间方向、增加一个正交空间维度。第一项看固定方向能否解释；第二项自由秩残差允许重选方向，用来区分方向错误与源数不足。"),
    ])
    section("量一的图线：方向不合适与维度不足有不同表现", [
        figure("02_space_projection", "A 中坐标是三通道向量沿固定方向及一个正交方向的投影，紫色竖线就是固定空间不能表示的部分；B 展开通道 2 的时间曲线；C 展示能量加法；D 用自由秩残差区分方向错误与维度不足。三例共享时间波形；旋转例只改变空间方向，额外维度例增加与原时间向量、原空间方向都正交的成分。"),
        paragraph(f"旋转例的残差平方和为 {space[1]['residual_energy']:.8f}，总平方和为 {space[1]['energy']:.8f}，相除为 0.25。自由秩 1 可以恢复旋转后的方向，因此其残差近于零。额外成分的长度是原成分的一半，能量是四分之一；残差占总能量为 0.25/(1+0.25)=20%，固定方向与自由秩 1 均得到该值。"),
        paragraph("对同一尺度、掩码和观测算子，这个自由轨迹投影残差是完整物理模型误差的下界：放开上游驱动和生理参数也不能把固定空间之外的部分拟合出来。它定位的是读出合同的局限；不能凭此给残差贴上头皮、仪器或脑源标签。"),
    ])

    section("量二的计算：Hb 内部闭合必须保留初态项", [
        equation(r"\begin{aligned}T&=O+R,\quad \rho=Q_0/P_0=0.35,\\"
                 r"B&=-\frac{(1-E_0)\log(1-E_0)}{E_0},\\"
                 r"\xi&=1-B,\quad\zeta=1-B/\alpha,\\"
                 r"h(t)&=(\tau D+1)R-\rho(\xi\tau D+\zeta)T,\\"
                 r"h(t)&=c_0e^{-t/\tau}\quad\text{under the linear model}.\end{aligned}",
                 f"D 表示时间求导。E0 是模型基线氧提取分数，α 控制体积与静脉流出的关系；B 是两种 Hb 对小流量变化的响应系数之差。本例 B={B:.6f}、ξ={XI:.6f}、ζ={ZETA:.6f}。τ 是共同运输时间，c0 代表允许的 Hb 与体积初态差。该关系不使用 EEG，也不依赖上游 κ 或神经血管增益。"),
        equation(r"\begin{aligned}a_j&=\int\phi_j R\,dt-\tau\int\phi'_jR\,dt,\\"
                 r"b_j&=\rho\left(\zeta\int\phi_jT\,dt-\xi\tau\int\phi'_jT\,dt\right),\\"
                 r"z_j&=\int\phi_j e^{-t/\tau}\,dt,\quad"
                 r"\widehat c_0=\frac{\sum_jz_j(a_j-b_j)}{\sum_jz_j^2},\\"
                 r"D_{\mathrm{Hb}}^{\mathrm{demo}}(\tau)&=\frac{\sum_j(a_j-b_j-\widehat c_0z_j)^2}"
                 r"{\sum_j(\int\phi_jR\,dt)^2+\rho^2\sum_j(\int\phi_jT\,dt)^2}.\end{aligned}",
                 "把曲线乘光滑权函数后积分；分部积分把导数移到已知权函数上，避免直接微分含噪 Hb。共用 50 个积分窗，总宽分别为 8、16 秒，中心间隔 2 秒；每个权函数的积分为 1。重叠积分窗不是独立样本。每个候选 τ 都重新估计 c0。"),
        paragraph("图三先固定真实 τ=2 秒，以便把扰动的作用单独显示；参数范围内能否减小残差，要看后面的剖面。共同缩放 O、R 会同时缩放分子与分母，故本教学量不变。单独改变 HbR 则一般会破坏闭合。若分母接近零，输出应是“信息不足”，不能把零信号当成强支持。"),
    ])
    index = int(np.flatnonzero((data["positions"][:, 0] == 24) & (data["positions"][:, 1] == 4))[0])
    hb_bad = data["hb_bad_result"]
    section("量二的图线：额外 HbR 成分造成持续闭合残差", [
        figure("03_hb_closure", "A 对合法 HbR 另加周期 10 秒、幅度为原 HbR 标准差 0.8 倍的正弦；HbO 和 EEG 保持不变。B 重新计算总 Hb。C 每个点是总宽 8 秒的加权积分，右侧包含由全部 50 个积分窗拟合的初态项。D 展开局部残差；总分仍由全部两种宽度的积分窗计算。"),
        table(["中心 24 秒、总宽 8 秒", "数值（模型 Hb 坐标）"], [
            ["左侧 a", f"{hb_bad['left'][index]:.8f}"],
            ["右侧 b + 初态项", f"{hb_bad['right'][index]:.8f}"],
            ["残差：左侧减右侧", f"{hb_bad['residual'][index]:.8f}"],
            ["全部积分窗残差平方和", f"{np.sum(hb_bad['residual']**2):.8f}"],
            ["固定分母", f"{hb_bad['denominator']:.8f}"],
        ], [1.5, 1.4]),
        paragraph(f"合法例 D={data['hb_good_result']['D']:.2e}；扰动后 D={hb_bad['D']:.6f}（{100*hb_bad['D']:.2f}%）。本例无噪，残差来自已知额外成分。实测残差还可能包含噪声、标定、非线性近似和预处理影响。"),
    ])

    section("一个必要的反例：同步不等于违反 Hb 动力学", [
        figure("04_legal_synchrony_and_spectrum", "上排为合法初态松弛：f=v=1，p=q=1+0.04 exp(-t/2)，驱动与信号状态为零。下排为另一组稳态正弦响应；蓝带由 τ=0.5–8 秒生成，是模型包络，不是置信区间或实测频谱。"),
        equation(r"\frac{R(\omega)}{O(\omega)}="
                 r"\frac{\rho(\zeta+i\omega\tau\xi)}{1-\rho\zeta+i\omega\tau(1-\rho\xi)}",
                 "ω=2π×频率；图的频率单位为 Hz。该式适用于处理了初态与窗边界的线性稳态；实测近零 Fourier 系数不应直接相除。"),
        paragraph(f"上排两条 Hb 曲线相关系数严格为 +1。允许初态后 D={data['relaxation_good']['D']:.2e}；错误地把初态项固定为零时 D={data['relaxation_wrong']['D']:.4f}，即 {100*data['relaxation_wrong']['D']:.2f}%。这就是只看相关、或遗漏合法瞬态的误判。"),
        paragraph("稳态低频极限的 HbR/HbO 比为 -0.35332，高频极限为 +0.06742。因而相位必须与幅度一起判断。初态松弛与高频稳态同相是不同机制，不能相互代替。30 秒窗的基本频率间隔约 0.033 Hz，无法独立完成 0.01 Hz 的整周期幅相检验。"),
    ])

    section("量三的计算：允许幅度重标定后，再检查共享时间结构", [
        equation(r"\begin{aligned}X&=\rho O-(1-\rho)R,\\"
                 r"(D^2+\kappa D+\gamma)(\tau D+1)X&=c(e-b_E),\\"
                 r"c&=gQ_0B\beta/a_E.\end{aligned}",
                 "X 为氧合组成对比；κ、γ 控制上游流量动态，τ 控制下游运输。复合增益 c 同时包含 Hb 标度、神经血管增益和 EEG 读出增益，拟合到 c 不代表独立识别出其中任何一个生理参数。本例偏置 b_E=0 已知。"),
        equation(r"\begin{aligned}\ell_j={}&-\tau\int\phi'''_jX\,dt"
                 r"+(1+\kappa\tau)\int\phi''_jX\,dt\\"
                 r"&-(\kappa+\gamma\tau)\int\phi'_jX\,dt+\gamma\int\phi_jX\,dt,\\"
                 r"u_j={}&\int\phi_j e\,dt,\qquad"
                 r"\widehat c=\max\left(0,\frac{\sum_ju_j\ell_j}{\sum_ju_j^2}\right),\\"
                 r"D_{\mathrm{EEG-Hb}}^{\mathrm{demo}}={}&\frac{\sum_j(\ell_j-\widehat c u_j)^2}"
                 r"{\gamma^2\sum_j(\int\phi_jX\,dt)^2}.\end{aligned}",
                 "左侧由 Hb 的四个加权积分得到，右侧由同一积分窗内的 EEG 特征得到。c 用全部 50 条约束统一拟合，不按局部窗口分别调幅。主例固定 τ=2 秒、κ=0.64；后续 τ 剖面会在每个候选点重新拟合 c。若 c 的最优值在零边界，需要单独报告。"),
        paragraph("多核实测模型还需要冻结的 EEG loading 和 Hb mixing 把两模态对应到同一驱动组合。任意一个 EEG 电极与任意一个 Hb 通道不能直接套用单核公式。实际时间滤波、偏置和掩码也必须进入观测设计；完整窗口重建属于条件诊断，不是未来预测。"),
    ])

    section("量三的图线：Hb 完全不变，EEG 错配仍会被发现", [
        figure("05_eeg_hb_closure", "只将 EEG 特征替换为原特征在 t+5 秒的值，即提前 5 秒；Hb 两条曲线完全不变。C 的一个点对应一条局部弱积分约束，正确配对应靠近 45° 线。两种宽度的积分窗重叠，因此不能把 50 个点当成 50 个独立样本。D 只展开总宽 8 秒的残差。"),
        paragraph(f"正确配对的跨模态 D={data['cross_good']['D']:.2e}；提前 5 秒后 D={data['cross_bad']['D']:.6f}，即 {100*data['cross_bad']['D']:.2f}%。尽管错配例重新拟合了复合增益（从 {data['cross_good']['gain']:.6f} 变为 {data['cross_bad']['gain']:.6f}），时间结构仍然对不上。Hb 内部闭合完全保持原值。"),
        paragraph("这个响应模式把问题定位到跨模态时间对应或共享驱动合同。实测中还需用正确对齐、错误配对、EEG 特征替换等控制进一步区分来源；单个跨模态高分不自动证明神经血管机制本身错误。"),
    ])

    section("把一段曲线变成一个数：弱形式的逐项计算", [
        figure("06_weak_integral_arithmetic", "A 展示中心 24 秒的权函数及 Hb 对比曲线。权函数总宽 8 秒，区间外为零；左轴为模型 Hb 坐标，右轴为权函数的每秒单位。B 前四根柱只由 Hb 决定，两个配对条件完全一样；第五项由 EEG 和全窗拟合的复合增益给出；第六根柱就是前五项的代数和。"),
        equation(r"\phi_j(t)=\frac{315}{256w_j}(1-u^2)^4\,\mathbf{1}_{|u|<1},"
                 r"\qquad u=\frac{t-m_j}{w_j}",
                 "m_j 为积分窗中心，w_j 为半宽，取 4 或 8 秒；完整宽度为 8 或 16 秒。函数及其前两阶导数在端点为零，所以三次分部积分没有边界项。导数由这条已知多项式解析得到；观测数据仅参与加权积分。"),
        paragraph("局部残差帮助定位冲突在什么时间尺度、什么位置出现。全窗指标将所有残差平方后相加；不能先把有正有负的残差直接相加，因为它们可能抵消。若要做噪声检验，必须使用这些重叠积分之间的完整协方差，不能把每项独立除以一个 SD 后当作独立检验。"),
    ])

    profiles = summary["parameter_slices"]
    profile_table = table(["图中情形", "实际计算出的交集", "能作出的判断"], [
        ["共同内点", f"{profiles[0]['feasible_min']:.2f}–{profiles[0]['feasible_max']:.2f} 秒，{profiles[0]['feasible_count']} 点", "声明域内存在共同解释；不是参数真值已被实测恢复。"],
        ["跨约束冲突", "0 / 151 个共同点", "在这个切片与容差内没有共同解释；仍不是整个连续参数域的不存在性证明。"],
        ["上界压力", f"{profiles[2]['feasible_min']:.2f}–{profiles[2]['feasible_max']:.2f} 秒，{profiles[2]['feasible_count']} 点", "所有相容点都在声明域最上方 10% 内。不能只报告最优值贴边。"],
        ["激励不足", "0.50–8.00 秒，全部 151 点", "常量观测没有时间常数辨识力；低残差只说明未检出偏离。"],
    ], [1, 1.3, 2])
    section("量四：需要的是共同参数，不是两套各自漂亮的拟合", [
        equation(r"\begin{aligned}\mathcal C_\tau=\{\tau\in[0.5,8]:\;&"
                 r"D_{\mathrm{Hb}}^{\mathrm{demo}}(\tau)\leq 0.01,\\"
                 r"&D_{\mathrm{EEG-Hb}}^{\mathrm{demo}}(\tau)\leq 0.01\}.\end{aligned}",
                 "这里只展示全参数集合在 κ、γ、α、E0 固定时的一维切片。τ 网格步长为 0.05 秒，共 151 点；c0 与 c 分别剖面拟合。1% 是为了展示集合形状而声明的教学能量容差，不是实测 5% 显著性界。"),
        figure("07_common_parameter_sets", "蓝实线为 Hb 内部闭合，紫虚线为跨模态闭合，红点线为教学容差，绿带为共同通过的网格范围。A 两模态由 τ=2 秒共同生成；B 的 Hb 由 τ=5 秒生成，再按 τ=1 秒的跨模态等式反构造 EEG；C 由 τ=7.9 秒生成；D 使用无时间变化的合法稳态。所有纵轴统一；小于 0.00000001% 的数值仅在绘图时显示于底线，原始数值保留。"),
    ])

    section("共同集合的四种结论，以及边界压力的计算", [
        profile_table,
        equation(r"b(\tau)=\min\left(\frac{\tau-0.5}{7.5},\frac{8-\tau}{7.5}\right),"
                 r"\qquad b_{\max}=\max_{\tau\in\mathcal C_\tau}b(\tau)",
                 "b 衡量候选点距离最近边界占整个声明域宽度的比例；b_max 问的是所有相容解中，最靠内部的一点能走多深。上界例的 b_max=0.10，说明找不到离边界超过全域宽度 10% 的相容点。这里的 10% 只是本例几何描述，不是普适生理阈值。"),
        paragraph("如果最优点在边界，但共同集合含有明显内点，数据不迫使解释依赖边界。若集合全部贴边，才需要检查参数域是否过窄、观测尺度是否有偏差、模型是否借极端参数吸收错误。扩大范围属于新敏感性条件，不能把新域的结果覆盖到原域上。"),
        paragraph("常量例在所有 τ 上都满足闭合，因为时间导数没有信息。若先去基线，它会变成零信号，教学归一化分母也为零，此时应输出“证据不足”。非常宽的集合、低 SNR 和缺乏激励都需要与低残差同时报告。"),
        paragraph("真正的共同参数集合需要对同一个参数向量同时检查所有适用约束，并保留参数相关性。把多维集合各自压成单参数区间再求交会丢失这种相关性；区间相交也不一定意味着存在共同多维解。本图只使用一维，故不发生该问题。完整非线性条件、空间合同与观测校准也须在适用时进入正式集合。"),
    ])

    section("量五的计算：从绝对 Hb 坐标重建，再检验完整守恒", [
        equation(r"\begin{aligned}p&=1+\frac{O+R}{gP_0},\qquad q=1+\frac{R}{gQ_0},\\"
                 r"d&=p-v,\qquad\tau\dot d=-(p-d)^{1/\alpha-1}d,\\"
                 r"v&=p-d,\quad L=v^{1/\alpha-1},\quad f=\tau\dot p+Lp,\\"
                 r"I_{\mathrm{Hb,NL}}&=\tau\dot q+Lq-\frac{fE(f)}{E_0},\\"
                 r"E(f)&=1-(1-E_0)^{1/f}.\end{aligned}",
                 "先只用 O、R 恢复 p、q；给定 d0 后积分得到 d，进而得到体积、流量与脱氧 Hb 流入。最后检查脱氧 Hb 变化加流出是否等于流入。计算残差时没有使用生成器中的隐藏体积和流量。完整轨迹必须满足 f、v、p、q 为正，以及 Q0q≤P0p；非法候选不得裁剪成合法数值。"),
        equation(r"D_{\mathrm{Hb,NL}}^{\mathrm{demo}}="
                 r"\min_{d_0\in\{-0.12,-0.115,\ldots,0.12\}}"
                 r"\frac{\operatorname{mean}_{t\in[1,59]} I_{\mathrm{Hb,NL}}(t)^2}{S_{\mathrm{NL}}^2}",
                 f"本例固定 τ=2 秒、α=E0=0.32、g=1，扫描 49 个初态差。固定分母 S_NL²={data['nl_scale_squared']:.10f}，等于合法大响应中“归一化脱氧流入减基线 1”的均方；三个例共用该值。该分母来自教学参照，不是噪声方差，亦不能与线性分数直接比大小。"),
        paragraph("主图的小响应与大响应使用同一双脉冲形状，驱动峰值系数从 0.012 改为 0.22；所有初始物理状态保持静息。额外氧合扰动在 HbR 中加入 Q0×0.04×sin(2πt/10) 再乘以 sin(πt/60) 的四次方，在 HbO 中减去同一量，因此总 Hb 保持不变。"),
        paragraph("这一层对坐标要求最强。本例知道绝对基线与共同增益，使用密集无噪合成曲线的三次样条导数。去基线、高通或空间混合后的实测 Hb 通常不能这样直接反演；缺少绝对坐标时应标记“不适用／待校准”，而不是强行补回状态。本图的条件结果也没有排除其他 τ 或标定参数下的解释。"),
    ])

    result = data["nl_best"][2]
    at = int(np.argmin(abs(data["t"]-34)))
    section("量五的图线：流入与变化加流出在哪里对不上", [
        figure("08_nonlinear_reconstruction", "A 对比合法大响应与只改变氧合组成的扰动。B 从扰动观测重建 p、q 和 f；本例最优初态差为零，因此体积等于 p，但一般不能作此假设。C 绿色虚线为脱氧 Hb 流入，紫线为脱氧 Hb 变化与流出之和；D 显示两者差值。评分去掉两端各 1 秒，曲线仍保留全长。"),
        table(["34 秒处的计算", "数值"], [
            ["p；q", f"{result['p'][at]:.6f}；{result['q'][at]:.6f}"],
            ["重建的体积 v；流量 f", f"{result['volume'][at]:.6f}；{result['flow'][at]:.6f}"],
            ["脱氧变化 + 流出", f"{result['left'][at]:.6f}"],
            ["脱氧流入", f"{result['right'][at]:.6f}"],
            ["非线性残差", f"{result['residual'][at]:.6f}"],
        ], [1.7, 1.3]),
        paragraph(f"合法大响应的非线性 D={data['nl_best'][1]['D']:.2e}；扰动例为 {result['D']:.6f}，即 {100*result['D']:.2f}%。49 个候选初态全部数值有效；在本例固定参数与候选初态范围内，允许初态变化仍不能把扰动残差降到零。"),
    ])

    section("非线性复核的价值：不要把合法大响应误写成模型失配", [
        figure("09_nonlinearity_vs_mismatch", "A 与 B 分别使用已定义的线性、非线性归一化，纵轴不可直接横向比较。应该比较同一列中的合法小响应、合法大响应与有意扰动。C 显示完整初态差剖面；绿色在零初态差达到数值误差水平，紫色仍保留正残差。"),
        paragraph(f"从小响应到大响应，线性 Hb 闭合 D 从 {100*data['nl_linear'][0]:.5f}% 增至 {100*data['nl_linear'][1]:.5f}%，约为原来的 {data['nl_linear'][1]/data['nl_linear'][0]:.1f} 倍；完整非线性闭合仍接近零。这说明线性残差可以随合法响应幅度增加。大响应本例的 0.67% 仍低于前面用于集合演示的 1% 容差，这里没有把它宣判为异常。"),
        paragraph(f"加入氧合扰动后，线性 D 为 {100*data['nl_linear'][2]:.5f}%，非线性 D 为 {100*data['nl_best'][2]['D']:.2f}%。相对于合法大响应，非线性复核能明确区分这个人工扰动。这里只验证了一个已知控制的响应，不代表已经证明对所有可能失配都有特异性。"),
        paragraph(f"积分步长减半的核验中，完整前向观测的最大绝对变化为 {data['nonlinear_refinement_error']:.2e} 个模型坐标单位。非线性初态剖面中的非法点和求解失败数均为零。记录这些数值状态，才能避免把求解器误差当成科学冲突。"),
    ])

    noise = summary["space_noise"]
    section("残差大到什么程度才有意义：加入噪声与检验力", [
        figure("10_noise_and_testability", "每次保留同一合法 3 通道波形，再加入独立高斯噪声；A 共 500 个独立噪声实现，不是 500 名被试。噪声标准差为无噪整块观测 RMS 的 15%，两个色团与所有通道相同。B 合法柱为这 500 次均值，额外维度柱为另一次同级噪声实现。"),
        equation(r"Q_{\mathrm{space}}=\frac{\|Y(I-MM^+)\|_F^2}{\sigma^2},"
                 r"\qquad \nu=2N_t(J-K)=964,\qquad"
                 r"\mathbb E[Q_{\mathrm{space}}/\nu]=1",
                 "本例时间点数 N_t=241、通道数 J=3、核数 K=1。这条期望关系只用于已知、独立、同方差的高斯噪声，且 M 固定、没有滤波或缺失。真实时间处理、相关噪声、参数选择与秩变化会改变残差分布，必须连同实际流程重新校准。"),
        paragraph(f"即使模型完全合法，空间残差的经验 95% 分位也达到 {100*noise['null_D_95_percentile']:.3f}%。白化后的 Q/ν 均值为 {noise['null_mean_Q_over_df']:.4f}，符合期望 1；额外维度加同级噪声后 D={100*noise['perturbed_D']:.3f}%，Q/ν={noise['perturbed_Q_over_df']:.3f}。噪声造成的偏离与结构偏离应在同一流程下比较。"),
        paragraph("前面用于讲解共同集合的 1% 不是这里估计的空间阈值，也不能跨指标通用。正式动力学检验要考虑各弱积分之间的协方差，或使用包含掩码、时间变换和允许初态的离散设计，在合成零分布中校准参数选择后的统计量。训练 SD 只平衡尺度，不能自动替代噪声 SD。"),
    ])

    section("全面实测后，怎样把五个结果连成结论", [
        table(["观察到的组合", "可支持的解释", "应进一步定位什么"], [
            ["固定空间高，自由秩低", "固定方向不够合适，源数未必不足。", "空间 loading、几何、通道增益。"],
            ["固定空间和自由秩都高", "当前 K 难以容纳观测；仍包含噪声影响。", "源数、额外成分、空间噪声结构。"],
            ["空间低，Hb 内部高", "问题已经出现在 Hb 层，不需要先归咎 EEG。", "Hb 标定、时间处理、初态和完整非线性。"],
            ["Hb 内部低，跨模态高", "Hb 可以自洽，但与给定 EEG 映射和时序不相容。", "时间对齐、EEG 特征、配对和共享驱动。"],
            ["两项各自低，共同集合空", "在声明搜索与容差内，两套约束没有共同解释。", "同一参数向量的联合可行性、搜索覆盖及标定敏感性。"],
            ["线性高，完整非线性低", "线性近似可解释至少部分异常。", "大响应幅度、可恢复绝对坐标、适用时间尺度。"],
            ["全部低，集合很宽", "当前分辨率未检出偏离，但参数信息不足。", "激励、SNR、观测时长和参数敏感度。"],
            ["全部低，集合有稳定内点", "声明模型与观测在当前检验力下相容。", "独立数据重复性与任务外验证；不能直接认定生理真值。"],
        ], [1.35, 2, 2]),
        paragraph("相容是相对当前模型、观测合同、噪声、参数域和检验力的判断。任何一项都不能单独完成成分归属、证明生理因果，或证明潜变量准确。空间投影下界具有明确数学保证；有限网格搜不到解与局部优化残差较高，则必须保留数值搜索层面的限定。"),
    ])

    section("从教学积分迁移到实测：让同一个观测设计处理边界", [
        equation(r"\begin{aligned}y&=A_\theta\eta+o_\theta+\epsilon,\\"
                 r"\widetilde y&=C_\epsilon^{-1/2}(y-o_\theta),\quad "
                 r"B_\theta=C_\epsilon^{-1/2}A_\theta,\\"
                 r"Q_\theta&=\|(I-B_\theta B_\theta^+)\widetilde y\|^2.\end{aligned}",
                 "y 是实际可见观测，η 收集允许的驱动与初态，o 是已知观测偏置。A 包含模型传播、空间读出、时间处理与可见行选择；C 是该支持上的噪声协方差。先白化，再投影到模型输出子空间，最后计算垂直残差。若协方差奇异，只能在有定义的支持上处理，不能机械求逆。"),
        table(["需要检验什么", "同一设计怎样使用"], [
            ["固定空间容量", "允许潜在 Hb 时间轨迹完全自由，只保留冻结空间读出与观测算子；其残差给出空间合同下界。"],
            ["Hb 内部关系", "在线性设计中只选择可见 Hb 行，保留足够自由的驱动和允许初态。"],
            ["EEG–Hb 共同解释", "加入对应 EEG 特征行与冻结映射，并在同一候选参数向量下计算；与 Hb-only 结果并列报告。"],
            ["噪声与参数选择", "在合法生成器上经过同一采样、预处理、缺失与参数搜索流程，校准最终选择后统计量；过程噪声是否属于零假设也必须预先声明。"],
        ], [1.2, 3.2]),
        paragraph("现有 build_shared_driver_design 已包含每采样驱动和五个初始血管坐标；fit_waveform_subspace 可用于允许轨迹空间的投影筛查。其现有 SD 加权不是自动成立的噪声白化似然，正式噪声检验仍需独立校准。把时间基设得过窄也会制造表示残差，不应误归为生理冲突。"),
        paragraph("这个离散计算与图解中的弱形式不是同一个数值分数：弱形式便于解释局部冲突，离散设计便于准确纳入实际处理边界。尤其要区分：自由空间轨迹的残差是完整模型下界；一般线性化动态设计的残差并不自动成为完整非线性模型下界。"),
    ])

    section("下一轮实测的输出应是什么样，而不是先给一个总通过率", [
        table(["计算层级", "建议保留的输出与分母"], [
            ["观测合同", "数据集、被试、会话、任务、通道或空间核心、时间支持、预处理、掩码、单位与冻结映射。K2 先核对跨模态驱动组合。"],
            ["局部窗口", "空间原始残差、自由秩残差、Hb 与跨模态弱残差随时间的曲线、信号能量、适用状态和数值失败。不能只保留成功窗口。"],
            ["会话或连续长段", "共同参数集合、宽度、边界距离、校准敏感性。低频检验使用足够长的有效观测支持；不因采样率高就声称频率分辨率高。"],
            ["控制与噪声", "合法同相初态、共同尺度变化、相对色团标定、错配、额外成分、非线性幅度及噪声水平；每项先验证零分布与检验力，再冻结判定规则。"],
            ["被试统计", "先按声明方式汇总同被试的窗口与会话，再跨被试汇总。报告计划数、可用数、失败数、证据不足数与各自分母；重叠窗口不是独立被试。"],
            ["群体比较", "同一被试、同一条件的配对变化；给出中位数/P90、被试层区间、数据集差异。若做显著性，预先声明重采样单位、参数冻结与多重比较处理。"],
        ], [1, 3.8]),
        paragraph("推荐按已核对的空间、Hb 内部、跨模态共同参数三个入口开始；非线性层只应用于满足绝对坐标与标定要求的记录。每个单位分别标注“超出已校准容差”“当前分辨率未检出偏离”“证据不足”“不适用”“数值失败”。这些状态不能合并成一个好坏二分类。"),
        paragraph("本次交付是计算与解释图解，尚未开展全面实测统计。后续报告中的偏离率、被试区间与参数分布应来自正式保留证据；本图解的 25%、20%、30.64% 等数值都是已知构造的教学例，不是当前数据集结论。"),
    ])

    return {"schema": "source_diagnostics_report_v1", "title": "五个模型相容性指标：计算与图解",
            "subtitle": "静态合成图解｜空间投影、Hb 闭合、跨模态闭合、共同参数与完整非线性",
            "date": "2026-10-23",
            "provenance": ["全部为可复算合成示例；无实测数据访问。数值来源为本次导出根目录的 summary.json 与 arrays.npz；报告是这些数值的解释。"],
            "summary": ["读图时依次问：输入是什么、约束两侧怎样计算、残差怎么汇总、参数是否共同可行，以及这个结果还不能说明什么。",
                        "图解提供 10 幅图及逐步算例。共同参数与噪声阈值明确标为教学条件；实测实验需要冻结自己的观测、噪声与统计合同。"],
            "sections": sections,
            "sources": [
                "输入论述：用户提供的五项模型相容性检验文本；原文以 2026-10-21 提交 81c3d304 为依据。本次重新核对当前本地实现。",
                "当前动力学与观测 owner：src/inference/t3a_balloon_robust_ssm.py；前向复用 src/inference/shared_driver_reconstruction.py:nonlinear_driver_forward。线性生成直接使用核心静息 Jacobian。",
                "已核对历史公开核心：https://github.com/Kevin65536/physiological-semantic-tokenizer/blob/81c3d3045c001b28a75e2789921e4f25647595a9/src/inference/shared_driver_rk4.py。当前本地源码快照与 Git 身份保存在本次来源记录。",
                "空间实验参数与窗口合同：experiments/configs/physiology_semantic_tokenizer/shared_driver_spatial_v1.yaml。候选域来源：shared_driver_parameter_stability_v1.yaml。本例一维切片不是该实测实验的重新评价。",
                "计算与图线 owner：experiments/scripts/illustrate_model_compatibility.py；保留 source_snapshot。数值 summary.json 与 arrays.npz；PDF/HTML 复用 render_source_diagnostics_report.py。",
                "实际实验的观测与数据规则仍由 docs/DATA_CONTRACT.md 和相应 owning protocol 决定；本图解不变更研究状态，也不建立新的实测授权。",
            ]}


def write_arrays(data, output):
    arrays = {}

    def collect(prefix, value):
        if isinstance(value, np.ndarray):
            arrays[prefix] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(prefix + "__" + key, item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(prefix + f"__{index}", item)

    for name, value in data.items():
        # Quadrature is exactly reconstructible from time and the source; do
        # not duplicate the largest deterministic design in retained arrays.
        if name != "bank":
            collect(name, value)
    np.savez_compressed(output / "arrays.npz", **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh versioned directory below the owning run root")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    root = ROOT / "experiments/runs/physiology_semantic_tokenizer/model_compatibility_guide"
    if not output.is_relative_to(root) or output == root:
        parser.error(f"--output must name a run below {root}")
    if output.exists():
        raise FileExistsError(f"use a fresh export; retained artifacts are not overwritten: {output}")
    data = prepare_examples()
    summary = numerical_summary(data)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_arrays(data, output)
    figures = draw_figures(data, output)
    content = report_content(data, figures, summary)
    content_path = output / "report_content.json"
    content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    sources = [Path(__file__).resolve(), ROOT / "src/inference/t3a_balloon_robust_ssm.py",
               ROOT / "src/inference/shared_driver_reconstruction.py",
               ROOT / "experiments/scripts/render_source_diagnostics_report.py"]
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    provenance = {"kind": "synthetic_visual_guide", "command": sys.argv,
                  "working_directory": str(ROOT), "python_executable": sys.executable,
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
                  "sources": [str(source.relative_to(ROOT)) for source in sources],
                  "data_access": "no dataset/run reads; generated synthetic arrays only",
                  "resources": "bounded small synthetic illustration; no training or measured-data sweep"}
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    from experiments.scripts.render_source_diagnostics_report import render_report
    checks = render_report(content_path, output / "report")
    audit = checks["exports"]["TECHNICAL_REPORT.pdf"]
    print(json.dumps({"output": str(output), "summary": str(output / "summary.json"),
                      "pdf_checks": {key: audit[key] for key in
                                     ("pages", "displayed_images", "image_objects",
                                      "text_outside_page", "full_page_raster_images")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
