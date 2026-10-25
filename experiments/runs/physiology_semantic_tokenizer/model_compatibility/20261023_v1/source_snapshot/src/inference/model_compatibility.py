"""Descriptive model-compatibility distances, with no data or run I/O.

These are projection/closure energies, not calibrated noise statistics or
physiological parameter confidence sets. The caller owns sampling support,
processing operators, physical channel identities and evaluation partitions.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.linalg import svd

from . import t3a_balloon_robust_ssm as core
from .shared_driver_reconstruction import build_shared_driver_design


def _positive_grid(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} must be a nonempty positive finite vector")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate nodes")
    return values


def _basis(matrix, rcond):
    """An orthonormal column basis and the actual numerical rank."""
    u, singular, _ = svd(matrix, full_matrices=False, check_finite=False)
    keep = singular > (rcond * singular[0] if len(singular) else 0.)
    return u[:, keep], int(keep.sum())


def _weak_bank(times, half_widths, center_spacing):
    """Integrate normalized (1-u**2)**4 kernels and their time derivative."""
    delta = np.diff(times)
    weights = np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]
    kernels, derivatives, positions = [], [], []
    for width in half_widths:
        centers = np.arange(times[0] + width, times[-1] - width + 1e-9, center_spacing)
        for center in centers:
            u = (times - center) / width
            inside = abs(u) < 1
            phi, derivative = np.zeros(len(times)), np.zeros(len(times))
            phi[inside] = (1 - u[inside] ** 2) ** 4 / (width * 256 / 315)
            derivative[inside] = -8 * u[inside] * (1 - u[inside] ** 2) ** 3 / (width**2 * 256 / 315)
            # The same discrete normalization multiplies phi and phi'.
            norm = phi @ weights
            if norm > 0:
                kernels.append(phi * weights / norm)
                derivatives.append(derivative * weights / norm)
                positions.append((center, width))
    return (np.asarray(kernels).reshape(-1, len(times)),
            np.asarray(derivatives).reshape(-1, len(times)),
            np.asarray(positions).reshape(-1, 2))


def weak_hb_profiles(hb, *, dt=None, times=None, taus=None, nominal_tau=2.,
                     half_widths=(4., 8.), center_spacing=2., rho=None,
                     zeta=None, xi=None, rcond=1e-10, return_details=False):
    """Batch weak Hb closure, preserving each HbO/HbR pair's relative scale.

    ``hb`` is [T,J,2] (or [T,2], treated as J=1), last axis [HbO,HbR].
    A window/channel pair with any nonfinite value is unavailable in full;
    missing values are never replaced or integrated across. Times are actual
    seconds; no resampling, filtering or differentiation of Hb is performed.

    Let R=HbR, T=HbO+HbR after subtracting their window time means. At tau,
    h=integral(phi R)-tau integral(phi' R)
      -rho[zeta integral(phi T)-xi tau integral(phi' T)].
    A constant and integral(phi exp(-(t-t0)/tau)) are independently profiled
    nuisance coordinates. The denominator, fixed across tau, is
    sum(integral(phi R)**2+rho**2 integral(phi T)**2). It is a descriptive
    dynamic-energy normalization, not an estimate of measurement noise.

    Overlapping kernels do not supply independent observations. The caller
    must split windows at any known acquisition concatenation boundary.
    """
    y = np.asarray(hb, dtype=float)
    if y.ndim == 2:
        y = y[:, None, :]
    if y.ndim != 3 or y.shape[-1] != 2 or y.shape[0] < 3:
        raise ValueError("hb must be [T,J,2] or [T,2], T>=3")
    if times is None:
        if dt is None or not np.isfinite(dt) or dt <= 0:
            raise ValueError("provide positive dt or actual times")
        time = np.arange(len(y), dtype=float) * dt
    else:
        time = np.asarray(times, dtype=float)
        if time.shape != (len(y),) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("times must be a finite strictly increasing [T] vector")
        if dt is not None and (not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(time), dt)):
            raise ValueError("dt and times disagree")
    tau_grid = _positive_grid(np.arange(.5, 8.0001, .25) if taus is None else taus, "taus")
    widths = _positive_grid(half_widths, "half_widths")
    if (not np.isfinite([nominal_tau, center_spacing, rcond]).all()
            or nominal_tau <= 0 or center_spacing <= 0 or not 0 < rcond < 1):
        raise ValueError("invalid nominal_tau, center_spacing or rcond")
    fixed = core.BalloonFixedParameters()
    b = -(1 - fixed.E0) * np.log1p(-fixed.E0) / fixed.E0
    rho = fixed.Q0 / fixed.P0 if rho is None else float(rho)
    xi = 1 - b if xi is None else float(xi)
    zeta = 1 - b / fixed.alpha if zeta is None else float(zeta)
    if not np.isfinite([rho, xi, zeta]).all() or rho <= 0:
        raise ValueError("rho must be positive, xi and zeta finite")
    bank, derivative, positions = _weak_bank(time, widths, center_spacing)
    count = y.shape[1]
    status = np.full(count, "missing_support", dtype=object)
    complete = np.isfinite(y).all(axis=(0, 2))
    profiles = np.full((count, len(tau_grid)), np.nan)
    nominal = np.full(count, np.nan)
    denominator = np.full(count, np.nan)
    nuisance_rank = np.zeros(len(tau_grid), dtype=int)
    # Operate only on complete pairs; constructing a zero-filled input would
    # silently turn a support failure into a successful physiological score.
    indices = np.flatnonzero(complete)
    status[indices] = "insufficient_support"
    if len(indices) and len(bank) > 2:
        values = y[:, indices].copy()
        # Referencing a sample first makes an exactly constant channel exactly
        # zero, including decimal constants whose floating mean rounds off.
        values -= values[:1].copy()
        values -= values.mean(axis=0, keepdims=True)
        r, total = values[:, :, 1], values.sum(axis=2)
        ir, it = bank @ r, bank @ total
        dr, dtotal = derivative @ r, derivative @ total
        energy = np.sum(ir**2 + rho**2 * it**2, axis=0)
        denominator[indices] = energy
        energetic = np.isfinite(energy) & (energy > np.finfo(float).tiny)
        status[indices] = "insufficient_energy"
        status[indices[energetic]] = "ok"

        def evaluate(tau, details=False):
            nuisance = np.column_stack((bank.sum(axis=1), bank @ np.exp(-(time-time[0])/tau)))
            basis, rank = _basis(nuisance, rcond)
            left = ir - tau * dr
            raw_right = rho * (zeta * it - xi * tau * dtotal)
            h = left - raw_right
            residual = h - basis @ (basis.T @ h)
            score = np.full(len(indices), np.nan)
            score[energetic] = np.sum(residual[:, energetic]**2, axis=0) / energy[energetic]
            if details:
                coefficients = np.linalg.lstsq(nuisance, h, rcond=rcond)[0]
                return score, rank, dict(left=left, raw_right=raw_right,
                    right=left-residual, residual=residual, fitted_nuisance=h-residual,
                    nuisance_coefficients=coefficients)
            return score, rank

        for k, tau in enumerate(tau_grid):
            profiles[indices, k], nuisance_rank[k] = evaluate(tau)
        nominal[indices], nominal_rank = evaluate(float(nominal_tau))
    else:
        nominal_rank = 0
    best_d, best_tau = np.full(count, np.nan), np.full(count, np.nan)
    good = status == "ok"
    if good.any():
        best_index = np.argmin(profiles[good], axis=1)
        best_d[good] = profiles[good, best_index]
        best_tau[good] = tau_grid[best_index]
    result = dict(nominal_D=nominal, best_D=best_d, best_tau=best_tau,
                profiles=profiles, taus=tau_grid, status=status,
                denominator=denominator, kernel_count=len(bank), kernel_positions=positions,
                nuisance_rank=nuisance_rank, nominal_nuisance_rank=nominal_rank,
                nominal_tau=float(nominal_tau), rho=rho, xi=xi, zeta=zeta)
    if return_details:
        details = {}
        for name in ("nominal_details", "best_details"):
            details[name] = {key: np.full((2 if key == "nuisance_coefficients" else len(bank), count), np.nan)
                for key in ("left", "raw_right", "right", "residual", "fitted_nuisance", "nuisance_coefficients")}
        if len(indices) and len(bank) > 2:
            _, _, nominal_details = evaluate(float(nominal_tau), details=True)
            for key, value in nominal_details.items():
                details["nominal_details"][key][:, indices] = value
            for tau in np.unique(best_tau[good]):
                _, _, at_tau = evaluate(tau, details=True)
                local = best_tau[indices] == tau
                for key, value in at_tau.items():
                    details["best_details"][key][:, indices[local]] = value[:, local]
        result.update(details)
    return result


def spatial_projection_residual(signal, mixing, *, mask=None, rcond=1e-10):
    """Fixed spatial readout distance and a separately labelled rank bound.

    Rows of signal [N,C] must already share the same temporal support/operator.
    Mixing [C,K] is fixed by the caller; no PCA or new mapping is fitted. The
    separate rank_D is the optimal rank-rank(mixing) lower bound, not the
    distance to the fixed readout. Energies retain the supplied native units.
    """
    y, m = np.asarray(signal, float), np.asarray(mixing, float)
    if y.ndim != 2 or m.ndim != 2 or y.shape[1] != m.shape[0] or not np.isfinite(m).all():
        raise ValueError("signal must be [N,C] and mixing finite [C,K]")
    if not 0 < rcond < 1:
        raise ValueError("rcond must be in (0,1)")
    selected = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != (len(y),):
        raise ValueError("mask must be boolean [N]")
    basis, rank = _basis(m, rcond)
    result = dict(status="insufficient_support", rank=rank, sample_count=int(selected.sum()),
                  energy=np.nan, residual_energy=np.nan, rank_residual_energy=np.nan,
                  D=np.nan, rank_D=np.nan)
    if not selected.any():
        return result
    values = y[selected]
    if not np.isfinite(values).all():
        result["status"] = "missing_support"
        return result
    energy = float(np.sum(values**2))
    residual = values - (values @ basis) @ basis.T
    singular = svd(values, compute_uv=False, check_finite=False)
    residual_energy = float(np.sum(residual**2))
    rank_energy = float(np.sum(singular[rank:]**2))
    result.update(energy=energy, residual_energy=residual_energy, rank_residual_energy=rank_energy)
    if energy <= np.finfo(float).tiny:
        result["status"] = "insufficient_energy"
    else:
        result.update(status="ok", D=residual_energy/energy, rank_D=rank_energy/energy)
    return result


def prepare_linear_panel_profiles(steps, dt, operator, sd, taus, kappas, score_mask=None,
                                  *, parameters=None, rcond=1e-10):
    """Build reusable exact discrete projection bases for one frozen panel.

    All bases include the existing unrestricted five-coordinate initial state
    plus per-modality unknown measurement constants propagated through the
    *same* supplied operator. No regularization, noise fit or new mean filter
    is introduced. The score mask also specifies the observations used to fit;
    unselected target entries cannot affect any score.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 3:
        raise ValueError("steps must be an integer >=3")
    if not np.isfinite(dt) or dt <= 0 or not 0 < rcond < 1:
        raise ValueError("invalid dt or rcond")
    op, scale = np.asarray(operator, float), np.asarray(sd, float)
    if op.shape != (3*steps, 3*steps) or not np.isfinite(op).all():
        raise ValueError("operator must be finite [3T,3T]")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("sd must be positive finite [3]")
    tau_grid, kappa_grid = _positive_grid(taus, "taus"), _positive_grid(kappas, "kappas")
    mask = np.ones((steps, 3), bool) if score_mask is None else np.asarray(score_mask)
    if mask.shape != (steps, 3) or mask.dtype != np.bool_ or not mask[:, 0].any() or not mask[:, 1:].any():
        raise ValueError("score_mask must select EEG and Hb observations in boolean [T,3]")
    p = core.BalloonParameters() if parameters is None else parameters
    p.validate()
    weight = np.broadcast_to(scale, (steps, 3))[mask]
    hb = np.tile([False, True, True], steps)[mask.ravel()]
    constants = op @ np.tile(np.eye(3), (steps, 1))
    constant_basis, constant_rank = _basis(constants[mask.ravel()] / weight[:, None], rcond)
    nodes = []
    for tau in tau_grid:
        for kappa in kappa_grid:
            candidate = replace(p, free=replace(p.free, tau=float(tau), kappa=float(kappa)))
            design = build_shared_driver_design(candidate, steps, dt, processed_mean_operator=op)
            x = np.column_stack((design.observation_design, constants))[mask.ravel()] / weight[:, None]
            joint_basis, joint_rank = _basis(x, rcond)
            hb_basis, hb_rank = _basis(x[hb], rcond)
            nodes.append(dict(tau=float(tau), kappa=float(kappa),
                              joint_basis=joint_basis, hb_basis=hb_basis,
                              joint_rank=joint_rank, hb_rank=hb_rank,
                              offset=design.offset[mask] / weight))
    return dict(steps=int(steps), dt=float(dt), operator=op.copy(), sd=scale.copy(),
                score_mask=mask.copy(), weight=weight, hb_rows=hb, nodes=nodes,
                taus=tau_grid.copy(), kappas=kappa_grid.copy(), rcond=float(rcond), parameters=p,
                constant_basis=constant_basis, constant_rank=constant_rank)


def linear_panel_profiles(target, operator, sd, taus, kappas, score_mask=None, *, dt,
                          parameters=None, prepared=None, rcond=1e-10,
                          nominal_tau=2., nominal_kappa=.64, return_predictions=False):
    """Engineering SD-normalized projection NMSE on the same parameter nodes.

    Target is [T,3], ordered [EEG,HbO,HbR]. The joint residual includes both Hb
    and cross-modal restrictions; eeg_nmse is its separate EEG component.
    cross_increment_nmse=(joint_SSE-Hb_only_SSE)/N_EEG also includes the Hb
    compromise needed to satisfy EEG. It is neither independent of joint/Hb
    scores nor a calibrated EEG likelihood. NMSE is SSE/selected entry count;
    sqrt(NMSE) is an engineering normalized RMS error, not a noise z score.
    zero_prediction_nmse scores the literal zero prediction in this processed
    target coordinate. dynamic_nmse removes only the actual operator's three
    constant columns. A numerically constant/zero target can have zero fit
    distance while interpretation='insufficient_excitation'; this cannot
    establish strong support for the model or identify its parameters.

    ``prepared`` permits reuse across targets with the *identical* operator,
    SD, grids and mask. Common feasible nodes must be intersected node by node
    by the caller; independent profile minima do not define a common theta.
    """
    y = np.asarray(target, float)
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("target must be [T,3]")
    if prepared is None:
        prepared = prepare_linear_panel_profiles(len(y), dt, operator, sd, taus, kappas,
            score_mask, parameters=parameters, rcond=rcond)
    else:
        mask = np.ones_like(y, bool) if score_mask is None else np.asarray(score_mask)
        checks = ((prepared["steps"] == len(y)), (prepared["dt"] == dt),
                  np.array_equal(prepared["operator"], operator), np.array_equal(prepared["sd"], sd),
                  np.array_equal(prepared["taus"], taus), np.array_equal(prepared["kappas"], kappas),
                  np.array_equal(prepared["score_mask"], mask), prepared["rcond"] == rcond,
                  prepared["parameters"] == (core.BalloonParameters() if parameters is None else parameters))
        if not all(checks):
            raise ValueError("prepared bases do not match the requested panel")
    mask, hb = prepared["score_mask"], prepared["hb_rows"]
    if not np.isfinite(y[mask]).all():
        return dict(status="missing_support", profiles=[], score_count=int(mask.sum()))
    values = y[mask] / prepared["weight"]
    zero_energy = float(values @ values)
    dynamic = values - prepared["constant_basis"] @ (prepared["constant_basis"].T @ values)
    dynamic_energy = float(dynamic @ dynamic)
    dynamic_support = dynamic_energy > max(np.finfo(float).tiny,
        64*np.finfo(float).eps**2*zero_energy)
    coordinates = np.broadcast_to(np.arange(3), y.shape)[mask]
    rows = []
    for node in prepared["nodes"]:
        rhs = values - node["offset"]
        joint = rhs - node["joint_basis"] @ (node["joint_basis"].T @ rhs)
        hbrhs = rhs[hb]
        hb_residual = hbrhs - node["hb_basis"] @ (node["hb_basis"].T @ hbrhs)
        hb_sse, joint_sse = float(hb_residual @ hb_residual), float(joint @ joint)
        row = dict(tau=node["tau"], kappa=node["kappa"],
                         hb_nmse=hb_sse/int(hb.sum()), joint_nmse=joint_sse/len(rhs),
                         joint_hb_nmse=float(joint[hb] @ joint[hb])/int(hb.sum()),
                         eeg_nmse=float(joint[~hb] @ joint[~hb])/int((~hb).sum()),
                         cross_increment_nmse=max(0., joint_sse-hb_sse)/int((~hb).sum()),
                         hb_sse=hb_sse, joint_sse=joint_sse,
                         hb_rank=node["hb_rank"], joint_rank=node["joint_rank"])
        for coordinate, name in ((1, "hbo"), (2, "hbr")):
            selected = coordinates == coordinate
            hselected = coordinates[hb] == coordinate
            row[f"joint_{name}_nmse"] = float(np.mean(joint[selected]**2)) if selected.any() else np.nan
            row[f"hb_{name}_nmse"] = float(np.mean(hb_residual[hselected]**2)) if hselected.any() else np.nan
        row["joint_eeg_nmse"] = row["eeg_nmse"]
        row["cross_increment_nrmse"] = float(np.sqrt(row["cross_increment_nmse"]))
        rows.append(row)
    result = dict(status="ok", profiles=rows, score_count=int(mask.sum()),
                hb_score_count=int(hb.sum()), eeg_score_count=int((~hb).sum()),
                zero_prediction_nmse=zero_energy/len(values),
                zero_prediction_energy=zero_energy,
                dynamic_nmse=dynamic_energy/len(values), dynamic_energy=dynamic_energy,
                dynamic_support=bool(dynamic_support),
                interpretation="descriptive_projection" if dynamic_support else "insufficient_excitation",
                constant_rank=prepared["constant_rank"],
                best_hb=min(rows, key=lambda row: row["hb_nmse"]).copy(),
                best_joint=min(rows, key=lambda row: row["joint_nmse"]).copy())
    result["nominal"] = next((row.copy() for row in rows
        if row["tau"] == nominal_tau and row["kappa"] == nominal_kappa), None)
    if return_predictions:
        predictions, cached = {}, {}
        for label in ("nominal", "best_hb", "best_joint"):
            row = result[label]
            if row is None:
                predictions[label] = None
                continue
            theta = row["tau"], row["kappa"]
            if theta not in cached:
                p = prepared["parameters"]
                candidate = replace(p, free=replace(p.free, tau=theta[0], kappa=theta[1]))
                design = build_shared_driver_design(candidate, len(y), dt,
                    processed_mean_operator=prepared["operator"])
                constants = prepared["operator"] @ np.tile(np.eye(3), (len(y), 1))
                full_x = np.column_stack((design.observation_design, constants))
                x = full_x[mask.ravel()] / prepared["weight"][:, None]
                rhs = (y[mask]-design.offset[mask])/prepared["weight"]
                joint_coef = np.linalg.lstsq(x, rhs, rcond=rcond)[0]
                hb_coef = np.linalg.lstsq(x[hb], rhs[hb], rcond=rcond)[0]
                joint_prediction = (full_x@joint_coef).reshape(y.shape)+design.offset
                hb_prediction = (full_x@hb_coef).reshape(y.shape)+design.offset
                hb_prediction[:, 0] = np.nan
                cached[theta] = dict(tau=theta[0], kappa=theta[1],
                    joint_prediction=joint_prediction, hb_prediction=hb_prediction,
                    joint_coefficients=joint_coef, hb_coefficients=hb_coef)
            predictions[label] = cached[theta]
        result["predictions"] = predictions
    return result
