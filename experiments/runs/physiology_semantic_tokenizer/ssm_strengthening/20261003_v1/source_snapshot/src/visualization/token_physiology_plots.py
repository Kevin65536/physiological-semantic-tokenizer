"""Truthful, reusable figures for token physiology analysis.

The functions in this module intentionally do not set a global Matplotlib
backend or style.  Callers (and tests) choose the backend, while every figure
uses Matplotlib's object-oriented API and constrained layout.

Missing values and estimates with insufficient support are not interchangeable:
missing cells use a dark grey ``x`` marker, while insufficient-support cells
use a light grey hatch.  Diverging physiological profiles always share one
symmetric colour scale centred on zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


MISSING_COLOR = "#6F6F6F"
INSUFFICIENT_SUPPORT_COLOR = "#D9D9D9"
SUPPORTED_COLOR = "#0072B2"
REFERENCE_COLOR = "#333333"
_SUPPORTED_FORMATS = ("png", "pdf", "svg")


@dataclass(frozen=True)
class FigureArtifacts:
    """Paths created by :func:`save_figure_atomic`."""

    figure_paths: tuple[Path, ...]


def _as_1d(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
    length: int | None = None,
    dtype: Any | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has length {len(array)}; expected {length}")
    return array


def _token_ids(token_ids: Sequence[Any] | np.ndarray, length: int) -> np.ndarray:
    ids = _as_1d(token_ids, name="token_ids", length=length)
    labels = [str(value) for value in ids.tolist()]
    if len(set(labels)) != len(labels):
        raise ValueError("token_ids must be unique after string conversion")
    return ids


def _support_flags(
    support_flags: Sequence[bool] | np.ndarray | None,
    length: int,
) -> np.ndarray:
    if support_flags is None:
        return np.ones(length, dtype=bool)
    flags = _as_1d(
        support_flags,
        name="support_flags",
        length=length,
        dtype=bool,
    )
    return flags


def _figure_size(
    rows: int,
    columns: int,
    figsize: tuple[float, float] | None,
    *,
    min_width: float = 6.0,
    min_height: float = 3.8,
) -> tuple[float, float]:
    if figsize is not None:
        if len(figsize) != 2 or min(figsize) <= 0:
            raise ValueError("figsize must contain two positive values")
        return float(figsize[0]), float(figsize[1])
    width = max(min_width, min(18.0, 1.0 + 0.72 * columns))
    height = max(min_height, min(24.0, 1.6 + 0.30 * rows))
    return width, height


def _set_token_ticks(
    ax: Axes,
    positions: np.ndarray,
    token_ids: np.ndarray,
    *,
    axis: str,
    max_labels: int = 40,
) -> None:
    """Label a deterministic subset without implying that omitted tokens vanish."""

    count = len(positions)
    if count <= max_labels:
        selected = np.arange(count)
    else:
        selected = np.unique(np.linspace(0, count - 1, max_labels, dtype=int))
    labels = [str(token_ids[index]) for index in selected]
    ticks = positions[selected]
    if axis == "x":
        ax.set_xticks(ticks, labels)
        rotation = 90 if len(selected) > 12 else 0
        for label in ax.get_xticklabels():
            label.set_rotation(rotation)
            label.set_ha("center")
    elif axis == "y":
        ax.set_yticks(ticks, labels)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def _symmetric_limit(values: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        limit = float(requested)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("value_limit must be a finite positive number")
        return limit
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.max(np.abs(finite)))
    return limit if limit > 0 else 1.0


def _feature_label(name: str, units: str | None) -> str:
    return f"{name} ({units})" if units else name


def plot_token_feature_heatmap(
    profile_matrix: Sequence[Sequence[float]] | np.ndarray,
    token_ids: Sequence[Any] | np.ndarray,
    feature_names: Sequence[str] | np.ndarray,
    support_flags: Sequence[bool] | np.ndarray | None = None,
    *,
    value_label: str = "Standardized feature enrichment",
    value_limit: float | None = None,
    title: str = "Token physiological feature profiles",
    cmap: str = "RdBu_r",
    missing_color: str = MISSING_COLOR,
    insufficient_color: str = INSUFFICIENT_SUPPORT_COLOR,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot token-by-feature profiles on one zero-centred colour scale.

    Parameters
    ----------
    profile_matrix:
        ``[n_tokens, n_features]`` values.  NaN/inf values are shown as
        explicitly missing and are never replaced with zero.
    support_flags:
        One Boolean per token.  Unsupported rows are grey and hatched even when
        a numerical estimate is present.
    value_label:
        Colorbar label, including the transformation and units when applicable.
    value_limit:
        Optional positive absolute limit.  Both colour limits are always
        ``(-value_limit, +value_limit)`` so panels/cells are comparable.
    """

    matrix = np.asarray(profile_matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(
            "profile_matrix must be a non-empty two-dimensional array; "
            f"got shape {matrix.shape}"
        )
    n_tokens, n_features = matrix.shape
    ids = _token_ids(token_ids, n_tokens)
    names = _as_1d(feature_names, name="feature_names", length=n_features)
    flags = _support_flags(support_flags, n_tokens)

    supported_values = matrix[flags]
    limit = _symmetric_limit(supported_values, value_limit)
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    colour_map = mpl.colormaps.get_cmap(cmap).with_extremes(bad=missing_color)

    display = matrix.copy()
    display[~flags, :] = np.nan
    display = np.ma.masked_invalid(display)

    with mpl.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(
            figsize=_figure_size(n_tokens, n_features, figsize),
            layout="constrained",
        )
        image = ax.imshow(
            display,
            aspect="auto",
            interpolation="nearest",
            cmap=colour_map,
            norm=norm,
        )
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(value_label)

        positions = np.arange(n_tokens)
        _set_token_ticks(ax, positions, ids, axis="y")
        ax.set_xticks(np.arange(n_features), [str(name) for name in names])
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        ax.set(xlabel="Feature", ylabel="Token ID", title=title)

        # Hatch is a non-colour cue for insufficient support.
        for row in np.flatnonzero(~flags):
            for column in range(n_features):
                ax.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1.0,
                        1.0,
                        facecolor=insufficient_color,
                        edgecolor=MISSING_COLOR,
                        linewidth=0.3,
                        hatch="///",
                        zorder=3,
                    )
                )

        # A printed cross makes missingness distinguishable without colour.
        missing = ~np.isfinite(matrix) & flags[:, None]
        for row, column in np.argwhere(missing):
            ax.text(
                column,
                row,
                "\u00d7",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=4,
            )

        legend_handles: list[Any] = []
        if np.any(missing):
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    linestyle="none",
                    color=missing_color,
                    markeredgewidth=1.5,
                    label="Missing value (\u00d7)",
                )
            )
        if np.any(~flags):
            legend_handles.append(
                Patch(
                    facecolor=insufficient_color,
                    edgecolor=MISSING_COLOR,
                    hatch="///",
                    label="Insufficient token support",
                )
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                frameon=False,
            )

    return fig, ax


def plot_token_support(
    token_ids: Sequence[Any] | np.ndarray,
    support_values: Sequence[float] | np.ndarray,
    support_flags: Sequence[bool] | np.ndarray | None = None,
    *,
    minimum_support: float | None = None,
    support_label: str = "Assigned patch count",
    title: str = "Token support",
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot token support with low-support and missing tokens made explicit."""

    values = _as_1d(support_values, name="support_values", dtype=float)
    ids = _token_ids(token_ids, len(values))
    if np.any(np.isfinite(values) & (values < 0)):
        raise ValueError("support_values must be non-negative")
    if minimum_support is not None:
        minimum_support = float(minimum_support)
        if not np.isfinite(minimum_support) or minimum_support < 0:
            raise ValueError("minimum_support must be finite and non-negative")

    if support_flags is None and minimum_support is not None:
        flags = np.isfinite(values) & (values >= minimum_support)
    else:
        flags = _support_flags(support_flags, len(values))

    missing = ~np.isfinite(values)
    positions = np.arange(len(values))
    heights = values.copy()
    heights[missing] = 0.0  # a marker below makes clear this is not observed zero
    colours = np.where(flags, SUPPORTED_COLOR, INSUFFICIENT_SUPPORT_COLOR)
    colours = np.where(missing, MISSING_COLOR, colours)

    with mpl.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(
            figsize=_figure_size(1, len(values), figsize, min_height=4.2),
            layout="constrained",
        )
        bars = ax.bar(
            positions,
            heights,
            color=colours,
            edgecolor=REFERENCE_COLOR,
            linewidth=0.45,
        )
        for index, bar in enumerate(bars):
            if not flags[index] and not missing[index]:
                bar.set_hatch("///")
            if missing[index]:
                bar.set_hatch("xx")

        finite = values[np.isfinite(values)]
        upper = max(float(np.max(finite)) if finite.size else 1.0, 1.0)
        if minimum_support is not None:
            ax.axhline(
                minimum_support,
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=1.0,
                label=f"Minimum support = {minimum_support:g}",
            )
            upper = max(upper, minimum_support)
        for index in np.flatnonzero(missing):
            ax.plot(
                positions[index],
                0.025 * upper,
                marker="x",
                color=MISSING_COLOR,
                markeredgewidth=1.8,
                linestyle="none",
                zorder=4,
            )

        _set_token_ticks(ax, positions, ids, axis="x")
        ax.set(
            xlabel="Token ID",
            ylabel=support_label,
            title=title,
            ylim=(0.0, upper * 1.10),
        )
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

        handles: list[Any] = [
            Patch(
                facecolor=SUPPORTED_COLOR,
                edgecolor=REFERENCE_COLOR,
                label="Sufficient support",
            )
        ]
        if np.any(~flags & ~missing):
            handles.append(
                Patch(
                    facecolor=INSUFFICIENT_SUPPORT_COLOR,
                    edgecolor=REFERENCE_COLOR,
                    hatch="///",
                    label="Insufficient support",
                )
            )
        if np.any(missing):
            handles.append(
                Patch(
                    facecolor=MISSING_COLOR,
                    edgecolor=REFERENCE_COLOR,
                    hatch="xx",
                    label="Missing support value",
                )
            )
        if minimum_support is not None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=REFERENCE_COLOR,
                    linestyle="--",
                    label=f"Minimum support = {minimum_support:g}",
                )
            )
        ax.legend(handles=handles, frameon=False)

    return fig, ax


def plot_token_feature_profile_ci(
    token_ids: Sequence[Any] | np.ndarray,
    estimates: Sequence[float] | np.ndarray,
    interval_lower: Sequence[float] | np.ndarray,
    interval_upper: Sequence[float] | np.ndarray,
    support_flags: Sequence[bool] | np.ndarray | None = None,
    *,
    feature_name: str,
    units: str | None = None,
    estimator_label: str = "Estimate",
    interval_label: str = "95% subject-bootstrap confidence interval",
    reference_value: float | None = 0.0,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Plot token-level estimates and explicitly named confidence intervals.

    Unsupported estimates remain visible as grey ``x`` marks but their
    intervals are suppressed to avoid presenting unstable precision. Missing
    estimates are counted in an annotation rather than placed at an invented
    y-coordinate.
    """

    values = _as_1d(estimates, name="estimates", dtype=float)
    ids = _token_ids(token_ids, len(values))
    lower = _as_1d(
        interval_lower,
        name="interval_lower",
        length=len(values),
        dtype=float,
    )
    upper = _as_1d(
        interval_upper,
        name="interval_upper",
        length=len(values),
        dtype=float,
    )
    flags = _support_flags(support_flags, len(values))
    if not interval_label.strip():
        raise ValueError("interval_label must explicitly name the interval")

    complete = np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper)
    invalid_order = complete & (
        (lower > values) | (values > upper) | (lower > upper)
    )
    if np.any(invalid_order):
        bad = np.flatnonzero(invalid_order).tolist()
        raise ValueError(
            "Intervals must satisfy lower <= estimate <= upper; "
            f"invalid token positions: {bad}"
        )

    positions = np.arange(len(values))
    supported = complete & flags
    insufficient = complete & ~flags
    missing = ~complete
    ylabel = _feature_label(feature_name, units)

    with mpl.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(
            figsize=_figure_size(1, len(values), figsize, min_height=4.3),
            layout="constrained",
        )
        if np.any(supported):
            yerr = np.vstack(
                (
                    values[supported] - lower[supported],
                    upper[supported] - values[supported],
                )
            )
            ax.errorbar(
                positions[supported],
                values[supported],
                yerr=yerr,
                fmt="o",
                linestyle="none",
                color=SUPPORTED_COLOR,
                ecolor=REFERENCE_COLOR,
                elinewidth=1.0,
                capsize=2.5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=f"{estimator_label}; {interval_label}",
                zorder=3,
            )
        if np.any(insufficient):
            ax.scatter(
                positions[insufficient],
                values[insufficient],
                marker="x",
                s=42,
                linewidths=1.5,
                color=MISSING_COLOR,
                label="Estimate shown; interval suppressed (insufficient support)",
                zorder=4,
            )
        if reference_value is not None:
            reference = float(reference_value)
            if not np.isfinite(reference):
                raise ValueError("reference_value must be finite or None")
            ax.axhline(
                reference,
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=0.9,
                label=f"Reference = {reference:g}",
                zorder=1,
            )

        _set_token_ticks(ax, positions, ids, axis="x")
        ax.set(
            xlabel="Token ID",
            ylabel=ylabel,
            title=title or f"{feature_name} by token",
        )
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        if np.any(missing):
            ax.text(
                0.01,
                0.99,
                f"Missing estimate or interval: {int(np.sum(missing))} token(s)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=MISSING_COLOR,
                fontsize=9,
            )
        handles, labels = ax.get_legend_handles_labels()
        if np.any(missing):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    linestyle="none",
                    color=MISSING_COLOR,
                    label="Missing estimate/interval (not positioned)",
                )
            )
            labels.append("Missing estimate/interval (not positioned)")
        if handles:
            ax.legend(handles=handles, labels=labels, frameon=False)

    return fig, ax


def _project_embedding(embedding: np.ndarray) -> tuple[np.ndarray, tuple[str, str]]:
    if embedding.ndim != 2 or embedding.shape[1] < 2 or embedding.shape[0] == 0:
        raise ValueError(
            "embedding must have shape [n_tokens, n_dimensions] with at least "
            f"two dimensions; got {embedding.shape}"
        )
    finite_rows = np.all(np.isfinite(embedding), axis=1)
    if embedding.shape[1] == 2:
        return embedding.copy(), ("Embedding dimension 1", "Embedding dimension 2")
    if np.sum(finite_rows) < 2:
        raise ValueError("At least two finite embedding rows are needed for PCA")

    finite = embedding[finite_rows]
    centred = finite - np.mean(finite, axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centred, full_matrices=False)
    projected = np.full((len(embedding), 2), np.nan, dtype=float)
    projected[finite_rows] = centred @ components[:2].T
    squared_singular_values = np.square(singular_values)
    total_variance = float(np.sum(squared_singular_values))
    if total_variance > 0.0:
        explained = squared_singular_values[:2] / total_variance
        labels = tuple(
            f"Codebook PCA component {index + 1} "
            f"({100.0 * ratio:.1f}% variance)"
            for index, ratio in enumerate(explained)
        )
    else:
        labels = ("Codebook PCA component 1", "Codebook PCA component 2")
    return projected, labels


def plot_codebook_embedding_colored(
    embedding: Sequence[Sequence[float]] | np.ndarray,
    feature_values: Sequence[float] | np.ndarray,
    token_ids: Sequence[Any] | np.ndarray,
    support_flags: Sequence[bool] | np.ndarray | None = None,
    *,
    feature_name: str,
    units: str | None = None,
    center: float | None = 0.0,
    value_limit: float | None = None,
    annotate_token_ids: bool = False,
    title: str | None = None,
    cmap: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Colour a 2-D codebook projection by a physiological feature.

    For embeddings with more than two dimensions, deterministic unscaled PCA
    scores are computed with NumPy SVD.  Marker direction redundantly encodes
    whether a supported value is below, at, or above the declared centre
    (or median when ``center=None``).
    """

    raw_embedding = np.asarray(embedding, dtype=float)
    projected, axis_labels = _project_embedding(raw_embedding)
    n_tokens = len(projected)
    values = _as_1d(
        feature_values,
        name="feature_values",
        length=n_tokens,
        dtype=float,
    )
    ids = _token_ids(token_ids, n_tokens)
    flags = _support_flags(support_flags, n_tokens)
    finite_coordinates = np.all(np.isfinite(projected), axis=1)
    finite_values = np.isfinite(values)
    supported = finite_coordinates & finite_values & flags

    if center is not None:
        centre = float(center)
        if not np.isfinite(centre):
            raise ValueError("center must be finite or None")
        deviations = values[supported] - centre
        limit = _symmetric_limit(deviations, value_limit)
        norm: mpl.colors.Normalize = mpl.colors.TwoSlopeNorm(
            vmin=centre - limit,
            vcenter=centre,
            vmax=centre + limit,
        )
        colour_map = mpl.colormaps.get_cmap(cmap or "RdBu_r").with_extremes(
            bad=MISSING_COLOR
        )
        comparison = centre
        comparison_label = f"reference {centre:g}"
    else:
        finite_supported_values = values[supported]
        if finite_supported_values.size == 0:
            vmin, vmax = 0.0, 1.0
            comparison = 0.5
        else:
            vmin = float(np.min(finite_supported_values))
            vmax = float(np.max(finite_supported_values))
            if vmin == vmax:
                padding = max(abs(vmin) * 0.05, 0.5)
                vmin, vmax = vmin - padding, vmax + padding
            comparison = float(np.median(finite_supported_values))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colour_map = mpl.colormaps.get_cmap(cmap or "viridis").with_extremes(
            bad=MISSING_COLOR
        )
        comparison_label = f"median {comparison:g}"

    scale = max(
        1.0,
        float(np.nanmax(np.abs(values[supported] - comparison)))
        if np.any(supported)
        else 1.0,
    )
    equal = supported & np.isclose(
        values,
        comparison,
        rtol=1e-9,
        atol=1e-12 * scale,
    )
    below = supported & ~equal & (values < comparison)
    above = supported & ~equal & (values > comparison)

    with mpl.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(
            figsize=figsize or (7.2, 5.8),
            layout="constrained",
        )
        for mask, marker, label in (
            (below, "v", f"Below {comparison_label}"),
            (equal, "o", f"At {comparison_label}"),
            (above, "^", f"Above {comparison_label}"),
        ):
            if np.any(mask):
                ax.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    c=values[mask],
                    cmap=colour_map,
                    norm=norm,
                    marker=marker,
                    s=54,
                    edgecolors=REFERENCE_COLOR,
                    linewidths=0.45,
                    label=label,
                    zorder=3,
                )

        insufficient = finite_coordinates & finite_values & ~flags
        if np.any(insufficient):
            ax.scatter(
                projected[insufficient, 0],
                projected[insufficient, 1],
                marker="X",
                s=54,
                facecolors=INSUFFICIENT_SUPPORT_COLOR,
                edgecolors=REFERENCE_COLOR,
                linewidths=0.7,
                label="Insufficient token support",
                zorder=2,
            )
        missing_values = finite_coordinates & ~finite_values
        if np.any(missing_values):
            ax.scatter(
                projected[missing_values, 0],
                projected[missing_values, 1],
                marker="x",
                s=48,
                color=MISSING_COLOR,
                linewidths=1.5,
                label="Missing feature value",
                zorder=2,
            )
        missing_coordinates = ~finite_coordinates
        if np.any(missing_coordinates):
            ax.text(
                0.01,
                0.99,
                "Tokens omitted for missing embedding coordinates: "
                f"{int(np.sum(missing_coordinates))}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=MISSING_COLOR,
                fontsize=9,
            )

        if annotate_token_ids:
            visible = finite_coordinates
            for index in np.flatnonzero(visible):
                ax.annotate(
                    str(ids[index]),
                    (projected[index, 0], projected[index, 1]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=REFERENCE_COLOR,
                )

        scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=colour_map)
        scalar_mappable.set_array(values[supported])
        colorbar = fig.colorbar(scalar_mappable, ax=ax)
        colorbar.set_label(_feature_label(feature_name, units))
        ax.set(
            xlabel=axis_labels[0],
            ylabel=axis_labels[1],
            title=title or f"Codebook geometry coloured by {feature_name}",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#D0D0D0", linewidth=0.6, alpha=0.65)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=labels, frameon=False)

    return fig, ax


def _normalise_formats(
    output: Path,
    formats: Sequence[str] | str | None,
) -> tuple[Path, tuple[str, ...]]:
    suffix = output.suffix.lower().lstrip(".")
    if formats is None:
        normalised = (suffix,) if suffix in _SUPPORTED_FORMATS else ("png",)
    else:
        candidates = (formats,) if isinstance(formats, str) else tuple(formats)
        normalised = tuple(str(item).lower().lstrip(".") for item in candidates)
    if not normalised:
        raise ValueError("formats must contain at least one format")
    if len(set(normalised)) != len(normalised):
        raise ValueError("formats must not contain duplicates")
    invalid = sorted(set(normalised) - set(_SUPPORTED_FORMATS))
    if invalid:
        raise ValueError(
            f"Unsupported figure format(s): {invalid}; "
            f"choose from {_SUPPORTED_FORMATS}"
        )
    if suffix in _SUPPORTED_FORMATS:
        if formats is not None and suffix not in normalised:
            raise ValueError(
                f"Output suffix '.{suffix}' must be included in formats"
            )
        stem = output.with_suffix("")
    else:
        stem = output
    return stem, normalised


def _target_with_suffix(stem: Path, suffix: str) -> Path:
    return Path(f"{stem}.{suffix}")


def _stage_path(parent: Path, stem_name: str, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=parent,
        prefix=f".{stem_name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(raw_path)


def _publish_without_overwrite(staged: Path, target: Path) -> None:
    """Atomically expose a complete staged file and refuse target replacement."""

    try:
        os.link(staged, target)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}") from error
    staged.unlink()


def save_figure_atomic(
    figure: Figure,
    output: str | Path,
    *,
    formats: Sequence[str] | str | None = None,
    dpi: int = 300,
) -> FigureArtifacts:
    """Atomically save a figure without ever replacing an existing artifact.

    ``output`` may be a stem or include one of the supported suffixes.  Passing
    ``formats=("png", "pdf", "svg")`` writes all three representations.  The
    exporter preserves the figure's physical page size (``bbox_inches=None``)
    and uses an opaque white background.

    Existing regular files, directories, or symlinks cause a preflight
    :class:`FileExistsError`.  There is deliberately no overwrite switch.
    """

    if not isinstance(figure, Figure):
        raise TypeError("figure must be a matplotlib.figure.Figure")
    output_path = Path(output)
    stem, normalised_formats = _normalise_formats(output_path, formats)
    if not stem.name:
        raise ValueError("output must include a non-empty filename stem")
    dpi = int(dpi)
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure_targets = tuple(
        _target_with_suffix(stem, format_name)
        for format_name in normalised_formats
    )
    existing = [
        target
        for target in figure_targets
        if target.exists() or target.is_symlink()
    ]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing artifact(s): {joined}")

    staged_paths: list[Path] = []
    staged_outputs: list[tuple[Path, Path, str]] = []
    try:
        save_rc = {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
        }
        with mpl.rc_context(save_rc):
            for target, format_name in zip(
                figure_targets,
                normalised_formats,
                strict=True,
            ):
                staged = _stage_path(
                    stem.parent,
                    stem.name,
                    f".{format_name}.tmp",
                )
                staged_paths.append(staged)
                figure.savefig(
                    staged,
                    format=format_name,
                    dpi=dpi,
                    bbox_inches=None,
                    facecolor="white",
                    edgecolor="white",
                    transparent=False,
                )
                staged_outputs.append((staged, target, format_name))

        for staged, target, _ in staged_outputs:
            _publish_without_overwrite(staged, target)
    finally:
        for staged in staged_paths:
            if staged.exists() or staged.is_symlink():
                staged.unlink()

    return FigureArtifacts(figure_paths=figure_targets)


__all__ = [
    "FigureArtifacts",
    "INSUFFICIENT_SUPPORT_COLOR",
    "MISSING_COLOR",
    "SUPPORTED_COLOR",
    "plot_codebook_embedding_colored",
    "plot_token_feature_heatmap",
    "plot_token_feature_profile_ci",
    "plot_token_support",
    "save_figure_atomic",
]
