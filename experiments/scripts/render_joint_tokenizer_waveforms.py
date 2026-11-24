"""Render frozen tokenizer waveform exports without signal rescaling or sample selection.

The evaluator owns the panel and exports normalized G0 arrays plus each sample's
frozen training scale. This renderer restores that coordinate once and displays
every exported sample, including invalid scoring support.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.dual_component_tokenizer import eeg_complete_patch_band_energy


TARGET = "#243d56"
PREDICTION = "#ca4f25"
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
COMPONENTS = (("X", "x", "prediction_sum"), ("S", "s", "prediction_S"),
              ("O", "o", "prediction_O"))
UNIT_LABELS = {"uV": "µV", "unknown_REFED_EEG_export": "REFED export unit (unresolved)",
               "subject_normalized_Hb": "subject-normalized Hb"}
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def valid_runs(mask):
    """Half-open intervals, preserving every invalid interval in the figure."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def shade_support(ax, mask, rate, token_seconds):
    for start, stop in valid_runs(~mask):
        ax.axvspan(start / rate, stop / rate, color="#c4c8cd", alpha=.6, zorder=0)
    alpha = .65 if token_seconds >= 2 else .22
    for time in np.arange(token_seconds, 30, token_seconds):
        ax.axvline(time, color="#808890", ls=":", lw=.7, alpha=alpha, zorder=0)


def paired_statistics(target, prediction, mask):
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    if not valid.any():
        return "scoring unavailable"
    y, p = target[valid], prediction[valid]
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    correlation = np.corrcoef(y, p)[0, 1] if np.std(y) > 0 and np.std(p) > 0 else np.nan
    nonfinite = int((mask & ~np.isfinite(prediction)).sum())
    failure = f" | nonfinite decode {nonfinite}" if nonfinite else ""
    return f"RMSE {rmse:.3g} | r {correlation:.3f} | scored {valid.sum()}/{len(valid)}{failure}"


def amplitude_limits(target, prediction, mask):
    values = np.r_[target[mask], prediction]
    values = values[np.isfinite(values)]
    if not len(values):
        return (-1., 1.)
    lower, upper = float(values.min()), float(values.max())
    padding = .06 * max(upper - lower, abs(upper) * .01, 1e-12)
    return lower - padding, upper + padding


def identity_text(item, model):
    start = item.get("start_s", item.get("example", [None, None, None])[1])
    selection = item.get("selection", "exported fixed panel")
    if isinstance(selection, list):
        selection = ", ".join(selection)
    model_path = Path(model)
    model_label = str(Path(*model_path.parts[-4:]).with_suffix("")) if "/" in model else model
    label = (f"{model_label} | {item['dataset']} | {item['subject']} | {item['record']}\n"
             f"{item['position_name']} | crop +{start}s | {selection}")
    if item.get("token_seconds", 2) != 2:
        label += f"\n{item['token_seconds']:g} s per ID | Diagnostic at increased token rate"
    return label


def curves(arrays, item, index):
    channels = 1 if item["modality"] == "eeg" else 2
    scale = float(item["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("positive frozen G0 training scale required")
    result = {}
    for key in ("x", "s", "o", "prediction_S", "prediction_O", "prediction_sum"):
        result[key] = np.asarray(arrays[key][index], float).reshape(-1, channels) * scale
    result["mask"] = np.asarray(arrays["mask"][index], bool).reshape(-1, channels)
    rate = 200 if item["modality"] == "eeg" else 10
    if any(value.shape != (rate * 30, channels) for value in result.values()):
        raise ValueError("the frozen 30-second EEG/Hb grid is required")
    np.testing.assert_allclose(result["prediction_sum"],
                               result["prediction_S"] + result["prediction_O"],
                               atol=1e-6 * scale, rtol=1e-5)
    return result, rate, channels


def waveform_figure(data, item, model, rate, channels, output, prediction_label, *, demean=False, dpi=220):
    rows = len(COMPONENTS) * channels
    fig, axes = plt.subplots(rows, 3, figsize=(15.4, 3.05 * rows), squeeze=False,
                             layout="constrained")
    time = np.arange(rate * 30) / rate
    ranges = ((0, 30), (14, 16), (13.5, 14.5))
    for component_index, (label, truth, estimate) in enumerate(COMPONENTS):
        for color in range(channels):
            row = component_index * channels + color
            y = data[truth][:, color].copy()
            p = data[estimate][:, color].copy()
            mask = data["mask"][:, color]
            if demean:
                patches = mask.reshape(15, rate * 2)
                for values in (y, p):
                    values_view = values.reshape(patches.shape)
                    count = patches.sum(axis=1, keepdims=True)
                    means = np.where(patches, values_view, 0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
                    values_view -= means
            color_label = "" if channels == 1 else (" HbO" if color == 0 else " HbR")
            target_visible = np.isfinite(y) if truth == "x" else mask
            limits = amplitude_limits(y, p, target_visible)
            for column, limits_time in enumerate(ranges):
                ax = axes[row, column]
                shade_support(ax, mask, rate, item["token_seconds"])
                ax.plot(time, np.where(target_visible, y, np.nan), color=TARGET, lw=.9, label="Target")
                ax.plot(time, p, color=PREDICTION, lw=.85,
                        ls="--", label=prediction_label)
                ax.set(xlim=limits_time, ylim=limits, xlabel="Time in visible crop (s)")
                if column == 0:
                    ax.set_ylabel(f"{label}{color_label}\n{UNIT_LABELS.get(item['source_unit'], item['source_unit'])}")
                if row == 0:
                    ax.set_title(("Full 30 s", "Fixed 2 s patch", "Fixed token boundary")[column])
                if column == 0:
                    ax.text(.01, .98, paired_statistics(y, p, mask), transform=ax.transAxes,
                            va="top", fontsize=9,
                            bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
                ax.grid(alpha=.12)
    axes[0, 0].legend(loc="lower left", framealpha=.9)
    diagnostic = "\n2 s patch means removed separately for shape diagnosis" if demean else ""
    fig.suptitle(identity_text(item, model) + diagnostic, fontsize=12)
    fig.supxlabel(f"Dotted lines mark {item['token_seconds']:g} s token boundaries. Gray regions have no scoring support.\n"
                  "All columns retain the same amplitude limits within each component/color.", fontsize=10)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def local_rms(values, mask, rate):
    """Centered 0.5-second RMS, requiring a complete contiguous valid footprint."""
    width = int(rate * .5)
    kernel = np.ones(width)
    squared = np.convolve(np.where(mask, values ** 2, 0), kernel, mode="valid") / width
    valid = np.convolve(mask.astype(int), kernel, mode="valid") == width
    time = (np.arange(len(squared)) + (width - 1) / 2) / rate
    return time, np.where(valid, np.sqrt(squared), np.nan)


def band_power(values, mask, rate):
    if rate != 200 or len(values) != 6000:
        raise ValueError("EEG band diagnostic requires the frozen 30-second 200-Hz grid")
    energies = eeg_complete_patch_band_energy(values.reshape(1, 15, 400, 1),
                                              mask.reshape(1, 15, 400, 1))
    return np.asarray([energies[band][0, 0] for band in BANDS])


def eeg_dynamics_figure(data, item, model, rate, output, prediction_label, *, dpi=220):
    fig, axes = plt.subplots(3, 2, figsize=(15.4, 10), layout="constrained")
    for row, (label, truth, estimate) in enumerate(COMPONENTS):
        y, p, mask = data[truth][:, 0], data[estimate][:, 0], data["mask"][:, 0]
        ax, bx = axes[row]
        shade_support(ax, mask, rate, item["token_seconds"])
        for values, name, color, style in ((y, "Target", TARGET, "-"),
                                           (p, prediction_label, PREDICTION, "--")):
            time, rms = local_rms(values, mask, rate)
            ax.plot(time, rms, color=color, ls=style, lw=1.15, label=name)
        ax.set(xlim=(0, 30), xlabel="Time in visible crop (s)",
               ylabel=f"{label} 0.5 s RMS\n{UNIT_LABELS.get(item['source_unit'], item['source_unit'])}")
        indices = np.arange(len(BANDS))
        target_power, predicted_power = band_power(y, mask, rate), band_power(p, mask, rate)
        for offset, values, name, color, hatch in ((-.19, target_power, "Target", TARGET, ""),
                                                 (.19, predicted_power, prediction_label, PREDICTION, "//")):
            bx.bar(indices + offset, values, .36, color=color, hatch=hatch, label=name)
        bx.set(xticks=indices, xticklabels=[f"{lo}–{hi}" for lo, hi in BANDS],
               xlabel="Frequency band (Hz)",
               ylabel=f"{label} band power\n({UNIT_LABELS.get(item['source_unit'], item['source_unit'])})²")
        complete_patches = int(mask.reshape(15, 400).all(1).sum())
        if not complete_patches:
            bx.text(.5, .5, "Spectrum unavailable without a complete 2 s patch",
                    transform=bx.transAxes, ha="center", wrap=True)
        else:
            bx.set_title(f"{complete_patches}/15 complete 2 s patches")
            if np.any(np.r_[target_power, predicted_power] > 0):
                bx.set_yscale("log")
            else:
                bx.text(.5, .5, "Both curves have zero power in the shown bands",
                        transform=bx.transAxes, ha="center", wrap=True)
        ax.grid(alpha=.12)
        bx.grid(axis="y", alpha=.12)
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.suptitle(identity_text(item, model) + " | EEG dynamics", fontsize=12)
    fig.supxlabel("RMS uses a complete 0.5 s footprint. Band power averages all complete 2 s patches,\n"
                  "each demeaned and Hann-windowed on the 1–45 Hz support. Positive power uses a logarithmic axis.", fontsize=10)
    fig.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_export(export, metadata, output, *, prediction_label=None, indices=None):
    """Render all preselected examples from a retained evaluation export."""
    metadata_path = Path(metadata)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("coordinate") != "G0_normalized":
        raise ValueError("explicit G0_normalized export coordinate required")
    examples = metadata["examples"]
    indices = list(range(len(examples))) if indices is None else list(indices)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(examples) for i in indices):
        raise ValueError("explicit export indices must be unique and in range")
    prediction_label = prediction_label or metadata.get("prediction_label", "Integer decode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with np.load(export) as arrays:
        if any(len(arrays[key]) != len(examples) for key in
               ("x", "s", "o", "mask", "prediction_S", "prediction_O", "prediction_sum")):
            raise ValueError("export arrays and fixed example identities disagree")
        for index in indices:
            item = examples[index]
            token_seconds = float(item.get("token_seconds", metadata.get("token_seconds", 2.)))
            if token_seconds <= 0 or not np.isclose(30 / token_seconds, round(30 / token_seconds)):
                raise ValueError("token duration must divide the visible 30-second context")
            item = dict(item, token_seconds=token_seconds,
                        interface=item.get("interface", metadata.get("interface", "one_ID_per_branch_per_2s")))
            current_prediction_label = item.get("prediction_label", prediction_label)
            data, rate, channels = curves(arrays, item, index)
            name = f"{index:03d}_{item['modality']}_{item['dataset']}_{item['subject']}"
            paths = [output / (name + "_waveforms.png")]
            waveform_figure(data, item, metadata["model"], rate, channels, paths[-1], current_prediction_label)
            if item["modality"] == "eeg":
                paths.append(output / (name + "_dynamics.png"))
                eeg_dynamics_figure(data, item, metadata["model"], rate, paths[-1], current_prediction_label)
            else:
                paths.append(output / (name + "_demeaned.png"))
                waveform_figure(data, item, metadata["model"], rate, channels, paths[-1], current_prediction_label, demean=True)
            records.append(dict(index=index, identity=item, figures=[p.name for p in paths],
                                prediction_label=current_prediction_label,
                                visual_audit="pending_actual_image_inspection"))
    (output / "manifest.json").write_text(json.dumps(dict(
        export=str(Path(export).resolve()), metadata=str(metadata_path.resolve()),
        model=metadata["model"], coordinate="restored_prepared_signal_coordinate",
        selection=("Every input entry rendered in declared order without filtering"
                   if indices == list(range(len(examples))) else
                   "Explicit retained export indices for failed-candidate inspection; no aggregate claim"),
        source_example_count=len(examples), rendered_indices=indices,
        examples=records), ensure_ascii=False, indent=2) + "\n")
    links = []
    for record in records:
        item = record["identity"]
        links.append("<h2>" + html.escape(identity_text(item, metadata["model"])) + "</h2>" +
                     "".join(f'<p><a href="{html.escape(name)}">{html.escape(name)}</a></p>'
                             for name in record["figures"]))
    (output / "index.html").write_text('<meta charset="utf-8"><title>Tokenizer waveforms</title>' +
                                      "\n".join(links))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-label", help="Explicit label for retained exports predating prediction labels")
    parser.add_argument("--indices", type=int, nargs="+", help="Explicit retained export indices for diagnostic inspection")
    args = parser.parse_args()
    render_export(args.export, args.metadata, args.output, prediction_label=args.prediction_label, indices=args.indices)
