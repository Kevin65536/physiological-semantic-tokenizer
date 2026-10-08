"""Read-only illustrations of retained synthetic cases and a frozen encoder.

The report function deliberately selects evaluation base 0 before reading scores.
It uses a fresh, isolated interpreter for the retained reader and generator.
No fitting, normalization update, or write to the evidence run is performed.
"""


def build_synthetic_waveform_figures(repo, run, out):
    """Return readable bitmap figures and their case-level audit manifest."""
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    repo, run, out = (Path(value).resolve() for value in (repo, run, out))
    frozen = run / "source_snapshot_analysis_v1"
    tokenizer = run / "tokenizer"
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    synw_prior_manifest_path = out / "synthetic_waveform_provenance.json"
    synw_prior_figures = {}
    if synw_prior_manifest_path.exists():
        synw_prior_figures = {item["key"]: item for item in json.loads(synw_prior_manifest_path.read_text())["figures"]}
    scenarios = ["baseline", "true_a_shape", "feature_gain_only", "true_a_amplitude",
                 "Hb_gain_only", "exact_joint_alias", "non_dct_impulse", "tau_change",
                 "correlated_Hb_component"]
    # The isolated interpreter prevents another report section's live src imports
    # from changing the namespace used by the frozen reader.
    synw_script = r'''
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
run, target = Path(sys.argv[1]), Path(sys.argv[2])
scenarios = json.loads(sys.argv[3])
frozen = run / "source_snapshot_analysis_v1"
sys.path.insert(0, str(frozen))
reader_path = frozen / "experiments/scripts/train_semantic_response_tokenizer.py"
spec = importlib.util.spec_from_file_location("synw_retained_reader", reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
from src.inference.semantic_response_synthetic import generate_response_case
import src.tokenizers.semantic_response_tokenizer as model_source
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root = run / "tokenizer"
resolved = json.loads((root / "resolved_tokenizer_config.json").read_text())
arm = "distribution_randomized_generator"
data = reader.load_prepared(root, arm, "evaluation", resolved["config_identity"])
base = str(data["base_id"][0])
indices = np.flatnonzero((data["base_id"] == base) & np.isin(data["scenario"], scenarios))
selected = {key: value[indices].copy() for key, value in data.items()}
del data
model, ckpt = reader.selected_model(root, arm, 101, resolved["options"], resolved["config_identity"], "cpu")
if set(selected["base_id"]) & set(ckpt["training_base_ids"] + ckpt["selection_base_ids"]):
    raise ValueError("selected illustration overlaps fitting/selection base IDs")
prediction = reader.predictions(model, selected, "cpu", len(indices), ckpt["amplitude_mean"], ckpt["amplitude_scale"])
arrays = {"data_" + key: value for key, value in selected.items()}
arrays["data_retained_row_index"] = indices
for modality, values in prediction.items():
    shape_sd, amplitude_sd = reader.uncertainty(arm, values, ckpt["calibration"][modality])
    values["shape_sd"] = shape_sd
    values["log_amplitude_sd"] = amplitude_sd
    branch = getattr(model, modality + "_branch")
    values["reconstruction_observation_coordinate"] = (values["reconstruction"] * branch.normalizer.scale.detach().numpy()
                                                       + branch.normalizer.mean.detach().numpy())
    for key, value in values.items():
        arrays["pred_" + modality + "_" + key] = value
for key in ("physical_hb", "observation_component", "hb_noise", "eeg_noise", "driver", "time_s"):
    arrays["truth_" + key] = []
checks = []
generated = {}
base_index = int(base.rsplit("/", 1)[-1])
generator_seed = int(resolved["options"]["generator_seed"])
for row, (scenario, variant) in enumerate(zip(selected["scenario"], selected["variant"])):
    if str(scenario) not in generated:
        generated[str(scenario)] = generate_response_case(generator_seed, str(scenario), "randomized",
                                                          partition="evaluation", base_id=base_index)
    truth = generated[str(scenario)][str(variant)]
    comparison = {}
    for key in ("eeg", "hb", "source_shape", "a"):
        rebuilt = np.asarray(truth[key], dtype=selected[key].dtype)
        delta = float(np.max(np.abs(rebuilt - selected[key][row])))
        comparison[key] = delta
        if not np.array_equal(rebuilt, selected[key][row]):
            raise ValueError(f"frozen generator no longer reproduces retained {scenario}/{variant}/{key}: {delta}")
    if float(truth["log_amplitude"]) != float(selected["log_amplitude"][row]):
        raise ValueError("retained amplitude identity differs")
    for key in ("physical_hb", "observation_component", "hb_noise", "eeg_noise", "driver", "time_s"):
        arrays["truth_" + key].append(truth[key])
    reconstructed_truth = (truth["metadata"]["hb_coordinate_gain"] * truth["physical_hb"]
                           + truth["observation_component"] + truth["hb_noise"])
    checks.append({"scenario": str(scenario), "variant": str(variant), "retained_row_index": int(indices[row]),
                   "bitwise_float32_replay_max_absolute_difference": comparison,
                   "generative_Hb_additive_closure_max_abs": float(np.max(np.abs(reconstructed_truth - truth["hb"])) )})
meta = {"base_id": base, "base_index": base_index, "arm": arm, "seed": 101, "generator_seed": generator_seed,
        "config_identity": resolved["config_identity"], "calibration": ckpt["calibration"],
        "amplitude_training_mean": ckpt["amplitude_mean"], "amplitude_training_scale": ckpt["amplitude_scale"],
        "reader_source": str(reader_path), "model_runtime_source": model_source.__file__,
        "generator_source": str(frozen / "src/inference/semantic_response_synthetic.py"),
        "selected_checkpoint": str(root / "fits" / arm / "seed_101/selected.pt"),
        "checkpoint_options": ckpt["options"], "replay_checks": checks,
        "inference_rows": len(indices), "inference_device": "cpu", "inference_threads": 1}
arrays["audit_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
np.savez_compressed(target, **{key: np.asarray(value) for key, value in arrays.items()})
'''
    with tempfile.TemporaryDirectory(prefix="semantic_waveforms_") as synw_tmp:
        synw_payload = Path(synw_tmp) / "frozen_subset.npz"
        synw_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run([sys.executable, "-c", synw_script, str(run), str(synw_payload), json.dumps(scenarios)],
                       cwd=frozen, env=synw_env, check=True, capture_output=True, text=True)
        with np.load(synw_payload, allow_pickle=False) as synw_handle:
            synw_arrays = {key: synw_handle[key].copy() for key in synw_handle.files}

    synw_audit = json.loads(str(synw_arrays.pop("audit_json")))
    synw_data = {key[5:]: value for key, value in synw_arrays.items() if key.startswith("data_")}
    synw_truth = {key[6:]: value for key, value in synw_arrays.items() if key.startswith("truth_")}
    synw_pred = {modality: {key[len("pred_" + modality + "_"):]: value
                           for key, value in synw_arrays.items() if key.startswith("pred_" + modality + "_")}
                 for modality in ("eeg", "hb")}
    synw_metadata = [json.loads(str(value)) for value in synw_data["metadata_json"]]
    synw_time = synw_truth["time_s"][0]
    synw_colors = {"control": "#68788a", "intervention": "#8054b3", "truth": "#182c42",
                   "eeg": "#0084a4", "hb": "#d45050", "component": "#dc8b27", "noise": "#a5aab2"}
    synw_font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    synw_font_name = FontProperties(fname=str(synw_font_path)).get_name() if synw_font_path.exists() else "DejaVu Sans"
    synw_previous_rc = plt.rcParams.copy()
    plt.rcParams.update({"font.family": synw_font_name, "font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 12, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "#fbfcfe"})
    synw_sources = {
        "prepared": str(tokenizer / "prepared/randomized/evaluation.npz"),
        "prepared_metadata": str(tokenizer / "prepared/randomized/evaluation.json"),
        "resolved_config": str(tokenizer / "resolved_tokenizer_config.json"),
        "checkpoint": synw_audit["selected_checkpoint"],
        "reader": synw_audit["reader_source"],
        "generator": synw_audit["generator_source"],
        "model": synw_audit["model_runtime_source"],
        "existing_evaluation_record": str(tokenizer / "evaluation/distribution_randomized_generator/seed_101/record.json"),
    }
    synw_figures = []

    def synw_pair(scenario):
        found = {}
        for variant in ("control", "intervention"):
            rows = np.flatnonzero((synw_data["scenario"] == scenario) & (synw_data["variant"] == variant))
            if len(rows) != 1:
                raise ValueError(f"illustration needs exactly one {scenario}/{variant}")
            found[variant] = int(rows[0])
        return found["control"], found["intervention"]

    def synw_style(ax, title, ylabel=None, time=True):
        ax.set_title(title, loc="left", pad=6)
        ax.grid(True, linewidth=.4, alpha=.24)
        if ylabel:
            ax.set_ylabel(ylabel)
        if time:
            ax.set_xlim(synw_time[0], synw_time[-1])
            ax.set_xlabel("时间 / s")
            ax.axvspan(0, 4.75, color="#aab4c0", alpha=.10, lw=0)

    def synw_figure(title, nrows=2, ncols=3, height=5.1, identity_footer=False):
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, height), squeeze=False)
        fig.suptitle(title, x=.04, ha="left", fontsize=14, fontweight="bold", y=.988)
        fig.subplots_adjust(left=.09, right=.985, top=.75, bottom=.15, hspace=.72, wspace=.42)
        if identity_footer:
            fig.text(.04, .025, "预定首例 evaluation/0  ·  distribution_randomized  ·  训练种子101  ·  灰底=前20点参考段；所有结果仅描述该个案",
                     fontsize=12, color="#566678")
        return fig, axes

    def synw_interval(ax, row, modality, title):
        truth = synw_data["source_shape"][row]
        mean, sd = synw_pred[modality]["shape_mean"][row], synw_pred[modality]["shape_sd"][row]
        ax.fill_between(synw_time, mean-1.959963984540054*sd, mean+1.959963984540054*sd,
                        color=synw_colors[modality], alpha=.16, label="95% 点边际预测区间")
        ax.plot(synw_time, truth, color=synw_colors["truth"], lw=1.6, label="生成器真形状")
        ax.plot(synw_time, mean, color=synw_colors[modality], lw=1.5, label=f"{modality.upper()}语义均值")
        synw_style(ax, title, "单位 RMS 形状")
        ax.legend(loc="best", frameon=False)

    def synw_amplitudes(ax, rows, title, label_variants=True):
        labels, truth_ys = [], []
        for item, row in enumerate(rows):
            role = "控制" if str(synw_data["variant"][row]) == "control" else "干预"
            y = float(item)
            truth = float(synw_data["log_amplitude"][row])
            ax.scatter(truth, y, c=synw_colors["truth"], marker="|", s=190, linewidths=1.8,
                       label="真 log幅值" if item == 0 else None, zorder=5)
            for delta, modality in ((-.15, "eeg"), (.15, "hb")):
                mu = float(synw_pred[modality]["log_amplitude_mean"][row])
                half = 1.959963984540054*float(synw_pred[modality]["log_amplitude_sd"][row])
                ax.errorbar(mu, y+delta, xerr=half, fmt="o", markersize=3.5, capsize=2,
                            color=synw_colors[modality], linewidth=1.1,
                            label=f"{modality.upper()}均值±95%区间" if item == 0 else None)
            labels.append(role if label_variants else str(item+1))
            truth_ys.append(y)
        ax.set_yticks(truth_ys, labels)
        ax.set_ylim(-.5, len(rows)-.5)
        ax.invert_yaxis()
        ax.set_xlabel("ln(RMS(a − 前20点均值))")
        synw_style(ax, title, time=False)
        ax.legend(loc="best", frameon=False)

    def synw_eeg_bands(ax, row, title, reconstruction=False):
        bands = synw_data["eeg"][row].reshape(120, 6, 5).mean(axis=1)
        palettes = ["#607d8b", "#bb9554", "#0084a4", "#d45050", "#8054b3"]
        # Only bands 2/3 are named alpha/beta by the retained generator.
        for band, name in enumerate(("坐标0", "坐标1", "α", "β", "坐标4")):
            ax.plot(synw_time, bands[:, band], color=palettes[band], lw=1.15, label=name)
            if reconstruction:
                rec = synw_pred["eeg"]["reconstruction_observation_coordinate"][row].reshape(120, 6, 5).mean(axis=1)
                ax.plot(synw_time, rec[:, band], color=palettes[band], lw=.8, ls="--", alpha=.70)
        synw_style(ax, title, "有效 Δlog-power")
        ax.legend(loc="upper right", ncol=3, frameon=False)

    def synw_observed_hb(ax, row, title, reconstruction=False):
        for channel, name, color in ((0, "HbO", "#d45050"), (1, "HbR", "#0084a4")):
            ax.plot(synw_time, synw_data["hb"][row, :, channel], lw=1.35, color=color, label=f"{name}观测")
            if reconstruction:
                ax.plot(synw_time, synw_pred["hb"]["reconstruction_observation_coordinate"][row, :, channel],
                        color=color, ls="--", lw=1.2, label=f"{name}完整重建")
        synw_style(ax, title, "合成 Hb 观测坐标")
        ax.legend(loc="best", frameon=False, ncol=2)

    def synw_delta(ax, control, intervention, title):
        ax.plot(synw_time, synw_data["source_shape"][intervention]-synw_data["source_shape"][control],
                color=synw_colors["truth"], lw=1.8, label="真 Δshape")
        for modality in ("eeg", "hb"):
            ax.plot(synw_time, synw_pred[modality]["shape_mean"][intervention]-synw_pred[modality]["shape_mean"][control],
                    color=synw_colors[modality], lw=1.35, label=f"{modality.upper()}估计变化")
        synw_style(ax, title, "干预 − 控制")
        ax.legend(frameon=False, loc="best")

    def synw_observed_pair(ax, control, intervention, modality, title, channels=None):
        if modality == "eeg":
            values = synw_data["eeg"].reshape(-1, 120, 6, 5).mean(axis=2)
            channels = [(2, "α"), (3, "β")] if channels is None else channels
            ylabel = "有效 Δlog-power"
        else:
            values = synw_data["hb"]
            channels = [(0, "HbO"), (1, "HbR")] if channels is None else channels
            ylabel = "合成 Hb 观测坐标"
        for channel, name in channels:
            for row, role, color in ((control, "控制", synw_colors["control"]),
                                     (intervention, "干预", synw_colors["intervention"])):
                ax.plot(synw_time, values[row, :, channel], lw=1.2, color=color,
                        ls="-" if channel == channels[0][0] else "--", label=f"{name}{role}")
        synw_style(ax, title, ylabel)
        ax.legend(frameon=False, ncol=2, loc="best")

    def synw_case_stats(scenario):
        control, intervention = synw_pair(scenario)
        stats = dict(scenario=scenario, base_id=synw_audit["base_id"],
                     case_ids=[f"{synw_audit['base_id']}/{scenario}/{role}" for role in ("control", "intervention")],
                     retained_row_indices=[int(synw_data["retained_row_index"][row]) for row in (control, intervention)],
                     mean_absolute_input_difference={modality: float(np.mean(np.abs(synw_data[modality][intervention]
                                                                                   -synw_data[modality][control])))
                                                      for modality in ("eeg", "hb")},
                     bitwise_identical_inputs={modality: bool(np.array_equal(synw_data[modality][intervention], synw_data[modality][control]))
                                              for modality in ("eeg", "hb")},
                     bitwise_identical_input_masks={modality: bool(np.array_equal(synw_data[modality+"_mask"][intervention], synw_data[modality+"_mask"][control]))
                                                    for modality in ("eeg", "hb")},
                     truth_shape_delta_RMS=float(np.sqrt(np.mean((synw_data["source_shape"][intervention]
                                                                -synw_data["source_shape"][control])**2))),
                     truth_log_amplitude_delta=float(synw_data["log_amplitude"][intervention]-synw_data["log_amplitude"][control]),
                     truth_parameter_control=synw_metadata[control]["parameters"]["free"],
                     truth_parameter_intervention=synw_metadata[intervention]["parameters"]["free"],
                     interpretation="preselected_single_case_not_scenario_aggregate", modalities={})
        for modality in ("eeg", "hb"):
            pred = synw_pred[modality]
            bitwise = {key: bool(np.array_equal(value[control], value[intervention]))
                       for key, value in pred.items()}
            stats["modalities"][modality] = {
                "estimated_shape_delta_RMS": float(np.sqrt(np.mean((pred["shape_mean"][intervention]-pred["shape_mean"][control])**2))),
                "estimated_log_amplitude_delta": float(pred["log_amplitude_mean"][intervention]-pred["log_amplitude_mean"][control]),
                "all_exported_outputs_bitwise_identical": all(bitwise.values()),
                "outputs_bitwise_equal_by_key": bitwise,
                "shape_MAE_control_intervention": [float(np.mean(np.abs(pred["shape_mean"][row]-synw_data["source_shape"][row])))
                                                   for row in (control, intervention)],
                "shape_95_point_marginal_case_coverage": [float(np.mean(np.abs(pred["shape_mean"][row]-synw_data["source_shape"][row])
                                                                        <=1.959963984540054*pred["shape_sd"][row]))
                                                         for row in (control, intervention)],
                "log_amplitude_95_intervals_control_intervention": [[float(pred["log_amplitude_mean"][row]-1.959963984540054*pred["log_amplitude_sd"][row]),
                                                                      float(pred["log_amplitude_mean"][row]+1.959963984540054*pred["log_amplitude_sd"][row])]
                                                                     for row in (control, intervention)],
                "log_amplitude_95_case_covered_control_intervention": [bool(abs(float(pred["log_amplitude_mean"][row]-synw_data["log_amplitude"][row]))
                                                                          <=1.959963984540054*float(pred["log_amplitude_sd"][row]))
                                                                      for row in (control, intervention)],
            }
        return stats

    synw_all_case_stats = {scenario: synw_case_stats(scenario) for scenario in scenarios}
    synw_manifest_path = out / "synthetic_waveform_provenance.json"

    def synw_shared_legend(fig, key):
        def synw_line(color, label, style="-"):
            return Line2D([], [], color=color, lw=1.7, ls=style), label

        truth = synw_line(synw_colors["truth"], "真shape / log幅值")
        eeg = synw_line(synw_colors["eeg"], "EEG语义")
        hb = synw_line(synw_colors["hb"], "Hb语义")
        interval = (Patch(facecolor="#8495a8", alpha=.20), "95%边际区间")
        reconstruction = synw_line("#566678", "完整重建（虚线）", "--")
        control = synw_line(synw_colors["control"], "控制")
        intervention = synw_line(synw_colors["intervention"], "干预")
        channel_styles = [synw_line("#566678", "α / HbO（实线）"),
                          synw_line("#566678", "β / HbR（虚线）", "--")]
        if key in ("01_baseline", "05_ood_impulse"):
            entries = [synw_line(color, label) for color, label in zip(
                ["#607d8b", "#bb9554", "#0084a4", "#d45050", "#8054b3"],
                ["坐标0", "坐标1", "α", "β", "坐标4"])]
            entries += [synw_line("#d45050", "HbO观测"), synw_line("#0084a4", "HbR观测"),
                        reconstruction, truth, eeg, hb, interval]
            columns = 6
        elif key == "02_shape_vs_eeg_gain":
            entries = [control, intervention, *channel_styles,
                       synw_line(synw_colors["truth"], "真Δshape"),
                       synw_line(synw_colors["eeg"], "EEG估计变化"),
                       synw_line(synw_colors["hb"], "Hb估计变化")]
            columns = 4
        elif key == "04_exact_joint_alias":
            entries = [control, intervention, *channel_styles,
                       synw_line(synw_colors["truth"], "真log幅值"), eeg, hb, interval]
            columns = 4
        elif key == "06_tau_response":
            entries = [control, synw_line(synw_colors["intervention"], "τ改变"), reconstruction,
                       truth, eeg, hb, interval,
                       synw_line("#d45050", "HbO响应Δ"), synw_line("#0084a4", "HbR响应Δ"),
                       synw_line("#566678", "观测Δ（虚线）", "--"),
                       synw_line("#566678", "α（实线）"), synw_line("#566678", "β（虚线）", "--")]
            columns = 6
        elif key == "07_correlated_hb":
            entries = [synw_line(synw_colors["truth"], "物理响应×gain"),
                       synw_line(synw_colors["component"], "真共同Hb项"),
                       synw_line(synw_colors["noise"], "配对噪声"),
                       synw_line(synw_colors["hb"], "分量和=观测", "--"),
                       synw_line("#d45050", "HbO观测"), synw_line("#0084a4", "HbR观测"),
                       reconstruction, truth, eeg, hb, interval]
            columns = 6
        else:
            return
        title_abbreviations = {
            "EEG：6通道平均的5个频带坐标": "EEG：6通道平均频带坐标",
            "Hb完整重建：D(S均值, O代码)": "Hb完整重建：D(S, O)",
            "EEG-only：连续 shape 与95%区间": "EEG-only：shape与95%区间",
            "Hb-only：连续 shape 与95%区间": "Hb-only：shape与95%区间",
            "幅值：两模态独立估计 ln(RMS)": "两模态独立log幅值估计",
            "语义真Δshape=0，估计是否跟着变？": "真Δshape=0：估计变化",
            "真幅值不变：log幅值估计与95%区间": "真幅值不变：估计95%区间",
            "真驱动与共同项：参考RMS内积=0.85": "真驱动与共同项：内积0.85",
            "真幅值不同，预测分布完全相同": "不同真幅值，同一预测分布",
            "生成器定位：变化由τ响应产生": "生成器定位：τ响应改变",
            "参考RMS幅值固定：控制与脉冲": "RMS幅值固定：控制vs脉冲",
            "网络：完整重建 D(S均值,O)": "网络完整重建：D(S, O)",
        }
        for ax in fig.axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            old_title = ax.get_title(loc="left")
            ax.set_title(title_abbreviations.get(old_title, old_title), loc="left", fontsize=12)
        legend = fig.legend([item[0] for item in entries], [item[1] for item in entries],
                            loc="upper center", bbox_to_anchor=(.535, .932), ncol=columns,
                            frameon=False, fontsize=12, columnspacing=1.15, handletextpad=.45)
        fig.canvas.draw()
        if legend.get_window_extent().x0 < 0 or legend.get_window_extent().x1 > fig.bbox.width:
            raise ValueError(f"wide_v2 legend exceeds figure width: {key}")

    def synw_save(fig, key, title, caption, cases):
        if not key.startswith("03"):
            synw_shared_legend(fig, key)
            key += "_wide_v2"
        path = figures_dir / f"syn_wave_{key}.png"
        fig.savefig(path, dpi=220, facecolor="white")
        plt.close(fig)
        record = dict(key=f"syn_wave_{key}", path=str(path), title=title, caption=caption,
                      source=synw_sources, stats={scenario: synw_all_case_stats[scenario] for scenario in cases},
                      statsmanifest=str(synw_manifest_path))
        synw_figures.append(record)

    control, intervention = synw_pair("baseline")
    fig, axes = synw_figure("合成基线：真实输入、完整重建与两模态独立语义输出")
    synw_eeg_bands(axes[0, 0], control, "EEG：6通道平均的5个频带坐标")
    synw_observed_hb(axes[0, 1], control, "HbO / HbR：实际合成输入")
    synw_observed_hb(axes[0, 2], control, "Hb完整重建：D(S均值, O代码)", reconstruction=True)
    synw_interval(axes[1, 0], control, "eeg", "EEG-only：连续 shape 与95%区间")
    synw_interval(axes[1, 1], control, "hb", "Hb-only：连续 shape 与95%区间")
    synw_amplitudes(axes[1, 2], [control], "幅值：两模态独立估计 ln(RMS)")
    synw_save(fig, "01_baseline", "合成基线：输入与独立语义",
              "预定 evaluation/0 基线，未按评分挑选。频带是有效基线相对 log-power 坐标，不是原始电压；源语义为首20点参考后的单位RMS形状与ln(RMS幅值)。各编码器只读自身模态。阴影为在selection冻结温度后给出的95%点边际预测区间，不是整条轨迹的联合可信带。Hb完整重建来自非线性D(S均值,O)，不能拆成可加的semantic Hb与observation Hb。",
              ["baseline"])

    fig, axes = synw_figure("反事实形状：真实驱动改变 vs. EEG读出增益改变")
    for nrow, (scenario, name) in enumerate((("true_a_shape", "真 shape 改变"), ("feature_gain_only", "EEG增益×1.8，真驱动不变"))):
        control, intervention = synw_pair(scenario)
        synw_observed_pair(axes[nrow, 0], control, intervention, "eeg",
                           "真shape改变：α / β输入" if nrow == 0 else "仅EEG增益×1.8：α / β输入")
        synw_observed_pair(axes[nrow, 1], control, intervention, "hb", "同一反事实：HbO / HbR输入")
        synw_delta(axes[nrow, 2], control, intervention, "语义变化：真值与估计的差")
    synw_save(fig, "02_shape_vs_eeg_gain", "形状变化还是读出尺度变化？",
              "两行使用同一基础轨迹与配对噪声。上行真shape改变但参考RMS幅值固定；下行仅EEG共同log-power坐标增益乘1.8，Hb和真shape均保持不变。右列直接画干预减控制的shape：非零估计且真值为零表示该个案的虚假语义变化。图中没有对差分画95%区间，因为模型未提供两次预测或时间点之间的协方差。",
              ["true_a_shape", "feature_gain_only"])

    # The original 12x5.9 draft is retained. Two 12x5.1 exports keep all labels
    # 12 pt before embedding: 12 * (4.72/5.1) > 11 pt in the slide frame.
    synw_amplitude_cases = [("true_a_amplitude", "真驱动幅值×2"),
                            ("feature_gain_only", "仅EEG增益×1.8"),
                            ("Hb_gain_only", "仅Hb增益×1.7")]
    synw_amplitude_caption = "真驱动×2使ln(RMS)改变ln2≈0.693；两种读出增益干预的真幅值均不变。图用实际α与HbO/HbR输入展示变化，再比较控制/干预的真log幅值及两个独立编码器的95%边际预测区间。EEG gain只乘有效log-power共同读出项，Hb gain只乘物理Hb均值响应；都不是人体绝对增益恢复。"
    synw_amplitude_keys = ("syn_wave_03a_amplitude_inputs_wide", "syn_wave_03b_amplitude_intervals_wide")
    if all(key in synw_prior_figures and Path(synw_prior_figures[key]["path"]).exists()
           for key in synw_amplitude_keys):
        # Approved amplitude exports are reused byte-for-byte, without redrawing.
        synw_figures.extend(synw_prior_figures[key] for key in synw_amplitude_keys)
    else:
        with plt.rc_context({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                             "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12}):
            fig, axes = synw_figure("反事实幅值①：三种干预的实际EEG与Hb输入", identity_footer=True)
            fig.subplots_adjust(left=.085, top=.86, bottom=.18, hspace=.65, wspace=.35)
            fig.texts[-1].set_text("预定 evaluation/0  ·  冻结训练种子101  ·  单个案（非汇总）；灰底=前20点参考段")
            fig.texts[-1].set_fontsize(12)
            for col, (scenario, name) in enumerate(synw_amplitude_cases):
                control, intervention = synw_pair(scenario)
                synw_observed_pair(axes[0, col], control, intervention, "eeg", name+"：EEG α", channels=[(2, "α")])
                synw_observed_pair(axes[1, col], control, intervention, "hb", "实际HbO / HbR输入")
            synw_save(fig, "03a_amplitude_inputs_wide", "幅值与增益：实际输入波形",
                      synw_amplitude_caption, [case[0] for case in synw_amplitude_cases])

            fig, axes = synw_figure("反事实幅值②：真log幅值与两个独立编码器的95%区间", nrows=1, identity_footer=True)
            fig.subplots_adjust(left=.067, top=.71, bottom=.16, wspace=.35)
            fig.texts[-1].set_text("预定 evaluation/0  ·  冻结训练种子101  ·  单个案（非汇总）；95%边际预测区间")
            fig.texts[-1].set_fontsize(12)
            for col, (scenario, name) in enumerate(synw_amplitude_cases):
                control, intervention = synw_pair(scenario)
                synw_amplitudes(axes[0, col], [control, intervention], name)
                if col == 0:
                    synw_handles, synw_labels = axes[0, col].get_legend_handles_labels()
                axes[0, col].get_legend().remove()
            fig.legend(synw_handles, synw_labels, loc="upper center", bbox_to_anchor=(.53, .86),
                       ncol=3, frameon=False, fontsize=12)
            synw_save(fig, "03b_amplitude_intervals_wide", "幅值与增益：语义输出区间",
                      synw_amplitude_caption, [case[0] for case in synw_amplitude_cases])

    control, intervention = synw_pair("exact_joint_alias")
    alias = synw_all_case_stats["exact_joint_alias"]
    if (not all(alias["bitwise_identical_inputs"].values())
            or not all(alias["bitwise_identical_input_masks"].values())
            or not all(value["all_exported_outputs_bitwise_identical"] for value in alias["modalities"].values())):
        raise ValueError("exact joint alias illustration violated identical input/output contract")
    fig, axes = synw_figure("精确联合 alias：同一EEG/Hb观测，对应两个不同真实幅值")
    synw_observed_pair(axes[0, 0], control, intervention, "eeg", "EEG输入逐位相同：重叠曲线")
    synw_observed_pair(axes[0, 1], control, intervention, "hb", "Hb输入逐位相同：重叠曲线")
    axes[0, 2].plot(synw_time, synw_data["a"][control], color=synw_colors["control"], label="控制真a")
    axes[0, 2].plot(synw_time, synw_data["a"][intervention], color=synw_colors["intervention"], label="干预真a=2×控制")
    synw_style(axes[0, 2], "生成真值不同：a(t) 幅值×2", "规定的合成驱动单位")
    axes[0, 2].legend(frameon=False)
    for col, modality in enumerate(("eeg", "hb")):
        for row, role, color, ls in ((control, "控制输出", synw_colors["control"], "-"), (intervention, "干预输出", synw_colors["intervention"], "--")):
            axes[1, col].plot(synw_time, synw_pred[modality]["shape_mean"][row], color=color, ls=ls, label=role)
        synw_style(axes[1, col], f"{modality.upper()} shape输出逐位相同", "单位RMS形状")
        axes[1, col].legend(frameon=False)
    synw_amplitudes(axes[1, 2], [control, intervention], "真幅值不同，预测分布完全相同")
    synw_save(fig, "04_exact_joint_alias", "完全相同输入的不可辨识反例",
              "冻结生成器显式构造a×2、EEG共同读出增益×1/2及Hb观测项精确补偿。全30个EEG通道和双Hb通道输入差MAE都为0，mask相同，两分支全部导出输出逐位相同，而真log幅值差为ln2。这是确定性的不可辨识构造；它不能通过网络容量解决，也不能因不符合普通噪声分布而从评价分母删除。该个案的区间是否覆盖两个幅值仅为局部演示。",
              ["exact_joint_alias"])

    control, intervention = synw_pair("non_dct_impulse")
    fig, axes = synw_figure("分布外驱动：慢DCT训练族之外的短脉冲")
    synw_eeg_bands(axes[0, 0], intervention, "短脉冲干预后的实际EEG频带输入")
    synw_observed_hb(axes[0, 1], intervention, "脉冲经血流动力学后的实际Hb")
    synw_observed_hb(axes[0, 2], intervention, "完整Hb重建与实际输入", reconstruction=True)
    synw_interval(axes[1, 0], intervention, "eeg", "EEG-only：尖峰真shape与预测")
    synw_interval(axes[1, 1], intervention, "hb", "Hb-only：尖峰真shape与预测")
    synw_amplitudes(axes[1, 2], [control, intervention], "参考RMS幅值固定：控制与脉冲")
    synw_save(fig, "05_ood_impulse", "分布外脉冲暴露形状失败",
              "保留的non_dct_impulse干预在索引29/43/64/83加入短脉冲，真源仍在token聚合前定义且RMS幅值保持固定。模型只有离线卷积上下文，训练source族为慢DCT脉冲；Hb动力学会平滑尖峰。完整观测重建较平滑或贴近输入，不保证尖峰shape恢复。阴影仅为selection校准后的点边际预测区间；本图的120点覆盖比例不能代表整个OOD场景或独立样本覆盖率。",
              ["non_dct_impulse"])

    control, intervention = synw_pair("tau_change")
    tau0 = synw_metadata[control]["parameters"]["free"]["tau"]
    tau1 = synw_metadata[intervention]["parameters"]["free"]["tau"]
    kappa = synw_metadata[control]["parameters"]["free"]["kappa"]
    fig, axes = synw_figure(f"响应参数改变：τ={tau0:.2f}→{tau1:.2f}s，κ={kappa:.3f}固定，真驱动不变")
    synw_observed_pair(axes[0, 0], control, intervention, "eeg", "EEG输入保持完全相同")
    for col, (channel, name) in enumerate(((0, "HbO"), (1, "HbR")), start=1):
        for row, role, color in ((control, "控制", synw_colors["control"]), (intervention, "τ变化", synw_colors["intervention"])):
            axes[0, col].plot(synw_time, synw_data["hb"][row, :, channel], color=color, lw=1.3, label=f"{role}观测")
            axes[0, col].plot(synw_time, synw_pred["hb"]["reconstruction_observation_coordinate"][row, :, channel],
                              color=color, ls="--", lw=1, label=f"{role}完整重建")
        synw_style(axes[0, col], name+"：响应形状改变", "合成Hb观测坐标")
        axes[0, col].legend(frameon=False, ncol=2)
    synw_delta(axes[1, 0], control, intervention, "语义真Δshape=0，估计是否跟着变？")
    synw_amplitudes(axes[1, 1], [control, intervention], "真幅值固定：估计分布的移动")
    for channel, name, color in ((0, "HbO", "#d45050"), (1, "HbR", "#0084a4")):
        delta_physical = synw_metadata[control]["hb_coordinate_gain"]*(synw_truth["physical_hb"][intervention, :, channel]
                                                                                    -synw_truth["physical_hb"][control, :, channel])
        axes[1, 2].plot(synw_time, delta_physical, color=color, label=name+"真物理响应变化")
        axes[1, 2].plot(synw_time, synw_data["hb"][intervention, :, channel]-synw_data["hb"][control, :, channel],
                       color=color, ls="--", label=name+"观测变化")
    synw_style(axes[1, 2], "生成器定位：变化由τ响应产生", "干预 − 控制")
    axes[1, 2].legend(frameon=False, ncol=2)
    synw_save(fig, "06_tau_response", "动力学变化会被归入驱动吗？",
              f"保留tau_change场景使τ从{tau0:.4f}s乘1.6到{tau1:.4f}s，κ={kappa:.4f}在该对中固定；本轮没有独立κ-change评价场景。真驱动、EEG、共同Hb项及配对噪声保持不变。右下由生成器真物理响应验证Hb差异来源，左下和中下显示编码器语义输出是否随动力学变化移动。网络没有显式τ/κ估计头，因此语义改变不能被解释成已分离恢复的血管参数。",
              ["tau_change"])

    control, intervention = synw_pair("correlated_Hb_component")
    fig, axes = synw_figure("相关共同Hb项：生成成分可定位，网络潜变量没有可加物理语义")
    for col, (channel, name) in enumerate(((0, "HbO"), (1, "HbR"))):
        gain = synw_metadata[intervention]["hb_coordinate_gain"]
        axes[0, col].plot(synw_time, gain*synw_truth["physical_hb"][intervention, :, channel],
                          color=synw_colors["truth"], lw=1.25, label="真物理响应×Hb gain")
        axes[0, col].plot(synw_time, synw_truth["observation_component"][intervention, :, channel],
                          color=synw_colors["component"], lw=1.25, label="真附加共同Hb项")
        axes[0, col].plot(synw_time, synw_truth["hb_noise"][intervention, :, channel],
                          color=synw_colors["noise"], lw=.7, label="配对噪声")
        axes[0, col].plot(synw_time, synw_data["hb"][intervention, :, channel],
                          color=synw_colors["hb"], ls="--", lw=1.1, label="三者相加=观测")
        synw_style(axes[0, col], name+"：生成器已知的可加真成分", "合成Hb观测坐标")
        axes[0, col].legend(frameon=False, ncol=2)
    synw_observed_hb(axes[0, 2], intervention, "网络：完整重建 D(S均值,O)", reconstruction=True)
    synw_delta(axes[1, 0], control, intervention, "真驱动不变：Hb语义是否误归属？")
    synw_amplitudes(axes[1, 1], [control, intervention], "真幅值不变：log幅值估计与95%区间")
    direction = np.asarray(synw_metadata[intervention]["component_direction"])
    component = synw_truth["observation_component"][intervention] @ direction
    component /= synw_metadata[intervention]["component_amplitude"]
    source = synw_data["source_shape"][intervention]
    corr = float(np.corrcoef(component, source)[0, 1])
    reference_rms_cosine = float(np.mean(component*source)/np.sqrt(np.mean(component**2)*np.mean(source**2)))
    if not np.isclose(reference_rms_cosine, .85, atol=1e-6):
        raise ValueError("correlated-Hb truth does not reproduce the reference-RMS inner product")
    axes[1, 2].plot(synw_time, source, color=synw_colors["truth"], lw=1.35, label="真驱动单位RMS shape")
    axes[1, 2].plot(synw_time, component, color=synw_colors["component"], lw=1.25, label="沿真loading的归一化共同Hb项")
    synw_style(axes[1, 2], "真驱动与共同项：参考RMS内积=0.85", "规定的归一化真成分")
    axes[1, 2].legend(frameon=False)
    synw_all_case_stats["correlated_Hb_component"]["true_component_source_Pearson_correlation"] = corr
    synw_all_case_stats["correlated_Hb_component"]["true_component_source_reference_RMS_inner_product"] = reference_rms_cosine
    synw_save(fig, "07_correlated_hb", "共同Hb项的语义归属边界",
              "干预把共同Hb项幅值增加0.008，并将其与规定驱动的参考RMS归一化内积设为0.85；该合同量不是全窗Pearson相关（本例Pearson为0.883）。真a、EEG与血管物理响应未变。上排左两图是生成器真值提供的物理响应、附加Hb项与噪声，三者在观测坐标闭合。上排右图是网络完整重建；非线性decoder拼接S均值与O代码，所以semantic decode + observation latent不是两条可相加的物理Hb成分。右下相关只是已知生成构造，不足以在实测中唯一识别来源；语义输出移动仅显示该个案的归属敏感性。",
              ["correlated_Hb_component"])

    synw_manifest = {
        "schema": "semantic_response_synthetic_waveform_report_v1", "report_date": "2026-10-03",
        "execution": "completed_read_only_frozen_inference", "figures": synw_figures,
        "source": synw_sources, "frozen_inference": synw_audit,
        "case_selection": {"rule": "first_base_in_retained_evaluation_order_before_scores",
                           "base_id": synw_audit["base_id"], "scenarios": scenarios,
                           "score_based_selection": False, "all_counterfactual_variants_same_base": True},
        "case_stats": synw_all_case_stats,
        "uncertainty_contract": {"continuous": "selection_calibrated_Gaussian_predictive_point_marginals",
                                 "level": .95, "joint_trajectory_band": False,
                                 "counterfactual_difference_intervals": False,
                                 "case_coverage_interpretation": "120_dependent_points_in_one_preselected_case_not_population_coverage"},
        "model_observation_contract": {"decoder": "D(concat(own_source_mean, own_observation_code))",
                                       "semantic_reconstruction_gradient": "detached_for_semantically_supervised_arms",
                                       "reconstruction_target": "complete_own_modality_observation_in_train_normalized_coordinate",
                                       "physical_additive_semantic_observation_decomposition": False,
                                       "displayed_reconstruction": "inverse_transform_of_same_frozen_training_normalizer",
                                       "generative_Hb_decomposition": "truth_Hb_gain_times_physical_response_plus_processed_common_component_plus_paired_noise"},
        "scientific_boundary": "generator_conditional_synthetic_cases_no_measured_or_unique_physiological_source_qualification",
        "writes_to_evidence_run": False, "fitting_or_training_performed": False,
        "raster_export": {"format": "PNG", "dpi": 220, "normal_figure_inches": [12, 5.1],
                          "amplitude_figure_inches": [12, 5.1], "amplitude_label_font_pt": 12,
                          "amplitude_label_font_pt_in_4_72_inch_slide_frame": 12*4.72/5.1,
                          "wide_v2_minimum_font_pt": 12,
                          "wide_v2_minimum_font_pt_in_4_72_inch_slide_frame": 12*4.72/5.1,
                          "wide_v2_source_annotation": "notes_and_manifest_only",
                          "wide_v2_legend": "figure_level_shared_top"},
        "retained_previous_draft_figures": [str(figures_dir / f"syn_wave_{key}.png") for key in (
            "01_baseline", "02_shape_vs_eeg_gain", "03_amplitude_vs_gains", "04_exact_joint_alias",
            "05_ood_impulse", "06_tau_response", "07_correlated_hb")],
    }
    synw_manifest_path.write_text(json.dumps(synw_manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    plt.rcParams.update(synw_previous_rc)
    return synw_manifest
