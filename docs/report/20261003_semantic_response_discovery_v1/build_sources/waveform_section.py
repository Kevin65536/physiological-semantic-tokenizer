"""Report-only native/prepared/retained-response evidence; no fitting or cache writes."""


def build_response_waveform_figures(repo, run, out):
    """Export adjacent native, Hb, driver/state and EEG PNGs for dataset cases.

    Case selection is descriptive, from evaluation regional rows at the fixed
    40-point Hb prefix. Raw reads use the registered native readers. The only
    replay is deterministic feature preparation and array reconstruction.
    """
    import csv
    import gc
    import json
    from pathlib import Path
    import time

    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import numpy as np
    import yaml

    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.registry import REGISTERED_DATASETS
    from src.data.unified_physiology import load_native_eeg_record, load_native_fnirs_record
    from src.inference.observation_baselines import eeg_band_power, native_feature_operators
    from src.inference.shared_driver_attribution import equal_capacity_hb_basis
    from src.inference.shared_driver_modes import spectral_loadings

    repo, run, out = (Path(p).resolve() for p in (repo, run, out))
    figroot = out / "figures"
    figroot.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def _wave_read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _wave_arrays(path):
        with np.load(path, allow_pickle=False) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}

    def _wave_error(actual, expected, label, atol=1e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"Waveform replay shape/finite failure: {label}")
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-10, atol=atol):
            raise ValueError(f"Waveform replay mismatch: {label}: {error}")
        return dict(max_absolute_error=error, rtol=1e-10, atol=atol, passed=True)

    def _wave_path(path, root):
        path = Path(path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Waveform source outside retained owning root")
        return path

    cfg = _wave_read(run / "resolved_config.json")
    plan = _wave_read(run / "measured_plan.json")
    parent = (repo / cfg["source_run"]).resolve()
    if (cfg.get("schema") != "semantic_response_discovery_v1"
            or cfg.get("protected_data") != "forbidden"
            or Path(plan["parent"]).resolve() != parent
            or cfg["tensor"]["steps"] != 120 or cfg["tensor"]["dt_s"] != .25):
        raise ValueError("Unexpected waveform evidence contract")
    source_cfg = yaml.safe_load((repo / cfg["source_config"]).read_text())
    evaluation = {x["ref"]["id"]: x for x in plan["windows"] if x["split"] == "evaluation"}
    if len(evaluation) != 360 or len({x["native_id"] for x in evaluation.values()}) != 120:
        raise ValueError("Expected exact 360 regional / 120 native evaluation inventory")
    datasets = ("eeg_fnirs_single_trial", "simultaneous_eeg_nirs", "visual_cognitive_motivation")
    if {x["ref"]["dataset"] for x in evaluation.values()} != set(datasets):
        raise ValueError("Unexpected waveform dataset before raw read")
    with (run / "measured_cells.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["prefix_steps"] == "40" and r["pairing"] == "real"
                and r["arm"] in ("best_stable", "gain_selected")]
    best = {r["id"]: r for r in rows if r["arm"] == "best_stable"}
    gain = {r["id"]: r for r in rows if r["arm"] == "gain_selected"}
    choices = []
    for ds in datasets:
        pairs = [(float(gain[k]["Hb_nrmse"]) - float(r["Hb_nrmse"]), k)
                 for k, r in best.items() if r["dataset"] == ds and k in evaluation and k in gain
                 and r["success"] == "True" and gain[k]["success"] == "True"]
        expected = sum(x["ref"]["dataset"] == ds for x in evaluation.values())
        if len(pairs) != expected:
            raise ValueError("Case selection requires the complete paired evaluation denominator")
        median = float(np.median([d for d, _ in pairs]))
        chosen = (min(pairs, key=lambda p: (abs(p[0] - median), p[1])), min(pairs))
        for label, (delta, identity) in zip(("median", "worst"), chosen):
            choices.append(dict(id=identity, dataset=ds, role=label, delta=delta,
                                paired_population=len(pairs), population_median_delta=median))

    # This index reads registered metadata, never signal arrays or protected splits.
    index = CleanPhysiologyCacheIndex(repo / source_cfg["data"]["cache_root"])
    native_op = native_feature_operators(120)
    hb_model_op = native_op["fnirs"] @ native_op["native_interpolation"]
    block = np.zeros((360, 360))
    block[0::3, 0::3] = native_op["eeg"]
    block[1::3, 1::3] = block[2::3, 2::3] = hb_model_op
    common_basis = equal_capacity_hb_basis(block, modes=4, rho=.35).reshape(120, 3, 4)[:, 1:]
    t = np.arange(120) * .25
    prefix_s, score_start_s, score_stop_s = 10., cfg["measured"]["score_start"] * .25, 30.
    names = {"eeg_fnirs_single_trial": "Single-Trial", "simultaneous_eeg_nirs": "Simultaneous",
             "visual_cognitive_motivation": "Visual"}
    palette = dict(observed="#17212b", physical="#386cb0", component="#d98b26",
                   prediction="#7b4ab5", gain="#7a7a7a", HbO="#d1495b", HbR="#2374ab")
    figures, cases, loaded = [], [], None
    raw_eeg, raw_hb = None, None

    def _wave_style(ax, shade=True):
        if shade:
            ax.axvspan(0, prefix_s, color="#80bfa3", alpha=.15, zorder=0)
            ax.axvspan(score_start_s, score_stop_s, color="#e6bd65", alpha=.20, zorder=0)
        ax.axvline(prefix_s, color="#548a70", ls="--", lw=.8)
        ax.axvline(score_start_s, color="#aa8130", ls="--", lw=.8)
        ax.axvline(5., color="#acb4bc", ls=":", lw=.7)
        ax.set_xlim(0, 30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(labelsize=11.5)

    def _wave_save(fig, key, title, caption, source, stats):
        path = figroot / f"{key}.png"
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        visible_text = [artist for artist in fig.findobj(match=Text)
                        if artist.get_visible() and artist.get_text()]
        outside = []
        for artist in visible_text:
            bounds = artist.get_window_extent(renderer)
            if (bounds.x0 < -.5 or bounds.y0 < -.5
                    or bounds.x1 > fig.bbox.width + .5 or bounds.y1 > fig.bbox.height + .5):
                outside.append(dict(text=artist.get_text(), bounds=list(bounds.extents)))
        if outside:
            raise ValueError(f"Waveform text outside image: {key}: {outside}")
        min_font = min(a.get_fontsize() for a in visible_text)
        fig.savefig(path, dpi=250, facecolor="white")
        plt.close(fig)
        item = dict(key=key, path=str(path), title=title, caption=caption,
                    source=source, stats=stats, export_version="wide_final",
                    figure_inches=list(fig.get_size_inches()), minimum_font_size_pt=min_font,
                    minimum_scaled_font_size_pt=min_font * min(12. / fig.get_figwidth(), 4.72 / fig.get_figheight()),
                    text_bounds_checked=len(visible_text), text_clipping=False, legends_outside_data=True)
        figures.append(item)
        return item

    with plt.rc_context({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                         "legend.fontsize": 11.5, "font.family": "DejaVu Sans"}):
        for number, chosen in enumerate(choices, 1):
            item = evaluation[chosen["id"]]
            ref = item["ref"]
            ds = ref["dataset"]
            if (ds not in datasets or (ds == "visual_cognitive_motivation"
                                      and ref["subject"] == "S06" and "Part1" in ref["record"])):
                raise ValueError("Excluded dataset/window before native read")
            expected_key = f'{ds}__{ref["subject"]}__{ref["record"]}'
            if ref["key"] != expected_key or ref["array_index"] != ref["window"]:
                raise ValueError("Native identity drift")
            array_path = _wave_path(ref["array_path"], parent / "prepared")
            if array_path != parent / "prepared" / expected_key / (ref["region"] + ".npz"):
                raise ValueError("Prepared path does not match native identity")
            prepared_record = _wave_read(array_path.parent / "record.json")
            spec = prepared_record["spec"]
            join = f'{ds}|{ref["subject"]}|{ref["record"]}'
            if spec["join_key"] != join:
                raise ValueError("Prepared/native join mismatch")
            anchor = next(a for a in prepared_record["prepared"] if a["region"] == ref["region"])
            if anchor["eeg_channels"] != ref["eeg_channels"] or anchor["hb_channel"] != ref["hb_channel"]:
                raise ValueError("Prepared/native channel mismatch")
            windows = [w for w in anchor["windows"] if w["window"] == ref["window"]]
            if (len(windows) != 1 or any(windows[0][k] != ref[k] for k in
                                        ("eeg_start_s", "hb_start_s", "event_id", "task", "condition"))):
                raise ValueError("Prepared/native time/event mismatch")
            records = index.records_by_join_key.get(join, [])
            if len(records) != 1:
                raise ValueError("Registered native identity is not unique")
            record = records[0]
            if loaded != join:
                raw_eeg, raw_hb = None, None
                gc.collect()
                raw_eeg = load_native_eeg_record(repo, record)
                raw_hb = load_native_fnirs_record(repo, record)
                loaded = join
            registered_root = (repo / REGISTERED_DATASETS[ds].default_root).resolve()
            _wave_path(raw_eeg.source_path, registered_root)
            for p in raw_hb["provenance"]["source_paths"]:
                _wave_path(p, registered_root)
            lookup = {str(n).upper(): i for i, n in enumerate(raw_eeg.channel_names)}
            eeg_indices = [lookup[n.upper()] for n in ref["eeg_channels"]]
            eeg_start = round(ref["eeg_start_s"] * raw_eeg.sample_rate_hz)
            eeg_length = round(30 * raw_eeg.sample_rate_hz)
            eeg_values = raw_eeg.values[eeg_start:eeg_start + eeg_length, eeg_indices]
            eeg_time = np.arange(eeg_length) / raw_eeg.sample_rate_hz
            if eeg_values.shape != (eeg_length, 6) or not np.isfinite(eeg_values).all():
                raise ValueError("Incomplete native EEG window")
            hb_pair = int(anchor["hb_pair"])
            if raw_hb["channel_names"][2 * hb_pair:2 * hb_pair + 2] != [
                    ref["hb_channel"] + "_HbO", ref["hb_channel"] + "_HbR"]:
                raise ValueError("Native Hb pair identity drift")
            native_hb_select = (raw_hb["time_s"] >= ref["hb_start_s"]) & (
                raw_hb["time_s"] < ref["hb_start_s"] + 30)
            native_hb_time = raw_hb["time_s"][native_hb_select] - ref["hb_start_s"]
            native_hb_values = raw_hb["values"][native_hb_select, hb_pair]
            hb_grid = ref["hb_start_s"] + np.arange(300) / 10
            if hb_grid[0] < raw_hb["time_s"][0] or hb_grid[-1] > raw_hb["time_s"][-1]:
                raise ValueError("Native Hb window outside actual support")
            hb_input = np.column_stack([np.interp(hb_grid, raw_hb["time_s"],
                                                  raw_hb["values"][:, hb_pair, j]) for j in range(2)])
            prepared = _wave_arrays(array_path)
            eeg_prepared = prepared["eeg_features"][ref["array_index"]]
            hb_prepared = prepared["hb"][ref["array_index"]]
            power = eeg_band_power(eeg_values, bands=cfg["tensor"]["eeg_bands_hz"],
                                   sample_rate=raw_eeg.sample_rate_hz, target_rate=4.)
            if power.shape != (120, 6, 5) or np.any(power <= 0):
                raise ValueError("Invalid raw-to-prepared EEG power")
            checks = dict(
                raw_to_prepared_eeg=_wave_error(native_op["eeg"] @ np.log(power).reshape(120, 30),
                                                eeg_prepared, "raw to prepared EEG"),
                raw_to_prepared_hb=_wave_error(native_op["fnirs"] @ hb_input, hb_prepared,
                                               "raw to prepared Hb"))
            feature_path = run / "features" / "evaluation" / (ref["id"] + ".npz")
            feature = _wave_arrays(feature_path)
            feature_meta = _wave_read(feature_path.with_suffix(".json"))
            if feature_meta["ref"] != ref or feature_meta["split"] != "evaluation":
                raise ValueError("Retained feature identity drift")
            cal = _wave_read(run / "calibration" / (item["calibration_key"] + ".json"))
            checks["prepared_to_scaled_eeg"] = _wave_error(eeg_prepared * cal["eeg_factor"], feature["eeg"], "scaled EEG")
            checks["prepared_to_scaled_hb"] = _wave_error(hb_prepared * cal["hb_factor"], feature["hb"], "scaled Hb")
            row = _wave_read(run / "measured" / ref["id"] / "best_stable" / "real" / "prefix_40.json")
            gain_row = _wave_read(run / "measured" / ref["id"] / "gain_selected" / "real" / "prefix_40.json")
            response_path = _wave_path(run / row["arrays"], run / "measured")
            gain_path = _wave_path(run / gain_row["arrays"], run / "measured")
            response, control = _wave_arrays(response_path), _wave_arrays(gain_path)
            if row["id"] != ref["id"] or gain_row["id"] != ref["id"]:
                raise ValueError("Response identity mismatch")
            prediction, physical, component = (response[k] for k in ("prediction", "physical_prediction", "observation_component"))
            checks["physical_plus_common_equals_prediction"] = _wave_error(physical + component, prediction, "component sum")
            checks["common_basis_reconstruction"] = _wave_error(common_basis @ response["component_coefficients"], component, "common basis")
            states = response["states"]
            fixed = source_cfg["fixed"]
            canonical = np.column_stack((fixed["P0"] * (states[:, 4] - 1) - fixed["Q0"] * (states[:, 5] - 1),
                                         fixed["Q0"] * (states[:, 5] - 1)))
            checks["states_to_physical_hb"] = _wave_error(hb_model_op @ canonical, physical, "state to Hb")
            modes, selected_spec = feature["eeg_modes"], row["selected_spec"]
            modes_path = run / "eeg_modes" / item["calibration_key"] / ref["id"] / "real__fixed.npz"
            mode_arrays = _wave_arrays(modes_path)
            checks["retained_feature_modes_match_EEG_cache"] = _wave_error(modes, mode_arrays["r"], "retained modes")
            loading = spectral_loadings()
            eeg_a = native_op["eeg"] @ np.outer(modes[:, 0], loading[:, 0]) / cal["eeg_factor"]
            eeg_b = native_op["eeg"] @ np.outer(modes[:, 1], loading[:, 1]) / cal["eeg_factor"]
            eeg_prediction = mode_arrays["prediction"] / cal["eeg_factor"]
            eeg_residual = eeg_prepared - eeg_prediction
            checks["EEG_a_plus_b_equals_retained_prediction"] = _wave_error(eeg_a + eeg_b, eeg_prediction, "EEG loading contributions")
            checks["prepared_EEG_equals_a_plus_b_plus_residual"] = _wave_error(eeg_a + eeg_b + eeg_residual, eeg_prepared, "EEG decomposition")
            g = selected_spec["gamma"]
            expected_driver = ((modes[:, 0] + g * modes[:, 1]) / np.sqrt(1 + g * g)
                               if g else selected_spec["gain"] * modes[:, 0])
            checks["eeg_modes_to_driver"] = _wave_error(expected_driver, response["driver"], "EEG-only driver")
            endpoint = response["endpoint_mask"]
            if endpoint.shape != (120, 2) or not np.array_equal(endpoint, control["endpoint_mask"]):
                raise ValueError("Paired score support mismatch")
            nrmse = float(np.sqrt(np.mean(((prediction - feature["hb"]) / np.asarray(cal["sd_hb"]))[endpoint] ** 2)))
            checks["retained_nrmse_recomputed"] = _wave_error(np.array(nrmse), np.array(row["Hb_nrmse"]), "NRMSE")
            hb_factor = float(cal["hb_factor"])
            if hb_factor <= 0:
                raise ValueError("Hb inverse scale must be common positive")
            prediction, physical, component, gain_prediction = (a / hb_factor for a in
                (prediction, physical, component, control["prediction"]))
            residual = hb_prepared - prediction
            checks["prepared_equals_physical_plus_common_plus_observation_residual"] = _wave_error(
                physical + component + residual, hb_prepared, "prepared decomposition")
            hb_unit = ("relative MBLL units" if ds == "eeg_fnirs_single_trial" else
                       "mmol/L" if ds == "simultaneous_eeg_nirs" else "released units (unreported)")
            hb_axis_unit = ("relative Hb" if ds == "eeg_fnirs_single_trial" else
                            "mmol/L" if ds == "simultaneous_eeg_nirs" else "released units")
            eeg_unit = (raw_eeg.channel_units[eeg_indices[0]] if raw_eeg.channel_units else raw_eeg.native_unit)
            raw_label = "raw released Hb" if ds != "eeg_fnirs_single_trial" else "raw photodetector intensity"
            short = f'{names[ds]} · {ref["subject"]} · {ref["record"]} · {ref["region"]} · window {ref["window"]}'
            detail = f'EEG clock start {ref["eeg_start_s"]:.3f}s; Hb clock start {ref["hb_start_s"]:.3f}s; {ref["condition"]}'
            role = "Closest to dataset median" if chosen["role"] == "median" else "Worst paired change"
            statline = f'{role}: gain NRMSE {float(gain_row["Hb_nrmse"]):.3f} → stable {nrmse:.3f}; Δ(gain−stable)={chosen["delta"]:+.3f}'
            stats = dict(chosen, best_stable_nrmse=nrmse, gain_nrmse=float(gain_row["Hb_nrmse"]),
                         chosen_arm=row["chosen_arm"], spec=selected_spec, numerical_checks=checks)

            compact = f'{names[ds]} {ref["subject"]} · {ref["region"]} · w{ref["window"]}'
            fig = plt.figure(figsize=(13.4, 5.3))
            grid = fig.add_gridspec(2, 2, height_ratios=[1., 1.5], left=.07, right=.935,
                                   top=.82, bottom=.22, hspace=.70, wspace=.24)
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
            fig.suptitle("Native signals → prepared EEG bands | " + compact,
                         x=.07, y=.982, ha="left", fontsize=16, weight="bold")
            fig.text(.07, .915, detail + "   |   " + role, fontsize=12, color="#4e5965")
            axes[0].plot(eeg_time, eeg_values[:, 0], color="#386cb0", lw=.55)
            axes[0].set_ylabel(f'EEG {eeg_unit}')
            axes[0].set_title(f'Native voltage: {ref["eeg_channels"][0]} (1 of 6 channels)', loc="left")
            _wave_style(axes[0])
            zoom = axes[0].inset_axes([.63, .47, .36, .51])
            zoom_mask = (eeg_time >= 20) & (eeg_time < 21)
            zoom.plot(eeg_time[zoom_mask], eeg_values[zoom_mask, 0], color="#386cb0", lw=.6)
            zoom.text(.5, .94, "20–21 s", transform=zoom.transAxes, ha="center", va="top", fontsize=11.5)
            zoom.set_xticks([])
            zoom.set_yticks([])
            zoom.set_facecolor("white")
            if ds == "eeg_fnirs_single_trial":
                values = raw_hb["optical_intensity"][native_hb_select, hb_pair]
                labels, colors, unit = ("760 nm", "850 nm"), ("#795da8", "#128c8d"), "optical V"
            else:
                values, labels, colors, unit = native_hb_values, ("HbO", "HbR"), (palette["HbO"], palette["HbR"]), hb_axis_unit
            for j, (label, color) in enumerate(zip(labels, colors)):
                axes[1].plot(native_hb_time, values[:, j], color=color, label=label, lw=1.3)
            raw_title = ("Raw optical" if ds == "eeg_fnirs_single_trial" else "Raw Hb")
            unknown_unit = " [unit unreported]" if ds == "visual_cognitive_motivation" else ""
            axes[1].set_title(f'{raw_title}: {ref["hb_channel"]}{unknown_unit}', loc="left")
            axes[1].set_ylabel(unit)
            axes[1].legend(loc="lower right", bbox_to_anchor=(1, 1.045), ncol=2,
                           frameon=False, borderaxespad=0)
            _wave_style(axes[1])
            extent = float(np.max(np.abs(eeg_prepared)))
            image = axes[2].imshow(eeg_prepared.T, aspect="auto", cmap="RdBu_r", vmin=-extent, vmax=extent,
                                   extent=(0, 30, 30, 0), interpolation="nearest")
            axes[2].set_yticks(np.arange(6) * 5 + 2.5, ref["eeg_channels"])
            for boundary in range(5, 30, 5):
                axes[2].axhline(boundary, color="white", lw=.7, alpha=.85)
            axes[2].set_title("Prepared EEG: 6 channels × 5 bands; log power relative to first 5 s", loc="left")
            axes[2].set_xlabel("Seconds from each registered modality window start")
            _wave_style(axes[2], shade=False)
            cax = fig.add_axes([.948, .22, .012, .26])
            colorbar = fig.colorbar(image, cax=cax)
            colorbar.ax.tick_params(labelsize=11.5)
            colorbar.ax.set_title("Δ log\npower", fontsize=11.5, pad=6)
            footer = "Bands: 1–4 / 4–8 / 8–13 / 13–30 / 30–45 Hz. Green: Hb input 0–10s; gold: score 20–30s."
            footer2 = ("Single-Trial optical V → natural-log OD → approximate relative MBLL Hb; the derived Hb is not a raw concentration."
                       if ds == "eeg_fnirs_single_trial" else
                       "Visual Hb has unreported physical units; it is not labelled µM." if ds == "visual_cognitive_motivation" else
                       "Released Hb remains in verified mmol/L; this plot applies no concentration-unit conversion.")
            fig.text(.07, .072, footer, fontsize=11.5, color="#4e5965")
            fig.text(.07, .024, footer2, fontsize=11.5, color="#4e5965")
            source = dict(raw_eeg=str(raw_eeg.source_path), raw_hb=raw_hb["provenance"]["source_paths"],
                          prepared=str(array_path), feature=str(feature_path), modes=str(modes_path),
                          response=str(response_path), control=str(gain_path))
            raw_fig = _wave_save(fig, f"waveform_{number:02d}_{chosen['role']}_native_wide", "原始信号与频带特征：" + short,
                "精确回溯本轮评价窗口。原始电压与发布Hb/光强保持各自单位；频带图展示已有局部特征变换，不把prepared称为raw。", source, stats)

            fig, axes = plt.subplots(3, 1, figsize=(12., 5.), sharex=True)
            fig.subplots_adjust(left=.12, right=.985, top=.77, bottom=.23, hspace=.27)
            fig.suptitle("Prepared Hb components | " + compact,
                         x=.12, y=.982, ha="left", fontsize=16, weight="bold")
            fig.text(.12, .915, f'Gain NRMSE {float(gain_row["Hb_nrmse"]):.3f} → stable {nrmse:.3f}; '
                     f'Δ={chosen["delta"]:+.3f} · {hb_unit} · {role}', fontsize=11.5, color="#4e5965")
            for j in range(2):
                ax = axes[j]
                for values, label, color, style, width in (
                    (hb_prepared, "Observed", palette["observed"], "-", 1.7),
                    (physical, "Physical", palette["physical"], "-", 1.25),
                    (component, "Common", palette["component"], "-", 1.15),
                    (prediction, "Phy + common", palette["prediction"], "-", 1.3),
                    (gain_prediction, "Gain ctrl", palette["gain"], "--", 1.1)):
                    ax.plot(t, values[:, j], label=label, color=color, ls=style, lw=width)
                ax.set_ylabel("HbO" if j == 0 else "HbR")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                _wave_style(ax)
            handles, labels = axes[0].get_legend_handles_labels()
            for j, label in enumerate(("HbO", "HbR")):
                axes[2].plot(t, residual[:, j], label=label + " residual", color=palette[label], lw=1.2)
            axes[2].axhline(0, color="#9ba4ae", lw=.6)
            axes[2].set_ylabel("Residual")
            axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            residual_handles, residual_labels = axes[2].get_legend_handles_labels()
            fig.legend(handles + residual_handles, labels + residual_labels, loc="upper left",
                       bbox_to_anchor=(.12, .87), ncol=7, frameon=False, fontsize=11.5,
                       columnspacing=.8, handlelength=1.5, handletextpad=.4)
            _wave_style(axes[2])
            axes[2].set_xlabel("Seconds from the registered window start")
            fig.text(.12, .063, "Residual = observed − (physical + common). Green: Hb input 0–10s; gold: score 20–30s.", fontsize=11.5, color="#4e5965")
            fig.text(.12, .020, "Common/residual terms do not certify physiological sources or sensor noise.", fontsize=11.5, color="#4e5965")
            component_fig = _wave_save(fig, f"waveform_{number:02d}_{chosen['role']}_components_wide", "Hb成分分离：" + short,
                "同一prepared Hb坐标内展示物理模型项、附加共同观测项和观测残差，并对照冻结增益控制；绿色可见前缀和金色评分区显式分开。成分命名不等于来源确证。", source, stats)

            fig, axes = plt.subplots(2, 1, figsize=(12., 5.), sharex=True)
            fig.subplots_adjust(left=.09, right=.985, top=.80, bottom=.23, hspace=.35)
            fig.suptitle("EEG-owned driver → vascular model states | " + compact,
                         x=.09, y=.982, ha="left", fontsize=16, weight="bold")
            fig.text(.09, .915, f'Frozen {row["chosen_arm"]}: gain={selected_spec["gain"]:g}, '
                     f'τn={selected_spec["tau_n"]:g}s, τv={selected_spec["tau_v"]:g}s · {role}', fontsize=12, color="#4e5965")
            axes[0].plot(t, modes[:, 0], color="#989fb0", lw=1.1, label="EEG mode a")
            axes[0].plot(t, modes[:, 1], color="#68aaa0", lw=1.1, label="EEG mode b")
            axes[0].plot(t, response["driver"], color="#7b4ab5", lw=1.6, label="Fixed driver d")
            axes[0].plot(t, states[:, 0], color="#cf8750", lw=1.4, ls="--", label="Vascular input u")
            axes[0].set_ylabel("Driver / modes\nmodel coord.")
            axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=4,
                           frameon=False, borderaxespad=0)
            _wave_style(axes[0])
            for j, label, color in ((2, "f−1", "#d1495b"), (3, "v−1", "#386cb0"),
                                    (4, "p−1", "#7b4ab5"), (5, "q−1", "#128c8d")):
                axes[1].plot(t, states[:, j] - 1, label=label, color=color, lw=1.3)
            axes[1].plot(t, states[:, 1], label="s", color="#7e7e7e", lw=1.1, ls=":")
            axes[1].set_ylabel("States\nrelative coord.")
            axes[1].legend(loc="lower left", bbox_to_anchor=(0, 1.025), ncol=5,
                           frameon=False, borderaxespad=0)
            axes[1].set_xlabel("Seconds from the registered window start")
            _wave_style(axes[1])
            fig.text(.09, .060, "Green: Hb input 0–10s; gold: score 20–30s. Driver and states use model coordinates, separate from Hb units.", fontsize=11.5, color="#4e5965")
            fig.text(.09, .017, "Full-window processing is offline; these trajectories do not establish a causal forecast or unique physiological source.", fontsize=11.5, color="#4e5965")
            state_fig = _wave_save(fig, f"waveform_{number:02d}_{chosen['role']}_states_wide", "EEG驱动与血管状态：" + short,
                "固定EEG-only驱动d与实际血管输入u显式区分；相对状态f/v/p/q以及s单列，不能与Hb直接相加。保留前缀/评分区，采用保存的拟合状态，不新增拟合。", source, stats)

            fig, gridaxes = plt.subplots(2, 3, figsize=(12., 5.))
            fig.subplots_adjust(left=.06, right=.985, top=.80, bottom=.235, hspace=.58, wspace=.27)
            axes = gridaxes.ravel()
            fig.suptitle("EEG log-power components | " + compact,
                         x=.06, y=.982, ha="left", fontsize=15.5, weight="bold")
            fig.text(.06, .915, f'Channel {ref["eeg_channels"][0]} · a: broad band; b: α/β contrast · observed = a + b + residual',
                     fontsize=12, color="#4e5965")
            for band, ax in enumerate(axes[:5]):
                for values, label, color, style, width in (
                    (eeg_prepared, "Observed", palette["observed"], "-", 1.5),
                    (eeg_a, "a contribution", palette["physical"], "-", 1.05),
                    (eeg_b, "b contribution", "#128c8d", "-", 1.05),
                    (eeg_prediction, "a + b prediction", palette["prediction"], "-", 1.15),
                    (eeg_residual, "Observed − predicted", palette["component"], "--", 1.05)):
                    ax.plot(t, values[:, band], color=color, label=label, ls=style, lw=width)
                low, high = cfg["tensor"]["eeg_bands_hz"][band]
                ax.set_title(f'{low}–{high} Hz', loc="left", pad=3)
                ax.set_ylabel("Δ log power" if band in (0, 3) else "")
                _wave_style(ax)
                ax.set_xticks([0, 10, 20, 30])
                if band >= 3:
                    ax.set_xlabel("Window time (s)")
            axes[5].axis("off")
            handles, labels = axes[0].get_legend_handles_labels()
            axes[5].legend(handles, labels, loc="upper left", frameon=False, fontsize=11.5, borderaxespad=0)
            fig.text(.06, .064, "All 30 features verified. Green: paired Hb input 0–10s; gold: score 20–30s. EEG uses the full processed window.", fontsize=11.5, color="#4e5965")
            fig.text(.06, .018, "These contributions add in prepared log-power coordinates. They do not add in native voltage; residual is not certified noise.",
                     fontsize=11.5, color="#4e5965")
            eeg_fig = _wave_save(fig, f"waveform_{number:02d}_{chosen['role']}_eeg_components_wide", "EEG频带成分分离：" + short,
                "从保存的EEG-only a/b与固定谱载荷，在prepared相对log-power坐标恢复贡献；观测=a贡献+b贡献+残差。示图为与原始电压相同的一条通道的五频带，全30维重构均核验。电压波形不能与这些特征贡献相加。", source, stats)
            transforms = [
                dict(modality="EEG", operation="native reader decoding; no artifact cleanup added", unit=eeg_unit,
                     reference=raw_eeg.reference, evidence=raw_eeg.unit_evidence),
                dict(modality="EEG", operation="30s local fourth-order Butterworth SOS forward/backward bandpass; mean squared power in nonoverlapping 0.25s bins; natural log; first20-bin mean removed",
                     bands_hz=cfg["tensor"]["eeg_bands_hz"], feature_order="channel_major_band_minor", power_floor="none; all powers strictly positive"),
                dict(modality="Hb", operation=raw_hb["provenance"]["transform"], raw_provenance=raw_hb["provenance"]),
                dict(modality="Hb", operation="record-relative linear interpolation to 300 samples at10Hz; window-local 0.01–0.2Hz third-order SOS forward/backward filter; polyphase resample2/5 to4Hz; first20-bin mean removed", unit=hb_unit),
                dict(modality="EEG/Hb", operation="retained new-training common positive scalar; report Hb inverse-scaled to parent prepared coordinates",
                     eeg_factor=cal["eeg_factor"], hb_factor=hb_factor, cal_path=str(run / "calibration" / (item["calibration_key"] + ".json")), fitted_now=False),
                dict(modality="model", operation="retained EEG-only modes; retained fixed-driver response, prefix-fit initial state and common term; no new inference or fitting", state_order=["u", "s", "f", "v", "p", "q"], spec=selected_spec),
                dict(modality="EEG_components", operation="native baseline operator times retained a/b outer products with fixed unit-norm spectral loading; inverse retained EEG scale; residual=prepared−predicted",
                     loading=loading.tolist(), displayed_feature_indices=list(range(5)),
                     saved_model_residual_convention="prediction−observed; report displays observed−prediction", all30features_verified=True),
            ]
            cases.append(dict(**chosen, ref=ref, split="evaluation", native_id=item["native_id"],
                chosen_reason="dataset regional-row gain_selected minus best_stable NRMSE at40-pointprefix; nearest median or minimum delta; lexicographic ID tie break; descriptive illustrations selected after evaluation",
                identity_owner=str(run / "measured_plan.json"), sources=source, source_record_join=join,
                eeg=dict(channel_names=ref["eeg_channels"], displayed_native_channel=ref["eeg_channels"][0],
                         native_rate_hz=raw_eeg.sample_rate_hz, source_unit=raw_eeg.native_unit,
                         displayed_unit=eeg_unit, unit_evidence=raw_eeg.unit_evidence,
                         requested_start_s=ref["eeg_start_s"], actual_start_s=eeg_start / raw_eeg.sample_rate_hz,
                         native_sample_indices=[eeg_start, eeg_start + eeg_length], plotted_native_samples=eeg_length),
                hb=dict(pair=ref["hb_channel"], native_rate_hz=raw_hb["sample_rate_hz"],
                        raw_unit=raw_hb["provenance"]["native_unit"], prepared_unit=hb_unit,
                        clock_start_s=ref["hb_start_s"], actual_native_support_s=[float(native_hb_time[0]), float(native_hb_time[-1])],
                        plotted_native_samples=len(native_hb_time), raw_semantics=raw_label),
                transforms=transforms, checks=checks, metrics=stats,
                visible_hb_interval_s=[0, prefix_s], scored_hb_interval_s=[score_start_s, score_stop_s],
                event_window_time_s=5., figures=[raw_fig["key"], component_fig["key"], state_fig["key"], eeg_fig["key"]]))
    result = dict(schema="semantic_response_discovery_waveform_evidence_v1", figures=figures, cases=cases,
                  run=str(run), parent=str(parent), native_windows=120, evaluation_subjects=15,
                  regional_evaluation_rows=360, example_cases=len(cases), fitted_now=False,
                  raw_loader_owner="src/data/unified_physiology.py",
                  data_contract="docs/DATA_CONTRACT.md", export_version="wide_final",
                  retained_draft=dict(source=str(out / "build_sources" / "waveform_section_draft.py"),
                                      provenance=str(out / "waveform_provenance_draft.json"), figures_suffix="without _wide"),
                  slide_frame_inches=[12., 4.72], minimum_scaled_font_size_pt=min(f["minimum_scaled_font_size_pt"] for f in figures),
                  seconds=time.monotonic() - started)
    (out / "waveform_provenance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
