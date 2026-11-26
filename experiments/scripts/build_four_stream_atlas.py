"""Export the first four sensor layers from completed frozen audit tables."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis.four_stream_atlas import (
    PERSON,
    check_partitions,
    hb_member_slopes,
    paired_readout,
    spatial_counts,
    summarize_sites,
)
from src.utils.io import write_json, write_yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "experiments/configs/physiology_semantic_tokenizer/four_stream_atlas_v1.yaml"
LAYERS = ("EEG-S", "EEG-O", "HB-S", "HB-O")
DATASETS = {
    "eeg_fnirs_single_trial": "Single-Trial",
    "simultaneous_eeg_nirs": "Simultaneous",
    "refed": "REFED",
    "visual_cognitive_motivation": "Visual",
}
READ = {"dtype": {"subject": str}}


def plot_atlas(nodes, members, readouts, contrasts, output):
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})
    shown = nodes.query("split == 'validation' and family == 4")
    datasets = sorted(shown.dataset.unique())
    fig, axes = plt.subplots(
        len(datasets), 4, figsize=(9, 2.6 * len(datasets)), layout="constrained", squeeze=False
    )
    for (row, ds), axes_row in zip(enumerate(datasets), axes):
        for ax, layer in zip(axes_row, LAYERS):
            sites = shown.loc[(shown.dataset == ds) & (shown.layer == layer)]
            ax.add_patch(plt.Circle((0, 0), 1, fc="#eeeeee", ec="#a6a6a6"))
            ax.plot([-0.1, 0, 0.1], [0.995, 1.08, 0.995], color="#999999")
            sc = ax.scatter(
                sites.x,
                sites.y,
                c=sites.occupancy,
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=115,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.13), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(layer)
            if layer == LAYERS[0]:
                ax.set_ylabel(DATASETS[ds])
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.65, label="Q4 占有效两秒片段的比例")
    fig.suptitle(
        "四流高属性词族 Q4 在开发被试传感器上的占用率\n"
        "每个点为原生位置，先在人内求比例再等权平均；灰色为未绘制区域"
    )
    fig.savefig(output / "sensor_occupancy.png", dpi=230)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.7), sharex=True, sharey=True, layout="constrained")
    for axes_row, layer in zip(axes, ["HB-S", "HB-O"]):
        for ax, tag, title in zip(axes_row, ["low", "middle", "high"], ["低频词", "中频词", "高频词"]):
            data = members.loc[
                (members.split == "validation")
                & (members.group == tag)
                & (members.layer == layer)
                & (members.signal == "X")
            ]
            points = data.groupby(PERSON)[["HbO_slope", "HbR_slope"]].median()
            ax.axhline(0, color="0.7", lw=0.8)
            ax.axvline(0, color="0.7", lw=0.8)
            ax.scatter(points.HbO_slope, points.HbR_slope, color="#087e81", alpha=0.8)
            ax.set_title(f"{layer}  {title} {data.word.iat[0]}  ·  {len(points)} 人")
    fig.suptitle(
        "训练期选词后，开发被试实测 HbO 与 HbR 的双色趋势\n"
        "低、中、高按训练出现次数选词；每点为一人所留成员的两秒斜率中位数"
    )
    fig.supxlabel("HbO 斜率（训练输入尺度 / 秒）")
    fig.supylabel("HbR 斜率（训练输入尺度 / 秒）")
    fig.savefig(output / "hb_member_trends.png", dpi=230)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, contrast in zip(axes, contrasts):
        mod = contrast["modality"]
        vals = readouts.loc[readouts.modality.eq(mod)]
        for i, (ds, group) in enumerate(vals.groupby("dataset")):
            ax.scatter(group.difference, i + np.linspace(-0.14, 0.14, len(group)), c="#245a86", s=28)
        ax.set_yticks(range(vals.dataset.nunique()), [DATASETS[k] for k in sorted(vals.dataset.unique())])
        ax.axvline(0, color="0.45", lw=0.8)
        ax.set_xlabel("S 单支 MSE 减去 S+O 的 MSE")
        metric = "EEG α 功率占比" if mod == "eeg" else "HbO 斜率"
        ax.set_title(f"{metric}  ·  {contrast['improved']}/{contrast['people']} 人改善")
    fig.suptitle(
        "同一两秒片段中，O 对实测属性读出的补充\n"
        "每点为一名开发被试；正值表示误差下降，两幅图按各自目标刻度显示"
    )
    fig.savefig(output / "so_readout.png", dpi=230)
    plt.close(fig)


def build(cfg, output):
    if (
        cfg["scope"] != ["train", "validation"]
        or cfg["shape"] != dict(sites_per_modality=4, bins_per_window=3, patches_per_bin=5, families=4)
        or cfg["support"]
        != dict(token_stride_s=2, context_s=30, preprocessing="offline_full_record", space="sensor_topview")
    ):
        raise ValueError("unsupported atlas scope, shape or information support")
    source = ROOT / cfg["source"]
    summary = json.loads((source / "summary.json").read_text())
    if (summary["experiment_id"], summary["status"], summary["seed"], summary["dimension"]) != (
        cfg["source_id"],
        "completed",
        cfg["seed"],
        cfg["dimension"],
    ):
        raise ValueError("frozen source identity mismatch")
    panel = pd.read_csv(source / "paired_panel.csv", **READ)
    positions = pd.read_csv(source / "positions.csv", **READ)
    with np.load(source / "paired_features.npz") as archive:
        counts = spatial_counts(panel, positions, archive)
    people, nodes = summarize_sites(counts)
    catalog, examples, slopes, scores, contrasts = [], [], [], [], []
    partitions = [panel[[*PERSON, "record", "split"]]]
    for mod, size in cfg["vocabulary"].items():
        folder = source / f"{mod}_{cfg['seed']}"
        for branch in "SO":
            family = json.loads((folder / f"{branch}_family.json").read_text())
            words = pd.read_csv(folder / f"{branch}_words.csv", float_precision="round_trip")
            mapping = np.asarray(family["mapping"])
            np.testing.assert_array_equal(words.id, np.arange(size))
            np.testing.assert_array_equal(mapping, words.family)
            np.testing.assert_array_equal(
                mapping, np.searchsorted(family["thresholds"], words.family_score, side="right")
            )
            layer = mod.upper() + "-" + branch
            catalog.append(
                words[["id", "train_patches", "train_people", "family_score"]].assign(
                    layer=layer,
                    family=mapping + 1,
                    attribute=family["attribute"],
                    definition_source=f"{folder.relative_to(ROOT)}/{branch}_family.json",
                )
            )
            manifest = pd.read_csv(folder / f"{branch}_members.csv", **READ)
            check_partitions(manifest)
            partitions.append(manifest[[*PERSON, "record", "split"]])
            members = manifest.loc[manifest.kind.eq("member")].copy()
            members["family"] = mapping[members.word] + 1
            members["layer"] = layer
            members["wave_file"] = f"{folder.relative_to(ROOT)}/{branch}_member_waves.npz"
            members["wave_key"] = (
                members.group + "_" + members.word.astype(str) + "_" + members.split + "_member"
            )
            examples.append(members)
            if mod == "hb":
                with np.load(folder / f"{branch}_member_waves.npz") as waves:
                    slopes.append(hb_member_slopes(manifest, waves, mapping, branch))
        subjects, result = paired_readout(pd.read_csv(folder / "conditional_subject.csv", **READ), mod)
        historical = summary["models"][mod]["O_gain_given_S"]
        np.testing.assert_allclose(result["relative_gain"], historical, rtol=1e-8)
        scores.append(subjects)
        contrasts.append(result)
    roster = pd.concat(partitions).drop_duplicates()
    check_partitions(roster)
    readouts = pd.concat(scores, ignore_index=True)
    dev = roster.loc[roster.split.eq("validation"), PERSON].drop_duplicates()
    if len(readouts.merge(dev, on=PERSON, validate="many_to_one")) != len(readouts):
        raise ValueError("readout outside development population")
    slopes = pd.concat(slopes, ignore_index=True)
    coverage = panel.groupby(["dataset", "split", "status"], as_index=False).agg(
        windows=("parent", "size"), subjects=("subject", "nunique")
    )
    for mod in ("eeg", "hb"):
        panel[f"{mod}_encoder_stop_s"] = panel[f"{mod}_start"] + cfg["support"]["context_s"]
    for name, frame in {
        "window_support": panel,
        "panel_coverage": coverage,
        "records": roster,
        "family_dictionary": pd.concat(catalog),
        "members": pd.concat(examples),
        "window_emissions": counts,
        "subject_occupancy": people,
        "atlas_nodes": nodes,
        "hb_member_slopes": slopes,
        "so_subjects": readouts,
        "so_contrasts": pd.DataFrame(contrasts),
    }.items():
        if name in {"window_emissions", "members", "family_dictionary"}:
            frame.to_csv(output / f"{name}.csv.gz", index=False)
        else:
            frame.to_csv(output / f"{name}.csv", index=False)
    plot_atlas(nodes, slopes, readouts, contrasts, output)
    write_yaml(output / "resolved.yaml", cfg)
    write_json(
        output / "summary.json",
        dict(
            experiment_id=cfg["experiment_id"],
            status="completed",
            parent=cfg["source"],
            layers=list(LAYERS),
            sensor_nodes=len(nodes.query("split=='validation'")),
            primary="atlas_nodes.csv",
            support=cfg["support"],
            panel=coverage.to_dict("records"),
            same_patch_readout=contrasts,
            future_prediction="unscored",
            cortical_mapping="unavailable",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh version in the four_stream_atlas suite")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONTRACT.read_text())
    default = ROOT / cfg["output"]
    dest = (args.output or default).resolve()
    if dest.parent != default.parent or dest.exists():
        parser.error("output must be a fresh run in the four_stream_atlas suite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()
        build(cfg, stage)
        stage.rename(dest)
    print(dest)


if __name__ == "__main__":
    main()
