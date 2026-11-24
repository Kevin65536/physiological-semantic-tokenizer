"""Build an offline code-by-code atlas from frozen IDs and G0 train/screen targets."""

import argparse
import base64
from collections import OrderedDict
import csv
from html import escape
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_training_data,
)


def encoded(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode(
        "ascii"
    )


def derived(waves, modality):
    if modality == "eeg":
        window = np.hanning(400)
        spec = (
            2
            * abs(np.fft.rfft((waves - waves.mean(1, keepdims=True)) * window, axis=1))
            ** 2
            / (200 * np.square(window).sum())
        )
        env = np.sqrt(np.mean(waves.reshape(-1, 8, 50) ** 2, axis=2))
        return [waves, spec[:, 2:91], env]
    wave = waves.reshape(-1, 20, 2)
    return [
        wave[:, :, 0],
        wave[:, :, 1],
        wave[:, :, 0] - wave[:, :, 0].mean(1, keepdims=True),
        wave[:, :, 1] - wave[:, :, 1].mean(1, keepdims=True),
    ]


def prepare(rows, scales, examples, folder, split, modality, branch):
    with np.load(folder / (split + "_IDs.npz"), allow_pickle=False) as ids:
        np.testing.assert_array_equal(examples, ids["examples"])
        codes = ids[branch].copy()
        band = (
            ids["fixed_band"].copy()
            if "fixed_band" in ids.files
            else np.zeros(len(examples), int)
        )
    expect = "train" if split == "train" else "validation"
    if any(
        rows[i]["modality"] != modality or rows[i]["split"] != expect
        for i, _, _ in examples
    ):
        raise ValueError("frozen split/modality identity mismatch before arrays")
    all_w = []
    all_keys = []
    all_ids = []
    cache = OrderedDict()
    for start in range(0, len(examples), 64):
        take = examples[start : start + 64]
        batch = base_tokenizer_batch(
            rows, take, scales, array_cache=cache, cache_limit=64
        )
        complete = batch["mask"].all(-1)
        sample, token = np.nonzero(complete)
        all_w.append(batch[branch.lower()][sample, token])
        all_keys.append(np.column_stack((sample + start, token)).astype("int32"))
        # A keyed pair avoids confusing conditional vocabularies.
        all_ids.append(
            np.column_stack((band[sample + start], codes[sample + start, token]))
        )
    w = np.concatenate(all_w)
    keys = np.concatenate(all_keys)
    ids = np.concatenate(all_ids)
    order = np.lexsort((ids[:, 1], ids[:, 0]))
    ids = ids[order]
    groups = {}
    edges = np.r_[0, np.nonzero((ids[1:] != ids[:-1]).any(1))[0] + 1, len(ids)]
    for i, j in zip(edges[:-1], edges[1:]):
        groups[tuple(ids[i])] = slice(i, j)
    return w[order], keys[order], groups


def records(waves, members, examples, rows, modality, rng):
    if len(waves) == 0:
        return dict(count=0, subjects=0, members=[], panels=[])
    take = rng.choice(len(waves), min(len(waves), 3), replace=False)
    identities = []
    for j in take:
        n, t = int(members[j, 0]), int(members[j, 1])
        i, start, p = examples[n]
        row = rows[i]
        identities.append(
            dict(
                dataset=row["dataset"],
                subject=row["subject"],
                record=row["record"],
                position=row["positions"][p],
                parent=row["parent"],
                crop_s=int(start),
                patch=int(t),
            )
        )
    signals = derived(waves, modality)
    panels = []
    for a in signals:
        panels.append(
            dict(
                length=a.shape[1],
                rows=encoded(
                    np.vstack([a.mean(0), np.quantile(a, [0.1, 0.9], axis=0), a[take]])
                ),
            )
        )
    subjects = {
        (rows[examples[n, 0]]["dataset"], rows[examples[n, 0]]["subject"])
        for n in np.unique(members[:, 0])
    }
    return dict(
        count=len(waves), subjects=len(subjects), members=identities, panels=panels
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ["run", "eeg", "hb", "output"]:
        parser.add_argument("--" + k, type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if (out / "atlas_index.js").exists():
        raise FileExistsError("retained atlas already exists")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(
        (
            ROOT
            / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest["rows"]
    with np.load(args.run / "panels.npz") as f:
        panels = {
            (m, s): f[m + suffix].copy()
            for m in ["eeg", "hb"]
            for s, suffix in [("train", "_pattern_train"), ("validation", "_screen")]
        }
    catalog = {}
    totals = {}
    csvrows = []
    for m in ("eeg", "hb"):
        src = args.eeg if m == "eeg" else args.hb
        for br in "SO":
            label = m.upper() + "-" + br
            train = prepare(rows, scales, panels[m, "train"], src, "train", m, br)
            val = prepare(
                rows, scales, panels[m, "validation"], src, "validation", m, br
            )
            if m == "eeg":
                with np.load(src / "training_attribute_profiles.npz") as f:
                    sizes = list(f[br + "_train_count"].shape)
            else:
                with np.load(src / (br + "_training_prototypes.npz")) as f:
                    sizes = [1, len(f["count"])]
            keys = [
                (band, code)
                for band in range(sizes[0])
                for code in range(1 if m == "eeg" else 0, sizes[1])
            ]
            keys.sort(
                key=lambda k: (
                    -(train[2][k].stop - train[2][k].start) if k in train[2] else 0,
                    k,
                )
            )
            corpus = len(train[0])
            cumulative = 0
            band_catalog = []
            chunk = []
            rng = np.random.default_rng(43029)
            (out / label).mkdir(exist_ok=True)
            for idx, k in enumerate(keys):
                participants = []
                for split, (waves, indices, groups) in [
                    ("train", train),
                    ("validation", val),
                ]:
                    subset = groups.get(k, slice(0, 0))
                    participants.append(
                        records(
                            waves[subset],
                            indices[subset],
                            panels[m, split],
                            rows,
                            m,
                            rng,
                        )
                    )
                tr, va = participants
                before = cumulative
                cumulative += tr["count"]
                row = dict(
                    rank=idx + 1,
                    band=int(k[0]),
                    ID=int(k[1]),
                    train=tr["count"],
                    validation=va["count"],
                    train_subjects=tr["subjects"],
                    validation_subjects=va["subjects"],
                    mass=tr["count"] / corpus,
                    cumulative=cumulative / corpus,
                    major=before < 0.9 * corpus,
                    low_support=tr["count"] < 8 or tr["subjects"] < 2,
                    chunk=idx // 32,
                )
                band_catalog.append(row)
                csvrows.append(dict(branch=label, **row))
                chunk.append(dict(train=tr, validation=va))
                if len(chunk) == 32 or idx + 1 == len(keys):
                    (out / label / (str(idx // 32) + ".js")).write_text(
                        "window.atlasData="
                        + json.dumps(chunk, separators=(",", ":"))
                        + ";"
                    )
                    chunk = []
            if cumulative != corpus or sum(
                row["validation"] for row in band_catalog
            ) != len(val[0]):
                raise ValueError(
                    f"{label} contains an ID outside its declared observed vocabulary"
                )
            totals[label] = dict(
                vocabulary_per_band=sizes[1],
                fixed_bands=sizes[0],
                real_codes=len(keys),
                train_complete=corpus,
                validation_complete=len(val[0]),
                major_codes=sum(v["major"] for v in band_catalog),
                dead_codes=sum(v["train"] == 0 for v in band_catalog),
                profile=str(src),
            )
            catalog[label] = band_catalog
            print(label, totals[label], flush=True)
    (out / "atlas_index.js").write_text(
        "window.atlasIndex=" + json.dumps(catalog, separators=(",", ":")) + ";"
    )
    with (out / "all_codes.csv").open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
    (out / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    template = Path(__file__).with_name("joint_member_atlas.html").read_text()
    profiles = " · ".join(
        f"{label} {escape(os.path.relpath(folder, args.run))}"
        for label, folder in [("EEG", args.eeg), ("Hb", args.hb)]
    )
    (out / "index.html").write_text(template.replace("{{profile_sources}}", profiles))


if __name__ == "__main__":
    main()
