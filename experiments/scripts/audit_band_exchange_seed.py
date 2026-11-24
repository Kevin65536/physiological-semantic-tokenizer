"""Compare independent EEG word dictionaries using fit-only role-wise label alignment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from evaluate_joint_tokenizer import write
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch
from src.tokenizers.band_exchange_words import decode_native

STATE = {}
FIELDS = {"power_4_8": (0, 1), "power_8_13": (1, 2), "rms_envelope": (2, 10)}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def align_role_words(reference, other, atom_count=1023, roles=5):
    """Fit a bijection from other IDs to reference IDs using only supplied fit members."""
    if (
        reference.shape != other.shape
        or reference.dtype.kind not in "iu"
        or other.dtype.kind not in "iu"
    ):
        raise ValueError("matched integer member arrays required")
    capacity = atom_count * roles + 1
    if (
        np.any(reference < 0)
        or np.any(other < 0)
        or np.any(reference >= capacity)
        or np.any(other >= capacity)
    ):
        raise ValueError("integer outside paid alphabet")
    reference_role = np.where(reference > 0, (reference.astype(np.int32) - 1) // atom_count, -1)
    other_role = np.where(other > 0, (other.astype(np.int32) - 1) // atom_count, -1)
    np.testing.assert_array_equal(reference_role, other_role)
    mapping = np.zeros(capacity, np.uint16)
    for role in range(roles):
        keep = reference_role == role
        first = reference[keep].astype(np.int64) - 1 - role * atom_count
        second = other[keep].astype(np.int64) - 1 - role * atom_count
        counts = np.bincount(first * atom_count + second, minlength=atom_count**2).reshape(
            atom_count, atom_count
        )
        original, new = linear_sum_assignment(counts, maximize=True)
        mapping[1 + role * atom_count + new] = 1 + role * atom_count + original
    np.testing.assert_array_equal(np.sort(mapping), np.arange(capacity))
    return mapping


def member_rates(reference, other, mapping, seed, atom_count=1023):
    rng = np.random.default_rng(seed)
    result = []
    for role in range(5):
        first_code = 1 + role * atom_count
        keep = (reference >= first_code) & (reference < first_code + atom_count)
        original = reference[keep].astype(np.int64) - first_code
        new = other[keep].astype(np.int64) - first_code
        counts = np.bincount(original * atom_count + new, minlength=atom_count**2).reshape(
            atom_count, atom_count
        )
        assignment = mapping[first_code : first_code + atom_count] - first_code
        n = int(counts.sum())
        fit_independence = float(np.sum(counts.sum(1)[assignment] * counts.sum(0)) / max(n * n, 1))
        random = [
            counts[rng.permutation(atom_count), np.arange(atom_count)].sum() / max(n, 1)
            for _ in range(100)
        ]
        result.append(
            dict(
                role=role,
                complete_patches=n,
                aligned_matches=int(counts[assignment, np.arange(atom_count)].sum()),
                match_fraction=float(counts[assignment, np.arange(atom_count)].sum() / max(n, 1)),
                marginal_independence_fraction=fit_independence,
                random_role_permutation_mean=float(np.mean(random)),
                random_role_permutation_P95=float(np.quantile(random, 0.95)),
            )
        )
    return result


def read_codes(directory, fold):
    examples = []
    output = {b: [] for b in "SO"}
    for dataset in STATE["names"]:
        with np.load(directory / dataset / f"fold_{fold}_ids.npz") as archive:
            examples.append(archive["examples"])
            for b in "SO":
                output[b].append(archive["exchange_balanced_" + b])
    np.testing.assert_array_equal(np.concatenate(examples), STATE["bank"]["examples"])
    return {b: np.concatenate(v) for b, v in output.items()}


def window_errors(predictions, truth, valid):
    errors = np.zeros((*valid.shape, len(predictions), 3), np.float64)
    errors[valid] = np.stack(
        [feature_means(np.square(value - truth), FIELDS) for value in predictions], axis=1
    )
    mean = errors.sum(1) / valid.sum(1).clip(1)[:, None, None]
    mean[~valid.any(1)] = np.nan
    return mean


def fold_review(fold):
    start = time.process_time()
    bank, original, replication = (STATE[k] for k in ("bank", "original", "replication"))
    first = read_codes(original, fold)
    second = read_codes(replication, fold)
    fit, held = bank["fold"] != fold, bank["fold"] == fold
    if set(bank["subject"][fit]) & set(bank["subject"][held]):
        raise ValueError("held person used to align labels")
    fm, hv = bank["valid"] & fit[:, None], bank["valid"][held]
    background = bank["background"]
    capacity = 5116
    results = {}
    arrays = dict(examples=bank["examples"][held], held_indices=np.flatnonzero(held))
    for branch in "SO":
        mapping = align_role_words(first[branch][fm], second[branch][fm])
        inverse = np.argsort(mapping)
        arrays[branch + "_mapping_211_to_107"] = mapping
        attribute = bank[branch + "_attributes"]
        profiles = {}
        for name, code, directory in (
            ("107", first[branch], original),
            ("211", second[branch], replication),
        ):
            audit = SparsePatternAudit(capacity, STATE["backgrounds"], [], 10, FIELDS)
            audit.fit(code[fm], background[fm], attribute[fm])
            with np.load(directory / f"exchange_balanced_{branch}_word_metrics.npz") as retained:
                np.testing.assert_array_equal(audit.local.count, retained[f"fit_{fold}"])
            profiles[name] = audit
        bg, _, p107 = profiles["107"].read(first[branch][held][hv], background[held][hv])
        bg211, _, p211 = profiles["211"].read(second[branch][held][hv], background[held][hv])
        # Unseen background cells fall back to a global mean. Its grouped
        # sum follows the vocabulary order and can change by a few double-
        # precision ulps after a label permutation. Observed cells must remain exact.
        supported_background = profiles["107"].background.count[background[held][hv]] > 0
        np.testing.assert_array_equal(bg[supported_background], bg211[supported_background])
        np.testing.assert_allclose(bg, bg211, rtol=32 * np.finfo(float).eps, atol=0)
        background_roundoff = dict(
            maximum_absolute_difference=float(np.max(abs(bg - bg211))),
            differing_values=int(np.sum(bg != bg211)),
            unseen_background_patches=int(np.sum(~supported_background)),
            relative_tolerance_in_float64_eps=32,
            observed_background_values_bitwise_equal=True,
        )
        _, _, p211_as107 = profiles["107"].read(
            mapping[second[branch][held][hv]], background[held][hv]
        )
        _, _, p107_as211 = profiles["211"].read(
            inverse[first[branch][held][hv]], background[held][hv]
        )
        errors = window_errors([bg, p107, p211, p211_as107, p107_as211], attribute[held][hv], hv)
        for i, (label, directory) in enumerate((("107", original), ("211", replication)), start=1):
            with np.load(directory / f"exchange_balanced_{branch}_word_metrics.npz") as retained:
                np.testing.assert_allclose(
                    errors[:, [0, i]], retained["word"][held], rtol=2e-14, atol=1e-14
                )
        arrays[branch + "_errors"] = errors
        fit_rates = member_rates(
            first[branch][fm], second[branch][fm], mapping, 211 + 10 * fold + (branch == "O")
        )
        held_rates = member_rates(
            first[branch][held][hv],
            second[branch][held][hv],
            mapping,
            311 + 10 * fold + (branch == "O"),
        )
        for partition, rates in (("fit", fit_rates), ("held", held_rates)):
            if sum(v["complete_patches"] for v in rates) != int(
                (fm if partition == "fit" else hv).sum()
            ):
                raise ValueError(
                    "missing zero on scored complete signal; preserve its denominator explicitly"
                )
        results[branch] = dict(
            fit_roles=fit_rates,
            held_roles=held_rates,
            fit_people=sorted(map(int, np.unique(bank["subject"][fit]))),
            held_people=sorted(map(int, np.unique(bank["subject"][held]))),
            background_roundoff=background_roundoff,
        )
    np.savez_compressed(replication / f"seed_alignment_fold_{fold}.npz", **arrays)
    results["CPU_seconds"] = time.process_time() - start
    write(replication / f"seed_alignment_fold_{fold}.json", results)
    return results


def half_profile_fold(fold):
    """Match fit-person exposure for own-seed and transferred categorical templates."""
    bank = STATE["bank"]
    first = read_codes(STATE["original"], fold)
    second = read_codes(STATE["replication"], fold)
    fit = bank["fold"] != fold
    held = ~fit
    hv = bank["valid"][held]
    dataset = STATE["datasets"]
    half_labels = np.full(len(fit), -1, np.int8)
    groups = []
    for d in STATE["names"]:
        persons = np.unique(bank["subject"][fit & (dataset == d)])
        ranked = sorted(
            map(int, persons),
            key=lambda person: hashlib.sha256(
                json.dumps([211, fold, d, person]).encode()
            ).hexdigest(),
        )
        for rank, person in enumerate(ranked):
            half_labels[fit & (dataset == d) & (bank["subject"] == person)] = rank % 2
        groups.append(
            dict(
                dataset=d,
                ordered_fit_people=ranked,
                first_half_people=ranked[::2],
                second_half_people=ranked[1::2],
            )
        )
    assert np.all(half_labels[fit] >= 0) and np.all(half_labels[held] == -1)
    records = []
    vectors = {}
    with np.load(STATE["replication"] / f"seed_alignment_fold_{fold}.npz") as archive:
        mappings = {b: archive[b + "_mapping_211_to_107"] for b in "SO"}
    for half in (0, 1):
        fit_rows = half_labels == half
        if set(bank["subject"][fit_rows]) & set(bank["subject"][held]):
            raise ValueError("held person in half-fit attribute model")
        fitmask = bank["valid"] & fit_rows[:, None]
        for branch in "SO":
            profiles = {}
            support = {}
            for seed, codes in ((107, first[branch]), (211, second[branch])):
                audit = SparsePatternAudit(5116, STATE["backgrounds"], [], 10, FIELDS)
                audit.fit(
                    codes[fitmask], bank["background"][fitmask], bank[branch + "_attributes"][fitmask]
                )
                people = np.zeros(5116, np.int32)
                for person in np.unique(bank["subject"][fit_rows]):
                    mask = bank["valid"] & fit_rows[:, None] & (bank["subject"][:, None] == person)
                    people[np.unique(codes[mask])] += 1
                profiles[seed] = audit
                support[seed] = people
            mapping = mappings[branch]
            inverse = np.argsort(mapping)
            held107 = first[branch][held][hv]
            held211 = second[branch][held][hv]
            targets = bank[branch + "_attributes"][held][hv]
            bg, _, own107 = profiles[107].read(held107, bank["background"][held][hv])
            bg211, _, own211 = profiles[211].read(held211, bank["background"][held][hv])
            np.testing.assert_allclose(bg, bg211, rtol=32 * np.finfo(float).eps, atol=0)
            _, _, mapped211 = profiles[107].read(mapping[held211], bank["background"][held][hv])
            _, _, mapped107 = profiles[211].read(inverse[held107], bank["background"][held][hv])
            vectors[f"half{half}_{branch}_errors"] = window_errors(
                [bg, own107, own211, mapped211, mapped107], targets, hv
            )
            measured = {}
            for name, seed, words in [
                ("seed107", 107, held107),
                ("seed211", 211, held211),
                ("211_to_107", 107, mapping[held211]),
                ("107_to_211", 211, inverse[held107]),
            ]:
                good = (profiles[seed].local.count[words] >= 8) & (support[seed][words] >= 2)
                measured[name] = dict(
                    qualified_patches=int(good.sum()),
                    complete_held_patches=len(words),
                    qualified_fraction=float(good.mean()),
                    held_background_ID_cells_seen=int(
                        sum(
                            int(k) in profiles[seed].joint.count
                            for k in bank["background"][held][hv] * 5116 + words
                        )
                    ),
                )
            records.append(
                dict(
                    fold=fold,
                    half=half,
                    branch=branch,
                    fit_people=sorted(map(int, np.unique(bank["subject"][fit_rows]))),
                    fit_complete_patches=int(fitmask.sum()),
                    held_people=sorted(map(int, np.unique(bank["subject"][held]))),
                    support=measured,
                )
            )
    np.savez_compressed(
        STATE["replication"] / f"half_fit_profile_fold_{fold}.npz",
        held_indices=np.flatnonzero(held),
        **vectors,
    )
    write(
        STATE["replication"] / f"half_fit_profile_fold_{fold}.json",
        dict(split=groups, profiles=records),
    )
    return records


def summarize_halves():
    bank = STATE["bank"]
    vectors = {(half, b): np.empty((len(bank["examples"]), 5, 3)) for half in (0, 1) for b in "SO"}
    support = []
    for fold in range(4):
        with np.load(STATE["replication"] / f"half_fit_profile_fold_{fold}.npz") as archive:
            for half in (0, 1):
                for b in "SO":
                    vectors[half, b][archive["held_indices"]] = archive[f"half{half}_{b}_errors"]
        support.extend(
            json.loads((STATE["replication"] / f"half_fit_profile_fold_{fold}.json").read_text())[
                "profiles"
            ]
        )
    results = {}
    for b in "SO":
        # Each half-template has exactly the same evaluation people. Average the
        # two squared-error estimates per held position before applying the hierarchy.
        error = (vectors[0, b] + vectors[1, b]) / 2
        h = base_training_aggregate({"nmse": error}, bank["examples"], STATE["rows"])["metrics"][
            "nmse"
        ]
        a = np.asarray(h["macro"])
        results[b] = dict(
            order=["background", "seed107", "seed211", "211_to_107_template", "107_to_211_template"],
            gain=1 - a[1:] / a[0],
            hierarchy=h,
            qualified_support={
                name: sum(
                    v["support"][name]["qualified_patches"] for v in support if v["branch"] == b
                )
                / sum(
                    v["support"][name]["complete_held_patches"] for v in support if v["branch"] == b
                )
                for name in ("seed107", "seed211", "211_to_107", "107_to_211")
            },
        )
    full = json.loads((STATE["replication"] / "seed_alignment.json").read_text())["readout"]
    contribution = {}
    for label, stats in [
        ("full_fit", full["O"]["hierarchy"]),
        ("half_fit_equal_exposure", results["O"]["hierarchy"]),
    ]:
        by = {}
        for d, values in stats["by_dataset"].items():
            a = np.asarray(values)
            by[d] = dict(
                theta_gain_percent=dict(
                    zip(
                        ["seed107", "seed211", "211_to_107", "107_to_211"],
                        100 * (1 - a[1:, 0] / a[0, 0]),
                    )
                ),
                theta_211_to_107_additional_MSE_vs_own_seed211=float(a[3, 0] - a[2, 0]),
                macro_difference_contribution=float((a[3, 0] - a[2, 0]) / 4),
            )
        contribution[label] = by
    write(
        STATE["replication"] / "half_fit_profiles.json",
        dict(
            status="complete",
            split="per-library fit people SHA211 ranked then alternating halves; common original held people",
            mapping="original full-fit-person member-overlap bijection frozen; neither tables nor mapping read held targets",
            scoring="two equal-exposure half-table squared errors averaged per held position then original hierarchy",
            readout=results,
            O_theta_dataset_attribution=contribution,
            fold_support=support,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )
    return results


def summarize_alignment():
    bank = STATE["bank"]
    gathered = {b: np.empty((len(bank["examples"]), 5, 3)) for b in "SO"}
    folds = []
    for fold in range(4):
        with np.load(STATE["replication"] / f"seed_alignment_fold_{fold}.npz") as archive:
            idx = archive["held_indices"]
            for b in "SO":
                gathered[b][idx] = archive[b + "_errors"]
        folds.append(
            json.loads((STATE["replication"] / f"seed_alignment_fold_{fold}.json").read_text())
        )
    result = {}
    for b, values in gathered.items():
        h = base_training_aggregate({"nmse": values}, bank["examples"], STATE["rows"])["metrics"][
            "nmse"
        ]
        macro = np.asarray(h["macro"])
        result[b] = dict(
            order=[
                "background",
                "seed107",
                "seed211",
                "seed211_mapped_to_107_fit_template",
                "seed107_mapped_to_211_fit_template",
            ],
            gain=1 - macro[1:] / macro[0],
            hierarchy=h,
        )
        for partition in ("fit", "held"):
            n = sum(v["complete_patches"] for f in folds for v in f[b][partition + "_roles"])
            hit = sum(v["aligned_matches"] for f in folds for v in f[b][partition + "_roles"])
            random = (
                sum(
                    v["random_role_permutation_mean"] * v["complete_patches"]
                    for f in folds
                    for v in f[b][partition + "_roles"]
                )
                / n
            )
            result[b][partition + "_alignment"] = dict(
                complete_patches=n,
                aligned_fraction=hit / n,
                random_within_role_permutation_fraction=random,
            )
    write(
        STATE["replication"] / "seed_alignment.json",
        dict(
            status="complete",
            readout=result,
            scope="same source and geometry, independent sampling and atomic dictionary seed; fit-person maximum-overlap bijections; all transferred scores use held people",
            alignment_folds=folds,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )
    return result


def render(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = STATE["replication"]
    old = json.loads((STATE["original"] / "summary.json").read_text())
    new = json.loads((out / "summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, b in zip(axes[0], "SO"):
        gain = np.asarray(result[b]["gain"]) * 100
        x = np.arange(3)
        for j, label in enumerate(
            ("seed 107", "seed 211", "211 -> 107 template", "107 -> 211 template")
        ):
            ax.bar(x + (j - 1.5) * 0.2, gain[j], width=0.19, label=label)
        ax.axhline(0, color="black", lw=0.6)
        ax.set(
            xticks=x,
            xticklabels=["theta 4-8 Hz", "alpha 8-13 Hz", "local RMS"],
            ylabel="Prediction error reduction (%)",
            title=f"Actual paid {b} ID added to dataset/site/X-RMS background",
        )
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    for j, (summary, label) in enumerate(((old, "seed 107"), (new, "seed 211"))):
        nmse = np.ravel(
            summary["arms"]["exchange_balanced_energy"]["summary"]["metrics"]["nmse"]["macro"]
        )
        ax.bar(np.arange(3) + (j - 0.5) * 0.3, nmse, width=0.29, label=label)
    ax.set(
        xticks=np.arange(3),
        xticklabels=["X", "S", "O"],
        ylabel="NMSE / frozen training variance",
        title="All 400 arrays, 17040 positions and 60 train people",
    )
    ax.legend()
    ax = axes[1, 1]
    for j, b in enumerate("SO"):
        ax.bar(
            np.arange(2) + (j - 0.5) * 0.3,
            [result[b][p + "_alignment"]["aligned_fraction"] for p in ("fit", "held")],
            width=0.29,
            label=b,
        )
    ax.set(
        xticks=np.arange(2),
        xticklabels=["fit members", "held members"],
        ylabel="Fit-aligned ID agreement",
        ylim=(0, 1),
        title="Independent initialization; per-role bijections",
    )
    ax.legend()
    fig.savefig(out / "seed_comparison.png", dpi=230)
    plt.close(fig)
    selected = json.loads(
        (STATE["original"] / "visual_review_v1/selected_examples.json").read_text()
    )["examples"]
    cases = STATE["population"]["cases"]
    plots = []
    for dataset in STATE["names"]:
        case = next(v for v in selected if v["dataset"] == dataset and v["method"] == "allocated")
        basecase = next(
            v
            for v in selected
            if v["dataset"] == dataset
            and v["method"] == "energy"
            and (v["row"], v["start_s"], v["position"])
            == (case["row"], case["start_s"], case["position"])
        )
        physical = next(
            c
            for c in cases
            if c["dataset"] == dataset
            and (c["crop"]["row"], c["crop"]["start_s"]) == (case["row"], case["start_s"])
        )
        fold = physical["fold"]
        books = {b: [] for b in "SO"}
        for b in "SO":
            for role in range(5):
                with np.load(out / "books" / f"fold_{fold}_{b}_{role}.npz") as saved:
                    books[b].append({k: saved[k] for k in saved if k != "metadata"})
        with np.load(
            STATE["run"]
            / "component_source_transfer/adaptive_band_v1"
            / dataset
            / f"fold_{fold}_model.npz"
        ) as saved:
            covariance = {b: saved[b] for b in "SO"}
        with np.load(out / dataset / f"fold_{fold}_ids.npz") as saved:
            location = {tuple(e): i for i, e in enumerate(saved["examples"])}
            ids = np.array(
                [
                    location[physical["crop"]["row"], physical["crop"]["start_s"], p]
                    for p in physical["crop"]["positions"]
                ]
            )
            inv = np.argsort(physical["native"])
            streams = {b: saved["exchange_balanced_" + b][ids[inv]] for b in "SO"}
        predicted = decode_native(streams, dataset, books, covariance, None, "balanced", "energy")
        native = physical["native"][physical["crop"]["positions"].index(case["position"])]
        ex = np.array([[case["row"], case["start_s"], case["position"]]])
        batch = base_tokenizer_batch(STATE["rows"], ex, STATE["scales"])
        actual = joint_reconstruction_metrics(
            {k: v[native : native + 1] for k, v in predicted.items()},
            batch,
            1,
            np.array([case["scale"]]),
        )["metrics"]["nmse"][0]
        with np.load(out / dataset / f"fold_{fold}_metrics.npz") as saved:
            score_index = next(i for i, e in enumerate(saved["examples"]) if tuple(e) == tuple(ex[0]))
            np.testing.assert_array_equal(
                actual, saved["exchange_balanced_energy__nmse"][score_index]
            )
        originalfile = (
            STATE["original"]
            / "visual_review_v1"
            / dataset
            / f"example_{basecase['audit_index']:03d}.npz"
        )
        with np.load(originalfile) as retained:
            for key in ("x", "s", "o", "mask"):
                np.testing.assert_array_equal(retained[key], batch[key][0])
            mask = retained["mask"].reshape(-1).astype(bool)
            t = np.arange(6000) / 200
            peak = int(np.flatnonzero(mask)[np.argmax(abs(retained["x"].reshape(-1)[mask]))])
            local = np.arange(max(0, peak - 200), min(6000, peak + 201))
            fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
            for line, (key, branch) in enumerate((("x", "sum"), ("s", "S"), ("o", "O"))):
                for col, ii in enumerate((np.arange(6000), local)):
                    ax = axes[line, col]
                    for name, wave, style in (
                        ("target", retained[key], "-"),
                        ("seed 107", retained["exchange_" + branch], "--"),
                        ("seed 211", predicted[branch][native], ":"),
                    ):
                        v = np.where(mask, wave.reshape(-1) * case["scale"], np.nan)
                        ax.plot(t[ii], v[ii], style, label=name, lw=1.1)
                    for bound in np.arange(0, 30, 2):
                        if t[ii[0]] <= bound <= t[ii[-1]]:
                            ax.axvline(bound, color="#bbb", lw=0.5)
                    ax.set(
                        title=f"{key.upper()} / {'30 s' if col == 0 else 'target-peak local'}",
                        xlabel="Time within visible 30 s (s)",
                        ylabel=case["source_unit"],
                    )
            axes[0, 0].legend()
            fig.suptitle(
                f"{dataset} / {case['subject']} / {case['position_name']} / crop+{case['start_s']} s; paid K5116"
            )
            path = out / (dataset + "_fixed_hard_seed_comparison.png")
            fig.savefig(path, dpi=220)
            plt.close(fig)
            plots.append(
                dict(
                    path=path.name,
                    row=case["row"],
                    start_s=case["start_s"],
                    position=case["position"],
                    subject=case["subject"],
                    fold=fold,
                    source_unit=case["source_unit"],
                    NMSE=actual,
                    score_replayed_exact=True,
                )
            )
    write(
        out / "figure_manifest.json",
        dict(
            figures=[dict(path="seed_comparison.png"), *plots],
            state="pending_actual_open",
            selection="four original hard identities; full 30s and same true maximum-absolute-X local range; true masks and same native coordinates",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--replication", required=True, type=Path)
    parser.add_argument("--half-profiles", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    out = args.replication.resolve()
    original = run / "component_source_transfer/band_exchange_words_v1"
    if not (out / "summary.json").is_file():
        raise ValueError("finish independent dictionary fitting and original scoring first")
    pop = json.loads(
        (
            run / "component_source_transfer/hard_as_v1/local_information_v1/population.json"
        ).read_text()
    )
    with np.load(
        run / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz"
    ) as saved:
        bank = {k: saved[k] for k in saved}
    cfg = yaml.safe_load(
        (
            ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml"
        ).read_text()
    )
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    if (pop["cfg_identity"], pop["scales_identity"]) != (
        manifest["cfg_identity"],
        manifest["scales_identity"],
    ):
        raise ValueError("changed population identity")
    assert all(manifest["rows"][i]["split"] == "train" for i, t, p in bank["examples"])
    STATE.update(
        run=run,
        replication=out,
        original=original,
        population=pop,
        bank=bank,
        rows=manifest["rows"],
        scales=scales,
        names=sorted(pop["native_positions"]),
        backgrounds=4 * len(pop["background_sites"]),
        datasets=np.array([manifest["rows"][i]["dataset"] for i, t, p in bank["examples"]]),
    )
    if args.half_profiles:
        with ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            list(pool.map(half_profile_fold, range(4)))
        summarize_halves()
        return
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        list(pool.map(fold_review, range(4)))
    result = summarize_alignment()
    render(result)
    write(
        out / "alignment_source_identity.json",
        dict(
            script_sha256=digest(Path(__file__)),
            bank_sha256=digest(
                run
                / "component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz"
            ),
            source_and_seed_book_count=40,
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


if __name__ == "__main__":
    main()
