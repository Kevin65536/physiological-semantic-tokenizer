"""Train-subject parent-label distillation from unchanged semantic Hb integers."""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_joint_tokenizer import (
    write,
    input_scales,
    load_source as load_panel,
    display_selection,
)
from run_dual_component_tokenizer import identity, owner_lock
from review_joint_tokenizer import score
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_batch,
)
from src.analysis.dual_component_tokenizer import (
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
)
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.tokenizers.contextual_latent_dictionary import load_model as load_current
from src.tokenizers.hb_parent_leaf import load_parent
from src.tokenizers.hb_parent_distillation import (
    validate_ids,
    fit_tables,
    parent_probabilities,
    decode_parent,
    preserve_O_mix_HbO,
    load_model,
)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def ranked(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parent_source(cfg, manifest, scales):
    config = yaml.safe_load((ROOT / cfg["source"]["parent_config"]).read_text())
    saved = torch.load(
        ROOT / cfg["source"]["parent"], map_location="cpu", weights_only=False
    )
    expected = dict(modality="hb", architecture="waveform", candidate="vq128")
    if (
        saved["step"] != 10000
        or saved["signature"]["config"] != identity(config)
        or saved["signature"]["input"] != identity(manifest)
        or saved["signature"]["scales"] != identity(scales)
        or saved["signature"]["task"] != expected
    ):
        raise ValueError("exact original G1 parent identity required")
    setting = dict(
        config["model"],
        **config["architectures"]["waveform"],
        input_dim=40,
        channels=2,
        sample_strides=[5],
    )
    meta = dict(schema="original_G1_Hb_parent", modality="hb", codebook_size=128)
    return load_parent(ROOT / cfg["source"]["parent"], setting, meta), setting, meta


def encode_parent(parent, x):
    with torch.inference_mode():
        z = parent.model.encode_continuous(torch.as_tensor(x, dtype=torch.float32))
        return {
            b: parent.model.branches[b]["quantizer"]
            ._assignment_logits(value.reshape(-1, value.shape[-1]))
            .argmax(-1)
            .reshape(value.shape[:2])
            .numpy()
            for b, value in z.items()
        }


def audit(prediction, bank, examples, rows, scales):
    result = joint_reconstruction_metrics(
        prediction, bank, 2, input_scales(rows, examples, scales)
    )
    if result["support"]["nonfinite_prediction_points"].any():
        raise ValueError("nonfinite waveform retained as failure")
    return result, joint_reconstruction_summary(
        result["metrics"], result["support"], examples.tolist(), rows
    )


def fold_assignment(examples, rows, cfg):
    folds = np.empty(len(examples), np.int8)
    for dataset in sorted({rows[i]["dataset"] for i, _, _ in examples}):
        subjects = sorted(
            {
                rows[i]["subject"]
                for i, _, _ in examples
                if rows[i]["dataset"] == dataset
            },
            key=lambda s: ranked([cfg["training"]["fold_seed"], dataset, s]),
        )
        for j, subject in enumerate(subjects):
            folds[
                [
                    rows[i]["dataset"] == dataset and rows[i]["subject"] == subject
                    for i, _, _ in examples
                ]
            ] = j % cfg["training"]["folds"]
    return folds


def batched_parent(parent, ids, probability, mode, size):
    arrays = defaultdict(list)
    for start in range(0, len(ids["S"]), size):
        end = start + size
        result = decode_parent(
            parent, {b: ids[b][start:end] for b in "SO"}, probability[start:end], mode
        )
        for b, value in result.items():
            arrays[b].append(value)
    return {b: np.concatenate(v) for b, v in arrays.items()}


def eligibility(summary, baseline, g1, cfg):
    m, b, ref = summary["metrics"], baseline["metrics"], g1["metrics"]
    rule = cfg["selection"]
    nm = m["nmse"]
    source = b["nmse"]
    gain = 1 - nm["macro"][0][0] / source["macro"][0][0]
    changes = {
        d: 1 - value[0][0] / source["by_dataset"][d][0][0]
        for d, value in nm["by_dataset"].items()
    }
    before = {
        (r["dataset"], r["subject"]): r["value"][0][0] for r in source["by_subject"]
    }
    improved = sum(
        r["value"][0][0] < before[r["dataset"], r["subject"]] for r in nm["by_subject"]
    )
    checks = dict(
        X_improvement=gain >= rule["minimum_CV_X_HbO_relative_improvement"],
        improved_datasets=sum(v > 0 for v in changes.values())
        >= rule["minimum_improved_datasets"],
        Visual=changes["visual_cognitive_motivation"] > 0,
        dataset_X_guard=min(changes.values())
        >= 1 - rule["maximum_dataset_X_HbO_degradation_ratio"],
        improved_subjects=improved / len(before)
        >= rule["minimum_improved_subject_fraction"],
        macro_component_NMSE=bool(
            np.all(
                np.asarray(nm["macro"])
                <= rule["component_color_NMSE_ratio_to_G1_max"]
                * np.asarray(ref["nmse"]["macro"])
            )
        ),
        dataset_component_NMSE=all(
            np.all(
                np.asarray(v)
                <= rule["dataset_component_color_NMSE_ratio_to_G1_max"]
                * np.asarray(ref["nmse"]["by_dataset"][d])
            )
            for d, v in nm["by_dataset"].items()
        ),
    )
    for key in ["macro", *nm["by_dataset"]]:
        get = lambda field: np.asarray(
            m[field]["macro"] if key == "macro" else m[field]["by_dataset"][key]
        )
        rms = get("rms_gain")
        checks["RMS_" + key] = bool(
            np.all(
                (rms >= rule["rms_gain_interval"][0])
                & (rms <= rule["rms_gain_interval"][1])
            )
        )
        checks["slope_" + key] = bool(
            np.all(get("slope_correlation") >= rule["slope_correlation_min"])
        )
        checks["curvature_" + key] = bool(
            np.all(
                get("curvature_correlation")
                > rule["curvature_correlation_min_exclusive"]
            )
        )
        checks["boundary_" + key] = bool(
            np.all(
                get("boundary_jump_difference_nmse")
                <= rule["boundary_jump_difference_nmse_max"]
            )
        )
    checks["positive_correlation"] = bool(
        np.all(np.asarray(m["correlation"]["macro"]) > 0)
    )
    return dict(
        eligible=all(checks.values()),
        checks=checks,
        relative_X_HbO_improvement=gain,
        dataset_improvements=changes,
        improved_subjects=improved,
        subjects=len(before),
    )


def prepare(cfg, out, manifest, scales, current, parent):
    bank_path = ROOT / cfg["source"]["bank"]
    if digest(bank_path) != cfg["source"]["bank_sha256"]:
        raise ValueError("frozen training bank changed")
    with np.load(bank_path) as stored:
        examples = stored["examples"].copy()
        if any(
            manifest["rows"][i]["split"] != "train"
            or manifest["rows"][i]["modality"] != "hb"
            for i, _, _ in examples
        ):
            raise ValueError("train-only bank boundary before signal arrays")
        bank = {key: stored[key].copy() for key in stored.files if key != "examples"}
    subjects = {
        (manifest["rows"][i]["dataset"], manifest["rows"][i]["subject"])
        for i, _, _ in examples
    }
    if len(subjects) != 60 or len(examples) != 944:
        raise ValueError("exact 944/60 frozen population required")
    a, b = current.source.model.state_dict(), parent.model.state_dict()
    shared = [key for key in a if key.startswith("branches.S.") or key == "position"]
    if any(key not in b or not torch.equal(a[key], b[key]) for key in shared):
        raise ValueError("ordinary S encoder and decoder must remain exactly G1")
    teachers = defaultdict(list)
    original = defaultdict(list)
    actual = defaultdict(list)
    ids = {b: bank["ids_" + b] for b in "SO"}
    _, quiet = validate_ids(ids)
    for start in range(0, len(examples), cfg["resources"]["batch_size"]):
        stop = start + cfg["resources"]["batch_size"]
        x = bank["x"][start:stop]
        teacher = encode_parent(parent, x)
        emitted = current.encode(x)
        for branch in "SO":
            np.testing.assert_array_equal(emitted[branch], ids[branch][start:stop])
            if branch == "S":
                np.testing.assert_array_equal(
                    teacher[branch][~quiet[start:stop]],
                    emitted[branch][~quiet[start:stop]],
                )
            teachers[branch].append(teacher[branch])
        for branch, value in parent.decode_ids(teacher).items():
            original[branch].append(value)
        for branch, value in current.decode_ids(emitted).items():
            np.testing.assert_array_equal(
                value, bank["prediction_" + branch][start:stop]
            )
            actual[branch].append(value)
    teacher_ids = {b: np.concatenate(v) for b, v in teachers.items()}
    teacher = {b: np.concatenate(v) for b, v in original.items()}
    prediction = {b: np.concatenate(v) for b, v in actual.items()}
    np.savez_compressed(
        out / "teacher_training.npz",
        examples=examples,
        **{"teacher_id_" + b: v for b, v in teacher_ids.items()},
        **{"teacher_" + b: v for b, v in teacher.items()},
    )
    write(
        out / "input_audit.json",
        dict(
            subjects=len(subjects),
            windows=len(examples),
            quiet_windows=int(quiet.sum()),
            exact_G1_S_state_tensors=len(shared),
            exact_ordinary_G1_S_IDs=int((~quiet).sum() * 15),
            current_IDs_and_outputs="exactly_equal_frozen_existing_bank",
            train_bank_sha256=digest(bank_path),
            source_exposure=cfg["training"]["exposure"],
            validation_arrays_read=0,
        ),
    )
    return examples, bank, ids, quiet, teacher_ids, teacher, prediction


def crossfit(
    cfg,
    out,
    rows,
    scales,
    examples,
    bank,
    ids,
    quiet,
    teacher_ids,
    teacher,
    current,
    parent,
):
    source_audit, baseline = audit(current, bank, examples, rows, scales)
    teacher_audit, reference = audit(teacher, bank, examples, rows, scales)
    oracle = preserve_O_mix_HbO(current, teacher, quiet, 1)
    _, ceiling = audit(oracle, bank, examples, rows, scales)
    write(
        out / "teacher_ceiling.json",
        dict(
            current=baseline,
            raw_original_G1=reference,
            teacher_HbO_X_with_current_O_and_unchanged_quiet_HbR=ceiling,
            teacher_X_HbO_improvement=1
            - ceiling["metrics"]["nmse"]["macro"][0][0]
            / baseline["metrics"]["nmse"]["macro"][0][0],
        ),
    )
    folds = fold_assignment(examples, rows, cfg)
    candidates = []
    metrics_archive = dict(
        examples=examples,
        folds=folds,
        source=source_audit["metrics"]["nmse"],
        G1=teacher_audit["metrics"]["nmse"],
    )
    predictions = {}
    classification = []
    size = cfg["resources"]["batch_size"]
    mask_fraction = bank["mask"].mean(-1)
    for shrink in cfg["model"]["shrinkage_token_equivalents"]:
        fold_tables = []
        for fold in range(cfg["training"]["folds"]):
            train = np.flatnonzero(folds != fold)
            weights = (
                training_weights(examples[train], rows)[:, None] * mask_fraction[train]
            )
            fitted = fit_tables(
                {b: ids[b][train] for b in "SO"},
                teacher_ids["O"][train],
                weights,
                shrink,
            )
            fold_tables.append(fitted)
        torch.save(fold_tables, out / f"fold_tables_shrink{shrink}.pt")
        for strength in cfg["model"]["context_strengths"]:
            probability = np.empty((len(examples), 15, 128))
            for fold, fitted in enumerate(fold_tables):
                held = np.flatnonzero(folds == fold)
                probability[held] = parent_probabilities(
                    {b: ids[b][held] for b in "SO"}, fitted, strength
                )
            w = (
                training_weights(examples, rows)[:, None]
                * mask_fraction
                * (~quiet[:, None])
            )
            chosen = np.take_along_axis(
                probability, teacher_ids["O"][..., None], axis=-1
            )[..., 0]
            classification.append(
                dict(
                    shrinkage=shrink,
                    context_strength=strength,
                    ordinary_token_accuracy=float(
                        np.sum(w * (probability.argmax(-1) == teacher_ids["O"]))
                        / w.sum()
                    ),
                    ordinary_token_log_loss=float(
                        np.sum(-w * np.log(chosen.clip(1e-300))) / w.sum()
                    ),
                )
            )
            for mode in cfg["model"]["parent_modes"]:
                name = f"shrink{shrink}__context{strength:g}__{mode}"
                predicted = batched_parent(parent, ids, probability, mode, size)
                _, parent_summary = audit(predicted, bank, examples, rows, scales)
                write(out / (name + "_predicted_parent.json"), parent_summary)
                for alpha in cfg["model"]["alpha_HbO"]:
                    key = name + f"__alpha{alpha:g}"
                    output = preserve_O_mix_HbO(current, predicted, quiet, alpha)
                    measured, summary = audit(output, bank, examples, rows, scales)
                    decision = eligibility(summary, baseline, reference, cfg)
                    record = dict(
                        candidate=key,
                        shrinkage=shrink,
                        context_strength=strength,
                        parent_mode=mode,
                        alpha=alpha,
                        CV_X_HbO_NMSE=summary["metrics"]["nmse"]["macro"][0][0],
                        **decision,
                        metrics=summary,
                    )
                    candidates.append(record)
                    metrics_archive[key] = measured["metrics"]["nmse"]
                    predictions[key] = output
                write(
                    out / "progress.json",
                    dict(
                        stage="parent_crossfit",
                        completed=len(candidates),
                        planned=64,
                        CPU_seconds=time.process_time(),
                    ),
                )
    eligible = [r for r in candidates if r["eligible"]]
    selected = None
    if eligible:
        best = min(r["CV_X_HbO_NMSE"] for r in eligible)
        comparable = [
            r
            for r in eligible
            if r["CV_X_HbO_NMSE"]
            <= best * cfg["selection"]["parsimony_X_objective_ratio"]
        ]
        selected = min(
            comparable,
            key=lambda r: (
                r["context_strength"] > 0,
                -r["shrinkage"],
                r["alpha"],
                r["parent_mode"] != "hard",
                r["context_strength"],
            ),
        )
    best = min(candidates, key=lambda r: r["CV_X_HbO_NMSE"])
    for label, candidate in [("best_diagnostic", best), ("selected", selected)]:
        if candidate:
            np.savez_compressed(
                out / (label + "_training_predictions.npz"),
                examples=examples,
                **predictions[candidate["candidate"]],
            )
    np.savez_compressed(out / "crossfit_nmse.npz", **metrics_archive)
    write(
        out / "training_crossfit.json",
        dict(
            status="completed",
            source=baseline,
            original_G1=reference,
            classification=classification,
            candidates=candidates,
            best_diagnostic=best["candidate"],
            selected=None if selected is None else selected["candidate"],
            validation_arrays_read=0,
        ),
    )
    return selected


def read_summary(folder):
    for name in ["summary_units_v2.json", "summary_v2.json", "summary.json"]:
        if (folder / name).exists():
            return json.loads((folder / name).read_text())
    raise FileNotFoundError(folder)


def evaluate(cfg, out, checkpoint, rows, scales, role):
    _, _, examples, _ = load_panel(
        argparse.Namespace(
            panel=ROOT / cfg["source"]["panel"], modality="hb", role=role
        )
    )
    with np.load(ROOT / cfg["source"]["integers"]) as stored:
        lookup = {tuple(e): j for j, e in enumerate(stored["examples"])}
        take = [lookup[tuple(e)] for e in examples]
        ids = {b: stored[b][take].copy() for b in "SO"}
    folder = out / role
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / "integer_ids.npz", examples=examples, **ids)
    with np.load(folder / "integer_ids.npz") as stored:
        ids = {b: stored[b].copy() for b in "SO"}
    model = load_model(checkpoint)
    replay = load_model(checkpoint)
    values = defaultdict(list)
    support = defaultdict(list)
    source_metrics = defaultdict(list)
    from collections import OrderedDict

    cache = OrderedDict()
    for start in range(0, len(examples), cfg["resources"]["batch_size"]):
        ex = examples[start : start + cfg["resources"]["batch_size"]]
        codes = {b: ids[b][start : start + len(ex)] for b in "SO"}
        batch = base_tokenizer_batch(
            rows, ex, scales, array_cache=cache, cache_limit=64
        )
        output = model.decode_ids(codes)
        repeated = replay.decode_ids(codes)
        before = model.source.decode_ids(codes)
        for b in ["S", "O", "sum"]:
            np.testing.assert_array_equal(output[b], repeated[b])
        np.testing.assert_array_equal(output["O"], before["O"])
        measured, _ = audit(output, batch, ex, rows, scales)
        original, _ = audit(before, batch, ex, rows, scales)
        for k, v in measured["metrics"].items():
            values[k].append(v)
        for k, v in measured["support"].items():
            support[k].append(v)
        for k, v in original["metrics"].items():
            source_metrics[k].append(v)
        write(
            folder / "progress.json",
            dict(
                completed=start + len(ex),
                planned=len(examples),
                CPU_seconds=time.process_time(),
            ),
        )
    metrics = {k: np.concatenate(v) for k, v in values.items()}
    counts = {k: np.concatenate(v) for k, v in support.items()}
    summary = joint_reconstruction_summary(metrics, counts, examples.tolist(), rows)
    summary.update(
        status="completed",
        modality="hb",
        role=role,
        integer_replay=dict(
            status="passed",
            independent_checkpoint_reload=True,
            waveform_max_abs_difference=0.0,
        ),
        O_HbR_quiet="all_exactly_unchanged",
        checkpoint=str(checkpoint),
    )
    np.savez_compressed(
        folder / "metrics.npz",
        examples=examples,
        **metrics,
        **{"support_" + k: v for k, v in counts.items()},
    )
    write(folder / "summary.json", summary)
    run = ROOT / Path(cfg["source"]["panel"]).parent
    root = run / ("baseline_audit" if role == "screen" else "development_audit") / "hb"
    decision = score(
        summary,
        read_summary(root / "vq128"),
        read_summary(root / "continuous"),
        yaml.safe_load(
            (
                ROOT
                / "experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml"
            ).read_text()
        ),
    )
    write(folder / "decision.json", decision)
    # Reuse the fixed historical display-selection rules, retaining hard examples.
    selection = display_selection(
        argparse.Namespace(reference_metrics=root / "vq128" / "metrics.npz"),
        metrics,
        rows,
        examples,
    )
    chosen = np.array(sorted(selection))
    batch = base_tokenizer_batch(rows, examples[chosen], scales)
    output = model.decode_ids({b: ids[b][chosen] for b in "SO"})
    np.savez_compressed(
        folder / "waveform_examples.npz",
        examples=examples[chosen],
        **batch,
        **{"prediction_" + b: v for b, v in output.items()},
    )
    write(
        folder / "waveform_examples.json",
        dict(
            selection=selection,
            scales=input_scales(rows, examples[chosen], scales),
            metadata=[
                dict(rows[i], start_s=int(t), position=int(p))
                for i, t, p in examples[chosen]
            ],
        ),
    )
    return decision


def run(args, cfg):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("completed distillation evidence retained")
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest["rows"]
    current = load_current(ROOT / cfg["source"]["current"])
    if (
        current.metadata["signature"]["input"] != identity(manifest)
        or current.metadata["scales_identity"] != manifest["scales_identity"]
    ):
        raise ValueError("current/G0 identity mismatch")
    parent, setting, parent_meta = parent_source(cfg, manifest, scales)
    write(
        out / "resolved_config.json",
        dict(
            config=cfg,
            input_identity=manifest["cfg_identity"],
            scales_identity=manifest["scales_identity"],
        ),
    )
    write(
        out / "duplicate_audit.json",
        dict(
            status="distinct",
            prior_waveform_residual="fits X residual on reconstructed waveforms",
            prior_G4="different encoders and jointly learned neural decoder",
            prior_parent_leaf="changes O assignment and transmitted child IDs",
            this="freeze semantic emitted integers and infer historical G1 O128 teacher labels, then static G1 decode with preserved current O",
        ),
    )
    examples, bank, ids, quiet, teacher_ids, teacher, current_prediction = prepare(
        cfg, out, manifest, scales, current, parent
    )
    selected = crossfit(
        cfg,
        out,
        rows,
        scales,
        examples,
        bank,
        ids,
        quiet,
        teacher_ids,
        teacher,
        current_prediction,
        parent,
    )
    if selected is None:
        write(
            out / "summary.json",
            dict(
                status="train_crossfit_rejected",
                screen_read=False,
                development_read=False,
                confirmation_read=False,
                test_read=False,
                CPU_seconds=time.process_time(),
            ),
        )
        return
    weights = training_weights(examples, rows)[:, None] * bank["mask"].mean(-1)
    fitted = fit_tables(ids, teacher_ids["O"], weights, selected["shrinkage"])
    for name, source in [
        ("current.pt", ROOT / cfg["source"]["current"]),
        ("parent.pt", ROOT / cfg["source"]["parent"]),
    ]:
        shutil.copy2(source, out / name)
    companion = (ROOT / cfg["source"]["current"]).parent / current.metadata[
        "source_file"
    ]
    shutil.copy2(companion, out / current.metadata["source_file"])
    metadata = deepcopy(current.metadata)
    metadata.update(
        schema="hb_parent_distillation_v1",
        current_file="current.pt",
        current_sha256=digest(out / "current.pt"),
        parent_file="parent.pt",
        parent_sha256=digest(out / "parent.pt"),
        context_strength=selected["context_strength"],
        parent_mode=selected["parent_mode"],
        alpha=selected["alpha"],
    )
    checkpoint = out / "distilled.pt"
    torch.save(
        dict(
            metadata=metadata,
            fitted=fitted,
            parent_setting=setting,
            parent_metadata=parent_meta,
        ),
        checkpoint,
    )
    write(
        out / "selected.json",
        dict(
            status="frozen_before_screen",
            candidate=selected["candidate"],
            emitted_IDs="unchanged_S129_O257",
            O_HbR_quiet="exact",
            train_config_identity=identity(cfg),
        ),
    )
    screen = evaluate(cfg, out, checkpoint, rows, scales, "screen")
    development = None
    if screen["reconstruction_passed"]:
        development = evaluate(cfg, out, checkpoint, rows, scales, "development")
    write(
        out / "summary.json",
        dict(
            status="completed",
            screen=screen,
            development=development,
            checkpoint=str(checkpoint),
            confirmation_read=False,
            test_read=False,
            CPU_seconds=time.process_time(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg["schema"] != "hb_parent_distillation_v1"
        or cfg["data"]["test_access"] != "forbidden"
    ):
        raise ValueError("frozen train-first distillation contract required")
    torch.set_num_threads(cfg["resources"]["torch_threads"])
    resource.setrlimit(
        resource.RLIMIT_CPU, (cfg["resources"]["process_CPU_limit_seconds"],) * 2
    )
    with owner_lock(args.output):
        try:
            run(args, cfg)
        except BaseException as error:
            write(
                args.output / "failure.json",
                dict(
                    type=type(error).__name__,
                    reason=str(error),
                    CPU_seconds=time.process_time(),
                ),
            )
            raise


if __name__ == "__main__":
    main()
