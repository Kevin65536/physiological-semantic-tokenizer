"""Real-X HbO residual heads from unchanged semantic integer sequences."""

import argparse
from collections import defaultdict, OrderedDict
from copy import deepcopy
import hashlib, json, resource, shutil, sys, time
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/scripts'))
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
    base_training_aggregate,
)
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.tokenizers.contextual_latent_dictionary import load_model as load_source
from src.tokenizers.hb_parent_distillation import validate_ids
from src.tokenizers.hb_native_id_readout import (
    smooth_basis,
    fit_head,
    apply_head,
    load_model,
)


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def ranked(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def prepare(cfg, out, manifest, scales, source):
    path = ROOT / cfg["source"]["bank"]
    if digest(path) != cfg["source"]["bank_sha256"]:
        raise ValueError("exact full pattern bank identity required")
    with np.load(ROOT / cfg["source"]["panel"]) as panel:
        expected = panel["hb_pattern_train"].copy()
    with np.load(path) as f:
        examples = f["examples"].copy()
        np.testing.assert_array_equal(examples, expected)
        if len(examples) != 15360 or any(
            manifest["rows"][i]["split"] != "train"
            or manifest["rows"][i]["modality"] != "hb"
            for i, _, _ in examples
        ):
            raise ValueError("full train-only boundary before target arrays")
        if (
            len(
                {
                    (manifest["rows"][i]["dataset"], manifest["rows"][i]["subject"])
                    for i, _, _ in examples
                }
            )
            != 60
        ):
            raise ValueError("all sixty training subjects required")
        bank = {
            k: f[k].copy()
            for k in ["x", "s", "o", "mask", "vx", "vs", "vo", "dvx", "dvs", "dvo"]
        }
        parent_S = f["id_S"].copy()
        g1 = {b: f["original_" + b].copy() for b in ["S", "O", "sum"]}
    for name in ["vx", "vs", "vo", "dvx", "dvs", "dvo"]:
        bank[name] = np.broadcast_to(
            np.tile(bank[name], (1, 20))[:, None, :], bank["x"].shape
        )
    cached = out / "current_training.npz"
    if cached.exists():
        with np.load(cached) as f:
            np.testing.assert_array_equal(f["examples"], examples)
            ids = {b: f[b].copy() for b in "SO"}
            prediction = {b: f["prediction_" + b].copy() for b in ["S", "O", "sum"]}
    else:
        allids = defaultdict(list)
        pred = defaultdict(list)
        for start in range(0, len(examples), cfg["resources"]["batch_size"]):
            x = bank["x"][start : start + cfg["resources"]["batch_size"]]
            codes = source.encode(x)
            decoded = source.decode_ids(codes)
            _, quiet = validate_ids(codes)
            np.testing.assert_array_equal(
                codes["S"][~quiet], parent_S[start : start + len(x)][~quiet]
            )
            for b, v in codes.items():
                allids[b].append(v)
            for b, v in decoded.items():
                pred[b].append(v)
            write(
                out / "progress.json",
                dict(
                    stage="frozen_full_pattern_integer_collection",
                    completed=start + len(x),
                    planned=len(examples),
                    CPU_seconds=time.process_time(),
                ),
            )
        ids = {b: np.concatenate(v) for b, v in allids.items()}
        prediction = {b: np.concatenate(v) for b, v in pred.items()}
        np.savez_compressed(
            cached,
            examples=examples,
            **ids,
            **{"prediction_" + b: v for b, v in prediction.items()},
        )
    _, quiet = validate_ids(ids)
    write(
        out / "input_audit.json",
        dict(
            subjects=60,
            windows=len(examples),
            ordinary_S_IDs_exact_G1=int((~quiet).sum() * 15),
            quiet_windows=int(quiet.sum()),
            source_SHA256=digest(ROOT / cfg["source"]["current"]),
            bank_SHA256=digest(path),
            source_exposure=cfg["training"]["exposure"],
            validation_reads=0,
        ),
    )
    return examples, bank, ids, prediction, g1


def protect(summary, current, g1, metrics, base_metrics, examples, rows, folds, cfg):
    m, b, g = summary["metrics"], current["metrics"], g1["metrics"]
    rule = cfg["selection"]
    nm = m["nmse"]
    old = b["nmse"]
    macro = 1 - nm["macro"][0][0] / old["macro"][0][0]
    datasets = {
        d: 1 - v[0][0] / old["by_dataset"][d][0][0] for d, v in nm["by_dataset"].items()
    }
    subject = {
        (v["dataset"], v["subject"]): v["value"][0][0] for v in old["by_subject"]
    }
    improved = sum(
        v["value"][0][0] < subject[v["dataset"], v["subject"]] for v in nm["by_subject"]
    )
    fold_results = []
    for fold in range(cfg["training"]["folds"]):
        take = np.flatnonzero(folds == fold)
        a = base_training_aggregate(
            {"nmse": metrics["nmse"][take]}, examples[take], rows
        )["metrics"]["nmse"]
        z = base_training_aggregate(
            {"nmse": base_metrics["nmse"][take]}, examples[take], rows
        )["metrics"]["nmse"]
        fold_results.append(
            dict(
                fold=fold,
                macro=1 - a["macro"][0][0] / z["macro"][0][0],
                datasets={
                    d: 1 - v[0][0] / z["by_dataset"][d][0][0]
                    for d, v in a["by_dataset"].items()
                },
            )
        )
    checks = dict(
        macro_X_gain=macro >= rule["minimum_macro_X_HbO_relative_improvement"],
        all_dataset_X_gain=min(datasets.values()) > 0,
        repeat_macro_folds=sum(r["macro"] > 0 for r in fold_results)
        >= rule["positive_macro_folds_min"],
        repeat_dataset_folds=all(
            sum(r["datasets"][d] > 0 for r in fold_results)
            >= rule["positive_folds_per_dataset_min"]
            for d in datasets
        ),
        improved_subjects=improved / 60 >= rule["improved_subject_fraction_min"],
    )
    checks["macro_SO"] = bool(
        np.all(
            np.asarray(nm["macro"])[1:, 0]
            <= rule["macro_SO_HbO_ratio_to_current_max"]
            * np.asarray(old["macro"])[1:, 0]
        )
    )
    checks["dataset_SO"] = all(
        np.all(
            np.asarray(v)[1:, 0]
            <= rule["dataset_SO_HbO_ratio_to_current_max"]
            * np.asarray(old["by_dataset"][d])[1:, 0]
        )
        for d, v in nm["by_dataset"].items()
    )
    checks["macro_G1"] = bool(
        np.all(
            np.asarray(nm["macro"])
            <= rule["macro_all_component_color_ratio_to_G1_max"]
            * np.asarray(g["nmse"]["macro"])
        )
    )
    checks["dataset_G1"] = all(
        np.all(
            np.asarray(v)
            <= rule["dataset_all_component_color_ratio_to_G1_max"]
            * np.asarray(g["nmse"]["by_dataset"][d])
        )
        for d, v in nm["by_dataset"].items()
    )
    for name in ["macro", *datasets]:
        get = lambda stats, key: np.asarray(
            stats[key]["macro"] if name == "macro" else stats[key]["by_dataset"][name]
        )[:2, 0]
        rms, base = get(m, "rms_gain"), get(b, "rms_gain")
        lo, hi = rule["changed_X_S_HbO_RMS_interval"]
        within = (rms >= lo) & (rms <= hi)
        old_within = (base >= lo) & (base <= hi)
        checks["amplitude_" + name] = bool(
            np.all(
                np.where(old_within, within, np.abs(rms - 1) <= np.abs(base - 1) + 0.01)
            )
        )
        checks["slope_" + name] = bool(
            np.all(
                get(m, "slope_correlation")
                >= get(b, "slope_correlation")
                - rule["changed_slope_correlation_drop_max"]
            )
        )
        checks["curvature_" + name] = bool(
            np.all(
                get(m, "curvature_correlation")
                >= get(b, "curvature_correlation")
                - rule["changed_curvature_correlation_drop_max"]
            )
        )
        checks["boundary_" + name] = bool(
            np.all(
                get(m, "boundary_jump_difference_nmse")
                <= np.maximum(4, 1.05 * get(b, "boundary_jump_difference_nmse"))
            )
        )
    return dict(
        eligible=all(checks.values()),
        checks=checks,
        relative_X_HbO_improvement=macro,
        dataset_improvements=datasets,
        improved_subjects=improved,
        subjects=60,
        fold_improvements=fold_results,
    )


def train(cfg, out, rows, scales, examples, bank, ids, prediction, g1):
    base_audit, baseline = audit(prediction, bank, examples, rows, scales)
    _, reference = audit(g1, bank, examples, rows, scales)
    np.savez_compressed(
        out / "source_metrics.npz", examples=examples, **base_audit["metrics"]
    )
    folds = fold_assignment(examples, rows, cfg)
    np.save(out / "training_folds.npy", folds)
    complete = bank["mask"].reshape(len(examples), 15, 20, 2).all((2, 3))
    _, quiet = validate_ids(ids)
    residual = (
        bank["x"].reshape(-1, 15, 20, 2)[:, :, :, 0]
        - prediction["sum"].reshape(-1, 15, 20, 2)[:, :, :, 0]
    )
    response = np.einsum("npt,tk->npk", residual, smooth_basis()) / 20
    subjects = np.array(
        [rows[i]["dataset"] + "/" + rows[i]["subject"] for i, _, _ in examples]
    )
    records = []
    best = 1e99
    bestkey = None
    eligible = []
    for family, ridge in itertools.product(
        cfg["model"]["families"], cfg["model"]["ridge_equivalent_tokens"]
    ):
        key = f"{family}__ridge{ridge}"
        output = {b: np.asarray(v, float).copy() for b, v in prediction.items()}
        heads = []
        for fold in range(cfg["training"]["folds"]):
            fitting = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds == fold)
            raw = (
                training_weights(examples[fitting], rows)[:, None]
                * complete[fitting]
                / 15
                * (~quiet[fitting, None])
            )
            w = (
                raw
                * (int((complete[fitting] & ~quiet[fitting, None]).sum()) / raw.sum())
                / bank["vx"][fitting, 0, 0, None]
            )
            fitted = fit_head(
                {b: ids[b][fitting] for b in "SO"},
                {b: v[fitting] for b, v in prediction.items()},
                response[fitting],
                w,
                complete[fitting],
                subjects[fitting],
                family,
                ridge,
                cfg["model"],
            )
            heads.append(fitted)
            corrected = apply_head(
                {b: ids[b][held] for b in "SO"},
                {b: v[held] for b, v in prediction.items()},
                fitted,
            )
            for b, v in corrected.items():
                output[b][held] = v
            write(
                out / "progress.json",
                dict(
                    stage="full_pattern_crossfit",
                    candidate=key,
                    completed_fold=fold + 1,
                    candidates_completed=len(records),
                    CPU_seconds=time.process_time(),
                ),
            )
        torch.save(heads, out / (key + "_fold_heads.pt"))
        for b in ["O"]:
            np.testing.assert_array_equal(output[b], prediction[b])
        for b in output:
            np.testing.assert_array_equal(output[b][quiet], prediction[b][quiet])
            np.testing.assert_array_equal(
                output[b][..., 1::2], prediction[b][..., 1::2]
            )
        measured, summary = audit(output, bank, examples, rows, scales)
        decision = protect(
            summary,
            baseline,
            reference,
            measured["metrics"],
            base_audit["metrics"],
            examples,
            rows,
            folds,
            cfg,
        )
        record = dict(
            candidate=key,
            family=family,
            ridge=ridge,
            scalar_coefficients=max(h["scalar_coefficients"] for h in heads),
            pair_counts=[h["pair_count"] for h in heads],
            CV_X_HbO_NMSE=summary["metrics"]["nmse"]["macro"][0][0],
            metrics=summary,
            **decision,
        )
        records.append(record)
        np.savez_compressed(
            out / (key + "_metrics.npz"), examples=examples, **measured["metrics"]
        )
        if record["CV_X_HbO_NMSE"] < best:
            best = record["CV_X_HbO_NMSE"]
            bestkey = key
            np.savez_compressed(
                out / "best_diagnostic_predictions.npz", examples=examples, **output
            )
        if record["eligible"]:
            eligible.append(record)
        write(
            out / "training_crossfit.json",
            dict(
                status="in_progress",
                source=baseline,
                G1=reference,
                candidates=records,
                validation_reads=0,
            ),
        )
    selected = (
        min(
            eligible,
            key=lambda r: (r["CV_X_HbO_NMSE"], r["scalar_coefficients"], -r["ridge"]),
        )
        if eligible
        else None
    )
    write(
        out / "training_crossfit.json",
        dict(
            status="completed",
            source=baseline,
            G1=reference,
            candidates=records,
            best_diagnostic=bestkey,
            selected=None if selected is None else selected["candidate"],
            validation_reads=0,
        ),
    )
    if selected is None:
        return None
    raw = training_weights(examples, rows)[:, None] * complete / 15 * (~quiet[:, None])
    w = (
        raw
        * (int((complete & ~quiet[:, None]).sum()) / raw.sum())
        / bank["vx"][:, 0, 0, None]
    )
    fitted = fit_head(
        ids,
        prediction,
        response,
        w,
        complete,
        subjects,
        selected["family"],
        selected["ridge"],
        cfg["model"],
    )
    return selected, fitted


def run(args, cfg):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest["rows"]
    if args.evaluate:
        decision = evaluate(
            cfg, out, out / "native_readout.pt", rows, scales, args.evaluate
        )
        write(
            out / (args.evaluate + "_completion.json"),
            dict(
                status="completed", decision=decision, CPU_seconds=time.process_time()
            ),
        )
        return
    if (out / "summary.json").exists():
        raise FileExistsError("retained completed full-pattern readout experiment")
    source = load_source(ROOT / cfg["source"]["current"])
    if (
        source.metadata["signature"]["input"] != identity(manifest)
        or source.metadata["scales_identity"] != manifest["scales_identity"]
    ):
        raise ValueError("frozen current/G0 identity drift")
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
            prior_32_coefficient="944-window waveform-only linear residual",
            prior_G4="different frozen G4 IDs and new neural decoder",
            prior_neighbor="cross-position waveforms not same-position categorical effects",
            prior_distillation="944-window labels target old G1 rather than real X",
            this="full15360 strong semantic ID pairs predict real X residual through strongly shrunk categorical tables and sparse interactions",
        ),
    )
    examples, bank, ids, prediction, g1 = prepare(cfg, out, manifest, scales, source)
    selected = train(cfg, out, rows, scales, examples, bank, ids, prediction, g1)
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
    record, fitted = selected
    shutil.copy2(ROOT / cfg["source"]["current"], out / "current.pt")
    companion = (ROOT / cfg["source"]["current"]).parent / source.metadata[
        "source_file"
    ]
    shutil.copy2(companion, out / source.metadata["source_file"])
    metadata = deepcopy(source.metadata)
    metadata.update(
        schema="hb_native_id_readout_v1",
        source_file="current.pt",
        source_sha256=digest(out / "current.pt"),
        extra_rate_bits_per_second=0,
        extra_scalar_coefficients=fitted["scalar_coefficients"],
        family=record["family"],
        ridge=record["ridge"],
    )
    torch.save(dict(metadata=metadata, fitted=fitted), out / "native_readout.pt")
    write(
        out / "selected.json",
        dict(
            status="frozen_before_screen",
            candidate=record["candidate"],
            extra_scalar_coefficients=fitted["scalar_coefficients"],
            extra_rate=0,
        ),
    )
    screen = evaluate(cfg, out, out / "native_readout.pt", rows, scales, "screen")
    write(
        out / "summary.json",
        dict(
            status="screen_requires_actual_visual_review"
            if screen["reconstruction_passed"]
            else "screen_numerical_rejected",
            screen=screen,
            development_read=False,
            confirmation_read=False,
            test_read=False,
            CPU_seconds=time.process_time(),
        ),
    )


def validate_pair_config(cfg):
    if (cfg['schema']!='hb_pair_allocation_v1' or cfg['data']['selection_splits']!=['train']
        or cfg['data']['confirmation_access']!='forbidden' or cfg['model']['unencoded_continuous_inputs']
        or cfg['model']['families']!=['local','window'] or cfg['model']['ridge_equivalent_tokens']!=[512,4096]
        or cfg['model']['branch_vocabularies']!={'S':129,'O':257}):
        raise ValueError('registered paired fixed-total train-first contract required')


def prepare_pair(cfg,out,manifest):
    from src.tokenizers.hb_native_id_readout import load_model as load_native
    native=ROOT/cfg['source']['native_run']
    source=load_native(ROOT/cfg['source']['current'])
    if source.metadata['signature']['input']!=identity(manifest) or source.metadata['scales_identity']!=manifest['scales_identity']:
        raise ValueError('frozen111 source and exact G0 identities differ')
    bank_path=ROOT/cfg['source']['bank']
    if digest(bank_path)!=cfg['source']['bank_sha256']:raise ValueError('pinned training truth bank changed')
    with np.load(ROOT/cfg['source']['panel']) as f:expected=f['hb_pattern_train'].copy()
    rows=manifest['rows']
    if len(expected)!=15360 or any(rows[i]['split']!='train' or rows[i]['modality']!='hb' for i,_,_ in expected):
        raise ValueError('15360 original train triples before target arrays')
    if len({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in expected})!=60:raise ValueError('original60 train subjects')
    with np.load(bank_path) as f:
        np.testing.assert_array_equal(f['examples'],expected)
        bank={k:f[k].copy() for k in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo')}
        g1={b:f['original_'+b].copy() for b in ('S','O','sum')}
    for k in ('vx','vs','vo','dvx','dvs','dvo'):
        bank[k]=np.broadcast_to(np.tile(bank[k],(1,20))[:,None,:],bank['x'].shape)
    with np.load(native/'current_training.npz') as f:
        np.testing.assert_array_equal(f['examples'],expected)
        ids={b:f[b].copy() for b in 'SO'}
        prediction={b:f['prediction_'+b].copy() for b in ('S','O','sum')}
    validate_ids(ids)
    folds=np.load(native/'training_folds.npy')
    np.testing.assert_array_equal(folds,fold_assignment(expected,rows,cfg))
    path=native/(cfg['training']['frozen_111_head']+'_fold_heads.pt')
    heads=torch.load(path,map_location='cpu',weights_only=False)
    if len(heads)!=4 or any(h['scalar_coefficients']!=111 or h['family']!='decoded_waveform' for h in heads):raise ValueError('four frozen111 heads required')
    # Verify the entire cached source from saved integers, using the semantic
    # source underneath111. The all-train111 head is never applied in outer CV.
    maximum=0.
    for start in range(0,len(expected),cfg['resources']['source_batch_size']):
        take=slice(start,start+cfg['resources']['source_batch_size'])
        decoded=source.source.decode_ids({b:v[take] for b,v in ids.items()})
        for b in prediction:
            maximum=max(maximum,float(abs(decoded[b]-prediction[b][take]).max()))
            np.testing.assert_array_equal(decoded[b],prediction[b][take])
    np.savez_compressed(out/'training_integer_ids.npz',examples=expected,**ids)
    write(out/'input_audit.json',dict(status='passed',windows=15360,subjects=60,source111_SHA256=digest(ROOT/cfg['source']['current']),
        truth_bank_SHA256=digest(bank_path),frozen_fold_heads_SHA256=digest(path),cached_source_integer_replay_max_abs=maximum,
        validation_arrays_read=0,source_exposure=cfg['training']['exposure'],fold_rule='exact_old_training_folds.npy_verified',
        new_layer_nesting='outer_k_source_head_k_for_both_fit_and_held; no mixture of other OOF heads'))
    return expected,bank,ids,prediction,g1,folds,heads,source


def pair_peak_error(prediction,bank,examples,rows):
    n=len(examples);valid=bank['mask'].reshape(n,300,2);truth=bank['s'].reshape(n,300,2)
    point=np.argmax(np.where(valid,abs(truth),-np.inf),axis=1)
    a=np.arange(n)[:,None];c=np.arange(2)[None]
    predicted=prediction['S'].reshape(n,300,2)
    error=(predicted[a,point,c]-truth[a,point,c])**2/bank['vs'][:,0,:2]
    return base_training_aggregate({'nmse':error[:,None]},examples,rows)['metrics']['nmse']


def pair_decision(summary,baseline,g1,metrics,base_metrics,g1_metrics,examples,rows,folds,cfg,peaks,base_peaks):
    rule=cfg['selection'];m,b,g=[v['metrics'] for v in (summary,baseline,g1)]
    a,z=np.asarray(m['nmse']['macro']),np.asarray(b['nmse']['macro'])
    gain=1-a[1:].sum()/z[1:].sum();color_gain=1-a[1:].sum(0)/z[1:].sum(0)
    dataset_gain={d:1-np.asarray(v)[1:].sum()/np.asarray(b['nmse']['by_dataset'][d])[1:].sum() for d,v in m['nmse']['by_dataset'].items()}
    subject={(v['dataset'],v['subject']):np.asarray(v['value'])[1:].sum() for v in b['nmse']['by_subject']}
    improved=sum(np.asarray(v['value'])[1:].sum()<subject[v['dataset'],v['subject']] for v in m['nmse']['by_subject'])
    fold_results=[]
    for fold in (range(4) if folds is not None else []):
        take=np.flatnonzero(folds==fold)
        aa=base_training_aggregate({'nmse':metrics['nmse'][take]},examples[take],rows)['metrics']['nmse']
        zz=base_training_aggregate({'nmse':base_metrics['nmse'][take]},examples[take],rows)['metrics']['nmse']
        fold_results.append(dict(fold=fold,gain=1-np.asarray(aa['macro'])[1:].sum()/np.asarray(zz['macro'])[1:].sum(),
            dataset_gain={d:1-np.asarray(v)[1:].sum()/np.asarray(zz['by_dataset'][d])[1:].sum() for d,v in aa['by_dataset'].items()}))
    checks=dict(joint=gain>=rule['minimum_joint_relative_improvement'],each_color=bool(np.all(color_gain>=rule['minimum_each_color_joint_relative_improvement'])),
        all_datasets=min(dataset_gain.values())>0,subjects=improved/len(subject)>=rule['improved_subject_fraction_min'],
        folds=sum(v['gain']>0 for v in fold_results)>=rule['positive_macro_folds_min'],
        dataset_folds=all(sum(v['dataset_gain'][d]>0 for v in fold_results)>=rule['positive_folds_per_dataset_min'] for d in dataset_gain),
        macro_current=bool(np.all(a<=rule['macro_component_color_ratio_to_current_max']*z)),
        macro_G1=bool(np.all(a<=rule['macro_component_color_ratio_to_G1_max']*np.asarray(g['nmse']['macro']))),
        dataset_current=all(np.all(np.asarray(v)<=rule['dataset_component_color_ratio_to_current_max']*np.asarray(b['nmse']['by_dataset'][d])) for d,v in m['nmse']['by_dataset'].items()),
        dataset_G1=all(np.all(np.asarray(v)<=rule['dataset_component_color_ratio_to_G1_max']*np.asarray(g['nmse']['by_dataset'][d])) for d,v in m['nmse']['by_dataset'].items()))
    if folds is None:
        for key in ('all_datasets','subjects','folds','dataset_folds'):checks.pop(key)
        checks['joint']=bool(gain>0);checks['each_color']=bool(np.all(color_gain>=0))
    lo,hi=rule['overall_RMS_interval'];rms=np.asarray(m['rms_gain']['macro'])
    checks['overall_RMS']=bool(np.all((rms>=lo)&(rms<=hi)))
    tails={}
    for name in ['macro',*dataset_gain]:
        get=lambda stats,key:np.asarray(stats[key]['macro'] if name=='macro' else stats[key]['by_dataset'][name])
        if name!='macro':
            rr,old=get(m,'rms_gain'),get(b,'rms_gain');inside=(old>=lo)&(old<=hi)
            checks['RMS_'+name]=bool(np.all(np.where(inside,(rr>=lo)&(rr<=hi),abs(rr-1)<=abs(old-1)+.01)))
        for metric in ['slope_correlation','curvature_correlation']:
            checks[metric+'_'+name]=bool(np.all(get(m,metric)[1:]>=get(b,metric)[1:]-rule['slope_curvature_correlation_drop_max']))
        checks['boundary_'+name]=bool(np.all(get(m,'boundary_jump_difference_nmse')[1:]<=np.maximum(4,1.05*get(b,'boundary_jump_difference_nmse')[1:])))
        peak=lambda v:np.asarray(v['macro'] if name=='macro' else v['by_dataset'][name])
        checks['true_S_peak_'+name]=bool(np.all(peak(peaks)<=rule['S_peak_abs_amplitude_error_ratio_max']*peak(base_peaks)+1e-15))
        use=np.ones(len(examples),bool) if name=='macro' else np.array([rows[i]['dataset']==name for i,_,_ in examples])
        for q in rule['difficulty_quantiles']:
            qa,qb,qg=[np.quantile(v['nmse'][use],q,axis=0) for v in (metrics,base_metrics,g1_metrics)]
            checks[f'tail{q}_{name}']=bool(np.all(qa<=rule['difficulty_source_ratio_max']*qb+1e-15) and np.all(qa<=np.maximum(rule['difficulty_G1_ratio_max']*qg,qb)+1e-15))
            tails[f'{name}_q{q}']=dict(candidate=qa,current=qb,G1=qg)
    return dict(eligible=all(checks.values()),checks=checks,joint_relative_improvement=gain,color_joint_relative_improvement=color_gain,
        dataset_gain=dataset_gain,improved_subjects=improved,subjects=len(subject),fold_results=fold_results,difficulty=tails)


def pair_render(out,examples,bank,source,prediction,rows,scales,metrics,base_metrics,selection=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    picked={}
    joint=base_metrics['nmse'][:,1:].mean((1,2));candidate=metrics['nmse'][:,1:].mean((1,2))
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples}):
        ids=[j for j,(i,_,_) in enumerate(examples) if rows[i]['dataset']==dataset]
        first=min(ids,key=lambda j:tuple(examples[j]));hard=max(ids,key=lambda j:joint[j])
        picked[first]='fixed_first';picked[hard]='source_joint_hardest'
    picked[int(candidate.argmax())]='candidate_joint_hardest'
    if selection is not None:picked={int(k):str(v) for k,v in selection.items()}
    folder=out/'waveforms';folder.mkdir(exist_ok=True);record=[]
    for index,label in sorted(picked.items()):
        i,start,p=examples[index];row=rows[i];unit=row.get('source_unit','native');scale=input_scales(rows,examples[index:index+1],scales)[0]
        mask=bank['mask'][index].reshape(300,2);t=np.arange(300)/10
        # Full window plus a true-S-peak and a token-boundary view, both colors.
        targetS=bank['s'][index].reshape(300,2);point=int(np.argmax(np.where(mask,abs(targetS),-np.inf))//2)
        center=np.clip(point/10,1,29);boundary=np.clip(round(center/2)*2,2,28)
        fig,axes=plt.subplots(6,2,figsize=(13,15))
        for r,(field,target) in enumerate((('sum','x'),('S','s'),('O','o'))):
            y=bank[target][index].reshape(300,2)*scale
            old=source[field][index].reshape(300,2)*scale;new=prediction[field][index].reshape(300,2)*scale
            for color in range(2):
                ax=axes[r,color]
                for value,name,ink in [(y,'target','black'),(old,'frozen111','#d98732'),(new,'paired allocation','#247db3')]:ax.plot(t,value[:,color],lw=.85,color=ink,label=name)
                for edge in np.arange(0,31,2):ax.axvline(edge,color='gray',lw=.4,alpha=.3)
                ax.fill_between(t,0,1,where=~mask[:,color],transform=ax.get_xaxis_transform(),color='gray',alpha=.2)
                ax.set(xlim=(0,30),ylabel=unit,title=field+' '+['HbO','HbR'][color]);ax.legend(fontsize=7)
        for r,(field,target,window) in enumerate((('S','s',(center-1,center+1)),('O','o',(center-1,center+1)),('S','s',(boundary-.5,boundary+.5))),start=3):
            take=(t>=window[0])&(t<window[1])
            for color in range(2):
                ax=axes[r,color]
                for value,name,ink in [(bank[target][index],'target','black'),(source[field][index],'frozen111','#d98732'),(prediction[field][index],'paired allocation','#247db3')]:
                    ax.plot(t[take],value.reshape(300,2)[take,color]*scale,lw=1,color=ink,label=name)
                for edge in np.arange(0,31,2):
                    if window[0]<=edge<=window[1]:ax.axvline(edge,color='gray',lw=.5)
                ax.set(xlim=window,ylabel=unit,xlabel='time(s)',title=field+' '+['HbO','HbR'][color]+(' true-S-peak vicinity' if r<5 else ' nearest token boundary'))
        fig.suptitle(f'{row["dataset"]}/{row["subject"]} row{i}+{start}s {row["positions"][p]} {label}\nThin vertical lines: 2s tokens; gray shading: original mask; all curves share axes')
        fig.tight_layout();path=folder/f'{row["dataset"]}_{label}_{index}.png';fig.savefig(path,dpi=200);plt.close(fig)
        record.append(dict(path=str(path),example=examples[index],dataset=row['dataset'],subject=row['subject'],position=row['positions'][p],label=label,
            source_joint_NMSE=joint[index],candidate_joint_NMSE=candidate[index],true_S_peak_time_s=point/10,status='rendered_awaiting_actual_review'))
    write(folder/'manifest.json',dict(figures=record))


def run_pair_allocation(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_native_id_readout import apply_head
    from src.tokenizers.hb_pair_allocation import sufficient_statistics,fit_from_statistics,apply_pair,optimal_delta
    validate_pair_config(cfg);out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT);rows=manifest['rows']
    if args.allocation_diagnostics:
        return pair_diagnostic_analysis(args,cfg,rows,scales)
    if args.evaluate:
        return evaluate_pair_allocation(args,cfg,rows,scales)
    if (out/'training_crossfit.json').exists() and json.loads((out/'training_crossfit.json').read_text())['status']=='completed':raise FileExistsError('completed allocation evidence retained')
    write(out/'resolved_config.json',dict(config=cfg,input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity']))
    examples,bank,ids,raw,g1,folds,heads,source=prepare_pair(cfg,out,manifest)
    if args.benchmark:
        fit=np.flatnonzero(folds!=0)[:128];base=apply_head({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in raw.items()},heads[0]);timings=[]
        for threads in cfg['resources']['benchmark_threads']:
            with threadpool_limits(threads):
                begin=time.monotonic();cpu=time.process_time()
                sufficient_statistics({b:v[fit] for b,v in ids.items()},base,{k:v[fit] for k,v in bank.items()},training_weights(examples[fit],rows),batch_size=64)
                timings.append(dict(threads=threads,wall_seconds=time.monotonic()-begin,CPU_seconds=time.process_time()-cpu,windows=len(fit)))
        write(out/'benchmark.json',dict(timings=timings,selected_threads=min(timings,key=lambda v:v['wall_seconds'])['threads'],CPU_seconds=time.process_time()));return
    native=ROOT/cfg['source']['native_run']
    for name in ['native_readout.pt','current.pt','model.pt']:
        if not (out/name).exists():shutil.copy2(native/name,out/name)
        if digest(out/name)!=digest(native/name):raise ValueError('frozen source bundle changed')
    threads=json.loads((out/'benchmark.json').read_text())['selected_threads']
    base={b:np.empty_like(np.asarray(raw[b],float)) for b in ('S','O','sum')}
    keys=[f'{family}__ridge{ridge}' for family in cfg['model']['families'] for ridge in cfg['model']['ridge_equivalent_tokens']]
    outputs={key:dict(S=np.empty_like(base['S']),O=np.empty_like(base['O']),sum=base['sum']) for key in keys};fitted_heads={key:[] for key in keys}
    scopes=[]
    with threadpool_limits(threads):
        for fold in range(4):
            fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
            post111=apply_head(ids,raw,heads[fold])
            for b in base:base[b][held]=post111[b][held]
            fit_subjects=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[fit]});held_subjects=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[held]})
            if set(fit_subjects)&set(held_subjects):raise ValueError('outer subject overlap')
            scopes.append(dict(fold=fold,source111_head_index=fold,fit_subjects=fit_subjects,held_subjects=held_subjects,source111_used_same_head_for_fit_and_held=True))
            path=out/f'fold{fold}_normal_equations.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['config_identity']!=identity(cfg):raise ValueError('normal-equation continuation identity changed')
                stats=saved['statistics']
            else:
                stats=sufficient_statistics({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in post111.items()},
                    {k:v[fit] for k,v in bank.items()},training_weights(examples[fit],rows),cfg['resources']['batch_size'])
                torch.save(dict(config_identity=identity(cfg),statistics=stats,scope=scopes[-1]),path)
            for family in cfg['model']['families']:
                for ridge in cfg['model']['ridge_equivalent_tokens']:
                    key=f'{family}__ridge{ridge}';fitted=fit_from_statistics(stats,family,ridge);fitted_heads[key].append(fitted)
                    metadata=deepcopy(source.metadata);metadata.update(schema='hb_pair_allocation_v1',source_file='native_readout.pt',source_sha256=digest(out/'native_readout.pt'),
                        extra_allocation_scalar_coefficients=fitted['scalar_coefficients'],extra_scalar_coefficients=111+fitted['scalar_coefficients'],fixed_sum_head=111,held_fold=fold)
                    torch.save(dict(metadata=metadata,fitted=fitted,source111_fold_override=heads[fold]),out/f'{key}_fold{fold}.pt')
                    corrected=apply_pair({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in post111.items()},fitted)
                    for b in ('S','O'):outputs[key][b][held]=corrected[b]
                    np.testing.assert_array_equal(corrected['sum'],post111['sum'][held])
            write(out/'progress.json',dict(stage='strict_nested_allocation_CV',completed_fold=fold+1,CPU_seconds=time.process_time()))
        write(out/'fold_scope.json',dict(groups=scopes,strict_outer_111_heads=True))
        measured0,baseline=audit(base,bank,examples,rows,scales);measuredg,reference=audit(g1,bank,examples,rows,scales)
        with np.load(ROOT/cfg['source']['native_run']/(cfg['training']['frozen_111_head']+'_metrics.npz')) as f:
            np.testing.assert_array_equal(f['examples'],examples)
            np.testing.assert_allclose(measured0['metrics']['nmse'],f['nmse'],rtol=0,atol=1e-12)
            source_difference=float(abs(measured0['metrics']['nmse']-f['nmse']).max())
        write(out/'source111_CV_summary.json',baseline);write(out/'G1_summary.json',reference)
        np.savez_compressed(out/'source111_CV_metrics.npz',examples=examples,**measured0['metrics'])
        peaks0=pair_peak_error(base,bank,examples,rows);_,quiet=validate_ids(ids)
        records=[];best=None;best_score=np.inf
        for key in keys:
            prediction=outputs[key]
            np.testing.assert_array_equal(prediction['sum'],base['sum'])
            for b in prediction:np.testing.assert_array_equal(prediction[b][quiet],base[b][quiet])
            measured,summary=audit(prediction,bank,examples,rows,scales)
            peaks=pair_peak_error(prediction,bank,examples,rows)
            decision=pair_decision(summary,baseline,reference,measured['metrics'],measured0['metrics'],measuredg['metrics'],examples,rows,folds,cfg,peaks,peaks0)
            value=np.asarray(summary['metrics']['nmse']['macro'])[1:].mean()
            records.append(dict(candidate=key,scalar_coefficients=fitted_heads[key][0]['scalar_coefficients'],joint_NMSE=value,metrics=summary,S_true_peak=peaks,**decision))
            torch.save(fitted_heads[key],out/(key+'_fold_heads.pt'));np.savez_compressed(out/(key+'_metrics.npz'),examples=examples,**measured['metrics'])
            if value<best_score:best,best_score=key,value
            write(out/'training_crossfit.json',dict(status='in_progress',candidates=records,validation_reads=0))
        eligible=[v for v in records if v['eligible']]
        selected=min(eligible,key=lambda v:(v['joint_NMSE'],v['scalar_coefficients'],-int(v['candidate'].split('ridge')[1]))) if eligible else None
        chosen=selected['candidate'] if selected else best
        pair_replay_training(out,chosen,ids,outputs[chosen],base,folds,bank,examples,rows,scales)
        np.savez_compressed(out/'diagnostic_predictions.npz',examples=examples,**outputs[chosen],**{'source_'+b:v for b,v in base.items()})
        delta=optimal_delta(base,bank).reshape(-1,15,40);delta[quiet]=0
        oracle=dict(S=base['S']-delta,O=base['O']+delta,sum=base['sum'])
        _,oracle_summary=audit(oracle,bank,examples,rows,scales);write(out/'target_informed_fixed_sum_oracle.json',oracle_summary)
        write(out/'training_crossfit.json',dict(status='completed',candidates=records,selected=None if selected is None else selected['candidate'],
            best_diagnostic=best,source111_original_CV_NMSE_max_abs_difference=source_difference,validation_reads=0))
        chosen_metrics=next(v for v in records if v['candidate']==chosen)
        with np.load(out/(chosen+'_metrics.npz')) as f:metrics={k:f[k].copy() for k in f.files if k!='examples'}
        pair_render(out,examples,bank,base,outputs[chosen],rows,scales,metrics,measured0['metrics'])
        if selected is None:
            write(out/'summary.json',dict(status='train_crossfit_rejected',best_diagnostic=best,screen_read=False,development_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()));return
        # Fit only the preselected family once on all train. The released111 head
        # now legitimately uses all train; selection used only nested heads.
        full=source.decode_ids(ids)
        stats=sufficient_statistics(ids,full,bank,training_weights(examples,rows),cfg['resources']['batch_size'])
        family,ridge=selected['candidate'].split('__ridge');fitted=fit_from_statistics(stats,family,int(ridge))
        native=ROOT/cfg['source']['native_run']
        for name in ['native_readout.pt','current.pt','model.pt']:shutil.copy2(native/name,out/name)
        metadata=deepcopy(source.metadata);metadata.update(schema='hb_pair_allocation_v1',source_file='native_readout.pt',source_sha256=digest(out/'native_readout.pt'),
            extra_allocation_scalar_coefficients=fitted['scalar_coefficients'],extra_scalar_coefficients=111+fitted['scalar_coefficients'],fixed_sum_head=111)
        torch.save(dict(metadata=metadata,fitted=fitted),out/'allocation.pt')
        write(out/'selected.json',dict(status='frozen_before_screen',candidate=selected['candidate'],scalar_coefficients=fitted['scalar_coefficients']))
        write(out/'summary.json',dict(status='train_eligible_awaiting_actual_visual_review_then_screen',selected=selected['candidate'],screen_read=False,development_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))


def pair_replay_training(out,key,ids,prediction,base,folds,bank,examples,rows,scales):
    from src.tokenizers.hb_pair_allocation import load_model as load_pair
    with np.load(out/'training_integer_ids.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);disk={b:f[b].copy() for b in 'SO'}
    maximum=0.;total_difference=0.;recovered={b:np.empty_like(prediction[b]) for b in prediction}
    for fold in range(4):
        model=load_pair(out/f'{key}_fold{fold}.pt')
        model.source.encode=lambda *_: (_ for _ in ()).throw(AssertionError('integer replay requested X'))
        held=np.flatnonzero(folds==fold)
        for start in range(0,len(held),128):
            take=held[start:start+128];code={b:v[take] for b,v in disk.items()};output=model.decode_ids(code)
            source=model.source.decode_ids(code)
            np.testing.assert_array_equal(output['sum'],source['sum'])
            _,quiet=validate_ids(code)
            for b in output:
                np.testing.assert_array_equal(output[b][quiet],source[b][quiet])
                np.testing.assert_allclose(output[b],prediction[b][take],atol=2e-6,rtol=1e-5)
                maximum=max(maximum,float(abs(output[b]-prediction[b][take]).max()));recovered[b][take]=output[b]
            total_difference=max(total_difference,float(abs(output['S']+output['O']-output['sum']).max()))
    measured,summary=audit(recovered,bank,examples,rows,scales)
    write(out/'integer_replay.json',dict(status='passed',candidate=key,independent_checkpoint_reload=True,saved_IDs_only=True,
        maximum_absolute_difference=maximum,sum_arithmetic_max_abs=total_difference,sum_field_exact_frozen111=True,quiet_all_outputs_exact=True,
        tolerance=dict(atol=2e-6,rtol=1e-5),fold111_override_matches_new_layer_outer_fold=True))
    write(out/'replayed_training_summary.json',summary)
    np.savez_compressed(out/'replayed_training_metrics.npz',examples=examples,**measured['metrics'])


def pair_diagnostic_analysis(args,cfg,rows,scales):
    from src.tokenizers.hb_pair_allocation import optimal_delta
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output;path=ROOT/cfg['source']['bank']
    if digest(path)!=cfg['source']['bank_sha256']:raise ValueError('training truth identity')
    with np.load(out/'diagnostic_predictions.npz') as f:
        examples=f['examples'].copy()
        if len(examples)!=15360 or any(rows[i]['split']!='train' for i,_,_ in examples):raise ValueError('diagnostic train boundary before arrays')
        source={b:f['source_'+b].copy() for b in ('S','O','sum')};prediction={b:f[b].copy() for b in ('S','O','sum')}
    with np.load(path) as f:
        np.testing.assert_array_equal(f['examples'],examples)
        bank={k:f[k].copy() for k in ('s','o','mask','vs','vo')}
    with np.load(out/'training_integer_ids.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);_,quiet=validate_ids({b:f[b].copy() for b in 'SO'})
    n=len(examples);mask=bank['mask'].reshape(n,300,2);count=mask.sum(1)
    average=lambda value:np.where(mask,value,0.).sum(1)/np.maximum(count,1)
    s0,o0=[source[b].reshape(n,300,2) for b in 'SO'];s1,o1=[prediction[b].reshape(n,300,2) for b in 'SO']
    st,ot=[bank[b].reshape(n,300,2) for b in 'so'];vs,vo=[bank[b][:,None] for b in ('vs','vo')]
    target=optimal_delta(source,bank);predicted=o1-o0
    j0=average((s0-st)**2/vs+(o0-ot)**2/vo);j1=average((s1-st)**2/vs+(o1-ot)**2/vo)
    floor=average((s0+o0-st-ot)**2/(vs+vo))
    a0=average((1/vs+1/vo)*target**2);a1=average((1/vs+1/vo)*(predicted-target)**2)
    np.testing.assert_allclose(j0,floor+a0,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(j1,floor+a1,rtol=1e-10,atol=1e-10)
    policy_floor=floor+np.where(quiet[:,None],a0,0.)
    reducible0=np.where(quiet[:,None],0,a0);reducible1=np.where(quiet[:,None],0,a1)
    fields=['source_joint_SO_sum','candidate_joint_SO_sum','mathematical_fixed_sum_floor','quiet_preserving_floor','source_reducible_allocation','candidate_reducible_allocation']
    values=np.stack([j0,j1,floor,policy_floor,reducible0,reducible1],axis=1)
    hierarchy=base_training_aggregate({'nmse':values},examples,rows)['metrics']['nmse'];macro=np.asarray(hierarchy['macro'])
    write(out/'allocation_decomposition.json',dict(status='passed_exact_quadratic_identity',field_order=fields,color_order=['HbO','HbR'],hierarchy=hierarchy,
        branch_reduction='sum_S_and_O_not_mean',reducible_allocation_fraction_source=macro[4]/macro[0],
        reducible_allocation_relative_reduction=1-macro[5]/macro[4],quiet_windows=int(quiet.sum()),
        max_point_decomposition_error=float(max(abs(j0-floor-a0).max(),abs(j1-floor-a1).max())),
        source_history='111 family and ridge were historically selected on old full60-person CV; this new layer excludes held targets from111 coefficient fitting via the corresponding old fold head; base tokenizer and scales already exposed to all train',
        validation_arrays_read=0))
    selected=[]
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples}):
        take=[j for j,(i,_,_) in enumerate(examples) if rows[i]['dataset']==dataset]
        selected.append(max(take,key=lambda j:j0[j].mean()))
    folder=out/'waveforms';fig,axes=plt.subplots(4,4,figsize=(16,10),layout='constrained');fig2,axes2=plt.subplots(4,2,figsize=(13,10),layout='constrained');t=np.arange(300)/10;identities=[]
    for row,index in enumerate(selected):
        i,start,p=examples[index];scale=input_scales(rows,examples[index:index+1],scales)[0];dataset=rows[i]['dataset'];unit=rows[i].get('source_unit','native')
        identities.append(dict(dataset=dataset,subject=rows[i]['subject'],example=examples[index],unit=unit,selection='maximum_source_joint_SO_NMSE_per_dataset'))
        for col,(b,target_key,c) in enumerate((('S','s',0),('S','s',1),('O','o',0),('O','o',1))):
            ax=axes[row,col]
            for wave,name,ink in [(bank[target_key],'target','black'),(source[b],'frozen111','#d98732'),(prediction[b],'paired allocation','#247db3')]:
                y=wave[index].reshape(300,2)[:,c]*scale;y=y-y[mask[index,:,c]].mean()
                ax.plot(t,y,color=ink,lw=.8,label=name)
            ax.set(xlim=(0,30),title=dataset+' '+b+' '+['HbO','HbR'][c],ylabel='Hb (native coord.)\nown mean removed',xlabel='time(s)')
            ax.title.set_fontsize(8);ax.yaxis.label.set_fontsize(8);ax.tick_params(labelsize=8)
            if row==0 and col==0:ax.legend(fontsize=7)
        for c in range(2):
            ax=axes2[row,c];ax.plot(t,target[index,:,c]*scale,color='black',lw=.9,label='exact target delta (uses hidden S/O truth)')
            ax.plot(t,predicted[index,:,c]*scale,color='#247db3',lw=.9,label='integer-decoded predicted delta');ax.axhline(0,color='gray',lw=.5)
            for edge in np.arange(0,31,2):ax.axvline(edge,color='gray',lw=.4,alpha=.3)
            ax.fill_between(t,0,1,where=~mask[index,:,c],transform=ax.get_xaxis_transform(),color='gray',alpha=.2)
            ax.set(xlim=(0,30),title=dataset+' '+['HbO','HbR'][c],ylabel=unit,xlabel='time(s)')
            if row==0 and c==0:ax.legend(fontsize=7)
    fig.suptitle('Shape diagnosis only: each curve centered; original-amplitude panels retained separately',fontsize=10)
    fig2.suptitle('Frozen best diagnostic CV head; no refitting or new candidate selection\nVertical lines: 2s tokens; target-informed delta is a diagnostic label only')
    fig.savefig(folder/'centered_shape_diagnostic_v2.png',dpi=200)
    if not (folder/'allocation_target_diagnostic.png').exists():fig2.savefig(folder/'allocation_target_diagnostic.png',dpi=200)
    plt.close(fig);plt.close(fig2)
    write(folder/'supplementary_manifest.json',dict(identities=identities,figures=['centered_shape_diagnostic_v2.png','allocation_target_diagnostic.png'],previous_render='centered_shape_diagnostic.png; long y labels clipped, numerical arrays unchanged',status='rendered_awaiting_actual_review'))


def evaluate_pair_allocation(args,cfg,rows,scales):
    from src.tokenizers.hb_pair_allocation import load_model as load_pair
    out=args.output
    training=json.loads((out/'training_crossfit.json').read_text())
    if training['selected'] is None or not (out/'allocation.pt').exists():raise ValueError('train-qualified frozen allocation checkpoint before validation')
    if args.evaluate=='development':
        review=json.loads((out/'screen'/'actual_review.json').read_text())
        if not review['all_checks_passed']:raise ValueError('screen numerical, ID-pattern and actual visual acceptance before full development')
    _,_,examples,_=load_panel(argparse.Namespace(panel=ROOT/cfg['source']['panel'],modality='hb',role=args.evaluate))
    with np.load(ROOT/cfg['source']['integers']) as f:
        lookup={tuple(v):j for j,v in enumerate(f['examples'])};take=[lookup[tuple(v)] for v in examples];ids={b:f[b][take].copy() for b in 'SO'}
    folder=out/args.evaluate;folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'integer_ids.npz',examples=examples,**ids)
    with np.load(folder/'integer_ids.npz') as f:disk={b:f[b].copy() for b in 'SO'}
    model=load_pair(out/'allocation.pt');replay=load_pair(out/'allocation.pt');replay.source.encode=lambda *_: (_ for _ in ()).throw(AssertionError('replay requested X'))
    values=defaultdict(list);counts=defaultdict(list);before_values=defaultdict(list);cache=OrderedDict();banks=defaultdict(list);outputs=defaultdict(list);sources=defaultdict(list)
    maximum=0.;encoder_difference=0
    for start in range(0,len(examples),128):
        ex=examples[start:start+128];codes={b:v[start:start+len(ex)] for b,v in disk.items()};batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=64)
        actual_ids=model.encode(batch['x'])
        for b in 'SO':
            encoder_difference+=int((actual_ids[b]!=codes[b]).sum());np.testing.assert_array_equal(actual_ids[b],codes[b])
        output=model.decode_ids(codes);repeated=replay.decode_ids(codes);before=model.source.decode_ids(codes);_,quiet=validate_ids(codes)
        np.testing.assert_array_equal(output['sum'],before['sum'])
        for b in output:
            np.testing.assert_array_equal(output[b],repeated[b]);np.testing.assert_array_equal(output[b][quiet],before[b][quiet]);maximum=max(maximum,float(abs(output[b]-repeated[b]).max()))
            outputs[b].append(output[b]);sources[b].append(before[b])
        measured,_=audit(output,batch,ex,rows,scales);original,_=audit(before,batch,ex,rows,scales)
        for k,v in measured['metrics'].items():values[k].append(v)
        for k,v in original['metrics'].items():before_values[k].append(v)
        for k,v in measured['support'].items():counts[k].append(v)
        for k,v in batch.items():banks[k].append(v)
        write(folder/'progress.json',dict(completed=start+len(ex),planned=len(examples),CPU_seconds=time.process_time()))
    metrics={k:np.concatenate(v) for k,v in values.items()};support={k:np.concatenate(v) for k,v in counts.items()};old={k:np.concatenate(v) for k,v in before_values.items()}
    summary=joint_reconstruction_summary(metrics,support,examples.tolist(),rows);summary.update(status='completed',role=args.evaluate,integer_replay_max_abs=maximum)
    before_summary=joint_reconstruction_summary(old,support,examples.tolist(),rows)
    write(folder/'summary.json',summary);write(folder/'source111_summary.json',before_summary)
    np.savez_compressed(folder/'metrics.npz',examples=examples,**metrics,**{'support_'+k:v for k,v in support.items()})
    run=(ROOT/cfg['source']['panel']).parent;reference=run/('baseline_audit' if args.evaluate=='screen' else 'development_audit')/'hb'
    contract=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml').read_text())
    decision=score(summary,read_summary(reference/'vq128'),read_summary(reference/'continuous'),contract)
    all_bank={k:np.concatenate(v) for k,v in banks.items()};all_source={b:np.concatenate(v) for b,v in sources.items()};all_output={b:np.concatenate(v) for b,v in outputs.items()}
    with np.load(reference/'vq128'/'metrics.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);G1_metrics={k:f[k].copy() for k in f.files if k not in ('examples',) and not k.startswith('support_')}
    protection=pair_decision(summary,before_summary,read_summary(reference/'vq128'),metrics,old,G1_metrics,examples,rows,None,cfg,
        pair_peak_error(all_output,all_bank,examples,rows),pair_peak_error(all_source,all_bank,examples,rows))
    write(folder/'paired_route_protection.json',protection)
    decision['original_reconstruction_passed']=decision['reconstruction_passed']
    decision['paired_route_protection_passed']=protection['eligible']
    decision['reconstruction_passed']=bool(decision['reconstruction_passed'] and protection['eligible']);write(folder/'decision.json',decision)
    write(folder/'pattern_inheritance.json',dict(status='exact_ID_equivalence',encoder_ID_mismatches=encoder_difference,
        training_panel_unchanged=True,target_masks_and_background_protocol_unchanged=True,
        single_ID_true_attribute_statistics='identical_by_exact_ID_and_original_target_identity',
        parent_evidence=str(ROOT/cfg['source']['native_run']/'pattern_inheritance_scope.json'),decoded_sequence_visuals_require_new_review=True,
        exposure='development subjects previously used in G1 and this model family; not a fresh independent test'))
    selection=display_selection(argparse.Namespace(reference_metrics=reference/'vq128'/'metrics.npz'),metrics,rows,examples)
    pair_render(folder,examples,all_bank,all_source,all_output,rows,scales,metrics,old,selection=selection)
    write(out/(args.evaluate+'_completion.json'),dict(status='awaiting_actual_visual_review',decision=decision,integer_replay_max_abs=maximum,
        exact_ID_inheritance=True,confirmation_access=False,test_access=False,CPU_seconds=time.process_time()))


def run_pair_signed_gain(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import apply_pair,optimal_delta,fit_signed_gains,fixed_color_gains,load_model as load_pair
    if (cfg['data']['allowed_splits']!=['train'] or cfg['fitting']['scopes']!=['global','dataset'] or cfg['fitting']['range']!=[-1,1]):
        raise ValueError('single train-only signed-gain comparison required')
    if args.evaluate:raise ValueError('finish train numerical and actual visual review before any screen')
    if args.gain_replay_only:return replay_pair_signed_gain(args,cfg)
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():raise FileExistsError('completed signed-gain evidence retained')
    parent=ROOT/cfg['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config']
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows']
    write(out/'resolved_config.json',dict(config=cfg,unchanged_original_selection=old['selection'],input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity']))
    pinned=['resolved_config.json','native_readout.pt','current.pt','model.pt','diagnostic_predictions.npz',*[f'window__ridge4096_fold{k}.pt' for k in range(4)]]
    write(out/'source_identity.json',dict(parent=str(parent),files={name:digest(parent/name) for name in pinned},frozen_family='window__ridge4096',original_failure_unchanged=True))
    examples,bank,ids,raw,g1,folds,heads,source=prepare_pair(old,out,manifest)
    datasets=np.array([rows[i]['dataset'] for i,_,_ in examples]);_,quiet=validate_ids(ids)
    for name in ['native_readout.pt','current.pt','model.pt']:shutil.copy2(parent/name,out/name)
    with np.load(parent/'diagnostic_predictions.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples)
        baseline={b:f['source_'+b].copy() for b in ('S','O','sum')}
        previous={b:f[b].copy() for b in ('S','O','sum')}
    outputs={scope:dict(S=np.empty_like(baseline['S']),O=np.empty_like(baseline['O']),sum=baseline['sum']) for scope in cfg['fitting']['scopes']}
    records=[];scope_records=[]
    with threadpool_limits(cfg['resources']['numerical_threads']):
        for fold in range(4):
            fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
            fit_people={(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[fit]};held_people={(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[held]}
            if fit_people&held_people:raise ValueError('outer held person entered gain fitting')
            frozen=torch.load(parent/f'window__ridge4096_fold{fold}.pt',map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(frozen['source111_fold_override']['coefficient'],heads[fold]['coefficient'])
            post111=apply_head(ids,raw,heads[fold])
            for b in baseline:np.testing.assert_allclose(post111[b][held],baseline[b][held],rtol=0,atol=2e-13)
            corrected=apply_pair(ids,post111,frozen['fitted']);delta=corrected['O']-post111['O']
            target=optimal_delta({b:v[fit] for b,v in post111.items()},{k:v[fit] for k,v in bank.items()})
            fits={scope:fit_signed_gains(delta[fit],target,{k:v[fit] for k,v in bank.items()},training_weights(examples[fit],rows),datasets[fit],scope) for scope in cfg['fitting']['scopes']}
            stat=torch.load(parent/f'fold{fold}_normal_equations.pt',map_location='cpu',weights_only=False)['statistics']
            for c in range(2):
                beta=frozen['fitted']['coefficient'][:,c]
                expected=np.clip((beta@stat['right'][c])/(beta@stat['gram'][c]@beta),-1,1)
                np.testing.assert_allclose(fits['global']['global']['gain'][c],expected,atol=2e-12,rtol=2e-12)
            for scope in cfg['fitting']['scopes']:
                fitted=deepcopy(frozen['fitted'])
                if scope=='global':fitted['global_color_gain']=fits[scope]['global']['gain']
                else:fitted['dataset_color_gains']={d:v['gain'] for d,v in fits[scope].items()}
                metadata=deepcopy(frozen['metadata']);metadata.update(schema='hb_pair_allocation_gain_v1',source_file='native_readout.pt',source_sha256=digest(out/'native_readout.pt'),gain_scope=scope,gain_scalar_coefficients=2 if scope=='global' else 8,legal_fixed_metadata=['dataset'] if scope=='dataset' else [],held_fold=fold)
                torch.save(dict(metadata=metadata,fitted=fitted,source111_fold_override=heads[fold]),out/f'{scope}_fold{fold}.pt')
                gains=fixed_color_gains(fitted,datasets[held],len(held))
                for b,sign in [('S',-1),('O',1)]:
                    outputs[scope][b][held]=(post111[b][held].reshape(len(held),300,2)+sign*delta[held].reshape(len(held),300,2)*gains[:,None]).reshape(len(held),15,40)
                    outputs[scope][b][held[quiet[held]]]=post111[b][held[quiet[held]]]
            scope_records.append(dict(fold=fold,fit_subjects=sorted(fit_people),held_subjects=sorted(held_people),
                source111_fold=fold,delta_fold=fold,gains=fits,held_targets_used_for_fitting=False))
            write(out/'fit_progress.json',dict(completed_folds=fold+1,gains=scope_records,CPU_seconds=time.process_time()))
            print('completed_fold',fold,'gains',fits,flush=True)
        write(out/'gain_fit_scope.json',dict(folds=scope_records,original_source_exposure=old['training']['exposure'],historical_family_selection='111 and delta families/ridges previously explored on original60train CV; no fresh end-to-end test'))
        with np.load(parent/'source111_CV_metrics.npz') as f:base_metrics={k:f[k].copy() for k in f.files if k!='examples'}
        base_summary=json.loads((parent/'source111_CV_summary.json').read_text());g1summary=json.loads((parent/'G1_summary.json').read_text())
        measured_g1,_=audit(g1,bank,examples,rows,scales);base_peak=pair_peak_error(baseline,bank,examples,rows)
        measured_all={}
        for scope in cfg['fitting']['scopes']:
            prediction=outputs[scope]
            for b in prediction:np.testing.assert_array_equal(prediction[b][quiet],baseline[b][quiet])
            np.testing.assert_array_equal(prediction['sum'],baseline['sum'])
            measured,summary=audit(prediction,bank,examples,rows,scales);measured_all[scope]=measured['metrics']
            result=pair_decision(summary,base_summary,g1summary,measured['metrics'],base_metrics,measured_g1['metrics'],examples,rows,folds,old,pair_peak_error(prediction,bank,examples,rows),base_peak)
            records.append(dict(candidate=scope,summary=summary,joint_NMSE=float(np.asarray(summary['metrics']['nmse']['macro'])[1:].mean()),**result))
            np.savez_compressed(out/f'{scope}_metrics.npz',examples=examples,**measured['metrics'])
        control_difference=max(float(abs(outputs['global'][b]-previous[b]).max()) for b in previous)
        if control_difference>2e-13:raise ValueError('matching global gain did not recover unchanged original delta')
        eligible=[v for v in records if v['eligible']];best=min(records,key=lambda v:v['joint_NMSE'])
        chosen=min(eligible,key=lambda v:(v['joint_NMSE'],2 if v['candidate']=='global' else 8)) if eligible else None
        write(out/'training_crossfit.json',dict(status='completed',candidates=records,selected=None if chosen is None else chosen['candidate'],best_diagnostic=best['candidate'],global_control_max_abs_difference=control_difference,validation_reads=0))
        key=best['candidate'] if chosen is None else chosen['candidate'];prediction=outputs[key]
        recovered={b:np.empty_like(v) for b,v in prediction.items()};maximum=0.;closure=0.
        for fold in range(4):
            model=load_pair(out/f'{key}_fold{fold}.pt');model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('integer replay requested X'))
            held=np.flatnonzero(folds==fold)
            for start in range(0,len(held),128):
                take=held[start:start+128];code={b:v[take] for b,v in ids.items()};decoded=model.decode_ids(code,dataset=datasets[take])
                for b in decoded:
                    np.testing.assert_allclose(decoded[b],prediction[b][take],atol=2e-6,rtol=1e-5)
                    maximum=max(maximum,float(abs(decoded[b]-prediction[b][take]).max()));recovered[b][take]=decoded[b]
                current111=model.source.decode_ids(code)
                np.testing.assert_array_equal(decoded['sum'],current111['sum']);_,q=validate_ids(code)
                for b in decoded:np.testing.assert_array_equal(decoded[b][q],current111[b][q])
                closure=max(closure,float(abs(decoded['S']+decoded['O']-decoded['sum']).max()))
        replayed,replay_summary=audit(recovered,bank,examples,rows,scales)
        difference=float(np.nanmax(abs(replayed['metrics']['nmse']-measured_all[key]['nmse'])))
        np.testing.assert_allclose(replayed['metrics']['nmse'],measured_all[key]['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
        write(out/'integer_replay.json',dict(status='passed',candidate=key,maximum_waveform_difference=maximum,maximum_NMSE_difference=difference,sum_field_exact=True,quiet_exact=True,sum_closure_max_abs=closure,inputs=['saved_integer_IDs','frozen_111_delta_and_gain_coefficients','fixed_dataset_identity'],source_and_delta_exclude_matching_outer_fold=True))
        write(out/'replayed_training_summary.json',replay_summary)
        np.savez_compressed(out/'diagnostic_predictions.npz',examples=examples,**prediction,**{'source_'+b:v for b,v in baseline.items()})
        pair_render(out,examples,bank,baseline,prediction,rows,scales,measured_all[key],base_metrics)
        write(out/'summary.json',dict(status='train_crossfit_rejected' if chosen is None else 'train_numeric_eligible_awaiting_actual_visual_review',selected=None if chosen is None else chosen['candidate'],best_diagnostic=best['candidate'],screen_read=False,development_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))
        print('selected',None if chosen is None else chosen['candidate'],'best',best['candidate'],'joint',best['joint_NMSE'],flush=True)


def replay_pair_signed_gain(args,cfg):
    """Resume only saved checkpoints/metrics; never repeat gain fitting."""
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import load_model as load_pair,fixed_color_gains
    out=args.output;parent=ROOT/cfg['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config']
    record=json.loads((out/'training_crossfit.json').read_text());key=record['selected'] or record['best_diagnostic']
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows']
    with np.load(out/'training_integer_ids.npz') as f:
        examples=f['examples'].copy()
        if len(examples)!=15360 or any(rows[i]['split']!='train' for i,_,_ in examples):raise ValueError('original train triples before replay')
        ids={b:f[b].copy() for b in 'SO'}
    datasets=np.array([rows[i]['dataset'] for i,_,_ in examples]);folds=np.load(ROOT/old['source']['native_run']/'training_folds.npy')
    np.testing.assert_array_equal(folds,fold_assignment(examples,rows,old))
    path=ROOT/old['source']['bank']
    if digest(path)!=old['source']['bank_sha256']:raise ValueError('frozen training truth identity')
    with np.load(path) as f:
        np.testing.assert_array_equal(f['examples'],examples)
        bank={k:f[k].copy() for k in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo')}
    for k in ('vx','vs','vo','dvx','dvs','dvo'):bank[k]=np.broadcast_to(np.tile(bank[k],(1,20))[:,None,:],bank['x'].shape)
    with np.load(parent/'diagnostic_predictions.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples)
        baseline={b:f['source_'+b].copy() for b in ('S','O','sum')};delta=f['O']-baseline['O']
    recovered={b:np.empty_like(v) for b,v in baseline.items()};maximum=closure=path_difference=0.
    with threadpool_limits(cfg['resources']['numerical_threads']):
        for fold in range(4):
            model=load_pair(out/f'{key}_fold{fold}.pt');model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('replay requested X'))
            held=np.flatnonzero(folds==fold)
            for start in range(0,len(held),128):
                take=held[start:start+128];code={b:v[take] for b,v in ids.items()};labels=datasets[take]
                decoded=model.decode_ids(code,dataset=labels);current111=model.source.decode_ids(code);_,quiet=validate_ids(code)
                np.testing.assert_array_equal(decoded['sum'],current111['sum'])
                for b in decoded:np.testing.assert_array_equal(decoded[b][quiet],current111[b][quiet])
                gains=fixed_color_gains(model.fitted,labels,len(take))
                expected={b:(baseline[b][take].reshape(len(take),300,2)+sign*delta[take].reshape(len(take),300,2)*gains[:,None]).reshape(len(take),15,40) for b,sign in [('S',-1),('O',1)]}|{'sum':baseline['sum'][take]}
                for b in decoded:
                    np.testing.assert_allclose(decoded[b],expected[b],atol=2e-6,rtol=1e-5)
                    maximum=max(maximum,float(abs(decoded[b]-expected[b]).max()));recovered[b][take]=decoded[b]
                path_difference=max(path_difference,float(abs(current111['sum']-baseline['sum'][take]).max()))
                closure=max(closure,float(abs(decoded['S']+decoded['O']-decoded['sum']).max()))
        measured,summary=audit(recovered,bank,examples,rows,scales)
        with np.load(out/f'{key}_metrics.npz') as f:
            np.testing.assert_array_equal(f['examples'],examples)
            difference=float(np.nanmax(abs(measured['metrics']['nmse']-f['nmse'])))
            np.testing.assert_allclose(measured['metrics']['nmse'],f['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
        write(out/'integer_replay.json',dict(status='passed',candidate=key,saved_IDs_only=True,independent_checkpoint_reload=True,
            maximum_waveform_difference=maximum,maximum_NMSE_difference=difference,sum_field_exact_same_111_decode=True,quiet_exact_same_111_decode=True,
            unchanged_tolerance=dict(atol=2e-6,rtol=1e-5),cross_batch_111_path_max_difference=path_difference,sum_closure_max_abs=closure,
            gain_refits=0,inputs=['saved_integer_IDs','frozen_111_delta_and_gain_coefficients','fixed_dataset_identity'],source_and_delta_exclude_matching_outer_fold=True))
        write(out/'replayed_training_summary.json',summary)
        np.savez_compressed(out/'diagnostic_predictions.npz',examples=examples,**recovered,**{'source_'+b:v for b,v in baseline.items()})
        with np.load(parent/'source111_CV_metrics.npz') as f:base_metrics={k:f[k].copy() for k in f.files if k!='examples'}
        pair_render(out,examples,bank,baseline,recovered,rows,scales,measured['metrics'],base_metrics)
    write(out/'summary.json',dict(status='train_crossfit_rejected' if record['selected'] is None else 'train_numeric_eligible_awaiting_actual_visual_review',selected=record['selected'],best_diagnostic=record['best_diagnostic'],
        resumed_without_fitting=True,retained_attempt_failure='replay_attempt001_failure.json',screen_read=False,development_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))
    print('replay_passed',key,'waveform_max',maximum,'NMSE_max',difference,flush=True)


def validate_spectral_config(cfg):
    if (cfg['schema']!='hb_spectral_allocation_v1' or cfg['data']['allowed_splits']!=['train']
        or cfg['data']['confirmation_access']!='forbidden' or cfg['data']['test_access']!='forbidden'
        or cfg['model']['groups_DCT_indices']!=[[0,1],[1,3],[3,13]] or cfg['model']['context_seconds']!=30
        or cfg['model']['sample_rate_hz']!=10 or cfg['model']['branch_vocabularies']!={'S':129,'O':257}
        or cfg['model']['unencoded_continuous_inputs'] or cfg['model']['extra_integers']!=0
        or cfg['fitting']['scopes']!=['global','dataset'] or cfg['fitting']['ridge_fractions']!=[.1,1.]):
        raise ValueError('registered train-only whole30s Hb spectral contract required')


def wholewindow_error_parts(prediction,bank,examples,rows):
    metrics={k:[] for k in ['nmse','whole30s_level_nmse','whole30s_centered_shape_nmse']};n=len(examples)
    mask=bank['mask'].reshape(n,300,2);count=mask.sum(1)
    for b,k in [('sum','x'),('S','s'),('O','o')]:
        error=np.asarray(prediction[b],float).reshape(n,300,2)-np.asarray(bank[k],float).reshape(n,300,2)
        mean=np.where(mask,error,0).sum(1)/np.maximum(count,1)
        mse=np.where(mask,error*error,0).sum(1)/np.maximum(count,1);variance=bank['v'+k][:,0,:2]
        metrics['nmse'].append(mse/variance);metrics['whole30s_level_nmse'].append(mean*mean/variance)
        metrics['whole30s_centered_shape_nmse'].append((mse-mean*mean)/variance)
    return base_training_aggregate({k:np.stack(v,1) for k,v in metrics.items()},examples,rows)


def run_pair_spectral(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import spectral_statistics,fit_spectral,apply_spectral,load_model as load_pair
    validate_spectral_config(cfg)
    if args.evaluate:raise ValueError('report original training numerical and actual visual verdict before a new screen')
    if args.spectral_review_all:return review_spectral_all(args,cfg)
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():raise FileExistsError('completed wholewindow spectral evidence retained')
    parent=ROOT/cfg['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config']
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows']
    write(out/'resolved_config.json',dict(config=cfg,unchanged_relative_rules=old['selection'],input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity']))
    write(out/'duplicate_audit.json',dict(status='distinct',prior_neighbor='same-time all-site4channel linear outputs on different earlier decoder/population; does not fix111 sum or separate whole30s frequencies',
        prior_native111='2s current/neighbor DCT predicts only X-HbO residual',prior_pair='2s localDCT with optional k0..2 fullwindow auxiliary; shared294coef head',prior_signed_gain='scalar scaling of prior local delta; no frequency-specific4input2color fit',
        current='per-dataset or matched-global three whole30s spectral groups;24coefficients per map; fixed111 total; all original15360 held-train masks'))
    examples,bank,ids,raw,g1,folds,heads,source=prepare_pair(old,out,manifest)
    datasets=np.array([rows[i]['dataset'] for i,_,_ in examples]);_,quiet=validate_ids(ids)
    native=ROOT/old['source']['native_run']
    for name in ['native_readout.pt','current.pt','model.pt']:shutil.copy2(native/name,out/name)
    with np.load(parent/'diagnostic_predictions.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);baseline={b:f['source_'+b].copy() for b in ('S','O','sum')}
    base_summary=json.loads((parent/'source111_CV_summary.json').read_text());g1summary=json.loads((parent/'G1_summary.json').read_text())
    with np.load(parent/'source111_CV_metrics.npz') as f:base_metrics={k:f[k].copy() for k in f.files if k!='examples'}
    keys=[(scope,ridge,f'{scope}__ridge{ridge:g}') for scope in cfg['fitting']['scopes'] for ridge in cfg['fitting']['ridge_fractions']]
    outputs={key:dict(S=np.empty_like(baseline['S']),O=np.empty_like(baseline['O']),sum=baseline['sum']) for _,_,key in keys};scopes=[];fitted_heads={key:[] for _,_,key in keys}
    benchmark_path=out/'benchmark.json'
    if benchmark_path.exists():threads=json.loads(benchmark_path.read_text())['selected_threads']
    else:
        fit=np.flatnonzero(folds!=0)[:128];p=apply_head({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in raw.items()},heads[0]);timings=[]
        for threads in [1,2]:
            with threadpool_limits(threads):
                begin=time.monotonic();cpu=time.process_time()
                spectral_statistics({b:v[fit] for b,v in ids.items()},p,{k:v[fit] for k,v in bank.items()},training_weights(examples[fit],rows),datasets[fit],batch_size=128)
                timings.append(dict(threads=threads,wall_seconds=time.monotonic()-begin,CPU_seconds=time.process_time()-cpu,fit_only_windows=len(fit)))
        threads=min(timings,key=lambda v:v['wall_seconds'])['threads'];write(benchmark_path,dict(timings=timings,selected_threads=threads))
    with threadpool_limits(threads):
        for fold in range(4):
            fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold);post111=apply_head(ids,raw,heads[fold])
            for b in baseline:np.testing.assert_allclose(post111[b][held],baseline[b][held],rtol=0,atol=2e-13)
            fit_people={(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[fit]};held_people={(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[held]}
            if fit_people&held_people:raise ValueError('outer held person entered spectral fitting')
            scope_record=dict(fold=fold,source111_head=fold,fit_subjects=sorted(fit_people),held_subjects=sorted(held_people),held_targets_used_for_fitting=False);scopes.append(scope_record)
            path=out/f'fold{fold}_normal_equations.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['config_identity']!=identity(cfg):raise ValueError('spectral continuation contract changed')
                statistics=saved['statistics']
            else:
                begin=time.monotonic();cpu=time.process_time()
                statistics=spectral_statistics({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in post111.items()},{k:v[fit] for k,v in bank.items()},training_weights(examples[fit],rows),datasets[fit],batch_size=cfg['resources']['batch_size'])
                torch.save(dict(config_identity=identity(cfg),statistics=statistics,scope=scope_record),path)
                print('fit_statistics',fold,'windows',len(fit),'wall',time.monotonic()-begin,'CPU',time.process_time()-cpu,flush=True)
            for scope,ridge,key in keys:
                fitted=fit_spectral(statistics,scope,ridge);fitted_heads[key].append(fitted)
                metadata=deepcopy(source.metadata);metadata.update(schema='hb_spectral_allocation_v1',source_file='native_readout.pt',source_sha256=digest(out/'native_readout.pt'),
                    fixed_sum_head=111,extra_allocation_scalar_coefficients=fitted['scalar_coefficients'],extra_scalar_coefficients=111+fitted['scalar_coefficients'],held_fold=fold,
                    spectral_groups=cfg['model']['groups_DCT_indices'],legal_fixed_metadata=['dataset'] if scope=='dataset' else [])
                torch.save(dict(metadata=metadata,fitted=fitted,source111_fold_override=heads[fold]),out/f'{key}_fold{fold}.pt')
                prediction=apply_spectral({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in post111.items()},fitted,dataset=datasets[held])
                for b in 'SO':outputs[key][b][held]=prediction[b]
                np.testing.assert_array_equal(prediction['sum'],post111['sum'][held])
                for b in prediction:np.testing.assert_array_equal(prediction[b][quiet[held]],post111[b][held][quiet[held]])
            write(out/'fit_progress.json',dict(completed_folds=fold+1,CPU_seconds=time.process_time()))
        write(out/'fold_scope.json',dict(groups=scopes,source_exposure=old['training']['exposure'],new_coefficients_fit_only=True))
        measured_g1,_=audit(g1,bank,examples,rows,scales);np.savez_compressed(out/'G1_metrics.npz',examples=examples,**measured_g1['metrics'])
        contract=yaml.safe_load((ROOT/cfg['selection']['absolute_owner']).read_text());peaks0=pair_peak_error(baseline,bank,examples,rows)
        reference=dict(g1summary,modality='hb',role='original_train_crossfit');records=[];measured_all={}
        for scope,ridge,key in keys:
            prediction=outputs[key];measured,summary=audit(prediction,bank,examples,rows,scales);measured_all[key]=measured['metrics']
            relative=pair_decision(summary,base_summary,g1summary,measured['metrics'],base_metrics,measured_g1['metrics'],examples,rows,folds,old,pair_peak_error(prediction,bank,examples,rows),peaks0)
            absolute=score(dict(summary,modality='hb',role='original_train_crossfit',checkpoint=key,integer_replay=dict(status='pending')),reference,reference,contract)
            numeric_absolute=absolute['finite_predictions'] and all(v['passed'] for v in absolute['checks'].values())
            records.append(dict(candidate=key,scope=scope,ridge=ridge,scalar_coefficients=fitted_heads[key][0]['scalar_coefficients'],summary=summary,joint_NMSE=float(np.asarray(summary['metrics']['nmse']['macro'])[1:].mean()),
                absolute_checks=absolute,absolute_numerical_passed=numeric_absolute,relative_checks=relative,eligible=relative['eligible'] and numeric_absolute))
            np.savez_compressed(out/f'{key}_metrics.npz',examples=examples,**measured['metrics'])
            write(out/'training_crossfit.json',dict(status='in_progress',candidates=records,validation_reads=0))
        eligible=[v for v in records if v['eligible']];best=min(records,key=lambda v:v['joint_NMSE'])
        selected=min(eligible,key=lambda v:(v['joint_NMSE'],v['scalar_coefficients'],-v['ridge'])) if eligible else None
        key=best['candidate'] if selected is None else selected['candidate'];prediction=outputs[key]
        write(out/'training_crossfit.json',dict(status='completed',candidates=records,selected=None if selected is None else key,best_diagnostic=best['candidate'],
            original_relative_rules_unchanged=True,absolute_Hb_shape_checked=True,validation_reads=0))
        recovered={b:np.empty_like(v) for b,v in prediction.items()};maximum=closure=path_difference=0.
        for fold in range(4):
            model=load_pair(out/f'{key}_fold{fold}.pt');model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('integer replay requested X'))
            held=np.flatnonzero(folds==fold)
            for start in range(0,len(held),128):
                take=held[start:start+128];code={b:v[take] for b,v in ids.items()};decoded=model.decode_ids(code,dataset=datasets[take]);current111=model.source.decode_ids(code);_,q=validate_ids(code)
                np.testing.assert_array_equal(decoded['sum'],current111['sum'])
                for b in decoded:
                    np.testing.assert_array_equal(decoded[b][q],current111[b][q]);np.testing.assert_allclose(decoded[b],prediction[b][take],atol=2e-6,rtol=1e-5)
                    maximum=max(maximum,float(abs(decoded[b]-prediction[b][take]).max()));recovered[b][take]=decoded[b]
                path_difference=max(path_difference,float(abs(current111['sum']-baseline['sum'][take]).max()));closure=max(closure,float(abs(decoded['S']+decoded['O']-decoded['sum']).max()))
        replayed,replay_summary=audit(recovered,bank,examples,rows,scales);difference=float(np.nanmax(abs(replayed['metrics']['nmse']-measured_all[key]['nmse'])))
        np.testing.assert_allclose(replayed['metrics']['nmse'],measured_all[key]['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
        write(out/'integer_replay.json',dict(status='passed',candidate=key,maximum_waveform_difference=maximum,maximum_NMSE_difference=difference,
            total_exact_same_frozen111_decode=True,quiet_all_outputs_exact=True,cross_batch_111_difference=path_difference,sum_closure_max_abs=closure,
            tolerance=dict(atol=2e-6,rtol=1e-5),inputs=['saved_S129_O257_integers','frozen_models','fixed_dataset_identity'],raw_X_reads=0))
        write(out/'replayed_training_summary.json',replay_summary)
        np.savez_compressed(out/'diagnostic_predictions.npz',examples=examples,**recovered,**{'source_'+b:v for b,v in baseline.items()})
        parts=dict(source111=wholewindow_error_parts(baseline,bank,examples,rows),complete=wholewindow_error_parts(recovered,bank,examples,rows),frozen_group_ablations={})
        for removed,name in enumerate(['DC','slow','dynamic']):
            ablated={b:np.empty_like(v) for b,v in baseline.items()}
            for fold in range(4):
                held=np.flatnonzero(folds==fold);output=apply_spectral({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in baseline.items()},fitted_heads[key][fold],dataset=datasets[held],groups=[g for g in range(3) if g!=removed])
                for b in ablated:ablated[b][held]=output[b]
            parts['frozen_group_ablations']['remove_'+name]=wholewindow_error_parts(ablated,bank,examples,rows)
        write(out/'wholewindow_allocation_diagnostics.json',dict(**parts,interpretation='unchanged original masked loss; each ablation removes one group from same fitted model with no refit; original full-amplitude score remains primary'))
        chosen=display_selection(argparse.Namespace(reference_metrics=out/'G1_metrics.npz'),measured_all[key],rows,examples)
        pair_render(out,examples,bank,baseline,recovered,rows,scales,measured_all[key],base_metrics,selection={i:'_'.join(v) for i,v in chosen.items()})
        write(out/'summary.json',dict(status='train_crossfit_rejected' if selected is None else 'train_numeric_eligible_awaiting_actual_visual_review_and_report',selected=None if selected is None else key,best_diagnostic=best['candidate'],
            screen_read=False,development_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))
        print('spectral_candidates',[(v['candidate'],v['joint_NMSE'],v['eligible']) for v in records],flush=True)


def render_spectral_comparison(folder,selected,examples,bank,baseline,predictions,rows,scales,source_label='frozen111'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    folder.mkdir(exist_ok=True,parents=True);t=np.arange(300)/10;manifest=[]
    styles=[('#0072b2','-.'),('#56b4e9',':'),('#b23a48','-'),('#009e73','-')]
    for number,index in enumerate(sorted(selected)):
        i,start,position=examples[index];r=rows[i];scale=input_scales(rows,examples[index:index+1],scales)[0];valid=bank['mask'][index].reshape(300,2)
        peak=int(np.argmax(np.where(valid,abs(bank['s'][index].reshape(300,2)),-np.inf))//2);center=float(np.clip(peak/10,1,29))
        views=[('sum','x',(0,30),'30s'),('S','s',(0,30),'30s'),('O','o',(0,30),'30s'),('S','s',(14,16),'fixed2s'),('O','o',(14,16),'fixed2s'),
               ('S','s',(13.5,14.5),'fixed boundary'),('O','o',(13.5,14.5),'fixed boundary'),('S','s',(center-1,center+1),'true S peak'),('O','o',(center-1,center+1),'true S peak')]
        fig,axes=plt.subplots(9,2,figsize=(13,21));shape,shape_axes=plt.subplots(4,2,figsize=(13,10))
        def draw(ax,b,target,span,color,centered=False):
            take=(t>=span[0])&(t<span[1]);v=valid[take,color]
            curves=[(bank[target][index],'target','black','-',1.1),(baseline[b][index],source_label,'#777777','--',.9)]
            curves += [(predictions[key][b][index],key,ink,style,.85) for key,(ink,style) in zip(predictions,styles)]
            for wave,label,ink,style,width in curves:
                y=np.asarray(wave).reshape(300,2)[take,color]*scale
                if centered:y=y-(y[v].mean() if v.any() else np.nan)
                ax.plot(t[take],y,label=label,c=ink,ls=style,lw=width)
            for edge in np.arange(0,31,2):
                if span[0]<=edge<=span[1]:ax.axvline(edge,c='gray',lw=.4,alpha=.35)
            ax.fill_between(t[take],0,1,where=~v,transform=ax.get_xaxis_transform(),color='gray',alpha=.25)
            ax.set(xlim=span,ylabel=b+' '+['HbO','HbR'][color]+'\n'+r['source_unit'],xlabel='time(s)');ax.tick_params(labelsize=8)
        for row,(b,target,span,label) in enumerate(views):
            for color in range(2):draw(axes[row,color],b,target,span,color);axes[row,color].set_title(label,fontsize=9)
        for row,(b,target,span) in enumerate([('S','s',(0,30)),('O','o',(0,30)),('S','s',(14,16)),('O','o',(14,16))]):
            for color in range(2):draw(shape_axes[row,color],b,target,span,color,True)
        title=textwrap.fill(f'{r["dataset"]}/{r["subject"]} row{i}+{start}s {r["positions"][position]} | '+','.join(selected[index]),width=135)
        fig.suptitle(title+'\nOriginal amplitude; gray=mask; vertical=2s tokens; all candidates preserve frozen111 X',fontsize=11)
        shape.suptitle(title+'\nShape diagnosis: each curve centered within displayed span; original amplitude retained separately',fontsize=11)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8);shape.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.tight_layout(rect=(0,.025,1,.97));shape.tight_layout(rect=(0,.045,1,.95))
        stem=f'{number:03d}_{r["dataset"]}_{r["subject"]}';main=folder/(stem+'_amplitude.png');centered=folder/(stem+'_centered.png')
        fig.savefig(main,dpi=180);shape.savefig(centered,dpi=180);plt.close(fig);plt.close(shape)
        manifest.append(dict(index=int(index),example=examples[index],dataset=r['dataset'],subject=r['subject'],position=r['positions'][position],selection=selected[index],
            amplitude=str(main),centered=str(centered),true_S_peak_time=peak/10,status='rendered_awaiting_actual_review'))
    write(folder/'manifest.json',dict(candidates=list(predictions),cases=manifest,selection='union of original fixed identity/random23017/G1 P90/G1 worst/each of four candidate worst; no waveform-based selection'))
    indices=np.array(sorted(selected));arrays={k:bank[k][indices] for k in ('x','s','o','mask')};arrays['examples']=examples[indices]
    arrays.update({'source111_'+b:baseline[b][indices] for b in baseline})
    arrays.update({key+'_'+b:v[indices] for key,p in predictions.items() for b,v in p.items()});np.savez_compressed(folder/'selected_examples.npz',**arrays)


def review_spectral_all(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import apply_spectral,load_model as load_pair
    out=args.output;folder=out/'four_arm_review_v1';folder.mkdir(exist_ok=True)
    if (folder/'completion.json').exists():raise FileExistsError('retained completed four-arm review')
    parent=ROOT/cfg['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config'];record=json.loads((out/'training_crossfit.json').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows']
    with np.load(out/'training_integer_ids.npz') as f:
        examples=f['examples'].copy()
        if len(examples)!=15360 or any(rows[i]['split']!='train' for i,_,_ in examples):raise ValueError('original train triples before four-arm arrays')
        ids={b:f[b].copy() for b in 'SO'}
    datasets=np.array([rows[i]['dataset'] for i,_,_ in examples]);folds=np.load(ROOT/old['source']['native_run']/'training_folds.npy')
    np.testing.assert_array_equal(folds,fold_assignment(examples,rows,old))
    path=ROOT/old['source']['bank']
    if digest(path)!=old['source']['bank_sha256']:raise ValueError('frozen training truth bank')
    with np.load(path) as f:
        np.testing.assert_array_equal(f['examples'],examples);bank={k:f[k].copy() for k in ('x','s','o','mask','vx','vs','vo','dvx','dvs','dvo')}
    for k in ('vx','vs','vo','dvx','dvs','dvo'):bank[k]=np.broadcast_to(np.tile(bank[k],(1,20))[:,None,:],bank['x'].shape)
    with np.load(out/'diagnostic_predictions.npz') as f:
        baseline={b:f['source_'+b].copy() for b in ('S','O','sum')};cached={b:f[b].copy() for b in ('S','O','sum')}
    previous=json.loads((out/'integer_replay.json').read_text());predictions={};review={};selected=defaultdict(list)
    threads=json.loads((out/'benchmark.json').read_text())['selected_threads']
    with threadpool_limits(threads):
        for candidate in record['candidates']:
            key=candidate['candidate'];maximum=closure=0.;models={fold:load_pair(out/f'{key}_fold{fold}.pt') for fold in range(4)}
            if previous['candidate']==key and previous['status']=='passed':
                recovered=cached;maximum=previous['maximum_waveform_difference'];closure=previous['sum_closure_max_abs'];origin='reused already independently integer-decoded full15360 arrays'
            else:
                recovered={b:np.empty_like(v) for b,v in baseline.items()};origin='fresh independent reload and saved IDs only'
                for fold,model in models.items():
                    model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('four-arm replay requested X'));held=np.flatnonzero(folds==fold)
                    for start in range(0,len(held),128):
                        take=held[start:start+128];code={b:v[take] for b,v in ids.items()};decoded=model.decode_ids(code,dataset=datasets[take]);source=model.source.decode_ids(code);_,q=validate_ids(code)
                        np.testing.assert_array_equal(decoded['sum'],source['sum'])
                        expected=apply_spectral(code,{b:v[take] for b,v in baseline.items()},model.fitted,dataset=datasets[take])
                        for b in decoded:
                            np.testing.assert_array_equal(decoded[b][q],source[b][q]);np.testing.assert_allclose(decoded[b],expected[b],atol=2e-6,rtol=1e-5)
                            maximum=max(maximum,float(abs(decoded[b]-expected[b]).max()));recovered[b][take]=decoded[b]
                        closure=max(closure,float(abs(decoded['S']+decoded['O']-decoded['sum']).max()))
            measured,summary=audit(recovered,bank,examples,rows,scales)
            with np.load(out/f'{key}_metrics.npz') as f:
                np.testing.assert_array_equal(f['examples'],examples);difference=float(np.nanmax(abs(f['nmse']-measured['metrics']['nmse'])))
                np.testing.assert_allclose(f['nmse'],measured['metrics']['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
            predictions[key]=recovered
            entry=dict(status='passed',origin=origin,maximum_waveform_difference=maximum,maximum_NMSE_difference=difference,sum_closure_max_abs=closure,
                total_and_quiet_exact_same_decode=True,summary=summary,wholewindow_parts=wholewindow_error_parts(recovered,bank,examples,rows),ablations={})
            for removed,name in enumerate(['DC','slow','dynamic']):
                ablated={b:np.empty_like(v) for b,v in baseline.items()}
                for fold,model in models.items():
                    held=np.flatnonzero(folds==fold);p=apply_spectral({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in baseline.items()},model.fitted,dataset=datasets[held],groups=[v for v in range(3) if v!=removed])
                    for b in ablated:ablated[b][held]=p[b]
                _,s=audit(ablated,bank,examples,rows,scales);entry['ablations']['remove_'+name]=dict(summary=s,wholewindow_parts=wholewindow_error_parts(ablated,bank,examples,rows))
            review[key]=entry;write(folder/'integer_replay_and_frequency_groups.json',dict(status='in_progress',candidates=review,validation_reads=0))
            choices=display_selection(argparse.Namespace(reference_metrics=out/'G1_metrics.npz'),measured['metrics'],rows,examples)
            for index,labels in choices.items():
                for label in labels:
                    tag=(key+':'+label) if label=='candidate_max' else label
                    if tag not in selected[index]:selected[index].append(tag)
            print('reviewed',key,'NMSE_max',difference,flush=True)
        write(folder/'integer_replay_and_frequency_groups.json',dict(status='completed',candidates=review,validation_reads=0,confirmation_reads=0,test_reads=0,
            group_omissions='counterfactual frozen coefficients only; no fitting or candidate reselection'))
        render_spectral_comparison(folder/'waveforms',selected,examples,bank,baseline,predictions,rows,scales)
    write(folder/'completion.json',dict(status='awaiting_actual_visual_review',candidates=4,cases=len(selected),refits=0,CPU_seconds=time.process_time()))


def validate_protected_config(cfg):
    if (cfg['schema']!='hb_spectral_protected_v1' or cfg['data']['allowed_splits']!=['train'] or cfg['source']['candidate']!='dataset__ridge1'
        or cfg['data']['confirmation_access']!='forbidden' or cfg['data']['test_access']!='forbidden'
        or cfg['fitting']['gain_bounds']!=[0.,1.] or cfg['fitting']['candidates']!=1 or cfg['model']['extra_integers']!=0
        or cfg['model']['groups_DCT_indices']!=[[0,1],[1,3],[3,13]] or cfg['model']['unencoded_continuous_inputs']):raise ValueError('single fit-only protected spectral contract required')


def run_spectral_protected(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import spectral_group_delta,protection_moments,fit_protected_gains,apply_spectral,load_model as load_pair
    validate_protected_config(cfg)
    if args.evaluate:raise ValueError('parent coordinates screen after original training numerical and visual verdict')
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():raise FileExistsError('completed protected spectral evidence retained')
    previous=ROOT/cfg['source']['spectral_run'];spec=json.loads((previous/'resolved_config.json').read_text())['config'];parent=ROOT/spec['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config']
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows'];write(out/'resolved_config.json',dict(config=cfg,unchanged_relative_rules=old['selection'],input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity']))
    examples,bank,ids,raw,g1,folds,heads,source=prepare_pair(old,out,manifest);datasets=np.array([rows[i]['dataset'] for i,_,_ in examples]);_,quiet=validate_ids(ids)
    for name in ['native_readout.pt','current.pt','model.pt']:shutil.copy2(previous/name,out/name)
    with np.load(parent/'diagnostic_predictions.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);baseline={b:f['source_'+b].copy() for b in ('S','O','sum')}
    base_summary=json.loads((parent/'source111_CV_summary.json').read_text());g1summary=json.loads((parent/'G1_summary.json').read_text())
    with np.load(parent/'source111_CV_metrics.npz') as f:base_metrics={k:f[k].copy() for k in f.files if k!='examples'}
    with np.load(previous/'G1_metrics.npz') as f:g1metrics={k:f[k].copy() for k in f.files if k!='examples'}
    prediction={b:np.empty_like(v) for b,v in baseline.items()};fits=[];scope=[]
    with threadpool_limits(1):
        for fold in range(4):
            fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
            fit_people=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[fit]});held_people=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[held]})
            if set(fit_people)&set(held_people):raise ValueError('outer person overlap')
            saved=torch.load(previous/f'dataset__ridge1_fold{fold}.pt',map_location='cpu',weights_only=False);fitted=deepcopy(saved['fitted'])
            post=apply_head(ids,raw,heads[fold]);directions=spectral_group_delta({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in post.items()},fitted,datasets[fit])
            fold_fits={}
            for dataset in sorted(set(datasets)):
                local=np.flatnonzero(datasets[fit]==dataset);use=fit[local];coefficient=np.asarray(fitted['coefficients'][dataset]).copy();colors=[]
                for color in range(2):
                    moment=protection_moments({b:v[use] for b,v in post.items()},{k:v[use] for k,v in bank.items()},directions[local],color,training_weights(examples[use],rows),
                        finite_weights=lambda finite:training_weights(examples[use][finite],rows))
                    result=fit_protected_gains(moment);result.update(fit_windows=len(use),fit_subjects=sorted({rows[i]['subject'] for i,_,_ in examples[use]}));colors.append(result)
                    for group,gain in enumerate(result['gain']):coefficient[group*4:(group+1)*4,color]*=gain
                fitted['coefficients'][dataset]=coefficient;fold_fits[dataset]=colors
            fitted['protection']='fit-only bounded three-band output-color shrink';fitted['base_ridge']=1.;fits.append(dict(fold=fold,groups=fold_fits))
            saved['fitted']=fitted;saved['metadata'].update(schema='hb_spectral_protected_v1',protection_gains_fit_only=True)
            torch.save(saved,out/f'protected_fold{fold}.pt')
            p=apply_spectral({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in post.items()},fitted,dataset=datasets[held])
            for b in prediction:prediction[b][held]=p[b]
            scope.append(dict(fold=fold,fit_people=fit_people,held_people=held_people,source111_head=fold,spectral_head=fold,held_targets_used_for_fitting=False))
            write(out/'fit_progress.json',dict(completed_folds=fold+1,CPU_seconds=time.process_time(),fits=fits));print('protected_fold',fold,'CPU',time.process_time(),flush=True)
        write(out/'fitting.json',dict(fits=fits,scope=scope,new_parameters_fit_only=True,held_gain_selection=False,base_source_exposure=old['training']['exposure']))
        measured,summary=audit(prediction,bank,examples,rows,scales);peaks=pair_peak_error(prediction,bank,examples,rows)
        decision=pair_decision(summary,base_summary,g1summary,measured['metrics'],base_metrics,g1metrics,examples,rows,folds,old,peaks,pair_peak_error(baseline,bank,examples,rows))
        contract=yaml.safe_load((ROOT/spec['selection']['absolute_owner']).read_text());reference=dict(g1summary,modality='hb',role='original_train_crossfit')
        absolute=score(dict(summary,modality='hb',role='original_train_crossfit',checkpoint='protected',integer_replay=dict(status='pending')),reference,reference,contract)
        passed=decision['eligible'] and absolute['finite_predictions'] and all(v['passed'] for v in absolute['checks'].values())
        np.savez_compressed(out/'metrics.npz',examples=examples,**measured['metrics']);write(out/'training_crossfit.json',dict(summary=summary,relative=decision,absolute=absolute,numerically_eligible=passed))
        recovered={b:np.empty_like(v) for b,v in prediction.items()};maximum=closure=0.
        for fold in range(4):
            model=load_pair(out/f'protected_fold{fold}.pt');model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('integer replay requested X'));held=np.flatnonzero(folds==fold)
            for start in range(0,len(held),128):
                take=held[start:start+128];codes={b:v[take] for b,v in ids.items()};p=model.decode_ids(codes,dataset=datasets[take]);same=model.source.decode_ids(codes);_,q=validate_ids(codes)
                np.testing.assert_array_equal(p['sum'],same['sum'])
                for b in p:
                    np.testing.assert_array_equal(p[b][q],same[b][q]);np.testing.assert_allclose(p[b],prediction[b][take],atol=2e-6,rtol=1e-5);maximum=max(maximum,float(abs(p[b]-prediction[b][take]).max()));recovered[b][take]=p[b]
                closure=max(closure,float(abs(p['S']+p['O']-p['sum']).max()))
        replayed,_=audit(recovered,bank,examples,rows,scales);difference=float(np.nanmax(abs(replayed['metrics']['nmse']-measured['metrics']['nmse'])))
        np.testing.assert_allclose(replayed['metrics']['nmse'],measured['metrics']['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
        write(out/'integer_replay.json',dict(status='passed',maximum_waveform_difference=maximum,maximum_NMSE_difference=difference,total_and_quiet_exact=True,sum_closure_max_abs=closure,raw_X_reads=0))
        choices=display_selection(argparse.Namespace(reference_metrics=previous/'G1_metrics.npz'),measured['metrics'],rows,examples)
        prior=json.loads((previous/'four_arm_review_v1/waveforms/manifest.json').read_text())
        for item in prior['cases']:
            labels=choices.setdefault(int(item['index']),[])
            if 'previous_four_arm_fixed_panel' not in labels:labels.append('previous_four_arm_fixed_panel')
        unprotected={b:np.empty_like(v) for b,v in baseline.items()}
        for fold in range(4):
            held=np.flatnonzero(folds==fold);saved=torch.load(previous/f'dataset__ridge1_fold{fold}.pt',map_location='cpu',weights_only=False)
            p=apply_spectral({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in baseline.items()},saved['fitted'],dataset=datasets[held])
            for b in p:unprotected[b][held]=p[b]
        render_spectral_comparison(out/'waveforms',choices,examples,bank,baseline,dict(unprotected_ridge1=unprotected,protected=recovered),rows,scales)
        write(out/'wholewindow_parts.json',dict(source111=wholewindow_error_parts(baseline,bank,examples,rows),protected=wholewindow_error_parts(recovered,bank,examples,rows)))
        write(out/'summary.json',dict(status='numeric_eligible_awaiting_actual_visual_review' if passed else 'train_crossfit_rejected',numerically_eligible=passed,joint_NMSE=float(np.asarray(summary['metrics']['nmse']['macro'])[1:].mean()),
            original_gates_unchanged=True,screen_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))
        print('protected_final',passed,decision['joint_relative_improvement'],[k for k,v in decision['checks'].items() if not v],flush=True)


def validate_phase_config(cfg):
    if (cfg['schema']!='hb_phase_companion_v1' or cfg['data']['allowed_splits']!=['train'] or cfg['data']['confirmation_access']!='forbidden'
        or cfg['data']['test_access']!='forbidden' or cfg['model']['groups_DCT_indices']!=[[1,3],[3,13]]
        or cfg['model']['virtual_edges_q']!=[0,300] or cfg['model']['context_seconds']!=30 or cfg['model']['rate_hz']!=10
        or cfg['oracle']['spaces']!=['even','odd','even_plus_odd'] or cfg['oracle']['numerical_rank_relative_cutoff']!=1e-10):
        raise ValueError('train-only non-circular Hb phase reachability contract required')


def run_phase_companion(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import phase_oracle,phase_design,load_model as load_pair
    validate_phase_config(cfg)
    if args.evaluate:raise ValueError('phase reachability upper bound must be adjudicated before any development')
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():raise FileExistsError('retained completed phase diagnostic')
    protected=ROOT/cfg['source']['protected_run'];pc=json.loads((protected/'resolved_config.json').read_text())['config'];spectral=ROOT/pc['source']['spectral_run']
    sc=json.loads((spectral/'resolved_config.json').read_text())['config'];parent=ROOT/sc['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config']
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows']
    write(out/'resolved_config.json',dict(config=cfg,unchanged_relative_rules=old['selection'],input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'],role='target-informed per-window reachability, not deployable'))
    examples,bank,ids,_,g1,folds,_,_=prepare_pair(old,out,manifest);labels=np.array([rows[i]['dataset'] for i,_,_ in examples]);_,quiet=validate_ids(ids)
    with np.load(parent/'diagnostic_predictions.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);baseline111={b:f['source_'+b].copy() for b in ('S','O','sum')}
    base_summary=json.loads((parent/'source111_CV_summary.json').read_text());g1summary=json.loads((parent/'G1_summary.json').read_text())
    with np.load(parent/'source111_CV_metrics.npz') as f:base_metrics={k:f[k].copy() for k in f.files if k!='examples'}
    with np.load(spectral/'G1_metrics.npz') as f:g1metrics={k:f[k].copy() for k in f.files if k!='examples'}
    source={b:np.empty_like(v) for b,v in baseline111.items()};predictions={};records=[];ranks={};reproduction={};selected=defaultdict(list)
    with threadpool_limits(1):
        # Checkpoints are independently loaded, and no source encoder receives X.
        for fold in range(4):
            model=load_pair(protected/f'protected_fold{fold}.pt');model.encode=lambda *_:(_ for _ in ()).throw(AssertionError('phase source requested X'));held=np.flatnonzero(folds==fold)
            for start in range(0,len(held),128):
                take=held[start:start+128];p=model.decode_ids({b:v[take] for b,v in ids.items()},dataset=labels[take])
                for b in source:source[b][take]=p[b]
        measured_source,source_summary=audit(source,bank,examples,rows,scales)
        with np.load(protected/'metrics.npz') as f:np.testing.assert_allclose(measured_source['metrics']['nmse'],f['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
        write(out/'source_protected_summary.json',source_summary)
        np.savez_compressed(out/'source_protected_metrics.npz',examples=examples,**measured_source['metrics'])
        timings=[]
        for count in [64,256]:
            take=slice(0,count);start=time.process_time();begin=time.monotonic()
            phase_oracle({b:v[take] for b,v in ids.items()},{b:v[take] for b,v in source.items()},{k:v[take] for k,v in bank.items()},'even_plus_odd',batch_size=128)
            timings.append(dict(windows=count,threads=1,CPU_seconds=time.process_time()-start,wall_seconds=time.monotonic()-begin))
        write(out/'benchmark.json',dict(timings=timings,threads=1,reason='at most300x16 per-window SVD; one thread avoids tiny-matrix oversubscription',planned_window_fits=3*15360))
        contract=yaml.safe_load((ROOT/sc['selection']['absolute_owner']).read_text());reference=dict(g1summary,modality='hb',role='train_target_informed_oracle')
        peaks0=pair_peak_error(baseline111,bank,examples,rows)
        for space in cfg['oracle']['spaces']:
            begin=time.process_time();p,coefficients,rank=phase_oracle(ids,source,bank,space,batch_size=cfg['resources']['batch_size']);ranks[space]=rank
            role='target_informed_'+space;predictions[role]=p
            np.savez_compressed(out/(space+'_oracle_coefficients.npz'),examples=examples,coefficients=coefficients,numerical_rank=rank)
            measured,summary=audit(p,bank,examples,rows,scales);relative=pair_decision(summary,base_summary,g1summary,measured['metrics'],base_metrics,g1metrics,examples,rows,folds,old,pair_peak_error(p,bank,examples,rows),peaks0)
            absolute=score(dict(summary,modality='hb',role='train_target_informed_oracle',checkpoint=role,integer_replay=dict(status='not_integer_only_truth_fitted_coefficients')),reference,reference,contract)
            np.savez_compressed(out/(space+'_metrics.npz'),examples=examples,**measured['metrics'])
            numerical=relative['eligible'] and absolute['finite_predictions'] and all(v['passed'] for v in absolute['checks'].values())
            records.append(dict(space=space,role='target-informed per-window oracle; each scored target used to fit its own coefficients',summary=summary,
                joint_NMSE=float(np.asarray(summary['metrics']['nmse']['macro'])[1:].mean()),relative_checks=relative,absolute_checks=absolute,numerical_all_original_pass=numerical,
                deployable=False,nominal_continuous_coefficients_per_window=int(coefficients.shape[1]*coefficients.shape[2]),rank_quantiles=np.quantile(rank,[0,.1,.5,.9,1],axis=0),
                max_abs_target_fitted_coefficient=float(abs(coefficients).max()),CPU_seconds=time.process_time()-begin))
            with np.load(out/(space+'_oracle_coefficients.npz')) as f:
                np.testing.assert_array_equal(f['examples'],examples);disk=f['coefficients'].copy()
            maximum=0.;closure=0.
            for start in range(0,len(examples),128):
                take=slice(start,start+128);a=phase_design({b:v[take] for b,v in source.items()},space);delta=np.einsum('ntg,ncg->ntc',a,disk[take]).reshape(-1,15,40)
                for b,sign in [('S',1),('O',-1)]:
                    recovered=source[b][take]+sign*delta;np.testing.assert_allclose(recovered,p[b][take],atol=2e-10,rtol=1e-10);maximum=max(maximum,float(abs(recovered-p[b][take]).max()))
                np.testing.assert_array_equal(p['sum'][take],source['sum'][take]);closure=max(closure,float(abs(p['S'][take]+p['O'][take]-p['sum'][take]).max()))
                for b in p:np.testing.assert_array_equal(p[b][take][quiet[take]],source[b][take][quiet[take]])
            reproduction[space]=dict(status='passed_with_explicit_target_fitted_continuous_coefficients',maximum_waveform_difference=maximum,sum_closure_max_abs=closure,not_integer_only=True,quiet_and_sum_exact=True)
            choices=display_selection(argparse.Namespace(reference_metrics=spectral/'G1_metrics.npz'),measured['metrics'],rows,examples)
            for index,tags in choices.items():
                for tag in tags:
                    tag=role+':'+tag if tag=='candidate_max' else tag
                    if tag not in selected[index]:selected[index].append(tag)
            write(out/'oracle_scores.json',dict(status='in_progress',candidates=records,selection=False,validation_reads=0));print('phase_oracle',space,records[-1]['joint_NMSE'],numerical,'CPU',time.process_time(),flush=True)
        original=json.loads((protected/'waveforms/manifest.json').read_text())['cases'];fixed=[]
        for case in original:
            index=int(case['index']);selected[index].append('original17');mask=bank['mask'][index].reshape(300,2);values={}
            for branch,key in [('S','s'),('O','o')]:
                target=bank[key][index].reshape(300,2);by_color=[]
                for color in range(2):
                    good=mask[:,color];y=target[:,color];scores={};indices=np.flatnonzero(good);true_peak=indices[np.argmax(abs(y[good]))]
                    for name,p in dict(frozen111=baseline111,frozen_protected=source,**predictions).items():
                        z=p[branch][index].reshape(300,2)[:,color];center=lambda v:v[good]-v[good].mean()
                        scores[name]=dict(centered_RMS_gain=float(np.linalg.norm(center(z))/np.linalg.norm(center(y))),centered_correlation=float(np.corrcoef(y[good],z[good])[0,1]),
                            maximum_time_s=indices[np.argmax(z[good])]/10,minimum_time_s=indices[np.argmin(z[good])]/10,at_true_abs_peak_ratio=float(z[true_peak]/y[true_peak]))
                    by_color.append(dict(target_maximum_time_s=indices[np.argmax(y[good])]/10,target_minimum_time_s=indices[np.argmin(y[good])]/10,target_abs_peak_time_s=true_peak/10,models=scores))
                values[branch]=by_color
            fixed.append(dict(example=examples[index],index=index,dataset=case['dataset'],subject=case['subject'],diagnostics=values))
        write(out/'fixed17_diagnostics.json',dict(role='target-informed low-dimensional reachability',cases=fixed,validation_reads=0))
        write(out/'oracle_scores.json',dict(status='completed',candidates=records,selection=False,validation_reads=0,confirmation_reads=0,test_reads=0,fold_results='descriptive partitions only, oracle uses each scored target and is not cross-fitted'))
        write(out/'oracle_reproduction.json',dict(candidates=reproduction,source_integer_replay_verified=True,oracle_is_not_integer_only=True))
        write(out/'new_freedom_and_boundary.json',dict(same_frequency_cosine_sine_dot='zero analytically on the300-point half-offset grid for equal k',
            original_shared_cosine_features=8,new_shared_sine_features=8,rank_increment_quantiles=np.quantile(ranks['even_plus_odd']-ranks['even'],[0,.1,.5,.9,1],axis=0),
            zero_increment_windows=int(np.sum(np.all(ranks['even_plus_odd']==ranks['even'],axis=1))),quiet_windows=int(quiet.sum()),
            boundary='sine modes zero at virtual q0/300, physical edges-.05/29.95seconds; no circular roll; sampled endpoints remain scored',
            rationale='limited cosine mixing cannot create the matched input same-frequency sine direction; full all-frequency DCT would be a different larger space'))
        render_spectral_comparison(out/'waveforms',selected,examples,bank,source,predictions,rows,scales,source_label='frozen_protected')
        write(out/'summary.json',dict(status='oracle_complete_awaiting_actual_visual_adjudication',deployable=False,oracle_scores='oracle_scores.json',cases=len(selected),screen_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))


def validate_phase_fit_config(cfg):
    if (cfg['schema']!='hb_phase_fit_v1' or cfg['data']['allowed_splits']!=['train'] or cfg['data']['confirmation_access']!='forbidden'
        or cfg['data']['test_access']!='forbidden' or cfg['fitting']['ridge']!=1. or cfg['fitting']['candidates']!=1
        or cfg['fitting']['gain_bounds']!=[0.,1.] or cfg['model']['extra_integers']!=0 or cfg['model']['groups_DCT_indices']!=[[1,3],[3,13]]
        or cfg['model']['context_seconds']!=30 or cfg['model']['rate_hz']!=10 or cfg['model']['unencoded_continuous_inputs']):
        raise ValueError('single train-only protected phase fit contract required')


def run_phase_fit(args,cfg):
    from threadpoolctl import threadpool_limits
    from src.tokenizers.hb_pair_allocation import phase_statistics,fit_phase,phase_group_delta,apply_phase,protection_moments,protection_values,fit_protected_gains,load_model as load_pair
    validate_phase_fit_config(cfg)
    if args.evaluate:raise ValueError('frozen phase training verdict required before development')
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    if (out/'summary.json').exists():raise FileExistsError('retained completed phase fitting')
    oracle=ROOT/cfg['source']['oracle_run'];protected=ROOT/cfg['source']['protected_run']
    review=json.loads((oracle/'visual_review.json').read_text())
    if not review['phase_reachability_supports_unique_fit_only']:raise ValueError('actual oracle visual review did not admit unique fit-only comparison')
    pc=json.loads((protected/'resolved_config.json').read_text())['config'];spectral=ROOT/pc['source']['spectral_run'];sc=json.loads((spectral/'resolved_config.json').read_text())['config']
    parent=ROOT/sc['source']['pair_run'];old=json.loads((parent/'resolved_config.json').read_text())['config']
    manifest,scales,_,_=base_tokenizer_training_data(old,ROOT);rows=manifest['rows']
    write(out/'resolved_config.json',dict(config=cfg,unchanged_relative_rules=old['selection'],input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity']))
    examples,bank,ids,_,_,folds,_,_=prepare_pair(old,out,manifest);labels=np.array([rows[i]['dataset'] for i,_,_ in examples])
    for name in ['native_readout.pt','current.pt','model.pt']:shutil.copy2(protected/name,out/name)
    with np.load(parent/'diagnostic_predictions.npz') as f:
        np.testing.assert_array_equal(f['examples'],examples);base={b:f['source_'+b].copy() for b in ('S','O','sum')}
    base_summary=json.loads((parent/'source111_CV_summary.json').read_text());g1summary=json.loads((parent/'G1_summary.json').read_text())
    with np.load(parent/'source111_CV_metrics.npz') as f:base_metrics={k:f[k].copy() for k in f.files if k!='examples'}
    with np.load(spectral/'G1_metrics.npz') as f:g1metrics={k:f[k].copy() for k in f.files if k!='examples'}
    prediction={b:np.empty_like(v) for b,v in base.items()};protected_oof={b:np.empty_like(v) for b,v in base.items()};fits=[];scope=[]
    with threadpool_limits(1):
        for fold in range(4):
            fit=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold);fit_people=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[fit]});held_people=sorted({(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in examples[held]})
            if set(fit_people)&set(held_people):raise ValueError('phase outer person overlap')
            model=load_pair(protected/f'protected_fold{fold}.pt');model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('phase fit requested X'))
            post={b:np.empty_like(v) for b,v in base.items()};post111={b:np.empty_like(v) for b,v in base.items()}
            for start in range(0,len(examples),128):
                take=slice(start,start+128);codes={b:v[take] for b,v in ids.items()};p=model.decode_ids(codes,dataset=labels[take]);p111=model.source.decode_ids(codes)
                for b in post:post[b][take]=p[b];post111[b][take]=p111[b]
            fitted=fit_phase(phase_statistics({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in post.items()},{k:v[fit] for k,v in bank.items()},training_weights(examples[fit],rows),labels[fit]),ridge=cfg['fitting']['ridge'])
            directions=-phase_group_delta({b:v[fit] for b,v in ids.items()},{b:v[fit] for b,v in post.items()},fitted,labels[fit]);fold_fits={}
            for dataset in sorted(set(labels)):
                local=np.flatnonzero(labels[fit]==dataset);use=fit[local];coefficient=np.asarray(fitted['coefficients'][dataset]).copy();colors=[]
                for color in range(2):
                    weights=training_weights(examples[use],rows);kwargs=dict(finite_weights=lambda finite:training_weights(examples[use][finite],rows))
                    moment=protection_moments({b:v[use] for b,v in post.items()},{k:v[use] for k,v in bank.items()},directions[local],color,weights,**kwargs)
                    reference_moment=protection_moments({b:v[use] for b,v in post111.items()},{k:v[use] for k,v in bank.items()},directions[local],color,weights,**kwargs)
                    reference=protection_values(reference_moment,np.zeros(4));result=fit_protected_gains(moment,protection_reference=reference)
                    result.update(fit_windows=len(use),fit_subjects=sorted({rows[i]['subject'] for i,_,_ in examples[use]}));colors.append(result)
                    for group,gain in enumerate(result['gain']):coefficient[group*4:(group+1)*4,color]*=gain
                fitted['coefficients'][dataset]=coefficient;fold_fits[dataset]=colors
            saved=torch.load(protected/f'protected_fold{fold}.pt',map_location='cpu',weights_only=False);saved['protected_fitted']=saved['fitted'];saved['fitted']=fitted;saved['metadata'].update(schema='hb_phase_fit_v1',phase_gains_fit_only=True)
            torch.save(saved,out/f'phase_fold{fold}.pt')
            p=apply_phase({b:v[held] for b,v in ids.items()},{b:v[held] for b,v in post.items()},fitted,labels[held])
            for b in prediction:prediction[b][held]=p[b];protected_oof[b][held]=post[b][held]
            fits.append(dict(fold=fold,groups=fold_fits));scope.append(dict(fold=fold,fit_people=fit_people,held_people=held_people,source111_and_protected_head=fold,held_targets_used_for_fitting=False))
            write(out/'fit_progress.json',dict(completed_folds=fold+1,CPU_seconds=time.process_time(),fits=fits));print('phase_fit_fold',fold,'CPU',time.process_time(),flush=True)
        write(out/'fitting.json',dict(fits=fits,scope=scope,held_gain_or_phase_selection=False,base_source_exposure=old['training']['exposure']))
        measured,summary=audit(prediction,bank,examples,rows,scales);relative=pair_decision(summary,base_summary,g1summary,measured['metrics'],base_metrics,g1metrics,examples,rows,folds,old,pair_peak_error(prediction,bank,examples,rows),pair_peak_error(base,bank,examples,rows))
        contract=yaml.safe_load((ROOT/sc['selection']['absolute_owner']).read_text());reference=dict(g1summary,modality='hb',role='original_train_crossfit')
        absolute=score(dict(summary,modality='hb',role='original_train_crossfit',checkpoint='phase',integer_replay=dict(status='pending')),reference,reference,contract)
        passed=relative['eligible'] and absolute['finite_predictions'] and all(v['passed'] for v in absolute['checks'].values())
        np.savez_compressed(out/'metrics.npz',examples=examples,**measured['metrics']);write(out/'training_crossfit.json',dict(summary=summary,relative=relative,absolute=absolute,numerically_eligible=passed))
        recovered={b:np.empty_like(v) for b,v in prediction.items()};maximum=closure=0.
        for fold in range(4):
            model=load_pair(out/f'phase_fold{fold}.pt');model.source.encode=lambda *_:(_ for _ in ()).throw(AssertionError('integer replay requested X'));held=np.flatnonzero(folds==fold)
            for start in range(0,len(held),128):
                take=held[start:start+128];codes={b:v[take] for b,v in ids.items()};p=model.decode_ids(codes,dataset=labels[take]);same=model.source.decode_ids(codes);_,q=validate_ids(codes)
                np.testing.assert_array_equal(p['sum'],same['sum'])
                for b in p:
                    np.testing.assert_array_equal(p[b][q],same[b][q]);np.testing.assert_allclose(p[b],prediction[b][take],atol=2e-6,rtol=1e-5);maximum=max(maximum,float(abs(p[b]-prediction[b][take]).max()));recovered[b][take]=p[b]
                closure=max(closure,float(abs(p['S']+p['O']-p['sum']).max()))
        replayed,_=audit(recovered,bank,examples,rows,scales);difference=float(np.nanmax(abs(replayed['metrics']['nmse']-measured['metrics']['nmse'])))
        np.testing.assert_allclose(replayed['metrics']['nmse'],measured['metrics']['nmse'],atol=1e-12,rtol=1e-12,equal_nan=True)
        write(out/'integer_replay.json',dict(status='passed',maximum_waveform_difference=maximum,maximum_NMSE_difference=difference,total_and_quiet_exact=True,sum_closure_max_abs=closure,raw_X_reads=0))
        choices=display_selection(argparse.Namespace(reference_metrics=spectral/'G1_metrics.npz'),measured['metrics'],rows,examples)
        for item in json.loads((protected/'waveforms/manifest.json').read_text())['cases']:choices.setdefault(int(item['index']),[]).append('original17')
        render_spectral_comparison(out/'waveforms',choices,examples,bank,base,dict(frozen_protected=protected_oof,fit_only_phase=recovered),rows,scales)
        write(out/'wholewindow_parts.json',dict(source111=wholewindow_error_parts(base,bank,examples,rows),phase=wholewindow_error_parts(recovered,bank,examples,rows)))
        write(out/'summary.json',dict(status='numeric_eligible_awaiting_actual_visual_review' if passed else 'train_crossfit_rejected',numerically_eligible=passed,joint_NMSE=float(np.asarray(summary['metrics']['nmse']['macro'])[1:].mean()),
            original_gates_unchanged=True,screen_read=False,confirmation_read=False,test_read=False,CPU_seconds=time.process_time()))
        print('phase_fit_final',passed,relative['joint_relative_improvement'],[k for k,v in relative['checks'].items() if not v],flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", choices=["screen", "development"])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--allocation-diagnostics", action="store_true")
    parser.add_argument('--gain-replay-only',action='store_true')
    parser.add_argument('--spectral-review-all',action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.gain_replay_only and cfg['schema']!='hb_pair_signed_gain_v1':
        raise ValueError('gain replay flag requires the signed-gain contract')
    if args.spectral_review_all and cfg['schema']!='hb_spectral_allocation_v1':
        raise ValueError('four-arm spectral review flag requires its spectral contract')
    if (args.benchmark or args.allocation_diagnostics) and cfg['schema']!='hb_pair_allocation_v1':
        raise ValueError('allocation-only diagnostic flag requires its versioned config')
    if (
        cfg["schema"] not in ("hb_native_id_readout_v1", "hb_pair_allocation_v1", "hb_pair_signed_gain_v1", "hb_spectral_allocation_v1", "hb_spectral_protected_v1", "hb_phase_companion_v1", "hb_phase_fit_v1")
        or cfg["data"]["test_access"] != "forbidden"
    ):
        raise ValueError("versioned frozen-ID train-first contract required")
    torch.set_num_threads(cfg["resources"]["torch_threads"])
    resource.setrlimit(
        resource.RLIMIT_CPU, (cfg["resources"]["process_CPU_limit_seconds"],) * 2
    )
    with owner_lock(args.output):
        try:
            if cfg["schema"] == "hb_pair_allocation_v1":
                run_pair_allocation(args, cfg)
            elif cfg['schema']=='hb_pair_signed_gain_v1':
                run_pair_signed_gain(args,cfg)
            elif cfg['schema']=='hb_spectral_allocation_v1':
                run_pair_spectral(args,cfg)
            elif cfg['schema']=='hb_spectral_protected_v1':
                run_spectral_protected(args,cfg)
            elif cfg['schema']=='hb_phase_companion_v1':
                run_phase_companion(args,cfg)
            elif cfg['schema']=='hb_phase_fit_v1':
                run_phase_fit(args,cfg)
            else:
                run(args, cfg)
        except BaseException as e:
            write(
                args.output / "failure.json",
                dict(
                    type=type(e).__name__,
                    reason=str(e),
                    CPU_seconds=time.process_time(),
                ),
            )
            raise


if __name__ == "__main__":
    import itertools

    main()
