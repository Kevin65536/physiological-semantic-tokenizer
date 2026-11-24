"""Learn one atomic supervised child ID per frozen G1 Hb-O parent assignment."""

import argparse
from collections import defaultdict, OrderedDict
import fcntl
import hashlib
import heapq
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_joint_tokenizer import write, input_scales, load_source as load_panel
from run_dual_component_tokenizer import identity
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_batch,
    base_tokenizer_examples,
)
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
)
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.tokenizers.hb_parent_leaf import (
    load_parent,
    encode_parent,
    fit_children,
    assign_leaf,
    synthesize_parent,
    HbParentLeaf,
    load_model,
    emitted_error_polynomials,
)
from audit_joint_tokenizer_patterns import pattern_attributes


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def expansion_plan(rows, original, spec):
    """Add only existing train score windows, balancing records before labels."""
    original=[tuple(map(int,x)) for x in original]
    members={(rows[i]['dataset'],rows[i]['subject']) for i,_,_ in original}
    legal={tuple(x) for x in base_tokenizer_examples(rows,'hb','train') if (rows[x[0]]['dataset'],rows[x[0]]['subject']) in members}
    if not set(original).issubset(legal):raise ValueError('all original fitting windows must remain legal')
    def physical(e):
        i,t,p=e;r=rows[i]
        return (r['dataset'],r['subject'],r['record'],r['positions'][p]),r['native_start_sample']/r['native_rate_hz']+t
    occupied=defaultdict(list)
    for e in original:
        key,t=physical(e);occupied[key].append(t)
    existing_overlap=sum(sum(b-a<30-1e-8 for a,b in zip(sorted(v),sorted(v)[1:])) for v in occupied.values())
    candidates=defaultdict(lambda:defaultdict(list));counts=defaultdict(lambda:defaultdict(int))
    for i,t,p in original:counts[rows[i]['dataset'],rows[i]['subject']][rows[i]['record']]+=1
    for e in legal-set(original):
        r=rows[e[0]];candidates[r['dataset'],r['subject']][r['record']].append(e)
    added=[];support=[]
    for member in sorted(members):
        queues={r:sorted(v,key=lambda e:digest_text([spec['selection_seed'],*member,r,*e]),reverse=True) for r,v in candidates[member].items()}
        heap=[(counts[member][r],digest_text([spec['selection_seed'],*member,r]),r) for r in queues];heapq.heapify(heap)
        total=sum(counts[member].values());before_records={r for r,v in counts[member].items() if v};local=[];skipped=0
        while total<spec['maximum_windows_per_subject'] and heap:
            count,rank,record=heapq.heappop(heap);queue=queues[record];chosen=None
            while queue:
                e=queue.pop();key,t=physical(e)
                if all(abs(t-old)>=30-1e-8 for old in occupied[key]):chosen=e;break
                skipped+=1
            if chosen is None:continue
            key,t=physical(chosen);occupied[key].append(t);local.append(chosen);total+=1;counts[member][record]+=1
            if queue:heapq.heappush(heap,(count+1,rank,record))
        added.extend(local);after_records={r for r,v in counts[member].items() if v}
        support.append(dict(dataset=member[0],subject=member[1],original_windows=total-len(local),added_windows=len(local),expanded_windows=total,original_records=len(before_records),expanded_records=len(after_records),new_records=sorted(after_records-before_records),by_record=dict(counts[member]),overlap_candidates_skipped=skipped))
    expanded=np.asarray(original+sorted(added),np.int32)
    if len({tuple(e) for e in expanded})!=len(expanded):raise ValueError('unique fitting window IDs required')
    return expanded,dict(subjects=support,original_contexts=len(original),added_contexts=len(added),expanded_contexts=len(expanded),original_overlap_pairs=existing_overlap,new_overlapping_windows=0,original_preserved_as_prefix=True,additional_subjects=0,score_role_only=True,selection_uses_signal_or_supervision=False)


def expand_collection(cfg,out,rows,scales,examples,parent):
    old_count=cfg['training']['expanded_fit']['evaluation_contexts']
    chunks=out/'fit_chunks';chunks.mkdir(exist_ok=True);entries=[]
    for start in range(old_count,len(examples),cfg['training']['expanded_fit']['chunk_windows']):
        stop=min(len(examples),start+cfg['training']['expanded_fit']['chunk_windows']);folder=chunks/f'{start:06d}';folder.mkdir(exist_ok=True)
        path=folder/'training_bank.npz'
        if path.exists():
            with np.load(path) as saved:np.testing.assert_array_equal(saved['examples'],examples[start:stop])
        else:collection(cfg,folder,rows,scales,examples[start:stop],parent)
        entries.append(dict(start=start,stop=stop,path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size))
        write(out/'fit_chunks.json',dict(status='in_progress',original_bank=cfg['source']['training_bank'],chunks=entries,CPU_seconds=time.process_time()))
        print('collected',stop-old_count,'/',len(examples)-old_count,'added windows',flush=True)
    write(out/'fit_chunks.json',dict(status='completed',original_bank=cfg['source']['training_bank'],chunks=entries,CPU_seconds=time.process_time()))


def expanded_readout_audit(cfg,out,rows,examples,bank):
    """Separate frozen routing and conditional-template sample sizes, without selection."""
    n_eval=cfg['training']['expanded_fit']['evaluation_contexts'];n=len(examples)
    plan=json.loads((out/'training_folds.json').read_text());folds=np.asarray(plan['folds']);subjects=np.asarray(plan['subjects'])
    valid=bank['mask'].all(-1);quiet,_=quiet_state(bank,cfg)
    attributes,fields=pattern_attributes(bank['o'].reshape(-1,40),'hb');attributes=attributes.reshape(n,15,-1)
    sites=sorted({(rows[i]['dataset'],rows[i]['positions'][p]) for i,_,p in examples});site={s:j for j,s in enumerate(sites)}
    locations=np.array([site[rows[i]['dataset'],rows[i]['positions'][p]] for i,_,p in examples])
    bg=locations[:,None]*4+np.searchsorted([.5,1.,2.],np.sqrt(np.mean(bank['x']**2,axis=-1)),side='right')
    results={}
    for routing in ['v2_ridge',out.name]:
        values={fit_n:dict(errors=np.full((n_eval,2,len(fields)),np.nan),support=np.zeros(n_eval)) for fit_n in [n_eval,n]}
        for fold in range(cfg['training']['folds']):
            fitted=torch.load(out.parent/routing/f'leaves4_fold{fold}.pt',map_location='cpu',weights_only=False)['fitted']
            leaf=assign_leaf(bank['id_O'],bank['features'],fitted);leaf[quiet]=len(fitted['parent'])
            held=np.flatnonzero(folds[:n_eval]==fold)
            for fit_n,v in values.items():
                train=np.flatnonzero(folds[:fit_n]!=fold)
                error,support=pattern_fold(leaf,leaf,bg,attributes,fields,valid,train,held,subjects)
                v['errors'][held]=error;v['support'][held]=support
        for fit_n,v in values.items():
            label=f'{routing}__readout_{fit_n}'
            results[label]=pattern_summary(v['errors'],v['support'],fields,examples[:n_eval],rows)
        anchor=json.loads((out.parent/routing/'training_crossfit.json').read_text())
        reference=next(c for c in anchor['candidates'] if c['candidate']=='leaves4_shape0.25')['pattern']
        comparison=results[f'{routing}__readout_{n_eval if routing=="v2_ridge" else n}']
        np.testing.assert_array_equal(list(comparison['gain'].values()),list(reference['gain'].values()))
    write(out/'scale_readout_audit.json',dict(status='completed_diagnostic_only',evaluation_contexts=n_eval,
          same_subject_folds=True,new_model_fitting_or_hyperparameters=False,validation_read=False,
          purpose='cross frozen routing size with independently fitted train-only background and background+ID sample size',conditions=results,CPU_seconds=time.process_time()))


def prepare_parent(cfg, manifest, scales):
    source_cfg = yaml.safe_load((ROOT / cfg["source"]["config"]).read_text())
    path = ROOT / cfg["source"]["checkpoint"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (
        saved["step"] != 10000
        or saved["signature"]["config"] != identity(source_cfg)
        or saved["signature"]["input"] != identity(manifest)
        or saved["signature"]["scales"] != identity(scales)
        or saved["signature"]["task"]
        != dict(modality="hb", architecture="waveform", candidate="vq128")
    ):
        raise ValueError("exact original G1 v3 K128 parent required")
    setting = dict(
        source_cfg["model"],
        **source_cfg["architectures"]["waveform"],
        input_dim=40,
        channels=2,
        sample_strides=[5],
    )
    meta = dict(
        schema="original_G1_Hb_parent",
        modality="hb",
        codebook_size=128,
        input_identity=manifest["cfg_identity"],
        scales_identity=manifest["scales_identity"],
    )
    parent = load_parent(path, setting, meta)
    return parent, setting, meta


def collection(cfg, out, rows, scales, examples, parent):
    data = defaultdict(list)
    cache = OrderedDict()
    for start in range(0, len(examples), 64):
        ex = examples[start : start + 64]
        batch = base_tokenizer_batch(
            rows, ex, scales, array_cache=cache, cache_limit=64
        )
        ids, features = encode_parent(parent, batch["x"])
        prediction = parent.decode_ids(ids)
        if start == 0:
            with torch.inference_mode():
                normal = parent.model(torch.as_tensor(batch["x"], dtype=torch.float32))
            for b in ("S", "O", "sum"):
                np.testing.assert_array_equal(normal[b].numpy(), prediction[b])
            for b in "SO":
                np.testing.assert_array_equal(normal["ids"][b].numpy(), ids[b])
        for k, v in batch.items():
            if k not in ("vx", "vs", "vo", "dvx", "dvs", "dvo"):
                data[k].append(v)
        for b in ("x", "s", "o"):
            data["v" + b].append(batch["v" + b][:, 0, :2])
            data["dv" + b].append(batch["dv" + b][:, 0, :2])
        for b in "SO":
            data["id_" + b].append(ids[b])
        for b in ("S", "O", "sum"):
            data["original_" + b].append(prediction[b])
        data["features"].append(features.astype("float32"))
        write(
            out / "progress.json",
            dict(
                stage="parent_training_panel",
                completed=start + len(ex),
                planned=len(examples),
                CPU_seconds=time.process_time(),
            ),
        )
    bank = {k: np.concatenate(v) for k, v in data.items()}
    np.savez_compressed(out / "training_bank.npz", examples=examples, **bank)
    return bank


def batch_view(bank, idx):
    result = {k: bank[k][idx] for k in ["x", "s", "o", "mask"]}
    for b in ("x", "s", "o"):
        for prefix in ["v", "dv"]:
            v = bank[prefix + b][idx]
            result[prefix + b] = np.tile(v[:, None, :], (1, 15, 20))
    return result


def quiet_state(bank, cfg):
    q = json.loads((ROOT / cfg["source"]["quiet_fit"]).read_text())["variants"][
        "training_tail"
    ]
    energy = np.square(bank["x"].astype(float)).mean((1, 2))
    error = np.square(np.asarray(bank["original_sum"], float) - bank["x"]).mean((1, 2))
    quiet = (np.sqrt(energy) <= q["threshold_input_sd"]) & (energy < error)
    return quiet, q


def color_gain(bank, idx, weights):
    mask = bank["mask"][idx].reshape(len(idx), 300, 2)
    target = bank["o"][idx].reshape(len(idx), 300, 2)
    prediction = bank["original_O"][idx].reshape(len(idx), 300, 2)
    normalized = weights[idx, None] / mask.sum(1).clip(1)
    numerator = np.sum(
        normalized * np.where(mask, target * target, 0).sum(1) / bank["vo"][idx], axis=0
    )
    denominator = np.sum(
        normalized
        * np.where(mask, prediction * prediction, 0).sum(1)
        / bank["vo"][idx],
        axis=0,
    )
    return np.clip(np.sqrt(numerator / denominator.clip(1e-12)), 0.9, 1.12)


def constrained_gain(bank, idx, examples, rows, quiet, cfg):
    """Historical v2/v3 gain-only approximation, excluding residual and quiet error."""
    weights = training_weights(examples[idx], rows)
    local_weights = np.zeros(len(examples))
    local_weights[idx] = weights
    desired = color_gain(bank, idx, local_weights)
    mask = bank['mask'][idx].reshape(len(idx), 300, 2)
    original = {b: bank['original_'+b][idx].reshape(len(idx),300,2).astype(float) for b in 'SO'}
    target = {b:bank[b.lower()][idx].reshape(len(idx),300,2) for b in 'SO'}
    datasets = np.array([rows[i]['dataset'] for i,_,_ in examples[idx]])
    groups = [np.ones(len(idx),bool)]+[datasets==d for d in sorted(set(datasets))]
    chosen=[]; review=[]
    grid=np.linspace(.9,1.12,45)
    for c in range(2):
        bounds=[]
        ordinary=~quiet[idx]
        baseO=original['O'][:,:,c]*ordinary[:,None]
        for b in 'SO':
            error=original[b][:,:,c]-target[b][:,:,c]
            valid=mask[:,:,c]&ordinary[:,None]
            denominator=bank['v'+b.lower()][idx,c]*mask[:,:,c].sum(1).clip(1)
            A=np.where(valid,baseO**2,0).sum(1)/denominator
            B=2*np.where(valid,error*baseO*(1 if b=='O' else -1),0).sum(1)/denominator
            C=np.where(valid,error**2,0).sum(1)/denominator
            for j,group in enumerate(groups):
                w=weights*group
                a,bb,cc=[np.sum(v*w)/w.sum() for v in (A,B,C)]
                ratio=cfg['selection']['frozen_numeric_components_ratio_max' if j==0 else 'frozen_dataset_components_ratio_max']
                bounds.append(a*(grid-1)**2+bb*(grid-1)+cc<=ratio*cc+1e-12)
        feasible=np.all(bounds,axis=0)
        if not feasible.any():raise ValueError('gain 1 must retain original components')
        value=grid[np.flatnonzero(feasible)[np.argmin(abs(grid[feasible]-desired[c]))]]
        chosen.append(float(value));review.append(dict(desired=float(desired[c]),selected=float(value),feasible_grid=grid[feasible]))
    return np.array(chosen),review


def constrained_emitted_gain(bank,idx,examples,rows,scales,quiet,cfg,leaf,fitted,q,parent,out,fold):
    """Fit the same gain grid against the complete emitted decoder and baseline."""
    weights=training_weights(examples[idx],rows);global_weights=np.zeros(len(examples));global_weights[idx]=weights
    desired=color_gain(bank,idx,global_weights)
    source={b:bank['original_'+b][idx] for b in ['S','O','sum']}
    meta=dict(color_gain=[1.,1.],residual_strength=.25,residual_lowpass_hz=cfg['model']['residual_lowpass_hz'],quiet_level_S=q['level_S'])
    at_one=synthesize_parent(source,leaf[idx],fitted,meta,quiet[idx])
    baseline={b:v.copy() for b,v in source.items()}
    for b in 'SO':baseline[b][quiet[idx]]=np.tile(q['level_S' if b=='S' else 'level_O'],20)[None,None]
    baseline['sum'][quiet[idx]]=0
    targets={b:bank[t][idx].reshape(len(idx),300,2) for b,t in [('sum','x'),('S','s'),('O','o')]}
    variances=np.stack([bank['v'+t][idx] for t in ['x','s','o']],axis=1)
    polynomials=emitted_error_polynomials(baseline,at_one,targets,bank['mask'][idx],variances,quiet[idx])
    aggregate=base_training_aggregate({'nmse':polynomials},examples[idx].tolist(),rows)['metrics']['nmse']
    groups=[('macro',aggregate['macro'],cfg['selection']['frozen_numeric_components_ratio_max'])]
    groups += [(d,v,cfg['selection']['frozen_dataset_components_ratio_max']) for d,v in aggregate['by_dataset'].items()]
    grid=np.linspace(.9,1.12,45);gain=[];review=[]
    for color in range(2):
        feasible=np.ones(len(grid),bool);constraints=[]
        for name,value,limit in groups:
            coeff=np.asarray(value)[:,color];scores=coeff[:,0,None]*(grid-1)**2+coeff[:,1,None]*(grid-1)+coeff[:,2,None]
            passed=(scores<=limit*coeff[:,3,None]+1e-12).all(0)&np.isfinite(scores).all(0)
            feasible &= passed;constraints.append(dict(group=name,limit=limit,quadratic_and_parent_NMSE=coeff,all_component_grid_pass=passed))
        selected=None if not feasible.any() else float(grid[np.flatnonzero(feasible)[np.argmin(abs(grid[feasible]-desired[color]))]])
        gain.append(selected);review.append(dict(desired=float(desired[color]),selected=selected,feasible_grid=grid[feasible],constraints=constraints))
    # Independent emitted integer decoding on fixed training-only grid checks.
    lookup={int(v):j for j,v in enumerate(idx)};selected=[]
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples[idx]}):
        selected.extend([int(j) for j in idx if rows[examples[j][0]]['dataset']==dataset][:32])
    selected=sorted(set(selected));blocks=sorted({j//64 for j in selected});largest=0.;checks=[]
    for point in [.9,1.,1.12]+([] if None in gain else [gain]):
        pair=np.repeat(point,2) if np.isscalar(point) else np.asarray(point)
        model=HbParentLeaf(parent,dict(meta,color_gain=pair.tolist(),branch_codebook_sizes={'S':129,'O':len(fitted['parent'])+1}),fitted)
        decoded={b:[] for b in ['S','O','sum']};take=[]
        for block in blocks:
            all_rows=np.arange(block*64,min((block+1)*64,len(examples)))
            ids={'S':bank['id_S'][all_rows].copy(),'O':leaf[all_rows].copy()}
            ids['S'][quiet[all_rows]]=128;ids['O'][quiet[all_rows]]=len(fitted['parent'])
            current=model.decode_ids(ids)
            keep=[j for j,v in enumerate(all_rows) if int(v) in selected]
            for b in decoded:decoded[b].append(current[b][keep])
            take.extend(all_rows[keep])
        pred={b:np.concatenate(v) for b,v in decoded.items()};take=np.asarray(take)
        actual=joint_reconstruction_metrics(pred,batch_view(bank,take),2,input_scales(rows,examples[take],scales))['metrics']['nmse']
        coefficients=polynomials[[lookup[int(j)] for j in take]]
        expected=coefficients[...,0]*(pair-1)[None,None,:]**2+coefficients[...,1]*(pair-1)[None,None,:]+coefficients[...,2]
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        difference=float(abs(actual-expected).max());largest=max(largest,difference)
        checks.append(dict(gain=pair,training_windows=len(take),max_abs_NMSE_difference=difference))
    write(out/f'emitted_gain_fit_{fold}.json',dict(status='feasible' if None not in gain else 'no_feasible_gain',
          emitted_alpha=.25,quiet_included=True,complete_hierarchical_denominator=True,fit_contexts=len(idx),
          original_grid=grid,color_reviews=review,independent_grid_checks=checks,max_abs_grid_check_difference=largest,
          held_scores_used_for_gain=False,legacy_gain=constrained_gain(bank,idx,examples,rows,quiet,cfg)[0]))
    return (None if None in gain else np.asarray(gain)),review


def attributes_for_split(bank):
    wave = bank["o"].reshape(-1, 20, 2)
    centered = wave - wave.mean(1, keepdims=True)
    t = (np.arange(20) - 9.5) / 10
    slope = np.einsum("ntc,t->nc", wave, t / (t @ t))
    return np.concatenate([centered.reshape(-1, 40), slope], axis=1)


def pattern_fold(
    train_ids,
    test_ids,
    bg,
    attribute,
    fields,
    valid,
    train_windows,
    test_windows,
    subjects,
):
    """Use the project's original background and background-plus-ID means."""
    k = max(int(train_ids.max()), int(test_ids.max())) + 1
    audit = SparsePatternAudit(
        k, int(bg.max()) + 1, ["all"], attribute.shape[-1], fields
    )
    fit_mask = valid[train_windows].ravel()
    train_flat = train_ids[train_windows].ravel()[fit_mask]
    train_bg = bg[train_windows].ravel()[fit_mask]
    train_values = attribute[train_windows].reshape(-1, attribute.shape[-1])[fit_mask]
    audit.fit(train_flat, train_bg, train_values)
    scores = np.full((len(test_windows), 2, len(fields)), np.nan)
    support = np.zeros(len(test_windows))
    count = np.bincount(train_flat, minlength=k)
    subcounts = np.zeros(k, int)
    full_train_subject = np.repeat(subjects[train_windows], 15)[fit_mask]
    for subject in np.unique(full_train_subject):
        subcounts[np.unique(train_flat[full_train_subject == subject])] += 1
    good = (count >= 8) & (subcounts >= 2)
    test_valid = valid[test_windows]
    flat = test_ids[test_windows].ravel()[test_valid.ravel()]
    background = bg[test_windows].ravel()[test_valid.ravel()]
    truth = attribute[test_windows].reshape(-1, attribute.shape[-1])[test_valid.ravel()]
    back, _, joint = audit.read(flat, background)
    errors = np.stack(
        [feature_means((p - truth) ** 2, fields) for p in (back, joint)], axis=1
    )
    restored = np.zeros((len(test_windows), 15, 2, len(fields)))
    restored[test_valid] = errors
    scores = restored.sum(1) / test_valid.sum(1)[:, None, None].clip(1)
    scores[~test_valid.any(1)] = np.nan
    mass = np.zeros(test_valid.shape)
    mass[test_valid] = good[flat]
    support = mass.sum(1) / test_valid.sum(1).clip(1)
    return scores, support


def metric_summary(pred, bank, idx, examples, rows, scales):
    batch = batch_view(bank, idx)
    a = joint_reconstruction_metrics(
        pred, batch, 2, input_scales(rows, examples[idx], scales)
    )
    return joint_reconstruction_summary(
        a["metrics"], a["support"], examples[idx].tolist(), rows
    ), a


def numeric_pass(candidate, base, cfg):
    s = cfg["selection"]
    m = candidate["metrics"]
    b = base["metrics"]
    gain = np.array(m["rms_gain"]["macro"])
    checks = dict(
        component_NMSE=bool(
            np.all(
                np.array(m["nmse"]["macro"])
                <= s["frozen_numeric_components_ratio_max"]
                * np.array(b["nmse"]["macro"])
            )
        ),
        dataset_NMSE=all(
            np.all(
                np.array(m["nmse"]["by_dataset"][d])
                <= s["frozen_dataset_components_ratio_max"]
                * np.array(b["nmse"]["by_dataset"][d])
            )
            for d in m["nmse"]["by_dataset"]
        ),
        gain=bool(
            np.all(
                (gain >= s["rms_gain_interval"][0])
                & (gain <= s["rms_gain_interval"][1])
            )
        ),
    )
    return checks


def pattern_summary(error, support, fields, examples, rows):
    h = base_training_aggregate({"nmse": error}, examples, rows)["metrics"]["nmse"]
    m = np.asarray(h["macro"])
    gain = 1 - m[1] / m[0].clip(1e-20)
    subject = np.array(
        [
            1 - np.array(v["value"])[1] / np.array(v["value"])[0].clip(1e-20)
            for v in h["by_subject"]
        ]
    )
    return dict(
        gain=dict(zip(fields, gain)),
        positive_subject_fraction=dict(zip(fields, np.mean(subject > 0, axis=0))),
        training_support_mass=float(np.nanmean(support)),
        hierarchy=h,
    )


def pattern_pass(pattern, cfg):
    s = cfg["selection"]
    g = pattern["gain"]
    fraction = pattern["positive_subject_fraction"]
    primary = ["HbO_slope", "HbR_slope", "HbO_centered", "HbR_centered"]
    return dict(
        slope=all(
            g[c + "_slope"] >= s["O_single_ID_slope_gain_min_each_color"]
            for c in ["HbO", "HbR"]
        ),
        shape=all(
            g[c + "_centered"] > s["O_single_ID_centered_gain_min_exclusive"]
            for c in ["HbO", "HbR"]
        ),
        subject=all(
            fraction[k] >= s["primary_positive_subject_fraction_min"] for k in primary
        ),
        support=pattern["training_support_mass"]
        >= s["validation_mass_on_8patch_2subject_leaf_min"],
    )


def crossfit(cfg, out, rows, scales, examples, bank, parent, base_meta):
    n = len(examples)
    evaluation_n=cfg["training"].get("expanded_fit",{}).get("evaluation_contexts",n)
    evaluation_idx=np.arange(evaluation_n)
    valid = bank["mask"].all(-1)
    quiet, q = quiet_state(bank, cfg)
    subjects = np.array(
        [rows[i]["dataset"] + "/" + rows[i]["subject"] for i, _, _ in examples]
    )
    datasets = np.array([rows[i]["dataset"] for i, _, _ in examples])
    folds = np.empty(n, int)
    for d in sorted(set(datasets)):
        people = sorted(
            set(subjects[datasets == d]),
            key=lambda x: digest_text([cfg["training"]["fold_seed"], x]),
        )
        for j, p in enumerate(people):
            folds[subjects == p] = j % cfg["training"]["folds"]
    if cfg['training'].get('expanded_fit'):
        previous=json.loads((out.parent.parent/cfg['training']['expanded_fit']['original_training_folds']).read_text())
        np.testing.assert_array_equal(examples[:evaluation_n],previous['examples'])
        np.testing.assert_array_equal(folds[:evaluation_n],previous['folds'])
    write(
        out / "training_folds.json",
        dict(examples=examples, folds=folds, subjects=subjects),
    )
    attrs, fields = pattern_attributes(bank["o"].reshape(-1, 40), "hb")
    attrs = attrs.reshape(n, 15, -1)
    superv = attributes_for_split(bank)
    residual = (bank["o"] - bank["original_O"]).reshape(-1, 40)
    f = bank["features"].reshape(-1, bank["features"].shape[-1])
    old = bank["id_O"].reshape(-1)
    sites = sorted(
        {(rows[i]["dataset"], rows[i]["positions"][p]) for i, _, p in examples}
    )
    site = {s: j for j, s in enumerate(sites)}
    locations = np.array(
        [site[rows[i]["dataset"], rows[i]["positions"][p]] for i, _, p in examples]
    )
    bg = locations[:, None] * 4 + np.searchsorted(
        [0.5, 1.0, 2.0], np.sqrt(np.mean(bank["x"] ** 2, axis=-1)), side="right"
    )
    weights = training_weights(examples, rows)
    original = {b: bank["original_" + b].copy() for b in ["S", "O", "sum"]}
    # Apply only the already trained quiet rule to the original parent baseline.
    for b in ["S", "O"]:
        original[b][quiet] = np.tile(q["level_S" if b == "S" else "level_O"], 20)[
            None, None
        ]
    original["sum"][quiet] = 0
    baseline, _ = metric_summary({b:v[:evaluation_n] for b,v in original.items()}, bank, evaluation_idx, examples, rows, scales)
    if cfg['training'].get('expanded_fit'):
        reference=json.loads((out.parent/'v2_ridge/training_crossfit.json').read_text())['baseline']
        np.testing.assert_array_equal(baseline['metrics']['nmse']['macro'],reference['metrics']['nmse']['macro'])
    records = []
    for maximum in cfg["model"]["max_leaf_options"]:
        predicted = {
            alpha: {
                b: np.empty_like(
                    original[b][:evaluation_n], dtype=float if b != "sum" else original[b].dtype
                )
                for b in original
            }
            for alpha in cfg["model"]["residual_strength_options"]
        }
        errors = np.full((evaluation_n, 2, len(fields)), np.nan)
        support = np.zeros(evaluation_n)
        original_readout_errors=np.full_like(errors,np.nan)
        original_readout_support=np.zeros(evaluation_n)
        fold_info = []
        for fold in range(cfg["training"]["folds"]):
            train = np.flatnonzero(folds != fold)
            held = np.flatnonzero(folds[:evaluation_n] == fold)
            train_mask = np.zeros(n, bool)
            train_mask[train] = True
            patch = train_mask[:, None] & valid & ~quiet[:, None]
            selected = np.flatnonzero(patch.ravel())
            if cfg['training'].get('expanded_fit'):
                if set(subjects[train])&set(subjects[held]):raise ValueError('held subjects in expanded fitting')
                people=sorted(set(subjects));subject_index={s:j for j,s in enumerate(people)}
                person=np.array([subject_index[s] for s in subjects])
                parent_person=np.zeros((2,128,len(people)),np.int64)
                for j,mask in enumerate([patch & (np.arange(n)<evaluation_n)[:,None],patch]):
                    a,b=np.nonzero(mask)
                    np.add.at(parent_person[j],(bank['id_O'][a,b],person[a]),1)
                np.savez_compressed(out/f'fit_support_fold{fold}.npz',original_and_expanded_parent_subject_complete_patches=parent_person,subjects=people)
            setting = dict(cfg["model"], max_leaves=maximum)
            if 'frozen_route_folds' in cfg['source']:
                path=ROOT/cfg['source']['frozen_route_folds']/f'leaves{maximum}_fold{fold}.pt'
                if digest(path)!=cfg['source']['frozen_route_sha256'][str(fold)]:raise ValueError('frozen routing weights changed')
                fitted=torch.load(path,map_location='cpu',weights_only=False)['fitted']
            else:
                fitted = fit_children(old[selected],f[selected],superv[selected],residual[selected],
                    np.repeat(subjects, 15)[selected],setting,cfg["training"]["fold_seed"] + fold)
            leaf = assign_leaf(bank["id_O"], bank["features"], fitted)
            np.testing.assert_array_equal(fitted["parent"][leaf], bank["id_O"])
            emitted = leaf.copy()
            emitted[quiet] = len(fitted["parent"])
            scores, mass = pattern_fold(
                emitted, emitted, bg, attrs, fields, valid, train, held, subjects
            )
            errors[held] = scores
            support[held] = mass
            if cfg['model'].get('emitted_gain_projection'):
                small_scores,small_mass=pattern_fold(emitted,emitted,bg,attrs,fields,valid,train[train<evaluation_n],held,subjects)
                original_readout_errors[held]=small_scores;original_readout_support[held]=small_mass
            gain_review = None
            if cfg['model'].get('emitted_gain_projection'):
                gain,gain_review=constrained_emitted_gain(bank,train,examples,rows,scales,quiet,cfg,leaf,fitted,q,parent,out,fold)
                if gain is None:
                    write(out/'training_crossfit.json',dict(status='completed_training_gain_infeasible',baseline=baseline,candidates=[],infeasible_fold=fold,validation_arrays_read=0))
                    return None,bank,quiet,q
            elif cfg['model'].get('gain_projection'):
                gain, gain_review = constrained_gain(bank, train, examples, rows, quiet, cfg)
            else:
                gain = color_gain(bank, train, weights)
            for alpha in predicted:
                meta = dict(
                    base_meta,
                    color_gain=gain.tolist(),
                    residual_strength=alpha,
                    residual_lowpass_hz=cfg["model"]["residual_lowpass_hz"],
                    quiet_level_S=q["level_S"],
                )
                current = synthesize_parent(
                    {b: bank["original_" + b][held] for b in original},
                    leaf[held],
                    fitted,
                    meta,
                    quiet[held],
                )
                ordinary = ~quiet[held]
                np.testing.assert_array_equal(
                    current["sum"][ordinary], bank["original_sum"][held][ordinary]
                )
                for b in original:
                    predicted[alpha][b][held] = current[b]
            fold_info.append(
                dict(
                    fold=fold,
                    fitting_contexts=len(train),
                    original_fitting_contexts=int(np.sum(train<evaluation_n)),
                    added_held_subject_contexts_excluded=int(np.sum(folds[evaluation_n:]==fold)),
                    fixed_evaluated_contexts=len(held),
                    ordinary_leaf_count=len(fitted["parent"]),
                    split_parents=sum(p["split"] for p in fitted["parents"]),
                    color_gain=gain,
                    gain_projection=gain_review,
                    min_split_patch_support=min(
                        (
                            v["complete_patches"]
                            for v in fitted["support"]
                            if fitted["parents"][v["parent"]]["split"]
                        ),
                        default=0,
                    ),
                )
            )
            torch.save(
                dict(fitted=fitted, gain=gain), out / f"leaves{maximum}_fold{fold}.pt"
            )
            write(
                out / "progress.json",
                dict(
                    stage="leaf_train_crossfit",
                    maximum_children=maximum,
                    completed_fold=fold + 1,
                    CPU_seconds=time.process_time(),
                    validation_arrays_read=0,
                ),
            )
        ps = pattern_summary(errors, support, fields, examples[:evaluation_n], rows)
        pg = pattern_pass(ps, cfg)
        small_ps=pattern_summary(original_readout_errors,original_readout_support,fields,examples[:evaluation_n],rows) if cfg['model'].get('emitted_gain_projection') else None
        small_pg=pattern_pass(small_ps,cfg) if small_ps else {}
        for alpha, est in predicted.items():
            summary, audit = metric_summary(
                est, bank, evaluation_idx, examples, rows, scales
            )
            ng = numeric_pass(summary, baseline, cfg)
            key = f"leaves{maximum}_shape{alpha:g}"
            np.savez_compressed(
                out / (key + "_metrics.npz"),
                examples=examples[:evaluation_n],
                nmse=audit["metrics"]["nmse"],
            )
            records.append(
                dict(
                    candidate=key,
                    max_leaves=maximum,
                    residual_strength=alpha,
                    pattern=ps,
                    pattern_gates=pg,
                    pattern_original_256_support=small_ps,
                    pattern_original_256_support_gates=small_pg,
                    numeric_gates=ng,
                    eligible=all(pg.values()) and all(ng.values()) and all(small_pg.values()),
                    summary=summary,
                    fold_models=fold_info,
                )
            )
        write(
            out / "training_crossfit.json",
            dict(
                status="in_progress",
                baseline=baseline,
                candidates=records,
                validation_arrays_read=0,
            ),
        )
    eligible = [r for r in records if r["eligible"] and r["max_leaves"] > 1]
    selected = (
        min(
            eligible,
            key=lambda r: (
                r["summary"]["selection_score"],
                r["max_leaves"],
                r["residual_strength"],
            ),
        )
        if eligible
        else None
    )
    write(
        out / "training_crossfit.json",
        dict(
            status="completed",
            baseline=baseline,
            candidates=records,
            selected=None if selected is None else selected["candidate"],
            validation_arrays_read=0,
        ),
    )
    return selected, bank, quiet, q


def digest_text(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def oracle_capacity(cfg,out,rows,examples,bank):
    """Train-only hidden-target quantile routing: a diagnostic, never a candidate."""
    source=ROOT/cfg['source']['panel']
    reference=out.parent/'v2_ridge'
    actual=json.loads((reference/'training_crossfit.json').read_text())
    if any(c['eligible'] for c in actual['candidates']):
        raise ValueError('admitted actual X arm takes priority over oracle diagnostic')
    plan=json.loads((reference/'training_folds.json').read_text())
    np.testing.assert_array_equal(examples,plan['examples'])
    folds=np.asarray(plan['folds']);subjects=np.asarray(plan['subjects'])
    n=len(examples);valid=bank['mask'].all(-1);quiet,_=quiet_state(bank,cfg)
    attrs,fields=pattern_attributes(bank['o'].reshape(-1,40),'hb');attrs=attrs.reshape(n,15,-1)
    true_slope=attributes_for_split(bank)[:,40].reshape(n,15)
    sites=sorted({(rows[i]['dataset'],rows[i]['positions'][p]) for i,_,p in examples})
    site={s:j for j,s in enumerate(sites)}
    locations=np.array([site[rows[i]['dataset'],rows[i]['positions'][p]] for i,_,p in examples])
    bg=locations[:,None]*4+np.searchsorted([.5,1.,2.],np.sqrt(np.mean(bank['x']**2,axis=-1)),side='right')
    records=[]
    for maximum in (2,4):
        error=np.full((n,2,len(fields)),np.nan);support=np.zeros(n);folds_out=[]
        actual_error=np.full_like(error,np.nan);actual_support=np.zeros(n)
        for fold in range(cfg['training']['folds']):
            train=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
            patch=np.zeros((n,15),bool);patch[train]=valid[train]&~quiet[train,None]
            emitted=np.zeros((n,15),np.int32);matched=np.zeros_like(emitted);lookup=[];parents=[]
            source_arm=out.parent/('v1' if maximum==2 else 'v2_ridge')
            fitted=torch.load(source_arm/f'leaves{maximum}_fold{fold}.pt',map_location='cpu',weights_only=False)['fitted']
            actual_leaf=assign_leaf(bank['id_O'],bank['features'],fitted)
            for code in range(128):
                fit=patch&(bank['id_O']==code);slopes=true_slope[fit];people=np.broadcast_to(subjects[:,None],patch.shape)[fit]
                thresholds=np.array([])
                wanted=fitted['parents'][code]['children']
                for count in ([wanted] if wanted>1 else []):
                    if len(slopes)<count*cfg['model']['minimum_leaf_complete_training_patches']:continue
                    candidate=np.unique(np.quantile(slopes,np.arange(1,count)/count))
                    labels=np.searchsorted(candidate,slopes,side='right')
                    if len(candidate)!=count-1:continue
                    if all(np.sum(labels==j)>=cfg['model']['minimum_leaf_complete_training_patches'] and
                           len(set(people[labels==j]))>=cfg['model']['minimum_leaf_training_subjects'] for j in range(count)):
                        thresholds=candidate;break
                offset=len(lookup);count=len(thresholds)+1
                use=bank['id_O']==code
                emitted[use]=offset+np.searchsorted(thresholds,true_slope[use],side='right')
                if count>1:
                    actual_codes=sorted(fitted['node_to_ID'][code].values())
                    matched[use]=offset+np.searchsorted(actual_codes,actual_leaf[use])
                else:matched[use]=offset
                lookup.extend([code]*count);parents.append(dict(parent=code,thresholds=thresholds,children=count,actual_original_children=wanted,both_collapsed_due_quantile_support=(wanted>count)))
            emitted[quiet]=len(lookup);matched[quiet]=len(lookup)
            ae,ass=pattern_fold(matched,matched,bg,attrs,fields,valid,train,held,subjects)
            actual_error[held]=ae;actual_support[held]=ass
            e,s=pattern_fold(emitted,emitted,bg,attrs,fields,valid,train,held,subjects)
            error[held]=e;support[held]=s
            np.savez_compressed(out/f'privileged_leaves{maximum}_fold{fold}.npz',IDs=emitted,matched_X_IDs=matched,leaf_to_parent=lookup)
            folds_out.append(dict(fold=fold,ordinary_leaf_count=len(lookup),parents=parents))
        ps=pattern_summary(error,support,fields,examples,rows)
        records.append(dict(maximum_children=maximum,pattern=ps,matched_actual_X_pattern=pattern_summary(actual_error,actual_support,fields,examples,rows),fold_models=folds_out))
    write(out/'summary.json',dict(status='completed_diagnostic_only',routing='hidden_true_O_HbO_local_slope_at_both_train_and_held_no_validation',
          candidate_eligible=False,screen_read=False,confirmation_read=False,test_read=False,actual_X_reference=str(reference),
          same_background_and_folds=True,candidates=records,CPU_seconds=time.process_time()))


def evaluate_frozen(cfg,out,role):
    """Score a frozen admitted leaf model, then replay saved integers after reload."""
    from types import SimpleNamespace
    from collections import defaultdict
    from evaluate_joint_tokenizer import export_examples
    from review_joint_tokenizer import score
    root=out.parent.parent
    panel=ROOT/cfg['source']['panel']
    rows,scales,examples,manifest=load_panel(SimpleNamespace(panel=panel,modality='hb',role=role))
    # The source module checks current manifest/split before signal arrays.
    forbidden=json.loads(panel.with_name('panel.json').read_text())['confirmation_subjects']
    if role not in ('screen','development') or any(rows[i]['subject']==forbidden[rows[i]['dataset']] for i,_,_ in examples):
        raise ValueError('original non-confirmation development panel required')
    trained=json.loads((out/'summary.json').read_text())
    if trained['status']!='train_admitted':raise ValueError('train admission required before validation waveform access')
    checkpoint=out/'model.pt';frozen_hash=digest(checkpoint);folder=out/role;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():raise FileExistsError('retained frozen evaluation')
    forward=load_model(checkpoint)
    if forward.metadata['cfg_identity']!=manifest['cfg_identity'] or forward.metadata['scales_identity']!=manifest['scales_identity']:
        raise ValueError('frozen leaf G0 coordinate mismatch')
    cache=OrderedDict();results=defaultdict(list);support=defaultdict(list);stored=defaultdict(list);parent_error=0.;quiet_windows=0
    for start in range(0,len(examples),64):
        take=examples[start:start+64];batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=32)
        ids=forward.encode(batch['x']);pred=forward.decode_ids(ids)
        ordinary=(ids['S']!=128).any(1)
        if ordinary.any():
            safe_S=np.where(ordinary[:,None],ids['S'],0)
            safe_O=np.where(ordinary[:,None],ids['O'],0)
            original_ids=dict(S=safe_S,O=forward.fitted['parent'][safe_O])
            expected=forward.parent.decode_ids(original_ids)
            np.testing.assert_array_equal(pred['sum'][ordinary],expected['sum'][ordinary])
            parent_error=max(parent_error,float(abs(pred['sum'][ordinary]-expected['sum'][ordinary]).max()))
        quiet_windows+=int((~ordinary).sum())
        audited=joint_reconstruction_metrics(pred,batch,2,input_scales(rows,take,scales))
        for k,v in audited['metrics'].items():results[k].append(v)
        for k,v in audited['support'].items():support[k].append(v)
        for k,v in ids.items():stored[k].append(v)
        write(folder/'progress.json',dict(stage='frozen_forward',completed=start+len(take),planned=len(examples)))
    results={k:np.concatenate(v) for k,v in results.items()};support={k:np.concatenate(v) for k,v in support.items()}
    ids={k:np.concatenate(v) for k,v in stored.items()};np.savez_compressed(folder/'integer_ids.npz',examples=examples,**ids)
    with np.load(folder/'integer_ids.npz') as disk:
        np.testing.assert_array_equal(examples,disk['examples']);ids={k:disk[k].copy() for k in 'SO'}
    reloaded=load_model(checkpoint);replayed=defaultdict(list);counts=defaultdict(list)
    for start in range(0,len(examples),64):
        take=examples[start:start+64];batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=32)
        pred=reloaded.decode_ids({k:v[start:start+len(take)] for k,v in ids.items()})
        audited=joint_reconstruction_metrics(pred,batch,2,input_scales(rows,take,scales))
        for k,v in audited['metrics'].items():replayed[k].append(v)
        for k,v in audited['support'].items():counts[k].append(v)
    for k,v in replayed.items():np.testing.assert_array_equal(np.concatenate(v),results[k])
    for k,v in counts.items():np.testing.assert_array_equal(np.concatenate(v),support[k])
    if digest(checkpoint)!=frozen_hash:raise ValueError('model changed during score')
    np.savez_compressed(folder/'metrics.npz',examples=examples,**results,**{'support_'+k:v for k,v in support.items()})
    summary=joint_reconstruction_summary(results,support,examples.tolist(),rows)
    summary.update(status='completed',modality='hb',role=role,checkpoint=str(checkpoint),checkpoint_sha256=frozen_hash,
                   integer_replay=dict(status='passed',independent_reload=True,metric_max_abs_difference=0.),quiet_windows=quiet_windows,
                   original_parent_sum_max_abs_difference_ordinary=parent_error,interface_eligible=True,context_seconds=30,patch_seconds=2)
    write(folder/'summary.json',summary)
    ref=root/('baseline_audit' if role=='screen' else 'development_audit')/'hb'
    def reference(kind):
        p=ref/kind;special=p/'summary_v2.json'
        return json.loads((special if special.exists() else p/'summary.json').read_text())
    contract=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml').read_text())
    verdict=score(summary,reference('vq128'),reference('continuous'),contract);write(folder/'reconstruction_decision.json',verdict)
    args=SimpleNamespace(reference_metrics=ref/'vq128/metrics.npz',continuous=False,kind='dictionary',checkpoint=checkpoint,
                         modality='hb',display_output=None,output=folder,role=role)
    export_examples(args,reloaded,results,rows,scales,examples,ids)
    from render_joint_tokenizer_waveforms import render_export
    render_export(folder/'waveform_examples.npz',folder/'waveform_examples.json',folder/'waveform_figures')
    if role=='screen':
        from audit_joint_tokenizer_patterns import collect
        with np.load(panel) as f:pane={'train':f['hb_pattern_train'].copy(),'validation':f['hb_screen'].copy()}
        collect(reloaded,rows,scales,pane,'hb',out/'screen_pattern',max_atlas_pages=3)
    write(folder/'rate_review.json',dict(branch_vocabulary=reloaded.metadata['branch_codebook_sizes'],
        fixed_bits_per_second={b:float(np.ceil(np.log2(k)))/2 for b,k in reloaded.metadata['branch_codebook_sizes'].items()},
        integer_count=sum(v.size for v in ids.values()),parent_lookup_is_fixed=True,decoder_continuous_window_side_information=0))
    print(role,summary['selection_score'],verdict['failed'])
    return verdict


def render_training_patterns(out,folder,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),constrained_layout=True)
    arms=['v1','v2_axis','v2_ridge'];labels=['v1\naxis','v2\naxis','v2 ridge\nfit 256/person']
    if cfg['training'].get('expanded_fit'):
        arms.append(out.name);labels.append('frozen v3 routes\nemitted gain' if cfg['model'].get('emitted_gain_projection') else 'same v2 ridge\nfit 1024/person')
    for c,color in enumerate(['HbO','HbR']):
        vals=[]
        for arm in arms:
            report=json.loads((out.parent/arm/'training_crossfit.json').read_text());x=next(v for v in report['candidates'] if v['candidate']=='leaves4_shape0.25');vals.append([x['pattern']['gain'][color+'_slope'],x['pattern']['gain'][color+'_centered']])
        for j,field in enumerate(['slope','centered']):axes[c].bar(np.arange(len(arms))+j*.35,np.array(vals)[:,j],.35,label=field)
        axes[c].axhline(.10,color='red',ls='--',label='slope requirement');axes[c].set_xticks(np.arange(len(arms))+.175,labels);axes[c].set_ylabel('1 - held background+ID MSE / background MSE');axes[c].set_title(color);axes[c].legend(fontsize=8)
    path=folder/'training_patterns.png'
    if path.exists():path=folder/'training_patterns_v2.png'
    if path.exists():raise FileExistsError('retained versioned pattern figure')
    fig.savefig(path,dpi=180);plt.close(fig)
    source=ROOT/cfg['source']['expanded_fit_run'] if 'expanded_fit_run' in cfg['source'] else out
    audit=source/'scale_readout_audit.json'
    if audit.exists():
        values=json.loads(audit.read_text())['conditions'];labels=['route 256\nreadout 256','route 256\nreadout 1024','route 1024\nreadout 256','route 1024\nreadout 1024']
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for ax,color in zip(axes,['HbO','HbR']):
            for j,field in enumerate(['slope','centered']):ax.bar(np.arange(4)+j*.35,[v['gain'][color+'_'+field] for v in values.values()],.35,label=field)
            ax.set_xticks(np.arange(4)+.175,labels);ax.set_title(color);ax.set_ylabel('1 - held conditional MSE / background MSE');ax.axhline(.1,color='red',ls='--',label='slope requirement');ax.legend(fontsize=8)
        fig.suptitle('Frozen routing x train-only template sample size; same original 15360 held windows, 60 subjects')
        path=folder/'readout_size_control.png'
        if path.exists():raise FileExistsError('retained readout diagnostic figure')
        fig.savefig(path,dpi=180);plt.close(fig)


def review_training(cfg,out,rows,scales,examples,bank,parent,base_meta):
    """Retain rejected frozen folds, integer replay, and rule-selected failures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plan=json.loads((out/'training_folds.json').read_text());folds=np.array(plan['folds'])[:len(examples)]
    qmask,q=quiet_state(bank,cfg);folder=out/'training_review_v2';folder.mkdir(exist_ok=True)
    cross=json.loads((out/'training_crossfit.json').read_text())
    record=next(c for c in cross['candidates'] if c['candidate']=='leaves4_shape0.25')
    with np.load(out/'leaves4_shape0.25_metrics.npz') as f:actual=f['nmse'].copy()
    baseline=np.zeros_like(actual)
    for j,(b,t) in enumerate([('sum','x'),('S','s'),('O','o')]):
        v=bank['original_'+b].copy()
        if b=='sum':v[qmask]=0
        else:v[qmask]=np.tile(q['level_S' if b=='S' else 'level_O'],20)[None,None]
        a=((v-bank[t])**2).reshape(len(examples),300,2)
        mask=bank['mask'].reshape(len(examples),300,2)
        baseline[:,j]=np.where(mask,a,0).sum(1)/mask.sum(1).clip(1)/bank['v'+t]
    dataset=np.array([rows[i]['dataset'] for i,_,_ in examples]);chosen={}
    for d in sorted(set(dataset)):
        ids=np.flatnonzero(dataset==d)
        fixed=min(ids,key=lambda j:digest_text(examples[j].tolist()))
        for j,rule in [(fixed,'fixed_SHA_identity'),(ids[np.argmax(baseline[ids,1:].mean((1,2)))],'hardest_parent_SO'),(ids[np.argmax((actual[ids,1:]-baseline[ids,1:]).mean((1,2)))],'largest_new_SO_degradation')]:
            chosen.setdefault(int(j),[]).append(rule)
    # Actual quiet behavior is retained explicitly, irrespective of its errors.
    for j in np.flatnonzero(qmask)[:1]:chosen.setdefault(int(j),[]).append('first_actual_quiet')
    models={};meta={};states={}
    for fold in range(cfg['training']['folds']):
        saved=torch.load(out/f'leaves4_fold{fold}.pt',map_location='cpu',weights_only=False)
        fitted=saved['fitted'];metadata=dict(base_meta,branch_codebook_sizes={'S':129,'O':len(fitted['parent'])+1},color_gain=saved['gain'].tolist(),residual_strength=.25,residual_lowpass_hz=cfg['model']['residual_lowpass_hz'],quiet_threshold=q['threshold_input_sd'],quiet_level_S=q['level_S'],diagnostic_only=True)
        m=HbParentLeaf(parent,metadata,fitted);path=folder/f'rejected_fold{fold}.pt';m.save(path);models[fold]=load_model(path);meta[fold]=metadata;states[fold]=fitted
    selected=np.array(sorted(chosen));pred={k:[] for k in ['S','O','sum']};stored={b:[] for b in 'SO'};sumerr=0.;replayerr=0.;rmsum=0.;parent_batch_drift=0.
    for j in selected:
        m=models[folds[j]];ids=m.encode(bank['x'][j:j+1]);p=m.decode_ids(ids)
        for b in 'SO':stored[b].append(ids[b][0])
        for b in pred:pred[b].append(p[b][0])
    pred={k:np.array(v) for k,v in pred.items()};stored={b:np.array(v) for b,v in stored.items()}
    np.savez_compressed(folder/'saved_integer_ids.npz',examples=examples[selected],training_panel_index=selected,**stored)
    with np.load(folder/'saved_integer_ids.npz') as disk:
        for a,j in enumerate(disk['training_panel_index']):
            m=load_model(folder/f'rejected_fold{folds[j]}.pt');p=m.decode_ids({b:disk[b][a:a+1] for b in 'SO'})
            for b in pred:np.testing.assert_array_equal(p[b][0],pred[b][a])
            if not qmask[j]:
                parent_ids=dict(S=disk['S'][a:a+1],O=m.fitted['parent'][disk['O'][a:a+1]])
                np.testing.assert_array_equal(parent_ids['S'][0],bank['id_S'][j])
                np.testing.assert_array_equal(parent_ids['O'][0],bank['id_O'][j])
                expected=m.parent.decode_ids(parent_ids)
                np.testing.assert_array_equal(p['sum'],expected['sum'])
                parent_batch_drift=max(parent_batch_drift,float(abs(p['sum'][0]-bank['original_sum'][j]).max()))
            rmsum=max(rmsum,float(abs(p['S']+p['O']-p['sum']).max()))
    factors=input_scales(rows,examples[selected],scales);manifest=[];time_axis=np.arange(300)/10
    for a,j in enumerate(selected):
        i,start,pos=examples[j];r=rows[i];mask=bank['mask'][j].reshape(300,2)
        # Zoom is centered on the interior token boundary with greatest target local slope.
        wave=bank['o'][j].reshape(300,2);boundaries=np.arange(20,281,20)
        boundary=int(boundaries[np.argmax(np.linalg.norm(wave[boundaries+1]-wave[boundaries-1],axis=1))])
        lo,hi=boundary/10-1,boundary/10+1
        fig,axes=plt.subplots(3,4,figsize=(17,8),constrained_layout=True)
        for row,(branch,target) in enumerate([('S','s'),('O','o'),('sum','x')]):
            curves=[bank[target][j],bank['original_'+branch][j],pred[branch][a]]
            curves=[v.reshape(300,2)*factors[a] for v in curves]
            if qmask[j]:curves[1]=np.zeros_like(curves[1]) if branch=='sum' else np.tile(q['level_S' if branch=='S' else 'level_O'],(300,1))*factors[a]
            for col in range(4):
                c=col%2;ax=axes[row,col]
                for values,color,label,style in zip(curves,['black','#d8870c','#1766ba'],['Target','Original G1 + quiet','Atomic leaf'],['-','--','-']):ax.plot(time_axis,values[:,c],color=color,lw=1.2,label=label,ls=style)
                for t in np.arange(0,31,2):ax.axvline(t,color='.75',lw=.5)
                for t in np.flatnonzero(~mask[:,c]):ax.axvspan(t/10,(t+1)/10,color='.7',alpha=.4,lw=0)
                ax.set_xlim((0,30) if col<2 else (lo,hi));ax.set_xlabel('seconds');ax.set_ylabel(branch+' / subject_normalized_Hb');ax.set_title(('HbO' if c==0 else 'HbR')+(' / full 30 s' if col<2 else ' / 2 s boundary zoom'));ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=8)
        title=f"{r['dataset']} / {r['subject']} / {r['record']} / {r['positions'][pos]} / start={start}s / fold={folds[j]}\n"+', '.join(chosen[int(j)])+f"; quiet={bool(qmask[j])}; parent SO NMSE={baseline[j,1:].mean():.4f}; leaf={actual[j,1:].mean():.4f}"
        fig.suptitle(title,fontsize=10);name=f'{r["dataset"]}_{a}.png';fig.savefig(folder/name,dpi=160);plt.close(fig)
        manifest.append(dict(index=int(j),dataset=r['dataset'],subject=r['subject'],record=r['record'],position=r['positions'][pos],start_s=int(start),fold=int(folds[j]),rules=chosen[int(j)],quiet=bool(qmask[j]),figure=name,original_SO_NMSE=float(baseline[j,1:].mean()),leaf_SO_NMSE=float(actual[j,1:].mean())))
    # Fold 0 training template and real held members for the ten most occupied O leaves.
    fold=0;fitted=states[fold];leaf=assign_leaf(bank['id_O'],bank['features'],fitted)
    train=(folds!=fold)[:,None]&bank['mask'].all(-1)&~qmask[:,None];held=(folds==fold)[:,None]&bank['mask'].all(-1)&~qmask[:,None]
    counts=np.bincount(leaf[train],minlength=len(fitted['parent']));order=np.argsort(-counts);wave=bank['o'].reshape(-1,20,2)
    atlas=[];fig,axes=plt.subplots(10,2,figsize=(12,23),constrained_layout=True)
    for row,code in enumerate(order[:10]):
        use=np.flatnonzero(train.ravel()&(leaf.ravel()==code));held_use=np.flatnonzero(held.ravel()&(leaf.ravel()==code));people=set(np.asarray(plan['subjects'])[use//15])
        atlas.append(dict(ID=int(code),parent=int(fitted['parent'][code]),patches=len(use),subjects=len(people),held_patches=len(held_use),occupancy=float(len(use)/train.sum()),cumulative_coverage=float(counts[order[:row+1]].sum()/train.sum())))
        for c in range(2):
            values=wave[use,:,c];values=values-values.mean(1,keepdims=True);ax=axes[row,c];t=np.arange(20)/10
            loq,median,hiq=np.quantile(values,[.1,.5,.9],axis=0);ax.fill_between(t,loq,hiq,color='#4c8ec6',alpha=.25,label='train P10-P90');ax.plot(t,median,color='black',lw=2,label='train median')
            for h in held_use[:3]:v=wave[h,:,c];ax.plot(t,v-v.mean(),lw=.9,alpha=.8,label='held real member' if h==held_use[0] else None)
            ax.set_title(f"{'HbO' if c==0 else 'HbR'} leaf {code} -> parent {fitted['parent'][code]}; n={len(use)}, people={len(people)}, held={len(held_use)}",fontsize=9);ax.set_xlabel('local seconds');ax.set_ylabel('centered / G0 train input SD');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Rejected fold 0: single-ID local target shape, train vs held subjects; original X-amplitude background used in score',fontsize=11);fig.savefig(folder/'O_leaf_atlas.png',dpi=150);plt.close(fig)
    write(folder/'figure_manifest.json',manifest);write(folder/'atlas_metadata.json',dict(major=atlas,all_codes=[dict(ID=int(k),patches=int(v),parent=int(fitted['parent'][k])) for k,v in enumerate(counts)]))
    render_training_patterns(out,folder,cfg)
    if cfg['training'].get('expanded_fit'):
        old=json.loads((out.parent/'v2_ridge/training_crossfit.json').read_text())
        old_pattern=next(c for c in old['candidates'] if c['candidate']=='leaves4_shape0.25')['pattern']
        new_pattern=record['pattern'];field_names=list(new_pattern['gain'])
        def people_gain(pattern):
            return {(r['dataset'],r['subject']):1-np.array(r['value'])[1]/np.array(r['value'])[0].clip(1e-20) for r in pattern['hierarchy']['by_subject']}
        old_people=people_gain(old_pattern);new_people=people_gain(new_pattern)
        if old_people.keys()!=new_people.keys():raise ValueError('matched original people required')
        fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
        for ax,field in zip(axes.ravel(),['HbO_slope','HbO_centered','HbR_slope','HbR_centered']):
            j=field_names.index(field);all_values=[]
            for d,marker in zip(sorted(set(dataset)),['o','s','^','D']):
                keys=sorted(k for k in old_people if k[0]==d);x=np.array([old_people[k][j] for k in keys]);y=np.array([new_people[k][j] for k in keys]);ax.scatter(x,y,label=d,marker=marker,s=35,alpha=.8);all_values.extend(x);all_values.extend(y)
            lo=min(0,min(all_values))-.05;hi=max(all_values)+.05;ax.plot([lo,hi],[lo,hi],color='.5',ls=':');ax.axhline(0,color='.5',lw=.7);ax.axvline(0,color='.5',lw=.7);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
            ax.set_xlabel('fit 256/person: conditional ID error reduction');ax.set_ylabel('fit 1024/person: conditional ID error reduction');ax.set_title(field+f"; positive {round(old_pattern['positive_subject_fraction'][field]*60)}/60 -> {round(new_pattern['positive_subject_fraction'][field]*60)}/60");ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle('Each point is one identical held subject; all subjects retained; original 15360 evaluation windows')
        fig.savefig(folder/'paired_subject_patterns.png',dpi=180);plt.close(fig)
    write(folder/'integer_replay_review.json',dict(status='passed',saved_integer_rows=len(selected),ordinary_original_parent_sum_same_batch_bitwise=True,original_parent_IDs_bitwise=True,parent_batch1_vs_bank_batch64_max_abs=parent_batch_drift,independent_checkpoint_reload_bitwise=True,component_addition_roundoff_max_abs=rmsum,decoder_window_side_information=0,declared_vocabulary_by_fold={str(f):m['branch_codebook_sizes'] for f,m in meta.items()},rate_bits_per_second_by_fold={str(f):sum(np.ceil(np.log2(k)) for k in m['branch_codebook_sizes'].values())/2 for f,m in meta.items()},note='crossfit diagnostic folds, no final admitted candidate'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", choices=["screen", "development"])
    parser.add_argument('--feature-mode', choices=['axis','ridge'])
    parser.add_argument('--oracle-capacity',action='store_true')
    parser.add_argument('--review-train',action='store_true')
    parser.add_argument('--expansion-stage',choices=['plan','collect','fit'])
    parser.add_argument('--scale-audit',action='store_true')
    parser.add_argument('--render-training-patterns',action='store_true')
    args = parser.parse_args()
    if args.scale_audit and args.expansion_stage!='fit':
        parser.error('--scale-audit requires the retained expanded fit banks')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.render_training_patterns:
        render_training_patterns(out,out/'training_review_v2',yaml.safe_load(args.config.read_text()));return
    if args.evaluate:
        cfg = yaml.safe_load(args.config.read_text())
        torch.set_num_threads(cfg['resources']['torch_threads'])
        evaluate_frozen(cfg, out, args.evaluate)
        return
    if (out / "summary.json").exists() and not args.review_train and not args.scale_audit:
        raise FileExistsError("retained leaf experiment")
    cfg = yaml.safe_load(args.config.read_text())
    if args.feature_mode:
        if args.feature_mode not in cfg['model']['matched_feature_modes']:
            raise ValueError('unregistered feature mode')
        cfg['model']['feature_mode']=args.feature_mode
    budget_key={'collect':'collection_CPU_budget_seconds','fit':'fit_CPU_budget_seconds'}.get(args.expansion_stage,'total_CPU_limit_seconds')
    cpu_limit=int(cfg['resources'].get(budget_key,cfg['resources']['total_CPU_limit_seconds']))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit-30, cpu_limit))
    torch.set_num_threads(cfg["resources"]["torch_threads"])
    g0 = yaml.safe_load((ROOT / cfg["source"]["config"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest["rows"]
    panel = ROOT / cfg["source"]["panel"]
    fingerprint = json.loads(panel.with_name("panel.json").read_text())
    if (
        digest(panel) != fingerprint["panel_sha256"]
        or manifest["cfg_identity"] != fingerprint["cfg_identity"]
        or manifest["scales_identity"] != fingerprint["scales_identity"]
    ):
        raise ValueError("frozen G0 identity")
    with np.load(panel) as data:
        examples = data["hb_pattern_train"].copy()
    if any(
        rows[i]["split"] != "train" or rows[i]["modality"] != "hb"
        for i, _, _ in examples
    ):
        raise ValueError("train-only before waveform access")
    if args.expansion_stage:
        spec=cfg['training']['expanded_fit']
        expanded,support=expansion_plan(rows,examples,spec)
        np.testing.assert_array_equal(expanded[:len(examples)],examples)
        if len(examples)!=spec['evaluation_contexts']:raise ValueError('fixed original evaluation denominator')
        cache_source=ROOT/cfg['source']['expanded_fit_run'] if 'expanded_fit_run' in cfg['source'] else out
        path=cache_source/'expanded_fit_plan.npz'
        if path.exists():
            with np.load(path) as previous:np.testing.assert_array_equal(previous['examples'],expanded)
        elif args.expansion_stage=='plan':
            np.savez_compressed(path,examples=expanded,evaluation_contexts=len(examples))
            write(out/'expanded_fit_support.json',dict(support,original_panel_sha256=digest(panel),cfg_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity']))
        else:raise ValueError('frozen metadata-only expansion plan required before arrays')
        if args.expansion_stage=='plan':
            write(out/'resolved_config.json',cfg)
            if cache_source!=out:write(out/'expanded_fit_reference.json',dict(run=str(cache_source),plan_sha256=digest(path),chunks_sha256=digest(cache_source/'fit_chunks.json')))
            print(support['expanded_contexts'],support['added_contexts']);return
        examples=expanded
    parent, setting, metadata = prepare_parent(cfg, manifest, scales)
    if args.expansion_stage=='collect':
        if 'expanded_fit_run' in cfg['source']:raise ValueError('reuse frozen expanded banks without new collection')
        expand_collection(cfg,out,rows,scales,examples,parent);return
    if not args.review_train and not args.expansion_stage:
        write(out / "resolved_config.json", cfg)
    if 'training_bank' in cfg['source']:
        retained=ROOT/cfg['source']['training_bank']
        if digest(retained)!=cfg['source']['training_bank_sha256']:
            raise ValueError('retained original parent bank changed')
        with np.load(retained) as data:
            np.testing.assert_array_equal(examples[:len(data['examples'])],data['examples'])
            bank={k:data[k].copy() for k in data.files if k!='examples'}
        if not args.review_train and not args.scale_audit:
            write(out/'training_bank_reference.json',dict(path=str(retained),sha256=digest(retained),source_parent_sha256=digest(ROOT/cfg['source']['checkpoint'])))
    else:
        bank = collection(cfg, out, rows, scales, examples, parent)
    if args.expansion_stage=='fit':
        cache_source=ROOT/cfg['source']['expanded_fit_run'] if 'expanded_fit_run' in cfg['source'] else out
        collected=json.loads((cache_source/'fit_chunks.json').read_text())
        if collected['status']!='completed':raise ValueError('completed partitioned fit banks required')
        parts={k:[v] for k,v in bank.items()}
        for chunk in collected['chunks']:
            if digest(chunk['path'])!=chunk['sha256']:raise ValueError('retained fit chunk changed')
            with np.load(chunk['path']) as data:
                np.testing.assert_array_equal(data['examples'],examples[chunk['start']:chunk['stop']])
                for k in parts:parts[k].append(data[k].copy())
        bank={k:np.concatenate(v) for k,v in parts.items()}
        del parts
        if any(len(v)!=len(examples) for v in bank.values()):raise ValueError('complete fit bank support required')
    base_meta = dict(
        schema="hb_parent_leaf_v1",
        modality="hb",
        parent_checkpoint=str((ROOT / cfg["source"]["checkpoint"]).resolve()),
        parent_sha256=digest(ROOT / cfg["source"]["checkpoint"]),
        parent_model_setting=setting,
        parent_metadata=metadata,
        input_identity=manifest["cfg_identity"],
        cfg_identity=manifest["cfg_identity"],
        scales_identity=manifest["scales_identity"],
    )
    if args.review_train:
        review_training(cfg,out,rows,scales,examples,bank,parent,base_meta)
        return
    if args.scale_audit:
        expanded_readout_audit(cfg,out,rows,examples,bank)
        return
    if args.oracle_capacity:
        oracle_capacity(cfg,out,rows,examples,bank)
        return
    chosen, bank, quiet, q = crossfit(
        cfg, out, rows, scales, examples, bank, parent, base_meta
    )
    if chosen is not None:
        valid = bank["mask"].all(-1) & ~quiet[:, None]
        use = valid.ravel()
        subjects = np.array(
            [rows[i]["dataset"] + "/" + rows[i]["subject"] for i, _, _ in examples]
        )
        attrs = attributes_for_split(bank)
        fitted = fit_children(
            bank["id_O"].ravel()[use],
            bank["features"].reshape(-1, bank["features"].shape[-1])[use],
            attrs[use],
            (bank["o"] - bank["original_O"]).reshape(-1, 40)[use],
            np.repeat(subjects, 15)[use],
            dict(cfg["model"], max_leaves=chosen["max_leaves"]),
            cfg["training"]["fold_seed"],
        )
        if cfg['model'].get('emitted_gain_projection'):
            leaf=assign_leaf(bank['id_O'],bank['features'],fitted)
            gain,_=constrained_emitted_gain(bank,np.arange(len(examples)),examples,rows,scales,quiet,cfg,leaf,fitted,q,parent,out,'all_train')
            if gain is None:raise ValueError('no feasible full-train emitted gain; no checkpoint released')
        elif cfg['model'].get('gain_projection'):
            gain,_=constrained_gain(bank,np.arange(len(examples)),examples,rows,quiet,cfg)
        else:
            gain = color_gain(bank, np.arange(len(examples)), training_weights(examples, rows))
        meta = dict(
            base_meta,
            branch_codebook_sizes={"S": 129, "O": len(fitted["parent"]) + 1},
            color_gain=gain.tolist(),
            residual_strength=chosen["residual_strength"],
            residual_lowpass_hz=cfg["model"]["residual_lowpass_hz"],
            quiet_threshold=q["threshold_input_sd"],
            quiet_level_S=q["level_S"],
        )
        HbParentLeaf(parent, meta, fitted).save(out / "model.pt")
    write(
        out / "summary.json",
        dict(
            status="train_admitted" if chosen else "train_rejected",
            selected=None if chosen is None else chosen["candidate"],
            CPU_seconds=time.process_time(),
            training_contexts=len(examples),
            evaluated_contexts=cfg['training'].get('expanded_fit',{}).get('evaluation_contexts',len(examples)),
            complete_patches=int(bank["mask"].all(-1).sum()),
            training_subjects=len(
                set((rows[i]["dataset"], rows[i]["subject"]) for i, _, _ in examples)
            ),
            screen_read=False,
            confirmation_read=False,
            test_read=False,
            source_parent_sum_identity="bitwise_for_every_ordinary_training_window_every_fold_and_strength",
        ),
    )
    print("selected", None if chosen is None else chosen["candidate"])


if __name__ == "__main__":
    main()
