"""Train paid source innovations conditioned on actual cross-branch pilot integers."""
from __future__ import annotations

import argparse
from collections import Counter,OrderedDict,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import fcntl,hashlib,json,multiprocessing,os
from pathlib import Path
import subprocess,sys,time

import numpy as np
from numpy.lib.format import open_memmap
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"experiments/scripts")]
from src.analysis.dual_component_tokenizer import base_training_aggregate,joint_reconstruction_metrics
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit,feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_joint_eeg,base_tokenizer_batch
from src.tokenizers.adaptive_band_allocation import band_waves
from src.tokenizers.band_exchange_words import load_fold_books,paid_coordinates,source_properties
from src.tokenizers.cross_branch_sampling import calibration
from src.tokenizers.cross_branch_innovations import (
    PILOT_SIZE,pilot_positions,pilot_means,fit_second_books,encode_second,paid_fields,decode_words,posterior_waveforms,
)
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.utils.io import file_sha256,save_npz_atomic

STATE={};MODES=("direct","innovation");HARD=tuple(m+"_"+r for m in MODES for r in ("mean","energy"))
ARMS=HARD+("pilot_only","continuous_second")


def write(path,value):
    def convert(a):
        if isinstance(a,np.ndarray):return a.tolist()
        if isinstance(a,np.generic):return a.item()
        raise TypeError(type(a).__name__)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=convert)+"\n")


def budget():
    unit=os.environ.get("INNOVATION_UNIT")
    if unit:
        used=int(subprocess.check_output(["systemctl","--user","show",unit,"-p","CPUUsageNSec","--value"],text=True))/1e9
        if used>STATE["cfg"]["resources"]["CPU_seconds"]-15:raise RuntimeError("registered innovation CPU budget reached")


@lru_cache(maxsize=8)
def old_books(fold,b):
    return load_fold_books(STATE["dictionaries"]/"books",fold,b)


@lru_cache(maxsize=8)
def new_books(fold,b):
    result=[]
    for role in range(5):
        with np.load(STATE["out"]/"books"/f"fold_{fold}_{b}_{role}.npz") as z:result.append({k:z[k].copy() for k in z if k!="metadata"})
    return result


def prepared(dataset,fold,b):
    folder=STATE["out"]/dataset/f"fold_{fold}_prepared"
    if not (folder/"ready.json").exists():raise ValueError("completed folded source preparation required")
    return {k:np.load(folder/(b+"_"+k+".npy"),mmap_mode="r") for k in ("source","residual","properties","role","seen","weight")}


def prepare_dataset(dataset):
    cases=STATE["cases"][dataset];names=STATE["population"]["native_positions"][dataset];n=len(names)
    expected=STATE["examples"][dataset];N=len(expected);native=np.concatenate([c["native"] for c in cases])
    groups=np.concatenate([np.full(c["count"],c["fold"]) for c in cases])
    samples=[]
    for c in cases:
        if STATE["rows"][c["crop"]["row"]]["split"]!="train":raise ValueError("train before observations")
        v,seen,_=base_tokenizer_joint_eeg(STATE["rows"],dict(c["crop"],positions=[]),STATE["scales"],names,allowed_splits=("train",))
        samples.append((v["x"].T.reshape(1,n,15,400).astype(np.float32),seen))
    for fold in range(4):
        budget();folder=STATE["out"]/dataset/f"fold_{fold}_prepared";folder.mkdir(parents=True,exist_ok=True)
        if (folder/"ready.json").exists():continue
        source_path=STATE["estimators"]/f"{dataset}_fold{fold}_transient.npz"
        source=load_source(source_path)
        fit={c["subject"] for c in cases if c["fold"]!=fold};held={c["subject"] for c in cases if c["fold"]==fold}
        if fit&held or set(source.metadata["fit_subjects"])!=fit:raise ValueError("source person boundary mismatch")
        model_path=STATE["sampling"]/dataset/f"fold_{fold}_model.npz"
        with np.load(model_path) as z:
            if set(z["fit_people"])!=fit or set(z["held_people"])!=held:raise ValueError("joint covariance fit boundary mismatch")
            if str(z["source_sha256"])!=file_sha256(source_path):raise ValueError("different frozen X source")
            state=dict(joint=z["joint"].copy(),pilot_gain=z["staggered_gain"].copy(),pilot_noise=z["staggered_noise"].copy())
        with np.load(STATE["sampling"]/dataset/f"fold_{fold}_ids.npz") as z:
            np.testing.assert_array_equal(z["examples"],expected);np.testing.assert_array_equal(z["fold"],groups)
            streams={b:z["staggered_"+b].copy() for b in "SO"}
        fitting=[c for c in cases if c["fold"]!=fold]
        count=Counter((c["subject"],STATE["rows"][c["crop"]["row"]]["record"]) for c in fitting);records=Counter(p for p,r in count)
        buffers={}
        for b in "SO":
            buffers[b]={k:open_memmap(folder/(b+"_"+k+".npy"),mode="w+",dtype=np.float32,shape=shape) for k,shape in
                        (("source",(N,15,400)),("residual",(N,15,400)),("properties",(N,15,3)),("weight",(N,)))}
            buffers[b]["role"]=open_memmap(folder/(b+"_role.npy"),mode="w+",dtype=np.int8,shape=(N,))
            buffers[b]["seen"]=open_memmap(folder/(b+"_seen.npy"),mode="w+",dtype=bool,shape=(N,))
        sq=np.zeros((5,2*n));mass=np.zeros_like(sq);ratio=[];offset=0
        for c,(x,seen) in zip(cases,samples):
            budget();take=np.arange(offset,offset+c["count"]);offset+=c["count"];site=np.array(c["native"]);order=site.argsort()
            full=source.predict_continuous(x,dataset,seen[None])
            coordinates=[];roles=[]
            for b in "SO":
                q,tags=paid_coordinates(streams[b][take[order]],old_books(fold,b),"balanced",True)
                if not np.all(tags==tags[:,:1]):raise ValueError("fixed staggered layout required")
                coordinates.append(q);roles.append(tags[:,0])
            role=np.concatenate(roles);pilot=pilot_positions(n)
            pilot_coord=np.concatenate(coordinates);pilot_coord[~pilot]=0
            means,conditional=pilot_means(pilot_coord,role,state["joint"],state["pilot_gain"],state["pilot_noise"])
            subject=c["subject"];record=STATE["rows"][c["crop"]["row"]]["record"]
            weight=1/(len(fit)*records[subject]*count[subject,record]) if c["fold"]!=fold else 0.
            metrics=[]
            for bi,b in enumerate("SO"):
                fields=band_waves(np.fft.rfft(full[b][0].reshape(n,6000)));features=source_properties(full[b][0])
                available=seen&~pilot[bi*n:(bi+1)*n]
                target=np.zeros((n,15,400));residual=np.zeros_like(target)
                for band in range(5):
                    choose=available&(roles[bi]==band);target[choose]=fields[band,choose]
                    residual[choose]=fields[band,choose]-means[band,bi*n:(bi+1)*n][choose]
                    if c["fold"]!=fold:
                        sq[band,bi*n+np.flatnonzero(choose)]+=weight*np.square(residual[choose]).mean((1,2))
                        mass[band,bi*n+np.flatnonzero(choose)]+=weight
                    before=np.square(target[choose]).sum();after=np.square(residual[choose]).sum()
                    metrics.append((b,band,int(choose.sum()),float(before),float(after)))
                buffers[b]["source"][take]=target[site];buffers[b]["residual"][take]=residual[site]
                buffers[b]["properties"][take]=features[site];buffers[b]["role"][take]=roles[bi][site]
                buffers[b]["seen"][take]=available[site]
                buffers[b]["weight"][take]=weight/max(int(seen.sum())*15,1)
            ratio.append(dict(row=c["crop"]["row"],start_s=c["crop"]["start_s"],subject=c["subject"],fit=c["fold"]!=fold,band_energy=metrics))
        scale=np.sqrt(np.divide(sq,mass,out=np.zeros_like(sq),where=mass>0))
        fallback=np.sqrt(np.diagonal(state["joint"],axis1=1,axis2=2).clip(1e-16))
        scale=np.where(mass>0,np.maximum(scale,1e-8*fallback),fallback)
        state["scale"]=scale
        for b in "SO":
            for array in buffers[b].values():array.flush()
        save_npz_atomic(folder/"state.npz",**state,native=native,fold=groups,examples=expected,fit_people=np.array(sorted(fit)),held_people=np.array(sorted(held)))
        write(folder/"prediction_energy.json",ratio)
        write(folder/"ready.json",dict(dataset=dataset,fold=fold,source_sha256=file_sha256(source_path),upstream_model_sha256=file_sha256(model_path),fit_people=sorted(fit),held_people=sorted(held),fixed_scale="fit residual RMS by native branch site band"))
    return dataset


def balanced_sample(indices,people,maximum,rng):
    if len(indices)<=maximum:return indices
    groups=[rng.permutation(indices[people[indices//15]==p]) for p in np.unique(people[indices//15])]
    selected=[];n=0
    while n<maximum:
        active=[g for g in groups if len(g)];quota=max(1,(maximum-n)//len(active));groups=[]
        for g in active:
            k=min(len(g),quota,maximum-n);selected.append(g[:k]);n+=k
            if len(g)>k:groups.append(g[k:])
    return np.concatenate(selected)


def fit_book(task):
    fold,b,role=task;path=STATE["out"]/"books"/f"fold_{fold}_{b}_{role}.npz"
    if path.exists():return
    budget();seed=STATE["cfg"]["training"]["seed"]+100*fold+10*(b=="O")+role;rng=np.random.default_rng(seed)
    values={m:[] for m in MODES};properties=[];weights=[];energy=[];records=[]
    for dataset,cases in STATE["cases"].items():
        data=prepared(dataset,fold,b);here=STATE["out"]/dataset/f"fold_{fold}_prepared"
        with np.load(here/"state.npz") as z:
            group=z["fold"];site=z["native"];scale=z["scale"];n=scale.shape[1]//2
        valid=data["seen"]&(data["role"]==role)&(group!=fold)
        indices=np.flatnonzero(np.broadcast_to(valid[:,None],(len(valid),15)))
        people=np.array([STATE["rows"][i]["subject"] for i,t,p in STATE["examples"][dataset]])
        keep=balanced_sample(indices,people,4096,rng);rows=keep//15
        divisor=scale[role,site[rows]+n*(b=="O")]
        source=data["source"].reshape(-1,400)[keep].copy()
        for mode,key in (("direct","source"),("innovation","residual")):
            values[mode].append(data[key].reshape(-1,400)[keep].copy()/divisor[:,None])
        properties.append(data["properties"].reshape(-1,3)[keep].copy())
        weights.append(data["weight"][rows].astype(float));energy.append(np.square(source.astype(float)/divisor[:,None]).sum(1))
        records.append(dict(dataset=dataset,available=len(indices),used=len(keep),fit_people=len(np.unique(people[rows]))))
    start=time.process_time()
    fitted=fit_second_books({k:np.concatenate(v) for k,v in values.items()},np.concatenate(properties),np.concatenate(energy),np.concatenate(weights),role,seed,STATE["second_capacity"])
    save_npz_atomic(path,**fitted,metadata=np.array(json.dumps(dict(fold=fold,branch=b,role=role,seed=seed,sampling=records,CPU_seconds=time.process_time()-start))))
    return (fold,b,role)


def encode_dataset(dataset):
    cases=STATE["cases"][dataset]
    for fold in range(4):
        budget();dest=STATE["out"]/dataset;path=dest/f"fold_{fold}_ids.npz"
        if path.exists():continue
        folder=dest/f"fold_{fold}_prepared"
        with np.load(folder/"state.npz") as z:state={k:z[k].copy() for k in z}
        with np.load(STATE["sampling"]/dataset/f"fold_{fold}_ids.npz") as z:
            fields=dict(examples=z["examples"].copy(),fold=z["fold"].copy());old={b:z["staggered_"+b].copy() for b in "SO"}
        N=len(fields["examples"]);n=state["scale"].shape[1]//2
        moments={arm:np.zeros((4,5,2*n)) for arm in HARD}
        for b in "SO":
            data=prepared(dataset,fold,b);books=new_books(fold,b);site=state["native"]+n*(b=="O")
            for mode in MODES:
                paid=old[b].copy();paid[data["seen"]]=0
                for role,book in enumerate(books):
                    rows=np.flatnonzero(data["seen"]&(data["role"]==role))
                    for start in range(0,len(rows),128):
                        take=rows[start:start+128];scale=state["scale"][role,site[take]]
                        key="source" if mode=="direct" else "residual"
                        wave=data[key][take].copy()/scale[:,None,None]
                        prop=data["properties"][take].copy()
                        words=encode_second(wave.reshape(-1,400),prop.reshape(-1,3),book,mode,role).reshape(len(take),15)
                        paid[take]=words
                        fit=fields["fold"][take]!=fold
                        if not fit.any():continue
                        use=take[fit];ids=(words[fit].astype(np.int32)-PILOT_SIZE)%STATE["second_capacity"]
                        target=data[key][use].astype(float)
                        for energy in (False,True):
                            q=book[mode+"_atoms"][ids].astype(float)
                            if energy:q*=book[mode+"_energy_gain"][ids,None]
                            q*=state["scale"][role,site[use],None,None]
                            terms=np.stack([np.square(target).mean((1,2)),(target*q).mean((1,2)),np.square(q).mean((1,2)),np.ones(len(use))])
                            w=data["weight"][use].astype(float)
                            moment=moments[mode+("_energy" if energy else "_mean")]
                            for j in range(4):np.add.at(moment[j,role],site[use],w*terms[j])
                fields[mode+"_"+b]=paid
        save_npz_atomic(path,**fields)
        for arm,moment in moments.items():state[arm+"_gain"],state[arm+"_noise"]=calibration(moment)
        save_npz_atomic(dest/f"fold_{fold}_state.npz",**state)
    return dataset



def pack_words(words):
    flat=words.astype(np.uint32).ravel()
    return np.packbits(((flat[:,None]>>np.arange(14))&1).astype(np.uint8).ravel(),bitorder="little").tobytes()


def unpack_words(blob,shape):
    n=int(np.prod(shape));bits=np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder="little")[:14*n]
    return (bits.reshape(n,14)@(1<<np.arange(14))).astype(np.uint16).reshape(shape)


def score_dataset(task):
    dataset,replay=task;dest=STATE["out"]/dataset;cache=OrderedDict()
    for fold in range(4):
        budget();path=dest/f"fold_{fold}_metrics.npz"
        if path.exists() and not replay:continue
        with np.load(dest/f"fold_{fold}_ids.npz") as z:
            examples=z["examples"].copy()
            streams={mode:{b:z[mode+"_"+b].copy() for b in "SO"} for mode in MODES}
        with np.load(dest/f"fold_{fold}_state.npz") as z:state={k:z[k].copy() for k in z}
        old={b:old_books(fold,b) for b in "SO"};books={b:new_books(fold,b) for b in "SO"}
        if replay:
            with np.load(path) as z:expected={k:z[k].copy() for k in z}
        else:source={b:prepared(dataset,fold,b)["source"] for b in "SO"}
        measurements=defaultdict(list);gallery={};offset=0;held=0;transport=[]
        for c in STATE["cases"][dataset]:
            take=np.arange(offset,offset+c["count"]);offset+=c["count"]
            if c["fold"]!=fold:continue
            budget();site=np.array(c["native"]);ix=take[site.argsort()]
            ex=examples[take];predictions={}
            for mode in MODES:
                paid={}
                for b in "SO":
                    q=streams[mode][b][ix];file=dest/f'{c["crop"]["row"]}_{c["crop"]["start_s"]}_{mode}_{b}.14bit'
                    if not replay:file.write_bytes(pack_words(q))
                    paid[b]=unpack_words(file.read_bytes(),q.shape);np.testing.assert_array_equal(paid[b],q)
                    transport.append(dict(file=file.name,shape=list(q.shape),integers=q.size,bytes=file.stat().st_size,sha256=file_sha256(file)))
                for energy in (False,True):
                    loader=np.load
                    if replay:
                        def reject(*args,**kwargs):raise AssertionError("integer decoder attempted an additional file read")
                        np.load=reject
                    try:values=decode_words(paid,old,books,state,state,mode,energy)
                    finally:np.load=loader
                    predictions[mode+("_energy" if energy else "_mean")]={k:v[site] for k,v in values.items()}
                if mode=="innovation" and not replay:
                    first,_,roles=paid_fields(paid,old,books,state["scale"],mode,False)
                    unit=np.ones_like(state["pilot_gain"]);zero=np.zeros_like(unit)
                    pilot=posterior_waveforms(first,np.zeros_like(first),roles,state["joint"],state["pilot_gain"],state["pilot_noise"],zero,unit,"innovation")
                    values=np.concatenate([source[b][ix] for b in "SO"]).astype(float)
                    continuous=posterior_waveforms(first,values,roles,state["joint"],state["pilot_gain"],state["pilot_noise"],unit,zero,"direct")
                    predictions["pilot_only"]={k:v[site] for k,v in pilot.items()}
                    predictions["continuous_second"]={k:v[site] for k,v in continuous.items()}
            if any(STATE["rows"][i]["split"]!="train" for i,t,p in ex):raise ValueError("train only before target scoring")
            batch=base_tokenizer_batch(STATE["rows"],ex,STATE["scales"],array_cache=cache,cache_limit=8)
            factors=np.array([STATE["scales"]["positions"]["eeg"][dataset][STATE["rows"][i]["positions"][p]]["scale"] for i,t,p in ex])
            measurements["examples"].append(ex)
            interval=slice(held,held+c["count"])
            for arm,pred in predictions.items():
                score=joint_reconstruction_metrics(pred,batch,1,factors)
                if score["support"]["nonfinite_prediction_points"].any():raise ValueError("nonfinite pilot innovation waveform")
                for key,value in score["metrics"].items():
                    if replay:np.testing.assert_array_equal(value,expected[arm+"__"+key][interval])
                    else:measurements[arm+"__"+key].append(value)
                for j,e in enumerate(ex):
                    if tuple(e) in STATE["display"]:
                        label=STATE["display"][tuple(e)]
                        for k,v in pred.items():gallery[f"{label}_{arm}_{k}"]=v[j]
                        if arm=="innovation_mean":
                            for k in ("x","s","o","mask"):gallery[f"{label}_{k}"]=batch[k][j]
            held+=c["count"]
        if replay:write(dest/f"fold_{fold}_integer_replay.json",dict(status="passed",positions=held,all4hard_arms_all_metric_arrays_exact=True,decoder_additional_file_reads=0))
        else:
            save_npz_atomic(path,**{k:np.concatenate(v) for k,v in measurements.items()})
            save_npz_atomic(dest/f"fold_{fold}_gallery.npz",**gallery)
            write(dest/f"fold_{fold}_payloads.json",transport)
    return dataset



def review():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw
    import csv

    out=STATE["out"];destination=out/"visual_review_v1";destination.mkdir(exist_ok=True);snapshots={}
    STATE["selected"]={tuple(c[k] for k in ("row","start_s","position")):dict(c,audit_index=i) for i,c in enumerate(STATE["gallery"])}
    for dataset in STATE["cases"]:
        examples=[];scores=[]
        for fold in range(4):
            with np.load(out/dataset/f"fold_{fold}_gallery.npz") as z:
                for key in z:snapshots[key]=z[key].copy()
            with np.load(out/dataset/f"fold_{fold}_metrics.npz") as z:
                examples.append(z["examples"]);scores.append(z["innovation_mean__nmse"])
        examples=np.concatenate(examples);scores=np.concatenate(scores)
        for label,value in (("new_innovation_joint_worst",scores.mean((1,2))),("new_innovation_S_worst",scores[:,1,0])):
            identity=tuple(map(int,examples[np.argmax(value)]))
            if identity in STATE["selected"]:continue
            i,t,p=identity;row=STATE["rows"][i]
            STATE["selected"][identity]=dict(modality="eeg",dataset=dataset,subject=row["subject"],record=row["record"],parent=row["parent"],
                row=i,start_s=t,position=p,position_name=row["positions"][p],scale=STATE["scales"]["positions"]["eeg"][dataset][row["positions"][p]]["scale"],
                source_unit=row["source_unit"],selection=[label],audit_index=len(STATE["selected"]))
    for identity,case in STATE["selected"].items():
        index=case["audit_index"]
        if f"{index}_x" in snapshots:continue
        dataset=case["dataset"];offset=0;chosen=None
        for c in STATE["cases"][dataset]:
            if (c["crop"]["row"],c["crop"]["start_s"])==identity[:2]:chosen=c;break
            offset+=c["count"]
        if chosen is None:raise ValueError("display outside source population")
        fold=chosen["fold"];site=np.array(chosen["native"]);take=np.arange(offset,offset+chosen["count"]);ix=take[site.argsort()]
        with np.load(out/dataset/f"fold_{fold}_state.npz") as z:state={k:z[k].copy() for k in z}
        old={b:old_books(fold,b) for b in "SO"};books={b:new_books(fold,b) for b in "SO"}
        predictions={}
        with np.load(out/dataset/f"fold_{fold}_ids.npz") as z:
            examples=z["examples"][take];j=int(np.flatnonzero(examples[:,2]==identity[2])[0])
            for mode in MODES:
                sent={b:z[mode+"_"+b][ix] for b in "SO"}
                values=decode_words(sent,old,books,state,state,mode,False)
                predictions[mode+"_mean"]={k:v[site] for k,v in values.items()}
            sent={b:z["innovation_"+b][ix] for b in "SO"}
        first,_,roles=paid_fields(sent,old,books,state["scale"],"innovation",False)
        unit=np.ones_like(state["pilot_gain"]);zero=np.zeros_like(unit)
        values=posterior_waveforms(first,np.zeros_like(first),roles,state["joint"],state["pilot_gain"],state["pilot_noise"],zero,unit,"innovation")
        predictions["pilot_only"]={k:v[site] for k,v in values.items()}
        source=np.concatenate([prepared(dataset,fold,b)["source"][ix] for b in "SO"]).astype(float)
        values=posterior_waveforms(first,source,roles,state["joint"],state["pilot_gain"],state["pilot_noise"],unit,zero,"direct")
        predictions["continuous_second"]={k:v[site] for k,v in values.items()}
        batch=base_tokenizer_batch(STATE["rows"],examples,STATE["scales"])
        for key in ("x","s","o","mask"):snapshots[f"{index}_{key}"]=batch[key][j]
        for arm,values in predictions.items():
            for key,value in values.items():snapshots[f"{index}_{arm}_{key}"]=value[j]
    save_npz_atomic(destination / "display_arrays.npz", **snapshots)
    names={"target":"target","direct_mean":"direct word","innovation_mean":"innovation word","continuous_second":"continuous second slots","pilot_only":"paid pilots only"}
    color={"target":"#203d52","direct_mean":"#be823d","innovation_mean":"#197b64","continuous_second":"#8a6da8","pilot_only":"#89939b"}
    styles={"target":"-","direct_mean":"--","innovation_mean":":","continuous_second":"-","pilot_only":"-."}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure_records = []
    numbers = []
    hz = np.fft.rfftfreq(400, 1 / 200)
    taxis = np.arange(6000) / 200
    for identity, case in sorted(STATE["selected"].items(), key=lambda z: z[1]["audit_index"]):
        label = case["audit_index"]
        scale = float(case["scale"])
        unit = case["source_unit"]
        waves = {
            "target": {
                b: snapshots[f"{label}_{k}"].reshape(6000) * scale
                for b, k in (("X", "x"), ("S", "s"), ("O", "o"))
            }
        }
        for arm in ("direct_mean", "innovation_mean", "continuous_second", "pilot_only"):
            waves[arm] = {
                b: snapshots[f"{label}_{arm}_{k}"].reshape(6000) * scale
                for b, k in (("X", "sum"), ("S", "S"), ("O", "O"))
            }
        valid = snapshots[f"{label}_mask"].reshape(6000).astype(bool)
        peak_time = np.argmax(np.where(valid, np.abs(waves["target"]["X"]), -1)) / 200
        local = (
            min(28, max(0, 2 * np.floor(peak_time / 2))),
            min(30, max(2, 2 * np.floor(peak_time / 2) + 2)),
        )
        fig, axes = plt.subplots(3, 3, figsize=(15, 8.6), layout="constrained")
        for row, branch in enumerate(("X", "S", "O")):
            values = np.array([waves[arm][branch] for arm in waves])
            lo, hi = values.min(), values.max()
            margin = 0.06 * (hi - lo or 1)
            for col, (left, right) in enumerate(((0, 30), local, (13.5, 14.5))):
                ax = axes[row, col]
                for arm in waves:
                    ax.plot(
                        taxis,
                        waves[arm][branch],
                        c=color[arm],
                        ls=styles[arm],
                        lw=0.7 if arm != "target" else 0.85,
                        label=names[arm],
                        alpha=0.75 if arm == "continuous_second" else 1,
                    )
                for edge in np.arange(0, 31, 2):
                    ax.axvline(edge, c="#b8c1c6", ls=":", lw=0.45)
                ax.fill_between(
                    taxis, lo - margin, hi + margin, where=~valid, color="#bababa", alpha=0.35
                )
                ax.set(
                    xlim=(left, right),
                    ylim=(lo - margin, hi + margin),
                    xlabel="Time within 30-s context (s)",
                )
                if col == 0:
                    ax.set_ylabel(branch + " / " + unit)
                if row == 0:
                    ax.set_title(
                        (
                            "Full 30 s",
                            f"2 s at largest |X| event ({peak_time:.3f} s)",
                            "13.5–14.5 s token boundary",
                        )[col]
                    )
            for arm in ("direct_mean", "innovation_mean", "continuous_second", "pilot_only"):
                y, pred = waves["target"][branch], waves[arm][branch]
                j = int(np.argmax(np.where(valid, np.abs(y), -1)))
                near = np.arange(max(0, j - 100), min(6000, j + 101))
                near = near[valid[near]]
                k = int(near[np.argmax(np.abs(pred[near]))])
                numbers.append(
                    dict(
                        case=label,
                        dataset=case["dataset"],
                        subject=case["subject"],
                        record=case["record"],
                        position=case["position_name"],
                        arm=arm,
                        branch=branch,
                        RMSE_native=float(np.sqrt(np.mean(np.square(pred[valid] - y[valid])))),
                        target_peak_time_s=j / 200,
                        target_peak=float(y[j]),
                        prediction_same_time=float(pred[j]),
                        prediction_local_abs_peak=float(pred[k]),
                        prediction_local_peak_time_s=k / 200,
                        selected_fraction=None,
                        unit=unit,
                    )
                )
        axes[0, 0].legend(fontsize=7, ncol=2)
        identity_text = f"{case['dataset']} | {case['subject']} | {case['record']} | {case['position_name']} | crop +{case['start_s']}s"
        fig.suptitle(identity_text + " | one 14-bit word per branch site and 2 s; actual paid pilots", fontsize=10)
        stem = f"{label:03d}_{case['dataset']}_{case['subject']}_{case['position_name']}"
        voltage = destination / (stem + "_voltage.png")
        fig.savefig(voltage, dpi=220, facecolor="white")
        plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(14, 8.4), layout="constrained")
        block_valid = valid.reshape(120, 50).all(1)
        patch_valid = valid.reshape(15, 400).all(1)
        for row, branch in enumerate(("X", "S", "O")):
            for j, arm in enumerate(waves):
                y = waves[arm][branch]
                envelope = np.sqrt(np.square(y.reshape(120, 50)).mean(1))
                envelope[~block_valid] = np.nan
                axes[row, 0].plot(
                    np.arange(120) * 0.25 + 0.125,
                    envelope,
                    c=color[arm],
                    ls=styles[arm],
                    lw=0.9,
                    label=names[arm],
                )
                patches = y.reshape(15, 400)[patch_valid]
                power = (
                    np.abs(
                        np.fft.rfft(
                            (patches - patches.mean(1, keepdims=True)) * np.hanning(400), axis=1
                        )
                    )
                    ** 2
                    * 2
                    / (400 * np.square(np.hanning(400)).sum())
                )
                energy = [
                    power[:, (hz >= a) & ((hz <= b) if b == 45 else (hz < b))].sum(1).mean()
                    for a, b in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
                ]
                axes[row, 1].bar(
                    np.arange(5) + (j - 2) * 0.16,
                    np.maximum(energy, 1e-25),
                    0.15,
                    color=color[arm],
                    label=names[arm],
                )
            axes[row, 0].set(
                xlim=(0, 30),
                xlabel="Time within 30-s context (s)",
                ylabel=branch + " quarter-second RMS / " + unit,
            )
            axes[row, 1].set(
                xticks=np.arange(5),
                xticklabels=["1–4", "4–8", "8–13", "13–30", "30–45"],
                yscale="log",
                xlabel="Frequency band (Hz)",
                ylabel=branch + " mean complete-patch power / unit²",
            )
        axes[0, 0].legend(fontsize=7, ncol=2)
        axes[0, 1].legend(fontsize=7, ncol=2)
        fig.suptitle(identity_text + " | complete-patch bands and quarter-second RMS", fontsize=10)
        dynamics = destination / (stem + "_dynamics.png")
        fig.savefig(dynamics, dpi=220, facecolor="white")
        plt.close(fig)
        figure_records.append(
            dict(
                identity=case,
                voltage=voltage.name,
                dynamics=dynamics.name,
                mask_valid_points=int(valid.sum()),
                status="rendered_pending_actual_view",
            )
        )
    with (destination / "example_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(numbers[0]))
        writer.writeheader()
        writer.writerows(numbers)
    write(
        destination / "figure_manifest.json",
        dict(
            examples=figure_records,
            figure_count=2 * len(figure_records),
            status="rendered_pending_actual_view",
            actual_interface="direct and innovation each send one 14bit ID per branch site per2s; continuous second slots are a capacity diagnostic",
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )
    for dataset in STATE["cases"]:
        chosen = [r for r in figure_records if r["identity"]["dataset"] == dataset]
        for kind in ("voltage", "dynamics"):
            thumbnails = []
            for r in chosen:
                im = Image.open(destination / r[kind])
                im.thumbnail((1000, 600))
                canvas = Image.new("RGB", (1020, 630), "white")
                canvas.paste(im, ((1020 - im.width) // 2, 25))
                ImageDraw.Draw(canvas).text((10, 5), str(r["identity"]["audit_index"]), fill="black")
                thumbnails.append(canvas)
            canvas = Image.new("RGB", (1020, 630 * len(thumbnails)), "white")
            for n, im in enumerate(thumbnails):
                canvas.paste(im, (0, 630 * n))
            canvas.save(destination / (dataset + "_" + kind + "_contact.jpg"), quality=90)
    html = "<html><head><meta charset='utf-8'><title>付费pilot与条件创新的单ID比较</title></head><body><h1>付费pilot与条件创新的单ID比较</h1>"
    html += (
        "<p>每支每位置每2秒发送一个14bit整数。先由pilot整数形成条件均值，再解码直接源或创新字。连续第二槽和pilot单独重建为诊断对照，保留原17例及每库新增最差。</p>"
    )
    for r in figure_records:
        c = r["identity"]
        html += f"<h2>{c['audit_index']} {c['dataset']} {c['subject']} {c['position_name']}</h2><p>{' / '.join(c['selection'])}</p>"
        html += "".join(
            f'<a href="{r[k]}"><img width="100%" src="{r[k]}"></a>' for k in ("voltage", "dynamics")
        )
    (destination / "index.html").write_text(html + "</body></html>")


def information():
    with np.load(STATE["run"]/"component_source_transfer/hard_as_v1/local_information_v1/training_attributes.npz") as z:bank={k:z[k].copy() for k in z}
    np.testing.assert_array_equal(bank["examples"],np.concatenate(list(STATE["examples"].values())))
    fields={"theta":(0,1),"alpha":(1,2),"RMS":(2,10)};backgrounds=4*len(STATE["population"]["background_sites"])
    results={}
    for mode in MODES:
        results[mode]={}
        for b in "SO":
            metrics={name:np.empty((len(bank["examples"]),2,3)) for name in ("word","type_role","within_type_role")}
            mass=defaultdict(lambda:[0,0]);usage={}
            for fold in range(4):
                budget();parts=[]
                for dataset in STATE["cases"]:
                    with np.load(STATE["out"]/dataset/f"fold_{fold}_ids.npz") as z:parts.append(z[mode+"_"+b].copy())
                words=np.concatenate(parts);stage=words>=PILOT_SIZE
                role=np.where(words>0,(words.astype(np.int32)-1)//1023+1,0)
                role[stage]=6+(words[stage].astype(np.int32)-PILOT_SIZE)//STATE["second_capacity"]
                fit=bank["fold"]!=fold;held=~fit;valid=bank["valid"]
                assert set(bank["subject"][fit]).isdisjoint(bank["subject"][held])
                for name,codes,background,k,n in (("word",words,bank["background"],STATE["cfg"]["interface"]["K_including_missing"],backgrounds),
                                                 ("type_role",role,bank["background"],11,backgrounds),
                                                 ("within_type_role",words,11*bank["background"]+role,STATE["cfg"]["interface"]["K_including_missing"],11*backgrounds)):
                    model=SparsePatternAudit(k,n,[],10,fields);take=valid&fit[:,None]
                    model.fit(codes[take],background[take],bank[b+"_attributes"][take])
                    test=valid[held];key=codes[held][test];bg=background[held][test]
                    baseline,_,prediction=model.read(key,bg);target=bank[b+"_attributes"][held][test]
                    error=np.zeros((*test.shape,2,3));error[test]=np.stack([feature_means(np.square(v-target),fields) for v in (baseline,prediction)],axis=1)
                    count=test.sum(1);window=error.sum(1)/count.clip(1)[:,None,None];window[count==0]=np.nan;metrics[name][held]=window
                    people=np.zeros(k,np.int32)
                    for person in np.unique(bank["subject"][fit]):
                        seen=valid&(bank["subject"][:,None]==person);people[np.unique(codes[seen])]+=1
                    qualified=(model.local.count[key]>=8)&(people[key]>=2)
                    mass[name][0]+=int(qualified.sum());mass[name][1]+=len(key)
                    if name=="word":
                        usage[f"fit_count_{fold}"]=model.local.count;usage[f"fit_people_{fold}"]=people
                        usage[f"held_count_{fold}"]=np.bincount(key,minlength=k)
                heldmask=valid&held[:,None];fitmask=valid&fit[:,None]
                # Full namespace and each type stay visible in retained usage.
                usage[f"fit_type_count_{fold}"]=np.bincount(stage[fitmask],minlength=2)
                usage[f"held_type_count_{fold}"]=np.bincount(stage[heldmask],minlength=2)
            branch={}
            for name,value in metrics.items():
                hierarchy=base_training_aggregate({"nmse":value},bank["examples"],STATE["rows"])["metrics"]["nmse"]
                m=np.asarray(hierarchy["macro"]);gain=1-m[1]/m[0]
                people=np.sum([np.array(v["value"])[1]<np.array(v["value"])[0] for v in hierarchy["by_subject"]],axis=0)
                support=mass[name][0]/mass[name][1]
                branch[name]=dict(gain=gain,positive_people=people,support=support,hierarchy=hierarchy,
                                  passed=bool(min(gain[:2])>0 and max(gain)>=.05 and min(people[:2])>=30 and support>=.9))
            results[mode][b]=branch
            save_npz_atomic(STATE["out"]/f"{mode}_{b}_word_metrics.npz",examples=bank["examples"],**metrics,**usage)
    write(STATE["out"]/"single_word_information.json",results)


def summarize():
    arrays=defaultdict(list)
    for dataset in STATE["cases"]:
        for fold in range(4):
            with np.load(STATE["out"]/dataset/f"fold_{fold}_metrics.npz") as z:
                for key in z:arrays[key].append(z[key])
    ex=np.concatenate(arrays.pop("examples"))
    if sorted(map(tuple,ex))!=sorted(map(tuple,np.concatenate(list(STATE["examples"].values())))):raise ValueError("changed original scored population")
    old=json.loads((STATE["reference"]/"summary.json").read_text())["arms"]["high_transient_source"]["metrics"]["nmse"]
    info=json.loads((STATE["out"]/"single_word_information.json").read_text());results={}
    for arm in ARMS:
        values={k.split("__",1)[1]:np.concatenate(v) for k,v in arrays.items() if k.startswith(arm+"__")}
        stats=base_training_aggregate(values,ex,STATE["rows"]);m=stats["metrics"];nm=np.ravel(m["nmse"]["macro"]);ref=np.ravel(old["macro"]);gain=np.ravel(m["rms_gain"]["macro"])
        checks=dict(X=nm[0]<=1.05*ref[0],components=np.all(nm[1:]<=1.1*ref[1:]),
                    datasets=all(np.all(np.asarray(v)<=1.1*np.asarray(old["by_dataset"][d])) for d,v in m["nmse"]["by_dataset"].items()),
                    RMS=np.all((gain>=.75)&(gain<=1.25)),bands=all(np.max(abs(np.asarray(m[k]["macro"])))<=3.0103 for k in ("band_4_8_db","band_8_13_db","band_4_8_abs_db","band_8_13_abs_db")),
                    correlation=min(np.ravel(m["correlation"]["macro"]))>0,envelope=min(np.ravel(m["local_rms_correlation"]["macro"]))>=.5)
        hard=arm in HARD;waveform=all(checks.values())
        if hard:
            mode=arm.rsplit("_",1)[0]
            checks.update({f"actual_word_{b}":info[mode][b]["word"]["passed"] for b in "SO"})
        results[arm]=dict(summary=stats,waveform_checks={k:v for k,v in checks.items() if not k.startswith("actual_word")},waveform_pass=waveform,
                          checks=checks,finite_single_ID_candidate=hard,passed=hard and all(checks.values()))
    write(STATE["out"]/"summary.json",dict(arms=results,eligible=[a for a,v in results.items() if v["passed"]],original_arrays=400,original_positions=17040,original_people=60,
        declared_K=STATE["cfg"]["interface"]["K_including_missing"],paid_bits_per_ID=14,integers_per_condition=511200,EEG_pair_bits_per_native_position_second=14.,validation_reads=0,confirmation_reads=0,test_reads=0))


def initialize(args):
    cfg=yaml.safe_load(args.config.read_text())
    if cfg["schema"]!="cross_branch_innovations_v1" or cfg["data"]["validation"]!="forbidden":raise ValueError("train-only conditional-word contract required")
    capacity=cfg["training"].get("second_capacity",1023)
    if capacity<1 or PILOT_SIZE+5*capacity!=cfg["interface"]["K_including_missing"] or (PILOT_SIZE+5*capacity-1).bit_length()!=14:
        raise ValueError("second-word capacity must match the full fixed 14-bit transport")
    STATE["second_capacity"]=capacity
    run=ROOT/cfg["joint_run"];population=json.loads((run/cfg["source"]["population"]).read_text())
    manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/"experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml").read_text()),ROOT)
    if (population["cfg_identity"],population["scales_identity"])!=(manifest["cfg_identity"],manifest["scales_identity"]):raise ValueError("G0 coordinate changed")
    cases={d:[c for c in population["cases"] if c["dataset"]==d] for d in sorted(population["native_positions"])}
    examples={d:np.array([(c["crop"]["row"],c["crop"]["start_s"],p) for c in cc for p in c["crop"]["positions"]]) for d,cc in cases.items()}
    if sum(map(len,examples.values()))!=17040 or any(manifest["rows"][i]["split"]!="train" for ex in examples.values() for i,t,p in ex):raise ValueError("fixed train population required before signal reads")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/"books").mkdir(exist_ok=True)
    identity=dict(config=cfg,population_sha256=file_sha256(run/cfg["source"]["population"]),input_identity=manifest["cfg_identity"],scales_identity=manifest["scales_identity"])
    file=out/"resolved_config.json"
    if file.exists() and json.loads(file.read_text())!=identity:raise ValueError("different retained pilot-innovation contract")
    write(file,identity)
    draws=json.loads((run/"component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json").read_text())["examples"]
    STATE.update(gallery=draws,display={tuple(c[k] for k in ("row","start_s","position")):i for i,c in enumerate(draws)},cfg=cfg,run=run,out=out,population=population,cases=cases,examples=examples,rows=manifest["rows"],scales=scales,
                 sampling=run/cfg["source"]["sampling"],dictionaries=run/cfg["source"]["dictionaries"],estimators=run/cfg["source"]["source_estimators"],
                 reference=run/cfg["source"]["reference"])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--stage",choices=("prepare","fit","encode","score","replay","information","summarize","review","evaluate","all"),default="all")
    args=p.parse_args();initialize(args)
    lock=(STATE["out"]/"controller.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stages=("prepare","fit","encode","score","information","summarize") if args.stage=="all" else (("score","information","summarize") if args.stage=="evaluate" else (args.stage,))
    for stage in stages:
        start=time.monotonic()
        if stage in ("information","summarize","review"):
            {"information":information,"summarize":summarize,"review":review}[stage]()
        else:
            function={"prepare":prepare_dataset,"fit":fit_book,"encode":encode_dataset,"score":score_dataset,"replay":score_dataset}[stage]
            tasks=[(f,b,r) for f in range(4) for b in "SO" for r in range(5)] if stage=="fit" else ([(d,stage=="replay") for d in STATE["cases"]] if stage in ("score","replay") else list(STATE["cases"]))
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("fork")) as pool:list(pool.map(function,tasks))
        write(STATE["out"]/(stage+"_complete.json"),dict(stage=stage,wall_seconds=time.monotonic()-start))
if __name__=="__main__":main()
