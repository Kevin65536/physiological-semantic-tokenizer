#!/usr/bin/env python3
"""Render a dated comparison from retained public results and published tables.

No fitting, raw signals, historical protected predictions or live queue writes.
Figures are PNG bitmaps; PDF body and tables remain searchable text.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sys

import fitz
import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.metrics.downstream_benchmark import score_predictions

TASKS = ["motor_imagery", "mental_arithmetic", "wg", "nback", "dsr", "visual"]
ALL_TASKS = TASKS + ["refed_regression"]
LABELS = dict(zip(ALL_TASKS, ["MI", "MA", "WG", "n-back", "DSR", "Visual", "REFED"]))
METHODS = ["biot", "cbramod", "reve", "efrm", "normwear", "brainfusion", "sta_net"]
NAMES = dict(zip(METHODS, ["BIOT", "CBraMod", "REVE", "EFRM", "NormWear", "BrainFusion", "STA-Net"]))
NAMES["signal_linear"] = "简单信号基线"
TRACK = {m: ("supervised" if m in {"sta_net", "brainfusion"} else "lp") for m in METHODS}
OLD_TASKS = {"Motor imagery": "motor_imagery", "Mental arithmetic": "mental_arithmetic", "Word generation": "wg",
             "n-back": "nback", "DSR": "dsr", "Visual": "visual", "REFED": "refed_regression"}
OLD_METHODS = {"BIOT": "biot", "CBraMod": "cbramod", "REVE": "reve", "EFRM": "efrm",
               "NormWear adapted": "normwear", "BrainFusion reimplementation": "brainfusion"}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    def default(item):
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        raise TypeError(f"unsupported report value: {type(item).__name__}")
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=default) + "\n")


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def table(headers, rows):
    def cell(v):
        return str(v).replace("|", "／").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(cell, row)) + " |" for row in rows]])


def number(value, scale=100):
    return "—" if value is None else f"{value * scale:.2f}"


def cell_key(cell):
    return cell["method"], cell["task"], cell["track"], cell.get("baseline_modality", "not_applicable")


def history_table(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 7 and parts[0] in OLD_METHODS:
            rows.append(dict(method=OLD_METHODS[parts[0]], task=OLD_TASKS[parts[1]], track=TRACK[OLD_METHODS[parts[0]]],
                             metric=parts[3], value=None if parts[4] == "—" else float(parts[4]),
                             fold_sd=None if parts[5] == "—" else float(parts[5]), decision=parts[6].strip("`")))
        elif len(parts) == 3 and parts[0] in OLD_TASKS and "±" in parts[2]:
            values = re.findall(r"[\d.]+", parts[2])
            scale = 1 if parts[0] == "REFED" else 100
            rows.append(dict(method="sta_net", task=OLD_TASKS[parts[0]], track="supervised", metric=parts[1],
                             value=float(values[0]) / scale, fold_sd=float(values[1]) / scale, decision="method_native_context_reference"))
    if len(rows) != 49:
        raise ValueError(f"expected the published 42 cells plus seven STA-Net context cells, got {len(rows)}")
    return rows


def bootstrap(values):
    x = np.asarray(values, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("undefined subject bootstrap input")
    indices = np.random.default_rng(20261016).integers(0, len(x), size=(2000, len(x)))
    return np.quantile(x[indices].mean(1), [.025, .975]).tolist()


def companions(summary, schedule, inventory):
    """Read only the exact valid job set in the dated aggregate, not live status."""
    full = {cell_key(c): c for c in summary["cells"] if c["coverage_complete"] and c["task"] != "refed_regression"}
    jobs = {j["id"]: j for j in schedule["jobs"]}
    by_subject = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    pooled = defaultdict(lambda: defaultdict(list))
    training = []
    input_files = []
    for source in summary["source_inputs"]:
        if not source.get("valid"):
            continue
        job = jobs[source["schedule_job_id"]]
        if job.get("method") == "signal_linear":
            resolved = read(job["job_path"])
            job = {**job, "baseline_modality": resolved["method_profile"]["baseline_modality"]}
        k = cell_key(job)
        if k not in full:
            continue
        result_path, prediction_path = Path(source["result_path"]), Path(source["predictions_path"])
        # These paths belong to the current public-v5 run, never a legacy campaign.
        if schedule["run_root"].split("/")[-1] not in prediction_path.parts or "protected_campaign" in prediction_path.parts:
            raise ValueError("unexpected prediction evidence boundary")
        result = read(result_path)
        if result["status"] != "completed" or result["job_identity"] != source["job_identity"]:
            raise ValueError("result identity differs from the retained aggregate")
        input_files += [str(result_path), str(prediction_path)]
        with np.load(prediction_path, allow_pickle=False) as z:
            target, pred, mask, subjects = [z[name] for name in ("target", "prediction", "mask", "subject_id")]
            scores = score_predictions(target, pred, mask, subjects, job["task"], len(inventory["tasks"][job["task"]]["classes"]))
            kclasses = len(inventory["tasks"][job["task"]]["classes"])
            for row in scores["subjects"]:
                selected = (subjects == row["subject_id"]) & mask
                data = dict(macro_f1=row["score"], accuracy=float(accuracy_score(target[selected], pred[selected])),
                            balanced_accuracy=float(balanced_accuracy_score(target[selected], pred[selected])))
                if job["seed"] in by_subject[k][row["subject_id"]]["seeds"]:
                    raise ValueError("a subject appears in multiple outer folds for one seed")
                by_subject[k][row["subject_id"]]["seeds"][job["seed"]] = data
            pooled[k][job["outer_fold"]].append(float(f1_score(target[mask], pred[mask], labels=np.arange(kclasses), average="macro", zero_division=0)))
        if job["track"] in {"ft", "supervised"}:
            detail = result.get("training", {})
            training.append(dict(method=job["method"], task=job["task"], track=job["track"], fold=job["outer_fold"],
                                 seed=job["seed"], selected_candidate=result.get("selected_candidate"),
                                 refit_epochs=detail.get("epochs_completed"), optimizer=detail.get("optimizer_spec"),
                                 result_path=str(result_path)))
    rows, subject_means = [], {}
    for k, cell in full.items():
        values = by_subject[k]
        if len(values) != cell["subject_count"]:
            raise ValueError("subject denominator changed")
        subject_means[k] = {s: {metric: float(np.mean([v[metric] for v in item["seeds"].values()]))
                               for metric in ("macro_f1", "accuracy", "balanced_accuracy")} for s, item in values.items()}
        f1 = np.mean([v["macro_f1"] for v in subject_means[k].values()])
        if not np.isclose(f1, cell["value"], atol=1e-12, rtol=0):
            raise ValueError("companion calculation did not reproduce owning primary")
        row = dict(method=k[0], task=k[1], track=k[2], baseline_modality=k[3], subject_count=cell["subject_count"],
                   primary_macro_f1=float(f1), current_fold_pooled_f1=float(np.mean([np.mean(v) for v in pooled[k].values()])))
        for metric in ("accuracy", "balanced_accuracy"):
            data = [v[metric] for v in subject_means[k].values()]
            row[metric] = float(np.mean(data)); row[metric + "_ci95"] = bootstrap(data)
        rows.append(row)
    paired = []
    for k, data in subject_means.items():
        targets = []
        if k[2] == "ft":
            targets.append((k[0], k[1], "lp", "not_applicable"))
        if k[0] != "signal_linear":
            modality = "eeg" if k[0] in {"biot", "cbramod", "reve"} else "multimodal"
            targets.append(("signal_linear", k[1], "lp", modality))
            if k[0] == "efrm": targets.append(("signal_linear", k[1], "lp", "fnirs"))
        for b in targets:
            if b not in subject_means:
                continue
            if set(data) != set(subject_means[b]):
                raise ValueError("paired comparison requires identical subject support")
            diff = [data[s]["macro_f1"] - subject_means[b][s]["macro_f1"] for s in sorted(data)]
            paired.append(dict(method=k[0], task=k[1], track=k[2], reference_method=b[0], reference_track=b[2],
                               reference_modality=b[3], difference=float(np.mean(diff)), ci95=bootstrap(diff),
                               subject_count=len(diff), interpretation="fixed-fit subject bootstrap, 2000 draws; exploratory; no multiplicity correction"))
    return rows, paired, training, input_files


def window_ledger(inventory):
    rows, exclusions, channels = [], [], []
    for e in inventory["exclusions"]:
        if e["join_key"].startswith("simultaneous_eeg_nirs|"):
            exclusions.append(e)
    for task in ALL_TASKS:
        method = "EFRM-PyTorch" if task == "refed_regression" else "BIOT"
        source = ROOT / f"comparative_methods/{method}/evidence/alignment_v2/{task}.json"
        old = read(source)
        old_count = old.get("public_audit", old.get("public_adapter_audit"))["unique_sample_count"]
        current = inventory["tasks"][task]
        rows.append(dict(task=task, old_public_audit_windows=old_count, current_windows=current["sample_count"],
                         delta=current["sample_count"] - old_count, current_subjects=current["subject_count"],
                         current_class_counts=current.get("class_counts"), old_source=str(source),
                         excluded_unverified_boundary={"wg":91,"nback":81,"dsr":680}.get(task,0),
                         newly_excluded_drift_windows=2880 if task=="dsr" else 0,
                         probe_coalescing_reduction=3860 if task=="visual" else 0))
        if task in TASKS:
            for method in ("BIOT", "CBraMod", "REVE"):
                old_cfg = yaml.safe_load((ROOT / f"comparative_methods/{method}/configs/alignment_v2.yaml").read_text())
                before = old_cfg["tasks"][task]["panel"]
                after = current["profiles"]["eeg_only"]["channel_names"]
                channels.append(dict(method=method, task=task, old_channels=before, new_channels=after,
                                     common_count=len(set(before) & set(after)), removed=sorted(set(before)-set(after)),
                                     added=sorted(set(after)-set(before))))
    counts = Counter((e["join_key"].split("|")[-1], e["reason"]) for e in exclusions)
    assert counts[("cnt_wg", "unverified_concatenation_interval")] == 91
    assert counts[("cnt_nback", "unverified_concatenation_interval")] == 81
    assert counts[("cnt_dsr", "unverified_concatenation_interval")] == 680
    newly_rejected = [e for e in exclusions if e["reason"] == "unadmitted_alignment" and "|VP005|" not in e["join_key"]]
    assert len(newly_rejected) == 8 and sum(e["event_count"] for e in newly_rejected) == 2880
    assert 8980 - 2880 - 680 == inventory["tasks"]["dsr"]["sample_count"]
    assert 7720 // 2 == inventory["tasks"]["visual"]["sample_count"]
    assert all(len(s["event_refs"]) == 2 for s in inventory["samples"] if s["task_id"] == "visual")
    assert all(s["event_refs"][0]["eeg_time_ms"]==s["event_refs"][1]["eeg_time_ms"]
               for s in inventory["samples"] if s["task_id"]=="visual")
    return rows, exclusions, channels


def make_figures(out, differences, paired, paper_accuracy):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    matplotlib.font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": FontProperties(fname=font_path).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "savefig.dpi": 250})
    figdir = out / "figures"
    figdir.mkdir()
    lookup = {(r["method"], r["task"]): r for r in differences}
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 7.5), layout="constrained")
    for ax, field, title in zip(axes, ["raw_difference", "old_style_difference"],
                               ["A  两轮主表直接相减（含汇总口径差异）", "B  本轮按旧口径重算后再相减"]):
        values = np.full((len(METHODS), len(TASKS)), np.nan)
        for i, method in enumerate(METHODS):
            for k, task in enumerate(TASKS):
                row = lookup.get((method, task))
                if row: values[i, k] = row[field] * 100
        cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#e9edf1")
        artist = ax.imshow(values, vmin=-10, vmax=10, cmap=cmap, aspect="auto")
        for i, method in enumerate(METHODS):
            for k, task in enumerate(TASKS):
                value = values[i, k]
                label = f"{value:+.2f}" if np.isfinite(value) else ("移出" if method == "normwear" else "不支持" if method == "brainfusion" else "未齐")
                ax.text(k, i, label, ha="center", va="center", fontsize=9.5,
                        color="white" if np.isfinite(value) and abs(value)>6 else "#172033")
        ax.set_xticks(range(6), [LABELS[t] for t in TASKS]); ax.set_yticks(range(7), [NAMES[m] for m in METHODS])
        ax.set_title(title, fontsize=12, loc="left")
    fig.colorbar(artist, ax=axes, shrink=.9, label="本轮 − 第一轮（macro-F1 百分点）")
    fig.savefig(figdir / "01_round_differences.png", bbox_inches="tight"); plt.close(fig)

    rows = sorted([r for r in paired if r["track"]=="ft" and r["reference_method"]==r["method"]],
                  key=lambda r: (METHODS.index(r["method"]), TASKS.index(r["task"])))
    fig, ax = plt.subplots(figsize=(7.1, 4.8), layout="constrained")
    for i, row in enumerate(rows):
        value = row["difference"] * 100; low, high = np.asarray(row["ci95"])*100
        color = "#2466a1" if row["method"]=="biot" else "#a34434"
        ax.errorbar(value, i, xerr=[[value-low],[high-value]], fmt="o", color=color, capsize=4, lw=1.6)
        ax.text(high+.45, i, f"{value:+.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [f"{NAMES[r['method']]} · {LABELS[r['task']]}  (n={r['subject_count']})" for r in rows])
    ax.invert_yaxis(); ax.axvline(0, color="#555", ls="--", lw=1)
    ax.set_xlim(-8, 24); ax.set_xlabel("FT − LP（macro-F1 百分点）；横线为探索性 95% 区间")
    ax.grid(axis="x", alpha=.2); ax.set_title("同一轮内，微调收益具有任务依赖性", loc="left", fontsize=13)
    fig.savefig(figdir / "02_finetuning_gain.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 4.1), layout="constrained")
    for i, row in enumerate(paper_accuracy):
        for offset, key, color in [(-.22, "paper_accuracy", "#b87624"), (0, "first_accuracy", "#7a838c"), (.22, "current_accuracy", "#2466a1")]:
            if row[key] is None: continue
            value = row[key]*100
            ax.barh(i+offset, value, height=.19, color=color)
            ax.text(value+.6, i+offset, f"{value:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(paper_accuracy)), [f"{NAMES[r['method']]} · {LABELS[r['task']]}" for r in paper_accuracy])
    ax.invert_yaxis(); ax.set_xlim(0,108); ax.set_xlabel("Accuracy（%）；同指标、不同协议，不能作因果差值")
    handles=[matplotlib.patches.Patch(color=c,label=l) for c,l in [("#b87624","作者原文"),("#7a838c","第一轮"),("#2466a1","第二轮")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,-.18), ncol=3, frameon=False, fontsize=9)
    ax.set_title("STA-Net 与 BrainFusion 的原文背景比较",loc="left",fontsize=13)
    fig.savefig(figdir / "03_paper_accuracy_context.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.1, 1.6)); ax.axis("off")
    for y, formula in [(0.80,r"$\mathrm{MacroF1}=\frac{1}{K}\sum_{c=1}^{K}\frac{2TP_c}{2TP_c+FP_c+FN_c}$"),
                       (0.40,r"$\mathrm{CCC}=\frac{2\,\mathrm{cov}(y,\hat y)}{\mathrm{var}(y)+\mathrm{var}(\hat y)+(\bar y-\bar{\hat y})^2}$"),
                       (0.02,r"$\mathrm{MAE}=\mathrm{mean}|y-\hat y|,\quad\mathrm{RMSE}=\sqrt{\mathrm{mean}(y-\hat y)^2}$")]:
        ax.text(.02,y,formula,fontsize=14,va="center")
    fig.savefig(figdir / "04_metric_formulas.png", bbox_inches="tight",pad_inches=.12);plt.close(fig)
    return sorted(figdir.glob("*.png"))


def make_report(summary, history, companion, paired, differences, ledger, channels, cfg, old_sta, training, paper_accuracy, cutoff, primary_csv):
    hist = {(r["method"], r["task"]): r for r in history}
    cells = {cell_key(c): c for c in summary["cells"]}
    full = [c for c in cells.values() if c["coverage_complete"]]
    def get(method, task, track=None, modality="not_applicable"):
        return cells.get((method,task,track or TRACK.get(method,"lp"),modality))
    def value(method, task, track=None, modality="not_applicable"):
        c=get(method,task,track,modality)
        if c and c["coverage_complete"]:return number(c["value"])
        if method=="normwear":return "已移出"
        return f"未齐 {c['completed_job_count']}/{c['planned_job_count']}" if c else "不支持"
    def ci_text(row):return f"[{row['ci95'][0]*100:.2f}, {row['ci95'][1]*100:.2f}]"
    ft = sorted([r for r in paired if r["track"]=="ft" and r["reference_method"]==r["method"]],key=lambda r:(METHODS.index(r["method"]),TASKS.index(r["task"])))
    old_rows=[[NAMES[m],*["—" if hist[m,t]["value"] is None else f"{hist[m,t]['value']*100:.2f} ± {hist[m,t]['fold_sd']*100:.2f}" for t in TASKS]] for m in METHODS]
    new_rows=[[NAMES[m]+("（重叠注记）" if m=="reve" else ""),*[value(m,t) for t in TASKS]] for m in METHODS]
    for mode,label in [("eeg","EEG 简单基线"),("fnirs","fNIRS 简单基线"),("multimodal","双模态简单基线")]:
        new_rows.append([label,*[value("signal_linear",t,"lp",mode) for t in TASKS]])
    strong=sorted([r for r in differences if abs(r["old_style_difference"])>=.02],key=lambda r:r["old_style_difference"])
    sample_reasons={"motor_imagery":"数量不变；并不代表输入坐标、通道或划分相同", "mental_arithmetic":"数量不变；60 个部分 fNIRS 支持窗保留 mask", "wg":"排除 91 个进入未验证拼接区间的 8 s 窗口", "nback":"排除 81 个进入未验证拼接区间的 8 s 窗口", "dsr":"新增 8 条漂移记录排除 2880 窗；另排除 680 个边界窗", "visual":"两个探头合并为一事件；7720 ÷ 2 = 3860", "refed_regression":"数量均为 2720；20 s 分窗，时间对齐与掩码语义仍有变化"}
    sections=[]
    sections.append(("阅读结论与范围", f"""
本报告比较三类证据：**第一轮已发布的下游结果、第二轮已完整完成的结果、作者原论文的对应或背景结果**。第一轮包括 2026-08-14 联合报告及独立的 2026-07-27 STA-Net 严格跨被试参考；第二轮截点为 **{cutoff}**。结果来自保留证据，本报告不启动训练，也不重开历史受保护预测。

第二轮源汇总含 **{summary['valid_evaluate_job_count']} 个有效评价作业、{len(full)} 个完整结果组合**；缩减后计划为 83 个结果组合。未齐折/种子的组合不给出正式均值。NormWear 已于 10 月 27 日从本轮执行范围移出，历史成绩保留，空白不能记成零分或运行失败。

关键发现：

- 两轮并非只更换随机种子：数据/时间对齐、事件组织、EEG 通道名单、标准化、选择预算和统计汇总均有变化。
- 同一批第二轮预测按第一轮口径重算后，31 个可对应分类结果中，5 个提高至少 2 点、11 个降低至少 2 点、15 个变化小于 2 点。2 点是描述性幅度标准，不是显著性阈值。
- 第二轮 CBraMod 微调在 MA、WG 上分别达到 **68.81、65.42**，较同轮 LP 提高 **14.34、10.05 点**；MI 未改善。LP 与 FT 必须分轨道报告。
- STA-Net 的 MA/WG/DSR/Visual 在重汇总后仍低于第一轮；窗口支持、dropout、选择过程同时变化，现阶段不能把差距归因于单一因素。
- 原文与项目的指标/划分经常不同。STA-Net 的 accuracy 可列出“原文—第一轮—第二轮”；大部分 foundation model 没有本项目同数据集的作者成绩。

LP：冻结编码器，只拟合线性读出。FT：用下游标签微调编码器和分类头。STA-Net/BrainFusion 为监督适配或重实现。本项目自己的新 tokenizer 尚未进入这版比较矩阵。

MI为运动想象，MA为心算，WG为词语生成，n-back为工作记忆任务；DSR按发布的Go/No-go事件评分；Visual保留原生RR/RF/FF/FR四类标签；REFED评价连续valence/arousal情绪评分。

所有分类表按 macro-F1×100 显示，差值为百分点；REFED 的一致性相关系数 CCC 无量纲。读图时区分：性能有效性、相对简单基线的收益、跨模态增量及与作者协议的可比性。
"""))
    sections.append(("两轮共同配置及统计对象", table(["项目","第一轮","第二轮"],[
        ["结果身份","六方法联合保护评价：540 作业；STA-Net 另列，非同输入预算","公共 v5 开发性重测；历史任务已被研究，不是新的确认性盲测"],
        ["数据与时钟","physiology_semantic_clean_v1；历史事件对齐","measurement v5 / motion=none；event alignment v2；连续记录离线预处理"],
        ["标准化","记录级 robust SD 坐标；头部变换只拟合相应训练集","每个内训练折拟合带 mask 的通道均值/SD；外层重训重新拟合"],
        ["外层与选择","共享严格跨被试 5 折，seed=42；登记 3 内折但选择使用 inner0；STA 单列同类结构","seed=20261016，5 外折×3 内折均参与选择；被试所有会话/记录/视频同组"],
        ["种子","联合方法下游 17/42/73；EFRM 源检查点 seed42；STA 正式参考仅 seed42","官方固定编码器 LP：确定性 seed17；FT/监督：17/42/73；EFRM 每个目标排除组 3 个独立预训练种子"],
        ["分类汇总","折内合并预测算 F1；先种子均值，再 5 折均值与 SD","先每被试算 F1、被试内种子均值，再被试等权；折 SD 与种子 SD 分列"],
        ["区间含义","主报告 ± 是折间样本 SD，不是 95% CI","2000 次被试块 bootstrap；固定已训练模型、划分和选择；不含重新训练不确定性"],
        ["完整分母","保留正式未通过数值准入的负结果","全部计划折/种子和目标覆盖齐全后才给正式组合值；失败/未完成不删除"],
        ["保护与状态","使用已发布结果报告及公开配置/审计，不读取旧保护预测","按冻结清单读取本轮公共结果；当前状态由 controller/registry 持有"]]) + "\n\n同为跨被试五折，并不意味着两轮使用相同被试分配或相同窗口。第二轮既有数据修复，也有不同实验设计；旧新差值只能作为历史比较，不能替代控制变量实验。[S1–S4]"))
    sections.append(("窗口数为什么变化：逐项对账", table(["任务","旧公开审计窗口","本轮窗口","本轮被试","原因"],[[LABELS[r['task']],r['old_public_audit_windows'],r['current_windows'],r['current_subjects'],sample_reasons[r['task']]] for r in ledger])+"""

旧列来自相应方法对五折公开 manifest 并集的 unique-sample 审计；不是把各测试折样本数相加，也不把通用 loader 的默认窗口数代替实际下游窗口。新列来自冻结 inventory。完整排除明细在 `tables/window_exclusions.csv`。[S3, S5]

**WG / n-back：**v2 时间对齐为每段保留可信支持。会话拼接出现偏移跳变时，原始文件没有精确 append 边界，两个相邻锚点之间不能擅自假定同步；8 s 请求窗口只要进入该区间即不纳入。分别减少 91 和 81 窗，仍保留全部 26 名被试。n-back 类别数量从各 234 变成 0-back 196、2-back 191、3-back 234。

**DSR：**旧 v1 已因连续时钟漂移排除 VP005；修正后的判定在实际毫秒坐标上、逐偏移段估计漂移，并先检查漂移再检查残差离散度。新增排除 VP010/012/013/015/019/022/025/026 的整条 DSR 记录，每条 360 窗，共 2880；剩余 17 名被试各有 40 个 2 s 窗进入未验证拼接区间，共 680。精确账本为 **8980 − 2880 − 680 = 5420**，被试由 25 变为 17。该数值下降不来自按模型表现挑样本。

**Visual：**第一轮一个事件的 Probe1/Probe2 分别成为样本；本轮按原生 epoch 合并真实两探头通道，产生一条预测：**7720 ÷ 2 = 3860**。fNIRS 由单探头 24 位置的 48 个 HbO/HbR 通道变为双探头 48 位置、96 通道。事件数没有因这项操作减半，也不能把两探头当作独立被试。两轮的 Visual 时钟实现仍不同。

本轮3860组探头配对的EEG事件锚点逐项相同，最大差为0 ms。对于EEG-only方法，合并减少了同事件因探头拆分而重复出现的样本；对于双模态方法，则把两个真实Hb探头放进同一条输入，不增加物理观察时长。

**MI / MA / REFED：**窗口数量不变。MI/MA 各 60 个部分 fNIRS 支持窗在本轮用 mask 保留；REFED 2720 窗中 480 个为部分末端支持窗。保留部分观测不等于填充值是真值；数量相同也不等于张量、目标时钟或评分分母完全相同。
"""))
    sections.append(("输入窗口、通道与方法配置", table(["方法","第一轮实际轨道/输入","第二轮实际轨道/输入"],[
        ["BIOT / CBraMod / REVE","官方权重冻结 LP；各数据集固定 16 EEG 通道；一般 8 s、DSR 2 s","同类官方权重 LP + FT；仍 16 EEG 通道，但重新选择较广头皮覆盖的名单"],
        ["EFRM","4 个目标数据集排除检查点，均 seed42；同步数据重训，非作者原检查点；LP","12 个 v5 新检查点（4×3种子）；LP + FT；完整目标数据集排除；REFED 使用时间 token 读出"],
        ["NormWear","官方编码器冻结 LP；EEG/HbO/HbR→65 Hz→CWT；通道特征拼接","曾计划 LP/FT，全部未启动即移出；不把等待或移出记成低分"],
        ["STA-Net","独立 PyTorch 实现；分类主要为 3 s EEG / 13 s Hb；任务特异 HPO 配置；seed42","窗口匹配监督适配：8/8 s，DSR 2/2 s，REFED 20/20 s；3种子；dropout 固定0.5"],
        ["BrainFusion","CSP/NVC+SVM/RF stacking 独立重实现；特征参数固定锚点，内部5折","外层3内折选择4个特征/正则候选；stacking内部3折；填补分别在各OOF训练分区拟合"],
        ["简单信号基线","不将后续 P0 公开探针混入第一轮正式主表","EEG五频带 log1p 能量及覆盖率；Hb均值/SD/斜率/覆盖率；双模态拼接；训练折尺度与线性读出"]])+"\n\n"+
        table(["数据集任务","旧/新 EEG 通道数","共同通道","变化含义"],[["MI / MA","16 / 16",channels[0]['common_count'],"10 个相同，6 个替换"],["WG / n-back / DSR","16 / 16",next(r['common_count'] for r in channels if r['task']=='wg'),"9 个相同，7 个替换"],["Visual","16 / 16",next(r['common_count'] for r in channels if r['task']=='visual'),"10 个相同，6 个替换"]])+"\n\n三种 EEG 方法使用共同的任务面板；完整有序名单及新增/移除通道在 `tables/channel_panels.csv`。因此不能把两轮称为“同一16通道输入”。一般采样率为 EEG 200 Hz、Hb 10 Hz；各模型再按自身确定性变换编码。DSR 的 Hb 是同步块上下文，不能称为事件特异血氧响应。[S3–S7]"))
    sections.append(("训练与选择预算：上限不等于实际训练轮数", """
第一轮 BIOT/CBraMod/REVE/NormWear 的线性头：学习率 {0.001,0.0003}×weight decay {0,0.01}，最多40轮，batch128，inverse-frequency 类别权重；选定后在外训练+公开验证成员上重拟合。第一轮 EFRM 头为 LayerNorm/Dropout/Linear，最多20轮，lr=0.001、wd=0.01、dropout=0.5，无类别权重；不是本轮确定性逻辑回归。

第二轮 LP 使用训练折特征标准化和逻辑回归，C∈{0.001,0.01,0.1,1,10,100}，无类别权重；REFED 为点式 ridge，alpha 同网格。外层评价不参与选择。普通神经选择最多200轮、至少观察50轮才允许早停、patience30、有效batch32；最终重训轮数为所选候选3个内折最佳轮数中位数向上取整。早于50轮的最佳点仍可被选中，所以最终重训可能小于50轮。

"""+table(["第二轮轨道","学习率 / weight decay 搜索","优化与调度"],[
        ["BIOT FT","{1e-5,1e-4,1e-3} × {1e-5,1e-4}","Adam，恒定lr，clip1"],
        ["CBraMod FT","{1e-5,3e-5,1e-4} × {0.01,0.05}","AdamW，cosine，clip1"],
        ["REVE FT（未完成）","{1e-5,3e-5,1e-4} × {0.001,0.01}","AdamW变体，β=(0.92,0.999)、eps1e-9，cosine，clip2"],
        ["EFRM FT（未完成）","{1e-5,3e-5,1e-4} × {0.01,0.05}","AdamW，warmup4 + cosine，clip1"],
        ["STA-Net","(1e-4,1e-4)、(3e-5,1e-4)、(3e-4,1e-4)、(1e-4,1e-3)","AdamW，恒定lr，clip1；括号为(lr,wd)"],
        ["BrainFusion","NVC对数16/32；CSP分量2/3；正则0.1/0.3；meta C0.1/1，四个固定组合","候选含结构及正则变化；SVM/RF与元分类器按组外预测拟合"]])+"""

固定官方骨干的FT/监督方法，seed42/73复用seed17的内层选择，再独立重训；EFRM预训练种子对应不同骨干，分别选择。cosine最终重训回放原200轮调度的前缀，不压缩到最终轮数。

EFRM两轮预训练均为目标数据集整体排除，Stage A选择、Stage B新初值重训；上限100、最早20、patience15、有效batch32、lr1e-4、wd0.01、Adam β=(0.9,0.95)。但第一轮为v1的平衡数据集/动态裁窗采样，第二轮为已冻结无标签8秒源窗逐轮遍历；检查点、采样及源划分种子不同。不能沿用第一轮检查点解释本轮。

第二轮DSR原生2秒窗不扩成8秒预训练样本；REFED只在允许的20秒父窗支持内取8秒源片段，保留来源和未使用区间，整个目标数据集仍排除。

全部已完成作业的实际重训轮数、所选候选及优化参数见 `tables/training_actual.csv`；精确配置路径见 `configuration_details.json`。[S2–S4, S7]
"""))
    sections.append(("第一轮完整历史结果", "分类值为 macro-F1×100；± 为种子先在折内平均后的五折 SD。STA-Net 为单seed42的独立参考，观察窗口不同。\n\n"+table(["方法",*[LABELS[t] for t in TASKS]],old_rows)+"""

REFED：EFRM CCC **0.0783 ± 0.0104**，STA-Net **0.081 ± 0.048**；其余方法未支持该回归任务。不能用分类表的百分比读法解释 CCC。

联合报告42个注册组合中，22个TABLE_READY_WITH_NOTE、12个REJECTED_VALUE、2个REVE重叠轨道、6个UNSUPPORTED。REJECTED_VALUE是实际观察到但未通过数值准入的负结果，不是运行失败；本报告仍完整呈现。STA-Net另列，不能把7种方法压成不分模态、输入预算的总排名。

NormWear的历史n-back表值最高，MA/WG有一定竞争力；其本轮移出不构成“重训失败”证据。REVE的MI/MA与预训练语料有重叠，必须保留注记。[S1, S2]
"""))
    sections.append(("第二轮完整同轨道结果与简单基线", "同轨道包括官方/重训骨干LP，以及STA-Net/BrainFusion监督适配。表中缺失有明确原因；尚未齐全的结果不以成功子集均值替代。\n\n"+table(["方法",*[LABELS[t] for t in TASKS]],new_rows)+"""

主指标为每被试计算后等权汇总。每个完整组合的95%区间、折间SD、种子SD、独立被试数、零基线与计划分母均见 `tables/current_cells.csv`。完整结果和未完成组合使用同一冻结任务清单，NormWear移除记录仍在原运行目录保留。

当前简单基线不等于性能上界。双模态模型优于EEG单模态，也不自动说明融合收益；需要同时比较对应fNIRS和双模态简单基线。尤其EFRM MA 61.83虽高于双模态简单基线57.41，但与fNIRS 61.21很接近，不能据此确认EEG额外贡献。[S3]
"""))
    sections.append(("第二轮微调结果：与同轮 LP 配对", table(["方法／任务","LP","FT","FT−LP","95%配对区间","被试数"],[[NAMES[r['method']]+" / "+LABELS[r['task']],value(r['method'],r['task'],"lp"),value(r['method'],r['task'],"ft"),f"{r['difference']*100:+.2f}",ci_text(r),r['subject_count']] for r in ft])+"""

![同轮微调增益](figures/02_finetuning_gain.png)

每个点对应一个完整任务组合，所有FT结果均为5折×3种子；先平均同一被试内的种子结果，再对同一被试FT−LP差值做2000次bootstrap。区间固定模型/选择且未作多重比较校正，属于探索性分析，不代表重新训练分布。

CBraMod MA/WG较LP提高14.34/10.05点，探索性区间分别为[10.17,18.31]/[5.63,14.45]；MI差值为−2.17。BIOT的MA、DSR也有较大改善。可得结论是“FT轨道的收益依赖任务”；不能把所有收益归于某一个编码器参数变化，因为LP与FT训练器和读出拟合方式也不同。[S3]
"""))
    sections.append(("两轮差异：先分离汇总口径", """
旧口径是在每个外折内合并样本算F1，再平均种子与折；新口径先逐被试算F1，再逐被试平均种子、最后被试等权。F1非线性，同一批预测的两种汇总可能不同。

本报告只对第二轮已完成预测作旧式重汇总。图B减去第一轮已发布均值；它消除了汇总方法这一算术项，剩余差值仍混合数据、通道、划分与训练变化。第二轮正式主指标没有被替换。

![原始差值与重汇总差值](figures/01_round_differences.png)

格子是“方法×任务”，蓝色为提高、红色为降低；同一色标±10个百分点，格内给出数字。灰色表示未齐、移出或不支持；REVE MI/MA仍有预训练语料重叠。颜色不是显著性标记。
"""))
    sections.append(("差异较大的组合及简要解释", table(["方法／任务","第一轮","本轮主值","本轮旧式汇总","旧式差值"],[[NAMES[r['method']]+" / "+LABELS[r['task']],number(r['first_value']),number(r['current_value']),number(r['current_old_style']),f"{r['old_style_difference']*100:+.2f}"] for r in strong])+"""

以上筛选仅使用|重汇总差值|≥2个百分点，展示全部满足条件的组合。完整31个分类配对与原始差值见 `tables/round_differences.csv`；没有旧新同被试预测配对，故没有旧新显著性或因果结论。

- **汇总影响很大：**BrainFusion MI直接比较54.93→52.68看似下降，但第二轮旧式汇总56.32；BIOT n-back直接下降5.73点，重汇总后只低1.32点。
- **STA-Net仍有较大负差：**MA/WG/DSR/Visual重汇总后分别低6.27/4.22/4.05/4.13点。输入时间支持改变、DSR时延候选退化、旧任务特异dropout改为0.5以及优化/选择变化，都是具体待检验因素。当前证据无法分配每项原因的贡献。
- **EEG通道和分母是共同混杂：**MI/MA替换6个通道；Simultaneous任务替换7个；DSR被试由25降到17。不能将CBraMod MA等剩余差值全部解释成骨干退化。
- **改善不等于足够有效：**EFRM MI旧式提高3.70点，但第二轮主值50.31仍接近零基线49.79，并低于简单双模态57.83。REVE MA旧式提高4.13点，但属于语料重叠轨道。
- **训练方式变化可以反转结论：**CBraMod MA第一轮LP61.75，本轮LP54.47，本轮FT68.81；第一轮到FT的直接变化不能称为“同一方法配置重跑的改善”。
"""))
    regression=[r for r in primary_csv if r['task']=='refed_regression' and r['value']]
    reg_rows=[]
    for r in regression:
        name=NAMES[r['method']]+(' / '+r['baseline_modality'] if r['method']=='signal_linear' else '')
        reg_rows.append([name,f"{float(r['value']):.4f}",f"[{float(r['ci_lower']):.4f}, {float(r['ci_upper']):.4f}]",f"{float(r['mae']):.2f}",f"{float(r['rmse']):.2f}"])
    sections.append(("REFED：一致性与绝对误差必须分别判断",table(["第二轮方法","CCC","95%区间","MAE","RMSE"],reg_rows)+"""

相同外训练分区拟合的两维训练均值基线：**MAE 45.65、RMSE 51.01**。STA-Net CCC为0.0551，高于零但误差更大；这支持有限一致性，不支持准确的连续评分恢复。三种简单基线同样没有降低误差。第一轮STA-Net CCC为0.081，EFRM为0.0783；本轮EFRM尚未完整完成，不能用缺失或成功子集作新旧比较。

![指标公式](figures/04_metric_formulas.png)

分类F1逐类计算并对固定类别集取均值，分母为0时该类记0；TP/FP/FN分别为真阳性、假阳性、假阴性计数。Accuracy是正确标签比例；balanced accuracy（BA，平衡准确率）是各真实类别召回率均值，不能与macro-F1互换。

REFED使用有效target mask内的原生valence/arousal评分。对每个记录、每个坐标在其有效点上计算CCC/MAE/RMSE，先坐标平均、再记录平均、再被试平均，最后保持种子在被试块内汇总。CCC无量纲；MAE/RMSE为原生标签评分单位，不是百分比或训练SD单位。上述训练均值只从外训练数据拟合，不使用测试标签均值。

报告未对第一轮CCC实施旧新统一记录级重算，故0.081→0.0551仅为历史原始数值对照；没有旧新配对区间。[S1–S3]
"""))
    sections.append(("作者原论文：按原指标、原任务列出","BA为平衡准确率，AUROC为ROC曲线下面积；weighted-F1按类别支持加权，macro-F1对类别等权。以下保留原文指标，不跨指标相减。\n\n"+table(["方法","作者报告的代表性成绩","与本项目的可比边界"],[
        ["BIOT [P1]","PREST：TUAB BA 79.07%；TUEV BA52.07%、weighted-F1 73.81%（Table5/6）","临床异常/事件数据，与本项目六分类任务不同；weighted-F1不是macro-F1；没有本项目同数据的作者目标值"],
        ["CBraMod [P2]","微调：PhysioNet四类MI BA64.17%；SHU二类MI BA63.70%；Zyma心算 BA72.56%","本项目MI/MA是Shin数据，任务同名不等于同数据；当前LP与原文FT也不同"],
        ["REVE [P3]","Base：PhysioNet MI FT BA64.80%，池化LP约53.7%；Zyma心算FT76.60%、池化LP72.5%","必须区分Base/Large及LP/FT；本项目MI/MA还涉及Shin2017A预训练语料重叠"],
        ["NormWear [P4]","作者预印本v2全样本LP：WESAD AUROC76.06%、DriverFatigue74.29%、GAMEEMO54.94%","原训练不含fNIRS；本项目为EEG–fNIRS适配；AUROC不能减macro-F1；本轮已移出"],
        ["EFRM [P5]","另一套8人fNIRS心算：2/4/6/8-shot accuracy90/81/85/94%（Table3）","本项目为29人Shin多模态心算；原MI数据用于预训练，没有对应下游MI结果；本项目无官方权重而重新训练"],
        ["STA-Net [P6]","被试特异accuracy：MI69.65%、MA85.14%、WG79.03%","项目使用跨被试及不同输入支持；三个任务可同指标展示，仍非同协议复制"],
        ["BrainFusion [P7]","被试内十折MI accuracy95.5%；EEG-CSP57.0%、HbO90.6%、HbR90.1%、NVC93.4%","项目使用跨被试、统一处理和独立完整流程重实现；不能把95.5当作当前协议应达到的基线"]])+"""

这里保留作者版本及表/节定位，避免把第三方后续论文对某模型的数字冒充该模型作者原始成绩。完整机器可读来源与限制在 `paper_reference_values.json`。STA-Net本次出版社全文请求返回403，原文结论限于保留的出版社摘要/预览及官方源码核验记录；本报告不补写未核实的原文实验细节。

本项目“复现/适配”的层级应明确：BIOT/CBraMod/REVE/NormWear使用官方预训练资源，但输入及读出为项目适配；EFRM是同步数据上的目标排除重训；STA-Net为PyTorch独立实现与跨任务适配；BrainFusion是独立重实现。结果差距同时包含源任务和项目协议差异。

EFRM原文预训练还包含868小时EEG-only、364小时fNIRS-only及15.5小时配对数据；本项目是不同的同步配对数据重训。原文8人心算94% accuracy与本项目29人MA不能直接作数值复现差。[P5]
"""))
    sections.append(("与原文同指标对照：STA-Net 与 BrainFusion",table(["方法／任务","作者accuracy","第一轮accuracy","第二轮accuracy","第二轮−作者"],[[NAMES[r['method']]+" / "+LABELS[r['task']],number(r['paper_accuracy']),number(r['first_accuracy']) if r['first_accuracy'] is not None else '未提供',number(r['current_accuracy']),f"{(r['current_accuracy']-r['paper_accuracy'])*100:+.2f}"] for r in paper_accuracy])+"""

![原文同指标背景比较](figures/03_paper_accuracy_context.png)

第一轮STA-Net accuracy来自其已发布`paper_table.md`，不是由macro-F1反推。BrainFusion第一轮已发布主表没有accuracy，本报告不重开旧保护预测补算。图中每组为一个任务，灰/蓝/棕分别表示第一轮、第二轮、作者结果；第一轮仍是旧汇总，第二轮为被试等权，原文为不同评价协议。

STA-Net MI与作者的accuracy差距从−13.15点缩小为−9.04点；MA从−21.82扩大为−27.71点；WG从−16.76扩大为−20.21点。数字说明项目条件下仍有较大差距，不能单独归因于跨被试难度。第一轮另存的sample-random心算accuracy为68.97%，仍低于作者85.14%；该分法也不是作者的被试特异协议，且未隔离记录/重叠窗口，所以只能作为边界说明。

BrainFusion第二轮MI accuracy56.97%，与作者95.5%相差38.53点。原文的高分伴随Hb单模态约90%的结果，应优先审查血流时间支持、被试特异信息、预处理和CSP/AutoML/stacking完整流程，而不能只增加训练预算。当前没有匹配作者条件的端到端重现证据，也没有据此否定作者结果的依据。[S2, P6, P7]
"""))
    sta_rows=[]
    for task in ALL_TASKS:
        old=old_sta[task]['training'];model=old_sta[task]['model']
        sta_rows.append([LABELS[task],f"{old['lr']:.3g}",f"{old['weight_decay']:.3g}",f"{model['dropout']:.3f}",old['grad_clip_norm']])
    epoch_rows=[]
    groups=defaultdict(list)
    for r in training:
        if r['refit_epochs'] is not None:groups[(r['method'],r['task'])].append(r['refit_epochs'])
    for (method,task),epochs in sorted(groups.items(),key=lambda kv:(METHODS.index(kv[0][0]),ALL_TASKS.index(kv[0][1]))):
        epoch_rows.append([NAMES[method],LABELS[task],len(epochs),f"{np.median(epochs):g}",f"{min(epochs)}–{max(epochs)}"])
    sections.append(("STA-Net 历史配置细节与解释边界",table(["旧任务","学习率","weight decay","dropout","clip"],sta_rows)+"""

旧正式协议最大轮数为300，使用ReduceLROnPlateau：8次无改善后lr×0.5，最低1e-6；至少40轮后、30次无改善才早停。上述为任务特异预选HPO配置；保留报告说明70个训练（严格跨被试和sample-random合计）都因验证收敛停止，而非触及上限。不能把基础runtime配置中的100误当正式任务最大轮数。

新STA-Net结构仍为embedding256、attention heads10、key dim256，但dropout为0.5、lr/wd改为四候选，普通选择上限200、最早50、patience30、恒定lr，最终重训由三个内折选择。分类输入也从3s EEG/13s Hb变为8s/8s；8s Hb提供6个3s片段，DSR只有一个2s片段，没有多个时延候选可选。

因此，“STA-Net下降由Hb窗口缩短造成”目前只能是机制假设。dropout、优化、通道及分母变化都是同一轮的共同变化，需控制变量才能判断。报告不为解释差距而修改正在运行的协议。[S2, S4, S7]
"""))
    sections.append(("实际最终重训轮数",table(["方法","任务","完整拟合数","轮数中位数","范围"],epoch_rows)+"""

该表从每个已完成作业的`training.epochs_completed`读取，未读检查点权重；只展示分类组合，REFED训练细节仍在其原结果记录。候选搜索、最后重训与最大预算是三个不同对象。长时间运行或更多epoch并不自动意味着更好的外层泛化。

`tables/training_actual.csv`还提供逐折/逐种子的所选候选、实际优化器和来源路径。模型尚未完成的组合不计入此表。[S3]
"""))
    sections.append(("综合分析与尚未解决的问题", """
**已核实的变化。** 旧新数据/时钟版本、窗口计数及事件单位、EEG通道名单、训练与特征标准化、内层选择、下游读出、种子和汇总层级均不同。这些是报告已对账的事实。

**有支持的局部结论。** CBraMod的MA/WG及BIOT的MA/DSR在本轮FT轨道取得收益；MI仍是弱项。REVE DSR LP约63.95，高于当前EEG简单基线53.62；该任务值得保留作为表征有用性的正例。EFRM MA不应仅凭超过双模态简单基线就宣称融合收益，因为fNIRS简单基线已达61.21。

**负结果仍须保留。** 当前完整Visual方法的主macro-F1均未超过零基线24.28；STA-Net Visual accuracy34.58但balanced accuracy25.52、macro-F1 19.10，说明accuracy受类别分布/预测偏置影响。REFED的正CCC没有转化为相对训练均值的误差改善。

**仍是假设的原因。** 早期血流信息不足、冻结表示不匹配、平均池化丢失空间结构、类别加权变化、模型更容易保留被试特征，都不能仅凭当前成绩确认为主要原因。历史EFRM近rank-one诊断属于旧检查点，不能套用到新EFRM或NormWear。

**后续解释优先级。** 保持当前冻结比较继续执行；如另行做机制研究，应固定数据和划分，分别检验通道名单、输入标准化/类别权重、读出方式、STA窗口和dropout。原文复现需另建匹配作者数据及划分的对照，不能通过追逐旧保护分数来调参。

本报告不维护实时队列；进度以原运行的controller和项目registry为准。图、表、附表都是指定截点证据的可复核视图，不能覆盖单个作业的失败或后来完成的结果。
"""))
    return sections


def render_pdf(text, path, archive):
    body=markdown.markdown(text,extensions=['tables'])
    css="""body {font-family:sans-serif;font-size:10pt;line-height:1.45;color:#172033;}
    h1 {font-size:21pt;color:#173d62;} h2 {font-size:16pt;color:#173d62;page-break-before:always;}
    h2:first-of-type {page-break-before:auto;} h3 {font-size:12pt;color:#173d62;}
    p {margin:7pt 0;} table {border-collapse:collapse;width:100%;font-size:8.6pt;margin:8pt 0;}
    th {background-color:#e8eef4;} td,th {border:0.5pt solid #a7b2bf;padding:4pt;}
    img {width:100%;height:auto;} a {color:#24598b;} li {margin:5pt 0;}"""
    story=fitz.Story(html='<html><body>'+body+'</body></html>',user_css=css,archive=fitz.Archive(str(archive)))
    media=fitz.paper_rect('a4');area=fitz.Rect(38,38,media.width-38,media.height-42)
    def rectfn(rect_num,filled):return media,area,None
    pdf=story.write_with_links(rectfn)
    for i,page in enumerate(pdf):page.insert_text((media.width/2-14,media.height-23),f'{i+1} / {len(pdf)}',fontsize=8,color=(.4,.4,.4))
    pdf.set_metadata({'title':'两轮下游任务性能、配置与原论文结果比较','subject':'Dated evidence report; bitmap figures and searchable text','author':'Project evidence review'})
    pdf.save(path,garbage=4,deflate=True)
    images={image[0] for page in pdf for image in page.get_images()}
    image_dpi=[]
    for page in pdf:
        for im in page.get_images():
            for rect in page.get_image_rects(im[0]):
                if rect.width: image_dpi.append(im[2]/(rect.width/72))
    if len(images)<4 or not all('�' not in page.get_text() for page in pdf):
        raise ValueError('PDF images or searchable glyph validation failed')
    if min(image_dpi)<190:raise ValueError('figure resolution below report requirement')
    check=dict(pages=len(pdf),unique_image_objects=len(images),minimum_figure_dpi=min(image_dpi),
               searchable_characters=sum(len(page.get_text()) for page in pdf))
    preview=path.parent/'preview'/path.stem.lower();preview.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(pdf):page.get_pixmap(matrix=fitz.Matrix(1.05,1.05)).save(preview/f'page_{i+1:02d}.png')
    pdf.close()
    return check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--paper-values',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise ValueError('dated exports require a fresh versioned output directory')
    summary=read(args.summary);schedule=read(summary['schedule_path']);run=Path(schedule['run_root'])
    if summary['schema']!='downstream_rebench_aggregate_v1' or summary['scope'] is None:
        raise ValueError('expected retained downstream rebenchmark summary')
    inventory=read(schedule['inventory_path'])
    historical_path=ROOT/'docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md'
    history=history_table(historical_path);lookup={(r['method'],r['task']):r for r in history}
    companion,paired,training,inputs=companions(summary,schedule,inventory)
    ledger,exclusions,channels=window_ledger(inventory)
    differences=[]
    cells={cell_key(c):c for c in summary['cells']}
    for r in companion:
        prior=lookup.get((r['method'],r['task']))
        if prior is None or r['track']!=prior['track'] or prior['value'] is None:continue
        differences.append(dict(method=r['method'],task=r['task'],track=r['track'],first_value=prior['value'],
                                first_fold_sd=prior['fold_sd'],current_value=r['primary_macro_f1'],
                                current_old_style=r['current_fold_pooled_f1'],
                                raw_difference=r['primary_macro_f1']-prior['value'],
                                old_style_difference=r['current_fold_pooled_f1']-prior['value'],
                                aggregation_effect=r['primary_macro_f1']-r['current_fold_pooled_f1']))
    expected_ft={(m,t) for m,tasks in [('biot',TASKS),('cbramod',TASKS[:3])] for t in tasks}
    actual_ft={(r['method'],r['task']) for r in companion if r['track']=='ft'}
    if len(differences)!=31 or actual_ft!=expected_ft:
        raise ValueError('this dated narrative requires the reviewed 31 historical pairs and nine complete FT cells; review a new export version for a later snapshot')
    sta_root=ROOT/'comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1'
    sta_freeze=read(sta_root/'protocol_freeze_manifest.json')
    old_sta={t:yaml.safe_load(Path(v['config']).read_text()) for t,v in sta_freeze['configs'].items()}
    sta_published=sta_root/'aggregate/paper_table.md'
    accuracy_section=sta_published.read_text().split('## Classification accuracy')[1].split('## Accuracy comparison')[0]
    first_accuracy={}
    for line in accuracy_section.splitlines():
        for task,prefix in [('motor_imagery','Motor imagery (MI)'),('mental_arithmetic','Mental arithmetic (MA)'),('wg','Word generation (WG)')]:
            if line.startswith('| '+prefix+' |'):first_accuracy[task]=float(line.split('|')[2].strip().split('±')[0])/100
    author_acc={('sta_net','motor_imagery'):.6965,('sta_net','mental_arithmetic'):.8514,('sta_net','wg'):.7903,('brainfusion','motor_imagery'):.955}
    paper_accuracy=[]
    for (method,task),author_value in author_acc.items():
        current=next(r for r in companion if r['method']==method and r['task']==task and r['track']=='supervised')
        paper_accuracy.append(dict(method=method,task=task,paper_accuracy=author_value,
                                   first_accuracy=first_accuracy.get(task) if method=='sta_net' else None,
                                   current_accuracy=current['accuracy'],current_ci95=current['accuracy_ci95'],
                                   comparability='same metric, different protocol; descriptive context only'))
    cfg_path=Path(schedule['config_path']);cfg=yaml.safe_load(cfg_path.read_text())
    with args.summary.with_name('cells.csv').open() as f:primary_csv=list(csv.DictReader(f))
    cutoff=datetime.fromtimestamp(args.summary.stat().st_mtime,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M UTC+08:00')
    sections=make_report(summary,history,companion,paired,differences,ledger,channels,cfg,old_sta,training,paper_accuracy,cutoff,primary_csv)
    out.mkdir(parents=True);(out/'tables').mkdir()
    csv_write(out/'tables/first_round.csv',history);csv_write(out/'tables/current_cells.csv',primary_csv)
    csv_write(out/'tables/round_differences.csv',differences);csv_write(out/'tables/metric_companions.csv',companion)
    csv_write(out/'tables/paired_current.csv',paired);csv_write(out/'tables/training_actual.csv',training)
    csv_write(out/'tables/window_count_ledger.csv',ledger);csv_write(out/'tables/window_exclusions.csv',exclusions)
    csv_write(out/'tables/channel_panels.csv',channels);csv_write(out/'tables/paper_accuracy_context.csv',paper_accuracy)
    paper_values=read(args.paper_values);dump(out/'paper_reference_values.json',paper_values)
    details={'second_round':cfg,'second_round_manifest':summary['schedule_path'],'old_sta_formal':{k:sta_freeze[k] for k in ['seed','maximum_epochs','selection_rule','protocols']},'old_sta_task_configs':old_sta}
    for name,path in [('old_biot','BIOT/configs/public_development_v2.yaml'),('old_cbramod','CBraMod/configs/public_development_v2.yaml'),('old_reve','REVE/configs/public_development_v2.yaml'),('old_normwear','NormWear/configs/public_development_v2.yaml'),('old_efrm','EFRM-PyTorch/configs/downstream_public_v2.yaml'),('old_efrm_pretraining','EFRM-PyTorch/configs/pretrain_sync.yaml'),('old_efrm_protocol','EFRM-PyTorch/sources/lodo_full_target_fivefold_v2.yaml'),('old_brainfusion','BrainFusion-NVC-CSP-Stacking/configs/public_development_v2.yaml')]:
        details[name]={'source':str(ROOT/'comparative_methods'/path),'config':yaml.safe_load((ROOT/'comparative_methods'/path).read_text())}
    dump(out/'configuration_details.json',details)
    figures=make_figures(out,differences,paired,paper_accuracy)
    sources=[('S1','第一轮联合发布报告',historical_path),('S2','第一轮STA-Net发布表及配置冻结',sta_published),
             ('S3','第二轮本次结果事实源',args.summary.resolve()),('S4','第二轮冻结执行合同',cfg_path),
             ('S5','本轮窗口库存与排除原因',Path(schedule['inventory_path'])),('S6','数据合同：时间对齐v2',ROOT/'docs/DATA_CONTRACT.md'),
             ('S7','本轮冻结工作器',Path(schedule['source_root'])/'comparative_methods/downstream_rebench_worker_v1.py')]
    reference_lines=[f'- [{tag}] [{label}]({path})' for tag,label,path in sources]
    order=['BIOT','CBraMod','REVE','NormWear','EFRM','STA-Net','BrainFusion']
    for i,name in enumerate(order,1):
        paper=next(p for p in paper_values if p['method']==name)
        reference_lines.append(f"- [P{i}] [{name} 作者原文]({paper['source']})；定位：{paper['locator']}。")
    reference_lines += ['','配套文件：`configuration_details.json`保存两轮配置摘录及源路径；`paper_reference_values.json`保留文献数值及可比性；`tables/`含全量结果、置信区间、差值、窗口排除和通道名单。','',
                       '旧数值继续以已发布历史表为准，新数值以第二轮源汇总和逐作业记录为准。本目录是带日期的交流材料，不是新的实验状态源。新完成结果应使用新版本导出，不覆盖本次快照。']
    sections.append(('证据入口与文件说明','\n'.join(reference_lines)))
    text='# 两轮下游任务性能、配置与原论文结果比较\n\n结果截点：'+cutoff+'。\n\n'+'\n\n'.join('## '+title+'\n\n'+body.strip() for title,body in sections)+'\n'
    text += '\n'+'\n'.join(f'[{tag}]: {path}' for tag,label,path in sources)+'\n'
    text += '\n'.join(f"[P{i}]: {next(p['source'] for p in paper_values if p['method']==name)}" for i,name in enumerate(order,1))+'\n'
    (out/'REPORT.md').write_text(text)
    pdf_check=render_pdf(text,out/'REPORT.pdf',out)
    atlas='# 下游比较图册\n\n与正文共享同一数值来源与位图。\n\n'+'\n\n'.join(f'## 图 {i+1}\n\n![图{i+1}](figures/{p.name})\n\n'+['原始差值与重汇总差值。灰格不是零分，REVE MI/MA保留语料重叠注记。','FT减同轮LP；2000次被试块bootstrap，固定训练结果、未多重校正。','作者、第一轮和第二轮accuracy；不同协议，不能作同条件因果比较。','指标定义；分类按被试汇总，REFED先坐标、再记录、再被试。'][i] for i,p in enumerate(figures))
    atlas_check=render_pdf(atlas,out/'FIGURES.pdf',out)
    manifest={'schema':'downstream_comparison_report_v1','created_at':datetime.now(timezone.utc).isoformat(),'result_cutoff':cutoff,
              'owning_summary':str(args.summary.resolve()),'owning_schedule':summary['schedule_path'],
              'render_command':[sys.executable,str(Path(__file__).resolve()),'--summary',str(args.summary.resolve()),'--paper-values',str(args.paper_values.resolve()),'--output',str(out)],
              'scope':'published historical results and public-v5 retained predictions; no old protected payload read; no fits',
              'complete_cells':sum(c['coverage_complete'] for c in summary['cells']),'valid_evaluations':summary['valid_evaluate_job_count'],
              'historical_classification_pairs':len(differences),'primary_recalculation_matches':True,
              'window_ledger_checks':'passed: WG91, nback81, DSR2880+680, Visual paired probes; REFED unchanged2720',
              'sources':[{'id':tag,'label':label,'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for tag,label,path in sources],
              'current_prediction_inputs':inputs,'report_pdf':pdf_check,'figure_atlas_pdf':atlas_check}
    dump(out/'source_manifest.json',manifest)
    shutil.copyfile(__file__,out/'renderer_source.py')
    print(json.dumps({'output':str(out),'report_pdf':pdf_check,'complete_cells':manifest['complete_cells'],'primary_checks':'passed'},ensure_ascii=False))


if __name__=='__main__':main()
