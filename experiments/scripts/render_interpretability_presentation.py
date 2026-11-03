#!/usr/bin/env python3
"""Build the dated interpretability presentation from explicit retained exports.

Report-only: no dataset loaders, model fitting, new statistical inference or
protected evaluation. Original figures and evidence remain at their owning paths.
Run with the repository Python and a fresh, versioned --output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/reports/interpretability_analysis"
RUNS = ROOT / "experiments/runs/physiology_semantic_tokenizer"
OV = BASE / "20261028_overview_v1"
PARAM = BASE / "staging/03_tau_beta_task_parameters_20261029_v2"
EXPANDED = BASE / "staging/02_parameter_individuality_expanded_20261028_v3"
MISMATCH = BASE / "20261029_fixed_tak_physiology_mismatch_v3"
VOLTAGE = RUNS / "dual_component_tokenizer/20261029_voltage_v2"
PILOT = RUNS / "dual_component_tokenizer/20261028_v3/analysis/summary.json"
DOWN = BASE / "20261029_downstream_best_available_v2"
REFED = BASE / "20261029_refed_progress_v1"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
INK, BLUE, MUTED, ORANGE = "203449", "236A94", "65717E", "B96322"
VISUAL = "Visual 的 EEG／联合拟合沿用已发现错误的通道映射，仅作历史工程诊断。"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def setup_plots():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
                         "font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


class Deck:
    def __init__(self, output):
        self.output = output
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333333), Inches(7.5)
        self.pages = []
        self.assets = {}
        self.formulas = []
        self.text_boxes = []
        for name in ("figures", "formulas", "previews"):
            (output / name).mkdir(exist_ok=True)

    def text(self, slide, text, x, y, w, h, size=19, bold=False, color=INK, align=None):
        font = ImageFont.truetype(str(FONT), round(size * 4))
        lines = 0
        for paragraph in text.split("\n"):
            lines += 1
            width = 0
            for character in paragraph:
                advance = font.getlength(character) / 4
                if width + advance > (w - .06) * 72:
                    lines += 1
                    width = 0
                width += advance
        needed = lines * size * 1.15 + max(0, len(text.split("\n")) - 1) * 4
        if needed > (h - .04) * 72 + 1:
            raise ValueError(f"Text too dense ({needed:.1f}>{h*72:.1f}): {text}")
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(.02)
        tf.margin_top = tf.margin_bottom = Inches(.01)
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = "Noto Sans CJK SC"
            p.font.size, p.font.bold = Pt(size), bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.space_after, p.line_spacing = Pt(4), 1.15
            if align is not None:
                p.alignment = align
            for run in p.runs:
                ea = OxmlElement("a:ea")
                ea.set("typeface", "Noto Sans CJK SC")
                run._r.get_or_add_rPr().append(ea)
        self.text_boxes.append({"slide": len(self.prs.slides), "text": text,
                                "box_inches": [x, y, w, h], "font_pt": size})
        return shape

    def base(self, title, subtitle, source, section="结果", notes=""):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.text(slide, section, .46, .14, 11.9, .25, 10.5, color=BLUE)
        self.text(slide, title, .45, .48, 12.42, .58, 26, True)
        self.text(slide, subtitle, .47, 1.12, 12.35, .42, 13.5, color=MUTED)
        sources = [Path(p) for p in source]
        for path in sources:
            if not path.exists():
                raise FileNotFoundError(path)
        label = "证据：" + "；".join(p.parent.name + "/" + p.name for p in sources[:2])
        if len(label) > 158:
            label = "证据：" + sources[0].parent.name + "（逐页来源见备注及 source_manifest.json）"
        footer = self.text(slide, label, .47, 7.08, 11.75, .24, 8.3, color=MUTED)
        if sources:
            footer.text_frame.paragraphs[0].runs[0].hyperlink.address = sources[0].as_uri()
        self.text(slide, f"{len(self.prs.slides):02d}", 12.33, 7.04, .5, .3, 11, color=MUTED)
        slide.notes_slide.notes_text_frame.text = notes + "\n\n来源（原证据为事实源）：\n" + "\n".join(str(p) for p in sources)
        self.pages.append({"slide": len(self.prs.slides), "section": section,
                           "title": title, "sources": [relative(p) for p in sources],
                           "figures": [], "notes": notes})
        return slide

    def picture(self, slide, path, x=.5, y=1.64, w=12.3, h=4.25):
        path = Path(path).resolve()
        if path not in self.assets:
            destination = self.output / "figures" / f"{len(self.assets)+1:03d}_{path.name}"
            shutil.copyfile(path, destination)
            self.assets[path] = destination
        local = self.assets[path]
        with Image.open(local) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        ww, hh = iw * scale, ih * scale
        slide.shapes.add_picture(str(local), Inches(x + (w-ww)/2), Inches(y + (h-hh)/2),
                                 width=Inches(ww), height=Inches(hh))
        self.pages[-1]["figures"].append({"source": relative(path),
            "embedded": str(local.relative_to(self.output)), "pixels": [iw, ih],
            "display_inches": [ww, hh], "effective_dpi": min(iw/ww, ih/hh)})

    def figure(self, title, subtitle, path, lines, source, section="结果", notes=""):
        slide = self.base(title, subtitle, source, section, notes)
        with Image.open(path) as im:
            ratio = im.width / im.height
        if ratio < 1.8:
            self.picture(slide, path, .48, 1.67, 8.25, 5.12)
            for index, line in enumerate(lines):
                self.text(slide, line, 9.0, 2.0 + 2.35 * index, 3.8, 2.22, 19)
        else:
            self.picture(slide, path)
            self.text(slide, "\n".join(lines), .55, 6.02, 12.2, .9, 17)
        return slide

    def table(self, slide, columns, rows, widths=None, y=1.75, height=4.3, size=17):
        nrows, ncols = len(rows)+1, len(columns)
        shape = slide.shapes.add_table(nrows, ncols, Inches(.55), Inches(y), Inches(12.2), Inches(height))
        table = shape.table
        if widths:
            for col, width in zip(table.columns, widths):
                col.width = Inches(width / sum(widths) * 12.2)
        for i, row in enumerate([columns] + rows):
            table.rows[i].height = Inches(height/nrows)
            for j, value in enumerate(row):
                cell = table.cell(i, j)
                cell.text = str(value)
                cell.margin_left = cell.margin_right = Inches(.09)
                cell.margin_top = cell.margin_bottom = Inches(.06)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor.from_string("EAF1F6" if i == 0 else ("FFFFFF" if i%2 else "F7F9FA"))
                for p in cell.text_frame.paragraphs:
                    p.font.name, p.font.size = "Noto Sans CJK SC", Pt(size)
                    p.font.bold = i == 0
                    p.font.color.rgb = RGBColor.from_string(INK)
        return table

    def equation(self, latex, name):
        stem = self.output / "formulas" / name
        tex = stem.with_suffix(".tex")
        tex.write_text("\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n"
            "\\usepackage[paperwidth=18in,paperheight=4in,margin=.2in]{geometry}\n"
            "\\pagestyle{empty}\n\\begin{document}\n\\fontsize{24}{30}\\selectfont\n\\["
            + latex + "\\]\n\\end{document}\n", encoding="utf-8")
        r = subprocess.run(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=tex.parent, capture_output=True, text=True, timeout=60)
        stem.with_suffix(".render.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"Formula failed: {tex}")
        with fitz.open(stem.with_suffix(".pdf")) as doc:
            boxes = [fitz.Rect(b[1]) for b in doc[0].get_bboxlog() if b[0] != "ignore-text"]
            clip = boxes[0]
            for box in boxes[1:]:
                clip |= box
            clip += (-8, -8, 8, 8)
            doc[0].get_pixmap(clip=clip, matrix=fitz.Matrix(300/72, 300/72), alpha=False).save(stem.with_suffix(".png"))
        self.formulas.append({"latex": latex, "source": str(tex.relative_to(self.output)),
                              "png": str(stem.with_suffix(".png").relative_to(self.output))})
        return stem.with_suffix(".png")


def architecture(output):
    """A readable overview; detailed implementation diagram is also retained."""
    fig, ax = plt.subplots(figsize=(12.2, 4.3))
    ax.set(xlim=(0, 12.2), ylim=(0, 4.3)); ax.axis("off")
    def box(x,y,w,h,text,color="#EAF1F6",edge="#236A94"):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor=edge,lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=14,linespacing=1.6,color="#203449")
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,lw=1.3,
                                    color="#7A65A3" if dashed else "#526477",
                                    linestyle="--" if dashed else "-"))
    box(.15,3.28,3.1,.8,"Tak 条件教师：联合 90 秒观测\n拟合驱动与四个初态", "#F4F0FA", "#8A73AC")
    box(4.15,3.28,3.25,.8,"冻结监督目标\n参照成分 S · 完整残差 O", "#F4F0FA", "#8A73AC")
    arrow((3.25,3.68),(4.15,3.68),True)
    for y,label in [(1.98,"EEG 电压"),(.7,"fNIRS：HbO / HbR")]:
        box(.15,y,2.25,.83,label)
        box(3.05,y,2.5,.83,"本模态两分支编码器\nS 分支 / O 分支")
        box(6.2,y,2.1,.83,"独立量化与码本\n整数 token")
        box(8.95,y,3.05,.83,"同分支硬 ID 独立解码\n成分波形 → 相加重建")
        for a,b in [((2.4,y+.42),(3.05,y+.42)),((5.55,y+.42),(6.2,y+.42)),((8.3,y+.42),(8.95,y+.42))]:arrow(a,b)
    ax.plot([7.4,12.12,12.12],[3.68,3.68,1.12],color="#7A65A3",ls="--",lw=1.3)
    arrow((12.12,2.4),(12,2.4),True);arrow((12.12,1.12),(12,1.12),True)
    ax.text(10.25,3.94,"仅训练监督",fontsize=11,color="#7A65A3",ha="center")
    ax.text(6.1,.15,"冻结四路 token → 掩码序列模型 → 任务头接口（各版本结果分别验证）",
            ha="center",va="center",fontsize=14,color="#236A94")
    fig.subplots_adjust(0,0,1,1)
    path = output/"figures/architecture_overview.png";fig.savefig(path,dpi=260);plt.close(fig)
    return path


def generated_charts(output):
    paths = {}
    voltage = read_json(VOLTAGE/"voltage_summary.json")
    def value(ds,modality,component,model,metric="NMSE"):
        rows=[r for r in voltage["metrics"] if all(r[k]==v for k,v in
              {"dataset":ds,"modality":modality,"component":component,"signal":"joint","model":model,"metric":metric}.items())]
        assert len(rows)==1
        return rows[0]["value"]
    datasets=["eeg_fnirs_single_trial","simultaneous_eeg_nirs"]
    fig,axs=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
    for ax,ds,label in zip(axs,datasets,["Single-Trial","Simultaneous"]):
        x=np.arange(3);w=.25
        for shift,model,color,label_m in [(-w,"hb_blocks_continuous","#89939B","连续学生"),
                (0,"hb_blocks","#B96322","硬 token 原解码"),(w,"selected","#236A94","硬 token 固定平滑")]:
            ys=[value(ds,"hb",c,model) for c in ["S","O","sum"]]
            ax.bar(x+shift,ys,w,color=color,label=label_m)
            for xx,yy in zip(x+shift,ys):ax.text(xx,yy+.002,f"{yy:.3f}",ha="center",fontsize=10)
        ax.set_xticks(x,["S 参照支","O 残差支","总重建"]);ax.set_title(label+" · 6 名公开测试被试")
        ax.set_ylim(0,.115);ax.set_ylabel("NMSE / 冻结训练 SD 平方 · 越低越好")
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    axs[0].legend(loc="upper left",ncol=3,bbox_to_anchor=(-.03,1.3),fontsize=10)
    fig.tight_layout();paths["hb_overall"]=output/"figures/hb_overall.png";fig.savefig(paths["hb_overall"],dpi=260);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.2))
    for ax,ds,label in zip(axs,datasets,["Single-Trial","Simultaneous"]):
        models=["eeg_voltage_continuous","selected","eeg_voltage_ordinary"]
        ys=[value(ds,"eeg","sum",m) for m in models]
        ax.bar(np.arange(3),ys,color=["#89939B","#236A94","#B96322"])
        for x,y in enumerate(ys):ax.text(x,y,f"{y:.5f}",ha="center",va="bottom",fontsize=13)
        ax.set_xticks(np.arange(3),["连续学生","双分支硬 token","普通硬量化"])
        ax.set_title(label+" · 独立纵轴");ax.set_ylim(0,max(ys)*1.25)
        ax.set_ylabel("总电压 NMSE / 冻结训练 RMS 平方")
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();paths["eeg_overall"]=output/"figures/eeg_overall.png";fig.savefig(paths["eeg_overall"],dpi=260);plt.close(fig)
    pilot=read_json(PILOT)
    fig,axs=plt.subplots(1,2,figsize=(12,4.2),sharex=True)
    controls=["own_reader","frozen_matched_donor","linear_interpolation_visible_decodes"]
    for ax,ds,label in zip(axs,datasets,["Single-Trial","Simultaneous"]):
        for i,c in enumerate(controls):
            rs=[r for r in pilot["masked"]["comparisons"] if all(r[k]==v for k,v in
                dict(route="hard_student",split="test",task="time",modality="hb",component="sum",metric="NMSE",second=c,dataset=ds).items())]
            assert len(rs)==1
            r=rs[0];mean=r["mean_difference"];low,high=r["CI95"]
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt="o",color="#236A94",capsize=4)
            ax.text(.216,i,f"{mean:+.4f}",ha="right",va="center",fontsize=12)
        ax.axvline(0,color="#A0A0A0",ls="--")
        ax.set_yticks(range(3),["自身模态 reader","冻结错配供体","可见解码的线性插值"])
        ax.set_xlim(-.035,.22);ax.set_ylim(2.5,-.5);ax.set_title(label+" · 6 名被试")
        ax.set_xlabel("真实配对减对照的 NMSE · 向左有利")
    fig.tight_layout();paths["masked"]=output/"figures/masked_controls.png";fig.savefig(paths["masked"],dpi=260);plt.close(fig)
    refs=read_csv(REFED/"tables/refed_method_diagnostics.csv")
    rows=[next(r for r in refs if r['method']==m) for m in ['sta_net','efrm']] if any(r['method']=='sta_net' for r in refs) else []
    if not rows:
        rows=[next(r for r in refs if r['method']=='stanet'), next(r for r in refs if r['method']=='efrm')]
    fig,axs=plt.subplots(1,3,figsize=(12.2,4.2))
    for ax,key,label in zip(axs,['ccc','mae','rmse'],['CCC · 越高越好','MAE · 越低越好','RMSE · 越低越好']):
        ys=[float(r[key]) for r in rows]+[0 if key=='ccc' else float(rows[0]['train_mean_'+key])]
        ax.bar(range(3),ys,color=['#236A94','#B96322','#89939B'])
        for x,y in enumerate(ys):ax.text(x,y,f'{y:.4f}' if key=='ccc' else f'{y:.2f}',ha='center',va='bottom',fontsize=12)
        ax.set_xticks(range(3),['STA-Net\n15/15','EFRM LP\n10/15 阶段性','训练均值'])
        ax.set_title(label);ax.set_ylim(0,max(ys)*1.22)
        ax.set_ylabel('无量纲' if key=='ccc' else '原始情绪评分单位')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();paths['refed']=output/'figures/refed_metrics.png';fig.savefig(paths['refed'],dpi=260);plt.close(fig)
    return paths


def build(output):
    setup_plots();d=Deck(output);charts=generated_charts(output)
    arch=architecture(output)
    s=d.base("EEG–fNIRS 可解释性分析", "最终汇报 · 2026 年 10 月 29 日 · 基于目录中已准备的保留结果", [BASE/"README.md",BASE/"paper_outline.md"], "汇报")
    d.text(s,"生理参照分解、双分支离散表征\n及其证据边界",.7,2.02,11.9,1.6,32,True)
    d.text(s,"总体架构  →  可解释性结果  →  下游性能  →  论文大纲",.72,4.15,11.8,.7,23,color=BLUE)
    d.text(s,"以已有图表为主；正文呈现主要结果，补充页保留历史诊断。\n重建、参数恢复、来源归属和任务效用分别判断。",.73,5.37,11.8,1.1,20)

    s=d.base("已准备结果覆盖六条解释路径", "同一结果的旧版不重复计数；当前 Tak 与历史 inlet-balance 结果保留版本身份", [BASE/"README.md",ROOT/"experiments/RESULTS_INDEX.md"], "汇报范围")
    d.table(s,["结果组","准备素材","回答的问题"],[
        ["观测与重建","双 Hb／EEG 样例、处理对照","拟合好在哪里，尺度与形状是否一致"],
        ["参数与空间","多片段重估、任务对照、头皮图","差异能否重复，是否存在可解释的空间结构"],
        ["模型失配","总体分布、重复性、共同残差、眼电对照","未解释信号是否对应稳定或独立的来源"],
        ["离散表征","硬 token 波形、总体误差、掩码先导","压缩后保留了什么，是否依赖额外信息"],
        ["下游效用","分类、REFED 回归、30 秒心算诊断","任务收益是否超过相应基线"],
        ["历史诊断","已知真值、空间／跨模态、原型与探针","解释能力的正对照、反例和泛化边界"]
    ],[2,4,6],height=4.55,size=16)
    d.text(s,"每页注明原结果入口；完整技术报告与图册继续保留，PPT 不替代实验事实源。",.6,6.5,12,.4,16,color=MUTED)

    d.figure("总体架构：以生理参照监督两条可独立解码的分支", "教师提供目标；各模态学生读取本模态观测；四路整数 token 接入后续序列模型", arch,
        ["S 是模型参照成分，O 是完整残差；两支都保留，再分别压缩和解码。",
         "这是总体接口示意；电压版的重建结果、旧版掩码结果和外部下游比较分别报告。"],
        [ROOT/"docs/DATA_CONTRACT.md",BASE/"20261029_project_framework_v10/figure_index.json"],"模型与读图")

    s=d.base("同一观测坐标内定义目标，再分别评价分支和总信号", "EEG＝脑电；fNIRS＝功能近红外；HbO／HbR＝氧合／脱氧血红蛋白", [ROOT/"docs/DATA_CONTRACT.md",BASE/"paper_outline.md"],"模型与读图")
    eq=d.equation(r"S_m(t)=\widehat y_{\mathrm{Tak},m}(t),\qquad O_m(t)=u_m(t)-S_m(t)","component_targets")
    d.picture(s,eq,.75,1.68,11.8,.72)
    d.text(s,"代理与 Hb：u 为处理后实测，S 为完整 Tak 输出；m 表示模态。教师可见完整 90 秒。\n电压版：S 为功率参照约束下的条件修正电压；O 为原电压减修正电压。",.7,2.52,11.9,1.0,18)
    eq=d.equation(r"\mathrm{NMSE}_j=\frac{1}{|\mathcal T_j|}\sum_{t\in\mathcal T_j}\left(\frac{\widehat y_j(t)-y_j(t)}{a_j}\right)^2", "normalized_error")
    d.picture(s,eq,.8,3.75,11.7,.83)
    d.text(s,"评分支持为有效时间集合；a 为冻结训练尺度：代理／Hb 用标准差 SD，电压用均方根 RMS。\nSD 扣除均值，RMS 保留均值；相关只评价去均值形状。汇总顺序以各实验为准。",.7,4.85,11.9,1.05,18)
    d.text(s,"当前窗口可见目标的条件重建，与隐藏目标预测、生理真值恢复是不同端点。",.7,6.3,11.9,.5,19,True,color=BLUE)

    d.figure("高形状相关的样例，仍有幅度低估", "历史 Tak 4 Hz · Visual S08 / Probe1 / CH7 · 90 秒 · 48 个计划片段中 44 个数值合格",
        BASE/"staging/01_reconstruction_20261028_v1/best_shape/02_visual_cognitive_motivation__S08__prefrontal__montage0__r1.png",
        ["按两色团较弱的局部相关选图：本例约 0.97；双 Hb NRMSE 为 0.326，幅比为 0.88／0.81。",
         "左列完整可见，右列另行拟合隐藏块；"+VISUAL],
        [BASE/"staging/01_reconstruction_20261028_v1/index.html",ROOT/"docs/DATA_CONTRACT.md"],"条件重建",
        "黑线为处理后观测，蓝线 Tak，灰虚线旧自由初态，橙线旧自由加估参。三行 EEG 代理、HbO、HbR，单位训练 SD。按20秒局部去趋势相关排序，选择性个例不代表总体。")
    d.figure("训练尺度误差很小，也可能弱于零预测", "历史 Tak 4 Hz · REFED 被试 11 / video_13 / CH49 · 同一被试的窗口不作独立重复",
        BASE/"staging/01_reconstruction_20261028_v1/lowest_error/01_refed__11__prefrontal__r0.png",
        ["本例双 Hb NRMSE 为 0.1097，零预测为 0.1033；HbO／HbR 局部相关仅 0.267／0.507。",
         "低波动片段在训练尺度下误差小；应同时核对峰谷、幅度和简单基线。"],
        [BASE/"staging/01_reconstruction_20261028_v1/index.html"],"条件重建",
        "按完整评分支持的双Hb训练SD均方误差开方升序选图。左列完整可见，右列隐藏评分阴影。灰线旧自由初态、橙线旧自由加估参、蓝线Tak，黑线观测。")

    d.figure("放开转运时间与有效增益，进一步降低条件重建误差", "10 Hz Tak · 24 名被试 · 共同成功片段比较 · 灰：固定；蓝：衰减＋转运；橙：转运＋增益",PARAM/"figures/09_reconstruction.png",
        ["主分析固定→转运＋增益：Single-Trial 0.527→0.484；Simultaneous 0.831→0.740；REFED 0.299→0.281。",
         "两自由臂每段仅放开一对参数，驱动与四初态均重新拟合；"+VISUAL],
        [PARAM/"index.html",ROOT/"experiments/configs/physiology_semantic_tokenizer/shared_driver_tak_task_parameters_v1.yaml"],"参数与空间",
        "NRMSE先在0–89.8秒899点及两色团平均训练SD平方误差，支持内、被试内支持等权、被试等权后开方。270/276片段可用；固定269/270合格，两自由臂270/270。主图为原168片段范围。")
    for param,label in [("tau","转运时间"),("beta","有效驱动增益")]:
        d.figure(f"{label}：同一被试六段数据的估计仍有波动", "每点是一段独立拟合的 90 秒数据；黑横线为六点几何均值，虚线为先验参考",BASE/f"20261029_tau_beta_final_candidates_v3/figures/{param}_six_points.png",
            ["每库 6 名被试、每人 6 点；REFED 混合悲伤、中性、恐惧与放松视频，属于跨情绪分布。",
             "共同 log 先验 SD 为 0.125；图中集中程度不能单独证明个体稳定性。Visual 仅作旧映射诊断。"],
            [BASE/"20261029_tau_beta_final_candidates_v3/index.html",PARAM/"index.html"],"参数与空间",
            "转运时间单位秒；beta为冻结观测坐标中的有效神经血管驱动增益，非独立标定的生理参数。REFED六视频展示不套用原三悲伤视频的重复性统计。")
    d.figure("两组参数都尚未确证稳定的个体区分", "被试间／被试内 log 分散比：大于 1 表示中心差异较大；区间为冻结拟合下的被试 bootstrap",PARAM/"figures/03_pair_dispersion.png",
        ["两种参数组合均为 0/8 项区间完全高于 1；每库仅 6 名独立被试，不能按窗口扩大样本量。",
         "数值多起点几乎一致，跨数据重估仍有变化：优化稳定性与个体重复性需要分开判断。"],
        [PARAM/"index.html",EXPANDED/"index.html"],"参数与空间")
    d.figure("任务参数读出未通过四项主检验的多重校正", "同被试留出连续支持，以其余支持的任务质心分类；缺支持计入完整分母",PARAM/"figures/07_heldout_task_discrimination.png",
        ["转运＋增益：Single-Trial 33.3%（Holm p=0.969）；Simultaneous 40.7%（p=0.124）。",
         "衰减＋转运分别为 61.1%／35.2%；任务中心差异尚未形成可靠留出区分。"],
        [PARAM/"index.html"],"参数与空间",
        "Single-Trial36次判别，Simultaneous54次；每库6人。10000次被试内及支持序号内标签置换重算质心，4项Holm校正。VP005 DSR六段因时钟漂移排除，弃判保留。")
    d.figure("已知参数变化可以被恢复，但存在收缩和串扰", "合成配对控制 · 三个独立生成种子、六个成对样例 · 真参数只用于评分",PARAM/"figures/08_synthetic_controls.png",
        ["真实增加 50% 时，转运时间平均恢复约 33.9%，有效增益约 41.6%；不变条件伪变化接近零。",
         "转运变化引起增益约 −5.6% 串扰，增益变化引起转运约 −3.7%；实测仍缺独立标定。"],
        [PARAM/"index.html"],"参数与空间")
    for subject,coverage in [(15,"36/36"),(23,"35/36"),(25,"35/36")]:
        d.figure(f"空间分布样例：被试 {subject} 的转运时间与有效增益", "Single-Trial · 首个 90 秒片段 · 左为转运时间（秒），右为有效增益；三人共用各参数色标",BASE/f"20261029_tau_beta_topomap_examples_v1/figures/subject_{subject}.png",
            [f"本例完成 {coverage} 个位置；黑点为实际位置，叉号及局部留白保留未完成位置。蓝低红高，白色为共同中心。",
             "按停止后首片段覆盖达到 90% 选入全部三人；插值和平滑仅用于显示，不增加分辨率或证明空间重复性。"],
            [BASE/"20261029_tau_beta_topomap_examples_v1/index.html"],"参数与空间",
            "选取停止后首片段覆盖达到90%的全部三名非Visual被试，无按参数极值二次选择。白中心为三人空间几何均值的几何平均。三角网对数线性插值、正权重显示平滑。不是皮层源定位或空间重复性验证。")

    s=d.base("固定 Tak 失配：区分条件拟合与独立预测", "完整标量：473,088 个窗口×Hb 通道对；形态／眼电面板：38,268 对、103 名被试",[MISMATCH/"content.json"],"模型失配")
    d.table(s,["条件／指标","实际计算","解释范围"],[
        ["Hb-only","当前 Hb 拟合驱动与初态，再重建 Hb","条件相容性；已看到评分 Hb"],
        ["EEG-only","由 EEG 约束驱动，再前向生成 Hb","独立 Hb 预测；需要匹配基线"],
        ["联合拟合","当前 EEG 与 Hb 同时拟合","共同条件重建，不能当隐藏预测"],
        ["Hb 失配 D1","两色团训练 SD 标准化误差的 RMS","越小越相容；跨库不是绝对浓度"],
        ["残差／零 MSE 比","共同支持的模型与零预测均方误差比","小于 1 才优于零；与绝对误差并列"]
    ],[2.3,5.1,4.8],height=3.5,size=16)
    d.text(s,"固定参数，逐窗驱动与初态独立拟合；窗口和通道对不是独立被试。\n全分母包含数值未通过、无支持、异常；完整失败分类见补充页与原报告。",.62,5.53,12,1.0,18)

    d.figure("Hb 可条件拟合，跨模态读出仍有较大失配", "每条阶梯线是被试等权累积分布；越靠左表示该项误差较低，同一指标跨库同轴",MISMATCH/"figures/figure_18.png",
        ["四库 Hb 失配中位数为 0.2287–0.3949；残差／零 MSE 比为 0.0681–0.1777。",
         "这些值来自可见 Hb 的条件拟合；下排含 EEG 的 Visual 曲线只保留错误映射下工程含义。"],
        [MISMATCH/"content.json"],"模型失配",
        "D1先在窗口内按有效时点及两个色团计算训练SD标准化RMS。后续按原表通道、窗口、记录及被试聚合。上两图为纯Hb-only，可按原Hb条件保留Visual。下左为Hb条件驱动对EEG读出误差，下右为两个驱动同初态血流重放差异，均不等于独立来源。")
    d.figure("同任务跨记录的失配排序可以短期复现", "A／B 为最早／最晚原生支持；每点是被试×任务；虚线为相等线",MISMATCH/"figures/figure_50.png",
        ["Single-Trial：心算排序相关 0.788，95% 区间 [0.549, 0.919]；运动想象 0.856 [0.682, 0.935]。",
         "两任务各 29 人；重复窗口不增加独立人数。这支持短期排序一致性，尚非长期生理特质。"],
        [MISMATCH/"content.json"],"模型失配",
        "图中58个Single-Trial点对应29人两个任务。Visual纯Hb D1跨Part有效11/16人，4人无重复元数据，1人端点无共同支持；同次访问不能当跨日可靠性。区间按完整被试bootstrap。")
    d.figure("同被试的任务失配差异依赖评分坐标", "Simultaneous：行是任务 A、列是任务 B；颜色为 B 减 A，点标记原报告校正后的比较",MISMATCH/"figures/figure_43.png",
        ["词生成减 n-back：Hb 失配增加 0.0454，BH q=0.0044；残差／零 MSE 比差 −0.0009，q=0.971。",
         "绝对训练尺度误差与相对零基线比例结论不同；固定任务顺序、疲劳与记录场景仍可能混杂。"],
        [MISMATCH/"content.json"],"模型失配")
    s=d.base("HbO 与 HbR 的残差常同向，但不能据此命名来源", "预定三通道面板；每条曲线为有效被试的累积分布，四库分别为 29／32／26／16 人",[MISMATCH/"content.json"],"模型失配")
    d.picture(s,MISMATCH/"figures/figure_03.png",.45,1.7,6.1,3.1)
    d.picture(s,MISMATCH/"figures/figure_04.png",6.8,1.7,6.1,3.1)
    eq=d.equation(r"c=(r_O+r_R)/\sqrt{2},\quad d=(r_O-r_R)/\sqrt{2},\quad F_c=\frac{\langle\mathrm{mean}_t(c^2)\rangle}{\langle\mathrm{mean}_t(c^2+d^2)\rangle}","common_residual_fraction")
    d.picture(s,eq,.8,4.93,11.7,.72)
    d.text(s,"残差相关中位数 0.946–0.957；共同方向平方比 0.837–0.973。r 为训练 SD 标准化残差。\n尖括号为记录等权汇总；同向 c 与反向 d 的平方分解是统计描述，不是生理来源占比。",.55,6.0,12.2,.94,17)
    d.figure("一个投影反例：独立噪声也能留下完全共同的残差", "解析几何示意 · 2,000 组独立标准正态噪声 · 不是实测被试",MISMATCH/"figures/figure_07.png",
        ["左：两个独立观测，模型只允许拟合反向方向；右：减去该方向后，两条残差完全相同。",
         "共同残差可由受限拟合本身产生；不能仅据同向波形认定系统、浅表或共同生理来源。"],
        [MISMATCH/"content.json"],"模型失配")
    d.figure("眼电回归后的误差下降，可被幅度对照复现", "raw：原代理；clean：眼电回归后代理；amp：仅匹配 clean 的均值与尺度；Hb 读出保持固定",MISMATCH/"figures/figure_11.png",
        ["clean 减 amp：Single-Trial +0.0063 [−0.0044, 0.0175]；Simultaneous −0.0015 [−0.0104, 0.0073]。",
         "两组区间均跨零，未检出额外改善；这不是等效检验，也不能量化眼电造成的 Hb 失配。"],
        [MISMATCH/"content.json"],"模型失配",
        "单位EEG训练SD平方；29/29与26/26人，记录与被试等权。各family内BH校正，clean−amp的q为0.3812/0.7533。回归改变评分目标，未重拟合Tak，Hb轨迹不变。")

    for ds,label,identity,denom in [("eeg_fnirs_single_trial","Single-Trial","被试 26 / session_03 / 左运动区",1941),
                                  ("simultaneous_eeg_nirs","Simultaneous","VP010 / 词生成 / 左运动区",1290)]:
        d.figure(f"硬 token 的 Hb 良好个例：{label}",f"电压 v2 的 Hb 支 · seed 17 · {identity} · 20 秒；S／O 分开解码",BASE/f"staging/04_hb_tokenizer_good_examples_20261029_v1/{ds}_hb_shape_best.png",
            ["黑为条件目标／观测，蓝为硬 token，橙为连续输出；四条分量相关均约 0.96–0.98。",
             f"从 {denom:,} 例／6 人按最差分量误差选出最好一例。选图用局部方差，NMSE 用训练 SD；不代表总体。"],
            [BASE/"staging/04_hb_tokenizer_good_examples_20261029_v1/source_manifest.json",VOLTAGE/"diagnostic_export_v3/summary.json"],"离散表征",
            "Hb编码器读取6秒同模态上下文，每2秒输出ID，两个残差量化层，同分支固定码字独立解码。黑色目标为条件模型S或完整残差O；总重建对照处理后观测。示例后验选择仅用于展示。")
    d.figure("Hb 总重建保真度较高，分支恢复仍弱于总和", "电压 v2 · 三个种子先在被试内平均；每库 6 名公开测试被试；两色团等权",charts["hb_overall"],
        ["固定平滑后总 NMSE 为 0.0158／0.0096；普通观测硬量化为 0.0035／0.0017，总误差更低。",
         "参照／残差支 NMSE 约 0.078–0.088；总和更准不等于每个分支准确或来源可识别。"],
        [VOLTAGE/"voltage_summary.json",VOLTAGE/"voltage_analysis_subjects.csv"],"离散表征",
        "公共测试计划1083试次，父教师失败6，保留1077；3区域Hb共有3231例。开发计划3925试次，父失败18，保留3907。选择在验证集冻结，测试是既有公开划分而非新确认性数据。")
    d.figure("EEG 电压量化的主要限制集中在 Simultaneous", "电压 v2 · 总观测重建 · 两库各 6 名公开测试被试；左右纵轴独立，不能比较柱高",charts["eeg_overall"],
        ["Simultaneous：连续学生总 NMSE 0.0150，硬 token 为 0.6249；Single-Trial 为 0.00225→0.00268。",
         "残差支相关仅 0.063／0.050。必须保留此负结果，不能用较好的总形状或 Hb 个例替代。"],
        [VOLTAGE/"voltage_summary.json"],"离散表征",
        "电压分母为训练RMS；EEG以2秒同电极上下文生成100ms/200Hz目标，4层硬量化。O为原电压减条件修正电压，不是纯噪声。")
    d.figure("旧版掩码先导：优于自身 reader，仍弱于简单插值", "Tak 双词表 v3 · 时间块遮挡 S／O 共 4 秒 · 整体质量筛查未过，仅小规模诊断",charts["masked"],
        ["相对自身 reader，双库 Hb NMSE 下降 0.0149／0.0176；相对插值却增加 0.0831／0.1247。",
         "真实与错配供体差值的区间均跨零；改良电压版尚无可直接沿用的匹配掩码证据。"],
        [PILOT,ROOT/"experiments/configs/physiology_semantic_tokenizer/dual_component_tokenizer_v1.yaml"],"离散表征",
        "3种子先被试内平均，1000次成对被试bootstrap，未多重校正。原先导15024窗口计划、14952保留、24试次父失败。输入为冻结单ROI整数ID；Hb完整记录带通使本实验是处理后离线补全。")
    d.figure("形状字典给出了可复核的波形语言", "历史局部形状探针 · 2026-10-25 · 训练使用最多的两个参照支与两个剩余支 ID",OV/"figures/20_shape_dictionary.png",
        ["红实线为 HbO，蓝虚线为 HbR；2 秒片段先去通道均值，再除以两色团共同 RMS。",
         "可描述反向斜坡或较快起伏；均值与幅度另存，不能把这些旧版 ID 直接命名为生理机制。"],
        [OV/"REPORT.md",RUNS/"functional_tokenizer/20261025_shape_v1/SHAPE_DICTIONARY_PROVENANCE.json"] if (RUNS/"functional_tokenizer/20261025_shape_v1/SHAPE_DICTIONARY_PROVENANCE.json").exists() else [OV/"REPORT.md"],"离散表征")

    s=d.base("下游性能：沿用统一跨被试协议，保留轨道与截点", "分类表采用 10 月 29 日 11:50 截点；REFED 采用 13:34 截点；这是一份结果快照",[DOWN/"REPORT.md",REFED/"REPORT.md"],"下游性能")
    d.table(s,["项目","口径"],[
        ["划分与统计单位","严格跨被试 5 外折；神经训练 3 种子；被试与种子等权"],
        ["输入支持","一般分类 8 秒；DSR 2 秒；REFED 20 秒；EEG-only 16 通道"],
        ["分类评分","macro-F1：分别计算各类别 F1，再类别等权；表值乘 100，越高越好"],
        ["轨道","L：冻结编码器线性探测；F：全参数微调；S：监督适配／独立重实现"],
        ["报告范围","只填完整且准入的组合；未齐和未达标保留；不混合旧轮次择优"],
        ["本项目主方法","最终 Tak 双分支版本尚缺同协议直接下游结果，不能借用比较模型成绩"]
    ],[2.4,9.8],height=4.6,size=17)

    s=d.base("分类表现因任务而异，简单基线仍有竞争力", "macro-F1 × 100 · 每格取已准入完整 L／F 轨道的较高点估计；属于概览，不是统一轨道排名",[DOWN/"best_available.csv",DOWN/"negative_results.csv"],"下游性能")
    records=read_csv(DOWN/"best_available.csv")
    tasks=["motor_imagery","mental_arithmetic","wg","nback","dsr"]
    methods=[]
    for r in records:
        k=(r['method'],r['display_name'])
        if k not in methods:methods.append(k)
    rows=[]
    for m,label in methods:
        vals=[]
        for task in tasks:
            r=next(x for x in records if x['method']==m and x['display_name']==label and x['task']==task)
            val=r['display_value'] or '—'
            if r['value']:
                track={'lp':'L','ft':'F','supervised':'S'}.get(r['track'],'S')
                val += ' '+track if '基线' not in label else ''
                if m=='reve' and task in ['motor_imagery','mental_arithmetic']:val+='*'
            vals.append(val)
        rows.append([label]+vals)
    d.table(s,["方法","MI 运动想象","MA 心算","WG 词生成","n-back 记忆","DSR"],rows,[2.5,2,2,2,2,1.7],height=3.75,size=15)
    d.text(s,"双模态简单基线 MI 57.83、n-back 44.16；CBraMod FT 心算 68.81；REVE LP 的 DSR 为 63.95。\n* REVE 的 MI／MA 有目标语料重叠；未达标＝完整但未通过准入，原值见负结果附表。",.58,5.7,12.2,1.0,17)

    d.figure("REFED：正一致性尚未转化为更低绝对误差", "CCC＝一致性相关系数；MAE／RMSE＝平均绝对／均方根误差 · 每完整种子 32 人、2,720 窗口",charts['refed'],
        ["STA-Net CCC 0.0551（95% 区间 0.0433–0.0666）；MAE 49.10，高于外训练集均值基线 45.65。",
         "EFRM LP 为 10/15 作业阶段性诊断；CCC 0.0321、MAE 47.37，不能进入完整方法排名。"],
        [REFED/"tables/refed_method_diagnostics.csv",REFED/"REPORT.md"],"下游性能",
        "CCC衡量均值、幅度与变化的一致性；MAE/RMSE为原始评分单位。每记录每坐标有效点算指标，坐标、记录、被试依次等权，多种子先被试内平均。2000次被试块bootstrap，固定训练与选择，未多重校正。训练均值仅用外层训练分区。")
    s=d.base("30 秒心算诊断有改善，但不能并入五折主表", "共同输入：16 EEG／72 Hb；事件前 5 秒至后 25 秒；15 名训练、8 名开发验证被试，单种子",[DOWN/"REPORT.md",ROOT/"docs/DATA_CONTRACT.md"],"下游性能")
    d.table(s,["方法","验证 macro-F1 × 100","比较说明"],[
        ["CBraMod FT","78.92","有序 token＋三层分类头；开发集选择 epoch"],
        ["STA-Net","70.62","同一输入下默认设置"],
        ["Hb／双模态简单基线","61.98／60.58","可作为同支持参考"],
        ["EEG／BrainFusion","44.66／43.21","负结果；后者有 stacking 不稳定"]
    ],[3.2,3,6],height=2.85,size=18)
    d.text(s,"Visual：已完成方法均未通过无信息参照检查，且 EEG 通道身份存在已确认缺陷。\n该配置的分数不能验证预定电极／几何方案；不将旧数值补入正常性能排名。",.65,5.08,12,1.2,20)

    s=d.base("目前支持的是可检查的条件表示，解释强度需要分层", "重建、稳定性、来源归属和任务效用分别评价",[BASE/"paper_outline.md",ROOT/"experiments/RESULTS_INDEX.md"],"结论与论文")
    d.table(s,["解释层次","直接支持","仍受限制"],[
        ["波形与模型参照","部分 Hb 可拟合、可量化、可从硬 ID 解码","总重建好不能替代分支准确性"],
        ["条件参数","可见个体／任务分布；合成变化有响应","重复性区间与任务检验尚不足"],
        ["失配表型","短期排序重复；残差具有共同方向","投影与幅度对照阻止直接来源归因"],
        ["跨模态信息","有利合成条件下存在检出能力","实测匹配优势和强基线收益未普遍成立"],
        ["脑状态评估","已建立外部比较及任务差异参照","最终主方法的统一下游证据仍待补"]
    ],[2.4,4.9,4.9],height=4.4,size=17)

    s=d.base("论文大纲：围绕参照是否适用、表示保留什么展开", "建议题目：生理模型引导的 EEG–fNIRS 离散表征与脑状态评估",[BASE/"paper_outline.md"],"结论与论文")
    d.table(s,["章节","正文内容","主要图表"],[
        ["引言","观测差异；模型参照与表示学习的动机；三个研究问题","研究问题与贡献"],
        ["1 材料和方法","数据、观测坐标、Tak 教师、两分支硬量化、冻结序列读出","图1 总框架；表1 数据及分母"],
        ["2.1 参照分解","条件重建与适用范围；数值失败和对照","重建样例与误差分布"],
        ["2.2 信息保留","连续到硬量化损失；分支及总重建；码率与上下文","波形、原型及公平基线"],
        ["2.3 可解释性","分支原型、个体差异、状态关联、模型失配","重复性、任务对照、来源反例"]
    ],[2.5,6.2,3.5],height=4.5,size=17)
    s=d.base("论文结果后半部分保留负结果与待补证据", "沿用已写好的 paper_outline.md；最终方法版本、统计分母和直接下游结果需要统一",[BASE/"paper_outline.md"],"结论与论文")
    d.table(s,["章节","已有内容","成稿边界"],[
        ["2.4 掩码与跨模态增量","旧版先导、错配对照、强插值基线","电压／长上下文版本需匹配其支持重做验证"],
        ["2.5 状态评估","独立模型比较、分类与连续回归参照","最终主方法与分支消融的直接成绩尚缺"],
        ["3 讨论","参照分解价值、码率代价、测量与模型条件性","不把观测语义升级为唯一生理来源"],
        ["4 结论","给出已验证性质、比较收益与适用范围","结论只覆盖实际结果，不填预期收益"],
        ["补充材料","完整参数、分母、失败、历史版本与复现入口","正文保留影响主张的负结果与公平条件"]
    ],[2.4,4.8,5],height=4.4,size=17)
    main_count=len(d.prs.slides)

    # Supplementary pages keep historical identities and complete evidence coverage.
    shutil.copyfile(BASE/"20261029_project_framework_v10/project_framework_token_first.png", output/"figures/full_architecture_reference.png")
    shutil.copyfile(BASE/"20261029_project_framework_v10/project_framework_token_first.svg", output/"figures/full_architecture_reference.svg")
    tak_source=RUNS/"shared_driver_reconstruction/20261027_tak_v1_run1/report_v4/figures/tak_equations.tex"
    tak_equations=d.equation(r"\begin{gathered}"+r"\\[10pt]".join(tak_source.read_text().splitlines())+r"\end{gathered}","tak_equations")
    d.figure("补充：Tak 血流核心与观测关系", "四个独立血管状态；驱动尺度由 EEG 读出固定；方程从原 LaTeX 源码重渲染为位图",tak_equations,
        ["驱动先影响血流调节与流量，再影响体积和脱氧状态；总 Hb 状态与体积绑定。",
         "转运／黏弹性时间以秒计；衰减系数为每秒；观测尺度常数未经独立标定时不代表统一绝对浓度。"],
        [ROOT/"docs/ARCHITECTURE.md",tak_source],"补充｜方法",
        "r驱动、s血流调节、f相对血流、v相对静脉体积、q相对脱氧Hb，p为派生量。gamma单位每秒平方；alpha和E0无量纲。完整Tak DCM观测推断体系未全部复现。")
    d.figure("补充：多片段实测重估区分优化稳定与数据稳定", "10 Hz · 衰减／转运参数 · 168 段均数值合格；主分析 126 段、REFED 跨情绪补充 42 段",EXPANDED/"figures/05_support_and_subject_dispersion.png",
        ["同支持转运时间乘性散布：Single-Trial 10.8%、Simultaneous 27.2%、Visual 22.6%。",
         "多起点最大跨度不到 0.004%；REFED 每视频只有一段，无法估计视频内波动。Visual 沿用旧映射。"],
        [EXPANDED/"index.html"],"补充｜参数",
        "先在支持内/支持间/被试间算log RMS，汇总平方后开方，再转成exp(RMS)−1的百分比。不是方差可加分解，各类柱分母不同；Visual跨记录仅3人。")
    d.figure("补充：任务中心可移动，留出判别仍需独立核对", "转运时间／有效增益；每条线连接同被试任务中心，不是生理时间轨迹",PARAM/"figures/05_tau_beta_tasks.png",
        ["Simultaneous n-back 相对词生成：转运时间 −19.4%，有效增益 −5.5%；各 6 名配对被试。",
         "Go/No-go 相对词生成：有效增益 −14.1%（5/6 人可配对）；不能推断因果任务效应。"],
        [PARAM/"index.html"],"补充｜参数")
    d.figure("补充：旧模型的相近拟合可伴随不同潜变量误差", "历史 inlet-balance 合成恢复 · 每点为真值×种子×模态×先验条件；红叉保留未收敛候选",OV/"figures/07_fit_vs_recovery.png",
        ["零显式先验下，加入 EEG 后 Hb NRMSE 几乎不变：0.003699→0.003688。",
         "衰减系数绝对误差却由 0.1098 降至 0.0065 每秒；驱动 NRMSE 由 0.0590 降至 0.0074。"],
        [ROOT/"docs/report/20261021_source_observability_v2/content.json"],"补充｜恢复与归属",
        "图横轴是原始模型坐标Hb RMSE而非正文标准化NRMSE。128个有噪声条件中27个未收敛有限候选用红叉；完整套件648任务，567完成，80数值失败，1旧源不可用。颜色为先验乘数，点形为可见模态。")
    d.figure("补充：拟合改善并不保证真实成分恢复改善", "历史 inlet-balance · 已知真值反事实 · 横轴观测误差改善，纵轴物理 Hb 恢复改善",OV/"figures/11_fit_vs_attribution.png",
        ["真实增益 1.25、无共同项：加额外项使观测 NRMSE 0.1952→0.1907，物理 Hb 却 0.1816→0.2193。",
         "转运时间正确率从 16/16 降为 8/16；右下象限是更好拟合与更差归属同时出现的反例。"],
        [ROOT/"docs/report/20261021_mechanism_attribution_v2/content.json"],"补充｜恢复与归属")
    d.figure("补充：真实异质性存在时，独立动态有明显收益", "历史模型合成控制 · 每点为生成种子，线为 8 种子均值；横轴是通道面板，每次独立重拟合",OV/"figures/14_spatial_synthetic.png",
        ["独立双源、完整面板：自由双动态隐藏 NRMSE 0.0294，绑定双动态 0.6616。",
         "相同双源反而为 0.0258 对 0.0238；增加自由度的价值依赖真实异质性。"],
        [ROOT/"docs/report/20261021_spatial_information_v3/content.json"],"补充｜空间与信息")
    d.figure("补充：实测空间收益没有达到合成条件的水平", "历史局部空间实验 · 柱为完整分母汇总，点为被试；更低隐藏误差更好",OV/"figures/15_spatial_measured.png",
        ["单动态约 1.172、双动态约 2.148；联合 ridge 约 0.998，双动态未优于同输入线性参考。",
         "失败分母仍保留；多通道、几何信息和独立动态自由度应分别比较。"],
        [ROOT/"docs/report/20261020_local_spatial_v5/LOCAL_SPATIAL_REPORT.pdf"],"补充｜空间与信息")
    d.figure("补充：实测正确 EEG 尚未同时胜过两项必要对照", "历史独立跨模态检验 · 29 个可错配窗／6 名被试 · 点为被试，正值有利于正确配对",OV/"figures/16_crossmodal_measured.png",
        ["固定 EEG 权重的单动态：相对 Hb-only 的 NMSE 改善 −0.00803，相对错误 EEG 为 +0.00146。",
         "两项 95% 区间均跨零；双动态的较大变化受失败惩罚影响，不能据此推断更多生理来源。"],
        [ROOT/"docs/report/20261021_crossmodal_information_v3/content.json"],"补充｜空间与信息")
    d.figure("补充：在已知有利条件下，可以检出独立 EEG 信息", "合成阳性控制 · 已知读出与同一噪声实现；上下两图采用独立纵轴",OV/"figures/17_crossmodal_positive.png",
        ["Hb 供体看不到第二动态、EEG 可以看到时：平均 NMSE 0.9851→0.2825，16/16 身份越过预定阈值。",
         "288 次拟合均成功；该阳性说明条件检出能力，不覆盖原实测背景的无稳定增量。"],
        [ROOT/"docs/report/20261021_crossmodal_information_v3/content.json"],"补充｜空间与信息")
    d.figure("补充：历史条件表示的读出收益不等于任务收益", "历史功能 tokenizer · CT 连续条件、QT 量化条件；C0／Q0 为对应普通表示",OV/"figures/18_token_readout.png",
        ["图为条件目标 NRMSE 相对降低，向右有利；EEG 为功率代理，X 为加权 Hb 对比坐标。",
         "成功子集平衡准确率差：QT 相对 Q0 的运动想象 −7.92、n-back +5.26 个百分点；父失败限制结论。"],
        [OV/"REPORT.md",ROOT/"docs/report/20261013_functional_tokenizer_v3/report.md"],"补充｜历史表征")
    d.figure("补充：真实 EEG 好于错配，仍可能差于自身 Hb 上下文", "历史功能 tokenizer · 六名评价被试，三个种子先被试内汇总；向右有利于真实配对",OV/"figures/19_token_crossmodal.png",
        ["相对仅 Hb 上下文，量化条件表示的隐藏 NRMSE 增加 0.01229／0.01583。",
         "中间错配对照的正向收益没有抵消对强基线的退化；不同代际结果不与当前掩码先导合并。"],
        [OV/"REPORT.md",ROOT/"docs/report/20261013_functional_tokenizer_v3/report.md"],"补充｜历史表征")
    d.figure("补充：形状与连续尺度保留了部分时空关系", "历史 2026-10-25 局部形状探针 · 各子图按自身指标解释，不比较不同指标柱高",OV/"figures/21_shape_probe.png",
        ["A：旧表示的观测读出；B：新形状 ID 与尺度预测后续 Hb；C／D：正确时间／区域配对的收益。",
         "形状加尺度有可用信息，但尚未建立 SSM 特有优势；尺度与可见上下文的贡献需保留。"],
        [OV/"REPORT.md"],"补充｜历史表征")
    d.figure("补充：处理链会明显改变相同身份的拟合表象", "历史选择性复测 · REFED 同一 90 秒样例 · 左旧读取链，右统一滤波链",OV/"figures/02_refed_filter.png",
        ["HbO 相关 0.416→0.789、NRMSE 0.727→0.178；HbR 相关 0.257→0.815、误差 1.309→0.248。",
         "处理影响不等于新生理信息；这组选择性样例不能更新全库存分布。"],
        [OV/"REPORT.md"],"补充｜观测与版本")
    d.figure("补充：历史四库汇总显示响应和增益选择的收益", "各数据集等权 · 相对衰减对照的 Hb NRMSE 改善；区间为原报告的探索性区间",OV/"figures/22_four_dataset_effects.png",
        ["最佳稳定响应改善 +0.0684 [0.0459, 0.0902]；增益选择 +0.0585 [0.0399, 0.0761]。",
         "谱权重选择差 −0.0006 [−0.0071, 0.0057]，区间跨零；不同改动的收益需分别判断。"],
        [ROOT/"docs/report/20261013_refed_four_dataset_v1/REFED_FOUR_DATASET_REPORT.pptx"],"补充｜观测与版本")
    s=d.base("补充：两轮下游比较的变化不能归因于单一因素", "沿用 2026-10-28 的两轮比较报告；不读取或重新汇总历史保护预测",[BASE/"20261028_downstream_comparison_v3/REPORT.md"],"补充｜下游协议")
    d.table(s,["变化","为什么影响解释"],[
        ["数据与时钟","旧 v1 → 公共 v5／对齐 v2；WG、n-back、DSR 的有效窗口改变"],
        ["通道与观察长度","同为16 EEG通道但名单改变；STA-Net旧3/13秒，新一般8/8秒"],
        ["选择与汇总","旧主要用inner0，新3内折；旧折内合并F1，新被试等权F1"],
        ["同轮 LP 与 FT","CBraMod心算提升14.34点、词生成提升10.05点，MI未改善"],
        ["作者原文比较","指标与划分不同者只列背景，不能拼成同协议排行榜"]
    ],[3.2,9],height=4.15,size=18)
    d.text(s,"31 个可对应分类结果按旧口径重算后：5 个提高至少 2 点、11 个降低至少 2 点、15 个小于 2 点。",.62,6.25,12,.55,17)
    s=d.base("补充：固定 Hb-only 拟合的完整失败分母", "单位为窗口×Hb 通道对；数值未通过、无支持、执行异常互斥，所有条件均无缺失终态",[MISMATCH/"content.json"],"补充｜覆盖与失败")
    d.table(s,["数据集","全计划","成功","数值未通过","无支持","执行异常"],[
        ["Single-Trial","146,376","142,340","3,750","280","6"],
        ["REFED","75,072","74,347","725","0","0"],
        ["Simultaneous","172,440","139,490","1,554","31,392","4"],
        ["Visual（纯 Hb）","79,200","78,270","928","0","2"]
    ],[3.1,1.9,1.9,1.9,1.8,1.6],height=3.2,size=18)
    d.text(s,"没有支持不等于数值失败；Hb-only 成功不替代 EEG-only 或联合拟合成功。\n正文 103 名被试来自四库原计划；时序／眼电的三通道面板采用另一分母，见原报告。",.6,5.5,12.1,1.1,19)
    s=d.base("补充：其余保留结果的解释用途", "补全目录的历史结果家族；只作背景，不与新 Tak／电压版合并统计",[BASE/"README.md",OV/"REPORT.md"],"补充｜证据覆盖")
    d.table(s,["保留结果家族","汇报中的用途／限制"],[
        ["单位、对齐、原生 Hb 方向、光学处理","说明测量链敏感性；方向不唯一标识神经来源"],
        ["联合可行性、容量方向、旧模型衰减","区分表示边界、额外自由度与数值失败；搜索失败不是不可行证明"],
        ["固定 Tak 传感器代理与多层控制","局部代理修改仍需全分母及隐藏预测对照；详细图册保留原入口"],
        ["teacher 稳健性、连续状态、有限响应","局部收益依赖先验、输入支持与已知生成条件"],
        ["T3／Step5、E／R、SSM／LC、Atlas","历史资格与负结果保留；可读表型不替代物理资格"],
        ["Croce 与早期求解器验证","物理实现与数值审计背景；未执行设计不算结果"]
    ],[4.4,7.8],height=4.6,size=16)
    return d,main_count


def finish(d, main_count):
    pptx=d.output/"可解释性分析_最终汇报.pptx"
    d.prs.save(pptx)
    shutil.copyfile(__file__,d.output/"renderer_source.py")
    with tempfile.TemporaryDirectory(prefix="lo_interpretability_") as profile:
        result=subprocess.run(["libreoffice",f"-env:UserInstallation={Path(profile).as_uri()}",
            "--headless","--convert-to","pdf:impress_pdf_Export:{\"ReduceImageResolution\":{\"type\":\"boolean\",\"value\":\"false\"}}",
            "--outdir",str(d.output),str(pptx)],capture_output=True,text=True,timeout=180)
    (d.output/"export.log").write_text(result.stdout+result.stderr,encoding="utf-8")
    pdf=pptx.with_suffix(".pdf")
    if result.returncode or not pdf.exists():raise RuntimeError("PDF export failed")
    checks={"slides":len(d.prs.slides),"main_slides":main_count,"supplementary_slides":len(d.prs.slides)-main_count,
            "pptx_shape_bounds":[],"pdf_pages":[],"figure_bitmap_only":True,
            "retained_bitmaps_below_200_dpi": [
                {"slide":p["slide"],"source":f["source"],"dpi":f["effective_dpi"]}
                for p in d.pages for f in p["figures"] if f["effective_dpi"] < 200],
            "retained_bitmap_policy":"Retain original pixels; no artificial upscaling. New plots use 260 dpi; new LaTeX uses 300 dpi."}
    def normalized(text):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).replace("\u200b", "")
    for i,slide in enumerate(d.prs.slides,1):
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0 or shape.left+shape.width>d.prs.slide_width+2 or shape.top+shape.height>d.prs.slide_height+2:
                checks["pptx_shape_bounds"].append({"slide":i,"shape":shape.name})
    with fitz.open(pdf) as doc:
        assert len(doc)==len(d.prs.slides)
        for i,page in enumerate(doc):
            expected=len(d.pages[i]["figures"])
            image_info=page.get_image_info()
            spans=[s for b in page.get_text("dict")["blocks"] if "lines" in b for l in b['lines'] for s in l['spans']]
            outside=[s['text'] for s in spans if not (page.rect+(-1,-1,1,1)).contains(fitz.Rect(s['bbox']))]
            row={"page":i+1,"images":len(image_info),"expected_figure_images":expected,
                 "searchable_characters":len(page.get_text()),"outside_text":outside}
            missing=[]
            pdf_text=normalized(page.get_text())
            for shape in d.prs.slides[i].shapes:
                frames=[shape.text_frame] if shape.has_text_frame else []
                if shape.has_table:
                    frames.extend(c.text_frame for row_ in shape.table.rows for c in row_.cells)
                for frame in frames:
                    for paragraph in frame.paragraphs:
                        value=normalized(paragraph.text)
                        if value and value not in pdf_text:
                            missing.append(paragraph.text)
            row["missing_pptx_paragraphs_in_pdf"]=missing
            assert not missing,(i+1,missing)
            assert len(image_info)>=expected,(i+1,len(image_info),expected)
            assert row['searchable_characters']>40
            assert not outside,(i+1,outside)
            page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(d.output/"previews"/f"page_{i+1:02d}.png")
            checks["pdf_pages"].append(row)
    assert not checks["pptx_shape_bounds"]
    for start in range(0,len(d.pages),12):
        sheet=Image.new('RGB',(1920,1080),'#DDE3E8');draw=ImageDraw.Draw(sheet)
        for n in range(start,min(start+12,len(d.pages))):
            im=Image.open(d.output/"previews"/f"page_{n+1:02d}.png");im.thumbnail((470,264))
            x=(n-start)%4*480;y=(n-start)//4*360
            sheet.paste(im,(x+5,y+28));draw.text((x+9,y+6),f"{n+1:02d}",fill='black')
        sheet.save(d.output/"previews"/f"contact_{start//12+1:02d}.png")
    manifest={"kind":"dated_communication_export","date":"2026-10-29", "generator":relative(Path(__file__)),
        "evidence_policy":"Original summaries, tables and protocol remain authoritative; no refits or new statistical inference.",
        "native_signal_reads":0,"protected_evaluation_reads":0,"main_slides":main_count,
        "detailed_architecture_reference":"figures/full_architecture_reference.png",
        "downstream_snapshot":"2026-10-29 11:50 Asia/Hong_Kong; REFED 13:34", "slides":d.pages,"formulas":d.formulas}
    (d.output/"source_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    (d.output/"delivery_checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"pptx":str(pptx),"pdf":str(pdf),"slides":len(d.prs.slides),"main":main_count},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() and any(out.glob("*.pptx")):
        parser.error("Preserve dated exports: choose a fresh versioned output.")
    out.mkdir(parents=True,exist_ok=True)
    deck,main=build(out)
    finish(deck,main)
