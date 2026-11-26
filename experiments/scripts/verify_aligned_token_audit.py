"""Verify the frozen support, all catalogue links, PDF bitmap figures and portable evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import audit_aligned_tokenizer as audit
from run_dual_component_tokenizer import write_json


def verify(report, preview):
    out = audit.OUT
    states = {p.stem: p for p in out.glob("*_status.json")}
    modes = {
        "extract",
        "members",
        "S",
        "O",
        "extended",
        "local",
        "spatial",
        "waveform",
        "specificity",
        "visibility",
        "budget",
    }
    needed = {m + "_" + s + "_status" for m in ("eeg", "hb") for s in modes} | {
        "eeg_" + s + "_status" for s in ("eye", "events", "event_controls", "cross")
    }
    assert needed == states.keys() and all(
        json.loads(p.read_text())["status"] == "completed" for p in states.values()
    )
    matrix = pd.read_csv(report / "evidence.csv")
    assert matrix.ID.tolist() == [f"A{i:02}" for i in range(1, 65)]
    for paths in matrix["证据文件"]:
        for name in paths.split("|"):
            assert (report / "tables" / name).is_file(), name
    with ZipFile(report / "evidence.xlsx") as x:
        assert x.testzip() is None
        assert len(ET.fromstring(x.read("xl/worksheets/sheet1.xml")).findall(".//{*}row")) == 65
    with ZipFile(report / "report_bundle.zip") as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {
            p.relative_to(report).as_posix()
            for p in report.rglob("*")
            if p.is_file() and p.name != "report_bundle.zip"
        }
    doc = fitz.open(report / "REPORT.pdf")
    pages = list(doc)
    assert len(doc.get_toc()) == len(pages)
    assert len(pages) > 40 and sorted(matrix["正文页"].unique())[-1] <= len(pages)
    preview.mkdir(exist_ok=True, parents=True)
    chart_pages = []
    dpis = []
    all_txt = []
    for n, p in enumerate(pages, 1):
        t = p.get_text()
        assert len(t) > 100 and str(n).zfill(2) in t
        for w in p.get_text("words"):
            assert w[0] >= 0 and w[1] >= 0 and w[2] <= p.rect.width + 1 and w[3] <= p.rect.height + 1
        imgs = p.get_image_info()
        if imgs:
            assert len(imgs) == 1
            z = imgs[0]
            bbox = fitz.Rect(z["bbox"])
            assert p.rect.contains(bbox)
            dpis.append(z["width"] * 72 / bbox.width)
            assert 200 <= dpis[-1] <= 300
            chart_pages.append(n)
        all_txt.append(t)
        pix = p.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        pix.save(preview / f"{n:02}.png")
    assert len(chart_pages) == len(list((report / "figures").glob("*.png")))
    contents = "".join(all_txt).replace(" ", "")
    for a in matrix.ID:
        assert a in contents
    for k in ["0.18861", "0.22834", "0.1672", "0.17726"]:
        assert k in contents
    for st in range(0, len(doc), 9):
        sheet = Image.new("RGB", (1260, 3 * 325), "#cdd6d9")
        for n in range(st, min(st + 9, len(doc))):
            pic = Image.open(preview / f"{n+1:02}.png").convert("RGB")
            pic.thumbnail((416, 303))
            x = (n - st) % 3 * 420
            y = (n - st) // 3 * 325
            sheet.paste(pic, (x, y))
            ImageDraw.Draw(sheet).text((x + 4, y + 303), str(n + 1), fill="#133d54")
        sheet.save(preview / f"sheet_{st:02}.jpg", quality=90)
    for p in fitz.open(report / "tables/blind/blinded_waveforms.pdf"):
        assert len(p.get_image_info()) == 4
    assert len(pd.read_csv(report / "tables/blind/ratings.csv")) == 32
    points = {}
    for mod, expected in [("eeg", 780210), ("hb", 838583)]:
        c = pd.read_csv(out / f"{mod}_17/coverage.csv").query("dataset=='pooled'").set_index("branch")
        assert set(c.valid) == {expected}
        con = (
            pd.read_csv(out / f"{mod}_17/conditional.csv")
            .query("target=='X' and reference=='S' and candidate=='SO'")
            .iloc[0]
        )
        v = (
            pd.read_csv(out / f"{mod}_order_budget.csv")
            .query("adjacent and branch=='S'")
            .groupby(["method", "dataset"])
            .mse.mean()
            .groupby("method")
            .mean()
        )
        points[mod] = dict(
            patches=expected,
            coverage=c.readable_fraction.to_dict(),
            O_gain_given_S=con.gain,
            equal_width_order_gain=1 - v.bigram_dwell / v.bag,
        )
    primary = pd.read_csv(out / "events_17/event_contrasts.csv").query("primary").iloc[0]
    h = (
        pd.read_csv(out / "events_17/coupling_contrasts.csv")
        .query("color=='two_color' and reference=='base'")
        .iloc[0]
    )
    extra = pd.read_csv(out / "systemic_controls.csv").groupby("method").nmse.mean()
    assert np.isclose(extra.base, h.reference_nmse) and np.isclose(extra.aligned, h.aligned_nmse)
    case = pd.read_csv(out / "paired_example.csv")
    with np.load(report / "tables/paired_example.npz") as z:
        assert len(case) == 2 and all(z[m + b + "_id"].shape == (15,) for m in ["eeg", "hb"] for b in "SO")
    summary = dict(
        status="completed",
        experiment_id=audit.PLAN["experiment_id"],
        verified=datetime.now(timezone.utc).isoformat(),
        seed=17,
        dimension=128,
        stages=len(states),
        catalogue=matrix["状态"].value_counts().to_dict(),
        models=points,
        primary_event=dict(
            people=int(primary.people),
            effect=primary.ma_minus_bl_change,
            exact_p=primary.primary_exact_p,
            subject_bootstrap_95=[primary.ci_low, primary.ci_high],
        ),
        measured_Hb=dict(
            people=int(h.people),
            trial_support=json.loads((out / "events_17/events_summary.json").read_text())["D"],
            history=h.reference_nmse,
            aligned=h.aligned_nmse,
            people_improved=int(h.improved),
            systemic=extra.to_dict(),
        ),
        report=report.relative_to(out).as_posix(),
        pages=len(doc),
        bitmap_figure_pages=len(chart_pages),
        effective_dpi_range=[min(dpis), max(dpis)],
        portable_zip_checked=True,
    )
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=audit.OUT / "report_v2")
    p.add_argument("--preview", type=Path, required=True)
    args = p.parse_args()
    verify(args.report, args.preview)
