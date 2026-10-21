"""Portable, synthetic report fixtures; no measured evidence is accessed."""
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw
import pytest

from experiments.scripts import render_source_diagnostics_report as renderer


def _content(tmp_path):
    image = Image.new("RGB", (1400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 30, 1350, 440), outline="#18324d", width=3)
    draw.line([(60, 400), (400, 220), (800, 330), (1330, 80)], fill="#2775ad", width=7)
    image.save(tmp_path / "fixture.png")
    return {
        "schema": renderer.SCHEMA,
        "title": "合成报告：拟合与参数恢复的边界",
        "subtitle": "仅用于验证排版，不包含实验结果。",
        "date": "2026-10-21",
        "provenance": ["临时非授权测试样例；正文、表格和图注应可选择。"],
        "summary": ["低重建误差只说明观测重建较好，参数恢复需要单独的真值对照。"],
        "sections": [{"title": "方法、图和跨页证据表", "blocks": [
            {"type": "paragraph", "text": "长路径必须自动换行：/synthetic/" + "long_path_" * 90},
            {"type": "figure", "path": "fixture.png", "caption": "整幅合成图以 PNG 嵌入；蓝线仅为排版样例。",
             "source": "synthetic fixture"},
            {"type": "table", "title": "跨页表格", "columns": ["测试单元", "解释与结果"],
             "widths": [1, 4], "rows": [[f"单元 {i:03}", f"保留第 {i} 项的真实数值标签。"] for i in range(70)] +
                 [["长行", "超长单元格需要跨页继续，且不截断。" * 160]],
             "caption": "所有行均来自临时样例，不代表独立被试。", "source": "synthetic rows"},
            {"type": "paragraph", "text": "报告末尾必须保留。"},
        ]}],
    }


def test_report_keeps_cjk_and_tables_searchable_and_figures_bitmap(tmp_path):
    content = _content(tmp_path)
    source = tmp_path / "report_content.json"
    source.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "export_v1"
    audit = renderer.render_report(source, output)
    checks = audit["exports"]["TECHNICAL_REPORT.pdf"]
    assert checks["pages"] >= 4
    assert checks["displayed_images"] == 1
    assert checks["image_objects"] == 1
    assert checks["text_outside_page"] == []
    assert checks["full_page_raster_images"] == 0
    with fitz.open(output / "TECHNICAL_REPORT.pdf") as document:
        text = "\n".join(page.get_text() for page in document)
        assert "报告末尾必须保留" in text
        assert "单元 069" in text
        assert sum("测试单元" in page.get_text() for page in document) >= 3
        assert "超长单元格需要跨页继续" in text
        assert document.get_toc()
    packed = json.loads((output / "content.json").read_text(encoding="utf-8"))
    image_path = packed["sections"][0]["blocks"][1]["path"]
    assert not Path(image_path).is_absolute()
    assert (output / image_path).is_file()
    assert "&lt;" not in (output / "TECHNICAL_REPORT.html").read_text(encoding="utf-8")
    placement = checks["image_placements"][0]
    assert placement["effective_dpi_x"] == pytest.approx(renderer.FIGURE_DPI, abs=.5)
    assert len(list((output / "preview").glob("page_*.png"))) == checks["pages"]
    original_pdf = (output / "TECHNICAL_REPORT.pdf").read_bytes()
    with pytest.raises(FileExistsError, match="must be fresh"):
        renderer.render_report(source, output)
    assert (output / "TECHNICAL_REPORT.pdf").read_bytes() == original_pdf
    # A two-line caption fits the current page while its source alone would
    # spill. Move the complete group instead of creating an orphan source line.
    flow = renderer._FlowPDF("图注来源分页", fitz.Font(renderer.FONT_NAME))
    flow.new_page()
    flow.y = renderer.BODY_BOTTOM - 150
    caption = "图注必须与来源相邻。" * 7
    flow.image(tmp_path / "fixture.png", 280, 100, caption, "synthetic_source")
    assert len(flow.doc) == 2
    assert "图注必须与来源相邻" not in flow.doc[0].get_text()
    assert "图注必须与来源相邻" in flow.doc[1].get_text()
    assert "synthetic_source" in flow.doc[1].get_text()
    flow.doc.close()
    table_flow = renderer._FlowPDF("表头分页", fitz.Font(renderer.FONT_NAME))
    table_flow.new_page()
    table_flow.y = renderer.BODY_BOTTOM - 60
    table_flow.table({"columns": ["测试单元", "解释"],
                      "rows": [["首行", "跨行解释必须跟在表头后面。" * 7]]})
    assert len(table_flow.doc) == 2
    assert "测试单元" not in table_flow.doc[0].get_text()
    assert "测试单元" in table_flow.doc[1].get_text()
    assert "首行" in table_flow.doc[1].get_text()
    table_flow.doc.close()
    title_flow = renderer._FlowPDF("标题分页", fitz.Font(renderer.FONT_NAME))
    title_flow.new_page()
    title_flow.y = renderer.BODY_BOTTOM - 115
    table = {"columns": ["测试单元", "解释"],
             "rows": [["首行", "跨行解释必须跟在标题和表头后面。" * 7]]}
    title_flow.heading("标题必须与表格同页", keep=title_flow.table_layout(table)["opening_height"])
    title_flow.table(table)
    assert len(title_flow.doc) == 2
    assert "标题必须与表格同页" not in title_flow.doc[0].get_text()
    assert "标题必须与表格同页" in title_flow.doc[1].get_text()
    assert "首行" in title_flow.doc[1].get_text()
    title_flow.doc.close()


def test_equation_uses_latex_png_in_both_exports_and_retains_source(tmp_path):
    content = _content(tmp_path)
    equation = {"type": "equation", "latex": r"\begin{aligned}y_t &= Hx_t + \epsilon_t\\x_{t+1}&=Ax_t\end{aligned}",
                "caption": "状态与观测的测试公式。", "explanation": "状态为隐变量，观测为评分输入；这里只验证排版。"}
    content["sections"][0]["blocks"] = [equation]
    source = tmp_path / "content.json"
    source.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "formulas_export"
    audit = renderer.render_report(source, output)
    assert audit["exports"]["TECHNICAL_REPORT.pdf"]["displayed_images"] == 1
    assert audit["exports"]["FIGURE_ATLAS.pdf"]["displayed_images"] == 1
    assert (output / "formulas/equation_01.tex").is_file()
    assert (output / "formulas/equation_01.png").is_file()
    html = (output / "TECHNICAL_REPORT.html").read_text(encoding="utf-8")
    assert 'src="formulas/equation_01.png"' in html
    assert r"\begin{aligned}" not in html
    assert "状态为隐变量" in html


def test_content_rejects_ambiguous_order_and_nonrectangular_tables(tmp_path):
    content = _content(tmp_path)
    content["sections"][0]["paragraphs"] = ["不能指定第二条展示路径。"]
    with pytest.raises(ValueError, match="either ordered blocks or grouped"):
        renderer.validate_content(content, tmp_path)
    content["sections"][0].pop("paragraphs")
    content["sections"][0]["blocks"][2]["rows"] = [["缺一列"]]
    with pytest.raises(ValueError, match="column count"):
        renderer.validate_content(content, tmp_path)


def test_wrapping_bounds_long_tokens_and_preserves_characters():
    font = fitz.Font(renderer.FONT_NAME)
    text = "解释路径" + "/a_very_long_path/" * 50
    lines = renderer.wrap_text(font, text, 100, 11)
    assert "".join(lines) == text
    assert all(font.text_length(line, fontsize=11) <= 100.001 for line in lines)
