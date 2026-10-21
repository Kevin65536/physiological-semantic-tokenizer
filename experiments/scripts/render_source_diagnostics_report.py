#!/usr/bin/env python3
"""Render an explicitly authored Chinese report without reading experiment data.

``--content report_content.json --output <fresh-directory>`` writes searchable
PDF/HTML reports, a bitmap figure atlas, editable LaTeX sources and delivery
checks. Relative figure paths are resolved against the content JSON. This
communication export does not aggregate runs, fit models or establish results.

Content has ``title``, optional ``subtitle``, ``date``, ``summary`` and
``provenance`` string lists, and ``sections``. A section has ``title`` and either
ordered ``blocks`` or ``paragraphs``, ``equations``, ``figures`` and ``tables``.
Blocks have ``type`` paragraph/heading/figure/equation/table. Paragraph/heading
use ``text``; figures use ``path`` and ``caption``; equations use bare ``latex``
(aligned and matrices are supported) and optional caption/explanation; tables
use ``columns`` and rectangular ``rows``, with optional relative ``widths``.
All non-text blocks may carry a ``source`` string. Optional top-level ``sources``
are rendered as an appendix. The input content is retained as ``content.json``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


SCHEMA = "source_diagnostics_report_v1"
# The Noto TTC maps several ordinary Chinese characters to compatibility
# codepoints in MuPDF's generated ToUnicode map (e.g. 留 -> U+F9CD). The bundled
# simplified-Chinese font preserves ordinary Unicode in search/copy operations.
FONT_NAME = "china-s"
PAGE_WIDTH, PAGE_HEIGHT = 595.276, 841.890
MARGIN, BODY_TOP, BODY_BOTTOM = 46.0, 58.0, 791.0
BODY_WIDTH = PAGE_WIDTH - 2 * MARGIN
FIGURE_DPI, FORMULA_DPI = 250, 300
INK, MUTED = (.095, .196, .302), (.31, .37, .43)
BLOCK_TYPES = {"paragraph", "heading", "figure", "equation", "table"}


def _string(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'a' if empty else 'a nonempty'} string")
    return value


def _strings(value, name):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of strings")
    for item in value:
        _string(item, name)
    return value


def section_blocks(section):
    """Return one presentation order, preserving an explicitly supplied order."""
    if "blocks" in section:
        return section["blocks"]
    blocks = [{"type": "paragraph", "text": text} for text in section.get("paragraphs", [])]
    for plural, kind in (("equations", "equation"), ("figures", "figure"), ("tables", "table")):
        blocks.extend({**entry, "type": kind} for entry in section.get(plural, []))
    return blocks


def validate_content(content, base):
    """Validate the communication contract and only explicitly named images."""
    from PIL import Image

    if not isinstance(content, dict):
        raise ValueError("content must be an object")
    if content.get("schema", SCHEMA) != SCHEMA:
        raise ValueError(f"unsupported content schema: {content.get('schema')}")
    _string(content.get("title"), "title")
    for key in ("subtitle", "date"):
        _string(content.get(key, ""), key, empty=True)
    for key in ("summary", "provenance", "sources"):
        _strings(content.get(key, []), key)
    sections = content.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("sections must be a nonempty list")
    counts = Counter()
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("each section must be an object")
        _string(section.get("title"), "section.title")
        if "blocks" in section and any(key in section for key in ("paragraphs", "equations", "figures", "tables")):
            raise ValueError("section must use either ordered blocks or grouped content")
        _strings(section.get("paragraphs", []), "section.paragraphs")
        for key in ("blocks", "equations", "figures", "tables"):
            if key in section and not isinstance(section[key], list):
                raise ValueError(f"section.{key} must be a list")
        for block in section_blocks(section):
            if not isinstance(block, dict) or block.get("type") not in BLOCK_TYPES:
                raise ValueError("unknown report block type")
            kind = block["type"]
            counts[kind] += 1
            for key in ("caption", "source", "explanation", "label"):
                _string(block.get(key, ""), key, empty=True)
            if kind in ("paragraph", "heading"):
                _string(block.get("text"), f"{kind}.text")
            elif kind == "figure":
                path = Path(_string(block.get("path"), "figure.path"))
                path = path if path.is_absolute() else base / path
                _string(block.get("caption"), "figure.caption")
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
                    raise ValueError(f"figure must be a bitmap: {path}")
                with Image.open(path) as image:
                    image.verify()
            elif kind == "equation":
                latex = _string(block.get("latex"), "equation.latex").strip()
                if latex.startswith(("$", r"\[", r"\begin{document}")):
                    raise ValueError("equation.latex must omit outer math/document delimiters")
            elif kind == "table":
                columns = _strings(block.get("columns"), "table.columns")
                rows = block.get("rows")
                if not columns or not isinstance(rows, list):
                    raise ValueError("table requires nonempty columns and a rows list")
                for row in rows:
                    if not isinstance(row, list) or len(row) != len(columns):
                        raise ValueError("each table row must match the column count")
                    if any(not isinstance(v, (str, int, float, bool, type(None))) for v in row):
                        raise ValueError("table cells must be scalar values")
                    if any(isinstance(v, float) and not math.isfinite(v) for v in row):
                        raise ValueError("table cells must use explicit labels for unavailable values")
                widths = block.get("widths", [1] * len(columns))
                if not isinstance(widths, list) or len(widths) != len(columns) or any(
                    not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in widths
                ):
                    raise ValueError("table.widths must contain one positive weight per column")
    return dict(counts)


def _cell(value):
    return "—" if value is None else str(value)


def wrap_text(font, text, width, size):
    """Wrap CJK, ordinary words and arbitrarily long paths by actual glyph width."""
    if width < size:
        raise ValueError("text column is narrower than one glyph")
    result = []
    for paragraph in str(text).split("\n"):
        if not paragraph:
            result.append("")
            continue
        line, used = "", 0.0
        for token in re.findall(r"[A-Za-z0-9_]+|[^\S\n]+|.", paragraph):
            token = " " if token.isspace() else token
            token_width = font.text_length(token, fontsize=size)
            if token_width <= width:
                if line and used + token_width > width + 1e-6:
                    result.append(line.rstrip())
                    line, used = "", 0.0
                if not line and token == " ":
                    continue
                line += token
                used += token_width
            else:
                for char in token:
                    char_width = font.text_length(char, fontsize=size)
                    if line and used + char_width > width + 1e-6:
                        result.append(line.rstrip())
                        line, used = "", 0.0
                    line += char
                    used += char_width
        if line:
            result.append(line.rstrip())
    return result or [""]


class _FlowPDF:
    """Small report-specific paginator; text and tables remain PDF text objects."""

    def __init__(self, title, font, date=""):
        import fitz

        self.doc = fitz.open()
        self.font, self.title, self.date = font, title, date
        self.page = None
        self.y = BODY_TOP
        self.toc = []
        self.drawn_lines = []
        self.missing_glyphs = set()

    def new_page(self):
        import fitz

        self.page = self.doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        self.y = BODY_TOP
        header = wrap_text(self.font, self.title, BODY_WIDTH, 8.5)[0]
        if len(header) < len(self.title):
            header = header[:-1] + "…"
        self._draw_lines([header], MARGIN, 22, 8.5, 13, MUTED)
        self.page.draw_line(fitz.Point(MARGIN, 43), fitz.Point(PAGE_WIDTH - MARGIN, 43),
                            color=(.80, .84, .88), width=.5)
        return self.page

    def ensure(self, height):
        if self.page is None or self.y + height > BODY_BOTTOM + .01:
            self.new_page()

    def _draw_lines(self, lines, x, y, size, leading, color=INK):
        import fitz

        writer = fitz.TextWriter(self.page.rect)
        for index, line in enumerate(lines):
            for char in line:
                if not char.isspace() and not self.font.has_glyph(ord(char)):
                    self.missing_glyphs.add(char)
            baseline = y + self.font.ascender * size + index * leading
            writer.append((x, baseline), line, font=self.font, fontsize=size)
            self.drawn_lines.append(line)
            for match in re.finditer(r"https?://\S+", line):
                left = x + self.font.text_length(line[:match.start()], fontsize=size)
                right = left + self.font.text_length(match.group(), fontsize=size)
                self.page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(left, baseline - size, right, baseline + 3),
                                       "uri": match.group().rstrip("，。；）")})
        writer.write_text(self.page, color=color)

    def text(self, text, *, size=11, gap=9, color=INK, keep=0):
        lines = wrap_text(self.font, text, BODY_WIDTH, size)
        leading = size * 1.62
        self.ensure(min(len(lines), 2) * leading + keep)
        while lines:
            available = int((BODY_BOTTOM - self.y + .01) // leading)
            if available <= 0 or (available == 1 and len(lines) > 1):
                self.new_page()
                continue
            count = min(len(lines), available)
            if len(lines) - count == 1 and count > 2:
                count -= 1
            self._draw_lines(lines[:count], MARGIN, self.y, size, leading, color)
            self.y += count * leading
            lines = lines[count:]
            if lines:
                self.new_page()
        self.y += gap

    def heading(self, text, *, level=1, keep=35):
        size = 16 if level == 1 else 12.5
        self.ensure(len(wrap_text(self.font, text, BODY_WIDTH, size)) * size * 1.62 + 13 + keep)
        if level == 1:
            self.toc.append([1, text, len(self.doc)])
        self.text(text, size=size, gap=13, keep=keep)

    def image(self, path, width, height, caption, source="", *, label=""):
        import fitz

        caption_text = (label + "　" if label else "") + caption
        caption_lines = wrap_text(self.font, caption_text, BODY_WIDTH, 9.6) if caption else []
        source_lines = wrap_text(self.font, "来源：" + source, BODY_WIDTH, 8.2) if source else []
        required = (height + 9 + (len(caption_lines) * 9.6 * 1.62 + 7 if caption_lines else 0)
                    + (len(source_lines) * 8.2 * 1.62 + 10 if source_lines else 0))
        # Keep a complete figure, caption and provenance together whenever the
        # group fits one fresh page; checking only two caption lines leaves a
        # source path stranded at the beginning of the following page.
        self.ensure(required if required <= BODY_BOTTOM - BODY_TOP
                    else height + 12 + min(len(caption_lines), 2) * 15.6)
        left = MARGIN + (BODY_WIDTH - width) / 2
        self.page.insert_image(fitz.Rect(left, self.y, left + width, self.y + height), filename=str(path))
        self.y += height + 9
        if caption:
            self.text(caption_text, size=9.6, gap=7)
        if source:
            self.text("来源：" + source, size=8.2, gap=10, color=MUTED)

    def table_layout(self, block):
        """Measure the same table opening for pagination and its preceding title."""
        columns = block["columns"]
        weights = block.get("widths", [1] * len(columns))
        widths = [BODY_WIDTH * w / sum(weights) for w in weights]
        size, leading, pad = 9.2, 14.4, 5.0
        header = [wrap_text(self.font, c, w - 2 * pad, size) for c, w in zip(columns, widths)]
        header_height = max(map(len, header)) * leading + 2 * pad
        title = block.get("title", "")
        if header_height > (BODY_BOTTOM - BODY_TOP) / 2:
            raise ValueError("table header is too tall; shorten labels or revise widths")
        title_height = (len(wrap_text(self.font, title, BODY_WIDTH, 10.5)) * 10.5 * 1.62 + 7
                        if title else 0)
        fresh_room = BODY_BOTTOM - BODY_TOP - header_height - title_height
        first_row_height = leading + 2 * pad
        if block["rows"]:
            first_lines = [wrap_text(self.font, _cell(value), width - 2 * pad, size)
                           for value, width in zip(block["rows"][0], widths)]
            full_height = max(map(len, first_lines)) * leading + 2 * pad
            if full_height <= fresh_room:
                first_row_height = full_height
        return dict(widths=widths, size=size, leading=leading, pad=pad, header=header,
                    header_height=header_height, title=title, title_height=title_height,
                    fresh_room=fresh_room, first_row_height=first_row_height,
                    opening_height=header_height + title_height + first_row_height)

    def table(self, block):
        import fitz

        layout = self.table_layout(block)
        widths, size, leading, pad = (layout[k] for k in ("widths", "size", "leading", "pad"))
        header, header_height, title, title_height = (layout[k] for k in
                                                    ("header", "header_height", "title", "title_height"))
        fresh_room = layout["fresh_room"]

        def row(lines, height, is_header=False):
            left = MARGIN
            for cell_lines, width in zip(lines, widths):
                self.page.draw_rect(fitz.Rect(left, self.y, left + width, self.y + height),
                                    color=(.74, .79, .84), width=.45,
                                    fill=(.91, .94, .97) if is_header else None)
                self._draw_lines(cell_lines, left + pad, self.y + pad, size, leading)
                left += width
            self.y += height

        def start(continuation=False, first_row_height=leading + 2 * pad):
            self.ensure(header_height + first_row_height + title_height)
            if title:
                self.text(title + ("（续）" if continuation else ""), size=10.5, gap=7,
                          keep=header_height + first_row_height)
            row(header, header_height, True)

        start(first_row_height=layout["first_row_height"])
        for values in block["rows"]:
            lines = [wrap_text(self.font, _cell(value), width - 2 * pad, size)
                     for value, width in zip(values, widths)]
            required = max(map(len, lines)) * leading + 2 * pad
            if self.y + required > BODY_BOTTOM and required <= fresh_room:
                self.new_page()
                start(True)
            while any(lines):
                capacity = int((BODY_BOTTOM - self.y - 2 * pad + .01) // leading)
                if capacity < 1:
                    self.new_page()
                    start(True)
                    continue
                count = min(max(map(len, lines)), capacity)
                row([cell_lines[:count] for cell_lines in lines], count * leading + 2 * pad)
                lines = [cell_lines[count:] for cell_lines in lines]
                if any(lines):
                    self.new_page()
                    start(True)
        self.y += 8
        if block.get("caption"):
            self.text(block["caption"], size=9.6, gap=7)
        if block.get("source"):
            self.text("来源：" + block["source"], size=8.2, gap=11, color=MUTED)

    def save(self, path):
        import fitz

        if self.missing_glyphs:
            raise ValueError("report font lacks glyphs: " + " ".join(sorted(self.missing_glyphs)) +
                             "; render mathematical notation as LaTeX or use ordinary text labels")
        for index in range(len(self.doc)):
            self.page = self.doc[index]
            footer = f"{self.date}　证据来源见正文　|　配套 HTML"
            self._draw_lines([footer], MARGIN, PAGE_HEIGHT - 32, 7.3, 10, MUTED)
            start = MARGIN + self.font.text_length(footer[:-8], fontsize=7.3)
            self.page.insert_link({"kind": fitz.LINK_LAUNCH,
                                   "from": fitz.Rect(start, PAGE_HEIGHT - 33, MARGIN + 280, PAGE_HEIGHT - 16),
                                   "file": "TECHNICAL_REPORT.html"})
            number = f"{index + 1} / {len(self.doc)}"
            width = self.font.text_length(number, fontsize=8)
            self._draw_lines([number], PAGE_WIDTH - MARGIN - width, PAGE_HEIGHT - 32, 8, 10, MUTED)
        if self.toc:
            self.doc.set_toc(self.toc)
        self.doc.set_metadata({"title": self.title, "creator": "render_source_diagnostics_report.py",
                               "subject": "Retained source-diagnostics evidence; authored communication export"})
        self.doc.subset_fonts(fallback=True)
        self.doc.save(path, garbage=4, deflate=True)
        self.doc.close()


def _png(source, destination, width, height, dpi):
    """Normalize a complete bitmap at its final display resolution."""
    from PIL import Image

    pixels = (max(1, round(width * dpi / 72)), max(1, round(height * dpi / 72)))
    with Image.open(source) as original:
        original.load()
        image = original.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        image = background.convert("RGB")
        if image.size != pixels:
            image = image.resize(pixels, Image.Resampling.LANCZOS)
        image.save(destination, dpi=(dpi, dpi), optimize=True)
    return pixels


def _formula(block, output, index):
    import fitz

    latex = block["latex"]
    stem = output / "formulas" / f"equation_{index:02}"
    tex = stem.with_suffix(".tex")
    tex.write_text("\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n"
                   "\\usepackage[paperwidth=18in,paperheight=8in,margin=.25in]{geometry}\n"
                   "\\pagestyle{empty}\n\\begin{document}\n\\fontsize{22}{29}\\selectfont\n\\[\n"
                   + latex + "\n\\]\n\\end{document}\n", encoding="utf-8")
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required to render equation PNGs")
    result = subprocess.run([executable, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                            cwd=tex.parent, capture_output=True, text=True, timeout=60)
    (stem.with_suffix(".render.log")).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise ValueError(f"LaTeX rendering failed for {tex.name}: {result.stdout[-1800:]}")
    with fitz.open(stem.with_suffix(".pdf")) as document:
        page = document[0]
        boxes = [fitz.Rect(entry[1]) for entry in page.get_bboxlog() if entry[0] != "ignore-text"]
        if not boxes or len(document) != 1:
            raise ValueError(f"equation must render to one nonempty page: {tex.name}")
        clip = boxes[0]
        for box in boxes[1:]:
            clip |= box
        clip += (-7, -7, 7, 7)
        if not page.rect.contains(clip):
            raise ValueError(f"equation exceeds its LaTeX canvas: {tex.name}")
        scale = min(14 / 22, BODY_WIDTH / clip.width, 350 / clip.height)
        if 22 * scale < 10.5:
            raise ValueError(f"equation is too long for readable A4 output; use aligned or split it: {tex.name}")
        width, height = clip.width * scale, clip.height * scale
        zoom = FORMULA_DPI * scale / 72
        page.get_pixmap(clip=clip, matrix=fitz.Matrix(zoom, zoom), alpha=False).save(stem.with_suffix(".png"))
    return stem.with_suffix(".png"), width, height, tex


def _html_text(text):
    return html.escape(str(text)).replace("\n", "<br>")


def _source_html(source):
    return '<p class="source">来源：' + _html_text(source) + "</p>" if source else ""


def _audit_pdf(path, expected_images, preview=None):
    import fitz

    outside, image_sizes, chars = [], [], 0
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            chars += len(page.get_text())
            for image in page.get_image_info():
                box = fitz.Rect(image["bbox"])
                image_sizes.append({"page": index + 1, "width_pixels": image["width"],
                                    "height_pixels": image["height"], "bbox": list(box),
                                    "effective_dpi_x": image["width"] * 72 / box.width,
                                    "effective_dpi_y": image["height"] * 72 / box.height})
                if not page.rect.contains(box):
                    raise ValueError(f"image extends outside page: {path.name} page {index + 1}")
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        box = fitz.Rect(span["bbox"])
                        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > page.rect.width + .5 or box.y1 > page.rect.height + .5:
                            outside.append({"page": index + 1, "text": span["text"], "bbox": list(box)})
            if preview is not None:
                page.get_pixmap(dpi=100, alpha=False).save(preview / f"page_{index + 1:03}.png")
        audit = {"pages": len(document), "image_objects": len({image[0] for page in document for image in page.get_images()}),
                 "displayed_images": len(image_sizes), "selectable_characters": chars,
                 "text_outside_page": outside, "image_placements": image_sizes,
                 "full_page_raster_images": sum((image["bbox"][2] - image["bbox"][0]) *
                     (image["bbox"][3] - image["bbox"][1]) > .9 * PAGE_WIDTH * PAGE_HEIGHT for image in image_sizes)}
    if outside or len(image_sizes) != expected_images or not chars or audit["full_page_raster_images"]:
        raise ValueError(f"PDF delivery checks failed: {path.name}: {audit}")
    return audit


def render_report(content_path, output):
    """Render a standalone, portable report from explicitly supplied content."""
    import fitz
    from PIL import Image

    content_path, output = Path(content_path).resolve(), Path(output).resolve()
    content = json.loads(content_path.read_text(encoding="utf-8"))
    counts = validate_content(content, content_path.parent)
    if output.exists():
        raise FileExistsError(f"report output must be fresh; retained exports are never overwritten: {output}")
    font = fitz.Font(FONT_NAME)
    output.mkdir(parents=True)
    for directory in ("figures", "formulas", "preview"):
        (output / directory).mkdir()
    report = _FlowPDF(content["title"], font, content.get("date", ""))
    report.text(content["title"], size=23, gap=18, keep=50)
    if content.get("subtitle"):
        report.text(content["subtitle"], size=12, gap=12)
    if content.get("date"):
        report.text(content["date"], size=10, gap=15, color=MUTED)
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>' + _html_text(content["title"]) + '</title><style>',
             'body{max-width:960px;margin:40px auto;padding:0 24px;color:#18324d;font:18px/1.75 "Noto Sans CJK SC",sans-serif}'
             'h1,h2,h3{line-height:1.4}h2{margin-top:2.5em}p,td,th{overflow-wrap:anywhere}figure{margin:1.7em 0}'
             'img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:16px;line-height:1.65;margin-top:12px}'
             '.source,.provenance{font-size:14px;color:#52616f}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;'
             'table-layout:fixed;font-size:15px;line-height:1.5}th,td{padding:9px;border:1px solid #bbc8d4;vertical-align:top}'
             'th{background:#e8eff6}caption{text-align:left;font-weight:bold;margin:10px 0}.equation img{max-width:100%}'
             '@media print{body{margin:0;font-size:11pt}h2{break-after:avoid}figure{break-inside:avoid}thead{display:table-header-group}}',
             '</style></head><body><h1>' + _html_text(content["title"]) + '</h1>']
    if content.get("subtitle"):
        parts.append('<p>' + _html_text(content["subtitle"]) + '</p>')
    if content.get("date"):
        parts.append('<p class="provenance">' + _html_text(content["date"]) + '</p>')
    for provenance in content.get("provenance", []):
        report.text(provenance, size=9.2, gap=9, color=MUTED)
        parts.append('<p class="provenance">' + _html_text(provenance) + '</p>')
    if content.get("summary"):
        report.heading("研究摘要")
        parts.append('<h2>研究摘要</h2>')
    for paragraph in content.get("summary", []):
        report.text(paragraph)
        parts.append('<p>' + _html_text(paragraph) + '</p>')
    assets, portable = [], json.loads(json.dumps(content, ensure_ascii=False))
    figure_index, equation_index = 0, 0
    for section, portable_section in zip(content["sections"], portable["sections"]):
        blocks, portable_blocks = section_blocks(section), section_blocks(portable_section)
        if section.get("page_break") and report.page is not None:
            report.new_page()
        keep = report.table_layout(blocks[0])["opening_height"] if blocks and blocks[0]["type"] == "table" else 35
        report.heading(section["title"], keep=keep)
        parts.append('<section><h2>' + _html_text(section["title"]) + '</h2>')
        for index, (block, portable_block) in enumerate(zip(blocks, portable_blocks)):
            kind = block["type"]
            if kind in ("paragraph", "heading"):
                if kind == "heading":
                    following = blocks[index + 1] if index + 1 < len(blocks) else None
                    keep = (report.table_layout(following)["opening_height"]
                            if following and following["type"] == "table" else 35)
                    report.heading(block["text"], level=2, keep=keep)
                    parts.append('<h3>' + _html_text(block["text"]) + '</h3>')
                else:
                    report.text(block["text"])
                    parts.append('<p>' + _html_text(block["text"]) + '</p>')
            elif kind in ("figure", "equation"):
                if kind == "figure":
                    figure_index += 1
                    source_path = Path(block["path"])
                    source_path = source_path if source_path.is_absolute() else content_path.parent / source_path
                    path = output / "figures" / f"figure_{figure_index:02}.png"
                    with Image.open(source_path) as image:
                        ratio = image.height / image.width
                        original = image.size
                    width = min(BODY_WIDTH, 540 / ratio)
                    height = width * ratio
                    pixels = _png(source_path, path, width, height, FIGURE_DPI)
                    label = block.get("label", f"图 {figure_index}")
                    extra = {"original_path": str(source_path.resolve()), "original_pixels": list(original),
                             "pixels": list(pixels), "dpi": FIGURE_DPI}
                    portable_block["path"] = path.relative_to(output).as_posix()
                else:
                    equation_index += 1
                    path, width, height, tex = _formula(block, output, equation_index)
                    label = block.get("label", f"式 {equation_index}")
                    extra = {"latex_source": tex.relative_to(output).as_posix(), "dpi": FORMULA_DPI}
                caption, source = block.get("caption", ""), block.get("source", "")
                report.image(path, width, height, caption, source, label=label)
                relative = path.relative_to(output).as_posix()
                assets.append({"kind": kind, "path": relative, "width_points": width, "height_points": height,
                               "label": label, "caption": caption, "source": source, **extra})
                parts.append('<figure class="' + kind + '"><img src="' + html.escape(relative, quote=True) +
                             '" style="width:' + f'{width * 96 / 72:.1f}' + 'px" alt="' +
                             html.escape(label + ("：" + caption if caption else "：数学公式"), quote=True) + '">')
                if caption:
                    parts.append('<figcaption>' + _html_text(label + "　" + caption) + '</figcaption>')
                parts.append(_source_html(source))
                if kind == "equation":
                    parts.append('<p class="source"><a href="' + extra["latex_source"] + '">LaTeX 源文件</a></p>')
                parts.append('</figure>')
                if block.get("explanation"):
                    report.text(block["explanation"])
                    parts.append('<p>' + _html_text(block["explanation"]) + '</p>')
            elif kind == "table":
                report.table(block)
                parts.append('<div class="table-wrap"><table>')
                if block.get("title"):
                    parts.append('<caption>' + _html_text(block["title"]) + '</caption>')
                weights = block.get("widths", [1] * len(block["columns"]))
                parts.append('<colgroup>' + ''.join('<col style="width:' + f'{100*w/sum(weights):.3f}' + '%">' for w in weights) + '</colgroup>')
                parts.append('<thead><tr>' + ''.join('<th>' + _html_text(c) + '</th>' for c in block["columns"]) + '</tr></thead><tbody>')
                for row in block["rows"]:
                    parts.append('<tr>' + ''.join('<td>' + _html_text(_cell(v)) + '</td>' for v in row) + '</tr>')
                parts.append('</tbody></table></div>')
                if block.get("caption"):
                    parts.append('<p>' + _html_text(block["caption"]) + '</p>')
                parts.append(_source_html(block.get("source", "")))
        if "blocks" in portable_section:
            portable_section["blocks"] = portable_blocks
        else:
            for plural, kind in (("equations", "equation"), ("figures", "figure"), ("tables", "table")):
                if plural in portable_section:
                    portable_section[plural] = [{k: v for k, v in block.items() if k != "type"}
                                                for block in portable_blocks if block["type"] == kind]
        parts.append('</section>')
    if content.get("sources"):
        report.heading("证据与实现入口")
        parts.append('<section><h2>证据与实现入口</h2>')
        for source in content["sources"]:
            report.text(source, size=9, gap=10, color=MUTED)
            parts.append('<p class="source">' + _html_text(source) + '</p>')
        parts.append('</section>')
    report.save(output / "TECHNICAL_REPORT.pdf")
    parts.append('</body></html>')
    (output / "TECHNICAL_REPORT.html").write_text("\n".join(parts) + "\n", encoding="utf-8")
    portable["schema"] = SCHEMA
    (output / "content.json").write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atlas = _FlowPDF(content["title"] + "：图与公式索引", font, content.get("date", ""))
    if not assets:
        atlas.text("本报告没有图像或公式。")
    for asset in assets:
        atlas.new_page()
        atlas.heading(asset["label"])
        atlas.image(output / asset["path"], asset["width_points"], asset["height_points"], asset["caption"], asset["source"])
    atlas.save(output / "FIGURE_ATLAS.pdf")
    checks = {"schema": SCHEMA, "content_source": str(content_path), "output": str(output),
              "kind": "authored_communication_export", "block_counts": counts,
              "bitmap_figures": True, "figure_dpi": FIGURE_DPI, "formula_dpi": FORMULA_DPI,
              "body_and_tables_selectable": True, "assets": assets,
              "exports": {"TECHNICAL_REPORT.pdf": _audit_pdf(output / "TECHNICAL_REPORT.pdf", len(assets), output / "preview"),
                          "FIGURE_ATLAS.pdf": _audit_pdf(output / "FIGURE_ATLAS.pdf", len(assets))},
              "visual_review": "Previews must be reviewed at actual display size for label readability and overlap."}
    (output / "delivery_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    checks = render_report(args.content, args.output)
    print(json.dumps({"output": checks["output"], "pages": checks["exports"]["TECHNICAL_REPORT.pdf"]["pages"],
                      "figures": checks["block_counts"].get("figure", 0),
                      "equations": checks["block_counts"].get("equation", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
