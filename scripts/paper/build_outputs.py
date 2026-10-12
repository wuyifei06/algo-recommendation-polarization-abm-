# -*- coding: utf-8 -*-
"""把 paper_revised.md 转为 (1) 排版良好的 HTML (2) 由 Edge 无头打印的 PDF (3) .docx。"""
import os, re, json, subprocess, sys, html as _html

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper_revised.md")
FIGS = os.path.join(HERE, "figs")
OUT_HTML = os.path.join(HERE, "paper_revised.html")
OUT_PDF = os.path.join(HERE, "算法推荐、信息暴露与情绪如何塑造舆论极化_修订版.pdf")
OUT_DOCX = os.path.join(HERE, "算法推荐、信息暴露与情绪如何塑造舆论极化_修订版.docx")
TOC_JSON = os.path.join(HERE, "toc_pages.json")
TOC_META = os.path.join(HERE, "toc_meta.json")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

FIGMAP = {
    "1": "fig1_rule_timeseries.png", "2": "fig2_llm_timeseries.png",
    "3": "fig3_rule_vs_llm_normalized.png", "4": "fig4_alpha_dose_response.png",
    "5": "fig5_poly_fit.png", "6": "fig6_mediation_path.png",
    "7": "fig7_robustness.png", "8": "fig8_crossmodel.png",
}


CITE_RX = re.compile(r"\[(\d{1,2}(?:\s*[-,，]\s*\d{1,2})*)\]")


def _valid_cite(seg: str) -> bool:
    """文献引用校验：全部编号为 1..42 的正整数（排除 [0,2] 区间、[−0.688,…] 置信区间等）。"""
    nums = re.findall(r"\d+", seg)
    return bool(nums) and all(1 <= int(x) <= 42 for x in nums)


def cite_segments(text: str):
    """把文本切分为 (片段, 是否引用标注) 序列；引用标注渲染为上标。"""
    out = []
    for part in re.split(r"(\*\*.+?\*\*)", text):
        bold = part.startswith("**") and part.endswith("**") and len(part) > 4
        core = part[2:-2] if bold else part
        for seg in CITE_RX.split(core):
            if not seg:
                continue
            out.append((seg, CITE_RX.fullmatch(seg) is not None and _valid_cite(seg), bold))
    return out


def inline(s, sup=True):
    s = _html.escape(s, quote=False)
    s = s.replace("\\|", "|")  # markdown 转义的竖线（表格单元格内）
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    if sup:
        def _sup(m):
            return f"<sup>[{m.group(1)}]</sup>" if _valid_cite(m.group(1)) else m.group(0)
        s = CITE_RX.sub(_sup, s)
    return s


def parse(md):
    """返回 block 列表: ('h',level,text) ('p',text) ('table',rows) ('fig',n,caption)"""
    lines = md.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1; continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip()
                if raw.startswith("|"):
                    raw = raw[1:]
                if raw.endswith("|"):
                    raw = raw[:-1]
                # 支持 \| 转义：只按未转义的竖线分列，并把 \| 还原为 |
                cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", raw)]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows)); continue
        m = re.match(r"^图(\d)　(.*)$", s)
        if m and m.group(1) in FIGMAP:
            blocks.append(("fig", m.group(1), s)); i += 1; continue
        if s.startswith("### "):
            blocks.append(("h", 3, s[4:])); i += 1; continue
        if s.startswith("## "):
            blocks.append(("h", 2, s[3:])); i += 1; continue
        if s.startswith("# "):
            blocks.append(("h", 1, s[2:])); i += 1; continue
        if s in ("摘要", "Abstract"):
            blocks.append(("h", 2, s)); i += 1; continue
        blocks.append(("p", s)); i += 1
    return blocks


CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
body { font-family: "Microsoft YaHei","SimSun",serif; font-size: 10.5pt; line-height: 1.75;
       color: #000; max-width: 900px; margin: 0 auto; padding: 0 12px; }
h1 { font-size: 19pt; text-align: center; margin: 26px 0 6px; line-height: 1.5; }
h2 { font-size: 14pt; margin: 22px 0 10px; border-left: 4px solid #A32D2D; padding-left: 8px; }
h3 { font-size: 12pt; margin: 16px 0 8px; }
p  { margin: 8px 0; text-align: justify; text-indent: 2em; }
h1 + p, h2 + p, h3 + p, .figcap + p { }
table { border-collapse: collapse; width: 100%; margin: 10px 0 14px; font-size: 9.2pt; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: center; line-height: 1.5; }
th { background: #F2F2F2; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 12px 0 16px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
.figcap { font-size: 9.5pt; text-align: center; text-indent: 0; color: #222; margin-top: 4px; }
p.note { font-size: 9pt; text-indent: 0; color: #333; margin: 4px 0 10px; padding-left: 1.2em; }
p.eq { display: flex; align-items: baseline; text-indent: 0; margin: 8px 0; }
p.eq .eqbody { flex: 1; text-align: center; }
p.eq .eqnum { flex: 0 0 auto; }
.sub { text-align: center; font-size: 12pt; text-indent: 0; margin: 4px 0 18px; }
.toc { margin: 6px 0 10px; page-break-after: always; }
.toc h2 { border-left: none; text-align: center; padding-left: 0; font-size: 14pt; margin: 10px 0 14px; }
.tocline { display: flex; align-items: baseline; font-size: 10.5pt; text-indent: 0;
           margin: 3.5px 0; line-height: 1.5; }
.tocline.lv3 { padding-left: 22px; font-size: 10pt; }
.tocline .txt { white-space: nowrap; }
.tocline .dots { flex: 1; border-bottom: 1px dotted #999; margin: 0 6px 3px; }
h2, h3 { page-break-after: avoid; }
"""


def toc_entries(blocks):
    """目录条目：二级标题（摘要/Abstract/一~六/参考文献/附录）与三级标题。"""
    return [(b[2], b[1]) for b in blocks if b[0] == "h" and b[1] in (2, 3)]


def load_toc_pages():
    if os.path.exists(TOC_JSON):
        return json.load(open(TOC_JSON, encoding="utf-8"))
    return {}


def load_toc_meta():
    if os.path.exists(TOC_META):
        return json.load(open(TOC_META, encoding="utf-8"))
    return {}


def to_roman(n):
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    s = ""
    for v, r in vals:
        while n >= v:
            s += r; n -= v
    return s


def display_page(p, meta):
    """页码显示口径：封面不显示（计 i）；前置部分（目录/摘要/Abstract）罗马数字；
    正文自“一、引言”所在物理页起用阿拉伯数字，从 1 起。"""
    bs = meta.get("body_start")
    if not bs or not p:
        return str(p)
    p = int(p)
    if p == 1:
        return ""
    if p < bs:
        return to_roman(p)
    return str(p - bs + 1)


def render_toc_html(entries, pages):
    meta = load_toc_meta()
    rows = ['<div class="toc" style="page-break-before: always"><h2>目　录</h2>']
    for txt, lv in entries:
        pg = display_page(pages.get(txt, ""), meta)
        cls = "tocline" if lv == 2 else "tocline lv3"
        rows.append(f'<div class="{cls}"><span class="txt">{inline(txt)}</span>'
                    f'<span class="dots"></span><span class="pg">{pg}</span></div>')
    rows.append("</div>")
    return "\n".join(rows)


def to_html(blocks):
    pages = load_toc_pages()
    out, inserted = [], False
    in_refs = False
    for b in blocks:
        if b[0] == "h":
            lvl, t = b[1], b[2]
            if lvl == 2 and t.startswith("参考文献"):
                in_refs = True
            elif lvl == 2 and t.startswith("附录"):
                in_refs = False
            # 目录移至摘要/Abstract 之后、正文之前
            if lvl == 2 and t.startswith("一、引言") and not inserted:
                out.append(render_toc_html(toc_entries(blocks), pages))
                inserted = True
            if lvl == 2 and t.startswith("一、引言"):
                out.append(f'<h2 style="page-break-before: always">{inline(t)}</h2>')
            else:
                out.append(f"<h{lvl}>{inline(t)}</h{lvl}>")
        elif b[0] == "p":
            t = b[1]
            if t.startswith("——"):
                out.append(f'<p class="sub">{inline(t, sup=not in_refs)}</p>')
            elif t.startswith("注："):
                out.append(f'<p class="note">{inline(t, sup=not in_refs)}</p>')
            elif "=" in t and re.search(r"（\d+）\s*$", t):
                # 公式行：正文居中、编号右对齐
                m = re.search(r"^(.*?)\s*（(\d+)）\s*$", t)
                body_, num_ = (m.group(1), m.group(2)) if m else (t, "")
                out.append(f'<p class="eq"><span class="eqbody">{inline(body_, sup=not in_refs)}</span>'
                           f'<span class="eqnum">（{num_}）</span></p>')
            else:
                out.append(f"<p>{inline(t, sup=not in_refs)}</p>")
        elif b[0] == "table":
            rows = b[1]
            th = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")
        elif b[0] == "fig":
            n, cap = b[1], b[2]
            src = os.path.join("figs", FIGMAP[n]).replace("\\", "/")
            out.append(f'<figure><img src="{src}"/><div class="figcap">{inline(cap)}</div></figure>')
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>" + "\n".join(out) + "</body></html>")


def stamp_page_numbers(pdf_path, meta=None):
    """在 PDF 页脚居中标注页码：封面不显示；前置部分罗马数字；正文自 1 起阿拉伯数字。"""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    meta = meta if meta is not None else load_toc_meta()
    d = pymupdf.open(pdf_path)
    for i, page in enumerate(d):
        txt = display_page(i + 1, meta)
        if not txt:
            continue
        w = pymupdf.get_text_length(txt, fontname="helv", fontsize=9)
        x = (page.rect.width - w) / 2
        y = page.rect.height - 22
        page.insert_text((x, y), txt, fontname="helv", fontsize=9,
                         color=(0.35, 0.35, 0.35))
    d.save(pdf_path, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
    d.close()


def build_pdf():
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", "file:///" + OUT_HTML.replace("\\", "/").replace(" ", "%20")],
                   check=False, timeout=300)
    stamp_page_numbers(OUT_PDF)
    print("PDF ->", OUT_PDF, os.path.exists(OUT_PDF))


def build_docx(blocks):
    from docx import Document
    from docx.shared import Pt, Cm, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
    from docx.enum.section import WD_SECTION
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Microsoft YaHei"; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for s in doc.sections:
        s.top_margin = Cm(2.0); s.bottom_margin = Cm(2.0)
        s.left_margin = Cm(1.8); s.right_margin = Cm(1.8)

    def add_p(text, style=None, align=None, indent=True, sup=True):
        p = doc.add_paragraph(style=style)
        if align is not None:
            p.alignment = align
        if indent:
            p.paragraph_format.first_line_indent = Pt(21)
        # 处理 **bold** 与 文献引用上标
        for seg, is_cite, is_bold in cite_segments(text):
            r = p.add_run(seg)
            if is_bold:
                r.bold = True
            if sup and is_cite:
                r.font.superscript = True
        return p

    def add_toc_docx():
        """Word 域目录：显示缓存内容，并在打开时自动更新为 Word 自己的页码。"""
        pages = load_toc_pages()
        entries = toc_entries(blocks)
        doc.add_page_break()
        h = doc.add_paragraph(); h.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = h.add_run("目　录"); r.bold = True; r.font.size = Pt(14)
        paras = []
        for txt, lv in entries:
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.left_indent = Pt(0 if lv == 2 else 21)
            pf.first_line_indent = Pt(0)
            pf.space_after = Pt(2)
            pf.tab_stops.add_tab_stop(Cm(16.4), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            p.add_run(txt)
            p.add_run("\t" + display_page(pages.get(txt, ""), load_toc_meta()))
            paras.append(p)
        if paras:
            first, last = paras[0], paras[-1]
            b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
            b.set(qn("w:dirty"), "true")
            instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
            instr.text = 'TOC \\o "1-3" \\h \\z \\u'
            sep = OxmlElement("w:fldChar"); sep.set(qn("w:fldCharType"), "separate")
            end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
            first.runs[0]._r.addprevious(b)
            first.runs[0]._r.addprevious(instr)
            first.runs[0]._r.addprevious(sep)
            last.runs[-1]._r.append(end)
        doc.add_paragraph()

    def set_pgnum(section, fmt=None, start=None):
        """设置节的页码格式（w:pgNumType）。"""
        sectPr = section._sectPr
        pg = sectPr.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType"); sectPr.append(pg)
        if fmt:
            pg.set(qn("w:fmt"), fmt)
        if start is not None:
            pg.set(qn("w:start"), str(start))

    def add_page_numbers_docx():
        """页脚居中 PAGE 域（封面不显示）；前置部分 roman、正文 decimal 自 1 起。"""
        from docx.enum.text import WD_ALIGN_PARAGRAPH as _A
        first = doc.sections[0]
        first.different_first_page_header_footer = True
        set_pgnum(first, fmt="lowerRoman", start=1)
        for s in doc.sections:
            footer = s.footer
            p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
            p.alignment = _A.CENTER
            run = p.add_run()
            b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin")
            instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
            instr.text = "PAGE"
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end")
            run._r.append(b); run._r.append(instr); run._r.append(e)

    in_refs = False
    toc_done = False
    for b in blocks:
        if b[0] == "h":
            lvl, t = b[1], b[2]
            if lvl == 2 and t.startswith("参考文献"):
                in_refs = True
            elif lvl == 2 and t.startswith("附录"):
                in_refs = False
            # 目录移至摘要/Abstract 之后、正文之前
            if lvl == 2 and t.startswith("一、引言") and not toc_done:
                add_toc_docx()
                toc_done = True
            if lvl == 1:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(t); r.bold = True; r.font.size = Pt(17)
            elif lvl == 2 and t.startswith("一、引言"):
                sec = doc.add_section(WD_SECTION.NEW_PAGE)
                set_pgnum(sec, fmt="decimal", start=1)
                sec.different_first_page_header_footer = False
                doc.add_heading(t, level=lvl)
            else:
                doc.add_heading(t, level=lvl)
        elif b[0] == "p":
            t = b[1]
            if t.startswith("——"):
                add_p(t, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, sup=not in_refs)
            elif t.startswith("注："):
                p = add_p(t, indent=False, sup=not in_refs)
                for run in p.runs:
                    run.font.size = Pt(9)
            elif "=" in t and re.search(r"（\d+）\s*$", t):
                m = re.search(r"^(.*?)\s*（(\d+)）\s*$", t)
                body_, num_ = (m.group(1), m.group(2)) if m else (t, "")
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.first_line_indent = Pt(0)
                pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf.tab_stops.add_tab_stop(Cm(8.2), WD_TAB_ALIGNMENT.CENTER, WD_TAB_LEADER.SPACES)
                pf.tab_stops.add_tab_stop(Cm(16.4), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.SPACES)
                p.add_run("\t" + body_)
                p.add_run("\t（" + num_ + "）")
            else:
                add_p(t, sup=not in_refs)
        elif b[0] == "table":
            rows = b[1]
            tb = doc.add_table(rows=0, cols=len(rows[0]))
            tb.style = "Table Grid"
            for ri, r in enumerate(rows):
                cells = tb.add_row().cells
                for ci, c in enumerate(r[:len(rows[0])]):
                    cells[ci].text = c
                    for par in cells[ci].paragraphs:
                        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for run in par.runs:
                            run.font.size = Pt(8.5)
                            if ri == 0:
                                run.bold = True
        elif b[0] == "fig":
            n, cap = b[1], b[2]
            doc.add_picture(os.path.join(FIGS, FIGMAP[n]), width=Inches(6.1))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_p(cap, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    add_page_numbers_docx()
    doc.save(OUT_DOCX)
    print("DOCX ->", OUT_DOCX, os.path.exists(OUT_DOCX))


if __name__ == "__main__":
    md = open(MD, encoding="utf-8").read()
    blocks = parse(md)
    open(OUT_HTML, "w", encoding="utf-8").write(to_html(blocks))
    print("HTML ->", OUT_HTML, "blocks:", len(blocks))
    build_pdf()
    build_docx(blocks)
