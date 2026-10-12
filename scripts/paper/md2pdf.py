# -*- coding: utf-8 -*-
"""通用 md → HTML → PDF（Edge 无头）小工具，用于生成索引/校验类报告。"""
import os, re, subprocess, sys, html as _html

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 17mm 16mm 17mm; }
body { font-family: "Microsoft YaHei","SimSun",serif; font-size: 10.5pt; line-height: 1.7;
       color: #111; max-width: 900px; margin: 0 auto; padding: 0 10px; }
h1 { font-size: 17pt; text-align: center; margin: 6px 0 14px; }
h2 { font-size: 13pt; margin: 20px 0 8px; border-left: 4px solid #A32D2D; padding-left: 8px; }
h3 { font-size: 11.5pt; margin: 14px 0 6px; }
p { margin: 6px 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; line-height: 1.45; vertical-align: top; }
th { background: #F2F2F2; font-weight: 600; text-align: center; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #F6F6F6; padding: 0 3px; }
ul { margin: 4px 0 8px 0; padding-left: 22px; }
li { margin: 3px 0; }
.ok { color: #1a7f37; font-weight: 600; }
.small { font-size: 9pt; color: #444; }
"""


def inline(s):
    s = _html.escape(s, quote=False)
    s = s.replace("\\|", "|")
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s


def conv(md):
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1; continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", raw)]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if s.startswith("### "):
            out.append(f"<h3>{inline(s[4:])}</h3>"); i += 1; continue
        if s.startswith("## "):
            out.append(f"<h2>{inline(s[3:])}</h2>"); i += 1; continue
        if s.startswith("# "):
            out.append(f"<h1>{inline(s[2:])}</h1>"); i += 1; continue
        if s.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{inline(lines[i].strip()[2:])}</li>"); i += 1
            out.append("<ul>" + "".join(items) + "</ul>"); continue
        out.append(f"<p>{inline(s)}</p>"); i += 1
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>" + "\n".join(out) + "</body></html>")


def build(md_path, out_pdf, out_html=None):
    md = open(md_path, encoding="utf-8").read()
    html = conv(md)
    out_html = out_html or (os.path.splitext(out_pdf)[0] + ".html")
    open(out_html, "w", encoding="utf-8").write(html)
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--user-data-dir=" + os.path.join(HERE, ".edge_tmp"),
                    f"--print-to-pdf={out_pdf}",
                    "file:///" + out_html.replace("\\", "/").replace(" ", "%20")],
                   check=False, timeout=300)
    print("PDF ->", out_pdf, os.path.exists(out_pdf))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
