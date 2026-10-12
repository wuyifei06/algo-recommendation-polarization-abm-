# -*- coding: utf-8 -*-
"""论文最终全量索引/格式复检 v2（修正正则误报）。"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper_revised.md")
HTML = os.path.join(HERE, "paper_revised.html")
PDF = os.path.join(HERE, "算法推荐、信息暴露与情绪如何塑造舆论极化_修订版.pdf")

md = open(MD, encoding="utf-8").read()
lines = md.split("\n")
issues, info = [], {}


def add(level, msg):
    issues.append((level, msg))


REF_HEAD = md.find("## 参考文献")
APP_HEAD = md.find("## 附录 A")
BODY = md[:REF_HEAD]          # 正文（含摘要）
APPENDIX = md[APP_HEAD:]

# ---------- 1. 结构 ----------
h2s = [l.strip()[3:].strip() for l in lines if l.strip().startswith("## ")]
info["h2"] = h2s
for k in ["一、引言", "二、文献综述与研究框架", "三、研究方法", "四、研究结果",
          "五、讨论", "六、结论", "参考文献"]:
    if k not in h2s:
        add("ERROR", f"缺少二级标题：{k}")
if "摘要" not in md.split("\n")[:20] and "## 摘要" not in md:
    add("ERROR", "缺少中文摘要标题")
if "Abstract" not in md.split("\n")[:30] and "## Abstract" not in md:
    add("ERROR", "缺少英文摘要标题")
for k in ["摘要", "Abstract"]:
    idx = [i for i, l in enumerate(lines) if l.strip() == k]
    if not idx and f"## {k}" not in h2s:
        add("ERROR", f"标题缺失：{k}")
info["h3_count"] = sum(1 for l in lines if l.strip().startswith("### "))

# ---------- 2. 图 ----------
fig_caps = [int(x) for x in re.findall(r"^图(\d+)[　\s]", md, re.M)]
info["fig_caps"] = sorted(set(fig_caps))
if sorted(set(fig_caps)) != list(range(1, 9)):
    add("ERROR", f"图注编号不连续/缺失：{sorted(set(fig_caps))}")
if len(fig_caps) != 8:
    add("ERROR", f"图注行数 {len(fig_caps)}（应为 8）")
for n in range(1, 9):
    body_hits = len(re.findall(rf"图{n}(?![0-9])", BODY))
    app_hits = len(re.findall(rf"图{n}(?![0-9])", APPENDIX))
    if body_hits == 0 and app_hits == 0:
        add("ERROR", f"图{n} 未被任何位置引用")
    info[f"fig{n}_ref"] = body_hits

# ---------- 3. 表 ----------
tab_caps = re.findall(r"^表([0-9A-Z]+)[　\s]", md, re.M)
info["tab_caps"] = tab_caps
main_tabs = sorted(int(x) for x in tab_caps if x.isdigit())
if main_tabs != list(range(1, 9)):
    add("ERROR", f"正表编号不连续：{main_tabs}")
app_tabs = [x for x in tab_caps if not x.isdigit()]
info["app_tabs"] = sorted(set(app_tabs))
for x in sorted(set(app_tabs)):
    if x not in ("A1", "B1", "C1"):
        add("WARN", f"出现未预期的附录表号：{x}")
for n in range(1, 8):
    if not re.search(rf"表{n}(?![0-9])", BODY):
        add("ERROR", f"表{n} 未在正文引用")

# ---------- 4. 参考文献闭环 ----------
ref_sec = md[REF_HEAD:APP_HEAD]
ref_nums = [int(x) for x in re.findall(r"^\[(\d+)\]", ref_sec, re.M)]
info["ref_count"] = len(ref_nums)
if ref_nums != list(range(1, len(ref_nums) + 1)):
    add("ERROR", f"参考文献编号不连续：{ref_nums}")

# 正文引用（排除区间写法：表格行内、∈ 语境、落在/限定在 语境）
cited = set()
for m in re.finditer(r"\[\d+(?:[-,]\d+)*\]", BODY):
    line_start = BODY.rfind("\n", 0, m.start()) + 1
    line_txt = BODY[line_start:m.start()]
    if line_txt.lstrip().startswith("|"):
        continue
    pre = BODY[max(0, m.start() - 30):m.start()]
    if "∈" in pre or "落在" in pre[-8:] or "限定在" in pre[-10:]:
        continue
    tok = m.group(0)[1:-1]
    if any(p.strip() == "0" for p in re.split(r"[-,]", tok)):
        continue          # [0,1] / [0,2] 等值域区间，非引用
    for part in tok.split(","):
        if "-" in part:
            a, b = part.split("-")[:2]
            cited.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            cited.add(int(part))
info["cited_count"] = len(cited)
unused = sorted(set(ref_nums) - cited)
over = sorted(cited - set(ref_nums))
if unused:
    add("ERROR", f"文献表中有未被正文引用的条目：{unused}")
if over:
    add("ERROR", f"正文引用了不存在的文献号：{over}")

# ---------- 5. 附录交叉引用 ----------
app_items = sorted(set(re.findall(r"^#{2,3}\s*A(\d)[　\s]", APPENDIX, re.M)))
info["appendix_items"] = app_items
if app_items != ["1", "2", "3", "4"]:
    add("ERROR", f"附录 A 分项异常：{app_items}")
for L in ["A", "B", "C"]:
    n_body = len(re.findall(rf"附录\s*{L}(?![\u4e00-\u9fa5])", BODY))
    n_all = len(re.findall(rf"附录\s*{L}(?![\u4e00-\u9fa5])", md))
    info[f"ref_{L}"] = n_body
    if n_body < 1:
        add("ERROR", f"附录 {L} 在正文中未被引用")
for i in ["A1", "A2", "A3", "A4"]:
    if not re.search(rf"附录\s*{i}|{i}\s*(报告|中|对此|纳入)", BODY):
        add("ERROR", f"附录 {i} 在正文中缺交叉引用")

# ---------- 6. 公式编号（只认行末公式号，排除“设计要点：（1）…”这类句内列号）----------
eqs = sorted(set(re.findall(r"[（(](\d)[）)]\s*$", BODY, re.M)))
info["eqs"] = eqs
for n in "12345":
    if n not in eqs:
        add("ERROR", f"公式编号（{n}）缺失")
extra_eq = [e for e in eqs if e not in "12345"]
if extra_eq:
    add("ERROR", f"多余公式号：{extra_eq}")

# ---------- 7. 禁用词/占位符 ----------
banned = ["旧实验", "旧版本", "作废", "废弃实验", "上一版实验", "此前实验", "原实验",
          "第一轮实验", "第二轮实验", "原始实验结果", "TODO", "XXX", "待补", "待填", "???",
          "placeholder", "TBD", "（略）", "此处略", "（空）", "省略不表",
          "主要驱动", "重要驱动", "主要载体", "完全传递", "决定性", "充分证明"]
for w in banned:
    n = md.count(w)
    if n:
        add("ERROR", f"禁用词命中：{w} × {n}")
for pat in [r"\{\{\s*待", r"\[\s*待\s*\]", r"<待[^>]*>"]:
    hits = re.findall(pat, md)
    if hits:
        add("ERROR", f"占位符：{hits[:6]}")

# ---------- 8. HTML ----------
html = open(HTML, encoding="utf-8").read()
tables = re.findall(r"<table>.*?</table>", html, re.S)
info["html_tables"] = len(tables)
for ti, t in enumerate(tables, 1):
    trs = re.findall(r"<tr>(.*?)</tr>", t, re.S)
    counts = [len(re.findall(r"<t[hd]>", tr)) for tr in trs]
    if len(set(counts)) != 1:
        add("ERROR", f"HTML 第 {ti} 个表列数不一致：{counts}")
    if any(c == 0 for c in counts):
        add("ERROR", f"HTML 第 {ti} 个表存在空行")
if len(tables) != len(tab_caps):
    add("ERROR", f"HTML 表格数 {len(tables)} ≠ markdown 表注数 {len(tab_caps)}")
if "\\|" in html:
    add("ERROR", "HTML 残留转义竖线 \\|")
imgs = re.findall(r'<img src="([^"]+)"', html)
info["html_imgs"] = len(imgs)
if len(imgs) != 8:
    add("ERROR", f"HTML 图片数 {len(imgs)}（应为 8）")
for s in imgs:
    if not os.path.exists(os.path.join(HERE, s.replace("/", os.sep))):
        add("ERROR", f"HTML 图片不存在：{s}")
# 图注与图片一一对应
caps_in_html = re.findall(r'<div class="figcap">(图\d+)', html)
if len(caps_in_html) != 8:
    add("ERROR", f"HTML 图注数 {len(caps_in_html)}")
info["html_figcaps"] = caps_in_html
if re.search(r"<td>[^<]*<strong>[^<]*</strong>[^<]*</td>", html):
    pass

# ---------- 9. 关键数字一致性 ----------
checks = [
    ("0.946", "延长档 HE t=30（与主实验 HE30=0.9458 三位小数一致）"),
    ("0.975", "延长档 HE t=100"),
    ("0.417", "延长档 PI t=100"),
    ("0.311", "延长档 PI t=30"),
    ("0.460", "延长档 PI t=60"),
    ("+0.232", "延长档 BC100 pooled β"),
    ("0.003", "延长档 BC100 p"),
    ("−0.144", "主实验 BC30 pooled β"),
    ("0.566", "Rule 主实验标准化 β"),
    ("0.405", "LLM 主实验 γ=1.0 斜率的置信区间基准"),
]
for val, desc in checks:
    if val not in md:
        add("WARN", f"关键数字 {val}（{desc}）未出现")

# ---------- 10. PDF ----------
info["pdf_exists"] = os.path.exists(PDF)
if not info["pdf_exists"]:
    add("ERROR", "PDF 未生成")
else:
    try:
        import fitz
        d = fitz.open(PDF)
        info["pdf_pages"] = d.page_count
        nimg = sum(len(p.get_images(full=True)) for p in d)
        info["pdf_images"] = nimg
        if d.page_count < 15:
            add("WARN", f"PDF 仅 {d.page_count} 页，可能截断")
    except Exception as e:
        add("WARN", f"PDF 无法解析：{e}")

# ---------- 输出 ----------
errs = [i for i in issues if i[0] == "ERROR"]
warns = [i for i in issues if i[0] == "WARN"]
print("=" * 72)
print("论文最终复检报告 v2")
print("=" * 72)
print(f"markdown 行数: {len(lines)}   二级标题: {len(h2s)}   三级标题: {info['h3_count']}")
print(f"图注: {info['fig_caps']}   正表: {main_tabs}   附录表: {info['app_tabs']}")
print(f"参考文献: {info['ref_count']} 条，被正文引用 {info['cited_count']} 条")
print(f"附录分项: {info['appendix_items']}   正文引用次数 A/B/C = "
      f"{info.get('ref_A')}/{info.get('ref_B')}/{info.get('ref_C')}")
print(f"公式号: {info['eqs']}")
print(f"HTML: 表 {info['html_tables']} 个，图 {info['html_imgs']} 张，图注 {len(info['html_figcaps'])}")
print(f"PDF: {info.get('pdf_pages')} 页，嵌入图片 {info.get('pdf_images')}")
print("-" * 72)
print(f"ERROR {len(errs)} 项 / WARN {len(warns)} 项")
for lv, m in issues:
    print(f"  [{lv}] {m}")
if not errs:
    print("  ✓ 索引与格式问题 ERROR 清零")
print("=" * 72)

json.dump({"errors": errs, "warns": warns, "info": info},
          open(os.path.join(HERE, "final_audit.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
