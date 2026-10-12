# -*- coding: utf-8 -*-
"""两遍构建：先排版 → 量出各章节实际页码 → 回填目录 → 再排版，迭代至页码稳定。"""
import os, re, json, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_outputs as B

PDF = B.OUT_PDF


def norm(s):
    return re.sub(r"[\s\u3000]+", "", s)


def parse_blocks():
    md = open(B.MD, encoding="utf-8").read()
    return B.parse(md)


def measure(entries):
    """按字号定位标题：h2=14pt、h3=12pt，而目录行仅 10/10.5pt，可精确区分。"""
    import pymupdf
    d = pymupdf.open(PDF)
    heads = []          # [(page_idx, line_text, size)]
    for i, p in enumerate(d):
        for blk in p.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                txt = norm("".join(s["text"] for s in line["spans"]))
                if not txt:
                    continue
                size = max(s["size"] for s in line["spans"])
                heads.append((i, txt, size))
    found, missing = {}, []
    for txt, lv in entries:
        key = norm(txt)
        hit = None
        for i, lt, size in heads:
            if size >= 11.5 and (lt == key or lt.startswith(key)):
                hit = i + 1
                break
        if hit:
            found[txt] = hit
        else:
            missing.append(txt)
    npages = d.page_count
    return found, missing, npages, []


def main():
    blocks = parse_blocks()
    entries = B.toc_entries(blocks)
    print(f"目录条目 {len(entries)} 条")
    prev = None
    for it in range(1, 6):
        # 构建 HTML + PDF（DOCX 最后单独生成）
        md = open(B.MD, encoding="utf-8").read()
        open(B.OUT_HTML, "w", encoding="utf-8").write(B.to_html(parse_blocks()))
        B.build_pdf()
        found, missing, npages, tocp = measure(entries)
        if found.get("一、引言"):
            json.dump({"body_start": found["一、引言"]}, open(B.TOC_META, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        if missing:
            print(f"  第{it}遍：{len(missing)} 条未定位 -> {missing[:6]}")
        print(f"  第{it}遍：共 {npages} 页，目录页 {[p+1 for p in tocp]}，"
              f"已定位 {len(found)}/{len(entries)}")
        if found == prev:
            print("  ✓ 页码已稳定")
            break
        json.dump(found, open(B.TOC_JSON, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        prev = found
    else:
        print("  ! 迭代 5 次仍未完全稳定，使用最后一次结果")
    # 终版 HTML + PDF + DOCX
    open(B.OUT_HTML, "w", encoding="utf-8").write(B.to_html(parse_blocks()))
    B.build_pdf()
    B.build_docx(parse_blocks())
    print("完成。目录页码：")
    for t, p in json.load(open(B.TOC_JSON, encoding="utf-8")).items():
        print(f"   {t}  {p}")


if __name__ == "__main__":
    main()
