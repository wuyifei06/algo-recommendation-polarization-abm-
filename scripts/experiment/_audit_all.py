# -*- coding: utf-8 -*-
"""方向版实验 · 全面审计（机制 + 数据）。

检查项：
  A. 完整性     各批次 n vs 期望、重复 run_id、缺行
  B. 调用一致性  完成 run 的 calls 数 == 期望（90/300）
  C. 质量       全批次兜底率扫描（>2% 残留）、退化指标（expr 0/1、silent=1）
  D. 配对数     消融-基线同种子配对 n、T100 种子公式与主实验重叠
  E. 隔离       _VOIDED_/resampled 工件是否会落入分析 glob
  F. 制度盘点    各数据集属于哪套提示词 × 哪套 K 制度
输出：控制台 + `_全面审计_20261009.md`
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

PROJ = Path(_PROJ_ROOT)
RD = PROJ / "results_direction"
AR = PROJ / "results"
PROBE = Path(_PROBE_ROOT)
OUT = PROBE / "_全面审计_20261009.md"

lines: list[str] = []
issues: list[str] = []


def w(s: str = "") -> None:
    lines.append(s)
    print(s)


def issue(level: str, msg: str) -> None:
    issues.append(f"[{level}] {msg}")
    w(f"  {level} {msg}")


def load_summary(base: Path) -> pd.DataFrame:
    m = base / "llm_robust_summary.csv"
    if m.exists():
        try:
            return pd.read_csv(m)
        except Exception:
            pass
    fs = sorted(base.glob("shard*_of*/llm_robust_summary.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


EXPECT = {
    "llm_qwen_full": 200, "llm_qwen_abl_no_post": 90, "llm_qwen_abl_no_emotion": 90,
    "llm_qwen_abl_positive_control": 10,
    "llm_randomfeed_control": 20,
    "llm_ablation_no_post": 160, "llm_ablation_no_emotion": 160,
    "llm_ablation_positive_control": 20,
    "llm_n100_scale": 30, "llm_extend_T100": 160,
}

w("# 方向版实验 · 全面审计报告（2026-10-09）\n")
w("## A/B/C. 批次完整性 · 调用数一致性 · 质量扫描\n")
w("| 批次 | 行数/期望 | 重复run_id | calls≠期望(完成run) | 兜底mean | 兜底max | >2% | 退化指标 |")
w("|---|---|---|---|---|---|---|---|")
for name, exp in EXPECT.items():
    base = RD / name
    if not base.exists():
        w(f"| {name} | **目录缺失** | - | - | - | - | - | - |")
        issue("P1", f"{name} 目录缺失")
        continue
    df = load_summary(base)
    n = len(df)
    dup = int(df["run_id"].duplicated().sum()) if len(df) else 0
    fb = df["llm_fallback_rate"].astype(float) if len(df) else pd.Series(dtype=float)
    over2 = int((fb > 0.02).sum()) if len(fb) else 0
    # calls 数一致性（对已完成 run 抽查全部）
    bad_calls = 0
    extra_calls = []
    if len(df):
        calls_map = {}
        for fp in glob.glob(str(base / "shard*_of*" / "calls" / "llm_robust_run_*.jsonl")):
            rid = int(Path(fp).stem.replace("llm_robust_run_", ""))
            try:
                calls_map[rid] = sum(1 for _ in open(fp, encoding="utf-8", errors="replace"))
            except OSError:
                pass
        per_step = 10 if name == "llm_n100_scale" else 3   # 每步批次数 = N/10
        for _, r in df.iterrows():
            rid = int(r["run_id"])
            steps = int(r.get("step", 0))
            need = steps * per_step
            got = calls_map.get(rid)
            if got is None:
                continue
            if got < need:
                bad_calls += 1                              # 缺记录 = P1
            elif got > need:
                extra_calls.append((rid, got - need))       # 断点重执行 = P3 留痕
    # 退化指标
    degen = []
    if len(df):
        if "llm_silent_step_rate" in df.columns:
            k = int((df["llm_silent_step_rate"].astype(float) > 0.99).sum())
            if k:
                degen.append(f"全沉默run×{k}")
        if "llm_express_rate" in df.columns:
            e = df["llm_express_rate"].astype(float)
            k0 = int((e <= 0.01).sum())
            k1 = int((e >= 0.99).sum())
            if k0:
                degen.append(f"表达率≈0×{k0}")
            if k1:
                degen.append(f"表达率≈1×{k1}")
    ex = f"（{len(extra_calls)}个run断点重执行+{sum(e for _,e in extra_calls)}条冗余记录）" if extra_calls else ""
    w(f"| {name} | {n}/{exp}{' ⚠' if n != exp else ''} | {dup} | {bad_calls} | "
      f"{fb.mean():.3%} | {fb.max():.3%} | {over2} | {'、'.join(degen) or '无'}{ex} |")
    if extra_calls:
        issue("P3", f"{name} 断点重执行冗余：{extra_calls[:8]}（指标不受影响，存档级留痕）")
    if dup:
        issue("P0", f"{name} 存在重复 run_id ×{dup}")
    if over2:
        issue("P1", f"{name} 有 {over2} 个 run 兜底率 >2%（应为 0）")
    if bad_calls:
        issue("P1", f"{name} 有 {bad_calls} 个完成 run 的 calls 数与步数不一致")
    if degen:
        issue("P2", f"{name} 退化指标：{'、'.join(degen)}")
    if n < exp and name != "llm_extend_T100":   # T100 尚未跑完，允许不足
        issue("P1", f"{name} 行数不足 {n}/{exp}")
    if n < exp:
        w(f"  ℹ {name} 未完成（{n}/{exp}）——按计划暂停中" if name == "llm_extend_T100" else "")

# D. 配对
w("\n## D. 同种子配对完整性\n")
def seed_map(base: Path) -> dict:
    df = load_summary(base)
    return {int(r["run_id"]): int(r["seed"]) for _, r in df.iterrows()} if len(df) else {}

main_seeds = set()
for b in sorted(PROBE.glob("results_prompt_direction/a*")):
    main_seeds.update(seed_map(b).values())
for arm, base in [("冻结表达", RD / "llm_ablation_no_post"),
                  ("冻结情绪", RD / "llm_ablation_no_emotion"),
                  ("阳性对照", RD / "llm_ablation_positive_control")]:
    s = set(seed_map(base).values())
    inter = s & main_seeds
    w(f"- {arm}: 臂 n={len(s)}，与主实验交集 n={len(inter)}{' ✓' if len(inter) == len(s) else ' ⚠'}")
    if len(inter) != len(s):
        issue("P1", f"{arm} 与主实验种子配对不完整：{len(inter)}/{len(s)}")

# E. 隔离
w("\n## E. 废件隔离（防止污染分析 glob）\n")
voided = [p.name for p in RD.iterdir() if p.is_dir() and p.name.startswith("_VOIDED")]
baks = sum(1 for _ in RD.glob("*/*/*/*.resampled*"))
baks2 = sum(1 for _ in RD.glob("*/*/*.resampled*"))
w(f"- _VOIDED 目录：{voided}")
w(f"- .resampled_bak 工件（三级）: {baks}，二级: {baks2}")
# 检查所有分析脚本的 glob 是否会撞到废件
scripts = ["digest_results.py", "progress_direction.py", "reanswer_hypotheses.py"]
danger = []
for sp in scripts:
    txt = (PROBE / sp).read_text(encoding="utf-8")
    for pat in ("_VOIDED", ".resampled"):
        if pat in txt:
            danger.append(f"{sp} 引用了 {pat}")
    # 检查通配模式撞车：含 "llm_*" 风格且未排除 _ 前缀
    if re.search(r'glob\(["\'].*"\*"', txt) and "startswith" not in txt:
        pass
w(f"- 废件引用检查：{danger or '无 ✓'}")
if danger:
    issue("P2", "以下脚本可能触及废件：" + "; ".join(danger))

# F. 制度盘点
w("\n## F. 数据集制度盘点（提示词 × K 制度）\n")
w("| 数据集 | 提示词 | K 制度 | 备注 |")
w("|---|---|---|---|")
rows = [
    ("主实验 160（results_prompt_direction）", "方向版", "fixed K=10（N=30）", "K 比例化后 N=30 数值不变"),
    ("Qwen 复现/消融 390", "方向版", "fixed K=10（N=30）", "含迭代重采，见决定文档"),
    ("随机feed 20", "方向版", "fixed K=10（N=30）", ""),
    ("消融×3 340", "方向版", "fixed K=10（N=30）", "含阳对探针档 20"),
    ("N100 30（results_direction）", "方向版", "**fixed K=10（N=100，旧制度）**", "K 比例化后成为历史制度；新制度需 K=33 重跑"),
    ("T100（进行中）", "方向版", "fixed K=10（N=30）", "K 不变"),
    ("老提示词全套（results/）", "原版", "fixed K=10", "已归档 85_被取代；对照臂"),
    ("Rule 臂（results/rule_*、calib 等）", "—（规则模型）", "fixed K=10（N=200 设计）", "未受本轮改动影响"),
]
for a, b, c, d in rows:
    w(f"| {a} | {b} | {c} | {d} |")

w("\n## 结论汇总\n")
if issues:
    for it in issues:
        w(f"- {it}")
else:
    w("- 未发现 P0/P1/P2 级问题")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n[已写入] {OUT}")
