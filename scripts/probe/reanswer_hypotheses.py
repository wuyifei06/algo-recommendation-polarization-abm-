# -*- coding: utf-8 -*-
"""在「方向版」提示词的完整网格（160 runs）上重答论文的 H1–H4。

对照基线：论文原稿（原版提示词）的结论
  H1 提高 α 会改变群体极化水平        → 原判：部分支持（Rule 支持 / LLM 未复现）
  H2 α 经 HE 与 EE 通道中介传导       → 原判：不支持（LLM b 路径不显著）
  H3 α 与极化存在非线性/阈值          → 原判：不支持（二次项均不显著）
  H4 Rule 与 LLM 的方向一致           → 原判：未获支持
Rule 臂参照（不经过提示词，不受影响）：β_std = −0.566（γ=1.0），R²=0.32。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
LAB = HERE / "results_prompt_direction"


def load(d: Path) -> pd.DataFrame:
    files = sorted(set(list(d.glob("runs/llm_robust_run_*.csv"))
                       + list(d.glob("*/runs/llm_robust_run_*.csv"))))
    rows = []
    for f in files:
        if f.name.endswith("_agents_final.csv"):
            continue
        t = pd.read_csv(f).sort_values("step")
        last = t.iloc[-1]
        rows.append(dict(alpha=float(t["alpha"].iloc[0]), gamma=float(t["gamma"].iloc[0]),
                         PI30=float(last["PI"]), HE30=float(last["HE"]), EE30=float(last["EE"]),
                         EM30=float(last["EM"]), ER30=float(last["ER"]), SR30=float(last["SR"])))
    return pd.DataFrame(rows)


def ols(y, X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, se, t, p, r2


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def sobel(a, sa, b, sb):
    ab = a * b
    se = np.sqrt(b ** 2 * sa ** 2 + a ** 2 * sb ** 2)
    z = ab / se
    return ab, se, z, 2 * (1 - stats.norm.cdf(abs(z)))


df = load(LAB)
print(f"方向版网格 runs={len(df)}")
print()

# ---------------- H1 ----------------
print("=" * 76)
print("H1　提高 α 会改变群体极化水平　（PI(30) ~ α）")
print("=" * 76)
for g in (0.1, 1.0):
    d = df[np.isclose(df["gamma"], g)]
    c, se, t, p, r2 = ols(d["PI30"].to_numpy(float), z(d["alpha"].to_numpy(float)))
    print(f"  γ={g}: β_std = {c[1]:+.4f} (SE {se[1]:.4f})  p = {p[1]:.4g}  R² = {r2:.4f}  n = {len(d)}")
cm, sem, tm, pm, r2m = ols(df["PI30"].to_numpy(float),
                           np.column_stack([z(df["alpha"]), z(df["gamma"])]))
print(f"  合并(γ协变量): β_std(α) = {cm[1]:+.4f} (SE {sem[1]:.4f})  p = {pm[1]:.4g}  R² = {r2m:.4f}")
print("  【原判】部分支持（Rule 支持 / LLM 未复现）")

# ---------------- H3 ----------------
print()
print("=" * 76)
print("H3　α 与极化的关系存在非线性（二次项）")
print("=" * 76)
for g in (0.1, 1.0):
    d = df[np.isclose(df["gamma"], g)]
    x = d["alpha"].to_numpy(float)
    c, se, t, p, r2 = ols(d["PI30"].to_numpy(float),
                          np.column_stack([z(x), z(x) ** 2]))
    print(f"  γ={g}: β_线性 = {c[1]:+.4f} (p={p[1]:.3g})  β_二次 = {c[2]:+.4f} (p={p[2]:.3g})  R² = {r2:.4f}")
print("  【原判】不支持（二次项均不显著）")

# ---------------- H2 ----------------
print()
print("=" * 76)
print("H2　α 经 HE / EE 通道中介传导（a 路径、b 路径、Sobel 间接效应）")
print("=" * 76)
for g in (0.1, 1.0):
    d = df[np.isclose(df["gamma"], g)]
    xa = z(d["alpha"].to_numpy(float))
    for med, lab in (("HE30", "HE"), ("EE30", "EE")):
        ca, sea, ta, pa, _ = ols(d[med].to_numpy(float), xa)          # a: M ~ X
        cb, seb, tb, pb, _ = ols(d["PI30"].to_numpy(float),
                                 np.column_stack([xa, z(d[med].to_numpy(float))]))  # b: Y ~ X + M
        ab, se_ab, zval, pv = sobel(ca[1], sea[1], cb[2], seb[2])
        print(f"  γ={g} {lab}: a = {ca[1]:+.4f}(p={pa[1]:.3g})  b = {cb[2]:+.4f}(p={pb[2]:.3g})"
              f"  间接 = {ab:+.4f} (Sobel z={zval:+.2f}, p={pv:.3g})  直接(α) = {cb[1]:+.4f} (p={pb[1]:.3g})")
print("  【原判】不支持（a 路径显著、b 路径不显著 → '通道存在、传导不成立'）")

# ---------------- H4 ----------------
print()
print("=" * 76)
print("H4　Rule 与 LLM 的 α 响应方向是否一致")
print("=" * 76)
print("  Rule 臂（不受提示词影响）：β_std = −0.566（γ=1.0），方向为负")
for g in (0.1, 1.0):
    d = df[np.isclose(df["gamma"], g)]
    c, se, t, p, r2 = ols(d["PI30"].to_numpy(float), z(d["alpha"].to_numpy(float)))
    print(f"  LLM 方向版 γ={g}: β_std = {c[1]:+.4f}（方向为{'负' if c[1]<0 else '正'}，p={p[1]:.4g}）")
print("  【原判】未获支持（Rule 负向且稳定，LLM 未达可识别程度）")

# ---------------- 上行 ----------------
print()
print("=" * 76)
print("附带：LLM 臂的 PI 上行（论文的第三个发现）")
print("=" * 76)
tr = []
for f in sorted(set(list(LAB.glob("runs/llm_robust_run_*.csv"))
                    + list(LAB.glob("*/runs/llm_robust_run_*.csv")))):
    if f.name.endswith("_agents_final.csv"):
        continue
    t = pd.read_csv(f).sort_values("step")
    tr.append(float(t["PI"].iloc[-1] - t["PI"].iloc[0]))
tr = np.array(tr)
print(f"  ΔPI = {tr.mean():+.4f} (sd {tr.std(ddof=1):.4f})；上升 run {int((tr>0).sum())}/{len(tr)}")
print(f"  分 γ：", end=" ")
d0 = load(LAB)
print()
