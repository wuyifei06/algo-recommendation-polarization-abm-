# -*- coding: utf-8 -*-
"""完整网格分析：标签版提示词 vs 存档原版（160 runs × 2，run_id→(α,γ)→种子 完全对齐）。

回答三个问题
------------
Q1  α 的零效应还在不在？（主要）：逐 γ 的 OLS 斜率（原始尺度 + 标准化）、R²、p、
    双侧 TOST 等价判定、MDE，并与存档同法重算的结果并排。
Q2  PI 上行还在不在？逐格 ΔPI = PI(末步) − PI(首步)，跨 α 汇总，两版本对照。
Q3  逐 run 配对（同种子）差异有多大？PI(30) 与 ΔPI 的配对差分布。

口径
----
- 存档：``results/llm_gradient_noleak``（原版提示词，base_seed=31345, R=10, 8α × 2γ）
- 标签版：``results_prompt_full``（字段全名 + 取值范围，同 base_seed/R/网格）
- 两者 run_id 一一对应且同种子（已逐条核实）。
- 等价界沿用论文口径：δ = 0.031，**位于原始（斜率）尺度**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ARCH = Path(_PROJ_ROOT + r"\results\llm_gradient_noleak")
LAB = HERE / "results_prompt_direction"      # 方向版（标签版 + feed_mean_opinion）完整网格

DELTA = 0.031                      # 等价界（原始尺度），沿用论文
MDE_FACTOR = 2.8                   # 80% 效力、双侧 0.05


def load(d: Path) -> pd.DataFrame:
    """支持两种布局：单目录 runs/，或「每档 α 一个子目录」的判定实验布局。"""
    files = sorted(set(list(d.glob("runs/llm_robust_run_*.csv"))
                       + list(d.glob("*/runs/llm_robust_run_*.csv"))))
    rows = []
    for f in files:
        if f.name.endswith("_agents_final.csv"):
            continue
        rid = int(f.stem.split("_")[-1])
        t = pd.read_csv(f).sort_values("step")
        rows.append({
            "run_id": rid,
            "alpha": float(t["alpha"].iloc[0]),
            "gamma": float(t["gamma"].iloc[0]),
            "PI1": float(t["PI"].iloc[0]),
            "PI30": float(t["PI"].iloc[-1]),
            "dPI": float(t["PI"].iloc[-1] - t["PI"].iloc[0]),
            "dEM": float(t["EM"].iloc[-1] - t["EM"].iloc[0]),
            "dSR": float(t["SR"].iloc[-1] - t["SR"].iloc[0]),
            "dER": float(t["ER"].iloc[-1] - t["ER"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def ols(y: np.ndarray, x: np.ndarray):
    """返回 slope, se, r2, p, n。"""
    n = len(y)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    dof = n - 2
    s2 = (resid ** 2).sum() / dof
    se = np.sqrt(s2 / sxx)
    t = b / se
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    r2 = 1 - (resid ** 2).sum() / ((y - ym) ** 2).sum()
    return b, se, r2, p, n


def tost(b: float, se: float, n: int, delta: float = DELTA):
    """双侧等价检验 H0: |β| ≥ δ。返回 (p_TOST, 通过?)"""
    dof = n - 2
    p1 = 1 - stats.t.cdf((b + delta) / se, dof)      # H0: β ≤ −δ
    p2 = stats.t.cdf((b - delta) / se, dof)          # H0: β ≥ +δ
    p = max(p1, p2)
    return p, p < 0.05


def z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / a.std(ddof=0)


def report(df: pd.DataFrame, name: str) -> dict:
    out = {}
    print(f"\n{'='*74}\n### {name}　（n={len(df)}）\n{'='*74}")
    for g in sorted(df["gamma"].unique()):
        d = df[np.isclose(df["gamma"], g)]
        y, x = d["PI30"].to_numpy(float), d["alpha"].to_numpy(float)
        b, se, r2, p, n = ols(y, x)
        bz, sez, r2z, pz, _ = ols(z(y), z(x))
        pt, ok = tost(b, se, n)
        mde = MDE_FACTOR * se
        excl = abs(b) + 1.96 * se
        print(f"\n  γ={g}　PI(30) ~ α")
        print(f"    原始尺度斜率  β = {b:+.5f}  (SE {se:.5f})  95%CI [{b-1.96*se:+.5f}, {b+1.96*se:+.5f}]")
        print(f"    标准化斜率    β_std = {bz:+.4f}  SE {sez:.4f}  R² = {r2z:.4f}  p = {pz:.4f}")
        print(f"    双侧 TOST（δ={DELTA}）: p = {pt:.4f}  → {'通过（可判定等价于零）' if ok else '未通过'}")
        print(f"    可排除上界 |β|+1.96·SE = {excl:.5f}；MDE(80%) = {mde:.5f}")
        out[f"g{g}"] = dict(b=b, se=se, bz=bz, r2z=r2z, pz=pz, tost_p=pt, tost_ok=ok,
                            excl=excl, mde=mde, n=n)
    # 合并（γ 作协变量）
    X = np.column_stack([np.ones(len(df)), df["alpha"].to_numpy(float), df["gamma"].to_numpy(float)])
    y = df["PI30"].to_numpy(float)
    Xz = np.column_stack([np.ones(len(df)), z(df["alpha"].to_numpy(float)), z(df["gamma"].to_numpy(float))])
    yz = z(y)
    cod, *_ = np.linalg.lstsq(X, y, rcond=None)
    codz, *_ = np.linalg.lstsq(Xz, yz, rcond=None)
    resid = y - X @ cod
    dof = len(y) - X.shape[1]
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se_b = np.sqrt(cov[1, 1])
    t = cod[1] / se_b
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot
    pt, ok = tost(cod[1], se_b, len(y) + 1)   # 用 dof 近似
    print(f"\n  合并（γ 为协变量，n={len(df)}）：α 系数 原始 β = {cod[1]:+.5f} (SE {se_b:.5f})"
          f"　标准化 β_std = {codz[1]:+.4f}　p = {p:.4f}　R² = {r2:.4f}")
    print(f"    双侧 TOST（δ={DELTA}）p = {pt:.4f} → {'通过' if ok else '未通过'}")
    out["merged"] = dict(b=cod[1], se=se_b, bz=codz[1], p=p, r2=r2, tost_p=pt, tost_ok=ok)
    # 上行
    for g in sorted(df["gamma"].unique()):
        d = df[np.isclose(df["gamma"], g)]
        print(f"  γ={g} 上行 ΔPI：均值 {d['dPI'].mean():+.4f}（sd {d['dPI'].std(ddof=1):.4f}），"
              f"上升 run {int((d['dPI']>0).sum())}/{len(d)}；"
              f"PI(1) {d['PI1'].mean():.4f} → PI(30) {d['PI30'].mean():.4f}")
        out[f"rise_g{g}"] = dict(mean=d["dPI"].mean(), sd=d["dPI"].std(ddof=1),
                                 n_up=int((d["dPI"] > 0).sum()), n=len(d))
    return out


arch = load(ARCH)
lab = load(LAB)
print(f"存档 runs={len(arch)}；标签版 runs={len(lab)}")
assert len(arch) == len(lab), "两臂 run 数不一致"

ra = report(arch, "存档（原版提示词：单字母 o/e/a/d/fe）")
rb = report(lab, "标签版提示词（字段全名 + 取值范围）")

print(f"\n\n{'='*74}\n### 逐 run 配对（同种子，n={len(arch)}）\n{'='*74}")
for col in ("PI30", "dPI"):
    a, b = lab[col].to_numpy(float), arch[col].to_numpy(float)
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    print(f"  {col:>5}: 标签版均值 {a.mean():+.4f} vs 原版 {b.mean():+.4f}；"
          f"配对差 {diff.mean():+.4f} (sd {diff.std(ddof=1):.4f})　t={t:+.2f}　p={p:.2e}")

print(f"\n{'='*74}\n### 判定\n{'='*74}")
for g in ("0.1", "1.0"):
    A, B = ra[f"g{g}"], rb[f"g{g}"]
    print(f"\n  γ={g}")
    print(f"    α 斜率：原版 β={A['b']:+.5f}（TOST {'通过' if A['tost_ok'] else '未通过'}）"
          f"　→　标签版 β={B['b']:+.5f}（TOST {'通过' if B['tost_ok'] else '未通过'}）")
    d = B["b"] - A["b"]
    print(f"    斜率变化 Δβ = {d:+.5f}；等价界 δ={DELTA} 的 {abs(d)/DELTA*100:.1f}%")
    print(f"    上行幅度：原版 {ra[f'rise_g{g}']['mean']:+.4f} → 标签版 {rb[f'rise_g{g}']['mean']:+.4f}"
          f"（比 {rb[f'rise_g{g}']['mean']/ra[f'rise_g{g}']['mean']:.3f}）")
