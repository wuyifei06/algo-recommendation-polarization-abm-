# -*- coding: utf-8 -*-
"""判定实验分析：「方向版」提示词（补回带符号的 feed_mean_opinion）下，α 有没有效应。

三臂对照（同种子，凡同种子处均做配对）：
  A. 原版（存档）       results/llm_gradient_noleak          单字母 o/e/a/d/fe，无方向
  B. 标签版（部分）     probe_提示词明确性/results_prompt_full  字段全名+范围，无方向（跑到约 80/160）
  C. 方向版（本批）     results_prompt_direction/{a0.0,a1.0,a2.0}  标签版 + feed_mean_opinion

问题：C 的 α 剖面还是不是平的？
  · 若 C 出现与 Rule 同号的显著负斜率 → 设计缺陷成立（信息集缺方向是主因），须改口径/重做
  · 若 C 仍然平（TOST 通过）→ 主体性质：给了方向 LLM 也不按曝光均值更新，原结论更硬
判读阈值**事先写死**，避免事后挑选解释。
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
DIR_ROOT = HERE / "results_prompt_direction"
LAB_FULL = HERE / "results_prompt_full"

ALPHAS = [0.0, 1.0, 2.0]
GAMMAS = [0.1, 1.0]
BLOCK_BASE = {0.0: 31345, 1.0: 31425, 2.0: 31485}

# —— 判读阈值（事先写死）——
RULE_RAW_SLOPE = -0.0215          # Rule 主实验 γ=1.0 的原始尺度斜率（量级参照）
THRESH_STRONG = 0.020             # |β| ≥ 此值 → 视为"出现效应"
THRESH_NULL = 0.010               # |β| < 此值且 TOST 通过 → 视为"仍无效应"


def load(rdir: Path, run_ids=None):
    rows = []
    for f in sorted(rdir.glob("runs/llm_robust_run_*.csv")):
        if f.name.endswith("_agents_final.csv"):
            continue
        rid = int(f.stem.split("_")[-1])
        if run_ids is not None and rid not in run_ids:
            continue
        t = pd.read_csv(f).sort_values("step")
        rows.append(dict(run_id=rid, alpha=float(t["alpha"].iloc[0]),
                         gamma=float(t["gamma"].iloc[0]),
                         PI1=float(t["PI"].iloc[0]), PI30=float(t["PI"].iloc[-1]),
                         dPI=float(t["PI"].iloc[-1] - t["PI"].iloc[0]),
                         EM30=float(t["EM"].iloc[-1]), ER30=float(t["ER"].iloc[-1])))
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True) if rows else pd.DataFrame()


def load_direction() -> pd.DataFrame:
    frames = []
    for a, base, _tag in [(0.0, 31345, "a0.0"), (1.0, 31425, "a1.0"), (2.0, 31485, "a2.0")]:
        d = load(DIR_ROOT / f"a{a:.1f}")
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ols(y, x):
    n = len(y)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0 or n < 3:
        return None
    b = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym - b * xm + b * x)
    dof = n - 2
    se = np.sqrt((resid ** 2).sum() / dof / sxx)
    p = 2 * (1 - stats.t.cdf(abs(b / se), dof))
    r2 = 1 - (resid ** 2).sum() / ((y - ym) ** 2).sum()
    bz = b * x.std(ddof=0) / y.std(ddof=0) if y.std(ddof=0) > 0 else np.nan
    return dict(b=b, se=se, p=p, r2=r2, bz=bz, n=n,
                ci=[b - stats.t.ppf(.975, dof) * se, b + stats.t.ppf(.975, dof) * se])


def tost(b, se, n, delta=0.031):
    dof = n - 2
    p1 = 1 - stats.t.cdf((b + delta) / se, dof)
    p2 = stats.t.cdf((b - delta) / se, dof)
    p = max(p1, p2)
    return p, p < 0.05


def paired(a, b):
    """返回 (均差, sd, t, p, n)。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    t, p = stats.ttest_rel(a, b)
    return d.mean(), d.std(ddof=1), t, p, len(d)


def profile(df, gamma):
    d = df[np.isclose(df["gamma"], gamma)] if len(df) else df
    return [(a, d[np.isclose(d["alpha"], a)]["PI30"].mean(),
             d[np.isclose(d["alpha"], a)]["dPI"].mean(),
             len(d[np.isclose(d["alpha"], a)])) for a in ALPHAS
            if len(df) and len(d[np.isclose(d["alpha"], a)])]


def main() -> None:
    C = load_direction()
    A = load(ARCH)
    # 标签版 full 臂只跑到约 run_id 86；取它与本实验可配对的部分
    B = load(LAB_FULL)
    print("=" * 78)
    print("判定实验：方向版提示词（补回带符号 feed_mean_opinion）下，α 有没有效应？")
    print("=" * 78)
    print(f"\n方向版 runs={len(C)}　存档 runs={len(A)}　标签版(部分) runs={len(B)}")

    for g in GAMMAS:
        print(f"\n{'='*74}\n### γ={g}\n{'='*74}")
        Cg = C[np.isclose(C["gamma"], g)] if len(C) else C
        Ag = A[np.isclose(A["gamma"], g)]
        Bg = B[np.isclose(B["gamma"], g)] if len(B) else B

        print(f"\n  α 剖面（PI(30) 均值 / 上行 ΔPI 均值 [n]）")
        print(f"  {'α':>5} {'C 方向版':>22} {'B 标签版':>22} {'A 原版':>22}")
        for a in ALPHAS:
            def fmt(d):
                if not len(d):
                    return "—"
                dd = d[np.isclose(d["alpha"], a)]
                if not len(dd):
                    return "—"
                return f"{dd['PI30'].mean():.4f} / {dd['dPI'].mean():+.4f} [{len(dd)}]"
            print(f"  {a:>5.1f} {fmt(Cg):>22} {fmt(Bg):>22} {fmt(Ag):>22}")

        if len(Cg) >= 6:
            r = ols(Cg["PI30"].to_numpy(float), Cg["alpha"].to_numpy(float))
            if r is None:
                na = Cg["alpha"].nunique()
                print(f"\n  C 方向版：α 尚只覆盖 {na} 档，暂不能估斜率")
                continue
            pt, ok = tost(r["b"], r["se"], r["n"])
            print(f"\n  C 方向版 OLS（PI(30)~α, n={r['n']}）：")
            print(f"    β(原始) = {r['b']:+.5f}  95%CI [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]")
            print(f"    β_std   = {r['bz']:+.4f}  R² = {r['r2']:.4f}  p = {r['p']:.4f}")
            print(f"    TOST(δ=0.031) p = {pt:.4f} → {'通过' if ok else '未通过'}")
            print(f"    量级参照：Rule 主实验原始斜率 = {RULE_RAW_SLOPE:+.5f}")

        # 同种子配对：C vs A、C vs B（按 seed=31345+run_id_archive 对齐）
        # 方向版 run_id 0-9/10-19/20-29 对应存档 run_id 0-19/80-99/140-159
        rid_map = {}
        for bi, (a, base, _t) in enumerate([(0.0, 31345, "a0.0"), (1.0, 31425, "a1.0"),
                                            (2.0, 31485, "a2.0")]):
            for k in range(10):
                rid_map[bi * 10 + k] = (a, base + k)
        print(f"\n  同种子配对（方向版 − 存档原版）：")
        for a in ALPHAS:
            csel = Cg[np.isclose(Cg["alpha"], a)] if len(Cg) else Cg
            if not len(csel):
                continue
            asel = Ag[np.isclose(Ag["alpha"], a)].copy()
            if not len(asel):
                continue
            csel = csel.copy()
            csel["seed_key"] = [BLOCK_BASE[a] + (int(r) % 20) for r in csel["run_id"]]
            asel["seed_key"] = 31345 + asel["run_id"].astype(int)
            m = csel.merge(asel[["seed_key", "PI30", "dPI"]], on="seed_key", suffixes=("_C", "_A"))
            if len(m) >= 3:
                dm, ds, t, p, n = paired(m["PI30_C"], m["PI30_A"])
                dm2, ds2, t2, p2, _ = paired(m["dPI_C"], m["dPI_A"])
                print(f"    α={a:.1f}: PI30 差 {dm:+.4f} (p={p:.2e})；ΔPI 差 {dm2:+.4f} (p={p2:.2e})，n={n}")

        # 同种子配对：C vs B（标签版，同 seed；仅 α=0 与 α=1.0 有重叠）
        if len(Bg):
            print(f"\n  同种子配对（方向版 − 标签版）：")
            for a in ALPHAS:
                csel = Cg[np.isclose(Cg["alpha"], a)] if len(Cg) else Cg
                if not len(csel):
                    continue
                bsel = Bg[np.isclose(Bg["alpha"], a)].copy()
                if not len(bsel):
                    continue
                csel = csel.copy()
                csel["seed_key"] = [BLOCK_BASE[a] + (int(r) % 20) for r in csel["run_id"]]
                bsel["seed_key"] = 31345 + bsel["run_id"].astype(int)
                m = csel.merge(bsel[["seed_key", "PI30", "dPI"]], on="seed_key",
                               suffixes=("_C", "_B"))
                if len(m) >= 3:
                    dm, ds, t, p, n = paired(m["PI30_C"], m["PI30_B"])
                    print(f"    α={a:.1f}: PI30 差 {dm:+.4f} (p={p:.2e})，n={n}")

    print(f"\n{'='*74}\n### 判定（阈值事先写死）\n{'='*74}")
    for g in GAMMAS:
        Cg = C[np.isclose(C["gamma"], g)] if len(C) else C
        if len(Cg) < 6:
            print(f"  γ={g}: 数据不足")
            continue
        r = ols(Cg["PI30"].to_numpy(float), Cg["alpha"].to_numpy(float))
        if r is None:
            print(f"  γ={g}: α 尚未覆盖 2 档以上，暂不能估斜率")
            continue
        pt, ok = tost(r["b"], r["se"], r["n"])
        if r["b"] <= -THRESH_STRONG and r["p"] < 0.05:
            v = ("设计缺陷成立：出现与 Rule 同号的显著负斜率 → 论文须改口径，"
                 "「主体模型依赖」需重新表述并考虑重做")
        elif abs(r["b"]) < THRESH_NULL and ok:
            v = "主体性质成立：补了方向信息 α 仍无效应 → 原结论更硬，论文只需加边界说明"
        else:
            v = (f"中间地带：|β|={abs(r['b']):.4f}，p={r['p']:.3f}，TOST {'通过' if ok else '未通过'}"
                 f" → 建议加重复后再判")
        print(f"  γ={g}: β={r['b']:+.5f} (95%CI [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}])"
              f"，p={r['p']:.4f}，TOST p={pt:.4f}")
        print(f"        → {v}")


if __name__ == "__main__":
    main()
