# -*- coding: utf-8 -*-
"""提示词明确性探针的结果分析：标签版 vs 同种子存档（原版提示词）。

口径
----
存档 ``results/llm_gradient_noleak``：seed = 31345 + run_id
  (α=1.0, γ=0.1) = run_id 80–89 = seed 31425–31434
  (α=1.0, γ=1.0) = run_id 90–99 = seed 31435–31444
探针 ``results_prompt_labeled``：base_seed=31425, CONDITIONS=[(1.0,0.1),(1.0,1.0)], R=10
  run 0–9  = (α=1.0, γ=0.1) = seed 31425–31434   → 与存档 run_id 80–89 同种子
  run 10–19= (α=1.0, γ=1.0) = seed 31435–31444   → 与存档 run_id 90–99 同种子

主要结果量：ΔPI = PI(末步) − PI(首步)，即"上行幅度"。配对（同种子）检验。
另报 PI 末步水平、ΔEM / ΔSR / ΔER，以及表述率与兜底率（数据质量）。
"""
from __future__ import annotations

import io
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


def traj(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.sort_values("step").reset_index(drop=True)


def summarize(path: Path) -> dict:
    d = traj(path)
    return {
        "n_steps": len(d),
        "PI1": float(d["PI"].iloc[0]),
        "PI30": float(d["PI"].iloc[-1]),
        "dPI": float(d["PI"].iloc[-1] - d["PI"].iloc[0]),
        "dEM": float(d["EM"].iloc[-1] - d["EM"].iloc[0]),
        "dSR": float(d["SR"].iloc[-1] - d["SR"].iloc[0]),
        "dER": float(d["ER"].iloc[-1] - d["ER"].iloc[0]),
        "dHE": float(d["HE"].iloc[-1] - d["HE"].iloc[0]),
    }


CELLS = [
    ("γ=0.1", HERE / "results_prompt_labeled" / "runs", range(0, 10),
     ARCH / "runs", range(80, 90), 31425),
    ("γ=1.0", HERE / "results_prompt_labeled" / "runs", range(10, 20),
     ARCH / "runs", range(90, 100), 31435),
]

rows_out = []
print("=" * 78)
print("提示词明确性探针 —— 标签版（字段全名 + 取值范围）vs 同种子存档（原版单字母）")
print("=" * 78)

for name, newdir, newids, olddir, oldids, seed0 in CELLS:
    print(f"\n### {name}   （配对 n={len(list(newids))}，同种子）")
    new, old, seeds = [], [], []
    for k, (ni, oi) in enumerate(zip(newids, oldids)):
        sn = summarize(newdir / f"llm_robust_run_{ni}.csv")
        so = summarize(olddir / f"llm_robust_run_{oi}.csv")
        new.append(sn)
        old.append(so)
        seeds.append(seed0 + k)
    f = lambda key, lst: np.array([x[key] for x in lst], float)  # noqa: E731

    print(f"{'seed':>7} {'PI(1)':>8} {'PI(30)':>8} {'ΔPI 标签版':>11} {'ΔPI 原版':>10} {'配对差':>9}")
    for i, s in enumerate(seeds):
        print(f"{s:>7} {new[i]['PI1']:>8.4f} {new[i]['PI30']:>8.4f} "
              f"{new[i]['dPI']:>+11.4f} {old[i]['dPI']:>+10.4f} "
              f"{new[i]['dPI']-old[i]['dPI']:>+9.4f}")

    d_new, d_old = f("dPI", new), f("dPI", old)
    diff = d_new - d_old
    t, p = stats.ttest_rel(d_new, d_old)
    print(f"\n  上行幅度 ΔPI：标签版 = {d_new.mean():+.4f} (sd {d_new.std(ddof=1):.4f})"
          f"；原版 = {d_old.mean():+.4f} (sd {d_old.std(ddof=1):.4f})")
    print(f"  配对差 = {diff.mean():+.4f} (sd {diff.std(ddof=1):.4f})，"
          f"95% CI [{diff.mean()-stats.t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)):+.4f}, "
          f"{diff.mean()+stats.t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)):+.4f}]")
    print(f"  配对 t = {t:+.3f}，p = {p:.4f}")
    print(f"  上行是否仍在：标签版 {sum(d_new>0)}/{len(d_new)} 个 run 上升"
          f"；原版 {sum(d_old>0)}/{len(d_old)} 个 run 上升")
    print(f"  上行幅度比（标签版/原版） = {d_new.mean()/d_old.mean():.3f}")
    for k in ("PI30", "dEM", "dSR", "dER", "dHE"):
        a, b = f(k, new), f(k, old)
        tk, pk = stats.ttest_rel(a, b)
        print(f"  {k:>5}: 标签版 {a.mean():+.4f} vs 原版 {b.mean():+.4f}"
              f"   配对差 {a.mean()-b.mean():+.4f}  p={pk:.4f}")
    rows_out.append((name, d_new.mean(), d_old.mean(), diff.mean(), p))

print("\n" + "=" * 78)
print("判读口径（写在前面，避免事后挑选解释）：")
print("  · 若标签版的 ΔPI 仍在、且与原版配对差不显著 → 上行不是'字段没解释清楚'造成的；")
print("  · 若标签版 ΔPI 明显变小或反转 → 原提示词确实在驱动该上行，须写进论文并重新表述。")
for name, a, b, d, p in rows_out:
    verdict = "仍在（无显著差异）" if (a > 0.10 and p > 0.05) else ("明显改变" if p <= 0.05 else "需更多重复")
    print(f"  [{name}] 标签版 {a:+.4f} vs 原版 {b:+.4f}，配对差 {d:+.4f}（p={p:.3f}）→ {verdict}")
