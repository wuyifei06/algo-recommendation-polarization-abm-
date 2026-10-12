# -*- coding: utf-8 -*-
"""从 probe 主实验(T=30) 的 checkpoint 重建逐步意见分布，计算 Sarle BC（与 shape_bc.py 同法）。"""
import glob, json, os, re
import numpy as np
import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

MAIN = _PROBE_ROOT + r"\results_prompt_direction"
OUT = _os.path.join(_WORK_ROOT, r"shape_bc_main.json")
TS = [1, 5, 10, 15, 20, 25, 30]


def bc(x):
    x = np.asarray(x, float); n = len(x)
    if n < 4 or x.std() == 0:
        return np.nan
    g1 = stats.skew(x); g2 = stats.kurtosis(x, fisher=True)
    denom = g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g1 ** 2 + 1) / denom if denom != 0 else np.nan


rows = []
cps = sorted(glob.glob(os.path.join(MAIN, "a*", "checkpoints", "*.checkpoint.json")))
print("checkpoints:", len(cps))
for f in cps:
    rid = int(re.search(r"run_(\d+)\.checkpoint", os.path.basename(f)).group(1))
    adir = os.path.basename(os.path.dirname(os.path.dirname(f)))
    d = json.load(open(f, encoding="utf-8"))
    posts = d.get("posts", [])
    a, g, N = d["alpha"], d["gamma"], d["N"]
    if not posts:
        continue
    by = {}
    for p in posts:
        by.setdefault(p["author_id"], []).append((p["created_step"], p["opinion"]))
    for k in by:
        by[k].sort()
    rec = {"run_id": f"{adir}:{rid}", "alpha": a, "gamma": g}
    for t in TS:
        vals = []
        for i in range(N):
            h = by.get(i, []); cur = None
            for cs, op in h:
                if cs <= t: cur = op
                else: break
            if cur is not None: vals.append(cur)
        if len(vals) >= 4:
            rec[f"BC_{t}"] = bc(vals)
            rec[f"frac_at_bound_{t}"] = float(np.mean(np.abs(vals) >= 0.999))
            rec[f"mean_abs_{t}"] = float(np.mean(np.abs(vals)))
    rows.append(rec)

df = pd.DataFrame(rows)
print("runs:", len(df))
res = {}


def z(x):
    x = np.asarray(x, float); s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; n, k = X.shape; dof = max(n - k, 1)
    s2 = (r @ r) / dof; se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se; p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, t, p, r2


print("\n=== MAIN(T=30) BC 均值 ===")
for t in TS:
    c = f"BC_{t}"
    if c in df:
        print(f"t={t:>3}  BC={df[c].mean():.4f}  越判据(>0.5556)占比={(df[c]>0.5556).mean():.3f}  frac_at_bound={df[f'frac_at_bound_{t}'].mean():.3f}  mean_abs={df[f'mean_abs_{t}'].mean():.3f}")
print("\n=== MAIN BC(30) ~ α ===")
for g in [0.1, 1.0]:
    s = df[np.isclose(df.gamma, g)]
    b, se, tt, pp, r2 = ols(z(s["BC_30"]), [z(s["alpha"])])
    res[f"BC30_g{g}"] = {"beta_std": round(float(b[1]), 4), "t": round(float(tt[1]), 2), "p": round(float(pp[1]), 4), "n": len(s)}
    print(f"γ={g}: β_std={b[1]:+.4f} t={tt[1]:+.2f} p={pp[1]:.4f} n={len(s)}")
b, se, tt, pp, r2 = ols(z(df["BC_30"]), [z(df["alpha"]), z(df["gamma"])])
res["BC30_pooled"] = {"beta_alpha_std": round(float(b[1]), 4), "t": round(float(tt[1]), 2), "p": round(float(pp[1]), 4)}
print(f"合并: β_α={b[1]:+.4f} t={tt[1]:+.2f} p={pp[1]:.4f}")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
df.to_csv(os.path.join(os.path.dirname(OUT), "shape_bc_main_runs.csv"), index=False, encoding="utf-8-sig")
print("written", OUT)
