# -*- coding: utf-8 -*-
"""规则臂时间交互检验：base(T=200) vs T400(T=400)。

背景：LLM 臂的时间交互已补算（α×批次 p=0.030）。规则臂在结构变异中存在
同种子延长数据（base 与 T400），可做同轨迹配对检验，回答：
  α 的效应在 t=200 与 t=400 是否不同？
先验证两批是否同轨迹（确定性系统，同 seed 应逐位一致）。
"""
import os, glob, re, json
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

RS = _PROJ_ROOT + r"\results_replay_20261010\rule_structure"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rule_time_interaction.json")
RES = {}


def load(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "rule_run_*.csv"))):
        d = pd.read_csv(f)
        if len(d):
            d["run_id"] = int(re.search(r"rule_run_(\d+)\.csv", f).group(1))
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


BASE = load(os.path.join(RS, "base"))
T400 = load(os.path.join(RS, "T400"))
print("base runs:", BASE.run_id.nunique(), " T400 runs:", T400.run_id.nunique())
print("base alpha:", sorted(BASE.alpha.unique()), " gamma:", sorted(BASE.gamma.unique()))

# ---- (0) 同轨迹验证：base 末步与 T400 的 step==200 ----
b200 = BASE[BASE.step == 200].set_index("run_id").sort_index()
t200 = T400[T400.step == 200].set_index("run_id").sort_index()
same_idx = b200.index.intersection(t200.index)
dPI = (b200.loc[same_idx, "PI"] - t200.loc[same_idx, "PI"]).abs()
print("\n[同轨迹验证] 共同 run:", len(same_idx), " max|ΔPI@t200| =", dPI.max())
RES["same_traj"] = dict(n=int(len(same_idx)), max_abs_dPI_at200=float(dPI.max()))

# α/γ/seed 一致性
chk = pd.merge(b200.reset_index()[["run_id", "alpha", "gamma", "seed"]],
               t200.reset_index()[["run_id", "alpha", "gamma", "seed"]],
               on="run_id", suffixes=("_b", "_t"))
print("alpha 一致:", (chk.alpha_b == chk.alpha_t).all(),
      " gamma 一致:", (chk.gamma_b == chk.gamma_t).all(),
      " seed 一致:", (chk.seed_b == chk.seed_t).all())

# ---- (1) 末步斜率对比 ----
t400 = T400[T400.step == 400].set_index("run_id").sort_index()
for nm, df in (("t=200", b200), ("t=400", t400)):
    y = df.PI.values
    Xa = df.alpha.astype(float).values
    Xg = df.gamma.astype(float).values
    X = np.column_stack([np.ones(len(y)), Xa, Xg])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    s2 = (r @ r) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    print(f"\n[{nm}] PI ~ α + γ：α b={b[1]:+.5f} t={t[1]:+.2f} p={p[1]:.3g} n={n}")
    RES[f"slope_{nm}"] = dict(b=float(b[1]), t=float(t[1]), p=float(p[1]), n=int(n))

# ---- (2) 配对时间交互：ΔPI(200→400) ~ α + γ ----
mg = pd.merge(t400.reset_index()[["run_id", "PI", "alpha", "gamma"]].rename(columns={"PI": "PI400"}),
              b200.reset_index()[["run_id", "PI"]].rename(columns={"PI": "PI200"}),
              on="run_id")
mg["dPI"] = mg.PI400 - mg.PI200
y = mg.dPI.values
X = np.column_stack([np.ones(len(y)), mg.alpha.astype(float).values, mg.gamma.astype(float).values])
b, *_ = np.linalg.lstsq(X, y, rcond=None)
r = y - X @ b
n, k = X.shape
s2 = (r @ r) / (n - k)
se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
t = b / se
p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
print(f"\n[ΔPI(200→400) ~ α + γ] α b={b[1]:+.5f} t={t[1]:+.2f} p={p[1]:.3g} n={n}")
print(f"    截距(平均增量)={b[0]:+.4f}")
RES["dPI"] = dict(b_alpha=float(b[1]), t=float(t[1]), p=float(p[1]),
                  intercept=float(b[0]), n=int(n))

# γ 分组
for g in (0.1, 1.0):
    s = mg[mg.gamma.astype(float) == g]
    y = s.dPI.values
    X = np.column_stack([np.ones(len(y)), s.alpha.astype(float).values])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    s2 = (r @ r) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    print(f"  γ={g}: ΔPI~α  b={b[1]:+.5f} t={t[1]:+.2f} p={p[1]:.3g} n={n}")
    RES[f"dPI_g{g}"] = dict(b=float(b[1]), t=float(t[1]), p=float(p[1]), n=int(n))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
print("\nSaved:", OUT)
