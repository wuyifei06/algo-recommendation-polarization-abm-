# -*- coding: utf-8 -*-
"""时间×α 交互检验：LLM 主实验(T=30) vs 延长档(T=100)。

背景：论文报告 α 效应在 T=30 显著、T=100 不显著，但“显著/不显著”本身
不构成“两时间窗口效应有差异”的证据，须做交互项检验。

两批为同 run→seed 映射的独立实现（非同一轨迹），故做非配对交互回归：
  (1) 末步口径：主实验 PI(t=30) vs 延长档 PI(t=100)
  (2) 对齐点口径：主实验 PI(t=30) vs 延长档 PI(t=30)
  (3) 延长档内部上行速率：ΔPI(t30→t100) ~ α
交互项显著 = 时间窗口改变 α 的效应。
"""
import os, glob, re, json
import numpy as np
import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_BASE_ROOT = _os.getenv("ABM_BASE_ROOT", r"D:\21130\Desktop\大创过程")
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
_VENV_PY = _os.getenv("ABM_VENV_PY", r"C:\Users\21130\.workbuddy\binaries\python\envs\default\Scripts\python.exe")
# -------------------------------------------------------------------

PROJ = _BASE_ROOT
BASE = os.path.join(PROJ, "实验代码", "最终代码和数据")
NEW = os.path.join(BASE, "results_direction")
PROBE = os.path.join(PROJ, "probe_提示词明确性", "results_prompt_direction")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "time_interaction.json")

RES = {}


def load_llm(root, keyfn=None, pat="**/runs/llm_robust_run_*.csv"):
    seen, rows = set(), []
    for f in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
        m = RX.match(os.path.basename(f))
        if not m:
            continue
        k = keyfn(f, m) if keyfn else int(m.group(1))
        if k in seen:
            continue
        seen.add(k)
        d = pd.read_csv(f)
        if len(d):
            d["run_id"] = k
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def ols(y, cols, names=None):
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = (r @ r) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    if names is None:
        names = [f"b{i}" for i in range(k)]
    return {nm: dict(b=float(bb), se=float(ss), t=float(tt), p=float(pp))
            for nm, bb, ss, tt, pp in zip(names, b, se, t, p)}, n, dof


# ---------- 读取 ----------
MAIN = load_llm(PROBE,
                lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{m.group(1)}",
                "a*/runs/llm_robust_run_*.csv")
EXT = load_llm(os.path.join(NEW, "llm_extend_T100"))

print("主实验行数(含全步):", len(MAIN), " runs:", MAIN.run_id.nunique())
print("延长档行数(含全步):", len(EXT), " runs:", EXT.run_id.nunique())
RES["n_main_runs"] = int(MAIN.run_id.nunique())
RES["n_ext_runs"] = int(EXT.run_id.nunique())

# 取主实验末步（=t30）与延长档 t=30 / t=100
main30 = MAIN.sort_values("step").groupby("run_id").tail(1).copy()
assert main30.step.max() == 30, main30.step.max()
ext30 = EXT[EXT.step == 30].copy()
ext100 = EXT.sort_values("step").groupby("run_id").tail(1).copy()
assert ext100.step.max() == 100, ext100.step.max()
print("main@t30:", len(main30), " ext@t30:", len(ext30), " ext@t100:", len(ext100))

main30["batch"] = 0.0
ext30b = ext30.copy(); ext30b["batch"] = 1.0
ext100b = ext100.copy(); ext100b["batch"] = 1.0

# ---------- (1) 末步口径交互：PI ~ α + batch + α×batch + γ ----------
d1 = pd.concat([main30, ext100b], ignore_index=True)
d1["alpha"] = d1["alpha"].astype(float)
d1["gamma"] = d1["gamma"].astype(float)
d1["axb"] = d1["alpha"] * d1["batch"]

res, n, dof = ols(d1.PI, [d1.alpha, d1.batch, d1.axb, d1.gamma],
                  ["intercept", "alpha", "batch(T100)", "alpha×batch", "gamma"])
print("\n=== (1) 末步口径：PI ~ α + batch + α×batch + γ（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["step_end"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# 分 γ 的交互（去掉 γ 协变量，各自做）
RES["step_end_by_gamma"] = {}
for g in ("0.1", "1.0"):
    dg = d1[d1.gamma == float(g)]
    res, n, dof = ols(dg.PI, [dg.alpha, dg.batch, dg.axb],
                      ["intercept", "alpha", "batch(T100)", "alpha×batch"])
    print(f"\n--- 分 γ={g}（n={n}）---")
    for k_, v in res.items():
        print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
    RES["step_end_by_gamma"][g] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# ---------- (2) 对齐点口径交互：主实验末步(t30) vs 延长档 t=30 ----------
d2 = pd.concat([main30, ext30b], ignore_index=True)
d2["alpha"] = d2["alpha"].astype(float)
d2["gamma"] = d2["gamma"].astype(float)
d2["axb"] = d2["alpha"] * d2["batch"]
res, n, dof = ols(d2.PI, [d2.alpha, d2.batch, d2.axb, d2.gamma],
                  ["intercept", "alpha", "batch(T100)", "alpha×batch", "gamma"])
print("\n=== (2) 对齐点(t=30)：主实验末步 vs 延长档 t=30（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["align30"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# ---------- (3) 延长档内部：上行速率 ΔPI ~ α ----------
a30 = ext30.set_index("run_id")[["PI"]].rename(columns={"PI": "PI30"})
a100 = ext100.set_index("run_id")[["PI", "alpha", "gamma"]].rename(columns={"PI": "PI100"})
mg = a100.join(a30, how="inner").reset_index()
mg["dPI"] = mg.PI100 - mg.PI30
mg["alpha"] = mg["alpha"].astype(float)
mg["gamma"] = mg["gamma"].astype(float)
res, n, dof = ols(mg.dPI, [mg.alpha, mg.gamma], ["intercept", "alpha", "gamma"])
print("\n=== (3) 延长档内部：ΔPI(t30→t100) ~ α + γ（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
b, se, t, p = res["alpha"]["b"], res["alpha"]["se"], res["alpha"]["t"], res["alpha"]["p"]
print(f"  平均增量对 α 的斜率: {b:+.4f}/单位α (t={t:+.2f}, p={p:.2g})")
RES["ext_upslope"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# γ=1.0 下 α=0 与 α=2 的平均增量
for a in (0.0, 2.0):
    s = mg[(mg.alpha == a) & (mg.gamma == 1.0)]
    print(f"  γ=1.0, α={a}: ΔPI 均值 {s.dPI.mean():+.4f} (n={len(s)})")
    RES[f"ext_dPI_a{a}_g1.0"] = float(s.dPI.mean())

# ---------- (4) 延长档 t=30 斜率（复核正文报的 −0.034/单位α, t=−4.31） ----------
res, n, dof = ols(ext30.PI, [ext30.alpha.astype(float), ext30.gamma.astype(float)],
                  ["intercept", "alpha", "gamma"])
print("\n=== (4) 延长档在 t=30 的 PI~α 原始斜率（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["ext_at30_slope"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# 主实验末步原始斜率（对照）
res, n, dof = ols(main30.PI, [main30.alpha.astype(float), main30.gamma.astype(float)],
                  ["intercept", "alpha", "gamma"])
print("\n=== (5) 主实验末步(t=30) PI~α 原始斜率（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["main_at30_slope"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
print("\nSaved:", OUT)
