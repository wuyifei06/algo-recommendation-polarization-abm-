# -*- coding: utf-8 -*-
"""三重交互检验：α×视野×模型（DeepSeek/Qwen × T=30/T=100）。

方法与已发表的 time_interaction.py（DS 批内交互 +0.028, t=2.18, n=320）同源：
逐 run 载入时间序列 CSV，取每 run 末步为一条观测（每 run 恰好一条，无需聚类），
四批合并做 OLS：PI ~ α + batch + model + γ + 全部二阶 + α×batch×model。

输出：three_way_interaction.json
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three_way_interaction.json")

RES = {}


def load_runs(root, keyfn=None, pat=None, expect_step=None, tag=""):
    """载入逐 run 时间序列，取每 run 末步为一条观测。"""
    if pat is None:
        pat = "**/runs/llm_robust_run_*.csv"
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
        if not len(d):
            continue
        last = d.sort_values("step").iloc[-1]
        rows.append({"run_key": k, "step": int(last["step"]), "PI": float(last["PI"]),
                     "alpha": float(last["alpha"]), "gamma": float(last["gamma"])})
    df = pd.DataFrame(rows)
    if expect_step is not None and len(df):
        bad = df[df.step != expect_step]
        if len(bad):
            print(f"WARN {tag}: {len(bad)} runs 末步 != {expect_step}（已保留）")
    print(f"{tag}: {len(df)} runs")
    RES[f"n_{tag}"] = int(len(df))
    return df


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


# ---------- 载入四批 ----------
ds30 = load_runs(PROBE,
                 lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{m.group(1)}",
                 "a*/runs/llm_robust_run_*.csv", 30, "DS_T30")
ds100 = load_runs(os.path.join(NEW, "llm_extend_T100"), None, None, 100, "DS_T100")
qw30 = load_runs(os.path.join(NEW, "llm_qwen_full"), None, None, 30, "Qwen_T30")
qw100 = load_runs(os.path.join(NEW, "llm_qwen_extend_T100"), None, None, 100, "Qwen_T100")

# ---------- 交叉验证：DS 批内交互应复现已发表 +0.028 ----------
d_ds = pd.concat([ds30.assign(batch=0.0), ds100.assign(batch=1.0)], ignore_index=True)
d_ds["axb"] = d_ds.alpha * d_ds.batch
res, n, dof = ols(d_ds.PI, [d_ds.alpha, d_ds.batch, d_ds.axb, d_ds.gamma],
                  ["intercept", "alpha", "batch", "alpha×batch", "gamma"])
print(f"\n[交叉验证] DS 批内 α×batch: b={res['alpha×batch']['b']:+.4f} "
      f"t={res['alpha×batch']['t']:+.2f} p={res['alpha×batch']['p']:.4g} (n={n})；已发表 +0.028/t=2.18/p=0.030")
RES["ds_interaction_check"] = dict(n=int(n), **res["alpha×batch"])

d_qw = pd.concat([qw30.assign(batch=0.0), qw100.assign(batch=1.0)], ignore_index=True)
d_qw["axb"] = d_qw.alpha * d_qw.batch
res, n, dof = ols(d_qw.PI, [d_qw.alpha, d_qw.batch, d_qw.axb, d_qw.gamma],
                  ["intercept", "alpha", "batch", "alpha×batch", "gamma"])
print(f"[交叉验证] Qwen 批内 α×batch: b={res['alpha×batch']['b']:+.4f} "
      f"t={res['alpha×batch']['t']:+.2f} p={res['alpha×batch']['p']:.4g} (n={n})")
RES["qw_interaction"] = dict(n=int(n), **res["alpha×batch"])

# ---------- 三重交互 ----------
d = pd.concat([d_ds.assign(model=0.0), d_qw.assign(model=1.0)], ignore_index=True)
d["axb"] = d.alpha * d.batch
d["axm"] = d.alpha * d.model
d["bxm"] = d.batch * d.model
d["axbm"] = d.alpha * d.batch * d.model
res, n, dof = ols(d.PI, [d.alpha, d.batch, d.model, d.gamma,
                         d.axb, d.axm, d.bxm, d.axbm],
                  ["intercept", "alpha", "batch(T100)", "model(Qwen)", "gamma",
                   "alpha×batch", "alpha×model", "batch×model", "α×视野×模型"])
print(f"\n=== 三重交互（n={n}）===")
for k_, v in res.items():
    print(f"  {k_:16s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["three_way"] = dict(n=int(n), **res)

OUT_JSON = OUT
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(RES, fh, ensure_ascii=False, indent=1)
print("JSON ->", OUT_JSON)
