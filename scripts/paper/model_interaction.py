# -*- coding: utf-8 -*-
"""模型类型×α 交互检验：Rule 受限规模(N=30,T=30) vs LLM 主实验(N=30,T=30)。

背景：论文 H4 称两类主体“方向一致”。但“各自显著”不等于“两者显著不同”，
须做交互项检验，才能支撑“方向一致、量级不同”的准确表述。

设定：α 五档对齐 {0, 0.5, 1.0, 1.5, 2.0}，含 γ 协变量。
  Rule 受限规模：5α×2γ×5R = 50 runs
  LLM 主实验（五档子集）：5α×2γ×10R = 100 runs
  合计 n=150
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
REPLAY = os.path.join(BASE, "results_replay_20261010")
PROBE = os.path.join(PROJ, "probe_提示词明确性", "results_prompt_direction")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_interaction.json")
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


def z(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


# Rule 受限规模：末步
RS = pd.read_csv(os.path.join(REPLAY, "rule_small_d4", "rule_small_summary.csv"))
weird = [c for c in RS.columns if c.lower() in ("step", "t")]
if weird:
    RS = RS.sort_values(weird[0]).groupby(["alpha", "gamma"]).tail(1) if False else RS
print("RS 列:", list(RS.columns)[:14])
print("RS 行数:", len(RS), " alpha 取值:", sorted(RS.alpha.unique()))

# LLM 主实验：末步、取 α 五档子集
MAIN = load_llm(PROBE,
                lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{m.group(1)}",
                "a*/runs/llm_robust_run_*.csv")
main30 = MAIN.sort_values("step").groupby("run_id").tail(1).copy()
main30["alpha"] = main30["alpha"].astype(float)
main30["gamma"] = main30["gamma"].astype(float)
sub = main30[main30.alpha.isin([0.0, 0.5, 1.0, 1.5, 2.0])].copy()
print("LLM 五档子集:", len(sub), " runs; gamma:", sorted(sub.gamma.unique()))

# 组装
rs = RS.copy()
rs["alpha"] = rs["alpha"].astype(float)
rs["gamma"] = rs["gamma"].astype(float)
rs["model"] = 0.0
sub["model"] = 1.0
keep = ["alpha", "gamma", "PI", "model"]
d = pd.concat([rs[keep], sub[keep]], ignore_index=True)
print("合并 n =", len(d))

d["axm"] = d["alpha"] * d["model"]

# 原始尺度交互
res, n, dof = ols(d.PI, [d.alpha, d.model, d.axm, d.gamma],
                  ["intercept", "alpha", "model(LLM)", "alpha×model", "gamma"])
print("\n=== 原始尺度：PI ~ α + model + α×model + γ（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["raw"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# 标准化尺度交互
d["PIz"] = z(d.PI)
res, n, dof = ols(d.PIz, [z(d.alpha), d.model, z(d.alpha) * d.model, z(d.gamma)],
                  ["intercept", "alpha_z", "model(LLM)", "alpha_z×model", "gamma_z"])
print("\n=== 标准化尺度：z(PI) ~ z(α) + model + z(α)×model + z(γ)（n=%d） ===" % n)
for k_, v in res.items():
    print(f"  {k_:14s} b={v['b']:+.4f}  t={v['t']:+.2f}  p={v['p']:.4g}")
RES["std"] = dict(n=int(n), **{k_: v for k_, v in res.items()})

# 分模型斜率（合并 γ，标准化）
for m, nm in ((0.0, "Rule"), (1.0, "LLM")):
    s = d[d.model == m]
    res, n, dof = ols(z(s.PI), [z(s.alpha), z(s.gamma)], ["intercept", "alpha_z", "gamma_z"])
    v = res["alpha_z"]
    print(f"\n{nm}：z(PI)~z(α)+z(γ)  alpha_z b={v['b']:+.4f} t={v['t']:+.2f} p={v['p']:.4g} n={n}")
    RES[f"slope_{nm}"] = dict(n=int(n), **{k_: vv for k_, vv in res.items()})

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
print("\nSaved:", OUT)
