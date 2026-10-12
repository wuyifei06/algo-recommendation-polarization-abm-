# -*- coding: utf-8 -*-
"""最终数字提取（修正版）。
   LLM 主实验(T=30)  = probe/results_prompt_direction   ← 修正点
   LLM 延长档(T=100) = results_direction/llm_extend_T100
   其余阶段          = results_direction/*
"""
import glob, os, re, json
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
MAIN_DIR = os.path.join(PROJ, "probe_提示词明确性", "results_prompt_direction")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_numbers.json")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")
KEEP = ["PI", "SR", "ER", "EM", "HE", "EE"]


def _load(pat, keyfn):
    seen, uniq = set(), []
    for f in sorted(glob.glob(pat, recursive=True)):
        b = os.path.basename(f); m = RX.match(b)
        if not m:
            continue
        k = keyfn(f, m)
        if k in seen:
            continue
        seen.add(k); uniq.append((k, f))
    out = []
    for k, f in uniq:
        d = pd.read_csv(f)
        if len(d):
            d["run_id"] = k; out.append(d)
    return pd.concat(out, ignore_index=True)


def main_llm():
    return _load(os.path.join(MAIN_DIR, "a*", "runs", "llm_robust_run_*.csv"),
                 lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{int(m.group(1))}")


def stage(st):
    return _load(os.path.join(NEW, st, "**", "runs", "llm_robust_run_*.csv"), lambda f, m: int(m.group(1)))


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
    return b, se, t, p, r2, dof


def at(df, s): return df[df["step"] == s].copy()


M = main_llm()
E = stage("llm_extend_T100")
R = {"_source": {"main": "probe/results_prompt_direction (T=30)", "ext": "results_direction/llm_extend_T100 (T=100)"}}

# 1) MAIN cells
R["main_cells"] = {}
for g in [0.1, 1.0]:
    s = at(M, 30)[np.isclose(at(M, 30).gamma, g)]
    for a, sub in s.groupby("alpha"):
        R["main_cells"][f"{a}|{g}"] = {k: round(float(sub[k].mean()), 4) for k in KEEP}
        R["main_cells"][f"{a}|{g}"]["n"] = int(len(sub))
# 2) MAIN dose 所有指标
R["main_dose"] = {}
for key, df, st in [("t30", M, 30), ("t100", E, 100)]:
    D = at(df, st); dd = {}
    for g in [0.1, 1.0]:
        sub = D[np.isclose(D.gamma, g)]
        e = {}
        for m in ["PI", "SR", "EM", "ER"]:
            b, se, t, p, r2, _ = ols(z(sub[m]), [z(sub.alpha)])
            e[m] = {"beta_std": round(float(b[1]), 4), "t": round(float(t[1]), 2), "p": round(float(p[1]), 5), "R2": round(float(r2), 4),
                    "ci_std": [round(float(b[1]-1.96*se[1]), 4), round(float(b[1]+1.96*se[1]), 4)]}
        lo, hi = sub[sub.alpha == sub.alpha.min()], sub[sub.alpha == sub.alpha.max()]
        sp = np.sqrt(((len(lo)-1)*lo.PI.var(ddof=1)+(len(hi)-1)*hi.PI.var(ddof=1))/(len(lo)+len(hi)-2))
        e["cohen_d"] = round(float((hi.PI.mean()-lo.PI.mean())/sp), 2)
        b2, se2, _, _, _, _ = ols(sub.PI.to_numpy(float), [sub.alpha.to_numpy(float)])
        e["slope_raw"] = round(float(b2[1]), 5)
        e["ci_raw"] = [round(float(b2[1]-1.96*se2[1]), 5), round(float(b2[1]+1.96*se2[1]), 5)]
        dd[f"g{g}"] = e
    bb, ss, tt, pp, rr, _ = ols(z(D.PI), [z(D.alpha), z(D.gamma)])
    dd["pooled"] = {"beta_alpha": round(float(bb[1]), 4), "t": round(float(tt[1]), 2), "p": round(float(pp[1]), 7), "R2": round(float(rr), 4),
                    "beta_gamma": round(float(bb[2]), 4), "t_gamma": round(float(tt[2]), 2), "p_gamma": round(float(pp[2]), 6)}
    R["main_dose"][key] = dd
# 3) MAIN timeline
R["main_timeline"] = {int(s): {m: round(float(sub[m].mean()), 4) for m in KEEP} for s, sub in M.groupby("step")}
# 4) MAIN upswing
up = {}
for g in [0.1, 1.0]:
    d = M[np.isclose(M.gamma, g)]
    p1 = d[d.step == 1].set_index("run_id").PI; p30 = d[d.step == 30].set_index("run_id").PI
    diff = (p30 - p1).dropna()
    up[f"g{g}"] = {"dPI": round(float(diff.mean()), 4), "sd": round(float(diff.std(ddof=1)), 4),
                   "t": round(float(diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff)))), 2),
                   "n_up": int((diff > 0).sum()), "n": int(len(diff))}
R["main_upswing"] = up
# 5) MAIN mediation
def mediation(D, st):
    D = at(D, st); az, gz = z(D.alpha), z(D.gamma); o = {}; rng = np.random.default_rng(11); n = len(D)
    for Mv in ["HE", "EE", "EM"]:
        mz, pz = z(D[Mv]), z(D.PI)
        a = ols(mz, [az, gz])[0][1]; bb = ols(pz, [az, gz, mz])[0][3]
        bt = ols(pz, [az, gz, mz])[2][3]; bp = ols(pz, [az, gz, mz])[3][3]
        at_ = ols(mz, [az, gz])[2][1]; ap = ols(mz, [az, gz])[3][1]
        bs = []
        for _ in range(5000):
            i = rng.integers(0, n, n)
            bs.append(ols(mz[i], [az[i], gz[i]])[0][1]*ols(pz[i], [az[i], gz[i], mz[i]])[0][3])
        bs = np.array(bs)
        o[Mv] = {"a": round(float(a), 4), "a_t": round(float(at_), 2), "a_p": round(float(ap), 5),
                 "b": round(float(bb), 4), "b_t": round(float(bt), 2), "b_p": round(float(bp), 5),
                 "ind": round(float(a*bb), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        o[Mv]["par_HE_EE"] = None
    c = ols(z(D.PI), [az, gz])[0][1]
    o["c_total"] = round(float(c), 4)
    # 平行中介 HE+EE
    mzH, mzE, pz = z(D.HE), z(D.EE), z(D.PI)
    aH = ols(mzH, [az, gz])[0][1]; aE = ols(mzE, [az, gz])[0][1]
    coef = ols(pz, [az, gz, mzH, mzE])[0]
    o["parallel_HE_EE"] = round(float(aH*coef[3] + aE*coef[4]), 4)
    # 直接效应
    o["c_prime"] = round(float(ols(pz, [az, gz, z(D.HE)])[0][1]), 4)
    return o
R["main_mediation"] = mediation(M, 30)
R["qwen_mediation"] = mediation(stage("llm_qwen_full"), 30)
# 6) MAIN poly fit
poly = {}
for g in [0.1, 1.0]:
    s = at(M, 30)[np.isclose(at(M, 30).gamma, g)]
    x, y = s.alpha.to_numpy(float), s.PI.to_numpy(float)
    r2_1 = np.corrcoef(x, y)[0, 1]**2
    c2 = np.polyfit(x, y, 2)
    yhat = np.polyval(c2, x); r2_2 = 1-((y-yhat)**2).sum()/((y-y.mean())**2).sum()
    # 二次项 p 值
    X = np.column_stack([np.ones(len(x)), x, x**2]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; dof = len(x)-3; s2 = (r@r)/dof
    se = np.sqrt(np.diag(s2*np.linalg.pinv(X.T@X))); tp = b[2]/se[2]
    p2 = 2*(1-stats.t.cdf(abs(tp), dof))
    rho = stats.spearmanr(x, y)[0]
    poly[f"g{g}"] = {"r2_lin": round(float(r2_1), 4), "r2_quad": round(float(r2_2), 4), "p_quad": round(float(p2), 4), "spearman": round(float(rho), 3)}
R["main_poly"] = poly
# 7) randomfeed paired vs MAIN
rf = at(stage("llm_randomfeed_control"), 30)
al = at(M, 30); al = al[np.isclose(al.alpha, 1.0)]
mm = pd.merge(rf, al, on=["gamma", "seed"], suffixes=("_rf", "_al"))
R["randomfeed"] = {"n": len(mm)}
for k in ["PI", "HE", "ER", "EM", "SR"]:
    d = (mm[f"{k}_rf"] - mm[f"{k}_al"]).to_numpy(float)
    se = d.std(ddof=1)/np.sqrt(len(d))
    R["randomfeed"][k] = {"rf": round(float(mm[f"{k}_rf"].mean()), 4), "algo": round(float(mm[f"{k}_al"].mean()), 4),
                          "diff": round(float(d.mean()), 4), "ci95": [round(float(d.mean()-1.96*se), 4), round(float(d.mean()+1.96*se), 4)],
                          "t": round(float(d.mean()/se), 2)}
# 8) ablations paired vs MAIN
base = at(M, 30).set_index("seed").PI - at(M, 1).set_index("seed").PI
base = base[~base.index.duplicated()]
R["ablation"] = {"base_dPI": round(float(base.mean()), 4), "n": len(base)}
for nm in ["llm_ablation_no_post", "llm_ablation_no_emotion", "llm_ablation_positive_control"]:
    arm = stage(nm)
    aseed = at(arm, 30).set_index("seed").PI - at(arm, 1).set_index("seed").PI
    aseed = aseed[~aseed.index.duplicated()]
    j = pd.concat([base.rename("base"), aseed.rename("arm")], axis=1).dropna()
    d = j["arm"] - j["base"]; se = d.std(ddof=1)/np.sqrt(len(d))
    R["ablation"][nm] = {"arm_dPI": round(float(j["arm"].mean()), 4), "diff": round(float(d.mean()), 4),
                         "ci95": [round(float(d.mean()-1.96*se), 4), round(float(d.mean()+1.96*se), 4)],
                         "t": round(float(d.mean()/se), 2), "n": len(j), "frac_neg": round(float((d < 0).mean()), 3)}
# 9) n100
n1 = at(stage("llm_n100_scale"), 30)
R["n100"] = {"cells": {f"{a}|{g}": round(float(s.PI.mean()), 4) for (a, g), s in n1.groupby(["alpha", "gamma"])},
             "overall": round(float(n1.PI.mean()), 4)}
b, se, t, p, r2, _ = ols(z(n1.PI), [z(n1.alpha)])
R["n100"]["dose"] = {"beta_std": round(float(b[1]), 4), "t": round(float(t[1]), 2), "p": round(float(p[1]), 4), "R2": round(float(r2), 4), "n": len(n1)}
# 10) qwen cells
qw = at(stage("llm_qwen_full"), 30)
R["qwen"] = {"cells": {f"{a}|{g}": {"PI": round(float(s.PI.mean()), 4), "n": len(s)} for (a, g), s in qw.groupby(["alpha", "gamma"])},
             "overall_PI30": round(float(qw.PI.mean()), 4),
             "EM30": round(float(qw.EM.mean()), 4), "SR30": round(float(qw.SR.mean()), 4),
             "ER30": round(float(qw.ER.mean()), 4), "HE30": round(float(qw.HE.mean()), 4),
             "PI1": round(float(at(stage("llm_qwen_full"), 1).PI.mean()), 4)}
qd = {}
for g in [0.1, 1.0]:
    sub = qw[np.isclose(qw.gamma, g)]
    b, se, t, p, r2, _ = ols(z(sub.PI), [z(sub.alpha)])
    qd[f"g{g}"] = {"beta_std": round(float(b[1]), 4), "t": round(float(t[1]), 2), "p": round(float(p[1]), 4), "n": len(sub)}
R["qwen"]["dose"] = qd
# DeepSeek 行为侧 t30
R["ds_behavior"] = {"EM30": round(float(at(M, 30).EM.mean()), 4), "SR30": round(float(at(M, 30).SR.mean()), 4),
                    "ER30": round(float(at(M, 30).ER.mean()), 4), "HE30": round(float(at(M, 30).HE.mean()), 4),
                    "PI1": round(float(at(M, 1).PI.mean()), 4), "PI30": round(float(at(M, 30).PI.mean()), 4)}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("written", OUT)
print()
print("MAIN dose t30:", json.dumps(R["main_dose"]["t30"], ensure_ascii=False, indent=1))
print("MAIN mediation:", json.dumps(R["main_mediation"], ensure_ascii=False))
print("randomfeed:", json.dumps(R["randomfeed"], ensure_ascii=False))
print("ablation:", json.dumps(R["ablation"], ensure_ascii=False))
print("poly:", json.dumps(R["main_poly"], ensure_ascii=False))
print("upswing:", json.dumps(R["main_upswing"], ensure_ascii=False))
print("ds_behavior:", json.dumps(R["ds_behavior"], ensure_ascii=False))
