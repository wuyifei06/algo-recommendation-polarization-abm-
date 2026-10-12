# -*- coding: utf-8 -*-
"""方向版（results_direction）全量数字提取 —— 供论文重写使用。"""
import os, glob, re, json
import numpy as np
import pandas as pd

BASE = _PROJ_ROOT
NEW = os.path.join(BASE, "results_direction")
OUT = _os.path.join(_WORK_ROOT, r"new_numbers.json")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")


def load(root, stage):
    seen, uniq = set(), []
    for f in sorted(glob.glob(os.path.join(root, stage, "**", "runs", "llm_robust_run_*.csv"), recursive=True)):
        b = os.path.basename(f); m = RX.match(b)
        if m and b not in seen:
            seen.add(b); uniq.append((int(m.group(1)), f))
    out = []
    for rid, f in uniq:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if len(d):
            d["run_id"] = rid
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else None


def z(x):
    x = np.asarray(x, float); s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; n, k = X.shape; dof = max(n - k, 1)
    s2 = (r @ r) / dof; se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, t, p, r2, dof


from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------


def at(df, s):
    return df[df["step"] == s].copy()


def last(df):
    return df.loc[df.groupby("run_id")["step"].idxmax()].copy()


R = {}

# ---------- 1. LLM 主实验（extend_T100） ----------
ext = load(NEW, "llm_extend_T100")
R["main"] = {}
for s in [30, 100]:
    D = at(ext, s)
    cells = {}
    for (a, g), sub in D.groupby(["alpha", "gamma"]):
        cells[f"{a}|{g}"] = {k: round(float(sub[k].mean()), 4) for k in ["PI", "SR", "ER", "EM", "HE", "EE"]}
        cells[f"{a}|{g}"]["n"] = int(len(sub))
    R["main"][f"cells_t{s}"] = cells
# 剂量响应
for s in [30, 100]:
    D = at(ext, s); dd = {}
    for g in sorted(D["gamma"].unique()):
        sub = D[D["gamma"] == g]
        b, se, t, p, r2, dof = ols(z(sub["PI"]), [z(sub["alpha"])])
        b2, se2, t2, p2, r22, _ = ols(sub["PI"].to_numpy(float), [sub["alpha"].to_numpy(float)])
        dd[f"g{g}"] = {"beta_std": round(float(b[1]), 4), "t": round(float(t[1]), 3), "p": round(float(p[1]), 5),
                       "R2": round(float(r2), 4), "n": len(sub),
                       "ci_std": [round(float(b[1] - 1.96 * se[1]), 4), round(float(b[1] + 1.96 * se[1]), 4)],
                       "slope_raw": round(float(b2[1]), 5),
                       "ci_raw": [round(float(b2[1] - 1.96 * se2[1]), 5), round(float(b2[1] + 1.96 * se2[1]), 5)]}
        # Cohen's d  α=0 vs α=max
        lo, hi = sub[sub["alpha"] == sub["alpha"].min()], sub[sub["alpha"] == sub["alpha"].max()]
        sp = np.sqrt(((len(lo) - 1) * lo["PI"].var(ddof=1) + (len(hi) - 1) * hi["PI"].var(ddof=1)) / (len(lo) + len(hi) - 2))
        dd[f"g{g}"]["cohen_d"] = round(float((hi["PI"].mean() - lo["PI"].mean()) / sp), 3)
    bb, ss, tt, pp, rr, _ = ols(z(D["PI"]), [z(D["alpha"]), z(D["gamma"])])
    dd["pooled"] = {"beta_alpha_std": round(float(bb[1]), 4), "t": round(float(tt[1]), 3), "p": round(float(pp[1]), 6),
                    "R2": round(float(rr), 4), "n": len(D), "beta_gamma_std": round(float(bb[2]), 4),
                    "t_gamma": round(float(tt[2]), 3), "p_gamma": round(float(pp[2]), 6)}
    R["main"][f"dose_t{s}"] = dd
# 时序
tl = {}
for s, sub in ext.groupby("step"):
    tl[int(s)] = {"PI": round(float(sub["PI"].mean()), 4), "EM": round(float(sub["EM"].mean()), 4),
                  "SR": round(float(sub["SR"].mean()), 4), "ER": round(float(sub["ER"].mean()), 4),
                  "HE": round(float(sub["HE"].mean()), 4), "EE": round(float(sub["EE"].mean()), 4)}
R["main"]["timeline"] = tl
# 上行幅度 by gamma + 上升 run 比例
up = {}
for g in sorted(ext["gamma"].unique()):
    d = ext[ext["gamma"] == g]
    p1 = d[d["step"] == 1].set_index("run_id")["PI"]
    for s in [30, 100]:
        ps = d[d["step"] == s].set_index("run_id")["PI"]
        diff = (ps - p1).dropna()
        up[f"g{g}_t{s}"] = {"dPI": round(float(diff.mean()), 4), "sd": round(float(diff.std(ddof=1)), 4),
                            "t": round(float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))), 2),
                            "n_up": int((diff > 0).sum()), "n": int(len(diff))}
R["main"]["upswing"] = up
# 末10步斜率（t=91..100）
sl = []
for rid, d in ext.groupby("run_id"):
    w = d[(d["step"] >= 91) & (d["step"] <= 100)]
    sl.append(np.polyfit(w["step"], w["PI"], 1)[0])
R["main"]["tail_slope_mean"] = round(float(np.mean(sl)), 5)
R["main"]["tail_up_frac"] = round(float(np.mean([x > 0 for x in sl])), 3)

# ---------- 2. 随机 feed 对照 ----------
rf = last(load(NEW, "llm_randomfeed_control"))
al = at(ext, 30); al = al[al["alpha"] == 1.0]
m = pd.merge(rf, al, on=["gamma", "seed"], suffixes=("_rf", "_al"))
R["randomfeed"] = {"n": len(m)}
for k in ["PI", "HE", "EM", "SR", "ER"]:
    d = (m[f"{k}_rf"] - m[f"{k}_al"]).to_numpy(float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    R["randomfeed"][k] = {"rf": round(float(m[f"{k}_rf"].mean()), 4), "algo": round(float(m[f"{k}_al"].mean()), 4),
                          "diff": round(float(d.mean()), 4),
                          "ci95": [round(float(d.mean() - 1.96 * se), 4), round(float(d.mean() + 1.96 * se), 4)]}

# ---------- 3. 在线消融 ----------
def dpi(df, s):
    a = at(df, s).set_index("run_id")["PI"]; b = at(df, 1).set_index("run_id")["PI"]
    return a - b

base = dpi(ext, 30)
R["ablation"] = {}
for nm in ["llm_ablation_no_post", "llm_ablation_no_emotion", "llm_ablation_positive_control"]:
    arm = load(NEW, nm)
    if arm is None:
        continue
    bseed = at(ext, 30)[["seed", "PI"]].set_index("seed")["PI"] - at(ext, 1)[["seed", "PI"]].set_index("seed")["PI"]
    aseed = at(arm, 30)[["seed", "PI"]].set_index("seed")["PI"] - at(arm, 1)[["seed", "PI"]].set_index("seed")["PI"]
    j = pd.concat([bseed.rename("base"), aseed.rename("arm")], axis=1).dropna()
    d = j["arm"] - j["base"]; se = d.std(ddof=1) / np.sqrt(len(d))
    from scipy import stats as st
    R["ablation"][nm] = {"base_dPI": round(float(j["base"].mean()), 4), "arm_dPI": round(float(j["arm"].mean()), 4),
                         "diff": round(float(d.mean()), 4),
                         "ci95": [round(float(d.mean() - 1.96 * se), 4), round(float(d.mean() + 1.96 * se), 4)],
                         "t": round(float(d.mean() / se), 2), "n": len(j),
                         "frac_neg": round(float((d < 0).mean()), 3)}
    # 臂末步水平
    R["ablation"][nm]["arm_PI30"] = round(float(at(arm, 30)["PI"].mean()), 4)

# ---------- 4. N=100 ----------
n1 = load(NEW, "llm_n100_scale")
R["n100"] = {"cells": {f"{a}|{g}": round(float(s["PI"].mean()), 4) for (a, g), s in at(n1, 30).groupby(["alpha", "gamma"])},
             "overall": round(float(at(n1, 30)["PI"].mean()), 4)}
b, se, t, p, r2, _ = ols(z(at(n1, 30)["PI"]), [z(at(n1, 30)["alpha"])])
R["n100"]["dose"] = {"beta_std": round(float(b[1]), 4), "t": round(float(t[1]), 3), "p": round(float(p[1]), 4),
                     "R2": round(float(r2), 4), "n": len(at(n1, 30))}

# ---------- 5. Qwen ----------
qw = load(NEW, "llm_qwen_full")
R["qwen"] = {"cells": {f"{a}|{g}": {"PI": round(float(s["PI"].mean()), 4), "n": len(s)} for (a, g), s in at(qw, 30).groupby(["alpha", "gamma"])},
             "overall_PI30": round(float(at(qw, 30)["PI"].mean()), 4),
             "PI1": round(float(at(qw, 1)["PI"].mean()), 4),
             "dPI": round(float(at(qw, 30)["PI"].mean() - at(qw, 1)["PI"].mean()), 4),
             "EM30": round(float(at(qw, 30)["EM"].mean()), 4), "SR30": round(float(at(qw, 30)["SR"].mean()), 4),
             "ER30": round(float(at(qw, 30)["ER"].mean()), 4), "HE30": round(float(at(qw, 30)["HE"].mean()), 4)}
dd = {}
for g in sorted(at(qw, 30)["gamma"].unique()):
    sub = at(qw, 30); sub = sub[sub["gamma"] == g]
    b, se, t, p, r2, _ = ols(z(sub["PI"]), [z(sub["alpha"])])
    dd[f"g{g}"] = {"beta_std": round(float(b[1]), 4), "t": round(float(t[1]), 3), "p": round(float(p[1]), 4),
                   "R2": round(float(r2), 4), "n": len(sub)}
R["qwen"]["dose"] = dd
# Qwen 消融
R["qwen_abl"] = {}
for nm in ["llm_qwen_abl_no_post", "llm_qwen_abl_no_emotion", "llm_qwen_abl_positive_control"]:
    arm = load(NEW, nm)
    if arm is None:
        continue
    bseed = at(qw, 30)[["seed", "PI"]].set_index("seed")["PI"] - at(qw, 1)[["seed", "PI"]].set_index("seed")["PI"]
    aseed = at(arm, 30)[["seed", "PI"]].set_index("seed")["PI"] - at(arm, 1)[["seed", "PI"]].set_index("seed")["PI"]
    bseed = bseed[~bseed.index.duplicated()]
    j = pd.concat([bseed.rename("base"), aseed.rename("arm")], axis=1).dropna()
    d = j["arm"] - j["base"]; se = d.std(ddof=1) / np.sqrt(len(d))
    R["qwen_abl"][nm] = {"base_dPI": round(float(j["base"].mean()), 4), "arm_dPI": round(float(j["arm"].mean()), 4),
                         "diff": round(float(d.mean()), 4),
                         "ci95": [round(float(d.mean() - 1.96 * se), 4), round(float(d.mean() + 1.96 * se), 4)],
                         "t": round(float(d.mean() / se), 2), "n": len(j)}

# ---------- 6. 中介（t=30, boot 5000） ----------
def mediation(df, s, label):
    D = at(df, s)
    az, gz = z(D["alpha"]), z(D["gamma"])
    out = {}
    rng = np.random.default_rng(11); n = len(D)
    for M in ["HE", "EE", "EM"]:
        mz, pz = z(D[M]), z(D["PI"])
        a, ase, at_, ap, ar2, _ = ols(mz, [az, gz])
        b, bse, bt, bp, br2, _ = ols(pz, [az, gz, mz])
        ind = a[1] * b[3]
        bs = []
        for _ in range(5000):
            i = rng.integers(0, n, n)
            aa, _, _, _, _, _ = ols(mz[i], [az[i], gz[i]])
            bb, _, _, _, _, _ = ols(pz[i], [az[i], gz[i], mz[i]])
            bs.append(aa[1] * bb[3])
        bs = np.array(bs)
        out[M] = {"a": round(float(a[1]), 4), "a_t": round(float(at_[1]), 2), "a_p": round(float(ap[1]), 5),
                  "b": round(float(b[3]), 4), "b_t": round(float(bt[3]), 2), "b_p": round(float(bp[3]), 5),
                  "indirect": round(float(ind), 4),
                  "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    c, cse, ct, cp, cr2, _ = ols(z(D["PI"]), [az, gz])
    out["total_c"] = {"c": round(float(c[1]), 4), "t": round(float(ct[1]), 2), "p": round(float(cp[1]), 6)}
    return out


R["mediation_main"] = mediation(ext, 30, "DS t30")
R["mediation_qwen"] = mediation(qw, 30, "Qwen t30")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("written", OUT)
print(json.dumps({k: R[k] for k in ["upswing", "randomfeed", "ablation", "n100"] if k in R} | {"main_upswing": R["main"]["upswing"]}, ensure_ascii=False, indent=2, default=str))
