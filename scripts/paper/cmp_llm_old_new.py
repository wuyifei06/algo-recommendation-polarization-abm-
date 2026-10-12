# -*- coding: utf-8 -*-
"""对比 LLM 部分 旧数据(results/) vs 方向版新数据(results_direction/) 的全部关键数字。"""
import os, glob, json
import numpy as np
import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

BASE = _PROJ_ROOT
NEW = os.path.join(BASE, "results_direction")
OLD = os.path.join(BASE, "results")
OUT = _os.path.join(_WORK_ROOT, r"llm_cmp.json")
STAGES = ["llm_extend_T100", "llm_randomfeed_control", "llm_qwen_full", "llm_n100_scale",
          "llm_ablation_no_post", "llm_ablation_no_emotion", "llm_ablation_positive_control"]


def load(root, stage):
    pat = os.path.join(root, stage, "**", "runs", "llm_robust_run_*.csv")
    files = sorted(glob.glob(pat, recursive=True))
    import re as _re
    rx = _re.compile(r"^llm_robust_run_\d+\.csv$")
    # 去重（shard 合并可能重复）
    seen, uniq = set(), []
    for f in files:
        b = os.path.basename(f)
        if not rx.match(b):
            continue
        if b in seen:
            continue
        seen.add(b); uniq.append(f)
    out = []
    for f in uniq:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if not len(d):
            continue
        d["run_id"] = int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0])
        out.append(d)
    if not out:
        return None
    return pd.concat(out, ignore_index=True)


def last_step(df):
    return df.loc[df.groupby("run_id")["step"].idxmax()].copy()


def zscore(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0.0


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / se
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, t, r2, dof


def report(root, tag):
    res = {}
    for st in STAGES:
        df = load(root, st)
        if df is None:
            res[st] = None
            continue
        L = last_step(df)
        info = {"n_runs": int(L["run_id"].nunique()), "T": int(L["step"].max())}
        if "N" in L:
            info["N"] = int(L["N"].iloc[0])
        if "model" in L:
            info["model"] = str(L["model"].iloc[0])
        # 按 alpha x gamma 末步均值
        g = L.groupby(["alpha", "gamma"])
        cell = {}
        for (a, gm), sub in g:
            cell[f"a{a}_g{gm}"] = {k: round(float(sub[k].mean()), 4)
                                   for k in ["PI", "SR", "ER", "EM", "HE", "EE"] if k in sub}
            cell[f"a{a}_g{gm}"]["n"] = int(len(sub))
        info["cells"] = cell
        info["overall_PI"] = round(float(L["PI"].mean()), 4)
        res[st] = info
    return res


def dose_response(root):
    df = load(root, "llm_extend_T100")
    if df is None:
        return None
    L = last_step(df)
    out = {}
    for gm in sorted(L["gamma"].unique()):
        sub = L[L["gamma"] == gm]
        y = zscore(sub["PI"])
        b, se, t, r2, dof = ols(y, [zscore(sub["alpha"])])
        out[f"gamma={gm}"] = {"beta_std": round(float(b[1]), 4), "se": round(float(se[1]), 4),
                              "t": round(float(t[1]), 3), "R2": round(float(r2), 4), "n": int(len(sub))}
    # 合并（alpha 标准化，gamma 作协变量）
    y = zscore(L["PI"])
    b, se, t, r2, dof = ols(y, [zscore(L["alpha"])] + [zscore(L["gamma"])])
    out["pooled"] = {"beta_alpha_std": round(float(b[1]), 4), "se": round(float(se[1]), 4),
                     "t": round(float(t[1]), 3), "R2": round(float(r2), 4),
                     "beta_gamma_std": round(float(b[2]), 4), "n": int(len(L))}
    # 原始尺度 slope
    for gm in sorted(L["gamma"].unique()):
        sub = L[L["gamma"] == gm]
        b, se, t, r2, dof = ols(sub["PI"].to_numpy(float), [sub["alpha"].to_numpy(float)])
        lo, hi = b[1] - 1.96 * se[1], b[1] + 1.96 * se[1]
        out[f"raw_gamma={gm}"] = {"slope": round(float(b[1]), 5), "se": round(float(se[1]), 5),
                                  "ci95": [round(float(lo), 5), round(float(hi), 5)]}
    # t=30 vs t=100
    tt = {}
    for a in sorted(L["alpha"].unique()):
        sub = L[L["alpha"] == a]
        if (df["step"] == 30).any():
            d30 = df[(df["alpha"] == a) & (df["step"] == 30)]
            tt[f"a{a}"] = {"t30_PI": round(float(d30["PI"].mean()), 4),
                           "t100_PI": round(float(sub["PI"].mean()), 4)}
    out["timeline"] = tt
    return out


def randomfeed_vs_algo(root):
    rf = load(root, "llm_randomfeed_control")
    ext = load(root, "llm_extend_T100")
    if rf is None or ext is None:
        return None
    R = last_step(rf)
    E = last_step(ext)
    # 按 (gamma, seed) 配对 alpha=1.0 的 run
    algo = E[E["alpha"] == 1.0][["gamma", "seed", "PI", "HE", "EM", "SR", "ER"]]
    m = pd.merge(R[["gamma", "seed", "PI", "HE", "EM", "SR", "ER"]], algo,
                 on=["gamma", "seed"], suffixes=("_rf", "_algo"))
    if not len(m):
        m = pd.merge(R, algo, on="gamma", suffixes=("_rf", "_algo"))
    out = {"n_pairs": int(len(m))}
    for k in ["PI", "HE", "EM", "SR", "ER"]:
        d = (m[f"{k}_rf"] - m[f"{k}_algo"]).to_numpy(float)
        out[k] = {"rf": round(float(m[f"{k}_rf"].mean()), 4),
                  "algo": round(float(m[f"{k}_algo"].mean()), 4),
                  "diff": round(float(d.mean()), 4),
                  "ci95": [round(float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))), 4),
                           round(float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))), 4)]}
    return out


def main():
    out = {}
    out["NEW_direction"] = report(NEW, "new")
    out["OLD_results"] = report(OLD, "old")
    out["NEW_dose"] = dose_response(NEW)
    out["OLD_dose"] = dose_response(OLD)
    out["NEW_randomfeed_vs_algo"] = randomfeed_vs_algo(NEW)
    out["OLD_randomfeed_vs_algo"] = randomfeed_vs_algo(OLD)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 打印精要
    print("=" * 70)
    print("末步 PI 总体均值（每 run 末步）")
    print(f"{'stage':<32}{'OLD results/':>14}{'NEW direction':>16}")
    for st in STAGES:
        o = out["OLD_results"].get(st)
        n = out["NEW_direction"].get(st)
        ov = o["overall_PI"] if o else None
        nv = n["overall_PI"] if n else None
        print(f"{st:<32}{str(ov):>14}{str(nv):>16}")
    print("=" * 70)
    print("α 剂量响应 (extend_T100)")
    for tag in ["OLD_dose", "NEW_dose"]:
        print(f"--- {tag} ---")
        for k, v in out[tag].items():
            if k == "timeline":
                print("  timeline:", v)
            else:
                print(f"  {k}: {v}")
    print("=" * 70)
    print("随机 feed vs 算法臂")
    for tag in ["OLD_randomfeed_vs_algo", "NEW_randomfeed_vs_algo"]:
        print(f"--- {tag} ---")
        print(" ", out[tag])
    print("=" * 70)
    print("各格末步 PI (NEW):")
    for st in STAGES:
        n = out["NEW_direction"].get(st)
        if not n:
            print(f"  {st}: <missing>"); continue
        cells = n["cells"]
        s = "  ".join(f"{k}:PI={v['PI']}(n={v['n']})" for k, v in sorted(cells.items()))
        print(f"  {st}: {s}")


if __name__ == "__main__":
    main()
