# -*- coding: utf-8 -*-
"""按论文报告口径，对 LLM 部分逐项重算 旧(results/) vs 新(results_direction/)。
输出 llm_diff.json + 控制台差异清单。"""
import os, glob, re, json
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
OUT = _os.path.join(_WORK_ROOT, r"llm_diff.json")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")


def load(root, stage):
    pat = os.path.join(root, stage, "**", "runs", "llm_robust_run_*.csv")
    seen, uniq = set(), []
    for f in sorted(glob.glob(pat, recursive=True)):
        b = os.path.basename(f)
        m = RX.match(b)
        if not m or b in seen:
            continue
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


def at_step(df, s):
    return df[df["step"] == s].copy()


def last(df):
    return df.loc[df.groupby("run_id")["step"].idxmax()].copy()


def z(x):
    x = np.asarray(x, float); s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; n, k = X.shape; dof = max(n - k, 1)
    s2 = (r @ r) / dof; se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, b / se, r2, dof


def dose(df, step):
    D = at_step(df, step)
    out = {}
    for gm in sorted(D["gamma"].unique()):
        s = D[D["gamma"] == gm]
        b, se, t, r2, dof = ols(z(s["PI"]), [z(s["alpha"])])
        lo, hi = b[1] - 1.96 * se[1], b[1] + 1.96 * se[1]
        out[f"g{gm}"] = {"beta_std": round(b[1], 4), "se_std": round(se[1], 4),
                         "t": round(t[1], 3), "R2": round(r2, 4), "n": len(s),
                         "beta_std_ci95": [round(lo, 4), round(hi, 4)]}
        # 原始尺度
        b2, se2, t2, r22, _ = ols(s["PI"].to_numpy(float), [s["alpha"].to_numpy(float)])
        out[f"g{gm}"]["slope_raw"] = round(b2[1], 5)
        out[f"g{gm}"]["slope_raw_ci95"] = [round(b2[1] - 1.96 * se2[1], 5),
                                           round(b2[1] + 1.96 * se2[1], 5)]
    b, se, t, r2, dof = ols(z(D["PI"]), [z(D["alpha"]), z(D["gamma"])])
    out["pooled"] = {"beta_alpha_std": round(b[1], 4), "se": round(se[1], 4),
                     "t": round(t[1], 3), "p_approx": round(2 * (1 - 0.5 * (1 + np.math.erf(abs(t[1]) / np.sqrt(2)))), 4) if False else None,
                     "R2": round(r2, 4), "n": len(D),
                     "beta_gamma_std": round(b[2], 4)}
    return out


def paired(a, b, cols):
    m = pd.merge(a, b, on=["gamma", "seed"], suffixes=("_1", "_2"))
    out = {"n": len(m)}
    for k in cols:
        d = (m[f"{k}_1"] - m[f"{k}_2"]).to_numpy(float)
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[k] = {"arm1": round(float(m[f"{k}_1"].mean()), 4),
                  "arm2": round(float(m[f"{k}_2"].mean()), 4),
                  "diff": round(float(d.mean()), 4),
                  "ci95": [round(float(d.mean() - 1.96 * se), 4), round(float(d.mean() + 1.96 * se), 4)]}
    return out


def mediation(df, step):
    D = at_step(df, step)
    az, gz = z(D["alpha"]), z(D["gamma"])
    res = {}
    for M in ["HE", "EE", "EM"]:
        mz, pz = z(D[M]), z(D["PI"])
        a, ase, at, ar2, _ = ols(mz, [az, gz])
        b, bse, bt, br2, _ = ols(pz, [az, gz, mz])
        ind = a[1] * b[3]
        # bootstrap
        rng = np.random.default_rng(7); bs = []
        n = len(D)
        for _ in range(2000):
            i = rng.integers(0, n, n)
            aa, _, _, _, _ = ols(mz[i], [az[i], gz[i]])
            bb, _, _, _, _ = ols(pz[i], [az[i], gz[i], mz[i]])
            bs.append(aa[1] * bb[3])
        bs = np.array(bs)
        res[M] = {"a": round(a[1], 4), "a_t": round(at[1], 3), "b": round(b[3], 4), "b_t": round(bt[3], 3),
                  "indirect": round(ind, 4),
                  "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(np.percentile(bs, 97.5), 4)]}
    # 总效应 c, 直接 c'
    pz = z(D["PI"])
    c, cse, ct, _, _ = ols(pz, [az, gz])
    res["total_c"] = {"c": round(c[1], 4), "t": round(ct[1], 3)}
    return res


def ablate(root, step):
    base = load(root, "llm_extend_T100")
    res = {}
    for nm in ["llm_ablation_no_post", "llm_ablation_no_emotion", "llm_ablation_positive_control"]:
        arm = load(root, nm)
        if arm is None:
            continue
        def dpi(df):
            return (at_step(df, step).set_index("run_id")["PI"] -
                    at_step(df, 1).set_index("run_id")["PI"])
        b = dpi(base); a = dpi(arm)
        common = b.index.intersection(a.index)
        # 配对按 seed
        bs = at_step(base, step)[["seed", "PI"]].set_index("seed")["PI"] - at_step(base, 1)[["seed", "PI"]].set_index("seed")["PI"]
        asr = at_step(arm, step)[["seed", "PI"]].set_index("seed")["PI"] - at_step(arm, 1)[["seed", "PI"]].set_index("seed")["PI"]
        j = pd.concat([bs.rename("base"), asr.rename("arm")], axis=1).dropna()
        d = j["arm"] - j["base"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        res[nm] = {"base_dPI": round(float(j["base"].mean()), 4),
                   "arm_dPI": round(float(j["arm"].mean()), 4),
                   "paired_diff": round(float(d.mean()), 4),
                   "ci95": [round(float(d.mean() - 1.96 * se), 4), round(float(d.mean() + 1.96 * se), 4)],
                   "retain_pct": round(float(j["arm"].mean() / j["base"].mean() * 100), 1), "n": len(j)}
    return res


def cells(df, step, keys=("PI", "SR", "ER", "EM", "HE", "EE")):
    D = at_step(df, step)
    out = {}
    for (a, gm), sub in D.groupby(["alpha", "gamma"]):
        out[f"a{a}_g{gm}"] = {k: round(float(sub[k].mean()), 4) for k in keys}
    return out


def timeline(df, keys=("PI", "EM", "SR", "ER")):
    out = {}
    for s, sub in df.groupby("step"):
        out[int(s)] = {k: round(float(sub[k].mean()), 4) for k in keys}
    return out


def main():
    R = {"new": {}, "old": {}}
    for tag, root in [("new", NEW), ("old", OLD)]:
        ext = load(root, "llm_extend_T100")
        R[tag]["extend_T100"] = {
            "cells_t30": cells(ext, 30), "cells_t100": cells(ext, 100),
            "overall_t30_PI": round(float(at_step(ext, 30)["PI"].mean()), 4),
            "overall_t100_PI": round(float(last(ext)["PI"].mean()), 4),
            "dose_t30": dose(ext, 30), "dose_t100": dose(ext, 100),
            "timeline": timeline(ext),
            "dPI_t1_t30": round(float((at_step(ext, 30)["PI"].mean() - at_step(ext, 1)["PI"].mean())), 4),
            "dPI_t1_t100": round(float((at_step(ext, 100)["PI"].mean() - at_step(ext, 1)["PI"].mean())), 4),
        }
        rf = load(root, "llm_randomfeed_control")
        algo30 = at_step(ext, 30); algo30 = algo30[algo30["alpha"] == 1.0]
        R[tag]["randomfeed_vs_algo_t30"] = paired(last(rf), algo30, ["PI", "HE", "EM", "SR", "ER"])
        qw = load(root, "llm_qwen_full")
        R[tag]["qwen"] = {"cells_t30": cells(qw, 30),
                          "overall_t30_PI": round(float(at_step(qw, 30)["PI"].mean()), 4),
                          "dose_t30": dose(qw, 30),
                          "dPI_t1_t30": round(float(at_step(qw, 30)["PI"].mean() - at_step(qw, 1)["PI"].mean()), 4),
                          "delta_vs_deepseek": round(float(at_step(qw, 30)["PI"].mean() - at_step(ext, 30)["PI"].mean()), 4)}
        n1 = load(root, "llm_n100_scale")
        R[tag]["n100"] = {"cells_t30": cells(n1, 30), "overall_t30_PI": round(float(at_step(n1, 30)["PI"].mean()), 4),
                          "dose_t30": dose(n1, 30)}
        R[tag]["ablation"] = ablate(root, 30)
        R[tag]["mediation_t30"] = mediation(ext, 30)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    return R


if __name__ == "__main__":
    R = main()
    n, o = R["new"], R["old"]
    print("#" * 78)
    print("## 1. LLM 主实验（extend_T100 的 t=30 切片, 8α×2γ×10=160 runs）")
    print(f"{'':<18}{'OLD':>10}{'NEW':>10}")
    print(f"{'PI(t=30)':<18}{o['extend_T100']['overall_t30_PI']:>10}{n['extend_T100']['overall_t30_PI']:>10}")
    print(f"{'PI(t=100)':<18}{o['extend_T100']['overall_t100_PI']:>10}{n['extend_T100']['overall_t100_PI']:>10}")
    print(f"{'ΔPI(1→30)':<18}{o['extend_T100']['dPI_t1_t30']:>10}{n['extend_T100']['dPI_t1_t30']:>10}")
    print(f"{'ΔPI(1→100)':<18}{o['extend_T100']['dPI_t1_t100']:>10}{n['extend_T100']['dPI_t1_t100']:>10}")
    print("\n### 剂量响应 β_std(α→PI)")
    for st in ["dose_t30", "dose_t100"]:
        print(f"-- {st} --")
        for gm in list(o['extend_T100'][st].keys()):
            oo = o['extend_T100'][st][gm]; nn = n['extend_T100'][st][gm]
            print(f"   {gm:<10} OLD={oo} \n              NEW={nn}")
    print("\n### 各格 PI(t=30)")
    print(f"{'cell':<16}{'OLD':>9}{'NEW':>9}")
    keys = sorted(n['extend_T100']['cells_t30'].keys())
    for k in keys:
        print(f"{k:<16}{o['extend_T100']['cells_t30'][k]['PI']:>9}{n['extend_T100']['cells_t30'][k]['PI']:>9}")
    print("\n" + "#" * 78)
    print("## 2. 随机 feed 对照（配对 t=30, α=1.0）")
    for tag, lbl in [("old", "OLD"), ("new", "NEW")]:
        print(f"-- {lbl} --", R[tag]["randomfeed_vs_algo_t30"])
    print("\n" + "#" * 78)
    print("## 3. Qwen 跨模型（t=30）")
    for tag, lbl in [("old", "OLD"), ("new", "NEW")]:
        q = R[tag]["qwen"]
        print(f"-- {lbl} -- PI={q['overall_t30_PI']}  ΔPI(1→30)={q['dPI_t1_t30']}  dose={q['dose_t30']}")
    print("\n" + "#" * 78)
    print("## 4. N=100 规模（t=30, γ=1.0）")
    for tag, lbl in [("old", "OLD"), ("new", "NEW")]:
        print(f"-- {lbl} -- ", R[tag]["n100"]["cells_t30"], " dose=", R[tag]["n100"]["dose_t30"])
    print("\n" + "#" * 78)
    print("## 5. 在线消融（ΔPI 配对）")
    for tag, lbl in [("old", "OLD"), ("new", "NEW")]:
        print(f"-- {lbl} --")
        for k, v in R[tag]["ablation"].items():
            print(f"   {k}: {v}")
    print("\n" + "#" * 78)
    print("## 6. 中介（t=30, γ 协变量, z 标准化, boot2000）")
    for tag, lbl in [("old", "OLD"), ("new", "NEW")]:
        print(f"-- {lbl} --")
        for k, v in R[tag]["mediation_t30"].items():
            print(f"   {k}: {v}")
