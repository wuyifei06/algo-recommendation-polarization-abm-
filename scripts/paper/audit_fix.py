# -*- coding: utf-8 -*-
"""核对：LLM 主实验(T=30) 的正确来源 = probe/results_prompt_direction；
   延长档 = results_direction/llm_extend_T100。逐项重算并对比旧（错误）值。"""
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
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")


def load_probe():
    out = []
    for f in sorted(glob.glob(os.path.join(MAIN_DIR, "a*", "runs", "llm_robust_run_*.csv"))):
        b = os.path.basename(f)
        if not RX.match(b):
            continue
        d = pd.read_csv(f)
        if len(d):
            adir = os.path.basename(os.path.dirname(os.path.dirname(f)))
            d["run_id"] = f"{adir}:{int(RX.match(b).group(1))}"
            out.append(d)
    return pd.concat(out, ignore_index=True)


def load_llm(stage, root=NEW):
    seen, uniq = set(), []
    for f in sorted(glob.glob(os.path.join(root, stage, "**", "runs", "llm_robust_run_*.csv"), recursive=True)):
        b = os.path.basename(f); m = RX.match(b)
        if m and b not in seen:
            seen.add(b); uniq.append((int(m.group(1)), f))
    out = []
    for rid, f in uniq:
        d = pd.read_csv(f)
        if len(d):
            d["run_id"] = rid; out.append(d)
    return pd.concat(out, ignore_index=True)


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


M = load_probe()                     # 主实验 T=30
E = load_llm("llm_extend_T100")      # 延长档 T=100
print("MAIN(probe) rows", len(M), "steps", sorted(M.step.unique()))
print("EXT(extend_T100) rows", len(E), "steps", sorted(E.step.unique()))
print()

# ---- 1. 剂量响应 ----
for tag, D, step in [("MAIN t=30", M, 30), ("EXT t=30", E, 30), ("EXT t=100", E, 100)]:
    D = D[D.step == step]
    for g in [0.1, 1.0]:
        s = D[np.isclose(D.gamma, g)]
        b, se, t, p, r2, _ = ols(z(s.PI), [z(s.alpha)])
        lo, hi = s[s.alpha == s.alpha.min()], s[s.alpha == s.alpha.max()]
        sp = np.sqrt(((len(lo)-1)*lo.PI.var(ddof=1)+(len(hi)-1)*hi.PI.var(ddof=1))/(len(lo)+len(hi)-2))
        print(f"{tag:11s} g={g}: beta_std={b[1]:+.4f} t={t[1]:+.2f} p={p[1]:.5f} R2={r2:.4f} d={(hi.PI.mean()-lo.PI.mean())/sp:+.2f}")
    b, se, t, p, r2, _ = ols(z(D.PI), [z(D.alpha), z(D.gamma)])
    print(f"{tag:11s} pooled: beta_a={b[1]:+.4f} t={t[1]:+.2f} p={p[1]:.6f} R2={r2:.4f} beta_g={b[2]:+.4f} t_g={t[2]:+.2f}")
print()

# ---- 2. 逐格均值（γ=1.0）----
print("MAIN t=30 γ=1.0 cells:", {round(a,2): round(v,4) for a, v in at(M,30)[np.isclose(at(M,30).gamma,1.0)].groupby('alpha').PI.mean().items()})
print("MAIN t=30 γ=0.1 cells:", {round(a,2): round(v,4) for a, v in at(M,30)[np.isclose(at(M,30).gamma,0.1)].groupby('alpha').PI.mean().items()})
print()

# ---- 3. 上行 ΔPI ----
for tag, D, s1 in [("MAIN", M, 30)]:
    for g in [0.1, 1.0]:
        d = D[np.isclose(D.gamma, g)]
        p1 = d[d.step == 1].set_index("run_id").PI
        ps = d[d.step == s1].set_index("run_id").PI
        diff = (ps - p1).dropna()
        print(f"{tag} g={g} ΔPI(t1→t{s1})={diff.mean():+.4f} sd={diff.std(ddof=1):.4f} t={diff.mean()/(diff.std(ddof=1)/np.sqrt(len(diff))):.2f} up={int((diff>0).sum())}/{len(diff)}")
print()

# ---- 4. 中介（MAIN, t=30, boot 5000）----
def mediation(D, step):
    D = D[D.step == step]
    az, gz = z(D.alpha), z(D.gamma); out = {}
    rng = np.random.default_rng(11); n = len(D)
    for Mv in ["HE", "EE", "EM"]:
        mz, pz = z(D[Mv]), z(D.PI)
        a = ols(mz, [az, gz])[0][1]; bb = ols(pz, [az, gz, mz])[0][3]
        bs = []
        for _ in range(5000):
            i = rng.integers(0, n, n)
            aa = ols(mz[i], [az[i], gz[i]])[0][1]; bbb = ols(pz[i], [az[i], gz[i], mz[i]])[0][3]
            bs.append(aa * bbb)
        bs = np.array(bs)
        out[Mv] = {"a": round(a, 4), "b": round(bb, 4), "ind": round(a*bb, 4),
                   "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    c = ols(z(D.PI), [az, gz])[0][1]
    out["c_total"] = round(float(c), 4)
    return out

print("MAIN mediation:", json.dumps(mediation(M, 30), ensure_ascii=False))
print()

# ---- 5. 随机 feed 配对（基线=MAIN）----
rf = load_llm("llm_randomfeed_control")
fr = at(rf, 30)
al = at(M, 30); al = al[np.isclose(al.alpha, 1.0)]
mm = pd.merge(fr, al, on=["gamma", "seed"], suffixes=("_rf", "_al"))
print("randomfeed pairing n=", len(mm), "(seeds rf:", sorted(fr.seed.unique())[:3], "..., main a1.0 seeds:", sorted(al.seed.unique())[:3], ")")
for k in ["PI", "HE", "ER", "EM", "SR"]:
    d = (mm[f"{k}_rf"] - mm[f"{k}_al"]).to_numpy(float)
    se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"  {k}: rf={mm[f'{k}_rf'].mean():.4f} algo={mm[f'{k}_al'].mean():.4f} diff={d.mean():+.4f} ci=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}] t={d.mean()/se:+.2f}")
print()

# ---- 6. 在线消融（基线=MAIN）----
def dpi(df, s):
    a = at(df, s).set_index("run_id").PI; b = at(df, 1).set_index("run_id").PI
    return a - b
base_main = at(M, 30).set_index("seed").PI - at(M, 1).set_index("seed").PI
print("MAIN baseline ΔPI (by seed) mean=", round(base_main.mean(), 4), "n=", len(base_main))
for nm in ["llm_ablation_no_post", "llm_ablation_no_emotion", "llm_ablation_positive_control"]:
    arm = load_llm(nm)
    aseed = at(arm, 30).set_index("seed").PI - at(arm, 1).set_index("seed").PI
    aseed = aseed[~aseed.index.duplicated()]
    j = pd.concat([base_main.rename("base"), aseed.rename("arm")], axis=1).dropna()
    d = j["arm"] - j["base"]; se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"  {nm}: armΔPI={j['arm'].mean():+.4f} baseΔPI={j['base'].mean():+.4f} diff={d.mean():+.4f} ci=[{d.mean()-1.96*se:+.4f},{d.mean()+1.96*se:+.4f}] t={d.mean()/se:+.2f} n={len(j)} frac_neg={(d<0).mean():.3f}")
