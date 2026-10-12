# -*- coding: utf-8 -*-
"""论文全部表格的数字复核：用新数据重算 表3/表4/表5/表6/表7 + 附录A1/A2/A4，与论文值逐项对比。"""
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
REPLAY = os.path.join(BASE, "results_replay_20261010")
PROBE = os.path.join(PROJ, "probe_提示词明确性", "results_prompt_direction")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")

OUT = []
def rec(item, paper, calc, tol=0.0101, fmt="{:.4f}", note=""):
    ok = "✔" if (calc is not None and abs(calc - paper) <= tol) else "✘"
    OUT.append((item, paper, calc, ok, note))


def z(x):
    x = np.asarray(x, float); s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; n, k = X.shape; dof = max(n - k, 1)
    s2 = (r @ r) / dof; se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se; p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - (r @ r) / (((y - y.mean()) @ (y - y.mean())) or 1)
    return b, se, t, p, r2, dof


def load_llm(root, keyfn=None, pat="**/runs/llm_robust_run_*.csv"):
    seen, rows = set(), []
    for f in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
        m = RX.match(os.path.basename(f))
        if not m: continue
        k = keyfn(f, m) if keyfn else int(m.group(1))
        if k in seen: continue
        seen.add(k)
        d = pd.read_csv(f)
        if len(d): d["run_id"] = k; rows.append(d)
    return pd.concat(rows, ignore_index=True)


def last(df):
    return df.sort_values("step").groupby(["alpha", "gamma", "run_id"]).tail(1)


MAIN = last(load_llm(PROBE, lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{m.group(1)}",
                     "a*/runs/llm_robust_run_*.csv"))
EXT = load_llm(os.path.join(NEW, "llm_extend_T100"))
RM = pd.read_csv(os.path.join(NEW, "_VPI_FIX_20261010", "rule_main_d4", "rule_summary.csv"))
RS = pd.read_csv(os.path.join(REPLAY, "rule_small_d4", "rule_small_summary.csv"))
SP = pd.read_csv(os.path.join(REPLAY, "rule_spiral", "rule_spiral_summary.csv"))
ST = pd.read_csv(os.path.join(REPLAY, "rule_structure", "_variant_dose_response.csv"))


def bstd(df, ycol="PI"):
    return ols(z(df[ycol]), [z(df["alpha"])])[0][1]


def d_cohen(df, ycol="PI"):
    a0 = df[df.alpha == 0][ycol]; a2 = df[df.alpha == 2.0][ycol]
    sp = np.sqrt(((len(a0) - 1) * a0.var(ddof=1) + (len(a2) - 1) * a2.var(ddof=1)) / (len(a0) + len(a2) - 2))
    return (a2.mean() - a0.mean()) / sp


# ============ 表3 条件均值（γ=1.0，末步） ============
T3 = [("Rule 主实验", 0.0, RM, 0.307, 0.953, 0.337, 0.284),
      ("Rule 主实验", 1.0, RM, 0.284, 0.923, 0.346, 0.281),
      ("Rule 主实验", 2.0, RM, 0.269, 0.905, 0.364, 0.287),
      ("Rule 受限规模", 0.0, RS, 0.248, 0.873, 0.333, 0.267),
      ("Rule 受限规模", 2.0, RS, 0.167, 0.816, 0.373, 0.187),
      ("LLM 主实验", 0.0, MAIN, 0.309, 0.761, 0.450, 0.293),
      ("LLM 主实验", 0.5, MAIN, 0.315, 0.785, 0.443, 0.307),
      ("LLM 主实验", 1.0, MAIN, 0.264, 0.735, 0.427, 0.237),
      ("LLM 主实验", 2.0, MAIN, 0.250, 0.753, 0.350, 0.230)]
for name, a, df, pPI, pEM, pSR, pER in T3:
    s = df[(df.alpha == a) & (df.gamma.astype(str) == "1.0")]
    if not len(s): continue
    rec(f"表3 {name} α={a} PI", pPI, s.PI.mean(), 0.0011, "{:.3f}")
    rec(f"表3 {name} α={a} EM", pEM, s.EM.mean(), 0.0011, "{:.3f}")
    rec(f"表3 {name} α={a} SR", pSR, s.SR.mean(), 0.0011, "{:.3f}")
    rec(f"表3 {name} α={a} ER", pER, s.ER.mean(), 0.0011, "{:.3f}")

# ============ 表4 效应量 ============
T4 = [("Rule 主实验", RM, 1.0, -0.566, 0.320, -2.07),
      ("Rule 主实验", RM, 0.1, -0.179, 0.032, -0.49),
      ("Rule 受限规模", RS, 1.0, -0.610, 0.372, -1.82),
      ("Rule 受限规模", RS, 0.1, -0.585, 0.342, -1.76),
      ("LLM 主实验", MAIN, 1.0, -0.405, 0.164, -1.26),
      ("LLM 主实验", MAIN, 0.1, -0.352, 0.124, -0.98)]
for name, df, g, pb, pr2, pd_ in T4:
    s = df[df.gamma.astype(str) == str(g)]
    if not len(s): continue
    b = bstd(s); r2 = ols(z(s.PI), [z(s.alpha)])[4]; dd = d_cohen(s)
    rec(f"表4 {name} γ={g} β_std", pb, b, 0.0020)
    rec(f"表4 {name} γ={g} R²", pr2, r2, 0.0020)
    rec(f"表4 {name} γ={g} Cohen's d", pd_, dd, 0.0100)
# 合并 γ 与延长档
bp = ols(z(MAIN.PI), [z(MAIN.alpha), z(MAIN.gamma.astype(float))])[0]
rec("表4 LLM主实验 合并γ β_std", -0.361, bp[1], 0.0020)
E100 = EXT[EXT.step == 100]
bpe = ols(z(E100.PI), [z(E100.alpha), z(E100.gamma.astype(float))])[0]
rec("表4 LLM延长档 合并γ β_std", -0.053, bpe[1], 0.0020)
for g, pb in (("1.0", -0.087), ("0.1", -0.014)):
    s = E100[E100.gamma.astype(str) == g]
    rec(f"表4 LLM延长档 γ={g} β_std", pb, bstd(s), 0.0020)

# ============ 表5 多项式拟合 ============
T5 = [("Rule 主实验", RM, 1.0, 0.320, 0.321, 0.487, -0.575),
      ("Rule 主实验", RM, 0.1, 0.032, 0.047, 0.054, -0.171),
      ("Rule 受限规模", RS, 1.0, 0.372, 0.399, 0.330, -0.667),
      ("Rule 受限规模", RS, 0.1, 0.342, 0.345, 0.777, -0.620),
      ("LLM 主实验", MAIN, 1.0, 0.164, 0.165, 0.842, -0.402),
      ("LLM 主实验", MAIN, 0.1, 0.124, 0.126, 0.654, -0.337)]
for name, df, g, pr1, pr2, pp, prho in T5:
    s = df[df.gamma.astype(str) == str(g)]
    if not len(s) or len(s) < 6: continue
    x = s.alpha.values; y = s.PI.values
    r1 = ols(y, [x])[4]
    X2 = np.column_stack([np.ones(len(x)), x, x ** 2])
    b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    r2_2 = 1 - ((y - X2 @ b2) @ (y - X2 @ b2)) / ((y - y.mean()) @ (y - y.mean()))
    n, k = X2.shape
    s2 = ((y - X2 @ b2) @ (y - X2 @ b2)) / (n - k)
    se2 = np.sqrt(np.diag(s2 * np.linalg.pinv(X2.T @ X2)))
    p2 = 2 * (1 - stats.t.cdf(abs(b2[2] / se2[2]), n - k))
    rho = stats.spearmanr(x, y).statistic
    rec(f"表5 {name} γ={g} 1次R²", pr1, r1, 0.0020)
    rec(f"表5 {name} γ={g} 2次R²", pr2, r2_2, 0.0020)
    rec(f"表5 {name} γ={g} 二次项p", pp, p2, 0.0060)
    rec(f"表5 {name} γ={g} Spearman", prho, rho, 0.0020)


# ============ 表5 附带：EM 的 U 型 ============
def quad_em(df, g):
    s = df[df.gamma.astype(str) == str(g)]
    x = s.alpha.values; y = s.EM.values
    X2 = np.column_stack([np.ones(len(x)), x, x ** 2])
    b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    r = y - X2 @ b2; n, k = X2.shape
    se2 = np.sqrt(np.diag(((r @ r) / (n - k)) * np.linalg.pinv(X2.T @ X2)))
    p2 = 2 * (1 - stats.t.cdf(abs(b2[2] / se2[2]), n - k))
    return p2, -b2[1] / (2 * b2[2])

for name, df, g, pp, vtx in [("Rule 主实验", RM, 0.1, 0.00002, 1.79),
                             ("Rule 主实验", RM, 1.0, 0.0028, 2.96),
                             ("Rule 受限规模", RS, 0.1, 0.0084, 1.31)]:
    p2, v = quad_em(df, g)
    rec(f"EM 二次项p {name} γ={g}", pp, p2, max(0.0005, pp * 0.3), "{:.5f}")
    rec(f"EM 顶点α {name} γ={g}", vtx, v, 0.02)
for g, pp in ((1.0, 0.76), (0.1, 0.08)):
    p2, v = quad_em(MAIN, g)
    rec(f"LLM EM 二次项p γ={g}", pp, p2, 0.006)

# ============ 表6 中介（点估计） ============
def med(df):
    y = z(df.PI); x = z(df.alpha); g = z(df.gamma.astype(float))
    c = ols(y, [x, g])[0][1]
    m = z(df.HE); a = ols(m, [x, g])[0][1]
    bb = ols(y, [m, x, g])[0]; b = bb[1]; cp = bb[2]
    return c, a, b, a * b, cp


c, a, b, ind, cp = med(MAIN)
rec("表6 LLM α→HE→PI 间接", -0.477, ind, 0.0030)
rec("表6 LLM a 路径", -0.757, a, 0.0030)
rec("表6 LLM b 路径", 0.630, b, 0.0030)
rec("表6 LLM 总效应 c", -0.361, c, 0.0030)
me = z(MAIN.EM); aem = ols(me, [z(MAIN.alpha), z(MAIN.gamma.astype(float))])[0][1]
rec("表6 LLM α→EM 的 a", -0.171, aem, 0.0030)
ey = z(MAIN.EE); aee = ols(ey, [z(MAIN.alpha), z(MAIN.gamma.astype(float))])[0][1]
rec("表6 LLM α→EE 的 a", -0.304, aee, 0.0030)
bEE = ols(z(MAIN.PI), [ey, z(MAIN.alpha), z(MAIN.gamma.astype(float))])[0]
rec("表6 LLM EE b 路径", -0.109, bEE[1], 0.0030)
bEM = ols(z(MAIN.PI), [me, z(MAIN.alpha), z(MAIN.gamma.astype(float))])[0]
rec("表6 LLM EM 间接", -0.012, aem * bEM[1], 0.0030)
# Qwen（必须取末步：与 extract_final.py 的 at(df,30) 同口径）
QW = last(load_llm(os.path.join(NEW, "llm_qwen_full")))
cq, aq, bq, iq, cpq = med(QW)
rec("表6 Qwen α→HE→PI 间接", -0.250, iq, 0.0030)
rec("表6 Qwen a 路径", -0.827, aq, 0.0030)
rec("表6 Qwen b 路径", 0.302, bq, 0.0030)
rec("表6 Qwen 总效应", 0.013, cq, 0.0030)
# Rule 主实验（汇总与分 γ）
cR, aR, bR, indR, cpR = med(RM)
rec("表6 Rule 汇总 α→HE→PI", -0.063, indR, 0.0030)
eR = z(RM.EE); aeeR = ols(eR, [z(RM.alpha), z(RM.gamma.astype(float))])[0][1]
bbR = ols(z(RM.PI), [eR, z(RM.alpha), z(RM.gamma.astype(float))])[0]
rec("表6 Rule 汇总 α→EE→PI", 0.147, aeeR * bbR[1], 0.0030)
rec("表6 Rule a(EE) 路径", -0.399, aeeR, 0.0030)
rec("表6 Rule b(EE) 路径", -0.368, bbR[1], 0.0030)

# ============ 附录 A2 结构变异 β ============
if len(ST):
    print("  [附录A2 变异表列]", list(ST.columns)[:14])
    cols = {c.lower(): c for c in ST.columns}
    vcol = cols.get("variant") or ST.columns[0]
    gcol = cols.get("gamma") or cols.get("γ")
    bcol = cols.get("beta_std") or cols.get("β_std")
    for v in ["base", "comp10", "comp40", "H000", "H020", "H050", "K20", "N500", "T400"]:
        s = ST[(ST[vcol] == v) & (ST[gcol].astype(str) == "1.0")]
        if len(s) and bcol:
            rec(f"附录A2 {v} γ=1.0 β", float("nan"), float(s[bcol].iloc[0]), 0.0020, "{:.3f}", "见下")

# ============ 附录 A4 延长档 ============
t30 = EXT[EXT.step == 30]
rec("附录A4 PI t=30", 0.311, t30.PI.mean(), 0.0011, "{:.3f}")
rec("附录A4 PI t=100", 0.417, E100.PI.mean(), 0.0011, "{:.3f}")
rec("附录A4 HE t=30", 0.946, t30.HE.mean(), 0.0011, "{:.3f}")
rec("附录A4 HE t=100", 0.975, E100.HE.mean(), 0.0011, "{:.3f}")
rec("附录A4 EM t=100", 0.86, E100.EM.mean(), 0.011, "{:.2f}")
rec("附录A4 SR t=100", 0.47, E100.SR.mean(), 0.011, "{:.2f}")
rec("附录A4 ER t=100", 0.41, E100.ER.mean(), 0.011, "{:.2f}")
if "boundary_frac" in EXT.columns:
    rec("附录A4 边界吸附 t=100", 0.210, E100.boundary_frac.mean(), 0.0020)

# ============ 表7 螺旋机制输入（重算数据源：重放汇总 + 确定性重跑命中率） ============
SPF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spiral_minority_rerun.csv")
if os.path.exists(SPF):
    Q = pd.read_csv(SPF)
    for a, phe, pdm, pdx, php in [(0.0, 0.9998, 0.00018, 0.0006, 0.0002),
                                  (0.5, 0.9988, 0.00121, 0.0037, 0.0006),
                                  (1.0, 0.9982, 0.00180, 0.0038, 0.0019),
                                  (1.5, 0.9977, 0.00235, 0.0060, 0.0022),
                                  (2.0, 0.9974, 0.00261, 0.0077, 0.0018)]:
        s = SP[np.isclose(SP.alpha, a)]
        m = Q[np.isclose(Q.alpha, a)]
        rec(f"表7 α={a} HE", phe, s.HE.mean(), 0.0001)
        rec(f"表7 α={a} d均值", pdm, s.mean_exposure_distance.mean(), 0.000005)
        rec(f"表7 α={a} d最大", pdx, s.mean_exposure_distance.max(), 0.00005)
        rec(f"表7 α={a} 命中率", php, m.hit_post_final.mean(), 0.00006)  # 论文为 4 位小数四舍五入值
        rec(f"表7 α={a} 1-c·d", 1 - 2 * pdx, 1 - 2 * s.mean_exposure_distance.max(), 0.0006, "{:.3f}")
    x = SP.alpha.to_numpy(float); y = SP.mean_exposure_distance.to_numpy(float)
    X = np.column_stack([np.ones(len(x)), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; n = len(x)
    se = np.sqrt(np.diag(((r @ r) / (n - 2)) * np.linalg.pinv(X.T @ X)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    rec("表7 d~α 斜率", 0.0012, b[1], 0.00005, "{:.4f}")
    rec("表7 d~α SE", 0.000033, se[1], 0.000001, "{:.6f}")
    rec("表7 d~α t", 37.7, b[1] / se[1], 0.10, "{:.1f}")
    rec("表7 d~α R²", 0.551, r2, 0.0020, "{:.3f}")
    d0 = SP[np.isclose(SP.alpha, 0)].mean_exposure_distance.mean()
    d2 = SP[np.isclose(SP.alpha, 2)].mean_exposure_distance.mean()
    rec("表7 d(α=2)/d(α=0)", 14.0, d2 / d0, 1.0, "{:.1f}")
    rec("表7 最坏个体削减(≤2%)", 0.0154, 2 * SP.mean_exposure_distance.max(), 0.0010, "{:.4f}")
    # 命中率上界（论文表述为"均低于 0.003"，故按单侧口径给 0.001 容差）
    mn = max(Q.groupby("alpha").hit_post_final.mean())
    rec("表7 命中率上界(<0.003)", 0.003, mn, 0.0010, "{:.4f}")

# ============ 输出 ============
print()
print(f"{'项目':<40}{'论文':>12}{'重算':>12}  判定")
print("-" * 78)
nbad = 0
for it, p, c_, ok, note in OUT:
    if ok == "✘" and note != "见下":
        nbad += 1
    ps = f"{p:.3f}" if isinstance(p, float) else str(p)
    cs = "—" if (c_ is None or (isinstance(c_, float) and not np.isfinite(c_))) else \
         (f"{c_:.4f}" if isinstance(c_, float) else str(c_))
    print(f"{it:<40}{ps:>12}{cs:>12}   {ok} {note}")
print("-" * 78)
print(f"共 {len(OUT)} 项，✘ {nbad} 项")
json.dump([[a, b, (float(c) if isinstance(c, (int, float)) and np.isfinite(c) else None), d, e]
           for a, b, c, d, e in OUT],
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_check.json"),
               "w", encoding="utf-8"), ensure_ascii=False, indent=1)
