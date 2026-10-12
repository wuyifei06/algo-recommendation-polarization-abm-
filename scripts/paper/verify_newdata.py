# -*- coding: utf-8 -*-
"""论文数据来源核查：把论文中的每一项关键数字，分别用「新数据」与「作废旧数据」重算并对比。
   新数据：probe_提示词明确性/results_prompt_direction（LLM 主实验 T=30）
           results_direction/*（LLM 延长档与各复核臂）
           results_replay_20261010/*（规则臂，本轮重放）
   旧数据：results/*（作废周期）
"""
import glob, json, os, re
import numpy as np
import pandas as pd
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
OLD = os.path.join(BASE, "results")
REPLAY = os.path.join(BASE, "results_replay_20261010")
PROBE = os.path.join(PROJ, "probe_提示词明确性", "results_prompt_direction")
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")


# ---------- 载入 ----------
def load(root, keyfn=None, pat="**/runs/llm_robust_run_*.csv"):
    seen, files = set(), []
    for f in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
        m = RX.match(os.path.basename(f))
        if not m:
            continue
        k = keyfn(f, m) if keyfn else int(m.group(1))
        if k in seen:
            continue
        seen.add(k); files.append(f)
    rows = []
    for f, k in zip(files, [keyfn(f, RX.match(os.path.basename(f))) if keyfn else int(RX.match(os.path.basename(f)).group(1)) for f in files]):
        d = pd.read_csv(f)
        if len(d):
            d["run_id"] = k
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def last(df):
    return df.sort_values("step").groupby(["alpha", "gamma", "run_id"]).tail(1)


def z(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    se = np.sqrt(np.diag(((r @ r) / (n - k)) * np.linalg.pinv(X.T @ X)))
    return b, b[1] / se[1]


def beta_std(df, ycol="PI"):
    y = z(df[ycol]); x = z(df["alpha"])
    b, _ = ols(y, [x])
    return b[1]


def beta_std_pooled(df, ycol="PI"):
    y = z(df[ycol]); x = z(df["alpha"]); g = z(df["gamma"].astype(float))
    b, _ = ols(y, [x, g])
    return b[1], b[2]


def mediation(df):
    """以 γ 为协变量、样本内 z 标准化。返回 (c, a, b, ind, cprime)。"""
    y = z(df["PI"]); x = z(df["alpha"]); g = z(df["gamma"].astype(float))
    bc, _ = ols(y, [x, g]); c = bc[1]
    m = z(df["HE"]); ba, _ = ols(m, [x, g]); a = ba[1]
    bb, _ = ols(y, [m, x, g]); b = bb[1]; cp = bb[2]
    return c, a, b, a * b, cp


def cellmean(df, a, g, col="PI"):
    s = df[(df["alpha"] == a) & (df["gamma"].astype(str) == str(g))]
    return s[col].mean()


# ---------- 载入各批 ----------
DATA = {}
DATA["LLM主实验_T30"] = ("new", load(PROBE, lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{m.group(1)}", "a*/runs/llm_robust_run_*.csv"))
DATA["LLM主实验_T30_old"] = ("old", load(os.path.join(OLD, "llm_robust_grid")))
for st in ["llm_extend_T100", "llm_randomfeed_control", "llm_ablation_no_post",
           "llm_ablation_no_emotion", "llm_ablation_positive_control",
           "llm_n100_scale", "llm_qwen_full", "llm_qwen_abl_no_post", "llm_qwen_abl_no_emotion"]:
    p = os.path.join(NEW, st)
    if os.path.isdir(p):
        DATA[st] = ("new", load(p))
    p = os.path.join(OLD, st)
    if os.path.isdir(p):
        DATA[st + "_old"] = ("old", load(p))

print("=" * 96)
print("已载入批次：")
for k, (src, d) in DATA.items():
    if len(d):
        print(f"  [{src.upper():>3}] {k:<32} runs={d['run_id'].nunique():>4}  "
              f"α={sorted(d['alpha'].unique())}  γ={sorted(d['gamma'].astype(str).unique())}")
    else:
        print(f"  [{src.upper():>3}] {k:<32} (空)")
print("=" * 96)

# ---------- 逐项对比 ----------
rows = []


def rec(item, paper, newv, oldv, fmt="{:.4f}", expect_old=True):
    """expect_old=False 表示该行是「旧数据对照」，本就不应与论文一致。"""
    def f(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return fmt.format(v) if isinstance(v, float) else str(v)
    ok_new = "✔" if (newv is not None and abs(newv - paper) <= 0.0101) else "✘"
    if not expect_old:
        ok_new = "◆对照(预期不符)"
    rows.append((item, f(paper), f(newv), f(oldv), ok_new))


# 1) LLM 主实验 T=30
d = DATA["LLM主实验_T30"][1]; do = DATA["LLM主实验_T30_old"][1]
L = last(d); Lo = last(do) if len(do) else pd.DataFrame()
for g, pv in ((1.0, -0.405), (0.1, -0.352)):
    rec(f"LLM主实验 β_std (γ={g})", pv, beta_std(L[L.gamma.astype(str) == str(g)]),
        beta_std(Lo[Lo.gamma.astype(str) == str(g)]) if len(Lo) else None)
rec("LLM主实验 β_std (合并 γ)", -0.361, beta_std_pooled(L)[0], beta_std_pooled(Lo)[0] if len(Lo) else None)
for a, pv in ((0.0, 0.309), (1.0, 0.264), (2.0, 0.250)):
    rec(f"LLM主实验 末步 PI (α={a}, γ=1.0)", pv, cellmean(L, a, "1.0"),
        cellmean(Lo, a, "1.0") if len(Lo) else None, "{:.3f}")
c, aa, bb, ind, cp = mediation(L)
rec("LLM主实验 a 路径 α→HE", -0.757, aa, mediation(Lo)[1] if len(Lo) else None)
rec("LLM主实验 b 路径 HE→PI", 0.630, bb, mediation(Lo)[2] if len(Lo) else None)
rec("LLM主实验 间接效应 HE", -0.477, ind, mediation(Lo)[3] if len(Lo) else None)
rec("LLM主实验 总效应 c", -0.361, c, mediation(Lo)[0] if len(Lo) else None)

# 2) LLM 延长档
for tag, pv in (("llm_extend_T100", None), ("llm_extend_T100_old", None)):
    if tag not in DATA:
        continue
    is_old = tag.endswith("_old")
    dd = DATA[tag][1]
    t30 = dd[dd.step == 30]; t100 = dd[dd.step == 100]
    rec(f"[{tag}] PI t=30", 0.311, t30.PI.mean(), None, "{:.3f}", expect_old=not is_old)
    rec(f"[{tag}] PI t=100", 0.417, t100.PI.mean(), None, "{:.3f}", expect_old=not is_old)
    rec(f"[{tag}] β_std (t=100 合并γ)", -0.053, beta_std_pooled(t100)[0], None, expect_old=not is_old)
    rec(f"[{tag}] HE t=30", 0.946, t30.HE.mean(), None, "{:.3f}", expect_old=not is_old)

# 3) 随机 feed 对照
if "llm_randomfeed_control" in DATA:
    rf = last(DATA["llm_randomfeed_control"][1])
    rec("随机feed 末步 PI (随机臂)", 0.161, rf[rf.alpha == -1].PI.mean() if (rf.alpha == -1).any() else rf.PI.mean(), None, "{:.3f}")
    rec("随机feed 末步 HE (随机臂)", 0.757, rf.HE.mean(), None, "{:.3f}")
    if "llm_randomfeed_control_old" in DATA:
        rfo = last(DATA["llm_randomfeed_control_old"][1])
        print(f"  [参考] 旧数据 随机feed PI={rfo.PI.mean():.3f} HE={rfo.HE.mean():.3f}")

# 4) 在线消融（配对差 = 干预臂 ΔPI − 同 (α,γ,seed) 的基线 ΔPI）
def delta_by_seed(df):
    g = df.sort_values("step").groupby(["alpha", "gamma", "seed"]).agg(d1=("PI", "first"), d2=("PI", "last"))
    g["dlt"] = g.d2 - g.d1
    return g.dlt


bdl = delta_by_seed(d)
print(f"  [基线] 主实验 ΔPI = {bdl.mean():+.4f}（论文 +0.0105）")
for tag, pv, pv_own in (("llm_ablation_no_post", -0.1150, -0.1045),
                        ("llm_ablation_no_emotion", -0.0183, -0.0078)):
    if tag in DATA:
        tdl = delta_by_seed(DATA[tag][1])
        pair = (tdl - bdl.reindex(tdl.index)).dropna()
        rec(f"在线消融 干预臂 ΔPI [{tag}]", pv_own, tdl.mean(), None)
        rec(f"在线消融 配对差 [{tag}]（n={len(pair)}）", pv, pair.mean(), None)

# 5) N=100
if "llm_n100_scale" in DATA:
    n1 = last(DATA["llm_n100_scale"][1])
    for a, pv in ((0.0, 0.348), (1.0, 0.363), (2.0, 0.353)):
        rec(f"N=100 末步 PI (α={a})", pv, n1[n1.alpha == a].PI.mean(), None, "{:.3f}")
    rec("N=100 β_std", 0.073, beta_std(n1), None, "{:.3f}")

# 6) Qwen
if "llm_qwen_full" in DATA:
    q = last(DATA["llm_qwen_full"][1])
    cq, aq, bq, iq, cpq = mediation(q)
    rec("Qwen a 路径 α→HE", -0.827, aq, None)
    rec("Qwen b 路径 HE→PI", 0.302, bq, None)
    rec("Qwen 间接效应", -0.250, iq, None)
    rec("Qwen 总效应 c", 0.013, cq, None)

# 7) 规则臂主实验（新=重放目录 / 旧=results）
def load_rule(root, st):
    p = os.path.join(root, st, "逐run时间序列")
    fs = sorted(glob.glob(os.path.join(p, "*.csv")))
    out = []
    for f in fs:
        d = pd.read_csv(f)
        if len(d):
            d["src_run"] = os.path.basename(f)
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


rm_new = load_rule(REPLAY, "rule_main_d4")
rm_old = load_rule(OLD, "rule_main_d4")
for src, tag in ((rm_new, "新(重放)"), (rm_old, "旧")):
    if not len(src):
        continue
    lastr = src.sort_values("step").groupby(["alpha", "gamma", "src_run"]).tail(1)
    for g, pv in ((1.0, -0.566), (0.1, -0.179)):
        rec(f"Rule主实验 β_std (γ={g}) [{tag}]", pv,
            beta_std(lastr[lastr.gamma.astype(str) == str(g)]), None)
    for a, pv in ((0.0, 0.307), (2.0, 0.269)):
        rec(f"Rule主实验 末步 PI (α={a}, γ=1.0) [{tag}]", pv, cellmean(lastr, a, "1.0"), None, "{:.3f}")

print()
print(f"{'项目':<46}{'论文':>10}{'新数据':>12}{'旧数据':>12}  判定")
print("-" * 96)
for it, p, n, o, ok in rows:
    print(f"{it:<46}{p:>10}{n:>12}{o:>12}   {ok}")
print("-" * 96)
bad = [r for r in rows if r[4] == "✘"]
print(f"共 {len(rows)} 项；新数据与论文一致 {len(rows)-len(bad)} 项，不一致 {len(bad)} 项")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "newdata_check.json"),
                     "w", encoding="utf-8"), ensure_ascii=False, indent=1)
