# -*- coding: utf-8 -*-
"""论文图 1-8（修正版）：
   图1 Rule 主实验（不变）；图2 上排=LLM 主实验 T=30、下排=延长档 T=100；
   图3-8 的 LLM 部分一律以 probe/results_prompt_direction（主实验 T=30）为准。
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
# --- portability shim (auto-added) ---------------------------------
import os as _os
_BASE_ROOT = _os.getenv("ABM_BASE_ROOT", r"D:\21130\Desktop\大创过程")
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
_VENV_PY = _os.getenv("ABM_VENV_PY", r"C:\Users\21130\.workbuddy\binaries\python\envs\default\Scripts\python.exe")
# -------------------------------------------------------------------

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

PROJ = _BASE_ROOT
BASE = os.path.join(PROJ, "实验代码", "最终代码和数据")
NEW = os.path.join(BASE, "results_direction")
OLD = os.path.join(BASE, "results")
MAIN_DIR = os.path.join(PROJ, "probe_提示词明确性", "results_prompt_direction")
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)

RULE_TS = os.path.join(NEW, "_VPI_FIX_20261010", "rule_main_d4", "逐run时间序列")
RULE_SUM = os.path.join(NEW, "_VPI_FIX_20261010", "rule_main_d4", "rule_summary.csv")
# rule_small 取自本轮规则臂重放目录（results_replay_20261010），不再引用作废的 results/
REPLAY = os.path.join(BASE, "results_replay_20261010")
RULESM_SUM = os.path.join(REPLAY, "rule_small_d4", "rule_small_summary.csv")

C_RULE = "#A32D2D"; C_LLM = "#185FA5"; C_QWEN = "#2E7D32"; C_EXT = "#7EA6C9"
RX = re.compile(r"^llm_robust_run_(\d+)\.csv$")


def _load(pat, keyfn):
    seen, uniq = set(), []
    for f in sorted(glob.glob(pat, recursive=True)):
        b = os.path.basename(f); m = RX.match(b)
        if not m: continue
        k = keyfn(f, m)
        if k in seen: continue
        seen.add(k); uniq.append((k, f))
    out = []
    for k, f in uniq:
        d = pd.read_csv(f)
        if len(d):
            d["run_id"] = k; out.append(d)
    return pd.concat(out, ignore_index=True)


def load_main():
    return _load(os.path.join(MAIN_DIR, "a*", "runs", "llm_robust_run_*.csv"),
                 lambda f, m: f"{os.path.basename(os.path.dirname(os.path.dirname(f)))}:{int(m.group(1))}")


def load_llm(stage, root=NEW):
    return _load(os.path.join(root, stage, "**", "runs", "llm_robust_run_*.csv"), lambda f, m: int(m.group(1)))


def load_rule_ts():
    out = []
    for f in sorted(glob.glob(os.path.join(RULE_TS, "rule_run_*.csv"))):
        d = pd.read_csv(f)
        d["run_id"] = int(re.search(r"rule_run_(\d+)", os.path.basename(f)).group(1))
        out.append(d)
    return pd.concat(out, ignore_index=True)


def band(ax, x, mat, color, ls="-", lw=1.5, alpha=0.12, label=None, z=3):
    m = np.nanmean(mat, axis=0)
    n = np.sum(~np.isnan(mat), axis=0)
    ci = 1.96 * np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.plot(x, m, ls, color=color, lw=lw, label=label, zorder=z)
    ax.fill_between(x, m - ci, m + ci, color=color, alpha=alpha, lw=0, zorder=z - 1)
    return m


def z(x):
    x = np.asarray(x, float); s = x.std(ddof=0)
    return (x - x.mean()) / s if s > 0 else x * 0


def ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


# ============================ 数据 ============================
MAIN = load_main()
EXT = load_llm("llm_extend_T100")
qw = load_llm("llm_qwen_full")
rf = load_llm("llm_randomfeed_control")
rule_ts = load_rule_ts()
rule_sum = pd.read_csv(RULE_SUM)
rulesm_sum = pd.read_csv(RULESM_SUM)

ALPHAS5 = [0.0, 0.5, 1.0, 1.5, 2.0]


# ============ 图1 Rule 主实验时序 ============
def fig1():
    steps = np.arange(1, 201)
    panels = [("PI", "舆论极化指数 PI"), ("ER", "极端率 ER"), ("EM", "平均情绪 EM"), ("SR", "沉默率 SR")]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(ALPHAS5)))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for a, c in zip(ALPHAS5, cols):
            for gm, ls in ((0.1, "--"), (1.0, "-")):
                g = rule_ts[np.isclose(rule_ts.alpha, a) & np.isclose(rule_ts.gamma, gm)]
                if g.empty: continue
                pv = g.pivot_table(index="run_id", columns="step", values=col).reindex(columns=steps)
                band(ax, steps, pv.to_numpy(float), c, ls=ls, lw=1.3,
                     label=rf"$\alpha$={a:g}" if gm == 1.0 else None)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("时间步 t"); ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("图1　规则主体主实验中 PI/ER/EM/SR 的时序演化（实线 γ=1.0，虚线 γ=0.1；N=200，T=200，R=50）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "fig1_rule_timeseries.png")


# ============ 图2 LLM 时序（上排=主实验 T=30，下排=延长档 T=100） ============
def _llm_panel(ax, df, col, T):
    steps = np.arange(1, T + 1)
    for a, c, ls, lab in [(0.0, "#7FB3D9", "-", "α=0 (γ=1.0)"), (2.0, C_LLM, "-", "α=2 (γ=1.0)"),
                          (0.0, "#B7D4EA", "--", "α=0 (γ=0.1)"), (2.0, "#0B2E4F", "--", "α=2 (γ=0.1)")]:
        gm = 1.0 if ls == "-" else 0.1
        g = df[np.isclose(df.alpha, a) & np.isclose(df.gamma, gm)]
        if g.empty: continue
        pv = g.pivot_table(index="run_id", columns="step", values=col).reindex(columns=steps)
        band(ax, steps, pv.to_numpy(float), c, ls=ls, lw=1.5, alpha=0.10, label=lab)
    ax.grid(alpha=0.25, lw=0.5)


def fig2():
    panels = [("PI", "舆论极化指数 PI"), ("ER", "极端率 ER"), ("EM", "平均情绪 EM"), ("SR", "沉默率 SR")]
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 6.8))
    for j, (col, title) in enumerate(panels):
        _llm_panel(axes[0, j], MAIN, col, 30)
        axes[0, j].axvline(30, color="#888888", lw=0.9, ls=":")
        axes[0, j].set_title(title, fontsize=11)
        axes[0, j].set_xlabel("时间步 t（主实验 T=30）")
        _llm_panel(axes[1, j], EXT, col, 100)
        axes[1, j].set_xlabel("时间步 t（延长档 T=100）")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("图2　LLM 主体时序演化（上排：主实验 N=30, T=30, R=10；下排：延长档 N=30, T=100, R=10；8α×2γ，含 95% CI）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, "fig2_llm_timeseries.png")


# ============ 图3 Rule vs LLM 归一化对比 ============
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    for ax, col, title in zip(axes, ["PI", "EM", "SR"], ["舆论极化指数 PI", "平均情绪 EM", "沉默率 SR"]):
        for gm, ls in ((1.0, "-"), (0.1, "--")):
            g = rule_ts[np.isclose(rule_ts.gamma, gm)]
            pv = g.pivot_table(index="run_id", columns="step", values=col)
            band(ax, np.arange(1, 201) / 200.0, pv.to_numpy(float), C_RULE, ls=ls, lw=1.8, alpha=0.10, label=f"Rule, γ={gm:g}")
        for gm, ls in ((1.0, "-"), (0.1, "--")):
            g = MAIN[np.isclose(MAIN.gamma, gm)]
            pv = g.pivot_table(index="run_id", columns="step", values=col)
            band(ax, np.arange(1, 31) / 30.0, pv.to_numpy(float), C_LLM, ls=ls, lw=1.8, alpha=0.10, label=f"LLM 主实验, γ={gm:g}")
        ax.set_title(title, fontsize=12); ax.set_xlabel("归一化时间 t/T"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("图3　规则主体（T=200）与 LLM 主体主实验（T=30）的时序对比（横轴归一化至 0–1）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); save(fig, "fig3_rule_vs_llm_normalized.png")


# ============ 图4 α 剂量—响应 ============
def fig4():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    E30 = MAIN[MAIN.step == 30]
    for gm in [0.1, 1.0]:
        s = E30[np.isclose(E30.gamma, gm)]
        ax.plot(s.groupby("alpha").PI.mean().index.to_numpy(), s.groupby("alpha").PI.mean().to_numpy(),
                "-" if gm == 1.0 else "--", marker="o", ms=4, color=C_LLM, lw=1.6, label=f"LLM 主实验, γ={gm:g}")
    for lab, df, gm, c, ls in [("Rule 主实验, γ=1.0", rule_sum, 1.0, C_RULE, "-"),
                               ("Rule 受限规模, γ=1.0", rulesm_sum, 1.0, "#D98C3F", "-")]:
        s = df[np.isclose(df.gamma, gm)]
        m = s.groupby("alpha").PI.mean()
        ax.plot(m.index.to_numpy(), m.to_numpy(), ls, marker="o", ms=4, color=c, lw=1.6, label=lab)
    ax.set_xlabel("互动推荐权重 α"); ax.set_ylabel("末步舆论极化指数 PI")
    ax.set_title("图4　PI 的 α 剂量—响应曲线", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(); save(fig, "fig4_alpha_dose_response.png")


# ============ 图5 线性/二次拟合 ============
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    sets = [("Rule 主实验, γ=1.0", rule_sum[np.isclose(rule_sum.gamma, 1.0)], C_RULE),
            ("Rule 受限规模, γ=1.0", rulesm_sum[np.isclose(rulesm_sum.gamma, 1.0)], "#D98C3F"),
            ("LLM 主实验, γ=1.0", MAIN[(MAIN.step == 30) & np.isclose(MAIN.gamma, 1.0)], C_LLM)]
    for ax, (lab, s, c) in zip(axes, sets):
        x = s["alpha"].to_numpy(float); y = s["PI"].to_numpy(float)
        xs = np.linspace(x.min(), x.max(), 100)
        b1 = np.polyfit(x, y, 1); b2 = np.polyfit(x, y, 2)
        ax.scatter(x, y, s=14, color=c, alpha=0.55, lw=0)
        ax.plot(xs, np.polyval(b1, xs), "-", color="#333333", lw=1.6, label="一次拟合")
        ax.plot(xs, np.polyval(b2, xs), "--", color="#C0392B", lw=1.6, label="二次拟合")
        ax.set_title(lab, fontsize=11); ax.set_xlabel("α"); ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("末步 PI"); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("图5　PI ~ α 的线性/二次多项式拟合对比", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); save(fig, "fig5_poly_fit.png")


# ============ 图6 中介路径图 ============
def mediation(df, step, cols=("HE", "EE", "EM")):
    D = df[df.step == step]
    az, gz = z(D["alpha"]), z(D["gamma"])
    out = {"total_c": float(ols(z(D["PI"]), [az, gz])[1])}
    for M in cols:
        mz, pz = z(D[M]), z(D["PI"])
        a = float(ols(mz, [az, gz])[1]); bb = float(ols(pz, [az, gz, mz])[3])
        out[M] = {"a": round(a, 3), "b": round(bb, 3), "ind": round(a * bb, 3)}
    return out


def fig6():
    R = mediation(rule_sum, 200); L = mediation(MAIN, 30)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    for ax, lab, M, color in [(axes[0], "规则主体（Rule 主实验，末步 n=500）", R, C_RULE),
                              (axes[1], "LLM 主体（主实验 t=30，n=160）", L, C_LLM)]:
        ax.set_xlim(0, 12); ax.set_ylim(0, 11.6); ax.axis("off")
        ax.text(6.0, 11.1, f"总效应 c = {M['total_c']:+.3f}", ha="center", fontsize=11, color="#111111")
        ax.add_patch(FancyBboxPatch((0.4, 5.0), 2.0, 1.2, boxstyle="round,pad=0.05", fc="#F2F2F2", ec="#555555", lw=1.0))
        ax.text(1.4, 5.6, "推荐强度 α", ha="center", va="center", fontsize=10)
        ax.add_patch(FancyBboxPatch((9.6, 5.0), 2.0, 1.2, boxstyle="round,pad=0.05", fc="#F2F2F2", ec="#555555", lw=1.0))
        ax.text(10.6, 5.6, "极化 PI", ha="center", va="center", fontsize=10)
        for m, y in zip(["HE", "EE", "EM"], [9.3, 5.6, 1.9]):
            ax.add_patch(FancyBboxPatch((5.0, y - 0.6), 2.0, 1.2, boxstyle="round,pad=0.05", fc="#EAF1F8", ec="#555555", lw=1.0))
            ax.text(6.0, y, m, ha="center", va="center", fontsize=10)
            v = M[m]
            ax.add_patch(FancyArrowPatch((2.5, 5.6), (4.95, y), arrowstyle="-|>", mutation_scale=11, color="#555555", lw=1.0))
            ax.text(3.4, (5.6 + y) / 2 + 0.30, f"a={v['a']}", fontsize=8.5, color="#333333")
            ax.add_patch(FancyArrowPatch((7.05, y), (9.55, 5.6), arrowstyle="-|>", mutation_scale=11, color="#555555", lw=1.0))
            ax.text(8.1, (5.6 + y) / 2 + 0.30, f"b={v['b']}", fontsize=8.5, color="#333333")
        s = "   ".join(f"{m}: {M[m]['ind']:+.3f}" for m in ["HE", "EE", "EM"])
        ax.text(6.0, 0.15, "间接效应  " + s, ha="center", fontsize=9.5, color="#111111")
        ax.plot([1.4, 1.4, 10.6, 10.6], [5.0, 0.62, 0.62, 5.0], ls="--", color="#111111", lw=1.1)
        ax.add_patch(FancyArrowPatch((9.4, 0.62), (10.6, 0.62), arrowstyle="-|>", mutation_scale=11, color="#111111", lw=1.1))
        ax.text(6.0, 0.95, "直接效应 c′", ha="center", fontsize=9, color="#111111")
        ax.set_title(lab, fontsize=11, pad=6)
    fig.suptitle("图6　中介效应路径图（系数为样本内 z 标准化）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, "fig6_mediation_path.png")


# ============ 图7 稳健性 ============
def fig7():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ax = axes[0]
    alg = MAIN[(MAIN.step == 30) & np.isclose(MAIN.alpha, 1.0)]
    r30 = rf[rf.step == 30]
    m = pd.merge(r30, alg, on=["gamma", "seed"], suffixes=("_rf", "_al"))
    metrics = ["PI", "HE", "ER"]; xs = np.arange(len(metrics)); w = 0.35
    ax.bar(xs - w / 2, [m[k + "_al"].mean() for k in metrics], w, color=C_LLM, label="算法臂 (α=1.0)")
    ax.bar(xs + w / 2, [m[k + "_rf"].mean() for k in metrics], w, color="#9AA6B2", label="均匀随机 feed")
    ax.set_ylim(0, 1.30)
    for i, k in enumerate(metrics):
        d = (m[k + "_rf"] - m[k + "_al"]); se = d.std(ddof=1) / np.sqrt(len(d))
        top = max(m[k + "_al"].mean(), m[k + "_rf"].mean())
        ax.text(i, top + 0.05, f"Δ={d.mean():+.3f}\n[{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels(["PI", "HE", "ER"])
    ax.set_title("(a) 随机 feed 对照（配对 n=20）", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(alpha=0.25, axis="y", lw=0.5)
    ax = axes[1]
    base = MAIN[MAIN.step == 30].set_index("seed")["PI"] - MAIN[MAIN.step == 1].set_index("seed")["PI"]
    base = base[~base.index.duplicated()]
    arms = [("llm_ablation_no_post", "冻结表达"), ("llm_ablation_no_emotion", "冻结情绪")]
    vals = []
    for st, lab in arms:
        a = load_llm(st)
        aseed = a[a.step == 30].set_index("seed")["PI"] - a[a.step == 1].set_index("seed")["PI"]
        aseed = aseed[~aseed.index.duplicated()]
        j = pd.concat([base.rename("b"), aseed.rename("a")], axis=1).dropna()
        vals.append((lab, j["a"].mean(), (j["a"] - j["b"]).mean(), (j["a"] - j["b"]).std(ddof=1) / np.sqrt(len(j))))
    ax.bar([0, 1], [v[1] for v in vals], 0.5, color=["#C0504D", "#E8A33D"])
    ax.axhline(0, color="#333333", lw=0.8)
    ax.errorbar([0, 1], [v[1] for v in vals], yerr=[1.96 * v[3] for v in vals], fmt="none", ecolor="#333333", capsize=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([v[0] for v in vals])
    ax.set_ylabel("ΔPI（末步 − 首步）"); ax.set_ylim(-0.20, 0.10)
    ax.set_title("(b) 在线消融的极化变化", fontsize=11); ax.grid(alpha=0.25, axis="y", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v[1] - 0.038, f"{v[1]:+.3f}\n配对差 {v[2]:+.3f}", ha="center", va="top", fontsize=7.5)
    ax = axes[2]
    E30 = MAIN[MAIN.step == 30]
    for gm, c, lab in [(1.0, C_LLM, "LLM γ=1.0"), (0.1, "#7FB3D9", "LLM γ=0.1")]:
        s = E30[np.isclose(E30.gamma, gm)]
        M = s.groupby("alpha")["PI"].mean()
        ax.scatter([gm], [(M.max() - M.min()) / s["PI"].std(ddof=1)], s=60, color=c, label=lab)
    s = rule_sum[np.isclose(rule_sum.gamma, 1.0)]
    Mr = s.groupby("alpha")["PI"].mean()
    ax.scatter([1.0], [(Mr.max() - Mr.min()) / s["PI"].std(ddof=1)], s=80, marker="s", color=C_RULE, label="Rule γ=1.0")
    ax.set_ylabel("跨 α 极差 / 运行间标准差"); ax.set_xticks([0.1, 1.0]); ax.set_xticklabels(["γ=0.1", "γ=1.0"])
    ax.set_title("(c) 条件间变化幅度与噪声之比", fontsize=11); ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("图7　稳健性与可辨识性", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); save(fig, "fig7_robustness.png")


# ============ 图8 跨模型 ============
def fig8():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    steps = np.arange(1, 31); ax = axes[0]
    for df, c, lab in [(MAIN, C_LLM, "DeepSeek-V4.1-Flash"), (qw, C_QWEN, "Qwen3.8-Flash")]:
        g = df[df.step <= 30]
        pv = g.pivot_table(index="run_id", columns="step", values="PI").reindex(columns=steps)
        band(ax, steps, pv.to_numpy(float), c, lw=1.8, label=lab)
    ax.set_title("(a) PI 的时间轨迹", fontsize=11); ax.set_xlabel("时间步 t"); ax.set_ylabel("PI")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, lw=0.5)
    ax = axes[1]
    for df, c, lab, mk in [(MAIN, C_LLM, "DeepSeek", "o"), (qw, C_QWEN, "Qwen", "s")]:
        for gm, ls, ec in [(0.1, "--", c), (1.0, "-", c)]:
            s = df[(df.step == 30) & np.isclose(df.gamma, gm)]
            M = s.groupby("alpha")["PI"].mean(); SE = s.groupby("alpha")["PI"].sem()
            ax.errorbar(M.index, M.values, yerr=1.96 * SE.values, fmt=mk, ms=4, color=c, ecolor=c,
                        alpha=0.9 if gm == 1.0 else 0.5, capsize=2, ls="none", label=f"{lab}, γ={gm:g}")
            xs = np.linspace(0, 2, 50)
            ax.plot(xs, np.polyval(np.polyfit(M.index.to_numpy(float), M.values, 1), xs), ls, color=c, lw=1.3, alpha=0.9 if gm == 1.0 else 0.5)
    ax.set_title("(b) PI 对推荐强度 α 的回归", fontsize=11); ax.set_xlabel("α")
    ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.grid(alpha=0.25, lw=0.5)
    ax = axes[2]
    metrics = ["EM", "SR", "ER", "HE"]; xs = np.arange(len(metrics)); w = 0.36
    ax.bar(xs - w / 2, [MAIN[MAIN.step == 30][k].mean() for k in metrics], w, color=C_LLM, label="DeepSeek")
    ax.bar(xs + w / 2, [qw[qw.step == 30][k].mean() for k in metrics], w, color=C_QWEN, label="Qwen")
    ax.set_xticks(xs); ax.set_xticklabels(metrics)
    ax.set_title("(c) 行为侧指标（t=30）", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25, axis="y", lw=0.5)
    fig.suptitle("图8　跨模型复现", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); save(fig, "fig8_crossmodel.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print("ALL DONE")
