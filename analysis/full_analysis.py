#!/usr/bin/env python
"""完整数据分析流水线（对应《方案总览》第四节分析计划）。

只读取已有实验结果，不修改任何原始数据，不运行仿真实验。
输出目录：results/analysis_report/
    tables.xlsx        表1-表8
    fig2_rule_timeseries.png
    fig3_llm_timeseries.png
    fig4_rule_vs_llm_timeseries.png
    fig5_alpha_dose_response.png
    fig6_nonlinearity.png
    fig7_robustness.png
    analysis_summary.md （由报告脚本另行撰写，本脚本只输出数据与图）

数据源：
    rule_main   results/run_by_rule/rule_summary.csv + results/run_by_rule/存档点/rule_run_*.csv
    rule_small  results/rule_small/rule_small_summary.csv + runs/rule_small_run_*.csv
    llm_main    results/run_by_llm/llm_robustness_v2/summary_all_runs.csv + runs/run_*.csv
    llm_robust  results/llm_robust/llm_robust_summary.csv + runs/llm_robust_run_*.csv
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from scipy import stats

METRICS = ["PI", "VPI", "SR", "ER", "EM", "HE", "EE"]
TS_METRICS = ["PI", "VPI", "EM", "SR"]
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.1, 1.0]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d627a8", "#7f7f7f"]
STYLES = ["-", "--", "-.", ":", "-."]

FINAL_SOURCES = {
    "rule_main": ("results/run_by_rule/rule_summary.csv", "Rule ABM 正式（N=200, T=200, 500 runs）"),
    "rule_small": ("results/rule_small/rule_small_summary.csv", "Rule ABM 受限规模（N=30, T=30, 50 runs）"),
    "llm_main": ("results/run_by_llm/llm_robustness_v2/summary_all_runs.csv", "LLM ABM 主实验（Qwen3.7-Flash, 50 runs）"),
    "llm_robust": ("results/llm_robust/llm_robust_summary.csv", "LLM ABM 稳健性（DeepSeek-V4.1-Flash, 9 runs）"),
}
TS_SOURCES = {
    "rule_main": ("results/run_by_rule/存档点", re.compile(r"rule_run_(\d+)\.csv$"), 200),
    "rule_small": ("results/rule_small/runs", re.compile(r"rule_small_run_(\d+)\.csv$"), 30),
    "llm_main": ("results/run_by_llm/llm_robustness_v2/runs", re.compile(r"run_(\d+)\.csv$"), 30),
    "llm_robust": ("results/llm_robust/runs", re.compile(r"llm_robust_run_(\d+)\.csv$"), 30),
}


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "config.py").exists() and (candidate / "model").is_dir():
            return candidate
    raise FileNotFoundError("无法定位项目根目录")


def read_csv_robust(path: Path) -> pd.DataFrame:
    last = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last = exc
    raise last or RuntimeError(f"无法读取 {path}")


def read_final(path: Path) -> pd.DataFrame:
    frame = read_csv_robust(path)
    if frame.columns[0].startswith("Unnamed"):
        frame = read_csv_robust(path) if False else frame
    frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    if "step" in frame.columns:
        frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    if "run_id" in frame.columns:
        frame = frame.drop_duplicates("run_id", keep="last")
    return frame


def load_timeseries(directory: Path, pattern: re.Pattern, steps: int, label: str) -> pd.DataFrame:
    records = []
    skipped = 0
    for path in sorted(directory.glob("*.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            skipped += 1
            continue
        if len(frame) != steps or not set(METRICS).issubset(frame.columns):
            skipped += 1
            continue
        run_id = int(match.group(1))
        # rule 存档点部分文件没有 alpha/gamma 列以外的一致命名，逐行取用现有列
        for _, row in frame.iterrows():
            record = {metric: float(row[metric]) for metric in METRICS if metric in frame.columns}
            record.update({
                "step": int(row["step"]),
                "run_id": run_id,
                "alpha": float(row["alpha"]),
                "gamma": float(row["gamma"]),
                "dataset": label,
            })
            records.append(record)
    if skipped:
        print(f"[时序] {label}: 跳过 {skipped} 个不完整文件")
    if not records:
        raise RuntimeError(f"{label} 没有可用的完整时序 run")
    data = pd.DataFrame(records)
    print(f"[时序] {label}: {data['run_id'].nunique()} runs × {steps} steps")
    return data


def ci95(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1])
    return 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def descriptive(final: pd.DataFrame, dataset: str, description: str) -> pd.DataFrame:
    rows = []
    for (alpha, gamma), group in final.groupby(["alpha", "gamma"]):
        row = {"dataset": dataset, "description": description, "alpha": float(alpha), "gamma": float(gamma), "n": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            half = 1.96 * sd / np.sqrt(len(values)) if len(values) > 1 else np.nan
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = float(np.mean(values) - half) if np.isfinite(half) else np.nan
            row[f"{metric}_ci_high"] = float(np.mean(values) + half) if np.isfinite(half) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "gamma", "alpha"])


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def effect_sizes(final: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for gamma in sorted(final["gamma"].unique()):
        subset = final.loc[np.isclose(final["gamma"], gamma)]
        for metric in METRICS:
            work = subset[["alpha", metric]].dropna().copy()
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
            work = work.dropna()
            if work["alpha"].nunique() < 2:
                continue
            x = sm.add_constant(work["alpha"].astype(float))
            model = sm.OLS(work[metric].astype(float), x).fit()
            beta = float(model.params["alpha"])
            sd_x = float(work["alpha"].std(ddof=1))
            sd_y = float(work[metric].std(ddof=1))
            rows.append({
                "dataset": dataset, "gamma": float(gamma), "metric": metric, "n": len(work),
                "beta_unstd": beta, "beta_std": beta * sd_x / sd_y if sd_y > 1e-12 else np.nan,
                "r2": float(model.rsquared), "p_alpha": float(model.pvalues["alpha"]),
                "cohens_d_alpha0_vs_alpha2": cohens_d(
                    work.loc[np.isclose(work["alpha"], 2.0), metric].to_numpy(float),
                    work.loc[np.isclose(work["alpha"], 0.0), metric].to_numpy(float),
                ),
            })
    return pd.DataFrame(rows)


def nonlinearity(final: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for gamma in sorted(final["gamma"].unique()):
        subset = final.loc[np.isclose(final["gamma"], gamma)].copy()
        for metric in ["PI", "VPI", "EM"]:
            work = subset[["alpha", metric]].dropna().astype(float)
            if work["alpha"].nunique() < 3:
                continue
            row = {"dataset": dataset, "gamma": float(gamma), "metric": metric, "n": len(work)}
            r2s = {}
            for degree in (1, 2, 3):
                x = np.column_stack([work["alpha"].to_numpy(float) ** k for k in range(1, degree + 1)])
                x = sm.add_constant(x)
                model = sm.OLS(work[metric].to_numpy(float), x).fit()
                r2s[degree] = float(model.rsquared)
                row[f"r2_deg{degree}"] = float(model.rsquared)
                row[f"aic_deg{degree}"] = float(model.aic)
                if degree == 2:
                    row["quad_coef"] = float(model.params[2])
                    row["quad_p"] = float(model.pvalues[2])
                    row["r2_gain_quad_over_linear"] = r2s[2] - r2s[1]
                    b1, b2 = float(model.params[1]), float(model.params[2])
                    row["vertex_alpha"] = -b1 / (2 * b2) if abs(b2) > 1e-12 else np.nan
                    row["shape"] = (
                        "U型（先降后升）" if b2 > 1e-9 and np.isfinite(row["vertex_alpha"]) and 0 <= row["vertex_alpha"] <= 2
                        else "倒U型（先升后降）" if b2 < -1e-9 and np.isfinite(row["vertex_alpha"]) and 0 <= row["vertex_alpha"] <= 2
                        else "单调/无内部极值"
                    )
            rho, p_rho = stats.spearmanr(work["alpha"], work[metric])
            row["spearman_rho"] = float(rho)
            row["spearman_p"] = float(p_rho)
            means = work.groupby("alpha")[metric].mean().sort_index()
            diffs = means.diff().abs()
            row["max_adjacent_shift"] = float(diffs.max()) if len(diffs) else np.nan
            row["max_adjacent_shift_at"] = float(diffs.idxmax()) if len(diffs) and np.isfinite(diffs.max()) else np.nan
            row["plateau_range"] = float(means.max() - means.min()) / (abs(means.mean()) + 1e-12)
            rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(new: pd.DataFrame, base: pd.DataFrame, new_label: str, base_label: str) -> pd.DataFrame:
    rows = []
    merged_keys = sorted(set(zip(new["alpha"], new["gamma"])) & set(zip(base["alpha"], base["gamma"])))
    for alpha, gamma in merged_keys:
        a = new.loc[np.isclose(new["alpha"], alpha) & np.isclose(new["gamma"], gamma)]
        b = base.loc[np.isclose(base["alpha"], alpha) & np.isclose(base["gamma"], gamma)]
        row = {"alpha": float(alpha), "gamma": float(gamma), f"n_{new_label}": len(a), f"n_{base_label}": len(b)}
        for metric in METRICS:
            va = pd.to_numeric(a[metric], errors="coerce").dropna().to_numpy(float)
            vb = pd.to_numeric(b[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_{new_label}_mean"] = float(np.mean(va)) if len(va) else np.nan
            row[f"{metric}_{base_label}_mean"] = float(np.mean(vb)) if len(vb) else np.nan
            row[f"{metric}_d"] = cohens_d(va, vb)
            if len(va) > 1 and len(vb) > 1:
                row[f"{metric}_p_mannwhitney"] = float(stats.mannwhitneyu(va, vb).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_timeseries(data: pd.DataFrame, dataset: str, output: Path, title: str, steps: int) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()
    titles = {"PI": r"观点极化 $PI$", "VPI": r"可见意见方差 $VPI$", "EM": r"平均情绪 $EM$", "SR": r"沉默率 $SR$"}
    for ax, metric in zip(axes, TS_METRICS):
        for alpha, color, style in zip(ALPHAS, COLORS, STYLES):
            for gamma, lw_alpha in zip(GAMMAS, [0.18, 0.10]):
                condition = data.loc[data["dataset"] == dataset]
                condition = condition.loc[np.isclose(condition["alpha"], alpha) & np.isclose(condition["gamma"], gamma)]
                if condition.empty:
                    continue
                pivot = condition.pivot(index="run_id", columns="step", values=metric)
                pivot = pivot.reindex(columns=list(range(1, steps + 1)))
                values = pivot.to_numpy(float)
                mean, ci = np.nanmean(values, axis=0), ci95(values)
                x = np.arange(1, steps + 1)
                ax.plot(x, mean, color=color, linestyle=style, linewidth=1.8,
                        label=rf"$\alpha$={alpha:g}, $\gamma$={gamma:g}" if metric == "PI" else None)
                if values.shape[0] > 1:
                    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=lw_alpha, linewidth=0)
        ax.set_title(titles[metric], fontsize=15)
        ax.set_xlabel("Step")
        ax.set_ylabel("均值 (95% CI)")
        ax.grid(alpha=0.25)
    handles = [
        Line2D([0], [0], color=COLORS[ALPHAS.index(a)], linestyle=STYLES[ALPHAS.index(a)], linewidth=1.8, label=rf"$\alpha$={a:g}")
        for a in ALPHAS
    ] + [
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=1.8, label=r"$\gamma$=0.1"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.8, label=r"$\gamma$=1.0"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=17)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {output}")


def plot_rule_vs_llm(rule_ts: pd.DataFrame, llm_ts: pd.DataFrame, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    titles = {"PI": r"观点极化 $PI$", "EM": r"平均情绪 $EM$", "SR": r"沉默率 $SR$"}
    for row_index, (metric, label) in enumerate(titles.items()):
        for col_index, (data, name, steps) in enumerate([(rule_ts, "Rule ABM", 200), (llm_ts, "LLM ABM", 30)]):
            ax = axes[row_index, col_index]
            for alpha, color, style in zip(ALPHAS, COLORS, STYLES):
                for gamma, band in zip(GAMMAS, [0.16, 0.09]):
                    condition = data.loc[np.isclose(data["alpha"], alpha) & np.isclose(data["gamma"], gamma)]
                    if condition.empty:
                        continue
                    pivot = condition.pivot(index="run_id", columns="step", values=metric).reindex(columns=list(range(1, steps + 1)))
                    values = pivot.to_numpy(float)
                    mean, ci = np.nanmean(values, axis=0), ci95(values)
                    x = np.linspace(0, 1, steps)
                    ax.plot(x, mean, color=color, linestyle=("-" if np.isclose(gamma, 0.1) else "--"),
                            linewidth=1.9, label=rf"$\alpha$={alpha:g}")
                    if values.shape[0] > 1:
                        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=band, linewidth=0)
            ax.set_title(f"{name} — {label}", fontsize=14)
            ax.set_xlabel("归一化时间步 (0=开始, 1=结束)")
            ax.set_ylabel("均值 (95% CI)")
            ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], color=COLORS[ALPHAS.index(a)], linewidth=1.9, label=rf"$\alpha$={a:g}") for a in ALPHAS]
    fig.legend(handles=handles, loc="upper right", ncol=5, frameon=False, fontsize=11)
    fig.suptitle("Rule ABM vs LLM ABM：时序对比（实现线为 γ=0.1，虚线为 γ=1.0）", fontsize=17)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {output}")


def plot_dose_response(final_map: dict, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()
    for ax, (name, frame) in zip(axes, final_map.items()):
        for gamma, marker in zip(GAMMAS, ["o", "s"]):
            subset = frame.loc[np.isclose(frame["gamma"], gamma)]
            means, lows, highs, xs = [], [], [], []
            for alpha in sorted(subset["alpha"].unique()):
                group = pd.to_numeric(subset.loc[np.isclose(subset["alpha"], alpha), "PI"], errors="coerce").dropna().to_numpy(float)
                if len(group) == 0:
                    continue
                sd = float(np.std(group, ddof=1)) if len(group) > 1 else 0.0
                half = 1.96 * sd / np.sqrt(len(group)) if len(group) > 1 else 0.0
                xs.append(alpha); means.append(float(np.mean(group))); lows.append(float(np.mean(group) - half)); highs.append(float(np.mean(group) + half))
            ax.errorbar(xs, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
                        marker=marker, capsize=4, linewidth=2, label=rf"$\gamma$={gamma:g}")
        ax.set_title(name, fontsize=14)
        ax.set_xlabel(r"$\alpha$（互动推荐权重）")
        ax.set_ylabel("PI 均值 (95% CI)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("剂量-响应：PI 随 α 的变化（四个实验）", fontsize=16)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {output}")


def plot_nonlinearity(final_map: dict, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for ax, (name, frame) in zip(axes, final_map.items()):
        subset = frame.loc[np.isclose(frame["gamma"], 1.0)]
        work = subset[["alpha", "PI"]].dropna().astype(float)
        if work["alpha"].nunique() < 3:
            ax.axis("off")
            ax.set_title(f"{name}（条件不足）")
            continue
        ax.scatter(work["alpha"], work["PI"], s=18, alpha=0.35, color="#94A3B8", label="run 观测")
        grid = np.linspace(0, 2, 200)
        for degree, color in [(1, "#2563EB"), (2, "#DC2626"), (3, "#0F766E")]:
            x = np.column_stack([work["alpha"].to_numpy(float) ** k for k in range(1, degree + 1)])
            model = sm.OLS(work["PI"].to_numpy(float), sm.add_constant(x)).fit()
            y = model.params[0] + sum(model.params[k] * grid ** k for k in range(1, degree + 1))
            ax.plot(grid, y, color=color, linewidth=2, label=f"{degree} 次（R²={model.rsquared:.3f}）")
        ax.set_title(f"{name}（γ=1.0）", fontsize=13)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("PI")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("非线性检验：PI ~ α 的多项式拟合", fontsize=16)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {output}")


def plot_robustness(final_map: dict, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    pairs = [("Rule 规模对照", "rule_main", "rule_small"), ("LLM 跨模型", "llm_main", "llm_robust")]
    for ax, (title, base_name, new_name) in zip(axes[:2], pairs):
        base, new = final_map[base_name], final_map[new_name]
        for gamma, marker in zip(GAMMAS, ["o", "s"]):
            for frame, name, color in [(base, base_name, "#2563EB"), (new, new_name, "#DC2626")]:
                subset = frame.loc[np.isclose(frame["gamma"], gamma)]
                means = subset.groupby("alpha")["PI"].mean()
                ax.plot(means.index, means.values, marker=marker, color=color, linewidth=2,
                        linestyle="-" if np.isclose(gamma, 0.1) else "--", label=f"{name}, γ={gamma:g}")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("PI 均值")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    ax = axes[2]
    ax.axis("off")
    lines = []
    for base_name, new_name in [("rule_main", "rule_small"), ("llm_main", "llm_robust")]:
        base, new = final_map[base_name], final_map[new_name]
        for metric in ["PI", "EM", "SR"]:
            b = base.loc[np.isclose(base["gamma"], 1.0), metric].mean()
            n = new.loc[np.isclose(new["gamma"], 1.0), metric].mean()
            lines.append(f"{base_name} vs {new_name} | {metric}: {b:.3f} → {n:.3f}")
    ax.text(0.02, 0.95, "γ=1.0 下的均值对比\n" + "\n".join(lines), va="top", fontsize=11, family="Microsoft YaHei")
    fig.suptitle("稳健性检验：规模对照与跨模型对照", fontsize=16)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {output}")


def main() -> None:
    root = find_project_root(Path(__file__).resolve().parent)
    output_dir = root / "results" / "analysis_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_map, desc_rows = {}, []
    for name, (rel, description) in FINAL_SOURCES.items():
        path = root / rel
        if not path.exists():
            print(f"[主流程] 跳过缺失数据集 {name}")
            continue
        frame = read_final(path)
        frame = frame.dropna(subset=["alpha", "gamma"])
        if "status" in frame.columns:
            frame = frame.loc[frame["status"].astype(str).str.startswith("completed")]
        final_map[name] = frame
        desc_rows.append(descriptive(frame, name, description))
        print(f"[主流程] {name}: {len(frame)} runs")

    descriptive_all = pd.concat(desc_rows, ignore_index=True)

    # 表1/表2：定义与设计
    t1 = pd.DataFrame({
        "变量": ["α", "γ", "HE", "EE", "EM", "PI", "VPI", "SR", "ER"],
        "含义": ["互动内容推荐权重", "情绪内容推荐权重", "信息暴露同质性", "情绪暴露强度", "平均情绪",
                 "总体意见极化", "可见意见极化", "沉默率", "极端表达比例"],
        "角色": ["自变量", "自变量/分组", "中介变量", "中介变量", "中介变量", "因变量", "因变量", "辅助指标", "辅助指标"],
        "取值/范围": ["{0, 0.5, 1, 1.5, 2}", "{0.1, 1.0}", "[0,1]", "[0,1]", "[0,1]", "[0,1]", "[0,1]", "[0,1]", "[0,1]"],
    })
    t2 = pd.DataFrame([
        {"实验": "Rule ABM", "N": 200, "T": 200, "条件": "5α×2γ", "R": 50, "运行数": 500, "模型": "规则模型", "用途": "主实验"},
        {"实验": "LLM ABM", "N": 30, "T": 30, "条件": "5α×2γ", "R": 5, "运行数": 50, "模型": "Qwen3.7-Flash", "用途": "机制验证"},
        {"实验": "Rule small", "N": 30, "T": 30, "条件": "5α×2γ", "R": 5, "运行数": 50, "模型": "规则模型", "用途": "规模对照（已完成）"},
        {"实验": "LLM robust", "N": 30, "T": 30, "条件": "3 条件", "R": 3, "运行数": 9, "模型": "DeepSeek-V4.1-Flash", "用途": "跨模型稳健性（已完成）"},
    ])

    # 表3-表7
    eff = pd.concat([effect_sizes(frame, name) for name, frame in final_map.items()], ignore_index=True)
    nonlin = pd.concat([nonlinearity(frame, name) for name, frame in final_map.items()], ignore_index=True)
    t_rule_robust = robustness_table(final_map["rule_small"], final_map["rule_main"], "small", "main")
    t_llm_robust = robustness_table(final_map["llm_robust"], final_map["llm_main"], "deepseek", "qwen")

    mediation_summary = pd.read_csv(root / "results" / "mediation" / "mediation_summary.csv")
    pooled = mediation_summary.loc[(mediation_summary["scope"] == "pooled_gamma_covariate")]
    med_table = pooled[["dataset", "outcome", "analysis_type", "mediator_set", "a", "b", "d21",
                        "c", "c_prime", "indirect_effect", "bootstrap_se", "ci_low", "ci_high",
                        "p_indirect", "proportion_mediated", "n", "vif_direct_max"]].copy()

    # 时序
    ts_map = {}
    for name, (rel_dir, pattern, steps) in TS_SOURCES.items():
        directory = root / rel_dir
        if not directory.exists():
            print(f"[主流程] 跳过缺失时序目录 {name}")
            continue
        ts_map[name] = load_timeseries(directory, pattern, steps, name)

    with pd.ExcelWriter(output_dir / "tables.xlsx", engine="openpyxl") as writer:
        t1.to_excel(writer, sheet_name="表1_变量定义", index=False)
        t2.to_excel(writer, sheet_name="表2_实验设计", index=False)
        descriptive_all.to_excel(writer, sheet_name="表3_描述性统计", index=False)
        med_table.to_excel(writer, sheet_name="表4_中介效应汇总", index=False)
        pd.concat([t_rule_robust.assign(comparison="rule_small_vs_rule_main"),
                   t_llm_robust.assign(comparison="llm_deepseek_vs_qwen")]).to_excel(writer, sheet_name="表5_稳健性对比", index=False)
        eff.to_excel(writer, sheet_name="表6_效应量", index=False)
        nonlin.to_excel(writer, sheet_name="表7_非线性检验", index=False)
        try:
            from openpyxl.styles import Alignment, Font, PatternFill
            for sheet in writer.book.worksheets:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cell in sheet[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor="1F4E78")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
        except Exception as exc:
            print(f"[主流程] 表头格式未应用：{exc}")

    # 图
    if "rule_main" in ts_map:
        plot_timeseries(ts_map["rule_main"], "rule_main", output_dir / "fig2_rule_timeseries.png",
                        "Rule ABM 时序（N=200, T=200, 500 runs）", 200)
    if "llm_main" in ts_map:
        plot_timeseries(ts_map["llm_main"], "llm_main", output_dir / "fig3_llm_timeseries.png",
                        "LLM ABM 时序（Qwen3.7-Flash, N=30, T=30, 50 runs）", 30)
    if "rule_main" in ts_map and "llm_main" in ts_map:
        plot_rule_vs_llm(ts_map["rule_main"], ts_map["llm_main"], output_dir / "fig4_rule_vs_llm_timeseries.png")
    plot_dose_response(final_map, output_dir / "fig5_alpha_dose_response.png")
    plot_nonlinearity(final_map, output_dir / "fig6_nonlinearity.png")
    plot_robustness(final_map, output_dir / "fig7_robustness.png")

    descriptive_all.to_csv(output_dir / "表3_描述性统计.csv", index=False, encoding="utf-8-sig")
    eff.to_csv(output_dir / "表6_效应量.csv", index=False, encoding="utf-8-sig")
    nonlin.to_csv(output_dir / "表7_非线性检验.csv", index=False, encoding="utf-8-sig")
    print(f"\n[主流程] 输出目录：{output_dir}")
    print("\n[关键结果] 表6 效应量（PI，γ=1.0）：")
    print(eff.loc[(eff["metric"] == "PI") & np.isclose(eff["gamma"], 1.0)].to_string(index=False))
    print("\n[关键结果] 表7 非线性（PI）：")
    print(nonlin.loc[nonlin["metric"] == "PI"][["dataset", "gamma", "r2_deg1", "r2_deg2", "quad_p", "r2_gain_quad_over_linear", "shape", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
