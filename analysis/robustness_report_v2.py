"""Aggregate, validate, and plot the versioned LLM robustness experiment.

The script never fills missing conditions.  It writes explicit coverage and
failure information, uses observed repeats for uncertainty estimates, and
marks absent conditions in figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULE_SUMMARY = PROJECT_ROOT / "results" / "run_by_rule" / "rule_summary.csv"
FORMAL_ROOT = PROJECT_ROOT / "results" / "llm_robustness_v2"
SMOKE_ROOT = PROJECT_ROOT / "results" / "llm_robustness_v2_smoke"
METRICS = ["SR", "EM", "PI", "VPI", "ER", "HE", "EE"]
PERCENT_METRICS = {"SR", "EM", "ER", "HE", "EE"}
EXPECTED_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
EXPECTED_GAMMAS = [0.1, 1.0]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _load_frames(root: Path) -> Dict[str, pd.DataFrame]:
    rule = _read_csv(RULE_SUMMARY)
    if not rule.empty:
        rule = rule.assign(model="Rule ABM")
    llm = _read_csv(root / "summary.csv")
    if not llm.empty:
        llm = llm.assign(model="LLM ABM v2")
    return {"rule": rule, "llm": llm}


def _expected_n(model: str, root: Path) -> int:
    if model == "Rule ABM":
        return 50
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return int(manifest.get("experiment", {}).get("repeats", 5))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 5


def _aggregate(frames: Dict[str, pd.DataFrame], root: Path) -> pd.DataFrame:
    rows = []
    for name, frame in frames.items():
        if frame.empty or not {"alpha", "gamma"}.issubset(frame.columns):
            continue
        model = "Rule ABM" if name == "rule" else "LLM ABM v2"
        expected_n = _expected_n(model, root)
        for (alpha, gamma), group in frame.groupby(["alpha", "gamma"], dropna=False):
            for metric in METRICS:
                if metric not in group.columns:
                    continue
                values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
                n = int(values.size)
                mean = float(np.mean(values)) if n else float("nan")
                std = float(np.std(values, ddof=1)) if n > 1 else 0.0 if n == 1 else float("nan")
                se = std / math.sqrt(n) if n else float("nan")
                ci95 = 1.96 * se if n else float("nan")
                rows.append(
                    {
                        "model": model,
                        "alpha": float(alpha),
                        "gamma": float(gamma),
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "se": se,
                        "ci95": ci95,
                        "n": n,
                        "expected_n": expected_n,
                        "missing_n": max(0, expected_n - n),
                    }
                )
    return pd.DataFrame(rows)


def _coverage(frames: Dict[str, pd.DataFrame], root: Path) -> pd.DataFrame:
    rows = []
    for model_name, frame in [("Rule ABM", frames["rule"]), ("LLM ABM v2", frames["llm"])]:
        expected_n = _expected_n(model_name, root)
        for alpha in EXPECTED_ALPHAS:
            for gamma in EXPECTED_GAMMAS:
                if frame.empty or not {"alpha", "gamma"}.issubset(frame.columns):
                    n = 0
                else:
                    n = int(
                        (
                            np.isclose(frame["alpha"].astype(float), alpha)
                            & np.isclose(frame["gamma"].astype(float), gamma)
                        ).sum()
                    )
                rows.append(
                    {
                        "model": model_name,
                        "alpha": alpha,
                        "gamma": gamma,
                        "observed_n": n,
                        "expected_n": expected_n,
                        "complete": n >= expected_n,
                        "missing_n": max(0, expected_n - n),
                    }
                )
    return pd.DataFrame(rows)


def _plot_metric_grid(aggregate: pd.DataFrame, coverage: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, len(METRICS), figsize=(24, 8.5), squeeze=False, constrained_layout=False)
    colors = {"Rule ABM": "#1f77b4", "LLM ABM v2": "#d62728"}
    styles = {"Rule ABM": "-", "LLM ABM v2": "--"}
    for row_idx, gamma in enumerate(EXPECTED_GAMMAS):
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            sub = aggregate[(np.isclose(aggregate["gamma"], gamma)) & (aggregate["metric"] == metric)] if not aggregate.empty else pd.DataFrame()
            for model in ("Rule ABM", "LLM ABM v2"):
                points = sub[sub["model"] == model].sort_values("alpha")
                if points.empty:
                    continue
                ax.errorbar(
                    points["alpha"],
                    points["mean"],
                    yerr=points["ci95"].fillna(0.0),
                    color=colors[model],
                    linestyle=styles[model],
                    marker="o",
                    capsize=3,
                    label=model,
                )
                for _, point in points.iterrows():
                    ax.annotate(
                        f"n={int(point['n'])}/{int(point['expected_n'])}",
                        (point["alpha"], point["mean"]),
                        textcoords="offset points",
                        xytext=(0, 5 if model == "Rule ABM" else -13),
                        ha="center",
                        fontsize=7,
                        color=colors[model],
                    )
            ax.set_title(f"{metric} | γ={gamma:g}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("mean ± 95% CI")
            ax.set_xticks(EXPECTED_ALPHAS)
            ax.grid(alpha=0.2)
            missing = coverage[(coverage["gamma"] == gamma)]
            for _, item in missing.iterrows():
                if not bool(item["complete"]):
                    ax.annotate(
                        f"missing {int(item['observed_n'])}/{int(item['expected_n'])}",
                        (float(item["alpha"]), 0.0),
                        xytext=(0, -29),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                        color="#666666",
                        rotation=90,
                    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.suptitle("Rule ABM vs LLM ABM v2: final metrics by condition", fontsize=16, y=0.985)
    fig.text(0.5, 0.01, "Points are observed repeat means; error bars are 95% CI. Labels show observed/expected n; missing conditions are not interpolated.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.35, hspace=0.55)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_subset(aggregate: pd.DataFrame, coverage: pd.DataFrame, metrics: List[str], title: str, filename: str, out: Path) -> Path:
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    colors = {"Rule ABM": "#1f77b4", "LLM ABM v2": "#d62728"}
    styles = {"Rule ABM": "-", "LLM ABM v2": "--"}
    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        sub = aggregate[aggregate["metric"] == metric] if not aggregate.empty else pd.DataFrame()
        for model in ("Rule ABM", "LLM ABM v2"):
            points = sub[sub["model"] == model].sort_values(["gamma", "alpha"])
            for gamma in EXPECTED_GAMMAS:
                curve = points[np.isclose(points["gamma"], gamma)]
                if curve.empty:
                    continue
                ax.errorbar(
                    curve["alpha"], curve["mean"], yerr=curve["ci95"].fillna(0.0),
                    color=colors[model], linestyle=styles[model], marker="o",
                    label=f"{model}, γ={gamma:g}", capsize=3,
                )
        ax.set_title(metric)
        ax.set_xlabel("alpha")
        ax.set_ylabel("mean ± 95% CI")
        ax.set_xticks(EXPECTED_ALPHAS)
        ax.grid(alpha=0.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.text(0.5, 0.01, "Only observed conditions are connected. See coverage.csv for exact n and missing runs.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.3)
    fig.savefig(out / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out / filename


def _plot_gamma_effect(aggregate: pd.DataFrame, out: Path) -> Path:
    metrics = ["EM", "SR", "ER"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), squeeze=False)
    colors = {"Rule ABM": "#1f77b4", "LLM ABM v2": "#d62728"}
    styles = {"Rule ABM": "-", "LLM ABM v2": "--"}
    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        sub = aggregate[aggregate["metric"] == metric] if not aggregate.empty else pd.DataFrame()
        for model in ("Rule ABM", "LLM ABM v2"):
            for alpha in EXPECTED_ALPHAS:
                points = sub[(sub["model"] == model) & np.isclose(sub["alpha"], alpha)].sort_values("gamma")
                if points.empty:
                    continue
                ax.errorbar(
                    points["gamma"], points["mean"], yerr=points["ci95"].fillna(0.0),
                    marker="o", linestyle=styles[model], color=colors[model], capsize=3,
                    label=f"{model}, α={alpha:g}", alpha=0.9,
                )
        ax.set_title(metric)
        ax.set_xlabel("gamma")
        ax.set_ylabel("mean ± 95% CI")
        ax.set_xticks(EXPECTED_GAMMAS)
        ax.grid(alpha=0.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Effect of gamma on emotion and related metrics", fontsize=16, y=0.985)
    fig.text(0.5, 0.01, "Labels and missing-condition coverage are reported in coverage.csv; no estimated values are plotted.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.3)
    fig.savefig(out / "gamma_effect_emotion_related.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out / "gamma_effect_emotion_related.png"


def _quality_summary(root: Path) -> Dict[str, float]:
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    rows = [row.get("quality", {}) for row in manifest.get("runs", []) if row.get("quality")]
    if not rows:
        return {}
    result = {}
    for key in ("logical_calls", "valid_responses", "fallbacks", "parse_failures", "provider_errors", "valid_response_rate", "fallback_rate", "parse_failure_rate"):
        values = [float(row.get(key, 0.0)) for row in rows]
        result[key] = float(np.sum(values)) if key.endswith("calls") or key in {"valid_responses", "fallbacks", "parse_failures", "provider_errors"} else float(np.mean(values))
    return result


def _describe_old_data(frames: Dict[str, pd.DataFrame]) -> str:
    old = _read_csv(PROJECT_ROOT / "results" / "run_by_llm" / "存档点" / "llm_run_1.csv")
    files = list((PROJECT_ROOT / "results" / "run_by_llm" / "存档点").glob("llm_run_*.csv"))
    if old.empty:
        return "旧 LLM 观测 CSV 未找到或无法读取。"
    sr = float(old.iloc[-1]["SR"]) if "SR" in old.columns else float("nan")
    return f"旧 LLM 存档目录中可见 {len(files)} 个 CSV；示例 run_1 的最终 SR={sr:g}，该存档不是 v2 正式结果。"


def _write_report(root: Path, frames: Dict[str, pd.DataFrame], aggregate: pd.DataFrame, coverage: pd.DataFrame) -> Path:
    manifest = {}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    llm_coverage = coverage[coverage["model"] == "LLM ABM v2"] if not coverage.empty else pd.DataFrame()
    complete_conditions = int(llm_coverage["complete"].sum()) if not llm_coverage.empty else 0
    planned = int(manifest.get("planned_runs", 50))
    completed_runs = int(manifest.get("completed_runs", len(frames["llm"])))
    failed_runs = manifest.get("failed_runs", [])
    quality = _quality_summary(root)
    expected_decisions = (
        int(manifest.get("planned_runs", 0))
        * int(manifest.get("experiment", {}).get("N", 0))
        * int(manifest.get("experiment", {}).get("steps", 0))
    )
    lines = [
        "# Rule ABM 与 LLM ABM 机制稳健性实验报告（v2）",
        "",
        "> 本报告区分正式实验配置、真实观测和 smoke test。缺失条件没有用插值或复制值填充。",
        "",
        "## 当前状态",
        "",
        f"- v2 结果目录：`{root}`。",
        f"- 计划运行：{planned} 个 run；已纳入汇总：{completed_runs} 个；完整条件：{complete_conditions}/10。",
        f"- 计划逻辑决策数：{expected_decisions:,}；正式实验只有在实际完成并写入 run 元数据后才计入汇总。",
        f"- 失败或未纳入的 run_id：{failed_runs if failed_runs else '无记录'}。",
        f"- 旧数据核查：{_describe_old_data(frames)}",
        "",
        "## 代码与机制修改",
        "",
        "- 新增 `llm/robust_client_v2.py`：真实模型请求、结构化响应解析、超范围/缺失/非法输出 fallback，以及逐调用质量日志。",
        "- 更新 `llm/prompt.py`：明确沉默条件、`express` 0/1、`stance_delta` [-0.05, 0.05]、`emotion_delta` [-0.1, 0.1]、α/γ 的行为含义和 JSON-only 输出。",
        "- Batch 化：每 10 个 Agent 共用一次 API 请求；N=30 时每轮约 3 次请求、T=30 时每 run 约 90 次请求。",
        "- 新增 `model/llm_robustness_v2.py`：复用 Rule ABM 的 Agent 初始化、帖子池、Top-K 推荐、时间步和统一指标函数；只替换行为决策接口。",
        "- 新增 `experiments/llm_robustness_v2.py`：固定 5×2 条件、每条件 5 次重复、独立 seed、checkpoint、运行元数据和 Agent 末状态。",
        "- 新增本报告脚本：统一计算均值、标准差、标准误、95% CI，并对缺失条件显式标注。",
        "",
        "## 环境差异",
        "",
        "- Rule ABM：N=200、200 轮、Top-K=10、α=[0.0, 0.5, 1.0, 1.5, 2.0]、γ=[0.1, 1.0]；现有规则结果每条件 n=50。",
        "- LLM ABM v2：按本项目最新设计使用 N=30、30 轮、Top-K=10、相同初始化逻辑/推荐评分/内容池/指标定义；正式目标为每条件 n=5。N 与 T 与 Rule ABM 不同，是实验规模差异，不应宣称完全同环境。",
        "- 必然存在的机制差异：LLM 输出 Batch 中每个 Agent 的 express/stance_delta/emotion_delta；v2 在每轮先固定同一推荐快照，再并发等待 Batch 结果并按 Agent 顺序应用，而规则模型保留其原有逐 Agent 顺序更新。",
        "",
        "## LLM 输出质量",
        "",
        f"- 逻辑调用数：{quality.get('logical_calls', 0):g}；有效响应：{quality.get('valid_responses', 0):g}；fallback：{quality.get('fallbacks', 0):g}。",
        f"- 解析失败率（按逻辑调用）：{quality.get('parse_failure_rate', float('nan')):.4f}；fallback 率：{quality.get('fallback_rate', float('nan')):.4f}；有效响应率：{quality.get('valid_response_rate', float('nan')):.4f}。",
        "",
        "## 新观测结果",
        "",
    ]
    if frames["llm"].empty:
        lines.append("- 当前目录没有纳入汇总的 v2 LLM 观测，因此没有可报告的 SR、EM、PI/VPI、ER、HE、EE 正式新结果。")
    else:
        for _, row in frames["llm"].sort_values(["alpha", "gamma", "repeat"]).iterrows():
            values = []
            for metric in ["SR", "EM", "PI", "VPI", "ER", "HE", "EE"]:
                if metric in row and pd.notna(row[metric]):
                    values.append(f"{metric}={float(row[metric]):.6f}")
            lines.append(f"- α={float(row['alpha']):g}, γ={float(row['gamma']):g}, repeat={int(row.get('repeat', 0))}, status={row.get('status', 'observed')}：" + ", ".join(values))
    if manifest.get("status") == "planned_only":
        lines.append("- 正式 v2 目前为 plan-only；50 runs 共需约 4,500 次真实 Batch API 请求。未完成的正式条件没有进入结果汇总。")
    lines.extend(["", "## 结果解释边界", ""])
    if completed_runs < planned or complete_conditions < 10:
        lines.append("- 当前 v2 数据尚未覆盖完整正式网格，因此不能据此下定论：LLM 是否仍出现 SR 接近 0、EM 趋势反转或高方差。图表中的 `n=observed/expected` 反映真实覆盖率。")
    else:
        lines.append("- 正式网格已完成；是否存在 SR 接近 0、EM 趋势反转或高方差，应以 aggregate_summary.csv 和图表中的真实 CI 逐条件判断。")
    lines.extend(
        [
            "- 可以写入论文的谨慎结论：规则模型用于揭示在固定推荐与更新规则下的稳定宏观机制；LLM 模型用于检验这些行为机制在自然语言决策接口下的稳健性。",
            "- 目前不能写入的结论：不能在正式网格未完成、有效响应不足或存在大量 fallback 时宣称两类模型结果一致。若指标方向不同，应报告差异并结合表达率、情绪增量、解析失败率和执行顺序分析来源。",
            "",
            "## 输出文件",
            "",
            "- `summary.csv`：纳入汇总的单次实验最终指标。",
            "- `aggregate_summary.csv`：按模型×α×γ×指标汇总的 mean/std/SE/95% CI/n。",
            "- `coverage.csv`：每个条件的 observed/expected n 和缺失数。",
            "- `figures/`：Rule/LLM 的 SR、EM、PI/VPI、ER、HE/EE、γ 影响图。",
            "- `runs/`：每次实验的时间序列、Agent 末状态、checkpoint、调用日志和元数据。",
        ]
    )
    report = root / "EXPERIMENT_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def generate_report(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(root)
    aggregate = _aggregate(frames, root)
    coverage = _coverage(frames, root)
    aggregate.to_csv(root / "aggregate_summary.csv", index=False)
    coverage.to_csv(root / "coverage.csv", index=False)
    _plot_metric_grid(aggregate, coverage, figures / "model_comparison_final_metrics.png")
    _plot_subset(aggregate, coverage, ["HE", "EE"], "Effect of alpha on information exposure", "alpha_information_exposure.png", figures)
    _plot_gamma_effect(aggregate, figures)
    return _write_report(root, frames, aggregate, coverage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="v2 result directory; defaults to formal directory")
    parser.add_argument("--smoke", action="store_true", help="use the smoke result directory")
    args = parser.parse_args()
    root = Path(args.root) if args.root else (SMOKE_ROOT if args.smoke else FORMAL_ROOT)
    print(generate_report(root))


if __name__ == "__main__":
    main()
