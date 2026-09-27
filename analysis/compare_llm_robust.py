#!/usr/bin/env python
"""LLM 稳健性实验 vs 主 LLM 实验：趋势对比分析。

读取：
- results/llm_robust/llm_robust_summary.csv（任务三输出，需先手动运行实验）
- 主 LLM 实验汇总：优先 results/run_by_llm/llm_robustness_v2/summary_all_runs.csv，
  缺失时回退拼接 results/run_by_llm 下的逐 run CSV。

输出（写入 results/llm_robust/comparison/）：
- llm_robust_vs_main.csv   按 (alpha, gamma) 聚合的均值/标准差对比表
- llm_robust_vs_main.png   PI/VPI/EM/HE/EE 随 alpha 变化的趋势对比图

只读取数据，不修改任何原始文件。
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["PI", "VPI", "SR", "ER", "EM", "HE", "EE"]
ROBUST_CONDITIONS = [(0.0, 0.1), (1.0, 1.0), (2.0, 1.0)]


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "config.py").exists() and (candidate / "model").is_dir():
            return candidate
    raise FileNotFoundError("无法定位包含 config.py 和 model/ 的项目根目录")


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or RuntimeError(f"无法读取 CSV: {path}")


def last_step_per_run(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["step"] = pd.to_numeric(work["step"], errors="coerce")
    work = work.dropna(subset=["run_id", "step"])
    max_step = work.groupby("run_id")["step"].transform("max")
    final = work.loc[work["step"].eq(max_step)].copy()
    return final.sort_values(["run_id", "step"]).drop_duplicates("run_id", keep="last")


def load_main_llm(root: Path) -> pd.DataFrame:
    preferred = root / "results" / "run_by_llm" / "llm_robustness_v2" / "summary_all_runs.csv"
    if preferred.exists():
        frame = read_csv_robust(preferred)
        return last_step_per_run(frame)
    base = root / "results" / "run_by_llm"
    frames = []
    for path in sorted(base.rglob("*.csv")):
        if "llm_robustness_v2" in path.parts:
            continue
        try:
            frame = read_csv_robust(path)
        except Exception as exc:
            warnings.warn(f"跳过无法读取的文件 {path}: {exc}")
            continue
        if {"run_id", "step", "alpha", "gamma"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"未找到主 LLM 实验数据：{base}")
    return last_step_per_run(pd.concat(frames, ignore_index=True))


def aggregate(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for (alpha, gamma), group in frame.groupby(["alpha", "gamma"]):
        row = {"dataset": label, "alpha": float(alpha), "gamma": float(gamma), "n": len(group)}
        for metric in METRICS:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def draw_trends(combined: pd.DataFrame, output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plot_metrics = ["PI", "VPI", "EM", "HE", "EE"]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.2 * len(plot_metrics), 4), sharex=False)
    colors = {"main": "#2563EB", "robust": "#DC2626"}
    for ax, metric in zip(axes, plot_metrics):
        for label, name in [("main", "主 LLM 实验"), ("robust", "稳健性实验")]:
            data = combined.loc[combined["dataset"] == label].sort_values(["gamma", "alpha"])
            for gamma, group in data.groupby("gamma"):
                style = "-" if float(gamma) == 0.1 else "--"
                ax.errorbar(
                    group["alpha"], group[f"{metric}_mean"], yerr=group[f"{metric}_sd"],
                    marker="o", linestyle=style, capsize=3, color=colors[label],
                    label=f"{name} γ={gamma:g}",
                )
        ax.set_title(metric)
        ax.set_xlabel("alpha")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("最后一步均值 ± SD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 稳健性实验 vs 主 LLM 实验趋势对比")
    parser.add_argument("--robust-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = find_project_root(Path(__file__).resolve().parent)

    robust_path = args.robust_summary or root / "results" / "llm_robust" / "llm_robust_summary.csv"
    if not robust_path.exists():
        print(f"[对比] 未找到稳健性实验结果：{robust_path}")
        print("[对比] 请先手动运行：python experiment_llm_robust.py")
        sys.exit(1)

    robust = last_step_per_run(read_csv_robust(robust_path))
    main_llm = load_main_llm(root)
    # 仅保留稳健性实验覆盖的三个条件，保证对比口径一致。
    def keep_conditions(frame: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(False, index=frame.index)
        for alpha, gamma in ROBUST_CONDITIONS:
            mask |= np.isclose(frame["alpha"], alpha) & np.isclose(frame["gamma"], gamma)
        return frame.loc[mask]

    combined = pd.concat(
        [aggregate(keep_conditions(main_llm), "main"), aggregate(keep_conditions(robust), "robust")],
        ignore_index=True,
    )
    output_dir = args.output_dir or root / "results" / "llm_robust" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "llm_robust_vs_main.csv"
    png_path = output_dir / "llm_robust_vs_main.png"
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    draw_trends(combined, png_path)
    print(f"[对比] 对比表：{csv_path}")
    print(f"[对比] 趋势图：{png_path}")
    key = combined.loc[combined["alpha"].isin([c[0] for c in ROBUST_CONDITIONS])]
    print(key[["dataset", "alpha", "gamma", "n", "PI_mean", "VPI_mean", "EM_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
