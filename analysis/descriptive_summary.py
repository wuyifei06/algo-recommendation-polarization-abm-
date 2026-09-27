#!/usr/bin/env python
"""四个实验数据集的描述性汇总（均值/SD），输出单个 xlsx 供分享。

数据集：
- rule_main   ：results/run_by_rule/rule_summary.csv（N=200, T=200, 500 runs）
- rule_small  ：results/rule_small/rule_small_summary.csv（N=30, T=30, 50 runs）
- llm_main    ：results/run_by_llm/llm_robustness_v2/summary_all_runs.csv（50 runs）
- llm_robust  ：results/llm_robust/llm_robust_summary.csv（DeepSeek，9 runs）

每个数据集按 (alpha, gamma) 聚合 PI/VPI/SR/ER/EM/HE/EE 的 mean/sd/n。
只读取数据，不修改原始文件。输出：results/analysis_summary/descriptive_summary.xlsx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["PI", "VPI", "SR", "ER", "EM", "HE", "EE"]

DATASETS = {
    "rule_main": ("results/run_by_rule/rule_summary.csv", "Rule 正式实验 N=200 T=200"),
    "rule_small": ("results/rule_small/rule_small_summary.csv", "Rule 受限规模 N=30 T=30"),
    "llm_main": ("results/run_by_llm/llm_robustness_v2/summary_all_runs.csv", "LLM 主实验 qwen3.7-flash"),
    "llm_robust": ("results/llm_robust/llm_robust_summary.csv", "LLM 稳健性 DeepSeek-V4.1-Flash"),
}


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "config.py").exists() and (candidate / "model").is_dir():
            return candidate
    raise FileNotFoundError("无法定位项目根目录")


def last_step_per_run(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["step"] = pd.to_numeric(work["step"], errors="coerce")
    if "run_id" not in work.columns:
        return work
    work = work.dropna(subset=["run_id", "step"])
    max_step = work.groupby("run_id")["step"].transform("max")
    final = work.loc[work["step"].eq(max_step)].copy()
    return final.sort_values(["run_id", "step"]).drop_duplicates("run_id", keep="last")


def aggregate(frame: pd.DataFrame, dataset: str, description: str) -> pd.DataFrame:
    rows = []
    for (alpha, gamma), group in frame.groupby(["alpha", "gamma"]):
        row = {
            "dataset": dataset,
            "description": description,
            "alpha": float(alpha),
            "gamma": float(gamma),
            "n": len(group),
        }
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = find_project_root(Path(__file__).resolve().parent)
    output_dir = root / "results" / "analysis_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "descriptive_summary.xlsx"

    combined = []
    per_dataset = {}
    for name, (rel_path, description) in DATASETS.items():
        path = root / rel_path
        if not path.exists():
            print(f"[汇总] 跳过缺失数据集 {name}: {path}")
            continue
        frame = last_step_per_run(pd.read_csv(path))
        agg = aggregate(frame, name, description)
        per_dataset[name] = agg
        combined.append(agg)
        print(f"[汇总] {name}: {len(frame)} runs -> {len(agg)} 条件")

    if not combined:
        raise RuntimeError("没有可用的数据集")
    combined_frame = pd.concat(combined, ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        combined_frame.to_excel(writer, sheet_name="combined", index=False)
        for name, agg in per_dataset.items():
            agg.to_excel(writer, sheet_name=name, index=False)
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
            print(f"[汇总] 表头格式未应用：{exc}")

    combined_frame.to_csv(output_dir / "descriptive_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[汇总] 输出：{output_path}")
    key = combined_frame[["dataset", "alpha", "gamma", "n", "PI_mean", "VPI_mean", "EM_mean", "HE_mean", "EE_mean"]]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
