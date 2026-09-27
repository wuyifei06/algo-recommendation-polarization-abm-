"""Generate rule-style time-series figures from complete LLM checkpoints.

Only observed, complete checkpoints are used.  Estimated workbooks and
incomplete checkpoints are intentionally excluded so the figure cannot
silently introduce fabricated trajectories.
"""

from __future__ import annotations

import json
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


METRICS = ["PI", "VPI", "ER", "EM", "SR"]
METRIC_TITLES = {
    "PI": r"Opinion polarization ($PI$)",
    "VPI": r"Visible opinion variance ($VPI$)",
    "ER": r"Extreme-opinion rate ($ER$)",
    "EM": r"Mean emotion ($EM$)",
    "SR": r"Silence rate ($SR$)",
}
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d627a8", "#7f7f7f"]
LINE_STYLES = ["-", "--", "-.", ":", "-."]
RUN_RE = re.compile(r"llm_run_(\d+)\.checkpoint\.json$")
V2_RUN_RE = re.compile(r"run_(\d+)\.csv$")
ROBUST_RUN_RE = re.compile(r"llm_robust_run_(\d+)\.csv$")


def _same_number(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))


def load_complete_observed_runs(checkpoint_dir: Path, steps: int = 30, ignored_run_ids=(11,)) -> pd.DataFrame:
    """Load complete time series from checkpoints, excluding estimates and ignored runs."""

    records = []
    ignored = {int(run_id) for run_id in ignored_run_ids}
    for path in sorted(checkpoint_dir.glob("llm_run_*.checkpoint.json")):
        match = RUN_RE.search(path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        if run_id in ignored:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[LLM 图表] 跳过无法读取的 checkpoint：{path.name} ({exc})")
            continue

        series = payload.get("metrics_time_series") or []
        if int(payload.get("current_step", 0)) < steps or len(series) < steps:
            print(
                f"[LLM 图表] 跳过未完成 run_id={run_id}："
                f"current_step={payload.get('current_step')}, series={len(series)}"
            )
            continue

        frame = pd.DataFrame(series).sort_values("step").reset_index(drop=True)
        if len(frame) != steps or list(frame["step"].astype(int)) != list(range(1, steps + 1)):
            print(f"[LLM 图表] 跳过 step 不完整的 run_id={run_id}")
            continue
        if any(metric not in frame.columns for metric in METRICS):
            print(f"[LLM 图表] 跳过缺少指标的 run_id={run_id}")
            continue

        for _, row in frame.iterrows():
            record = {metric: float(row[metric]) for metric in METRICS}
            record.update(
                {
                    "step": int(row["step"]),
                    "run_id": run_id,
                    "alpha": float(payload["alpha"]),
                    "gamma": float(payload["gamma"]),
                    "seed": int(payload.get("seed", -1)),
                }
            )
            records.append(record)

    if not records:
        raise RuntimeError("没有找到可用于制图的完整 LLM 观测 checkpoint。")
    return pd.DataFrame(records)


def load_v2_observed_runs(runs_dir: Path, steps: int = 30) -> pd.DataFrame:
    """Load complete time series from the formal LLM02 run CSV files.

    The formal v2 directory contains one time-series CSV and one metadata JSON
    per run.  Completed runs with fallback are retained because they are real
    observations; failed or incomplete runs are excluded.
    """

    records = []
    for path in sorted(runs_dir.glob("run_*.csv")):
        match = V2_RUN_RE.fullmatch(path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        metadata_path = path.with_name(f"run_{run_id}.metadata.json")
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[LLM02 图表] 跳过无法读取的 metadata：{metadata_path.name} ({exc})")
                continue

        status = str(metadata.get("status", "completed"))
        if status not in {"completed", "completed_with_fallback"}:
            print(f"[LLM02 图表] 跳过失败 run_id={run_id}：status={status}")
            continue

        try:
            frame = pd.read_csv(path).sort_values("step").reset_index(drop=True)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[LLM02 图表] 跳过无法读取的 run：{path.name} ({exc})")
            continue

        required = set(METRICS) | {"step", "alpha", "gamma"}
        if len(frame) != steps or not required.issubset(frame.columns):
            print(f"[LLM02 图表] 跳过不完整 run_id={run_id}：rows={len(frame)}")
            continue
        if list(frame["step"].astype(int)) != list(range(1, steps + 1)):
            print(f"[LLM02 图表] 跳过 step 不完整的 run_id={run_id}")
            continue

        spec = metadata.get("spec", {})
        alpha = float(spec.get("alpha", frame["alpha"].iloc[0]))
        gamma = float(spec.get("gamma", frame["gamma"].iloc[0]))
        seed = int(spec.get("seed", frame.get("seed", pd.Series([-1])).iloc[0]))
        for _, row in frame.iterrows():
            record = {metric: float(row[metric]) for metric in METRICS}
            record.update(
                {
                    "step": int(row["step"]),
                    "run_id": run_id,
                    "alpha": alpha,
                    "gamma": gamma,
                    "seed": seed,
                    "status": status,
                }
            )
            records.append(record)

    if not records:
        raise RuntimeError("没有找到可用于制图的完整 LLM02 观测 run。")
    return pd.DataFrame(records)


def load_llm_robust_runs(runs_dir: Path, steps: int = 30) -> pd.DataFrame:
    """Load complete time series from the DeepSeek robustness experiment run CSVs.

    ``results/llm_robust/runs/llm_robust_run_<id>.csv`` 每个文件是一个
    完整 run 的 30 步时间序列，列内自带 alpha/gamma/seed。
    """

    records = []
    for path in sorted(runs_dir.glob("llm_robust_run_*.csv")):
        match = ROBUST_RUN_RE.fullmatch(path.name)
        if not match:
            continue
        run_id = int(match.group(1))
        try:
            frame = pd.read_csv(path).sort_values("step").reset_index(drop=True)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[LLM robust 图表] 跳过无法读取的 run：{path.name} ({exc})")
            continue

        required = set(METRICS) | {"step", "alpha", "gamma"}
        if len(frame) != steps or not required.issubset(frame.columns):
            print(f"[LLM robust 图表] 跳过不完整 run_id={run_id}：rows={len(frame)}")
            continue
        if list(frame["step"].astype(int)) != list(range(1, steps + 1)):
            print(f"[LLM robust 图表] 跳过 step 不完整的 run_id={run_id}")
            continue

        alpha = float(frame["alpha"].iloc[0])
        gamma = float(frame["gamma"].iloc[0])
        seed = int(frame["seed"].iloc[0]) if "seed" in frame.columns else -1
        for _, row in frame.iterrows():
            record = {metric: float(row[metric]) for metric in METRICS}
            record.update(
                {
                    "step": int(row["step"]),
                    "run_id": run_id,
                    "alpha": alpha,
                    "gamma": gamma,
                    "seed": seed,
                }
            )
            records.append(record)

    if not records:
        raise RuntimeError("没有找到可用于制图的完整 LLM robustness 观测 run。")
    return pd.DataFrame(records)


def _ci95(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1], dtype=float)
    return 1.96 * np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _condition_label(gamma: float) -> str:
    return rf"Condition ($\gamma$ = {gamma:g})"


def plot_llm_figures(
    data: pd.DataFrame,
    out_dir: Path,
    steps: int = 30,
    filename_prefix: str = "llm",
    title: str = "LLM-driven ABM: time-series means across observed runs",
) -> list[Path]:
    """Create one five-panel figure per observed gamma condition."""

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    observed_conditions = (
        data[["alpha", "gamma"]]
        .drop_duplicates()
        .sort_values(["gamma", "alpha"])
        .itertuples(index=False)
    )
    gammas = sorted({float(row.gamma) for row in observed_conditions})

    for gamma in gammas:
        fig, axes = plt.subplots(3, 2, figsize=(18, 13.5), constrained_layout=False)
        axes = axes.flatten()
        fig.suptitle(
            title,
            fontsize=22,
            y=0.985,
        )

        plotted_alphas: list[float] = []
        alpha_counts: dict[float, int] = {}
        for axis, metric in zip(axes[:5], METRICS):
            metric_frame = data[data["gamma"].apply(lambda value: _same_number(value, gamma))]
            for alpha, color, line_style in zip(ALPHAS, COLORS, LINE_STYLES):
                condition = metric_frame[
                    metric_frame["alpha"].apply(lambda value: _same_number(value, alpha))
                ]
                if condition.empty:
                    continue
                pivot = condition.pivot(index="run_id", columns="step", values=metric)
                pivot = pivot.reindex(columns=list(range(1, steps + 1)))
                values = pivot.to_numpy(dtype=float)
                mean = np.nanmean(values, axis=0)
                ci = _ci95(values)
                x = np.arange(1, steps + 1)
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=line_style,
                    linewidth=2.5,
                    label=rf"$\alpha$ = {alpha:g}",
                )
                if values.shape[0] > 1:
                    axis.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.12, linewidth=0)
                if alpha not in plotted_alphas:
                    plotted_alphas.append(alpha)
                alpha_counts[alpha] = int(values.shape[0])

            axis.set_title(METRIC_TITLES[metric], fontsize=17)
            axis.set_xlabel("Step", fontsize=14)
            axis.set_ylabel("Mean (95% CI)", fontsize=14)
            axis.grid(True, alpha=0.22)
            axis.tick_params(labelsize=12)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color=COLORS[ALPHAS.index(alpha)],
                linestyle=LINE_STYLES[ALPHAS.index(alpha)],
                linewidth=2.5,
                label=rf"$\alpha$ = {alpha:g} (n={alpha_counts[alpha]})",
            )
            for alpha in ALPHAS
            if alpha in plotted_alphas
        ]
        axes[5].axis("off")
        axes[5].legend(
            handles=legend_handles,
            title=_condition_label(gamma),
            loc="center",
            frameon=False,
            fontsize=14,
            title_fontsize=16,
        )

        condition_rows = data[data["gamma"].apply(lambda value: _same_number(value, gamma))]
        condition_counts = condition_rows.groupby("alpha")["run_id"].nunique().to_dict()
        missing = [alpha for alpha in ALPHAS if alpha not in plotted_alphas]
        note = "Observed runs only; n shown in legend for each α×γ condition; no estimated data."
        if missing:
            note += " Missing observed α: " + ", ".join(f"{alpha:g}" for alpha in missing) + "."
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=10, color="0.35")
        fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.065, wspace=0.14, hspace=0.25)

        output = out_dir / f"{filename_prefix}_gamma_{gamma:g}.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
        print(f"[LLM 图表] 已生成：{output}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM time-series figures.")
    parser.add_argument(
        "--source",
        choices=["legacy", "v2", "llm_robust"],
        default="legacy",
        help="legacy=old checkpoints; v2=formal LLM02 run CSVs; llm_robust=DeepSeek robustness run CSVs",
    )
    parser.add_argument("--v2-root", type=Path, default=None, help="LLM02 result directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="figure output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if args.source == "llm_robust":
        robust_root = args.v2_root or (project_root / "results" / "llm_robust")
        out_dir = args.out_dir or (robust_root / "数据图")
        data = load_llm_robust_runs(robust_root / "runs", steps=30)
        filename_prefix = "llm_robust"
        title = "DeepSeek LLM ABM (robustness): time-series means across observed runs"
    elif args.source == "v2":
        v2_root = args.v2_root or (project_root / "results" / "run_by_llm" / "llm_robustness_v2")
        out_dir = args.out_dir or (project_root / "results" / "run_by_llm" / "数据图" / "llm02")
        data = load_v2_observed_runs(v2_root / "runs", steps=30)
        filename_prefix = "llm02"
        title = "LLM02-driven ABM: time-series means across observed runs"
    else:
        checkpoint_dir = project_root / "results" / "run_by_llm" / "存档点"
        out_dir = args.out_dir or (project_root / "results" / "run_by_llm" / "数据图")
        data = load_complete_observed_runs(checkpoint_dir, steps=30, ignored_run_ids=(11,))
        filename_prefix = "llm"
        title = "LLM-driven ABM: time-series means across observed runs"

    print(
        f"[LLM 图表] 使用 {data['run_id'].nunique()} 个完整观测 run，"
        f"条件数={data[['alpha', 'gamma']].drop_duplicates().shape[0]}"
    )
    plot_llm_figures(data, out_dir, steps=30, filename_prefix=filename_prefix, title=title)


if __name__ == "__main__":
    main()
