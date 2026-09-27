#!/usr/bin/env python
"""Rule-based ABM 受限规模实验。

使用与正式实验相同的 ``SimulationModel``、初始化、推荐算法和指标计算，
仅把设计固定为 N=30、T=30、R=5、5 个 alpha × 2 个 gamma，共 50 runs。
原始正式实验目录不会被读取写入，也不会被覆盖。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "config.py").exists() and (candidate / "model").is_dir():
            return candidate
    raise FileNotFoundError("无法定位包含 config.py 和 model/ 的项目根目录")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))

from config import conf
from model.model import SimulationModel


ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.1, 1.0]
REPEATS = 5
N_AGENTS = 30
STEPS = 30


def run_experiment(output_dir: Path, base_seed: int, force: bool = False) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rule_small_summary.csv"
    expected_runs = len(ALPHAS) * len(GAMMAS) * REPEATS
    if summary_path.exists() and not force:
        existing = pd.read_csv(summary_path)
        if len(existing) == expected_runs and set(existing["step"].astype(int)) == {STEPS}:
            print(f"[Rule small] 已存在完整结果，跳过重跑：{summary_path}")
            return existing

    results = []
    run_id = 0
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            for repeat in range(REPEATS):
                seed = base_seed + run_id
                print(f"[Rule small] run {run_id + 1}/{expected_runs}: alpha={alpha}, gamma={gamma}, repeat={repeat}", flush=True)
                model = SimulationModel(seed=seed, N=N_AGENTS, steps=STEPS, alpha=alpha, gamma=gamma, debug=False)
                model.run()
                if not model.metrics_time_series:
                    raise RuntimeError(f"run_id={run_id} 没有产生指标")
                time_series_path = runs_dir / f"rule_small_run_{run_id}.csv"
                model.save_results(str(time_series_path))
                final = dict(model.metrics_time_series[-1])
                final.update({
                    "run_id": run_id,
                    "repeat": repeat,
                    "alpha": alpha,
                    "gamma": gamma,
                    "seed": seed,
                    "N": N_AGENTS,
                    "steps": STEPS,
                })
                results.append(final)
                pd.DataFrame(results).sort_values("run_id").to_csv(summary_path, index=False)
                run_id += 1

    summary = pd.DataFrame(results).sort_values("run_id").reset_index(drop=True)
    manifest = {
        "experiment": "rule_small",
        "N": N_AGENTS,
        "T": STEPS,
        "R": REPEATS,
        "alphas": ALPHAS,
        "gammas": GAMMAS,
        "total_runs": len(summary),
        "base_seed": base_seed,
        "model_source": "model.model.SimulationModel",
        "final_metrics": ["PI", "VPI", "SR", "ER", "EM", "HE", "EE"],
        "note": "与正式 Rule ABM 使用相同初始化、推荐算法和 metrics.metrics.compute_group_metrics；仅改变 N、T、R。",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 N=30, T=30, R=5 的 Rule ABM 受限规模实验")
    parser.add_argument("--output-dir", type=Path, default=None, help="默认写入 <项目根>/results/rule_small")
    parser.add_argument("--base-seed", type=int, default=conf.BASE_SEED)
    parser.add_argument("--force", action="store_true", help="即使已有完整小规模结果也重新运行")
    args = parser.parse_args()
    root = find_project_root(Path(__file__).resolve().parent)
    output_dir = (args.output_dir or root / "results" / "rule_small").resolve()
    summary = run_experiment(output_dir, args.base_seed, args.force)
    print(f"[Rule small] 完成 {len(summary)} runs，输出：{output_dir}")
    print(summary.groupby(["alpha", "gamma"]).size().to_string())


if __name__ == "__main__":
    main()
