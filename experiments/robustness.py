"""Robustness 实验（第二个 LLM 的小网格测试）。

按照文档要求对少量条件执行稳健性检验，产生 `results/robustness.csv`。
"""
import os
import random
import pandas as pd
from config import conf
from model.model import SimulationModel
from utils.results import candidate_run_paths, read_completed_run, resolve_result_dirs


def run_robustness_experiment():
    alphas = conf.LLM_ALPHAS
    gamma = 1.0
    repeats = 3
    results = []
    out_dir, run_dir = resolve_result_dirs("robustness")
    total_runs = len(alphas) * repeats
    completed_runs = 0
    run_id = 0
    for alpha in alphas:
        for r in range(repeats):
            filename = f"robustness_run_{run_id}.csv"
            out_path = os.path.join(run_dir, filename)
            existing_path, final = read_completed_run(
                candidate_run_paths(out_dir, run_dir, filename),
                expected_step=conf.LLM_STEPS,
                alpha=alpha,
                gamma=gamma,
            )
            if existing_path is not None and final is not None:
                final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma})
                results.append(final)
                run_id += 1
                completed_runs += 1
                print(f"[稳健性实验] 已复用已有结果 {completed_runs}/{total_runs} 次，剩余 {total_runs - completed_runs} 次", flush=True)
                continue

            seed = conf.BASE_SEED + run_id
            random.seed(seed)
            model = SimulationModel(seed=seed, N=conf.LLM_N_AGENTS, steps=conf.LLM_STEPS, alpha=alpha, gamma=gamma)
            model.run()
            if model.metrics_time_series:
                final = model.metrics_time_series[-1]
                final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma})
                results.append(final)
                pd.DataFrame([final]).to_csv(out_path, index=False)
            run_id += 1
            completed_runs += 1
            print(f"[稳健性实验] 已完成 {completed_runs}/{total_runs} 次，剩余 {total_runs - completed_runs} 次", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "robustness.csv"), index=False)
