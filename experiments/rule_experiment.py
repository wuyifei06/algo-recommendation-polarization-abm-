"""Rule ABM 实验驱动。

负责按照 config 中的参数网格（alpha × gamma × repeats）运行仿真，
并将每次 run 的时间序列与汇总结果保存到 `results/` 目录。
在正式运行前请使用 `--mode debug` 进行小规模验证。
"""
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from config import conf
from model.model import SimulationModel
import pandas as pd
from utils.results import candidate_run_paths, read_completed_run, resolve_result_dirs


def _run_rule_once(task):
    """运行一个彼此独立的规则实验任务。

    每个任务显式使用自己的 seed，因此放到独立进程后不会共享随机状态，
    也不会改变单个 run 的实验内容。
    """
    run_id, N, steps, alpha, gamma, debug, out_path, seed = task
    model = SimulationModel(seed=seed, N=N, steps=steps, alpha=alpha, gamma=gamma, debug=debug)
    model.run()
    model.save_results(out_path)
    if not model.metrics_time_series:
        return None
    final = dict(model.metrics_time_series[-1])
    final["run_id"] = run_id
    return final


def _rule_worker_count(pending_count: int, debug: bool) -> int:
    """确定规则实验并行度；可用 RULE_WORKERS 覆盖默认值。"""
    if debug or pending_count <= 1:
        return 1
    configured = os.getenv("RULE_WORKERS")
    if configured:
        try:
            return max(1, min(int(configured), pending_count))
        except ValueError:
            pass
    # 规则 run 彼此独立，默认使用适度的进程并行，避免占满机器。
    return min(4, os.cpu_count() or 1, pending_count)


def run_rule_experiment(debug: bool = False):
    # debug smaller settings
    if debug:
        N = 20
        steps = 20
        repeats = 2
        alphas = [0.5]
        gammas = [0.1]
    else:
        N = conf.N_AGENTS
        steps = conf.STEPS
        repeats = conf.REPEATS
        alphas = conf.ALPHAS
        gammas = conf.GAMMAS

    results = []
    summary_dir, run_dir = resolve_result_dirs("rule")
    # debug 运行永远写入独立目录，避免覆盖正式实验的 rule_run_0.csv。
    if debug:
        run_dir = os.path.join(summary_dir, "debug")
        os.makedirs(run_dir, exist_ok=True)
    out_dir = run_dir if debug else summary_dir
    total_runs = len(alphas) * len(gammas) * repeats
    completed_runs = 0
    run_id = 0
    pending = []
    print(f"[规则实验] 共 {total_runs} 次，开始运行", flush=True)
    for alpha in alphas:
        for gamma in gammas:
            for r in range(repeats):
                filename = f"rule_run_{run_id}.csv"
                out_path = os.path.join(run_dir, filename)
                if not debug:
                    existing_path, final = read_completed_run(
                        candidate_run_paths(summary_dir, run_dir, filename),
                        expected_step=steps,
                        alpha=alpha,
                        gamma=gamma,
                    )
                    if existing_path is not None and final is not None:
                        # 以当前网格为准写入标识，避免旧文件中的 run_id
                        # 与本次参数网格不一致时污染汇总。
                        final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma})
                        results.append(final)
                    completed_runs += 1
                    run_id += 1
                    if existing_path is not None:
                        print(f"[规则实验] 已复用已有结果 {completed_runs}/{total_runs} 次，剩余 {total_runs - completed_runs} 次", flush=True)
                        continue
                    # 文件不存在、为空或只运行到中途时，回退 completed_runs
                    # 并把这个条件加入待运行队列。
                    completed_runs -= 1
                    run_id -= 1

                seed = conf.BASE_SEED + run_id
                pending.append(
                    (
                        run_id,
                        N,
                        steps,
                        alpha,
                        gamma,
                        debug,
                        out_path,
                        seed,
                    )
                )
                run_id += 1

    workers = _rule_worker_count(len(pending), debug)
    if workers > 1:
        print(f"[规则实验] 剩余 {len(pending)} 次，使用 {workers} 个进程并行运行", flush=True)

    def collect(final):
        nonlocal completed_runs
        if final is not None:
            results.append(final)
        completed_runs += 1
        print(f"[规则实验] 已完成 {completed_runs}/{total_runs} 次，剩余 {total_runs - completed_runs} 次", flush=True)

    if workers > 1:
        processed = 0
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for final in executor.map(_run_rule_once, pending):
                    collect(final)
                    processed += 1
        except (PermissionError, OSError, BrokenProcessPool) as exc:
            # 某些受限环境禁止创建子进程；已经完成的任务保留，
            # 其余任务回退到串行，保证规则实验入口仍可用。
            print(f"[规则实验] 并行不可用（{type(exc).__name__}），回退到串行", flush=True)
            for task in pending[processed:]:
                run_id_i, _, _, alpha, gamma, _, _, _ = task
                print(
                    f"[规则实验] 正在进行第 {completed_runs + 1}/{total_runs} 次，"
                    f"剩余 {total_runs - completed_runs} 次，alpha={alpha}, gamma={gamma}",
                    flush=True,
                )
                collect(_run_rule_once(task))
    else:
        for task in pending:
            run_id_i, _, _, alpha, gamma, _, _, _ = task
            print(
                f"[规则实验] 正在进行第 {completed_runs + 1}/{total_runs} 次，"
                f"剩余 {total_runs - completed_runs} 次，alpha={alpha}, gamma={gamma}",
                flush=True,
            )
            collect(_run_rule_once(task))

    # 并行完成顺序可能与 run_id 不同，按原实验顺序写汇总文件。
    df = pd.DataFrame(sorted(results, key=lambda row: row.get("run_id", -1)))
    df.to_csv(os.path.join(out_dir, "rule_summary.csv"), index=False)
