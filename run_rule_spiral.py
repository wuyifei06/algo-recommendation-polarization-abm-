"""规则臂「沉默的螺旋」扩展实验（独立目录，不覆盖任何既有论文数据）。

背景
----
2026-09-21 的结构审计发现：规则臂的表达概率
``P = sigmoid(b0 + bE*emotion + bA*activity)``
里没有任何"气候感知 / 少数派 / 孤立恐惧"项，因此**无法表达**沉默的螺旋。
而 LLM 臂在离线重放检验中自发出现了跨立场沉默（控制 α 后 d 系数 −0.29***，
minority −0.035**）。本脚本把该机制以**理论驱动**的方式补进规则臂，
作为 baseline 之外的**扩展设定**，而不是替换 baseline。

模型（`model/dynamics.py::rule_expression_probability`）
-------------------------------------------------------
    P = sigmoid(b0 + bE*emotion + bA*activity)
        * (1 - c * d)            # d = Top-K 曝光的平均立场距离 ∈[0,1]
        * (1 - m * minority)     # minority = 自身立场与 feed 均值符号相反

    c = 0 且 m = 0 时与 baseline **逐位等价**（c=0 这一档就是本实验的自带对照）。

设计要点
--------
1. **种子配对**：seed 只由 (alpha, gamma, repeat) 决定，与 c 无关 →
   同一 seed 下不同 c 的 run 可做配对比较，方差大幅降低。
2. **独立目录 + 独立文件名前缀**，且 run_id 编码 c，不会与既有 rule_run_*.csv 冲突。
3. c 网格自带敏感性分析，直接回答"系数取多少才稳"。

用法
----
    python run_rule_spiral.py --quick                       # 冒烟（N=40/T=40）
    python run_rule_spiral.py --repeats 50                  # 正式：5c × 5α × 2γ × 50
    python run_rule_spiral.py --c-grid 0,0.6,1.5 --spiral-m 0.5
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import conf  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "rule_spiral"


def _run_once(task):
    """单个 run。参数全部显式传入，进程间不共享随机状态。"""
    (run_id, N, steps, alpha, gamma, spiral_c, spiral_m, seed, out_path, debug) = task
    from model.model import SimulationModel

    model = SimulationModel(
        seed=seed, N=N, steps=steps, alpha=alpha, gamma=gamma,
        spiral_c=spiral_c, spiral_m=spiral_m, debug=debug,
    )
    model.run()
    model.save_results(out_path)
    if not model.metrics_time_series:
        return None
    final = dict(model.metrics_time_series[-1])
    final.update({
        "run_id": run_id, "alpha": alpha, "gamma": gamma,
        "spiral_c": spiral_c, "spiral_m": spiral_m, "seed": seed,
        "N": N, "steps": steps,
    })
    return final


def _existing_ok(path: str, steps: int, spiral_c: float, spiral_m: float) -> bool:
    """判断已有文件能否复用（按有效行数 + 扩展参数校验，避免混入别的设定）。"""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) < steps:
        return False
    if "spiral_c" in df.columns and abs(float(df["spiral_c"].iloc[-1]) - float(spiral_c)) > 1e-12:
        return False
    if "spiral_m" in df.columns and abs(float(df["spiral_m"].iloc[-1]) - float(spiral_m)) > 1e-12:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="规则臂沉默的螺旋扩展实验")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--c-grid", type=str, default=None,
                    help="逗号分隔；默认取 config.SPIRAL_C_GRID（含 0 = baseline 对照）")
    ap.add_argument("--spiral-m", type=float, default=None)
    ap.add_argument("--workers", type=int, default=0, help="0 = 自动（min(4, cpu, 任务数)）")
    ap.add_argument("--quick", action="store_true", help="冒烟：N=40/T=40/R=2/单一 c")
    ap.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    args = ap.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "逐run时间序列"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        N, steps, repeats = 40, 40, 2
        c_grid = [0.0, 1.0]
    else:
        N, steps = conf.N_AGENTS, conf.STEPS
        repeats = args.repeats if args.repeats is not None else conf.REPEATS
        c_grid = ([float(x) for x in args.c_grid.split(",") if x.strip() != ""]
                  if args.c_grid else list(conf.SPIRAL_C_GRID))
    spiral_m = args.spiral_m if args.spiral_m is not None else float(conf.SPIRAL_M_DEFAULT)
    alphas = list(conf.ALPHAS)
    gammas = list(conf.GAMMAS)

    n_cell = len(alphas) * len(gammas) * repeats
    total_runs = len(c_grid) * n_cell
    print(f"[rule-spiral] 输出目录：{output_dir}")
    print(f"[rule-spiral] N={N} / T={steps} / {len(c_grid)}c × {len(alphas)}α × {len(gammas)}γ × R={repeats} = {total_runs} runs")
    print(f"[rule-spiral] c 网格 = {c_grid} ；m = {spiral_m}")
    print("[rule-spiral] 种子只由 (alpha,gamma,repeat) 决定 → 不同 c 之间配对", flush=True)

    results = []
    pending = []
    reused = 0
    for ci, c in enumerate(c_grid):
        for ai, alpha in enumerate(alphas):
            for gi, gamma in enumerate(gammas):
                for r in range(repeats):
                    # ⚠️ 配对种子：与 c 无关
                    base_idx = (ai * len(gammas) + gi) * repeats + r
                    seed = conf.BASE_SEED + base_idx
                    run_id = ci * n_cell + base_idx
                    out_path = str(run_dir / f"rule_spiral_run_{run_id}.csv")
                    if not args.force and _existing_ok(out_path, steps, c, spiral_m):
                        df = pd.read_csv(out_path)
                        final = dict(df.iloc[-1])
                        final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma,
                                      "spiral_c": c, "spiral_m": spiral_m, "seed": seed,
                                      "N": N, "steps": steps})
                        results.append(final)
                        reused += 1
                        continue
                    pending.append((run_id, N, steps, alpha, gamma, c, spiral_m, seed, out_path, False))

    if reused:
        print(f"[rule-spiral] 复用已有结果 {reused}/{total_runs}，待运行 {len(pending)}", flush=True)

    done = reused

    def collect(final):
        nonlocal done
        if final is not None:
            results.append(final)
        done += 1
        if done % 10 == 0 or done == total_runs:
            print(f"[rule-spiral] 已完成 {done}/{total_runs}", flush=True)

    workers = args.workers or min(4, os.cpu_count() or 1, max(1, len(pending)))
    if pending:
        if workers > 1:
            print(f"[rule-spiral] 使用 {workers} 个进程并行", flush=True)
            processed = 0
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for final in ex.map(_run_once, pending):
                        collect(final)
                        processed += 1
            except (PermissionError, OSError, BrokenProcessPool) as exc:
                print(f"[rule-spiral] 并行不可用（{type(exc).__name__}），回退串行", flush=True)
                for task in pending[processed:]:
                    collect(_run_once(task))
        else:
            for task in pending:
                try:
                    collect(_run_once(task))
                except Exception:
                    print(f"[rule-spiral] run {task[0]} 失败\n{traceback.format_exc()}", flush=True)

    df = pd.DataFrame(results).sort_values("run_id")
    summary_path = output_dir / "rule_spiral_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"[rule-spiral] 完成，汇总：{summary_path}（{len(df)} 行）")


if __name__ == "__main__":
    main()
