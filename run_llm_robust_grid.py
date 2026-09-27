"""DeepSeek-V4.1-Flash 稳健性实验：50-run 全网格启动器。

背景
----
原先的 `experiment_llm_robust.py` 跑的是「3 条件 × 3 重复 = 9 runs」。
本启动器把条件网格与重复数在**本进程内**覆盖为与 Qwen 主实验完全同构的
「10 条件 × 5 重复 = 50 runs」，以便做逐格跨模型对照。

设计原则（刻意为之）
--------------------
1. **不修改 `experiment_llm_robust.py` 一个字符**，原 9-run 跑法保持原样可复现。
2. 只覆盖两个模块常量：`CONDITIONS`、`REPEATS`。
   其余（N=30 / T=30 / model / prompt / 更新语义）一律沿用原值，保证可比性。
3. 输出写到独立目录 `results/llm_robust_grid/`，
   避免与既有 `results/llm_robust/` 的 9-run 结果混合
   —— 因为 run_id → 条件的映射随 REPEATS 变化，混跑会被续跑快路径误读。

用法
----
    python run_llm_robust_grid.py --self-test      # 免费自检，不发起任何调用
    python run_llm_robust_grid.py                  # 正式跑 50 runs，写入 results/llm_robust_grid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import experiment_llm_robust as E

# 与 Qwen 主实验 results/run_by_llm/llm_robustness_v2/summary_all_runs.csv 完全一致的网格与顺序
GRID = [
    (0.0, 0.1), (0.0, 1.0),
    (0.5, 0.1), (0.5, 1.0),
    (1.0, 0.1), (1.0, 1.0),
    (1.5, 0.1), (1.5, 1.0),
    (2.0, 0.1), (2.0, 1.0),
]
GRID_REPEATS = 5
DEFAULT_OUTPUT_DIR = E.PROJECT_ROOT / "results" / "llm_robust_grid"


def main() -> None:
    # 进程内覆盖，退出即失效，不影响原作者文件
    E.CONDITIONS = GRID
    E.REPEATS = GRID_REPEATS

    parser = argparse.ArgumentParser(
        description="DeepSeek LLM ABM 稳健性实验（v2 栈，10 条件 × R=5，共 50 runs）"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="默认写入 <项目根>/results/llm_robust_grid")
    parser.add_argument("--base-seed", type=int, default=22345)
    parser.add_argument("--max-llm-calls", type=int, default=None,
                        help="可选调用上限（cost guard），不传则不限")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    parser.add_argument("--self-test", action="store_true", help="只运行跑前自检，不发起任何调用")
    args = parser.parse_args()

    expected = len(GRID) * GRID_REPEATS
    print(f"[grid] 条件数={len(GRID)}，重复数={GRID_REPEATS}，共 {expected} runs；输出：{args.output_dir}")

    if args.self_test:
        E.self_test()
        print("[grid] self-test-only 完成（未发起任何网络调用）。")
        return

    E.run_experiment(args)


if __name__ == "__main__":
    main()
