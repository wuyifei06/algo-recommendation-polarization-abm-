"""DeepSeek-only LLM 梯度测试启动器（加密 α 网格，定位倒 U 峰值）。

为什么要做梯度测试
------------------
现有 50-run 粗网格只在 α ∈ {0, 0.5, 1.0, 1.5, 2.0} 取点。而 DeepSeek 上
PI(α) 的两个候选峰值（拟合顶点 α≈0.39 与 α≈0.76）**都落在格点之间**：

  * α∈[0,0.5] 之间、[0.5,1.0] 之间完全无观测，峰值位置只能靠二次拟合外推；
  * 峰值幅度因此被低估，"净放大 +16.1%" 目前只有 p=0.043，勉强显著。

本启动器把 α 在曲率集中的 [0, 1.5] 区间按 0.25 步长加密，长尾（1.5→2.0）
保持稀疏（该段单调下降，已由粗网格确认），以提高：

  1. **峰值位置的分辨率**：从"5 点插值"变成"8 点直接观测"；
  2. **峰值幅度的检验力**：提高重复数 R 压低标准误。

设计原则（与 run_llm_robust_grid.py 一致）
------------------------------------------
1. **不修改 `experiment_llm_robust.py` 一个字符**，粗网格跑法保持原样可复现。
2. 只覆盖 `CONDITIONS` / `REPEATS` 两个模块常量；N=30 / T=30 / model / prompt /
   更新语义一律沿用原值，保证与粗网格、与 Rule 小规模实验可比。
3. 输出写到**独立目录** `results/llm_gradient_deepseek/`，避免与粗网格混合
   （run_id → 条件的映射随 REPEATS 变化，混跑会被续跑快路径误读）。
4. **基准种子默认 31345**（与粗网格的 22345 不同）：
   这样与粗网格共享的格点（α=0/0.5/1.0/1.5/2.0）是**独立重复**而非同种子复制，
   日后若要合并成 n=15 不会重复计数。
5. 只使用 DeepSeek（.env: LLM_PROVIDER=deepseek）。不涉及、不读取任何 Qwen 数据。

用法
----
    python run_llm_gradient_deepseek.py --plan        # 免费：打印设计与成本估算
    python run_llm_gradient_deepseek.py --self-test   # 免费：跑前字段/解析自检
    python run_llm_gradient_deepseek.py               # 正式跑（默认 R=10），可中断续跑
"""
from __future__ import annotations

import argparse
from pathlib import Path

import experiment_llm_robust as E

# 加密网格：曲率区 [0,1.5] 按 0.25 步长，长尾保留 2.0 作为下降段锚点。
# γ 仍为 {0.1, 1.0}，与粗网格 / Rule 小规模实验完全一致。
GRADIENT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
GRADIENT_GAMMAS = [0.1, 1.0]
GRADIENT = [(a, g) for a in GRADIENT_ALPHAS for g in GRADIENT_GAMMAS]

DEFAULT_REPEATS = 10
DEFAULT_BASE_SEED = 31345          # 与粗网格 22345 不同 → 共享格点为独立重复
DEFAULT_OUTPUT_DIR = E.PROJECT_ROOT / "results" / "llm_gradient_deepseek"

# 实测单价与耗时（来自 results/llm_robust_grid/ 的 50-run manifest 时间戳）
COST_PER_RUN_YUAN = 0.0702
MINUTES_PER_RUN = 2.2


def print_plan() -> None:
    runs = len(GRADIENT) * DEFAULT_REPEATS
    print("[gradient] DeepSeek-only LLM 梯度测试设计")
    print(f"[gradient]   alpha 网格（{len(GRADIENT_ALPHAS)} 点）: {GRADIENT_ALPHAS}")
    print(f"[gradient]   gamma 取值（{len(GRADIENT_GAMMAS)} 点）: {GRADIENT_GAMMAS}")
    print(f"[gradient]   条件数 = {len(GRADIENT)}，重复数 R = {DEFAULT_REPEATS}，共 {runs} runs")
    print(f"[gradient]   N = {E.N_AGENTS}，T = {E.STEPS}（与粗网格一致）")
    print(f"[gradient]   基准种子 = {DEFAULT_BASE_SEED}（与粗网格 22345 独立）")
    print(f"[gradient]   输出目录 = {DEFAULT_OUTPUT_DIR}")
    print(f"[gradient]   估算成本 ≈ ¥{runs * COST_PER_RUN_YUAN:.1f}；"
          f"估算耗时 ≈ {runs * MINUTES_PER_RUN / 60:.1f} h（约 {MINUTES_PER_RUN} min/run）")
    print("[gradient]   （成本/耗时按 results/llm_robust_grid 实测单价与时间戳外推）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek-only LLM ABM 梯度测试（加密 alpha 网格，定位倒 U 峰值）"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="默认写入 <项目根>/results/llm_gradient_deepseek")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"每个条件的重复数，默认 {DEFAULT_REPEATS}")
    parser.add_argument("--max-llm-calls", type=int, default=None,
                        help="可选调用上限（cost guard），不传则不限")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    parser.add_argument("--self-test", action="store_true", help="只运行跑前自检，不发起任何调用")
    parser.add_argument("--plan", action="store_true", help="只打印设计与成本估算，不运行")
    args = parser.parse_args()

    if args.plan:
        print_plan()
        return

    # 进程内覆盖，退出即失效，不影响原作者文件
    E.CONDITIONS = GRADIENT
    E.REPEATS = int(args.repeats)

    expected = len(GRADIENT) * int(args.repeats)
    print(f"[gradient] 条件数={len(GRADIENT)}，重复数={args.repeats}，共 {expected} runs；"
          f"输出：{args.output_dir}")
    print(f"[gradient] 估算成本 ≈ ¥{expected * COST_PER_RUN_YUAN:.1f}，"
          f"估算耗时 ≈ {expected * MINUTES_PER_RUN / 60:.1f} h")

    if args.self_test:
        E.self_test()
        print("[gradient] self-test-only 完成（未发起任何网络调用）。")
        return

    E.run_experiment(args)


if __name__ == "__main__":
    main()
