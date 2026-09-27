"""D1/D2/D3/D4 修复后的 Rule 正式实验重跑（输出到独立目录，不覆盖论文现有数据）。

为什么要重跑
------------
D1 改变了 engagement 的生成方式（随机漂移 → 互动驱动累加 + 秩归一化）、
D2 给每个 agent 抽了独立的 λi、D4 让推荐候选集排除 agent 自己发的帖子，
三者都会改变 Rule 的演化轨迹，因此论文里现有的 Rule 数值
（`results/run_by_rule/`，由旧代码产出）与新代码已不可比。
旧数据**原样保留**用于审计，新结果写入 `--output-dir` 指定的目录。

与 `experiments/rule_experiment.py` 的关系
-----------------------------------------
完全不改原文件；只在本进程内替换 `resolve_result_dirs`，
把输出目录从 `results/run_by_rule/` 改为目标目录。
这样也顺便绕开了原脚本的"复用已完成 run"快路径
（它会直接复用旧代码产出的 CSV，导致新旧数据混在一起）。

规模与原始正式实验完全一致：N=200、T=200、5α×2γ、R=50 = 500 runs，
BASE_SEED 仍为 12345（run_id → seed 映射与原始实验一致，便于逐 run 对照）。

用法
----
    python run_rule_main_fixed.py                          # 写入 results/rule_main_fixed
    python run_rule_main_fixed.py --output-dir results/rule_main_d4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.rule_experiment as R

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "rule_main_fixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule 正式实验重跑（D1-D4 修复后）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="默认写入 <项目根>/results/rule_main_fixed")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)

    # 进程内替换输出目录解析；原文件不被修改。
    def resolve_fixed(kind: str):
        assert kind == "rule", f"本启动器只用于 rule 实验，收到 {kind!r}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir), str(output_dir / "逐run时间序列")

    R.resolve_result_dirs = resolve_fixed
    print(f"[rule-fixed] 输出目录：{output_dir}")
    print("[rule-fixed] N=200 / T=200 / 5 alpha × 2 gamma × R=50 = 500 runs")
    R.run_rule_experiment(debug=False)
    print(f"[rule-fixed] 完成，汇总：{output_dir / 'rule_summary.csv'}")


if __name__ == "__main__":
    main()
