"""主入口：支持多种运行模式。

本脚本解析命令行参数并分发到不同的实验驱动函数：
- debug: 小规模快速验证
- rule: 正式规则 ABM 实验
- llm: LLM 驱动的实验
- robustness: LLM 稳健性实验
- all: 依次运行以上所有实验

注意：任何使用 LLM 的入口都必须先验证真实大模型配置，
不能回退到本地 stub。
"""
import argparse
from config import conf, validate_llm_config
from experiments.rule_experiment import run_rule_experiment
from experiments.llm_experiment import run_llm_experiment
from experiments.robustness import run_robustness_experiment
from experiments.llm_robustness_v2 import run_experiment as run_llm_robustness_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "debug",
            "rule",
            "llm01",
            "llm02",
            "llm",  # legacy alias for llm01
            "robustness",
            "llm_robustness_v2",  # legacy alias for llm02
            "all",
        ],
        default="debug",
        help="llm01=原LLM实验，llm02=新版Batch稳健性实验；llm和llm_robustness_v2为兼容别名",
    )
    parser.add_argument(
        "--llm-run-id",
        type=int,
        default=None,
        help="只补跑指定的旧版 LLM run；当前缺失的第六次实验使用 --llm-run-id 6",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="for --mode llm_robustness_v2, run the separate small smoke experiment",
    )
    parser.add_argument("--trial", type=int, choices=[1, 2, 3], help="run requested LLM trial stage")
    args = parser.parse_args()

    mode = args.mode
    if mode == "debug":
        run_rule_experiment(debug=True)
    elif mode == "rule":
        run_rule_experiment(debug=False)
    elif mode in {"llm01", "llm"}:
        validate_llm_config()
        run_llm_experiment(run_ids=[args.llm_run_id] if args.llm_run_id is not None else None)
    elif mode == "robustness":
        validate_llm_config()
        run_robustness_experiment()
    elif mode in {"llm02", "llm_robustness_v2"}:
        run_llm_robustness_v2(smoke=args.smoke, trial=args.trial)
    elif mode == "all":
        run_rule_experiment(debug=False)
        validate_llm_config()
        run_llm_experiment(run_ids=[args.llm_run_id] if args.llm_run_id is not None else None)
        run_robustness_experiment()


if __name__ == "__main__":
    main()
