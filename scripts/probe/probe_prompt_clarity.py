# -*- coding: utf-8 -*-
"""提示词明确性探针：把输入字段写全名并声明取值范围，检验 LLM 臂的 PI 上行是否仍然存在。

要检验的假设
------------
原提示词只给 ``o / e / a / d / fe`` 五个单字母字段、且不声明量纲与含义，模型只能自行揣摩。
若模型因此"自己编一套理解"（人格化解读），则 PI 上行可能是该提示词缺陷的产物，
而非 LLM 主体的性质。

本探针只改**提示词里的字段说明**，不动任何其它东西：
  不动模型与采样参数、不动 N/T、不动更新语义、不动决策解析器、不动任何已有的实验代码。
  红线继续保持：alpha / gamma / top_k / social_pressure 一律不出现在提示词中
  （--self-test 会渲染带污染键的 state 并断言不泄漏）。

配对设计（关键）
----------------
原存档 ``results/llm_gradient_noleak`` 的种子为 ``seed = 31345 + run_id``，已逐条核实；
其中 (α=1.0, γ=0.1) 落在 run_id 80–89（seed 31425–31434）、
     (α=1.0, γ=1.0) 落在 run_id 90–99（seed 31435–31444），两组连续。
故本探针取 CONDITIONS=[(1.0,0.1),(1.0,1.0)]、REPEATS=10、base_seed=31425，
使探针 run 0–9 / 10–19 与存档**同种子**（共享初始条件，不共享轨迹——温度采样，
与论文在线消融的配对口径一致）。

同时跑一个"原提示词"小型同日对照（3 runs），用于排查模型别名的版本漂移
（``deepseek-flash`` 无固定快照标签，跨月对比需同日锚点）。

成本（按 ``run_llm_gradient_deepseek.py`` 的实测单价 ¥0.0702/run、2.2 min/run）
  labeled  20 runs ≈ ¥1.4
  original  3 runs ≈ ¥0.2
  合计     23 runs ≈ ¥1.6，约 50 分钟

用法
----
    python probe_prompt_clarity.py --plan                    # 免费
    python probe_prompt_clarity.py --self-test               # 免费（含红线断言）
    python probe_prompt_clarity.py --run --arm labeled       # 正式跑（必须显式 --run）
    python probe_prompt_clarity.py --run --arm original

结果写入本目录下 ``results_prompt_<arm>/``，与项目内任何既有结果目录隔离。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(_PROJ_ROOT)
sys.path.insert(0, str(PROJECT))

import experiment_llm_robust as E           # noqa: E402  仅调用其既有驱动，不修改其源码
import llm.prompt as P                      # noqa: E402
import llm.robust_client_v2 as RC           # noqa: E402
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 标签版提示词
# 只做两件事：① 加字段图例；② 把单字母字段名换成全名。其余一字不改。
LABELED_RULES = (
    "Return only a JSON array with one object per input id, in the same order. "
    "Each object must contain only express (0/1), stance_delta [-0.05,0.05], "
    "emotion_delta [-0.1,0.1]. Decide from the shown feed and your own state; "
    "you may stay silent or keep your stance unchanged when you judge it "
    "appropriate, and not everyone needs to express. "
    "No comments or explanations.\n"
    "Input fields (one object per agent, in the given order):\n"
    "- id: this agent's identifier.\n"
    "- opinion: this agent's own current stance on the issue, a continuous value "
    "in [-1, 1], where -1 and +1 are the two extreme stances and 0 is neutral.\n"
    "- emotion: this agent's own current emotional intensity, in [0, 1].\n"
    "- activity: this agent's own current level of activity, in [0, 1]; "
    "a higher value means the agent tends to be more active in the discussion.\n"
    "- feed_stance_distance: the mean absolute distance between this agent's own stance "
    "and the stances of the posts shown to it in its recommendation feed, in [0, 1]; "
    "a larger value means the shown feed differs more from the agent's own stance.\n"
    "- feed_emotion: the mean emotional intensity of the posts shown to it in its "
    "recommendation feed, in [0, 1]."
)


def labeled_build_batch_prompt(batch: list) -> str:
    """与 llm/prompt.py 的 build_batch_prompt 逐行同构，只换字段名与图例。"""
    compact = []
    for item in batch:
        state = item["state"]
        compact.append(
            {
                "id": item["agent_id"],
                "opinion": round(float(state["opinion"]), 4),
                "emotion": round(float(state["emotion"]), 4),
                "activity": round(float(state["activity"]), 4),
                "feed_stance_distance": round(float(state.get("mean_stance_distance", 0.0)), 4),
                "feed_emotion": round(float(state.get("mean_feed_emotion", 0.0)), 4),
            }
        )
    return LABELED_RULES + "\nINPUT=" + json.dumps(
        compact, ensure_ascii=False, separators=(",", ":")
    )


def install_labeled_prompt() -> None:
    """把标签版装进两处命名空间：模块本身，以及 client 已经 import 进去的名字。"""
    P.build_batch_prompt = labeled_build_batch_prompt
    RC.build_batch_prompt = labeled_build_batch_prompt


# ------------------------------------------------- 判定实验：补回「带符号的曝光方向」
# 诊断见 诊断_信息集不对称_20261008.md：
#   Rule 主体拿到 o_feed = Σp.opinion/K（带符号），而 LLM 主体只拿到 mean_stance_distance（无符号）。
#   本臂在原字段之外**只增加一个** feed_mean_opinion（推荐内容的平均立场，[-1,1]，带符号），
#   其余一切不动，用来判定"α 在 LLM 臂失效"是不是该信息缺失造成的。
#   红线合规：该字段是主体在情境下本可感知的内容量，不含 alpha/gamma/top_k/social_pressure。
DIRECTION_RULES = LABELED_RULES.replace(
    "- feed_stance_distance:",
    "- feed_mean_opinion: the mean stance of the posts shown to it in its recommendation "
    "feed, a continuous value in [-1, 1], where -1 and +1 are the two extreme stances "
    "and 0 is neutral.\n"
    "- feed_stance_distance:",
)
assert "feed_mean_opinion" in DIRECTION_RULES, "方向版提示词拼装失败"


def direction_build_batch_prompt(batch: list) -> str:
    """标签版 + 一个带符号的 feed_mean_opinion 字段。"""
    compact = []
    for item in batch:
        state = item["state"]
        compact.append(
            {
                "id": item["agent_id"],
                "opinion": round(float(state["opinion"]), 4),
                "emotion": round(float(state["emotion"]), 4),
                "activity": round(float(state["activity"]), 4),
                "feed_mean_opinion": round(float(state.get("mean_feed_opinion", 0.0)), 4),
                "feed_stance_distance": round(float(state.get("mean_stance_distance", 0.0)), 4),
                "feed_emotion": round(float(state.get("mean_feed_emotion", 0.0)), 4),
            }
        )
    return DIRECTION_RULES + "\nINPUT=" + json.dumps(
        compact, ensure_ascii=False, separators=(",", ":")
    )


def install_direction_prompt() -> None:
    """装「方向版」提示词，并给模型补上 mean_feed_opinion 这个特征。

    state 由 ``{**features}`` 展开而来，所以只要扩展 ``_feed_features()`` 的返回，
    该字段就会自动进入 state、进而进入提示词；不需要改项目里的任何文件。
    """
    import numpy as _np
    import model.llm_robustness_v2 as M

    if not getattr(M.LLMRobustnessModel, "_direction_patched", False):
        _orig = M.LLMRobustnessModel._feed_features

        def _feed_features_with_direction(self, agent, recs):   # noqa: ANN001
            feats = _orig(self, agent, recs)
            feats["mean_feed_opinion"] = (
                float(_np.mean([p.opinion for p in recs])) if recs else float(agent.opinion)
            )
            return feats

        M.LLMRobustnessModel._feed_features = _feed_features_with_direction
        M.LLMRobustnessModel._direction_patched = True

    P.build_batch_prompt = direction_build_batch_prompt
    RC.build_batch_prompt = direction_build_batch_prompt


# ---------------------------------------------------------------- 设计参数
PROBE_BASE_SEED = 31425                       # = 31345 + 80，对齐存档 (α=1.0, γ=0.1) 首 run
FULL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
FULL_GAMMAS = [0.1, 1.0]
ARM_CONDITIONS = {
    "labeled": [(1.0, 0.1), (1.0, 1.0)],      # 与存档同种子，逐一配对
    "original": [(1.0, 0.1)],                 # 同日漂移锚点，仅 γ=0.1
    # 完整主实验网格：与存档 160 runs 的 run_id→种子 完全相同（base_seed=31345）
    "full": [(a, g) for a in FULL_ALPHAS for g in FULL_GAMMAS],
}
ARM_REPEATS = {"labeled": 10, "original": 3, "full": 10, "direction": 10,
               "direction-rest": 10}
ARM_BASE_SEED = {"labeled": 31425, "original": 31425, "full": 31345,
                 "direction": None, "direction-rest": None}
# 「方向版」判定实验：三档 α 各跑一个区块，每块的 base_seed 都对齐存档的同种子 run_id
#   存档 seed = 31345 + run_id；(α,γ) 块在网格里连续：
#     α=0.0 → run_id 0–19  → base 31345
#     α=1.0 → run_id 80–99 → base 31425
#     α=2.0 → run_id 140–159 → base 31485
DIRECTION_BLOCKS = [(0.0, 31345, "a0.0"), (1.0, 31425, "a1.0"), (2.0, 31485, "a2.0")]
# 补跑区块：α=0.25/0.5/0.75/1.25/1.5（base 取该 α 的 γ=0.1 格；其 γ=1.0 格种子恰与前者连续，
# 已从存档 checkpoints 逐格核实：α=0.25→31365/31375、0.5→31385/31395、0.75→31405/31415、
# 1.25→31445/31455、1.5→31465/31475）
DIRECTION_REST = [(0.25, 31365, "a0.25"), (0.5, 31385, "a0.50"), (0.75, 31405, "a0.75"),
                  (1.25, 31445, "a1.25"), (1.5, 31465, "a1.50")]
COST_PER_RUN_YUAN = 0.0702
MINUTES_PER_RUN = 2.2


def print_plan(arm: str, repeats: int, base_seed) -> None:
    if arm == "direction-rest":
        runs = len(DIRECTION_REST) * 2 * repeats
        print("[probe] arm = direction-rest（补跑缺失的 5 档 α：0.25/0.5/0.75/1.25/1.5）")
        for a, base, tag in DIRECTION_REST:
            print(f"[probe]     α={a}: base_seed={base} → {HERE / 'results_prompt_direction' / tag}")
        print(f"[probe]   共 {runs} runs；N = {E.N_AGENTS}，T = {E.STEPS}")
        print(f"[probe]   估算成本 ≈ ¥{runs * COST_PER_RUN_YUAN:.2f}；"
              f"估算耗时 ≈ {runs * MINUTES_PER_RUN:.0f} min")
        _print_env()
        return
    if arm == "direction":
        runs = len(DIRECTION_BLOCKS) * 2 * repeats
        print(f"[probe] arm = direction（判定实验：标签版 + 带符号 feed_mean_opinion）")
        print(f"[probe]   α 档 = {[a for a, _, _ in DIRECTION_BLOCKS]}，γ 档 = [0.1, 1.0]，"
              f"重复数 = {repeats}")
        for a, base, tag in DIRECTION_BLOCKS:
            print(f"[probe]     区块 α={a}: base_seed={base}（种子 {base}–{base + 2*repeats - 1}）"
                  f" → {HERE / 'results_prompt_direction' / tag}")
        print(f"[probe]   共 {runs} runs；N = {E.N_AGENTS}，T = {E.STEPS}")
        print(f"[probe]   估算成本 ≈ ¥{runs * COST_PER_RUN_YUAN:.2f}；"
              f"估算耗时 ≈ {runs * MINUTES_PER_RUN:.0f} min")
        _print_env()
        return
    conds = ARM_CONDITIONS[arm]
    runs = len(conds) * repeats
    print(f"[probe] arm = {arm}")
    print(f"[probe]   提示词 = {'标签版（字段全名 + 取值范围）' if arm != 'original' else '原版（单字母 o/e/a/d/fe）'}")
    print(f"[probe]   条件数 = {len(conds)}，重复数 = {repeats}，共 {runs} runs")
    if len(conds) <= 4:
        print(f"[probe]   条件 = {conds}")
    print(f"[probe]   N = {E.N_AGENTS}，T = {E.STEPS}，base_seed = {base_seed}")
    print(f"[probe]   种子范围 = {base_seed} … {base_seed + runs - 1}")
    print(f"[probe]   输出目录 = {HERE / ('results_prompt_' + arm)}")
    print(f"[probe]   估算成本 ≈ ¥{runs * COST_PER_RUN_YUAN:.2f}；"
          f"估算耗时 ≈ {runs * MINUTES_PER_RUN:.0f} min")
    _print_env()


def _print_env() -> None:
    try:
        import config
        env = config.load_env()
        key = env.get("LLM_API_KEY") or ""
        print(f"[probe]   provider = {env.get('LLM_PROVIDER')}；model = {env.get('LLM_MODEL')}"
              f"；endpoint = {env.get('LLM_API_ENDPOINT')}")
        print(f"[probe]   temperature = {env.get('LLM_TEMPERATURE')}；"
              f"API key {'已配置' if key else '**未配置**'}（长度 {len(key)}，不显示内容）")
        missing = [k for k in ("LLM_API_KEY", "LLM_API_ENDPOINT", "LLM_MODEL") if not env.get(k)]
        if missing:
            print(f"[probe]   ⚠ 缺配置：{missing}——正式跑会直接报错，请先补 .env")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[probe]   （无法读取 .env：{exc}）")


def run_direction(args, repeats: int) -> None:
    """判定实验：三档 α 各跑一个区块，逐块对齐存档同种子。"""
    root = args.output_dir or (HERE / "results_prompt_direction")
    for alpha, base, tag in DIRECTION_BLOCKS:
        E.CONDITIONS = [(alpha, 0.1), (alpha, 1.0)]
        E.REPEATS = repeats
        sub = argparse.Namespace(**vars(args))
        sub.output_dir = Path(root) / tag
        sub.base_seed = base
        sub.repeats = repeats
        sub.force = False
        print(f"\n[probe] === 区块 α={alpha}　base_seed={base}　→ {sub.output_dir}")
        E.run_experiment(sub)
    print(f"\n[probe] 三个区块全部完成；汇总目录：{root}")


def run_direction_rest(args, repeats: int) -> None:
    """补跑缺失的 5 档 α（运行器不会跳过已完成的 run_id，故绝不能重跑已有格）。"""
    root = args.output_dir or (HERE / "results_prompt_direction")
    for alpha, base, tag in DIRECTION_REST:
        E.CONDITIONS = [(alpha, 0.1), (alpha, 1.0)]
        E.REPEATS = repeats
        sub = argparse.Namespace(**vars(args))
        sub.output_dir = Path(root) / tag
        sub.base_seed = base
        sub.repeats = repeats
        sub.force = False
        print(f"\n[probe] === 补跑区块 α={alpha}　base_seed={base}　→ {sub.output_dir}")
        E.run_experiment(sub)
    print(f"\n[probe] 补跑完成；完整 8 档 α 网格汇总于：{root}")


def main() -> None:
    ap = argparse.ArgumentParser(description="提示词明确性探针")
    ap.add_argument("--arm", choices=["labeled", "original", "full", "direction",
                                      "direction-rest"], default="labeled")
    ap.add_argument("--run", action="store_true", help="必须显式指定才会真正发起调用并花钱")
    ap.add_argument("--self-test", action="store_true", help="跑前自检（含红线断言），零调用")
    ap.add_argument("--plan", action="store_true", help="只打印设计与成本")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=None)
    ap.add_argument("--max-llm-calls", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    arm = args.arm
    repeats = int(args.repeats if args.repeats is not None else ARM_REPEATS[arm])

    if arm in ("direction", "direction-rest"):
        install_direction_prompt()
        print("[probe] 已装入「方向版」提示词：标签版 + feed_mean_opinion（带符号）")
        print("[probe] 已扩展 _feed_features() 返回 mean_feed_opinion（不改动项目文件）")
    elif arm != "original":
        install_labeled_prompt()
        print("[probe] 已装入标签版提示词（llm.prompt + llm.robust_client_v2 两处命名空间）")
    else:
        print("[probe] 使用原版提示词（不做任何 patch）")

    if args.self_test:
        demo_state = {"opinion": -1.0, "emotion": 0.8449, "activity": 0.6857,
                      "mean_feed_opinion": 0.4213,
                      "mean_stance_distance": 0.298, "mean_feed_emotion": 0.5023}
        builder = (direction_build_batch_prompt
                   if arm in ("direction", "direction-rest") else labeled_build_batch_prompt)
        print(f"\n--- {arm} 臂提示词渲染示例 ---")
        print(builder([{"agent_id": 0, "state": demo_state}]))
        print("--- 示例结束 ---\n")
        E.self_test()
        print("[probe] self-test 完成（未发起任何网络调用）。")
        return

    base_seed = args.base_seed if args.base_seed is not None else ARM_BASE_SEED.get(arm)

    if args.plan or not args.run:
        print_plan(arm, repeats, base_seed)
        if not args.run:
            print("[probe] 未指定 --run，仅打印设计（未发起任何调用）。")
        return

    print_plan(arm, repeats, base_seed)
    if arm == "direction":
        run_direction(args, repeats)
        return
    if arm == "direction-rest":
        run_direction_rest(args, repeats)
        return

    E.CONDITIONS = ARM_CONDITIONS[arm]
    E.REPEATS = repeats
    if args.output_dir is None:
        args.output_dir = HERE / f"results_prompt_{arm}"
    args.base_seed = int(base_seed)
    args.repeats = repeats
    E.run_experiment(args)


if __name__ == "__main__":
    main()
