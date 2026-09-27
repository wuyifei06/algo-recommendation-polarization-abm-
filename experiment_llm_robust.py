#!/usr/bin/env python
"""DeepSeek-V4.1-Flash LLM ABM 三条件稳健性实验（v2 栈）。

实验设计（固定，不可通过参数修改）：
    条件：(alpha=0, gamma=0.1)、(alpha=1, gamma=1.0)、(alpha=2, gamma=1.0)
    每条件 R=3，N=30、T=30，共 9 runs，输出至 results/llm_robust/。

输出（与 ``analysis/full_analysis.py``、``analysis/compare_llm_robust.py`` 对接）：
    results/llm_robust/llm_robust_summary.csv      每个 run 的最终指标 + 用量
    results/llm_robust/runs/llm_robust_run_<i>.csv 每个 run 的 30 步时间序列
    results/llm_robust/calls/llm_robust_run_<i>.jsonl  逐请求存档（含解析结果）
    results/llm_robust/manifest.json               运行元数据与质量统计

⚠️ 为什么不再复用 legacy 路径（2026-09-20 修复）
------------------------------------------------------------------
本脚本原先复用 ``model.model.SimulationModel(use_llm=True)`` +
``llm.client.LLMClient``。``llm/prompt.py::build_prompt`` 早已改为要求模型
返回 ``{express, stance_delta, emotion_delta}``（**相对增量**），但该 legacy
路径仍按**绝对量** ``stance`` / ``emotion`` 解析：

  1. ``llm/client.py::parse_json_object`` 要求三键齐全 → 缺 ``stance`` → 失败；
  2. 退到 ``parse_partial_json_fragment`` → 只匹配到 ``express``，
     把缺失的 ``stance`` / ``emotion`` **静默填成 0.0**，不抛错；
  3. ``model/model.py::_step_llm`` 随后执行
     ``agent.opinion = clip(decision["stance"])``（绝对值赋值）
     → 第 1 步全体 agent 立场被写成 0.0 并永久锁死。

后果：PI / VPI / ER 恒为 0、HE 恒为 1.0，看起来像"模型导致完全趋同"，
实际是解析器默认值的产物（存档实测 8,121 条调用 100% ``stance == 0.0``）。
该批数据已作废，必须重跑。

因此本脚本改用与主实验（Qwen）**完全同构**的 v2 栈：
    experiments/llm_robustness_v2.py  →  model/llm_robustness_v2.py
                                      →  llm/robust_client_v2.py
即同一 Prompt、同一批量协议（10 agent / 请求）、同一兜底策略、同一增量更新
语义（``opinion += stance_delta``）。

三道防线（防止同类问题再次静默通过）：
  * 跑前 ``self_test()``：断言 prompt 要求的字段就是所用解析器能还原的字段；
  * 跑后 ``assert_not_degenerate()``：检测"所有决策取值完全相同""出现 legacy
    绝对量字段""无任何有效响应"等退化指纹，命中直接报错而不是写出假结果；
  * 跑后 ``report_stratified_health()``：按 alpha×gamma 分层输出表达率与
    "整步全沉默率"，写入 ``results/llm_robust/llm_robust_health_by_condition.csv``。
    这是行为侧健康指标——字段齐全、解析成功也可能"整步全沉默"，属于模型行为
    而非 bug，但比例过高会把 alpha 效应压平成地板效应，必须留痕并在论文中如实
    报告（主实验 Qwen 与 DeepSeek 在这一点上存在明显跨模型差异）。

⚠️ 关于推理型模型的 token 预算（2026-09-20 修复）
------------------------------------------------------------------
DeepSeek 的推理型模型不认 ``enable_thinking`` / ``thinking_budget`` /
``reasoning_effort``，``max_tokens`` 会被 ``reasoning_tokens`` 全部吃光 →
``content=""``、``finish_reason="length"`` → 全部决策走兜底，最终抛
``RuntimeError: 没有任何有效模型响应``。实测 1024 / 4096 / 8192 全部如此，
**抬高 max_tokens 不是修复**。真正的开关是 provider 原生字段
``thinking={"type": "disabled"}``，已在 ``llm/robust_client_v2.py``
的 ``_thinking_off_payload()`` 中实现（默认只对 deepseek 生效，可用
``LLM_DISABLE_THINKING=0/1`` 覆盖），并在首次命中时打印告警。

本脚本只做真实调用：缺少 API Key 时直接报错退出，不会生成伪造结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

CONDITIONS = [(0.0, 0.1), (1.0, 1.0), (2.0, 1.0)]
REPEATS = 3
N_AGENTS = 30
STEPS = 30
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_ENDPOINT = "https://api.deepseek.com/v1"

EXPECTED_KEYS = {"express", "stance_delta", "emotion_delta"}
LEGACY_KEYS = {"express", "stance", "emotion"}


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "config.py").exists() and (candidate / "model").is_dir():
            return candidate
    raise FileNotFoundError("无法定位包含 config.py 和 model/ 的项目根目录")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------
# 跑前自检：确保"prompt 要求的字段" == "解析器能还原的字段"
# --------------------------------------------------------------------------
def self_test() -> None:
    """回归自检（不发起任何网络请求）。

    这条断言的存在意义：历史上 prompt 改成 ``stance_delta`` / ``emotion_delta``
    后，legacy 解析器仍按 ``stance`` / ``emotion`` 解析并静默填 0，导致整批数据
    退化为 PI ≡ 0。self_test 会在跑实验之前把这类"字段不匹配"直接暴露出来。
    """
    from config import load_env  # noqa: F401  (.env 在 import config 时已加载)
    from llm.prompt import build_batch_prompt, build_prompt
    from llm.robust_client_v2 import ResponseValidationError, RobustLLMClient
    from model.llm_robustness_v2 import LLMRobustnessModel
    from model.model import SimulationModel

    # 1) prompt 必须要求增量字段
    state = {
        "opinion": 0.30, "emotion": 0.40, "activity": 0.50, "agent_type": "moderate",
        "top_k": 10,
        "mean_stance_distance": 0.20, "mean_feed_emotion": 0.50,
    }
    items = [{"opinion": 0.10, "emotion": 0.60, "engagement": 0.50, "created_step": 0}]
    prompt = build_prompt(state, items)
    for field in ("stance_delta", "emotion_delta", "express"):
        if field not in prompt:
            raise AssertionError(f"prompt 未要求字段 {field}；prompt 与解析器可能不匹配。")
    if "stance_delta" not in prompt:
        raise AssertionError("prompt 要求的是绝对量 stance，与 v2 增量语义不一致。")

    # 1b) 自变量泄漏防线（2026-09-21）
    #     算法参数 alpha / gamma 及其派生量 social_pressure 一律不得出现在 prompt 文本里。
    #     这里故意把被污染的键塞回 state：即使上游手滑传了，渲染结果也必须干净。
    leak_state = dict(state)
    leak_state.update({"alpha": 1.0, "gamma": 1.0, "social_pressure": 0.30})
    leak_batch = [{"agent_id": 1, "state": leak_state}]
    for name, rendered in (
        ("build_prompt", build_prompt(leak_state, items)),
        ("build_batch_prompt", build_batch_prompt(leak_batch)),
    ):
        low = rendered.lower()
        for bad in ("alpha", "gamma", "social_pressure"):
            if bad in low:
                raise AssertionError(
                    f"{name} 泄漏自变量：渲染出的 prompt 含 {bad!r}。"
                    f"算法参数不得进入 prompt —— 详见 llm/prompt.py 顶部设计红线与 "
                    f"D4_同质推荐角点_改动验证与机制发现_20260920.md §10。"
                )

    # 2) 本脚本实际使用的解析器必须能把该格式还原成非零增量
    cases = [
        {"express": 1, "stance_delta": 0.03, "emotion_delta": -0.04},
        {"choices": [{"message": {"content": '{"express": 1, "stance_delta": 0.03, "emotion_delta": -0.04}'}}]},
    ]
    for case in cases:
        action, quality = RobustLLMClient.normalize_action(case, 0.5)
        if abs(float(action["stance_delta"]) - 0.03) > 1e-9:
            raise AssertionError(f"解析器未能还原 stance_delta：{action}")
        if abs(float(action["emotion_delta"]) - (-0.04)) > 1e-9:
            raise AssertionError(f"解析器未能还原 emotion_delta：{action}")
        if int(action["express"]) != 1:
            raise AssertionError(f"解析器未能还原 express：{action}")

    # 3) 越界必须抛错（而不是被静默裁剪成默认值）
    try:
        RobustLLMClient.normalize_action({"express": 1, "stance_delta": 0.9, "emotion_delta": 0.0}, 0.5)
    except ResponseValidationError:
        pass
    else:
        raise AssertionError("越界的 stance_delta 未被拒绝，解析器可能仍在静默填默认值。")

    # 4) 所用的模型必须覆盖 _step_llm（即真的是增量语义的 v2 实现）
    if LLMRobustnessModel._step_llm is SimulationModel._step_llm:
        raise AssertionError("LLMRobustnessModel 未覆盖 _step_llm，会退回 legacy 绝对量赋值。")

    print("[LLM robust] self_test 通过：prompt 字段与解析器语义一致，模型为 v2 增量实现。", flush=True)


# --------------------------------------------------------------------------
# 跑后体检：识别"解析器退化"指纹
# --------------------------------------------------------------------------
def assert_not_degenerate(calls_path: Path, quality: Dict) -> Dict:
    """扫描一个 run 的调用存档，命中退化指纹即报错。

    退化指纹（历史事故的确切特征）：
      * 出现 legacy 绝对量字段（express/stance/emotion）→ 说明又走回 legacy 路径；
      * 所有决策的 stance_delta 完全相同（尤其全为 0.0）→ 解析器默认值产物；
      * 没有任何有效响应（全是兜底）→ 该 run 不属于真实模型结果。
    """
    entries = 0
    valid = 0
    express_yes = 0
    keysets: Dict[str, int] = {}
    stance_values: Dict[str, int] = {}
    step_total: Dict[int, int] = {}
    step_express: Dict[int, int] = {}
    if calls_path.exists():
        with calls_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries += 1
                step = item.get("step")
                result = item.get("result")
                actions = result if isinstance(result, list) else [result]
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    valid += 1
                    expressed = 1 if int(action.get("express") or 0) == 1 else 0
                    express_yes += expressed
                    if step is not None:
                        step_total[int(step)] = step_total.get(int(step), 0) + 1
                        if expressed:
                            step_express[int(step)] = step_express.get(int(step), 0) + 1
                    key = ",".join(sorted(action.keys()))
                    keysets[key] = keysets.get(key, 0) + 1
                    if "stance_delta" in action:
                        stance_values[repr(round(float(action["stance_delta"]), 6))] = (
                            stance_values.get(repr(round(float(action["stance_delta"]), 6)), 0) + 1
                        )

    # 整步全沉默：该时间步内**所有** agent 的 express 都是 0。
    # 这不是 bug（字段齐全、解析成功），而是模型行为；但比例过高会把 alpha
    # 效应压平成地板效应，因此必须显式记录下来、随 summary 一起留痕。
    silent_steps = sum(1 for s, total in step_total.items() if total > 0 and step_express.get(s, 0) == 0)

    report = {
        "archive_entries": entries,
        "parsed_actions": valid,
        "keysets": keysets,
        "distinct_stance_delta": len(stance_values),
        "valid_responses": int(quality.get("valid_responses", 0)),
        "fallbacks": int(quality.get("fallbacks", 0)),
        "fallback_rate": float(quality.get("fallback_rate", 0.0)),
        "parse_failures": int(quality.get("parse_failures", 0)),
        "reasoning_starved": int(quality.get("reasoning_starved", 0)),
        "express_rate": (express_yes / valid) if valid else 0.0,
        "steps_observed": len(step_total),
        "silent_steps": silent_steps,
        "silent_step_rate": (silent_steps / len(step_total)) if step_total else 0.0,
    }

    if valid == 0:
        raise RuntimeError(f"存档里没有任何解析结果：{calls_path}")
    if report["valid_responses"] == 0:
        raise RuntimeError("没有任何有效模型响应（全部走了兜底），该 run 不计入正式结果。")

    for keys in keysets:
        keyset = set(keys.split(","))
        if keyset == LEGACY_KEYS or ("stance" in keyset and "stance_delta" not in keyset):
            raise RuntimeError(
                "检测到 legacy 绝对量字段 "
                f"({keys})：解析器与 prompt 的字段格式不匹配，"
                "请检查是否被改回 llm/client.py 这条 legacy 路径。"
            )
        if not EXPECTED_KEYS.issubset(keyset):
            raise RuntimeError(f"解析结果字段异常：{keys}，期望包含 {sorted(EXPECTED_KEYS)}")

    if report["distinct_stance_delta"] <= 1:
        raise RuntimeError(
            "所有决策的 stance_delta 取值完全相同（疑似解析器默认值退化，"
            "历史上表现为 PI/VPI/ER 恒为 0）。请检查解析链路后再重跑。"
        )

    return report


def report_stratified_health(summary: pd.DataFrame, output_dir: Path) -> None:
    """按 alpha×gamma 打印分层健康报告，并把结果写成 CSV 留痕。

    为什么必须分层看：整体兜底率正常，也可能存在"高 alpha 条件下整步全沉默"
    这类被平均值掩盖的地板效应——DeepSeek 在 alpha=2 时的全沉默比例明显高于
    主实验 Qwen。分层后这种差异会立刻显形，便于在论文里如实报告跨模型差异，
    而不是等复现失败时才回头排查。
    """
    required = {"alpha", "gamma", "llm_express_rate", "llm_silent_step_rate"}
    missing = required - set(summary.columns)
    if missing:
        print(
            f"[LLM robust] 分层健康报告跳过：summary 缺少列 {sorted(missing)}"
            "（可能是复用了旧版 summary）。",
            flush=True,
        )
        return

    grouped = (
        summary.groupby(["alpha", "gamma"], dropna=False)
        .agg(
            runs=("run_id", "count"),
            express_rate=("llm_express_rate", "mean"),
            silent_step_rate=("llm_silent_step_rate", "mean"),
            fallback_rate=("llm_fallback_rate", "mean"),
        )
        .reset_index()
        .sort_values(["alpha", "gamma"])
    )
    print("[LLM robust] 分层健康报告（按 alpha×gamma）", flush=True)
    for _, row in grouped.iterrows():
        print(
            f"[LLM robust]   alpha={row['alpha']}, gamma={row['gamma']}: "
            f"runs={int(row['runs'])} 表达率={row['express_rate']:.3f} "
            f"整步全沉默率={row['silent_step_rate']:.3f} "
            f"兜底率={row['fallback_rate']:.2%}",
            flush=True,
        )
    health_path = output_dir / "llm_robust_health_by_condition.csv"
    grouped.to_csv(health_path, index=False)
    print(f"[LLM robust] 分层健康报告已写入：{health_path}", flush=True)


# --------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------
def resolve_llm_settings() -> Dict[str, str]:
    """v2 客户端只从 .env / 环境变量读取配置，这里只做回声与校验。"""
    from config import load_env, validate_llm_config

    validate_llm_config()
    env = load_env()
    return {
        "provider": env["LLM_PROVIDER"],
        "model": env["LLM_MODEL"],
        "endpoint": env["LLM_API_ENDPOINT"],
        "key_tail": (env["LLM_API_KEY"] or "")[-4:],
    }


def iter_specs(base_seed: int) -> Iterable[Dict]:
    run_id = 0
    for alpha, gamma in CONDITIONS:
        for repeat in range(REPEATS):
            yield {
                "run_id": run_id,
                "repeat": repeat,
                "alpha": float(alpha),
                "gamma": float(gamma),
                "seed": int(base_seed + run_id),
            }
            run_id += 1


def build_model(spec: Dict, calls_path: Path, max_llm_calls: int):
    from model.llm_robustness_v2 import LLMRobustnessModel

    return LLMRobustnessModel(
        seed=spec["seed"],
        N=N_AGENTS,
        steps=STEPS,
        alpha=spec["alpha"],
        gamma=spec["gamma"],
        use_llm=True,
        llm_archive_path=str(calls_path),
        max_llm_calls=max_llm_calls,
        run_id=spec["run_id"],
    )


def load_or_resume(spec: Dict, checkpoint_path: Path, calls_path: Path, max_llm_calls: int):
    """优先从 run 内 checkpoint 恢复；参数不一致则从头重跑。"""
    from model.llm_robustness_v2 import LLMRobustnessModel

    if checkpoint_path.exists():
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            matches = (
                int(payload.get("seed", -1)) == spec["seed"]
                and int(payload.get("N", -1)) == N_AGENTS
                and int(payload.get("steps", -1)) == STEPS
                and abs(float(payload.get("alpha")) - spec["alpha"]) <= 1e-12
                and abs(float(payload.get("gamma")) - spec["gamma"]) <= 1e-12
                and int(payload.get("current_step", STEPS)) < STEPS
            )
            if matches:
                model = LLMRobustnessModel.load_checkpoint(
                    str(checkpoint_path), use_llm=True, llm_archive_path=str(calls_path)
                )
                model.run_id = spec["run_id"]
                model.llm_client.run_id = spec["run_id"]
                if max_llm_calls:
                    model.llm_client.max_calls = int(max_llm_calls)
                    model.llm_client.cost_guard.max_calls = int(max_llm_calls)
                print(
                    f"[LLM robust] run {spec['run_id']} 从 checkpoint 恢复："
                    f"已完成 step {model.current_step}/{STEPS}。",
                    flush=True,
                )
                return model, "resumed_checkpoint"
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"[LLM robust] run {spec['run_id']} checkpoint 不可用（{exc!r}），从头重跑。", flush=True)

    return build_model(spec, calls_path, max_llm_calls), "generated"


def save_agent_snapshot(model, path: Path) -> None:
    rows = []
    for agent in model.agents:
        rows.append({
            "agent_id": agent.id,
            "opinion": agent.opinion,
            "emotion": agent.emotion,
            "agent_type": agent.agent_type,
            "activity": agent.activity,
            "expression_probability": agent.expression_probability,
            "alpha": model.alpha,
            "gamma": model.gamma,
            "seed": model.seed,
            "step": model.current_step,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def run_experiment(args) -> pd.DataFrame:
    self_test()
    settings = resolve_llm_settings()
    print(
        f"[LLM robust] provider={settings['provider']} model={settings['model']} "
        f"endpoint={settings['endpoint']} key=***{settings['key_tail']}",
        flush=True,
    )
    max_llm_calls = int(args.max_llm_calls or 0)

    output_dir = (args.output_dir or PROJECT_ROOT / "results" / "llm_robust").resolve()
    runs_dir = output_dir / "runs"
    checkpoints_dir = output_dir / "checkpoints"
    calls_dir = output_dir / "calls"
    for directory in (output_dir, runs_dir, checkpoints_dir, calls_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "llm_robust_summary.csv"

    specs = list(iter_specs(args.base_seed))
    expected_runs = len(specs)

    # 断点续跑：summary 中最后一个 step 达标且状态正常的 run 直接复用。
    completed: Dict[int, Dict] = {}
    if summary_path.exists() and not args.force:
        existing = pd.read_csv(summary_path)
        for _, row in existing.iterrows():
            status = str(row.get("status", "completed"))
            if int(row.get("step", -1)) == STEPS and status in {"completed", "completed_with_fallback"}:
                completed[int(row["run_id"])] = row.to_dict()
        if completed:
            print(f"[LLM robust] 检测到 {len(completed)} 个已完成 run，将只补跑缺失部分。", flush=True)
        if len(completed) == expected_runs:
            print(f"[LLM robust] 已存在完整结果，跳过重跑：{summary_path}")
            report_stratified_health(existing, output_dir)
            return existing

    manifest = {
        "experiment": "llm_robust",
        "stack": "model/llm_robustness_v2.py + llm/robust_client_v2.py",
        "provider": settings["provider"],
        "model": settings["model"],
        "endpoint": settings["endpoint"],
        "N": N_AGENTS,
        "T": STEPS,
        "R": REPEATS,
        "conditions": [{"alpha": a, "gamma": g} for a, g in CONDITIONS],
        "temperature_source": "RobustLLMClient（.env: LLM_TEMPERATURE）",
        "prompt_source": "llm.prompt.build_prompt（未修改）",
        "update_semantics": "opinion += stance_delta; emotion += emotion_delta（与主实验一致）",
        "fixed_from": "2026-09-20 修复：原 legacy 路径字段不匹配导致 stance 恒为 0.0，旧数据作废",
        "cost_rates_per_1k": {
            "input": float(os.getenv("LLM_INPUT_COST_PER_1K", "0.0008")),
            "output": float(os.getenv("LLM_OUTPUT_COST_PER_1K", "0.002")),
        },
        "runs": [],
    }
    records: List[Dict] = []
    for spec in specs:
        run_id = spec["run_id"]
        if run_id in completed:
            records.append(completed[run_id])
            manifest["runs"].append({
                "run_id": run_id, "alpha": spec["alpha"], "gamma": spec["gamma"],
                "repeat": spec["repeat"], "seed": spec["seed"],
                "status": "completed", "resumed_from_existing": True,
            })
            continue

        print(
            f"[LLM robust] run {run_id + 1}/{expected_runs}: "
            f"alpha={spec['alpha']}, gamma={spec['gamma']}, repeat={spec['repeat']}",
            flush=True,
        )
        calls_path = calls_dir / f"llm_robust_run_{run_id}.jsonl"
        checkpoint_path = checkpoints_dir / f"llm_robust_run_{run_id}.checkpoint.json"
        model, source = load_or_resume(spec, checkpoint_path, calls_path, max_llm_calls)
        model.checkpoint_callback = lambda model_obj, p=checkpoint_path: model_obj.save_checkpoint(str(p))

        try:
            model.run()
        except Exception as exc:
            manifest["runs"].append({
                "run_id": run_id, "alpha": spec["alpha"], "gamma": spec["gamma"],
                "repeat": spec["repeat"], "seed": spec["seed"],
                "status": "failed", "error": repr(exc),
            })
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise

        if len(model.metrics_time_series) != STEPS:
            raise RuntimeError(
                f"run_id={run_id} 时间序列不完整："
                f"{len(model.metrics_time_series)}/{STEPS} 步，未写出结果。"
            )

        quality = model.llm_client.quality_summary()
        guard = assert_not_degenerate(calls_path, quality)

        ts_path = runs_dir / f"llm_robust_run_{run_id}.csv"
        model.save_results(str(ts_path))
        save_agent_snapshot(model, runs_dir / f"llm_robust_run_{run_id}_agents_final.csv")
        model.save_checkpoint(str(checkpoint_path))

        final = dict(model.metrics_time_series[-1])
        input_tokens = int(quality.get("input_tokens", 0))
        output_tokens = int(quality.get("output_tokens", 0))
        final.update({
            "run_id": run_id, "repeat": spec["repeat"], "alpha": spec["alpha"],
            "gamma": spec["gamma"], "seed": spec["seed"], "N": N_AGENTS, "steps": STEPS,
            "model": settings["model"],
            "llm_calls": int(quality.get("api_calls", 0)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost": float(quality.get("estimated_cost", 0.0)),
            "llm_valid_responses": guard["valid_responses"],
            "llm_fallbacks": guard["fallbacks"],
            "llm_fallback_rate": round(guard["fallback_rate"], 6),
            "llm_parse_failures": guard["parse_failures"],
            # 行为侧健康指标：字段齐全也可能"整步全沉默"，必须与兜底率一起看，
            # 否则地板效应会被平均值掩盖。
            "llm_reasoning_starved": guard.get("reasoning_starved", 0),
            "llm_express_rate": round(guard.get("express_rate", 0.0), 6),
            "llm_silent_step_rate": round(guard.get("silent_step_rate", 0.0), 6),
            "status": "completed_with_fallback" if guard["fallbacks"] else "completed",
        })
        records.append(final)
        pd.DataFrame(records).sort_values("run_id").to_csv(summary_path, index=False)

        manifest["runs"].append({
            "run_id": run_id, "alpha": spec["alpha"], "gamma": spec["gamma"],
            "repeat": spec["repeat"], "seed": spec["seed"], "source": source,
            "status": final["status"], "llm_calls": final["llm_calls"],
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "estimated_cost": final["estimated_cost"], "guard": guard,
        })
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[LLM robust] run {run_id}: PI={final.get('PI')} EM={final.get('EM')} "
            f"fallback={guard['fallback_rate']:.2%} 决策取值数={guard['distinct_stance_delta']}",
            flush=True,
        )

    summary = pd.DataFrame(records).sort_values("run_id").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    total_cost = float(summary["estimated_cost"].sum()) if "estimated_cost" in summary else 0.0
    total_in = int(summary["input_tokens"].sum()) if "input_tokens" in summary else 0
    total_out = int(summary["output_tokens"].sum()) if "output_tokens" in summary else 0
    print(f"[LLM robust] 完成 {len(summary)} runs，输出：{output_dir}")
    print(f"[LLM robust] tokens: input={total_in}, output={total_out}; 估算成本合计={total_cost:.4f}")
    report_stratified_health(summary, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 DeepSeek-V4.1-Flash LLM ABM 三条件稳健性实验（v2 栈，3 条件 × R=3，共 9 runs）"
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认写入 <项目根>/results/llm_robust")
    parser.add_argument("--base-seed", type=int, default=22345)
    parser.add_argument("--max-llm-calls", type=int, default=None,
                        help="可选调用上限（cost guard），不传则不限")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    parser.add_argument("--self-test", action="store_true", help="只运行跑前自检，不发起任何调用")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("[LLM robust] self-test-only 完成。")
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
