"""LLM Agent 的行为约束 Prompt。

Prompt 不负责计算 PI、SR、HE 等群体指标；它只要求模型对单个 Agent
给出可解析的行为决策。``emotion`` 保留为旧版客户端兼容字段，新的
稳健性实验优先使用 ``emotion_delta``。

⚠️ 设计红线（2026-09-21）
------------------------
**算法参数（alpha / gamma / top_k）及其派生的 ``social_pressure`` 一律不得
出现在 prompt 里。** 模型只能感知"推荐内容本身"及其可观测量（立场距离、
情绪、互动），不能知道实验设计。

原因：把这些数值写进 prompt 会让模型"按提示词表演"，自变量因此多了一条
直接通路，跨臂对比随之失效。实测证据见
``deepseek-bug-check/D4_同质推荐角点_改动验证与机制发现_20260920.md`` §10：
控制"模型实际输出幅度"后 alpha 对极化的效应从 p<1e-4 变为 p=0.60。

任何新增字段前请自问：**真人用户能不能"感觉到"这个量？** 不能，就不要给。
"""
from typing import Dict, Any
import json


def build_prompt(state: Dict[str, Any], items: list) -> str:
    """Build a bounded, JSON-only decision prompt.

    The input fields are deliberately explicit so the LLM sees the same
    recommendation snapshot that the rule model uses.  The model is asked to
    decide behaviour only; all population metrics remain program-computed.
    """
    lines = [
        "你是一个社交媒体用户，正在根据当前推荐的 Top-K 内容做一次行为决策。",
        "你只能返回一个 JSON 对象，不能输出解释、Markdown、思考过程或额外文本。",
        f"当前立场={state.get('opinion')}",
        f"当前情绪={state.get('emotion')}",
        f"活动性={state.get('activity')}",
        f"Agent类型={state.get('agent_type')}",
        f"推荐内容平均立场距离={state.get('mean_stance_distance')}",
        f"推荐内容平均情绪={state.get('mean_feed_emotion')}",
        "推荐内容（已经由程序按统一的 Top-K 规则排序）：",
    ]
    for i, it in enumerate(items[: int(state.get("top_k") or 10)], 1):
        lines.append(
            f"{i}: 立场={it.get('opinion')}, 情绪={it.get('emotion')}, "
            f"互动={it.get('engagement')}, 创建轮次={it.get('created_step')}"
        )
    lines.extend(
        [
            "行为约束：",
            "1. express 必须是整数 0 或 1。你可以根据推荐内容与自身状态决定是否表达，也可以选择不表达；不要默认每个 Agent 都表达。",
            "2. emotion_delta 必须在 [-0.1, 0.1] 内，只根据当前情绪、推荐内容情绪、立场距离和互动反馈作有限变化；不凭空制造极端情绪。",
            "3. stance_delta 必须在 [-0.05, 0.05] 内；不要计算任何群体指标。",
            "严格返回：{\"express\":0或1,\"stance_delta\":[-0.05,0.05]内数值,\"emotion_delta\":[-0.1,0.1]内数值}",
        ]
    )
    return "\n".join(lines)


def build_batch_prompt(batch: list) -> str:
    """Compact JSON-only prompt for a batch of Agent decisions.

    The recommendation function has already selected the Top-K feed.  To keep
    requests small, each Agent is represented by bounded summary features of
    that exact feed rather than repeating all post text.
    """
    compact = []
    for item in batch:
        state = item["state"]
        compact.append(
            {
                "id": item["agent_id"],
                "o": round(float(state["opinion"]), 4),
                "e": round(float(state["emotion"]), 4),
                "a": round(float(state["activity"]), 4),
                "d": round(float(state.get("mean_stance_distance", 0.0)), 4),
                "fe": round(float(state.get("mean_feed_emotion", 0.0)), 4),
            }
        )
    rules = (
        "Return only a JSON array with one object per input id, in the same order. "
        "Each object must contain only express (0/1), stance_delta [-0.05,0.05], "
        "emotion_delta [-0.1,0.1]. Decide from the shown feed and your own state; "
        "you may stay silent or keep your stance unchanged when you judge it "
        "appropriate, and not everyone needs to express. "
        "No comments or explanations."
    )
    return rules + "\nINPUT=" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
