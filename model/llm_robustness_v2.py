"""LLM robustness model that reuses the Rule ABM environment.

The Rule ABM branch in :mod:`model.model` is not changed.  This subclass only
replaces the decision mechanism used by the LLM branch while retaining the
same agent initialization, post pool, recommendation function, Top-K, metric
function, aging, and time-series schema.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, List

import numpy as np

from config import conf
from llm.robust_client_v2 import RobustLLMClient
from model.dynamics import expression_probability, rule_expression_probability
from model.model import SimulationModel
from model.post import Post
from model.recommendation import recommend_from_arrays


class LLMRobustnessModel(SimulationModel):
    """Simulation model for the mechanism-level LLM robustness check."""

    # 在线消融开关的合法取值（见 __init__ 中的说明）。
    VALID_ABLATIONS = frozenset({"none", "positive_control", "no_post", "no_emotion"})

    def __init__(
        self,
        seed: int,
        N: int = None,
        steps: int = None,
        alpha: float = 0.0,
        gamma: float = 0.1,
        spiral_c: float = 0.0,
        spiral_m: float = 0.0,
        debug: bool = False,
        use_llm: bool = True,
        llm_archive_path: str = None,
        max_llm_calls: int = 0,
        cost_log_path: str = None,
        run_id: int = None,
        ablation: str = None,
    ):
        # Calling the base constructor with use_llm=False avoids constructing
        # the legacy client.  Initialization remains exactly the base model's
        # initialization for a given seed, N, alpha, and gamma.
        #
        # ⚠️ 2026-09-22 修复（断点续跑）：``SimulationModel.load_checkpoint``
        # 会用 ``cls(..., spiral_c=..., spiral_m=...)`` 重建模型，而原先本类
        # 的 ``__init__`` 不接收这两个参数 → 抛
        # ``TypeError: got an unexpected keyword argument 'spiral_c'``
        # → 每个 checkpoint 都被判为"不可用"、run 永远从头重跑。
        # 对本实验（单 run ~9 分钟、整批数小时）意味着任何中断都只能整段作废。
        # 现在显式接收并透传给基类；默认 0.0 与基类默认值一致，
        # 未显式传参时轨迹**逐位不变**。
        super().__init__(
            seed=seed,
            N=N,
            steps=steps,
            alpha=alpha,
            gamma=gamma,
            spiral_c=spiral_c,
            spiral_m=spiral_m,
            debug=debug,
            use_llm=False,
        )
        self.use_llm = True
        self.llm_client = RobustLLMClient(
            archive_path=llm_archive_path or "results/llm_robustness_v2/default.calls.jsonl",
            max_calls=max_llm_calls,
            cost_log_path=cost_log_path,
            run_id=run_id,
        )
        self.run_id = run_id
        # ------------------------------------------------------------------
        # 在线消融（do-干预）开关。默认 "none" 时**逐位不变**（与主实验一致）。
        #   none              ：不做任何干预（主实验行为）
        #   positive_control  ：do(stance_delta=0)，阳性对照/操纵检验
        #   no_post           ：禁用 _append_post，内容池冻结 → 切断表达→内容池→曝光 自反馈回路
        #   no_emotion        ：do(emotion_delta=0)，切断 LLM→情绪 通道
        # ⚠️ 取值优先取显式入参；未传时回落到环境变量 LLM_ABLATION，最后才是 "none"。
        #    之所以要有环境变量回落：基类 load_checkpoint 会用 cls(...) 重建模型，
        #    若不回落，断点续跑时消融开关会丢失（静默退回 none → 数据混入污染）。
        # ------------------------------------------------------------------
        raw_ablation = ablation if ablation is not None else os.getenv("LLM_ABLATION", "none")
        self.ablation = str(raw_ablation or "none").strip().lower()
        if self.ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"未知 ablation={self.ablation!r}，可选：{sorted(self.VALID_ABLATIONS)}"
            )
        self.llm_parallel_workers = max(1, int(getattr(conf, "LLM_PARALLEL_WORKERS", 4)))
        self.llm_call_count = self.llm_client.call_count
        self.llm_quality_stats = self.llm_client.quality_summary()

    def to_checkpoint_dict(self):
        payload = super().to_checkpoint_dict()
        payload["model_version"] = "llm_robustness_v2"
        payload["llm_quality"] = copy.deepcopy(self.llm_client.quality_summary())
        return payload

    def _restore_from_payload(self, payload):
        super()._restore_from_payload(payload)
        saved_quality = payload.get("llm_quality")
        if isinstance(saved_quality, dict):
            for key, value in saved_quality.items():
                if key in self.llm_client._quality:
                    self.llm_client._quality[key] = float(value) if key == "estimated_cost" else int(value)
            self.llm_client._api_call_counter = int(saved_quality.get("api_calls", 0))
        self.llm_quality_stats = self.llm_client.quality_summary()

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    def _feed_features(self, agent, recs: List[Post]) -> Dict[str, float]:
        """只返回模型"能感知"的推荐内容特征。

        ⚠️ 设计红线（2026-09-21）：**不得**加入 ``alpha`` / ``gamma`` / ``top_k``
        或其任何派生量（原先的 ``social_pressure`` 就是
        ``0.5*mean_engagement + 0.25*alpha``，属于隐藏泄漏）。
        详见 ``llm/prompt.py`` 顶部的说明与
        ``deepseek-bug-check/D4_同质推荐角点_改动验证与机制发现_20260920.md`` §10。
        """
        if not recs:
            return {"mean_stance_distance": 0.0, "mean_feed_emotion": agent.emotion}
        distances = [abs(agent.opinion - p.opinion) / 2.0 for p in recs]
        return {
            "mean_stance_distance": float(np.mean(distances)),
            "mean_feed_emotion": float(np.mean([p.emotion for p in recs])),
        }

    def _fallback_decision(self, agent, recs: List[Post], t: int) -> Dict[str, float]:
        """Bounded deterministic fallback, used only for bad/failed outputs."""
        if recs:
            o_feed = float(np.mean([p.opinion for p in recs]))
            e_feed = float(np.mean([p.emotion for p in recs]))
            mean_distance = float(np.mean([abs(agent.opinion - p.opinion) / 2.0 for p in recs]))
            mean_engagement = float(np.mean([p.engagement for p in recs]))
        else:
            o_feed, e_feed, mean_distance, mean_engagement = agent.opinion, agent.emotion, 0.0, 0.0

        # ⚠️ 兜底路径同样不得含 alpha（与 LLM 路径保持一致，避免兜底样本带自变量效应）
        social_pressure = self._clip(float(mean_engagement), 0.0, 1.0)
        # ⚠️ 2026-09-21 【同步死代码】：原先这里硬编码 `*= 1.0 - 0.60 * mean_distance`
        # —— 那是"跨立场沉默"项，但它只存在于兜底分支、且从无生效（兜底率 0%），
        # 同时使规则臂与 LLM 兜底各写一套表达模型（口径不一致）。
        # 现统一调用 model.dynamics.rule_expression_probability：
        #   c = SPIRAL_C_DEFAULT = 0.6（与旧的 1-0.60*d 完全一致）
        #   m = 0.0（本路径不启用少数派惩罚，保持与旧行为一致）
        # 乘法顺序也保持 (1-0.6d) → (0.5+0.5e) → (1-0.35·sp)，故本路径**逐位不变**。
        expression_probability_bounded = rule_expression_probability(
            agent, mean_distance, o_feed,
            conf.B0, conf.BE, conf.BA,
            float(getattr(conf, "SPIRAL_C_DEFAULT", 0.6)),
            0.0,
        )
        expression_probability_bounded *= 0.50 + 0.50 * agent.emotion
        expression_probability_bounded *= 1.0 - 0.35 * social_pressure
        expression_probability_bounded = self._clip(expression_probability_bounded, 0.0, 1.0)

        # Avoid dependence on thread completion order while retaining a random
        # experiment path for fallback decisions.
        fallback_rng = random.Random(self.seed + 1000003 * int(t) + int(agent.id))
        expressed = int(fallback_rng.random() < expression_probability_bounded)
        stance_delta = self._clip(conf.MU * (o_feed - agent.opinion), -0.05, 0.05)
        emotion_delta = self._clip(
            conf.RHO * agent.lambda_sensitivity * (e_feed - agent.emotion), -0.1, 0.1
        )
        return {"express": expressed, "stance_delta": stance_delta, "emotion_delta": emotion_delta}

    def _step_llm(self, t: int):
        """Decide from one immutable Top-K snapshot, then apply in agent order."""
        count = self._post_cache_count
        requests = []
        for agent in self.agents:
            recs = recommend_from_arrays(
                agent,
                self.posts,
                self._post_opinions[:count],
                self._post_engagements[:count],
                self._post_emotions[:count],
                self.alpha,
                self.gamma,
                self.top_k,
                post_author_ids=self._post_author_ids[:count],
                exclude_author_id=agent.id,
            )
            features = self._feed_features(agent, recs)
            # 2026-09-20：把本步该 agent 的实际曝光集合交给基类 step()，
            # 供 HE/EE 指标与 engagement 互动累加共用（不再各自重算一套 feed）。
            self._step_exposure[agent.id] = recs
            state = {
                "opinion": float(agent.opinion),
                "emotion": float(agent.emotion),
                "activity": float(agent.activity),
                "agent_type": agent.agent_type,
                # ⚠️ 不传 alpha / gamma（算法参数不得进入 prompt）；
                # top_k 仅用于 build_prompt 截断渲染条数，不会渲染给模型。
                "top_k": int(self.top_k),
                **features,
            }
            requests.append({"agent": agent, "recs": recs, "state": state, "agent_id": agent.id})

        batches = []
        for start in range(0, len(requests), 10):
            batch = requests[start : start + 10]
            fallbacks = [
                (lambda item=item: self._fallback_decision(item["agent"], item["recs"], t))
                for item in batch
            ]
            batches.append((batch, fallbacks))

        def run_batch(pair):
            batch, fallbacks = pair
            return self.llm_client.generate_batch_actions(batch, self.run_id, t, fallbacks)

        # Each batch sees the same immutable start-of-step snapshot.  Requests
        # may therefore wait concurrently, while results are flattened in the
        # original Agent order before any state is mutated.
        if len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(batches)) as executor:
                batch_results = list(executor.map(run_batch, batches))
        else:
            batch_results = [run_batch(batches[0])]
        decisions = [decision for result in batch_results for decision in result]

        for item, decision in zip(requests, decisions):
            agent = item["agent"]
            # --- 在线消融（do-干预）作用于"决策被消费"的这一步，不改 prompt、不改调用次数 ---
            # Abl-1 阳性对照：do(stance_delta=0)
            stance_delta = float(decision.get("stance_delta", 0.0))
            if self.ablation == "positive_control":
                stance_delta = 0.0
            agent.opinion = self._clip(agent.opinion + stance_delta, -1.0, 1.0)
            emotion_delta = self._clip(decision.get("emotion_delta", 0.0), -0.1, 0.1)
            # Abl-3：do(emotion_delta=0)，情绪冻结在初值
            if self.ablation == "no_emotion":
                emotion_delta = 0.0
            agent.emotion = self._clip(agent.emotion + emotion_delta, 0.0, 1.0)
            agent.expression_probability = 1.0 if int(decision.get("express", 0)) == 1 else 0.0
            # Abl-2：禁用写入新帖 → 内容池冻结，切断"表达→内容池→曝光"自反馈回路。
            # 注意：expression_probability 仍照常记录（HE/EE 指标需要），只是不 append。
            if agent.expression_probability == 1.0 and self.ablation != "no_post":
                self._append_post(agent.id, agent.opinion, agent.emotion, created_step=t)

        self.llm_call_count = self.llm_client.call_count
        self.llm_quality_stats = self.llm_client.quality_summary()
