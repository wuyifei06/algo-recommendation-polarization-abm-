"""核心仿真模型：负责时间循环、内容池与度量记录。

此模块实现 `SimulationModel`，包含：
- Agent 与初始 Post 的创建
- 每步推荐、情绪/意见更新、表达生成
- 指标计算并保存时间序列结果

设计原则：保持仿真内部逻辑清晰、避免将 I/O 逻辑散落在模型内部。
保存结果使用 `save_results(path)` 将 time-series 写为 CSV。
"""
from typing import List, Dict, Any
import json
import random
import os
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from llm.client import LLMClient
from model.agent import Agent
from model.post import Post
from model.recommendation import recommend_from_arrays
from model.dynamics import update_emotion, update_opinion, expression_probability, rule_expression_probability
from metrics.metrics import compute_group_metrics
from config import conf
import pandas as pd
import numpy as np


class SimulationModel:
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
        use_llm: bool = False,
        llm_archive_path: str = None,
    ):
        self.seed = seed
        random.seed(seed)
        self.N = N or conf.N_AGENTS
        self.steps = steps or conf.STEPS
        self.alpha = alpha
        self.gamma = gamma
        # ⚠️ 沉默的螺旋扩展（2026-09-21）：默认 0.0 = 关闭，
        # 此时规则臂表达概率走原来的 expression_probability，轨迹逐位不变。
        # 只有显式传入 >0 才会启用（见 run_rule_spiral.py）。
        self.spiral_c = float(spiral_c)
        self.spiral_m = float(spiral_m)
        self.top_k = conf.TOP_K
        self.max_posts = conf.MAX_POSTS
        self.h = conf.H
        self.rho = conf.RHO
        self.mu = conf.MU
        self.debug = debug
        self.use_llm = use_llm
        self.llm_client = LLMClient(archive_path=llm_archive_path) if self.use_llm else None
        self.llm_parallel_workers = max(1, int(getattr(conf, "LLM_PARALLEL_WORKERS", 1))) if self.use_llm else 1
        self.llm_call_count = self.llm_client._call_counter if self.llm_client else 0
        self.current_step = 0
        self.checkpoint_callback = None

        self.agents: List[Agent] = []
        self.posts: List[Post] = []
        self.metrics_time_series: List[Dict] = []
        self._post_id_counter = 0
        # 帖子字段缓存。规则实验中每个 agent 都会对同一帖子池做一次推荐，
        # 维护列式缓存可以避免每次推荐都重新遍历 Post 对象并组装二维数组。
        self._post_opinions = np.empty(max(0, self.max_posts), dtype=float)
        self._post_engagements = np.empty(max(0, self.max_posts), dtype=float)
        self._post_emotions = np.empty(max(0, self.max_posts), dtype=float)
        # ⚠️ 2026-09-20 修复（D4）：帖子作者 id 的列式缓存。推荐需要把
        # "agent 自己发的帖子"排除出候选集，若每次推荐都遍历 Post 对象取
        # author_id，就会抵消列式缓存带来的全部收益，故一并缓存。
        self._post_author_ids = np.empty(max(0, self.max_posts), dtype=np.int64)
        # ⚠️ 2026-09-20 修复（D1）：engagement 的原始互动计数（未归一化）。
        # "热度"语义要求 engagement 随其他 agent 的互动逐级累加，因此必须与
        # 用于推荐的归一化值分开存放；否则每步归一化都会把累加结果覆盖掉。
        self._post_raw_engagement = np.empty(max(0, self.max_posts), dtype=float)
        self._post_cache_count = 0
        # 本步每个 agent 实际收到的推荐集合（agent_id -> List[Post]）。
        # 供 HE/EE 指标与互动累加使用，避免二者各自再按"纯相似度"重算一套 feed。
        self._step_exposure: Dict[int, List[Post]] = {}

        # 初始化代理和初始内容池
        self._create_agents()
        self._seed_initial_posts()

    def save_checkpoint(self, path: str):
        """原子 + 抗瞬时占用地写 checkpoint。

        ⚠️ 2026-09-21（崩溃修复）：原来的实现是 ``open(path, "w")`` 直接覆盖写入，
        有两个脆弱点：
          1. Windows 上写文件可能因**进程外瞬时占用**（实时杀毒扫描 / 桌面同步盘）
             抛 ``PermissionError``。checkpoint 每步都写（30 步 × 160 run ≈ 4800 次），
             概率虽低但足以拖垮一次数小时的实验 —— 事实上 llm_gradient_noleak
             就是同类错误（写 calls/*.jsonl）把整个进程打崩的。
          2. ``"w"`` 先截断再写，若写入中途进程被杀，checkpoint 会变成**半截文件**，
             而它随后会被当作有效断点读入并续跑。
        改为「写临时文件 → os.replace 原子替换」并退避重试：
        文件要么是旧的完整版本，要么是新的完整版本，永远不会半截。
        """
        payload = self.to_checkpoint_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp = f"{path}.tmp"
        last_exc = None
        for attempt in range(1, 6):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, path)
                return
            except PermissionError as exc:  # pragma: no cover - 依赖运行环境
                last_exc = exc
                if attempt < 5:
                    time.sleep(0.15 * attempt)
        assert last_exc is not None
        raise last_exc

    def to_checkpoint_dict(self):
        """返回可 JSON 序列化的完整模型状态。"""
        return {
            "seed": self.seed,
            "N": self.N,
            "steps": self.steps,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "spiral_c": self.spiral_c,
            "spiral_m": self.spiral_m,
            "use_llm": self.use_llm,
            "current_step": self.current_step,
            "llm_call_count": self.llm_call_count,
            "agents": [copy.deepcopy(agent.__dict__) for agent in self.agents],
            "posts": [copy.deepcopy(post.__dict__) for post in self.posts],
            "metrics_time_series": copy.deepcopy(self.metrics_time_series),
            "_post_id_counter": self._post_id_counter,
        }

    def _restore_from_payload(self, payload):
        """把 checkpoint 字典恢复到当前对象。"""
        self.current_step = int(payload.get("current_step", 0))
        self.llm_call_count = int(payload.get("llm_call_count", 0))
        self._post_id_counter = int(payload.get("_post_id_counter", len(payload.get("posts", []))))
        self.agents = [
            Agent(
                id=int(a.get("id", i)),
                opinion=float(a.get("opinion", 0.0)),
                emotion=float(a.get("emotion", 0.0)),
                agent_type=str(a.get("agent_type", "moderate")),
                activity=float(a.get("activity", 0.0)),
                expression_probability=float(a.get("expression_probability", 0.0)),
                interaction_history=list(a.get("interaction_history", [])),
                exposure_history=list(a.get("exposure_history", [])),
                lambda_sensitivity=float(a.get("lambda_sensitivity", 0.5)),
            )
            for i, a in enumerate(payload.get("agents", []))
        ]
        self.posts = [
            Post(
                post_id=int(p.get("post_id", i)),
                author_id=int(p.get("author_id", 0)),
                opinion=float(p.get("opinion", 0.0)),
                emotion=float(p.get("emotion", 0.0)),
                engagement=float(p.get("engagement", 0.0)),
                created_step=int(p.get("created_step", 0)),
            )
            for i, p in enumerate(payload.get("posts", []))
        ]
        self.metrics_time_series = copy.deepcopy(payload.get("metrics_time_series", []))
        self._rebuild_post_cache()

    @classmethod
    def load_checkpoint(cls, path: str, use_llm: bool = False, llm_archive_path: str = None):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        model = cls(
            seed=int(payload["seed"]),
            N=int(payload["N"]),
            steps=int(payload["steps"]),
            alpha=float(payload.get("alpha", 0.0)),
            gamma=float(payload.get("gamma", 0.1)),
            spiral_c=float(payload.get("spiral_c", 0.0)),
            spiral_m=float(payload.get("spiral_m", 0.0)),
            use_llm=bool(payload.get("use_llm", use_llm)),
            llm_archive_path=llm_archive_path,
        )
        model._restore_from_payload(payload)
        if model.llm_client is not None:
            # checkpoint 中的计数属于模型状态；同步到客户端，避免恢复后
            # 调用编号从 1 重新开始覆盖/混淆已有调用存档。
            model.llm_client._call_counter = max(
                model.llm_client._call_counter,
                model.llm_call_count,
            )
        return model

    def _rebuild_post_cache(self):
        """从帖子对象重建列式缓存（用于加载 checkpoint）。"""
        count = min(len(self.posts), max(0, self.max_posts))
        if count == 0:
            self._post_cache_count = 0
            return
        # checkpoint 中的帖子本来就不应超过 max_posts；截取尾部也使这里
        # 对旧版本/手工 checkpoint 更稳健。
        if len(self.posts) > count:
            self.posts = self.posts[-count:]
        self._post_cache_count = count
        self._post_opinions[:count] = [p.opinion for p in self.posts]
        self._post_engagements[:count] = [p.engagement for p in self.posts]
        self._post_emotions[:count] = [p.emotion for p in self.posts]
        self._post_author_ids[:count] = [p.author_id for p in self.posts]
        # checkpoint 只保存归一化后的 engagement；恢复时以它作为原始互动计数
        # 的近似起点（重启后排名顺序与中断前一致，量级差异会在下一步归一化时抹平）。
        self._post_raw_engagement[:count] = [p.engagement for p in self.posts]

    def _create_agents(self):
        # 按照指定比例创建三类 Agent：extreme / moderate / swing
        # ⚠️ 2026-09-20 修复（D2）：论文 §三（三）声明个体情绪敏感性
        # λi ∈ [0.5, 1.5]，但旧实现从未写入该字段，全部沿用 Agent 默认值 0.5，
        # 使"情绪敏感性异质性"实际上不存在。现按论文声明抽样赋值。
        n_ext = int(self.N * 0.2)
        n_mod = int(self.N * 0.5)
        n_swing = self.N - n_ext - n_mod
        idx = 0
        for _ in range(n_ext):
            a = Agent(idx, opinion=random.choice([-1.0, 1.0]), emotion=random.random(), agent_type="extreme", activity=random.random(), lambda_sensitivity=random.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1
        for _ in range(n_mod):
            a = Agent(idx, opinion=random.uniform(-0.3, 0.3), emotion=random.random(), agent_type="moderate", activity=random.random(), lambda_sensitivity=random.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1
        for _ in range(n_swing):
            a = Agent(idx, opinion=random.uniform(-1.0, 1.0), emotion=random.random(), agent_type="swing", activity=random.random(), lambda_sensitivity=random.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1

    def _seed_initial_posts(self):
        # 每个 Agent 在 t=0 时发布一条初始 Post，保证内容池非空
        for a in self.agents:
            self._append_post(a.id, a.opinion, a.emotion, created_step=0)

    def _append_post(self, author_id: int, opinion: float, emotion: float, created_step: int):
        # ⚠️ 2026-09-20 修复（D1）：新帖的原始互动计数从一个随机先验起步
        # （仅用于打破排名并列），而归一化后的 engagement 置 0 —— 新帖此刻
        # 尚未获得任何互动。之后由 _accumulate_engagement 按实际互动累加。
        raw_engagement = random.random()
        p = Post(self._post_id_counter, author_id, opinion, emotion, engagement=0.0, created_step=created_step)
        self._post_id_counter += 1

        if self.max_posts <= 0:
            self.posts.clear()
            self._post_cache_count = 0
            return

        # 与原来的 ``append + 删除最旧帖子`` 保持相同的帖子池顺序，
        # 同时对缓存做同样的左移，避免每次推荐重新扫描对象字段。
        if self._post_cache_count >= self.max_posts:
            del self.posts[0]
            self._post_opinions[:-1] = self._post_opinions[1:self.max_posts]
            self._post_engagements[:-1] = self._post_engagements[1:self.max_posts]
            self._post_emotions[:-1] = self._post_emotions[1:self.max_posts]
            self._post_author_ids[:-1] = self._post_author_ids[1:self.max_posts]
            self._post_raw_engagement[:-1] = self._post_raw_engagement[1:self.max_posts]
            self._post_cache_count = self.max_posts - 1

        self.posts.append(p)
        idx = self._post_cache_count
        self._post_opinions[idx] = p.opinion
        self._post_engagements[idx] = p.engagement
        self._post_emotions[idx] = p.emotion
        self._post_author_ids[idx] = p.author_id
        self._post_raw_engagement[idx] = raw_engagement
        self._post_cache_count += 1

    def _normalize_engagement(self):
        """把原始互动计数归一化为推荐打分用的 [0,1] 权重。

        ⚠️ 2026-09-20 修复（D1）：原实现用"除以当前最大值"。互动计数是
        重尾分布（少数帖子被反复曝光，大多数长期为 0），除以最大值会把绝大多数
        帖子压到 ≈0，只留下一个帖子 ≈1。于是 α 一旦变大，所有 agent 都被指向
        同一条帖子（等价于全局广播），PI 在 α≥1.0 会整批塌到 0 —— 这是归一化
        的数值伪影，不是模型行为。

        改用**秩归一化**：engagement 只在"谁比谁更热"的比较中起作用，
        与绝对量级无关，因此秩是最自然且尺度无关的归一化方式；α∈[0,2] 的
        标定也因此保持稳定（不会再出现 α≥1 的塌缩）。
        """
        count = self._post_cache_count
        if count <= 0:
            return
        raw = self._post_raw_engagement[:count]
        # stable argsort + 逆置换 → 秩；并列按帖子先后顺序确定性地打破。
        order = np.argsort(raw, kind="stable")
        ranks = np.empty(count, dtype=float)
        ranks[order] = np.arange(count, dtype=float)
        ranks = ranks / (count - 1) if count > 1 else np.full(count, 0.5)
        self._post_engagements[:count] = ranks
        for i in range(count):
            self.posts[i].engagement = float(ranks[i])

    def _accumulate_engagement(self):
        """按本步每个 agent 实际收到的推荐集合，累加各帖的互动计数。

        ⚠️ 2026-09-20 修复（D1）：原实现每步执行
        ``p.engagement += random.random() * 0.01``（注释自称 "aging"），
        与任何互动行为无关 —— 实测 engagement 与帖子年龄/情绪/观点的相关
        全部为噪声（-0.049 / -0.030 / +0.007）。这使论文 §三（二）所述
        "Engagej 为该内容归一化后的互动量""α 提升意味着也推高互动内容"
        在代码中并不成立。

        现按原始设计恢复互动语义：agent 对 feed 中每条内容以
        interaction_prob = clip(0.05 + 0.25·emotion + 0.20·similarity, 0, 0.9)
        的概率产生一次互动，命中则该帖计数 +1。交互强度取决于该 agent 与
        该帖的观点相似度以及帖子情绪，因此 engagement 重新成为内生量：
        高互动内容 = 高相似度或高情绪内容，α 也因此重新具备"社会证明"语义。
        """
        exposure = self._step_exposure
        if not exposure or self._post_cache_count <= 0:
            return
        # 帖子池索引映射（本步内构建一次，O(池大小)）
        id_to_index = {p.post_id: i for i, p in enumerate(self.posts)}
        for agent in self.agents:
            recs = exposure.get(agent.id)
            if not recs:
                continue
            for post in recs:
                i = id_to_index.get(post.post_id)
                if i is None:
                    continue
                sim = 1.0 - abs(agent.opinion - post.opinion) / 2.0
                if sim < 0.0:
                    sim = 0.0
                prob = 0.05 + 0.25 * post.emotion + 0.20 * sim
                if prob > 0.9:
                    prob = 0.9
                if random.random() < prob:
                    self._post_raw_engagement[i] += 1.0

    def _step_llm(self, t: int):
        """完成一个 LLM 时间步，并行等待多个 agent 的模型响应。

        所有请求使用本时间步开始时的推荐快照，结果仍按 agent 原顺序
        应用；因此 agent 数量、时间步数和调用次数不变，只减少网络等待。
        """
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
            self._step_exposure[agent.id] = recs
            state = {"opinion": float(agent.opinion), "emotion": float(agent.emotion)}
            feed = {"items": [{"opinion": float(p.opinion), "emotion": float(p.emotion)} for p in recs]}
            requests.append((agent, state, feed))

        def request_action(item):
            _, state, feed = item
            return self.llm_client.generate_agent_action(state, feed, timeout=self.llm_client.timeout)

        if self.llm_parallel_workers == 1:
            decisions = [request_action(item) for item in requests]
        else:
            workers = min(self.llm_parallel_workers, max(1, len(requests)))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                decisions = list(executor.map(request_action, requests))

        for (agent, _, _), decision in zip(requests, decisions):
            self.llm_call_count += 1
            # ⚠️ 2026-09-20 修复：prompt 要求的是**相对增量**（stance_delta /
            # emotion_delta），旧实现却按绝对量赋值（agent.opinion = stance），
            # 叠加解析器把缺失字段静默填 0.0，导致第 1 步全体立场被重置为 0
            # 并永久锁死（PI/VPI/ER ≡ 0 的假结果）。
            # 现在：有增量按增量累加；仅旧格式才按绝对量赋值；两者都缺 → 报错。
            if "stance_delta" in decision:
                agent.opinion = max(-1.0, min(1.0, float(agent.opinion) + float(decision["stance_delta"])))
            elif "stance" in decision:
                agent.opinion = max(-1.0, min(1.0, float(decision["stance"])))
            else:
                raise ValueError("LLM 决策缺少立场字段（stance_delta 或 stance），拒绝静默兜底")

            if "emotion_delta" in decision:
                agent.emotion = max(0.0, min(1.0, float(agent.emotion) + float(decision["emotion_delta"])))
            elif "emotion" in decision:
                agent.emotion = max(0.0, min(1.0, float(decision["emotion"])))
            else:
                raise ValueError("LLM 决策缺少情绪字段（emotion_delta 或 emotion），拒绝静默兜底")

            expresses = bool(decision.get("express", False))
            agent.expression_probability = 1.0 if expresses else 0.0
            if expresses:
                self._append_post(agent.id, agent.opinion, agent.emotion, created_step=t)

    def step(self, t: int):
        # 本步实际曝光集合（每个 agent 的 Top-K），由下面的分支写入，
        # 供 metric 计算（HE/EE）与互动累加（engagement）共用同一份 feed。
        self._step_exposure = {}

        # normalization
        self._normalize_engagement()

        d_expo_step_mean = None  # 仅规则臂写入；LLM 臂为 None → 不产生该列
        if self.use_llm:
            self._step_llm(t)
        else:
            # ⚠️ 2026-09-21：本步 Top-K 曝光的平均立场距离（机制通道诊断量）。
            #    只做「记录」，不参与任何决策 → 不改变轨迹、不消耗随机数。
            #    spiral 关闭时也会记录，用于回答"α 是否真的提高了跨立场曝光"。
            _d_expo_sum = 0.0
            # 规则模型保持原有的逐 agent 更新顺序。
            for agent in self.agents:
                # 为每个 agent 获取推荐内容（Top-K）
                count = self._post_cache_count
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
                self._step_exposure[agent.id] = recs
                K = len(recs) if recs else 0
                if K == 0:
                    o_feed = agent.opinion
                    e_feed = agent.emotion
                else:
                    o_feed = sum(p.opinion for p in recs) / K
                    e_feed = sum(p.emotion for p in recs) / K

                # ⚠️ 2026-09-21（沉默的螺旋扩展）：曝光立场距离必须在**状态更新之前**
                # 取值，与 LLM 臂 `_feed_features` 的取值时点保持一致；否则两臂的 d
                # 口径不同（一个用更新前立场、一个用更新后立场），不可比。
                # 这里**无条件计算**（含 spiral 关闭时）以便记录机制通道；
                # 数值上对 spiral 开启的情形没有影响，因为它本就是同一个式子。
                d_expo = (sum(abs(agent.opinion - p.opinion) for p in recs) / (2.0 * K)) if K else 0.0
                _d_expo_sum += d_expo

                # update emotion
                agent.emotion = update_emotion(agent, e_feed, self.rho)
                # update opinion
                agent.opinion = update_opinion(agent, o_feed, self.mu)

                # expression decision
                if self.spiral_c > 0.0 or self.spiral_m > 0.0:
                    p_expr = rule_expression_probability(
                        agent, d_expo, o_feed, conf.B0, conf.BE, conf.BA,
                        self.spiral_c, self.spiral_m,
                    )
                else:
                    p_expr = expression_probability(agent, conf.B0, conf.BE, conf.BA)
                expressed = random.random() < p_expr
                # ⚠️ 2026-09-20 修复（口径不一致）：metrics 的 SR = 1 - mean(agent.expression_probability)，
                # 而 LLM 分支（本文件 _step_llm 与 model/llm_robustness_v2.py）写入的是
                # 【本步是否真的表达 0/1】。旧实现此处写入的是【表达倾向 p_expr】，
                # 导致 SR 在 Rule 臂是"预期沉默率"、在 LLM 臂是"实际沉默率"，
                # 两臂 SR 不可直接比较（而"沉默率上升"正是论文的核心发现之一）。
                # 现统一为实际表达结果；p_expr 仍用于本步决策，轨迹不变。
                agent.expression_probability = 1.0 if expressed else 0.0
                if expressed:
                    # expression generates a new post
                    self._append_post(agent.id, agent.opinion, agent.emotion, created_step=t)

            if self.agents:
                d_expo_step_mean = _d_expo_sum / len(self.agents)

        # ⚠️ 2026-09-20 修复（D1）：engagement 改为互动驱动累加，
        # 取代原来的 ``+= random.random() * 0.01``（与互动无关的随机漂移）。
        self._accumulate_engagement()

        # compute metrics
        count = self._post_cache_count
        metrics = compute_group_metrics(
            self.agents,
            self.posts,
            top_k=self.top_k,
            post_opinions=self._post_opinions[:count],
            post_emotions=self._post_emotions[:count],
            agent_exposure=self._step_exposure,
        )
        metrics.update({"step": t, "alpha": self.alpha, "gamma": self.gamma, "seed": self.seed,
                        "spiral_c": self.spiral_c, "spiral_m": self.spiral_m})
        if d_expo_step_mean is not None:
            metrics["mean_exposure_distance"] = d_expo_step_mean
        self.metrics_time_series.append(metrics)
        self.current_step = t
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(self)

    def run(self):
        start = self.current_step + 1
        for t in range(start, self.steps + 1):
            # 若 LLM 在一个 step 的中途失败，step 内可能已经修改了部分
            # agent。回滚到该 step 开始状态，checkpoint 才能安全续跑。
            before_step = copy.deepcopy(self.to_checkpoint_dict())
            try:
                self.step(t)
            except Exception:
                self._restore_from_payload(before_step)
                raise

    def save_results(self, path: str):
        """原子 + 抗瞬时占用地写时间序列 CSV（理由同 `save_checkpoint`）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.metrics_time_series)
        tmp = f"{path}.tmp"
        last_exc = None
        for attempt in range(1, 6):
            try:
                df.to_csv(tmp, index=False)
                os.replace(tmp, path)
                return
            except PermissionError as exc:  # pragma: no cover - 依赖运行环境
                last_exc = exc
                if attempt < 5:
                    time.sleep(0.15 * attempt)
        assert last_exc is not None
        raise last_exc
