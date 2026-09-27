"""计算群体级别的舆论极化等指标。

返回指标包括：PI, VPI, SR, ER, EM, HE, EE。
HE 与 EE 按每个 agent 的 Top-K **实际曝光项**（由调用方通过 agent_exposure 传入
真实推荐集合）计算并汇总为群体均值；未传入 agent_exposure 时才回退为按纯相似度
重取 Top-K 的旧近似算法。
"""
from typing import List, Dict
from model.agent import Agent
from model.post import Post
import numpy as np


def compute_group_metrics(
    agents: List[Agent],
    posts: List[Post],
    top_k: int = 10,
    post_opinions: np.ndarray = None,
    post_emotions: np.ndarray = None,
    agent_exposure: Dict[int, List[Post]] = None,
) -> Dict:
    opinions = np.array([a.opinion for a in agents])
    emotions = np.array([a.emotion for a in agents])

    # PI
    PI = float(np.var(opinions))

    # expressed posts authors
    if posts:
        expressed_agent_ids = set(p.author_id for p in posts)
        expressed_opinions = np.array([a.opinion for a in agents if a.id in expressed_agent_ids])
        VPI = float(np.var(expressed_opinions)) if expressed_opinions.size > 0 else 0.0
    else:
        VPI = 0.0

    # SR = 1 - expression_rate
    expression_rate = np.mean([a.expression_probability for a in agents]) if agents else 0.0
    SR = 1.0 - expression_rate

    # ER: proportion |opinion| >= 0.7
    ER = float(np.mean((np.abs(opinions) >= 0.7).astype(float)))

    # EM: mean emotion
    EM = float(np.mean(emotions)) if emotions.size > 0 else 0.0

    # HE 与 EE：按每个 agent 的**实际推荐集合**计算（2026-09-20 修复 D3）。
    # 旧实现无视 alpha/gamma，对所有 agent 一律重新按"纯相似度"取 Top-K，
    # 因此 HE 在构造上必然接近其上限（≈1.0）、EE 又与 EM 共线，导致论文 RQ2 的
    # 中介通道在构造上不可识别。现在优先使用调用方传入的真实曝光集合；
    # 未提供时才回退到旧的纯相似度近似（保持向后兼容）。
    HE_list = []
    EE_list = []
    if posts:
        if post_opinions is None:
            post_opinions = np.asarray([p.opinion for p in posts], dtype=float)
        if post_emotions is None:
            post_emotions = np.asarray([p.emotion for p in posts], dtype=float)

    for a in agents:
        exposure = agent_exposure.get(a.id) if agent_exposure else None
        if exposure is not None:
            feed_opinions = [float(p.opinion) for p in exposure]
            feed_emotions = [float(p.emotion) for p in exposure]
        elif post_opinions is not None and post_opinions.size > 0:
            sims = np.maximum(0.0, 1.0 - np.abs(a.opinion - post_opinions) / 2.0)
            count = min(int(top_k), len(sims))
            candidates = np.argpartition(sims, -count)[-count:] if count else []
            idx = candidates[np.argsort(sims[candidates])[::-1]] if count else []
            feed_opinions = [float(post_opinions[i]) for i in idx]
            feed_emotions = [float(post_emotions[i]) for i in idx]
        else:
            feed_opinions = []
            feed_emotions = []

        K = len(feed_opinions)
        if K == 0:
            HE_list.append(1.0)
            EE_list.append(0.0)
            continue
        D_i = sum(abs(a.opinion - o) for o in feed_opinions) / K
        HE_list.append(1.0 - D_i / 2.0)
        EE_list.append(sum(feed_emotions) / K)

    HE = float(np.mean(HE_list)) if HE_list else 0.0
    EE = float(np.mean(EE_list)) if EE_list else 0.0

    return {
        "PI": PI,
        "VPI": VPI,
        "SR": SR,
        "ER": ER,
        "EM": EM,
        "HE": HE,
        "EE": EE,
    }
