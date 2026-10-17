"""推荐模块：计算相似度与得分并返回 Top-K。

实现推荐得分：Score = H * Similarity + alpha * Engagement + gamma * Emotion
其中 Similarity 按用户观点差距线性映射到 [0,1]。注意：engagement
应在模型中事先归一化。

⚠️ 2026-09-20 修复（D4）：推荐候选集排除 agent **自己发的帖子**。
原实现不排除自身帖，而 ``sim = 1 - |Δopinion|/2`` 对自己发的帖严格等于 1
（观点完全相同），对其他帖严格小于 1 → α=0（纯相似度）时 Top-K 被自己的帖子
占满（N=200/T=200 实测：占 87.5% 的席位，87% 的 agent 十席全满），
于是 ``o_feed = o_self``、观点更新位移为 0，群体观点被**冻结在初值**。
论文的 α=0 应表示"有社会互动的同质推荐"，故推荐候选必须排除自身帖；
自身帖仍留在全局内容池中（可被推荐给其他 agent）。
"""
from typing import List
from model.agent import Agent
from model.post import Post
import numpy as np


def similarity(op_i: float, op_j: float) -> float:
    """观点相似度函数，返回范围 [0,1]。"""
    return max(0.0, 1.0 - abs(op_i - op_j) / 2.0)


def _stable_topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """降序取前 k 个下标；并列时按索引先后（确定性）。

    默认走 O(n) 的 argpartition；只有当第 k 名与第 k+1 名分数**完全相等**
    （即截断处存在并列）时，才退回完整稳定排序，保证跨平台逐位可复现，
    同时避免每次推荐都做 O(n log n) 排序带来的性能损失。
    """
    n = int(values.size)
    k = min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.argsort(-values, kind="stable")
    part = np.partition(values, n - k)
    thr = part[n - k]
    nxt = np.partition(values, n - k - 1)[n - k - 1]
    if nxt == thr:                                  # 截断处并列 → 稳定排序
        return np.argsort(-values, kind="stable")[:k]
    cand = np.argpartition(values, n - k)[n - k:]   # 无并列 → 快速路径
    return cand[np.argsort(-values[cand], kind="stable")]


def score_for_post(agent: Agent, post: Post, alpha: float, gamma: float, H: float = 1.0) -> float:
    """计算单条 Post 对 agent 的推荐得分。

    参数 alpha 表示算法对 engagement 的偏好权重，gamma 表示算法对
    内容情绪的偏好权重。H 为同质性项的系数（通常为 1.0）。
    """
    sim = similarity(agent.opinion, post.opinion)
    # engagement 期望在调用前已被归一化到 [0,1]
    return H * sim + alpha * post.engagement + gamma * post.emotion


def recommend(agent: Agent, posts: List[Post], alpha: float, gamma: float, top_k: int) -> List[Post]:
    """返回按得分排序的 Top-K Posts（可能少于 top_k）。"""
    if not posts:
        return []
    post_values = np.asarray([(p.opinion, p.engagement, p.emotion) for p in posts], dtype=float)
    return recommend_from_arrays(
        agent,
        posts,
        post_values[:, 0],
        post_values[:, 1],
        post_values[:, 2],
        alpha,
        gamma,
        top_k,
        post_author_ids=np.asarray([p.author_id for p in posts], dtype=np.int64),
    )


def recommend_from_arrays(
    agent: Agent,
    posts: List[Post],
    post_opinions: np.ndarray,
    post_engagements: np.ndarray,
    post_emotions: np.ndarray,
    alpha: float,
    gamma: float,
    top_k: int,
    post_author_ids: np.ndarray = None,
    exclude_author_id: int = None,
) -> List[Post]:
    """使用已经准备好的帖子字段数组返回 Top-K。

    规则实验中每个 agent 都会访问同一个帖子池。把帖子对象逐条转换成
    ``(opinion, engagement, emotion)`` 数组是一个很大的重复开销，因此模型
    会维护这些数组并在帖子新增/淘汰时增量更新。

    ``post_author_ids`` / ``exclude_author_id``：当两者都提供时，作者等于
    ``exclude_author_id`` 的帖子会被**排除出候选集**（推荐不推给自己的帖子）。
    这是默认行为（调用方一般传 ``exclude_author_id=agent.id``）；未提供
    ``post_author_ids`` 时退回到不排除的旧行为，以兼容外部调用。
    """
    if not posts:
        return []

    scores = (
        np.maximum(0.0, 1.0 - np.abs(agent.opinion - post_opinions) / 2.0)
        + alpha * post_engagements
        + gamma * post_emotions
    )

    # 候选集：排除自身帖。
    if exclude_author_id is not None and post_author_ids is not None:
        pool = np.flatnonzero(post_author_ids != exclude_author_id)
        n_valid = int(pool.size)
    else:
        pool = None
        n_valid = int(scores.size)

    if n_valid == 0:
        return []

    idx = min(int(top_k), n_valid)
    if pool is None:
        top_indices = _stable_topk_indices(scores, idx)
        return [posts[i] for i in top_indices]

    sub = scores[pool]
    top_indices = pool[_stable_topk_indices(sub, idx)]
    return [posts[i] for i in top_indices]
