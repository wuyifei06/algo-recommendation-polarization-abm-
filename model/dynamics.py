"""情绪与意见更新、表达概率计算。

包含情绪更新公式、意见更新公式和表达概率（sigmoid）。
注：情绪被裁剪到 [0,1]，意见裁剪到 [-1,1]。
"""
from model.agent import Agent
import math


def update_emotion(agent: Agent, e_feed: float, rho: float) -> float:
    """按照公式更新情绪：E(t+1)=E(t)+rho*lambda*(E_feed-E(t))，并裁剪到 [0,1]。"""
    lam = agent.lambda_sensitivity
    new = agent.emotion + rho * lam * (e_feed - agent.emotion)
    return min(1.0, max(0.0, new))


def update_opinion(agent: Agent, o_feed: float, mu: float) -> float:
    """按照公式更新意见：O(t+1)=O(t)+mu*(O_feed-O(t))，并裁剪到 [-1,1]。"""
    new = agent.opinion + mu * (o_feed - agent.opinion)
    return min(1.0, max(-1.0, new))


def sigmoid(x: float) -> float:
    """Sigmoid 函数，返回 (0,1) 之间的概率值。"""
    return 1.0 / (1.0 + math.exp(-x))


def expression_probability(agent: Agent, b0: float, bE: float, bA: float) -> float:
    """基于情绪与活动性的表达概率计算：P= sigmoid(b0 + bE*emotion + bA*activity)。"""
    x = b0 + bE * agent.emotion + bA * agent.activity
    return float(sigmoid(x))


def rule_expression_probability(
    agent: Agent,
    mean_stance_distance: float,
    feed_mean_opinion: float,
    b0: float,
    bE: float,
    bA: float,
    c: float = 0.0,
    m: float = 0.0,
) -> float:
    """规则臂表达概率（含可选的沉默的螺旋扩展项）。

    基线等价性（重要）
    ------------------
    ``c = 0.0`` 且 ``m = 0.0`` 时，本函数与 ``expression_probability`` **逐位等价**。
    因此只要调用方不显式打开扩展，规则臂的轨迹与既有 baseline 完全一致。

    两个扩展项（2026-09-21 新增，默认关闭）
    --------------------------------------
    - ``c > 0``：跨立场沉默。异见距离越大越不愿表达，对应 Noelle-Neumann
      沉默的螺旋中"感知到与自己相左的意见气候 → 收敛表达"这一环。
      这里用的是 agent **自己能看到**的量：Top-K 推荐中的平均立场距离
      ``mean_stance_distance``（= mean|自身立场 − 帖子立场| / 2，与 LLM 臂
      prompt 里给的 ``d`` 同口径）。
    - ``m > 0``：少数派惩罚。当自身立场与所看到的 feed 平均立场符号相反时，
      表达概率按 ``(1 - m)`` 衰减。同样只用 agent 可感知的 feed 气候，
      不引入任何全局信息（否则违反"真人能不能感觉到"的红线）。

    参数
    ----
    mean_stance_distance : 本步 Top-K 曝光集合相对自身的平均立场距离，∈[0,1]
    feed_mean_opinion    : 本步 Top-K 曝光集合的立场均值（即 o_feed）
    """
    prob = sigmoid(b0 + bE * agent.emotion + bA * agent.activity)
    if c > 0.0:
        prob *= max(0.0, 1.0 - c * float(mean_stance_distance))
    if m > 0.0:
        minority = 1.0 if (float(agent.opinion) * float(feed_mean_opinion)) < 0.0 else 0.0
        prob *= 1.0 - m * minority
    return float(min(1.0, max(0.0, prob)))
