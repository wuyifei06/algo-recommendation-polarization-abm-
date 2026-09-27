"""Agent 定义。

每个 Agent 保持最小必要状态：`id, opinion, emotion, agent_type, activity`。
包含历史记录字段用于后续扩展（交互历史与暴露历史），并保留
`lambda_sensitivity` 表示情绪敏感性。不要在 dataclass 中放置
大型不可序列化字段，以保持结果可保存为 CSV/JSON。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Agent:
    id: int
    opinion: float  # [-1,1]
    emotion: float  # [0,1]
    agent_type: str  # 'extreme','moderate','swing'
    activity: float  # 0-1
    expression_probability: float = 0.0
    interaction_history: List[int] = field(default_factory=list)
    exposure_history: List[int] = field(default_factory=list)
    lambda_sensitivity: float = 0.5  # 情绪敏感性（用于情绪更新公式）

    def to_dict(self):
        """将 Agent 转为字典，方便保存或打印。

        返回值只包含关键字段，避免暴露内部大型历史数组。
        """
        return {
            "id": self.id,
            "opinion": self.opinion,
            "emotion": self.emotion,
            "agent_type": self.agent_type,
            "activity": self.activity,
        }
