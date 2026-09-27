"""Post 定义。

每条 Post 表示一次表达（或内容单元），包含发表者、情绪、观点和
互动强度（engagement）。engagement 在模型中会被归一化处理。
"""
from dataclasses import dataclass


@dataclass
class Post:
    post_id: int
    author_id: int
    opinion: float
    emotion: float
    engagement: float
    created_step: int

    def to_dict(self):
        """将 Post 转为字典，便于保存到 CSV/JSON。"""
        return {
            "post_id": self.post_id,
            "author_id": self.author_id,
            "opinion": self.opinion,
            "emotion": self.emotion,
            "engagement": self.engagement,
            "created_step": self.created_step,
        }
