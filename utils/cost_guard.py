from typing import Optional
import warnings
import threading


class CostGuard:
    """简单的成本保护器，用于限制 LLM 调用次数与预计费用。

    用法示例：在每次调用前检查 `allow_call(est_cost)`，若返回 False 则停止
    后续调用或切换到本地 stub。该实现为轻量级估算，适合实验流程控制。
    """
    def __init__(
        self,
        max_calls: int = 0,
        max_cost: float = 300.0,
        warning_cost: float = 250.0,
        stop_cost: float = 280.0,
    ):
        self.max_calls = max_calls
        self.max_cost = max_cost
        self.warning_cost = warning_cost
        self.stop_cost = stop_cost
        self.calls = 0
        self.estimated_cost = 0.0
        self.warning_emitted = False
        self.stopped = False
        self._lock = threading.Lock()

    def allow_call(self, est_cost: float = 0.01) -> bool:
        """检查是否允许本次调用；若允许则计入调用次数与估算费用并返回 True。"""
        with self._lock:
            if self.max_calls and self.calls + 1 > self.max_calls:
                return False
            projected = self.estimated_cost + est_cost
            if projected >= self.max_cost:
                raise RuntimeError(
                    f"LLM estimated cost reached MAX_LLM_COST={self.max_cost:.2f}."
                )
            if projected >= self.stop_cost:
                self.stopped = True
                return False
            # 批准本次调用并更新计数与费用
            self.calls += 1
            self.estimated_cost += est_cost
            if self.estimated_cost >= self.warning_cost and not self.warning_emitted:
                self.warning_emitted = True
                warnings.warn(
                    f"LLM estimated cost reached warning threshold {self.warning_cost:.2f}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return True

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "estimated_cost": self.estimated_cost,
                "max_cost": self.max_cost,
                "warning_cost": self.warning_cost,
                "stop_cost": self.stop_cost,
                "stopped": self.stopped,
            }
