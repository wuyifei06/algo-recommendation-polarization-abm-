"""实验参数配置。支持通过命令行覆盖。

本模块采用统一的 `.env` 入口，所有 LLM 相关参数都在这里读取。
如果用户指定使用 LLM 实验，则必须强制提供真实的模型 endpoint/key/model，
不能回退到本地 stub。
"""
from dataclasses import dataclass
from typing import List
import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()


@dataclass
class Config:
    # Rule ABM
    N_AGENTS: int = 200
    STEPS: int = 200
    REPEATS: int = 50
    ALPHAS: List[float] = None
    GAMMAS: List[float] = None
    TOP_K: int = 10
    # ⚠️ 2026-10-09 设计决策（最终）：**K=10 固定，定位为"缩小版邓巴数"**——
    # 个体信息消费容量是认知常数，不随群体规模 N 变化（真实平台中用户也只刷一屏）。
    # 因此全部规模（N=30/100/200）与两个主体（Rule/LLM）统一 K=10。
    # 备查：K∝N（比例化）曾被探索以分离"feed 覆盖率稀释"，结论为**无现实对应物**，
    # 不作设计；如需再调查可 TOP_K_MODE=auto + TOP_K_N_RATIO 开启（默认 fixed）。
    TOP_K_N_RATIO: float = float(os.getenv("TOP_K_N_RATIO", "0.3333333333333333"))
    TOP_K_MODE: str = os.getenv("TOP_K_MODE", "fixed")
    MAX_POSTS: int = 5000
    HOMOPHILY: float = 1.0
    RHO: float = 0.2
    MU: float = 0.03
    H: float = 1.0

    # ── 方向版提示词开关（2026-11-25）────────────────────────────────
    # True ：给 LLM 主体补上「推荐内容平均立场」（带符号，[-1,1]），与规则主体
    #        拿到的 o_feed = Σp.opinion/K 信息集对齐 —— 这是论文采用的口径。
    # False：原版（只给无符号距离 mean|o_i−p_j|/2），用于复现旧数据（LD 臂零效应）。
    # 环境变量 LLM_FEED_SIGNED=0 可切回原版。
    LLM_SIGNED_FEED: bool = True

    # LLM experiment
    LLM_N_AGENTS: int = 30
    LLM_STEPS: int = 30
    # LLM robustness design: 5 alpha × 2 gamma × 5 actual repeats.
    LLM_REPEATS: int = 5
    LLM_ALPHAS: List[float] = None
    LLM_GAMMAS: List[float] = None

    # LLM cost guard defaults
    MAX_LLM_CALLS: int = 1000
    MAX_ESTIMATED_COST: float = 100.0

    # LLM provider / model configuration
    LLM_PROVIDER: str = ""
    LLM_API_KEY: str = ""
    LLM_API_ENDPOINT: str = ""
    LLM_MODEL: str = ""
    LLM_MODEL_NAME: str = ""
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT_SECONDS: int = 20
    LLM_MAX_RETRIES: int = 3
    # 同一时间步内并发请求多个 agent；不改变 agent 数量、步数或调用总量。
    LLM_PARALLEL_WORKERS: int = 4

    # seeds
    BASE_SEED: int = 12345

    # expression probability params
    B0: float = -1.0
    BE: float = 1.5
    BA: float = 0.5

    # 沉默的螺旋 / 跨立场沉默（规则臂扩展设定，2026-09-21）
    # ⚠️ 默认 **不启用**：模型构造时 spiral_c/spiral_m 默认 0.0，baseline 逐位不变。
    #    只有显式传入（driver 的 --c-grid / --spiral-m）才会打开。
    SPIRAL_C_GRID: List[float] = None      # 敏感性网格：d 对表达的抑制强度
    SPIRAL_M_DEFAULT: float = 0.5          # 少数派身份惩罚的默认强度
    # LLM 兜底路径沿用的取值。0.6 与历史上硬编码的 (1-0.60*d) 完全一致，
    # 因此把兜底统一到共享函数后行为逐位不变。
    SPIRAL_C_DEFAULT: float = 0.6

    def __post_init__(self):
        env = load_env()
        self.LLM_PROVIDER = (self.LLM_PROVIDER or env.get("LLM_PROVIDER") or "dashscope").strip().lower()
        self.LLM_API_KEY = (self.LLM_API_KEY or env.get("LLM_API_KEY") or "").strip()
        self.LLM_API_ENDPOINT = (self.LLM_API_ENDPOINT or env.get("LLM_API_ENDPOINT") or "").strip()
        self.LLM_MODEL = (self.LLM_MODEL or env.get("LLM_MODEL") or env.get("LLM_MODEL_NAME") or "").strip()
        self.LLM_MODEL_NAME = (self.LLM_MODEL_NAME or self.LLM_MODEL).strip()
        _signed = str(env.get("LLM_FEED_SIGNED", os.getenv("LLM_FEED_SIGNED", "1"))).strip().lower()
        self.LLM_SIGNED_FEED = _signed not in ("0", "false", "no", "off")

        self.LLM_TEMPERATURE = float(self.LLM_TEMPERATURE if self.LLM_TEMPERATURE != 0.7 else env.get("LLM_TEMPERATURE", 0.7))
        self.LLM_TIMEOUT_SECONDS = int(self.LLM_TIMEOUT_SECONDS if self.LLM_TIMEOUT_SECONDS != 20 else env.get("LLM_TIMEOUT_SECONDS", 20))
        self.LLM_MAX_RETRIES = int(self.LLM_MAX_RETRIES if self.LLM_MAX_RETRIES != 3 else env.get("LLM_MAX_RETRIES", 3))
        self.LLM_PARALLEL_WORKERS = max(
            1,
            int(self.LLM_PARALLEL_WORKERS if self.LLM_PARALLEL_WORKERS != 4 else env.get("LLM_PARALLEL_WORKERS", 4)),
        )

        if self.ALPHAS is None:
            self.ALPHAS = [0, 0.5, 1.0, 1.5, 2.0]
        if self.GAMMAS is None:
            self.GAMMAS = [0.1, 1.0]
        if self.LLM_ALPHAS is None:
            self.LLM_ALPHAS = [0, 0.5, 1.0, 1.5, 2.0]
        if self.LLM_GAMMAS is None:
            self.LLM_GAMMAS = [0.1, 1.0]
        if self.SPIRAL_C_GRID is None:
            # c=0 即 baseline（同一网格内自带对照，便于配对比较）
            self.SPIRAL_C_GRID = [0.0, 0.3, 0.6, 1.0, 1.5]

        # LLM 实验的规模由上面的配置字段决定。不要在这里静默覆盖，
        # 否则断点恢复时可能出现“结果文件属于另一组实验参数”的问题。


def load_env():
    """读取统一 LLM 配置，只使用 .env / 环境变量，不做 stub 回退。"""
    provider = (os.getenv("LLM_PROVIDER") or "dashscope").strip().lower()
    return {
        "LLM_API_KEY": (os.getenv("LLM_API_KEY") or "").strip(),
        "LLM_API_ENDPOINT": (os.getenv("LLM_API_ENDPOINT") or "").strip(),
        "LLM_PROVIDER": provider,
        "LLM_MODEL": (os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NAME") or "").strip(),
        "LLM_MODEL_NAME": (os.getenv("LLM_MODEL_NAME") or os.getenv("LLM_MODEL") or "").strip(),
        "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "LLM_TIMEOUT_SECONDS": int(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        "LLM_MAX_RETRIES": int(os.getenv("LLM_MAX_RETRIES", "3")),
        "LLM_PARALLEL_WORKERS": int(os.getenv("LLM_PARALLEL_WORKERS", "4")),
    }


def validate_llm_config():
    """确保 LLM 实验必须使用真实模型配置。"""
    env = load_env()
    missing = []
    if not env.get("LLM_API_KEY"):
        missing.append("LLM_API_KEY")
    if not env.get("LLM_API_ENDPOINT"):
        missing.append("LLM_API_ENDPOINT")
    if not env.get("LLM_MODEL"):
        missing.append("LLM_MODEL")

    if missing:
        raise ValueError(
            "LLM mode requires real model configuration. Missing: "
            + ", ".join(missing)
            + ". Please set them in .env before running LLM experiments."
        )
    return True


conf = Config()


__all__ = ["conf", "load_env", "validate_llm_config"]
