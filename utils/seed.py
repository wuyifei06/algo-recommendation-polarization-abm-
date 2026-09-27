import random
import numpy as np


def set_seed(seed: int):
    """设置随机种子，保证实验可复现。

    同时设置 Python 的 `random` 和 NumPy 的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
