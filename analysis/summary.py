"""简单的汇总统计工具。

提供均值、标准差、中位数以及 95% 置信区间的计算方法，便于对
不同实验条件（alpha × gamma）进行分组统计。
"""
import pandas as pd
import numpy as np


def summarize_results(df: pd.DataFrame, group_cols: list, metric: str) -> pd.DataFrame:
    """对 DataFrame 按 group_cols 分组并对指定 metric 计算统计量。

    返回包含 mean, std, median, count, ci95 的 DataFrame。
    """
    def ci95(x):
        m = np.mean(x)
        se = np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return 1.96 * se

    agg = df.groupby(group_cols)[metric].agg(["mean", "std", "median", "count"]).reset_index()
    agg["ci95"] = df.groupby(group_cols)[metric].apply(lambda x: ci95(x)).values
    return agg
