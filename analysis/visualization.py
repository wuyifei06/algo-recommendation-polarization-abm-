"""生成基础图表（使用 matplotlib）。

当前实现包含示例绘图逻辑，可基于 `analysis.summary` 的输出
生成不同 alpha 条件下的曲线图。保存路径为 `figures/`。
"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(summary_df: pd.DataFrame, out_dir: str = "figures"):
    """根据 summary DataFrame 绘制示例图并保存到 out_dir。"""
    os.makedirs(out_dir, exist_ok=True)
    # 示例：绘制 PI 随 alpha 的变化曲线
    if "alpha" in summary_df.columns and "mean" in summary_df.columns:
        for metric in ["PI"]:
            sub = summary_df[summary_df.get("metric", "PI")==metric] if "metric" in summary_df.columns else summary_df
            plt.figure()
            plt.plot(sub["alpha"], sub["mean"], marker="o")
            plt.xlabel("alpha")
            plt.ylabel(metric)
            plt.title(f"{metric} by alpha")
            plt.savefig(os.path.join(out_dir, f"{metric}_by_alpha.png"))
            plt.close()
