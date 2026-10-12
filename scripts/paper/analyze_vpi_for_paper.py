# -*- coding: utf-8 -*-
"""为论文修订计算 VPI 全部统计量。"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "vpi_dataset.csv"))

last = df[df.groupby(["experiment", "run_id"])["step"].transform("max") == df["step"]].copy()
last["PI_minus_VPI"] = last["PI"] - last["VPI_fixed"]

print("=" * 96)
print("【1】末步：各 α 条件的 PI 与 VPI_fixed")
for exp, label in [("rule_main", "Rule 主实验 (N=200,T=200,R=50)"),
                   ("rule_small", "Rule 受限规模 (N=30,T=30,R=5)"),
                   ("llm_ds", "LLM 主实验 DeepSeek (N=30,T=30,R=10)")]:
    g = last[last.experiment == exp]
    print(f"\n  {label}")
    print(f"  {'γ':>4} {'α':>5} {'n':>4} {'PI':>8} {'VPI':>8} {'PI−VPI':>8} {'r(PI,VPI)':>10}")
    for gm in sorted(g.gamma.unique()):
        for al in sorted(g[g.gamma == gm].alpha.unique()):
            s = g[(g.gamma == gm) & (g.alpha == al)]
            r = np.corrcoef(s["PI"], s["VPI_fixed"])[0, 1] if len(s) > 2 and s["PI"].std() > 0 else np.nan
            print(f"  {gm:>4} {al:>5} {len(s):>4} {s['PI'].mean():>8.4f} {s['VPI_fixed'].mean():>8.4f} "
                  f"{(s['PI'].mean()-s['VPI_fixed'].mean()):>+8.4f} {r:>10.3f}")

print("\n" + "=" * 96)
print("【2】VPI 的 α 剂量—响应（标准化 β，末步）")
print(f"  {'实验':16s} {'γ':>4} {'PI β':>9} {'PI R²':>7} {'VPI β':>9} {'VPI R²':>7} {'VPI p':>10}")
for exp, label in [("rule_main", "Rule 主"), ("rule_small", "Rule 小"), ("llm_ds", "LLM DeepSeek")]:
    for gm in sorted(last[last.experiment == exp].gamma.unique()):
        g = last[(last.experiment == exp) & (np.isclose(last.gamma, gm))]
        out = []
        for col in ["PI", "VPI_fixed"]:
            x = ((g["alpha"] - g["alpha"].mean()) / g["alpha"].std(ddof=1)).to_numpy(float)
            y = ((g[col] - g[col].mean()) / g[col].std(ddof=1)).to_numpy(float)
            m = sm.OLS(y, sm.add_constant(x)).fit()
            out.append((m.params[1], m.rsquared, m.pvalues[1]))
        print(f"  {label:16s} {gm:>4} {out[0][0]:>+9.4f} {out[0][1]:>7.3f} "
              f"{out[1][0]:>+9.4f} {out[1][1]:>7.3f} {out[1][2]:>10.3g}")

print("\n" + "=" * 96)
print("【3】全程 PI−VPI 的范围（逐 run 逐步）")
df["d"] = df["PI"] - df["VPI_fixed"]
for exp, label in [("rule_main", "Rule 主"), ("rule_small", "Rule 小"), ("llm_ds", "LLM DeepSeek")]:
    g = df[df.experiment == exp]
    d = g["d"]
    print(f"  {label:16s} 均值={d.mean():+.4f}  范围=[{d.min():+.4f}, {d.max():+.4f}]  "
          f"全程为负? {(d < 0).all()}  为负比例={100*(d<0).mean():.1f}%")
    # 分时段（前/中/后三分之一）
    T = int(g["step"].max())
    for name, lo, hi in [("前 1/3", 1, T//3), ("中 1/3", T//3+1, 2*T//3), ("后 1/3", 2*T//3+1, T)]:
        s = g[(g.step >= lo) & (g.step <= hi)]["d"]
        print(f"      {name}: 均值={s.mean():+.4f}")

print("\n" + "=" * 96)
print("【4】LLM 臂：表达者比例与立场差（按 α）")
g = last[last.experiment == "llm_ds"]
for al in sorted(g.alpha.unique()):
    s = g[np.isclose(g.alpha, al)]
    print(f"  α={al:<5} PI={s['PI'].mean():.4f}  VPI={s['VPI_fixed'].mean():.4f}  "
          f"PI−VPI={s['d'].mean():+.4f}  (n={len(s)})")

print("\n" + "=" * 96)
print("【5】与论文原文的交叉核对")
gl = df[df.experiment == "llm_ds"]
glast = gl[gl.step == gl.step.max()]
print(f"  论文原句：DeepSeek T=30 末步 PI−VPI_valid = −0.052，全程范围 [−0.101, −0.018]")
print(f"  本次全量：末步 PI−VPI 均值 = {glast['d'].mean():+.4f}")
dall = gl["d"]
print(f"            全程范围 = [{dall.min():+.4f}, {dall.max():+.4f}]")
print(f"  论文原句：VPI（旧）与 PI 完全相同，max|VPI−PI|=0")
print(f"  本次核对：{np.allclose(df['VPI_old'], df['PI'])}  (max diff={np.abs(df['VPI_old']-df['PI']).max():.2e})")
