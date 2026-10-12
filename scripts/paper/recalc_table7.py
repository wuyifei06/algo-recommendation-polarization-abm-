# -*- coding: utf-8 -*-
"""表7（沉默的螺旋机制输入）全量重算：HE / d 均值 / d 最大 / 少数派命中率 / 1−c·d。

数据源：
  - 指标：results_replay_20261010/rule_spiral/rule_spiral_summary.csv（1159 runs，末步）
  - 少数派命中率：spiral_minority_rerun.csv（确定性重跑日志，逐位复现校验通过）

口径：全部 1159 个 run 的末步；按 α 汇总。命中率两种定义都报告：
  post：sign(更新后立场)×sign(feed 均值) < 0（与模型实现一致）
  pre ：sign(更新前立场)×sign(feed 均值) < 0
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_BASE_ROOT = _os.getenv("ABM_BASE_ROOT", r"D:\21130\Desktop\大创过程")
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
_VENV_PY = _os.getenv("ABM_VENV_PY", r"C:\Users\21130\.workbuddy\binaries\python\envs\default\Scripts\python.exe")
# -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY = _PROJ_ROOT + r"\results_replay_20261010\rule_spiral\rule_spiral_summary.csv"
MINORITY = os.path.join(HERE, "spiral_minority_rerun.csv")

D = pd.read_csv(REPLAY)
M = pd.read_csv(MINORITY)
print("指标表：%d runs；命中率表：%d runs；逐位复现：%d/%d max|Δ|=%.1e" % (
    len(D), len(M), (M.chk_max_abs_diff.fillna(0) <= 1e-12).sum(), len(M),
    float(np.nanmax(M.chk_max_abs_diff))))
print()

rows = []
print("%-6s %6s %9s %9s %11s %11s %9s" % ("α", "n", "HE", "d均值", "d最大", "命中(post)", "命中(pre)"))
for a in sorted(D.alpha.unique()):
    s = D[np.isclose(D.alpha, a)]
    m = M[np.isclose(M.alpha, a)]
    hp = m.hit_post_final.mean()
    hr = m.hit_pre_final.mean()
    rows.append((a, len(s), s.HE.mean(), s.mean_exposure_distance.mean(),
                 s.mean_exposure_distance.max(), hp, hr))
    print("%-6s %6d %9.4f %9.5f %11.4f %11.4f %9.4f" % (a, len(s), s.HE.mean(),
          s.mean_exposure_distance.mean(), s.mean_exposure_distance.max(), hp, hr))
print()

# 末步口径（全池化）
p_all = M.hit_post_final.mean()
print("5 档 α 的末步命中率（post）：", ["%.4f" % r[5] for r in rows], " 池化 = %.4f" % p_all)
print("最大档 = %.4f" % max(r[5] for r in rows))
print("非零档数（按 run 均值 >0）= %d / %d" % (sum(1 for r in rows if r[5] > 0), len(rows)))
print("非零档数（池化 agent-step >0）= %d / %d（见下方逐档计数）" % (
    sum(1 for a in sorted(D.alpha.unique())
        if M[M.alpha == a].hit_post_final.sum() > 0), len(rows)))
for a in sorted(D.alpha.unique()):
    m = M[np.isclose(M.alpha, a)]
    nz = int((m.hit_post_final > 0).sum())
    print("   α=%.1f：%d/%d 个 run 末步命中率 >0；run 均值 %.5f" % (a, nz, len(m), m.hit_post_final.mean()))
print()

# 回归与倍数
x = D.alpha.to_numpy(float); y = D.mean_exposure_distance.to_numpy(float)
X = np.column_stack([np.ones(len(x)), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
r = y - X @ b; n = len(x)
se = np.sqrt(np.diag(((r @ r) / (n - 2)) * np.linalg.pinv(X.T @ X)))
r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
print("d~α：斜率 %+.6f  SE %.6f  t=%.2f  R²=%.3f  n=%d" % (b[1], se[1], b[1] / se[1], r2, n))
d0 = D[np.isclose(D.alpha, 0)].mean_exposure_distance.mean()
d2 = D[np.isclose(D.alpha, 2)].mean_exposure_distance.mean()
print("d(α=0)=%.6f  d(α=2)=%.6f  倍数=%.1f" % (d0, d2, d2 / d0))
dmax = D.mean_exposure_distance.max()
print("最坏个体（c=2）：1−2·dmax = %.4f → 削减 %.2f%%" % (1 - 2 * dmax, 200 * dmax))
print("平均（α=2，c=2）：1−2·d = %.4f → 削减 %.2f%%" % (1 - 2 * d2, 200 * d2))
print()

# ===== 表7 行（按论文表格精度）=====
print("=== 表7 建议行（α=0 / 1 / 2）===")
for a, n_, he, dm, dx, hp, hr in rows:
    if a in (0.0, 1.0, 2.0):
        print("| %.1f | %.4f | %.5f | %.4f | %.3f | %.3f | ≡ 1 |" % (a, he, dm, dx, hp, 1 - 2 * dx))
print()

# ===== 跨种子 SR 95% CI 半宽 =====
hws = []
for (a, g), s in D.groupby(["alpha", "gamma"]):
    hws.append(1.96 * s.SR.std(ddof=1) / np.sqrt(len(s)))
print("SR 95%CI 半宽：min ±%.4f  max ±%.4f" % (min(hws), max(hws)))
