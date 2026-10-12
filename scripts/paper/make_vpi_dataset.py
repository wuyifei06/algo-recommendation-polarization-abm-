# -*- coding: utf-8 -*-
"""为论文修订准备统一的 VPI 数据。

三个实验：
  rule_main_d4          500 runs, N=200 T=200  → 确定性重放（已有产物，直接读）
  rule_small_d4          50 runs, N=30  T=30   → 确定性重放（本次补做）
  llm_gradient_noleak   160 runs, N=30  T=30   → 从末步 checkpoint 的帖池离线反算

口径统一：VPI_fixed(t) = var( { p.opinion | p.created_step == t } )
输出长表 _work/paper_rev/vpi_dataset.csv
"""
from __future__ import annotations
import csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

ROOT = _PROJ_ROOT
RES = os.path.join(ROOT, "results")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vpi_dataset.csv")


# ---------- rule_small 重放 ----------
def replay_small(task):
    run_id, alpha, gamma, seed = task
    sys.path.insert(0, ROOT)
    import numpy as np
    from config import conf
    from model.model import SimulationModel
    m = SimulationModel(seed=seed, N=30, steps=30, alpha=float(alpha), gamma=float(gamma))
    rows = []
    for t in range(1, 31):
        m.step(t)
        mt = m.metrics_time_series[-1]
        ops = [p.opinion for p in m.posts if p.created_step == t]
        rows.append(dict(experiment="rule_small", run_id=run_id, alpha=float(alpha),
                         gamma=float(gamma), step=t,
                         PI=mt["PI"], VPI_old=mt["VPI"],
                         VPI_fixed=float(np.var(ops)) if ops else 0.0,
                         n_poster=len(ops)))
    return rows


def replay_rule_main_read():
    """rule_main 的重放产物已存在，直接读并规范化列名。"""
    src = os.path.join(ROOT, "results_direction", "_VPI_FIX_20261010", "rule_main_d4", "逐run时间序列")
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "rule_run_*.csv"))):
        rid = int(os.path.basename(f)[len("rule_run_"):-4])
        for r in csv.DictReader(open(f, encoding="utf-8")):
            rows.append(dict(experiment="rule_main", run_id=rid,
                             alpha=float(r["alpha"]), gamma=float(r["gamma"]),
                             step=int(r["step"]), PI=float(r["PI"]),
                             VPI_old=float(r["PI"]),   # 已验证旧 VPI ≡ PI
                             VPI_fixed=float(r["VPI"]),
                             n_poster=-1))
    return rows


def rebuild_llm():
    """llm_gradient_noleak：从末步 checkpoint 的帖池按 created_step 分组反算 VPI。"""
    cps = glob.glob(os.path.join(RES, "llm_gradient_noleak", "checkpoints", "*.checkpoint.json"))
    cps = [c for c in cps if "ABANDONED" not in c and "VOIDED" not in c]
    by_run = {}
    for p in cps:
        base = os.path.basename(p)
        rid = base[len("llm_robust_run_"):-len(".checkpoint.json")]
        if not rid.isdigit():
            continue
        try:
            d = json.loads(open(p, encoding="utf-8").read())
        except Exception:
            continue
        step = int(d.get("current_step") or 0)
        prev = by_run.get(int(rid))
        if prev is None or step > prev[0]:
            by_run[int(rid)] = (step, d)
    print(f"  llm_gradient_noleak: checkpoint {len(cps)} 个, 覆盖 run {len(by_run)} 个")

    # 读逐步 PI（runs/*.csv）
    pi_by_run = {}
    for f in sorted(glob.glob(os.path.join(RES, "llm_gradient_noleak", "runs", "llm_robust_run_*.csv"))):
        if f.endswith("_agents_final.csv"):
            continue
        rid = int(os.path.basename(f)[len("llm_robust_run_"):-4])
        pi_by_run[rid] = {int(r["step"]): r for r in csv.DictReader(open(f, encoding="utf-8"))}

    rows = []
    for rid, (step, d) in by_run.items():
        groups = {}
        for q in d.get("posts") or []:
            groups.setdefault(int(q["created_step"]), []).append(float(q["opinion"]))
        for t in range(1, step + 1):
            src = pi_by_run.get(rid, {}).get(t)
            if src is None:
                continue
            ops = groups.get(t, [])
            rows.append(dict(experiment="llm_ds", run_id=rid,
                             alpha=float(src["alpha"]), gamma=float(src["gamma"]),
                             step=t, PI=float(src["PI"]),
                             VPI_old=float(src["VPI"]),
                             VPI_fixed=float(np.var(ops)) if ops else 0.0,
                             n_poster=len(ops)))
    return rows


def main():
    # rule_small
    summ = list(csv.DictReader(open(os.path.join(RES, "rule_small_d4", "rule_small_summary.csv"), encoding="utf-8")))
    tasks = sorted((int(r["run_id"]), float(r["alpha"]), float(r["gamma"]), int(r["seed"])) for r in summ)
    print(f"  rule_small: 重放 {len(tasks)} runs …")
    rows_small = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for rs in ex.map(replay_small, tasks):
            rows_small += rs
    print(f"  rule_small: 完成 {len(rows_small)} 行")

    rows_main = replay_rule_main_read()
    print(f"  rule_main: 读入 {len(rows_main)} 行")
    rows_llm = rebuild_llm()
    print(f"  llm_ds: 反算 {len(rows_llm)} 行")

    allrows = rows_main + rows_small + rows_llm
    df = pd.DataFrame(allrows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n总行数 {len(df)} → {OUT}")

    # 快速核对
    print("\n各实验末步汇总（VPI_fixed vs PI）:")
    last = df[df.groupby(["experiment", "run_id"])["step"].transform("max") == df["step"]]
    for exp, g in last.groupby("experiment"):
        r = np.corrcoef(g["PI"], g["VPI_fixed"])[0, 1]
        print(f"  {exp:12s} runs={len(g):4d}  PI={g['PI'].mean():.4f}  VPI_fix={g['VPI_fixed'].mean():.4f}  "
              f"VPI_old==PI? {np.allclose(g['VPI_old'], g['PI'])}  r={r:.4f}")


if __name__ == "__main__":
    main()
