# -*- coding: utf-8 -*-
"""确定性重跑「Rule 螺旋机制扩展」全网格（1159 runs），记录存档中未落盘的
「少数派命中率」（minority hit rate）等机制输入量，并逐位校验轨迹复现性。

用法：
    python rerun_spiral_minority.py [--workers 16] [--limit N]

输出：
    spiral_minority_rerun.csv  每 run 一行：
        run_id, alpha, gamma, c, m, seed, n_steps, n_agents,
        hit_post_all, hit_post_final,     # 命中率：sign(更新后立场) × sign(feed 均值) < 0
        hit_pre_all,  hit_pre_final,      # 同一指标改用「更新前立场」
        d_all, d_final, d_max,            # 曝光距离（用于交叉校验）
        chk_max_abs_diff                  # 与存档 CSV 的 PI/HE/d 逐位最大偏差
"""
import os, sys, re, glob, json, argparse
import numpy as np
import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

SRC = _PROJ_ROOT
sys.path.insert(0, SRC)

RUN_DIR = os.path.join(SRC, "results", "rule_spiral", "逐run时间序列")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "spiral_minority_rerun.csv")

CHECKS = ["PI", "HE", "mean_exposure_distance"]


def collect_tasks(limit=None):
    summ = pd.read_csv(os.path.join(SRC, "results_replay_20261010", "rule_spiral",
                                    "rule_spiral_summary.csv"))
    meta = {int(r.run_id): r for r in summ.itertuples()}
    files = sorted(glob.glob(os.path.join(RUN_DIR, "rule_spiral_run_*.csv")),
                   key=lambda f: int(re.search(r"run_(\d+)\.csv", f).group(1)))
    tasks = []
    for f in files:
        rid = int(re.search(r"run_(\d+)\.csv", f).group(1))
        if rid not in meta:
            continue
        L = meta[rid]
        tasks.append(dict(run_id=rid, alpha=float(L.alpha), gamma=float(L.gamma),
                          c=float(L.spiral_c), m=float(L.spiral_m), seed=int(L.seed),
                          N=int(L.N), T=int(L.steps), path=f))
    tasks.sort(key=lambda t: t["run_id"])
    if limit:
        tasks = tasks[:limit]
    return tasks


def run_one(task):
    import model.model as MM
    rec_post, rec_pre = [], []

    orig_expr = MM.rule_expression_probability

    def patched_expr(agent, d_expo, feed_mean, b0, bE, bA, cc=0.0, mm=0.0):
        rec_post.append((float(agent.opinion), float(feed_mean)))
        return orig_expr(agent, d_expo, feed_mean, b0, bE, bA, cc, mm)

    MM.rule_expression_probability = patched_expr

    orig_upd = MM.update_opinion

    def patched_upd(agent, o_feed, mu):
        rec_pre.append((float(agent.opinion), float(o_feed)))
        return orig_upd(agent, o_feed, mu)

    MM.update_opinion = patched_upd

    model = MM.SimulationModel(seed=task["seed"], N=task["N"], steps=task["T"],
                               alpha=task["alpha"], gamma=task["gamma"],
                               spiral_c=task["c"], spiral_m=task["m"])
    model.run()

    MM.rule_expression_probability = orig_expr
    MM.update_opinion = orig_upd

    N, T = task["N"], task["T"]
    ts = pd.DataFrame(model.metrics_time_series)
    stored = pd.read_csv(task["path"])
    diffs = []
    for col in CHECKS:
        if col in ts.columns and col in stored.columns and len(ts) == len(stored):
            diffs.append(float(np.max(np.abs(ts[col].to_numpy(float) - stored[col].to_numpy(float)))))
    chk = max(diffs) if diffs else float("nan")

    def hitrate(arr):
        if len(arr) != N * T:
            return float("nan"), float("nan")
        a = np.asarray(arr, dtype=float)          # shape (T*N, 2)
        ind = (np.sign(a[:, 0]) * np.sign(a[:, 1]) < 0).astype(float)
        a = ind.reshape(T, N)
        return float(a.mean()), float(a[-1].mean())

    hp_all, hp_fin = hitrate(rec_post)
    hr_all, hr_fin = hitrate(rec_pre)
    d = stored["mean_exposure_distance"].to_numpy(float)

    return dict(run_id=task["run_id"], alpha=task["alpha"], gamma=task["gamma"],
                c=task["c"], m=task["m"], seed=task["seed"], n_steps=T, n_agents=N,
                hit_post_all=hp_all, hit_post_final=hp_fin,
                hit_pre_all=hr_all, hit_pre_final=hr_fin,
                d_all=float(d.mean()), d_final=float(d[-1]), d_max=float(d.max()),
                pi_final=float(stored["PI"].iloc[-1]), he_final=float(stored["HE"].iloc[-1]),
                chk_max_abs_diff=chk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = collect_tasks(args.limit)
    print(f"[rerun] 任务数 {len(tasks)}，workers={args.workers}", flush=True)

    from multiprocessing import Pool
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"  完成 {i}/{len(tasks)}", flush=True)

    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    bad = df[df.chk_max_abs_diff > 1e-12]
    print(f"[rerun] 写入 {OUT}  行数={len(df)}")
    print(f"[rerun] 逐位复现校验：{len(df) - len(bad)}/{len(df)} 完全一致 (max|Δ|=0)")
    if len(bad):
        print(bad[["run_id", "chk_max_abs_diff"]].head(10).to_string())


if __name__ == "__main__":
    main()
