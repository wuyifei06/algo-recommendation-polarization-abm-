"""规则臂四批次确定性重放（2026-10-10）。

目的：让论文规则臂全部数据出自本轮实验周期。Rule 主体为固定种子确定性仿真
（random.seed(seed)，seed = 12345 + run_id），重放结果与原始运行逐位一致。

批次与原始数据来源：
  1. rule_main_d4   500 runs  N=200 T=200  5a x 2g x R=50   <- run_rule_main_fixed.py
  2. rule_small_d4   50 runs  N=30  T=30  5a x 2g x R=5    <- experiment_rule_small.py
  3. rule_spiral   1159 runs  N=200 T=200  c{0,0.5}=500 各 + c1.0 前 159（提前终止）
  4. rule_structure 9 x 200 runs  变体: base/comp10/comp40/H000/H020/H050/K20/N500/T400
     每变体 5a x 2g x R=20, seed = 12345 + run_id, run_id = (ai*2+gi)*20 + r

变体参数（已用 base run0 == rule_main run0 逐位一致验证）：
  H000/H020/H050 -> conf.H = 0.0/0.2/0.5
  K20            -> conf.TOP_K = 20
  N500           -> N=500
  T400           -> steps=400
  comp40         -> 构型 40/20/40（extreme/moderate/swing）
  comp10         -> 构型 10/80/10
  base           -> 默认 20/50/30

输出：<OUTPUT_ROOT>/<批次>/...，不触碰任何原始数据目录。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

PROJ = _PROJ_ROOT
OUT = os.path.join(PROJ, "results_replay_20261010")
sys.path.insert(0, PROJ)

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
GAMMAS = [0.1, 1.0]
BASE_SEED = 12345


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- 批次 1/2：main 与 small 用现成驱动 ----------

def run_main():
    import experiments.rule_experiment as R

    out_dir = os.path.join(OUT, "rule_main_d4")
    os.makedirs(out_dir, exist_ok=True)

    def resolve_fixed(kind):
        assert kind == "rule"
        return out_dir, os.path.join(out_dir, "逐run时间序列")

    R.resolve_result_dirs = resolve_fixed
    R.run_rule_experiment(debug=False)


def run_small():
    import experiment_rule_small as S
    from pathlib import Path
    S.run_experiment(Path(OUT) / "rule_small_d4", BASE_SEED, force=True)


# ---------- 批次 3：spiral（c 三值，第三个提前终止于 159） ----------

SPIRAL_C_GRID = [0.0, 0.5, 1.0]
SPIRAL_LAST_BLOCK_N = 159  # c=1.0 只跑 base_idx 0..158
SPIRAL_REPEATS = 50


def _spiral_tasks():
    tasks = []
    n_cell = len(ALPHAS) * len(GAMMAS) * SPIRAL_REPEATS  # 500
    for ci, c in enumerate(SPIRAL_C_GRID):
        for ai, alpha in enumerate(ALPHAS):
            for gi, gamma in enumerate(GAMMAS):
                for r in range(SPIRAL_REPEATS):
                    base_idx = (ai * len(GAMMAS) + gi) * SPIRAL_REPEATS + r
                    if ci == len(SPIRAL_C_GRID) - 1 and base_idx >= SPIRAL_LAST_BLOCK_N:
                        continue
                    run_id = ci * n_cell + base_idx
                    tasks.append((run_id, alpha, gamma, c, BASE_SEED + base_idx))
    return tasks


def _spiral_worker(task):
    import pandas as pd
    from model.model import SimulationModel
    run_id, alpha, gamma, c, seed = task
    out = os.path.join(OUT, "rule_spiral", "逐run时间序列", f"rule_spiral_run_{run_id}.csv")
    # 断点续跑：文件已存在且行数完整（表头 + 200 步）则直接复用
    if os.path.exists(out) and os.path.getsize(out) > 0:
        try:
            df = pd.read_csv(out)
            if len(df) >= 200:
                final = dict(df.iloc[-1])
                final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma,
                              "spiral_c": c, "spiral_m": 0.5, "seed": seed,
                              "N": 200, "steps": 200})
                return final
        except Exception:
            pass
    m = SimulationModel(seed=seed, N=200, steps=200, alpha=alpha, gamma=gamma,
                        spiral_c=c, spiral_m=0.5, debug=False)
    m.run()
    m.save_results(out)
    final = dict(m.metrics_time_series[-1])
    final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma,
                  "spiral_c": c, "spiral_m": 0.5, "seed": seed, "N": 200, "steps": 200})
    return final


def run_spiral(workers):
    tasks = _spiral_tasks()
    _log(f"spiral: {len(tasks)} runs")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, final in enumerate(ex.map(_spiral_worker, tasks, chunksize=4)):
            results.append(final)
            if (i + 1) % 100 == 0:
                _log(f"spiral {i+1}/{len(tasks)}")
    import pandas as pd
    df = pd.DataFrame(sorted(results, key=lambda x: x["run_id"]))
    df.to_csv(os.path.join(OUT, "rule_spiral", "rule_spiral_summary.csv"), index=False)


# ---------- 批次 4：structure 九变体 ----------

STRUCT_VARIANTS = {
    # name: (H, top_k, N, steps, mix(ext,mod,swing) or None=默认)
    "base":   (1.0, 10, 200, 200, None),
    "H050":   (0.5, 10, 200, 200, None),
    "H020":   (0.2, 10, 200, 200, None),
    "H000":   (0.0, 10, 200, 200, None),
    "K20":    (1.0, 20, 200, 200, None),
    "N500":   (1.0, 10, 500, 200, None),
    "T400":   (1.0, 10, 200, 400, None),
    "comp40": (1.0, 10, 200, 200, (0.4, 0.2, 0.4)),
    "comp10": (1.0, 10, 200, 200, (0.1, 0.8, 0.1)),
}
STRUCT_REPEATS = 20


def _make_default_create_agents():
    """与 model.py 原始 _create_agents 完全一致的通用版（用 self.N，20/50/30）。

    随机调用顺序必须逐位保持：extreme(choice,random,random,uniform) ->
    moderate(uniform,random,random,uniform) -> swing(uniform,random,random,uniform)。
    """
    import random as _rd
    import model.model as _MM

    def _create_agents(self):
        n_ext = int(self.N * 0.2)
        n_mod = int(self.N * 0.5)
        n_swing = self.N - n_ext - n_mod
        idx = 0
        for _ in range(n_ext):
            a = _MM.Agent(idx, opinion=_rd.choice([-1.0, 1.0]), emotion=_rd.random(),
                         agent_type="extreme", activity=_rd.random(),
                         lambda_sensitivity=_rd.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1
        for _ in range(n_mod):
            a = _MM.Agent(idx, opinion=_rd.uniform(-0.3, 0.3), emotion=_rd.random(),
                         agent_type="moderate", activity=_rd.random(),
                         lambda_sensitivity=_rd.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1
        for _ in range(n_swing):
            a = _MM.Agent(idx, opinion=_rd.uniform(-1.0, 1.0), emotion=_rd.random(),
                         agent_type="swing", activity=_rd.random(),
                         lambda_sensitivity=_rd.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1

    return _create_agents


def _make_mixed_create_agents(mix):
    """构型变体版：mix = (extreme 占比, moderate 占比, 其余 swing)。"""
    import random as _rd
    import model.model as _MM

    def _create_agents(self):
        n_ext = int(self.N * mix[0])
        n_mod = int(self.N * mix[1])
        n_swing = self.N - n_ext - n_mod
        idx = 0
        for _ in range(n_ext):
            a = _MM.Agent(idx, opinion=_rd.choice([-1.0, 1.0]), emotion=_rd.random(),
                         agent_type="extreme", activity=_rd.random(),
                         lambda_sensitivity=_rd.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1
        for _ in range(n_mod):
            a = _MM.Agent(idx, opinion=_rd.uniform(-0.3, 0.3), emotion=_rd.random(),
                         agent_type="moderate", activity=_rd.random(),
                         lambda_sensitivity=_rd.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1
        for _ in range(n_swing):
            a = _MM.Agent(idx, opinion=_rd.uniform(-1.0, 1.0), emotion=_rd.random(),
                         agent_type="swing", activity=_rd.random(),
                         lambda_sensitivity=_rd.uniform(0.5, 1.5))
            self.agents.append(a); idx += 1

    return _create_agents


def _make_rec_with_h(H):
    """带 H 系数的 recommend_from_arrays（与 model.recommendation 逐位等价 + H·sim 项）。

    背景：当前代码库的 recommend_from_arrays 把相似度项系数硬编码为 1.0
    （H 通道丢失），而 rule_structure 的 H000/H020/H050 变体当初是以
    Score = H·sim + α·eng + γ·emo 生成的。H=1.0 时两式逐位相等，故仅
    H≠1.0 的变体需要 patch。随机数调用：本函数不消耗随机数。
    """
    import numpy as np

    def recommend_from_arrays(agent, posts, post_opinions, post_engagements,
                              post_emotions, alpha, gamma, top_k,
                              post_author_ids=None, exclude_author_id=None):
        if not posts:
            return []
        scores = (
            H * np.maximum(0.0, 1.0 - np.abs(agent.opinion - post_opinions) / 2.0)
            + alpha * post_engagements
            + gamma * post_emotions
        )
        if exclude_author_id is not None and post_author_ids is not None:
            pool = np.flatnonzero(post_author_ids != exclude_author_id)
            n_valid = int(pool.size)
        else:
            pool = None
            n_valid = int(scores.size)
        if n_valid == 0:
            return []
        idx = min(int(top_k), n_valid)
        if pool is None:
            candidate_indices = np.argpartition(scores, -idx)[-idx:]
            top_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
            return [posts[i] for i in top_indices]
        sub = scores[pool]
        candidate_local = np.argpartition(sub, -idx)[-idx:]
        top_indices = pool[candidate_local[np.argsort(sub[candidate_local])[::-1]]]
        return [posts[i] for i in top_indices]

    return recommend_from_arrays


def _structure_worker(task):
    import model.model as MM
    from config import conf
    variant, run_id, alpha, gamma, seed = task
    H, top_k, N, steps, mix = STRUCT_VARIANTS[variant]

    # ⚠️ worker 进程会被复用，必须每个任务无条件重设 patch，
    # 否则 comp 变体的构型会泄漏到同进程后续的默认构型任务。
    conf.H = H
    conf.TOP_K = top_k
    MM.SimulationModel._create_agents = (
        _make_mixed_create_agents(mix) if mix is not None else _make_default_create_agents()
    )
    # H 打分通道：当前代码库相似度项系数硬编码 1.0，H≠1.0 的变体
    # 必须在 model.model 命名空间替换 recommend_from_arrays（H=1.0 时不 patch）。
    if abs(H - 1.0) > 1e-12:
        MM.recommend_from_arrays = _make_rec_with_h(H)
    else:
        from model.recommendation import recommend_from_arrays as _orig_rec
        MM.recommend_from_arrays = _orig_rec

    out = os.path.join(OUT, "rule_structure", variant, f"rule_run_{run_id}.csv")
    # 断点续跑：文件已存在且行数与 steps 匹配则直接复用
    if os.path.exists(out) and os.path.getsize(out) > 0:
        try:
            import pandas as _pd
            df = _pd.read_csv(out)
            if len(df) >= steps:
                final = dict(df.iloc[-1])
                final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma, "seed": seed,
                              "spiral_c": 0.0, "spiral_m": 0.0, "variant": variant,
                              "top_k": top_k, "h_coef": H})
                return final
        except Exception:
            pass
    m = MM.SimulationModel(seed=seed, N=N, steps=steps, alpha=alpha, gamma=gamma, debug=False)
    m.run()
    m.save_results(out)
    final = dict(m.metrics_time_series[-1])
    final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma, "seed": seed,
                  "spiral_c": 0.0, "spiral_m": 0.0, "variant": variant,
                  "top_k": top_k, "h_coef": H})
    return final


def _structure_tasks():
    tasks = []
    for variant in STRUCT_VARIANTS:
        for ai, alpha in enumerate(ALPHAS):
            for gi, gamma in enumerate(GAMMAS):
                for r in range(STRUCT_REPEATS):
                    run_id = (ai * len(GAMMAS) + gi) * STRUCT_REPEATS + r
                    tasks.append((variant, run_id, alpha, gamma, BASE_SEED + run_id))
    return tasks


def run_structure(workers):
    tasks = _structure_tasks()
    _log(f"structure: {len(tasks)} runs (9 variants x 200)")
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, final in enumerate(ex.map(_structure_worker, tasks, chunksize=2)):
            results.append(final)
            if (i + 1) % 200 == 0:
                _log(f"structure {i+1}/{len(tasks)}")
    import pandas as pd
    df = pd.DataFrame(sorted(results, key=lambda x: (x["variant"], x["run_id"])))
    cols = ["PI", "VPI", "SR", "ER", "EM", "HE", "EE", "step", "alpha", "gamma",
            "seed", "spiral_c", "spiral_m", "mean_exposure_distance",
            "run_id", "variant", "top_k", "h_coef"]
    df = df[cols]
    df.to_csv(os.path.join(OUT, "rule_structure", "_all_variants_merged.csv"),
              index=False, encoding="utf-8-sig")


# ---------- 汇总重算（dose_response 等，供与原档案交叉校验） ----------

def build_reports():
    import glob
    import json
    import numpy as np
    import pandas as pd

    def ols(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        res = y - yhat
        sse = float(res @ res)
        sxx = float(((x - x.mean()) ** 2).sum())
        se_b = np.sqrt(sse / (n - 2) / sxx) if sxx > 0 else np.nan
        t = b / se_b if se_b else np.nan
        r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
        return b, se_b, t, r2, n

    # structure dose response
    rows = []
    merged = pd.read_csv(os.path.join(OUT, "rule_structure", "_all_variants_merged.csv"))
    for (v, g), sub in merged.groupby(["variant", "gamma"]):
        b, se, t, r2, n = ols(sub["alpha"], sub["PI"])
        # beta_std
        sx, sy = sub["alpha"].std(ddof=1), sub["PI"].std(ddof=1)
        rows.append({"variant": v, "gamma": g, "beta": b, "se": se, "r2": r2,
                     "beta_std": b * sx / sy, "n": n, "t": t,
                     "pi0": sub[sub.alpha == 0]["PI"].mean(),
                     "pi2": sub[sub.alpha == 2]["PI"].mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rule_structure", "_variant_dose_response.csv"),
                              index=False)

    # spiral summary 校验用统计
    sp = pd.read_csv(os.path.join(OUT, "rule_spiral", "rule_spiral_summary.csv"))
    stats = {"n_runs": len(sp), "c_values": sorted(sp.spiral_c.unique().tolist()),
             "per_c": sp.groupby("spiral_c").size().to_dict()}
    with open(os.path.join(OUT, "_replay_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    _log("reports done")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    workers = int(os.getenv("REPLAY_WORKERS", "12"))
    t0 = time.time()
    if which in ("all", "main"):
        _log("=== rule_main_d4 begin ===")
        run_main()
    if which in ("all", "small"):
        _log("=== rule_small_d4 begin ===")
        run_small()
    if which in ("all", "spiral"):
        _log("=== rule_spiral begin ===")
        run_spiral(workers)
    if which in ("all", "structure"):
        _log("=== rule_structure begin ===")
        run_structure(workers)
    if which == "reports":
        build_reports()
    _log(f"=== {which} done in {time.time()-t0:.0f}s ===")
