# -*- coding: utf-8 -*-
"""方向版提示词 · 全套原始实验重跑统一 runner（Qwen + DeepSeek 共用）。

背景
----
2026-10-08 判定：原提示词缺少字段图例与带符号的曝光方向，已在
``probe_prompt_clarity.py`` 中构造「方向版」提示词（标签版 + feed_mean_opinion），
并跑完主实验 160 runs（probe/results_prompt_direction，种子 31345–31504）。
本脚本把**原设计套件的其余批次**在方向版下全部重跑，种子锚定与存档逐格一致，
输出全部写入 ``results_direction/``，不碰任何存档目录。

阶段一览（--stage）
-------------------
  qwen-full   Qwen 跨模型复现      200 runs（spec 逐行取自存档 llm_qwen_full summary）
  qwen-abl    Qwen 在线消融 ×3 臂   10 + 90 + 90 = 190 runs（种子取自 llm_qwen_full）
  randomfeed  随机 feed 对照         20 runs（种子 31425–31444，与方向版 α=1.0 块配对）
  abl         DeepSeek 在线消融 ×3  3 × 160 runs（seed = 31345 + gi*10 + rep，配对核验）
  r20         DeepSeek R=20 加强档  160 runs（run_id 160–319，seed 531345–531504，取自存档）
  n100        DeepSeek N=100 规模   30 runs（spec 取自存档，N 覆盖为 100）
  t100        DeepSeek T=100 延长  160 runs（seed = 31345 + run_id，steps 覆盖为 100）
  qwen-t100   Qwen T=100 延长（完整版）200 runs（与主实验逐格同构，8α×2γ；种子逐格取自方向版 qwen_full）
  n100b       N=100 规模档补重复   60 runs（每格 +20；新种子段 531505-531564，N 覆盖为 100）
  qwen-all    = qwen-full + qwen-abl（今晚用）
  abl-all     = abl 三臂顺序（调度器用）

红线：提示词由 probe_prompt_clarity.install_direction_prompt() 统一安装，
      alpha/gamma/top_k/social_pressure 永不进入提示词；本脚本不修改项目任何文件。

用法
----
  --plan        零成本打印设计与成本
  --self-test   零网络自检（提示词拼装 + 红线断言 + spec 核验 + 模型构造）
  --workers W   父进程模式：拉起 W 个分片子进程，结束后自动合并
  --shard k --shards W   子进程模式
  --status      打印各阶段完成度（JSON），供调度器判断
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
# --- portability shim (auto-added) ---------------------------------
import os as _os
# --- portability shim (auto-added) ---------------------------------
import os as _os
_BASE_ROOT = _os.getenv("ABM_BASE_ROOT", r"D:\21130\Desktop\大创过程")
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
_VENV_PY = _os.getenv("ABM_VENV_PY", r"C:\Users\21130\.workbuddy\binaries\python\envs\default\Scripts\python.exe")
# -------------------------------------------------------------------
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

PROJ = Path(_PROJ_ROOT)
HERE = Path(__file__).resolve().parent
OUT_ROOT = PROJ / "results_direction"
ENV_QWEN = PROJ / ".env.qwen"
PYTHON = _PYTHON
# 实测单价（¥/run）：方向版 DeepSeek T30/N30 = 0.0935；N100/T100 调用量 ×3.33 ≈ 0.311；
# Qwen（百炼·北京，无峰谷）≈ 存档 0.0638 × 提示词系数 1.53 ≈ 0.098。
COST_PER_RUN = {"t30n30_ds": 0.0935, "t30n30_qwen": 0.098, "x3_ds": 0.311,
                "x3_qwen": 0.326}   # Qwen T100/N30 调用量 ×3.33 ≈ 0.098×3.33
MIN_PER_RUN_T30 = 2.2          # 单进程实测
MIN_PER_CALL = MIN_PER_RUN_T30 / 90.0

FULL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
FULL_GAMMAS = [0.1, 1.0]
FULL_GRID = [(a, g) for a in FULL_ALPHAS for g in FULL_GAMMAS]

ABL_ARMS_DS = ["no_post", "no_emotion", "positive_control"]
ABL_ARMS_QWEN = ["no_post", "no_emotion", "positive_control"]
QWEN_MECH_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
QWEN_MECH_REPS = 9
QWEN_POS_CELL = (1.0, 1.0)
QWEN_POS_REPS = 10

# 阶段 → [(输出目录名, 期望完成 runs, 单价键)]
STAGE_DIRS = {
    "qwen-full": [("llm_qwen_full", 200, "t30n30_qwen")],
    "qwen-abl": [("llm_qwen_abl_positive_control", 10, "t30n30_qwen"),
                 ("llm_qwen_abl_no_post", 90, "t30n30_qwen"),
                 ("llm_qwen_abl_no_emotion", 90, "t30n30_qwen")],
    "randomfeed": [("llm_randomfeed_control", 20, "t30n30_ds")],
    "abl": [("llm_ablation_no_post", 160, "t30n30_ds"),
            ("llm_ablation_no_emotion", 160, "t30n30_ds"),
            ("llm_ablation_positive_control", 20, "t30n30_ds")],
    "r20": [("llm_gradient_r20", 160, "t30n30_ds")],
    "n100": [("llm_n100_scale", 30, "x3_ds")],
    "n100r": [("llm_n100_ratio_scale", 30, "x3_ds")],   # K 比例化新制度：N=100 → K=33
    "t100": [("llm_extend_T100", 160, "x3_ds")],
    "qwen-t100": [("llm_qwen_extend_T100", 200, "x3_qwen")],
    "n100b": [("llm_n100_boost", 60, "x3_ds")],
}


# ---------------------------------------------------------------- 环境注入
def clear_proxies() -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def inject_qwen_env() -> None:
    """把 .env.qwen 注入进程环境（setdefault，不覆盖已有变量）；必须在 import config 前调用。"""
    if not ENV_QWEN.exists():
        raise SystemExit(f"缺少 {ENV_QWEN}")
    for line in ENV_QWEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
    clear_proxies()
    os.environ["NO_PROXY"] = "dashscope.aliyuncs.com"


def setup_stage_env(stage: str) -> None:
    if stage.startswith("qwen"):
        inject_qwen_env()
    else:
        clear_proxies()
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


# ---------------------------------------------------------------- 项目导入 + 方向版提示词
_E = None


def import_project(stage: str):
    """导入项目模块并安装方向版提示词（父/子进程都走这里，补丁随进程生效）。"""
    global _E
    if str(PROJ) not in sys.path:
        sys.path.insert(0, str(PROJ))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import experiment_llm_robust as E
    import probe_prompt_clarity as PC

    PC.install_direction_prompt()          # 两处命名空间 + _feed_features 扩展
    if "LLM_ABLATION" in os.environ:       # 非消融阶段绝不残留消融开关
        del os.environ["LLM_ABLATION"]
    _E = E
    return E, PC


# ---------------------------------------------------------------- spec 生成
def _read_archive_rows(path: Path) -> list[dict]:
    """从存档 summary csv 读 (run_id, alpha, gamma, repeat, seed) 行。"""
    import pandas as pd

    df = pd.read_csv(path)
    out = []
    for i, r in df.iterrows():
        out.append({
            "run_id": int(r["run_id"]) if "run_id" in df.columns else i,
            "alpha": round(float(r["alpha"]), 4),
            "gamma": round(float(r["gamma"]), 4),
            "repeat": int(r["repeat"]),
            "seed": int(r["seed"]),
        })
    return out


def specs_qwen_full() -> list[dict]:
    """Qwen 复现批：spec 逐行取自存档 summary（200 行，非均衡设计，seed 为唯一锚）。"""
    rows = _read_archive_rows(PROJ / "results" / "llm_qwen_full" / "llm_robust_summary.csv")
    assert len(rows) == 200, f"存档 qwen_full 行数 {len(rows)} != 200"
    assert len({r["seed"] for r in rows}) == 200, "存档 qwen_full seed 有重复"
    return rows


def specs_qwen_abl(arm: str) -> list[dict]:
    """Qwen 消融：种子取自 llm_qwen_full（同 seed + 同 N/T ⇒ 初始条件逐位相同）。

    机制臂：5α × 2γ × 9 reps；阳性对照：α=1.0/γ=1.0 × 10 reps（照抄 _qwen_ablation.py）。
    """
    import pandas as pd

    rows = _read_archive_rows(PROJ / "results" / "llm_qwen_full" / "llm_robust_summary.csv")
    cells = [(QWEN_POS_CELL[0], QWEN_POS_CELL[1])] if arm == "positive_control" \
        else [(a, g) for g in FULL_GAMMAS for a in QWEN_MECH_ALPHAS]
    reps = QWEN_POS_REPS if arm == "positive_control" else QWEN_MECH_REPS
    specs = []
    for a, g in cells:
        cell = sorted((r for r in rows if r["gamma"] == g and r["alpha"] == a),
                      key=lambda r: r["seed"])
        if len(cell) < reps:
            raise SystemExit(f"(α={a}, γ={g}) 在 qwen_full 只有 {len(cell)} 行，不足 {reps}")
        picked = cell[:reps]
        seeds = [r["seed"] for r in picked]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"(α={a}, γ={g}) seed 重复：{seeds}")
        for r in picked:
            specs.append({"run_id": len(specs), "repeat": int(r["repeat"]),
                          "alpha": float(a), "gamma": float(g), "seed": int(r["seed"])})
    assert len(specs) == (10 if arm == "positive_control" else 90)
    return specs


def specs_randomfeed() -> list[dict]:
    """随机 feed 对照：种子与方向版 α=1.0 块（存档 noleak run 80–99）逐格配对。"""
    specs = []
    for gamma, lo in ((0.1, 31425), (1.0, 31435)):
        for rep in range(10):
            specs.append({"run_id": len(specs), "repeat": rep,
                          "alpha": 1.0, "gamma": gamma, "seed": lo + rep})
    assert len(specs) == 20
    return specs


def specs_abl(arm: str) -> list[dict]:
    """DeepSeek 消融：复现 run_llm_ablation 的锚定公式 seed = 31345 + gi*10 + rep。

    ⚠️ 2026-10-09 削减决定：阳性对照臂缩为探针档（α=1.0×2γ×R=10 = 20 runs）。
    理由：do(stance_delta=0) 效应量 dz≈3.9，原团队在 Qwen 侧同逻辑仅用 10 runs
    轻验证；且探针种子（31425–31444）是全网格 α=1.0 格子的子集，将来可无损扩到 160。
    """
    if arm == "positive_control":
        specs = []
        for gamma, lo in ((0.1, 31425), (1.0, 31435)):
            for rep in range(10):
                specs.append({"run_id": len(specs), "repeat": rep,
                              "alpha": 1.0, "gamma": gamma, "seed": lo + rep})
        assert len(specs) == 20
        return specs
    specs = []
    for alpha, gamma in FULL_GRID:
        gi = FULL_ALPHAS.index(alpha) * len(FULL_GAMMAS) + FULL_GAMMAS.index(gamma)
        for rep in range(10):
            specs.append({"run_id": gi * 10 + rep, "repeat": rep,
                          "alpha": alpha, "gamma": gamma, "seed": 31345 + gi * 10 + rep})
    assert len(specs) == 160
    return specs


def specs_r20() -> list[dict]:
    """R=20 加强档：spec 逐行取自存档（run_id 160–319，seed 531345–531504）。"""
    m = PROJ / "results" / "llm_gradient_r20" / "manifest.json"
    if m.exists():
        runs = json.loads(m.read_text(encoding="utf-8"))["runs"]
        rows = [{"run_id": int(r["run_id"]), "alpha": round(float(r["alpha"]), 4),
                 "gamma": round(float(r["gamma"]), 4), "repeat": int(r["repeat"]),
                 "seed": int(r["seed"])} for r in runs]
    else:
        rows = _read_archive_rows(PROJ / "results" / "llm_gradient_r20" / "llm_robust_summary.csv")
    assert len(rows) == 160, f"存档 r20 行数 {len(rows)} != 160"
    assert min(r["seed"] for r in rows) == 531345 and max(r["seed"] for r in rows) == 531504
    return rows


def specs_n100() -> list[dict]:
    """N=100 规模：spec 逐行取自存档（30 runs，γ=1.0，α∈{0,1,2}）。"""
    rows = _read_archive_rows(PROJ / "results" / "llm_n100_scale" / "llm_robust_summary.csv")
    assert len(rows) == 30, f"存档 n100 行数 {len(rows)} != 30"
    assert len({r["seed"] for r in rows}) == 30
    return rows


def specs_t100() -> list[dict]:
    """T=100 延长：全网格 160 runs，seed = 31345 + run_id（与存档延长批同种子）。"""
    specs = []
    for alpha, gamma in FULL_GRID:
        gi = FULL_ALPHAS.index(alpha) * len(FULL_GAMMAS) + FULL_GAMMAS.index(gamma)
        for rep in range(10):
            specs.append({"run_id": gi * 10 + rep, "repeat": rep,
                          "alpha": alpha, "gamma": gamma, "seed": 31345 + gi * 10 + rep})
    # 与存档延长批逐格核对种子
    p = PROJ / "results" / "llm_extend_T100" / "llm_extend_summary.csv"
    if p.exists():
        import pandas as pd

        df = pd.read_csv(p).dropna(subset=["alpha"])
        arch = {int(r["run_id"]): int(r["seed"]) for _, r in df.iterrows()}
        bad = [s for s in specs if s["run_id"] in arch and arch[s["run_id"]] != s["seed"]]
        if bad:
            raise SystemExit(f"t100 种子与存档不一致：{bad[:5]}")
    assert len(specs) == 160
    return specs


def specs_qwen_t100() -> list[dict]:
    """Qwen T=100 延长（完整版）：与方向版 T=30 主实验逐格同构（200 runs）。

    种子逐格取自方向版存档 results_direction/llm_qwen_full（同 seed + 同 N
    ⇒ 共享初始条件；T=30/T=100 为两次独立实现，配对含义为共享初始化）。
    每格重复数与 qwen_full 完全一致：γ=0.1 八档各 10；γ=1.0 八档各 10、
    端点 α∈{0,2} 各再 +20（即 30）。run_id 0-199（0-139 已先行完成，顺序不变；
    140-199 为追加的三档 α∈{0.25,0.75,1.25}×2γ，各 10）。
    """
    rows = _read_archive_rows(OUT_ROOT / "llm_qwen_full" / "llm_robust_summary.csv")
    assert len(rows) == 200, f"方向版 qwen_full 行数 {len(rows)} != 200"
    expect = {(0.0, 0.1): 10, (0.5, 0.1): 10, (1.0, 0.1): 10, (1.5, 0.1): 10, (2.0, 0.1): 10,
              (0.0, 1.0): 30, (0.5, 1.0): 10, (1.0, 1.0): 10, (1.5, 1.0): 10, (2.0, 1.0): 30}
    specs = []
    for (a, g) in expect:
        cell = sorted((r for r in rows if r["alpha"] == a and r["gamma"] == g),
                      key=lambda r: r["seed"])
        if len(cell) != expect[(a, g)]:
            raise SystemExit(f"(α={a}, γ={g}) 方向版 qwen_full 实际 {len(cell)} 行，"
                             f"期望 {expect[(a, g)]}（存档变动则拒绝开跑）")
        for r in cell:
            specs.append({"run_id": len(specs), "repeat": int(r["repeat"]),
                          "alpha": float(a), "gamma": float(g), "seed": int(r["seed"])})
    assert len(specs) == 140, "前 140（5α 子集）顺序必须与既有存档一致"
    # 完整版追加：其余三档 α × 2γ，逐格种子取自 qwen_full（run_id 140-199）
    for a in (0.25, 0.75, 1.25):
        for g in (0.1, 1.0):
            cell = sorted((r for r in rows if r["alpha"] == a and r["gamma"] == g),
                          key=lambda r: r["seed"])
            if len(cell) != 10:
                raise SystemExit(f"(α={a}, γ={g}) qwen_full 实际 {len(cell)} 行，期望 10")
            for r in cell:
                specs.append({"run_id": len(specs), "repeat": int(r["repeat"]),
                              "alpha": float(a), "gamma": float(g), "seed": int(r["seed"])})
    assert len(specs) == 200
    assert len({s["seed"] for s in specs}) == 200, "qwen-t100 seed 重复"
    return specs


# N=100 补重复：三格（α=0/1/2, γ=1.0），每格 +20 次（repeat 10-29）
N100B_CELLS = [
    # (alpha, run_id 基, seed 基)  run_id 沿用原档公式 = 网格序号×10 + repeat
    (0.0, 20, 531505),
    (1.0, 100, 531525),
    (2.0, 160, 531545),
]


def specs_n100b() -> list[dict]:
    """N=100 规模档补重复：每格 +20 次（repeat 10-29），共 60 runs（用户确认完整版）。

    * run_id 沿用原档公式（α=0 格 10+rep、α=1 格 90+rep、α=2 格 150+rep），
      repeat 10-29 → run_id 20-29 / 100-119 / 160-179；与存档 30 个零重叠，
      合并后每格 run_id 连续（10-29 / 90-119 / 150-179），n=90。
    * 种子独立扩展段 531505-531564（接续 r20 的 531345-531504，无任何批次重叠）。
    * 输出独立目录 llm_n100_boost；分析时与 llm_n100_scale 合并成 90 runs。
    """
    import pandas as pd

    specs = []
    for alpha, rid0, seed0 in N100B_CELLS:
        for k in range(20):
            specs.append({"run_id": rid0 + k, "repeat": 10 + k,
                          "alpha": alpha, "gamma": 1.0, "seed": seed0 + k})
    assert len(specs) == 60
    assert len({s["seed"] for s in specs}) == 60, "n100b seed 重复"
    d = pd.read_csv(OUT_ROOT / "llm_n100_scale" / "llm_robust_summary.csv")
    old_ids = {int(x) for x in d.run_id}
    old_seeds = {int(x) for x in d.seed}
    assert not (old_ids & {s["run_id"] for s in specs}), "n100b run_id 与存档冲突"
    assert not (old_seeds & {s["seed"] for s in specs}), "n100b seed 与存档冲突"
    return specs


# ---------------------------------------------------------------- 配对核验（DeepSeek 消融）
def verify_abl_pairing(specs: list[dict]) -> None:
    """与存档 noleak manifest 逐条核对 seed；不通过直接拒绝开跑。"""
    p = PROJ / "results" / "llm_gradient_noleak" / "manifest.json"
    baseline = {}
    if p.exists():
        for r in json.loads(p.read_text(encoding="utf-8"))["runs"]:
            baseline[(round(float(r["alpha"]), 4), round(float(r["gamma"]), 4), int(r["repeat"]))] = int(r["seed"])
    problems = []
    for s in specs:
        key = (s["alpha"], s["gamma"], s["repeat"])
        if key in baseline and baseline[key] != s["seed"]:
            problems.append(f"{key}: 存档 {baseline[key]} vs 本批 {s['seed']}")
    if problems:
        raise SystemExit("消融配对核验失败，拒绝开跑：\n  " + "\n  ".join(problems[:10]))
    return len(baseline)


# ---------------------------------------------------------------- 随机 feed 补丁（照抄 run_llm_randomfeed.py）
_RF = {"rng": random.Random(0)}
_RF_STATUS = {"calls": 0, "picked": 0, "pool": 0, "dist": 0.0}


def _random_recommend(agent, posts, opinions, engagements, emotions,
                      alpha, gamma, top_k, post_author_ids=None,
                      exclude_author_id=None):
    """等价签名，均匀随机抽样替换算法排序（α/γ 完全不参与）。"""
    n = len(opinions)
    if n == 0 or top_k <= 0:
        return []
    if post_author_ids is not None:
        pool = [i for i in range(n) if int(post_author_ids[i]) != exclude_author_id]
    else:
        pool = [i for i in range(n) if posts[i].author_id != exclude_author_id]
    k = min(int(top_k), len(pool))
    if k <= 0:
        return []
    pick = _RF["rng"].sample(pool, k)
    _RF_STATUS["calls"] += 1
    _RF_STATUS["picked"] += k
    _RF_STATUS["pool"] += len(pool)
    _RF_STATUS["dist"] += sum(abs(agent.opinion - posts[i].opinion) / 2.0 for i in pick) / k
    return [posts[i] for i in pick]


def install_randomfeed() -> None:
    """模块级替换 recommend_from_arrays + 按 run seed 重播种（子进程重新 import 即生效）。"""
    import model.llm_robustness_v2 as M

    orig_run = M.LLMRobustnessModel.run

    def _patched_run(self):
        _RF["rng"] = random.Random(int(self.seed) * 1000003 + 17)
        return orig_run(self)

    M.recommend_from_arrays = _random_recommend
    M.LLMRobustnessModel.run = _patched_run


# ---------------------------------------------------------------- 单臂执行（子进程/单进程）
def run_one_batch(stage: str, arm: str, specs: list[dict], out_dir: Path,
                  max_llm_calls, force: bool) -> None:
    E = _E
    E.CONDITIONS = sorted({(s["alpha"], s["gamma"]) for s in specs},
                          key=lambda t: (t[1], t[0]))
    E.REPEATS = max(len({s["repeat"] for s in specs}), 1)

    if stage in ("n100", "n100r", "n100b"):
        E.N_AGENTS = 100
    if stage in ("t100", "qwen-t100"):
        E.STEPS = 100
    if arm:                                # 消融臂（DeepSeek abl 与 Qwen qwen-abl 通用）
        os.environ["LLM_ABLATION"] = arm
    if stage == "randomfeed":
        install_randomfeed()

    mine = specs                       # 目录已按 shard 隔离，无需再切
    def _it(_base_seed: int):
        for s in mine:
            yield dict(s)

    E.iter_specs = _it
    out_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(output_dir=out_dir,
                              base_seed=int(specs[0]["seed"]) if specs else 0,
                              repeats=E.REPEATS, max_llm_calls=max_llm_calls,
                              force=force, self_test=False, plan=False)
    print(f"[stage:{stage}{('/' + arm) if arm else ''}] {len(mine)} runs -> {out_dir}", flush=True)
    E.run_experiment(args)

    # 质量门：兜底率超限 = 本批不是真实模型行为
    import pandas as pd

    s = out_dir / "llm_robust_summary.csv"
    if s.exists():
        df = pd.read_csv(s)
        fb = float(df["llm_fallback_rate"].mean()) if "llm_fallback_rate" in df.columns else 0.0
        cost = float(df["estimated_cost"].sum()) if "estimated_cost" in df.columns else 0.0
        print(f"[stage:{stage}] 完成 {len(df)} runs | 兜底率均值 {fb:.2%} | 成本 ¥{cost:.2f}", flush=True)
        if fb > 0.05:
            raise RuntimeError(f"兜底率 {fb:.1%} > 5%：本批作废，排查网络/代理后重跑。")
        if stage == "randomfeed" and _RF_STATUS["calls"]:
            print(f"[stage:randomfeed] 随机抽样统计：曝光立场距离均值 "
                  f"{_RF_STATUS['dist'] / _RF_STATUS['calls']:.4f}", flush=True)


# ---------------------------------------------------------------- 合并 + 完成度
def shard_dirs(base: Path) -> list[Path]:
    return sorted(p for p in base.glob("shard*_of*") if p.is_dir())


def completed_rows(dirs: list[Path], steps: int) -> int:
    import pandas as pd

    n = 0
    for d in dirs:
        s = d / "llm_robust_summary.csv"
        if not s.exists():
            continue
        try:
            df = pd.read_csv(s)
        except Exception:
            continue
        if "step" not in df.columns:
            continue
        ok = df["step"].astype(float) == float(steps)
        if "status" in df.columns:
            ok &= df["status"].astype(str).isin({"completed", "completed_with_fallback"})
        n += int(ok.sum())
    return n


def merge_dir(base: Path) -> None:
    import pandas as pd

    shards = shard_dirs(base)
    if not shards:
        return
    frames, runs = [], []
    for d in shards:
        s = d / "llm_robust_summary.csv"
        if s.exists():
            frames.append(pd.read_csv(s))
        m = d / "manifest.json"
        if m.exists():
            runs.extend(json.loads(m.read_text(encoding="utf-8")).get("runs", []))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("run_id").sort_values("run_id")
    merged.to_csv(base / "llm_robust_summary.csv", index=False)
    by_run = {int(r["run_id"]): r for r in runs}
    (base / "manifest.json").write_text(
        json.dumps({"experiment": "direction_rerun", "source": "merged_from_shards",
                    "runs": [by_run[k] for k in sorted(by_run)]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] {base.name}: {len(merged)} runs -> {base / 'llm_robust_summary.csv'}", flush=True)


def stage_progress(stage: str) -> dict:
    steps = 100 if stage in ("t100", "qwen-t100") else 30
    out = {}
    for name, expected, _cost in STAGE_DIRS[stage]:
        base = OUT_ROOT / name
        if (base / "llm_robust_summary.csv").exists():
            import pandas as pd

            try:
                df = pd.read_csv(base / "llm_robust_summary.csv")
                ok = df["step"].astype(float) == float(steps)
                if "status" in df.columns:
                    ok &= df["status"].astype(str).isin({"completed", "completed_with_fallback"})
                done = int(ok.sum())
            except Exception:
                done = 0
        else:
            done = completed_rows(shard_dirs(base), steps)
        out[name] = {"done": done, "expected": expected}
    return out


# ---------------------------------------------------------------- 父进程
def run_stage_group(stage: str, arm: str, workers: int, max_llm_calls, force: bool) -> None:
    """父进程：按臂顺序执行（每臂拉起 W 个分片并等待，随后合并 + 质量检查）。"""
    E, PC = import_project(stage)
    if stage == "qwen-full":
        jobs = [("llm_qwen_full", specs_qwen_full(), None)]
    elif stage == "qwen-abl":
        arms = ABL_ARMS_QWEN if arm in (None, "", "all") else [arm]
        jobs = [(f"llm_qwen_abl_{a}", specs_qwen_abl(a), a) for a in arms]
    elif stage == "randomfeed":
        jobs = [("llm_randomfeed_control", specs_randomfeed(), None)]
    elif stage == "abl":
        arms = ABL_ARMS_DS if arm in (None, "", "all") else [arm]
        jobs = [(f"llm_ablation_{a}", specs_abl(a), a) for a in arms]
    elif stage == "r20":
        jobs = [("llm_gradient_r20", specs_r20(), None)]
    elif stage == "n100":
        jobs = [("llm_n100_scale", specs_n100(), None)]
    elif stage == "n100r":
        jobs = [("llm_n100_ratio_scale", specs_n100(), None)]
    elif stage == "t100":
        jobs = [("llm_extend_T100", specs_t100(), None)]
    elif stage == "qwen-t100":
        jobs = [("llm_qwen_extend_T100", specs_qwen_t100(), None)]
    elif stage == "n100b":
        jobs = [("llm_n100_boost", specs_n100b(), None)]
    else:
        raise SystemExit(f"未知 stage: {stage}")

    for dirname, specs, job_arm in jobs:
        base = OUT_ROOT / dirname
        steps = 100 if stage in ("t100", "qwen-t100") else 30
        done = completed_rows([base], steps) if (base / "llm_robust_summary.csv").exists() \
            else completed_rows(shard_dirs(base), steps)
        if done >= len(specs) and not force:
            print(f"[parent] {dirname} 已完成 {done}/{len(specs)}，跳过", flush=True)
            continue

        t0 = time.time()
        if workers <= 1:
            run_one_batch(stage, job_arm or "", specs, base, max_llm_calls, force)
            merge_dir(base)
        else:
            logs = base / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            procs = []
            script = str(Path(__file__).resolve())
            for k in range(workers):
                log = open(logs / f"shard{k:02d}.log", "w", encoding="utf-8")
                cmd = [PYTHON, script, "--stage", stage, "--worker",
                       "--shards", str(workers), "--shard", str(k)]
                if job_arm:
                    cmd += ["--arm", job_arm]
                if max_llm_calls:
                    cmd += ["--max-llm-calls", str(max_llm_calls)]
                if force:
                    cmd += ["--force"]
                procs.append((k, subprocess.Popen(cmd, cwd=str(PROJ), stdout=log,
                                                  stderr=subprocess.STDOUT,
                                                  env=dict(os.environ)), log))
            print(f"[parent] {dirname}: 已拉起 {workers} 分片，共 {len(specs)} runs", flush=True)
            while any(p.poll() is None for _, p, _ in procs):
                time.sleep(20)
            for _, _, log in procs:
                log.close()
            codes = {k: p.returncode for k, p, _ in procs}
            bad = {k: c for k, c in codes.items() if c != 0}
            if bad:
                raise SystemExit(f"[parent] {dirname} 分片非零退出 {bad}，看 {logs}/shard??.log")
            merge_dir(base)
            done = completed_rows([base], steps)
            if done < len(specs):
                raise SystemExit(f"[parent] {dirname} 完成数 {done} < 期望 {len(specs)}")
        print(f"[parent] {dirname} 完成，用时 {(time.time() - t0) / 60:.1f} min", flush=True)


def run_shard(stage: str, arm: str, shard: int, shards: int, max_llm_calls, force: bool) -> None:
    E, PC = import_project(stage)
    if stage == "qwen-full":
        specs, dirname, job_arm = specs_qwen_full(), "llm_qwen_full", None
    elif stage == "qwen-abl":
        job_arm = arm if arm else "no_post"
        specs, dirname = specs_qwen_abl(job_arm), f"llm_qwen_abl_{job_arm}"
    elif stage == "randomfeed":
        specs, dirname, job_arm = specs_randomfeed(), "llm_randomfeed_control", None
    elif stage == "abl":
        job_arm = arm if arm else "no_post"
        specs, dirname = specs_abl(job_arm), f"llm_ablation_{job_arm}"
    elif stage == "r20":
        specs, dirname, job_arm = specs_r20(), "llm_gradient_r20", None
    elif stage == "n100":
        specs, dirname, job_arm = specs_n100(), "llm_n100_scale", None
    elif stage == "n100r":
        specs, dirname, job_arm = specs_n100(), "llm_n100_ratio_scale", None
    elif stage == "t100":
        specs, dirname, job_arm = specs_t100(), "llm_extend_T100", None
    elif stage == "qwen-t100":
        specs, dirname, job_arm = specs_qwen_t100(), "llm_qwen_extend_T100", None
    elif stage == "n100b":
        specs, dirname, job_arm = specs_n100b(), "llm_n100_boost", None
    else:
        raise SystemExit(f"未知 stage: {stage}")

    mine = [s for i, s in enumerate(specs) if i % shards == shard]
    out_dir = OUT_ROOT / dirname / f"shard{shard:02d}_of{shards:02d}"
    run_one_batch(stage, job_arm or "", mine, out_dir, max_llm_calls, force)


# ---------------------------------------------------------------- plan / self-test
def stage_specs(stage: str, arm: str):
    if stage == "qwen-full":
        return [("llm_qwen_full", specs_qwen_full(), None)]
    if stage == "qwen-abl":
        arms = ABL_ARMS_QWEN if arm in (None, "", "all") else [arm]
        return [(f"llm_qwen_abl_{a}", specs_qwen_abl(a), a) for a in arms]
    if stage == "randomfeed":
        return [("llm_randomfeed_control", specs_randomfeed(), None)]
    if stage == "abl":
        arms = ABL_ARMS_DS if arm in (None, "", "all") else [arm]
        return [(f"llm_ablation_{a}", specs_abl(a), a) for a in arms]
    if stage == "r20":
        return [("llm_gradient_r20", specs_r20(), None)]
    if stage == "n100":
        return [("llm_n100_scale", specs_n100(), None)]
    if stage == "n100r":
        return [("llm_n100_ratio_scale", specs_n100(), None)]
    if stage == "t100":
        return [("llm_extend_T100", specs_t100(), None)]
    if stage == "qwen-t100":
        return [("llm_qwen_extend_T100", specs_qwen_t100(), None)]
    if stage == "n100b":
        return [("llm_n100_boost", specs_n100b(), None)]
    raise SystemExit(f"未知 stage: {stage}")


def print_plan(stage: str, arm: str, workers: int) -> None:
    total_runs, total_cost = 0, 0.0
    print("=" * 78)
    for dirname, specs, job_arm in stage_specs(stage, arm):
        n = len(specs)
        cells = sorted({(s["alpha"], s["gamma"]) for s in specs})
        if stage == "qwen-t100":
            cost_key = "x3_qwen"
        else:
            cost_key = "t30n30_qwen" if stage.startswith("qwen") else ("x3_ds" if stage in ("n100", "n100r", "n100b", "t100") else "t30n30_ds")
        unit = COST_PER_RUN[cost_key]
        calls_per_run = 90 if stage not in ("n100", "n100r", "n100b", "t100", "qwen-t100") else 300
        min_per_run = calls_per_run * MIN_PER_CALL
        wall = n * min_per_run / max(workers, 1) / 60.0
        total_runs += n
        total_cost += n * unit
        print(f"[plan] {dirname}: {n} runs | {len(cells)} 格 | seed [{min(s['seed'] for s in specs)}"
              f", {max(s['seed'] for s in specs)}] | 单价 ¥{unit:.4f} | 成本 ¥{n * unit:.2f}"
              f" | {workers} 分片 ≈ {wall:.1f} h")
        if job_arm:
            print(f"[plan]    消融臂 = {job_arm}（LLM_ABLATION 环境变量生效）")
    print(f"[plan] 合计 {total_runs} runs，估算成本 ¥{total_cost:.2f}"
          f"（DeepSeek 空闲时段半价，可再降）")
    print(f"[plan] 输出根目录 {OUT_ROOT}")
    print("=" * 78)


def self_test(stage: str, arm: str) -> None:
    setup_stage_env(stage)
    E, PC = import_project(stage)
    print(f"[selftest] provider={os.environ.get('LLM_PROVIDER')} "
          f"model={os.environ.get('LLM_MODEL')} endpoint={os.environ.get('LLM_API_ENDPOINT')}")
    assert os.environ.get("LLM_API_KEY"), "LLM_API_KEY 未配置"
    for name, specs, job_arm in stage_specs(stage, arm):
        seeds = [s["seed"] for s in specs]
        assert len(seeds) == len(set(seeds)), f"{name} seed 重复"
        print(f"[selftest] OK {name}: {len(specs)} runs，seed 唯一，"
              f"范围 [{min(seeds)}, {max(seeds)}]")
    if stage == "abl":
        n_pair = verify_abl_pairing(specs_abl("no_post"))
        print(f"[selftest] OK 消融配对核验通过（对照 noleak manifest {n_pair} 格）")
    # 红线断言：方向版提示词不得含 alpha/gamma/top_k/social_pressure，必须含 feed_mean_opinion
    fake = [{"agent_id": 1, "state": {"opinion": 0.1, "emotion": 0.2, "activity": 0.3,
                                      "mean_feed_opinion": 0.05,
                                      "mean_stance_distance": 0.4,
                                      "mean_feed_emotion": 0.5}}]
    prompt = PC.direction_build_batch_prompt(fake)
    for banned in ("alpha", "gamma", "top_k", "social_pressure"):
        assert banned not in prompt, f"红线泄漏：{banned} 出现在提示词"
    assert "feed_mean_opinion" in prompt
    print("[selftest] OK 提示词红线断言通过（含 feed_mean_opinion，无参数语义）")
    # 模型构造（零网络）
    from model.llm_robustness_v2 import LLMRobustnessModel
    import tempfile
    tmp = str(Path(tempfile.gettempdir()) / "_stage_direction_selftest.jsonl")
    name, specs, job_arm = stage_specs(stage, arm)[0]
    if job_arm:
        os.environ["LLM_ABLATION"] = job_arm
    m = LLMRobustnessModel(seed=specs[0]["seed"], N=100 if stage in ("n100", "n100r", "n100b") else E.N_AGENTS,
                           steps=100 if stage in ("t100", "qwen-t100") else E.STEPS,
                           alpha=specs[0]["alpha"], gamma=specs[0]["gamma"],
                           use_llm=True, llm_archive_path=tmp, run_id=specs[0]["run_id"])
    if job_arm:
        assert m.ablation == job_arm, f"消融开关未生效: {m.ablation!r}"
    if stage == "randomfeed":
        import model.llm_robustness_v2 as M
        install_randomfeed()
        assert M.recommend_from_arrays is _random_recommend
    print(f"[selftest] OK 模型构造成功（ablation={getattr(m, 'ablation', '-')!r}，零网络调用）")
    print("[selftest] PASS")


def expand_stages(raw: str) -> list:
    """组合阶段展开为原子阶段序列。"""
    if raw == "qwen-all":
        return ["qwen-full", "qwen-abl"]
    if raw == "abl-all":
        return ["abl"]
    return [raw]


def main() -> None:
    ap = argparse.ArgumentParser(description="方向版提示词 · 原始实验套件重跑统一 runner")
    ap.add_argument("--stage", required=True,
                    choices=["qwen-full", "qwen-abl", "qwen-all", "randomfeed", "abl",
                             "abl-all", "r20", "n100", "n100r", "n100b", "t100", "qwen-t100"])
    ap.add_argument("--arm", default="all", help="消融臂（qwen-abl / abl 用）")
    ap.add_argument("--workers", type=int, default=8, help="父进程并行分片数")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--worker", action="store_true", help="子进程模式")
    ap.add_argument("--max-llm-calls", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        out = {}
        for s in ["qwen-full", "qwen-abl", "randomfeed", "abl", "r20", "n100", "n100r", "n100b",
                  "t100", "qwen-t100"]:
            out[s] = stage_progress(s)
        print(json.dumps(out, ensure_ascii=False))
        return

    stages = expand_stages(args.stage)
    setup_stage_env(stages[0])

    if args.plan:
        for s in stages:
            print_plan(s, "" if args.arm == "all" else args.arm, args.workers)
        return
    if args.self_test:
        for s in stages:
            print(f"===== self-test: {s} =====")
            self_test(s, "" if args.arm == "all" else args.arm)
        return

    if args.worker:
        # 子进程只可能被派发原子阶段
        stage = stages[0]
        arm = args.arm if args.arm != "all" else ""
        run_shard(stage, arm, args.shard, args.shards, args.max_llm_calls, args.force)
    else:
        for s in stages:
            arm = "all" if s in ("qwen-abl", "abl") else ""
            run_stage_group(s, arm, args.workers, args.max_llm_calls, args.force)
            print(f"[done] stage={s} 全部完成；进度：", flush=True)
            print(json.dumps(stage_progress(s), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
