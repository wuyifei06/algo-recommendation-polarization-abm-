"""在线消融（do-干预）启动器：Abl-1 / Abl-2 / Abl-3 机制分解。

为什么必须是**在线**消融
------------------------
离线重放（``_llm_extend/replay_cf.py``）只能复现"决策被消费"的结果，因为重放世界里
观点更新只消费存档的 ``stance_delta``。在重放里把 ``emotion_delta`` / ``express``
置零，观点轨迹**按构造不动** —— 那得到的零效应是人为构造的，不能解读为"该通道不重要"。
凡是要问"如果没有这条通道会怎样"，必须在真实 LLM 调用下重跑。见
``_llm_extend/_online_ablation_plan.md``。

三条在线消融
------------
  * ``positive_control`` —— Abl-1：``do(stance_delta=0)``。阳性对照 / 操纵检验。
  * ``no_post``          —— Abl-2：禁用 ``_append_post``，内容池冻结在 t=0 那批初始帖，
                            切断"表达 → 内容池 → 下一轮曝光"的自反馈回路。**最关键的一条**。
  * ``no_emotion``       —— Abl-3：``do(emotion_delta=0)``，情绪冻结在初值。

（Abl-4 随机 feed 已完成：``results/llm_randomfeed_control``。Abl-5 离线随机决策零 API。）

设计原则（与 run_llm_gradient_deepseek.py / run_llm_robust_grid.py 一致）
------------------------------------------------------------------------
1. **不修改 ``experiment_llm_robust.py`` 一个字符**：只在本进程内覆盖
   ``E.CONDITIONS`` / ``E.REPEATS`` / ``E.iter_specs``，退出即失效。
2. **prompt / 调用次数逐字不变**：消融只改"决策被消费"的那一步（
   ``model/llm_robustness_v2.py::_step_llm`` 的应用循环），不碰 prompt、不碰 Top-K、
   不增减 LLM 调用 → **成本与 baseline 完全一致**（每 run 90 次调用）。
3. **与 baseline 逐位配对**：``results/llm_gradient_noleak`` 用的基准种子是 31345，
   完整网格 8α × 2γ、R=10。本启动器把 run_id / seed 锚定在**同一网格顺序**上，
   于是同一 (α, γ, repeat) 的消融 run 与 baseline run **共享完全相同的初始条件**
   （观点 / 情绪 / 活动性 / λ 逐位相同）→ 消融 vs baseline 是**配对比较**，
   而不是两组独立样本。运行前会用 baseline 的 manifest 逐条核对种子。

用法
----
    # 免费：打印设计与成本（含与 baseline 的配对核对）
    python run_llm_ablation.py --ablation no_post --scale probe --plan

    # 免费：跑前自检（字段/解析器/模型 + 配对核对），不发起任何网络调用
    python run_llm_ablation.py --ablation no_post --scale probe --self-test

    # 正式跑：Abl-2 探针（α=1.0 × 2γ × R=10 = 20 runs）
    python run_llm_ablation.py --ablation no_post --scale probe

    # 正式跑：Abl-2 标准档（8α × 2γ × R=10 = 160 runs），16 路并行示例：
    for i in $(seq 0 15); do
      python run_llm_ablation.py --ablation no_post --scale standard \
          --shards 16 --shard $i &
    done; wait
    python run_llm_ablation.py --ablation no_post --scale standard --merge

断点续跑：同一 output-dir 重复执行即可，已完成的 run 会被跳过（沿用父脚本快路径）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import experiment_llm_robust as E

# --------------------------------------------------------------------------
# 与 baseline 配对所必需的锚点（不得随意改动）
# --------------------------------------------------------------------------
BASELINE_DIR = E.PROJECT_ROOT / "results" / "llm_gradient_noleak"
BASELINE_BASE_SEED = 31345          # 见 results/llm_gradient_noleak/manifest.json
BASELINE_REPEATS = 10

# 完整梯度网格（**顺序即 baseline 的 run_id 顺序**）。消融臂沿用同一顺序 +
# 同一 seed 公式，于是只跑子集时 run_id → seed 仍与 baseline 对齐。
FULL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
FULL_GAMMAS = [0.1, 1.0]
FULL_GRID: List[Tuple[float, float]] = [(a, g) for a in FULL_ALPHAS for g in FULL_GAMMAS]

# 加强档新增重复用的种子偏移块（与 baseline 种子块不重叠）
EXT_SEED_OFFSET = 500000

ABLATIONS: Dict[str, Dict[str, str]] = {
    "positive_control": {
        "label": "Abl-1 阳性对照 do(stance_delta=0)",
        "dirname": "llm_ablation_positive_control",
    },
    "no_post": {
        "label": "Abl-2 切断自反馈回路（内容池冻结）",
        "dirname": "llm_ablation_no_post",
    },
    "no_emotion": {
        "label": "Abl-3 do(emotion_delta=0)（情绪冻结）",
        "dirname": "llm_ablation_no_emotion",
    },
}

SCALES: Dict[str, Dict] = {
    "probe": {
        "label": "探针",
        "conds": [(1.0, 0.1), (1.0, 1.0)],
        "repeats": BASELINE_REPEATS,
        "suffix": "_probe",
    },
    "standard": {
        "label": "标准",
        "conds": FULL_GRID,
        "repeats": BASELINE_REPEATS,
        "suffix": "",
    },
    "strong": {
        "label": "加强",
        "conds": FULL_GRID,
        "repeats": 2 * BASELINE_REPEATS,
        "suffix": "_strong",
    },
}

# 实测单价与耗时：来自 results/llm_gradient_noleak 的 160-run 实测（¥0.061019/run）。
MEASURED_COST_PER_RUN_YUAN = 0.0610
MEASURED_MIN_PER_RUN = 2.2


# --------------------------------------------------------------------------
# run_id / seed 锚定
# --------------------------------------------------------------------------
def _grid_index(alpha: float, gamma: float) -> int:
    i = FULL_ALPHAS.index(float(alpha))
    j = FULL_GAMMAS.index(float(gamma))
    return i * len(FULL_GAMMAS) + j


def _anchored_seed(grid_index: int, repeat: int, repeats: int) -> int:
    """复现 baseline 的 seed 公式；超出 baseline 重复数的部分进入独立偏移块。

    baseline: seed = 31345 + grid_index*10 + repeat（repeat ∈ [0,10)）。
    """
    if repeat < BASELINE_REPEATS:
        return BASELINE_BASE_SEED + grid_index * BASELINE_REPEATS + repeat
    extra = repeats - BASELINE_REPEATS
    return (
        BASELINE_BASE_SEED
        + EXT_SEED_OFFSET
        + grid_index * max(extra, 1)
        + (repeat - BASELINE_REPEATS)
    )


def make_specs(conds: List[Tuple[float, float]], repeats: int) -> List[Dict]:
    """生成与 baseline 锚定的 spec 列表（run_id 唯一，seed 与 baseline 对齐）。"""
    specs: List[Dict] = []
    for alpha, gamma in conds:
        gi = _grid_index(alpha, gamma)
        for repeat in range(repeats):
            specs.append({
                "run_id": gi * repeats + repeat,
                "repeat": repeat,
                "alpha": float(alpha),
                "gamma": float(gamma),
                "seed": int(_anchored_seed(gi, repeat, repeats)),
            })
    return specs


def load_baseline_index() -> Dict[Tuple[float, float, int], int]:
    """从 baseline manifest 建立 {(alpha, gamma, repeat): seed} 索引。"""
    path = BASELINE_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 baseline manifest：{path}。配对核对无法进行 —— 请先确认 "
            f"results/llm_gradient_noleak 存在。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    index: Dict[Tuple[float, float, int], int] = {}
    for run in payload.get("runs", []):
        key = (float(run["alpha"]), float(run["gamma"]), int(run["repeat"]))
        index[key] = int(run["seed"])
    return index


def verify_pairing(specs: List[Dict]) -> Tuple[int, List[str]]:
    """逐条核对：baseline 覆盖到的 (α, γ, repeat) 其 seed 必须完全一致。"""
    baseline = load_baseline_index()
    checked = 0
    problems: List[str] = []
    for spec in specs:
        key = (spec["alpha"], spec["gamma"], spec["repeat"])
        if key in baseline:
            checked += 1
            if baseline[key] != spec["seed"]:
                problems.append(
                    f"(α={spec['alpha']}, γ={spec['gamma']}, rep={spec['repeat']}) "
                    f"expected seed={baseline[key]}, got {spec['seed']}"
                )
    return checked, problems


# --------------------------------------------------------------------------
# 输出目录
# --------------------------------------------------------------------------
def resolve_output_dir(ablation: str, scale: str) -> Path:
    info = SCALES[scale]
    name = ABLATIONS[ablation]["dirname"] + info["suffix"]
    return E.PROJECT_ROOT / "results" / name


def shard_output_dir(base: Path, shard: int, shards: int) -> Path:
    if shards <= 1:
        return base
    return base / f"shard{shard:02d}_of{shards:02d}"


def clear_proxies() -> None:
    """沙箱/内网环境里的 HTTP(S)_PROXY 会让 requests 撞 ProxyError，必须清掉再直连。"""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ.setdefault("NO_PROXY", "*")


def shard_completed(shard_dir: Path, steps: int) -> int:
    """读分片 summary，返回"已完成（且 step 达标、状态正常）"的 run 数。"""
    import pandas as pd

    summary = shard_dir / "llm_robust_summary.csv"
    if not summary.exists():
        return 0
    try:
        df = pd.read_csv(summary)
    except Exception:
        return 0
    if "step" not in df.columns:
        return 0
    ok = (df["step"].astype(float) == float(steps))
    if "status" in df.columns:
        ok &= df["status"].astype(str).isin({"completed", "completed_with_fallback"})
    return int(ok.sum())


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------
def print_plan(ablation: str, scale: str) -> None:
    info = SCALES[scale]
    specs = make_specs(info["conds"], info["repeats"])
    n = len(specs)
    out = resolve_output_dir(ablation, scale)
    checked, problems = verify_pairing(specs)

    print("=" * 78)
    print(f"[ablation] 消融臂：{ABLATIONS[ablation]['label']}")
    print(f"[ablation] 档位：{info['label']}（{scale}）")
    print(f"[ablation] α 取值：{[a for a, _ in info['conds']]}")
    print(f"[ablation] γ 取值：{sorted({g for _, g in info['conds']})}")
    print(f"[ablation] R = {info['repeats']}，共 {n} runs")
    print(f"[ablation] N = {E.N_AGENTS}，T = {E.STEPS}（与 baseline 一致）")
    print(f"[ablation] 与 baseline 配对锚点：{BASELINE_DIR}")
    print(f"[ablation] 配对核对：{checked}/{n} 个 spec 在 baseline 中有对应，"
          f"seed 冲突 {len(problems)} 条")
    for p in problems[:10]:
        print(f"[ablation]   ⚠ {p}")
    print(f"[ablation] 输出目录：{out}")
    print(f"[ablation] 实测单价 ¥{MEASURED_COST_PER_RUN_YUAN:.4f}/run（DeepSeek-flash，T=30）")
    print(f"[ablation] 估算成本：¥{n * MEASURED_COST_PER_RUN_YUAN:.2f}")
    print(f"[ablation] 单进程耗时 ≈ {n * MEASURED_MIN_PER_RUN / 60:.1f} h"
          f"（约 {MEASURED_MIN_PER_RUN} min/run）；"
          f"16 路并行 ≈ {n * MEASURED_MIN_PER_RUN / 16 / 60:.2f} h")
    print("=" * 78)


def print_three_arm_table() -> None:
    print("\n三臂 × 三档 成本矩阵（¥，按 ¥0.0610/run 实测单价）")
    print(f"{'档位':<8}{'runs/臂':>9}{'¥/臂':>10}{'三臂合计':>12}")
    for scale in ("probe", "standard", "strong"):
        info = SCALES[scale]
        n = len(info["conds"]) * info["repeats"]
        per = n * MEASURED_COST_PER_RUN_YUAN
        print(f"{info['label']:<8}{n:>9}{per:>10.2f}{3 * per:>12.2f}")
    print()


# --------------------------------------------------------------------------
# merge（多分片合并）
# --------------------------------------------------------------------------
def merge_shards(base: Path) -> None:
    import pandas as pd

    shard_dirs = sorted(p for p in base.glob("shard*_of*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"{base} 下没有分片目录，无需合并。")

    frames = []
    all_runs: List[Dict] = []
    for d in shard_dirs:
        summary = d / "llm_robust_summary.csv"
        if not summary.exists():
            print(f"[ablation][merge] 跳过（无 summary）：{d}")
            continue
        frames.append(pd.read_csv(summary))
        manifest = d / "manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            all_runs.extend(payload.get("runs", []))

    if not frames:
        raise RuntimeError("没有可合并的 summary。")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates("run_id")
    merged = merged.sort_values("run_id").reset_index(drop=True)
    out_summary = base / "llm_robust_summary.csv"
    merged.to_csv(out_summary, index=False)

    by_run = {int(r["run_id"]): r for r in all_runs}
    merged_manifest = {
        "experiment": "llm_robust_ablation",
        "source": "merged_from_shards",
        "shards": [d.name for d in shard_dirs],
        "runs": [by_run[k] for k in sorted(by_run)],
    }
    for name in ("ablation", "scale"):
        side = base / f"_{name}.json"
        if side.exists():
            merged_manifest[name] = json.loads(side.read_text(encoding="utf-8"))
    (base / "manifest.json").write_text(
        json.dumps(merged_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_cost = float(merged["estimated_cost"].sum())
    print(f"[ablation][merge] 合并 {len(merged)} runs → {out_summary}")
    print(f"[ablation][merge] 合计成本 ¥{total_cost:.4f}")


# --------------------------------------------------------------------------
# 分片进度 / 父进程并行 / 质量门
# --------------------------------------------------------------------------
def infer_shards(base_out: Path) -> int:
    """从 <out>/shardNN_ofMM 目录名反推分片数；没有分片目录则返回 1。"""
    dirs = sorted(base_out.glob("shard*_of*"))
    if not dirs:
        return 1
    m = re.search(r"_of(\d+)$", dirs[0].name)
    return int(m.group(1)) if m else 1


def _arm_progress(base_out: Path, expected: int) -> Tuple[int, float, float]:
    """返回 (已完成 runs, 已用分钟, 估算成本)。用最早 calls 文件 mtime 当起点。"""
    shards = infer_shards(base_out)
    done = sum(shard_completed(shard_output_dir(base_out, k, shards), E.STEPS)
               for k in range(shards))
    files = glob.glob(str(base_out / "shard*_of*" / "calls" / "*.jsonl")) + \
        glob.glob(str(base_out / "calls" / "*.jsonl"))
    elapsed_min = 0.0
    if files:
        t0 = min(os.path.getmtime(f) for f in files)
        elapsed_min = max(0.0, (time.time() - t0) / 60.0)
    cost = 0.0
    summary = base_out / "llm_robust_summary.csv"
    if summary.exists():
        try:
            import pandas as pd
            cost = float(pd.read_csv(summary)["estimated_cost"].sum())
        except Exception:
            cost = done * MEASURED_COST_PER_RUN_YUAN
    else:
        cost = done * MEASURED_COST_PER_RUN_YUAN
    return done, elapsed_min, cost


def report_all_status(scale: str) -> None:
    """零参数进度：扫描三臂（分片数自动推断），不需要指定 --ablation/--shards。"""
    info = SCALES[scale]
    expected = len(info["conds"]) * info["repeats"]

    def _display(base_out, name, label):
        done, el, cost = _arm_progress(base_out, expected)
        pct = 100.0 * done / expected if expected else 0.0
        filled = int(round(pct / 5))
        bar = "█" * filled + "·" * (20 - filled)
        eta = ""
        if 0 < done < expected and el > 1:
            eta = f"  ETA≈{el / done * (expected - done):.0f}min"
        print(f"  [{bar}] {pct:5.1f}%  {done:>4}/{expected}  ¥{cost:>6.2f}  "
              f"已用{el:>5.1f}min{eta}   {label}")
        print(f"       {base_out}")

    print(f"[status] 档位={info['label']}（{scale}）｜每臂 {expected} runs｜三臂预算 ¥{3*expected*MEASURED_COST_PER_RUN_YUAN:.2f}")
    total_done, total_cost = 0, 0.0
    for name, meta in ABLATIONS.items():
        base_out = resolve_output_dir(name, scale)
        d, _, c = _arm_progress(base_out, expected)
        total_done += d
        total_cost += c
        _display(base_out, name, f"{name} — {meta['label']}")
    nproc = len([1 for _ in subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"],
        capture_output=True, text=True).stdout.splitlines() if "python" in _.lower()])
    print(f"[status] 三臂合计 {total_done}/{3*expected} runs，已花 ≈¥{total_cost:.2f}"
          f"｜存活 python 进程 {nproc} 个")


def report_status(base_out: Path, shards: int, expected: int) -> None:
    total = 0
    print(f"[ablation][status] {base_out}")
    for k in range(max(1, shards)):
        d = shard_output_dir(base_out, k, shards)
        c = shard_completed(d, E.STEPS)
        total += c
        print(f"[ablation][status]   shard{k:02d}: {c} runs  {'(不存在)' if not d.exists() else ''}")
    print(f"[ablation][status] 合计 {total}/{expected} runs；成本≈¥{total * MEASURED_COST_PER_RUN_YUAN:.2f}")


def summarize_quality(base_out: Path, expected: int) -> None:
    """合并后的质量门：分层兜底率表 + 标出高兜底 run（>2%）。"""
    import pandas as pd

    summary = base_out / "llm_robust_summary.csv"
    if not summary.exists():
        print("[ablation][quality] 无 summary，跳过质量门")
        return
    df = pd.read_csv(summary)
    print(f"[ablation][quality] {len(df)}/{expected} runs；"
          f"成本 ¥{df['estimated_cost'].sum():.4f}")
    if "llm_fallback_rate" in df.columns:
        g = df.groupby(["alpha", "gamma"])["llm_fallback_rate"].agg(["mean", "max", "count"])
        print("[ablation][quality] α×γ 兜底率（mean / max / n）：")
        for (a, gm), row in g.iterrows():
            print(f"[ablation][quality]   α={a:<5} γ={gm:<4} "
                  f"mean={row['mean']:.4%} max={row['max']:.4%} n={int(row['count'])}")
        bad = df[df["llm_fallback_rate"] > 0.02]
        if len(bad):
            print(f"[ablation][quality] ⚠️ {len(bad)} 个 run 兜底率 >2%（建议重采）："
                  f"{sorted(int(r) for r in bad['run_id'])[:40]}")
        else:
            print("[ablation][quality] ✓ 无 run 兜底率 >2%")
    if "llm_silent_step_rate" in df.columns:
        print(f"[ablation][quality] 整步全沉默率 mean={df['llm_silent_step_rate'].mean():.4f} "
              f"max={df['llm_silent_step_rate'].max():.4f}")


def run_parallel_parent(args, base_out: Path, repeats: int) -> None:
    """父进程：拉起 W 个分片子进程 → 轮询 → 合并 → 质量门。"""
    info = SCALES[args.scale]
    expected = len(info["conds"]) * repeats
    W = int(args.parallel)
    logs_dir = base_out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # 臂/档位标识写进合并目录，供 merge_shards 汇进总 manifest（写成合法 JSON 值）
    (base_out / "_ablation.json").write_text(json.dumps(args.ablation), encoding="utf-8")
    (base_out / "_scale.json").write_text(json.dumps(args.scale), encoding="utf-8")
    # 预先建好全部分片目录：16 个进程并发 mkdir 会撞杀毒/同步盘的瞬占 PermissionError
    for k in range(W):
        shard_output_dir(base_out, k, W).mkdir(parents=True, exist_ok=True)

    script = str(Path(__file__).resolve())
    cmd_base = [sys.executable, script, "--ablation", args.ablation, "--scale", args.scale,
                "--base-seed", str(args.base_seed), "--shards", str(W)]
    if args.output_dir:
        cmd_base += ["--output-dir", str(args.output_dir)]
    if args.force:
        cmd_base += ["--force"]
    if args.max_llm_calls:
        cmd_base += ["--max-llm-calls", str(args.max_llm_calls)]

    procs = []
    for k in range(W):
        log = open(logs_dir / f"shard{k:02d}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(cmd_base + ["--shard", str(k)],
                                cwd=str(E.PROJECT_ROOT), stdout=log,
                                stderr=subprocess.STDOUT, env=dict(os.environ))
        procs.append((k, proc, log))
    print(f"[ablation][parent] 已拉起 {W} 分片 | {ABLATIONS[args.ablation]['label']} | "
          f"{info['label']} | {expected} runs | 预算 ¥{expected * MEASURED_COST_PER_RUN_YUAN:.2f}")
    print(f"[ablation][parent] 日志：{logs_dir}")

    t0 = time.time()
    try:
        while True:
            alive = [k for k, p, _ in procs if p.poll() is None]
            done = sum(shard_completed(shard_output_dir(base_out, k, W), E.STEPS)
                       for k, _, _ in procs)
            print(f"[ablation][parent] {done}/{expected} runs | {(time.time()-t0)/60:.1f} min | "
                  f"存活 {len(alive)}/{W} | 成本≈¥{done * MEASURED_COST_PER_RUN_YUAN:.2f}", flush=True)
            if not alive:
                break
            time.sleep(max(5, args.poll))
    finally:
        for _, _, log in procs:
            log.close()

    codes = {k: p.returncode for k, p, _ in procs}
    bad = {k: c for k, c in codes.items() if c != 0}
    print(f"[ablation][parent] 退出码：{codes}")
    if bad:
        print(f"[ablation][parent] ⚠️ 非零退出 {bad} → 看 {logs_dir}/shard??.log")

    merge_shards(base_out)
    summarize_quality(base_out, expected)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="在线消融（do-干预）启动器：Abl-1/2/3 机制分解（与 baseline 逐位配对）"
    )
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), required=False,
                        help="消融类型：positive_control / no_post / no_emotion")
    parser.add_argument("--scale", choices=sorted(SCALES), default="probe",
                        help="档位：probe(20) / standard(160) / strong(320)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="覆盖默认输出目录")
    parser.add_argument("--base-seed", type=int, default=BASELINE_BASE_SEED,
                        help=f"必须等于 baseline 基准种子 {BASELINE_BASE_SEED}（保证配对）")
    parser.add_argument("--shards", type=int, default=1,
                        help="并行分片总数（>1 时写入 <out>/shardNN_ofMM/）")
    parser.add_argument("--shard", type=int, default=0, help="本进程处理的分片下标（从 0 起）")
    parser.add_argument("--parallel", type=int, default=0,
                        help="父进程模式：由本进程拉起 W 个分片子进程并等待，最后自动合并")
    parser.add_argument("--poll", type=int, default=60, help="父进程进度轮询间隔（秒）")
    parser.add_argument("--status", action="store_true", help="只打印各分片进度后退出")
    parser.add_argument("--max-llm-calls", type=int, default=None, help="可选调用上限（cost guard）")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    parser.add_argument("--self-test", action="store_true", help="只跑前自检，不发起任何调用")
    parser.add_argument("--plan", action="store_true", help="只打印设计与成本估算，不运行")
    parser.add_argument("--merge", action="store_true", help="合并 --shards 产生的分片结果后退出")
    parser.add_argument("--table", action="store_true", help="只打印三臂×三档成本矩阵")
    args = parser.parse_args()
    # 沙箱/内网里的 HTTP(S)_PROXY 会让 requests 撞 ProxyError ⇒ 一律清掉再直连。
    clear_proxies()

    if args.table:
        print_three_arm_table()
        return

    if not args.ablation:
        parser.error("--ablation 为必填（positive_control / no_post / no_emotion）")

    info = SCALES[args.scale]
    base_out = (args.output_dir or resolve_output_dir(args.ablation, args.scale)).resolve()

    if args.merge:
        merge_shards(base_out)
        return

    if args.status:
        report_status(base_out, args.parallel or args.shards,
                      len(info["conds"]) * info["repeats"])
        return

    if args.base_seed != BASELINE_BASE_SEED:
        raise SystemExit(
            f"--base-seed 必须为 {BASELINE_BASE_SEED} 才能与 baseline 逐位配对；"
            f"收到 {args.base_seed}。"
        )

    if args.plan:
        print_plan(args.ablation, args.scale)
        print_three_arm_table()
        return

    # ---- 父进程模式：拉起 W 个分片子进程 → 等待 → 合并 ----
    if args.parallel and args.parallel > 1:
        run_parallel_parent(args, base_out, info["repeats"])
        return

    # ---- 进程内覆盖（不落盘，不影响原作者文件）----
    E.CONDITIONS = list(info["conds"])
    E.REPEATS = int(info["repeats"])

    specs_all = make_specs(info["conds"], info["repeats"])
    specs = [s for s in specs_all if (s["run_id"] % args.shards) == args.shard]
    # ⚠️ 分片内**顺序随机化**（固定 seed，可复现）：run_id 随 α/γ 递增，
    #    若按 run_id 升序处理，"批次内时间"会与实验因子共线 —— 任何一段服务抖动
    #    都会精确砸在某一档因子上，伪装成效应（见 skill llm-sim-batch-orchestration §7.1）。
    random.Random(20260922 + args.shard).shuffle(specs)

    # 配对核对：硬性门槛，不通过直接拒绝开跑（防止与 baseline 无法配对的种子被使用）
    checked, problems = verify_pairing(specs_all)
    if problems:
        raise SystemExit(
            "配对核对失败，拒绝开跑（否则消融与 baseline 无法配对）：\n  "
            + "\n  ".join(problems[:10])
        )

    out_dir = shard_output_dir(base_out, args.shard, args.shards)

    # ⚠️ 关键（2026-09-22 实盘踩到）：必须把**分片目录显式交给 run_experiment**。
    #    父脚本 `experiment_llm_robust.run_experiment` 用的是 `args.output_dir`，
    #    若为 None 会回落到 `PROJECT_ROOT/results/llm_robust` —— 于是 12 个分片
    #    会同时写同一个目录、在 `llm_robust_summary.csv` 上互相抢 Windows 文件锁，
    #    表现为进程 100% CPU 空转、结果散落到错误的目录（表面像"静默死掉"）。
    args.output_dir = out_dir
    assert Path(args.output_dir).resolve() == out_dir.resolve(), "输出目录解析不一致"

    # 消融开关：写进环境变量，供 model 的 __init__ 回落读取（断点续跑也生效）
    os.environ["LLM_ABLATION"] = args.ablation

    # 替换 iter_specs：返回锚定后的 spec（保证 run_id → seed 与 baseline 对齐）
    def anchored_iter_specs(_base_seed: int) -> Iterable[Dict]:
        for spec in specs:
            yield dict(spec)

    E.iter_specs = anchored_iter_specs

    print(f"[ablation] {ABLATIONS[args.ablation]['label']} | 档位={info['label']} | "
          f"本分片 runs={len(specs)}/{len(specs_all)}（shard {args.shard}/{args.shards}）")
    print(f"[ablation] 配对核对通过：{checked}/{len(specs_all)} 个 spec 与 baseline 种子一致")
    print(f"[ablation] 输出：{out_dir}")
    print(f"[ablation] 估算成本 ≈ ¥{len(specs) * MEASURED_COST_PER_RUN_YUAN:.2f}"
          f"（整臂 ¥{len(specs_all) * MEASURED_COST_PER_RUN_YUAN:.2f}）")

    if args.self_test:
        E.self_test()
        from model.llm_robustness_v2 import LLMRobustnessModel
        import tempfile
        # 自检：消融开关必须被模型正确接收（不发起任何调用，也不落任何 run 目录）
        prev = os.environ.get("LLM_ABLATION")
        selftest_archive = str(Path(tempfile.gettempdir()) / "_ablation_selftest.jsonl")
        try:
            for name in sorted(LLMRobustnessModel.VALID_ABLATIONS):
                os.environ["LLM_ABLATION"] = name
                probe = LLMRobustnessModel(
                    seed=1, N=E.N_AGENTS, steps=E.STEPS, alpha=1.0, gamma=0.1,
                    use_llm=True, llm_archive_path=selftest_archive, run_id=0,
                )
                assert probe.ablation == name, f"ablation 未生效：{probe.ablation!r}"
            # 未知取值必须报错
            try:
                LLMRobustnessModel(seed=1, N=E.N_AGENTS, steps=E.STEPS,
                                   ablation="bogus", use_llm=True,
                                   llm_archive_path=selftest_archive, run_id=0)
            except ValueError:
                pass
            else:
                raise AssertionError("未知 ablation 未被拒绝。")
        finally:
            if prev is None:
                os.environ.pop("LLM_ABLATION", None)
            else:
                os.environ["LLM_ABLATION"] = prev
        print("[ablation] self-test 通过：消融开关生效、未知取值被拒；未发起任何网络调用。")
        return

    # 正式跑：此刻才创建 run 目录与标识文件（避免 self-test 留下空目录）
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_ablation.json").write_text(
        json.dumps({
            "ablation": args.ablation,
            "label": ABLATIONS[args.ablation]["label"],
            "scale": args.scale,
            "base_seed": args.base_seed,
            "baseline_dir": str(BASELINE_DIR),
            "runs": len(specs),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "ABLATION.txt").write_text(
        f"{ABLATIONS[args.ablation]['label']} | scale={args.scale} | "
        f"shard={args.shard}/{args.shards}\n"
        f"环境变量 LLM_ABLATION={args.ablation}（成本与 baseline 完全一致：90 calls/run）\n",
        encoding="utf-8",
    )

    E.run_experiment(args)

    # 兜底断言：结果必须落在预期目录，否则说明又写跑偏了（立即炸掉而不是静默污染别处）
    if not (out_dir / "llm_robust_summary.csv").exists():
        raise RuntimeError(
            f"summary 未落在预期目录 {out_dir}（可能又回落到默认 results/llm_robust）！"
        )

    # 跑完后给 manifest 打上消融标签（父脚本 manifest 是 llm_robust 通用格式）
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["experiment"] = "llm_robust_ablation"
        payload["ablation"] = args.ablation
        payload["ablation_label"] = ABLATIONS[args.ablation]["label"]
        payload["scale"] = args.scale
        payload["shard"] = args.shard
        payload["shards"] = args.shards
        payload["baseline_dir"] = str(BASELINE_DIR)
        payload["update_semantics"] = (
            "消融：仅改变 _step_llm 应用循环的消费方式；prompt/Top-K/调用次数逐字不变"
        )
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"[ablation] 完成。结果：{out_dir}")


if __name__ == "__main__":
    main()
