# -*- coding: utf-8 -*-
"""超兜底率 run 的自动重掷（论文既有口径：run 兜底率 >2% 即同种子重掷）。

流程：读当前 summary → 找出兜底率 >2% 的 T100 run → 删除其产物（checkpoint/calls/runs
+ 过滤 summary 行）→ 分块前台重跑（每块 ≤9 分钟，断点安全）→ 复检。

用法：venv python -X utf8 _reroll_over2.py [--chunks 2]
只处理 llm_qwen_extend_T100。无违例时直接退出。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
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

PY = _PYTHON
STAGE = Path(_PROJ_ROOT + r"\results_direction\llm_qwen_extend_T100")
RUNNER_CWD = Path(_PROBE_ROOT)


def load_runs():
    rows = {}
    for sf in glob.glob(str(STAGE / "shard*_of*" / "llm_robust_summary.csv")):
        try:
            with open(sf, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if int(float(row.get("step", -1))) == 100 and \
                       str(row.get("status", "")) in ("completed", "completed_with_fallback"):
                        rows[int(float(row["run_id"]))] = {
                            "rate": float(row["llm_fallback_rate"]),
                            "shard_csv": sf,
                        }
        except Exception:
            continue
    return rows


def find_shard(run_id: int) -> Path | None:
    for f in glob.glob(str(STAGE / "shard*_of*" / "checkpoints" / f"llm_robust_run_{run_id}.checkpoint.json")):
        return Path(f).parent.parent
    return None


def wipe_run(run_id: int, shard: Path):
    for pat in [f"checkpoints/llm_robust_run_{run_id}.*",
                f"calls/llm_robust_run_{run_id}.jsonl",
                f"runs/llm_robust_run_{run_id}_*"]:
        for f in glob.glob(str(shard / pat)):
            try:
                os.remove(f)
                print(f"  deleted {Path(f).name}")
            except OSError:
                pass
    # 过滤 summary 行（shard 级）
    for sf in [shard / "llm_robust_summary.csv"]:
        if not sf.exists():
            continue
        rows = list(csv.DictReader(open(sf, encoding="utf-8")))
        keep = [r for r in rows if int(float(r["run_id"])) != run_id]
        if len(keep) != len(rows):
            with open(sf, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(keep)
            print(f"  summary 过滤 run_{run_id}: {len(rows)}→{len(keep)}")


def chunk_run(minutes: float = 9.0):
    """前台跑 runner 至多 minutes 分钟，超时杀树（断点安全）。"""
    proc = subprocess.Popen(
        [PY, "-X", "utf8", "run_stage_direction.py", "--stage", "qwen-t100", "--workers", "8"],
        cwd=str(RUNNER_CWD), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time() + minutes * 60
    while time.time() < end:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(5)
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    return -9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2)
    args = ap.parse_args()

    runs = load_runs()
    bad = {rid: v for rid, v in runs.items() if v["rate"] > 0.02}
    if not bad:
        print("NO-OVER2: 全部 run 兜底率 ≤2%，无需重掷。")
        return

    print(f"发现 {len(bad)} 个超 2% run：{sorted(bad)}")
    for rid, v in sorted(bad.items()):
        shard = find_shard(rid)
        if shard:
            print(f"重掷 run_{rid} (rate={v['rate']:.3%}, shard={shard.name})")
            wipe_run(rid, shard)

    for i in range(args.chunks):
        runs = load_runs()
        bad = {rid: v for rid, v in runs.items() if v["rate"] > 0.02}
        if not bad:
            break
        print(f"[chunk {i+1}/{args.chunks}] 重跑中（{sorted(bad)}）...")
        chunk_run(9.0)

    runs = load_runs()
    bad = {rid: v["rate"] for rid, v in runs.items() if v["rate"] > 0.02}
    if bad:
        print(f"STILL-OVER2: {bad}（保留，报告给用户）")
    else:
        print("ALL-CLEAN: 重掷后全部 ≤2%。")


if __name__ == "__main__":
    main()
