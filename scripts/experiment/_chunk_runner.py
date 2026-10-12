# -*- coding: utf-8 -*-
"""T100 分块执行器（定时任务专用）：跑 N 分钟后受控退出。

用途：定时任务会话无法维持长命后台进程（会话结束→进程树被杀）。
本脚本在"单次会话内"以 8 个分片进程跑满 N 分钟，然后主动 taskkill 分片
（checkpoint 每步落盘，断点安全），脚本自身正常退出。定时任务每小时
可多次调用本脚本推进 T100，直到其完成。

用法：
  python _chunk_runner.py            # 默认跑 9 分钟
  python _chunk_runner.py --minutes 8
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
PYTHON = _PYTHON
RD = Path(_PROJ_ROOT + r"\results_direction")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=9.0)
    ap.add_argument("--stage", default="t100")
    args = ap.parse_args()

    done_flag = RD / "_scheduler_done.flag"
    if done_flag.exists():
        print("全部完成（done flag），无需分块。")
        return

    shards = 8
    procs = []
    for s in range(shards):
        procs.append(subprocess.Popen(
            [PYTHON, "-X", "utf8", "run_stage_direction.py", "--stage", args.stage,
             "--shards", str(shards), "--shard", str(s)],
            cwd=str(HERE),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    end = time.time() + args.minutes * 60
    while time.time() < end:
        if all(p.poll() is not None for p in procs):
            break                      # 全部自然退出（阶段完成）
        time.sleep(5)

    killed = 0
    for p in procs:
        if p.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            killed += 1
    print(f"[chunk] 完成 {args.stage} 分块：{args.minutes} 分钟，主动终止在途分片 {killed} 个（断点已落盘）")


if __name__ == "__main__":
    main()
