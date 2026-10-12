# -*- coding: utf-8 -*-
"""N=100 规模档补重复（n100b，60 runs）启动器 —— 接力模式。

由 Windows 计划任务 N100B_Launch 自 21:30 起每 15 分钟调用一次；
每次运行按以下顺序判定，任一条件命中即退出（可安全反复执行）：

  1) n100b 已完成（boost 目录 60 个 run 全部 step=30 达标） -> SKIP-ALREADY-DONE
  2) n100b runner 正在跑                                  -> SKIP-ALREADY-RUNNING
  3) qwen-t100 runner 正在跑（不与其并行，避免兜底率升高） -> SKIP-WAIT-QWEN
  4) 否则：脱附启动 n100b（8 分片，DETACHED_PROCESS）

用法:
  python -X utf8 _launch_n100b.py            # 按上述判定
  python -X utf8 _launch_n100b.py --dry-run  # 只判定与写日志，不启动
"""
from __future__ import annotations

import argparse
import csv
import glob
import subprocess
import sys
import time
from datetime import datetime
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
WD = Path(_PROBE_ROOT)
OUT_ROOT = Path(_PROJ_ROOT + r"\results_direction")
BOOST_DIR = OUT_ROOT / "llm_n100_boost"
QWEN_DIR = OUT_ROOT / "llm_qwen_extend_T100"
LOG = WD / "_n100b_run.log"

EXPECTED_RUNS = 60
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

# 文件新鲜度阈值（秒）：分片日志/calls 在此窗口内有更新 => 该 stage 仍在推进。
# 8 分片每 run ≈8.6 分钟，300s 阈值安全。用于 CIM 失效时的兜底（尤其是"等 Qwen"门）。
ALIVE_WINDOW_S = 300


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [launcher] {msg}\n")


def probe_pids(keyword: str) -> list[int]:
    """命令行含 keyword 的 python 进程 PID（PowerShell CIM，只回传数字）。"""
    ps = ("$procs = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          f"Where-Object {{ $_.CommandLine -like '*{keyword}*' }}; "
          "if ($procs) { $procs | ForEach-Object { [string]$_.ProcessId } }")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=90)
        return [int(ln.strip()) for ln in (out.stdout or "").splitlines() if ln.strip().isdigit()]
    except Exception as exc:
        log(f"WARN: probe({keyword}) failed ({exc}); assume none")
        return []


def stage_alive_by_files(stage_dir: Path, window_s: int = ALIVE_WINDOW_S) -> bool:
    """文件新鲜度判活（不依赖进程枚举）：分片日志或 calls jsonl 近期有更新。"""
    now = time.time()
    pats = [str(stage_dir / "logs" / "shard*.log"),
            str(stage_dir / "shard*_of*" / "calls" / "*.jsonl")]
    for pat in pats:
        for f in glob.glob(pat):
            try:
                if now - Path(f).stat().st_mtime < window_s:
                    return True
            except OSError:
                continue
    return False


def done_runs() -> int:
    """已达标 run 数（boost 目录根 summary + 各分片 summary，按 run_id 去重）。"""
    done: set[int] = set()
    cands = [BOOST_DIR / "llm_robust_summary.csv"]
    cands += sorted(BOOST_DIR.glob("shard*_of*/llm_robust_summary.csv"))
    for path in cands:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if int(float(row.get("step", -1))) == 30 and str(row.get("status", "")) in {
                            "completed", "completed_with_fallback"}:
                        done.add(int(float(row["run_id"])))
        except Exception:
            continue
    return len(done)


def _spawn() -> int:
    """以 DETACHED 方式拉起 n100b runner 父进程（8 分片），返回 PID。"""
    with open(LOG, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [PY, "-X", "utf8", "run_stage_direction.py", "--stage", "n100b", "--workers", "8"],
            cwd=str(WD), stdout=fh, stderr=subprocess.STDOUT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True)
    return proc.pid


def _boost_log_bytes() -> int:
    return sum(Path(f).stat().st_size
               for f in glob.glob(str(BOOST_DIR / "logs" / "shard*.log")) if Path(f).exists())


def launch_with_heal(probe_wait: int = 30) -> None:
    """启动后校验分片日志是否增长；启动即死则清理残留并重试一次。"""
    before = _boost_log_bytes()
    pid = _spawn()
    log(f"LAUNCHED: detached n100b runner pid={pid} (8 shards, {EXPECTED_RUNS} runs); "
        f"verifying in {probe_wait}s.")
    print(f"LAUNCHED pid={pid}")

    time.sleep(probe_wait)
    after = _boost_log_bytes()
    if after > before:
        log(f"VERIFY-OK: n100b shard logs grew {before}->{after} bytes; runner alive.")
        print("VERIFY-OK")
        return

    log(f"VERIFY-FAIL: n100b logs did NOT grow ({before}->{after}); retry after cleanup.")
    print("VERIFY-FAIL-RETRY")
    import shutil
    shutil.rmtree(BOOST_DIR, ignore_errors=True)
    log(f"CLEANUP: removed partial {BOOST_DIR}")
    time.sleep(3)
    pid2 = _spawn()
    log(f"RETRY-LAUNCHED: detached n100b runner pid={pid2}; verifying in {probe_wait}s.")
    print(f"RETRY-LAUNCHED pid={pid2}")
    time.sleep(probe_wait)
    after2 = _boost_log_bytes()
    if after2 > 0:
        log(f"RETRY-VERIFY-OK: n100b logs now {after2} bytes; runner alive.")
        print("RETRY-VERIFY-OK")
    else:
        log("RETRY-VERIFY-FAIL: still no n100b log growth; needs manual attention.")
        print("RETRY-VERIFY-FAIL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-wait", type=int, default=30,
                    help="启动后校验存活的等待秒数（默认 30）")
    args = ap.parse_args()

    n_done = done_runs()
    if n_done >= EXPECTED_RUNS:
        log(f"SKIP-ALREADY-DONE: {n_done}/{EXPECTED_RUNS} runs completed.")
        print(f"SKIP-ALREADY-DONE {n_done}/{EXPECTED_RUNS}")
        return

    # 关键字用 "--stage n100b"（runner 命令行形式），避免命中本启动器文件名自身
    if probe_pids("--stage n100b") or stage_alive_by_files(BOOST_DIR):
        log(f"SKIP-ALREADY-RUNNING: n100b runner alive ({n_done}/{EXPECTED_RUNS} done).")
        print(f"SKIP-ALREADY-RUNNING {n_done}/{EXPECTED_RUNS}")
        return

    # 门 3：qwen 在跑则等待（双通道：进程 OR 文件新鲜度）——文件通道专防 CIM 失效导致误并行
    if probe_pids("--stage qwen-t100") or stage_alive_by_files(QWEN_DIR):
        log("SKIP-WAIT-QWEN: qwen-t100 still running; will retry on next tick.")
        print("SKIP-WAIT-QWEN")
        return

    if args.dry_run:
        log(f"DRYRUN: all gates passed ({n_done}/{EXPECTED_RUNS}); would launch now.")
        print("DRYRUN-OK")
        return

    launch_with_heal(probe_wait=args.probe_wait)


if __name__ == "__main__":
    main()
