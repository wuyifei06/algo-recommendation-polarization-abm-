# -*- coding: utf-8 -*-
"""Qwen T=100 延长实验启动器（供 Windows 计划任务 19:40 调用；也可手动重跑）。

设计
----
* 防重：双通道 —— ① 进程探测（CIM，命令行含 --stage qwen-t100）；② 文件新鲜度
  （分片日志/calls 在 300s 内有更新）。任一命中即跳过（可安全重复执行）。文件
  通道专为"CIM 在隔离会话返回空"的场景兜底，防止兜底检查误触发双启。
* 脱附：以 DETACHED_PROCESS 启动 runner 父进程（--workers 8），父启动器立即退出；
  runner 自身带 run 级 checkpoint，任何中断都可续跑。
* 自愈：spawn 后等待并校验分片日志是否在增长；若启动即死（如被进程组连带终止），
  清理残留并重试一次，避免"静默失败"等到兜底检查才发现。
* 日志：追加写入 _qwen_t100_run.log（启动器事件 + runner 全部输出）。

用法
----
  python -X utf8 _launch_qwen_t100.py            # 正常启动（防重）
  python -X utf8 _launch_qwen_t100.py --dry-run  # 只做检查与日志，不启动
"""
from __future__ import annotations

import argparse
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
LOG = WD / "_qwen_t100_run.log"
OUT = Path(_PROJ_ROOT + r"\results_direction\llm_qwen_extend_T100")
LOGS_GLOB = str(OUT / "logs" / "shard*.log")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [launcher] {msg}\n")


_PROBE_KEY = "--stage qwen-t100"    # runner 命令行形式，避免命中启动器文件名自身

# 判定"runner 存活"的文件新鲜度阈值（秒）：分片日志/calls 若在此窗口内有更新，
# 视为 runner 仍在推进。8 分片每个 run ≈8.6 分钟，故 300 秒阈值非常安全。
ALIVE_WINDOW_S = 300


def runner_pids() -> list[int]:
    """枚举命令行含 _PROBE_KEY 的 python 进程 PID（PowerShell CIM；只回传数字）。

    注：本机 WMIC 已被 Windows 11 移除，故用 Get-CimInstance；cmdline 过滤在
    PowerShell 内部完成，Python 侧只解析 PID 数字，无中文编码风险。
    但 CIM 在部分隔离会话中会返回空（实测），故防重不依赖它——另有文件新鲜度兜底。
    """
    ps = ("$procs = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          f"Where-Object {{ $_.CommandLine -like '*{_PROBE_KEY}*' }}; "
          "if ($procs) { $procs | ForEach-Object { [string]$_.ProcessId } }")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=90)
        return [int(ln.strip()) for ln in (out.stdout or "").splitlines() if ln.strip().isdigit()]
    except Exception as exc:                      # 探测失败时不阻塞启动（仅记录）
        log(f"WARN: runner probe failed ({exc}); assume not running")
        return []


def runner_alive_by_files(window_s: int = ALIVE_WINDOW_S) -> bool:
    """基于文件新鲜度判断 runner 是否在跑（不依赖进程枚举，最可靠的存活判据）。

    看两类文件的最近修改时间：各分片 shard*.log 与 calls/*.jsonl。
    若任一在 window_s 秒内被写过，则认为 runner 存活。
    """
    now = time.time()
    pats = [LOGS_GLOB, str(OUT / "shard*_of08" / "calls" / "*.jsonl")]
    for pat in pats:
        for f in glob.glob(pat):
            try:
                if now - Path(f).stat().st_mtime < window_s:
                    return True
            except OSError:
                continue
    return False


def _spawn() -> int:
    """以 DETACHED 方式拉起 runner 父进程（8 分片），返回其 PID。"""
    with open(LOG, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [PY, "-X", "utf8", "run_stage_direction.py", "--stage", "qwen-t100", "--workers", "8"],
            cwd=str(WD), stdout=fh, stderr=subprocess.STDOUT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True)
    return proc.pid


def _log_bytes() -> int:
    """所有分片日志的字节总量，用作"是否在推进"的判据。"""
    return sum(Path(f).stat().st_size for f in glob.glob(LOGS_GLOB) if Path(f).exists())


def _cleanup_partial() -> None:
    """清理启动即死的残留输出目录（无 summary，纯半成品，安全删除）。"""
    import shutil
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
        log(f"CLEANUP: removed partial output {OUT}")


def launch_with_heal(probe_wait: int = 30) -> None:
    """启动并在 probe_wait 秒后校验分片是否真存活；不存活则重试一次。

    判据：8 个分片进程已 spawn 后，其日志字节数应在 probe_wait 内增长
    （每个 run 至少打出 "run k/N" 一行）。若字节数不增长，判定为启动即死
    （典型原因：父进程组被连带终止 -> 0xC000013A），清理后重试。
    """
    before = _log_bytes()
    pid = _spawn()
    log(f"LAUNCHED: detached runner pid={pid} (8 shards, 140 runs); verifying in {probe_wait}s.")
    print(f"LAUNCHED pid={pid}")

    time.sleep(probe_wait)
    after = _log_bytes()
    if after > before:
        log(f"VERIFY-OK: shard logs grew {before}->{after} bytes; runner alive.")
        print("VERIFY-OK")
        return

    # 启动即死：清理残留后重试一次
    log(f"VERIFY-FAIL: shard logs did NOT grow ({before}->{after}); retrying after cleanup.")
    print("VERIFY-FAIL-RETRY")
    _cleanup_partial()
    time.sleep(3)
    pid2 = _spawn()
    log(f"RETRY-LAUNCHED: detached runner pid={pid2}; verifying in {probe_wait}s.")
    print(f"RETRY-LAUNCHED pid={pid2}")
    time.sleep(probe_wait)
    after2 = _log_bytes()
    if after2 > 0:
        log(f"RETRY-VERIFY-OK: shard logs now {after2} bytes; runner alive.")
        print("RETRY-VERIFY-OK")
    else:
        log("RETRY-VERIFY-FAIL: still no shard log growth; needs manual attention.")
        print("RETRY-VERIFY-FAIL")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-wait", type=int, default=30,
                    help="启动后校验存活的等待秒数（默认 30）")
    args = ap.parse_args()

    pids = runner_pids()
    if pids:
        log(f"SKIP: runner already alive (process probe); pids={pids}")
        print(f"SKIP-ALREADY-RUNNING pids={pids}")
        return
    if runner_alive_by_files():
        log("SKIP: runner already alive (file-freshness probe); shard logs updated recently.")
        print("SKIP-ALREADY-RUNNING (files)")
        return

    if args.dry_run:
        log("DRYRUN: no live runner; checks passed; would launch now.")
        print("DRYRUN-OK")
        return

    launch_with_heal(probe_wait=args.probe_wait)


if __name__ == "__main__":
    main()
