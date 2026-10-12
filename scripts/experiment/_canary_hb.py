# -*- coding: utf-8 -*-
"""Canary：验证 Task Scheduler 拉起的进程能否长期存活（每 5 秒写一次心跳，持续 30 分钟）。"""
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "_canary_task_hb.txt"
t0 = time.time()
with open(OUT, "a", encoding="utf-8") as fh:
    fh.write(f"=== canary start {time.strftime('%F %T')} ===\n")
for i in range(360):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} tick={i} elapsed={int(time.time()-t0)}s\n")
    time.sleep(5)
with open(OUT, "a", encoding="utf-8") as fh:
    fh.write(f"=== canary end {time.strftime('%F %T')} (30min full) ===\n")
