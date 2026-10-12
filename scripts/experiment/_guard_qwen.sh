#!/usr/bin/env bash
# Qwen T=100 守护循环：保持 runner 存活直到 140 runs 完成。
# 设计：run 在循环内前台执行；若 runner 被杀/退出，15 秒后自动重启（checkpoint 续跑，几乎零损失）。
# 心跳文件 _guard_qwen.heartbeat 供外部判断守护是否存活；日志 _guard_qwen.log。
set -u
# --- portability shim (auto-added) ---------------------------------
PROBE="${ABM_PROBE_ROOT:-/d/21130/Desktop/大创过程/probe_提示词明确性}"
PROJ="${ABM_PROJ_ROOT:-/d/21130/Desktop/大创过程/实验代码/最终代码和数据}"
PY312="${ABM_PYTHON:-/c/Users/21130/AppData/Local/Programs/Python/Python312/python.exe}"
# -------------------------------------------------------------------
cd "$PROBE" || exit 1
D="$PROJ/results_direction/llm_qwen_extend_T100"
RUNLOG="_qwen_t100_run.log"
GLOG="_guard_qwen.log"
HB="_guard_qwen.heartbeat"

log() { echo "[$(date '+%F %T')] $*" >> "$GLOG"; }

# 防双守护：若已有心跳 <60 秒，说明另一个守护在跑，退出
if [ -f "$HB" ]; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
  if [ "$AGE" -lt 60 ]; then
    log "ANOTHER-GUARD-ACTIVE (heartbeat age ${AGE}s); exit."
    exit 0
  fi
fi

# 启动时检查：若 stage 文件 300 秒内仍在更新，说明已有 runner 在跑，不重复启动
NEWEST=$(find "$D" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
NOW=$(date +%s)
if [ -n "$NEWEST" ] && [ $(( NOW - NEWEST )) -lt 300 ]; then
  log "STARTUP: runner appears ALIVE (files fresh $(( NOW - NEWEST ))s ago); monitor-only mode."
fi

OWN=0   # 1 = 我们启动的 runner 刚退出（已知其不在跑）
while true; do
  date '+%F %T' > "$HB"

  # 暂停旗标（监控器兜底率超红线时写入）：不拉起，空转等待
  if [ -f "$PROBE/_pause.flag" ]; then
    log "PAUSED (flag present); idling."
    OWN=0
    sleep 60
    continue
  fi

  # 完成检查：completed run 数（agents_final 文件）—— 完整版目标 200
  DONE=$(find "$D"/shard*_of08/runs -name "*agents_final*" 2>/dev/null | wc -l)
  if [ "$DONE" -ge 200 ]; then
    log "ALL-DONE: $DONE/200 completed; guard exit."
    exit 0
  fi

  # 非自有场景：若文件新鲜，别人在跑，等待
  if [ "$OWN" = "0" ]; then
    NEWEST=$(find "$D" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)
    NOW=$(date +%s)
    if [ -n "$NEWEST" ] && [ $(( NOW - NEWEST )) -lt 300 ]; then
      log "RUNNING-ELSEWHERE (fresh $(( NOW - NEWEST ))s); wait."
      sleep 25
      continue
    fi
  fi

  log "START runner (done=$DONE/140)."
  "$PY312" -X utf8 run_stage_direction.py --stage qwen-t100 --workers 8 >> "$RUNLOG" 2>&1
  RC=$?
  log "RUNNER-EXIT rc=$RC (done≈$(find "$D"/shard*_of08/runs -name "*agents_final*" 2>/dev/null | wc -l)/140)."
  OWN=1
  sleep 15
done
