#!/usr/bin/env bash
# N100B 会话守护：等 Qwen T=100 完成(140/140)后接力启动 n100b（60 runs），带重启自愈。
# 心跳 _guard_n100b.heartbeat；日志 _guard_n100b.log。
set -u
# --- portability shim (auto-added) ---------------------------------
PROBE="${ABM_PROBE_ROOT:-/d/21130/Desktop/大创过程/probe_提示词明确性}"
PROJ="${ABM_PROJ_ROOT:-/d/21130/Desktop/大创过程/实验代码/最终代码和数据}"
PY312="${ABM_PYTHON:-/c/Users/21130/AppData/Local/Programs/Python/Python312/python.exe}"
# -------------------------------------------------------------------
cd "$PROBE" || exit 1
QD="$PROJ/results_direction/llm_qwen_extend_T100"
BD="$PROJ/results_direction/llm_n100_boost"
GLOG="_guard_n100b.log"
HB="_guard_n100b.heartbeat"
log(){ echo "[$(date '+%F %T')] $*" >> "$GLOG"; }
qwen_done(){ find "$QD"/shard*_of08/runs -name "*agents_final*" 2>/dev/null | wc -l; }
nb_done(){ find "$BD"/shard*_of08/runs -name "*agents_final*" 2>/dev/null | wc -l; }
fresh(){ local N=$(find "$1" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1); [ -z "$N" ] && return 1; [ $(( $(date +%s) - N )) -lt "$2" ]; }

# 防双守护
if [ -f "$HB" ]; then
  AGE=$(( $(date +%s) - $(stat -c %Y "$HB" 2>/dev/null || echo 0) ))
  if [ "$AGE" -lt 90 ]; then log "ANOTHER-GUARD-ACTIVE (hb age ${AGE}s); exit."; exit 0; fi
fi
log "guard start."

OWN=0
while true; do
  date '+%F %T' > "$HB"

  # 暂停旗标（监控器兜底率超红线时写入）：不拉起，空转等待
  if [ -f "$PROBE/_pause.flag" ]; then
    log "PAUSED (flag present); idling."
    OWN=0
    sleep 60
    continue
  fi

  ND=$(nb_done)
  if [ "$ND" -ge 60 ]; then log "ALL-DONE n100b: $ND/60; guard exit."; exit 0; fi

  QN=$(qwen_done)
  if [ "$QN" -lt 200 ]; then
    log "WAIT-QWEN (qwen=$QN/200, n100b=$ND/60)."
    sleep 60
    continue
  fi

  # Qwen 已完成 → 启动/续跑 n100b
  if [ "$OWN" = "0" ] && fresh "$BD" 300; then
    log "N100B-RUNNING-ELSEWHERE (files fresh); wait."
    sleep 60
    continue
  fi
  log "START n100b (done=$ND/60, qwen=$QN/140)."
  "$PY312" -X utf8 run_stage_direction.py --stage n100b --workers 8 >> _n100b_run.log 2>&1
  log "RUNNER-EXIT rc=$? (done=$(nb_done)/60)."
  OWN=1
  sleep 15
done
