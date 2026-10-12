# -*- coding: utf-8 -*-
"""兜底率实时监控器 —— Qwen T=100 / N100B 两批。

用法：
  python -X utf8 _fb_watch.py --once     # 单次检查并打印（供自动化调用）
  python -X utf8 _fb_watch.py            # 循环模式，每 4 分钟检查一次（本机后台任务）

输出：
  _fb_watch.log     每行一条状态记录
  _fb_alerts.log    等级>=ALERT 时的详细警报
  _fb_watch_state.json  上次快照（算边际率用）

等级判定（用户硬约束：兜底率 ≤2%）：
  OK     累积/边际均 <=2%，无已完成 run 超 2%
  WATCH  累积 >2%（早期爆发稀释中，边际正常）——信息级
  ALERT  边际 >2%（持续）或 任一已完成 run >2%
  CRIT   边际 >=5% 或 累积 >=4%
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime
from pathlib import Path
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

PROBE = Path(_PROBE_ROOT)
RD = Path(_PROJ_ROOT + r"\results_direction")
WATCH_LOG = PROBE / "_fb_watch.log"
ALERT_LOG = PROBE / "_fb_alerts.log"
STATE = PROBE / "_fb_watch_state.json"

STAGES = [
    # (stage名, 目录, 分片数, 每run步数, 每步响应数)
    ("qwen-t100", RD / "llm_qwen_extend_T100", 8, 100, 30),
    ("n100b", RD / "llm_n100_boost", 8, 30, 30),
]


def newest_checkpoint_per_shard(stage_dir: Path):
    """每分片取 mtime 最新的一个 checkpoint，返回 [(shard, data)]。"""
    out = []
    for sh in sorted(stage_dir.glob("shard*_of*")):
        cps = glob.glob(str(sh / "checkpoints" / "*.checkpoint.json"))
        if not cps:
            continue
        f = max(cps, key=lambda p: os.path.getmtime(p))
        try:
            out.append((sh.name, json.load(open(f, encoding="utf-8"))))
        except Exception:
            continue
    return out


def completed_stats(stage_dir: Path, steps: int):
    """已完成 run 的兜底率统计（从 shard summary 读）。"""
    rates = []
    for sf in glob.glob(str(stage_dir / "shard*_of*" / "llm_robust_summary.csv")):
        try:
            import csv
            with open(sf, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if int(float(row.get("step", -1))) == steps and str(row.get("status", "")) in {
                            "completed", "completed_with_fallback"}:
                        r = row.get("llm_fallback_rate", "")
                        if r not in ("", None):
                            rates.append(float(r))
        except Exception:
            continue
    return rates


def check_stage(name: str, stage_dir: Path, shards: int, steps: int, n_agents: int, prev: dict):
    """返回 (状态dict, 更新后的prev条目)。"""
    cps = newest_checkpoint_per_shard(stage_dir)
    tot_fb = sum((d.get("llm_quality", {}) or {}).get("fallbacks", 0) for _, d in cps)
    tot_lc = sum((d.get("llm_quality", {}) or {}).get("logical_calls", 0) for _, d in cps)
    tot_pe = sum((d.get("llm_quality", {}) or {}).get("provider_errors", 0) for _, d in cps)
    rows = []
    for sh, d in cps:
        q = d.get("llm_quality", {}) or {}
        rows.append((sh, d.get("current_step", 0), q.get("fallbacks", 0), q.get("logical_calls", 0)))

    # 边际率（相对上次）
    marg = None
    if prev and tot_lc > prev.get("lc", 0):
        d_fb = tot_fb - prev.get("fb", 0)
        d_lc = tot_lc - prev.get("lc", 0)
        marg = d_fb / d_lc if d_lc > 0 else 0.0

    # 文件新鲜度（是否在推进）
    newest = 0.0
    for f in glob.glob(str(stage_dir / "**" / "*"), recursive=True):
        try:
            m = os.path.getmtime(f)
            if m > newest:
                newest = m
        except OSError:
            continue
    alive = (time.time() - newest) < 180 if newest else None

    rates = completed_stats(stage_dir, steps)
    n_done = len(rates)

    res = {
        "stage": name,
        "shards_alive": len(cps),
        "steps": [r[1] for r in rows],
        "tot_fb": tot_fb, "tot_lc": tot_lc,
        "cum_rate": (tot_fb / tot_lc) if tot_lc else 0.0,
        "marg_rate": marg,
        "provider_err": tot_pe,
        "runs_done": n_done,
        "run_rate_max": max(rates) if rates else None,
        "run_rate_mean": (sum(rates) / len(rates)) if rates else None,
        "runs_over_2pct": sum(1 for r in rates if r > 0.02),
        "alive": alive,
        "per_shard": [f"{r[0]}:{r[2]}/{r[3]}={r[2]/max(1,r[3]):.2%}" for r in rows],
    }
    newprev = {"fb": tot_fb, "lc": tot_lc, "ts": time.time()}
    return res, newprev


def level_of(res: dict) -> str:
    if res["tot_lc"] == 0:
        return "NA"
    if res["marg_rate"] is not None and res["marg_rate"] >= 0.05:
        return "CRIT"
    if res["cum_rate"] >= 0.04:
        return "CRIT"
    if (res["run_rate_max"] is not None and res["run_rate_max"] > 0.02) or \
       (res["marg_rate"] is not None and res["marg_rate"] > 0.02):
        return "ALERT"
    if res["cum_rate"] > 0.02:
        return "WATCH"
    return "OK"


def fmt(res: dict) -> str:
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    lvl = level_of(res)
    parts = [f"[{ts}] {res['stage']} {lvl}"]
    parts.append(f"shards={res['shards_alive']} alive={'Y' if res['alive'] else 'N'}")
    parts.append(f"cum={res['cum_rate']:.3%}({res['tot_fb']}/{res['tot_lc']})")
    parts.append(f"marg={'%.3f%%' % (res['marg_rate']*100) if res['marg_rate'] is not None else 'NA'}")
    parts.append(f"done={res['runs_done']}" + (f" max={res['run_rate_max']:.3%} over2={res['runs_over_2pct']}" if res['run_rate_max'] is not None else ""))
    if res["provider_err"]:
        parts.append(f"prov_err={res['provider_err']}")
    return " | ".join(parts)


def do_check(prev_all: dict, print_out: bool = False):
    state = {}
    lines = []
    alerts = []
    for name, d, shards, steps, n_agents in STAGES:
        if not d.exists():
            state[name] = prev_all.get(name, {})
            if print_out:
                lines.append(f"[{datetime.now():%H:%M:%S}] {name} NA (dir missing)")
            continue
        res, newprev = check_stage(name, d, shards, steps, n_agents, prev_all.get(name, {}))
        state[name] = newprev
        line = fmt(res)
        lines.append(line)
        if level_of(res) in ("ALERT", "CRIT"):
            alerts.append(line + "\n  per_shard: " + " ".join(res["per_shard"]))
    return state, lines, alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=240)
    ap.add_argument("--until", default="03:30", help="循环模式自动结束时刻（HH:MM）")
    args = ap.parse_args()

    prev_all = {}
    if STATE.exists():
        try:
            prev_all = json.load(open(STATE, encoding="utf-8"))
        except Exception:
            prev_all = {}

    def run_cycle(print_out):
        nonlocal prev_all
        state, lines, alerts = do_check(prev_all, print_out=print_out)
        prev_all = state
        try:
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass
        with open(WATCH_LOG, "a", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")
        if alerts:
            with open(ALERT_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"===== {datetime.now():%F %T} =====\n")
                for a in alerts:
                    fh.write(a + "\n")
        if print_out:
            for ln in lines:
                print(ln)

    if args.once:
        run_cycle(True)
        return

    # 循环模式
    stop_hm = args.until
    try:
        _sh, _sm = map(int, stop_hm.split(":"))
    except Exception:
        _sh, _sm = 3, 30
    stop_min = _sh * 60 + _sm
    while True:
        try:
            run_cycle(False)
        except Exception as exc:
            with open(WATCH_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"[{datetime.now():%F %T}] WATCHER-ERROR {exc}\n")
        _now = datetime.now()
        now_min = _now.hour * 60 + _now.minute
        # 跨零点安全：距停止时刻 0~12 小时内则退出（默认 03:30 停）
        if 0 <= (now_min - stop_min) < 720:
            with open(WATCH_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"[{datetime.now():%F %T}] watcher exits (until {stop_hm})\n")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
