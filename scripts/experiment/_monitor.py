# -*- coding: utf-8 -*-
"""实验一体化监控器（唯一监控入口）：进度 + 兜底率 + 红线执行。

用法:
  python -X utf8 _monitor.py            # 循环模式：每 3 分钟检查，03:30 自动停
  python -X utf8 _monitor.py --once     # 单次检查并打印（供自动化/人工调用）

监控内容（两批：qwen-t100 / n100b）:
  进度    完成run数/目标、各分片在途step、速度(steps/min)、预计完成时间
  兜底率  cum=累计、marg=边际(相对上次检查)、各分片、已完成run终值(max/超2%清单)

红线（用户硬约束：兜底率 <= 2%）:
  1) 已完成 run 终值 > 2%   -> 记入 _over2_runs.txt（重跑清单），不停机
  2) 边际率连续 2 次 > 2%，或单次 >= 5% -> 自动暂停：
     写 _pause.flag（守护循环见到即不再拉起 runner）+ 终止在跑 runner，日志记 PAUSE
     恢复方法：删除 _pause.flag，守护循环会自动续跑（checkpoint 无损）

日志: _monitor.log（全部状态都在这）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

PROBE = Path(__file__).resolve().parent
RD = Path(_PROJ_ROOT + r"\results_direction")
LOG = PROBE / "_monitor.log"
PAUSE_FLAG = PROBE / "_pause.flag"
OVER2 = PROBE / "_over2_runs.txt"
STATE = PROBE / "_monitor_state.json"
N_AGENTS = 30

STAGES = [
    # (stage名, 目录, 目标runs, 每run步数, 真实成本倍率, 成本上限¥, 账户基线)
    ("qwen-t100", RD / "llm_qwen_extend_T100", 200, 100, 1.0, 72.0,
     {"balance": 32.76, "spent_at_baseline": 46.98, "note": "DashScope，用户充值后实报 ¥32.76 @00:32"}),
    ("n100b",     RD / "llm_n100_boost",       60,  30, 1.554, 33.0,
     {"balance": 55.57, "spent_at_baseline": 0.0, "floor": 5.0,
      "note": "DeepSeek，用户充值后 API 实查 ¥55.57 @00:33"}),
]
_CKPT_CACHE = {}  # checkpoint 路径 -> [mtime, cost]（避免每轮全量重解析）


def stage_cost(stage_dir: Path) -> float:
    """本阶段累计花费（checkpoint 的 estimated_cost 求和，带缓存只解析有变化的文件）。"""
    tot = 0.0
    for f in glob.glob(str(stage_dir / "shard*_of*" / "checkpoints" / "*.checkpoint.json")):
        try:
            m = os.path.getmtime(f)
        except OSError:
            continue
        c = _CKPT_CACHE.get(f)
        if c and abs(c[0] - m) < 1e-6:
            tot += c[1]
            continue
        try:
            v = float(json.load(open(f, encoding="utf-8")).get("llm_quality", {}).get("estimated_cost", 0))
        except Exception:
            v = 0.0
        _CKPT_CACHE[f] = [m, v]
        tot += v
    return tot


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}\n")


def kill_stage(keyword: str) -> int:
    """终止命令行含 keyword 的 python 进程（runner 及其分片），返回杀掉的 PID 数。"""
    ps = ("$p = Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
          f"Where-Object {{ $_.CommandLine -like '*{keyword}*' }}; "
          "if ($p) { $p | ForEach-Object { [string]$_.ProcessId } }")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=90)
        pids = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip().isdigit()]
    except Exception:
        return 0
    n = 0
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
        n += 1
    return n


def scan_stage(name: str, sdir: Path, target: int, steps: int, prev: dict):
    """扫描一批：返回 (res, newprev)。"""
    # 已完成 run：runs/*_agents_final 计数 + summary 终值
    done = len(glob.glob(str(sdir / "shard*_of*" / "runs" / "*_agents_final*")))
    finals = {}   # run_id -> rate（合并根/shard summary，去重）
    for sf in glob.glob(str(sdir / "shard*_of*" / "llm_robust_summary.csv")) + \
              glob.glob(str(sdir / "llm_robust_summary.csv")):
        try:
            with open(sf, encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        if int(float(row.get("step", -1))) != steps:
                            continue
                        if str(row.get("status", "")) not in ("completed", "completed_with_fallback"):
                            continue
                        rid = int(float(row["run_id"]))
                        r = row.get("llm_fallback_rate", "")
                        if r not in ("", None):
                            finals[rid] = float(r)
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            continue

    # 在途：每分片最新 checkpoint
    inf_steps, tot_fb, tot_lc, per_shard = [], 0, 0, []
    for sh in sorted(sdir.glob("shard*_of*")):
        cps = glob.glob(str(sh / "checkpoints" / "*.checkpoint.json"))
        if not cps:
            continue
        f = max(cps, key=os.path.getmtime)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        q = d.get("llm_quality", {}) or {}
        st = d.get("current_step", 0)
        inf_steps.append(st)
        tot_fb += q.get("fallbacks", 0)
        tot_lc += q.get("logical_calls", 0)
        per_shard.append(f"{sh.name[5:7]}:{st}")

    newest = 0.0
    for f in glob.glob(str(sdir / "**" / "*"), recursive=True):
        try:
            m = os.path.getmtime(f)
            if m > newest:
                newest = m
        except OSError:
            continue
    alive = bool(newest) and (time.time() - newest) < 240

    marg = None
    if prev and tot_lc > prev.get("lc", 0):
        d_fb, d_lc = tot_fb - prev.get("fb", 0), tot_lc - prev.get("lc", 0)
        marg = (d_fb / d_lc) if d_lc > 0 else 0.0

    # 速度与 ETA（用 总步数 进度：done*steps + 在途step之和）
    prog_units = done * steps + sum(inf_steps)
    pace = None
    if prev and prev.get("prog") is not None and prog_units > prev["prog"] and prev.get("ts"):
        dt_min = (time.time() - prev["ts"]) / 60.0
        if dt_min > 0.2:
            pace = (prog_units - prev["prog"]) / dt_min   # steps/min
    remaining = target * steps - prog_units
    eta_min = (remaining / pace) if (pace and pace > 0 and remaining > 0) else None

    res = {
        "stage": name, "done": done, "target": target,
        "inf_steps": inf_steps, "per_shard": " ".join(per_shard),
        "tot_fb": tot_fb, "tot_lc": tot_lc,
        "cum": (tot_fb / tot_lc) if tot_lc else 0.0,
        "marg": marg, "alive": alive,
        "finals": finals, "pace": pace, "eta_min": eta_min,
        "prog": prog_units,
    }
    newprev = {"fb": tot_fb, "lc": tot_lc, "prog": prog_units, "ts": time.time()}
    return res, newprev


def fmt(r: dict) -> str:
    eta = f"{r['eta_min']:.0f}min" if r["eta_min"] else "NA"
    pace = f"{r['pace']:.1f}step/min" if r["pace"] else "NA"
    marg = f"{r['marg']:.2%}" if r["marg"] is not None else "NA"
    fin = ""
    if r["finals"]:
        mx = max(r["finals"].values())
        over = sum(1 for v in r["finals"].values() if v > 0.02)
        fin = f" 终值n={len(r['finals'])} max={mx:.2%} 超2%={over}"
    cost = ""
    if "spent_real" in r:
        rate = f"¥{r['rate_real']:.2f}/min" if r.get("rate_real") is not None else "NA"
        cost = (f" | 花费 ¥{r['spent_real']:.1f}(真实) 速率 {rate} "
                f"预测全程 ¥{r.get('proj_real', 0) or 0:.1f}/上限 ¥{r.get('ceiling', 0) or 0:.0f} "
                f"账户预估余额 ¥{r.get('balance_est', 0) or 0:.1f}")
    return (f"{r['stage']} 进度={r['done']}/{r['target']} 速度={pace} ETA={eta} "
            f"| 兜底 cum={r['cum']:.2%} marg={marg}{fin} "
            f"| 存活={'Y' if r['alive'] else 'N'} 在途[{r['per_shard']}]{cost}")


def check_pause(res: dict, marg_hist: list) -> tuple[bool, str]:
    """返回 (是否需要暂停, 原因)。"""
    if res["tot_lc"] == 0:
        return False, ""
    h = marg_hist[-2:]
    if len(h) == 2 and all(m is not None and m > 0.02 for m in h):
        return True, f"边际率连续2次>2% ({h[0]:.2%}, {h[1]:.2%})"
    if res["marg"] is not None and res["marg"] >= 0.05:
        return True, f"边际率单次>=5% ({res['marg']:.2%})"
    return False, ""


def run_cycle(prev_all: dict, marg_hist: dict, state_all: dict, once: bool):
    lines = []
    all_done = True
    for name, sdir, target, steps, real_mult, ceiling, acct in STAGES:
        if not sdir.exists():
            lines.append(f"{name} NA(目录不存在)")
            all_done = False
            continue
        res, newprev = scan_stage(name, sdir, target, steps, prev_all.get(name, {}))
        # ---- 成本监控（严格模式）----
        spent_code = stage_cost(sdir)
        spent_real = spent_code * real_mult
        rate_real = None
        if prev_all.get(name, {}).get("spent_real") is not None:
            dt_min = (time.time() - prev_all[name].get("ts", time.time())) / 60.0
            if dt_min > 0.2:
                rate_real = (spent_real - prev_all[name]["spent_real"]) / dt_min
        avg_real = None
        if res["finals"]:
            # 用已完成 run 的 summary 均价（排除在途部分成本摊薄导致的虚高）
            import csv as _csv
            fc = []
            for sf in glob.glob(str(sdir / "shard*_of*" / "llm_robust_summary.csv")) + \
                      glob.glob(str(sdir / "llm_robust_summary.csv")):
                try:
                    for row in _csv.DictReader(open(sf, encoding="utf-8")):
                        if int(float(row.get("step", -1))) == steps and \
                           str(row.get("status", "")) in ("completed", "completed_with_fallback"):
                            fc.append(float(row.get("estimated_cost", 0) or 0))
                except Exception:
                    continue
            if fc:
                avg_real = (sum(fc) / len(fc)) * real_mult
        if avg_real is None:
            avg_real = ({"qwen-t100": 0.336, "n100b": 0.48}.get(name, 0.4))
        proj_real = spent_real + (target - res["done"]) * avg_real
        bal_est = acct["balance"] - (spent_real - acct["spent_at_baseline"])
        floor = acct.get("floor", 3.0)
        res["spent_real"] = spent_real
        res["rate_real"] = rate_real
        res["proj_real"] = proj_real
        res["ceiling"] = ceiling
        res["balance_est"] = bal_est
        newprev["spent_real"] = spent_real
        # 成本红线：预测全程超上限 10%，或账户预估余额耗尽 → 自动暂停
        if not PAUSE_FLAG.exists() and res["done"] < target:
            if proj_real > ceiling * 1.10:
                PAUSE_FLAG.write_text(
                    f"{datetime.now():%F %T} {name} 成本红线: 预测全程 ¥{proj_real:.1f} > 上限 ¥{ceiling:.0f}×1.1\n",
                    encoding="utf-8")
                kill_stage(f"--stage {name}")
                log(f"!!!! COST-PAUSE: {name} 预测 ¥{proj_real:.1f} 超上限；已暂停，待人工确认")
            elif bal_est <= floor:
                PAUSE_FLAG.write_text(
                    f"{datetime.now():%F %T} {name} 账户预估余额 ¥{bal_est:.2f} ≤{floor}，继续将耗尽\n",
                    encoding="utf-8")
                kill_stage(f"--stage {name}")
                log(f"!!!! BALANCE-PAUSE: {name} 预估余额 ¥{bal_est:.2f} ≤{floor}；已暂停，充值后删 _pause.flag 恢复")
        prev_all[name] = newprev
        lines.append(fmt(res))
        if res["done"] < target:
            all_done = False

        # 红线1：已完成 run 终值 > 2% -> 重跑清单
        seen = state_all.setdefault("over2_seen", {})
        for rid, v in sorted(res["finals"].items()):
            key = f"{name}:{rid}"
            if v > 0.02 and not seen.get(key):
                seen[key] = v
                with open(OVER2, "a", encoding="utf-8") as fh:
                    fh.write(f"{datetime.now():%F %T} {name} run_{rid} fallback={v:.4%}\n")
                log(f"WARNING 超2%终值: {name} run_{rid} = {v:.4%}（已记入 {OVER2.name}，择时重跑）")

        # 红线2：边际率持续超 -> 自动暂停
        hist = marg_hist.setdefault(name, [])
        hist.append(res["marg"])
        del hist[:-3]
        if not PAUSE_FLAG.exists():
            need, reason = check_pause(res, hist)
            if need:
                PAUSE_FLAG.write_text(
                    f"{datetime.now():%F %T} {name} {reason}\n", encoding="utf-8")
                killed = kill_stage(f"--stage {name}")
                log(f"!!!! PAUSE 触发: {name} {reason}；已终止 {killed} 个进程；"
                    f"删除 _pause.flag 后守护会自动续跑")

    paused = PAUSE_FLAG.exists()
    if paused and not once:
        lines.append(f"== PAUSED：{PAUSE_FLAG.read_text(encoding='utf-8').strip()} —— 删除 _pause.flag 恢复 ==")
    if all_done and not paused:
        lines.append("== 全部完成 ==")
    for ln in lines:
        log(ln)
    if once:
        print("\n".join(lines))
    return all_done, paused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--until", default="03:30")
    args = ap.parse_args()

    prev_all, marg_hist, state_all = {}, {}, {}
    if STATE.exists():
        try:
            d = json.load(open(STATE, encoding="utf-8"))
            prev_all = d.get("prev", {})
            state_all = d.get("over2_seen", {}) and {"over2_seen": d["over2_seen"]} or {}
            state_all = {"over2_seen": d.get("over2_seen", {})}
        except Exception:
            pass

    log("=== monitor start ===")
    try:
        _sh, _sm = map(int, args.until.split(":"))
    except Exception:
        _sh, _sm = 3, 30
    stop_min = _sh * 60 + _sm

    while True:
        try:
            all_done, paused = run_cycle(prev_all, marg_hist, state_all, args.once)
        except Exception as exc:
            log(f"MONITOR-ERROR {exc}")
            all_done, paused = False, False
        try:
            STATE.write_text(json.dumps(
                {"prev": prev_all, "over2_seen": state_all.get("over2_seen", {})},
                ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        if args.once or all_done:
            break
        now_min = datetime.now().hour * 60 + datetime.now().minute
        if 0 <= (now_min - stop_min) < 720:
            log("monitor exits (until)")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
