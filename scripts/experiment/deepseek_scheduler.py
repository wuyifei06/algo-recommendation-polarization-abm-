# -*- coding: utf-8 -*-
"""DeepSeek 方向版全套重跑 · 定时 + 低谷窗口调度器。

设计
----
* 2026-10-10 12:00（用户指定）后才允许开跑；由 Windows 计划任务一次性拉起。
* 只在 DeepSeek 空闲时段发起调用：高峰 = 北京时间 9:00–12:00、14:00–18:00
  （api-docs.deepseek.com 现行资费页；空闲时段价格为高峰一半）。
  保守起见高峰窗口**每天**生效（部分资费口径称周末全时段空闲，可用
  --weekend-full 放开周末，省 4 小时空等）。
* 阶段顺序（贵/久在后，断了不影响前面；全部断点续跑）：
    randomfeed(20) -> abl no_post(160) -> abl no_emotion(160)
    -> abl positive_control(160) -> r20(160) -> n100(30) -> t100(160)
* 若某阶段跑到一半撞上高峰窗口：taskkill 终止进程树（run 级 checkpoint 已落盘，
  无损失），等窗口结束自动续跑同一阶段。
* 全部阶段完成后写 _scheduler_done.flag 并退出。

用法
----
  python deepseek_scheduler.py --dry-run        # 打印计划与当前判定，不执行
  python deepseek_scheduler.py --weekend-full   # 周末不放高峰假
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
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
RUNNER = HERE / "run_stage_direction.py"
OUT_ROOT = Path(_PROJ_ROOT + r"\results_direction")
PROJ = OUT_ROOT.parent          # 最终代码和数据
LOG = OUT_ROOT / "_scheduler.log"
DONE_FLAG = OUT_ROOT / "_scheduler_done.flag"
LOCK = OUT_ROOT / "_scheduler.lock"
ENV_PATH = PROJ / ".env"
PYTHON = _PYTHON
BALANCE_FLOOR = 3.0          # 余额低于此值 → 停止调度（保留干净断点，防止欠费毒化 run）
BALANCE_CHECK_EVERY = 4      # 每 4 个 poll（≈2 分钟）查一次余额

WAIT_UNTIL = dt.datetime(2026, 10, 9, 12, 0)          # 用户指定：今天中午 12 点装填
PEAK_WINDOWS = ((9 * 60, 12 * 60), (14 * 60, 18 * 60))  # 北京时间，分钟数
POLL_SEC = 30

STAGES = [
    ("randomfeed", None),
    ("abl", "no_post"),
    ("abl", "no_emotion"),
    ("abl", "positive_control"),
    ("n100", None),
    ("t100", None),
]
ARM_DIRS = {
    ("randomfeed", None): "llm_randomfeed_control",
    ("abl", "no_post"): "llm_ablation_no_post",
    ("abl", "no_emotion"): "llm_ablation_no_emotion",
    ("abl", "positive_control"): "llm_ablation_positive_control",
    ("n100", None): "llm_n100_scale",
    ("t100", None): "llm_extend_T100",
}

# 2026-10-09 削减决定：R20 加强档移除（其"零效应加固"功能因 α 效应被检出而失效）；
# 阳性对照缩为探针档 20 runs（见 run_stage_direction.specs_abl）。T100 保持全网格。


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def is_peak(now: dt.datetime, weekend_full: bool) -> bool:
    if weekend_full and now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return any(lo <= m < hi for lo, hi in PEAK_WINDOWS)


def minutes_to_window_end(now: dt.datetime) -> int:
    m = now.hour * 60 + now.minute + 1
    for lo, hi in PEAK_WINDOWS:
        if lo <= m < hi:
            return hi - m + 1
    return 1


def runner_status() -> dict:
    out = subprocess.run([PYTHON, "-X", "utf8", str(RUNNER), "--stage", "abl-all",
                          "--status"], capture_output=True, text=True, timeout=120)
    return json.loads(out.stdout.strip().splitlines()[-1])


def deepseek_balance() -> float | None:
    """查 DeepSeek 真实余额（免费接口）；失败返回 None（视为本次不判定）。"""
    try:
        import requests
        headers = {}
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("LLM_API_KEY="):
                headers["Authorization"] = "Bearer " + line.partition("=")[2].strip()
        if "Authorization" not in headers:
            return None
        r = requests.get("https://api.deepseek.com/user/balance",
                         headers=headers, timeout=15)
        if r.status_code == 200:
            return float(r.json()["balance_infos"][0]["total_balance"])
    except Exception as exc:                                   # noqa: BLE001
        log(f"余额查询失败（忽略本次）：{type(exc).__name__}: {exc}")
    return None


def net_alive() -> bool:
    """直连连通性探测（绕过系统代理，与 runner 清代理后的网络路径一致）；零成本。"""
    try:
        import requests
        key = ""
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LLM_API_KEY="):
                key = line.partition("=")[2].strip()
        s = requests.Session()
        s.trust_env = False
        r = s.get("https://api.deepseek.com/user/balance",
                  headers={"Authorization": "Bearer " + key}, timeout=8)
        return r.status_code == 200
    except Exception:
        return False


def scan_provider_errors(stage: str, arm) -> list:
    """扫描该阶段目录近 60 分钟写入的调用日志是否含 provider 错误（网络污染标记）。"""
    import glob
    d = OUT_ROOT / ARM_DIRS[(stage, arm)]
    cutoff = time.time() - 3600
    bad = set()
    for fp in glob.glob(str(d / "shard*_of*" / "calls" / "*.jsonl")):
        p = Path(fp)
        try:
            if p.stat().st_mtime < cutoff:
                continue
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in txt.splitlines():
            if "fallback_reason" in line and "Error" in line:
                bad.add(int(p.stem.replace("llm_robust_run_", "")))
                break
    return sorted(bad)


def stage_done(stage: str, arm, status: dict) -> bool:
    """按（stage, arm）精确判断：arm 指定时只看该臂对应目录，避免整段聚合导致空转。"""
    v = status[stage][ARM_DIRS[(stage, arm)]]
    return v["done"] >= v["expected"]


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True)
    return str(pid) in (out.stdout or "")


def acquire_lock() -> bool:
    """单实例守卫：已有存活实例则返回 False（避免定时任务与手动启动双开）。"""
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            pid = 0
        if pid and _pid_alive(pid):
            return False
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--weekend-full", action="store_true",
                    help="周末不执行高峰暂停（按周末全空闲口径）")
    args = ap.parse_args()

    if DONE_FLAG.exists():
        log("全部阶段已完成（检测到完成标志），退出。")
        return

    if not acquire_lock():
        log("检测到另一个调度器实例正在运行（锁文件），本实例退出。")
        return

    status = runner_status()
    log(f"启动：待跑阶段进度 {json.dumps(status, ensure_ascii=False)}")
    if args.dry_run:
        now = dt.datetime.now()
        log(f"dry-run：now={now:%F %T} weekday={now.weekday()} "
            f"is_peak={is_peak(now, args.weekend_full)} wait_until={WAIT_UNTIL}")
        for st, arm in STAGES:
            done = stage_done(st, arm, status)
            log(f"  {'[完成]' if done else '[待跑]'} stage={st} arm={arm}")
        return

    while True:
        now = dt.datetime.now()
        if now < WAIT_UNTIL:
            remain = (WAIT_UNTIL - now).total_seconds()
            if remain > 1800:
                log(f"未到装填时间 {WAIT_UNTIL}，还差 {remain / 3600:.1f} h，休眠。")
                time.sleep(max(60, remain - 1800))
            else:
                time.sleep(60)
            continue

        status = runner_status()
        pending = [(st, arm) for st, arm in STAGES if not stage_done(st, arm, status)]
        if not pending:
            DONE_FLAG.write_text(f"all done at {dt.datetime.now():%F %T}\n", encoding="utf-8")
            if LOCK.exists():
                LOCK.unlink()
            log("全部 7 个阶段完成，写完成标志，退出。")
            return

        if is_peak(now, args.weekend_full):
            wait_min = minutes_to_window_end(now) + 1
            log(f"高峰时段（9:00-12:00 / 14:00-18:00），暂停 {wait_min} 分钟后继续。")
            time.sleep(wait_min * 60)
            continue

        # ---- 网络门禁：连续 3 次探测成功（间隔 20s）才放行启动 ----
        streak = 0
        while streak < 3:
            if net_alive():
                streak += 1
            else:
                streak = 0
                log("网络探测失败（断网中），等待恢复……")
            if streak < 3:
                time.sleep(20)
        log("网络连通性确认（3 连成功），放行。")

        st, arm = pending[0]
        cmd = [PYTHON, "-X", "utf8", str(RUNNER), "--stage", st, "--workers", "8"]
        if arm:
            cmd += ["--arm", arm]
        log(f"启动阶段 stage={st} arm={arm} -> {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=str(HERE), stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        polls = 0
        net_fails = 0
        while proc.poll() is None:
            time.sleep(POLL_SEC)
            polls += 1
            if not net_alive():
                net_fails += 1
                log(f"⚠ 网络探测失败（连续 {net_fails} 次）")
                if net_fails >= 2:
                    log("连续断网 → 冻结当前阶段进程树（checkpoint 已落盘），"
                        "等待网络恢复后自动续跑。")
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                    break
            else:
                if net_fails:
                    log("网络已恢复。")
                net_fails = 0
            if polls % BALANCE_CHECK_EVERY == 0:
                bal = deepseek_balance()
                if bal is not None:
                    log(f"余额检查：¥{bal:.2f}")
                    if bal < BALANCE_FLOOR:
                        log(f"余额 ¥{bal:.2f} < 下限 ¥{BALANCE_FLOOR:.2f}，"
                            f"停止调度并终止当前阶段（断点干净，充值后重启即续跑）。")
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                       capture_output=True)
                        if LOCK.exists():
                            LOCK.unlink()
                        log("调度器退出。明晓充值后重新运行本脚本即可从断点续跑。")
                        return
            if is_peak(dt.datetime.now(), args.weekend_full):
                log("撞上高峰窗口，终止本阶段进程树（checkpoint 已落盘，稍后自动续跑）。")
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
                break
        code = proc.poll()
        if code == 0:
            log(f"阶段 stage={st} arm={arm} 正常退出。")
            bad = scan_provider_errors(st, arm)
            if bad:
                log(f"⚠⚠ 阶段 {st}/{arm} 检出 {len(bad)} 个 run 含 provider 错误"
                    f"（网络波动污染）：{bad}")
                log("    → 需按封存流程清理并重跑这些 run，勿直接用于分析！")
        else:
            log(f"阶段 stage={st} arm={arm} 退出码 {code}（可能被高峰/断网暂停或出错），"
                f"稍后重查进度。")
            bad = scan_provider_errors(st, arm)
            if bad:
                log(f"⚠⚠ 阶段 {st}/{arm} 检出 {len(bad)} 个 run 含 provider 错误"
                    f"（网络波动污染）：{bad}")
                log("    → 需按封存流程清理并重跑这些 run，勿直接用于分析！")
            bal = deepseek_balance()
            if bal is not None and bal < BALANCE_FLOOR:
                if LOCK.exists():
                    LOCK.unlink()
                log(f"余额 ¥{bal:.2f} 已耗尽，调度器退出；充值后重新运行即续跑。")
                return
            time.sleep(120)


if __name__ == "__main__":
    main()
