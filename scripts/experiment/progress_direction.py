# -*- coding: utf-8 -*-
"""方向版重跑套件 · 实时监控台（进度 / 兜底率 / token / 真实成本 / 余额 / 停机预测）。

用法：
  python progress_direction.py            # 单次快照（推荐）
  python progress_direction.py -i 30      # 每 30 秒自动刷新

要点：
* 真实成本按官方价目表（DeepSeek 空闲 in ¥1.5/M + out ¥4.5/M；Qwen in ¥0.8/M + out ¥2.7/M），
  不用项目内 estimated_cost（该列系统性低估约 2 倍）。
* DeepSeek 余额实时查询（免费）；燃烧速率与触底时间从 _scheduler.log 余额记录推算。
* 兜底率取各批次 summary 的 llm_fallback_rate（canonical 口径）。
"""
from __future__ import annotations

import argparse
import glob
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

PROJ = Path(_PROJ_ROOT)
RD = PROJ / "results_direction"
PROBE = Path(_PROBE_ROOT)
ENV = PROJ / ".env"
SCHED_LOG = RD / "_scheduler.log"

STAGES = [
    ("主实验(方向版)", sorted(PROBE.glob("results_prompt_direction/a*")), 30, "ds"),
    ("Qwen 复现", [RD / "llm_qwen_full"], 30, "qw"),
    ("Qwen 消融", sorted(RD.glob("llm_qwen_abl_*")), 30, "qw"),
    ("随机feed", [RD / "llm_randomfeed_control"], 30, "ds"),
    ("DS消融×3", sorted(RD.glob("llm_ablation_*")), 30, "ds"),
    ("N100规模", [RD / "llm_n100_scale"], 30, "ds"),
    ("T100延长", [RD / "llm_extend_T100"], 100, "ds"),
]
EXPECTED = {"主实验(方向版)": 160, "Qwen 复现": 200, "Qwen 消融": 190,
            "随机feed": 20, "DS消融×3": 340, "N100规模": 30, "T100延长": 160}
REAL_RATE = {"ds": (1.5e-6, 4.5e-6), "qw": (0.8e-6, 2.7e-6)}
# 未完成的执行顺序 + 实测单价（¥/run，T30 或 T100 口径）
RUN_ORDER = [("DS消融×3", 0.192), ("N100规模", 0.64), ("T100延长", 0.64)]


def scan_dir(d: Path, steps: int):
    """(完成runs, 超2%数, 兜底mean, 兜底max, 输入tok, 输出tok)。"""
    done = over2 = 0
    fb_sum = 0.0
    fb_max = 0.0
    tok_in = tok_out = 0
    frames = []
    if (d / "llm_robust_summary.csv").exists():
        frames = [pd.read_csv(d / "llm_robust_summary.csv")]
    else:
        for fp in sorted(d.glob("shard*_of*/llm_robust_summary.csv")):
            try:
                frames.append(pd.read_csv(fp))
            except Exception:
                pass
    n = 0
    for df in frames:
        for _, r in df.iterrows():
            n += 1
            try:
                if int(float(r.get("step", -1))) == steps and \
                        str(r.get("status", "completed")).startswith("completed"):
                    done += 1
            except (ValueError, TypeError):
                pass
            fb = float(r.get("llm_fallback_rate") or 0)
            fb_sum += fb
            fb_max = max(fb_max, fb)
            if fb > 0.02:
                over2 += 1
            tok_in += int(float(r.get("input_tokens") or 0))
            tok_out += int(float(r.get("output_tokens") or 0))
    return done, over2, fb_sum / max(n, 1), fb_max, tok_in, tok_out


def live_calls() -> int:
    now = time.time()
    n = 0
    for p in glob.glob(str(RD / "*" / "shard*" / "calls" / "*.jsonl")):
        try:
            if now - Path(p).stat().st_mtime < 120:
                n += 1
        except OSError:
            pass
    return n


def ds_balance() -> float | None:
    try:
        import requests
        key = ""
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("LLM_API_KEY="):
                key = line.partition("=")[2].strip()
        r = requests.get("https://api.deepseek.com/user/balance",
                         headers={"Authorization": "Bearer " + key}, timeout=15)
        if r.status_code == 200:
            return float(r.json()["balance_infos"][0]["total_balance"])
    except Exception:
        pass
    return None


def burn_rate():
    """从 _scheduler.log 最近两条余额记录推 (¥/小时)；暂停期返回 None。"""
    try:
        lines = [l for l in SCHED_LOG.read_text(encoding="utf-8").splitlines()
                 if "余额检查" in l][-2:]
        if len(lines) < 2:
            return None
        t1 = datetime.strptime(lines[0][1:20], "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(lines[1][1:20], "%Y-%m-%d %H:%M:%S")
        v1 = float(lines[0].split("¥")[-1])
        v2 = float(lines[1].split("¥")[-1])
        hours = (t2 - t1).total_seconds() / 3600
        if hours <= 0 or v1 <= v2:
            return None
        return (v1 - v2) / hours
    except Exception:
        return None


def snapshot() -> None:
    g_done = g_exp = g_in = g_out = g_over = 0
    g_real = 0.0
    st = {}
    print("=" * 108)
    print(f"══ 方向版重跑 · 实时监控 · {datetime.now():%Y-%m-%d %H:%M:%S} ══")
    print("-" * 108)
    print(f"{'阶段':<15}{'完成':>9}{'兜底mean':>9}{'兜底max':>9}{'超2%':>5}"
          f"{'输入tok':>13}{'输出tok':>11}{'真实¥':>8}")
    for name, dirs, steps, prov in STAGES:
        done = over2 = tok_in = tok_out = 0
        m_avg, m_max = 0.0, 0.0
        for d in dirs:
            if not d.exists():
                continue
            dn, o2, avg, mx, ti, to = scan_dir(d, steps)
            done += dn
            over2 += o2
            m_avg = max(m_avg, avg)
            m_max = max(m_max, mx)
            tok_in += ti
            tok_out += to
        rate = REAL_RATE[prov]
        real = tok_in * rate[0] + tok_out * rate[1]
        exp = EXPECTED[name]
        st[name] = (done, exp)
        g_done += done
        g_exp += exp
        g_over += over2
        g_in += tok_in
        g_out += tok_out
        g_real += real
        mark = "100%" if done >= exp else f"{100 * done / exp:.0f}%"
        print(f"{name:<15}{done:>5}/{exp:<4}{mark:>5}"
              f"{m_avg:>9.2%}{m_max:>9.2%}{over2:>5}"
              f"{tok_in:>13,}{tok_out:>11,}{real:>8.2f}")
    print("-" * 108)
    print(f"{'合计':<15}{g_done:>5}/{g_exp:<3}{100 * g_done / g_exp:>3.0f}%"
          f"{'':>19}{g_over:>5}{g_in:>13,}{g_out:>11,}{g_real:>8.2f}")
    print(f"总 token = {g_in + g_out:,} ｜ 活跃调用文件(120s内) = {live_calls()}")

    current = next((n for n, _ in RUN_ORDER if st.get(n, (0, 0))[0] < st.get(n, (0, 0))[1]), None)
    remaining_cost = sum(max(st.get(n, (0, 0))[1] - st.get(n, (0, 0))[0], 0) * u
                         for n, u in RUN_ORDER)

    bal = ds_balance()
    burn = burn_rate()
    seg = []
    if bal is not None:
        seg.append(f"DeepSeek 余额 ¥{bal:.2f}")
    if burn:
        seg.append(f"燃烧 ¥{burn:.1f}/h")
        if bal is not None:
            mins = max((bal - 3.0) / (burn / 60), 0)
            seg.append(f"预计 {(datetime.now() + timedelta(minutes=mins)):%H:%M} 触底(¥3)")
    seg.append(f"剩余阶段需 ¥{remaining_cost:.0f}")
    if bal is not None:
        need = remaining_cost + 3
        seg.append("资金" + ("充足 ✓" if bal >= need else f"缺口 ¥{need - bal:.0f}"))
    print(" ｜ ".join(seg))
    print(f"当前阶段: {current or '（全部完成 ✓）'}")
    print("=" * 108)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=int, default=0)
    args = ap.parse_args()
    if args.interval > 0:
        while True:
            snapshot()
            time.sleep(args.interval)
    else:
        snapshot()


if __name__ == "__main__":
    main()
