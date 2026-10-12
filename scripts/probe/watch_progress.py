# -*- coding: utf-8 -*-
"""终端实时进度监视器（提示词明确性探针 · 完整网格批次）。

打印内容：
  * 总体进度条 + 百分比（已完成 run，含"进行中"的小数折算）
  * 已用时间 / 预计剩余 / 已花成本 / 预计总成本
  * 正在跑的 run_id 与当前步
  * **实时结果面板**：每完成一批就把当前累计的 α 斜率、TOST、上行幅度算出来，
    并与存档「原版提示词」的同口径结果并排显示 —— 跑的过程中就能看到结论在不在变。

用法：
  python watch_progress.py                    # 单次快照
  python watch_progress.py --loop             # 实时刷新（默认每 10 秒，Ctrl-C 退出）
  python watch_progress.py --loop -i 3        # 每 3 秒刷新
  python watch_progress.py --ascii            # 终端不支持方块字符时用 #/- 画条
  python watch_progress.py --root D:\\...\\results_prompt_labeled --total 20

数据来源：<root>/llm_robust_summary.csv（增量写入）、<root>/calls/*.jsonl（判断在跑的 run）、
<root>/runs/*.csv（逐 run 轨迹，用于算上行幅度）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

# ---- 编码策略（沿用项目内 watch_progress.py 的做法）----
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
else:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
# 若终端仍显示乱码，先执行：chcp 65001

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "results_prompt_full"
ARCHIVE = Path(_PROJ_ROOT + r"\results\llm_gradient_noleak")
DELTA = 0.031                      # 等价界（原始尺度），与论文一致
BAR_FULL, BAR_EMPTY = "█", "░"
ASCII_FULL, ASCII_EMPTY = "#", "-"


def ols(y, x):
    n = len(y)
    if n < 3:
        return None
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return None
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    dof = n - 2
    se = np.sqrt((resid ** 2).sum() / dof / sxx)
    t = b / se
    p = 2 * (1 - stats.t.cdf(abs(t), dof))
    r2 = 1 - (resid ** 2).sum() / ((y - ym) ** 2).sum()
    bz = b * x.std(ddof=0) / y.std(ddof=0) if y.std(ddof=0) > 0 else np.nan
    return dict(b=b, se=se, p=p, r2=r2, bz=bz, n=n)


def tost_ok(b, se, n, delta=DELTA):
    """双侧等价检验是否通过（δ 在原始尺度）。"""
    if n < 3:
        return None, None
    dof = n - 2
    p1 = 1 - stats.t.cdf((b + delta) / se, dof)
    p2 = stats.t.cdf((b - delta) / se, dof)
    p = max(p1, p2)
    return p, p < 0.05


def sub_roots(root: Path):
    """支持两种布局：单目录，或「三档 α 各一个子目录」的判定实验布局。"""
    if (root / "llm_robust_summary.csv").exists():
        return [root]
    if not root.exists():
        return []
    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and (d / "llm_robust_summary.csv").exists()]
    return subs


def is_multi(root: Path) -> bool:
    """多区块布局：根目录自己没有 summary，但有带 summary 的子目录。"""
    return (not (root / "llm_robust_summary.csv").exists()) and len(sub_roots(root)) >= 1


def read_summary(root: Path) -> pd.DataFrame:
    frames = []
    for r in sub_roots(root):
        p = r / "llm_robust_summary.csv"
        try:
            d = pd.read_csv(p)
            if "step" in d:
                d = d[d["step"].notna()]
            if len(d):
                d = d.copy()
                d["_sub"] = r.name
                frames.append(d)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def live_run(root: Path, steps: int):
    """返回 (run_id, 当前步, 最新 calls 文件 mtime, 所属子目录)；无则在跑的 None。"""
    best = None
    for r in sub_roots(root):
        cdir = r / "calls"
        if not cdir.exists():
            continue
        for p in cdir.glob("*.jsonl"):
            m = p.stat().st_mtime
            if best is None or m > best[0]:
                best = (m, p, r)
    if best is None:
        return None, None, None, None
    mtime, p, r = best
    rid = None
    try:
        rid = int(p.stem.rsplit("_", 1)[-1])
    except Exception:
        pass
    step = None
    try:
        import json
        tail = p.read_bytes()[-4000:].decode("utf-8", errors="replace")
        for line in reversed([x for x in tail.splitlines() if x.strip()]):
            try:
                step = int(json.loads(line).get("step"))
                break
            except Exception:
                continue
    except Exception:
        pass
    return rid, step, mtime, r.name


def first_mtime(root: Path):
    mt = []
    for r in sub_roots(root):
        cdir = r / "calls"
        if cdir.exists():
            mt += [p.stat().st_mtime for p in cdir.glob("*.jsonl")]
    return min(mt) if mt else None


def rise_table(root: Path, steps: int) -> dict:
    """逐 γ 的上行：ΔPI、PI(1)→PI(30)、上升 run 数。"""
    out = {}
    acc = {}
    for r in sub_roots(root):
        rdir = r / "runs"
        if not rdir.exists():
            continue
        for f in sorted(rdir.glob("llm_robust_run_*.csv")):
            if f.name.endswith("_agents_final.csv"):
                continue
            try:
                t = pd.read_csv(f).sort_values("step")
                if len(t) < 2:
                    continue
                g = float(t["gamma"].iloc[0])
                acc.setdefault(g, []).append((float(t["PI"].iloc[0]), float(t["PI"].iloc[-1])))
            except Exception:
                continue
    for g, v in acc.items():
        if len(v) >= 2:
            p1 = np.array([x[0] for x in v])
            p30 = np.array([x[1] for x in v])
            out[g] = dict(pi1=float(p1.mean()), pi30=float(p30.mean()),
                          mean=float((p30 - p1).mean()), sd=float((p30 - p1).std(ddof=1)),
                          n_up=int((p30 - p1 > 0).sum()), n=len(v))
    return out


def panel(df: pd.DataFrame, rises: dict, title: str, total_per_g: int = 80) -> None:
    print(f"  ── 实时结果｜{title}　（已完成 {len(df)} runs）" + " " + "─" * 18)
    if len(df) == 0:
        print("     （尚无已完成的 run）")
        return
    for g in sorted(df["gamma"].unique()):
        d = df[np.isclose(df["gamma"], g)]
        n_a = int(d["alpha"].nunique())
        r = ols(d["PI"].to_numpy(float), d["alpha"].to_numpy(float)) if n_a >= 2 else None
        if r is None:
            print(f"    γ={g:<4} n={len(d):>3}/{total_per_g}　"
                  f"α 只覆盖 {n_a} 档，尚不能估斜率")
        else:
            pt, ok = tost_ok(r["b"], r["se"], r["n"])
            print(f"    γ={g:<4} n={len(d):>3}/{total_per_g}　α 覆盖 {n_a}/8　"
                  f"β(原始)={r['b']:+.5f}　β_std={r['bz']:+.4f}　p={r['p']:.3f}")
            if pt is not None:
                verdict = ("通过（可判定等价于零）" if ok else "未通过") if n_a >= 5 else "（α 档数不足，暂不判）"
                print(f"             TOST(δ={DELTA}) p={pt:.4f} → {verdict}"
                      f"　可排除上界 |β|+1.96SE={abs(r['b'])+1.96*r['se']:.5f}")
        if g in rises:
            v = rises[g]
            print(f"             上行：PI(1)={v['pi1']:.4f} → PI(30)={v['pi30']:.4f}　"
                  f"ΔPI={v['mean']:+.4f} (sd {v['sd']:.4f})　上升 {v['n_up']}/{v['n']}")


def fmt_hms(sec: float) -> str:
    sec = max(0.0, sec)
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def bar(frac: float, width: int, ascii_mode: bool) -> str:
    f, e = (ASCII_FULL, ASCII_EMPTY) if ascii_mode else (BAR_FULL, BAR_EMPTY)
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return f * n + e * (width - n)


def snapshot(root: Path, total: int, steps: int, ascii_mode: bool, show_ref: bool) -> None:
    df = read_summary(root)
    done = len(df)
    rid, step, mtime, sub = live_run(root, steps)
    frac_live = min(1.0, max(0.0, (step or 0) / float(steps)))
    done_frac = done + (frac_live if done < total else 0.0)
    t0 = first_mtime(root)
    now = time.time()
    elapsed = (now - t0) if t0 else 0.0
    rate = (done_frac / elapsed) if elapsed > 0 and done_frac > 0 else 0.0
    eta = ((total - done_frac) / rate) if rate > 0 else float("inf")
    cost = float(df["estimated_cost"].sum()) if "estimated_cost" in df else 0.0
    proj = (cost / done_frac * total) if done_frac > 0 else 0.0

    print("=" * 78)
    layout = "多区块（判定实验）" if is_multi(root) else "单批次"
    print(f"提示词明确性探针　[{root.name}]　布局：{layout}")
    print("=" * 78)
    print(f"  进度  {bar(done_frac / total, 40, ascii_mode)}  "
          f"{done_frac:5.1f}/{total}  ({done_frac/total*100:5.1f}%)")
    print(f"  已完成 {done} runs" + (f"　进行中 {sub}/run_id={rid}，当前步 {step}/{steps}"
                                      if done < total and rid is not None else ""))
    if t0:
        print(f"  已用 {fmt_hms(elapsed)}　速率 {rate*60:.2f} run/min"
              f"　预计剩余 {fmt_hms(eta) if np.isfinite(eta) else '—'}"
              f"　预计完成 {time.strftime('%H:%M', time.localtime(now + eta)) if np.isfinite(eta) else '—'}")
    print(f"  成本 已花 ¥{cost:.2f}　预计总计 ¥{proj:.2f}")
    if mtime and done < total and (now - mtime) > 300:
        print(f"  ⚠ 最近一次调用写盘已过 {fmt_hms(now-mtime)}，进程可能已停止")

    if done:
        rises = rise_table(root, steps)
        panel(df, rises, "本批次（标签版提示词）", total_per_g=max(1, total // 2))

    if show_ref and ARCHIVE.exists():
        adf = read_summary(ARCHIVE)
        if len(adf):
            # 若本批次只覆盖部分 α 档，则把存档对照也裁到同样的档位，避免口径不一致
            alphas = set(np.round(df["alpha"].unique(), 6)) if done else set()
            if done and len(alphas) < adf["alpha"].nunique():
                adf = adf[adf["alpha"].round(6).isin(alphas)]
                ref_rises = rise_table_filtered(ARCHIVE, steps, alphas)
                panel(adf, ref_rises, "对照：存档（原版提示词，同 α 档）",
                      total_per_g=max(1, len(adf) // max(1, len(alphas))))
            else:
                panel(adf, rise_table(ARCHIVE, steps), "对照：存档（原版提示词）",
                      total_per_g=max(1, len(adf) // 2))
    print("=" * 78)


def rise_table_filtered(root: Path, steps: int, alphas) -> dict:
    """存档对照：只保留指定 α 档的上行统计。"""
    out = {}
    acc = {}
    rdir = root / "runs"
    if not rdir.exists():
        return out
    for f in sorted(rdir.glob("llm_robust_run_*.csv")):
        if f.name.endswith("_agents_final.csv"):
            continue
        try:
            t = pd.read_csv(f).sort_values("step")
            if len(t) < 2 or round(float(t["alpha"].iloc[0]), 6) not in alphas:
                continue
            g = float(t["gamma"].iloc[0])
            acc.setdefault(g, []).append((float(t["PI"].iloc[0]), float(t["PI"].iloc[-1])))
        except Exception:
            continue
    for g, v in acc.items():
        if len(v) >= 2:
            p1 = np.array([x[0] for x in v])
            p30 = np.array([x[1] for x in v])
            out[g] = dict(pi1=float(p1.mean()), pi30=float(p30.mean()),
                          mean=float((p30 - p1).mean()), sd=float((p30 - p1).std(ddof=1)),
                          n_up=int((p30 - p1 > 0).sum()), n=len(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="提示词明确性探针 · 进度监视器")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--total", type=int, default=None,
                    help="默认按布局自动判断（单批次 160 / 多区块 60）")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--loop", action="store_true", help="持续刷新，Ctrl-C 退出")
    ap.add_argument("-i", "--interval", type=float, default=10.0, help="刷新间隔秒（默认 10）")
    ap.add_argument("--ascii", action="store_true", help="用 #/- 画进度条")
    ap.add_argument("--no-ref", action="store_true", help="不显示存档对照面板")
    args = ap.parse_args()

    if args.total is None:
        args.total = 60 if is_multi(args.root) else 160

    if not args.loop:
        snapshot(args.root, args.total, args.steps, args.ascii, not args.no_ref)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")      # 清屏 + 光标归位
            snapshot(args.root, args.total, args.steps, args.ascii, not args.no_ref)
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n已退出监视（实验仍在后台继续）。")


if __name__ == "__main__":
    main()
