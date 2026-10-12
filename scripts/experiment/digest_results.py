# -*- coding: utf-8 -*-
"""结果数据汇总（纯数据，无解释）。输出 markdown 到 results_direction/_结果数据.md。"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

RD = Path(_PROJ_ROOT + r"\results_direction")
PROBE = Path(_PROBE_ROOT)
OUT = RD / "_结果数据_20261009.md"

METRICS = ["PI", "HE", "EE", "EM", "SR", "ER"]
lines: list[str] = []


def w(s: str = "") -> None:
    lines.append(s)
    print(s)


def load_summary(base: Path) -> pd.DataFrame:
    m = base / "llm_robust_summary.csv"
    if m.exists():
        return pd.read_csv(m)
    frames = []
    for fp in sorted(base.glob("shard*_of*/llm_robust_summary.csv")):
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def delta_pi(base: Path) -> dict:
    """seed -> ΔPI（末步 PI − 首步 PI）。"""
    out = {}
    pats = [str(base / "shard*_of*" / "runs" / "llm_robust_run_*.csv"),
            str(base / "runs" / "llm_robust_run_*.csv")]
    files: list[str] = []
    for p in pats:
        files += [f for f in glob.glob(p) if "_agents_final" not in f]
    for fp in files:
        m = re.search(r"llm_robust_run_(\d+)\.csv$", fp)
        if not m:
            continue
        rid = int(m.group(1))
        try:
            df = pd.read_csv(fp).sort_values("step")
            if len(df) >= 2:
                out[rid] = float(df["PI"].iloc[-1] - df["PI"].iloc[0])
        except Exception:
            continue
    return out


def seed_of(base: Path) -> dict:
    df = load_summary(base)
    if not len(df):
        return {}
    return {int(r["run_id"]): int(r["seed"]) for _, r in df.iterrows()}


def cell_table(df: pd.DataFrame, title: str) -> None:
    w(f"\n### {title}")
    w("| γ | α | n | PI | HE | EE | EM | SR | ER |")
    w("|---|---|---|---|---|---|---|---|---|")
    for g in sorted(df["gamma"].unique()):
        for a in sorted(df["alpha"].unique()):
            sub = df[(df["gamma"] == g) & (df["alpha"] == a)]
            if not len(sub):
                continue
            vals = " | ".join(f"{sub[c].mean():.4f}" for c in METRICS)
            w(f"| {g} | {a} | {len(sub)} | {vals} |")


def paired(arm_dir: Path, base_dir: Path, label: str, by_seed=None) -> None:
    arm_df = load_summary(arm_dir)
    base_df = load_summary(base_dir)
    if by_seed is None:
        by_seed = seed_of
    bmap = {int(r["seed"]): r for _, r in base_df.iterrows()}
    d_arm = delta_pi(arm_dir)
    d_base = delta_pi(base_dir)
    # run_id(arm) -> seed -> base run_id
    pairs = []
    rid2seed_arm = {int(r["run_id"]): int(r["seed"]) for _, r in arm_df.iterrows()}
    seed2rid_base = {int(r["seed"]): int(r["run_id"]) for _, r in base_df.iterrows()}
    for rid_a, seed in rid2seed_arm.items():
        rid_b = seed2rid_base.get(seed)
        if rid_b is None:
            continue
        if rid_a in d_arm and rid_b in d_base:
            pairs.append((d_arm[rid_a], d_base[rid_b]))
    if not pairs:
        w(f"\n### {label}\n（无法配对计算）")
        return
    arm = [p[0] for p in pairs]
    base = [p[1] for p in pairs]
    diff = [a - b for a, b in pairs]
    t, p = stats.ttest_rel(arm, base)
    w(f"\n### {label}")
    w(f"配对 n = {len(pairs)}")
    w(f"- 基线 ΔPI 均值 = {sum(base)/len(base):.4f}")
    w(f"- 本臂 ΔPI 均值 = {sum(arm)/len(arm):.4f}")
    w(f"- 配对差均值 = {sum(diff)/len(diff):.4f}（SD {pd.Series(diff).std():.4f}）")
    w(f"- t = {t:.3f}，p = {p:.3g}")


def main() -> None:
    # 1. 方向版主实验
    blocks = sorted(PROBE.glob("results_prompt_direction/a*"))
    main_df = pd.concat([load_summary(b) for b in blocks], ignore_index=True)
    w("# 结果数据（2026-10-09）\n")
    w("## 1. 方向版主实验（DeepSeek，160 runs）")
    cell_table(main_df, "逐格均值")

    # 2. 随机 feed
    rf = load_summary(RD / "llm_randomfeed_control")
    if len(rf):
        w("\n## 2. 随机 feed 对照（20 runs）")
        vals = " | ".join(f"{rf[c].mean():.4f}" for c in METRICS)
        w(f"| n | PI | HE | EE | EM | SR | ER |")
        w("|---|---|---|---|---|---|---|")
        w(f"| {len(rf)} | {vals} |")

    # 3. DeepSeek 消融（已完成的臂；主实验基线用多块拼接配对）
    def paired_multi(arm_dir: Path, base_dirs: list[Path], label: str) -> None:
        def seed_delta(base: Path) -> dict:
            df = load_summary(base)
            if not len(df):
                return {}
            rid2seed = {int(r["run_id"]): int(r["seed"]) for _, r in df.iterrows()}
            out = {}
            for rid, v in delta_pi(base).items():
                if rid in rid2seed:
                    out[rid2seed[rid]] = v
            return out

        base_delta = {}
        for bd in base_dirs:
            base_delta.update(seed_delta(bd))
        arm_delta = seed_delta(arm_dir)
        seeds = sorted(set(arm_delta) & set(base_delta))
        if not seeds:
            w(f"\n### {label}\n（无法配对计算）")
            return
        arm = [arm_delta[s] for s in seeds]
        base = [base_delta[s] for s in seeds]
        diff = [a - b for a, b in zip(arm, base)]
        t, p = stats.ttest_rel(arm, base)
        w(f"\n## {label}")
        w(f"配对 n = {len(seeds)}")
        w(f"- 基线 ΔPI 均值 = {sum(base)/len(base):.4f}")
        w(f"- 本臂 ΔPI 均值 = {sum(arm)/len(arm):.4f}")
        w(f"- 配对差均值 = {sum(diff)/len(diff):.4f}（SD {pd.Series(diff).std():.4f}）")
        w(f"- t = {t:.3f}，p = {p:.3g}")

    paired_multi(RD / "llm_ablation_no_post", blocks,
                 "3. DeepSeek 消融·冻结表达（vs 主实验，同种子配对）")

    # 4. Qwen 复现
    qf = load_summary(RD / "llm_qwen_full")
    w("\n## 4. Qwen 跨模型复现（200 runs）")
    cell_table(qf, "逐格均值")

    # 5. Qwen 消融
    w("\n## 5. Qwen 在线消融（vs Qwen 复现，同种子配对）")
    for arm, label in [("llm_qwen_abl_no_post", "5.1 冻结表达"),
                       ("llm_qwen_abl_no_emotion", "5.2 冻结情绪"),
                       ("llm_qwen_abl_positive_control", "5.3 阳性对照")]:
        paired_multi(RD / arm, [RD / "llm_qwen_full"], label)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[已写入] {OUT}")


if __name__ == "__main__":
    main()
