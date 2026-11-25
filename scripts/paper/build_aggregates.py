# -*- coding: utf-8 -*-
"""Build the paper-reproduction aggregate tables from raw run outputs.

Design goals
------------
- Everything in ``results/aggregates/`` is derived from raw experiment outputs
  and small enough to live in git. Raw per-step series, checkpoints and call
  logs stay out of git (see .gitignore).
- Only **paper-口径** batches are included:
    * rule arms: rule_main_d4, rule_small_d4, rule_spiral
    * paper LLM arms (signed-feed prompt): probe/results_prompt_direction and
      results_direction/* (T=100, random-feed, ablations, N=100, Qwen)
    * legacy prompt variants used in the README's "Prompt variants" table:
      results/llm_gradient_noleak (DeepSeek unsigned),
      results/llm_qwen_full (Qwen unsigned)
  Deprecated / non-paper batches are intentionally excluded:
  ``llm_decouple_*``, ``llm_robust_grid``, old unsigned ablation arms, old
  N=100 / random-feed batches, and the quarantined Qwen batches
  (llm_qwen_r10*, llm_qwen_slope_boost*, llm_qwen_check, llm_qwen_fb_fix,
  llm_gradient_r20).

Outputs
-------
- runs_laststep.csv : one row per run (last step)
- pi_by_condition.csv : per dataset x alpha x gamma descriptive stats
- effect_sizes.csv : per dataset x gamma OLS of PI ~ alpha
- sensitivity_runs.csv : unchanged (175-run parameter scan, shipped earlier)

Usage
-----
    python scripts/paper/build_aggregates.py

Environment override for the probe directory (outside this repo):
    ABM_PROBE_ROOT=/path/to/probe_提示词明确性
"""
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]          # repo root（最终代码和数据）
OUT = ROOT / "results" / "aggregates"
PROBE = Path(os.environ.get("ABM_PROBE_ROOT", ROOT.parent.parent / "probe_提示词明确性"))

RUN_RE = re.compile(r"^llm_robust_run_(\d+)\.csv$")
SPIRAL_RE = re.compile(r"^rule_spiral_run_(\d+)\.csv$")

COLS = ["dataset", "alpha", "gamma", "seed", "run_id", "steps",
        "PI", "HE", "EE", "EM", "SR"]
EXTRA = ["spiral_c", "spiral_m", "mean_exposure_distance"]   # 仅螺旋扩展批有；其余为 NaN


# ---------------------------------------------------------------- loaders
def _last_rows(files, id_fn):
    rows = []
    for f in sorted(files):
        try:
            d = pd.read_csv(f)
        except Exception as exc:                       # pragma: no cover
            print(f"[warn] skip {f}: {exc}", file=sys.stderr)
            continue
        if not len(d):
            continue
        last = d.loc[d["step"].idxmax()].to_dict()
        last["run_id"] = id_fn(f)
        rows.append(last)
    return pd.DataFrame(rows)


def load_rule_table(path):
    d = pd.read_csv(path)
    # summary files hold one row per run（末步）
    return d.loc[d.groupby("run_id")["step"].idxmax()].reset_index(drop=True)


def load_rule_spiral(root):
    files = glob.glob(str(root / "逐run时间序列" / "rule_spiral_run_*.csv"))
    return _last_rows(files, lambda f: int(SPIRAL_RE.match(os.path.basename(f)).group(1)))


def load_llm_runs(root):
    files = [f for f in glob.glob(str(root / "**" / "runs" / "llm_robust_run_*.csv"),
                                  recursive=True)
             if RUN_RE.match(os.path.basename(f))]
    if not files:
        files = [f for f in glob.glob(str(root / "**" / "llm_robust_run_*.csv"),
                                      recursive=True)
                 if RUN_RE.match(os.path.basename(f))]
    return _last_rows(files, lambda f: int(RUN_RE.match(os.path.basename(f)).group(1)))


DATASETS = [
    ("results/rule_main_d4", ("csv", "results/rule_main_d4/rule_summary.csv")),
    ("results/rule_small_d4", ("csv", "results/rule_small_d4/rule_small_summary.csv")),
    ("results/rule_spiral", ("spiral", "results/rule_spiral")),
    ("probe/results_prompt_direction", ("llm", PROBE / "results_prompt_direction")),
    ("results_direction/llm_extend_T100", ("llm", "results_direction/llm_extend_T100")),
    ("results_direction/llm_randomfeed_control", ("llm", "results_direction/llm_randomfeed_control")),
    ("results_direction/llm_ablation_no_post", ("llm", "results_direction/llm_ablation_no_post")),
    ("results_direction/llm_ablation_no_emotion", ("llm", "results_direction/llm_ablation_no_emotion")),
    ("results_direction/llm_ablation_positive_control", ("llm", "results_direction/llm_ablation_positive_control")),
    ("results_direction/llm_n100_scale", ("llm", "results_direction/llm_n100_scale")),
    ("results_direction/llm_n100_boost", ("llm", "results_direction/llm_n100_boost")),
    ("results_direction/llm_qwen_full", ("llm", "results_direction/llm_qwen_full")),
    ("results_direction/llm_qwen_extend_T100", ("llm", "results_direction/llm_qwen_extend_T100")),
    ("results_direction/llm_qwen_abl_no_post", ("llm", "results_direction/llm_qwen_abl_no_post")),
    ("results_direction/llm_qwen_abl_no_emotion", ("llm", "results_direction/llm_qwen_abl_no_emotion")),
    ("results_direction/llm_qwen_abl_positive_control", ("llm", "results_direction/llm_qwen_abl_positive_control")),
    ("results/llm_gradient_noleak", ("llm", "results/llm_gradient_noleak")),
    ("results/llm_qwen_full", ("llm", "results/llm_qwen_full")),
]


def load_all():
    frames = []
    for label, (kind, src) in DATASETS:
        if kind == "csv":
            d = load_rule_table(ROOT / src)
        elif kind == "spiral":
            d = load_rule_spiral(ROOT / src)
        else:
            d = load_llm_runs(Path(src) if Path(src).is_absolute() else ROOT / src)
        if "step" in d.columns and "steps" in d.columns:
            d = d.drop(columns=["steps"])
        d = d.rename(columns={"step": "steps"})
        for c in ["PI", "HE", "EE", "EM", "SR", "alpha", "gamma", "seed", "run_id", "steps"]:
            if c not in d.columns:
                raise KeyError(f"{label}: missing column {c}")
        # 同一文件可能被多个 glob 命中；按 run 身份去重（spiral 的配对种子会跨 c 复用，
        # 所以不能只用 alpha/gamma/seed）
        d = d.drop_duplicates(subset=["alpha", "gamma", "seed", "run_id"]).copy()
        for c in EXTRA:
            if c not in d.columns:
                d[c] = float("nan")
        d["dataset"] = label
        d["alpha"] = d["alpha"].astype(float).round(4)
        d["gamma"] = d["gamma"].astype(float).round(4)
        d["seed"] = d["seed"].astype(int)
        d["run_id"] = d["run_id"].astype(int)
        d["steps"] = d["steps"].astype(int)
        frames.append(d[COLS + EXTRA])
        print(f"{label:55s} n={len(d):5d}  steps={int(d['steps'].iloc[0])}")
    runs = pd.concat(frames, ignore_index=True)
    order = {d: i for i, (d, _) in enumerate(DATASETS)}
    runs["_o"] = runs.dataset.map(order)
    runs = (runs.sort_values(["_o", "alpha", "gamma", "seed"], kind="mergesort")
                .drop(columns="_o").reset_index(drop=True))
    return runs


# ---------------------------------------------------------------- stats
def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def ols(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    X = np.column_stack([np.ones(len(y)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, k = X.shape
    dof = n - k
    s2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, t, p, r2


def save_csv(df, name):
    """Write CSV with LF endings（仓库统一 LF，避免 Windows 换行把整文件标成改动）。"""
    (OUT / name).write_bytes(
        df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )


def build_tables(runs):
    save_csv(runs.round(10), "runs_laststep.csv")

    rows = []
    for (ds, a, g), d in runs.groupby(["dataset", "alpha", "gamma"], sort=False):
        n = len(d)
        sd = float(d.PI.std(ddof=1)) if n > 1 else float("nan")
        half = 1.96 * sd / np.sqrt(n) if n > 1 else float("nan")
        rows.append({
            "dataset": ds, "alpha": a, "gamma": g, "n": n,
            "PI_mean": float(d.PI.mean()), "PI_sd": sd,
            "PI_lo": float(d.PI.mean() - half), "PI_hi": float(d.PI.mean() + half),
            "HE": float(d.HE.mean()), "EE": float(d.EE.mean()),
            "EM": float(d.EM.mean()), "SR": float(d.SR.mean()),
        })
    save_csv(pd.DataFrame(rows).round(6), "pi_by_condition.csv")

    rows = []
    for (ds, g), d in runs.groupby(["dataset", "gamma"], sort=False):
        k = d.alpha.nunique()
        if ds == "results/rule_spiral":   # 螺旋批操纵的是 c/m，不估计 alpha 斜率
            continue
        if k < 3 or d.PI.std(ddof=1) == 0:
            continue
        b, se, t, p, r2 = ols(d.PI, d.alpha)
        b_std = float(b[1] * d.alpha.std(ddof=1) / d.PI.std(ddof=1))
        rows.append({
            "dataset": ds, "gamma": g, "n": len(d), "k": int(k),
            "raw_slope": float(b[1]), "se": float(se[1]),
            "t": float(t[1]), "p": float(p[1]),
            "r2": float(r2), "std_beta": b_std,
        })
    eff = pd.DataFrame(rows).round(6)
    save_csv(eff, "effect_sizes.csv")
    return eff


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "sensitivity_runs.csv").exists():
        print("[error] sensitivity_runs.csv missing; keep the shipped 175-run scan",
              file=sys.stderr)
        sys.exit(1)
    runs = load_all()
    eff = build_tables(runs)
    print(f"\nrows={len(runs)}  datasets={runs.dataset.nunique()}  effect_sizes={len(eff)}")
    key = eff[eff.dataset.isin(["results/rule_main_d4", "results/rule_small_d4",
                                "probe/results_prompt_direction",
                                "results_direction/llm_extend_T100",
                                "results_direction/llm_qwen_full",
                                "results/llm_qwen_full"])]
    print(key[["dataset", "gamma", "n", "k", "std_beta", "p"]].to_string(index=False))
