# -*- coding: utf-8 -*-
"""今晚两批新实验的终版统计分析（预注册方案，确定性输出）。

覆盖：
  A. Qwen T=100 剂量-响应（分 γ 与合并）
  B. Qwen 时间×α 交互（T30 vs T100，同格合并回归）
  C. t=30 对齐点比较（T100 的 PI(t=30) vs T30 终点）
  D. Qwen T100 轨迹与边界吸附（t1/t30/t100 均值、末 10 步斜率、|opinion|>=0.999 占比）
  E. N=100 规模复核（n=100 规模档 60 runs 合并：均值、β、CI、与 N=30 主实验对照）
  F. 质量门核查：全部 run 兜底率 <= 2%（违例列出到 _over2_runs.txt 供重掷）

输出：
  new_batches_results.json   全部数字（供 integrate_paper.py 使用）
  new_batches_report.md      人读摘要

用法：venv python（需 pandas/scipy/numpy）。任何数据缺失都降级输出，不抛异常。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

RD = Path(_PROJ_ROOT + r"\results_direction")
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "new_batches_results.json"
OUT_MD = HERE / "new_batches_report.md"
OVER2 = Path(_PROBE_ROOT + r"\_over2_runs.txt")

T100_DIR = RD / "llm_qwen_extend_T100"
T30_CSV = RD / "llm_qwen_full" / "llm_robust_summary.csv"
N100_SCALE = RD / "llm_n100_scale" / "llm_robust_summary.csv"
N100_BOOST_DIR = RD / "llm_n100_boost"

ALPHA_SUBSET = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)  # 完整版：全部 8 档
# 已发表论文中的 DeepSeek 常数（不重算，防口径漂移）
DS_CONST = {"beta_t30": -0.405, "beta_t100": -0.053, "interact_t": 2.18, "interact_p": 0.030}


def ols(y, X):
    """多变量 OLS（自动带截距）：返回 (coef dict, n)。X 为 dict列名->list。"""
    names = ["const"] + list(X.keys())
    Xm = np.column_stack([np.ones(len(next(iter(X.values()))))] +
                         [np.asarray(X[k], dtype=float) for k in names[1:]])
    yv = np.asarray(y, dtype=float)
    ok = ~(np.isnan(Xm).any(axis=1) | np.isnan(yv))
    Xm, yv = Xm[ok], yv[ok]
    n = len(yv)
    if n <= len(names) + 1:
        return None, n
    coef, res, rank, _ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ coef
    dof = n - rank
    if dof <= 0:
        return None, n
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xm.T @ Xm)
    se = np.sqrt(np.diag(cov))
    out = {}
    for i, k in enumerate(names):
        t = coef[i] / se[i] if se[i] > 0 else 0.0
        p = 2 * (1 - stats.t.cdf(abs(t), dof))
        out[k] = {"coef": float(coef[i]), "se": float(se[i]), "t": float(t), "p": float(p)}
    tss = float(((yv - yv.mean()) ** 2).sum())
    r2 = float(1 - (resid @ resid) / tss) if tss > 0 else 0.0
    return {"coefs": out, "n": n, "r2": r2}, n


def load_stage_summaries(stage_dir: Path, steps: int) -> pd.DataFrame:
    rows = []
    for sf in glob.glob(str(stage_dir / "shard*_of*" / "llm_robust_summary.csv")) + \
              glob.glob(str(stage_dir / "llm_robust_summary.csv")):
        try:
            d = pd.read_csv(sf)
        except Exception:
            continue
        d = d[(d.get("step") == steps) & (d.get("status").isin(["completed", "completed_with_fallback"]))]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["run_id"])
    return d


def qwen_t100_block(res):
    """A/B/C/D：T100 剂量响应、时间交互、对齐点、轨迹与吸附。"""
    out = {}
    d = load_stage_summaries(T100_DIR, 100)
    if d.empty:
        return {"error": "T100 summary 缺失"}
    d = d[d["alpha"].isin(ALPHA_SUBSET)]
    out["n_runs"] = int(len(d))
    out["fb_max"] = float(d["llm_fallback_rate"].max())
    out["fb_mean"] = float(d["llm_fallback_rate"].mean())

    # A. 剂量-响应
    for g, sub in [("1.0", d[d.gamma == 1.0]), ("0.1", d[d.gamma == 0.1]), ("combined", d)]:
        r, n = ols(sub["PI"], {"alpha": sub["alpha"], **({"gamma": (sub["gamma"] == 0.1).astype(int)} if g == "combined" else {})})
        key = "beta_combined" if g == "combined" else f"beta_g{g}"
        if r:
            c = r["coefs"]["alpha"]
            out[key] = {"beta": c["coef"], "se": c["se"], "t": c["t"], "p": c["p"],
                        "r2": r["r2"], "n": r["n"]}
            if g == "combined":
                cg = r["coefs"].get("gamma")
                out["gamma_coef_combined"] = cg

    # B. 时间×α 交互（T30 同格 + T100）
    t30 = pd.read_csv(T30_CSV)
    t30 = t30[(t30["step"] == 30) & t30["status"].isin(["completed", "completed_with_fallback"])]
    t30 = t30[t30["alpha"].isin(ALPHA_SUBSET)]
    pooled = pd.concat([
        t30[["PI", "alpha", "gamma"]].assign(batch=0),
        d[["PI", "alpha", "gamma"]].assign(batch=1),
    ], ignore_index=True)
    r, n = ols(pooled["PI"], {"alpha": pooled["alpha"], "batch": pooled["batch"],
                              "axb": pooled["alpha"] * pooled["batch"],
                              "gamma": (pooled["gamma"] == 0.1).astype(int)})
    if r:
        c = r["coefs"]["axb"]
        out["time_interaction"] = {"coef_per_alpha": c["coef"], "se": c["se"], "t": c["t"],
                                   "p": c["p"], "n": n,
                                   "note": "PI~α+batch+α×batch+γ；T30 n=%d, T100 n=%d" % (
                                       int((pooled.batch == 0).sum()), int((pooled.batch == 1).sum()))}

    # C. t=30 对齐点 + D. 轨迹/吸附（从 checkpoint 时间序列）
    ts_rows, clip_rows = [], []
    for f in glob.glob(str(T100_DIR / "shard*_of*" / "checkpoints" / "*.checkpoint.json")):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ts = {int(x["step"]): x for x in j.get("metrics_time_series", [])}
        get = lambda s, k: ts.get(s, {}).get(k)
        if get(100, "PI") is None:
            continue
        ts_rows.append({"alpha": j["alpha"], "gamma": j["gamma"],
                        "PI_t1": get(1, "PI"), "PI_t30": get(30, "PI"), "PI_t100": get(100, "PI")})
        ags = j.get("agents", [])
        if ags:
            clip_rows.append(sum(1 for a in ags if abs(a.get("opinion", 0)) >= 0.999) / len(ags))
    if ts_rows:
        tdf = pd.DataFrame(ts_rows)
        out["traj"] = {"PI_t1": float(tdf.PI_t1.mean()), "PI_t30": float(tdf.PI_t30.mean()),
                       "PI_t100": float(tdf.PI_t100.mean()), "n": int(len(tdf))}
        # 末 10 步斜率：t100-t91 简化口径 + 每 run 线性斜率
        slopes = []
        for f in glob.glob(str(T100_DIR / "shard*_of*" / "checkpoints" / "*.checkpoint.json")):
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            ts = {int(x["step"]): x["PI"] for x in j.get("metrics_time_series", [])}
            xs = np.array([s for s in range(91, 101) if s in ts])
            ys = np.array([ts[s] for s in xs])
            if len(xs) >= 5:
                slopes.append(np.polyfit(xs, ys, 1)[0])
        if slopes:
            out["last10_slope_mean"] = float(np.mean(slopes))
        if clip_rows:
            out["boundary_clip_t100"] = float(np.mean(clip_rows))
        # 对齐点：T100 的 PI_t30 vs T30 终点（同格回归差）
        ali = pd.concat([
            t30[["PI", "alpha"]].assign(batch=0),
            tdf[["PI_t30", "alpha"]].rename(columns={"PI_t30": "PI"}).assign(batch=1),
        ], ignore_index=True)
        r2, n2 = ols(ali["PI"], {"alpha": ali["alpha"], "batch": ali["batch"],
                                 "axb": ali["alpha"] * ali["batch"]})
        if r2:
            c = r2["coefs"]["batch"]
            out["aligned_t30"] = {"batch_diff": c["coef"], "p": c["p"], "n": n2,
                                  "axb_p": r2["coefs"]["axb"]["p"]}
    return out


def n100_block(res):
    """E. 规模复核（scale 30 + boost 30 → 60）。"""
    out = {}
    parts = []
    try:
        s = pd.read_csv(N100_SCALE)
        s = s[(s["step"] == 30) & s["status"].isin(["completed", "completed_with_fallback"])]
        parts.append(s)
        out["n_scale"] = int(len(s))
    except Exception:
        pass
    b = load_stage_summaries(N100_BOOST_DIR, 30)
    if not b.empty:
        parts.append(b)
        out["n_boost"] = int(len(b))
    if not parts:
        return {"error": "N100 数据缺失"}
    d = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["run_id"])
    out["n_total"] = int(len(d))
    cells = d.groupby("alpha")[["PI", "EM", "SR", "ER", "HE"]].mean().round(4)
    out["cell_means"] = {str(a): {k: float(v) for k, v in row.items()} for a, row in cells.iterrows()}
    r, n = ols(d["PI"], {"alpha": d["alpha"]})
    if r:
        c = r["coefs"]["alpha"]
        # 95% CI（t 临界近似用正态 1.96）
        ci = 1.96 * c["se"]
        # 标准化口径（与论文既有规模段落一致）：β_std = β_raw × sd_x / sd_y
        sx, sy = float(d["alpha"].std(ddof=1)), float(d["PI"].std(ddof=1))
        beta_std = c["coef"] * sx / sy
        ci_std = ci * sx / sy
        out["beta"] = {"beta": c["coef"], "se": c["se"], "t": c["t"], "p": c["p"],
                       "ci95": [c["coef"] - ci, c["coef"] + ci],
                       "beta_std": beta_std, "ci95_std": [beta_std - ci_std, beta_std + ci_std],
                       "n": n}
        out["reversal"] = bool(c["p"] < 0.05 and c["coef"] > 0)
        out["covers_main"] = bool(c["coef"] - ci <= DS_CONST["beta_t30"] <= c["coef"] + ci)
    return out


def quality_gate():
    """F. 两批全部 run 的兜底率门（<=2%）。"""
    bad = []
    for stage_dir, steps in [(T100_DIR, 100), (N100_BOOST_DIR, 30)]:
        d = load_stage_summaries(stage_dir, steps)
        for _, row in d.iterrows():
            if float(row.get("llm_fallback_rate", 0)) > 0.02:
                bad.append({"stage": stage_dir.name, "run_id": int(row["run_id"]),
                            "alpha": float(row["alpha"]), "gamma": float(row["gamma"]),
                            "rate": float(row["llm_fallback_rate"])})
    return bad


def main():
    res = {"generated": __import__("datetime").datetime.now().strftime("%F %T")}
    try:
        res["qwen_t100"] = qwen_t100_block(res)
    except Exception as e:
        res["qwen_t100"] = {"error": str(e)}
    try:
        res["n100"] = n100_block(res)
    except Exception as e:
        res["n100"] = {"error": str(e)}
    try:
        bad = quality_gate()
        res["over2_runs"] = bad
        if bad:
            OVER2.parent.mkdir(parents=True, exist_ok=True)
            with open(OVER2, "a", encoding="utf-8") as fh:
                for b in bad:
                    fh.write(f"{res['generated']} {b['stage']} run_{b['run_id']} a={b['alpha']} g={b['gamma']} rate={b['rate']:.4%}\n")
    except Exception as e:
        res["over2_runs"] = {"error": str(e)}
    res["ds_constants"] = DS_CONST

    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# 新批次分析报告（自动生成）", ""]
    q = res.get("qwen_t100", {})
    md.append("## Qwen T=100（%s runs，兜底 max %.3f%%）" % (q.get("n_runs", "?"), q.get("fb_max", 0) * 100))
    for k in ("beta_g1.0", "beta_g0.1", "beta_combined"):
        if k in q:
            b = q[k]
            md.append(f"- {k}: β={b['beta']:.4f} (t={b['t']:.2f}, p={b['p']:.4g}, R²={b['r2']:.3f}, n={b['n']})")
    if "time_interaction" in q:
        t = q["time_interaction"]
        md.append(f"- 时间×α 交互: {t['coef_per_alpha']:+.4f}/单位α (t={t['t']:.2f}, p={t['p']:.4g}, n={t['n']})")
    if "traj" in q:
        t = q["traj"]
        md.append(f"- 轨迹: PI t1={t['PI_t1']:.4f} → t30={t['PI_t30']:.4f} → t100={t['PI_t100']:.4f}")
    if "last10_slope_mean" in q:
        md.append(f"- 末10步斜率均值: {q['last10_slope_mean']:+.5f}")
    if "boundary_clip_t100" in q:
        md.append(f"- 边界吸附占比(t100): {q['boundary_clip_t100']:.3f}")
    if "aligned_t30" in q:
        a = q["aligned_t30"]
        md.append(f"- t=30 对齐点批间差: {a['batch_diff']:+.4f} (p={a['p']:.4g})")
    md.append("")
    n1 = res.get("n100", {})
    md.append("## N=100 规模复核（合计 %s runs）" % n1.get("n_total", "?"))
    if "cell_means" in n1:
        for a, m in n1["cell_means"].items():
            md.append(f"- α={a}: PI={m['PI']:.4f} EM={m['EM']:.4f} SR={m['SR']:.4f} ER={m['ER']:.4f} HE={m['HE']:.4f}")
    if "beta" in n1:
        b = n1["beta"]
        md.append(f"- β={b['beta']:+.4f} (95% CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}], p={b['p']:.4g}, n={b['n']})")
        md.append(f"- 反转: {'是' if n1.get('reversal') else '未检出'}; CI 覆盖主实验 β(−0.405): {'是' if n1.get('covers_main') else '否'}")
    md.append("")
    bad = res.get("over2_runs", [])
    md.append("## 质量门（≤2%%）：%d 个 run 超标" % (len(bad) if isinstance(bad, list) else -1))
    if isinstance(bad, list):
        for b in bad:
            md.append(f"- {b['stage']} run_{b['run_id']} α={b['alpha']} γ={b['gamma']} rate={b['rate']:.4%}")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))
    print("\nJSON ->", OUT_JSON)


if __name__ == "__main__":
    main()
