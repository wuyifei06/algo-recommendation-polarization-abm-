"""重放结果 vs 原始档案 全量逐位校验。

对每个 run 文件逐列（float 精确相等）比对：
  rule_main_d4   <-> 成果归档/20_Rule验证/A_主实验_α剂量响应/rule_main_d4/逐run时间序列
  rule_small_d4  <-> .../rule_small_d4/runs
  rule_spiral    <-> .../B_螺旋放大/rule_spiral/逐run时间序列
  rule_structure <-> .../C_结构稳健性/rule_structure/<variant>
输出：replay_verify_report.json + 控制台摘要
"""
import glob
import json
import os
import re
import sys

import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_BASE_ROOT = _os.getenv("ABM_BASE_ROOT", r"D:\21130\Desktop\大创过程")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

PROJ = _PROJ_ROOT
REPLAY = os.path.join(PROJ, "results_replay_20261010")
ARC = _os.path.join(_BASE_ROOT, "成果归档", "20_Rule验证")

COLS = ["PI", "VPI", "SR", "ER", "EM", "HE", "EE", "mean_exposure_distance"]


def compare_dir(rep_dir, orig_dir, pat):
    rep_files = sorted(glob.glob(os.path.join(rep_dir, pat)))
    n_ok = n_diff = n_missing_orig = 0
    diffs = []
    for rf in rep_files:
        rid = re.search(r"run_(\d+)\.csv", rf).group(1)
        # 原始文件名与重放文件名同前缀（rule_run_/rule_small_run_/rule_spiral_run_）
        of = os.path.join(orig_dir, os.path.basename(rf))
        if not os.path.exists(of):
            n_missing_orig += 1
            diffs.append({"run": rid, "issue": "missing_orig"})
            continue
        a, b = pd.read_csv(of), pd.read_csv(rf)
        if len(a) != len(b):
            n_diff += 1
            diffs.append({"run": rid, "issue": "len", "la": len(a), "lb": len(b)})
            continue
        bad = [c for c in COLS if c in a.columns and c in b.columns
               and not (a[c].values == b[c].values).all()]
        if bad:
            n_diff += 1
            diffs.append({"run": rid, "issue": "cols", "cols": bad})
        else:
            n_ok += 1
    return {"n_rep": len(rep_files), "n_ok": n_ok, "n_diff": n_diff,
            "n_missing_orig": n_missing_orig, "diffs": diffs[:20]}


def main():
    report = {}

    report["rule_main_d4"] = compare_dir(
        os.path.join(REPLAY, "rule_main_d4", "逐run时间序列"),
        os.path.join(ARC, "A_主实验_α剂量响应", "rule_main_d4", "逐run时间序列"),
        "rule_run_*.csv")

    report["rule_small_d4"] = compare_dir(
        os.path.join(REPLAY, "rule_small_d4", "runs"),
        os.path.join(ARC, "A_主实验_α剂量响应", "rule_small_d4", "runs"),
        "rule_small_run_*.csv")

    report["rule_spiral"] = compare_dir(
        os.path.join(REPLAY, "rule_spiral", "逐run时间序列"),
        os.path.join(ARC, "B_螺旋放大", "rule_spiral", "逐run时间序列"),
        "rule_spiral_run_*.csv")

    report["rule_structure"] = {}
    for v in ["base", "comp10", "comp40", "H000", "H020", "H050", "K20", "N500", "T400"]:
        report["rule_structure"][v] = compare_dir(
            os.path.join(REPLAY, "rule_structure", v),
            os.path.join(ARC, "C_结构稳健性", "rule_structure", v),
            "rule_run_*.csv")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_verify_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 摘要
    total_ok = total_diff = 0
    for k, v in report.items():
        if k == "rule_structure":
            for v2, s in v.items():
                total_ok += s["n_ok"]; total_diff += s["n_diff"]
                print(f"structure/{v2:8s}: rep={s['n_rep']:4d} ok={s['n_ok']:4d} diff={s['n_diff']} missing_orig={s['n_missing_orig']}")
                if s["diffs"]:
                    print("   ", s["diffs"][:3])
        else:
            total_ok += v["n_ok"]; total_diff += v["n_diff"]
            print(f"{k:14s}: rep={v['n_rep']:4d} ok={v['n_ok']:4d} diff={v['n_diff']} missing_orig={v['n_missing_orig']}")
            if v["diffs"]:
                print("   ", v["diffs"][:3])
    print(f"\nTOTAL: ok={total_ok} diff={total_diff}")
    print("report:", out)
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
