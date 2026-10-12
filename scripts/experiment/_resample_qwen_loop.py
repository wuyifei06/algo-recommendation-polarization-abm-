# -*- coding: utf-8 -*-
"""Qwen 循环重采：重复"扫描超标 → 留档 → 归档旧工件 → 同种子重采"，直到 0 个 run 超 2%。

与 _resample_qwen.py 的区别
--------------------------
* 单轮版（pass 1）用于把 47 个超标 run 拉回口径；本脚本做**后续迭代 pass**，
  只处理仍然 >2% 的 run，直到全部达标（用户 2026-10-09 明确要求"直到全部达标"）。
* 每轮旧值写入 `_resample_qwen_pass_history.json`（含 pass 号），工件以
  `.resampled_p<N>_bak` 归档（**不删除任何文件**，绕开沙箱 safe-delete）。

方法学留档（论文如实披露）
--------------------------
"重采到全部 ≤2%"比原团队的单轮默认更进取，含义是按结果分布收缩样本；
用户已知情并明确选择。论文必须报告：轮数、每轮移出/移入的 run、旧值分布。
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
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
RD = Path(_PROJ_ROOT + r"\results_direction")
RUNNER = HERE / "run_stage_direction.py"
PYTHON = _PYTHON
HISTORY = HERE / "_resample_qwen_pass_history.json"
LOOP_LOG = HERE / "_resample_loop.log"
THRESH = 0.02
MAX_PASSES = 12
TARGETS = ["llm_qwen_full", "llm_qwen_abl_no_post",
           "llm_qwen_abl_no_emotion", "llm_qwen_abl_positive_control"]
EXPECTED = {"llm_qwen_full": 200, "llm_qwen_abl_no_post": 90,
            "llm_qwen_abl_no_emotion": 90, "llm_qwen_abl_positive_control": 10}
ENV_QWEN = Path(_PROJ_ROOT + r"\.env.qwen")


def balance_alive() -> bool | None:
    """最小化测试调用判断百炼是否可用（200=可用；Arrearage/其他=不可用；异常=None）。"""
    try:
        import requests
        env = {}
        for line in ENV_QWEN.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip())
        r = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": "Bearer " + env["LLM_API_KEY"],
                     "Content-Type": "application/json"},
            json={"model": env["LLM_MODEL"],
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1}, timeout=30)
        if r.status_code == 200:
            return True
        code = ""
        try:
            code = (r.json().get("error") or {}).get("code", "")
        except Exception:
            pass
        if code == "Arrearage" or r.status_code == 400:
            return False
        return None
    except Exception:
        return None


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOOP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def scan_over() -> dict:
    """{批次名: [(run_id, fallback_rate)]}，只含当前可见数据 >2% 的 run。

    数据源 = 各分片 summary 的并集（分片是实时真源；合并版在批间可能滞后）。
    """
    out = {}
    for name in TARGETS:
        base = RD / name
        rows = []
        for sh in sorted(base.glob("shard*_of*")):
            s = sh / "llm_robust_summary.csv"
            if s.exists():
                try:
                    rows.append(pd.read_csv(s))
                except Exception:
                    pass
        if not rows:
            m = base / "llm_robust_summary.csv"
            if m.exists():
                try:
                    rows = [pd.read_csv(m)]
                except Exception:
                    pass
        hits = []
        for df in rows:
            for _, r in df.iterrows():
                try:
                    if float(r["llm_fallback_rate"]) > THRESH:
                        hits.append((int(r["run_id"]), float(r["llm_fallback_rate"])))
                except (KeyError, TypeError, ValueError):
                    continue
        # 去重（同 run 只应属一个分片，这里防御性处理）
        seen = {}
        for rid, fb in hits:
            seen[rid] = fb
        if seen:
            out[name] = sorted(seen.items())
    return out


def incomplete_batches() -> dict:
    """{批次名: (当前行数, 期望)}，只含行数不足的批次（分片并集口径）。"""
    bad = {}
    for name in TARGETS:
        base = RD / name
        n = 0
        for sh in sorted(base.glob("shard*_of*")):
            s = sh / "llm_robust_summary.csv"
            if s.exists():
                try:
                    n += len(pd.read_csv(s))
                except Exception:
                    pass
        if n < EXPECTED[name]:
            bad[name] = (n, EXPECTED[name])
    return bad


def park(f: Path, pas: int) -> bool:
    dst = f.with_name(f.name + (".resampled_bak" if pas == 1 else f".resampled_p{pas}_bak"))
    if not f.exists():
        return True
    try:
        os.replace(f, dst)      # 覆盖同轮旧档；不删除
        return True
    except OSError as exc:
        log(f"  ⚠ 归档失败 {f.name}: {exc}")
        return False


def clear_one(name: str, rid: int, pas: int, history: dict) -> None:
    base = RD / name
    # 1) 留档当前行（合并版优先）
    row = None
    for sv in [base / "llm_robust_summary.csv"] + \
              [d / "llm_robust_summary.csv" for d in sorted(base.glob("shard*_of*"))]:
        if not sv.exists():
            continue
        try:
            df = pd.read_csv(sv)
        except Exception:
            continue
        hit = df[df["run_id"].astype(int) == rid]
        if len(hit):
            if row is None:
                row = {k: (None if pd.isna(v) else v) for k, v in hit.iloc[0].to_dict().items()}
            df[df["run_id"].astype(int) != rid].to_csv(sv, index=False)
    history.setdefault(name, {}).setdefault(str(rid), []).append(
        {"pass": pas, "old": row})
    HISTORY.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    # 2) 归档工件 + 清 manifest
    for sh in (sorted(base.glob("shard*_of*")) or [base]):
        for sub, suffix in (("checkpoints", ".checkpoint.json"),
                            ("runs", ".csv"), ("calls", ".jsonl")):
            park(sh / sub / f"llm_robust_run_{rid}{suffix}", pas)
        for mv in (sh / "manifest.json",):
            if mv.exists():
                try:
                    payload = json.loads(mv.read_text(encoding="utf-8"))
                    new = [r for r in payload.get("runs", []) if int(r.get("run_id", -1)) != rid]
                    if len(new) != len(payload.get("runs", [])):
                        payload["runs"] = new
                        mv.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                except Exception:
                    pass
    for mv in (base / "manifest.json",):
        if mv.exists():
            try:
                payload = json.loads(mv.read_text(encoding="utf-8"))
                new = [r for r in payload.get("runs", []) if int(r.get("run_id", -1)) != rid]
                if len(new) != len(payload.get("runs", [])):
                    payload["runs"] = new
                    mv.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            except Exception:
                pass


def main() -> None:
    history = json.loads(HISTORY.read_text(encoding="utf-8")) if HISTORY.exists() else {}

    def run_runner(tag: str) -> bool:
        """跑一轮 runner（补齐缺失 + 重采已归档的），返回是否正常退出。"""
        alive = balance_alive()
        if alive is False:
            log(f"⚠ 百炼账户不可用（欠费/其他 400）→ 干净退出，充值后重跑本脚本即可接续。")
            return False
        code = subprocess.run(
            [PYTHON, "-X", "utf8", str(RUNNER), "--stage", "qwen-all", "--workers", "8"],
            cwd=str(HERE))
        ok = code.returncode == 0
        if not ok:
            alive = balance_alive()
            if alive is False:
                log(f"⚠ {tag} 因欠费失败 → 干净退出；充值后重跑本脚本接续。")
                return False
            log(f"⚠ {tag} 退出码 {code.returncode}（非欠费），进入下一轮观察。")
        return ok

    for pas in range(2, MAX_PASSES + 1):
        incomplete = incomplete_batches()
        if incomplete:
            log(f"=== pass {pas} 前置：批次行数不足 {incomplete} → 先补齐缺失 run")
            if not run_runner("补齐"):
                return
            continue

        over = scan_over()
        if not over:
            log("全部达标（0 个 run >2%）且行数完整，循环结束。")
            return
        ids = {k: [r for r, _ in v] for k, v in over.items()}
        total = sum(len(v) for v in over.values())
        log(f"=== pass {pas}：{total} 个超标 run {ids} → 留档并重采")
        for name, hits in over.items():
            for rid, fb in hits:
                clear_one(name, rid, pas, history)
        if not run_runner(f"pass {pas}"):
            return

    log(f"达到最大轮数 {MAX_PASSES}，停止；仍需人工处理：{scan_over()}")


if __name__ == "__main__":
    main()
