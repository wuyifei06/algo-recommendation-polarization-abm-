# -*- coding: utf-8 -*-
"""清理被 Arrearage（欠费期 400 兜底）污染的 run。

判定：run 的 calls jsonl 中出现 fallback_reason 含 HTTPError 的记录 ⇒ 该 run
在欠费窗口内产生过兜底假决策 ⇒ 整体作废（数据不可信，不能续跑）。

动作：删除该 run 的 checkpoint / runs csv / calls jsonl，并从分片 summary 与
manifest 中移除对应行 ⇒ 下次 run_experiment 会把它当新 run 完整重跑。
其余 run（含昨晚欠费前完成、今天干净续跑的）一律不动。

用法：python _cleanup_arrearage.py            # 只报告
      python _cleanup_arrearage.py --wipe    # 实际删除
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import pandas as pd
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
# -------------------------------------------------------------------

ROOT = Path(_PROJ_ROOT + r"\results_direction")


def scan_and_clean(arm_dir: Path, wipe: bool) -> None:
    bad: dict[Path, set] = {}
    for fp in glob.glob(str(arm_dir / "shard*" / "calls" / "llm_robust_run_*.jsonl")):
        p = Path(fp)
        run_id = int(p.stem.replace("llm_robust_run_", ""))
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if "HTTPError" not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad.setdefault(p.parent, set()).add(run_id)
                break
            if (rec.get("quality") or {}).get("fallback_reason", "").find("HTTPError") >= 0:
                bad.setdefault(p.parent, set()).add(run_id)
                break

    for calls_dir, run_ids in sorted(bad.items()):
        shard = calls_dir.parent
        print(f"{shard.name}: 污染 run_id = {sorted(run_ids)}")
        if not wipe:
            continue
        for rid in sorted(run_ids):
            stem = f"llm_robust_run_{rid}"
            for sub in ("checkpoints", "runs", "calls"):
                for suffix in (".checkpoint.json", ".csv", ".jsonl"):
                    f = shard / sub / (stem + suffix)
                    if f.exists():
                        f.unlink()
            s = shard / "llm_robust_summary.csv"
            if s.exists():
                df = pd.read_csv(s)
                df = df[df["run_id"].astype(int) != rid]
                df.to_csv(s, index=False)
            m = shard / "manifest.json"
            if m.exists():
                payload = json.loads(m.read_text(encoding="utf-8"))
                payload["runs"] = [r for r in payload.get("runs", [])
                                   if int(r.get("run_id", -1)) != rid]
                m.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"  -> 已删除并从 summary/manifest 移除（重跑时自动补齐）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true", help="实际删除（默认只报告）")
    args = ap.parse_args()
    for arm_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        if not list(arm_dir.glob("shard*/calls")):
            continue
        scan_and_clean(arm_dir, args.wipe)
    print("完成。" if args.wipe else "（仅报告；加 --wipe 执行删除）")


if __name__ == "__main__":
    main()
