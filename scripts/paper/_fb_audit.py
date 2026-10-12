# -*- coding: utf-8 -*-
"""兜底率审计：解析 calls jsonl 的 raw_preview，用与 robust_client_v2 相同的 range 规则
统计 out_of_range 率。对比 当前 T100 vs 历史 T30（qwen_full）。"""
import json
import glob
import os
import sys
from collections import defaultdict
# --- portability shim (auto-added) ---------------------------------
import os as _os
_PROJ_ROOT = _os.getenv("ABM_PROJ_ROOT", r"D:\21130\Desktop\大创过程\实验代码\最终代码和数据")
_PROBE_ROOT = _os.getenv("ABM_PROBE_ROOT", r"D:\21130\Desktop\大创过程\probe_提示词明确性")
_WORK_ROOT = _os.getenv("ABM_WORK_ROOT", r"C:\Users\21130\WorkBuddy\2026-10-08-15-15-14\_work\paper_rev")
_PYTHON = _os.getenv("ABM_PYTHON", r"C:\Users\21130\AppData\Local\Programs\Python\Python312\python.exe")
# -------------------------------------------------------------------

RANGE_STANCE = 0.05
RANGE_EMOTION = 0.1


def extract_items(raw_preview: str):
    """从 raw_preview（API 原始 JSON 字符串）提取 content 数组的 items。"""
    try:
        raw = json.loads(raw_preview)
    except Exception:
        return None
    # 兼容 OpenAI 格式：choices[0].message.content 是字符串(JSON数组)
    choices = raw.get("choices")
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str):
        return None
    # 去掉可能的 markdown 围栏
    c = content.strip()
    if c.startswith("```"):
        c = c.split("```")[1] if "```" in c[3:] else c[3:]
        if c.startswith("json"):
            c = c[4:]
    try:
        items = json.loads(c)
    except Exception:
        return None
    if not isinstance(items, list):
        return None
    return items


def check_items(items):
    """返回 (n_viol_stance, n_viol_emotion, n_items)。"""
    vs = ve = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        sd = it.get("stance_delta")
        ed = it.get("emotion_delta")
        if isinstance(sd, (int, float)) and not (-RANGE_STANCE <= sd <= RANGE_STANCE):
            vs += 1
        if isinstance(ed, (int, float)) and not (-RANGE_EMOTION <= ed <= RANGE_EMOTION):
            ve += 1
    return vs, ve, len(items)


def audit_run(calls_file: str, max_step=None):
    """审计单个 run 的 calls 文件。返回逐 step 统计。"""
    per_step = defaultdict(lambda: {"stance": 0, "emotion": 0, "items": 0, "calls": 0})
    with open(calls_file, encoding="utf-8") as fh:
        for ln in fh:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if not d.get("batch"):
                continue
            st = d.get("step", -1)
            if max_step is not None and st > max_step:
                continue
            rp = d.get("raw_preview")
            if not rp:
                continue
            items = extract_items(rp)
            if items is None:
                per_step[st]["calls"] += 1  # 解析不了也算一次调用
                continue
            vs, ve, n = check_items(items)
            per_step[st]["stance"] += vs
            per_step[st]["emotion"] += ve
            per_step[st]["items"] += n
            per_step[st]["calls"] += 1
    return per_step


def summarize(per_step, max_step=None):
    S = E = I = C = 0
    steps = []
    for st in sorted(per_step):
        if max_step is not None and st > max_step:
            continue
        d = per_step[st]
        S += d["stance"]; E += d["emotion"]; I += d["items"]; C += d["calls"]
        steps.append(st)
    return {"stance": S, "emotion": E, "items": I, "calls": C,
            "viol": S + E, "rate": (S + E) / I if I else 0.0,
            "steps": (steps[0], steps[-1]) if steps else None}


if __name__ == "__main__":
    CUR = _PROJ_ROOT + r"\results_direction\llm_qwen_extend_T100"
    HIST = _PROJ_ROOT + r"\results_direction\llm_qwen_full"
    # ---------- 当前 T100：全部 8 分片 ----------
    print("=" * 70)
    print("【当前 T100】各分片 raw_preview 逐条审核（全期）")
    print("=" * 70)
    tot = {"stance": 0, "emotion": 0, "items": 0, "calls": 0, "viol": 0}
    for sh in sorted(glob.glob(os.path.join(CUR, "shard*_of08"))):
        cfiles = sorted(glob.glob(os.path.join(sh, "calls", "*.jsonl")))
        if not cfiles:
            continue
        # 每分片可能有多个 run 文件（当前都在 run 首个）
        agg = defaultdict(lambda: {"stance": 0, "emotion": 0, "items": 0, "calls": 0})
        for cf in cfiles:
            ps = audit_run(cf)
            for st, d in ps.items():
                for k in agg[st]:
                    agg[st][k] += d[k]
        s = summarize(agg)
        shname = os.path.basename(sh)
        print(f"  {shname}: steps {s['steps']} | calls={s['calls']} items={s['items']} "
              f"viol={s['viol']} (stance {s['stance']} + emotion {s['emotion']}) rate={s['rate']:.4%}")
        for k in ("stance", "emotion", "items", "calls", "viol"):
            tot[k] += s[k]
    print(f"  >>> 合计: viol={tot['viol']}/{tot['items']} = {tot['viol']/tot['items']:.4%}")

    # ---------- 历史 T30：抽样 8 个 run（每分片 1 个）----------
    print()
    print("=" * 70)
    print("【历史 T30 (qwen_full)】抽样对比：取每分片第 1 个 run 的前 22 步")
    print("=" * 70)
    htot = {"stance": 0, "emotion": 0, "items": 0, "calls": 0, "viol": 0}
    for sh in sorted(glob.glob(os.path.join(HIST, "shard*_of08"))):
        cfiles = sorted(glob.glob(os.path.join(sh, "calls", "*.jsonl")))
        if not cfiles:
            continue
        cf = cfiles[0]
        ps = audit_run(cf, max_step=22)
        s = summarize(ps, max_step=22)
        shname = os.path.basename(sh)
        run_id = os.path.basename(cf).replace("llm_robust_run_", "").replace(".jsonl", "")
        print(f"  {shname} run{run_id}: items={s['items']} viol={s['viol']} "
              f"(stance {s['stance']} + emotion {s['emotion']}) rate={s['rate']:.4%}")
        for k in ("stance", "emotion", "items", "calls", "viol"):
            htot[k] += s[k]
    print(f"  >>> 历史合计(前22步): viol={htot['viol']}/{htot['items']} = "
          f"{htot['viol']/htot['items']:.4%}" if htot['items'] else "无数据")
