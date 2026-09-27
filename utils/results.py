"""实验结果目录与单次 run 文件的兼容处理。

项目早期把结果直接写在 ``results/``，当前数据目录则按实验类型放在
``results/run_by_rule`` 和 ``results/run_by_llm``，并把单次 run 放在
``存档点``。这里统一解析两种布局，避免续跑时因为工作目录或目录结构
不同而重复运行已有实验。
"""
import os
from typing import Iterable, Optional, Tuple

import pandas as pd


def resolve_result_dirs(kind: str) -> Tuple[str, str]:
    """返回 ``(summary_dir, run_dir)``。

    若检测到当前项目已经采用按实验分类的目录，则沿用该目录；否则保留
    旧版直接使用 ``results/`` 的行为。也可以用 ``RESULTS_ROOT`` 指定根目录。
    """
    if kind not in {"rule", "llm", "robustness"}:
        raise ValueError(f"unknown experiment kind: {kind}")

    root = os.getenv("RESULTS_ROOT", "results")
    nested = os.path.join(root, f"run_by_{kind}")
    if os.path.isdir(nested):
        summary_dir = nested
        archive_dir = os.path.join(nested, "存档点")
        run_dir = archive_dir if os.path.isdir(archive_dir) else summary_dir
    else:
        summary_dir = root
        run_dir = root

    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    return summary_dir, run_dir


def candidate_run_paths(
    summary_dir: str,
    run_dir: str,
    filename: str,
) -> Iterable[str]:
    """按当前布局和旧版布局返回不重复的候选路径。"""
    seen = set()
    for path in (os.path.join(run_dir, filename), os.path.join(summary_dir, filename)):
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized not in seen:
            seen.add(normalized)
            yield path


def read_completed_run(
    paths: Iterable[str],
    expected_step: int,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
) -> Tuple[Optional[str], Optional[dict]]:
    """读取一个已完成的单次 run。

    只有存在最终步、且（若提供）alpha/gamma 与当前实验条件一致时，才会
    返回结果。估算 xlsx、空文件和中途 checkpoint 不会被识别为完成结果。
    """
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if frame.empty or "step" not in frame.columns:
            continue

        last = frame.iloc[-1].to_dict()
        try:
            if int(float(last["step"])) != int(expected_step):
                continue
        except (TypeError, ValueError):
            continue

        def matches(name: str, expected: Optional[float]) -> bool:
            if expected is None or name not in last:
                return True
            try:
                return abs(float(last[name]) - float(expected)) <= 1e-12
            except (TypeError, ValueError):
                return False

        if not matches("alpha", alpha) or not matches("gamma", gamma):
            continue
        return path, last
    return None, None
