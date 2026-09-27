"""LLM 实验驱动与可恢复的缺失 run 补全。

恢复原则：

* 只把达到目标步数的 CSV 视为真实完成结果；估算 xlsx 永远不会被读取。
* 优先继续同一 ``run_id`` 的 checkpoint。
* 旧版本曾因续跑游标错位产生“文件名与条件不一致”的 CSV。对这类完整
  观测，按 ``(alpha, gamma, repeat)`` 复用，并在 manifest 中记录来源。
* 每个 run 使用独立的 LLM 调用存档，网络中断后再次执行即可继续。
"""
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import conf
from model.model import SimulationModel
from utils.results import read_completed_run, resolve_result_dirs


_RUN_FILE_RE = re.compile(r"^llm_run_(\d+)\.csv$")


def _checkpoint_paths(summary_dir: str, run_dir: str, run_id: int) -> List[str]:
    filename = f"llm_run_{run_id}.checkpoint.json"
    paths = [os.path.join(run_dir, filename), os.path.join(summary_dir, filename)]
    result = []
    seen = set()
    for path in paths:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _archive_path(run_dir: str, run_id: int) -> str:
    return os.path.join(run_dir, f"llm_run_{run_id}.calls.jsonl")


def _remove_estimated_artifacts(run_dir: str) -> None:
    """删除用户明确标记为估算的两个旧结果文件。"""
    for filename in ("llm_run_6_estimated.xlsx", "llm_run_11_estimated.xlsx"):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"[LLM 实验] 已删除估算文件：{path}", flush=True)


def _all_run_csv_paths(summary_dir: str, run_dir: str) -> Iterable[Tuple[int, str]]:
    """枚举两种历史布局中的单次 LLM CSV，去除重复路径。"""
    seen = set()
    for directory in (run_dir, summary_dir):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            match = _RUN_FILE_RE.match(name)
            if not match:
                continue
            path = os.path.join(directory, name)
            key = os.path.normcase(os.path.abspath(path))
            if key in seen:
                continue
            seen.add(key)
            yield int(match.group(1)), path


def _float_equal(left, right) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _checkpoint_matches(path: str, spec: dict) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        current_step = int(payload.get("current_step", -1))
        return (
            int(payload.get("N", -1)) == int(spec["N"])
            and int(payload.get("steps", -1)) == int(spec["steps"])
            and int(payload.get("seed", -1)) == int(spec["seed"])
            and _float_equal(payload.get("alpha"), spec["alpha"])
            and _float_equal(payload.get("gamma"), spec["gamma"])
            and current_step < int(spec["steps"])
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _result_for_summary(row: dict, run_id: int, alpha: float, gamma: float) -> dict:
    result = dict(row)
    result.update({"run_id": run_id, "alpha": alpha, "gamma": gamma})
    return result


def _write_resume_state(
    path: str,
    results_by_id: Dict[int, dict],
    current_run: Optional[dict] = None,
) -> None:
    payload = {
        "version": 2,
        "completed_run_ids": sorted(results_by_id),
        "completed_runs": len(results_by_id),
        "results": [results_by_id[key] for key in sorted(results_by_id)],
    }
    if current_run is not None:
        payload["current_run"] = current_run
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _scan_observed_runs(
    summary_dir: str,
    run_dir: str,
    steps: int,
) -> Dict[Tuple[float, float], List[dict]]:
    """收集完整的历史观测，按实验条件建立可复用候选列表。"""
    observed: Dict[Tuple[float, float], List[dict]] = {}
    for source_run_id, path in _all_run_csv_paths(summary_dir, run_dir):
        _, row = read_completed_run([path], expected_step=steps)
        if row is None or "alpha" not in row or "gamma" not in row:
            continue
        try:
            key = (float(row["alpha"]), float(row["gamma"]))
        except (TypeError, ValueError):
            continue
        observed.setdefault(key, []).append(
            {"source_run_id": source_run_id, "path": path, "row": row}
        )
    for candidates in observed.values():
        candidates.sort(key=lambda item: item["source_run_id"])
    return observed


def _requested_specs(run_ids, N: int, steps: int) -> List[dict]:
    """生成本次要处理的 run 规格。

    不传 run_ids 时保留完整的 5×2 网格。指定旧版 ``run_id=6`` 时，
    使用存档中第六次实验的原始条件 alpha=1.0、gamma=1.0、seed=12351；
    第 11 次不属于当前修复目标，因此不会被加入任务列表。
    """
    full_specs = []
    run_id = 0
    for alpha in conf.LLM_ALPHAS:
        for gamma in conf.LLM_GAMMAS:
            for repeat in range(conf.LLM_REPEATS):
                full_specs.append(
                    {
                        "run_id": run_id,
                        "repeat": repeat,
                        "alpha": alpha,
                        "gamma": gamma,
                        "N": N,
                        "steps": steps,
                        "seed": conf.BASE_SEED + run_id,
                    }
                )
                run_id += 1

    if run_ids is None:
        return full_specs

    requested = sorted({int(value) for value in run_ids})
    selected = []
    for run_id in requested:
        if run_id == 6:
            selected.append(
                {
                    "run_id": 6,
                    "repeat": 0,
                    "alpha": 1.0,
                    "gamma": 1.0,
                    "N": N,
                    "steps": steps,
                    "seed": conf.BASE_SEED + 6,
                }
            )
        elif 0 <= run_id < len(full_specs):
            selected.append(full_specs[run_id])
        else:
            print(f"[LLM 实验] 忽略不在当前实验范围内的 run_id={run_id}", flush=True)
    return selected


def run_llm_experiment(run_ids=None):
    N = conf.LLM_N_AGENTS
    steps = conf.LLM_STEPS
    repeats = conf.LLM_REPEATS
    alphas = conf.LLM_ALPHAS
    gammas = conf.LLM_GAMMAS

    summary_dir, run_dir = resolve_result_dirs("llm")
    _remove_estimated_artifacts(run_dir)
    resume_path = os.path.join(summary_dir, "llm_resume.json")
    manifest_path = os.path.join(summary_dir, "llm_data_manifest.json")
    results_by_id: Dict[int, dict] = {}
    used_observed_paths = set()
    observed = _scan_observed_runs(summary_dir, run_dir, steps)
    specs = _requested_specs(run_ids, N, steps)
    total_runs = len(specs)
    if not specs:
        print("[LLM 实验] 没有需要处理的有效 run，已结束。", flush=True)
        return

    print(
        f"[LLM 实验] 真实大模型模式：N={N}, steps={steps}, repeats={repeats}, "
        f"alpha={alphas}, gamma={gammas}, target_runs={[spec['run_id'] for spec in specs]}",
        flush=True,
    )

    manifest = {
        "version": 2,
        "experiment": {"N": N, "steps": steps, "repeats": repeats, "alphas": alphas, "gammas": gammas},
        "runs": [],
    }

    for spec in specs:
                run_id = spec["run_id"]
                alpha = spec["alpha"]
                gamma = spec["gamma"]
                repeat = spec["repeat"]
                filename = f"llm_run_{run_id}.csv"
                out_path = os.path.join(run_dir, filename)
                status = None
                source_run_id = None

                # 1. 精确文件优先：只有条件和目标步数都相符才算完成。
                exact_path, exact_row = read_completed_run(
                    [os.path.join(run_dir, filename), os.path.join(summary_dir, filename)],
                    expected_step=steps,
                    alpha=alpha,
                    gamma=gamma,
                )
                if exact_row is not None:
                    results_by_id[run_id] = _result_for_summary(exact_row, run_id, alpha, gamma)
                    used_observed_paths.add(os.path.normcase(os.path.abspath(exact_path)))
                    status = "observed_exact"
                    source_run_id = run_id
                else:
                    # 2. 同一 run 的 checkpoint 优先于其它条件相同的旧 run。
                    checkpoint_path = next(
                        (
                            path
                            for path in _checkpoint_paths(summary_dir, run_dir, run_id)
                            if _checkpoint_matches(path, spec)
                        ),
                        None,
                    )
                    if checkpoint_path is not None:
                        status = "resumed_checkpoint"
                    else:
                        # 3. 旧版本可能只错位了文件名；复用完整观测，但不
                        # 把估算 xlsx 或同一文件重复分配给多个条件。
                        key = (float(alpha), float(gamma))
                        candidate = next(
                            (
                                item
                                for item in observed.get(key, [])
                                if os.path.normcase(os.path.abspath(item["path"])) not in used_observed_paths
                            ),
                            None,
                        )
                        if candidate is not None:
                            results_by_id[run_id] = _result_for_summary(candidate["row"], run_id, alpha, gamma)
                            used_observed_paths.add(os.path.normcase(os.path.abspath(candidate["path"])))
                            status = "observed_reused"
                            source_run_id = candidate["source_run_id"]

                if status in {"observed_exact", "observed_reused"}:
                    manifest["runs"].append({**spec, "status": status, "source_run_id": source_run_id})
                    run_id += 1
                    continue

                archive_path = _archive_path(run_dir, run_id)
                if status == "resumed_checkpoint":
                    model = SimulationModel.load_checkpoint(
                        checkpoint_path,
                        use_llm=True,
                        llm_archive_path=archive_path,
                    )
                else:
                    model = SimulationModel(
                        seed=spec["seed"],
                        N=N,
                        steps=steps,
                        alpha=alpha,
                        gamma=gamma,
                        use_llm=True,
                        llm_archive_path=archive_path,
                    )
                    checkpoint_path = os.path.join(run_dir, f"llm_run_{run_id}.checkpoint.json")

                def save_run_checkpoint(model_obj):
                    model_obj.save_checkpoint(checkpoint_path)
                    _write_resume_state(resume_path, results_by_id, current_run=spec)

                model.checkpoint_callback = save_run_checkpoint
                try:
                    model.run()
                except Exception as exc:
                    # checkpoint_callback 已在最近一个完整 step 保存；异常
                    # 直接抛出，用户再次运行同一入口即可继续，不再等待交互输入。
                    save_run_checkpoint(model)
                    print(
                        f"[LLM 断点] run_id={run_id} 在 {type(exc).__name__} 处中断，"
                        f"已保存到 {checkpoint_path}，请修复连接后重新运行。",
                        flush=True,
                    )
                    raise

                if not model.metrics_time_series:
                    raise RuntimeError(f"LLM run {run_id} finished without metrics")
                final = dict(model.metrics_time_series[-1])
                final.update({"run_id": run_id, "alpha": alpha, "gamma": gamma, "llm_calls": model.llm_call_count})
                results_by_id[run_id] = final
                pd.DataFrame([final]).to_csv(out_path, index=False)
                manifest["runs"].append(
                    {**spec, "status": status or "generated", "source_run_id": run_id, "llm_calls": model.llm_call_count}
                )
                _write_resume_state(resume_path, results_by_id)
                print(
                    f"[LLM 实验] 已完成 {len(results_by_id)}/{total_runs} 次，"
                    f"剩余 {total_runs - len(results_by_id)} 次，LLM调用次数={model.llm_call_count}",
                    flush=True,
                )
                run_id += 1

    df = pd.DataFrame([results_by_id[key] for key in sorted(results_by_id)])
    df.to_csv(os.path.join(summary_dir, "llm_summary.csv"), index=False)
    manifest["runs"].sort(key=lambda item: item["run_id"])
    manifest["completed_runs"] = len(results_by_id)
    target_ids = {spec["run_id"] for spec in specs}
    manifest["missing_runs"] = sorted(target_ids - set(results_by_id))
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    _write_resume_state(resume_path, results_by_id)
