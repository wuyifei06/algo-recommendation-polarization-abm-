"""Versioned LLM ABM robustness experiment.

The formal design is fixed at the Rule ABM environment size and the requested
5 repeats per alpha-gamma condition.  Smoke runs are written to a separate
directory and never count as formal observations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from config import conf, validate_llm_config
from model.llm_robustness_v2 import LLMRobustnessModel
from utils.xlsx_writer import write_xlsx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = PROJECT_ROOT / "results" / "llm_robustness_v2"
SMOKE_ROOT = PROJECT_ROOT / "results" / "llm_robustness_v2_smoke"
COST_LOG = PROJECT_ROOT / "results" / "cost" / "cost_log.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _config(smoke: bool, trial: int = None) -> dict:
    if trial == 1:
        return {"mode": "trial1", "N": 10, "steps": 5, "repeats": 1, "alphas": [0.0], "gammas": [0.1]}
    if trial == 2:
        return {"mode": "trial2", "N": 30, "steps": 30, "repeats": 1, "alphas": [0.0], "gammas": [0.1]}
    if trial == 3:
        return {"mode": "trial3", "N": 30, "steps": 30, "repeats": 2, "alphas": [0.0], "gammas": [0.1]}
    if smoke:
        return {
            "mode": "smoke",
            "N": 3,
            "steps": 2,
            "repeats": 1,
            "alphas": [0.0],
            "gammas": [0.1],
        }
    return {
        "mode": "formal",
        "N": int(conf.LLM_N_AGENTS),
        "steps": int(conf.LLM_STEPS),
        "repeats": 5,
        "alphas": [float(value) for value in conf.ALPHAS],
        "gammas": [float(value) for value in conf.GAMMAS],
    }


def _iter_specs(config: dict) -> Iterable[dict]:
    run_id = 0
    for alpha in config["alphas"]:
        for gamma in config["gammas"]:
            for repeat in range(config["repeats"]):
                yield {
                    "run_id": run_id,
                    "repeat": repeat,
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "N": int(config["N"]),
                    "steps": int(config["steps"]),
                    # Keep v2 seeds separate from all legacy runs.
                    "seed": int(conf.BASE_SEED + 100000 + run_id),
                }
                run_id += 1


def _paths(root: Path, run_id: int) -> Dict[str, Path]:
    runs = root / "runs"
    return {
        "csv": runs / f"run_{run_id}.csv",
        "agents": runs / f"run_{run_id}_agents_final.csv",
        "checkpoint": runs / f"run_{run_id}.checkpoint.json",
        "calls": runs / f"run_{run_id}.calls.jsonl",
        "metadata": runs / f"run_{run_id}.metadata.json",
        "cost": COST_LOG,
    }


def _is_complete(paths: Dict[str, Path], spec: dict) -> bool:
    if not paths["csv"].exists() or not paths["metadata"].exists():
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if metadata.get("status") not in {"completed", "completed_with_fallback"}:
            return False
        if metadata.get("spec", {}).get("seed") != spec["seed"]:
            return False
        frame = pd.read_csv(paths["csv"])
        return (
            len(frame) == spec["steps"]
            and int(float(frame.iloc[-1]["step"])) == spec["steps"]
            and abs(float(frame.iloc[-1]["alpha"]) - spec["alpha"]) <= 1e-12
            and abs(float(frame.iloc[-1]["gamma"]) - spec["gamma"]) <= 1e-12
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def _save_agent_snapshot(model: LLMRobustnessModel, path: Path) -> None:
    rows = []
    for agent in model.agents:
        rows.append(
            {
                "agent_id": agent.id,
                "opinion": agent.opinion,
                "emotion": agent.emotion,
                "agent_type": agent.agent_type,
                "activity": agent.activity,
                "expression_probability": agent.expression_probability,
                "alpha": model.alpha,
                "gamma": model.gamma,
                "seed": model.seed,
                "step": model.current_step,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _load_or_create_model(spec: dict, paths: Dict[str, Path], max_llm_calls: int):
    if paths["checkpoint"].exists():
        try:
            payload = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
            matches = (
                int(payload.get("seed", -1)) == spec["seed"]
                and int(payload.get("N", -1)) == spec["N"]
                and int(payload.get("steps", -1)) == spec["steps"]
                and abs(float(payload.get("alpha")) - spec["alpha"]) <= 1e-12
                and abs(float(payload.get("gamma")) - spec["gamma"]) <= 1e-12
                and int(payload.get("current_step", spec["steps"])) < spec["steps"]
            )
            if matches:
                model = LLMRobustnessModel.load_checkpoint(
                    str(paths["checkpoint"]),
                    use_llm=True,
                    llm_archive_path=str(paths["calls"]),
                )
                model.run_id = spec["run_id"]
                model.llm_client.run_id = spec["run_id"]
                model.llm_client.cost_log_path = str(paths["cost"])
                return model, "resumed_checkpoint"
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass

    return (
        LLMRobustnessModel(
            seed=spec["seed"],
            N=spec["N"],
            steps=spec["steps"],
            alpha=spec["alpha"],
            gamma=spec["gamma"],
            use_llm=True,
            llm_archive_path=str(paths["calls"]),
            max_llm_calls=max_llm_calls,
            cost_log_path=str(paths["cost"]),
            run_id=spec["run_id"],
        ),
        "generated",
    )


def _run_one(spec: dict, root: Path, max_llm_calls: int) -> dict:
    paths = _paths(root, spec["run_id"])
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    if _is_complete(paths, spec):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        metadata["status"] = metadata.get("status", "completed")
        return metadata

    started = time.time()
    metadata = {"spec": spec, "started_at": _now(), "status": "running", "errors": []}
    _write_json(paths["metadata"], metadata)
    try:
        model, source = _load_or_create_model(spec, paths, max_llm_calls)

        def checkpoint_callback(model_obj):
            model_obj.save_checkpoint(str(paths["checkpoint"]))

        model.checkpoint_callback = checkpoint_callback
        model.run()
        if len(model.metrics_time_series) != spec["steps"]:
            raise RuntimeError(
                f"incomplete time series: {len(model.metrics_time_series)}/{spec['steps']} steps"
            )
        model.save_results(str(paths["csv"]))
        _save_agent_snapshot(model, paths["agents"])
        quality = model.llm_client.quality_summary()
        series = pd.read_csv(paths["csv"])
        series["run_id"] = spec["run_id"]
        series["repeat"] = spec["repeat"]
        series["llm_calls"] = quality.get("api_calls", 0)
        series["input_tokens"] = quality.get("input_tokens", 0)
        series["output_tokens"] = quality.get("output_tokens", 0)
        series["estimated_cost"] = quality.get("estimated_cost", 0.0)
        series.to_csv(paths["csv"], index=False)
        if quality["valid_responses"] == 0:
            status = "failed_no_valid_llm"
            error = "No valid real-model response was observed; fallback-only run is excluded from formal summaries."
            metadata["errors"].append(error)
        elif quality["fallbacks"]:
            status = "completed_with_fallback"
            error = None
        else:
            status = "completed"
            error = None

        metadata.update(
            {
                "status": status,
                "source": source,
                "finished_at": _now(),
                "runtime_seconds": round(time.time() - started, 3),
                "quality": quality,
                "error": error,
                "final_metrics": model.metrics_time_series[-1],
            }
        )
        model.save_checkpoint(str(paths["checkpoint"]))
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "finished_at": _now(),
                "runtime_seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        metadata["errors"].append(metadata["error"])
    _write_json(paths["metadata"], metadata)
    return metadata


def _summary_row(metadata: dict) -> dict:
    row = dict(metadata.get("final_metrics") or {})
    row.update(metadata.get("spec") or {})
    row["status"] = metadata.get("status")
    row["runtime_seconds"] = metadata.get("runtime_seconds")
    row["error"] = metadata.get("error")
    quality = metadata.get("quality") or {}
    row["llm_calls"] = quality.get("api_calls", 0)
    row["input_tokens"] = quality.get("input_tokens", 0)
    row["output_tokens"] = quality.get("output_tokens", 0)
    row["estimated_cost"] = quality.get("estimated_cost", 0.0)
    for key, value in quality.items():
        row[f"llm_{key}"] = value
    return row


def _condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["SR", "EM", "PI", "VPI", "ER", "HE", "EE"]
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=["alpha", "gamma", "metric", "Mean", "SD", "SE", "CI95", "n"])
    for (alpha, gamma), group in frame.groupby(["alpha", "gamma"]):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(values)
            sd = float(values.std(ddof=1)) if n > 1 else 0.0 if n == 1 else float("nan")
            se = sd / (n ** 0.5) if n else float("nan")
            rows.append({"alpha": alpha, "gamma": gamma, "metric": metric, "Mean": float(values.mean()) if n else float("nan"), "SD": sd, "SE": se, "CI95": 1.96 * se if n else float("nan"), "n": n})
    return pd.DataFrame(rows)


def run_experiment(
    smoke: bool = False,
    output_dir: str = None,
    max_llm_calls: int = None,
    plan_only: bool = False,
    trial: int = None,
) -> Path:
    if not plan_only:
        validate_llm_config()
    config = _config(smoke, trial)
    if output_dir:
        root = Path(output_dir)
    elif trial:
        root = PROJECT_ROOT / "results" / f"llm_robustness_v2_trial{trial}"
    else:
        root = Path(output_dir) if output_dir else (SMOKE_ROOT if smoke else FORMAL_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    if max_llm_calls is None:
        if smoke:
            max_llm_calls = 100
        else:
            max_llm_calls = int(os.getenv("LLM_V2_MAX_CALLS", "0") or 0)

    specs = list(_iter_specs(config))
    manifest = {
        "version": "llm_robustness_v2",
        "created_at": _now(),
        "project_root": str(PROJECT_ROOT),
        "experiment": config,
        "formal_target": {
            "N": int(conf.LLM_N_AGENTS),
            "steps": int(conf.LLM_STEPS),
            "repeats": 5,
            "alphas": [float(value) for value in conf.ALPHAS],
            "gammas": [float(value) for value in conf.GAMMAS],
            "top_k": int(conf.TOP_K),
            "base_seed": int(conf.BASE_SEED),
        },
        "max_llm_calls": int(max_llm_calls),
        "planned_runs": len(specs),
        "runs": [],
    }
    _write_json(root / "manifest.json", manifest)

    if plan_only:
        manifest["status"] = "planned_only"
        manifest["finished_at"] = _now()
        _write_json(root / "manifest.json", manifest)
        print(
            f"[LLM robustness v2] plan-only written: {root} "
            f"planned_runs={len(specs)} expected_decisions={len(specs) * config['N'] * config['steps']}",
            flush=True,
        )
        return root

    print(
        f"[LLM robustness v2] mode={config['mode']} N={config['N']} steps={config['steps']} "
        f"repeats={config['repeats']} conditions={len(config['alphas']) * len(config['gammas'])} "
        f"planned_runs={len(specs)} expected_decisions={len(specs) * config['N'] * config['steps']}",
        flush=True,
    )
    metadata_rows = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[LLM robustness v2] run {index}/{len(specs)} "
            f"alpha={spec['alpha']} gamma={spec['gamma']} repeat={spec['repeat']}",
            flush=True,
        )
        metadata = _run_one(spec, root, int(max_llm_calls))
        metadata_rows.append(metadata)
        _write_json(root / "manifest.json", {**manifest, "runs": metadata_rows})
        pd.DataFrame([_summary_row(row) for row in metadata_rows]).to_csv(
            root / "summary_all_runs.csv", index=False
        )

    usable = [
        row for row in metadata_rows if row.get("status") in {"completed", "completed_with_fallback"}
    ]
    summary_frame = pd.DataFrame([_summary_row(row) for row in usable])
    summary_frame.to_csv(root / "summary.csv", index=False)
    all_frame = pd.DataFrame([_summary_row(row) for row in metadata_rows])
    condition_frame = _condition_summary(summary_frame)
    write_xlsx(root / "llm_all_runs_summary.xlsx", {"all_runs": all_frame})
    write_xlsx(root / "condition_summary.xlsx", {"conditions": condition_frame})
    manifest["runs"] = metadata_rows
    manifest["completed_runs"] = len(usable)
    manifest["failed_runs"] = [row.get("spec", {}).get("run_id") for row in metadata_rows if row.get("status") not in {"completed", "completed_with_fallback"}]
    manifest["finished_at"] = _now()
    _write_json(root / "manifest.json", manifest)
    return root


def parser_test() -> None:
    from llm.robust_client_v2 import RobustLLMClient

    cases = [
        {"express": 0, "stance_delta": -0.025, "emotion_delta": 0.05},
        {"choices": [{"message": {"content": '{"express": 1, "stance_delta": 0.03, "emotion_delta": -0.04}'}}]},
        {"express": True, "stance_delta": 0.0, "emotion": 0.55},
    ]
    for case in cases:
        action, quality = RobustLLMClient.normalize_action(case, 0.5)
        print(json.dumps({"action": action, "quality": quality}, ensure_ascii=False))
    print("parser_test=passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run only the separate 3-agent/2-step smoke grid")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-llm-calls", type=int, default=None)
    parser.add_argument("--parser-test", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="write the formal grid manifest without making LLM calls")
    parser.add_argument("--trial", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.parser_test:
        parser_test()
        return
    run_experiment(args.smoke, args.output_dir, args.max_llm_calls, args.plan_only, args.trial)


if __name__ == "__main__":
    main()
