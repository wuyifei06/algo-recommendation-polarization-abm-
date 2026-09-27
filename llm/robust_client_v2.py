"""Robust real-model client for the LLM robustness experiment.

This client is intentionally separate from the legacy client.  It records one
structured quality record per logical Agent decision, accepts common provider
response envelopes, and uses a deterministic caller-supplied fallback for
invalid, missing, out-of-range, or failed responses.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests

from config import load_env
from llm.prompt import build_batch_prompt, build_prompt
from utils.cost_guard import CostGuard


class ResponseValidationError(ValueError):
    """Raised when a provider response cannot be converted to one action."""


def _retry_on_permission_error(fn: Callable[[], Any], attempts: int = 6,
                               base_delay: float = 0.15) -> Any:
    """执行文件写入，遇到瞬时占用自动退避重试。

    背景（2026-09-21）：``llm_gradient_noleak`` 跑到 run 44 时因
    ``PermissionError: [Errno 13]`` 写 ``calls/llm_robust_run_44.jsonl`` 失败，
    导致整个 4 小时实验进程崩溃。
    进程内互斥已由 ``self._archive_lock`` 保证，所以该错误只能来自**进程外**
    的短暂占用（实时杀毒扫描、桌面同步盘、索引服务），通常 <1s 即释放。
    这里最多重试 6 次（退避 0.15→0.90s，合计约 3s）。

    无害性：只影响"落盘记录"这一步，不改变任何 agent 决策、调用次数、
    也不改变写入内容；全部重试仍失败时，照旧抛出原异常。
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except PermissionError as exc:  # pragma: no cover - 依赖运行环境
            last_exc = exc
            if attempt < attempts:
                time.sleep(base_delay * attempt)
    assert last_exc is not None
    raise last_exc


class RobustLLMClient:
    def __init__(self, archive_path: str, max_calls: int = 0, cost_log_path: str = None, run_id: int = None):
        env = load_env()
        self.provider = (env.get("LLM_PROVIDER") or "dashscope").strip().lower()
        self.model_name = (
            env.get("LLM_MODEL")
            or env.get("LLM_MODEL_NAME")
            or ""
        ).strip()
        self.api_key = (env.get("LLM_API_KEY") or "").strip()
        self.endpoint = self._normalize_endpoint(env.get("LLM_API_ENDPOINT") or "")
        self.temperature = float(env.get("LLM_TEMPERATURE", 0.7))
        self.timeout = int(env.get("LLM_TIMEOUT_SECONDS", 60))
        self.max_retries = max(1, int(env.get("LLM_MAX_RETRIES", 3)))
        self.max_calls = max(0, int(max_calls or 0))

        if not self.api_key or not self.endpoint or not self.model_name:
            raise ValueError(
                "LLM robustness v2 requires LLM_API_KEY, LLM_API_ENDPOINT and LLM_MODEL."
            )

        self.archive_path = archive_path
        os.makedirs(os.path.dirname(self.archive_path) or ".", exist_ok=True)
        self._archive_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._call_counter = self._read_last_call_no(self.archive_path)
        self._quality = {
            "logical_calls": 0,
            "api_calls": 0,
            "batch_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost": 0.0,
            "valid_responses": 0,
            "fallbacks": 0,
            "parse_failures": 0,
            "provider_errors": 0,
            "out_of_range": 0,
            "missing_fields": 0,
            "legacy_emotion": 0,
            "provider_attempts": 0,
            "reasoning_starved": 0,
        }
        self.run_id = run_id
        self.cost_log_path = cost_log_path
        if self.cost_log_path:
            os.makedirs(os.path.dirname(self.cost_log_path) or ".", exist_ok=True)
        self.cost_guard = CostGuard(max_calls=max_calls, max_cost=300.0, warning_cost=250.0, stop_cost=280.0)
        self._api_call_counter = 0
        self._starvation_warned = False

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        base = endpoint.strip().rstrip("/")
        if not base:
            return ""
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1") or base.endswith("/api/v1"):
            return f"{base}/chat/completions"
        return base

    def _thinking_off_payload(self) -> Dict[str, Any]:
        """返回 provider 原生的「关闭思考」字段。

        背景（2026-09-20 实测，见 deepseek-bug-check/probe_batch_content.py）：
        deepseek 的推理型模型**完全不认** ``enable_thinking`` /
        ``thinking_budget`` / ``reasoning_effort``。此时 ``max_tokens`` 会被
        ``reasoning_tokens`` 吃光（1024 -> 1024、4096 -> 4096、8192 -> 8192 全部
        如此），``content`` 恒为空、``finish_reason="length"``，整批决策解析失败
        并全部走兜底，最终抛
        ``RuntimeError: 没有任何有效模型响应（全部走了兜底）``。

        实测有效的是顶层 ``thinking={"type": "disabled"}``：completion_tokens 从
        4096 降到约 340，content 正常返回，10/10 解析通过、3~8 种不同
        stance_delta（远高于退化守卫 distinct_stance_delta<=1 的阈值）。

        默认只对 deepseek 打开，避免给 dashscope / Qwen 等 provider 发未知字段；
        可用 ``LLM_DISABLE_THINKING=0/1`` 覆盖。
        """
        flag = os.getenv("LLM_DISABLE_THINKING")
        if flag is None:
            enabled = self.provider == "deepseek"
        else:
            enabled = flag.strip().lower() in ("1", "true", "yes", "on")
        return {"thinking": {"type": "disabled"}} if enabled else {}

    def _warn_if_thinking_starved(self, raw: Any) -> None:
        """识别「思考吃光 max_tokens」这一失败模式，首次出现时大声告警。

        这是本 bug 最难查的地方：HTTP 200、没有任何异常，只是 content 为空，
        错误要等到整轮跑完才以「全部走了兜底」的形式暴露。这里把它变成
        第一时间可见、且带修复指引的提示。
        """
        if self._starvation_warned or not isinstance(raw, dict):
            return
        try:
            choice = (raw.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = raw.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            completion = int(usage.get("completion_tokens") or 0)
            reasoning = int(details.get("reasoning_tokens") or 0)
            content = message.get("content") or ""
        except (TypeError, ValueError, AttributeError, IndexError):
            return
        starved = (
            not content
            and choice.get("finish_reason") == "length"
            and reasoning > 0
            and reasoning >= completion
        )
        if not starved:
            return
        self._starvation_warned = True
        self._quality["reasoning_starved"] += 1
        sys.stderr.write(
            "\n[RobustLLMClient] ⚠️ 检测到「思考吃光 max_tokens」："
            f"completion_tokens={completion}、reasoning_tokens={reasoning}、content 为空。\n"
            "  该模式下所有决策都会走兜底，整轮结果不可用。\n"
            "  修复：在 .env 中设置 LLM_DISABLE_THINKING=1（或改用非推理模型），"
            "然后重新运行。\n\n"
        )
        sys.stderr.flush()

    @staticmethod
    def _read_last_call_no(path: str) -> int:
        if not os.path.exists(path):
            return 0
        latest = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        latest = max(latest, int(json.loads(line).get("call_no", 0) or 0))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
        except OSError:
            return 0
        return latest

    @property
    def call_count(self) -> int:
        return self._api_call_counter

    def quality_summary(self) -> Dict[str, Any]:
        result = dict(self._quality)
        calls = max(1, int(result["logical_calls"]))
        result["fallback_rate"] = result["fallbacks"] / calls
        result["parse_failure_rate"] = result["parse_failures"] / calls
        result["provider_error_rate"] = result["provider_errors"] / calls
        result["valid_response_rate"] = result["valid_responses"] / calls
        return result

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_rate = float(os.getenv("LLM_INPUT_COST_PER_1K", "0.0008"))
        output_rate = float(os.getenv("LLM_OUTPUT_COST_PER_1K", "0.002"))
        return input_tokens / 1000.0 * input_rate + output_tokens / 1000.0 * output_rate

    def _write_cost_log(self, run_id, step, agent_id, model, input_tokens, output_tokens, estimated_cost):
        if not self.cost_log_path:
            return
        fields = ["run_id", "step", "agent_id", "model", "input_tokens", "output_tokens", "estimated_cost"]

        def _do_write():
            with self._archive_lock:
                exists = os.path.exists(self.cost_log_path)
                with open(self.cost_log_path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fields)
                    if not exists:
                        writer.writeheader()
                    writer.writerow({
                        "run_id": run_id,
                        "step": step,
                        "agent_id": agent_id,
                        "model": model,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "estimated_cost": f"{float(estimated_cost):.8f}",
                    })

        _retry_on_permission_error(_do_write)

    def _write_archive(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"

        def _do_write():
            with self._archive_lock:
                with open(self.archive_path, "a", encoding="utf-8") as fh:
                    fh.write(line)

        _retry_on_permission_error(_do_write)

    def generate_agent_action(
        self,
        state: Dict[str, Any],
        feed: Dict[str, Any],
        fallback: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the real model once and return a bounded action or fallback."""
        with self._counter_lock:
            self._call_counter += 1
            self._api_call_counter += 1
            call_no = self._call_counter
            self._quality["logical_calls"] += 1

        if self.max_calls and self._quality["logical_calls"] > self.max_calls:
            self._quality["provider_errors"] += 1
            return self._fallback_result(
                call_no, state, feed, fallback, "call_budget_exceeded"
            )

        raw = None
        attempts = 0
        error = None
        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            try:
                raw = self._request(state, feed)
                self._quality["provider_attempts"] += attempt
                break
            except Exception as exc:  # network/provider errors are logged and fall back
                error = f"{type(exc).__name__}: {exc}"
                self._quality["provider_attempts"] += 1
                if attempt < self.max_retries:
                    time.sleep(min(1.5 * attempt, 3.0))

        if raw is None:
            self._quality["provider_errors"] += 1
            return self._fallback_result(call_no, state, feed, fallback, error or "provider_error", attempts)

        try:
            action, quality = self.normalize_action(raw, float(state.get("emotion", 0.0)))
            self._quality["valid_responses"] += 1
            for key in ("out_of_range", "legacy_emotion"):
                self._quality[key] += int(quality.get(key, 0))
            self._write_archive(
                {
                    "call_no": call_no,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "state": state,
                    "feed": feed,
                    "result": action,
                    "quality": {**quality, "attempts": attempts, "fallback": False},
                }
            )
            return action
        except ResponseValidationError as exc:
            reason = str(exc)
            if "missing" in reason:
                self._quality["missing_fields"] += 1
            if "out_of_range" in reason:
                self._quality["out_of_range"] += 1
            self._quality["parse_failures"] += 1
            return self._fallback_result(call_no, state, feed, fallback, reason, attempts)

    def generate_batch_actions(self, batch: list, run_id: int, step: int, fallbacks: list) -> list:
        """Make one API request for up to ten Agents and return one action each."""
        count = len(batch)
        if count == 0:
            return []
        prompt = build_batch_prompt(batch)
        input_tokens = max(1, len(prompt) // 4)
        expected_output_tokens = max(32, count * 18)
        estimated_cost = self._estimate_cost(input_tokens, expected_output_tokens)
        self._quality["logical_calls"] += count
        try:
            allowed = self.cost_guard.allow_call(estimated_cost)
        except RuntimeError as exc:
            allowed = False
            reason = str(exc)
        else:
            reason = "cost_guard_stop" if not allowed else ""
        if not allowed:
            self._quality["provider_errors"] += count
            self._quality["fallbacks"] += count
            actions = [fallback() for fallback in fallbacks]
            for item, action in zip(batch, actions):
                self._write_cost_log(run_id, step, item["agent_id"], self.model_name, input_tokens // count, 0, 0.0)
            self._write_archive({"call_no": self._call_counter, "batch": True, "run_id": run_id, "step": step, "quality": {"fallback": True, "fallback_reason": reason}, "result": actions})
            return actions

        with self._counter_lock:
            self._call_counter += 1
            self._api_call_counter += 1
            call_no = self._call_counter
        self._quality["api_calls"] += 1
        self._quality["batch_requests"] += 1
        raw = None
        attempts = 0
        error = None
        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            try:
                raw = self._request_batch(prompt)
                self._quality["provider_attempts"] += attempt
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._quality["provider_attempts"] += 1
                if attempt < self.max_retries:
                    time.sleep(min(1.5 * attempt, 3.0))
        output_tokens = self._response_tokens(raw, expected_output_tokens)
        actual_cost = self._estimate_cost(input_tokens, output_tokens)
        self._quality["input_tokens"] += input_tokens
        self._quality["output_tokens"] += output_tokens
        self._quality["estimated_cost"] += actual_cost
        per_input = max(1, input_tokens // count)
        per_output = max(0, output_tokens // count)
        per_cost = actual_cost / count
        for item in batch:
            self._write_cost_log(run_id, step, item["agent_id"], self.model_name, per_input, per_output, per_cost)

        if raw is None:
            self._quality["provider_errors"] += count
            self._quality["fallbacks"] += count
            actions = [fallback() for fallback in fallbacks]
            self._write_archive({"call_no": call_no, "batch": True, "run_id": run_id, "step": step, "quality": {"fallback": True, "fallback_reason": error or "provider_error", "attempts": attempts}, "result": actions})
            return actions

        candidates = self._find_batch_candidates(raw)
        actions = []
        for index, (item, fallback) in enumerate(zip(batch, fallbacks)):
            candidate = candidates[index] if index < len(candidates) else None
            try:
                action, quality = self.normalize_action(candidate, float(item["state"].get("emotion", 0.0)))
                self._quality["valid_responses"] += 1
                self._quality["out_of_range"] += int(quality.get("out_of_range", 0))
                self._quality["legacy_emotion"] += int(quality.get("legacy_emotion", 0))
            except ResponseValidationError as exc:
                reason = str(exc)
                self._quality["parse_failures"] += 1
                if "missing" in reason:
                    self._quality["missing_fields"] += 1
                if "out_of_range" in reason:
                    self._quality["out_of_range"] += 1
                self._quality["fallbacks"] += 1
                action = fallback()
            actions.append(action)
        self._write_archive({"call_no": call_no, "batch": True, "run_id": run_id, "step": step, "quality": {"fallback": self._quality["fallbacks"] > 0, "attempts": attempts, "response_count": len(candidates)}, "raw_preview": json.dumps(raw, ensure_ascii=False, default=str)[:6000], "result": actions})
        return actions

    def _request_batch(self, prompt: str) -> Any:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "x-api-key": self.api_key}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": min(max(self.temperature, 0.0), 1.0),
            "max_tokens": 1024,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False, "thinking_budget": 0, "reasoning_effort": "none"},
            # ⚠️ 关键修复（2026-09-20）：deepseek 推理型模型不认上面的
            # enable_thinking / extra_body，必须靠 provider 原生字段关掉思考，
            # 否则 max_tokens 会被 reasoning_tokens 吃光、content 恒为空。
            **self._thinking_off_payload(),
        }
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json()
        self._warn_if_thinking_starved(raw)
        return raw

    @staticmethod
    def _response_tokens(raw: Any, default: int) -> int:
        if isinstance(raw, dict):
            usage = raw.get("usage") or {}
            for key in ("completion_tokens", "output_tokens"):
                if key in usage:
                    try:
                        return int(usage[key])
                    except (TypeError, ValueError):
                        pass
        return default

    @classmethod
    def _find_batch_candidates(cls, data: Any) -> list:
        if isinstance(data, list):
            if all(isinstance(item, dict) and any(key in item for key in ("express", "stance_delta", "emotion_delta", "stance")) for item in data):
                return data
            for item in data:
                found = cls._find_batch_candidates(item)
                if found:
                    return found
            return []
        if isinstance(data, dict):
            for key in ("decisions", "actions", "responses", "results", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
            for key in ("content", "text", "output_text", "output", "result", "message", "choices", "response", "reasoning_content"):
                value = data.get(key)
                if isinstance(value, str):
                    parsed = cls._parse_text(value)
                    if isinstance(parsed, list):
                        return parsed
                elif isinstance(value, (dict, list)):
                    found = cls._find_batch_candidates(value)
                    if found:
                        return found
            if any(key in data for key in ("express", "stance_delta", "emotion_delta", "stance")):
                return [data]
        return []

    def _fallback_result(
        self,
        call_no: int,
        state: Dict[str, Any],
        feed: Dict[str, Any],
        fallback: Callable[[], Dict[str, Any]],
        reason: str,
        attempts: int = 0,
    ) -> Dict[str, Any]:
        action = fallback()
        self._quality["fallbacks"] += 1
        self._write_archive(
            {
                "call_no": call_no,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "state": state,
                "feed": feed,
                "result": action,
                "quality": {"fallback": True, "fallback_reason": reason, "attempts": attempts},
                "error": reason,
            }
        )
        return action

    def _request(self, state: Dict[str, Any], feed: Dict[str, Any]) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": build_prompt(state, feed.get("items", []))}],
            "temperature": min(max(self.temperature, 0.0), 1.0),
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False, "reasoning_effort": "none"},
            **self._thinking_off_payload(),
        }
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json()
        self._warn_if_thinking_starved(raw)
        return raw

    @classmethod
    def normalize_action(cls, data: Any, current_emotion: float) -> Tuple[Dict[str, Any], Dict[str, int]]:
        candidate = cls._find_candidate(data)
        if candidate is None:
            raise ResponseValidationError("invalid_or_missing_json")

        required = ["express"]
        if "stance_delta" not in candidate and "stance" not in candidate:
            required.append("stance_delta")
        missing = [key for key in required if key not in candidate]
        if "emotion_delta" not in candidate and "emotion" not in candidate:
            missing.append("emotion_delta")
        if missing:
            raise ResponseValidationError("missing_fields:" + ",".join(missing))

        express = cls._parse_express(candidate["express"])
        legacy_stance = 0
        if "stance_delta" in candidate:
            stance_delta = cls._parse_number(candidate["stance_delta"], "stance_delta")
        else:
            stance_delta = cls._parse_number(candidate["stance"], "stance")
            legacy_stance = 1
        legacy = 0
        if "emotion_delta" in candidate:
            delta = cls._parse_number(candidate["emotion_delta"], "emotion_delta")
        else:
            emotion = cls._parse_number(candidate["emotion"], "emotion")
            delta = emotion - current_emotion
            legacy = 1

        if not (-0.05 <= stance_delta <= 0.05):
            raise ResponseValidationError("out_of_range:stance_delta")
        if not (-0.1 <= delta <= 0.1):
            raise ResponseValidationError("out_of_range:emotion_delta")
        action = {"express": int(express), "stance_delta": float(stance_delta), "emotion_delta": float(delta)}
        if legacy_stance:
            action["stance"] = float(candidate["stance"])
        return (action, {"out_of_range": 0, "legacy_emotion": legacy, "legacy_stance": legacy_stance})

    @classmethod
    def _parse_express(cls, value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        if isinstance(value, str) and value.strip().lower() in {"0", "1", "true", "false"}:
            return int(value.strip().lower() in {"1", "true"})
        raise ResponseValidationError("invalid_express")

    @staticmethod
    def _parse_number(value: Any, field: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ResponseValidationError("invalid_" + field)
        if not math.isfinite(number):
            raise ResponseValidationError("invalid_" + field)
        return number

    @classmethod
    def _find_candidate(cls, data: Any) -> Optional[Dict[str, Any]]:
        if isinstance(data, dict):
            if any(key in data for key in ("express", "stance_delta", "stance", "emotion_delta", "emotion")):
                return data
            for key in ("content", "text", "output", "result"):
                value = data.get(key)
                if isinstance(value, str):
                    parsed = cls._parse_text(value)
                    if parsed is not None:
                        return parsed
                elif isinstance(value, (dict, list)):
                    parsed = cls._find_candidate(value)
                    if parsed is not None:
                        return parsed
            for key in ("message", "choices", "data"):
                if key in data:
                    parsed = cls._find_candidate(data[key])
                    if parsed is not None:
                        return parsed
        elif isinstance(data, list):
            for value in data:
                parsed = cls._find_candidate(value)
                if parsed is not None:
                    return parsed
        elif isinstance(data, str):
            return cls._parse_text(data)
        return None

    @classmethod
    def _parse_text(cls, text: str) -> Any:
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for target in ("[", "{"):
            starts = [match.start() for match in re.finditer(re.escape(target), cleaned)]
            for left in reversed(starts):
                try:
                    parsed, _ = decoder.raw_decode(cleaned[left:])
                    if target == "[" and isinstance(parsed, list):
                        return parsed
                    if target == "{" and isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        patterns = {
            "express": r"['\"]?express['\"]?\s*[:=]\s*(true|false|0|1)",
            "stance_delta": r"['\"]?stance_delta['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            "stance": r"['\"]?stance['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            "emotion_delta": r"['\"]?emotion_delta['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
            "emotion": r"['\"]?emotion['\"]?\s*[:=]\s*(\d+(?:\.\d+)?)",
        }
        parsed: Dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                parsed[key] = match.group(1)
        return parsed or None
