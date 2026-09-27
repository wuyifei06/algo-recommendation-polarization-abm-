"""LLM 客户端统一接口。

本项目要求：如果用户选择使用 LLM 实验，就必须强制使用真实大模型，
不能在没有配置时自动回退到本地 stub。所有缺失配置都会直接抛出错误，
方便用户及时修正 `.env`。
"""
import json
import os
import time
import threading
from typing import Dict, Any, Optional

import requests

from config import load_env
from llm.prompt import build_prompt
from utils.cost_guard import CostGuard


class LLMClient:
    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        endpoint: str = None,
        archive_path: str = None,
    ):
        env = load_env()
        self.provider = (provider or env.get("LLM_PROVIDER") or "dashscope").lower()

        alias_map = {
            "qwen": "dashscope",
            "ali": "aliyun",
            "alibaba": "aliyun",
            "dash": "dashscope",
            "openai": "openai",
        }
        self.provider = alias_map.get(self.provider, self.provider)

        self.model_name = (
            model_name
            or env.get("LLM_MODEL")
            or env.get("LLM_MODEL_NAME")
            or {
                "dashscope": env.get("DASHSCOPE_MODEL_NAME", "qwen-plus"),
                "deepseek": env.get("DEEPSEEK_MODEL_NAME", "deepseek-chat"),
                "aliyun": env.get("ALIBABA_MODEL_NAME", "qwen-plus"),
                "openai": env.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
            }.get(self.provider, "qwen-plus")
        ).strip()
        self.temperature = float(env.get("LLM_TEMPERATURE", 0.7))
        self.timeout = int(env.get("LLM_TIMEOUT_SECONDS", 20))
        self.max_retries = int(env.get("LLM_MAX_RETRIES", 3))
        self.verbose = (os.getenv("LLM_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"})

        self._env = env
        key_map = {
            "dashscope": env.get("LLM_API_KEY") or env.get("DASHSCOPE_API_KEY"),
            "deepseek": env.get("LLM_API_KEY") or env.get("DEEPSEEK_API_KEY"),
            "aliyun": env.get("LLM_API_KEY") or env.get("ALIBABA_API_KEY"),
            "openai": env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY"),
        }
        self.api_key = (key_map.get(self.provider) or "").strip()

        endpoint_map = {
            "dashscope": env.get("LLM_API_ENDPOINT") or env.get("DASHSCOPE_API_ENDPOINT"),
            "deepseek": env.get("LLM_API_ENDPOINT") or env.get("DEEPSEEK_API_ENDPOINT"),
            "aliyun": env.get("LLM_API_ENDPOINT") or env.get("ALIBABA_API_ENDPOINT"),
            "openai": env.get("LLM_API_ENDPOINT") or env.get("OPENAI_API_ENDPOINT"),
        }
        self.endpoint = self._normalize_endpoint(endpoint or endpoint_map.get(self.provider) or "")
        self.cost_guard = CostGuard()
        self._archive_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        # 累计 token 用量（线程安全）。仅做记录，不改变任何请求/解析行为。
        self._usage_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        # 每个 run 使用独立调用存档。这样恢复某个 checkpoint 时不会把别的
        # run 的调用混进来，也不会因为旧的全局 archive 而重复编号。
        self.archive_path = archive_path or os.path.join("results", "llm_call_archive.jsonl")
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
        self._call_counter = self._read_last_call_no(self.archive_path)

        if not self.api_key or not self.endpoint or not self.model_name:
            raise ValueError(
                "LLM mode requires a real model configuration: set LLM_API_KEY, "
                "LLM_API_ENDPOINT and LLM_MODEL in .env. Stub mode is forbidden."
            )

    @staticmethod
    def _read_last_call_no(path: str) -> int:
        """从已有存档中读取最大 call_no，确保续跑时编号连续。"""
        if not os.path.exists(path):
            return 0
        max_no = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    call_no = int(item.get("call_no", 0) or 0)
                    if call_no > max_no:
                        max_no = call_no
        except Exception:
            return 0
        return max_no

    def _normalize_endpoint(self, endpoint: str) -> str:
        """把兼容 OpenAI 的 base URL 转成真正的 completion 路径。"""
        if not endpoint:
            return ""
        base = endpoint.strip().rstrip('/')
        if base.endswith('/chat/completions'):
            return base
        if base.endswith('/v1') or base.endswith('/api/v1'):
            return f"{base}/chat/completions"
        return base

    def _write_archive(self, call_no: int, state: Dict[str, Any], feed: Dict[str, Any], result: Optional[Dict[str, Any]] = None, error: Optional[str] = None, usage: Optional[Dict[str, Any]] = None):
        """把每一次模型返回或失败都写入一个临时 JSONL 存档文件，用作断点恢复。"""
        entry = {
            "call_no": call_no,
            "model": self.model_name,
            "state": state,
            "feed": feed,
            "result": result,
            "error": error,
            "usage": usage,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with self._archive_lock:
            with open(self.archive_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _record_usage(self, usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """累计并返回标准化的 token 用量；没有 usage 字段时返回 None。"""
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        try:
            input_tokens = int(input_tokens) if input_tokens is not None else 0
            output_tokens = int(output_tokens) if output_tokens is not None else 0
            total_tokens = int(total_tokens) if total_tokens is not None else input_tokens + output_tokens
        except (TypeError, ValueError):
            return None
        normalized = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        with self._usage_lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total_tokens
        return normalized

    def usage_snapshot(self) -> Dict[str, int]:
        """返回当前累计 token 用量快照。"""
        with self._usage_lock:
            return {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
            }

    def generate_agent_action(self, state: Dict[str, Any], feed: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """强制调用真实大模型，不允许回退到 local stub。"""
        if not self.api_key or not self.endpoint or not self.model_name:
            raise ValueError(
                "LLM mode requires a real model configuration. "
                "Missing LLM_API_KEY / LLM_API_ENDPOINT / LLM_MODEL in .env."
            )

        if not self.cost_guard.allow_call(est_cost=0.01):
            raise RuntimeError("LLM call budget exceeded. Please increase MAX_LLM_CALLS or MAX_ESTIMATED_COST.")

        with self._counter_lock:
            self._call_counter += 1
            call_no = self._call_counter
        try:
            result, usage = self._call_provider(state, feed, timeout=timeout)
            self._write_archive(call_no, state, feed, result=result, usage=usage)
            if self.verbose:
                print(f"[LLM DEBUG] 第 {call_no} 次调用返回: {json.dumps(result, ensure_ascii=False)}", flush=True)
            return result
        except Exception as exc:
            self._write_archive(call_no, state, feed, error=str(exc))
            print(f"[LLM DEBUG] 第 {call_no} 次调用失败: {exc}", flush=True)
            raise RuntimeError(f"LLM real model call failed: {exc}") from exc

    def _call_provider(self, state: Dict[str, Any], feed: Dict[str, Any], timeout: int = 10):
        """对指定 provider 发起真实请求，并尽量兼容 OpenAI/通义千问/阿里云返回格式。

        返回 ``(parsed_result, usage)``：``parsed_result`` 为
        ``{express, + (stance_delta, emotion_delta)}`` 或旧的
        ``{express, + (stance, emotion)}``——与 ``build_prompt`` 要求的
        **相对增量**语义保持一致，同时兼容绝对量字段；``usage`` 为标准化
        token 用量（provider 未返回 usage 时为 None）。请求 payload 与重试
        逻辑保持不变。
        """
        prompts = [
            build_prompt(state, feed.get("items", [])),
            "Return only JSON with keys express, stance_delta, emotion_delta. No prose. "
            "express boolean, stance_delta number in [-0.05,0.05], "
            "emotion_delta number in [-0.1,0.1].",
        ]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            prompt = prompts[min(attempt - 1, len(prompts) - 1)]
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                # 推理型模型（如 deepseek-flash）会先消耗 token 进行思考，
                # max_tokens=32 时 content 必然为空。默认 1024，可用
                # LLM_MAX_TOKENS 在 .env 中按需调整。
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
                "response_format": {"type": "json_object"},
                "extra_body": {
                    "enable_thinking": False,
                    "reasoning_effort": "none",
                },
            }
            try:
                resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
                if resp.status_code >= 400:
                    body_preview = (resp.text or "")[:800]
                    raise requests.HTTPError(f"HTTP {resp.status_code} from provider; body_preview={body_preview}", response=resp)
                data = resp.json()
                usage = self._record_usage(data.get("usage") if isinstance(data, dict) else None)
                parsed = self._parse_provider_response(data)
                if parsed is not None:
                    if self.verbose:
                        print(
                            f"[LLM DEBUG] model={self.model_name} response={json.dumps(parsed, ensure_ascii=False)}",
                            flush=True,
                        )
                    return parsed, usage
                if self.verbose:
                    print(
                        f"[LLM DEBUG] model={self.model_name} raw_response={json.dumps(data, ensure_ascii=False, default=str)}",
                        flush=True,
                    )
                raw_preview = json.dumps(data, ensure_ascii=False, default=str)[:800]
                raise ValueError(f"Provider response missing required JSON fields; raw_preview={raw_preview}")
            except Exception as e:
                last_exc = e
                # HTTP 错误（4xx/5xx）不重试：多半是模型名/参数问题，重试只会浪费调用。
                if isinstance(e, requests.HTTPError):
                    raise
                if attempt < self.max_retries:
                    time.sleep(min(1.5 * attempt, 3.0))
                    continue
                raise last_exc

    def _parse_provider_response(self, data: Any) -> Optional[Dict[str, Any]]:
        """从多种 provider 返回结构中提取一次行动决策。

        字段支持两套语义：
          * 增量（当前 ``build_prompt`` 的约定）：``stance_delta`` / ``emotion_delta``；
          * 绝对量（旧 prompt 遗留）：``stance`` / ``emotion``。

        ⚠️ 2026-09-20 修复：旧实现在某个字段缺失时会把 ``stance`` / ``emotion``
        **静默填成 0.0**（``pick_bool`` 还会把缺失的 ``express`` 默认成 True）。
        配合 ``model.model._step_llm`` 的绝对值赋值，第 1 步就把全体 agent 立场
        重置为 0.0 并永久锁死，产出 ``PI/VPI/ER ≡ 0`` 的假结果。
        现在改为：``express`` 与立场/情绪字段**必须齐全**，缺任何一个都返回
        ``None`` → 由调用方抛错，绝不凭空造值。
        """
        if not isinstance(data, dict):
            return None

        # 两套字段语义：(增量, 增量) 优先，其次 (绝对量, 绝对量)。
        action_numeric_fields = (("stance_delta", "emotion_delta"), ("stance", "emotion"))

        def extract_action(obj: Any) -> Optional[Dict[str, Any]]:
            """从 dict 中抽出行动决策，兼容增量与绝对量两套字段。

            必需字段不全时返回 ``None``——**不再静默填默认值**。
            """
            if not isinstance(obj, dict) or "express" not in obj:
                return None
            for stance_key, emotion_key in action_numeric_fields:
                if stance_key in obj and emotion_key in obj:
                    try:
                        return {
                            "express": bool(obj["express"]),
                            stance_key: float(obj[stance_key]),
                            emotion_key: float(obj[emotion_key]),
                        }
                    except (TypeError, ValueError):
                        return None
            return None

        def parse_partial_json_fragment(text: str) -> Optional[Dict[str, Any]]:
            if not isinstance(text, str):
                return None
            cleaned = text.strip()
            if not cleaned:
                return None
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`\n ")
                if "json" in cleaned.lower():
                    cleaned = cleaned.split("json", 1)[1].strip()
            cleaned = cleaned.replace("'", '"')

            if not cleaned.startswith("{"):
                # some models emit a leading prefix before the JSON object.
                left = cleaned.find("{")
                if left >= 0:
                    cleaned = cleaned[left:]
                else:
                    return None

            import re
            expr_match = re.search(r'(?:"|\')?express(?:"|\')?\s*[:=]\s*(true|false|1|0)', cleaned, re.I)
            stance_match = re.search(r'(?:"|\')?(stance_delta|stance)(?:"|\')?\s*[:=]\s*(-?\d+(?:\.\d+)?)', cleaned)
            emotion_match = re.search(r'(?:"|\')?(emotion_delta|emotion)(?:"|\')?\s*[:=]\s*(-?\d+(?:\.\d+)?)', cleaned)

            # 三键必须齐全：缺任何一个都判为解析失败（旧实现会静默补 0.0）。
            if not (expr_match and stance_match and emotion_match):
                return None

            return {
                "express": expr_match.group(1).lower() in ("true", "1"),
                stance_match.group(1): float(stance_match.group(2)),
                emotion_match.group(1): float(emotion_match.group(2)),
            }

        def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
            if not isinstance(text, str):
                return None
            cleaned = text.strip()
            if not cleaned:
                return None
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`\n ")
                if "json" in cleaned.lower():
                    cleaned = cleaned.split("json", 1)[1].strip()
            if cleaned.startswith("{"):
                try:
                    action = extract_action(json.loads(cleaned))
                    if action is not None:
                        return action
                except Exception:
                    pass
                return parse_partial_json_fragment(cleaned)
            return None

        def parse_candidate(obj: Any) -> Optional[Dict[str, Any]]:
            if isinstance(obj, dict):
                action = extract_action(obj)
                if action is not None:
                    return action

                for key in ("content", "text", "output", "result"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        parsed = parse_json_object(val)
                        if parsed is not None:
                            return parsed
                    elif isinstance(val, dict):
                        parsed = parse_candidate(val)
                        if parsed is not None:
                            return parsed

                if "message" in obj and isinstance(obj["message"], dict):
                    parsed = parse_candidate(obj["message"])
                    if parsed is not None:
                        return parsed
                if "choices" in obj and isinstance(obj["choices"], list):
                    for item in obj["choices"]:
                        parsed = parse_candidate(item)
                        if parsed is not None:
                            return parsed

            elif isinstance(obj, list):
                for item in obj:
                    parsed = parse_candidate(item)
                    if parsed is not None:
                        return parsed

            return None

        direct = parse_candidate(data)
        if direct is not None:
            return direct

        for key in ("output", "result", "choices", "message", "data"):
            if key not in data:
                continue
            val = data[key]
            parsed = parse_candidate(val)
            if parsed is not None:
                return parsed

        return None
