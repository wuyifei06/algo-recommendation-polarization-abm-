"""LLM Agent wrapper (用于将 state/feeds 转换为 LLM 调用)。"""
from typing import Dict, Any
from .client import LLMClient
from .prompt import build_prompt


class LLMAgent:
    def __init__(self, client: LLMClient = None):
        self.client = client or LLMClient()

    def decide(self, state: Dict[str, Any], feed_items: list) -> Dict[str, Any]:
        prompt = build_prompt(state, feed_items)
        # In real client, use prompt to call LLM. Here call stub API method.
        payload = {"items": feed_items}
        return self.client.generate_agent_action(state, payload)
