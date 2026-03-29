from __future__ import annotations

from typing import Tuple

from src.chat.qwen_api import dashscope_chat, extract_json_object

VALID_INTENTS: Tuple[str, ...] = ("chitchat", "factual", "compare", "policy", "howto")

INTENT_SYSTEM = """你是电商售前场景的查询意图分类器。只输出 JSON，不要其他文字。
从用户问题中选出唯一一个意图标签：

- chitchat: 寒暄、感谢、与商品无关的闲聊
- factual: 参数、规格、兼容性、硬件配置等事实性问题
- compare: 明确对比两个或多个商品/方案（含「对比」「哪个好」「A还是B」）
- policy: 促销、优惠券叠加、退换货、保修政策、活动规则
- howto: 连接、配对、安装、设置、使用步骤

输出格式严格为：
{"intent":"<标签>","confidence":0.0到1.0的小数}
标签必须是: chitchat, factual, compare, policy, howto 之一。"""


def classify_intent_qwen(
    query: str,
    *,
    api_key: str,
    model: str,
    timeout_sec: float = 15.0,
    temperature: float = 0.0,
) -> Tuple[str, float]:
    content = dashscope_chat(
        api_key,
        model,
        [
            {"role": "system", "content": INTENT_SYSTEM},
            {"role": "user", "content": f"用户问题：{query}"},
        ],
        temperature=temperature,
        max_tokens=128,
        timeout_sec=timeout_sec,
    )
    obj = extract_json_object(content)
    intent = str(obj.get("intent", "factual")).strip().lower()
    if intent not in VALID_INTENTS:
        intent = "factual"
    conf = obj.get("confidence", 0.5)
    try:
        confidence = float(conf)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return intent, confidence
