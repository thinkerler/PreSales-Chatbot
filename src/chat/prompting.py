from __future__ import annotations

from typing import Dict, List, Literal

from src.common.models import RetrievalHit

PromptStyle = Literal["engineering", "customer"]


def _intent_system_addon_engineering(intent: str) -> str:
    addons = {
        "factual": "当前为事实问答：逐条对照证据回答参数与规格，缺证据则说明证据不足。",
        "compare": (
            "当前为对比场景：用表格或分点对比各选项，每一项必须引用证据编号 [E1] 等；"
            "若证据不足以支撑对比，明确说明并建议人工客服。"
        ),
        "policy": (
            "当前为规则/政策场景：仅复述证据中的规则原文含义，禁止推测未写明的叠加规则；"
            "证据未覆盖时必须说证据不足。"
        ),
        "howto": (
            "当前为操作指导场景：用有序步骤（1. 2. 3.）回答，每一步尽量标注对应证据编号。"
        ),
    }
    return addons.get(intent, addons["factual"])


def _intent_system_addon_customer(intent: str) -> str:
    addons = {
        "factual": "用户问参数或事实：只根据证据介绍，没有的信息就说暂时查不到，别猜。",
        "compare": "用户在对比商品：用口语分点说差异，证据不够就说清楚并建议看详情页或问人工。",
        "policy": "用户问活动或保修规则：只复述证据里有的，没有就说以页面或客服为准，别脑补叠加规则。",
        "howto": "用户问怎么用：按步骤说清楚，某步证据没有就说这一步需要看说明书或问客服。",
    }
    return addons.get(intent, addons["factual"])


def _intent_user_addon_engineering(intent: str) -> str:
    if intent == "chitchat":
        return "请简短自然回复，不必引用证据编号。"
    if intent == "compare":
        return "请输出对比结论，并确保每个要点对应证据编号。"
    if intent == "howto":
        return "请按步骤说明，并引用证据编号。"
    if intent == "policy":
        return "请严格依据证据说明规则，不要扩展未出现的条款。"
    return "请引用证据编号作答。"


def _intent_user_addon_customer(intent: str) -> str:
    if intent == "chitchat":
        return "随便聊两句即可，别编造商品信息。"
    return (
        "请像真人客服一样直接回复用户；正文中不要用 [E1] 这种编号；"
        "若用到知识库内容，可用「根据说明」「资料里写到」等自然说法。"
    )


def build_system_prompt(
    user_profile: Dict[str, str] | None = None,
    *,
    intent: str = "factual",
    prompt_style: PromptStyle = "engineering",
) -> str:
    profile_text = ""
    if user_profile:
        profile_text = (
            f"用户画像: 预算={user_profile.get('budget', '未知')}, "
            f"偏好={user_profile.get('preference', '未知')}, "
            f"核心需求={user_profile.get('need', '未知')}。"
        )

    if intent == "chitchat":
        if prompt_style == "customer":
            return (
                "你是淘天店铺里的智能导购，正在和顾客聊天。"
                "语气轻松友好、一两句话即可；不要编造参数和活动；"
                "若对方想买东西，可以温和引导他说预算和需求。"
                + profile_text
            )
        return (
            "你是淘天售前导购助手。"
            "当前为寒暄/闲聊：回复简短友好，不要编造商品参数或活动规则；"
            "若用户开始咨询商品，可礼貌引导其描述需求。"
            + profile_text
        )

    if prompt_style == "customer":
        base = (
            "你是淘天店铺的智能导购客服，正在网页聊天窗口里回复顾客。"
            "语气亲切、口语化，像真人，不要用「结论」「理由」「风险提示」等报告式标题。"
            "只用下面提供的知识库片段回答；没有提到的信息就坦诚说暂时查不到，建议看商品详情页或联系人工客服。"
            "不要编造活动规则和价格。若用户画像里有预算、偏好、需求，请自然融入推荐话术。"
            "回答用短段落即可，必要时用少量列表，避免长篇大论。"
        )
        return base + profile_text + _intent_system_addon_customer(intent)

    base = (
        "你是淘天售前导购助手。回答必须基于给定证据，禁止编造活动规则。"
        "若证据不足，请明确说“当前证据不足，建议查看商品页或咨询人工客服”。"
        "回答结构: 结论 -> 理由 -> 风险提示。"
        "每个关键结论后标注对应证据编号，例如 [E1]。"
    )
    return base + profile_text + _intent_system_addon_engineering(intent)


def build_context(hits: List[RetrievalHit]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f"[E{i}] doc_id={hit.doc_id}; title={hit.title}; "
            f"category={hit.metadata.get('category', '')}; content={hit.content}"
        )
    return "\n".join(parts)


def build_user_prompt(
    query: str,
    context: str,
    *,
    intent: str = "factual",
    prompt_style: PromptStyle = "engineering",
) -> str:
    if prompt_style == "customer":
        suffix = _intent_user_addon_customer(intent)
        if intent == "chitchat":
            return f"用户说: {query}\n{suffix}"
        return f"顾客问题：{query}\n\n店内知识库摘录：\n{context}\n\n{suffix}"

    suffix = _intent_user_addon_engineering(intent)
    if intent == "chitchat":
        return f"用户说: {query}\n{suffix}"
    return f"问题: {query}\n证据:\n{context}\n{suffix}"
