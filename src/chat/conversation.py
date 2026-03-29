from __future__ import annotations

from typing import Any


def build_retrieval_query(query: str, messages: list[dict[str, Any]] | None) -> str:
    """用上文用户句与当前句拼成检索查询，缓解「它/这款」指代丢失。"""
    if not messages:
        return query.strip()
    user_msgs = [str(m.get("content", "")).strip() for m in messages if m.get("role") == "user"]
    tail = user_msgs[-3:] if len(user_msgs) > 3 else user_msgs
    parts = tail + [query.strip()]
    joined = " ".join(p for p in parts if p)
    if len(joined) > 1200:
        joined = " ".join(parts[-2:]) if len(parts) >= 2 else joined[:1200]
    return joined.strip() or query.strip()


def format_conversation_prefix(messages: list[dict[str, Any]] | None, *, max_messages: int = 12) -> str:
    """把最近若干轮对话拼进用户侧提示，便于模型理解指代。"""
    if not messages:
        return ""
    tail = messages[-max_messages:]
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if len(content) > 800:
            content = content[:800] + "…"
        if role == "user":
            lines.append(f"用户: {content}")
        elif role == "assistant":
            lines.append(f"助手: {content}")
    if not lines:
        return ""
    return "对话上文:\n" + "\n".join(lines) + "\n\n"
