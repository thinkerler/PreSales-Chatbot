from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import httpx

DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def dashscope_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout_sec: float = 60.0,
) -> str:
    if not api_key.strip():
        raise ValueError("DASHSCOPE_API_KEY is empty")
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(DASHSCOPE_CHAT_URL, headers=headers, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = resp.text[:500] if resp.text else str(exc)
            raise RuntimeError(f"DashScope HTTP {resp.status_code}: {detail}") from exc
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"DashScope empty choices: {json.dumps(data, ensure_ascii=False)[:500]}")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    return content


def dashscope_chat_stream(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout_sec: float = 120.0,
):
    """OpenAI 兼容流式：逐段 yield 文本增量。"""
    if not api_key.strip():
        raise ValueError("DASHSCOPE_API_KEY is empty")
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with httpx.Client(timeout=timeout_sec) as client:
        with client.stream("POST", DASHSCOPE_CHAT_URL, headers=headers, json=payload) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = resp.text[:500] if resp.text else str(exc)
                raise RuntimeError(f"DashScope HTTP {resp.status_code}: {detail}") from exc
            for line in resp.iter_lines():
                if line is None:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    yield piece


def ollama_chat_stream(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float = 0.2,
    timeout_sec: float = 120.0,
):
    """Ollama /api/chat stream=true，逐段 yield 文本增量。"""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    with httpx.Client(timeout=timeout_sec) as client:
        with client.stream("POST", "http://127.0.0.1:11434/api/chat", json=payload) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = resp.text[:500] if resp.text else str(exc)
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {detail}") from exc
            for line in resp.iter_lines():
                if line is None:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    yield piece


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse first JSON object from model output (handles optional markdown fences)."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in: {raw[:200]}")
    return json.loads(raw[start : end + 1])
