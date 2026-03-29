from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import gradio as gr
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_settings

settings = load_settings()
API_BASE = "http://127.0.0.1:8000"
TIMEOUT_SEC = max(30, int(settings["runtime"]["request_timeout_sec"]) + 10)


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_SEC)


def _ask_api(
    query: str,
    user_profile: dict[str, str],
    profile_id: str | None,
    *,
    prompt_style: str,
    messages: list[dict] | None = None,
) -> dict:
    trace_id = str(uuid.uuid4())
    payload: dict = {
        "query": query,
        "user_profile": user_profile,
        "prompt_style": prompt_style,
    }
    if profile_id:
        payload["profile_id"] = profile_id
    if messages:
        payload["messages"] = messages
    with _client() as client:
        resp = client.post(f"{API_BASE}/ask", json=payload, headers={"x-trace-id": trace_id})
        resp.raise_for_status()
        body = resp.json()
        body["trace_id"] = resp.headers.get("x-trace-id", body.get("trace_id", trace_id))
        return body


def _iter_ask_stream(
    query: str,
    user_profile: dict[str, str],
    profile_id: str | None,
    *,
    prompt_style: str,
    messages: list[dict] | None = None,
):
    trace_id = str(uuid.uuid4())
    payload: dict = {
        "query": query,
        "user_profile": user_profile,
        "prompt_style": prompt_style,
    }
    if profile_id:
        payload["profile_id"] = profile_id
    if messages:
        payload["messages"] = messages
    with httpx.Client(timeout=TIMEOUT_SEC) as client:
        with client.stream(
            "POST",
            f"{API_BASE}/ask/stream",
            json=payload,
            headers={"x-trace-id": trace_id},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                yield obj


def _list_profiles() -> list[dict]:
    try:
        with _client() as client:
            r = client.get(f"{API_BASE}/profiles")
            r.raise_for_status()
            return r.json()
    except Exception:
        return []


def _save_profile(
    display_name: str, budget: str, preference: str, need: str, existing_id: str
) -> tuple[str, str]:
    if not (display_name or "").strip():
        return existing_id or "", "请填写画像名称后再保存。"
    body: dict = {
        "display_name": display_name.strip(),
        "budget": budget or "",
        "preference": preference or "",
        "need": need or "",
    }
    if (existing_id or "").strip():
        body["profile_id"] = existing_id.strip()
    try:
        with _client() as client:
            r = client.post(f"{API_BASE}/profiles", json=body)
            r.raise_for_status()
            pid = r.json().get("profile_id", "")
            return pid, f"已保存画像，id={pid}"
    except Exception as exc:
        return existing_id or "", f"保存失败: {exc}"


def _dropdown_update():
    rows = _list_profiles()
    choices: list[tuple[str, str]] = [("（不选用已保存画像）", "")]
    for p in rows:
        pid = p.get("id", "")
        name = p.get("display_name", "未命名")
        choices.append((f"{name} · {pid[:8]}…", pid))
    return gr.Dropdown(choices=choices, value=None)


def _load_profile_fields(profile_id: str) -> tuple[str, str, str, str, str]:
    if not profile_id:
        return "", "", "", "", ""
    try:
        with _client() as client:
            r = client.get(f"{API_BASE}/profiles/{profile_id}")
            if r.status_code == 404:
                return "", "", "", "", ""
            r.raise_for_status()
            p = r.json()
            return (
                p.get("display_name", ""),
                p.get("budget", ""),
                p.get("preference", ""),
                p.get("need", ""),
                p.get("id", profile_id),
            )
    except Exception:
        return "", "", "", "", ""


def _normalize_chat_history(history) -> list:
    if not history:
        return []
    out: list = []
    for item in history:
        if isinstance(item, dict) and "role" in item and "content" in item:
            out.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append({"role": "user", "content": str(item[0])})
            out.append({"role": "assistant", "content": str(item[1])})
    return out


def _render_debug(result: dict) -> str:
    retrieval_debug = result.get("retrieval_debug", {})
    return "\n".join(
        [
            f"trace_id: {result.get('trace_id', '')}",
            f"prompt_style: {retrieval_debug.get('prompt_style', '')}",
            f"intent: {result.get('intent', '')}",
            f"latency_ms: {result.get('latency_ms', 0)}",
            f"confidence: {result.get('confidence', 0)}",
            f"reject_reason: {result.get('reject_reason', '')}",
            f"profile_id: {result.get('profile_id', '')}",
            "retrieval_debug:",
            json.dumps(retrieval_debug, ensure_ascii=False, indent=2),
        ]
    )


_EMPTY_PROFILE = {"budget": "", "preference": "", "need": ""}


def chat_fn_customer(
    history,
    msg: str,
    budget: str,
    preference: str,
    need: str,
    profile_dd_value: str,
    state,
):
    """与研发调试共用同一套画像组件；仅在调试 Tab 可见，但请求会带上 user_profile / profile_id。"""
    query = (msg or "").strip()
    if not query:
        yield history, "", state
        return
    history = list(_normalize_chat_history(history or []))
    prior = [dict(m) for m in history]
    user_profile = {"budget": budget or "", "preference": preference or "", "need": need or ""}
    pid = (profile_dd_value or "").strip() or None
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": ""})
    yield history, "", state
    acc = ""
    try:
        for obj in _iter_ask_stream(
            query=query,
            user_profile=user_profile,
            profile_id=pid,
            prompt_style="customer",
            messages=prior,
        ):
            t = obj.get("type")
            if t == "start":
                continue
            if t == "delta":
                acc += obj.get("text") or ""
                history[-1]["content"] = acc
                yield history, "", state
            elif t == "end":
                state = {
                    "last_query": query,
                    "last_messages": prior,
                    "last_profile": user_profile,
                    "last_profile_id": pid or "",
                    "prompt_style": "customer",
                }
                yield history, "", state
                return
    except Exception as exc:
        history[-1]["content"] = f"暂时无法连接服务：{exc}"
        yield history, "", state


def chat_fn_debug(
    history,
    query: str,
    budget: str,
    preference: str,
    need: str,
    profile_dd_value: str,
    state,
):
    history = list(_normalize_chat_history(history or []))
    prior = [dict(m) for m in history]
    user_profile = {"budget": budget, "preference": preference, "need": need}
    pid = (profile_dd_value or "").strip() or None
    q = (query or "").strip()
    if not q:
        yield history, query, "", "", state
        return
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": ""})
    yield history, "", "", "", state
    acc = ""
    try:
        for obj in _iter_ask_stream(
            query=q,
            user_profile=user_profile,
            profile_id=pid,
            prompt_style="engineering",
            messages=prior,
        ):
            t = obj.get("type")
            if t == "start":
                continue
            if t == "delta":
                acc += obj.get("text") or ""
                history[-1]["content"] = acc
                yield history, "", "", "", state
            elif t == "end":
                evidence = "\n".join([f"- {doc_id}" for doc_id in obj.get("evidence_doc_ids", [])])
                debug = _render_debug(obj)
                state = {
                    "last_query": q,
                    "last_messages": prior,
                    "last_profile": user_profile,
                    "last_profile_id": pid or "",
                    "prompt_style": "engineering",
                }
                yield history, "", evidence, debug, state
                return
    except Exception as exc:
        history[-1]["content"] = f"请求失败: {exc}"
        yield history, "", "", f"error: {exc}", state


def retry_fn_customer(history, state):
    if not state or not state.get("last_query"):
        yield history, "", state
        return
    up = state.get("last_profile") or _EMPTY_PROFILE
    pid = (state.get("last_profile_id") or "").strip() or None
    prior = state.get("last_messages") or []
    history = list(_normalize_chat_history(history or []))
    history.append({"role": "user", "content": "重试上一次"})
    history.append({"role": "assistant", "content": ""})
    yield history, "", state
    acc = ""
    try:
        for obj in _iter_ask_stream(
            query=state["last_query"],
            user_profile=up,
            profile_id=pid,
            prompt_style="customer",
            messages=prior,
        ):
            t = obj.get("type")
            if t == "start":
                continue
            if t == "delta":
                acc += obj.get("text") or ""
                history[-1]["content"] = acc
                yield history, "", state
            elif t == "end":
                yield history, "", state
                return
    except Exception:
        yield history, "", state


def retry_fn_debug(history, state):
    if not state or not state.get("last_query"):
        yield history, "", "没有可重试的请求", state
        return
    pid = (state.get("last_profile_id") or "").strip() or None
    prior = state.get("last_messages") or []
    history = list(_normalize_chat_history(history or []))
    history.append({"role": "user", "content": "重试上一次"})
    history.append({"role": "assistant", "content": ""})
    yield history, "", "", state
    acc = ""
    try:
        for obj in _iter_ask_stream(
            query=state["last_query"],
            user_profile=state.get("last_profile") or _EMPTY_PROFILE,
            profile_id=pid,
            prompt_style="engineering",
            messages=prior,
        ):
            t = obj.get("type")
            if t == "start":
                continue
            if t == "delta":
                acc += obj.get("text") or ""
                history[-1]["content"] = f"[重试] {acc}"
                yield history, "", "", state
            elif t == "end":
                evidence = "\n".join([f"- {doc_id}" for doc_id in obj.get("evidence_doc_ids", [])])
                debug = _render_debug(obj)
                yield history, evidence, debug, state
                return
    except Exception as exc:
        yield history, "", f"retry_error: {exc}", state


def do_save(display_name, budget, preference, need, saved_id_state):
    pid, msg = _save_profile(display_name, budget, preference, need, saved_id_state)
    return msg, pid, _dropdown_update()


CUSTOM_CSS = """
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="智能导购") as demo:
    gr.Markdown("# 智能导购")

    with gr.Tabs():
        with gr.Tab("智能客服"):
            state_c = gr.State({})
            chat_c = gr.Chatbot(
                label="",
                height=560,
                show_label=False,
                layout="bubble",
                placeholder="您好，需要了解哪款商品？",
            )
            with gr.Row():
                msg_c = gr.Textbox(
                    placeholder="输入消息…",
                    lines=1,
                    scale=5,
                    show_label=False,
                    container=False,
                )
                send_c = gr.Button("发送", variant="primary", scale=1, min_width=100)
            retry_c = gr.Button("重试上一次", variant="secondary", size="sm")

        with gr.Tab("研发调试"):
            gr.Markdown(
                "##### 用户画像\n"
                "在此配置预算、偏好、已保存画像等；**智能客服**页对话会沿用当前画像（嵌入导购 Prompt），无需在客服页重复填写。"
            )
            profile_dd = gr.Dropdown(
                label="选用已保存画像",
                choices=[("（不选用已保存画像）", "")],
                value=None,
            )
            display_name = gr.Textbox(label="画像名称（保存）", placeholder="例如：男大学生-游戏")
            with gr.Row():
                budget = gr.Textbox(label="预算", placeholder="如 500 元内", scale=1)
                preference = gr.Textbox(label="偏好", placeholder="如 无线、静音", scale=1)
                need = gr.Textbox(label="核心需求", placeholder="如 传感器、续航", scale=1)
            with gr.Row():
                save_btn = gr.Button("保存 / 更新画像", variant="secondary", size="sm")
                refresh_btn = gr.Button("刷新画像列表", size="sm")
            save_msg = gr.Textbox(label="保存结果", interactive=False, lines=1)
            saved_id_state = gr.State("")

            gr.Markdown("##### 对话与 trace")
            state_d = gr.State({})
            chat_d = gr.Chatbot(label="对话", height=360)
            query_d = gr.Textbox(label="问题", placeholder="例如：G304 支持蓝牙吗？", lines=2)
            with gr.Row():
                send_d = gr.Button("发送", variant="primary")
                retry_d = gr.Button("重试上一次")
            evidence = gr.Textbox(label="证据 doc_id", lines=5)
            debug = gr.Textbox(label="调试信息", lines=14)

    demo.load(_dropdown_update, outputs=profile_dd)

    profile_dd.change(
        lambda pid: _load_profile_fields(pid or ""),
        inputs=[profile_dd],
        outputs=[display_name, budget, preference, need, saved_id_state],
    )
    refresh_btn.click(_dropdown_update, outputs=profile_dd)
    save_btn.click(
        do_save,
        inputs=[display_name, budget, preference, need, saved_id_state],
        outputs=[save_msg, saved_id_state, profile_dd],
    )

    send_c.click(
        chat_fn_customer,
        inputs=[chat_c, msg_c, budget, preference, need, profile_dd, state_c],
        outputs=[chat_c, msg_c, state_c],
    )
    msg_c.submit(
        chat_fn_customer,
        inputs=[chat_c, msg_c, budget, preference, need, profile_dd, state_c],
        outputs=[chat_c, msg_c, state_c],
    )
    retry_c.click(
        retry_fn_customer,
        inputs=[chat_c, state_c],
        outputs=[chat_c, msg_c, state_c],
    )

    send_d.click(
        chat_fn_debug,
        inputs=[chat_d, query_d, budget, preference, need, profile_dd, state_d],
        outputs=[chat_d, query_d, evidence, debug, state_d],
    )
    retry_d.click(
        retry_fn_debug,
        inputs=[chat_d, state_d],
        outputs=[chat_d, evidence, debug, state_d],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
