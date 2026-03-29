from __future__ import annotations

import time
from typing import Any, Dict, Generator, List, Optional

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.chat.conversation import build_retrieval_query, format_conversation_prefix
from src.chat.intent_router import classify_intent_qwen
from src.chat.prompting import PromptStyle, build_context, build_system_prompt, build_user_prompt
from src.chat.qwen_api import dashscope_chat, dashscope_chat_stream, ollama_chat_stream
from src.common.models import RetrievalHit


def _extract_evidence_refs(hits: List[RetrievalHit]) -> List[str]:
    return [h.doc_id for h in hits]


def _rule_based_answer(query: str, hits: List[RetrievalHit]) -> str:
    joined = " ".join([h.content for h in hits])
    if "证据不足" in joined or not hits:
        return "当前证据不足，建议查看商品页或咨询人工客服。"
    return (
        "结论：基于当前检索结果可提供初步建议。\n"
        f"理由：已匹配到 {len(hits)} 条相关证据，重点覆盖规则和商品参数。\n"
        "风险提示：活动叠加和库存请以下单结算页为准。"
    )


class ChatPipeline:
    def __init__(
        self,
        retriever,
        *,
        llm_backend: str = "ollama",
        dashscope_api_key: Optional[str] = None,
        dashscope_model: str = "qwen-plus",
        dashscope_intent_model: str = "qwen-turbo",
        use_intent_router: bool = True,
        use_ollama: bool = True,
        ollama_model: str = "qwen2.5:7b",
        temperature: float = 0.2,
        max_tokens: int = 600,
        score_threshold: float = 0.26,
        timeout_sec: int = 20,
        intent_timeout_sec: float = 12.0,
    ) -> None:
        self.retriever = retriever
        self.llm_backend = (llm_backend or "ollama").lower().strip()
        self.dashscope_api_key = (dashscope_api_key or "").strip() or None
        self.dashscope_model = dashscope_model
        self.dashscope_intent_model = dashscope_intent_model
        self.use_intent_router = bool(use_intent_router) and bool(self.dashscope_api_key)
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.score_threshold = score_threshold
        self.timeout_sec = timeout_sec
        self.intent_timeout_sec = intent_timeout_sec
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("user", "{user_prompt}"),
            ]
        )
        self._llm_chain = (
            self._prompt
            | RunnableLambda(self._invoke_from_prompt_value)
            | StrOutputParser()
        )

    def _can_run_dashscope_chat(self) -> bool:
        return self.llm_backend == "dashscope" and bool(self.dashscope_api_key)

    def _can_run_llm(self) -> bool:
        if self._can_run_dashscope_chat():
            return True
        return self.use_ollama

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post("http://127.0.0.1:11434/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

    def _call_dashscope(self, system_prompt: str, user_prompt: str) -> str:
        assert self.dashscope_api_key
        return dashscope_chat(
            self.dashscope_api_key,
            self.dashscope_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_sec=float(self.timeout_sec),
        )

    def _invoke_from_prompt_value(self, prompt_value) -> str:
        msgs = prompt_value.to_messages()
        system_prompt = ""
        user_prompt = ""
        for msg in msgs:
            if msg.type == "system":
                system_prompt = msg.content
            elif msg.type == "human":
                user_prompt = msg.content
        if self._can_run_dashscope_chat():
            return self._call_dashscope(system_prompt, user_prompt)
        return self._call_ollama(system_prompt, user_prompt)

    def _iter_llm_text_stream(self, system_prompt: str, user_prompt: str):
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self._can_run_dashscope_chat():
            assert self.dashscope_api_key
            yield from dashscope_chat_stream(
                self.dashscope_api_key,
                self.dashscope_model,
                msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_sec=float(self.timeout_sec),
            )
            return
        if self.use_ollama:
            yield from ollama_chat_stream(
                msgs,
                model=self.ollama_model,
                temperature=self.temperature,
                timeout_sec=float(self.timeout_sec),
            )
            return
        yield "（本机未配置可用的语言模型：请设置环境变量 DASHSCOPE_API_KEY 或启动 Ollama。）"

    def _resolve_intent(self, query: str) -> tuple[str, float, float]:
        if not self.use_intent_router or not self.dashscope_api_key:
            return "factual", 1.0, 0.0
        t0 = time.perf_counter()
        try:
            intent, confidence = classify_intent_qwen(
                query,
                api_key=self.dashscope_api_key,
                model=self.dashscope_intent_model,
                timeout_sec=self.intent_timeout_sec,
                temperature=0.0,
            )
        except Exception:
            intent, confidence = "factual", 0.0
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return intent, confidence, elapsed_ms

    def ask(
        self,
        query: str,
        user_profile: Dict[str, str] | None = None,
        *,
        prompt_style: PromptStyle = "engineering",
        messages: list[dict[str, Any]] | None = None,
    ) -> Dict:
        messages = messages or []
        started = time.perf_counter()
        retrieval_query = build_retrieval_query(query, messages)
        intent, intent_confidence, intent_ms = self._resolve_intent(retrieval_query)
        conv_prefix = format_conversation_prefix(messages)

        if intent == "chitchat":
            t_gen_start = time.perf_counter()
            system_prompt = build_system_prompt(
                user_profile=user_profile, intent=intent, prompt_style=prompt_style
            )
            user_prompt = conv_prefix + build_user_prompt(
                query, "", intent=intent, prompt_style=prompt_style
            )
            reject_reason = ""
            if self._can_run_llm():
                try:
                    answer = self._llm_chain.invoke(
                        {"system_prompt": system_prompt, "user_prompt": user_prompt}
                    )
                except Exception:
                    answer = "你好，我是售前导购助手，有商品或优惠相关问题可以随时问我。"
                    reject_reason = "llm_failed_fallback"
            else:
                answer = "你好，我是售前导购助手，有商品或优惠相关问题可以随时问我。"
                reject_reason = "llm_disabled_fallback"
            generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            cost_ms = round((time.perf_counter() - started) * 1000, 2)
            retrieval_debug: Dict = {
                "prompt_style": prompt_style,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "intent_latency_ms": intent_ms,
                "skipped_retrieval": True,
                "retrieval_query": retrieval_query,
                "stage_latency_ms": {
                    "intent": intent_ms,
                    "retrieve": 0.0,
                    "prompt": 0.0,
                    "generate": generation_ms,
                },
            }
            return {
                "query": query,
                "answer": answer,
                "evidence_doc_ids": [],
                "retrieval_debug": retrieval_debug,
                "latency_ms": cost_ms,
                "confidence": round(float(intent_confidence), 4),
                "reject_reason": reject_reason,
                "intent": intent,
            }

        t_retrieve_start = time.perf_counter()
        retrieval = self.retriever.retrieve(retrieval_query)
        t_retrieve_end = time.perf_counter()
        max_score = max([h.score for h in retrieval.hits], default=0.0)
        use_reranker = bool(getattr(self.retriever, "use_reranker", False))
        should_reject = len(retrieval.hits) == 0
        if (not use_reranker) and (max_score < self.score_threshold):
            should_reject = True

        reject_reason = ""
        generation_ms = 0.0
        prompt_ms = 0.0
        if should_reject:
            answer = "当前证据不足，建议查看商品页或咨询人工客服。"
            reject_reason = "no_evidence" if not retrieval.hits else "low_confidence"
        else:
            t_prompt_start = time.perf_counter()
            system_prompt = build_system_prompt(
                user_profile=user_profile, intent=intent, prompt_style=prompt_style
            )
            context = build_context(retrieval.hits)
            user_prompt = conv_prefix + build_user_prompt(
                query, context, intent=intent, prompt_style=prompt_style
            )
            t_prompt_end = time.perf_counter()
            t_gen_start = time.perf_counter()
            if self._can_run_llm():
                try:
                    answer = self._llm_chain.invoke(
                        {"system_prompt": system_prompt, "user_prompt": user_prompt}
                    )
                except Exception:
                    answer = _rule_based_answer(query, retrieval.hits)
                    reject_reason = "llm_failed_fallback"
            else:
                answer = _rule_based_answer(query, retrieval.hits)
                reject_reason = "llm_disabled_fallback"
            generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            prompt_ms = round((t_prompt_end - t_prompt_start) * 1000, 2)

        cost_ms = round((time.perf_counter() - started) * 1000, 2)
        evidence_refs = _extract_evidence_refs(retrieval.hits)
        if prompt_style == "customer":
            cited_answer = answer
        else:
            cited_answer = answer + (
                ("\n\n证据引用: " + ", ".join(evidence_refs)) if evidence_refs else ""
            )
        confidence = round(float(max_score), 4)
        retrieval_stage_ms = round((t_retrieve_end - t_retrieve_start) * 1000, 2)
        stage_latency = {
            "intent": intent_ms,
            "retrieve": retrieval_stage_ms,
            "generate": generation_ms,
        }
        if not should_reject:
            stage_latency["prompt"] = prompt_ms
        retrieval_debug = dict(retrieval.debug)
        retrieval_debug["prompt_style"] = prompt_style
        retrieval_debug["intent"] = intent
        retrieval_debug["intent_confidence"] = intent_confidence
        retrieval_debug["intent_latency_ms"] = intent_ms
        retrieval_debug["confidence"] = confidence
        retrieval_debug["retrieval_query"] = retrieval_query
        retrieval_debug["stage_latency_ms"] = {
            **retrieval_debug.get("stage_latency_ms", {}),
            **stage_latency,
        }
        return {
            "query": query,
            "answer": cited_answer,
            "evidence_doc_ids": evidence_refs,
            "retrieval_debug": retrieval_debug,
            "latency_ms": cost_ms,
            "confidence": confidence,
            "reject_reason": reject_reason,
            "intent": intent,
        }

    def ask_stream(
        self,
        query: str,
        user_profile: Dict[str, str] | None = None,
        *,
        prompt_style: PromptStyle = "engineering",
        messages: list[dict[str, Any]] | None = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """NDJSON 流：多次 {type:delta,text}，最后 {type:end,...} 与 ask() 字段对齐。"""
        messages = messages or []
        started = time.perf_counter()
        retrieval_query = build_retrieval_query(query, messages)
        intent, intent_confidence, intent_ms = self._resolve_intent(retrieval_query)
        conv_prefix = format_conversation_prefix(messages)

        def _finish(
            answer: str,
            *,
            evidence_refs: List[str],
            retrieval_debug: Dict,
            reject_reason: str,
            confidence: float,
        ) -> Dict[str, Any]:
            cost_ms = round((time.perf_counter() - started) * 1000, 2)
            cited = answer
            if prompt_style != "customer" and evidence_refs:
                cited = answer + ("\n\n证据引用: " + ", ".join(evidence_refs))
            return {
                "type": "end",
                "query": query,
                "answer": cited,
                "evidence_doc_ids": evidence_refs,
                "retrieval_debug": retrieval_debug,
                "latency_ms": cost_ms,
                "confidence": confidence,
                "reject_reason": reject_reason,
                "intent": intent,
            }

        if intent == "chitchat":
            system_prompt = build_system_prompt(
                user_profile=user_profile, intent=intent, prompt_style=prompt_style
            )
            user_prompt = conv_prefix + build_user_prompt(
                query, "", intent=intent, prompt_style=prompt_style
            )
            reject_reason = ""
            full = ""
            t_gen_start = time.perf_counter()
            if self._can_run_llm():
                try:
                    for chunk in self._iter_llm_text_stream(system_prompt, user_prompt):
                        full += chunk
                        yield {"type": "delta", "text": chunk}
                except Exception:
                    full = "你好，我是售前导购助手，有商品或优惠相关问题可以随时问我。"
                    reject_reason = "llm_failed_fallback"
                    yield {"type": "delta", "text": full}
            else:
                full = "你好，我是售前导购助手，有商品或优惠相关问题可以随时问我。"
                reject_reason = "llm_disabled_fallback"
                yield {"type": "delta", "text": full}
            generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            retrieval_debug = {
                "prompt_style": prompt_style,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "intent_latency_ms": intent_ms,
                "skipped_retrieval": True,
                "retrieval_query": retrieval_query,
                "stage_latency_ms": {
                    "intent": intent_ms,
                    "retrieve": 0.0,
                    "prompt": 0.0,
                    "generate": generation_ms,
                },
            }
            yield _finish(
                full,
                evidence_refs=[],
                retrieval_debug=retrieval_debug,
                reject_reason=reject_reason,
                confidence=round(float(intent_confidence), 4),
            )
            return

        t_retrieve_start = time.perf_counter()
        retrieval = self.retriever.retrieve(retrieval_query)
        t_retrieve_end = time.perf_counter()
        max_score = max([h.score for h in retrieval.hits], default=0.0)
        use_reranker = bool(getattr(self.retriever, "use_reranker", False))
        should_reject = len(retrieval.hits) == 0
        if (not use_reranker) and (max_score < self.score_threshold):
            should_reject = True

        evidence_refs = _extract_evidence_refs(retrieval.hits)
        confidence = round(float(max_score), 4)
        retrieval_stage_ms = round((t_retrieve_end - t_retrieve_start) * 1000, 2)
        retrieval_debug = dict(retrieval.debug)
        retrieval_debug["prompt_style"] = prompt_style
        retrieval_debug["intent"] = intent
        retrieval_debug["intent_confidence"] = intent_confidence
        retrieval_debug["intent_latency_ms"] = intent_ms
        retrieval_debug["confidence"] = confidence
        retrieval_debug["retrieval_query"] = retrieval_query

        if should_reject:
            answer = "当前证据不足，建议查看商品页或咨询人工客服。"
            reject_reason = "no_evidence" if not retrieval.hits else "low_confidence"
            yield {"type": "delta", "text": answer}
            retrieval_debug["stage_latency_ms"] = {
                **retrieval_debug.get("stage_latency_ms", {}),
                **{
                    "intent": intent_ms,
                    "retrieve": retrieval_stage_ms,
                    "generate": 0.0,
                },
            }
            yield _finish(
                answer,
                evidence_refs=evidence_refs,
                retrieval_debug=retrieval_debug,
                reject_reason=reject_reason,
                confidence=confidence,
            )
            return

        t_prompt_start = time.perf_counter()
        system_prompt = build_system_prompt(
            user_profile=user_profile, intent=intent, prompt_style=prompt_style
        )
        context = build_context(retrieval.hits)
        user_prompt = conv_prefix + build_user_prompt(
            query, context, intent=intent, prompt_style=prompt_style
        )
        t_prompt_end = time.perf_counter()
        prompt_ms = round((t_prompt_end - t_prompt_start) * 1000, 2)
        t_gen_start = time.perf_counter()
        reject_reason = ""
        full = ""
        if self._can_run_llm():
            try:
                for chunk in self._iter_llm_text_stream(system_prompt, user_prompt):
                    full += chunk
                    yield {"type": "delta", "text": chunk}
            except Exception:
                full = _rule_based_answer(query, retrieval.hits)
                reject_reason = "llm_failed_fallback"
                yield {"type": "delta", "text": full}
        else:
            full = _rule_based_answer(query, retrieval.hits)
            reject_reason = "llm_disabled_fallback"
            yield {"type": "delta", "text": full}
        generation_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
        retrieval_debug["stage_latency_ms"] = {
            **retrieval_debug.get("stage_latency_ms", {}),
            **{
                "intent": intent_ms,
                "retrieve": retrieval_stage_ms,
                "prompt": prompt_ms,
                "generate": generation_ms,
            },
        }
        yield _finish(
            full,
            evidence_refs=evidence_refs,
            retrieval_debug=retrieval_debug,
            reject_reason=reject_reason,
            confidence=confidence,
        )
