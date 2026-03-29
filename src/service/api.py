from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.chat.pipeline import ChatPipeline
from src.common.config import load_settings
from src.common.logger import setup_logger
from src.ingestion.build_indexes import build_and_save_indexes
from src.retrieval.hybrid_retriever import HybridRetriever
from src.service.profile_store import (
    delete_profile,
    get_profile,
    list_profiles,
    merge_user_profile,
    upsert_profile,
)


class ChatMessage(BaseModel):
    """兼容 Gradio 6：content 可能为 str 或文本块列表；role 可能大小写不一致。"""

    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=8000)

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v: object) -> str:
        s = str(v).strip().lower() if v is not None else "user"
        if s in ("assistant", "bot", "model"):
            return "assistant"
        return "user"

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v[:8000]
        if isinstance(v, list):
            parts: List[str] = []
            for p in v:
                if isinstance(p, dict):
                    if "text" in p:
                        parts.append(str(p["text"]))
                    elif p.get("type") == "text" and "text" in p:
                        parts.append(str(p["text"]))
                    else:
                        parts.append(json.dumps(p, ensure_ascii=False))
                else:
                    parts.append(str(p))
            return " ".join(parts)[:8000]
        return str(v)[:8000]


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    user_profile: Optional[Dict[str, str]] = None
    profile_id: Optional[str] = Field(default=None, max_length=64)
    # engineering: 报告式+证据编号；customer: 导购口语、无[E1]格式
    prompt_style: Literal["engineering", "customer"] = "engineering"
    # 当前句之前的对话轮次（不含本句 query）；用于指代消解与流式多轮
    messages: Optional[List[ChatMessage]] = None


class ProfileUpsert(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)
    budget: str = Field(default="", max_length=200)
    preference: str = Field(default="", max_length=200)
    need: str = Field(default="", max_length=400)
    profile_id: Optional[str] = Field(default=None, max_length=64)


def _component_health(settings: Dict, retriever: HybridRetriever) -> Dict[str, str]:
    index_dir = settings["paths"]["index_dir"]
    docs_path = f"{index_dir}/docs.jsonl"
    lc_faiss_path = f"{index_dir}/lc_faiss/index.pkl"
    llm_backend = settings.get("generation", {}).get("llm_backend", "ollama")
    dash_ok = bool(os.environ.get("DASHSCOPE_API_KEY", "").strip())
    return {
        "index_docs": "ok" if Path(docs_path).exists() else "missing",
        "langchain_faiss": "ok" if Path(lc_faiss_path).exists() else "missing",
        "reranker": "enabled" if getattr(retriever, "use_reranker", False) else "disabled",
        "llm_backend": llm_backend,
        "dashscope_api_key": "configured" if dash_ok else "missing",
    }


def build_app() -> FastAPI:
    settings = load_settings()
    logger = setup_logger("api", settings["app"]["log_level"])
    profile_db = settings["paths"].get("profile_db", "data/profiles.db")

    ing = settings.get("ingestion", {})
    index_dir_path = Path(settings["paths"]["index_dir"])
    docs_jsonl = index_dir_path / "docs.jsonl"
    lc_faiss_pkl = index_dir_path / "lc_faiss" / "index.pkl"
    build_on_startup = bool(ing.get("build_on_startup", False))

    if build_on_startup:
        logger.info("ingestion.build_on_startup=true, running full index build...")
        build_and_save_indexes(
            kb_file=settings["paths"]["kb_file"],
            index_dir=settings["paths"]["index_dir"],
            chunk_size=settings["retrieval"]["chunk_size"],
            chunk_overlap=settings["retrieval"]["chunk_overlap"],
            splitter_type=settings["retrieval"].get("splitter_type", "recursive"),
            source_type=ing.get("source_type", "jsonl"),
            mineru_output_dir=ing.get("mineru_output_dir"),
            mineru_auto_run=ing.get("mineru_auto_run", False),
            mineru_input_dir=ing.get("mineru_input_dir"),
            mineru_command_template=ing.get(
                "mineru_command_template",
                'magic-pdf -p "{input_file}" -o "{output_dir}"',
            ),
            mineru_mode=ing.get("mineru_mode", "flash"),
            mineru_token=ing.get("mineru_token"),
            mineru_split_pages=ing.get("mineru_split_pages", True),
            mineru_language=ing.get("mineru_language", "ch"),
            mineru_timeout=ing.get("mineru_timeout", 1200),
            faiss_index_type=settings["retrieval"].get("faiss_index_type", "flatip"),
            faiss_nlist=settings["retrieval"].get("faiss_nlist", 64),
        )
    elif not docs_jsonl.is_file() or not lc_faiss_pkl.is_file():
        raise RuntimeError(
            f"Index missing under {index_dir_path} (need docs.jsonl and lc_faiss/index.pkl). "
            "Run ingestion once, e.g. `python scripts/run_mineru_pipeline.py`, "
            "or set ingestion.build_on_startup: true in configs/settings.yaml."
        )
    else:
        logger.info("Skipping ingestion (ingestion.build_on_startup=false), loading existing index.")

    retriever = HybridRetriever(
        index_dir=settings["paths"]["index_dir"],
        top_k_dense=settings["retrieval"]["top_k_dense"],
        top_k_sparse=settings["retrieval"]["top_k_sparse"],
        top_k_final=settings["retrieval"]["top_k_final"],
        rrf_k=settings["retrieval"]["rrf_k"],
        use_reranker=settings["retrieval"]["use_reranker"],
        reranker_model=settings["retrieval"]["reranker_model"],
        faiss_nprobe=settings["retrieval"].get("faiss_nprobe", 8),
    )
    gen = settings.get("generation", {})
    pipeline = ChatPipeline(
        retriever=retriever,
        llm_backend=gen.get("llm_backend", "ollama"),
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "").strip() or None,
        dashscope_model=gen.get("dashscope_model", "qwen-plus"),
        dashscope_intent_model=gen.get("dashscope_intent_model", "qwen-turbo"),
        use_intent_router=gen.get("use_intent_router", True),
        use_ollama=gen.get("use_ollama", True),
        ollama_model=gen.get("ollama_model", "qwen2.5:7b"),
        temperature=float(gen.get("temperature", 0.2)),
        max_tokens=int(gen.get("max_tokens", 600)),
        score_threshold=settings["retrieval"]["score_threshold"],
        timeout_sec=settings["runtime"]["request_timeout_sec"],
        intent_timeout_sec=float(gen.get("intent_timeout_sec", 12.0)),
    )
    semaphore = asyncio.Semaphore(settings["runtime"]["max_concurrency"])
    max_retries = settings["runtime"]["max_retries"]
    timeout_sec = settings["runtime"]["request_timeout_sec"]

    app = FastAPI(title="Taotian Presale RAG API", version="0.2.0")

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-trace-id"] = trace_id
        logger.info(
            "request_done",
            extra={"extra": {"trace_id": trace_id, "path": request.url.path, "latency_ms": elapsed_ms}},
        )
        return response

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "components": _component_health(settings, retriever),
        }

    @app.get("/profiles")
    async def profiles_list() -> List[Dict]:
        return list_profiles(profile_db)

    @app.get("/profiles/{profile_id}")
    async def profiles_get(profile_id: str) -> Dict:
        row = get_profile(profile_db, profile_id)
        if not row:
            raise HTTPException(status_code=404, detail="profile_not_found")
        return row

    @app.post("/profiles")
    async def profiles_save(body: ProfileUpsert) -> Dict[str, str]:
        pid = upsert_profile(
            profile_db,
            display_name=body.display_name,
            budget=body.budget,
            preference=body.preference,
            need=body.need,
            profile_id=body.profile_id,
        )
        return {"profile_id": pid}

    @app.delete("/profiles/{profile_id}")
    async def profiles_delete(profile_id: str) -> Dict[str, bool]:
        ok = delete_profile(profile_db, profile_id)
        if not ok:
            raise HTTPException(status_code=404, detail="profile_not_found")
        return {"ok": True}

    def _messages_as_dicts(req: AskRequest) -> Optional[List[Dict[str, str]]]:
        if not req.messages:
            return None
        return [m.model_dump() for m in req.messages]

    @app.post("/ask")
    async def ask(req: AskRequest, request: Request):
        trace_id = request.state.trace_id
        merged_profile: Optional[Dict[str, str]] = None
        if req.profile_id:
            stored = get_profile(profile_db, req.profile_id)
            if not stored:
                raise HTTPException(status_code=404, detail="profile_not_found")
            merged_profile = merge_user_profile(stored, req.user_profile)
        elif req.user_profile:
            merged_profile = merge_user_profile(None, req.user_profile)
        msg_list = _messages_as_dicts(req)
        async with semaphore:
            for i in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            pipeline.ask,
                            req.query,
                            merged_profile,
                            prompt_style=req.prompt_style,
                            messages=msg_list,
                        ),
                        timeout=timeout_sec,
                    )
                    stage_latency = result.get("retrieval_debug", {}).get("stage_latency_ms", {})
                    logger.info(
                        "ask_success",
                        extra={
                            "extra": {
                                "trace_id": trace_id,
                                "query": req.query,
                                "intent": result.get("intent", ""),
                                "prompt_style": req.prompt_style,
                                "profile_id": req.profile_id or "",
                                "latency_ms": result["latency_ms"],
                                "evidence_count": len(result["evidence_doc_ids"]),
                                "confidence": result.get("confidence", 0.0),
                                "reject_reason": result.get("reject_reason", ""),
                                "stage_latency_ms": stage_latency,
                            }
                        },
                    )
                    result["trace_id"] = trace_id
                    if req.profile_id:
                        result["profile_id"] = req.profile_id
                    return result
                except Exception as exc:  # pragma: no cover
                    if i == max_retries:
                        logger.error(
                            "ask_failed",
                            extra={"extra": {"trace_id": trace_id, "error": str(exc)}},
                        )
                        raise HTTPException(status_code=500, detail="internal_error") from exc
                    await asyncio.sleep(0.2 * (i + 1))

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest, request: Request):
        trace_id = request.state.trace_id
        merged_profile: Optional[Dict[str, str]] = None
        if req.profile_id:
            stored = get_profile(profile_db, req.profile_id)
            if not stored:
                raise HTTPException(status_code=404, detail="profile_not_found")
            merged_profile = merge_user_profile(stored, req.user_profile)
        elif req.user_profile:
            merged_profile = merge_user_profile(None, req.user_profile)
        msg_list = _messages_as_dicts(req)

        async def ndjson_bytes():
            first = json.dumps({"type": "start", "trace_id": trace_id}, ensure_ascii=False) + "\n"
            yield first.encode("utf-8")
            loop = asyncio.get_event_loop()
            it = iter(
                pipeline.ask_stream(
                    req.query,
                    merged_profile,
                    prompt_style=req.prompt_style,
                    messages=msg_list,
                )
            )

            def _next_chunk():
                try:
                    return next(it)
                except StopIteration:
                    return None

            async with semaphore:
                while True:
                    chunk = await loop.run_in_executor(None, _next_chunk)
                    if chunk is None:
                        break
                    line = json.dumps(chunk, ensure_ascii=False) + "\n"
                    yield line.encode("utf-8")

        return StreamingResponse(ndjson_bytes(), media_type="application/x-ndjson")

    return app


app = build_app()
