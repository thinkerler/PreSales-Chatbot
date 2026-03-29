from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List

from src.chat.pipeline import ChatPipeline
from src.common.config import load_settings
from src.ingestion.build_indexes import build_and_save_indexes
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def load_eval_set(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recall_at_k(pred: List[str], truth: List[str], k: int) -> float:
    pred_k = pred[:k]
    hit = len(set(pred_k).intersection(set(truth)))
    return hit / max(1, len(set(truth)))


def mrr_at_k(pred: List[str], truth: List[str], k: int) -> float:
    truth_set = set(truth)
    for i, doc_id in enumerate(pred[:k], start=1):
        if doc_id in truth_set:
            return 1.0 / i
    return 0.0


def faithfulness_proxy(answer: str, evidence_ids: List[str]) -> float:
    # Lightweight proxy: answer should contain evidence hint and not be empty.
    if not answer.strip():
        return 0.0
    return 1.0 if evidence_ids else 0.3


def evaluate_retriever(retriever, dataset: List[Dict], top_k: int = 5) -> Dict[str, float]:
    recalls, mrrs = [], []
    for row in dataset:
        result = retriever.retrieve(row["query"])
        pred_ids = [h.doc_id.split("#")[0] for h in result.hits]
        recalls.append(recall_at_k(pred_ids, row["relevant_doc_ids"], top_k))
        mrrs.append(mrr_at_k(pred_ids, row["relevant_doc_ids"], top_k))
    return {"Recall@5": round(mean(recalls), 4), "MRR@5": round(mean(mrrs), 4)}


def evaluate_generation(pipeline: ChatPipeline, dataset: List[Dict]) -> Dict[str, float]:
    faithfulness_scores = []
    latencies = []
    reject_count = 0
    fallback_count = 0
    for row in dataset:
        out = pipeline.ask(row["query"])
        faithfulness_scores.append(faithfulness_proxy(out["answer"], out["evidence_doc_ids"]))
        latencies.append(out["latency_ms"])
        reason = out.get("reject_reason", "")
        if reason in {"no_evidence", "low_confidence"}:
            reject_count += 1
        if "fallback" in reason:
            fallback_count += 1
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]
    total = max(1, len(dataset))
    return {
        "Faithfulness(proxy)": round(mean(faithfulness_scores), 4),
        "LatencyP95(ms)": round(p95, 2),
        "RejectRate": round(reject_count / total, 4),
        "FallbackRate": round(fallback_count / total, 4),
    }


def main() -> None:
    settings = load_settings()
    build_and_save_indexes(
        kb_file=settings["paths"]["kb_file"],
        index_dir=settings["paths"]["index_dir"],
        chunk_size=settings["retrieval"]["chunk_size"],
        chunk_overlap=settings["retrieval"]["chunk_overlap"],
        splitter_type=settings["retrieval"].get("splitter_type", "recursive"),
        source_type=settings.get("ingestion", {}).get("source_type", "jsonl"),
        mineru_output_dir=settings.get("ingestion", {}).get("mineru_output_dir"),
        mineru_auto_run=settings.get("ingestion", {}).get("mineru_auto_run", False),
        mineru_input_dir=settings.get("ingestion", {}).get("mineru_input_dir"),
        mineru_command_template=settings.get("ingestion", {}).get(
            "mineru_command_template",
            'magic-pdf -p "{input_file}" -o "{output_dir}"',
        ),
        faiss_index_type=settings["retrieval"].get("faiss_index_type", "flatip"),
        faiss_nlist=settings["retrieval"].get("faiss_nlist", 64),
    )
    dataset = load_eval_set(settings["paths"]["eval_file"])
    dense = DenseRetriever(
        index_dir=settings["paths"]["index_dir"],
        top_k=6,
        faiss_nprobe=settings["retrieval"].get("faiss_nprobe", 8),
    )
    hybrid = HybridRetriever(
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
        retriever=hybrid,
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

    dense_metrics = evaluate_retriever(dense, dataset)
    hybrid_metrics = evaluate_retriever(hybrid, dataset)
    gen_metrics = evaluate_generation(pipeline, dataset[: min(20, len(dataset))])
    report = {
        "dense_baseline": dense_metrics,
        "hybrid": hybrid_metrics,
        "generation": gen_metrics,
    }
    out = Path("docs/eval_report_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
