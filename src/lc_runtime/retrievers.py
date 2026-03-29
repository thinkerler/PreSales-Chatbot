from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from sentence_transformers import CrossEncoder

from src.common.models import RetrievalResult
from src.lc_runtime.adapters import hits_from_lc_documents
from src.lc_runtime.embeddings import SentenceTransformerEmbeddings


@dataclass
class HybridRetrieverConfig:
    index_dir: str
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 6
    rrf_k: int = 60
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"


def _doc_key(doc: LCDocument) -> str:
    return str(doc.metadata.get("doc_id", ""))


def _rrf_fuse(
    dense_docs: List[LCDocument],
    sparse_docs: List[LCDocument],
    rrf_k: int,
) -> List[Tuple[LCDocument, float]]:
    dense_map = {_doc_key(doc): doc for doc in dense_docs}
    sparse_map = {_doc_key(doc): doc for doc in sparse_docs}
    doc_map = {**dense_map, **sparse_map}

    scores: Dict[str, float] = {}
    for rank, doc in enumerate(dense_docs):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc in enumerate(sparse_docs):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in ranked if key in doc_map]


class LangChainHybridRetriever:
    def __init__(self, config: HybridRetrieverConfig) -> None:
        self.config = config
        self.use_reranker = config.use_reranker
        self.index_dir = Path(config.index_dir)
        embeddings = SentenceTransformerEmbeddings(config.embedding_model)
        lc_dir = self.index_dir / "lc_faiss"
        self.vectorstore = FAISS.load_local(
            str(lc_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        self.sparse_retriever = self._build_sparse_retriever()
        self.reranker = CrossEncoder(config.reranker_model) if config.use_reranker else None
        self._last_debug: Dict[str, object] = {}

    def _build_sparse_retriever(self) -> BM25Retriever:
        docs_path = self.index_dir / "docs.jsonl"
        docs: List[LCDocument] = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                metadata = dict(item.get("metadata", {}))
                metadata["doc_id"] = item.get("doc_id", "")
                metadata["title"] = item.get("title", "")
                docs.append(
                    LCDocument(
                        page_content=item.get("content", ""),
                        metadata=metadata,
                    )
                )
        sparse = BM25Retriever.from_documents(docs)
        sparse.k = self.config.top_k_sparse
        return sparse

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[LCDocument, float]],
    ) -> List[Tuple[LCDocument, float]]:
        if not self.reranker or not candidates:
            return candidates
        pairs = [[query, doc.page_content] for doc, _ in candidates]
        scores = self.reranker.predict(pairs)
        reranked = [(doc, float(score)) for (doc, _), score in zip(candidates, scores)]
        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def retrieve(self, query: str) -> RetrievalResult:
        t0 = time.perf_counter()
        dense_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.config.top_k_dense,
        )
        t1 = time.perf_counter()
        sparse_docs = self.sparse_retriever.invoke(query)
        t2 = time.perf_counter()

        dense_docs = [doc for doc, _ in dense_with_scores]
        dense_scores = [float(score) for _, score in dense_with_scores]
        fused = _rrf_fuse(dense_docs, sparse_docs, self.config.rrf_k)[: self.config.top_k_final * 2]
        t3 = time.perf_counter()
        final_ranked = self._rerank(query, fused)[: self.config.top_k_final]
        t4 = time.perf_counter()

        hits = hits_from_lc_documents(
            [doc for doc, _ in final_ranked],
            [score for _, score in final_ranked],
        )
        self._last_debug = {
            "dense_candidates": len(dense_docs),
            "sparse_candidates": len(sparse_docs),
            "fused_candidates": len(fused),
            "dense_doc_ids": [str(d.metadata.get("doc_id", "")) for d in dense_docs],
            "sparse_doc_ids": [str(d.metadata.get("doc_id", "")) for d in sparse_docs],
            "fused_doc_ids": [str(d.metadata.get("doc_id", "")) for d, _ in fused],
            "dense_scores": dense_scores,
            "stage_latency_ms": {
                "dense": round((t1 - t0) * 1000, 2),
                "sparse": round((t2 - t1) * 1000, 2),
                "fusion": round((t3 - t2) * 1000, 2),
                "rerank": round((t4 - t3) * 1000, 2),
            },
        }
        return RetrievalResult(query=query, hits=hits, debug=self._last_debug)
