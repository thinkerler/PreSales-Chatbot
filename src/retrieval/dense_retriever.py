from __future__ import annotations

from typing import List

from langchain_community.vectorstores import FAISS

from src.common.models import RetrievalHit, RetrievalResult
from src.lc_runtime.embeddings import SentenceTransformerEmbeddings


class DenseRetriever:
    def __init__(self, index_dir: str, top_k: int = 6, faiss_nprobe: int = 8) -> None:
        del faiss_nprobe  # kept for API compatibility
        self.top_k = top_k
        embeddings = SentenceTransformerEmbeddings("BAAI/bge-small-zh-v1.5")
        self.vectorstore = FAISS.load_local(
            f"{index_dir}/lc_faiss",
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str) -> RetrievalResult:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        hits: List[RetrievalHit] = []
        for doc, score in docs_with_scores:
            hits.append(
                RetrievalHit(
                    doc_id=str(doc.metadata.get("doc_id", "")),
                    score=float(score),
                    content=doc.page_content,
                    title=str(doc.metadata.get("title", "")),
                    metadata={k: str(v) for k, v in doc.metadata.items()},
                )
            )
        return RetrievalResult(query=query, hits=hits, debug={"dense_only": 1.0})
