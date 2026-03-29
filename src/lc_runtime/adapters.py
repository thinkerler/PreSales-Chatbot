from __future__ import annotations

from typing import List

from langchain_core.documents import Document as LCDocument

from src.common.models import Document, RetrievalHit


def to_lc_documents(docs: List[Document]) -> List[LCDocument]:
    results: List[LCDocument] = []
    for doc in docs:
        metadata = dict(doc.metadata)
        metadata["doc_id"] = doc.doc_id
        metadata["title"] = doc.title
        results.append(LCDocument(page_content=doc.content, metadata=metadata))
    return results


def hits_from_lc_documents(docs: List[LCDocument], scores: List[float]) -> List[RetrievalHit]:
    hits: List[RetrievalHit] = []
    for doc, score in zip(docs, scores):
        hits.append(
            RetrievalHit(
                doc_id=str(doc.metadata.get("doc_id", "")),
                score=float(score),
                content=doc.page_content,
                title=str(doc.metadata.get("title", "")),
                metadata={k: str(v) for k, v in doc.metadata.items()},
            )
        )
    return hits
