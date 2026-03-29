from __future__ import annotations

from typing import List, Literal

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from src.common.models import Document


SplitterType = Literal["sliding_window", "character", "recursive"]


def _sliding_window_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _langchain_character_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def _langchain_recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: SplitterType = "recursive",
) -> List[str]:
    if splitter_type == "sliding_window":
        return _sliding_window_chunk(text, chunk_size, chunk_overlap)
    if splitter_type == "character":
        return _langchain_character_chunk(text, chunk_size, chunk_overlap)
    return _langchain_recursive_chunk(text, chunk_size, chunk_overlap)


def build_chunked_documents(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: SplitterType = "recursive",
) -> List[Document]:
    results: List[Document] = []
    for doc in docs:
        chunks = chunk_text(
            doc.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type,
        )
        if len(chunks) == 1:
            results.append(doc)
            continue
        for i, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = str(i)
            metadata["splitter_type"] = splitter_type
            results.append(
                Document(
                    doc_id=f"{doc.doc_id}#c{i}",
                    title=doc.title,
                    content=chunk,
                    metadata=metadata,
                )
            )
    return results