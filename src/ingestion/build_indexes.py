from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from src.common.models import Document
from src.ingestion.loaders import load_documents
from src.ingestion.mineru import run_mineru_batch
from src.ingestion.splitter import build_chunked_documents
from src.lc_runtime.adapters import to_lc_documents
from src.lc_runtime.embeddings import SentenceTransformerEmbeddings


def _safe_model_name() -> str:
    # Small bilingual embedding model that works for Chinese tasks.
    return "BAAI/bge-small-zh-v1.5"


def build_dense_matrix(docs: List[Document]) -> np.ndarray:
    model = SentenceTransformer(_safe_model_name())
    embeddings = model.encode([d.content for d in docs], normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def persist_index(
    index_dir: str | Path,
    docs: List[Document],
    dense: np.ndarray,
    faiss_index_type: str = "flatip",
    faiss_nlist: int = 64,
) -> None:
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    np.save(index_path / "dense.npy", dense)
    dense_index = np.asarray(dense, dtype=np.float32).copy()
    faiss.normalize_L2(dense_index)
    dim = dense_index.shape[1]

    if faiss_index_type.lower() == "ivfflat" and len(dense_index) > 1:
        nlist = max(1, min(int(faiss_nlist), len(dense_index)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(dense_index)
        index.add(dense_index)
        index_meta = {"index_type": "ivfflat", "dim": dim, "nlist": nlist}
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(dense_index)
        index_meta = {"index_type": "flatip", "dim": dim}

    faiss.write_index(index, str(index_path / "dense.faiss"))
    (index_path / "faiss_meta.json").write_text(
        json.dumps(index_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (index_path / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "content": d.content,
                        "metadata": d.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def persist_langchain_index(index_dir: str | Path, docs: List[Document]) -> None:
    index_path = Path(index_dir)
    lc_dir = index_path / "lc_faiss"
    lc_docs = to_lc_documents(docs)
    embeddings = SentenceTransformerEmbeddings(_safe_model_name())
    vs = FAISS.from_documents(lc_docs, embeddings)
    lc_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(lc_dir))


def persist_manifest(
    index_dir: str | Path,
    *,
    source_type: str,
    splitter_type: str,
    chunk_size: int,
    chunk_overlap: int,
    faiss_index_type: str,
    faiss_nlist: int,
    raw_doc_count: int,
    chunk_count: int,
) -> None:
    index_path = Path(index_dir)
    manifest = {
        "embedding_model": _safe_model_name(),
        "source_type": source_type,
        "splitter_type": splitter_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "faiss_index_type": faiss_index_type,
        "faiss_nlist": int(faiss_nlist),
        "raw_doc_count": raw_doc_count,
        "chunk_count": chunk_count,
    }
    (index_path / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_and_save_indexes(
    kb_file: str,
    index_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: str = "recursive",
    source_type: str = "jsonl",
    mineru_output_dir: str | None = None,
    mineru_auto_run: bool = False,
    mineru_input_dir: str | None = None,
    mineru_command_template: str = 'magic-pdf -p "{input_file}" -o "{output_dir}"',
    mineru_mode: str = "flash",
    mineru_token: str | None = None,
    mineru_split_pages: bool = True,
    mineru_language: str = "ch",
    mineru_timeout: int = 1200,
    faiss_index_type: str = "flatip",
    faiss_nlist: int = 64,
) -> Tuple[int, int]:
    if source_type == "mineru_markdown" and mineru_auto_run:
        if not mineru_input_dir or not mineru_output_dir:
            raise ValueError("mineru_input_dir and mineru_output_dir are required when mineru_auto_run=true")
        run_mineru_batch(
            input_dir=mineru_input_dir,
            output_dir=mineru_output_dir,
            command_template=mineru_command_template,
        )

    raw_docs = load_documents(
        source_type=source_type,
        kb_file=kb_file,
        mineru_output_dir=mineru_output_dir,
        mineru_input_dir=mineru_input_dir,
        mineru_mode=mineru_mode,
        mineru_token=mineru_token,
        mineru_split_pages=mineru_split_pages,
        mineru_language=mineru_language,
        mineru_timeout=mineru_timeout,
    )
    if not raw_docs:
        raise ValueError(
            "No documents loaded. For MinerU flow, ensure mineru_output_dir contains .md files "
            "or set mineru_auto_run=true with a valid mineru_input_dir."
        )
    chunked_docs = build_chunked_documents(
        raw_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type=splitter_type,
    )
    dense = build_dense_matrix(chunked_docs)
    persist_index(
        index_dir=index_dir,
        docs=chunked_docs,
        dense=dense,
        faiss_index_type=faiss_index_type,
        faiss_nlist=faiss_nlist,
    )
    persist_langchain_index(index_dir=index_dir, docs=chunked_docs)
    persist_manifest(
        index_dir=index_dir,
        source_type=source_type,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        faiss_index_type=faiss_index_type,
        faiss_nlist=faiss_nlist,
        raw_doc_count=len(raw_docs),
        chunk_count=len(chunked_docs),
    )
    return len(raw_docs), len(chunked_docs)


if __name__ == "__main__":
    from src.common.config import load_settings

    settings: Dict = load_settings()
    raw_count, chunk_count = build_and_save_indexes(
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
        mineru_mode=settings.get("ingestion", {}).get("mineru_mode", "flash"),
        mineru_token=settings.get("ingestion", {}).get("mineru_token"),
        mineru_split_pages=settings.get("ingestion", {}).get("mineru_split_pages", True),
        mineru_language=settings.get("ingestion", {}).get("mineru_language", "ch"),
        mineru_timeout=settings.get("ingestion", {}).get("mineru_timeout", 1200),
        faiss_index_type=settings["retrieval"].get("faiss_index_type", "flatip"),
        faiss_nlist=settings["retrieval"].get("faiss_nlist", 64),
    )
    print(f"Built index with {raw_count} raw docs and {chunk_count} chunks")
