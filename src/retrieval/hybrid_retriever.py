from __future__ import annotations

from src.lc_runtime.retrievers import HybridRetrieverConfig, LangChainHybridRetriever


class HybridRetriever(LangChainHybridRetriever):
    def __init__(
        self,
        index_dir: str,
        top_k_dense: int = 8,
        top_k_sparse: int = 8,
        top_k_final: int = 6,
        rrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
        faiss_nprobe: int = 8,
    ) -> None:
        del faiss_nprobe  # kept for backward-compatible constructor signature
        super().__init__(
            HybridRetrieverConfig(
                index_dir=index_dir,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                top_k_final=top_k_final,
                rrf_k=rrf_k,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
            )
        )
