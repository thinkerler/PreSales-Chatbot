from __future__ import annotations

import sys
from pathlib import Path

# Allow running via "python scripts/run_mineru_pipeline.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_settings
from src.ingestion.build_indexes import build_and_save_indexes


def main() -> None:
    settings = load_settings()
    ingestion = settings.get("ingestion", {})
    retrieval = settings["retrieval"]
    paths = settings["paths"]

    raw_count, chunk_count = build_and_save_indexes(
        kb_file=paths["kb_file"],
        index_dir=paths["index_dir"],
        chunk_size=retrieval["chunk_size"],
        chunk_overlap=retrieval["chunk_overlap"],
        splitter_type=retrieval.get("splitter_type", "recursive"),
        source_type=ingestion.get("source_type", "mineru_markdown"),
        mineru_output_dir=ingestion.get("mineru_output_dir"),
        mineru_auto_run=ingestion.get("mineru_auto_run", False),
        mineru_input_dir=ingestion.get("mineru_input_dir"),
        mineru_command_template=ingestion.get(
            "mineru_command_template",
            'magic-pdf -p "{input_file}" -o "{output_dir}"',
        ),
        mineru_mode=ingestion.get("mineru_mode", "flash"),
        mineru_token=ingestion.get("mineru_token"),
        mineru_split_pages=ingestion.get("mineru_split_pages", True),
        mineru_language=ingestion.get("mineru_language", "ch"),
        mineru_timeout=ingestion.get("mineru_timeout", 1200),
        faiss_index_type=retrieval.get("faiss_index_type", "flatip"),
        faiss_nlist=retrieval.get("faiss_nlist", 64),
    )
    print(f"MinerU pipeline done. raw_docs={raw_count}, chunked_docs={chunk_count}")


if __name__ == "__main__":
    main()
