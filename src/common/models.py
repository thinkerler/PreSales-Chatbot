from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, str]


@dataclass
class RetrievalHit:
    doc_id: str
    score: float
    content: str
    title: str
    metadata: Dict[str, str]


@dataclass
class RetrievalResult:
    query: str
    hits: List[RetrievalHit]
    debug: Dict[str, Union[float, int, str, list, dict]]
