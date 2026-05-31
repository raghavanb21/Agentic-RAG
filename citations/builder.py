"""Standardized citation builder.

Assembles `Citation` Pydantic models from RetrievedChunk metadata. The
Citation schema is the single source of truth for source metadata; raw
dicts are never used for citations outside the chromadb_tool boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from graph.state import Citation, RetrievedChunk


def _coerce_page(value) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def build_citation(chunk: RetrievedChunk) -> Citation:
    md = dict(chunk.metadata or {})
    source_type = md.get("source_type") or chunk.citation.source_type
    if source_type not in ("pdf", "arxiv", "web", "blog"):
        source_type = "pdf"
    payload = {
        "source_type": source_type,
        "title": md.get("title") or md.get("source_file") or chunk.citation.title or "Unknown",
        "author": md.get("authors") or md.get("author") or chunk.citation.author,
        "url": md.get("url") or chunk.citation.url,
        "page": _coerce_page(md.get("page", chunk.citation.page)),
        "section": md.get("section") or chunk.citation.section,
        "published_date": md.get("published_date") or chunk.citation.published_date,
        "retrieved_at": md.get("retrieved_at")
        or chunk.citation.retrieved_at
        or datetime.now(timezone.utc).isoformat(),
        "chunk_text": chunk.content,
    }
    return Citation.model_validate(payload)


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [build_citation(c) for c in chunks]
