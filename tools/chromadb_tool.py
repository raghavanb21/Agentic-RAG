"""ChromaDB read/write for `pdf_text` and `web_cache`.

The ChromaDB client and the bge-base embedding model are module-level
singletons — they are created exactly once when this module is first
imported.

ArXiv retrieval is vectorless (PageIndex JSON trees, see `tools/arxiv_tool.py`)
and does NOT use ChromaDB.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import (
    CHROMA_PERSIST_DIR,
    EMBED_MODEL,
    PDF_COLLECTION,
    WEB_COLLECTION,
)
from graph.state import Citation, RetrievedChunk

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

_CHROMA_CLIENT = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False, allow_reset=False),
)

_EMBEDDER = SentenceTransformer(EMBED_MODEL)

_COLLECTIONS = {
    PDF_COLLECTION: _CHROMA_CLIENT.get_or_create_collection(
        name=PDF_COLLECTION, metadata={"hnsw:space": "cosine"}
    ),
    WEB_COLLECTION: _CHROMA_CLIENT.get_or_create_collection(
        name=WEB_COLLECTION, metadata={"hnsw:space": "cosine"}
    ),
}


def get_collection(name: str):
    if name not in _COLLECTIONS:
        raise ValueError(f"Unknown collection: {name}")
    return _COLLECTIONS[name]


def embed_query(text: str) -> np.ndarray:
    return _EMBEDDER.encode(text, normalize_embeddings=True, convert_to_numpy=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    return _EMBEDDER.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )


def _metadata_to_citation(metadata: dict, chunk_text: str) -> Citation:
    source_type = metadata.get("source_type", "pdf")
    if source_type not in ("pdf", "arxiv", "web", "blog"):
        source_type = "pdf"
    page_val = metadata.get("page")
    page_int: Optional[int] = None
    if isinstance(page_val, (int, float)):
        page_int = int(page_val)
    return Citation(
        source_type=source_type,
        title=metadata.get("title") or metadata.get("source_file") or "Unknown",
        author=metadata.get("authors") or metadata.get("author"),
        url=metadata.get("url"),
        page=page_int,
        section=metadata.get("section"),
        published_date=metadata.get("published_date"),
        retrieved_at=metadata.get("retrieved_at", datetime.now(timezone.utc).isoformat()),
        chunk_text=chunk_text,
    )


def search_collection(name: str, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
    collection = get_collection(name)
    if collection.count() == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[RetrievedChunk] = []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    for content, metadata, distance in zip(docs, metas, dists):
        metadata = metadata or {}
        similarity = 1.0 - float(distance)
        out.append(
            RetrievedChunk(
                content=content,
                metadata=dict(metadata),
                similarity=similarity,
                citation=_metadata_to_citation(dict(metadata), content),
            )
        )
    return out


def store_chunks(
    name: str,
    contents: list[str],
    metadatas: list[dict],
    ids: list[str],
    embeddings: Optional[list[list[float]]] = None,
) -> None:
    if not contents:
        return
    collection = get_collection(name)
    if embeddings is None:
        embeddings = embed_batch(contents).tolist()
    collection.upsert(
        ids=ids,
        documents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


def get_by_url(name: str, url: str, k: int) -> list[RetrievedChunk]:
    """Return cached chunks whose `metadata.url` exactly matches `url`."""
    collection = get_collection(name)
    if collection.count() == 0:
        return []
    try:
        results = collection.get(
            where={"url": url},
            limit=k,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []
    out: list[RetrievedChunk] = []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    for content, metadata in zip(docs, metas):
        metadata = metadata or {}
        out.append(
            RetrievedChunk(
                content=content,
                metadata=dict(metadata),
                similarity=1.0,
                citation=_metadata_to_citation(dict(metadata), content),
            )
        )
    return out


def has_source(name: str, source_key: str, source_value: str) -> bool:
    collection = get_collection(name)
    if collection.count() == 0:
        return False
    try:
        result = collection.get(where={source_key: source_value}, limit=1)
        return bool(result.get("ids"))
    except Exception:
        return False
