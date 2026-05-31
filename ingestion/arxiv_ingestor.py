"""Ingest `data/arxiv/<arxiv_id>.pdf` into the cloud PageIndex API.

Vectorless. Each PDF is uploaded via `PageIndexClient.submit_document(...)`
and then polled via `is_retrieval_ready` until the server-side index
is queryable. Submissions and polls run inside a thread pool, so
ingesting N papers takes about the wall time of the slowest paper —
not the sum.

We persist only the `arxiv_id → doc_id` map locally
(`data/arxiv_index/doc_ids.json`). The tree itself lives on the
PageIndex server, keyed by that doc_id. The same map is consulted at
runtime by `tools.arxiv_tool.retrieve_for_query` to look up a paper
the user mentions in chat.

Idempotent: arxiv ids already in the mapping are skipped (we still
re-poll to confirm the server-side index is ready).
"""
from __future__ import annotations

from pathlib import Path

from config.settings import PAGEINDEX_PARALLEL_WORKERS
from graph.state import IngestionResult
from tools import arxiv_tool


def ingest_folder(folder: str) -> list[IngestionResult]:
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"ArXiv folder not found: {folder}")
        return []
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder}")
        return []

    pairs: list[tuple[str, str]] = []
    pre_cached: list[IngestionResult] = []
    for pdf_path in pdf_files:
        arxiv_id = pdf_path.stem
        if arxiv_tool.get_doc_id(arxiv_id):
            print(f"  ↳ Already mapped: {pdf_path.name} (doc_id={arxiv_tool.get_doc_id(arxiv_id)})")
        pairs.append((arxiv_id, str(pdf_path)))

    print(
        f"Found {len(pdf_files)} arxiv PDFs — submitting in parallel "
        f"(max {PAGEINDEX_PARALLEL_WORKERS} workers)..."
    )
    triples = arxiv_tool.index_pdfs_parallel(pairs)

    results: list[IngestionResult] = list(pre_cached)
    for arxiv_id, doc_id, status in triples:
        success = status in ("ready", "cached")
        if status == "ready":
            print(f"  ↳ Indexed {arxiv_id}.pdf → doc_id={doc_id}")
        elif status == "cached":
            print(f"  ↳ Cached  {arxiv_id}.pdf → doc_id={doc_id}")
        elif status == "cached_not_ready":
            print(f"  ↳ Cached  {arxiv_id}.pdf → doc_id={doc_id} (server-side index not yet ready)")
        elif status == "timeout":
            print(f"  ↳ Timeout {arxiv_id}.pdf → doc_id={doc_id} (still processing — retry later)")
        else:
            print(f"  ↳ FAILED  {arxiv_id}.pdf — {status}")
        results.append(
            IngestionResult(
                source=arxiv_id,
                total_chunks=1 if success else 0,
                success=success,
                error=None if success else status,
            )
        )
    return results
