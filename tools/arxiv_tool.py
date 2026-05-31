"""ArXiv tool — wraps the **cloud** PageIndex API.

Trees and page text live on VectifyAI's servers, keyed by `doc_id`.
The only thing this project persists locally is a JSON map of
`arxiv_id → doc_id` at `data/arxiv_index/doc_ids.json`, so we can
re-use the server-side index across runs without re-uploading.

API surface (all sync from the caller's perspective):
  - `submit_pdf(pdf_path)` — uploads a PDF, returns the new doc_id
    immediately. Server-side indexing is still in progress when this
    returns.
  - `wait_ready(doc_id)` — polls `is_retrieval_ready` until the
    server-side index is queryable or the timeout fires.
  - `index_pdfs_parallel(pairs)` — submits and polls many PDFs
    concurrently via a thread pool. Total wall time is bounded by
    the slowest paper, not the sum.
  - `fetch_and_index(arxiv_id)` — runtime path: downloads via the
    `arxiv` Python library, submits, polls, registers the mapping.
  - `retrieve_for_query(arxiv_id, question)` — submits a query
    against the server-side agent and polls `get_retrieval` until
    completed, then turns the `retrieved_nodes` into
    `RetrievedChunk` Pydantic models.

The `PageIndexClient` instance and the in-memory mirror of the
`arxiv_id → doc_id` mapping are module-level singletons. Writes to
the on-disk mapping are guarded by a threading lock so the parallel
ingestion path is safe.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import arxiv
from pageindex import PageIndexClient


def _log(msg: str) -> None:
    """Print to stderr so messages are visible even when stdout is captured."""
    print(f"[arxiv_tool] {msg}", file=sys.stderr, flush=True)

from config.settings import (
    PAGEINDEX_API_KEY,
    PAGEINDEX_INDEX_MAX_WAIT_S,
    PAGEINDEX_MAPPING_PATH,
    PAGEINDEX_PARALLEL_WORKERS,
    PAGEINDEX_POLL_INTERVAL_S,
    PAGEINDEX_QUERY_MAX_WAIT_S,
    PAGEINDEX_QUERY_POLL_INTERVAL_S,
    TOP_K_CHUNKS,
)
from graph.state import Citation, RetrievedChunk


_CLIENT = (
    PageIndexClient(api_key=PAGEINDEX_API_KEY) if PAGEINDEX_API_KEY else PageIndexClient()
)

_MAPPING_PATH = Path(PAGEINDEX_MAPPING_PATH)
_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
_MAPPING_LOCK = threading.Lock()


def _load_mapping() -> dict[str, dict]:
    """Load doc_ids.json. Accepts both the legacy ``{arxiv_id: "doc_id"}``
    string format and the new ``{arxiv_id: {doc_id, title, authors, published_date}}``
    dict format. Legacy entries are upgraded in-memory.
    """
    if not _MAPPING_PATH.exists():
        return {}
    try:
        data = json.loads(_MAPPING_PATH.read_text())
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for k, v in data.items():
        if isinstance(v, str) and v:
            out[str(k)] = {"doc_id": v}
        elif isinstance(v, dict) and v.get("doc_id"):
            out[str(k)] = {
                "doc_id": str(v["doc_id"]),
                "title": v.get("title") or None,
                "authors": list(v.get("authors") or []),
                "published_date": v.get("published_date") or "",
            }
    return out


_MAPPING: dict[str, dict] = _load_mapping()


def _save_mapping_locked() -> None:
    _MAPPING_PATH.write_text(json.dumps(_MAPPING, indent=2, sort_keys=True))


def get_doc_id(arxiv_id: str) -> Optional[str]:
    entry = _MAPPING.get(arxiv_id)
    if isinstance(entry, dict):
        return entry.get("doc_id")
    return None


def get_cached_metadata(arxiv_id: str) -> Optional[tuple[str, list[str], str]]:
    """Return (title, authors, published_date) from doc_ids.json if cached. None otherwise."""
    entry = _MAPPING.get(arxiv_id)
    if not isinstance(entry, dict):
        return None
    title = entry.get("title")
    if not title:
        return None
    return title, list(entry.get("authors") or []), entry.get("published_date") or ""


def _register(
    arxiv_id: str,
    doc_id: str,
    *,
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    published_date: Optional[str] = None,
) -> None:
    """Persist doc_id and (optionally) arxiv metadata. Existing fields are preserved
    so a metadata-only update doesn't wipe the doc_id, and vice versa.
    """
    with _MAPPING_LOCK:
        entry = dict(_MAPPING.get(arxiv_id) or {})
        if doc_id:
            entry["doc_id"] = doc_id
        if title:
            entry["title"] = title
        if authors:
            entry["authors"] = list(authors)
        if published_date:
            entry["published_date"] = published_date
        _MAPPING[arxiv_id] = entry
        _save_mapping_locked()


def submit_pdf(pdf_path: str) -> Optional[str]:
    """Upload a PDF. Returns doc_id immediately; indexing is async server-side."""
    try:
        info = _CLIENT.submit_document(pdf_path)
    except Exception as e:
        _log(f"submit_document FAILED for {pdf_path}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None
    if not isinstance(info, dict):
        _log(f"submit_document returned non-dict for {pdf_path}: {info!r}")
        return None
    doc_id = info.get("doc_id")
    if not doc_id:
        _log(f"submit_document returned dict with no doc_id: {info!r}")
        return None
    return str(doc_id)


def wait_ready(
    doc_id: str,
    max_wait_s: int = PAGEINDEX_INDEX_MAX_WAIT_S,
    poll_interval_s: float = PAGEINDEX_POLL_INTERVAL_S,
) -> bool:
    """Poll `is_retrieval_ready` until the server-side index is queryable.

    Short-circuits on the second-and-onward call for the same doc_id within a
    process: once a doc has been confirmed ready, it stays ready, so we don't
    burn another HTTP round-trip on every user turn.
    """
    with _KNOWN_READY_LOCK:
        if doc_id in _KNOWN_READY:
            return True
    deadline = time.time() + max_wait_s
    attempts = 0
    while time.time() < deadline:
        try:
            ready = _CLIENT.is_retrieval_ready(doc_id)
        except Exception as e:
            _log(f"is_retrieval_ready raised for doc_id={doc_id}: {e}")
            return False
        if ready:
            with _KNOWN_READY_LOCK:
                _KNOWN_READY.add(doc_id)
            return True
        attempts += 1
        if attempts == 1 or attempts % 6 == 0:
            _log(f"doc_id={doc_id} still indexing (attempt {attempts})…")
        time.sleep(poll_interval_s)
    _log(f"doc_id={doc_id} did not become ready within {max_wait_s}s")
    return False


def _index_one(arxiv_id: str, pdf_path: str) -> tuple[str, Optional[str], str]:
    """Submit one PDF and poll until ready. Returns (arxiv_id, doc_id, status)."""
    existing = get_doc_id(arxiv_id)
    if existing:
        if wait_ready(existing, max_wait_s=PAGEINDEX_INDEX_MAX_WAIT_S):
            # Opportunistically fetch metadata if we haven't already cached it,
            # so runtime queries never need to hit arxiv.org.
            if get_cached_metadata(arxiv_id) is None:
                _fetch_arxiv_metadata(arxiv_id)
            return arxiv_id, existing, "cached"
        return arxiv_id, existing, "cached_not_ready"

    doc_id = submit_pdf(pdf_path)
    if not doc_id:
        return arxiv_id, None, "submit_failed"

    # Persist immediately so a crash mid-poll doesn't lose the doc_id.
    _register(arxiv_id, doc_id)

    # Pre-fetch metadata at ingestion. The arxiv client's 3 s delay paces this
    # safely; failures degrade silently and will retry on first user query.
    _fetch_arxiv_metadata(arxiv_id)

    if wait_ready(doc_id):
        return arxiv_id, doc_id, "ready"
    return arxiv_id, doc_id, "timeout"


def index_pdfs_parallel(
    pairs: list[tuple[str, str]],
    max_workers: int = PAGEINDEX_PARALLEL_WORKERS,
) -> list[tuple[str, Optional[str], str]]:
    """Submit & poll many PDFs concurrently. Wall time ≈ slowest paper, not sum."""
    if not pairs:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda p: _index_one(p[0], p[1]), pairs))


# Default arxiv API client (matches `arxiv.Client()` from the library's example
# code: page_size=100, delay_seconds=3.0, num_retries=3). Module-level singleton
# so the 3-second spacing between requests is enforced across the whole process.
_ARXIV_CLIENT = arxiv.Client()

# Per-process cache of arxiv.Result by id, so we only hit the API once per paper
# even though we use the result twice (PDF download URL + title/authors/published).
_ARXIV_RESULT_CACHE: dict[str, arxiv.Result] = {}
_ARXIV_LOCK = threading.Lock()

# Session-level set of doc_ids we have already confirmed queryable. Saves up to
# 90 s of redundant `is_retrieval_ready` polling on every user turn after the
# first one.
_KNOWN_READY: set[str] = set()
_KNOWN_READY_LOCK = threading.Lock()

# Session-level set of arxiv_ids whose background metadata fetch is in flight,
# so duplicate user queries don't spawn parallel threads hammering arxiv.
_METADATA_IN_FLIGHT: set[str] = set()
_METADATA_INFLIGHT_LOCK = threading.Lock()


def _arxiv_result(arxiv_id: str) -> Optional[arxiv.Result]:
    """Fetch a single arxiv paper by id using the canonical lib pattern:

        client = arxiv.Client()
        search_by_id = arxiv.Search(id_list=["2508.10146"])
        first_result = next(client.results(search_by_id))

    Returns None if the id doesn't resolve (StopIteration) or the API call
    is throttled/errors. Caller (`fetch_and_index`) then submits the PDF
    to PageIndex and queries it for the answer context.
    """
    with _ARXIV_LOCK:
        cached = _ARXIV_RESULT_CACHE.get(arxiv_id)
    if cached is not None:
        return cached

    # Construct the default API client (reused module-level singleton).
    client = _ARXIV_CLIENT
    search_by_id = arxiv.Search(id_list=[arxiv_id])
    try:
        # Reuse client to fetch the paper.
        first_result = next(client.results(search_by_id))
    except StopIteration:
        _log(f"arxiv.Client found no paper with id={arxiv_id}")
        return None
    except Exception as e:
        # 429/503 from arxiv just mean we're throttled — log one line, no traceback.
        msg = str(e)
        if "429" in msg or "503" in msg or "HTTPError" in msg:
            _log(f"arxiv API throttled for id={arxiv_id} ({msg.splitlines()[0][:120]})")
        else:
            _log(f"arxiv.Client lookup FAILED for id={arxiv_id}: {e}")
            traceback.print_exc(file=sys.stderr)
        return None
    with _ARXIV_LOCK:
        _ARXIV_RESULT_CACHE[arxiv_id] = first_result
    return first_result


def _download_arxiv_pdf(arxiv_id: str) -> Optional[str]:
    result = _arxiv_result(arxiv_id)
    if result is None:
        return None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="arxiv_")
        path = result.download_pdf(dirpath=tmp_dir, filename=f"{arxiv_id}.pdf")
        _log(f"downloaded arxiv {arxiv_id} → {path}")
        return path
    except Exception as e:
        _log(f"download_pdf FAILED for id={arxiv_id}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None


def _fetch_arxiv_metadata(arxiv_id: str) -> tuple[str, list[str], str]:
    """Title/authors/published for a paper. Cache-first; persists on first success
    so future calls never hit arxiv.org again, regardless of throttling. This
    call MAY block on the arxiv API; for the user-facing retrieval path use
    :func:`metadata_for_retrieval` instead, which never blocks.
    """
    cached = get_cached_metadata(arxiv_id)
    if cached is not None:
        return cached
    result = _arxiv_result(arxiv_id)
    if result is None:
        return arxiv_id, [], ""
    title = result.title or arxiv_id
    authors = [a.name for a in (result.authors or [])]
    published = (
        result.published.isoformat() if getattr(result, "published", None) else ""
    )
    doc_id = get_doc_id(arxiv_id)
    if doc_id:
        _register(arxiv_id, doc_id, title=title, authors=authors, published_date=published)
    return title, authors, published


def _fetch_metadata_async(arxiv_id: str) -> None:
    """Spawn one background thread to fetch+persist metadata. Deduplicated by id."""
    with _METADATA_INFLIGHT_LOCK:
        if arxiv_id in _METADATA_IN_FLIGHT:
            return
        _METADATA_IN_FLIGHT.add(arxiv_id)

    def _go() -> None:
        try:
            _fetch_arxiv_metadata(arxiv_id)
        except Exception as e:
            _log(f"background metadata fetch for {arxiv_id} failed: {e}")
        finally:
            with _METADATA_INFLIGHT_LOCK:
                _METADATA_IN_FLIGHT.discard(arxiv_id)

    threading.Thread(target=_go, daemon=True).start()


def metadata_for_retrieval(arxiv_id: str) -> tuple[str, list[str], str]:
    """Non-blocking metadata for the user-facing retrieval path.

    Cache hit → return cached values immediately.
    Cache miss → spawn a background thread to populate the cache for the NEXT
    query and return ``(arxiv_id, [], "")`` right now so the user doesn't wait
    on arxiv.org under throttle. The Citation Pydantic model tolerates the
    empty author / published fields.
    """
    cached = get_cached_metadata(arxiv_id)
    if cached is not None:
        return cached
    _fetch_metadata_async(arxiv_id)
    return arxiv_id, [], ""


def fetch_and_index(arxiv_id: str) -> Optional[str]:
    """Runtime path: ensure a doc_id exists for this arxiv_id.

    If the arxiv_id is already in our local `doc_ids.json` mapping, we
    just confirm the server-side index is ready and return.

    Otherwise we fetch the paper via the `arxiv` Python library, submit
    it to PageIndex, persist the new `arxiv_id → doc_id` mapping (so
    future runs reuse it), then poll until the server says the index
    is queryable.

    Every failure point prints to stderr so the user can see *why* a
    paper couldn't be retrieved instead of getting a silent fallback
    to `direct_llm`.
    """
    existing = get_doc_id(arxiv_id)
    if existing:
        _log(f"arxiv_id={arxiv_id} already mapped → doc_id={existing}; polling readiness…")
        if wait_ready(existing, max_wait_s=PAGEINDEX_INDEX_MAX_WAIT_S):
            return existing
        _log(f"arxiv_id={arxiv_id} doc_id={existing} never became ready")
        return None

    _log(f"arxiv_id={arxiv_id} not in mapping — fetching via arxiv library…")
    pdf_path = _download_arxiv_pdf(arxiv_id)
    if not pdf_path:
        return None

    # `_download_arxiv_pdf` populated the result cache — capture metadata now so
    # we can persist it alongside doc_id in a single mapping write.
    md_title: Optional[str] = None
    md_authors: Optional[list[str]] = None
    md_published: Optional[str] = None
    with _ARXIV_LOCK:
        cached_result = _ARXIV_RESULT_CACHE.get(arxiv_id)
    if cached_result is not None:
        md_title = cached_result.title or arxiv_id
        md_authors = [a.name for a in (cached_result.authors or [])]
        md_published = (
            cached_result.published.isoformat()
            if getattr(cached_result, "published", None)
            else ""
        )

    _log(f"arxiv_id={arxiv_id} submitting to PageIndex…")
    try:
        doc_id = submit_pdf(pdf_path)
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass
    if not doc_id:
        _log(f"arxiv_id={arxiv_id} submit_pdf returned no doc_id")
        return None

    _register(
        arxiv_id,
        doc_id,
        title=md_title,
        authors=md_authors,
        published_date=md_published,
    )
    _log(f"arxiv_id={arxiv_id} → doc_id={doc_id} (+metadata) persisted; waiting for indexing…")
    if wait_ready(doc_id, max_wait_s=PAGEINDEX_INDEX_MAX_WAIT_S):
        _log(f"arxiv_id={arxiv_id} doc_id={doc_id} READY")
        return doc_id
    _log(f"arxiv_id={arxiv_id} doc_id={doc_id} indexing timeout")
    return None


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


def _wait_retrieval(retrieval_id: str) -> Optional[dict]:
    deadline = time.time() + PAGEINDEX_QUERY_MAX_WAIT_S
    while time.time() < deadline:
        try:
            retrieval = _CLIENT.get_retrieval(retrieval_id)
        except Exception:
            return None
        if not isinstance(retrieval, dict):
            return None
        status = retrieval.get("status")
        if status == "completed":
            return retrieval
        if status == "failed":
            return None
        time.sleep(PAGEINDEX_QUERY_POLL_INTERVAL_S)
    return None


def _nodes_to_chunks(
    nodes: list[dict],
    arxiv_id: str,
    title: str,
    authors: list[str],
    published: str,
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    authors_joined = ", ".join(authors) if authors else "unknown"
    url = f"https://arxiv.org/abs/{arxiv_id}"
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id", "")
        node_title = node.get("title") or node.get("node_title") or ""
        page = _coerce_page(node.get("start_index") or node.get("page"))

        groups = node.get("relevant_contents", []) or []
        for group in groups:
            if not isinstance(group, list):
                group = [group]
            for item in group:
                if not isinstance(item, dict):
                    continue
                content = item.get("relevant_content") or item.get("content")
                if not content:
                    continue
                citation = Citation(
                    source_type="arxiv",
                    title=title,
                    author=authors_joined if authors else None,
                    url=url,
                    page=page,
                    section=node_title or None,
                    published_date=published or None,
                    retrieved_at=retrieved_at,
                    chunk_text=content[:1000],
                )
                chunks.append(
                    RetrievedChunk(
                        content=content,
                        metadata={
                            "source_type": "arxiv",
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "authors": authors_joined,
                            "node_id": node_id,
                            "section": node_title,
                            "url": url,
                            "retrieved_at": retrieved_at,
                        },
                        similarity=1.0,
                        citation=citation,
                    )
                )
    return chunks


def retrieve_for_query(
    arxiv_id: str, question: str, top_k: int = TOP_K_CHUNKS
) -> list[RetrievedChunk]:
    """End-to-end retrieval: ensure server-side index, run query, return chunks."""
    _log(f"retrieve_for_query(arxiv_id={arxiv_id})")
    doc_id = get_doc_id(arxiv_id) or fetch_and_index(arxiv_id)
    if not doc_id:
        _log(f"retrieve_for_query: no doc_id for {arxiv_id} — giving up")
        return []

    if not wait_ready(
        doc_id,
        max_wait_s=PAGEINDEX_QUERY_MAX_WAIT_S,
        poll_interval_s=PAGEINDEX_QUERY_POLL_INTERVAL_S,
    ):
        _log(f"retrieve_for_query: doc_id={doc_id} not ready in query window")
        return []

    try:
        response = _CLIENT.submit_query(doc_id=doc_id, query=question)
    except Exception as e:
        _log(f"submit_query FAILED for doc_id={doc_id}: {e}")
        traceback.print_exc(file=sys.stderr)
        return []
    if not isinstance(response, dict):
        _log(f"submit_query returned non-dict: {response!r}")
        return []
    retrieval_id = response.get("retrieval_id")
    if not retrieval_id:
        _log(f"submit_query returned no retrieval_id: {response!r}")
        return []

    retrieval = _wait_retrieval(retrieval_id)
    if retrieval is None:
        _log(f"retrieval {retrieval_id} failed or timed out")
        return []

    nodes = retrieval.get("retrieved_nodes", [])[:top_k]
    _log(f"retrieve_for_query: got {len(nodes)} retrieved_nodes")
    # Non-blocking: cache hit → instant; miss → spawn background fetch and
    # return degraded values so the user response isn't gated on arxiv.org.
    title, authors, published = metadata_for_retrieval(arxiv_id)
    return _nodes_to_chunks(nodes, arxiv_id, title, authors, published)
