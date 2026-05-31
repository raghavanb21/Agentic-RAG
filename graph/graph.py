"""All LangGraph nodes + edges for the Agentic RAG pipeline (single file).

Read top to bottom — each node is immediately followed by its outgoing
conditional edge function (where one exists). `build_graph()` at the
bottom wires every node and edge into a compiled StateGraph.

Pipeline (execution order):

    START
      ↓
    normalize_node            (deterministic regex normalization, no LLM)
      ↓
    jailbreak_node            (deterministic keyword/regex check, no LLM)
      ↓  BLOCK → END (state.final_answer set to safe response)
      ↓  SAFE
    orchestrator_node         (llama-3.1-8b-instant → RouteDecision)
      ├─ direct_llm     → END (app.py streams direct_llm_node)
      ├─ arxiv          → arxiv_node → citation_builder_node → END
      ├─ blog_url       → web_cache_lookup_node →
      │                     hit?  → citation_builder_node → END
      │                     miss? → web_search_node → injection_check_node → …
      └─ pdf_chromadb   → pdf_retriever_node → crag_grader_node →
                            correct?   → citation_builder_node → END
                            incorrect? → query_rewriter_node → web_search_node → …
                            ambiguous? → query_rewriter_node → web_search_node → …
                                          (web chunks MERGED with pdf chunks)

    web_search_node           (Firecrawl scrape or search)
      ↓
    injection_check_node      (per-doc prompt-guard classifier)
      ↓  UNSAFE → END (state.final_answer set to block message)
      ↓  SAFE   → background insert into web_cache + citation_builder_node → END

    citation_builder_node     (no LLM, Citation Pydantic assembly)
      ↓
    END   (app.py picks up state and streams answer_node or direct_llm_node)

Every node:
  - is decorated with @traceable for LangSmith,
  - measures wall-clock latency with time.perf_counter(),
  - appends a NodeLatency entry to state["node_latencies"].

The two streaming functions (`answer_node`, `direct_llm_node`) live
at the bottom of this file but are NOT in the StateGraph — they are
invoked directly from `app.py` so Streamlit's `st.write_stream(...)`
can consume the Groq SSE generator in the main thread.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Generator, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from agents import answer_generator, crag_grader, jailbreak_guard, normalizer, orchestrator
from citations.builder import build_citations
from config.settings import (
    MAX_REWRITE_COUNT,
    PDF_COLLECTION,
    TOP_K_CHUNKS,
    WEB_COLLECTION,
)
from graph.state import (
    AgentState,
    CRAGResult,
    Citation,
    GuardDecision,
    NodeLatency,
    RetrievedChunk,
    RouteDecision,
    SkippedDoc,
)
from tools import arxiv_tool, chromadb_tool, firecrawl_tool


# ──────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────

def _append_latency(state: AgentState, name: str, t0: float) -> list[NodeLatency]:
    latency_ms = (time.perf_counter() - t0) * 1000
    return (state.get("node_latencies") or []) + [
        NodeLatency(node_name=name, latency_ms=latency_ms)
    ]


def _log(msg: str) -> None:
    print(f"[graph] {msg}", file=sys.stderr, flush=True)


_BLOCK_RESPONSE = (
    "I can't help with that request. If you have a legitimate question about AI "
    "concepts, models, or research papers, please rephrase and I'll do my best."
)


# ──────────────────────────────────────────────────────────────────
# 1. NORMALIZE NODE — deterministic, no LLM
# ──────────────────────────────────────────────────────────────────

@traceable(name="normalize_node")
def normalize_node(state: AgentState) -> AgentState:
    """Lowercase, strip, collapse whitespace via regex."""
    t0 = time.perf_counter()
    query = state.get("query", "")
    normalized = normalizer.normalize(query)
    return {
        "normalized_query": normalized,
        "rewrite_count": state.get("rewrite_count", 0),
        "node_latencies": _append_latency(state, "normalize_node", t0),
    }


# Unconditional edge:  normalize_node → jailbreak_node


# ──────────────────────────────────────────────────────────────────
# 2. JAILBREAK NODE — deterministic regex check, no LLM
# ──────────────────────────────────────────────────────────────────

_JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore (?:all |any )?(?:previous|prior|above|preceding) instructions?\b", re.IGNORECASE),
    re.compile(r"\b(?:reveal|show|print|display|expose|leak) (?:your |the |me your |me the )?(?:system|hidden|internal) prompt\b", re.IGNORECASE),
    re.compile(r"\bdisregard (?:all |any )?(?:previous|prior|above|preceding) (?:instructions?|rules)\b", re.IGNORECASE),
    re.compile(r"\byou are now (?:a |an )?(?:dan|unrestricted|jailbroken|evil|malicious|uncensored)\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdo anything now\b", re.IGNORECASE),
    re.compile(r"\bdan mode\b", re.IGNORECASE),
    re.compile(r"\bact (?:as )?(?:if you are )?(?:an? )?(?:unrestricted|uncensored|evil|malicious|harmful)\b", re.IGNORECASE),
    re.compile(r"\bpretend (?:you are |to be )(?:an? )?(?:unrestricted|uncensored|evil|malicious|harmful)\b", re.IGNORECASE),
    re.compile(r"\b(?:override|bypass) (?:your |the )?(?:safety|guard|rules|filter)\b", re.IGNORECASE),
]


@traceable(name="jailbreak_node")
def jailbreak_node(state: AgentState) -> AgentState:
    """Match a curated list of jailbreak / prompt-exfil patterns. No LLM."""
    t0 = time.perf_counter()
    text = state.get("normalized_query") or state.get("query", "")
    matched: Optional[str] = next(
        (p.pattern for p in _JAILBREAK_PATTERNS if p.search(text)), None
    )
    if matched:
        return {
            "guard_decision": GuardDecision(
                decision="BLOCK",
                cleaned_query=None,
                reason=f"matched jailbreak pattern: {matched[:80]}",
            ),
            "final_answer": _BLOCK_RESPONSE,
            "retrieved_chunks": [],
            "citations": [],
            "skipped_docs": [],
            "node_latencies": _append_latency(state, "jailbreak_node", t0),
        }
    return {
        "guard_decision": GuardDecision(
            decision="SAFE", cleaned_query=None, reason="no jailbreak pattern matched"
        ),
        "node_latencies": _append_latency(state, "jailbreak_node", t0),
    }


def route_after_jailbreak(state: AgentState) -> str:
    decision = state.get("guard_decision")
    if decision is not None and decision.decision == "BLOCK":
        return END
    return "orchestrator_node"


# ──────────────────────────────────────────────────────────────────
# 3. ORCHESTRATOR NODE — LLM routing
# ──────────────────────────────────────────────────────────────────

@traceable(name="orchestrator_node")
def orchestrator_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state.get("query", "")
    decision = orchestrator.route(query)
    return {
        "route_decision": decision,
        "node_latencies": _append_latency(state, "orchestrator_node", t0),
    }


def route_after_orchestrator(state: AgentState) -> str:
    decision = state.get("route_decision")
    if decision is None:
        return END
    if decision.route == "pdf_chromadb":
        return "pdf_retriever_node"
    if decision.route == "arxiv":
        return "arxiv_node"
    if decision.route == "blog_url":
        return "web_cache_lookup_node"
    return END  # direct_llm — app.py streams direct_llm_node


# ──────────────────────────────────────────────────────────────────
# 4. PDF RETRIEVER NODE — ChromaDB top-k vector search
# ──────────────────────────────────────────────────────────────────

@traceable(name="pdf_retriever_node")
def pdf_retriever_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state.get("query", "")
    emb = chromadb_tool.embed_query(query)
    chunks = chromadb_tool.search_collection(PDF_COLLECTION, emb, TOP_K_CHUNKS)
    return {
        "retrieved_chunks": chunks,
        "node_latencies": _append_latency(state, "pdf_retriever_node", t0),
    }


# Unconditional edge:  pdf_retriever_node → crag_grader_node


# ──────────────────────────────────────────────────────────────────
# 5. CRAG GRADER NODE — LLM, returns labelled CRAGResult
# ──────────────────────────────────────────────────────────────────

@traceable(name="crag_grader_node")
def crag_grader_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks") or []
    result = crag_grader.grade(query, chunks)
    return {
        "crag_result": result,
        "node_latencies": _append_latency(state, "crag_grader_node", t0),
    }


def route_after_crag(state: AgentState) -> str:
    result = state.get("crag_result")
    if result is None or result.label == "correct":
        return "citation_builder_node"
    # Both `incorrect` and `ambiguous` trigger query rewrite + web search.
    # The ambiguous branch additionally MERGES the original pdf chunks
    # with the web chunks — see injection_check_node below.
    return "query_rewriter_node"


# ──────────────────────────────────────────────────────────────────
# 6. QUERY REWRITER NODE — LLM
# ──────────────────────────────────────────────────────────────────

@traceable(name="query_rewriter_node")
def query_rewriter_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state.get("query", "")
    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count >= MAX_REWRITE_COUNT:
        return {
            "node_latencies": _append_latency(state, "query_rewriter_node", t0),
        }
    rewritten = crag_grader.rewrite(query)
    return {
        "normalized_query": rewritten,
        "rewrite_count": rewrite_count + 1,
        "node_latencies": _append_latency(state, "query_rewriter_node", t0),
    }


# Unconditional edge:  query_rewriter_node → web_search_node


# ──────────────────────────────────────────────────────────────────
# 7. WEB CACHE LOOKUP NODE — ChromaDB metadata.url match (no LLM)
# ──────────────────────────────────────────────────────────────────

@traceable(name="web_cache_lookup_node")
def web_cache_lookup_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    route = state.get("route_decision")
    url = route.blog_url if route else None
    chunks: list[RetrievedChunk] = []
    if url:
        chunks = chromadb_tool.get_by_url(WEB_COLLECTION, url, TOP_K_CHUNKS)
    return {
        "retrieved_chunks": chunks,
        "node_latencies": _append_latency(state, "web_cache_lookup_node", t0),
    }


def route_after_web_cache(state: AgentState) -> str:
    if state.get("retrieved_chunks"):
        return "citation_builder_node"
    return "web_search_node"


# ──────────────────────────────────────────────────────────────────
# 8. WEB SEARCH NODE — Firecrawl scrape (blog_url) or search (CRAG fallback)
# ──────────────────────────────────────────────────────────────────

@traceable(name="web_search_node")
def web_search_node(state: AgentState) -> AgentState:
    """Firecrawl entry point — picks the right skill operation by context.

    Maps to the skills described in `tools/firecrawl.md`:
      - explicit `blog_url`  → `firecrawl_scrape`  (firecrawl-scrape skill)
      - CRAG fallback search → `firecrawl_search`  (firecrawl-search skill)
    """
    t0 = time.perf_counter()
    route = state.get("route_decision")
    raw_docs: list[dict] = []
    if route and route.route == "blog_url" and route.blog_url:
        fetched = firecrawl_tool.firecrawl_scrape(route.blog_url)
        if fetched:
            raw_docs = [fetched]
    else:
        query = state.get("normalized_query") or state.get("query", "")
        raw_docs = firecrawl_tool.firecrawl_search(query, limit=TOP_K_CHUNKS)
    return {
        "raw_web_docs": raw_docs,
        "node_latencies": _append_latency(state, "web_search_node", t0),
    }


# Unconditional edge:  web_search_node → injection_check_node


# ──────────────────────────────────────────────────────────────────
# 9. INJECTION CHECK NODE — per-doc prompt-guard classifier
# ──────────────────────────────────────────────────────────────────

def _docs_to_chunks(docs: list[dict], source_type: str) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    retrieved_at_default = datetime.now(timezone.utc).isoformat()
    for d in docs:
        metadata = {
            "source_type": source_type,
            "title": d.get("title") or d.get("url", "web"),
            "url": d.get("url"),
            "retrieved_at": d.get("retrieved_at", retrieved_at_default),
        }
        citation = Citation(
            source_type=source_type,  # type: ignore[arg-type]
            title=metadata["title"],
            url=metadata["url"],
            retrieved_at=metadata["retrieved_at"],
            chunk_text=d.get("content", "")[:1000],
        )
        chunks.append(
            RetrievedChunk(
                content=d.get("content", ""),
                metadata=metadata,
                similarity=1.0,
                citation=citation,
            )
        )
    return chunks


def _store_web_docs_async(docs: list[dict], source_type: str) -> None:
    """Fire-and-forget insert into the web_cache collection."""
    if not docs:
        return

    def _store():
        try:
            contents = [d.get("content", "") for d in docs]
            metadatas = [
                {
                    "source_type": source_type,
                    "title": d.get("title") or d.get("url", "web"),
                    "url": d.get("url"),
                    "retrieved_at": d.get("retrieved_at", ""),
                }
                for d in docs
            ]
            ids = [
                f"{source_type}::{abs(hash((d.get('url', ''), d.get('content', '')[:80])))}"
                for d in docs
            ]
            chromadb_tool.store_chunks(WEB_COLLECTION, contents, metadatas, ids)
        except Exception as e:
            _log(f"background web_cache insert failed: {e}")

    threading.Thread(target=_store, daemon=True).start()


@traceable(name="injection_check_node")
def injection_check_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    raw_docs = state.get("raw_web_docs") or []
    clean: list[dict] = []
    skipped: list[SkippedDoc] = list(state.get("skipped_docs") or [])

    for d in raw_docs:
        content = d.get("content", "")
        result = jailbreak_guard.classify_doc(content)
        if result.flagged:
            skipped.append(
                SkippedDoc(
                    url=d.get("url", ""),
                    reason=result.reason or "flagged by injection checker",
                )
            )
        else:
            clean.append(d)

    if raw_docs and not clean:
        # Everything was flagged → block the response.
        return {
            "skipped_docs": skipped,
            "final_answer": _BLOCK_RESPONSE,
            "retrieved_chunks": [],
            "citations": [],
            "node_latencies": _append_latency(state, "injection_check_node", t0),
        }

    web_chunks = _docs_to_chunks(clean, "web")

    # CRAG `ambiguous` → merge the original pdf chunks with the web chunks.
    crag = state.get("crag_result")
    if crag is not None and crag.label == "ambiguous":
        merged = list(state.get("retrieved_chunks") or []) + web_chunks
    else:
        merged = web_chunks

    # Background insert: clean docs into ChromaDB web_cache for future reuse.
    _store_web_docs_async(clean, "web")

    return {
        "retrieved_chunks": merged,
        "skipped_docs": skipped,
        "node_latencies": _append_latency(state, "injection_check_node", t0),
    }


def route_after_injection(state: AgentState) -> str:
    if state.get("final_answer"):
        return END  # blocked
    return "citation_builder_node"


# ──────────────────────────────────────────────────────────────────
# 10. ARXIV NODE — cloud PageIndex retrieval, with runtime fetch+index
# ──────────────────────────────────────────────────────────────────

@traceable(name="arxiv_node")
def arxiv_node(state: AgentState) -> AgentState:
    """Vectorless PageIndex retrieval.

    `arxiv_tool.retrieve_for_query` checks the local `doc_ids.json`
    mapping first; on miss it downloads the paper via the `arxiv`
    Python library, submits to PageIndex, waits for indexing, then
    queries. If no chunks come back we flip the route to direct_llm
    so the answer model can still respond.
    """
    t0 = time.perf_counter()
    route = state.get("route_decision")
    arxiv_id = route.arxiv_id if route else None
    query = state.get("normalized_query") or state.get("query", "")

    chunks: list[RetrievedChunk] = []
    if arxiv_id:
        chunks = arxiv_tool.retrieve_for_query(arxiv_id, query)

    if not chunks:
        flipped = RouteDecision(
            route="direct_llm",
            arxiv_id=arxiv_id,
            blog_url=None,
            reason="arxiv retrieval returned no chunks, falling back to direct_llm",
        )
        return {
            "retrieved_chunks": [],
            "route_decision": flipped,
            "node_latencies": _append_latency(state, "arxiv_node", t0),
        }
    return {
        "retrieved_chunks": chunks,
        "node_latencies": _append_latency(state, "arxiv_node", t0),
    }


def route_after_arxiv(state: AgentState) -> str:
    if state.get("retrieved_chunks"):
        return "citation_builder_node"
    return END  # arxiv flipped to direct_llm — app.py streams direct_llm_node


# ──────────────────────────────────────────────────────────────────
# 11. CITATION BUILDER NODE — no LLM, pure metadata assembly
# ──────────────────────────────────────────────────────────────────

@traceable(name="citation_builder_node")
def citation_builder_node(state: AgentState) -> AgentState:
    t0 = time.perf_counter()
    chunks = state.get("retrieved_chunks") or []
    citations = build_citations(chunks)
    return {
        "citations": citations,
        "node_latencies": _append_latency(state, "citation_builder_node", t0),
    }


# Unconditional edge:  citation_builder_node → END  (app.py streams answer_node)


# ──────────────────────────────────────────────────────────────────
# Streaming functions — NOT in the StateGraph; called by app.py
# ──────────────────────────────────────────────────────────────────

@traceable(name="answer_node")
def answer_node(
    state: AgentState, history: list[BaseMessage]
) -> tuple[Generator[str, None, None], AgentState]:
    """Streaming generator using state.retrieved_chunks as context."""
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks") or []
    gen = answer_generator.answer_stream(query, history, chunks=chunks)
    return gen, {"_answer_t0": t0, "_answer_node_name": "answer_node"}


@traceable(name="direct_llm_node")
def direct_llm_node(
    state: AgentState, history: list[BaseMessage]
) -> tuple[Generator[str, None, None], AgentState]:
    """Streaming generator with NO retrieval context. Greetings / non-AI queries."""
    t0 = time.perf_counter()
    query = state.get("normalized_query") or state.get("query", "")
    gen = answer_generator.answer_stream(query, history, chunks=None)
    return gen, {"_answer_t0": t0, "_answer_node_name": "direct_llm_node"}


def finalize_answer_latency(
    state: AgentState, partial: AgentState, final_text: str
) -> AgentState:
    """Merge the streaming-node latency back into state after the stream completes."""
    t0 = partial.get("_answer_t0")
    name = partial.get("_answer_node_name", "answer_node")
    latency_ms = (time.perf_counter() - t0) * 1000 if t0 is not None else 0.0
    new_latencies = (state.get("node_latencies") or []) + [
        NodeLatency(node_name=name, latency_ms=latency_ms)
    ]
    new_messages = (state.get("messages") or []) + [
        HumanMessage(content=state.get("query", "")),
        AIMessage(content=final_text),
    ]
    return {
        **state,
        "final_answer": final_text,
        "node_latencies": new_latencies,
        "messages": new_messages,
    }


# ──────────────────────────────────────────────────────────────────
# build_graph() — wire everything into a compiled StateGraph
# ──────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("normalize_node", normalize_node)
    builder.add_node("jailbreak_node", jailbreak_node)
    builder.add_node("orchestrator_node", orchestrator_node)
    builder.add_node("pdf_retriever_node", pdf_retriever_node)
    builder.add_node("crag_grader_node", crag_grader_node)
    builder.add_node("query_rewriter_node", query_rewriter_node)
    builder.add_node("web_cache_lookup_node", web_cache_lookup_node)
    builder.add_node("web_search_node", web_search_node)
    builder.add_node("injection_check_node", injection_check_node)
    builder.add_node("arxiv_node", arxiv_node)
    builder.add_node("citation_builder_node", citation_builder_node)

    # Edges — exactly mirror the conditional functions above.
    builder.add_edge(START, "normalize_node")
    builder.add_edge("normalize_node", "jailbreak_node")

    builder.add_conditional_edges(
        "jailbreak_node",
        route_after_jailbreak,
        {"orchestrator_node": "orchestrator_node", END: END},
    )

    builder.add_conditional_edges(
        "orchestrator_node",
        route_after_orchestrator,
        {
            "pdf_retriever_node": "pdf_retriever_node",
            "arxiv_node": "arxiv_node",
            "web_cache_lookup_node": "web_cache_lookup_node",
            END: END,
        },
    )

    builder.add_edge("pdf_retriever_node", "crag_grader_node")

    builder.add_conditional_edges(
        "crag_grader_node",
        route_after_crag,
        {
            "citation_builder_node": "citation_builder_node",
            "query_rewriter_node": "query_rewriter_node",
        },
    )

    builder.add_edge("query_rewriter_node", "web_search_node")

    builder.add_conditional_edges(
        "web_cache_lookup_node",
        route_after_web_cache,
        {
            "citation_builder_node": "citation_builder_node",
            "web_search_node": "web_search_node",
        },
    )

    builder.add_edge("web_search_node", "injection_check_node")

    builder.add_conditional_edges(
        "injection_check_node",
        route_after_injection,
        {"citation_builder_node": "citation_builder_node", END: END},
    )

    builder.add_conditional_edges(
        "arxiv_node",
        route_after_arxiv,
        {"citation_builder_node": "citation_builder_node", END: END},
    )

    builder.add_edge("citation_builder_node", END)

    return builder.compile(checkpointer=MemorySaver())
