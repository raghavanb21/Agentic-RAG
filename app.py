"""Streamlit chat UI for the Agentic RAG system.

A fresh session begins on every refresh. Within one session, the
LangGraph checkpointer (MemorySaver) persists conversation state under
a single thread_id, so follow-up questions inherit context. The graph
runs from normalize → guard → orchestrator → retrieval → citation
builder. The streaming answer (or direct_llm) step is then invoked
directly from this file so `st.write_stream(...)` can pipe Groq SSE
tokens straight to the UI.

Render order on each response:
    1. assistant message (streamed token-by-token)
    2. latency caption + node-latency expander
    3. Sources expander (Citations)
    4. Skipped Sources expander (only if any skipped)
"""
from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph.graph import (
    build_graph,
    answer_node as streaming_answer_node,
    direct_llm_node as streaming_direct_llm_node,
    finalize_answer_latency,
)
from graph.state import Citation, LatencyReport, NodeLatency, SkippedDoc

st.set_page_config(page_title="Agentic RAG — AI Concepts", page_icon="🧠", layout="wide")


def _init_session() -> None:
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []
    if "langgraph_state" not in st.session_state:
        st.session_state.langgraph_state = {
            "messages": [],
            "rewrite_count": 0,
            "node_latencies": [],
            "retrieved_chunks": [],
            "citations": [],
            "skipped_docs": [],
        }
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def _sidebar() -> None:
    with st.sidebar:
        st.title("Agentic RAG")
        st.caption("AI Concepts Research Assistant")
        st.divider()
        st.markdown(f"**Session:** `{st.session_state.thread_id[:8]}`")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            for key in ("messages", "langgraph_state", "thread_id", "graph"):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.divider()
        with st.expander("Routes"):
            st.markdown(
                "- **pdf_chromadb** — local AI-concepts PDFs\n"
                "- **arxiv** — ArXiv paper (id or title)\n"
                "- **blog_url** — explicit URL via Firecrawl\n"
                "- **direct_llm** — greetings / non-AI"
            )


def _render_latency(latencies: list[NodeLatency]) -> None:
    if not latencies:
        return
    report = LatencyReport(
        total_latency_ms=sum(n.latency_ms for n in latencies),
        per_node=latencies,
    )
    st.caption(f"⏱ Total latency: {report.total_latency_ms:.0f}ms")
    with st.expander("Node latencies"):
        for n in report.per_node:
            st.caption(f"{n.node_name}: {n.latency_ms:.0f}ms")


def _render_citations(citations: list[Citation]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for i, c in enumerate(citations, start=1):
            header = f"**[{i}] {c.source_type}** · {c.title}"
            if c.page is not None:
                header += f" · p.{c.page}"
            st.markdown(header)
            if c.url:
                st.markdown(f"[{c.url}]({c.url})")
            preview = c.chunk_text[:280] + ("…" if len(c.chunk_text) > 280 else "")
            st.markdown(f"> {preview}")
            st.markdown("---")


def _render_skipped(skipped: list[SkippedDoc]) -> None:
    if not skipped:
        return
    with st.expander(f"Skipped sources ({len(skipped)})"):
        for s in skipped:
            st.markdown(f"- `{s.url or '—'}` — {s.reason}")


def _build_history():
    return st.session_state.langgraph_state.get("messages", [])


def _run_graph(query: str) -> dict:
    state_in = {
        **st.session_state.langgraph_state,
        "query": query,
        "rewrite_count": 0,
        "node_latencies": [],
        "retrieved_chunks": [],
        "citations": [],
        "skipped_docs": [],
        "guard_decision": None,
        "route_decision": None,
        "final_answer": "",
    }
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    final_state = st.session_state.graph.invoke(state_in, config=config)
    return final_state


def _is_blocked(state: dict) -> bool:
    """True for both jailbreak BLOCK and injection-unsafe outcomes — both set final_answer."""
    return bool(state.get("final_answer"))


def _is_direct_llm(state: dict) -> bool:
    decision = state.get("route_decision")
    return decision is not None and decision.route == "direct_llm"


def main() -> None:
    _init_session()
    _sidebar()

    st.title("🧠 Agentic RAG")
    st.caption("Ask about AI concepts, arXiv papers, or paste a blog URL.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about AI…")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Routing…"):
        graph_state = _run_graph(user_input)

    if _is_blocked(graph_state):
        text = graph_state.get("final_answer") or ""
        with st.chat_message("assistant"):
            st.markdown(text)
            _render_latency(graph_state.get("node_latencies") or [])
            _render_skipped(graph_state.get("skipped_docs") or [])
        st.session_state.messages.append({"role": "assistant", "content": text})
        graph_state["messages"] = (graph_state.get("messages") or []) + [
            HumanMessage(content=user_input),
            AIMessage(content=text),
        ]
        st.session_state.langgraph_state = graph_state
        return

    history = _build_history()
    if _is_direct_llm(graph_state):
        gen, partial = streaming_direct_llm_node(graph_state, history)
    else:
        gen, partial = streaming_answer_node(graph_state, history)

    with st.chat_message("assistant"):
        final_text = st.write_stream(gen)
        merged_state = finalize_answer_latency(graph_state, partial, final_text or "")
        _render_latency(merged_state.get("node_latencies") or [])
        _render_citations(merged_state.get("citations") or [])
        _render_skipped(merged_state.get("skipped_docs") or [])

    st.session_state.messages.append({"role": "assistant", "content": final_text or ""})
    st.session_state.langgraph_state = merged_state


if __name__ == "__main__":
    main()
