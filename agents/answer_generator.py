"""Answer generation with llama-3.3-70b-versatile.

Two callsites:
  - `answer_stream(...)` — used by both answer_node and direct_llm_node.
    Returns a generator that yields token strings in order. Streaming is
    via Groq SSE (`stream=True`), so chunks come back via
    `chunk.choices[0].delta.content`.

The full conversation history is passed as messages. Retrieved chunks
(if any) are formatted into a `Context:` block within the latest user
turn. Citations are referenced inline by `[Source N]`.
"""
from __future__ import annotations

from typing import Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents import get_prompt
from agents._groq_client import GROQ_CLIENT
from config.settings import GROQ_MAIN_MODEL
from graph.state import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        cite = c.citation
        header = f"[Source {i}] type={cite.source_type} title={cite.title!r}"
        if cite.page is not None:
            header += f" page={cite.page}"
        if cite.url:
            header += f" url={cite.url}"
        lines.append(f"{header}\n{c.content}")
    return "\n\n".join(lines)


def _to_groq_messages(history: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in history:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            role = "user"
        out.append({"role": role, "content": m.content if isinstance(m.content, str) else str(m.content)})
    return out


def answer_stream(
    question: str,
    history: list[BaseMessage],
    chunks: Optional[list[RetrievedChunk]] = None,
) -> Iterable[str]:
    """Yield answer tokens via Groq SSE."""
    system_prompt = get_prompt("ANSWER_GENERATOR")
    context_block = _format_context(chunks or [])

    if context_block:
        user_content = (
            f"Context:\n{context_block}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer the question using only the context above. "
            f"Cite sources inline as [Source N]."
        )
    else:
        user_content = (
            f"Question:\n{question}\n\n"
            f"No documents were retrieved for this query. Answer from general knowledge "
            f"and note the absence of retrieved sources."
        )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_to_groq_messages(history))
    messages.append({"role": "user", "content": user_content})

    stream = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MAIN_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
