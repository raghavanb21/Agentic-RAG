"""CRAG grader + query rewriter (both `llama-3.1-8b-instant`).

- `grade(question, chunks)` returns a labelled `CRAGResult`
  (`correct` / `incorrect` / `ambiguous` + score + reason). The score
  range thresholds match those documented in `agents.md`'s CRAG_GRADER
  section so the label is deterministic given the score.
- `rewrite(question)` returns a single improved query string for the
  CRAG fallback path. It's invoked by the `query_rewriter_node`.
"""
from __future__ import annotations

import json
from typing import Optional

from agents import get_prompt
from agents._groq_client import GROQ_CLIENT
from config.settings import GROQ_FAST_MODEL
from graph.state import CRAGResult, RetrievedChunk


def _try_json(content: str) -> Optional[dict]:
    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                return None
    return None


def _format_chunks(chunks: list[RetrievedChunk], max_chars: int = 2000) -> str:
    parts: list[str] = []
    budget = max_chars
    for i, c in enumerate(chunks, start=1):
        snippet = c.content[: min(len(c.content), budget)]
        parts.append(f"[Chunk {i}] {snippet}")
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n".join(parts)


def _label_from_score(score: float) -> str:
    if score >= 0.7:
        return "correct"
    if score >= 0.4:
        return "ambiguous"
    return "incorrect"


def grade(question: str, chunks: list[RetrievedChunk]) -> CRAGResult:
    if not chunks:
        return CRAGResult(label="incorrect", score=0.0, reason="no chunks retrieved")

    system_prompt = get_prompt("CRAG_GRADER")
    user_msg = (
        f"Question:\n{question}\n\n"
        f"Retrieved chunks:\n{_format_chunks(chunks)}"
    )
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            stream=False,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        return CRAGResult(label="incorrect", score=0.0, reason=f"grader error: {e}")

    parsed = _try_json(content)
    if not parsed:
        return CRAGResult(label="incorrect", score=0.0, reason="unparseable grader output")
    try:
        result = CRAGResult.model_validate(parsed)
    except Exception as e:
        return CRAGResult(label="incorrect", score=0.0, reason=f"validation error: {e}")

    # Enforce the label↔score contract from agents.md regardless of what
    # the model emitted: the score is the source of truth.
    return CRAGResult(
        label=_label_from_score(result.score),
        score=result.score,
        reason=result.reason,
    )


def rewrite(question: str) -> str:
    system_prompt = get_prompt("QUERY_REWRITER")
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            stream=False,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return question

    parsed = _try_json(content)
    if not parsed:
        return question
    rewritten = parsed.get("rewritten_query")
    if isinstance(rewritten, str) and rewritten.strip():
        return rewritten.strip()
    return question
