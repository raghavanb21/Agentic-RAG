"""Routing agent — decides which retrieval route the query takes.

Uses llama-3.1-8b-instant with JSON mode. The raw response is parsed
into a RouteDecision Pydantic model.
"""
from __future__ import annotations

import json
from typing import Optional

from agents import get_prompt
from agents._groq_client import GROQ_CLIENT
from config.settings import GROQ_FAST_MODEL
from graph.state import RouteDecision


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


def route(query: str) -> RouteDecision:
    system_prompt = get_prompt("ORCHESTRATOR")
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_FAST_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            stream=False,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
    except Exception as e:
        return RouteDecision(
            route="direct_llm", arxiv_id=None, blog_url=None, reason=f"orchestrator error: {e}"
        )

    parsed = _try_json(content)
    if parsed is None:
        return RouteDecision(
            route="direct_llm",
            arxiv_id=None,
            blog_url=None,
            reason="orchestrator returned unparseable JSON",
        )
    try:
        return RouteDecision.model_validate(parsed)
    except Exception as e:
        return RouteDecision(
            route="direct_llm",
            arxiv_id=None,
            blog_url=None,
            reason=f"orchestrator validation error: {e}",
        )
