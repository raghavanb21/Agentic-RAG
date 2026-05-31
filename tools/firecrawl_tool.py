"""Firecrawl client — Path B (build skills) integration per tools/firecrawl.md.

This module is the in-app build-skill site that the LangGraph
`web_search_node` calls. The skill operations from `tools/firecrawl.md`
map to functions here:

  - `firecrawl_scrape(url)`             ↔ firecrawl-scrape  (known URL → clean markdown)
  - `firecrawl_search(query, limit)`    ↔ firecrawl-search  (discovery → ranked pages)
  - `firecrawl_interact(url, actions)`  ↔ firecrawl-interact (click/form/login then scrape)
  - `firecrawl_ask(question, job_id)`   ↔ firecrawl-ask     (diagnose a failing call)

All operations are HTTPS calls under the hood (we use the `firecrawl-py`
SDK rather than shelling out to the CLI), and all of them require
`FIRECRAWL_API_KEY` to be set. When the key is missing, every operation
degrades to "return nothing" — the LangGraph pipeline then falls back
to `direct_llm` (no documents retrieved) instead of crashing.

The legacy names `scrape_url` and `web_search` are kept as thin
aliases so existing imports continue to work.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from firecrawl import FirecrawlApp

from config.settings import FIRECRAWL_API_KEY


def _log(msg: str) -> None:
    print(f"[firecrawl] {msg}", file=sys.stderr, flush=True)


def _client() -> Optional[FirecrawlApp]:
    if not FIRECRAWL_API_KEY:
        _log("FIRECRAWL_API_KEY not set — skipping Firecrawl call")
        return None
    try:
        return FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    except Exception as e:
        _log(f"FirecrawlApp init failed: {e}")
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────
# firecrawl-scrape — single known URL
# ──────────────────────────────────────────────────────────────────

def firecrawl_scrape(url: str) -> Optional[dict]:
    """Scrape one URL → {url, title, content, retrieved_at} or None.

    Maps to the `firecrawl-scrape` skill. Use when you already have the
    URL — for the LangGraph `blog_url` route.
    """
    app = _client()
    if app is None:
        return None
    try:
        result = app.scrape_url(url, params={"formats": ["markdown"]})
    except Exception as e:
        _log(f"scrape_url FAILED for {url}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None

    data = _unwrap(result)
    content = data.get("markdown") or data.get("content") or ""
    metadata = data.get("metadata") or {}
    title = metadata.get("title") or url
    if not content:
        _log(f"scrape_url returned empty content for {url}")
        return None
    return {"url": url, "title": title, "content": content, "retrieved_at": _now()}


# ──────────────────────────────────────────────────────────────────
# firecrawl-search — discovery
# ──────────────────────────────────────────────────────────────────

def firecrawl_search(query: str, limit: int = 5) -> list[dict]:
    """Search the web by query → list of {url, title, content, retrieved_at}.

    Maps to the `firecrawl-search` skill. Use when you do NOT have the
    URL yet — for the CRAG fallback path.
    """
    app = _client()
    if app is None:
        return []
    try:
        result = app.search(
            query,
            params={"limit": limit, "scrapeOptions": {"formats": ["markdown"]}},
        )
    except Exception as e:
        _log(f"search FAILED for query={query!r}: {e}")
        traceback.print_exc(file=sys.stderr)
        return []

    docs_out: list[dict] = []
    items = _items(result)
    for item in items:
        url = item.get("url") or ""
        title = item.get("title") or url
        content = item.get("markdown") or item.get("content") or item.get("description") or ""
        if not (url and content):
            continue
        docs_out.append(
            {"url": url, "title": title, "content": content, "retrieved_at": _now()}
        )
    return docs_out


# ──────────────────────────────────────────────────────────────────
# firecrawl-interact — page actions then scrape
# ──────────────────────────────────────────────────────────────────

def firecrawl_interact(url: str, actions: list[dict[str, Any]]) -> Optional[dict]:
    """Run a sequence of browser actions on `url`, then scrape the result.

    Maps to the `firecrawl-interact` skill. Use when plain scraping fails
    because the page needs clicks, scrolls, form input, or login.

    `actions` follows Firecrawl's action schema, e.g.
        [{"type": "click", "selector": "#accept-cookies"},
         {"type": "wait",  "milliseconds": 500},
         {"type": "scrape"}]

    Returns the same {url, title, content, retrieved_at} shape as
    firecrawl_scrape so the LangGraph pipeline downstream is identical.
    """
    if not actions:
        return firecrawl_scrape(url)
    app = _client()
    if app is None:
        return None
    try:
        result = app.scrape_url(
            url, params={"formats": ["markdown"], "actions": actions}
        )
    except Exception as e:
        _log(f"interact FAILED for {url}: {e}")
        traceback.print_exc(file=sys.stderr)
        return None
    data = _unwrap(result)
    content = data.get("markdown") or data.get("content") or ""
    metadata = data.get("metadata") or {}
    title = metadata.get("title") or url
    if not content:
        return None
    return {"url": url, "title": title, "content": content, "retrieved_at": _now()}


# ──────────────────────────────────────────────────────────────────
# firecrawl-ask — diagnose a failing call
# ──────────────────────────────────────────────────────────────────

def firecrawl_ask(question: str, job_id: Optional[str] = None) -> Optional[dict]:
    """Diagnostic helper. Maps to the `firecrawl-ask` skill.

    Returns {answer, fix_parameters} or None if the support endpoint
    isn't reachable. Useful for surfacing why a search/scrape returned
    nothing, instead of silently degrading.
    """
    app = _client()
    if app is None:
        return None
    method = getattr(app, "support_ask", None) or getattr(app, "ask", None)
    if method is None:
        _log("support_ask not available on this firecrawl-py version")
        return None
    try:
        payload = {"question": question}
        if job_id:
            payload["jobId"] = job_id
        result = method(**payload) if _accepts_kwargs(method) else method(payload)
    except Exception as e:
        _log(f"support_ask FAILED: {e}")
        return None
    data = _unwrap(result)
    return {
        "answer": data.get("answer", ""),
        "fix_parameters": data.get("fixParameters") or data.get("fix_parameters") or {},
    }


# ──────────────────────────────────────────────────────────────────
# Internal helpers + legacy aliases
# ──────────────────────────────────────────────────────────────────

def _unwrap(result: Any) -> dict:
    """Firecrawl responses sometimes nest under `data`; normalize to a dict."""
    if isinstance(result, dict):
        return result.get("data", result) if "data" in result else result
    return {}


def _items(result: Any) -> list[dict]:
    if isinstance(result, dict):
        if isinstance(result.get("data"), list):
            return [x for x in result["data"] if isinstance(x, dict)]
        if isinstance(result.get("results"), list):
            return [x for x in result["results"] if isinstance(x, dict)]
    return []


def _accepts_kwargs(fn) -> bool:
    try:
        import inspect

        sig = inspect.signature(fn)
        return any(
            p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in sig.parameters.values()
        )
    except Exception:
        return False


# Legacy aliases — keep older imports working.
scrape_url = firecrawl_scrape
web_search = firecrawl_search
