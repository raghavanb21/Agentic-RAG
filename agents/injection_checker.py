"""Per-document injection checker. Thin wrapper over jailbreak_guard.classify_doc."""
from __future__ import annotations

from agents.jailbreak_guard import classify_doc
from graph.state import InjectionResult


def check(content: str) -> InjectionResult:
    return classify_doc(content)
