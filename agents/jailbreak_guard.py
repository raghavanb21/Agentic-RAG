"""Jailbreak / prompt-injection classifier using llama-prompt-guard-2-86m.
"""
from __future__ import annotations

import json
from typing import Optional

from agents._groq_client import GROQ_CLIENT
from config.settings import GROQ_GUARD_MODEL, GUARD_MAX_CHARS
from graph.state import GuardDecision, InjectionResult

_JAILBREAK_LABELS = {"jailbreak", "malicious", "unsafe", "harmful"}
_INJECTION_LABELS = {"injection", "prompt_injection", "prompt-injection"}
_BENIGN_LABELS = {"benign", "safe", "ok"}


def _label_to_guard(label: str, raw_query: str) -> Optional[GuardDecision]:
    low = label.strip().lower()
    if not low:
        return None
    if low in _JAILBREAK_LABELS:
        return GuardDecision(decision="BLOCK", cleaned_query=None, reason=f"guard label: {low}")
    if low in _INJECTION_LABELS:
        return GuardDecision(
            decision="REDIRECT",
            cleaned_query=raw_query,
            reason=f"guard label: {low}",
        )
    if low in _BENIGN_LABELS:
        return GuardDecision(decision="SAFE", cleaned_query=None, reason=f"guard label: {low}")
    return None


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


def classify_query(query: str) -> GuardDecision:
    truncated = query[:GUARD_MAX_CHARS]
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_GUARD_MODEL,
            messages=[{"role": "user", "content": truncated}],
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as e:
        return GuardDecision(decision="SAFE", cleaned_query=None, reason=f"guard error: {e}")

    label_decision = _label_to_guard(content, query)
    if label_decision is not None:
        return label_decision

    parsed = _try_json(content)
    if parsed is not None:
        try:
            return GuardDecision.model_validate(parsed)
        except Exception:
            pass

    return GuardDecision(decision="SAFE", cleaned_query=None, reason="guard produced no parseable label")


def classify_doc(content: str) -> InjectionResult:
    truncated = content[:GUARD_MAX_CHARS]
    try:
        response = GROQ_CLIENT.chat.completions.create(
            model=GROQ_GUARD_MODEL,
            messages=[{"role": "user", "content": truncated}],
            stream=False,
        )
        out = (response.choices[0].message.content or "").strip()
    except Exception as e:
        return InjectionResult(flagged=False, reason=f"checker error: {e}")

    low = out.lower()
    if any(lbl in low for lbl in _INJECTION_LABELS) or any(lbl in low for lbl in _JAILBREAK_LABELS):
        return InjectionResult(flagged=True, reason=f"checker label: {low.strip()[:80]}")
    if any(lbl in low for lbl in _BENIGN_LABELS):
        return InjectionResult(flagged=False, reason=None)

    parsed = _try_json(out)
    if parsed is not None:
        try:
            return InjectionResult.model_validate(parsed)
        except Exception:
            pass

    return InjectionResult(flagged=False, reason=None)
