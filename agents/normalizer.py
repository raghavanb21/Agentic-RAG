"""Pure string normalization. No model call."""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())
