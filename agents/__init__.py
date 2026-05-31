"""Parses `agents.md` and exposes one system prompt per section header.

`agents.md` is the single source of truth for prompts. The file is
read once at import time. Each `## SECTION_NAME` header maps to a
prompt string in `PROMPTS`.
"""
from __future__ import annotations

import re
from pathlib import Path

from config.settings import AGENTS_MD_PATH


def _load_prompts(path: str) -> dict[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    pattern = re.compile(r"^##\s+([A-Z][A-Z0-9_]+)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip().strip("-").strip()
        sections[name] = body
    return sections


PROMPTS: dict[str, str] = _load_prompts(AGENTS_MD_PATH)


def get_prompt(name: str) -> str:
    if name not in PROMPTS:
        raise KeyError(
            f"Prompt section '{name}' not found in agents.md. "
            f"Available: {sorted(PROMPTS.keys())}"
        )
    return PROMPTS[name]
