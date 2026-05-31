"""Single shared Groq client instance for all agents."""
from __future__ import annotations

from groq import Groq

from config.settings import GROQ_API_KEY

GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
