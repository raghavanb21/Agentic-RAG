"""AgentState TypedDict and all Pydantic models used across the graph.

All LLM JSON responses MUST be parsed into these Pydantic models with
`model.model_validate(...)`. Never use raw dict access on LLM outputs.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["pdf_chromadb", "arxiv", "blog_url", "direct_llm"]
    arxiv_id: Optional[str] = None
    blog_url: Optional[str] = None
    reason: str


class GuardDecision(BaseModel):
    decision: Literal["BLOCK", "REDIRECT", "SAFE"]
    cleaned_query: Optional[str] = None
    reason: str


class InjectionResult(BaseModel):
    flagged: bool
    reason: Optional[str] = None


class CRAGResult(BaseModel):
    """CRAG grader output. `label` is set strictly from `score`:
    correct >= 0.7, 0.4 <= ambiguous < 0.7, incorrect < 0.4.
    """

    label: Literal["correct", "incorrect", "ambiguous"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class Citation(BaseModel):
    source_type: Literal["pdf", "arxiv", "web", "blog"]
    title: str
    author: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    published_date: Optional[str] = None
    retrieved_at: str
    chunk_text: str


class SkippedDoc(BaseModel):
    url: str
    reason: str


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict
    similarity: float
    citation: Citation


class NodeLatency(BaseModel):
    node_name: str
    latency_ms: float


class LatencyReport(BaseModel):
    total_latency_ms: float
    per_node: list[NodeLatency]


class PDFChunk(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    chunk_index: int
    source_file: str
    page: int


class IngestionResult(BaseModel):
    source: str
    total_chunks: int
    success: bool
    error: Optional[str] = None


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    normalized_query: str
    guard_decision: Optional[GuardDecision]
    route_decision: Optional[RouteDecision]
    retrieved_chunks: list[RetrievedChunk]
    citations: list[Citation]
    rewrite_count: int
    skipped_docs: list[SkippedDoc]
    final_answer: str
    node_latencies: list[NodeLatency]
    crag_result: Optional[CRAGResult]
    raw_web_docs: list[dict]
