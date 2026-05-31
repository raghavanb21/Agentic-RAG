"""Ingest data/pdfs/*.pdf into ChromaDB collection `pdf_text`.

PyMuPDF extracts page-by-page text. RecursiveCharacterTextSplitter
chunks the text. bge-base-en-v1.5 (loaded once in chromadb_tool)
produces embeddings. Each chunk is validated as a PDFChunk Pydantic
model before being stored. Ingestion is idempotent — files already in
the collection are skipped.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, PDF_COLLECTION
from graph.state import IngestionResult, PDFChunk
from tools import chromadb_tool


_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                pages.append((i, text))
    return pages


def _chunks_for_pdf(pdf_path: str) -> list[PDFChunk]:
    source_file = Path(pdf_path).name
    title = Path(pdf_path).stem
    retrieved_at = datetime.now(timezone.utc).isoformat()
    out: list[PDFChunk] = []
    idx = 0
    for page_no, page_text in _extract_pages(pdf_path):
        for piece in _SPLITTER.split_text(page_text):
            if not piece.strip():
                continue
            metadata = {
                "source_type": "pdf",
                "source_file": source_file,
                "page": page_no,
                "chunk_index": idx,
                "title": title,
                "retrieved_at": retrieved_at,
            }
            out.append(
                PDFChunk(
                    content=piece,
                    metadata=metadata,
                    chunk_index=idx,
                    source_file=source_file,
                    page=page_no,
                )
            )
            idx += 1
    return out


def ingest_file(pdf_path: str) -> IngestionResult:
    source_file = Path(pdf_path).name
    if chromadb_tool.has_source(PDF_COLLECTION, "source_file", source_file):
        print(f"  ↳ Skipping {source_file} (already ingested)")
        return IngestionResult(source=source_file, total_chunks=0, success=True, error=None)

    try:
        chunks = _chunks_for_pdf(pdf_path)
        if not chunks:
            return IngestionResult(
                source=source_file, total_chunks=0, success=False, error="no text extracted"
            )
        contents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"pdf::{source_file}::{c.chunk_index}" for c in chunks]
        chromadb_tool.store_chunks(PDF_COLLECTION, contents, metadatas, ids)
        print(f"  ↳ Ingesting {source_file}... {len(chunks)} chunks stored")
        return IngestionResult(
            source=source_file, total_chunks=len(chunks), success=True, error=None
        )
    except Exception as e:
        return IngestionResult(source=source_file, total_chunks=0, success=False, error=str(e))


def ingest_folder(folder: str) -> list[IngestionResult]:
    results: list[IngestionResult] = []
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"PDF folder not found: {folder}")
        return results
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {folder}")
        return results
    print(f"Found {len(pdf_files)} PDFs in {folder}")
    for pdf_path in pdf_files:
        results.append(ingest_file(str(pdf_path)))
    return results
