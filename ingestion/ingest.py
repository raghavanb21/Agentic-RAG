"""Single entry-point: ingest both pdf and arxiv folders. Idempotent."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import ARXIV_DATA_DIR, PDF_DATA_DIR  # noqa: E402
from ingestion import arxiv_ingestor, pdf_ingestor  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("Agentic RAG — Ingestion")
    print("=" * 60)

    print("\n[1/2] PDFs → pdf_text")
    pdf_results = pdf_ingestor.ingest_folder(PDF_DATA_DIR)

    print("\n[2/2] ArXiv PDFs → PageIndex workspace (vectorless)")
    arxiv_results = arxiv_ingestor.ingest_folder(ARXIV_DATA_DIR)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    pdf_total = sum(r.total_chunks for r in pdf_results)
    arxiv_indexed = sum(1 for r in arxiv_results if r.success)
    print(f"PDF chunks ingested:        {pdf_total} ({len(pdf_results)} files)")
    print(f"ArXiv papers in PageIndex:  {arxiv_indexed} ({len(arxiv_results)} files)")
    print()
    for r in pdf_results + arxiv_results:
        status = "OK " if r.success else "ERR"
        err = f"  ({r.error})" if r.error else ""
        print(f"  [{status}] {r.source}: {r.total_chunks} chunks{err}")
    print()


if __name__ == "__main__":
    main()
