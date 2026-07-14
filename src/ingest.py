"""
FinRAG - Document Ingestion Pipeline
Loads financial PDFs (annual reports, RBI circulars, SEBI filings),
chunks them intelligently, embeds them, and stores in ChromaDB.

Usage:
    python src/ingest.py                    # ingest everything in data/documents/
    python src/ingest.py --reset           # wipe DB and re-ingest from scratch
"""

import argparse
import hashlib
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "financial_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"   # free, fast, runs locally
CHUNK_SIZE = 1000                   # characters per chunk
CHUNK_OVERLAP = 200                 # overlap so context isn't cut mid-sentence


def load_pdf(path: Path) -> list[dict]:
    """Extract text from a PDF, page by page, with metadata."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # normalise whitespace
        if len(text) > 50:  # skip near-empty pages
            pages.append({
                "text": text,
                "source": path.name,
                "page": i + 1,
            })
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, breaking at sentence boundaries
    where possible so chunks stay coherent.
    """
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # try to break at the last sentence end inside the window
            window = text[start:end]
            last_period = max(window.rfind(". "), window.rfind(".\n"))
            if last_period > size // 2:
                end = start + last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def make_chunk_id(source: str, page: int, idx: int, text: str) -> str:
    """Stable, unique ID so re-running ingestion doesn't duplicate chunks."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{source}::p{page}::c{idx}::{h}"


def ingest(reset: bool = False):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}")
        print("Download some annual reports / RBI circulars and drop them there.")
        print("Run:  python scripts/download_sample_docs.py  to fetch samples.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(DB_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing collection deleted.")
        except Exception as e:
            # Expected on a fresh DB (no collection to delete yet); print so a
            # real failure (corrupt DB, permissions) isn't silently swallowed.
            print(f"No existing collection to delete ({e})")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() else set()

    total_new = 0
    for pdf_path in pdfs:
        print(f"\nProcessing: {pdf_path.name}")
        pages = load_pdf(pdf_path)
        print(f"  {len(pages)} pages with text")

        ids, docs, metas = [], [], []
        for page in pages:
            for idx, chunk in enumerate(chunk_text(page["text"])):
                cid = make_chunk_id(page["source"], page["page"], idx, chunk)
                if cid in existing_ids:
                    continue
                ids.append(cid)
                docs.append(chunk)
                metas.append({"source": page["source"], "page": page["page"]})

        # add in batches (Chroma has batch limits)
        BATCH = 256
        for i in range(0, len(ids), BATCH):
            collection.add(
                ids=ids[i:i + BATCH],
                documents=docs[i:i + BATCH],
                metadatas=metas[i:i + BATCH],
            )
        total_new += len(ids)
        print(f"  +{len(ids)} new chunks")

    print(f"\nDone. {total_new} new chunks added. Collection size: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="wipe DB and re-ingest")
    args = parser.parse_args()
    ingest(reset=args.reset)
