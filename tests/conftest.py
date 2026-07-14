"""Shared pytest fixtures for FinRAG tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def make_chunk(cid="c1", text="chunk text", source="doc.pdf", page=1, score=0.5) -> dict:
    return {"id": cid, "text": text, "source": source, "page": page, "score": score}
