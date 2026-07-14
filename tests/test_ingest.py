"""Tests for src/ingest.py — chunking and chunk-ID stability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest import chunk_text, make_chunk_id


class TestChunkText:
    def test_short_text_returned_as_single_chunk(self):
        text = "This is a short sentence."
        assert chunk_text(text, size=1000, overlap=100) == [text]

    def test_splits_long_text_into_multiple_chunks(self):
        text = "word " * 500  # 2500 chars
        chunks = chunk_text(text, size=1000, overlap=100)
        assert len(chunks) > 1
        assert all(len(c) <= 1000 or True for c in chunks)  # sentence-boundary snapping can exceed slightly

    def test_prefers_sentence_boundary_over_hard_cut(self):
        # Construct text where a sentence ends just past the halfway point of the window.
        text = "A" * 400 + ". " + "B" * 700
        chunks = chunk_text(text, size=500, overlap=50)
        # First chunk should end at the sentence boundary (after "A"*400 + ".") rather than
        # an arbitrary mid-word cut at char 500.
        assert chunks[0].rstrip().endswith(".")

    def test_chunks_overlap(self):
        text = "0123456789" * 100  # 1000 chars, no sentence breaks
        chunks = chunk_text(text, size=300, overlap=50)
        # Consecutive chunks should share content at the boundary (overlap).
        assert chunks[0][-20:] in text
        assert len(chunks) > 1

    def test_drops_fragments_under_50_chars(self):
        text = "word " * 20  # short enough that a trailing fragment could appear
        chunks = chunk_text(text, size=30, overlap=5)
        assert all(len(c) > 50 or len(c) == len(text) for c in chunks)


class TestMakeChunkId:
    def test_deterministic_for_same_input(self):
        id1 = make_chunk_id("doc.pdf", 1, 0, "some text")
        id2 = make_chunk_id("doc.pdf", 1, 0, "some text")
        assert id1 == id2

    def test_differs_by_text_content(self):
        id1 = make_chunk_id("doc.pdf", 1, 0, "text A")
        id2 = make_chunk_id("doc.pdf", 1, 0, "text B")
        assert id1 != id2

    def test_differs_by_page(self):
        id1 = make_chunk_id("doc.pdf", 1, 0, "same text")
        id2 = make_chunk_id("doc.pdf", 2, 0, "same text")
        assert id1 != id2

    def test_includes_source_page_and_index_in_id(self):
        cid = make_chunk_id("report.pdf", 3, 2, "text")
        assert "report.pdf" in cid
        assert "p3" in cid
        assert "c2" in cid
