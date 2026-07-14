"""Tests for src/rag.py — keyword scoring, RRF fusion, context building,
confidence, and faithfulness checking (LLM calls mocked)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import rag
from rag import FinRAG, _keyword_score, _rrf_fuse
from conftest import make_chunk


class TestKeywordScore:
    def test_scores_zero_for_no_overlap(self):
        assert _keyword_score("apple banana", "completely unrelated content") == 0.0

    def test_scores_positive_for_overlap(self):
        assert _keyword_score("NPA ratio", "the NPA ratio improved this year") > 0.0

    def test_empty_document_scores_zero(self):
        assert _keyword_score("query", "") == 0.0

    def test_case_insensitive(self):
        assert _keyword_score("NPA", "the npa ratio") == _keyword_score("npa", "the NPA ratio")


class TestRRFFuse:
    def test_combines_and_ranks_by_fused_score(self):
        dense = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        keyword = [make_chunk("b"), make_chunk("a"), make_chunk("c")]
        fused = _rrf_fuse(dense, keyword, k=60)
        fused_ids = [c["id"] for c in fused]
        assert set(fused_ids) == {"a", "b", "c"}
        # "a" and "b" both rank high in both lists, "c" is last in both -> c should rank lowest.
        assert fused_ids[-1] == "c"

    def test_chunk_present_in_only_one_list_still_included(self):
        dense = [make_chunk("only_dense")]
        fused = _rrf_fuse(dense, [], k=60)
        assert len(fused) == 1
        assert fused[0]["id"] == "only_dense"

    def test_empty_inputs_return_empty(self):
        assert _rrf_fuse([], [], k=60) == []


class TestBuildContext:
    def test_formats_chunks_with_metadata(self):
        instance = FinRAG.__new__(FinRAG)  # bypass __init__ (no chromadb/embedding model needed)
        chunks = [make_chunk("a", text="hello world", source="doc.pdf", page=3, score=0.72)]
        context = instance.build_context(chunks)
        assert "doc.pdf" in context
        assert "page 3" in context
        assert "hello world" in context
        assert "0.72" in context

    def test_truncates_at_max_context_chars(self):
        instance = FinRAG.__new__(FinRAG)
        big_chunk = make_chunk("a", text="x" * 10000)
        small_chunk = make_chunk("b", text="short")
        context = instance.build_context([big_chunk, small_chunk])
        # The big chunk alone exceeds MAX_CONTEXT_CHARS-ish budget with metadata overhead,
        # so the second chunk should be dropped rather than the context growing unbounded.
        assert len(context) <= rag.MAX_CONTEXT_CHARS + 500  # allow for entry header overhead
        assert "short" not in context or len(context) < 10000 + 500


class TestConfidence:
    def test_averages_chunk_scores(self):
        instance = FinRAG.__new__(FinRAG)
        chunks = [make_chunk(score=0.8), make_chunk(score=0.4)]
        assert instance.confidence(chunks) == 0.6

    def test_empty_chunks_returns_zero(self):
        instance = FinRAG.__new__(FinRAG)
        assert instance.confidence([]) == 0.0


class TestCheckFaithfulness:
    def test_faithful_verdict(self, monkeypatch):
        instance = FinRAG.__new__(FinRAG)
        monkeypatch.setattr(rag, "_chat", lambda *a, **k: "FAITHFUL - the answer matches the context")
        result = instance.check_faithfulness("some answer", [make_chunk()])
        assert result == "FAITHFUL"

    def test_unfaithful_verdict(self, monkeypatch):
        instance = FinRAG.__new__(FinRAG)
        monkeypatch.setattr(rag, "_chat", lambda *a, **k: "UNFAITHFUL - contains unsupported claims")
        result = instance.check_faithfulness("some answer", [make_chunk()])
        assert result == "UNFAITHFUL"

    def test_llm_error_returns_unknown(self, monkeypatch):
        instance = FinRAG.__new__(FinRAG)

        def raise_error(*a, **k):
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(rag, "_chat", raise_error)
        result = instance.check_faithfulness("some answer", [make_chunk()])
        assert result == "UNKNOWN"
