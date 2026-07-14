"""
FinRAG v2 - Retrieval + Generation Core

Enhancements over v1:
  - Hybrid retrieval: dense (semantic) + sparse (BM25-style keyword) with RRF fusion
  - Chunk reranking by relevance score threshold
  - Conversation memory: multi-turn Q&A with context window
  - Confidence scoring on answers (based on avg chunk similarity)
  - Anthropic Claude API support (falls back to Ollama)
  - Answer faithfulness check (hallucination guard)
  - Source deduplication in context
"""

import os
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ── LLM backend ──────────────────────────────────────────────────
try:
    import anthropic
    _ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    _USE_ANTHROPIC = bool(_ANTHROPIC_KEY)
except ImportError:
    _USE_ANTHROPIC = False

if not _USE_ANTHROPIC:
    try:
        import ollama as _ollama
        _USE_OLLAMA = True
    except ImportError:
        _USE_OLLAMA = False

if not _USE_ANTHROPIC and not _USE_OLLAMA:
    raise RuntimeError(
        "No LLM backend available. Either set ANTHROPIC_API_KEY, or install and run Ollama "
        "(pip install ollama, then `ollama pull llama3.2` and start the Ollama server)."
    )

# ── Config ────────────────────────────────────────────────────────
DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "financial_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-6"
LLM_MODEL_OLLAMA = "llama3.2"
TOP_K = 6
SIMILARITY_THRESHOLD = 0.30   # discard chunks below this similarity
MAX_CONTEXT_CHARS = 8000       # keep context within LLM window
MAX_TURNS_IN_MEMORY = 6        # number of past Q&A turns to include

SYSTEM_PROMPT = """You are FinRAG, a financial document analyst assistant.
Answer the user's question using ONLY the provided context from financial documents.

Rules:
- Ground every claim in the context. If the context doesn't contain the answer, say "The provided documents do not contain information about this."
- Cite the source document and page number for key claims, like: (Source: HDFC_AR_2024.pdf, p. 42)
- Be precise with numbers. Never invent figures.
- Keep answers concise and structured. Use bullet points for lists.
- If multiple documents give conflicting figures, flag the discrepancy explicitly.
"""

FAITHFULNESS_PROMPT = """You are a faithfulness checker. Given an answer and the context it was based on,
rate whether the answer contains only information present in the context.
Respond with exactly one word: FAITHFUL or UNFAITHFUL, followed by a brief reason."""


# ── LLM helper ────────────────────────────────────────────────────
def _chat(messages: list[dict], system: str = "", temperature: float = 0.1, stream: bool = False):
    if _USE_ANTHROPIC:
        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        sys_prompt = system or next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        if stream:
            with client.messages.stream(
                model=LLM_MODEL_ANTHROPIC,
                max_tokens=1500,
                system=sys_prompt,
                messages=user_msgs,
            ) as s:
                for text in s.text_stream:
                    yield text
        else:
            resp = client.messages.create(
                model=LLM_MODEL_ANTHROPIC,
                max_tokens=1500,
                system=sys_prompt,
                messages=user_msgs,
            )
            return resp.content[0].text
    else:
        if stream:
            s = _ollama.chat(model=LLM_MODEL_OLLAMA, messages=messages,
                             options={"temperature": temperature}, stream=True)
            for part in s:
                yield part["message"]["content"]
        else:
            resp = _ollama.chat(model=LLM_MODEL_OLLAMA, messages=messages,
                                options={"temperature": temperature})
            return resp["message"]["content"]


# ── BM25-style keyword scorer ────────────────────────────────────
def _keyword_score(query: str, text: str) -> float:
    """Simple TF-based keyword overlap score (no IDF — lightweight)."""
    query_terms = set(re.findall(r"\w+", query.lower()))
    doc_terms = re.findall(r"\w+", text.lower())
    if not doc_terms:
        return 0.0
    hits = sum(1 for t in doc_terms if t in query_terms)
    return hits / len(doc_terms) * 10  # scale to roughly match cosine similarity range


# ── Reciprocal Rank Fusion ────────────────────────────────────────
def _rrf_fuse(dense_chunks: list[dict], keyword_chunks: list[dict], k: int = 60) -> list[dict]:
    """Combine dense and keyword rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_chunks):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunks_by_id[cid] = chunk

    for rank, chunk in enumerate(keyword_chunks):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunks_by_id[cid] = chunk

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
    return [chunks_by_id[cid] for cid in sorted_ids]


class FinRAG:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )
        self.conversation_history: list[dict] = []

    # ── Retrieval ─────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Hybrid retrieval: dense semantic + keyword, fused with RRF."""
        if self.collection.count() == 0:
            return []

        # Dense retrieval
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances", "ids"],
        )

        dense_chunks = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            sim = round(1 - dist, 3)
            if sim < SIMILARITY_THRESHOLD:
                continue
            dense_chunks.append({
                "id": cid,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "score": sim,
            })

        # Keyword ranking of same corpus
        keyword_chunks = sorted(
            dense_chunks,
            key=lambda c: _keyword_score(query, c["text"]),
            reverse=True,
        )

        # Fuse rankings
        fused = _rrf_fuse(dense_chunks, keyword_chunks)

        # Deduplicate by source+page (keep highest-scored chunk per page)
        seen_pages: set[str] = set()
        deduped = []
        for c in fused:
            page_key = f"{c['source']}::{c['page']}"
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                deduped.append(c)

        return deduped[:k]

    def build_context(self, chunks: list[dict]) -> str:
        parts = []
        total_chars = 0
        for i, c in enumerate(chunks, 1):
            entry = (
                f"[Chunk {i} | Source: {c['source']}, page {c['page']} | "
                f"similarity: {c['score']}]\n{c['text']}"
            )
            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                break
            parts.append(entry)
            total_chars += len(entry)
        return "\n\n---\n\n".join(parts)

    def confidence(self, chunks: list[dict]) -> float:
        """Average similarity of retrieved chunks as a proxy for answer confidence."""
        if not chunks:
            return 0.0
        return round(sum(c["score"] for c in chunks) / len(chunks), 3)

    # ── Answer ────────────────────────────────────────────────────
    def answer(self, query: str, k: int = TOP_K) -> dict:
        """Non-streaming answer with conversation memory."""
        chunks = self.retrieve(query, k)
        if not chunks:
            return {
                "answer": "No documents have been ingested yet. Run `python src/ingest.py` first.",
                "chunks": [],
                "confidence": 0.0,
            }

        context = self.build_context(chunks)

        # Build messages with memory
        messages = []
        for turn in self.conversation_history[-MAX_TURNS_IN_MEMORY:]:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})

        user_msg = f"Context from financial documents:\n\n{context}\n\n---\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_msg})

        answer_text = _chat(messages, system=SYSTEM_PROMPT)

        # Update memory
        self.conversation_history.append({"q": query, "a": answer_text})

        return {
            "answer": answer_text,
            "chunks": chunks,
            "confidence": self.confidence(chunks),
        }

    def answer_stream(self, query: str, k: int = TOP_K):
        """Streaming answer for the UI. Yields (token, chunks, confidence)."""
        chunks = self.retrieve(query, k)
        if not chunks:
            yield "No documents have been ingested yet. Run `python src/ingest.py` first.", [], 0.0
            return

        context = self.build_context(chunks)
        conf = self.confidence(chunks)

        messages = []
        for turn in self.conversation_history[-MAX_TURNS_IN_MEMORY:]:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({
            "role": "user",
            "content": f"Context from financial documents:\n\n{context}\n\n---\n\nQuestion: {query}",
        })

        full = ""
        for token in _chat(messages, system=SYSTEM_PROMPT, stream=True):
            full += token
            yield token, chunks, conf

        self.conversation_history.append({"q": query, "a": full})

    def check_faithfulness(self, answer: str, chunks: list[dict]) -> str:
        """Quick hallucination check — returns 'FAITHFUL', 'UNFAITHFUL', or 'UNKNOWN'."""
        context = self.build_context(chunks[:3])
        prompt = (
            f"Answer:\n{answer}\n\n"
            f"Context:\n{context}\n\n"
            f"Is the answer faithful to the context?"
        )
        try:
            verdict = _chat(
                [{"role": "user", "content": prompt}],
                system=FAITHFULNESS_PROMPT,
            )
            return "UNFAITHFUL" if "UNFAITHFUL" in verdict.upper() else "FAITHFUL"
        except Exception:
            return "UNKNOWN"

    def clear_memory(self):
        self.conversation_history.clear()

    def get_document_list(self) -> list[str]:
        """Return unique source documents in the collection."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in results["metadatas"]}
        return sorted(sources)


if __name__ == "__main__":
    rag = FinRAG()
    print("FinRAG v2 CLI — type a question (or 'quit' / 'clear')\n")
    while True:
        q = input("Q: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() == "clear":
            rag.clear_memory()
            print("Conversation memory cleared.\n")
            continue
        result = rag.answer(q)
        print(f"\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print("Sources:")
        for c in result["chunks"]:
            print(f"  - {c['source']} p.{c['page']} (sim: {c['score']})")
        print()
