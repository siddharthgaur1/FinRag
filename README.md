# FinRAG

[![CI](https://github.com/siddharthgaur1/finrag/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthgaur1/finrag/actions/workflows/ci.yml) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Financial documents — annual reports, RBI circulars, SEBI filings — are long,
dense PDFs that don't hold up to keyword search or a single embedding lookup:
a question like "what changed in the RBI repo rate circular" needs an exact
term match, while "how has the NPA ratio trended" needs semantic retrieval
across paraphrased mentions. FinRAG ingests PDFs, indexes them with hybrid
(dense + keyword) retrieval, and answers questions with page-cited grounding
and an optional hallucination check, so an answer can be traced back to the
exact page it came from.

## Architecture

```
PDF files (data/documents/)
        │
        ▼
   src/ingest.py
   ├── pypdf text extraction, page by page
   ├── sentence-boundary chunking (1000 chars, 200 overlap)
   ├── MD5 chunk-ID dedup (re-running ingest never duplicates)
   └── ChromaDB storage (all-MiniLM-L6-v2 embeddings, local, free)

                    Query
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
   Dense retrieval          Keyword scoring
   (ChromaDB cosine)        (TF term overlap)
           │                     │
           └──────────┬──────────┘
                       ▼
             Reciprocal Rank Fusion
             + same-page deduplication
                       │
                       ▼
              Claude (or local Ollama)
                       │
              ┌────────┴────────┐
              ▼                 ▼
      Cited answer +    Faithfulness check
      confidence score   (optional 2nd LLM call)
```

`src/ingest.py` builds the index; `src/rag.py` (`FinRAG` class) holds
retrieval + generation; `src/app.py` is the Streamlit UI over it.

## Tech stack

| Choice | Why |
|---|---|
| ChromaDB (local, file-based) | Zero infra for a project this size — no server to run, persists to `data/chroma_db/`. |
| `all-MiniLM-L6-v2` embeddings | Runs locally, free, fast enough for interactive ingest — no embedding API cost or latency. |
| TF keyword scoring (not real BM25) | Dense embeddings miss exact terms ("Tier 1 capital", specific circular numbers); a lightweight term-overlap score catches those without adding a BM25 library dependency for what's a secondary signal. |
| Reciprocal Rank Fusion | Combines dense + keyword rankings without needing their scores on a common scale — rank-based fusion sidesteps that entirely. |
| Claude, with Ollama fallback | Claude for answer quality; Ollama lets the whole thing run fully offline/free for local development or demoing without an API key. |
| Streamlit | Fastest path to an interactive chat UI with file upload, no separate frontend. |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or install Ollama for a free local backend

python scripts/download_sample_docs.py   # or drop your own PDFs into data/documents/
python src/ingest.py
```

No `ANTHROPIC_API_KEY` and no Ollama installed → the app raises a clear error
at startup rather than failing confusingly on the first question.

## Running it

```bash
streamlit run src/app.py
pytest tests/ -v
```

## Design decisions

**Reciprocal Rank Fusion, not a weighted score blend.** Cosine similarity and
TF term-overlap live on incomparable scales — averaging them directly would
implicitly (and arbitrarily) weight one signal over the other depending on
their respective ranges. RRF only looks at each list's *rank*, so a chunk
that's #1 in both lists always outranks one that's #1 in only one, regardless
of the underlying score magnitudes.

**Confidence is retrieval similarity, not a calibrated probability.** The
displayed confidence score is the mean cosine similarity of the retrieved
chunks — a proxy for "how relevant is the context," not "how likely is this
answer correct." It's cheap and directionally useful (low similarity reliably
correlates with "the documents don't cover this"), but it isn't a substitute
for the faithfulness check for catching a wrong answer built on relevant-but-
insufficient context.

**Faithfulness check is a separate, optional LLM call, not inline
self-critique.** A model asked to both answer and grade its own answer in one
pass tends to rubber-stamp itself. A second call, given only the answer and
the (top-3) source chunks with no memory of having generated the answer,
gives an independent check — at the cost of a second LLM round-trip, which is
why it's a toggle rather than always-on.

## What I'd improve with more time

1. **No eval harness.** This is the biggest gap — there's no way to answer
   "did the last prompt/chunking change make this better or worse" other than
   spot-checking manually. `llm-regression-detector` (a companion project)
   is built exactly for this — a golden Q&A set plus the same regression-gate
   pattern would apply directly here.
2. **Real BM25 instead of TF overlap.** The current keyword score
   (`hits / len(doc_terms) * 10`) has no IDF term, so common words are
   weighted the same as rare, discriminating ones (e.g. "the" counts the same
   as "NPA"). A proper BM25 implementation (`rank_bm25`, or reusing the
   pattern from `rag-hybrid-search`) would meaningfully improve the sparse
   side of the fusion.
3. **Confidence calibration.** Replace the raw-similarity proxy with
   something that's actually validated against "was this answer correct" —
   requires the eval harness above to exist first.
4. **Chunking strategy comparison.** Fixed 1000-char/200-overlap chunking is
   a reasonable default but untested against alternatives (structure-aware,
   semantic) — see `rag-hybrid-search` for a chunking-strategy comparison
   harness that could be adapted here.

## Related projects

- [llm-regression-detector](https://github.com/siddharthgaur1/llm-regression-detector) — the eval-gate pattern this project is missing.
- [rag-hybrid-search](https://github.com/siddharthgaur1/rag-hybrid-search) — a more fully-built hybrid RAG pipeline (real BM25, cross-encoder reranking, citation verification) that this project's "what I'd improve" items point toward.
- [ipo-gmp](https://github.com/siddharthgaur1/ipo-gmp) — XGBoost IPO listing-return predictor.
