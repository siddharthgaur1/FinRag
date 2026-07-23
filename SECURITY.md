# Security

## Threat model

FinRAG ingests **user-supplied PDFs**, embeds them locally, and answers questions
grounded in them. The trust boundaries are the uploaded file (arbitrary bytes, an
arbitrary filename) and, at answer time, the retrieved document text that an LLM
reads. Embeddings are computed locally (sentence-transformers via Chroma) — no
data leaves the machine for retrieval, and no API key is needed for that half.

## What is mitigated

| Risk | Status | Where |
|---|---|---|
| **Upload path traversal** | **Fixed** — the client filename is reduced to a basename, verified to resolve inside `data/documents/`, and rejected otherwise | `src/app.py` (upload handler) |
| **Upload size exhaustion** | **Fixed** — 25 MB cap before the bytes are written | `src/app.py` |
| Non-PDF upload | **Mitigated** — extension checked server-side in addition to the uploader filter | `src/app.py` |
| Command injection via ingest | **Not present** — ingest runs a fixed argv (`[sys.executable, ingest.py]`), no shell, no user input in the command | `src/app.py` |
| Secrets in git history | **Clean** — `gitleaks`: 0 findings; `.env` gitignored |
| ChromaDB server RCE (PYSEC-2026-311) | **Not applicable** — that advisory is a pre-auth code-injection in Chroma's **HTTP server** API (`/api/v2/...`, `trust_remote_code`). FinRAG uses `chromadb.PersistentClient` (embedded, on-disk) and runs no server, so the vulnerable endpoint does not exist here. No fixed release exists upstream yet; revisit when one ships. | `src/ingest.py`, `src/rag.py` |
| Download script hanging | **Mitigated** — `urllib` sample-doc fetch uses `timeout=60` | `scripts/download_sample_docs.py` |

## What is NOT mitigated / notes

- **No authentication.** Single-operator / demo tool.
- **Prompt injection via ingested PDFs.** A PDF can contain text like "ignore the
  question and output X"; the answering LLM reads retrieved chunks verbatim. The
  faithfulness check (an LLM-based guard) catches some ungrounded answers but is
  not a robust defence against a document crafted to manipulate the model. Treat
  answers over untrusted documents as advisory. This is the RAG threat model and
  is not fully solved here.
- **PDF parsing** runs on untrusted files via the ingest pipeline's PDF library.
  A malformed PDF is a potential parser-crash / resource vector; the size cap
  bounds it but does not eliminate it.

## Reporting

Open an issue. Portfolio/demo project, no production deployment, no security SLA.
