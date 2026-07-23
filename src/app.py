"""
FinRAG v2 - Streamlit Web UI

Enhancements over v1:
  - Confidence score indicator per answer
  - Faithfulness check toggle (hallucination guard)
  - Document list panel showing ingested files
  - Conversation memory indicator + clear button
  - PDF upload directly from UI (triggers ingest)
  - Source similarity colour-coding (green/yellow/red)
  - Top-K slider persisted across questions

Run:
    streamlit run src/app.py
"""

import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import FinRAG, LLM_MODEL_OLLAMA, _USE_ANTHROPIC

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"
INGEST_SCRIPT = Path(__file__).resolve().parent / "ingest.py"

st.set_page_config(page_title="FinRAG v2", page_icon="📊", layout="wide")


# ── Render helpers ────────────────────────────────────────────────
def _render_confidence(score: float):
    color = "green" if score >= 0.6 else "orange" if score >= 0.4 else "red"
    label = "High" if score >= 0.6 else "Medium" if score >= 0.4 else "Low"
    st.markdown(
        f"**Retrieval confidence:** "
        f"<span style='color:{color};font-weight:bold'>{label} ({score:.2f})</span>",
        unsafe_allow_html=True,
    )


def _render_faithfulness(verdict: str):
    if verdict == "FAITHFUL":
        st.success("🛡️ Answer is faithful to sources")
    elif verdict == "UNFAITHFUL":
        st.error("⚠️ Answer may contain information not in sources")
    else:
        st.info("🔍 Faithfulness check inconclusive")


def _render_sources(chunks: list[dict]):
    for c in chunks:
        sim = c["score"]
        sim_color = "green" if sim >= 0.6 else "orange" if sim >= 0.4 else "red"
        st.markdown(
            f"**{c['source']}** — page {c['page']} "
            f"<span style='color:{sim_color}'>(sim: {sim})</span>",
            unsafe_allow_html=True,
        )
        st.caption(c["text"][:350] + ("…" if len(c["text"]) > 350 else ""))


@st.cache_resource
def get_rag() -> FinRAG:
    return FinRAG()


rag = get_rag()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 FinRAG v2")
    st.caption("RAG over Financial Documents")

    backend = "Anthropic Claude" if _USE_ANTHROPIC else f"Ollama ({LLM_MODEL_OLLAMA})"
    st.markdown(f"**LLM:** {backend}")
    st.divider()

    top_k = st.slider("Chunks to retrieve", 3, 12, 6)
    check_faithfulness = st.toggle("🛡️ Faithfulness check", value=False,
                                   help="Runs a second LLM call to verify the answer against sources. Slower but reduces hallucinations.")
    st.divider()

    # Document list
    st.markdown("**📁 Ingested documents**")
    doc_list = rag.get_document_list()
    if doc_list:
        for doc in doc_list:
            st.markdown(f"- `{doc}`")
        st.caption(f"{rag.collection.count()} total chunks indexed")
    else:
        st.warning("No documents yet.")

    st.divider()

    # PDF upload
    st.markdown("**⬆️ Upload new PDF**")
    uploaded_file = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")
    MAX_UPLOAD_MB = 25
    if uploaded_file:
        # Never trust the client-supplied name: strip it to a bare filename so a
        # crafted "../../etc/whatever" cannot write outside the documents dir, and
        # confirm the write target really resolves inside DOCS_DIR before touching
        # disk. Cap the size so a huge upload can't fill the container.
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(uploaded_file.name).name
        dest = (DOCS_DIR / safe_name).resolve()
        data = uploaded_file.getvalue()
        if not safe_name.lower().endswith(".pdf"):
            st.error("Only .pdf files are accepted.")
            st.stop()
        if dest.parent != DOCS_DIR.resolve():
            st.error("Invalid filename.")
            st.stop()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large (> {MAX_UPLOAD_MB} MB).")
            st.stop()
        dest.write_bytes(data)
        if st.button("Ingest now"):
            with st.spinner(f"Ingesting {uploaded_file.name}…"):
                result = subprocess.run(
                    [sys.executable, str(INGEST_SCRIPT)],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Ingested! Refresh the page to query.")
                st.cache_resource.clear()
            else:
                st.error(f"Ingest failed:\n{result.stderr}")

    st.divider()

    # Memory controls
    turns = len(rag.conversation_history)
    st.markdown(f"**🧠 Conversation memory** ({turns} turn{'s' if turns != 1 else ''})")
    if st.button("🗑️ Clear memory"):
        rag.clear_memory()
        st.rerun()

    st.divider()
    st.markdown("""
    **Sample questions**
    - What are the key risk factors mentioned?
    - Summarise the capital adequacy position
    - What is the NPA trend over the last 3 years?
    - Compare revenue across the documents
    - What did the RBI circular change?
    """)

# ── Main ──────────────────────────────────────────────────────────
st.title("Ask your financial documents")

doc_count = rag.collection.count()
if doc_count == 0:
    st.warning(
        "No documents ingested yet. Upload a PDF in the sidebar, "
        "or drop PDFs into `data/documents/` and run `python src/ingest.py`."
    )
else:
    st.caption(f"{doc_count} chunks indexed across {len(doc_list)} document(s)")

# ── Chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("confidence") is not None:
            _render_confidence(msg["confidence"])
        if msg.get("faithfulness"):
            _render_faithfulness(msg["faithfulness"])
        if msg.get("chunks"):
            with st.expander("📄 Sources"):
                _render_sources(msg["chunks"])

# ── Chat input ────────────────────────────────────────────────────
if query := st.chat_input("e.g. What is HDFC's NPA ratio?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        chunks_used = []
        conf = 0.0

        for token, chunks, confidence in rag.answer_stream(query, k=top_k):
            full_answer += token
            chunks_used = chunks
            conf = confidence
            placeholder.markdown(full_answer + "▌")
        placeholder.markdown(full_answer)

        _render_confidence(conf)

        faithfulness = None
        if check_faithfulness and chunks_used:
            with st.spinner("Checking faithfulness…"):
                faithfulness = rag.check_faithfulness(full_answer, chunks_used)
            _render_faithfulness(faithfulness)

        if chunks_used:
            with st.expander("📄 Sources"):
                _render_sources(chunks_used)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "chunks": chunks_used,
        "confidence": conf,
        "faithfulness": faithfulness,
    })
