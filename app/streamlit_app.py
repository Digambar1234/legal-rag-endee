"""
Legal RAG Assistant — Streamlit UI
Run with:  streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time
import json
from pathlib import Path
from typing import Optional

# ------------------------------------------------------------------ #
#  Page config (must be first Streamlit call)                         #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Legal RAG Assistant | Powered by Endee",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS                                                          #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .main-header h1 { font-size: 2.2rem; margin: 0; }
  .main-header p  { opacity: 0.8; margin: 0.5rem 0 0; font-size: 1rem; }

  .endee-badge {
    background: #0f3460;
    color: #e2b04f;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    display: inline-block;
    margin-bottom: 0.5rem;
  }

  .context-card {
    background: #f8fafc;
    border-left: 4px solid #0f3460;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
    font-size: 0.9rem;
  }
  .context-card .score-badge {
    background: #0f3460;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    float: right;
  }

  .answer-box {
    background: #eef7ee;
    border: 1px solid #b2d8b2;
    padding: 1.2rem;
    border-radius: 10px;
    margin-top: 1rem;
  }

  .metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 0.5rem 0;
  }
  .metric-pill {
    background: #e8edf5;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #1a1a2e;
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session state initialisation                                       #
# ------------------------------------------------------------------ #
def init_state():
    defaults = {
        "pipeline": None,
        "indexed": False,
        "chat_history": [],
        "index_stats": {},
        "endee_connected": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ------------------------------------------------------------------ #
#  Sidebar — configuration & document upload                          #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.image("https://endee.io/favicon.ico", width=40)
    st.markdown("### ⚙️ Configuration")

    endee_url = st.text_input(
        "Endee Server URL",
        value="http://localhost:8080",
        help="URL of your running Endee instance.",
    )
    auth_token = st.text_input(
        "Auth Token (optional)",
        value="",
        type="password",
        help="Leave blank if running Endee without authentication.",
    )

    st.divider()

    st.markdown("### 🗄️ Endee Index")
    index_name = st.text_input("Index Name", value="legal_docs")
    top_k = st.slider("Top-K Results", min_value=1, max_value=15, value=5)

    st.divider()

    st.markdown("### 📄 Document Ingestion")
    upload_mode = st.radio("Source", ["Upload Files", "Use Sample Documents"])

    uploaded_files = []
    if upload_mode == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "md"],
        )

    if st.button("🚀 Index Documents", use_container_width=True, type="primary"):
        from app.rag_pipeline import LegalRAGPipeline
        import tempfile, shutil

        try:
            pipeline = LegalRAGPipeline(
                endee_url=endee_url,
                endee_auth_token=auth_token or None,
                index_name=index_name,
                top_k=top_k,
            )

            with st.spinner("Connecting to Endee…"):
                if not pipeline.endee.health_check():
                    st.error(
                        "❌ Cannot reach Endee server.\n\n"
                        "Start it with:\n```\ndocker compose up\n```\n"
                        "or `./run.sh` from the Endee repo."
                    )
                    st.stop()

            st.session_state.endee_connected = True

            # Determine source directory
            if upload_mode == "Upload Files" and uploaded_files:
                tmp_dir = tempfile.mkdtemp()
                for uf in uploaded_files:
                    path = os.path.join(tmp_dir, uf.name)
                    with open(path, "wb") as f:
                        f.write(uf.getbuffer())
                source = tmp_dir
            else:
                source = "./data/sample_docs"

            with st.spinner("Indexing… this may take a moment."):
                result = pipeline.index_documents(source, recreate=True)

            st.session_state.pipeline = pipeline
            st.session_state.indexed = True
            st.session_state.index_stats = result

            st.success(
                f"✅ Indexed {result['documents_indexed']} document(s), "
                f"{result['chunks_indexed']} chunks."
            )

        except Exception as e:
            st.error(f"Indexing failed: {e}")

    if st.session_state.indexed and st.session_state.index_stats:
        stats = st.session_state.index_stats
        st.markdown("**Index Stats**")
        st.json({
            "Documents": stats.get("documents_indexed", "—"),
            "Chunks": stats.get("chunks_indexed", "—"),
        })

    st.divider()
    st.markdown(
        '<div class="endee-badge">Powered by Endee</div>', unsafe_allow_html=True
    )
    st.caption("High-performance vector DB — up to 1B vectors on a single node.")


# ------------------------------------------------------------------ #
#  Main area                                                           #
# ------------------------------------------------------------------ #
st.markdown("""
<div class="main-header">
  <div class="endee-badge">ENDEE VECTOR DATABASE</div>
  <h1>⚖️ Legal RAG Assistant</h1>
  <p>Ask natural-language questions over your legal documents. Powered by semantic search via Endee + AI-generated answers.</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["💬 Q&A Chat", "🔍 Semantic Search", "📊 System Info"])


# ---- Tab 1: Q&A Chat ------------------------------------------------
with tabs[0]:
    if not st.session_state.indexed:
        st.info("👈 Configure and index documents using the sidebar first.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            question = st.text_input(
                "Ask a legal question",
                placeholder="e.g. What are the termination clauses in the contract?",
                key="question_input",
            )
        with col2:
            doc_filter = st.selectbox(
                "Filter by document type",
                ["All", "contract", "policy", "court_order", "ip_document", "legal_document"],
                index=0,
            )

        if st.button("Ask ⚡", type="primary") and question:
            pipeline = st.session_state.pipeline
            filter_val = None if doc_filter == "All" else doc_filter

            with st.spinner("Searching Endee for relevant context…"):
                response = pipeline.ask(question, doc_type_filter=filter_val)

            # Store in chat history
            st.session_state.chat_history.append(response)

        # Display latest response
        if st.session_state.chat_history:
            resp = st.session_state.chat_history[-1]

            st.markdown(f"**Q:** {resp.question}")

            st.markdown(
                f'<div class="answer-box">{resp.answer}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="metric-row">'
                f'<span class="metric-pill">⏱ {resp.latency_ms:.0f} ms</span>'
                f'<span class="metric-pill">📚 {len(resp.contexts)} contexts</span>'
                f'<span class="metric-pill">🤖 {resp.model_used}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if resp.contexts:
                with st.expander(f"📎 Source Contexts ({len(resp.contexts)})"):
                    for i, ctx in enumerate(resp.contexts, 1):
                        st.markdown(
                            f'<div class="context-card">'
                            f'<span class="score-badge">Score: {ctx.score:.3f}</span>'
                            f'<strong>[{i}] {ctx.filename}</strong> — {ctx.section}<br/><br/>'
                            f'{ctx.text[:400]}{"…" if len(ctx.text) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # Chat history (older entries)
        if len(st.session_state.chat_history) > 1:
            with st.expander("🕘 Previous Questions"):
                for r in reversed(st.session_state.chat_history[:-1]):
                    st.markdown(f"**Q:** {r.question}")
                    st.markdown(f"_{r.answer[:200]}…_")
                    st.divider()


# ---- Tab 2: Semantic Search -----------------------------------------
with tabs[1]:
    if not st.session_state.indexed:
        st.info("👈 Index documents first.")
    else:
        query = st.text_input(
            "Semantic Search Query",
            placeholder="e.g. intellectual property ownership",
            key="search_input",
        )
        k = st.slider("Number of results", 1, 20, 8, key="search_k")

        if st.button("Search 🔎") and query:
            pipeline = st.session_state.pipeline
            pipeline.retriever.top_k = k

            with st.spinner("Querying Endee…"):
                results = pipeline.semantic_search(query, top_k=k)

            st.success(f"Found {len(results)} results")

            for i, ctx in enumerate(results, 1):
                with st.expander(
                    f"[{i}] {ctx.filename} | Score: {ctx.score:.4f} | {ctx.section[:60]}"
                ):
                    st.markdown(f"**Document:** {ctx.filename}  |  **Type:** `{ctx.doc_type}`")
                    st.markdown(f"**Section:** {ctx.section}")
                    st.markdown(f"**Relevance Score:** {ctx.score:.4f}")
                    st.text_area("Excerpt", value=ctx.text, height=120, key=f"exc_{i}")


# ---- Tab 3: System Info ---------------------------------------------
with tabs[2]:
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Legal RAG Assistant                              │
    │                                                                     │
    │  ┌──────────┐    ┌───────────────┐    ┌─────────────────────────┐  │
    │  │  Document│───▶│ Text Chunker  │───▶│   Embedding Engine      │  │
    │  │  Loader  │    │ (overlap=150) │    │ (all-MiniLM-L6-v2 384d) │  │
    │  └──────────┘    └───────────────┘    └────────────┬────────────┘  │
    │                                                    │               │
    │                                                    ▼               │
    │                                       ┌────────────────────────┐   │
    │                                       │   ENDEE VECTOR DB      │   │
    │                                       │  (cosine, HNSW index)  │   │
    │                                       └────────────┬───────────┘   │
    │                                                    │               │
    │  ┌──────────────┐    ┌────────────────┐            │               │
    │  │   Streamlit  │◀───│ Answer         │◀───────────┘               │
    │  │   UI / API   │    │ Generator      │  ANN search results        │
    │  └──────────────┘    └────────────────┘                            │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### 🔧 Why Endee?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Vectors", "1 Billion", "Single node")
    with col2:
        st.metric("Index Type", "HNSW", "Approximate NN")
    with col3:
        st.metric("Metrics", "Cosine / L2 / Dot", "Configurable")

    st.markdown("""
    Endee is a high-performance open-source vector database that makes this
    project possible. Key advantages used here:
    - **HNSW indexing** for fast approximate nearest-neighbour search
    - **REST API** for language-agnostic integration
    - **Metadata filtering** for document-type-aware retrieval
    - **Docker deployment** — no cloud dependency, fully local
    """)

    if st.session_state.endee_connected:
        st.success("✅ Endee server is connected and healthy")
    else:
        st.warning("⚠️ Endee not yet connected — index a document to connect.")
