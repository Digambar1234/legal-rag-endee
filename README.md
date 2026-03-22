# ⚖️ Legal RAG Assistant — Powered by Endee Vector Database

> **Retrieval-Augmented Generation for Legal Documents**  
> Ask natural-language questions over contracts, policies, and court orders — answers grounded in your documents, retrieved at speed by [Endee](https://github.com/endee-io/endee).

---

## 📋 Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)
- [System Design & Technical Approach](#system-design--technical-approach)
- [How Endee Is Used](#how-endee-is-used)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Sample Queries](#sample-queries)
- [Tech Stack](#tech-stack)

---

## Overview & Problem Statement

Legal professionals and students regularly need to extract specific information from long, dense documents — NDAs, employment contracts, privacy policies, court orders. Reading every document manually is slow and error-prone.

**This project solves that problem** by building a Retrieval-Augmented Generation (RAG) system that:

1. **Ingests** legal documents (PDF, DOCX, TXT)
2. **Chunks and embeds** them into 384-dimensional semantic vectors
3. **Stores** those vectors in **Endee** — a high-performance open-source vector database
4. **Retrieves** the most relevant chunks using Approximate Nearest Neighbour (ANN) search
5. **Generates** a grounded, cited answer using the retrieved context

The result: a legal assistant that can answer "What is the notice period in the employment contract?" in under 500ms, with direct citations to the source document.

---

## System Design & Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                           │
│                                                                     │
│  PDF/DOCX/TXT  ──►  DocumentLoader  ──►  TextChunker               │
│                                          (800 chars, 150 overlap)   │
│                                               │                     │
│                                               ▼                     │
│                                    EmbeddingEngine                  │
│                                    (all-MiniLM-L6-v2, 384-dim)      │
│                                               │                     │
│                                               ▼                     │
│                                    ┌──────────────────┐             │
│                                    │   ENDEE (nDD)    │             │
│                                    │  Vector Database  │             │
│                                    │  HNSW + Cosine   │             │
│                                    │  Up to 1B vectors│             │
│                                    └──────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
│                                                                     │
│  User Question  ──►  embed_query()  ──►  Endee ANN Search           │
│                                               │                     │
│                                       Top-K chunks + scores         │
│                                               │                     │
│                                               ▼                     │
│                                    AnswerGenerator                  │
│                                    (context-grounded)               │
│                                               │                     │
│                        ┌─────────────────────┼────────────────┐    │
│                         ▼                     ▼                ▼    │
│                    Streamlit UI           FastAPI          CLI Tool  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector DB | **Endee** | High-performance HNSW, open-source, REST API, Docker-native |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | Fast CPU inference, excellent semantic quality |
| Chunking strategy | Overlap with section-aware splitting | Preserves context across clause boundaries |
| Distance metric | **Cosine similarity** | Language embeddings are best compared directionally |
| Answer generation | Template + optional local LLM | Always works; no API key required |

### Chunking Strategy

Legal documents have a natural hierarchical structure (Articles → Sections → Clauses). The `TextChunker` class:

1. Detects section headings via regex (ALL-CAPS, numbered sections, "ARTICLE N")
2. Splits each section into paragraph-level chunks of ≤800 characters
3. Applies 150-character overlap between chunks to avoid losing cross-boundary context
4. Falls back to sliding-window splitting for exceptionally long single paragraphs

---

## How Endee Is Used

[Endee](https://github.com/endee-io/endee) is the **core vector storage and retrieval engine** of this project. Every interaction with indexed documents goes through Endee.

### Endee API Endpoints Used

| Operation | Endee Endpoint | Used When |
|-----------|---------------|-----------|
| Create index | `POST /api/v1/index/create` | First time indexing |
| Upsert vectors | `POST /api/v1/index/{name}/vectors/upsert` | Adding document chunks |
| Query (ANN search) | `POST /api/v1/index/{name}/query` | Every user question |
| Get stats | `GET /api/v1/index/{name}/stats` | Dashboard display |
| Delete index | `DELETE /api/v1/index/{name}` | Re-indexing from scratch |
| List indexes | `GET /api/v1/index/list` | Health check |

### Index Configuration

```python
endee.create_index(
    name="legal_docs",
    dimension=384,        # matches all-MiniLM-L6-v2 output
    metric="cosine",      # cosine similarity for normalized embeddings
    description="Legal document RAG index powered by Endee",
)
```

### Vector Schema in Endee

Each vector stored in Endee carries rich metadata:

```json
{
  "id": "uuid-chunk-id",
  "values": [0.023, -0.145, ..., 0.087],
  "metadata": {
    "chunk_id": "uuid",
    "doc_id": "uuid",
    "filename": "employment_contract.txt",
    "doc_type": "contract",
    "section": "SECTION 5 – TERMINATION",
    "text": "Either party may terminate this Contract by providing 90 days' written notice…",
    "char_start": 4812,
    "char_end": 5400,
    "page_number": 3
  }
}
```

### Metadata Filtering (Endee Feature)

Endee supports **server-side metadata filtering**, enabling targeted retrieval:

```python
# Only search within contract-type documents
endee.query(
    index_name="legal_docs",
    vector=query_embedding,
    top_k=5,
    filter={"doc_type": {"$eq": "contract"}},
)
```

This allows users to restrict answers to specific document categories without post-filtering.

---

## Project Structure

```
legal-rag-endee/
│
├── app/
│   ├── __init__.py
│   ├── endee_client.py       # Endee REST API client wrapper
│   ├── document_processor.py # Document loader + intelligent text chunker
│   ├── embedder.py           # Sentence-transformer embedding engine
│   ├── rag_pipeline.py       # Full RAG orchestrator (Indexer + Retriever + Generator)
│   ├── streamlit_app.py      # Streamlit web UI
│   └── api.py                # FastAPI REST backend
│
├── data/
│   └── sample_docs/          # Sample legal documents for demo
│       ├── nda_agreement.txt
│       ├── employment_contract.txt
│       └── privacy_policy.txt
│
├── scripts/
│   └── index_and_query.py    # CLI tool for indexing and querying
│
├── tests/
│   └── test_all.py           # Unit + integration test suite (pytest)
│
├── docs/
│   └── architecture.mermaid  # System architecture diagram
│
├── demo.py                   # Standalone demo (no Endee server needed)
├── docker-compose.yml        # One-command deployment (Endee + UI + API)
├── Dockerfile                # Python app container
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Endee)
- Git

### Step 1 — Fork and Clone the Endee Repository

As required by the project guidelines:

```bash
# 1. Star the repo at https://github.com/endee-io/endee
# 2. Fork it to your GitHub account
# 3. Clone YOUR fork:
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee
```

### Step 2 — Clone This Project

```bash
git clone https://github.com/YOUR_USERNAME/legal-rag-endee.git
cd legal-rag-endee
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Start Endee Vector Database

**Option A — Docker (recommended, no build needed):**

```bash
docker compose up endee -d
```

**Option B — From your forked Endee repo:**

```bash
cd /path/to/your/endee-fork
./install.sh --release --avx2      # Intel/AMD
# OR
./install.sh --release --neon      # Apple Silicon
./run.sh
```

Verify Endee is running:

```bash
curl http://localhost:8080/api/v1/index/list
# Expected: {"indexes": []}
```

---

## Running the Application

### 🖥️ Streamlit Web UI (recommended)

```bash
streamlit run app/streamlit_app.py
```

Open: http://localhost:8501

1. Click **"Index Documents"** in the sidebar (uses sample docs automatically)
2. Type a question in the **Q&A Chat** tab
3. Explore the **Semantic Search** tab for raw vector search results

### 🚀 Full Stack with Docker Compose

Starts Endee + Streamlit UI + FastAPI all together:

```bash
docker compose up
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI + Swagger | http://localhost:8000/docs |
| Endee | http://localhost:8080 |

### 🔧 CLI Tool

```bash
# Index sample documents
python scripts/index_and_query.py index ./data/sample_docs

# Ask a question
python scripts/index_and_query.py ask "What is the notice period for termination?"

# Semantic search
python scripts/index_and_query.py search "intellectual property assignment"

# Show index stats
python scripts/index_and_query.py stats
```

### 🎯 Demo Mode (no Endee server required)

```bash
python demo.py
```

Runs the full pipeline with an in-memory Endee mock — great for quick demonstrations.

---

## API Reference

The FastAPI server (port 8000) exposes:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Endee connection status |
| POST | `/index/files` | Upload and index document files |
| POST | `/index/directory` | Index from a server-side path |
| POST | `/ask` | Answer a question via RAG |
| POST | `/search` | Semantic search (no generation) |
| GET | `/index/stats` | Endee index statistics |
| DELETE | `/index` | Clear the index |

**Example — Ask a question:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the data retention period?", "doc_type_filter": "policy"}'
```

**Example — Semantic Search:**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "confidentiality obligations after termination", "top_k": 5}'
```

---

## Running Tests

```bash
# Unit tests (no Endee server needed)
pytest tests/ -v

# Unit + integration tests (requires live Endee on localhost:8080)
pytest tests/ -v -m integration
```

---

## Sample Queries

These questions work against the included sample documents:

| Question | Expected Source |
|----------|----------------|
| What is the notice period for termination? | employment_contract.txt |
| How long does the NDA confidentiality obligation last after termination? | nda_agreement.txt |
| What data security measures are described? | privacy_policy.txt |
| Can the employee work for a competitor after leaving? | employment_contract.txt |
| What rights do users have over their personal data? | privacy_policy.txt |
| Who owns intellectual property created during employment? | employment_contract.txt |
| What information qualifies as Confidential Information? | nda_agreement.txt |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) (HNSW, cosine, REST API) |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` (384-dim) |
| **Web UI** | [Streamlit](https://streamlit.io/) |
| **REST API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **PDF Parsing** | [pypdf](https://pypdf.readthedocs.io/) |
| **DOCX Parsing** | [python-docx](https://python-docx.readthedocs.io/) |
| **Containerisation** | Docker + Docker Compose |
| **Testing** | pytest |
| **Language** | Python 3.11 |

---

## Author

Built as part of the Endee AI/ML project submission.

**Repository:** https://github.com/YOUR_USERNAME/legal-rag-endee  
**Endee Fork:** https://github.com/YOUR_USERNAME/endee  

> ⭐ Don't forget to star [endee-io/endee](https://github.com/endee-io/endee)!
