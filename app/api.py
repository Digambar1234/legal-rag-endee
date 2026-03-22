"""
Legal RAG Assistant — REST API
FastAPI server exposing the RAG pipeline as HTTP endpoints.

Run with:
    uvicorn app.api:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_pipeline import LegalRAGPipeline, RAGResponse, RetrievedContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  App setup                                                           #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="Legal RAG Assistant API",
    description=(
        "A Retrieval-Augmented Generation system for legal documents, "
        "powered by **Endee** vector database."
    ),
    version="1.0.0",
    contact={"name": "Legal RAG", "url": "https://github.com/endee-io/endee"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Pipeline (singleton)                                               #
# ------------------------------------------------------------------ #

_pipeline: Optional[LegalRAGPipeline] = None

def get_pipeline() -> LegalRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LegalRAGPipeline(
            endee_url=os.getenv("ENDEE_URL", "http://localhost:8080"),
            endee_auth_token=os.getenv("ENDEE_AUTH_TOKEN") or None,
            index_name=os.getenv("ENDEE_INDEX", "legal_docs"),
            top_k=int(os.getenv("TOP_K", "5")),
        )
    return _pipeline


# ------------------------------------------------------------------ #
#  Pydantic models                                                     #
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language legal question")
    doc_type_filter: Optional[str] = Field(
        None,
        description="Optionally restrict to doc type: contract | policy | court_order | ip_document",
    )

class ContextOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    filename: str
    doc_type: str
    section: str
    page_number: int

class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: List[ContextOut]
    latency_ms: float
    index_name: str
    model_used: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(5, ge=1, le=50)
    doc_type_filter: Optional[str] = None

class IndexResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    message: str

class HealthResponse(BaseModel):
    status: str
    endee_connected: bool
    index_name: str


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/", tags=["General"])
def root():
    return {
        "name": "Legal RAG Assistant",
        "vector_db": "Endee (endee-io/endee)",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    pipeline = get_pipeline()
    connected = pipeline.endee.health_check()
    return HealthResponse(
        status="ok" if connected else "degraded",
        endee_connected=connected,
        index_name=pipeline.index_name,
    )


@app.post("/index/files", response_model=IndexResponse, tags=["Indexing"])
async def index_uploaded_files(
    files: List[UploadFile] = File(...),
    recreate: bool = Form(False),
):
    """Upload and index document files (PDF, DOCX, TXT)."""
    pipeline = get_pipeline()

    if not pipeline.endee.health_check():
        raise HTTPException(503, "Endee server is not reachable. Start it first.")

    tmp_dir = tempfile.mkdtemp()
    for uf in files:
        dest = Path(tmp_dir) / uf.filename
        content = await uf.read()
        dest.write_bytes(content)

    try:
        result = pipeline.index_documents(tmp_dir, recreate=recreate)
    except Exception as e:
        raise HTTPException(500, str(e))

    return IndexResponse(
        documents_indexed=result["documents_indexed"],
        chunks_indexed=result["chunks_indexed"],
        message="Indexing complete",
    )


@app.post("/index/directory", response_model=IndexResponse, tags=["Indexing"])
def index_directory(
    directory: str = Form("./data/sample_docs"),
    recreate: bool = Form(False),
):
    """Index documents from a server-side directory path."""
    pipeline = get_pipeline()

    if not pipeline.endee.health_check():
        raise HTTPException(503, "Endee server is not reachable.")

    if not Path(directory).exists():
        raise HTTPException(404, f"Directory not found: {directory}")

    try:
        result = pipeline.index_documents(directory, recreate=recreate)
    except Exception as e:
        raise HTTPException(500, str(e))

    return IndexResponse(
        documents_indexed=result["documents_indexed"],
        chunks_indexed=result["chunks_indexed"],
        message="Indexing complete",
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(req: AskRequest):
    """
    Answer a legal question using RAG backed by Endee vector search.
    """
    pipeline = get_pipeline()

    if not pipeline.endee.health_check():
        raise HTTPException(503, "Endee server is not reachable.")

    try:
        response: RAGResponse = pipeline.ask(req.question, req.doc_type_filter)
    except Exception as e:
        raise HTTPException(500, str(e))

    return AskResponse(
        question=response.question,
        answer=response.answer,
        contexts=[
            ContextOut(
                chunk_id=c.chunk_id,
                score=c.score,
                text=c.text,
                filename=c.filename,
                doc_type=c.doc_type,
                section=c.section,
                page_number=c.page_number,
            )
            for c in response.contexts
        ],
        latency_ms=response.latency_ms,
        index_name=response.index_name,
        model_used=response.model_used,
    )


@app.post("/search", response_model=List[ContextOut], tags=["RAG"])
def semantic_search(req: SearchRequest):
    """
    Pure semantic search — returns ranked document chunks without answer generation.
    """
    pipeline = get_pipeline()

    if not pipeline.endee.health_check():
        raise HTTPException(503, "Endee server is not reachable.")

    try:
        results = pipeline.semantic_search(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, str(e))

    return [
        ContextOut(
            chunk_id=c.chunk_id,
            score=c.score,
            text=c.text,
            filename=c.filename,
            doc_type=c.doc_type,
            section=c.section,
            page_number=c.page_number,
        )
        for c in results
    ]


@app.get("/index/stats", tags=["Indexing"])
def index_stats():
    """Return Endee index statistics."""
    pipeline = get_pipeline()
    try:
        return pipeline.get_index_stats()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/index", tags=["Indexing"])
def delete_index():
    """Delete the current Endee index (destructive!)."""
    pipeline = get_pipeline()
    try:
        pipeline.endee.delete_index(pipeline.index_name)
        return {"message": f"Index '{pipeline.index_name}' deleted."}
    except Exception as e:
        raise HTTPException(500, str(e))
