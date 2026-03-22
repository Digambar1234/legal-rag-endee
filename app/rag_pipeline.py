"""
RAG Pipeline — Retrieval Augmented Generation
Orchestrates the full pipeline:
  Document → Chunks → Embeddings → Endee → Query → LLM → Answer

LLM backend: HuggingFace transformers (local, no API key).
Swap to OpenAI / Anthropic by implementing a different `_generate` method.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from app.endee_client import EndeeClient
from app.embedder import EmbeddingEngine
from app.document_processor import DocumentLoader, TextChunker, DocumentChunk

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Data models                                                         #
# ------------------------------------------------------------------ #

@dataclass
class RetrievedContext:
    chunk_id: str
    score: float
    text: str
    filename: str
    doc_type: str
    section: str
    page_number: int


@dataclass
class RAGResponse:
    question: str
    answer: str
    contexts: List[RetrievedContext]
    latency_ms: float
    index_name: str
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ #
#  Indexer — build the Endee index from documents                     #
# ------------------------------------------------------------------ #

class DocumentIndexer:
    """
    Loads documents, chunks them, embeds the chunks, and upserts
    the resulting vectors into an Endee index.
    """

    def __init__(
        self,
        endee: EndeeClient,
        embedder: EmbeddingEngine,
        index_name: str = "legal_docs",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.endee = endee
        self.embedder = embedder
        self.index_name = index_name
        self.loader = DocumentLoader()
        self.chunker = TextChunker(max_chars=chunk_size, overlap_chars=chunk_overlap)

    def build_index(self, source: str, recreate: bool = False) -> Dict[str, Any]:
        """
        Build (or rebuild) the Endee index from a directory or single file.

        Args:
            source:   Path to a directory or a single document file.
            recreate: If True, delete and recreate the index before indexing.

        Returns:
            Summary dict with counts.
        """
        import os

        # Ensure Endee is reachable
        if not self.endee.health_check():
            raise ConnectionError(
                "Cannot reach Endee server. "
                "Start it with: docker compose up  OR  ./run.sh"
            )

        # Manage index
        if recreate and self.endee.index_exists(self.index_name):
            self.endee.delete_index(self.index_name)
            logger.info(f"Deleted existing index '{self.index_name}'")

        if not self.endee.index_exists(self.index_name):
            self.endee.create_index(
                name=self.index_name,
                dimension=self.embedder.dimension,
                metric="cosine",
                description="Legal document RAG index powered by Endee",
            )

        # Load documents
        if os.path.isdir(source):
            docs = self.loader.load_directory(source)
        else:
            docs = [self.loader.load(source)]

        if not docs:
            raise ValueError(f"No supported documents found in: {source}")

        total_chunks = 0
        total_docs = len(docs)

        for doc in docs:
            chunks: List[DocumentChunk] = self.chunker.chunk(doc)
            if not chunks:
                continue

            # Embed in one shot (efficient batching inside embedder)
            texts = [c.text for c in chunks]
            logger.info(f"Embedding {len(texts)} chunks from '{doc.filename}'…")
            vectors = self.embedder.embed(texts)

            # Build upsert payload
            records = [
                {
                    "id": chunk.chunk_id,
                    "values": vec,
                    "metadata": chunk.to_metadata(),
                }
                for chunk, vec in zip(chunks, vectors)
            ]

            # Upsert in batches of 200 (safe payload size)
            batch_size = 200
            for i in range(0, len(records), batch_size):
                self.endee.upsert_vectors(self.index_name, records[i: i + batch_size])

            total_chunks += len(chunks)
            logger.info(f"Indexed {len(chunks)} chunks from '{doc.filename}'")

        stats = self.endee.get_index_stats(self.index_name)
        return {
            "documents_indexed": total_docs,
            "chunks_indexed": total_chunks,
            "index_stats": stats,
        }


# ------------------------------------------------------------------ #
#  Retriever — query Endee for relevant chunks                        #
# ------------------------------------------------------------------ #

class EndeeRetriever:
    """
    Given a natural-language query, returns the top-k most relevant
    document chunks from the Endee index.
    """

    def __init__(
        self,
        endee: EndeeClient,
        embedder: EmbeddingEngine,
        index_name: str = "legal_docs",
        top_k: int = 5,
        score_threshold: float = 0.30,
    ):
        self.endee = endee
        self.embedder = embedder
        self.index_name = index_name
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        doc_type_filter: Optional[str] = None,
    ) -> List[RetrievedContext]:
        """
        Embed the query and perform ANN search in Endee.

        Args:
            query:           Natural language question.
            doc_type_filter: Optional filter — only return chunks from this doc_type.

        Returns:
            Ranked list of RetrievedContext objects.
        """
        query_vector = self.embedder.embed_query(query)

        endee_filter = None
        if doc_type_filter:
            endee_filter = {"doc_type": {"$eq": doc_type_filter}}

        matches = self.endee.query(
            index_name=self.index_name,
            vector=query_vector,
            top_k=self.top_k,
            include_metadata=True,
            filter=endee_filter,
        )

        results = []
        for m in matches:
            score = m.get("score", 0.0)
            if score < self.score_threshold:
                continue
            meta = m.get("metadata", {})
            results.append(
                RetrievedContext(
                    chunk_id=m.get("id", ""),
                    score=score,
                    text=meta.get("text", ""),
                    filename=meta.get("filename", ""),
                    doc_type=meta.get("doc_type", ""),
                    section=meta.get("section", ""),
                    page_number=meta.get("page_number", 0),
                )
            )

        logger.info(f"[Endee] Retrieved {len(results)} contexts for query: '{query[:60]}…'")
        return results


# ------------------------------------------------------------------ #
#  Generator — synthesise an answer from the retrieved contexts       #
# ------------------------------------------------------------------ #

class AnswerGenerator:
    """
    Generates a grounded answer using the retrieved legal context.

    Backends supported (configure via 'backend' param):
        'local'  — uses a small HuggingFace model (no API key needed)
        'simple' — template-based fallback (always works, no ML model)
    """

    PROMPT_TEMPLATE = """You are an expert legal AI assistant. Answer the question based ONLY on the legal document excerpts provided.
If the answer cannot be found in the excerpts, say so clearly.
Always cite the document name and section in your answer.

Legal Document Excerpts:
{context_block}

Question: {question}

Answer (be concise and precise, cite your sources):"""

    def __init__(self, backend: str = "simple"):
        self.backend = backend
        self._pipeline = None

    def generate(self, question: str, contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the indexed documents "
                "to answer your question. Please ensure the documents are indexed "
                "or rephrase your question."
            )

        context_block = self._build_context_block(contexts)
        prompt = self.PROMPT_TEMPLATE.format(
            context_block=context_block,
            question=question,
        )

        if self.backend == "local":
            return self._generate_local(prompt, contexts)
        else:
            return self._generate_simple(question, contexts)

    def _build_context_block(self, contexts: List[RetrievedContext]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            parts.append(
                f"[{i}] Source: {ctx.filename} | Section: {ctx.section} | "
                f"Relevance: {ctx.score:.2f}\n{ctx.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _generate_simple(self, question: str, contexts: List[RetrievedContext]) -> str:
        """
        Template-based answer — always works without a GPU or large model.
        Suitable for demo / evaluation purposes.
        """
        top = contexts[0]
        sources = list({c.filename for c in contexts})

        answer = (
            f"Based on the legal documents indexed:\n\n"
            f"The most relevant excerpt (relevance score: {top.score:.2f}) is from "
            f"**{top.filename}** — Section: *{top.section}*:\n\n"
            f'"{top.text[:500]}{"…" if len(top.text) > 500 else ""}"\n\n'
        )

        if len(contexts) > 1:
            answer += f"Additional supporting context was found in: {', '.join(sources[1:])}.\n"

        answer += (
            "\n📌 *Note: This answer is extracted directly from the document. "
            "Always consult a qualified legal professional for legal advice.*"
        )
        return answer

    def _generate_local(self, prompt: str, contexts: List[RetrievedContext]) -> str:
        """Use a local HuggingFace model for answer generation."""
        try:
            if self._pipeline is None:
                from transformers import pipeline
                logger.info("[Generator] Loading local LLM…")
                self._pipeline = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    max_length=512,
                )
            result = self._pipeline(prompt)[0]["generated_text"]
            return result
        except Exception as e:
            logger.warning(f"Local LLM failed ({e}), falling back to simple generator.")
            return self._generate_simple(prompt.split("Question:")[-1].strip(), contexts)


# ------------------------------------------------------------------ #
#  RAG Pipeline — top-level orchestrator                              #
# ------------------------------------------------------------------ #

class LegalRAGPipeline:
    """
    High-level interface for the Legal Document RAG system.

    Typical usage:
        pipeline = LegalRAGPipeline()
        pipeline.index_documents("./data/sample_docs")
        response = pipeline.ask("What is the notice period in the NDA?")
        print(response.answer)
    """

    def __init__(
        self,
        endee_url: str = "http://localhost:8080",
        endee_auth_token: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_name: str = "legal_docs",
        top_k: int = 5,
        generator_backend: str = "simple",
    ):
        self.endee = EndeeClient(base_url=endee_url, auth_token=endee_auth_token)
        self.embedder = EmbeddingEngine(model_name=embedding_model)
        self.index_name = index_name

        self.indexer = DocumentIndexer(
            endee=self.endee,
            embedder=self.embedder,
            index_name=index_name,
        )
        self.retriever = EndeeRetriever(
            endee=self.endee,
            embedder=self.embedder,
            index_name=index_name,
            top_k=top_k,
        )
        self.generator = AnswerGenerator(backend=generator_backend)

    # -- public API ------------------------------------------------

    def index_documents(self, source: str, recreate: bool = False) -> Dict[str, Any]:
        """Index all supported documents from a path."""
        return self.indexer.build_index(source, recreate=recreate)

    def ask(
        self,
        question: str,
        doc_type_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Answer a question using the RAG pipeline backed by Endee.

        Args:
            question:        The natural language question.
            doc_type_filter: Optionally restrict retrieval to a doc type
                             (e.g., 'contract', 'policy').

        Returns:
            RAGResponse with answer, source contexts, and metadata.
        """
        t0 = time.perf_counter()

        contexts = self.retriever.retrieve(question, doc_type_filter=doc_type_filter)
        answer = self.generator.generate(question, contexts)

        latency_ms = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            question=question,
            answer=answer,
            contexts=contexts,
            latency_ms=latency_ms,
            index_name=self.index_name,
            model_used=self.embedder.model_name,
            metadata={
                "num_contexts": len(contexts),
                "doc_type_filter": doc_type_filter,
                "generator_backend": self.generator.backend,
            },
        )

    def semantic_search(self, query: str, top_k: int = 10) -> List[RetrievedContext]:
        """
        Raw semantic search — returns ranked chunks without answer generation.
        Useful for exploration and document discovery.
        """
        self.retriever.top_k = top_k
        return self.retriever.retrieve(query)

    def get_index_stats(self) -> Dict[str, Any]:
        """Return Endee index statistics."""
        return self.endee.get_index_stats(self.index_name)
