"""
demo.py — Run the Legal RAG Assistant without a live Endee server.

This demo uses an in-memory mock of Endee so you can see the full
pipeline working (chunking, embedding, retrieval, answer generation)
without having to start Docker.

Run with:
    python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import uuid
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.WARNING)

# ------------------------------------------------------------------ #
#  In-memory mock of Endee for demo/testing                           #
# ------------------------------------------------------------------ #

class InMemoryEndee:
    """
    Lightweight in-memory vector store that mirrors the Endee API surface.
    Uses brute-force cosine similarity (fine for demo with small corpora).
    """

    def __init__(self):
        self._indexes: Dict[str, Dict] = {}

    def health_check(self) -> bool:
        return True

    def create_index(self, name, dimension, metric="cosine", description=""):
        self._indexes[name] = {"dimension": dimension, "metric": metric, "vectors": {}}
        return {"status": "created", "name": name}

    def delete_index(self, name):
        self._indexes.pop(name, None)
        return {"status": "deleted"}

    def list_indexes(self):
        return list(self._indexes.keys())

    def index_exists(self, name):
        return name in self._indexes

    def upsert_vectors(self, index_name, vectors):
        store = self._indexes[index_name]["vectors"]
        for v in vectors:
            store[v["id"]] = {"values": v["values"], "metadata": v.get("metadata", {})}
        return {"upserted": len(vectors)}

    def query(self, index_name, vector, top_k=5, include_metadata=True, filter=None):
        store = self._indexes[index_name]["vectors"]
        scored = []
        for vid, vdata in store.items():
            score = self._cosine(vector, vdata["values"])
            scored.append({"id": vid, "score": score, "metadata": vdata["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_index_stats(self, index_name):
        store = self._indexes.get(index_name, {})
        return {"vector_count": len(store.get("vectors", {}))}

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ------------------------------------------------------------------ #
#  Demo runner                                                         #
# ------------------------------------------------------------------ #

def run_demo():
    from app.document_processor import DocumentLoader, TextChunker
    from app.embedder import EmbeddingEngine
    from app.rag_pipeline import EndeeRetriever, AnswerGenerator, RAGResponse
    import time

    print("\n" + "="*65)
    print("  ⚖️  Legal RAG Assistant — Demo (In-Memory Endee Mock)")
    print("="*65 + "\n")

    # Step 1: Load documents
    print("📄 Step 1: Loading legal documents from ./data/sample_docs …")
    loader = DocumentLoader()
    docs = loader.load_directory("./data/sample_docs")
    print(f"   Loaded {len(docs)} documents: {[d.filename for d in docs]}\n")

    # Step 2: Chunk
    print("✂️  Step 2: Chunking documents into overlapping text segments …")
    chunker = TextChunker(max_chars=600, overlap_chars=100)
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        print(f"   {doc.filename}: {len(chunks)} chunks")
    print(f"   Total: {len(all_chunks)} chunks\n")

    # Step 3: Embed
    print("🤖 Step 3: Generating embeddings with 'all-MiniLM-L6-v2' (384-dim) …")
    embedder = EmbeddingEngine("all-MiniLM-L6-v2")
    t0 = time.time()
    texts = [c.text for c in all_chunks]
    vectors = embedder.embed(texts)
    embed_time = time.time() - t0
    print(f"   Embedded {len(vectors)} chunks in {embed_time:.2f}s\n")

    # Step 4: Upsert into in-memory Endee mock
    print("🗄️  Step 4: Upserting vectors into Endee (in-memory mock) …")
    endee = InMemoryEndee()
    endee.create_index("legal_docs", dimension=embedder.dimension, metric="cosine")

    records = [
        {"id": c.chunk_id, "values": vec, "metadata": c.to_metadata()}
        for c, vec in zip(all_chunks, vectors)
    ]
    endee.upsert_vectors("legal_docs", records)
    stats = endee.get_index_stats("legal_docs")
    print(f"   Index 'legal_docs' now has {stats['vector_count']} vectors\n")

    # Step 5: RAG Q&A
    print("💬 Step 5: Answering questions via Endee vector search + RAG\n")
    retriever = EndeeRetriever(endee, embedder, index_name="legal_docs", top_k=3, score_threshold=0.20)
    generator = AnswerGenerator(backend="simple")

    questions = [
        "What is the notice period for termination of employment?",
        "How long must confidential information be kept secret after the NDA ends?",
        "What rights do users have under the privacy policy?",
        "What happens to intellectual property created during employment?",
        "What are the data security measures in place?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        t0 = time.perf_counter()
        contexts = retriever.retrieve(question)
        answer = generator.generate(question, contexts)
        latency = (time.perf_counter() - t0) * 1000

        print(f"   📚 Retrieved {len(contexts)} contexts from Endee")
        if contexts:
            top = contexts[0]
            print(f"   🏆 Top match: {top.filename} | Score: {top.score:.4f} | Section: {top.section[:50]}")
        print(f"   ⏱  Latency: {latency:.1f}ms")
        print(f"   💡 Answer preview: {answer[:200].strip()}…")
        print()

    # Step 6: Semantic search demo
    print("="*65)
    print("🔍 Step 6: Pure Semantic Search (no generation)\n")
    query = "intellectual property ownership assignment"
    print(f"   Query: '{query}'")
    retriever.top_k = 5
    results = retriever.retrieve(query)
    print(f"   Top {len(results)} results from Endee:\n")
    for j, r in enumerate(results, 1):
        print(f"   [{j}] {r.filename} | Score: {r.score:.4f}")
        print(f"       Section: {r.section[:60]}")
        print(f"       Excerpt: {r.text[:120].strip()}…\n")

    print("="*65)
    print("✅ Demo complete! All components working correctly.")
    print()
    print("🚀 To run the full app with real Endee:")
    print("   docker compose up")
    print("   streamlit run app/streamlit_app.py")
    print("   uvicorn app.api:app --reload  (for the REST API)")
    print("="*65 + "\n")


if __name__ == "__main__":
    run_demo()
