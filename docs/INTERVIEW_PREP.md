# 🎯 Interview Preparation Guide — Legal RAG Assistant

This guide helps you explain your project confidently in an interview.
Read it once carefully and you'll be able to answer every question.

---

## 🏗️ How To Explain The Project In 60 Seconds

> "I built a Legal Document Q&A system using RAG — Retrieval Augmented Generation.
> The user uploads legal documents like contracts or NDAs. My system chunks those
> documents into small pieces, converts each chunk into a 384-dimensional vector
> using sentence-transformers, and stores those vectors in Endee — the vector database
> we were required to use. When someone asks a question, I embed that question the
> same way, and Endee finds the most similar document chunks using approximate
> nearest-neighbour search. I then use the retrieved chunks to construct a grounded
> answer. The whole thing runs as a Streamlit web app with a FastAPI backend, and
> everything is containerised with Docker."

---

## ❓ Common Interview Questions and Answers

---

### Q1: What is RAG and why did you use it?

**Answer:**
RAG stands for Retrieval-Augmented Generation. Instead of training or fine-tuning a language
model on legal documents (which is expensive and slow), I embed the documents as vectors and
retrieve relevant chunks at query time. The retrieved chunks are then given to the language
model as context.

This has three advantages:
1. **Accuracy** — The model answers based only on real document content, reducing hallucination
2. **Up-to-date** — Adding new documents doesn't require retraining, just re-indexing
3. **Citations** — We know exactly which document the answer came from

---

### Q2: Why did you choose Endee as the vector database?

**Answer:**
The project required Endee, but it's also genuinely a good fit. Endee is designed for
high-performance vector search — up to 1 billion vectors on a single node using HNSW indexing.
For a legal document system at enterprise scale, that headroom matters. It's also open-source,
deployable via Docker without any cloud dependency, and exposes a clean REST API that I could
wrap in a Python client class.

Compared to alternatives like Pinecone (managed cloud, paid) or FAISS (no persistence, in-memory
only), Endee gives us persistent storage, fast ANN search, and metadata filtering in one package.

---

### Q3: What is HNSW? How does Endee find similar vectors?

**Answer:**
HNSW stands for Hierarchical Navigable Small World. It's a graph-based index structure for
approximate nearest-neighbour search. Instead of comparing a query vector against every stored
vector (which would be O(n)), HNSW builds a multi-layer graph where each node connects to its
nearest neighbours. Searching starts at the top layer with few nodes and navigates down,
getting more precise at each level.

The result is sub-linear search time — for millions of vectors you might need to compare only
a few thousand candidates, not all of them. The trade-off is it's approximate (not exact), but
for semantic search that's fine — we don't need the mathematically perfect nearest neighbour,
just a good one.

---

### Q4: Why cosine similarity and not Euclidean distance?

**Answer:**
Language embeddings encode meaning as a direction in vector space, not as an absolute position.
Two sentences with similar meaning point in the same direction even if their magnitudes differ.

Cosine similarity measures the angle between two vectors (cos θ), so it captures this
directional similarity regardless of vector length. Euclidean distance would incorrectly
penalize vectors that are semantically similar but have different magnitudes.

I normalize all embeddings before storing them (the sentence-transformers library does this by
default when `normalize_embeddings=True`), which also makes cosine similarity equivalent to
dot product — the fastest operation.

---

### Q5: Why did you chunk the documents? Why not embed the whole document?

**Answer:**
Three reasons:

1. **Embedding model limits** — `all-MiniLM-L6-v2` has a max token length of 256 tokens
   (~1,000 characters). A 50-page contract would be truncated.

2. **Retrieval precision** — If I embed a whole document, one vector has to represent hundreds
   of clauses. It will match some queries but not others. Smaller chunks are more focused and
   retrieve more precisely.

3. **Citation quality** — With chunk-level retrieval I can show the user exactly which clause
   or section answered their question, not just "it's in this document".

I use 800-character chunks with 150-character overlap so context isn't lost at chunk boundaries.

---

### Q6: Walk me through what happens when a user asks a question.

**Answer:**
Step by step:

1. User types a question in the Streamlit UI
2. The `EmbeddingEngine` encodes the question into a 384-dimensional vector using `all-MiniLM-L6-v2`
3. That vector is sent to Endee via `POST /api/v1/index/legal_docs/query`
4. Endee uses HNSW to find the top-5 most similar chunk vectors (by cosine similarity)
5. Endee returns the chunk IDs, scores, and metadata (including the original text)
6. The `AnswerGenerator` constructs an answer using the retrieved texts, citing filename and section
7. The response is displayed in the UI with latency, source citations, and individual chunk scores

The total latency is typically 50–300ms depending on hardware.

---

### Q7: What embedding model did you use and why?

**Answer:**
I used `sentence-transformers/all-MiniLM-L6-v2`. Reasons:
- **Free and local** — No API key, runs on CPU
- **384 dimensions** — Small enough to be fast, large enough for good quality
- **Trained for semantic similarity** — The training objective matches our use case exactly
- **Normalized outputs** — Makes cosine similarity computation efficient

For production with higher accuracy requirements, I'd upgrade to `all-mpnet-base-v2` (768-dim)
or use a domain-specific legal embedding model like `legal-bert`.

---

### Q8: What does your text chunker do differently for legal documents?

**Answer:**
I built a section-aware chunker rather than blindly splitting on character count. Legal documents
have natural structure — Articles, Sections, numbered clauses. My chunker:

1. Uses regex to detect section headings (ALL-CAPS lines, "ARTICLE N", "SECTION N", numbered like "1.1")
2. Splits the document into sections first, then chunks within each section
3. Stores the section name in the chunk metadata

This means a chunk about "TERMINATION" in the employment contract carries that label, so the
retriever knows which part of the document it came from, and the answer can cite it precisely.

---

### Q9: How does the metadata filtering in Endee work?

**Answer:**
When I upsert vectors into Endee, each vector has a metadata payload — a JSON object with
fields like `doc_type`, `filename`, `section`. Endee stores this alongside the vector.

When querying, I can pass a filter like `{"doc_type": {"$eq": "contract"}}`. Endee applies
this filter server-side before or during the ANN search, so only matching vectors are
considered. This is much more efficient than fetching all results and filtering in Python.

In the UI, the user can select "Filter by document type" to restrict answers to only contracts,
or only policies, etc.

---

### Q10: How would you scale this to millions of documents?

**Answer:**
Endee is designed for this — it supports up to 1 billion vectors on a single node.
For millions of documents at ~10 chunks each, that's ~10 million vectors, well within range.

Beyond that:
- Use a larger embedding model with better recall (e.g., `e5-large`)
- Implement sparse+dense hybrid search (BM25 + vector search) for keyword-heavy legal queries
- Add a re-ranker (cross-encoder) to re-score the top-K results more precisely
- Use Endee's metadata filtering to shard by document type or organization
- Add a caching layer for frequent queries

---

### Q11: What would you improve if you had more time?

**Answer:**
Several things:

1. **Better generation** — Integrate an actual LLM (Llama 3, GPT-4) for fluent answers
   instead of the template-based generator I used for the demo

2. **Hybrid search** — Combine vector search with BM25 keyword search; legal queries often
   have exact terms ("Section 5.1", "Force Majeure") that semantic search misses

3. **Document hierarchy** — Build a knowledge graph of clause relationships for multi-hop
   reasoning ("does clause 3.1 contradict clause 7.2?")

4. **Evaluation** — Implement RAGAS (Retrieval-Augmented Generation Assessment) metrics
   to systematically measure retrieval accuracy and answer faithfulness

5. **Auth and multi-tenancy** — Use Endee's auth token + namespace support for
   organization-level document isolation

---

## 💡 Technical Terms to Know

| Term | Simple Explanation |
|------|-------------------|
| **RAG** | Retrieval + Generation: find relevant docs, then generate an answer |
| **Vector embedding** | A list of numbers that captures the meaning of a sentence |
| **HNSW** | A fast graph-based algorithm for finding similar vectors |
| **Cosine similarity** | Measures the angle between two vectors (1 = identical meaning) |
| **Chunk** | A small piece of a document (800 chars), stored as one vector |
| **Top-K** | Return the K most similar results from the vector search |
| **ANN** | Approximate Nearest Neighbour — fast but not 100% exact |
| **Metadata filtering** | Filter vectors by their associated JSON data before searching |
| **Dimension** | The length of the embedding vector (384 for our model) |

---

## 🔴 Things NOT to Say in the Interview

- "I don't know how the embeddings work" — You do. They convert text to numbers capturing meaning.
- "I just used libraries" — You designed the architecture, chunking strategy, and integration.
- "Endee is just like any other database" — Endee is a *vector* database optimised for ANN search, very different from SQL/NoSQL.
- "The code just runs" — You can explain every file. See the project structure section.

---

Good luck! You have a solid project — just explain it with confidence.
