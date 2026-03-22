"""
Test suite for the Legal RAG Assistant.

Run with:  pytest tests/ -v

Tests are organised into:
  - unit tests (no Endee server required, using mocks)
  - integration tests (require live Endee on localhost:8080)
"""

import pytest
import json
from unittest.mock import MagicMock, patch
from typing import List


# ================================================================== #
#  Unit Tests — Document Processing                                   #
# ================================================================== #

class TestDocumentProcessor:

    def test_chunk_splits_on_paragraphs(self, tmp_path):
        from app.document_processor import DocumentLoader, TextChunker, Document

        text = "\n\n".join([f"Paragraph {i}. " + "Word " * 50 for i in range(10)])
        doc = Document(
            doc_id="test-123",
            filename="test.txt",
            doc_type="legal_document",
            full_text=text,
            pages=[text],
        )

        chunker = TextChunker(max_chars=200, overlap_chars=40)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 300   # allow slight overflow at word boundaries
            assert c.doc_id == "test-123"
            assert c.filename == "test.txt"

    def test_chunk_ids_are_unique(self):
        from app.document_processor import TextChunker, Document

        text = "\n\n".join(["Clause. " * 20 for _ in range(5)])
        doc = Document("id1", "doc.txt", "contract", text, [text])
        chunker = TextChunker(max_chars=100, overlap_chars=20)
        chunks = chunker.chunk(doc)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_serialisation(self):
        from app.document_processor import DocumentChunk

        chunk = DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            filename="nda.txt",
            doc_type="contract",
            section="ARTICLE 1",
            text="Sample clause text.",
            char_start=0,
            char_end=18,
            page_number=1,
        )
        meta = chunk.to_metadata()
        assert meta["chunk_id"] == "c1"
        assert meta["doc_type"] == "contract"
        assert meta["section"] == "ARTICLE 1"
        assert isinstance(meta["text"], str)

    def test_infer_doc_type(self):
        from app.document_processor import DocumentLoader
        loader = DocumentLoader()
        assert loader._infer_doc_type("nda_agreement") == "contract"
        assert loader._infer_doc_type("privacy_policy") == "policy"
        assert loader._infer_doc_type("court_judgment_2024") == "court_order"
        assert loader._infer_doc_type("random_file") == "legal_document"

    def test_load_txt_file(self, tmp_path):
        from app.document_processor import DocumentLoader
        f = tmp_path / "agreement.txt"
        f.write_text("This is a test legal document. It has some content.")
        loader = DocumentLoader()
        doc = loader.load(str(f))
        assert "test legal document" in doc.full_text
        assert doc.filename == "agreement.txt"

    def test_unsupported_file_type_raises(self, tmp_path):
        from app.document_processor import DocumentLoader
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"fake")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(str(f))


# ================================================================== #
#  Unit Tests — Embedding Engine                                      #
# ================================================================== #

class TestEmbeddingEngine:

    def test_dimension_property(self):
        from app.embedder import EmbeddingEngine
        e = EmbeddingEngine("all-MiniLM-L6-v2")
        assert e.dimension == 384

    def test_cosine_similarity_identical_vectors(self):
        from app.embedder import EmbeddingEngine
        e = EmbeddingEngine()
        v = [0.5, 0.5, 0.5, 0.5]
        # normalise manually
        import math
        norm = math.sqrt(sum(x**2 for x in v))
        v_norm = [x / norm for x in v]
        sim = e.cosine_similarity(v_norm, v_norm)
        assert abs(sim - 1.0) < 1e-5


# ================================================================== #
#  Unit Tests — Endee Client (mocked)                                 #
# ================================================================== #

class TestEndeeClientMocked:

    @pytest.fixture
    def client(self):
        from app.endee_client import EndeeClient
        return EndeeClient(base_url="http://localhost:8080")

    def test_health_check_returns_false_on_connection_error(self, client):
        import requests
        with patch.object(client.session, "get", side_effect=requests.exceptions.ConnectionError):
            assert client.health_check() is False

    def test_url_construction(self, client):
        assert client._url("/api/v1/index/list") == "http://localhost:8080/api/v1/index/list"

    def test_create_index_sends_correct_payload(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "created"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            client.create_index("test_index", dimension=384, metric="cosine")
            call_kwargs = mock_post.call_args
            payload = call_kwargs[1]["json"]
            assert payload["name"] == "test_index"
            assert payload["dimension"] == 384
            assert payload["metric"] == "cosine"

    def test_upsert_vectors_sends_correct_payload(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"upserted": 2}

        vectors = [
            {"id": "v1", "values": [0.1, 0.2], "metadata": {"text": "hello"}},
            {"id": "v2", "values": [0.3, 0.4], "metadata": {"text": "world"}},
        ]

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            client.upsert_vectors("legal_docs", vectors)
            payload = mock_post.call_args[1]["json"]
            assert len(payload["vectors"]) == 2
            assert payload["vectors"][0]["id"] == "v1"

    def test_query_sends_vector_and_top_k(self, client):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"matches": [
            {"id": "c1", "score": 0.92, "metadata": {"text": "relevant clause"}}
        ]}

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            results = client.query("legal_docs", vector=[0.1] * 384, top_k=3)
            payload = mock_post.call_args[1]["json"]
            assert payload["top_k"] == 3
            assert len(payload["vector"]) == 384
            assert results[0]["score"] == 0.92


# ================================================================== #
#  Unit Tests — RAG Pipeline (mocked)                                 #
# ================================================================== #

class TestRAGPipelineMocked:

    @pytest.fixture
    def pipeline(self):
        from app.rag_pipeline import LegalRAGPipeline
        p = LegalRAGPipeline.__new__(LegalRAGPipeline)
        return p

    def test_answer_generator_simple_with_no_context(self):
        from app.rag_pipeline import AnswerGenerator
        gen = AnswerGenerator(backend="simple")
        answer = gen.generate("What is the notice period?", contexts=[])
        assert "could not find" in answer.lower() or "no" in answer.lower()

    def test_answer_generator_simple_with_context(self):
        from app.rag_pipeline import AnswerGenerator, RetrievedContext
        gen = AnswerGenerator(backend="simple")
        ctx = RetrievedContext(
            chunk_id="c1",
            score=0.88,
            text="The notice period shall be 90 days.",
            filename="employment_contract.txt",
            doc_type="contract",
            section="SECTION 5",
            page_number=2,
        )
        answer = gen.generate("What is the notice period?", [ctx])
        assert "employment_contract.txt" in answer
        assert "90 days" in answer

    def test_retriever_filters_low_scores(self):
        from app.rag_pipeline import EndeeRetriever

        mock_endee = MagicMock()
        mock_endee.query.return_value = [
            {"id": "c1", "score": 0.85, "metadata": {"text": "hi", "filename": "a.txt", "doc_type": "contract", "section": "S1", "page_number": 1}},
            {"id": "c2", "score": 0.10, "metadata": {"text": "low", "filename": "b.txt", "doc_type": "policy", "section": "S2", "page_number": 1}},
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.0] * 384

        retriever = EndeeRetriever(mock_endee, mock_embedder, score_threshold=0.30)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].score == 0.85


# ================================================================== #
#  Integration Tests — require live Endee on localhost:8080           #
# ================================================================== #

@pytest.mark.integration
class TestIntegration:
    """
    These tests require a running Endee instance.
    Run with:  pytest tests/ -v -m integration
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_endee(self):
        from app.endee_client import EndeeClient
        client = EndeeClient()
        if not client.health_check():
            pytest.skip("Endee server not available — skipping integration tests")

    def test_full_index_and_query(self, tmp_path):
        from app.rag_pipeline import LegalRAGPipeline

        # Write a tiny test document
        doc = tmp_path / "test_contract.txt"
        doc.write_text(
            "TERMINATION CLAUSE\n\n"
            "Either party may terminate this agreement with 60 days written notice.\n\n"
            "PAYMENT TERMS\n\n"
            "Payment shall be made within 30 days of invoice."
        )

        pipeline = LegalRAGPipeline(index_name="test_rag_index")
        result = pipeline.index_documents(str(tmp_path), recreate=True)

        assert result["documents_indexed"] == 1
        assert result["chunks_indexed"] >= 1

        response = pipeline.ask("How many days notice is required for termination?")

        assert response.answer
        assert len(response.contexts) >= 1
        assert response.latency_ms > 0

        # Clean up
        pipeline.endee.delete_index("test_rag_index")

    def test_semantic_search_returns_ranked_results(self, tmp_path):
        from app.rag_pipeline import LegalRAGPipeline

        doc = tmp_path / "policy.txt"
        doc.write_text(
            "DATA RETENTION POLICY\n\n"
            "User data shall be retained for 7 years as required by law.\n\n"
            "SECURITY POLICY\n\n"
            "All data is encrypted using AES-256."
        )

        pipeline = LegalRAGPipeline(index_name="test_search_index")
        pipeline.index_documents(str(tmp_path), recreate=True)

        results = pipeline.semantic_search("data encryption security", top_k=3)
        assert len(results) >= 1
        # Results should be ranked by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        pipeline.endee.delete_index("test_search_index")
