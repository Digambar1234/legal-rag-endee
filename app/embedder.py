"""
Embedding Engine
Converts text chunks to dense vector embeddings using
sentence-transformers (free, local, no API key required).

Model choice: 'all-MiniLM-L6-v2'
  - Dimension: 384
  - Speed: very fast on CPU
  - Quality: excellent for semantic similarity tasks
  - Size: ~90 MB

For higher quality (at the cost of speed), swap to:
  - 'all-mpnet-base-v2'        (768-dim)
  - 'multi-qa-mpnet-base-cos-v1' (768-dim, tuned for Q&A)
"""

import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)

# Default model & dimension — change here to swap models project-wide
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class EmbeddingEngine:
    """
    Wraps sentence-transformers to produce normalised embeddings.

    Usage:
        engine = EmbeddingEngine()
        vectors = engine.embed(["Hello world", "Legal document text…"])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy-loaded

    @property
    def dimension(self) -> int:
        """Return embedding dimension without loading model."""
        model_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "multi-qa-mpnet-base-cos-v1": 768,
            "all-MiniLM-L12-v2": 384,
        }
        return model_dims.get(self.model_name, 384)

    def _load(self):
        if self._model is None:
            logger.info(f"[Embedder] Loading model '{self.model_name}'…")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("[Embedder] Model ready.")

    def embed(self, texts: Union[str, List[str]], batch_size: int = 64) -> List[List[float]]:
        """
        Encode texts into normalised float32 embedding vectors.

        Args:
            texts:      A single string or list of strings.
            batch_size: Number of texts per encoding batch.

        Returns:
            List of embedding vectors (list[float]).
        """
        self._load()

        if isinstance(texts, str):
            texts = [texts]

        # sentence-transformers handles batching internally
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # cosine similarity friendly
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Convenience wrapper for embedding a single query string.
        Prefixes with 'query:' for asymmetric retrieval models.
        """
        return self.embed([query])[0]

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors (for unit-testing).
        Both vectors should already be normalised (embed() normalises by default).
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np))
