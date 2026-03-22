"""
Endee Vector Database Client
Wraps the Endee HTTP API for easy integration into the RAG pipeline.
Endee runs as a local service on port 8080.
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EndeeClient:
    """
    Client for interacting with the Endee vector database.
    Endee exposes a REST API on http://localhost:8080 by default.
    """

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if auth_token:
            self.session.headers.update({"Authorization": auth_token})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health_check(self) -> bool:
        """Check if Endee server is running."""
        try:
            resp = self.session.get(self._url("/api/v1/index/list"), timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    # ------------------------------------------------------------------ #
    #  Index management                                                    #
    # ------------------------------------------------------------------ #

    def create_index(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new vector index in Endee.

        Args:
            name:       Index name (alphanumeric + underscores).
            dimension:  Vector dimension (must match your embedding model).
            metric:     Distance metric — 'cosine', 'dot', or 'l2'.
            description: Human-readable description stored as metadata.
        """
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "description": description,
        }
        resp = self.session.post(self._url("/api/v1/index/create"), json=payload)
        resp.raise_for_status()
        logger.info(f"[Endee] Created index '{name}' (dim={dimension}, metric={metric})")
        return resp.json()

    def delete_index(self, name: str) -> Dict[str, Any]:
        resp = self.session.delete(self._url(f"/api/v1/index/{name}"))
        resp.raise_for_status()
        return resp.json()

    def list_indexes(self) -> List[str]:
        resp = self.session.get(self._url("/api/v1/index/list"))
        resp.raise_for_status()
        return resp.json().get("indexes", [])

    def index_exists(self, name: str) -> bool:
        return name in self.list_indexes()

    # ------------------------------------------------------------------ #
    #  Vector operations                                                   #
    # ------------------------------------------------------------------ #

    def upsert_vectors(
        self,
        index_name: str,
        vectors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in an index.

        Each vector dict must have:
            id       (str)   – unique identifier
            values   (list)  – float list of length == index dimension
            metadata (dict)  – arbitrary JSON payload stored alongside vector
        """
        payload = {"vectors": vectors}
        resp = self.session.post(
            self._url(f"/api/v1/index/{index_name}/vectors/upsert"),
            json=payload,
        )
        resp.raise_for_status()
        logger.info(f"[Endee] Upserted {len(vectors)} vectors into '{index_name}'")
        return resp.json()

    def query(
        self,
        index_name: str,
        vector: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Nearest-neighbour search in Endee.

        Returns a ranked list of matches, each with:
            id, score, metadata (if include_metadata=True)
        """
        payload: Dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }
        if filter:
            payload["filter"] = filter

        resp = self.session.post(
            self._url(f"/api/v1/index/{index_name}/query"),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json().get("matches", [])

    def delete_vectors(self, index_name: str, ids: List[str]) -> Dict[str, Any]:
        payload = {"ids": ids}
        resp = self.session.delete(
            self._url(f"/api/v1/index/{index_name}/vectors"),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        resp = self.session.get(self._url(f"/api/v1/index/{index_name}/stats"))
        resp.raise_for_status()
        return resp.json()
