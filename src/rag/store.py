"""
Vector store wrapper around Qdrant Cloud.
Handles collection lifecycle, embedding, upsert, and search.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.models import Chunk, SearchResult
from utils.config import settings


class VectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self._openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._collection = settings.QDRANT_COLLECTION
        self._dims = settings.EMBEDDING_DIMS
        self._embed_model = settings.EMBEDDING_MODEL
        self._batch_size = settings.RAG_EMBED_BATCH_SIZE

    # Collection lifecycle
    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        try:
            info = self._client.get_collection(self._collection)
            vectors = info.config.params.vectors
            # vectors is VectorParams for unnamed collections, dict for named ones
            existing_dims = (
                next(iter(vectors.values())).size
                if isinstance(vectors, dict)
                else vectors.size
            )
            if existing_dims != self._dims:
                raise RuntimeError(
                    f"Collection '{self._collection}' exists with {existing_dims} dims "
                    f"but config expects {self._dims}. Run reset() to rebuild."
                )
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dims,
                    distance=Distance.COSINE,
                ),
            )

    def reset(self) -> None:
        """Delete and recreate the collection (clean re-ingestion)."""
        self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._dims,
                distance=Distance.COSINE,
            ),
        )

    def collection_info(self) -> dict[str, Any]:
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()))
            dims, distance = first.size, first.distance
        else:
            dims, distance = vectors.size, vectors.distance
        return {
            "name": self._collection,
            "points_count": info.points_count,
            "dims": dims,
            "distance": distance,
        }

    def upsert(self, chunks: list[Chunk]) -> None:
        """Embed chunks in batches and upsert into the collection."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self._embed_batch(texts)

        points = [
            PointStruct(
                id=c.id,
                vector=vec,
                payload={"text": c.text, **c.metadata},
            )
            for c, vec in zip(chunks, vectors)
        ]
        for i in range(0, len(points), self._batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + self._batch_size],
            )

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed query and return top-k results. Optionally filter by metadata."""
        query_vec = self._embed_batch([query])[0]
        qdrant_filter = _build_filter(filters) if filters else None
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={key: val for key, val in hit.payload.items() if key != "text"},
            )
            for hit in response.points
        ]

    def fetch_by_ids(self, ids: list[str]) -> dict[str, SearchResult]:
        """Fetch points by exact ID. Returns {point_id: SearchResult} for existing points."""
        records = self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return {
            str(r.id): SearchResult(
                text=r.payload.get("text", ""),
                score=1.0,
                metadata={k: v for k, v in r.payload.items() if k != "text"},
            )
            for r in records
        }

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to minimise API round-trips."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._openai.embeddings.create(
                model=self._embed_model,
                input=batch,
            )
            # Response items are ordered to match input order.
            vectors.extend(item.embedding for item in response.data)

        actual_dims = len(vectors[0]) if vectors else 0
        if actual_dims and actual_dims != self._dims:
            raise RuntimeError(
                f"Embedding model returned {actual_dims} dims but config expects "
                f"{self._dims}. Update EMBEDDING_DIMS in config."
            )
        return vectors


def _build_filter(filters: dict[str, Any]) -> Filter:
    """Convert a flat {field: value} dict to a Qdrant AND filter."""
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filters.items()
    ]
    return Filter(must=conditions)
