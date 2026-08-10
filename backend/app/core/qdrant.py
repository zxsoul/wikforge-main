"""Qdrant vector database client and collection management.

Provides:
- Qdrant client initialization with connection settings
- Collection creation with Dense (1024-dim) + Sparse vector configuration
- Collection existence check and setup utilities
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Collection name for document chunks
COLLECTION_NAME = "document_chunks"

# Dense vector dimensions (matches LiteLLM embedding output)
DENSE_VECTOR_DIM = 1024

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client instance.

    Returns:
        QdrantClient connected to the configured Qdrant server
    """
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        kwargs: dict = {
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
            "timeout": 30,
        }
        if settings.QDRANT_API_KEY:
            kwargs["api_key"] = settings.QDRANT_API_KEY
        _qdrant_client = QdrantClient(**kwargs)
    return _qdrant_client


def close_qdrant_client() -> None:
    """Close the Qdrant client connection."""
    global _qdrant_client
    if _qdrant_client is not None:
        _qdrant_client.close()
        _qdrant_client = None


def reset_qdrant_client() -> None:
    """Drop the cached Qdrant client without closing it.

    Primarily for tests that patch ``QdrantClient`` and need a fresh singleton.
    """
    global _qdrant_client
    _qdrant_client = None


def ensure_collection_exists() -> None:
    """Ensure the document_chunks collection exists with proper configuration.

    Creates the collection if it doesn't exist, with:
    - Dense vector: 1024 dimensions, Cosine distance
    - Sparse vector: SPLADE-style sparse embeddings

    This is idempotent - safe to call multiple times.
    """
    client = get_qdrant_client()

    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME in collection_names:
            logger.debug(f"Collection '{COLLECTION_NAME}' already exists")
            return

        logger.info(f"Creating collection '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created successfully")

    except UnexpectedResponse as e:
        logger.error(f"Failed to ensure collection exists: {e}")
        raise


def delete_collection() -> None:
    """Delete the document_chunks collection. Use with caution.

    Primarily for testing and reset scenarios.
    """
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' deleted")
    except UnexpectedResponse:
        logger.debug(f"Collection '{COLLECTION_NAME}' does not exist, nothing to delete")
