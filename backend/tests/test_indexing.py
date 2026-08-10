"""Unit tests for the vectorization and indexing pipeline.

Tests cover:
- Qdrant client configuration and collection setup
- OpenSearch client configuration and index setup
- Embedding service (dense + sparse generation)
- Indexing service (dual-write with rollback)
- Cascade delete logic
- Pipeline status updates
"""

import math
from unittest.mock import MagicMock, patch
import pytest

from app.services.embedding_service import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingError,
    DENSE_VECTOR_DIM,
)
from app.services.indexing_service import (
    ChunkPayload,
    IndexingError,
    IndexingService,
    update_pipeline_progress,
)


# ─── Embedding Service Tests ──────────────────────────────────────────


class TestEmbeddingServiceSparse:
    """Tests for sparse embedding generation (TF-IDF based)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EmbeddingService()

    def test_sparse_embedding_basic_english(self):
        """Sparse embeddings should produce non-empty indices and values for English text."""
        texts = ["The quick brown fox jumps over the lazy dog"]
        results = self.service._generate_sparse_embeddings(texts)

        assert len(results) == 1
        assert len(results[0]["indices"]) > 0
        assert len(results[0]["values"]) > 0
        assert len(results[0]["indices"]) == len(results[0]["values"])

    def test_sparse_embedding_basic_chinese(self):
        """Sparse embeddings should handle Chinese text with character and bigram tokens."""
        texts = ["企业知识库系统支持多格式文档导入"]
        results = self.service._generate_sparse_embeddings(texts)

        assert len(results) == 1
        assert len(results[0]["indices"]) > 0
        assert len(results[0]["values"]) > 0

    def test_sparse_embedding_empty_text(self):
        """Sparse embeddings for empty text should return empty vectors."""
        texts = [""]
        results = self.service._generate_sparse_embeddings(texts)

        assert len(results) == 1
        assert results[0]["indices"] == []
        assert results[0]["values"] == []

    def test_sparse_embedding_multiple_texts(self):
        """Sparse embeddings should handle multiple texts in a batch."""
        texts = [
            "Document processing pipeline",
            "Vector database indexing",
            "Full text search with OpenSearch",
        ]
        results = self.service._generate_sparse_embeddings(texts)

        assert len(results) == 3
        for result in results:
            assert len(result["indices"]) > 0
            assert len(result["values"]) > 0

    def test_sparse_embedding_indices_sorted(self):
        """Sparse embedding indices should be sorted in ascending order."""
        texts = ["The quick brown fox jumps over the lazy dog multiple times"]
        results = self.service._generate_sparse_embeddings(texts)

        indices = results[0]["indices"]
        assert indices == sorted(indices)

    def test_sparse_embedding_values_positive(self):
        """All sparse embedding values should be positive (TF-IDF weights)."""
        texts = ["Testing positive values in sparse embeddings"]
        results = self.service._generate_sparse_embeddings(texts)

        for value in results[0]["values"]:
            assert value > 0

    def test_sparse_embedding_no_duplicate_indices(self):
        """Sparse embedding indices should not contain duplicates."""
        texts = ["repeated word word word word word"]
        results = self.service._generate_sparse_embeddings(texts)

        indices = results[0]["indices"]
        assert len(indices) == len(set(indices))


class TestEmbeddingServiceTokenize:
    """Tests for the tokenization method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EmbeddingService()

    def test_tokenize_english(self):
        """English text should be tokenized into words."""
        tokens = self.service._tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_chinese(self):
        """Chinese text should produce character and bigram tokens."""
        tokens = self.service._tokenize("知识库")
        assert "知" in tokens
        assert "识" in tokens
        assert "库" in tokens
        assert "知识" in tokens
        assert "识库" in tokens

    def test_tokenize_mixed(self):
        """Mixed Chinese and English text should produce both types of tokens."""
        tokens = self.service._tokenize("知识库 system")
        assert "知识" in tokens
        assert "system" in tokens

    def test_tokenize_single_char_english_filtered(self):
        """Single character English tokens should be filtered out."""
        tokens = self.service._tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens

    def test_tokenize_numbers(self):
        """Numeric tokens should be included."""
        tokens = self.service._tokenize("version 123 test")
        assert "123" in tokens
        assert "version" in tokens


class TestEmbeddingServiceDense:
    """Tests for dense embedding generation via LiteLLM."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = EmbeddingService()

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_dense_embedding_calls_litellm(self, mock_aembedding):
        """Dense embedding should call litellm.aembedding with correct params."""
        # Mock response
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1] * DENSE_VECTOR_DIM},
        ]
        mock_aembedding.return_value = mock_response

        vectors = await self.service._generate_dense_embeddings(["test text"])

        assert len(vectors) == 1
        assert len(vectors[0]) == DENSE_VECTOR_DIM
        mock_aembedding.assert_called_once()

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_dense_embedding_pads_short_vectors(self, mock_aembedding):
        """Dense embedding should pad vectors shorter than 1024 dims."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.5] * 512},  # Only 512 dims
        ]
        mock_aembedding.return_value = mock_response

        vectors = await self.service._generate_dense_embeddings(["test"])

        assert len(vectors[0]) == DENSE_VECTOR_DIM
        assert vectors[0][511] == 0.5
        assert vectors[0][512] == 0.0  # Padded

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_dense_embedding_truncates_long_vectors(self, mock_aembedding):
        """Dense embedding should truncate vectors longer than 1024 dims."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.3] * 2048},  # 2048 dims
        ]
        mock_aembedding.return_value = mock_response

        vectors = await self.service._generate_dense_embeddings(["test"])

        assert len(vectors[0]) == DENSE_VECTOR_DIM

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_dense_embedding_batching(self, mock_aembedding):
        """Dense embedding should batch texts according to batch_size."""
        self.service.batch_size = 2

        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1] * DENSE_VECTOR_DIM},
            {"embedding": [0.2] * DENSE_VECTOR_DIM},
        ]
        mock_aembedding.return_value = mock_response

        texts = ["text1", "text2", "text3", "text4"]
        vectors = await self.service._generate_dense_embeddings(texts)

        # Should be called twice (2 batches of 2)
        assert mock_aembedding.call_count == 2
        assert len(vectors) == 4

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_embed_chunks_combines_dense_and_sparse(self, mock_aembedding):
        """embed_chunks should return both dense and sparse vectors."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1] * DENSE_VECTOR_DIM},
        ]
        mock_aembedding.return_value = mock_response

        chunks = [{"id": "chunk-1", "text": "Hello world testing"}]
        results = await self.service.embed_chunks(chunks)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
        assert len(results[0].dense_vector) == DENSE_VECTOR_DIM
        assert len(results[0].sparse_indices) > 0
        assert len(results[0].sparse_values) > 0

    @pytest.mark.asyncio
    @patch("litellm.aembedding")
    async def test_embed_chunks_empty_list(self, mock_aembedding):
        """embed_chunks with empty list should return empty results."""
        results = await self.service.embed_chunks([])
        assert results == []
        mock_aembedding.assert_not_called()


# ─── Indexing Service Tests ────────────────────────────────────────────


class TestIndexingService:
    """Tests for the dual-write indexing service."""

    def _make_payload(self, chunk_id=None, document_id=None):
        """Create a test ChunkPayload.

        Defaults to fresh UUIDs because Qdrant requires UUID-string or
        unsigned-int point IDs and ``IndexingService._upsert_qdrant`` validates
        ``chunk_id`` accordingly.
        """
        import uuid as _uuid

        return ChunkPayload(
            chunk_id=chunk_id or str(_uuid.uuid4()),
            document_id=document_id or str(_uuid.uuid4()),
            space_id="space-1",
            chunk_index=0,
            title_chain="H1 > H2",
            source_file="test.pdf",
            page_number=1,
            content="Test chunk content",
            parent_chunk_id=None,
            depth=1,
            token_count=10,
            allowed_user_ids=["user-1"],
            access_level="read",
        )

    def _make_embedding(self, chunk_id="chunk-1"):
        """Create a test EmbeddingResult."""
        return EmbeddingResult(
            chunk_id=chunk_id,
            dense_vector=[0.1] * DENSE_VECTOR_DIM,
            sparse_indices=[1, 5, 10, 100],
            sparse_values=[0.5, 0.3, 0.8, 0.2],
        )

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_index_chunks_success(self, mock_qdrant_fn, mock_os_fn):
        """Dual-write should succeed when both backends succeed."""
        mock_qdrant = MagicMock()
        mock_qdrant_fn.return_value = mock_qdrant

        mock_os = MagicMock()
        mock_os_fn.return_value = mock_os

        # Mock OpenSearch bulk (imported inside the method from opensearchpy.helpers)
        with patch("opensearchpy.helpers.bulk", return_value=(1, [])):
            service = IndexingService()
            service._qdrant = mock_qdrant
            service._opensearch = mock_os

            payloads = [self._make_payload()]
            embeddings = [self._make_embedding()]

            result = service.index_chunks(payloads, embeddings)

            assert result["qdrant_count"] == 1
            assert result["opensearch_count"] == 1
            mock_qdrant.upsert.assert_called()

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_index_chunks_qdrant_failure(self, mock_qdrant_fn, mock_os_fn):
        """Should raise IndexingError when Qdrant write fails."""
        mock_qdrant = MagicMock()
        mock_qdrant.upsert.side_effect = Exception("Qdrant connection error")
        mock_qdrant_fn.return_value = mock_qdrant

        mock_os = MagicMock()
        mock_os_fn.return_value = mock_os

        service = IndexingService()
        service._qdrant = mock_qdrant
        service._opensearch = mock_os

        payloads = [self._make_payload()]
        embeddings = [self._make_embedding()]

        with pytest.raises(IndexingError, match="Qdrant write failed"):
            service.index_chunks(payloads, embeddings)

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_index_chunks_opensearch_failure_triggers_rollback(
        self, mock_qdrant_fn, mock_os_fn
    ):
        """Should rollback Qdrant when OpenSearch write fails."""
        mock_qdrant = MagicMock()
        mock_qdrant_fn.return_value = mock_qdrant

        mock_os = MagicMock()
        mock_os_fn.return_value = mock_os

        with patch(
            "opensearchpy.helpers.bulk",
            side_effect=Exception("OpenSearch error"),
        ):
            service = IndexingService()
            service._qdrant = mock_qdrant
            service._opensearch = mock_os

            payloads = [self._make_payload()]
            embeddings = [self._make_embedding()]

            with pytest.raises(IndexingError, match="OpenSearch write failed"):
                service.index_chunks(payloads, embeddings)

            # Verify Qdrant rollback was attempted
            mock_qdrant.delete.assert_called()

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_index_chunks_empty_input(self, mock_qdrant_fn, mock_os_fn):
        """Should return zero counts for empty input."""
        service = IndexingService()
        result = service.index_chunks([], [])

        assert result["qdrant_count"] == 0
        assert result["opensearch_count"] == 0

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_index_chunks_mismatched_lengths(self, mock_qdrant_fn, mock_os_fn):
        """Should raise IndexingError when payload and embedding counts differ."""
        service = IndexingService()
        payloads = [self._make_payload(), self._make_payload(chunk_id="chunk-2")]
        embeddings = [self._make_embedding()]

        with pytest.raises(IndexingError, match="Payload count"):
            service.index_chunks(payloads, embeddings)

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_delete_document_chunks(self, mock_qdrant_fn, mock_os_fn):
        """Should delete chunks from both Qdrant and OpenSearch."""
        mock_qdrant = MagicMock()
        mock_qdrant_fn.return_value = mock_qdrant

        mock_os = MagicMock()
        mock_os.delete_by_query.return_value = {"deleted": 5}
        mock_os_fn.return_value = mock_os

        service = IndexingService()
        service._qdrant = mock_qdrant
        service._opensearch = mock_os

        result = service.delete_document_chunks("doc-123")

        assert result["opensearch_deleted"] == 5
        mock_qdrant.delete.assert_called_once()
        mock_os.delete_by_query.assert_called_once()

    @patch("app.services.indexing_service.get_opensearch_client")
    @patch("app.services.indexing_service.get_qdrant_client")
    def test_delete_document_chunks_qdrant_failure(self, mock_qdrant_fn, mock_os_fn):
        """Should raise IndexingError when Qdrant deletion fails."""
        mock_qdrant = MagicMock()
        mock_qdrant.delete.side_effect = Exception("Qdrant error")
        mock_qdrant_fn.return_value = mock_qdrant

        service = IndexingService()
        service._qdrant = mock_qdrant

        with pytest.raises(IndexingError, match="Qdrant deletion failed"):
            service.delete_document_chunks("doc-123")


# ─── Qdrant Client Tests ──────────────────────────────────────────────


class TestQdrantSetup:
    """Tests for Qdrant client and collection setup."""

    @patch("app.core.qdrant.QdrantClient")
    def test_get_qdrant_client_creates_client(self, mock_client_class):
        """Should create a QdrantClient with correct settings."""
        import app.core.qdrant as qdrant_module

        # Reset the global client
        qdrant_module._qdrant_client = None

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = qdrant_module.get_qdrant_client()

        assert client == mock_client
        mock_client_class.assert_called_once()

        # Cleanup
        qdrant_module._qdrant_client = None

    @patch("app.core.qdrant.get_qdrant_client")
    def test_ensure_collection_exists_creates_when_missing(self, mock_get_client):
        """Should create collection when it doesn't exist."""
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []  # No existing collections
        mock_client.get_collections.return_value = mock_collections
        mock_get_client.return_value = mock_client

        from app.core.qdrant import ensure_collection_exists

        ensure_collection_exists()

        mock_client.create_collection.assert_called_once()

    @patch("app.core.qdrant.get_qdrant_client")
    def test_ensure_collection_exists_skips_when_exists(self, mock_get_client):
        """Should not create collection when it already exists."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "document_chunks"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_get_client.return_value = mock_client

        from app.core.qdrant import ensure_collection_exists

        ensure_collection_exists()

        mock_client.create_collection.assert_not_called()


# ─── OpenSearch Client Tests ───────────────────────────────────────────


class TestOpenSearchSetup:
    """Tests for OpenSearch client and index setup."""

    @patch("app.core.opensearch.OpenSearch")
    def test_get_opensearch_client_creates_client(self, mock_os_class):
        """Should create an OpenSearch client with correct settings."""
        import app.core.opensearch as os_module

        # Reset the global client
        os_module._opensearch_client = None

        mock_client = MagicMock()
        mock_os_class.return_value = mock_client

        client = os_module.get_opensearch_client()

        assert client == mock_client
        mock_os_class.assert_called_once()

        # Cleanup
        os_module._opensearch_client = None

    @patch("app.core.opensearch.get_opensearch_client")
    def test_ensure_index_exists_creates_when_missing(self, mock_get_client):
        """Should create index when it doesn't exist."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_get_client.return_value = mock_client

        from app.core.opensearch import ensure_index_exists

        ensure_index_exists()

        mock_client.indices.create.assert_called_once()

    @patch("app.core.opensearch.get_opensearch_client")
    def test_ensure_index_exists_skips_when_exists(self, mock_get_client):
        """Should not create index when it already exists."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_get_client.return_value = mock_client

        from app.core.opensearch import ensure_index_exists

        ensure_index_exists()

        mock_client.indices.create.assert_not_called()

    @patch("app.core.opensearch.get_opensearch_client")
    def test_ensure_index_falls_back_without_ik(self, mock_get_client):
        """Should fall back to standard analyzer when IK is not available."""
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        # First call fails (IK not available), second succeeds (fallback)
        mock_client.indices.create.side_effect = [
            Exception("analyzer [ik_max_word] not found"),
            None,
        ]
        mock_get_client.return_value = mock_client

        from app.core.opensearch import ensure_index_exists

        ensure_index_exists()

        assert mock_client.indices.create.call_count == 2


# ─── Pipeline Progress Tests ──────────────────────────────────────────


class TestPipelineProgress:
    """Tests for pipeline status update functions."""

    @patch("redis.Redis")
    def test_update_pipeline_progress(self, mock_redis_class):
        """Should update Redis hash with stage and progress."""
        mock_redis = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis

        update_pipeline_progress("doc-1", "embedding", 50)

        mock_redis_class.from_url.assert_called_once()
        mock_redis.hset.assert_called_once()
        # Verify the key was passed correctly (positional or keyword)
        call_args = mock_redis.hset.call_args
        key_arg = call_args[0][0] if call_args[0] else call_args[1].get("name", "")
        assert key_arg == "doc:status:doc-1"


# ─── ChunkPayload Tests ───────────────────────────────────────────────


class TestChunkPayload:
    """Tests for ChunkPayload data class."""

    def test_default_values(self):
        """ChunkPayload should have sensible defaults."""
        payload = ChunkPayload()
        assert payload.chunk_id == ""
        assert payload.document_id == ""
        assert payload.space_id == ""
        assert payload.chunk_index == 0
        assert payload.depth == 1
        assert payload.allowed_user_ids == []
        assert payload.access_level == "read"

    def test_custom_values(self):
        """ChunkPayload should accept custom values."""
        payload = ChunkPayload(
            chunk_id="c-1",
            document_id="d-1",
            space_id="s-1",
            chunk_index=5,
            title_chain="Title > Sub",
            source_file="doc.pdf",
            page_number=3,
            content="Hello world",
            parent_chunk_id="c-0",
            depth=2,
            token_count=100,
            allowed_user_ids=["u-1", "u-2"],
            access_level="write",
        )
        assert payload.chunk_id == "c-1"
        assert payload.document_id == "d-1"
        assert payload.content == "Hello world"
        assert payload.allowed_user_ids == ["u-1", "u-2"]
        assert payload.access_level == "write"
