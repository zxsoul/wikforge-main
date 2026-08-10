"""Unit tests for Qdrant client and ``document_chunks`` collection setup.

任务 12.1：配置 Qdrant 客户端和 Collection 创建
（document_chunks，Dense 1024 维 + Sparse）。

覆盖点：
- ``get_qdrant_client`` 是单例：多次调用返回同一实例，``QdrantClient``
  构造函数仅被调用一次，且使用 ``QDRANT_HOST`` / ``QDRANT_PORT`` 配置。
- ``ensure_collection_exists`` 在 collection 缺失时创建，并配置：
  * Dense 向量名 ``dense``、维度 1024、Cosine 距离；
  * Sparse 向量名 ``sparse``。
- ``ensure_collection_exists`` 在 collection 已存在时为 no-op（幂等）。
- ``ensure_collection_exists`` 在底层抛 ``UnexpectedResponse`` 时向上抛出。
- ``reset_qdrant_client`` 释放单例，下次调用会重新构造客户端。
- ``delete_collection`` 在不存在时静默忽略。

Validates: Requirements 4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.core.qdrant as qdrant_module
from app.core.qdrant import (
    COLLECTION_NAME,
    DENSE_VECTOR_DIM,
    delete_collection,
    ensure_collection_exists,
    get_qdrant_client,
    reset_qdrant_client,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_singleton():
    """每个测试前后都清空模块级 Qdrant 客户端缓存。"""
    qdrant_module._qdrant_client = None
    yield
    qdrant_module._qdrant_client = None


# ─── Constants ─────────────────────────────────────────────────────────


class TestModuleConstants:
    """模块级常量必须与设计文档保持一致。"""

    def test_collection_name_is_document_chunks(self):
        assert COLLECTION_NAME == "document_chunks"

    def test_dense_vector_dim_is_1024(self):
        assert DENSE_VECTOR_DIM == 1024


# ─── Client singleton ──────────────────────────────────────────────────


class TestGetQdrantClient:
    """``get_qdrant_client`` 行为校验。"""

    @patch("app.core.qdrant.QdrantClient")
    def test_creates_client_with_host_and_port(self, mock_client_class):
        """客户端构造时使用 settings 中的 host/port。"""
        mock_client_class.return_value = MagicMock()

        client = get_qdrant_client()

        assert client is mock_client_class.return_value
        mock_client_class.assert_called_once()
        kwargs = mock_client_class.call_args.kwargs
        assert "host" in kwargs
        assert "port" in kwargs
        # timeout 必须设置，避免请求挂死
        assert kwargs.get("timeout") is not None

    @patch("app.core.qdrant.QdrantClient")
    def test_returns_same_instance_on_repeated_calls(self, mock_client_class):
        """单例：多次调用复用同一客户端，构造函数只跑一次。"""
        mock_client_class.return_value = MagicMock()

        first = get_qdrant_client()
        second = get_qdrant_client()
        third = get_qdrant_client()

        assert first is second is third
        assert mock_client_class.call_count == 1

    @patch("app.core.qdrant.get_settings")
    @patch("app.core.qdrant.QdrantClient")
    def test_passes_api_key_when_configured(self, mock_client_class, mock_get_settings):
        """配置了 API key 时透传到 QdrantClient。"""
        settings = MagicMock()
        settings.QDRANT_HOST = "qdrant.example"
        settings.QDRANT_PORT = 6333
        settings.QDRANT_API_KEY = "secret-token"
        mock_get_settings.return_value = settings

        get_qdrant_client()

        kwargs = mock_client_class.call_args.kwargs
        assert kwargs.get("api_key") == "secret-token"
        assert kwargs.get("host") == "qdrant.example"
        assert kwargs.get("port") == 6333

    @patch("app.core.qdrant.get_settings")
    @patch("app.core.qdrant.QdrantClient")
    def test_omits_api_key_when_blank(self, mock_client_class, mock_get_settings):
        """未配置 API key 时不应把空字符串传给 QdrantClient。"""
        settings = MagicMock()
        settings.QDRANT_HOST = "qdrant"
        settings.QDRANT_PORT = 6333
        settings.QDRANT_API_KEY = ""
        mock_get_settings.return_value = settings

        get_qdrant_client()

        kwargs = mock_client_class.call_args.kwargs
        assert "api_key" not in kwargs


# ─── Reset helper ──────────────────────────────────────────────────────


class TestResetQdrantClient:
    """``reset_qdrant_client`` 必须清空单例状态。"""

    @patch("app.core.qdrant.QdrantClient")
    def test_reset_forces_new_client_on_next_call(self, mock_client_class):
        first_inst = MagicMock(name="first")
        second_inst = MagicMock(name="second")
        mock_client_class.side_effect = [first_inst, second_inst]

        first = get_qdrant_client()
        assert first is first_inst

        reset_qdrant_client()

        second = get_qdrant_client()
        assert second is second_inst
        assert mock_client_class.call_count == 2

    @patch("app.core.qdrant.QdrantClient")
    def test_reset_is_safe_when_no_client_cached(self, mock_client_class):
        """从未调用 ``get_qdrant_client`` 时 reset 也应静默通过。"""
        # 不应抛错
        reset_qdrant_client()
        assert qdrant_module._qdrant_client is None


# ─── ensure_collection_exists ─────────────────────────────────────────


class TestEnsureCollectionExists:
    """``ensure_collection_exists`` 幂等创建 ``document_chunks`` collection。"""

    @patch("app.core.qdrant.get_qdrant_client")
    def test_creates_collection_when_missing(self, mock_get_client):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        mock_get_client.return_value = client

        ensure_collection_exists()

        client.create_collection.assert_called_once()

    @patch("app.core.qdrant.get_qdrant_client")
    def test_creates_collection_with_correct_name(self, mock_get_client):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        mock_get_client.return_value = client

        ensure_collection_exists()

        kwargs = client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "document_chunks"

    @patch("app.core.qdrant.get_qdrant_client")
    def test_dense_vector_config_is_1024_cosine(self, mock_get_client):
        """Dense 向量必须命名为 ``dense``、1024 维、Cosine 距离。"""
        from qdrant_client.models import Distance, VectorParams

        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        mock_get_client.return_value = client

        ensure_collection_exists()

        kwargs = client.create_collection.call_args.kwargs
        vectors_config = kwargs["vectors_config"]

        assert "dense" in vectors_config
        dense_params = vectors_config["dense"]
        assert isinstance(dense_params, VectorParams)
        assert dense_params.size == 1024
        assert dense_params.distance == Distance.COSINE

    @patch("app.core.qdrant.get_qdrant_client")
    def test_sparse_vector_config_is_present(self, mock_get_client):
        """Sparse 向量必须命名为 ``sparse``，使用 SparseVectorParams。"""
        from qdrant_client.models import SparseVectorParams

        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        mock_get_client.return_value = client

        ensure_collection_exists()

        kwargs = client.create_collection.call_args.kwargs
        sparse_config = kwargs["sparse_vectors_config"]

        assert "sparse" in sparse_config
        assert isinstance(sparse_config["sparse"], SparseVectorParams)

    @patch("app.core.qdrant.get_qdrant_client")
    def test_skips_creation_when_collection_exists(self, mock_get_client):
        """已存在时不再调用 create_collection（幂等）。"""
        client = MagicMock()
        existing = MagicMock()
        existing.name = "document_chunks"
        client.get_collections.return_value = MagicMock(collections=[existing])
        mock_get_client.return_value = client

        ensure_collection_exists()

        client.create_collection.assert_not_called()

    @patch("app.core.qdrant.get_qdrant_client")
    def test_idempotent_across_repeated_calls(self, mock_get_client):
        """连续两次调用：首次创建，第二次因已存在而跳过。"""
        client = MagicMock()
        existing = MagicMock()
        existing.name = "document_chunks"
        # 第一次：collection 不存在；第二次：已存在
        client.get_collections.side_effect = [
            MagicMock(collections=[]),
            MagicMock(collections=[existing]),
        ]
        mock_get_client.return_value = client

        ensure_collection_exists()
        ensure_collection_exists()

        assert client.create_collection.call_count == 1

    @patch("app.core.qdrant.get_qdrant_client")
    def test_propagates_unexpected_response(self, mock_get_client):
        """底层 UnexpectedResponse 应向上抛，方便上游处理。"""
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collections.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"boom",
            headers=None,
        )
        mock_get_client.return_value = client

        with pytest.raises(UnexpectedResponse):
            ensure_collection_exists()


# ─── delete_collection ───────────────────────────────────────────────


class TestDeleteCollection:
    """``delete_collection`` 用于测试/重置场景。"""

    @patch("app.core.qdrant.get_qdrant_client")
    def test_calls_qdrant_delete_collection(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client

        delete_collection()

        client.delete_collection.assert_called_once_with(
            collection_name="document_chunks"
        )

    @patch("app.core.qdrant.get_qdrant_client")
    def test_swallows_unexpected_response_when_missing(self, mock_get_client):
        """collection 不存在时不应抛错。"""
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.delete_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"missing",
            headers=None,
        )
        mock_get_client.return_value = client

        # 不应抛错
        delete_collection()
