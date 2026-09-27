"""任务 12.3：Embedding 服务（Dense 向量）单元测试。

仅覆盖 Dense 向量相关行为；Sparse 向量由任务 12.4 单独测试，已在
``tests/test_indexing.py`` 中保留。

2026-09 重构：EmbeddingService 不再调用 litellm SDK，改为通过
``openai`` SDK 直连 OpenAI 兼容端点（默认阿里百炼）。测试通过 patch
``app.services.llm_gateway.get_openai_client`` 注入可控客户端 stub，
避免依赖真实网络与大体量依赖。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embedding_service import (
    DENSE_VECTOR_DIM,
    EmbeddingError,
    EmbeddingResult,
    EmbeddingService,
)


# ─── openai client stub ──────────────────────────────────────────────


@pytest.fixture
def openai_client_stub():
    """Patch ``get_openai_client``，注入可控的 AsyncOpenAI 客户端 stub。

    yield 出 ``(client, get_client_mock)``：
    - ``client.embeddings.create`` 是 AsyncMock，可设置返回值/异常；
    - ``get_client_mock`` 可断言客户端以哪个 (api_base, api_key) 被获取。
    """
    client = MagicMock()
    client.embeddings.create = AsyncMock()
    with patch(
        "app.services.llm_gateway.get_openai_client",
        return_value=client,
    ) as get_client_mock:
        yield client, get_client_mock


def _make_response(vectors: list[list[float]]) -> MagicMock:
    """构造一个 ``embeddings.create`` 的响应对象（item.embedding 属性）。"""
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


def _make_settings(
    chat_model: str = "gpt-4o",
    embedding_model: str = "",
    chat_api_base: str = "",
    chat_api_key: str = "",
    embedding_api_base: str = "",
    embedding_api_key: str = "",
) -> MagicMock:
    """构造符合新配置命名（CHAT_* / EMBEDDING_*）的 stub Settings。"""
    return MagicMock(
        CHAT_MODEL=chat_model,
        CHAT_API_BASE=chat_api_base,
        CHAT_API_KEY=chat_api_key,
        EMBEDDING_MODEL=embedding_model,
        EMBEDDING_API_BASE=embedding_api_base,
        EMBEDDING_API_KEY=embedding_api_key,
        EMBEDDING_DIMENSIONS=1024,
        EMBEDDING_TIMEOUT=30.0,
        EMBEDDING_MAX_INPUT_CHARS=6000,
        EMBEDDING_MAX_RETRIES=2,
    )


# ─── 配置默认值 ───────────────────────────────────────────────────────


class TestEmbeddingServiceConfiguration:
    """构造函数读取 settings 并允许显式覆盖。"""

    def test_dense_dimension_defaults_to_1024(self):
        """默认维度必须等于 Settings.EMBEDDING_DIMENSIONS（1024，与 Qdrant 对齐）。"""
        service = EmbeddingService()
        assert service.dimensions == 1024
        assert DENSE_VECTOR_DIM == 1024

    def test_constructor_reads_embedding_model_from_settings(self):
        """优先使用专用 EMBEDDING_MODEL；未配置时回退到 CHAT_MODEL。"""
        with patch("app.services.embedding_service.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                embedding_model="text-embedding-3-large",
            )
            service = EmbeddingService()
            assert service.model == "text-embedding-3-large"

    def test_constructor_falls_back_to_chat_model(self):
        """EMBEDDING_MODEL 为空字符串时，必须回退到 CHAT_MODEL。"""
        with patch("app.services.embedding_service.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(embedding_model="")
            service = EmbeddingService()
            assert service.model == "gpt-4o"

    def test_constructor_prefers_embedding_endpoint_over_chat(self):
        """配置了 EMBEDDING_API_BASE/KEY 时优先于 CHAT_API_BASE/KEY。"""
        with patch("app.services.embedding_service.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                chat_api_base="https://chat.example.com/v1",
                chat_api_key="chat-key",
                embedding_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                embedding_api_key="ali-key",
            )
            service = EmbeddingService()
            assert service.api_base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            assert service.api_key == "ali-key"

    def test_constructor_falls_back_to_chat_endpoint(self):
        """EMBEDDING_API_BASE/KEY 为空时回退到 CHAT_API_BASE/KEY。"""
        with patch("app.services.embedding_service.get_settings") as mock_settings:
            mock_settings.return_value = _make_settings(
                chat_api_base="https://chat.example.com/v1",
                chat_api_key="chat-key",
            )
            service = EmbeddingService()
            assert service.api_base == "https://chat.example.com/v1"
            assert service.api_key == "chat-key"

    def test_explicit_args_override_settings(self):
        """显式参数优先于 Settings。"""
        service = EmbeddingService(
            model="custom-embed",
            api_base="https://example.com",
            api_key="secret",
            batch_size=8,
            dimensions=512,
            timeout=5.0,
            max_input_chars=100,
            max_retries=0,
        )
        assert service.model == "custom-embed"
        assert service.api_base == "https://example.com"
        assert service.api_key == "secret"
        assert service.batch_size == 8
        assert service.dimensions == 512
        assert service.timeout == 5.0
        assert service.max_input_chars == 100
        assert service.max_retries == 0


# ─── Dense 向量基础行为 ──────────────────────────────────────────────


class TestDenseEmbeddingShape:
    """1024 维输出、批量、空输入。"""

    @pytest.mark.asyncio
    async def test_returns_1024_dim_vectors(self, openai_client_stub):
        """每个向量长度必须等于 1024 维（Qdrant 维度）。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response(
            [[0.1] * 1024, [0.2] * 1024]
        )
        service = EmbeddingService()
        chunks = [
            {"id": "c-1", "text": "alpha"},
            {"id": "c-2", "text": "beta"},
        ]
        results = await service.embed_chunks(chunks)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, EmbeddingResult)
            assert len(r.dense_vector) == 1024

    @pytest.mark.asyncio
    async def test_pads_short_vectors_to_dimensions(self, openai_client_stub):
        """API 返回不足 1024 维时，必须用 0 补齐。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.5] * 512])
        service = EmbeddingService()
        results = await service.embed_chunks([{"id": "c", "text": "x"}])
        v = results[0].dense_vector
        assert len(v) == 1024
        assert v[511] == 0.5
        assert v[512] == 0.0
        assert v[1023] == 0.0

    @pytest.mark.asyncio
    async def test_truncates_long_vectors_to_dimensions(self, openai_client_stub):
        """API 返回多于 1024 维时，必须截断。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.3] * 4096])
        service = EmbeddingService()
        results = await service.embed_chunks([{"id": "c", "text": "x"}])
        assert len(results[0].dense_vector) == 1024

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, openai_client_stub):
        """空 chunk 列表必须直接返回 []，不调用 embedding API。"""
        client, _ = openai_client_stub
        service = EmbeddingService()
        results = await service.embed_chunks([])
        assert results == []
        client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_preserves_chunk_ids(self, openai_client_stub):
        """每个 EmbeddingResult.chunk_id 必须与输入一一对应。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response(
            [[0.0] * 1024, [0.0] * 1024, [0.0] * 1024]
        )
        service = EmbeddingService()
        chunks = [
            {"id": "doc1#0", "text": "a"},
            {"id": "doc1#1", "text": "b"},
            {"id": "doc1#2", "text": "c"},
        ]
        results = await service.embed_chunks(chunks)
        assert [r.chunk_id for r in results] == ["doc1#0", "doc1#1", "doc1#2"]


# ─── 批处理 ─────────────────────────────────────────────────────────


class TestDenseEmbeddingBatching:
    """按 batch_size 分批调用 embedding API。"""

    @pytest.mark.asyncio
    async def test_splits_into_batches_by_batch_size(self, openai_client_stub):
        """5 条文本、batch_size=2 应产生 3 次 API 调用。"""
        client, _ = openai_client_stub

        # 每次调用根据传入数量返回等量向量。
        async def fake_embed(**kwargs):
            n = len(kwargs["input"])
            return _make_response([[0.1] * 1024 for _ in range(n)])

        client.embeddings.create.side_effect = fake_embed
        service = EmbeddingService(batch_size=2)
        chunks = [{"id": f"c-{i}", "text": f"t{i}"} for i in range(5)]
        results = await service.embed_chunks(chunks)
        assert len(results) == 5
        assert client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_passes_correct_model_to_api(self, openai_client_stub):
        """embedding 调用必须原样使用配置的 model 参数。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(model="bge-large-zh", batch_size=4)
        await service.embed_chunks([{"id": "c", "text": "hi"}])
        kwargs = client.embeddings.create.call_args.kwargs
        assert kwargs["model"] == "bge-large-zh"
        assert kwargs["input"] == ["hi"]

    @pytest.mark.asyncio
    async def test_passes_api_base_and_key_to_client(self, openai_client_stub):
        """配置的 api_base / api_key 必须用于获取 openai 客户端。"""
        client, get_client_mock = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(
            api_base="https://gateway.example.com",
            api_key="sk-test",
        )
        await service.embed_chunks([{"id": "c", "text": "hi"}])
        get_client_mock.assert_called_with("https://gateway.example.com", "sk-test")

    @pytest.mark.asyncio
    async def test_blank_endpoint_falls_back_to_blank_client(
        self, openai_client_stub, monkeypatch
    ):
        """settings 与显式参数均为空时，以空端点获取客户端（SDK 用默认）。

        显式 mock settings 让所有相关字段为空,以隔离测试环境的真实 .env。
        """
        monkeypatch.setattr(
            "app.services.embedding_service.get_settings",
            lambda: _make_settings(),
        )

        client, get_client_mock = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(api_base="", api_key="")
        await service.embed_chunks([{"id": "c", "text": "hi"}])
        get_client_mock.assert_called_with("", "")


# ─── 输入截断 ────────────────────────────────────────────────────────


class TestDenseEmbeddingInputTruncation:
    """超长文本必须截断；空字符串必须替换。"""

    @pytest.mark.asyncio
    async def test_long_text_is_truncated_before_call(self, openai_client_stub):
        """超过 max_input_chars 的文本被裁剪到 max_input_chars。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(max_input_chars=50)
        long_text = "x" * 1000
        await service.embed_chunks([{"id": "c", "text": long_text}])
        passed = client.embeddings.create.call_args.kwargs["input"][0]
        assert len(passed) == 50

    @pytest.mark.asyncio
    async def test_short_text_is_unchanged(self, openai_client_stub):
        """短文本必须按原样发送。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService(max_input_chars=50)
        await service.embed_chunks([{"id": "c", "text": "hello world"}])
        passed = client.embeddings.create.call_args.kwargs["input"][0]
        assert passed == "hello world"

    @pytest.mark.asyncio
    async def test_empty_text_replaced_with_space(self, openai_client_stub):
        """空字符串会被替换为单空格，避免被 embedding API 拒绝。"""
        client, _ = openai_client_stub
        client.embeddings.create.return_value = _make_response([[0.0] * 1024])
        service = EmbeddingService()
        await service.embed_chunks([{"id": "c", "text": ""}])
        passed = client.embeddings.create.call_args.kwargs["input"][0]
        assert passed != ""
        assert passed.strip() == ""


# ─── 失败处理 ────────────────────────────────────────────────────────


class TestDenseEmbeddingErrorHandling:
    """超时、API 错误、重试。"""

    @pytest.mark.asyncio
    async def test_api_failure_raises_embedding_error(self, openai_client_stub):
        """API 抛错时必须包装成 EmbeddingError 并暴露上下文。"""
        client, _ = openai_client_stub
        client.embeddings.create.side_effect = RuntimeError("upstream 503")
        # max_retries=0 让失败直接到达终态，避免测试等待 backoff。
        service = EmbeddingService(max_retries=0, timeout=1.0, model="bge-m3")
        with pytest.raises(EmbeddingError) as exc_info:
            await service.embed_chunks([{"id": "c", "text": "x"}])
        message = str(exc_info.value)
        # 错误信息必须可定位：模型名、原始错误。
        assert "bge-m3" in message
        assert "upstream 503" in message
        # 原始异常作为 cause 链接。
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_timeout_raises_embedding_error(self, openai_client_stub):
        """单批次超过 timeout 时必须抛 EmbeddingError。"""
        client, _ = openai_client_stub

        async def slow_embed(**kwargs):
            await asyncio.sleep(0.5)
            return _make_response([[0.0] * 1024])

        client.embeddings.create.side_effect = slow_embed
        service = EmbeddingService(max_retries=0, timeout=0.05)
        with pytest.raises(EmbeddingError):
            await service.embed_chunks([{"id": "c", "text": "x"}])

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, openai_client_stub, monkeypatch):
        """前两次失败、第三次成功必须最终返回向量。"""
        client, _ = openai_client_stub
        # 让 backoff 不消耗实际墙钟时间。
        monkeypatch.setattr(
            "app.services.embedding_service.asyncio.sleep",
            AsyncMock(),
        )
        calls = {"n": 0}

        async def flaky(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return _make_response([[0.7] * 1024])

        client.embeddings.create.side_effect = flaky
        service = EmbeddingService(max_retries=2, timeout=1.0)
        results = await service.embed_chunks([{"id": "c", "text": "x"}])
        assert calls["n"] == 3
        assert len(results) == 1
        assert len(results[0].dense_vector) == 1024
        assert results[0].dense_vector[0] == 0.7

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_fails(self, openai_client_stub, monkeypatch):
        """重试用尽仍失败时抛 EmbeddingError，调用次数 = max_retries+1。"""
        client, _ = openai_client_stub
        monkeypatch.setattr(
            "app.services.embedding_service.asyncio.sleep",
            AsyncMock(),
        )
        client.embeddings.create.side_effect = RuntimeError("permanent")
        service = EmbeddingService(max_retries=2, timeout=1.0)
        with pytest.raises(EmbeddingError):
            await service.embed_chunks([{"id": "c", "text": "x"}])
        assert client.embeddings.create.call_count == 3
