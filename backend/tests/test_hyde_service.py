"""``HyDEService`` 单元测试（任务 15.2）。

覆盖需求 7.2 关键场景：
- LLM 正常生成 1-3 段假设文档并被 embed 成向量
- LLM 输出超过 3 段时被截断为 3 段
- 空白 query 直接返回空列表（不调 LLM、不调 embedding）
- LLM 超时（3 秒）降级返回空列表
- LLM 抛异常时降级返回空列表
- Embedding 服务整体异常时降级
- 单段 embedding 失败时仅丢弃该段，其它成功段落仍返回
- 全部段落 embedding 失败时返回空列表
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.embedding_service import EmbeddingResult
from app.services.hyde_service import (
    HYDE_TIMEOUT_SECONDS,
    MAX_HYPOTHETICAL_DOCUMENTS,
    HyDEService,
)
from app.services.llm_gateway import LLMGatewayError

# ─── Fixtures ──────────────────────────────────────────────────────────


def _make_response(content: str) -> MagicMock:
    """构造模拟的 ``LLMResponse`` 对象。"""
    response = MagicMock()
    response.content = content
    return response


def _make_embedding(vector: list[float] | None = None) -> EmbeddingResult:
    """构造模拟的 ``EmbeddingResult``，默认返回 1024 维全零向量。"""
    return EmbeddingResult(
        chunk_id="query",
        dense_vector=vector if vector is not None else [0.1] * 1024,
        sparse_indices=[],
        sparse_values=[],
    )


@pytest.fixture
def mock_llm() -> AsyncMock:
    """提供带 ``complete`` 异步方法的 LLMGateway mock。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock()
    return gateway


@pytest.fixture
def mock_embedding() -> AsyncMock:
    """提供带 ``embed_query`` 异步方法的 EmbeddingService mock。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(return_value=_make_embedding())
    return service


@pytest.fixture
def service(mock_llm: AsyncMock, mock_embedding: AsyncMock) -> HyDEService:
    """注入 mock 的 ``HyDEService`` 默认实例（3s 超时，3 段上限）。"""
    return HyDEService(
        llm_gateway=mock_llm,
        embedding_service=mock_embedding,
    )


# ─── 输入校验 ──────────────────────────────────────────────────────────


class TestInputValidation:
    """测试输入校验：空字符串 / 空白不调用任何下游服务。"""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_without_calls(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """空字符串应直接返回空列表，且不触发 LLM / embedding 调用。"""
        result = await service.generate_hypothetical_embeddings("")

        assert result == []
        mock_llm.complete.assert_not_called()
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty_without_calls(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """仅空白字符的查询同样不应触发任何下游调用。"""
        result = await service.generate_hypothetical_embeddings("   \n\t  ")

        assert result == []
        mock_llm.complete.assert_not_called()
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_max_documents_returns_empty(
        self,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """``max_documents=0`` 时应直接返回空列表，不调 LLM。"""
        s = HyDEService(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
            max_documents=0,
        )

        result = await s.generate_hypothetical_embeddings("机器学习")

        assert result == []
        mock_llm.complete.assert_not_called()


# ─── 正常生成路径 ──────────────────────────────────────────────────────


class TestSuccessfulGeneration:
    """测试 LLM 正常返回时的解析、embedding 与裁剪。"""

    @pytest.mark.asyncio
    async def test_returns_one_to_three_vectors(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 输出 2 段假设文档时应返回 2 个 dense 向量。"""
        mock_llm.complete.return_value = _make_response(
            '['
            '"机器学习是一类通过数据训练统计模型来完成预测、分类、聚类等任务的方法，'
            '广泛应用于推荐系统、计算机视觉和自然语言处理。",'
            '"机器学习算法可分为监督学习、无监督学习与强化学习三大类，'
            '常见模型包括线性回归、决策树、神经网络等。"'
            ']'
        )
        # 让两次 embed 返回不同向量，便于断言顺序
        mock_embedding.embed_query.side_effect = [
            _make_embedding([0.1] * 1024),
            _make_embedding([0.2] * 1024),
        ]

        result = await service.generate_hypothetical_embeddings("机器学习是什么")

        assert len(result) == 2
        # 每个向量应是 dense 向量且长度 1024
        for vec in result:
            assert isinstance(vec, list)
            assert len(vec) == 1024
        # embed_query 应被精确调用 2 次
        assert mock_embedding.embed_query.call_count == 2

    @pytest.mark.asyncio
    async def test_truncates_to_max_documents(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 返回多于 3 段时应被截断到 3 段，且 embedding 也只调用 3 次。"""
        import json as _json

        many_docs = [
            f"这是第{i}段足够长的假设文档段落，用于测试截断逻辑是否生效。" * 2
            for i in range(6)
        ]
        mock_llm.complete.return_value = _make_response(
            _json.dumps(many_docs, ensure_ascii=False)
        )

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert len(result) == MAX_HYPOTHETICAL_DOCUMENTS
        assert mock_embedding.embed_query.call_count == MAX_HYPOTHETICAL_DOCUMENTS

    @pytest.mark.asyncio
    async def test_filters_short_paragraphs(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """过短的段落（< 20 字符）应被丢弃，不进入 embedding。"""
        mock_llm.complete.return_value = _make_response(
            '["短", "这是一段足够长的假设文档，用于覆盖向量检索的语义空间。"]'
        )

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert len(result) == 1
        assert mock_embedding.embed_query.call_count == 1

    @pytest.mark.asyncio
    async def test_dedupes_repeated_paragraphs(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """完全相同的段落应去重，仅 embed 一次。"""
        mock_llm.complete.return_value = _make_response(
            '['
            '"这是一段足够长的假设文档，用于覆盖向量检索的语义空间。",'
            '"这是一段足够长的假设文档，用于覆盖向量检索的语义空间。"'
            ']'
        )

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert len(result) == 1
        assert mock_embedding.embed_query.call_count == 1

    @pytest.mark.asyncio
    async def test_passes_user_query_to_llm_prompt(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
    ) -> None:
        """LLM 调用的 prompt 中应包含原始查询，保证语义对齐。"""
        mock_llm.complete.return_value = _make_response(
            '["这是一段足够长的假设性回答，用于补充向量检索。"]'
        )

        await service.generate_hypothetical_embeddings("水泥生产工艺")

        mock_llm.complete.assert_called_once()
        kwargs = mock_llm.complete.call_args.kwargs
        assert "水泥生产工艺" in kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_falls_back_to_paragraph_split_when_no_json(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 没有输出合法 JSON 时按段落分隔解析。"""
        mock_llm.complete.return_value = _make_response(
            "1. 这是第一段足够长的假设性回答，用于补充向量检索的语义空间。\n\n"
            "2. 这是第二段足够长的假设性回答，描述了相关技术细节与适用场景。"
        )

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert len(result) == 2
        # 验证段落前缀编号被剥离：传入 embedding 的文本不应以 "1." 开头
        for call in mock_embedding.embed_query.await_args_list:
            text = call.args[0]
            assert not text.startswith("1.")
            assert not text.startswith("2.")


# ─── 降级路径：超时 / 异常 / 非法输出 ─────────────────────────────────


class TestDegradation:
    """测试各种失败场景下的降级行为。"""

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_empty(
        self,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 超过总超时未返回时应降级为空列表。

        通过将外层 timeout 设为很小的值（0.05s）+ 让 mock 睡更久（0.5s）
        来在测试中可靠触发 ``asyncio.TimeoutError``，避免真实等待 3s。
        """

        async def slow_complete(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            return _make_response('["不该被看到"]')

        mock_llm.complete.side_effect = slow_complete
        s = HyDEService(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
            timeout=0.05,
        )

        result = await s.generate_hypothetical_embeddings("测试查询")

        assert result == []
        # embedding 不应被调用，因为 LLM 阶段就已超时
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_timeout_is_three_seconds(self) -> None:
        """默认超时常量需符合设计文档（HyDE 3s 预算）。"""
        assert HYDE_TIMEOUT_SECONDS == 3.0

    @pytest.mark.asyncio
    async def test_llm_gateway_error_returns_empty(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 网关抛错（限流 / 鉴权失败等）应降级。"""
        mock_llm.complete.side_effect = LLMGatewayError(
            "rate limited", reason="rate_limit"
        )

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert result == []
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_unexpected_llm_exception_returns_empty(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 抛任意未预期异常应降级，不向上抛。"""
        mock_llm.complete.side_effect = RuntimeError("network down")

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert result == []
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_empty(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 返回空内容时应降级，不调 embedding。"""
        mock_llm.complete.return_value = _make_response("   \n\n  \t  ")

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert result == []
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_short_paragraphs_returns_empty(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """LLM 输出全部段落都过短时（< 20 字符）应降级。"""
        mock_llm.complete.return_value = _make_response('["短", "也短", "还是短"]')

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert result == []
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_embedding_failure_keeps_successful(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """部分段落 embedding 失败时，仅丢弃失败段，其它成功段落仍返回。"""
        mock_llm.complete.return_value = _make_response(
            '['
            '"第一段足够长的假设文档，用于覆盖向量检索的语义空间。",'
            '"第二段足够长的假设文档，描述了相关技术细节与适用场景。",'
            '"第三段足够长的假设文档，从不同角度补充答案细节。"'
            ']'
        )
        # 第二段失败，其它段成功
        mock_embedding.embed_query.side_effect = [
            _make_embedding([0.1] * 1024),
            RuntimeError("embedding api down"),
            _make_embedding([0.3] * 1024),
        ]

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert len(result) == 2
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.3] * 1024

    @pytest.mark.asyncio
    async def test_all_embeddings_fail_returns_empty(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """所有段落 embedding 都失败时返回空列表。"""
        mock_llm.complete.return_value = _make_response(
            '['
            '"第一段足够长的假设文档，用于覆盖向量检索的语义空间。",'
            '"第二段足够长的假设文档，描述了相关技术细节与适用场景。"'
            ']'
        )
        mock_embedding.embed_query.side_effect = RuntimeError("embedding api down")

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_empty_dense_vector_is_dropped(
        self,
        service: HyDEService,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
    ) -> None:
        """embedding 返回空 dense 向量的段落应被丢弃。"""
        mock_llm.complete.return_value = _make_response(
            '['
            '"第一段足够长的假设文档，用于覆盖向量检索的语义空间。",'
            '"第二段足够长的假设文档，描述了相关技术细节与适用场景。"'
            ']'
        )
        mock_embedding.embed_query.side_effect = [
            _make_embedding([]),  # 第一段返回空向量
            _make_embedding([0.2] * 1024),
        ]

        result = await service.generate_hypothetical_embeddings("测试查询")

        assert len(result) == 1
        assert result[0] == [0.2] * 1024
