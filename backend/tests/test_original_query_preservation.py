"""任务 15.4：原始查询保留逻辑单元测试。

需求 7.4：搜索引擎在查询增强过程中，必须将原始查询作为必选检索条件纳入最终检索，
确保返回结果始终包含与原始查询直接匹配的内容。

本文件聚焦 ``EnhancedQuery.all_text_queries`` 这一新增字段的契约：

- ``enhance(query)`` 返回的 ``all_text_queries`` 第一项**永远是** ``query``
- 各子模块（改写 / HyDE / 分解）即使全部返回空、超时或抛异常，``all_text_queries``
  也至少包含 ``query`` 这一条
- 当 LLM 返回的改写或子查询恰好与原始查询相同时，不会在合集中产生重复
- 改写、子查询合并时按首次出现顺序去重；同一字符串只出现一次
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.llm_gateway import LLMGatewayError
from app.services.query_enhancer import (
    EnhancedQuery,
    QueryEnhancer,
    QueryEnhancerConfig,
)


# ─── 工具函数 ──────────────────────────────────────────────────────────


def _make_llm_response(content: str):
    """构造 LLM 的伪响应对象。"""
    response = MagicMock()
    response.content = content
    return response


@pytest.fixture
def mock_llm_gateway():
    """注入可控的 LLM 网关。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock()
    return gateway


@pytest.fixture
def mock_embedding_service():
    """注入可控的 embedding 服务，避免真实网络调用。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(
        return_value=MagicMock(
            dense_vector=[0.1] * 1024,
            sparse_indices=[],
            sparse_values=[],
        )
    )
    return service


@pytest.fixture
def enhancer(mock_llm_gateway, mock_embedding_service):
    """默认全部子模块启用的增强器。"""
    return QueryEnhancer(
        llm_gateway=mock_llm_gateway,
        embedding_service=mock_embedding_service,
    )


# ─── 1. 原始查询始终首位且至少存在 ─────────────────────────────────────


class TestAllTextQueriesAlwaysContainsOriginal:
    """验证 ``all_text_queries`` 始终包含且首位为原始查询。"""

    @pytest.mark.asyncio
    async def test_hello_present_with_normal_response(
        self, enhancer, mock_llm_gateway
    ):
        """正常路径：``all_text_queries[0] == "hello"`` 且包含 hello。"""
        # rewrite + decompose 都用同一个返回；HyDE 只取 dense vector 不参与文本合集
        mock_llm_gateway.complete.return_value = _make_llm_response(
            '["hello world", "say hello"]'
        )

        result = await enhancer.enhance("hello")

        assert isinstance(result, EnhancedQuery)
        assert result.original == "hello"
        assert result.all_text_queries, "all_text_queries 不应为空"
        assert result.all_text_queries[0] == "hello"
        assert "hello" in result.all_text_queries

    @pytest.mark.asyncio
    async def test_alias_property_original_query(
        self, enhancer, mock_llm_gateway
    ):
        """``original_query`` 别名属性应等于 ``original``。"""
        mock_llm_gateway.complete.return_value = _make_llm_response("[]")

        result = await enhancer.enhance("hello")

        assert result.original_query == result.original == "hello"

    @pytest.mark.asyncio
    async def test_hypothetical_embeddings_alias(
        self, enhancer, mock_llm_gateway, mock_embedding_service
    ):
        """``hypothetical_embeddings`` 别名等于 ``hyde_embeddings``。"""
        mock_llm_gateway.complete.return_value = _make_llm_response(
            "这是一个足够长的假设文档段落，用于覆盖测试场景验证。"
        )

        result = await enhancer.enhance("机器学习")

        assert result.hypothetical_embeddings is result.hyde_embeddings


# ─── 2. 子模块全部返回空时 → 仅含原始查询 ─────────────────────────────


class TestAllSubModulesEmpty:
    """所有子模块返回空时，``all_text_queries`` 仅包含原始查询。"""

    @pytest.mark.asyncio
    async def test_empty_rewrites_and_decompose(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """改写与分解都返回 ``[]``，HyDE 关闭：``all_text_queries == ["hello"]``。"""
        config = QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=False,
            enable_decomposition=True,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )

        # query_enhancer 内嵌的 _parse_variants 是按行解析；
        # 空字符串 → 解析得 []；分解模块判断不到 2 条子查询时也返回 []
        mock_llm_gateway.complete.return_value = _make_llm_response("")

        result = await enhancer.enhance("hello")

        assert result.variants == []
        assert result.sub_queries == []
        assert result.all_text_queries == ["hello"]

    @pytest.mark.asyncio
    async def test_all_features_disabled(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """所有功能关闭：``all_text_queries == ["hello"]`` 且不调用 LLM。"""
        config = QueryEnhancerConfig(
            enable_rewrite=False,
            enable_hyde=False,
            enable_decomposition=False,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )

        result = await enhancer.enhance("hello")

        assert result.original == "hello"
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        assert result.all_text_queries == ["hello"]
        mock_llm_gateway.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_collection(self, enhancer):
        """空查询：``all_text_queries`` 为 ``[]``，不强行注入空字符串。"""
        result = await enhancer.enhance("")

        assert result.original == ""
        assert result.all_text_queries == []

    @pytest.mark.asyncio
    async def test_whitespace_input_returns_empty_collection(self, enhancer):
        """仅空白的查询：``all_text_queries`` 为 ``[]``。"""
        result = await enhancer.enhance("   ")

        assert result.all_text_queries == []


# ─── 3. 重复去重：rewrites 含原始查询时不重复 ──────────────────────────


class TestNoDuplicateOfOriginal:
    """改写或子查询恰好与原始查询相同时，不在合集中产生重复。"""

    @pytest.mark.asyncio
    async def test_rewrite_contains_original_no_duplicate(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """改写中包含 ``"hello"``：``all_text_queries`` 中 hello 仅出现一次。"""
        config = QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=False,
            enable_decomposition=False,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )
        # 注：内部 _parse_variants 会按行解析；这里直接给出多行
        mock_llm_gateway.complete.return_value = _make_llm_response(
            "hello\nhi\nhello world"
        )

        result = await enhancer.enhance("hello")

        # hello 只出现一次（首位），且其余补充查询保留
        assert result.all_text_queries[0] == "hello"
        assert result.all_text_queries.count("hello") == 1
        # hi 与 hello world 至少其中之一会出现（具体取决于解析；用包含断言）
        non_original = [q for q in result.all_text_queries if q != "hello"]
        assert any("hi" in q or "world" in q for q in non_original)

    @pytest.mark.asyncio
    async def test_rewrite_with_whitespace_padded_original(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """改写返回 ``"hello "``（带空白）也视为重复，不再追加。"""
        config = QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=False,
            enable_decomposition=False,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )
        mock_llm_gateway.complete.return_value = _make_llm_response(
            "hello \n hi"
        )

        result = await enhancer.enhance("hello")

        # 仅一次 hello（首位为 ``"hello"`` 原文，"hello " 被识别为重复）
        assert result.all_text_queries[0] == "hello"
        # 不应出现仅末尾空白差异的重复 "hello " 项
        assert "hello " not in result.all_text_queries

    @pytest.mark.asyncio
    async def test_subquery_overlap_with_rewrite_dedup(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """子查询与改写之间也按首次出现顺序去重。"""
        config = QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=False,
            enable_decomposition=True,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )
        # 第一次调用为改写，第二次为分解
        mock_llm_gateway.complete.side_effect = [
            _make_llm_response("greeting hello\nsay hi"),
            _make_llm_response("greeting hello\nhow to say hi"),
        ]

        result = await enhancer.enhance("hello")

        # 首位为原始查询
        assert result.all_text_queries[0] == "hello"
        # "greeting hello" 在改写与子查询中各出现一次，合集中只保留一次
        assert result.all_text_queries.count("greeting hello") == 1


# ─── 4. 子模块超时 → 仍保留原始查询 ────────────────────────────────────


class TestPreservationOnSubModuleTimeout:
    """子模块或整体超时仍保留原始查询。"""

    @pytest.mark.asyncio
    async def test_individual_module_timeout_still_keeps_original(
        self, enhancer, mock_llm_gateway
    ):
        """单个子模块超时不影响原始查询保留。"""

        async def slow_complete(*args, **kwargs):
            # 单次调用超过子模块自身的超时（2-3 秒），但低于整体 5 秒
            await asyncio.sleep(4.0)
            return _make_llm_response("[]")

        mock_llm_gateway.complete.side_effect = slow_complete

        result = await enhancer.enhance("hello")

        assert result.original == "hello"
        # 各子模块都返回空，all_text_queries 至少含原始查询
        assert result.all_text_queries == ["hello"]

    @pytest.mark.asyncio
    async def test_overall_timeout_falls_back_to_original_only(
        self, enhancer, mock_llm_gateway
    ):
        """整体 5 秒超时降级时仍保留原始查询。"""

        async def very_slow_complete(*args, **kwargs):
            await asyncio.sleep(20)  # 超过整体 5s 总预算
            return _make_llm_response("不会到这里")

        mock_llm_gateway.complete.side_effect = very_slow_complete

        result = await enhancer.enhance("hello")

        assert result.original == "hello"
        assert result.all_text_queries == ["hello"]
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []


# ─── 5. 子模块抛异常 → 仍保留原始查询 ──────────────────────────────────


class TestPreservationOnSubModuleException:
    """子模块抛任意异常时仍保留原始查询。"""

    @pytest.mark.asyncio
    async def test_llm_gateway_error_keeps_original(
        self, enhancer, mock_llm_gateway
    ):
        """LLM 网关错误（限流/鉴权失败等）：保留原始查询。"""
        mock_llm_gateway.complete.side_effect = LLMGatewayError(
            "rate limited", reason="rate_limit"
        )

        result = await enhancer.enhance("hello")

        assert result.original == "hello"
        assert result.all_text_queries == ["hello"]

    @pytest.mark.asyncio
    async def test_unexpected_exception_keeps_original(
        self, enhancer, mock_llm_gateway
    ):
        """子模块抛未预期异常：保留原始查询，整体降级返回。"""
        mock_llm_gateway.complete.side_effect = RuntimeError("boom")

        result = await enhancer.enhance("hello")

        assert result.original == "hello"
        # 即使所有子模块都炸，原始查询也必须出现在合集中
        assert "hello" in result.all_text_queries
        assert result.all_text_queries[0] == "hello"

    @pytest.mark.asyncio
    async def test_partial_exception_other_modules_still_contribute(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """部分子模块抛异常时，其它成功结果仍能合并到 ``all_text_queries``。"""
        config = QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=False,
            enable_decomposition=True,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )

        call_count = [0]

        async def selective(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # 改写：成功返回两条变体
                return _make_llm_response("greeting\nsay hi")
            # 分解：抛 LLMGatewayError，应被各自子模块吞掉降级为 []
            raise LLMGatewayError("boom", reason="server_error")

        mock_llm_gateway.complete.side_effect = selective

        result = await enhancer.enhance("hello")

        # 原始查询首位
        assert result.all_text_queries[0] == "hello"
        # 改写的两条变体都应进入合集
        assert "greeting" in result.all_text_queries
        assert "say hi" in result.all_text_queries
        # 分解失败：sub_queries 仍为空
        assert result.sub_queries == []


# ─── 6. _build_all_text_queries 直接测试 ──────────────────────────────


class TestBuildAllTextQueries:
    """直接测试静态合并函数 ``_build_all_text_queries``。"""

    def test_empty_query_returns_empty(self):
        result = QueryEnhancer._build_all_text_queries("", [], [])
        assert result == []

    def test_whitespace_query_returns_empty(self):
        result = QueryEnhancer._build_all_text_queries("   ", ["a"], ["b"])
        assert result == []

    def test_original_first_then_rewrites_then_subs(self):
        result = QueryEnhancer._build_all_text_queries(
            "hello", ["hi", "greet"], ["how do you say hi"]
        )
        assert result[0] == "hello"
        assert result == ["hello", "hi", "greet", "how do you say hi"]

    def test_dedup_across_rewrites_and_sub_queries(self):
        result = QueryEnhancer._build_all_text_queries(
            "hello", ["hi", "greet"], ["greet", "another"]
        )
        # greet 只出现一次
        assert result.count("greet") == 1
        assert result == ["hello", "hi", "greet", "another"]

    def test_dedup_when_rewrite_equals_original(self):
        result = QueryEnhancer._build_all_text_queries(
            "hello", ["hello", "hi"], []
        )
        assert result.count("hello") == 1
        assert result == ["hello", "hi"]

    def test_skips_empty_and_whitespace_items(self):
        result = QueryEnhancer._build_all_text_queries(
            "hello", ["", "   ", "hi"], []
        )
        assert result == ["hello", "hi"]
