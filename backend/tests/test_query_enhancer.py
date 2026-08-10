"""Unit tests for the QueryEnhancer service.

Tests cover:
- Query rewriting (LLM generates up to 5 semantic variants, 2s timeout)
- HyDE (generates 1-3 hypothetical document embeddings)
- Sub-query decomposition (splits multi-part queries into ≤5 sub-queries)
- Original query preservation (always included in results)
- Timeout degradation (5s overall timeout, fallback to original query)
- Config toggle (enable/disable rewrite, HyDE, decomposition)
"""

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.query_enhancer import (
    DECOMPOSE_TIMEOUT,
    HYDE_TIMEOUT,
    MAX_HYDE_DOCUMENTS,
    MAX_REWRITE_VARIANTS,
    MAX_SUB_QUERIES,
    OVERALL_TIMEOUT,
    REWRITE_TIMEOUT,
    EnhancedQuery,
    QueryEnhancer,
    QueryEnhancerConfig,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_gateway():
    """Create a mock LLM gateway."""
    gateway = AsyncMock()
    gateway.complete = AsyncMock()
    return gateway


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.embed_query = AsyncMock(return_value=MagicMock(
        dense_vector=[0.1] * 1024,
        sparse_indices=[1, 5, 10],
        sparse_values=[0.5, 0.3, 0.8],
    ))
    return service


@pytest.fixture
def enhancer(mock_llm_gateway, mock_embedding_service):
    """Create a QueryEnhancer with mocked dependencies."""
    return QueryEnhancer(
        llm_gateway=mock_llm_gateway,
        embedding_service=mock_embedding_service,
    )


@pytest.fixture
def enhancer_all_disabled(mock_llm_gateway, mock_embedding_service):
    """Create a QueryEnhancer with all features disabled."""
    config = QueryEnhancerConfig(
        enable_rewrite=False,
        enable_hyde=False,
        enable_decomposition=False,
    )
    return QueryEnhancer(
        llm_gateway=mock_llm_gateway,
        embedding_service=mock_embedding_service,
        config=config,
    )


def make_llm_response(content: str):
    """Helper to create a mock LLM response."""
    response = MagicMock()
    response.content = content
    return response


# ─── Original Query Preservation Tests ─────────────────────────────────


class TestOriginalQueryPreservation:
    """Tests that original query is always preserved in results."""

    @pytest.mark.asyncio
    async def test_original_query_always_present(self, enhancer, mock_llm_gateway):
        """Enhanced result should always contain the original query."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "改写1\n改写2\n改写3"
        )

        result = await enhancer.enhance("机器学习算法")

        assert result.original == "机器学习算法"

    @pytest.mark.asyncio
    async def test_original_query_on_empty_input(self, enhancer):
        """Empty query should return EnhancedQuery with empty original."""
        result = await enhancer.enhance("")

        assert result.original == ""
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_original_query_on_whitespace_input(self, enhancer):
        """Whitespace-only query should return without enhancement."""
        result = await enhancer.enhance("   ")

        assert result.original == "   "
        assert result.variants == []

    @pytest.mark.asyncio
    async def test_original_query_preserved_on_failure(
        self, enhancer, mock_llm_gateway
    ):
        """Original query should be preserved even when all enhancements fail."""
        mock_llm_gateway.complete.side_effect = Exception("LLM unavailable")

        result = await enhancer.enhance("测试查询")

        assert result.original == "测试查询"


# ─── Query Rewrite Tests ──────────────────────────────────────────────


class TestQueryRewrite:
    """Tests for query rewriting functionality."""

    @pytest.mark.asyncio
    async def test_rewrite_generates_variants(self, enhancer, mock_llm_gateway):
        """Rewrite should generate semantic variants from LLM response."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "如何使用机器学习\n机器学习的应用方法\nML算法实践指南"
        )

        result = await enhancer.enhance("机器学习怎么用")

        assert len(result.variants) == 3
        assert "如何使用机器学习" in result.variants
        assert "机器学习的应用方法" in result.variants
        assert "ML算法实践指南" in result.variants

    @pytest.mark.asyncio
    async def test_rewrite_max_5_variants(self, enhancer, mock_llm_gateway):
        """Rewrite should return at most 5 variants."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "变体1\n变体2\n变体3\n变体4\n变体5\n变体6\n变体7"
        )

        result = await enhancer.enhance("测试查询")

        assert len(result.variants) <= MAX_REWRITE_VARIANTS

    @pytest.mark.asyncio
    async def test_rewrite_handles_numbered_list(self, enhancer, mock_llm_gateway):
        """Rewrite should parse numbered list format."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "1. 第一个改写\n2. 第二个改写\n3. 第三个改写"
        )

        result = await enhancer.enhance("原始查询")

        assert len(result.variants) == 3
        assert "第一个改写" in result.variants

    @pytest.mark.asyncio
    async def test_rewrite_handles_bullet_list(self, enhancer, mock_llm_gateway):
        """Rewrite should parse bullet point format."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "- 改写A\n- 改写B\n- 改写C"
        )

        result = await enhancer.enhance("原始查询")

        assert len(result.variants) == 3
        assert "改写A" in result.variants

    @pytest.mark.asyncio
    async def test_rewrite_timeout_returns_empty(
        self, enhancer, mock_llm_gateway
    ):
        """Rewrite should return empty list on timeout."""

        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(10)
            return make_llm_response("too late")

        mock_llm_gateway.complete.side_effect = slow_complete

        result = await enhancer.enhance("测试查询")

        assert result.variants == []
        assert result.original == "测试查询"

    @pytest.mark.asyncio
    async def test_rewrite_llm_error_returns_empty(
        self, enhancer, mock_llm_gateway
    ):
        """Rewrite should return empty list on LLM error."""
        from app.services.llm_gateway import LLMGatewayError

        mock_llm_gateway.complete.side_effect = LLMGatewayError(
            "Rate limited", reason="rate_limit"
        )

        result = await enhancer.enhance("测试查询")

        assert result.variants == []

    @pytest.mark.asyncio
    async def test_rewrite_empty_response(self, enhancer, mock_llm_gateway):
        """Rewrite should handle empty LLM response."""
        mock_llm_gateway.complete.return_value = make_llm_response("")

        result = await enhancer.enhance("测试查询")

        assert result.variants == []

    @pytest.mark.asyncio
    async def test_rewrite_filters_short_variants(
        self, enhancer, mock_llm_gateway
    ):
        """Rewrite should filter out variants shorter than 2 characters."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "a\n有效改写\n\nb\n另一个有效改写"
        )

        result = await enhancer.enhance("测试")

        # Only variants with length >= 2 should be included
        for variant in result.variants:
            assert len(variant) >= 2


# ─── HyDE Tests ────────────────────────────────────────────────────────


class TestHyDE:
    """Tests for Hypothetical Document Embedding generation."""

    @pytest.mark.asyncio
    async def test_hyde_generates_embeddings(
        self, enhancer, mock_llm_gateway, mock_embedding_service
    ):
        """HyDE should generate embeddings from hypothetical documents."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "这是第一个假设文档，描述了机器学习的基本概念和应用场景。\n\n"
            "这是第二个假设文档，介绍了深度学习在自然语言处理中的应用。"
        )

        result = await enhancer.enhance("机器学习应用")

        assert len(result.hyde_embeddings) == 2
        assert len(result.hyde_embeddings[0]) == 1024
        # Embedding service should be called for each hypothetical doc
        assert mock_embedding_service.embed_query.call_count >= 2

    @pytest.mark.asyncio
    async def test_hyde_max_3_documents(
        self, enhancer, mock_llm_gateway, mock_embedding_service
    ):
        """HyDE should generate at most 3 hypothetical document embeddings."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "假设文档一，内容足够长以通过过滤。\n\n"
            "假设文档二，内容足够长以通过过滤。\n\n"
            "假设文档三，内容足够长以通过过滤。\n\n"
            "假设文档四，内容足够长以通过过滤。\n\n"
            "假设文档五，内容足够长以通过过滤。"
        )

        result = await enhancer.enhance("测试查询")

        assert len(result.hyde_embeddings) <= MAX_HYDE_DOCUMENTS

    @pytest.mark.asyncio
    async def test_hyde_timeout_returns_empty(
        self, enhancer, mock_llm_gateway
    ):
        """HyDE should return empty list on timeout."""

        async def slow_complete(*args, **kwargs):
            await asyncio.sleep(10)
            return make_llm_response("too late")

        mock_llm_gateway.complete.side_effect = slow_complete

        result = await enhancer.enhance("测试查询")

        assert result.hyde_embeddings == []

    @pytest.mark.asyncio
    async def test_hyde_llm_error_returns_empty(
        self, enhancer, mock_llm_gateway
    ):
        """HyDE should return empty list on LLM error."""
        from app.services.llm_gateway import LLMGatewayError

        mock_llm_gateway.complete.side_effect = LLMGatewayError(
            "Timeout", reason="timeout"
        )

        result = await enhancer.enhance("测试查询")

        assert result.hyde_embeddings == []

    @pytest.mark.asyncio
    async def test_hyde_filters_short_paragraphs(
        self, enhancer, mock_llm_gateway, mock_embedding_service
    ):
        """HyDE should filter out paragraphs shorter than 20 characters."""
        mock_llm_gateway.complete.return_value = make_llm_response(
            "短\n\n这是一个足够长的假设文档段落，描述了相关内容。"
        )

        result = await enhancer.enhance("测试查询")

        # Only the long paragraph should be embedded
        assert len(result.hyde_embeddings) == 1


# ─── Sub-query Decomposition Tests ────────────────────────────────────


class TestSubQueryDecomposition:
    """Tests for sub-query decomposition functionality."""

    @pytest.mark.asyncio
    async def test_decompose_multi_part_query(
        self, enhancer, mock_llm_gateway
    ):
        """Should decompose multi-part queries into sub-queries."""
        # First call for rewrite, second for hyde, third for decompose
        mock_llm_gateway.complete.side_effect = [
            make_llm_response("改写1\n改写2"),  # rewrite
            make_llm_response("假设文档内容足够长以通过过滤条件。"),  # hyde
            make_llm_response(
                "水泥的生产工艺流程是什么\n水泥的质量标准有哪些"
            ),  # decompose
        ]

        result = await enhancer.enhance("水泥的生产工艺和质量标准是什么")

        assert len(result.sub_queries) == 2
        assert "水泥的生产工艺流程是什么" in result.sub_queries
        assert "水泥的质量标准有哪些" in result.sub_queries

    @pytest.mark.asyncio
    async def test_decompose_simple_query_no_split(
        self, enhancer, mock_llm_gateway
    ):
        """Simple queries should not be decomposed."""
        mock_llm_gateway.complete.side_effect = [
            make_llm_response("改写1\n改写2"),  # rewrite
            make_llm_response("假设文档内容足够长以通过过滤条件。"),  # hyde
            make_llm_response("无需分解"),  # decompose
        ]

        result = await enhancer.enhance("什么是机器学习")

        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_decompose_max_5_sub_queries(
        self, enhancer, mock_llm_gateway
    ):
        """Decomposition should return at most 5 sub-queries."""
        mock_llm_gateway.complete.side_effect = [
            make_llm_response("改写1\n改写2"),  # rewrite
            make_llm_response("假设文档内容足够长以通过过滤条件。"),  # hyde
            make_llm_response(
                "子查询1\n子查询2\n子查询3\n子查询4\n子查询5\n子查询6\n子查询7"
            ),  # decompose
        ]

        result = await enhancer.enhance("复杂查询")

        assert len(result.sub_queries) <= MAX_SUB_QUERIES

    @pytest.mark.asyncio
    async def test_decompose_single_result_returns_empty(
        self, enhancer, mock_llm_gateway
    ):
        """If decomposition yields only 1 sub-query, return empty (not worth it)."""
        mock_llm_gateway.complete.side_effect = [
            make_llm_response("改写1\n改写2"),  # rewrite
            make_llm_response("假设文档内容足够长以通过过滤条件。"),  # hyde
            make_llm_response("只有一个子查询"),  # decompose
        ]

        result = await enhancer.enhance("简单查询")

        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_decompose_timeout_returns_empty(
        self, enhancer, mock_llm_gateway
    ):
        """Decomposition should return empty on timeout."""
        call_count = [0]

        async def selective_slow(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:  # Third call is decompose
                await asyncio.sleep(10)
            return make_llm_response("改写1\n改写2")

        mock_llm_gateway.complete.side_effect = selective_slow

        result = await enhancer.enhance("测试查询")

        assert result.sub_queries == []


# ─── Timeout Degradation Tests ─────────────────────────────────────────


class TestTimeoutDegradation:
    """Tests for overall timeout degradation behavior."""

    @pytest.mark.asyncio
    async def test_overall_timeout_fallback(self, enhancer, mock_llm_gateway):
        """Should fall back to original query if overall timeout exceeded."""

        async def very_slow_complete(*args, **kwargs):
            await asyncio.sleep(10)  # Exceeds 5s overall timeout
            return make_llm_response("too late")

        mock_llm_gateway.complete.side_effect = very_slow_complete

        result = await enhancer.enhance("测试查询")

        # Should still have original query
        assert result.original == "测试查询"
        # Enhancements should be empty (timed out)
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_partial_success_on_individual_timeout(
        self, enhancer, mock_llm_gateway, mock_embedding_service
    ):
        """If one enhancement times out, others should still succeed."""
        call_count = [0]

        async def selective_response(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Rewrite succeeds quickly
                return make_llm_response("改写变体1\n改写变体2")
            elif call_count[0] == 2:
                # HyDE times out
                await asyncio.sleep(10)
                return make_llm_response("too late")
            else:
                # Decompose returns no decomposition
                return make_llm_response("无需分解")

        mock_llm_gateway.complete.side_effect = selective_response

        result = await enhancer.enhance("测试查询")

        # Original always present
        assert result.original == "测试查询"
        # Rewrite should succeed (first call)
        assert len(result.variants) >= 1

    @pytest.mark.asyncio
    async def test_exception_fallback(self, enhancer, mock_llm_gateway):
        """Should fall back to original query on unexpected exceptions."""
        mock_llm_gateway.complete.side_effect = RuntimeError("Unexpected error")

        result = await enhancer.enhance("测试查询")

        assert result.original == "测试查询"
        assert result.variants == []


# ─── Config Toggle Tests ───────────────────────────────────────────────


class TestConfigToggle:
    """Tests for feature enable/disable configuration."""

    @pytest.mark.asyncio
    async def test_all_disabled_returns_original_only(
        self, enhancer_all_disabled, mock_llm_gateway
    ):
        """With all features disabled, should return only original query."""
        result = await enhancer_all_disabled.enhance("测试查询")

        assert result.original == "测试查询"
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        # LLM should not be called
        mock_llm_gateway.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_rewrite_enabled(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """With only rewrite enabled, should only generate variants."""
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

        mock_llm_gateway.complete.return_value = make_llm_response(
            "改写1\n改写2\n改写3"
        )

        result = await enhancer.enhance("测试查询")

        assert result.original == "测试查询"
        assert len(result.variants) == 3
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        # LLM should be called only once (for rewrite)
        assert mock_llm_gateway.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_only_hyde_enabled(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """With only HyDE enabled, should only generate embeddings."""
        config = QueryEnhancerConfig(
            enable_rewrite=False,
            enable_hyde=True,
            enable_decomposition=False,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )

        mock_llm_gateway.complete.return_value = make_llm_response(
            "这是一个假设文档段落，描述了相关的技术内容和应用场景。"
        )

        result = await enhancer.enhance("测试查询")

        assert result.original == "测试查询"
        assert result.variants == []
        assert len(result.hyde_embeddings) >= 1
        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_only_decomposition_enabled(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """With only decomposition enabled, should only generate sub-queries."""
        config = QueryEnhancerConfig(
            enable_rewrite=False,
            enable_hyde=False,
            enable_decomposition=True,
        )
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            config=config,
        )

        mock_llm_gateway.complete.return_value = make_llm_response(
            "子查询A\n子查询B\n子查询C"
        )

        result = await enhancer.enhance("复合查询")

        assert result.original == "复合查询"
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert len(result.sub_queries) == 3

    @pytest.mark.asyncio
    async def test_config_can_be_changed(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """Config should be changeable after initialization."""
        enhancer = QueryEnhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
        )

        # Initially all enabled
        assert enhancer.config.enable_rewrite is True
        assert enhancer.config.enable_hyde is True
        assert enhancer.config.enable_decomposition is True

        # Disable all
        enhancer.config = QueryEnhancerConfig(
            enable_rewrite=False,
            enable_hyde=False,
            enable_decomposition=False,
        )

        mock_llm_gateway.complete.return_value = make_llm_response("改写")

        result = await enhancer.enhance("测试")

        assert result.variants == []
        mock_llm_gateway.complete.assert_not_called()


# ─── Parsing Helper Tests ──────────────────────────────────────────────


class TestParsingHelpers:
    """Tests for internal parsing helper methods."""

    def test_parse_variants_plain_lines(self, enhancer):
        """Should parse plain text lines as variants."""
        content = "变体一\n变体二\n变体三"
        result = enhancer._parse_variants(content, 5)

        assert result == ["变体一", "变体二", "变体三"]

    def test_parse_variants_numbered(self, enhancer):
        """Should strip numbering from variants."""
        content = "1. 第一个\n2. 第二个\n3. 第三个"
        result = enhancer._parse_variants(content, 5)

        assert "第一个" in result
        assert "第二个" in result
        assert "第三个" in result

    def test_parse_variants_chinese_numbered(self, enhancer):
        """Should strip Chinese numbering from variants."""
        content = "1、第一个\n2、第二个\n3、第三个"
        result = enhancer._parse_variants(content, 5)

        assert "第一个" in result
        assert "第二个" in result

    def test_parse_variants_bullet_points(self, enhancer):
        """Should strip bullet points from variants."""
        content = "- 变体A\n- 变体B\n* 变体C"
        result = enhancer._parse_variants(content, 5)

        assert "变体A" in result
        assert "变体B" in result
        assert "变体C" in result

    def test_parse_variants_empty_content(self, enhancer):
        """Should return empty list for empty content."""
        result = enhancer._parse_variants("", 5)
        assert result == []

    def test_parse_variants_respects_max_count(self, enhancer):
        """Should respect max_count limit."""
        content = "a1\na2\na3\na4\na5\na6"
        result = enhancer._parse_variants(content, 3)

        assert len(result) == 3

    def test_parse_variants_skips_empty_lines(self, enhancer):
        """Should skip empty lines."""
        content = "变体1\n\n\n变体2\n\n变体3"
        result = enhancer._parse_variants(content, 5)

        assert len(result) == 3

    def test_parse_hypothetical_documents_double_newline(self, enhancer):
        """Should split by double newlines."""
        content = (
            "这是第一个假设文档，描述了机器学习的基本概念和应用场景，内容足够长。\n\n"
            "这是第二个假设文档，介绍了深度学习在自然语言处理中的应用，内容也足够长。"
        )
        result = enhancer._parse_hypothetical_documents(content, 3)

        assert len(result) == 2

    def test_parse_hypothetical_documents_filters_short(self, enhancer):
        """Should filter paragraphs shorter than 20 characters."""
        content = "短\n\n这是一个足够长的段落，超过了二十个字符的最低要求。"
        result = enhancer._parse_hypothetical_documents(content, 3)

        assert len(result) == 1
        assert "足够长" in result[0]

    def test_parse_hypothetical_documents_max_count(self, enhancer):
        """Should respect max_count limit."""
        content = "\n\n".join(
            [f"这是第{i}个假设文档段落，内容足够长以通过过滤。" for i in range(10)]
        )
        result = enhancer._parse_hypothetical_documents(content, 3)

        assert len(result) <= 3

    def test_parse_hypothetical_documents_empty(self, enhancer):
        """Should return empty list for empty content."""
        result = enhancer._parse_hypothetical_documents("", 3)
        assert result == []


# ─── EnhancedQuery Dataclass Tests ─────────────────────────────────────


class TestEnhancedQueryDataclass:
    """Tests for the EnhancedQuery dataclass."""

    def test_default_values(self):
        """EnhancedQuery should have sensible defaults."""
        eq = EnhancedQuery(original="test")

        assert eq.original == "test"
        assert eq.variants == []
        assert eq.hyde_embeddings == []
        assert eq.sub_queries == []

    def test_with_all_fields(self):
        """EnhancedQuery should hold all enhancement data."""
        eq = EnhancedQuery(
            original="原始查询",
            variants=["变体1", "变体2"],
            hyde_embeddings=[[0.1] * 1024],
            sub_queries=["子查询1", "子查询2"],
        )

        assert eq.original == "原始查询"
        assert len(eq.variants) == 2
        assert len(eq.hyde_embeddings) == 1
        assert len(eq.sub_queries) == 2


# ─── QueryEnhancerConfig Tests ─────────────────────────────────────────


class TestQueryEnhancerConfig:
    """Tests for the QueryEnhancerConfig dataclass."""

    def test_default_all_enabled(self):
        """Default config should have all features enabled."""
        config = QueryEnhancerConfig()

        assert config.enable_rewrite is True
        assert config.enable_hyde is True
        assert config.enable_decomposition is True

    def test_custom_config(self):
        """Should support custom configuration."""
        config = QueryEnhancerConfig(
            enable_rewrite=False,
            enable_hyde=True,
            enable_decomposition=False,
        )

        assert config.enable_rewrite is False
        assert config.enable_hyde is True
        assert config.enable_decomposition is False
