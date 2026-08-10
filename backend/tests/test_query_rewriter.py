"""``QueryRewriter`` 单元测试（任务 15.1）。

覆盖需求 7.1 关键场景：
- LLM 正常生成 ≤ 5 个语义变体
- LLM 输出超过 5 个时被截断
- 输出包含原始 query 时被去重移除
- 空白 query 直接返回空列表（不调 LLM）
- LLM 超时（2 秒）降级
- LLM 抛异常时降级
- LLM 返回非法 JSON 时按行解析降级
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.llm_gateway import LLMGatewayError
from app.services.query_rewriter import (
    MAX_REWRITE_VARIANTS,
    REWRITE_TIMEOUT_SECONDS,
    QueryRewriter,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


def _make_response(content: str) -> MagicMock:
    """构造模拟的 ``LLMResponse`` 对象。"""
    response = MagicMock()
    response.content = content
    return response


@pytest.fixture
def mock_llm() -> AsyncMock:
    """提供带 ``complete`` 异步方法的 LLMGateway mock。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock()
    return gateway


@pytest.fixture
def rewriter(mock_llm: AsyncMock) -> QueryRewriter:
    """注入 mock 的 ``QueryRewriter`` 默认实例（2s 超时，5 变体上限）。"""
    return QueryRewriter(llm_gateway=mock_llm)


# ─── 输入校验 ──────────────────────────────────────────────────────────


class TestInputValidation:
    """测试输入校验：空字符串 / 空白不调用 LLM。"""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_without_llm_call(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """空字符串应直接返回空列表，且不触发 LLM 调用。"""
        result = await rewriter.rewrite("")

        assert result == []
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty_without_llm_call(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """仅空白字符的查询同样不应调用 LLM。"""
        result = await rewriter.rewrite("   \n\t  ")

        assert result == []
        mock_llm.complete.assert_not_called()


# ─── 正常生成路径 ──────────────────────────────────────────────────────


class TestSuccessfulRewrite:
    """测试 LLM 正常返回时的解析与裁剪。"""

    @pytest.mark.asyncio
    async def test_returns_variants_from_json_array(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 输出标准 JSON 数组时应被完整解析。"""
        mock_llm.complete.return_value = _make_response(
            '["如何使用机器学习", "机器学习的应用方法", "ML 算法实践"]'
        )

        result = await rewriter.rewrite("机器学习怎么用")

        assert len(result) == 3
        assert "如何使用机器学习" in result
        assert "机器学习的应用方法" in result
        assert "ML 算法实践" in result

    @pytest.mark.asyncio
    async def test_truncates_to_max_variants(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 返回多于 5 个变体时应被截断到 5 个。"""
        many_variants = [f"变体{i}" for i in range(10)]
        # 用 JSON 序列化保证格式合法，避免中文转义问题
        import json as _json

        mock_llm.complete.return_value = _make_response(
            _json.dumps(many_variants, ensure_ascii=False)
        )

        result = await rewriter.rewrite("测试查询")

        assert len(result) == MAX_REWRITE_VARIANTS
        assert result == [f"变体{i}" for i in range(MAX_REWRITE_VARIANTS)]

    @pytest.mark.asyncio
    async def test_dedupes_and_removes_original_query(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """与原查询相同的变体被剔除；重复项被去重；保留首次出现顺序。"""
        mock_llm.complete.return_value = _make_response(
            '["机器学习入门", "机器学习入门指南", "机器学习入门", "深度学习简介"]'
        )

        result = await rewriter.rewrite("机器学习入门")

        # 与原查询相同的 "机器学习入门" 应被移除
        assert "机器学习入门" not in result
        # 重复的 "机器学习入门指南" 只保留一次
        assert result.count("机器学习入门指南") == 1
        # 仍保留两个去重后的变体
        assert result == ["机器学习入门指南", "深度学习简介"]

    @pytest.mark.asyncio
    async def test_strips_short_and_empty_variants(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """空字符串与超短（<2 字符）的变体应被丢弃。"""
        mock_llm.complete.return_value = _make_response(
            '["a", "", "  ", "有效改写一", "有效改写二"]'
        )

        result = await rewriter.rewrite("测试查询")

        assert "有效改写一" in result
        assert "有效改写二" in result
        # "a" 长度仅 1，应被过滤
        for variant in result:
            assert len(variant) >= 2

    @pytest.mark.asyncio
    async def test_extracts_json_from_markdown_fence(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 在 JSON 数组外包裹了 markdown / 解释文本时仍能提取。"""
        mock_llm.complete.return_value = _make_response(
            "```json\n"
            '["改写一", "改写二", "改写三"]\n'
            "```\n"
            "以上是为你生成的查询变体。"
        )

        result = await rewriter.rewrite("原始查询")

        assert result == ["改写一", "改写二", "改写三"]

    @pytest.mark.asyncio
    async def test_passes_user_query_to_llm_prompt(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 调用的 prompt 中应包含原始查询，保证语义对齐。"""
        mock_llm.complete.return_value = _make_response('["变体1"]')

        await rewriter.rewrite("水泥生产工艺")

        mock_llm.complete.assert_called_once()
        kwargs = mock_llm.complete.call_args.kwargs
        assert "水泥生产工艺" in kwargs["prompt"]


# ─── 降级路径：超时 / 异常 / 非法输出 ─────────────────────────────────


class TestDegradation:
    """测试各种失败场景下的降级行为。"""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self, mock_llm: AsyncMock) -> None:
        """LLM 超过 2 秒未返回时应降级为空列表。

        通过将外层 timeout 设为很小的值（0.05s）+ 让 mock 睡更久（0.5s）
        来在测试中可靠触发 ``asyncio.TimeoutError``，避免真实等待 2s。
        """

        async def slow_complete(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            return _make_response('["不该被看到"]')

        mock_llm.complete.side_effect = slow_complete
        rewriter = QueryRewriter(llm_gateway=mock_llm, timeout=0.05)

        result = await rewriter.rewrite("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_default_timeout_is_two_seconds(self) -> None:
        """默认超时常量需符合需求 7.1（"2 秒内完成"）。"""
        assert REWRITE_TIMEOUT_SECONDS == 2.0

    @pytest.mark.asyncio
    async def test_llm_gateway_error_returns_empty(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 网关抛错（限流 / 鉴权失败等）应降级。"""
        mock_llm.complete.side_effect = LLMGatewayError(
            "rate limited", reason="rate_limit"
        )

        result = await rewriter.rewrite("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_empty(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """任意未预期的异常都应被吞掉并降级，避免影响整体搜索。"""
        mock_llm.complete.side_effect = RuntimeError("network down")

        result = await rewriter.rewrite("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_line_parsing(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 没有输出合法 JSON，但每行一个变体时应按行解析。"""
        mock_llm.complete.return_value = _make_response(
            "1. 第一个变体\n"
            "2. 第二个变体\n"
            "- 第三个变体\n"
            "* 第四个变体"
        )

        result = await rewriter.rewrite("原始查询")

        assert "第一个变体" in result
        assert "第二个变体" in result
        assert "第三个变体" in result
        assert "第四个变体" in result

    @pytest.mark.asyncio
    async def test_completely_unparseable_returns_empty(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """LLM 输出完全空白时降级为空列表。"""
        mock_llm.complete.return_value = _make_response("   \n\n  \t  ")

        result = await rewriter.rewrite("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_array_falls_back(
        self, rewriter: QueryRewriter, mock_llm: AsyncMock
    ) -> None:
        """JSON 数组语法错误时退化为按行解析，仍尽量返回有效内容。"""
        # 缺少右括号的非法 JSON；按行解析应能提取出"变体一"和"变体二"
        mock_llm.complete.return_value = _make_response(
            '"变体一"\n"变体二"'
        )

        result = await rewriter.rewrite("测试查询")

        assert "变体一" in result
        assert "变体二" in result
