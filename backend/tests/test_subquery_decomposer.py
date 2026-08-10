"""``SubqueryDecomposer`` 单元测试（任务 15.3）。

覆盖需求 7.3 关键场景：
- 多子问题查询返回多个子查询
- 单一问题查询返回空列表（LLM 输出 ``[]``）
- LLM 输出超过 5 个时被截断
- 空白 query 直接返回空列表（不调 LLM）
- LLM 超时（2 秒）降级
- LLM 抛异常时降级
- LLM 返回非法 JSON 时按行解析降级
- 仅返回 1 条子查询视为单一问题（< 2 个子问题）
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.llm_gateway import LLMGatewayError
from app.services.subquery_decomposer import (
    DECOMPOSE_TIMEOUT_SECONDS,
    MAX_SUB_QUERIES,
    SubqueryDecomposer,
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
def decomposer(mock_llm: AsyncMock) -> SubqueryDecomposer:
    """注入 mock 的 ``SubqueryDecomposer`` 默认实例（2s 超时，5 子查询上限）。"""
    return SubqueryDecomposer(llm_gateway=mock_llm)


# ─── 输入校验 ──────────────────────────────────────────────────────────


class TestInputValidation:
    """测试输入校验：空字符串 / 空白不调用 LLM。"""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_without_llm_call(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """空字符串应直接返回空列表，且不触发 LLM 调用。"""
        result = await decomposer.decompose("")

        assert result == []
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty_without_llm_call(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """仅空白字符的查询同样不应调用 LLM。"""
        result = await decomposer.decompose("   \n\t  ")

        assert result == []
        mock_llm.complete.assert_not_called()


# ─── 多子问题分解 ───────────────────────────────────────────────────────


class TestMultiQueryDecomposition:
    """测试 LLM 正常返回多子查询时的解析与裁剪。"""

    @pytest.mark.asyncio
    async def test_decomposes_multi_question_query(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """包含「和」连接词的多子问题查询应被分解为多个子查询。"""
        mock_llm.complete.return_value = _make_response(
            '["Python 的特点", "Java 的特点", "两者的区别"]'
        )

        result = await decomposer.decompose("Python 和 Java 的区别是什么")

        assert len(result) == 3
        assert "Python 的特点" in result
        assert "Java 的特点" in result
        assert "两者的区别" in result

    @pytest.mark.asyncio
    async def test_truncates_to_max_sub_queries(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 返回多于 5 个子查询时应被截断到 5 个。"""
        many_sub_queries = [f"子查询{i}" for i in range(10)]
        mock_llm.complete.return_value = _make_response(
            json.dumps(many_sub_queries, ensure_ascii=False)
        )

        result = await decomposer.decompose("一个包含许多子问题的复杂查询")

        assert len(result) == MAX_SUB_QUERIES
        assert result == [f"子查询{i}" for i in range(MAX_SUB_QUERIES)]

    @pytest.mark.asyncio
    async def test_dedupes_and_removes_original_query(
        self,
        decomposer: SubqueryDecomposer,
        mock_llm: AsyncMock,
    ) -> None:
        """与原查询相同的子查询被剔除；重复项被去重；保留首次出现顺序。"""
        mock_llm.complete.return_value = _make_response(
            '["问题甲", "问题乙", "问题甲", "原始查询", "问题丙"]'
        )

        result = await decomposer.decompose("原始查询")

        # 与原查询相同的项应被移除
        assert "原始查询" not in result
        # 重复的 "问题甲" 只保留一次
        assert result.count("问题甲") == 1
        # 仍保留三个去重后的子查询，按首次出现顺序
        assert result == ["问题甲", "问题乙", "问题丙"]

    @pytest.mark.asyncio
    async def test_strips_short_and_empty_sub_queries(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """空字符串与超短（<2 字符）的子查询应被丢弃。"""
        mock_llm.complete.return_value = _make_response(
            '["a", "", "  ", "有效子查询一", "有效子查询二"]'
        )

        result = await decomposer.decompose("一个含多个独立子问题的查询")

        assert "有效子查询一" in result
        assert "有效子查询二" in result
        # "a" 长度仅 1，应被过滤
        for sub_query in result:
            assert len(sub_query) >= 2

    @pytest.mark.asyncio
    async def test_extracts_json_from_markdown_fence(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 在 JSON 数组外包裹了 markdown / 解释文本时仍能提取。"""
        mock_llm.complete.return_value = _make_response(
            "```json\n"
            '["子查询一", "子查询二", "子查询三"]\n'
            "```\n"
            "以上是分解后的子查询。"
        )

        result = await decomposer.decompose(
            "请同时介绍 A、B 和 C 三个主题"
        )

        assert result == ["子查询一", "子查询二", "子查询三"]

    @pytest.mark.asyncio
    async def test_passes_user_query_to_llm_prompt(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 调用的 prompt 中应包含原始查询，保证语义对齐。"""
        mock_llm.complete.return_value = _make_response(
            '["水泥的成分", "水泥的强度等级"]'
        )

        await decomposer.decompose("水泥的成分和强度等级分别是什么")

        mock_llm.complete.assert_called_once()
        kwargs = mock_llm.complete.call_args.kwargs
        assert "水泥的成分和强度等级分别是什么" in kwargs["prompt"]


# ─── 单一问题（不分解）─────────────────────────────────────────────────


class TestSingleQueryNoDecomposition:
    """测试单一问题查询的空数组返回。"""

    @pytest.mark.asyncio
    async def test_single_question_returns_empty_list(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 判断为单一问题时输出 ``[]``，本组件原样返回 ``[]``。"""
        mock_llm.complete.return_value = _make_response("[]")

        result = await decomposer.decompose("机器学习入门")

        assert result == []

    @pytest.mark.asyncio
    async def test_single_sub_query_treated_as_single_question(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 仅返回 1 条子查询时，按需求 7.3「2 个及以上」原则视为单一问题。"""
        mock_llm.complete.return_value = _make_response('["唯一子查询"]')

        result = await decomposer.decompose("某个查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_only_original_query_in_response_returns_empty(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 仅返回原查询本身时，去重后等价于单一问题，应返回空列表。"""
        mock_llm.complete.return_value = _make_response('["原始查询"]')

        result = await decomposer.decompose("原始查询")

        assert result == []


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
        decomposer = SubqueryDecomposer(llm_gateway=mock_llm, timeout=0.05)

        result = await decomposer.decompose(
            "一个包含 A 和 B 两个独立子问题的查询"
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_default_timeout_is_two_seconds(self) -> None:
        """默认超时常量需符合「与改写一致的 2 秒预算」。"""
        assert DECOMPOSE_TIMEOUT_SECONDS == 2.0

    @pytest.mark.asyncio
    async def test_llm_gateway_error_returns_empty(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 网关抛错（限流 / 鉴权失败等）应降级。"""
        mock_llm.complete.side_effect = LLMGatewayError(
            "rate limited", reason="rate_limit"
        )

        result = await decomposer.decompose("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_empty(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """任意未预期的异常都应被吞掉并降级，避免影响整体搜索。"""
        mock_llm.complete.side_effect = RuntimeError("network down")

        result = await decomposer.decompose("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_line_parsing(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 没有输出合法 JSON，但每行一个子查询时应按行解析。"""
        mock_llm.complete.return_value = _make_response(
            "1. 第一个子查询\n"
            "2. 第二个子查询\n"
            "- 第三个子查询\n"
            "* 第四个子查询"
        )

        result = await decomposer.decompose(
            "一个含多个独立子问题的复杂查询"
        )

        assert "第一个子查询" in result
        assert "第二个子查询" in result
        assert "第三个子查询" in result
        assert "第四个子查询" in result

    @pytest.mark.asyncio
    async def test_completely_unparseable_returns_empty(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """LLM 输出完全空白时降级为空列表。"""
        mock_llm.complete.return_value = _make_response("   \n\n  \t  ")

        result = await decomposer.decompose("测试查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_array_falls_back(
        self, decomposer: SubqueryDecomposer, mock_llm: AsyncMock
    ) -> None:
        """JSON 数组语法错误时退化为按行解析，仍尽量返回有效内容。"""
        # 缺少右括号的非法 JSON；按行解析应能提取出 子查询一 / 子查询二
        mock_llm.complete.return_value = _make_response(
            '"子查询一"\n"子查询二"'
        )

        result = await decomposer.decompose("测试查询")

        assert "子查询一" in result
        assert "子查询二" in result
