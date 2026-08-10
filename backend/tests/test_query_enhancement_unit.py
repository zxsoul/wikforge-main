"""任务 15.7：查询增强综合单元测试（收口）。

本文件作为「查询增强」整体单元测试的收口，承担以下职责：

1. 把 ``QueryRewriter`` / ``HyDEService`` / ``SubqueryDecomposer`` 三个独立
   组件的综合协同与边界统一覆盖（避免被 15.1/15.2/15.3 三份单独的测试文件
   错过的"组合场景"漏网）；
2. 验证 ``QueryEnhancer.enhance()`` 在 **配置全开** 时返回的 ``EnhancedQuery``
   结构同时包含三类结果（变体 / HyDE 向量 / 子查询）以及合集首项为原始查询；
3. 验证 **空查询 / 极长查询 / 特殊字符查询** 三类边界输入下的鲁棒性；
4. 把"三类独立组件的输入校验、降级、超时"用同一组参数化生成器统一回归一遍，
   确保它们在维护过程中不会在某一类边界上单独退化。

定位说明：

- 既有 ``test_query_enhancer.py`` 覆盖 ``QueryEnhancer`` 的特性级单元；
- 既有 ``test_query_rewriter.py`` / ``test_hyde_service.py`` /
  ``test_subquery_decomposer.py`` 各自覆盖独立组件；
- 既有 ``test_original_query_preservation.py`` / ``test_query_enhancement_timeout.py`` /
  ``test_query_enhancement_config.py`` 覆盖 15.4 / 15.5 / 15.6 任务级要求；
- 本文件**只补缺**：综合协同、配置全开结构验证、边界输入鲁棒性、跨组件一致性。

关联需求：第 7 章「查询增强」整体（7.1 ~ 7.5），任务 15.1 ~ 15.6。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.embedding_service import EmbeddingResult
from app.services.hyde_service import HyDEService
from app.services.llm_gateway import LLMGatewayError
from app.services.query_enhancer import (
    EnhancedQuery,
    QueryEnhancer,
    QueryEnhancerConfig,
    build_query_enhancer,
)
from app.services.query_rewriter import QueryRewriter
from app.services.subquery_decomposer import SubqueryDecomposer


# ─── 共用工具 ──────────────────────────────────────────────────────────


def _make_response(content: str) -> MagicMock:
    """构造模拟的 ``LLMResponse``。"""
    response = MagicMock()
    response.content = content
    return response


def _make_embedding(vector: list[float] | None = None) -> EmbeddingResult:
    """构造模拟的 ``EmbeddingResult``，默认返回 1024 维占位向量。"""
    return EmbeddingResult(
        chunk_id="query",
        dense_vector=vector if vector is not None else [0.1] * 1024,
        sparse_indices=[],
        sparse_values=[],
    )


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm() -> AsyncMock:
    """LLM 网关 mock。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock()
    return gateway


@pytest.fixture
def mock_embedding() -> AsyncMock:
    """Embedding 服务 mock，每次 embed_query 返回 1024 维向量。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(return_value=_make_embedding())
    return service


@pytest.fixture
def enhancer_all_enabled(mock_llm, mock_embedding) -> QueryEnhancer:
    """三项开关全开的 ``QueryEnhancer``。"""
    return QueryEnhancer(
        llm_gateway=mock_llm,
        embedding_service=mock_embedding,
        config=QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=True,
            enable_decomposition=True,
        ),
    )


# ─── 1. 配置全开：综合返回结构完整 ───────────────────────────────────


class TestEnhanceFullyEnabledShape:
    """配置全开时 ``EnhancedQuery`` 同时包含三类结果且符合契约。"""

    @pytest.mark.asyncio
    async def test_full_enhancement_contains_all_three_categories(
        self, enhancer_all_enabled, mock_llm, mock_embedding
    ):
        """三类结果都非空 + ``all_text_queries`` 首项为原始查询 + 不重复。

        ``QueryEnhancer._enhance_internal`` 并发调度 rewrite / hyde / decompose，
        三个 ``asyncio.create_task`` 间的调度顺序在不同事件循环实现下不保证稳定，
        因此用一个共享 ``side_effect`` 函数按 prompt 内容分发响应，避免依赖
        调用顺序导致测试 flaky。
        """

        async def respond_by_prompt(*_args, **kwargs):
            prompt = kwargs.get("prompt", "")
            system = kwargs.get("system_prompt", "")
            # ``QueryEnhancer`` 内置三段 system_prompt 关键字可识别：
            # - rewrite: "搜索查询改写助手"
            # - hyde: "文档生成助手"
            # - decompose: "查询分析助手"
            if "查询分析助手" in system or "无需分解" in prompt:
                return _make_response(
                    "企业知识库的检索流程\n企业知识库的索引结构"
                )
            if "文档生成助手" in system:
                return _make_response(
                    "这是一段足够长的假设文档段落，描述了企业知识库的检索流程。"
                )
            # 默认按 rewrite 返回
            return _make_response(
                "如何检索企业知识库\n企业知识库检索方法\n企业搜索流程"
            )

        mock_llm.complete.side_effect = respond_by_prompt

        result = await enhancer_all_enabled.enhance("企业知识库的检索流程是什么")

        # 1. 类型与原始查询保留
        assert isinstance(result, EnhancedQuery)
        assert result.original == "企业知识库的检索流程是什么"
        assert result.original_query == result.original  # 别名一致
        # 2. 三类结果同时存在
        assert len(result.variants) >= 1, "改写变体不应为空"
        assert len(result.hyde_embeddings) >= 1, "HyDE 向量不应为空"
        assert len(result.sub_queries) >= 2, "子查询应至少 2 个"
        # 3. 别名属性
        assert result.rewrites is result.variants
        assert result.hypothetical_embeddings is result.hyde_embeddings
        # 4. 文本合集：首项原始查询，且无重复
        assert result.all_text_queries[0] == result.original
        assert len(result.all_text_queries) == len(set(result.all_text_queries))
        # 5. 文本合集严格大于 1（不是降级路径）
        assert len(result.all_text_queries) > 1
        # 6. embed_query 至少被调用一次（HyDE 路径）
        assert mock_embedding.embed_query.await_count >= 1

    @pytest.mark.asyncio
    async def test_full_enhancement_text_collection_includes_rewrites_and_subqueries(
        self, enhancer_all_enabled, mock_llm
    ):
        """``all_text_queries`` 应同时包含改写变体与子查询，按首次出现顺序。"""

        async def respond_by_prompt(*_args, **kwargs):
            prompt = kwargs.get("prompt", "")
            system = kwargs.get("system_prompt", "")
            if "查询分析助手" in system:
                return _make_response("子查询甲\n子查询乙")
            if "文档生成助手" in system:
                # 让 HyDE 的 LLM 返回空，便于隔离 HyDE 不影响文本合集断言
                return _make_response("")
            assert "改写助手" in system or "改写" in prompt
            return _make_response("改写A\n改写B")

        mock_llm.complete.side_effect = respond_by_prompt

        result = await enhancer_all_enabled.enhance("综合查询")

        # 改写、子查询都进合集；HyDE 不参与文本合集
        assert "改写A" in result.all_text_queries
        assert "改写B" in result.all_text_queries
        assert "子查询甲" in result.all_text_queries
        assert "子查询乙" in result.all_text_queries
        # 首项原始查询
        assert result.all_text_queries[0] == "综合查询"


# ─── 2. 边界输入：空 / 极长 / 特殊字符 ────────────────────────────────


class TestBoundaryInputs:
    """对 ``QueryEnhancer.enhance()`` 的边界输入鲁棒性。"""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_collection_no_calls(
        self, enhancer_all_enabled, mock_llm, mock_embedding
    ):
        """空查询：原始保留为空字符串，``all_text_queries`` 也为空。

        关键不变量：空查询时**不应**强行注入空字符串到 ``all_text_queries``，
        避免下游误以为存在一条"空文本查询"。LLM / Embedding 都不应被调用。
        """
        result = await enhancer_all_enabled.enhance("")

        assert result.original == ""
        assert result.all_text_queries == []
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        mock_llm.complete.assert_not_called()
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_query_is_treated_as_empty(
        self, enhancer_all_enabled, mock_llm, mock_embedding
    ):
        """仅空白字符的查询同样被视为空查询，不调用任何外部依赖。"""
        result = await enhancer_all_enabled.enhance("   \n\t  ")

        assert result.all_text_queries == []
        mock_llm.complete.assert_not_called()
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_very_long_query_does_not_crash(
        self, enhancer_all_enabled, mock_llm
    ):
        """极长查询（> 5000 字符）不应导致 enhance 崩溃，原始查询应原样保留。"""
        long_query = "深度学习" * 2000  # 4 字符 × 2000 = 8000 字符

        # 让 LLM 返回固定内容，避免为长 prompt 编造响应
        mock_llm.complete.return_value = _make_response("[]")

        result = await enhancer_all_enabled.enhance(long_query)

        # 原始查询完整保留（不截断）
        assert result.original == long_query
        assert len(result.original) == 8000
        # 文本合集首项是完整原始查询
        assert result.all_text_queries[0] == long_query
        # 不论 LLM 返回什么，原始查询都至少出现在合集中
        assert long_query in result.all_text_queries

    @pytest.mark.asyncio
    async def test_long_query_is_passed_to_llm_unchanged(
        self, enhancer_all_enabled, mock_llm
    ):
        """长查询应原样作为 prompt 传给 LLM（由 LLM 网关层负责截断/限流）。"""
        long_query = "X" * 1500

        mock_llm.complete.return_value = _make_response("[]")
        await enhancer_all_enabled.enhance(long_query)

        # 至少有一次 LLM 调用，且其 prompt 包含完整长查询
        assert mock_llm.complete.await_count >= 1
        for call in mock_llm.complete.await_args_list:
            assert long_query in call.kwargs["prompt"]

    @pytest.mark.parametrize(
        "special_query",
        [
            "C++ 11/14/17 新特性",
            "🔥 ChatGPT 与 🦙 Llama 的对比",
            "<script>alert('xss')</script>",
            "SELECT * FROM users WHERE 1=1",
            'query with "double quotes" and \'single quotes\'',
            "包含\n换行符\t制表符的查询",
            "包含\\转义字符\\的查询",
            "中英 mixed 查询 with 数字 123 和符号 !@#$%^&*()",
            "重复字符aaaaaa 重复字符 bbbbbbb",
        ],
    )
    @pytest.mark.asyncio
    async def test_special_characters_preserved_and_passed_through(
        self, enhancer_all_enabled, mock_llm, special_query: str
    ):
        """各种特殊字符查询都能被原样保留，不会引发异常。

        包括但不限于：HTML 标签、SQL 片段、emoji、引号、换行/制表符、转义符、
        中英混排、特殊符号等。这些都是真实搜索框可能输入的内容，``QueryEnhancer``
        必须保持透明传递，不做任何过滤或转义（过滤是 SearchService 职责）。
        """
        mock_llm.complete.return_value = _make_response("[]")

        result = await enhancer_all_enabled.enhance(special_query)

        # 原始查询逐字符保留
        assert result.original == special_query
        # 合集首项是原始查询（关键不变量）
        assert result.all_text_queries[0] == special_query


# ─── 3. 三独立组件输入校验一致性 ───────────────────────────────────────


class TestIndependentComponentsInputValidation:
    """三独立组件在边界输入下行为一致：均返回空 + 不调外部依赖。"""

    @pytest.fixture
    def rewriter(self, mock_llm) -> QueryRewriter:
        return QueryRewriter(llm_gateway=mock_llm)

    @pytest.fixture
    def hyde(self, mock_llm, mock_embedding) -> HyDEService:
        return HyDEService(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
        )

    @pytest.fixture
    def decomposer(self, mock_llm) -> SubqueryDecomposer:
        return SubqueryDecomposer(llm_gateway=mock_llm)

    @pytest.mark.parametrize(
        "blank_input",
        ["", "   ", "\n", "\t\t", "  \n\t  "],
    )
    @pytest.mark.asyncio
    async def test_all_components_short_circuit_on_blank_input(
        self,
        rewriter: QueryRewriter,
        hyde: HyDEService,
        decomposer: SubqueryDecomposer,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
        blank_input: str,
    ):
        """三独立组件遇到空白输入时统一返回空且不触发外部调用。"""
        rewrites = await rewriter.rewrite(blank_input)
        vectors = await hyde.generate_hypothetical_embeddings(blank_input)
        sub_queries = await decomposer.decompose(blank_input)

        assert rewrites == []
        assert vectors == []
        assert sub_queries == []
        mock_llm.complete.assert_not_called()
        mock_embedding.embed_query.assert_not_called()


# ─── 4. 三独立组件降级一致性：LLM 异常一律返回空 ─────────────────────


class TestIndependentComponentsDegradation:
    """三独立组件遇到 LLM 错误 / 未知异常时降级一致性。"""

    @pytest.fixture
    def rewriter(self, mock_llm) -> QueryRewriter:
        return QueryRewriter(llm_gateway=mock_llm)

    @pytest.fixture
    def hyde(self, mock_llm, mock_embedding) -> HyDEService:
        return HyDEService(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
        )

    @pytest.fixture
    def decomposer(self, mock_llm) -> SubqueryDecomposer:
        return SubqueryDecomposer(llm_gateway=mock_llm)

    @pytest.mark.parametrize(
        "exc",
        [
            LLMGatewayError("rate limited", reason="rate_limit"),
            LLMGatewayError("auth failed", reason="auth"),
            RuntimeError("network down"),
            ValueError("unexpected payload"),
        ],
    )
    @pytest.mark.asyncio
    async def test_all_components_degrade_on_llm_exception(
        self,
        rewriter: QueryRewriter,
        hyde: HyDEService,
        decomposer: SubqueryDecomposer,
        mock_llm: AsyncMock,
        exc: Exception,
    ):
        """LLM 抛各类异常时，三独立组件都降级为空列表，不向上抛。"""
        mock_llm.complete.side_effect = exc

        rewrites = await rewriter.rewrite("机器学习是什么")
        vectors = await hyde.generate_hypothetical_embeddings("机器学习是什么")
        sub_queries = await decomposer.decompose(
            "Python 和 Java 的区别是什么"
        )

        assert rewrites == []
        assert vectors == []
        assert sub_queries == []


# ─── 5. 三独立组件超时一致性：均 ≤ 各自超时常量 ─────────────────────


class TestIndependentComponentsTimeout:
    """三独立组件超时降级行为一致：均返回 ``[]``，不抛 TimeoutError。"""

    @pytest.mark.asyncio
    async def test_rewriter_timeout_returns_empty(self, mock_llm):
        """``QueryRewriter`` 超时降级。"""

        async def slow(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            return _make_response('["不该被看到"]')

        mock_llm.complete.side_effect = slow
        rewriter = QueryRewriter(llm_gateway=mock_llm, timeout=0.05)

        result = await rewriter.rewrite("查询")

        assert result == []

    @pytest.mark.asyncio
    async def test_hyde_timeout_returns_empty(self, mock_llm, mock_embedding):
        """``HyDEService`` 超时降级，且 embedding 不应被调用。"""

        async def slow(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            return _make_response('["不该被看到"]')

        mock_llm.complete.side_effect = slow
        hyde = HyDEService(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
            timeout=0.05,
        )

        result = await hyde.generate_hypothetical_embeddings("查询")

        assert result == []
        mock_embedding.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_decomposer_timeout_returns_empty(self, mock_llm):
        """``SubqueryDecomposer`` 超时降级。"""

        async def slow(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            return _make_response('["不该被看到"]')

        mock_llm.complete.side_effect = slow
        decomposer = SubqueryDecomposer(llm_gateway=mock_llm, timeout=0.05)

        result = await decomposer.decompose("一个含 A 和 B 的查询")

        assert result == []


# ─── 6. 三独立组件与 QueryEnhancer 的对外约束一致 ─────────────────────


class TestSharedConstraints:
    """三独立组件对外约束的一致性：上限、不返回原始查询、去重等。"""

    @pytest.mark.asyncio
    async def test_rewriter_never_returns_original(self, mock_llm):
        """``QueryRewriter`` 输出永远不包含原始查询字面。"""
        mock_llm.complete.return_value = _make_response(
            '["机器学习", "ML 学习路径", "如何学 ML"]'
        )
        rewriter = QueryRewriter(llm_gateway=mock_llm)

        result = await rewriter.rewrite("机器学习")

        assert "机器学习" not in result

    @pytest.mark.asyncio
    async def test_decomposer_never_returns_original(self, mock_llm):
        """``SubqueryDecomposer`` 输出永远不包含原始查询字面。"""
        mock_llm.complete.return_value = _make_response(
            '["原始查询", "另一个子查询", "再一个子查询"]'
        )
        decomposer = SubqueryDecomposer(llm_gateway=mock_llm)

        result = await decomposer.decompose("原始查询")

        assert "原始查询" not in result

    @pytest.mark.asyncio
    async def test_rewriter_respects_max_5_variants(self, mock_llm):
        """``QueryRewriter`` 输出严格 ≤ 5 条，与需求 7.1 一致。"""
        import json as _json

        mock_llm.complete.return_value = _make_response(
            _json.dumps([f"变体{i}" for i in range(20)], ensure_ascii=False)
        )
        rewriter = QueryRewriter(llm_gateway=mock_llm)

        result = await rewriter.rewrite("某查询")

        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_hyde_respects_max_3_documents(
        self, mock_llm, mock_embedding
    ):
        """``HyDEService`` 输出严格 ≤ 3 条向量，与需求 7.2 一致。"""
        import json as _json

        mock_llm.complete.return_value = _make_response(
            _json.dumps(
                [
                    f"这是第{i}段足够长的假设文档段落，用于测试上限。" * 2
                    for i in range(20)
                ],
                ensure_ascii=False,
            )
        )
        hyde = HyDEService(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
        )

        result = await hyde.generate_hypothetical_embeddings("某查询")

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_decomposer_respects_max_5_subqueries(self, mock_llm):
        """``SubqueryDecomposer`` 输出严格 ≤ 5 条，与需求 7.3 一致。"""
        import json as _json

        mock_llm.complete.return_value = _make_response(
            _json.dumps([f"子查询{i}" for i in range(20)], ensure_ascii=False)
        )
        decomposer = SubqueryDecomposer(llm_gateway=mock_llm)

        result = await decomposer.decompose("一个含很多子问题的复杂查询")

        assert len(result) <= 5


# ─── 7. 配置开关 + 工厂：组合矩阵 ─────────────────────────────────────


class TestConfigToggleMatrix:
    """通过 ``build_query_enhancer`` 工厂以"开关组合矩阵"覆盖运行时入口。"""

    @pytest.mark.parametrize(
        "rewrite, hyde, decompose, expected_has_variants, "
        "expected_has_hyde, expected_has_subs",
        [
            # 仅 rewrite
            (True, False, False, True, False, False),
            # 仅 hyde
            (False, True, False, False, True, False),
            # 仅 decompose
            (False, False, True, False, False, True),
            # rewrite + decompose（不调用 embedding）
            (True, False, True, True, False, True),
            # 全开
            (True, True, True, True, True, True),
            # 全关：仅原始查询
            (False, False, False, False, False, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_enhance_respects_each_toggle_combination(
        self,
        mock_llm: AsyncMock,
        mock_embedding: AsyncMock,
        rewrite: bool,
        hyde: bool,
        decompose: bool,
        expected_has_variants: bool,
        expected_has_hyde: bool,
        expected_has_subs: bool,
    ):
        """工厂构造的 ``QueryEnhancer.enhance()`` 严格按开关组合产出结果。

        本测试用一个能识别 system_prompt 的 mock，既能覆盖每个开关的"开"路径，
        也能验证"关"路径下对应能力**不会**触发 LLM / Embedding 调用。
        """

        async def respond_by_prompt(*_args, **kwargs):
            system = kwargs.get("system_prompt", "")
            if "查询分析助手" in system:
                # decompose 至少 2 条才会被认为是多子问题
                return _make_response("子查询1\n子查询2")
            if "文档生成助手" in system:
                return _make_response(
                    "这是一段足够长的假设文档段落，用于测试 HyDE 路径。"
                )
            return _make_response("改写A\n改写B")

        mock_llm.complete.side_effect = respond_by_prompt

        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=rewrite,
            QUERY_ENHANCEMENT_ENABLE_HYDE=hyde,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=decompose,
        )
        enhancer = build_query_enhancer(
            llm_gateway=mock_llm,
            embedding_service=mock_embedding,
            settings=fake_settings,
        )

        result = await enhancer.enhance("综合查询")

        # 原始查询保留
        assert result.original == "综合查询"
        assert result.all_text_queries[0] == "综合查询"
        # 各分项严格按开关产出
        assert bool(result.variants) is expected_has_variants
        assert bool(result.hyde_embeddings) is expected_has_hyde
        assert bool(result.sub_queries) is expected_has_subs
        # 全关时不调任何外部依赖
        if not (rewrite or hyde or decompose):
            mock_llm.complete.assert_not_called()
            mock_embedding.embed_query.assert_not_called()
        # HyDE 关闭意味着 embed_query 不被调用
        if not hyde:
            mock_embedding.embed_query.assert_not_called()


# ─── 8. ``EnhancedQuery`` 数据契约综合验证 ────────────────────────────


class TestEnhancedQueryContract:
    """``EnhancedQuery`` 数据契约综合验证，保证别名与不变量稳定。"""

    def test_aliases_share_underlying_objects(self):
        """``original_query`` / ``rewrites`` / ``hypothetical_embeddings``
        三个别名属性应分别返回与原字段相同的对象，以便上层无论使用哪套
        命名都能拿到同一份数据。"""
        eq = EnhancedQuery(
            original="原始",
            variants=["变体1", "变体2"],
            hyde_embeddings=[[0.1] * 1024],
            sub_queries=["子查询1"],
            all_text_queries=["原始", "变体1", "变体2", "子查询1"],
        )

        assert eq.original_query == eq.original
        assert eq.rewrites is eq.variants
        assert eq.hypothetical_embeddings is eq.hyde_embeddings

    def test_default_all_text_queries_is_empty_list(self):
        """构造时未传 ``all_text_queries`` 默认为空列表，由 enhance() 负责填充。"""
        eq = EnhancedQuery(original="X")

        assert eq.all_text_queries == []
        assert eq.variants == []
        assert eq.hyde_embeddings == []
        assert eq.sub_queries == []

    def test_build_all_text_queries_handles_typical_inputs(self):
        """``_build_all_text_queries`` 静态方法对典型输入产出稳定。"""
        result = QueryEnhancer._build_all_text_queries(
            original="hello",
            rewrites=["hi", "greet"],
            sub_queries=["hello", "say hi"],
        )

        # 原始查询置首；与 sub_queries 中的 "hello" 重复被去重
        assert result == ["hello", "hi", "greet", "say hi"]
        assert result.count("hello") == 1
