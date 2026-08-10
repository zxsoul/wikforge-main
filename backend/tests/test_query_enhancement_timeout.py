"""任务 15.5 — 查询增强 5 秒超时降级专项测试。

需求 7.5：「IF 查询增强过程（改写、HyDE 生成或子查询分解）在 5 秒内未完成或发生错误，
THEN THE Search_Engine SHALL 回退至使用用户原始查询直接执行检索，并向用户返回检索结果
而不显示错误信息」。

本文件聚焦 ``QueryEnhancer.enhance()`` 的整体 5 秒超时降级语义，与
``test_query_enhancer.py`` 中通用的特性测试互补：

- 整体超时（≥ 5s）触发 → 仅返回 ``[original]``，不抛错
- 5s 内完成 → 返回完整增强结果（多于 1 个文本查询）
- 部分子模块完成 + 部分子模块拖慢到整体超时 → 触发整体超时，仅原始查询
- 子模块各自的子超时不会拖累其他子模块（rewrite 完成 + HyDE 子超时 → rewrite 保留）
- ``all_text_queries[0] == 原始查询`` 始终成立
- 超时分支日志中带 ``event=query_enhancement_timeout`` 关键字（运维可观测）

为保证测试快速可靠，本文件通过 ``monkeypatch`` 把模块级超时常量缩小到亚秒级，
而不是真的等 5 秒；这样既能验证降级路径，又不会拖慢 CI。
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services import query_enhancer as qe_mod
from app.services.query_enhancer import (
    EnhancedQuery,
    QueryEnhancer,
    QueryEnhancerConfig,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fast_timeouts(monkeypatch):
    """将查询增强器的所有超时常量缩小到亚秒级，加速测试。

    模拟生产配置（整体 5s 大于子超时 2/3/2s）的"子超时优先触发"场景：

    - OVERALL_TIMEOUT: 0.5s（模拟 5s）
    - REWRITE_TIMEOUT: 0.2s（模拟 2s）
    - HYDE_TIMEOUT:    0.3s（模拟 3s）
    - DECOMPOSE_TIMEOUT: 0.2s（模拟 2s）

    这些常量在 ``enhance()`` 与子方法中均通过模块属性查找读取，所以
    ``monkeypatch.setattr`` 在测试间生效后再恢复。
    """
    monkeypatch.setattr(qe_mod, "OVERALL_TIMEOUT", 0.5)
    monkeypatch.setattr(qe_mod, "REWRITE_TIMEOUT", 0.2)
    monkeypatch.setattr(qe_mod, "HYDE_TIMEOUT", 0.3)
    monkeypatch.setattr(qe_mod, "DECOMPOSE_TIMEOUT", 0.2)
    return {
        "overall": 0.5,
        "rewrite": 0.2,
        "hyde": 0.3,
        "decompose": 0.2,
    }


@pytest.fixture
def overall_first_timeouts(monkeypatch):
    """让"整体超时优先于子超时"的病态配置，专门测试整体降级兜底分支。

    生产中整体 5s > 子超时（2/3/2s），子模块通常会先在自己的 wait_for 里返回 []，
    整体超时分支只在"系统调度延迟 / Embedding 慢调用累积"等极端情况触发。
    本 fixture 把 overall 设小、子超时设大，模拟该极端情况，让 LLM mock 的
    sleep 时长落在 ``OVERALL_TIMEOUT < sleep < 子超时`` 区间，从而强制走整体降级。

    - OVERALL_TIMEOUT: 0.2s
    - REWRITE_TIMEOUT / HYDE_TIMEOUT / DECOMPOSE_TIMEOUT: 1.5s
    """
    monkeypatch.setattr(qe_mod, "OVERALL_TIMEOUT", 0.2)
    monkeypatch.setattr(qe_mod, "REWRITE_TIMEOUT", 1.5)
    monkeypatch.setattr(qe_mod, "HYDE_TIMEOUT", 1.5)
    monkeypatch.setattr(qe_mod, "DECOMPOSE_TIMEOUT", 1.5)
    return {
        "overall": 0.2,
        "rewrite": 1.5,
        "hyde": 1.5,
        "decompose": 1.5,
    }


@pytest.fixture
def overall_first_enhancer(
    overall_first_timeouts, mock_llm_gateway, mock_embedding_service
):
    """搭配 ``overall_first_timeouts`` 的增强器实例。"""
    return QueryEnhancer(
        llm_gateway=mock_llm_gateway,
        embedding_service=mock_embedding_service,
        config=QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=True,
            enable_decomposition=True,
        ),
    )


@pytest.fixture
def mock_llm_gateway():
    """LLM 网关 mock。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock()
    return gateway


@pytest.fixture
def mock_embedding_service():
    """Embedding 服务 mock。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(
        return_value=MagicMock(
            dense_vector=[0.1] * 1024,
            sparse_indices=[1, 5, 10],
            sparse_values=[0.5, 0.3, 0.8],
        )
    )
    return service


@pytest.fixture
def enhancer(fast_timeouts, mock_llm_gateway, mock_embedding_service):
    """带快速超时常量与 mock 依赖的 QueryEnhancer。"""
    return QueryEnhancer(
        llm_gateway=mock_llm_gateway,
        embedding_service=mock_embedding_service,
        config=QueryEnhancerConfig(
            enable_rewrite=True,
            enable_hyde=True,
            enable_decomposition=True,
        ),
    )


def _make_response(content: str):
    """构建一个具有 ``content`` 属性的 LLM 响应对象。"""
    response = MagicMock()
    response.content = content
    return response


# ─── 1. 整体超时触发降级 ─────────────────────────────────────────────


class TestOverallTimeoutFallback:
    """整体 5s（测试缩短为 0.2s）超时未完成 → 仅返回原始查询。"""

    @pytest.mark.asyncio
    async def test_overall_timeout_returns_original_only(
        self, overall_first_enhancer, mock_llm_gateway, overall_first_timeouts
    ):
        """所有子模块均拖慢到超过整体超时 → 仅 ``[original]``。"""

        async def hang(*_args, **_kwargs):
            # 介于整体超时与子超时之间：触发整体 wait_for 的 TimeoutError
            await asyncio.sleep(overall_first_timeouts["overall"] * 3)
            return _make_response("应不会被使用")

        mock_llm_gateway.complete.side_effect = hang

        result = await overall_first_enhancer.enhance("企业知识库的检索流程是什么")

        # 原始查询保留
        assert result.original == "企业知识库的检索流程是什么"
        # 各子模块结果均为空
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        # 文本查询合集首项必为原始查询，且仅含原始查询
        assert result.all_text_queries == ["企业知识库的检索流程是什么"]
        assert result.all_text_queries[0] == result.original

    @pytest.mark.asyncio
    async def test_overall_timeout_emits_structured_log(
        self,
        overall_first_enhancer,
        mock_llm_gateway,
        overall_first_timeouts,
        caplog,
    ):
        """整体超时分支必须打 ``event=query_enhancement_timeout`` 结构化日志。"""

        async def hang(*_args, **_kwargs):
            await asyncio.sleep(overall_first_timeouts["overall"] * 3)
            return _make_response("late")

        mock_llm_gateway.complete.side_effect = hang

        with caplog.at_level(logging.WARNING, logger="app.services.query_enhancer"):
            result = await overall_first_enhancer.enhance("超时观测查询")

        assert result.all_text_queries == ["超时观测查询"]
        # 至少有一条 warning 包含约定的事件名
        timeout_logs = [
            r for r in caplog.records
            if "event=query_enhancement_timeout" in r.getMessage()
        ]
        assert len(timeout_logs) == 1, (
            f"应有 1 条 query_enhancement_timeout 事件日志，"
            f"实际：{[r.getMessage() for r in caplog.records]}"
        )
        # 日志须带具体超时阈值，便于运维定位
        assert "0.2" in timeout_logs[0].getMessage()


# ─── 2. 5s 内完成 → 返回完整增强结果 ──────────────────────────────────


class TestFastEnhancementSucceeds:
    """5s（测试缩短为 0.5s）内完成 → 增强结果完整保留。"""

    @pytest.mark.asyncio
    async def test_within_budget_returns_full_enhancement(
        self, enhancer, mock_llm_gateway
    ):
        """所有 LLM 调用均瞬时返回 → 改写 / HyDE / 分解全部生效。"""

        # 三次 complete 调用：rewrite / hyde / decompose
        responses = [
            _make_response("如何检索企业知识库\n企业知识库检索方法\n企业搜索流程"),
            _make_response("这是一段足够长的假设文档段落，描述了企业知识库的检索流程。"),
            _make_response("企业知识库的检索流程\n企业知识库的索引结构"),
        ]
        mock_llm_gateway.complete.side_effect = responses

        result = await enhancer.enhance("企业知识库的检索流程是什么")

        # 原始查询置首
        assert result.original == "企业知识库的检索流程是什么"
        assert result.all_text_queries[0] == "企业知识库的检索流程是什么"
        # 改写、HyDE、分解都拿到结果
        assert len(result.variants) >= 1
        assert len(result.hyde_embeddings) >= 1
        assert len(result.sub_queries) >= 2
        # 合集长度应严格大于 1（不是退化到只有原始查询）
        assert len(result.all_text_queries) > 1
        # 合集中不应该有重复
        assert len(result.all_text_queries) == len(set(result.all_text_queries))


# ─── 3. 部分子模块超时不拖累其他子模块 ────────────────────────────────


class TestPartialSubmoduleTimeout:
    """单个子模块的子超时不应导致整体降级；其他子模块仍保留结果。"""

    @pytest.mark.asyncio
    async def test_rewrite_succeeds_when_hyde_subtimeouts(
        self, enhancer, mock_llm_gateway, fast_timeouts
    ):
        """改写快速完成、HyDE 子超时 → 改写保留、HyDE 为空、原始查询仍置首。

        ``QueryEnhancer`` 给每个子模块独立 ``asyncio.wait_for`` 子超时，
        因此 HyDE 触发自己的 timeout 后只是返回 ``[]``，不会影响 rewrite。
        """
        call_index = {"n": 0}

        async def selective(*_args, **_kwargs):
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                # rewrite — 立即返回
                return _make_response("改写A\n改写B")
            if n == 2:
                # hyde — 故意超过 hyde 子超时，但仍小于整体超时
                # 这样 hyde 自己 timeout 返回 []，不会触发整体 wait_for
                await asyncio.sleep(fast_timeouts["hyde"] * 2)
                return _make_response("late hyde")
            # decompose — 立即返回，无需分解
            return _make_response("无需分解")

        mock_llm_gateway.complete.side_effect = selective

        result = await enhancer.enhance("混合查询")

        # 原始查询保留
        assert result.original == "混合查询"
        assert result.all_text_queries[0] == "混合查询"
        # rewrite 完整保留
        assert "改写A" in result.variants
        assert "改写B" in result.variants
        # hyde 因子超时被降级为空
        assert result.hyde_embeddings == []
        # decompose 显式 "无需分解"
        assert result.sub_queries == []
        # all_text_queries 至少包含原始 + 两条改写
        assert len(result.all_text_queries) >= 3

    @pytest.mark.asyncio
    async def test_overall_timeout_when_any_task_exceeds_budget(
        self, overall_first_enhancer, mock_llm_gateway, overall_first_timeouts
    ):
        """若有任意子模块拖到超过整体超时 → 整体降级。

        即便 rewrite 已经完成，由于 ``_enhance_internal`` 使用
        ``asyncio.wait(..., return_when=ALL_COMPLETED)`` 等待所有任务，
        外层 ``wait_for(OVERALL_TIMEOUT)`` 会在最后时刻抛 ``TimeoutError``。
        这是需求 7.5 的预期行为：5s 内未完成 → 直接降级。

        本测试使用 ``overall_first_*`` 配置（OVERALL=0.2s < 子超时 1.5s），
        让 sleep 落在 ``OVERALL < sleep < 子超时`` 区间，强制走整体降级路径。
        """
        call_index = {"n": 0}

        async def selective(*_args, **_kwargs):
            call_index["n"] += 1
            n = call_index["n"]
            if n == 1:
                # rewrite 立刻返回
                return _make_response("改写A\n改写B")
            # 其他子模块拖到超过整体超时（但小于子超时，避免子模块自己 timeout）
            await asyncio.sleep(overall_first_timeouts["overall"] * 3)
            return _make_response("late")

        mock_llm_gateway.complete.side_effect = selective

        result = await overall_first_enhancer.enhance("整体超时查询")

        # 整体降级：仅原始查询
        assert result.original == "整体超时查询"
        assert result.all_text_queries == ["整体超时查询"]
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []


# ─── 4. 异常路径同样降级 ─────────────────────────────────────────────


class TestExceptionFallback:
    """非超时异常（如 RuntimeError）也应触发降级，仅返回原始查询。"""

    @pytest.mark.asyncio
    async def test_runtime_error_falls_back_to_original(
        self, enhancer, mock_llm_gateway, monkeypatch
    ):
        """``_enhance_internal`` 内部抛出未捕获异常 → 整体捕获并降级。"""

        async def boom(_self, _query):
            raise RuntimeError("internal boom")

        monkeypatch.setattr(QueryEnhancer, "_enhance_internal", boom)

        result = await enhancer.enhance("异常路径查询")

        assert result.original == "异常路径查询"
        assert result.all_text_queries == ["异常路径查询"]
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []

    @pytest.mark.asyncio
    async def test_runtime_error_emits_structured_log(
        self, enhancer, monkeypatch, caplog
    ):
        """异常分支日志带 ``event=query_enhancement_failed``。"""

        async def boom(_self, _query):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(QueryEnhancer, "_enhance_internal", boom)

        with caplog.at_level(logging.WARNING, logger="app.services.query_enhancer"):
            result = await enhancer.enhance("异常日志查询")

        assert result.all_text_queries == ["异常日志查询"]
        failed_logs = [
            r for r in caplog.records
            if "event=query_enhancement_failed" in r.getMessage()
        ]
        assert len(failed_logs) == 1, (
            f"应有 1 条 query_enhancement_failed 事件日志，"
            f"实际：{[r.getMessage() for r in caplog.records]}"
        )
        # 日志中带异常文本
        assert "simulated failure" in failed_logs[0].getMessage()


# ─── 5. 不变量：all_text_queries[0] 始终为原始查询 ────────────────────


class TestAllTextQueriesInvariant:
    """需求 7.4 + 7.5 联合不变量：``all_text_queries[0]`` 永远等于 ``original``。"""

    @pytest.mark.asyncio
    async def test_invariant_on_overall_timeout(
        self, overall_first_enhancer, mock_llm_gateway, overall_first_timeouts
    ):
        """整体超时路径下不变量保持。"""

        async def hang(*_args, **_kwargs):
            await asyncio.sleep(overall_first_timeouts["overall"] * 3)
            return _make_response("late")

        mock_llm_gateway.complete.side_effect = hang

        result = await overall_first_enhancer.enhance("不变量超时查询")

        assert result.all_text_queries[0] == result.original

    @pytest.mark.asyncio
    async def test_invariant_on_success(self, enhancer, mock_llm_gateway):
        """成功路径下不变量保持。"""
        mock_llm_gateway.complete.side_effect = [
            _make_response("改写X\n改写Y"),
            _make_response("足够长的假设文档段落，覆盖原始查询语义。"),
            _make_response("子查询甲\n子查询乙"),
        ]

        result = await enhancer.enhance("不变量成功查询")

        assert result.all_text_queries[0] == result.original

    @pytest.mark.asyncio
    async def test_invariant_when_rewrite_returns_original_text(
        self, enhancer, mock_llm_gateway
    ):
        """改写碰巧产出与原始查询同字面 → 仍以原始查询为首项，不重复。"""
        mock_llm_gateway.complete.side_effect = [
            _make_response("不变量去重\n另一个改写"),  # 第一条与原始同字面
            _make_response("足够长的假设文档段落，覆盖原始查询语义。"),
            _make_response("无需分解"),
        ]

        result = await enhancer.enhance("不变量去重")

        assert result.all_text_queries[0] == "不变量去重"
        # 原始查询不应在合集中重复出现
        assert result.all_text_queries.count("不变量去重") == 1


# ─── 6. EnhancedQuery 降级语义构造一致性 ─────────────────────────────


class TestEnhancedQueryFallbackShape:
    """超时降级返回的 ``EnhancedQuery`` 在结构上应与"成功路径"一致。"""

    @pytest.mark.asyncio
    async def test_fallback_shape_matches_contract(
        self, overall_first_enhancer, mock_llm_gateway, overall_first_timeouts
    ):
        """超时降级返回的对象具备：``original``、空 variants/hyde/sub、
        ``all_text_queries=[original]``，且各别名属性可访问。"""

        async def hang(*_args, **_kwargs):
            await asyncio.sleep(overall_first_timeouts["overall"] * 3)
            return _make_response("late")

        mock_llm_gateway.complete.side_effect = hang

        result = await overall_first_enhancer.enhance("结构一致性")

        assert isinstance(result, EnhancedQuery)
        assert result.original == "结构一致性"
        # 别名属性（任务 15.4 引入）也可用
        assert result.original_query == "结构一致性"
        assert result.rewrites == []
        assert result.hypothetical_embeddings == []
        # all_text_queries 仅含原始查询
        assert result.all_text_queries == ["结构一致性"]
