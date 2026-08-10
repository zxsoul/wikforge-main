"""RAG 相似度阈值过滤单元测试（任务 16.6 / 需求 8.6）。

覆盖点：

- ``RAGService.similarity_threshold`` 默认值为 ``0.5``
- 构造函数 ``similarity_threshold`` 参数可显式覆盖默认阈值
- ``Settings.SIMILARITY_THRESHOLD`` 的配置生效（构造未显式传值时）
- 当 SearchService 返回的所有候选 chunk 分数都低于阈值时：
    * ``answer`` 返回 ``NO_CONTEXT_MESSAGE``，``sources`` 为空、``usage`` 为空
    * 不调用 LLM
- 当部分 chunk 分数 ≥ 阈值时：仅保留高分 chunk 进入 Prompt 与 sources
- ``score == 0.5`` 边界值：按"满足条件"处理（包含）
- ``answer_stream`` 同样应用阈值过滤
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.llm_gateway import LLMResponse
from app.services.rag_service import (
    DEFAULT_SIMILARITY_THRESHOLD,
    NO_CONTEXT_MESSAGE,
    RAGService,
)
from app.services.search_service import SearchResponse, SearchResult

# ─── Helpers ────────────────────────────────────────────────────────────


def _make_result(
    *,
    chunk_id: str,
    score: float,
    document_id: str = "doc-1",
    title_chain: str = "章节",
    source_file: str = "doc.pdf",
    page_number: int = 1,
    highlight: str = "示例片段",
    chunk_index: int = 0,
) -> SearchResult:
    """构造测试用的 SearchResult。"""
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        title_chain=title_chain,
        source_file=source_file,
        page_number=page_number,
        score=score,
        highlight=highlight,
    )


def _make_search_service(results: list[SearchResult]) -> AsyncMock:
    service = AsyncMock()
    service.search = AsyncMock(
        return_value=SearchResponse(
            results=results,
            total=len(results),
            page=1,
            page_size=len(results),
        )
    )
    return service


def _make_llm_gateway(content: str = "答案 [1]") -> AsyncMock:
    gateway = AsyncMock()
    gateway.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="test-model",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
        )
    )
    return gateway


def _make_conversation_service() -> AsyncMock:
    """ConversationService 在本套测试中不需要真实行为，仅返回空历史。"""
    svc = AsyncMock()
    svc.get_history = AsyncMock(return_value=[])
    svc.append = AsyncMock()
    return svc


# ─── 默认阈值与显式覆盖 ──────────────────────────────────────────────────


class TestThresholdConfiguration:
    """阈值的来源与覆盖优先级。"""

    def test_default_threshold_is_module_constant(self):
        """模块常量 ``DEFAULT_SIMILARITY_THRESHOLD`` 必须为 0.5（需求 8.6）。"""
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.5

    def test_default_instance_uses_settings_value(self):
        """未显式传参时 ``similarity_threshold`` 来自 ``Settings``。"""
        from app.core.config import get_settings

        service = RAGService(
            search_service=_make_search_service([]),
            llm_gateway=_make_llm_gateway(),
            conversation_service=_make_conversation_service(),
        )

        assert service.similarity_threshold == get_settings().SIMILARITY_THRESHOLD
        # 默认 settings 值应为 0.5
        assert service.similarity_threshold == 0.5

    def test_constructor_overrides_default(self):
        """构造函数显式传值优先于 ``Settings``。"""
        service = RAGService(
            search_service=_make_search_service([]),
            llm_gateway=_make_llm_gateway(),
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.8,
        )

        assert service.similarity_threshold == 0.8

    def test_settings_threshold_is_picked_up(self, monkeypatch):
        """修改 ``Settings.SIMILARITY_THRESHOLD`` 会被 RAGService 读取。

        ``get_settings`` 使用 ``lru_cache``，需手动清空缓存以读到 patch 后的值；
        Settings 是 pydantic BaseSettings 实例，直接 ``setattr`` 实例字段。
        """
        from app.core import config as config_module

        # 先清空缓存，让下面的 get_settings() 返回一个新实例
        config_module.get_settings.cache_clear()
        settings_instance = config_module.get_settings()
        monkeypatch.setattr(
            settings_instance, "SIMILARITY_THRESHOLD", 0.42, raising=False
        )
        try:
            service = RAGService(
                search_service=_make_search_service([]),
                llm_gateway=_make_llm_gateway(),
                conversation_service=_make_conversation_service(),
            )
            assert service.similarity_threshold == pytest.approx(0.42)
        finally:
            # 还原缓存供后续测试使用
            config_module.get_settings.cache_clear()


# ─── 过滤行为（answer） ──────────────────────────────────────────────────


class TestAnswerThresholdFiltering:
    """``RAGService.answer`` 在阈值过滤下的行为。"""

    @pytest.mark.asyncio
    async def test_all_below_threshold_returns_no_context_message(self):
        """全部 chunk 分数 < 阈值时返回固定提示，不调用 LLM。"""
        results = [
            _make_result(chunk_id="c1", score=0.49),
            _make_result(chunk_id="c2", score=0.30),
            _make_result(chunk_id="c3", score=0.10),
        ]
        search = _make_search_service(results)
        llm = _make_llm_gateway()
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.5,
        )

        answer = await service.answer(
            query="低相关问题", user_id="u", allowed_space_ids=["s"]
        )

        assert answer.answer == NO_CONTEXT_MESSAGE
        assert answer.sources == []
        assert answer.usage == {}
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_above_threshold_keeps_only_high_scores(self):
        """部分 chunk 分数 ≥ 阈值时，仅高分 chunk 进入 sources 与 Prompt。"""
        results = [
            _make_result(
                chunk_id="c-high",
                score=0.9,
                source_file="high.pdf",
                highlight="高分片段",
            ),
            _make_result(
                chunk_id="c-low",
                score=0.3,
                source_file="low.pdf",
                highlight="低分片段",
            ),
            _make_result(
                chunk_id="c-mid",
                score=0.6,
                source_file="mid.pdf",
                highlight="中分片段",
            ),
        ]
        search = _make_search_service(results)
        llm = _make_llm_gateway(content="基于高分片段的答案 [1][2]")
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.5,
        )

        answer = await service.answer(
            query="混合相关性问题", user_id="u", allowed_space_ids=["s"]
        )

        # sources 仅包含高分两条，按原始顺序
        assert [s.chunk_id for s in answer.sources] == ["c-high", "c-mid"]
        # 编号应在过滤后重新分配
        assert [s.index for s in answer.sources] == [1, 2]

        # LLM 看到的 Prompt 不应包含低分内容
        llm.complete.assert_awaited_once()
        prompt = llm.complete.await_args.kwargs["prompt"]
        assert "高分片段" in prompt
        assert "中分片段" in prompt
        assert "低分片段" not in prompt
        assert "low.pdf" not in prompt

    @pytest.mark.asyncio
    async def test_score_equal_to_threshold_is_included(self):
        """边界值：``score == threshold`` 应保留（与需求'低于'的措辞一致）。"""
        results = [
            _make_result(chunk_id="c-edge", score=0.5),
            _make_result(chunk_id="c-below", score=0.499),
        ]
        search = _make_search_service(results)
        llm = _make_llm_gateway(content="OK [1]")
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.5,
        )

        answer = await service.answer(
            query="边界问题", user_id="u", allowed_space_ids=["s"]
        )

        assert [s.chunk_id for s in answer.sources] == ["c-edge"]
        assert answer.answer == "OK [1]"

    @pytest.mark.asyncio
    async def test_custom_threshold_filters_more_aggressively(self):
        """构造函数传入更高阈值时，过滤会更严格。"""
        results = [
            _make_result(chunk_id="c1", score=0.6),
            _make_result(chunk_id="c2", score=0.85),
        ]
        search = _make_search_service(results)
        llm = _make_llm_gateway()
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.8,
        )

        answer = await service.answer(
            query="高门槛问题", user_id="u", allowed_space_ids=["s"]
        )

        assert [s.chunk_id for s in answer.sources] == ["c2"]


# ─── 过滤行为（answer_stream） ──────────────────────────────────────────


class TestStreamThresholdFiltering:
    """流式版本同样应用相似度阈值。"""

    @pytest.mark.asyncio
    async def test_stream_all_below_threshold_returns_no_context(self):
        """所有 chunk 分数低于阈值时，流式产出固定提示，不进入 LLM。"""
        results = [
            _make_result(chunk_id="c1", score=0.2),
            _make_result(chunk_id="c2", score=0.4),
        ]

        async def _should_not_be_called(**_kwargs):
            # 一旦被调用就让测试失败
            raise AssertionError("LLM stream should not be invoked")
            yield  # pragma: no cover - 仅为让函数成为 async generator

        gateway = MagicMock()
        gateway.stream = _should_not_be_called

        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=gateway,
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.5,
        )

        events = []
        async for ev in service.answer_stream(
            query="问", user_id="u", allowed_space_ids=["s"]
        ):
            events.append(ev)

        kinds = [e.event for e in events]
        assert kinds == ["token", "sources", "done"]
        assert events[0].data == {"text": NO_CONTEXT_MESSAGE}
        assert events[1].data == {"sources": []}

    @pytest.mark.asyncio
    async def test_stream_partial_above_threshold_filters_sources(self):
        """流式版本在部分 chunk 通过阈值时，sources 事件仅含高分 chunk。"""
        results = [
            _make_result(chunk_id="c-low", score=0.2, source_file="low.pdf"),
            _make_result(chunk_id="c-high", score=0.9, source_file="high.pdf"),
        ]

        async def _stream(**_kwargs):
            for tok in ["答案 ", "[1]"]:
                yield tok

        gateway = MagicMock()
        gateway.stream = _stream

        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=gateway,
            conversation_service=_make_conversation_service(),
            similarity_threshold=0.5,
        )

        events = []
        async for ev in service.answer_stream(
            query="问", user_id="u", allowed_space_ids=["s"]
        ):
            events.append(ev)

        # 找出 sources 事件并校验
        source_events = [e for e in events if e.event == "sources"]
        assert len(source_events) == 1
        sources_payload = source_events[0].data["sources"]
        assert [s["chunk_id"] for s in sources_payload] == ["c-high"]
        assert sources_payload[0]["index"] == 1
