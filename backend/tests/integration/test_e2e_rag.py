"""RAG 问答端到端测试（任务 25.4）。

覆盖：导入文档 → 提问 → 验证回答包含引用 的多场景查询。

策略：
- ``SearchService`` mock 为预设语料，便于断言 LLM 拿到了正确上下文
- ``LLMGateway`` mock 为可控响应（含/不含 ``[n]`` 引用）
- 通过 ``RAGService.answer`` 直连，确保 prompt → citation 解析端到端
- 兼容路由层 SSE：另一个测试用 ``RAGEngine.chat`` 流式接口

Validates: Requirements 8
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.services.llm_gateway import LLMResponse
from app.services.rag_service import (
    NO_CONTEXT_MESSAGE,
    RAGService,
)
from app.services.search_service import SearchResponse, SearchResult


pytestmark = pytest.mark.integration


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def conversation_service():
    """fakeredis 驱动的真实 ConversationService。"""
    fakeredis = pytest.importorskip("fakeredis")
    from app.services.conversation_service import ConversationService

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield ConversationService(redis_client=redis)
    finally:
        await redis.aclose()


def _make_result(*, chunk_id: str, score: float, highlight: str, source: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=0,
        title_chain="测试章节",
        source_file=source,
        page_number=1,
        score=score,
        highlight=highlight,
    )


def _make_search(results: list[SearchResult]) -> AsyncMock:
    svc = AsyncMock()
    svc.search = AsyncMock(
        return_value=SearchResponse(
            results=results,
            total=len(results),
            page=1,
            page_size=max(1, len(results)),
        )
    )
    return svc


def _make_llm(content: str) -> AsyncMock:
    gw = AsyncMock()
    gw.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="stub",
            usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            finish_reason="stop",
        )
    )
    return gw


# ─── 测试 ─────────────────────────────────────────────────────────────


class TestRAGAnswerQuality:
    """多场景 RAG 问答：验证答案质量与引用标注。"""

    @pytest.mark.asyncio
    async def test_factual_query_returns_answer_with_citation(
        self, conversation_service
    ):
        """事实型查询：答案应包含 [1] 引用，sources 含正确 cited 标记。"""
        results = [
            _make_result(
                chunk_id="c-1",
                score=0.92,
                highlight="RAG 通过检索增强生成。",
                source="rag.pdf",
            ),
            _make_result(
                chunk_id="c-2",
                score=0.74,
                highlight="向量检索使用近似最近邻。",
                source="vec.pdf",
            ),
        ]
        service = RAGService(
            search_service=_make_search(results),
            llm_gateway=_make_llm("RAG 是检索增强生成 [1]，常配合向量检索使用 [2]。"),
            conversation_service=conversation_service,
        )

        result = await service.answer(
            query="什么是 RAG？",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-fact",
        )

        assert "[1]" in result.answer and "[2]" in result.answer
        assert {s.chunk_id: s.cited for s in result.sources} == {
            "c-1": True,
            "c-2": True,
        }

    @pytest.mark.asyncio
    async def test_no_context_query_returns_fallback_message(
        self, conversation_service
    ):
        """检索为空时，应返回固定提示且不调用 LLM。"""
        llm = _make_llm("不该被调用")
        service = RAGService(
            search_service=_make_search([]),
            llm_gateway=llm,
            conversation_service=conversation_service,
        )

        result = await service.answer(
            query="知识库里没有的问题",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-empty",
        )

        assert result.answer == NO_CONTEXT_MESSAGE
        assert result.sources == []
        llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_low_similarity_chunks_filtered_out(self, conversation_service):
        """分数低于阈值的 chunk 不应进入 prompt 与 sources。"""
        results = [
            _make_result(chunk_id="c-1", score=0.92, highlight="高分相关", source="a.pdf"),
            _make_result(chunk_id="c-low", score=0.30, highlight="无关内容", source="b.pdf"),
        ]
        llm = _make_llm("答案 [1]")
        service = RAGService(
            search_service=_make_search(results),
            llm_gateway=llm,
            conversation_service=conversation_service,
            similarity_threshold=0.5,
        )

        result = await service.answer(
            query="问题",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-thresh",
        )

        assert {s.chunk_id for s in result.sources} == {"c-1"}
        prompt = llm.complete.await_args.kwargs["prompt"]
        assert "高分相关" in prompt
        assert "无关内容" not in prompt

    @pytest.mark.asyncio
    async def test_streaming_answer_yields_tokens_then_sources_then_done(
        self, conversation_service
    ):
        """流式接口先发 token，再发 sources，最后 done。"""
        from app.services.rag_service import (
            STREAM_EVENT_DONE,
            STREAM_EVENT_SOURCES,
            STREAM_EVENT_TOKEN,
        )

        results = [_make_result(chunk_id="c-1", score=0.9, highlight="片段", source="a.pdf")]

        async def _stream(*_args, **_kwargs) -> AsyncIterator[str]:
            for tok in ["答案 [", "1", "]"]:
                yield tok

        gw = MagicMock()
        gw.stream = _stream
        service = RAGService(
            search_service=_make_search(results),
            llm_gateway=gw,
            conversation_service=conversation_service,
        )

        events = []
        async for ev in service.answer_stream(
            query="问",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-stream",
        ):
            events.append(ev)

        kinds = [e.event for e in events]
        assert STREAM_EVENT_TOKEN in kinds
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]

        full_answer = "".join(
            e.data["text"] for e in events if e.event == STREAM_EVENT_TOKEN
        )
        assert full_answer == "答案 [1]"
