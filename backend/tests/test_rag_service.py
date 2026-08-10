"""RAG 问答核心服务单元测试。

覆盖任务 16.2 的关键场景：

- 正常路径：检索到 chunks、调用 LLM、返回 ``RAGAnswer``
- ``sources`` 字段与输入 chunks 一一对应
- 检索为空时返回固定提示且不调用 LLM
- LLM 失败时抛出 ``RAGServiceError``
- ``top_k`` 参数控制检索条数（含夹紧到 ``[1, 20]``）
- Prompt 中包含每个 chunk 的内容与编号
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.llm_gateway import LLMGatewayError, LLMResponse
from app.services.rag_service import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    NO_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    RAGAnswer,
    RAGService,
    RAGServiceError,
    Source,
)
from app.services.search_service import SearchResponse, SearchResult

# ─── Helpers ────────────────────────────────────────────────────────────


def _make_result(
    *,
    chunk_id: str,
    document_id: str = "doc-1",
    title_chain: str = "第一章 > 第二节",
    source_file: str = "handbook.pdf",
    page_number: int = 3,
    score: float = 0.8,
    highlight: str = "示例正文片段",
    chunk_index: int = 0,
) -> SearchResult:
    """构造一个用于测试的 SearchResult。"""
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


def _make_response(results: list[SearchResult]) -> SearchResponse:
    """把 results 包装成 SearchResponse。"""
    return SearchResponse(
        results=results,
        total=len(results),
        page=1,
        page_size=len(results),
    )


def _make_search_service(results: list[SearchResult]) -> AsyncMock:
    """构造一个 mock SearchService，``search`` 始终返回给定结果。"""
    service = AsyncMock()
    service.search = AsyncMock(return_value=_make_response(results))
    return service


def _make_llm_gateway(
    *, content: str = "答案 [1]", usage: dict | None = None
) -> AsyncMock:
    """构造一个 mock LLMGateway，``complete`` 返回固定文本。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="test-model",
            usage=usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )
    )
    return gateway


# ─── 正常路径 ───────────────────────────────────────────────────────────


class TestAnswerHappyPath:
    """正常 RAG 流程的端到端行为。"""

    @pytest.mark.asyncio
    async def test_returns_rag_answer_with_llm_content(self):
        """LLM 返回的答案应原样写入 RAGAnswer.answer，且包含 token 用量。"""
        results = [
            _make_result(chunk_id="c1", highlight="片段一"),
            _make_result(chunk_id="c2", highlight="片段二"),
            _make_result(chunk_id="c3", highlight="片段三"),
        ]
        search = _make_search_service(results)
        llm = _make_llm_gateway(
            content="综合三段资料的答案 [1]",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
        service = RAGService(search_service=search, llm_gateway=llm)

        answer = await service.answer(
            query="什么是 RAG？",
            user_id="u-1",
            allowed_space_ids=["s-1"],
        )

        assert isinstance(answer, RAGAnswer)
        assert answer.answer == "综合三段资料的答案 [1]"
        assert answer.usage == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
        # 默认 top_k=5 透传给 SearchService
        search.search.assert_awaited_once()
        call_kwargs = search.search.await_args.kwargs
        assert call_kwargs["query"] == "什么是 RAG？"
        assert call_kwargs["user_id"] == "u-1"
        assert call_kwargs["allowed_space_ids"] == ["s-1"]
        assert call_kwargs["page"] == 1
        assert call_kwargs["page_size"] == DEFAULT_TOP_K

    @pytest.mark.asyncio
    async def test_sources_match_retrieved_chunks(self):
        """sources 列表必须与检索到的 chunks 一一对应（顺序、字段都对齐）。"""
        results = [
            _make_result(
                chunk_id="c-A",
                document_id="d-A",
                title_chain="A > A1",
                source_file="a.pdf",
                page_number=1,
                score=0.9,
            ),
            _make_result(
                chunk_id="c-B",
                document_id="d-B",
                title_chain="B > B1",
                source_file="b.pdf",
                page_number=12,
                score=0.6,
            ),
        ]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway(),
        )

        answer = await service.answer(
            query="问题", user_id="u", allowed_space_ids=["s"]
        )

        assert len(answer.sources) == 2
        first, second = answer.sources
        assert isinstance(first, Source)
        assert (first.index, first.chunk_id, first.document_id) == (1, "c-A", "d-A")
        assert first.source_file == "a.pdf"
        assert first.title_chain == "A > A1"
        assert first.page_number == 1
        assert first.score == pytest.approx(0.9)

        assert (second.index, second.chunk_id, second.document_id) == (2, "c-B", "d-B")
        assert second.page_number == 12
        assert second.score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_prompt_contains_all_chunks_with_numbering(self):
        """构造的 Prompt 必须包含每个 chunk 的编号、来源标签与内容。"""
        results = [
            _make_result(
                chunk_id=f"c{i}",
                source_file=f"file_{i}.pdf",
                title_chain=f"章节{i}",
                page_number=i,
                highlight=f"内容片段编号 {i}",
            )
            for i in range(1, 4)
        ]
        llm = _make_llm_gateway()
        service = RAGService(
            search_service=_make_search_service(results), llm_gateway=llm
        )

        await service.answer(query="问题 X", user_id="u", allowed_space_ids=["s"])

        llm.complete.assert_awaited_once()
        call_kwargs = llm.complete.await_args.kwargs
        prompt = call_kwargs["prompt"]
        system_prompt = call_kwargs["system_prompt"]

        assert system_prompt == SYSTEM_PROMPT

        for i in range(1, 4):
            # 编号
            assert f"[{i}]" in prompt
            # 文件名
            assert f"file_{i}.pdf" in prompt
            # 章节名
            assert f"章节{i}" in prompt
            # 页码
            assert f"页:{i}" in prompt
            # 内容片段
            assert f"内容片段编号 {i}" in prompt

        # 用户问题最终被附加
        assert "问题：问题 X" in prompt


# ─── 检索为空 ───────────────────────────────────────────────────────────


class TestEmptyRetrieval:
    """检索结果为空时的退化行为。"""

    @pytest.mark.asyncio
    async def test_returns_fixed_message_when_no_results(self):
        """检索为空时应返回固定提示且不调用 LLM。"""
        search = _make_search_service([])
        llm = _make_llm_gateway()
        service = RAGService(search_service=search, llm_gateway=llm)

        answer = await service.answer(
            query="冷僻问题", user_id="u", allowed_space_ids=["s"]
        )

        assert answer.answer == NO_CONTEXT_MESSAGE
        assert answer.sources == []
        assert answer.usage == {}
        llm.complete.assert_not_awaited()


# ─── LLM 失败 ───────────────────────────────────────────────────────────


class TestLLMFailure:
    """LLM 调用失败时应统一抛出 RAGServiceError。"""

    @pytest.mark.asyncio
    async def test_llm_timeout_raises_rag_service_error(self):
        results = [_make_result(chunk_id="c1")]
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=LLMGatewayError("LLM call timed out", reason="timeout")
        )
        service = RAGService(
            search_service=_make_search_service(results), llm_gateway=llm
        )

        with pytest.raises(RAGServiceError) as exc_info:
            await service.answer(query="Q", user_id="u", allowed_space_ids=["s"])

        assert exc_info.value.reason == "timeout"

    @pytest.mark.asyncio
    async def test_llm_auth_error_propagates_reason(self):
        results = [_make_result(chunk_id="c1")]
        llm = AsyncMock()
        llm.complete = AsyncMock(
            side_effect=LLMGatewayError("auth failed", reason="auth")
        )
        service = RAGService(
            search_service=_make_search_service(results), llm_gateway=llm
        )

        with pytest.raises(RAGServiceError) as exc_info:
            await service.answer(query="Q", user_id="u", allowed_space_ids=["s"])

        assert exc_info.value.reason == "auth"


# ─── top_k 控制 ─────────────────────────────────────────────────────────


class TestTopKControl:
    """top_k 参数应直接控制 SearchService.search 的 page_size 并被夹紧。"""

    @pytest.mark.asyncio
    async def test_custom_top_k_passed_to_search(self):
        results = [_make_result(chunk_id=f"c{i}") for i in range(3)]
        search = _make_search_service(results)
        service = RAGService(
            search_service=search, llm_gateway=_make_llm_gateway()
        )

        await service.answer(
            query="Q", user_id="u", allowed_space_ids=["s"], top_k=3
        )

        call_kwargs = search.search.await_args.kwargs
        assert call_kwargs["page_size"] == 3

    @pytest.mark.asyncio
    async def test_top_k_above_max_is_clamped(self):
        results = [_make_result(chunk_id="c1")]
        search = _make_search_service(results)
        service = RAGService(
            search_service=search, llm_gateway=_make_llm_gateway()
        )

        await service.answer(
            query="Q", user_id="u", allowed_space_ids=["s"], top_k=999
        )

        assert search.search.await_args.kwargs["page_size"] == MAX_TOP_K

    @pytest.mark.asyncio
    async def test_top_k_below_min_is_clamped(self):
        results = [_make_result(chunk_id="c1")]
        search = _make_search_service(results)
        service = RAGService(
            search_service=search, llm_gateway=_make_llm_gateway()
        )

        await service.answer(
            query="Q", user_id="u", allowed_space_ids=["s"], top_k=0
        )

        assert search.search.await_args.kwargs["page_size"] == MIN_TOP_K


# ─── 会话历史集成（任务 16.5） ─────────────────────────────────────────


class TestConversationIntegration:
    """RAGService 与 ConversationService 的对接行为。

    覆盖点：

    - 不传 ``conversation_id`` 时不读不写历史
    - 传 ``conversation_id`` 时把历史以 ``用户：`` / ``助手：`` 拼到 Prompt 中
    - LLM 完成后把当前轮（user + assistant）写回会话
    - 流式版本同样支持以上行为
    - Redis 写入失败不影响主答案返回
    """

    @staticmethod
    def _conv_service(
        *,
        history: list[dict] | None = None,
        append_side_effect: BaseException | None = None,
    ):
        from unittest.mock import AsyncMock

        svc = AsyncMock()
        svc.get_history = AsyncMock(return_value=list(history or []))
        if append_side_effect is not None:
            svc.append = AsyncMock(side_effect=append_side_effect)
        else:
            svc.append = AsyncMock()
        return svc

    @pytest.mark.asyncio
    async def test_no_conversation_id_skips_history_io(self):
        """conversation_id 为 None 时既不读也不写历史。"""
        results = [_make_result(chunk_id="c1")]
        conv = self._conv_service(history=[{"role": "user", "content": "旧"}])
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway(content="答案"),
            conversation_service=conv,
        )

        await service.answer(query="Q", user_id="u", allowed_space_ids=["s"])

        conv.get_history.assert_not_awaited()
        conv.append.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_history_is_inlined_into_prompt(self):
        """传入 conversation_id 时历史应作为'对话历史：'段拼入 user prompt。"""
        results = [_make_result(chunk_id="c1", highlight="片段")]
        history = [
            {"role": "user", "content": "上一轮问题"},
            {"role": "assistant", "content": "上一轮答案 [1]"},
            # 脏数据/未知 role 应被忽略
            {"role": "system", "content": "should-be-dropped"},
        ]
        llm = _make_llm_gateway(content="本轮答案")
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=llm,
            conversation_service=self._conv_service(history=history),
        )

        await service.answer(
            query="本轮问题",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-xyz",
        )

        prompt = llm.complete.await_args.kwargs["prompt"]
        assert "对话历史" in prompt
        assert "用户：上一轮问题" in prompt
        assert "助手：上一轮答案 [1]" in prompt
        assert "should-be-dropped" not in prompt
        # 当前问题仍然在最后
        assert prompt.rstrip().endswith("问题：本轮问题")

    @pytest.mark.asyncio
    async def test_persists_current_turn_after_completion(self):
        """LLM 返回成功后必须把 user + assistant 各 append 一次。"""
        results = [_make_result(chunk_id="c1")]
        conv = self._conv_service(history=[])
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway(content="本轮答案"),
            conversation_service=conv,
        )

        await service.answer(
            query="本轮问题",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-1",
        )

        assert conv.append.await_count == 2
        first_call, second_call = conv.append.await_args_list
        assert first_call.args == ("conv-1", "user", "本轮问题")
        assert second_call.args == ("conv-1", "assistant", "本轮答案")

    @pytest.mark.asyncio
    async def test_no_append_when_retrieval_empty(self):
        """检索为空时直接返回固定提示，不应当把空 LLM 答案写入历史。"""
        conv = self._conv_service(history=[])
        service = RAGService(
            search_service=_make_search_service([]),
            llm_gateway=_make_llm_gateway(),
            conversation_service=conv,
        )

        await service.answer(
            query="冷僻问题",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-1",
        )

        # 检索为空走短路分支：既然 LLM 没被调用，也不应当把"知识库中未找到相关
        # 内容。"作为助手答案污染历史。
        conv.append.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_append_failure_does_not_break_answer(self):
        """会话写入失败不应让 answer() 抛错——答案对用户已经返回成功。"""
        results = [_make_result(chunk_id="c1")]
        conv = self._conv_service(
            history=[],
            append_side_effect=RuntimeError("redis down"),
        )
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway(content="OK"),
            conversation_service=conv,
        )

        # 不抛异常即可
        result = await service.answer(
            query="Q",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-1",
        )
        assert result.answer == "OK"

    @pytest.mark.asyncio
    async def test_streaming_persists_full_answer(self):
        """answer_stream 在正常结束时也应把完整答案写回会话。"""
        from unittest.mock import MagicMock

        results = [_make_result(chunk_id="c1")]
        conv = self._conv_service(history=[])

        async def _stream(**kwargs):
            for tok in ["第一段", "+第二段"]:
                yield tok

        gateway = MagicMock()
        gateway.stream = _stream
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=gateway,
            conversation_service=conv,
        )

        events = []
        async for ev in service.answer_stream(
            query="问",
            user_id="u",
            allowed_space_ids=["s"],
            conversation_id="conv-stream",
        ):
            events.append(ev)

        # 至少包含 token + sources + done
        kinds = [e.event for e in events]
        assert kinds[-2:] == ["sources", "done"]
        # 完整答案应当为两段拼接结果
        assert conv.append.await_count == 2
        assert conv.append.await_args_list[0].args == (
            "conv-stream", "user", "问",
        )
        assert conv.append.await_args_list[1].args == (
            "conv-stream", "assistant", "第一段+第二段",
        )

    @pytest.mark.asyncio
    async def test_streaming_does_not_persist_on_first_token_timeout(self):
        """首 token 超时时不应当把空答案写回历史。"""
        import asyncio
        from unittest.mock import MagicMock

        results = [_make_result(chunk_id="c1")]
        conv = self._conv_service(history=[])

        async def _slow_stream(**kwargs):
            await asyncio.sleep(0.5)
            yield "too-late"

        gateway = MagicMock()
        gateway.stream = _slow_stream
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=gateway,
            conversation_service=conv,
        )

        events = []
        async for ev in service.answer_stream(
            query="问",
            user_id="u",
            allowed_space_ids=["s"],
            first_token_timeout=0.05,
            conversation_id="conv-stream",
        ):
            events.append(ev)

        assert events[-1].event == "error"
        assert events[-1].data["code"] == "first_token_timeout"
        conv.append.assert_not_awaited()
