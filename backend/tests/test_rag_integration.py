"""任务 16.10：RAG 引擎端到端集成测试。

定位
----

任务 16.1–16.9 已经为 RAG 引擎的各个组件（LLMGateway、RAGService、流式 SSE、
引用标注、ConversationService、相似度阈值、LLM 超时、会话过期、RAG API）写了
专门的单元测试。本模块作为收尾，专门覆盖**多个组件协同工作**的端到端场景：

- ``Search → Filter（相似度阈值） → Prompt → LLM → Citation 解析 → 会话写入``
  这条主链路上的真实代码路径都跑一遍。
- 通过 ``fakeredis`` 注入真实的 :class:`ConversationService`，让多轮对话、TTL、
  20 轮容量上限的逻辑都按真实 Redis 协议执行。
- LLM 与底层 SearchService 仍以 mock 形式注入，避免触达真实模型与 OpenSearch /
  Qdrant，但 RAGService 内部组装 Prompt、解析 ``[n]`` 引用、过滤低分 chunk、
  写入会话等所有逻辑都走真实代码。

四类场景
~~~~~~~~

1. 完整流程：单轮问答中各步骤串联是否正确（含 cited 标记、usage 透传、
   会话写入、相似度阈值过滤）。
2. 多轮对话：第一轮无历史、第二轮把历史拼进 Prompt、第三轮起触发 20 轮容量
   上限自动驱逐最旧消息。
3. 错误恢复：LLM 超时后下一次调用应当能正常返回；首次失败不污染会话历史。
4. 权限过滤：``allowed_space_ids`` 应原样透传给 SearchService，且不同用户的
   会话彼此隔离。
"""

from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.services.conversation_service import (
    KEY_PREFIX,
    MAX_TURNS,
    ConversationService,
)
from app.services.llm_gateway import LLMGatewayError, LLMResponse
from app.services.rag_service import (
    NO_CONTEXT_MESSAGE,
    STREAM_EVENT_DONE,
    STREAM_EVENT_ERROR,
    STREAM_EVENT_SOURCES,
    STREAM_EVENT_TOKEN,
    RAGService,
    RAGServiceError,
)
from app.services.search_service import SearchResponse, SearchResult

# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_redis_client():
    """fakeredis 异步客户端,用于真实驱动 ConversationService。"""
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def conversation_service(fake_redis_client) -> ConversationService:
    """注入 fakeredis 的真实 ConversationService。"""
    return ConversationService(redis_client=fake_redis_client)


# ─── 工厂函数 ──────────────────────────────────────────────────────────


def _make_search_result(
    *,
    chunk_id: str,
    score: float,
    document_id: str = "doc-1",
    title_chain: str = "第一章 > 第一节",
    source_file: str = "manual.pdf",
    page_number: int = 1,
    highlight: str = "示例片段",
    chunk_index: int = 0,
) -> SearchResult:
    """构造一个 SearchResult 测试样本。"""
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


def _make_search_response(results: list[SearchResult]) -> SearchResponse:
    """把 results 包装成 SearchResponse。"""
    return SearchResponse(
        results=results,
        total=len(results),
        page=1,
        page_size=max(1, len(results)),
    )


def _make_search_service(
    results_or_func,
) -> AsyncMock:
    """构造 mock SearchService。

    Args:
        results_or_func: 直接传 ``list[SearchResult]`` 时所有调用都返回同一份;
            若传可调用对象,则按 SearchService.search 的关键字参数动态计算
            返回值,便于校验权限传递等场景。
    """
    service = AsyncMock()
    if callable(results_or_func):
        async def _dynamic_search(**kwargs):
            results = results_or_func(**kwargs)
            return _make_search_response(results)
        service.search = AsyncMock(side_effect=_dynamic_search)
    else:
        service.search = AsyncMock(
            return_value=_make_search_response(results_or_func)
        )
    return service


def _make_llm_gateway(
    *,
    content: str = "答案 [1]",
    usage: dict | None = None,
) -> AsyncMock:
    """构造 mock LLMGateway,``complete`` 返回固定文本。"""
    gateway = AsyncMock()
    gateway.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="stub-model",
            usage=usage or {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
            },
            finish_reason="stop",
        )
    )
    return gateway


def _make_streaming_gateway(tokens: list[str]) -> MagicMock:
    """构造 mock LLMGateway,``stream`` 按顺序异步产出 tokens。"""
    gateway = MagicMock()

    async def _stream(*_args, **_kwargs) -> AsyncIterator[str]:
        for tok in tokens:
            yield tok

    gateway.stream = _stream
    return gateway


# ─── 1. 完整 RAG 流程端到端 ────────────────────────────────────────────


class TestFullRAGPipeline:
    """单轮问答中,Search → Filter → Prompt → LLM → Citation → Session 全链路。"""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_with_filter_citation_and_persist(
        self, conversation_service, fake_redis_client
    ):
        """整条主链路一次性跑通,逐项校验输出与副作用。"""
        # 5 个候选 chunk:其中 c-low 分数低于阈值,应被过滤掉
        results = [
            _make_search_result(
                chunk_id="c-1",
                score=0.92,
                source_file="rag.pdf",
                title_chain="RAG 简介",
                highlight="RAG 是检索增强生成。",
                page_number=2,
            ),
            _make_search_result(
                chunk_id="c-2",
                score=0.81,
                source_file="rag.pdf",
                title_chain="RAG 实践",
                highlight="它结合了检索和生成两步。",
                page_number=5,
            ),
            _make_search_result(
                chunk_id="c-3",
                score=0.73,
                source_file="vector.pdf",
                title_chain="向量检索",
                highlight="向量检索用 ANN。",
                page_number=1,
            ),
            _make_search_result(
                chunk_id="c-4",
                score=0.55,
                source_file="bm25.pdf",
                title_chain="BM25",
                highlight="BM25 是经典稀疏算法。",
                page_number=3,
            ),
            _make_search_result(
                chunk_id="c-low",
                score=0.30,
                source_file="legacy.pdf",
                title_chain="无关章节",
                highlight="完全不相关的内容。",
                page_number=8,
            ),
        ]
        search = _make_search_service(results)
        # LLM 答案引用 [1] 和 [3];[5] 越界(过滤后只有 4 条),应被忽略
        llm = _make_llm_gateway(
            content="RAG 包含检索 [1] 与生成两步,常见检索是向量检索 [3]。",
            usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        )
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=conversation_service,
            similarity_threshold=0.5,
        )

        result = await service.answer(
            query="什么是 RAG?",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            top_k=5,
            conversation_id="conv-e2e",
        )

        # 1) Search 调用参数透传完整
        search.search.assert_awaited_once()
        call_kwargs = search.search.await_args.kwargs
        assert call_kwargs["query"] == "什么是 RAG?"
        assert call_kwargs["user_id"] == "u-1"
        assert call_kwargs["allowed_space_ids"] == ["space-A"]
        assert call_kwargs["page_size"] == 5

        # 2) 阈值过滤:c-low 被过滤,sources 只剩 4 条且重新编号 1..4
        assert [s.chunk_id for s in result.sources] == [
            "c-1", "c-2", "c-3", "c-4",
        ]
        assert [s.index for s in result.sources] == [1, 2, 3, 4]

        # 3) Prompt 应包含 4 个候选的 highlight 与来源标签,但不含被过滤的 c-low
        llm.complete.assert_awaited_once()
        prompt = llm.complete.await_args.kwargs["prompt"]
        assert "RAG 是检索增强生成。" in prompt
        assert "向量检索用 ANN。" in prompt
        assert "BM25 是经典稀疏算法。" in prompt
        assert "完全不相关的内容。" not in prompt
        assert "legacy.pdf" not in prompt
        # 第一段编号必为 [1],最后段编号必为 [4]
        assert "[1]" in prompt and "[4]" in prompt
        assert "[5]" not in prompt
        # 用户问题位于末尾
        assert prompt.rstrip().endswith("问题:什么是 RAG?".replace(":", "："))

        # 4) Citation 解析:[1] / [3] 命中,c-2 与 c-4 标记为未引用
        cited_map = {s.chunk_id: s.cited for s in result.sources}
        assert cited_map == {
            "c-1": True,
            "c-2": False,
            "c-3": True,
            "c-4": False,
        }

        # 5) Usage 完整透传
        assert result.usage == {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }

        # 6) 会话已写入 Redis(真实 fakeredis 路径)
        history = await conversation_service.get_history("conv-e2e")
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "什么是 RAG?"
        assert history[1]["content"] == result.answer
        # TTL 应约等于 30 分钟(允许浮动)
        ttl = await conversation_service.ttl("conv-e2e")
        assert 1700 < ttl <= 1800

    @pytest.mark.asyncio
    async def test_end_to_end_streaming_pipeline(self, conversation_service):
        """流式版本同样跑通端到端,事件顺序与会话写入正确。"""
        results = [
            _make_search_result(chunk_id="c-1", score=0.9),
            _make_search_result(chunk_id="c-2", score=0.7),
        ]
        search = _make_search_service(results)
        # 把 [1] 拆到不同 token,验证 cited 解析基于完整答案
        gateway = _make_streaming_gateway(
            ["根据 [", "1", "] 资料,RAG 的核心是", "检索 + 生成。"]
        )
        service = RAGService(
            search_service=search,
            llm_gateway=gateway,
            conversation_service=conversation_service,
            similarity_threshold=0.5,
        )

        events = []
        async for ev in service.answer_stream(
            query="解释 RAG",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            top_k=3,
            conversation_id="conv-stream",
        ):
            events.append(ev)

        # 事件顺序:多个 token → sources → done
        kinds = [e.event for e in events]
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]
        token_events = [e for e in events if e.event == STREAM_EVENT_TOKEN]
        assert len(token_events) == 4
        full_answer = "".join(e.data["text"] for e in token_events)
        assert full_answer == "根据 [1] 资料,RAG 的核心是检索 + 生成。"

        # sources 事件携带 cited 标记
        sources_event = next(e for e in events if e.event == STREAM_EVENT_SOURCES)
        sources_payload = sources_event.data["sources"]
        assert {s["chunk_id"]: s["cited"] for s in sources_payload} == {
            "c-1": True,
            "c-2": False,
        }

        # 流式正常结束后,完整答案已写入会话
        history = await conversation_service.get_history("conv-stream")
        assert [m["content"] for m in history] == [
            "解释 RAG",
            full_answer,
        ]


# ─── 2. 多轮对话 ───────────────────────────────────────────────────────


class TestMultiTurnConversation:
    """连续多轮问答中,会话历史的拼接、写入与容量上限行为。"""

    @pytest.mark.asyncio
    async def test_first_turn_has_no_history_in_prompt(
        self, conversation_service
    ):
        """第一轮问答时 Prompt 中不应当出现"对话历史"段。"""
        results = [_make_search_result(chunk_id="c-1", score=0.9)]
        search = _make_search_service(results)
        llm = _make_llm_gateway(content="第一轮答案 [1]")
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=conversation_service,
        )

        await service.answer(
            query="第一轮问题",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-multi",
        )

        prompt = llm.complete.await_args.kwargs["prompt"]
        assert "对话历史" not in prompt
        # 历史现在应已写入
        history = await conversation_service.get_history("conv-multi")
        assert [m["content"] for m in history] == [
            "第一轮问题", "第一轮答案 [1]"
        ]

    @pytest.mark.asyncio
    async def test_second_turn_includes_prior_turn_in_prompt(
        self, conversation_service
    ):
        """第二轮问答时,上一轮的 user/assistant 应被拼接到 Prompt 历史段。"""
        results = [_make_search_result(chunk_id="c-1", score=0.9)]
        search = _make_search_service(results)
        # 给同一个 service 用两个不同 LLM 响应
        first_llm = _make_llm_gateway(content="第一轮答案 [1]")
        service = RAGService(
            search_service=search,
            llm_gateway=first_llm,
            conversation_service=conversation_service,
        )

        # 第一轮
        await service.answer(
            query="第一轮问题",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-multi",
        )

        # 第二轮:换一个 LLM mock,能看到前一轮内容
        second_llm = _make_llm_gateway(content="第二轮答案 [1]")
        service._llm_gateway = second_llm  # 仅测试用,直接替换

        await service.answer(
            query="第二轮问题",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-multi",
        )

        prompt = second_llm.complete.await_args.kwargs["prompt"]
        # 历史段必须出现
        assert "对话历史" in prompt
        # 上一轮 user / assistant 都被拼接(角色前缀使用全角冒号)
        assert "用户:第一轮问题" in prompt or "用户：第一轮问题" in prompt
        assert "助手:第一轮答案 [1]" in prompt or "助手：第一轮答案 [1]" in prompt
        # 当前问题在末尾
        assert "第二轮问题" in prompt
        assert prompt.rstrip().endswith("问题:第二轮问题".replace(":", "："))

        # 会话历史现已包含两轮共 4 条消息
        history = await conversation_service.get_history("conv-multi")
        assert len(history) == 4
        assert [m["content"] for m in history] == [
            "第一轮问题", "第一轮答案 [1]",
            "第二轮问题", "第二轮答案 [1]",
        ]

    @pytest.mark.asyncio
    async def test_capacity_limit_evicts_oldest_turns(
        self, conversation_service
    ):
        """超过 20 轮(40 条消息)时,最旧的轮次应被自动驱逐。"""
        results = [_make_search_result(chunk_id="c-1", score=0.9)]
        search = _make_search_service(results)
        service = RAGService(
            search_service=search,
            llm_gateway=_make_llm_gateway(),
            conversation_service=conversation_service,
        )

        # 连续问 25 轮
        for i in range(25):
            # 每轮换一个 LLM 响应,便于断言
            service._llm_gateway = _make_llm_gateway(
                content=f"回答-{i} [1]"
            )
            await service.answer(
                query=f"问题-{i}",
                user_id="u-1",
                allowed_space_ids=["space-A"],
                conversation_id="conv-cap",
            )

        # ConversationService 强制最多 20 轮 = 40 条消息
        history = await conversation_service.get_history("conv-cap")
        assert len(history) == MAX_TURNS * 2 == 40
        # 最旧的应当是第 5 轮(下标 5)的 user
        assert history[0] == {"role": "user", "content": "问题-5"}
        # 最新的应当是第 24 轮(下标 24)的 assistant
        assert history[-1] == {"role": "assistant", "content": "回答-24 [1]"}

    @pytest.mark.asyncio
    async def test_capacity_limit_prompt_only_sees_recent_history(
        self, conversation_service
    ):
        """触发上限后,下一轮 Prompt 中也不应出现已被驱逐的最旧轮次。"""
        results = [_make_search_result(chunk_id="c-1", score=0.9)]
        search = _make_search_service(results)
        service = RAGService(
            search_service=search,
            llm_gateway=_make_llm_gateway(),
            conversation_service=conversation_service,
        )

        # 先填 21 轮:第 1 轮("问题-0")会被挤出
        for i in range(21):
            service._llm_gateway = _make_llm_gateway(content=f"回答-{i}")
            await service.answer(
                query=f"问题-{i}",
                user_id="u-1",
                allowed_space_ids=["space-A"],
                conversation_id="conv-cap2",
            )

        # 第 22 轮:Prompt 中应不再包含"问题-0"
        next_llm = _make_llm_gateway(content="新答案")
        service._llm_gateway = next_llm
        await service.answer(
            query="第二十二轮问题",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-cap2",
        )

        prompt = next_llm.complete.await_args.kwargs["prompt"]
        # 历史段一定存在
        assert "对话历史" in prompt
        # 已被驱逐的最旧消息不应出现
        assert "问题-0" not in prompt
        assert "回答-0" not in prompt
        # 但近期消息(如第 20 轮)应仍在
        assert "问题-20" in prompt or "回答-20" in prompt


# ─── 3. 错误恢复 ───────────────────────────────────────────────────────


class TestErrorRecovery:
    """LLM 失败后,后续请求能正常恢复;失败请求不污染会话历史。"""

    @pytest.mark.asyncio
    async def test_llm_timeout_then_normal_recovery(
        self, conversation_service
    ):
        """LLM 超时一次 → 抛 RAGServiceError;再调用 → 成功且历史只含成功的轮次。"""
        results = [_make_search_result(chunk_id="c-1", score=0.9)]
        search = _make_search_service(results)

        # 第一次:LLM 抛 timeout
        failing_llm = AsyncMock()
        failing_llm.complete = AsyncMock(
            side_effect=LLMGatewayError(
                "LLM call timed out", reason="timeout"
            )
        )
        service = RAGService(
            search_service=search,
            llm_gateway=failing_llm,
            conversation_service=conversation_service,
        )

        with pytest.raises(RAGServiceError) as exc_info:
            await service.answer(
                query="第一次问题",
                user_id="u-1",
                allowed_space_ids=["space-A"],
                conversation_id="conv-recover",
            )
        assert exc_info.value.reason == "timeout"

        # 失败时不写历史
        history = await conversation_service.get_history("conv-recover")
        assert history == [], "LLM 失败时不应当把任何消息写入会话历史"

        # 第二次:LLM 恢复正常
        good_llm = _make_llm_gateway(content="恢复后的答案 [1]")
        service._llm_gateway = good_llm

        result = await service.answer(
            query="重试问题",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-recover",
        )

        assert result.answer == "恢复后的答案 [1]"
        # 现在历史只含成功这一轮
        history = await conversation_service.get_history("conv-recover")
        assert [m["content"] for m in history] == [
            "重试问题", "恢复后的答案 [1]",
        ]
        # 失败那次的 query 不应出现在历史里
        assert all("第一次问题" != m["content"] for m in history)

    @pytest.mark.asyncio
    async def test_streaming_first_token_timeout_then_recovery(
        self, conversation_service
    ):
        """流式首 token 超时 → error 事件;再调用 → 流式成功完成。"""
        import asyncio

        results = [_make_search_result(chunk_id="c-1", score=0.9)]
        search = _make_search_service(results)

        # 第一次:首 token 永远不到达
        async def _never_yield(*_args, **_kwargs):
            await asyncio.sleep(1.0)
            yield "too-late"

        slow_gateway = MagicMock()
        slow_gateway.stream = _never_yield

        service = RAGService(
            search_service=search,
            llm_gateway=slow_gateway,
            conversation_service=conversation_service,
        )

        events = []
        async for ev in service.answer_stream(
            query="慢请求",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            first_token_timeout=0.05,
            conversation_id="conv-stream-recover",
        ):
            events.append(ev)

        assert len(events) == 1
        assert events[0].event == STREAM_EVENT_ERROR
        assert events[0].data["code"] == "first_token_timeout"
        # 失败时不应写入会话历史
        assert await conversation_service.get_history("conv-stream-recover") == []

        # 第二次:恢复正常
        good_gateway = _make_streaming_gateway(["恢复 [", "1", "]"])
        service._llm_gateway = good_gateway

        events = []
        async for ev in service.answer_stream(
            query="再次提问",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-stream-recover",
        ):
            events.append(ev)

        # 应有 token + sources + done
        kinds = [e.event for e in events]
        assert STREAM_EVENT_TOKEN in kinds
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]

        # 历史只有这一轮成功的内容
        history = await conversation_service.get_history(
            "conv-stream-recover"
        )
        assert [m["content"] for m in history] == [
            "再次提问", "恢复 [1]",
        ]


# ─── 4. 权限过滤 ───────────────────────────────────────────────────────


class TestPermissionFiltering:
    """用户的 ``allowed_space_ids`` 必须如实传到 SearchService;不同用户彼此隔离。"""

    @pytest.mark.asyncio
    async def test_allowed_space_ids_passed_through_to_search(
        self, conversation_service
    ):
        """RAGService 不应当修改 ``allowed_space_ids``,原样透传给 SearchService。"""
        # 让 SearchService 根据 allowed_space_ids 返回不同结果——此处仅
        # 用来做副作用断言。
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return [
                _make_search_result(
                    chunk_id="c-1",
                    score=0.9,
                    document_id="doc-from-" + ",".join(
                        kwargs["allowed_space_ids"]
                    ),
                )
            ]

        search = _make_search_service(_capture)
        service = RAGService(
            search_service=search,
            llm_gateway=_make_llm_gateway(),
            conversation_service=conversation_service,
        )

        await service.answer(
            query="带权限的问题",
            user_id="u-1",
            allowed_space_ids=["space-A", "space-B", "space-C"],
            top_k=5,
            conversation_id="conv-perm",
        )

        # 透传完整、顺序保留
        assert captured["user_id"] == "u-1"
        assert captured["allowed_space_ids"] == [
            "space-A", "space-B", "space-C"
        ]

    @pytest.mark.asyncio
    async def test_users_are_isolated_by_allowed_space_ids(
        self, conversation_service, fake_redis_client
    ):
        """两个用户用各自 allowed_space_ids 提问,SearchService 收到的过滤参数互不干扰。

        同时验证不同 conversation_id 之间的会话历史也是隔离的——这是权限场景里
        前端常见的真实诉求(用户 A 不应看到用户 B 的对话)。
        """
        captured_calls: list[dict] = []

        def _by_user(**kwargs):
            captured_calls.append({
                "user_id": kwargs["user_id"],
                "allowed_space_ids": list(kwargs["allowed_space_ids"]),
            })
            # 仅返回与当前用户匹配的"虚拟"chunk
            uid = kwargs["user_id"]
            return [
                _make_search_result(
                    chunk_id=f"chunk-{uid}",
                    score=0.9,
                    document_id=f"doc-{uid}",
                    source_file=f"{uid}.pdf",
                    highlight=f"属于 {uid} 的内容",
                )
            ]

        search = _make_search_service(_by_user)
        service = RAGService(
            search_service=search,
            llm_gateway=_make_llm_gateway(content="A 的答案 [1]"),
            conversation_service=conversation_service,
        )

        # 用户 A 提问(只能访问 space-A)
        await service.answer(
            query="A 的问题",
            user_id="user-A",
            allowed_space_ids=["space-A"],
            conversation_id="conv-A",
        )

        # 切换 LLM 响应,用户 B 提问(只能访问 space-B)
        service._llm_gateway = _make_llm_gateway(content="B 的答案 [1]")
        await service.answer(
            query="B 的问题",
            user_id="user-B",
            allowed_space_ids=["space-B"],
            conversation_id="conv-B",
        )

        # 1) 两次 SearchService 调用各自携带正确的过滤参数
        assert captured_calls == [
            {"user_id": "user-A", "allowed_space_ids": ["space-A"]},
            {"user_id": "user-B", "allowed_space_ids": ["space-B"]},
        ]

        # 2) 会话历史在 Redis 中真正隔离
        history_a = await conversation_service.get_history("conv-A")
        history_b = await conversation_service.get_history("conv-B")
        assert [m["content"] for m in history_a] == [
            "A 的问题", "A 的答案 [1]"
        ]
        assert [m["content"] for m in history_b] == [
            "B 的问题", "B 的答案 [1]"
        ]
        # 两个 key 各自独立存在
        assert await fake_redis_client.exists(f"{KEY_PREFIX}conv-A") == 1
        assert await fake_redis_client.exists(f"{KEY_PREFIX}conv-B") == 1

    @pytest.mark.asyncio
    async def test_empty_allowed_space_ids_returns_no_context(
        self, conversation_service
    ):
        """``allowed_space_ids`` 为空时(无权限),SearchService 通常返回空——
        RAGService 应当走"未找到相关内容"分支,且不调用 LLM。"""
        # 模拟 SearchService 在无权限时返回空
        search = _make_search_service([])
        llm = _make_llm_gateway()
        service = RAGService(
            search_service=search,
            llm_gateway=llm,
            conversation_service=conversation_service,
        )

        result = await service.answer(
            query="无权限用户的问题",
            user_id="u-no-access",
            allowed_space_ids=[],
            conversation_id="conv-no-access",
        )

        # 走固定提示分支
        assert result.answer == NO_CONTEXT_MESSAGE
        assert result.sources == []
        assert result.usage == {}
        # 不调用 LLM
        llm.complete.assert_not_awaited()
        # 检索为空时不写会话历史(避免污染)
        assert await conversation_service.get_history("conv-no-access") == []


# ─── 5. 端到端额外校验:cited 与 source 字段在序列化里完整呈现 ────────


class TestStreamingPayloadStructure:
    """流式 ``sources`` 事件的 JSON 载荷结构稳定可序列化。"""

    @pytest.mark.asyncio
    async def test_sources_event_payload_is_json_serializable(
        self, conversation_service
    ):
        """``sources`` 事件中的每条 source 都能被 ``json.dumps`` 处理。"""
        results = [
            _make_search_result(chunk_id="c-1", score=0.9),
            _make_search_result(chunk_id="c-2", score=0.7),
        ]
        gateway = _make_streaming_gateway(["仅 [", "2", "] 被引用"])
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=gateway,
            conversation_service=conversation_service,
        )

        events = []
        async for ev in service.answer_stream(
            query="问",
            user_id="u-1",
            allowed_space_ids=["space-A"],
            conversation_id="conv-json",
        ):
            events.append(ev)

        sources_event = next(e for e in events if e.event == STREAM_EVENT_SOURCES)
        # 完整 JSON 序列化不应抛异常
        payload = json.dumps(sources_event.data, ensure_ascii=False)
        # cited 字段必须出现在序列化里
        assert "\"cited\": true" in payload
        assert "\"cited\": false" in payload
        assert "c-1" in payload and "c-2" in payload
