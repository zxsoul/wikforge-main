"""任务 16.3：流式 RAG 输出（SSE）单元测试。

覆盖点：

- ``RAGService.answer_stream`` 异步生成器逐 token 产出
- 流结束后产出 ``sources`` + ``done`` 事件
- 检索为空时短路：固定提示 + 空 ``sources``
- LLM 失败 → ``error`` 事件并提前结束
- 首 token 超时 → ``error(code=first_token_timeout)``
- ``POST /api/qa/ask/stream`` 返回 ``text/event-stream`` 且按 SSE 格式编码
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.qa import get_rag_service
from app.api.qa import router as qa_router
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.services.llm_gateway import LLMGatewayError
from app.services.rag_service import (
    NO_CONTEXT_MESSAGE,
    STREAM_EVENT_DONE,
    STREAM_EVENT_ERROR,
    STREAM_EVENT_SOURCES,
    STREAM_EVENT_TOKEN,
    RAGService,
    StreamEvent,
)
from app.services.search_service import SearchResponse, SearchResult

# ─── Helpers ───────────────────────────────────────────────────────────


def _make_result(
    *,
    chunk_id: str = "c1",
    document_id: str = "d1",
    title_chain: str = "章节1 > 章节2",
    source_file: str = "doc.pdf",
    page_number: int = 1,
    score: float = 0.8,
    highlight: str = "片段",
    chunk_index: int = 0,
) -> SearchResult:
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
    """构造返回固定 results 的 mock SearchService。"""
    service = AsyncMock()
    service.search = AsyncMock(
        return_value=SearchResponse(
            results=results,
            total=len(results),
            page=1,
            page_size=max(1, len(results)),
        )
    )
    return service


def _make_streaming_gateway(
    tokens: list[str],
    *,
    first_token_delay: float = 0.0,
    error_after: int | None = None,
    error: LLMGatewayError | None = None,
) -> MagicMock:
    """构造一个支持 ``stream`` 异步生成的 mock LLMGateway。

    Args:
        tokens: 依次产出的 token 列表
        first_token_delay: 首 token 前的延迟（秒），用于触发首 token 超时
        error_after: 在产出第 N 个 token 后抛出 ``error`` （从 0 开始）
        error: 要抛出的 LLMGatewayError；缺省时构造 timeout reason
    """
    gateway = MagicMock()

    async def _stream(*args, **kwargs) -> AsyncIterator[str]:
        produced = 0
        for i, token in enumerate(tokens):
            if i == 0 and first_token_delay > 0:
                await asyncio.sleep(first_token_delay)
            if error_after is not None and produced == error_after:
                raise error or LLMGatewayError("stream broken", reason="timeout")
            yield token
            produced += 1
        # tokens 耗尽后仍可能要抛错（适配 error_after >= len(tokens) 的用例）
        if error_after is not None and produced == error_after:
            raise error or LLMGatewayError("stream broken", reason="timeout")

    gateway.stream = _stream
    return gateway


async def _collect(stream) -> list[StreamEvent]:
    """把 ``answer_stream`` 异步生成器收集成事件列表。"""
    events: list[StreamEvent] = []
    async for ev in stream:
        events.append(ev)
    return events


# ─── 服务层：正常 token 流 ─────────────────────────────────────────────


class TestAnswerStreamHappyPath:
    """正常路径：逐 token 产出 + 末尾 sources + done。"""

    @pytest.mark.asyncio
    async def test_yields_each_token_in_order(self):
        results = [_make_result(chunk_id="c1"), _make_result(chunk_id="c2")]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_streaming_gateway(["第一", "段", "答案"]),
        )

        events = await _collect(
            service.answer_stream(
                query="问题", user_id="u", allowed_space_ids=["s"]
            )
        )

        token_events = [e for e in events if e.event == STREAM_EVENT_TOKEN]
        assert [e.data["text"] for e in token_events] == ["第一", "段", "答案"]

    @pytest.mark.asyncio
    async def test_sources_event_emitted_after_tokens(self):
        results = [
            _make_result(
                chunk_id="c1",
                document_id="d1",
                title_chain="第一章",
                source_file="a.pdf",
                page_number=2,
                score=0.91,
            )
        ]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_streaming_gateway(["ans"]),
        )

        events = await _collect(
            service.answer_stream(
                query="Q", user_id="u", allowed_space_ids=["s"]
            )
        )

        # 顺序：先 token，再 sources，最后 done
        kinds = [e.event for e in events]
        assert kinds.count(STREAM_EVENT_TOKEN) == 1
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]

        sources_event = next(e for e in events if e.event == STREAM_EVENT_SOURCES)
        sources = sources_event.data["sources"]
        assert len(sources) == 1
        src = sources[0]
        assert src["index"] == 1
        assert src["chunk_id"] == "c1"
        assert src["document_id"] == "d1"
        assert src["title_chain"] == "第一章"
        assert src["source_file"] == "a.pdf"
        assert src["page_number"] == 2
        assert src["score"] == pytest.approx(0.91)


# ─── 检索为空 ─────────────────────────────────────────────────────────


class TestAnswerStreamEmptyRetrieval:
    """检索结果为空时不调用 LLM，直接给出固定提示。"""

    @pytest.mark.asyncio
    async def test_no_context_message_and_empty_sources(self):
        gateway = _make_streaming_gateway(["不应被调用"])
        # 用 spy 替换 stream，确保它不会被触发
        gateway.stream = MagicMock(
            side_effect=AssertionError("LLM should not be called")
        )
        service = RAGService(
            search_service=_make_search_service([]),
            llm_gateway=gateway,
        )

        events = await _collect(
            service.answer_stream(
                query="冷僻问题", user_id="u", allowed_space_ids=["s"]
            )
        )

        assert [e.event for e in events] == [
            STREAM_EVENT_TOKEN,
            STREAM_EVENT_SOURCES,
            STREAM_EVENT_DONE,
        ]
        assert events[0].data == {"text": NO_CONTEXT_MESSAGE}
        assert events[1].data == {"sources": []}
        assert events[2].data == {}


# ─── LLM 失败 ─────────────────────────────────────────────────────────


class TestAnswerStreamLLMFailure:
    """LLM 错误必须转成 ``error`` 事件，且不再产出后续 sources/done。"""

    @pytest.mark.asyncio
    async def test_first_token_timeout_emits_error(self):
        # 首 token 延迟 0.5s，调用方设置 first_token_timeout=0.05s 必触发
        gateway = _make_streaming_gateway(
            ["should-not-arrive"], first_token_delay=0.5
        )
        service = RAGService(
            search_service=_make_search_service([_make_result()]),
            llm_gateway=gateway,
        )

        events = await _collect(
            service.answer_stream(
                query="Q",
                user_id="u",
                allowed_space_ids=["s"],
                first_token_timeout=0.05,
            )
        )

        assert len(events) == 1
        assert events[0].event == STREAM_EVENT_ERROR
        assert events[0].data["code"] == "first_token_timeout"
        assert "未返回" in events[0].data["message"]

    @pytest.mark.asyncio
    async def test_llm_error_mid_stream_emits_error(self):
        # 第 1 个 token 后抛 LLMGatewayError(reason="rate_limit")
        gateway = _make_streaming_gateway(
            ["ok"],
            error_after=1,
            error=LLMGatewayError("rate limited", reason="rate_limit"),
        )
        service = RAGService(
            search_service=_make_search_service([_make_result()]),
            llm_gateway=gateway,
        )

        events = await _collect(
            service.answer_stream(
                query="Q", user_id="u", allowed_space_ids=["s"]
            )
        )

        kinds = [e.event for e in events]
        # 第一个 token 已经产出，然后是 error；不会再有 sources/done
        assert kinds == [STREAM_EVENT_TOKEN, STREAM_EVENT_ERROR]
        assert events[1].data["code"] == "rate_limit"


# ─── SSE 路由 ──────────────────────────────────────────────────────────


def _make_app(rag_service: RAGService) -> FastAPI:
    """构造一个带 mock 鉴权 / DB / RAGService 的隔离 FastAPI app。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(qa_router)

    fake_user = MagicMock()
    fake_user.id = uuid.uuid4()

    async def _override_user():
        return fake_user

    db_session = AsyncMock()
    db_result = MagicMock()
    db_result.scalars.return_value.all.return_value = [str(uuid.uuid4())]
    db_session.execute = AsyncMock(return_value=db_result)

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    return app


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体拆成 ``[(event, data_dict), ...]``。"""
    parsed: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        data_str = "\n".join(data_lines) or "{}"
        parsed.append((event_name, json.loads(data_str)))
    return parsed


class TestQAStreamRoute:
    """``POST /api/qa/ask/stream`` 行为测试。"""

    def test_returns_event_stream_media_type(self):
        service = RAGService(
            search_service=_make_search_service([_make_result()]),
            llm_gateway=_make_streaming_gateway(["你", "好"]),
        )
        app = _make_app(service)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/api/qa/ask/stream",
            json={"question": "Hello"},
        ) as resp:
            assert resp.status_code == 200
            content_type = resp.headers["content-type"]
            assert content_type.startswith("text/event-stream")
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        kinds = [name for name, _ in events]

        # 必须包含至少一个 token、最后是 sources + done
        assert STREAM_EVENT_TOKEN in kinds
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]

        token_texts = [d["text"] for n, d in events if n == STREAM_EVENT_TOKEN]
        assert token_texts == ["你", "好"]

    def test_empty_retrieval_route_emits_no_context_message(self):
        service = RAGService(
            search_service=_make_search_service([]),
            llm_gateway=MagicMock(stream=MagicMock(
                side_effect=AssertionError("LLM should not be called")
            )),
        )
        app = _make_app(service)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/api/qa/ask/stream",
            json={"question": "x"},
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        assert events[0] == (STREAM_EVENT_TOKEN, {"text": NO_CONTEXT_MESSAGE})
        assert events[1] == (STREAM_EVENT_SOURCES, {"sources": []})
        assert events[2] == (STREAM_EVENT_DONE, {})

    def test_llm_error_route_emits_error_event(self):
        gateway = _make_streaming_gateway(
            [],
            error_after=0,
            error=LLMGatewayError("auth failed", reason="auth"),
        )
        service = RAGService(
            search_service=_make_search_service([_make_result()]),
            llm_gateway=gateway,
        )
        app = _make_app(service)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/api/qa/ask/stream",
            json={"question": "x"},
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        # 最后一个事件必为 error
        last_name, last_data = events[-1]
        assert last_name == STREAM_EVENT_ERROR
        assert last_data["code"] == "auth"
