"""任务 16.4：引用来源标注（解析答案中的 ``[n]``）单元测试。

覆盖点：

- ``SYSTEM_PROMPT`` 显式要求 LLM 使用 ``[n]`` 标注引用
- ``RAGService.parse_citations`` 正确从答案中提取编号集合
- 非流式 ``answer``：sources 列表中的 ``cited`` 字段反映真实引用
- 越界编号（如 ``[99]``）不污染输出，cited 仅基于实际范围 1..K 判定
- 流式 ``answer_stream``：最终 sources 事件中的 source dict 含 ``cited``
- 端到端 ``POST /api/qa/ask``：返回的 sources 包含 ``cited`` 字段
"""

from __future__ import annotations

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
from app.services.llm_gateway import LLMResponse
from app.services.rag_service import (
    STREAM_EVENT_DONE,
    STREAM_EVENT_SOURCES,
    STREAM_EVENT_TOKEN,
    SYSTEM_PROMPT,
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


def _make_llm_gateway(content: str) -> AsyncMock:
    """构造返回固定文本的 mock LLMGateway（非流式）。"""
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


def _make_streaming_gateway(tokens: list[str]) -> MagicMock:
    """构造按顺序产出 tokens 的 mock LLMGateway（流式）。"""
    gateway = MagicMock()

    async def _stream(*_args, **_kwargs) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    gateway.stream = _stream
    return gateway


async def _collect(stream) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for ev in stream:
        events.append(ev)
    return events


# ─── 1. System Prompt 必须显式要求 [n] 标注 ───────────────────────────


class TestSystemPromptInstructsCitationFormat:
    """需求 8.4：System Prompt 必须明确要求 LLM 使用 [n] 形式标注引用。"""

    def test_system_prompt_mentions_bracket_number_format(self):
        # 必须出现 [1] 与 [2] 等示例，且强调"标注"语义
        assert "[1]" in SYSTEM_PROMPT
        assert "[2]" in SYSTEM_PROMPT
        assert ("标注" in SYSTEM_PROMPT) or ("引用" in SYSTEM_PROMPT)
        # 必须出现 [n] 形式的占位符说明，使指令一般化
        assert "[n]" in SYSTEM_PROMPT


# ─── 2. parse_citations 行为 ─────────────────────────────────────────


class TestParseCitations:
    """``RAGService.parse_citations`` 静态方法的单元测试。"""

    def test_extracts_multiple_unique_indices(self):
        result = RAGService.parse_citations("根据资料 [1] 和 [2][3]，结论...")
        assert result == {1, 2, 3}

    def test_returns_empty_for_plain_text(self):
        assert RAGService.parse_citations("纯文本无引用") == set()

    def test_handles_multi_digit_indices(self):
        assert RAGService.parse_citations("[1] [10] [25]") == {1, 10, 25}

    def test_deduplicates_repeated_indices(self):
        # 重复引用同一编号只统计一次
        assert RAGService.parse_citations("[1][1] foo [1]") == {1}

    def test_ignores_non_numeric_brackets(self):
        # 非纯数字方括号不应误命中
        assert RAGService.parse_citations("[abc] [1a] [a1]") == set()

    def test_empty_string_returns_empty_set(self):
        assert RAGService.parse_citations("") == set()


# ─── 3. 非流式 answer：cited 字段 ─────────────────────────────────────


class TestAnswerCitedField:
    """非流式 ``answer`` 应基于 LLM 回答标记每个 source 的 ``cited``。"""

    @pytest.mark.asyncio
    async def test_cited_marks_only_referenced_sources(self):
        # 三个候选 chunk，但答案只引用了 [1] 和 [3]
        results = [
            _make_result(chunk_id="c1"),
            _make_result(chunk_id="c2"),
            _make_result(chunk_id="c3"),
        ]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway("结论 A [1]，结论 B [3]。"),
        )

        answer = await service.answer(
            query="Q", user_id="u", allowed_space_ids=["s"]
        )

        # 全部 source 都返回（保留全部上下文），但 cited 只标记被引用的
        cited_map = {s.index: s.cited for s in answer.sources}
        assert cited_map == {1: True, 2: False, 3: True}

    @pytest.mark.asyncio
    async def test_no_citations_marks_all_uncited(self):
        # 答案中完全没有 [n] 标注
        results = [_make_result(chunk_id="c1"), _make_result(chunk_id="c2")]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway("仅有的纯文本回答没有引用。"),
        )

        answer = await service.answer(
            query="Q", user_id="u", allowed_space_ids=["s"]
        )

        assert all(s.cited is False for s in answer.sources)
        # sources 仍按检索顺序保留
        assert [s.chunk_id for s in answer.sources] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_out_of_range_indices_are_ignored(self):
        # 答案引用了不存在的 [99]，不应影响输出
        results = [_make_result(chunk_id="c1"), _make_result(chunk_id="c2")]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway("根据 [1] 和 [99] 推断。"),
        )

        answer = await service.answer(
            query="Q", user_id="u", allowed_space_ids=["s"]
        )

        # [1] 命中实际范围；[99] 越界被忽略；[2] 没被引用
        cited_map = {s.index: s.cited for s in answer.sources}
        assert cited_map == {1: True, 2: False}
        # 不会因越界编号产生额外条目
        assert len(answer.sources) == 2


# ─── 4. 流式 answer_stream：sources 事件含 cited ───────────────────────


class TestAnswerStreamCitedField:
    """流式 ``answer_stream`` 在最终 sources 事件中应包含 ``cited`` 字段。"""

    @pytest.mark.asyncio
    async def test_sources_event_includes_cited_after_full_answer(self):
        # 把 [2] 拆分到不同 token，验证基于"完整答案"而非单 token 解析
        results = [
            _make_result(chunk_id="c1"),
            _make_result(chunk_id="c2"),
            _make_result(chunk_id="c3"),
        ]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_streaming_gateway(
                ["先看 [", "1", "]，再看 [", "2", "]。"]
            ),
        )

        events = await _collect(
            service.answer_stream(
                query="Q", user_id="u", allowed_space_ids=["s"]
            )
        )

        # 顺序应为多 token + sources + done
        kinds = [e.event for e in events]
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]

        sources_event = next(e for e in events if e.event == STREAM_EVENT_SOURCES)
        sources = sources_event.data["sources"]
        cited_map = {s["index"]: s["cited"] for s in sources}
        assert cited_map == {1: True, 2: True, 3: False}
        # 全部条目都必须含 cited 字段
        assert all("cited" in s for s in sources)

    @pytest.mark.asyncio
    async def test_stream_out_of_range_citation_does_not_crash(self):
        # 流式场景下越界编号同样应被忽略
        results = [_make_result(chunk_id="c1")]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_streaming_gateway(["参 [", "99", "]"]),
        )

        events = await _collect(
            service.answer_stream(
                query="Q", user_id="u", allowed_space_ids=["s"]
            )
        )

        sources_event = next(e for e in events if e.event == STREAM_EVENT_SOURCES)
        sources = sources_event.data["sources"]
        assert len(sources) == 1
        assert sources[0]["index"] == 1
        assert sources[0]["cited"] is False


# ─── 5. 端到端：POST /api/qa/ask 返回 cited ───────────────────────────


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


class TestQAAskRouteCitedField:
    """``POST /api/qa/ask`` 返回 sources 必须包含 ``cited`` 字段。"""

    def test_non_streaming_route_returns_cited_per_source(self):
        results = [
            _make_result(chunk_id="c1"),
            _make_result(chunk_id="c2"),
            _make_result(chunk_id="c3"),
        ]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_llm_gateway("根据 [1] 和 [3] 总结。"),
        )
        app = _make_app(service)
        client = TestClient(app)

        resp = client.post("/api/qa/ask", json={"question": "Q"})
        assert resp.status_code == 200
        body = resp.json()

        cited_map = {s["index"]: s["cited"] for s in body["sources"]}
        assert cited_map == {1: True, 2: False, 3: True}

    def test_streaming_route_sources_event_has_cited(self):
        results = [
            _make_result(chunk_id="c1"),
            _make_result(chunk_id="c2"),
        ]
        service = RAGService(
            search_service=_make_search_service(results),
            llm_gateway=_make_streaming_gateway(["仅引用 [", "2", "]"]),
        )
        app = _make_app(service)
        client = TestClient(app)

        with client.stream(
            "POST", "/api/qa/ask/stream", json={"question": "Q"}
        ) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        kinds = [n for n, _ in events]
        assert kinds[-2:] == [STREAM_EVENT_SOURCES, STREAM_EVENT_DONE]
        # token 事件至少 1 条
        assert kinds.count(STREAM_EVENT_TOKEN) >= 1

        sources = next(d for n, d in events if n == STREAM_EVENT_SOURCES)["sources"]
        cited_map = {s["index"]: s["cited"] for s in sources}
        assert cited_map == {1: False, 2: True}
