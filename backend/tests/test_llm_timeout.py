"""任务 16.7：LLM 调用超时处理（默认 60 秒，失败返回错误提示）单元测试。

需求 8.7：
    IF LLM_Gateway 调用失败或响应超时（超时阈值为可配置参数，默认 60 秒），
    THEN THE RAG_Engine SHALL 向用户返回服务暂时不可用的错误提示，并保留
    当前对话上下文不丢失。

本测试模块从端到端覆盖该路径：

- ``LLMGateway`` 默认超时回落到 ``Settings.LLM_TIMEOUT``（默认 60 秒），
  且支持通过环境变量调整。
- ``LLMGateway`` 在 ``acompletion`` 长时间不返回时通过 ``asyncio.wait_for``
  转为 ``LLMGatewayError(reason="timeout")``。
- ``RAGService.answer`` 把 ``LLMGatewayError`` 映射为
  ``RAGServiceError`` 且复用相同的 ``reason``。
- ``POST /api/qa/ask`` 在 ``RAGServiceError`` 出现时不抛 5xx，而是返回
  中文友好提示，且不暴露原始英文异常文本。
- ``POST /api/qa/ask/stream`` 在 LLM 错误时 SSE ``error`` 事件的
  ``message`` 字段同样替换为中文友好提示，``code`` 保留原始 reason。
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.qa import (
    _DEFAULT_LLM_FAILURE_MESSAGE,
    _LLM_FAILURE_MESSAGES,
    get_rag_service,
)
from app.api.qa import router as qa_router
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.services.llm_gateway import (
    DEFAULT_LLM_TIMEOUT,
    LLMGateway,
    LLMGatewayError,
)
from app.services.rag_service import (
    RAGService,
    RAGServiceError,
    StreamEvent,
)
from app.services.search_service import SearchResponse, SearchResult


# ─── 公用工具 ───────────────────────────────────────────────────────────


@pytest.fixture
def litellm_stub():
    """把 ``litellm`` 替换为可控 stub，避免依赖真实 LLM。"""
    original = sys.modules.get("litellm")
    stub = types.ModuleType("litellm")
    stub.acompletion = AsyncMock()
    sys.modules["litellm"] = stub
    try:
        yield stub
    finally:
        if original is not None:
            sys.modules["litellm"] = original
        else:
            sys.modules.pop("litellm", None)


def _make_search_response(results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(
        results=results,
        total=len(results),
        page=1,
        page_size=max(1, len(results)),
    )


def _make_result(score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        document_id="d1",
        chunk_index=0,
        title_chain="第一章",
        source_file="doc.pdf",
        page_number=1,
        score=score,
        highlight="片段",
    )


# ─── 1. LLMGateway 默认 timeout 来自 Settings ─────────────────────────


class TestLLMGatewayDefaultTimeout:
    """需求 8.7：超时阈值默认 60 秒，可配置。"""

    @patch("app.services.llm_gateway.get_settings")
    def test_default_timeout_is_60_seconds(self, mock_settings):
        """未传入 ``timeout`` 时，默认值为 60.0 秒。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="",
            LITELLM_API_KEY="",
            LLM_TIMEOUT=60.0,
        )

        gateway = LLMGateway()

        assert gateway.timeout == 60.0
        assert DEFAULT_LLM_TIMEOUT == 60.0

    @patch("app.services.llm_gateway.get_settings")
    def test_settings_override_default(self, mock_settings):
        """``Settings.LLM_TIMEOUT`` 可调整默认值。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="",
            LITELLM_API_KEY="",
            LLM_TIMEOUT=15.0,
        )

        gateway = LLMGateway()

        assert gateway.timeout == 15.0

    @patch("app.services.llm_gateway.get_settings")
    def test_explicit_timeout_overrides_settings(self, mock_settings):
        """显式构造参数始终优先于 Settings。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="",
            LITELLM_API_KEY="",
            LLM_TIMEOUT=60.0,
        )

        gateway = LLMGateway(timeout=5.0)

        assert gateway.timeout == 5.0

    @patch("app.services.llm_gateway.get_settings")
    def test_falls_back_to_default_when_settings_missing(self, mock_settings):
        """Settings 缺少 ``LLM_TIMEOUT`` 字段时退回模块默认 60 秒。

        测试场景：旧测试用 ``MagicMock(LITELLM_MODEL=...)`` 构造 settings 时
        没列出 ``LLM_TIMEOUT``，``MagicMock`` 默认会返回一个新的 MagicMock，
        ``float(...)`` 会抛错——LLMGateway 必须优雅退化为 60 秒，避免破坏
        现有调用方。
        """
        # 故意不提供 LLM_TIMEOUT，但 MagicMock 会返回一个 MagicMock 对象，
        # 模拟"该字段不存在"的最坏情况。
        bad_settings = MagicMock(spec=["LITELLM_MODEL", "LITELLM_API_BASE", "LITELLM_API_KEY"])
        bad_settings.LITELLM_MODEL = "gpt-4o"
        bad_settings.LITELLM_API_BASE = ""
        bad_settings.LITELLM_API_KEY = ""
        mock_settings.return_value = bad_settings

        gateway = LLMGateway()

        assert gateway.timeout == DEFAULT_LLM_TIMEOUT


# ─── 2. LLMGateway 超时映射为 LLMGatewayError(reason="timeout") ──────


class TestLLMGatewayTimeoutMapping:
    """``acompletion`` 长时间无返回 → ``LLMGatewayError(reason="timeout")``。"""

    @pytest.mark.asyncio
    async def test_complete_timeout_maps_to_timeout_reason(self, litellm_stub):
        """``complete`` 超时必须抛 ``LLMGatewayError(reason="timeout")``。"""
        async def _hang(**_kwargs):
            await asyncio.sleep(10)

        litellm_stub.acompletion.side_effect = _hang

        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
                LLM_TIMEOUT=0.05,
            )
            gateway = LLMGateway()

        with pytest.raises(LLMGatewayError) as exc_info:
            await gateway.complete(prompt="hi")

        assert exc_info.value.reason == "timeout"
        # 任务 16.7：错误信息至少包含超时秒数，便于排查。
        assert "0.05" in str(exc_info.value) or "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_stream_timeout_maps_to_timeout_reason(self, litellm_stub):
        """``stream`` 在首 chunk 之前超时也必须映射为 ``timeout``。"""
        async def _hang(**_kwargs):
            await asyncio.sleep(10)

        litellm_stub.acompletion.side_effect = _hang

        with patch("app.services.llm_gateway.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                LITELLM_API_BASE="",
                LITELLM_API_KEY="",
                LLM_TIMEOUT=0.05,
            )
            gateway = LLMGateway()

        with pytest.raises(LLMGatewayError) as exc_info:
            async for _ in gateway.stream(prompt="hi"):
                pass

        assert exc_info.value.reason == "timeout"


# ─── 3. RAGService.answer 把 timeout 映射为 RAGServiceError ───────────


class TestRAGServiceTimeoutPropagation:
    """LLM 超时必须沿用 reason 抛 ``RAGServiceError``，不应吞掉。"""

    @pytest.mark.asyncio
    async def test_answer_propagates_timeout_reason(self):
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=_make_search_response([_make_result()])
        )
        gateway = AsyncMock()
        gateway.complete = AsyncMock(
            side_effect=LLMGatewayError(
                "LLM call timed out after 60 seconds", reason="timeout"
            )
        )
        service = RAGService(
            search_service=search, llm_gateway=gateway
        )

        with pytest.raises(RAGServiceError) as exc_info:
            await service.answer(
                query="问题", user_id="u", allowed_space_ids=["s"]
            )

        assert exc_info.value.reason == "timeout"


# ─── 4. /api/qa/ask 在 timeout 下返回中文友好提示 ─────────────────────


def _make_app(rag_service: RAGService) -> FastAPI:
    """构造一个隔离的 FastAPI app，注入 mock 鉴权 / DB / RAGService。"""
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


def _make_failing_rag_service(reason: str) -> RAGService:
    """构造一个 LLM 一定失败的 RAGService（绕过实例方法直接 patch）。"""
    search = AsyncMock()
    search.search = AsyncMock(
        return_value=_make_search_response([_make_result()])
    )
    gateway = AsyncMock()
    gateway.complete = AsyncMock(
        side_effect=LLMGatewayError(f"raw english error: {reason}", reason=reason)
    )
    return RAGService(search_service=search, llm_gateway=gateway)


class TestQAAskRouteFriendlyError:
    """``POST /api/qa/ask`` 在 LLM 失败时返回中文友好提示。"""

    @pytest.mark.parametrize(
        "reason, expected_message",
        [
            ("timeout", _LLM_FAILURE_MESSAGES["timeout"]),
            ("rate_limit", _LLM_FAILURE_MESSAGES["rate_limit"]),
            ("auth", _LLM_FAILURE_MESSAGES["auth"]),
            ("model_unavailable", _LLM_FAILURE_MESSAGES["model_unavailable"]),
            ("unknown", _DEFAULT_LLM_FAILURE_MESSAGE),
            ("anything_else", _DEFAULT_LLM_FAILURE_MESSAGE),
        ],
    )
    def test_returns_chinese_friendly_message(
        self, reason: str, expected_message: str
    ):
        """各 ``reason`` 都应映射到中文友好提示，且不暴露原始英文异常。"""
        service = _make_failing_rag_service(reason)
        app = _make_app(service)
        client = TestClient(app)

        resp = client.post("/api/qa/ask", json={"question": "Q"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == expected_message
        assert body["sources"] == []
        assert body["usage"] == {}
        # 不应回显原始英文异常文本
        assert "raw english error" not in body["answer"]

    def test_timeout_message_mentions_retry(self):
        """``timeout`` 文案应当至少提示用户稍后重试。"""
        service = _make_failing_rag_service("timeout")
        app = _make_app(service)
        client = TestClient(app)

        resp = client.post("/api/qa/ask", json={"question": "Q"})

        assert resp.status_code == 200
        assert "重试" in resp.json()["answer"]

    def test_response_does_not_drop_request_payload(self):
        """LLM 失败不应让请求本身被拒（仍是 200，而非 5xx）。"""
        service = _make_failing_rag_service("timeout")
        app = _make_app(service)
        client = TestClient(app)

        resp = client.post(
            "/api/qa/ask",
            json={"question": "Q", "top_k": 3},
        )

        # 需求 8.7：仍能正常响应，"保留当前对话上下文不丢失"
        assert resp.status_code == 200


# ─── 5. SSE error 事件 message 也是中文友好提示 ───────────────────────


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


def _make_streaming_failing_service(reason: str) -> RAGService:
    """构造一个 stream 立即抛错的 RAGService。"""
    search = AsyncMock()
    search.search = AsyncMock(
        return_value=_make_search_response([_make_result()])
    )

    async def _stream(*_args, **_kwargs):
        raise LLMGatewayError(f"raw english stream error: {reason}", reason=reason)
        yield  # pragma: no cover - 让函数成为 async generator

    gateway = MagicMock()
    gateway.stream = _stream
    return RAGService(search_service=search, llm_gateway=gateway)


class TestQAAskStreamRouteFriendlyError:
    """SSE 流式错误事件的 ``message`` 字段也是中文，``code`` 保留 reason。"""

    @pytest.mark.parametrize(
        "reason, expected_message",
        [
            ("timeout", _LLM_FAILURE_MESSAGES["timeout"]),
            ("rate_limit", _LLM_FAILURE_MESSAGES["rate_limit"]),
            ("auth", _LLM_FAILURE_MESSAGES["auth"]),
            ("model_unavailable", _LLM_FAILURE_MESSAGES["model_unavailable"]),
            ("unknown", _DEFAULT_LLM_FAILURE_MESSAGE),
        ],
    )
    def test_sse_error_message_is_chinese(
        self, reason: str, expected_message: str
    ):
        service = _make_streaming_failing_service(reason)
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
        assert events, "至少应当产出一个 SSE 事件"
        last_name, last_data = events[-1]
        assert last_name == "error"
        assert last_data["code"] == reason
        assert last_data["message"] == expected_message
        assert "raw english" not in last_data["message"]

    def test_first_token_timeout_uses_timeout_message(self):
        """首 token 超时同样映射到 timeout 文案，让用户感知一致。"""
        # RAGService.answer_stream 在首 token 超过 first_token_timeout 时
        # 会自行产出 ``error(code=first_token_timeout)``，而非由 LLMGateway
        # 抛 LLMGatewayError。路由层需要把 ``first_token_timeout`` 也归类
        # 到"超时"提示，给终端用户一致的体验。
        search = AsyncMock()
        search.search = AsyncMock(
            return_value=_make_search_response([_make_result()])
        )

        async def _slow_stream(*_args, **_kwargs):
            await asyncio.sleep(0.5)
            yield "too-late"

        gateway = MagicMock()
        gateway.stream = _slow_stream

        # 注入一个 first_token_timeout 极短的 RAGService 子类，避免改动
        # 真实 FIRST_TOKEN_TIMEOUT_SECONDS。
        class _FastTimeoutRAG(RAGService):
            async def answer_stream(self, *args, **kwargs):  # type: ignore[override]
                kwargs["first_token_timeout"] = 0.05
                async for event in super().answer_stream(*args, **kwargs):
                    yield event

        service = _FastTimeoutRAG(
            search_service=search, llm_gateway=gateway
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
        last_name, last_data = events[-1]
        assert last_name == "error"
        # code 保留原始 first_token_timeout，供前端按"超时类"处理
        assert last_data["code"] == "first_token_timeout"
        # message 使用 timeout 文案，统一用户感知
        assert last_data["message"] == _LLM_FAILURE_MESSAGES["timeout"]


# ─── 6. StreamEvent 不变（防御回归） ─────────────────────────────────


class TestStreamEventUnchanged:
    """StreamEvent dataclass 字段稳定，避免无意中破坏其他消费者。"""

    def test_stream_event_has_expected_fields(self):
        ev = StreamEvent(event="error", data={"code": "x", "message": "y"})
        assert ev.event == "error"
        assert ev.data == {"code": "x", "message": "y"}
