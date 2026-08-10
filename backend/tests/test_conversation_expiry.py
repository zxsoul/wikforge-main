"""会话过期逻辑测试（任务 16.8 / 需求 8.8）。

需求 8.8：
    IF 对话会话超过 30 分钟无新消息, THEN THE RAG_Engine SHALL 将该会话
    标记为过期，用户下次提问时开启新的对话会话。

实现路径：

- :class:`ConversationService` 写入时把 TTL 重置为 1800 秒；
  Redis 自动过期会在 30 分钟无活动后删除整个 List，
  这就是"标记为过期"的物理表达。
- :meth:`ConversationService.is_active` 通过 ``EXISTS`` 判断当前是否活跃。
- :class:`RAGService` 收到已过期/未知 ``conversation_id`` 时，
  ``_load_history`` 返回空列表，等同于"开始新的会话"——LLM 看不到旧上下文，
  随后写入会重新建立 List。
- ``GET /api/qa/conversations/{conversation_id}/status`` 暴露给前端，
  让前端能主动判断是否需要丢弃旧会话 ID。

本模块覆盖三类场景：

1. ``ConversationService.is_active``：新会话 / 活跃会话 / 已删除（过期）
   会话三种状态下的返回值。
2. ``RAGService.answer``：传入"已过期"会话 ID 时按空历史处理，
   仍能成功对话且把当前轮写回 Redis（重建会话）。
3. ``GET /api/qa/conversations/{id}/status`` 路由：基于 fakeredis 的端到端
   行为校验。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.qa import get_conversation_service, router as qa_router
from app.core.exceptions import register_exception_handlers
from app.services.conversation_service import (
    KEY_PREFIX,
    ConversationService,
)
from app.services.llm_gateway import LLMResponse
from app.services.rag_service import RAGService
from app.services.search_service import SearchResponse, SearchResult


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_redis_client():
    """fakeredis 异步客户端。"""
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def conversation_service(fake_redis_client) -> ConversationService:
    """注入 fakeredis 的 ConversationService。"""
    return ConversationService(redis_client=fake_redis_client)


# ─── 1. is_active 行为 ────────────────────────────────────────────────


class TestIsActive:
    """``ConversationService.is_active`` 在三种典型状态下的返回值。"""

    @pytest.mark.asyncio
    async def test_new_conversation_returns_false(self, conversation_service):
        """从未创建的会话 → 不活跃。"""
        assert await conversation_service.is_active("never-existed") is False

    @pytest.mark.asyncio
    async def test_active_conversation_returns_true(self, conversation_service):
        """刚 append 过消息的会话 → 活跃。"""
        await conversation_service.append("conv-active", "user", "你好")
        assert await conversation_service.is_active("conv-active") is True

    @pytest.mark.asyncio
    async def test_expired_conversation_returns_false(
        self, conversation_service, fake_redis_client
    ):
        """模拟 TTL 到期：直接删除 key 等同于 Redis 自动过期。"""
        await conversation_service.append("conv-expire", "user", "你好")
        assert await conversation_service.is_active("conv-expire") is True

        # fakeredis 不会自然推进时间，直接 DEL 模拟 Redis TTL 到期后的清理。
        await fake_redis_client.delete(f"{KEY_PREFIX}conv-expire")

        assert (
            await conversation_service.is_active("conv-expire") is False
        ), "TTL 到期后会话应被视为不活跃"

    @pytest.mark.asyncio
    async def test_is_active_does_not_refresh_ttl(
        self, conversation_service, fake_redis_client
    ):
        """只读访问不应延长 TTL，避免长时间不交互的会话被永远续命。"""
        await conversation_service.append("conv-ttl", "user", "x")
        # 把 TTL 缩到极小，模拟接近过期
        await fake_redis_client.expire(f"{KEY_PREFIX}conv-ttl", 5)

        await conversation_service.is_active("conv-ttl")

        ttl = await conversation_service.ttl("conv-ttl")
        assert ttl <= 5, f"is_active 不应刷新 TTL，但实际 ttl={ttl}"

    @pytest.mark.asyncio
    async def test_is_active_rejects_empty_conversation_id(
        self, conversation_service
    ):
        """空 conversation_id 与其它读写方法保持一致：ValueError。"""
        with pytest.raises(ValueError):
            await conversation_service.is_active("")


# ─── 2. RAGService 对过期会话按"新会话"处理 ─────────────────────────


def _make_search_response(results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(
        results=results, total=len(results), page=1, page_size=max(1, len(results))
    )


def _make_search_result(score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        document_id="d1",
        chunk_index=0,
        title_chain="第一章",
        source_file="doc.pdf",
        page_number=1,
        score=score,
        highlight="片段内容",
    )


class TestRAGServiceTreatsExpiredAsNewConversation:
    """传入已过期 / 未知 conversation_id 时，RAGService 按空历史处理。"""

    @pytest.mark.asyncio
    async def test_expired_conversation_id_starts_fresh(
        self, conversation_service, fake_redis_client
    ):
        """过期会话再次提问：history 应为空，且当前轮次写入后会重建 key。"""
        # 1) 先建立一个会话并写入若干历史消息
        await conversation_service.append(
            "conv-X", "user", "上一轮问题"
        )
        await conversation_service.append(
            "conv-X", "assistant", "上一轮答案"
        )
        assert await conversation_service.is_active("conv-X") is True

        # 2) 模拟 Redis TTL 到期清理
        await fake_redis_client.delete(f"{KEY_PREFIX}conv-X")
        assert await conversation_service.is_active("conv-X") is False

        # 3) 用同一个 conversation_id 调 RAGService.answer，
        #    LLM 网关接收到的 prompt 不应包含旧"上一轮"内容。
        captured: dict = {}

        async def _capture_complete(**kwargs):
            captured.update(kwargs)
            return LLMResponse(
                content="新答案 [1]",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
                model="stub",
                finish_reason="stop",
            )

        search = AsyncMock()
        search.search = AsyncMock(
            return_value=_make_search_response([_make_search_result()])
        )
        gateway = MagicMock()
        gateway.complete = AsyncMock(side_effect=_capture_complete)

        service = RAGService(
            search_service=search,
            llm_gateway=gateway,
            conversation_service=conversation_service,
        )

        result = await service.answer(
            query="新一轮问题",
            user_id="u-1",
            allowed_space_ids=["s-1"],
            conversation_id="conv-X",
        )

        # 4) 校验：LLM Prompt 不含旧历史
        prompt = captured.get("prompt", "")
        assert "上一轮问题" not in prompt
        assert "上一轮答案" not in prompt
        # 仍然能正常作答
        assert result.answer == "新答案 [1]"

        # 5) 写入后会话被重新激活，本轮 user/assistant 已 append
        assert await conversation_service.is_active("conv-X") is True
        history = await conversation_service.get_history("conv-X")
        assert [m["content"] for m in history] == [
            "新一轮问题",
            "新答案 [1]",
        ]
        # 不包含已过期的旧轮次
        assert not any(
            m["content"] in {"上一轮问题", "上一轮答案"} for m in history
        )

    @pytest.mark.asyncio
    async def test_unknown_conversation_id_behaves_like_new(
        self, conversation_service
    ):
        """从未使用过的 conversation_id 与"过期会话"行为一致——开新会话。"""
        captured: dict = {}

        async def _capture_complete(**kwargs):
            captured.update(kwargs)
            return LLMResponse(
                content="答案 [1]",
                usage={},
                model="stub",
                finish_reason="stop",
            )

        search = AsyncMock()
        search.search = AsyncMock(
            return_value=_make_search_response([_make_search_result()])
        )
        gateway = MagicMock()
        gateway.complete = AsyncMock(side_effect=_capture_complete)

        service = RAGService(
            search_service=search,
            llm_gateway=gateway,
            conversation_service=conversation_service,
        )

        result = await service.answer(
            query="第一次提问",
            user_id="u-1",
            allowed_space_ids=["s-1"],
            conversation_id="brand-new-id",
        )

        # Prompt 中无对话历史段落
        assert "对话历史" not in captured.get("prompt", "")
        assert result.answer == "答案 [1]"
        # 新会话已建立
        history = await conversation_service.get_history("brand-new-id")
        assert [m["role"] for m in history] == ["user", "assistant"]


# ─── 3. GET /api/qa/conversations/{id}/status 路由 ──────────────────


def _make_status_app(svc) -> FastAPI:
    """构造隔离的 FastAPI app，注入 fake user 与传入的 conversation_service。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(qa_router)

    fake_user = MagicMock()
    fake_user.id = uuid.uuid4()

    async def _override_user():
        return fake_user

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_conversation_service] = lambda: svc
    return app


class TestConversationStatusRoute:
    """``GET /api/qa/conversations/{id}/status`` 在三种状态下的返回。

    路由层的关键契约：

    1. 透传 ``conversation_id``；
    2. ``is_active`` 与 ``ttl`` 的返回值如实映射到响应字段；
    3. 不论会话是否存在，HTTP 状态码均为 200——状态信息通过 ``exists`` 字段
       表达，避免前端用 4xx 误判为客户端错误。

    路由本身不直接操作 Redis，因此用 ``AsyncMock`` 注入桩 service 即可，
    fakeredis 的真实交互留给 ``TestIsActive`` 一组断言。
    """

    def _make_stub_service(
        self, *, exists: bool, ttl_seconds: int
    ):
        stub = MagicMock(spec=ConversationService)
        stub.is_active = AsyncMock(return_value=exists)
        stub.ttl = AsyncMock(return_value=ttl_seconds)
        return stub

    def test_active_session_returns_exists_true_with_positive_ttl(self):
        svc = self._make_stub_service(exists=True, ttl_seconds=1799)
        client = TestClient(_make_status_app(svc))

        resp = client.get("/api/qa/conversations/conv-A/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["conversation_id"] == "conv-A"
        assert body["exists"] is True
        assert body["ttl_seconds"] == 1799
        # 透传 path 参数
        svc.is_active.assert_awaited_once_with("conv-A")
        svc.ttl.assert_awaited_once_with("conv-A")

    def test_unknown_session_returns_exists_false_ttl_minus_two(self):
        svc = self._make_stub_service(exists=False, ttl_seconds=-2)
        client = TestClient(_make_status_app(svc))

        resp = client.get("/api/qa/conversations/never-existed/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["exists"] is False
        # Redis TTL 协议：不存在的 key 返回 -2
        assert body["ttl_seconds"] == -2

    def test_expired_session_returns_exists_false(self):
        """TTL 到期后路由应明确告诉调用方会话已过期。"""
        svc = self._make_stub_service(exists=False, ttl_seconds=-2)
        client = TestClient(_make_status_app(svc))

        resp = client.get("/api/qa/conversations/conv-Z/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["exists"] is False
        assert body["ttl_seconds"] == -2

    def test_route_response_uses_200_not_404_for_missing_session(self):
        """需求 8.8：会话不存在不是错误，前端基于 ``exists`` 决策即可。"""
        svc = self._make_stub_service(exists=False, ttl_seconds=-2)
        client = TestClient(_make_status_app(svc))

        resp = client.get("/api/qa/conversations/anything/status")

        assert resp.status_code == 200, (
            "会话不存在应返回 200 + exists=False，而非 404"
        )
