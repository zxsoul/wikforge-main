"""POST /api/search 路由层单元测试（任务 14.9）。

测试策略：
- FastAPI ``TestClient`` + ``dependency_overrides`` 注入 mock
  :class:`SearchService`、当前用户、DB session
- 不连接真实 DB / OpenSearch / Qdrant，行为通过 ``AsyncMock`` 控制
- 覆盖：
  - 正常请求结构、默认分页、page_size 上限/越界
  - query 长度校验（空 / 过长）
  - 未登录 401
  - 5 秒整体超时降级为 504

Validates: Requirements 6.5
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.search import (
    SEARCH_TOTAL_TIMEOUT,
    get_search_service,
)
from app.api.search import (
    router as search_router,
)
from app.core.database import get_db
from app.core.exceptions import (
    UnauthorizedException,
    register_exception_handlers,
)
from app.services.search_service import (
    SearchResponse,
    SearchResult,
    SearchService,
)

# ─── Helpers ──────────────────────────────────────────────────────────


def _make_search_result(
    chunk_id: str | None = None,
    document_id: str | None = None,
    score: float = 0.85,
    highlight: str = "<mark>测试</mark>命中片段",
    title_chain: str = "章节 1 > 子章节 2",
    source_file: str = "demo.pdf",
    chunk_index: int = 0,
    page_number: int = 1,
) -> SearchResult:
    """构造一条 ``SearchResult`` 用例数据。"""
    return SearchResult(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=document_id or str(uuid.uuid4()),
        chunk_index=chunk_index,
        title_chain=title_chain,
        source_file=source_file,
        score=score,
        highlight=highlight,
        page_number=page_number,
    )


def _make_app(
    *,
    search_service: SearchService | AsyncMock | None = None,
    current_user: MagicMock | None = None,
    auth_error: Exception | None = None,
    allowed_space_ids: list[str] | None = None,
) -> FastAPI:
    """构造一个隔离的 FastAPI 应用，注入测试所需的依赖覆盖。

    - 当 ``auth_error`` 不为空时，``get_current_user`` 会抛出该异常
    - 否则使用 ``current_user``（默认随机 UUID 用户）
    - ``get_db`` 注入一个返回 ``allowed_space_ids`` 的 mock session
    """
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(search_router)

    # 鉴权依赖：401 场景下抛 UnauthorizedException，
    # 由全局异常处理器返回 401 + 标准信封
    if auth_error is not None:
        async def _override_user():
            raise auth_error
    else:
        fake_user = current_user or MagicMock()
        if not hasattr(fake_user, "id"):
            fake_user.id = uuid.uuid4()

        async def _override_user():
            return fake_user

    # DB session：测试中 ``_get_user_allowed_space_ids`` 通过 db.execute 拿空间集合
    space_ids = allowed_space_ids if allowed_space_ids is not None else [str(uuid.uuid4())]

    db_session = AsyncMock()
    db_result = MagicMock()
    db_result.scalars.return_value.all.return_value = space_ids
    db_session.execute = AsyncMock(return_value=db_result)

    async def _override_db():
        yield db_session

    # 默认搜索服务：返回单条结果
    if search_service is None:
        default_service = AsyncMock(spec=SearchService)
        default_service.search = AsyncMock(
            return_value=SearchResponse(
                results=[_make_search_result()],
                total=1,
                page=1,
                page_size=10,
            )
        )
        search_service = default_service

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_search_service] = lambda: search_service

    return app


# ─── Tests ────────────────────────────────────────────────────────────


class TestSearchApiSuccess:
    """正常路径：响应结构 + 分页参数透传。"""

    def test_returns_200_with_expected_structure(self):
        """正常请求返回 200，响应字段与契约一致。"""
        app = _make_app()
        client = TestClient(app)

        resp = client.post("/api/search", json={"query": "知识库搜索"})

        assert resp.status_code == 200
        body = resp.json()

        # 顶层字段
        assert set(body.keys()) >= {"results", "total", "page", "page_size"}
        assert body["total"] == 1
        assert body["page"] == 1
        assert body["page_size"] == 10
        assert len(body["results"]) == 1

        # 结果字段
        item = body["results"][0]
        for key in (
            "chunk_id",
            "document_id",
            "chunk_index",
            "title_chain",
            "source_file",
            "score",
            "highlight",
        ):
            assert key in item, f"missing key: {key}"
        assert 0.0 <= item["score"] <= 1.0
        assert len(item["highlight"]) <= 200

    def test_default_page_and_page_size(self):
        """未传分页参数时使用默认 page=1, page_size=10。"""
        captured: dict = {}

        async def _spy_search(**kwargs):
            captured.update(kwargs)
            return SearchResponse(results=[], total=0, page=1, page_size=10)

        service = AsyncMock(spec=SearchService)
        service.search.side_effect = _spy_search

        app = _make_app(search_service=service)
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "你好"})

        assert resp.status_code == 200
        assert captured["page"] == 1
        assert captured["page_size"] == 10

    def test_page_size_50_is_accepted(self):
        """``page_size=50`` 是允许的上限。"""
        captured: dict = {}

        async def _spy_search(**kwargs):
            captured.update(kwargs)
            return SearchResponse(results=[], total=0, page=1, page_size=50)

        service = AsyncMock(spec=SearchService)
        service.search.side_effect = _spy_search

        app = _make_app(search_service=service)
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "测试", "page_size": 50})

        assert resp.status_code == 200
        assert captured["page_size"] == 50

    def test_passes_user_id_and_spaces_to_service(self):
        """API 层应把 user_id 与可访问空间列表透传给 SearchService。"""
        captured: dict = {}

        async def _spy_search(**kwargs):
            captured.update(kwargs)
            return SearchResponse(results=[], total=0, page=1, page_size=10)

        service = AsyncMock(spec=SearchService)
        service.search.side_effect = _spy_search

        user_id = uuid.uuid4()
        fake_user = MagicMock()
        fake_user.id = user_id
        space_a = str(uuid.uuid4())
        space_b = str(uuid.uuid4())

        app = _make_app(
            search_service=service,
            current_user=fake_user,
            allowed_space_ids=[space_a, space_b],
        )
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "权限"})

        assert resp.status_code == 200
        assert captured["user_id"] == str(user_id)
        assert captured["allowed_space_ids"] == [space_a, space_b]


class TestSearchApiValidation:
    """请求体校验。"""

    def test_page_size_over_limit_returns_422(self):
        """``page_size=51`` 超过上限，Pydantic 返回 422。"""
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/search", json={"query": "测试", "page_size": 51}
        )
        assert resp.status_code == 422

    def test_page_zero_returns_422(self):
        """``page=0`` 不满足 ``ge=1`` 返回 422。"""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "测试", "page": 0})
        assert resp.status_code == 422

    def test_empty_query_returns_422(self):
        """空查询字符串触发 ``min_length=1`` 校验。"""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self):
        """查询超过 500 字符返回 422。"""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "x" * 501})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self):
        """请求体缺少必填 ``query`` 字段时返回 422。"""
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/api/search", json={})
        assert resp.status_code == 422


class TestSearchApiAuth:
    """鉴权场景。"""

    def test_unauthenticated_returns_401(self):
        """``get_current_user`` 抛 ``UnauthorizedException`` 时映射为 401。"""
        app = _make_app(auth_error=UnauthorizedException("缺少认证令牌"))
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "未授权访问"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "Unauthorized"

    def test_user_with_no_spaces_returns_empty(self):
        """无可访问空间时 SearchService 返回空结果。"""
        service = AsyncMock(spec=SearchService)
        service.search = AsyncMock(
            return_value=SearchResponse(
                results=[], total=0, page=1, page_size=10
            )
        )

        app = _make_app(search_service=service, allowed_space_ids=[])
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "无权限用户"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["results"] == []
        assert body["total"] == 0


class TestSearchApiTimeout:
    """5 秒整体超时降级为 504。"""

    def test_search_total_timeout_returns_504(self):
        """SearchService 超过 5 秒时 API 返回 504。

        通过让 ``search`` 协程在事件循环上长时间挂起，触发
        :func:`asyncio.wait_for` 的 ``TimeoutError`` 分支。为避免测试
        慢吞吞，这里 monkeypatch 把超时阈值降到 0.05 秒。
        """
        from app.api import search as search_module

        async def _slow_search(**kwargs):
            await asyncio.sleep(2.0)
            return SearchResponse(results=[], total=0, page=1, page_size=10)

        service = AsyncMock(spec=SearchService)
        service.search.side_effect = _slow_search

        original_timeout = search_module.SEARCH_TOTAL_TIMEOUT
        try:
            search_module.SEARCH_TOTAL_TIMEOUT = 0.05
            app = _make_app(search_service=service)
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "慢查询"})
        finally:
            search_module.SEARCH_TOTAL_TIMEOUT = original_timeout

        assert resp.status_code == 504
        body = resp.json()
        assert body["error"]["code"] == "SearchTimeout"

    def test_total_timeout_constant_is_5_seconds(self):
        """生产配置应保持 5 秒整体超时（与需求 6.7 一致）。"""
        assert SEARCH_TOTAL_TIMEOUT == pytest.approx(5.0)
