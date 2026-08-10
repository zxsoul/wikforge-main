"""``POST /api/feedback`` 路由层单元测试（任务 17.1）。

测试目标（需求 9.1 / 18.1）：

- 三种 ``feedback_type``（``thumbs_up`` / ``thumbs_down`` / ``issue``）都能写入
- 未登录返回 401（``UnauthorizedException`` → 标准化错误信封）
- ``query`` / ``feedback_type`` 缺失时 Pydantic 触发 422
- 非法 ``feedback_type`` 在服务层被拒绝并映射为 422
- ``comment`` 为可选字段
- ``feedback_type='issue'`` 必须附带 ``issue_category``，否则 422
- API 把当前登录用户的 ``id`` 透传给 :class:`FeedbackService`

策略：
- ``TestClient`` + ``app.dependency_overrides`` 注入 mock，
  避免真实数据库 / Redis 依赖
- :class:`FeedbackService` 由 ``AsyncMock`` 替身，重点验证 API 层契约

Validates: Requirements 9.1
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.feedback import router as feedback_router
from app.core.database import get_db
from app.core.exceptions import (
    UnauthorizedException,
    register_exception_handlers,
)


# ─── Helpers ────────────────────────────────────────────────────────


class _FakeFeedback:
    """轻量替身，模拟 :class:`SearchFeedback` ORM 实例的可读字段。"""

    def __init__(
        self,
        *,
        user_id: uuid.UUID,
        query: str,
        returned_results: list[str],
        feedback_type: str,
        issue_category: str | None,
        comment: str | None,
        related_profile_id: uuid.UUID | None,
    ):
        self.id = uuid.uuid4()
        self.user_id = user_id
        self.query = query
        self.returned_results = returned_results
        self.feedback_type = feedback_type
        self.issue_category = issue_category
        self.comment = comment
        self.related_profile_id = related_profile_id
        self.created_at = datetime.now(timezone.utc)


def _make_app(
    *,
    create_side_effect=None,
    create_return_value: _FakeFeedback | None = None,
    current_user: MagicMock | None = None,
    auth_error: Exception | None = None,
) -> tuple[FastAPI, MagicMock | None, AsyncMock]:
    """构造一个隔离的 FastAPI 应用并注入依赖覆盖。

    Returns:
        (app, fake_user, service_mock) 三元组：
        - ``app``：可直接给 ``TestClient`` 使用
        - ``fake_user``：当前用户替身（鉴权失败时为 None）
        - ``service_mock``：``FeedbackService`` 的 AsyncMock，便于断言入参
    """
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(feedback_router)

    # 鉴权依赖
    if auth_error is not None:
        async def _override_user():
            raise auth_error

        fake_user = None
    else:
        fake_user = current_user or MagicMock()
        if not hasattr(fake_user, "id"):
            fake_user.id = uuid.uuid4()

        async def _override_user():
            return fake_user

    # DB 依赖：路由层不直接使用 db，service 用 AsyncMock 代替
    async def _override_db():
        yield AsyncMock()

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db

    # 替换 FeedbackService 的 create_feedback，使其按测试要求行为
    service_mock = AsyncMock()

    # 默认 mock 行为复现 FeedbackService.create_feedback 中的字段级校验，
    # 确保非法 feedback_type / 缺失 issue_category 等场景能触发 ValueError
    # 进而被路由层映射为 422，而不是被 mock 静默放行返回 201。
    _VALID_FEEDBACK_TYPES = {"thumbs_up", "thumbs_down", "issue"}
    _VALID_ISSUE_CATEGORIES = {
        "irrelevant",
        "missing_info",
        "citation_error",
        "format",
        "other",
    }

    async def _create(**kwargs):
        if create_side_effect is not None:
            return create_side_effect(**kwargs)
        if create_return_value is not None:
            return create_return_value

        feedback_type = kwargs["feedback_type"]
        issue_category = kwargs.get("issue_category")
        comment = kwargs.get("comment")

        if feedback_type not in _VALID_FEEDBACK_TYPES:
            raise ValueError(f"Invalid feedback_type '{feedback_type}'")
        if feedback_type == "issue":
            if not issue_category:
                raise ValueError(
                    "issue_category is required when feedback_type is 'issue'"
                )
            if issue_category not in _VALID_ISSUE_CATEGORIES:
                raise ValueError(f"Invalid issue_category '{issue_category}'")
        if comment and len(comment) > 500:
            raise ValueError("Comment must not exceed 500 characters")

        related = kwargs.get("related_profile_id")
        return _FakeFeedback(
            user_id=kwargs["user_id"],
            query=kwargs["query"],
            returned_results=kwargs.get("returned_results") or [],
            feedback_type=feedback_type,
            issue_category=issue_category,
            comment=comment,
            related_profile_id=uuid.UUID(related) if related else None,
        )

    service_mock.create_feedback = AsyncMock(side_effect=_create)

    # 在路由层会通过 ``FeedbackService(db)`` 实例化 —— 用 monkeypatch 拦截
    # 不需要真正 patch 类，由 conftest 之外的 fixture 注入更合适。这里
    # 直接 patch 模块级 ``FeedbackService`` 引用即可。
    from app.api import feedback as feedback_module

    feedback_module.FeedbackService = lambda _db: service_mock  # type: ignore[assignment]

    return app, fake_user, service_mock


# ─── Tests ──────────────────────────────────────────────────────────


class TestFeedbackTypes:
    """需求 9.1：三种反馈类型都能写入。"""

    @pytest.mark.parametrize("feedback_type", ["thumbs_up", "thumbs_down"])
    def test_thumbs_feedback_returns_201(self, feedback_type: str):
        """``thumbs_up`` / ``thumbs_down`` 返回 201 + 完整响应。"""
        app, fake_user, service_mock = _make_app()
        client = TestClient(app)

        resp = client.post(
            "/api/feedback",
            json={
                "query": "如何接入 SSO？",
                "returned_results": ["chunk-1", "chunk-2"],
                "feedback_type": feedback_type,
            },
        )

        assert resp.status_code == 201
        body = resp.json()

        assert body["feedback_type"] == feedback_type
        assert body["query"] == "如何接入 SSO？"
        assert body["returned_results"] == ["chunk-1", "chunk-2"]
        assert body["issue_category"] is None
        assert body["comment"] is None
        assert body["user_id"] == str(fake_user.id)
        # API 把当前登录用户透传给 service
        service_mock.create_feedback.assert_awaited_once()
        call_kwargs = service_mock.create_feedback.await_args.kwargs
        assert call_kwargs["user_id"] == fake_user.id
        assert call_kwargs["feedback_type"] == feedback_type

    def test_issue_feedback_with_category_returns_201(self):
        """``issue`` 类型 + 合法 ``issue_category`` 写入成功。"""
        app, _, service_mock = _make_app()
        client = TestClient(app)

        resp = client.post(
            "/api/feedback",
            json={
                "query": "搜索没有返回最新的合同模板",
                "returned_results": ["chunk-9"],
                "feedback_type": "issue",
                "issue_category": "missing_info",
                "comment": "应该包含 2024 年版本",
            },
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["feedback_type"] == "issue"
        assert body["issue_category"] == "missing_info"
        assert body["comment"] == "应该包含 2024 年版本"

        call_kwargs = service_mock.create_feedback.await_args.kwargs
        assert call_kwargs["feedback_type"] == "issue"
        assert call_kwargs["issue_category"] == "missing_info"
        assert call_kwargs["comment"] == "应该包含 2024 年版本"


class TestOptionalFields:
    """``comment`` 等可选字段：缺省时仍返回 201。"""

    def test_comment_is_optional(self):
        """不带 ``comment`` 也可以提交。"""
        app, _, _ = _make_app()
        client = TestClient(app)

        resp = client.post(
            "/api/feedback",
            json={
                "query": "怎样配置部门权限？",
                "feedback_type": "thumbs_up",
            },
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["comment"] is None
        # ``returned_results`` 默认空列表
        assert body["returned_results"] == []


class TestRequestValidation:
    """请求体字段级校验（Pydantic）。"""

    def test_missing_query_returns_422(self):
        """``query`` 必填。"""
        app, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={"feedback_type": "thumbs_up"},
        )
        assert resp.status_code == 422

    def test_missing_feedback_type_returns_422(self):
        """``feedback_type`` 必填。"""
        app, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={"query": "测试"},
        )
        assert resp.status_code == 422

    def test_empty_query_returns_422(self):
        """空字符串触发 ``min_length=1`` 校验。"""
        app, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={"query": "", "feedback_type": "thumbs_up"},
        )
        assert resp.status_code == 422


class TestFeedbackTypeEnumValidation:
    """``feedback_type`` 枚举校验由服务层执行，错误映射 422。"""

    def test_invalid_feedback_type_returns_422(self):
        """未知 ``feedback_type`` 返回 422。"""
        app, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={
                "query": "测试",
                "feedback_type": "not_a_real_type",
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "ValidationError"

    def test_issue_without_category_returns_422(self):
        """``issue`` 类型必须附带 ``issue_category``。"""
        app, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={
                "query": "测试",
                "feedback_type": "issue",
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "ValidationError"

    def test_invalid_issue_category_returns_422(self):
        """非法 ``issue_category`` 返回 422。"""
        app, _, _ = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={
                "query": "测试",
                "feedback_type": "issue",
                "issue_category": "not_a_category",
            },
        )
        assert resp.status_code == 422


class TestAuth:
    """鉴权场景。"""

    def test_unauthenticated_returns_401(self):
        """未登录访问返回 401，错误信封 ``code='Unauthorized'``。"""
        app, _, _ = _make_app(
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/feedback",
            json={"query": "未授权访问", "feedback_type": "thumbs_up"},
        )
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"]["code"] == "Unauthorized"
