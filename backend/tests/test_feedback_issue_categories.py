"""反馈问题类型选项测试（任务 17.2）。

验证目标（需求 9.2 / 18.2）：

- 5 种 ``issue_category``（``irrelevant``、``missing_info``、``citation_error``、
  ``format``、``other``）都能通过 ``POST /api/feedback`` 正常写入
- ``GET /api/feedback/issue-categories`` 返回完整列表，与服务层枚举一致
- 非法 ``issue_category`` 返回 422，错误信封 ``code='ValidationError'``

测试策略：复用 ``test_feedback_api.py`` 中的轻量替身模式 —— 通过
``app.dependency_overrides`` 注入 mock 的 ``get_current_user`` / ``get_db``，
并把 :class:`FeedbackService` 替换为按字段级校验执行的 ``AsyncMock``，
不依赖真实数据库 / Redis。

Validates: Requirements 9.2
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
from app.core.exceptions import register_exception_handlers
from app.services.feedback_service import (
    ISSUE_CATEGORY_LABELS,
    IssueCategory,
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


# 与 FeedbackService.create_feedback 中的字段级校验保持一致，
# 确保非法值能触发 ValueError → 路由层映射 422。
_VALID_FEEDBACK_TYPES = {"thumbs_up", "thumbs_down", "issue"}
_VALID_ISSUE_CATEGORIES = {ic.value for ic in IssueCategory}


def _make_app() -> tuple[FastAPI, MagicMock, AsyncMock]:
    """构造一个隔离的 FastAPI 应用并注入依赖覆盖。

    Returns:
        ``(app, fake_user, service_mock)`` 三元组。
    """
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(feedback_router)

    fake_user = MagicMock()
    fake_user.id = uuid.uuid4()

    async def _override_user():
        return fake_user

    async def _override_db():
        yield AsyncMock()

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db

    service_mock = AsyncMock()

    async def _create(**kwargs):
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

    from app.api import feedback as feedback_module

    feedback_module.FeedbackService = lambda _db: service_mock  # type: ignore[assignment]

    return app, fake_user, service_mock


# ─── Tests ──────────────────────────────────────────────────────────


class TestIssueCategoriesEndpoint:
    """``GET /api/feedback/issue-categories`` 返回全部 5 种可选项。"""

    def test_returns_all_five_categories(self):
        """响应包含 5 种类别，且 value/label 与服务层枚举一致。"""
        app, _, _ = _make_app()
        client = TestClient(app)

        resp = client.get("/api/feedback/issue-categories")

        assert resp.status_code == 200
        body = resp.json()
        assert "categories" in body
        categories = body["categories"]
        assert len(categories) == 5

        values = [item["value"] for item in categories]
        assert values == [
            "irrelevant",
            "missing_info",
            "citation_error",
            "format",
            "other",
        ]

        # 每个 value 都对应一个非空中文 label
        for item in categories:
            assert isinstance(item["label"], str)
            assert item["label"] != ""
            assert ISSUE_CATEGORY_LABELS[item["value"]] == item["label"]

    def test_endpoint_does_not_require_auth(self):
        """该端点用于前端公开渲染下拉框，不应依赖登录态。

        注：路由层未挂 ``Depends(get_current_user)``，TestClient 调用
        即使没有 ``Authorization`` 头也能 200。
        """
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(feedback_router)

        client = TestClient(app)
        resp = client.get("/api/feedback/issue-categories")
        assert resp.status_code == 200


class TestAllFiveCategoriesAcceptedByPostFeedback:
    """``POST /api/feedback`` 接受所有 5 种合法 ``issue_category``。"""

    @pytest.mark.parametrize(
        "category",
        [
            "irrelevant",
            "missing_info",
            "citation_error",
            "format",
            "other",
        ],
    )
    def test_each_category_is_accepted(self, category: str):
        """5 种 ``issue_category`` 均能写入并回显。"""
        app, _, service_mock = _make_app()
        client = TestClient(app)

        resp = client.post(
            "/api/feedback",
            json={
                "query": "搜索结果不准确",
                "feedback_type": "issue",
                "issue_category": category,
            },
        )

        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["feedback_type"] == "issue"
        assert body["issue_category"] == category

        # 服务层收到的 issue_category 与请求一致
        call_kwargs = service_mock.create_feedback.await_args.kwargs
        assert call_kwargs["issue_category"] == category


class TestInvalidCategoryRejected:
    """非法 ``issue_category`` 必须返回 422。"""

    @pytest.mark.parametrize(
        "bad_category",
        [
            "not_a_category",  # 完全未知
            "Irrelevant",  # 大小写不匹配
            "missing-info",  # 连字符 vs 下划线
            " irrelevant ",  # 带空白
            "",  # 空字符串
        ],
    )
    def test_invalid_category_returns_422(self, bad_category: str):
        """各种非法值均被拒绝。"""
        app, _, _ = _make_app()
        client = TestClient(app)

        resp = client.post(
            "/api/feedback",
            json={
                "query": "测试非法类别",
                "feedback_type": "issue",
                "issue_category": bad_category,
            },
        )

        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "ValidationError"
