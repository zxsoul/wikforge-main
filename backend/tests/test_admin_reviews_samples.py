"""Admin Reviews Samples API 集成测试（任务 11.12）。

覆盖 ``GET /api/admin/reviews/samples``——修正样本收集端点：

- 正常路径：返回包含 ``profile_id`` / ``profile_name`` /
  ``original_text`` / ``corrected_text`` / ``quality_score_snapshot``
  / ``reviewed_at`` / ``corrected_at`` 的样本列表
- ``profile_name`` 通过 ``DocumentProfile`` 外连接补齐；文档无匹配
  Profile 时为 ``None``
- ``corrected_at == reviewed_at``（语义别名）
- ``quality_score_snapshot`` 包含 ``overall`` / ``components`` / ``issues``
- 过滤：``profile_id`` / ``space_id`` / ``date_from`` / ``date_to``
- 分页：``skip`` / ``limit`` 元数据回填
- 401（未登录）/ 403（非管理员）
- 400（非法 UUID / 非法日期格式）

策略与 ``test_admin_reviews_list.py`` 一致：FastAPI TestClient +
``dependency_overrides`` 注入 AsyncMock DB session；不连真实 DB。

Validates: Requirements 17
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin_reviews import router as admin_reviews_router
from app.api.auth import require_admin
from app.core.database import get_db
from app.core.exceptions import (
    ForbiddenException,
    UnauthorizedException,
    register_exception_handlers,
)
from app.models.document_review import ReviewStatus


# ─── Mock helpers ──────────────────────────────────────────────────────


def _scalar_result(value):
    """Mock for ``await db.execute(...)``; ``.scalar()`` returns *value*。"""
    r = MagicMock()
    r.scalar.return_value = value
    return r


def _all_result(rows):
    """Mock for ``await db.execute(...)``; ``.all()`` returns *rows*。"""
    r = MagicMock()
    r.all.return_value = list(rows)
    return r


def _build_sample_row(
    *,
    review_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    space_id: uuid.UUID | None = None,
    profile_id: uuid.UUID | None = None,
    profile_name: str | None = None,
    overall: float = 0.42,
    components: dict | None = None,
    issues: list[str] | None = None,
    original_markdown: str = "# 原始\n\n旧的",
    corrected_markdown: str = "# 修正\n\n新的",
    reviewer_note: str | None = "标题层级错了，已修",
    reviewed_at: datetime | None = None,
    correction_timestamp: str | None = None,
) -> tuple:
    """Build a row matching the SELECT shape used by ``list_correction_samples``。

    Route returns rows of:
      ``(DocumentReview, Document.space_id,
         Document.matched_profile_id, DocumentProfile.name)``
    """
    review = MagicMock()
    review.id = review_id or uuid.uuid4()
    review.document_id = document_id or uuid.uuid4()
    review.status = ReviewStatus.corrected
    review.reviewer_note = reviewer_note
    review.reviewed_at = reviewed_at or datetime(
        2024, 6, 2, 9, 0, 0, tzinfo=timezone.utc
    )
    review.created_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    review.quality_score = {
        "overall": overall,
        "components": components or {"text_retention": 0.5},
        "issues": issues or ["text retention low"],
        "original_markdown": original_markdown,
        "corrected_markdown": corrected_markdown,
        "correction_timestamp": correction_timestamp
        or review.reviewed_at.isoformat(),
    }
    return (
        review,
        space_id or uuid.uuid4(),
        profile_id,
        profile_name,
    )


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def admin_user() -> MagicMock:
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "admin@wikforge.local"
    user.display_name = "Admin"
    return user


@pytest.fixture
def app(mock_db: AsyncMock, admin_user: MagicMock) -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(admin_reviews_router)

    async def _override_get_db():
        yield mock_db

    async def _override_require_admin():
        return admin_user

    application.dependency_overrides[get_db] = _override_get_db
    application.dependency_overrides[require_admin] = _override_require_admin
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ─── Authorization ─────────────────────────────────────────────────────


class TestAuthorization:
    """``require_admin`` 守门：401 / 403 路径。"""

    def test_unauthenticated_returns_401(self, mock_db):
        application = FastAPI()
        register_exception_handlers(application)
        application.include_router(admin_reviews_router)

        async def _override_get_db():
            yield mock_db

        async def _override_require_admin():
            raise UnauthorizedException("缺少认证令牌")

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin
        client = TestClient(application)

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 401
        # 守门拒绝后不应触达 DB。
        mock_db.execute.assert_not_called()

    def test_non_admin_returns_403(self, mock_db):
        application = FastAPI()
        register_exception_handlers(application)
        application.include_router(admin_reviews_router)

        async def _override_get_db():
            yield mock_db

        async def _override_require_admin():
            raise ForbiddenException("需要管理员权限")

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin
        client = TestClient(application)

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 403
        mock_db.execute.assert_not_called()


# ─── Default response shape ────────────────────────────────────────────


class TestDefaultResponse:
    """``GET /api/admin/reviews/samples`` 默认行为：返回 corrected 样本。"""

    def test_returns_profile_name_when_document_has_profile(
        self, client, mock_db
    ):
        """文档匹配 Profile 时，``profile_name`` 透传 DocumentProfile.name。"""
        profile_id = uuid.uuid4()
        rows = [
            _build_sample_row(
                profile_id=profile_id,
                profile_name="cement-spec-v1",
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] == 1
        assert len(body["samples"]) == 1
        sample = body["samples"][0]
        assert sample["profile_id"] == str(profile_id)
        assert sample["profile_name"] == "cement-spec-v1"

    def test_profile_name_null_when_no_match(self, client, mock_db):
        """文档无匹配 Profile（外连接返回 None） → ``profile_name=None``。"""
        rows = [
            _build_sample_row(profile_id=None, profile_name=None),
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200
        body = response.json()
        sample = body["samples"][0]
        assert sample["profile_id"] is None
        assert sample["profile_name"] is None

    def test_quality_score_snapshot_contains_overall_components_issues(
        self, client, mock_db
    ):
        """``quality_score_snapshot`` 必须有 overall + components + issues。"""
        rows = [
            _build_sample_row(
                overall=0.55,
                components={
                    "text_retention": 0.7,
                    "heading_detection": 0.4,
                },
                issues=["heading detection low"],
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200
        snapshot = response.json()["samples"][0]["quality_score_snapshot"]
        assert snapshot["overall"] == pytest.approx(0.55)
        assert snapshot["components"] == {
            "text_retention": 0.7,
            "heading_detection": 0.4,
        }
        assert snapshot["issues"] == ["heading detection low"]

    def test_corrected_at_equals_reviewed_at(self, client, mock_db):
        """``corrected_at`` 与 ``reviewed_at`` 应当为同一时间。"""
        ts = datetime(2024, 7, 1, 9, 30, 0, tzinfo=timezone.utc)
        rows = [_build_sample_row(reviewed_at=ts)]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200
        sample = response.json()["samples"][0]
        assert sample["reviewed_at"] == sample["corrected_at"]
        # 同时也兼容旧的 ``created_at`` 别名（指向修正时间）。
        assert sample["created_at"] == sample["reviewed_at"]

    def test_returns_original_and_corrected_text(self, client, mock_db):
        rows = [
            _build_sample_row(
                original_markdown="# 原文\n\n第一版",
                corrected_markdown="# 原文\n\n第二版（已校正）",
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews/samples")
        sample = response.json()["samples"][0]
        assert sample["original_text"] == "# 原文\n\n第一版"
        assert sample["corrected_text"] == "# 原文\n\n第二版（已校正）"

    def test_empty_returns_zero_total(self, client, mock_db):
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(0), _all_result([])]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["samples"] == []
        assert body["skip"] == 0
        assert body["limit"] == 20


# ─── Filters ───────────────────────────────────────────────────────────


class TestFilters:
    """``profile_id`` / ``space_id`` / ``date_from`` / ``date_to`` 过滤。"""

    def test_filter_by_profile_id(self, client, mock_db):
        """``profile_id`` 过滤参数被正确接受，返回对应样本。"""
        profile_id = uuid.uuid4()
        rows = [
            _build_sample_row(
                profile_id=profile_id, profile_name="cement-spec-v1"
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get(
            f"/api/admin/reviews/samples?profile_id={profile_id}"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["samples"][0]["profile_id"] == str(profile_id)

    def test_filter_by_space_id(self, client, mock_db):
        space_id = uuid.uuid4()
        rows = [_build_sample_row(space_id=space_id)]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get(
            f"/api/admin/reviews/samples?space_id={space_id}"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["samples"][0]["space_id"] == str(space_id)

    def test_filter_by_date_range(self, client, mock_db):
        """``date_from`` / ``date_to`` 都可解析（接口接受时一定调到 DB）。"""
        rows = [
            _build_sample_row(
                reviewed_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get(
            "/api/admin/reviews/samples"
            "?date_from=2024-06-01&date_to=2024-06-30T23:59:59Z"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] == 1
        # 校验过滤参数被走进了 DB execute（count + list 共 2 次）。
        assert mock_db.execute.await_count == 2

    def test_filter_date_from_only(self, client, mock_db):
        rows = [_build_sample_row()]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get(
            "/api/admin/reviews/samples?date_from=2024-06-01T00:00:00Z"
        )
        assert response.status_code == 200


# ─── Validation errors ─────────────────────────────────────────────────


class TestValidationErrors:
    """非法过滤参数：400 / 422 路径。"""

    def test_invalid_profile_id_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get(
            "/api/admin/reviews/samples?profile_id=not-a-uuid"
        )
        assert response.status_code == 400
        mock_db.execute.assert_not_called()

    def test_invalid_space_id_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get(
            "/api/admin/reviews/samples?space_id=not-a-uuid"
        )
        assert response.status_code == 400
        mock_db.execute.assert_not_called()

    def test_invalid_date_from_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get(
            "/api/admin/reviews/samples?date_from=not-a-date"
        )
        assert response.status_code == 400
        mock_db.execute.assert_not_called()

    def test_invalid_date_to_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get(
            "/api/admin/reviews/samples?date_to=2024/06/01"
        )
        assert response.status_code == 400
        mock_db.execute.assert_not_called()

    def test_invalid_skip_returns_422(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get("/api/admin/reviews/samples?skip=-1")
        assert response.status_code == 422

    def test_invalid_limit_returns_422(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get("/api/admin/reviews/samples?limit=0")
        assert response.status_code == 422

    def test_limit_above_max_returns_422(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get("/api/admin/reviews/samples?limit=101")
        assert response.status_code == 422


# ─── Pagination ────────────────────────────────────────────────────────


class TestPagination:
    """``skip`` / ``limit`` 元数据回填。"""

    def test_skip_and_limit_metadata(self, client, mock_db):
        rows = [_build_sample_row() for _ in range(3)]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(50), _all_result(rows)]
        )

        response = client.get(
            "/api/admin/reviews/samples?skip=10&limit=3"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 50
        assert body["skip"] == 10
        assert body["limit"] == 3
        assert len(body["samples"]) == 3

    def test_default_pagination(self, client, mock_db):
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(0), _all_result([])]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200
        body = response.json()
        # 默认 skip=0 / limit=20
        assert body["skip"] == 0
        assert body["limit"] == 20
