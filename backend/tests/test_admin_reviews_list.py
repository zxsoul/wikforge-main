"""Admin Reviews List API 集成测试（任务 11.9）。

覆盖 ``GET /api/admin/reviews``：

- 默认 ``status='pending'`` + ``sort_by='quality_score_asc'`` 排序，分数最低
  排在最前面（design.md 审核队列页要求）。
- 按 ``profile_id`` / ``space_id`` 过滤。
- 分页元数据（``page`` / ``page_size`` / ``total``）正确。
- 空结果返回 ``total=0`` / ``items=[]``。
- 非管理员返回 403；未登录返回 401。
- ``sort_by='created_at_desc'`` 按创建时间倒序。
- 非法参数返回 400。

策略与 ``test_admin_profiles.py`` 一致：
- FastAPI TestClient + ``dependency_overrides`` 注入 AsyncMock DB session
- 通过覆盖 ``require_admin`` 依赖模拟「管理员 / 非管理员 / 未登录」三种场景
- 不连接真实 DB
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
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
    r.scalar_one_or_none.return_value = value
    return r


def _all_result(rows):
    """Mock for ``await db.execute(...)``; ``.all()`` returns *rows*。"""
    r = MagicMock()
    r.all.return_value = list(rows)
    return r


def _build_review_row(
    *,
    review_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    title: str = "样例文档",
    space_id: uuid.UUID | None = None,
    profile_id: uuid.UUID | None = None,
    profile_name: str | None = None,
    overall: float = 0.5,
    components: dict | None = None,
    issues: list[str] | None = None,
    status: ReviewStatus = ReviewStatus.pending,
    created_at: datetime | None = None,
    reviewed_at: datetime | None = None,
) -> tuple:
    """Build a row matching the SELECT shape used by ``list_reviews``。

    The route returns rows of:
      ``(DocumentReview, Document.title, Document.space_id,
         Document.matched_profile_id, DocumentProfile.name)``

    Tests don't need a real ORM instance — a ``MagicMock`` with the right
    attributes is sufficient because Pydantic only reads the fields that
    are referenced when constructing ``ReviewListItem``.
    """
    review = MagicMock()
    review.id = review_id or uuid.uuid4()
    review.document_id = document_id or uuid.uuid4()
    review.quality_score = {
        "overall": overall,
        "components": components or {},
        "issues": issues or [],
    }
    review.status = status
    review.created_at = created_at or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    review.reviewed_at = reviewed_at
    return (
        review,
        title,
        space_id or uuid.uuid4(),
        profile_id,
        profile_name,
    )


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    """Async session mock。 ``execute`` 在每个测试里按需 patch。"""
    db = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def admin_user() -> MagicMock:
    """A fake authenticated admin user。"""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "admin@wikforge.local"
    user.display_name = "Admin"
    return user


@pytest.fixture
def app(mock_db: AsyncMock, admin_user: MagicMock) -> FastAPI:
    """FastAPI app with ``require_admin`` overridden to return *admin_user*。"""
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
        """未登录（没有 token）→ ``UnauthorizedException`` → 401。"""
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

        response = client.get("/api/admin/reviews")
        assert response.status_code == 401

    def test_non_admin_returns_403(self, mock_db):
        """已登录但邮箱不匹配管理员 → ``ForbiddenException`` → 403。"""
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

        response = client.get("/api/admin/reviews")
        assert response.status_code == 403


# ─── Default sort + filters ────────────────────────────────────────────


class TestListDefaults:
    """默认行为：``status='pending'`` + ``sort_by='quality_score_asc'``。"""

    def test_default_returns_pending_sorted_lowest_first(self, client, mock_db):
        # 准备三条 pending review，分数从低到高排（DB 已按 ASC 返回）。
        rows = [
            _build_review_row(title="差", overall=0.20, profile_name="generic-text"),
            _build_review_row(title="一般", overall=0.45, profile_name="generic-text"),
            _build_review_row(title="尚可", overall=0.65, profile_name="generic-text"),
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(3), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["page"] == 1
        assert body["page_size"] == 20
        assert body["total"] == 3
        assert len(body["items"]) == 3
        # Lowest-score-first ordering preserved as-returned by the (mocked) query.
        titles = [item["document_title"] for item in body["items"]]
        assert titles == ["差", "一般", "尚可"]
        # 综合分原貌透出。
        assert body["items"][0]["quality_score"]["overall"] == pytest.approx(0.20)
        # profile_name 透出。
        assert body["items"][0]["profile_name"] == "generic-text"

    def test_response_item_shape(self, client, mock_db):
        """ReviewListItem 字段齐全。"""
        review_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        space_id = uuid.uuid4()
        profile_id = uuid.uuid4()
        reviewed_at = datetime(2024, 7, 1, 9, 0, 0, tzinfo=timezone.utc)
        rows = [
            _build_review_row(
                review_id=review_id,
                document_id=doc_id,
                title="详细字段",
                space_id=space_id,
                profile_id=profile_id,
                profile_name="chinese-technical-spec",
                overall=0.42,
                components={"text_retention": 0.5},
                issues=["heading low"],
                reviewed_at=reviewed_at,
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews")
        body = response.json()
        item = body["items"][0]

        assert item["review_id"] == str(review_id)
        assert item["document_id"] == str(doc_id)
        assert item["document_title"] == "详细字段"
        assert item["space_id"] == str(space_id)
        assert item["profile_id"] == str(profile_id)
        assert item["profile_name"] == "chinese-technical-spec"
        assert item["quality_score"] == {
            "overall": 0.42,
            "components": {"text_retention": 0.5},
            "issues": ["heading low"],
        }
        assert item["status"] == "pending"
        assert item["created_at"] is not None
        assert item["reviewed_at"] == "2024-07-01T09:00:00Z"

    def test_profile_name_null_when_no_match(self, client, mock_db):
        """文档无匹配 Profile（外连接返回 None） → ``profile_name=None``。"""
        rows = [
            _build_review_row(
                title="无 profile",
                profile_id=None,
                profile_name=None,
                overall=0.30,
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews")
        item = response.json()["items"][0]
        assert item["profile_id"] is None
        assert item["profile_name"] is None


# ─── Filters ───────────────────────────────────────────────────────────


class TestFilters:
    """``status`` / ``profile_id`` / ``space_id`` 过滤。"""

    def test_filter_by_profile_id(self, client, mock_db):
        profile_id = uuid.uuid4()
        rows = [
            _build_review_row(
                title="过滤后只剩它",
                profile_id=profile_id,
                profile_name="custom",
                overall=0.4,
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get(f"/api/admin/reviews?profile_id={profile_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["items"][0]["profile_id"] == str(profile_id)

    def test_filter_by_space_id(self, client, mock_db):
        space_id = uuid.uuid4()
        rows = [
            _build_review_row(
                title="同空间 1",
                space_id=space_id,
                overall=0.3,
            ),
            _build_review_row(
                title="同空间 2",
                space_id=space_id,
                overall=0.5,
            ),
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(2), _all_result(rows)]
        )

        response = client.get(f"/api/admin/reviews?space_id={space_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert all(item["space_id"] == str(space_id) for item in body["items"])

    def test_filter_by_status_approved(self, client, mock_db):
        rows = [
            _build_review_row(
                title="已通过",
                overall=0.75,
                status=ReviewStatus.approved,
                reviewed_at=datetime(2024, 6, 5, tzinfo=timezone.utc),
            )
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(1), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews?status=approved")
        assert response.status_code == 200
        body = response.json()
        assert body["items"][0]["status"] == "approved"

    def test_invalid_status_returns_400(self, client, mock_db):
        """非法 status 在校验阶段就 400，不应触达 DB。"""
        mock_db.execute = AsyncMock()  # should not be called

        response = client.get("/api/admin/reviews?status=banana")
        assert response.status_code == 400
        assert "banana" in response.text or "Invalid status" in response.text
        mock_db.execute.assert_not_called()

    def test_invalid_profile_id_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()

        response = client.get("/api/admin/reviews?profile_id=not-a-uuid")
        assert response.status_code == 400
        mock_db.execute.assert_not_called()

    def test_invalid_space_id_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()

        response = client.get("/api/admin/reviews?space_id=not-a-uuid")
        assert response.status_code == 400
        mock_db.execute.assert_not_called()


# ─── Pagination ────────────────────────────────────────────────────────


class TestPagination:
    """分页元数据正确性。"""

    def test_pagination_metadata(self, client, mock_db):
        rows = [
            _build_review_row(title=f"r{i}", overall=0.1 + i * 0.05)
            for i in range(5)
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(42), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews?page=2&page_size=5")
        assert response.status_code == 200
        body = response.json()
        assert body["page"] == 2
        assert body["page_size"] == 5
        assert body["total"] == 42
        assert len(body["items"]) == 5

    def test_invalid_page_returns_422(self, client):
        """``page`` 必须 ``ge=1``，0 或负数应返回 422（Pydantic 校验）。"""
        response = client.get("/api/admin/reviews?page=0")
        assert response.status_code == 422

    def test_invalid_page_size_returns_422(self, client):
        response = client.get("/api/admin/reviews?page_size=0")
        assert response.status_code == 422

    def test_page_size_above_max_returns_422(self, client):
        response = client.get("/api/admin/reviews?page_size=101")
        assert response.status_code == 422


# ─── Empty results ─────────────────────────────────────────────────────


class TestEmptyResults:
    """空结果应当返回 ``total=0`` / ``items=[]``。"""

    def test_empty_returns_zero_total_and_empty_items(self, client, mock_db):
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(0), _all_result([])]
        )

        response = client.get("/api/admin/reviews")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["items"] == []
        assert body["page"] == 1
        assert body["page_size"] == 20

    def test_empty_with_filters_still_returns_metadata(self, client, mock_db):
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(0), _all_result([])]
        )

        space_id = uuid.uuid4()
        response = client.get(f"/api/admin/reviews?space_id={space_id}&page_size=5")
        body = response.json()
        assert body["total"] == 0
        assert body["items"] == []
        assert body["page_size"] == 5


# ─── Sort by created_at_desc ───────────────────────────────────────────


class TestSortByCreatedAt:
    """``sort_by='created_at_desc'`` 按创建时间倒序。"""

    def test_created_at_desc(self, client, mock_db):
        now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        rows = [
            _build_review_row(title="新", overall=0.7, created_at=now),
            _build_review_row(
                title="次新", overall=0.3, created_at=now - timedelta(hours=1)
            ),
            _build_review_row(
                title="旧", overall=0.5, created_at=now - timedelta(days=1)
            ),
        ]
        mock_db.execute = AsyncMock(
            side_effect=[_scalar_result(3), _all_result(rows)]
        )

        response = client.get("/api/admin/reviews?sort_by=created_at_desc")
        assert response.status_code == 200
        titles = [item["document_title"] for item in response.json()["items"]]
        # 顺序由 mock 保证（这里仅校验路由原样透传）。
        assert titles == ["新", "次新", "旧"]

    def test_invalid_sort_by_returns_400(self, client, mock_db):
        mock_db.execute = AsyncMock()
        response = client.get("/api/admin/reviews?sort_by=score_desc")
        assert response.status_code == 400
        assert "sort_by" in response.text or "Invalid" in response.text
