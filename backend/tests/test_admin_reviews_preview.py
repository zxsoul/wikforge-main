"""Admin Reviews Preview API 集成测试（任务 11.10）。

覆盖 ``GET /api/admin/reviews/{review_id}/preview``：

- 正常路径：返回 ``review_id`` / ``document_id`` / ``document_title``
  / ``original_file_url`` / ``parsed_markdown`` / ``quality_score`` /
  ``status`` 全字段
- ``original_file_url`` 在 MinIO 可用时是预签名 URL
- ``original_file_url`` 在 MinIO 不可用时退化为
  ``/api/documents/{document_id}/download``
- ``parsed_markdown`` 在 ``quality_score`` JSONB 写入了该字段时直接透出
- ``parsed_markdown`` 在 JSONB 缺失该字段时返回明确占位文本
- ``review_id`` 不存在 → 404
- ``review`` 关联的 ``document`` 缺失 → 404
- 非法 ``review_id``（不是 UUID）→ 400
- 未登录 → 401；非管理员 → 403

策略与 ``test_admin_reviews_list.py`` 一致：
- FastAPI TestClient + ``dependency_overrides`` 注入 AsyncMock DB session
- 通过覆盖 ``require_admin`` 依赖模拟「管理员 / 非管理员 / 未登录」三种场景
- 通过 patch ``app.api.admin_reviews.generate_presigned_get_url`` 控制 MinIO
  路径，无需连接真实 MinIO

Validates: Requirements 17
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Mock for ``await db.execute(...)``; ``.scalar_one_or_none()`` returns *value*。"""
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _make_review(
    *,
    review_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    document_title: str = "样例文档.pdf",
    storage_path: str = "spaces/abc/2024/sample.pdf",
    quality_score: dict | None = None,
    status: ReviewStatus = ReviewStatus.pending,
    document_present: bool = True,
):
    """Build a mock ``DocumentReview`` instance with selectin-loaded ``document``。

    The route reads ``review.document.id`` / ``review.document.title`` /
    ``review.document.storage_path``，所以 mock 只需暴露这几个属性。
    """
    review = MagicMock()
    review.id = review_id or uuid.uuid4()
    review.document_id = document_id or uuid.uuid4()
    review.quality_score = quality_score or {
        "overall": 0.42,
        "components": {
            "text_retention": 0.5,
            "heading_detection": 0.4,
            "table_completeness": 0.4,
            "numeric_protection": 0.4,
            "boilerplate_removal": 0.4,
        },
        "issues": ["text retention low"],
    }
    review.status = status
    review.created_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    review.reviewed_at = None

    if document_present:
        doc = MagicMock()
        doc.id = review.document_id
        doc.title = document_title
        doc.storage_path = storage_path
        review.document = doc
    else:
        # 模拟 document 被并发删除 / lazy=selectin 拿到 None 的极端情况。
        review.document = None

    return review


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

        response = client.get(f"/api/admin/reviews/{uuid.uuid4()}/preview")
        assert response.status_code == 401
        # 不应触达 DB（依赖在路由处理前拒绝）。
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

        response = client.get(f"/api/admin/reviews/{uuid.uuid4()}/preview")
        assert response.status_code == 403
        mock_db.execute.assert_not_called()


# ─── Successful preview ────────────────────────────────────────────────


class TestPreviewSuccess:
    """正常路径：DB 命中 + 返回完整字段。"""

    def test_returns_full_payload_with_presigned_url(self, client, mock_db):
        review = _make_review(
            quality_score={
                "overall": 0.42,
                "components": {"text_retention": 0.5},
                "issues": ["text retention low"],
                "parsed_markdown": "# 样例文档\n\n这是清洗后的 Markdown。",
            },
        )
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value="https://minio.example/presigned?sig=abc",
        ) as mock_presign:
            response = client.get(f"/api/admin/reviews/{review.id}/preview")

        assert response.status_code == 200, response.text
        body = response.json()

        # 关键字段全部存在。
        assert body["review_id"] == str(review.id)
        assert body["document_id"] == str(review.document.id)
        assert body["document_title"] == review.document.title
        assert body["status"] == "pending"

        # 预签名 URL 透出。
        assert body["original_file_url"] == "https://minio.example/presigned?sig=abc"
        # 解析后 Markdown 透出。
        assert body["parsed_markdown"] == "# 样例文档\n\n这是清洗后的 Markdown。"

        # 质量分原貌透出。
        assert body["quality_score"]["overall"] == pytest.approx(0.42)
        assert body["quality_score"]["components"] == {"text_retention": 0.5}
        assert body["quality_score"]["issues"] == ["text retention low"]

        # 预签名生成器被传入正确的 storage_path。
        mock_presign.assert_called_once_with(review.document.storage_path)

    def test_falls_back_to_download_path_when_presign_fails(self, client, mock_db):
        """MinIO 不可用 → presign 返回 None → 退化到 ``/api/documents/{id}/download``。"""
        review = _make_review()
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value=None,
        ):
            response = client.get(f"/api/admin/reviews/{review.id}/preview")

        assert response.status_code == 200
        body = response.json()
        assert body["original_file_url"] == f"/api/documents/{review.document.id}/download"

    def test_parsed_markdown_placeholder_when_missing(self, client, mock_db):
        """JSONB 没写 ``parsed_markdown`` → 返回明确占位，便于前端区分「字段缺失」。"""
        review = _make_review(
            document_title="历史文档.docx",
            quality_score={
                "overall": 0.5,
                "components": {},
                "issues": [],
                # 故意不写 parsed_markdown。
            },
        )
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value="https://minio.example/x",
        ):
            response = client.get(f"/api/admin/reviews/{review.id}/preview")

        assert response.status_code == 200
        body = response.json()
        assert body["parsed_markdown"]  # not empty
        # 占位包含文档标题，便于前端排查。
        assert "历史文档.docx" in body["parsed_markdown"]
        assert "暂未存储" in body["parsed_markdown"]

    def test_parsed_markdown_placeholder_when_quality_score_null(self, client, mock_db):
        """整个 ``quality_score`` 列为 NULL（极旧数据） → 占位 + 0 分 + 空列表。"""
        review = _make_review(
            document_title="legacy.pdf",
            quality_score=None,
        )
        # 同时让 status 字段保持 pending，便于断言 status 透出
        review.status = ReviewStatus.pending
        # 极旧数据可能没有 quality_score 字段；模拟 None
        review.quality_score = None  # type: ignore[assignment]

        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value="https://minio.example/x",
        ):
            response = client.get(f"/api/admin/reviews/{review.id}/preview")

        assert response.status_code == 200
        body = response.json()
        assert body["quality_score"] == {
            "overall": 0.0,
            "components": {},
            "issues": [],
        }
        assert "legacy.pdf" in body["parsed_markdown"]


# ─── Error paths ───────────────────────────────────────────────────────


class TestPreviewErrors:
    """404 / 400 路径。"""

    def test_review_not_found_returns_404(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))

        response = client.get(f"/api/admin/reviews/{uuid.uuid4()}/preview")
        assert response.status_code == 404
        assert "Review not found" in response.text

    def test_document_detached_returns_404(self, client, mock_db):
        """``review.document is None`` （并发删除场景） → 404 而不是 500。"""
        review = _make_review(document_present=False)
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value="https://minio.example/x",
        ):
            response = client.get(f"/api/admin/reviews/{review.id}/preview")

        assert response.status_code == 404
        assert "document" in response.text.lower()

    def test_invalid_review_id_returns_400(self, client, mock_db):
        """非 UUID → 400，不应触达 DB。"""
        mock_db.execute = AsyncMock()
        response = client.get("/api/admin/reviews/not-a-uuid/preview")
        assert response.status_code == 400
        mock_db.execute.assert_not_called()


# ─── Storage-path edge cases ───────────────────────────────────────────


class TestStoragePathHandling:
    """``storage_path`` 缺失时不应当试图调用 MinIO presign。"""

    def test_empty_storage_path_uses_download_fallback(self, client, mock_db):
        """没有 storage_path 时直接用 ``/api/documents/{id}/download``，不调 presign。"""
        review = _make_review(storage_path="")
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value="should-not-be-used",
        ) as mock_presign:
            response = client.get(f"/api/admin/reviews/{review.id}/preview")

        assert response.status_code == 200
        body = response.json()
        # 退化路径，不调用 presign。
        mock_presign.assert_not_called()
        assert body["original_file_url"] == f"/api/documents/{review.document.id}/download"
