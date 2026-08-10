"""Admin Reviews Correct API 集成测试（任务 11.11）。

覆盖 ``POST /api/admin/reviews/{review_id}/correct``：

- 正常路径：200 + ``status='corrected'`` + 修正后 Markdown 持久化到
  ``DocumentReview.quality_score`` JSONB + Celery reprocess 链被触发
- ``corrected_markdown`` 仅含空白 → 400
- ``corrected_markdown`` 超大（> 5MB）→ 422 (Pydantic max_length)
- 非法 ``review_id``（非 UUID）→ 400
- ``review`` 不存在 → 404
- 已经是 ``approved`` / ``corrected`` 的审核 → 400 (cannot correct twice)
- 未登录 → 401；非管理员 → 403
- Celery 不可用 / 提交失败 → 仍 200，``status='corrected'``，提示文案变更
- Sample collection (``/samples`` endpoint) 能看到刚刚修正的文档

测试策略与 ``test_admin_reviews_preview.py`` 一致：FastAPI TestClient +
``dependency_overrides`` 注入 AsyncMock DB session；通过 patch
``app.api.admin_reviews.submit_reprocess_from_markdown`` 隔离 Celery 调用。

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
    r.scalar.return_value = value
    return r


def _make_review(
    *,
    review_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    quality_score: dict | None = None,
    status: ReviewStatus = ReviewStatus.pending,
):
    """Build a mock ``DocumentReview`` instance with selectin-loaded ``document``。

    ``submit_correction`` 在更新阶段会修改 ``review.status`` /
    ``review.reviewer_note`` / ``review.reviewed_at`` / ``review.quality_score``，
    所以 mock 必须允许属性赋值——``MagicMock`` 默认就支持。
    """
    review = MagicMock()
    review.id = review_id or uuid.uuid4()
    review.document_id = document_id or uuid.uuid4()
    review.quality_score = quality_score if quality_score is not None else {
        "overall": 0.42,
        "components": {"text_retention": 0.5},
        "issues": ["text retention low"],
        "parsed_markdown": "# 原始解析\n\n这是解析阶段产出的 markdown。",
    }
    review.status = status
    review.reviewer_note = None
    review.reviewed_at = None
    review.created_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return review


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db() -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
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

        response = client.post(
            f"/api/admin/reviews/{uuid.uuid4()}/correct",
            json={"corrected_markdown": "# Fixed\n\nbody"},
        )
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

        response = client.post(
            f"/api/admin/reviews/{uuid.uuid4()}/correct",
            json={"corrected_markdown": "# Fixed\n\nbody"},
        )
        assert response.status_code == 403
        mock_db.execute.assert_not_called()


# ─── Successful correction ─────────────────────────────────────────────


class TestSubmitCorrectionSuccess:
    """正常路径：DB 命中 + JSONB 写入 + Celery 触发。"""

    def test_happy_path_persists_and_triggers_reprocess(self, client, mock_db):
        review = _make_review()
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.submit_reprocess_from_markdown",
            return_value=True,
        ) as mock_reprocess:
            response = client.post(
                f"/api/admin/reviews/{review.id}/correct",
                json={
                    "corrected_markdown": "# 修正后标题\n\n这是修正过的正文。",
                    "reviewer_note": "标题层级错了，已修",
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["review_id"] == str(review.id)
        assert body["status"] == "corrected"
        assert "重新触发" in body["message"]

        # 状态机：审核状态置为 corrected。
        assert review.status == ReviewStatus.corrected
        assert review.reviewer_note == "标题层级错了，已修"
        assert review.reviewed_at is not None

        # JSONB 留底：corrected_markdown / original_markdown / 时间戳。
        assert review.quality_score["corrected_markdown"] == (
            "# 修正后标题\n\n这是修正过的正文。"
        )
        assert review.quality_score["original_markdown"] == (
            "# 原始解析\n\n这是解析阶段产出的 markdown。"
        )
        assert "correction_timestamp" in review.quality_score
        # 评分维度仍保留：``ParseQualityScore.from_dict`` 的往返一致性。
        assert review.quality_score["overall"] == pytest.approx(0.42)
        assert "components" in review.quality_score
        assert review.quality_score["issues"] == ["text retention low"]

        # Celery 任务被触发，参数是 (document_id_str, corrected_markdown)。
        mock_reprocess.assert_called_once()
        args = mock_reprocess.call_args.args
        assert args[0] == str(review.document_id)
        assert args[1] == "# 修正后标题\n\n这是修正过的正文。"

        # DB 写入路径：flush + refresh 都被调用。
        mock_db.flush.assert_called()
        mock_db.refresh.assert_called()

    def test_rejected_review_can_be_corrected(self, client, mock_db):
        """``rejected`` → ``corrected`` 是允许的（审核员驳回后再重新提供修正）。"""
        review = _make_review(status=ReviewStatus.rejected)
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.submit_reprocess_from_markdown",
            return_value=True,
        ):
            response = client.post(
                f"/api/admin/reviews/{review.id}/correct",
                json={"corrected_markdown": "# Reverted heading\n\nbody"},
            )
        assert response.status_code == 200
        assert review.status == ReviewStatus.corrected

    def test_missing_parsed_markdown_falls_back_to_empty(self, client, mock_db):
        """``parsed_markdown`` 缺失时 ``original_markdown`` 落空字符串，不抛异常。"""
        review = _make_review(
            quality_score={
                "overall": 0.5,
                "components": {},
                "issues": [],
                # 故意不写 parsed_markdown / original_markdown
            }
        )
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.submit_reprocess_from_markdown",
            return_value=True,
        ):
            response = client.post(
                f"/api/admin/reviews/{review.id}/correct",
                json={"corrected_markdown": "# Fixed\n\nbody"},
            )

        assert response.status_code == 200
        assert review.quality_score["original_markdown"] == ""
        assert review.quality_score["corrected_markdown"] == "# Fixed\n\nbody"


# ─── Celery unavailability ─────────────────────────────────────────────


class TestCeleryUnavailable:
    """Celery 不可用 / broker 离线时 API 仍 200，仅修改提示文案。"""

    def test_celery_unavailable_still_marks_corrected(self, client, mock_db, caplog):
        review = _make_review()
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.submit_reprocess_from_markdown",
            return_value=False,
        ) as mock_reprocess:
            response = client.post(
                f"/api/admin/reviews/{review.id}/correct",
                json={"corrected_markdown": "# Still saves\n\nbody"},
            )

        # 即使 reprocess 失败，API 仍然 200，状态仍然 corrected。
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "corrected"
        # 提示文案变更，提醒运维侧手动触发。
        assert "消息队列" in body["message"] or "运维" in body["message"]

        # JSONB 仍然完成持久化。
        assert review.status == ReviewStatus.corrected
        assert review.quality_score["corrected_markdown"] == "# Still saves\n\nbody"

        mock_reprocess.assert_called_once()

    def test_reprocess_raises_is_swallowed(self, client, mock_db, caplog):
        """``submit_reprocess_from_markdown`` 抛异常时也不应让 API 5xx。"""
        review = _make_review()
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        with patch(
            "app.api.admin_reviews.submit_reprocess_from_markdown",
            side_effect=RuntimeError("broker connection refused"),
        ):
            with caplog.at_level("WARNING"):
                response = client.post(
                    f"/api/admin/reviews/{review.id}/correct",
                    json={"corrected_markdown": "# Still saves\n\nbody"},
                )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "corrected"
        # 兜底 WARNING 被打印。
        assert any(
            "submit_reprocess_from_markdown raised" in record.message
            for record in caplog.records
        )


# ─── Validation errors ─────────────────────────────────────────────────


class TestValidationErrors:
    """请求参数校验：400 / 422 路径。"""

    def test_invalid_review_id_returns_400(self, client, mock_db):
        """非 UUID → 400，不应触达 DB。"""
        mock_db.execute = AsyncMock()
        response = client.post(
            "/api/admin/reviews/not-a-uuid/correct",
            json={"corrected_markdown": "# Fixed\n\nbody"},
        )
        assert response.status_code == 400
        mock_db.execute.assert_not_called()

    def test_empty_markdown_returns_422(self, client, mock_db):
        """``min_length=1`` 拒绝完全空字符串（Pydantic 422）。"""
        mock_db.execute = AsyncMock()
        response = client.post(
            f"/api/admin/reviews/{uuid.uuid4()}/correct",
            json={"corrected_markdown": ""},
        )
        assert response.status_code == 422
        mock_db.execute.assert_not_called()

    def test_whitespace_only_markdown_returns_400(self, client, mock_db):
        """全空白（``"   \\n\\t  "``）→ 400，不写 JSONB。"""
        mock_db.execute = AsyncMock()
        response = client.post(
            f"/api/admin/reviews/{uuid.uuid4()}/correct",
            json={"corrected_markdown": "   \n\t  "},
        )
        assert response.status_code == 400
        # 提前在 Pydantic 之后、DB 之前就拒绝。
        mock_db.execute.assert_not_called()

    def test_oversized_markdown_returns_422(self, client, mock_db):
        """超出 5 MB 上限的 payload 由 Pydantic 拒绝（422），不打 DB。"""
        mock_db.execute = AsyncMock()
        # 5 MB + 1 byte
        oversized = "a" * (5 * 1024 * 1024 + 1)
        response = client.post(
            f"/api/admin/reviews/{uuid.uuid4()}/correct",
            json={"corrected_markdown": oversized},
        )
        assert response.status_code == 422
        mock_db.execute.assert_not_called()


# ─── 404 / state errors ────────────────────────────────────────────────


class TestStateErrors:
    """``review`` 不存在 / 状态不允许修正。"""

    def test_review_not_found_returns_404(self, client, mock_db):
        mock_db.execute = AsyncMock(return_value=_scalar_result(None))
        response = client.post(
            f"/api/admin/reviews/{uuid.uuid4()}/correct",
            json={"corrected_markdown": "# Fixed\n\nbody"},
        )
        assert response.status_code == 404
        assert "Review not found" in response.text

    @pytest.mark.parametrize(
        "status", [ReviewStatus.approved, ReviewStatus.corrected]
    )
    def test_already_finalized_review_returns_400(self, client, mock_db, status):
        """``approved`` 与 ``corrected`` 都不能再次修正。"""
        review = _make_review(status=status)
        mock_db.execute = AsyncMock(return_value=_scalar_result(review))

        response = client.post(
            f"/api/admin/reviews/{review.id}/correct",
            json={"corrected_markdown": "# Try again\n\nbody"},
        )
        assert response.status_code == 400
        assert "Cannot correct" in response.text
        # 状态保持不变，不进入修正分支。
        assert review.status == status


# ─── Samples endpoint surfacing ────────────────────────────────────────


class TestSampleCollectionSurfacing:
    """``GET /api/admin/reviews/samples`` 应能看到刚修正的文档。"""

    def test_corrected_review_appears_in_samples(self, client, mock_db, admin_user):
        # 模拟 list_correction_samples 的两次 execute：
        #   1) count → 1
        #   2) 主查询 → 一条 (review, space_id, profile_id, profile_name)
        review = _make_review(status=ReviewStatus.corrected)
        review.reviewer_note = "标题层级错了，已修"
        review.reviewed_at = datetime(2024, 6, 2, 9, 0, 0, tzinfo=timezone.utc)
        review.quality_score = {
            "overall": 0.42,
            "components": {"text_retention": 0.5},
            "issues": [],
            "original_markdown": "# 原始\n\n旧的",
            "corrected_markdown": "# 修正\n\n新的",
            "correction_timestamp": "2024-06-02T09:00:00+00:00",
        }
        space_id = uuid.uuid4()
        profile_id = uuid.uuid4()

        # 主查询返回元组列表（与 list_correction_samples 的 select 一致）
        list_result_mock = MagicMock()
        list_result_mock.all.return_value = [
            (review, space_id, profile_id, "generic-text")
        ]

        # count 查询：``.scalar() -> 1``
        count_result_mock = MagicMock()
        count_result_mock.scalar.return_value = 1

        # 第一个 execute 是 count（list_correction_samples 先 count 再 list）
        mock_db.execute = AsyncMock(
            side_effect=[count_result_mock, list_result_mock]
        )

        response = client.get("/api/admin/reviews/samples")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["total"] == 1
        assert len(body["samples"]) == 1
        sample = body["samples"][0]
        assert sample["original_text"] == "# 原始\n\n旧的"
        assert sample["corrected_text"] == "# 修正\n\n新的"
        assert sample["reviewer_note"] == "标题层级错了，已修"
