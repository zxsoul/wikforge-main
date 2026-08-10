"""反馈聚合分析 API 与服务层测试（任务 17.4）。

测试目标（需求 9.4 / 18.4）：

> 系统应提供反馈聚合分析 API，支持按 Profile / 文档 / 查询类型 / 时间范围进行聚合。

覆盖三层：

1. **服务层维度聚合**：``FeedbackService.aggregate_feedback`` 按 Profile / 文档 /
   feedback_type + issue_category / 日期 多维度同时输出统计，且 returned_results
   内重复 ID 不会被重复计数。
2. **服务层过滤**：``FeedbackFilter`` 同时支持 ``profile_id`` / ``document_id`` /
   ``feedback_type`` / ``issue_category`` / 时间范围 / ``user_id``，并能正确
   叠加（多个条件之间 AND 关系）。
3. **API 层路由契约**：
   - ``GET /api/admin/feedback/analysis`` 通过 ``require_admin`` 守门，未登录
     401、非管理员 403、管理员 200。
   - 所有过滤参数（含 ``document_id``）正确透传到 :class:`FeedbackFilter`。
   - 响应中包含 ``by_document`` 字段，承载文档维度聚合。

为了避免引入真实 PostgreSQL 依赖（``returned_results`` 是 JSONB 列），服务层测试
使用内存中的反馈样本 + ``execute`` mock 来还原「过滤已经发生」之后的样本子集，
重点覆盖 service 内的统计/累加逻辑；过滤条件构造由专门的单元测试单独覆盖。

Validates: Requirements 9.4
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import require_admin
from app.api.feedback import router as feedback_router
from app.core.database import get_db
from app.core.exceptions import (
    ForbiddenException,
    UnauthorizedException,
    register_exception_handlers,
)
from app.services.feedback_service import (
    FeedbackAggregation,
    FeedbackFilter,
    FeedbackService,
)

# ─── Helpers ────────────────────────────────────────────────────────


def _feedback(
    *,
    feedback_type: str,
    profile_id: uuid.UUID | None = None,
    issue_category: str | None = None,
    returned_results: list[str] | None = None,
    created_at: datetime | None = None,
    user_id: uuid.UUID | None = None,
) -> SimpleNamespace:
    """构造一条「足够用」的反馈样本。

    使用 :class:`SimpleNamespace` 替身而非真实 ORM 实例，避免依赖 SQLAlchemy
    元数据初始化；服务层只读取这些属性即可完成聚合。
    """
    return SimpleNamespace(
        feedback_type=feedback_type,
        related_profile_id=profile_id,
        issue_category=issue_category,
        returned_results=returned_results or [],
        created_at=created_at or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        user_id=user_id,
    )


def _stub_execute_returning(feedbacks: list[SimpleNamespace]) -> AsyncMock:
    """构造 ``db.execute`` 的 AsyncMock：返回的对象 ``.scalars().all()`` 输出 *feedbacks*。"""
    scalars = MagicMock()
    scalars.all.return_value = list(feedbacks)
    result = MagicMock()
    result.scalars.return_value = scalars
    return AsyncMock(return_value=result)


# ─── 1. 服务层维度聚合 ──────────────────────────────────────────────


class TestServiceAggregationDimensions:
    """``aggregate_feedback`` 应在一次扫描中给出全部维度的计数。"""

    @pytest.mark.asyncio
    async def test_aggregates_all_dimensions(self):
        profile_a = uuid.uuid4()
        profile_b = uuid.uuid4()
        doc_x = "doc-X"
        doc_y = "doc-Y"

        # 构造覆盖各维度的样本：
        # - 不同 feedback_type / issue_category
        # - 不同 profile / 不同 returned_results / 不同日期
        feedbacks = [
            _feedback(
                feedback_type="thumbs_up",
                profile_id=profile_a,
                returned_results=[doc_x],
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            _feedback(
                feedback_type="thumbs_down",
                profile_id=profile_a,
                returned_results=[doc_x, doc_y],
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            _feedback(
                feedback_type="issue",
                profile_id=profile_b,
                issue_category="missing_info",
                returned_results=[doc_y],
                created_at=datetime(2024, 6, 2, tzinfo=timezone.utc),
            ),
            _feedback(
                feedback_type="issue",
                profile_id=profile_b,
                issue_category="missing_info",
                returned_results=[doc_y],
                created_at=datetime(2024, 6, 2, tzinfo=timezone.utc),
            ),
            _feedback(
                feedback_type="issue",
                profile_id=None,  # 无 Profile：不计入 by_profile
                issue_category="format",
                returned_results=[],  # 无文档引用：不计入 by_document
                created_at=datetime(2024, 6, 3, tzinfo=timezone.utc),
            ),
        ]

        db = AsyncMock()
        db.execute = _stub_execute_returning(feedbacks)
        service = FeedbackService(db)

        agg = await service.aggregate_feedback()

        # 反馈总数与三类计数
        assert agg.total_count == 5
        assert agg.thumbs_up_count == 1
        assert agg.thumbs_down_count == 1
        assert agg.issue_count == 3

        # 问题类型聚合：仅统计带有 issue_category 的样本
        assert agg.by_issue_category == {"missing_info": 2, "format": 1}

        # Profile 聚合：使用 str(uuid) 作为 key，无 Profile 的不计入
        assert agg.by_profile == {str(profile_a): 2, str(profile_b): 2}

        # 文档（returned_results）聚合：每条反馈对同一 ID 仅计 1 次
        assert agg.by_document == {doc_x: 2, doc_y: 3}

        # 时间范围（按日）聚合
        assert agg.by_date == {
            "2024-06-01": 2,
            "2024-06-02": 2,
            "2024-06-03": 1,
        }

    @pytest.mark.asyncio
    async def test_duplicate_results_in_same_feedback_count_once(self):
        """同一条反馈内重复出现的 ID 只计 1 次，避免 chunk 列表重复放大计数。"""
        feedbacks = [
            _feedback(
                feedback_type="thumbs_down",
                returned_results=["doc-1", "doc-1", "doc-1"],
            ),
        ]
        db = AsyncMock()
        db.execute = _stub_execute_returning(feedbacks)
        service = FeedbackService(db)

        agg = await service.aggregate_feedback()

        assert agg.by_document == {"doc-1": 1}

    @pytest.mark.asyncio
    async def test_empty_feedbacks_returns_zero_counts(self):
        """无反馈时各维度计数均为 0 / 空字典。"""
        db = AsyncMock()
        db.execute = _stub_execute_returning([])
        service = FeedbackService(db)

        agg = await service.aggregate_feedback()
        assert agg == FeedbackAggregation()
        assert agg.total_count == 0
        assert agg.by_profile == {}
        assert agg.by_document == {}
        assert agg.by_issue_category == {}
        assert agg.by_date == {}

    @pytest.mark.asyncio
    async def test_ignores_non_string_returned_results(self):
        """``returned_results`` 中的非字符串项被静默忽略，避免 JSONB 异常数据破坏聚合。"""
        feedbacks = [
            _feedback(
                feedback_type="thumbs_down",
                returned_results=["doc-1", None, "", 123, "doc-2"],  # type: ignore[list-item]
            ),
        ]
        db = AsyncMock()
        db.execute = _stub_execute_returning(feedbacks)
        service = FeedbackService(db)

        agg = await service.aggregate_feedback()
        assert agg.by_document == {"doc-1": 1, "doc-2": 1}


# ─── 2. 服务层过滤构造 ──────────────────────────────────────────────


class TestFilterApplication:
    """``_apply_filters`` 把 :class:`FeedbackFilter` 转换为 SQL where 条件。"""

    def test_document_id_uses_jsonb_contains(self):
        """``document_id`` 过滤使用 JSONB ``contains`` 包装为列表字面量。"""
        from sqlalchemy import select
        from sqlalchemy.dialects import postgresql

        from app.models.search_feedback import SearchFeedback

        service = FeedbackService(AsyncMock())
        filter = FeedbackFilter(document_id="doc-42")

        # 直接复用真实 select() 构造，校验 where 子句使用了 JSONB ``@>``
        # 包含运算符（PostgreSQL 方言下编译为 ``returned_results @> ...``），
        # 并且绑定参数中保留了目标 ID（包装成列表字面量）。
        query = service._apply_filters(select(SearchFeedback), filter)
        compiled = query.compile(dialect=postgresql.dialect())
        compiled_sql = str(compiled)

        assert "returned_results" in compiled_sql
        assert "@>" in compiled_sql
        # 绑定参数中应当出现「[doc-42]」结构，确保 JSONB literal 是数组
        bound_values = list(compiled.params.values())
        assert ["doc-42"] in bound_values

    def test_multiple_filters_combine_with_and(self):
        """同时设置多种过滤时各条件以 AND 关系叠加。"""
        from sqlalchemy import select
        from sqlalchemy.dialects import postgresql

        from app.models.search_feedback import SearchFeedback

        service = FeedbackService(AsyncMock())
        profile_id = uuid.uuid4()
        user_id = uuid.uuid4()
        filter = FeedbackFilter(
            profile_id=str(profile_id),
            document_id="doc-7",
            feedback_type="issue",
            issue_category="missing_info",
            start_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
            user_id=str(user_id),
        )

        query = service._apply_filters(select(SearchFeedback), filter)
        compiled_sql = str(query.compile(dialect=postgresql.dialect()))

        # 各过滤维度都应反映在 SQL 中
        assert "related_profile_id" in compiled_sql
        assert "feedback_type" in compiled_sql
        assert "issue_category" in compiled_sql
        assert "created_at" in compiled_sql
        assert "user_id" in compiled_sql
        assert "returned_results" in compiled_sql
        assert "@>" in compiled_sql
        # 多个条件之间使用 AND（PostgreSQL 编译输出包含 ``AND``）
        assert " AND " in compiled_sql.upper()

    def test_no_filter_returns_query_unchanged(self):
        """``filter=None`` 不应增加 where 子句。"""
        from sqlalchemy import select

        from app.models.search_feedback import SearchFeedback

        service = FeedbackService(AsyncMock())
        original = select(SearchFeedback)
        result = service._apply_filters(original, None)
        # SQLAlchemy 的 Select 不可变，返回的应是原始查询
        assert result is original


# ─── 3. API 路由契约 ────────────────────────────────────────────────


class _ServiceProxy:
    """轻量替身，转发到 ``aggregate_feedback`` 的 AsyncMock。

    ``app.api.feedback`` 在请求时会执行 ``FeedbackService(db)``，
    因此 patch 模块级名字到一个工厂闭包即可。
    """

    def __init__(self, mock: AsyncMock):
        self._mock = mock

    async def aggregate_feedback(self, *, filter: FeedbackFilter | None = None):
        return await self._mock(filter=filter)


def _build_app(
    *,
    aggregate_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    auth_error: Exception | None = None,
) -> FastAPI:
    """构造一个隔离 FastAPI 应用，注入聚合依赖与鉴权替身。

    通过 ``monkeypatch`` 对 ``app.api.feedback.FeedbackService`` 打桩，
    pytest 会在测试结束时自动还原全局名字，避免对其他测试模块造成污染。
    """
    from app.api import feedback as feedback_module

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(feedback_router)

    if auth_error is not None:
        async def _override_admin():
            raise auth_error
    else:
        async def _override_admin():
            user = MagicMock()
            user.id = uuid.uuid4()
            user.email = "admin@wikforge.local"
            return user

    async def _override_db():
        yield AsyncMock()

    app.dependency_overrides[require_admin] = _override_admin
    app.dependency_overrides[get_db] = _override_db

    proxy = _ServiceProxy(aggregate_mock)
    monkeypatch.setattr(feedback_module, "FeedbackService", lambda _db: proxy)

    return app


def _aggregation_result(**overrides) -> FeedbackAggregation:
    base = FeedbackAggregation()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


class TestAnalysisAuthorization:
    """``GET /api/admin/feedback/analysis`` 必须经过 ``require_admin``。"""

    def test_unauthenticated_returns_401(self, monkeypatch):
        aggregate = AsyncMock()
        app = _build_app(
            aggregate_mock=aggregate,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/analysis")
        assert resp.status_code == 401
        aggregate.assert_not_awaited()

    def test_non_admin_returns_403(self, monkeypatch):
        aggregate = AsyncMock()
        app = _build_app(
            aggregate_mock=aggregate,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/analysis")
        assert resp.status_code == 403
        aggregate.assert_not_awaited()


class TestAnalysisFilters:
    """各过滤参数应正确透传到 :class:`FeedbackFilter`。"""

    def test_no_filters_default_values(self, monkeypatch):
        aggregate = AsyncMock(return_value=_aggregation_result(total_count=0))
        app = _build_app(aggregate_mock=aggregate, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/analysis")
        assert resp.status_code == 200, resp.text

        aggregate.assert_awaited_once()
        passed: FeedbackFilter = aggregate.await_args.kwargs["filter"]
        assert passed.profile_id is None
        assert passed.document_id is None
        assert passed.feedback_type is None
        assert passed.issue_category is None
        assert passed.start_date is None
        assert passed.end_date is None

    def test_all_filters_propagated(self, monkeypatch):
        aggregate = AsyncMock(return_value=_aggregation_result())
        app = _build_app(aggregate_mock=aggregate, monkeypatch=monkeypatch)
        client = TestClient(app)

        profile_id = str(uuid.uuid4())
        params = {
            "profile_id": profile_id,
            "document_id": "doc-77",
            "feedback_type": "issue",
            "issue_category": "missing_info",
            "start_date": "2024-06-01T00:00:00",
            "end_date": "2024-06-30T23:59:59",
        }
        resp = client.get("/api/admin/feedback/analysis", params=params)
        assert resp.status_code == 200, resp.text

        passed: FeedbackFilter = aggregate.await_args.kwargs["filter"]
        assert passed.profile_id == profile_id
        assert passed.document_id == "doc-77"
        assert passed.feedback_type == "issue"
        assert passed.issue_category == "missing_info"
        assert passed.start_date == datetime.fromisoformat("2024-06-01T00:00:00")
        assert passed.end_date == datetime.fromisoformat("2024-06-30T23:59:59")

    def test_response_carries_all_aggregation_dimensions(self, monkeypatch):
        """响应字段必须包含 ``by_document``，否则前端无法按文档维度展示。"""
        agg = _aggregation_result(
            total_count=4,
            thumbs_up_count=1,
            thumbs_down_count=1,
            issue_count=2,
            by_issue_category={"missing_info": 2},
            by_profile={"profile-a": 3},
            by_document={"doc-1": 2, "doc-2": 1},
            by_date={"2024-06-01": 4},
        )
        aggregate = AsyncMock(return_value=agg)
        app = _build_app(aggregate_mock=aggregate, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/analysis")
        body = resp.json()
        assert body["total_count"] == 4
        assert body["thumbs_up_count"] == 1
        assert body["thumbs_down_count"] == 1
        assert body["issue_count"] == 2
        assert body["by_issue_category"] == {"missing_info": 2}
        assert body["by_profile"] == {"profile-a": 3}
        assert body["by_document"] == {"doc-1": 2, "doc-2": 1}
        assert body["by_date"] == {"2024-06-01": 4}
