"""错误模式检测测试（任务 17.5）。

测试目标（需求 9.5 / 18.5）：

> 系统应检测错误模式：同一 Profile 下相同 issue_category 重复出现 N 次
> （默认 N=3）触发优化建议生成。

覆盖三层：

1. **服务层算法**：``FeedbackService.detect_error_patterns`` 按
   ``(profile_id, issue_category)`` 分组、过滤
   ``occurrence_count >= min_occurrences``、收集样本查询、计算时间窗。
2. **边界条件**：低于阈值不触发、时间窗外被过滤、sample_queries 上限 5、
   thumbs_down 无 issue_category 时归为 ``general_negative``、无 profile
   关联的反馈不参与统计。
3. **API 层**：``GET /api/admin/feedback/patterns`` 通过 ``require_admin``
   守门，未登录 401、非管理员 403、管理员 200，且 ``min_occurrences``
   /``days_lookback`` 参数能正确透传。

为避免引入真实 PostgreSQL 依赖，服务层测试通过 mock 化 ``db.execute`` 直接
喂入「过滤后」的反馈样本，重点覆盖分组 / 阈值 / 样本上限 / 时间戳计算。
时间窗过滤本身由专门的查询条件单测覆盖。

Validates: Requirements 9.5
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
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
    ErrorPattern,
    FeedbackService,
)


# ─── Helpers ────────────────────────────────────────────────────────


def _feedback(
    *,
    feedback_type: str,
    profile_id: uuid.UUID | None = None,
    issue_category: str | None = None,
    query: str = "默认查询",
    created_at: datetime | None = None,
) -> SimpleNamespace:
    """构造一条「足够用」的反馈样本。

    使用 :class:`SimpleNamespace` 替身而非真实 ORM 实例，避免依赖 SQLAlchemy
    元数据初始化；服务层只读取 ``related_profile_id`` / ``issue_category`` /
    ``query`` / ``created_at`` 等属性即可完成分组。
    """
    return SimpleNamespace(
        feedback_type=feedback_type,
        related_profile_id=profile_id,
        issue_category=issue_category,
        query=query,
        created_at=created_at or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _stub_db(
    feedbacks: list[SimpleNamespace],
    profile_names: dict[uuid.UUID, str] | None = None,
) -> AsyncMock:
    """构造 ``db.execute`` 的双行为 mock：

    - 首次调用（``select(SearchFeedback)``）：返回反馈列表
    - 后续调用（``select(DocumentProfile.name)``）：根据 profile_id 返回名称

    通过 ``side_effect`` 让 mock 按顺序响应不同的查询，避免 patcher 重复设置。
    """
    profile_names = profile_names or {}

    # 第一次返回反馈集合
    feedback_scalars = MagicMock()
    feedback_scalars.all.return_value = list(feedbacks)
    feedback_result = MagicMock()
    feedback_result.scalars.return_value = feedback_scalars

    # 后续每次返回的名称根据查询参数决定 —— 这里用一个简单的 FIFO 策略：
    # 服务层按 pattern_groups 顺序逐个调用，正好对应 _get_profile_name。
    name_results: list[MagicMock] = []
    for fb in feedbacks:
        name_result = MagicMock()
        name_result.scalar_one_or_none.return_value = profile_names.get(
            fb.related_profile_id
        )
        name_results.append(name_result)

    call_log: list = [feedback_result, *name_results]
    call_iter = iter(call_log)

    db = AsyncMock()

    async def _execute(*_args, **_kwargs):
        try:
            return next(call_iter)
        except StopIteration:
            # 兜底：意外调用时返回空结果，避免测试无关报错
            empty = MagicMock()
            empty.scalar_one_or_none.return_value = None
            empty.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))
            return empty

    db.execute = AsyncMock(side_effect=_execute)
    return db


# ─── 1. 服务层算法 ──────────────────────────────────────────────────


class TestErrorPatternDetectionAlgorithm:
    """``detect_error_patterns`` 应按 ``(profile_id, issue_category)`` 分组并过滤。"""

    @pytest.mark.asyncio
    async def test_single_profile_repeated_issue_category_detected(self):
        """同一 Profile 下同一 issue_category 重复 N 次 → 检测到一个 pattern。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="missing_info",
                query=f"查询 {i}",
                created_at=datetime(2024, 6, i + 1, tzinfo=timezone.utc),
            )
            for i in range(3)
        ]

        db = _stub_db(feedbacks, profile_names={profile_id: "中式技术规范"})
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)

        assert len(patterns) == 1
        p = patterns[0]
        assert p.profile_id == str(profile_id)
        assert p.profile_name == "中式技术规范"
        assert p.issue_category == "missing_info"
        assert p.occurrence_count == 3
        assert p.first_seen == datetime(2024, 6, 1, tzinfo=timezone.utc)
        assert p.last_seen == datetime(2024, 6, 3, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_multiple_profiles_each_with_distinct_category(self):
        """多个 Profile 各自有不同 issue_category 重复 → 检测到多个 pattern。"""
        profile_a = uuid.uuid4()
        profile_b = uuid.uuid4()
        feedbacks = [
            # Profile A：3 次 irrelevant
            *[
                _feedback(
                    feedback_type="issue",
                    profile_id=profile_a,
                    issue_category="irrelevant",
                    query=f"A-{i}",
                )
                for i in range(3)
            ],
            # Profile B：4 次 format
            *[
                _feedback(
                    feedback_type="issue",
                    profile_id=profile_b,
                    issue_category="format",
                    query=f"B-{i}",
                )
                for i in range(4)
            ],
        ]

        db = _stub_db(
            feedbacks,
            profile_names={profile_a: "Profile A", profile_b: "Profile B"},
        )
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)

        # 排序按出现次数降序
        assert len(patterns) == 2
        assert patterns[0].occurrence_count == 4
        assert patterns[0].profile_id == str(profile_b)
        assert patterns[0].issue_category == "format"
        assert patterns[1].occurrence_count == 3
        assert patterns[1].profile_id == str(profile_a)
        assert patterns[1].issue_category == "irrelevant"

    @pytest.mark.asyncio
    async def test_below_threshold_does_not_trigger(self):
        """occurrence_count < min_occurrences 的分组不应触发 pattern。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="missing_info",
            )
            for _ in range(2)  # 低于默认阈值 3
        ]

        db = _stub_db(feedbacks)
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)
        assert patterns == []

    @pytest.mark.asyncio
    async def test_custom_min_occurrences_threshold(self):
        """自定义阈值应改变触发条件。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="format",
            )
            for _ in range(5)
        ]

        db = _stub_db(feedbacks, profile_names={profile_id: "P"})
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=10)
        assert patterns == []

    @pytest.mark.asyncio
    async def test_sample_queries_capped_at_five(self):
        """``sample_queries`` 最多保留 5 个样本，避免响应膨胀。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="irrelevant",
                query=f"查询 {i}",
                created_at=datetime(2024, 6, 1, 12, i, 0, tzinfo=timezone.utc),
            )
            for i in range(8)
        ]

        db = _stub_db(feedbacks, profile_names={profile_id: "P"})
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)
        assert len(patterns) == 1
        assert patterns[0].occurrence_count == 8
        assert len(patterns[0].sample_queries) == 5
        # 必须是字符串列表，且都来源于反馈样本
        assert all(isinstance(q, str) for q in patterns[0].sample_queries)
        assert all(q.startswith("查询") for q in patterns[0].sample_queries)

    @pytest.mark.asyncio
    async def test_thumbs_down_without_issue_category_grouped_as_general(self):
        """thumbs_down 即使无 issue_category 也算入 ``general_negative`` 分组。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            _feedback(
                feedback_type="thumbs_down",
                profile_id=profile_id,
                issue_category=None,
                query=f"未明确问题 {i}",
            )
            for i in range(3)
        ]

        db = _stub_db(feedbacks, profile_names={profile_id: "通用文本"})
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)

        assert len(patterns) == 1
        assert patterns[0].issue_category == "general_negative"
        assert patterns[0].occurrence_count == 3
        assert patterns[0].profile_name == "通用文本"

    @pytest.mark.asyncio
    async def test_first_seen_and_last_seen_reflect_extreme_timestamps(self):
        """first_seen / last_seen 必须分别等于该分组内的最早与最晚时间。"""
        profile_id = uuid.uuid4()
        # 故意打乱顺序，确保实现使用 min/max 而非首尾
        feedbacks = [
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="format",
                created_at=datetime(2024, 6, 5, tzinfo=timezone.utc),
            ),
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="format",
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="format",
                created_at=datetime(2024, 6, 10, tzinfo=timezone.utc),
            ),
        ]

        db = _stub_db(feedbacks, profile_names={profile_id: "P"})
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)
        assert len(patterns) == 1
        assert patterns[0].first_seen == datetime(2024, 6, 1, tzinfo=timezone.utc)
        assert patterns[0].last_seen == datetime(2024, 6, 10, tzinfo=timezone.utc)

    @pytest.mark.asyncio
    async def test_unknown_profile_name_falls_back_to_placeholder(self):
        """Profile 已被删除（名称查询返回 None）时使用占位符，保证响应完整。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            _feedback(
                feedback_type="issue",
                profile_id=profile_id,
                issue_category="other",
            )
            for _ in range(3)
        ]

        # profile_names 缺失 → _get_profile_name 返回 None
        db = _stub_db(feedbacks)
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)
        assert len(patterns) == 1
        assert patterns[0].profile_name == "Unknown"

    @pytest.mark.asyncio
    async def test_mixed_categories_under_same_profile_are_separate_patterns(self):
        """同一 Profile 下不同 issue_category 应分组为独立的 pattern。"""
        profile_id = uuid.uuid4()
        feedbacks = [
            *[
                _feedback(
                    feedback_type="issue",
                    profile_id=profile_id,
                    issue_category="irrelevant",
                )
                for _ in range(3)
            ],
            *[
                _feedback(
                    feedback_type="issue",
                    profile_id=profile_id,
                    issue_category="missing_info",
                )
                for _ in range(3)
            ],
        ]

        db = _stub_db(feedbacks, profile_names={profile_id: "P"})
        service = FeedbackService(db)

        patterns = await service.detect_error_patterns(min_occurrences=3)
        assert len(patterns) == 2
        categories = {p.issue_category for p in patterns}
        assert categories == {"irrelevant", "missing_info"}


# ─── 2. 时间窗过滤的查询条件 ────────────────────────────────────────


class TestLookbackWindowFilter:
    """``days_lookback`` 必须被编译为 ``created_at >= cutoff`` 的 SQL where。"""

    @pytest.mark.asyncio
    async def test_query_includes_created_at_lower_bound(self):
        """detect_error_patterns 构造的查询必须带有 created_at 时间下界。"""
        from sqlalchemy import select  # noqa: F401  ensure available
        from sqlalchemy.dialects import postgresql

        captured: dict[str, object] = {}

        async def _capture_execute(query):
            captured["query"] = query
            scalars = MagicMock()
            scalars.all.return_value = []
            res = MagicMock()
            res.scalars.return_value = scalars
            return res

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=_capture_execute)
        service = FeedbackService(db)

        before = datetime.now(timezone.utc)
        await service.detect_error_patterns(min_occurrences=3, days_lookback=7)
        after = datetime.now(timezone.utc)

        compiled = captured["query"].compile(  # type: ignore[union-attr]
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
        compiled_sql = str(compiled)

        # created_at 下界 + 必须存在 profile 关联 + 反馈类型为负面集合
        assert "created_at" in compiled_sql
        assert "related_profile_id" in compiled_sql
        assert "feedback_type" in compiled_sql

        # 绑定值中应当包含一个时间戳，且应当落在 [before-7d, after-7d] 区间内
        bound_values = list(compiled.params.values())
        cutoff_candidates = [
            v for v in bound_values if isinstance(v, datetime)
        ]
        assert cutoff_candidates, "查询未绑定 created_at 截止时间"
        cutoff = cutoff_candidates[0]
        assert before - timedelta(days=7, seconds=1) <= cutoff <= after - timedelta(days=7) + timedelta(seconds=1)


# ─── 3. API 路由契约 ────────────────────────────────────────────────


class _ServiceProxy:
    """轻量替身，转发到 ``detect_error_patterns`` 的 AsyncMock。"""

    def __init__(self, mock: AsyncMock):
        self._mock = mock

    async def detect_error_patterns(
        self,
        *,
        min_occurrences: int = 3,
        days_lookback: int = 30,
    ):
        return await self._mock(
            min_occurrences=min_occurrences,
            days_lookback=days_lookback,
        )


def _build_app(
    *,
    detect_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    auth_error: Exception | None = None,
) -> FastAPI:
    """构造隔离 FastAPI 应用，注入鉴权与服务替身。"""
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

    proxy = _ServiceProxy(detect_mock)
    monkeypatch.setattr(feedback_module, "FeedbackService", lambda _db: proxy)

    return app


def _sample_pattern(
    *,
    profile_id: str | None = None,
    issue_category: str = "missing_info",
    occurrence_count: int = 3,
) -> ErrorPattern:
    return ErrorPattern(
        profile_id=profile_id or str(uuid.uuid4()),
        profile_name="中式技术规范",
        issue_category=issue_category,
        occurrence_count=occurrence_count,
        sample_queries=["查询 A", "查询 B"],
        first_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
        last_seen=datetime(2024, 6, 3, tzinfo=timezone.utc),
    )


class TestPatternsAuthorization:
    """``GET /api/admin/feedback/patterns`` 必须经过 ``require_admin``。"""

    def test_unauthenticated_returns_401(self, monkeypatch):
        detect = AsyncMock()
        app = _build_app(
            detect_mock=detect,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/patterns")
        assert resp.status_code == 401
        detect.assert_not_awaited()

    def test_non_admin_returns_403(self, monkeypatch):
        detect = AsyncMock()
        app = _build_app(
            detect_mock=detect,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/patterns")
        assert resp.status_code == 403
        detect.assert_not_awaited()


class TestPatternsApiContract:
    """API 应正确透传查询参数并完整响应识别到的模式。"""

    def test_default_parameters_are_used(self, monkeypatch):
        detect = AsyncMock(return_value=[])
        app = _build_app(detect_mock=detect, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/patterns")
        assert resp.status_code == 200
        assert resp.json() == []
        detect.assert_awaited_once_with(min_occurrences=3, days_lookback=30)

    def test_custom_parameters_propagated(self, monkeypatch):
        detect = AsyncMock(return_value=[])
        app = _build_app(detect_mock=detect, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get(
            "/api/admin/feedback/patterns",
            params={"min_occurrences": 5, "days_lookback": 14},
        )
        assert resp.status_code == 200
        detect.assert_awaited_once_with(min_occurrences=5, days_lookback=14)

    def test_response_carries_all_pattern_fields(self, monkeypatch):
        pattern = _sample_pattern(occurrence_count=4)
        detect = AsyncMock(return_value=[pattern])
        app = _build_app(detect_mock=detect, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/patterns")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        item = body[0]
        assert item["profile_id"] == pattern.profile_id
        assert item["profile_name"] == pattern.profile_name
        assert item["issue_category"] == pattern.issue_category
        assert item["occurrence_count"] == pattern.occurrence_count
        assert item["sample_queries"] == pattern.sample_queries
        # FastAPI 序列化 datetime 为 ISO 字符串
        assert item["first_seen"].startswith("2024-06-01")
        assert item["last_seen"].startswith("2024-06-03")

    def test_min_occurrences_below_one_rejected(self, monkeypatch):
        """``min_occurrences`` 必须 >= 1，否则 422。"""
        detect = AsyncMock(return_value=[])
        app = _build_app(detect_mock=detect, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get(
            "/api/admin/feedback/patterns",
            params={"min_occurrences": 0},
        )
        assert resp.status_code == 422
        detect.assert_not_awaited()

    def test_days_lookback_out_of_range_rejected(self, monkeypatch):
        """``days_lookback`` 取值必须在 [1, 365] 内，否则 422。"""
        detect = AsyncMock(return_value=[])
        app = _build_app(detect_mock=detect, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get(
            "/api/admin/feedback/patterns",
            params={"days_lookback": 400},
        )
        assert resp.status_code == 422
        detect.assert_not_awaited()
