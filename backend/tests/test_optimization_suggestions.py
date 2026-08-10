"""优化建议生成测试（任务 17.6）。

测试目标（需求 9.6 / 18.6）：

> 系统应基于错误模式生成优化建议，建议类型包括 ``adjust_chunking``、
> ``add_term``、``update_boilerplate``、``adjust_heading_rules`` 等。

覆盖三层：

1. **服务层映射**：``FeedbackService.generate_suggestions`` /
   ``_pattern_to_suggestion`` 应按 ``issue_category`` 把错误模式翻译为正确的
   :class:`SuggestionType`，并回填 ``target_id`` / ``target_name`` /
   ``description`` / ``evidence``。
2. **置信度计算**：``confidence = min(occurrence_count / 10, 1.0)``，
   ``general_negative`` 类型在此基础上额外打 0.7 折扣。
3. **API 路由契约**：``GET /api/admin/feedback/suggestions`` 通过
   ``require_admin`` 守门（未登录 401、非管理员 403），且能完整返回所有字段。

为避免引入真实数据库依赖，测试通过 mock ``detect_error_patterns`` 直接喂入
错误模式；映射逻辑本身是纯函数，可独立断言。

Validates: Requirements 9.6
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
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
    IssueCategory,
    OptimizationSuggestion,
    SuggestionType,
)

# ─── Helpers ────────────────────────────────────────────────────────


def _pattern(
    *,
    issue_category: str,
    occurrence_count: int = 3,
    profile_id: str | None = None,
    profile_name: str = "中式技术规范",
    sample_queries: list[str] | None = None,
) -> ErrorPattern:
    """构造一个错误模式样本。

    默认 ``occurrence_count=3``（恰好命中 PATTERN_DETECTION_THRESHOLD），
    ``sample_queries`` 默认两条，兼顾 evidence 字段的内容覆盖。
    """
    return ErrorPattern(
        profile_id=profile_id or str(uuid.uuid4()),
        profile_name=profile_name,
        issue_category=issue_category,
        occurrence_count=occurrence_count,
        sample_queries=sample_queries or ["查询 A", "查询 B"],
        first_seen=datetime(2024, 6, 1, tzinfo=timezone.utc),
        last_seen=datetime(2024, 6, 3, tzinfo=timezone.utc),
    )


def _make_service(patterns: list[ErrorPattern]) -> FeedbackService:
    """构造一个 :class:`FeedbackService` 替身，``detect_error_patterns`` 返回固定模式。"""
    db = AsyncMock()
    service = FeedbackService(db)

    async def _fake_detect(min_occurrences=3, days_lookback=30):  # noqa: ARG001
        return patterns

    # 直接替换实例方法，避免触发真实数据库查询
    service.detect_error_patterns = _fake_detect  # type: ignore[assignment]
    return service


# ─── 1. issue_category → SuggestionType 映射 ────────────────────────


class TestSuggestionTypeMapping:
    """需求 9.6：每种 issue_category 都应被映射到正确的 SuggestionType。"""

    @pytest.mark.asyncio
    async def test_irrelevant_maps_to_adjust_chunking(self):
        """``irrelevant`` 反馈意味着分块过大，应建议 ``adjust_chunking``。"""
        pattern = _pattern(issue_category=IssueCategory.IRRELEVANT.value)
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.type == SuggestionType.ADJUST_CHUNKING.value
        assert s.target_id == pattern.profile_id
        assert s.target_name == pattern.profile_name
        # recommendation 必须给出可执行动作
        assert s.recommendation["action"] == "reduce_max_tokens"
        assert "suggested_max_tokens" in s.recommendation

    @pytest.mark.asyncio
    async def test_missing_info_maps_to_adjust_chunking_with_overlap(self):
        """``missing_info`` 反馈意味着上下文被切断，应建议增加 ``overlap``。"""
        pattern = _pattern(issue_category=IssueCategory.MISSING_INFO.value)
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.type == SuggestionType.ADJUST_CHUNKING.value
        assert s.recommendation["action"] == "increase_overlap"
        assert "suggested_overlap_tokens" in s.recommendation

    @pytest.mark.asyncio
    async def test_citation_error_maps_to_update_boilerplate(self):
        """``citation_error`` 反馈意味着噪声未被清洗，应建议 ``update_boilerplate``。"""
        pattern = _pattern(issue_category=IssueCategory.CITATION_ERROR.value)
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.type == SuggestionType.UPDATE_BOILERPLATE.value
        assert s.recommendation["action"] == "review_boilerplate_patterns"

    @pytest.mark.asyncio
    async def test_format_maps_to_adjust_heading_rules(self):
        """``format`` 反馈意味着标题层级未识别，应建议 ``adjust_heading_rules``。"""
        pattern = _pattern(issue_category=IssueCategory.FORMAT.value)
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.type == SuggestionType.ADJUST_HEADING_RULES.value
        assert s.recommendation["action"] == "review_heading_rules"

    @pytest.mark.asyncio
    async def test_general_negative_maps_to_add_term(self):
        """``general_negative``（无明确分类的 thumbs_down）映射到 ``add_term``。

        ``ErrorPattern`` 在 ``thumbs_down`` 且未带 ``issue_category`` 时把分组
        键设为 ``general_negative``，应被翻译为「领域词典缺词」建议。
        """
        pattern = _pattern(issue_category="general_negative")
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.type == SuggestionType.ADD_TERM.value
        assert s.recommendation["action"] == "review_terminology"
        # ``general_negative`` 建议的 recommendation 必须把样本查询带出，便于
        # 管理员人工审阅候选术语
        assert s.recommendation["sample_queries"] == pattern.sample_queries

    @pytest.mark.asyncio
    async def test_other_category_yields_no_suggestion(self):
        """``IssueCategory.OTHER`` 没有对应的优化动作，应被忽略而非生成空建议。"""
        pattern = _pattern(issue_category=IssueCategory.OTHER.value)
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()

        # ``other`` 不映射到任何已知 SuggestionType，避免向管理员推送无用建议
        assert suggestions == []


# ─── 2. 置信度 / 描述 / 证据 ────────────────────────────────────────


class TestSuggestionConfidence:
    """``confidence = min(occurrence_count / 10, 1.0)``，``general_negative`` 额外 ×0.7。"""

    @pytest.mark.asyncio
    async def test_confidence_scales_with_occurrence_count(self):
        """``occurrence_count=3`` → ``0.3``。"""
        pattern = _pattern(
            issue_category=IssueCategory.IRRELEVANT.value,
            occurrence_count=3,
        )
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()
        assert suggestions[0].confidence == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_confidence_caps_at_one(self):
        """``occurrence_count >= 10`` 时置信度被截断到 1.0，不会超出。"""
        pattern = _pattern(
            issue_category=IssueCategory.FORMAT.value,
            occurrence_count=25,
        )
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()
        assert suggestions[0].confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_general_negative_confidence_is_discounted(self):
        """``general_negative`` 在基础值上再乘 0.7（信号弱于明确问题）。"""
        pattern = _pattern(
            issue_category="general_negative",
            occurrence_count=10,
        )
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()
        # 1.0 * 0.7 = 0.7
        assert suggestions[0].confidence == pytest.approx(0.7)


class TestSuggestionDescriptionAndEvidence:
    """description 应包含可读的 profile 名称和出现次数；evidence 应包含样本查询。"""

    @pytest.mark.asyncio
    async def test_description_contains_profile_name_and_count(self):
        """description 文案里必须能读到 profile_name 和 occurrence_count。"""
        pattern = _pattern(
            issue_category=IssueCategory.IRRELEVANT.value,
            occurrence_count=7,
            profile_name="水泥行业规范",
        )
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()
        desc = suggestions[0].description
        assert "水泥行业规范" in desc
        assert "7" in desc

    @pytest.mark.asyncio
    async def test_evidence_contains_sample_queries(self):
        """evidence 字段必须把 sample_queries 编码成可追溯条目。"""
        pattern = _pattern(
            issue_category=IssueCategory.MISSING_INFO.value,
            sample_queries=["合同期限是多久", "续约条款"],
        )
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()
        evidence = suggestions[0].evidence
        # 每条 evidence 都应能在内容中识别到原始查询
        joined = "\n".join(evidence)
        assert "合同期限是多久" in joined
        assert "续约条款" in joined

    @pytest.mark.asyncio
    async def test_target_id_and_name_match_pattern(self):
        """target_id / target_name 必须分别等于 pattern 的 profile_id / profile_name。"""
        profile_id = str(uuid.uuid4())
        pattern = _pattern(
            issue_category=IssueCategory.CITATION_ERROR.value,
            profile_id=profile_id,
            profile_name="技术白皮书",
        )
        service = _make_service([pattern])

        suggestions = await service.generate_suggestions()
        s = suggestions[0]
        assert s.target_id == profile_id
        assert s.target_name == "技术白皮书"


# ─── 3. 多模式 / 空模式 ─────────────────────────────────────────────


class TestMultiplePatterns:
    """多个错误模式应各自生成独立建议；无模式时返回空列表。"""

    @pytest.mark.asyncio
    async def test_multiple_patterns_each_yield_one_suggestion(self):
        """三个不同的 issue_category 应生成三条不同类型的建议。"""
        patterns = [
            _pattern(issue_category=IssueCategory.IRRELEVANT.value),
            _pattern(issue_category=IssueCategory.CITATION_ERROR.value),
            _pattern(issue_category=IssueCategory.FORMAT.value),
        ]
        service = _make_service(patterns)

        suggestions = await service.generate_suggestions()

        assert len(suggestions) == 3
        types = {s.type for s in suggestions}
        assert types == {
            SuggestionType.ADJUST_CHUNKING.value,
            SuggestionType.UPDATE_BOILERPLATE.value,
            SuggestionType.ADJUST_HEADING_RULES.value,
        }

    @pytest.mark.asyncio
    async def test_empty_patterns_returns_empty_list(self):
        """没有错误模式时应返回空列表，而非 None。"""
        service = _make_service([])

        suggestions = await service.generate_suggestions()
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_pre_detected_patterns_used_directly(self):
        """显式传入 patterns 时不应再调用 ``detect_error_patterns``。"""
        db = AsyncMock()
        service = FeedbackService(db)

        # 监听器：若被调用则失败
        async def _should_not_be_called(**_kwargs):
            raise AssertionError("不应在显式传入 patterns 时再去检测")

        service.detect_error_patterns = _should_not_be_called  # type: ignore[assignment]

        patterns = [_pattern(issue_category=IssueCategory.IRRELEVANT.value)]
        suggestions = await service.generate_suggestions(patterns=patterns)
        assert len(suggestions) == 1
        assert suggestions[0].type == SuggestionType.ADJUST_CHUNKING.value


# ─── 4. API 路由契约 ────────────────────────────────────────────────


class _ServiceProxy:
    """轻量替身，转发到 ``generate_suggestions`` 的 AsyncMock。"""

    def __init__(self, mock: AsyncMock):
        self._mock = mock

    async def generate_suggestions(self):
        return await self._mock()


def _build_app(
    *,
    suggestions_mock: AsyncMock,
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

    proxy = _ServiceProxy(suggestions_mock)
    monkeypatch.setattr(feedback_module, "FeedbackService", lambda _db: proxy)

    return app


def _sample_suggestion(
    *,
    type_: str = SuggestionType.ADJUST_CHUNKING.value,
    confidence: float = 0.5,
) -> OptimizationSuggestion:
    return OptimizationSuggestion(
        type=type_,
        target_id=str(uuid.uuid4()),
        target_name="中式技术规范",
        recommendation={"action": "reduce_max_tokens", "suggested_max_tokens": 512},
        evidence=["query: 合同期限"],
        confidence=confidence,
        description="Profile '中式技术规范' 下有 5 次反馈",
    )


class TestSuggestionsAuthorization:
    """``GET /api/admin/feedback/suggestions`` 必须经过 ``require_admin``。"""

    def test_unauthenticated_returns_401(self, monkeypatch):
        suggestions = AsyncMock()
        app = _build_app(
            suggestions_mock=suggestions,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/suggestions")
        assert resp.status_code == 401
        suggestions.assert_not_awaited()

    def test_non_admin_returns_403(self, monkeypatch):
        suggestions = AsyncMock()
        app = _build_app(
            suggestions_mock=suggestions,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/suggestions")
        assert resp.status_code == 403
        suggestions.assert_not_awaited()


class TestSuggestionsApiContract:
    """API 应完整地序列化 :class:`OptimizationSuggestion` 字段。"""

    def test_empty_suggestions_returns_empty_list(self, monkeypatch):
        suggestions = AsyncMock(return_value=[])
        app = _build_app(suggestions_mock=suggestions, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/suggestions")
        assert resp.status_code == 200
        assert resp.json() == []
        suggestions.assert_awaited_once()

    def test_response_carries_all_suggestion_fields(self, monkeypatch):
        suggestion = _sample_suggestion(confidence=0.42)
        suggestions = AsyncMock(return_value=[suggestion])
        app = _build_app(suggestions_mock=suggestions, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/suggestions")
        assert resp.status_code == 200

        body = resp.json()
        assert len(body) == 1
        item = body[0]
        assert item["type"] == suggestion.type
        assert item["target_id"] == suggestion.target_id
        assert item["target_name"] == suggestion.target_name
        assert item["recommendation"] == suggestion.recommendation
        assert item["evidence"] == suggestion.evidence
        assert item["confidence"] == pytest.approx(0.42)
        assert item["description"] == suggestion.description

    def test_multiple_suggestions_preserve_order(self, monkeypatch):
        first = _sample_suggestion(type_=SuggestionType.ADJUST_CHUNKING.value)
        second = _sample_suggestion(type_=SuggestionType.UPDATE_BOILERPLATE.value)
        third = _sample_suggestion(type_=SuggestionType.ADD_TERM.value)
        suggestions = AsyncMock(return_value=[first, second, third])
        app = _build_app(suggestions_mock=suggestions, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/suggestions")
        assert resp.status_code == 200
        body = resp.json()
        assert [item["type"] for item in body] == [
            SuggestionType.ADJUST_CHUNKING.value,
            SuggestionType.UPDATE_BOILERPLATE.value,
            SuggestionType.ADD_TERM.value,
        ]
