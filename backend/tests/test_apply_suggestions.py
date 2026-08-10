"""一键更新 Profile / 词典 API 测试（任务 17.7 / 需求 9.7 / 18.6）。

被测路由：

- ``POST /api/admin/feedback/apply/profile``
- ``POST /api/admin/feedback/apply/dictionary``

覆盖维度：

1. **路由层契约**：
   - 应用 Profile 更新成功 → ``profile.version`` +1，返回 ``ApplyResponse``。
   - 应用 Dictionary 更新成功 → 新 terms 合并去重。
   - Profile / Dictionary 不存在 → 404。
   - 受影响文档非空 → 返回 ``reprocessing_task_id``；为空 → ``None``。
   - 鉴权：未登录返回 401；登录但非管理员返回 403，且不调用业务服务。
2. **服务层去重**：
   :meth:`FeedbackService.apply_dictionary_update` 必须按 ``word`` 字段
   去重，避免相同术语重复进入 ``DomainDictionary.terms``。

策略：
- 使用 ``app.dependency_overrides`` 替换鉴权依赖，避免真实 JWT 流程。
- 通过 ``monkeypatch`` 把模块级 ``FeedbackService`` 替换为代理对象，
  转发到一个 ``AsyncMock`` 服务替身，便于断言调用参数。
- 服务层的合并去重逻辑直接用真实 :class:`FeedbackService`（搭配
  AsyncSession Mock），确保产生回归保护。

Validates: Requirements 9.7
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import feedback as feedback_module
from app.api.auth import require_admin
from app.api.feedback import router as feedback_router
from app.core.database import get_db
from app.core.exceptions import (
    ForbiddenException,
    UnauthorizedException,
    register_exception_handlers,
)
from app.services.feedback_service import FeedbackService, ReprocessingTask


# ─── 路由层 fixture ────────────────────────────────────────────────


class _ServiceProxy:
    """轻量替身：把 ``FeedbackService(db)`` 调用转发到注入的 AsyncMock。

    通过这种方式可以让多个异步方法（``apply_profile_update`` /
    ``apply_dictionary_update`` / ``get_affected_documents`` /
    ``trigger_reprocessing``）使用同一个 mock，断言更直观。
    """

    def __init__(self, mock: AsyncMock):
        self._mock = mock

    async def apply_profile_update(self, **kwargs):
        return await self._mock.apply_profile_update(**kwargs)

    async def apply_dictionary_update(self, **kwargs):
        return await self._mock.apply_dictionary_update(**kwargs)

    async def get_affected_documents(self, **kwargs):
        return await self._mock.get_affected_documents(**kwargs)

    async def trigger_reprocessing(self, document_ids):
        return await self._mock.trigger_reprocessing(document_ids)


def _build_app(
    *,
    service_mock: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
    auth_error: Exception | None = None,
) -> FastAPI:
    """构造隔离 FastAPI 应用，注入鉴权与服务替身。"""
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

    proxy = _ServiceProxy(service_mock)
    monkeypatch.setattr(feedback_module, "FeedbackService", lambda _db: proxy)

    return app


def _fake_profile(*, name: str = "中式技术规范", version: int = 2) -> MagicMock:
    """构造 Profile ORM 对象的轻量替身。"""
    profile = MagicMock()
    profile.id = uuid.uuid4()
    profile.name = name
    profile.version = version
    return profile


def _fake_dictionary(*, name: str = "电力术语", terms: list | None = None) -> MagicMock:
    dictionary = MagicMock()
    dictionary.id = uuid.uuid4()
    dictionary.name = name
    dictionary.terms = terms if terms is not None else []
    return dictionary


def _fake_document() -> MagicMock:
    doc = MagicMock()
    doc.id = uuid.uuid4()
    return doc


def _fake_task(task_id: str = "task-123", total: int = 5) -> ReprocessingTask:
    return ReprocessingTask(
        task_id=task_id,
        total_documents=total,
        processed_documents=0,
        status="running",
        created_at=datetime.now(timezone.utc),
    )


# ─── 1. POST /api/admin/feedback/apply/profile ─────────────────────


class TestApplyProfile:
    """``POST /api/admin/feedback/apply/profile`` 行为契约。"""

    def test_apply_success_returns_response_with_task_id(self, monkeypatch):
        """成功路径：Profile 已更新到 version+1，且返回重处理任务 ID。"""
        profile_id = str(uuid.uuid4())
        profile = _fake_profile(name="中式技术规范", version=3)
        documents = [_fake_document() for _ in range(2)]
        task = _fake_task(task_id="task-abc", total=2)

        service = AsyncMock()
        service.apply_profile_update = AsyncMock(return_value=profile)
        service.get_affected_documents = AsyncMock(return_value=documents)
        service.trigger_reprocessing = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/profile",
            json={
                "profile_id": profile_id,
                "updates": {
                    "chunking": {"max_tokens": 512, "overlap_tokens": 100},
                },
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["reprocessing_task_id"] == "task-abc"
        # 文案应当揭示版本号与重处理文档数量
        assert "version 3" in body["message"]
        assert "2 documents" in body["message"]
        # 入参透传：profile_id / updates
        service.apply_profile_update.assert_awaited_once_with(
            profile_id=profile_id,
            updates={"chunking": {"max_tokens": 512, "overlap_tokens": 100}},
        )
        # 受影响文档查询使用 profile_id
        service.get_affected_documents.assert_awaited_once_with(profile_id=profile_id)
        # 重处理任务被提交，参数为文档 id 列表
        triggered_doc_ids = service.trigger_reprocessing.await_args.args[0]
        assert triggered_doc_ids == [str(doc.id) for doc in documents]

    def test_apply_success_without_affected_documents_skips_reprocessing(
        self, monkeypatch
    ):
        """没有受影响文档时 ``reprocessing_task_id`` 应为 ``None``，且不触发任务。"""
        profile = _fake_profile(name="通用文本文档", version=2)

        service = AsyncMock()
        service.apply_profile_update = AsyncMock(return_value=profile)
        service.get_affected_documents = AsyncMock(return_value=[])
        service.trigger_reprocessing = AsyncMock()

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/profile",
            json={
                "profile_id": str(uuid.uuid4()),
                "updates": {"boilerplate": {"patterns": []}},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["reprocessing_task_id"] is None
        assert "0 documents" in body["message"]
        # 没有文档时不应调用 trigger_reprocessing
        service.trigger_reprocessing.assert_not_awaited()

    def test_apply_unknown_profile_returns_404(self, monkeypatch):
        """Profile 不存在 → 404，且不应再触发后续重处理。"""
        service = AsyncMock()
        service.apply_profile_update = AsyncMock(return_value=None)
        service.get_affected_documents = AsyncMock()
        service.trigger_reprocessing = AsyncMock()

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/profile",
            json={
                "profile_id": str(uuid.uuid4()),
                "updates": {"chunking": {"max_tokens": 256}},
            },
        )

        assert resp.status_code == 404
        # Profile 不存在时不应继续查询受影响文档或触发重处理
        service.get_affected_documents.assert_not_awaited()
        service.trigger_reprocessing.assert_not_awaited()

    def test_apply_missing_fields_returns_422(self, monkeypatch):
        """缺少必填字段触发 Pydantic 422。"""
        service = AsyncMock()
        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/profile",
            json={"profile_id": str(uuid.uuid4())},  # 缺少 updates
        )
        assert resp.status_code == 422
        service.apply_profile_update.assert_not_awaited()


# ─── 2. POST /api/admin/feedback/apply/dictionary ──────────────────


class TestApplyDictionary:
    """``POST /api/admin/feedback/apply/dictionary`` 行为契约。"""

    def test_apply_success_returns_response_with_task_id(self, monkeypatch):
        """成功路径：词典合并新 terms，并触发重处理任务。"""
        dictionary_id = str(uuid.uuid4())
        dictionary = _fake_dictionary(
            name="电力术语",
            terms=[{"word": "变压器", "weight": 1.0}],
        )
        documents = [_fake_document() for _ in range(3)]
        task = _fake_task(task_id="task-dict-1", total=3)

        service = AsyncMock()
        service.apply_dictionary_update = AsyncMock(return_value=dictionary)
        service.get_affected_documents = AsyncMock(return_value=documents)
        service.trigger_reprocessing = AsyncMock(return_value=task)

        new_terms = [
            {"word": "继电器", "weight": 0.8},
            {"word": "断路器", "weight": 0.9},
        ]
        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/dictionary",
            json={"dictionary_id": dictionary_id, "new_terms": new_terms},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["reprocessing_task_id"] == "task-dict-1"
        assert "2 terms" in body["message"]
        assert "3 documents" in body["message"]
        # 入参透传
        service.apply_dictionary_update.assert_awaited_once_with(
            dictionary_id=dictionary_id, new_terms=new_terms
        )
        service.get_affected_documents.assert_awaited_once_with(
            dictionary_id=dictionary_id
        )

    def test_apply_success_without_affected_documents_skips_reprocessing(
        self, monkeypatch
    ):
        """无关联文档时同样允许返回 200，且 ``reprocessing_task_id`` 为 ``None``。"""
        dictionary = _fake_dictionary(name="通用中文停用词")

        service = AsyncMock()
        service.apply_dictionary_update = AsyncMock(return_value=dictionary)
        service.get_affected_documents = AsyncMock(return_value=[])
        service.trigger_reprocessing = AsyncMock()

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/dictionary",
            json={
                "dictionary_id": str(uuid.uuid4()),
                "new_terms": [{"word": "兆瓦"}],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["reprocessing_task_id"] is None
        service.trigger_reprocessing.assert_not_awaited()

    def test_apply_unknown_dictionary_returns_404(self, monkeypatch):
        """词典不存在 → 404。"""
        service = AsyncMock()
        service.apply_dictionary_update = AsyncMock(return_value=None)
        service.get_affected_documents = AsyncMock()
        service.trigger_reprocessing = AsyncMock()

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/apply/dictionary",
            json={
                "dictionary_id": str(uuid.uuid4()),
                "new_terms": [{"word": "测试"}],
            },
        )

        assert resp.status_code == 404
        service.get_affected_documents.assert_not_awaited()
        service.trigger_reprocessing.assert_not_awaited()


# ─── 3. 鉴权：未登录 / 非管理员 ─────────────────────────────────────


class TestApplyAuthorization:
    """``apply/profile`` 与 ``apply/dictionary`` 必须通过 ``require_admin``。"""

    def test_profile_unauthenticated_returns_401(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/admin/feedback/apply/profile",
            json={"profile_id": str(uuid.uuid4()), "updates": {}},
        )
        assert resp.status_code == 401
        # 鉴权失败不应继续触达业务服务
        service.apply_profile_update.assert_not_awaited()

    def test_profile_non_admin_returns_403(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/admin/feedback/apply/profile",
            json={"profile_id": str(uuid.uuid4()), "updates": {}},
        )
        assert resp.status_code == 403
        service.apply_profile_update.assert_not_awaited()

    def test_dictionary_unauthenticated_returns_401(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/admin/feedback/apply/dictionary",
            json={"dictionary_id": str(uuid.uuid4()), "new_terms": []},
        )
        assert resp.status_code == 401
        service.apply_dictionary_update.assert_not_awaited()

    def test_dictionary_non_admin_returns_403(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/admin/feedback/apply/dictionary",
            json={"dictionary_id": str(uuid.uuid4()), "new_terms": []},
        )
        assert resp.status_code == 403
        service.apply_dictionary_update.assert_not_awaited()


# ─── 4. 服务层去重 / 版本递增（直接调用 FeedbackService）───────────


class _FakeScalarResult:
    """模拟 ``await db.execute(...)`` 返回的标量结果。"""

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _DbStub:
    """轻量 AsyncSession 替身：仅模拟 ``execute`` / ``flush`` / ``refresh``。"""

    def __init__(self, scalar_value):
        self._scalar_value = scalar_value
        self.flush_called = False

    async def execute(self, _stmt):
        return _FakeScalarResult(self._scalar_value)

    async def flush(self):
        self.flush_called = True

    async def refresh(self, _obj):
        return None


class _ProfileFake:
    """简化的 DocumentProfile 行为：可改变 chunking / boilerplate / version。"""

    def __init__(self, name: str, version: int = 1):
        self.id = uuid.uuid4()
        self.name = name
        self.version = version
        self.chunking = {}
        self.boilerplate = {}
        self.heading_rules = []


class _DictionaryFake:
    def __init__(self, name: str, terms: list | None = None):
        self.id = uuid.uuid4()
        self.name = name
        self.terms = list(terms) if terms is not None else []


class TestApplyProfileServiceVersionBump:
    """:meth:`FeedbackService.apply_profile_update` 必须把 ``version`` +1。"""

    @pytest.mark.asyncio
    async def test_version_increments_on_update(self):
        """传入 ``chunking`` 字段后，profile.version 严格 +1。"""
        profile = _ProfileFake(name="通用文本文档", version=2)
        db = _DbStub(scalar_value=profile)

        service = FeedbackService(db)  # type: ignore[arg-type]

        result = await service.apply_profile_update(
            profile_id=str(profile.id),
            updates={
                "chunking": {"max_tokens": 512},
                "boilerplate": {"patterns": [r"^第\d+页$"]},
                "heading_rules": [{"pattern": "^# "}],
            },
        )

        assert result is profile
        assert profile.version == 3
        assert profile.chunking == {"max_tokens": 512}
        assert profile.boilerplate == {"patterns": [r"^第\d+页$"]}
        assert profile.heading_rules == [{"pattern": "^# "}]
        assert db.flush_called is True

    @pytest.mark.asyncio
    async def test_returns_none_when_profile_missing(self):
        """目标 profile 不存在时返回 ``None``，不应抛异常。"""
        db = _DbStub(scalar_value=None)
        service = FeedbackService(db)  # type: ignore[arg-type]

        result = await service.apply_profile_update(
            profile_id=str(uuid.uuid4()),
            updates={"chunking": {"max_tokens": 256}},
        )
        assert result is None


class TestApplyDictionaryServiceMergeDedup:
    """:meth:`FeedbackService.apply_dictionary_update` 按 ``word`` 去重。"""

    @pytest.mark.asyncio
    async def test_new_terms_merged_without_duplicates(self):
        """已有术语不会被再次写入，保留首次出现的 term 字典。"""
        dictionary = _DictionaryFake(
            name="电力术语",
            terms=[{"word": "变压器", "weight": 1.0}],
        )
        db = _DbStub(scalar_value=dictionary)
        service = FeedbackService(db)  # type: ignore[arg-type]

        result = await service.apply_dictionary_update(
            dictionary_id=str(dictionary.id),
            new_terms=[
                {"word": "变压器", "weight": 0.5},  # 已存在，应被跳过
                {"word": "断路器", "weight": 0.9},
                {"word": "继电器", "weight": 0.8},
            ],
        )

        assert result is dictionary
        words = [t["word"] if isinstance(t, dict) else t for t in dictionary.terms]
        # 顺序：原有在前，新增按入参顺序追加；"变压器" 不应重复
        assert words == ["变压器", "断路器", "继电器"]
        # 已有的 "变压器" 权重保持原值，不被覆盖
        first_term = dictionary.terms[0]
        assert isinstance(first_term, dict)
        assert first_term["weight"] == 1.0
        assert db.flush_called is True

    @pytest.mark.asyncio
    async def test_dedup_within_new_terms_batch(self):
        """同一批 ``new_terms`` 中重复出现的 word 也只写入一次。"""
        dictionary = _DictionaryFake(name="电力术语", terms=[])
        db = _DbStub(scalar_value=dictionary)
        service = FeedbackService(db)  # type: ignore[arg-type]

        await service.apply_dictionary_update(
            dictionary_id=str(dictionary.id),
            new_terms=[
                {"word": "断路器", "weight": 0.9},
                {"word": "断路器", "weight": 1.0},  # 同批重复
                {"word": "继电器", "weight": 0.8},
            ],
        )

        words = [t["word"] for t in dictionary.terms]
        assert words == ["断路器", "继电器"]

    @pytest.mark.asyncio
    async def test_returns_none_when_dictionary_missing(self):
        """目标词典不存在时返回 ``None``。"""
        db = _DbStub(scalar_value=None)
        service = FeedbackService(db)  # type: ignore[arg-type]

        result = await service.apply_dictionary_update(
            dictionary_id=str(uuid.uuid4()),
            new_terms=[{"word": "无关词"}],
        )
        assert result is None
