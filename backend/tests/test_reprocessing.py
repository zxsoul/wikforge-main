"""受影响文档批量重处理 API 测试（任务 17.8 / 需求 9.8）。

被测路由：

- ``POST /api/admin/feedback/reprocess``
- ``GET  /api/admin/feedback/reprocess/{task_id}``

覆盖维度：

1. **路由层契约**：
   - ``profile_id`` 入参 → 解析受影响文档并触发批量重处理；返回 ``task_id``。
   - ``dictionary_id`` 入参 → 同上，但通过 ``dictionary_id`` 解析受影响文档。
   - ``document_ids`` 入参 → 直接入队，不再调用 ``get_affected_documents``。
   - 三者均未提供 → 422 ``Must provide profile_id, dictionary_id, or document_ids``。
   - 入参齐全但解析后无任何文档 → 404 ``No documents found for reprocessing``。
   - 进度查询命中 → 200，含 ``processed_documents`` / ``progress_percent``。
   - 进度查询未命中（任务不存在或 24h TTL 过期）→ 404。
   - 鉴权：未登录返回 401；登录但非管理员返回 403，且不调用业务服务。
2. **服务层副作用**：
   - :meth:`FeedbackService.trigger_reprocessing` 对每篇文档调用一次
     ``celery_app.send_task("app.tasks.reprocess_document", ...)``，
     队列固定为 ``reprocessing``。
   - 同步写入 Redis hash ``reprocess:task:{task_id}``，含 ``total`` / ``status``
     / ``created_at``，并设置 86400 秒（24h）TTL。

策略：
- 使用 ``app.dependency_overrides`` 替换鉴权依赖，避免真实 JWT 流程。
- 通过 ``monkeypatch`` 把模块级 ``FeedbackService`` 替换为代理对象，
  转发到一个 ``AsyncMock`` 服务替身，便于断言调用参数。
- 服务层副作用直接调用真实 :class:`FeedbackService`（搭配 AsyncMock 的
  Celery / Redis 客户端），确保对 Celery 任务名 / 队列 / Redis key /
  TTL 有回归保护。

Validates: Requirements 9.8
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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
    """轻量替身：把 ``FeedbackService(db)`` 调用转发到注入的 AsyncMock。"""

    def __init__(self, mock: AsyncMock):
        self._mock = mock

    async def get_affected_documents(self, **kwargs):
        return await self._mock.get_affected_documents(**kwargs)

    async def trigger_reprocessing(self, document_ids):
        return await self._mock.trigger_reprocessing(document_ids)

    async def get_reprocessing_progress(self, task_id):
        return await self._mock.get_reprocessing_progress(task_id)


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


def _fake_document() -> MagicMock:
    doc = MagicMock()
    doc.id = uuid.uuid4()
    return doc


def _fake_task(
    task_id: str = "task-123",
    *,
    total: int = 5,
    processed: int = 0,
    status: str = "running",
    error: str | None = None,
) -> ReprocessingTask:
    return ReprocessingTask(
        task_id=task_id,
        total_documents=total,
        processed_documents=processed,
        status=status,
        created_at=datetime.now(timezone.utc),
        error=error,
    )


# ─── 1. POST /api/admin/feedback/reprocess ─────────────────────────


class TestTriggerReprocessing:
    """``POST /api/admin/feedback/reprocess`` 行为契约。"""

    def test_with_profile_id_resolves_and_triggers(self, monkeypatch):
        """提供 ``profile_id`` → 解析受影响文档并触发批量重处理。"""
        profile_id = str(uuid.uuid4())
        documents = [_fake_document() for _ in range(3)]
        task = _fake_task(task_id="task-prof-1", total=3)

        service = AsyncMock()
        service.get_affected_documents = AsyncMock(return_value=documents)
        service.trigger_reprocessing = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"profile_id": profile_id},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-prof-1"
        assert body["total_documents"] == 3
        assert body["processed_documents"] == 0
        assert body["status"] == "running"
        # 进度百分比基于 processed/total 计算，初始为 0.0
        assert body["progress_percent"] == 0.0
        assert body["error"] is None
        # 解析路径：先按 profile_id 查询，再触发
        service.get_affected_documents.assert_awaited_once_with(
            profile_id=profile_id, dictionary_id=None
        )
        triggered_doc_ids = service.trigger_reprocessing.await_args.args[0]
        assert triggered_doc_ids == [str(d.id) for d in documents]

    def test_with_dictionary_id_resolves_and_triggers(self, monkeypatch):
        """提供 ``dictionary_id`` → 同样解析受影响文档并触发。"""
        dictionary_id = str(uuid.uuid4())
        documents = [_fake_document() for _ in range(2)]
        task = _fake_task(task_id="task-dict-9", total=2)

        service = AsyncMock()
        service.get_affected_documents = AsyncMock(return_value=documents)
        service.trigger_reprocessing = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"dictionary_id": dictionary_id},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-dict-9"
        assert body["total_documents"] == 2
        service.get_affected_documents.assert_awaited_once_with(
            profile_id=None, dictionary_id=dictionary_id
        )
        triggered_doc_ids = service.trigger_reprocessing.await_args.args[0]
        assert triggered_doc_ids == [str(d.id) for d in documents]

    def test_with_explicit_document_ids_skips_resolution(self, monkeypatch):
        """提供 ``document_ids`` → 直接入队，不再调用 ``get_affected_documents``。"""
        doc_ids = [str(uuid.uuid4()) for _ in range(4)]
        task = _fake_task(task_id="task-explicit", total=4)

        service = AsyncMock()
        service.get_affected_documents = AsyncMock()
        service.trigger_reprocessing = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"document_ids": doc_ids},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-explicit"
        assert body["total_documents"] == 4
        # 显式入参时不应再去解析受影响文档
        service.get_affected_documents.assert_not_awaited()
        # 文档 ID 列表透传，不重新生成
        triggered_doc_ids = service.trigger_reprocessing.await_args.args[0]
        assert triggered_doc_ids == doc_ids

    def test_progress_percent_reflects_processed_ratio(self, monkeypatch):
        """``progress_percent`` 必须等于 ``processed / total * 100``。"""
        task = _fake_task(
            task_id="task-progress",
            total=4,
            processed=1,
            status="running",
        )
        service = AsyncMock()
        service.trigger_reprocessing = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"document_ids": ["a", "b", "c", "d"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["processed_documents"] == 1
        assert body["progress_percent"] == 25.0

    def test_missing_all_inputs_returns_422(self, monkeypatch):
        """三者均未提供 → 422。"""
        service = AsyncMock()
        service.get_affected_documents = AsyncMock()
        service.trigger_reprocessing = AsyncMock()

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post("/api/admin/feedback/reprocess", json={})

        assert resp.status_code == 422
        # 422 不应触达后续业务调用
        service.get_affected_documents.assert_not_awaited()
        service.trigger_reprocessing.assert_not_awaited()

    def test_no_affected_documents_returns_404(self, monkeypatch):
        """``profile_id`` 解析后没有匹配文档 → 404。"""
        service = AsyncMock()
        service.get_affected_documents = AsyncMock(return_value=[])
        service.trigger_reprocessing = AsyncMock()

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"profile_id": str(uuid.uuid4())},
        )

        assert resp.status_code == 404
        # 404 不应触发后续 trigger
        service.trigger_reprocessing.assert_not_awaited()

    def test_explicit_empty_document_ids_falls_through_to_422(self, monkeypatch):
        """``document_ids=[]`` 视为「未提供」，命中 422 兜底分支。

        ``request.document_ids`` 为空列表时 ``if request.document_ids`` 为假，
        若同时也未给出 ``profile_id`` / ``dictionary_id``，路由层应明确返回
        422 ``Must provide ...``，而不是让任务以 0 篇文档继续往下走。
        """
        service = AsyncMock()
        service.trigger_reprocessing = AsyncMock()
        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"document_ids": []},
        )

        assert resp.status_code == 422
        service.trigger_reprocessing.assert_not_awaited()


# ─── 2. GET /api/admin/feedback/reprocess/{task_id} ────────────────


class TestGetReprocessingProgress:
    """``GET /api/admin/feedback/reprocess/{task_id}`` 行为契约。"""

    def test_returns_progress_when_task_exists(self, monkeypatch):
        """命中：返回任务进度 + 计算后的百分比。"""
        task = _fake_task(
            task_id="task-running",
            total=10,
            processed=3,
            status="running",
        )
        service = AsyncMock()
        service.get_reprocessing_progress = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/reprocess/task-running")

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-running"
        assert body["total_documents"] == 10
        assert body["processed_documents"] == 3
        assert body["status"] == "running"
        assert body["progress_percent"] == 30.0
        service.get_reprocessing_progress.assert_awaited_once_with("task-running")

    def test_returns_404_when_task_missing(self, monkeypatch):
        """未命中（不存在或 24h TTL 过期）→ 404。"""
        service = AsyncMock()
        service.get_reprocessing_progress = AsyncMock(return_value=None)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/reprocess/missing-id")

        assert resp.status_code == 404

    def test_progress_percent_zero_when_total_is_zero(self, monkeypatch):
        """``total_documents == 0`` 时百分比稳健回落到 0.0，不抛除零异常。"""
        task = _fake_task(
            task_id="task-empty",
            total=0,
            processed=0,
            status="completed",
        )
        service = AsyncMock()
        service.get_reprocessing_progress = AsyncMock(return_value=task)

        app = _build_app(service_mock=service, monkeypatch=monkeypatch)
        client = TestClient(app)

        resp = client.get("/api/admin/feedback/reprocess/task-empty")

        assert resp.status_code == 200
        body = resp.json()
        assert body["progress_percent"] == 0.0


# ─── 3. 鉴权：未登录 / 非管理员 ─────────────────────────────────────


class TestReprocessingAuthorization:
    """``reprocess`` 与进度查询路由必须通过 ``require_admin``。"""

    def test_trigger_unauthenticated_returns_401(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"profile_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 401
        # 鉴权失败不应触达业务服务
        service.get_affected_documents.assert_not_awaited()
        service.trigger_reprocessing.assert_not_awaited()

    def test_trigger_non_admin_returns_403(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.post(
            "/api/admin/feedback/reprocess",
            json={"profile_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 403
        service.get_affected_documents.assert_not_awaited()
        service.trigger_reprocessing.assert_not_awaited()

    def test_progress_unauthenticated_returns_401(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=UnauthorizedException("缺少认证令牌"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/reprocess/some-task-id")
        assert resp.status_code == 401
        service.get_reprocessing_progress.assert_not_awaited()

    def test_progress_non_admin_returns_403(self, monkeypatch):
        service = AsyncMock()
        app = _build_app(
            service_mock=service,
            monkeypatch=monkeypatch,
            auth_error=ForbiddenException("需要管理员权限"),
        )
        client = TestClient(app)
        resp = client.get("/api/admin/feedback/reprocess/some-task-id")
        assert resp.status_code == 403
        service.get_reprocessing_progress.assert_not_awaited()


# ─── 4. 服务层副作用：Celery 任务名 / 队列 + Redis 写入 / TTL ────────


class TestTriggerReprocessingServiceEffects:
    """:meth:`FeedbackService.trigger_reprocessing` 的副作用回归保护。

    需求 9.8：「记录任务进度可查」+「每个文档独立提交，不阻塞其它文档」。

    通过 patch 模块 ``app.core.celery_app`` 与 ``app.core.redis``，断言：

    - 每个 ``document_id`` 都会调用一次 ``celery_app.send_task``，任务名固定
      为 ``app.tasks.reprocess_document``，队列固定为 ``reprocessing``。
    - 提交完成后，Redis hash ``reprocess:task:{id}`` 被写入 ``total`` /
      ``status`` / ``created_at`` 等字段，并设置 86400 秒（24h）TTL。
    - Celery 不可用时回落到 ``status=='failed'`` 但 Redis 仍记录失败原因，
      以便前端能查询到失败任务而不是「无任何痕迹」。
    """

    @pytest.mark.asyncio
    async def test_send_task_per_document_with_redis_progress(self):
        """正常路径：每篇文档一条 send_task；Redis 写入 24h TTL。"""
        document_ids = [str(uuid.uuid4()) for _ in range(3)]

        fake_celery = MagicMock()
        fake_redis = AsyncMock()
        fake_redis.hset = AsyncMock()
        fake_redis.expire = AsyncMock()

        with (
            patch(
                "app.core.celery_app.celery_app",
                fake_celery,
            ),
            patch(
                "app.core.redis.get_redis",
                AsyncMock(return_value=fake_redis),
            ),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.trigger_reprocessing(document_ids)

        # send_task 被调用 3 次，每次入参均为正确的 (任务名, args, queue)
        assert fake_celery.send_task.call_count == 3
        for call, doc_id in zip(
            fake_celery.send_task.call_args_list, document_ids, strict=True
        ):
            assert call.args[0] == "app.tasks.reprocess_document"
            assert call.kwargs["args"] == [doc_id, task.task_id]
            assert call.kwargs["queue"] == "reprocessing"

        # 任务对象状态被推进为 running，total_documents 等于入参数量
        assert task.status == "running"
        assert task.total_documents == 3
        assert task.processed_documents == 0
        assert task.error is None

        # Redis hash 写入：key 形如 ``reprocess:task:{uuid}``，含必要字段
        fake_redis.hset.assert_awaited_once()
        hset_args = fake_redis.hset.await_args
        assert hset_args.args[0] == f"reprocess:task:{task.task_id}"
        mapping = hset_args.kwargs["mapping"]
        assert mapping["total"] == "3"
        assert mapping["processed"] == "0"
        assert mapping["status"] == "running"
        # ISO 时间字符串能成功解析回 datetime
        assert datetime.fromisoformat(mapping["created_at"]) is not None
        assert mapping["error"] == ""

        # TTL = 86400 秒（24h），与需求 9.8「进度可查」相吻合
        fake_redis.expire.assert_awaited_once_with(
            f"reprocess:task:{task.task_id}", 86400
        )

    @pytest.mark.asyncio
    async def test_celery_send_failure_marks_task_as_failed(self):
        """Celery 提交抛异常时：任务状态 ``failed``、错误原因仍写入 Redis。"""
        fake_celery = MagicMock()
        fake_celery.send_task.side_effect = RuntimeError("broker unavailable")

        fake_redis = AsyncMock()
        fake_redis.hset = AsyncMock()
        fake_redis.expire = AsyncMock()

        with (
            patch(
                "app.core.celery_app.celery_app",
                fake_celery,
            ),
            patch(
                "app.core.redis.get_redis",
                AsyncMock(return_value=fake_redis),
            ),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.trigger_reprocessing([str(uuid.uuid4())])

        assert task.status == "failed"
        assert task.error == "broker unavailable"
        # 即使 Celery 不可用，Redis 仍记录任务，便于前端查询失败状态
        fake_redis.hset.assert_awaited_once()
        mapping = fake_redis.hset.await_args.kwargs["mapping"]
        assert mapping["status"] == "failed"
        assert mapping["error"] == "broker unavailable"


class TestGetReprocessingProgressServiceEffects:
    """:meth:`FeedbackService.get_reprocessing_progress` 服务层契约。"""

    @pytest.mark.asyncio
    async def test_returns_task_when_redis_hit(self):
        """Redis 命中：完整回填 ``ReprocessingTask`` 字段。"""
        created_at = datetime.now(timezone.utc)
        fake_redis = AsyncMock()
        fake_redis.hgetall = AsyncMock(
            return_value={
                "total": "10",
                "processed": "4",
                "status": "running",
                "created_at": created_at.isoformat(),
                "error": "",
            }
        )

        with patch(
            "app.core.redis.get_redis",
            AsyncMock(return_value=fake_redis),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.get_reprocessing_progress("task-xyz")

        assert task is not None
        assert task.task_id == "task-xyz"
        assert task.total_documents == 10
        assert task.processed_documents == 4
        assert task.status == "running"
        assert task.error is None
        # Redis key 形态：``reprocess:task:{id}``
        fake_redis.hgetall.assert_awaited_once_with("reprocess:task:task-xyz")

    @pytest.mark.asyncio
    async def test_returns_none_when_redis_miss(self):
        """Redis 未命中（key 不存在或 TTL 过期）→ ``None``。"""
        fake_redis = AsyncMock()
        fake_redis.hgetall = AsyncMock(return_value={})

        with patch(
            "app.core.redis.get_redis",
            AsyncMock(return_value=fake_redis),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.get_reprocessing_progress("missing-id")

        assert task is None
