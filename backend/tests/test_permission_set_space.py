"""空间级权限设置测试 (任务 4.2, 4.5, 4.6)。

覆盖：
- ``set_space_permission`` 创建新权限
- 已存在记录时基于 (resource_id, resource_type, user_id) 唯一约束更新而非新增
- 设置成功后触发 Qdrant 同步与缓存失效
- 异步 Celery 任务 ``sync_space_permissions_async`` 被派发
- API 路由 ``PUT /api/permissions/spaces/{id}`` 端到端验证（TestClient）

Validates: Requirements 10
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.permissions import get_permission_service, router as permissions_router
from app.api.auth import get_current_user
from app.core.exceptions import register_exception_handlers
from app.models.permission import AccessLevel, Permission, ResourceType
from app.services.permission_service import PermissionService
from tests._permission_helpers import (
    permission_service,
    perm_db,
    perm_redis,
    scalar_one_or_none_result,
)


# Celery 未安装时，``set_space_permission`` 内会延迟 import ``app.tasks.permission_tasks``
# 触发 ``ModuleNotFoundError``。Service 层测试在该情况下整体跳过；
# API 层测试使用纯 mock service，不受影响。
_celery_available = True
try:  # pragma: no cover - 仅运行环境差异
    import celery  # noqa: F401
except Exception:  # pragma: no cover
    _celery_available = False

requires_celery = pytest.mark.skipif(
    not _celery_available, reason="celery 未安装，跳过 Service 层 Celery 派发测试"
)


# ─── Service 层测试 ─────────────────────────────────────────────────


@requires_celery
class TestSetSpacePermissionService:
    """``PermissionService.set_space_permission`` 行为验证。"""

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_space_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    @patch(
        "app.tasks.permission_tasks.sync_space_permissions_async.delay"
    )
    async def test_create_new_permission(
        self, mock_celery, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """无现有记录时应创建新的 Permission，并触发同步与缓存失效。"""
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # 1) _get_permission_record（rollback 前快照）→ None
        # 2) _upsert_permission 内部又一次 _get_permission_record → None
        # 之后无 DB 查询
        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(None),
                scalar_one_or_none_result(None),
            ]
        )

        result = await permission_service.set_space_permission(
            space_id=space_id,
            user_id=user_id,
            access_level=AccessLevel.read,
        )

        # 新增记录：db.add 被调用一次
        assert perm_db.add.call_count == 1
        added = perm_db.add.call_args.args[0]
        assert isinstance(added, Permission)
        assert added.resource_id == space_id
        assert added.resource_type == ResourceType.space
        assert added.user_id == user_id
        assert added.access_level == AccessLevel.read

        # Qdrant 同步被调用、缓存失效、Celery 异步任务被派发
        mock_qdrant_sync.assert_awaited_once_with(space_id)
        perm_redis.delete.assert_awaited_once()
        mock_celery.assert_called_once_with(str(space_id))

        assert result is added

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_space_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    @patch(
        "app.tasks.permission_tasks.sync_space_permissions_async.delay"
    )
    async def test_update_existing_permission(
        self, mock_celery, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """已存在 (resource_id, resource_type, user_id) 三元组时应更新 access_level，不新增行。"""
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        existing = MagicMock(spec=Permission)
        existing.access_level = AccessLevel.read

        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(existing),
            ]
        )

        result = await permission_service.set_space_permission(
            space_id=space_id,
            user_id=user_id,
            access_level=AccessLevel.write,
        )

        # 没有新增（add 不被调用），仅更新 access_level
        perm_db.add.assert_not_called()
        assert existing.access_level == AccessLevel.write
        assert result is existing

        # 同步与异步任务仍然触发
        mock_qdrant_sync.assert_awaited_once_with(space_id)
        mock_celery.assert_called_once_with(str(space_id))

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_space_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    @patch(
        "app.tasks.permission_tasks.sync_space_permissions_async.delay"
    )
    async def test_invisible_level_accepted(
        self, mock_celery, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """invisible 也是合法等级，应被正常持久化。"""
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(None),
                scalar_one_or_none_result(None),
            ]
        )

        await permission_service.set_space_permission(
            space_id=space_id,
            user_id=user_id,
            access_level=AccessLevel.invisible,
        )
        added = perm_db.add.call_args.args[0]
        assert added.access_level == AccessLevel.invisible
        mock_celery.assert_called_once()


# ─── API 路由层测试 ─────────────────────────────────────────────────


def _make_app(service: PermissionService) -> FastAPI:
    """构造仅包含 permissions 路由的迷你 FastAPI app，依赖注入 mock 服务。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(permissions_router)

    fake_user = MagicMock()
    fake_user.id = uuid.uuid4()
    app.dependency_overrides[get_permission_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: fake_user
    return app


class TestSetSpacePermissionAPI:
    """``PUT /api/permissions/spaces/{id}`` 端到端验证。"""

    def test_put_returns_200_with_permission(self):
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # 构造一个 mock service（不走真实 DB）
        service = AsyncMock(spec=PermissionService)
        returned_perm = MagicMock(spec=Permission)
        returned_perm.id = uuid.uuid4()
        returned_perm.resource_id = space_id
        returned_perm.resource_type = ResourceType.space
        returned_perm.user_id = user_id
        returned_perm.access_level = AccessLevel.write
        service.set_space_permission = AsyncMock(return_value=returned_perm)

        client = TestClient(_make_app(service))
        resp = client.put(
            f"/api/permissions/spaces/{space_id}",
            json={"user_id": str(user_id), "access_level": "write"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["resource_id"] == str(space_id)
        assert body["user_id"] == str(user_id)
        assert body["access_level"] == "write"
        assert body["resource_type"] == "space"

        service.set_space_permission.assert_awaited_once()
        kwargs = service.set_space_permission.await_args.kwargs
        assert kwargs["space_id"] == space_id
        assert kwargs["user_id"] == user_id
        assert kwargs["access_level"] == AccessLevel.write

    def test_put_with_invalid_access_level_returns_422(self):
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()
        service = AsyncMock(spec=PermissionService)

        client = TestClient(_make_app(service))
        resp = client.put(
            f"/api/permissions/spaces/{space_id}",
            json={"user_id": str(user_id), "access_level": "owner"},
        )
        assert resp.status_code == 422

    def test_put_qdrant_failure_returns_500(self):
        """Service 抛 RuntimeError 时路由应返回 500（同步失败回滚后的语义）。"""
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        service = AsyncMock(spec=PermissionService)
        service.set_space_permission = AsyncMock(
            side_effect=RuntimeError("权限同步到 Qdrant 失败")
        )

        client = TestClient(_make_app(service))
        resp = client.put(
            f"/api/permissions/spaces/{space_id}",
            json={"user_id": str(user_id), "access_level": "read"},
        )
        assert resp.status_code == 500
        # 全局异常处理器把 detail 放到 ``error.message``
        body = resp.json()
        message = body.get("detail") or body.get("error", {}).get("message", "")
        assert "Qdrant" in message
