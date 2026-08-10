"""文档级权限设置测试 (任务 4.3, 4.4, 4.5)。

覆盖：
- ``set_document_permission`` 创建/更新
- 文档级权限覆盖空间级（通过 effective_permission 验证语义）
- 设置后触发 Qdrant 同步与缓存失效
- API 路由 ``PUT /api/permissions/documents/{id}``

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


# ─── Service 层测试 ─────────────────────────────────────────────────


class TestSetDocumentPermissionService:
    """``PermissionService.set_document_permission`` 行为验证。"""

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_document_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_create_new_document_permission(
        self, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """创建新的文档权限：DB add → Qdrant 同步 → 缓存失效。"""
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        # 1) 快照查询 → None
        # 2) upsert 内查询 → None
        # 3) 缓存失效前 _get_document_space_id → space_id
        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(None),
                scalar_one_or_none_result(None),
                scalar_one_or_none_result(space_id),
            ]
        )

        result = await permission_service.set_document_permission(
            document_id=document_id,
            user_id=user_id,
            access_level=AccessLevel.write,
        )

        assert perm_db.add.call_count == 1
        added = perm_db.add.call_args.args[0]
        assert added.resource_type == ResourceType.document
        assert added.resource_id == document_id
        assert added.access_level == AccessLevel.write

        mock_qdrant_sync.assert_awaited_once_with(document_id)
        # 文档权限变更也会失效该用户在文档所属空间的权限缓存
        perm_redis.delete.assert_awaited_once()
        assert result is added

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_document_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_document_invisible_overrides_space_via_effective(
        self, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """文档级 invisible 应覆盖空间级 → ``get_effective_permission`` 返回 invisible。"""
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # ``get_effective_permission`` 优先查文档级权限
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(AccessLevel.invisible)
        )

        effective = await permission_service.get_effective_permission(
            user_id, document_id
        )
        assert effective == AccessLevel.invisible

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_document_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_document_no_explicit_perm_inherits_space(
        self, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """文档无显式权限时 ``get_effective_permission`` 返回空间级权限。"""
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        # 1) 文档级 → None
        # 2) 文档所属空间 → space_id
        # 3) 空间级权限 → read
        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(None),
                scalar_one_or_none_result(space_id),
                scalar_one_or_none_result(AccessLevel.read),
            ]
        )

        effective = await permission_service.get_effective_permission(
            user_id, document_id
        )
        assert effective == AccessLevel.read

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_document_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_update_existing_document_permission(
        self, mock_qdrant_sync, permission_service, perm_db, perm_redis
    ):
        """更新已存在的文档权限：仅修改 access_level，不新增行。"""
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        existing = MagicMock(spec=Permission)
        existing.access_level = AccessLevel.read

        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(space_id),
            ]
        )

        await permission_service.set_document_permission(
            document_id=document_id,
            user_id=user_id,
            access_level=AccessLevel.write,
        )

        perm_db.add.assert_not_called()
        assert existing.access_level == AccessLevel.write
        mock_qdrant_sync.assert_awaited_once_with(document_id)


# ─── API 路由层测试 ─────────────────────────────────────────────────


def _make_app(service: PermissionService) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(permissions_router)

    fake_user = MagicMock()
    fake_user.id = uuid.uuid4()
    app.dependency_overrides[get_permission_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: fake_user
    return app


class TestSetDocumentPermissionAPI:
    """``PUT /api/permissions/documents/{id}`` 端到端验证。"""

    def test_put_document_permission_returns_200(self):
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()

        service = AsyncMock(spec=PermissionService)
        returned = MagicMock(spec=Permission)
        returned.id = uuid.uuid4()
        returned.resource_id = document_id
        returned.resource_type = ResourceType.document
        returned.user_id = user_id
        returned.access_level = AccessLevel.read
        service.set_document_permission = AsyncMock(return_value=returned)

        client = TestClient(_make_app(service))
        resp = client.put(
            f"/api/permissions/documents/{document_id}",
            json={"user_id": str(user_id), "access_level": "read"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["resource_type"] == "document"
        assert body["access_level"] == "read"

    def test_get_effective_permission(self):
        """``GET /api/permissions/users/{id}/effective/{doc_id}`` 返回文档级覆盖结果。"""
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()

        service = AsyncMock(spec=PermissionService)
        service.get_effective_permission = AsyncMock(return_value=AccessLevel.write)

        client = TestClient(_make_app(service))
        resp = client.get(
            f"/api/permissions/users/{user_id}/effective/{document_id}"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["access_level"] == "write"

    def test_get_effective_permission_none_when_no_record(self):
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()

        service = AsyncMock(spec=PermissionService)
        service.get_effective_permission = AsyncMock(return_value=None)

        client = TestClient(_make_app(service))
        resp = client.get(
            f"/api/permissions/users/{user_id}/effective/{document_id}"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["access_level"] is None
