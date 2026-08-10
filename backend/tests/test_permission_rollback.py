"""权限同步失败回滚测试 (任务 4.8)。

覆盖：
- ``_rollback_permission`` 在 ``old_access_level`` 为 None 时删除新增的记录
- ``old_access_level`` 不为 None 时恢复旧值
- ``set_space_permission`` Qdrant 同步抛错 → DB 回滚 + 抛 ``RuntimeError``
- ``set_document_permission`` Qdrant 同步抛错 → DB 回滚 + 抛 ``RuntimeError``

Validates: Requirements 10
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.permission import AccessLevel, Permission, ResourceType
from tests._permission_helpers import (
    permission_service,
    perm_db,
    perm_redis,
    scalar_one_or_none_result,
)


# ─── _rollback_permission 直接行为 ─────────────────────────────────


class TestRollbackPrimitive:
    """直接调用 ``_rollback_permission`` 的两条分支。"""

    @pytest.mark.asyncio
    async def test_rollback_reverts_to_old_level(
        self, permission_service, perm_db
    ):
        """``old_access_level`` 不为 None → 恢复到旧值。"""
        existing = MagicMock(spec=Permission)
        existing.access_level = AccessLevel.write  # 当前 (失败更新后)

        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(existing)
        )

        await permission_service._rollback_permission(
            resource_id=uuid.uuid4(),
            resource_type=ResourceType.space,
            user_id=uuid.uuid4(),
            old_access_level=AccessLevel.read,
        )

        assert existing.access_level == AccessLevel.read
        perm_db.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_rollback_deletes_when_old_was_none(
        self, permission_service, perm_db
    ):
        """``old_access_level=None`` → 删除新创建的记录。"""
        existing = MagicMock(spec=Permission)
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(existing)
        )

        await permission_service._rollback_permission(
            resource_id=uuid.uuid4(),
            resource_type=ResourceType.space,
            user_id=uuid.uuid4(),
            old_access_level=None,
        )

        perm_db.delete.assert_awaited_once_with(existing)
        perm_db.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_rollback_noop_when_no_record_found(
        self, permission_service, perm_db
    ):
        """记录已不存在时回滚应安静返回。"""
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(None)
        )

        await permission_service._rollback_permission(
            resource_id=uuid.uuid4(),
            resource_type=ResourceType.space,
            user_id=uuid.uuid4(),
            old_access_level=AccessLevel.read,
        )
        perm_db.delete.assert_not_called()


# ─── set_*_permission 的回滚集成 ──────────────────────────────────


class TestSetPermissionRollback:
    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_space_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_set_space_permission_rolls_back_on_qdrant_failure(
        self, mock_sync, permission_service, perm_db, perm_redis
    ):
        """空间权限同步 Qdrant 失败 → 删除新建记录并抛 ``RuntimeError``。"""
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        # 创建分支：不存在 → upsert 新增（add）
        # 失败后 _rollback_permission 再次 _get_permission_record
        # → 应找到刚 add 的 Permission 并 delete 它
        added: dict = {}

        def _capture_add(perm):
            added["perm"] = perm

        perm_db.add.side_effect = _capture_add

        def _execute_side_effect(*_args, **_kwargs):
            # 1) 快照查询 → None
            # 2) upsert 内查询 → None（触发 add）
            # 3) rollback 查询 → 返回刚 add 的 Permission
            call = perm_db.execute.call_count
            if call <= 2:
                return scalar_one_or_none_result(None)
            return scalar_one_or_none_result(added.get("perm"))

        perm_db.execute = AsyncMock(side_effect=_execute_side_effect)
        mock_sync.side_effect = Exception("Qdrant connection timeout")

        with pytest.raises(RuntimeError, match="权限同步到 Qdrant 失败"):
            await permission_service.set_space_permission(
                space_id=space_id,
                user_id=user_id,
                access_level=AccessLevel.read,
            )

        # 回滚应删除刚刚添加的权限
        perm_db.delete.assert_awaited()
        # 没有触发 Celery 派发（应在 sync 后才派发）
        # —— 这里不能直接断言，因为 Celery 任务在 try 块外仍未执行。
        # 由 ``raise`` 中断流程已隐含验证。

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_space_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_set_space_permission_rolls_back_to_old_level(
        self, mock_sync, permission_service, perm_db, perm_redis
    ):
        """更新已有空间权限失败 → 恢复到旧 access_level。"""
        space_id = uuid.uuid4()
        user_id = uuid.uuid4()

        existing = MagicMock(spec=Permission)
        existing.access_level = AccessLevel.read  # 旧值

        # 1) 快照查询 → existing（access_level=read）
        # 2) upsert 内查询 → existing（直接修改 access_level=write）
        # 3) rollback 查询 → existing（应被恢复到 read）
        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(existing),
            ]
        )
        mock_sync.side_effect = RuntimeError("transient failure")

        with pytest.raises(RuntimeError, match="权限同步到 Qdrant 失败"):
            await permission_service.set_space_permission(
                space_id=space_id,
                user_id=user_id,
                access_level=AccessLevel.write,
            )

        # 回滚后应恢复为 read
        assert existing.access_level == AccessLevel.read

    @pytest.mark.asyncio
    @patch(
        "app.services.permission_service.PermissionService._sync_document_permissions_to_qdrant",
        new_callable=AsyncMock,
    )
    async def test_set_document_permission_rolls_back_on_qdrant_failure(
        self, mock_sync, permission_service, perm_db, perm_redis
    ):
        """文档权限同步 Qdrant 失败 → 回滚 + 抛 ``RuntimeError``。"""
        document_id = uuid.uuid4()
        user_id = uuid.uuid4()

        existing = MagicMock(spec=Permission)
        existing.access_level = AccessLevel.read

        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(existing),
                scalar_one_or_none_result(existing),
            ]
        )
        mock_sync.side_effect = Exception("Qdrant down")

        with pytest.raises(RuntimeError, match="权限同步到 Qdrant 失败"):
            await permission_service.set_document_permission(
                document_id=document_id,
                user_id=user_id,
                access_level=AccessLevel.write,
            )
        assert existing.access_level == AccessLevel.read
