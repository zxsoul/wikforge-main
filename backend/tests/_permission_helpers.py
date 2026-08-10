"""权限服务测试共享 fixtures 与辅助函数。

故意不放在 conftest.py 中，避免污染其他模块（如认证）的 fixture 命名空间。
各权限测试文件通过 ``from tests._permission_helpers import ...`` 引入。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.permission import AccessLevel, Permission, ResourceType
from app.services.permission_service import PermissionService


@pytest.fixture
def perm_db() -> AsyncMock:
    """SQLAlchemy AsyncSession mock，行为与 conftest 的 mock_db 一致但独立命名。"""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.delete = AsyncMock()
    db.refresh = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
def perm_redis() -> AsyncMock:
    """Redis 客户端 mock。

    默认 ``get`` 返回 None、``set``/``delete`` 异步无返回值。
    ``scan_iter`` 在 redis-py 中是异步迭代器，这里通过自定义 wrapper 实现。
    """
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)

    async def _empty_scan_iter(*_args, **_kwargs):
        # 使其成为异步生成器；测试中可按需替换为产出具体 key 的实现
        if False:  # pragma: no cover
            yield ""
        return

    redis.scan_iter = _empty_scan_iter
    return redis


@pytest.fixture
def permission_service(perm_db: AsyncMock, perm_redis: AsyncMock) -> PermissionService:
    """提供注入了 mock DB / Redis 的 PermissionService 实例。"""
    return PermissionService(db=perm_db, redis=perm_redis)


def make_permission(
    resource_id: uuid.UUID,
    resource_type: ResourceType,
    user_id: uuid.UUID,
    access_level: AccessLevel,
) -> MagicMock:
    """构造一个貌似 Permission ORM 实例的 MagicMock。"""
    perm = MagicMock(spec=Permission)
    perm.id = uuid.uuid4()
    perm.resource_id = resource_id
    perm.resource_type = resource_type
    perm.user_id = user_id
    perm.access_level = access_level
    return perm


def scalar_one_or_none_result(value):
    """构造 ``await db.execute(...).scalar_one_or_none()`` 链式返回的 mock。"""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def scalars_all_result(values):
    """构造 ``(await db.execute(...)).scalars().all()`` 链式返回的 mock。"""
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = list(values)
    result.scalars.return_value = scalars
    return result


def rows_all_result(rows):
    """构造 ``(await db.execute(...)).all()`` 链式返回的 mock。"""
    result = MagicMock()
    result.all.return_value = list(rows)
    return result
