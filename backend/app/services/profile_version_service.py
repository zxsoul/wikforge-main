"""Shared helpers for ``DocumentProfile`` 版本快照与变更人查询。

任务 10.6 把 ``_create_version_snapshot`` / ``_get_admin_user_id`` 从
``app.api.admin_profiles`` 抽到这里，让候选 Profile 服务层（同任务）以及现有
管理路由都能复用，且不引入循环依赖。

注意：
- 这里不直接 ``await db.commit()`` —— 调用方在请求结束时由 ``get_db`` 统一
  提交，本模块只负责把 ORM 实例加入 session。
- ``get_admin_user_id`` 暂时返回数据库中的第一个用户作为占位（与原实现一致）。
  后续接入鉴权后可以替换。
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_profile import DocumentProfile
from app.models.profile_version import ProfileVersion


async def create_version_snapshot(
    db: AsyncSession,
    profile: DocumentProfile,
    changed_by: uuid.UUID,
    change_note: str | None = None,
) -> ProfileVersion:
    """为当前 Profile 状态写入一条 ``ProfileVersion`` 快照。

    Args:
        db: SQLAlchemy 异步 Session。
        profile: 已经更新到目标状态、并且 ``profile.version`` 也已自增的 ORM 实例。
        changed_by: 操作者用户 ID。
        change_note: 可选的变更说明。

    Returns:
        新建的 ``ProfileVersion``（已加入 session，未 commit）。
    """
    snapshot = {
        "name": profile.name,
        "description": profile.description,
        "priority": profile.priority,
        "enabled": profile.enabled,
        "match_rules": profile.match_rules,
        "heading_rules": profile.heading_rules,
        "boilerplate": profile.boilerplate,
        "tables": profile.tables,
        "chunking": profile.chunking,
        "domain_dictionary_id": (
            str(profile.domain_dictionary_id) if profile.domain_dictionary_id else None
        ),
    }

    version_entry = ProfileVersion(
        profile_id=profile.id,
        version=profile.version,
        snapshot=snapshot,
        changed_by=changed_by,
        change_note=change_note,
    )
    db.add(version_entry)
    return version_entry


async def get_admin_user_id(db: AsyncSession) -> uuid.UUID | None:
    """返回用于版本追踪的管理员用户 ID。

    生产环境接入鉴权后，应改为从请求上下文里取 ``current_user``。当前 fallback
    为数据库中的第一个 User —— 在初始化脚本里这就是预置管理员。
    """
    from app.models.user import User  # 延迟导入，避免在迁移脚本中触发循环

    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()
    return user.id if user else None
