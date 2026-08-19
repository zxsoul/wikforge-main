"""管理员按用户分配空间级权限 API（新增文件，不改动既有代码）。

设计说明:
- 只覆盖「空间级别」权限分配, 文档级权限仍走既有 /api/permissions 端点。
- 写路径完全复用 :class:`PermissionService.set_space_permission`,
  因此每次变更自带: DB upsert -> Qdrant payload 同步(失败回滚)
  -> Redis 权限缓存失效 -> Celery 异步全量同步空间内文档,
  RAG / 搜索的 Pre-Filtering 立即能感知权限变化。
- 权限模型沿用既有 ABAC: access_level ∈ {invisible, read, write},
  无记录等价于 invisible (默认不可见)。
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import require_admin
from app.core.database import get_db
from app.core.redis import get_redis
from app.models.permission import AccessLevel, Permission, ResourceType
from app.models.space import Space
from app.models.user import User
from app.services.permission_service import PermissionService

router = APIRouter(
    prefix="/api/admin/users", tags=["admin", "users", "space-permissions"]
)


# ─── Schemas ────────────────────────────────────────────────────────────


class UserSpacePermissionItem(BaseModel):
    """单个空间及其对目标用户的当前权限。"""

    space_id: str
    space_name: str
    description: str | None = None
    # None 表示无权限记录 (等价于 invisible / 默认不可见)
    access_level: str | None = None


class UserSpacePermissionListResponse(BaseModel):
    """目标用户在全部空间上的权限视图。"""

    user_id: str
    items: list[UserSpacePermissionItem]


class SetUserSpacePermissionItem(BaseModel):
    """批量设置中的单项。"""

    space_id: str
    access_level: AccessLevel


class SetUserSpacePermissionsRequest(BaseModel):
    """批量设置目标用户在多个空间上的权限。"""

    items: list[SetUserSpacePermissionItem]


class SetUserSpacePermissionResult(BaseModel):
    """单项设置结果。"""

    space_id: str
    access_level: str
    status: str  # "ok" | "failed"
    detail: str | None = None


class SetUserSpacePermissionsResponse(BaseModel):
    """批量设置响应, 逐项汇报成功/失败。"""

    user_id: str
    results: list[SetUserSpacePermissionResult]


# ─── Helpers ────────────────────────────────────────────────────────────


async def _get_target_user(user_id: str, db: AsyncSession) -> User:
    """解析并校验目标用户存在。"""
    try:
        target_uuid = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="user_id 不是合法 UUID") from exc

    user = (
        await db.execute(select(User).where(User.id == target_uuid))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# ─── Endpoints ──────────────────────────────────────────────────────────


@router.get(
    "/{user_id}/space-permissions",
    response_model=UserSpacePermissionListResponse,
)
async def list_user_space_permissions(
    user_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserSpacePermissionListResponse:
    """列出全部空间以及目标用户在每个空间上的当前权限。

    无权限记录的空间 access_level 为 null (语义等同 invisible)。
    """
    user = await _get_target_user(user_id, db)

    spaces = (await db.execute(select(Space).order_by(Space.created_at))).scalars().all()

    perm_stmt = select(Permission.resource_id, Permission.access_level).where(
        Permission.user_id == user.id,
        Permission.resource_type == ResourceType.space,
    )
    perm_rows = (await db.execute(perm_stmt)).all()
    perm_map = {row[0]: row[1].value for row in perm_rows}

    return UserSpacePermissionListResponse(
        user_id=str(user.id),
        items=[
            UserSpacePermissionItem(
                space_id=str(space.id),
                space_name=space.name,
                description=space.description,
                access_level=perm_map.get(space.id),
            )
            for space in spaces
        ],
    )


@router.put(
    "/{user_id}/space-permissions",
    response_model=SetUserSpacePermissionsResponse,
)
async def set_user_space_permissions(
    user_id: str,
    body: SetUserSpacePermissionsRequest,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> SetUserSpacePermissionsResponse:
    """批量设置目标用户在多个空间上的权限。

    每项独立调用 PermissionService.set_space_permission:
    - 成功: DB 已更新, Qdrant / Redis / Celery 同步链路已触发;
    - 失败: 该项已回滚, 不影响其他项, 结果中标记 failed。
    """
    user = await _get_target_user(user_id, db)
    service = PermissionService(db=db, redis=redis)

    results: list[SetUserSpacePermissionResult] = []
    for item in body.items:
        try:
            space_uuid = uuid.UUID(item.space_id)
        except ValueError:
            results.append(
                SetUserSpacePermissionResult(
                    space_id=item.space_id,
                    access_level=item.access_level.value,
                    status="failed",
                    detail="space_id 不是合法 UUID",
                )
            )
            continue

        try:
            await service.set_space_permission(
                space_id=space_uuid,
                user_id=user.id,
                access_level=item.access_level,
            )
            results.append(
                SetUserSpacePermissionResult(
                    space_id=item.space_id,
                    access_level=item.access_level.value,
                    status="ok",
                )
            )
        except RuntimeError as exc:
            # Qdrant 同步失败已回滚, 记录失败继续处理其余项
            results.append(
                SetUserSpacePermissionResult(
                    space_id=item.space_id,
                    access_level=item.access_level.value,
                    status="failed",
                    detail=str(exc),
                )
            )

    return SetUserSpacePermissionsResponse(user_id=str(user.id), results=results)
