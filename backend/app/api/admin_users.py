"""管理员用户管理 API。

设计说明:
- 当前采用「单一管理员邮箱」模型 (require_admin 按 INITIAL_ADMIN_EMAIL 匹配),
  没有完整的 RBAC role 字段。本路由提供:
  * 列表 / 搜索 / 分页 (GET /api/admin/users)
  * 切换启用状态 (PUT /api/admin/users/{id}/active) — 通过设置 locked_until 实现
  * 角色变更 (PUT /api/admin/users/{id}/role) — 当前模型不支持,返回 409 提示

后续如果引入完整 RBAC, 可以替换实现而不动路由签名。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import is_admin_user, require_admin
from app.core.database import get_db
from app.models.user import User

router = APIRouter(prefix="/api/admin/users", tags=["admin", "users"])


# ─── Schemas ────────────────────────────────────────────────────────────


class UserListItem(BaseModel):
    """用户列表中的单条记录。"""

    id: str
    email: str
    display_name: str = ""
    is_admin: bool
    is_active: bool
    oidc_provider: str | None = None
    failed_login_count: int = 0
    locked_until: datetime | None = None
    created_at: datetime
    updated_at: datetime


class UserListResponse(BaseModel):
    """分页响应。"""

    items: list[UserListItem]
    total: int
    page: int
    page_size: int


class SetActiveRequest(BaseModel):
    """启用 / 禁用用户。"""

    is_active: bool


class SetRoleRequest(BaseModel):
    """修改用户角色 (当前模型仅占位, 不支持真实变更)。"""

    role: str = Field(..., description="目标角色, 当前只接受 'user' 或 'admin'")


# ─── Helpers ────────────────────────────────────────────────────────────


def _to_list_item(user: User) -> UserListItem:
    """User ORM -> Pydantic 列表项。"""
    now = datetime.now(timezone.utc)
    locked = user.locked_until
    is_active = locked is None or locked <= now
    return UserListItem(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name or "",
        is_admin=is_admin_user(user),
        is_active=is_active,
        oidc_provider=user.oidc_provider,
        failed_login_count=user.failed_login_count,
        locked_until=locked,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


# ─── Endpoints ──────────────────────────────────────────────────────────


@router.get("", response_model=UserListResponse)
async def list_users(
    keyword: str | None = Query(
        None,
        description="按邮箱或显示名模糊搜索,大小写不敏感",
        max_length=100,
    ),
    page: int = Query(1, ge=1, description="页码 (从 1 开始)"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数,上限 100"),
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserListResponse:
    """列出系统用户,支持邮箱 / 显示名模糊搜索。"""
    base_stmt = select(User)
    count_stmt = select(func.count()).select_from(User)

    if keyword:
        like_pattern = f"%{keyword.lower()}%"
        condition = or_(
            func.lower(User.email).like(like_pattern),
            func.lower(func.coalesce(User.display_name, "")).like(like_pattern),
        )
        base_stmt = base_stmt.where(condition)
        count_stmt = count_stmt.where(condition)

    total_result = await db.execute(count_stmt)
    total = int(total_result.scalar() or 0)

    offset = (page - 1) * page_size
    page_stmt = (
        base_stmt.order_by(User.created_at.desc()).offset(offset).limit(page_size)
    )
    rows = (await db.execute(page_stmt)).scalars().all()

    return UserListResponse(
        items=[_to_list_item(u) for u in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.put("/{user_id}/active", response_model=UserListItem)
async def set_user_active(
    user_id: str,
    body: SetActiveRequest,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserListItem:
    """启用 / 禁用用户。

    当前实现复用 ``locked_until``: 禁用 = 锁定 100 年, 启用 = 解锁。
    """
    try:
        target_uuid = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="user_id 不是合法 UUID") from exc

    user = (
        await db.execute(select(User).where(User.id == target_uuid))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 不允许禁用 admin 自身, 防止自锁
    if not body.is_active and is_admin_user(user):
        raise HTTPException(
            status_code=409, detail="不能禁用管理员账号"
        )

    if body.is_active:
        user.locked_until = None
        user.failed_login_count = 0
    else:
        user.locked_until = datetime.now(timezone.utc) + timedelta(days=365 * 100)

    await db.flush()
    await db.refresh(user)
    return _to_list_item(user)


@router.put("/{user_id}/role", status_code=status.HTTP_409_CONFLICT)
async def set_user_role(
    user_id: str,
    body: SetRoleRequest,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """修改用户角色 (当前模型不支持)。

    系统当前采用「单一管理员邮箱」模型 (INITIAL_ADMIN_EMAIL),
    没有真正的 role 字段。前端为了未来扩展会调用这个端点,
    后端统一返回 409 + 提示。
    """
    raise HTTPException(
        status_code=409,
        detail=(
            "当前部署未启用 RBAC: 管理员身份由 INITIAL_ADMIN_EMAIL 决定, "
            "不支持运行时变更角色。如需扩展请引入完整 RBAC。"
        ),
    )


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除用户 (硬删除)。

    禁止删除当前管理员邮箱以防自锁。
    Permission / ChatSession / 等关联记录通过 FK ON DELETE CASCADE 同步清理。
    """
    try:
        target_uuid = uuid.UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="user_id 不是合法 UUID") from exc

    user = (
        await db.execute(select(User).where(User.id == target_uuid))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if is_admin_user(user):
        raise HTTPException(status_code=409, detail="不能删除管理员账号")

    await db.delete(user)
    await db.flush()
    return None
