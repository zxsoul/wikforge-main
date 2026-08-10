"""Permission management API routes."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.core.database import get_db
from app.core.redis import get_redis
from app.models.permission import AccessLevel, ResourceType
from app.models.user import User
from app.services.permission_service import Action, PermissionService

router = APIRouter(prefix="/api/permissions", tags=["permissions"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class SetPermissionRequest(BaseModel):
    """Request body for setting permissions."""

    user_id: str
    access_level: AccessLevel


class PermissionResponse(BaseModel):
    """Response for a single permission record."""

    id: str
    resource_id: str
    resource_type: str
    user_id: str
    access_level: str

    model_config = {"from_attributes": True}


class EffectivePermissionResponse(BaseModel):
    """Response for effective permission query."""

    user_id: str
    resource_id: str
    access_level: str | None


# ─── Dependencies ──────────────────────────────────────────────────────


async def get_permission_service(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> PermissionService:
    """Dependency to get PermissionService instance."""
    return PermissionService(db=db, redis=redis)


# ─── Space Permission Endpoints ────────────────────────────────────────


@router.get("/spaces/{space_id}", response_model=list[PermissionResponse])
async def get_space_permissions(
    space_id: str,
    current_user: User = Depends(get_current_user),
    service: PermissionService = Depends(get_permission_service),
):
    """Get all permissions for a space."""
    permissions = await service.get_space_permissions(uuid.UUID(space_id))
    return [
        PermissionResponse(
            id=str(p.id),
            resource_id=str(p.resource_id),
            resource_type=p.resource_type.value,
            user_id=str(p.user_id),
            access_level=p.access_level.value,
        )
        for p in permissions
    ]


@router.put("/spaces/{space_id}", response_model=PermissionResponse)
async def set_space_permission(
    space_id: str,
    body: SetPermissionRequest,
    current_user: User = Depends(get_current_user),
    service: PermissionService = Depends(get_permission_service),
):
    """
    Set space-level permission for a user.

    Access levels:
    - invisible: space is hidden from the user
    - read: user can browse and read documents
    - write: user can browse, read, and edit documents
    """
    try:
        permission = await service.set_space_permission(
            space_id=uuid.UUID(space_id),
            user_id=uuid.UUID(body.user_id),
            access_level=body.access_level,
        )
        return PermissionResponse(
            id=str(permission.id),
            resource_id=str(permission.resource_id),
            resource_type=permission.resource_type.value,
            user_id=str(permission.user_id),
            access_level=permission.access_level.value,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Document Permission Endpoints ────────────────────────────────────


@router.get("/documents/{document_id}", response_model=list[PermissionResponse])
async def get_document_permissions(
    document_id: str,
    current_user: User = Depends(get_current_user),
    service: PermissionService = Depends(get_permission_service),
):
    """Get all permissions for a document."""
    permissions = await service.get_document_permissions(uuid.UUID(document_id))
    return [
        PermissionResponse(
            id=str(p.id),
            resource_id=str(p.resource_id),
            resource_type=p.resource_type.value,
            user_id=str(p.user_id),
            access_level=p.access_level.value,
        )
        for p in permissions
    ]


@router.put("/documents/{document_id}", response_model=PermissionResponse)
async def set_document_permission(
    document_id: str,
    body: SetPermissionRequest,
    current_user: User = Depends(get_current_user),
    service: PermissionService = Depends(get_permission_service),
):
    """
    Set document-level permission for a user.

    Document permissions override space permissions.
    Documents without explicit permissions inherit from their space.

    Access levels:
    - invisible: document is hidden from the user
    - read: user can read the document
    - write: user can read and edit the document
    """
    try:
        permission = await service.set_document_permission(
            document_id=uuid.UUID(document_id),
            user_id=uuid.UUID(body.user_id),
            access_level=body.access_level,
        )
        return PermissionResponse(
            id=str(permission.id),
            resource_id=str(permission.resource_id),
            resource_type=permission.resource_type.value,
            user_id=str(permission.user_id),
            access_level=permission.access_level.value,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Effective Permission Endpoint ─────────────────────────────────────


@router.get(
    "/users/{user_id}/effective/{document_id}",
    response_model=EffectivePermissionResponse,
)
async def get_effective_permission(
    user_id: str,
    document_id: str,
    current_user: User = Depends(get_current_user),
    service: PermissionService = Depends(get_permission_service),
):
    """
    Get the effective permission for a user on a document.
    Takes into account document-level overrides and space inheritance.
    """
    access_level = await service.get_effective_permission(
        user_id=uuid.UUID(user_id),
        document_id=uuid.UUID(document_id),
    )
    return EffectivePermissionResponse(
        user_id=user_id,
        resource_id=document_id,
        access_level=access_level.value if access_level else None,
    )


# ─── Access Check Endpoint ─────────────────────────────────────────────


@router.get("/check")
async def check_access(
    resource_id: str,
    resource_type: ResourceType,
    action: Action,
    current_user: User = Depends(get_current_user),
    service: PermissionService = Depends(get_permission_service),
):
    """
    Check if the current user has permission to perform an action on a resource.
    Returns {"allowed": true/false}.
    """
    allowed = await service.check_access(
        user_id=current_user.id,
        resource_id=uuid.UUID(resource_id),
        resource_type=resource_type,
        action=action,
    )
    return {"allowed": allowed}
