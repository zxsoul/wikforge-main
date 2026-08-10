"""Permission service implementing ABAC access control model.

Provides:
- check_access: ABAC permission check (<50ms with Redis cache)
- Space-level and document-level permission management
- Permission inheritance (document inherits from space unless overridden)
- Qdrant payload sync for Pre-Filtering
- Redis caching with TTL 5 minutes and active invalidation
- Rollback mechanism on Qdrant sync failure
"""

import logging
import uuid
from enum import Enum

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.permission import AccessLevel, Permission, ResourceType

logger = logging.getLogger(__name__)

# Redis cache key pattern and TTL
PERM_CACHE_KEY_PATTERN = "perm:user:{user_id}:space:{space_id}"
PERM_CACHE_TTL = 300  # 5 minutes


class Action(str, Enum):
    """Actions that can be performed on resources."""

    browse = "browse"
    read = "read"
    write = "write"


# Mapping of access levels to allowed actions
ACCESS_LEVEL_ACTIONS: dict[AccessLevel, set[str]] = {
    AccessLevel.invisible: set(),
    AccessLevel.read: {Action.browse, Action.read},
    AccessLevel.write: {Action.browse, Action.read, Action.write},
}


class PermissionService:
    """ABAC-based permission service with caching and Qdrant sync."""

    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

    # ─── Core ABAC Check ───────────────────────────────────────────────

    async def check_access(
        self,
        user_id: uuid.UUID,
        resource_id: uuid.UUID,
        resource_type: ResourceType,
        action: Action,
    ) -> bool:
        """
        ABAC permission check. Target: <50ms response via Redis cache.

        Logic:
        1. For documents: check document-level permission first, fall back to space.
        2. For spaces: check space-level permission directly.
        """
        if resource_type == ResourceType.document:
            return await self._check_document_access(user_id, resource_id, action)
        else:
            return await self._check_space_access(user_id, resource_id, action)

    async def _check_space_access(
        self, user_id: uuid.UUID, space_id: uuid.UUID, action: Action
    ) -> bool:
        """Check space-level access with Redis cache."""
        access_level = await self._get_space_permission_cached(user_id, space_id)
        if access_level is None:
            # No permission set means invisible (default)
            return False
        allowed_actions = ACCESS_LEVEL_ACTIONS.get(access_level, set())
        return action in allowed_actions

    async def _check_document_access(
        self, user_id: uuid.UUID, document_id: uuid.UUID, action: Action
    ) -> bool:
        """
        Check document-level access.
        Document permission overrides space permission.
        If no document-level permission, inherit from space.
        """
        # Check document-level permission first
        doc_perm = await self._get_document_permission(user_id, document_id)
        if doc_perm is not None:
            allowed_actions = ACCESS_LEVEL_ACTIONS.get(doc_perm, set())
            return action in allowed_actions

        # Fall back to space permission (inheritance)
        space_id = await self._get_document_space_id(document_id)
        if space_id is None:
            return False
        return await self._check_space_access(user_id, space_id, action)

    # ─── Permission CRUD ───────────────────────────────────────────────

    async def set_space_permission(
        self,
        space_id: uuid.UUID,
        user_id: uuid.UUID,
        access_level: AccessLevel,
    ) -> Permission:
        """
        Set space-level permission for a user.
        Creates or updates the permission record.
        Syncs to Qdrant and invalidates cache.
        On Qdrant sync failure, rolls back the DB change.
        """
        # Save current state for rollback
        existing = await self._get_permission_record(
            space_id, ResourceType.space, user_id
        )
        old_access_level = existing.access_level if existing else None

        # Upsert permission in DB
        permission = await self._upsert_permission(
            resource_id=space_id,
            resource_type=ResourceType.space,
            user_id=user_id,
            access_level=access_level,
        )

        # Sync to Qdrant
        try:
            await self._sync_space_permissions_to_qdrant(space_id)
        except Exception as e:
            logger.error(f"Qdrant sync failed for space {space_id}: {e}")
            # Rollback: revert permission in DB
            await self._rollback_permission(
                resource_id=space_id,
                resource_type=ResourceType.space,
                user_id=user_id,
                old_access_level=old_access_level,
            )
            raise RuntimeError(
                f"权限同步到 Qdrant 失败，已回滚权限变更: {e}"
            ) from e

        # Invalidate cache
        await self._invalidate_space_cache(user_id, space_id)

        # Trigger async bulk update for all documents in the space
        from app.tasks.permission_tasks import sync_space_permissions_async

        sync_space_permissions_async.delay(str(space_id))

        return permission

    async def set_document_permission(
        self,
        document_id: uuid.UUID,
        user_id: uuid.UUID,
        access_level: AccessLevel,
    ) -> Permission:
        """
        Set document-level permission for a user.
        Document permissions override space permissions.
        Syncs to Qdrant and invalidates cache.
        On Qdrant sync failure, rolls back the DB change.
        """
        # Save current state for rollback
        existing = await self._get_permission_record(
            document_id, ResourceType.document, user_id
        )
        old_access_level = existing.access_level if existing else None

        # Upsert permission in DB
        permission = await self._upsert_permission(
            resource_id=document_id,
            resource_type=ResourceType.document,
            user_id=user_id,
            access_level=access_level,
        )

        # Sync to Qdrant
        try:
            await self._sync_document_permissions_to_qdrant(document_id)
        except Exception as e:
            logger.error(f"Qdrant sync failed for document {document_id}: {e}")
            # Rollback: revert permission in DB
            await self._rollback_permission(
                resource_id=document_id,
                resource_type=ResourceType.document,
                user_id=user_id,
                old_access_level=old_access_level,
            )
            raise RuntimeError(
                f"权限同步到 Qdrant 失败，已回滚权限变更: {e}"
            ) from e

        # Invalidate cache for the document's space
        space_id = await self._get_document_space_id(document_id)
        if space_id:
            await self._invalidate_space_cache(user_id, space_id)

        return permission

    async def get_space_permissions(
        self, space_id: uuid.UUID
    ) -> list[Permission]:
        """Get all permissions for a space."""
        stmt = select(Permission).where(
            Permission.resource_id == space_id,
            Permission.resource_type == ResourceType.space,
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_document_permissions(
        self, document_id: uuid.UUID
    ) -> list[Permission]:
        """Get all permissions for a document."""
        stmt = select(Permission).where(
            Permission.resource_id == document_id,
            Permission.resource_type == ResourceType.document,
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_effective_permission(
        self, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> AccessLevel | None:
        """
        Get effective permission for a user on a document.
        Document-level overrides space-level.
        """
        # Check document-level first
        doc_perm = await self._get_document_permission(user_id, document_id)
        if doc_perm is not None:
            return doc_perm

        # Fall back to space
        space_id = await self._get_document_space_id(document_id)
        if space_id is None:
            return None
        return await self._get_space_permission(user_id, space_id)

    # ─── Qdrant Sync ──────────────────────────────────────────────────

    async def _sync_space_permissions_to_qdrant(
        self, space_id: uuid.UUID
    ) -> None:
        """
        Sync permissions for all documents in a space to Qdrant payload.
        Updates allowed_user_ids field on all chunks belonging to documents in this space.
        Must complete within 3 seconds for the immediate sync.

        实现说明：
        Qdrant 官方客户端为同步实现，``set_payload`` 是非阻塞的轻量 HTTP 调用，
        即使在 FastAPI 异步上下文中直接调用也只会短暂阻塞事件循环（且 timeout=3s
        与设计中"3 秒内完成"约束一致）。如果未来对延迟敏感，可以替换为
        ``AsyncQdrantClient`` 或 ``run_in_executor``，但当前不引入额外依赖。
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from app.core.config import get_settings

        settings = get_settings()

        # Get all users with read or write access to this space
        allowed_user_ids = await self._get_allowed_user_ids_for_space(space_id)

        # Connect to Qdrant
        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=3.0,
        )

        try:
            # Update all points with this space_id
            client.set_payload(
                collection_name="document_chunks",
                payload={"allowed_user_ids": [str(uid) for uid in allowed_user_ids]},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="space_id",
                            match=MatchValue(value=str(space_id)),
                        )
                    ]
                ),
            )
        finally:
            client.close()

    async def _sync_document_permissions_to_qdrant(
        self, document_id: uuid.UUID
    ) -> None:
        """
        Sync permissions for a specific document to Qdrant payload.
        Updates allowed_user_ids field on all chunks belonging to this document.
        Must complete within 3 seconds.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from app.core.config import get_settings

        settings = get_settings()

        # Get allowed user IDs for this document
        allowed_user_ids = await self._get_allowed_user_ids_for_document(document_id)

        # Connect to Qdrant
        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=3.0,
        )

        try:
            # Update all points with this document_id
            client.set_payload(
                collection_name="document_chunks",
                payload={"allowed_user_ids": [str(uid) for uid in allowed_user_ids]},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )
        finally:
            client.close()

    async def _get_allowed_user_ids_for_space(
        self, space_id: uuid.UUID
    ) -> list[uuid.UUID]:
        """Get list of user IDs that have read or write access to a space."""
        stmt = select(Permission.user_id).where(
            Permission.resource_id == space_id,
            Permission.resource_type == ResourceType.space,
            Permission.access_level.in_([AccessLevel.read, AccessLevel.write]),
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def _get_allowed_user_ids_for_document(
        self, document_id: uuid.UUID
    ) -> list[uuid.UUID]:
        """
        Get list of user IDs that have read or write access to a document.
        Considers both document-level and inherited space-level permissions.
        """
        # Get document-level permissions
        doc_stmt = select(Permission.user_id, Permission.access_level).where(
            Permission.resource_id == document_id,
            Permission.resource_type == ResourceType.document,
        )
        doc_result = await self.db.execute(doc_stmt)
        doc_perms = {row[0]: row[1] for row in doc_result.all()}

        # Get space-level permissions
        space_id = await self._get_document_space_id(document_id)
        space_perms: dict[uuid.UUID, AccessLevel] = {}
        if space_id:
            space_stmt = select(Permission.user_id, Permission.access_level).where(
                Permission.resource_id == space_id,
                Permission.resource_type == ResourceType.space,
            )
            space_result = await self.db.execute(space_stmt)
            space_perms = {row[0]: row[1] for row in space_result.all()}

        # Merge: document-level overrides space-level
        all_user_ids = set(doc_perms.keys()) | set(space_perms.keys())
        allowed: list[uuid.UUID] = []
        for uid in all_user_ids:
            # Document-level takes priority
            if uid in doc_perms:
                level = doc_perms[uid]
            else:
                level = space_perms[uid]
            if level in (AccessLevel.read, AccessLevel.write):
                allowed.append(uid)

        return allowed

    # ─── Redis Cache ───────────────────────────────────────────────────

    async def _get_space_permission_cached(
        self, user_id: uuid.UUID, space_id: uuid.UUID
    ) -> AccessLevel | None:
        """Get space permission from cache, falling back to DB."""
        cache_key = PERM_CACHE_KEY_PATTERN.format(
            user_id=str(user_id), space_id=str(space_id)
        )

        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached is not None:
            if cached == "__none__":
                return None
            try:
                return AccessLevel(cached)
            except ValueError:
                pass

        # Cache miss: query DB
        access_level = await self._get_space_permission(user_id, space_id)

        # Store in cache
        cache_value = access_level.value if access_level else "__none__"
        await self.redis.set(cache_key, cache_value, ex=PERM_CACHE_TTL)

        return access_level

    async def _invalidate_space_cache(
        self, user_id: uuid.UUID, space_id: uuid.UUID
    ) -> None:
        """Invalidate the Redis cache for a specific user-space permission."""
        cache_key = PERM_CACHE_KEY_PATTERN.format(
            user_id=str(user_id), space_id=str(space_id)
        )
        await self.redis.delete(cache_key)

    async def invalidate_all_space_cache(self, space_id: uuid.UUID) -> None:
        """Invalidate all cached permissions for a space (all users)."""
        pattern = f"perm:user:*:space:{space_id}"
        async for key in self.redis.scan_iter(match=pattern):
            await self.redis.delete(key)

    # ─── Internal Helpers ──────────────────────────────────────────────

    async def _get_space_permission(
        self, user_id: uuid.UUID, space_id: uuid.UUID
    ) -> AccessLevel | None:
        """Get space permission from DB."""
        stmt = select(Permission.access_level).where(
            Permission.resource_id == space_id,
            Permission.resource_type == ResourceType.space,
            Permission.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        row = result.scalar_one_or_none()
        return row

    async def _get_document_permission(
        self, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> AccessLevel | None:
        """Get document-level permission from DB."""
        stmt = select(Permission.access_level).where(
            Permission.resource_id == document_id,
            Permission.resource_type == ResourceType.document,
            Permission.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_document_space_id(
        self, document_id: uuid.UUID
    ) -> uuid.UUID | None:
        """Get the space_id for a document."""
        stmt = select(Document.space_id).where(Document.id == document_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_permission_record(
        self,
        resource_id: uuid.UUID,
        resource_type: ResourceType,
        user_id: uuid.UUID,
    ) -> Permission | None:
        """Get a specific permission record."""
        stmt = select(Permission).where(
            Permission.resource_id == resource_id,
            Permission.resource_type == resource_type,
            Permission.user_id == user_id,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def _upsert_permission(
        self,
        resource_id: uuid.UUID,
        resource_type: ResourceType,
        user_id: uuid.UUID,
        access_level: AccessLevel,
    ) -> Permission:
        """Create or update a permission record."""
        existing = await self._get_permission_record(
            resource_id, resource_type, user_id
        )
        if existing:
            existing.access_level = access_level
            await self.db.flush()
            return existing
        else:
            permission = Permission(
                resource_id=resource_id,
                resource_type=resource_type,
                user_id=user_id,
                access_level=access_level,
            )
            self.db.add(permission)
            await self.db.flush()
            return permission

    async def _rollback_permission(
        self,
        resource_id: uuid.UUID,
        resource_type: ResourceType,
        user_id: uuid.UUID,
        old_access_level: AccessLevel | None,
    ) -> None:
        """Rollback a permission change to its previous state."""
        if old_access_level is None:
            # Permission didn't exist before, delete it
            existing = await self._get_permission_record(
                resource_id, resource_type, user_id
            )
            if existing:
                await self.db.delete(existing)
                await self.db.flush()
        else:
            # Revert to old access level
            existing = await self._get_permission_record(
                resource_id, resource_type, user_id
            )
            if existing:
                existing.access_level = old_access_level
                await self.db.flush()
