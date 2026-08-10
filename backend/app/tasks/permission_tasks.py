"""Celery tasks for asynchronous permission synchronization to Qdrant.

These tasks handle bulk updates when permissions change, ensuring all
affected document chunks in Qdrant have their allowed_user_ids updated
within 60 seconds.

设计参考（design.md §6 权限服务）：
- ``time_limit=60`` 与 ``max_retries=3`` 与设计中"权限变更 60 秒内同步"约束一致；
- 任务使用 ``asyncio.run`` 在独立事件循环中执行异步逻辑，避免与 worker 线程
  共享事件循环。``asyncio.get_event_loop()`` 在 Python 3.12+ 已弃用，因此
  统一改为 ``asyncio.run``，并保留 try/except 兜底（极少数情况下当前线程
  仍持有运行中的事件循环时，通过新建独立循环再回退一次）。
"""

import asyncio
import logging
import uuid
from typing import Awaitable, TypeVar

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _run_async(coro: Awaitable[T]) -> T:
    """在 Celery worker（同步上下文）中运行异步协程。

    优先 ``asyncio.run``；若调用线程已存在运行中的事件循环（理论上 Celery
    prefork worker 不会出现，但在某些 eventlet/gevent 池下可能命中），
    则新建独立事件循环执行后再关闭。
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


@celery_app.task(
    bind=True,
    name="permissions.sync_space_permissions_async",
    max_retries=3,
    default_retry_delay=5,
    soft_time_limit=55,
    time_limit=60,
)
def sync_space_permissions_async(self, space_id: str) -> dict:
    """
    Async Celery task to sync all document permissions in a space to Qdrant.

    This task is triggered when space-level permissions change.
    It updates the allowed_user_ids payload field for all chunks
    belonging to documents in the specified space.

    Must complete within 60 seconds.
    """
    return _run_async(_sync_space_permissions(space_id))


@celery_app.task(
    bind=True,
    name="permissions.sync_document_permissions_async",
    max_retries=3,
    default_retry_delay=5,
    soft_time_limit=55,
    time_limit=60,
)
def sync_document_permissions_async(self, document_id: str) -> dict:
    """
    Async Celery task to sync a single document's permissions to Qdrant.

    This task is triggered when document-level permissions change.
    It updates the allowed_user_ids payload field for all chunks
    belonging to the specified document.

    Must complete within 60 seconds.
    """
    return _run_async(_sync_document_permissions(document_id))


async def _sync_space_permissions(space_id: str) -> dict:
    """
    Internal async function to sync space permissions to Qdrant.

    Queries all documents in the space, computes allowed_user_ids for each,
    and updates Qdrant payload in bulk.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from app.core.config import get_settings
    from app.models.document import Document
    from app.models.permission import AccessLevel, Permission, ResourceType

    settings = get_settings()

    engine = create_async_engine(settings.DATABASE_URL, echo=False)

    async with AsyncSession(engine) as db:
        # Get all users with read/write access to this space
        space_uuid = uuid.UUID(space_id)
        space_perm_stmt = select(Permission.user_id, Permission.access_level).where(
            Permission.resource_id == space_uuid,
            Permission.resource_type == ResourceType.space,
        )
        space_result = await db.execute(space_perm_stmt)
        space_perms = {row[0]: row[1] for row in space_result.all()}

        # Get all documents in this space
        doc_stmt = select(Document.id).where(Document.space_id == space_uuid)
        doc_result = await db.execute(doc_stmt)
        document_ids = [row[0] for row in doc_result.all()]

        if not document_ids:
            logger.info(f"No documents in space {space_id}, skipping Qdrant sync")
            await engine.dispose()
            return {"status": "success", "documents_updated": 0}

        # Get all document-level permissions for documents in this space
        doc_perm_stmt = select(
            Permission.resource_id, Permission.user_id, Permission.access_level
        ).where(
            Permission.resource_id.in_(document_ids),
            Permission.resource_type == ResourceType.document,
        )
        doc_perm_result = await db.execute(doc_perm_stmt)
        doc_perms: dict[uuid.UUID, dict[uuid.UUID, AccessLevel]] = {}
        for row in doc_perm_result.all():
            doc_id, uid, level = row
            if doc_id not in doc_perms:
                doc_perms[doc_id] = {}
            doc_perms[doc_id][uid] = level

    await engine.dispose()

    # Connect to Qdrant and update each document's chunks
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY or None,
        timeout=30.0,
    )

    updated_count = 0
    try:
        for doc_id in document_ids:
            # Compute allowed users for this document
            # Document-level overrides space-level
            doc_specific = doc_perms.get(doc_id, {})
            all_users = set(space_perms.keys()) | set(doc_specific.keys())

            allowed_user_ids = []
            for uid in all_users:
                if uid in doc_specific:
                    level = doc_specific[uid]
                else:
                    level = space_perms.get(uid, AccessLevel.invisible)
                if level in (AccessLevel.read, AccessLevel.write):
                    allowed_user_ids.append(str(uid))

            # Update Qdrant payload
            client.set_payload(
                collection_name="document_chunks",
                payload={"allowed_user_ids": allowed_user_ids},
                points=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(doc_id)),
                        )
                    ]
                ),
            )
            updated_count += 1
    finally:
        client.close()

    logger.info(
        f"Space {space_id} permission sync complete: {updated_count} documents updated"
    )
    return {"status": "success", "documents_updated": updated_count}


async def _sync_document_permissions(document_id: str) -> dict:
    """
    Internal async function to sync a single document's permissions to Qdrant.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from app.core.config import get_settings
    from app.models.document import Document
    from app.models.permission import AccessLevel, Permission, ResourceType

    settings = get_settings()

    engine = create_async_engine(settings.DATABASE_URL, echo=False)

    async with AsyncSession(engine) as db:
        doc_uuid = uuid.UUID(document_id)

        # Get document's space_id
        doc_stmt = select(Document.space_id).where(Document.id == doc_uuid)
        doc_result = await db.execute(doc_stmt)
        space_id = doc_result.scalar_one_or_none()

        if space_id is None:
            logger.warning(f"Document {document_id} not found, skipping sync")
            await engine.dispose()
            return {"status": "skipped", "reason": "document_not_found"}

        # Get document-level permissions
        doc_perm_stmt = select(Permission.user_id, Permission.access_level).where(
            Permission.resource_id == doc_uuid,
            Permission.resource_type == ResourceType.document,
        )
        doc_perm_result = await db.execute(doc_perm_stmt)
        doc_perms = {row[0]: row[1] for row in doc_perm_result.all()}

        # Get space-level permissions
        space_perm_stmt = select(Permission.user_id, Permission.access_level).where(
            Permission.resource_id == space_id,
            Permission.resource_type == ResourceType.space,
        )
        space_perm_result = await db.execute(space_perm_stmt)
        space_perms = {row[0]: row[1] for row in space_perm_result.all()}

    await engine.dispose()

    # Compute allowed users (document-level overrides space-level)
    all_users = set(doc_perms.keys()) | set(space_perms.keys())
    allowed_user_ids = []
    for uid in all_users:
        if uid in doc_perms:
            level = doc_perms[uid]
        else:
            level = space_perms.get(uid, AccessLevel.invisible)
        if level in (AccessLevel.read, AccessLevel.write):
            allowed_user_ids.append(str(uid))

    # Update Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY or None,
        timeout=30.0,
    )

    try:
        client.set_payload(
            collection_name="document_chunks",
            payload={"allowed_user_ids": allowed_user_ids},
            points=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
    finally:
        client.close()

    logger.info(
        f"Document {document_id} permission sync complete: "
        f"{len(allowed_user_ids)} users allowed"
    )
    return {"status": "success", "allowed_users": len(allowed_user_ids)}
