"""权限 Celery 异步同步任务测试 (任务 4.6)。

覆盖：
- 任务 ``time_limit=60`` / ``soft_time_limit=55`` / ``max_retries=3`` 与 design 一致
- 任务名称符合 ``permissions.*`` 命名空间
- ``_sync_space_permissions`` 协程：mock DB + Qdrant 客户端，
  验证空间内全部文档都被遍历更新
- ``_sync_document_permissions`` 协程：仅更新单个文档的 chunks
- 文档级权限覆盖空间级（合并语义）

Validates: Requirements 10
"""

from __future__ import annotations

import sys
import types
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Celery 在最小测试环境下可能未安装；该模块整体跳过即可。
pytest.importorskip("celery", reason="celery 未安装，跳过 Celery 任务测试")

from app.models.permission import AccessLevel  # noqa: E402
from app.tasks.permission_tasks import (  # noqa: E402
    sync_document_permissions_async,
    sync_space_permissions_async,
    _sync_space_permissions,
    _sync_document_permissions,
)


# ─── Celery 任务声明性约束 ───────────────────────────────────────


class TestTaskDeclaration:
    def test_space_task_time_limits_and_retries(self):
        """空间同步任务的 time_limit、soft_time_limit、max_retries 与设计一致。"""
        opts = sync_space_permissions_async
        assert opts.time_limit == 60
        assert opts.soft_time_limit == 55
        assert opts.max_retries == 3
        assert opts.name == "permissions.sync_space_permissions_async"

    def test_document_task_time_limits_and_retries(self):
        opts = sync_document_permissions_async
        assert opts.time_limit == 60
        assert opts.soft_time_limit == 55
        assert opts.max_retries == 3
        assert opts.name == "permissions.sync_document_permissions_async"


# ─── 异步内部函数测试（绕过 Celery 调度，直接验证业务逻辑）────────


def _scalar_one_or_none(value):
    r = MagicMock()
    r.scalar_one_or_none.return_value = value
    return r


def _rows_all(rows):
    r = MagicMock()
    r.all.return_value = list(rows)
    return r


@contextmanager
def _patched_db_engine(execute_side_effect):
    """构造一个伪造的 ``create_async_engine``，使 ``AsyncSession`` 上下文返回 mock db。

    设计：``permission_tasks._sync_*`` 内部 ``async with AsyncSession(engine) as db``，
    我们 patch ``AsyncSession`` 类本身让其返回支持上下文的对象。
    """
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(side_effect=execute_side_effect)

    fake_session_cm = AsyncMock()
    fake_session_cm.__aenter__ = AsyncMock(return_value=fake_db)
    fake_session_cm.__aexit__ = AsyncMock(return_value=None)

    def _async_session_factory(_engine):
        return fake_session_cm

    fake_engine = AsyncMock()
    fake_engine.dispose = AsyncMock()

    with patch(
        "app.tasks.permission_tasks.uuid",  # 占位（实际 patch 在外部）
        uuid,
    ):
        # 真正的 patch：让 sqlalchemy 引擎/会话使用 mock
        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=fake_engine,
        ), patch(
            "sqlalchemy.ext.asyncio.AsyncSession",
            side_effect=_async_session_factory,
        ):
            yield fake_db, fake_engine


# 由于 ``_sync_space_permissions`` 内部对 sqlalchemy 模块做了延迟 import，
# 直接 patch ``sqlalchemy.ext.asyncio.AsyncSession`` 即可生效。


class _FakeQdrantClient:
    """记录 set_payload 调用以便断言遍历范围。"""

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, list[str]]] = []
        self.closed = False

    def set_payload(self, *, collection_name, payload, points):
        # 从 Filter 中提取 document_id 文本
        doc_id = None
        try:
            doc_id = points.must[0].match.value
        except Exception:  # pragma: no cover
            doc_id = None
        self.calls.append((doc_id, list(payload.get("allowed_user_ids", []))))

    def close(self):
        self.closed = True


class TestSyncSpacePermissionsCoroutine:
    """``_sync_space_permissions`` 应遍历空间内所有文档并调用 Qdrant set_payload。"""

    @pytest.mark.asyncio
    async def test_iterates_all_documents_in_space(self):
        space_id = uuid.uuid4()
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        doc_1 = uuid.uuid4()
        doc_2 = uuid.uuid4()

        # 1) 空间级权限：user_a=read, user_b=write
        space_perms_rows = _rows_all(
            [(user_a, AccessLevel.read), (user_b, AccessLevel.write)]
        )
        # 2) 空间内文档列表：[doc_1, doc_2]
        documents_rows = _rows_all([(doc_1,), (doc_2,)])
        # 3) 文档级权限：仅 doc_1 上 user_a=invisible 覆盖
        doc_perm_rows = _rows_all([(doc_1, user_a, AccessLevel.invisible)])

        execute_results = iter([space_perms_rows, documents_rows, doc_perm_rows])

        fake_db = AsyncMock()
        fake_db.execute = AsyncMock(side_effect=lambda *a, **kw: next(execute_results))

        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()

        # 构造 ``async with AsyncSession(engine) as db:`` 上下文
        @contextmanager
        def _noop():
            yield

        class _SessionCM:
            async def __aenter__(self_inner):
                return fake_db

            async def __aexit__(self_inner, *exc):
                return None

        fake_qdrant = _FakeQdrantClient()

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=fake_engine,
        ), patch(
            "sqlalchemy.ext.asyncio.AsyncSession",
            return_value=_SessionCM(),
        ), patch(
            "qdrant_client.QdrantClient", return_value=fake_qdrant
        ):
            result = await _sync_space_permissions(str(space_id))

        assert result == {"status": "success", "documents_updated": 2}
        # Qdrant 被调用两次：每个文档一次
        assert len(fake_qdrant.calls) == 2
        # 收集每个文档的 allowed_user_ids
        by_doc = {call[0]: set(call[1]) for call in fake_qdrant.calls}

        # doc_1: user_a 被 invisible 覆盖 → 仅 user_b
        # doc_2: 继承空间 → user_a + user_b
        assert by_doc[str(doc_1)] == {str(user_b)}
        assert by_doc[str(doc_2)] == {str(user_a), str(user_b)}
        assert fake_qdrant.closed is True
        fake_engine.dispose.assert_awaited()

    @pytest.mark.asyncio
    async def test_no_documents_in_space_skips_qdrant(self):
        space_id = uuid.uuid4()

        space_perms_rows = _rows_all([])
        documents_rows = _rows_all([])
        execute_results = iter([space_perms_rows, documents_rows])

        fake_db = AsyncMock()
        fake_db.execute = AsyncMock(side_effect=lambda *a, **kw: next(execute_results))

        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()

        class _SessionCM:
            async def __aenter__(self_inner):
                return fake_db

            async def __aexit__(self_inner, *exc):
                return None

        fake_qdrant = _FakeQdrantClient()

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=fake_engine,
        ), patch(
            "sqlalchemy.ext.asyncio.AsyncSession",
            return_value=_SessionCM(),
        ), patch(
            "qdrant_client.QdrantClient", return_value=fake_qdrant
        ):
            result = await _sync_space_permissions(str(space_id))

        assert result == {"status": "success", "documents_updated": 0}
        # 没有调用 Qdrant
        assert fake_qdrant.calls == []


class TestSyncDocumentPermissionsCoroutine:
    """``_sync_document_permissions`` 仅同步单个文档的 chunks。"""

    @pytest.mark.asyncio
    async def test_updates_only_target_document(self):
        document_id = uuid.uuid4()
        space_id = uuid.uuid4()
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        # 1) 文档 → space_id
        # 2) 文档级权限：user_a=write
        # 3) 空间级权限：user_b=read
        execute_results = iter([
            _scalar_one_or_none(space_id),
            _rows_all([(user_a, AccessLevel.write)]),
            _rows_all([(user_b, AccessLevel.read)]),
        ])

        fake_db = AsyncMock()
        fake_db.execute = AsyncMock(side_effect=lambda *a, **kw: next(execute_results))

        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()

        class _SessionCM:
            async def __aenter__(self_inner):
                return fake_db

            async def __aexit__(self_inner, *exc):
                return None

        fake_qdrant = _FakeQdrantClient()

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=fake_engine,
        ), patch(
            "sqlalchemy.ext.asyncio.AsyncSession",
            return_value=_SessionCM(),
        ), patch(
            "qdrant_client.QdrantClient", return_value=fake_qdrant
        ):
            result = await _sync_document_permissions(str(document_id))

        assert result["status"] == "success"
        # 仅一次 set_payload 调用
        assert len(fake_qdrant.calls) == 1
        called_doc_id, allowed = fake_qdrant.calls[0]
        assert called_doc_id == str(document_id)
        # 合并：user_a (doc-level write) + user_b (space-level read)
        assert set(allowed) == {str(user_a), str(user_b)}
        assert fake_qdrant.closed is True

    @pytest.mark.asyncio
    async def test_document_not_found_skips_sync(self):
        document_id = uuid.uuid4()

        execute_results = iter([_scalar_one_or_none(None)])

        fake_db = AsyncMock()
        fake_db.execute = AsyncMock(side_effect=lambda *a, **kw: next(execute_results))

        fake_engine = MagicMock()
        fake_engine.dispose = AsyncMock()

        class _SessionCM:
            async def __aenter__(self_inner):
                return fake_db

            async def __aexit__(self_inner, *exc):
                return None

        fake_qdrant = _FakeQdrantClient()

        with patch(
            "sqlalchemy.ext.asyncio.create_async_engine",
            return_value=fake_engine,
        ), patch(
            "sqlalchemy.ext.asyncio.AsyncSession",
            return_value=_SessionCM(),
        ), patch(
            "qdrant_client.QdrantClient", return_value=fake_qdrant
        ):
            result = await _sync_document_permissions(str(document_id))

        assert result == {"status": "skipped", "reason": "document_not_found"}
        assert fake_qdrant.calls == []
