"""Service 层 Qdrant 同步逻辑测试 (任务 4.5)。

覆盖：
- ``_get_allowed_user_ids_for_space`` 仅返回 read/write 用户
- ``_get_allowed_user_ids_for_document`` 中文档级覆盖空间级
- ``_sync_*_to_qdrant`` 调用 Qdrant 客户端 set_payload，timeout=3 与设计一致

Validates: Requirements 10
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.models.permission import AccessLevel, ResourceType
from tests._permission_helpers import (
    permission_service,
    perm_db,
    perm_redis,
    rows_all_result,
    scalar_one_or_none_result,
    scalars_all_result,
)


class TestAllowedUserIds:
    @pytest.mark.asyncio
    async def test_space_returns_read_and_write_users(
        self, permission_service, perm_db
    ):
        space_id = uuid.uuid4()
        u_read = uuid.uuid4()
        u_write = uuid.uuid4()
        # SQL where 已过滤掉 invisible，因此结果只含 read/write
        from unittest.mock import AsyncMock

        perm_db.execute = AsyncMock(
            return_value=scalars_all_result([u_read, u_write])
        )

        result = await permission_service._get_allowed_user_ids_for_space(space_id)
        assert set(result) == {u_read, u_write}

    @pytest.mark.asyncio
    async def test_document_doc_level_overrides_space_level(
        self, permission_service, perm_db
    ):
        document_id = uuid.uuid4()
        space_id = uuid.uuid4()
        u_a = uuid.uuid4()
        u_b = uuid.uuid4()

        from unittest.mock import AsyncMock

        # 1) 文档级权限：u_a 被 invisible（覆盖空间级 read）
        # 2) 文档所属空间 → space_id
        # 3) 空间级权限：u_a=read, u_b=write
        perm_db.execute = AsyncMock(
            side_effect=[
                rows_all_result([(u_a, AccessLevel.invisible)]),
                scalar_one_or_none_result(space_id),
                rows_all_result(
                    [(u_a, AccessLevel.read), (u_b, AccessLevel.write)]
                ),
            ]
        )

        result = await permission_service._get_allowed_user_ids_for_document(
            document_id
        )
        assert u_a not in result
        assert u_b in result

    @pytest.mark.asyncio
    async def test_document_no_space_returns_doc_level_only(
        self, permission_service, perm_db
    ):
        document_id = uuid.uuid4()
        u_a = uuid.uuid4()

        from unittest.mock import AsyncMock

        perm_db.execute = AsyncMock(
            side_effect=[
                rows_all_result([(u_a, AccessLevel.write)]),
                scalar_one_or_none_result(None),  # 无 space
            ]
        )

        result = await permission_service._get_allowed_user_ids_for_document(
            document_id
        )
        assert result == [u_a]


class TestQdrantSync:
    """``_sync_*_to_qdrant`` 在 mock Qdrant 客户端下的行为。"""

    @pytest.mark.asyncio
    async def test_sync_space_calls_set_payload_with_allowed_users(
        self, permission_service, perm_db
    ):
        space_id = uuid.uuid4()
        u_read = uuid.uuid4()

        from unittest.mock import AsyncMock

        perm_db.execute = AsyncMock(
            return_value=scalars_all_result([u_read])
        )

        fake_client = MagicMock()
        fake_client.set_payload = MagicMock(return_value=None)
        fake_client.close = MagicMock()

        with patch(
            "qdrant_client.QdrantClient", return_value=fake_client
        ) as ctor:
            await permission_service._sync_space_permissions_to_qdrant(space_id)

        # 客户端用 timeout=3.0 构造（design 约束）
        _, kwargs = ctor.call_args
        assert kwargs.get("timeout") == 3.0

        fake_client.set_payload.assert_called_once()
        call_kwargs = fake_client.set_payload.call_args.kwargs
        assert call_kwargs["collection_name"] == "document_chunks"
        assert call_kwargs["payload"] == {
            "allowed_user_ids": [str(u_read)]
        }
        # Filter 必须按 space_id 限定
        flt = call_kwargs["points"]
        assert flt.must[0].key == "space_id"
        assert flt.must[0].match.value == str(space_id)

        fake_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_document_calls_set_payload_with_doc_filter(
        self, permission_service, perm_db
    ):
        document_id = uuid.uuid4()
        space_id = uuid.uuid4()
        u = uuid.uuid4()

        from unittest.mock import AsyncMock

        perm_db.execute = AsyncMock(
            side_effect=[
                rows_all_result([(u, AccessLevel.write)]),
                scalar_one_or_none_result(space_id),
                rows_all_result([]),
            ]
        )

        fake_client = MagicMock()
        fake_client.set_payload = MagicMock(return_value=None)
        fake_client.close = MagicMock()

        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            await permission_service._sync_document_permissions_to_qdrant(
                document_id
            )

        call_kwargs = fake_client.set_payload.call_args.kwargs
        flt = call_kwargs["points"]
        assert flt.must[0].key == "document_id"
        assert flt.must[0].match.value == str(document_id)
        assert call_kwargs["payload"] == {"allowed_user_ids": [str(u)]}

    @pytest.mark.asyncio
    async def test_sync_closes_client_on_qdrant_error(
        self, permission_service, perm_db
    ):
        """即便 set_payload 抛错也要关闭客户端。"""
        space_id = uuid.uuid4()
        from unittest.mock import AsyncMock

        perm_db.execute = AsyncMock(
            return_value=scalars_all_result([])
        )

        fake_client = MagicMock()
        fake_client.set_payload = MagicMock(side_effect=RuntimeError("boom"))
        fake_client.close = MagicMock()

        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            with pytest.raises(RuntimeError, match="boom"):
                await permission_service._sync_space_permissions_to_qdrant(
                    space_id
                )

        fake_client.close.assert_called_once()
