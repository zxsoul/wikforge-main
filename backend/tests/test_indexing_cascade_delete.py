"""文档删除时的级联清理（task 12.9）。

聚焦点：
- ``IndexingService.delete_document_chunks`` 同时操作 Qdrant 和 OpenSearch；
- Qdrant 失败抛 ``IndexingError``，OpenSearch **不会** 被调用（避免半清理）；
- OpenSearch 失败 *在* Qdrant 成功 *之后* 不抛错，而是返回
  ``opensearch_error`` 字段（部分清理契约，让上层 SQL 删除继续提交）；
- ``DocumentService.delete_document`` / ``delete_space`` 在删行前调用
  ``delete_document_chunks``，确保 PostgreSQL 与搜索后端保持一致。

只测 task 12.9 关心的契约；不重复 ``test_indexing.py`` 已覆盖的双写路径。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.document import Document, DocumentStatus
from app.models.space import Space
from app.services.document_service import DocumentService
from app.services.indexing_service import (
    IndexingError,
    IndexingService,
)


# ─── IndexingService.delete_document_chunks ───────────────────────────


class TestDeleteDocumentChunks:
    """级联清理在 IndexingService 层的契约。"""

    def _make_service(
        self,
        *,
        os_response: dict | None = None,
        os_exc: Exception | None = None,
        qdrant_exc: Exception | None = None,
    ) -> tuple[IndexingService, MagicMock, MagicMock]:
        """构造已注入 mock 后端的 IndexingService。"""
        mock_qdrant = MagicMock()
        if qdrant_exc is not None:
            mock_qdrant.delete.side_effect = qdrant_exc

        mock_os = MagicMock()
        if os_exc is not None:
            mock_os.delete_by_query.side_effect = os_exc
        else:
            mock_os.delete_by_query.return_value = os_response or {"deleted": 0}

        service = IndexingService()
        # 直接注入避免命中真实客户端构造（lazy property 会触发网络配置加载）。
        service._qdrant = mock_qdrant
        service._opensearch = mock_os
        return service, mock_qdrant, mock_os

    def test_calls_qdrant_filter_delete_by_document_id(self) -> None:
        """Qdrant 删除使用 ``document_id`` payload filter（而不是按点 ID 列表）。"""
        from qdrant_client.models import Filter

        service, mock_qdrant, _ = self._make_service(os_response={"deleted": 3})

        service.delete_document_chunks("doc-abc")

        mock_qdrant.delete.assert_called_once()
        kwargs = mock_qdrant.delete.call_args.kwargs
        # collection_name 必须显式指定（避免误删其他 collection）。
        assert kwargs["collection_name"] == "document_chunks"
        selector = kwargs["points_selector"]
        # 选择器是 Filter（按 payload 过滤）而不是简单的 ID 列表。
        assert isinstance(selector, Filter)
        # 过滤条件锁定到目标 document_id。
        must = selector.must or []
        assert len(must) == 1
        assert must[0].key == "document_id"
        assert must[0].match.value == "doc-abc"

    def test_calls_opensearch_delete_by_query_with_refresh(self) -> None:
        """OpenSearch 删除按 ``document_id`` term 查询，且 ``refresh=True``
        以让删除立即对后续搜索可见。"""
        service, _, mock_os = self._make_service(os_response={"deleted": 7})

        service.delete_document_chunks("doc-xyz")

        mock_os.delete_by_query.assert_called_once()
        kwargs = mock_os.delete_by_query.call_args.kwargs
        assert kwargs["index"] == "chunks"
        assert kwargs["refresh"] is True
        assert kwargs["body"] == {
            "query": {"term": {"document_id": "doc-xyz"}}
        }

    def test_returns_deletion_counts(self) -> None:
        """返回值包含 Qdrant 哨兵值和 OpenSearch 实际删除条数。"""
        service, _, _ = self._make_service(os_response={"deleted": 12})

        result = service.delete_document_chunks("doc-1")

        # Qdrant filter-delete 不返回计数，使用 -1 哨兵。
        assert result["qdrant_deleted"] == -1
        assert result["opensearch_deleted"] == 12
        # 全成功时不应出现 opensearch_error 字段，避免上层误判为部分清理。
        assert "opensearch_error" not in result

    def test_qdrant_failure_raises_indexing_error_and_skips_opensearch(self) -> None:
        """Qdrant 失败抛 ``IndexingError``，且 **不** 触碰 OpenSearch。

        这是关键的"全有或全无"语义：让上层中止 SQL 删除，避免
        PostgreSQL 与 Qdrant 不一致——若此时去清 OpenSearch 反而
        会制造另一种半清理。"""
        service, mock_qdrant, mock_os = self._make_service(
            qdrant_exc=RuntimeError("qdrant unreachable"),
        )

        with pytest.raises(IndexingError, match="Qdrant deletion failed"):
            service.delete_document_chunks("doc-1")

        mock_qdrant.delete.assert_called_once()
        # OpenSearch 路径不应被触发。
        mock_os.delete_by_query.assert_not_called()

    def test_opensearch_failure_after_qdrant_returns_partial_cleanup(self) -> None:
        """Qdrant 已成功、OpenSearch 失败：不抛错，返回 ``opensearch_error``。

        部分清理契约：再抛 ``IndexingError`` 会逼上层回滚 Qdrant，
        而 Qdrant 已经清干净了——这反而是更糟的状态。"""
        service, mock_qdrant, _ = self._make_service(
            os_exc=RuntimeError("opensearch timeout"),
        )

        result = service.delete_document_chunks("doc-2")

        # Qdrant 仍按预期成功调用了一次。
        mock_qdrant.delete.assert_called_once()
        # 哨兵值确认 Qdrant 路径走完。
        assert result["qdrant_deleted"] == -1
        # OpenSearch 失败原因被透传给调用方做日志/对账。
        assert "opensearch_error" in result
        assert "opensearch timeout" in result["opensearch_error"]


# ─── DocumentService 与 IndexingService 的接线 ────────────────────────


def _make_space_row(space_id: uuid.UUID | None = None) -> MagicMock:
    space = MagicMock(spec=Space)
    space.id = space_id or uuid.uuid4()
    space.name = "Test Space"
    space.description = None
    space.created_by = uuid.uuid4()
    space.created_at = datetime.now(timezone.utc)
    space.updated_at = datetime.now(timezone.utc)
    return space


def _make_document_row(
    doc_id: uuid.UUID | None = None,
    space_id: uuid.UUID | None = None,
) -> MagicMock:
    doc = MagicMock(spec=Document)
    doc.id = doc_id or uuid.uuid4()
    doc.space_id = space_id or uuid.uuid4()
    doc.folder_id = None
    doc.title = "Doc"
    doc.file_type = "pdf"
    doc.file_size = 100
    doc.storage_path = "/p"
    doc.status = DocumentStatus.completed
    doc.created_at = datetime.now(timezone.utc)
    doc.updated_at = datetime.now(timezone.utc)
    return doc


class TestDocumentServiceWiring:
    """删除流程是否真的把 ``IndexingService.delete_document_chunks`` 接上了。"""

    @pytest.fixture
    def mock_db(self) -> AsyncMock:
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.delete = AsyncMock()
        db.refresh = AsyncMock()
        return db

    @pytest.fixture
    def service(self, mock_db: AsyncMock) -> DocumentService:
        return DocumentService(db=mock_db)

    @pytest.mark.asyncio
    @patch("app.services.indexing_service.IndexingService")
    async def test_delete_document_invokes_index_cleanup(
        self,
        mock_indexing_cls: MagicMock,
        service: DocumentService,
        mock_db: AsyncMock,
    ) -> None:
        """``DocumentService.delete_document`` 必须先做向量清理再删 SQL 行。"""
        doc = _make_document_row()

        # _get_document 内部 select(Document)
        get_doc_result = MagicMock()
        get_doc_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=get_doc_result)

        mock_service = MagicMock()
        mock_service.delete_document_chunks.return_value = {
            "qdrant_deleted": -1,
            "opensearch_deleted": 0,
        }
        mock_indexing_cls.return_value = mock_service

        await service.delete_document(doc.id)

        # 关键断言：清理被调用，且参数是 document_id 的字符串形式。
        mock_service.delete_document_chunks.assert_called_once_with(str(doc.id))
        # SQL 删除也照常发生。
        mock_db.delete.assert_called_once_with(doc)
        mock_db.flush.assert_called()

    @pytest.mark.asyncio
    @patch("app.services.indexing_service.IndexingService")
    async def test_delete_document_qdrant_failure_aborts_sql_delete(
        self,
        mock_indexing_cls: MagicMock,
        service: DocumentService,
        mock_db: AsyncMock,
    ) -> None:
        """Qdrant 清理失败 → IndexingError 透传 → SQL 行不被删除。"""
        doc = _make_document_row()

        get_doc_result = MagicMock()
        get_doc_result.scalar_one_or_none.return_value = doc
        mock_db.execute = AsyncMock(return_value=get_doc_result)

        mock_service = MagicMock()
        mock_service.delete_document_chunks.side_effect = IndexingError(
            "Qdrant deletion failed: boom"
        )
        mock_indexing_cls.return_value = mock_service

        with pytest.raises(IndexingError):
            await service.delete_document(doc.id)

        # 清理失败必须阻止 SQL 删除（保持 PostgreSQL 与搜索后端一致）。
        mock_db.delete.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.services.indexing_service.IndexingService")
    async def test_delete_space_cleans_each_documents_indices(
        self,
        mock_indexing_cls: MagicMock,
        service: DocumentService,
        mock_db: AsyncMock,
    ) -> None:
        """删除空间时，应对空间内每篇文档分别调用 ``delete_document_chunks``。"""
        space = _make_space_row()
        doc_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]

        # 1. get_space → 返回 space
        # 2. select(Document.id) where space_id == … → 返回 [(doc_id,), …]
        get_space_result = MagicMock()
        get_space_result.scalar_one_or_none.return_value = space

        list_doc_ids_result = MagicMock()
        list_doc_ids_result.all.return_value = [(d,) for d in doc_ids]

        mock_db.execute = AsyncMock(
            side_effect=[get_space_result, list_doc_ids_result]
        )

        mock_service = MagicMock()
        mock_service.delete_document_chunks.return_value = {
            "qdrant_deleted": -1,
            "opensearch_deleted": 0,
        }
        mock_indexing_cls.return_value = mock_service

        await service.delete_space(space.id)

        # 每篇文档都得到一次清理调用。
        assert mock_service.delete_document_chunks.call_count == len(doc_ids)
        called_ids = {
            call.args[0] for call in mock_service.delete_document_chunks.call_args_list
        }
        assert called_ids == {str(d) for d in doc_ids}
        # SQL 行删除照常进行。
        mock_db.delete.assert_called_once_with(space)
