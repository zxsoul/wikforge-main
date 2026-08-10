"""向量化与入库集成测试（任务 12.10）。

聚焦点：把任务 12.1–12.9 单独锤过的零件**串起来**跑通一遍 —— 验证它们能
按管线契约协作完成 ``embed → index → search/cleanup`` 闭环。

与已有 12.x 单元测试的边界划分：

- ``test_embedding_service.py`` (12.3) / ``test_embedding_sparse.py`` (12.4)
  各自固化 EmbeddingService 一侧的契约（dense 维度 / sparse 形状）。
- ``test_indexing_qdrant_write.py`` (12.5) / ``test_indexing_opensearch_bulk.py``
  (12.6) 各自锁定单后端写路径。
- ``test_indexing_dual_write.py`` (12.7) 单独锁定双写事务的回滚分支。
- ``test_indexing_cascade_delete.py`` (12.9) 单独锁定级联清理的语义。
- ``test_pipeline_status_updates.py`` (12.8) 单独锁定 Redis/PG 状态过渡。

本模块**不重复**这些细粒度断言，而是**只**验证它们能拼出端到端正确的
``PointStruct`` / OpenSearch action / 删除路径，以及状态在边界处确实被
触发——任何一个零件契约破裂时本测试也会一起红灯，方便快速定位。

后端策略：
- ``qdrant_client`` 复用 ``test_indexing_qdrant_write.py`` 安装的轻量 stub
  （``PointStruct`` / ``SparseVector`` / ``Filter`` 等都是 dataclass）。
- ``litellm.aembedding`` 注入异步 stub，返回固定形状的 dense 向量。
- ``opensearchpy.helpers.bulk`` 用 ``patch`` 拦截，捕获实际 actions。
- Qdrant / OpenSearch 客户端实例直接注入 MagicMock（避免触发 lazy
  initialization 走真实网络）。

整个测试不依赖任何真实网络服务。

Validates: Requirements 4
"""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ``qdrant_client`` 的轻量 stub —— 复用 12.5 测试模块中的安装函数。
from tests import test_indexing_qdrant_write as _qd_stubs  # noqa: F401

from app.services.embedding_service import (  # noqa: E402
    DENSE_VECTOR_DIM,
    EmbeddingResult,
    EmbeddingService,
)
from app.services.indexing_service import (  # noqa: E402
    ChunkPayload,
    IndexingService,
)


# ─── 共用 fixtures ───────────────────────────────────────────────────


@pytest.fixture
def litellm_stub():
    """注入异步 ``litellm.aembedding`` stub，返回固定 1024 维向量。

    与 ``test_embedding_service.py`` 的 fixture 同模式，但默认 side_effect
    会按输入数量动态生成对应数量的向量，让批量调用自动配齐。
    """
    original = sys.modules.get("litellm")
    stub = types.ModuleType("litellm")

    async def fake_aembedding(**kwargs: Any) -> MagicMock:
        texts = kwargs.get("input", [])
        response = MagicMock()
        # 用与文本数量相等的、互不相同的常量向量，方便断言每条 chunk 拿到
        # 自己的那条（dense_vector[0] = 文本在 batch 中的下标 / 100）。
        response.data = [
            {"embedding": [(i + 1) / 100.0] * DENSE_VECTOR_DIM}
            for i, _ in enumerate(texts)
        ]
        return response

    stub.aembedding = AsyncMock(side_effect=fake_aembedding)
    sys.modules["litellm"] = stub
    try:
        yield stub
    finally:
        if original is not None:
            sys.modules["litellm"] = original
        else:
            sys.modules.pop("litellm", None)


@pytest.fixture
def indexing_service():
    """构造 IndexingService 并预注入 mock 后端，避免真实客户端构造。"""
    service = IndexingService()
    service._qdrant = MagicMock(name="qdrant_client")
    service._opensearch = MagicMock(name="opensearch_client")
    return service


def _make_chunk_payload(
    *,
    document_id: str,
    space_id: str,
    chunk_index: int,
    content: str,
    user_ids: list[str],
) -> ChunkPayload:
    """构造一个真实 ChunkPayload —— chunk_id 是合法 UUID（Qdrant 强制要求）。"""
    return ChunkPayload(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        space_id=space_id,
        chunk_index=chunk_index,
        title_chain=f"第一章 > 1.{chunk_index + 1}",
        source_file="handbook.pdf",
        page_number=chunk_index + 1,
        content=content,
        parent_chunk_id=None,
        depth=2,
        token_count=max(1, len(content)),
        allowed_user_ids=user_ids,
        access_level="read",
    )


# ─── 集成测试 ────────────────────────────────────────────────────────


class TestEmbedAndIndexIntegration:
    """``EmbeddingService.embed_chunks`` → ``IndexingService.index_chunks`` 串联。

    单一 happy-path 集成场景：
    1. 真实 EmbeddingService 跑出 dense + sparse 向量；
    2. 把 ``EmbeddingResult`` 对应到 ``ChunkPayload`` 一并喂给 IndexingService；
    3. 检查 Qdrant 收到的 ``PointStruct`` 与 OpenSearch 收到的 bulk action
       在数量、ID、payload 字段上彼此一致——任何一侧零件的契约破裂都会
       立刻引爆这个断言序列。
    """

    def test_embed_then_index_dual_write_uses_consistent_chunk_ids_and_payloads(
        self, litellm_stub, indexing_service
    ):
        document_id = str(uuid.uuid4())
        space_id = str(uuid.uuid4())
        user_a, user_b = str(uuid.uuid4()), str(uuid.uuid4())

        # 1) 真实文本 → ChunkPayload。两条中文 + 一条英文，能在 sparse
        #    路径上同时覆盖 Chinese / English tokenizer 分支。
        chunks_text = [
            "企业知识库系统支持多格式文档导入和向量化处理",
            "解析 / 清洗 / 分块 / 向量化 / 入库五步流水线",
            "Hybrid retrieval combines dense and sparse vectors",
        ]
        payloads = [
            _make_chunk_payload(
                document_id=document_id,
                space_id=space_id,
                chunk_index=i,
                content=text,
                user_ids=[user_a, user_b],
            )
            for i, text in enumerate(chunks_text)
        ]

        # 2) 真实 EmbeddingService —— 走 LiteLLM stub 出 dense + 真 TF-IDF sparse。
        embedding_service = EmbeddingService()
        embed_inputs = [{"id": p.chunk_id, "text": p.content} for p in payloads]
        results = asyncio.run(embedding_service.embed_chunks(embed_inputs))

        # 验证 EmbeddingService 的产物形状（不重测 12.3/12.4 的细节，只确认
        # 后续 IndexingService 拿到的是它能消费的类型）。
        assert len(results) == len(payloads)
        for r in results:
            assert isinstance(r, EmbeddingResult)
            assert len(r.dense_vector) == DENSE_VECTOR_DIM
            # 中文 / 英文分词都应该至少有 token，sparse 不能整批为空。
        assert any(r.sparse_indices for r in results)

        # 3) 用真实 IndexingService 跑双写。OpenSearch bulk 用 patch 拦截，
        #    Qdrant 客户端是注入的 MagicMock。
        with patch(
            "opensearchpy.helpers.bulk", return_value=(len(payloads), [])
        ) as mock_bulk:
            outcome = indexing_service.index_chunks(payloads, results)

        assert outcome == {
            "qdrant_count": len(payloads),
            "opensearch_count": len(payloads),
        }

        # ─── Qdrant 侧断言 ───
        # 全部 payload 一次能装下（QDRANT_BATCH_SIZE = 100），仅一次 upsert。
        indexing_service._qdrant.upsert.assert_called_once()
        qd_kwargs = indexing_service._qdrant.upsert.call_args.kwargs
        points = qd_kwargs["points"]
        assert [p.id for p in points] == [pl.chunk_id for pl in payloads]

        # 每个 PointStruct 的 dense 向量长度 = 1024，且与 EmbeddingResult 一致。
        for point, embedding, payload in zip(points, results, payloads):
            assert point.vector["dense"] == list(embedding.dense_vector)
            # sparse 仅在有信号时附带（与 12.5 契约一致）。
            if embedding.sparse_indices and embedding.sparse_values:
                assert "sparse" in point.vector
                assert point.vector["sparse"].indices == list(embedding.sparse_indices)
                assert point.vector["sparse"].values == list(embedding.sparse_values)
            else:
                assert "sparse" not in point.vector
            # payload 中的 ABAC 关键字段必须忠实复刻 ChunkPayload。
            assert point.payload["document_id"] == document_id
            assert point.payload["space_id"] == space_id
            assert point.payload["allowed_user_ids"] == [user_a, user_b]
            assert point.payload["content"] == payload.content
            assert point.payload["chunk_index"] == payload.chunk_index

        # ─── OpenSearch 侧断言 ───
        # 一次 bulk 调用，actions 与 payloads 一一对应。
        assert mock_bulk.call_count == 1
        actions = list(mock_bulk.call_args.args[1])
        assert len(actions) == len(payloads)
        # _id 必须等于 chunk_id（保证 re-index 是 idempotent upsert）。
        assert [a["_id"] for a in actions] == [p.chunk_id for p in payloads]
        # 关键字段在两侧一致（Qdrant payload ↔ OpenSearch _source）。
        for action, payload in zip(actions, payloads):
            src = action["_source"]
            assert src["chunk_id"] == payload.chunk_id
            assert src["document_id"] == document_id
            assert src["space_id"] == space_id
            assert src["content"] == payload.content
            assert src["allowed_user_ids"] == [user_a, user_b]


class TestEmbedAndIndexFailureRollsBackQdrant:
    """OpenSearch 失败时，IndexingService 必须回滚刚才上去的 Qdrant points。

    这是 12.7 契约的端到端体现——在真实 ``embed → index`` 链路上验证
    EmbeddingService 的产物与回滚路径上送进 Qdrant ``delete`` 的 ID 列表
    完全对齐（不会漏 / 重）。
    """

    def test_opensearch_failure_triggers_qdrant_rollback_with_exact_ids(
        self, litellm_stub, indexing_service
    ):
        from app.services.indexing_service import IndexingError

        document_id = str(uuid.uuid4())
        space_id = str(uuid.uuid4())
        user_a = str(uuid.uuid4())

        payloads = [
            _make_chunk_payload(
                document_id=document_id,
                space_id=space_id,
                chunk_index=i,
                content=f"chunk content {i}",
                user_ids=[user_a],
            )
            for i in range(3)
        ]
        expected_ids = [p.chunk_id for p in payloads]

        embedding_service = EmbeddingService()
        results = asyncio.run(
            embedding_service.embed_chunks(
                [{"id": p.chunk_id, "text": p.content} for p in payloads]
            )
        )

        # OpenSearch bulk 抛错（模拟集群不可用），Qdrant 客户端正常。
        with patch(
            "opensearchpy.helpers.bulk", side_effect=RuntimeError("opensearch down")
        ):
            with pytest.raises(IndexingError, match="OpenSearch write failed"):
                indexing_service.index_chunks(payloads, results)

        # Qdrant 经历 1 次 upsert + 1 次回滚 delete。
        assert indexing_service._qdrant.upsert.call_count == 1
        assert indexing_service._qdrant.delete.call_count == 1
        rollback_ids = indexing_service._qdrant.delete.call_args.kwargs[
            "points_selector"
        ]
        # 回滚送进的是真实写入过的 chunk_id 列表（顺序保留）。
        assert rollback_ids == expected_ids


class TestCascadeDeleteAfterIndex:
    """文档级联删除：``delete_document_chunks`` 必须同时打 Qdrant + OpenSearch。

    这是 12.9 在 ``embed → index → cleanup`` 闭环里的体现：先把 chunks
    写进双后端，再用同一个 IndexingService 实例触发清理，验证两端的
    清理调用都按 ``document_id`` 锁定（而不是按 chunk_id 列表）。
    """

    def test_index_then_delete_document_invokes_both_backends_with_document_id(
        self, litellm_stub, indexing_service
    ):
        from qdrant_client.models import Filter

        document_id = str(uuid.uuid4())
        space_id = str(uuid.uuid4())
        user_a = str(uuid.uuid4())

        payloads = [
            _make_chunk_payload(
                document_id=document_id,
                space_id=space_id,
                chunk_index=i,
                content=f"section {i}",
                user_ids=[user_a],
            )
            for i in range(2)
        ]
        embedding_service = EmbeddingService()
        results = asyncio.run(
            embedding_service.embed_chunks(
                [{"id": p.chunk_id, "text": p.content} for p in payloads]
            )
        )

        # 第一阶段：写入。
        with patch(
            "opensearchpy.helpers.bulk", return_value=(len(payloads), [])
        ):
            indexing_service.index_chunks(payloads, results)

        # 第二阶段：删除。OpenSearch ``delete_by_query`` 走客户端 mock，
        # 返回 deleted=2；Qdrant filter-delete 走客户端 mock。
        indexing_service._opensearch.delete_by_query.return_value = {"deleted": 2}

        cleanup = indexing_service.delete_document_chunks(document_id)

        # Qdrant：按 ``document_id`` payload filter 删除（不是按点 ID 列表，
        # 这才能在不知道全部 chunk_id 的情况下批量清理）。
        indexing_service._qdrant.delete.assert_called_once()
        qd_kwargs = indexing_service._qdrant.delete.call_args.kwargs
        selector = qd_kwargs["points_selector"]
        assert isinstance(selector, Filter)
        assert selector.must[0].key == "document_id"
        assert selector.must[0].match.value == document_id

        # OpenSearch：按 ``document_id`` term query delete_by_query，refresh=True
        # 才能让后续搜索立刻看不到被删的 chunks。
        indexing_service._opensearch.delete_by_query.assert_called_once()
        os_kwargs = indexing_service._opensearch.delete_by_query.call_args.kwargs
        assert os_kwargs["body"] == {
            "query": {"term": {"document_id": document_id}}
        }
        assert os_kwargs["refresh"] is True

        # 返回值：Qdrant 用 -1 哨兵（filter-delete 不回 count），OpenSearch
        # 反映客户端 ``deleted`` 字段。两侧都成功 → 不应出现 partial-cleanup
        # 的 ``opensearch_error`` 字段。
        assert cleanup["qdrant_deleted"] == -1
        assert cleanup["opensearch_deleted"] == 2
        assert "opensearch_error" not in cleanup


class TestPipelineStatusBoundaryAtIndexStage:
    """管线状态：``index_chunks`` 任务在入口写 ``indexing/0``、出口写 ``done/100``。

    这是 12.8 在 ``embed → index`` 任务边界上的体现 —— 不重测 Redis/PG 写
    法（已在 12.8 覆盖），只验证 Celery 任务在跑通真实 IndexingService
    时确实按设计文档要求的边界点触发了状态过渡。
    """

    def test_index_chunks_task_writes_status_at_entry_and_completion(
        self, litellm_stub, indexing_service
    ):
        from app.tasks.pipeline import index_chunks

        document_id = str(uuid.uuid4())
        space_id = str(uuid.uuid4())

        payloads = [
            _make_chunk_payload(
                document_id=document_id,
                space_id=space_id,
                chunk_index=0,
                content="hello",
                user_ids=[str(uuid.uuid4())],
            )
        ]
        embedding_service = EmbeddingService()
        results = asyncio.run(
            embedding_service.embed_chunks(
                [{"id": p.chunk_id, "text": p.content} for p in payloads]
            )
        )

        # Pipeline 任务消费 dict 形态的 chunks/embeddings，重建为
        # ChunkPayload / EmbeddingResult。这里复用真实 results。
        embed_result = {
            "document_id": document_id,
            "chunks": [
                {
                    "id": p.chunk_id,
                    "text": p.content,
                    "chunk_index": p.chunk_index,
                    "page_number": p.page_number,
                    "title_chain": p.title_chain,
                    "source_file": p.source_file,
                    "permission_ids": list(p.allowed_user_ids),
                }
                for p in payloads
            ],
            "embeddings": [
                {
                    "chunk_id": r.chunk_id,
                    "dense_vector": r.dense_vector,
                    "sparse_indices": r.sparse_indices,
                    "sparse_values": r.sparse_values,
                }
                for r in results
            ],
            "metadata": {},
            "profile_id": None,
        }

        recorded: list[tuple[str, str, int]] = []

        def _record(doc_id: str, stage: str, progress: int = 0) -> None:
            recorded.append((doc_id, stage, progress))

        # 用 patch 拦截 IndexingService 类，让任务用我们注入的 service
        # 实例（已含 mock 后端），从而避免真实客户端工厂被触发。
        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=_record
        ), patch(
            "app.tasks.pipeline._get_document_info",
            return_value={"storage_path": "h.pdf", "file_type": "pdf"},
        ), patch(
            "app.tasks.pipeline._get_document_space_id", return_value=space_id
        ), patch(
            "app.services.indexing_service.IndexingService",
            return_value=indexing_service,
        ), patch(
            "app.services.indexing_service.update_document_db_status"
        ) as mock_pg, patch(
            "app.services.indexing_service.update_pipeline_progress"
        ), patch(
            "app.core.qdrant.ensure_collection_exists"
        ), patch(
            "app.core.opensearch.ensure_index_exists"
        ), patch(
            "opensearchpy.helpers.bulk", return_value=(len(payloads), [])
        ):
            outcome = index_chunks.run(embed_result)

        # 任务返回 completed 状态，indexed_chunks 反映 Qdrant 上行计数。
        assert outcome["status"] == "completed"
        assert outcome["indexed_chunks"] == len(payloads)

        # 边界点：入口 ``indexing/0``、出口 ``done/100`` 都被记录。
        stages_at = {progress: stage for _, stage, progress in recorded}
        assert stages_at.get(0) == "indexing"
        assert stages_at.get(100) == "done"

        # PG 显式被推到 completed（任务尾部的 update_document_db_status 调用）。
        mock_pg.assert_called_once_with(document_id, "completed", "done", 100)
